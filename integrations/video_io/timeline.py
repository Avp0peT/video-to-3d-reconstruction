"""Strict, standard-library-only video timestamp and export validation.

Only ffprobe's integer ``best_effort_timestamp`` defines the input timeline.
Frame-rate and floating-point timestamp metadata never define frame identity.
"""

from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
import math
import re
from typing import Any


_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_ROTATIONS = {0, 90, -90, 180, -180, 270, -270}
_COLOR_FIELDS = ("color_range", "color_space", "color_transfer", "color_primaries")
_SHOWINFO_CONFIG = re.compile(r"config in time_base:\s*([+-]?[0-9]+)\s*/\s*([+-]?[0-9]+)")


def _integer(value: Any, label: str) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and _INTEGER.fullmatch(value):
        return int(value)
    raise ValueError(f"{label} must be an integer or canonical integer string")


def _positive_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _time_base(value: Any, label: str) -> Fraction:
    if isinstance(value, str):
        parts = value.split("/")
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        raise ValueError(f"{label} must contain an integer numerator and denominator")
    if len(parts) != 2:
        raise ValueError(f"{label} must contain an integer numerator and denominator")
    numerator = _integer(parts[0], label)
    denominator = _integer(parts[1], label)
    if denominator == 0:
        raise ValueError(f"{label} denominator must not be zero")
    result = Fraction(numerator, denominator)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _square_pixels(value: Any, label: str) -> str:
    if not isinstance(value, str) or value not in ("1:1", "1/1"):
        raise ValueError(f"{label} must explicitly declare square pixels as 1:1 or 1/1")
    return "1:1"


def _rotation(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be a supported right-angle rotation")
    try:
        angle = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be a supported right-angle rotation") from error
    if angle.denominator != 1 or angle.numerator not in _ROTATIONS:
        raise ValueError(f"{label} must be 0, +/-90, +/-180, or +/-270")
    return angle.numerator


def _display_matrix(value: Any, label: str) -> list[int]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ffprobe displaymatrix string")
    lines = [line for line in value.splitlines() if line.strip()]
    matrix: list[int] = []
    if len(lines) != 3:
        raise ValueError(f"{label} must have exactly three matrix rows")
    for line in lines:
        if ":" not in line:
            raise ValueError(f"{label} has an invalid matrix row")
        address, row = line.split(":", 1)
        if not re.fullmatch(r"[0-9a-fA-F]+", address.strip()):
            raise ValueError(f"{label} has an invalid matrix row address")
        values = row.split()
        if len(values) != 3:
            raise ValueError(f"{label} must have three values per row")
        matrix.extend(_integer(item, label) for item in values)
    scale = 65536
    a, b, _, c, d, _, _, _, _ = matrix
    rows_and_columns = ((a, b), (c, d), (a, c), (b, d))
    if (
        any(sum(item != 0 for item in group) != 1 for group in rows_and_columns)
        or any(item not in {0, scale, -scale} for item in (a, b, c, d))
        or a * d - b * c != scale * scale
        or any(matrix[index] != 0 for index in (2, 5, 6, 7))
        or matrix[8] != 1073741824
    ):
        raise ValueError(f"{label} only supports pure right-angle rotation matrices")
    return matrix


def _rotation_metadata(
    record: dict[str, Any], label: str, fallback: int = 0
) -> tuple[int, list[int] | None]:
    side_data = record.get("side_data_list", [])
    if not isinstance(side_data, list):
        raise ValueError(f"{label}.side_data_list must be a list")
    angle: int | None = None
    matrix: list[int] | None = None
    for entry in side_data:
        if not isinstance(entry, dict):
            raise ValueError(f"{label}.side_data_list entries must be objects")
        kind = str(entry.get("side_data_type", "")).lower().replace(" ", "")
        if kind != "displaymatrix" and "displaymatrix" not in entry:
            continue
        if "rotation" not in entry:
            raise ValueError(f"{label} displaymatrix is missing rotation")
        if "displaymatrix" not in entry:
            raise ValueError(f"{label} Display Matrix side-data is missing displaymatrix")
        candidate = _rotation(entry["rotation"], f"{label}.rotation")
        if angle is not None and candidate % 360 != angle % 360:
            raise ValueError(f"{label} has conflicting displaymatrix rotations")
        angle = candidate
        candidate_matrix = _display_matrix(entry["displaymatrix"], f"{label}.displaymatrix")
        if matrix is not None and candidate_matrix != matrix:
            raise ValueError(f"{label} has conflicting displaymatrix values")
        matrix = candidate_matrix
    result = fallback if angle is None else angle
    tags = record.get("tags", {})
    if not isinstance(tags, dict):
        raise ValueError(f"{label}.tags must be an object")
    if "rotate" in tags and _rotation(tags["rotate"], f"{label}.tags.rotate") % 360 != result % 360:
        raise ValueError(f"{label} rotate tag conflicts with displaymatrix rotation")
    return result, matrix


def parse_probe(data: dict) -> dict:
    """Validate ffprobe JSON and return a serializable, strictly ordered timeline.

    Per-frame dimensions, pixel format, and square-pixel aspect ratio must be
    present and static. Display matrices are restricted to pure right-angle
    rotations, excluding mirrors, scale, translation, perspective, and shear.
    Declared stream color metadata is checked against present per-frame values;
    absent color metadata makes no claim of validated color interpretation.
    """
    if not isinstance(data, dict):
        raise ValueError("probe data must be an object")
    streams = data.get("streams")
    if not isinstance(streams, list) or not all(isinstance(stream, dict) for stream in streams):
        raise ValueError("probe streams must be a list of objects")
    video_streams = []
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        disposition = stream.get("disposition", {})
        if not isinstance(disposition, dict):
            raise ValueError("stream disposition must be an object")
        attached = _integer(disposition.get("attached_pic", 0), "stream attached_pic")
        if attached not in (0, 1):
            raise ValueError("stream attached_pic must be zero or one")
        if not attached:
            video_streams.append(stream)
    if len(video_streams) != 1:
        raise ValueError("probe must contain exactly one non-attached-picture video stream")
    stream = video_streams[0]
    stream_index = _integer(stream.get("index"), "stream index")
    if stream_index < 0:
        raise ValueError("stream index must be nonnegative")
    base = _time_base(stream.get("time_base"), "stream time_base")
    width = _positive_integer(stream.get("width"), "stream width")
    height = _positive_integer(stream.get("height"), "stream height")
    pix_fmt = stream.get("pix_fmt")
    if not isinstance(pix_fmt, str) or not pix_fmt.strip():
        raise ValueError("stream pix_fmt must be present")
    sample_aspect_ratio = _square_pixels(stream.get("sample_aspect_ratio"), "stream sample_aspect_ratio")
    color_metadata = {field: stream[field] for field in _COLOR_FIELDS if field in stream}
    rotation, matrix = _rotation_metadata(stream, "stream")
    raw_frames = data.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("probe frames must be a nonempty list")
    frames = []
    previous_pts: int | None = None
    for decode_index, frame in enumerate(raw_frames):
        label = f"frame {decode_index}"
        if not isinstance(frame, dict):
            raise ValueError(f"{label} must be an object")
        if frame.get("media_type") != "video":
            raise ValueError(f"{label} media_type must be video")
        if _integer(frame.get("stream_index"), f"{label} stream_index") != stream_index:
            raise ValueError(f"{label} belongs to a different stream")
        if (
            _positive_integer(frame.get("width"), f"{label} width") != width
            or _positive_integer(frame.get("height"), f"{label} height") != height
            or frame.get("pix_fmt") != pix_fmt
        ):
            raise ValueError(f"{label} dimensions or pix_fmt changed from the stream")
        if _square_pixels(frame.get("sample_aspect_ratio"), f"{label} sample_aspect_ratio") != sample_aspect_ratio:
            raise ValueError(f"{label} sample_aspect_ratio changed from the stream")
        for field, expected in color_metadata.items():
            if field in frame and frame[field] != expected:
                raise ValueError(f"{label} {field} changed from the stream")
        frame_rotation, frame_matrix = _rotation_metadata(frame, label, rotation)
        if frame_rotation % 360 != rotation % 360:
            raise ValueError(f"{label} rotation changed from the stream")
        if frame_matrix is not None and frame_matrix != matrix:
            raise ValueError(f"{label} displaymatrix changed from the stream")
        pts = _integer(frame.get("best_effort_timestamp"), f"{label} best_effort_timestamp")
        if previous_pts is not None and pts <= previous_pts:
            raise ValueError("frame best_effort_timestamp values must be strictly increasing")
        previous_pts = pts
        try:
            seconds = float(pts * base)
        except OverflowError as error:
            raise ValueError(f"{label} timestamp cannot be represented as finite seconds") from error
        if not math.isfinite(seconds):
            raise ValueError(f"{label} timestamp cannot be represented as finite seconds")
        frames.append({
            "decode_index": decode_index,
            "pts": pts,
            "time_base": _pair(base),
            "time_seconds": seconds,
        })
    return {
        "stream_index": stream_index,
        "time_base": _pair(base),
        "width": width,
        "height": height,
        "pix_fmt": pix_fmt,
        "sample_aspect_ratio": sample_aspect_ratio,
        "color_metadata": color_metadata,
        "rotation_degrees": rotation,
        "display_matrix": matrix,
        "pts_field": "best_effort_timestamp",
        "frames": frames,
    }


def select_uniform(timeline: dict, count: int) -> list[dict]:
    """Select nearest frames to evenly spaced exact presentation timestamps.

    Ties choose the earlier frame. Repeated nearest frames are deduplicated;
    sparse regions never cause synthesized frames or index-based gap filling.
    """
    if type(count) is not int or not 2 <= count <= 96:
        raise ValueError("count must be an integer between 2 and 96")
    frames = timeline.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("timeline frames must be a nonempty list")
    times = [
        _integer(frame.get("pts"), "frame pts") * _time_base(frame.get("time_base"), "frame time_base")
        for frame in frames
    ]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timeline timestamps must be strictly increasing")
    if len(frames) <= count:
        return list(frames)
    chosen = set()
    for target_index in range(count):
        target = times[0] + (times[-1] - times[0]) * Fraction(target_index, count - 1)
        right = bisect_left(times, target)
        if right == 0:
            nearest = 0
        elif right == len(times):
            nearest = right - 1
        elif target - times[right - 1] <= times[right] - target:
            nearest = right - 1
        else:
            nearest = right
        chosen.add(nearest)
    return [frames[index] for index in sorted(chosen)]


def select_all(timeline: dict) -> list[dict]:
    """Whole-video-192-v1: retain every original decoded frame, or reject."""
    frames = timeline.get("frames")
    if not isinstance(frames, list) or not 2 <= len(frames) <= 192:
        raise ValueError("whole-video-192-v1 requires 2..192 decoded frames")
    times = []
    for index, frame in enumerate(frames):
        if type(frame.get("decode_index")) is not int or frame["decode_index"] != index:
            raise ValueError("whole-video decode indices must be complete and ordered")
        times.append(_integer(frame.get("pts"), "frame pts") * _time_base(frame.get("time_base"), "frame time_base"))
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timeline timestamps must be strictly increasing")
    return list(frames)


def parse_showinfo(text: str) -> list[dict]:
    """Parse showinfo frame rows with their declared input time base.

    Configuration and color-detail rows are otherwise ignored. A frame without
    a preceding input time base, changed time base, or reset index is rejected.
    No frame rows returns an empty list for the caller's count check.
    """
    if not isinstance(text, str):
        raise ValueError("showinfo text must be a string")
    frames = []
    base: Fraction | None = None
    for line in text.splitlines():
        if "showinfo" not in line.lower():
            continue
        if "config in time_base:" in line:
            match = _SHOWINFO_CONFIG.search(line)
            if match is None:
                raise ValueError("showinfo config has an invalid input time_base")
            candidate = _time_base([match.group(1), match.group(2)], "showinfo time_base")
            if base is not None and candidate != base:
                raise ValueError("showinfo input time_base changed")
            base = candidate
            continue
        if not re.search(r"\bn\s*:", line):
            continue
        index_match = re.search(r"\bn\s*:\s*(\S+)", line)
        pts_match = re.search(r"\bpts\s*:\s*(\S+)", line)
        size_match = re.search(r"\bs\s*:\s*([0-9]+)x([0-9]+)(?=\s|$)", line)
        if index_match is None or pts_match is None or size_match is None:
            raise ValueError("showinfo frame row is missing n, pts, or s")
        if base is None:
            raise ValueError("showinfo frame row is missing input time_base config")
        index = _integer(index_match.group(1), "showinfo output_index")
        if index != len(frames):
            raise ValueError("showinfo output indices must be consecutive starting at zero")
        frames.append({
            "output_index": index,
            "pts": _integer(pts_match.group(1), "showinfo pts"),
            "width": _positive_integer(size_match.group(1), "showinfo width"),
            "height": _positive_integer(size_match.group(2), "showinfo height"),
            "time_base": _pair(base),
        })
    return frames


def validate_export(timeline: dict, selected: list[dict], showinfo: list[dict]) -> bool:
    """Verify output count and exact source presentation timestamps.

    FFmpeg can rescale integer PTS into a different time base. Equality is
    therefore checked on exact rational timestamps, never rounded seconds or
    frame-rate calculations. Image dimensions are checked by the caller.
    """
    frames = timeline.get("frames")
    if not isinstance(frames, list) or not isinstance(selected, list) or not selected:
        raise ValueError("export requires a nonempty selection from the timeline")
    if not isinstance(showinfo, list) or len(showinfo) != len(selected):
        raise ValueError("export frame count does not match selected frame count")
    previous_index = -1
    for output_index, (source, actual) in enumerate(zip(selected, showinfo)):
        if not isinstance(source, dict) or not isinstance(actual, dict):
            raise ValueError("selected and showinfo frames must be objects")
        decode_index = _integer(source.get("decode_index"), "selected decode_index")
        if not previous_index < decode_index < len(frames):
            raise ValueError("selected frames must preserve unique source decode order")
        previous_index = decode_index
        original = frames[decode_index]
        original_pts = _integer(original.get("pts"), "original pts")
        original_base = _time_base(original.get("time_base"), "original time_base")
        if (
            _integer(source.get("pts"), "selected pts") != original_pts
            or _time_base(source.get("time_base"), "selected time_base") != original_base
        ):
            raise ValueError("selected frame does not match its original timeline PTS")
        if _integer(actual.get("output_index"), "showinfo output_index") != output_index:
            raise ValueError("showinfo output indices must be consecutive starting at zero")
        actual_time = (
            _integer(actual.get("pts"), "showinfo pts")
            * _time_base(actual.get("time_base"), "showinfo time_base")
        )
        if actual_time != original_pts * original_base:
            raise ValueError(f"export PTS mismatch at output frame {output_index}")
    return True
