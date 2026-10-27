"""Strict, read-only parser for the OpenMVS 2.4.0 MVSI version 7 profile.

The layout follows ``libs/MVS/Interface.h`` in the frozen source audit. MVSI
uses native scalar serialization; this profile explicitly supports little-
endian IEEE-754 files only. It does not read Boost archives or rewrite scenes.
Camera intrinsics retain their stored pixel resolution; actual depth-map
intrinsics must be read from the depth map, never inferred here.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import struct
import unicodedata
from typing import Any


MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_IMAGES = 256
MAX_PLATFORMS = 256
MAX_CAMERAS = 256
MAX_POSES = 4096
MAX_VERTICES = 1_000_000
MAX_LINES = 1_000_000
MAX_OBSERVATIONS = 10_000_000
MAX_STRING_BYTES = 4096
ROTATION_TOLERANCE = 1e-8
NO_ID = 0xFFFFFFFF


class SceneFormatError(ValueError):
    """The input is outside the bounded MVSI7 reader profile."""


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0
        self.observations = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def fail(self, message: str) -> None:
        raise SceneFormatError(f"MVSI7 byte {self.offset}: {message}")

    def unpack(self, fmt: str, label: str) -> tuple:
        size = struct.calcsize("<" + fmt)
        if size > self.remaining:
            self.fail(f"truncated {label}: need {size}, have {self.remaining}")
        result = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += size
        return result

    def u32(self, label: str) -> int:
        return self.unpack("I", label)[0]

    def count(self, label: str, limit: int, minimum: int) -> int:
        value = self.unpack("Q", label)[0]
        if value > limit:
            self.fail(f"{label} count {value} exceeds limit {limit}")
        if value * minimum > self.remaining:
            self.fail(f"{label} count {value} exceeds remaining bytes")
        return value

    def string(self, label: str) -> str:
        size = self.count(label, MAX_STRING_BYTES, 1)
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        try:
            value = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SceneFormatError(f"{label}: invalid UTF-8") from exc
        if "\x00" in value:
            self.fail(f"{label}: NUL is forbidden")
        return value

    def floats(self, count: int, label: str, scalar: str = "d") -> list[float]:
        result = list(self.unpack(f"{count}{scalar}", label))
        if not all(math.isfinite(value) for value in result):
            self.fail(f"{label}: non-finite value")
        return result

    def matrix(self, side: int, label: str) -> list[list[float]]:
        values = self.floats(side * side, label)
        return [values[index : index + side] for index in range(0, len(values), side)]


def _rotation(matrix: list[list[float]], label: str) -> None:
    for row in range(3):
        for other in range(3):
            dot = sum(matrix[row][j] * matrix[other][j] for j in range(3))
            if abs(dot - float(row == other)) > ROTATION_TOLERANCE:
                raise SceneFormatError(f"{label}: not an orthonormal rotation")
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(determinant - 1.0) > ROTATION_TOLERANCE:
        raise SceneFormatError(f"{label}: not a proper rotation (determinant must be +1)")


def _intrinsics(matrix: list[list[float]], label: str) -> None:
    if matrix[0][0] <= 0 or matrix[1][1] <= 0:
        raise SceneFormatError(f"{label}: focal lengths must be positive")
    if any(abs(matrix[row][col]) > 1e-8 for row, col in ((1, 0), (2, 0), (2, 1))):
        raise SceneFormatError(f"{label}: invalid pinhole intrinsics")
    if abs(matrix[2][2] - 1.0) > 1e-8:
        raise SceneFormatError(f"{label}: K[2][2] must be 1")


def _relative_image_name(value: str) -> str:
    """Keep Unicode spelling; remove only the platform separator difference."""
    name = value.replace("\\", "/")
    parts = name.split("/")
    if (
        not name
        or name.startswith("/")
        or ":" in name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in parts)
    ):
        raise SceneFormatError(f"unsafe relative image path: {value!r}")
    forbidden = '<>"|?*'
    devices = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
    }
    if any(char in forbidden for char in name) or any(
        part.split(".", 1)[0].casefold() in devices for part in parts
    ):
        raise SceneFormatError(f"unsafe Windows image path: {value!r}")
    return name


def _compose(camera: dict, pose: dict) -> tuple[list[list[float]], list[float]]:
    # Interface::Platform::GetPose: camera.R * pose.R,
    #                              transpose(pose.R) * camera.C + pose.C.
    relative, platform = camera["R"], pose["R"]
    rotation = [
        [sum(relative[row][k] * platform[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]
    center = [
        sum(platform[k][row] * camera["C"][k] for k in range(3)) + pose["C"][row]
        for row in range(3)
    ]
    if not all(math.isfinite(value) for value in center):
        raise SceneFormatError("composed camera center is non-finite")
    _rotation(rotation, "composed image R")
    return rotation, center


def _read_platforms(reader: _Reader) -> list[dict]:
    platforms = []
    camera_total = pose_total = 0
    for platform_index in range(reader.count("platforms", MAX_PLATFORMS, 24)):
        name = reader.string("platform.name")
        camera_count = reader.count("platform.cameras", MAX_CAMERAS - camera_total, 192)
        camera_total += camera_count
        cameras = []
        for camera_index in range(camera_count):
            label = f"platform[{platform_index}].camera[{camera_index}]"
            camera = {"name": reader.string(label + ".name"), "band_name": reader.string(label + ".bandName")}
            camera["width"] = reader.u32(label + ".width")
            camera["height"] = reader.u32(label + ".height")
            if not camera["width"] or not camera["height"]:
                reader.fail(label + ": normalized/unknown-resolution camera is unsupported")
            camera["K"] = reader.matrix(3, label + ".K")
            camera["R"] = reader.matrix(3, label + ".R")
            camera["C"] = reader.floats(3, label + ".C")
            _intrinsics(camera["K"], label + ".K")
            _rotation(camera["R"], label + ".R")
            cameras.append(camera)
        pose_count = reader.count("platform.poses", MAX_POSES - pose_total, 96)
        pose_total += pose_count
        poses = []
        for pose_index in range(pose_count):
            label = f"platform[{platform_index}].pose[{pose_index}]"
            pose = {"R": reader.matrix(3, label + ".R"), "C": reader.floats(3, label + ".C")}
            _rotation(pose["R"], label + ".R")
            poses.append(pose)
        platforms.append({"name": name, "cameras": cameras, "poses": poses})
    return platforms


def _read_images(reader: _Reader, platforms: list[dict]) -> list[dict]:
    count = reader.count("images", MAX_IMAGES, 52)
    images, global_ids, path_keys = [], set(), set()
    for index in range(count):
        raw_name = reader.string("image.name")
        name = _relative_image_name(raw_name)
        key = unicodedata.normalize("NFC", name).casefold()
        if key in path_keys:
            reader.fail(f"duplicate/aliased image path: {name!r}")
        path_keys.add(key)
        record: dict[str, Any] = {
            "index": index,
            "name": name,
            "name_raw": raw_name,
            "mask_name": reader.string("image.maskName"),
            "platform_id": reader.u32("image.platformID"),
            "camera_id": reader.u32("image.cameraID"),
            "pose_id": reader.u32("image.poseID"),
            "global_id": reader.u32("image.ID"),
        }
        pid, cid, poseid = record["platform_id"], record["camera_id"], record["pose_id"]
        if pid >= len(platforms):
            reader.fail(f"image[{index}]: platform reference out of range")
        platform = platforms[pid]
        if cid >= len(platform["cameras"]):
            reader.fail(f"image[{index}]: camera reference out of range")
        if poseid >= len(platform["poses"]):
            reader.fail(f"image[{index}]: pose reference out of range")
        if record["global_id"] == NO_ID or record["global_id"] in global_ids:
            reader.fail(f"image[{index}]: missing or duplicate global ID")
        global_ids.add(record["global_id"])
        record["min_depth"], record["avg_depth"], record["max_depth"] = reader.floats(3, "image.depths", "f")
        scores = []
        for _ in range(reader.count("image.viewScores", MAX_IMAGES, 24)):
            local_id, points = reader.unpack("II", "viewScore IDs/points")
            if local_id >= count:
                reader.fail("viewScore.ID is not a valid local image index")
            scale, angle, area, score = reader.floats(4, "viewScore values", "f")
            scores.append({"image_index": local_id, "points": points, "scale": scale, "angle": angle, "area": area, "score": score})
        record["view_scores"] = scores
        camera = platform["cameras"][cid]
        rotation, center = _compose(camera, platform["poses"][poseid])
        record.update(width=camera["width"], height=camera["height"], K=camera["K"], R=rotation, C=center)
        images.append(record)
    return images


def _read_observations(reader: _Reader, image_count: int, label: str) -> int:
    count = reader.count(label + ".views", MAX_OBSERVATIONS - reader.observations, 8)
    reader.observations += count
    for _ in range(count):
        image_index = reader.u32(label + ".view.imageID")
        if image_index >= image_count:
            reader.fail(label + ".view.imageID is not a valid local image index")
        reader.floats(1, label + ".view.confidence", "f")
    return count


def _read_features(reader: _Reader, image_count: int, label: str, limit: int, endpoints: int) -> tuple[int, dict | None, int]:
    count = reader.count(label, limit, endpoints * 12 + 8)
    low, high = [math.inf] * 3, [-math.inf] * 3
    minimum_views = None
    for _ in range(count):
        for _ in range(endpoints):
            point = reader.floats(3, label + ".position", "f")
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
        views = _read_observations(reader, image_count, label)
        minimum_views = views if minimum_views is None else min(minimum_views, views)
    return count, {"min": low, "max": high} if count else None, minimum_views if count else 0


def _read_attributes(reader: _Reader, label: str, feature_count: int) -> tuple[int, int]:
    normals = reader.count(label + "Normal", feature_count, 12)
    if normals not in (0, feature_count):
        reader.fail(label + "Normal must be empty or match feature count")
    for _ in range(normals):
        reader.floats(3, label + "Normal.value", "f")
    colors = reader.count(label + "Color", feature_count, 3)
    if colors not in (0, feature_count):
        reader.fail(label + "Color must be empty or match feature count")
    reader.offset += colors * 3  # Count/remaining bound was checked above.
    return normals, colors


def read_scene(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one immutable-content snapshot, validate all fields, return metadata.

    Bounds are this workflow's policy, not limits of the native MVSI format.
    Points/lines and their observations are validated without retaining large
    lists. ``sha256`` covers precisely the bytes parsed. No native code runs.
    """
    source = Path(path)
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > MAX_FILE_BYTES:
            raise SceneFormatError(f"scene exceeds {MAX_FILE_BYTES} byte limit")
        data = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(data) > MAX_FILE_BYTES:
        raise SceneFormatError(f"scene exceeds {MAX_FILE_BYTES} byte limit")
    if len(data) != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SceneFormatError("scene changed while reading")
    reader = _Reader(data)
    magic, version, reserved = reader.unpack("4sII", "header")
    if magic != b"MVSI" or version != 7:
        reader.fail("only little-endian MVSI version 7 is supported")
    if reserved != 0:
        reader.fail("nonzero header reserved field is unsupported")
    platforms = _read_platforms(reader)
    images = _read_images(reader, platforms)
    point_count, point_bbox, min_point_views = _read_features(reader, len(images), "vertices", MAX_VERTICES, 1)
    point_normals, point_colors = _read_attributes(reader, "vertices", point_count)
    line_count, line_bbox, _ = _read_features(reader, len(images), "lines", MAX_LINES, 2)
    line_normals, line_colors = _read_attributes(reader, "lines", line_count)
    transform = reader.matrix(4, "transform")
    obb = {"rot": reader.matrix(3, "obb.rot"), "pt_min": reader.floats(3, "obb.ptMin"), "pt_max": reader.floats(3, "obb.ptMax")}
    if reader.remaining:
        reader.fail(f"unexpected trailing bytes: {reader.remaining}")
    return {
        "path": str(source.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "version": version,
        "byte_order": "little",
        "platforms": platforms,
        "images": images,
        "point_count": point_count,
        "min_point_views": min_point_views,
        "point_bbox": point_bbox,
        "line_count": line_count,
        "line_bbox": line_bbox,
        "observation_count": reader.observations,
        "point_normal_count": point_normals,
        "point_color_count": point_colors,
        "line_normal_count": line_normals,
        "line_color_count": line_colors,
        "transform": transform,
        "obb": obb,
    }
