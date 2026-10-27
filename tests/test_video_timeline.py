"""Synthetic timestamp contract tests; no ffmpeg process or video is required."""

from __future__ import annotations

from copy import deepcopy

import pytest

from integrations.video_io.timeline import (
    parse_probe,
    parse_showinfo,
    select_uniform,
    validate_export,
)


def probe(pts: tuple[int, ...] = (0, 10, 20, 90, 100)) -> dict:
    return {
        "streams": [{
            "index": 2,
            "codec_type": "video",
            "width": 640,
            "height": 360,
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "1:1",
            "time_base": "1/1000",
            "disposition": {"attached_pic": 0},
        }],
        "frames": [{
            "media_type": "video",
            "stream_index": 2,
            "best_effort_timestamp": value,
            "width": 640,
            "height": 360,
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "1:1",
        } for value in pts],
    }


def matrix_text(values: list[int]) -> str:
    return "\n".join(
        f"{row:08d}: " + " ".join(str(item) for item in values[row * 3:row * 3 + 3])
        for row in range(3)
    )


def matrix_entry(rotation: int = 0, values: list[int] | None = None) -> dict:
    result = {"side_data_type": "Display Matrix", "rotation": rotation}
    if values is None:
        # Explicit, known pure rotation matrices keep valid fixtures complete.
        coefficients = {
            0: (65536, 0, 0, 65536),
            90: (0, -65536, 65536, 0),
            180: (-65536, 0, 0, -65536),
            270: (0, 65536, -65536, 0),
        }
        key = rotation % 360 if type(rotation) is int else 0
        a, b, c, d = coefficients.get(key, coefficients[0])
        values = [a, b, 0, c, d, 0, 0, 0, 1073741824]
    result["displaymatrix"] = matrix_text(values)
    return result


def showinfo_text(pts: list[int], base: str = "1/1000") -> str:
    rows = [f"[Parsed_showinfo_1 @ 0x1] config in time_base: {base}, frame_rate: 999/1"]
    rows.extend(
        f"[Parsed_showinfo_1 @ 0x1] n: {index:3d} pts: {value:6d} "
        "pts_time:999 fmt:rgb24 sar:1/1 s:640x360 i:P iskey:1 type:I checksum:00000000"
        for index, value in enumerate(pts)
    )
    return "\n".join(rows)


def test_probe_records_integer_pts_and_exact_time_base() -> None:
    data = probe((-200, 0, 1400))
    data["streams"][0]["time_base"] = "2/2000"
    data["frames"][0]["best_effort_timestamp"] = "-200"
    timeline = parse_probe(data)
    assert timeline["pts_field"] == "best_effort_timestamp"
    assert timeline["stream_index"] == 2
    assert timeline["time_base"] == [1, 1000]
    assert timeline["rotation_degrees"] == 0
    assert timeline["frames"][0] == {
        "decode_index": 0, "pts": -200, "time_base": [1, 1000], "time_seconds": -0.2,
    }


def test_uniform_selection_uses_vfr_timestamps_and_earlier_ties() -> None:
    timeline = parse_probe(probe((0, 1, 2, 40, 60, 100)))
    assert [frame["pts"] for frame in select_uniform(timeline, 3)] == [0, 40, 100]


@pytest.mark.parametrize("offset", [-1000, 1000], ids=["negative", "positive"])
def test_uniform_selection_preserves_nonzero_origin(offset: int) -> None:
    timeline = parse_probe(probe(tuple(offset + value for value in (0, 1, 2, 40, 60, 100))))
    assert [frame["pts"] for frame in select_uniform(timeline, 3)] == [offset, offset + 40, offset + 100]


def test_wrong_fps_and_float_timestamp_metadata_cannot_change_selection() -> None:
    data = probe()
    before = select_uniform(parse_probe(data), 3)
    data["streams"][0].update({"avg_frame_rate": "999/1", "r_frame_rate": "1/999", "start_time": "800"})
    for frame in data["frames"]:
        frame.update({"best_effort_timestamp_time": "1234567", "pts_time": "-999", "pkt_duration_time": "100"})
    assert select_uniform(parse_probe(data), 3) == before


def test_uniform_selection_deduplicates_without_gap_filling() -> None:
    timeline = parse_probe(probe((0, 1, 2, 3, 4, 100)))
    assert [frame["pts"] for frame in select_uniform(timeline, 5)] == [0, 4, 100]


def test_short_video_returns_all_real_frames() -> None:
    timeline = parse_probe(probe((71, 99)))
    assert select_uniform(timeline, 96) == timeline["frames"]
    assert select_uniform(parse_probe(probe((71,))), 2)[0]["pts"] == 71


def test_large_pts_selection_does_not_use_rounded_float_seconds() -> None:
    origin = 2 ** 60
    timeline = parse_probe(probe(tuple(origin + value for value in (0, 1, 2, 3, 4, 5))))
    assert [frame["pts"] for frame in select_uniform(timeline, 3)] == [origin, origin + 2, origin + 5]


@pytest.mark.parametrize("count", [True, 1, 97, 3.0, "3"], ids=["bool", "small", "large", "float", "text"])
def test_invalid_sample_count_is_rejected(count: object) -> None:
    with pytest.raises(ValueError, match="count"):
        select_uniform(parse_probe(probe()), count)


@pytest.mark.parametrize(
    "bad_pts", [None, True, 1.5, "1.0", "01", "+1", "-0", " 1", "N/A"],
    ids=["missing", "bool", "float", "decimal", "leading", "plus", "negzero", "space", "na"],
)
def test_invalid_source_pts_is_rejected(bad_pts: object) -> None:
    data = probe()
    data["frames"][1]["best_effort_timestamp"] = bad_pts
    with pytest.raises(ValueError, match="best_effort_timestamp"):
        parse_probe(data)


@pytest.mark.parametrize("pts", [(0, 1, 1), (10, 5, 20)], ids=["duplicate", "backwards"])
def test_non_monotonic_pts_is_rejected(pts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_probe(probe(pts))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("stream_index", 0, "different stream"), ("media_type", "audio", "media_type"),
     ("width", 320, "changed"), ("height", 240, "changed"), ("pix_fmt", "yuv444p", "changed")],
    ids=["stream", "media", "width", "height", "format"],
)
def test_frame_stream_or_format_change_is_rejected(field: str, value: object, message: str) -> None:
    data = probe()
    data["frames"][1][field] = value
    with pytest.raises(ValueError, match=message):
        parse_probe(data)


@pytest.mark.parametrize("field", ["width", "height", "pix_fmt", "best_effort_timestamp"], ids=["width", "height", "format", "pts"])
def test_frame_required_fields_must_be_present(field: str) -> None:
    data = probe()
    del data["frames"][1][field]
    with pytest.raises(ValueError):
        parse_probe(data)


@pytest.mark.parametrize("target", ["stream", "frame"], ids=["stream", "frame"])
@pytest.mark.parametrize(
    "ratio", [None, "0:1", "N/A", "unknown", "4:3", "2:2", True],
    ids=["missing", "zero", "na", "unknown", "nonsquare", "unnormalized", "bool"],
)
def test_square_pixel_aspect_ratio_is_required(target: str, ratio: object) -> None:
    data = probe()
    record = data["streams"][0] if target == "stream" else data["frames"][1]
    if ratio is None:
        del record["sample_aspect_ratio"]
    else:
        record["sample_aspect_ratio"] = ratio
    with pytest.raises(ValueError, match="sample_aspect_ratio"):
        parse_probe(data)


def test_equivalent_square_pixel_notations_are_accepted() -> None:
    data = probe()
    data["streams"][0]["sample_aspect_ratio"] = "1/1"
    data["frames"][1]["sample_aspect_ratio"] = "1/1"
    assert parse_probe(data)["sample_aspect_ratio"] == "1:1"


@pytest.mark.parametrize(
    "field", ["color_range", "color_space", "color_transfer", "color_primaries"],
    ids=["range", "space", "transfer", "primaries"],
)
def test_declared_color_metadata_cannot_change_per_frame(field: str) -> None:
    data = probe()
    data["streams"][0][field] = "declared"
    data["frames"][0][field] = "declared"
    assert parse_probe(data)["color_metadata"] == {field: "declared"}
    data["frames"][1][field] = "different"
    with pytest.raises(ValueError, match=field + " changed"):
        parse_probe(data)


def test_absent_stream_color_metadata_is_not_claimed_as_validated() -> None:
    data = probe()
    data["frames"][1]["color_space"] = "bt709"
    assert parse_probe(data)["color_metadata"] == {}


@pytest.mark.parametrize("time_base", ["0/1", "1/0", "-1/1000", "NaN", None, True], ids=["zero", "zeroden", "negative", "nan", "missing", "bool"])
def test_invalid_time_base_is_rejected(time_base: object) -> None:
    data = probe()
    data["streams"][0]["time_base"] = time_base
    with pytest.raises(ValueError, match="time_base"):
        parse_probe(data)


def test_multiple_real_video_streams_are_rejected_but_attached_picture_is_ignored() -> None:
    data = probe()
    extra = deepcopy(data["streams"][0])
    extra["index"] = 8
    data["streams"].append(extra)
    with pytest.raises(ValueError, match="exactly one"):
        parse_probe(data)
    extra["disposition"]["attached_pic"] = 1
    assert parse_probe(data)["stream_index"] == 2


@pytest.mark.parametrize("angle", [0, 90, -90, 180, -180, 270, -270], ids=["zero", "p90", "n90", "p180", "n180", "p270", "n270"])
def test_supported_rotations_are_recorded(angle: int) -> None:
    data = probe()
    data["streams"][0]["side_data_list"] = [matrix_entry(angle)]
    assert parse_probe(data)["rotation_degrees"] == angle


@pytest.mark.parametrize("angle", [45, 360, "NaN", True], ids=["oblique", "turn", "nan", "bool"])
def test_unsupported_rotation_is_rejected(angle: object) -> None:
    data = probe()
    data["streams"][0]["side_data_list"] = [matrix_entry(angle)]
    with pytest.raises(ValueError, match="rotation|90"):
        parse_probe(data)


def test_rotate_tag_conflict_and_missing_display_rotation_are_rejected() -> None:
    data = probe()
    data["streams"][0]["side_data_list"] = [matrix_entry(-90)]
    data["streams"][0]["tags"] = {"rotate": "90"}
    with pytest.raises(ValueError, match="conflicts"):
        parse_probe(data)
    data["streams"][0]["tags"]["rotate"] = "270"
    assert parse_probe(data)["rotation_degrees"] == -90
    del data["streams"][0]["side_data_list"][0]["rotation"]
    with pytest.raises(ValueError, match="missing rotation"):
        parse_probe(data)


def test_dynamic_frame_rotation_is_rejected() -> None:
    data = probe()
    data["streams"][0]["side_data_list"] = [matrix_entry(90)]
    data["frames"][1]["side_data_list"] = [matrix_entry(180)]
    with pytest.raises(ValueError, match="rotation changed"):
        parse_probe(data)


@pytest.mark.parametrize("target", ["stream", "frame"], ids=["stream", "frame"])
def test_display_matrix_side_data_requires_matrix_payload(target: str) -> None:
    data = probe()
    record = data["streams"][0] if target == "stream" else data["frames"][1]
    record["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": 0}]
    with pytest.raises(ValueError, match="missing displaymatrix"):
        parse_probe(data)


def test_pure_matrix_is_recorded_and_static_frame_matrix_is_allowed() -> None:
    data = probe()
    values = [0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824]
    data["streams"][0]["side_data_list"] = [matrix_entry(90, values)]
    data["frames"][1]["side_data_list"] = [matrix_entry(90, values)]
    assert parse_probe(data)["display_matrix"] == values


@pytest.mark.parametrize(
    "values",
    [[-65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824],
     [65536, 1, 0, 0, 65536, 0, 0, 0, 1073741824],
     [65536, 0, 1, 0, 65536, 0, 0, 0, 1073741824],
     [32768, 0, 0, 0, 32768, 0, 0, 0, 1073741824],
     [65536, 0, 0, 0, 65536, 0, 0, 0, 1]],
    ids=["mirror", "shear", "translate", "scale", "perspective"],
)
def test_nonstandard_display_matrix_is_rejected(values: list[int]) -> None:
    data = probe()
    data["streams"][0]["side_data_list"] = [matrix_entry(0, values)]
    with pytest.raises(ValueError, match="pure right-angle"):
        parse_probe(data)


def test_dynamic_frame_matrix_is_rejected() -> None:
    data = probe()
    data["streams"][0]["side_data_list"] = [matrix_entry(0, [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824])]
    data["frames"][1]["side_data_list"] = [matrix_entry(0, [-65536, 0, 0, 0, -65536, 0, 0, 0, 1073741824])]
    with pytest.raises(ValueError, match="displaymatrix changed"):
        parse_probe(data)


def test_showinfo_parses_negative_pts_and_ignores_unrelated_rows() -> None:
    text = showinfo_text([-20, 0, 99])
    text += "\n[Parsed_showinfo_1 @ 0x1] color_range:unknown color_space:unknown"
    text += "\n[Other_filter @ 0x1] n: 9 pts: 999 s:1x1"
    parsed = parse_showinfo(text)
    assert parsed[0] == {"output_index": 0, "pts": -20, "width": 640, "height": 360, "time_base": [1, 1000]}
    assert [frame["pts"] for frame in parsed] == [-20, 0, 99]
    assert parse_showinfo("no showinfo frame rows") == []


@pytest.mark.parametrize("replacement", ["n:   0", "n:   2"], ids=["duplicate", "skipped"])
def test_showinfo_duplicate_or_skipped_index_is_rejected(replacement: str) -> None:
    text = showinfo_text([0, 40]).replace("n:   1", replacement)
    with pytest.raises(ValueError, match="consecutive"):
        parse_showinfo(text)


def test_showinfo_missing_or_changed_time_base_is_rejected() -> None:
    text = showinfo_text([0, 40])
    with pytest.raises(ValueError, match="missing input time_base"):
        parse_showinfo("\n".join(text.splitlines()[1:]))
    text += "\n[Parsed_showinfo_1 @ 0x1] config in time_base: 1/2000, frame_rate: 25/1"
    with pytest.raises(ValueError, match="time_base changed"):
        parse_showinfo(text)


def test_export_matches_exact_source_pts_with_rescaled_time_base() -> None:
    timeline = parse_probe(probe((-20, 10, 20, 90, 100)))
    selected = select_uniform(timeline, 3)
    output_pts = [frame["pts"] * 2 for frame in selected]
    assert validate_export(timeline, selected, parse_showinfo(showinfo_text(output_pts, "1/2000")))


def test_export_rejects_pts_error_even_when_float_seconds_round_equal() -> None:
    origin = 2 ** 60
    timeline = parse_probe(probe((origin, origin + 1)))
    selected = select_uniform(timeline, 2)
    wrong = parse_showinfo(showinfo_text([origin, origin + 2]))
    with pytest.raises(ValueError, match="PTS mismatch"):
        validate_export(timeline, selected, wrong)


def test_export_rejects_count_mismatch_and_wrong_time_base() -> None:
    timeline = parse_probe(probe((10, 20)))
    selected = select_uniform(timeline, 2)
    with pytest.raises(ValueError, match="frame count"):
        validate_export(timeline, selected, parse_showinfo(showinfo_text([10])))
    with pytest.raises(ValueError, match="PTS mismatch"):
        validate_export(timeline, selected, parse_showinfo(showinfo_text([10, 20], "1/500")))


def test_export_rejects_tampered_selection_and_duplicate_output_index() -> None:
    timeline = parse_probe(probe((10, 20)))
    selected = deepcopy(select_uniform(timeline, 2))
    output = parse_showinfo(showinfo_text([10, 20]))
    selected[0]["pts"] = 11
    with pytest.raises(ValueError, match="original timeline PTS"):
        validate_export(timeline, selected, output)
    selected[0]["pts"] = 10
    output[1]["output_index"] = 0
    with pytest.raises(ValueError, match="consecutive"):
        validate_export(timeline, selected, output)
