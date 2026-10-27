"""TEST_DATA: independent explicit MVSI7 writer; no native/OpenMVS execution.

These tests prove bounded parsing of synthetic bytes, not real scene import,
image relocation, actual-depth camera correctness, or surface reconstruction.
"""

from __future__ import annotations

import hashlib
import struct

import pytest

from integrations.mvs_surface import scene


def _fixture(names=("images/第一张.jpg", "images/second.jpg")):
    """Encode field order from Interface.h, without production helpers/constants."""
    data = bytearray()
    positions = {}

    def put(label, fmt, *values):
        positions[label] = len(data)
        data.extend(struct.pack("<" + fmt, *values))

    def string(label, value):
        raw = value if isinstance(value, bytes) else value.encode("utf-8")
        put(label + ".length", "Q", len(raw))
        positions[label] = len(data)
        data.extend(raw)

    put("header", "4sII", b"MVSI", 7, 0)
    put("platform_count", "Q", 1)
    string("platform_name", "mobile rig")
    put("camera_count", "Q", 1)
    string("camera_name", "left camera")
    string("band_name", "RGB")
    put("camera_dimensions", "II", 640, 480)
    put("camera_K", "9d", 500, 2, 319.5, 0, 510, 239.5, 0, 0, 1)
    put("camera_R", "9d", 1, 0, 0, 0, 0, -1, 0, 1, 0)
    put("camera_C", "3d", 1, 2, 3)
    put("pose_count", "Q", 1)
    put("pose_R", "9d", 0, -1, 0, 1, 0, 0, 0, 0, 1)
    put("pose_C", "3d", 10, 20, 30)
    put("image_count", "Q", 2)
    for index, name in enumerate(names):
        prefix = f"image{index}"
        string(prefix + ".name", name)
        string(prefix + ".mask", "")
        put(prefix + ".platform", "I", 0)
        put(prefix + ".camera", "I", 0)
        put(prefix + ".pose", "I", 0)
        put(prefix + ".global", "I", (17, 91)[index])
        # Native optional depths may all be zero; preserve them.
        put(prefix + ".depth", "3f", 0, 0, 0)
        put(prefix + ".score_count", "Q", 1)
        put(prefix + ".score_id", "I", 1 - index)
        put(prefix + ".score_points", "I", 2)
        put(prefix + ".score_values", "4f", 1, 0.25, 0.5, 7)
    put("vertex_count", "Q", 2)
    for index, point in enumerate(((1, 2, 3), (-1, 4, 5))):
        put(f"vertex{index}.X", "3f", *point)
        put(f"vertex{index}.view_count", "Q", 1)
        put(f"vertex{index}.view_id", "I", index)
        put(f"vertex{index}.confidence", "f", 0.5)
    put("vertex_normal_count", "Q", 2)
    put("vertex_normals", "6f", 0, 0, 1, 0, 1, 0)
    put("vertex_color_count", "Q", 2)
    put("vertex_colors", "6B", 10, 20, 30, 40, 50, 60)
    put("line_count", "Q", 1)
    put("line_endpoints", "6f", -2, 0, 1, 2, 1, 4)
    put("line_view_count", "Q", 1)
    put("line_view_id", "I", 0)
    put("line_confidence", "f", 0)
    put("line_normal_count", "Q", 1)
    put("line_normals", "3f", 0, 0, 1)
    put("line_color_count", "Q", 1)
    put("line_colors", "3B", 1, 2, 3)
    put("transform", "16d", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    put("obb_rotation", "9d", 1, 0, 0, 0, 1, 0, 0, 0, 1)
    put("obb_min", "3d", -10, -20, -30)
    put("obb_max", "3d", 10, 20, 30)
    return bytes(data), positions


def _read(tmp_path, data):
    path = tmp_path / "test-data-scene.mvs"
    path.write_bytes(data)
    return scene.read_scene(path)


def _change(data, positions, label, fmt, *values):
    output = bytearray(data)
    struct.pack_into("<" + fmt, output, positions[label], *values)
    return bytes(output)


def test_v7_complete_layout_and_noncommutative_pose_composition(tmp_path):
    data, _ = _fixture()
    result = _read(tmp_path, data)
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["bytes"] == len(data)
    assert result["version"] == 7
    assert result["byte_order"] == "little"
    assert result["point_count"] == 2
    assert result["min_point_views"] == 1  # Parse native-ineligible input; runner gates >= 2.
    assert result["line_count"] == 1
    assert result["observation_count"] == 3
    assert result["point_bbox"] == {"min": [-1, 2, 3], "max": [1, 4, 5]}
    assert result["line_bbox"] == {"min": [-2, 0, 1], "max": [2, 1, 4]}
    assert "vertices" not in result
    assert result["point_normal_count"] == result["point_color_count"] == 2
    assert result["line_normal_count"] == result["line_color_count"] == 1
    first, second = result["images"]
    assert (first["index"], first["global_id"], second["global_id"]) == (0, 17, 91)
    assert first["name"] == "images/第一张.jpg"
    assert first["mask_name"] == ""
    assert (first["width"], first["height"]) == (640, 480)
    assert first["K"] == [[500, 2, 319.5], [0, 510, 239.5], [0, 0, 1]]
    assert first["R"] == [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    assert first["C"] == [12, 19, 33]
    assert first["view_scores"][0] == {"image_index": 1, "points": 2, "scale": 1, "angle": 0.25, "area": 0.5, "score": 7}
    assert [first[key] for key in ("min_depth", "avg_depth", "max_depth")] == [0, 0, 0]
    assert result["obb"]["pt_min"] == [-10, -20, -30]
    assert result["obb"]["pt_max"] == [10, 20, 30]


def test_scene_transform_is_preserved_without_applying_it_twice(tmp_path):
    data, pos = _fixture()
    altered = _change(data, pos, "transform", "4d", 1, 0, 0, 123)
    result = _read(tmp_path, altered)
    assert result["transform"][0][3] == 123
    assert result["images"][0]["C"] == [12, 19, 33]


@pytest.mark.parametrize("label,value", [
    ("platform_count", 257), ("camera_count", 257), ("pose_count", 4097),
    ("image_count", 257), ("vertex_count", 1_000_001), ("line_count", 1_000_001),
    ("vertex0.view_count", 10_000_001), ("image0.name.length", 4097),
    ("image0.score_count", 257), ("vertex_count", 999_999),
    ("platform_count", 2**64 - 1),
])
def test_counts_are_bounded_before_allocation_or_iteration(tmp_path, label, value):
    data, pos = _fixture()
    with pytest.raises(scene.SceneFormatError, match="count.*(limit|remaining)"):
        _read(tmp_path, _change(data, pos, label, "Q", value))


@pytest.mark.parametrize("remove", [1, 8, 120, 248])
def test_full_tail_must_be_present(tmp_path, remove):
    data, _ = _fixture()
    with pytest.raises(scene.SceneFormatError, match="truncated"):
        _read(tmp_path, data[:-remove])


@pytest.mark.parametrize("payload", [b"", b"MVSI", b"MVSI\x07\x00\x00\x00"])
def test_truncated_header(tmp_path, payload):
    with pytest.raises(scene.SceneFormatError, match="truncated header"):
        _read(tmp_path, payload)


def test_trailing_bytes_are_not_silently_ignored(tmp_path):
    data, _ = _fixture()
    with pytest.raises(scene.SceneFormatError, match="trailing bytes"):
        _read(tmp_path, data + b"unexpected")


@pytest.mark.parametrize("header", [
    struct.pack("<4sII", b"MVSB", 7, 0),
    struct.pack("<4sII", b"MVSI", 6, 0),
    struct.pack(">4sII", b"MVSI", 7, 0),
    struct.pack("<4sII", b"MVSI", 7, 1),
])
def test_wrong_magic_version_endian_or_reserved(tmp_path, header):
    data, _ = _fixture()
    with pytest.raises(scene.SceneFormatError):
        _read(tmp_path, header + data[12:])


@pytest.mark.parametrize("label,value,match", [
    ("image0.platform", 1, "platform reference"),
    ("image0.camera", 1, "camera reference"),
    ("image0.pose", 1, "pose reference"),
    ("image0.pose", 0xFFFFFFFF, "pose reference"),
    ("image0.global", 0xFFFFFFFF, "global ID"),
    ("image1.global", 17, "global ID"),
    # 17 is a valid global ID, deliberately invalid as a local array index.
    ("image0.score_id", 17, "local image index"),
    ("vertex0.view_id", 17, "local image index"),
    ("line_view_id", 17, "local image index"),
])
def test_invalid_references_and_local_global_id_distinction(tmp_path, label, value, match):
    data, pos = _fixture()
    with pytest.raises(scene.SceneFormatError, match=match):
        _read(tmp_path, _change(data, pos, label, "I", value))


@pytest.mark.parametrize("name", [
    "../a.jpg", "images/../a.jpg", "images/./a.jpg", "/absolute.jpg",
    "\\\\server\\share\\a.jpg", "C:\\a.jpg", "images/a.jpg:stream",
    "images//a.jpg", "images/a.jpg.", "images/a.jpg ", "images/CON.jpg",
    "images/a\x00.jpg", "images/a\n.jpg", "", b"images/\xff.jpg",
])
def test_unsafe_or_ambiguous_image_paths_rejected(tmp_path, name):
    data, _ = _fixture((name, "images/second.jpg"))
    with pytest.raises(scene.SceneFormatError):
        _read(tmp_path, data)


@pytest.mark.parametrize("names", [
    ("images/a.jpg", "images\\a.jpg"),
    ("images/a.jpg", "IMAGES/A.JPG"),
    ("images/café.jpg", "images/cafe\u0301.jpg"),
])
def test_path_aliases_cannot_overwrite_staged_images(tmp_path, names):
    data, _ = _fixture(names)
    with pytest.raises(scene.SceneFormatError, match="aliased image path"):
        _read(tmp_path, data)


def test_unicode_spelling_preserved_only_separators_normalized(tmp_path):
    name = "images\\cafe\u0301.jpg"
    data, _ = _fixture((name, "images/second.jpg"))
    first = _read(tmp_path, data)["images"][0]
    assert first["name_raw"] == name
    assert first["name"] == "images/cafe\u0301.jpg"


@pytest.mark.parametrize("label,fmt", [
    ("camera_K", "d"), ("camera_R", "d"), ("camera_C", "d"),
    ("pose_R", "d"), ("pose_C", "d"), ("image0.depth", "f"),
    ("image0.score_values", "f"), ("vertex0.X", "f"),
    ("vertex0.confidence", "f"), ("vertex_normals", "f"),
    ("line_endpoints", "f"), ("line_confidence", "f"),
    ("line_normals", "f"), ("transform", "d"), ("obb_rotation", "d"),
    ("obb_min", "d"), ("obb_max", "d"),
])
def test_nonfinite_values_rejected_even_in_optional_tail(tmp_path, label, fmt):
    data, pos = _fixture()
    with pytest.raises(scene.SceneFormatError, match="non-finite"):
        _read(tmp_path, _change(data, pos, label, fmt, float("nan")))


@pytest.mark.parametrize("rotation,match", [
    ((1, 0, 0, 0, 1, 0, 0, 0, -1), "proper rotation"),
    ((2, 0, 0, 0, 1, 0, 0, 0, 1), "orthonormal rotation"),
])
def test_reflections_and_nonrigid_rotation_rejected(tmp_path, rotation, match):
    data, pos = _fixture()
    with pytest.raises(scene.SceneFormatError, match=match):
        _read(tmp_path, _change(data, pos, "camera_R", "9d", *rotation))


def test_normalized_camera_is_outside_runner_profile(tmp_path):
    data, pos = _fixture()
    with pytest.raises(scene.SceneFormatError, match="normalized/unknown-resolution"):
        _read(tmp_path, _change(data, pos, "camera_dimensions", "II", 0, 0))


def test_nonpositive_focal_length_rejected(tmp_path):
    data, pos = _fixture()
    with pytest.raises(scene.SceneFormatError, match="focal lengths"):
        _read(tmp_path, _change(data, pos, "camera_K", "d", 0))


@pytest.mark.parametrize("label", ["vertex_normal_count", "vertex_color_count"])
def test_attributes_must_match_point_count_or_be_empty(tmp_path, label):
    data, pos = _fixture()
    with pytest.raises(scene.SceneFormatError, match="empty or match"):
        _read(tmp_path, _change(data, pos, label, "Q", 1))


def test_observation_limit_is_cumulative_across_points_and_lines(tmp_path, monkeypatch):
    data, _ = _fixture()
    monkeypatch.setattr(scene, "MAX_OBSERVATIONS", 2)
    with pytest.raises(scene.SceneFormatError, match="lines.views count 1 exceeds limit 0"):
        _read(tmp_path, data)


def test_file_size_limit_checked_before_parsing(tmp_path, monkeypatch):
    data, _ = _fixture()
    monkeypatch.setattr(scene, "MAX_FILE_BYTES", 100)
    with pytest.raises(scene.SceneFormatError, match="byte limit"):
        _read(tmp_path, data)


def test_optional_attributes_can_be_empty(tmp_path):
    data, pos = _fixture()
    # Drop complete optional arrays while retaining their independent count fields.
    output = bytearray(data)
    for label, size in (("line_colors", 3), ("line_normals", 12), ("vertex_colors", 6), ("vertex_normals", 24)):
        offset = pos[label]
        del output[offset : offset + size]
        struct.pack_into("<Q", output, offset - 8, 0)
    result = _read(tmp_path, bytes(output))
    assert result["point_normal_count"] == result["point_color_count"] == 0
    assert result["line_normal_count"] == result["line_color_count"] == 0
