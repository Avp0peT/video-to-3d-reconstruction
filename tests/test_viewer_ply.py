"""PLY viewer tests use explicitly synthetic bytes; no native geometry imports."""
import hashlib
import math
from pathlib import Path
import struct
import types

import pytest


MODULE = Path(__file__).resolve().parents[1] / "integrations/viewer/ply_data.py"
ply = types.ModuleType("viewer_ply_under_test")
exec(compile(MODULE.read_text(encoding="utf-8"), str(MODULE), "exec"), ply.__dict__)


def header(encoding="ascii", vertices=3, faces=None, vertex_props=None, face_props=None):
    vertex_props = vertex_props or "property double x\nproperty double y\nproperty double z\n"
    data = f"ply\nformat {encoding} 1.0\ncomment synthetic test fixture\nelement vertex {vertices}\n{vertex_props}"
    if faces is not None:
        face_props = face_props or "property list uchar int vertex_indices\n"
        data += f"element face {faces}\n{face_props}"
    return (data + "end_header\n").encode("ascii")


def write(tmp_path, data):
    path = tmp_path / "input.ply"
    path.write_bytes(data)
    return path


def mesh():
    return header(faces=1) + b"1000000000 0 0\n1000000001 0 0\n1000000000 1 0\n3 2 0 1\n"


@pytest.mark.parametrize("encoding,endian", [("binary_little_endian", "<"), ("binary_big_endian", ">")])
def test_binary_rgb_point_cloud_preserves_all_2393_records(tmp_path, encoding, endian):
    props = "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
    points = [[float(index), -float(index), index / 4] for index in range(2393)]
    colors = [[index % 256, (index * 3) % 256, 255 - index % 256] for index in range(2393)]
    data = header(encoding, vertices=len(points), vertex_props=props)
    data += b"".join(struct.pack(endian + "fffBBB", *point, *color) for point, color in zip(points, colors))
    path = write(tmp_path, data)
    result = ply.read_ply(path)
    assert result["vertices"] == points
    assert result["colors"] == colors
    assert result["faces"] == []
    assert result["format"] == encoding
    assert result["bytes"] == len(data)
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["ignored_properties"] == []
    assert path.read_bytes() == data


def test_ascii_double_202_vertex_400_face_mesh_preserves_precision_and_order(tmp_path):
    points = [[1e9, 1e9, 1e9 + 0.125], [1e9, 1e9, 1e9 - 0.125]]
    points += [[1e9 + math.cos(index * math.tau / 200), 1e9 + math.sin(index * math.tau / 200), 1e9] for index in range(200)]
    triangles = []
    for index in range(200):
        current, following = 2 + index, 2 + (index + 1) % 200
        triangles.extend([[0, current, following], [1, following, current]])
    data = header(vertices=202, faces=400)
    data += "".join(" ".join(format(value, ".17g") for value in point) + "\n" for point in points).encode("ascii")
    data += "".join("3 " + " ".join(map(str, triangle)) + "\n" for triangle in triangles).encode("ascii")
    result = ply.read_ply(write(tmp_path, data))
    assert result["vertices"] == points
    assert result["faces"] == triangles
    assert result["colors"] is None
    assert result["vertices"][0][2] - result["vertices"][1][2] == 0.25


def test_reordered_properties_rgb_and_ignored_scalars(tmp_path):
    props = "property uchar blue\nproperty double z\nproperty float nx\nproperty uint8 red\nproperty double x\nproperty double y\nproperty uchar green\n"
    face_props = "property float confidence\nproperty list uint8 int32 vertex_index\nproperty short label\n"
    data = header(vertices=3, faces=1, vertex_props=props, face_props=face_props)
    data += b"31 3 .5 11 1 2 21\n32 6 .75 12 4 5 22\n33 9 1 13 7 8 23\n.25 3 2 0 1 -7\n"
    result = ply.read_ply(write(tmp_path, data))
    assert result["vertices"] == [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    assert result["colors"] == [[11, 21, 31], [12, 22, 32], [13, 23, 33]]
    assert result["faces"] == [[2, 0, 1]]
    assert result["ignored_properties"] == [
        {"element": "vertex", "name": "nx", "type": "float"},
        {"element": "face", "name": "confidence", "type": "float"},
        {"element": "face", "name": "label", "type": "short"},
    ]


def test_face_before_vertices_and_crlf(tmp_path):
    data = (b"ply\r\nformat ascii 1.0\r\nelement face 1\r\nproperty list uchar int vertex_indices\r\n"
            b"element vertex 3\r\nproperty float64 x\r\nproperty float64 y\r\nproperty float64 z\r\nend_header\r\n"
            b"3 2 0 1\n0 0 0\n1 0 0\n0 1 0\n")
    result = ply.read_ply(write(tmp_path, data))
    assert result["faces"] == [[2, 0, 1]]
    assert result["vertices"] == [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]


def test_empty_cloud_and_declared_empty_faces(tmp_path):
    for faces in (None, 0):
        result = ply.read_ply(write(tmp_path, header(vertices=0, faces=faces)))
        assert result["vertices"] == result["faces"] == []
        assert result["colors"] is None


@pytest.mark.parametrize("value", [b"nan", b"inf", b"-inf", b"1e999", b"1_000", b"garbage"])
def test_invalid_coordinates(tmp_path, value):
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, mesh().replace(b"1000000000", value, 1)))


@pytest.mark.parametrize("suffix", [b"2 0 1", b"4 0 1 2 0", b"3 -1 1 2", b"3 0 1 3", b"3 0 0 1", b"3 0 1.0 2", b"3 0 1", b"3 0 1 2147483648"])
def test_invalid_triangles_and_truncation(tmp_path, suffix):
    data = mesh().rsplit(b"3 2 0 1\n", 1)[0] + suffix
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, data))


@pytest.mark.parametrize("encoding,endian", [("binary_little_endian", "<"), ("binary_big_endian", ">")])
def test_binary_truncation_trailing_bytes_and_nonfinite(tmp_path, encoding, endian):
    data = struct.pack(endian + "9dB3i", 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 2, 0, 1)
    for damaged in (data[:-1], data + b"\n", struct.pack(endian + "d", float("nan")) + data[8:]):
        with pytest.raises(ValueError):
            ply.read_ply(write(tmp_path, header(encoding, faces=1) + damaged))


def test_ascii_trailing_data_is_rejected_but_whitespace_allowed(tmp_path):
    with pytest.raises(ValueError, match="trailing"):
        ply.read_ply(write(tmp_path, mesh() + b"unexpected"))
    assert ply.read_ply(write(tmp_path, mesh() + b"\n \t\r\n"))["faces"] == [[2, 0, 1]]


@pytest.mark.parametrize("extra,value", [("property uchar red\nproperty uchar green\nproperty uchar blue\n", "256 0 0"),
                                         ("property float nx\n", "nan"), ("property float nx\n", "3.5e38"),
                                         ("property short label\n", "32768")])
def test_color_and_ignored_scalar_validation(tmp_path, extra, value):
    props = "property double x\nproperty double y\nproperty double z\n" + extra
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, header(vertices=1, vertex_props=props) + f"0 0 0 {value}\n".encode("ascii")))


@pytest.mark.parametrize("data", [
    b"notply\n", b"ply\nformat ascii 1.0\n", mesh().replace(b"ascii 1.0", b"ascii 2.0"),
    mesh().replace(b"format ascii 1.0", b"format ascii 1.0\nformat ascii 1.0"),
    mesh().replace(b"element face 1", b"element edge 1"),
    mesh().replace(b"element face 1", b"element vertex 1"),
    mesh().replace(b"element vertex 3", b"element vertex -1"),
    mesh().replace(b"property double z", b"property double x"),
    mesh().replace(b"property double z\n", b""),
    mesh().replace(b"property double z", b"property list uchar float z"),
    mesh().replace(b"property double z", b"property double z\nproperty uchar red"),
    mesh().replace(b"property double z", b"property double z\nproperty uchar red\nproperty float green\nproperty uchar blue"),
    mesh().replace(b"list uchar int", b"list uchar double"),
    mesh().replace(b"vertex_indices", b"unsupported_list"),
    mesh().replace(b"end_header", b"property list uchar int vertex_index\nend_header"),
    mesh().replace(b"end_header", b"unknown directive\nend_header"),
    b"ply\ncomment " + b"x" * 65536 + b"\n",
], ids=["magic", "truncated", "version", "duplicate-format", "edge", "duplicate-element", "negative-count", "duplicate-property", "missing-z", "vertex-list", "partial-rgb", "float-rgb", "float-index", "extra-list", "second-list", "unknown", "oversized-header"])
def test_unsupported_or_malformed_headers(tmp_path, data):
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, data))


@pytest.mark.parametrize("kwargs", [dict(max_bytes=20), dict(max_vertices=2), dict(max_faces=True),
                                   dict(max_bytes=0), dict(max_vertices=-1), dict(max_faces=1.5)])
def test_budget_validation(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, mesh()), **kwargs)


def test_exact_budget_boundary_and_header_count_budget(tmp_path):
    data = mesh()
    result = ply.read_ply(write(tmp_path, data), max_bytes=len(data), max_vertices=3, max_faces=1)
    assert len(result["vertices"]) == 3
    for data, message in ((header(vertices=100001), "vertex count"), (header(faces=200001), "face count")):
        with pytest.raises(ValueError, match=message):
            ply.read_ply(write(tmp_path, data))


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="cannot read PLY"):
        ply.read_ply(tmp_path / "missing.ply")
