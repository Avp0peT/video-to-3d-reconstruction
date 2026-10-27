"""Restricted PLY reader checks; stdlib input generation and no native imports."""
import importlib.util
from pathlib import Path
import struct

import pytest


MODULE = Path(__file__).resolve().parents[1] / "integrations/meshprep/ply_io.py"
SPEC = importlib.util.spec_from_file_location("meshprep_ply_under_test", MODULE)
ply = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ply)


def header(encoding="ascii", vertex_props="property double x\nproperty double y\nproperty double z\n",
           face_props="property list uchar int vertex_indices\n", vertices=3, faces=1):
    return (f"ply\nformat {encoding} 1.0\ncomment fixture\nelement vertex {vertices}\n{vertex_props}"
            f"element face {faces}\n{face_props}end_header\n").encode("ascii")


def write(tmp_path, data):
    path = tmp_path / "input.ply"
    path.write_bytes(data)
    return path


def basic():
    return header() + b"1000000000 1000000000 1000000000\n1000000001 1000000000 1000000000\n1000000000 1000000001 1000000000\n3 0 1 2\n"


def test_ascii_double_preserves_unit_edges_at_large_origin_and_source(tmp_path):
    path = write(tmp_path, basic())
    before = path.read_bytes()
    points, faces, meta = ply.read_ply(path)
    assert points == [(1e9, 1e9, 1e9), (1e9 + 1, 1e9, 1e9), (1e9, 1e9 + 1, 1e9)]
    assert faces == [(0, 1, 2)]
    assert meta["coordinate_types"] == dict(x="double", y="double", z="double")
    assert meta["repairs_applied"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("encoding,endian", [("binary_little_endian", "<"), ("binary_big_endian", ">")])
def test_binary_declared_order_double_float_color_and_face_scalar(tmp_path, encoding, endian):
    props = "property uchar red\nproperty double z\nproperty float nx\nproperty double x\nproperty double y\nproperty ushort green\n"
    face_props = "property uint material\nproperty list uchar int vertex_index\nproperty float confidence\n"
    data = header(encoding, props, face_props)
    expected = [(1e9, 1e9, 1e9), (1e9 + 1, 1e9, 1e9), (1e9, 1e9 + 1, 1e9)]
    for x, y, z in expected:
        data += struct.pack(endian + "BdfddH", 23, z, .25, x, y, 65535)
    data += struct.pack(endian + "IBiiif", 42, 3, 0, 1, 2, .5)
    points, faces, meta = ply.read_ply(write(tmp_path, data))
    assert points == expected and faces == [(0, 1, 2)]
    assert meta["format"] == encoding
    assert [(p["element"], p["name"]) for p in meta["ignored_scalar_properties"]] == [
        ("vertex", "red"), ("vertex", "nx"), ("vertex", "green"), ("face", "material"), ("face", "confidence")]


def test_ascii_reordered_attributes_are_parsed_not_skipped_as_lines(tmp_path):
    data = header(vertex_props="property uchar red\nproperty double y\nproperty double x\nproperty double z\n")
    data += b"7 2 1 3 8 5 4 6 9 8 7 9 3 0 1 2\n"
    points, faces, meta = ply.read_ply(write(tmp_path, data))
    assert points == [(1., 2., 3.), (4., 5., 6.), (7., 8., 9.)]
    assert faces == [(0, 1, 2)]
    assert meta["ignored_scalar_properties"] == [{"element": "vertex", "name": "red", "type": "uchar"}]


def test_face_element_can_precede_vertices_and_crlf_header(tmp_path):
    data = (b"ply\r\nformat ascii 1.0\r\nelement face 1\r\nproperty list uint8 int32 vertex_indices\r\n"
            b"element vertex 3\r\nproperty float64 x\r\nproperty float64 y\r\nproperty float64 z\r\nend_header\r\n"
            b"3 0 1 2\n0 0 0\n1 0 0\n0 1 0\n")
    points, faces, _ = ply.read_ply(write(tmp_path, data))
    assert points[1] == (1., 0., 0.) and faces == [(0, 1, 2)]


@pytest.mark.parametrize("data", [
    b"notply\n", b"ply\nformat ascii 1.0\n", basic().replace(b"ascii 1.0", b"ascii 2.0"),
    basic().replace(b"format ascii 1.0", b"format ascii 1.0\nformat ascii 1.0"),
    basic().replace(b"element face 1", b"element edge 1"),
    basic().replace(b"element face 1", b"element vertex 1"),
    basic().replace(b"property double x", b"property int x"),
    basic().replace(b"property double z", b"property double x"),
    basic().replace(b"property double z\n", b""),
    basic().replace(b"property double z", b"property list uchar float z"),
    basic().replace(b"list uchar int", b"list uchar double"),
    basic().replace(b"list uchar int", b"list float int"),
    basic().replace(b"vertex_indices", b"something_else"),
    basic().replace(b"end_header", b"property list uchar int vertex_index\nend_header"),
    basic().replace(b"end_header", b"unknown directive\nend_header"),
    b"ply\ncomment " + b"x" * 65536 + b"\n",
], ids=lambda data: "header_" + str(len(data)) + "_" + data[:12].hex())
def test_bad_or_unsupported_headers_are_rejected(tmp_path, data):
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, data))


@pytest.mark.parametrize("suffix", [b"2 0 1\n", b"4 0 1 2 0\n", b"3 -1 1 2\n", b"3 0 1 3\n",
                                   b"3 0 true 2\n", b"3 0 1.0 2\n", b"3 0 1\n", b"3 0 1 2147483648\n"])
def test_bad_triangles_indices_and_truncation(tmp_path, suffix):
    data = basic().rsplit(b"3 0 1 2\n", 1)[0] + suffix
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, data))


@pytest.mark.parametrize("value", [b"nan", b"inf", b"-inf", b"1e999", b"1_000", b"garbage"])
def test_nonfinite_or_malformed_coordinates_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, basic().replace(b"1000000000", value, 1)))


@pytest.mark.parametrize("encoding,endian", [("binary_little_endian", "<"), ("binary_big_endian", ">")])
def test_binary_truncation_extra_bytes_and_nonfinite(tmp_path, encoding, endian):
    payload = struct.pack(endian + "9dB3i", 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 1, 2)
    for damaged in (payload[:-1], payload + b"\n", struct.pack(endian + "d", float("nan")) + payload[8:]):
        with pytest.raises(ValueError):
            ply.read_ply(write(tmp_path, header(encoding) + damaged))


def test_ascii_extra_data_and_invalid_ignored_scalar_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, basic() + b"unexpected"))
    data = header(vertex_props="property uchar red\nproperty double x\nproperty double y\nproperty double z\n")
    data += b"256 0 0 0\n1 1 0 0\n2 0 1 0\n3 0 1 2\n"
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, data))


@pytest.mark.parametrize("kwargs", [dict(max_bytes=20), dict(max_vertices=2), dict(max_faces=True),
                                   dict(max_bytes=0), dict(max_vertices=-1), dict(max_faces=1.5)])
def test_budgets_are_strict(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ply.read_ply(write(tmp_path, basic()), **kwargs)


def test_declared_face_limit_is_checked_before_payload(tmp_path):
    with pytest.raises(ValueError, match="face count exceeds"):
        ply.read_ply(write(tmp_path, header(faces=200001)))


def test_missing_file_is_value_error(tmp_path):
    with pytest.raises(ValueError):
        ply.read_ply(tmp_path / "missing.ply")
