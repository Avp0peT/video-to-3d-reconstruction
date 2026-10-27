"""TEST_DATA: independent synthetic OpenMVS DMAP parser tests.

The byte writer below uses the upstream HeaderDepthDataRaw layout and
ExportDepthDataRaw write order, never the parser's constants or serializers.
Passing these tests does not establish real OpenMVS depth or camera correctness.
All test files are confined to pytest's tmp_path.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.openmvs_depth import dmap


_K = (110.0, 3.0, 1.0, 4.0, 120.0, 0.5, 0.25, 0.125, 1.0)
_R = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
_C = (1.25, -2.5, 3.75)
_DEPTH = (0.0, 1.25, 2.5, 7.5, 12.0, 20.0)
_NORMAL = (
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
    0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
)
_CONFIDENCE = (0.0, 0.125, 0.25, 0.5, 0.75, 2.0)
_VIEWS = (255, 255, 255, 255, 0, 255, 255, 255, 1, 255, 255, 255,
          0, 1, 255, 255, 1, 0, 255, 255, 0, 1, 255, 255)
_NAME = b"images/\xe4\xb8\xad\xff-view.png"  # Deliberately not valid UTF-8.


def _fixture(
    *,
    flags: int = 15,
    reserved: int = 0,
    image_size: tuple[int, int] = (6, 4),
    depth_size: tuple[int, int] = (3, 2),
    image_name: bytes = _NAME,
    ids: tuple[int, ...] = (7, 11, 13),
    d_min: float = 1.0,
    d_max: float = 20.0,
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """Pack explicit little-endian TEST_DATA and record section byte boundaries."""
    parts: list[bytes] = []
    sections: dict[str, tuple[int, int]] = {}
    offset = 0

    def append(name: str, data: bytes) -> None:
        nonlocal offset
        sections[name] = (offset, offset + len(data))
        parts.append(data)
        offset += len(data)

    append("header", struct.pack("<2sBBIIIIff", b"DR", flags, reserved,
                                 *image_size, *depth_size, d_min, d_max))
    append("name_length", struct.pack("<H", len(image_name)))
    append("name", image_name)
    append("id_count", struct.pack("<I", len(ids)))
    append("ids", struct.pack(f"<{len(ids)}I", *ids))
    append("K", struct.pack("<9d", *_K))
    append("R", struct.pack("<9d", *_R))
    append("C", struct.pack("<3d", *_C))
    append("depth", struct.pack("<6f", *_DEPTH))
    if flags & 2:
        append("normal", struct.pack("<18f", *_NORMAL))
    if flags & 4:
        append("confidence", struct.pack("<6f", *_CONFIDENCE))
    if flags & 8:
        append("views", struct.pack("<24B", *_VIEWS))
    return b"".join(parts), sections


def _write(tmp_path: Path, raw: bytes) -> Path:
    path = tmp_path / "synthetic-test-data.dmap"
    path.write_bytes(raw)
    return path


def test_windows_path_and_fd_ctime_channels_may_differ(tmp_path, monkeypatch):
    raw, _ = _fixture()
    path = _write(tmp_path, raw)
    real_fstat = dmap.os.fstat

    def changed_channel(fd):
        s = real_fstat(fd)
        return SimpleNamespace(**{k: getattr(s, k) for k in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_mode')},
            st_ctime_ns=s.st_ctime_ns + 123456789)

    monkeypatch.setattr(dmap.os, 'fstat', changed_channel)
    assert dmap.read_dmap(path).sha256 == hashlib.sha256(raw).hexdigest()


def test_ctime_change_within_fd_channel_still_rejected(tmp_path, monkeypatch):
    raw, _ = _fixture()
    path = _write(tmp_path, raw)
    real_fstat = dmap.os.fstat
    calls = 0

    def changed_channel(fd):
        nonlocal calls
        calls += 1
        s = real_fstat(fd)
        return SimpleNamespace(**{k: getattr(s, k) for k in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_mode')},
            st_ctime_ns=s.st_ctime_ns + calls)

    monkeypatch.setattr(dmap.os, 'fstat', changed_channel)
    with pytest.raises(dmap.DMapError, match='changed'):
        dmap.read_dmap(path)


def _patch(raw: bytes, offset: int, fmt: str, value: int | float) -> bytes:
    replacement = struct.pack(fmt, value)
    return raw[:offset] + replacement + raw[offset + len(replacement):]


def _forbid_array_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid or oversized input reached NumPy frombuffer")

    monkeypatch.setattr(dmap.np, "frombuffer", forbidden)


@pytest.mark.parametrize("flags", [1, 3, 5, 7, 9, 11, 13, 15])
def test_all_optional_combinations_preserve_exact_layout(tmp_path: Path, flags: int) -> None:
    raw, sections = _fixture(flags=flags, reserved=0xA5)
    path = _write(tmp_path, raw)
    result = dmap.read_dmap(path)

    assert sections["header"] == (0, 28)
    assert isinstance(result, dmap.DepthMap)
    assert result.path == str(path.resolve())
    assert result.sha256 == hashlib.sha256(raw).hexdigest()
    assert result.file_bytes == len(raw)
    assert result.flags == flags
    assert result.reserved == 0xA5
    assert result.image_size == (6, 4)
    assert result.depth_size == (3, 2)
    assert result.d_min == 1.0
    assert result.d_max == 20.0
    assert result.image_name_bytes == _NAME
    assert result.view_ids == (7, 11, 13)

    expected = {
        "K": (_K, (3, 3), "<f8"),
        "R": (_R, (3, 3), "<f8"),
        "C": (_C, (3,), "<f8"),
        "depth": (_DEPTH, (2, 3), "<f4"),
        "normal": (_NORMAL, (2, 3, 3), "<f4") if flags & 2 else None,
        "confidence": (_CONFIDENCE, (2, 3), "<f4") if flags & 4 else None,
        "views": (_VIEWS, (2, 3, 4), "u1") if flags & 8 else None,
    }
    for name, specification in expected.items():
        actual = getattr(result, name)
        if specification is None:
            assert actual is None, name
            continue
        values, shape, dtype = specification
        assert actual.shape == shape, name
        assert actual.dtype == np.dtype(dtype), name
        assert actual.flags.c_contiguous, name
        np.testing.assert_array_equal(actual, np.array(values, dtype=dtype).reshape(shape))
    assert path.read_bytes() == raw


def test_frozen_record_and_array_bases_cannot_be_made_writeable(tmp_path: Path) -> None:
    raw, _ = _fixture()
    result = dmap.read_dmap(_write(tmp_path, raw))
    with pytest.raises(FrozenInstanceError):
        result.flags = 1
    for name in ("K", "R", "C", "depth", "normal", "confidence", "views"):
        array = getattr(result, name)
        with pytest.raises(ValueError):
            array.flat[0] = 99
        with pytest.raises(ValueError):
            array.setflags(write=True)
        base = array.base
        while isinstance(base, np.ndarray):
            assert not base.flags.writeable
            with pytest.raises(ValueError):
                base.setflags(write=True)
            base = base.base
        assert isinstance(base, bytes) or (isinstance(base, memoryview) and base.readonly)


def test_equal_resolutions_and_depth_range_tolerance_are_accepted(tmp_path: Path) -> None:
    # The native importer supports equal resolutions. dMin/dMax are metadata,
    # not strict per-pixel cutoffs; zero is the invalid-depth sentinel.
    raw, _ = _fixture(image_size=(3, 2), d_min=2.0, d_max=12.0)
    result = dmap.read_dmap(_write(tmp_path, raw))
    assert result.image_size == result.depth_size == (3, 2)
    np.testing.assert_array_equal(result.depth, np.array(_DEPTH, dtype="<f4").reshape(2, 3))
    assert result.confidence[1, 2] == 2.0  # Confidence is not universally <= 1.


def test_filename_decoding_is_explicit_and_strict(tmp_path: Path) -> None:
    raw, _ = _fixture()
    result = dmap.read_dmap(_write(tmp_path, raw))
    with pytest.raises(UnicodeDecodeError):
        _ = result.image_name_utf8
    name = "images/\u6d4b\u8bd5.png"
    raw, _ = _fixture(image_name=name.encode("utf-8"))
    assert dmap.read_dmap(_write(tmp_path, raw)).image_name_utf8 == name


@pytest.mark.parametrize("section", [
    "header", "name_length", "name", "id_count", "ids", "K", "R", "C",
    "depth", "normal", "confidence", "views",
])
def test_truncation_at_each_structural_boundary(tmp_path: Path, section: str) -> None:
    raw, sections = _fixture()
    cut = sections[section][1] - 1
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, raw[:cut]))


@pytest.mark.parametrize("corruption", ["empty", "bad_magic", "reversed_magic", "trailing"])
def test_invalid_envelope(tmp_path: Path, corruption: str) -> None:
    raw, _ = _fixture()
    corruptions = {"empty": b"", "bad_magic": b"XX" + raw[2:],
                   "reversed_magic": b"RD" + raw[2:], "trailing": raw + b"\x00"}
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, corruptions[corruption]))


@pytest.mark.parametrize("flags", [0, 2, 14, 17, 129, 255])
def test_missing_depth_and_unknown_flags_are_rejected(tmp_path: Path, flags: int) -> None:
    raw, _ = _fixture()
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, _patch(raw, 2, "<B", flags)))


@pytest.mark.parametrize("offset,value", [(4, 0), (8, 0), (12, 0), (16, 0), (12, 7), (16, 5)])
def test_invalid_dimensions(tmp_path: Path, offset: int, value: int) -> None:
    raw, _ = _fixture()
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, _patch(raw, offset, "<I", value)))


@pytest.mark.parametrize("image_name", [b"", b"image\x00.png"])
def test_empty_or_nul_containing_image_name(tmp_path: Path, image_name: bytes) -> None:
    raw, _ = _fixture(image_name=image_name)
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, raw))


@pytest.mark.parametrize("ids", [(), (7, 7), (0xFFFFFFFF,), tuple(range(256))])
def test_invalid_view_ids(tmp_path: Path, ids: tuple[int, ...]) -> None:
    raw, _ = _fixture(ids=ids)
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, raw))


@pytest.mark.parametrize("ids", [(0,), tuple(range(255))])
def test_view_id_count_endpoints(tmp_path: Path, ids: tuple[int, ...]) -> None:
    raw, _ = _fixture(flags=1, ids=ids)
    assert dmap.read_dmap(_write(tmp_path, raw)).view_ids == ids


@pytest.mark.parametrize("d_min,d_max", [
    (0.0, 20.0), (-1.0, 20.0), (21.0, 20.0), (1.0, 0.0),
    (float("nan"), 20.0), (float("inf"), 20.0), (1.0, float("nan")), (1.0, float("inf")),
])
def test_invalid_depth_range_metadata(tmp_path: Path, d_min: float, d_max: float) -> None:
    raw, _ = _fixture(d_min=d_min, d_max=d_max)
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, raw))


@pytest.mark.parametrize("section", ["K", "R", "C", "depth", "normal", "confidence"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_array_values(tmp_path: Path, section: str, value: float) -> None:
    raw, sections = _fixture()
    fmt = "<d" if section in ("K", "R", "C") else "<f"
    bad = _patch(raw, sections[section][0], fmt, value)
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, bad))


@pytest.mark.parametrize("section", ["depth", "confidence"])
def test_negative_depth_and_confidence(tmp_path: Path, section: str) -> None:
    raw, sections = _fixture()
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, _patch(raw, sections[section][0], "<f", -0.25)))


def test_exact_resource_limit_endpoints(tmp_path: Path) -> None:
    raw, _ = _fixture(image_size=(3, 2))
    path = _write(tmp_path, raw)
    assert dmap.read_dmap(path, max_pixels=6, max_bytes=len(raw)).file_bytes == len(raw)


@pytest.mark.parametrize("case", ["pixel_limit", "image_pixel_limit", "byte_limit", "huge_dimensions",
                                  "huge_id_count", "overlong_name", "truncated", "trailing"])
def test_resource_and_length_guards_precede_numpy_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    raw, sections = _fixture(image_size=(3, 2))
    kwargs: dict[str, int] = {}
    if case == "pixel_limit":
        kwargs["max_pixels"] = 5
    elif case == "image_pixel_limit":
        raw = _patch(raw, 4, "<I", 4)
        kwargs["max_pixels"] = 6
    elif case == "byte_limit":
        kwargs["max_bytes"] = len(raw) - 1
    elif case == "huge_dimensions":
        raw = raw[:4] + struct.pack("<IIII", *([0xFFFFFFFF] * 4)) + raw[20:]
    elif case == "huge_id_count":
        raw = _patch(raw, sections["id_count"][0], "<I", 0xFFFFFFFF)
    elif case == "overlong_name":
        raw = _patch(raw, sections["name_length"][0], "<H", 0xFFFF)
    elif case == "truncated":
        raw = raw[:-1]
    elif case == "trailing":
        raw += b"\x00"
    path = _write(tmp_path, raw)
    _forbid_array_materialization(monkeypatch)
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(path, **kwargs)


@pytest.mark.parametrize("limit", ["max_pixels", "max_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_resource_limit_types_and_values(tmp_path: Path, limit: str, value: object) -> None:
    raw, _ = _fixture()
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, raw), **{limit: value})


@pytest.mark.parametrize("limit,value", [("max_pixels", 16_777_217), ("max_bytes", 536_870_913)])
def test_hard_resource_ceilings_cannot_be_raised(tmp_path: Path, limit: str, value: int) -> None:
    raw, _ = _fixture()
    with pytest.raises(dmap.DMapError):
        dmap.read_dmap(_write(tmp_path, raw), **{limit: value})
