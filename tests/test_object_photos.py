"""Synthetic, independent photo geometry and source-format boundary checks."""

from __future__ import annotations

import hashlib
from io import BytesIO
import struct
import zlib

import numpy as np
from PIL import Image
import pytest

from integrations.object_input import photos
from kozlov_lab.io.errors import FrameIssue


def _encode(image: Image.Image, *, orientation: int | None = None,
            container: str = "PNG", **options: object) -> bytes:
    buffer = BytesIO()
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(buffer, format=container, **options)
    return buffer.getvalue()


def _output_pixels(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        assert image.format == "PNG" and image.mode == "RGB"
        assert image.getexif().get(274) is None
        assert "exif" not in image.info
        pixels = np.asarray(image).copy()
    assert pixels.dtype == np.uint8
    return pixels


def _chunk(kind: bytes, body: bytes) -> bytes:
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))


def _raw_png(width: int, height: int, *, depth: int = 8,
             color_type: int = 2, row: bytes = b"") -> bytes:
    # Deliberately construct encoded depth independently of Pillow's decoder.
    header = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress((b"\x00" + row) * height))
            + _chunk(b"IEND", b""))


def _forbid_decode(*args: object, **kwargs: object) -> None:
    raise AssertionError("full decode must not run before header/byte-limit rejection")


@pytest.mark.parametrize("orientation,expected_labels", [
    (1, [[1, 2, 3], [4, 5, 6]]),
    (2, [[3, 2, 1], [6, 5, 4]]),
    (3, [[6, 5, 4], [3, 2, 1]]),
    (4, [[4, 5, 6], [1, 2, 3]]),
    (5, [[1, 4], [2, 5], [3, 6]]),
    (6, [[4, 1], [5, 2], [6, 3]]),
    (7, [[6, 3], [5, 2], [4, 1]]),
    (8, [[3, 6], [2, 5], [1, 4]]),
])
def test_all_exif_orientations_have_explicit_pixels_and_apply_once(
    orientation: int, expected_labels: list[list[int]],
) -> None:
    colors = np.array([[i, i + 20, 200 - i] for i in range(7)], dtype=np.uint8)
    raw = colors[np.array([[1, 2, 3], [4, 5, 6]])]
    payload = _encode(Image.fromarray(raw), orientation=orientation)
    output, meta = photos.prepare_photo(payload)
    expected = colors[np.asarray(expected_labels)]
    np.testing.assert_array_equal(_output_pixels(output), expected)
    assert meta["raw_dimensions"] == [3, 2]
    assert meta["oriented_dimensions"] == [expected.shape[1], expected.shape[0]]
    assert meta["prepared_dimensions"] == meta["oriented_dimensions"]
    assert meta["exif_orientation_original"] == orientation
    assert meta["orientation_applied"] == orientation
    assert meta["resample"] == "NONE" and meta["lossy_resampling"] is False
    for y in range(2):
        for x in range(3):
            expected_y, expected_x = np.argwhere(np.asarray(expected_labels) == y * 3 + x + 1)[0]
            np.testing.assert_array_equal(
                np.asarray(meta["raw_to_prepared"]) @ [x, y, 1],
                [expected_x, expected_y, 1],
            )
    again, second_meta = photos.prepare_photo(output)
    np.testing.assert_array_equal(_output_pixels(again), expected)
    assert second_meta["exif_orientation_original"] is None
    assert second_meta["orientation_applied"] == 1


def test_greyscale_channels_are_replicated_without_quantisation() -> None:
    grey = np.array([[0, 1, 128], [254, 255, 70]], dtype=np.uint8)
    output, meta = photos.prepare_photo(_encode(Image.fromarray(grey)))
    pixels = _output_pixels(output)
    for channel in range(3):
        np.testing.assert_array_equal(pixels[:, :, channel], grey)
    assert meta["source_mode"] == "L" and meta["source_format"] == "PNG"


def test_ordinary_jpeg_preserves_its_decoded_samples() -> None:
    payload = _encode(Image.new("RGB", (9, 7), (10, 50, 90)), container="JPEG")
    with Image.open(BytesIO(payload)) as source:
        expected = np.asarray(source).copy()
    output, meta = photos.prepare_photo(payload)
    np.testing.assert_array_equal(_output_pixels(output), expected)
    assert meta["source_format"] == "JPEG" and meta["source_mode"] == "RGB"


@pytest.mark.parametrize("size,expected", [
    ((2048, 1025), [1024, 513]),  # 512.5 rounds up, not ties-to-even.
    ((1025, 2048), [513, 1024]),
    ((3000, 1000), [1024, 341]),
    ((1024, 1024), [1024, 1024]),
    ((17, 9), [17, 9]),
])
def test_max_side_half_up_and_no_upsampling(size: tuple[int, int], expected: list[int]) -> None:
    output, meta = photos.prepare_photo(_encode(Image.new("RGB", size, (30, 40, 50))))
    assert meta["prepared_dimensions"] == expected
    assert _output_pixels(output).shape == (expected[1], expected[0], 3)
    assert meta["resize_scale_rational"] == {"x": [expected[0], size[0]],
                                               "y": [expected[1], size[1]]}
    assert max(expected) <= 1024


def test_composed_mapping_matches_independently_written_rotation_and_pixel_centres() -> None:
    output, meta = photos.prepare_photo(_encode(Image.new("RGB", (2048, 1025)), orientation=6))
    assert meta["oriented_dimensions"] == [1025, 2048]
    assert meta["prepared_dimensions"] == [513, 1024]
    assert meta["raw_to_oriented"] == [[0, -1, 1024], [1, 0, 0], [0, 0, 1]]
    np.testing.assert_allclose(meta["oriented_to_prepared"], [
        [513 / 1025, 0, (513 / 1025 - 1) / 2],
        [0, 0.5, -0.25], [0, 0, 1],
    ])
    mapping = np.asarray(meta["raw_to_prepared"])
    assert meta["decoded_rgb_sha256"] == hashlib.sha256(
        b"513x1024:RGB:" + _output_pixels(output).tobytes()
    ).hexdigest()
    for x, y in ((0, 0), (2047, 1024), (17, 29), (1023.5, 512)):
        expected = [(1025 - y - 0.5) * 513 / 1025 - 0.5,
                    (x + 0.5) / 2 - 0.5, 1]
        np.testing.assert_allclose(mapping @ [x, y, 1], expected, atol=1e-12)


def test_small_photo_does_not_call_a_resampler(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _encode(Image.new("RGB", (10, 8)))
    monkeypatch.setattr(Image.Image, "resize", _forbid_decode)
    _, meta = photos.prepare_photo(payload)
    assert meta["resample"] == "NONE" and not meta["lossy_resampling"]


def test_downsampling_uses_lanczos_and_declares_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    original_resize = Image.Image.resize
    seen: list[object] = []

    def observe(self: Image.Image, size: tuple[int, int], *, resample: object) -> Image.Image:
        seen.append(resample)
        return original_resize(self, size, resample=resample)

    monkeypatch.setattr(Image.Image, "resize", observe)
    _, meta = photos.prepare_photo(_encode(Image.new("RGB", (1536, 1024))))
    assert seen == [Image.Resampling.LANCZOS]
    assert meta["resample"] == "LANCZOS" and meta["lossy_resampling"]


def test_prepared_pixel_digest_ignores_source_compression_and_matches_documented_prefix() -> None:
    source = Image.new("RGB", (8, 6), (10, 20, 30))
    first = _encode(source, compress_level=0)
    second = _encode(source, compress_level=9)
    assert first != second
    output, left = photos.prepare_photo(first)
    _, right = photos.prepare_photo(second)
    assert left["decoded_rgb_sha256"] == right["decoded_rgb_sha256"]
    assert left["decoded_rgb_sha256"] == hashlib.sha256(
        b"8x6:RGB:" + bytes([10, 20, 30]) * 48
    ).hexdigest()
    assert left["output_sha256"] == hashlib.sha256(output).hexdigest()


def test_prepared_pixel_digest_includes_dimensions() -> None:
    _, left = photos.prepare_photo(_encode(Image.new("RGB", (8, 6))))
    _, right = photos.prepare_photo(_encode(Image.new("RGB", (6, 8))))
    assert left["decoded_rgb_sha256"] != right["decoded_rgb_sha256"]


@pytest.mark.parametrize("encoding", ["rgba", "trns", "rgb16", "grey16", "grey2"])
def test_invalid_source_encoding_is_not_coerced(encoding: str) -> None:
    if encoding == "rgba":
        payload = _encode(Image.new("RGBA", (2, 2), (0, 0, 0, 255)))
    elif encoding == "trns":
        payload = _encode(Image.new("RGB", (2, 2)), transparency=(0, 0, 0))
    elif encoding == "rgb16":
        payload = _raw_png(2, 2, depth=16, color_type=2,
                           row=struct.pack(">HHH", 65535, 256, 1) * 2)
    elif encoding == "grey16":
        payload = _raw_png(2, 2, depth=16, color_type=0, row=struct.pack(">HH", 0, 65535))
    else:
        payload = _raw_png(2, 2, depth=2, color_type=0, row=b"\x30")
    with pytest.raises(FrameIssue) as caught:
        photos.prepare_photo(payload)
    assert caught.value.issue.code == "UNSUPPORTED_IMAGE"


def test_multiframe_png_is_rejected() -> None:
    payload = _encode(Image.new("RGB", (3, 2), (0, 0, 0)), save_all=True,
                      append_images=[Image.new("RGB", (3, 2), (255, 255, 255))])
    with Image.open(BytesIO(payload)) as image:
        assert image.n_frames == 2
    with pytest.raises(FrameIssue) as caught:
        photos.prepare_photo(payload)
    assert caught.value.issue.code == "UNSUPPORTED_IMAGE"


def test_other_container_is_rejected_before_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _encode(Image.new("RGB", (3, 2)), container="BMP")
    monkeypatch.setattr(photos, "decode_image", _forbid_decode)
    with pytest.raises(FrameIssue, match="UNSUPPORTED_IMAGE"):
        photos.prepare_photo(payload)


def test_invalid_exif_is_not_silently_defaulted() -> None:
    with pytest.raises(FrameIssue, match="INVALID_ORIENTATION"):
        photos.prepare_photo(_encode(Image.new("RGB", (3, 2)), orientation=9))


@pytest.mark.parametrize("payload", [b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_invalid_payload_is_a_validation_error(payload: bytes) -> None:
    with pytest.raises((ValueError, FrameIssue)):
        photos.prepare_photo(payload)


@pytest.mark.parametrize("size", [(1, 9), (9, 1)])
def test_raw_side_below_two_is_rejected_before_decode(
    size: tuple[int, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _encode(Image.new("RGB", size))
    monkeypatch.setattr(photos, "decode_image", _forbid_decode)
    with pytest.raises(ValueError, match="at least 2"):
        photos.prepare_photo(payload)


def test_oversized_header_is_rejected_before_pixel_load(monkeypatch: pytest.MonkeyPatch) -> None:
    # The header is valid but intentionally has no valid image data.  A decoder
    # must never reach that data or allocate the advertised pixel array.
    payload = _raw_png(6001, 4000)
    monkeypatch.setattr(photos, "decode_image", _forbid_decode)
    monkeypatch.setattr(Image.Image, "load", _forbid_decode)
    with pytest.raises(ValueError, match="24000000 pixels"):
        photos.prepare_photo(payload)


def test_encoded_byte_limit_is_checked_before_header_or_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(photos, "decode_image", _forbid_decode)
    monkeypatch.setattr(Image, "open", _forbid_decode)
    with pytest.raises(ValueError, match="67108864 bytes"):
        photos.prepare_photo(b"x" * (64 * 1024 * 1024 + 1))


@pytest.mark.parametrize("pillow_limit", [20, 10])
def test_pillow_decompression_warning_and_error_are_validation_errors(
    pillow_limit: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _encode(Image.new("RGB", (6, 6)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
    monkeypatch.setattr(photos, "decode_image", _forbid_decode)
    with pytest.raises(ValueError, match="decompression safety limit"):
        photos.prepare_photo(payload)


def test_extreme_aspect_ratio_does_not_emit_a_one_pixel_side() -> None:
    with pytest.raises(ValueError, match="output side below 2"):
        photos.prepare_photo(_encode(Image.new("RGB", (3000, 2))))


def test_non_bytes_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be bytes"):
        photos.prepare_photo(bytearray(b"bad"))
