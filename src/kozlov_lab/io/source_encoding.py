"""Source bit depth, read from the encoded container (``photo-io-v1`` §3).

Why the decoded mode is not evidence
------------------------------------
``photo-io-v1`` requires *"源每通道位深为 8"* — the **source** file, not the
decoded array.  Those differ, and the difference is silent:

* a valid **16-bit RGB PNG** opens in Pillow as mode ``RGB``, and the samples are
  reduced to the high byte — ``(65535, 256, 1)`` becomes ``(255, 1, 0)``;
* a **2-bit or 4-bit greyscale PNG** opens as mode ``L``, with the low-depth
  samples scaled up to 0-255.

In both cases the resulting array is a perfectly well-formed ``uint8 RGB`` / ``L``
array with the *right shape and dtype*, so every check performed on the decoded
image passes.  Only the pixel values are wrong, and nothing marks them as wrong.
That is the failure this module exists to prevent: a check that reads the depth
from ``mode``/``dtype`` is not a check.

So the depth is read from the container itself:

* **PNG** — the ``IHDR`` chunk, which is required by the spec to be the first
  chunk and carries bit depth and colour type at fixed offsets;
* **JPEG** — the ``SOF`` marker (``SOF0``-``SOF15`` excluding ``DHT``/``JPG``/
  ``DAC``), whose frame header begins with a one-byte sample precision.

Both readers are pure byte inspection: no decoding, no re-encoding, no Pillow.
That keeps the judgement independent of the library that performs the decode, and
the tests exercise the readers directly against bytes whose depth is known by
construction.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .errors import FrameIssue

__all__ = [
    "BITS_PER_CHANNEL_REQUIRED",
    "JpegFrame",
    "PNG_SIGNATURE",
    "PngHeader",
    "SourceEncodingError",
    "assert_eight_bit_source",
    "jpeg_frame",
    "png_header",
    "source_bit_depth",
]

#: The only per-sample depth ``photo-io-v1`` accepts.
BITS_PER_CHANNEL_REQUIRED = 8

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: ``SOFn`` markers carry a frame header; these three share the range but are not
#: frame headers (``DHT``, reserved ``JPG``, ``DAC``).
_NOT_FRAME_HEADERS = frozenset({0xC4, 0xC8, 0xCC})

#: Markers with no length field of their own.
_STANDALONE = frozenset({0x01}) | frozenset(range(0xD0, 0xDA))


class SourceEncodingError(ValueError):
    """The encoded container does not expose a readable bit depth."""


@dataclass(frozen=True)
class PngHeader:
    """The fields of ``IHDR`` that decide whether a PNG is acceptable."""

    width: int
    height: int
    bit_depth: int
    colour_type: int


@dataclass(frozen=True)
class JpegFrame:
    """The sample precision declared by a JPEG frame header."""

    precision: int
    marker: int
    height: int
    width: int
    components: int


def png_header(payload: bytes) -> PngHeader:
    """Parse ``IHDR`` from encoded PNG bytes.

    The PNG specification requires ``IHDR`` to be the first chunk, so its fields
    sit at fixed offsets: 8 signature + 4 length + 4 type, then width, height,
    bit depth, colour type.
    """
    if not payload.startswith(PNG_SIGNATURE):
        raise SourceEncodingError("missing PNG signature")
    if len(payload) < 33:
        raise SourceEncodingError("truncated PNG, no complete IHDR chunk")
    if payload[12:16] != b"IHDR":
        raise SourceEncodingError(f"first chunk is {payload[12:16]!r}, not IHDR")
    length = struct.unpack(">I", payload[8:12])[0]
    if length != 13:
        raise SourceEncodingError(f"IHDR declares length {length}, expected 13")

    width, height, bit_depth, colour_type = struct.unpack(">IIBB", payload[16:26])
    return PngHeader(
        width=width,
        height=height,
        bit_depth=bit_depth,
        colour_type=colour_type,
    )


def jpeg_frame(payload: bytes) -> JpegFrame:
    """Parse the first ``SOFn`` frame header from encoded JPEG bytes.

    Stops at ``SOS``: the frame header must precede the scan, so scanning past it
    would only wander into entropy-coded data.  Stand-alone markers (``TEM``,
    ``RSTn``, ``SOI``, ``EOI``) carry no length and are skipped as such.
    """
    if payload[:2] != b"\xff\xd8":
        raise SourceEncodingError("missing JPEG SOI")

    position = 2
    while position + 3 < len(payload):
        if payload[position] != 0xFF:
            raise SourceEncodingError(f"expected a marker at offset {position}")
        while position < len(payload) and payload[position] == 0xFF:
            position += 1
        if position >= len(payload):
            break
        marker = payload[position]
        position += 1

        if marker == 0xDA:  # start of scan; a frame header must already have appeared
            break
        if marker in _STANDALONE:
            continue

        if position + 2 > len(payload):
            break
        length = struct.unpack(">H", payload[position : position + 2])[0]
        if length < 2:
            raise SourceEncodingError(f"marker 0x{marker:02X} declares length {length}")

        if 0xC0 <= marker <= 0xCF and marker not in _NOT_FRAME_HEADERS:
            if position + 8 > len(payload):
                raise SourceEncodingError("truncated JPEG frame header")
            precision = payload[position + 2]
            height, width = struct.unpack(">HH", payload[position + 3 : position + 7])
            components = payload[position + 7]
            return JpegFrame(
                precision=precision,
                marker=marker,
                height=height,
                width=width,
                components=components,
            )
        position += length

    raise SourceEncodingError("no JPEG frame header found before the scan")


def source_bit_depth(payload: bytes, container: str) -> int:
    """Bits per sample as recorded in the encoded file.

    Deliberately *not* derived from a decoded mode or dtype.
    """
    if container == "PNG":
        return png_header(payload).bit_depth
    if container == "JPEG":
        return jpeg_frame(payload).precision
    raise SourceEncodingError(f"no source bit-depth rule for container {container!r}")


def assert_eight_bit_source(payload: bytes, container: str, *, code: str, subject: str) -> int:
    """Refuse a source whose per-sample depth is not exactly 8.

    Returns the observed depth so callers can record it.  ``code`` is supplied by
    the caller because the contract uses different vocabularies for photos
    (``UNSUPPORTED_IMAGE``) and masks (``INVALID_MASK``).

    A source whose depth cannot be read is reported as ``DECODE_ERROR`` rather
    than being waved through: "unreadable" must never be treated as "fine".
    """
    try:
        depth = source_bit_depth(payload, container)
    except SourceEncodingError as exc:
        raise FrameIssue(
            "DECODE_ERROR",
            f"cannot determine the {subject} source bit depth: {exc}",
        ) from exc

    if depth != BITS_PER_CHANNEL_REQUIRED:
        raise FrameIssue(
            code,
            f"{subject} source is {depth} bit per channel; only "
            f"{BITS_PER_CHANNEL_REQUIRED} is accepted, and no requantisation is performed",
        )
    return depth
