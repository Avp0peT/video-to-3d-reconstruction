"""Photo decoding and validation (``photo-io-v1`` §3).

The whole point of this module is to **refuse** rather than to coerce.  A silent
``astype(uint8)`` on a 16-bit PNG, or quietly dropping an alpha channel, would
produce a plausible-looking ``uint8 RGB`` array whose pixel values no longer
mean what the caller thinks.  The contract therefore lists the rejected shapes
explicitly and this module implements that list literally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import FrameIssue
from .orientation import read_exif_orientation
from .source_encoding import assert_eight_bit_source

__all__ = ["DecodedImage", "as_uint8_rgb", "decode_image", "encode_rgb_png", "SUPPORTED_FORMATS"]

#: Container formats accepted by ``photo-io-v1``.
SUPPORTED_FORMATS = ("JPEG", "PNG")

#: Decoded modes accepted; ``L`` is expanded to RGB, everything else is refused.
_ACCEPTED_MODES = ("RGB", "L")


@dataclass(frozen=True)
class DecodedImage:
    """A validated photo: canonical-ready RGB pixels plus raw geometry."""

    rgb: np.ndarray
    raw_width: int
    raw_height: int
    exif_orientation_original: int | None
    source_mode: str
    source_format: str


def as_uint8_rgb(array: Any) -> np.ndarray:
    """Validate that ``array`` is a ``uint8`` ``(H, W, 3)`` array, and return it.

    Deliberately does **no** conversion: an ``int16`` or out-of-range input is an
    error, not something to be clipped or cast.  This is the boundary the
    contract calls out as "API 输入不满足范围/类型时显式报错".
    """
    candidate = np.asarray(array)
    if candidate.dtype != np.uint8:
        raise ValueError(f"expected dtype uint8, got {candidate.dtype}")
    if candidate.ndim != 3 or candidate.shape[2] != 3:
        raise ValueError(f"expected shape (H, W, 3), got {candidate.shape}")
    return candidate


def decode_image(payload: bytes) -> DecodedImage:
    """Decode ``payload`` into a validated ``uint8 RGB`` array.

    Raises :class:`~kozlov_lab.io.errors.FrameIssue` with ``DECODE_ERROR``,
    ``UNSUPPORTED_IMAGE`` or ``INVALID_ORIENTATION``.
    """
    try:
        image = Image.open(_as_stream(payload))
        image.load()
    except FrameIssue:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise FrameIssue("DECODE_ERROR", f"cannot decode image: {exc}") from exc

    source_format = (image.format or "").upper()
    source_mode = image.mode

    if source_format not in SUPPORTED_FORMATS:
        raise FrameIssue(
            "UNSUPPORTED_IMAGE",
            f"format {source_format or 'unknown'} is not one of {SUPPORTED_FORMATS}",
        )

    frame_count = int(getattr(image, "n_frames", 1))
    if frame_count != 1:
        raise FrameIssue("UNSUPPORTED_IMAGE", f"multi-frame input has {frame_count} frames")

    if "transparency" in image.info:
        raise FrameIssue("UNSUPPORTED_IMAGE", "transparency (including tRNS) is not accepted")

    if source_mode not in _ACCEPTED_MODES:
        raise FrameIssue(
            "UNSUPPORTED_IMAGE",
            f"mode {source_mode!r} is not one of {_ACCEPTED_MODES}; "
            "no silent alpha drop or bit-depth quantisation is performed",
        )

    # Checked against the *encoded container*, not the decoded mode: Pillow
    # reports a valid 16-bit RGB PNG as mode ``RGB`` and a 2/4-bit greyscale PNG
    # as mode ``L``, so the mode cannot answer what the source depth was.
    assert_eight_bit_source(payload, source_format, code="UNSUPPORTED_IMAGE", subject="image")

    orientation = read_exif_orientation(image)
    width, height = int(image.width), int(image.height)

    if source_mode == "L":
        # Lossless expansion: replicate the single channel three times.
        grey = np.asarray(image, dtype=np.uint8)
        rgb = np.repeat(grey[:, :, None], 3, axis=2)
    else:
        rgb = np.asarray(image, dtype=np.uint8)

    return DecodedImage(
        rgb=as_uint8_rgb(rgb),
        raw_width=width,
        raw_height=height,
        exif_orientation_original=orientation,
        source_mode=source_mode,
        source_format=source_format,
    )


def encode_rgb_png(array: np.ndarray) -> bytes:
    """Serialise a validated ``uint8 RGB`` array as PNG bytes for writing.

    ``ascontiguousarray`` is required, not cosmetic: the orientation transforms
    are pure views (``fliplr`` has negative strides, ``swapaxes`` is
    non-contiguous), and only a C-contiguous buffer can be handed to Pillow
    without reinterpreting the strides.

    ``Image.fromarray`` infers mode ``RGB`` from a ``(H, W, 3)`` ``uint8`` array;
    the ``mode=`` argument is deprecated in Pillow 12, so the result is asserted
    instead of requested.
    """
    from io import BytesIO

    validated = as_uint8_rgb(array)
    image = Image.fromarray(np.ascontiguousarray(validated))
    if image.mode != "RGB":
        raise ValueError(f"expected the encoded photo to be mode RGB, got {image.mode}")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _as_stream(payload: bytes):
    from io import BytesIO

    stream = BytesIO(payload)
    stream.seek(0)
    return stream
