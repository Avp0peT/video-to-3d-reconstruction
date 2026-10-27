"""Bounded original-photo preparation for ``object-input-v1``.

EXIF is a lossless reindexing and is applied exactly once.  Downsampling, when
needed, uses Pillow LANCZOS and is explicitly reported as lossy.  The matrices
map continuous pixel-centre coordinates (integer coordinates locate centres):
``x_out = (x_in + 0.5) * out_width / in_width - 0.5``.  They describe geometry,
not nearest-pixel lookup, the LANCZOS sample weights, or camera calibration.

This adapter deliberately delegates source encoding checks to ``photo-io-v1``;
it must never make a rejected source acceptable by coercing mode or bit depth.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Final, Literal, TypedDict
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

from kozlov_lab.io.errors import FrameIssue
from kozlov_lab.io.images import decode_image, encode_rgb_png
from kozlov_lab.io.orientation import apply_orientation, orientation_matrix

PROFILE_ID: Final[str] = "object-input-v1"
MAX_INPUT_BYTES: Final[int] = 64 * 1024 * 1024
MAX_INPUT_PIXELS: Final[int] = 24_000_000
MAX_PREPARED_SIDE: Final[int] = 1024


class ResizeScaleRational(TypedDict):
    """Exact, unreduced ``[output_dimension, input_dimension]`` scale ratios."""

    x: list[int]
    y: list[int]


class PhotoMetadata(TypedDict):
    source_format: str
    source_mode: str
    raw_dimensions: list[int]
    oriented_dimensions: list[int]
    prepared_dimensions: list[int]
    exif_orientation_original: int | None
    orientation_applied: int
    raw_to_oriented: list[list[int]]
    oriented_to_prepared: list[list[float]]
    raw_to_prepared: list[list[float]]
    resize_scale_rational: ResizeScaleRational
    resample: Literal["LANCZOS", "NONE"]
    decoded_rgb_sha256: str
    output_sha256: str
    lossy_resampling: bool


def _check_header(payload: bytes) -> tuple[int, int]:
    """Inspect lazy Pillow header dimensions before any full pixel decode."""
    with Image.open(BytesIO(payload)) as header:
        if header.format not in ("JPEG", "PNG"):
            raise FrameIssue("UNSUPPORTED_IMAGE", "only JPEG and PNG photos are accepted")
        width, height = header.size
        if min(width, height) < 2:
            raise ValueError("photo dimensions must each be at least 2 pixels")
        if width * height > MAX_INPUT_PIXELS:
            raise ValueError(f"photo header exceeds {MAX_INPUT_PIXELS} pixels")
        return width, height


def _prepared_size(width: int, height: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= MAX_PREPARED_SIDE:
        return width, height
    # Integer half-up rounding avoids platform-dependent floating-point ties.
    sides = [
        max(1, (2 * side * MAX_PREPARED_SIDE + longest) // (2 * longest))
        for side in (width, height)
    ]
    return sides[0], sides[1]


def prepare_photo(payload: bytes) -> tuple[bytes, PhotoMetadata]:
    """Return an EXIF-free RGB8 PNG and JSON-safe provenance metadata.

    Accept only single-frame, genuinely 8-bit RGB/L JPEG or PNG, at most 64 MiB
    encoded and 24 million raw pixels.  The existing strict decoder replicates
    L channels without quantisation.  Preserve aspect ratio with integer
    half-up dimensions, never upscale, and cap the longest output side at
    1024.  Rounding may make the two exact recorded scale ratios differ.

    ``orientation_applied`` is the effective EXIF integer (1 when absent).
    ``decoded_rgb_sha256`` identifies the *prepared* pixels, using ASCII
    ``"{width}x{height}:RGB:"`` followed by row-major RGB bytes, matching the
    existing video entry's identity convention.  ``output_sha256`` hashes the
    returned PNG bytes.  Validation failures raise ValueError or FrameIssue.
    """
    if not isinstance(payload, bytes):
        raise ValueError("photo payload must be bytes")
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError(f"encoded photo exceeds {MAX_INPUT_BYTES} bytes")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            header_size = _check_header(payload)
            decoded = decode_image(payload)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("photo exceeds Pillow decompression safety limit") from exc
    except FrameIssue:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, EOFError) as exc:
        raise FrameIssue("DECODE_ERROR", f"cannot decode photo: {exc}") from exc

    raw_width, raw_height = decoded.raw_width, decoded.raw_height
    if (raw_width, raw_height) != header_size:
        raise ValueError("decoded dimensions disagree with the inspected header")
    orientation = decoded.exif_orientation_original
    applied = 1 if orientation is None else orientation
    oriented = apply_orientation(decoded.rgb, applied)
    oriented_height, oriented_width = oriented.shape[:2]
    prepared_width, prepared_height = _prepared_size(oriented_width, oriented_height)
    if min(prepared_width, prepared_height) < 2:
        raise ValueError("photo aspect ratio would produce an output side below 2 pixels")

    resized = (prepared_width, prepared_height) != (oriented_width, oriented_height)
    if resized:
        image = Image.fromarray(np.ascontiguousarray(oriented))
        prepared = np.asarray(
            image.resize((prepared_width, prepared_height), resample=Image.Resampling.LANCZOS)
        )
    else:
        prepared = oriented

    output = encode_rgb_png(prepared)
    raw_to_oriented = orientation_matrix(applied, raw_width, raw_height)
    sx, sy = prepared_width / oriented_width, prepared_height / oriented_height
    oriented_to_prepared = [
        [sx, 0.0, (sx - 1.0) / 2.0],
        [0.0, sy, (sy - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ]
    raw_to_prepared = (
        np.asarray(oriented_to_prepared) @ np.asarray(raw_to_oriented)
    ).tolist()
    pixel_digest = hashlib.sha256(f"{prepared_width}x{prepared_height}:RGB:".encode("ascii"))
    pixel_digest.update(np.ascontiguousarray(prepared).tobytes())
    metadata: PhotoMetadata = {
        "source_format": decoded.source_format,
        "source_mode": decoded.source_mode,
        "raw_dimensions": [raw_width, raw_height],
        "oriented_dimensions": [oriented_width, oriented_height],
        "prepared_dimensions": [prepared_width, prepared_height],
        "exif_orientation_original": orientation,
        "orientation_applied": applied,
        "raw_to_oriented": raw_to_oriented,
        "oriented_to_prepared": oriented_to_prepared,
        "raw_to_prepared": raw_to_prepared,
        "resize_scale_rational": {
            "x": [prepared_width, oriented_width],
            "y": [prepared_height, oriented_height],
        },
        "resample": "LANCZOS" if resized else "NONE",
        "decoded_rgb_sha256": pixel_digest.hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "lossy_resampling": resized,
    }
    return output, metadata
