"""Manual binary masks: decode, validate, and re-encode (``photo-io-v1`` §4).

Why this module refuses so much
-------------------------------
A mask is *ground truth for the foreground boundary*.  If a reader silently
thresholds a greyscale image, or accepts a stray ``128``, or accepts ``1`` and
``255`` in the same file, the boundary it reports is no longer the boundary a
human drew — but it still looks like a perfectly good boolean array.  The
resulting numbers would be wrong without ever being obviously broken, so every
ambiguity here is a hard error instead of a guess.

The contract therefore fixes the entire accepted domain:

* single-frame PNG, mode ``L``, 8 bit, no transparency;
* **raw** width/height exactly equal to the original photo's raw size — a mask
  is drawn in raw pixel coordinates and is *not* pre-rotated by EXIF;
* the mask's own ``Orientation`` tag must be absent or ``1``; anything else is
  rejected so the same rotation can never be applied twice;
* pixel values drawn only from ``{0, 1}`` or ``{0, 255}``, never a mixture, with
  no thresholding and no quantisation.

The canonical form is written as an ``L`` PNG whose values are exactly 0 or 255.
``apply_orientation`` from :mod:`kozlov_lab.io.orientation` is reused — the mask
must undergo the *same* transform as the photo, so it deliberately does not have
its own copy of the eight-orientation table.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..io.errors import FrameIssue
from ..io.orientation import apply_orientation, read_exif_orientation
from ..io.source_encoding import assert_eight_bit_source

__all__ = [
    "MASK_ENCODINGS",
    "DecodedMask",
    "decode_manual_mask",
    "encode_mask_png",
    "expected_canonical_mask_size",
]

#: ``mask_encoding`` labels the contract defines, plus the all-zero special case.
MASK_ENCODINGS = ("zero_one", "zero_255", "all_zero")

#: Foreground value written into canonical output PNGs.
_FOREGROUND_OUT = 255

_ZERO_ONE = frozenset({0, 1})
_ZERO_255 = frozenset({0, 255})


@dataclass(frozen=True)
class DecodedMask:
    """A validated manual mask in **raw** orientation."""

    foreground: np.ndarray  # bool, shape (raw_height, raw_width)
    raw_width: int
    raw_height: int
    encoding: str
    foreground_pixels: int
    exif_orientation_original: int | None

    @property
    def is_empty(self) -> bool:
        """Whether the mask contains no foreground pixel at all."""
        return self.foreground_pixels == 0


def expected_canonical_mask_size(
    orientation: int, raw_width: int, raw_height: int
) -> tuple[int, int]:
    """Canonical ``(width, height)`` a mask must have after the shared transform.

    Provided so a caller can recompute the expectation instead of copying it.
    """
    from ..io.orientation import canonical_size

    return canonical_size(orientation, raw_width, raw_height)


def decode_manual_mask(payload: bytes) -> DecodedMask:
    """Decode and validate a manual mask.

    Raises :class:`~kozlov_lab.io.errors.FrameIssue` with ``DECODE_ERROR``,
    ``INVALID_MASK`` or ``INVALID_ORIENTATION``.  The size check against the
    photo is **not** performed here: this function has no photo to compare with,
    and the caller reports ``MASK_SIZE_MISMATCH`` in the row record.
    """
    try:
        image = Image.open(_as_stream(payload))
        image.load()
    except FrameIssue:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise FrameIssue("DECODE_ERROR", f"cannot decode mask: {exc}") from exc

    if (image.format or "").upper() != "PNG":
        raise FrameIssue("INVALID_MASK", f"mask must be a PNG, got {image.format or 'unknown'}")

    if int(getattr(image, "n_frames", 1)) != 1:
        raise FrameIssue("INVALID_MASK", "mask must be a single-frame image")

    if image.mode != "L":
        raise FrameIssue(
            "INVALID_MASK",
            f"mask must be 8-bit mode L, got mode {image.mode!r} "
            "(no bit-depth quantisation or alpha handling is performed)",
        )

    if "transparency" in image.info:
        raise FrameIssue("INVALID_MASK", "mask must not carry a transparency channel")

    # A 2- or 4-bit greyscale PNG decodes to mode ``L``, so the decoded mode
    # cannot show that the mask's depth is not 8.  Read it from the IHDR instead.
    assert_eight_bit_source(payload, "PNG", code="INVALID_MASK", subject="mask")

    orientation = read_exif_orientation(image)
    if orientation not in (None, 1):
        raise FrameIssue(
            "INVALID_ORIENTATION",
            f"mask Orientation must be absent or 1 (drawn in raw pixels), got {orientation}",
        )

    raw_width, raw_height = int(image.width), int(image.height)
    values = np.asarray(image, dtype=np.uint8)

    encoding, foreground = _classify_values(values)

    foreground = np.ascontiguousarray(foreground)
    return DecodedMask(
        foreground=foreground,
        raw_width=raw_width,
        raw_height=raw_height,
        encoding=encoding,
        foreground_pixels=int(np.count_nonzero(foreground)),
        exif_orientation_original=orientation,
    )


def encode_mask_png(foreground: np.ndarray) -> bytes:
    """Serialise a boolean foreground array as a canonical ``L`` PNG (0/255).

    The written file carries no EXIF, so re-reading it cannot trigger a second
    rotation.  ``Image.fromarray`` infers mode ``L`` from a 2-D ``uint8`` array;
    the ``mode=`` argument is deprecated in Pillow 12, so the result is asserted
    against the contract's requirement rather than requested.
    """
    array = np.asarray(foreground)
    if array.dtype != np.bool_:
        raise ValueError(f"expected a boolean array, got dtype {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"expected shape (H, W), got {array.shape}")

    canonical = np.where(array, _FOREGROUND_OUT, 0).astype(np.uint8)
    image = Image.fromarray(canonical)
    if image.mode != "L":
        raise ValueError(f"expected the encoded mask to be mode L, got {image.mode}")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def canonical_mask_array(mask: DecodedMask, orientation: int) -> np.ndarray:
    """Apply the photo's orientation transform to a validated mask."""
    return apply_orientation(mask.foreground, orientation)


def _classify_values(values: np.ndarray) -> tuple[str, np.ndarray]:
    """Return ``(encoding, foreground_bool)`` or raise for any ambiguous domain.

    The all-zero mask is checked first because it is a legal member of *both*
    encodings; the contract names it ``all_zero`` so that the record does not
    silently claim one encoding over the other.

    The mixture branch is ordered before the general domain check on purpose.
    Both orders reject ``{0, 1, 255}``, but only this one produces a diagnosis
    that names the real problem: the general branch would compute
    ``present - {0, 1, 255} == {}`` and report "unexpected value(s): []", sending
    the reader after a stray pixel that does not exist.
    """
    present = frozenset(int(value) for value in np.unique(values))

    if present == {0}:
        return "all_zero", np.zeros(values.shape, dtype=bool)

    if 1 in present and 255 in present:
        raise FrameIssue(
            "INVALID_MASK",
            "mask mixes foreground values 1 and 255; exactly one encoding is required",
        )

    unexpected = sorted(present - {0, 1, 255})
    if unexpected:
        raise FrameIssue(
            "INVALID_MASK",
            "mask pixel values must be drawn from {0,1} or {0,255}; "
            f"unexpected value(s): {unexpected}",
        )

    # Only subsets of one encoding can reach here, because the mixture and the
    # out-of-domain cases were rejected above.
    if present <= _ZERO_ONE:
        return "zero_one", values != 0
    if present <= _ZERO_255:
        return "zero_255", values != 0
    raise FrameIssue(
        "INVALID_MASK",
        f"mask value domain could not be classified: {sorted(present)}",
    )


def _as_stream(payload: bytes) -> Any:
    stream = BytesIO(payload)
    stream.seek(0)
    return stream
