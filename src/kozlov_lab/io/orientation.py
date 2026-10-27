"""EXIF orientation: the matrix, the pixel transform, and their agreement.

Two independent expressions of the same contract clause (``photo-io-v1`` §3) are
implemented here **on purpose**:

* :func:`orientation_matrix` builds the integer 3x3 homogeneous matrix straight
  from the table in the contract;
* :func:`apply_orientation` performs the rotation/flip with NumPy slicing.

Neither is derived from the other, so :func:`matrix_matches_transform` can be
used as a cross-check: a test can assert that mapping every source pixel through
the matrix lands on the pixel the array transform placed there.  A single
implementation would have no way to catch a transposed or sign-flipped case.

All eight transforms are pure reindexings — no resampling, no interpolation, no
rounding.  That is what makes ``v1`` lossless with respect to pixel values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import FrameIssue

__all__ = [
    "VALID_ORIENTATIONS",
    "apply_orientation",
    "canonical_size",
    "matrix_matches_transform",
    "orientation_matrix",
    "read_exif_orientation",
]

#: EXIF Orientation tags the contract accepts.  Anything else is an error.
VALID_ORIENTATIONS = (1, 2, 3, 4, 5, 6, 7, 8)

#: EXIF tag id for ``Orientation``.
_ORIENTATION_TAG = 274

#: Orientations 5-8 exchange the axes, so canonical width/height swap.
_SWAPPING = (5, 6, 7, 8)


def canonical_size(orientation: int, width: int, height: int) -> tuple[int, int]:
    """Canonical ``(width, height)`` for a raw ``width`` x ``height`` image."""
    _check(orientation)
    if orientation in _SWAPPING:
        return height, width
    return width, height


def orientation_matrix(orientation: int, width: int, height: int) -> list[list[int]]:
    """Integer 3x3 ``raw_to_canonical`` matrix with ``[x', y', 1]^T = T [x, y, 1]^T``.

    Transcribed directly from the contract table; ``W``/``H`` are the **raw**
    dimensions.  Kept integral so the stored JSON stays exact.
    """
    _check(orientation)
    w, h = int(width), int(height)
    if orientation == 1:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if orientation == 2:
        return [[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]]
    if orientation == 3:
        return [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]]
    if orientation == 4:
        return [[1, 0, 0], [0, -1, h - 1], [0, 0, 1]]
    if orientation == 5:
        return [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    if orientation == 6:
        return [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]]
    if orientation == 7:
        return [[0, -1, h - 1], [-1, 0, w - 1], [0, 0, 1]]
    return [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]]


def apply_orientation(array: np.ndarray, orientation: int) -> np.ndarray:
    """Return the canonically-oriented copy of ``array`` (H, W[, C]).

    Implemented with slicing so no pixel value is ever recomputed.  For
    orientation 1 the *same* array is returned unchanged (callers must not
    mutate it in place; the row processor only reads).
    """
    _check(orientation)
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D array, got shape {array.shape}")
    if orientation == 1:
        return array
    if orientation == 2:
        return np.fliplr(array)
    if orientation == 3:
        return np.rot90(array, 2)
    if orientation == 4:
        return np.flipud(array)
    if orientation == 5:
        return np.swapaxes(array, 0, 1)
    if orientation == 6:
        return np.rot90(array, 3)
    if orientation == 7:
        return np.rot90(np.swapaxes(array, 0, 1), 2)
    return np.rot90(array, 1)


def matrix_matches_transform(orientation: int, width: int, height: int) -> bool:
    """Whether the matrix and the array transform agree on **every** pixel.

    Builds a 1-channel index image, pushes it through both routes, and compares
    the resulting placement.  Used by the tests as an independent oracle.
    """
    _check(orientation)
    rows, cols = np.indices((height, width))
    labelled = (rows * width + cols).astype(np.int64)

    via_transform = apply_orientation(labelled, orientation)

    matrix = np.asarray(orientation_matrix(orientation, width, height), dtype=np.int64)
    ys, xs = np.indices((height, width))
    ones = np.ones_like(xs)
    # Flatten in row-major order so ``k`` indexes the same pixel in both routes.
    stacked = np.stack([xs.reshape(-1), ys.reshape(-1), ones.reshape(-1)], axis=0)
    mapped = matrix @ stacked
    x_out, y_out = mapped[0], mapped[1]

    canonical_width, canonical_height = canonical_size(orientation, width, height)
    if x_out.min() < 0 or x_out.max() >= canonical_width:
        return False
    if y_out.min() < 0 or y_out.max() >= canonical_height:
        return False
    if len(np.unique(np.stack([x_out, y_out], axis=1), axis=0)) != x_out.size:
        # Two raw pixels landing on the same canonical pixel would mean the
        # assignment below silently drops one of them.
        return False

    via_matrix = np.empty((canonical_height, canonical_width), dtype=np.int64)
    via_matrix[y_out, x_out] = labelled.reshape(-1)
    return bool(np.array_equal(via_matrix, via_transform))


def read_exif_orientation(image: Any) -> int | None:
    """Read the EXIF ``Orientation`` tag, or ``None`` when it is absent.

    Only integers 1-8 are accepted; anything else raises
    ``INVALID_ORIENTATION`` rather than being coerced into a default.
    """
    try:
        exif = image.getexif()
    except Exception:  # pragma: no cover - Pillow defensive path
        return None
    if exif is None:
        return None
    raw = exif.get(_ORIENTATION_TAG)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise FrameIssue(
            "INVALID_ORIENTATION",
            f"EXIF Orientation must be an integer 1-8, got {type(raw).__name__}",
        )
    if raw not in VALID_ORIENTATIONS:
        raise FrameIssue("INVALID_ORIENTATION", f"EXIF Orientation {raw} is outside 1-8")
    return int(raw)


def _check(orientation: int) -> None:
    if orientation not in VALID_ORIENTATIONS:
        raise ValueError(f"orientation must be one of {VALID_ORIENTATIONS}, got {orientation!r}")
