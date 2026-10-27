"""PNG read/write helpers with explicit, verified semantics.

Contract requirements enforced here (docs/contracts_object_masks_v1.md):
  * masks are single-frame 8-bit ``L`` PNGs, size exactly equal to the canonical
    image, EXIF Orientation absent or 1, values only {0, 255};
  * 0 is background, 255 is foreground;
  * the probability -> binary rule actually used is recorded, never hidden
    behind a silent Pillow conversion;
  * no re-encoding of the source photo is performed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

FOREGROUND = 255
BACKGROUND = 0

#: EXIF Orientation tag
_ORIENTATION_TAG = 274


class ImageError(ValueError):
    """Raised when an image fails an explicit validation rule."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _orientation(image: Image.Image):
    try:
        exif = image.getexif()
    except Exception:  # pragma: no cover - exotic PNGs
        return None
    if not exif:
        return None
    value = exif.get(_ORIENTATION_TAG)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def read_canonical_rgb(path: str | Path) -> tuple[int, int, str]:
    """Validate a photo-io-v1 canonical image; return (width, height, mode).

    The pixel data itself is not needed by the adapter (SAM3 reads the file via
    PIL on its own), so we only assert the domain invariants.
    """
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = image.size
        orientation = _orientation(image)
    if mode != "RGB":
        raise ImageError(f"canonical image must be mode RGB, got {mode!r}")
    if orientation not in (None, 1):
        raise ImageError(f"canonical image must not carry Orientation != 1, got {orientation!r}")
    return size[0], size[1], mode


def read_mask_l(path: str | Path) -> tuple[bytes, int, int]:
    """Strictly read a binary 8-bit L PNG; return (flat bytes, width, height)."""
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        size = image.size
        orientation = _orientation(image)
        raw = image.tobytes()
    if mode != "L":
        raise ImageError(f"mask must be mode L, got {mode!r}")
    if orientation not in (None, 1):
        raise ImageError(f"mask must not carry Orientation != 1, got {orientation!r}")
    bbox = set(raw)
    if not bbox <= {BACKGROUND, FOREGROUND}:
        extras = sorted(bbox - {BACKGROUND, FOREGROUND})[:5]
        raise ImageError(f"mask values must be a subset of {{0, 255}}, found {extras!r}")
    return raw, size[0], size[1]


def write_mask_l(path: str | Path, flat: bytes, width: int, height: int) -> str:
    """Write a verified binary mask PNG and return its sha256.

    The buffer is written byte-for-byte (no thresholding, no resampling, no
    palette conversion) and then re-read to prove the file honours the contract.
    """
    if len(flat) != width * height:
        raise ImageError(f"mask buffer length {len(flat)} != {width}*{height}")
    extras = sorted(set(flat) - {BACKGROUND, FOREGROUND})
    if extras:
        raise ImageError(f"refusing to write non-binary mask, found {extras[:5]!r}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.frombytes("L", (width, height), flat)
    image.save(target, format="PNG", optimize=False, compress_level=6)
    image.close()

    back, back_w, back_h = read_mask_l(target)
    if (back_w, back_h) != (width, height):
        raise ImageError(f"round-trip size mismatch: wrote {(width, height)} read {(back_w, back_h)}")
    if back != flat:
        raise ImageError("round-trip pixel mismatch")
    # the recorded digest is the digest of the FILE, never of the decoded buffer
    return sha256_file(target)


def mask_from_boolean(truth) -> bytes:
    """Convert a 2-D boolean/0-1 array (or nested lists) into flat 0/255 bytes."""
    height = len(truth)
    if height == 0:
        return b""
    width = len(truth[0])
    out = bytearray(height * width)
    for y, row in enumerate(truth):
        if len(row) != width:
            raise ImageError("ragged truth array")
        base = y * width
        for x, value in enumerate(row):
            out[base + x] = FOREGROUND if value else BACKGROUND
    return bytes(out)


def foreground_pixels(flat: bytes) -> int:
    return flat.count(FOREGROUND)


def write_overlay(source_path: str | Path, flat: bytes, width: int, height: int, path: str | Path) -> str:
    """Write a red-tinted overlay of a mask over the canonical photo.

    Visual aid only -- never an authoritative mask (contract §mask and candidates).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as base:
        base = base.convert("RGB")
        if base.size != (width, height):
            raise ImageError("overlay source size mismatch")
        pixels = bytearray(base.tobytes())
    for index, value in enumerate(flat):
        if value:
            offset = index * 3
            pixels[offset] = min(255, int(pixels[offset] * 0.35) + 165)
            pixels[offset + 1] = int(pixels[offset + 1] * 0.35)
            pixels[offset + 2] = int(pixels[offset + 2] * 0.35)
    overlay = Image.frombytes("RGB", (width, height), bytes(pixels))
    overlay.save(target, format="PNG")
    overlay.close()
    return sha256_file(target)
