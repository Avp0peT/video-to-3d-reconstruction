"""Bounded, read-only parser for the OpenMVS v2.4.0 little-endian DR format.

The layout follows ``HeaderDepthDataRaw`` in ``libs/MVS/Interface.h`` and
``ExportDepthDataRaw`` in ``libs/MVS/DepthMap.cpp`` from the official v2.4.0
source. This independent implementation supports the x86 little-endian layout,
not arbitrary native C++ ABIs. It does not load images or execute OpenMVS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import stat
import struct

import numpy as np


HAS_DEPTH = 1
HAS_NORMAL = 2
HAS_CONF = 4
HAS_VIEWS = 8
KNOWN_FLAGS = HAS_DEPTH | HAS_NORMAL | HAS_CONF | HAS_VIEWS
NO_ID = 0xFFFFFFFF
MAX_PIXELS = 16_777_216
MAX_BYTES = 536_870_912

_HEADER = struct.Struct("<2sBBIIIIff")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_POSE = struct.Struct("<21d")


class DMapError(ValueError):
    """The input is inaccessible, exceeds the bounds, or is not a valid DR map."""


@dataclass(frozen=True, eq=False)
class DepthMap:
    """A parsed snapshot with immutable, row-major NumPy array storage.

    Sizes are ``(width, height)``; map arrays are indexed ``[row, column]``.
    ``K`` and ``R`` are (3, 3), ``C`` is (3,), and optional normal/views maps
    add trailing dimensions of 3/4. Camera matrices are kept at image
    resolution, exactly as stored. No camera, pose, or visibility semantics
    are inferred here. Identity equality avoids ambiguous NumPy array equality.
    """

    path: str
    sha256: str
    file_bytes: int
    flags: int
    reserved: int
    image_size: tuple[int, int]
    depth_size: tuple[int, int]
    d_min: float
    d_max: float
    image_name_bytes: bytes
    view_ids: tuple[int, ...]
    K: np.ndarray = field(repr=False)
    R: np.ndarray = field(repr=False)
    C: np.ndarray = field(repr=False)
    depth: np.ndarray = field(repr=False)
    normal: np.ndarray | None = field(repr=False)
    confidence: np.ndarray | None = field(repr=False)
    views: np.ndarray | None = field(repr=False)

    @property
    def image_name_utf8(self) -> str:
        """Decode explicitly as UTF-8, raising UnicodeDecodeError if invalid.

        OpenMVS stores filename bytes without an encoding declaration. Parsing
        therefore does not require UTF-8 and never opens this referenced name.
        """
        return self.image_name_bytes.decode("utf-8", errors="strict")


def _check_limit(name: str, value: int, ceiling: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DMapError(f"{name} must be an integer, excluding bool")
    if not 1 <= value <= ceiling:
        raise DMapError(f"{name} must be between 1 and {ceiling}")


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _identity_signature(info: os.stat_result) -> tuple[int, int, int, int]:
    # Windows/Python can expose birth time through path.stat().st_ctime and
    # change time through fstat().st_ctime. Compare ctime only within the same
    # query channel; do not mistake that API difference for a changed file.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read_snapshot(path: str | os.PathLike[str], max_bytes: int) -> tuple[str, bytes]:
    """Read once, rejecting special files and observable concurrent changes."""
    try:
        source = Path(path).expanduser().resolve(strict=True)
        initial = source.stat()
        if not stat.S_ISREG(initial.st_mode):
            raise DMapError("dmap input must be a regular file")
        if initial.st_size > max_bytes:
            raise DMapError(f"dmap file exceeds max_bytes ({max_bytes})")
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise DMapError("dmap input must be a regular file")
            if _identity_signature(initial) != _identity_signature(before):
                raise DMapError("dmap file changed before reading")
            # This is the sole read, bounded by the checked file size. The
            # post-read descriptor/path stats detect observable growth,
            # truncation, replacement, and in-place modification.
            data = stream.read(before.st_size)
            after = os.fstat(stream.fileno())
        final = source.stat()
        if (
            len(data) != before.st_size
            or _stat_signature(before) != _stat_signature(after)
            or _stat_signature(initial) != _stat_signature(final)
            or _identity_signature(before) != _identity_signature(final)
        ):
            raise DMapError("dmap file changed or was truncated while reading")
    except DMapError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise DMapError(f"cannot read dmap input: {exc}") from exc
    return str(source), data


def _require_bytes(data: bytes, offset: int, count: int, label: str) -> None:
    if offset + count > len(data):
        raise DMapError(f"truncated dmap {label}")


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise DMapError(f"{name} contains non-finite values")


def read_dmap(
    path: str | os.PathLike[str],
    *,
    max_pixels: int = MAX_PIXELS,
    max_bytes: int = MAX_BYTES,
) -> DepthMap:
    """Read and validate one OpenMVS DR snapshot without writing any files.

    Limits may be lowered but not raised beyond 16,777,216 pixels per image
    or depth map and 512 MiB per file. All lengths and the exact expected
    end-of-file are checked before constructing NumPy arrays. Returned arrays
    share immutable bytes storage, so setting their writable flag is rejected.

    Depth is required; zero depth is preserved as an invalid-pixel marker.
    Stored depths need not lie exactly within the header range. Confidence is
    nonnegative without an upper bound. The reserved byte is preserved as
    opaque metadata. Finite camera values are required; camera geometry,
    normal normalization, and view-index interpretation belong to a verifier.
    File-change checks detect metadata-visible changes, not adversarial edits
    that restore all checked metadata.
    """
    _check_limit("max_pixels", max_pixels, MAX_PIXELS)
    _check_limit("max_bytes", max_bytes, MAX_BYTES)
    absolute_path, data = _read_snapshot(path, max_bytes)
    _require_bytes(data, 0, _HEADER.size, "header")
    (
        magic,
        flags,
        reserved,
        image_w,
        image_h,
        depth_w,
        depth_h,
        d_min,
        d_max,
    ) = _HEADER.unpack_from(data)
    if magic != b"DR":
        raise DMapError("invalid dmap magic; expected little-endian DR")
    if flags & ~KNOWN_FLAGS or not flags & HAS_DEPTH:
        raise DMapError("invalid dmap flags; known flags and depth are required")
    if not image_w or not image_h or not depth_w or not depth_h:
        raise DMapError("image and depth dimensions must be positive")
    if depth_w > image_w or depth_h > image_h:
        raise DMapError("depth dimensions must not exceed image dimensions")
    pixels = depth_w * depth_h
    if pixels > max_pixels or image_w * image_h > max_pixels:
        raise DMapError(f"image or depth dimensions exceed max_pixels ({max_pixels})")
    if not math.isfinite(d_min) or not math.isfinite(d_max):
        raise DMapError("depth range must be finite")
    if d_min <= 0 or d_max < d_min:
        raise DMapError("depth range must satisfy 0 < d_min <= d_max")

    offset = _HEADER.size
    _require_bytes(data, offset, _U16.size, "image-name length")
    name_size = _U16.unpack_from(data, offset)[0]
    offset += _U16.size
    if name_size == 0:
        raise DMapError("image name must not be empty")
    _require_bytes(data, offset, name_size, "image name")
    image_name = data[offset:offset + name_size]
    if b"\0" in image_name:
        raise DMapError("image name must not contain NUL bytes")
    offset += name_size

    _require_bytes(data, offset, _U32.size, "view-ID count")
    id_count = _U32.unpack_from(data, offset)[0]
    offset += _U32.size
    if not 1 <= id_count <= 255:
        raise DMapError("view-ID count must be between 1 and 255")
    _require_bytes(data, offset, id_count * _U32.size, "view IDs")
    view_ids = struct.unpack_from(f"<{id_count}I", data, offset)
    if NO_ID in view_ids or len(set(view_ids)) != id_count:
        raise DMapError("view IDs must be unique and must not contain NO_ID")
    offset += id_count * _U32.size

    pose_offset = offset
    _require_bytes(data, pose_offset, _POSE.size, "camera matrices")
    offset += _POSE.size
    bytes_per_pixel = 4
    if flags & HAS_NORMAL:
        bytes_per_pixel += 12
    if flags & HAS_CONF:
        bytes_per_pixel += 4
    if flags & HAS_VIEWS:
        bytes_per_pixel += 4
    expected_size = offset + pixels * bytes_per_pixel
    if expected_size > max_bytes:
        raise DMapError(f"dmap layout exceeds max_bytes ({max_bytes})")
    if len(data) < expected_size:
        raise DMapError("truncated dmap array payload")
    if len(data) > expected_size:
        raise DMapError("unexpected trailing bytes after dmap array payload")
    if not all(math.isfinite(value) for value in _POSE.unpack_from(data, pose_offset)):
        raise DMapError("camera matrices contain non-finite values")

    # Each array is a view of the immutable bytes snapshot, never a writable
    # owning array whose WRITEABLE flag could later be re-enabled.
    K = np.frombuffer(data, dtype="<f8", count=9, offset=pose_offset).reshape(3, 3)
    R = np.frombuffer(data, dtype="<f8", count=9, offset=pose_offset + 72).reshape(3, 3)
    C = np.frombuffer(data, dtype="<f8", count=3, offset=pose_offset + 144)
    depth = np.frombuffer(data, dtype="<f4", count=pixels, offset=offset).reshape(
        depth_h, depth_w
    )
    offset += pixels * 4
    _require_finite("depth", depth)
    if (depth < 0).any():
        raise DMapError("depth contains negative values")

    normal = None
    if flags & HAS_NORMAL:
        normal = np.frombuffer(
            data, dtype="<f4", count=pixels * 3, offset=offset
        ).reshape(depth_h, depth_w, 3)
        offset += pixels * 12
        _require_finite("normal", normal)
    confidence = None
    if flags & HAS_CONF:
        confidence = np.frombuffer(
            data, dtype="<f4", count=pixels, offset=offset
        ).reshape(depth_h, depth_w)
        offset += pixels * 4
        _require_finite("confidence", confidence)
        if (confidence < 0).any():
            raise DMapError("confidence contains negative values")
    views = None
    if flags & HAS_VIEWS:
        views = np.frombuffer(
            data, dtype="u1", count=pixels * 4, offset=offset
        ).reshape(depth_h, depth_w, 4)

    return DepthMap(
        path=absolute_path,
        sha256=hashlib.sha256(data).hexdigest(),
        file_bytes=len(data),
        flags=flags,
        reserved=reserved,
        image_size=(image_w, image_h),
        depth_size=(depth_w, depth_h),
        d_min=d_min,
        d_max=d_max,
        image_name_bytes=image_name,
        view_ids=view_ids,
        K=K,
        R=R,
        C=C,
        depth=depth,
        normal=normal,
        confidence=confidence,
        views=views,
    )
