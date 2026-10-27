"""Compare saved depth outcomes; this API never certifies native execution provenance.

Input geometry is supplied at actual depth resolution by the caller. Binary masks
are already in the undistorted image domain, not SAM/photo-canonical coordinates.
Resizing is a documented diagnostic reference, not proof of a native OpenCV build.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image

from .dmap import read_dmap

CONTRACT = "openmvs-depth-outcome-v1"
MAX_PIXELS = 16_777_216
MAX_IMAGE_BYTES = 64 * 1024 * 1024


class DepthVerificationError(ValueError):
    """Input identity, format, or declared comparison precondition is invalid."""


def _inside(root: Path, path: str | Path) -> Path:
    try:
        path = Path(path).resolve(strict=True)
        path.relative_to(root)
        if not path.is_file():
            raise ValueError("not a regular file")
        return path
    except (OSError, ValueError) as exc:
        raise DepthVerificationError(f"file missing or outside input root: {path}") from exc


def _read(path: Path, expected_sha256: str) -> bytes:
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise DepthVerificationError("SHA-256 must be 64 lowercase hex digits")
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= MAX_IMAGE_BYTES:
            raise DepthVerificationError("image exceeds byte bound or is empty")
        data = stream.read(MAX_IMAGE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(data) != before.st_size or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise DepthVerificationError("input changed while reading")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise DepthVerificationError(f"SHA-256 mismatch: {path.name}")
    return data


def _pixels(data: bytes, *, label: bool) -> tuple[tuple[int, int], np.ndarray | None]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            w, h = image.size
            if min(w, h) <= 0 or w * h > MAX_PIXELS:
                raise DepthVerificationError("image exceeds pixel bound")
            if getattr(image, 'n_frames', 1) != 1 or image.getexif().get(274, 1) != 1:
                raise DepthVerificationError("multi-frame or oriented image is unsupported")
            if label:
                # Check source PNG encoding rather than trusting a decoded conversion.
                if (image.format != 'PNG' or image.mode != 'L' or len(data) < 26
                        or data[:8] != b'\x89PNG\r\n\x1a\n'
                        or data[24:26] != b'\x08\x00' or 'transparency' in image.info):
                    raise DepthVerificationError("mask must be 8-bit grayscale PNG without alpha")
                array = np.array(image, dtype=np.uint8)
                if not np.isin(array, (0, 255)).all():
                    raise DepthVerificationError("mask labels must be exactly 0/255")
                return (w, h), array
            image.load()  # Reject a plausible header with a truncated/corrupt pixel payload.
            return (w, h), None
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, DepthVerificationError):
            raise
        raise DepthVerificationError("invalid image encoding") from exc


def legacy_nearest_reference(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """OpenCV 4.10 scalar legacy NN reference; native build equivalence is unverified.

    Deliberately use double division then reciprocal as the source does. This is
    not PIL nearest and not an assertion about every SIMD/OpenCV implementation.
    """
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.uint8
            or labels.ndim != 2 or labels.size == 0 or labels.size > MAX_PIXELS
            or not np.isin(labels, (0, 255)).all()):
        raise DepthVerificationError("reference sampler needs bounded uint8 binary labels")
    if (not isinstance(size, tuple) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or math.prod(size) > MAX_PIXELS):
        raise DepthVerificationError("invalid target size")
    w, h = size
    sh, sw = labels.shape
    xs = np.minimum(np.floor(np.arange(w, dtype=np.float64) * (1.0 / (w / sw))).astype(np.int64), sw - 1)
    ys = np.minimum(np.floor(np.arange(h, dtype=np.float64) * (1.0 / (h / sh))).astype(np.int64), sh - 1)
    return labels[np.ix_(ys, xs)]


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    # Validate shape without copying a huge broadcast view or accepting a
    # custom __array__ implementation. At most nine scalar values are copied.
    def valid_structure(item: Any, wanted: tuple[int, ...]) -> bool:
        if type(item) is np.ndarray:
            return item.shape == wanted and item.dtype.kind in 'iuf'
        if not wanted:
            return type(item) in (int, float) or isinstance(item, (np.integer, np.floating))
        return (type(item) in (list, tuple) and len(item) == wanted[0]
                and all(valid_structure(row, wanted[1:]) for row in item))
    if not valid_structure(value, shape):
        raise DepthVerificationError(f"invalid {name} shape or scalar type")
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DepthVerificationError(f"invalid {name}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise DepthVerificationError(f"invalid {name}")
    return array


def _linked_image(dmap: Any, root: Path, expected: Path) -> None:
    try:
        name = dmap.image_name_bytes.decode('utf-8', errors='strict')
    except UnicodeDecodeError as exc:
        raise DepthVerificationError("image path encoding unsupported; explicit identity needed") from exc
    win = PureWindowsPath(name)
    if win.drive or win.root or name.startswith('/') or ':' in name:
        raise DepthVerificationError("native image path must be relative")
    # Native Windows and POSIX separators are both understood for these ASCII paths.
    linked = _inside(root, Path(dmap.path).parent / name.replace('\\', '/'))
    if linked != expected:
        raise DepthVerificationError("native image reference does not match supplied image identity")


def verify_depth_pair(
    masked_path: str | Path, control_path: str | Path, *,
    input_root: str | Path, masked_image_path: str | Path, control_image_path: str | Path,
    image_sha256: str, mask_path: str | Path, mask_sha256: str,
    validity_path: str | Path, validity_sha256: str,
    reference_id: int, expected_K: Any, expected_R: Any, expected_C: Any,
    masked_ignore_label: int, control_ignore_label: int, postprocess_dmaps: int,
) -> dict:
    """Read two separate artifacts and report a bounded per-reference outcome.

    Parameters declare a matched pair; this API cannot authenticate argv, scene,
    native binary, freshness, masks actually read, or clean output directories.
    The returned native_consumption_status therefore ALWAYS remains NOT_VERIFIED.
    SUPPORTED means only that saved arrays exhibit the declared exclusion effect.
    A resized mask always yields INCONCLUSIVE until actual native sampling is bound.
    """
    root = Path(input_root).resolve(strict=True)
    if not root.is_dir():
        raise DepthVerificationError("input_root must be a directory")
    if type(reference_id) is not int or not 0 <= reference_id < 0xffffffff:
        raise DepthVerificationError("reference_id must be a global native uint32 ID")
    if (type(masked_ignore_label) is not int or masked_ignore_label != 0
            or type(control_ignore_label) is not int or control_ignore_label != -1):
        raise DepthVerificationError("comparison requires explicit ignore labels 0 and -1")
    if type(postprocess_dmaps) is not int or not 0 <= postprocess_dmaps <= 15:
        raise DepthVerificationError("invalid postprocess_dmaps flags")
    paths = {name: _inside(root, path) for name, path in {
        'masked_depth': masked_path, 'control_depth': control_path,
        'masked_image': masked_image_path, 'control_image': control_image_path,
        'mask': mask_path, 'validity': validity_path}.items()}
    if os.path.samefile(paths['masked_depth'], paths['control_depth']):
        raise DepthVerificationError("control must be a distinct depth artifact")
    masked, control = (read_dmap(paths[n]) for n in ('masked_depth', 'control_depth'))
    if masked.depth_size != control.depth_size or masked.view_ids != control.view_ids:
        raise DepthVerificationError("depth dimensions or reference/neighbor identities differ")
    if masked.view_ids[0] != reference_id:
        raise DepthVerificationError("wrong native reference ID")
    # This checker targets Densify DepthData::Save, which passes depth size as image size.
    if any(d.image_size != d.depth_size for d in (masked, control)):
        raise DepthVerificationError("unsupported image/depth header convention")
    K = _matrix(expected_K, (3, 3), 'K')
    R = _matrix(expected_R, (3, 3), 'R')
    C = _matrix(expected_C, (3,), 'C')
    if (min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-7, rtol=0)
            or abs(K[1, 0]) > 1e-7 or np.linalg.slogdet(K)[0] <= 0):
        raise DepthVerificationError("expected K is not a positive pinhole calibration")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(R) - 1) > 1e-8:
        raise DepthVerificationError("expected R is not a proper rotation")
    for d in (masked, control):
        for name, wanted, tolerance in [('K', K, 1e-7), ('R', R, 1e-8), ('C', C, 1e-8)]:
            if not np.allclose(getattr(d, name), wanted, atol=tolerance, rtol=0):
                raise DepthVerificationError(f"native {name} differs from supplied depth-resolution geometry")
    _linked_image(masked, root, paths['masked_image'])
    _linked_image(control, root, paths['control_image'])
    masked_bytes = _read(paths['masked_image'], image_sha256)
    control_bytes = _read(paths['control_image'], image_sha256)
    source_size, _ = _pixels(masked_bytes, label=False)
    control_size, _ = _pixels(control_bytes, label=False)
    mask_bytes = _read(paths['mask'], mask_sha256)
    validity_bytes = _read(paths['validity'], validity_sha256)
    mask_size, labels = _pixels(mask_bytes, label=True)
    valid_size, valid = _pixels(validity_bytes, label=True)
    if not (source_size == control_size == mask_size == valid_size):
        raise DepthVerificationError("image/mask/validity dimensions disagree")
    if np.any((labels == 255) & (valid == 0)):
        raise DepthVerificationError("foreground is marked geometrically unknown")
    resized = mask_size != masked.depth_size
    if resized:
        labels = legacy_nearest_reference(labels, masked.depth_size)
        valid = legacy_nearest_reference(valid, masked.depth_size)
    ignored = labels == 0
    known_background = ignored & (valid == 255)
    unknown = valid == 0
    foreground = labels == 255
    md, cd = masked.depth > 0, control.depth > 0
    count = lambda array: int(np.count_nonzero(array))
    counts = {
        'total_pixels': int(labels.size), 'ignored_pixels': count(ignored),
        'known_background_pixels': count(known_background), 'unknown_pixels': count(unknown),
        'foreground_pixels': count(foreground),
        'masked_ignored_nonzero': count(md & ignored),
        'masked_known_background_nonzero': count(md & known_background),
        'masked_unknown_nonzero': count(md & unknown),
        'masked_foreground_nonzero': count(md & foreground),
        'control_known_background_nonzero': count(cd & known_background),
        'control_foreground_nonzero': count(cd & foreground),
        'known_background_removed': count(cd & ~md & known_background),
    }
    reasons = []
    if resized:
        reasons.append('NATIVE_RESIZE_MAPPING_UNVERIFIED')
    if postprocess_dmaps != 0:
        reasons.append('POSTPROCESS_ENABLED_OUTSIDE_FROZEN_COMPARISON')
    if reasons:
        outcome = 'INCONCLUSIVE'
    elif counts['masked_ignored_nonzero']:
        outcome = 'VIOLATION'
        reasons.append('NONZERO_DEPTH_IN_DECLARED_IGNORED_PIXELS')
    else:
        for key, reason in [
            ('known_background_pixels', 'NO_KNOWN_BACKGROUND'),
            ('masked_foreground_nonzero', 'NO_MASKED_FOREGROUND_DEPTH'),
            ('control_foreground_nonzero', 'NO_CONTROL_FOREGROUND_DEPTH'),
            ('control_known_background_nonzero', 'NO_CONTROL_BACKGROUND_SIGNAL'),
        ]:
            if counts[key] == 0:
                reasons.append(reason)
        outcome = 'INCONCLUSIVE' if reasons else 'SUPPORTED'
    bindings = [
        {'role': 'masked_depth', 'path': masked.path, 'bytes': masked.file_bytes, 'sha256': masked.sha256},
        {'role': 'control_depth', 'path': control.path, 'bytes': control.file_bytes, 'sha256': control.sha256},
    ]
    for role, data, digest in [('masked_image', masked_bytes, image_sha256),
            ('control_image', control_bytes, image_sha256), ('mask', mask_bytes, mask_sha256),
            ('validity', validity_bytes, validity_sha256)]:
        bindings.append({'role': role, 'path': str(paths[role]), 'bytes': len(data), 'sha256': digest})
    # Detect path replacement as well as in-place mutation during this complete check.
    for binding in bindings:
        path = _inside(root, binding['path'])
        limit = 536870912 if binding['role'].endswith('depth') else MAX_IMAGE_BYTES
        with path.open('rb') as stream:
            if os.fstat(stream.fileno()).st_size != binding['bytes']:
                raise DepthVerificationError("input changed during comparison")
            digest = hashlib.sha256()
            total = 0
            while block := stream.read(1024 * 1024):
                total += len(block)
                if total > limit:
                    raise DepthVerificationError("input grew during comparison")
                digest.update(block)
        if total != binding['bytes'] or digest.hexdigest() != binding['sha256']:
            raise DepthVerificationError("input changed during comparison")
    return {
        'contract_version': CONTRACT, 'outcome_status': outcome, 'reasons': reasons,
        'comparison_scope': 'one_reference_saved_depth_outcome',
        'native_consumption_status': 'NOT_VERIFIED', 'quality_status': 'NOT_EVALUATED',
        'runtime_provenance_status': 'NOT_VERIFIED', 'mask_alignment_status': 'CALLER_SUPPLIED',
        'configuration_status': 'DECLARED_NOT_NATIVE_VERIFIED',
        'sampling_status': 'REFERENCE_ONLY' if resized else 'IDENTITY_NO_RESIZE',
        'sampling_algorithm': 'opencv-4.10-scalar-legacy-nearest-reference' if resized else 'identity',
        'source_image_size': list(source_size), 'actual_depth_size': list(masked.depth_size),
        'native_reference_id': reference_id, 'view_ids': list(masked.view_ids),
        'geometry_tolerance': {'K_absolute': 1e-7, 'R_C_absolute': 1e-8, 'relative': 0},
        'expected_depth_geometry': {'K': K.tolist(), 'R': R.tolist(), 'C': C.tolist()},
        'declared_options': {'masked_ignore_label': masked_ignore_label,
                             'control_ignore_label': control_ignore_label,
                             'postprocess_dmaps': postprocess_dmaps},
        'counts': counts, 'inputs': bindings, 'inputs_stable': True,
    }
