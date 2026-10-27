"""Native model reading, describing and verification (``sfm-native-v1`` §4, §5).

Why this module imports pycolmap lazily
---------------------------------------
``cli`` imports this module.  With a top-level ``import pycolmap``, running the
CLI without the native backend raised ``ModuleNotFoundError`` during import —
before any error handling existed — so the entry point died with exit 1 instead of
reporting ``BACKEND_UNAVAILABLE``/exit 3 and writing its evidence.  Deferring the
import to the call sites makes "no backend" an ordinary, reportable condition, and
keeps ``--help`` working.

What "verified" means here
--------------------------
Comparing two exports against each other only shows they agree; it does not show
either matches the *source*.  Rig and frame identity is the sharpest example: the
native writers happily round-trip a model whose rig and frame ids were all shifted
by +100, so a count-only comparison reports no difference on a changed model.  The
description below therefore carries identities, relations and poses, and the
comparison walks them recursively — exact equality for ids and topology,
tolerance for numbers.
"""

from __future__ import annotations

import enum
import json
import math
from typing import Any

import numpy as np

from .colmap_adapter import (
    COORDINATE_CONVENTION,
    MODEL_NOT_USABLE_MIN_IMAGES,
    MODEL_NOT_USABLE_MIN_POINTS,
    BackendUnavailable,
    canonical_digest,
    sha256_file,
)

__all__ = [
    "COORDINATE_CONVENTION",
    "camera_center",
    "camera_record",
    "compare_descriptions",
    "compare_models",
    "describe_model",
    "is_usable",
    "model_files_digest",
    "quaternion_xyzw_to_matrix",
    "read_binary_model",
    "read_text_model",
    "require_pycolmap",
]

_USABLE_MIN_REGISTERED_IMAGES = MODEL_NOT_USABLE_MIN_IMAGES
_USABLE_MIN_POINTS = MODEL_NOT_USABLE_MIN_POINTS


def require_pycolmap() -> Any:
    """Return the native module, or raise a *reportable* backend error."""
    try:
        import pycolmap
    except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
        raise BackendUnavailable(
            f"the native backend cannot be imported: {type(exc).__name__}: {exc}"
        ) from exc
    return pycolmap


def read_binary_model(binary_dir) -> Any:
    """Read a reconstruction from a COLMAP binary model directory."""
    import pathlib

    pycolmap = require_pycolmap()
    directory = pathlib.Path(binary_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"binary model directory does not exist: {directory}")
    return pycolmap.Reconstruction(str(directory))


def read_text_model(text_dir) -> Any:
    """Read a reconstruction from a COLMAP text model directory."""
    import pathlib

    pycolmap = require_pycolmap()
    directory = pathlib.Path(text_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"text model directory does not exist: {directory}")
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(directory))
    return reconstruction


def _plain(value: Any) -> Any:
    """Convert native values to plain, order-stable, JSON-comparable data.

    Uses the version's own ``todict`` where it exists, so the description is the
    library's serialization rather than a hand-written summary of it.  Sets are
    sorted by their serialized form: ``Rig.sensor_ids`` returns a *set*, and an
    unordered comparison would make the verification result depend on hash order.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        rendered = [_plain(item) for item in value]
        return sorted(rendered, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "todict"):
        return _plain(value.todict())
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.ravel().tolist()]
    if hasattr(value, "tolist"):
        try:
            return _plain(value.tolist())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except Exception:  # noqa: BLE001
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def quaternion_xyzw_to_matrix(quaternion: Any) -> np.ndarray:
    """Hamilton ``wxyz`` quaternion to a rotation matrix.

    Computed here rather than via the native ``rotation.matrix()`` so the camera
    centre below follows the contract's own formula rather than an API whose
    convention would need separate verification.
    """
    values = np.asarray(quaternion, dtype=float).ravel()
    if values.size != 4:
        raise ValueError(f"expected 4 quaternion components, got {values.size}")
    x, y, z, w = values
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm == 0.0:
        raise ValueError("degenerate quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_center(quaternion_xyzw: Any, translation: Any) -> list[float]:
    """``C = -R^T t`` for the world-to-camera pose ``Xc = R Xw + t``."""
    rotation = quaternion_xyzw_to_matrix(quaternion_xyzw)
    return [float(v) for v in (-rotation.T @ np.asarray(translation, dtype=float)).ravel()]


def camera_record(camera: Any) -> dict[str, Any]:
    """Full camera description, including the complete parameter vector."""
    return {
        "camera_id": int(camera.camera_id),
        "model_name": camera.model_name,
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(value) for value in np.asarray(camera.params).ravel()],
        "has_prior_focal_length": bool(camera.has_prior_focal_length),
    }


def _rig_record(rig: Any) -> dict[str, Any]:
    """Rig identity, its sensors (as a sorted list) and their rig-relative poses."""
    record: dict[str, Any] = {
        "rig_id": int(rig.rig_id),
        "sensor_ids": _plain(rig.sensor_ids()),
        "ref_sensor_id": _plain(rig.ref_sensor_id),
        "num_sensors": int(rig.num_sensors()),
    }
    poses = {}
    for sensor in rig.sensor_ids():
        key = _plain(sensor)
        try:
            poses[key if isinstance(key, str) else str(key)] = _plain(
                rig.sensor_from_rig(sensor)
            )
        except Exception:  # noqa: BLE001 - surfaced through the comparison
            poses[str(key)] = None
    record["sensor_from_rig"] = dict(sorted(poses.items()))
    return record


def _frame_record(frame: Any) -> dict[str, Any]:
    """Frame identity, its rig link, its data ids and its world-to-rig pose."""
    record: dict[str, Any] = {
        "frame_id": int(frame.frame_id),
        "rig_id": int(frame.rig_id),
        "data_ids": _plain(frame.data_ids() if callable(frame.data_ids) else frame.data_ids),
        "image_ids": _plain(frame.image_ids),
    }
    if frame.has_pose():
        # In pycolmap 4.2.1 this is a Rigid3d property, not a method.
        pose = frame.rig_from_world
        if pose is None:
            raise ValueError(f"frame {frame.frame_id} reports a pose but has no rig_from_world")
        record["rig_from_world"] = _plain(pose)
    else:
        record["rig_from_world"] = None
    return record


def describe_model(reconstruction: Any) -> dict[str, Any]:
    """A complete, comparable description of one native component.

    Identities and relations are included deliberately: counts alone cannot
    distinguish a correct model from one whose rig and frame ids were all
    renumbered, and the native format preserves both.
    """
    cameras = {str(int(cid)): camera_record(reconstruction.camera(cid))
               for cid in sorted(int(c) for c in reconstruction.cameras)}

    images = {}
    for image_id in sorted(int(i) for i in reconstruction.images):
        image = reconstruction.image(image_id)
        pose = image.cam_from_world()
        quaternion = [float(v) for v in np.asarray(pose.rotation.quat).ravel()]
        translation = [float(v) for v in np.asarray(pose.translation).ravel()]
        images[str(image_id)] = {
            "name": image.name,
            "camera_id": int(image.camera_id),
            "image_id": int(image.image_id),
            "frame_id": int(image.frame_id),
            "quaternion_xyzw": quaternion,
            "translation": translation,
            "camera_center": camera_center(quaternion, translation),
            "points2D": [
                [float(v) for v in np.asarray(image.points2D[i].xy).ravel()]
                for i in range(image.num_points2D())
            ],
            "point3D_ids": [int(image.points2D[i].point3D_id) for i in range(image.num_points2D())],
        }

    points = {}
    for point_id in sorted(int(p) for p in reconstruction.points3D):
        point = reconstruction.point3D(point_id)
        points[str(point_id)] = {
            "xyz": [float(v) for v in np.asarray(point.xyz).ravel()],
            "track": [
                [int(element.image_id), int(element.point2D_idx)]
                for element in point.track.elements
            ],
            "color": [int(v) for v in np.asarray(point.color).ravel()],
        }

    rigs = {str(int(rig_id)): _rig_record(reconstruction.rig(rig_id))
            for rig_id in sorted(int(r) for r in reconstruction.rigs)}
    frames = {str(int(frame_id)): _frame_record(reconstruction.frame(frame_id))
              for frame_id in sorted(int(f) for f in reconstruction.frames)}

    return {
        "num_cameras": int(reconstruction.num_cameras()),
        "num_images": int(reconstruction.num_images()),
        "num_points3D": int(reconstruction.num_points3D()),
        "num_frames": int(reconstruction.num_frames()),
        "num_rigs": int(reconstruction.num_rigs()),
        "num_reg_images": int(reconstruction.num_reg_images()),
        "registered_image_ids": sorted(int(i) for i in reconstruction.reg_image_ids()),
        "cameras": cameras,
        "images": images,
        "points3D": points,
        "rigs": rigs,
        "frames": frames,
    }


def compare_descriptions(left: Any, right: Any, *, tolerance: float) -> list[str]:
    """Recursive difference between two model descriptions.

    Numbers compare with ``tolerance``; ids, names, topology and everything else
    compare exactly, because "these two ids differ" is never a rounding issue.
    """
    differences: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                differences.append(f"{path}: {a!r} != {b!r}")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=0.0):
                differences.append(f"{path}: {a!r} != {b!r}")
        elif isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    differences.append(f"{path}.{key}: missing on the left")
                elif key not in b:
                    differences.append(f"{path}.{key}: missing on the right")
                else:
                    walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append(f"{path}: length {len(a)} != {len(b)}")
            else:
                for index, (item_a, item_b) in enumerate(zip(a, b)):
                    walk(item_a, item_b, f"{path}[{index}]")
        elif a != b:
            differences.append(f"{path}: {a!r} != {b!r}")

    walk(left, right, "$")
    return differences


def compare_models(left: Any, right: Any, *, tolerance: float) -> list[str]:
    """Field-by-field differences between two components."""
    return compare_descriptions(
        describe_model(left), describe_model(right), tolerance=tolerance
    )


def is_usable(reconstruction: Any) -> tuple[bool, list[str]]:
    """Whether the component meets the contract's *structural* floor.

    A model passing here is well formed, not accurate, and says nothing about
    object completeness.
    """
    reasons: list[str] = []
    registered = int(reconstruction.num_reg_images())
    points = int(reconstruction.num_points3D())
    if registered < _USABLE_MIN_REGISTERED_IMAGES:
        reasons.append(f"only {registered} registered image(s), need >= {_USABLE_MIN_REGISTERED_IMAGES}")
    if points < _USABLE_MIN_POINTS:
        reasons.append(f"only {points} point(s), need >= {_USABLE_MIN_POINTS}")
    for camera_id in sorted(int(c) for c in reconstruction.cameras):
        try:
            camera = reconstruction.camera(camera_id)
            if not camera.verify_params():
                reasons.append(f"camera {camera_id} has invalid parameters")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"camera {camera_id} is unreadable: {type(exc).__name__}")
    return (not reasons), reasons


def model_files_digest(model_dir) -> str:
    """Canonical digest over the files of one exported component."""
    import pathlib

    model_dir = pathlib.Path(model_dir)
    pairs = [
        [path.relative_to(model_dir).as_posix(), sha256_file(path)]
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    ]
    return canonical_digest(pairs)
