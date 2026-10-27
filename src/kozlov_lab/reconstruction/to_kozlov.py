"""Identity-preserving pinhole SfM adapter (engineering, not a Kozlov algorithm).

The native sidecar retains multiplicity and observations; PointSet3D is a
separate exact-coordinate set view. No reconstruction, undistortion, half-pixel
shift, normalization, scale inference or combinatorial code search occurs here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .export import describe_model, quaternion_xyzw_to_matrix
from ..stereo.baseline.camera import Camera
from ..stereo.model import PointSet3D

CONTRACT = "research-native-pinhole-v1"
MAX_POINTS = 100_000
MAX_FEATURES = 1_000_000
MAX_IMAGES = 256
MAX_MODEL_BYTES = 64 * 1024**2
INVALID_POINT_ID = 2**64 - 1
PROJECTION_TOLERANCE = 1e-8  # pixels: numerical compatibility, not model accuracy
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")


class AdapterError(ValueError):
    """Invalid, unsupported or inconsistent input; no successful scene produced."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = INVALID_POINT_ID - 1) -> int:
    _require(type(value) is int and minimum <= value <= maximum, f"invalid integer {name}")
    return value


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    def walk(x):
        if isinstance(x, (list, tuple)):
            return all(walk(v) for v in x)
        return type(x) in (int, float) and math.isfinite(x)
    _require(walk(value), f"invalid/nonfinite numeric value in {name}")
    try:
        a = np.array(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AdapterError(f"invalid shape in {name}") from exc
    _require(a.shape == shape and np.isfinite(a).all(), f"invalid shape/nonfinite {name}")
    return a


def _records(value: Any, name: str, maximum: int) -> dict[int, dict]:
    _require(isinstance(value, dict) and len(value) <= maximum, f"invalid/oversized {name}")
    result = {}
    for key, row in value.items():
        _require(isinstance(key, str) and key.isascii() and key.isdecimal(), f"invalid {name} key")
        ident = int(key)
        _require(str(ident) == key, f"noncanonical {name} key")
        _integer(ident, name + " ID", minimum=1, maximum=INVALID_POINT_ID - 1 if name == "points3D" else 2**32 - 2)
        _require(isinstance(row, dict), f"invalid {name} row")
        result[ident] = row
    return result


def _pose(row: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    _require(isinstance(row, dict), f"missing {name} pose")
    q = _array(row["rotation"]["quat"], (4,), name + " quaternion")
    _require(abs(math.hypot(*q) - 1) <= 1e-10, f"nonunit {name} quaternion")
    t = _array(row["translation"], (3,), name + " translation")
    if "params" in row:
        params = _array(row["params"], (7,), name + " params")
        _require(np.allclose(params, np.r_[q, t], rtol=0, atol=1e-12), f"inconsistent {name} pose params")
    return quaternion_xyzw_to_matrix(q), t


def _sensor(row: dict) -> int:
    _require(isinstance(row, dict) and row["type"] == "SensorType.CAMERA", "only camera sensors are supported")
    return _integer(row["id"], "sensor camera ID", minimum=1)


@dataclass
class ResearchScene:
    native_description: dict
    point_ids: np.ndarray
    native_xyz: np.ndarray
    body: PointSet3D
    point_id_to_body_row: dict[int, int]
    cameras: dict[int, Camera]
    duplicate_groups: list[list[int]]
    report: dict

    def rays(self, image_id: int, xy: Any) -> tuple[np.ndarray, np.ndarray]:
        """World camera centre and z=1 rays; does not call legacy Camera.ray."""
        _integer(image_id, "ray image ID", minimum=1)
        _require(image_id in self.cameras, "ray image absent")
        _require(isinstance(xy, (list, tuple)) and len(xy) > 0, "rays require a nonempty list of pixel pairs")
        pixels = _array(xy, (len(xy), 2), "ray pixels")
        c = self.cameras[image_id]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized = (pixels - c.intrinsics[:2, 2]) / np.array([c.intrinsics[0, 0], c.intrinsics[1, 1]])
            directions = np.c_[normalized, np.ones(len(pixels))] @ c.rotation
            center = -c.rotation.T @ c.translation
        _require(np.isfinite(directions).all() and np.isfinite(center).all(), "nonfinite world ray")
        return center, directions

    def payload(self) -> dict:
        """Strict JSON scene, without discarding the original observations."""
        return {
            "contract": CONTRACT,
            "native_description": copy.deepcopy(self.native_description),
            "point_ids": [int(v) for v in self.point_ids],
            "native_xyz": self.native_xyz.tolist(),
            "body_xyz": self.body.as_array().tolist(),
            "point_id_to_body_row": {str(k): v for k, v in self.point_id_to_body_row.items()},
            "duplicate_groups": copy.deepcopy(self.duplicate_groups),
            "research_cameras": {str(k): {
                "K": c.intrinsics.tolist(), "R": c.rotation.tolist(),
                "t": c.translation.tolist(), "P": c.projection.tolist(),
                "metadata": copy.deepcopy(c.metadata),
            } for k, c in self.cameras.items()},
            "report": copy.deepcopy(self.report),
        }


def adapt_description(description: dict) -> ResearchScene:
    """Validate a describe_model snapshot and construct existing research types.

    Only registered, already-pinhole components are supported. The sidecar is
    not claimed to replace the complete native binary representation.
    """
    try:
        return _adapt_description(description)
    except (KeyError, TypeError, OverflowError) as exc:
        raise AdapterError(f"malformed native description: {exc}") from exc


def _adapt_description(description: dict) -> ResearchScene:
    _require(isinstance(description, dict), "description must be an object")
    # Check sizes before copying or allocating feature/point arrays.
    points = _records(description["points3D"], "points3D", MAX_POINTS)
    images = _records(description["images"], "images", MAX_IMAGES)
    cameras = _records(description["cameras"], "cameras", MAX_IMAGES)
    frames = _records(description["frames"], "frames", MAX_IMAGES)
    rigs = _records(description["rigs"], "rigs", MAX_IMAGES)
    _require(points and 2 <= len(images), "need a nonempty component with at least two images")
    for name, records in (("cameras", cameras), ("images", images), ("points3D", points), ("frames", frames), ("rigs", rigs)):
        _require(_integer(description["num_" + name], "num_" + name) == len(records), f"count mismatch {name}")
    registered = description["registered_image_ids"]
    _require(isinstance(registered, list), "invalid registered image list")
    for ident in registered:
        _integer(ident, "registered image ID", minimum=1)
    _require(len(registered) == len(set(registered)) and set(registered) == set(images), "unregistered/missing/duplicate image")
    _require(_integer(description["num_reg_images"], "num_reg_images") == len(images), "registered count mismatch")
    feature_count = 0
    for row in images.values():
        _require(isinstance(row["points2D"], list), "features must be a list")
        feature_count += len(row["points2D"])
    _require(feature_count <= MAX_FEATURES, "feature limit exceeded")
    # Reject non-JSON/nonfinite values also in preserved metadata, before copying.
    try:
        json.dumps(description, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AdapterError("description contains nonfinite or non-JSON metadata") from exc

    intrinsics = {}
    for cid, row in cameras.items():
        _require(_integer(row["camera_id"], "camera_id", minimum=1) == cid, "camera key/ID mismatch")
        _integer(row["width"], "camera width", minimum=1, maximum=65536)
        _integer(row["height"], "camera height", minimum=1, maximum=65536)
        model = row["model_name"]
        _require(model in ("PINHOLE", "SIMPLE_PINHOLE"), "only already-undistorted PINHOLE/SIMPLE_PINHOLE is supported")
        params = _array(row["params"], (4 if model == "PINHOLE" else 3,), "intrinsics")
        if model == "PINHOLE":
            fx, fy, cx, cy = params
        else:
            fx, cx, cy = params
            fy = fx
        _require(fx > 0 and fy > 0, "focal length must be positive")
        intrinsics[cid] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    sensor_poses = {}
    for rid, row in rigs.items():
        _require(_integer(row["rig_id"], "rig_id", minimum=1) == rid, "rig key/ID mismatch")
        sensors = row["sensor_ids"]
        _require(isinstance(sensors, list) and sensors, "empty/invalid rig sensors")
        sensor_ids = [_sensor(sensor) for sensor in sensors]
        _require(len(set(sensor_ids)) == len(sensor_ids) and set(sensor_ids) <= set(cameras), "unknown/duplicate rig sensor")
        _require(_integer(row["num_sensors"], "num_sensors", minimum=1) == len(sensors), "rig sensor count mismatch")
        reference = _sensor(row["ref_sensor_id"])
        _require(reference in sensor_ids, "rig reference sensor absent")
        poses = row["sensor_from_rig"]
        _require(isinstance(poses, dict) and set(poses) == {str(sensor) for sensor in sensors}, "rig sensor pose keys mismatch")
        for sensor, cid in zip(sensors, sensor_ids):
            pose = poses[str(sensor)]
            if cid == reference and pose is None:
                rotation, translation = np.eye(3), np.zeros(3)
            else:
                rotation, translation = _pose(pose, "sensor_from_rig")
            if cid == reference:
                _require(np.allclose(rotation, np.eye(3), rtol=0, atol=1e-12) and np.allclose(translation, 0, rtol=0, atol=1e-12), "reference sensor must be identity")
            sensor_poses[rid, cid] = rotation, translation
    frame_members = {}
    frame_poses = {}
    for fid, row in frames.items():
        _require(_integer(row["frame_id"], "frame_id", minimum=1) == fid, "frame key/ID mismatch")
        _require(_integer(row["rig_id"], "frame rig_id", minimum=1) in rigs, "frame refers to absent rig")
        _require(isinstance(row["image_ids"], list), "invalid frame image members")
        _require(isinstance(row["data_ids"], list) and sorted(json.dumps(v, sort_keys=True) for v in row["data_ids"]) == sorted(json.dumps(v, sort_keys=True) for v in row["image_ids"]), "frame data/image members mismatch")
        frame_poses[fid] = _pose(row["rig_from_world"], "rig_from_world")
        members = set()
        for entry in row["image_ids"]:
            iid = _integer(entry["id"], "frame image ID", minimum=1)
            cid = _integer(entry["sensor_id"]["id"], "frame sensor ID", minimum=1)
            _require(entry["sensor_id"]["type"] == "SensorType.CAMERA", "noncamera image sensor")
            _require((row["rig_id"], cid) in sensor_poses, "frame sensor absent from rig")
            _require(iid in images and iid not in members, "missing/duplicate frame member")
            _require(images[iid]["frame_id"] == fid and images[iid]["camera_id"] == cid, "frame/image membership mismatch")
            members.add(iid)
        frame_members[fid] = members

    research_cameras = {}
    image_features = {}
    forward = set()
    names = set()
    for iid, row in images.items():
        _require(_integer(row["image_id"], "image_id", minimum=1) == iid, "image key/ID mismatch")
        cid = _integer(row["camera_id"], "image camera_id", minimum=1)
        fid = _integer(row["frame_id"], "image frame_id", minimum=1)
        _require(cid in intrinsics and fid in frames and iid in frame_members[fid], "missing camera/frame/member")
        _require(isinstance(row["name"], str) and row["name"] and row["name"] not in names, "empty/duplicate image name")
        names.add(row["name"])
        q = _array(row["quaternion_xyzw"], (4,), "quaternion")
        _require(abs(math.hypot(*q) - 1) <= 1e-10, "quaternion must be unit")
        rotation = quaternion_xyzw_to_matrix(q)
        translation = _array(row["translation"], (3,), "translation")
        center = _array(row["camera_center"], (3,), "camera center")
        with np.errstate(over="ignore", invalid="ignore"):
            expected_center = -rotation.T @ translation
            center_matches = np.allclose(center, expected_center, rtol=1e-12, atol=1e-12)
        _require(np.isfinite(expected_center).all() and center_matches, "camera center mismatch/nonfinite")
        rs, ts = sensor_poses[frames[fid]["rig_id"], cid]
        rf, tf = frame_poses[fid]
        with np.errstate(over="ignore", invalid="ignore"):
            composed_r, composed_t = rs @ rf, rs @ tf + ts
        _require(np.allclose(rotation, composed_r, rtol=0, atol=1e-10) and np.allclose(translation, composed_t, rtol=1e-12, atol=1e-12), "image pose inconsistent with rig/frame")
        with np.errstate(over="ignore", invalid="ignore"):
            camera = Camera.from_parts(intrinsics[cid], rotation, translation,
                name=row["name"], native_image_id=iid, native_camera_id=cid, native_frame_id=fid,
                pixel_convention="COLMAP native: top-left pixel center (0.5,0.5); unchanged",
                world_convention="native world, cam_from_world: Xc=R Xw+t; x-right y-down z-forward")
        _require(np.isfinite(camera.projection).all(), "nonfinite camera projection matrix")
        research_cameras[iid] = camera
        count = len(row["points2D"])
        xy = _array(row["points2D"], (count, 2), "points2D") if count else np.empty((0, 2))
        ids = row["point3D_ids"]
        _require(isinstance(ids, list) and len(ids) == count, "feature ID count mismatch")
        image_features[iid] = xy
        for index, pid in enumerate(ids):
            _integer(pid, "feature point ID", minimum=1, maximum=INVALID_POINT_ID)
            if pid != INVALID_POINT_ID:
                _require(pid in points, "feature refers to absent point")
                forward.add((pid, iid, index))

    point_ids = sorted(points)
    xyz = np.empty((len(points), 3), dtype=float)
    reverse = set()
    coordinate_to_row = {}
    body_xyz, group_ids = [], []
    point_id_to_body_row = {}
    for index, pid in enumerate(point_ids):
        row = points[pid]
        xyz[index] = _array(row["xyz"], (3,), "point xyz")
        color = row["color"]
        _require(isinstance(color, list) and len(color) == 3, "invalid point RGB")
        for value in color:
            _integer(value, "color channel", maximum=255)
        track = row["track"]
        _require(isinstance(track, list) and 2 <= len(track) <= len(images), "invalid track length")
        seen = set()
        for pair in track:
            _require(isinstance(pair, list) and len(pair) == 2, "invalid track entry")
            iid = _integer(pair[0], "track image", minimum=1)
            idx = _integer(pair[1], "track point2D index")
            _require(iid in images and iid not in seen, "unregistered or duplicate track view")
            seen.add(iid)
            _require(idx < len(images[iid]["points2D"]), "track feature index outside image")
            _require(images[iid]["point3D_ids"][idx] == pid, "track reciprocity mismatch")
            reverse.add((pid, iid, idx))
        key = tuple(xyz[index])
        if key not in coordinate_to_row:
            coordinate_to_row[key] = len(body_xyz)
            body_xyz.append(xyz[index].copy())
            group_ids.append([])
        body_row = coordinate_to_row[key]
        group_ids[body_row].append(pid)
        point_id_to_body_row[pid] = body_row
    _require(forward == reverse, "image/track reciprocity mismatch")

    # Residuals describe retained measurements, never a physical-accuracy gate.
    residuals = []
    id_to_index = {pid: i for i, pid in enumerate(point_ids)}
    observations_by_image = {iid: [] for iid in images}
    for pid, iid, idx in sorted(forward):
        observations_by_image[iid].append((pid, idx))
    for iid, entries in observations_by_image.items():
        if not entries:
            continue
        world = xyz[[id_to_index[pid] for pid, _ in entries]]
        camera = research_cameras[iid]
        camera_xyz = world @ camera.rotation.T + camera.translation
        _require(np.isfinite(camera_xyz).all() and np.all(camera_xyz[:, 2] > 1e-12), "tracked point behind/on camera plane or nonfinite")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = camera.project(world)
            measured = image_features[iid][[idx for _, idx in entries]]
            delta = np.linalg.norm(projected - measured, axis=1)
        _require(np.isfinite(projected).all() and np.isfinite(delta).all(), "nonfinite projection/residual")
        residuals.extend(delta.tolist())
    identity = np.eye(4).tolist()
    duplicates = [group for group in group_ids if len(group) > 1]
    report = {
        "status": "ADAPTED_PINHOLE_NATIVE", "contract": CONTRACT,
        "native_points": len(points), "body_points": len(body_xyz), "images": len(images),
        "all_features": feature_count, "tracked_observations": len(forward),
        "untriangulated_features": feature_count - len(forward),
        "duplicate_coordinate_groups": len(duplicates),
        "scale_status": "unknown", "length_unit": None,
        "native_to_research": identity, "research_to_native": copy.deepcopy(identity),
        "transform_policy": "identity only; no centering, rescaling, axis or pixel shift",
        "set_policy": "exact xyz equality only; all native IDs/records remain in sidecar",
        "reprojection_residual_pixels": {"count": len(residuals), "max": max(residuals),
            "rms": float(np.linalg.norm(np.asarray(residuals)) / math.sqrt(len(residuals)))},
        "physical_accuracy": "NOT_EVALUATED", "kozlov_code_or_oracle": "NOT_RUN",
    }
    _require(math.isfinite(report["reprojection_residual_pixels"]["rms"]), "residual aggregate overflow")
    body = PointSet3D(np.array(body_xyz, dtype=float), name="native exact-coordinate set")
    return ResearchScene(copy.deepcopy(description), np.array(point_ids, dtype=np.uint64), xyz.copy(),
        body, point_id_to_body_row, research_cameras, duplicates, report)


def _binding(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_native_scene(model_dir: str | Path) -> ResearchScene:
    """Read the five bound 4.2.1 binary files, preserving source bytes.

    Size/count bounds are input ceilings, not an OS memory/time supervisor.
    Callers running untrusted inputs must provide their own process budget.
    """
    directory = Path(model_dir).resolve()
    _require(directory.is_dir(), "native model directory missing")
    files = [directory / name for name in MODEL_FILES]
    _require(all(p.is_file() for p in files), "five native binary model files are required")
    _require(not any(directory.glob("*.txt")), "mixed binary/text model directory is ambiguous")
    _require(sum(p.stat().st_size for p in files) <= MAX_MODEL_BYTES, "native model byte limit exceeded")
    before = [_binding(p) for p in files]
    import pycolmap  # lazy: pure description adaptation needs only NumPy
    _require(pycolmap.__version__ == "4.2.1", "native adapter supports validated pycolmap 4.2.1 only")
    model = pycolmap.Reconstruction(str(directory))
    _require(model.num_points3D() <= MAX_POINTS and model.num_images() <= MAX_IMAGES, "native model count limit exceeded")
    _require(sum(im.num_points2D() for im in model.images.values()) <= MAX_FEATURES, "native feature limit exceeded")
    _require(model.num_reg_images() == model.num_images(), "unregistered native images are unsupported")
    description = describe_model(model)
    scene = adapt_description(description)
    errors = {str(pid): float(p.error) for pid, p in model.points3D.items()}
    _require(all(math.isfinite(v) for v in errors.values()), "nonfinite native point error")
    # Independent native projection and inverse-intrinsic comparisons; no model fitting.
    max_projection = max_ray = 0.0
    for iid, research in scene.cameras.items():
        image = model.images[iid]
        camera = model.cameras[image.camera_id]
        entries = [(j, int(p.point3D_id)) for j, p in enumerate(image.points2D) if p.has_point3D()]
        if entries:
            xyz = np.array([model.points3D[pid].xyz for _, pid in entries])
            native_camera_xyz = image.cam_from_world() * xyz
            native_pixels = np.asarray(camera.img_from_cam(native_camera_xyz))
            ours = research.project(xyz)
            _require(native_pixels.shape == ours.shape and np.isfinite(native_pixels).all(), "invalid native projection")
            max_projection = max(max_projection, float(np.max(np.abs(native_pixels - ours))))
        if image.num_points2D():
            features = np.array([p.xy for p in image.points2D])
            rays = np.asarray(camera.cam_from_img(features))
            k = research.intrinsics
            ours = (features - k[:2, 2]) / np.array([k[0, 0], k[1, 1]])
            _require(rays.shape == ours.shape and np.isfinite(rays).all(), "invalid native inverse projection")
            max_ray = max(max_ray, float(np.max(np.abs(rays - ours))))
    _require(max_projection <= PROJECTION_TOLERANCE and max_ray <= 1e-12, "research/native projection mismatch")
    after = [_binding(p) for p in files]
    _require(before == after, "source model changed during adaptation")
    scene.report.update(source_files=before, native_backend_version=pycolmap.__version__,
        native_point_errors=errors, native_projection_max_abs_pixels=max_projection,
        native_inverse_intrinsics_max_abs=max_ray, projection_tolerance_pixels=PROJECTION_TOLERANCE)
    return scene


def write_scene(scene: ResearchScene, output: str | Path) -> dict:
    """Write a new self-contained research scene. No overwrite or fake CAD."""
    target = Path(output).resolve()
    root = Path(__file__).resolve().parents[3] / "outputs"
    _require(target.is_relative_to(root.resolve()) and target != root.resolve(), "output must be a new project outputs directory")
    _require(not target.exists(), "output already exists")
    payload = scene.payload()
    encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    _require(len(encoded) <= 128 * 1024**2, "scene JSON output limit exceeded")
    # Before publication, detect drift of native files bound by the loader.
    for item in scene.report.get("source_files", []):
        _require(_binding(Path(item["path"])) == item, "source model drift before publication")
    target.mkdir(parents=True, exist_ok=False)
    (target / "research_scene.json").write_bytes(encoded)
    result = {"status": "EXPORTED_RESEARCH_SCENE", "contract": CONTRACT,
        "artifact": _binding(target / "research_scene.json"), "report": scene.report,
        "implementation": [_binding(Path(__file__)), _binding(Path(__file__).with_name("export.py")),
            _binding(Path(__file__).parents[1] / "stereo/model.py"),
            _binding(Path(__file__).parents[1] / "stereo/baseline/camera.py")]}
    (target / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = write_scene(load_native_scene(args.model), args.output)
    except (AdapterError, OSError, ImportError) as exc:
        print(json.dumps({"status": "ADAPTER_ERROR", "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
