"""Export a bound regional component for a separately supervised surface run.

No model fitting, mesh generation, camera changes, or implicit point repair is
performed. JPEG bytes and decoded RGB pixels have separate identities. Optional
whole-point exclusion requires an exact, caller-supplied defect allowlist.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer, not bool")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")
    value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)
    _require(isinstance(value, dict), f"object JSON required: {path}")
    return value


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _safe_name(name: Any) -> str:
    _require(type(name) is str and bool(name) and "\\" not in name and ":" not in name,
             "unsafe native image name")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and not PureWindowsPath(name).is_absolute()
             and all(part not in ("", ".", "..") for part in name.split("/"))
             and not any(ord(char) < 32 for char in name), "unsafe native image name")
    return name


def _unique(rows: Any, key: str) -> dict:
    _require(isinstance(rows, list), f"{key} rows must be a list")
    result = {}
    for row in rows:
        _require(isinstance(row, dict), f"{key} row must be an object")
        value = _integer(row.get(key), key)
        _require(value not in result, f"duplicate {key}")
        result[value] = row
    return result


def _finite(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item, context)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item, context)
    elif type(value) is float:
        _require(math.isfinite(value), f"nonfinite native field: {context}")


def _describe(model) -> dict:
    from kozlov_lab.reconstruction.export import describe_model
    value = describe_model(model)
    # The shared description covers full camera, rig, frame and observation
    # identities. Point residuals are also native fields and must be preserved.
    for pid, point in model.points3D.items():
        value["points3D"][str(pid)]["error"] = float(point.error)
    _finite(value, "model")
    return value


def _equal(left: Any, right: Any, *, tolerance: float = 0.0, path: str = "model") -> None:
    """Compare integer identities exactly; tolerance applies only to floats."""
    _require(type(left) is type(right), f"native type changed at {path}")
    if isinstance(left, dict):
        _require(set(left) == set(right), f"native keys changed at {path}")
        for key in left:
            _equal(left[key], right[key], tolerance=tolerance, path=f"{path}.{key}")
    elif isinstance(left, list):
        _require(len(left) == len(right), f"native list length changed at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, tolerance=tolerance, path=f"{path}[{index}]")
    elif type(left) is float:
        _require(math.isfinite(left) and math.isfinite(right)
                 and math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance), f"native numeric field changed at {path}")
    else:
        _require(left == right, f"native field changed at {path}")


def _raw_binary_structure(directory: Path) -> dict:
    """Check serialized reciprocity before a native reader can normalize it.

    The little-endian images/points layouts are documented by COLMAP's own
    scripts/python/read_write_model.py (3.12.6); the current format documentation
    explicitly preserves compatibility of these two files across rig versions.
    This is an independent bounded parser, not a native model repair routine.
    """
    class Reader:
        def __init__(self, path):
            self.data = path.read_bytes()
            self.offset = 0

        def take(self, format):
            size = struct.calcsize("<" + format)
            _require(size <= len(self.data) - self.offset, "truncated native binary")
            value = struct.unpack_from("<" + format, self.data, self.offset)
            self.offset += size
            return value

        def count(self, minimum_record_size):
            count, = self.take("Q")
            _require(count <= (len(self.data) - self.offset) // minimum_record_size,
                     "native binary count exceeds available bytes")
            return count

        def name(self):
            end = self.data.find(b"\x00", self.offset)
            _require(end >= self.offset, "unterminated native image name")
            value = self.data[self.offset:end].decode("utf-8")
            self.offset = end + 1
            return _safe_name(value)

        def finish(self):
            _require(self.offset == len(self.data), "trailing native binary bytes")

    reader = Reader(directory / "images.bin")
    images = {}
    for _ in range(reader.count(73)):
        header = reader.take("idddddddi")
        iid, camera_id = header[0], header[-1]
        _integer(iid, "serialized image ID")
        _integer(camera_id, "serialized camera ID")
        _require(iid not in images, "duplicate serialized image ID")
        name = reader.name()
        rows = [list(reader.take("ddQ")) for _ in range(reader.count(24))]
        images[iid] = dict(name=name, camera_id=camera_id, rows=rows)
        _finite(list(header[1:-1]) + rows, "serialized image")
    reader.finish()
    reader = Reader(directory / "points3D.bin")
    points = {}
    for _ in range(reader.count(51)):
        header = reader.take("QdddBBBd")
        pid = header[0]
        _require(pid not in points, "duplicate serialized point ID")
        track = [list(reader.take("II")) for _ in range(reader.count(8))]
        points[pid] = dict(xyz=list(header[1:4]), color=list(header[4:7]), error=header[7], track=track)
        _finite(points[pid], "serialized point")
    reader.finish()
    forward = set()
    for pid, point in points.items():
        for iid, row in point["track"]:
            _require(iid in images, f"point {pid} has unregistered view in serialized model")
            _require(row < len(images[iid]["rows"]), f"point {pid} index outside serialized image")
            _require(images[iid]["rows"][row][2] == pid, f"point {pid} broken track reciprocity in serialized model")
            forward.add((iid, row, pid))
    invalid = (1 << 64) - 1
    reverse = {(iid, row, values[2]) for iid, image in images.items()
               for row, values in enumerate(image["rows"]) if values[2] != invalid}
    _require(forward == reverse, "serialized image observation missing reverse track")
    return dict(images=images, points=points)


def _verify_raw_loaded(raw: dict, model) -> None:
    _require(set(raw["images"]) == set(model.images) and set(raw["points"]) == set(model.points3D),
             "native read changed serialized identities")
    for iid, image in raw["images"].items():
        loaded = model.images[iid]
        _require((image["name"], image["camera_id"]) == (loaded.name, loaded.camera_id),
                 "native read changed serialized image identity")
        rows = [[float(p.xy[0]), float(p.xy[1]), int(p.point3D_id)] for p in loaded.points2D]
        _equal(image["rows"], rows, path=f"serialized images.{iid}")
    for pid, point in raw["points"].items():
        loaded = model.points3D[pid]
        actual = dict(xyz=loaded.xyz.tolist(), color=loaded.color.tolist(), error=float(loaded.error),
                      track=[[int(e.image_id), int(e.point2D_idx)] for e in loaded.track.elements])
        _equal(point, actual, path=f"serialized points.{pid}")


def _inspect_structure(model) -> tuple[dict, dict[int, list[str]]]:
    """Like prepare_mvs.structure, but classify only explicit whole-point defects.

    Missing references and broken reciprocity are always fatal, including on a
    point that also has a removable duplicate-view defect.
    """
    ids = set(model.reg_image_ids())
    _require(len(ids) >= 2 and len(model.points3D) > 0, "empty/insufficient sparse geometry")
    _require(ids == set(model.images), "unregistered images in native component")
    names = [_safe_name(model.images[i].name) for i in sorted(ids)]
    _require(len(names) == len(set(names)), "duplicate native image names")
    for camera in model.cameras.values():
        _require(camera.width > 0 and camera.height > 0 and camera.verify_params(), "invalid native camera")
    for iid, image in model.images.items():
        _require(image.camera_id in model.cameras and image.frame_id in model.frames, "image camera/frame reference absent")
        frame = model.frames[image.frame_id]
        _require(frame.rig_id in model.rigs and frame.has_pose(), "frame rig/pose absent")
        rig = model.rigs[frame.rig_id]
        _require(image.data_id in frame.data_ids and image.data_id.sensor_id in rig.sensor_ids(),
                 "image frame/sensor reciprocity broken")
        _require(image.data_id.id == iid and image.data_id.sensor_id.id == image.camera_id,
                 "native image data identity mismatch")
    for fid, frame in model.frames.items():
        _require(frame.rig_id in model.rigs, "unknown frame rig")
        for data in frame.data_ids:
            _require(data.id in model.images and model.images[data.id].frame_id == fid
                     and model.images[data.id].data_id == data, "frame data/image reciprocity broken")
    defects = {}
    pairs = set()
    raw_observations = 0
    for pid, point in model.points3D.items():
        image_ids = []
        for element in point.track.elements:
            iid, row = int(element.image_id), int(element.point2D_idx)
            _require(iid in ids, f"point {pid} has unregistered view")
            _require(0 <= row < len(model.images[iid].points2D), f"point {pid} index outside image")
            _require(model.images[iid].points2D[row].point3D_id == pid, f"point {pid} broken track reciprocity")
            image_ids.append(iid)
            pairs.add((iid, row, int(pid)))
            raw_observations += 1
        reasons = []
        if len(image_ids) != len(set(image_ids)):
            reasons.append("DUPLICATE_IMAGE_VIEW")
        if len(set(image_ids)) < 2:
            reasons.append("FEWER_THAN_TWO_DISTINCT_VIEWS")
        if reasons:
            defects[int(pid)] = reasons
    reverse_pairs = set()
    for iid, image in model.images.items():
        for row, point in enumerate(image.points2D):
            if point.has_point3D():
                _require(point.point3D_id in model.points3D, "image references absent 3D point")
                reverse_pairs.add((int(iid), row, int(point.point3D_id)))
    _require(pairs == reverse_pairs, "image observation missing reverse track")
    _describe(model)
    return dict(images=len(ids), cameras=len(model.cameras), rigs=len(model.rigs), frames=len(model.frames),
                points=len(model.points3D), observations=raw_observations,
                unique_observations=len(pairs)), defects


def _derive(model, expected_excluded_point_ids):
    import pycolmap as pc
    before, defects = _inspect_structure(model)
    actual = sorted(defects)
    if expected_excluded_point_ids is None:
        _require(not actual, f"native structure requires explicit whole-point exclusion: {actual}")
        approved = []
    else:
        _require(isinstance(expected_excluded_point_ids, list), "expected_excluded_point_ids must be a list")
        approved = [_integer(pid, "excluded point ID") for pid in expected_excluded_point_ids]
        _require(len(approved) == len(set(approved)), "duplicate point exclusion IDs")
        approved = sorted(approved)
        _require(approved == actual, f"point exclusion allowlist differs from actual defects: {actual}")
    expected = _describe(model)
    child = pc.Reconstruction(model)
    removed = []
    for pid in approved:
        point = model.points3D[pid]
        removed.append(dict(point3D_id=pid, reasons=defects[pid], xyz=point.xyz.tolist(),
                            color_rgb=point.color.tolist(), error=float(point.error),
                            track=[dict(image_id=int(e.image_id), point2D_idx=int(e.point2D_idx),
                                        xy=model.images[e.image_id].points2D[e.point2D_idx].xy.tolist())
                                   for e in point.track.elements]))
        child.delete_point3D(pid)
        del expected["points3D"][str(pid)]
    expected["num_points3D"] -= len(approved)
    invalid_id = int(pc.Point2D().point3D_id)
    for image in expected["images"].values():
        image["point3D_ids"] = [invalid_id if pid in approved else pid for pid in image["point3D_ids"]]
    _equal(expected, _describe(child))
    after, remaining = _inspect_structure(child)
    _require(not remaining, "derived component still has invalid point tracks")
    derivation = dict(schema="region-surface-derivation-v1",
                      status="EXPLICIT_WHOLE_POINT_EXCLUSION" if approved else "NONE_REQUIRED",
                      expected_excluded_point_ids=expected_excluded_point_ids,
                      excluded_point_ids=approved, removed_points=removed, before=before, after=after,
                      source_modified=False, points_repaired=False, remaining_geometry_unchanged=True,
                      all_2d_coordinates_unchanged=True, cameras_poses_rigs_frames_unchanged=True,
                      only_removed_point_associations_invalidated=True, geometry_accepted=False)
    return child, derivation


def _pixels(path: Path, width: int, height: int) -> tuple[str, str]:
    from PIL import Image
    with Image.open(path) as image:
        _require(image.format in ("PNG", "JPEG") and image.mode == "RGB", "RGB PNG/JPEG image required")
        _require(image.size == (width, height), "image camera/domain size mismatch")
        _require(image.getexif().get(274, 1) == 1, "image EXIF orientation must be identity")
        return image.format, hashlib.sha256(image.tobytes()).hexdigest()


def export(source_run, output, model_id: int = 0, *, expected_excluded_point_ids=None) -> dict:
    """Export one component to a fresh directory; return the saved EXPORT record.

    All original native fields survive except safe PNG image names. An explicit
    exact defect allowlist optionally creates a separately documented reference
    model first; the default refuses all invalid tracks. ``geometry_unchanged``
    means the PNG export equals that reference; ``source_geometry_unchanged``
    records whether the reference itself excluded any whole source points.
    """
    import pycolmap as pc
    from PIL import Image
    from kozlov_lab.reconstruction import export as native_export
    _require(pc.__version__ == "4.2.1", "pycolmap 4.2.1 required")
    _integer(model_id, "model_id")
    source = Path(source_run).resolve(strict=True)
    out = Path(output).resolve()
    _require(source.is_dir() and not out.exists() and not out.is_relative_to(source)
             and not source.is_relative_to(out), "fresh disjoint output required")
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        _require(path.is_file() and not path.is_relative_to(out), "invalid source file/output overlap")
        digest = _sha(path)
        if expected is not None:
            _require(type(expected) is str and len(expected) == 64 and digest == expected, f"source binding drift: {path}")
        record = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
        _require(str(path) not in bindings or bindings[str(path)] == record, f"source changed during export: {path}")
        bindings[str(path)] = record
        return path

    geometry_plan_path = bind(source / "geometry/PLAN.json")
    result_path = bind(source / "geometry/RESULT.json")
    input_path = bind(source / "input/PLAN.json")
    geometry_plan, result, input_plan = _read(geometry_plan_path), _read(result_path), _read(input_path)
    _require(result.get("schema") == "glove-region-geometry-v1", "regional geometry result required")
    _require(input_plan.get("schema") in ("glove-region-plan-v1", "object-region-plan-v1"), "regional input plan required")
    refs = geometry_plan.get("source_bindings")
    _require(isinstance(refs, dict) and result.get("source_unchanged") is True
             and result.get("source_before") == refs and result.get("source_after") == refs,
             "geometry result/plan source binding contract differs")
    normalized_refs = {str(Path(path).resolve()): digest for path, digest in refs.items()}
    _require(normalized_refs.get(str(input_path)) == bindings[str(input_path)]["sha256"], "geometry does not bind current input PLAN")
    models = _unique(result.get("models"), "model_id")
    _require(model_id in models, "requested native model missing")
    selected = models[model_id]
    binary = Path(selected["binary_path"]).resolve(strict=True)
    _require(binary.is_dir() and binary.is_relative_to(source / "geometry"), "model binary path outside source geometry")
    for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
        bind(binary / name)
    bind(Path(__file__))
    bind(Path(native_export.__file__))
    import pycolmap._core as core
    bind(Path(core.__file__))
    raw = _raw_binary_structure(binary)
    try:
        original = pc.Reconstruction(str(binary))
    except (RuntimeError, IndexError) as exc:
        raise ValueError(f"invalid native model: {exc}") from exc
    _verify_raw_loaded(raw, original)
    reference, derivation = _derive(original, expected_excluded_point_ids)
    _require(selected.get("registered_views") == len(original.reg_image_ids())
             and selected.get("points") == len(original.points3D), "native model/result counts differ")
    frames = _unique(geometry_plan.get("frames"), "image_id")
    _unique(geometry_plan["frames"], "input_index")
    _unique(geometry_plan["frames"], "decode_index")
    timeline = _unique(input_plan.get("timeline"), "decode_index")
    endpoints = _unique([r for r in input_plan["timeline"] if r.get("input_index") is not None], "input_index")
    registered_indices = []
    records = []
    for iid in sorted(original.reg_image_ids()):
        _require(iid in frames, "native image absent from geometry frame map")
        frame, image = frames[iid], original.images[iid]
        index = _integer(frame.get("input_index"), "input_index")
        decode = _integer(frame.get("decode_index"), "decode_index")
        _require(index in endpoints and decode in timeline and endpoints[index] is timeline[decode],
                 "geometry/input frame identity differs")
        row = endpoints[index]
        _require(frame.get("name") == image.name, "native image name/frame map differs")
        camera = original.cameras[image.camera_id]
        _require((row.get("width"), row.get("height")) == (camera.width, camera.height), "input camera/domain size mismatch")
        image_source = Path(frame["image_path"]).resolve(strict=True)
        canonical = Path(row["image_path"]).resolve(strict=True)
        mask = Path(row["mask_path"]).resolve(strict=True)
        for path, expected in ((image_source, row["image_sha256"]), (canonical, row["image_sha256"]), (mask, row["mask_sha256"])):
            _require(normalized_refs.get(str(path)) == expected, f"geometry does not bind selected frame bytes: {path}")
            bind(path, expected)
        source_format, pixel_hash = _pixels(image_source, camera.width, camera.height)
        _require(_pixels(canonical, camera.width, camera.height)[1] == pixel_hash, "canonical image pixels differ")
        with Image.open(mask) as mask_image:
            _require(mask_image.mode == "L" and mask_image.size == (camera.width, camera.height), "mask domain mismatch")
            values = set(mask_image.tobytes())
            _require(values <= {0, 255} and 255 in values, "nonempty binary canonical mask required")
        pose = image.cam_from_world()
        new_name = f"image_{iid:08d}.png"
        records.append(dict(export_image_id=int(iid), image_id=int(iid), original_name=image.name,
                            exported_name=new_name, input_index=index, decode_index=decode,
                            source_image_path=str(image_source), source_image_sha256=row["image_sha256"],
                            source_image_format=source_format, source_image_bytes=image_source.stat().st_size,
                            canonical_image_path=str(canonical), canonical_image_sha256=row["image_sha256"],
                            source_mask_path=str(mask), source_mask_sha256=row["mask_sha256"],
                            canonical_mask_path=str(mask), canonical_mask_sha256=row["mask_sha256"],
                            source_camera=dict(camera_id=int(image.camera_id), model=camera.model_name,
                                               model_name=camera.model_name, width=int(camera.width), height=int(camera.height),
                                               params=camera.params.tolist()),
                            source_pose=dict(quaternion_xyzw=pose.rotation.quat.tolist(), translation=pose.translation.tolist(),
                                             rotation_matrix=pose.rotation.matrix().tolist()),
                            native_frame_id=int(image.frame_id),
                            exported_png_path=str(out / "images" / new_name),
                            decoded_pixel_sha256=pixel_hash, pixel_hash_domain="RGB uint8 row-major bytes"))
        registered_indices.append(index)
    _require(sorted(registered_indices) == selected.get("registered_input_indices"), "native/result registered input identities differ")
    expected = _describe(reference)
    exported = pc.Reconstruction(reference)
    for record in records:
        iid = record["image_id"]
        exported.images[iid].name = record["exported_name"]
        expected["images"][str(iid)]["name"] = record["exported_name"]
    _equal(expected, _describe(exported))
    # No output is created until all source identities and structure are valid.
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "binary", "text"):
        (out / folder).mkdir()
    for record in records:
        src = bind(record["source_image_path"], record["source_image_sha256"])
        dest = Path(record["exported_png_path"])
        if record["source_image_format"] == "PNG":
            shutil.copyfile(src, dest)
            record["conversion"] = "PNG_BYTE_COPY"
        else:
            with Image.open(src) as image:
                image.save(dest, format="PNG")
            record["conversion"] = "JPEG_DECODE_TO_LOSSLESS_PNG"
        output_format, pixels = _pixels(dest, record["source_camera"]["width"], record["source_camera"]["height"])
        _require(output_format == "PNG" and pixels == record["decoded_pixel_sha256"], "exported PNG pixel readback differs")
        record.update(exported_png_sha256=_sha(dest), exported_pixel_sha256=pixels, pixel_equal=True)
    exported.write_binary(str(out / "binary"))
    exported.write_text(str(out / "text"))
    comparisons = {}
    for kind, tolerance in (("binary", 0.0), ("text", 1e-12)):
        back = pc.Reconstruction(str(out / kind))
        _equal(expected, _describe(back), tolerance=tolerance)
        counts, defects = _inspect_structure(back)
        _require(not defects, "exported native structure is invalid")
        comparisons[kind] = dict(status="PASS", all_native_fields_compared=True, counts=counts,
                                 numeric_absolute_tolerance=tolerance, integer_ids_exact=True,
                                 tolerance_is_geometry_quality_threshold=False)
    for record in list(bindings.values()):
        bind(record["path"], record["sha256"])
    frame_map = out / "FRAME_MAP.json"
    _save(frame_map, dict(schema="region-surface-frame-map-v1", source_run=str(source),
                         source_model_id=model_id, frames=records))
    derivation_path = out / "DERIVATION.json"
    derivation.update(source_model_id=model_id, source_binary_path=str(binary),
                      source_model_bindings=[r for r in bindings.values() if Path(r["path"]).parent == binary])
    _save(derivation_path, derivation)
    outputs = [dict(path=str(path), sha256=_sha(path), bytes=path.stat().st_size)
               for path in sorted(out.rglob("*")) if path.is_file()]
    answer = dict(schema="region-surface-export-v1", status="EXPORTED_NOT_GEOMETRY_ACCEPTED",
                  source_run=str(source), source_model_id=model_id, source_binary_path=str(binary),
                  binary_path=str(out / "binary"), sparse=str(out / "binary"), text_path=str(out / "text"),
                  image_root=str(out / "images"), frame_map_path=str(frame_map), frame_map_sha256=_sha(frame_map),
                  derivation_path=str(derivation_path), derivation_sha256=_sha(derivation_path),
                  frames=records, registered_views=len(records), points=len(reference.points3D),
                  excluded_point_ids=derivation["excluded_point_ids"],
                  explicit_point_exclusion=bool(derivation["excluded_point_ids"]),
                  geometry_unchanged=True, geometry_unchanged_reference="explicit derived model before PNG image renaming",
                  source_geometry_unchanged=not derivation["excluded_point_ids"], source_unchanged=True,
                  source_bindings=sorted(bindings.values(), key=lambda item: item["path"]), outputs=outputs,
                  source_binding_scope="Selected native component, current geometry/input contracts, and every selected staged/canonical image and mask; unselected upstream bindings are inherited, not rehashed.",
                  native_readback=comparisons, geometry_accepted=False, quality_status="NOT_EVALUATED",
                  raw_binary_reciprocity_verified=True,
                  surface_status="NOT_RUN", scale_status="unknown", cad_status="NOT_EVALUATED",
                  model_fitting=False, native_solver_executed=False, pycolmap_version=pc.__version__)
    _save(out / "EXPORT.json", answer)
    return answer
