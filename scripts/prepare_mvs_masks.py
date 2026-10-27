"""Bounded mvs-mask-v1 producer: object-mask-v1 manifests to OpenMVS 2.4.0 masks.

Consumes an accepted COLMAP->OpenMVS bridge, the source SFM run and one or more
object-mask-v1 manifests, and writes undistorted-domain 0/255 masks plus a
separate validity raster for each bound frame. It reuses the tested geometry core
in ``integrations/mvs_masks/mask_geometry.py`` and never reimplements distortion.

It does not run SfM, densification, meshing, CAD, SAM3 or OpenMVS. Masks are
predicted labels, not human truth, and producing them proves nothing about
OpenMVS consuming them: the suggested argument array is unverified, carries
``verified: false``, and the number of actual OpenMVS invocations is always zero.

Two failure regimes are kept deliberately apart:

* a rejected input (schema, hash, identity, path, model, pose, collision, output
  already present) is decided **before** any output directory is created, so an
  early rejection never claims an output directory and never touches somebody
  else's;
* a failure **after** the directory has been claimed still writes a failure
  report with the per-frame states and the artifacts already produced, and keeps
  the first cause; a failure of that finalisation is recorded separately.
"""
from __future__ import annotations

import os

THREAD_ENV = {name: "2" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)}
os.environ.update(THREAD_ENV)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
import json
import math
import sys
import traceback
from pathlib import Path

# A direct script launch starts with scripts/ on sys.path. Derived provenance
# validators import both integrations.* and kozlov_lab.*; do not depend on the
# caller's working directory or inherited PYTHONPATH for these package roots.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "integrations")]

from mvs_masks import manifest_io as mio  # noqa: E402
from mvs_masks.mask_geometry import ALGORITHM as GEOMETRY_ALGORITHM  # noqa: E402
from mvs_masks.mask_geometry import warp_binary_mask  # noqa: E402

POSE_RTOL = POSE_ATOL = 1e-8
TILE_ROWS = 64
MAX_PIXELS = 16_777_216

BINDING_BOUNDARIES = [
    "INPUT_BINDINGS.json declares the bridge's own inputs (source images and source model files). "
    "Those entries are verified one by one against the actual bytes on disk.",
    "The bridge's own products (undistorted sparse model, undistorted images, scene.mvs) have no "
    "upstream declaration in the bridge, so they are bound with method local_hash_only: their "
    "current bytes are recorded. No historical declaration is invented for them.",
    "PLAN.json records the paths the bridge used and does not declare hashes, so those files are "
    "bound through the INPUT_BINDINGS entries plus their current bytes.",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def detect_drift(snapshot) -> list:
    """Inputs whose bytes no longer match the digest taken before the run."""
    drift = []
    for record in snapshot:
        path = Path(record["path"])
        if not path.is_file():
            drift.append({"path": str(path), "reason": "disappeared"})
        elif mio.sha256_file(path) != record["sha256"]:
            drift.append({"path": str(path), "reason": "bytes changed",
                          "declared_sha256": record["sha256"], "actual_sha256": mio.sha256_file(path)})
    return drift


def edge(source, target, *, declared=None, actual=None, method, ok, note=None) -> dict:
    return {"from": source, "to": target, "method": method, "declared": declared, "actual": actual,
            "agrees": bool(ok), "note": note}


# --------------------------------------------------------------------------- #
# bridge
# --------------------------------------------------------------------------- #
def bridge_artifacts(bridge_dir) -> dict:
    """Resolve the bridge from its own PLAN and verify every declared input byte."""
    root = Path(bridge_dir).resolve()
    if not root.is_dir():
        raise mio.InputError(f"bridge directory missing: {root}")
    for name in ("PLAN.json", "RESULT.json", "INPUT_BINDINGS.json", "UNDISTORT_COMPARISON.json", "scene.mvs"):
        if not (root / name).is_file():
            raise mio.InputError(f"bridge is incomplete, {name} missing: {root}")
    plan = mio.load_json_strict(root / "PLAN.json", "bridge PLAN")
    result = mio.load_json_strict(root / "RESULT.json", "bridge RESULT")
    declared_status = result.get("status")
    if declared_status != "PASS":
        raise mio.InputError(f"bridge RESULT.json declared status is {declared_status!r}, not PASS")
    source_sparse = Path(mio.require_str(plan.get("source_sparse"), "bridge PLAN.source_sparse")).resolve()
    source_images = Path(mio.require_str(plan.get("source_images"), "bridge PLAN.source_images")).resolve()
    if not source_sparse.is_dir() or not source_images.is_dir():
        raise mio.InputError("bridge PLAN does not resolve to existing source model and image directories")

    raw = mio.load_json_array_strict(root / "INPUT_BINDINGS.json", "bridge INPUT_BINDINGS")
    if not raw:
        raise mio.InputError("bridge INPUT_BINDINGS.json must be a non-empty array")
    edges, declared_paths = [], set()
    for position, item in enumerate(raw):
        spot = f"INPUT_BINDINGS[{position}]"
        if not isinstance(item, dict):
            raise mio.InputError(f"{spot} must be an object")
        path = mio.require_str(item.get("path"), f"{spot}.path")
        declared_bytes = mio.require_int(item.get("bytes"), f"{spot}.bytes")
        declared_sha = mio.require_sha256(item.get("sha256"), f"{spot}.sha256")
        target = Path(path)
        if not target.is_file():
            raise mio.InputError(f"{spot} declares a file that does not exist: {target}")
        if mio.normalised(target) in declared_paths:
            raise mio.InputError(f"{spot} repeats an already declared path: {target}")
        actual_bytes, actual_sha = target.stat().st_size, mio.sha256_file(target)
        if actual_bytes != declared_bytes or actual_sha != declared_sha:
            raise mio.InputError(f"{spot} does not match the bytes on disk: {target}")
        declared_paths.add(mio.normalised(target))
        edges.append(edge("bridge INPUT_BINDINGS", str(target),
                          declared={"bytes": declared_bytes, "sha256": declared_sha},
                          actual={"bytes": actual_bytes, "sha256": actual_sha}, method="sha256", ok=True))
    scene = mio.file_record(root / "scene.mvs")
    edges.append(edge("bridge scene.mvs", scene["path"], declared=None,
                      actual={"bytes": scene["bytes"], "sha256": scene["sha256"]},
                      method="local_hash_only", ok=True,
                      note="no upstream declaration exists for the bridge's own scene file"))
    return {"bridge": root, "plan": plan, "result": result, "declared_result_status": declared_status,
            "source_sparse": source_sparse, "source_images": source_images,
            "undistorted_sparse": (root / "undistorted" / "sparse").resolve(),
            "undistorted_images": (root / "undistorted" / "images").resolve(),
            "input_binding_edges": edges, "input_binding_paths": declared_paths, "scene": scene,
            "scene_sha256": scene["sha256"]}


def frame_index(bridge, sfm_run, t01_root, selected_names) -> dict:
    """Join the selected bridge images to frames_map and T01 identity.

    The consumption set is decided by the selected native bridge model, not by
    the frames_map; rows outside that set are recorded and ignored instead of
    rejecting the whole input.

    Hash identity follows HASH_IDENTITY_RULING.md: the T01 ``image_sha256`` is the
    *original* input digest and is kept as a reference, while the canonical file
    that ``image_output`` points at must be byte-identical to the staged copy and
    to ``frames_map.canonical_image_sha256``.
    """
    sfm_root = Path(sfm_run).resolve()
    if sfm_root != Path(bridge["plan"]["source_images"]).resolve():
        raise mio.InputError(
            f"--sfm-run does not match the bridge PLAN source_images: {sfm_root} vs {bridge['plan']['source_images']}"
        )
    sfm_run_json = mio.load_json_strict(sfm_root / "run.json", "sfm run.json")
    t01_run_json = mio.load_json_strict(Path(t01_root) / "run.json", "T01 run.json")
    upstream = sfm_run_json.get("upstream_io_run_id")
    if upstream != t01_run_json.get("run_id"):
        raise mio.InputError(
            f"SFM upstream_io_run_id {upstream!r} does not join the T01 run_id {t01_run_json.get('run_id')!r}"
        )
    rows = mio.read_jsonl_strict(sfm_root / "frames_map.jsonl", "sfm frames_map.jsonl")
    t01_rows = mio.read_jsonl_strict(Path(t01_root) / "frames.jsonl", "T01 frames.jsonl")
    t01_by_key = {}
    for row in t01_rows:
        mio.require_int(row.get("input_index"), "T01.input_index")
        mio.require_str(row.get("frame_id"), "T01.frame_id")
        key = (row.get("input_index"), row.get("frame_id"))
        if key in t01_by_key:
            raise mio.InputError(f"T01 frames.jsonl repeats input_index/frame_id {key}")
        t01_by_key[key] = row
    by_staged = {}
    excluded_foreground = []
    map_keys = set()
    foreground = sfm_run_json.get('foreground') or {}
    object_first_verified = (isinstance(foreground,dict)
        and foreground.get('contract_version') == 'sfm-foreground-v1'
        and foreground.get('requested') is True
        and foreground.get('mask_consumption_verified') is True)
    for row in rows:
        mio.require_int(row.get("input_index"), "frames_map.input_index")
        mio.require_str(row.get("frame_id"), "frames_map.frame_id")
        key = (row['input_index'],row['frame_id'])
        if key in map_keys:
            raise mio.InputError(f'frames_map repeats input_index/frame_id {key}')
        map_keys.add(key)
        t01 = t01_by_key.get(key)
        if t01 is None or row.get('upstream_io_run_id') != upstream or row.get('upstream_image_output') != t01.get('image_output'):
            raise mio.InputError(f'frames_map row {key} has a conflicting or missing T01 identity')
        if row.get('staged_image_name') is None:
            # An excluded object-first row is still an original input identity,
            # not a staged/native image and never a missing-mask fallback.
            if not (object_first_verified
                and all(field in row for field in ('staged_image_name','foreground_selected','database_image_id','representative_input_index','registrations'))
                and row.get('status') == 'EXCLUDED_FOREGROUND'
                and row.get('foreground_selected') is False
                and row.get('database_image_id') is None
                and row.get('representative_input_index') is None
                and row.get('registrations') == []):
                raise mio.InputError(f'frames_map row {key} has an invalid unstaged foreground exclusion')
            declared = mio.require_sha256(row.get('canonical_image_sha256'),f'frames_map[{key}].canonical_image_sha256')
            relative = mio.require_str(t01.get('image_output'),f'T01[{key}].image_output')
            canonical = mio.ensure_inside(Path(t01_root),Path(t01_root)/relative,f'T01[{key}].image_output')
            if t01.get('status') != 'OK' or mio.sha256_file(canonical) != declared:
                raise mio.InputError(f'excluded foreground row {key} canonical identity differs')
            excluded_foreground.append(dict(input_index=row['input_index'],frame_id=row['frame_id'],
                status=row['status'],reason=row.get('reason'),canonical_image_sha256=declared))
            continue
        name = mio.require_str(row.get("staged_image_name"), "frames_map.staged_image_name")
        if row.get('status') == 'EXCLUDED_FOREGROUND':
            raise mio.InputError(f'excluded foreground row {key} cannot name a staged image')
        if name in by_staged:
            raise mio.InputError(f"frames_map repeats staged_image_name {name!r}")
        by_staged[name] = row
    if map_keys != set(t01_by_key):
        raise mio.InputError('frames_map and T01 original row identity coverage differs')
    selected = list(selected_names)
    missing = [n for n in selected if n not in by_staged]
    if missing:
        raise mio.InputError(f"selected bridge images missing from frames_map: {sorted(missing)[:5]}")
    index, edges = {}, []
    for name in selected:
        row = by_staged[name]
        key = (row.get("input_index"), row.get("frame_id"))
        t01 = t01_by_key.get(key)
        if t01 is None:
            raise mio.InputError(f"frames_map row {key} has no matching T01 line")
        if row.get("upstream_io_run_id") != upstream or row.get("upstream_image_output") != t01.get("image_output"):
            raise mio.InputError(f"frames_map row {key} has a conflicting upstream identity")
        declared_canonical = mio.require_sha256(row.get("canonical_image_sha256"),
                                               f"frames_map[{key}].canonical_image_sha256")
        staged = mio.ensure_inside(sfm_root, sfm_root / name, f"frames_map[{key}].staged_image_name")
        if mio.normalised(staged) not in bridge["input_binding_paths"]:
            raise mio.InputError(f"bridge INPUT_BINDINGS does not declare the staged image {name!r}")
        staged_record = mio.file_record(staged)
        if staged_record["sha256"] != declared_canonical:
            raise mio.InputError(f"staged image {name!r} does not match frames_map.canonical_image_sha256")
        if t01.get("status") != "OK":
            raise mio.InputError(f"T01 line {key} status is {t01.get('status')!r}, not OK")
        relative = t01.get("image_output")
        if not isinstance(relative, str) or not relative:
            raise mio.InputError(f"T01 line {key} has no image_output")
        canonical_image = mio.ensure_inside(Path(t01_root), Path(t01_root) / relative, f"T01[{key}].image_output")
        canonical_record = mio.file_record(canonical_image)
        if canonical_record["sha256"] != declared_canonical:
            raise mio.InputError(
                f"T01 canonical image {relative!r} is not byte-identical to the staged copy "
                f"(frames_map.canonical_image_sha256)"
            )
        edges.append(edge("staged actual bytes", str(staged),
                          declared={"sha256": declared_canonical},
                          actual={"sha256": staged_record["sha256"], "bytes": staged_record["bytes"]},
                          method="sha256", ok=True))
        edges.append(edge("T01 image_output actual bytes", str(canonical_image),
                          declared={"sha256": declared_canonical},
                          actual={"sha256": canonical_record["sha256"], "bytes": canonical_record["bytes"]},
                          method="sha256", ok=True,
                          note="canonical and staged must be exact byte copies; the original digest is a separate reference"))
        for key_name, expected in (("image_dtype", "uint8"), ("image_mode", "RGB")):
            if key_name in t01 and t01.get(key_name) != expected:
                raise mio.InputError(f"T01 line {key} {key_name} is {t01.get(key_name)!r}, expected {expected!r}")
        width, height = t01.get("canonical_width"), t01.get("canonical_height")
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise mio.InputError(f"T01 line {key} has no usable canonical size")
        verify_canonical_image(canonical_image, width, height)
        original = mio.require_sha256(t01.get("image_sha256"), f"T01[{key}].image_sha256")
        index[name] = {
            "input_index": row.get("input_index"), "frame_id": row.get("frame_id"),
            "staged_path": str(staged), "staged_sha256": staged_record["sha256"],
            "canonical_image_sha256": declared_canonical,
            "canonical_image_path": str(canonical_image),
            "canonical_actual_sha256": canonical_record["sha256"],
            "original_image_sha256": original,
            "original_hash_is_a_reference": True,
            "canonical_width": width, "canonical_height": height,
        }
    unselected = sorted(set(by_staged) - set(selected))
    unregistered = sorted({r.get("input_index") for r in rows if r.get("status") != "REGISTERED"})
    return {"sfm_root": sfm_root, "t01_root": Path(t01_root).resolve(), "by_name": index,
            "run_json": sfm_run_json, "t01_run_json": t01_run_json, "edges": edges,
            "frames_map_rows": len(rows), "selected_rows": len(selected),
            "unselected_staged_names": unselected, "unregistered_input_indices": unregistered,
            'excluded_foreground_rows':excluded_foreground,
            "by_staged": by_staged, "t01_by_key": t01_by_key,
            "map_by_key": {(r["input_index"], r["frame_id"]): r for r in rows}}


def verify_canonical_image(path, width, height):
    """Read canonical RGB8 pixels, not just a declared size or a file digest."""
    from PIL import Image
    try:
        with open(path, "rb") as stream:
            header = stream.read(26)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) < 26 or header[24:26] != b"\x08\x02":
            raise mio.InputError(f"canonical image must have RGB8 PNG encoding: {path}")
        with Image.open(path) as image:
            image.load()
            if (image.format != "PNG" or image.mode != "RGB" or image.size != (width, height)
                    or getattr(image, "n_frames", 1) != 1 or "transparency" in image.info
                    or image.getexif().get(274, 1) not in (None, 1)):
                raise mio.InputError(f"canonical image must be RGB8 PNG at {(width, height)}: {path}")
    except mio.InputError:
        raise
    except Exception as exc:
        raise mio.InputError(f"canonical image cannot be decoded: {path}: {exc}") from exc


def bind_frames(bridge_names, index, manifests, *, validate_all_sources=False, certified_excluded=None) -> dict:
    """Attach at most one usable mask to each bridge image, refusing duplicates."""
    by_key = {(entry["input_index"], entry["frame_id"]): name for name, entry in index["by_name"].items()}
    bound, claimed, seen_frames, source_io_records, object_ids = {}, {}, {}, {}, set()
    problems, edges, ignored, ignored_files = [], [], [], []
    for manifest in manifests:
        key_io = str(manifest["source_io"].get("path"))
        source_io_records.setdefault(key_io, mio.resolve_source_io(manifest["source_io"], manifest["manifest_path"]))
        object_ids.add(manifest["object_id"])
        for frame in manifest["frames"]:
            lookup = (frame["input_index"], frame["frame_id"])
            where = f"{manifest['manifest_path']} input_index={frame['input_index']} frame_id={frame['frame_id']}"
            if lookup in seen_frames:
                raise mio.InputError(f"image appears in more than one manifest frame: {where} and {seen_frames[lookup]}")
            seen_frames[lookup] = where
            name = by_key.get(lookup)
            if name is None:
                upstream = index["t01_by_key"].get(lookup)
                mapping = index["map_by_key"].get(lookup)
                certified = validate_all_sources and lookup in (certified_excluded or set())
                if (upstream is None or mapping is None or (not certified and (frame["status"] == "OK"
                        or frame["candidates"] or frame["chosen"] is not None))):
                    raise mio.InputError(f"{where}: nonselected frame is not an identified frame without a mask")
                if (mapping.get("upstream_io_run_id") != index["t01_run_json"]["run_id"]
                        or mapping.get("upstream_image_output") != upstream.get("image_output")
                        or frame["photo_io_image_sha256"] != upstream.get("image_sha256")
                        or (frame["width"], frame["height"]) !=
                        (upstream.get("canonical_width"), upstream.get("canonical_height"))):
                    raise mio.InputError(f"{where}: nonselected frame has conflicting upstream identity")
                canonical = mio.ensure_inside(index["t01_root"], index["t01_root"] / upstream["image_output"])
                if frame["source_image"] is not None:
                    source = mio.ensure_inside(index["t01_root"], index["t01_root"] / frame["source_image"])
                    if mio.normalised(source) != mio.normalised(canonical):
                        raise mio.InputError(f"{where}: nonselected source_image differs from T01 image_output")
                    actual = mio.file_record(source)
                    if (actual["sha256"] != frame["source_image_sha256"]
                            or actual["sha256"] != mapping.get("canonical_image_sha256")):
                        raise mio.InputError(f"{where}: nonselected canonical hash conflict")
                    verify_canonical_image(source, frame["width"], frame["height"])
                    ignored_files.append(actual)
                ignored.append({"input_index": frame["input_index"], "frame_id": frame["frame_id"],
                                "status": frame["status"], "reason": "outside selected model; upstream identity checked"})
                continue
            entry = index["by_name"][name]
            if (frame["width"], frame["height"]) != (entry["canonical_width"], entry["canonical_height"]):
                raise mio.InputError(
                    f"{where}: declared size {(frame['width'], frame['height'])} is not the T01 canonical size "
                    f"{(entry['canonical_width'], entry['canonical_height'])}"
                )
            if frame["photo_io_image_sha256"] != entry["original_image_sha256"]:
                raise mio.InputError(
                    f"{where}: photo_io_image_sha256 does not equal the T01 original image_sha256 reference"
                )
            if frame["status"] != "OK" and not validate_all_sources:
                problems.append({"frame": where, "reason": f"source frame status {frame['status']}"})
                continue
            if name in claimed:
                raise mio.InputError(f"bridge image {name!r} already carries a mask from {claimed[name]}")
            chosen = frame["chosen"]
            source_image = frame["source_image"]
            candidate_path = Path(source_image)
            if not candidate_path.is_absolute():
                candidate_path = index["t01_root"] / candidate_path
            resolved_source = mio.ensure_inside(index["t01_root"], candidate_path, f"{where}.source_image")
            if mio.normalised(resolved_source) != mio.normalised(entry["canonical_image_path"]):
                raise mio.InputError(
                    f"{where}: source_image must be the T01 line image_output for this frame, got {source_image!r}"
                )
            actual = mio.file_record(resolved_source)
            if frame["source_image_sha256"] != actual["sha256"]:
                raise mio.InputError(
                    f"{where}: source_image_sha256 does not equal the actual canonical bytes"
                )
            if actual["sha256"] != entry["canonical_actual_sha256"]:
                raise mio.InputError(f"{where}: canonical bytes are not the staged canonical bytes")
            edges.append(edge("manifest source_image_sha256", str(resolved_source),
                              declared={"sha256": frame["source_image_sha256"]},
                              actual={"sha256": actual["sha256"], "bytes": actual["bytes"]},
                              method="sha256", ok=True))
            if frame["status"] != "OK":
                verify_canonical_image(resolved_source, frame["width"], frame["height"])
                ignored_files.append(actual)
                problems.append({"frame": where, "reason": f"source frame status {frame['status']}"})
                continue
            array, record = mio.read_label_png(chosen["resolved_mask_path"], entry["canonical_width"],
                                               entry["canonical_height"])
            if record["sha256"] != chosen["mask_sha256"]:
                raise mio.InputError(f"{where}: mask bytes are not the declared selected sha256")
            if record["foreground_pixels"] == 0:
                raise mio.InputError(f"{where}: selected mask has no foreground pixel")
            if record["foreground_pixels"] != chosen["foreground_pixels"]:
                raise mio.InputError(f"{where}: selected foreground_pixels does not match actual mask pixels")
            edges.append(edge("manifest selected_mask.sha256", chosen["resolved_mask_path"],
                              declared={"sha256": chosen["mask_sha256"]},
                              actual={"sha256": record["sha256"], "bytes": record["bytes"]}, method="sha256", ok=True))
            claimed[name] = where
            bound[name] = {"frame": frame, "entry": entry, "manifest_path": manifest["manifest_path"],
                           "object_id": manifest["object_id"], "where": where, "array": array, "source": record}
    if len(object_ids) > 1:
        raise mio.InputError(f"manifests declare different object_id values: {sorted(object_ids)}")
    return {"bound": bound, "problems": problems, "source_io": source_io_records,
            "object_ids": sorted(object_ids), "edges": edges,
            "ignored_nonselected": ignored, "ignored_files": ignored_files}


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def scene_binding(bridge, records) -> dict:
    """Verify the MVSI scene content against the selected images, reusing read_scene.

    The scene path is resolved relative to the bridge root that owns it. A
    placeholder or unparseable file is rejected; the parser is reused, never
    rewritten, and no OpenMVS code runs.
    """
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
    from mvs_surface.scene import SceneFormatError, read_scene

    path = (bridge["bridge"] / "scene.mvs").resolve()
    if not path.is_file():
        raise mio.InputError(f"bridge scene missing: {path}")
    try:
        scene = read_scene(path)
    except SceneFormatError as exc:
        raise mio.InputError(f"scene.mvs is not a usable MVSI v7 scene: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise mio.InputError(f"scene.mvs cannot be read: {type(exc).__name__}: {exc}") from exc
    images = scene["images"]
    selected = sorted(records)
    if len(images) != len(selected):
        raise mio.InputError(
            f"scene image count {len(images)} does not match the selected model's {len(selected)} images"
        )
    # Same explicit name rule as mvs_surface.photo_inputs: InterfaceCOLMAP's
    # declared image folder relative to this bridge, plus the full native name.
    scene_by_name, mapping = {}, []
    for image in images:
        scene_by_name[image["name"]] = image
    prefix = bridge["undistorted_images"].relative_to(bridge["bridge"]).as_posix()
    expected_names = {prefix + "/" + name.replace("\\", "/"): name for name in selected}
    if set(scene_by_name) != set(expected_names):
        missing = sorted(set(expected_names) - set(scene_by_name))[:5]
        extra = sorted(set(scene_by_name) - set(expected_names))[:5]
        raise mio.InputError(f"scene images do not correspond to the selected model: missing {missing} extra {extra}")
    for scene_name, image in sorted(scene_by_name.items()):
        name = expected_names[scene_name]
        record = records[name]
        camera = record["target_camera"]
        actual_path = mio.ensure_inside(bridge["bridge"], bridge["bridge"] / scene_name, "scene image")
        if mio.normalised(actual_path) != mio.normalised(record["undistorted_image"]["path"]):
            raise mio.InputError(f"scene image path differs from the native target: {scene_name}")
        if image["global_id"] != record["target_image_id"] or image["global_id"] != record["source_image_id"]:
            raise mio.InputError(f"scene global/native image identity mismatch: {scene_name}")
        if (image["width"], image["height"]) != (camera["width"], camera["height"]):
            raise mio.InputError(
                f"scene image {image['name']!r} is {(image['width'], image['height'])}, "
                f"the target camera is {(camera['width'], camera['height'])}"
            )
        # InterfaceCOLMAP v2.4.0 converts COLMAP pixel centers by subtracting
        # 0.5 from the two principal-point entries; no offset is applied to warp.
        expected_k = np.asarray(record["target_K"], dtype=float).copy()
        expected_k[0, 2] -= 0.5
        expected_k[1, 2] -= 0.5
        gaps = {}
        for field, expected, tolerance in (("K", expected_k, 1e-7),
                                            ("R", record["target_R"], 1e-8),
                                            ("C", record["target_C"], 1e-8)):
            actual = np.asarray(image[field], dtype=float)
            expected = np.asarray(expected, dtype=float)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise mio.InputError(f"scene {field} shape/nonfinite mismatch: {scene_name}")
            gaps[field] = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=0 if field == "K" else 1e-8, atol=tolerance):
                raise mio.InputError(f"scene {field} mismatch for {scene_name}: {gaps[field]}")
        mapping.append({"model_image": name, "scene_name": image["name"], "scene_name_raw": image["name_raw"],
                        "scene_global_id": image["global_id"], "scene_platform_id": image["platform_id"],
                        "scene_camera_id": image["camera_id"], "scene_pose_id": image["pose_id"],
                        "native_image_id": record["target_image_id"], "actual_image": str(actual_path),
                        "K": image["K"], "R": image["R"], "C": image["C"], "max_abs_difference": gaps,
                        "width": image["width"], "height": image["height"]})
    return {"path": str(path), "bytes": scene["bytes"], "sha256": scene["sha256"], "version": scene["version"],
            "image_count": len(images), "mapping": mapping, "verified": True,
            "edges": [edge("bridge scene.mvs images", str(path),
                           declared={"count": len(selected), "names": "full native names under bridge undistorted/images"},
                           actual={"count": len(images), "names": [i["name"] for i in images[:3]]},
                           method="mvsi7_exact_path_global_id_size_K_R_C", ok=True,
                           note="local bytes plus parsed MVSI v7 content; no upstream product declaration exists")]}


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def model_bindings(bridge) -> dict:
    """Connect names, ids, cameras, poses and bytes of both models and the images."""
    import numpy as np
    import pycolmap
    from PIL import Image

    source_model = pycolmap.Reconstruction(str(bridge["source_sparse"]))
    target_model = pycolmap.Reconstruction(str(bridge["undistorted_sparse"]))
    source_names = {source_model.images[i].name for i in source_model.reg_image_ids()}
    target_names = {target_model.images[i].name for i in target_model.reg_image_ids()}
    if source_names != target_names:
        raise mio.InputError(
            f"image name sets differ between source and undistorted models: "
            f"missing {sorted(source_names - target_names)[:5]} extra {sorted(target_names - source_names)[:5]}"
        )
    if not source_names:
        raise mio.InputError("the selected native model registers no image")
    records, worst = {}, {"rotation": 0.0, "translation": 0.0, "center": 0.0}
    for iid in sorted(source_model.reg_image_ids()):
        name = source_model.images[iid].name
        source_image = source_model.images[iid]
        target_image = next(target_model.images[j] for j in target_model.reg_image_ids()
                            if target_model.images[j].name == name)
        source_camera = source_model.cameras[source_image.camera_id]
        target_camera = target_model.cameras[target_image.camera_id]
        if target_camera.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise mio.InputError(f"target camera for {name!r} is not undistorted pinhole: {target_camera.model_name}")
        for field, left, right in (
            ("rotation", source_image.cam_from_world().rotation.matrix(), target_image.cam_from_world().rotation.matrix()),
            ("translation", source_image.cam_from_world().translation, target_image.cam_from_world().translation),
            ("center", source_image.projection_center(), target_image.projection_center()),
        ):
            gap = float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))
            if not math.isfinite(gap) or not np.allclose(left, right, rtol=POSE_RTOL, atol=POSE_ATOL):
                raise mio.InputError(f"pose mismatch for {name!r}: {field} differs by {gap}")
            worst[field] = max(worst[field], gap)
        undistorted_image = mio.ensure_inside(bridge["undistorted_images"],
                                              bridge["undistorted_images"] / name, "native target image")
        if not undistorted_image.is_file():
            raise mio.InputError(f"undistorted image missing for {name!r}: {undistorted_image}")
        with Image.open(undistorted_image) as handle:
            size = handle.size
        if size != (int(target_camera.width), int(target_camera.height)):
            raise mio.InputError(
                f"undistorted image {name!r} is {size}, target camera is "
                f"{(int(target_camera.width), int(target_camera.height))}"
            )
        records[name] = {
            "source_image_id": int(iid), "target_image_id": int(target_image.image_id),
            "source_camera_id": int(source_image.camera_id), "target_camera_id": int(target_image.camera_id),
            "source_camera": {"model": source_camera.model_name, "width": int(source_camera.width),
                              "height": int(source_camera.height),
                              "params": [float(v) for v in source_camera.params]},
            "target_camera": {"model": target_camera.model_name, "width": int(target_camera.width),
                              "height": int(target_camera.height),
                              "params": [float(v) for v in target_camera.params]},
            "pose_max_abs": {key: worst[key] for key in ("rotation", "translation", "center")},
            "target_K": np.asarray(target_camera.calibration_matrix()).tolist(),
            "target_R": np.asarray(target_image.cam_from_world().rotation.matrix()).tolist(),
            "target_C": np.asarray(target_image.projection_center()).tolist(),
            "undistorted_image": mio.file_record(undistorted_image),
            "source_model_class": source_model.__class__.__name__,
        }
    return {"records": records, "source_model": source_model, "target_model": target_model}


# --------------------------------------------------------------------------- #
# preflight / execute
# --------------------------------------------------------------------------- #
def preflight(args) -> dict:
    bridge = bridge_artifacts(args.bridge)
    parsed = [mio.parse_object_manifest(p) for p in args.mask_manifest]
    resolved_io = mio.resolve_source_io(parsed[0]["source_io"], parsed[0]["manifest_path"])
    for other in parsed[1:]:
        if mio.normalised(other["source_io"]["path"]) != mio.normalised(parsed[0]["source_io"]["path"]):
            raise mio.InputError("manifests point at different source_io.path values")
    undeclared = [p.name for p in sorted(bridge["source_sparse"].iterdir())
                  if p.is_file() and mio.normalised(p) not in bridge["input_binding_paths"]]
    if undeclared:
        raise mio.InputError(f"bridge INPUT_BINDINGS does not declare the source model files {undeclared}")
    models = model_bindings(bridge)
    bridge_names = sorted(models["records"])
    scene = scene_binding(bridge, models["records"])
    index = frame_index(bridge, args.sfm_run, resolved_io["path"], bridge_names)
    for name, entry in index["by_name"].items():
        camera = models["records"][name]["source_camera"]
        if (entry["canonical_width"], entry["canonical_height"]) != (camera["width"], camera["height"]):
            raise mio.InputError(f"canonical dimensions do not match the source camera: {name}")
    plan = mio.output_plan(bridge_names)
    selection = None
    if getattr(args, "selection_ledger", None):
        from mvs_masks.selection import select_sources
        selection = select_sources(args.selection_ledger, parsed, index, derived_model=bridge['source_sparse'])
        # Validate every original row through the existing identity/pixel checks,
        # including sources the ledger does not choose. Duplicates are allowed
        # only across separately validated manifests, never within one source.
        source_checks = [bind_frames(bridge_names, index, [m], validate_all_sources=True,
                                    certified_excluded=selection.get('certified_excluded')) for m in parsed]
        binding = bind_frames(bridge_names, index, selection["effective_manifests"])
        binding["edges"].extend(item for check in source_checks for item in check["edges"])
        binding["ignored_files"].extend(item for check in source_checks for item in check["ignored_files"])
    else:
        binding = bind_frames(bridge_names, index, parsed)

    snapshot = []
    for path in ("PLAN.json", "RESULT.json", "INPUT_BINDINGS.json", "UNDISTORT_COMPARISON.json"):
        snapshot.append(mio.file_record(bridge["bridge"] / path))
    snapshot.append({"path": bridge["scene"]["path"], "bytes": bridge["scene"]["bytes"],
                     "sha256": bridge["scene"]["sha256"]})
    for sparse in (bridge["source_sparse"], bridge["undistorted_sparse"]):
        for path in sorted(p for p in sparse.iterdir() if p.is_file()):
            snapshot.append(mio.file_record(path))
    for rel in ("run.json", "frames_map.jsonl"):
        snapshot.append(mio.file_record(index["sfm_root"] / rel))
    for rel in ("run.json", "frames.jsonl"):
        snapshot.append(mio.file_record(index["t01_root"] / rel))
    for manifest in parsed:
        snapshot.append({"path": manifest["manifest_path"], "bytes": manifest["manifest_bytes"],
                         "sha256": manifest["manifest_sha256"]})
    for name in sorted(index["by_name"]):
        entry = index["by_name"][name]
        snapshot.append({"path": entry["staged_path"], "sha256": entry["staged_sha256"]})
        snapshot.append({"path": entry["canonical_image_path"], "sha256": entry["canonical_actual_sha256"]})
    for name in sorted(binding["bound"]):
        snapshot.append(mio.file_record(binding["bound"][name]["frame"]["chosen"]["resolved_mask_path"]))
    for record in models["records"].values():
        snapshot.append(record["undistorted_image"])
    snapshot.extend(binding["ignored_files"])
    if selection is not None:
        snapshot.extend(selection["bindings"])
    return {"bridge": bridge, "manifests": parsed, "source_io": resolved_io, "index": index,
            "models": models, "scene": scene, "bridge_names": bridge_names, "output_plan": plan,
            "binding": binding, "snapshot": snapshot, "selection": selection}


def suggested_arguments(root, *, verified: bool, reason: str) -> dict:
    if not verified:
        return {"available": False, "reason": reason, "actual_mvs_invocations": 0}
    return {"available": True, "verified": False,
            "argv_fragment": ["--mask-path", str(Path(root).resolve() / "masks"), "--ignore-mask-label", "0"],
            "note": "Unverified downstream argument fragment. mask_path still refers to the undistorted model's "
                    "image names; the C task must confirm the native consumer really ignores label 0.",
            "actual_mvs_invocations": 0}


def execute(args, plan, root, state) -> dict:
    import numpy as np

    bridge, index, binding, models = plan["bridge"], plan["index"], plan["binding"], plan["models"]
    bridge_names, output_plan, records = plan["bridge_names"], plan["output_plan"], models["records"]
    source_model, target_model = models["source_model"], models["target_model"]
    missing = sorted(set(bridge_names) - set(binding["bound"]))

    frames, written = state["frames"], state["written"]
    outputs_verified = True
    for name in bridge_names:
        frames.append(_frame_shell(index, records, output_plan, name, status="NOT_RUN",
                                   reason="not started"))
    slot = {frame["image_name"]: frame for frame in frames}
    if missing and not args.allow_partial:
        strict_reason = (f"strict mode: {len(missing)} of {len(bridge_names)} bridge images have no usable mask "
                         f"(for example {missing[:3]}); rerun with --allow-partial to keep what exists")
        for name in bridge_names:
            slot[name].update(status="NOT_RUN" if name in binding["bound"] else "MISSING",
                              reason=strict_reason)
        counts = mio.counts_from_frames(frames)
        drift = detect_drift(plan["snapshot"])
        manifest = _assemble(args, plan, root, frames, counts, written, input_stable=not drift,
                             outputs_verified=False, drift=drift, strict_reason=strict_reason)
        save_json(root / "manifest.json", manifest)
        return manifest
    for name in bridge_names:
        item = binding["bound"].get(name)
        if item is None:
            slot[name].update(status="MISSING",
                              reason="no usable object-mask-v1 frame for this bridge image")
            continue
        try:
            frame = _warp_one(name, item, index, records, source_model, target_model, output_plan, root, written)
        except Exception as exc:
            slot[name].update(status="INPUT_ERROR", reason=f"{type(exc).__name__}: {exc}",
                              failed_during_execution=True, source_mask=item["source"],
                              source_candidate_id=item["frame"]["selected_candidate_id"])
            raise
        slot[name].update(frame)
        if frame["status"] == "OK":
                outputs_verified &= bool(frame.get("readback_verified"))
        else:
            outputs_verified = False
    counts = mio.counts_from_frames(frames)
    drift = detect_drift(plan["snapshot"])
    manifest = _assemble(args, plan, root, frames, counts, written,
                         input_stable=not drift, outputs_verified=outputs_verified, drift=drift)
    save_json(root / "manifest.json", manifest)
    return manifest


def _frame_shell(index, records, output_plan, name, *, status, reason):
    entry, camera = index["by_name"][name], records[name]
    return {"input_index": entry["input_index"], "frame_id": entry["frame_id"], "image_name": name,
            "target_mask": output_plan["by_image"][name], "status": status, "reason": reason,
            "source_candidate_id": None, "source_mask": None, "source_image": None,
            "source_image_sha256": None, "photo_io_image_sha256": None,
            "source_camera": camera["source_camera"], "target_camera": camera["target_camera"],
            "pose_max_abs": camera["pose_max_abs"],
            "target_size": [camera["target_camera"]["width"], camera["target_camera"]["height"]],
            "mask": None, "validity": None, "core_counts": None, "readback_verified": False}


def _warp_one(name, item, index, records, source_model, target_model, output_plan, root, written):
    import numpy as np

    frame, entry = item["frame"], item["entry"]
    camera = records[name]
    source_iid = camera["source_image_id"]
    source_camera = source_model.cameras[source_model.images[source_iid].camera_id]
    target_image = next(target_model.images[j] for j in target_model.reg_image_ids()
                        if target_model.images[j].name == name)
    target_camera = target_model.cameras[target_image.camera_id]
    record = {
        "input_index": entry["input_index"], "frame_id": entry["frame_id"], "image_name": name,
        "target_mask": output_plan["by_image"][name], "source_candidate_id": frame["selected_candidate_id"],
        "selection": frame["selection"], "source_mask": item["source"],
        "source_image": frame["source_image"], "source_image_sha256": frame["source_image_sha256"],
        "photo_io_image_sha256": frame["photo_io_image_sha256"],
        "source_camera": camera["source_camera"], "target_camera": camera["target_camera"],
        "pose_max_abs": camera["pose_max_abs"],
        "target_size": [camera["target_camera"]["width"], camera["target_camera"]["height"]],
    }
    warped = warp_binary_mask(item["array"], source_camera, target_camera, tile_rows=TILE_ROWS, max_pixels=MAX_PIXELS)
    if warped.counts["foreground"] == 0:
        record.update({"status": "EMPTY_AFTER_WARP", "reason": "empty foreground after warp",
                       "core_counts": warped.counts, "mask": None, "validity": None, "readback_verified": False})
        return record
    stem = output_plan["by_image"][name][: -len(mio.MASK_SUFFIX)]
    mask_rel, validity_rel = f"masks/{stem}{mio.MASK_SUFFIX}", f"validity/{stem}{mio.VALIDITY_SUFFIX}"
    width, height = camera["target_camera"]["width"], camera["target_camera"]["height"]
    mask_written = mio.write_label_png(root / mask_rel, warped.mask)
    written.append(mask_rel)
    validity_written = mio.write_label_png(root / validity_rel, warped.valid)
    written.append(validity_rel)
    mask_read = mio.verify_label_png(root / mask_rel, warped.mask, width, height)
    validity_read = mio.verify_label_png(root / validity_rel, warped.valid, width, height)
    if mask_read["sha256"] != mask_written["sha256"] or validity_read["sha256"] != validity_written["sha256"]:
        raise mio.ProductError(f"read-back digest differs from the written digest for {name!r}")
    unknown = int(np.count_nonzero(np.asarray(warped.valid, dtype=np.uint8) == 0))
    if unknown != warped.counts["unknown"]:
        raise mio.ProductError(f"validity raster does not encode the unknown domain for {name!r}")
    if np.any((np.asarray(warped.valid, dtype=np.uint8) == 0) & (np.asarray(warped.mask, dtype=np.uint8) != 0)):
        raise mio.ProductError(f"foreground sits on an unknown pixel for {name!r}")
    record.update({"status": "OK", "reason": None, "core_counts": warped.counts,
                   "mask": {"path": mask_rel, "bytes": mask_read["bytes"], "sha256": mask_read["sha256"],
                            "labels": mask_read["labels"], "absolute_path": mask_read["path"]},
                   "validity": {"path": validity_rel, "bytes": validity_read["bytes"],
                                "sha256": validity_read["sha256"], "labels": validity_read["labels"],
                                "unknown_pixels": unknown, "absolute_path": validity_read["path"]},
                   "readback_verified": True})
    return record


def _assemble(args, plan, root, frames, counts, written, *, input_stable, outputs_verified,
              drift=None, strict_reason=None) -> dict:
    bridge, binding, index = plan["bridge"], plan["binding"], plan["index"]
    coverage = mio.coverage_status(counts)
    status, _ = mio.classify_run(counts, allow_partial=args.allow_partial)
    if drift:
        status = "INPUT_ERROR"
    if strict_reason:
        status = "INPUT_ERROR"
    edges = bridge["input_binding_edges"] + index["edges"] + binding["edges"] + plan["scene"]["edges"]
    unverified = [{"edge": item["from"], "to": item["to"], "note": item["note"]}
                  for item in edges if not item["agrees"]]
    if unverified or not plan["scene"]["verified"]:
        status = "INPUT_ERROR"
    verified = (status == "SUCCESS" and coverage == "COMPLETE" and input_stable and outputs_verified
                and counts["error"] == 0 and not binding["problems"]
                and plan["scene"]["verified"] and not unverified)
    if verified:
        reason = ""
    elif status != "SUCCESS":
        reason = f"run status is {status}, not SUCCESS"
    elif coverage != "COMPLETE":
        reason = "coverage is not COMPLETE"
    elif not input_stable:
        reason = "inputs drifted during the run"
    elif not outputs_verified:
        reason = "output read-back did not verify for every frame"
    else:
        reason = "unbound object-mask frames remain"
    result = {
        "contract_version": mio.CONTRACT_OUT,
        "run_id": root.name,
        "started_utc": args._started_utc,
        "finished_utc": utcnow(),
        "status": status,
        "image_domain": mio.IMAGE_DOMAIN,
        "source_image_domain": mio.SOURCE_IMAGE_DOMAIN,
        "object_id": binding["object_ids"][0] if binding["object_ids"] else None,
        "evidence_level": ("synthetic_interface_fixture"
                           if args._fixture or any("synthetic" in str(m.get("evidence_level", "")).casefold()
                                                   for m in plan["manifests"]) else "real_object_mask"),
        "object_mask_manifests": [{"path": m["manifest_path"], "sha256": m["manifest_sha256"],
                                  "bytes": m["manifest_bytes"], "run_id": m["run_id"], "status": m["status"],
                                  "object_id": m["object_id"], "quality_status": m["quality_status"],
                                  "cross_view_status": m["cross_view_status"],
                                  "evidence_level": m["evidence_level"]} for m in plan["manifests"]],
        "inputs": {
            "bridge": str(bridge["bridge"]), "bridge_declared_result_status": bridge["declared_result_status"],
            "bridge_scene_sha256": bridge["scene_sha256"],
            "sfm_run": str(index["sfm_root"]), "sfm_run_id": index["run_json"].get("run_id"),
            "sfm_upstream_io_run_id": index["run_json"].get("upstream_io_run_id"),
            "t01_run": str(index["t01_root"]),
            "source_io": binding["source_io"],
            "verified_edges": edges, "binding_boundaries": BINDING_BOUNDARIES,
            "files": plan["snapshot"], "input_bindings_unchanged": input_stable, "drift": drift or [],
        },
        "implementation": {
            "producer_sha256": mio.sha256_file(__file__),
            "consumer_sha256": mio.sha256_file(Path(mio.__file__)),
            "geometry_core_sha256": mio.sha256_file(Path(warp_binary_mask.__globals__["__file__"])),
            "algorithm": GEOMETRY_ALGORITHM, "tile_rows": TILE_ROWS, "max_pixels": MAX_PIXELS,
        },
        "frames": frames,
        "counts": counts,
        "coverage_status": coverage,
        "quality_status": "NOT_EVALUATED",
        "cross_view_status": "NOT_EVALUATED",
        "mvs_consumption_status": "NOT_RUN",
        "mvs_mask_arguments": {"available": False, "reason": "not published", "actual_mvs_invocations": 0}
        if not verified else suggested_arguments(root, verified=True, reason=""),
        "publishable_arguments": verified,
        "scene": {"path": plan["scene"]["path"], "bytes": plan["scene"]["bytes"],
                  "sha256": plan["scene"]["sha256"], "version": plan["scene"]["version"],
                  "image_count": plan["scene"]["image_count"], "verified": plan["scene"]["verified"],
                  "mapping": plan["scene"]["mapping"]},
        "unverified_edges": unverified,
        "selection_summary": {
            "selected_model_images": len(plan["bridge_names"]),
            "frames_map_rows": index["frames_map_rows"],
            "frames_map_rows_unselected": index['frames_map_rows']-index['selected_rows'],
            'excluded_foreground_rows':index['excluded_foreground_rows'],
            "unregistered_input_indices": index["unregistered_input_indices"],
            "object_mask_frames_requested": sum(m["counts"]["requested"] for m in plan["manifests"]),
        },
        "written_outputs": written,
        "unbound_object_mask_frames": binding["problems"],
        "ignored_nonselected_frames": binding["ignored_nonselected"],
        "strict_reason": strict_reason,
        "notes": [
            "Masks are predicted labels from the declared object-mask-v1 manifests, not human truth.",
            "No OpenMVS executable was invoked; mvs_consumption_status is NOT_RUN.",
            "Distortion handling is delegated to the tested geometry core; the source mask is never resized.",
        ],
    }
    if plan.get("selection") is not None:
        result["mask_selection"] = plan["selection"]["record"]
    return result


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", required=True)
    p.add_argument("--sfm-run", required=True)
    p.add_argument("--mask-manifest", required=True, action="append")
    p.add_argument("--selection-ledger", help="explicit object-mask-selection-v1 ledger; default still rejects duplicates")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-partial", action="store_true")
    return p


def _failure(root, state, status, code, exc) -> int:
    report = {"contract_version": mio.CONTRACT_OUT, "run_id": root.name, "status": status, "failed": True,
              "first_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
              "frames": state["frames"], "written_outputs": state["written"],
              "finalization_errors": list(state["finalization_errors"])}
    try:
        save_json(root / "FAILURE.json", report)
    except Exception as secondary:  # noqa: BLE001 - finalisation must not mask the first cause
        state["finalization_errors"].append(f"{type(secondary).__name__}: {secondary}")
        try:
            (root / "FINALIZATION_ERROR.txt").write_text(
                f"FAILURE.json could not be written: {type(secondary).__name__}: {secondary}\n"
                f"first cause: {report['first_cause']}\n", encoding="utf-8", newline="\n")
        except Exception:  # noqa: BLE001 - nothing else we can do, still report on stderr
            pass
        print(f"finalisation failed: {type(secondary).__name__}: {secondary}", file=sys.stderr)
    print(f"{status.lower()}: {report['first_cause']}", file=sys.stderr)
    return code


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args._started_utc = utcnow()
    args._fixture = any("fixture" in Path(p).as_posix().casefold() for p in args.mask_manifest)
    root = Path(args.output).resolve()
    if root.exists():
        print(f"output directory already exists, refusing to overwrite: {root}", file=sys.stderr)
        return 2
    try:
        plan = preflight(args)
    except mio.InputError as exc:
        print(f"input error (no output directory was claimed): {exc}", file=sys.stderr)
        return 2
    except mio.ProductError as exc:
        print(f"product error (no output directory was claimed): {exc}", file=sys.stderr)
        return 4
    except (OSError, UnicodeError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"input read error (no output directory was claimed): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        root.mkdir(parents=True)
    except OSError as exc:
        print(f"cannot create output directory {root}: {exc}", file=sys.stderr)
        return 2
    state = {"frames": [], "written": [], "finalization_errors": []}
    try:
        manifest = execute(args, plan, root, state)
    except mio.InputError as exc:
        return _failure(root, state, "INPUT_ERROR", 2, exc)
    except mio.ProductError as exc:
        return _failure(root, state, "BACKEND_ERROR", 4, exc)
    except Exception as exc:  # noqa: BLE001 - never claim success, but let cancellation through
        return _failure(root, state, "BACKEND_ERROR", 4, exc)
    print(f"status={manifest['status']} coverage={manifest['coverage_status']} counts={manifest['counts']}")
    return mio.exit_code_for(manifest["status"])


if __name__ == "__main__":
    raise SystemExit(main())
