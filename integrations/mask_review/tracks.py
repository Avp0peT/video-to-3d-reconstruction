"""Bounded mask-label diagnostics on a saved native-description sidecar.

No native library is imported and no model is reconstructed or reloaded. The
caller supplies already validated canonical frames and an evidence-binding reader.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Callable

from PIL import Image

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_IMAGES = 256
MAX_POINTS = 100_000
MAX_FEATURES = 1_000_000
MAX_PAIR_WORK = 2_000_000
MAX_PAIRS = MAX_IMAGES * (MAX_IMAGES - 1) // 2
MAX_MASK_PIXELS = 16_777_216
MAX_TOTAL_MASK_PIXELS = 64 * 1024 * 1024
MAX_STAGED_BYTES = 256 * 1024 * 1024
CONVENTION = "colmap_native_pixel_center_0.5_world_to_camera"
DOMAIN = "photo_io_canonical"
INVALID_POINT_IDS = {-1, 18446744073709551615}


def _need(condition, message):
    if not condition:
        raise ValueError("track diagnostics: " + message)


def _int(value, label, minimum=0):
    _need(type(value) is int and value >= minimum, label + " must be an integer")
    return value


def _obj(value, label):
    _need(isinstance(value, dict), label + " must be an object")
    return value


def _list(value, label, maximum):
    _need(isinstance(value, list) and len(value) <= maximum, label + " list exceeds limit or is invalid")
    return value


def _sha(value):
    _need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256")
    return value


def _safe(root, relative):
    _need(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
          "unsafe relative path")
    p = PurePosixPath(relative)
    _need(not p.is_absolute() and all(x not in ("", ".", "..") for x in relative.split("/")),
          "unsafe relative path")
    target = (root / Path(*p.parts)).resolve()
    _need(target.is_relative_to(root), "path escapes source run")
    return target


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("track diagnostics: nonfinite JSON constant " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _finite_vector(value, size, label):
    _need(isinstance(value, list) and len(value) == size, label + " has wrong shape")
    _need(all(type(x) in (int, float) and math.isfinite(x) for x in value), label + " is not finite")


def _records(value, label, maximum):
    value = _obj(value, label)
    _need(len(value) <= maximum, label + " exceeds limit")
    out = {}
    for key, record in value.items():
        _need(isinstance(key, str) and re.fullmatch(r"[1-9][0-9]*", key) is not None,
              label + " has invalid ID key")
        out[int(key)] = _obj(record, label + " record")
    return out


def _ratio(a, b):
    return a / b if b else None


def _mask(frame):
    # The review input validator has already checked and bound the PNG file.
    # This interface accepts its decoded row-major L plane, never PNG bytes.
    pixels = frame["mask_bytes"]
    _need(isinstance(pixels, bytes) and len(pixels) == frame["width"] * frame["height"]
          and len(pixels) <= MAX_MASK_PIXELS, "decoded mask must contain exactly width*height bytes")
    _need(set(pixels) <= {0, 255}, "decoded mask labels must be 0/255")
    return pixels


def diagnose(sfm_run: Path, model_id: str, frames: list[dict], source_io_run_id: str,
             read_bytes: Callable[[Path, int], bytes]) -> dict:
    """Return JSON diagnostics; reject broken bindings/topology with ValueError.

    Frames also require ``image_domain=photo_io_canonical``; ``mask_bytes`` is
    a decoded row-major L plane, not an encoded PNG. The caller binds source
    PNG hashes. Coordinates come from the ``binary`` object of the chosen
    ``descriptions_path``. This is evidence about that sidecar, never a current
    native-file roundtrip check.
    """
    root = Path(sfm_run).resolve()
    _need(root.is_dir(), "source run is not a directory")
    _need(isinstance(model_id, str) and model_id, "model_id must be explicit")
    _need(isinstance(source_io_run_id, str) and source_io_run_id, "source IO run ID missing")
    _list(frames, "frames", MAX_IMAGES)

    def read(path):
        payload = read_bytes(path, MAX_FILE_BYTES)
        _need(isinstance(payload, bytes) and len(payload) <= MAX_FILE_BYTES, "input file exceeds limit")
        return payload

    run = _obj(_json(read(_safe(root, "run.json"))), "run")
    _need(run.get("upstream_io_run_id") == source_io_run_id, "source IO run mismatch")
    _need(run.get("coordinate_convention") == CONVENTION, "coordinate convention/domain mismatch")
    models = _list(run.get("models"), "models", MAX_IMAGES)
    matching = [m for m in models if isinstance(m, dict) and m.get("model_id") == model_id]
    _need(len(matching) == 1, "chosen model missing or duplicated")
    model_row = matching[0]
    description_path = _safe(root, model_row.get("descriptions_path"))
    sidecar = _obj(_json(read(description_path)), "descriptions")
    model = _obj(sidecar.get("binary"), "binary description")
    cameras = _records(model.get("cameras"), "cameras", MAX_IMAGES)
    images = _records(model.get("images"), "images", MAX_IMAGES)
    points = _records(model.get("points3D"), "points", MAX_POINTS)
    for field, expected in (("num_images", len(images)), ("num_cameras", len(cameras)),
                            ("num_points3D", len(points))):
        _need(type(model.get(field)) is int and model[field] == expected, field + " count mismatch")
    registered = _list(model.get("registered_image_ids"), "registered IDs", MAX_IMAGES)
    _need(all(type(i) is int and i in images for i in registered) and len(set(registered)) == len(registered),
          "invalid registered IDs")
    scene_ids = set(registered)
    _need(model.get("num_reg_images") == len(scene_ids), "registered count mismatch")
    _need(model_row.get("num_registered_images") == len(scene_ids)
          and model_row.get("num_points3D") == len(points), "run/description counts differ")
    _need(model_row.get("registered_image_ids") == registered, "run/description registered IDs differ")
    for cid, camera in cameras.items():
        _need(type(camera.get("camera_id")) is int and camera["camera_id"] == cid, "camera ID mismatch")
        w, h = _int(camera.get("width"), "camera width", 1), _int(camera.get("height"), "camera height", 1)
        _need(w * h <= MAX_MASK_PIXELS, "camera pixel limit exceeded")
        params = camera.get("params")
        _need(isinstance(params, list) and 0 < len(params) <= 32, "camera parameters invalid")
        _finite_vector(params, len(params), "camera parameters")

    names = set()
    feature_total = 0
    for iid, image in images.items():
        _need(type(image.get("image_id")) is int and image["image_id"] == iid, "image ID mismatch")
        cid = _int(image.get("camera_id"), "image camera ID", 1)
        _need(cid in cameras, "image camera absent")
        name = image.get("name")
        _safe(root, name)
        _need(name not in names, "duplicate image name")
        names.add(name)
        xy = _list(image.get("points2D"), "features", MAX_FEATURES)
        pids = _list(image.get("point3D_ids"), "feature point IDs", MAX_FEATURES)
        _need(len(xy) == len(pids), "feature arrays have different lengths")
        feature_total += len(xy)
        _need(feature_total <= MAX_FEATURES, "total feature limit exceeded")
        for coord, pid in zip(xy, pids):
            _finite_vector(coord, 2, "feature coordinate")
            _need(type(pid) is int and (pid in INVALID_POINT_IDS or pid in points), "feature references absent point")

    reciprocal = set()
    duplicates = []
    kept = {}
    raw_observations = 0
    pair_work = 0
    for pid, point in points.items():
        _finite_vector(point.get("xyz"), 3, "point XYZ")
        track = _list(point.get("track"), "track", MAX_FEATURES)
        _need(len(track) >= 2, "point has fewer than two observations")
        raw_observations += len(track)
        _need(raw_observations <= MAX_FEATURES, "total track observation limit exceeded")
        views = set()
        elements = set()
        for item in track:
            _need(isinstance(item, list) and len(item) == 2, "invalid track element")
            iid, idx = item
            _int(iid, "track image ID", 1)
            _int(idx, "point2D index")
            _need(iid in scene_ids, "track references unregistered image")
            _need(idx < len(images[iid]["point3D_ids"]), "track index outside feature array")
            _need(images[iid]["point3D_ids"][idx] == pid, "track reciprocity mismatch")
            _need((iid, idx) not in elements, "duplicate exact track element")
            elements.add((iid, idx))
            views.add(iid)
            reciprocal.add((iid, idx, pid))
        if len(views) != len(track):
            duplicates.append({"point3D_id": pid, "observations": len(track), "distinct_views": len(views),
                               "reason": "DUPLICATE_SAME_VIEW_ENTIRE_POINT_EXCLUDED"})
        else:
            kept[pid] = track
            pair_work += len(track) * (len(track) - 1) // 2
            _need(pair_work <= MAX_PAIR_WORK, "pair work limit exceeded")
    reverse_count = 0
    for iid, image in images.items():
        for idx, pid in enumerate(image["point3D_ids"]):
            if pid not in INVALID_POINT_IDS:
                _need((iid, idx, pid) in reciprocal, "feature missing reciprocal track")
                reverse_count += 1
    _need(reverse_count == raw_observations, "reciprocal observation count mismatch")

    map_rows = []
    for line in read(_safe(root, "frames_map.jsonl")).decode("utf-8").splitlines():
        if line.strip():
            _need(len(map_rows) < MAX_IMAGES, "frames map row limit exceeded")
            map_rows.append(_obj(_json(line), "frame map row"))
    map_by_index = {}
    registrations = {}
    for row in map_rows:
        index = _int(row.get("input_index"), "map input_index")
        _need(index not in map_by_index, "duplicate map input_index")
        _need(row.get("upstream_io_run_id") == source_io_run_id, "map source IO mismatch")
        map_by_index[index] = row
        regs = _list(row.get("registrations"), "registrations", MAX_IMAGES)
        selected = [r for r in regs if isinstance(r, dict) and r.get("model_id") == model_id]
        _need(len(selected) <= 1, "duplicate chosen registration")
        if selected:
            reg = selected[0]
            iid = _int(reg.get("image_id"), "registered image ID", 1)
            cid = _int(reg.get("camera_id"), "registered camera ID", 1)
            _need(iid in scene_ids and images[iid]["camera_id"] == cid, "registration IDs differ from sidecar")
            _need(row.get("staged_image_name") == images[iid]["name"], "registration image name mismatch")
            _safe(root, row["staged_image_name"])
            _sha(row.get("canonical_image_sha256"))
            registrations[index] = iid
    _need(set(registrations.values()) == scene_ids, "frames map does not cover scene images")

    labels = {}
    label_hashes = {}
    bound = []
    absent = []
    seen_indices = set()
    staged_cache = {}
    staged_bytes = 0
    mask_pixels = 0
    for frame in frames:
        _obj(frame, "frame")
        index = _int(frame.get("input_index"), "frame input_index")
        _need(index not in seen_indices, "duplicate requested input_index")
        seen_indices.add(index)
        _need(frame.get("image_domain") == DOMAIN, "mask image domain mismatch")
        _need(index in map_by_index, "frame missing from frames map")
        row = map_by_index[index]
        _need(frame.get("frame_id") == row.get("frame_id") and isinstance(frame.get("frame_id"), str),
              "frame identity mismatch")
        _need(frame.get("image_output") == row.get("upstream_image_output"), "upstream image path mismatch")
        _sha(frame.get("photo_io_image_sha256"))
        _need(_sha(frame.get("source_image_sha256")) == _sha(row.get("canonical_image_sha256")),
              "canonical source hash mismatch")
        if index not in registrations:
            absent.append({"input_index": index, "reason": "NOT_REGISTERED_IN_SELECTED_MODEL"})
            continue
        iid = registrations[index]
        image = images[iid]
        camera = cameras[image["camera_id"]]
        _need(type(frame.get("width")) is int and type(frame.get("height")) is int
              and (frame["width"], frame["height"]) == (camera["width"], camera["height"]),
              "canonical/camera dimensions differ")
        staged_path = _safe(root, image["name"])
        if staged_path not in staged_cache:
            payload = read(staged_path)
            staged_bytes += len(payload)
            _need(staged_bytes <= MAX_STAGED_BYTES, "total staged image byte limit exceeded")
            digest = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as rgb:
                _need(rgb.format == "PNG" and rgb.mode == "RGB" and getattr(rgb, "n_frames", 1) == 1
                      and rgb.size == (camera["width"], camera["height"])
                      and rgb.getexif().get(274, 1) == 1, "staged canonical image/domain differs")
                rgb.load()
            staged_cache[staged_path] = digest
        _need(staged_cache[staged_path] == frame["source_image_sha256"], "actual staged image hash mismatch")
        mask = None
        mask_hash = None
        if frame.get("status") == "OK":
            mask_pixels += frame["width"] * frame["height"]
            _need(mask_pixels <= MAX_TOTAL_MASK_PIXELS, "total mask pixel limit exceeded")
            mask = _mask(frame)
            mask_hash = hashlib.sha256(frame["mask_bytes"]).hexdigest()
        else:
            _need(frame.get("mask_bytes") is None, "non-OK frame has selected mask bytes")
        if iid in labels:
            _need(label_hashes[iid] == mask_hash and labels[iid] == mask,
                  "aliases have different selected masks")
        else:
            labels[iid] = mask
            label_hashes[iid] = mask_hash
        bound.append({"input_index": index, "frame_id": frame["frame_id"], "image_id": iid,
                      "image_name": image["name"], "source_image_sha256": frame["source_image_sha256"],
                      "photo_io_image_sha256": frame["photo_io_image_sha256"],
                      "selected_mask_pixels_sha256": mask_hash, "has_mask": mask is not None})

    observations = {"foreground": 0, "background": 0, "missing_mask": 0, "out_of_bounds": 0}
    point_counts = {"no_valid_observations": 0, "one_valid_observation": 0, "comparable": 0,
                    "all_foreground": 0, "all_background": 0, "mixed": 0,
                    "any_observed_foreground": 0, "comparable_foreground_union": 0}
    pair_counts = {pair: {"shared_points": 0, "FF": 0, "FB": 0, "BB": 0,
                          "excluded_unknown": 0}
                   for pair in itertools.combinations(sorted(scene_ids), 2)}
    _need(len(pair_counts) <= MAX_PAIRS, "view pair limit exceeded")
    for track in kept.values():
        sampled = {}
        for iid, idx in track:
            mask = labels.get(iid)
            x, y = images[iid]["points2D"][idx]
            camera = cameras[images[iid]["camera_id"]]
            label = None
            if mask is None:
                observations["missing_mask"] += 1
            elif not (0 <= x < camera["width"] and 0 <= y < camera["height"]):
                observations["out_of_bounds"] += 1
            else:
                label = mask[math.floor(y) * camera["width"] + math.floor(x)] == 255
                observations["foreground" if label else "background"] += 1
            sampled[iid] = label
        valid = [v for v in sampled.values() if v is not None]
        fg, bg = sum(valid), len(valid) - sum(valid)
        point_counts["any_observed_foreground"] += int(fg > 0)
        if len(valid) < 2:
            point_counts["no_valid_observations" if not valid else "one_valid_observation"] += 1
        else:
            point_counts["comparable"] += 1
            point_counts["comparable_foreground_union"] += int(fg > 0)
            point_counts["mixed" if fg and bg else "all_foreground" if fg else "all_background"] += 1
        for pair in itertools.combinations(sorted(sampled), 2):
            record = pair_counts[pair]
            record["shared_points"] += 1
            a, b = sampled[pair[0]], sampled[pair[1]]
            if a is None or b is None:
                record["excluded_unknown"] += 1
            else:
                record["FF" if a and b else "BB" if not a and not b else "FB"] += 1
    pairs = []
    totals = {key: 0 for key in ("shared_points", "FF", "FB", "BB", "excluded_unknown")}
    supported = 0
    comparable_pairs = 0
    for (a, b), record in pair_counts.items():
        for key in totals:
            totals[key] += record[key]
        n = record["FF"] + record["FB"] + record["BB"]
        union = record["FF"] + record["FB"]
        supported += int(record["shared_points"] > 0)
        comparable_pairs += int(n > 0)
        pairs.append({"image_ids": [a, b], **record, "valid_shared_points": n,
                      "foreground_union": union, "agreement": _ratio(record["FF"] + record["BB"], n),
                      "foreground_conditioned_disagreement": _ratio(record["FB"], union)})
    n = totals["FF"] + totals["FB"] + totals["BB"]
    union = totals["FF"] + totals["FB"]
    totals.update(valid_shared_points=n, foreground_union=union,
                  agreement=_ratio(totals["FF"] + totals["BB"], n),
                  foreground_conditioned_disagreement=_ratio(totals["FB"], union))
    point_counts.update(mixed_fraction=_ratio(point_counts["mixed"], point_counts["comparable"]),
                        foreground_conditioned_disagreement=_ratio(point_counts["mixed"], point_counts["comparable_foreground_union"]))
    flags = ["DIAGNOSTICS_ONLY_NOT_OBJECT_IDENTITY_CERTIFICATION", "SOURCE_SIDECAR_NOT_NATIVE_RELOADED"]
    if duplicates:
        flags.append("DUPLICATE_SAME_VIEW_POINTS_EXCLUDED")
    if not n:
        flags.append("NO_COMPARABLE_POINT_PAIRS")
    if not union:
        flags.append("NO_COMPARABLE_FOREGROUND_SUPPORT")
    if totals["FB"]:
        flags.append("OBSERVED_FOREGROUND_BACKGROUND_DISAGREEMENT")
    return {
        "status": "DIAGNOSTICS_ONLY", "cross_view_status": "NOT_EVALUATED", "quality_status": "NOT_EVALUATED",
        "source_evidence": "SOURCE_SIDECAR_NOT_NATIVE_RELOADED", "source_sfm_run": str(root),
        "model_id": model_id, "source_io_run_id": source_io_run_id, "descriptions_path": str(description_path),
        "coordinate_convention": CONVENTION, "image_domain": DOMAIN,
        "sampling": "floor(native x),floor(native y); finite raster support [0,W)x[0,H); no offset or clamp",
        "limitations": "Saved binary-description observations only; no current native reload, object identity, surface coverage, geometry accuracy, or quality certification.",
        "limits": {"file_bytes": MAX_FILE_BYTES, "images": MAX_IMAGES, "points": MAX_POINTS,
                   "features": MAX_FEATURES, "pair_work": MAX_PAIR_WORK, "view_pairs": MAX_PAIRS,
                   "total_mask_pixels": MAX_TOTAL_MASK_PIXELS, "total_staged_bytes": MAX_STAGED_BYTES},
        "bound_frames": bound, "unregistered_requested_frames": absent,
        "scene": {"images": len(scene_ids), "points": len(points), "features": feature_total,
                  "raw_observations": raw_observations, "eligible_points": len(kept),
                  "eligible_observations": sum(len(t) for t in kept.values()), "pair_work": pair_work},
        "excluded_duplicate_points": duplicates,
        "excluded_duplicate_point_count": len(duplicates),
        "excluded_duplicate_observations": sum(r["observations"] for r in duplicates),
        "coverage": {"requested_frames": len(frames), "bound_unique_images": len(labels),
                     "images_with_masks": sum(m is not None for m in labels.values()),
                     "scene_images": len(scene_ids), "scene_image_mask_fraction": _ratio(sum(m is not None for m in labels.values()), len(scene_ids)),
                     "all_scene_view_pairs": len(pair_counts), "pairs_with_shared_points": supported,
                     "pairs_with_valid_samples": comparable_pairs,
                     "valid_pair_coverage": _ratio(comparable_pairs, len(pair_counts))},
        "observations": observations, "points": point_counts, "pair_totals": totals, "pairs": pairs, "flags": flags,
    }
