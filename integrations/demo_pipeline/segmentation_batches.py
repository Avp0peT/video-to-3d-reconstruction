"""Lossless aggregation of completed object-mask-v1 SAM batches (no inference)."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import uuid


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit):
    _need(path.stat().st_size <= limit, "artifact exceeds byte limit")
    raw = path.read_bytes()
    _need(len(raw) <= limit, "artifact exceeds byte limit")
    return raw


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _float(value):
    result = float(value)
    _need(math.isfinite(result), "non-finite JSON number")
    return result


def _constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def _append(copies, destination, data):
    _need(sum(len(raw) for _, raw in copies) + len(data) <= 128 << 20, "aggregate copies exceed byte limit")
    copies.append((destination, data))


def _contained(root, relative):
    _need(isinstance(relative, str) and relative, "missing artifact path")
    win = PureWindowsPath(relative)
    _need(not win.is_absolute() and not win.drive and not Path(relative).is_absolute(), "absolute artifact path")
    parts = relative.replace("\\", "/").split("/")
    _need(".." not in parts, "artifact path traversal")
    target = (root / Path(*parts)).resolve()
    _need(target.is_relative_to(root.resolve()), "artifact escapes source directory")
    return target


def _counts(frames):
    c = Counter(f["status"] for f in frames)
    return dict(requested=len(frames), success=c["OK"], no_match=c["NO_MATCH"],
                needs_selection=c["NEEDS_SELECTION"], error=0, not_run=0)


def _identity(doc, key):
    value = deepcopy(doc.get(key))
    if key == "backend" and isinstance(value, dict):
        value.pop("backend_unavailable_reason", None)
    if key == "backend_bindings" and isinstance(value, dict):
        weights = value.get("weights")
        if isinstance(weights, dict):
            for transient in ("hf_download_calls", "token_present"):
                weights.pop(transient, None)
    return value


def merge_manifests(paths, output, expected_indices) -> dict:
    """Validate exact coverage and copy bytes into a *new* aggregate directory.

    Raises ValueError on invalid evidence. No output is created until all source
    artifacts have been validated. Source manifests and masks are never written.
    """
    output = Path(output).resolve()
    _need(not output.exists(), "output must be a new directory")
    expected = list(expected_indices)
    _need(0 < len(expected) <= 192 and all(type(i) is int and i >= 0 for i in expected), "invalid expected indices")
    _need(len(set(expected)) == len(expected), "duplicate expected index")
    paths = list(paths)
    _need(0 < len(paths) <= 2, "one or two source batches required")
    docs, sources, copies, frames = [], [], [], []
    seen, frame_ids, manifest_paths = set(), set(), set()
    for batch, filename in enumerate(paths):
        path = Path(filename).resolve()
        _need(path not in manifest_paths, "duplicate source manifest")
        manifest_paths.add(path)
        raw = _read(path, 8 << 20)
        doc = json.loads(raw, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)
        _need(isinstance(doc, dict), "manifest must be object")
        _need(doc.get("contract_version") == "object-mask-v1", "unsupported manifest contract")
        _need(doc.get("status") in ("SUCCESS", "PARTIAL"), "batch is not completed SUCCESS/PARTIAL")
        for key in ("source_io", "object", "backend"):
            _need(isinstance(doc.get(key), dict) and doc[key], "missing identity: " + key)
        if docs:
            for key in ("source_io", "object", "backend", "backend_bindings"):
                _need(_identity(doc, key) == _identity(docs[0], key), "batch identity mismatch: " + key)
        fs = doc.get("frames")
        _need(isinstance(fs, list) and 0 < len(fs) <= 96, "batch must contain 1..96 frames")
        _need(all(isinstance(f, dict) and f.get("status") in ("OK", "NO_MATCH", "NEEDS_SELECTION") for f in fs), "error/not_run frame prohibited")
        counts = _counts(fs)
        _need(isinstance(doc.get("counts"), dict) and all(type(v) is int for v in doc["counts"].values()), "counts must be plain integers")
        _need(doc.get("counts") == counts, "source counts mismatch")
        _need((doc["status"] == "SUCCESS") == (counts["success"] == len(fs)), "batch status inconsistent")
        origin = dict(manifest=str(path), manifest_sha256=_sha(raw), run_id=doc.get("run_id"), status=doc["status"])
        sources.append(origin)
        _append(copies, Path("sources") / f"batch_{batch:03d}" / "manifest.json", raw)
        for original in fs:
            f = deepcopy(original)
            idx = f.get("input_index")
            _need(type(idx) is int and idx in expected, "unexpected input index")
            _need(idx not in seen, "duplicate input index")
            _need(isinstance(f.get("frame_id"), str) and f["frame_id"] not in frame_ids, "duplicate/missing frame ID")
            seen.add(idx)
            frame_ids.add(f["frame_id"])
            for dimension in ("width", "height"):
                if dimension in f:
                    _need(type(f[dimension]) is int and 0 < f[dimension] <= 1024, "unsupported frame dimensions")
            candidates = f.get("candidates")
            _need(isinstance(candidates, list) and len(candidates) <= 16 and all(isinstance(c, dict) for c in candidates), "invalid candidates")
            found = {}
            for ordinal, candidate in enumerate(candidates):
                cid = candidate.get("candidate_id")
                _need(isinstance(cid, str) and cid not in found, "duplicate/missing candidate ID")
                src = _contained(path.parent, candidate.get("mask_path"))
                _need(src.suffix.lower() == ".png", "candidate must be PNG")
                data = _read(src, 16 << 20)
                _need(_sha(data) == candidate.get("mask_sha256"), "candidate byte hash mismatch")
                dest = Path("masks") / f"{idx:06d}_{ordinal:03d}.png"
                _append(copies, dest, data)
                found[cid] = (candidate["mask_path"], candidate["mask_sha256"], dest.as_posix())
                candidate["mask_path"] = dest.as_posix()
            sid, selected = f.get("selected_candidate_id"), f.get("selected_mask")
            if f["status"] == "OK":
                _need(sid in found and isinstance(selected, dict), "selected candidate missing")
                oldpath, sha, newpath = found[sid]
                _need(selected.get("path") == oldpath and selected.get("sha256") == sha, "selected mask inconsistent")
                selected["path"] = newpath
            else:
                _need(sid is None and selected is None, "non-OK frame has selection")
                positive = sum(c.get("foreground_pixels", 1) > 0 for c in candidates)
                _need(positive == 0 if f["status"] == "NO_MATCH" else positive > 1, "non-OK candidates inconsistent")
            overlay = path.parent / "overlays" / f"{idx:06d}.png"
            artifacts = []
            if overlay.exists():
                overlay = _contained(path.parent, f"overlays/{idx:06d}.png")
                data = _read(overlay, 16 << 20)
                dest = Path("overlays") / f"{idx:06d}.png"
                _append(copies, dest, data)
                artifacts.append(dict(path=dest.as_posix(), sha256=_sha(data)))
            f["origin"] = dict(**origin, input_index=idx, overlays=artifacts)
            frames.append(f)
        docs.append(doc)
    _need(docs and seen == set(expected), "missing expected input indices")
    _need(sum(len(data) for _, data in copies) <= 128 << 20, "aggregate copies exceed byte limit")
    frames.sort(key=lambda f: f["input_index"])
    now = datetime.now(timezone.utc).isoformat()
    result = {k: deepcopy(docs[0][k]) for k in ("contract_version", "source_io", "object", "backend", "backend_bindings") if k in docs[0]}
    counts = _counts(frames)
    status = "SUCCESS" if counts["success"] == len(frames) else "PARTIAL"
    result.update(run_id="aggregate_" + uuid.uuid4().hex, started_utc=now, finished_utc=now,
                  status=status, frames=frames, counts=counts, quality_status="NOT_EVALUATED",
                  cross_view_status="NOT_EVALUATED", evidence_level="AGGREGATE_OF_SAM_BATCHES",
                  execution_mode="manifest_aggregation_no_model_execution", errors=[],
                  aggregation=dict(source_manifests=sources, sam_execution=False,
                                   model_load_count="NOT_AGGREGATED", resource_peaks="NOT_AGGREGATED"))
    output.mkdir(parents=True, exist_ok=False)
    for relative, data in copies:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        _need(_sha(target.read_bytes()) == _sha(data), "copied artifact byte hash drift")
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(status=status, exit_code=0 if status == "SUCCESS" else 1,
                run_dir=str(output), counts=counts, manifest=str(output / "manifest.json"))
