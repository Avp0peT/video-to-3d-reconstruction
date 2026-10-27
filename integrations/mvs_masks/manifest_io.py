"""mvs-mask-v1 consumer: object-mask-v1 manifests to undistorted-domain mask files.

This module owns the file and format layer only: strict JSON, byte-level hashing,
path safety, manifest schema validation, mask PNG label/identity checks and the
OpenMVS 2.4.0 mask file naming rule. It does not read COLMAP models, does not
warp pixels and does not call OpenMVS. Geometry lives in ``mask_geometry.py`` and
is reused, never reimplemented.

The accepted manifest dialect is the one the real producer writes
(``integrations/sam3/sam3_adapter/runner.py`` and ``worker.py``):

* a candidate carries ``mask_path`` / ``mask_sha256``;
* ``selected_mask`` carries exactly ``path`` / ``sha256`` pointing at that
  candidate's raster;
* ``selection`` carries ``method`` / ``source`` / ``mask_source``;
* ``status == "OK"`` always comes with a non-null ``selected_candidate_id``,
  ``selected_mask`` and ``selection``, because selection only succeeds when a
  usable candidate was chosen; any other status carries none of them.

Nothing here can upgrade a predicted mask into human truth, and no function claims
that OpenMVS actually consumed the labels.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

CONTRACT_IN = "object-mask-v1"
CONTRACT_OUT = "mvs-mask-v1"
IMAGE_DOMAIN = "colmap_undistorted"
SOURCE_IMAGE_DOMAIN = "photo_io_canonical"
MASK_SUFFIX = ".mask.png"
VALIDITY_SUFFIX = ".validity.png"
MASK_LABELS = (0, 255)
ALGORITHM_ID = "colmap-center-nearest-label-v1"

MANIFEST_RUN_STATUS = (
    "SUCCESS", "PARTIAL", "INPUT_ERROR", "BACKEND_UNAVAILABLE", "BACKEND_ERROR",
    "AUTH_REQUIRED", "RESOURCE_LIMIT",
)
FRAME_STATUS = (
    "OK", "NO_MATCH", "NEEDS_SELECTION", "INPUT_ERROR", "BACKEND_ERROR",
    "RESOURCE_LIMIT", "NOT_RUN",
)
#: per-frame status -> the manifest counts key it belongs to
STATUS_TO_COUNT = {
    "OK": "success", "NO_MATCH": "no_match", "NEEDS_SELECTION": "needs_selection",
    "INPUT_ERROR": "error", "BACKEND_ERROR": "error", "RESOURCE_LIMIT": "error",
    "NOT_RUN": "not_run",
}
MASK_SOURCES = ("automatic_sam3", "automatic_sam3_human_edited", "video_temporal_union")
SELECTION_METHODS = ("unique_candidate", "explicit_candidate")
OUTPUT_RUN_STATUS = ("SUCCESS", "PARTIAL", "INPUT_ERROR", "BACKEND_ERROR")
_OUTPUT_STATUS_TO_EXIT = {"SUCCESS": 0, "PARTIAL": 1, "INPUT_ERROR": 2, "BACKEND_ERROR": 4}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InputError(Exception):
    """A declared or referenced input is missing, inconsistent or corrupt (exit 2)."""


class ProductError(Exception):
    """Our own output could not be produced or written correctly (exit 4)."""


# --------------------------------------------------------------------------- #
# bytes, strict JSON, paths
# --------------------------------------------------------------------------- #
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path) -> dict:
    """Actual bytes binding for one existing file."""
    target = Path(path)
    if not target.is_file():
        raise InputError(f"referenced file is missing: {target}")
    return {"path": str(target.resolve()), "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def _reject_constant(name):
    raise InputError(f"non-finite JSON constant is not allowed in evidence: {name}")


def _object_pairs(pairs):
    """Reject duplicate keys instead of silently keeping the last one."""
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise InputError(f"duplicate JSON key {key!r}")
        seen[key] = value
    return seen


def walk_finite(value, where: str) -> None:
    """Reject any non-finite number, including 1e999 which json parses to inf."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InputError(f"{where}: non-finite number is not allowed")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            walk_finite(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            walk_finite(item, f"{where}[{position}]")


def load_json_strict(path, where: str = "json"):
    target = Path(path)
    if not target.is_file():
        raise InputError(f"{where} missing: {target}")
    text = target.read_text(encoding="utf-8")
    try:
        doc = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_object_pairs)
    except InputError:
        raise
    except json.JSONDecodeError as exc:
        raise InputError(f"{where} is not valid JSON: {target}: {exc}") from exc
    if not isinstance(doc, dict):
        raise InputError(f"{where} must be a JSON object: {target}")
    walk_finite(doc, where)
    return doc


def load_json_array_strict(path, where: str = "json array") -> list:
    """Strict loader for a top-level JSON array (bridge INPUT_BINDINGS.json).

    The same rules apply as for objects: duplicate keys anywhere, non-finite
    numbers including overflowing literals, and invalid UTF-8 are rejected.
    """
    target = Path(path)
    if not target.is_file():
        raise InputError(f"{where} missing: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{where} is not valid UTF-8: {target}: {exc}") from exc
    try:
        doc = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_object_pairs)
    except InputError:
        raise
    except json.JSONDecodeError as exc:
        raise InputError(f"{where} is not valid JSON: {target}: {exc}") from exc
    if not isinstance(doc, list):
        raise InputError(f"{where} must be a JSON array: {target}")
    walk_finite(doc, where)
    return doc


def read_jsonl_strict(path, where: str = "jsonl") -> list:
    target = Path(path)
    if not target.is_file():
        raise InputError(f"{where} missing: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{where} is not valid UTF-8: {target}: {exc}") from exc
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_object_pairs)
        except InputError:
            raise
        except json.JSONDecodeError as exc:
            raise InputError(f"{where} line {number} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise InputError(f"{where} line {number} must be a JSON object")
        walk_finite(row, f"{where} line {number}")
        rows.append(row)
    if not rows:
        raise InputError(f"{where} is empty: {target}")
    return rows


def normalised(path) -> str:
    """Case-folded, separator-normalised absolute path for identity comparisons."""
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def ensure_inside(root, path, what: str = "path") -> Path:
    resolved_root, resolved = Path(root).resolve(), Path(path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise InputError(f"{what} escapes its declared root: {resolved} not inside {resolved_root}")
    return resolved


def require_sha256(value, what: str) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        raise InputError(f"{what} must be a lowercase sha256 hex string, got {value!r}")
    return value


def require_int(value, what: str, minimum: int = 0) -> int:
    """Plain integers only; bool is rejected because it is not a count."""
    if type(value) is not int or value < minimum:
        raise InputError(f"{what} must be a plain integer >= {minimum}, got {value!r}")
    return value


def require_str(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{what} must be a non-empty string, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# OpenMVS 2.4.0 mask file naming (Util::getFileName: basename, last extension dropped)
# --------------------------------------------------------------------------- #
def mask_output_name(image_name: str) -> str:
    """images/000017.png -> 000017.mask.png (not 000017.png.mask.png)."""
    if not isinstance(image_name, str) or not image_name:
        raise InputError(f"image name must be a non-empty string, got {image_name!r}")
    text = image_name.replace("\\", "/")
    base = PurePosixPath(text).name or PureWindowsPath(image_name).name
    if "." in base[1:]:
        base = base[: base.rindex(".")]
    if not base or base in (".", ".."):
        raise InputError(f"image name has no usable basename stem: {image_name!r}")
    return base + MASK_SUFFIX


def output_plan(image_names) -> dict:
    names = [str(n) for n in image_names]
    mapping, seen, collisions = {}, {}, []
    for name in names:
        out = mask_output_name(name)
        key = out.casefold()
        if key in seen:
            collisions.append(sorted([seen[key], name]))
            continue
        seen[key] = name
        mapping[name] = out
    if collisions:
        raise InputError(f"OpenMVS mask file name collision (Windows is case-insensitive): {collisions}")
    return {"by_image": mapping, "count": len(mapping)}


# --------------------------------------------------------------------------- #
# mask PNG label identity
# --------------------------------------------------------------------------- #
def _pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ProductError(f"Pillow is required for mask IO: {exc}") from exc
    return Image


def read_label_png(path, expected_width: int, expected_height: int):
    """Strict single-frame 8-bit grayscale label raster. Returns (array, record)."""
    import numpy as np

    target = Path(path)
    if not target.is_file():
        raise InputError(f"mask file missing: {target}")
    Image = _pillow()
    try:
        with Image.open(target) as image:
            image.load()
            if image.format != "PNG":
                raise InputError(f"mask must be PNG: {target} is {image.format}")
            if image.mode != "L":
                raise InputError(f"mask must be single-channel L mode, got {image.mode}: {target}")
            if getattr(image, "n_frames", 1) != 1:
                raise InputError(f"mask must be a single frame: {target}")
            if "transparency" in image.info:
                raise InputError(f"mask must not carry transparency: {target}")
            orientation = image.getexif().get(274, 1)
            if orientation not in (None, 1):
                raise InputError(f"mask must be orientation 1 or absent, got {orientation}: {target}")
            array = np.asarray(image, dtype=np.uint8)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"mask PNG cannot be decoded: {target}: {exc}") from exc
    if array.shape != (expected_height, expected_width):
        raise InputError(
            f"mask size {array.shape[::-1]} does not equal canonical size "
            f"{(expected_width, expected_height)}: {target}"
        )
    labels = sorted(int(v) for v in np.unique(array))
    if any(value not in MASK_LABELS for value in labels):
        raise InputError(f"mask labels must be 0/255 with no implicit threshold, found {labels}: {target}")
    return array, {
        "path": str(target.resolve()),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "width": int(expected_width),
        "height": int(expected_height),
        "labels": labels,
        "foreground_pixels": int(np.count_nonzero(array)),
    }


def verify_label_png(path, expected_array, expected_width: int, expected_height: int) -> dict:
    """Read a written raster back and require identical pixels and legal labels."""
    import numpy as np

    array, record = read_label_png(path, expected_width, expected_height)
    expected = np.asarray(expected_array)
    if array.shape != expected.shape or not np.array_equal(array, expected):
        raise ProductError(f"written raster does not match the array handed to the writer: {record['path']}")
    return record


def write_label_png(path, array) -> dict:
    """Write an 8-bit L-mode PNG; return the bytes binding read back from disk."""
    import numpy as np

    data = np.asarray(array)
    if data.dtype != np.uint8 or data.ndim != 2:
        raise ProductError("label output must be a 2-D uint8 array")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image = _pillow()
    Image.fromarray(data, mode="L").save(target, format="PNG", optimize=False)
    return file_record(target)


# --------------------------------------------------------------------------- #
# coverage accounting and run status
# --------------------------------------------------------------------------- #
def counts_from_frames(frames) -> dict:
    """expected/ok/missing/error/not_run, recomputed from the frame list alone."""
    allowed = ("OK", "MISSING", "INPUT_ERROR", "EMPTY_AFTER_WARP", "NOT_RUN")
    for frame in frames:
        if frame.get("status") not in allowed:
            raise ProductError(f"unknown output frame status {frame.get('status')!r}")
    counts = {
        "expected": len(frames),
        "ok": sum(1 for f in frames if f["status"] == "OK"),
        "missing": sum(1 for f in frames if f["status"] == "MISSING"),
        "error": sum(1 for f in frames if f["status"] in ("INPUT_ERROR", "EMPTY_AFTER_WARP")),
        "not_run": sum(1 for f in frames if f["status"] == "NOT_RUN"),
    }
    if sum(counts[k] for k in ("ok", "missing", "error", "not_run")) != counts["expected"]:
        raise ProductError(f"frame statuses do not sum to expected: {counts}")
    return counts


def coverage_status(counts) -> str:
    if counts["expected"] == 0:
        raise ProductError("no bridge images to cover")
    return "COMPLETE" if counts["ok"] == counts["expected"] else "PARTIAL"


def classify_run(counts, *, allow_partial: bool) -> tuple:
    """Map coverage onto (run_status, exit_code) without upgrading partial results."""
    coverage = coverage_status(counts)
    if counts["error"] > 0:
        return "INPUT_ERROR", 2
    if coverage == "COMPLETE":
        return "SUCCESS", 0
    if counts["ok"] == 0:
        return "INPUT_ERROR", 2
    return ("PARTIAL", 1) if allow_partial else ("INPUT_ERROR", 2)


def exit_code_for(status: str) -> int:
    if status not in _OUTPUT_STATUS_TO_EXIT:
        raise ProductError(f"unknown run status {status!r}")
    return _OUTPUT_STATUS_TO_EXIT[status]


# --------------------------------------------------------------------------- #
# object-mask-v1 manifest (production dialect)
# --------------------------------------------------------------------------- #
def _parse_candidate(candidate, spot: str, run_root):
    if not isinstance(candidate, dict):
        raise InputError(f"{spot}: each candidate must be an object")
    cid = require_str(candidate.get("candidate_id"), f"{spot}.candidate_id")
    rel = require_str(candidate.get("mask_path"), f"{spot}.mask_path")
    digest = require_sha256(candidate.get("mask_sha256"), f"{spot}.mask_sha256")
    resolved = ensure_inside(run_root, run_root / rel, f"{spot}.mask_path")
    foreground = candidate.get("foreground_pixels")
    if foreground is not None:
        foreground = require_int(foreground, f"{spot}.foreground_pixels")
    return {"candidate_id": cid, "mask_path": rel, "mask_sha256": digest,
            "resolved_mask_path": str(resolved), "foreground_pixels": foreground,
            "score": candidate.get("score"), "score_semantics": candidate.get("score_semantics")}


def parse_object_manifest(path) -> dict:
    """Validate one object-mask-v1 manifest written by the production adapter."""
    manifest_path = Path(path).resolve()
    doc = load_json_strict(manifest_path, "object-mask manifest")
    if (doc.get('backend', {}).get('name') == 'video_temporal_union'
            or any(f.get('selection', {}).get('mask_source') == 'video_temporal_union'
                   for f in doc.get('frames', []) if isinstance(f, dict) and isinstance(f.get('selection'), dict))):
        from integrations.video_masks.reconstruction_input import validate_derivation
        try:
            validate_derivation(doc, manifest_path.parent)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise InputError('temporal union provenance invalid: ' + str(exc)) from exc
    where = f"object-mask manifest {manifest_path}"
    if doc.get("contract_version") != CONTRACT_IN:
        raise InputError(f"{where}: contract_version must be {CONTRACT_IN!r}, got {doc.get('contract_version')!r}")
    run_id = require_str(doc.get("run_id"), f"{where}.run_id")
    status = require_str(doc.get("status"), f"{where}.status")
    if status not in MANIFEST_RUN_STATUS:
        raise InputError(f"{where}: unknown status {status!r}")
    for key in ("started_utc", "finished_utc"):
        require_str(doc.get(key), f"{where}.{key}")
    source_io = doc.get("source_io")
    if not isinstance(source_io, dict):
        raise InputError(f"{where}: source_io must be an object")
    if not isinstance(doc.get("backend"), dict):
        raise InputError(f"{where}: backend must be an object")
    object_block = doc.get("object")
    if not isinstance(object_block, dict):
        raise InputError(f"{where}: object must be an object")
    object_id = require_str(object_block.get("object_id"), f"{where}.object.object_id")
    frames = doc.get("frames")
    if not isinstance(frames, list) or not frames:
        raise InputError(f"{where}: frames must be a non-empty list")
    counts = doc.get("counts")
    if not isinstance(counts, dict):
        raise InputError(f"{where}: counts must be an object")
    declared = {key: require_int(counts.get(key), f"{where}.counts.{key}")
                for key in ("requested", "success", "no_match", "needs_selection", "error", "not_run")}
    if declared["requested"] != len(frames):
        raise InputError(f"{where}: counts.requested {declared['requested']} != len(frames) {len(frames)}")

    run_root = manifest_path.parent
    parsed, tally = [], {key: 0 for key in STATUS_TO_COUNT.values()}
    for position, frame in enumerate(frames):
        spot = f"{where}.frames[{position}]"
        if not isinstance(frame, dict):
            raise InputError(f"{spot} must be an object")
        input_index = require_int(frame.get("input_index"), f"{spot}.input_index")
        frame_status = require_str(frame.get("status"), f"{spot}.status")
        if frame_status not in FRAME_STATUS:
            raise InputError(f"{spot}: unknown frame status {frame_status!r}")
        tally[STATUS_TO_COUNT[frame_status]] += 1
        domain = frame.get("image_domain")
        if domain != SOURCE_IMAGE_DOMAIN:
            raise InputError(f"{spot}: image_domain must be {SOURCE_IMAGE_DOMAIN!r}, got {domain!r}")
        if frame_status == "OK" or frame.get("source_image") is not None:
            require_str(frame.get("source_image"), f"{spot}.source_image")
        frame_id = frame.get("frame_id")
        if frame_id is not None:
            require_str(frame_id, f"{spot}.frame_id")
        width, height = frame.get("width"), frame.get("height")
        if frame_status == "OK" or frame.get("source_image") is not None:
            width = require_int(width, f"{spot}.width", 1)
            height = require_int(height, f"{spot}.height", 1)
        candidates = frame.get("candidates")
        if not isinstance(candidates, list):
            raise InputError(f"{spot}: candidates must be a list")
        parsed_candidates = [_parse_candidate(c, spot, run_root) for c in candidates]
        ids = [c["candidate_id"] for c in parsed_candidates]
        if len(set(ids)) != len(ids):
            raise InputError(f"{spot}: duplicate candidate_id inside one frame")
        selected_id = frame.get("selected_candidate_id")
        selected_mask = frame.get("selected_mask")
        selection = frame.get("selection")
        chosen = None
        if frame_status == "OK":
            if not isinstance(selected_id, str) or not isinstance(selected_mask, dict) or not isinstance(selection, dict):
                raise InputError(f"{spot}: status OK requires selected_candidate_id, selected_mask and selection")
            if set(selected_mask) != {"path", "sha256"}:
                raise InputError(f"{spot}: selected_mask must carry exactly path and sha256, got {sorted(selected_mask)}")
            chosen = next((c for c in parsed_candidates if c["candidate_id"] == selected_id), None)
            if chosen is None:
                raise InputError(f"{spot}: selected_candidate_id {selected_id!r} is not among its candidates")
            if selected_mask["path"] != chosen["mask_path"]:
                raise InputError(f"{spot}: selected_mask.path does not match the selected candidate mask_path")
            require_sha256(selected_mask["sha256"], f"{spot}.selected_mask.sha256")
            if selected_mask["sha256"] != chosen["mask_sha256"]:
                raise InputError(f"{spot}: selected_mask.sha256 does not match the selected candidate mask_sha256")
            require_str(selection.get("method"), f"{spot}.selection.method")
            if selection["method"] not in SELECTION_METHODS:
                raise InputError(f"{spot}: unknown selection.method {selection['method']!r}")
            require_str(selection.get("source"), f"{spot}.selection.source")
            if selection.get("mask_source") not in MASK_SOURCES:
                raise InputError(f"{spot}: selection.mask_source {selection.get('mask_source')!r} is not accepted")
            if selection["method"] == "unique_candidate" and len(parsed_candidates) != 1:
                raise InputError(
                    f"{spot}: selection.method unique_candidate requires exactly one candidate, "
                    f"found {len(parsed_candidates)}"
                )
            foreground = chosen["foreground_pixels"]
            if foreground is None or foreground <= 0:
                raise InputError(f"{spot}: the selected candidate must declare a positive foreground_pixels")
        elif selected_id is not None or selected_mask is not None or selection is not None:
            raise InputError(f"{spot}: status {frame_status} must not carry a selection")
        if frame_status == "NO_MATCH" and parsed_candidates:
            raise InputError(f"{spot}: NO_MATCH must have no candidate")
        if frame_status == "NEEDS_SELECTION" and len(parsed_candidates) < 2:
            raise InputError(f"{spot}: NEEDS_SELECTION requires multiple candidates")
        if frame_status == "NOT_RUN":
            require_str(frame.get("not_run_reason"), f"{spot}.not_run_reason")
        parsed.append({
            "input_index": input_index, "frame_id": frame_id, "status": frame_status,
            "source_image": frame.get("source_image"), "source_image_sha256": frame.get("source_image_sha256"),
            "photo_io_image_sha256": frame.get("photo_io_image_sha256"),
            "image_domain": frame.get("image_domain"), "width": width, "height": height,
            "candidates": parsed_candidates, "selected_candidate_id": selected_id,
            "selected_mask": ({"path": selected_mask["path"], "sha256": selected_mask["sha256"]}
                              if isinstance(selected_mask, dict) else None),
            "selection": ({key: selection.get(key) for key in ("method", "source", "mask_source")}
                          if isinstance(selection, dict) else None),
            "chosen": chosen,
        })
    declared_statuses = {key: declared[key] for key in STATUS_TO_COUNT.values()}
    if declared_statuses != tally:
        raise InputError(f"{where}: counts {declared_statuses} disagree with the per-frame statuses {tally}")
    if status == "SUCCESS" and tally["success"] != len(frames):
        raise InputError(
            f"{where}: manifest declares SUCCESS but {len(frames) - tally['success']} frame(s) are not OK"
        )
    if status != "SUCCESS" and tally["success"] == len(frames):
        raise InputError(f"{where}: every frame is OK but the manifest declares {status!r}, not SUCCESS")
    if status == "PARTIAL" and (tally["error"] or tally["not_run"]
                                or not (tally["no_match"] or tally["needs_selection"])):
        raise InputError(f"{where}: PARTIAL requires completed inference with NO_MATCH/NEEDS_SELECTION and no error/NOT_RUN")
    return {
        "manifest_path": str(manifest_path), "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": sha256_file(manifest_path), "run_id": run_id, "status": status,
        "object_id": object_id, "source_io": source_io, "backend": doc["backend"],
        "quality_status": doc.get("quality_status"), "cross_view_status": doc.get("cross_view_status"),
        "evidence_level": doc.get("evidence_level"), "frames": parsed, "counts": declared,
    }


def resolve_source_io(source_io, where: str) -> dict:
    """Locate the T01 run from source_io.path and verify its declared hashes."""
    if not isinstance(source_io, dict):
        raise InputError(f"{where}: source_io must be an object")
    root_declared = require_str(source_io.get("path"), f"{where}.source_io.path")
    declared_run_id = require_str(source_io.get("run_id"), f"{where}.source_io.run_id")
    declared_run_sha = require_sha256(source_io.get("run_json_sha256"), f"{where}.source_io.run_json_sha256")
    declared_frames_sha = require_sha256(source_io.get("frames_jsonl_sha256"), f"{where}.source_io.frames_jsonl_sha256")
    root = Path(root_declared).resolve()
    if not root.is_dir():
        raise InputError(f"{where}: source_io.path is not a directory: {root}")
    run_json, frames_jsonl = root / "run.json", root / "frames.jsonl"
    record = {"path": str(root), "declared_run_id": declared_run_id,
              "run_json": file_record(run_json), "frames_jsonl": file_record(frames_jsonl)}
    if record["run_json"]["sha256"] != declared_run_sha:
        raise InputError(f"{where}: source_io.run_json_sha256 mismatch for {run_json}")
    if record["frames_jsonl"]["sha256"] != declared_frames_sha:
        raise InputError(f"{where}: source_io.frames_jsonl_sha256 mismatch for {frames_jsonl}")
    upstream = load_json_strict(run_json, "source_io run.json")
    if upstream.get("contract_version") != "photo-io-v1":
        raise InputError(f"{where}: source_io.path is not a photo-io-v1 run: {run_json}")
    if upstream.get("run_id") != declared_run_id:
        raise InputError(f"{where}: source_io.run_id {declared_run_id!r} != run.json run_id {upstream.get('run_id')!r}")
    if upstream.get("status") != "SUCCESS":
        raise InputError(f"{where}: source_io run status {upstream.get('status')!r} is not SUCCESS")
    record["run_id"] = declared_run_id
    return record
