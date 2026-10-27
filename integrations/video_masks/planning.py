"""Read and freeze a bounded propagation experiment from existing evidence.

No inference, image conversion, mask writing, or ground-truth assertion occurs.
Invalid bindings raise ValueError; a valid dataset without a bracket returns
NOT_APPLICABLE. Paths in a returned plan are absolute.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _index(rows: list[dict[str, Any]], key: str) -> dict[int, dict[str, Any]]:
    result = {}
    for row in rows:
        index = _integer(row[key], key)
        _require(index not in result, f"duplicate {key}")
        result[index] = row
    return result


def _validated_evidence(pipeline_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every source binding before any interval selection."""
    pipeline = Path(pipeline_dir).resolve()
    pipeline_plan_path = pipeline / "PLAN.json"
    pipeline_plan = _read(pipeline_plan_path)
    video_result_path = pipeline / "input/video/RESULT.json"
    video = _read(video_result_path)
    outcome_path = pipeline / "segmentation/PRODUCT_OUTCOME.json"
    outcome = _read(outcome_path)
    if outcome.get("manifest"):
        manifest_path = _path(outcome["manifest"], outcome_path.parent)
    else:
        manifest_path = _path(outcome["run_dir"], outcome_path.parent) / "manifest.json"
    manifest = _read(manifest_path)

    source = pipeline_plan["source"]
    _require(source["kind"] == "video", "pipeline source is not video")
    source_path = _path(source["path"], pipeline)
    _require(_path(video["source_video"], video_result_path.parent) == source_path,
             "source video path mismatch")
    source_files = source["files"]
    _require(len(source_files) == 1, "expected one source video binding")
    _require(_path(source_files[0]["path"], pipeline) == source_path,
             "PLAN source file path mismatch")
    source_hash = _hash(source_path)
    _require(source_hash == source_files[0]["sha256"] == video["source_sha256"]
             == video["source_sha256_after"], "source video hash mismatch")
    _require(video["status"] == "SUCCESS", "video preparation not successful")

    io_binding = manifest["source_io"]
    io_dir = _path(io_binding["path"], manifest_path.parent)
    io_run_path = io_dir / "run.json"
    io_frames_path = io_dir / "frames.jsonl"
    io_run = _read(io_run_path)
    io_run_hash, io_frames_hash = _hash(io_run_path), _hash(io_frames_path)
    _require(_path(video["photo_io"]["run"], video_result_path.parent) == io_run_path,
             "video photo_io run path mismatch")
    _require(io_run_hash == io_binding["run_json_sha256"] == video["photo_io"]["sha256"],
             "photo_io run hash mismatch")
    _require(io_frames_hash == io_binding["frames_jsonl_sha256"]
             == video["photo_io"]["frames_sha256"], "photo_io frames hash mismatch")
    _require(io_run["run_id"] == io_binding["run_id"], "photo_io run identity mismatch")
    _require(io_run["status"] == "SUCCESS", "photo_io run not successful")
    with io_frames_path.open(encoding="utf-8") as stream:
        io_rows = _index([json.loads(line) for line in stream if line.strip()], "input_index")
    frames = _index(manifest["frames"], "input_index")
    sampled = _index([row for row in video["frames"] if "photo_input_index" in row],
                     "photo_input_index")
    _require(set(frames) == set(io_rows) == set(sampled), "sampled input coverage mismatch")
    _require(io_run["input_count"] == len(io_rows), "photo_io input count mismatch")
    timeline_path = pipeline / "input/video/timeline.json"
    timeline = _read(timeline_path)
    timeline_rows = _index(timeline["frames"], "decode_index")

    validated = []
    for index in sorted(frames):
        row, io, sample = frames[index], io_rows[index], sampled[index]
        _require(row["frame_id"] == io["frame_id"] == sample["photo_frame_id"],
                 f"frame_id mismatch at {index}")
        _require(row["photo_io_image_sha256"] == io["image_sha256"] == sample["sha256"],
                 f"photo_io image hash binding mismatch at {index}")
        _require(io["status"] == "OK" and sample["status"] == "PHOTO_IO_OK",
                 f"input status mismatch at {index}")
        _require(sample["export_pts_verified"] is True, f"unverified exported PTS at {index}")
        decode = _integer(sample["decode_index"], "decode_index")
        _require(decode in timeline_rows, f"decode index outside timeline at {index}")
        native = timeline_rows[decode]
        _require(sample["pts"] == native["pts"]
                 and sample["time_base"] == native["time_base"]
                 and sample["time_seconds"] == native["time_seconds"],
                 f"timeline mismatch at {index}")
        width = _integer(row["width"], "width", 1)
        height = _integer(row["height"], "height", 1)
        _require(width == io["canonical_width"] == sample["width"]
                 and height == io["canonical_height"] == sample["height"],
                 f"image dimensions mismatch at {index}")
        image_path = _path(row["source_image"], manifest_path.parent)
        _require(image_path == _path(io["image_output"], io_dir),
                 f"source image path mismatch at {index}")
        _require(_hash(image_path) == row["source_image_sha256"],
                 f"source image hash mismatch at {index}")
        entry = {"input_index": index, "frame_id": row["frame_id"],
                 "decode_index": decode, "pts": sample["pts"],
                 "time_base": sample["time_base"], "time_seconds": sample["time_seconds"],
                 "original_status": row["status"], "source_image": str(image_path),
                 "source_image_sha256": row["source_image_sha256"],
                 "width": width, "height": height}
        if row["status"] == "OK":
            selected = row["selected_mask"]
            _require(isinstance(selected, dict), f"OK missing selected mask at {index}")
            mask_path = _path(selected["path"], manifest_path.parent)
            _require(_hash(mask_path) == selected["sha256"], f"mask hash mismatch at {index}")
            entry.update(mask_path=str(mask_path), mask_sha256=selected["sha256"])
        validated.append(entry)
    ordered = sorted(validated, key=lambda row: row["decode_index"])
    _require(len({row["decode_index"] for row in ordered}) == len(ordered),
             "duplicate sampled decode index")
    bindings = {
        "source_video_path": str(source_path), "source_video_sha256": source_hash,
        "source_manifest_path": str(manifest_path), "source_manifest_sha256": _hash(manifest_path),
        "video_result_path": str(video_result_path), "video_result_sha256": _hash(video_result_path),
        "pipeline_plan_path": str(pipeline_plan_path), "pipeline_plan_sha256": _hash(pipeline_plan_path),
        "io_run_path": str(io_run_path), "io_run_sha256": io_run_hash,
        "io_frames_path": str(io_frames_path), "io_frames_sha256": io_frames_hash,
        "timeline_path": str(timeline_path), "timeline_sha256": _hash(timeline_path),
    }
    return ordered, bindings


def _interval_plan(start, end, interval, bindings, max_frames):
    start_index = start["decode_index"]
    count = end["decode_index"] - start_index + 1
    _require(all((row["width"], row["height"]) == (start["width"], start["height"])
                 for row in interval), "interval image dimensions differ")
    targets = [dict(row, local_frame_index=row["decode_index"] - start_index) for row in interval]
    frozen_seeds = [dict(row, local_frame_index=row["decode_index"] - start_index)
                    for row in (start, end)]
    return dict(bindings, **{
        "schema": "video-mask-propagation-plan-v1", "status": "PLANNED",
        "seed_quality": "NOT_EVALUATED", "candidate_quality": "NOT_EVALUATED",
        "start_decode_index": start_index, "end_decode_index": end["decode_index"],
        "frame_count": count, "max_frames": max_frames,
        "width": start["width"], "height": start["height"],
        "seeds": frozen_seeds, "targets": targets,
        "original_masks_preserved": True, "output_classification": "review_candidate_only",
    })

def make_plan(pipeline_dir: str | Path, max_frames: int = 48) -> dict[str, Any]:
    """Preserve shortest inclusive OK-NO_MATCH-OK selection."""
    _integer(max_frames, "max_frames", 3)
    _require(max_frames <= 96, "max_frames must be <= 96")
    ordered, bindings = _validated_evidence(pipeline_dir)
    choices = _candidates(ordered, max_frames, {"NO_MATCH"})
    if not choices:
        return {"status": "NOT_APPLICABLE", "reason": "no_bounded_OK_NO_MATCH_OK_interval",
                "max_frames": max_frames, "seed_quality": "NOT_EVALUATED"}
    start, end, interval = min(choices, key=lambda c: (c[1]["decode_index"] - c[0]["decode_index"], c[0]["decode_index"]))
    return _interval_plan(start, end, interval, bindings, max_frames)


def _candidates(ordered, max_frames, statuses):
    seeds = [row for row in ordered if row["original_status"] == "OK"]
    return [(start, end, [row for row in ordered if start["decode_index"] <= row["decode_index"] <= end["decode_index"]])
            for start, end in zip(seeds, seeds[1:])
            if end["decode_index"] - start["decode_index"] + 1 <= max_frames
            and any(start["decode_index"] < row["decode_index"] < end["decode_index"]
                    and row["original_status"] in statuses for row in ordered)]


def make_coverage_plan(pipeline_dir: str | Path, max_frames: int = 96,
                       max_intervals: int = 6, max_total_frames: int = 384) -> dict[str, Any]:
    """Plan bounded review candidates; ledger coverage never asserts model success."""
    _integer(max_frames, "max_frames", 3)
    _require(max_frames <= 96, "max_frames must be <= 96")
    _integer(max_intervals, "max_intervals", 1)
    _require(max_intervals <= 6, "max_intervals must be <= 6")
    _integer(max_total_frames, "max_total_frames", 1)
    _require(max_total_frames <= 384, "max_total_frames must be <= 384")
    ordered, bindings = _validated_evidence(pipeline_dir)
    statuses = {"NO_MATCH", "NEEDS_SELECTION"}
    choices = _candidates(ordered, max_frames, statuses)
    choices.sort(key=lambda c: (-sum(r["original_status"] in statuses for r in c[2]), c[0]["decode_index"]))
    selected, total = [], 0
    for start, end, rows in choices:
        count = end["decode_index"] - start["decode_index"] + 1
        if len(selected) < max_intervals and total + count <= max_total_frames:
            selected.append((start, end, rows))
            total += count
    selected.sort(key=lambda c: c[0]["decode_index"])
    intervals = [dict(_interval_plan(a, b, rows, bindings, max_frames), interval_id=f"clip_{i:03d}")
                 for i, (a, b, rows) in enumerate(selected)]
    ledger = []
    for row in ordered:
        memberships = [p["interval_id"] for p in intervals
                       if p["start_decode_index"] <= row["decode_index"] <= p["end_decode_index"]]
        bounded = any(a["decode_index"] <= row["decode_index"] <= b["decode_index"] for a, b, _ in choices)
        if row["original_status"] == "OK":
            classification, reason = "source_OK", None
        elif memberships:
            classification, reason = "selected_interval", None
        elif bounded:
            classification, reason = "budget_deferred", "interval_or_total_frame_budget"
        else:
            classification = "unbounded_gap"
            before = any(r["original_status"] == "OK" and r["decode_index"] < row["decode_index"] for r in ordered)
            after = any(r["original_status"] == "OK" and r["decode_index"] > row["decode_index"] for r in ordered)
            reason = "no_bracket_start" if not before else "no_bracket_end" if not after else "exceeds_max_frames"
        ledger.append(dict(row, coverage=classification, reason=reason, interval_ids=memberships,
                           propagation_status="NOT_RUN", candidate_quality="NOT_EVALUATED"))
    unique = len({i for p in intervals for i in range(p["start_decode_index"], p["end_decode_index"] + 1)})
    counts = {key: sum(r["coverage"] == key for r in ledger)
              for key in ("source_OK", "selected_interval", "unbounded_gap", "budget_deferred")}
    return dict(bindings, schema="video-mask-coverage-plan-v1", status="PLANNED" if intervals else "NOT_APPLICABLE",
                max_frames=max_frames, max_intervals=max_intervals, max_total_frames=max_total_frames,
                candidate_interval_count=len(choices), intervals=intervals, coverage_ledger=ledger,
                sampled_frame_count=len(ordered), coverage_counts=counts,
                sum_frame_count=total, unique_decode_frame_count=unique, repeated_decode_frame_count=total-unique,
                unresolved_outside_count=counts["unbounded_gap"] + counts["budget_deferred"],
                seed_quality="NOT_EVALUATED", candidate_quality="NOT_EVALUATED", propagation_status="NOT_RUN",
                original_masks_preserved=True, output_classification="review_candidate_only")
