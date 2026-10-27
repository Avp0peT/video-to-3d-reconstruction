"""Explicit diagnostic worker profile layered around the sealed mmap worker.

Ordinary segmentation still uses mmap_review_worker. This profile observes
scores only; it never changes the prompt, selection, threshold or mask bytes.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

from . import runtime as rt

PROCESSOR = "integrations/sam3/src/sam3-repo/sam3/model/sam3_image_processor.py"
OWN_FILES = ("integrations/demo_pipeline/runtime.py",
             "integrations/demo_pipeline/sam_score_worker.py",
             "integrations/demo_pipeline/sam_score_diagnostic.py",
             "scripts/diagnose_sam3_scores.py")
POLICY = "sam3-actual-prefilter-score-observation-v1"


def configure(config, env):
    files = [{"path": str(config.root / rel), "sha256": rt.sha256(config.root / rel)}
             for rel in (*OWN_FILES, PROCESSOR)]
    binding = {"policy": POLICY, "files": files, "expected_indices": config.indices,
               "processor_path": str(config.root / PROCESSOR),
               "processor_sha256": files[-1]["sha256"],
               "sealed_worker": str(config.worker_package / "worker_entry/mmap_review_worker.py"),
               "sealed_worker_sha256": rt.sha256(config.worker_package / "worker_entry/mmap_review_worker.py"),
               "output": str(config.output_dir), "max_queries": 256,
               "max_record_bytes": 65536, "max_total_bytes": 4 << 20,
               "timing_comparison_valid": False,
               "intervention": "thread-local Python trace and bounded CPU score copies; may synchronize CUDA"}
    env["PYTHONPATH"] += os.pathsep + str(config.root)
    env["SAM3_SCORE_BINDING"] = str(config.output_dir / "SCORE_DIAGNOSTIC_BINDING.json")
    return binding


def _check_binding(binding):
    if binding["policy"] != POLICY:
        raise ValueError("score diagnostic policy mismatch")
    for row in binding["files"]:
        path = Path(row["path"])
        rt._no_links(path)
        if rt.sha256(path) != row["sha256"]:
            raise ValueError("score diagnostic binding drift: " + str(path))
    if rt.sha256(Path(binding["sealed_worker"])) != binding["sealed_worker_sha256"]:
        raise ValueError("sealed mmap worker drift")


def assess(config, binding):
    result = {"status": "INCOMPLETE", "observed_indices": [], "errors": []}
    try:
        _check_binding(binding)
        output = config.output_dir
        status = rt.read_json(output / "score_status.json")
        data = (output / "prefilter_scores.jsonl").read_bytes()
        if len(data) > binding["max_total_bytes"]:
            raise ValueError("score ledger exceeds bound")
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        result["observed_indices"] = [row["input_index"] for row in rows]
        result["worker_status"] = status
        if result["observed_indices"] != config.indices:
            raise ValueError("diagnostic frame sequence incomplete or changed")
        if not all(row.get("retained_scores_match") is True and
                   row.get("processor_source_sha256") == binding["processor_sha256"] and
                   row.get("threshold") == 0.5 for row in rows):
            raise ValueError("actual filter evidence failed")
        if (status.get("business_rc") != 0 or status.get("error") is not None or
                status.get("completed_indices") != config.indices or
                status.get("restored") is not True or status.get("trace_restored") is not True or
                status.get("events_sha256") != rt.sha256(output / "prefilter_scores.jsonl")):
            raise ValueError("worker diagnostic evidence incomplete")
        result["status"] = "COMPLETE"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def main(argv=None):
    from .sam_score_diagnostic import ScoreDiagnostic, DiagnosticError

    binding_path = Path(os.environ["SAM3_SCORE_BINDING"])
    binding = rt.read_json(binding_path)
    _check_binding(binding)
    output = Path(binding["output"])
    if binding_path.resolve() != (output / "SCORE_DIAGNOSTIC_BINDING.json").resolve():
        raise ValueError("score binding output mismatch")
    entry = importlib.import_module("mmap_review_worker")
    if Path(entry.__file__).resolve() != Path(binding["sealed_worker"]).resolve():
        raise ValueError("mmap worker came from another package")
    base = importlib.import_module("real3_review_worker")
    if entry.base is not base:
        raise ValueError("mmap audit module identity differs")
    original = base.AuditedBackend.segment
    expected = binding["expected_indices"]
    if not isinstance(expected, list) or not 1 <= len(expected) <= 96 or any(type(x) is not int for x in expected):
        raise ValueError("invalid diagnostic frame sequence")
    status = {"policy": POLICY, "business_rc": None, "error": None,
              "started_indices": [], "completed_indices": [], "restored": False,
              "trace_restored": False, "events_sha256": None}
    ledger = output / "prefilter_scores.jsonl"
    with ledger.open("x", encoding="utf-8", newline="\n") as stream:
        def sink(line):
            stream.write(line)
            stream.flush()
            status["completed_indices"].append(json.loads(line)["input_index"])

        diagnostic = ScoreDiagnostic(sink, expected_source_sha256=binding["processor_sha256"],
            max_frames=len(expected), max_queries=binding["max_queries"],
            max_record_bytes=binding["max_record_bytes"], max_total_bytes=binding["max_total_bytes"])

        def segment(self, *args, **kwargs):
            index = kwargs.get("input_index", args[5] if len(args) > 5 else None)
            position = len(status["started_indices"])
            if position >= len(expected) or index != expected[position]:
                raise DiagnosticError("diagnostic frame order differs from request")
            status["started_indices"].append(index)
            processor = getattr(self.inner, "_processor", None)
            if processor is None:
                raise DiagnosticError("loaded backend has no actual processor")
            with diagnostic.observe(processor, index):
                return original(self, *args, **kwargs)

        base.AuditedBackend.segment = segment
        failure = None
        try:
            status["business_rc"] = entry.main(argv)
        except BaseException as exc:
            failure = exc
            status["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            owned = base.AuditedBackend.segment is segment
            base.AuditedBackend.segment = original
            status["restored"] = owned and base.AuditedBackend.segment is original
            status["trace_restored"] = sys.gettrace() is None
            stream.flush()
            try:
                status["events_sha256"] = rt.sha256(ledger)
                _check_binding(binding)
                rt.write_json(output / "score_status.json", status)
            except Exception as exc:
                if failure is not None:
                    failure.add_note("score finalisation failed: " + str(exc))
                else:
                    raise
    if status["business_rc"] != 0:
        return status["business_rc"]
    return 0 if status["completed_indices"] == expected and status["restored"] and status["trace_restored"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
