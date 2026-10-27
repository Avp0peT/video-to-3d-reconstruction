"""Pure-data execute/_drive_worker regression; no model, OS query or child launch.

Run only through the guarded entry in
outputs/delegation/SAM3_FRAME_STATE/20261008_main/run_tests.py.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
# This suite exercises event reconciliation, never image decoding. Supply just
# hashing at the imagery boundary in a private package, without replacing the
# normal sam3_adapter modules or affecting other tests during collection.
PACKAGE = "_sam3_frame_state_fixture"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "integrations" / "sam3" / "sam3_adapter")]
sys.modules[PACKAGE] = package
imagery = ModuleType(PACKAGE + ".imagery")
imagery.sha256_bytes = lambda data: hashlib.sha256(data).hexdigest()
imagery.sha256_file = lambda path: imagery.sha256_bytes(Path(path).read_bytes())
sys.modules[imagery.__name__] = imagery

runner = importlib.import_module(PACKAGE + ".runner")
sourceio, worker = runner.sourceio, runner.worker
counts_consistent = importlib.import_module(PACKAGE + ".contract").counts_consistent
assert Path(runner.__file__).resolve() == ROOT / "integrations" / "sam3" / "sam3_adapter" / "runner.py"

FIRST_REASON = "worker_interpreter pid=4321: RAM commit 13285302272 B > limit 12884901888 B"


def frame(index, seq, status="OK"):
    selected = {"path": f"masks/{index:06d}_c00.png", "sha256": "a" * 64}
    return {"type": "frame", "seq": seq, "input_index": index, "status": status,
            "candidates": [{"candidate_id": "c00", "mask_path": selected["path"],
                            "mask_sha256": selected["sha256"], "foreground_pixels": 1,
                            "score": 0.9, "score_semantics": "fixture"}] if status == "OK" else [],
            "selected_candidate_id": "c00" if status == "OK" else None,
            "selected_mask": selected if status == "OK" else None,
            "selection": {"method": "unique_candidate", "source": "automatic",
                          "mask_source": "automatic_sam3"} if status == "OK" else None,
            "errors": ([{"code": "BACKEND_ERROR", "message": "frame failure"}]
                       if status == "BACKEND_ERROR" else []),
            "not_run_reason": None}


def stream():
    return [{"type": "ready", "seq": 1, "pid": 4321},
            {"type": "stage", "seq": 2, "stage": "load", "state": "end"},
            {"type": "frame_begin", "seq": 3, "input_index": 17}, frame(17, 4),
            {"type": "frame_begin", "seq": 5, "input_index": 34}]


def run_case(tmp_path, monkeypatch, *, live=None, final=None, latch="RESOURCE_LIMIT",
             raise_after_start=False, close_error=False, orphan=False, torn=False):
    live = stream() if live is None else live
    final = live if final is None else final
    io = tmp_path / "io"
    io.mkdir()
    (io / "run.json").write_bytes(b"{}")
    (io / "frames.jsonl").write_bytes(b"[]")
    source = sourceio.SourceIo(io, "fixture", b"{}", b"[]")
    resolved = {}
    for index in (17, 34, 59):
        image = io / f"{index}.raw"
        image.write_bytes(bytes([index]))
        digest = imagery.sha256_file(image)
        resolved[index] = sourceio.SourceFrame(index, f"frame-{index}", image.name,
                                               image, digest, digest, 1, 1)
    monkeypatch.setattr(sourceio, "load_source_io", lambda path: source)
    monkeypatch.setattr(sourceio, "resolve_frame", lambda source, index: resolved[index])
    backend = SimpleNamespace(describe=lambda: {"name": "fake"}, bindings=lambda: {})

    def close():
        if close_error:
            raise RuntimeError("later cleanup failure")

    backend.close = close
    monkeypatch.setattr(runner, "_build_backend", lambda request: backend)
    monkeypatch.setattr(runner, "ResourceMonitor", lambda **kwargs: SimpleNamespace(
        channel_report=lambda: {"process_commit_bytes": 1024}))
    # Metadata gathering must not reach platform's possible subprocess probes.
    monkeypatch.setattr(runner.platform, "platform", lambda: "offline-fixture")
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "offline-fixture")
    process = SimpleNamespace(pid=4321, poll=lambda: 0)
    trace = []

    def launch(python, job_path, *args, **kwargs):
        trace.append("fake_launch")
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        events_path = Path(job["events_path"])
        data = "".join(json.dumps(event) + "\n" for event in final)
        events_path.write_text(data + ('{"type":"frame"' if torn else ""), encoding="utf-8")
        if orphan:
            mask = Path(job["run_dir"]) / "masks" / "000034_c00.png"
            mask.parent.mkdir()
            mask.write_bytes(b"orphan artifact must never be inspected or selected")
        return process

    class Ownership:
        def __init__(self, process, *args, **kwargs):
            self.process = process

        def note_ready(self, event):
            pass

        def members(self):
            return {"unresolved_members": [], "enumeration_available": True, "members": []}

        def identity(self):
            return {"pid": 4321, "fixture": True}

        def reclaim(self, reason, **kwargs):
            trace.append("fake_reclaim")
            return {"state": "COMPLETE", "exited": True, "residual_unknown": []}

        def close_handles(self):
            return {"errors": []}

    class Supervisor:
        def __init__(self, events_path, owner, ledger, stop_state, *args, **kwargs):
            self.ledger, self.stop_state = ledger, stop_state
            self._stop_event = threading.Event()

        def start(self):
            trace.append("live_apply")
            self.ledger.apply_all(live)
            if latch:
                self.stop_state.latch(latch, FIRST_REASON if latch == "RESOURCE_LIMIT"
                                      else "worker protocol failure", "fixture_observer")
                # A subsequent observation must never replace the first cause.
                self.stop_state.latch("BACKEND_ERROR", "later failure", "fixture_cleanup")
            if raise_after_start:
                raise RuntimeError("injected supervision failure after frame begin")

        def stop(self, *args):
            trace.append("supervisor_stop")
            return {"joined_cleanly": True}

    monkeypatch.setattr(worker, "launch_worker", launch)
    monkeypatch.setattr(runner.observer_module, "WorkerOwnership", Ownership)
    monkeypatch.setattr(runner, "_WorkerSupervisor", Supervisor)
    read_events = worker.read_events_strict

    def read(path, allow_trailing_partial):
        assert allow_trailing_partial is False
        assert "supervisor_stop" in trace
        trace.append("final_drain")
        return read_events(path, allow_trailing_partial)

    monkeypatch.setattr(worker, "read_events_strict", read)
    request = runner.RunRequest(io_run=io, indices=[17, 34, 59], object_id="fixture",
                                text_prompt="fixture", output_root=tmp_path / "out",
                                repo_dir=ROOT / "integrations" / "sam3", backend_kind="fake",
                                device="cpu", use_worker=True, monitor_interval_seconds=0)
    outcome = runner.execute(request)
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == outcome.manifest
    assert counts_consistent(manifest["counts"], manifest["frames"]) == []
    assert trace.count("fake_launch") == trace.count("final_drain") == 1
    return outcome, manifest


def assert_uncommitted(manifest, status):
    record = manifest["frames"][1]
    assert record["status"] == status and record["not_run_reason"] is None
    assert record["candidates"] == []
    assert record["selected_candidate_id"] is record["selected_mask"] is record["selection"] is None
    assert record["errors"][0]["code"] == "FRAME_RESULT_UNCOMMITTED"
    return record


@pytest.mark.parametrize("raise_after_start", [False, True])
def test_started_uncommitted_preserves_first_cause_and_orphan(tmp_path, monkeypatch, raise_after_start):
    outcome, manifest = run_case(tmp_path, monkeypatch, orphan=True, close_error=True,
                                 raise_after_start=raise_after_start)
    assert [row["status"] for row in manifest["frames"]] == ["OK", "RESOURCE_LIMIT", "NOT_RUN"]
    assert outcome.status == "RESOURCE_LIMIT" and outcome.exit_code == manifest["execution"]["exit_code"] == 5
    record = assert_uncommitted(manifest, "RESOURCE_LIMIT")
    assert FIRST_REASON in record["errors"][0]["message"]
    assert manifest["stop_state"]["reason"] == manifest["frames"][2]["not_run_reason"] == FIRST_REASON
    assert (outcome.run_dir / "masks" / "000034_c00.png").read_bytes().startswith(b"orphan")
    assert manifest["counts"] == dict(requested=3, success=1, no_match=0,
                                       needs_selection=0, error=1, not_run=1)
    if not raise_after_start:
        assert manifest["worker"]["ledger_summary"]["frames_seen"] == [17]


@pytest.mark.parametrize("status", ["OK", "NO_MATCH", "NEEDS_SELECTION", "BACKEND_ERROR"])
def test_final_drain_committed_result_wins_after_latch(tmp_path, monkeypatch, status):
    late = frame(34, 6, status)
    outcome, manifest = run_case(tmp_path, monkeypatch, final=stream() + [late, late])
    record = manifest["frames"][1]
    assert record["status"] == status
    assert record["selected_mask"] == late["selected_mask"]
    assert record["errors"] == late["errors"]
    assert manifest["worker"]["ledger_summary"]["frames_seen"] == [17, 34]
    assert outcome.exit_code == 5 and manifest["stop_state"]["reason"] == FIRST_REASON
    assert manifest["frames"][2]["status"] == "NOT_RUN"


def test_begin_seen_only_in_final_drain_is_still_started(tmp_path, monkeypatch):
    _, manifest = run_case(tmp_path, monkeypatch, live=stream()[:-1], final=stream())
    assert_uncommitted(manifest, "RESOURCE_LIMIT")


@pytest.mark.parametrize("raise_after_start", [False, True])
def test_missing_result_without_resource_stop_is_backend_error(tmp_path, monkeypatch, raise_after_start):
    outcome, manifest = run_case(tmp_path, monkeypatch, latch=None, raise_after_start=raise_after_start)
    assert outcome.status == "BACKEND_ERROR" and outcome.exit_code == 4
    assert_uncommitted(manifest, "BACKEND_ERROR")
    assert manifest["frames"][0]["status"] == "OK"
    assert manifest["frames"][2]["status"] == "NOT_RUN"


def test_torn_final_result_cannot_publish_success(tmp_path, monkeypatch):
    outcome, manifest = run_case(tmp_path, monkeypatch, torn=True)
    assert_uncommitted(manifest, "RESOURCE_LIMIT")
    assert outcome.exit_code == 5
    assert any(error["code"] == "EVENT_RECORD_CORRUPT" for error in manifest["errors"])


def test_limit_after_all_results_preserves_completed_frames(tmp_path, monkeypatch):
    events = stream() + [frame(34, 6), {"type": "frame_begin", "seq": 7, "input_index": 59}, frame(59, 8)]
    outcome, manifest = run_case(tmp_path, monkeypatch, live=events)
    assert [row["status"] for row in manifest["frames"]] == ["OK", "OK", "OK"]
    assert outcome.exit_code == 5 and manifest["counts"]["success"] == 3


def test_no_begin_and_orphan_file_remain_not_run(tmp_path, monkeypatch):
    outcome, manifest = run_case(tmp_path, monkeypatch, live=stream()[:2], orphan=True)
    assert [row["status"] for row in manifest["frames"]] == ["NOT_RUN"] * 3
    assert all(row["not_run_reason"] == FIRST_REASON for row in manifest["frames"])
    assert outcome.exit_code == 5 and manifest["counts"]["not_run"] == 3
