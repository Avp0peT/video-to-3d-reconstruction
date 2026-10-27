"""Task-local supervisor checks; one short Python child, no models or network."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

if os.name != "nt":
    pytest.skip("Windows Job Object supervisor", allow_module_level=True)

from integrations.demo_pipeline import supervision as sup


def arguments(tmp_path):
    return dict(cwd=tmp_path, log_path=tmp_path / "events.jsonl",
                stdout_path=tmp_path / "stdout.txt", stderr_path=tmp_path / "stderr.txt",
                poll_seconds=0.02, reclamation_seconds=1.0, wall_clock_seconds=4.0)


@pytest.fixture
def fake_native(monkeypatch):
    """Keep the real supervision and reclaim decision logic, fake Win32 calls."""
    calls = []
    state = {"cancel": False, "exited": False, "exit_code": 37, "reclaim_errors": []}
    proc = SimpleNamespace(pid=31415, process_handle="root", thread_handle="thread",
                           in_job=False, resumed=False, closed=False)
    proc.as_dict = lambda: {"pid": proc.pid, "in_job": proc.in_job,
                           "resumed": proc.resumed, "closed": proc.closed}

    def close_handles():
        calls.append("close_process")
        proc.closed = True

    proc.close_handles = close_handles

    class Job:
        def set_limits(self, **kwargs):
            calls.append("limits")
            return kwargs

        def assign(self, handle):
            assert handle == "root"
            calls.append("assign")

        def terminate(self, exit_code):
            calls.append("terminate_job")
            state["exited"] = True
            return {"terminated": True}

        def accounting(self):
            return {"active_processes": int(not state["exited"]), "total_processes": 1}

        def extended_limit_information(self):
            return {"peak_job_memory_used_bytes": 0}

        def close(self):
            calls.append("close_job")

    def create(*args):
        calls.append("create")
        state["cancel"] = True
        return proc

    def terminate(handle, **kwargs):
        assert handle == "root"
        calls.append("terminate_root")
        state["exited"] = True
        return {"terminated": True, "confirmed_exit_code": state["exit_code"],
                "errors": state["reclaim_errors"]}

    def wait(handle, timeout_ms):
        assert handle == "root"
        calls.append("wait")
        return {"exited": state["exited"],
                "exit_code": state["exit_code"] if state["exited"] else None,
                "error": None if state["exited"] else "timeout"}

    def resume(handle):
        assert handle == "thread"
        calls.append("resume")
        state["exited"] = True
        state["exit_code"] = 0
        return 1

    monkeypatch.setattr(sup, "JobObject", Job)
    monkeypatch.setattr(sup, "create_suspended", create)
    monkeypatch.setattr(sup, "terminate_and_wait", terminate)
    monkeypatch.setattr(sup, "wait_for_exit", wait)
    monkeypatch.setattr(sup, "proc_creation_time_utc", lambda pid: "fixture")
    monkeypatch.setattr(sup, "kernel32", SimpleNamespace(ResumeThread=resume))
    return calls, state, proc


@pytest.mark.parametrize("reclaim_errors,confirmed", [([], True), (["fixture query failure"], False)])
def test_cancel_latched_during_create_reclaims_before_assignment(
        tmp_path, fake_native, reclaim_errors, confirmed):
    calls, state, proc = fake_native
    state["reclaim_errors"] = reclaim_errors
    result = sup.supervise(["fixture"], **arguments(tmp_path),
                           cancel_requested=lambda: state["cancel"])
    assert result["status"] == result["termination_reason"] == "user_cancelled"
    assert result["pid"] == proc.pid
    assert result["root_exit_code"] == 37
    assert result["reclamation"]["path"] == "own_handle_terminate_and_wait"
    assert result["reclaim_confirmed"] is confirmed
    assert result["reclamation"]["in_job"] is False
    assert "assign" not in calls and "resume" not in calls
    assert calls.index("terminate_root") < calls.index("wait") < calls.index("close_process")
    assert proc.closed and not proc.resumed


def test_default_callback_retains_normal_completion(tmp_path, fake_native):
    calls, state, proc = fake_native
    result = sup.supervise(["fixture"], **arguments(tmp_path))
    assert result["status"] == "completed"
    assert result["termination_reason"] == "normal_exit"
    assert result["root_exit_code"] == 0
    assert result["reclaim_confirmed"] is True
    assert "cancellation_exit_probe" not in result
    assert calls.index("assign") < calls.index("resume") < calls.index("terminate_job")
    assert proc.closed and proc.resumed


def test_cancel_after_assignment_never_resumes(tmp_path, fake_native):
    calls, state, proc = fake_native
    result = sup.supervise(["fixture"], **arguments(tmp_path),
                           cancel_requested=lambda: "assign" in calls)
    assert result["status"] == result["termination_reason"] == "user_cancelled"
    assert result["root_exit_code"] == 37
    assert result["reclamation"]["path"] == "job_tree"
    assert result["reclaim_confirmed"] is True
    assert "assign" in calls and "resume" not in calls
    assert calls.index("assign") < calls.index("terminate_job") < calls.index("close_process")
    assert proc.closed and not proc.resumed


def test_shared_cad_supervisor_remains_frozen():
    original = Path(__file__).resolve().parents[1] / "integrations/cad_delivery/supervise.py"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == (
        "077a4a021741d1b432d6312bcedbab1bc31696cc6621b66d61dc65dda0d7e4be")


def test_active_python_child_is_cancelled_and_reclaimed(tmp_path):
    # If cancellation breaks, this child exits itself after three seconds; the
    # supervisor also has a four-second deadline. It loads only Python stdlib.
    marker = tmp_path / "active.txt"
    script = tmp_path / "bounded_child.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "deadline = time.monotonic() + 3.0\n"
        "pathlib.Path(sys.argv[1]).write_text('active', encoding='utf-8')\n"
        "while time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n", encoding="utf-8")
    callback_observations = []

    def cancel_requested():
        active = marker.exists()
        callback_observations.append(active)
        return active

    result = sup.supervise([sys.executable, "-B", str(script), str(marker)],
                           **arguments(tmp_path), cancel_requested=cancel_requested)
    (tmp_path / "RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert callback_observations[-1] is True
    assert "resumed" in [item["kind"] for item in events]
    assert result["status"] == result["termination_reason"] == "user_cancelled"
    assert result["reclamation"]["path"] == "job_tree"
    assert result["reclaim_confirmed"] is True
    assert result["reclamation"]["all_members_dead"] is True
    assert result["root_exit_code"] == 1  # observed Windows TerminateJobObject code
    assert result["cancellation_exit_probe"]["exit_code"] == result["root_exit_code"]
    assert result["errors"] == []
    assert events[-1]["kind"] == "cleanup"
    assert events[-1]["reclaim_confirmed"] is True
