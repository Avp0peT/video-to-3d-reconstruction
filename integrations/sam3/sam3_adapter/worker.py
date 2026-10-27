"""Bounded worker: the single controlled execution mechanism for real work.

The parent (``runner``) owns the budgets, the stop latch and the manifest; the
heavy load and the per-frame forward passes happen in a child of this task. One
child serves the whole batch, so a single load is reused across frames.

Two properties are load-bearing:

* **bounded reclaim.** Waiting, pipe handling, ``kill`` and the final ``wait``
  all share one reclaim deadline. There is no unbounded ``communicate()``: the
  child's stdout/stderr go to files, and every wait is a bounded poll. A member
  that has not been *proved* gone is recorded as residual/UNKNOWN -- a
  successful ``kill`` call is not evidence of exit.
* **kill scope.** Only the process we started and its descendants. We never
  target other processes and never terminate "all Python".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from itertools import count
from pathlib import Path

JOB_SCHEMA = "sam3-adapter-worker-job-v1"

#: per-process monotonic event sequence: event identity is the sequence number,
#: never the second-resolution timestamp
_SEQUENCE = count(1)

EVENT_READY = "ready"
EVENT_STAGE = "stage"
EVENT_LOAD_ERROR = "load_error"
EVENT_FRAME_BEGIN = "frame_begin"
EVENT_FRAME = "frame"
EVENT_SAMPLE = "sample"
EVENT_STOP = "stop"
EVENT_DONE = "done"
EVENT_FATAL = "fatal"

#: exit codes used by the child, aligned with the contract
CHILD_EXIT = {
    "SUCCESS": 0, "PARTIAL": 1, "INPUT_ERROR": 2, "BACKEND_UNAVAILABLE": 3,
    "AUTH_REQUIRED": 3, "BACKEND_ERROR": 4, "RESOURCE_LIMIT": 5,
}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# job / event transport
# --------------------------------------------------------------------------- #
def write_job(path: str | Path, payload: dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["schema"] = JOB_SCHEMA
    text = json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return text


def read_job(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != JOB_SCHEMA:
        raise ValueError(f"job file is not a {JOB_SCHEMA} document")
    return payload


def append_event(path: str | Path, payload: dict) -> None:
    """Append one event line and flush, so the parent can tail it immediately."""
    body = dict(payload)
    body["seq"] = next(_SEQUENCE)
    body.setdefault("at_utc", _now_utc())
    body.setdefault("t", round(time.monotonic(), 6))
    line = json.dumps(body, ensure_ascii=False, allow_nan=False) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path: str | Path) -> list[dict]:
    """Read whatever events exist; a torn final line is ignored, not fatal."""
    target = Path(path)
    if not target.is_file():
        return []
    events: list[dict] = []
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def read_events_strict(path: str | Path, allow_trailing_partial: bool) -> tuple[list[dict], list[dict]]:
    """Read events and report malformed lines instead of silently skipping them.

    ``allow_trailing_partial`` is for the live window, when the last line may
    still be mid-write. Once the writer has exited, an unparsable line is
    incomplete evidence and is returned as a problem.
    """
    target = Path(path)
    if not target.is_file():
        return [], [{"code": "EVENT_FILE_MISSING", "message": f"no event file at {target}"}]
    events: list[dict] = []
    problems: list[dict] = []
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            is_last = index == len(lines)
            problems.append({
                "code": "EVENT_RECORD_CORRUPT",
                "message": f"line {index} is not valid JSON: {exc}",
                "trailing_partial": bool(is_last and allow_trailing_partial),
                "tolerated": bool(is_last and allow_trailing_partial),
            })
            continue
        if not isinstance(payload, dict):
            problems.append({"code": "EVENT_RECORD_NOT_OBJECT",
                             "message": f"line {index} is {type(payload).__name__}",
                             "trailing_partial": False, "tolerated": False})
            continue
        events.append(payload)
    return events, problems


def classify_load_error(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "AuthRequired" or "AuthRequired" in repr(exc):
        return "AUTH_REQUIRED"
    if name == "BackendUnavailable" or "BackendUnavailable" in repr(exc):
        return "BACKEND_UNAVAILABLE"
    if name == "ResourceLimit" or "ResourceLimit" in repr(exc):
        return "RESOURCE_LIMIT"
    return "BACKEND_ERROR"


# --------------------------------------------------------------------------- #
# bounded process management
# --------------------------------------------------------------------------- #
def launch_worker(
    python: str,
    job_path: str | Path,
    adapter_dir: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    extra_env: dict | None = None,
    cwd: str | Path | None = None,
    module: str = "sam3_adapter.worker",
) -> subprocess.Popen:
    """Start the child with its streams redirected to files (never to memory)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(adapter_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    # the child runs with cwd=adapter_dir, so the job path must be absolute
    absolute_job = str(Path(job_path).resolve())
    stdout_handle = open(stdout_path, "w", encoding="utf-8", newline="\n")
    stderr_handle = open(stderr_path, "w", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(
            [python, "-m", module, "--job", absolute_job],
            cwd=str(cwd) if cwd is not None else str(adapter_dir),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creation_flags,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    return process


def wait_bounded(process: subprocess.Popen, deadline: float, interval: float = 0.02) -> dict:
    """Poll until exit or the absolute deadline; never blocks indefinitely."""
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            return {"exited": True, "returncode": returncode, "timed_out": False}
        time.sleep(interval)
    returncode = process.poll()
    return {"exited": returncode is not None, "returncode": returncode, "timed_out": True}


def kill_tree(process: subprocess.Popen, deadline: float, interval: float = 0.02) -> dict:
    """Terminate the child's own process tree within one shared reclaim deadline."""
    record = {
        "pid": process.pid,
        "tree_kill_attempted": False,
        "tree_kill_returncode": None,
        "escalated_to_kill": False,
        "exited": process.poll() is not None,
        "returncode": process.poll(),
        "residual_unknown": False,
        "reclaim_deadline_monotonic": round(deadline, 6),
    }
    if record["exited"]:
        return record
    if os.name == "nt":
        record["tree_kill_attempted"] = True
        remaining = max(deadline - time.monotonic(), 0.1)
        try:
            completed = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=min(3.0, remaining), check=False,
            )
            record["tree_kill_returncode"] = completed.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            record["tree_kill_returncode"] = f"{type(exc).__name__}"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(interval)
    if process.poll() is None:
        record["escalated_to_kill"] = True
        try:
            process.kill()
        except OSError as exc:  # pragma: no cover
            record["kill_error"] = f"{type(exc).__name__}: {exc}"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(interval)
    record["exited"] = process.poll() is not None
    record["returncode"] = process.poll()
    # a successful kill call is NOT proof of exit
    record["residual_unknown"] = not record["exited"]
    return record


def run_bounded(
    argv: list[str],
    deadline_seconds: float,
    cwd: str | Path | None = None,
    env: dict | None = None,
    log_dir: str | Path | None = None,
    reclaim_seconds: float = 5.0,
) -> dict:
    """Run ``argv`` with a hard wall-clock bound and a bounded reclaim.

    Streams go to files under ``log_dir`` (or DEVNULL), so nothing unbounded is
    ever held in memory and no unbounded ``communicate()`` is used.
    """
    started = time.monotonic()
    deadline = started + deadline_seconds
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    stdout_path = stderr_path = None
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stdout_path = Path(log_dir) / f"stdout_{int(started * 1000)}.txt"
        stderr_path = Path(log_dir) / f"stderr_{int(started * 1000)}.txt"
        stdout_handle = open(stdout_path, "w", encoding="utf-8", newline="\n")
        stderr_handle = open(stderr_path, "w", encoding="utf-8", newline="\n")
    else:
        stdout_handle = subprocess.DEVNULL
        stderr_handle = subprocess.DEVNULL
    try:
        process = subprocess.Popen(argv, cwd=str(cwd) if cwd is not None else None, env=env,
                                   stdout=stdout_handle, stderr=stderr_handle,
                                   creationflags=creation_flags)
    finally:
        for handle in (stdout_handle, stderr_handle):
            if hasattr(handle, "close"):
                handle.close()
    record = {
        "argv": list(argv),
        "deadline_seconds": deadline_seconds,
        "pid": process.pid,
        "timed_out": False,
        "returncode": None,
        "wall_seconds": 0.0,
        "kill": None,
        "stdout_path": str(stdout_path) if stdout_path else None,
        "stderr_path": str(stderr_path) if stderr_path else None,
    }
    outcome = wait_bounded(process, deadline)
    if not outcome["exited"]:
        record["timed_out"] = True
        record["kill"] = kill_tree(process, time.monotonic() + reclaim_seconds)
    record["returncode"] = process.poll()
    record["wall_seconds"] = round(time.monotonic() - started, 6)
    record["exited"] = process.poll() is not None
    record["residual_unknown"] = not record["exited"]
    return record


# --------------------------------------------------------------------------- #
# child side
# --------------------------------------------------------------------------- #
def child_main(job_path: str | Path) -> int:
    """Run one job inside this child process; every event is appended to a file."""
    from . import imagery, resources, runner  # noqa: PLC0415 - local to the child

    job = read_job(job_path)
    events_path = job["events_path"]
    run_dir = Path(job["run_dir"])
    append_event(events_path, {"type": EVENT_READY, "pid": os.getpid(),
                               "python": sys.version.split()[0], "executable": sys.executable})

    # Validate visual inputs and the job's actual image paths before monitoring,
    # model resolution or loading. Rebind against T01, not only parent metadata.
    payload = dict(job["request"])
    payload["explicit_selection"] = {int(k): v
                                     for k, v in (payload.get("explicit_selection") or {}).items()}
    request = runner.RunRequest(**payload)
    from . import sourceio
    from .visual_prompt import VisualPromptError, validate_visual_prompt, validate_bound_frame
    try:
        visual = validate_visual_prompt(request.visual_prompt, request.indices, request.text_prompt)
        if visual is not None:
            source = sourceio.load_source_io(request.io_run)
            items = job["frames"]
            if ([r["input_index"] for r in items] != request.indices
                    or any(type(r["input_index"]) is not int for r in items)):
                raise VisualPromptError("worker box frame membership differs from request")
            for item in items:
                frame = sourceio.resolve_frame(source, item["input_index"])
                validate_bound_frame(visual, frame)
                if (Path(item["source_image"]).resolve() != frame.source_image.resolve()
                        or type(item["width"]) is not int or type(item["height"]) is not int
                        or (item["width"], item["height"]) != (frame.width, frame.height)):
                    raise VisualPromptError("worker box image path or dimensions differ from T01")
            request.visual_prompt = visual
    except (VisualPromptError, sourceio.SourceIoError, KeyError, TypeError, ValueError) as exc:
        append_event(events_path, {"type": EVENT_LOAD_ERROR, "kind": "INPUT_ERROR",
                                   "stage": "visual_prompt", "message": str(exc)})
        return CHILD_EXIT["INPUT_ERROR"]

    monitor = None
    if job.get("monitor_device"):
        try:
            monitor = resources.ResourceMonitor(policy=resources.ResourcePolicy(
                ram_commit_limit_bytes=job.get("ram_limit_bytes", 12 << 30),
                vram_limit_bytes=job.get("vram_limit_bytes", 7 << 30),
                cpu_threads=job.get("cpu_threads", 2),
            ), device=job["monitor_device"])
        except Exception as exc:  # noqa: BLE001
            monitor = None
            append_event(events_path, {"type": EVENT_FATAL, "kind": "BACKEND_ERROR",
                                       "stage": "monitor_init", "message": f"{type(exc).__name__}: {exc}"})
            return CHILD_EXIT["BACKEND_ERROR"]

    def sample(stage: str, input_index: int | None = None) -> list[str]:
        if monitor is None:
            return []
        try:
            payload = monitor.sample()
        except Exception as exc:  # noqa: BLE001 - a failed sample is not safety
            append_event(events_path, {"type": EVENT_SAMPLE, "stage": stage, "failed": True,
                                       "message": f"{type(exc).__name__}: {exc}"})
            return [f"monitor sample failed at {stage}: {type(exc).__name__}: {exc}"]
        payload.update({"type": EVENT_SAMPLE, "stage": stage, "failed": False,
                        "input_index": input_index, "pid": os.getpid(),
                        "scope": "this worker process only"})
        append_event(events_path, payload)
        try:
            return monitor.violations(payload)
        except Exception as exc:  # noqa: BLE001
            append_event(events_path, {"type": EVENT_SAMPLE, "stage": stage, "failed": True,
                                       "message": f"violations() raised {type(exc).__name__}: {exc}"})
            return [f"violations() raised {type(exc).__name__}: {exc}"]

    def stop(kind: str, reason: str, stage: str, input_index: int | None = None) -> None:
        append_event(events_path, {"type": EVENT_STOP, "kind": kind, "reason": reason,
                                   "stage": stage, "input_index": input_index, "pid": os.getpid()})

    # --------------------------------------------------------------- backend
    try:
        backend = runner._build_backend(request)
        resolver = getattr(backend, "resolve_weights", None)
        if resolver is not None:
            resolver()
    except Exception as exc:  # noqa: BLE001
        kind = classify_load_error(exc)
        append_event(events_path, {"type": EVENT_LOAD_ERROR, "kind": kind, "stage": "resolve_weights",
                                   "message": f"{type(exc).__name__}: {exc}"})
        return CHILD_EXIT.get(kind, 4)

    for reason in sample("pre_load"):
        stop("RESOURCE_LIMIT", reason, "pre_load")
        append_event(events_path, {"type": EVENT_DONE, "stopped": True})
        return CHILD_EXIT["RESOURCE_LIMIT"]

    append_event(events_path, {"type": EVENT_STAGE, "stage": "load", "state": "begin"})
    try:
        backend.load()
    except Exception as exc:  # noqa: BLE001
        kind = classify_load_error(exc)
        append_event(events_path, {"type": EVENT_LOAD_ERROR, "kind": kind, "stage": "load",
                                   "message": f"{type(exc).__name__}: {exc}"})
        return CHILD_EXIT.get(kind, 4)
    append_event(events_path, {"type": EVENT_STAGE, "stage": "load", "state": "end"})
    for reason in sample("post_load"):
        stop("RESOURCE_LIMIT", reason, "post_load")
        append_event(events_path, {"type": EVENT_DONE, "stopped": True})
        return CHILD_EXIT["RESOURCE_LIMIT"]

    # ----------------------------------------------------------------- frames
    completed_ok = 0
    attempted = 0
    try:
        for item in job["frames"]:
            index = item["input_index"]
            append_event(events_path, {"type": EVENT_FRAME_BEGIN, "input_index": index, "pid": os.getpid()})
            try:
                candidates = backend.segment(Path(item["source_image"]), request.text_prompt,
                                            request.object_id, item["width"], item["height"], index)
            except Exception as exc:  # noqa: BLE001
                # Resource failures remain distinct through the worker protocol;
                # a generic frame handler must not turn a true OOM into exit 4.
                kind = "RESOURCE_LIMIT" if isinstance(exc, runner.ResourceLimit) else "BACKEND_ERROR"
                append_event(events_path, {"type": EVENT_FRAME, "input_index": index, "status": kind,
                                           "candidates": [], "errors": [
                                               {"code": kind,
                                                "message": f"{type(exc).__name__}: {exc}"}]})
                stop(kind, f"frame[{index}] failed: {type(exc).__name__}: {exc}", "frame", index)
                append_event(events_path, {"type": EVENT_DONE, "stopped": True})
                return CHILD_EXIT[kind]
            attempted += 1
            try:
                records = runner._write_candidates(candidates, run_dir, index)
                dropped = sum(1 for candidate in candidates
                              if imagery.foreground_pixels(candidate.flat) == 0)
                status, chosen, problem = runner._select_index(
                    candidates, request.explicit_selection.get(index))
                errors = [problem] if problem else []
                frame_record = {
                    "type": EVENT_FRAME, "input_index": index, "status": status,
                    "candidates": records, "candidates_dropped_empty": dropped,
                    "selected_candidate_id": None, "selected_mask": None, "selection": None,
                    "errors": errors, "not_run_reason": None,
                }
                if chosen is not None:
                    selected = candidates[chosen]
                    frame_record["selected_candidate_id"] = selected.candidate_id
                    frame_record["selection"] = {
                        "method": "explicit_candidate" if index in request.explicit_selection
                        else "unique_candidate",
                        "source": "cli_argument" if index in request.explicit_selection else "automatic",
                        "mask_source": runner.MASK_SOURCE,
                    }
                    mask_rel = records[chosen]["mask_path"]
                    frame_record["selected_mask"] = {
                        "path": mask_rel,
                        "sha256": imagery.sha256_file(run_dir / mask_rel),
                    }
                    imagery.write_overlay(Path(item["source_image"]), selected.flat,
                                          selected.width, selected.height,
                                          run_dir / "overlays" / f"{index:06d}.png")
            except Exception as exc:  # noqa: BLE001 - artifacts count as backend failure
                append_event(events_path, {"type": EVENT_FRAME, "input_index": index,
                                           "status": "BACKEND_ERROR", "candidates": [],
                                           "errors": [{"code": "ARTIFACT_WRITE_FAILED",
                                                       "message": f"{type(exc).__name__}: {exc}"}]})
                stop("BACKEND_ERROR", f"artifact write failed for frame[{index}]: {exc}", "frame", index)
                append_event(events_path, {"type": EVENT_DONE, "stopped": True})
                return CHILD_EXIT["BACKEND_ERROR"]
            append_event(events_path, frame_record)
            if status == "OK":
                completed_ok += 1
            for reason in sample("post_frame", index):
                stop("RESOURCE_LIMIT", reason, "post_frame", index)
                append_event(events_path, {"type": EVENT_DONE, "stopped": True,
                                           "frames_attempted": attempted})
                return CHILD_EXIT["RESOURCE_LIMIT"]
    finally:
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 - cleanup failure is recorded, never hidden
            append_event(events_path, {"type": EVENT_STAGE, "stage": "close", "state": "error",
                                       "message": f"{type(exc).__name__}: {exc}"})

    append_event(events_path, {"type": EVENT_DONE, "stopped": False, "frames_attempted": attempted,
                               "frames_ok": completed_ok})
    return CHILD_EXIT["SUCCESS"]


# --------------------------------------------------------------------------- #
# self-test entry (fixtures only)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="bounded worker")
    parser.add_argument("--job", default=None, help="run a runner job in this process")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--deadline", type=float, default=5.0)
    parser.add_argument("--spawn-child", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.job:
        return child_main(args.job)

    if args.spawn_child:
        child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({args.sleep})"])
        time.sleep(args.sleep)
        child.wait()
    else:
        time.sleep(args.sleep)
    record = {"slept": args.sleep, "deadline": args.deadline, "child_spawned": args.spawn_child}
    if args.out:
        Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8", newline="\n")
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
