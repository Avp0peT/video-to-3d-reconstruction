"""Run orchestration for the object-mask-v1 contract.

Execution modes
---------------
``use_worker=False`` (in-process) is kept for deterministic interface tests.
``use_worker=True`` is the **product** path used by the CLI: the parent owns the
budgets, the stop latch, the stage bookkeeping and the manifest, while the heavy
model load and every per-frame forward pass happen inside one bounded worker
child of this task -- one child per batch, so a single load serves all frames.

Control rules
-------------
* every stop cause is recorded in one irrevocable :class:`StopState`; the first
  latch wins, so a sample that falls back below the threshold, or a later
  cleanup failure, cannot clear or overwrite it;
* a frame is never started once the stop latch is set, and the parent terminates
  the worker's own process tree within one shared reclaim deadline;
* the manifest is always written: every extra query at the end is wrapped, its
  failure recorded, and the last known snapshot / an explicit unknown is used
  instead;
* failures are never downgraded to ``NO_MATCH`` and cleanup errors never
  overwrite the real stop reason.
"""

from __future__ import annotations

import json
import platform
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import imagery, observer as observer_module, sourceio, worker
from .backends import (
    AuthRequired,
    BackendError,
    BackendUnavailable,
    FakeBackend,
    ResourceLimit,
    Sam3Backend,
)
from .contract import (
    CONTRACT_VERSION,
    EVIDENCE_LEVEL_FAKE,
    EVIDENCE_LEVEL_REAL,
    EXIT_CODES,
    IMAGE_DOMAIN,
    MASK_SOURCE,
    MONITOR_INTERVAL_DEFAULT_SECONDS,
    NOT_EVALUATED,
    SAM3_BACKEND_NAME,
    SAM3_INTERNAL_RESOLUTION,
    SAM3_MASK_BINARIZATION,
    SAM3_SCORE_SEMANTICS,
    SELECTION_POLICY,
    counts_from_frames,
)
from .deadline import Budgets, Deadline, DeadlineExceeded, StopState
from .resources import ResourceMonitor, ResourcePolicy
from .visual_prompt import VisualPromptError, validate_visual_prompt, validate_bound_frame


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


@dataclass
class RunRequest:
    io_run: Path
    indices: list[int]
    object_id: str
    text_prompt: str | None
    output_root: Path
    repo_dir: Path
    backend_kind: str = SAM3_BACKEND_NAME
    fake_spec: Path | None = None
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    config_path: Path | None = None
    config_sha256: str | None = None
    hf_revision: str | None = None
    hf_cache_dir: Path | None = None
    device: str = "cuda"
    confidence_threshold: float = 0.5
    resolution: int = SAM3_INTERNAL_RESOLUTION
    allow_hf_download: bool = False
    explicit_selection: dict[int, str] = field(default_factory=dict)
    argv: list[str] = field(default_factory=list)
    cwd: str = ""
    env_allowlist: list[str] = field(default_factory=list)
    host_probe: dict | None = None
    resource_policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    budgets: Budgets = field(default_factory=Budgets)
    monitor_interval_seconds: float = MONITOR_INTERVAL_DEFAULT_SECONDS
    use_worker: bool = False
    worker_python: str | None = None
    worker_adapter_dir: Path | None = None
    #: module the child runs; fixtures point this at a wrapper that injects a
    #: fake backend while the real supervisor/ownership/observer stay in place
    worker_module: str = "sam3_adapter.worker"
    worker_extra_env: dict | None = None
    #: continuous device observation: "auto" (default) wires the bounded driver query on
    #: CUDA and is NOT_APPLICABLE on CPU; "none" disables a required channel and is rejected
    device_observe: str = "auto"
    device_observe_timeout: float = 5.0
    visual_prompt: dict | None = None


@dataclass
class RunOutcome:
    run_dir: Path | None
    status: str
    exit_code: int
    manifest: dict | None


class OutputRefused(RuntimeError):
    """Output directory already exists or cannot be created."""


# --------------------------------------------------------------------------- #
# sample series
# --------------------------------------------------------------------------- #
def _sample_peaks(samples: list[dict]) -> dict:
    """Peaks over the whole series, never just the last sample.

    Module-level on purpose: the monitor object may be substituted in tests, and
    peak reporting must not depend on it.
    """
    def peak(key: str):
        values = [s[key] for s in samples
                  if isinstance(s.get(key), (int, float)) and not isinstance(s.get(key), bool)]
        return max(values) if values else None

    return {
        "n_samples": len(samples),
        "sample_first_utc": samples[0]["utc"] if samples else None,
        "sample_last_utc": samples[-1]["utc"] if samples else None,
        "ram_commit_peak_bytes": peak("process_commit_bytes"),
        "ram_rss_peak_bytes": peak("process_rss_bytes"),
        "device_used_peak_bytes": peak("device_used_bytes"),
        "torch_reserved_peak_bytes": peak("torch_reserved_bytes"),
        "torch_allocated_peak_bytes": peak("torch_allocated_bytes"),
        "unmeasurable_samples": sum(1 for s in samples if s.get("ram_commit_measurable") is False),
        "note": "peaks are computed over all samples; a single final sample is never reported as the peak",
    }


class _BackgroundSampler(threading.Thread):
    """Light soft-monitor thread covering the load and inference windows.

    The stop flag is deliberately named ``_stop_event``: ``threading.Thread`` has
    private internals called ``_stop`` on CPython 3.12, and shadowing them with an
    Event makes ``join()`` raise ``TypeError: 'Event' object is not callable``.
    """

    def __init__(self, monitor, interval: float, sink: list, errors: list, stop_state: StopState,
                 on_violation=None, scope: str = "parent process"):
        super().__init__(daemon=True, name="sam3-adapter-monitor")
        self.monitor = monitor
        self.interval = max(interval, 0.01)
        self.sink = sink
        self.errors = errors
        self.stop_state = stop_state
        self.on_violation = on_violation
        self.scope = scope
        self._stop_event = threading.Event()
        self.joined_cleanly: bool | None = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self.monitor.sample()
                payload.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                payload["sampler_scope"] = self.scope
                self.sink.append(payload)
            except Exception as exc:  # noqa: BLE001 - a failed sample is not safety
                self.errors.append({"code": "MONITOR_SAMPLE_FAILED",
                                    "message": f"{type(exc).__name__}: {exc}",
                                    "scope": self.scope})
                self.stop_state.latch("RESOURCE_LIMIT",
                                      f"monitor sample failed: {type(exc).__name__}: {exc}", "sampler")
                return
            try:
                violations = self.monitor.violations(self.sink[-1])
            except Exception as exc:  # noqa: BLE001
                violations = [f"violations() raised {type(exc).__name__}: {exc}"]
            if violations:
                self.errors.append({"code": "RESOURCE_THRESHOLD_OBSERVED",
                                    "message": violations[0], "scope": self.scope})
                self.stop_state.latch("RESOURCE_LIMIT", violations[0], "sampler")
                if self.on_violation is not None:
                    try:
                        self.on_violation(violations[0])
                    except Exception:  # pragma: no cover - defensive
                        pass
                return
            self._stop_event.wait(self.interval)

    def stop(self) -> dict:
        """Stop and join with a bound; a failed join is recorded, never hidden."""
        self._stop_event.set()
        try:
            self.join(timeout=5.0)
            self.joined_cleanly = not self.is_alive()
        except Exception as exc:  # noqa: BLE001 - the 3.12 name-shadowing class of bug
            self.joined_cleanly = False
            self.errors.append({"code": "SAMPLER_JOIN_FAILED",
                                "message": f"{type(exc).__name__}: {exc}", "scope": self.scope})
        if self.joined_cleanly is False:
            self.errors.append({"code": "SAMPLER_NOT_JOINED",
                                "message": "sampler thread did not stop within the join bound",
                                "scope": self.scope})
        return {"scope": self.scope, "joined_cleanly": self.joined_cleanly}


# --------------------------------------------------------------------------- #
# manifest helpers
# --------------------------------------------------------------------------- #
def _blank_frame(input_index: int, status: str, reason: str) -> dict:
    return {
        "input_index": input_index,
        "frame_id": None,
        "source_image": None,
        "source_image_sha256": None,
        "photo_io_image_sha256": None,
        "image_domain": IMAGE_DOMAIN,
        "width": None,
        "height": None,
        "status": status,
        "candidates": [],
        "candidates_dropped_empty": 0,
        "selected_candidate_id": None,
        "selected_mask": None,
        "selection": None,
        "errors": [{"code": "INPUT_ERROR", "message": reason}] if status == "INPUT_ERROR" else [],
        "not_run_reason": None,
    }


def _frame_record(source: sourceio.SourceFrame) -> dict:
    record = _blank_frame(source.input_index, "INPUT_ERROR" if source.error else "NOT_RUN",
                          source.error["message"] if source.error else "pending")
    record.update({
        "frame_id": source.frame_id,
        "source_image": str(source.source_image) if source.source_image else None,
        "source_image_sha256": source.source_image_sha256,
        "photo_io_image_sha256": source.photo_io_image_sha256,
        "width": source.width,
        "height": source.height,
        "errors": [source.error] if source.error else [],
        "not_run_reason": None if source.error else "pending",
    })
    return record


def _backend_block(backend) -> dict:
    try:
        described = backend.describe()
    except Exception as exc:  # pragma: no cover - defensive
        described = {"name": getattr(backend, "name", "unknown")}
        described["describe_error"] = f"{type(exc).__name__}: {exc}"
    block = {
        "name": described.get("name"),
        "code_commit": described.get("code_commit"),
        "checkpoint_path": described.get("checkpoint_path"),
        "checkpoint_sha256": described.get("checkpoint_sha256"),
        "config_sha256": described.get("config_sha256"),
        "python_version": described.get("python_version"),
        "torch_version": described.get("torch_version"),
        "cuda_runtime": described.get("cuda_runtime"),
        "device": described.get("device"),
    }
    if "describe_error" in described:
        block["describe_error"] = described["describe_error"]
    return block


def _safe_bindings(backend) -> dict | None:
    getter = getattr(backend, "bindings", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception as exc:  # pragma: no cover - defensive
        return {"binding_error": f"{type(exc).__name__}: {exc}"}


def _build_backend(request: RunRequest):
    visual = validate_visual_prompt(request.visual_prompt, request.indices, request.text_prompt)
    if request.backend_kind == "fake":
        if visual is not None:
            raise VisualPromptError("positive_boxes requires the SAM3 backend")
        if request.fake_spec is None:
            raise BackendUnavailable("fake backend requires --fake-spec")
        return FakeBackend(request.fake_spec)
    return Sam3Backend(
        repo_dir=request.repo_dir,
        checkpoint_path=request.checkpoint,
        checkpoint_sha256=request.checkpoint_sha256,
        device=request.device,
        confidence_threshold=request.confidence_threshold,
        resolution=request.resolution,
        allow_hf_download=request.allow_hf_download,
        config_path=request.config_path,
        config_sha256=request.config_sha256,
        hf_revision=request.hf_revision,
        hf_cache_dir=request.hf_cache_dir,
        visual_prompt=visual,
    )


def _stop_status_for(exc: Exception) -> str:
    if isinstance(exc, AuthRequired):
        return "AUTH_REQUIRED"
    if isinstance(exc, BackendUnavailable):
        return "BACKEND_UNAVAILABLE"
    if isinstance(exc, ResourceLimit):
        return "RESOURCE_LIMIT"
    return "BACKEND_ERROR"


def _select_index(candidates: list, explicit_id: str | None):
    """Return (status, index_into_candidates_or_None, problem_or_None)."""
    usable = [i for i, c in enumerate(candidates) if imagery.foreground_pixels(c.flat) > 0]
    if explicit_id is not None:
        chosen = next((i for i in usable if candidates[i].candidate_id == explicit_id), None)
        if chosen is None:
            return "NEEDS_SELECTION", None, {
                "code": "EXPLICIT_CANDIDATE_NOT_USABLE",
                "message": f"explicit candidate {explicit_id!r} missing or has zero foreground pixels",
            }
        return "OK", chosen, None
    if not usable:
        return "NO_MATCH", None, None
    if len(usable) == 1:
        return "OK", usable[0], None
    return "NEEDS_SELECTION", None, None


def _serialize_request(request: RunRequest, resolved: dict | None = None) -> dict:
    """Serializable request for the worker; resolved weight paths are authoritative."""
    payload = {
        # every path is absolute: the child runs with cwd=integration dir
        "io_run": str(Path(request.io_run).resolve()),
        "indices": list(request.indices),
        "object_id": request.object_id,
        "text_prompt": request.text_prompt,
        "visual_prompt": validate_visual_prompt(request.visual_prompt, request.indices, request.text_prompt),
        "output_root": str(Path(request.output_root).resolve()),
        "repo_dir": str(Path(request.repo_dir).resolve()),
        "backend_kind": request.backend_kind,
        "fake_spec": str(Path(request.fake_spec).resolve()) if request.fake_spec else None,
        "checkpoint": str(request.checkpoint) if request.checkpoint else None,
        "checkpoint_sha256": request.checkpoint_sha256,
        "config_path": str(request.config_path) if request.config_path else None,
        "config_sha256": request.config_sha256,
        "hf_revision": request.hf_revision,
        "hf_cache_dir": str(request.hf_cache_dir) if request.hf_cache_dir else None,
        "device": request.device,
        "confidence_threshold": request.confidence_threshold,
        "resolution": request.resolution,
        # the parent already resolved (and hashed) the weights: the child must not
        # download a second time
        "allow_hf_download": False,
        "explicit_selection": {str(k): v for k, v in request.explicit_selection.items()},
        "argv": list(request.argv),
        "cwd": request.cwd,
        "env_allowlist": list(request.env_allowlist),
        "host_probe": None,
        "use_worker": False,
    }
    if resolved:
        payload.update(resolved)
    return payload


# --------------------------------------------------------------------------- #
# worker supervision
# --------------------------------------------------------------------------- #
class _EventLedger:
    """The single validate/apply/dedup path for live reads AND the final drain.

    Event identity is the writer's monotonic ``seq``, never the second-resolution
    timestamp. Malformed records are reported; a torn final line is tolerated only
    while the writer is still alive.
    """

    def __init__(self, stop_state: StopState, samples: list, frame_records: dict,
                 errors: list, finalisation_errors: list, deadline: Deadline, on_ready=None,
                 started_frames: set[int] | None = None):
        self.stop_state = stop_state
        self.samples = samples
        self.frame_records = frame_records
        # Owned by execute so starts survive a later _drive_worker exception.
        # Only committed frame events belong in frame_records / frames_seen.
        self.started_frames = started_frames if started_frames is not None else set()
        self.errors = errors
        self.finalisation_errors = finalisation_errors
        self.deadline = deadline
        self.on_ready = on_ready
        self.seen_seq: dict[int, str] = {}
        self.seen_keys: set = set()
        self.conflicting_seq: list[dict] = []
        self.applied = 0
        self.duplicates = 0
        self.state: dict = {
            "ready": None, "load_started": None, "load_ended": None,
            "current_frame": None, "current_frame_deadline": None,
            "frames_seen": [], "load_error": None, "close_error": None,
            "done": None, "stopped": None, "fatal": [], "unknown_events": [],
            "missing_seq_records": 0, "seq_max": None, "drain_authoritative": None,
        }

    # --------------------------------------------------------------- dedup
    def _is_new(self, event: dict) -> bool:
        seq = event.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            digest = imagery.sha256_bytes(
                json.dumps(event, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            previous = self.seen_seq.get(seq)
            if previous is not None:
                self.duplicates += 1
                if previous != digest:
                    entry = {"code": "EVENT_SEQ_CONFLICT",
                             "message": f"seq {seq} was re-read with different content; the second "
                                        "record is rejected"}
                    self.conflicting_seq.append({"seq": seq})
                    self.errors.append(entry)
                    self.finalisation_errors.append(entry)
                return False
            self.seen_seq[seq] = digest
            self.state["seq_max"] = seq if self.state["seq_max"] is None else max(self.state["seq_max"], seq)
            return True
        key = (event.get("type"), event.get("input_index"), event.get("stage"),
               event.get("state"), event.get("message"))
        if key in self.seen_keys:
            self.duplicates += 1
            return False
        self.seen_keys.add(key)
        self.state["missing_seq_records"] += 1
        return True

    # --------------------------------------------------------------- apply
    def apply(self, event: dict) -> bool:
        if not isinstance(event, dict):
            self.errors.append({"code": "EVENT_RECORD_NOT_OBJECT",
                                "message": f"ignored non-object event {type(event).__name__}"})
            return False
        if not self._is_new(event):
            return False
        kind = event.get("type")
        if kind == worker.EVENT_READY:
            self.state["ready"] = {"pid": event.get("pid"), "at_utc": event.get("at_utc"),
                                   "python": event.get("python")}
            if self.on_ready is not None:
                try:
                    self.on_ready(event)
                except Exception:  # pragma: no cover - defensive
                    pass
        elif kind == worker.EVENT_STAGE:
            stage = event.get("stage")
            if stage == "load":
                if event.get("state") == "begin":
                    self.state["load_started"] = time.monotonic()
                else:
                    self.state["load_ended"] = time.monotonic()
            elif stage == "close" and event.get("state") == "error":
                self.state["close_error"] = {"message": event.get("message"),
                                             "at_utc": event.get("at_utc")}
                entry = {"code": "BACKEND_CLOSE_FAILED",
                         "message": f"worker reported a close error: {event.get('message')}"}
                self.finalisation_errors.append(entry)
                self.errors.append(entry)
        elif kind == worker.EVENT_LOAD_ERROR:
            self.state["load_error"] = {"stage": event.get("stage"), "kind": event.get("kind"),
                                        "message": event.get("message")}
            self.stop_state.latch(event.get("kind", "BACKEND_ERROR"),
                                  f"{event.get('stage')}: {event.get('message')}", "worker")
        elif kind == worker.EVENT_FRAME_BEGIN:
            index = event.get("input_index")
            self.started_frames.add(index)
            self.state["current_frame"] = index
            self.state["current_frame_deadline"] = self.deadline.frame_deadline_for(index)
        elif kind == worker.EVENT_FRAME:
            missing = [key for key in ("input_index", "status") if key not in event]
            if missing:
                entry = {"code": "EVENT_FRAME_MISSING_FIELDS",
                         "message": f"frame event is missing {missing}; it was not applied"}
                self.errors.append(entry)
                self.finalisation_errors.append(entry)
                return False
            self.frame_records[event["input_index"]] = event
            self.state["frames_seen"].append(event["input_index"])
            self.state["current_frame"] = None
            self.state["current_frame_deadline"] = None
        elif kind == worker.EVENT_SAMPLE:
            payload = dict(event)
            payload.setdefault("scope", "worker process only")
            payload["reported_by"] = "worker"
            payload["sampler_scope"] = "worker process"
            self.samples.append(payload)
            if event.get("failed"):
                self.stop_state.latch("RESOURCE_LIMIT",
                                      f"worker sample failed: {event.get('message')}", "worker")
        elif kind == worker.EVENT_STOP:
            self.stop_state.latch(event.get("kind", "BACKEND_ERROR"), str(event.get("reason")), "worker")
        elif kind == worker.EVENT_FATAL:
            self.state["fatal"].append({"message": event.get("message"), "stage": event.get("stage")})
            self.stop_state.latch("BACKEND_ERROR", str(event.get("message")), "worker_fatal")
        elif kind == worker.EVENT_DONE:
            self.state["done"] = {"stopped": event.get("stopped"),
                                  "frames_attempted": event.get("frames_attempted"),
                                  "frames_ok": event.get("frames_ok")}
            self.state["stopped"] = bool(event.get("stopped"))
            attempted = event.get("frames_attempted")
            observed = len(self.frame_records)
            if isinstance(attempted, int) and attempted != observed:
                entry = {"code": "EVENT_TERMINAL_COUNT_MISMATCH",
                         "message": f"done reports frames_attempted={attempted} but {observed} frame "
                                    f"events were drained"}
                self.errors.append(entry)
                self.finalisation_errors.append(entry)
        else:
            # an unrecognised event is evidence we cannot interpret: it must reach the
            # verdict, not just decorate the error list
            self.state["unknown_events"].append({"type": kind})
            entry = {"code": "EVENT_TYPE_UNKNOWN",
                     "message": f"unhandled event type {kind!r} was not applied"}
            self.errors.append(entry)
            self.finalisation_errors.append(entry)
        self.applied += 1
        return True

    def apply_all(self, events: list[dict]) -> int:
        return sum(1 for event in events if self.apply(event))

    # ------------------------------------------------------------ budgets
    def load_overdue(self) -> bool:
        if self.state["load_error"] or self.state["done"] is not None:
            return False
        return self.state["load_ended"] is None and time.monotonic() > self.deadline.load_deadline

    def frame_overdue(self) -> tuple[bool, int | None]:
        deadline_at = self.state["current_frame_deadline"]
        if deadline_at is None or self.state["done"] is not None:
            return False, None
        return time.monotonic() > deadline_at, self.state["current_frame"]

    def summary(self) -> dict:
        return {
            "events_applied": self.applied,
            "duplicates_ignored": self.duplicates,
            "seq_max": self.state["seq_max"],
            "records_without_seq": self.state["missing_seq_records"],
            "unknown_events": self.state["unknown_events"],
            "ready": self.state["ready"],
            "load_started": self.state["load_started"] is not None,
            "load_ended": self.state["load_ended"] is not None,
            "frames_seen": sorted(self.frame_records),
            "current_frame": self.state["current_frame"],
            "close_error": self.state["close_error"],
            "fatal": self.state["fatal"],
            "done": self.state["done"],
            "dedup_key": "monotonic seq from the writer; structural key only as a fallback",
        }


class _WorkerSupervisor(threading.Thread):
    """Tails the worker's event file and enforces budgets on the *live* child."""

    def __init__(self, events_path: Path, ownership, ledger: _EventLedger, stop_state: StopState,
                 errors: list, on_kill, interval: float = 0.05, wake=None):
        super().__init__(daemon=True, name="sam3-adapter-supervisor")
        self.events_path = events_path
        self.ownership = ownership
        self.ledger = ledger
        self.stop_state = stop_state
        self.errors = errors
        self.on_kill = on_kill
        self.interval = interval
        self.wake = wake
        self._stop_event = threading.Event()
        self.killed_by: dict | None = None
        self.joined_cleanly: bool | None = None
        self.stop_deadlines_hit: list[str] = []

    def _check_budgets(self) -> None:
        if self.ledger.load_overdue():
            self.stop_deadlines_hit.append("load")
            self.stop_state.latch(
                "RESOURCE_LIMIT",
                "load deadline exceeded before the worker finished loading "
                f"(budget {self.deadline_seconds()}s)", "parent_deadline")
            return
        overdue, index = self.ledger.frame_overdue()
        if overdue:
            self.stop_deadlines_hit.append(f"frame[{index}]")
            self.stop_state.latch(
                "RESOURCE_LIMIT",
                f"frame[{index}] deadline exceeded while the worker was inside segment() "
                f"(budget {self.deadline_seconds()}s)", "parent_deadline")
            return
        if time.monotonic() > self.deadline_batch():
            self.stop_deadlines_hit.append("batch")
            self.stop_state.latch("RESOURCE_LIMIT",
                                  "batch deadline exceeded while the worker was still running",
                                  "parent_deadline")

    def deadline_seconds(self) -> float:
        return self.ledger.deadline.budgets.load_seconds

    def deadline_batch(self) -> float:
        return self.ledger.deadline.batch_deadline

    def run(self) -> None:
        while not self._stop_event.is_set():
            events, problems = worker.read_events_strict(self.events_path, allow_trailing_partial=True)
            self.ledger.apply_all(events)
            for problem in problems:
                if not problem.get("tolerated"):
                    self.errors.append(problem)
            self._check_budgets()
            if self.stop_state.latched and self.ownership.process.poll() is None:
                try:
                    self.killed_by = self.on_kill(self.stop_state.reason or "latched stop")
                except Exception as exc:  # noqa: BLE001
                    self.errors.append({"code": "WORKER_KILL_FAILED",
                                        "message": f"{type(exc).__name__}: {exc}"})
                if self.wake is not None:
                    self.wake.set()
                return
            if self.ownership.process.poll() is not None:
                return
            self._stop_event.wait(self.interval)

    def stop(self, deadline: float | None = None) -> dict:
        self._stop_event.set()
        if self.wake is not None:
            self.wake.set()
        if self.ident is None:
            return {"joined_cleanly": None, "note": "supervisor never started"}
        try:
            self.join(timeout=10.0 if deadline is None else max(deadline - time.monotonic(), 0.0))
            self.joined_cleanly = not self.is_alive()
        except Exception as exc:  # noqa: BLE001
            self.joined_cleanly = False
            self.errors.append({"code": "SUPERVISOR_JOIN_FAILED",
                                "message": f"{type(exc).__name__}: {exc}"})
        if self.joined_cleanly is False:
            self.errors.append({"code": "SUPERVISOR_NOT_JOINED",
                                "message": "supervisor thread did not stop within the join bound"})
        return {"joined_cleanly": self.joined_cleanly, "killed_by": self.killed_by,
                "stop_deadlines_hit": self.stop_deadlines_hit}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def execute(request: RunRequest) -> RunOutcome:  # noqa: PLR0912, PLR0915
    run_id = new_run_id()
    started_utc = utc_now()
    run_dir = Path(request.output_root) / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise OutputRefused(f"output directory already exists, refusing to overwrite: {run_dir}") from exc
    except OSError as exc:
        raise OutputRefused(f"output directory is not writable: {run_dir} ({exc})") from exc

    deadline = Deadline(budgets=request.budgets)
    stop_state = StopState()
    run_errors: list[dict] = []
    frames: list[dict] = [_blank_frame(index, "INPUT_ERROR", "unresolved") for index in request.indices]
    resolved: dict[int, sourceio.SourceFrame] = {}
    source: sourceio.SourceIo | None = None
    backend = None
    bindings: dict | None = None
    backend_block: dict | None = None
    fatal_status: str | None = None
    skip_reason: str | None = None
    monitor = None
    samples: list[dict] = []
    monitor_errors: list[dict] = []
    sampler = None
    sampler_report: dict | None = None
    supervisor: _WorkerSupervisor | None = None
    worker_report: dict | None = None
    ownership: dict = {"handle": None, "records": []}
    observer = None
    ledger = None
    finalisation_errors: list[dict] = []
    frame_records: dict[int, dict] = {}
    started_frames: set[int] = set()
    cached_channels: dict | None = None
    upstream_recheck: dict = {"attempted": False, "ok": None, "problems": []}
    source_io_block = {
        "path": str(Path(request.io_run).expanduser().resolve()),
        "run_id": None,
        "run_json_sha256": None,
        "frames_jsonl_sha256": None,
    }
    execution_mode = "bounded_worker" if request.use_worker else "in_process"

    def stop_batch(reason: str | None) -> None:
        for record in frames:
            if record["status"] in ("INPUT_ERROR", "OK", "NO_MATCH", "NEEDS_SELECTION",
                                    "BACKEND_ERROR", "RESOURCE_LIMIT"):
                continue
            record["status"] = "NOT_RUN"
            record["not_run_reason"] = reason
            record["selected_candidate_id"] = None
            record["selected_mask"] = None
            record["selection"] = None

    def kill_worker(reason: str) -> dict:
        """Reclaim the process that is *actually running* right now.

        Ownership is published the moment the worker is launched, so this is reachable
        from the supervisor's stop callback while ``_drive_worker`` is still blocking.
        Reclaim is idempotent and reuses one fixed deadline.
        """
        handle = ownership["handle"]
        if handle is None:
            return {"exited": None, "residual_unknown": True,
                    "note": "no worker ownership yet: nothing was launched"}
        record = handle.reclaim(reason, deadline=deadline.reclaim_deadline_from_now())
        ownership["records"].append(record)
        return record

    try:
        # ---------------------------------------------------- bind the upstream
        try:
            source = sourceio.load_source_io(request.io_run)
            source_io_block = source.binding()
            for position, index in enumerate(request.indices):
                frame = sourceio.resolve_frame(source, index)
                resolved[index] = frame
                frames[position] = _frame_record(frame)
            visual = validate_visual_prompt(request.visual_prompt, request.indices, request.text_prompt)
            if visual is not None:
                for index in request.indices:
                    validate_bound_frame(visual, resolved[index])
                request.visual_prompt = visual
        except VisualPromptError as exc:
            reason = f"visual prompt could not be bound: {exc}"
            run_errors.append({"code": "VISUAL_PROMPT_INVALID", "message": reason})
            fatal_status = "INPUT_ERROR"
            skip_reason = reason
            for record in frames:
                record.update(status="INPUT_ERROR", errors=[{"code": "VISUAL_PROMPT_INVALID", "message": reason}])
        except sourceio.SourceIoError as exc:
            reason = f"upstream photo-io-v1 run could not be bound: {exc}"
            run_errors.append({"code": "SOURCE_IO_UNBOUND", "message": reason})
            fatal_status = "INPUT_ERROR"
            skip_reason = reason
            for position, index in enumerate(request.indices):
                frames[position] = _blank_frame(index, "INPUT_ERROR", reason)

        # ------------------------------------------------------------ backend
        if fatal_status is None:
            # the device policy is resolved BEFORE anything is launched: explicitly
            # disabling a required channel on CUDA is an input error, not a late stop
            device_gate = observer_module.resolve_device_policy(request)
            if request.use_worker and device_gate.get("input_error"):
                fatal_status = "INPUT_ERROR"
                skip_reason = f"device policy: {device_gate['input_error']}"
                run_errors.append({"code": "DEVICE_POLICY_INPUT_ERROR", "message": skip_reason,
                                   "device_policy": device_gate})
        if fatal_status is None:
            monitor_device = "cpu" if request.backend_kind == "fake" else request.device
            try:
                deadline.begin_load()
                backend = _build_backend(request)
                backend_block = _backend_block(backend)
                resolver = getattr(backend, "resolve_weights", None)
                if resolver is not None:  # weights/config bound BEFORE any heavy import
                    resolver()
                backend_block = _backend_block(backend)
                bindings = _safe_bindings(backend)
                # the parent monitor is created only AFTER the weights are bound, and it
                # never imports torch itself, so heavy torch/CUDA initialisation stays
                # inside the controlled worker domain
                monitor = ResourceMonitor(policy=request.resource_policy,
                                          device=monitor_device, import_torch=False)
                if request.use_worker:
                    channels = monitor.channel_report()
                    if channels.get("process_commit_bytes") is None:
                        raise ResourceLimit(
                            "headroom cannot be proven: the parent commit charge is unmeasurable")
                    headroom_note = (
                        "worker mode: the parent proves its own commit charge before launch; the device "
                        "channel is observed FROM THE WORKER during load and every frame, and any "
                        "unmeasurable or over-limit device sample latches RESOURCE_LIMIT instead of "
                        "being treated as within budget")
                else:
                    provable, why = monitor.can_prove_headroom()
                    if not provable:
                        raise ResourceLimit(f"headroom cannot be proven: {why}")
                    headroom_note = why
                if not request.use_worker:
                    if request.monitor_interval_seconds > 0:  # 0 disables the thread for
                        sampler = _BackgroundSampler(  # deterministic single-check tests
                            monitor, request.monitor_interval_seconds,
                            samples, monitor_errors, stop_state)
                        sampler.start()
                    backend.load()
                    deadline.end_load()
                    backend_block = _backend_block(backend)
                    bindings = _safe_bindings(backend)
            except DeadlineExceeded as exc:
                fatal_status = "RESOURCE_LIMIT"
                skip_reason = str(exc)
                run_errors.append({"code": "DEADLINE_EXCEEDED", "message": skip_reason})
                if backend is not None:
                    backend_block = _backend_block(backend)
                    bindings = _safe_bindings(backend)
            except Exception as exc:  # noqa: BLE001 - mapped onto contract statuses
                fatal_status = _stop_status_for(exc)
                skip_reason = f"{type(exc).__name__}: {exc}"
                run_errors.append({"code": f"STOP_{fatal_status}", "message": skip_reason})
                if backend is not None:
                    backend_block = _backend_block(backend)
                    bindings = _safe_bindings(backend)

        if backend_block is None:
            backend_block = {
                "name": SAM3_BACKEND_NAME if request.backend_kind != "fake" else "fake",
                "code_commit": None,
                "checkpoint_path": str(request.checkpoint) if request.checkpoint else None,
                "checkpoint_sha256": request.checkpoint_sha256,
                "config_sha256": None,
                "python_version": platform.python_version(),
                "torch_version": None,
                "cuda_runtime": None,
                "device": None,
            }

        # ------------------------------------------- frames: worker or in-process
        if fatal_status is None:
            if request.use_worker:
                worker_report = _drive_worker(
                    request, run_dir, resolved, deadline, stop_state, samples,
                    frame_records, run_errors, kill_worker, ownership, monitor,
                    monitor_errors, finalisation_errors, started_frames=started_frames)
                supervisor = worker_report.pop("supervisor", None)
                observer = worker_report.pop("observer", None)
                ledger = worker_report.pop("ledger", None)
                worker_report.pop("process", None)
                worker_report.pop("ownership", None)
                if worker_report.get("fatal_kind") and not stop_state.latched:
                    fatal_status = worker_report["fatal_kind"]
                    skip_reason = worker_report.get("fatal_reason")
            else:
                _drive_in_process(request, run_dir, resolved, deadline, stop_state, samples,
                                  frame_records, run_errors, monitor, monitor_errors, backend)

        if stop_state.latched or fatal_status is not None:
            stop_batch(stop_state.reason or skip_reason)

    except Exception as exc:  # noqa: BLE001 - the run must still leave a manifest
        fatal_status = stop_state.kind if stop_state.latched else "BACKEND_ERROR"
        skip_reason = f"unexpected adapter error: {type(exc).__name__}: {exc}"
        run_errors.append({"code": "UNEXPECTED_ADAPTER_ERROR", "message": skip_reason})
        stop_batch(skip_reason)
    finally:
        if sampler is not None:
            try:
                sampler_report = sampler.stop()
            except Exception as exc:  # noqa: BLE001
                run_errors.append({"code": "SAMPLER_STOP_FAILED",
                                   "message": f"{type(exc).__name__}: {exc}"})
        if observer is not None:
            try:
                observer.stop()
            except Exception as exc:  # noqa: BLE001
                entry = {"code": "OBSERVER_STOP_FAILED",
                         "message": f"{type(exc).__name__}: {exc}"}
                finalisation_errors.append(entry)
                run_errors.append(entry)
        if supervisor is not None:
            try:
                report = supervisor.stop()
                if worker_report is not None:
                    worker_report["supervisor_join"] = report
            except Exception as exc:  # noqa: BLE001
                entry = {"code": "SUPERVISOR_STOP_FAILED",
                         "message": f"{type(exc).__name__}: {exc}"}
                finalisation_errors.append(entry)
                run_errors.append(entry)
        # the channel snapshot is taken ONCE, inside a guard: a failing extra query
        # must not stop the manifest from being written
        if monitor is not None:
            try:
                cached_channels = monitor.channel_report()
            except Exception as exc:  # noqa: BLE001
                entry = {"code": "CHANNEL_QUERY_FAILED",
                         "message": f"{type(exc).__name__}: {exc}",
                         "fallback": "last known / unknown channels"}
                finalisation_errors.append(entry)
                run_errors.append(entry)
        if source is not None:
            upstream_recheck["attempted"] = True
            try:
                before = (imagery.sha256_bytes(source.run_json_bytes),
                          imagery.sha256_bytes(source.frames_jsonl_bytes))
                if before != source.fingerprint():
                    upstream_recheck["problems"].append("run.json/frames.jsonl bytes changed during the run")
                for index, frame in resolved.items():
                    if frame.source_image is None:
                        continue
                    if imagery.sha256_file(frame.source_image) != frame.source_image_sha256:
                        upstream_recheck["problems"].append(
                            f"input_index {index} source image bytes changed during the run")
                upstream_recheck["ok"] = not upstream_recheck["problems"]
                if not upstream_recheck["ok"] and fatal_status is None and not stop_state.latched:
                    fatal_status = "INPUT_ERROR"
                    run_errors.append({"code": "UPSTREAM_INPUT_CHANGED",
                                       "message": "; ".join(upstream_recheck["problems"])})
            except Exception as exc:  # noqa: BLE001 - record the inability, not a pass
                upstream_recheck["ok"] = None
                upstream_recheck["problems"].append(f"re-check impossible: {type(exc).__name__}: {exc}")
                run_errors.append({"code": "UPSTREAM_RECHECK_FAILED",
                                   "message": upstream_recheck["problems"][-1]})
        else:
            upstream_recheck["problems"].append("upstream never bound; nothing to re-check")
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - must not overwrite the original error
                entry = {"code": "BACKEND_CLOSE_FAILED",
                         "message": f"{type(exc).__name__}: {exc}"}
                finalisation_errors.append(entry)
                run_errors.append(entry)

    # ---------------------------------------------------------------- verdict
    run_errors.extend(monitor_errors)

    for record in frames:  # fold worker-reported frames into the manifest
        event = frame_records.get(record["input_index"])
        if event is None:
            if record["input_index"] in started_frames and record["status"] != "INPUT_ERROR":
                # Final drain has already run. A begin without a committed result
                # is interrupted work, never NOT_RUN or a disk-derived success.
                cause = stop_state.kind or fatal_status
                reason = stop_state.reason or skip_reason or "worker ended without a frame result event"
                record.update({
                    "status": "RESOURCE_LIMIT" if cause == "RESOURCE_LIMIT" else "BACKEND_ERROR",
                    "candidates": [], "candidates_dropped_empty": 0,
                    "selected_candidate_id": None, "selected_mask": None, "selection": None,
                    "not_run_reason": None,
                })
                record["errors"].append({
                    "code": "FRAME_RESULT_UNCOMMITTED",
                    "message": f"frame[{record['input_index']}] started but no frame result event "
                               f"was accepted after final drain: {reason}",
                })
            continue
        record.update({
            "status": event.get("status", record["status"]),
            "candidates": event.get("candidates", []),
            "candidates_dropped_empty": event.get("candidates_dropped_empty", 0),
            "selected_candidate_id": event.get("selected_candidate_id"),
            "selected_mask": event.get("selected_mask"),
            "selection": event.get("selection"),
            "not_run_reason": event.get("not_run_reason"),
        })
        if event.get("errors"):
            record["errors"] = list(record.get("errors", [])) + list(event["errors"])
    if stop_state.latched or fatal_status is not None:
        stop_batch(stop_state.reason or skip_reason)

    statuses = [record["status"] for record in frames]
    if fatal_status is not None:
        status = fatal_status
    elif stop_state.latched:
        status = stop_state.kind if stop_state.kind != "BACKEND_UNAVAILABLE" else "BACKEND_UNAVAILABLE"
    elif "BACKEND_ERROR" in statuses:
        status = "BACKEND_ERROR"
    elif "RESOURCE_LIMIT" in statuses:
        status = "RESOURCE_LIMIT"
    elif "INPUT_ERROR" in statuses:
        status = "INPUT_ERROR"
    elif frames and all(s == "OK" for s in statuses):
        status = "SUCCESS"
    elif frames:
        status = "PARTIAL"
    else:
        status = "INPUT_ERROR"

    if status in ("SUCCESS", "PARTIAL") and request.use_worker:
        # both a clean success and a partial result must rest on an explainable terminal
        # state and on finalisation that did not report failures; the first primary cause
        # (RESOURCE_LIMIT and friends) is never overwritten by this gate
        terminal = ledger.state.get("done") if ledger is not None else None
        if terminal is None:
            status = "BACKEND_ERROR"
            run_errors.append({
                "code": "WORKER_TERMINAL_STATE_UNEXPLAINED",
                "message": "worker mode: no terminal done event was drained, so the completion "
                           "cannot be explained",
            })
        elif finalisation_errors:
            status = "BACKEND_ERROR"
            run_errors.append({
                "code": "FINALISATION_FAILED",
                "message": "finalisation failures present: " + "; ".join(
                    entry["message"] for entry in finalisation_errors),
            })
        elif ledger is not None and ledger.conflicting_seq:
            status = "BACKEND_ERROR"
            run_errors.append({
                "code": "EVENT_SEQ_CONFLICT",
                "message": f"{len(ledger.conflicting_seq)} conflicting seq records were rejected",
            })
        elif ledger is not None and (terminal.get("frames_attempted") is not None
                                     and terminal["frames_attempted"] != len(ledger.frame_records)):
            status = "BACKEND_ERROR"
            run_errors.append({
                "code": "EVENT_TERMINAL_COUNT_MISMATCH",
                "message": "the terminal done count disagrees with the drained frame events",
            })

    if request.backend_kind == SAM3_BACKEND_NAME:
        backend_block["backend_unavailable_reason"] = (
            None if status == "SUCCESS" else (skip_reason or stop_state.reason)
        )
        if status == "SUCCESS" and (
            backend_block.get("code_commit") is None
            or backend_block.get("checkpoint_sha256") is None
            or backend_block.get("config_sha256") is None
        ):
            status = "BACKEND_UNAVAILABLE"
            backend_block["backend_unavailable_reason"] = "model/config hash could not be obtained"
            run_errors.append({"code": "BACKEND_HASH_MISSING",
                               "message": backend_block["backend_unavailable_reason"]})
    else:
        backend_block["backend_unavailable_reason"] = None
        backend_block["evidence_caveat"] = (
            "fake backend: this run exercises the object-mask-v1 interface only and must NOT be "
            "reported as SAM3 evidence"
        )

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "status": status,
        "source_io": source_io_block,
        "backend": backend_block,
        "backend_bindings": bindings,
        "object": {
            "object_id": request.object_id,
            "text_prompt": request.text_prompt,
            "visual_prompt": request.visual_prompt,
            "selection_policy": SELECTION_POLICY,
            "submitted_options": {
                "visual_prompt": "positive_boxes" if request.visual_prompt is not None else "REJECTED_NOT_IMPLEMENTED",
                "confidence_threshold": request.confidence_threshold,
                "resolution": request.resolution,
                "device": request.device,
                "allow_hf_download": request.allow_hf_download,
            },
        },
        "frames": frames,
        "counts": counts_from_frames(frames),
        "quality_status": NOT_EVALUATED,
        "cross_view_status": NOT_EVALUATED,
        "evidence_level": EVIDENCE_LEVEL_REAL if request.backend_kind == SAM3_BACKEND_NAME else EVIDENCE_LEVEL_FAKE,
        "backend_notes": {
            "internal_resize": (
                "SAM3 Sam3Processor resizes the input to a square "
                f"{request.resolution}x{request.resolution} grid (aspect ratio NOT preserved) "
                "before the forward pass"
            ),
            "output_remap": (
                "the pinned processor returns masks already resampled back to the canonical "
                "(height,width) as an N x 1 x H x W boolean tensor, thresholded at 0.5; the adapter "
                "validates all four dimensions, removes the singleton channel and applies no further "
                "geometric transform, and never re-encodes the source photo"
            ),
            "mask_binarization": SAM3_MASK_BINARIZATION,
            "score_semantics": SAM3_SCORE_SEMANTICS,
            "selected_mask_source": MASK_SOURCE,
            "empty_candidate_rule": (
                "candidates with zero foreground pixels are exported for review but excluded from "
                "selection, so a selected mask is never empty"
            ),
            "overlays": "overlays/ are visual aids only and are NOT authoritative masks",
            "stop_latch": (
                "the first observed stop cause is irrevocable; a later sample falling back below the "
                "threshold cannot clear it, and cleanup errors cannot overwrite it"
            ),
        },
        "resource_policy": request.resource_policy.as_dict(),
        "stop_state": stop_state.snapshot(),
        "resource_observation": {
            "samples": samples,
            "peaks": _sample_peaks(samples),
            "channels": cached_channels,
            "channels_note": (
                "captured once inside a guard; a failing extra query is recorded in errors and the "
                "manifest is still written with the last known / unknown channels"
            ),
            "monitor_interval_seconds": request.monitor_interval_seconds,
            "headroom_gate": locals().get("headroom_note", "in-process gate via can_prove_headroom()"),
            "monitor_coverage": (
                "bounded_worker mode: the parent samples itself AND the worker reports its own "
                "commit/RSS per frame, so RAM coverage includes the worker rather than the idle parent"
                if request.use_worker else
                "in_process mode: the sampler covers this process only"
            ),
            "sampler": sampler_report,
            "worker_reported_samples": sum(1 for s in samples if s.get("reported_by") == "worker"),
            "thresholds_hit": [e for e in run_errors if e["code"] == "RESOURCE_THRESHOLD_OBSERVED"],
        },
        "deadlines": deadline.snapshot(),
        "execution_mode": execution_mode,
        "worker": worker_report,
        "upstream_recheck": upstream_recheck,
        "host_probe": request.host_probe,
        "execution": {
            "argv": request.argv,
            "cwd": request.cwd,
            "env_allowlist": request.env_allowlist,
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "exit_code": EXIT_CODES[status],
        },
        "errors": run_errors,
    }

    text = _dumps_or_error(manifest, run_errors)
    (run_dir / "manifest.json").write_text(text, encoding="utf-8", newline="\n")
    return RunOutcome(run_dir=run_dir, status=status, exit_code=EXIT_CODES[status], manifest=manifest)


# --------------------------------------------------------------------------- #
# frame drivers
# --------------------------------------------------------------------------- #
def _drive_in_process(request, run_dir, resolved, deadline, stop_state, samples, frame_records,
                      run_errors, monitor, monitor_errors, backend) -> None:
    for index in request.indices:
        if stop_state.latched:
            return
        frame = resolved.get(index)
        if frame is None or frame.error is not None:
            continue
        record = frame_records.setdefault(index, {"input_index": index})
        try:
            deadline.check_batch()
            deadline.begin_frame(index)
            candidates = backend.segment(frame.source_image, request.text_prompt, request.object_id,
                                         frame.width, frame.height, index)
            deadline.end_frame()
        except DeadlineExceeded as exc:
            record.update({"status": "RESOURCE_LIMIT", "candidates": [],
                           "errors": [{"code": "DEADLINE_EXCEEDED", "message": str(exc)}]})
            stop_state.latch("RESOURCE_LIMIT", str(exc), "parent_deadline")
            return
        except ResourceLimit as exc:
            record.update({"status": "RESOURCE_LIMIT", "candidates": [],
                           "errors": [{"code": "RESOURCE_LIMIT", "message": str(exc)}]})
            stop_state.latch("RESOURCE_LIMIT", str(exc), "parent")
            return
        except (BackendError, BackendUnavailable, AuthRequired) as exc:
            record.update({"status": "BACKEND_ERROR", "candidates": [],
                           "errors": [{"code": "BACKEND_ERROR", "message": str(exc)}]})
            stop_state.latch("BACKEND_ERROR", str(exc), "parent")
            return
        except imagery.ImageError as exc:
            record.update({"status": "BACKEND_ERROR", "candidates": [],
                           "errors": [{"code": "BACKEND_MASK_INVALID", "message": str(exc)}]})
            stop_state.latch("BACKEND_ERROR", str(exc), "parent")
            return
        try:
            records = _write_candidates(candidates, run_dir, index)
            dropped = sum(1 for candidate in candidates if imagery.foreground_pixels(candidate.flat) == 0)
            status, chosen, problem = _select_index(candidates, request.explicit_selection.get(index))
            record.update({"status": status, "candidates": records, "candidates_dropped_empty": dropped,
                           "selected_candidate_id": None, "selected_mask": None, "selection": None,
                           "errors": [problem] if problem else [], "not_run_reason": None})
            if chosen is not None:
                selected = candidates[chosen]
                record["selected_candidate_id"] = selected.candidate_id
                record["selection"] = {
                    "method": "explicit_candidate" if index in request.explicit_selection else "unique_candidate",
                    "source": "cli_argument" if index in request.explicit_selection else "automatic",
                    "mask_source": MASK_SOURCE,
                }
                mask_rel = records[chosen]["mask_path"]
                record["selected_mask"] = {"path": mask_rel,
                                           "sha256": imagery.sha256_file(run_dir / mask_rel)}
                imagery.write_overlay(frame.source_image, selected.flat, selected.width, selected.height,
                                      run_dir / "overlays" / f"{index:06d}.png")
        except Exception as exc:  # noqa: BLE001 - artifacts count as backend failure
            record.update({"status": "BACKEND_ERROR", "candidates": [], "selected_candidate_id": None,
                           "selected_mask": None, "selection": None,
                           "errors": [{"code": "ARTIFACT_WRITE_FAILED",
                                       "message": f"{type(exc).__name__}: {exc}"}]})
            stop_state.latch("BACKEND_ERROR", f"artifact failure on frame[{index}]: {exc}", "parent")
            return
        if monitor is not None:
            try:
                payload = monitor.sample()
                payload["sampler_scope"] = "parent process"
                samples.append(payload)
            except Exception as exc:  # noqa: BLE001
                monitor_errors.append({"code": "MONITOR_SAMPLE_FAILED",
                                       "message": f"{type(exc).__name__}: {exc}",
                                       "scope": "parent process"})
                stop_state.latch("RESOURCE_LIMIT",
                                 "monitoring failed; a failed sample is not treated as within budget",
                                 "parent")
                return
            try:
                violations = monitor.violations(payload)
            except Exception as exc:  # noqa: BLE001
                violations = [f"violations() raised {type(exc).__name__}: {exc}"]
            if violations:
                monitor_errors.append({"code": "RESOURCE_THRESHOLD_OBSERVED",
                                       "message": violations[0], "scope": "parent process"})
                stop_state.latch("RESOURCE_LIMIT", violations[0], "parent")
                return


def _drive_worker(request, run_dir, resolved, deadline, stop_state, samples, frame_records,
                  run_errors, kill_worker, ownership, monitor=None, monitor_errors=None,
                  finalisation_errors=None, started_frames=None) -> dict:
    """One guarded lifetime; all cleanup consumers share the first reclaim deadline."""
    monitor_errors = monitor_errors if monitor_errors is not None else []
    finalisation_errors = finalisation_errors if finalisation_errors is not None else []
    device_policy = observer_module.resolve_device_policy(request)
    worker_dir = (run_dir / "worker").resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    events_path = worker_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    payload = _serialize_request(request, resolved={
        "checkpoint": str(request.checkpoint) if request.checkpoint else None,
        "checkpoint_sha256": request.checkpoint_sha256,
        "config_path": str(request.config_path) if request.config_path else None})
    job = {"request": payload, "events_path": str(events_path), "run_dir": str(run_dir.resolve()),
           "monitor_device": request.device, "device_policy": device_policy,
           "ram_limit_bytes": request.resource_policy.ram_commit_limit_bytes,
           "vram_limit_bytes": request.resource_policy.vram_limit_bytes,
           "cpu_threads": request.resource_policy.cpu_threads,
           "frames": [{"input_index": index, "source_image": str(resolved[index].source_image),
                       "width": resolved[index].width, "height": resolved[index].height}
                      for index in request.indices if index in resolved and resolved[index].error is None]}
    worker.write_job(worker_dir / "job.json", job)
    python = request.worker_python or sys.executable
    adapter_dir = request.worker_adapter_dir or Path(__file__).resolve().parent.parent
    owner = observer = supervisor = ledger = None
    original_error = None
    report = {"mode": "bounded_worker", "python": python, "adapter_dir": str(adapter_dir),
              "job_path": str(worker_dir / "job.json"), "events_path": str(events_path),
              "stdout_path": str(worker_dir / "worker.stdout.txt"),
              "stderr_path": str(worker_dir / "worker.stderr.txt"),
              "device_policy": device_policy, "frames_declared": len(job["frames"]),
              "reclaim": None, "fatal_kind": None, "fatal_reason": None}

    def final_error(code, message):
        entry = {"code": code, "message": message}
        finalisation_errors.append(entry)
        run_errors.append(entry)

    process = worker.launch_worker(python, worker_dir / "job.json", adapter_dir,
                                   worker_dir / "worker.stdout.txt", worker_dir / "worker.stderr.txt",
                                   extra_env=request.worker_extra_env, module=request.worker_module)
    try:
        report["pid"] = process.pid
        owner = observer_module.WorkerOwnership(process, python, str(adapter_dir), str(worker_dir / "job.json"),
                                                reclaim_seconds=request.budgets.reclaim_seconds)
        ownership["handle"] = owner
        ledger = _EventLedger(stop_state, samples, frame_records, run_errors, finalisation_errors,
                              deadline, on_ready=owner.note_ready, started_frames=started_frames)
        report["ledger"] = ledger
        wake = threading.Event()
        device_sampler = observer_module.make_device_sampler(
            "driver" if device_policy["device_channel_applies"] else "none",
            request.device, request.device_observe_timeout, device_policy=device_policy)
        if request.monitor_interval_seconds > 0:
            observer = observer_module.WorkerObserver(owner, request.monitor_interval_seconds,
                samples, monitor_errors, stop_state, policy=request.resource_policy,
                device_sampler=device_sampler, device_policy=device_policy)
            observer.start()
        supervisor = _WorkerSupervisor(events_path, owner, ledger, stop_state,
                                       run_errors, kill_worker, interval=0.05, wake=wake)
        supervisor.start()
        while process.poll() is None and not stop_state.latched:
            if time.monotonic() > deadline.batch_deadline:
                stop_state.latch("RESOURCE_LIMIT", "batch deadline reached while worker active", "parent_deadline")
                break
            wake.wait(0.02)
            wake.clear()
    except BaseException as exc:
        original_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end = deadline.reclaim_deadline_from_now()
        cleanup = {"stage": "post_popen_lifecycle", "original_error": original_error}
        # Signal both users before reclaim/join; do not close identity handles until they stop.
        for thread in (observer, supervisor):
            if thread is not None:
                thread._stop_event.set()
        try:
            members = owner.members() if owner is not None else None
            needs_reclaim = (original_error is not None or process.poll() is None or
                             members is None or bool(members.get("unresolved_members")) or
                             not members.get("enumeration_available"))
            if needs_reclaim:
                reason = original_error or stop_state.reason or "owned members not confirmed exited"
                if owner is not None:
                    cleanup["reclaim"] = kill_worker(reason)
                else:
                    cleanup["reclaim"] = observer_module.make_session_cleanup(
                        process, {"handle": None}, request.budgets.reclaim_seconds,
                        deadline_factory=lambda: end)(reason)
                report["reclaim"] = cleanup["reclaim"]
        except BaseException as exc:
            cleanup["reclaim"] = {"error": f"{type(exc).__name__}: {exc}", "exited": None}
            # The original Popen handle remains available even if member enumeration failed.
            try:
                cleanup["fallback"] = observer_module.make_session_cleanup(
                    process, {"handle": None}, request.budgets.reclaim_seconds,
                    deadline_factory=lambda: end)("ownership cleanup failed")
            except BaseException as fallback_exc:
                cleanup["fallback"] = {"error": f"{type(fallback_exc).__name__}: {fallback_exc}"}
            final_error("WORKER_RECLAIM_FAILED", cleanup["reclaim"]["error"])

        threads_stopped = True
        for label, thread in (("observer", observer), ("supervisor", supervisor)):
            if thread is None:
                continue
            try:
                joined = thread.stop(end)
                cleanup[label + "_join"] = report[label + "_join"] = joined
                if joined.get("joined_cleanly") is False:
                    threads_stopped = False
                    final_error(label.upper() + "_NOT_JOINED", label + " did not stop before shared deadline")
            except BaseException as exc:
                threads_stopped = False
                cleanup[label + "_join"] = {"error": f"{type(exc).__name__}: {exc}"}
                final_error(label.upper() + "_STOP_FAILED", cleanup[label + "_join"]["error"])

        if ledger is not None and threads_stopped:
            try:
                events, problems = worker.read_events_strict(events_path, allow_trailing_partial=False)
                ledger.apply_all(events)
                for problem in problems:
                    final_error(problem.get("code", "EVENT_PROBLEM"), problem.get("message", "invalid event"))
                ledger.state["drain_authoritative"] = True
                report["ledger_summary"] = ledger.summary()
            except BaseException as exc:
                final_error("EVENT_DRAIN_FAILED", f"{type(exc).__name__}: {exc}")
        try:
            returncode = process.poll()
            report.update(returncode=returncode, exited=returncode is not None, frames_seen=len(frame_records))
            if owner is not None:
                members = owner.members()
                residual = list(members["unresolved_members"])
                if not members["enumeration_available"]:
                    residual.append({"role": "enumeration", "state": "UNKNOWN"})
                report.update(members_after=members, ownership_identity=owner.identity(),
                              member_states=members["members"], top_level_exit_is_not_whole_tree_exit=True)
            else:
                residual = [] if returncode is not None else [{"role": "launcher", "state": "UNKNOWN"}]
            report.update(residual_details=residual, residual_unknown=bool(residual))
            if residual or returncode is None:
                final_error("MEMBERS_NOT_CONFIRMED_EXITED", "owned members/identity/enumeration not confirmed clean")
            if returncode not in (0, 1) and not stop_state.latched:
                report["fatal_kind"] = {3: "AUTH_REQUIRED", 5: "RESOURCE_LIMIT"}.get(returncode, "BACKEND_ERROR")
                report["fatal_reason"] = f"worker exit code {returncode!r}"
        except BaseException as exc:
            report.update(residual_unknown=True, residual_details=[{"role": "query", "state": "UNKNOWN"}])
            final_error("MEMBER_FINAL_QUERY_FAILED", f"{type(exc).__name__}: {exc}")

        if owner is not None and threads_stopped:
            try:
                cleanup["handles"] = owner.close_handles()
                if cleanup["handles"]["errors"]:
                    final_error("HANDLE_RELEASE_FAILED", "; ".join(cleanup["handles"]["errors"]))
            except BaseException as exc:
                final_error("HANDLE_RELEASE_FAILED", f"{type(exc).__name__}: {exc}")
        elif owner is not None:
            cleanup["handles"] = {"state": "UNKNOWN", "reason": "threads still using identity handles"}
        ownership["cleanup"] = cleanup
        report["cleanup"] = cleanup
        report["finalisation_errors"] = finalisation_errors
        if original_error is not None:
            run_errors.append({"code": "POST_LAUNCH_CLEANUP", "message": original_error, "cleanup": cleanup})
    return report


def _dumps_or_error(manifest: dict, run_errors: list[dict]) -> str:
    """Strictly finite JSON; a serialization failure becomes a visible error."""
    try:
        return json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except ValueError as exc:
        run_errors.append({"code": "MANIFEST_SERIALIZATION_REJECTED",
                           "message": f"non-finite number reached the manifest: {exc}"})
        manifest["status"] = "BACKEND_ERROR"
        manifest["execution"]["exit_code"] = EXIT_CODES["BACKEND_ERROR"]
        manifest["errors"] = run_errors
        for record in manifest["frames"]:
            for candidate in record["candidates"]:
                score = candidate.get("score")
                if isinstance(score, float) and (score != score or score in (float("inf"), float("-inf"))):
                    candidate["score"] = None
                    candidate["score_semantics"] = "REJECTED_NON_FINITE"
        return json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _write_candidates(candidates, run_dir: Path, input_index: int) -> list[dict]:
    records: list[dict] = []
    for candidate in candidates:
        rel = f"masks/{input_index:06d}_{candidate.candidate_id}.png"
        digest = imagery.write_mask_l(run_dir / rel, candidate.flat, candidate.width, candidate.height)
        records.append({
            "candidate_id": candidate.candidate_id,
            "mask_path": rel,
            "mask_sha256": digest,
            "foreground_pixels": imagery.foreground_pixels(candidate.flat),
            "score": candidate.score,
            "score_semantics": candidate.score_semantics,
        })
    return records


__all__ = ["OutputRefused", "RunOutcome", "RunRequest", "execute", "new_run_id"]
