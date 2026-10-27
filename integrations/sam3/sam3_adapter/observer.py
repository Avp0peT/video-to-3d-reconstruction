"""Live worker ownership and parent-side observation.

Design rules proven necessary by the A-R3 review:

* **observe the process that is actually running.** The launcher pid and the
  interpreter that does the work are usually two different members. The READY pid
  is accepted only after the OS relationship and creation time corroborate it, and
  every target is recorded with the evidence that identified it.
* **a null reading is not a pass.** ``read_process_memory_for_pid`` reports
  failures as null plus a reason instead of raising, so a required channel that
  returns null while the target is alive must latch a stop. A *confirmed exited*
  target is a different thing and is marked closed.
* **members survive the top-level exit.** Descendants are registered while the top
  level is alive and keep being observed afterwards; an unavailable enumeration is
  reported as UNKNOWN rather than as an empty (and therefore "safe") set.
* **identity, not a bare PID.** Creation time takes part in every decision: a PID
  whose creation time changed is a different process and is never terminated.
* **one deadline for everything.** reclaim, per-member handling, joins and waits
  all consume the same remaining budget; nothing renews it and no blocking control
  command is started once it is exhausted.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .contract import MONITOR_MIN_INTERVAL_SECONDS
from .resources import RAM_COMMIT_LIMIT_BYTES, VRAM_LIMIT_BYTES, read_process_memory

TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001
WAIT_TIMEOUT = 0x00000102
WAIT_OBJECT_0 = 0x00000000
ERROR_INVALID_PARAMETER = 87


class _PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("cntUsage", ctypes.c_ulong),
        ("th32ProcessID", ctypes.c_ulong),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", ctypes.c_ulong),
        ("cntThreads", ctypes.c_ulong),
        ("th32ParentProcessID", ctypes.c_ulong),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_ulong),
        ("szExeFile", ctypes.c_char * 260),
    ]


def _k32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.Process32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESSENTRY32)]
    kernel32.Process32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESSENTRY32)]
    kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p]
    return kernel32


def open_identity_handle(pid: int) -> int | None:
    """A handle usable both for exit waits and for creation-time checks."""
    if os.name != "nt" or not pid:
        return None
    try:
        handle = _k32().OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, 0, int(pid))
        return int(handle) if handle else None
    except Exception:  # pragma: no cover
        return None


def close_handle(handle: int | None) -> None:
    if os.name != "nt" or not handle:
        return
    if not _k32().CloseHandle(ctypes.c_void_p(handle)):
        raise OSError(ctypes.get_last_error(), "CloseHandle failed")


def handle_creation_time(handle: int) -> int | None:
    created, exited, kernel, user = (ctypes.c_ulonglong() for _ in range(4))
    ok = _k32().GetProcessTimes(ctypes.c_void_p(handle), ctypes.byref(created),
                               ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user))
    return int(created.value) if ok else None


def handle_alive(handle: int | None) -> bool | None:
    """Exit state via an already-held handle: no PID-reuse window."""
    if os.name != "nt" or not handle:
        return None
    try:
        result = _k32().WaitForSingleObject(ctypes.c_void_p(handle), 0)
    except Exception:  # pragma: no cover
        return None
    if result == WAIT_OBJECT_0:
        return False
    if result == WAIT_TIMEOUT:
        return True
    return None


def process_alive(pid: int) -> bool | None:
    """True/False when the OS can answer, None when it cannot (UNKNOWN)."""
    if os.name != "nt" or not pid:
        return None
    try:
        kernel32 = _k32()
        handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
    except Exception:  # pragma: no cover
        return None
    if not handle:
        return False if ctypes.get_last_error() == ERROR_INVALID_PARAMETER else None
    try:
        result = kernel32.WaitForSingleObject(handle, 0)
        if result == WAIT_OBJECT_0:
            return False
        if result == WAIT_TIMEOUT:
            return True
        return None
    finally:
        kernel32.CloseHandle(handle)


def process_creation_time(pid: int) -> int | None:
    """OS-recorded creation time (100 ns ticks); part of the real identity."""
    if os.name != "nt" or not pid:
        return None
    try:
        kernel32 = _k32()
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
        if not handle:
            return None
        try:
            created = ctypes.c_ulonglong(0)
            exited = ctypes.c_ulonglong(0)
            kernel = ctypes.c_ulonglong(0)
            user = ctypes.c_ulonglong(0)
            ok = kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel), ctypes.byref(user))
            return int(created.value) if ok else None
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # pragma: no cover
        return None


def enumerate_processes() -> dict[int, dict] | None:
    """{pid: {ppid, exe}} from an OS snapshot; None when enumeration is unavailable."""
    if os.name != "nt":
        return None
    try:
        kernel32 = _k32()
        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == INVALID_HANDLE_VALUE or not snapshot:
            return None
    except Exception:  # pragma: no cover
        return None
    table: dict[int, dict] = {}
    try:
        entry = _PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel32.Process32First(snapshot, ctypes.byref(entry)):
            return None
        while True:
            table[int(entry.th32ProcessID)] = {
                "ppid": int(entry.th32ParentProcessID),
                "exe": entry.szExeFile.decode("mbcs", "replace"),
            }
            if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def descendants_of(pid: int, table: dict[int, dict] | None = None) -> list[dict] | None:
    """Live descendants of ``pid``; None when the process table is unavailable."""
    table = enumerate_processes() if table is None else table
    if table is None:
        return None
    children: dict[int, list[int]] = {}
    for child_pid, info in table.items():
        children.setdefault(info["ppid"], []).append(child_pid)
    found: list[dict] = []
    queue = list(children.get(pid, []))
    seen: set[int] = set()
    while queue:
        child = queue.pop(0)
        if child in seen:
            continue
        seen.add(child)
        found.append({"pid": child, "exe": table.get(child, {}).get("exe"),
                      "creation_time": process_creation_time(child)})
        queue.extend(children.get(child, []))
    return found


# --------------------------------------------------------------------------- #
# effective device policy (one rule for CLI / runner / job / observer)
# --------------------------------------------------------------------------- #
DEVICE_NOT_APPLICABLE = "NOT_APPLICABLE"


def effective_device(device: str | None) -> dict:
    """What the run's device really is, and whether a device channel applies."""
    kind, index = parse_device_string(device)
    return {"kind": kind, "device_kind": kind, "logical_index": index,
            "device_channel_applies": kind == "cuda",
            "raw": device}


def parse_device_string(device: str | None) -> tuple[str, int | None]:
    text = "cpu" if device is None else str(device).strip().lower()
    if text == "cpu":
        return "cpu", None
    if text == "cuda":
        return "cuda", 0
    if text.startswith("cuda:"):
        tail = text[5:]
        if not tail.isdigit():
            raise ValueError(f"unparsable CUDA device index in {device!r}")
        return "cuda", int(tail)
    raise ValueError(f"unsupported device {device!r}; expected cpu, cuda or cuda:N")


def resolve_device_identity(device: str | None, env: dict | None = None) -> dict:
    """Map the logical CUDA index to a nvidia-smi index, or say why we cannot.

    ``CUDA_VISIBLE_DEVICES`` renumbers the logical index, so the logical index is
    NOT assumed to be the physical nvidia-smi index.
    """
    env = os.environ if env is None else env
    policy = effective_device(device)
    policy.update({"visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                   "identity_provable": policy["kind"] != "cuda",
                   "physical_index": None, "reason": "cpu run: no device identity needed"})
    if policy["kind"] != "cuda":
        return policy
    visible = env.get("CUDA_VISIBLE_DEVICES")
    logical = policy["logical_index"] or 0
    if visible is None:
        policy["physical_index"] = logical
        policy["identity_provable"] = True
        policy["reason"] = "CUDA_VISIBLE_DEVICES unset, so the logical index is the physical index"
        return policy
    entries = [item.strip() for item in visible.split(",")]
    if (logical >= len(entries) or not all(item.isdigit() for item in entries)
            or len(set(entries)) != len(entries)):
        policy["reason"] = (f"CUDA_VISIBLE_DEVICES={visible!r} does not map logical index {logical} to a "
                            f"physical GPU id; the mapping cannot be proven")
        return policy
    policy["physical_index"] = int(entries[logical])
    policy["identity_provable"] = True
    policy["reason"] = f"logical {logical} -> physical {policy['physical_index']} via CUDA_VISIBLE_DEVICES"
    return policy


def validate_required_bytes(value, label: str, limit: int | None = None,
                            allow_none: bool = False) -> tuple[int | None, str | None]:
    """A required byte reading must be a non-bool, finite, non-negative integer."""
    if value is None:
        return (None, None) if allow_none else (None, f"{label} is missing")
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"{label} must be a non-boolean integer, got {type(value).__name__}"
    if value < 0:
        return None, f"{label} must be non-negative, got {value}"
    if limit is not None and value > limit:
        return value, None
    if limit is None and value > (1 << 62):
        return None, f"{label} is implausibly large ({value})"
    return value, None


def resolve_device_policy(request) -> dict:
    """One effective device policy for CLI / runner / job / observer.

    * CPU: the device channel does not apply; no driver query is issued, and RAM and
      process-identity monitoring stay switched on.
    * CUDA: continuous observation of the selected device is required and wired by
      default through the existing bounded driver query.
    * CUDA with the channel explicitly disabled, or a device whose identity cannot be
      proven (CUDA_VISIBLE_DEVICES remap), is an input error raised before launch.
    """
    observe_mode = str(getattr(request, "device_observe", "auto") or "auto").lower()
    timeout = float(getattr(request, "device_observe_timeout", 5.0) or 5.0)
    try:
        identity = resolve_device_identity(getattr(request, "device", "cpu"))
    except ValueError as exc:
        return {"device_kind": "invalid", "device_channel_applies": False, "observe_mode": observe_mode,
                "freshness_seconds": timeout, "input_error": f"invalid device: {exc}",
                "reason": str(exc), "identity_provable": False, "physical_index": None}
    policy = dict(identity)
    policy["observe_mode"] = observe_mode
    policy["freshness_seconds"] = timeout
    policy["input_error"] = None
    if observe_mode not in ("auto", "driver", "none") or not math.isfinite(timeout) or timeout <= 0:
        policy["input_error"] = "invalid device observation mode or timeout"
        return policy
    if identity["kind"] != "cuda":
        policy.update({"device_channel_applies": False, "required": False,
                       "reason": "effective CPU run: the device channel is NOT_APPLICABLE and no driver "
                                 "query is issued; RAM and process-identity monitoring remain active"})
        return policy
    policy["required"] = True
    policy["device_channel_applies"] = True
    if observe_mode == "none":
        policy["input_error"] = ("a CUDA run may not disable the required device channel "
                                 "(--device-observe none); use auto")
        return policy
    if not identity.get("identity_provable"):
        policy["input_error"] = ("the selected CUDA device identity cannot be proven: "
                                 f"{identity.get('reason')}")
        return policy
    return policy


def make_session_cleanup(process, ownership_holder: dict, reclaim_seconds: float, deadline_factory=None):
    """Bounded cleanup that works even before WorkerOwnership exists."""
    fixed_deadline = None
    def cleanup(reason: str) -> dict:
        nonlocal fixed_deadline
        record = {"reason": reason, "used": None, "exited": None, "residual_unknown": [],
                  "wall_seconds": None}
        started = time.monotonic()
        if fixed_deadline is None:
            fixed_deadline = deadline_factory() if deadline_factory else started + max(reclaim_seconds, 0.0)
        deadline = fixed_deadline
        ownership = ownership_holder.get("handle")
        if ownership is not None:
            record["used"] = "WorkerOwnership.reclaim"
            record.update(ownership.reclaim(reason, deadline=deadline))
            record["wall_seconds"] = round(time.monotonic() - started, 6)
            return record
        record["used"] = "held Popen handle only (ownership was never constructed)"
        if process is not None and process.poll() is None and time.monotonic() < deadline:
            try:
                process.kill()
            except OSError as exc:  # pragma: no cover
                record["kill_error"] = f"{type(exc).__name__}: {exc}"
        while process is not None and time.monotonic() < deadline and process.poll() is None:
            time.sleep(min(0.02, max(deadline - time.monotonic(), 0.0)))
        returncode = process.poll() if process is not None else None
        record["exited"] = returncode is not None
        record["returncode"] = returncode
        if returncode is None:
            record["residual_unknown"] = [{"pid": getattr(process, "pid", None),
                                           "role": "launcher", "state": "UNKNOWN"}]
        record["wall_seconds"] = round(time.monotonic() - started, 6)
        return record
    return cleanup



def device_violations(device, policy: dict, vram_limit: int) -> list[str]:
    """Strict device-channel validation, honouring the effective device policy."""
    applies = bool(policy.get("device_channel_applies"))
    if not applies:
        if isinstance(device, dict) and device.get("device_used_bytes") is not None:
            return ["device readings were taken on a run whose device channel is NOT_APPLICABLE"]
        return []
    problems: list[str] = []
    if device is None:
        return ["device channel is required for this device but was not sampled"]
    if not isinstance(device, dict):
        return [f"device reading must be an object, got {type(device).__name__}"]
    if device.get("error"):
        return [f"device query failed: {device['error']}"]
    used, problem = validate_required_bytes(device.get("device_used_bytes"), "device_used_bytes")
    if problem:
        problems.append(problem)
    total, problem = validate_required_bytes(device.get("device_total_bytes"), "device_total_bytes")
    if problem:
        problems.append(problem)
    if used is not None and used > vram_limit:
        problems.append(f"device used {used} B > limit {vram_limit} B")
    if used is not None and total is not None and used > total:
        problems.append(f"device used {used} B exceeds device total {total} B")
    expected = policy.get("physical_index")
    if (expected is not None and (type(device.get("device_index")) is not int
                                 or device.get("device_index") != expected)):
        problems.append(f"device reading is for index {device.get('device_index')!r} but the selected "
                        f"device is {expected}")
    if device.get("fresh") is not True:
        problems.append(f"device reading is stale (age {device.get('age_seconds')}s)")
    freshness = policy.get("freshness_seconds")
    age = device.get("age_seconds")
    if (isinstance(age, bool) or not isinstance(age, (int, float))
            or not math.isfinite(age) or age < 0):
        problems.append("device reading age must be present, finite and non-negative")
    elif freshness is None or age > freshness:
        problems.append(f"device reading age {age}s exceeds the {freshness}s freshness bound")
    if not policy.get("identity_provable", True):
        problems.append(f"the selected device's identity cannot be proven: {policy.get('reason')}")
    return problems



def terminate_member(member: "Member") -> dict:
    """Terminate a member using its identity handle when we hold one."""
    if member.identity_still_matches() is not True:
        return {"method": "skipped", "skipped": "identity not proven"}
    if member.handle:
        try:
            kernel32 = _k32()
            ok = bool(kernel32.TerminateProcess(ctypes.c_void_p(member.handle), 1))
            return {"method": "TerminateProcess(handle)", "argv": None,
                    "accepted": ok, "returncode": 0 if ok else None,
                    "win32_error": None if ok else ctypes.get_last_error()}
        except Exception as exc:  # pragma: no cover
            return {"method": "TerminateProcess(handle)", "returncode": None,
                    "error": f"{type(exc).__name__}: {exc}"}
    return {"method": "none", "skipped": "no identity handle; bare PID termination forbidden"}


@dataclass
class Member:
    """A registered member with an identity guarantee, not just a PID."""

    role: str
    pid: int
    creation_time: int | None
    handle: int | None = None
    exe: str | None = None
    origin: str = "os_enumeration"

    def identity(self) -> dict:
        return {"role": self.role, "pid": self.pid, "creation_time": self.creation_time,
                "exe": self.exe, "has_handle": bool(self.handle), "origin": self.origin}

    def alive(self) -> bool | None:
        if self.handle:
            return handle_alive(self.handle)
        return process_alive(self.pid)

    def identity_still_matches(self) -> bool | None:
        """False when the PID now belongs to a different process."""
        if self.creation_time is None:
            return None
        current = handle_creation_time(self.handle) if self.handle else process_creation_time(self.pid)
        if current is None:
            return None
        return current == self.creation_time


class WorkerOwnership:
    """Owns the live worker from the instant it exists and reclaims it once."""

    def __init__(self, popen, python: str, adapter_dir: str, job_path: str,
                 reclaim_seconds: float = 5.0):
        self.process = popen
        self.python = python
        self.adapter_dir = adapter_dir
        self.job_path = job_path
        self.reclaim_seconds = reclaim_seconds
        self.launcher_pid = popen.pid
        self.launcher_creation = process_creation_time(popen.pid)
        self.launcher_handle = open_identity_handle(popen.pid)
        self.self_reported_pid: int | None = None
        self.self_reported_at: str | None = None
        self.worker_pid: int | None = None
        self.worker_creation: int | None = None
        self.worker_handle: int | None = None
        self.worker_identity_evidence: dict | None = None
        self.members_registry: dict[int, Member] = {}
        self.created_monotonic = time.monotonic()
        self.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.reclaim_deadline: float | None = None
        self.reclaim_records: list[dict] = []
        self._lock = threading.RLock()
        self._reclaimed = False
        try:
            self.register_launcher()
        except BaseException:
            close_handle(self.launcher_handle)
            raise

    # ------------------------------------------------------------ registry
    def register_launcher(self) -> Member:
        with self._lock:
            member = Member(role="launcher", pid=self.launcher_pid, creation_time=self.launcher_creation,
                            handle=self.launcher_handle, origin="popen")
            self.members_registry[self.launcher_pid] = member
            return member

    def refresh_members(self) -> dict:
        """Registers newly seen descendants; never drops registered members."""
        with self._lock:
            table = enumerate_processes()
            if table is None:
                return {"enumeration_available": False, "new_members": 0}
            added = 0
            for pid, info in table.items():
                if pid in self.members_registry:
                    continue
                parent = self.members_registry.get(info["ppid"])
                if (parent is not None and parent.alive() is True
                        and process_creation_time(parent.pid) == parent.creation_time):
                    self.members_registry[pid] = Member(role="descendant", pid=pid,
                                                        creation_time=process_creation_time(pid),
                                                        handle=open_identity_handle(pid),
                                                        exe=info.get("exe"))
                    added += 1
            return {"enumeration_available": True, "new_members": added}

    def note_ready(self, event: dict) -> None:
        """Bind the real worker identity using OS evidence, never the claim alone."""
        with self._lock:
            pid = event.get("pid")
            self.self_reported_pid = pid
            self.self_reported_at = event.get("at_utc")
            if type(pid) is not int or pid <= 0:
                self.worker_identity_evidence = {"accepted": False,
                                                 "reason": "READY pid missing or not an int"}
                return
            creation = process_creation_time(pid)
            alive = process_alive(pid)
            is_launcher = pid == self.launcher_pid
            table = enumerate_processes()
            descendant = False
            if table is not None and not is_launcher:
                descendant = table.get(pid, {}).get("ppid") == self.launcher_pid
            accepted = bool(is_launcher or descendant) and alive is True and creation is not None
            self.worker_identity_evidence = {
                "accepted": accepted,
                "ready_pid": pid,
                "creation_time": creation,
                "alive_at_bind": alive,
                "is_launcher": is_launcher,
                "is_os_descendant_of_launcher": descendant,
                "enumeration_available": table is not None,
                "pid_equality_assumed": False,
                "rule": "the READY pid is trusted only once the OS confirms the parent relationship or that it "
                        "is the launcher; its creation time is captured and re-checked before any termination",
            }
            if accepted:
                self.worker_pid = pid
                self.worker_creation = creation
                existing = self.members_registry.get(pid)
                self.worker_handle = (existing.handle if existing and existing.creation_time == creation
                                      else open_identity_handle(pid))
                role = "worker_launcher" if is_launcher else "worker_interpreter"
                self.members_registry[pid] = Member(role=role, pid=pid, creation_time=creation,
                                                    handle=self.worker_handle,
                                                    origin="ready_event+os_evidence")

    def observation_targets(self) -> list[dict]:
        """The processes that must be observed, each with its identity evidence."""
        with self._lock:
            self.refresh_members()
            targets: list[dict] = []
            for pid, member in sorted(self.members_registry.items()):
                targets.append({
                    "pid": pid,
                    "role": member.role,
                    "creation_time": member.creation_time,
                    "is_actual_worker": bool(self.worker_pid is not None and pid == self.worker_pid),
                    "identified_by": member.origin,
                })
            if not targets:  # nothing bound yet: at least observe the launcher
                targets.append({"pid": self.launcher_pid, "role": "launcher",
                                "creation_time": self.launcher_creation,
                                "is_actual_worker": False, "identified_by": "popen"})
            return targets

    # ------------------------------------------------------------- identity
    def identity(self) -> dict:
        with self._lock:
            return {
                "launcher_pid": self.launcher_pid,
                "launcher_creation_time": self.launcher_creation,
                "launcher_python": self.python,
                "self_reported_pid": self.self_reported_pid,
                "self_reported_at": self.self_reported_at,
                "worker_pid": self.worker_pid,
                "worker_creation_time": self.worker_creation,
                "worker_identity_evidence": self.worker_identity_evidence,
                "pid_equality_assumed": False,
                "registered_members": [member.identity() for member in self.members_registry.values()],
                "identity_note": "creation time participates in every identity check; a PID whose creation time "
                                 "changed is a different process and is never terminated",
            }

    def members(self) -> dict:
        """Registered members plus freshly enumerated descendants; never hides them."""
        with self._lock:
            refresh = self.refresh_members()
            rows = []
            for member in list(self.members_registry.values()):
                matches = member.identity_still_matches()
                alive = member.alive()
                rows.append({**member.identity(), "alive": alive, "identity_matches": matches,
                             "state": ("UNKNOWN" if matches is not True else
                                       "EXITED" if alive is False else
                                       "ALIVE" if alive is True else "UNKNOWN")})
            live = [row for row in rows if row["alive"] is True]
            # every owned role counts: naming only descendants would let a live worker
            # interpreter slip past the leftover check
            unresolved = [row for row in rows if row["state"] != "EXITED"]
            return {
                "members": rows,
                "member_count": len(rows),
                # kept for callers written against the earlier shape: the live subset is derived
                # here and is never hidden merely because the top level has exited
                "live_members": live,
                "live_member_count": len(live),
                "unresolved_members": [{"pid": row["pid"], "role": row["role"], "state": row["state"]}
                                      for row in unresolved],
                "live_descendants": [row for row in live if row["role"] == "descendant"],
                "descendant_count": sum(1 for row in live if row["role"] == "descendant"),
                "enumeration_available": refresh["enumeration_available"],
                "new_members": refresh["new_members"],
                "top_level_exited": self.process.poll() is not None,
                "unknown_note": "alive=None or identity_matches=None is reported as UNKNOWN, never as exited; an "
                                "unavailable enumeration is UNKNOWN, not an empty set",
            }

    # -------------------------------------------------------------- reclaim
    def reclaim(self, reason: str, deadline: float | None = None) -> dict:
        with self._lock:
            if self._reclaimed:
                return {**self.reclaim_records[0], "already_reclaimed": True}
            self._reclaimed = True
            if self.reclaim_deadline is None:
                self.reclaim_deadline = (deadline if deadline is not None else
                                         time.monotonic() + self.reclaim_seconds)
            elif deadline is not None:
                self.reclaim_deadline = min(self.reclaim_deadline, deadline)
            end = self.reclaim_deadline
            record = {"reason": reason, "state": "IN_PROGRESS", "actions": [],
                      "reclaim_deadline_monotonic": end, "exited": None,
                      "residual_unknown": [], "members_before": None, "members_after": None,
                      "top_level_exit_is_not_whole_tree_exit": True}
            self.reclaim_records.append(record)
        started = time.monotonic()
        remaining = lambda: max(end - time.monotonic(), 0.0)
        try:
            before = self.members()
            record["members_before"] = before
            for row in before["members"]:
                if row["state"] == "EXITED":
                    continue
                action = {"label": f"terminate member {row['pid']} ({row['role']})"}
                if remaining() <= 0:
                    action["skipped"] = "shared budget exhausted"
                elif row["identity_matches"] is not True:
                    action["why"] = "identity not proven; no termination"
                elif row["alive"] is True:
                    with self._lock:
                        member = self.members_registry[row["pid"]]
                    action.update(terminate_member(member))
                else:
                    action["skipped"] = "process state UNKNOWN"
                record["actions"].append(action)
            # Popen owns its original process handle, independent of PID reuse.
            if remaining() > 0 and self.process.poll() is None:
                self.process.kill()
                record["actions"].append({"method": "Popen.kill held handle", "pid": self.launcher_pid})
            while remaining() > 0:
                current = self.members()
                if not any(row["alive"] is True for row in current["members"]):
                    break
                time.sleep(min(0.02, remaining()))
            after = self.members()
            record["members_after"] = after
            residual = list(after["unresolved_members"])
            if not after["enumeration_available"]:
                residual.append({"pid": None, "role": "enumeration", "state": "UNKNOWN"})
            record["residual_unknown"] = residual
            record["all_members_exited"] = not residual
            record["exited"] = self.process.poll() is not None and not residual
            record["state"] = "COMPLETE"
        except BaseException as exc:
            record["state"] = "FAILED"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["residual_unknown"] = [{"role": "reclaim", "state": "UNKNOWN"}]
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - started
            record["budget_exhausted"] = remaining() <= 0
        return record

    def close_handles(self) -> dict:
        """Called only after both users of the registry have stopped."""
        errors, closed = [], []
        with self._lock:
            handles = {member.handle for member in self.members_registry.values() if member.handle}
            handles.update(h for h in (self.launcher_handle, self.worker_handle) if h)
            for handle in handles:
                try:
                    close_handle(handle)
                    closed.append(handle)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            for member in self.members_registry.values():
                if member.handle in closed:
                    member.handle = None
            if self.launcher_handle in closed:
                self.launcher_handle = None
            if self.worker_handle in closed:
                self.worker_handle = None
        return {"closed_count": len(closed), "errors": errors}


def driver_device_query(index: int, timeout: float = 5.0, freshness: float = 2.0) -> dict:
    """Bounded query; age is a conservative query duration, not a driver timestamp.

    The sampler passes the resolved freshness policy explicitly. Keep the direct
    function's historical default for callers that intentionally use this API.
    """
    started = time.monotonic()
    result = {"device_index": index, "source": "nvidia-smi", "device_used_bytes": None,
              "device_total_bytes": None, "error": None, "age_seconds": None, "fresh": False,
              "freshness_seconds": freshness, "query_timeout_seconds": timeout, "query_rc": None,
              "query_started_monotonic": started, "query_finished_monotonic": None,
              "query_elapsed_seconds": None,
              "age_semantics": "conservative query elapsed duration; not a device measurement timestamp"}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace")
        result["query_rc"] = completed.returncode
        if completed.returncode != 0:
            result["error"] = f"nvidia-smi rc={completed.returncode}: {(completed.stderr or '').strip()[:120]}"
        else:
            try:
                used_mib, total_mib = (chunk.strip()
                                       for chunk in completed.stdout.strip().splitlines()[0].split(","))
                used_bytes, total_bytes = int(used_mib) * (1 << 20), int(total_mib) * (1 << 20)
                result["device_used_bytes"] = used_bytes
                result["device_total_bytes"] = total_bytes
            except (IndexError, ValueError) as exc:
                result["error"] = f"unparsable nvidia-smi output: {exc}"
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        finished = time.monotonic()
        elapsed = finished - started
        result.update(query_finished_monotonic=finished, query_elapsed_seconds=elapsed,
                      age_seconds=elapsed)
        # Compare the unrounded duration: even a sub-microsecond overrun fails.
        result["fresh"] = (result["error"] is None and math.isfinite(elapsed)
                           and 0 <= elapsed <= freshness)
    return result


def make_device_sampler(kind: str | None, device: str, timeout: float = 5.0, *, device_policy=None):
    """Consume the already-resolved device identity; never reparse a logical index."""
    if kind in (None, "", "none"):
        return None
    policy = device_policy if device_policy is not None else resolve_device_identity(device)
    if not policy.get("identity_provable") or policy.get("physical_index") is None:
        raise ValueError("device identity cannot be established")
    if kind == "driver":
        freshness = policy.get("freshness_seconds", timeout)
        return lambda: driver_device_query(policy["physical_index"], timeout=timeout, freshness=freshness)
    raise ValueError(f"unknown device observation mode: {kind!r}")


class WorkerObserver(threading.Thread):
    """Bounded parent-side sampling of the *actual* worker during load and frames."""

    def __init__(self, ownership, interval: float, samples: list, errors: list,
                 stop_state, policy=None, device_sampler=None, device_policy=None):
        super().__init__(daemon=True, name="sam3-adapter-worker-observer")
        self.ownership = ownership
        self.interval = max(interval, MONITOR_MIN_INTERVAL_SECONDS)
        self.samples = samples
        self.errors = errors
        self.stop_state = stop_state
        self.policy = policy
        self.device_sampler = device_sampler
        self.device_policy = device_policy or {"device_channel_applies": False}
        self._stop_event = threading.Event()
        self.joined_cleanly: bool | None = None
        self.observations = 0
        self.targets_seen: list[dict] = []
        self.started = False

    def start(self) -> None:  # noqa: D102
        super().start()
        self.started = True  # set only once the thread really exists

    # ------------------------------------------------------------- sampling
    def observe(self, *, ram_ready: Callable[[dict], None] | None = None) -> dict:
        from .resources import read_process_memory_for_pid  # noqa: PLC0415

        device_applicable = bool(self.device_policy.get("device_channel_applies"))

        getter = getattr(self.ownership, "observation_targets", None)
        targets = getter() if getter is not None else [
            {"pid": getattr(self.ownership, "launcher_pid", None), "role": "launcher",
             "creation_time": None, "is_actual_worker": False, "identified_by": "popen"}]
        self.targets_seen = targets
        ram_started = time.monotonic()
        parent_started = time.monotonic()
        parent = read_process_memory()
        parent_finished = time.monotonic()
        per_target = []
        for target in targets:
            pid = target.get("pid")
            member = self.ownership.members_registry.get(pid)
            matches = member.identity_still_matches() if member else None
            alive = member.alive() if member and matches is True else None
            reading = None
            ram_interval = None
            if alive is not False:
                read_started = time.monotonic()
                try:
                    reading = read_process_memory_for_pid(pid)
                except Exception as exc:  # noqa: BLE001
                    reading = {"commit_bytes": None, "rss_bytes": None, "pid": pid,
                               "commit_channel": f"exception {type(exc).__name__}: {exc}"}
                finally:
                    ram_interval = {"started_monotonic": read_started,
                                    "finished_monotonic": time.monotonic()}
            per_target.append({
                "pid": pid,
                "role": target.get("role"),
                "is_actual_worker": bool(target.get("is_actual_worker")),
                "identified_by": target.get("identified_by"),
                "alive": alive,
                "commit_bytes": (reading or {}).get("commit_bytes"),
                "rss_bytes": (reading or {}).get("rss_bytes"),
                "channel_reason": (reading or {}).get("commit_channel"),
                "confirmed_exited": alive is False,
                "ram_sample_interval": ram_interval,
            })
        observation = {
            "interval_seconds": self.interval,
            "reported_by": "parent",
            "sampler_scope": "launcher + actual worker + registered descendants, sampled from the OS",
            "targets": per_target,
            "covers_pids": [row["pid"] for row in per_target],
            "actual_worker_pids": [row["pid"] for row in per_target if row["is_actual_worker"]],
            "parent_commit_bytes": parent.get("commit_bytes"),
            "parent_rss_bytes": parent.get("rss_bytes"),
            "parent_ram_sample_interval": {"started_monotonic": parent_started,
                                           "finished_monotonic": parent_finished},
            "ram_phase_interval": {"started_monotonic": ram_started,
                                   "finished_monotonic": time.monotonic()},
        }
        # The run loop can adjudicate RAM before a synchronous device query.
        # Direct observe() callers retain observation-only behavior.
        if ram_ready is not None:
            ram_ready(observation)
        device = None
        device_interval = None
        if self.device_sampler is not None:
            device_started = time.monotonic()
            try:
                device = self.device_sampler()
            except Exception as exc:  # noqa: BLE001
                device = {"error": f"{type(exc).__name__}: {exc}", "device_used_bytes": None}
            finally:
                device_interval = {"started_monotonic": device_started,
                                   "finished_monotonic": time.monotonic()}
        observation.update({
            "t": round(time.monotonic(), 6),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
            "device_sample_interval": device_interval,
            "sampling_time_semantics": "intervals bound host read/query calls, not device measurement timestamps; "
                                       "t is observation completion; null interval means no read/query call",
            "device_channel": ("observed" if device_applicable else DEVICE_NOT_APPLICABLE),
            "coverage_note": "each target is sampled separately from the OS; a null reading carries its "
                             "channel reason and is treated as a failure, not as a pass; device totals are "
                             "device-wide and a soft sampler cannot claim to catch arbitrarily short peaks",
        })
        return observation

    def ram_violations(self, observation: dict) -> list[str]:
        """RAM rules shared by early adjudication and complete observations."""
        policy = self.policy
        ram_limit = getattr(policy, "ram_commit_limit_bytes", RAM_COMMIT_LIMIT_BYTES)
        problems: list[str] = []
        for row in observation.get("targets") or []:
            label = f"{row.get('role')} pid={row.get('pid')}"
            if row.get("confirmed_exited"):
                continue  # a confirmed exit is not a measurement failure
            if row.get("alive") is None:
                problems.append(f"{label}: process state UNKNOWN, its resource use cannot be ruled out")
                continue
            commit = row.get("commit_bytes")
            if commit is None:
                problems.append(f"{label}: commit charge unmeasurable while alive "
                                f"({row.get('channel_reason')})")
            else:
                value, problem = validate_required_bytes(commit, f"{label} commit_bytes")
                if problem:
                    problems.append(problem)
                elif value > ram_limit:
                    problems.append(f"{label}: RAM commit {value} B > limit {ram_limit} B")
        parent_commit = observation.get("parent_commit_bytes")
        value, problem = validate_required_bytes(parent_commit, "parent commit_bytes")
        if problem:
            problems.append(problem)
        elif value > ram_limit:
            problems.append(f"parent RAM commit {value} B > limit {ram_limit} B")
        return problems

    def violations(self, observation: dict) -> list[str]:
        problems = self.ram_violations(observation)
        vram_limit = getattr(self.policy, "vram_limit_bytes", VRAM_LIMIT_BYTES)
        problems.extend(device_violations(observation.get("device"),
                                          self.device_policy or {}, vram_limit))
        return problems

    def run(self) -> None:
        while not self._stop_event.is_set():
            ram_problems: list[str] = []

            def ram_ready(observation: dict) -> None:
                ram_problems.extend(self.ram_violations(observation))
                if ram_problems:
                    observation["ram_stop_decision_monotonic"] = time.monotonic()
                    self.errors.append({"code": "RESOURCE_THRESHOLD_OBSERVED",
                                        "message": ram_problems[0], "scope": "worker observation",
                                        "channel": "ram"})
                    self.stop_state.latch("RESOURCE_LIMIT", ram_problems[0], "observer")

            try:
                members = self.ownership.members()
                if (members.get("top_level_exited") and members.get("enumeration_available")
                        and not members.get("unresolved_members")):
                    return
                observation = self.observe(ram_ready=ram_ready)
            except Exception as exc:  # noqa: BLE001 - a failed sample is not safety
                self.errors.append({"code": "WORKER_OBSERVATION_FAILED",
                                    "message": f"{type(exc).__name__}: {exc}"})
                self.stop_state.latch("RESOURCE_LIMIT",
                                      f"worker observation failed: {type(exc).__name__}: {exc}",
                                      "observer")
                return
            self.observations += 1
            self.samples.append(observation)
            if ram_problems:
                # Preserve device failures even when RAM already stopped the run.
                # The existing bounded query still runs once; do not re-latch RAM
                # or let its secondary device outcome replace the first cause.
                for problem in device_violations(observation.get("device"), self.device_policy,
                        getattr(self.policy, "vram_limit_bytes", VRAM_LIMIT_BYTES)):
                    self.errors.append({"code": "RESOURCE_THRESHOLD_OBSERVED", "message": problem,
                                        "scope": "worker observation", "channel": "device"})
                return
            violations = self.violations(observation)
            if violations:
                self.errors.append({"code": "RESOURCE_THRESHOLD_OBSERVED",
                                    "message": violations[0], "scope": "worker observation"})
                self.stop_state.latch("RESOURCE_LIMIT", violations[0], "observer")
                return
            self._stop_event.wait(self.interval)

    def stop(self, deadline: float | None = None) -> dict:
        self._stop_event.set()
        if not self.started:
            return {"joined_cleanly": None, "observations": self.observations,
                    "note": "the observer thread was never started, so nothing was joined"}
        timeout = 5.0 if deadline is None else max(deadline - time.monotonic(), 0.0)
        try:
            self.join(timeout=timeout)
            self.joined_cleanly = not self.is_alive()
        except Exception as exc:  # noqa: BLE001
            self.joined_cleanly = False
            self.errors.append({"code": "OBSERVER_JOIN_FAILED",
                                "message": f"{type(exc).__name__}: {exc}"})
        if self.joined_cleanly is False:
            self.errors.append({"code": "OBSERVER_NOT_JOINED",
                                "message": "observer thread did not stop within the join budget"})
        return {"joined_cleanly": self.joined_cleanly, "observations": self.observations,
                "interval_seconds": self.interval, "targets": self.targets_seen,
                "join_budget_seconds": timeout}


def dump(payload: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def host_python() -> str:
    return sys.executable
