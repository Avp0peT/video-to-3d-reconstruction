"""Windows Job Object supervisor, R1 — a real, callable supervision entry point.

This module exists to be *called by a real task*, not just by tests. It provides:

* :func:`supervise` -- the single entry point. It performs the full lifecycle:
  start event -> suspended create -> assign to Job -> verified resume -> monitoring
  (wall-clock deadline, run-dir threshold, combined-roots threshold, periodic
  sampling) -> termination on failure/timeout/threshold/normal-exit -> whole-tree
  reclamation with confirmation.
* :func:`observer_state` -- a three-valued liveness probe (``alive``/``dead``/
  ``unknown``). A failed query is ``unknown``; it is NEVER reported as ``dead``.
* :class:`JobObject`, :func:`create_suspended`, :class:`EventLog` and the ctypes
  ABI layer.

Design rules enforced here (from the R1 review):

* Every successfully-created branch records identity (PID, creation time, its own
  handles) and keeps its own handles until exit is confirmed. Nothing closes a
  handle and then throws it away.
* ``assign`` failure keeps the process suspended and reclaims it through its own
  process handle (``TerminateJobObject`` cannot reach an unassigned process).
* ``ResumeThread`` failure captures the real return value and the immediate Win32
  error, then reclaims the Job and our own handles.
* An exception, a failed query, or a failed log write all have an explicit failure
  state and fallback cleanup.
* The Job's actual member list is recorded; the launcher/venv interpreter is not
  assumed to be exactly one OS process.

Pure ``ctypes``; no third-party dependencies. Python 3.13.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import os
import pathlib
import subprocess
import time
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

JobObjectBasicAccountingInformation = 1
JobObjectExtendedLimitInformation = 9

CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_NO_WINDOW = 0x08000000

STARTF_USESTDHANDLES = 0x00000100

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
CREATE_ALWAYS = 2
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x00000080

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010

WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF
STILL_ACTIVE = 259

MI = 1024 * 1024
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# --------------------------------------------------------------------------- #
# ctypes ABI -- explicit restype AND argtypes on every entry, returns checked
# --------------------------------------------------------------------------- #

kernel32.CreateJobObjectW.restype = wt.HANDLE
kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]

kernel32.SetInformationJobObject.restype = wt.BOOL
kernel32.SetInformationJobObject.argtypes = [
    wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]

kernel32.QueryInformationJobObject.restype = wt.BOOL
kernel32.QueryInformationJobObject.argtypes = [
    wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD)]

kernel32.AssignProcessToJobObject.restype = wt.BOOL
kernel32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]

kernel32.TerminateJobObject.restype = wt.BOOL
kernel32.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]

kernel32.CloseHandle.restype = wt.BOOL
kernel32.CloseHandle.argtypes = [wt.HANDLE]

kernel32.CreateFileW.restype = wt.HANDLE
kernel32.CreateFileW.argtypes = [
    wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]

kernel32.CreateProcessW.restype = wt.BOOL
kernel32.CreateProcessW.argtypes = [
    wt.LPCWSTR, wt.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wt.BOOL, wt.DWORD,
    ctypes.c_void_p, wt.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p]

kernel32.ResumeThread.restype = wt.DWORD
kernel32.ResumeThread.argtypes = [wt.HANDLE]

kernel32.WaitForSingleObject.restype = wt.DWORD
kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]

kernel32.GetExitCodeProcess.restype = wt.BOOL
kernel32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]

kernel32.TerminateProcess.restype = wt.BOOL
kernel32.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]

kernel32.OpenProcess.restype = wt.HANDLE
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]

kernel32.GetProcessTimes.restype = wt.BOOL
kernel32.GetProcessTimes.argtypes = [
    wt.HANDLE, ctypes.POINTER(wt.FILETIME), ctypes.POINTER(wt.FILETIME),
    ctypes.POINTER(wt.FILETIME), ctypes.POINTER(wt.FILETIME)]

kernel32.GetCurrentProcess.restype = wt.HANDLE
kernel32.GetCurrentProcess.argtypes = []


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wt.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wt.DWORD),
        ("TotalProcesses", wt.DWORD),
        ("ActiveProcesses", wt.DWORD),
        ("TotalTerminatedProcesses", wt.DWORD),
    ]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR),
        ("lpTitle", wt.LPWSTR), ("dwX", wt.DWORD), ("dwY", wt.DWORD),
        ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD), ("dwXCountChars", wt.DWORD),
        ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
        ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
        ("lpReserved2", ctypes.c_void_p), ("hStdInput", wt.HANDLE),
        ("hStdOutput", wt.HANDLE), ("hStdError", wt.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wt.HANDLE), ("hThread", wt.HANDLE),
        ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD),
    ]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wt.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wt.BOOL),
    ]


# --------------------------------------------------------------------------- #
# Errors and handle helpers
# --------------------------------------------------------------------------- #

class SuperviseError(RuntimeError):
    """Base class; carries the Win32 error where one exists."""

    def __init__(self, message: str, win32_error: int | None = None) -> None:
        super().__init__(message)
        self.win32_error = win32_error


class JobError(SuperviseError):
    pass


class CreateProcessError(SuperviseError):
    pass


class LogError(SuperviseError):
    pass


def _handle_valid(handle: Any) -> bool:
    if handle is None:
        return False
    if isinstance(handle, int) and handle == 0:
        return False
    try:
        value = int(handle)
    except (TypeError, ValueError):
        return False
    return value not in (0, INVALID_HANDLE_VALUE)


def _last_win32_error() -> int:
    return int(ctypes.get_last_error())


def _close_quietly(handle: Any) -> bool:
    if not _handle_valid(handle):
        return False
    return bool(kernel32.CloseHandle(handle))


def _environment_block(env: dict[str, str]) -> ctypes.Array:
    """Sorted, double-NUL-terminated Unicode block.

    MUST be paired with CREATE_UNICODE_ENVIRONMENT (see create_suspended).
    """
    parts = [f"{key}={value}" for key, value in sorted(env.items())]
    return ctypes.create_unicode_buffer("\0".join(parts) + "\0")


def directory_size(root: pathlib.Path) -> int:
    """Total byte size under ``root``. Missing paths count as 0."""
    total = 0
    root = pathlib.Path(root)
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


# --------------------------------------------------------------------------- #
# Three-valued observer
# --------------------------------------------------------------------------- #

def observer_state(pid: int | None) -> dict[str, Any]:
    """Liveness of ``pid`` as ``alive`` / ``dead`` / ``unknown``.

    A failed OpenProcess/GetExitCodeProcess is ``unknown`` -- a refused or failed
    query must never be reported as death. This is the property the earlier
    delivery got wrong.
    """
    if not pid:
        return {"state": "unknown", "pid": pid, "win32_error": None,
                "reason": "no pid supplied"}
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not _handle_valid(handle):
        err = _last_win32_error()
        # ERROR_INVALID_PARAMETER (87) means no such process -> genuinely dead.
        if err == 87:
            return {"state": "dead", "pid": int(pid), "win32_error": err,
                    "reason": "OpenProcess: no such process (87)"}
        return {"state": "unknown", "pid": int(pid), "win32_error": err,
                "reason": f"OpenProcess failed (errno {err})"}
    try:
        code = wt.DWORD(0)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            err = _last_win32_error()
            return {"state": "unknown", "pid": int(pid), "win32_error": err,
                    "reason": f"GetExitCodeProcess failed (errno {err})"}
        if int(code.value) == STILL_ACTIVE:
            return {"state": "alive", "pid": int(pid), "win32_error": None,
                    "reason": "STILL_ACTIVE"}
        return {"state": "dead", "pid": int(pid), "win32_error": None,
                "reason": f"exit code {int(code.value)}"}
    finally:
        _close_quietly(handle)


def proc_creation_time_utc(pid: int) -> str | None:
    """Creation time, used to bind identity (guards against PID reuse)."""
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not _handle_valid(handle):
        return None
    try:
        creation = wt.FILETIME()
        exit_t = wt.FILETIME()
        kernel_t = wt.FILETIME()
        user_t = wt.FILETIME()
        ok = kernel32.GetProcessTimes(handle, ctypes.byref(creation),
                                      ctypes.byref(exit_t), ctypes.byref(kernel_t),
                                      ctypes.byref(user_t))
        if not ok:
            return None
        ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime((ticks - 116444736000000000) / 10_000_000))
    finally:
        _close_quietly(handle)


def own_peak_working_set_bytes() -> int | None:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, os.getpid())
    if not _handle_valid(handle):
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    finally:
        _close_quietly(handle)


# --------------------------------------------------------------------------- #
# Job object
# --------------------------------------------------------------------------- #

class JobObject:
    """A Job whose every query is checked and whose failures raise."""

    def __init__(self, name: str | None = None) -> None:
        handle = kernel32.CreateJobObjectW(None, name)
        if not _handle_valid(handle):
            err = _last_win32_error()
            raise JobError(f"CreateJobObjectW failed (errno {err})", err)
        self.handle = handle
        self.name = name
        self.closed = False

    def set_limits(
        self,
        *,
        job_memory_limit_bytes: int | None = None,
        active_process_limit: int | None = None,
        kill_on_close: bool = True,
        per_process_user_time_100ns: int | None = None,
    ) -> dict[str, Any]:
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = 0
        if job_memory_limit_bytes is not None:
            info.JobMemoryLimit = job_memory_limit_bytes
            flags |= JOB_OBJECT_LIMIT_JOB_MEMORY
        if active_process_limit is not None:
            info.BasicLimitInformation.ActiveProcessLimit = active_process_limit
            flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        if per_process_user_time_100ns is not None:
            info.BasicLimitInformation.PerProcessUserTimeLimit = per_process_user_time_100ns
            flags |= JOB_OBJECT_LIMIT_PROCESS_TIME
        if kill_on_close:
            flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.BasicLimitInformation.LimitFlags = flags
        ok = kernel32.SetInformationJobObject(
            self.handle, JobObjectExtendedLimitInformation, ctypes.byref(info),
            ctypes.sizeof(info))
        if not ok:
            err = _last_win32_error()
            raise JobError(f"SetInformationJobObject failed (errno {err})", err)
        return self.extended_limit_information()

    def extended_limit_information(self) -> dict[str, Any]:
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        returned = wt.DWORD(0)
        ok = kernel32.QueryInformationJobObject(
            self.handle, JobObjectExtendedLimitInformation, ctypes.byref(info),
            ctypes.sizeof(info), ctypes.byref(returned))
        if not ok:
            err = _last_win32_error()
            raise JobError(
                f"QueryInformationJobObject(extended) failed (errno {err})", err)
        return {
            "limit_flags": int(info.BasicLimitInformation.LimitFlags),
            "job_memory_limit_bytes": int(info.JobMemoryLimit),
            "process_memory_limit_bytes": int(info.ProcessMemoryLimit),
            "active_process_limit": int(info.BasicLimitInformation.ActiveProcessLimit),
            "peak_job_memory_used_bytes": int(info.PeakJobMemoryUsed),
            "peak_process_memory_used_bytes": int(info.PeakProcessMemoryUsed),
            "struct_size_bytes": ctypes.sizeof(info),
            "bytes_returned": int(returned.value),
        }

    def accounting(self) -> dict[str, Any]:
        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = wt.DWORD(0)
        ok = kernel32.QueryInformationJobObject(
            self.handle, JobObjectBasicAccountingInformation, ctypes.byref(info),
            ctypes.sizeof(info), ctypes.byref(returned))
        if not ok:
            err = _last_win32_error()
            # Never return zero-initialised memory as if it were a measurement.
            raise JobError(
                f"QueryInformationJobObject(accounting) failed (errno {err})", err)
        return {
            "total_processes": int(info.TotalProcesses),
            "active_processes": int(info.ActiveProcesses),
            "total_terminated_processes": int(info.TotalTerminatedProcesses),
            "total_page_fault_count": int(info.TotalPageFaultCount),
            "struct_size_bytes": ctypes.sizeof(info),
            "bytes_returned": int(returned.value),
        }

    @property
    def peak_job_memory_used_bytes(self) -> int:
        return self.extended_limit_information()["peak_job_memory_used_bytes"]

    def assign(self, process_handle: Any) -> None:
        ok = kernel32.AssignProcessToJobObject(self.handle, process_handle)
        if not ok:
            err = _last_win32_error()
            raise JobError(f"AssignProcessToJobObject failed (errno {err})", err)

    def terminate(self, exit_code: int = 1) -> bool:
        if self.closed or not _handle_valid(self.handle):
            return False
        return bool(kernel32.TerminateJobObject(self.handle, exit_code))

    def close(self) -> bool:
        if self.closed:
            return True
        self.closed = True
        return _close_quietly(self.handle)


# --------------------------------------------------------------------------- #
# Process creation and reclamation
# --------------------------------------------------------------------------- #

class ProcessRecord:
    """Identity + handles of one process we created. Kept until exit confirmed."""

    __slots__ = ("pid", "tid", "creation_utc", "process_handle", "thread_handle",
                 "creation_flags", "resumed", "closed", "in_job")

    def __init__(self, pid: int, tid: int, creation_utc: str, process_handle: Any,
                 thread_handle: Any, creation_flags: int) -> None:
        self.pid = pid
        self.tid = tid
        self.creation_utc = creation_utc
        self.process_handle = process_handle
        self.thread_handle = thread_handle
        self.creation_flags = creation_flags
        self.resumed = False
        self.closed = False
        # R2: whether AssignProcessToJobObject has already succeeded. This decides
        # which reclaim path is correct -- a process that is NOT yet in the Job
        # cannot be reached by TerminateJobObject.
        self.in_job = False

    def as_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "tid": self.tid, "creation_utc": self.creation_utc,
                "creation_flags": hex(self.creation_flags), "resumed": self.resumed,
                "closed": self.closed, "in_job": self.in_job}

    def close_handles(self) -> None:
        if self.closed:
            return
        _close_quietly(self.thread_handle)
        _close_quietly(self.process_handle)
        self.thread_handle = None
        self.process_handle = None
        self.closed = True


def create_suspended(
    argv: Sequence[str],
    cwd: str | pathlib.Path,
    env: dict[str, str],
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    *,
    extra_flags: int = 0,
    omit_unicode_env_flag: bool = False,
) -> ProcessRecord:
    """Create a suspended process with a correct Unicode environment block.

    The caller owns the returned :class:`ProcessRecord` and must reclaim it (via
    the Job, or :func:`terminate_and_wait`) before calling ``close_handles``.
    """
    security = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, True)
    handles: dict[str, Any] = {}
    h_stdin = None
    try:
        for tag, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            handle = kernel32.CreateFileW(
                str(path), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                ctypes.byref(security), CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
            if not _handle_valid(handle):
                err = _last_win32_error()
                raise CreateProcessError(
                    f"CreateFileW failed for {tag} (errno {err})", err)
            handles[tag] = handle
        h_stdin = kernel32.CreateFileW(
            "NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            ctypes.byref(security), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
        if not _handle_valid(h_stdin):
            err = _last_win32_error()
            raise CreateProcessError(f"CreateFileW(NUL) failed (errno {err})", err)

        startup = STARTUPINFOW()
        startup.cb = ctypes.sizeof(STARTUPINFOW)
        startup.dwFlags = STARTF_USESTDHANDLES
        startup.hStdInput = h_stdin
        startup.hStdOutput = handles["stdout"]
        startup.hStdError = handles["stderr"]

        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
        environment = _environment_block(env)
        created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        flags = CREATE_SUSPENDED | CREATE_NO_WINDOW
        if not omit_unicode_env_flag:
            flags |= CREATE_UNICODE_ENVIRONMENT
        flags |= int(extra_flags)

        info = PROCESS_INFORMATION()
        ctypes.set_last_error(0)
        ok = kernel32.CreateProcessW(
            None, command, None, None, True, flags,
            ctypes.cast(environment, ctypes.c_void_p), str(cwd),
            ctypes.byref(startup), ctypes.byref(info))
        err = _last_win32_error()   # sampled BEFORE any cleanup call
        if not ok:
            raise CreateProcessError(f"CreateProcessW failed (errno {err})", err)
        if not _handle_valid(info.hProcess) or not _handle_valid(info.hThread):
            _close_quietly(info.hProcess)
            _close_quietly(info.hThread)
            raise CreateProcessError(
                "CreateProcessW returned TRUE with invalid handles", 0)
        return ProcessRecord(int(info.dwProcessId), int(info.dwThreadId),
                             created_utc, info.hProcess, info.hThread, flags)
    finally:
        for handle in handles.values():
            _close_quietly(handle)
        if h_stdin is not None:
            _close_quietly(h_stdin)


def terminate_and_wait(
    process_handle: Any,
    *,
    exit_code: int = 1,
    wait_ms: int = 20_000,
) -> dict[str, Any]:
    """Terminate a process we own and confirm it is dead.

    This is the only correct reclaim path for a process that was created but never
    assigned to a Job.
    """
    result: dict[str, Any] = {"terminated": False, "wait_result": None,
                              "confirmed_exit_code": None, "waited_ms": None,
                              "errors": []}
    if not _handle_valid(process_handle):
        result["errors"].append("invalid process handle")
        return result
    if not kernel32.TerminateProcess(process_handle, exit_code):
        result["errors"].append(f"TerminateProcess errno {_last_win32_error()}")
    started = time.monotonic()
    wait = kernel32.WaitForSingleObject(process_handle, int(wait_ms))
    result["waited_ms"] = round((time.monotonic() - started) * 1000.0, 3)
    result["wait_result"] = int(wait)
    if wait == 0:
        result["terminated"] = True
    elif wait == WAIT_TIMEOUT:
        result["errors"].append("wait timed out; exit not confirmed")
    else:
        result["errors"].append(f"WaitForSingleObject returned {wait}")
    code = wt.DWORD(0)
    if kernel32.GetExitCodeProcess(process_handle, ctypes.byref(code)):
        result["confirmed_exit_code"] = int(code.value)
    else:
        result["errors"].append(f"GetExitCodeProcess errno {_last_win32_error()}")
    return result


def wait_for_exit(process_handle: Any, timeout_ms: int) -> dict[str, Any]:
    """Wait for a process to exit on its own; reports the real exit code."""
    started = time.monotonic()
    wait = kernel32.WaitForSingleObject(process_handle, int(timeout_ms))
    elapsed = round((time.monotonic() - started) * 1000.0, 3)
    if wait == 0:
        code = wt.DWORD(0)
        if not kernel32.GetExitCodeProcess(process_handle, ctypes.byref(code)):
            err = _last_win32_error()
            return {"exited": True, "exit_code": None, "elapsed_ms": elapsed,
                    "error": f"GetExitCodeProcess errno {err}", "win32_error": err}
        return {"exited": True, "exit_code": int(code.value),
                "elapsed_ms": elapsed, "error": None, "win32_error": None}
    if wait == WAIT_TIMEOUT:
        return {"exited": False, "exit_code": None, "elapsed_ms": elapsed,
                "error": "timeout", "win32_error": None}
    return {"exited": False, "exit_code": None, "elapsed_ms": elapsed,
            "error": f"WaitForSingleObject returned {wait}",
            "win32_error": _last_win32_error()}


# --------------------------------------------------------------------------- #
# Incremental log
# --------------------------------------------------------------------------- #

class EventLog:
    """Append-only JSONL log, flushed+fsynced per record.

    Each record is on disk before the next lifecycle step begins, so the log is
    valid evidence even if the supervisor is killed mid-flight. Open mode is
    append, never CREATE_ALWAYS-style overwrite of a previous attempt's log.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise LogError(f"log append failed: {exc}", None) from exc


def sha256_file(path: pathlib.Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def env_summary(
    overrides: dict[str, str] | None = None,
    inherited_count: int | None = None,
    secrets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Desensitised environment description.

    Only this task's own whitelisted overrides are recorded, values included.
    Inherited variables are recorded as a **count only** -- never names, never
    values (names such as ``OPENAI_API_KEY`` are themselves a disclosure).
    """
    overrides = dict(overrides or {})
    secrets = list(secrets or [])
    safe: dict[str, str] = {}
    redacted: list[str] = []
    for key, value in overrides.items():
        if any(secret and secret in str(value) for secret in secrets):
            safe[key] = "<redacted:fake-secret>"
            redacted.append(key)
        else:
            safe[key] = value
    return {
        "task_env_overrides": safe,
        "redacted_override_keys": sorted(redacted),
        "inherited_env_count": inherited_count,
        "inherited_env_names_recorded": False,
        "inherited_env_values_recorded": False,
        "env_mode": "inherit_plus_overrides",
    }


# --------------------------------------------------------------------------- #
# THE ENTRY POINT
# --------------------------------------------------------------------------- #

TERMINATION_REASONS = (
    "normal_exit", "wall_clock_exceeded", "run_dir_threshold",
    "total_threshold", "spawn_failure", "assign_failure", "resume_failure",
    "log_failure", "supervisor_error", "job_member_timeout",
    # R2: a stop that would have been `normal_exit` but whose reclamation could not
    # confirm every member gone.
    "reclaim_unconfirmed",
    # R2: the `created` event could not be logged, so the lifecycle aborted before
    # assignment and reclaimed the still-suspended process through its own handle.
    "created_log_failure",
)


def supervise(
    argv: Sequence[str],
    *,
    cwd: str | pathlib.Path,
    env_overrides: dict[str, str] | None = None,
    log_path: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    job_memory_limit_bytes: int = 256 * MI,
    wall_clock_seconds: float = 90.0,
    run_dir: str | pathlib.Path | None = None,
    run_dir_threshold_bytes: int | None = None,
    total_roots: Iterable[str | pathlib.Path] = (),
    total_threshold_bytes: int | None = None,
    poll_seconds: float = 0.5,
    reclamation_seconds: float = 20.0,
    label: str = "run",
    script_sha256: str | None = None,
    secrets: Sequence[str] | None = None,
    inherit_env: bool = True,
) -> dict[str, Any]:
    """Supervise ``argv`` to completion inside a Job Object.

    Lifecycle (all of it happens here, not in the caller):

    1. write the ``start`` record and flush it **before** creating the process;
    2. create suspended (Unicode env block + ``CREATE_UNICODE_ENVIRONMENT``);
    3. assign to the Job (still suspended);
    4. ``ResumeThread`` and check the result -- a failure reclaims Job + handles;
    5. monitor until: normal root exit, wall-clock deadline, run-dir threshold,
       or combined-roots threshold -- sampling Job accounting each period;
    6. terminate and reclaim the whole tree, confirming every Job member is gone
       within ``reclamation_seconds``;
    7. return a structured result with the real exit codes and the reason.

    The wall-clock deadline governs the *entire* run, including the case where the
    root process has exited but Job members remain.
    """
    started_monotonic = time.monotonic()
    deadline = started_monotonic + float(wall_clock_seconds)
    log = EventLog(log_path)
    run_dir = pathlib.Path(run_dir) if run_dir is not None else None
    total_roots = [pathlib.Path(p) for p in total_roots]

    result: dict[str, Any] = {
        "label": label,
        "argv": [str(a) for a in argv],
        "cwd": str(cwd),
        "python": sys_executable(),
        "script_sha256": script_sha256,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_monotonic": started_monotonic,
        "job_memory_limit_bytes": job_memory_limit_bytes,
        "wall_clock_seconds": wall_clock_seconds,
        "poll_seconds": poll_seconds,
        "reclamation_seconds": reclamation_seconds,
        "run_dir": str(run_dir) if run_dir else None,
        "run_dir_threshold_bytes": run_dir_threshold_bytes,
        "total_roots": [str(p) for p in total_roots],
        "total_threshold_bytes": total_threshold_bytes,
        "pid": None,
        "root_exit_code": None,
        "termination_reason": None,
        "job_members": None,
        "peak_job_memory_used_bytes": None,
        "final_accounting": None,
        "samples": [],
        "reclamation": None,
        "reclaim_confirmed": None,
        "log_errors": [],
        "errors": [],
        "status": "unknown",
    }

    proc: ProcessRecord | None = None
    job: JobObject | None = None
    root_handle_kept = False

    try:
        # ---- 1. start record, flushed before anything is created ------------ #
        try:
            log.append({
                "kind": "start",
                "label": label,
                "argv": result["argv"],
                "cwd": result["cwd"],
                "python": result["python"],
                "script_sha256": script_sha256,
                "started_utc": result["started_utc"],
                "started_monotonic": started_monotonic,
                "budgets": {
                    "wall_clock_seconds": wall_clock_seconds,
                    "job_memory_limit_bytes": job_memory_limit_bytes,
                    "run_dir_threshold_bytes": run_dir_threshold_bytes,
                    "total_threshold_bytes": total_threshold_bytes,
                },
                "env": env_summary(overrides=env_overrides,
                                   inherited_count=len(os.environ),
                                   secrets=secrets),
            })
        except LogError as exc:
            result["status"] = "log_failure"
            result["termination_reason"] = "log_failure"
            result["errors"].append(str(exc))
            return result

        # ---- 2/3. create suspended, then assign --------------------------- #
        env = dict(os.environ) if inherit_env else {}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        effective_overrides = {"PYTHONDONTWRITEBYTECODE": "1"}
        if env_overrides:
            env.update(env_overrides)
            effective_overrides.update(env_overrides)

        job = JobObject()
        limits = job.set_limits(job_memory_limit_bytes=job_memory_limit_bytes,
                                kill_on_close=True)
        log.append({"kind": "job_created", "limits": limits})

        try:
            proc = create_suspended(argv, cwd, env, stdout_path, stderr_path)
        except CreateProcessError as exc:
            result["status"] = "spawn_failure"
            result["termination_reason"] = "spawn_failure"
            result["errors"].append(str(exc))
            log.append({"kind": "spawn_failure", "error": str(exc),
                        "win32_error": exc.win32_error})
            return result

        result["pid"] = proc.pid
        root_handle_kept = True
        # R2 fix for review defect 1. If we cannot even WRITE the `created` record,
        # the run's evidence chain is already broken, so we abort the lifecycle at
        # this point. The process is suspended and has NEVER been assigned to the
        # Job, so `TerminateJobObject` cannot reach it; reclamation MUST go through
        # the process's own handle. Critically, the log failure must NOT skip that
        # reclamation, and the failure itself must not be silently swallowed.
        try:
            log.append({"kind": "created", **proc.as_dict(),
                        "creation_time_utc_recheck": proc_creation_time_utc(proc.pid)})
        except LogError as exc:
            result["log_errors"] = result.get("log_errors", [])
            result["log_errors"].append({"kind": "created", "error": str(exc)})
            result["errors"].append(f"created-event log failure: {exc}")
            result["reclamation"] = reclaim_created_process(job, proc, log,
                                                            reclamation_seconds)
            result["reclaim_confirmed"] = reclaim_confirmed(result["reclamation"])
            result["status"] = "created_log_failure"
            result["termination_reason"] = "created_log_failure"
            # No ResumeThread happened, so the process is still suspended; record
            # the reclaim outcome too (best-effort: the log itself is failing).
            try:
                log.append({"kind": "created_log_failure", "error": str(exc),
                            "reclaim": result["reclamation"]})
            except LogError:
                pass
            return result

        try:
            job.assign(proc.process_handle)
            proc.in_job = True
            log.append({"kind": "assigned", "pid": proc.pid})
        except JobError as exc:
            # Process is created but NOT in the Job. TerminateJobObject cannot
            # reach it, so reclaim through the process's own handle. It is still
            # suspended, so nothing it could have spawned exists yet.
            result["reclamation"] = reclaim_created_process(job, proc, log,
                                                            reclamation_seconds)
            result["status"] = "assign_failure"
            result["termination_reason"] = "assign_failure"
            result["errors"].append(str(exc))
            try:
                log.append({"kind": "assign_failure", "error": str(exc),
                            "win32_error": exc.win32_error,
                            "reclaim": result["reclamation"]})
            except LogError as log_exc:
                result.setdefault("log_errors", []).append(
                    {"kind": "assign_failure", "error": str(log_exc)})
            return result

        # ---- 4. resume, checking the result ------------------------------- #
        resumed = kernel32.ResumeThread(proc.thread_handle)
        if resumed == 0xFFFFFFFF:
            err = _last_win32_error()
            result["status"] = "resume_failure"
            result["termination_reason"] = "resume_failure"
            result["errors"].append(f"ResumeThread failed (errno {err})")
            try:
                log.append({"kind": "resume_failure", "win32_error": err})
            except LogError as log_exc:
                result.setdefault("log_errors", []).append(
                    {"kind": "resume_failure", "error": str(log_exc)})
            # Process is in the Job and suspended; the unified reclaim uses the
            # Job path because proc.in_job is True.
            result["reclamation"] = reclaim_created_process(job, proc, log,
                                                            reclamation_seconds)
            return result
        proc.resumed = True
        log.append({"kind": "resumed", "pid": proc.pid, "resume_count": int(resumed)})

        # ---- 5. monitor --------------------------------------------------- #
        reason = "normal_exit"
        while True:
            now = time.monotonic()
            sample: dict[str, Any] = {"monotonic": round(now - started_monotonic, 3)}
            try:
                sample["accounting"] = job.accounting()
            except JobError as exc:
                sample["accounting_error"] = str(exc)
                result["errors"].append(f"accounting sample failed: {exc}")
            if run_dir is not None:
                sample["run_dir_bytes"] = directory_size(run_dir)
            if total_roots:
                sample["total_bytes"] = sum(directory_size(p) for p in total_roots)
            result["samples"].append(sample)
            try:
                log.append({"kind": "sample", **sample})
            except LogError as exc:
                result["errors"].append(str(exc))

            if now >= deadline:
                reason = "wall_clock_exceeded"
                break
            if (run_dir_threshold_bytes is not None and run_dir is not None
                    and sample.get("run_dir_bytes", 0) > run_dir_threshold_bytes):
                reason = "run_dir_threshold"
                break
            if (total_threshold_bytes is not None and total_roots
                    and sample.get("total_bytes", 0) > total_threshold_bytes):
                reason = "total_threshold"
                break
            exit_probe = wait_for_exit(proc.process_handle, int(poll_seconds * 1000))
            if exit_probe["exited"]:
                result["root_exit_code"] = exit_probe["exit_code"]
                log.append({"kind": "root_exited", **exit_probe})
                # Root exited, but Job members may remain: the deadline still
                # governs, and reclamation below must confirm everyone is gone.
                try:
                    acct = job.accounting()
                    if acct["active_processes"] > 0:
                        reason = "normal_exit"
                except JobError as exc:
                    result["errors"].append(f"post-exit accounting failed: {exc}")
                break

        result["termination_reason"] = reason

        # ---- 6. terminate + reclaim the whole tree ------------------------ #
        result["reclamation"] = reclaim_created_process(job, proc, log,
                                                        reclamation_seconds)

        # ---- 7. final accounting (recorded before the Job is closed) ------ #
        try:
            final = job.extended_limit_information()
            result["peak_job_memory_used_bytes"] = final["peak_job_memory_used_bytes"]
            result["final_accounting"] = final
            log.append({"kind": "final_job_accounting", **final})
        except JobError as exc:
            result["errors"].append(f"final accounting failed: {exc}")
            try:
                log.append({"kind": "final_job_accounting_failed", "error": str(exc)})
            except LogError as log_exc:
                result.setdefault("log_errors", []).append(
                    {"kind": "final_job_accounting", "error": str(log_exc)})
        try:
            result["final_basic_accounting"] = job.accounting()
            log.append({"kind": "final_basic_accounting",
                        **result["final_basic_accounting"]})
        except JobError as exc:
            result["errors"].append(f"final basic accounting failed: {exc}")

        # ---- 8. status: reclamation must NOT be masked by a normal exit --- #
        # R2 fix for review defect 2. A run is `completed` only when the stop
        # reason really was a normal exit AND reclamation explicitly confirmed
        # every member gone AND nothing else errored. Anything else is surfaced.
        result["reclaim_confirmed"] = reclaim_confirmed(result["reclamation"])
        if reason == "normal_exit":
            if not result["reclaim_confirmed"]:
                result["status"] = "reclaim_unconfirmed"
                result["termination_reason"] = "reclaim_unconfirmed"
            elif result["errors"]:
                result["status"] = "completed_with_errors"
            else:
                result["status"] = "completed"
        else:
            result["status"] = reason
        return result

    except Exception as exc:  # noqa: BLE001 -- fallback cleanup must still run
        result["status"] = "supervisor_error"
        result["termination_reason"] = result["termination_reason"] or "supervisor_error"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        try:
            log.append({"kind": "supervisor_error", "error": f"{type(exc).__name__}: {exc}"})
        except LogError:
            pass
        # R2: the fallback reclaim must ALSO choose the path from proc.in_job, or a
        # process created-but-not-yet-assigned would be left suspended.
        if proc is not None:
            try:
                result["reclamation"] = reclaim_created_process(
                    job, proc, log, reclamation_seconds)
            except Exception as exc2:  # noqa: BLE001
                result["errors"].append(f"fallback reclaim failed: {exc2}")
                result["reclamation"] = {
                    "all_members_dead": False,
                    "errors": [f"fallback reclaim failed: {exc2}"],
                    "path": "fallback_failed",
                }
        result["reclaim_confirmed"] = reclaim_confirmed(result.get("reclamation"))
        return result
    finally:
        # Handles are closed only AFTER the reclaim attempt and its outcome have
        # been recorded.
        if job is not None:
            job.close()
        if proc is not None and root_handle_kept:
            proc.close_handles()
        try:
            log.append({"kind": "cleanup", "pid": result.get("pid"),
                        "status": result.get("status"),
                        "reclaim_confirmed": result.get("reclaim_confirmed")})
        except LogError:
            pass


def reclaim_created_process(
    job: "JobObject | None",
    proc: "ProcessRecord | None",
    log: "EventLog",
    reclamation_seconds: float,
) -> dict[str, Any]:
    """Reclaim a process we created, choosing the path from its REAL state.

    R2 fix for review defect 1 and 2. The only thing that decides the path is
    whether ``AssignProcessToJobObject`` has already succeeded:

    * **not yet in the Job** -- ``TerminateJobObject`` cannot reach it, so terminate
      and wait through the process's *own* handle. A process created suspended and
      never resumed cannot have spawned children, so its own handle is sufficient.
    * **already in the Job** -- terminate the Job and confirm the whole tree is gone.

    A logging failure must never skip this. The caller invokes this from
    ``except``-and-``finally`` paths as well as the happy path, and a failure inside
    reclamation is recorded rather than raised, so the caller always gets a bounded,
    structured result and never spins.

    The caller is responsible for closing the handles *after* this returns.
    """
    outcome: dict[str, Any] = {
        "path": None,
        "in_job": None,
        "job_terminated": None,
        "root_terminate_and_wait": None,
        "all_members_dead": None,
        "elapsed_ms": None,
        "errors": [],
    }
    started = time.monotonic()
    if proc is None:
        outcome["path"] = "nothing_created"
        outcome["errors"].append("no created process to reclaim")
        outcome["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        return outcome

    outcome["in_job"] = bool(proc.in_job)
    try:
        if not proc.in_job:
            # Created but never assigned: reclaim through our own handle only.
            outcome["path"] = "own_handle_terminate_and_wait"
            outcome["root_terminate_and_wait"] = terminate_and_wait(
                proc.process_handle, exit_code=1,
                wait_ms=int(reclamation_seconds * 1000))
            tw = outcome["root_terminate_and_wait"]
            # R3: the nested errors from `terminate_and_wait` MUST be propagated to
            # the top-level reclaim errors. A wait that observed the process dead
            # while a native reclaim call reported an error is NOT a clean reclaim:
            # the death observation is kept, but the error may not be hidden, and
            # `reclaim_confirmed` must not be true while it stands.
            tw_errors = tw.get("errors")
            if tw_errors:
                if not isinstance(tw_errors, (list, tuple)):
                    tw_errors = [tw_errors]
                for err in tw_errors:
                    outcome["errors"].append(f"own-handle reclaim: {err}")
            outcome["all_members_dead"] = bool(
                tw.get("terminated") and tw.get("confirmed_exit_code") is not None)
        else:
            # In the Job: kill the Job, then confirm every member is gone.
            outcome["path"] = "job_tree"
            if job is not None:
                outcome["job_terminated"] = job.terminate(1)
            tree = _reclaim_tree(job, proc, log, reclamation_seconds) if job else {
                "all_members_dead": False, "errors": ["no job object"],
                "elapsed_ms": 0.0, "job_terminated": None, "observations": [],
                "members_before": None}
            outcome["job_terminated"] = tree.get("job_terminated")
            outcome["all_members_dead"] = tree.get("all_members_dead")
            outcome["tree"] = tree
            if tree.get("errors"):
                outcome["errors"].extend(tree["errors"])
    except Exception as exc:  # noqa: BLE001 -- reclamation must stay bounded
        outcome["all_members_dead"] = False
        outcome["errors"].append(f"reclaim raised {type(exc).__name__}: {exc}")
    outcome["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
    return outcome


def reclaim_confirmed(outcome: dict[str, Any] | None) -> bool:
    """True only when reclamation *explicitly* confirmed every member is gone.

    ``None``, a missing key, a failed query (``false``) and any recorded error all
    mean "not confirmed" -- never "assumed clean".
    """
    if not outcome:
        return False
    if outcome.get("all_members_dead") is not True:
        return False
    if outcome.get("errors"):
        return False
    return True


def sys_executable() -> str:
    import sys
    return sys.executable


def _reclaim_tree(job: JobObject, proc: ProcessRecord, log: EventLog,
                  reclamation_seconds: float) -> dict[str, Any]:
    """Terminate the Job and confirm every member is gone within the deadline."""
    outcome: dict[str, Any] = {
        "job_terminated": None,
        "members_before": None,
        "observations": [],
        "all_members_dead": None,
        "elapsed_ms": None,
        "errors": [],
    }
    started = time.monotonic()
    try:
        outcome["members_before"] = job.accounting()
    except JobError as exc:
        outcome["errors"].append(f"accounting before reclaim failed: {exc}")

    outcome["job_terminated"] = job.terminate(1)

    deadline = started + float(reclamation_seconds)
    root_pid = proc.pid
    while True:
        observations: list[dict[str, Any]] = []
        # Confirm the root via its own handle (identity-bound), and the Job via
        # its own accounting.
        root_wait = wait_for_exit(proc.process_handle, 200)
        observations.append({"root_pid": root_pid, "root_exited": root_wait["exited"],
                             "root_exit_code": root_wait["exit_code"]})
        try:
            acct = job.accounting()
            observations.append({"active_processes": acct["active_processes"],
                                 "total_processes": acct["total_processes"]})
            members_gone = acct["active_processes"] == 0 and root_wait["exited"]
        except JobError as exc:
            # A failed query is UNKNOWN, not "zero". Do not claim success.
            observations.append({"accounting_error": str(exc)})
            members_gone = False
            outcome["errors"].append(f"reclaim accounting failed: {exc}")
        outcome["observations"].append({"monotonic": round(time.monotonic() - started, 3),
                                        "observations": observations})
        if members_gone:
            outcome["all_members_dead"] = True
            break
        if time.monotonic() >= deadline:
            outcome["all_members_dead"] = False
            outcome["errors"].append("reclamation deadline exceeded; members may remain")
            break
        time.sleep(0.1)

    outcome["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
    try:
        log.append({"kind": "reclaim", **outcome})
    except LogError:
        pass
    return outcome
