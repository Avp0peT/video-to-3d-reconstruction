"""Resource policy, capacity/usage separation and *soft* monitoring.

Hard-won distinctions encoded here (see REVIEW_AB_20261007.md, defect 1):

* **capacity is not usage.** A device whose total capacity exceeds the frozen
  budget is perfectly legal; only *measured usage* may violate the budget.
* Every channel is reported separately with an explicit coverage statement:
  process commit (this task's process), device total capacity, device
  used/free (whole device, other processes included), torch allocated/reserved
  (this process only).
* On Windows, RSS/USS are NOT the commit charge; the commit charge comes from
  ``GetProcessMemoryInfo(...).PagefileUsage``. RSS is reported only as a
  secondary, differently-scoped number and is never substituted for it.
* The thresholds are **observed**, never enforced by the OS. No hard cap is
  claimed anywhere.
* If a required channel cannot be measured, headroom cannot be proven and the
  real backend must not start.
"""

from __future__ import annotations

import ctypes
import platform
import sys
import time
from dataclasses import dataclass, field

MIB = 1 << 20
GIB = 1 << 30

#: frozen defaults -- unchanged by this repair
RAM_COMMIT_LIMIT_BYTES = 12 * GIB
VRAM_LIMIT_BYTES = 7 * GIB

ENFORCEMENT = "soft_observed_not_hard_capped"

CHANNEL_COVERAGE = {
    "process_commit_bytes": "this task's process only; Windows commit charge (PagefileUsage), not RSS/USS",
    "process_rss_bytes": "this task's process only; secondary, differently scoped, never used as the commit charge",
    "device_total_bytes": "whole device capacity; informational only, NOT a usage number",
    "device_used_bytes": "whole device usage; includes other processes, so it upper-bounds this task",
    "device_free_bytes": "whole device free memory; device-wide, not task-specific",
    "torch_allocated_bytes": "this process' torch allocations only",
    "torch_reserved_bytes": "this process' torch caching allocator reservation only",
}


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _windows_counters() -> tuple[int, int] | None:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        # HANDLEs are pointer-sized: without an explicit restype ctypes truncates
        # GetCurrentProcess() to 32 bits and the measurement silently fails,
        # which would (correctly) refuse every real run as "unmeasurable".
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
                                              ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PagefileUsage), int(counters.WorkingSetSize)
    except Exception:  # pragma: no cover - non-Windows / API unavailable
        return None


def _psutil_rss() -> int | None:
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover
        return None


def read_process_memory() -> dict:
    """Return commit and RSS separately; never conflate the two."""
    if sys.platform == "win32":
        measured = _windows_counters()
        if measured is not None:
            commit, rss = measured
            return {"commit_bytes": commit, "rss_bytes": rss,
                    "commit_channel": "psapi:PagefileUsage",
                    "rss_channel": "psapi:WorkingSetSize"}
    rss = _psutil_rss()
    return {"commit_bytes": None, "rss_bytes": rss,
            "commit_channel": "unavailable",
            "rss_channel": "psutil:rss" if rss is not None else "unavailable"}


def read_process_memory_for_pid(pid: int) -> dict:
    """Commit charge of *another* process (the worker) via an OS handle.

    This is what lets the parent keep observing a worker that is blocked inside a
    native call. Query failures are returned as null + a channel note; they are
    never silently replaced by the parent's own numbers.
    """
    result = {"commit_bytes": None, "rss_bytes": None, "pid": int(pid),
              "commit_channel": "unavailable", "rss_channel": "unavailable"}
    if sys.platform != "win32" or not pid:
        return result
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
                                              ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.OpenProcess(0x1000 | 0x0400, 0, int(pid))  # QUERY_LIMITED | VM_READ
        if not handle:
            result["commit_channel"] = f"OpenProcess failed (err={ctypes.get_last_error()})"
            return result
        try:
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                result["commit_channel"] = f"GetProcessMemoryInfo failed (err={ctypes.get_last_error()})"
                return result
            result["commit_bytes"] = int(counters.PagefileUsage)
            result["rss_bytes"] = int(counters.WorkingSetSize)
            result["commit_channel"] = "psapi:PagefileUsage (target pid)"
            result["rss_channel"] = "psapi:WorkingSetSize (target pid)"
        finally:
            kernel32.CloseHandle(handle)
    except Exception as exc:  # pragma: no cover - non-Windows
        result["commit_channel"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_cuda_device(device: str | None) -> tuple[str, int | None]:
    """Split a device string into (kind, index).

    ``cuda`` -> index None (torch's current device), ``cuda:1`` -> 1, ``cpu`` ->
    None. An unparsable value is rejected rather than silently falling back to
    device 0.
    """
    if device is None:
        return "cpu", None
    text = str(device).strip().lower()
    if text == "cpu":
        return "cpu", None
    if text == "cuda":
        return "cuda", None
    if text.startswith("cuda:"):
        tail = text[5:]
        if not tail.isdigit():
            raise ValueError(f"unparsable CUDA device index in {device!r}")
        return "cuda", int(tail)
    raise ValueError(f"unsupported device {device!r}; expected cpu, cuda or cuda:N")


@dataclass
class ResourcePolicy:
    ram_commit_limit_bytes: int = RAM_COMMIT_LIMIT_BYTES
    vram_limit_bytes: int = VRAM_LIMIT_BYTES
    enforcement: str = ENFORCEMENT
    cpu_threads: int = 2

    def as_dict(self) -> dict:
        return {
            "ram_commit_limit_bytes": self.ram_commit_limit_bytes,
            "ram_commit_limit_gib": round(self.ram_commit_limit_bytes / GIB, 3),
            "vram_limit_bytes": self.vram_limit_bytes,
            "vram_limit_gib": round(self.vram_limit_bytes / GIB, 3),
            "cpu_threads": self.cpu_threads,
            "enforcement": self.enforcement,
            "limits_apply_to": "measured usage only; device total capacity is never compared to the budget",
            "note": "observed soft thresholds; process and device are not hard-capped by the adapter",
        }


@dataclass
class ResourceMonitor:
    policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    device: str = "cuda"
    import_torch: bool = True
    _torch: object | None = None

    def __post_init__(self) -> None:
        self.device_kind, self.device_index = parse_cuda_device(self.device)
        if not self.import_torch:
            # the parent must not pull torch/CUDA in before the weights are bound;
            # the device channel then comes from the worker's own reports
            self._torch = None
            return
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            self._torch = None
        else:
            self._torch = torch
            torch.set_num_threads(self.policy.cpu_threads)

    # ---------------------------------------------------------------- channels
    def _device_index(self) -> int:
        if self.device_index is not None:
            return self.device_index
        torch = self._torch
        if torch is not None and getattr(torch, "cuda", None) is not None:
            try:
                return int(torch.cuda.current_device())
            except Exception:  # pragma: no cover
                return 0
        return 0

    def channel_report(self) -> dict:
        """One snapshot of every channel, each with its own stated coverage."""
        memory = read_process_memory()
        report = {
            "process_commit_bytes": memory["commit_bytes"],
            "process_rss_bytes": memory["rss_bytes"],
            "process_commit_channel": memory["commit_channel"],
            "process_rss_channel": memory["rss_channel"],
            "device_kind": self.device_kind,
            "cuda_available": False,
            "torch_available": self._torch is not None,
            "coverage": dict(CHANNEL_COVERAGE),
        }
        if self.device_kind != "cuda":
            return report
        torch = self._torch
        if torch is None:
            report["device_query_error"] = "torch not importable"
            return report
        try:
            available = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - driver dependent
            report["device_query_error"] = f"{type(exc).__name__}: {exc}"
            return report
        report["cuda_available"] = available
        if not available:
            return report
        index = self._device_index()
        report["device_index"] = index
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            report["device_total_bytes"] = int(total_bytes)
            report["device_free_bytes"] = int(free_bytes)
            report["device_used_bytes"] = int(total_bytes - free_bytes)
            report["device_name"] = str(torch.cuda.get_device_properties(index).name)
        except Exception as exc:  # pragma: no cover - driver dependent
            report["device_query_error"] = f"{type(exc).__name__}: {exc}"
        return report

    def can_prove_headroom(self) -> tuple[bool, str]:
        """May the real backend start? Refuses whenever something is unmeasurable."""
        report = self.channel_report()
        commit = report["process_commit_bytes"]
        if commit is None:
            return False, "process commit charge cannot be measured"
        if commit > self.policy.ram_commit_limit_bytes:
            return False, (
                f"process commit {commit} B already exceeds the frozen "
                f"{self.policy.ram_commit_limit_bytes} B RAM budget"
            )
        if self.device_kind != "cuda":
            return True, "cpu device; process commit measurable"
        if not report["torch_available"]:
            return False, "torch unavailable, device usage cannot be measured"
        if not report["cuda_available"]:
            return False, "CUDA reports as unavailable"
        if "device_used_bytes" not in report:
            return False, f"device usage cannot be queried ({report.get('device_query_error')})"
        used = report["device_used_bytes"]
        if used > self.policy.vram_limit_bytes:
            return False, (
                f"device already in use for {used} B, above the frozen "
                f"{self.policy.vram_limit_bytes} B VRAM budget"
            )
        return True, (
            "monitoring channels verified; device total capacity "
            f"{report.get('device_total_bytes')} B is informational and is NOT compared to the budget"
        )

    # ------------------------------------------------------------------ sample
    def sample(self) -> dict:
        report = self.channel_report()
        payload = {
            "t": round(time.monotonic(), 6),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "process_commit_bytes": report["process_commit_bytes"],
            "process_rss_bytes": report["process_rss_bytes"],
            "ram_commit_measurable": report["process_commit_bytes"] is not None,
        }
        for key in ("device_used_bytes", "device_free_bytes", "device_total_bytes", "device_index"):
            if key in report:
                payload[key] = report[key]
        try:
            torch = self._torch
            if torch is not None and report.get("cuda_available"):
                index = self._device_index()
                payload["torch_allocated_bytes"] = int(torch.cuda.memory_allocated(index))
                payload["torch_reserved_bytes"] = int(torch.cuda.memory_reserved(index))
        except Exception:  # pragma: no cover - defensive
            payload["torch_memory_query_error"] = True
        return payload

    def violations(self, sample: dict | None = None) -> list[str]:
        """Threshold breaches, including 'unmeasurable' -- which is never safety."""
        sample = sample or self.sample()
        problems = []
        commit = sample.get("process_commit_bytes")
        if commit is None:
            problems.append("RAM commit charge unmeasurable in this sample; cannot be treated as within budget")
        elif commit > self.policy.ram_commit_limit_bytes:
            problems.append(f"RAM commit {commit} B > limit {self.policy.ram_commit_limit_bytes} B")
        if self.device_kind == "cuda":
            used = sample.get("device_used_bytes")
            if used is None:
                problems.append("device usage unmeasurable in this sample; cannot be treated as within budget")
            elif used > self.policy.vram_limit_bytes:
                problems.append(f"VRAM used {used} B > limit {self.policy.vram_limit_bytes} B")
        return problems

    @staticmethod
    def peaks(samples: list[dict]) -> dict:
        """Peaks over the whole series -- never 'the last sample'."""
        def peak(key: str):
            values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
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


def host_platform() -> str:
    return platform.platform()
