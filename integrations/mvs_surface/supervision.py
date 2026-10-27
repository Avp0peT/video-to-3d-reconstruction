"""Strict local adapter for the accepted Windows CAD Job supervisor.

Only the private module loaded for a call is adapted. The accepted supervisor
file is never changed. Job committed memory is a hard limit; elapsed time and
directory size are sampled stop thresholds, not hard wall/disk quotas.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import stat
import time
from types import ModuleType
from typing import Any, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_PATH = ROOT / "integrations/cad_delivery/supervise.py"
SUPERVISOR_SHA256 = "077a4a021741d1b432d6312bcedbab1bc31696cc6621b66d61dc65dda0d7e4be"
POLL_SECONDS = 0.1
RECLAMATION_SECONDS = 5.0
MAX_WALL_SECONDS = 1800.0
MIN_MEMORY_BYTES = 128 * 1024**2
MAX_MEMORY_BYTES = 12 * 1024**3
MAX_OUTPUT_BYTES = 8 * 1024**3
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
THREAD_ENV = {
    name: "2" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
}


class ObservationError(RuntimeError):
    """A resource observation is unknown and must stop the supervised work."""


def _no_link(info: os.stat_result, path: Path, *, file_links: bool = False) -> None:
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & REPARSE_POINT:
        raise ObservationError(f"symlink/reparse path is forbidden: {path}")
    if file_links and stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise ObservationError(f"non-unit file link count is forbidden (nlink={info.st_nlink}): {path}")


def _absolute_existing(value: str | os.PathLike[str], *, directory: bool) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"an absolute path without parent traversal is required: {path}")
    # Check ancestors before resolving, so resolution cannot hide a junction.
    for part in reversed((path, *path.parents)):
        info = part.lstat()
        _no_link(info, part)
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode):
        raise ValueError(f"expected {'directory' if directory else 'regular file'}: {path}")
    return path.resolve(strict=True)


def strict_directory_size(root: str | os.PathLike[str]) -> int:
    """Count regular files without following links or swallowing I/O failures.

    A disappearing file is also an unknown observation. This deliberately stops
    on it rather than silently undercounting. The check is not a filesystem
    sandbox; the runner must keep its run tree exclusive to this execution.
    """
    root = Path(root)
    try:
        root_info = root.lstat()
        _no_link(root_info, root)
        if not stat.S_ISDIR(root_info.st_mode):
            raise ObservationError(f"monitored root is not a directory: {root}")
        pending = [root]
        total = 0
        while pending:
            directory = pending.pop()
            info = directory.lstat()
            _no_link(info, directory)
            if not stat.S_ISDIR(info.st_mode):
                raise ObservationError(f"monitored directory changed type: {directory}")
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    # Windows DirEntry.stat may expose cached directory-entry
                    # metadata with st_nlink/st_ino/st_dev set to zero. lstat
                    # asks for the full file identity without following links.
                    info = path.lstat()
                    _no_link(info, path, file_links=True)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(path)
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                    else:
                        raise ObservationError(f"non-regular output entry is forbidden: {path}")
        return total
    except OSError as exc:
        raise ObservationError(f"directory observation failed for {root}: {exc}") from exc


RUNTIME_MAX_SKIPPED_ENTRIES = 128
RUNTIME_MAX_SKIP_SAMPLES = 512


def _runtime_root_identity(root: Path) -> tuple[int, int]:
    try:
        info = root.lstat()
    except FileNotFoundError as exc:
        raise ObservationError(f"monitored directory disappeared: {root}") from exc
    _no_link(info, root)
    if not stat.S_ISDIR(info.st_mode):
        raise ObservationError(f"monitored path is not a directory: {root}")
    if not info.st_ino:
        raise ObservationError(f"directory identity unavailable: {root}")
    return info.st_dev, info.st_ino


def runtime_directory_observation(root: str | os.PathLike[str]) -> dict[str, Any]:
    """One non-atomic byte sample, tolerating only proven regular-file ENOENT.

    Root and every discovered directory retain their identity throughout the
    walk. A vanished regular file contributes unknown bytes, never an invented
    zero-size file. The sum is sequential observed bytes, not bytes at one instant: files
    counted earlier may vanish and renames may be counted twice. It is not a
    mathematical bound, complete snapshot or hard disk quota. Preflight/final scans stay strict.
    """
    root = Path(root)
    directories = {root: _runtime_root_identity(root)}
    pending = [root]; total = 0; skipped = []; file_count = 0; metadata_refreshes = []

    def verify_directories():
        for path, identity in directories.items():
            if _runtime_root_identity(path) != identity:
                raise ObservationError(f"monitored directory identity changed: {path}")

    try:
        while pending:
            directory = pending.pop()
            verify_directories()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    was_file = entry.is_file(follow_symlinks=False)
                    was_directory = entry.is_dir(follow_symlinks=False)
                    try:
                        info = path.lstat()
                    except FileNotFoundError:
                        # A missing directory, symlink or unknown entry is not
                        # a supported native temporary-file rotation.
                        if not was_file or was_directory:
                            raise ObservationError(f"missing non-regular/unknown entry: {path}")
                        verify_directories()
                        skipped.append(dict(path=str(path), reason="ENUMERATED_REGULAR_FILE_ENOENT"))
                        if len(skipped) > RUNTIME_MAX_SKIPPED_ENTRIES:
                            raise ObservationError("runtime vanished-entry observation bound exceeded")
                        continue
                    _no_link(info, path, file_links=False)
                    # Windows can fall back to incomplete path metadata during
                    # DeleteFile(raw) -> MoveFile(geo, raw): nlink/ino/dev = 0.
                    # Do not treat zero as a hardlink or as a measured zero size.
                    # Query exactly once more; only independently valid regular
                    # metadata or an explicit disappearance is admissible.
                    if stat.S_ISREG(info.st_mode) and info.st_nlink == 0:
                        if not was_file or was_directory:
                            raise ObservationError(f"zero-link metadata for unknown entry: {path}")
                        refresh = dict(path=str(path), first_nlink=0,
                            first_dev=getattr(info, 'st_dev', None), first_ino=getattr(info, 'st_ino', None),
                            first_size=info.st_size, reason='ZERO_LINK_METADATA_SINGLE_REFRESH')
                        verify_directories()
                        if len(metadata_refreshes) >= RUNTIME_MAX_SKIPPED_ENTRIES:
                            raise ObservationError('runtime metadata-refresh bound exceeded')
                        try:
                            refreshed = path.lstat()
                        except FileNotFoundError:
                            verify_directories()
                            refresh['outcome'] = 'CONFIRMED_DISAPPEARED_UNKNOWN_BYTES'
                            metadata_refreshes.append(refresh)
                            skipped.append(dict(path=str(path), reason='ZERO_LINK_THEN_ENOENT'))
                            if len(skipped) > RUNTIME_MAX_SKIPPED_ENTRIES:
                                raise ObservationError('runtime vanished-entry observation bound exceeded')
                            continue
                        _no_link(refreshed, path, file_links=True)
                        if not stat.S_ISREG(refreshed.st_mode) or not refreshed.st_ino:
                            raise ObservationError(f'refresh did not establish regular file identity: {path}')
                        verify_directories()
                        refresh.update(outcome='VALID_REGULAR_METADATA', nlink=refreshed.st_nlink,
                            dev=refreshed.st_dev, ino=refreshed.st_ino, size=refreshed.st_size)
                        metadata_refreshes.append(refresh)
                        info = refreshed
                    _no_link(info, path, file_links=True)
                    if stat.S_ISDIR(info.st_mode):
                        if not was_directory:
                            raise ObservationError(f"enumerated entry changed type: {path}")
                        identity = (info.st_dev, info.st_ino)
                        if not identity[1]:
                            raise ObservationError(f"directory identity unavailable: {path}")
                        previous = directories.setdefault(path, identity)
                        if previous != identity:
                            raise ObservationError(f"monitored directory identity changed: {path}")
                        pending.append(path)
                    elif stat.S_ISREG(info.st_mode):
                        if not was_file:
                            raise ObservationError(f"enumerated entry changed type: {path}")
                        total += info.st_size; file_count += 1
                    else:
                        raise ObservationError(f"non-regular output entry is forbidden: {path}")
            verify_directories()
        verify_directories()
        if skipped and total == 0:
            raise ObservationError("no measured bytes while regular entries disappeared; unknown is not zero")
        return dict(bytes=total, regular_files_observed=file_count, directories_checked=len(directories),
                    root_identity=list(directories[root]), skipped_entries=skipped,
                    skipped_entry_count=len(skipped), metadata_refreshes=metadata_refreshes,
                    metadata_refresh_count=len(metadata_refreshes), scan_attempts=1,
                    semantics="NONATOMIC_OBSERVED_BYTES_WITH_DISAPPEARING_REGULAR_FILES", complete_snapshot=False)
    except OSError as exc:
        raise ObservationError(f"runtime directory observation failed for {root}: {exc}") from exc


def runtime_directory_size(root: str | os.PathLike[str]) -> int:
    """Compatibility integer API; supervision additionally records its sample."""
    return runtime_directory_observation(root)["bytes"]


def _load_supervisor() -> tuple[ModuleType, dict[str, Any]]:
    source = SUPERVISOR_PATH.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SUPERVISOR_SHA256:
        raise ValueError("accepted CAD supervisor hash does not match")
    name = f"_mvs_surface_supervisor_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SUPERVISOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct the private supervisor module")
    module = importlib.util.module_from_spec(spec)
    # Execute the exact bytes whose hash was checked, not a second loader read.
    exec(compile(source, str(SUPERVISOR_PATH), "exec"), module.__dict__)
    original_job = module.JobObject
    original_log = module.EventLog
    log_failures: list[dict[str, Any]] = []
    directory_samples: dict[str, Any] = dict(samples=0, skipped_samples=[],
        semantics='NONATOMIC_OBSERVED_BYTES_WITH_DISAPPEARING_REGULAR_FILES', complete_snapshot=False)
    pinned_roots: dict[str, tuple[int, int]] = {}

    def observed_directory_size(root):
        key = str(Path(root))
        identity = _runtime_root_identity(Path(root))
        if pinned_roots.setdefault(key, identity) != identity:
            raise ObservationError(f'monitored root identity changed between samples: {root}')
        sample = runtime_directory_observation(root)
        if tuple(sample['root_identity']) != identity:
            raise ObservationError(f'monitored root identity changed during sample: {root}')
        directory_samples['samples'] += 1
        directory_samples['last_sample'] = sample
        if sample['skipped_entry_count'] or sample.get('metadata_refresh_count', 0):
            if len(directory_samples['skipped_samples']) >= RUNTIME_MAX_SKIP_SAMPLES:
                raise ObservationError('runtime skipped-sample audit bound exceeded')
            directory_samples['skipped_samples'].append(dict(sample_index=directory_samples['samples'], **sample))
        return sample['bytes']

    class StrictJobObject(original_job):
        def accounting(self) -> dict[str, Any]:
            try:
                return super().accounting()
            except module.JobError as exc:
                # This is deliberately not JobError: the old sampling handler
                # must not catch it and continue. Outer fallback reclaims first.
                # If it recurs during reclaim, reclaim remains unconfirmed.
                raise ObservationError(f"Job accounting observation failed: {exc}") from exc

    class ObservedEventLog(original_log):
        def append(self, record: dict[str, Any]) -> None:
            try:
                super().append(record)
            except Exception as exc:
                # Preserve the original exception type: the accepted supervisor
                # relies on LogError handlers to finish reclamation. Some late
                # writes are intentionally best-effort there, so retain their
                # failure here until even its final cleanup block has returned.
                log_failures.append({
                    "kind": record.get("kind"),
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
                raise

    module.JobObject = StrictJobObject
    module.EventLog = ObservedEventLog
    module.directory_size = observed_directory_size
    module._mvs_directory_samples = directory_samples
    module._mvs_eventlog_failures = log_failures
    return module, {"path": str(SUPERVISOR_PATH), "bytes": len(source), "sha256": digest}


def _apply_log_failure_gate(
    result: dict[str, Any], failures: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Keep swallowed reclaim/finally log failures from becoming success."""
    if failures:
        result.setdefault("log_errors", []).extend(
            {**failure, "source": "mvs_surface_event_log"} for failure in failures
        )
        if result.get("status") == "completed":
            result["status"] = "completed_with_errors"
    return result


def _bounded_bytes(value: int, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _registry_program_w6432() -> str:
    """Read the OS installation path when our own parent is also isolated."""
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"SOFTWARE\Microsoft\Windows\CurrentVersion", 0,
                        winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
        value, kind = winreg.QueryValueEx(key, "ProgramW6432Dir")
    if kind != winreg.REG_SZ or not isinstance(value, str) or not value:
        raise ValueError("Windows ProgramW6432Dir is not a nonempty directory path")
    return value


def _windows_platform_environment() -> dict[str, str]:
    """Retain the Windows runtime settings required by native entry points.

    This does not inherit the host environment. A local single-variable probe
    established that omitting ProgramW6432 makes nvidia-smi return 255, while
    restoring just this path permits the same query under the same Job limits.
    PowerShell 5 also needs PATHEXT to recognize .exe native commands. Use a
    fixed minimal extension list, not host script/file-association extensions.
    No GPU is queried here, and no resource/freshness gate changes.
    """
    value = os.environ.get("ProgramW6432")
    if value is None:
        value = _registry_program_w6432()
    directory = _absolute_existing(value, directory=True)
    return {"ProgramW6432": str(directory), "PATHEXT": ".COM;.EXE;.BAT;.CMD"}


def run_supervised(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str],
    output: str | os.PathLike[str],
    wall_seconds: float,
    memory_bytes: int,
    output_bytes: int,
    label: str,
) -> dict[str, Any]:
    """Run one stage; the caller owns overall deadlines and stage decisions.

    ``cwd`` is the exclusive whole run tree monitored for size. ``output`` is
    an existing stage evidence directory within it, unused by any earlier
    launch. This function returns the supervisor result and adapter metadata;
    the caller must persist it even when the native stage failed.
    """
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise ValueError("argv must be a nonempty sequence of strings")
    if any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
        raise ValueError("argv entries must be NUL-free strings")
    if not argv[0]:
        raise ValueError("argv executable cannot be empty")
    if type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError(f"wall_seconds must be finite in (0, {MAX_WALL_SECONDS}]")
    memory_bytes = _bounded_bytes(memory_bytes, "memory_bytes", MIN_MEMORY_BYTES, MAX_MEMORY_BYTES)
    output_bytes = _bounded_bytes(output_bytes, "output_bytes", 1, MAX_OUTPUT_BYTES)
    if not isinstance(label, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label) is None:
        raise ValueError("label must contain 1-64 safe ASCII identifier characters")
    work = _absolute_existing(cwd, directory=True)
    evidence = _absolute_existing(output, directory=True)
    executable = _absolute_existing(argv[0], directory=False)
    if not evidence.is_relative_to(work):
        raise ValueError("stage evidence output must be inside the monitored cwd")
    for name in ("supervisor.jsonl", "stdout.txt", "stderr.txt", "temp"):
        if os.path.lexists(evidence / name):
            raise ValueError(f"stage evidence path already exists: {evidence / name}")
    if strict_directory_size(work) > output_bytes:
        raise ValueError("run tree already exceeds the output stop threshold")
    if os.name != "nt":
        raise RuntimeError("the accepted Job supervisor requires Windows")
    system_root = _absolute_existing(os.environ.get("SystemRoot", r"C:\Windows"), directory=True)
    system32 = _absolute_existing(system_root / "System32", directory=True)
    module, binding = _load_supervisor()
    adapter_path = Path(__file__).resolve(strict=True)
    adapter_bytes = adapter_path.read_bytes()
    adapter_binding = {
        "path": str(adapter_path), "bytes": len(adapter_bytes),
        "sha256": hashlib.sha256(adapter_bytes).hexdigest(),
    }
    temp = evidence / "temp"
    temp.mkdir(exist_ok=False)
    environment = {
        "SystemRoot": str(system_root), "WINDIR": str(system_root),
        "TEMP": str(temp), "TMP": str(temp),
        "PATH": os.pathsep.join((str(executable.parent), str(system32))),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTHONIOENCODING": "utf-8", **THREAD_ENV,
        **_windows_platform_environment(),
    }
    result = module.supervise(
        list(argv), cwd=work, env_overrides=environment, inherit_env=False,
        log_path=evidence / "supervisor.jsonl",
        stdout_path=evidence / "stdout.txt", stderr_path=evidence / "stderr.txt",
        job_memory_limit_bytes=memory_bytes, wall_clock_seconds=float(wall_seconds),
        run_dir=work, run_dir_threshold_bytes=output_bytes,
        poll_seconds=POLL_SECONDS, reclamation_seconds=RECLAMATION_SECONDS,
        label=label, script_sha256=adapter_binding["sha256"],
    )
    _apply_log_failure_gate(result, module._mvs_eventlog_failures)
    result["mvs_surface_adapter"] = {
        "schema": "mvs-surface-supervision-v1", "binding": adapter_binding,
        "accepted_supervisor": binding, "inherit_env": False,
        "environment": environment, "output": str(evidence),
        "limits": {
            "job_committed_memory": "HARD_JOB_LIMIT",
            "wall_clock": "POLLED_STOP_THRESHOLD",
            "directory_bytes": "POLLED_STOP_THRESHOLD",
            "cpu": "THREAD_ARGUMENT_AND_ENVIRONMENT_ONLY",
            "reclamation_seconds_after_stop": RECLAMATION_SECONDS,
        },
        "accounting_failure": "STOP_AND_ATTEMPT_RECLAMATION",
        "directory_observation_failure": "STOP_AND_ATTEMPT_RECLAMATION",
        "directory_disappearing_entry": "ONLY_ENUMERATED_REGULAR_FILE_ENOENT_WITH_ALL_DIRECTORY_IDENTITIES_STABLE",
        "directory_runtime_scan_attempts": 1,
        "directory_runtime_retry_backoff_seconds": [],
        "zero_link_metadata_policy": "ONE_LSTAT_REFRESH_REQUIRE_VALID_REGULAR_IDENTITY_OR_ENOENT",
        "directory_runtime_observation": module._mvs_directory_samples,
        "directory_runtime_max_skipped_entries_per_sample": RUNTIME_MAX_SKIPPED_ENTRIES,
        "directory_runtime_max_skipped_samples": RUNTIME_MAX_SKIP_SAMPLES,
        "logging_failure": "ALL_APPEND_FAILURES_RECORDED_AND_REJECT_SUCCESS_NO_IMMEDIATE_STOP_GUARANTEE",
        "native_launch_count": int(result.get("pid") is not None),
    }
    return result
