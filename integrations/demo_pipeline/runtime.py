"""One segmentation stage using the unchanged, pinned SAM3 mmap worker.

The caller supplies the outer Job supervisor. This module imports only standard
library modules until preflight passes; torch and SAM3 belong in the worker.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_SHA256 = "6704d9a2331bb6c82500bd0fa58d276e21188cc407c84296539b60c76d36f843"
PACKAGE_RELATIVE = "outputs/delegation/DEMO_RUNTIME/20261011_text_v2"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PATH_FIELDS = ("root", "io_run", "output_dir", "checkpoint", "worker_package", "sam3_python")
FIELDS = {*PATH_FIELDS, "checkpoint_sha256", "text_prompt", "object_id", "indices"}
RESERVED = ("SEGMENT_ENTRY.json", "PRECONDITIONS.json", "REQUEST.json", "PRODUCT_LAUNCH.json",
            "PRODUCT_OUTCOME.json", "POST_BINDINGS.json", "product_run")


@dataclass(frozen=True)
class SegmentConfig:
    root: Path
    io_run: Path
    output_dir: Path
    checkpoint: Path
    checkpoint_sha256: str
    worker_package: Path
    sam3_python: Path
    text_prompt: str
    object_id: str
    indices: list[int]
    git_exe: Path | None = None


def discover_git(explicit=None) -> Path | None:
    """Resolve Git once; isolated child PATH need not contain its installation."""
    if explicit is not None:
        candidates=[Path(explicit).expanduser()]
    else:
        found=shutil.which('git')
        candidates=([Path(found)] if found else [])
        if os.name=='nt':
            candidates += [Path(r'C:\Program Files\Git\cmd\git.exe'),
                           Path(r'C:\Program Files (x86)\Git\cmd\git.exe')]
    for candidate in candidates:
        if candidate.is_file():
            _no_links(candidate)
            return candidate.resolve(strict=True)
    return None


def read_json(path: Path):
    if path.stat().st_size > 8 << 20:
        raise ValueError(f"JSON exceeds 8 MiB: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(pending, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp() -> dict:
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic": time.monotonic()}


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"linked/reparse path refused: {part}")


def load_config(path: str | Path) -> SegmentConfig:
    raw = read_json(Path(path))
    if not isinstance(raw, dict) or set(raw) not in (FIELDS,FIELDS|{'git_exe'}):
        raise ValueError("segmentation config must contain exactly: " + ", ".join(sorted(FIELDS)))
    for key in PATH_FIELDS:
        if not isinstance(raw[key], str) or not Path(raw[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
        candidate = Path(raw[key])
        _no_links(candidate)
        raw[key] = candidate.resolve(strict=True)
    if raw["root"] != PROJECT_ROOT.resolve():
        raise ValueError("root must equal this project; the audited worker is bound to that project")
    for key in ("root", "io_run", "output_dir", "worker_package"):
        if not raw[key].is_dir():
            raise ValueError(f"{key} must be an existing directory")
    if raw["worker_package"] != (raw["root"] / PACKAGE_RELATIVE).resolve():
        raise ValueError("worker_package must be the accepted text_v2 runtime profile")
    if raw["sam3_python"] != (raw["root"] / "integrations/sam3/env/Scripts/python.exe").resolve():
        raise ValueError("sam3_python must be the existing isolated SAM3 interpreter")
    output = raw["output_dir"]
    if not output.is_relative_to(raw["root"] / "outputs") or output == raw["root"] / "outputs":
        raise ValueError("output_dir must be strictly within project outputs")
    for protected in (raw["io_run"], raw["worker_package"]):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("segmentation output must be disjoint from inputs and sealed package")
    for key in ("text_prompt", "object_id"):
        if not isinstance(raw[key], str) or not raw[key].strip() or len(raw[key]) > 2048:
            raise ValueError(f"{key} must be a nonempty bounded string")
    indices = raw["indices"]
    if (not isinstance(indices, list) or not 1 <= len(indices) <= 96
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("indices must contain 1..96 unique nonnegative integers")
    digest = raw["checkpoint_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256")
    if 'git_exe' in raw and (not isinstance(raw['git_exe'],str) or not Path(raw['git_exe']).is_absolute()):
        raise ValueError('git_exe must be an absolute path')
    raw['git_exe']=discover_git(raw.get('git_exe'))
    if raw['git_exe'] is None:raise ValueError('Git executable missing; specify git_exe absolute path')
    return SegmentConfig(**raw)


def _runtime_dependency(relative: str) -> bool:
    return (relative.startswith(("integrations/sam3/sam3_adapter/",
                                 "integrations/sam3/src/sam3-repo/"))
            or relative in {"integrations/sam3/VERSIONS.json",
                "integrations/sam3/env/Lib/site-packages/torch/serialization.py",
                "integrations/sam3/env/Lib/site-packages/torch/nn/modules/module.py"}
            or relative.startswith("outputs/delegation/SAM3_GROUNDING_DIAG/20261008_main_r1/worker_entry/"))


def verify_bindings(config: SegmentConfig) -> dict:
    """Check pinned package bytes and its explicit runtime dependency subset.

    Historical input images, docs and tests do not become new dataset inputs.
    The original manifests remain unchanged; every excluded path is reported.
    """
    package = config.worker_package
    actual_bundle = sha256(package / "BUNDLE.json")
    if actual_bundle != BUNDLE_SHA256:
        raise ValueError("sealed BUNDLE.json digest differs from accepted text_v2 profile")
    bundle = read_json(package / "BUNDLE.json")
    checked, excluded, errors = [], [], []
    manifests = [("bundle", package, bundle, False)]
    # Verify the manifest files through the bundle before trusting their contents.
    for row in bundle["files"]:
        if row["path"] in ("START_BASELINE.json", "AUTHORITY_BINDINGS.json"):
            if sha256(package / row["path"]) != row["sha256"]:
                raise ValueError("sealed manifest drift: " + row["path"])
    for name in ("START_BASELINE.json", "AUTHORITY_BINDINGS.json"):
        manifests.append((name, config.root, read_json(package / name), True))
    for name, base, document, filtered in manifests:
        seen = set()
        for row in document["files"]:
            relative = row["path"].replace("\\", "/")
            target = base / relative
            if relative in seen or Path(relative).is_absolute() or not target.resolve().is_relative_to(base):
                raise ValueError(f"invalid or duplicate protected path in {name}: {relative}")
            seen.add(relative)
            if filtered and not _runtime_dependency(relative):
                excluded.append({"manifest": name, "path": relative,
                                 "reason": "not read by new segmentation runtime; historical provenance retained"})
                continue
            _no_links(target)
            actual = sha256(target) if target.is_file() else None
            checked.append({"manifest": name, "path": relative, "sha256": actual,
                            "expected_sha256": row["sha256"], "match": actual == row["sha256"]})
            if actual != row["sha256"]:
                errors.append(f"{name}: {relative}")
    return {"bundle_sha256": actual_bundle, "checked": checked, "excluded": excluded,
            "errors": errors, "match": not errors}


def _heavy_modules() -> list[str]:
    return sorted(name for name in sys.modules if name.split(".", 1)[0] in {"torch", "sam3"})


def preflight(config: SegmentConfig) -> dict:
    report = {"status": "CHECKING", "started": stamp(), "executable": sys.executable,
              "checkpoint_rehashed": False, "heavy_imports": _heavy_modules(), "errors": []}
    try:
        if Path(sys.executable).resolve() != config.sam3_python:
            raise ValueError("wrong interpreter: use the isolated SAM3 Python")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValueError("CUDA_VISIBLE_DEVICES must be unset for the audited device identity gate")
        if report["heavy_imports"]:
            raise ValueError("parent imported torch/SAM3 before preflight")
        git_exe=discover_git(config.git_exe)
        if git_exe is None:raise ValueError('Git executable missing; specify git_exe absolute path')
        report['git_tool_binding']={'path':str(git_exe),'bytes':git_exe.stat().st_size,'sha256':sha256(git_exe)}
        report['git_commands']=[]
        report["bindings"] = verify_bindings(config)
        if not report["bindings"]["match"]:
            raise ValueError("protected runtime source drift: " + "; ".join(report["bindings"]["errors"]))
        base = read_json(config.worker_package / "START_BASELINE.json")
        authority = read_json(config.worker_package / "AUTHORITY_BINDINGS.json")
        if Path(authority["project_root"]).resolve() != config.root:
            raise ValueError("audited worker authority binds a different project root")
        bound = base["checkpoint"]
        if config.checkpoint != Path(bound["path"]).resolve() or config.checkpoint_sha256 != bound["sha256"]:
            raise ValueError("checkpoint path/digest differ from the audited local checkpoint")
        info = config.checkpoint.stat()
        expected = authority["checkpoint_stat_only"]
        if (info.st_size, info.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
            raise ValueError("checkpoint stat differs from worker authority")
        report["checkpoint_actual_sha256"] = sha256(config.checkpoint)
        report["checkpoint_rehashed"] = True
        if report["checkpoint_actual_sha256"] != config.checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 mismatch")
        site = config.sam3_python.parent.parent / "Lib/site-packages"
        packages = sorted(({"name": d.metadata["Name"], "version": d.version}
                           for d in metadata.distributions(path=[str(site)])), key=lambda item: item["name"].lower())
        report["packages_checked"] = len(packages)
        report["package_inventory_match"] = packages == base["package_inventory"]
        if not report["package_inventory_match"]:
            raise ValueError("SAM3 environment package inventory differs from accepted baseline")
        repo = config.root / "integrations/sam3/src/sam3-repo"
        for key, arguments in (("checkout_head", ["rev-parse", "HEAD"]),
                               ("checkout_status", ["status", "--porcelain"])):
            argv=[str(git_exe), '-C', str(repo), *arguments]
            report['git_commands'].append(argv)
            result = subprocess.run(argv, capture_output=True,
                                    text=True, check=True, timeout=10)
            report[key] = result.stdout.strip()
        if report["checkout_head"] != base["checkout_head"] or report["checkout_status"]:
            raise ValueError("SAM3 checkout revision or clean-state differs")
        if report['git_tool_binding']!={'path':str(git_exe),'bytes':git_exe.stat().st_size,'sha256':sha256(git_exe)}:
            raise ValueError('Git executable changed during checkout validation')
        report["free_disk_bytes"] = shutil.disk_usage(config.output_dir).free
        if report["free_disk_bytes"] < 10 << 30:
            raise ValueError("less than 10 GiB free disk for segmentation")
        report["input_bindings"] = {name: sha256(config.io_run / name) for name in ("run.json", "frames.jsonl")}
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "PRECONDITION_BLOCKED"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["finished"] = stamp()
    return report


def _environment(config: SegmentConfig) -> dict[str, str]:
    output = config.output_dir
    values = {key: "2" for key in THREAD_ENV}
    values.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        PYTHONPATH=os.pathsep.join([str(config.worker_package / "worker_entry"), str(config.root / "integrations/sam3")]),
        SAM3_REAL3_AUDIT_PATH=str(output / "gate_events.jsonl"),
        SAM3_MEMORY_DIAG_PATH=str(output / "stage_memory.jsonl"),
        SAM3_MEMORY_DIAG_STATUS_PATH=str(output / "diag_status.json"),
        SAM3_MMAP_DIAG_PATH=str(output / "mmap_forward.jsonl"),
        SAM3_MMAP_STATUS_PATH=str(output / "mmap_status.json"), SAM3_AUDIT_IMPORT_ONLY="0")
    for key, folder in (("TEMP", "tmp"), ("TMP", "tmp"), ("TORCH_HOME", "torch_cache"),
                        ("TRITON_CACHE_DIR", "triton_cache"), ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache"),
                        ("CUDA_CACHE_PATH", "cuda_cache")):
        target = output / folder
        _no_links(target)
        target.mkdir(exist_ok=True)
        values[key] = str(target)
    return values


def _adapter(config: SegmentConfig):
    sys.path.insert(0, str(config.root / "integrations/sam3"))
    from sam3_adapter import runner, worker
    from sam3_adapter.deadline import Budgets
    from sam3_adapter.resources import ResourcePolicy
    for module in (runner, worker):
        if not Path(module.__file__).resolve().is_relative_to(config.root / "integrations/sam3/sam3_adapter"):
            raise ValueError("already-imported adapter came from another checkout")
    return SimpleNamespace(runner=runner, worker=worker, Budgets=Budgets, ResourcePolicy=ResourcePolicy)


def _evidence(output: Path) -> dict:
    """Summarize actual worker evidence, without importing its assessment code."""
    summary = {"mmap_status": "NOT_AVAILABLE", "kernel_gate": "NOT_AVAILABLE",
               "independent_full_assessment": "NOT_RUN", "errors": []}
    try:
        status = read_json(output / "mmap_status.json")
        summary["mmap_worker_record"] = status
        complete = (status.get("complete") is True and status.get("business_rc") == 0
            and all(status.get(key) == 0 for key in ("write_failures", "dropped_events",
                "restoration_errors", "diagnostic_errors", "active_calls"))
            and sha256(output / "mmap_forward.jsonl") == status.get("events_sha256"))
        summary["mmap_status"] = "WORKER_EVIDENCE_COMPLETE" if complete else "INCOMPLETE"
        path = output / "gate_events.jsonl"
        if path.stat().st_size > 16 << 20:
            raise ValueError("gate ledger exceeds 16 MiB")
        gates = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        gate_a = [row for row in gates if row.get("section") == "gate_a"]
        if len(gate_a) == 1:
            summary["kernel_gate"] = gate_a[0].get("status", "UNKNOWN")
            summary["kernel_gate_record"] = gate_a[0]
    except (OSError, ValueError, TypeError) as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    return summary


def _run(config: SegmentConfig, env: dict, argv: list[str], *, worker_module="mmap_review_worker") -> dict:
    api = _adapter(config)
    request = api.runner.RunRequest(io_run=config.io_run, indices=config.indices,
        object_id=config.object_id, text_prompt=config.text_prompt, output_root=config.output_dir / "product_run",
        repo_dir=config.root / "integrations/sam3/src/sam3-repo", checkpoint=config.checkpoint,
        checkpoint_sha256=config.checkpoint_sha256, device="cuda", confidence_threshold=0.5,
        resolution=1008, allow_hf_download=False, use_worker=True, worker_python=str(config.sam3_python),
        worker_adapter_dir=config.root / "integrations/sam3", worker_module=worker_module,
        worker_extra_env=env, device_observe="auto",
        resource_policy=api.ResourcePolicy(ram_commit_limit_bytes=12 << 30, vram_limit_bytes=7 << 30, cpu_threads=2),
        budgets=api.Budgets(load_seconds=300, frame_seconds=300, batch_seconds=900, reclaim_seconds=10),
        argv=argv, cwd=str(config.root), env_allowlist=list(env))
    original = api.worker.launch_worker
    launch_path = config.output_dir / "PRODUCT_LAUNCH.json"
    record = {"launched": False, "pid": None, "returncode": None}
    process = None
    result = None
    failure = None

    def recorded_launch(python, job_path, adapter_dir, stdout_path, stderr_path,
                        extra_env=None, cwd=None, module="sam3_adapter.worker"):
        nonlocal process
        if "before" in record:
            raise RuntimeError("this segmentation stage allows only one worker launch attempt")
        record.update(before=stamp(), argv=[python, "-m", module, "--job", str(Path(job_path).resolve())],
                      cwd=str(cwd or adapter_dir), env_overrides=extra_env)
        write_json(launch_path, record)
        process = original(python, job_path, adapter_dir, stdout_path, stderr_path, extra_env, cwd, module)
        record.update(launched=True, pid=process.pid, actual_popen_args=process.args, after_create=stamp())
        try:
            write_json(launch_path, record)
        except BaseException:
            api.worker.kill_tree(process, time.monotonic() + 5)
            raise
        return process

    api.worker.launch_worker = recorded_launch
    try:
        outcome = api.runner.execute(request)
        manifest = outcome.manifest or {}
        record["ownership_identity"] = manifest.get("worker", {}).get("ownership_identity")
        result = {"status": outcome.status, "exit_code": outcome.exit_code,
                "run_dir": str(outcome.run_dir) if outcome.run_dir else None,
                "counts": manifest.get("counts"), "stop_state": manifest.get("stop_state")}
    except BaseException as exc:
        failure = exc
        record["error"] = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            try:
                record["fallback_cleanup"] = api.worker.kill_tree(process, time.monotonic() + 5)
            except Exception as cleanup_error:
                record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        raise
    finally:
        api.worker.launch_worker = original
        record["end"] = stamp()
        record["returncode"] = process.poll() if process is not None else None
        try:
            write_json(launch_path, record)
        except Exception as evidence_error:
            message = f"launch evidence write failed: {type(evidence_error).__name__}: {evidence_error}"
            if result is not None:
                result.update(product_exit_code=result["exit_code"], product_status=result["status"],
                              status="BACKEND_ERROR", exit_code=4, error=message)
            elif failure is not None:
                failure.add_note(message)
            else:
                raise
    return result


def execute_config(path: str | Path, *, argv: list[str] | None = None, score_diagnostics=False) -> int:
    """Run once from JSON; return product/wrapper exit code and preserve failure facts."""
    config = load_config(path)
    output = config.output_dir
    if any((output / name).exists() for name in RESERVED):
        raise ValueError("segmentation stage already has execution artifacts; no retry or overwrite")
    config_path = Path(path).resolve()
    for child in output.iterdir():
        _no_links(child)
        if child.resolve() != config_path or not child.is_file():
            raise ValueError("segmentation output must contain only its new config file before launch")
    actual_argv = argv or [sys.executable, *sys.argv]
    with (output / "SEGMENT_ENTRY.json").open("x", encoding="utf-8") as handle:
        json.dump({"started": stamp(), "argv": actual_argv, "config_path": str(Path(path).resolve()),
                   "config_sha256": sha256(Path(path)), "one_worker_attempt": True}, handle, indent=2)
    write_json(output / "PRODUCT_LAUNCH.json", {"launched": False, "pid": None, "returncode": None})
    result = {"status": "BACKEND_ERROR", "exit_code": 4, "run_dir": None, "counts": None}
    previous_env, previous_path = dict(os.environ), list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    try:
        os.environ.update({key: "2" for key in THREAD_ENV})
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        checks = preflight(config)
        write_json(output / "PRECONDITIONS.json", checks)
        if checks["status"] != "PASS":
            result.update(status="PRECONDITION_BLOCKED", exit_code=3, errors=checks["errors"])
        else:
            env = _environment(config)
            diagnostic_binding = None
            if score_diagnostics:
                from integrations.demo_pipeline import sam_score_worker
                diagnostic_binding = sam_score_worker.configure(config, env)
                write_json(output / "SCORE_DIAGNOSTIC_BINDING.json", diagnostic_binding)
            os.environ.update({key: value for key, value in env.items() if key != "PYTHONPATH"})
            write_json(output / "REQUEST.json", {key: str(value) if isinstance(value, Path) else value
                                                  for key, value in vars(config).items()})
            if score_diagnostics:
                result = _run(config, env, actual_argv,
                              worker_module="integrations.demo_pipeline.sam_score_worker")
            else:
                result = _run(config, env, actual_argv)
            result.setdefault("product_exit_code", result["exit_code"])
            result["evidence"] = _evidence(output)
            if score_diagnostics:
                result["score_diagnostics"] = sam_score_worker.assess(config, diagnostic_binding)
                if result["exit_code"] in (0, 1) and result["score_diagnostics"]["status"] != "COMPLETE":
                    result.update(status="BACKEND_ERROR", exit_code=4,
                                  error="score diagnostic evidence incomplete; business result retained")
            post = verify_bindings(config)
            write_json(output / "POST_BINDINGS.json", post)
            if not post["match"] or _heavy_modules():
                result.update(status="BACKEND_ERROR", exit_code=4, error="runtime binding drift or parent heavy import")
            if result["exit_code"] == 0 and (result["evidence"]["mmap_status"] != "WORKER_EVIDENCE_COMPLETE"
                                            or result["evidence"]["kernel_gate"] != "PASS"):
                result.update(status="BACKEND_ERROR", exit_code=4, error="required worker evidence incomplete")
    except Exception as exc:
        result.update(status="BACKEND_ERROR", exit_code=4, error=f"{type(exc).__name__}: {exc}")
        if getattr(exc, "__notes__", None):
            result["evidence_errors"] = exc.__notes__
        if not (output / "PRECONDITIONS.json").exists():
            write_json(output / "PRECONDITIONS.json", {"status": "PRECONDITION_BLOCKED", "errors": [result["error"]]})
    finally:
        os.environ.clear()
        os.environ.update(previous_env)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        result["finished"] = stamp()
        write_json(output / "PRODUCT_OUTCOME.json", result)
    return result["exit_code"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One new SAM3 stage under an outer Job supervisor")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args(argv)
    try:
        return execute_config(arguments.config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"INPUT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
