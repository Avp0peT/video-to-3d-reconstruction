"""Environment / CUDA / Hugging Face authentication probes.

These run in the isolated SAM 3 interpreter and print a single JSON document on
stdout so the calling harness can archive it verbatim. No credential value is
ever printed -- only whether a token is present.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import imagery
from .backends import (
    SAM3_CONFIG_FILE,
    SAM3_HF_URL,
    SAM3_REPO_ID,
    AuthRequired,
    BackendUnavailable,
    Sam3Backend,
)

#: variables whose *names* (never values) are recorded as the process allowlist
ENV_ALLOWLIST_NAMES = (
    "PATH",
    "SystemRoot",
    "windir",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PROCESSOR_ARCHITECTURE",
)


def env_allowlist() -> dict:
    present = [name for name in ENV_ALLOWLIST_NAMES if os.environ.get(name) is not None]
    return {"names_present": present, "values_recorded": False}


def env_probe() -> dict:
    report = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "in_virtualenv": sys.prefix != sys.base_prefix,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "env_allowlist": env_allowlist(),
        "packages": {},
    }
    for module_name in ("torch", "torchvision", "numpy", "PIL", "timm", "huggingface_hub", "sam3"):
        try:
            module = __import__(module_name)
        except Exception as exc:  # noqa: BLE001
            report["packages"][module_name] = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        entry = {"importable": True, "version": getattr(module, "__version__", None)}
        if module_name == "torch":
            entry["cuda_runtime"] = getattr(getattr(module, "version", None), "cuda", None)
            entry["cuda_kernels"] = getattr(module.version, "cuda", None)
        report["packages"][module_name] = entry
    return report


def cuda_probe(device_index: int = 0) -> dict:
    """Minimal CUDA device probe plus one small kernel execution."""
    report = {"probe": "cuda", "device_index": device_index, "ok": False}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            report["nvidia_smi"] = out.stdout.strip() or out.stderr.strip()
            report["nvidia_smi_returncode"] = out.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            report["nvidia_smi"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        report["error"] = f"torch not importable: {exc}"
        return report
    if not torch.cuda.is_available():
        report["error"] = "torch.cuda.is_available() is False"
        return report
    properties = torch.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    left = torch.arange(1024, dtype=torch.float32, device=f"cuda:{device_index}")
    right = torch.arange(1024, dtype=torch.float32, device=f"cuda:{device_index}")
    product = float((left * right).sum().item())
    torch.cuda.synchronize()
    report.update({
        "ok": True,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "mem_get_info_total_bytes": int(total_bytes),
        "mem_get_info_free_bytes": int(free_bytes),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "kernel_check_sum": product,
    })
    return report


def hf_probe(checkpoint: str | None, allow_hf_download: bool, repo_dir: str | None,
             hf_revision: str | None = None, hf_cache_dir: str | None = None) -> dict:
    """Probe weight availability *without* importing torch.

    Runs one resolution only: ``resolve_weights()`` is memoised inside the
    backend, so the same round does not download twice.
    """
    report = {
        "probe": "hf_weights",
        "repo_id": SAM3_REPO_ID,
        "checkpoint_file": SAM3_CHECKPOINT_FILE,
        "config_file": SAM3_CONFIG_FILE,
        "official_entry": SAM3_HF_URL,
        "hf_revision": hf_revision,
        "hf_revision_pinned": hf_revision is not None,
        "hf_cache_dir": hf_cache_dir,
        "token_present": None,
        "outcome": None,
        "detail": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "config_sha256": None,
        "hf_download_calls": [],
    }
    try:
        from huggingface_hub import get_token  # noqa: PLC0415
    except ImportError as exc:
        report["outcome"] = "BACKEND_UNAVAILABLE"
        report["detail"] = f"huggingface_hub not importable: {exc}"
        return report
    try:
        report["token_present"] = bool(get_token())
    except Exception as exc:  # pragma: no cover
        report["detail"] = f"could not read token state: {type(exc).__name__}"

    backend = Sam3Backend(
        repo_dir=repo_dir or ".",
        checkpoint_path=checkpoint,
        checkpoint_sha256=None,
        device="cuda",
        confidence_threshold=0.5,
        allow_hf_download=allow_hf_download,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
    )
    try:
        backend.resolve_weights()
    except AuthRequired as exc:
        report["outcome"] = "AUTH_REQUIRED"
        report["detail"] = str(exc)
        report["hf_download_calls"] = backend.hf_calls
        return report
    except BackendUnavailable as exc:
        report["outcome"] = "BACKEND_UNAVAILABLE"
        report["detail"] = str(exc)
        return report
    report["outcome"] = "WEIGHTS_RESOLVED"
    report["checkpoint_path"] = str(backend.checkpoint_path)
    report["hf_download_calls"] = backend.hf_calls
    report["weights_binding"] = backend.bindings()["weights"]
    if backend.checkpoint_path and Path(backend.checkpoint_path).is_file():
        report["checkpoint_sha256"] = imagery.sha256_file(backend.checkpoint_path)
        report["checkpoint_bytes"] = Path(backend.checkpoint_path).stat().st_size
    if backend.config_path and Path(backend.config_path).is_file():
        report["config_sha256"] = imagery.sha256_file(backend.config_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAM3 adapter environment probes")
    parser.add_argument("--mode", choices=("env", "cuda", "hf", "all"), default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    args = parser.parse_args(argv)

    payload: dict = {}
    if args.mode in ("env", "all"):
        payload["env"] = env_probe()
    if args.mode in ("cuda", "all"):
        payload["cuda"] = cuda_probe(args.device_index)
    if args.mode in ("hf", "all"):
        payload["hf"] = hf_probe(args.checkpoint, args.allow_hf_download, args.repo_dir,
                                 args.hf_revision, args.hf_cache_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
