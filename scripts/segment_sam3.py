#!/usr/bin/env python
"""Standalone SAM 3 image segmentation entry point (object-mask-v1 contract).

Thread / bytecode policy is applied **explicitly** before any heavy import: the
values are assigned, not defaulted, so an inherited ``OMP_NUM_THREADS=32`` cannot
survive into the run. The script never imports torch at module import time and
never touches the project's NumPy 2 SFM environment: it must be executed with the
isolated interpreter under ``integrations/sam3/env``.

Usage
-----
    python scripts/segment_sam3.py \
        --io-run outputs/delegation/T01A/.../20261006T153857Z_5c3795b195b6 \
        --indices 17,34,59 --object-id skull-01 --text-prompt skull \
        --checkpoint ABS --checkpoint-sha256 <sha256> --output <NEW DIR>

Exit codes follow docs/contracts_object_masks_v1.md:
0 SUCCESS, 1 PARTIAL, 2 INPUT_ERROR, 3 BACKEND_UNAVAILABLE/AUTH_REQUIRED,
4 BACKEND_ERROR, 5 RESOURCE_LIMIT.  Argument problems also exit 2 (INPUT_ERROR).
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------- #
# process policy -- assigned before numpy / torch / PIL are imported
# --------------------------------------------------------------------------- #
CPU_THREADS = "2"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = CPU_THREADS
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "integrations" / "sam3"
DEFAULT_REPO_DIR = ADAPTER_DIR / "src" / "sam3-repo"
FROZEN_HF_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from sam3_adapter import probe as probe_module  # noqa: E402
from sam3_adapter.contract import (  # noqa: E402
    EXIT_CODES,
    SAM3_BACKEND_NAME,
    SAM3_INTERNAL_RESOLUTION,
)
from sam3_adapter.deadline import Budgets  # noqa: E402
from sam3_adapter.resources import (  # noqa: E402
    RAM_COMMIT_LIMIT_BYTES,
    VRAM_LIMIT_BYTES,
    ResourcePolicy,
)
from sam3_adapter.runner import OutputRefused, RunRequest, execute  # noqa: E402
from sam3_adapter.visual_prompt import validate_visual_prompt  # noqa: E402

#: options that exist in the contract discussion but are NOT implemented here
UNIMPLEMENTED_OPTIONS = {
}


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #
def _parse_indices(text: str) -> list[int]:
    indices: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"input_index must be an integer: {chunk!r}") from exc
        if value < 0:
            raise argparse.ArgumentTypeError(f"input_index must be non-negative: {value}")
        if value in indices:
            raise argparse.ArgumentTypeError(f"duplicate input_index: {value}")
        indices.append(value)
    if not indices:
        raise argparse.ArgumentTypeError("--indices must list at least one input_index")
    return indices


def _parse_selection(values: list[str] | None, allowed_indices: list[int]) -> dict[int, str]:
    selection: dict[int, str] = {}
    for raw in values or []:
        if ":" not in raw:
            raise argparse.ArgumentTypeError(f"--select expects INPUT_INDEX:CANDIDATE_ID, got {raw!r}")
        index_text, candidate_id = raw.split(":", 1)
        try:
            index = int(index_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--select index must be an integer: {index_text!r}") from exc
        if index not in allowed_indices:
            raise argparse.ArgumentTypeError(f"--select index {index} is not in --indices")
        if index in selection:
            raise argparse.ArgumentTypeError(f"--select given more than once for input_index {index}")
        if not (len(candidate_id) == 3 and candidate_id[0] == "c" and candidate_id[1:].isdigit()):
            raise argparse.ArgumentTypeError(
                f"candidate id must look like c00/c01, got {candidate_id!r}")
        selection[index] = candidate_id
    return selection


def _confidence_threshold(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--confidence-threshold must be a number: {text!r}") from exc
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("--confidence-threshold must lie strictly inside (0, 1)")
    return value


def _resolution(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--resolution must be an integer: {text!r}") from exc
    if value < 14 or value % 14 != 0:
        raise argparse.ArgumentTypeError("--resolution must be a positive multiple of the ViT patch size 14")
    return value


def _positive_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected seconds, got {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("budgets must be positive")
    return value


def _device_index_of(device: str) -> int:
    """Index used by the probe; an unparsable device is rejected, not defaulted."""
    from sam3_adapter.resources import parse_cuda_device  # noqa: PLC0415
    _, index = parse_cuda_device(device)
    return 0 if index is None else index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM 3 image segmentation -> object-mask-v1 manifest")
    parser.add_argument("--io-run", required=True, help="absolute path of a successful photo-io-v1 run")
    parser.add_argument("--indices", required=True, type=_parse_indices,
                        help="comma separated input_index list")
    parser.add_argument("--object-id", required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--text-prompt")
    prompts.add_argument("--visual-prompt", type=Path,
                         help="JSON file with identity-bound AI-assisted positive pixel boxes; box-only, not truth masks")
    parser.add_argument("--output", required=True, help="output root; a new <root>/<run_id>/ is created")
    parser.add_argument("--backend", choices=(SAM3_BACKEND_NAME, "fake"), default=SAM3_BACKEND_NAME)
    parser.add_argument("--checkpoint", default=None,
                        help="local sam3.pt; REQUIRES --checkpoint-sha256")
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--config", default=None,
                        help="declared config file; recorded as NOT consumed by the image builder; "
                             "REQUIRES --config-sha256")
    parser.add_argument("--config-sha256", default=None)
    parser.add_argument("--hf-revision", default=FROZEN_HF_REVISION,
                        help="pinned revision for HF resolution; never a rolling 'main'")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--fake-spec", default=None)
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--device", default="cuda", help="cpu | cuda | cuda:N")
    parser.add_argument("--confidence-threshold", type=_confidence_threshold, default=0.5)
    parser.add_argument("--resolution", type=_resolution, default=SAM3_INTERNAL_RESOLUTION)
    parser.add_argument("--allow-hf-download", action="store_true",
                        help="resolve the gated facebook/sam3 weights through the existing HF login")
    parser.add_argument("--select", action="append", default=None,
                        help="explicit candidate choice INPUT_INDEX:CANDIDATE_ID (repeatable)")
    parser.add_argument("--in-process", action="store_true",
                        help="run the frames inside this process instead of the bounded worker; "
                             "intended for deterministic interface tests only")
    parser.add_argument("--worker-python", default=None,
                        help="interpreter for the bounded worker (defaults to this interpreter)")
    parser.add_argument("--env-probe", action="store_true",
                        help="run the environment/CUDA probe as an explicit bounded pre-step; the probe is "
                             "NEVER run implicitly, so it can never bypass supervision")
    parser.add_argument("--no-host-probe", action="store_true",
                        help="deprecated no-op kept for call compatibility: the probe is never run "
                             "implicitly, so there is nothing to suppress")
    parser.add_argument("--ram-commit-limit-gib", type=float, default=RAM_COMMIT_LIMIT_BYTES / (1 << 30))
    parser.add_argument("--vram-limit-gib", type=float, default=VRAM_LIMIT_BYTES / (1 << 30))
    parser.add_argument("--load-budget-seconds", type=_positive_seconds, default=300.0)
    parser.add_argument("--frame-budget-seconds", type=_positive_seconds, default=300.0)
    parser.add_argument("--batch-budget-seconds", type=_positive_seconds, default=900.0)
    parser.add_argument("--monitor-interval-seconds", type=_positive_seconds, default=0.25)
    parser.add_argument("--device-observe", choices=("auto", "none"), default="auto",
                        help="continuous device observation: 'auto' (default) wires the existing bounded "
                             "device-identified driver query for a CUDA run and is NOT_APPLICABLE for a CPU "
                             "run; 'none' explicitly disables a required channel and is rejected before "
                             "launch on CUDA")
    parser.add_argument("--device-observe-timeout-seconds", type=_positive_seconds, default=5.0)
    for option in UNIMPLEMENTED_OPTIONS:
        parser.add_argument(option, default=None,
                            help="NOT IMPLEMENTED -- passing it is rejected")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    for option in UNIMPLEMENTED_OPTIONS:
        if option in raw_argv or any(item.startswith(option + "=") for item in raw_argv):
            print(f"REJECTED: {option} is not implemented in this deliverable: "
                  f"{UNIMPLEMENTED_OPTIONS[option]}", file=sys.stderr)
            return EXIT_CODES["INPUT_ERROR"]

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    try:
        explicit_selection = _parse_selection(args.select, args.indices)
        visual = None
        if args.visual_prompt is not None:
            if args.backend != SAM3_BACKEND_NAME:
                raise ValueError("--visual-prompt requires the SAM3 backend")
            if args.visual_prompt.stat().st_size > 1 << 20:
                raise ValueError("visual prompt JSON exceeds 1 MiB")
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError("duplicate visual prompt JSON key: " + key)
                    result[key] = value
                return result
            def invalid(value):
                raise ValueError("nonfinite visual prompt JSON: " + value)
            visual = validate_visual_prompt(json.loads(args.visual_prompt.read_text(encoding="utf-8-sig"),
                object_pairs_hook=pairs, parse_constant=invalid), args.indices, args.text_prompt)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    if args.backend == SAM3_BACKEND_NAME:
        if args.checkpoint and not args.checkpoint_sha256:
            parser.error("--checkpoint requires --checkpoint-sha256 (weights must never be recorded unverified)")
        if args.config and not args.config_sha256:
            parser.error("--config requires --config-sha256")
        if not args.checkpoint and not args.allow_hf_download:
            parser.error("provide --checkpoint/--checkpoint-sha256, or --allow-hf-download for the gated weights")

    argv_effective = [sys.executable, str(Path(__file__).resolve()), *raw_argv]

    # ------------------------------------------------------------------ probe
    # The environment/CUDA probe is NOT run implicitly: an implicit probe would
    # execute outside the controlled path. With --env-probe it runs as an
    # explicit bounded pre-step in its own worker, under the same mechanism.
    probe_dir = Path(args.output) / "_env_probe"
    if args.env_probe:
        from sam3_adapter import worker as worker_module  # noqa: PLC0415
        probe_logs = probe_dir / "logs"
        probe_dir.mkdir(parents=True, exist_ok=True)
        record = worker_module.run_bounded(
            [args.worker_python or sys.executable, "-m", "sam3_adapter.probe", "--mode", "all",
             "--repo-dir", args.repo_dir, "--device-index", str(_device_index_of(args.device))],
            deadline_seconds=60.0, cwd=ADAPTER_DIR, log_dir=probe_logs,
            env=dict(os.environ, PYTHONPATH=str(ADAPTER_DIR),
                     OMP_NUM_THREADS=CPU_THREADS, PYTHONDONTWRITEBYTECODE="1"),
        )
        stdout_path = record.get("stdout_path")
        payload = None
        if stdout_path and Path(stdout_path).is_file():
            try:
                payload = json.loads(Path(stdout_path).read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                payload = {"probe_error": f"probe output was not JSON: {exc}"}
        (probe_dir / "PROBE_RECORD.json").write_text(
            json.dumps({"mode": "explicit_bounded_pre_step", "record": {k: v for k, v in record.items()},
                        "payload": payload}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n")
        host_probe = {"mode": "explicit_bounded_pre_step", "deadline_seconds": 60.0,
                      "timed_out": record.get("timed_out"), "returncode": record.get("returncode"),
                      "record_path": str(probe_dir / "PROBE_RECORD.json"), "payload": payload}
    else:
        host_probe = {
            "status": "NOT_RUN",
            "reason": "the environment/CUDA probe is never run implicitly by the CLI; pass --env-probe "
                      "to run it as an explicit bounded pre-step",
            "implicit_probe_is_bypass": False,
        }

    request = RunRequest(
        io_run=Path(args.io_run),
        indices=list(args.indices),
        object_id=args.object_id,
        text_prompt=args.text_prompt,
        visual_prompt=visual,
        output_root=Path(args.output),
        repo_dir=Path(args.repo_dir),
        backend_kind=args.backend,
        fake_spec=Path(args.fake_spec) if args.fake_spec else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        checkpoint_sha256=args.checkpoint_sha256,
        config_path=Path(args.config) if args.config else None,
        config_sha256=args.config_sha256,
        hf_revision=args.hf_revision,
        hf_cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        resolution=args.resolution,
        allow_hf_download=args.allow_hf_download,
        explicit_selection=explicit_selection,
        argv=argv_effective,
        cwd=str(Path.cwd()),
        env_allowlist=probe_module.env_allowlist()["names_present"],
        host_probe=host_probe,
        resource_policy=ResourcePolicy(
            ram_commit_limit_bytes=int(args.ram_commit_limit_gib * (1 << 30)),
            vram_limit_bytes=int(args.vram_limit_gib * (1 << 30)),
        ),
        budgets=Budgets(
            load_seconds=args.load_budget_seconds,
            frame_seconds=args.frame_budget_seconds,
            batch_seconds=args.batch_budget_seconds,
        ),
        monitor_interval_seconds=args.monitor_interval_seconds,
        device_observe=args.device_observe,
        device_observe_timeout=args.device_observe_timeout_seconds,
        use_worker=not args.in_process,
        worker_python=args.worker_python,
        worker_adapter_dir=ADAPTER_DIR,
    )

    try:
        outcome = execute(request)
    except OutputRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_CODES["INPUT_ERROR"]
    except Exception as exc:  # noqa: BLE001 - unexpected internal failure
        print(f"INTERNAL_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CODES["BACKEND_ERROR"]

    summary = {
        "run_dir": str(outcome.run_dir),
        "status": outcome.status,
        "exit_code": outcome.exit_code,
        "execution_mode": outcome.manifest.get("execution_mode") if outcome.manifest else None,
        "evidence_level": outcome.manifest["evidence_level"] if outcome.manifest else None,
        "counts": outcome.manifest["counts"] if outcome.manifest else None,
        "stop_state": outcome.manifest.get("stop_state") if outcome.manifest else None,
        "worker": {k: v for k, v in (outcome.manifest.get("worker") or {}).items()
                   if k in ("pid", "returncode", "exited", "residual_unknown", "frames_seen",
                            "kill_reason", "reclaim")} if outcome.manifest else None,
        "backend_unavailable_reason": (
            outcome.manifest["backend"].get("backend_unavailable_reason") if outcome.manifest else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
