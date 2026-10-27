"""Command line entry point (``photo-io-v1`` §5).

    python -m kozlov_lab.io --manifest <manifest.json> \\
        --output outputs/delegation/T01A/runs --data-kind synthetic|real

Exit codes are part of the contract and are **not** to be adjusted by callers:

* ``0`` — every input row succeeded and all artifacts were written;
* ``1`` — the run completed but at least one row is an error;
* ``2`` — a top-level problem: bad arguments, an unusable output directory, a
  manifest that cannot be parsed, or an output root outside ``outputs/``.

The last case deliberately writes **nothing**: claiming evidence was saved when
the destination is unwritable would be worse than failing loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .hashing import CONTRACT_VERSION
from .manifest import DATA_KINDS, load_manifest
from .paths import find_project_root
from .runner import execute_run

__all__ = ["CliError", "build_parser", "main"]

#: ``--output`` must live strictly inside this project-relative directory.
_OUTPUT_PARENT = "outputs"


class CliError(Exception):
    """A top-level problem that must exit ``2`` without writing artifacts."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kozlov_lab.io",
        description=f"Photo + manual-mask IO for contract {CONTRACT_VERSION}.",
    )
    parser.add_argument("--manifest", required=True, help="path to the input manifest (UTF-8 JSON)")
    parser.add_argument(
        "--output",
        required=True,
        help=f"artifact base directory; must be inside the project '{_OUTPUT_PARENT}/'",
    )
    parser.add_argument(
        "--data-kind",
        required=True,
        choices=DATA_KINDS,
        help="must match the manifest's data_kind",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # argparse exits 2 on missing/invalid arguments

    try:
        project_root = _project_root()
        output_base = _resolve_output_base(args.output, project_root)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest)
    # ``load_manifest`` never raises for a bad manifest: a run record still has
    # to be written, so the failure is carried as data rather than an exception.
    manifest_load = load_manifest(manifest_path, args.data_kind)

    if manifest_load.ok:
        assert manifest_load.manifest is not None
        dataset_root = (project_root / manifest_load.manifest.dataset_root).resolve()
        if output_base == dataset_root or dataset_root in output_base.parents:
            print(
                "error: --output must not be inside the dataset root "
                f"({manifest_load.manifest.dataset_root!r})",
                file=sys.stderr,
            )
            return 2

    try:
        result = execute_run(
            manifest_load,
            output_base=output_base,
            data_kind=args.data_kind,
            project_root=project_root,
        )
    except OSError as exc:
        print(f"error: cannot create run directory under {output_base}: {exc}", file=sys.stderr)
        return 2

    _report(result)
    return result.exit_code


def _project_root() -> Path:
    root = find_project_root(Path.cwd())
    if not (root / "pyproject.toml").is_file():
        raise CliError("run this command from the project root (no pyproject.toml found above the cwd)")
    return root


def _resolve_output_base(raw: str, project_root: Path) -> Path:
    """Absolute, ``outputs/``-contained artifact base, or raise :class:`CliError`.

    Requiring the base to be *strictly* inside ``outputs/`` keeps stray run
    directories from accumulating directly in the project's output root.
    """
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()

    outputs_dir = (project_root / _OUTPUT_PARENT).resolve()
    if outputs_dir not in resolved.parents:
        raise CliError(f"--output must lie strictly inside {outputs_dir}")
    return resolved


def _report(result) -> None:
    """One short summary line; the detailed evidence lives in the run directory."""
    print(
        f"run {result.run_id}: status={result.status} "
        f"inputs={result.input_count} ok={result.ok_count} "
        f"errors={result.error_count} warnings={result.warning_count}"
    )
    print(f"artifacts: {result.run_dir}")
