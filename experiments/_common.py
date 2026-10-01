"""Shared helpers for the experiment scripts.

Every result file is written with a **sidecar provenance record**::

    experiments/results/
        phase2_gate.json          phase2_gate.meta.json
        invariant_matrix.csv      invariant_matrix.meta.json
        precision_stress.csv      precision_stress.meta.json
        shape_generalization.csv  shape_generalization.meta.json

The sidecar carries ``seed``, ``timestamp``, ``git_commit``, ``python_version``,
``numpy_version`` and ``experiment_config``, so any number quoted in a report can
be traced back to the exact code, tolerances and parameters that produced it.

Why a sidecar rather than repeated columns: a CSV with the provenance in every row
is unreadable, and a JSON blob inside a CSV breaks naive consumers.  One file per
concern keeps both machine-readable.

``git_dirty`` is reported rather than hidden — a result produced from a modified
working tree must not be quoted as if it came from a tagged commit.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git unavailable
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(
    *,
    seed: int | Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce the run."""
    import numpy as np

    from kozlov_lab import __version__
    from kozlov_lab import config as tolerance_config

    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "seed": list(seed) if isinstance(seed, (list, tuple)) else seed,
        "tolerances": {
            "FLOAT_TOL": tolerance_config.FLOAT_TOL,
            "COLLINEAR_TOL": tolerance_config.COLLINEAR_TOL,
            "MATCH_TOL": tolerance_config.MATCH_TOL,
            "SIGNATURE_DECIMALS": tolerance_config.SIGNATURE_DECIMALS,
            "MAX_CONDITION_NUMBER": tolerance_config.MAX_CONDITION_NUMBER,
            "UNUSABLE_CONDITION_NUMBER": tolerance_config.UNUSABLE_CONDITION_NUMBER,
            "UNUSABLE_ACCURACY": tolerance_config.UNUSABLE_ACCURACY,
            "TEACHING_CODE_DECIMALS": tolerance_config.TEACHING_CODE_DECIMALS,
        },
        "experiment_config": dict(config) if config else {},
    }


def _write_sidecar(name: str, *, seed: Any, config: Mapping[str, Any] | None) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    meta = RESULTS_ROOT / f"{name}.meta.json"
    meta.write_text(
        json.dumps(
            provenance(seed=seed, config=config), indent=2, ensure_ascii=False, default=str
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return meta


def write_json(
    name: str,
    payload: Mapping[str, Any],
    *,
    seed: int | Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``<name>.json`` plus its provenance sidecar."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    target = RESULTS_ROOT / f"{name}.json"
    body = {"name": name, "provenance": provenance(seed=seed, config=config), **dict(payload)}
    target.write_text(
        json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target, _write_sidecar(name, seed=seed, config=config)


def write_csv(
    name: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    seed: int | Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``<name>.csv`` plus its provenance sidecar.

    The column order comes from ``fieldnames`` when given, so the file does not
    depend on dict ordering at the call site.

    Refuses to write an empty file: an empty CSV looks like "no problems found"
    and is the classic way a broken experiment gets reported as a clean one.
    """
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"write_csv({name!r}): refusing to write an empty result file")
    names = list(fieldnames) if fieldnames else list(rows[0])
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    target = RESULTS_ROOT / f"{name}.csv"
    # ``newline=""`` is required by the csv module; ``lineterminator="\n"`` is what
    # actually keeps the file LF — the csv default is ``"\r\n"`` on every platform,
    # which silently produced CRLF files in a repository that declares ``eol=lf``.
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=names, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in names})
    return target, _write_sidecar(name, seed=seed, config=config)


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# curated per-run output (used by the older report-style experiment scripts)
# --------------------------------------------------------------------------- #
def result_dir(exp_id: str, *, date: str | None = None) -> Path:
    """Create and return ``experiments/results/<exp_id>_<date>/``."""
    day = date or _dt.date.today().isoformat()
    path = RESULTS_ROOT / f"{exp_id}_{day}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_outputs(
    exp_id: str,
    summary: Mapping[str, Any],
    markdown: str,
    *,
    seed: int | Sequence[int] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write ``summary.json``, ``summary.md`` and ``provenance.json`` for a curated run."""
    target = out_dir or result_dir(exp_id)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(
        json.dumps(dict(summary), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "summary.md").write_text(markdown, encoding="utf-8", newline="\n")
    (target / "provenance.json").write_text(
        json.dumps(provenance(seed=seed), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


# --------------------------------------------------------------------------- #
# exact-compatible corpora
# --------------------------------------------------------------------------- #
def quantized_rational_image(points, *, decimals: int = 3):
    """Coordinates as **exact rationals** at the given decimal precision.

    Why this exists: the project's synthetic images are built in floating point
    (``A + 0.6*(B-A)``), so their stored coordinates are not integers and the *exact* recogniser
    refuses them.  The first Phase 5 run fed those floats straight into
    ``recognize_affine_exact``, got ``INVALID_INPUT`` for nearly every case, and the harness
    reported the refusals as "code undefined after compression" — producing a "0 / 768 preserved"
    headline that was entirely an artefact.

    Two honest ways to fix it, and the choice matters:

    * ``Fraction(float(v))`` is exact but imports binary-rounding noise into a corpus that is
      supposed to be noise-free (``0.1`` becomes a 55-bit rational);
    * reading the value **as the decimal it was written as** matches how the images were
      designed, and gives denominators of ``10**decimals`` — small, exact, and free of noise.

    This function does the second, and the rounding is explicit in its name so no caller can
    think they are getting the raw float.
    """
    from fractions import Fraction

    scale = 10**decimals
    return [
        (Fraction(round(float(x) * scale), scale), Fraction(round(float(y) * scale), scale))
        for x, y in points
    ]


def apply_rational_affine(matrix, translation, points):
    """Apply an affine map with **exact rational** arithmetic (integer matrices welcome)."""
    a, b, c, d = matrix
    tx, ty = translation
    return [
        (a * x + b * y + tx, c * x + d * y + ty)
        for x, y in points
    ]
