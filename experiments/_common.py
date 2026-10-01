"""Shared helpers for the experiment scripts.

Every experiment writes a *self-describing* result directory::

    results/<EXP-ID>_<YYYY-MM-DD>/
        summary.json      machine-readable results
        summary.md        the same thing as a readable table
        provenance.json   git commit, dirty flag, package version, tolerances, seed

The provenance stamp exists so that a number quoted in the report can always be
traced back to the exact code and tolerances that produced it.  Without it a
result table is unverifiable two weeks later.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def result_dir(exp_id: str, *, date: str | None = None) -> Path:
    """Create and return ``results/<exp_id>_<date>/``."""
    day = date or _dt.date.today().isoformat()
    path = RESULTS_ROOT / f"{exp_id}_{day}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git available
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(seed: int | None = None) -> dict[str, Any]:
    """Everything needed to reproduce the run.

    ``dirty`` is reported rather than hidden: a result produced from a modified
    working tree must not be quoted as if it came from a tagged commit.
    """
    import numpy as np

    from kozlov_lab import __version__, config

    status = _git("status", "--porcelain")
    return {
        "experiment_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "tolerances": {
            "FLOAT_TOL": config.FLOAT_TOL,
            "COLLINEAR_TOL": config.COLLINEAR_TOL,
            "MATCH_TOL": config.MATCH_TOL,
            "SIGNATURE_DECIMALS": config.SIGNATURE_DECIMALS,
            "MAX_CONDITION_NUMBER": config.MAX_CONDITION_NUMBER,
        },
        "seed": seed,
    }


def write_outputs(
    exp_id: str,
    summary: dict[str, Any],
    markdown: str,
    *,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write ``summary.json``, ``summary.md`` and ``provenance.json``."""
    target = out_dir or result_dir(exp_id)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (target / "summary.md").write_text(markdown, encoding="utf-8", newline="\n")
    (target / "provenance.json").write_text(
        json.dumps(provenance(seed), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target
