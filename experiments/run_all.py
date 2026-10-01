"""Sprint 3E — run every reproducibility experiment and verify the outputs.

Runs the four experiment scripts in separate processes (so no in-process state can
leak between them), then checks that every expected result file exists, is
non-empty, and records its SHA-256 together with the git commit in
``experiments/results/MANIFEST.json``.

The manifest is what makes the directory quotable: a report can name a file *and*
its digest, and a reader can verify that the file they have is the file that was
produced.

Run::

    python experiments/run_all.py
    python experiments/run_all.py --quick        # smaller sample sizes
    python experiments/run_all.py --verify-only  # reuse artifacts, rewrite the manifest only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _common import RESULTS_ROOT, provenance, print_header  # noqa: E402

HERE = Path(__file__).resolve().parent

SCRIPTS: tuple[tuple[str, str], ...] = (
    ("phase2_gate", "exp_phase2_gate.py"),
    ("invariant_matrix", "exp_invariant_matrix.py"),
    ("precision_stress", "exp_precision_stress.py"),
    ("shape_generalization", "exp_shape_generalization.py"),
    ("official_code_definition", "exp_official_code_definition.py"),
    ("phase4_scaling", "exp_phase4_scaling.py"),
    ("phase4_differential", "exp_phase4_differential.py"),
    ("phase5_compression", "exp_phase5_compression.py"),
    ("phase5_sanity", "exp_phase5_sanity.py"),
    ("phase6_stereo", "exp_phase6_stereo.py"),
)

#: Required result files (Sprint 3E names them explicitly).
REQUIRED = (
    "phase2_gate.json",
    "invariant_matrix.csv",
    "precision_stress.csv",
    "shape_generalization.csv",
    # Q1 evidence: verifies the *definition* of the formal code.  Deliberately a
    # result file rather than a module — Phase 3 is gated on G3.1.
    "official_code_definition.json",
    # phase 4: the O(n^6) structure and the differential gate
    "phase4_scaling.csv",
    "phase4_differential.json",
    # phase 5: equivariance and recognition preservation under compression
    "phase5_equivariance.csv",
    "phase5_recognition.csv",
    "phase5_hand_trace.json",
    # phase 6: the sanity rule, the optimum-set composition, the degeneracy taxonomy
    "phase6_stereo.csv",
)

QUICK_FLAGS: dict[str, tuple[str, ...]] = {
    "exp_phase2_gate.py": ("--n", "200", "--falsifiability-n", "100"),
    "exp_shape_generalization.py": ("--repetitions", "3"),
    "exp_phase4_scaling.py": ("--max-n", "12", "--entry-cap", "60000"),
    "exp_phase4_differential.py": ("--accepted", "600"),
    "exp_phase5_compression.py": ("--quick",),
    "exp_phase6_stereo.py": ("--quick",),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller sample sizes")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "do not run the experiment scripts; only re-verify the artifacts already on disk "
            "and rewrite MANIFEST.json.  Needed where a wall-clock cap kills the process: the "
            "full suite takes ~15 minutes end to end (phase5_compression alone is 8.5), so the "
            "scripts can be run one at a time and the manifest built afterwards.  Because this "
            "mode does not re-run anything, MANIFEST.json records it: a digest written by "
            "--verify-only attests to what is on disk, not to a fresh execution."
        ),
    )
    args = parser.parse_args(argv)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    if args.verify_only:
        print_header("skipping execution (--verify-only): verifying artifacts on disk")
    for name, script in (() if args.verify_only else SCRIPTS):
        extra = QUICK_FLAGS.get(script, ()) if args.quick else ()
        command = [sys.executable, str(HERE / script), *extra]
        print_header(f"running {script}")
        result = subprocess.run(command, cwd=HERE.parent, capture_output=True, text=True)
        tail = result.stdout.strip().splitlines()[-4:]
        for line in tail:
            print(f"  | {line}")
        if result.returncode != 0:
            failures.append(f"{script} exited with {result.returncode}")
            print(result.stderr.strip()[-2000:], file=sys.stderr)

    print_header("verification")
    manifest: dict[str, object] = {
        "provenance": provenance(config={"quick": args.quick, "verify_only": args.verify_only}),
        "files": {},
        "required": list(REQUIRED),
        "missing": [],
        "empty": [],
    }
    for required in REQUIRED:
        path = RESULTS_ROOT / required
        if not path.exists():
            manifest["missing"].append(required)
            failures.append(f"missing result file: {required}")
            print(f"  MISSING {required}")
            continue
        size = path.stat().st_size
        if size == 0:
            manifest["empty"].append(required)
            failures.append(f"empty result file: {required}")
        rows = path.read_text(encoding="utf-8").count("\n")
        manifest["files"][required] = {
            "bytes": size,
            "lines": rows,
            "sha256": _digest(path),
        }
        print(f"  {required:28s} {size:>9,d} bytes  {rows:>5d} lines  {_digest(path)[:16]}")

    sidecars = sorted(RESULTS_ROOT.glob("*.meta.json"))
    manifest["sidecars"] = [p.name for p in sidecars]
    print(f"  provenance sidecars: {len(sidecars)}")

    manifest["failures"] = failures
    manifest_path = RESULTS_ROOT / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"  [written] {manifest_path}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll reproducibility experiments completed and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
