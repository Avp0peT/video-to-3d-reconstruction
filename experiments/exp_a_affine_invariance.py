"""Experiment A — affine invariance of the internal coordinates (README section 14).

Protocol
--------
A fixed image ``X`` (12 points) and a fixed, well-conditioned affine frame are
chosen.  ``N`` random invertible affine maps ``T(x) = Ax + t`` are drawn with a
*bounded condition number* (see
:meth:`~kozlov_lab.geometry.transforms.AffineTransform.random_invertible`).
For every map the frame is carried along with the image and the internal
coordinates are recomputed; the reported error is

    max_i |alpha_i(T X) - alpha_i(X)|,   max_i |beta_i(T X) - beta_i(X)|.

Acceptance criterion (README section 14): both ``< 1e-9``.

What this script reports beyond the criterion
---------------------------------------------
* the **measured worst case** and the **theoretical bound** ``cond * eps * C``,
  per configuration — a number without its error budget is not a result;
* the **ill-conditioned arm**: the same experiment on frames whose flatness is
  ``1e-2 … 5e-9``.  The README's ``1e-9`` is a statement about *well-conditioned*
  frames; this arm shows where it stops being true and by how much, which is the
  honest form of the claim;
* the **translation arm**: the degradation law
  ``max|delta alpha| ~ eps * |t| / |B - A|``, which the conditioning bound does
  *not* predict because the information is lost in the input, before the solve.

Run::

    python experiments/exp_a_affine_invariance.py            # 1000 maps
    python experiments/exp_a_affine_invariance.py --n 5000   # longer run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab import __version__  # noqa: E402
from kozlov_lab.config import tolerance_for_condition  # noqa: E402
from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.geometry import AffineFrame, AffineTransform  # noqa: E402
from _common import write_outputs  # noqa: E402

EPS = float(np.finfo(np.float64).eps)
REQUIRED = 1e-9
FRAME_INDICES = (0, 3, 5)


def run_arm(
    x: np.ndarray,
    frame: AffineFrame,
    n: int,
    *,
    seed: int,
    max_condition: float = 50.0,
) -> dict:
    """One arm of the experiment: ``n`` random affine maps, fixed frame."""
    rng = np.random.default_rng(seed)
    before = frame.coordinates_batch(x)
    delta_alpha: list[float] = []
    delta_beta: list[float] = []
    conditions: list[float] = []
    for _ in range(n):
        t = AffineTransform.random_invertible(rng, max_condition=max_condition)
        moved_frame = frame.transformed(t)
        after = moved_frame.coordinates_batch(t.apply(x))
        delta_alpha.append(float(np.abs(after["alpha"] - before["alpha"]).max()))
        delta_beta.append(float(np.abs(after["beta"] - before["beta"]).max()))
        conditions.append(max(float(before["condition_number"]), float(moved_frame.condition_number)))

    da = np.array(delta_alpha)
    db = np.array(delta_beta)
    worst = float(max(da.max(), db.max()))
    cond_array = np.array(conditions)
    bound = float(max(tolerance_for_condition(float(cond_array.max())), EPS * 64.0))
    return {
        "n_transforms": n,
        "seed": seed,
        "max_condition_budget": max_condition,
        "condition_number": {
            "min": float(cond_array.min()),
            "median": float(np.median(cond_array)),
            "max": float(cond_array.max()),
        },
        "delta_alpha": {
            "max": float(da.max()),
            "median": float(np.median(da)),
            "p99": float(np.percentile(da, 99)),
        },
        "delta_beta": {
            "max": float(db.max()),
            "median": float(np.median(db)),
            "p99": float(np.percentile(db, 99)),
        },
        "worst_delta": worst,
        "theoretical_bound": bound,
        "margin_to_bound": bound / worst if worst > 0 else None,
        "required": REQUIRED,
        "passed": worst < REQUIRED,
        "non_zero_measurement": worst > 0.0,
    }


def run_translation_arm(x: np.ndarray, frame: AffineFrame, translation: float) -> dict:
    """Translation-only arm: measures the ``eps * |t| / frame_scale`` law."""
    frame_size = float(np.linalg.norm(frame.b - frame.a))
    t = AffineTransform.translation_of(translation, -translation)
    try:
        delta = frame.invariance(t, x)
        measured = delta.max_delta
        refused = False
    except ValueError as exc:
        measured = None
        refused = True
        delta = None

    predicted = EPS * translation / frame_size
    if measured is None:
        ratio = None
    else:
        ratio = measured / predicted if predicted > 0 else None
    return {
        "translation": translation,
        "frame_scale": frame_size,
        "predicted_error": predicted,
        "measured_error": measured,
        "measured_over_predicted": ratio,
        "within_a_factor_of_8": (ratio is not None and ratio <= 8.0),
        "library_refused_degenerate_frame": refused,
        "note": (
            "library raised: the translated image no longer resolves its own structure"
            if refused
            else None
        ),
    }


def run_ill_conditioned_arm(n_per_flatness: int, seed: int) -> list[dict]:
    """The same experiment on deliberately flat frames.

    Isometries are used so that the frame keeps its conditioning exactly; with a
    general affine map the transformed frame can itself become collinear, which
    would confound "the accuracy claim failed" with "the frame was rejected".
    """
    rows = []
    for flatness in (1e-2, 1e-4, 1e-6, 1e-8, 5e-9):
        shape = shapes.near_collinear_triangle(flatness)
        x = shape.as_array()
        frame = AffineFrame.from_indices(x, (0, 1, 2))
        if not frame.is_valid():
            rows.append({"flatness": flatness, "skipped": "below COLLINEAR_TOL"})
            continue

        rng = np.random.default_rng(seed)
        before = frame.coordinates_batch(x)
        worst = 0.0
        for _ in range(n_per_flatness):
            t = AffineTransform.rotation(float(rng.uniform(0.0, 360.0)))
            moved_frame = frame.transformed(t)
            after = moved_frame.coordinates_batch(t.apply(x))
            worst = max(
                worst,
                float(np.abs(after["alpha"] - before["alpha"]).max()),
                float(np.abs(after["beta"] - before["beta"]).max()),
            )
        bound = tolerance_for_condition(frame.condition_number)
        rows.append(
            {
                "flatness": flatness,
                "condition_number": frame.condition_number,
                "max_delta": worst,
                "theoretical_bound": bound,
                "required": REQUIRED,
                "meets_required_1e_9": worst < REQUIRED,
                "within_bound": worst <= bound,
                "n_transforms": n_per_flatness,
            }
        )
    return rows


def markdown(summary: dict) -> str:
    main = summary["experiment_a"]
    lines = [
        f"# Experiment A — affine invariance of internal coordinates",
        "",
        f"* requirement: `max|d alpha|, max|d beta| < {REQUIRED:.0e}` (README section 14)",
        f"* image: `square_3x` (12 points), frame indices {FRAME_INDICES}",
        f"* package version: `{summary['package_version']}`",
        "",
        "## Main arm",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| transforms | {main['n_transforms']} |",
        f"| condition number (min / median / max) | "
        f"{main['condition_number']['min']:.3f} / {main['condition_number']['median']:.3f} / "
        f"{main['condition_number']['max']:.3f} |",
        f"| max abs delta alpha | {main['delta_alpha']['max']:.3e} |",
        f"| max abs delta beta | {main['delta_beta']['max']:.3e} |",
        f"| median abs delta alpha | {main['delta_alpha']['median']:.3e} |",
        f"| theoretical bound cond*eps*64 | {main['theoretical_bound']:.3e} |",
        f"| margin (bound / worst) | {main['margin_to_bound']:.1f}x |",
        f"| **passes requirement** | **{'YES' if main['passed'] else 'NO'}** |",
        f"| measurement is non-trivial (worst > 0) | {main['non_zero_measurement']} |",
        "",
        "## Ill-conditioned arm (isometries only)",
        "",
        "| flatness | cond([B-A,C-A]) | max abs delta | bound cond*eps*64 | meets 1e-9 | within bound |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["ill_conditioned"]:
        if "skipped" in row:
            lines.append(f"| {row['flatness']:.0e} | - | - | - | - | {row['skipped']} |")
            continue
        lines.append(
            f"| {row['flatness']:.0e} | {row['condition_number']:.3e} | {row['max_delta']:.3e} | "
            f"{row['theoretical_bound']:.3e} | {'yes' if row['meets_required_1e_9'] else '**NO**'} | "
            f"{'yes' if row['within_bound'] else '**NO**'} |"
        )
    lines += [
        "",
        "## Translation arm — measured degradation law",
        "",
        "Prediction: `max|d alpha| ~ eps * |t| / |B - A|` (information lost in the input,",
        "not in the solve; the conditioning bound does not see it).",
        "",
        "| translation | frame scale | predicted | measured | measured/predicted |",
        "|---|---|---|---|---|",
    ]
    for row in summary["translation_arm"]:
        measured = (
            f"{row['measured_error']:.3e}"
            if row["measured_error"] is not None
            else f"refused ({row['note']})"
        )
        ratio = f"{row['measured_over_predicted']:.2f}" if row["measured_over_predicted"] else "-"
        lines.append(
            f"| {row['translation']:g} | {row['frame_scale']:.3e} | {row['predicted_error']:.3e} | "
            f"{measured} | {ratio} |"
        )
    lines += [
        "",
        "## Reading of the result",
        "",
        f"* The README acceptance criterion holds with a margin of "
        f"{main['margin_to_bound']:.0f}x on the conditioning bound and "
        f"{REQUIRED / main['worst_delta']:.0e}x on the required tolerance.",
        "* The `1e-9` criterion is a statement about well-conditioned frames: in the",
        "  ill-conditioned arm the *reported bound* already exceeds `1e-9`, so the library",
        "  never claims the criterion there. Every measured value stays within that bound.",
        "* The translation arm shows a second, independent failure mode that the condition",
        "  number cannot express: comparing coordinates across very different coordinate",
        "  magnitudes costs accuracy before the linear algebra starts.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="number of random affine maps")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-condition", type=float, default=50.0)
    parser.add_argument("--ill-n", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="print the raw JSON to stdout")
    args = parser.parse_args(argv)

    x = shapes.square(2.0, n_per_side=3).as_array()
    frame = AffineFrame.from_indices(x, FRAME_INDICES)
    assert frame.is_valid(), "fixture frame must be valid"

    started = time.perf_counter()
    main_arm = run_arm(x, frame, args.n, seed=args.seed, max_condition=args.max_condition)
    ill = run_ill_conditioned_arm(args.ill_n, args.seed + 1)
    translations = [
        run_translation_arm(x, frame, t) for t in (1.0, 1e3, 1e6)
    ]
    elapsed = time.perf_counter() - started

    summary = {
        "experiment": "A",
        "title": "affine invariance of internal affine coordinates",
        "package_version": __version__,
        "elapsed_seconds": elapsed,
        "experiment_a": main_arm,
        "ill_conditioned": ill,
        "translation_arm": translations,
    }
    target = write_outputs("EXP-A", summary, markdown(summary), seed=args.seed, out_dir=args.out)

    print(markdown(summary))
    print(f"[written] {target}")
    if args.json:
        print(json.dumps(summary, indent=2))

    failures = []
    if not main_arm["passed"]:
        failures.append("GATE FAILED: main arm exceeded 1e-9")
    if not main_arm["non_zero_measurement"]:
        failures.append("main arm measured an exact zero everywhere (not measuring)")
    for row in ill:
        if "skipped" not in row and not row["within_bound"]:
            failures.append(f"ill-conditioned arm exceeded its own bound at flatness {row['flatness']:.0e}")
    for row in translations:
        if row["measured_over_predicted"] and not row["within_a_factor_of_8"]:
            failures.append(f"translation law violated at |t|={row['translation']:g}")
    if failures:
        print("\n".join(["FAIL:", *failures]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
