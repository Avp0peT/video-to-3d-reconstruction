"""Sprint 3E — reproduce the Phase 2 acceptance gate as a machine-readable file.

Writes ``experiments/results/phase2_gate.json`` (+ ``.meta.json``) containing:

* the main arm: ``N`` random invertible affine maps, the measured
  ``max|d alpha|`` / ``max|d beta|``, and the per-row theoretical bound;
* the **non-triviality** check (the measurement must not be an exact zero);
* the **falsifiability** check: an ill-conditioned frame under isometries must
  *exceed* the ``1e-9`` threshold, otherwise the threshold has no discriminating
  power;
* the order-independence check (shuffled points, per-index comparison);
* the verdicts.

Run::

    python experiments/exp_phase2_gate.py --n 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.config import MATCH_TOL, tolerance_for_condition  # noqa: E402
from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.geometry import AffineFrame, AffineTransform  # noqa: E402
from _common import print_header, write_json  # noqa: E402

EPS = float(np.finfo(np.float64).eps)
FRAME_INDICES = (0, 3, 5)
DEFAULT_SEED = 20261001


def _measure(
    x: np.ndarray, frame: AffineFrame, n: int, *, seed: int, max_condition: float
) -> dict:
    rng = np.random.default_rng(seed)
    before = frame.coordinates_batch(x)
    errors: list[float] = []
    ratios: list[float] = []
    conds: list[float] = []
    for _ in range(n):
        transform = AffineTransform.random_invertible(rng, max_condition=max_condition)
        moved_frame = frame.transformed(transform)
        after = moved_frame.coordinates_batch(transform.apply(x))
        error = max(
            float(np.abs(after["alpha"] - before["alpha"]).max()),
            float(np.abs(after["beta"] - before["beta"]).max()),
        )
        cond = max(float(before["condition_number"]), float(moved_frame.condition_number))
        bound = max(tolerance_for_condition(cond), EPS * 64.0)
        errors.append(error)
        conds.append(cond)
        ratios.append(error / bound)

    errors_array = np.asarray(errors)
    return {
        "n_transforms": n,
        "seed": seed,
        "max_condition_budget": max_condition,
        "max_delta": float(errors_array.max()),
        "median_delta": float(np.median(errors_array)),
        "p99_delta": float(np.percentile(errors_array, 99)),
        "condition_number_min": float(min(conds)),
        "condition_number_median": float(np.median(conds)),
        "condition_number_max": float(max(conds)),
        "worst_error_over_bound": float(max(ratios)),
    }


def _falsifiability_arm(n: int, *, seed: int, flatness: float) -> dict:
    """Ill-conditioned frame under isometries: must break the 1e-9 threshold."""
    x = shapes.near_collinear_triangle(flatness).as_array()
    frame = AffineFrame.from_indices(x, (0, 1, 2))
    if not frame.is_valid():
        return {"flatness": flatness, "skipped": "below COLLINEAR_TOL"}

    rng = np.random.default_rng(seed)
    before = frame.coordinates_batch(x)
    worst = 0.0
    conds: list[float] = []
    for _ in range(n):
        transform = AffineTransform.rotation(float(rng.uniform(0.0, 360.0)))
        moved_frame = frame.transformed(transform)
        conds.append(float(moved_frame.condition_number))
        after = moved_frame.coordinates_batch(transform.apply(x))
        worst = max(
            worst,
            float(np.abs(after["alpha"] - before["alpha"]).max()),
            float(np.abs(after["beta"] - before["beta"]).max()),
        )
    return {
        "flatness": flatness,
        "condition_number": frame.condition_number,
        "condition_number_after_isometry_max": float(max(conds)),
        "max_delta": worst,
        "reported_bound": tolerance_for_condition(frame.condition_number),
        "exceeds_required_1e_9": worst > MATCH_TOL,
        "within_reported_bound": worst <= tolerance_for_condition(frame.condition_number) * 10.0,
        "n_transforms": n,
    }


def _order_independence_arm(x: np.ndarray, frame: AffineFrame, *, seed: int, trials: int) -> dict:
    rng = np.random.default_rng(seed)
    before = frame.coordinates_batch(x)
    transform = AffineTransform.random_invertible(rng, max_condition=20.0)
    moved_frame = frame.transformed(transform)
    moved = transform.apply(x)

    worst_follows_permutation = 0.0
    worst_invariance = 0.0
    for _ in range(trials):
        perm = rng.permutation(x.shape[0])
        shuffled = frame.coordinates_batch(x[perm])
        worst_follows_permutation = max(
            worst_follows_permutation,
            float(np.abs(shuffled["alpha"] - before["alpha"][perm]).max()),
            float(np.abs(shuffled["beta"] - before["beta"][perm]).max()),
        )
        after = moved_frame.coordinates_batch(moved[perm])
        worst_invariance = max(
            worst_invariance,
            float(np.abs(after["alpha"] - before["alpha"][perm]).max()),
            float(np.abs(after["beta"] - before["beta"][perm]).max()),
        )
    return {
        "trials": trials,
        "max_error_following_the_permutation": worst_follows_permutation,
        "max_error_of_the_invariance": worst_invariance,
        "permutation_is_exact": worst_follows_permutation <= 1e-15,
        "invariance_still_holds": worst_invariance < MATCH_TOL,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-condition", type=float, default=50.0)
    parser.add_argument("--falsifiability-n", type=int, default=500)
    parser.add_argument("--order-trials", type=int, default=25)
    args = parser.parse_args(argv)

    x = shapes.square(2.0, n_per_side=3).as_array()
    frame = AffineFrame.from_indices(x, FRAME_INDICES)
    assert frame.is_valid()

    config = {
        "image": "square_3x",
        "frame_indices": list(FRAME_INDICES),
        "n_transforms": args.n,
        "max_condition_budget": args.max_condition,
        "required_accuracy": MATCH_TOL,
        "falsifiability_n": args.falsifiability_n,
        "order_trials": args.order_trials,
    }

    print_header("Phase 2 gate — Experiment A (internal affine coordinates)")
    main_arm = _measure(x, frame, args.n, seed=args.seed, max_condition=args.max_condition)
    print(
        f"  main arm: n={main_arm['n_transforms']}  max|d|={main_arm['max_delta']:.3e}  "
        f"median={main_arm['median_delta']:.3e}  cond_max={main_arm['condition_number_max']:.3f}"
    )

    falsifiability = [
        _falsifiability_arm(args.falsifiability_n, seed=args.seed + 1, flatness=f)
        for f in (1e-6, 1e-8, 5e-9)
    ]
    for row in falsifiability:
        if "skipped" in row:
            print(f"  falsifiability flatness={row['flatness']:.0e}: skipped")
        else:
            print(
                f"  falsifiability flatness={row['flatness']:.0e}: cond={row['condition_number']:.3e} "
                f"max|d|={row['max_delta']:.3e}  exceeds_1e-9={row['exceeds_required_1e_9']}"
            )

    order = _order_independence_arm(x, frame, seed=args.seed + 2, trials=args.order_trials)
    print(
        f"  order independence: perm_exact={order['permutation_is_exact']} "
        f"invariance_holds={order['invariance_still_holds']}"
    )

    passed = (
        main_arm["max_delta"] < MATCH_TOL
        and main_arm["max_delta"] > 0.0
        and any(row.get("exceeds_required_1e_9") for row in falsifiability)
        and all(row.get("within_reported_bound", True) for row in falsifiability)
        and order["permutation_is_exact"]
        and order["invariance_still_holds"]
    )
    verdicts = {
        "gate_passed": bool(passed),
        "criterion": {"max_delta_alpha_beta": f"< {MATCH_TOL:.0e}"},
        "non_trivial_measurement": main_arm["max_delta"] > 0.0,
        "threshold_is_falsifiable": any(row.get("exceeds_required_1e_9") for row in falsifiability),
        "ill_conditioned_arm_within_its_own_bound": all(
            row.get("within_reported_bound", True) for row in falsifiability
        ),
        "order_independence_verified": order["permutation_is_exact"] and order["invariance_still_holds"],
        "note": (
            "The 1e-9 criterion is asserted for well-conditioned frames only; the "
            "ill-conditioned arm records where it stops holding and the library's own "
            "reported bound for those configurations."
        ),
    }

    target, meta = write_json(
        "phase2_gate",
        {
            "experiment": "phase2_gate",
            "title": "Phase 2 acceptance gate (Experiment A)",
            "main_arm": main_arm,
            "falsifiability_arm": falsifiability,
            "order_independence_arm": order,
            "verdicts": verdicts,
        },
        seed=args.seed,
        config=config,
    )
    print(f"\n  gate_passed = {passed}")
    print(f"  [written] {target}")
    print(f"  [written] {meta}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
