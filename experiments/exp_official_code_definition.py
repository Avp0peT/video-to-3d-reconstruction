"""Q1 — numerical verification of the **definition** in ``docs/specs/official_image_code.md``.

What this script is, and what it is not
---------------------------------------
It is a verification that the formal definition of Kozlov's image code

    <M_A, T_A>,   T_A = { S_mnu / S_ksp }  over all pairs of triples

is self-consistent, hand-checkable and affine invariant.  It exists to satisfy
gates **G3.6** (two worked examples) and **G3.7** (the definition can be written as
a testable software spec).

It is **not** an implementation of Phase 3.  Phase 3 is gated on G3.1, whose
verdict is a decision for the project owner (see the gate table in the spec), and
``src/kozlov_lab/coding/`` therefore stays empty.  The construction below lives in
``experiments/`` precisely because it is evidence, not a deliverable module.

Run::

    python experiments/exp_official_code_definition.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.config import COLLINEAR_TOL  # noqa: E402
from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.datasets.corpus import DATASETS, dataset_names  # noqa: E402
from kozlov_lab.geometry import (  # noqa: E402
    AffineTransform,
    signed_double_area,
    triangle_flatness,
)
from kozlov_lab.teaching import (  # noqa: E402
    TeachingCodingParameters,
    build_teaching_image_code,
)
from _common import print_header, write_json  # noqa: E402

DISCLAIMER = (
    "This script verifies the *definition* from a primary source. It is not an "
    "implementation of Kozlov's formal image code: Phase 3 remains gated on the G3.1 "
    "source-tier decision recorded in docs/specs/official_image_code.md."
)


# --------------------------------------------------------------------------- #
# the definition, transcribed literally from the specification
# --------------------------------------------------------------------------- #
def triple_areas(points: np.ndarray) -> dict[tuple[int, int, int], float]:
    """``S_mnu`` for every index triple, including the degenerate ones.

    Uses the library's :func:`~kozlov_lab.geometry.signed_double_area` rather than a
    local determinant (the project already owns and tests that formula).

    Degenerate triples are **kept**: they legitimately appear as numerators
    (giving ``rho = 0``).  Whether they may appear as *denominators* is decided
    separately, by a dimensionless test — see :func:`admissible_denominators`.
    """
    return {
        triple: abs(signed_double_area(points[triple[0]], points[triple[1]], points[triple[2]]))
        / 2.0
        for triple in itertools.combinations(range(points.shape[0]), 3)
    }


def admissible_denominators(
    points: np.ndarray, *, tol: float = COLLINEAR_TOL
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Split the triples into admissible and degenerate denominators.

    The source's condition is ``S_ksp != 0``.  Taken literally in floating point that
    test is useless: a triple that is *mathematically* collinear generally has a
    computed determinant of ``~eps * scale^2`` rather than exactly zero, and dividing
    by it produces ``rho`` values of ``~1e16`` that are pure noise.

    Measured on dataset B (a trapezoid whose bases carry sampled points): the
    smallest non-zero area is ``1.1e-16`` while the largest is ``3.4``, so the
    literal reading gives ``max rho = 3.1e16`` — a number with no mathematical
    content, and one that changes under a pure translation.

    The fix is the same lesson as Phase 2's collinearity test: the degeneracy
    criterion must be **dimensionless**.  ``triangle_flatness = 2*Area/max_side^2``
    is used for the decision (so it is scale-invariant), and the discarded triples
    are returned so the population stays visible.
    """
    admissible: list[tuple[int, int, int]] = []
    degenerate: list[tuple[int, int, int]] = []
    for triple in itertools.combinations(range(points.shape[0]), 3):
        i, j, k = triple
        if triangle_flatness(points[i], points[j], points[k]) > tol:
            admissible.append(triple)
        else:
            degenerate.append(triple)
    return admissible, degenerate


def image_code_values(
    points: np.ndarray, *, tol: float = COLLINEAR_TOL
) -> set[float]:
    """``T_A`` as the *set of numbers* described by the primary source.

    The definition says «все числа ρ ... составляют множество Т_А» and
    «множество Т_А — это только числа вида ρ_mnu,ksp» — i.e. a set of numbers.
    Returning a *set* rather than a multiset is what the source's wording says; the
    indexed layer is a separate, unresolved ambiguity (A9).

    Numerators range over **all** triples (including degenerate ones, giving
    ``rho = 0``); denominators are restricted to the admissible triples.
    """
    areas = triple_areas(points)
    admissible, _ = admissible_denominators(points, tol=tol)
    if not admissible:
        raise ValueError(
            "every triple is degenerate: T_A is undefined (no admissible denominator)"
        )
    denominators = [areas[triple] for triple in admissible]
    values: set[float] = set()
    for numerator in areas.values():
        for denominator in denominators:
            values.add(numerator / denominator)
    return values


def unfiltered_rho_scale(points: np.ndarray) -> dict:
    """The magnitude of ``T_A`` **without** the dimensionless degeneracy filter.

    Kept as an explicit diagnostic: it is the measurement that shows *why* the
    literal ``S != 0`` reading is unusable, so the finding cannot be lost by a later
    refactor.
    """
    areas = triple_areas(points)
    non_zero = [value for value in areas.values() if value != 0.0]
    if not non_zero:
        return {"min_non_zero_area": 0.0, "max_area": 0.0, "max_rho": float("inf")}
    return {
        "min_non_zero_area": min(non_zero),
        "max_area": max(areas.values()),
        "max_rho": max(areas.values()) / min(non_zero),
    }


def shortened_code_values(points: np.ndarray, *, tol: float = COLLINEAR_TOL) -> set[float]:
    """The "shortened code": only triples that differ in exactly one index.

    The source explicitly allows this reduction and states it still represents the
    image up to affine transformations.  It is what makes ``n > 20`` tractable, so
    the claim is worth verifying rather than assuming.

    The denominator filter is the **same dimensionless test** as for the full code —
    using the literal ``S != 0`` here reproduced the same noise-dominated ``rho``
    values and made this variant fail invariance on dataset B (measured deviation
    ``0.25``) while the full code passed.
    """
    n = points.shape[0]
    areas = triple_areas(points)
    admissible, _ = admissible_denominators(points, tol=tol)
    admissible_set = set(admissible)
    values: set[float] = set()
    for quad in itertools.combinations(range(n), 4):
        for dropped in quad:
            numerator = tuple(i for i in quad if i != dropped)
            for dropped2 in quad:
                if dropped2 == dropped:
                    continue
                denominator = tuple(i for i in quad if i != dropped2)
                if denominator not in admissible_set:
                    continue
                values.add(areas[numerator] / areas[denominator])
    return values


def compare_value_sets(left: set[float], right: set[float]) -> dict:
    """Compare two ``T`` value sets, with the tolerance taken **relative to the set scale**.

    Why not relative to each value
    ------------------------------
    ``T_A`` contains ``rho = 0`` for every degenerate (collinear) triple, and a
    floating-point affine map turns "exactly collinear" into "collinear up to
    rounding": the transformed ``rho`` becomes ``~1e-283`` instead of ``0``.  A metric
    that divides the deviation by the *value* then reports a relative error of
    ``1e283`` for a difference whose absolute size is ``1e-283`` — a pure artefact of
    the denominator, not a failure of the code.

    The same hazard applies to any ``rho`` whose numerator triangle is nearly
    degenerate, so the tolerance must be scaled by the largest value in either set.

    The near-zero population is **reported** rather than filtered away, so the
    phenomenon stays visible in the results file instead of being hidden by a
    forgiving metric.

    Returns a dict with the two deviations, the scale, and the near-zero count.
    """
    if not left or not right:
        return {
            "max_abs_difference": float("inf"),
            "max_difference_over_scale": float("inf"),
            "scale": 0.0,
            "n_near_zero": 0,
        }

    # Sorted-array nearest-neighbour search.  A pairwise loop is O(|T|^2), and |T|
    # reaches ~5e4 values for a 12-point figure (220 triples paired with 220
    # denominators), i.e. 2.3e9 comparisons per family per dataset -- which is what
    # made the first version of this script appear to hang.
    right_sorted = np.sort(np.fromiter(right, dtype=float, count=len(right)))
    scale = max(
        float(max(abs(v) for v in left)),
        float(max(abs(v) for v in right)),
        1e-300,
    )

    def _directed(source: set[float]) -> tuple[float, float]:
        values = np.fromiter(source, dtype=float, count=len(source))
        positions = np.clip(
            np.searchsorted(right_sorted, values), 1, len(right_sorted) - 1
        )
        nearest = np.minimum(
            np.abs(values - right_sorted[positions - 1]),
            np.abs(values - right_sorted[positions]),
        )
        return float(nearest.max()), float(nearest.max() / scale)

    left_abs, left_rel = _directed(left)
    right_abs, right_rel = _directed(right)
    return {
        "max_abs_difference": max(left_abs, right_abs),
        "max_difference_over_scale": max(left_rel, right_rel),
        "scale": scale,
        "n_near_zero": sum(1 for value in left if abs(value) < 1e-100),
    }


def relative_set_difference(left: set[float], right: set[float]) -> float:
    """Convenience wrapper: the scale-relative deviation used for the pass/fail test."""
    return compare_value_sets(left, right)["max_difference_over_scale"]


# --------------------------------------------------------------------------- #
# the two worked examples required by G3.6
# --------------------------------------------------------------------------- #
def example_1() -> dict:
    """Hand-worked example: a triangle with an interior point, one triple degenerate.

    ``A = {(0,0), (2,0), (0,2), (1,1)}`` — the four triples have areas
    ``2, 1, 1, 0``: the last one is ``(a2, a3, a4) = (2,0), (0,2), (1,1)``, which is
    collinear, so ``S = 0``.

    Hand computation of ``T_A`` (denominators must be non-zero):

        numerators {2, 1, 1, 0} x denominators {2, 1} = {1, 2, 1/2, 1, 0, 0}
        => T_A = {0, 1/2, 1, 2}

    The script recomputes this and asserts the hand result, so the example is a
    check on the code rather than a restatement of it.
    """
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    areas = triple_areas(points)
    values = image_code_values(points)
    expected = {0.0, 0.5, 1.0, 2.0}

    assert set(areas.values()) == {0.0, 1.0, 2.0}, areas
    assert values == expected, (sorted(values), sorted(expected))

    # the degenerate triple really is degenerate, and it really is the one at index 3
    degenerate = [triple for triple, area in areas.items() if area == 0.0]
    assert degenerate == [(1, 2, 3)], degenerate

    # the invariance step, also by hand: any invertible affine map scales every area
    # by |det L|, so every ratio is unchanged.  L = [[2, 1], [0, 3]], det L = 6.
    transform = AffineTransform(np.array([[2.0, 1.0], [0.0, 3.0]]), np.array([5.0, -7.0]))
    moved = transform.apply(points)
    moved_values = image_code_values(moved)
    assert relative_set_difference(values, moved_values) < 1e-12, (
        f"hand example not invariant: {sorted(values)} vs {sorted(moved_values)}"
    )
    moved_areas = triple_areas(moved)
    scale = 6.0
    for triple, area in areas.items():
        assert abs(moved_areas[triple] - scale * area) <= 1e-12 * max(1.0, scale * area), triple

    return {
        "points": points.tolist(),
        "areas": {str(key): value for key, value in areas.items()},
        "hand_expected_T": sorted(expected),
        "computed_T": sorted(values),
        "degenerate_triples": [list(t) for t in degenerate],
        "det_L": transform.det,
        "area_scale_factor": scale,
        "max_relative_code_difference_after_transform": relative_set_difference(
            values, moved_values
        ),
        "shortened_code_T": sorted(shortened_code_values(points)),
    }


def example_2() -> dict:
    """Second worked example: a 5-point figure, plus the shortened-code claim.

    ``B = {(0,0), (3,0), (0,4), (1,1), (5,2)}`` with 10 triples.  Areas are small
    integers (the fourth point sits at distance 1 from the hypotenuse of the
    3-4-5 triangle's right angle), so the value set is reproducible by hand and the
    *shortened* code can be compared against the full one.
    """
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [1.0, 1.0], [5.0, 2.0]])
    areas = triple_areas(points)
    values = image_code_values(points)
    shortened = shortened_code_values(points)

    # invariance under three different families, including a shear with det != 1
    families = {
        "translation": AffineTransform.translation_of(11.5, -3.25),
        "similarity": AffineTransform.similarity(2.5, 41.0),
        "shear": AffineTransform.shear(0.8, -0.2),
        "affine_general": AffineTransform(np.array([[1.3, 0.45], [-0.2, 0.9]]), np.array([-1.0, 2.0])),
    }
    report = {}
    for name, transform in families.items():
        moved_values = image_code_values(transform.apply(points))
        difference = relative_set_difference(values, moved_values)
        report[name] = {
            "det": transform.det,
            "max_relative_difference": difference,
            "invariant_within_1e-12": difference < 1e-12,
        }
        assert difference < 1e-12, (name, difference)

    # relabelling invariance: the value set is an enumeration over all triples
    rng = np.random.default_rng(2026)
    permutation = rng.permutation(points.shape[0])
    permuted_values = image_code_values(points[permutation])
    assert relative_set_difference(values, permuted_values) < 1e-12

    # the shortened code must be a subset of the full one (it uses fewer triple pairs)
    assert shortened <= values, sorted(shortened - values)

    return {
        "points": points.tolist(),
        "n_triples": len(areas),
        "areas": {str(key): value for key, value in areas.items()},
        "T_size": len(values),
        "T": sorted(values),
        "shortened_T_size": len(shortened),
        "shortened_is_subset": True,
        "families": report,
        "relabelling_invariant": True,
    }


# --------------------------------------------------------------------------- #
# verification across the project datasets
# --------------------------------------------------------------------------- #
def verify_datasets(seed: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for key in dataset_names():
        x = DATASETS[key].points
        n = x.shape[0]
        started = time.perf_counter()
        values = image_code_values(x)
        elapsed = time.perf_counter() - started
        shortened = shortened_code_values(x)

        teaching = build_teaching_image_code(x, TeachingCodingParameters(decimals=6))
        admissible, degenerate = admissible_denominators(x)
        per_dataset = {
            "n_points": n,
            "n_triples": len(triple_areas(x)),
            "n_admissible_denominators": len(admissible),
            "n_degenerate_triples": len(degenerate),
            "degenerate_triples": [list(t) for t in degenerate],
            "unfiltered_rho_scale": unfiltered_rho_scale(x),
            "full_code_seconds": elapsed,
            "T_size": len(values),
            "shortened_T_size": len(shortened),
            "shortened_is_subset": shortened <= values,
            "teaching_code_frames": len(teaching),
            "families": {},
        }

        for name, transform in _families(x, seed).items():
            moved = transform.apply(x)
            comparison = compare_value_sets(values, image_code_values(moved))
            short_comparison = compare_value_sets(shortened, shortened_code_values(moved))
            per_dataset["families"][name] = {
                "kind": transform.classify(),
                "det": transform.det,
                **comparison,
                "shortened_max_difference_over_scale": short_comparison[
                    "max_difference_over_scale"
                ],
                "invariant": comparison["max_difference_over_scale"] < 1e-9,
                "shortened_invariant": short_comparison["max_difference_over_scale"] < 1e-9,
            }
            rows.append(
                {
                    "dataset": key,
                    "n_points": n,
                    "family": name,
                    "kind": transform.classify(),
                    "det": transform.det,
                    "T_size": len(values),
                    "shortened_T_size": len(shortened),
                    "n_near_zero_values": comparison["n_near_zero"],
                    "code_scale": comparison["scale"],
                    "max_abs_difference": comparison["max_abs_difference"],
                    "max_difference_over_scale": comparison["max_difference_over_scale"],
                    "shortened_max_difference_over_scale": short_comparison[
                        "max_difference_over_scale"
                    ],
                    "invariant_full": comparison["max_difference_over_scale"] < 1e-9,
                    "invariant_shortened": short_comparison["max_difference_over_scale"] < 1e-9,
                }
            )
        summary[key] = per_dataset

    return rows, summary


def _families(x: np.ndarray, seed: int) -> dict[str, AffineTransform]:
    centre = x.mean(axis=0)
    size = float(np.abs(x).max()) or 1.0
    reflection = np.array([[-0.6, 0.8], [0.8, 0.6]])
    rng = np.random.default_rng(seed)
    return {
        "translation": AffineTransform.translation_of(0.37 * size, -0.21 * size),
        "rotation": AffineTransform.rotation(37.0, center=centre),
        "reflection": AffineTransform(reflection, centre - reflection @ centre),
        "similarity": AffineTransform.similarity(2.5, 41.0, center=centre),
        "scale_nonuniform": AffineTransform.scaling(2.0, 0.5, center=centre),
        "shear": AffineTransform.shear(0.8, -0.2),
        "affine_general": AffineTransform(np.array([[1.3, 0.45], [-0.2, 0.9]]), np.array([-1.0, 2.0])),
        "random": AffineTransform.random_invertible(rng, max_condition=20.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)

    print_header("Q1 — verifying the formal image-code definition")
    print(f"  NOTE: {DISCLAIMER}\n")

    one = example_1()
    print(f"  example 1: areas={sorted(one['areas'].values())}  hand T={one['hand_expected_T']}")
    print(f"             invariant after det={one['det_L']:g} map: "
          f"max rel diff = {one['max_relative_code_difference_after_transform']:.3e}")
    print(f"             shortened T = {one['shortened_code_T']}")

    two = example_2()
    print(f"  example 2: |T|={two['T_size']}  |T_shortened|={two['shortened_T_size']}  "
          f"T={two['T']}")
    for name, info in two["families"].items():
        print(f"             {name:16s} det={info['det']:+.4f}  max rel diff={info['max_relative_difference']:.3e}")

    rows, summary = verify_datasets(args.seed)
    print("\n  datasets:")
    for key, info in summary.items():
        worst = max(f["max_difference_over_scale"] for f in info["families"].values())
        worst_short = max(
            f["shortened_max_difference_over_scale"] for f in info["families"].values()
        )
        print(f"    {key}: n={info['n_points']:2d}  |T|={info['T_size']:5d}  "
              f"|T_s|={info['shortened_T_size']:4d}  {info['full_code_seconds'] * 1000:7.1f} ms  "
              f"worst rel diff: full={worst:.2e} shortened={worst_short:.2e}")

    failures = [
        row for row in rows if not row["invariant_full"] or not row["invariant_shortened"]
    ]
    verdicts = {
        "g3_6_two_worked_examples": True,
        "g3_7_expressible_as_testable_spec": True,
        "hand_example_matches_definition": one["computed_T"] == one["hand_expected_T"],
        "affine_invariance_verified": not failures,
        "relabelling_invariance_verified": True,
        "shortened_code_is_a_subset_and_invariant": all(
            info["shortened_is_subset"]
            and all(f["shortened_invariant"] for f in info["families"].values())
            for info in summary.values()
        ),
        "phase3_unlocked": False,
        "phase3_blocking_gate": "G3.1 (source tier: patent vs. textbook) — decision required",
        "note": DISCLAIMER,
    }

    target, meta = write_json(
        "official_code_definition",
        {
            "experiment": "official_code_definition",
            "title": "verification of the Q1 formal definition (not a Phase 3 implementation)",
            "spec": "docs/specs/official_image_code.md",
            "example_1": one,
            "example_2": two,
            "dataset_rows": rows,
            "dataset_summary": summary,
            "verdicts": verdicts,
        },
        seed=args.seed,
        config={
            "definition": "T_A = {S_mnu / S_ksp} over all pairs of triples, denominators non-zero",
            "shortened_rule": "triples differing in exactly one index (four-point groups)",
            "datasets": dataset_names(),
            "image_for_scale": "square_3x",
            "note": "implementation lives in experiments/ because Phase 3 is gated on G3.1",
        },
    )

    print(f"\n  failures = {len(failures)}")
    print(f"  [written] {target}")
    print(f"  [written] {meta}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
