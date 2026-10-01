"""Q1 evidence — the gate-G3.6 worked examples and the library-level record.

History, kept because the file changed role
-------------------------------------------
This script was written **before** Phase 3 was unlocked, when ``kozlov_lab.coding`` had to
stay empty.  It therefore contained its own transcription of the definition — appropriate
then, and a liability now: two implementations of one definition eventually diverge.  It has
been rewritten to call the library and to produce only what the gate asked for:

* **G3.6** — two worked examples, each asserted against a value computed by hand;
* **G3.7** — evidence that the definition runs end to end on the project's own data.

The authoritative tests are in ``tests/test_kozlov_exact_oracle.py``; this script exists to
put the numbers on the record in ``experiments/results/``.

Run::

    python experiments/exp_official_code_definition.py
"""

from __future__ import annotations

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.coding import (  # noqa: E402
    affine_equivalent_exact,
    build_kozlov_image_code_exact,
    cross_check_exact,
    validate_kozlov_image,
)
from kozlov_lab.datasets.corpus import DATASETS, dataset_names  # noqa: E402
from _common import print_header, write_json  # noqa: E402

NOTE = (
    "This script produces gate evidence (G3.6, G3.7) by calling the Phase 3 library. It "
    "contains no implementation of the definition: two implementations of one definition "
    "would eventually diverge, and the library plus tests/test_kozlov_exact_oracle.py are "
    "the single source of truth."
)


# --------------------------------------------------------------------------- #
# G3.6 — the two worked examples
# --------------------------------------------------------------------------- #
def example_1() -> dict:
    """``(0,0), (2,0), (0,2), (1,1)`` — hand-checked entry count and value multiset.

    Hand computation.  Doubled areas: ``(0,1,2)=4``, ``(0,1,3)=2``, ``(0,2,3)=2``,
    ``(1,2,3)=0``.  Three of the four triples are admissible denominators, and the two
    triples of a ratio must be *distinct*, so

        |T_A| = 4 x 3 - 3 = 9

    with values ``{2, 2, 1/2, 1, 1/2, 1, 0, 0, 0}``, i.e. ``0 (x3), 1/2 (x2), 1 (x2), 2 (x2)``.

    Second hand check: apply ``M = [[2,1],[0,3]]``, ``det M = 6``.  Every doubled area scales
    by 6, so every ratio is unchanged, and the indexed entries must be **identical** (not
    merely close) because the arithmetic is exact.
    """
    points = [[0, 0], [2, 0], [0, 2], [1, 1]]
    code = build_kozlov_image_code_exact(points)

    expected_multiset = {
        Fraction(0): 3,
        Fraction(1, 2): 2,
        Fraction(1): 2,
        Fraction(2): 2,
    }
    assert len(code) == 9, len(code)
    assert code.value_multiset() == expected_multiset, code.value_multiset()
    assert code.degenerate_triples == frozenset({frozenset({1, 2, 3})})
    assert code.is_complete

    matrix = ((2, 1), (0, 3))
    translation = (5, -7)
    moved = [
        [matrix[0][0] * x + matrix[0][1] * y + translation[0],
         matrix[1][0] * x + matrix[1][1] * y + translation[1]]
        for x, y in points
    ]
    moved_code = build_kozlov_image_code_exact(moved)
    assert moved_code.entries == code.entries, "an integer affine map must preserve entries exactly"

    return {
        "points": points,
        "n_entries": len(code),
        "hand_expected_value_multiset": {str(k): v for k, v in expected_multiset.items()},
        "computed_value_multiset": {str(k): v for k, v in code.value_multiset().items()},
        "degenerate_triples": [sorted(t) for t in code.degenerate_triples],
        "det_M": 6,
        "entries_identical_after_transform": True,
        "domain": code.domain,
    }


def example_2() -> dict:
    """A 5-point figure, four integer affine maps, and the relabelling cross-check.

    The geometric comparator of the library is used as a second, independent route: the
    code route and the geometric route share nothing but the input coordinates, so their
    agreement is evidence about the implementation rather than about itself.
    """
    points = [[0, 0], [3, 0], [0, 4], [1, 1], [5, 2]]
    code = build_kozlov_image_code_exact(points)
    assert len(code) > 0

    families = {
        "translation": ((1, 0, 0, 1), (11, -9)),
        "similarity": ((3, 0, 0, 3), (0, 0)),
        "shear": ((1, 1, 0, 2), (0, 0)),
        "generic": ((1, 2, 3, 5), (-4, 6)),
    }
    rows: dict[str, dict] = {}
    for name, (matrix, translation) in families.items():
        moved = [
            [matrix[0] * x + matrix[1] * y + translation[0],
             matrix[2] * x + matrix[3] * y + translation[1]]
            for x, y in points
        ]
        moved_code = build_kozlov_image_code_exact(moved)
        geometry = affine_equivalent_exact(points, moved)
        rows[name] = {
            "entries_identical": moved_code.entries == code.entries,
            "geometry_route_found_a_map": geometry.equivalent,
            "det": matrix[0] * matrix[3] - matrix[1] * matrix[2],
        }
        assert rows[name]["entries_identical"], name
        assert geometry.equivalent, name

    permutation = [4, 2, 0, 3, 1]
    relabelled = [points[i] for i in permutation]
    report = cross_check_exact(points, relabelled)
    assert report["routes_agree"] is True, report

    return {
        "points": points,
        "n_entries": len(code),
        "families": rows,
        "relabelling": {
            "permutation": permutation,
            "code_route_equivalent": report["code_route"]["equivalent"],
            "geometry_route_equivalent": report["geometry_route"]["equivalent"],
            "routes_agree": report["routes_agree"],
        },
    }


# --------------------------------------------------------------------------- #
# dataset-level record
# --------------------------------------------------------------------------- #
def dataset_record() -> tuple[list[dict], dict]:
    """Validate each dataset and record which of the three layers applied.

    The project datasets are built in floating point, so most are *not* exactly
    representable and the exact layer refuses them.  That refusal is itself the datum: it
    shows the adapter doing its job instead of silently rounding.
    """
    from fractions import Fraction as F

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for key in dataset_names():
        points = DATASETS[key].points
        row: dict = {"dataset": key, "n_points": int(points.shape[0])}
        try:
            verification = validate_kozlov_image(points)
            row.update(
                layer="exact",
                domain=verification.domain,
                theorem_applicable=verification.theorem_applicable,
                numerically_well_conditioned=verification.numerically_well_conditioned,
                n_degenerate_triples=len(verification.degenerate_triples),
            )
        except Exception as exc:  # noqa: BLE001 - the *kind* of refusal is the datum
            row.update(layer="refused", reason=type(exc).__name__, note=str(exc)[:200])
        rows.append(row)
        summary[key] = row

    # one dataset re-expressed in exact rationals, to exercise the exact path end to end
    trapezoid = [
        (F(0), F(0)),
        (F(4), F(1, 2)),
        (F(11, 4), F(41, 20)),
        (F(3, 4), F(9, 5)),
        (F(5, 4), F(149, 80)),
    ]
    verification = validate_kozlov_image(trapezoid)
    summary["rational_trapezoid"] = {
        "layer": "exact",
        "n_points": len(trapezoid),
        "domain": verification.domain,
        "code_definable": verification.code_definable,
        "theorem_applicable": verification.theorem_applicable,
        "note": "the two bases are parallel exactly, so the code exists but theorem 1 does not apply",
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)

    print_header("Q1 evidence — gate G3.6 (worked examples) and G3.7 (implementable spec)")
    print(f"  {NOTE}\n")

    started = time.perf_counter()
    one = example_1()
    print(f"  example 1: |T_A| = {one['n_entries']} (hand: 9)  values {one['computed_value_multiset']}")
    print(f"             entries identical after a det=6 affine map: {one['entries_identical_after_transform']}")

    two = example_2()
    print(f"  example 2: |T_A| = {two['n_entries']}, {len(two['families'])} families")
    for name, info in two["families"].items():
        print(f"             {name:12s} det={info['det']:+d}  entries identical={info['entries_identical']}")
    print(f"             relabelling: routes_agree={two['relabelling']['routes_agree']}")

    rows, summary = dataset_record()
    elapsed = time.perf_counter() - started
    print("\n  datasets (the layer that applied is the datum):")
    print("    note: the A-E datasets are BUILT in floating point, so many coordinates are")
    print("          not exactly representable and the exact layer refuses them by design.")
    print("          That refusal is the datum: use the numeric layer for float data.")
    for row in rows:
        if row["layer"] == "exact":
            print(f"    {row['dataset']}: exact, domain={row['domain']}")
        else:
            print(f"    {row['dataset']}: {row['layer']} ({row['reason']})")
    extra = summary["rational_trapezoid"]
    print(f"    rational_trapezoid: domain={extra['domain']}, theorem_applicable={extra['theorem_applicable']}")

    verdicts = {
        "g3_6_two_worked_examples": True,
        "g3_7_expressible_as_testable_spec": True,
        "hand_example_matches": one["n_entries"] == 9,
        "affine_map_preserves_entries_exactly": one["entries_identical_after_transform"],
        "both_routes_agree_on_relabelling": two["relabelling"]["routes_agree"] is True,
        "three_layers_are_distinguished": extra["theorem_applicable"] is False,
        "phase3_unlocked": True,
        "authoritative_tests": (
            "tests/test_kozlov_exact_oracle.py, test_kozlov_domain_and_validation.py, "
            "test_kozlov_equivalence.py"
        ),
        "note": NOTE,
    }

    target, meta = write_json(
        "official_code_definition",
        {
            "experiment": "official_code_definition",
            "title": "Q1 gate evidence: worked examples and dataset-level record",
            "spec": "docs/specs/official_image_code.md",
            "example_1": one,
            "example_2": two,
            "dataset_rows": rows,
            "dataset_summary": summary,
            "verdicts": verdicts,
            "elapsed_seconds": elapsed,
        },
        seed=args.seed,
        config={
            "definition": "T_A = {S_mnu / S_ksp}; triples unordered and distinct; denominators non-degenerate",
            "source": "Kozlov 2009, Fundam. Prikl. Mat. 15:5, 95-110 (primary full text)",
            "layers_checked": ["code_definable", "theorem_applicable", "numerically_well_conditioned"],
        },
    )
    print(f"\n  [written] {target}")
    print(f"  [written] {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
