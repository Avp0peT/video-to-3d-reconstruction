"""F5.2 — compression equivariance and recognition preservation, with controls separated.

The first version of this experiment reported "0 / 768 recognitions preserved" and was wrong:
it fed raw floating-point coordinates to the *exact* recogniser, which refused them, and the
harness counted the refusals as "code undefined after compression" (see
``experiments/exp_phase5_sanity.py`` for the audit that caught it, and the three defects it
found: non-exact corpus, a compressor that coerced its input to float64, and a numeric rule
that called an exactly-zero residual "indeterminate").

What this version does differently:

* the corpus is **exact rationals** (denominator ``10**3``), so the exact pipeline applies;
* the transform families are **integer maps**, so coordinates stay exact through the transform;
* **controls are run and reported separately** from nontrivial compression.  Mixing them is how
  a pipeline failure hides as a scientific finding;
* every row carries an F5.1 ``RecognitionPreservation`` category, and ``RECOGNIZER_DISCREPANCY``
  exists precisely so a Phase 4 inconsistency cannot be reported as a compression effect.

Run::

    python experiments/exp_phase5_compression.py
    python experiments/exp_phase5_compression.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.coding import (  # noqa: E402
    CollinearImageError,
    InternalConsistencyError,
    RecognitionStatus,
    TooFewPointsError,
    recognize_affine_exact,
)
from kozlov_lab.compression import (  # noqa: E402
    code_size_reduction,
    coverage_report,
    get_compressor,
    list_compressors,
)
from _common import (  # noqa: E402
    apply_rational_affine,
    print_header,
    quantized_rational_image,
    write_csv,
    write_json,
)

#: F5.1 — seven categories, because "preserved: yes/no" cannot express the failure modes.
class Preservation(str, Enum):
    """F5.1's categories, plus one the specification's list turned out to lack.

    ``AFFINE_EQUIVALENCE_LOST`` had to be added, and the reason is worth recording: F5.1's seven
    categories have no slot for *the compression broke an equivalence that held before*, which is
    the single most common real outcome.  Without it those rows fell into
    ``RECOGNIZER_DISCREPANCY`` — a category that means "a Phase 4 inconsistency appeared" — and
    the headline would have reported 32 implementation defects where the correct reading is 32
    genuine losses.  A taxonomy that cannot express its own most frequent result will
    misattribute it.
    """

    EXACTLY_PRESERVED = "EXACTLY_PRESERVED"
    #: Held before compression, does not hold after.  The core Q2 outcome.
    AFFINE_EQUIVALENCE_LOST = "AFFINE_EQUIVALENCE_LOST"
    STRUCTURE_LOST_CARDINALITY = "STRUCTURE_LOST_CARDINALITY"
    CODE_UNDEFINED = "CODE_UNDEFINED"
    OUTSIDE_THEOREM_DOMAIN = "OUTSIDE_THEOREM_DOMAIN"
    CODE_COLLISION_OUTSIDE_DOMAIN = "CODE_COLLISION_OUTSIDE_DOMAIN"
    NUMERIC_INDETERMINATE = "NUMERIC_INDETERMINATE"
    #: A Phase 4 internal inconsistency — must never be produced by a legitimate negative.
    RECOGNIZER_DISCREPANCY = "RECOGNIZER_DISCREPANCY"
    INVALID_INPUT_AFTER_COMPRESSION = "INVALID_INPUT_AFTER_COMPRESSION"


#: Integer maps: det != 0, and rational coordinates stay rational under them.
FAMILIES: dict[str, tuple[tuple, tuple]] = {
    "translation": ((1, 0, 0, 1), (3, -2)),
    "rotation": ((0, -1, 1, 0), (2, 1)),
    "similarity": ((2, 0, 0, 2), (-1, 1)),
    "affine": ((2, 1, 0, 3), (1, -1)),
    "affine_shear": ((1, 1, 0, 2), (0, 0)),
}


def shapes() -> dict[str, list[tuple]]:
    """Six shapes (G5.7 asks for five), as exact rationals."""
    from kozlov_lab.datasets.corpus import DATASETS

    data: dict[str, list[tuple]] = {}
    for key in ("A", "B", "C", "D", "E"):
        data[f"dataset_{key}"] = quantized_rational_image(DATASETS[key].points)
    data["stair"] = quantized_rational_image(
        [[i * 1.3 + (i % 3) * 0.4, i * i * 0.05 + (i % 5) * 0.7] for i in range(9)]
    )
    return data


def grid(quick: bool) -> list[tuple[str, str, list[float], bool]]:
    """``(compressor, parameter, strengths, is_control)``.

    The **controls come first and are flagged**, so no report can average them together with
    real compression without saying so.
    """
    if quick:
        return [
            ("IdentityCompressor", "-", [0.0], True),
            ("KozlovSketch", "r", [0.0], True),
            ("KozlovSketch", "r", [0.3, 1.0], False),
            ("TeachingEpsilonCover", "epsilon", [0.3, 1.0], False),
            ("TeachingFarthestPoint", "k", [6.0, 4.0], False),
            ("TeachingGridReduce", "cell", [0.3, 1.2], False),
        ]
    return [
        ("IdentityCompressor", "-", [0.0], True),
        ("KozlovSketch", "r", [0.0], True),
        ("KozlovSkeleton", "r", [0.0], True),
        ("KozlovSketch", "r", [0.05, 0.2, 0.5, 1.0, 1.5], False),
        ("KozlovSkeleton", "r", [0.05, 0.2, 0.5, 1.0], False),
        ("TeachingEpsilonCover", "epsilon", [0.05, 0.2, 0.5, 1.0], False),
        ("TeachingFarthestPoint", "k", [8.0, 6.0, 5.0, 4.0], False),
        ("TeachingGridReduce", "cell", [0.05, 0.3, 0.8, 1.5], False),
        ("AffineAwareSampler", "r", [0.05, 0.2, 0.5], False),
    ]


def _apply(compressor, points, parameter: str, strength: float):
    kwargs = {} if parameter == "-" else {parameter: int(strength) if parameter == "k" else strength}
    return compressor.fit(points, **kwargs)


def _categorise(result, n_x: int, n_y: int) -> Preservation:
    """F5.1's taxonomy, applied to a recogniser result plus the two cardinalities."""
    if n_x != n_y:
        return Preservation.STRUCTURE_LOST_CARDINALITY
    if result.status is RecognitionStatus.EXACT_AFFINE_EQUIVALENT:
        return Preservation.EXACTLY_PRESERVED
    if result.status is RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT:
        # a *legitimate* negative: the compression destroyed a relation that held. Reporting this
        # as a recogniser discrepancy would turn every real loss into a phantom bug.
        return Preservation.AFFINE_EQUIVALENCE_LOST
    if result.status is RecognitionStatus.CODE_COLLISION_OUTSIDE_THEOREM_DOMAIN:
        return Preservation.CODE_COLLISION_OUTSIDE_DOMAIN
    if result.status is RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE:
        return Preservation.OUTSIDE_THEOREM_DOMAIN
    if result.status is RecognitionStatus.NUMERIC_INDETERMINATE:
        return Preservation.NUMERIC_INDETERMINATE
    if result.status is RecognitionStatus.CODE_UNDEFINED:
        return Preservation.CODE_UNDEFINED
    return Preservation.INVALID_INPUT_AFTER_COMPRESSION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)

    print_header("F5.2 — compression: equivariance and recognition preservation")
    print("  contracts (family names carry the provenance)")
    for contract in list_compressors():
        print(f"    {contract['name']:<24} {contract['family']:<13} "
              f"cover={contract['cover_guarantee']:<21} card={contract['cardinality_guarantee']}")

    equivariance: list[dict] = []
    recognition: list[dict] = []
    started = time.perf_counter()

    for shape_name, points in shapes().items():
        scale = float(np.ptp(np.asarray(points, dtype=float), axis=0).max()) or 1.0
        for name, parameter, strengths, is_control in grid(args.quick):
            compressor = get_compressor(name)()
            for strength in strengths:
                try:
                    reference = _apply(compressor, points, parameter, strength)
                except Exception as exc:  # noqa: BLE001 - a refusal is a result
                    equivariance.append({
                        "shape": shape_name, "compressor": name, "parameter": parameter,
                        "strength": strength, "family": "-", "is_control": is_control,
                        "n_before": len(points), "n_after": 0, "verdict": "refused",
                        "note": f"{type(exc).__name__}: {exc}"[:200],
                    })
                    continue

                coverage = coverage_report(
                    np.asarray(points, dtype=float), np.asarray(reference.points, dtype=float)
                )
                for family, (matrix, translation) in FAMILIES.items():
                    moved = apply_rational_affine(matrix, translation, points)
                    try:
                        moved_sketch = _apply(compressor, moved, parameter, strength)
                    except Exception as exc:  # noqa: BLE001
                        equivariance.append({
                            "shape": shape_name, "compressor": name, "parameter": parameter,
                            "strength": strength, "family": family, "is_control": is_control,
                            "n_before": len(points), "n_after": reference.n_after,
                            "verdict": "refused_after_transform",
                            "note": f"{type(exc).__name__}: {exc}"[:200],
                        })
                        continue

                    predicted = apply_rational_affine(matrix, translation, reference.points)
                    left = np.asarray(predicted, dtype=float)
                    right = np.asarray(moved_sketch.points, dtype=float)
                    if left.shape[0] != right.shape[0]:
                        verdict, discrepancy = "not", float("inf")
                    else:
                        discrepancy = float(
                            np.abs(np.array(sorted(map(tuple, left)))
                                   - np.array(sorted(map(tuple, right)))).max()
                        )
                        tolerance = 1e-9 * max(scale, 1.0)
                        verdict = (
                            "exact" if discrepancy <= tolerance
                            else "approximately" if discrepancy <= 0.05 * scale
                            else "not"
                        )
                    equivariance.append({
                        "shape": shape_name, "compressor": name, "parameter": parameter,
                        "strength": strength, "family": family, "is_control": is_control,
                        "n_before": len(points), "n_after": reference.n_after,
                        "n_after_transformed": moved_sketch.n_after,
                        "compression_ratio": reference.compression_ratio,
                        "coverage_radius": reference.achieved_radius,
                        "hausdorff_original_sketch": coverage["hausdorff"],
                        "sketch_is_subset": coverage["sketch_is_subset"],
                        "verdict": verdict, "discrepancy": discrepancy, "note": "",
                    })

                for family, (matrix, translation) in FAMILIES.items():
                    moved = apply_rational_affine(matrix, translation, points)
                    try:
                        moved_sketch = _apply(compressor, moved, parameter, strength)
                    except Exception:  # noqa: BLE001
                        continue

                    category: Preservation
                    status = ""
                    note = ""
                    residual = ""
                    elapsed_ms = float("nan")
                    if reference.n_after != moved_sketch.n_after:
                        category = Preservation.STRUCTURE_LOST_CARDINALITY
                        note = (f"|S_X| = {reference.n_after} vs |S_Y| = {moved_sketch.n_after}: "
                                "the compressed images differ in cardinality before any geometry")
                    else:
                        try:
                            t0 = time.perf_counter()
                            result = recognize_affine_exact(reference.points, moved_sketch.points)
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            status = str(result.status)
                            residual = result.residual
                            category = _categorise(result, reference.n_after, moved_sketch.n_after)
                        except InternalConsistencyError as exc:
                            category = Preservation.RECOGNIZER_DISCREPANCY
                            note = str(exc)[:200]
                        except (CollinearImageError, TooFewPointsError) as exc:
                            # |S| < 3 or all-collinear *after* compression: the code needs a
                            # non-degenerate triple, and the input had one before compression.
                            category = Preservation.CODE_UNDEFINED
                            status = type(exc).__name__
                            note = f"{type(exc).__name__}: {exc}"[:200]
                        except Exception as exc:  # noqa: BLE001
                            category = Preservation.INVALID_INPUT_AFTER_COMPRESSION
                            status = type(exc).__name__
                            note = f"{type(exc).__name__}: {exc}"[:200]

                    sizes = code_size_reduction(len(points), reference.n_after)
                    recognition.append({
                        "shape": shape_name, "compressor": name, "parameter": parameter,
                        "strength": strength, "family": family, "is_control": is_control,
                        "n_original": len(points), "n_compressed": reference.n_after,
                        "compression_ratio": reference.compression_ratio,
                        "code_entries_before": sizes["n_entries_before"],
                        "code_entries_after": sizes["n_entries_after"],
                        "code_size_ratio": sizes["code_size_ratio"],
                        "code_size_speedup": sizes["code_size_speedup"],
                        "recognition_status": status, "category": category.value,
                        "residual": residual, "compression_ms": reference.build_ms,
                        "recognition_ms": elapsed_ms, "note": note,
                    })

    elapsed = time.perf_counter() - started

    equivariance_target, equivariance_meta = write_csv(
        "phase5_equivariance", equivariance,
        fieldnames=("shape", "compressor", "parameter", "strength", "family", "is_control",
                    "n_before", "n_after", "n_after_transformed", "compression_ratio",
                    "coverage_radius", "hausdorff_original_sketch", "sketch_is_subset",
                    "verdict", "discrepancy", "note"),
        seed=args.seed,
        config={"quick": args.quick, "corpus": "exact rationals (10^-3)",
                "families": list(FAMILIES), "integer_maps": True},
    )
    recognition_target, recognition_meta = write_csv(
        "phase5_recognition", recognition,
        fieldnames=("shape", "compressor", "parameter", "strength", "family", "is_control",
                    "n_original", "n_compressed", "compression_ratio", "code_entries_before",
                    "code_entries_after", "code_size_ratio", "code_size_speedup",
                    "recognition_status", "category", "residual", "compression_ms",
                    "recognition_ms", "note"),
        seed=args.seed,
        config={"quick": args.quick, "corpus": "exact rationals (10^-3)", "taxonomy": "F5.1"},
    )

    # -- report controls and nontrivial compression **separately** -------- #
    def tally(rows: list[dict], *, control: bool) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            if bool(row["is_control"]) is not control:
                continue
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        return counts

    control_counts = tally(recognition, control=True)
    real_counts = tally(recognition, control=False)
    control_total = sum(control_counts.values())
    real_total = sum(real_counts.values())

    print("\n  CONTROLS (identity, sketch at r = 0) — must be 100% preserved")
    print(f"    preserved/total: {control_counts.get('EXACTLY_PRESERVED', 0)}/{control_total}")
    for key, value in sorted(control_counts.items()):
        print(f"      {key:<36} {value}")

    print("\n  NONTRIVIAL COMPRESSION")
    print(f"    preserved/total: {real_counts.get('EXACTLY_PRESERVED', 0)}/{real_total}")
    for key, value in sorted(real_counts.items()):
        print(f"      {key:<36} {value}")

    by_family: dict[str, dict[str, int]] = {}
    for row in equivariance:
        if row["is_control"]:
            continue
        key = f"{row['compressor']}|{row['family']}"
        by_family.setdefault(key, {})
        by_family[key][str(row["verdict"])] = by_family[key].get(str(row["verdict"]), 0) + 1
    print("\n  EQUIVARIANCE by compressor and family (nontrivial only, G5.4)")
    for key in sorted(by_family):
        print(f"    {key:<40} {by_family[key]}")

    print(f"\n  rows: equivariance={len(equivariance)} recognition={len(recognition)}  "
          f"elapsed={elapsed:.1f}s")

    summary, summary_meta = write_json(
        "phase5_compression_summary",
        {
            "experiment": "phase5_compression",
            "title": "compression: equivariance and recognition preservation (controls separated)",
            "controls": {"counts": control_counts, "total": control_total,
                         "preserved": control_counts.get("EXACTLY_PRESERVED", 0)},
            "nontrivial": {"counts": real_counts, "total": real_total,
                           "preserved": real_counts.get("EXACTLY_PRESERVED", 0),
                           "preservation_rate": (real_counts.get("EXACTLY_PRESERVED", 0) / real_total)
                           if real_total else None},
            "equivariance_by_family": by_family,
            "contracts": list_compressors(),
            "elapsed_seconds": elapsed,
            "note": (
                "controls are reported separately on purpose: the earlier run mixed them with "
                "real compression and produced a headline that was an artefact of the harness"
            ),
        },
        seed=args.seed,
        config={"quick": args.quick},
    )
    for path in (equivariance_target, equivariance_meta, recognition_target, recognition_meta,
                 summary, summary_meta):
        print(f"  [written] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
