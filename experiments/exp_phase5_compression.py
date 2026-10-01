"""5D/5E/5F/5H/5I — equivariance, recognition-after-compression, and the cost coupling.

The specification is explicit that this is the phase's most important experiment, and that the
compression *ratio* is not the headline:

    C(T X)  ==?  T(C X)          (equivariance, 5D)
    S_X ~_aff S_Y  ?             (recognition preservation, 5E)

Both are measured over **5+ shapes × several compression strengths × 4 transform families**
(G5.7), and every failure is recorded with its **reason** rather than counted as a generic
"recognition failure" (5F).

Two things this script deliberately does *not* do:

* it does not assume the Euclidean cover is affine-compatible.  Euclidean distance is not
  preserved by a general affine map, so the expectation — stated in the specification and
  confirmed here — is that equivariance holds for translation, degrades for rotation, and fails
  for a general affine map.  That is a *result*, not a bug;
* it does not report a bare "recognition rate".  ``STRUCTURE_LOST`` (the two sketches have
  different cardinalities) is a different outcome from a genuine recognition failure, and the
  taxonomy keeps them apart.

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
    InternalConsistencyError,
    RecognitionStatus,
    recognize_affine_exact,
)
from kozlov_lab.compression import (  # noqa: E402
    code_size_reduction,
    coverage_report,
    get_compressor,
    hausdorff,
    list_compressors,
)
from _common import print_header, write_csv, write_json  # noqa: E402


class FailureKind(str, Enum):
    """5F's failure taxonomy — every outcome keeps its reason."""

    RECOGNISED = "RECOGNISED"
    STRUCTURE_LOST = "STRUCTURE_LOST"
    COMPRESSOR_NON_EQUIVARIANCE = "COMPRESSOR_NON_EQUIVARIANCE"
    CODE_UNDEFINED_AFTER_COMPRESSION = "CODE_UNDEFINED_AFTER_COMPRESSION"
    OUTSIDE_THEOREM_DOMAIN = "OUTSIDE_THEOREM_DOMAIN"
    NUMERIC_INDETERMINATE = "NUMERIC_INDETERMINATE"
    RECOGNISER_FAILURE = "RECOGNISER_FAILURE"
    INTERNAL_INCONSISTENCY = "INTERNAL_INCONSISTENCY"


#: Five shapes (G5.7), all asymmetric for the reason Phases 3–4 established.
def shapes() -> dict[str, list[list[float]]]:
    from kozlov_lab.datasets.corpus import DATASETS

    data: dict[str, list[list[float]]] = {}
    for key in ("A", "B", "C", "D", "E"):
        data[f"dataset_{key}"] = [list(map(float, point)) for point in DATASETS[key].points]
    # a 6th, larger shape so the O(n^6) coupling is visible
    data["stair"] = [[i * 1.3 + (i % 3) * 0.4, i * i * 0.05 + (i % 5) * 0.7] for i in range(9)]
    return data


#: (name, callable, strengths) — the strength grid is per compressor because they are
#: parameterised differently (radius vs cardinality vs cell size).
def compressor_grid(quick: bool) -> list[tuple[str, str, list[float]]]:
    from kozlov_lab.compression import teaching

    del teaching
    if quick:
        return [
            ("KozlovSketch", "r", [0.9]),
            ("KozlovSkeleton", "r", [0.9]),
            ("TeachingEpsilonCover", "epsilon", [0.9]),
            ("TeachingFarthestPoint", "k", [6.0]),
            ("TeachingGridReduce", "cell", [1.2]),
            ("AffineAwareSampler", "r", [0.7]),
        ]
    # The strength grid must reach the **low-compression** end, otherwise the trade-off curve
    # has no left-hand side: a first version only used strong compression, every sketch came out
    # degenerate, and "recognition is never preserved" would have looked like a property of the
    # method rather than of the strengths chosen.
    return [
        ("KozlovSketch", "r", [0.05, 0.2, 0.4, 0.7, 1.0, 1.5]),
        ("KozlovSkeleton", "r", [0.05, 0.2, 0.4, 0.7, 1.0]),
        ("TeachingEpsilonCover", "epsilon", [0.05, 0.2, 0.4, 0.7, 1.0, 1.5]),
        ("TeachingFarthestPoint", "k", [8.0, 7.0, 6.0, 5.0, 4.0]),
        ("TeachingGridReduce", "cell", [0.05, 0.2, 0.4, 0.8, 1.2, 1.6]),
        ("AffineAwareSampler", "r", [0.05, 0.15, 0.3, 0.5, 0.8]),
    ]


def transform_families(points: np.ndarray) -> dict[str, tuple[tuple, tuple]]:
    """The four families of 5D, as explicit integer/rational affine maps ``(matrix, translation)``."""
    centre = points.mean(axis=0)
    return {
        "translation": ((1.0, 0.0, 0.0, 1.0), (3.0, -2.0)),
        "rotation": ((0.0, -1.0, 1.0, 0.0), (-centre[1], centre[0])),  # 90 degrees about the centre
        "similarity": ((1.5, 0.0, 0.0, 1.5), (-0.5 * centre[0], -0.5 * centre[1])),
        "affine": ((1.3, 0.5, -0.2, 0.9), (1.0, -1.0)),
    }


def apply(matrix: tuple, translation: tuple, points: np.ndarray) -> np.ndarray:
    a, b, c, d = matrix
    tx, ty = translation
    return np.array([[a * x + b * y + tx, c * x + d * y + ty] for x, y in points], dtype=float)


def equivariance_verdict(
    transformed_sketch: np.ndarray, sketch_of_transformed: np.ndarray, *, scale: float
) -> tuple[str, float]:
    """``exact`` / ``approximately`` / ``not`` equivariant, plus the measured discrepancy.

    "Approximately" is defined against the image's own scale rather than an absolute epsilon:
    a discrepancy of ``1e-3`` is a different thing on an image of size 1 and one of size 1e3,
    and an absolute threshold would silently mean different things on different shapes.
    """
    if transformed_sketch.shape[0] != sketch_of_transformed.shape[0]:
        return "not", float("inf")
    left = np.array(sorted(map(tuple, transformed_sketch)))
    right = np.array(sorted(map(tuple, sketch_of_transformed)))
    discrepancy = float(np.abs(left - right).max())
    tolerance = 1e-9 * max(scale, 1.0)
    if discrepancy <= tolerance:
        return "exact", discrepancy
    if discrepancy <= 0.05 * scale:
        return "approximately", discrepancy
    return "not", discrepancy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)

    print_header("Phase 5D/5E — compression equivariance and recognition preservation")
    print("  contracts")
    for contract in list_compressors():
        print(
            f"    {contract['name']:<24} {contract['family']:<12} "
            f"cover={contract['cover_guarantee']:<22} card={contract['cardinality_guarantee']}"
        )

    equivariance_rows: list[dict] = []
    recognition_rows: list[dict] = []
    started = time.perf_counter()

    for shape_name, points in shapes().items():
        matrix = np.asarray(points, dtype=float)
        scale = float(np.ptp(matrix, axis=0).max()) or 1.0
        for compressor_name, parameter, strengths in compressor_grid(args.quick):
            compressor = get_compressor(compressor_name)()
            for strength in strengths:
                kwargs = {parameter: int(strength) if parameter == "k" else strength}
                try:
                    reference = compressor.fit(matrix, **kwargs)
                except Exception as exc:  # noqa: BLE001 - a refusal is a result
                    equivariance_rows.append(
                        {
                            "shape": shape_name,
                            "compressor": compressor_name,
                            "parameter": parameter,
                            "strength": strength,
                            "family": "-",
                            "n_before": matrix.shape[0],
                            "n_after": 0,
                            "compression_ratio": "",
                            "verdict": "refused",
                            "discrepancy": "",
                            "note": f"{type(exc).__name__}: {exc}"[:200],
                        }
                    )
                    continue

                coverage = coverage_report(matrix, reference.points)
                for family, (map_matrix, translation) in transform_families(matrix).items():
                    moved = apply(map_matrix, translation, matrix)
                    try:
                        moved_sketch = compressor.fit(moved, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        equivariance_rows.append(
                            {
                                "shape": shape_name,
                                "compressor": compressor_name,
                                "parameter": parameter,
                                "strength": strength,
                                "family": family,
                                "n_before": matrix.shape[0],
                                "n_after": reference.n_after,
                                "compression_ratio": reference.compression_ratio,
                                "verdict": "refused_after_transform",
                                "discrepancy": "",
                                "note": f"{type(exc).__name__}: {exc}"[:200],
                            }
                        )
                        continue

                    predicted = apply(map_matrix, translation, reference.points)
                    verdict, discrepancy = equivariance_verdict(
                        predicted, moved_sketch.points, scale=scale
                    )
                    equivariance_rows.append(
                        {
                            "shape": shape_name,
                            "compressor": compressor_name,
                            "parameter": parameter,
                            "strength": strength,
                            "family": family,
                            "n_before": matrix.shape[0],
                            "n_after": reference.n_after,
                            "n_after_transformed": moved_sketch.n_after,
                            "compression_ratio": reference.compression_ratio,
                            "coverage_radius": reference.achieved_radius,
                            "hausdorff_original_sketch": coverage["hausdorff"],
                            "sketch_is_subset": coverage["sketch_is_subset"],
                            "verdict": verdict,
                            "discrepancy": discrepancy,
                            "note": "",
                        }
                    )

                # -- 5E: recognition after compression --------------------- #
                for family, (map_matrix, translation) in transform_families(matrix).items():
                    moved = apply(map_matrix, translation, matrix)
                    try:
                        moved_sketch = compressor.fit(moved, **kwargs)
                    except Exception:  # noqa: BLE001
                        continue
                    failure = FailureKind.RECOGNISED
                    status = ""
                    note = ""
                    residual = ""
                    if reference.n_after != moved_sketch.n_after:
                        failure = FailureKind.STRUCTURE_LOST
                        note = (
                            f"|S_X| = {reference.n_after} but |S_Y| = {moved_sketch.n_after}; the two "
                            "compressed images do not even have the same cardinality"
                        )
                    elif reference.n_after < 3:
                        failure = FailureKind.CODE_UNDEFINED_AFTER_COMPRESSION
                        note = f"|S| = {reference.n_after} is below the minimum for a code"
                    else:
                        try:
                            t0 = time.perf_counter()
                            result = recognize_affine_exact(
                                reference.points.tolist(), moved_sketch.points.tolist()
                            )
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            status = str(result.status)
                            residual = result.residual
                            if result.status is RecognitionStatus.EXACT_AFFINE_EQUIVALENT:
                                failure = FailureKind.RECOGNISED
                            elif result.status is RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT:
                                failure = FailureKind.RECOGNISER_FAILURE
                            elif result.status is RecognitionStatus.CODE_COLLISION_OUTSIDE_THEOREM_DOMAIN:
                                failure = FailureKind.OUTSIDE_THEOREM_DOMAIN
                            elif result.status in (
                                RecognitionStatus.NUMERIC_INDETERMINATE,
                                RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE,
                            ):
                                failure = FailureKind.NUMERIC_INDETERMINATE
                            else:
                                failure = FailureKind.CODE_UNDEFINED_AFTER_COMPRESSION
                                note = str(result.status)
                        except InternalConsistencyError as exc:
                            elapsed_ms = float("nan")
                            failure = FailureKind.INTERNAL_INCONSISTENCY
                            note = str(exc)[:200]
                        except Exception as exc:  # noqa: BLE001
                            from kozlov_lab.coding import (
                                CollinearImageError,
                                TooFewPointsError,
                            )

                            elapsed_ms = float("nan")
                            # 5F: "not codeable after compression" and "the recogniser failed"
                            # are different outcomes, and bundling every exception into the first
                            # would have made the taxonomy look clean while hiding the reason.
                            if isinstance(exc, (CollinearImageError, TooFewPointsError)):
                                failure = FailureKind.CODE_UNDEFINED_AFTER_COMPRESSION
                            else:
                                failure = FailureKind.RECOGNISER_FAILURE
                            note = f"{type(exc).__name__}: {exc}"[:200]

                    sizes = code_size_reduction(reference.n_before, reference.n_after)
                    recognition_rows.append(
                        {
                            "shape": shape_name,
                            "compressor": compressor_name,
                            "parameter": parameter,
                            "strength": strength,
                            "family": family,
                            "n_original": reference.n_before,
                            "n_compressed": reference.n_after,
                            "compression_ratio": reference.compression_ratio,
                            "code_entries_before": sizes["n_entries_before"],
                            "code_entries_after": sizes["n_entries_after"],
                            "code_size_ratio": sizes["code_size_ratio"],
                            "code_size_speedup": sizes["code_size_speedup"],
                            "recognition_status": status,
                            "reason": failure.value,
                            "residual": residual,
                            "compression_ms": reference.build_ms,
                            "note": note,
                        }
                    )

    elapsed = time.perf_counter() - started

    equivariance_rows_out = [
        {key: ("" if value is None else value) for key, value in row.items()} for row in equivariance_rows
    ]
    target, meta = write_csv(
        "phase5_equivariance",
        equivariance_rows_out,
        fieldnames=(
            "shape",
            "compressor",
            "parameter",
            "strength",
            "family",
            "n_before",
            "n_after",
            "n_after_transformed",
            "compression_ratio",
            "coverage_radius",
            "hausdorff_original_sketch",
            "sketch_is_subset",
            "verdict",
            "discrepancy",
            "note",
        ),
        seed=args.seed,
        config={"quick": args.quick, "shapes": sorted(shapes()), "families": ["translation", "rotation", "similarity", "affine"]},
    )
    rec_target, rec_meta = write_csv(
        "phase5_recognition",
        recognition_rows,
        fieldnames=(
            "shape",
            "compressor",
            "parameter",
            "strength",
            "family",
            "n_original",
            "n_compressed",
            "compression_ratio",
            "code_entries_before",
            "code_entries_after",
            "code_size_ratio",
            "code_size_speedup",
            "recognition_status",
            "reason",
            "residual",
            "compression_ms",
            "note",
        ),
        seed=args.seed,
        config={"quick": args.quick},
    )

    # -- summaries -------------------------------------------------------- #
    verdict_counts: dict[str, dict[str, int]] = {}
    for row in equivariance_rows_out:
        key = f"{row['compressor']}|{row['family']}"
        verdict_counts.setdefault(key, {})
        verdict_counts[key][str(row["verdict"])] = verdict_counts[key].get(str(row["verdict"]), 0) + 1

    reason_counts: dict[str, int] = {}
    for row in recognition_rows:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1

    print("\n  equivariance verdicts (compressor | family → counts)")
    for key in sorted(verdict_counts):
        print(f"    {key:<40} {verdict_counts[key]}")
    print("\n  recognition outcomes (5F taxonomy)")
    for key in sorted(reason_counts):
        print(f"    {key:<36} {reason_counts[key]}")
    print(f"\n  rows: equivariance={len(equivariance_rows_out)} recognition={len(recognition_rows)}")
    print(f"  elapsed: {elapsed:.1f}s")

    summary_target, summary_meta = write_json(
        "phase5_compression_summary",
        {
            "experiment": "phase5_compression",
            "title": "equivariance and recognition preservation under compression",
            "verdicts": verdict_counts,
            "reasons": reason_counts,
            "contracts": list_compressors(),
            "elapsed_seconds": elapsed,
            "note": (
                "Euclidean-distance-based compressors are not expected to be affine-equivariant; "
                "the measurement is the result. STRUCTURE_LOST is kept apart from a genuine "
                "recognition failure."
            ),
        },
        seed=args.seed,
        config={"quick": args.quick},
    )
    print(f"\n  [written] {target}")
    print(f"  [written] {meta}")
    print(f"  [written] {rec_target}")
    print(f"  [written] {rec_meta}")
    print(f"  [written] {summary_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
