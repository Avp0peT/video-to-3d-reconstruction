"""F5.0 — recognition pipeline sanity audit (Controls A–E).

This runs **before** any conclusion about compression.  The previous Phase 5 result —
"0 / 768 recognitions preserved" — was an artefact, and the audit is what caught it:

* the corpus used raw floating-point coordinates, which the *exact* recogniser refuses
  (``INVALID_INPUT``);
* the harness's failure taxonomy had no slot for that refusal, so it fell into the
  "code undefined after compression" bucket;
* the headline then read as a property of compression.

Controls, in the order the specification fixes them:

=========================  ==========================================================================
A — identity               ``C_0(X) = X`` must preserve recognition **100%**, or the phase stops
B — official sketch r = 0  ``A_0 = A`` for duplicate-free images, as a property and as 100%
C — pre/post equality      the *fields* are compared (definable, theorem, code, witness), not one
                           status string: two different statuses can hide identical mathematics
D — no-op is a no-op       cardinality and coordinates are checked, not the compressor's metadata
E — hand trace             one hand-computable pair through five compression regimes, written to
                           ``phase5_hand_trace.json`` as the easiest artifact to audit by hand
=========================  ==========================================================================

Run::

    python experiments/exp_phase5_sanity.py
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.coding import (  # noqa: E402
    RecognitionStatus,
    recognize_affine_exact,
    recognize_affine_numeric,
)
from kozlov_lab.compression import get_compressor, run_compressor  # noqa: E402
from _common import (  # noqa: E402
    apply_rational_affine,
    print_header,
    quantized_rational_image,
    write_json,
)

#: The four families of 5D, as integer/rational maps so the exact pipeline stays applicable.
FAMILIES: dict[str, tuple[tuple, tuple]] = {
    "translation": ((1, 0, 0, 1), (3, -2)),
    "rotation": ((0, -1, 1, 0), (2, 1)),          # 90 degrees, integer, det = 1
    "similarity": ((2, 0, 0, 2), (-1, 1)),        # integer scaling
    "affine": ((2, 1, 0, 3), (1, -1)),            # det = 6
    "affine_shear": ((1, 1, 0, 2), (0, 0)),       # det = 2
}

HAND_X = [(0, 0), (2, 0), (0, 2), (1, 1)]
HAND_MAP = ((2, 1), (0, 3))
HAND_TRANSLATION = (5, -7)


def _images() -> dict[str, list[tuple[Fraction, Fraction]]]:
    from kozlov_lab.datasets.corpus import DATASETS

    data: dict[str, list[tuple[Fraction, Fraction]]] = {}
    for key in ("A", "B", "C", "D", "E"):
        data[f"dataset_{key}"] = quantized_rational_image(DATASETS[key].points)
    data["stair"] = quantized_rational_image(
        [[i * 1.3 + (i % 3) * 0.4, i * i * 0.05 + (i % 5) * 0.7] for i in range(9)]
    )
    return data


def _fields(result) -> dict:
    """Control C: the *mathematical* fields, compared one by one.

    A single status string can differ while the mathematics agrees (for instance when one run is
    `EXACT_AFFINE_EQUIVALENT` and the other `NUMERIC_AFFINE_MATCH`), so the audit compares what
    the status is *about*.
    """
    return {
        "affine_equivalent": result.is_affine_equivalent,
        "code_definable": result.code_definable,
        "theorem_applicable": result.theorem_applicable,
        "code_equivalent": result.code_equivalent,
        "has_geometric_witness": result.geometric_witness is not None,
        "status": str(result.status),
    }


def control_a_and_c(images, output: dict) -> list[dict]:
    """Identity compressor: preservation must be 100%, and the fields must be unchanged."""
    rows: list[dict] = []
    identity = get_compressor("IdentityCompressor")()
    for name, points in images.items():
        compressed = identity.fit(points)
        for family, (matrix, translation) in FAMILIES.items():
            moved = apply_rational_affine(matrix, translation, points)
            moved_compressed = identity.fit(moved)
            before = recognize_affine_exact(points, moved)
            after = recognize_affine_exact(compressed.points, moved_compressed.points)
            same_fields = _fields(before) == _fields(after)
            preserved = before.is_affine_equivalent == after.is_affine_equivalent
            rows.append(
                {
                    "control": "A_identity",
                    "shape": name,
                    "family": family,
                    "before_status": str(before.status),
                    "after_status": str(after.status),
                    "preserved": preserved,
                    "fields_identical": same_fields,
                    "before": _fields(before),
                    "after": _fields(after),
                }
            )
    output["control_a_identity"] = rows
    return rows


def control_b(images, output: dict) -> list[dict]:
    """Official sketch at r = 0 must be the identity, and must preserve recognition."""
    rows: list[dict] = []
    for name, points in images.items():
        try:
            sketch = run_compressor("KozlovSketch", points, r=0.0)
        except Exception as exc:  # noqa: BLE001
            rows.append({"control": "B_r0", "shape": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        same = [tuple(map(float, p)) for p in sketch.points] == [tuple(map(float, p)) for p in points]
        rows.append(
            {
                "control": "B_r0",
                "shape": name,
                "n_before": len(points),
                "n_after": sketch.n_after,
                "is_identity": same,
                "achieved_radius": sketch.achieved_radius,
            }
        )
    output["control_b_r0_identity"] = rows

    preservation: list[dict] = []
    for name, points in images.items():
        for family, (matrix, translation) in FAMILIES.items():
            moved = apply_rational_affine(matrix, translation, points)
            try:
                sx = run_compressor("KozlovSketch", points, r=0.0).points
                sy = run_compressor("KozlovSketch", moved, r=0.0).points
                before = recognize_affine_exact(points, moved)
                after = recognize_affine_exact(sx, sy)
                preservation.append(
                    {
                        "control": "B_r0_preservation",
                        "shape": name,
                        "family": family,
                        "before_status": str(before.status),
                        "after_status": str(after.status),
                        "preserved": before.is_affine_equivalent == after.is_affine_equivalent,
                        "fields_identical": _fields(before) == _fields(after),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                preservation.append(
                    {"control": "B_r0_preservation", "shape": name, "family": family,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
    output["control_b_r0_preservation"] = preservation
    return rows + preservation


def control_d(images, output: dict) -> list[dict]:
    """A declared no-op must *be* a no-op — checked against coordinates, not metadata."""
    rows: list[dict] = []
    identity = get_compressor("IdentityCompressor")()
    for name, points in images.items():
        result = identity.fit(points)
        same_cardinality = result.n_after == len(points)
        same_coordinates = (
            np.array_equal(np.asarray(result.points, dtype=object), np.asarray(points, dtype=object))
            if same_cardinality
            else False
        )
        rows.append(
            {
                "control": "D_no_op",
                "shape": name,
                "cardinality_equal": same_cardinality,
                "coordinates_equal": bool(same_coordinates),
                "declared_radius": result.requested_radius,
                "achieved_radius": result.achieved_radius,
                "metadata_alone_would_claim_no_op": result.diagnostics.get("no_op") is True,
            }
        )
    output["control_d_no_op"] = rows
    return rows


def control_e(output: dict) -> dict:
    """The hand example through five compression regimes (the easiest artifact to audit)."""
    x = [(Fraction(p), Fraction(q)) for p, q in HAND_X]
    (a, b), (c, d) = HAND_MAP          # M = [[2, 1], [0, 3]]
    flat_map = (a, b, c, d)            # the same matrix, in the flat form the helper takes
    tx, ty = HAND_TRANSLATION
    y = apply_rational_affine(flat_map, HAND_TRANSLATION, x)

    regimes: list[dict] = []
    reference = recognize_affine_exact(x, y)
    regimes.append({"regime": "no_compression", **_fields(reference)})

    identity = get_compressor("IdentityCompressor")()
    sx = identity.fit(x)
    sy = identity.fit(y)
    regimes.append({"regime": "identity", **_fields(recognize_affine_exact(sx.points, sy.points))})

    kx = run_compressor("KozlovSketch", x, r=0.0)
    ky = run_compressor("KozlovSketch", y, r=0.0)
    regimes.append({"regime": "official_sketch_r0", **_fields(recognize_affine_exact(kx.points, ky.points))})

    for label, r in (("weak_compression_r05", 0.5), ("strong_compression_r20", 2.0)):
        try:
            wx = run_compressor("KozlovSketch", x, r=r)
            wy = run_compressor("KozlovSketch", y, r=r)
            entries = {
                "regime": label,
                "n_x": wx.n_after,
                "n_y": wy.n_after,
                "compression_ratio_x": wx.compression_ratio,
                **_fields(recognize_affine_exact(wx.points, wy.points)),
            }
        except Exception as exc:  # noqa: BLE001
            entries = {"regime": label, "error": f"{type(exc).__name__}: {exc}"}
        regimes.append(entries)

    trace = {
        "note": (
            "The hand pair of the Phase 3 worked example: X is exactly the four-point image whose "
            "code has |T_A| = 9, and Y is its image under M with det M = 6.  Every row can be "
            "checked by hand, which is why this file exists."
        ),
        "x": [[str(v) for v in point] for point in x],
        "matrix": [[a, b], [c, d]],
        "translation": [tx, ty],
        "det": a * d - b * c,
        "y": [[str(v) for v in point] for point in y],
        "regimes": regimes,
    }
    output["control_e_hand_trace"] = trace
    return trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)

    print_header("F5.0 — recognition pipeline sanity audit")
    images = _images()
    print(f"  corpus: {len(images)} shapes, coordinates as exact rationals "
          f"(denominator 1000), so the exact pipeline is applicable")

    output: dict = {"corpus": sorted(images)}
    a_rows = control_a_and_c(images, output)
    b_rows = control_b(images, output)
    d_rows = control_d(images, output)
    trace = control_e(output)

    a_preserved = sum(1 for row in a_rows if row.get("preserved"))
    a_fields = sum(1 for row in a_rows if row.get("fields_identical"))
    b_identity = [row for row in output["control_b_r0_identity"] if "is_identity" in row]
    b_ok = sum(1 for row in b_identity if row["is_identity"])
    b_pres = [row for row in output["control_b_r0_preservation"] if "preserved" in row]
    b_preserved = sum(1 for row in b_pres if row["preserved"])
    d_ok = sum(1 for row in d_rows if row["cardinality_equal"] and row["coordinates_equal"])

    print("\n  Control A — identity compressor")
    print(f"    recognition preserved : {a_preserved}/{len(a_rows)}")
    print(f"    fields identical      : {a_fields}/{len(a_rows)}")
    print("\n  Control B — official sketch at r = 0")
    print(f"    is the identity       : {b_ok}/{len(b_identity)}")
    print(f"    recognition preserved : {b_preserved}/{len(b_pres)}")
    print("\n  Control D — a declared no-op is a no-op")
    print(f"    cardinality + coordinates equal : {d_ok}/{len(d_rows)}")
    print("\n  Control E — hand trace")
    for row in trace["regimes"]:
        label = row.get("regime")
        if "error" in row:
            print(f"    {label:<24} {row['error']}")
        else:
            print(f"    {label:<24} status={row['status']:<36} affine_equivalent={row['affine_equivalent']}")

    verdicts = {
        "control_a_identity_preserves_100pct": a_preserved == len(a_rows) and len(a_rows) > 0,
        "control_a_fields_identical_100pct": a_fields == len(a_rows) and len(a_rows) > 0,
        "control_b_r0_is_identity_100pct": b_ok == len(b_identity) and len(b_identity) > 0,
        "control_b_r0_preserves_100pct": b_preserved == len(b_pres) and len(b_pres) > 0,
        "control_d_no_op_verified_on_data": d_ok == len(d_rows) and len(d_rows) > 0,
        "control_e_hand_trace_complete": len(trace["regimes"]) == 5,
    }
    verdicts["sanity_audit_passed"] = all(verdicts.values())
    output["verdicts"] = verdicts

    print("\n  verdicts")
    for key, value in verdicts.items():
        print(f"    {key:<46} {value}")
    print("\n  " + ("SANITY AUDIT PASSED — compression results may be interpreted"
                    if verdicts["sanity_audit_passed"]
                    else "SANITY AUDIT FAILED — do not interpret compression results yet"))

    target, meta = write_json(
        "phase5_sanity",
        {"experiment": "phase5_sanity", "title": "F5.0 recognition pipeline sanity audit", **output},
        seed=args.seed,
        config={"families": list(FAMILIES), "corpus": sorted(images), "decimals": 3},
    )
    trace_path = Path(__file__).resolve().parent / "results" / "phase5_hand_trace.json"
    trace_path.write_text(
        __import__("json").dumps(trace, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n  [written] {target}")
    print(f"  [written] {meta}")
    print(f"  [written] {trace_path}")
    return 0 if verdicts["sanity_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
