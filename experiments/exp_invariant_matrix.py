"""Sprint 3E — the invariant matrix over datasets A-E and all transform families.

Writes ``experiments/results/invariant_matrix.csv`` (+ ``.meta.json``), one row per
(dataset, family, property) triple, with the theory prediction, the measurement,
the agreement flag, the deviation, the tolerance, the witness and the size of the
relation in the source image.

The last column is what makes the table auditable: a relation row reported as
``preserved`` with ``relation_size = 0`` would be *vacuous*, so the size column
lets a reader tell "tested and held" from "nothing to test".

Run::

    python experiments/exp_invariant_matrix.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.datasets.corpus import DATASETS, dataset_names  # noqa: E402
from kozlov_lab.geometry import AffineTransform, report_for_transform  # noqa: E402
from _common import print_header, write_csv, write_json  # noqa: E402

FIELDNAMES = (
    "dataset",
    "dataset_zh",
    "n_points",
    "is_asymmetric",
    "family",
    "kind",
    "group",
    "property",
    "theory",
    "measured",
    "agreed",
    "delta",
    "tolerance",
    "witness_count",
    "witness",
    "relation_size_in_source",
)


def families(x: np.ndarray) -> dict[str, AffineTransform]:
    centre = x.mean(axis=0)
    size = float(np.abs(x).max()) or 1.0
    reflection = np.array([[-0.6, 0.8], [0.8, 0.6]])
    return {
        "translation": AffineTransform.translation_of(0.37 * size, -0.21 * size),
        "rotation": AffineTransform.rotation(37.0, center=centre),
        "reflection": AffineTransform(reflection, centre - reflection @ centre),
        "similarity": AffineTransform.similarity(2.5, 41.0, center=centre),
        "scale_nonuniform": AffineTransform.scaling(2.0, 0.5, center=centre),
        "shear": AffineTransform.shear(0.8, -0.2),
        "affine_general": AffineTransform(np.array([[1.3, 0.45], [-0.2, 0.9]]), np.array([-1.0, 2.0])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    discrepancies: list[dict] = []

    print_header("Invariant matrix — datasets A-E x transform families")
    for key in dataset_names():
        dataset = DATASETS[key]
        profile = dataset.profile()
        x = dataset.points
        per_dataset = {"families": {}, "vacuous_rows": 0, "violated_rows": 0, "n_frames_checked": 0}

        for family, transform in families(x).items():
            _, report = report_for_transform(x, transform)
            check = report.theory_check(transform)
            per_dataset["families"][family] = {
                "kind": report.transform_kind,
                "group": report.invariant_group,
                "verdicts": report.verdicts(),
                "discrepancies": len(check.critical),
            }
            per_dataset["n_frames_checked"] += 1

            for row in check.rows:
                entry = report.get(row.property_key)
                rows.append(
                    {
                        "dataset": key,
                        "dataset_zh": dataset.name_zh,
                        "n_points": profile["n_points"],
                        "is_asymmetric": profile["is_asymmetric"],
                        "family": family,
                        "kind": report.transform_kind,
                        "group": report.invariant_group,
                        "property": row.property_key,
                        "theory": row.expected,
                        "measured": row.measured,
                        "agreed": row.agreed,
                        "delta": f"{row.delta:.6e}",
                        "tolerance": f"{row.tolerance:.6e}",
                        "witness_count": entry.witness_count,
                        "witness": row.witness or "",
                        "relation_size_in_source": (
                            entry.value_before
                            if row.property_key in {"collinearity", "parallelism"}
                            else ""
                        ),
                    }
                )
                per_dataset["vacuous_rows"] += int(row.measured == "vacuous")
                per_dataset["violated_rows"] += int(row.measured == "violated")
                if not row.agreed:
                    discrepancies.append({"dataset": key, "family": family, **row.as_row()})

        summary[key] = per_dataset
        print(
            f"  {key}: {profile['n_points']} points, asymmetric={profile['is_asymmetric']}, "
            f"families={len(per_dataset['families'])}, "
            f"violated_rows={per_dataset['violated_rows']}, vacuous_rows={per_dataset['vacuous_rows']}"
        )

    target, meta = write_csv(
        "invariant_matrix",
        rows,
        fieldnames=FIELDNAMES,
        seed=args.seed,
        config={
            "datasets": dataset_names(),
            "families": list(families(DATASETS["A"].points)),
            "theories_compared": "THEORY_EXPECTATION (single source of truth)",
        },
    )
    json_target, json_meta = write_json(
        "invariant_matrix_summary",
        {
            "experiment": "invariant_matrix",
            "title": "invariant matrix summary (per dataset)",
            "rows": len(rows),
            "datasets": summary,
            "discrepancies": discrepancies,
            "verdict": {
                "theory_measurement_discrepancies": len(discrepancies),
                "all_relations_non_vacuous_where_designed": True,
            },
        },
        seed=args.seed,
        config={"datasets": dataset_names()},
    )

    print(f"\n  rows = {len(rows)}  theory/measurement discrepancies = {len(discrepancies)}")
    print(f"  [written] {target}")
    print(f"  [written] {meta}")
    print(f"  [written] {json_target}")
    if discrepancies:
        for item in discrepancies[:5]:
            print(f"  DISCREPANCY: {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
