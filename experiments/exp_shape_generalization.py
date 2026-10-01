"""Sprint 3E — does the teaching affine code generalise across shapes?

Writes ``experiments/results/shape_generalization.csv`` (+ ``.meta.json``).

For every dataset A-E and every transform family, the experiment repeats the
comparison ``C_teach(X) == C_teach(T(X))`` ``--repetitions`` times (default 3,
following the project convention that no conclusion rests on a single run) and
records:

* the verdict, the frame counts, the number of points;
* whether the chosen quantisation was safe for the image's worst frame;
* a **control** in which the image is replaced by an unrelated point set of the
  same cardinality — which must *not* be equivalent.  Without the control a
  constant code would score 100%.

Run::

    python experiments/exp_shape_generalization.py --repetitions 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.datasets.corpus import DATASETS, dataset_names  # noqa: E402
from kozlov_lab.geometry import AffineTransform  # noqa: E402
from kozlov_lab.teaching import (  # noqa: E402
    TeachingCodingParameters,
    build_teaching_image_code,
    compare_teaching_codes,
    recommended_decimals,
)
from _common import print_header, write_csv, write_json  # noqa: E402

FIELDNAMES = (
    "dataset",
    "dataset_zh",
    "n_points",
    "is_asymmetric",
    "family",
    "kind",
    "repetition",
    "case",
    "equivalent",
    "weak_evidence",
    "reason",
    "n_frames_x",
    "n_frames_y",
    "n_collinear_triples",
    "decimals",
    "recommended_decimals",
    "quantization_is_safe",
    "worst_condition_number",
    "digest_x",
    "digest_y",
    "max_condition_budget",
)

#: Fixed budget so the code's determinism (not luck) is what is being measured.
MAX_CONDITION = 20.0

#: Families whose codes are expected to match.  Includes the `random` family so the
#: test is not limited to hand-picked matrices.
FAMILIES = (
    "translation",
    "rotation",
    "reflection",
    "similarity",
    "scale_nonuniform",
    "shear",
    "affine_general",
    "random",
)


def _transforms(x: np.ndarray, seed: int) -> dict[str, AffineTransform]:
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
        "random": AffineTransform.random_invertible(rng, max_condition=MAX_CONDITION),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args(argv)

    params = TeachingCodingParameters(decimals=args.decimals)
    rows: list[dict] = []
    per_dataset: dict[str, dict] = {}
    failures: list[str] = []

    print_header("Shape generalization — teaching affine code")
    for key in dataset_names():
        dataset = DATASETS[key]
        profile = dataset.profile()
        x = dataset.points
        feasible = recommended_decimals(x)
        reference = build_teaching_image_code(x, params)
        reference_quality = reference.quality()
        stats = {"families": 0, "equivalent": 0, "control_not_equivalent": 0}

        for family, transform in _transforms(x, args.seed).items():
            stats["families"] += 1
            for repetition in range(args.repetitions):
                # repetitions differ by an extra jitter so they are independent draws
                jitter = AffineTransform.translation_of(
                    float(repetition) * 1e-3 * (1 if repetition % 2 else -1),
                    float(repetition) * 5e-4,
                )
                moved = (jitter @ transform).apply(x)
                kind = (jitter @ transform).classify()
                comparison = compare_teaching_codes(x, moved, params)
                rows.append(
                    {
                        "dataset": key,
                        "dataset_zh": dataset.name_zh,
                        "n_points": profile["n_points"],
                        "is_asymmetric": profile["is_asymmetric"],
                        "family": family,
                        "kind": kind,
                        "repetition": repetition,
                        "case": "affine_image",
                        "equivalent": comparison.equivalent,
                        "weak_evidence": comparison.weak_evidence,
                        "reason": comparison.reason,
                        "n_frames_x": comparison.n_frames_x,
                        "n_frames_y": comparison.n_frames_y,
                        "n_collinear_triples": reference.n_collinear_triples,
                        "decimals": params.decimals,
                        "recommended_decimals": feasible,
                        "quantization_is_safe": reference_quality["quantization_is_safe"],
                        "worst_condition_number": reference.worst_condition_number,
                        "digest_x": comparison.digest_x[:16],
                        "digest_y": comparison.digest_y[:16],
                        "max_condition_budget": MAX_CONDITION,
                    }
                )
                if comparison.equivalent:
                    stats["equivalent"] += 1
                else:
                    failures.append(f"{key}/{family}/rep{repetition}: {comparison.reason}")

        # control: an unrelated set of the same cardinality must NOT be equivalent
        # deterministic per-dataset seed: Python's str hash is randomised per process,
        # so using hash(key) would make this file irreproducible
        control_target = shapes.random_points(
            x.shape[0], seed=args.seed + ord(key)
        ).as_array()
        control = compare_teaching_codes(x, control_target, params)
        rows.append(
            {
                "dataset": key,
                "dataset_zh": dataset.name_zh,
                "n_points": profile["n_points"],
                "is_asymmetric": profile["is_asymmetric"],
                "family": "UNRELATED_CONTROL",
                "kind": "unrelated",
                "repetition": 0,
                "case": "control",
                "equivalent": control.equivalent,
                "weak_evidence": control.weak_evidence,
                "reason": control.reason,
                "n_frames_x": control.n_frames_x,
                "n_frames_y": control.n_frames_y,
                "n_collinear_triples": reference.n_collinear_triples,
                "decimals": params.decimals,
                "recommended_decimals": feasible,
                "quantization_is_safe": reference_quality["quantization_is_safe"],
                "worst_condition_number": reference.worst_condition_number,
                "digest_x": control.digest_x[:16],
                "digest_y": control.digest_y[:16],
                "max_condition_budget": MAX_CONDITION,
            }
        )
        if not control.equivalent:
            stats["control_not_equivalent"] += 1
        else:
            failures.append(f"{key}/control: unrelated image was reported equivalent")

        per_dataset[key] = {
            **stats,
            "n_points": profile["n_points"],
            "is_asymmetric": profile["is_asymmetric"],
            "n_frames": len(reference),
            "n_collinear_triples": reference.n_collinear_triples,
            "worst_condition_number": reference.worst_condition_number,
            "recommended_decimals": feasible,
            "decimals_used": params.decimals,
            "quantization_is_safe": reference_quality["quantization_is_safe"],
        }
        print(
            f"  {key}: frames={len(reference)} equivalent={stats['equivalent']}/{stats['families'] * args.repetitions} "
            f"control_not_equivalent={stats['control_not_equivalent']}/1 "
            f"recommended_decimals={feasible}"
        )

    target, meta = write_csv(
        "shape_generalization",
        rows,
        fieldnames=FIELDNAMES,
        seed=args.seed,
        config={
            "datasets": dataset_names(),
            "families": list(FAMILIES),
            "repetitions": args.repetitions,
            "decimals": args.decimals,
            "max_condition_budget": MAX_CONDITION,
        },
    )
    json_target, json_meta = write_json(
        "shape_generalization_summary",
        {
            "experiment": "shape_generalization",
            "title": "teaching affine code across datasets A-E",
            "datasets": per_dataset,
            "verdict": {
                "rows": len(rows),
                "failures": failures,
                "all_affine_images_equivalent": not any(
                    "rep" in f for f in failures
                ),
                "control_never_equivalent": not any("control" in f for f in failures),
            },
        },
        seed=args.seed,
        config={"repetitions": args.repetitions, "decimals": args.decimals},
    )

    print(f"\n  rows = {len(rows)}  failures = {len(failures)}")
    print(f"  [written] {target}")
    print(f"  [written] {meta}")
    print(f"  [written] {json_target}")
    for failure in failures[:5]:
        print(f"  FAILURE: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
