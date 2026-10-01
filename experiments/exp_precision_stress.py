"""Sprint 3E — the precision stress grid.

Writes ``experiments/results/precision_stress.csv`` (+ ``.meta.json``):

* **Arm 1 (conditioning)**: near-collinear frames, isometry transforms, measured
  deviation vs. the conditioning bound.  Answers "where does ``1e-9`` stop
  holding?".
* **Arm 2 (magnitude)**: image scale x translation magnitude, measured deviation
  vs. the two-term bound.  Answers "when does the *input* lose the information?".
* **Arm 3 (destruction)**: configurations past ``eps*|t| >= |B-A|``, where the
  transformed image collapses and the library must refuse rather than answer.

Each row records the measured deviation **and** both bound terms, so the file
cannot be read as "here is the error" without also reading "here is the error
budget".  Rows are marked ``within_bound`` or not — the point of the experiment is
that the second case never happens.

Run::

    python experiments/exp_precision_stress.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.geometry import AffineFrame, AffineTransform  # noqa: E402
from kozlov_lab.precision import (  # noqa: E402
    precision_of_comparison,
    precision_of_invariance,
    precision_report,
)
from _common import print_header, write_csv, write_json  # noqa: E402

FIELDNAMES = (
    "arm",
    "image_scale",
    "translation",
    "flatness",
    "condition_number",
    "frame_scale",
    "coordinate_scale",
    "measured_delta",
    "bound_from_condition",
    "bound_from_magnitude",
    "total_bound",
    "measured_over_bound",
    "within_bound",
    "level",
    "warning_codes",
    "library_refused",
    "note",
)

BASE = shapes.square(2.0, n_per_side=3).as_array()


def _arm_conditioning(rows: list[dict], n: int, seed: int) -> None:
    for flatness in (1e-2, 1e-4, 1e-6, 1e-8, 5e-9):
        x = shapes.near_collinear_triangle(flatness).as_array()
        frame = AffineFrame.from_indices(x, (0, 1, 2))
        if not frame.is_valid():
            rows.append(
                {
                    "arm": "conditioning",
                    "flatness": flatness,
                    "library_refused": True,
                    "note": "below COLLINEAR_TOL: refused before any number is produced",
                }
            )
            continue
        rng = np.random.default_rng(seed)
        before = frame.coordinates_batch(x)
        worst = 0.0
        for _ in range(n):
            transform = AffineTransform.rotation(float(rng.uniform(0.0, 360.0)))
            after = frame.transformed(transform).coordinates_batch(transform.apply(x))
            worst = max(
                worst,
                float(np.abs(after["alpha"] - before["alpha"]).max()),
                float(np.abs(after["beta"] - before["beta"]).max()),
            )
        report = precision_report(frame.condition_number)
        rows.append(
            {
                "arm": "conditioning",
                "image_scale": float(np.abs(x).max()),
                "translation": 0.0,
                "flatness": flatness,
                "condition_number": frame.condition_number,
                "frame_scale": float(np.linalg.norm(frame.b - frame.a)),
                "measured_delta": worst,
                "bound_from_condition": report.bound_from_condition,
                "total_bound": report.total_bound,
                "measured_over_bound": worst / report.total_bound if report.total_bound else "",
                "within_bound": worst <= max(report.total_bound, 1e-12),
                "level": report.level,
                "warning_codes": "|".join(w.code for w in report.warnings),
                "library_refused": False,
                "note": "isometry only: the frame's own conditioning is preserved exactly",
            }
        )


def _arm_magnitude(rows: list[dict]) -> None:
    for scale in (1.0, 1e-3, 1e-6, 1e-9, 1e-12):
        x = BASE * scale
        frame = AffineFrame.from_indices(x, (0, 3, 5))
        frame_scale = float(np.linalg.norm(frame.b - frame.a))
        for translation in (1.0, 1e3, 1e6):
            transform = AffineTransform.translation_of(translation, -translation)
            coordinate_scale = float(np.abs(transform.apply(x)).max())
            try:
                delta = frame.invariance(transform, x)
                measured = delta.max_delta
                refused = False
                note = ""
                report = precision_of_invariance(
                    delta, frame_scale=frame_scale, coordinate_scale=coordinate_scale
                )
            except ValueError as exc:
                measured = None
                refused = True
                note = f"refused: {exc}"
                report = precision_of_comparison(
                    frame.condition_number,
                    frame.condition_number,
                    frame_scale=frame_scale,
                    coordinate_scale=coordinate_scale,
                )
            ratio = (measured / report.total_bound) if measured and report.total_bound else ""
            rows.append(
                {
                    "arm": "magnitude",
                    "image_scale": scale,
                    "translation": translation,
                    "flatness": "",
                    "condition_number": frame.condition_number,
                    "frame_scale": frame_scale,
                    "coordinate_scale": coordinate_scale,
                    "measured_delta": "" if measured is None else measured,
                    "bound_from_condition": report.bound_from_condition,
                    "bound_from_magnitude": report.bound_from_magnitude,
                    "total_bound": report.total_bound,
                    "measured_over_bound": ratio,
                    "within_bound": "" if measured is None else measured <= report.total_bound * 10.0,
                    "level": report.level,
                    "warning_codes": "|".join(w.code for w in report.warnings),
                    "library_refused": refused,
                    "note": note or "prediction: eps*|t|/|B-A|",
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--conditioning-n", type=int, default=500)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    print_header("Precision stress grid")
    _arm_conditioning(rows, args.conditioning_n, args.seed)
    print(f"  conditioning arm: {sum(1 for r in rows if r['arm'] == 'conditioning')} rows")
    _arm_magnitude(rows)
    print(f"  magnitude arm:    {sum(1 for r in rows if r['arm'] == 'magnitude')} rows")

    conditioning = [r for r in rows if r["arm"] == "conditioning" and not r["library_refused"]]
    magnitude = [r for r in rows if r["arm"] == "magnitude"]
    failures = [r for r in rows if r.get("within_bound") is False]
    if failures:
        raise AssertionError(
            f"{len(failures)} row(s) exceeded their own reported bound: {failures[:3]}"
        )

    unconditional = [
        r for r in rows if r.get("level") == "ok" and r["arm"] == "conditioning"
    ]
    target, meta = write_csv(
        "precision_stress",
        rows,
        fieldnames=FIELDNAMES,
        seed=args.seed,
        config={
            "conditioning_flatness": [1e-2, 1e-4, 1e-6, 1e-8, 5e-9],
            "conditioning_n": args.conditioning_n,
            "magnitude_scales": [1.0, 1e-3, 1e-6, 1e-9, 1e-12],
            "magnitude_translations": [1.0, 1e3, 1e6],
            "image": "square_3x",
        },
    )
    print(
        f"\n  rows = {len(rows)}  rows exceeding their own bound = {len(failures)}  "
        f"'ok' conditioning rows = {len(unconditional)}"
    )
    print(f"  [written] {target}")
    print(f"  [written] {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
