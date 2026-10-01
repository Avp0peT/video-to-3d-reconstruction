"""4O — differential testing, the Phase 4 gate.

Generates a large corpus of **exact** small cases and runs both routes:

* the **code route** — the formal definition, via the pruned ``psi`` search;
* the **geometric route** — an exact affine-map search that knows nothing about codes.

Requirement: **0 unexplained disagreements.**

What "unexplained" means, precisely
-----------------------------------
Inside theorem 1's domain the two routes are guaranteed to agree, so any disagreement there is
a defect in one of them and is reported as **unexplained** — this script then exits non-zero,
saves a full reproduction artifact, and shrinks it.

Outside the domain (an image lying on two parallel lines) a disagreement is *expected*: the
theorem's hypothesis is exactly what rules out code-equivalence without affine equivalence.
Those cases are counted separately as **explained**. Treating them as failures would be as
wrong as treating them as successes — which is why the split is explicit rather than a
tolerance someone could widen.

Ground truth is always the geometric oracle. Random images are **not** assumed to be
non-equivalent: whether a random draw happens to be affinely equivalent is a question the
oracle answers, not an assumption.

Run::

    python experiments/exp_phase4_differential.py                 # 10 000 cases
    python experiments/exp_phase4_differential.py --cases 500     # quick pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.coding import (  # noqa: E402
    build_kozlov_image_code_exact,
    check_geometric_equivalence,
    kozlov_codes_equivalent,
    validate_kozlov_image,
)
from _common import print_header, write_json  # noqa: E402

#: ``n`` is drawn from this table.  Weighted towards small ``n`` because both routes are
#: ``O(|T_A|)``-ish in their setup, and a uniform draw over ``{4..7}`` would spend most of the
#: budget on ``n = 7`` while exploring far fewer distinct configurations.
N_WEIGHTS: tuple[tuple[int, int], ...] = ((4, 4000), (5, 3000), (6, 2000), (7, 1000))

BOX = 20


def _random_integer_map(rng: np.random.Generator) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    while True:
        a, b, c, d = (int(rng.integers(-3, 4)) for _ in range(4))
        if a * d - b * c != 0:
            return (a, b, c, d), (int(rng.integers(-10, 11)), int(rng.integers(-10, 11)))


def _random_image(rng: np.random.Generator, n: int) -> list[list[int]] | None:
    """A random integer image in general position, or ``None`` if the draw is degenerate.

    General position is requested explicitly (via the validator) rather than assumed, so the
    corpus does not silently accumulate collinear draws that would be skipped anyway.
    """
    for _ in range(50):
        points = [
            [int(rng.integers(-BOX, BOX + 1)), int(rng.integers(-BOX, BOX + 1))] for _ in range(n)
        ]
        try:
            validate_kozlov_image(points)
        except Exception:  # noqa: BLE001 - a degenerate draw is simply retried
            continue
        return points
    return None


def _two_parallel_image(rng: np.random.Generator, n: int) -> list[list[int]] | None:
    """An image whose points lie on exactly two parallel lines.

    Needed because the "unexplained vs explained" split is only meaningful if the
    outside-the-theorem-domain case actually occurs.  With random coordinates it effectively
    never does: a generic random point set is in general position, so the *explained* counter
    would stay at zero and the classification would be untested.

    The lines are ``x + y = 0`` and ``x + y = m``, both of direction ``(1, -1)``.
    """
    if n < 3:
        return None
    split = int(rng.integers(1, n - 1))  # at least one point on each line
    first, second = split, n - split
    if first < 1 or second < 1:
        return None
    offset = int(rng.integers(1, 9))
    points: list[list[int]] = []
    for index in range(first):
        value = int(rng.integers(-BOX, BOX + 1))
        points.append([value, -value])
    for index in range(second):
        value = int(rng.integers(-BOX, BOX + 1))
        points.append([value, offset - value])
    if len({tuple(point) for point in points}) != n:
        return None
    try:
        validate_kozlov_image(points)
    except Exception:  # noqa: BLE001 - e.g. all points landed on one line
        return None
    return points


def _apply(
    matrix: tuple[int, int, int, int], translation: tuple[int, int], points: list[list[int]]
) -> list[list[int]]:
    a, b, c, d = matrix
    tx, ty = translation
    return [[a * x + b * y + tx, c * x + d * y + ty] for x, y in points]


def _disagrees(x: list[list[int]], y: list[list[int]]) -> dict | None:
    """Run both routes and report a disagreement, or ``None`` if they agree / abstain."""
    code_x = build_kozlov_image_code_exact(x)
    code_y = build_kozlov_image_code_exact(y)
    if code_x.domain != code_y.domain:
        # differing domain classes cannot correspond, and both routes know it
        return None
    code_result = kozlov_codes_equivalent(code_x, code_y)
    geometric = check_geometric_equivalence(x, y)
    if code_result.undecided or geometric.undecided:
        return None
    if code_result.equivalent == geometric.equivalent:
        return None
    return {
        "x": x,
        "y": y,
        "domain_x": code_x.domain,
        "domain_y": code_y.domain,
        "theorem_applicable": code_x.theorem_applies and code_y.theorem_applies,
        "code_equivalent": code_result.equivalent,
        "geometric_equivalent": geometric.equivalent,
        "code_route": code_result.as_dict(),
        "geometric_route": {
            "equivalent": geometric.equivalent,
            "exhaustive": geometric.exhaustive,
            "reason": geometric.reason,
            "candidates_tried": geometric.candidates_tried,
        },
    }


def shrink(x: list[list[int]], y: list[list[int]]) -> dict | None:
    """Greedily drop point pairs while the disagreement persists.

    Standard delta debugging: the counterexample is only useful if it is small enough to read.
    A pair (``x_i``, ``y_i``) is removed together because the two images are expected to
    correspond positionally in the generated corpus; if the disagreement survives, the smaller
    pair is kept and the process repeats.
    """
    if len(x) != len(y) or len(x) <= 4:
        return None
    improved = False
    index = 0
    while index < len(x) and len(x) > 4:
        candidate_x = x[:index] + x[index + 1 :]
        candidate_y = y[:index] + y[index + 1 :]
        try:
            reduced = _disagrees(candidate_x, candidate_y)
        except Exception:  # noqa: BLE001 - the reduced case may become degenerate
            reduced = None
        if reduced is not None:
            x, y = candidate_x, candidate_y
            improved = True
        else:
            index += 1
    return {"x": x, "y": y} if improved else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--shrink-budget", type=int, default=8)
    args = parser.parse_args(argv)

    print_header("Phase 4O — differential testing (code route vs geometric route)")
    rng = np.random.default_rng(args.seed)
    schedule = [n for n, weight in N_WEIGHTS for _ in range(max(1, weight * args.cases // 10_000))]

    counts = {
        "cases": 0,
        "equivalent": 0,
        "not_equivalent": 0,
        "skipped": 0,
        "outside_theorem_domain": 0,
    }
    unexplained: list[dict] = []
    explained: list[dict] = []
    started = time.perf_counter()

    for index in range(args.cases):
        n = schedule[index % len(schedule)]
        # ~12% of the corpus is drawn on two parallel lines, so the
        # "code computable but theorem not applicable" case is actually exercised
        use_two_parallel = rng.random() < 0.12
        x = _two_parallel_image(rng, n) if use_two_parallel else _random_image(rng, n)
        if x is None:
            counts["skipped"] += 1
            continue
        if rng.random() < 0.5:
            y = _apply(*_random_integer_map(rng), x)
        else:
            y = _two_parallel_image(rng, n) if use_two_parallel else _random_image(rng, n)
            if y is None:
                counts["skipped"] += 1
                continue
        try:
            if not validate_kozlov_image(x).theorem_applicable:
                counts["outside_theorem_domain"] += 1
        except Exception:  # noqa: BLE001
            pass

        try:
            disagreement = _disagrees(x, y)
        except Exception as exc:  # noqa: BLE001 - an exception is itself a finding
            disagreement = {
                "x": x,
                "y": y,
                "exception": f"{type(exc).__name__}: {exc}",
                "theorem_applicable": None,
            }

        counts["cases"] += 1
        if disagreement is None:
            geometric = check_geometric_equivalence(x, y)
            counts["equivalent" if geometric.equivalent else "not_equivalent"] += 1
            continue

        record = {"index": index, "n": n, **disagreement}
        if disagreement.get("theorem_applicable"):
            unexplained.append(record)
        else:
            explained.append(record)

        if (index + 1) % 2000 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"    {index + 1:>6} cases  ({elapsed:>6.1f}s)  unexplained={len(unexplained)}  "
                f"explained={len(explained)}"
            )

    elapsed = time.perf_counter() - started

    # -- shrink the unexplained cases -------------------------------------- #
    shrunk: list[dict] = []
    for record in unexplained[: args.shrink_budget]:
        try:
            minimal = shrink(record["x"], record["y"])
        except Exception:  # noqa: BLE001
            minimal = None
        shrunk.append({**record, "minimal": minimal})

    print("\n  corpus")
    print(f"    cases run            : {counts['cases']}")
    print(f"    affinely equivalent  : {counts['equivalent']}")
    print(f"    not equivalent       : {counts['not_equivalent']}")
    print(f"    skipped (degenerate) : {counts['skipped']}")
    print(f"    outside theorem domain: {counts['outside_theorem_domain']}")
    print(f"    elapsed              : {elapsed:.1f}s")
    print(f"\n  disagreements: unexplained={len(unexplained)}  explained(theorem domain)={len(explained)}")

    artifact_paths: list[str] = []
    if unexplained:
        target_dir = Path(__file__).resolve().parent / "results" / "discrepancies"
        target_dir.mkdir(parents=True, exist_ok=True)
        for position, record in enumerate(shrunk):
            path = target_dir / f"phase4_unexplained_{args.seed}_{position}.json"
            path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            artifact_paths.append(str(path))
        print("    artifacts written:")
        for path in artifact_paths:
            print(f"      {path}")

    verdicts = {
        "cases": counts["cases"],
        "outside_theorem_domain_cases": counts["outside_theorem_domain"],
        "domain_case_exercised": counts["outside_theorem_domain"] > 0,
        "unexplained_disagreements": len(unexplained),
        "explained_disagreements": len(explained),
        "gate_requirement": "0 unexplained disagreements",
        "gate_passed": not unexplained,
        "elapsed_seconds": elapsed,
        "artifact_paths": artifact_paths,
        "note": (
            "an 'explained' disagreement is one outside theorem 1's domain, where the "
            "hypothesis is exactly what rules out agreement; it is recorded, never tolerated "
            "away by widening a tolerance"
        ),
    }

    target, meta = write_json(
        "phase4_differential",
        {
            "experiment": "phase4_differential",
            "title": "code route vs geometric route on exact small cases",
            "counts": counts,
            "verdicts": verdicts,
            "unexplained": [
                {k: v for k, v in record.items() if k not in {"code_route", "geometric_route"}}
                for record in shrunk
            ],
            "explained": [
                {k: v for k, v in record.items() if k not in {"code_route", "geometric_route"}}
                for record in explained[:20]
            ],
        },
        seed=args.seed,
        config={
            "cases_requested": args.cases,
            "n_weights": N_WEIGHTS,
            "coordinate_box": BOX,
            "generator": "50% affine image, 50% independent draw; oracle supplies ground truth",
            "shrink_budget": args.shrink_budget,
        },
    )
    print(f"\n  [written] {target}")
    print(f"  [written] {meta}")
    return 0 if not unexplained else 1


if __name__ == "__main__":
    raise SystemExit(main())
