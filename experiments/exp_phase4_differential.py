"""F4.1 — differential test gate, counted by **accepted** cases.

Earlier versions reported the *requested* case count, which is not the same thing: a skipped
degenerate draw is not a compared case.  The gate is satisfied by

    accepted_and_compared >= 10_000   and   unexplained_disagreements == 0

so this script keeps generating until **accepted** reaches the target, and reports the full
breakdown the specification lists::

    generated, accepted, skipped_degenerate,
    inside_theorem_domain, outside_theorem_domain,
    code_equal_geometry_equal, code_unequal_geometry_unequal,
    code_equal_geometry_unequal, code_unequal_geometry_equal,
    unexplained_disagreements

Each verdict comes from :func:`~kozlov_lab.coding.recognize_affine_exact` — the production
pipeline — so this harness exercises the F4.2 truth table instead of re-deriving it.  A
partition check asserts every accepted case lands in exactly one bucket, because a counter that
silently loses cases is how a "0 disagreements" result stops meaning anything.

Run::

    python experiments/exp_phase4_differential.py                 # accepted >= 10 000
    python experiments/exp_phase4_differential.py --accepted 600  # quick pass
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
    InternalConsistencyError,
    RecognitionStatus,
    recognize_affine_exact,
    validate_kozlov_image,
)
from _common import print_header, write_json  # noqa: E402

#: ``n`` is drawn from this table, weighted towards small ``n``: both routes do
#: ``O(|T_A|)``-ish setup work, so a uniform draw over ``{4..7}`` would spend most of the budget
#: on ``n = 7`` while exploring far fewer distinct configurations.
N_WEIGHTS: tuple[tuple[int, int], ...] = ((4, 4000), (5, 3000), (6, 2000), (7, 1000))

BOX = 20

COUNTERS: tuple[str, ...] = (
    "generated",
    "accepted",
    "skipped_degenerate",
    "inside_theorem_domain",
    "outside_theorem_domain",
    "code_equal_geometry_equal",
    "code_unequal_geometry_unequal",
    "code_equal_geometry_unequal",
    "code_unequal_geometry_equal",
    "indeterminate",
    "unexplained_disagreements",
)


def _random_integer_map(rng: np.random.Generator) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    while True:
        a, b, c, d = (int(rng.integers(-3, 4)) for _ in range(4))
        if a * d - b * c != 0:
            return (a, b, c, d), (int(rng.integers(-10, 11)), int(rng.integers(-10, 11)))


def _random_image(rng: np.random.Generator, n: int) -> list[list[int]] | None:
    """A random integer image the code can be built for, or ``None`` for a degenerate draw."""
    points = [
        [int(rng.integers(-BOX, BOX + 1)), int(rng.integers(-BOX, BOX + 1))] for _ in range(n)
    ]
    try:
        validate_kozlov_image(points)
    except Exception:  # noqa: BLE001 - a degenerate draw is skipped and counted as such
        return None
    return points


def _two_parallel_image(rng: np.random.Generator, n: int) -> list[list[int]] | None:
    """An image on exactly two parallel lines, so the out-of-domain branch is exercised.

    Without a family like this the "outside the theorem domain" counter stays at zero for a
    random integer corpus: the branch never runs, yet the run looks complete.
    """
    if n < 3:
        return None
    split = int(rng.integers(1, n - 1))
    offset = int(rng.integers(1, 9))
    points = [[value, -value] for value in (int(rng.integers(-BOX, BOX + 1)) for _ in range(split))]
    points += [
        [value, offset - value]
        for value in (int(rng.integers(-BOX, BOX + 1)) for _ in range(n - split))
    ]
    if len({tuple(point) for point in points}) != n:
        return None
    try:
        validate_kozlov_image(points)
    except Exception:  # noqa: BLE001
        return None
    return points


def _apply(
    matrix: tuple[int, int, int, int], translation: tuple[int, int], points: list[list[int]]
) -> list[list[int]]:
    a, b, c, d = matrix
    tx, ty = translation
    return [[a * x + b * y + tx, c * x + d * y + ty] for x, y in points]


def _still_disagrees(x: list[list[int]], y: list[list[int]]) -> bool:
    try:
        recognize_affine_exact(x, y)
        return False
    except InternalConsistencyError:
        return True
    except Exception:  # noqa: BLE001 - a reduced case may become degenerate, which is not a finding
        return False


def shrink(x: list[list[int]], y: list[list[int]]) -> dict | None:
    """Greedily drop point pairs while the disagreement persists (delta debugging).

    A counterexample is only useful if it is small enough to read.  A removal is kept only when
    the disagreement survives it, so whatever is returned is still a genuine counterexample.
    """
    if len(x) != len(y) or len(x) <= 4 or not _still_disagrees(x, y):
        return None
    improved = False
    index = 0
    while index < len(x) and len(x) > 4:
        candidate_x = x[:index] + x[index + 1 :]
        candidate_y = y[:index] + y[index + 1 :]
        if _still_disagrees(candidate_x, candidate_y):
            x, y = candidate_x, candidate_y
            improved = True
        else:
            index += 1
    return {"x": x, "y": y} if improved else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accepted", type=int, default=10_000,
                        help="required number of ACCEPTED (compared) cases")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="generation cap; 0 means 4x the accepted target")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--shrink-budget", type=int, default=8)
    args = parser.parse_args(argv)

    attempts_cap = args.max_attempts or args.accepted * 4
    print_header("F4.1 — differential gate (accepted-based)")
    rng = np.random.default_rng(args.seed)
    schedule = [n for n, weight in N_WEIGHTS for _ in range(max(1, weight * 1000 // 10_000))]

    counts = {name: 0 for name in COUNTERS}
    unexplained: list[dict] = []
    explained: list[dict] = []
    started = time.perf_counter()
    next_report = 2000

    while counts["accepted"] < args.accepted and counts["generated"] < attempts_cap:
        index = counts["generated"]
        counts["generated"] += 1
        n = schedule[index % len(schedule)]
        use_two_parallel = rng.random() < 0.12

        x = _two_parallel_image(rng, n) if use_two_parallel else _random_image(rng, n)
        if x is None:
            counts["skipped_degenerate"] += 1
            continue
        if rng.random() < 0.5:
            y = _apply(*_random_integer_map(rng), x)
        else:
            y = _two_parallel_image(rng, n) if use_two_parallel else _random_image(rng, n)
            if y is None:
                counts["skipped_degenerate"] += 1
                continue

        try:
            pair_theorem = (
                validate_kozlov_image(x).theorem_applicable
                and validate_kozlov_image(y).theorem_applicable
            )
        except Exception:  # noqa: BLE001
            counts["skipped_degenerate"] += 1
            continue

        counts["inside_theorem_domain" if pair_theorem else "outside_theorem_domain"] += 1
        counts["accepted"] += 1

        try:
            result = recognize_affine_exact(x, y)
            status = result.status
        except InternalConsistencyError as exc:
            counts["unexplained_disagreements"] += 1
            unexplained.append({"index": index, "n": n, "artifact": exc.artifact})
            status = None
        except Exception as exc:  # noqa: BLE001 - any other exception is also a finding
            counts["unexplained_disagreements"] += 1
            unexplained.append(
                {"index": index, "n": n, "exception": f"{type(exc).__name__}: {exc}",
                 "artifact": {"inputs": {"x": x, "y": y}}}
            )
            status = None

        if status is RecognitionStatus.EXACT_AFFINE_EQUIVALENT:
            counts["code_equal_geometry_equal"] += 1
        elif status is RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT:
            counts["code_unequal_geometry_unequal"] += 1
        elif status is RecognitionStatus.CODE_COLLISION_OUTSIDE_THEOREM_DOMAIN:
            counts["code_equal_geometry_unequal"] += 1
            explained.append({"index": index, "n": n, "status": status.value})
        elif status in (
            RecognitionStatus.NUMERIC_INDETERMINATE,
            RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE,
        ):
            counts["indeterminate"] += 1

        if counts["accepted"] >= next_report:
            elapsed = time.perf_counter() - started
            print(f"    accepted={counts['accepted']:>6}  ({elapsed:>6.1f}s)  "
                  f"unexplained={counts['unexplained_disagreements']}")
            next_report += 2000

    elapsed = time.perf_counter() - started
    print("\n  counters")
    for name in COUNTERS:
        print(f"    {name:<32} {counts[name]}")

    classified = (
        counts["code_equal_geometry_equal"]
        + counts["code_unequal_geometry_unequal"]
        + counts["code_equal_geometry_unequal"]
        + counts["code_unequal_geometry_equal"]
        + counts["indeterminate"]
        + counts["unexplained_disagreements"]
    )
    gate = {
        "accepted": counts["accepted"],
        "required_accepted": args.accepted,
        "unexplained_disagreements": counts["unexplained_disagreements"],
        "accepted_requirement_met": counts["accepted"] >= args.accepted,
        "zero_unexplained_requirement_met": counts["unexplained_disagreements"] == 0,
        "gate_passed": (
            counts["accepted"] >= args.accepted and counts["unexplained_disagreements"] == 0
        ),
        "partition_is_complete": classified == counts["accepted"],
        "classified": classified,
        "domain_branch_exercised": counts["outside_theorem_domain"] > 0,
        "elapsed_seconds": elapsed,
    }
    print(f"\n  gate: accepted {counts['accepted']} >= {args.accepted} ? "
          f"{gate['accepted_requirement_met']}  |  unexplained == 0 ? "
          f"{gate['zero_unexplained_requirement_met']}")
    print(f"  partition complete ({classified}/{counts['accepted']} classified): "
          f"{gate['partition_is_complete']}")

    shrunk: list[dict] = []
    artifact_paths: list[str] = []
    if unexplained:
        target_dir = Path(__file__).resolve().parent / "results" / "discrepancies"
        target_dir.mkdir(parents=True, exist_ok=True)
        for position, record in enumerate(unexplained[: args.shrink_budget]):
            inputs = record.get("artifact", {}).get("inputs", {})
            minimal = None
            if inputs.get("x") and inputs.get("y"):
                try:
                    minimal = shrink(inputs["x"], inputs["y"])
                except Exception:  # noqa: BLE001
                    minimal = None
            shrunk.append({**record, "minimal": minimal})
            path = target_dir / f"phase4_unexplained_{args.seed}_{position}.json"
            path.write_text(
                json.dumps({**record, "minimal": minimal}, indent=2, ensure_ascii=False, default=str)
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            artifact_paths.append(str(path))
        print(f"  {len(artifact_paths)} artifact(s) written to {target_dir}")

    target, meta = write_json(
        "phase4_differential",
        {
            "experiment": "phase4_differential",
            "title": "differential gate counted by accepted cases (F4.1)",
            "counters": counts,
            "gate": gate,
            "explained": explained[:20],
            "unexplained": [{k: v for k, v in record.items() if k != "artifact"} for record in shrunk],
            "artifact_paths": artifact_paths,
            "note": (
                "verdicts come from recognize_affine_exact, so this harness exercises the F4.2 "
                "truth table rather than re-deriving it"
            ),
        },
        seed=args.seed,
        config={
            "accepted_target": args.accepted,
            "max_attempts": attempts_cap,
            "n_weights": N_WEIGHTS,
            "coordinate_box": BOX,
            "generator": "50% affine image, 50% independent draw; ~12% on two parallel lines",
            "shrink_budget": args.shrink_budget,
        },
    )
    print(f"\n  [written] {target}")
    print(f"  [written] {meta}")
    return 0 if gate["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
