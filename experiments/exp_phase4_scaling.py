"""4P — how the ``O(n^6)`` structure behaves in the real implementation.

The specification asks for measurements of, for ``n = 4, 5, 6, …`` until resources grow
visibly:

* number of triples;
* number of formal ratio entries ``|T_A| = q(N-1)``;
* build time, **peak memory**, equivalence time, geometric-oracle time;
* integrated **materialised vs lazy**, so the memory claim in
  :mod:`kozlov_lab.coding.lazy` is measured rather than asserted.

The point is not "big data": it is to confirm that the ``O(n^6)`` growth is real, to say
where the materialised representation stops being usable, and to record that boundary as a
number.

Run::

    python experiments/exp_phase4_scaling.py
    python experiments/exp_phase4_scaling.py --max-n 16 --entry-cap 200000
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab.coding import (  # noqa: E402
    build_kozlov_image_code_exact,
    build_kozlov_image_code_lazy_exact,
    check_geometric_equivalence,
    kozlov_codes_equivalent,
)
from _common import print_header, write_csv, write_json  # noqa: E402

FIELDNAMES = (
    "n_points",
    "n_triples",
    "n_admissible_denominators",
    "n_entries",
    "entries_over_triples_squared",
    "materialized_build_ms",
    "materialized_peak_bytes",
    "lazy_build_ms",
    "lazy_peak_bytes",
    "memory_ratio",
    "lazy_ratio_query_us",
    "equivalence_ms",
    "equivalence_nodes",
    "geometric_oracle_ms",
    "stopped_reason",
)


def _image(n: int) -> list[list[int]]:
    """A deterministic general-position integer image with no gratuitous symmetry.

    Coordinates come from a quadratic residue pattern rather than a grid: a grid produces many
    collinear triples and many parallel segments, which would distort both the ratio count
    (fewer admissible denominators) and the domain class.
    """
    points: list[list[int]] = []
    for index in range(n):
        x = index * index + 3 * index
        y = 2 * index * index + index * 7 + (index % 3) * 5
        points.append([x, y])
    return points


def measure(n: int, *, entry_cap: int) -> dict:
    points = _image(n)
    row: dict[str, float | int | str] = {"n_points": n}

    # -- materialised ---------------------------------------------------- #
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    materialized = build_kozlov_image_code_exact(points)
    build_ms = (time.perf_counter() - started) * 1000.0
    _, peak_materialized = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row.update(
        n_triples=materialized.n_triples,
        n_admissible_denominators=materialized.n_admissible_denominators,
        n_entries=len(materialized),
        entries_over_triples_squared=(
            len(materialized) / (materialized.n_triples**2) if materialized.n_triples else 0.0
        ),
        materialized_build_ms=build_ms,
        materialized_peak_bytes=peak_materialized,
    )

    # -- lazy ------------------------------------------------------------ #
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    lazy = build_kozlov_image_code_lazy_exact(points)
    lazy_ms = (time.perf_counter() - started) * 1000.0
    _, peak_lazy = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # a few ratio queries, to show that the lazy view is meant for exactly that
    triples = sorted(lazy.view.triples, key=sorted)
    admissible = sorted(lazy.view.admissible_triples, key=sorted)
    started = time.perf_counter()
    samples = 0
    for numerator in triples[:8]:
        for denominator in admissible[:8]:
            if numerator != denominator:
                lazy.ratio(numerator, denominator)
                samples += 1
    query_us = ((time.perf_counter() - started) / max(samples, 1)) * 1e6

    row.update(
        lazy_build_ms=lazy_ms,
        lazy_peak_bytes=peak_lazy,
        memory_ratio=(peak_materialized / peak_lazy) if peak_lazy else 0.0,
        lazy_ratio_query_us=query_us,
    )

    # -- equivalence (pruned) vs geometric oracle ------------------------- #
    permutation = list(np.random.default_rng(n).permutation(n))
    relabelled = [points[i] for i in permutation]

    # the second code is built **outside** the timed region: otherwise this column would
    # silently report "build + compare" and overstate the comparison by an order of magnitude
    other_code = build_kozlov_image_code_exact(relabelled)
    started = time.perf_counter()
    code_result = kozlov_codes_equivalent(materialized, other_code)
    equivalence_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    geometric = check_geometric_equivalence(points, relabelled)
    oracle_ms = (time.perf_counter() - started) * 1000.0

    assert code_result.equivalent and geometric.equivalent, (n, code_result.reason, geometric.reason)

    row.update(
        equivalence_ms=equivalence_ms,
        equivalence_nodes=code_result.candidates_examined,
        geometric_oracle_ms=oracle_ms,
        stopped_reason="",
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-n", type=int, default=4)
    parser.add_argument("--max-n", type=int, default=18)
    parser.add_argument("--entry-cap", type=int, default=400_000,
                        help="stop before the materialised entry count exceeds this")
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args(argv)

    print_header("Phase 4P — O(n^6) structure in the real implementation")
    rows: list[dict] = []
    stopped: str | None = None

    print(f"  {'n':>3} {'triples':>8} {'|T_A|':>10} {'mat ms':>9} {'mat peak':>11} "
          f"{'lazy ms':>8} {'lazy peak':>10} {'ratio':>7} {'equiv ms':>9} {'oracle ms':>10}")
    for n in range(args.min_n, args.max_n + 1):
        # predict the entry count first, so the cap stops us before allocating
        triples = n * (n - 1) * (n - 2) // 6
        projected = triples * (triples - 1)
        if projected > args.entry_cap:
            stopped = (
                f"stopped before n={n}: |T_A| would reach {projected}, above the entry cap "
                f"{args.entry_cap}. Raising the cap means allocating ~{projected} RatioEntry "
                "objects, which is the boundary this experiment is meant to locate."
            )
            break
        row = measure(n, entry_cap=args.entry_cap)
        rows.append(row)
        print(
            f"  {row['n_points']:>3} {row['n_triples']:>8} {row['n_entries']:>10} "
            f"{row['materialized_build_ms']:>9.1f} {row['materialized_peak_bytes'] / 1024:>10.0f}K "
            f"{row['lazy_build_ms']:>8.1f} {row['lazy_peak_bytes'] / 1024:>9.0f}K "
            f"{row['memory_ratio']:>6.1f}x {row['equivalence_ms']:>9.1f} {row['geometric_oracle_ms']:>10.2f}"
        )

    if stopped:
        print(f"\n  {stopped}")

    if not rows:
        raise SystemExit("no measurements produced; raise --entry-cap")

    first, last = rows[0], rows[-1]
    growth = {
        "n_first": first["n_points"],
        "n_last": last["n_points"],
        "n_ratio": last["n_points"] / first["n_points"],
        "entries_ratio": last["n_entries"] / max(first["n_entries"], 1),
        "entries_over_triples_squared_first": first["entries_over_triples_squared"],
        "entries_over_triples_squared_last": last["entries_over_triples_squared"],
        "memory_ratio_last": last["memory_ratio"],
        "stopped_reason": stopped,
    }
    print(f"\n  |T_A| grew {growth['entries_ratio']:.1f}x while n grew {growth['n_ratio']:.2f}x")
    print(f"  |T_A| / C(n,3)^2 tends to 1 as n grows: {growth['entries_over_triples_squared_last']:.4f}")
    print(f"  materialised/lazy peak-memory ratio at the last n: {growth['memory_ratio_last']:.1f}x")

    target, meta = write_csv(
        "phase4_scaling",
        rows,
        fieldnames=FIELDNAMES,
        seed=args.seed,
        config={
            "min_n": args.min_n,
            "max_n": args.max_n,
            "entry_cap": args.entry_cap,
            "stop_rule": "projected |T_A| = C(n,3)(C(n,3)-1) must stay under the entry cap",
            "image": "quadratic pattern, deterministic per n",
        },
    )
    json_target, json_meta = write_json(
        "phase4_scaling_summary",
        {
            "experiment": "phase4_scaling",
            "title": "materialised vs lazy growth of the formal code",
            "growth": growth,
            "rows": rows,
            "note": (
                "peak memory is tracemalloc-based and per-measurement (reset_peak before each "
                "build); it measures the Python objects the code allocates, not the process RSS"
            ),
        },
        seed=args.seed,
        config={"min_n": args.min_n, "max_n": args.max_n, "entry_cap": args.entry_cap},
    )
    print(f"\n  [written] {target}")
    print(f"  [written] {meta}")
    print(f"  [written] {json_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
