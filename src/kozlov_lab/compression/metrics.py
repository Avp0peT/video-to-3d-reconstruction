"""Compression metrics (5C) and the cost coupling to the formal code (5H).

The specification is explicit that ``300 points → 82 points`` is not a result.  What is
reported instead:

* **compression ratio** — ``CR = 1 - |S|/|X|``, plus the raw sizes it came from;
* **directed coverage error** ``E_{X→S} = max_{x∈X} min_{s∈S} ||x-s||`` — the quantity the
  source's эскиз is *defined* by, so it is the one that must be checked, not eyeballed;
* **Hausdorff distance** ``d_H(X,S)`` — two-sided, and computed through the same API even when
  ``S ⊆ X`` (where the reverse direction is trivially 0).  Skipping it "because it is zero"
  would remove the only place a non-subset compressor could be caught;
* **runtime and peak memory**;
* **formal-code size** — ``|T_X|`` against ``|T_S|``, which is where the ``O(n^6)`` structure
  turns a small drop in points into a large drop in work (5H).
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Mapping

import numpy as np

__all__ = [
    "compression_ratio",
    "size_reduction",
    "directed_hausdorff",
    "hausdorff",
    "coverage_report",
    "formal_code_size",
    "measure",
]


def compression_ratio(n_before: int, n_after: int) -> float:
    """``CR = 1 - |S|/|X|``.  Deliberately takes the two counts, not the sets.

    A ratio computed from ``len()`` of whatever was handy is how "300 → 82" ends up being
    reported as "0.73" without anyone checking which side was which.
    """
    if n_before < 0 or n_after < 0:
        raise ValueError("cardinalities must be non-negative")
    if n_before == 0:
        raise ValueError("cannot compress an empty image")
    return 1.0 - n_after / n_before


def size_reduction(n_before: int, n_after: int) -> dict[str, Any]:
    """All the size figures a report needs, so none of them is computed ad hoc."""
    return {
        "n_before": n_before,
        "n_after": n_after,
        "removed": n_before - n_after,
        "compression_ratio": compression_ratio(n_before, n_after),
        "retention": n_after / n_before,
    }


def _as_matrix(points: Any, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim == 1 and array.shape == (2,):
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name}: expected shape (n, 2), got {array.shape}")
    return array


def directed_hausdorff(source: Any, target: Any) -> float:
    """``max_{x ∈ source} min_{t ∈ target} ||x - t||`` — the source's ``l(B')`` quantity.

    Computed with an explicit pairwise distance matrix rather than a ``cdist`` import: the
    matrix is ``O(|source| · |target|)`` and for the sizes this project handles that is both
    cheaper than a dependency and easier to reason about.
    """
    src = _as_matrix(source, "source")
    dst = _as_matrix(target, "target")
    if src.shape[0] == 0:
        raise ValueError("source is empty")
    if dst.shape[0] == 0:
        return float("inf")
    differences = src[:, None, :] - dst[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))
    return float(distances.min(axis=1).max())


def hausdorff(first: Any, second: Any) -> float:
    """Two-sided Hausdorff distance.

    Always computed through :func:`directed_hausdorff` in both directions, even when one side
    is a subset of the other and the reverse term is structurally zero — see the module
    docstring.
    """
    return max(directed_hausdorff(first, second), directed_hausdorff(second, first))


def coverage_report(original: Any, sketch: Any) -> dict[str, Any]:
    """The coverage block the specification asks for, in one call."""
    return {
        "directed_original_to_sketch": directed_hausdorff(original, sketch),
        "directed_sketch_to_original": directed_hausdorff(sketch, original),
        "hausdorff": hausdorff(original, sketch),
        "sketch_is_subset": _is_subset(sketch, original),
    }


def _is_subset(sketch: Any, original: Any) -> bool:
    """Whether every sketch point occurs in the original (up to float equality)."""
    pool = {tuple(point) for point in _as_matrix(original, "original")}
    return all(tuple(point) in pool for point in _as_matrix(sketch, "sketch"))


def formal_code_size(n_points: int) -> dict[str, Any]:
    """Formal-code size for an image of ``n`` points, with **no degeneracies**.

    ``N = C(n,3)`` triples and ``|T_A| = N(N-1)`` entries — the ``O(n^6)`` structure of
    ``docs/theory/complexity.md``.  The maximum is quoted rather than a measured value because
    the point is the *scaling*: a compressor that turns ``n`` into ``r·n`` turns the code into
    roughly ``r^6`` of its size, which is why a modest compression can be a large speed-up.
    """
    if n_points < 3:
        return {"n_points": n_points, "n_triples": 0, "n_entries_max": 0}
    triples = n_points * (n_points - 1) * (n_points - 2) // 6
    return {
        "n_points": n_points,
        "n_triples": triples,
        "n_entries_max": triples * (triples - 1),
    }


def code_size_reduction(n_before: int, n_after: int) -> dict[str, Any]:
    """How much the *formal code* shrinks, which is not how much the point count shrinks."""
    before = formal_code_size(n_before)
    after = formal_code_size(n_after)
    ratio = (after["n_entries_max"] / before["n_entries_max"]) if before["n_entries_max"] else 0.0
    return {
        "n_points_before": n_before,
        "n_points_after": n_after,
        "n_triples_before": before["n_triples"],
        "n_triples_after": after["n_triples"],
        "n_entries_before": before["n_entries_max"],
        "n_entries_after": after["n_entries_max"],
        "code_size_ratio": ratio,
        "code_size_speedup": (1.0 / ratio) if ratio else 0.0,
        "point_ratio": (n_after / n_before) if n_before else 0.0,
    }


def measure(callable_: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Run ``callable_`` and report wall time and **peak** traced memory (not current).

    Peak, because a build that allocates a large temporary and frees it would look innocent by
    a current-memory reading — and the temporaries are exactly what the ``O(n^6)`` construction
    is made of.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    value = callable_()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {"build_ms": elapsed_ms, "peak_bytes": int(peak)}
