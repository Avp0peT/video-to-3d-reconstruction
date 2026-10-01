"""Euclidean $\\varepsilon$-cover — **baseline**, not from the source.

Centres may lie anywhere in the plane, so this is a true $\\varepsilon$-net rather than the
source's эскиз (whose centres must be points of the image).  The distinction is not pedantic:
it is exactly why the two can have different cardinalities on the same input, and why only the
эскиз version is comparable with the primary text.

The returned points are therefore **not** a subset of the input when centres are optimized, so
``surjective_on=False`` and the coverage direction that matters is
``directed_hausdorff(original, sketch)``.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from ..metrics import directed_hausdorff
from ..model import CompressionResult, Compressor, CompressorContract

__all__ = ["TeachingEpsilonCover", "farthest_point_centres"]


def farthest_point_centres(points: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point (maximin) seeding, used here to place the $\\varepsilon$-cover centres.

    Deterministic: the first centre is the point closest to the centroid, so the result does not
    depend on the input order.
    """
    n = points.shape[0]
    k = min(k, n)
    centre_of_mass = points.mean(axis=0)
    first = int(np.argmin(((points - centre_of_mass) ** 2).sum(axis=1)))
    chosen = [first]
    nearest = np.linalg.norm(points - points[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[nxt], axis=1))
    return points[chosen]


class TeachingEpsilonCover(Compressor):
    """Cover the image with as few balls of radius $\\varepsilon$ as a greedy pass needs.

    **Baseline.**  The construction is the classical greedy set cover on a Euclidean ball
    intersection graph; it is *not* in the primary source and must not be called an эскиз.
    Coverage is **guaranteed** (each centre covers every point within $\\varepsilon$ by
    construction) and cardinality is an *upper bound* (greedy cover is not minimum).
    """

    contract = CompressorContract(
        name="TeachingEpsilonCover",
        family="teaching",
        surjective_on=False,
        cover_guarantee="guaranteed",
        cardinality_guarantee="upper_bound",
        deterministic=True,
        complexity="O(k n m) with k = |cover|, m = |candidate centres|",
        source="greedy set cover over input points plus a lattice (classical); NOT from Kozlov 2002",
    )

    def fit(self, points: Any, *, epsilon: float, **_: Any) -> CompressionResult:
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        matrix = self._matrix(points)
        if epsilon == 0:
            # a zero-radius cover is the image itself; no lattice can help
            candidates = matrix.copy()
        else:
            spacing = epsilon / np.sqrt(2.0)
            low = matrix.min(axis=0) - epsilon
            high = matrix.max(axis=0) + epsilon
            counts = np.maximum(np.ceil((high - low) / spacing).astype(int), 1)
            axes = [np.linspace(low[k], high[k], counts[k] + 1) for k in range(2)]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
            candidates = np.vstack([matrix, grid])

        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        differences = matrix[:, None, :] - candidates[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))
        uncovered = set(range(matrix.shape[0]))
        chosen: list[int] = []
        while uncovered:
            best_index, best_cover = -1, -1
            for candidate in range(candidates.shape[0]):
                covered = sum(1 for j in uncovered if distances[j, candidate] <= epsilon + 1e-12)
                if covered > best_cover:
                    best_index, best_cover = candidate, covered
            if best_cover <= 0:  # pragma: no cover - the lattice guarantees otherwise
                raise AssertionError("no candidate centre covers any uncovered point")
            chosen.append(best_index)
            uncovered -= {j for j in uncovered if distances[j, best_index] <= epsilon + 1e-12}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        sketch = candidates[chosen]
        achieved = directed_hausdorff(matrix, sketch)
        if achieved > epsilon + 1e-12:
            raise AssertionError(
                f"TeachingEpsilonCover promised epsilon={epsilon:.3e} but achieved {achieved:.3e}"
            )
        from ..metrics import _is_subset

        return CompressionResult(
            points=sketch,
            indices=tuple(-1 for _ in chosen),
            contract=self.contract,
            requested_radius=float(epsilon),
            achieved_radius=achieved,
            build_ms=elapsed_ms,
            peak_bytes=int(peak),
            diagnostics={
                "n_before": int(matrix.shape[0]),
                "solver": "greedy set cover",
                "candidate_centres": int(candidates.shape[0]),
                "lattice_spacing": float(epsilon / np.sqrt(2.0)) if epsilon else 0.0,
                "centre_placement": "input points plus a lattice (may be off-image)",
                "sketch_is_subset": _is_subset(sketch, matrix),
            },
        )
