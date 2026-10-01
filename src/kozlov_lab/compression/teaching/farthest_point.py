"""Farthest-point sampling — **baseline**, not from the source.

Fixes the **cardinality** $k$ and picks well-spread representatives.  The covering radius is an
*outcome*, not an input, and the guarantee is only an upper bound: the maximin greedy is
$O(\\log k)$-competitive for the $k$-centre problem, which is an approximation statement and
must not be reported as optimality.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from ..metrics import directed_hausdorff
from ..model import CompressionResult, Compressor, CompressorContract

__all__ = ["TeachingFarthestPoint"]


class TeachingFarthestPoint(Compressor):
    """``k`` representatives by maximin (farthest-point) sampling.

    **Baseline.**  Deterministic given the input order for the first centre, which is chosen as
    the point nearest the centroid so that the result does not depend on how the image happens
    to be written down — the same relabelling concern that Phase 4 had to fix.
    """

    contract = CompressorContract(
        name="TeachingFarthestPoint",
        family="teaching",
        surjective_on=True,
        cover_guarantee="measured",
        cardinality_guarantee="fixed",
        deterministic=True,
        complexity="O(k n)",
        source="maximin / farthest-point sampling (classical); NOT from Kozlov 2002",
    )

    def fit(self, points: Any, *, k: int, **_: Any) -> CompressionResult:
        matrix = self._matrix(points)
        n = matrix.shape[0]
        if k < 1:
            raise ValueError("k must be at least 1")
        k = min(int(k), n)
        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        centre_of_mass = matrix.mean(axis=0)
        first = int(np.argmin(((matrix - centre_of_mass) ** 2).sum(axis=1)))
        chosen = [first]
        nearest = np.linalg.norm(matrix - matrix[first], axis=1)
        while len(chosen) < k:
            nxt = int(np.argmax(nearest))
            chosen.append(nxt)
            nearest = np.minimum(nearest, np.linalg.norm(matrix - matrix[nxt], axis=1))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        achieved = directed_hausdorff(matrix, matrix[chosen])
        return self._result(
            matrix, sorted(chosen), self.contract,
            requested_radius=None, achieved_radius=achieved,
            build_ms=elapsed_ms, peak_bytes=int(peak),
            diagnostics={
                "k_requested": int(k),
                "guarantee": "upper bound on the covering radius for this k; not optimal",
            },
        )
