"""Grid reduction — **engineering baseline**, no mathematical guarantee.

Snaps points to a grid of the given cell size and keeps the grid representatives that are
closest to a cell centroid.  Recorded because a "obviously fine" downsampling is the thing a
practitioner reaches for first, and its behaviour under transforms is worth measuring rather
than assuming — a grid is tied to the coordinate axes, so it has no reason to commute with a
rotation at all.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from ..metrics import directed_hausdorff
from ..model import CompressionResult, Compressor, CompressorContract

__all__ = ["TeachingGridReduce"]


class TeachingGridReduce(Compressor):
    """One representative per occupied grid cell (the point nearest the cell centre).

    **Baseline with no guarantee**: the covering radius is whatever it turns out to be, and it
    is reported as ``measured``.
    """

    contract = CompressorContract(
        name="TeachingGridReduce",
        family="teaching",
        surjective_on=True,
        cover_guarantee="measured",
        cardinality_guarantee="upper_bound",
        deterministic=True,
        complexity="O(n)",
        source="grid downsampling (engineering baseline); NOT from Kozlov 2002",
    )

    def fit(self, points: Any, *, cell: float, **_: Any) -> CompressionResult:
        if cell <= 0:
            raise ValueError("the grid cell size must be positive")
        matrix = self._matrix(points)
        float_matrix = self._float_view(matrix)
        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        keys = np.floor(float_matrix / cell).astype(np.int64)
        chosen: list[int] = []
        seen: dict[tuple[int, int], int] = {}
        for index, key in enumerate(map(tuple, keys)):
            if key not in seen:
                seen[key] = index
                chosen.append(index)
                continue
            # keep the point nearest the cell centre, so the result does not depend on order
            centre = (np.asarray(key, dtype=float) + 0.5) * cell
            current = chosen[seen[key]]
            if np.linalg.norm(float_matrix[index] - centre) < np.linalg.norm(
                float_matrix[current] - centre
            ):
                chosen[chosen.index(current)] = index
                seen[key] = index
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        achieved = directed_hausdorff(float_matrix, float_matrix[sorted(chosen)])
        return self._result(
            matrix, sorted(chosen), self.contract,
            requested_radius=None, achieved_radius=achieved,
            build_ms=elapsed_ms, peak_bytes=int(peak),
            diagnostics={"cell": float(cell), "guarantee": "none (measured radius only)"},
        )
