"""Affine-aware sampling — **our own** experimental compressor (Phase 5G).

The question this exists to ask: *can a compression rule be built that suits affine recognition
better than a Euclidean cover does?*  The Euclidean cover cannot be expected to commute with a
general affine map, because it is defined by Euclidean distances and those are not preserved.

Two ingredients, both of them this project's:

1. **area-weighted importance.**  A point is important when it participates in large doubled
   areas — equivalently when removing it would change the code's ``T_A`` a lot.  The weight uses
   ``signed_double_area`` from the Phase 1 geometry layer.
2. **affine-normalized distance.**  Coverage is measured after normalizing the image by its own
   covariance ellipse, which is an affine-invariant preconditioning.  Coverage is then claimed
   in the *normalized* metric, not in the Euclidean one — and the contract says so, because
   reporting a normalized radius as a Euclidean radius would be a false guarantee.

Under a general affine map the covariance normalization is equivariant by construction, so this
sampler has a *reason* to behave better than the baselines.  Whether it actually does is the
experiment (``experiments/exp_phase5_equivariance.py``); the code makes no claim here.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from ..metrics import directed_hausdorff
from ..model import CompressionResult, Compressor, CompressorContract

__all__ = ["AffineAwareSampler", "normalized_coordinates", "area_importance"]


def normalized_coordinates(points: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map the image to coordinates normalized by its own covariance (an affine invariant).

    Returns ``(normalized, transform)`` with ``normalized = (points - mean) @ T``; the transform
    is returned so the caller can invert it.  If the covariance is singular (a 1-dimensional
    image) the identity is used and the caller can see that from ``transform``.
    """
    matrix = np.asarray(points, dtype=float).reshape(-1, 2)
    centre = matrix.mean(axis=0)
    centred = matrix - centre
    covariance = centred.T @ centred / max(matrix.shape[0] - 1, 1)
    try:
        # whitening: the inverse square root of the covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    except np.linalg.LinAlgError:  # pragma: no cover - eigh is robust
        whitening = np.eye(2)
    return centred @ whitening, whitening


def area_importance(points: Any) -> np.ndarray:
    """Per-point importance: the largest doubled area of a triangle containing the point.

    Uses the Phase 1 geometry primitive rather than a local determinant, so the quantity is the
    same one the formal code is built from.  All-zero for a 1-dimensional image, in which case
    every point is equally important and the sampler falls back to uniform weights.
    """
    from ...geometry.affine_frame import signed_double_area

    matrix = np.asarray(points, dtype=float).reshape(-1, 2)
    n = matrix.shape[0]
    weights = np.zeros(n)
    for i in range(n):
        best = 0.0
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                best = max(best, abs(signed_double_area(matrix[i], matrix[j], matrix[k])))
        weights[i] = best
    if weights.max() <= 0:
        return np.ones(n)
    return weights / weights.max()


class AffineAwareSampler(Compressor):
    """Greedy cover in **normalized** coordinates, weighted by area importance.

    **Experimental, this project's own.**  The contract is deliberately narrow: coverage is
    guaranteed in the normalized metric (``cover_guarantee="guaranteed_normalized"``), and no
    Euclidean radius is claimed.  A caller who needs a Euclidean guarantee should use
    :class:`~kozlov_lab.compression.official.KozlovSketch`.
    """

    contract = CompressorContract(
        name="AffineAwareSampler",
        family="experimental",
        surjective_on=True,
        cover_guarantee="guaranteed_normalized",
        cardinality_guarantee="upper_bound",
        deterministic=True,
        complexity="O(n^3) for the importance weights, O(n^2) for the cover",
        source="this project's own exploratory construction; NO author attribution",
    )

    def fit(self, points: Any, *, r: float, **_: Any) -> CompressionResult:
        if r < 0:
            raise ValueError("the normalized covering radius r must be non-negative")
        matrix = self._matrix(points)
        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        normalized, _ = normalized_coordinates(matrix)
        importance = area_importance(matrix)

        differences = normalized[:, None, :] - normalized[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))
        uncovered = set(range(matrix.shape[0]))
        chosen: list[int] = []
        while uncovered:
            best_index, best_score = -1, -1.0
            for candidate in sorted(uncovered):
                covered = sum(1 for j in uncovered if distances[candidate, j] <= r + 1e-12)
                score = covered * (0.5 + importance[candidate])
                if score > best_score:
                    best_index, best_score = candidate, score
            chosen.append(best_index)
            uncovered -= {j for j in uncovered if distances[best_index, j] <= r + 1e-12}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        normalized_radius = float(distances[list(range(matrix.shape[0])), :][:, chosen].min(axis=1).max())
        euclidean_radius = directed_hausdorff(matrix, matrix[chosen])
        if normalized_radius > r + 1e-9:
            raise AssertionError(
                f"AffineAwareSampler promised normalized radius {r:.3e} but achieved "
                f"{normalized_radius:.3e}"
            )
        return self._result(
            matrix, sorted(chosen), self.contract,
            requested_radius=float(r), achieved_radius=euclidean_radius,
            build_ms=elapsed_ms, peak_bytes=int(peak),
            diagnostics={
                "normalized_radius_achieved": normalized_radius,
                "euclidean_radius_achieved": euclidean_radius,
                "guarantee": "coverage holds in the normalized metric only",
                "mean_importance_of_chosen": float(importance[sorted(chosen)].mean()),
            },
        )
