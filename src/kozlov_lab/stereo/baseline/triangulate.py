"""DLT triangulation — **BASELINE**.

The standard linear method: for ``p1 ~ P1 X`` and ``p2 ~ P2 X``,

    [u1 P1_3 - P1_1]                    A X = 0
    [v1 P1_3 - P1_2]  X = 0      solved by the right singular vector of A.
    [u2 P2_3 - P2_1]
    [v2 P2_3 - P2_2]

**This is not "Kozlov triangulation".**  The primary source uses parallel projection and defines
the reconstruction through tetrahedron geometry; it contains no DLT, no SVD and no camera matrix.
The method is attributed to the standard literature and named accordingly (6D of the directive).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["triangulate_dlt", "triangulate_many"]


def triangulate_dlt(
    projection_a: Any, point_a: Sequence[float],
    projection_b: Any, point_b: Sequence[float],
) -> np.ndarray:
    """One 3D point from two image points.  Raises on a rank-deficient system.

    A rank-deficient ``A`` means the two rays do not determine a point (identical cameras, or
    parallel rays) — degeneracy D1/D4 of the taxonomy.  Returning *some* point there would be the
    "false stable 3D" the directive's Control C forbids.
    """
    p1 = np.asarray(projection_a, dtype=float)
    p2 = np.asarray(projection_b, dtype=float)
    u1, v1 = float(point_a[0]), float(point_a[1])
    u2, v2 = float(point_b[0]), float(point_b[1])
    rows = np.vstack([
        u1 * p1[2] - p1[0],
        v1 * p1[2] - p1[1],
        u2 * p2[2] - p2[0],
        v2 * p2[2] - p2[1],
    ])
    _, singular, vh = np.linalg.svd(rows)
    if singular[2] <= 1e-12 * max(singular[0], 1e-300):
        raise ValueError(
            "the triangulation system is rank-deficient (identical cameras or parallel rays); "
            "the depth is not determined, and inventing one would be a false stable 3D"
        )
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        raise ValueError("the triangulated point is at infinity")
    return homogeneous[:3] / homogeneous[3]


def triangulate_many(
    projection_a: Any, points_a: Any, projection_b: Any, points_b: Any,
) -> tuple[np.ndarray, list[int]]:
    """Triangulate a whole correspondence.  Returns ``(points, failed_indices)``.

    Failures are **returned, not raised**: one degenerate point should not abort a reconstruction
    of the others, but it must be visible.  Silently dropping it would make the failure rate
    unreportable.
    """
    left = np.asarray(points_a, dtype=float).reshape(-1, 2)
    right = np.asarray(points_b, dtype=float).reshape(-1, 2)
    if left.shape != right.shape:
        raise ValueError("both images must have the same number of points")
    points: list[np.ndarray] = []
    failed: list[int] = []
    for index, (a, b) in enumerate(zip(left, right)):
        try:
            points.append(triangulate_dlt(projection_a, a, projection_b, b))
        except ValueError:
            failed.append(index)
            points.append(np.full(3, np.nan))
    return np.asarray(points), failed
