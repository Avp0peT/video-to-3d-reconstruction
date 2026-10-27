"""Reconstruction metrics (6N) and geometry comparison (6O).

Two rules from the directive shape this module:

* **absolute coordinates are not the target.**  Théorème 3 says the 3D code determines the body
  up to 3D affine equivalence, so comparing ``X̂`` with ``X`` pointwise is only meaningful when
  the reconstruction is already in the same frame — and the invariants (:func:`pairwise_distance_matrix`,
  :func:`collinearity_relation`, :func:`coplanarity_relation`) are what must agree in general.
* **no metric is reported without its conditioning.**  A 3D RMSE from nearly-parallel rays is a
  number with no content, so :func:`ray_angle_degrees` and :func:`condition_number` travel with it.
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np

__all__ = [
    "reprojection_error",
    "rmse_3d",
    "relative_3d_error",
    "max_3d_error",
    "pairwise_distance_matrix",
    "collinearity_relation",
    "coplanarity_relation",
    "volume_ratio_profile",
    "profile_relative_difference",
    "ray_angle_degrees",
    "condition_number",
]


def _as3(points: Any, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim == 1 and array.shape == (3,):
        array = array.reshape(1, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name}: expected shape (n, 3), got {array.shape}")
    return array


def reprojection_error(camera: Any, points3d: Any, observed: Any) -> float:
    """Largest reprojection distance of the reconstruction, in pixels.

    A **maximum**, not a mean: a mean hides the single point that does not fit, and one bad
    correspondence is exactly what this phase has to detect.
    """
    body = _as3(points3d, "points3d")
    seen = np.asarray(observed, dtype=float).reshape(-1, 2)
    if body.shape[0] != seen.shape[0]:
        raise ValueError("the number of points and observations must match")
    projected = camera.project(body)
    return float(np.linalg.norm(projected - seen, axis=1).max())


def rmse_3d(estimated: Any, reference: Any) -> float:
    """``sqrt(mean ||X̂ - X||²)`` — only meaningful in a common frame."""
    left = _as3(estimated, "estimated")
    right = _as3(reference, "reference")
    if left.shape != right.shape:
        raise ValueError("RMSE needs equal shapes")
    return float(np.sqrt(((left - right) ** 2).sum(axis=1).mean()))


def max_3d_error(estimated: Any, reference: Any) -> float:
    left = _as3(estimated, "estimated")
    right = _as3(reference, "reference")
    return float(np.linalg.norm(left - right, axis=1).max())


def relative_3d_error(estimated: Any, reference: Any) -> float:
    """``rmse_3d / diameter(reference)`` — the accuracy **as a fraction of the object's own size**.

    A raw RMSE is uninterpretable without the scale: 0.01 is excellent for a body 10 units across
    and useless for one 0.02 units across.  This is the number to quote next to a conditioning
    figure; the raw RMSE is for reproducing the geometry.
    """
    left = _as3(estimated, "estimated")
    right = _as3(reference, "reference")
    diameter = float(np.sqrt(((right[:, None, :] - right[None, :, :]) ** 2).sum(axis=-1)).max())
    if diameter <= 0:
        raise ValueError("the reference body is a single point: a relative error has no meaning")
    return rmse_3d(left, right) / diameter


def volume_ratio_profile(points: Any, *, degeneracy_tol: float = 1e-9) -> tuple[float, ...]:
    """The **sorted** multiset of ``V_q / V_ref`` over admissible quadruples — an affine invariant.

    Why this is here: théoreme 3 says the body is determined by its code up to affine equivalence,
    and tetrahedron volumes scale by ``|det A|`` under an affine map, so **the ratios of volumes are
    invariant** while the coordinates are not.  That is what makes 6O's demand — compare invariants,
    not coordinates — checkable rather than rhetorical.

    **Admissibility is decided dimensionlessly**, ``6V / max_edge³ > degeneracy_tol``, and that is
    not a detail.  A literal ``V > 0`` test — which the source's wording invites, since the source
    only says a zero-volume quadruple cannot be a denominator — is unusable in floating point, and
    this was measured rather than assumed: the cube has 12 exactly-coplanar quadruples, so its body
    admits 58 denominators, while a *correct* reconstruction of that same cube perturbs those
    quadruples to ``V ~ 1e-17`` and admits all 70.  Two bodies that are affinely equivalent then
    report profiles of different lengths and the invariant appears to "detect" a difference that is
    purely arithmetic.  This is the same failure Phase 3 documented for ``S != 0`` in 2D, and the
    same fix applies: normalise the quantity before thresholding it.

    Read the caveats, they matter:

    * this is a **measurement**, not the code and not an equivalence test.  The complete code is
      indexed (``official.indexed_code3d.indexed_body_code``); ``kozlov_body_code`` is also only a
      numerical summary. Equivalence acts on one global labelling bijection; a sorted
      bag cannot decide equivalence and must never be promoted to a criterion (the Phase 3 rule).
      :func:`~kozlov_lab.stereo.is_a_equivalent` is a numerical geometric oracle, not this code;
    * ``V_ref`` is the first admissible quadruple, so the profile is anchored; that is fine for a
      distance between two profiles but it is *not* invariant under a relabelling of the points
      that changes which quadruple is first. Sorting the values does not repair a changed
      reference denominator; relabelled bodies require an actual global correspondence test.
    """
    body = _as3(points, "points")
    volumes: dict[tuple[int, int, int, int], float] = {}
    relative: dict[tuple[int, int, int, int], float] = {}
    for quad in itertools.combinations(range(body.shape[0]), 4):
        corners = [body[i] for i in quad]
        a, b, c, d = corners
        volume6 = abs(float(np.dot(np.cross(b - a, c - a), d - a)))
        edges = [
            float(np.linalg.norm(corners[i] - corners[j]))
            for i in range(4) for j in range(i + 1, 4)
        ]
        longest = max(edges, default=0.0)
        volumes[quad] = volume6 / 6.0
        relative[quad] = 0.0 if longest <= 0 else volume6 / longest**3
    admissible = sorted((q for q in volumes if relative[q] > degeneracy_tol), key=lambda q: q)
    if not admissible:
        raise ValueError(
            "every quadruple is coplanar (dimensionless test): the volume-ratio profile is undefined"
        )
    reference = volumes[admissible[0]]
    return tuple(sorted(volumes[q] / reference for q in admissible))


def profile_relative_difference(first: Sequence[float], second: Sequence[float]) -> float:
    """Scale-free distance between two :func:`volume_ratio_profile` results.

    ``inf`` when the profiles have different lengths: that is a different *structure*, not a large
    distance, and collapsing the two would hide it (structure loss is a different failure from
    geometric drift — the distinction Phase 5 had to make between cardinality and equivalence).
    """
    left = tuple(float(v) for v in first)
    right = tuple(float(v) for v in second)
    if len(left) != len(right):
        return float("inf")
    if not left:
        return 0.0
    scale = max(max(abs(v) for v in left), max(abs(v) for v in right), 1e-300)
    return float(max(abs(a - b) for a, b in zip(left, right)) / scale)


def pairwise_distance_matrix(points: Any) -> np.ndarray:
    """All pairwise distances — the simplest affine invariant of a body.

    An affine map changes distances, so this is *not* invariant as a matrix; but the body can be
    compared **up to scale and ordering** through it, which is what the reconstruction checks use.
    (The genuinely affine-invariant quantities are the tetrahedron volume ratios of the 3D code.)
    """
    body = _as3(points, "points")
    differences = body[:, None, :] - body[None, :, :]
    return np.sqrt((differences**2).sum(axis=-1))


def collinearity_relation(points: Any, tol: float = 1e-9) -> frozenset[tuple[int, int, int]]:
    """Triples that are collinear — a *relation*, order-independent by construction."""
    body = _as3(points, "points")
    found = []
    for i, j, k in itertools.combinations(range(body.shape[0]), 3):
        u = body[j] - body[i]
        v = body[k] - body[i]
        cross = float(np.linalg.norm(np.cross(u, v)))
        scale = max(float(np.linalg.norm(u)), float(np.linalg.norm(v)), 1e-300)
        if cross / scale <= tol * scale:
            found.append((i, j, k))
    return frozenset(found)


def coplanarity_relation(points: Any, tol: float = 1e-9) -> frozenset[tuple[int, int, int, int]]:
    """Quadruples that are coplanar.  The 3D analogue of collinearity, and the degeneracy that
    makes a tetrahedron's volume zero — hence the one the 3D code must exclude."""
    body = _as3(points, "points")
    found = []
    for quad in itertools.combinations(range(body.shape[0]), 4):
        a, b, c, d = (body[i] for i in quad)
        u, v, w = b - a, c - a, d - a
        volume6 = abs(float(np.dot(np.cross(u, v), w)))
        edges = max(
            float(np.linalg.norm(u)), float(np.linalg.norm(v)), float(np.linalg.norm(w)), 1e-300
        )
        if volume6 / edges**3 <= tol:
            found.append(quad)
    return frozenset(found)


def ray_angle_degrees(direction_a: np.ndarray, direction_b: np.ndarray) -> float:
    """Angle between the two viewing rays of one point — the depth-conditioning driver.

    Near 0° the triangulation is ill-posed (the rays are nearly parallel), which is degeneracy
    D4 of the taxonomy; near 180° the point is behind one camera.
    """
    a = np.asarray(direction_a, dtype=float).reshape(3)
    b = np.asarray(direction_b, dtype=float).reshape(3)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("a viewing ray cannot be the zero vector")
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def condition_number(points3d: Any) -> float:
    """Conditioning of the point cloud: the ratio of its largest to smallest spread.

    Reported next to every RMSE because a reconstruction of a nearly-degenerate configuration
    (all points on a plane, or a very thin slab) can have a small residual and a meaningless 3D
    accuracy at the same time.
    """
    body = _as3(points3d, "points3d")
    centred = body - body.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    smallest = float(singular[-1])
    return float("inf") if smallest <= 0 else float(singular[0] / smallest)
