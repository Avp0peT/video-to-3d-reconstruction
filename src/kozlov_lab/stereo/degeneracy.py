"""Degeneracy taxonomy (6J) — six named failures, never one word "failed".

The directive is explicit that ``reconstruction failed`` is not an acceptable report.  Each of
D1–D6 has a different cause, a different consequence and a different remedy, so they are separate
values and each carries the evidence that produced it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

import numpy as np

__all__ = ["DegeneracyKind", "classify_geometry"]


class DegeneracyKind(str, Enum):
    """The directive's six cases, plus two this phase had to add.

    ``D7``/``D8`` were not in the directive's list, and they are not decoration: they are the
    degeneracies the axis-aligned cube actually exhibits, and without names for them the taxonomy
    would have to report that configuration as ``OK`` while its correspondence demonstrably cannot
    be identified.  A taxonomy that cannot express its own most common failure will mis-attribute
    it — the lesson Phase 5 learned when a missing category pushed 32 real losses into
    ``RECOGNIZER_DISCREPANCY``.
    """

    OK = "OK"
    D1_ZERO_BASELINE = "D1_ZERO_BASELINE"
    D2_SMALL_BASELINE = "D2_SMALL_BASELINE"
    D3_POINT_NEAR_EPIPOLE = "D3_POINT_NEAR_EPIPOLE"
    D4_PARALLEL_RAYS = "D4_PARALLEL_RAYS"
    D5_PLANAR_STRUCTURE = "D5_PLANAR_STRUCTURE"
    D6_SYMMETRIC_OBJECT = "D6_SYMMETRIC_OBJECT"
    D7_COINCIDENT_RAYS = "D7_COINCIDENT_RAYS"
    D8_EPIPOLAR_PLANE_MULTIPLICITY = "D8_EPIPOLAR_PLANE_MULTIPLICITY"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    from .metrics import ray_angle_degrees

    return ray_angle_degrees(a, b)


def _coincident_ray_pairs(cloud: np.ndarray, centre: np.ndarray, *, tol_degrees: float) -> list[tuple[int, int]]:
    """Point pairs on the **same ray** from ``centre`` — i.e. the same pixel.

    Same ray, not merely parallel: the directions must agree in sign, or the two points are on
    opposite sides of the camera and their projections differ.
    """
    found: list[tuple[int, int]] = []
    directions = cloud - centre
    for i in range(len(cloud)):
        for j in range(i + 1, len(cloud)):
            u, v = directions[i], directions[j]
            if np.dot(u, v) <= 0:  # opposite sides: not the same ray
                continue
            try:
                if _angle(u, v) <= tol_degrees:
                    found.append((i, j))
            except ValueError:
                continue
    return found


def _epipolar_plane_pairs(cloud: np.ndarray, first: np.ndarray, second: np.ndarray, *, tol: float) -> list[tuple[int, int]]:
    """Point pairs that lie in one **epipolar plane** — the normalised triple product vanishes.

    This is the general form of the failure that ``D7`` names at its extreme.  A left pixel
    constrains its 3D point to a ray, but the ray and its whole epipolar *plane* project to the
    same epipolar *line* in the other view.  So if two body points share an epipolar plane, the
    right-image pixel of the second lies on the epipolar line of the first and the pair
    ``(left[i], right[j])`` triangulates with **zero residual** even though they are different
    points.  Every such pair is a zero-cost correspondence hypothesis, which is what makes the
    answer non-unique.

    Measured on the cube: 8 of the 28 pairs are coplanar with the baseline (the four vertices with
    ``y = 0``), and that is what turns 1 correct answer into 96 zero-cost ones.
    """
    baseline = second - first
    norm_baseline = float(np.linalg.norm(baseline))
    found: list[tuple[int, int]] = []
    if norm_baseline == 0:
        return found
    for i in range(len(cloud)):
        for j in range(i + 1, len(cloud)):
            u, v = cloud[i] - first, cloud[j] - first
            nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            if nu == 0 or nv == 0:
                continue
            sine = abs(float(np.dot(np.cross(u, v), baseline))) / (nu * nv * norm_baseline)
            if sine <= tol:
                found.append((i, j))
    return found


def _symmetric_pairs(cloud: np.ndarray, *, decimals: int = 9) -> list[tuple[int, int]]:
    """Point pairs with **identical normalised distance profiles** — interchangeable vertices.

    The cheap necessary condition for a symmetry of the body: if a similarity maps the body to
    itself and swaps ``i`` and ``j``, then the multiset of distances from ``i`` equals that from
    ``j`` up to the body's own scale.  Scaling by the diameter makes it dimensionless.

    This is a **necessary** test, hence an over-approximation — it can call a coincidental pair
    symmetric.  That is acceptable for a *label* and is why ``D6`` is only reported when nothing
    else fires; it is never used to decide anything (the equivalence decision is
    :func:`~kozlov_lab.stereo.is_a_equivalent`).
    """
    if len(cloud) < 2:
        return []
    distances = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=-1)
    diameter = float(distances.max()) or 1.0
    profiles = [tuple(sorted(np.round(distances[i] / diameter, decimals))) for i in range(len(cloud))]
    return [
        (i, j)
        for i in range(len(cloud))
        for j in range(i + 1, len(cloud))
        if profiles[i] == profiles[j]
    ]


def classify_geometry(
    centres: Sequence[Sequence[float]],
    body: Any,
    *,
    ray_angle_threshold_degrees: float = 1.0,
    baseline_ratio_threshold: float = 1e-3,
    epipole_angle_threshold_degrees: float = 1e-2,
    coincident_ray_threshold_degrees: float = 1e-6,
    epipolar_plane_tolerance: float = 1e-9,
) -> tuple[DegeneracyKind, dict[str, Any]]:
    """Name the degeneracy — or ``OK`` — with the evidence that decided it.

    The **check order** is documented rather than implied, and it encodes a priority:

    1. ``D1``/``D2`` — a vanishing baseline makes every later test meaningless, so it is first;
    2. ``D5`` — planarity/thinness is **intrinsic to the body** (no camera arrangement fixes it,
       and it removes the 3D code's denominators), so it precedes everything configuration-dependent;
    3. ``D3`` — a single point on the baseline is the **most specific** statement available, and it
       must come before ``D8``: a point on the baseline lies in *every* epipolar plane through it,
       so the general pair test would otherwise always fire first and D3 would never be reported;
    4. ``D7``/``D8`` — failures of **identity**: the pixel evidence admits several zero-cost
       correspondences, so *who is who* is undecidable.  Identity fails before conditioning does;
    5. ``D4`` — near-parallel viewing rays, where the answer is determinate but its precision is not;
    6. ``D6`` — symmetry, reported **last and on purpose**.

    Why ``D6`` is last: symmetry is *benign*.  Measured, not assumed — a square pyramid rotated
    30° about the vertical axis is just as symmetric as the unrotated one (6 interchangeable pairs
    either way, ``D6`` both times), but it has a **unique** correspondence, while the axis-aligned
    one has four.  The ambiguity came from the alignment with the epipolar geometry (``D8``), not
    from the symmetry.  Symmetry alone means several answers are *correct*; it does not mean the
    answer is *unknown*.  Filing a benign property above a real one would hide the real one.
    """
    from .metrics import condition_number

    evidence: dict[str, Any] = {}
    first = np.asarray(centres[0], dtype=float)
    second = np.asarray(centres[1], dtype=float)
    baseline_vector = second - first
    baseline = float(np.linalg.norm(baseline_vector))
    cloud = np.asarray(body, dtype=float).reshape(-1, 3)
    scale = float(np.ptp(cloud, axis=0).max()) or 1.0
    evidence["baseline"] = baseline
    evidence["scale"] = scale
    evidence["baseline_over_scale"] = baseline / scale

    # 1. baseline
    if baseline <= 1e-12:
        return DegeneracyKind.D1_ZERO_BASELINE, evidence
    if baseline / scale < baseline_ratio_threshold:
        return DegeneracyKind.D2_SMALL_BASELINE, evidence

    # 2. intrinsic structure: a planar (or very thin) cloud has no admissible denominator for the
    #    3D code, which is a property of the body and independent of where the cameras are.
    condition = condition_number(cloud)
    evidence["condition_number"] = condition
    if condition > 1e6:
        return DegeneracyKind.D5_PLANAR_STRUCTURE, evidence

    # 3. the most *specific* identity/conditioning statement first: a single point sitting on the
    #    baseline itself.  D3 must precede D8 or it is unreachable — a point on the baseline is in
    #    *every* epipolar plane through the baseline, so the general pair test would always fire
    #    first and the specific diagnosis would never be reported.  (Measured: with D8 first, no
    #    configuration in this corpus ever produced D3.)
    #
    #    And this is *not* a restatement of D2: the earlier implementation compared
    #    ``baseline / scale`` a second time, with a looser threshold, so it could only ever
    #    re-report a small baseline and never detected a genuine near-epipole point.
    epipole_angles: list[float] = []
    for point in cloud:
        for centre, other in ((first, second), (second, first)):
            try:
                epipole_angles.append(_angle(point - centre, other - centre))
            except ValueError:
                continue
    evidence["min_epipole_angle_degrees"] = min(epipole_angles) if epipole_angles else float("nan")
    if epipole_angles and min(epipole_angles) < epipole_angle_threshold_degrees:
        return DegeneracyKind.D3_POINT_NEAR_EPIPOLE, evidence

    # 4. pairs: can the correspondence be identified from the pixels at all?
    collisions = sorted(
        {pair for centre in (first, second)
         for pair in _coincident_ray_pairs(cloud, centre, tol_degrees=coincident_ray_threshold_degrees)}
    )
    evidence["n_coincident_ray_pairs"] = len(collisions)
    evidence["coincident_ray_pairs"] = collisions
    if collisions:
        return DegeneracyKind.D7_COINCIDENT_RAYS, evidence

    epipolar_pairs = _epipolar_plane_pairs(cloud, first, second, tol=epipolar_plane_tolerance)
    evidence["n_epipolar_plane_pairs"] = len(epipolar_pairs)
    evidence["epipolar_plane_pairs"] = epipolar_pairs
    if epipolar_pairs:
        return DegeneracyKind.D8_EPIPOLAR_PLANE_MULTIPLICITY, evidence

    # 5. the remaining numeric depth conditioning
    angles = []
    for point in cloud:
        ray_a = point - first
        ray_b = point - second
        try:
            angles.append(_angle(ray_a, ray_b))
        except ValueError:
            pass
    evidence["min_ray_angle_degrees"] = min(angles) if angles else float("nan")
    if angles and min(angles) < ray_angle_threshold_degrees:
        return DegeneracyKind.D4_PARALLEL_RAYS, evidence

    # 5. benign: symmetry, reported only when nothing worse applies.
    symmetric = _symmetric_pairs(cloud)
    evidence["n_symmetric_pairs"] = len(symmetric)
    evidence["symmetric_pairs"] = symmetric
    if symmetric:
        return DegeneracyKind.D6_SYMMETRIC_OBJECT, evidence

    return DegeneracyKind.OK, evidence
