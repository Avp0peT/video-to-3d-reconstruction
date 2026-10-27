"""Source 3D definitions, numerical code summaries and an affine geometric oracle.

Quoted definitions (Kozlov 1996, «Математическое моделирование зрительного восприятия»,
Математические вопросы кибернетики 6, 321–338) are in ``docs/specs/stereo_kozlov.md``:

* the body is a finite set of points in ``R^3``;
* the code is ``rho = V_mnuv / V_kspq`` over pairs of **quadruples**, with ``V`` the tetrahedron
  volume — the verbatim lift of the 2D triangle-area construction;
* projections are **parallel**, defined directly on S-1996 printed page 331;
* théorème 3 concerns the source's *volume images*, which exclude sets contained in one plane
  or in two parallel planes: code-equivalence **iff** 3D affine equivalence on that domain;
* theorem 4 concerns compatible, already labelled, a'-different parallel projections and
  AlgT' construction; it does not recover unknown correspondences or metric dimensions.

Degeneracy note that the source makes explicit: a quadruple with zero volume (four coplanar
points) cannot serve as a denominator, exactly as a collinear triple cannot in 2D.

``is_a_equivalent`` is a floating-point geometric search with a bijection witness. It does not
compare the indexed source codes, and its numerical outcomes are not proofs of théorème 3.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Any, Sequence

import numpy as np

__all__ = [
    "SOURCE_3D",
    "tetrahedron_volumes",
    "kozlov_body_code",
    "parallel_project",
    "is_a_equivalent",
    "correspondence_tetrahedron_principle",
]

SOURCE_3D = (
    "Козлов В. Н., «Математическое моделирование зрительного восприятия», Математические "
    "вопросы кибернетики, вып. 6, М.: Наука, 1996, с. 321–338 (primary full text)"
)


def _sextuple_volume(a, b, c, d) -> float:
    """``6V`` for the tetrahedron ``abcd`` — the determinant, which is six volumes."""
    u = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(a, dtype=float)
    w = np.asarray(d, dtype=float) - np.asarray(a, dtype=float)
    return float(np.dot(np.cross(u, v), w))


def tetrahedron_volumes(points: Any) -> dict[tuple[int, int, int, int], float]:
    """``V`` for every **unordered** quadruple (the source's «порядок не важен»).

    Degenerate quadruples are kept with volume 0 rather than dropped: they are legitimate
    *numerators* (giving ``rho = 0``) and only forbidden as denominators, exactly as in 2D.
    """
    body = np.asarray(points, dtype=float).reshape(-1, 3)
    return {
        quad: abs(_sextuple_volume(*(body[i] for i in quad))) / 6.0
        for quad in itertools.combinations(range(body.shape[0]), 4)
    }


def kozlov_body_code(points: Any, *, decimals: int = 9, degeneracy_tol: float = 1e-9) -> dict[str, Any]:
    """Legacy numerical summary of the body's tetrahedron-volume ratios.

    Returns counts and a deduplicated set of quantised values, NOT indexed records.
    For a complete bounded exact code use ``indexed_code3d.indexed_body_code``.
    ``is_a_equivalent`` remains a separate floating geometric oracle.

    Scale: quadruples grow like ``C(n,4)``, so the ratio set grows like ``C(n,4)^2`` — i.e.
    ``O(n^8)``, two orders above the 2D code's ``O(n^6)``.  That is a property of the definition
    and is reported rather than hidden.

    ``degeneracy_tol`` is a **numeric strategy, not part of the definition** — exactly the
    separation Phase 3 established.  The source says only that a zero-volume quadruple cannot be a
    denominator; it does not say how to test "zero" in floating point, and a literal ``V > 0`` is
    unusable there: a reconstruction of a body with exactly-coplanar quadruples perturbs them to
    ``V ~ 1e-17`` and admits quadruples the body itself excludes, so two affinely equivalent bodies
    get different denominators.  The test is therefore dimensionless,
    ``6V / max_edge³ > degeneracy_tol``, the 3D analogue of the 2D ``2·Area / max_side²``.
    """
    body = np.asarray(points, dtype=float).reshape(-1, 3)
    volumes = tetrahedron_volumes(body)
    admissible = []
    for quad, volume in volumes.items():
        corners = [body[i] for i in quad]
        longest = max(
            (float(np.linalg.norm(corners[i] - corners[j]))
             for i in range(4) for j in range(i + 1, 4)),
            default=0.0,
        )
        relative = 0.0 if longest <= 0 else (6.0 * volume) / longest**3
        if relative > degeneracy_tol:
            admissible.append(quad)
    admissible.sort()
    quantum = 10**decimals
    values: set[int] = set()
    for numerator in volumes:
        for denominator in admissible:
            if numerator == denominator:
                continue
            scaled = volumes[numerator] / volumes[denominator]
            values.add(int(round(scaled * quantum)))
    return {
        "n_points": int(body.shape[0]),
        "n_quadruples": len(volumes),
        "n_admissible_denominators": len(admissible),
        "n_degenerate_coplanar_quadruples": len(volumes) - len(admissible),
        "n_entries": len(admissible) * (len(volumes) - 1),
        "decimals": decimals,
        "degeneracy_tol": degeneracy_tol,
        "value_signature": tuple(sorted(values)),
        "source": SOURCE_3D,
    }


def parallel_project(points: Any, direction: Sequence[float], *, basis: Sequence[Sequence[float]] | None = None) -> np.ndarray:
    """Parallel projection along ``direction`` — **SOURCE-EXACT** (the paper's projection type).

    A parallel projection is an affine map onto a plane, so it is represented by two orthonormal
    vectors ``basis`` (the image plane) plus the shift along ``direction`` being discarded:

        p = ( (X - X0) . e1 , (X - X0) . e2 )

    No intrinsics, no focal length, no camera matrix: those are baseline concepts and the source
    has none of them.
    """
    body = np.asarray(points, dtype=float).reshape(-1, 3)
    d = np.asarray(direction, dtype=float).reshape(3)
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("the projection direction cannot be zero")
    d = d / norm
    if basis is None:
        # an arbitrary but deterministic orthonormal pair perpendicular to d
        helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(d, helper)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(d, e1)
    else:
        e1 = np.asarray(basis[0], dtype=float)
        e2 = np.asarray(basis[1], dtype=float)
    return np.column_stack([body @ e1, body @ e2])


def _point_bijection(moved: np.ndarray, target: np.ndarray, atol: float) -> list[int] | None:
    """Match every source row to a distinct target row within coordinate tolerance.

    Nearest neighbours (even in both directions) need not form a bijection. Augmenting paths
    also avoid rejecting a valid matching just because an earlier row consumed its only target.
    """
    admissible = np.max(np.abs(moved[:, None, :] - target[None, :, :]), axis=2) <= atol
    if not admissible.any(axis=1).all() or not admissible.any(axis=0).all():
        return None
    neighbours = [np.flatnonzero(row).tolist() for row in admissible]
    owners = [-1] * len(target)

    def augment(source_index: int, seen: set[int]) -> bool:
        for target_index in neighbours[source_index]:
            if target_index in seen:
                continue
            seen.add(target_index)
            if owners[target_index] == -1 or augment(owners[target_index], seen):
                owners[target_index] = source_index
                return True
        return False

    for source_index in range(len(moved)):
        if not augment(source_index, set()):
            return None
    permutation = [-1] * len(moved)
    for target_index, source_index in enumerate(owners):
        permutation[source_index] = target_index
    return permutation


def is_a_equivalent(first: Any, second: Any, *, tol: float = 1e-9) -> dict[str, Any]:
    """Numerically test affine equivalence of small unordered 3D point sets.

    Fix one non-coplanar source quadruple, enumerate ordered target quadruples, solve their affine
    maps, and verify a bijection of *all* points. ``permutation[i]`` identifies the row of ``second``
    matched to ``first[i]`` by the returned matrix and translation. This is an exhaustive geometric
    oracle for small full-rank inputs, not an indexed-code comparison or an exact arithmetic proof.

    ``tol`` is a numerical strategy: controls require absolute sextuple volume greater than it,
    and matched coordinate residuals must be at most ``tol * max(max(abs(second)), 1)``. Near
    degeneracy and poorly conditioned data therefore remain tolerance dependent. If neither body
    has an admissible control, return ``equivalent=None, undecided=True`` rather than a positive
    claim; lower-dimensional affine equivalence is outside this oracle's scope.
    """
    a = np.asarray(first, dtype=float).reshape(-1, 3)
    b = np.asarray(second, dtype=float).reshape(-1, 3)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("body coordinates must be finite")
    if a.shape != b.shape:
        return {"equivalent": False, "reason": f"shapes differ: {a.shape} vs {b.shape}"}

    def find_control(body: np.ndarray) -> tuple[int, ...] | None:
        return next((quad for quad in itertools.combinations(range(len(body)), 4)
                     if abs(_sextuple_volume(*(body[i] for i in quad))) > tol), None)

    control = find_control(a)
    target_control = find_control(b)
    if control is None and target_control is None:
        return {"equivalent": None, "undecided": True,
                "reason": "neither body has a non-coplanar control at this tolerance"}
    if control is None or target_control is None:
        return {"equivalent": False,
                "reason": "only one body has a non-coplanar control at this tolerance"}
    source = a[list(control)]
    design = np.hstack([source, np.ones((4, 1))])
    tolerance = tol * max(float(np.abs(b).max()), 1.0)
    for ordered in itertools.permutations(range(b.shape[0]), 4):
        target = np.array([b[i] for i in ordered])
        if abs(_sextuple_volume(*target)) <= tol:
            continue
        # Solve for the affine map directly: design @ solution = target with
        # design = [s_i | 1] (4x4) and solution = [[A^T], [t]] (4x3), so A = solution[:3].T.
        # A first version built the wrong system and produced a 3x4 "matrix"; the shape error
        # surfaced immediately, which is why the solve is done in one explicit step here.
        try:
            solution = np.linalg.solve(design, target)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(solution).all():
            continue
        matrix = solution[:3].T
        translation = solution[3]
        moved = a @ matrix.T + translation
        permutation = _point_bijection(moved, b, tolerance)
        if permutation is not None:
            residual = float(np.abs(moved - b[permutation]).max())
            return {"equivalent": True, "matrix": matrix.tolist(),
                    "translation": translation.tolist(), "permutation": permutation,
                    "max_residual": residual, "tolerance": tolerance}
    return {"equivalent": False, "reason": "no affine bijection found at this tolerance"}


def correspondence_tetrahedron_principle(points_2d: Any, *, tol: float = 1e-9) -> dict[str, Any]:
    """Legacy name: count single-view noncollinear triples and describe prerequisites.

    This does not establish correspondence or certify source anchors. S-1996 p332 defines
    off-plane using TWO already matched views: align abc and compare the matched fourth point.
    That point may lie inside the drawn 2D triangle. AlgT'/AlgSo assume the matching is given;
    unknown-label acceptance is a separate procedure on p338. See stereo_algorithms_1996.md.
    """
    image = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    n = image.shape[0]
    non_degenerate_triples = []
    for triple in itertools.combinations(range(n), 3):
        a, b, c = (image[i] for i in triple)
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 > tol:
            non_degenerate_triples.append(triple)
    return {
        "n_points": n,
        "n_non_degenerate_triples": len(non_degenerate_triples),
        "requires": (
            "two compatible labelled parallel projections with abc noncollinear in both; "
            "after aligning matched abc, the fourth matched point e must not coincide. "
            "This two-view condition is not checked by this single-view triangle counter; "
            "e need not lie outside the drawn triangle and correspondence is not recovered"
        ),
        "source": SOURCE_3D,
    }
