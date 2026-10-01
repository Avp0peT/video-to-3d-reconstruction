"""The 3D body, its code, parallel projection and ``a'``-equivalence — **SOURCE-EXACT**.

Quoted definitions (Kozlov 1996, «Математическое моделирование зрительного восприятия»,
Математические вопросы кибернетики 6, 321–338) are in ``docs/specs/stereo_kozlov.md``:

* the body is a finite set of points in ``R^3``;
* the code is ``rho = V_mnuv / V_kspq`` over pairs of **quadruples**, with ``V`` the tetrahedron
  volume — the verbatim lift of the 2D triangle-area construction;
* projections are **parallel** (the paper says «проекции по разным направлениям», and the word for
  central/perspective projection never occurs);
* théorème 3: two bodies are code-equivalent **iff** they are 3D affinely equivalent;
* théorème 4: a body rebuilt from two differently-directed projections is ``a'``-equivalent to the
  original.

Degeneracy note that the source makes explicit: a quadruple with zero volume (four coplanar
points) cannot serve as a denominator, exactly as a collinear triple cannot in 2D.
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
    """The body's code ``<M_A, T_A>``: indexed ratios of tetrahedron volumes.

    Returns the counts and the set of quantised ratio values.  The **indexing is preserved** for
    the same reason as in 2D: the equivalence relation acts on it.

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


def is_a_equivalent(first: Any, second: Any, *, tol: float = 1e-9) -> dict[str, Any]:
    """Whether two bodies are 3D affinely equivalent — the equivalence of théorème 3.

    Decided by solving for the affine map from four non-coplanar control points and checking the
    remaining points, the 3D lift of the Phase 3 geometric oracle.  Returns the map when it exists,
    so the caller gets a witness rather than a bare ``True``.
    """
    a = np.asarray(first, dtype=float).reshape(-1, 3)
    b = np.asarray(second, dtype=float).reshape(-1, 3)
    if a.shape != b.shape:
        return {"equivalent": False, "reason": f"shapes differ: {a.shape} vs {b.shape}"}
    control = None
    for quad in itertools.combinations(range(a.shape[0]), 4):
        if abs(_sextuple_volume(*(a[i] for i in quad))) > tol:
            control = quad
            break
    if control is None:
        return {"equivalent": True, "undecided": True, "reason": "every quadruple of A is coplanar"}
    source = np.array([a[i] for i in control])
    target_all = list(itertools.permutations(range(b.shape[0]), 4))
    for ordered in target_all:
        target = np.array([b[i] for i in ordered])
        if abs(_sextuple_volume(*target)) <= tol:
            continue
        # Solve for the affine map directly: design @ solution = target with
        # design = [s_i | 1] (4x4) and solution = [[A^T], [t]] (4x3), so A = solution[:3].T.
        # A first version built the wrong system and produced a 3x4 "matrix"; the shape error
        # surfaced immediately, which is why the solve is done in one explicit step here.
        design = np.hstack([source, np.ones((4, 1))])
        try:
            solution = np.linalg.solve(design, target)
        except np.linalg.LinAlgError:
            continue
        matrix = solution[:3].T
        translation = solution[3]
        moved = a @ matrix.T + translation
        if np.abs(moved - b).max() <= 1e-9 * max(float(np.abs(b).max()), 1.0):
            return {"equivalent": True, "matrix": matrix.tolist(), "translation": translation.tolist()}
    return {"equivalent": False, "reason": "no affine map reproduces B from A"}


def correspondence_tetrahedron_principle(points_2d: Any, *, tol: float = 1e-9) -> dict[str, Any]:
    """The source's correspondence principle: a **tetrahedron** (triangle + off-plane point).

    > «Выберем на S1 и S2 точки a1,b1,c1,e1 и a2,b2,c2,e2 такие, что a1b1c1 и a2b2c2 —
    > треугольники и точка e1 лежит вне плоскости треугольника a1b1c1.»

    In a projection this is a quadruple of 2D points whose pre-images are **not coplanar**; the
    projection cannot detect that directly, so the principle is reported here as the structure the
    source requires, and the *detection* of such quadruples in a real pair of projections is the
    correspondence problem itself (6E).  Returning the requirement explicitly is more useful than
    pretending it is automatic.
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
            "a quadruple whose 2D projection is a triangle plus a point outside it, AND whose "
            "pre-image quadruple is non-coplanar; the second condition is not checkable from one "
            "projection alone, which is why the source calls correspondence the main problem"
        ),
        "source": SOURCE_3D,
    }
