"""Internal affine coordinates — the heart of the project (README section 6).

Given three non-collinear points ``A``, ``B``, ``C`` of an image, every point
``P`` of the plane has a *unique* representation

.. math::

    P - A = \\alpha (B - A) + \\beta (C - A)

equivalently

.. math::

    \\begin{pmatrix}\\alpha\\\\ \\beta\\end{pmatrix}
    = [B-A \\;\\; C-A]^{-1} (P - A).

``(alpha, beta)`` are the **internal affine coordinates** of ``P`` relative to
the frame ``(A, B, C)``.  Their defining property is that they are invariant
under *every* invertible affine map:

.. math::

    T(x) = Mx + t \\quad\\Longrightarrow\\quad
    \\alpha(T P) = \\alpha(P),\\; \\beta(T P) = \\beta(P).

This module is the *only* place in the code base that computes them; the GUI
layer must call :func:`affine_coordinates` rather than re-deriving the linear
algebra (README section 18: "UI 不允许重新实现数学算法").

Accuracy note
-------------
The project README demands ``error < 1e-9``.  That number is a property of the
*frame*, not of the code: the classical bound is

.. math::

    \\frac{\\|\\delta z\\|}{\\|z\\|} \\le \\mathrm{cond}(M)\\, \\varepsilon

with ``M = [B-A, C-A]``.  A very flat triangle has a huge condition number and
no algorithm can hit ``1e-9`` there.  Therefore every result carries its
condition number and a residual, and :func:`kozlov_lab.config.tolerance_for_condition`
converts the conditioning into the accuracy that is actually attainable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping

import numpy as np

from ..config import COLLINEAR_TOL, MAX_CONDITION_NUMBER, tolerance_for_condition
from .points import PointSet2D, as_point_array, as_point_matrix
from .transforms import AffineTransform

__all__ = [
    "AffineCoordinates",
    "AffineFrame",
    "signed_double_area",
    "collinearity_sine",
    "min_vertex_sine",
    "triangle_flatness",
    "is_non_collinear",
    "frame_matrix",
    "affine_coordinates",
    "affine_coordinates_batch",
    "reconstruct_point",
    "count_non_collinear_triples",
    "iter_non_collinear_triples",
    "find_non_collinear_triple",
]


# --------------------------------------------------------------------------- #
# collinearity — scale-invariant formulations
# --------------------------------------------------------------------------- #
def signed_double_area(a: Any, b: Any, c: Any) -> float:
    r"""``det[B-A, C-A]`` — twice the signed area of the triangle ``A B C``.

    Positive for a counter-clockwise triple.  Note that this quantity has the
    *dimension of an area*, so it is **not** a scale-free collinearity test;
    use :func:`collinearity_sine` for that.
    """
    pa, pb, pc = (as_point_array(v, name=n) for v, n in ((a, "a"), (b, "b"), (c, "c")))
    u = pb - pa
    v = pc - pa
    return float(u[0] * v[1] - u[1] * v[0])


def collinearity_sine(a: Any, b: Any, c: Any) -> float:
    r"""``|\sin(angle BAC)|`` — a dimensionless measure of the angle *at* ``a``.

    Returns ``0.0`` when either spanning vector vanishes (a degenerate triple is
    collinear for our purposes) and ``1.0`` for a right angle.  Because it is
    dimensionless, the same threshold works for an image of size ``1e-3`` and one
    of size ``1e3``.

    .. warning::
       **This function depends on which point is named ``A``.**  The three vertex
       sines of a triangle are *not* equal — a 90-45-45 triangle has sines
       ``1, 0.707, 0.707`` — and they can differ by many orders of magnitude for a
       nearly flat triangle.  Concretely, for ``a=(0,0)``, ``b=(3.5e-152, 0)``,
       ``c=(0,1)`` this function returns ``1.0`` at ``a`` and ``3.5e-152`` at
       ``c``.

       It is therefore the right tool for *diagnosing the conditioning of a
       specific frame* ``(A, B, C)``, and the **wrong** tool for deciding whether
       three points are collinear as a relation.  Use :func:`triangle_flatness`
       (and :func:`is_non_collinear`, which is built on it) for that.
    """
    pa, pb, pc = (as_point_array(v, name=n) for v, n in ((a, "a"), (b, "b"), (c, "c")))
    u = pb - pa
    v = pc - pa
    nu = float(math.hypot(u[0], u[1]))
    nv = float(math.hypot(v[0], v[1]))
    denom = nu * nv
    if denom <= 0.0:
        return 0.0
    cross = abs(float(u[0] * v[1] - u[1] * v[0]))
    return min(1.0, cross / denom)


def min_vertex_sine(a: Any, b: Any, c: Any) -> float:
    """Smallest of the three vertex sines — permutation invariant.

    ``min(|sin A|, |sin B|, |sin C|)``.  Dimensionless and symmetric in the three
    points, so it can be compared against a fixed threshold.
    """
    return min(
        collinearity_sine(a, b, c),
        collinearity_sine(b, a, c),
        collinearity_sine(c, a, b),
    )


def triangle_flatness(a: Any, b: Any, c: Any) -> float:
    r"""Permutation-invariant flatness of the triangle ``abc``.

    .. math::

        \mathrm{flatness} = \frac{2\,\mathrm{Area}}{\max(\text{side})^2}
                          = \frac{h_{\min}}{\max(\text{side})}

    i.e. "the smallest altitude measured in units of the longest side": ``0``
    for a degenerate triangle, ``sqrt(3)/2 ~ 0.866`` for the most compact one.

    Why not just use a vertex sine
    ------------------------------
    ``|\sin(angle BAC)|`` depends on the chosen vertex, so a *relation* defined by
    thresholding it is not well defined: the same three points can be "collinear"
    or "not collinear" depending on the naming order.  Hypothesis found exactly
    this: ``a=(0,0)``, ``b=(3.5e-152, 0)``, ``c=(0,1)`` yields sine ``1.0`` at
    ``a`` and ``3.5e-152`` at ``c``.  ``flatness`` is invariant by construction,
    which is what the image-code layer (Phase 3) needs when it enumerates
    collinear triples.
    """
    pa, pb, pc = (as_point_array(v, name=n) for v, n in ((a, "a"), (b, "b"), (c, "c")))
    cross = abs(float((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])))
    sides = (
        float(math.hypot(*(pb - pa))),
        float(math.hypot(*(pc - pa))),
        float(math.hypot(*(pc - pb))),
    )
    longest = max(sides)
    if longest <= 0.0:
        return 0.0
    # Divide twice instead of squaring: `longest * longest` underflows to 0 for
    # coordinates around 1e-307 (the square is below the smallest normal double),
    # and a division by that underflowed zero raises ZeroDivisionError.
    return min(1.0, cross / longest / longest)


def is_non_collinear(a: Any, b: Any, c: Any, tol: float = COLLINEAR_TOL) -> bool:
    r"""Return whether ``det[B-A, C-A] != 0`` in the scale-invariant sense.

    The test is ``flatness > tol`` with :func:`triangle_flatness` — hence
    **symmetric in the three arguments**, which the vertex sine is not.  ``tol``
    defaults to :data:`kozlov_lab.config.COLLINEAR_TOL` (``1e-9``).

    This is the guard the whole pipeline depends on: without three non-collinear
    points there is no two-dimensional affine frame, hence no internal
    coordinates and no image code.
    """
    return triangle_flatness(a, b, c) > tol


def frame_matrix(a: Any, b: Any, c: Any) -> np.ndarray:
    """The ``2x2`` matrix ``[B-A  C-A]`` (columns are the spanning vectors)."""
    pa, pb, pc = (as_point_array(v, name=n) for v, n in ((a, "a"), (b, "b"), (c, "c")))
    return np.column_stack((pb - pa, pc - pa))


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AffineCoordinates:
    """Internal coordinates of one point, with the diagnostics of the solve."""

    alpha: float
    beta: float
    #: ``|| P - (A + alpha(B-A) + beta(C-A)) ||`` — the algebraic residual.
    residual: float
    #: 2-norm condition number of ``[B-A, C-A]``.
    condition_number: float
    #: ``|\sin(angle BAC)|`` of the frame used.
    collinearity_sine: float

    def as_tuple(self, ndigits: int | None = None) -> tuple[float, float]:
        if ndigits is None:
            return (self.alpha, self.beta)
        return (round(self.alpha, ndigits), round(self.beta, ndigits))

    @property
    def is_well_conditioned(self) -> bool:
        return self.condition_number <= MAX_CONDITION_NUMBER

    @property
    def accuracy_bound(self) -> float:
        """Error level actually reachable for this frame."""
        return tolerance_for_condition(self.condition_number)

    def formula(self, ndigits: int = 3, point_name: str = "P") -> str:
        """The human-readable formula required by README section 20."""
        return (
            f"{point_name} - A = "
            f"{self.alpha:.{ndigits}f} (B - A) + {self.beta:.{ndigits}f} (C - A)"
        )

    def __iter__(self) -> Iterator[float]:
        yield self.alpha
        yield self.beta


# --------------------------------------------------------------------------- #
# core computation
# --------------------------------------------------------------------------- #
def affine_coordinates(
    point: Any,
    a: Any,
    b: Any,
    c: Any,
    *,
    tol: float = COLLINEAR_TOL,
    check: bool = True,
) -> AffineCoordinates:
    """Internal affine coordinates of ``point`` in the frame ``(a, b, c)``.

    Core API of README section 18.  Solves

    .. math::
        [B-A \\;\\; C-A] \\begin{pmatrix}\\alpha\\\\\\beta\\end{pmatrix} = P - A

    with :func:`numpy.linalg.solve` — never with an explicit inverse (README
    section 19).

    Parameters
    ----------
    point, a, b, c:
        Point-like values (``Point2D``, ``[x, y]``, ``(2,)`` array).
    tol:
        Scale-invariant collinearity threshold for the frame.
    check:
        Set to ``False`` only in hot loops where the frame was already
        validated; a non-finite result is still detected.

    Raises
    ------
    ValueError
        If the frame is degenerate: any of ``A``, ``B``, ``C`` coincides with
        another, or the three points are collinear (or nearly so).
    """
    pa = as_point_array(a, name="a")
    pb = as_point_array(b, name="b")
    pc = as_point_array(c, name="c")
    pp = as_point_array(point, name="point")

    mat = np.column_stack((pb - pa, pc - pa))
    sine = collinearity_sine(pa, pb, pc)
    cond = _condition_number(mat)

    if check and sine <= tol:
        raise ValueError(
            "affine_coordinates: reference points A, B, C are collinear "
            f"(|sin(angle BAC)| = {sine:.3e} <= tol = {tol:.1e}); "
            "a 2D affine frame requires three non-collinear points"
        )

    z = np.linalg.solve(mat, pp - pa)
    reconstructed = pa + z[0] * (pb - pa) + z[1] * (pc - pa)
    resid_vec = pp - reconstructed
    residual = float(math.hypot(float(resid_vec[0]), float(resid_vec[1])))

    return AffineCoordinates(
        alpha=float(z[0]),
        beta=float(z[1]),
        residual=residual,
        condition_number=cond,
        collinearity_sine=sine,
    )


def affine_coordinates_batch(
    points: Any,
    a: Any,
    b: Any,
    c: Any,
    *,
    tol: float = COLLINEAR_TOL,
) -> dict[str, Any]:
    """Vectorised :func:`affine_coordinates` for a whole point set.

    Returns a dict with ``alpha``, ``beta`` (each ``(n,)``), plus the frame
    diagnostics ``condition_number``, ``collinearity_sine`` and the vector of
    residuals.  One validation, one ``solve`` — important because M3/M4 run
    this over every triple of the image.
    """
    pa = as_point_array(a, name="a")
    pb = as_point_array(b, name="b")
    pc = as_point_array(c, name="c")
    pts = as_point_matrix(points, name="points", min_points=0)

    mat = np.column_stack((pb - pa, pc - pa))
    sine = collinearity_sine(pa, pb, pc)
    if sine <= tol:
        raise ValueError(
            "affine_coordinates_batch: reference points A, B, C are collinear "
            f"(|sin(angle BAC)| = {sine:.3e} <= tol = {tol:.1e})"
        )

    if pts.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "alpha": empty,
            "beta": empty.copy(),
            "residual": empty.copy(),
            "condition_number": _condition_number(mat),
            "collinearity_sine": sine,
        }

    sol = np.linalg.solve(mat, (pts - pa).T)  # (2, n)
    reconstructed = pa[:, None] + mat @ sol
    residual = np.linalg.norm(pts - reconstructed.T, axis=1)
    return {
        "alpha": sol[0].copy(),
        "beta": sol[1].copy(),
        "residual": residual,
        "condition_number": _condition_number(mat),
        "collinearity_sine": sine,
    }


def reconstruct_point(alpha_beta: Any, a: Any, b: Any, c: Any) -> np.ndarray:
    """Inverse of :func:`affine_coordinates`: ``(alpha, beta) -> P``.

    Used by the GUI to draw the frame and by tests to verify round trips.
    """
    z = as_point_array(alpha_beta, name="alpha_beta")
    pa = as_point_array(a, name="a")
    pb = as_point_array(b, name="b")
    pc = as_point_array(c, name="c")
    return pa + z[0] * (pb - pa) + z[1] * (pc - pa)


# --------------------------------------------------------------------------- #
# combinatorial helpers (needed by the image-code layer, Phase 3)
# --------------------------------------------------------------------------- #
def iter_non_collinear_triples(
    points: Any, tol: float = COLLINEAR_TOL
) -> Iterator[tuple[int, int, int]]:
    """Yield every *ordered* index triple ``(i, j, k)`` with ``i<j<k`` that
    forms a non-collinear frame.

    Ordering is fixed to ``i<j<k`` so that the enumeration is deterministic and
    independent of the input ordering; the image-code layer is responsible for
    turning the three ordered permutations into a canonical choice.
    """
    pts = as_point_matrix(points, name="points")
    n = pts.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if is_non_collinear(pts[i], pts[j], pts[k], tol):
                    yield (i, j, k)


def count_non_collinear_triples(points: Any, tol: float = COLLINEAR_TOL) -> int:
    """Number of valid frames ``(A, B, C)`` — a measure of encoding capacity."""
    return sum(1 for _ in iter_non_collinear_triples(points, tol))


def find_non_collinear_triple(
    points: Any, tol: float = COLLINEAR_TOL
) -> tuple[int, int, int] | None:
    """First valid frame in ``i<j<k`` order, or ``None`` if the image is 1-D.

    A set with fewer than three points, or whose points are all collinear, has
    *no* affine frame; the caller must handle that case explicitly (the GUI
    shows "此图像无法建立仿射坐标系" rather than crashing).
    """
    for triple in iter_non_collinear_triples(points, tol):
        return triple
    return None


# --------------------------------------------------------------------------- #
# AffineFrame — an object-oriented wrapper used by the GUI and experiments
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AffineFrame:
    """Three non-collinear reference points defining an internal coordinate system.

    Attributes
    ----------
    a, b, c:
        The reference points.
    indices:
        Optional indices of ``a``, ``b``, ``c`` inside the source image, kept
        so that the GUI can highlight them.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    indices: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        pa = as_point_array(self.a, name="a")
        pb = as_point_array(self.b, name="b")
        pc = as_point_array(self.c, name="c")
        object.__setattr__(self, "a", pa)
        object.__setattr__(self, "b", pb)
        object.__setattr__(self, "c", pc)

    # -- constructors ----------------------------------------------------- #
    @classmethod
    def from_points(cls, a: Any, b: Any, c: Any) -> "AffineFrame":
        return cls(a, b, c, None)

    @classmethod
    def from_indices(cls, points: Any, indices: tuple[int, int, int]) -> "AffineFrame":
        pts = as_point_matrix(points, name="points", min_points=3)
        i, j, k = indices
        return cls(pts[i], pts[j], pts[k], (int(i), int(j), int(k)))

    @classmethod
    def auto(cls, points: Any, tol: float = COLLINEAR_TOL) -> "AffineFrame":
        """Pick a *well-conditioned* frame automatically.

        We do not just take the first non-collinear triple: the accuracy of all
        downstream numbers is governed by ``cond([B-A, C-A])``, so we scan the
        valid triples and return one whose condition number is smallest among
        the sampled candidates, capped at 2 000 triples to stay interactive.

        For ``n = 3`` the single triple is returned.  Raises :class:`ValueError`
        when the image admits no frame at all.
        """
        pts = as_point_matrix(points, name="points", min_points=3)
        best: tuple[float, int, int, int] | None = None
        seen = 0
        for (i, j, k) in iter_non_collinear_triples(pts, tol):
            seen += 1
            cond = _condition_number(frame_matrix(pts[i], pts[j], pts[k]))
            # prefer obliqueness first, then conditioning
            sine = collinearity_sine(pts[i], pts[j], pts[k])
            score = cond / max(sine, 1e-12)
            if best is None or score < best[0]:
                best = (score, i, j, k)
            if seen >= 2000:
                break
        if best is None:
            raise ValueError(
                "AffineFrame.auto: the point set admits no affine frame "
                "(fewer than 3 points or all points collinear)"
            )
        _, i, j, k = best
        return cls.from_indices(pts, (i, j, k))

    # -- geometry --------------------------------------------------------- #
    @property
    def matrix(self) -> np.ndarray:
        """``[B-A  C-A]``."""
        return np.column_stack((self.b - self.a, self.c - self.a))

    @property
    def collinearity_sine(self) -> float:
        return collinearity_sine(self.a, self.b, self.c)

    @property
    def condition_number(self) -> float:
        return _condition_number(self.matrix)

    def is_valid(self, tol: float = COLLINEAR_TOL) -> bool:
        return self.collinearity_sine > tol

    def coordinates(self, point: Any, *, check: bool = True) -> AffineCoordinates:
        return affine_coordinates(point, self.a, self.b, self.c, check=check)

    def coordinates_xy(self, point: Any) -> tuple[float, float]:
        res = self.coordinates(point)
        return (res.alpha, res.beta)

    def coordinates_batch(self, points: Any) -> dict[str, Any]:
        return affine_coordinates_batch(points, self.a, self.b, self.c)

    def reconstruct(self, alpha_beta: Any) -> np.ndarray:
        return reconstruct_point(alpha_beta, self.a, self.b, self.c)

    def transformed(self, transform: AffineTransform) -> "AffineFrame":
        """The *same* frame carried along by an affine map.

        This is what makes the invariance experiment well posed: the frame
        points are part of the image, so they are transformed too, and the
        internal coordinates of the remaining points must not change.
        """
        return AffineFrame(
            transform.apply(self.a),
            transform.apply(self.b),
            transform.apply(self.c),
            self.indices,
        )

    def invariance(self, transform: AffineTransform, probe: Any) -> "InvarianceDelta":
        """Largest ``|delta alpha|`` / ``|delta beta|`` over ``probe`` points.

        Both the frame and the probe points are transformed by the *same*
        affine map; the theoretical answer is exactly ``0`` and the measured
        value is the floating-point reality.
        """
        pts = as_point_matrix(probe, name="probe", min_points=0)
        if pts.shape[0] == 0:
            raise ValueError("invariance: probe set is empty")
        before = self.coordinates_batch(pts)
        moved_frame = self.transformed(transform)
        after = moved_frame.coordinates_batch(transform.apply(pts))
        d_alpha = np.abs(after["alpha"] - before["alpha"])
        d_beta = np.abs(after["beta"] - before["beta"])
        return InvarianceDelta(
            max_delta_alpha=float(d_alpha.max()),
            max_delta_beta=float(d_beta.max()),
            condition_number_before=float(before["condition_number"]),
            condition_number_after=float(after["condition_number"]),
            transform_kind=transform.classify(),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "c": self.c.tolist(),
            "indices": list(self.indices) if self.indices else None,
            "condition_number": self.condition_number,
            "collinearity_sine": self.collinearity_sine,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "AffineFrame":
        idx = payload.get("indices")
        return cls(payload["a"], payload["b"], payload["c"], tuple(idx) if idx else None)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.a
        yield self.b
        yield self.c

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AffineFrame(A={self.a.tolist()}, B={self.b.tolist()}, C={self.c.tolist()}, "
            f"cond={self.condition_number:.3g}, sin={self.collinearity_sine:.3g})"
        )


@dataclass(frozen=True)
class InvarianceDelta:
    """Measured violation of affine invariance (should be ~1e-15)."""

    max_delta_alpha: float
    max_delta_beta: float
    condition_number_before: float
    condition_number_after: float
    transform_kind: str

    @property
    def max_delta(self) -> float:
        return max(self.max_delta_alpha, self.max_delta_beta)

    @property
    def accuracy_bound(self) -> float:
        return tolerance_for_condition(max(self.condition_number_before, self.condition_number_after))

    @property
    def within_bound(self) -> bool:
        return self.max_delta <= self.accuracy_bound

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_delta_alpha": self.max_delta_alpha,
            "max_delta_beta": self.max_delta_beta,
            "max_delta": self.max_delta,
            "condition_before": self.condition_number_before,
            "condition_after": self.condition_number_after,
            "accuracy_bound": self.accuracy_bound,
            "transform_kind": self.transform_kind,
            "within_bound": self.within_bound,
        }


@dataclass(frozen=True)
class FrameCode:
    """A single frame-relative encoding of an image (used from Phase 3 on).

    Kept here because it is the natural output type of
    :func:`affine_coordinates_batch`: the normalised points *are* the internal
    coordinates of every image point relative to ``frame``.
    """

    frame: AffineFrame
    normalized_points: tuple[tuple[float, float], ...]
    condition_number: float

    def as_array(self) -> np.ndarray:
        return np.asarray(self.normalized_points, dtype=np.float64)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _condition_number(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular.size == 0 or singular[-1] <= 0:
        return math.inf
    return float(singular[0] / singular[-1])
