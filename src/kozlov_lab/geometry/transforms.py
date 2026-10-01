"""Geometric transformations of finite point sets (README section 5).

The four families required by the curriculum, from weakest to strongest:

============  ==============================================  ====================
name          action on a point                               preserved up to
============  ==============================================  ====================
translation   ``x' = x + t``                                   isometry
isometry      ``x' = R x + t``, ``R`` orthogonal, ``det R = 1``  angles, distances
similarity    ``x' = s R x + t``, ``s > 0``                     angles, length ratios
affine        ``x' = A x + t``, ``det A != 0``                   collinearity, parallelism, affine coordinates
============  ==============================================  ====================

A single :class:`AffineTransform` represents all four; :meth:`AffineTransform.classify`
tells you which of them a concrete instance actually is.  This keeps the
"Invariant Explorer" (README section 9) honest: the user drags sliders, and the
laboratory *measures* which properties survived instead of asserting a table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..config import FLOAT_TOL
from .points import Point2D, PointSet2D, as_point_array, as_point_matrix

__all__ = [
    "AffineTransform",
    "IDENTITY",
    "rotation_matrix",
    "apply_affine",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rotation_matrix(degrees: float) -> np.ndarray:
    """Counter-clockwise rotation matrix of ``degrees`` degrees."""
    theta = math.radians(float(degrees))
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def apply_affine(points: Any, matrix: Any, translation: Any) -> np.ndarray:
    """Core API (README section 18): ``x -> A x + t`` for many points at once.

    Parameters
    ----------
    points:
        ``(n, 2)`` array-like, a sequence of ``[x, y]``, or a
        :class:`PointSet2D`.
    matrix:
        ``(2, 2)`` array-like.  Not required to be invertible here — the
        "is this a legal image transform?" question is answered by
        :func:`is_invertible` / :meth:`AffineTransform.is_invertible`.
    translation:
        Length-2 array-like.

    Returns
    -------
    numpy.ndarray
        New ``(n, 2)`` array; the input is never modified.
    """
    pts = as_point_matrix(points, name="points", min_points=0)
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (2, 2):
        raise ValueError(f"matrix: expected shape (2, 2), got {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("matrix: entries must be finite")
    vec = as_point_array(translation, name="translation")
    if pts.shape[0] == 0:
        return pts.copy()
    return pts @ mat.T + vec


# --------------------------------------------------------------------------- #
# AffineTransform
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class AffineTransform:
    """An affine map ``T(x) = A x + t`` on ``R^2``.

    The composition convention is "matrix-like": ``(f @ g)(x) = f(g(x))``,
    i.e. ``g`` is applied first.
    """

    matrix: np.ndarray
    translation: np.ndarray

    # -- construction ----------------------------------------------------- #
    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=np.float64)
        if mat.shape != (2, 2):
            raise ValueError(f"matrix: expected shape (2, 2), got {mat.shape}")
        if not np.all(np.isfinite(mat)):
            raise ValueError("matrix: entries must be finite")
        vec = np.asarray(self.translation, dtype=np.float64)
        if vec.shape == (2, 1):
            vec = vec.reshape(2)
        if vec.shape != (2,):
            raise ValueError(f"translation: expected shape (2,), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("translation: entries must be finite")
        object.__setattr__(self, "matrix", np.ascontiguousarray(mat))
        object.__setattr__(self, "translation", np.ascontiguousarray(vec))

    @classmethod
    def identity(cls) -> "AffineTransform":
        return cls(np.eye(2), np.zeros(2))

    @classmethod
    def translation_of(cls, tx: float, ty: float) -> "AffineTransform":
        """Pure translation ``x' = x + t``."""
        return cls(np.eye(2), [tx, ty])

    @classmethod
    def rotation(cls, degrees: float, center: Any | None = None) -> "AffineTransform":
        """Rotation by ``degrees`` CCW, optionally about ``center``."""
        mat = rotation_matrix(degrees)
        if center is None:
            return cls(mat, np.zeros(2))
        c = as_point_array(center, name="center")
        return cls(mat, c - mat @ c)

    @classmethod
    def scaling(
        cls,
        sx: float,
        sy: float | None = None,
        center: Any | None = None,
    ) -> "AffineTransform":
        """Axis scaling; ``sy`` defaults to ``sx`` (uniform scaling)."""
        sy = sx if sy is None else sy
        mat = np.diag([float(sx), float(sy)])
        if center is None:
            return cls(mat, np.zeros(2))
        c = as_point_array(center, name="center")
        return cls(mat, c - mat @ c)

    @classmethod
    def shear(cls, kx: float = 0.0, ky: float = 0.0) -> "AffineTransform":
        """Shear with ``x' = x + kx*y``, ``y' = ky*x + y``.

        The matrix is ``[[1, kx], [ky, 1]]`` and is invertible whenever
        ``kx * ky != 1``.
        """
        mat = np.array([[1.0, float(kx)], [float(ky), 1.0]], dtype=np.float64)
        return cls(mat, np.zeros(2))

    @classmethod
    def similarity(
        cls,
        scale: float = 1.0,
        degrees: float = 0.0,
        center: Any | None = None,
    ) -> "AffineTransform":
        """``x' = s R x + t`` with ``s > 0``."""
        if scale <= 0:
            raise ValueError("similarity scale must be > 0")
        mat = float(scale) * rotation_matrix(degrees)
        if center is None:
            return cls(mat, np.zeros(2))
        c = as_point_array(center, name="center")
        return cls(mat, c - mat @ c)

    @classmethod
    def from_control_points(cls, source: Any, target: Any) -> "AffineTransform":
        """Exact affine map sending three non-collinear points onto three points.

        This is "method B" of README section 8 in its most basic form, and is
        also the natural way to check an affine-invariance claim by hand: the
        map is *determined* by the triple.

        Raises
        ------
        ValueError
            If the source triple is collinear (no unique affine map exists) or
            the shapes do not match.
        """
        src = as_point_matrix(source, name="source", min_points=3)
        dst = as_point_matrix(target, name="target", min_points=3)
        if src.shape[0] != 3 or dst.shape[0] != 3:
            raise ValueError("from_control_points needs exactly 3 source and 3 target points")
        basis = (src[1:] - src[0]).T  # columns: B-A, C-A
        if abs(float(np.linalg.det(basis))) <= FLOAT_TOL * _basis_scale(basis):
            raise ValueError("source control points are collinear; affine map is not unique")
        images = (dst[1:] - dst[0]).T  # columns: B'-A', C'-A'
        mat = np.linalg.solve(basis.T, images.T).T
        return cls(mat, dst[0] - mat @ src[0])

    @classmethod
    def random_invertible(
        cls,
        rng: np.random.Generator | int | None = None,
        *,
        scale_range: tuple[float, float] = (0.5, 2.0),
        rotation_range: tuple[float, float] = (0.0, 360.0),
        shear_range: tuple[float, float] = (-0.5, 0.5),
        translation_range: tuple[float, float] = (-5.0, 5.0),
        allow_reflection: bool = False,
        max_condition: float = 50.0,
    ) -> "AffineTransform":
        """Random well-conditioned affine map — the workhorse of Experiment A/B.

        Built as ``A = R2 @ diag(s1, s2) @ R1`` (plus optional shear) rather
        than by drawing four i.i.d. entries: the singular-value construction
        lets us *bound the condition number*, which is exactly the quantity
        that controls the achievable accuracy of the internal coordinates.

        Parameters
        ----------
        max_condition:
            Upper bound on ``cond(A)``.  It is enforced by clamping the singular
            values, so the bound holds exactly rather than "usually".

        Notes
        -----
        Conditioning is not a detail here: ``cond(A)`` multiplies the frame's own
        condition number, and together they set the accuracy of every internal
        coordinate.  Drawing four i.i.d. matrix entries would produce nearly
        singular maps with non-negligible probability and silently turn
        Experiment A into an experiment about ill-conditioning.
        """
        g = np.random.default_rng(rng)

        def _u(a: tuple[float, float]) -> float:
            return float(g.uniform(a[0], a[1]))

        s1 = _u(scale_range)
        ratio = float(g.uniform(1.0 / max_condition, 1.0))
        s2 = s1 * ratio
        r1 = rotation_matrix(_u(rotation_range))
        r2 = rotation_matrix(_u(rotation_range))
        mat = r2 @ np.diag([s1, s2]) @ r1

        kx, ky = _u(shear_range), _u(shear_range)
        if abs(kx) > FLOAT_TOL or abs(ky) > FLOAT_TOL:
            if abs(kx * ky - 1.0) > 1e-3:  # keep the shear invertible
                mat = mat @ np.array([[1.0, kx], [ky, 1.0]], dtype=np.float64)

        if allow_reflection and bool(g.integers(0, 2)):
            mat = mat @ np.diag([-1.0, 1.0])

        # Enforce the condition-number budget by *clamping the singular values*.
        #
        # Note the trap: dividing the whole matrix by a scalar does nothing to the
        # condition number, because cond = s_max / s_min and the scale cancels.
        # The budget is a constraint on the *ratio* of the singular values, so the
        # only correct fix is to raise the smallest one (or lower the largest).
        # This matters because cond(A) is exactly the quantity that governs the
        # achievable accuracy of the internal coordinates.
        u, singular, vt = np.linalg.svd(mat)
        if singular[-1] <= 0 or not np.isfinite(singular[0] / singular[-1]):
            raise RuntimeError("random_invertible produced a singular matrix")  # pragma: no cover
        if singular[0] / singular[-1] > max_condition:
            singular = np.array(
                [singular[0], max(singular[-1], singular[0] / max_condition)], dtype=np.float64
            )
            mat = (u * singular) @ vt

        t = np.array([_u(translation_range), _u(translation_range)], dtype=np.float64)
        return cls(mat, t)

    # -- action ----------------------------------------------------------- #
    def apply(self, points: Any) -> np.ndarray:
        """Apply the map; the output shape follows the input shape.

        * a single point (:class:`Point2D`, ``[x, y]``, a ``(2,)`` array) gives a
          ``(2,)`` vector;
        * a collection of points gives an ``(n, 2)`` array.

        Vectorised callers that always want a 2-D result should use
        :func:`apply_affine` directly, which is unconditionally ``(n, 2)``.
        """
        single = _is_single_point(points)
        out = apply_affine(points, self.matrix, self.translation)
        return out[0] if single else out

    def apply_set(self, point_set: PointSet2D, *, name: str | None = None) -> PointSet2D:
        """Apply the map to a :class:`PointSet2D`, returning a new point set.

        Metadata is preserved and the label defaults to ``"T(<original>)"``.
        """
        if not isinstance(point_set, PointSet2D):  # pragma: no cover - defensive
            raise TypeError("apply_set expects a PointSet2D")
        label = name if name is not None else (
            f"T({point_set.name})" if point_set.name else None
        )
        meta = {**point_set.meta, "apply_affine_matrix": self.matrix.tolist(),
                "apply_affine_translation": self.translation.tolist()}
        return PointSet2D(self.apply(point_set.as_array()), name=label, meta=meta)

    def __call__(self, points: Any) -> np.ndarray:
        return self.apply(points)

    # -- algebra ---------------------------------------------------------- #
    def compose(self, inner: "AffineTransform") -> "AffineTransform":
        """Return ``self ∘ inner`` (``inner`` applied first)."""
        return AffineTransform(
            self.matrix @ inner.matrix,
            self.matrix @ inner.translation + self.translation,
        )

    def __matmul__(self, inner: "AffineTransform") -> "AffineTransform":
        return self.compose(inner)

    def inverse(self) -> "AffineTransform":
        """Inverse map, computed with ``solve`` rather than ``inv`` (README section 19)."""
        if not self.is_invertible:
            raise ValueError("transform is singular and cannot be inverted")
        inv_mat = np.linalg.solve(self.matrix, np.eye(2))
        return AffineTransform(inv_mat, -inv_mat @ self.translation)

    @property
    def det(self) -> float:
        return float(np.linalg.det(self.matrix))

    @property
    def is_invertible(self) -> bool:
        return abs(self.det) > _relative_det_floor(self.matrix)

    @property
    def singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.matrix, compute_uv=False)

    @property
    def condition_number(self) -> float:
        """2-norm condition number — the accuracy budget of downstream solves."""
        singular = self.singular_values
        if singular[-1] <= 0:
            return math.inf
        return float(singular[0] / singular[-1])

    @property
    def scale_factor(self) -> float:
        """``sqrt(|det A|)``: the area-equivalent length factor."""
        return math.sqrt(abs(self.det))

    # -- classification --------------------------------------------------- #
    def classify(self, tol: float = 1e-9) -> str:
        """Return the *strongest* family this transform belongs to.

        One of ``"translation"``, ``"identity"``, ``"rotation"``,
        ``"reflection"``, ``"isometry"``, ``"similarity"``, ``"affine"``,
        ``"singular"``.

        The classification is deliberately based on the *linear part* only,
        because that is what decides which invariants survive: a rotation about
        a point far from the origin is still an isometry even though its
        translation vector is large.
        """
        if not self.is_invertible:
            return "singular"
        mat = self.matrix
        is_orthogonal = np.allclose(mat @ mat.T, np.eye(2), atol=tol, rtol=0)
        if is_orthogonal:
            if self._is_identity_linear(tol):
                return "identity" if self._is_zero_translation(tol) else "translation"
            if float(np.linalg.det(mat)) > 0:
                return "rotation"
            return "reflection"
        # orthogonal up to a positive scalar?
        scale = self.scale_factor
        if scale > 0 and np.allclose(mat @ mat.T, (scale**2) * np.eye(2), atol=tol * max(1.0, scale**2), rtol=0):
            return "similarity"
        return "affine"

    def invariant_group(self) -> str:
        """Coarse group name used by the invariant table (README section 9)."""
        kind = self.classify()
        if kind in {"identity", "translation", "rotation", "reflection", "isometry"}:
            return "isometry"
        if kind == "similarity":
            return "similarity"
        if kind == "singular":
            return "singular"
        return "affine"

    def rotation_angle_degrees(self) -> float:
        """Signed rotation angle of the linear part (``0`` when undefined)."""
        angle = math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0]))
        return float(angle)

    def _is_identity_linear(self, tol: float) -> bool:
        return bool(np.allclose(self.matrix, np.eye(2), atol=tol, rtol=0))

    def _is_zero_translation(self, tol: float) -> bool:
        return bool(np.allclose(self.translation, 0.0, atol=tol, rtol=0))

    # -- UI support ------------------------------------------------------- #
    def parameters(self) -> dict[str, float]:
        """Decomposed parameters for the parameter sliders in the GUI.

        Decomposition: ``A = R(rotation_deg) @ Shear(kx, ky) @ diag(sx, sy)``
        is *not* used; instead we report the entries that a user actually
        recognises, plus a few derived diagnostics.
        """
        return {
            "a11": float(self.matrix[0, 0]),
            "a12": float(self.matrix[0, 1]),
            "a21": float(self.matrix[1, 0]),
            "a22": float(self.matrix[1, 1]),
            "tx": float(self.translation[0]),
            "ty": float(self.translation[1]),
            "rotation_deg": self.rotation_angle_degrees(),
            "scale_factor": self.scale_factor,
            "det": self.det,
            "condition_number": self.condition_number,
        }

    def as_matrix3(self) -> np.ndarray:
        """Homogeneous ``3x3`` matrix form."""
        out = np.eye(3)
        out[:2, :2] = self.matrix
        out[:2, 2] = self.translation
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return bool(
            np.allclose(self.matrix, other.matrix, atol=FLOAT_TOL, rtol=0)
            and np.allclose(self.translation, other.translation, atol=FLOAT_TOL, rtol=0)
        )

    def __hash__(self) -> int:
        return hash((self.matrix.tobytes(), self.translation.tobytes()))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "AffineTransform("
            f"A={np.array2string(self.matrix, precision=4)}, "
            f"t={np.array2string(self.translation, precision=4)}, "
            f"kind={self.classify()})"
        )

    # -- serialisation ---------------------------------------------------- #
    def to_json(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.tolist(),
            "translation": self.translation.tolist(),
            "kind": self.classify(),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "AffineTransform":
        return cls(payload["matrix"], payload["translation"])


IDENTITY = AffineTransform.identity()


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _is_single_point(value: Any) -> bool:
    """True when ``value`` denotes one point rather than a collection of points.

    Needed because ``[[0, 0]]`` (one point) and ``[0, 0]`` (two 1-D points, which
    is *not* valid input here) are distinguishable, while ``[0, 0]`` (one 2-D
    point) and ``[[0, 0], [1, 1]]`` (two points) must be told apart reliably.
    """
    if isinstance(value, Point2D):
        return True
    if isinstance(value, np.ndarray):
        return value.shape in {(2,), (1, 2)}
    if isinstance(value, Mapping):
        return False
    try:
        items = list(value)
    except TypeError:
        return False
    if len(items) != 2:
        return False
    return not isinstance(items[0], (list, tuple, np.ndarray, Point2D))


def _basis_scale(basis: np.ndarray) -> float:
    """Product of the column norms — makes the collinearity test scale-free."""
    norms = np.linalg.norm(basis, axis=0)
    scale = float(norms[0] * norms[1])
    return scale if scale > 0 else 1.0


def _relative_det_floor(matrix: np.ndarray) -> float:
    """A scale-invariant "is this matrix singular?" threshold."""
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] <= 0:
        return FLOAT_TOL
    return FLOAT_TOL * float(singular[0]) ** 2
