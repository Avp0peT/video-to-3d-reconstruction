"""3D finite point sets and 3D affine maps (Phase 6B).

The 3D object is the direct lift of the 2D one, and the source says so: «Назовем
трехмерным изображением А или **телом** конечное множество точек в трехмерном евклидовом
пространстве.»  (Kozlov 1996, §2).  Everything here is the 3D counterpart of
``kozlov_lab.geometry``; none of it is new mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["Point3D", "PointSet3D", "AffineTransform3D", "build_scene", "scene_names"]


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class PointSet3D:
    """A finite set of points in R^3, with **set** semantics (no repeated points).

    Kept deliberately parallel to :class:`~kozlov_lab.geometry.points.PointSet2D`, including
    the relabelling-invariance property: storage order is not part of the object.
    """

    def __init__(self, points: Any, name: str = "") -> None:
        array = np.asarray(points, dtype=float)
        if array.ndim == 1 and array.shape == (3,):
            array = array.reshape(1, 3)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"expected shape (n, 3), got {array.shape}")
        if array.shape[0] == 0:
            raise ValueError("a body must be non-empty")
        self._points = array
        self.name = name

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def n(self) -> int:
        return int(self._points.shape[0])

    def as_array(self) -> np.ndarray:
        return self._points.copy()

    def centroid(self) -> np.ndarray:
        return self._points.mean(axis=0)

    def diameter(self) -> float:
        if self.n < 2:
            return 0.0
        differences = self._points[:, None, :] - self._points[None, :, :]
        return float(np.sqrt((differences**2).sum(axis=-1)).max())

    def is_coplanar(self, tol: float = 1e-9) -> bool:
        """Whether every point lies in one plane (the 3D analogue of "all collinear")."""
        if self.n <= 3:
            return True
        centred = self._points - self._points.mean(axis=0)
        scale = float(np.abs(centred).max()) or 1.0
        singular = np.linalg.svd(centred, compute_uv=False)
        return bool(singular[2] / scale <= tol)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointSet3D):
            return NotImplemented
        if self.n != other.n:
            return False
        return {tuple(row) for row in self._points} == {tuple(row) for row in other._points}

    def signature(self) -> tuple:
        return tuple(sorted(tuple(row) for row in self._points))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PointSet3D(n={self.n}, name={self.name!r})"


@dataclass(frozen=True)
class AffineTransform3D:
    """``x -> A x + t`` with ``A`` invertible — the group of théorème 3's equivalence."""

    matrix: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"matrix must be 3x3, got {matrix.shape}")
        translation = np.asarray(self.translation, dtype=float).reshape(3)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "translation", translation)

    @property
    def det(self) -> float:
        return float(np.linalg.det(self.matrix))

    @property
    def is_invertible(self) -> bool:
        return abs(self.det) > 1e-12

    def apply(self, points: Any) -> np.ndarray:
        array = np.asarray(points, dtype=float)
        single = array.ndim == 1
        if single:
            array = array.reshape(1, 3)
        moved = array @ self.matrix.T + self.translation
        return moved[0] if single else moved

    # -- named families --------------------------------------------------- #
    @staticmethod
    def translation_of(dx: float, dy: float, dz: float) -> "AffineTransform3D":
        return AffineTransform3D(np.eye(3), [dx, dy, dz])

    @staticmethod
    def scaling(sx: float, sy: float, sz: float) -> "AffineTransform3D":
        return AffineTransform3D(np.diag([sx, sy, sz]), np.zeros(3))

    @staticmethod
    def rotation(axis: str, degrees: float) -> "AffineTransform3D":
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        if axis == "x":
            matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError("axis must be 'x', 'y' or 'z'")
        return AffineTransform3D(matrix, np.zeros(3))

    @staticmethod
    def shear(kxy: float = 0.0, kxz: float = 0.0, kyz: float = 0.0) -> "AffineTransform3D":
        return AffineTransform3D(
            np.array([[1, kxy, kxz], [0, 1, kyz], [0, 0, 1]]), np.zeros(3)
        )

    @staticmethod
    def random_invertible(seed: int, max_condition: float = 10.0) -> "AffineTransform3D":
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            matrix = rng.normal(size=(3, 3))
            singular = np.linalg.svd(matrix, compute_uv=False)
            if singular[-1] <= 0:
                continue
            if singular[0] / singular[-1] <= max_condition:
                return AffineTransform3D(matrix, rng.normal(size=3) * 2.0)
        raise RuntimeError("could not draw a well-conditioned matrix")


# --------------------------------------------------------------------------- #
# scenes (6B)
# --------------------------------------------------------------------------- #
def _cube(a: float = 2.0) -> np.ndarray:
    return np.array([(x * a, y * a, z * a) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)


def _cuboid() -> np.ndarray:
    return np.array(
        [(x * 3.0, y * 1.0, z * 1.5) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float
    )


def _tetrahedron() -> np.ndarray:
    return np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)


def _pyramid() -> np.ndarray:
    return np.array(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1.0, 1.0, 2.2]], dtype=float
    )


def _irregular(seed: int = 20261007, n: int = 8) -> np.ndarray:
    """An irregular body in general position (not coplanar, no gratuitous symmetry).

    The same discipline as Phase 3: a symmetric object makes correspondence ambiguous, which is
    a legitimate outcome but a poor default for testing whether reconstruction works at all.
    """
    rng = np.random.default_rng(seed)
    for _ in range(200):
        points = rng.normal(size=(n, 3)) * 1.5
        body = PointSet3D(points)
        if not body.is_coplanar(tol=1e-3):
            return points
    raise RuntimeError("could not draw a non-coplanar body")


SCENES: dict[str, Any] = {
    "cube": _cube,
    "cuboid": _cuboid,
    "tetrahedron": _tetrahedron,
    "pyramid": _pyramid,
    "irregular": _irregular,
}


def scene_names() -> list[str]:
    return sorted(SCENES)


def build_scene(name: str, **kwargs: Any) -> PointSet3D:
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; available: {', '.join(scene_names())}")
    return PointSet3D(SCENES[name](**kwargs), name=name)
