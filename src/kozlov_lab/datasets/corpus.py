"""Experiment datasets A–E — deliberately **asymmetric** point configurations.

Motivation (the reason this module exists)
------------------------------------------
The Phase 0–2 invariant experiments were validated on a single shape
(``square_3x``).  That is not enough, and the failure mode is specific:
**a symmetric shape can mask four different bugs**

1. *wrong point correspondence* — with a symmetry, several correspondences look
   equally good, so a broken matcher still scores 100%;
2. *wrong frame orientation* — a symmetric frame makes the wrong orientation
   numerically indistinguishable from the right one;
3. *permutation bug* — if the image is invariant under some relabelling, a
   permutation bug is invisible;
4. *reflection ambiguity* — ``det A = +1`` and ``det A = -1`` give the same
   result on a reflection-symmetric image, so a sign error passes.

Datasets A–E are therefore asymmetric **by construction**, and the asymmetry is
*measured* (see :func:`measure_asymmetry`) rather than assumed.

Two structural relations are also deliberate
--------------------------------------------
The invariant table has two *relation-valued* rows (collinearity, parallelism)
whose verdict is ``vacuous`` when the image contains no instance of the relation
(see ``docs/specs/invariants.md`` §2).  A dataset that silently produces vacuous
rows would "pass" without testing anything.  Datasets A–C therefore contain
**deliberately collinear points**, and A/B contain a **deliberately parallel
segment pair**, so those rows are non-vacuous.  Dataset E is the opposite case on
purpose: a genuine general-position set, where the vacuous verdict is the correct
outcome and is asserted as such.

Every profile is computed by measurement at call time, never hard-coded, so it
cannot drift away from the actual points.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from ..config import COLLINEAR_TOL
from ..geometry.affine_frame import triangle_flatness
from ..geometry.invariants import (
    collinearity_relation_change,
    parallelism_relation_change,
)
from ..geometry.points import PointSet2D, as_point_matrix

__all__ = [
    "Dataset",
    "DATASETS",
    "dataset_names",
    "build_dataset",
    "measure_asymmetry",
    "asymmetric_triangle_with_internal",
    "asymmetric_quadrilateral",
    "irregular_pentagon",
    "star_like_configuration",
    "random_general_position",
]

#: Safety margin used to *build* dataset E.  We reject a random draw if any
#: triple is flatter than this, or any segment pair is more parallel than this.
#: It is 1000x the relation threshold (``COLLINEAR_TOL``), so the intended
#: vacuity of the relation rows has three orders of magnitude of headroom rather
#: than sitting at the decision boundary.
GENERAL_POSITION_MARGIN: float = COLLINEAR_TOL * 1e3


# --------------------------------------------------------------------------- #
# symmetry measurement (the premise of "asymmetric", made checkable)
# --------------------------------------------------------------------------- #
#: A plane isometry, stored as ``(matrix, translation)``.  A stacked ``3x2``
#: array would be inhomogeneous (the translation has no third row), so the pair
#: is kept explicit.
_Isometry = tuple[np.ndarray, np.ndarray]


def _reflection_about_line(angle_deg: float, center: np.ndarray) -> _Isometry:
    """Reflection about the line through ``center`` with direction ``angle_deg``."""
    t = math.radians(2.0 * angle_deg)
    m = np.array([[math.cos(t), math.sin(t)], [math.sin(t), -math.cos(t)]])
    return m, -m @ center + center


def _rotation(angle_deg: float, center: np.ndarray) -> _Isometry:
    t = math.radians(angle_deg)
    m = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return m, -m @ center + center


def _apply(transform: _Isometry, points: np.ndarray) -> np.ndarray:
    matrix, translation = transform
    return points @ matrix.T + translation


def measure_asymmetry(
    points: Any, *, tol: float = 1e-9
) -> dict[str, Any]:
    """Test a finite list of candidate isometries for being a symmetry of ``X``.

    Returns ``{"symmetric_under": [...], "is_asymmetric": bool, "candidates": int}``.

    Scope, stated honestly: this checks a **finite candidate list** (identity,
    rotations by 90/180/270 degrees about the bounding-box centre, reflections
    about the two bounding-box axes and both diagonals, and the two 45-degree
    axes).  It is therefore a *sufficient* condition for asymmetry only with
    respect to those candidates — a shape with some exotic symmetry would pass.
    For the shapes in this module that is not an issue, and the alternative (a
    full symmetry-group computation) would be far more machinery than the claim
    needs.  The limitation is recorded rather than hidden.
    """
    x = as_point_matrix(points, name="points", min_points=1)
    centre = np.array(
        [(x[:, 0].min() + x[:, 0].max()) / 2.0, (x[:, 1].min() + x[:, 1].max()) / 2.0]
    )
    scale = float(np.abs(x).max()) or 1.0

    candidates: list[tuple[str, _Isometry]] = [("identity", (np.eye(2), np.zeros(2)))]
    for deg in (90.0, 180.0, 270.0):
        candidates.append((f"rotation_{deg:g}", _rotation(deg, centre)))
    for name, deg in (
        ("reflect_about_x_axis", 0.0),
        ("reflect_about_y_axis", 90.0),
        ("reflect_about_main_diagonal", 45.0),
        ("reflect_about_anti_diagonal", 135.0),
    ):
        candidates.append((name, _reflection_about_line(deg, centre)))

    symmetric: list[str] = []
    for name, transform in candidates:
        if name == "identity":
            continue
        moved = _apply(transform, x)
        # symmetric iff the two point sets coincide (Hausdorff distance ~ 0)
        diff = x[:, None, :] - moved[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        hausdorff = max(float(distances.min(axis=1).max()), float(distances.min(axis=0).max()))
        if hausdorff <= tol * (1.0 + scale):
            symmetric.append(name)

    return {
        "symmetric_under": tuple(symmetric),
        "is_asymmetric": not symmetric,
        "candidates": len(candidates) - 1,
    }


# --------------------------------------------------------------------------- #
# the datasets
# --------------------------------------------------------------------------- #
def asymmetric_triangle_with_internal(name: str = "A_asymmetric_triangle") -> PointSet2D:
    """Dataset A — asymmetric triangle plus interior structure (9 points).

    Deliberate content:

    * ``A, S, B`` collinear (``S`` on the segment ``AB``) — collinearity witness;
    * ``A, M1, M2`` collinear (both on the median from ``A``) — a second,
      differently-oriented witness;
    * ``P1 P2`` parallel to ``AB`` by construction (``P2 - P1 = B - A``) —
      parallelism witness;
    * ``G`` a generic interior point so the configuration is not rigid;
    * all distances irrational-ish and no mirror axis, hence asymmetric.
    """
    A = np.array([0.0, 0.0])
    B = np.array([3.1, 0.2])
    C = np.array([0.6, 2.4])
    offset = np.array([0.45, 0.30])
    midpoint_BC = (B + C) / 2.0

    points = [
        A,
        B,
        C,
        A + 0.6 * (B - A),              # S: on segment AB
        A + 0.35 * (midpoint_BC - A),   # M1: on the median from A
        A + 0.70 * (midpoint_BC - A),   # M2: on the same median
        A + offset,                      # P1
        B + offset,                      # P2  => P1P2 parallel to AB
        (A + B + C) / 3.0 + np.array([0.15, -0.20]),  # G: generic
    ]
    return PointSet2D(points, name=name)


def asymmetric_quadrilateral(name: str = "B_asymmetric_quadrilateral") -> PointSet2D:
    """Dataset B — asymmetric trapezoid (unequal legs) with sampled sides (10 points).

    The two bases are parallel **exactly** (the upper base is the lower base
    scaled by 0.5, a power of two, so the cross product is exactly zero), giving a
    non-vacuous parallelism witness; the legs have different lengths and
    directions, so the shape has no mirror axis.
    """
    lower_a = np.array([0.0, 0.0])
    lower_b = np.array([4.0, 0.5])
    upper_a = np.array([0.75, 1.8])
    upper_b = upper_a + 0.5 * (lower_b - lower_a)  # exactly parallel to the lower base

    points: list[np.ndarray] = [lower_a, lower_b, upper_b, upper_a]
    points += [lower_a + t * (lower_b - lower_a) for t in (0.25, 0.50, 0.75)]
    points += [upper_a + 0.5 * (upper_b - upper_a)]
    points += [
        lower_a + 0.4 * (upper_a - lower_a) + np.array([0.35, 0.0]),
        lower_b + 0.55 * (upper_b - lower_b) + np.array([-0.2, 0.15]),
    ]
    return PointSet2D(points, name=name)


def irregular_pentagon(name: str = "C_irregular_pentagon") -> PointSet2D:
    """Dataset C — irregular pentagon with one sampled side (7 points)."""
    vertices = [
        np.array([0.00, 0.00]),
        np.array([2.30, 0.35]),
        np.array([2.90, 1.90]),
        np.array([1.20, 2.70]),
        np.array([0.15, 1.50]),
    ]
    on_side = vertices[0] + 0.45 * (vertices[1] - vertices[0])   # collinear triple on a side
    interior = np.array([1.35, 1.30])
    return PointSet2D([*vertices, on_side, interior], name=name)


def star_like_configuration(name: str = "D_star_like") -> PointSet2D:
    """Dataset D — irregular star with an off-centre midpoint on one spike (8 points).

    Radii and angles are irregular, so the figure is concave and asymmetric.  One
    point is placed at exactly half of the longest spike: ``centre``, that
    midpoint and the spike tip are then collinear, which keeps the collinearity
    row non-vacuous on a shape that would otherwise have no collinear triple.
    """
    centre = np.array([0.33, -0.21])
    angles = np.array([8.0, 74.0, 137.0, 205.0, 268.0, 322.0])
    radii = np.array([1.90, 1.25, 1.55, 1.05, 1.70, 1.40])
    tips = [
        centre + r * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        for a, r in zip(angles, radii)
    ]
    longest = 0  # index of the longest spike
    half = centre + 0.5 * (tips[longest] - centre)
    return PointSet2D([centre, *tips, half], name=name)


def random_general_position(
    n: int = 12, *, seed: int = 20261002, box: tuple[float, float] = (-2.0, 2.0)
) -> PointSet2D:
    """Dataset E — random points in general position (no collinear triple, no parallel pair).

    Rejection-sampled so that the *intended* vacuity actually holds: the sampler
    rejects a draw if any three points are collinear or any two disjoint segments
    are parallel.  Without this the dataset would be "random" but its profile
    would vary, and the assertion "these rows are vacuous" would be luck.

    Raises
    ------
    RuntimeError
        If no such configuration is found within the attempt budget — a silent
        fallback would defeat the purpose of the dataset.
    """
    g = np.random.default_rng(seed)
    lo, hi = box
    margin = GENERAL_POSITION_MARGIN
    for _ in range(2000):
        pts = g.uniform(lo, hi, size=(n, 2))
        if min(
            float(np.linalg.norm(pts[i] - pts[j]))
            for i in range(n)
            for j in range(i + 1, n)
        ) < 0.2:
            continue
        if any(
            triangle_flatness(pts[i], pts[j], pts[k]) <= margin
            for i in range(n)
            for j in range(i + 1, n)
            for k in range(j + 1, n)
        ):
            continue
        if parallelism_relation_change(pts, pts, tol=margin)[2] != 0:
            continue
        return PointSet2D(pts, name="E_random_general_position")
    raise RuntimeError(  # pragma: no cover - the sampler succeeds in practice
        f"could not draw a general-position set of {n} points in {box} at margin {margin:g}"
    )


# --------------------------------------------------------------------------- #
# registry with measured profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dataset:
    """A named point configuration plus its *measured* structural profile.

    The profile is always recomputed from the points, never stored: a stored copy
    could drift away from the shape it describes, and a stale "has a collinear
    triple" flag would silently turn a non-vacuous row into a vacuous one.
    """

    key: str
    name_zh: str
    purpose_zh: str
    builder: Callable[[], PointSet2D]
    designed_for: tuple[str, ...]

    @property
    def image(self) -> PointSet2D:
        return self.builder()

    @property
    def points(self) -> np.ndarray:
        return self.builder().as_array()

    # -- measured profile ------------------------------------------------- #
    def n_collinear_triples(self) -> int:
        x = self.points
        return collinearity_relation_change(x, x)[2]

    def n_parallel_pairs(self) -> int:
        x = self.points
        return parallelism_relation_change(x, x)[2]

    def asymmetry(self) -> dict[str, Any]:
        return measure_asymmetry(self.points)

    def profile(self) -> dict[str, Any]:
        """Everything a report needs to describe this dataset, by measurement."""
        x = self.points
        asym = self.asymmetry()
        return {
            "key": self.key,
            "name_zh": self.name_zh,
            "n_points": int(x.shape[0]),
            "purpose_zh": self.purpose_zh,
            "designed_for": list(self.designed_for),
            "n_collinear_triples": self.n_collinear_triples(),
            "n_parallel_pairs": self.n_parallel_pairs(),
            "collinearity_row_is_vacuous": self.n_collinear_triples() == 0,
            "parallelism_row_is_vacuous": self.n_parallel_pairs() == 0,
            "symmetric_under": list(asym["symmetric_under"]),
            "is_asymmetric": asym["is_asymmetric"],
            "symmetry_candidates_checked": asym["candidates"],
            "diameter": float(
                max(
                    float(np.linalg.norm(x[i] - x[j]))
                    for i in range(x.shape[0])
                    for j in range(i + 1, x.shape[0])
                )
            ),
        }


DATASETS: dict[str, Dataset] = {
    "A": Dataset(
        key="A",
        name_zh="非对称三角形 + 内部点",
        purpose_zh="最小可解标架 + 两条不同方向的共线见证 + 一对平行线段见证",
        builder=asymmetric_triangle_with_internal,
        designed_for=("collinearity_nonvacuous", "parallelism_nonvacuous", "asymmetry"),
    ),
    "B": Dataset(
        key="B",
        name_zh="非对称四边形（梯形，不等腰）",
        purpose_zh="精确平行的两底（平行性见证）+ 底边上的共线点；不等腰保证无镜面对称",
        builder=asymmetric_quadrilateral,
        designed_for=("collinearity_nonvacuous", "parallelism_nonvacuous", "asymmetry"),
    ),
    "C": Dataset(
        key="C",
        name_zh="不规则五边形",
        purpose_zh="无平行边的凸图形：平行性行预期为 vacuous，共线性行非空",
        builder=irregular_pentagon,
        designed_for=("collinearity_nonvacuous", "parallelism_vacuous", "asymmetry"),
    ),
    "D": Dataset(
        key="D",
        name_zh="非规则星形",
        purpose_zh="凹图形 + 不规则半径；一条长尖上的中点制造唯一的共线三元组",
        builder=star_like_configuration,
        designed_for=("collinearity_nonvacuous", "concave", "asymmetry"),
    ),
    "E": Dataset(
        key="E",
        name_zh="随机一般位置点集",
        purpose_zh="对照组：刻意保证无共线三元组、无平行线段，用于验证 vacuous 语义",
        builder=random_general_position,
        designed_for=("collinearity_vacuous", "parallelism_vacuous", "asymmetry"),
    ),
}


def dataset_names() -> list[str]:
    """Sorted dataset keys."""
    return sorted(DATASETS)


def build_dataset(key: str) -> PointSet2D:
    """Instantiate the point set of dataset ``key`` (name set to the dataset key)."""
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; available: {', '.join(dataset_names())}")
    dataset = DATASETS[key]
    image = dataset.image
    return image.with_points(image.as_array(), name=f"{key}:{image.name}")


def all_profiles() -> list[dict[str, Any]]:
    """Profiles of every dataset, in key order."""
    return [DATASETS[key].profile() for key in dataset_names()]
