"""kozlov-lab — interactive laboratory for V. N. Kozlov's mathematical theory of
visual perception.

Pipeline implemented by this package (README section 1)::

    图像 → 有限点集 → 几何变换 → 内部坐标 → 图像编码 → 等价判定 → 识别

Layering (README section 2) — dependencies point strictly downwards::

    数学算法层   kozlov_lab.geometry / coding / recognition / compression / stereo
        ↓
    实验层       experiments/
        ↓
    可视化层     app/
        ↓
    教学解释层   docs/

The GUI must never re-implement a mathematical algorithm: it calls the public
functions exported here.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .config import (
    COLLINEAR_TOL,
    EPS,
    FLOAT_TOL,
    MATCH_TOL,
    MAX_CONDITION_NUMBER,
    SIGNATURE_DECIMALS,
    tolerance_for_condition,
)
from .geometry import (
    AffineCoordinates,
    AffineFrame,
    AffineTransform,
    InvariantReport,
    IDENTITY,
    Point2D,
    PointSet2D,
    affine_coordinates,
    affine_coordinates_batch,
    apply_affine,
    collinearity_sine,
    count_non_collinear_triples,
    find_non_collinear_triple,
    frame_matrix,
    invariant_report,
    is_non_collinear,
    iter_non_collinear_triples,
    min_vertex_sine,
    reconstruct_point,
    report_for_transform,
    signed_double_area,
    triangle_flatness,
)

__all__ = [
    "__version__",
    # tolerances
    "EPS",
    "FLOAT_TOL",
    "COLLINEAR_TOL",
    "MATCH_TOL",
    "SIGNATURE_DECIMALS",
    "MAX_CONDITION_NUMBER",
    "tolerance_for_condition",
    # points
    "Point2D",
    "PointSet2D",
    # transforms
    "AffineTransform",
    "IDENTITY",
    "apply_affine",
    # internal coordinates
    "AffineFrame",
    "AffineCoordinates",
    "affine_coordinates",
    "affine_coordinates_batch",
    "reconstruct_point",
    "frame_matrix",
    "is_non_collinear",
    "collinearity_sine",
    "min_vertex_sine",
    "triangle_flatness",
    "signed_double_area",
    "count_non_collinear_triples",
    "iter_non_collinear_triples",
    "find_non_collinear_triple",
    # invariants
    "invariant_report",
    "report_for_transform",
    "InvariantReport",
]
