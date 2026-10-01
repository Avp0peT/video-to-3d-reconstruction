"""Geometry layer: point sets, geometric transformations, internal affine
coordinates and invariant measurements (Phases 1–2 of README section 24).

Everything in this package is pure mathematics with no UI concerns: no
plotting, no Streamlit, no I/O other than the JSON format of a point set.
"""

from __future__ import annotations

from .affine_frame import (
    AffineCoordinates,
    AffineFrame,
    InvarianceDelta,
    affine_coordinates,
    affine_coordinates_batch,
    collinearity_sine,
    count_non_collinear_triples,
    find_non_collinear_triple,
    frame_matrix,
    is_non_collinear,
    iter_non_collinear_triples,
    min_vertex_sine,
    reconstruct_point,
    signed_double_area,
    triangle_flatness,
)
from .invariants import (
    FAMILY_LABELS_ZH,
    PROPERTY_KEYS,
    THEORY_EXPECTATION,
    Discrepancy,
    InvariantEntry,
    InvariantReport,
    TheoryCheck,
    TheoryRow,
    absolute_coordinate_change,
    affine_coordinate_change,
    angle_change,
    check_against_theory,
    collinearity_relation_change,
    distance_change,
    invariant_report,
    length_ratio_change,
    parallelism_relation_change,
    report_for_transform,
    theory_verdict,
)
from .points import Point2D, PointSet2D, as_point_array, as_point_matrix
from .transforms import IDENTITY, AffineTransform, apply_affine, rotation_matrix

__all__ = [
    # points
    "Point2D",
    "PointSet2D",
    "as_point_array",
    "as_point_matrix",
    # transforms
    "AffineTransform",
    "IDENTITY",
    "apply_affine",
    "rotation_matrix",
    # internal coordinates
    "AffineFrame",
    "AffineCoordinates",
    "InvarianceDelta",
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
    "PROPERTY_KEYS",
    "THEORY_EXPECTATION",
    "FAMILY_LABELS_ZH",
    "theory_verdict",
    "check_against_theory",
    "InvariantEntry",
    "InvariantReport",
    "TheoryRow",
    "TheoryCheck",
    "Discrepancy",
    "invariant_report",
    "report_for_transform",
    "absolute_coordinate_change",
    "distance_change",
    "angle_change",
    "length_ratio_change",
    "collinearity_relation_change",
    "parallelism_relation_change",
    "affine_coordinate_change",
]
