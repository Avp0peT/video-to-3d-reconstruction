"""Stereo vision and 3D reconstruction (Phase 6).

Three layers, separated the same way the previous phases are:

======================  ===================================================================
``official/``           **SOURCE-EXACT** constructions from Kozlov 1996: the body as a finite
                        point set, the 3D code (tetrahedron volume ratios), parallel
                        projection, a'-equivalence, and the tetrahedron correspondence
                        principle.  ``AlgT'``/``AlgSo`` are **not** implemented — their
                        step-by-step description has not been extracted, and the 6A gate
                        forbids giving that name to a reconstruction procedure meanwhile.
``baseline/``           standard multi-view geometry (pinhole camera, DLT triangulation):
                        **BASELINE**, not the author's content.
``correspondence.py``   our own search over point correspondences: **EXPERIMENTAL**.
======================  ===================================================================
"""

from __future__ import annotations

from .baseline.camera import Camera, project_points
from .baseline.triangulate import triangulate_dlt, triangulate_many
from .correspondence import (
    CorrespondenceResult,
    CorrespondenceVerdict,
    EngineAgreement,
    RecoveryOutcome,
    engines_agree,
    match_correspondences,
    oracle_correspondence,
    reconstruction_outcome,
)
from .degeneracy import DegeneracyKind, classify_geometry
from .official import (
    SOURCE_3D,
    correspondence_tetrahedron_principle,
    is_a_equivalent,
    kozlov_body_code,
    parallel_project,
    tetrahedron_volumes,
)
from .metrics import (
    condition_number,
    collinearity_relation,
    coplanarity_relation,
    max_3d_error,
    pairwise_distance_matrix,
    profile_relative_difference,
    ray_angle_degrees,
    relative_3d_error,
    reprojection_error,
    rmse_3d,
    volume_ratio_profile,
)
from .model import AffineTransform3D, Point3D, PointSet3D, build_scene, scene_names

__all__ = [
    "Point3D",
    "PointSet3D",
    "AffineTransform3D",
    "build_scene",
    "scene_names",
    "Camera",
    "project_points",
    "triangulate_dlt",
    "triangulate_many",
    "CorrespondenceVerdict",
    "CorrespondenceResult",
    "EngineAgreement",
    "RecoveryOutcome",
    "match_correspondences",
    "oracle_correspondence",
    "engines_agree",
    "reconstruction_outcome",
    "DegeneracyKind",
    "classify_geometry",
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
    # SOURCE-EXACT (Kozlov 1996)
    "SOURCE_3D",
    "tetrahedron_volumes",
    "kozlov_body_code",
    "parallel_project",
    "is_a_equivalent",
    "correspondence_tetrahedron_principle",
]
