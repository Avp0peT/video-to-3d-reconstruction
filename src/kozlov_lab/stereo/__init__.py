"""Stereo vision: 2D projections and 3D recovery — **not implemented yet**
(Phase 6, README section 11).

Planned content::

    camera.py        pinhole camera (intrinsics K, rotation R, centre C) and P matrices
    projection.py    project_points(points3d, camera)  ->  2D image points
    triangulation.py triangulate(left_points, right_points, cameras) -> 3D

Planned API::

    camera_matrix(camera) -> (3, 4) ndarray
    project_points(points3d, camera) -> (n, 2) ndarray
    triangulate(left, right, cameras) -> {"points3d": (n, 3), "reprojection_error_left": ...,
                                          "reprojection_error_right": ...}

Two versions as required by README section 11:

1. **known correspondence** — the i-th left point is the i-th right point;
2. **corresponding point problem** — find which point of the right image
   corresponds to a given point of the left image.  Note that correspondence in
   a stereo *pair* is a different mathematical question from the affine
   equivalence of Phase 4: an epipolar constraint is a *projective* notion, and
   the two must not be conflated in the report.
"""

from __future__ import annotations
