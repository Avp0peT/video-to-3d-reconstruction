"""Standard multi-view geometry — **BASELINE**, not the author's content.

Kozlov 1996 uses **parallel (directional) projection**; nothing in that paper mentions intrinsics,
focal lengths, a projection matrix, the epipolar constraint or DLT.  Everything here is the
standard textbook machinery, kept in its own package so that no result computed with it can be
reported as the author's method.  The naming rule is enforced by ``tests/test_stereo.py``.
"""

from __future__ import annotations

from .camera import Camera, project_points
from .triangulate import triangulate_dlt, triangulate_many

__all__ = ["Camera", "project_points", "triangulate_dlt", "triangulate_many"]
