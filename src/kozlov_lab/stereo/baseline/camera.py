"""Pinhole camera ``p ~ P X`` — **BASELINE**.

``P = K [R | t]`` with the standard conventions: ``X`` in world coordinates (homogeneous),
``p = (u, v, w)`` in image coordinates, and the image point ``(u/w, v/w)``.  Points at infinity
(``w = 0``) are reported rather than silently divided by zero — the same refusal discipline the
rest of the project uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = ["Camera", "project_points", "look_at_camera"]


@dataclass(frozen=True)
class Camera:
    """A pinhole camera, stored as the 3x4 projection matrix ``P`` plus its parts."""

    projection: np.ndarray
    intrinsics: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        projection = np.asarray(self.projection, dtype=float)
        if projection.shape != (3, 4):
            raise ValueError(f"P must be 3x4, got {projection.shape}")
        object.__setattr__(self, "projection", projection)

    @staticmethod
    def from_parts(intrinsics: Sequence[Sequence[float]], rotation: Sequence[Sequence[float]],
                   translation: Sequence[float], name: str = "", **metadata: Any) -> "Camera":
        k = np.asarray(intrinsics, dtype=float)
        r = np.asarray(rotation, dtype=float)
        t = np.asarray(translation, dtype=float).reshape(3, 1)
        projection = k @ np.hstack([r, t])
        return Camera(projection, k, r, t.reshape(3), name=name, metadata=metadata)

    @property
    def centre(self) -> np.ndarray:
        """Camera centre from the right null vector of ``P``."""
        _, _, vh = np.linalg.svd(self.projection)
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-12:
            raise ValueError("the camera centre is at infinity (an affine projection matrix)")
        return homogeneous[:3] / homogeneous[3]

    def project(self, points: Any) -> np.ndarray:
        return project_points(self.projection, points)

    def ray(self, point2d: Sequence[float]) -> np.ndarray:
        """A direction vector of the viewing ray through an image point (best-effort).

        Uses the camera centre and the pseudo-inverse of ``P``, which is exact for a finite
        centre.  Returned **unnormalised**; callers that need an angle normalise themselves.
        """
        u, v = (float(point2d[0]), float(point2d[1]))
        rows = self.projection[:2] - np.array([[u], [v]]) * self.projection[2]
        # solve rows @ X = 0 for the nearest point to the centre, then take the direction
        _, _, vh = np.linalg.svd(rows)
        point = vh[-1]
        if abs(point[3]) < 1e-12:
            raise ValueError("cannot recover a finite ray point from this camera and image point")
        world = point[:3] / point[3]
        return world - self.centre


def project_points(projection: Any, points: Any) -> np.ndarray:
    """Project ``(n, 3)`` points; raises on a point at infinity rather than dividing by zero."""
    matrix = np.asarray(projection, dtype=float)
    body = np.asarray(points, dtype=float)
    single = body.ndim == 1
    if single:
        body = body.reshape(1, 3)
    homogeneous = np.hstack([body, np.ones((body.shape[0], 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError(
            "a point projects to w = 0 (it lies on the camera plane); the image point is at "
            "infinity and refusing is better than dividing by zero"
        )
    result = projected[:, :2] / w[:, None]
    return result[0] if single else result


def look_at_camera(eye: Sequence[float], target: Sequence[float], up: Sequence[float],
                   focal: float = 500.0, principal: Sequence[float] = (0.0, 0.0),
                   name: str = "", **metadata: Any) -> Camera:
    """A camera looking from ``eye`` at ``target`` — the standard construction, used by the UI."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, -down, forward])
    intrinsics = np.array(
        [[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]]
    )
    return Camera.from_parts(
        intrinsics, rotation, -rotation @ eye, name=name,
        eye=eye.tolist(), target=target.tolist(), **metadata
    )
