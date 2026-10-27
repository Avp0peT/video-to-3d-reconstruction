"""SOURCE-EXACT constructions from Kozlov 1996 — see ``docs/specs/stereo_kozlov.md``.

Source definitions and their numerical/exact software representations are distinguished.
AlgT'/AlgSo take already labelled projections. Their source steps have been reviewed in
docs/specs/stereo_algorithms_1996.md, but implementation and algorithm gates remain pending.
The legacy tetrahedron-named helper only counts single-view triangles and reports prerequisites;
it neither recovers correspondence nor verifies the source's two-view coplanarity predicate.
"""

from __future__ import annotations

from .kozlov_code3d import (
    SOURCE_3D,
    correspondence_tetrahedron_principle,
    is_a_equivalent,
    kozlov_body_code,
    parallel_project,
    tetrahedron_volumes,
)

__all__ = [
    "SOURCE_3D",
    "tetrahedron_volumes",
    "kozlov_body_code",
    "parallel_project",
    "is_a_equivalent",
    "correspondence_tetrahedron_principle",
]
