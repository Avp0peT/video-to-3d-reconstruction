"""SOURCE-EXACT constructions from Kozlov 1996 — see ``docs/specs/stereo_kozlov.md``.

Only what the primary text states lives here: the body, the 3D code of tetrahedron volume ratios,
parallel projection, ``a'``-equivalence, and the tetrahedron correspondence principle.
``AlgT'`` and ``AlgSo`` are **deliberately absent**: their step-by-step description has not been
extracted from the source, and the 6A gate forbids attaching the author's name to a reconstruction
procedure meanwhile.
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
