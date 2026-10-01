"""Author-derived constructions (Phase 5A/5B).

Everything re-exported here comes from the primary text of Kozlov 2002 and is quoted in
``docs/specs/image_compression.md``.  Anything that is *not* in that text lives in
``compression/teaching`` or ``compression/experimental`` instead — the split is enforced by
``tests/test_compression_naming.py`` (gate G5.9).
"""

from __future__ import annotations

from .kozlov_sketch import (
    KOZLOV_SOURCE,
    KozlovSketch,
    KozlovSkeleton,
    certified_nearest,
    distance_matrix,
    greedy_sketch,
    kozlov_image_distance,
    kozlov_sketch_distance_bounds,
    minimum_cover_size,
    skeleton_hierarchy,
    verify_sketch,
)

__all__ = [
    "KOZLOV_SOURCE",
    "KozlovSketch",
    "KozlovSkeleton",
    "verify_sketch",
    "greedy_sketch",
    "minimum_cover_size",
    "skeleton_hierarchy",
    "kozlov_image_distance",
    "kozlov_sketch_distance_bounds",
    "certified_nearest",
    "distance_matrix",
]
