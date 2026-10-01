"""Compression layer — Phase 5.

The object compressed here is the same finite point set as everywhere else in this project.
The primary source for the *construction* is V. N. Kozlov, «Об одном подходе к сжатию
информации при обработке и распознавании изображений», ЖВМиМФ 42:2 (2002) 256–268; its full
text is in ``.sources/kozlov2002.txt`` and the extracted definitions are quoted in
``docs/specs/image_compression.md``.

Three families, and the separation is enforced by a test rather than by convention
(G5.9, ``tests/test_compression_naming.py``):

======================  ========================================================================
``official/``           constructions **from the primary text**: the эскиз ($r$-cover with
                        centres in $A$), the остов (minimum-cardinality sketch), the
                        similarity-invariant distance, and the two-sided bound of Теорема 4
``teaching/``           generic baselines that are *not* in the source and must not wear its
                        name: Euclidean $\varepsilon$-cover, farthest-point sampling, grid
                        reduction
``experimental/``       our own exploratory ideas: affine-aware sampling
======================  ========================================================================

Nothing in ``teaching/`` or ``experimental/`` may be described as "the Kozlov method", and
nothing in ``official/`` may claim more than the source states.
"""

from __future__ import annotations

from .metrics import (
    code_size_reduction,
    coverage_report,
    compression_ratio,
    directed_hausdorff,
    formal_code_size,
    hausdorff,
    measure,
    size_reduction,
)
from .experimental.affine_aware import AffineAwareSampler
from .model import (
    COMPRESSORS,
    CompressionResult,
    Compressor,
    CompressorContract,
    get_compressor,
    list_compressors,
    run_compressor,
)
from .official.kozlov_sketch import (
    KOZLOV_SOURCE,
    KozlovSketch,
    KozlovSkeleton,
    certified_nearest,
    greedy_sketch,
    kozlov_image_distance,
    kozlov_sketch_distance_bounds,
    minimum_cover_size,
    skeleton_hierarchy,
    verify_sketch,
)
from .teaching.epsilon_cover import TeachingEpsilonCover
from .teaching.farthest_point import TeachingFarthestPoint
from .teaching.grid_reduce import TeachingGridReduce

__all__ = [
    # contract and result types
    "Compressor",
    "CompressorContract",
    "CompressionResult",
    "COMPRESSORS",
    "get_compressor",
    "list_compressors",
    "run_compressor",
    # official (author-derived, Kozlov 2002)
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
    # teaching baselines (not from the source)
    "TeachingEpsilonCover",
    "TeachingFarthestPoint",
    "TeachingGridReduce",
    # experimental (this project's own)
    "AffineAwareSampler",
    # metrics
    "compression_ratio",
    "code_size_reduction",
    "size_reduction",
    "directed_hausdorff",
    "hausdorff",
    "coverage_report",
    "formal_code_size",
    "measure",
]
