"""Baseline compressors — **not** from the primary source (Phase 5B).

These are generic, well-known constructions implemented here as honest points of comparison.
None of them appears in Kozlov 2002, so none may be described as "the Kozlov method"; the
naming rule is enforced by ``tests/test_compression_naming.py``.

Their contracts differ from the official ones in a way that matters:

* :class:`TeachingEpsilonCover` allows centres *anywhere* (a true $\\varepsilon$-net), whereas
  the source's эскиз requires centres to be points of the image;
* :class:`TeachingFarthestPoint` guarantees an **upper bound** on the covering radius for a
  fixed cardinality and nothing more;
* :class:`TeachingGridReduce` is an engineering baseline with no mathematical guarantee at all.
"""

from __future__ import annotations

from .epsilon_cover import TeachingEpsilonCover
from .farthest_point import TeachingFarthestPoint
from .grid_reduce import TeachingGridReduce

__all__ = ["TeachingEpsilonCover", "TeachingFarthestPoint", "TeachingGridReduce"]
