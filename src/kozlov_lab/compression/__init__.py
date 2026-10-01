"""Image-information compression — **not implemented yet** (Phase 5, README section 10).

Planned content::

    cover.py     epsilon-cover  (priority), greedy representative points
    sampling.py  farthest-point sampling, grid reduction

Planned API::

    compress_point_set(points, epsilon) -> {
        "sketch":        PointSet2D,
        "compression_ratio": 1 - |S(X)| / |X|,
        "covering_radius":   float,   # must be <= epsilon (verified, not assumed)
        "method":        str,
    }

Methodological requirement for the experiment layer: every compression result
must be *verified* to actually cover the original set within ``epsilon``.
Reporting a compression ratio without checking the covering radius would make
the whole Experiment D meaningless.
"""

from __future__ import annotations
