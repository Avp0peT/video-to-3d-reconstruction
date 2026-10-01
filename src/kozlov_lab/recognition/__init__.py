"""Affine-equivalence recognition — **not implemented yet** (Phase 4, README section 8).

Blocked by the Phase 3 gate:

    "Phase 3 无法保证点排列无关性，不允许进入 Phase 4。"

Planned API (frozen here so that the GUI and the experiments can be written
against it in advance)::

    affine_equivalent(x, y) -> {
        "equivalent":   bool,
        "matrix":       (2, 2) ndarray | None,
        "translation":  (2,)   ndarray | None,
        "point_mapping": list[tuple[int, int]] | None,
        "residual":     float,
        "method":       "code" | "control_points",
        "cost":         {"triples_tested": int, "points_verified": int},
    }

Two methods must be implemented and compared on accuracy, runtime and
complexity growth in the number of points (README section 8):

* **A** — compare the canonical image codes of Phase 3;
* **B** — enumerate non-collinear control triple pairs, solve the affine map,
  then verify all remaining points.

``approximate.py`` will hold the *approximately* equivalent case (noisy points,
missing points, extra points) and must report it separately: README section 13
asks explicitly what the difference between exact and approximate recognition
is.
"""

from __future__ import annotations
