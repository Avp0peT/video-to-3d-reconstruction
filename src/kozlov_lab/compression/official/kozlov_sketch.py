"""Author-derived compression: the эскиз, the остов, the distance, and Теорема 4.

Everything in this module is **from the primary text** — V. N. Kozlov, «Об одном подходе к
сжатию информации при обработке и распознавании изображений», ЖВМиМФ 42:2 (2002) 256–268 —
except where a docstring says otherwise.  The full text is in ``.sources/kozlov2002.txt`` and
the verbatim quotations are in ``docs/specs/image_compression.md``.

эскиз (sketch)
--------------
> «...чтобы каждая точка из А находилась на расстоянии не большем $r$ от какой-либо точки из
> $A_r$. ... $A_r$ — фрагмент изображения А ... Изображение $A_r$ можно трактовать как покрытие
> изображения А кругами радиуса $r$ с центрами в точках из $A_r$ ... поэтому $A_r$ называем
> **эскизом** изображения А.»

$A_r \\subseteq A$ and every point of $A$ is within $r$ of some point of $A_r$: an
**$r$-cover whose centres are themselves points of $A$**.  Note that the parameter is called
$r$ in the source — the symbol $\\varepsilon$ does not occur in that paper at all.

остов (skeleton)
----------------
> «...в $\\{A_r\\}$ есть изображение ... с **наименьшим** ... числом точек. Такое изображение
> будем называть **остовом** изображения А.»

The minimum-cardinality sketch.  The source **defines** it and does **not** give an algorithm
for finding it; the solver here is therefore ours, and it says so: the exact branch-and-bound
path reports ``proved=True``, and the greedy fallback never claims minimality.

Теорема 4
---------
> «Теорема 4 позволяет заменить вычисление расстояний между изображениями А и В **оценками**
> этого расстояния, причем чем меньше величина $r$, тем **уже интервал**, в который заключено
> $B(A,B)$...»

The theorem's own proof, recovered from the text, gives the two-sided bound

.. math::

    R_{r_1 r_2} \\le R_0 + \\max(r_1, r_2),
    \\qquad
    R_0 \\le R_{r_1 r_2} + \\max(r_1, r_2),

i.e. ``|R_{sketch} - R_true| <= max(r1, r2)``: the sketch distance approximates the true
distance to within the coarser cover radius.  The source's *use* of it is a certified nearest
neighbour test: if the interval for one candidate does not overlap the others', that candidate
is provably the nearest.  :func:`certified_nearest` implements exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
import tracemalloc
from typing import Any, Sequence

import numpy as np

from ..metrics import directed_hausdorff, hausdorff
from ..model import CompressionResult, Compressor, CompressorContract

__all__ = [
    "distance_matrix",
    "verify_sketch",
    "greedy_sketch",
    "minimum_cover_size",
    "skeleton_hierarchy",
    "kozlov_image_distance",
    "kozlov_sketch_distance_bounds",
    "certified_nearest",
    "KozlovSketch",
    "KozlovSkeleton",
]


def distance_matrix(points: Any) -> np.ndarray:
    """Pairwise Euclidean distances — the only geometric primitive the source needs."""
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    differences = array[:, None, :] - array[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))


def verify_sketch(points: Any, sketch: Any, r: float) -> bool:
    """Whether ``sketch`` really is an эскиз of ``points`` at radius ``r``.

    The definition is a **claim to be checked**, not an assumption: a compressor that returns
    points and a radius has not produced an эскиз until this returns ``True``.
    """
    if r < 0:
        raise ValueError("the covering radius must be non-negative")
    source = np.asarray(points, dtype=float).reshape(-1, 2)
    target = np.asarray(sketch, dtype=float).reshape(-1, 2)
    if target.shape[0] == 0:
        return False
    return directed_hausdorff(source, target) <= r + 1e-12


def greedy_sketch(points: Any, r: float) -> list[int]:
    """A valid эскиз at radius ``r``: greedy cover by points of ``A``.

    **Guaranteed** to cover within ``r`` (each chosen centre covers every still-uncovered point
    within ``r`` by construction), and **not** guaranteed minimum — that is what the остов is
    for.  Deterministic: ties are broken by index.
    """
    matrix = np.asarray(points, dtype=float).reshape(-1, 2)
    distances = distance_matrix(matrix)
    n = matrix.shape[0]
    uncovered = set(range(n))
    chosen: list[int] = []
    while uncovered:
        best_index, best_cover = -1, -1
        for candidate in sorted(uncovered):
            covered = sum(1 for j in uncovered if distances[candidate, j] <= r + 1e-12)
            if covered > best_cover:
                best_index, best_cover = candidate, covered
        chosen.append(best_index)
        uncovered -= {j for j in uncovered if distances[best_index, j] <= r + 1e-12}
    return chosen


def minimum_cover_size(
    points: Any, r: float, *, node_budget: int = 500_000
) -> tuple[int | None, list[int], bool]:
    """The остов: minimum-cardinality эскиз at radius ``r``.

    Returns ``(size_or_None, indices, proved)``.  Exact iterative-deepening depth-first search
    over set cover, branching on the **least-covered** point (the standard most-constrained
    branching, which keeps the tree small).  ``proved=True`` only when the search completed
    within the node budget; otherwise the greedy solution is returned with ``proved=False`` and
    the caller must not report it as an остов.

    Deliberately *not* named after the author: the source defines the остов but gives no
    algorithm, so the algorithm is this project's and the naming rule (G5.9) applies.
    """
    matrix = np.asarray(points, dtype=float).reshape(-1, 2)
    n = matrix.shape[0]
    distances = distance_matrix(matrix)
    covers = [
        frozenset(j for j in range(n) if distances[i, j] <= r + 1e-12) for i in range(n)
    ]
    if any(not c for c in covers):
        return None, [], False  # unreachable: d(i,i)=0 <= r for r >= 0
    universe = frozenset(range(n))
    nodes = 0

    def search(remaining: frozenset[int], depth: int) -> list[int] | None:
        nonlocal nodes
        if not remaining:
            return []
        if depth == 0:
            return None
        nodes += 1
        if nodes > node_budget:
            raise _Budget()
        # branch on the point covered by the fewest remaining candidates
        pivot = min(remaining, key=lambda j: sum(1 for c in covers if j in c))
        for candidate in range(n):
            if pivot not in covers[candidate]:
                continue
            found = search(remaining - covers[candidate], depth - 1)
            if found is not None:
                return [candidate, *found]
        return None

    for size in range(1, n + 1):
        try:
            solution = search(universe, size)
        except _Budget:
            return None, greedy_sketch(matrix, r), False
        if solution is not None:
            return size, solution, True
    return n, list(range(n)), True  # pragma: no cover - r >= diameter covers with one centre


class _Budget(RuntimeError):
    """Internal signal: the exact search exceeded its node budget."""


def skeleton_hierarchy(points: Any, radii: Sequence[float], **kwargs: Any) -> list[dict[str, Any]]:
    """「иерархия остовов」: skeleton size as a function of ``r``.

    The source notes the smallest skeleton is one point (for large ``r``) and the largest
    equals the image itself (for ``r = 0``); this function reproduces that ladder.
    """
    matrix = np.asarray(points, dtype=float).reshape(-1, 2)
    rows: list[dict[str, Any]] = []
    for r in radii:
        size, indices, proved = minimum_cover_size(matrix, r, **kwargs)
        rows.append(
            {
                "r": float(r),
                "n_skeleton": len(indices),
                "proved_minimum": proved,
                "achieved_radius": directed_hausdorff(matrix, matrix[indices]),
                "n_points": int(matrix.shape[0]),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# the distance and the bound
# --------------------------------------------------------------------------- #
def kozlov_image_distance(
    a: Any,
    b: Any,
    *,
    align: bool = True,
    n_rotations: int = 24,
    n_scales: int = 5,
) -> dict[str, Any]:
    """The source's distance ``B(A,B)``: Hausdorff minimized over the similarity class.

    > «Расстоянием $B(A,B)$ между изображениями А и В назовем наименьшую из величин $l(B')$,
    > полученных для всех $B'$ из $B^*$.»

    $B^*$ is the class of images similar to $B$ (the paper's section heading is
    «...ЗАДАННЫХ С ТОЧНОСТЬЮ ДО ПОДОБИЯ»), so the quantity is **similarity-invariant**: the
    minimum is taken over the placement of the two images relative to one another.

    Honest limitation: minimizing over the whole similarity group is a non-convex search, and
    this implementation approximates it by aligning centroids and sweeping a **discrete grid**
    of rotations and scales.  The grid is returned in the result so the approximation is
    visible; ``align=False`` gives the plain (placement-sensitive) Hausdorff distance, which is
    what the equivariance experiment needs.
    """
    first = np.asarray(a, dtype=float).reshape(-1, 2)
    second = np.asarray(b, dtype=float).reshape(-1, 2)
    if not align:
        return {
            "distance": hausdorff(first, second),
            "aligned": False,
            "grid": None,
        }

    def _place(points: np.ndarray) -> np.ndarray:
        centre = points.mean(axis=0)
        centred = points - centre
        scale = float(np.sqrt((centred**2).sum(axis=1).mean())) or 1.0
        return centred / scale

    base_first = _place(first)
    base_second = _place(second)
    scales = np.linspace(1.0 / n_scales, float(n_scales), n_scales)
    best = math.inf
    best_grid: dict[str, float] | None = None
    radius = float(np.sqrt((base_second**2).sum(axis=1).mean())) or 1.0
    for index in range(n_rotations):
        angle = 2 * math.pi * index / n_rotations
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        rotated = base_second @ rotation.T
        for factor in scales:
            candidate = rotated * factor * radius
            value = hausdorff(base_first, candidate)
            if value < best:
                best = value
                best_grid = {"angle_rad": angle, "scale": float(factor), "radius": radius}
    return {
        "distance": float(best),
        "aligned": True,
        "grid": best_grid,
        "grid_resolution": {"n_rotations": n_rotations, "n_scales": n_scales},
        "note": (
            "the true minimum over the similarity group is approximated by a discrete search; "
            "the grid is reported so the approximation is visible rather than implied"
        ),
    }


def kozlov_sketch_distance_bounds(
    sketch_distance: float, r1: float, r2: float
) -> tuple[float, float]:
    """Теорема 4's two-sided bound for the distance between the originals.

    From the theorem's proof (recovered from ``.sources/kozlov2002.txt``):

        R_{r1 r2} <= R_0 + max(r1, r2),   R_0 <= R_{r1 r2} + max(r1, r2)

    hence ``R_0 ∈ [R_{r1r2} - max(r1,r2), R_{r1r2} + max(r1,r2)]``, where ``R_{r1r2}`` is the
    distance between the sketches and ``R_0 = B(A,B)`` the distance between the originals.

    The bound is *tight in the direction the source emphasises*: as the cover radii shrink the
    interval narrows, which is what makes the sketch distance a usable substitute.
    """
    if r1 < 0 or r2 < 0:
        raise ValueError("cover radii must be non-negative")
    r = max(float(r1), float(r2))
    return (float(sketch_distance) - r, float(sketch_distance) + r)


def certified_nearest(
    query: Any, candidates: Sequence[Any], radii: Sequence[float], *, r_query: float
) -> dict[str, Any]:
    """The source's certified nearest neighbour from sketch distances.

    > «...пусть интервал для $B_1$ **не пересекается** с интервалами для остальных $B_i$.
    > Тогда **заведомо** ... расстояние $B(A,B_1)$ наименьшее.»

    Computes each candidate's interval from :func:`kozlov_sketch_distance_bounds` and reports
    whether one candidate is *provably* the nearest.  When intervals overlap the answer is
    ``certified=False`` — the honest outcome, and the one a plain "argmin" would hide.
    """
    if len(candidates) != len(radii):
        raise ValueError("one cover radius is required per candidate")
    rows: list[dict[str, Any]] = []
    for candidate, r_candidate in zip(candidates, radii):
        distance = kozlov_image_distance(query, candidate, align=False)["distance"]
        low, high = kozlov_sketch_distance_bounds(distance, r_query, float(r_candidate))
        rows.append(
            {"sketch_distance": distance, "interval": [low, high], "r": float(r_candidate)}
        )

    best_index = int(np.argmin([row["sketch_distance"] for row in rows])) if rows else -1
    certified = False
    if best_index >= 0:
        best_high = rows[best_index]["interval"][1]
        others_low = [row["interval"][0] for position, row in enumerate(rows) if position != best_index]
        certified = all(best_high < low for low in others_low) if others_low else True
    return {
        "nearest_index": best_index if certified else None,
        "certified": certified,
        "rows": rows,
        "note": (
            "intervals overlap, so no candidate is *proven* nearest; a plain argmin would have "
            "returned an answer with no guarantee behind it"
            if not certified
            else "intervals are disjoint, so the nearest is certified by the theorem"
        ),
    }


# --------------------------------------------------------------------------- #
# compressors
# --------------------------------------------------------------------------- #
KOZLOV_SOURCE = (
    "Kozlov 2002, «Об одном подходе к сжатию информации при обработке и распознавании "
    "изображений», ЖВМиМФ 42:2, 256-268 (primary full text)"
)


class KozlovSketch(Compressor):
    """An эскиз at a requested radius ``r`` (greedy cover, guaranteed coverage).

    The *definition* is the source's; the *solver* is a greedy cover, so the cardinality claim
    is ``upper_bound``: the result is a valid эскиз, not necessarily an остов.
    """

    contract = CompressorContract(
        name="KozlovSketch",
        family="official",
        surjective_on=True,
        cover_guarantee="guaranteed",
        cardinality_guarantee="upper_bound",
        deterministic=True,
        complexity="O(n^2) to build the distance matrix, O(n^2) for the cover",
        source=KOZLOV_SOURCE + " (definition); greedy solver is this project's",
        transform_equivariance={"measured_in": "experiments/results/phase5_equivariance.csv"},
    )

    def fit(self, points: Any, *, r: float, **_: Any) -> CompressionResult:
        if r < 0:
            raise ValueError("the covering radius r must be non-negative")
        matrix = self._matrix(points)
        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        indices = greedy_sketch(matrix, float(r))
        # rescan to the last uncovered point so the double loop above cannot be left early
        if not indices:
            tracemalloc.stop()
            raise AssertionError("greedy_sketch returned nothing")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        achieved = directed_hausdorff(matrix, matrix[indices])
        if not verify_sketch(matrix, matrix[indices], float(r)):
            raise AssertionError(
                f"KozlovSketch failed its own guarantee: achieved radius {achieved:.3e} > r = {r:.3e}"
            )
        return self._result(
            matrix,
            indices,
            self.contract,
            requested_radius=float(r),
            achieved_radius=achieved,
            build_ms=elapsed_ms,
            peak_bytes=int(peak),
            diagnostics={"solver": "greedy cover", "guarantee_verified": True},
        )


class KozlovSkeleton(Compressor):
    """The остов: the minimum-cardinality эскиз at radius ``r``.

    The exact branch-and-bound path reports ``proved_minimum=True``.  When the node budget is
    exceeded the greedy solution is returned with ``proved_minimum=False`` and the contract's
    cardinality claim downgraded to ``upper_bound`` — the object then is a valid эскиз but is
    **not** claimed to be an остов.
    """

    contract = CompressorContract(
        name="KozlovSkeleton",
        family="official",
        surjective_on=True,
        cover_guarantee="guaranteed",
        cardinality_guarantee="minimum",
        deterministic=True,
        complexity="exact branch and bound (sets: O(n^2)); greedy fallback O(n^2)",
        source=KOZLOV_SOURCE + " (definition); solver is this project's",
        transform_equivariance={"measured_in": "experiments/results/phase5_equivariance.csv"},
    )

    def fit(self, points: Any, *, r: float, node_budget: int = 500_000, **_: Any) -> CompressionResult:
        if r < 0:
            raise ValueError("the covering radius r must be non-negative")
        matrix = self._matrix(points)
        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        size, indices, proved = minimum_cover_size(matrix, float(r), node_budget=node_budget)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        achieved = directed_hausdorff(matrix, matrix[indices])
        if not verify_sketch(matrix, matrix[indices], float(r)):
            raise AssertionError("KozlovSkeleton failed its own coverage guarantee")

        contract = self.contract
        if not proved:
            contract = CompressorContract(
                **{**self.contract.as_dict(), "cardinality_guarantee": "upper_bound",
                   "transform_equivariance": dict(self.contract.transform_equivariance)}
            )
        return self._result(
            matrix,
            indices,
            contract,
            requested_radius=float(r),
            achieved_radius=achieved,
            build_ms=elapsed_ms,
            peak_bytes=int(peak),
            diagnostics={
                "proved_minimum": proved,
                "claimed_size": size,
                "solver": "branch and bound" if proved else "greedy fallback (budget exceeded)",
            },
        )
