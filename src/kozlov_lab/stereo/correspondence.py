"""The correspondence problem (6E–6H) — **EXPERIMENTAL**, ours.

The source names correspondence as *the* main problem («Основной вопрос здесь — проблема
установления соответствия между точками изображений на разных проекциях») and its own mechanism
is structural: fix a triangle plus an off-plane point.  That principle is quoted in
``official/kozlov_code3d.py``; the *search* over correspondences implemented here is not from the
source, so it is labelled EXPERIMENTAL and kept out of ``official/``.

Two engines, deliberately:

* :func:`oracle_correspondence` — brute force over all ``n!`` bijections for ``n <= 8``.  It is the
  **ground-truth oracle**, not a method: its job is to be what the efficient engine is tested
  against (G6.5, G6.6);
* :func:`match_correspondences` — epipolar pruning plus branch-and-bound.  Same scores, far fewer
  hypotheses, and it must agree with the oracle or the disagreement is a finding.

Why a bare boolean is not the output: a symmetric body admits **several equally good**
correspondences, and calling that "failure" would misreport a property of the object as a defect of
the method — the same distinction Phase 4 had to make for ``psi``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "CorrespondenceVerdict",
    "CorrespondenceResult",
    "EngineAgreement",
    "RecoveryOutcome",
    "oracle_correspondence",
    "match_correspondences",
    "engines_agree",
    "reconstruction_outcome",
]


class CorrespondenceVerdict(str, Enum):
    """Four outcomes.  ``AMBIGUOUS`` is a result, not an error."""

    UNIQUE = "UNIQUE"
    AMBIGUOUS = "AMBIGUOUS"
    NO_CONSISTENT_MAPPING = "NO_CONSISTENT_MAPPING"
    DEGENERATE_GEOMETRY = "DEGENERATE_GEOMETRY"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class EngineAgreement(str, Enum):
    """How the pruned engine's answer relates to the oracle's — **verdict-aware**.

    A bare ``pruned.mapping == oracle.mapping`` is the wrong comparison, and it fails in a way
    that manufactures a *false defect*: when the body is symmetric (or two body points share a
    ray in one view) several bijections have **exactly** zero cost, so "which one" is genuinely
    undetermined and two correct engines legitimately return different representatives.  Comparing
    raw dictionaries would then report a disagreement out of a tie.

    So the comparison is split by outcome:

    ``EXACT``                both engines returned the same bijection;
    ``AMBIGUOUS_CONSISTENT`` the answers differ, but the engine's answer is a member of the
                             oracle's optimum set — a tie, not an error;
    ``DISAGREEMENT``         the engine's answer is **not** an optimum: a real defect;
    ``PRUNED_FAILED``        the engine produced no mapping at all.

    ``DISAGREEMENT`` is the falsifying counterpart of ``AMBIGUOUS_CONSISTENT``: without a way to
    produce it, "the engines agree" would be a claim that cannot fail.
    """

    EXACT = "EXACT"
    AMBIGUOUS_CONSISTENT = "AMBIGUOUS_CONSISTENT"
    DISAGREEMENT = "DISAGREEMENT"
    PRUNED_FAILED = "PRUNED_FAILED"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class RecoveryOutcome(str, Enum):
    """What the recovered correspondence actually reconstructs.

    This exists because **zero reprojection error does not imply a correct correspondence**.
    A body with two points on a common ray from one camera admits several bijections with cost
    exactly 0, and some of them reconstruct a *different* body.  A pipeline that only reports
    ``reprojection_error ≈ 0`` therefore reports success on a wrong answer.

    ``EXACT_MATCH``             the mapping is the true one;
    ``EQUIVALENT_VIA_SYMMETRY`` the mapping differs, but the reconstruction is 3D affine
                                equivalent to the body — a legitimate answer (théorème 3);
    ``INVALID_RECONSTRUCTION``  the mapping differs and the reconstruction is **not** equivalent:
                                the dangerous case above;
    ``UNDECIDED``               the equivalence test abstained (every quadruple of the body is
                                coplanar, so :func:`~kozlov_lab.stereo.is_a_equivalent` has no
                                control quadruple).  This exists because folding it into
                                ``EQUIVALENT_VIA_SYMMETRY`` would turn an *abstention* into a
                                **positive result** — the same three-valued discipline the 2D
                                phases enforce, where ``vacuous`` must never render as a pass;
    ``NO_CORRESPONDENCE``       the engine returned nothing.
    """

    NO_CORRESPONDENCE = "NO_CORRESPONDENCE"
    EXACT_MATCH = "EXACT_MATCH"
    EQUIVALENT_VIA_SYMMETRY = "EQUIVALENT_VIA_SYMMETRY"
    INVALID_RECONSTRUCTION = "INVALID_RECONSTRUCTION"
    UNDECIDED = "UNDECIDED"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class CorrespondenceResult:
    """The mapping, its score, and **how close the runner-up is**."""

    mapping: Mapping[int, int] | None
    verdict: CorrespondenceVerdict
    cost: float
    runner_up_cost: float | None
    confidence_gap: float | None
    n_valid_mappings: int
    candidate_counts: tuple[int, ...]
    reprojection_error: float
    degenerate: str | None
    #: Every bijection within ``ambiguity_ratio`` of the optimum, as index tuples.
    #: Only the oracle can enumerate these, and only they make ``engines_agree`` meaningful:
    #: without the optimum set there is no way to tell a tie from a defect.
    good_mappings: tuple[tuple[int, ...], ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ambiguous(self) -> bool:
        return self.verdict is CorrespondenceVerdict.AMBIGUOUS

    @property
    def permutation(self) -> tuple[int, ...] | None:
        """The mapping as an index tuple ``(column_of_0, column_of_1, ...)``."""
        if self.mapping is None:
            return None
        return tuple(int(self.mapping[i]) for i in range(len(self.mapping)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "mapping": dict(sorted(self.mapping.items())) if self.mapping else None,
            "cost": self.cost,
            "runner_up_cost": self.runner_up_cost,
            "confidence_gap": self.confidence_gap,
            "n_valid_mappings": self.n_valid_mappings,
            "reprojection_error": self.reprojection_error,
            "degenerate": self.degenerate,
            "n_good_mappings": len(self.good_mappings),
            "diagnostics": dict(self.diagnostics),
        }


# --------------------------------------------------------------------------- #
# pair scoring
# --------------------------------------------------------------------------- #
def _pair_cost(camera_a: Any, camera_b: Any, left: np.ndarray, right: np.ndarray, i: int, j: int) -> float:
    """Cost of hypothesising that ``left[i]`` and ``right[j]`` are the same 3D point.

    Triangulate, then measure how far the reconstruction lands from **both** observations.  A
    degenerate pair (failed triangulation) costs ``inf`` rather than being given a large finite
    value, so it can never win by accident.
    """
    from .baseline.triangulate import triangulate_dlt

    try:
        point = triangulate_dlt(camera_a.projection, left[i], camera_b.projection, right[j])
    except ValueError:
        return math.inf
    if not np.all(np.isfinite(point)):
        return math.inf
    back_a = camera_a.project(point)
    back_b = camera_b.project(point)
    return float(
        ((back_a - left[i]) ** 2).sum() + ((back_b - right[j]) ** 2).sum()
    )


def _cost_matrix(camera_a: Any, camera_b: Any, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    n = left.shape[0]
    matrix = np.full((n, n), math.inf)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = _pair_cost(camera_a, camera_b, left, right, i, j)
    return matrix


def _total_cost(matrix: np.ndarray, permutation: Sequence[int]) -> float:
    total = 0.0
    for index, column in enumerate(permutation):
        value = matrix[index, column]
        if not math.isfinite(value):
            return math.inf
        total += value
    return total


def _image_scale(*images: np.ndarray) -> float:
    """The image extent, used to turn a pixel tolerance into a cost tolerance."""
    stacked = np.vstack([np.asarray(a, dtype=float).reshape(-1, 2) for a in images])
    return float(np.ptp(stacked, axis=0).max()) or 1.0


def _tie_tolerance(best_cost: float, image_scale: float, *, tie_relative_pixels: float) -> float:
    """The cost window inside which a second correspondence is a **tie**, not a runner-up.

    Dimensionless the way this project requires: the tolerance is a **sub-pixel fraction of the
    image extent**, squared (the cost is a sum of squared pixel distances), so it follows the data
    instead of being a fixed number.

    Why a bare ratio will not do — this was a real defect.  The first version compared costs as
    ``cost <= best_cost * 1.05``.  An exact synthetic fit has ``best_cost ~ 1e-26``, so the window
    became ``[1.04e-26, 1.10e-26]`` — a range that separates **floating-point noise**.  The *true*
    correspondence then fell outside the "optimum set" while four wrong ones fell inside, and
    ``engines_agree`` would have reported a correct engine as a disagreement.  A tolerance applied
    to the noise level of the quantity it measures manufactures false defects.

    The returned value is an **absolute** cost window (``best + tol`` is the comparison), and it is
    derived from the image scale, so no dimensionful constant is hard-coded.
    """
    del best_cost  # kept in the signature to document that the window is *not* a ratio of it
    return (tie_relative_pixels * image_scale) ** 2


def _epipolar_filter(matrix: np.ndarray, *, keep: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Keep the ``keep`` cheapest candidates per left point — candidate generation (6H).

    A **pruning step**, not a decision: the oracle has no such limit, so the differential test
    between the two engines measures exactly what the pruning costs in correctness.  That is the
    point of having both.
    """
    pruned = np.full_like(matrix, math.inf)
    counts: list[int] = []
    for row in range(matrix.shape[0]):
        order = np.argsort(matrix[row], kind="stable")
        chosen = [c for c in order if math.isfinite(matrix[row, c])][:keep]
        counts.append(len(chosen))
        for column in chosen:
            pruned[row, column] = matrix[row, column]
    return pruned, tuple(counts)


def _branch_and_bound(matrix: np.ndarray, *, budget: int = 200_000) -> tuple[list[int] | None, int, bool]:
    """Assignment by depth-first search with a cheapest-remaining lower bound.

    Returns ``(permutation, nodes, proved)``; ``proved=False`` means the budget ran out and the
    incumbent is not claimed optimal.
    """
    n = matrix.shape[0]
    order = sorted(range(n), key=lambda row: float(np.nanmin(matrix[row])) if np.isfinite(matrix[row]).any() else math.inf)
    best: list[int] | None = None
    best_cost = math.inf
    nodes = 0
    proved = True

    def recurse(position: int, used: set[int], cost: float, partial: list[int]) -> None:
        nonlocal best, best_cost, nodes, proved
        nodes += 1
        if nodes > budget:
            proved = False
            return
        if position == n:
            if cost < best_cost:
                best_cost, best = cost, list(partial)
            return
        row = order[position]
        remaining = [c for c in range(n) if c not in used and math.isfinite(matrix[row, c])]
        for column in sorted(remaining, key=lambda c: matrix[row, c]):
            new_cost = cost + float(matrix[row, column])
            if new_cost >= best_cost:
                continue
            used.add(column)
            partial.append(column)
            recurse(position + 1, used, new_cost, partial)
            partial.pop()
            used.remove(column)
            if not proved:
                return

    recurse(0, set(), 0.0, [])
    if best is None:
        return None, nodes, proved
    # put the incumbent back in row order
    permutation = [0] * n
    for position, row in enumerate(order):
        permutation[row] = best[position]
    return permutation, nodes, proved


# --------------------------------------------------------------------------- #
# the two engines
# --------------------------------------------------------------------------- #
def oracle_correspondence(
    camera_a: Any, camera_b: Any, left: Any, right: Any, *,
    tie_relative_pixels: float = 1e-6, max_n: int = 8,
) -> CorrespondenceResult:
    """Brute force over every bijection — the **ground-truth oracle** (6F).

    ``tie_relative_pixels`` defines "equally good": a second mapping whose cost is within that
    sub-pixel fraction of the optimum counts as a genuine alternative rather than a runner-up.
    It is an input, not a hidden constant, because "how close is close enough" is a modelling
    choice — but it is expressed **relative to the image scale**, see :func:`_tie_tolerance` for
    why a ratio of the optimum is unsound here.
    """
    image_left = np.asarray(left, dtype=float).reshape(-1, 2)
    image_right = np.asarray(right, dtype=float).reshape(-1, 2)
    if image_left.shape != image_right.shape:
        raise ValueError("both images must have the same number of points")
    n = image_left.shape[0]
    if n > max_n:
        raise ValueError(
            f"the oracle enumerates n! bijections and is capped at n = {max_n}; got {n}. "
            "The cap is the point: the oracle is a reference, not a method."
        )
    if n == 0:
        raise ValueError("an image pair must be non-empty")

    matrix = _cost_matrix(camera_a, camera_b, image_left, image_right)
    costs: list[tuple[float, tuple[int, ...]]] = []
    for permutation in itertools.permutations(range(n)):
        value = _total_cost(matrix, permutation)
        if math.isfinite(value):
            costs.append((value, permutation))
    costs.sort(key=lambda item: item[0])

    if not costs:
        return CorrespondenceResult(
            mapping=None, verdict=CorrespondenceVerdict.DEGENERATE_GEOMETRY, cost=math.inf,
            runner_up_cost=None, confidence_gap=None, n_valid_mappings=0,
            candidate_counts=tuple(int(np.isfinite(matrix[row]).sum()) for row in range(n)),
            reprojection_error=math.inf, degenerate="no bijection has a finite triangulation cost",
            diagnostics={"engine": "oracle"},
        )

    scale = _image_scale(image_left, image_right)
    best_cost, best_permutation = costs[0]
    window = _tie_tolerance(best_cost, scale, tie_relative_pixels=tie_relative_pixels)
    good = [c for c in costs if c[0] <= best_cost + window]
    runner_up = costs[1][0] if len(costs) > 1 else None
    mapping = {row: int(column) for row, column in enumerate(best_permutation)}
    verdict = (
        CorrespondenceVerdict.AMBIGUOUS if len(good) > 1 else CorrespondenceVerdict.UNIQUE
    )
    return CorrespondenceResult(
        mapping=mapping, verdict=verdict, cost=best_cost, runner_up_cost=runner_up,
        confidence_gap=(None if runner_up is None else runner_up - best_cost),
        n_valid_mappings=len(good),
        candidate_counts=tuple(int(np.isfinite(matrix[row]).sum()) for row in range(n)),
        reprojection_error=_reprojection_of(camera_a, camera_b, image_left, image_right, best_permutation),
        degenerate=None,
        # the optimum set is what makes `engines_agree` a real test rather than a dict comparison
        good_mappings=tuple(sorted(tuple(p) for _, p in good)),
        diagnostics={
            "engine": "oracle",
            "n_bijections_examined": math.factorial(n),
            "n_finite_bijections": len(costs),
            "tie_relative_pixels": tie_relative_pixels,
            "tie_cost_window": window,
            "image_scale": scale,
        },
    )


def match_correspondences(
    camera_a: Any, camera_b: Any, left: Any, right: Any, *,
    candidates_per_point: int = 3, tie_relative_pixels: float = 1e-6, budget: int = 200_000,
) -> CorrespondenceResult:
    """Candidate generation → pruning → assignment → triangulation → scoring (6H).

    Fewer hypotheses than the oracle by construction, and the same scoring function, so any
    difference in the answer is attributable to the pruning alone.
    """
    image_left = np.asarray(left, dtype=float).reshape(-1, 2)
    image_right = np.asarray(right, dtype=float).reshape(-1, 2)
    if image_left.shape != image_right.shape:
        raise ValueError("both images must have the same number of points")
    n = image_left.shape[0]
    matrix = _cost_matrix(camera_a, camera_b, image_left, image_right)
    if candidates_per_point < 1:
        raise ValueError("at least one candidate per point is required")
    pruned, counts = _epipolar_filter(matrix, keep=candidates_per_point)

    permutation, nodes, proved = _branch_and_bound(pruned, budget=budget)
    if permutation is None:
        return CorrespondenceResult(
            mapping=None, verdict=CorrespondenceVerdict.NO_CONSISTENT_MAPPING, cost=math.inf,
            runner_up_cost=None, confidence_gap=None, n_valid_mappings=0,
            candidate_counts=counts, reprojection_error=math.inf, degenerate=None,
            diagnostics={"engine": "pruned", "nodes": nodes, "proved": proved,
                         "note": "the pruned candidate sets admit no perfect matching"},
        )

    scale = _image_scale(image_left, image_right)
    best_cost = _total_cost(matrix, permutation)
    window = _tie_tolerance(best_cost, scale, tie_relative_pixels=tie_relative_pixels)
    # ambiguity is re-checked on the *pruned* hypothesis space: the pruned engine cannot see
    # alternatives its own filter removed, and saying otherwise would overstate its knowledge
    alternatives = 0
    for other in itertools.permutations(range(n)):
        if other == tuple(permutation):
            continue
        value = _total_cost(pruned, other)
        if math.isfinite(value) and value <= best_cost + window:
            alternatives += 1
    mapping = {row: int(column) for row, column in enumerate(permutation)}
    return CorrespondenceResult(
        mapping=mapping,
        verdict=CorrespondenceVerdict.AMBIGUOUS if alternatives else CorrespondenceVerdict.UNIQUE,
        cost=best_cost, runner_up_cost=None,
        confidence_gap=None, n_valid_mappings=1 + alternatives,
        candidate_counts=counts,
        reprojection_error=_reprojection_of(camera_a, camera_b, image_left, image_right, permutation),
        degenerate=None,
        diagnostics={
            "engine": "pruned", "nodes": nodes, "proved": proved,
            "candidates_per_point": candidates_per_point,
            "tie_relative_pixels": tie_relative_pixels,
            "tie_cost_window": window,
            "note": "ambiguity is assessed within the pruned hypothesis space only",
        },
    )


def _reprojection_of(camera_a: Any, camera_b: Any, left: np.ndarray, right: np.ndarray,
                     permutation: Sequence[int]) -> float:
    """Largest reprojection distance under a given correspondence."""
    from .baseline.triangulate import triangulate_many

    ordered_right = right[list(permutation)]
    points, failed = triangulate_many(camera_a.projection, left, camera_b.projection, ordered_right)
    if failed:
        return math.inf
    back_a = camera_a.project(points)
    back_b = camera_b.project(points)
    return float(
        max(
            np.linalg.norm(back_a - left, axis=1).max(),
            np.linalg.norm(back_b - ordered_right, axis=1).max(),
        )
    )


# --------------------------------------------------------------------------- #
# comparing the two engines — verdict-aware, so a tie is not reported as a defect
# --------------------------------------------------------------------------- #
def engines_agree(oracle: CorrespondenceResult, pruned: CorrespondenceResult) -> EngineAgreement:
    """Compare the pruned engine with the oracle **without confusing a tie for a defect**.

    The rule: an answer that differs from the oracle's is acceptable **iff it is itself an
    optimum**.  That is the only comparison the mathematics supports — the correspondence is
    determined up to the object's own symmetries, so demanding a particular representative would
    be asserting something the definition never said.

    A directly wrong answer (an optimum exists, but the engine returned a non-optimum) is
    ``DISAGREEMENT``, and that is the case the differential test must be able to produce.
    """
    if pruned.permutation is None:
        return EngineAgreement.PRUNED_FAILED
    if oracle.permutation is None:
        return EngineAgreement.DISAGREEMENT
    if pruned.permutation == oracle.permutation:
        return EngineAgreement.EXACT
    if pruned.permutation in oracle.good_mappings:
        return EngineAgreement.AMBIGUOUS_CONSISTENT
    return EngineAgreement.DISAGREEMENT


def reconstruction_outcome(
    result: CorrespondenceResult,
    camera_a: Any, camera_b: Any,
    image_left: Any, image_right: Any,
    reference_body: Any,
    *,
    true_mapping: Mapping[int, int] | None = None,
) -> RecoveryOutcome:
    """What the recovered mapping *reconstructs* — not merely its reprojection error.

    This is the stereo counterpart of the three-valued domain of Phase 3: "the reprojection fits"
    and "the body was recovered" are different questions, and a pipeline that reports only the
    first will report success on a wrong answer whenever the configuration is ambiguous.

    ``true_mapping`` is the generator's permutation when it is known (synthetic data); pass
    ``None`` when it is not, and the outcome collapses to the two-valued
    ``EQUIVALENT`` / ``INVALID`` distinction.
    """
    from .baseline.triangulate import triangulate_many
    from .official import is_a_equivalent

    permutation = result.permutation
    if permutation is None:
        return RecoveryOutcome.NO_CORRESPONDENCE

    left = np.asarray(image_left, dtype=float).reshape(-1, 2)
    right = np.asarray(image_right, dtype=float).reshape(-1, 2)
    points, failed = triangulate_many(
        camera_a.projection, left, camera_b.projection, right[list(permutation)]
    )
    if failed:
        return RecoveryOutcome.INVALID_RECONSTRUCTION

    equivalence = is_a_equivalent(points, reference_body)
    if equivalence.get("undecided"):
        # an abstention is not a positive result: with every quadruple coplanar there is no
        # control quadruple, so "equivalent" was never actually established.
        return RecoveryOutcome.UNDECIDED
    if not equivalence["equivalent"]:
        return RecoveryOutcome.INVALID_RECONSTRUCTION

    if true_mapping is not None:
        truth = tuple(int(true_mapping[i]) for i in range(len(true_mapping)))
        if permutation == truth:
            return RecoveryOutcome.EXACT_MATCH
    return RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY
