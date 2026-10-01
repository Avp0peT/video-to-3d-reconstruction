"""Exact geometric affine-equivalence oracle — independent of the Kozlov code.

Why a separate module
---------------------
The point of this route is to be a **cross-check**, so it must not share machinery with the
thing it checks.  Its dependencies are deliberately narrow:

* :func:`~kozlov_lab.geometry.signed_double_area` — the Phase 1 geometry layer, reviewed and
  tested long before any of this existed;
* :func:`~kozlov_lab.coding.validation._as_exact_points` — so that "what counts as exact
  input" has **one** definition rather than two.

It imports **nothing** from :mod:`~kozlov_lab.coding.exact`,
:mod:`~kozlov_lab.coding.numeric`, :mod:`~kozlov_lab.coding.model` or
:mod:`~kozlov_lab.coding.equivalence`.  In particular it never constructs a
:class:`~kozlov_lab.coding.model.RatioEntry`, never looks at ``T_A``, and never computes a
ratio.  It answers one question: *is there an affine map carrying one point set onto the
other?* — and if so, it returns the map.

Why not enumerate the ``n!`` point bijections
---------------------------------------------
Because it is unnecessary.  An affine map is determined by the image of **any**
non-degenerate triple, so: fix one non-degenerate triple of ``X``; enumerate the ordered
non-degenerate triples of ``Y``; solve for ``M, t`` exactly; then test whether the images
of all of ``X`` reproduce ``Y`` as a **set**.  The set comparison needs no correspondence
guess: with exact rational arithmetic, ``sorted(T(X)) == sorted(Y)`` is an exact equality.

Complexity is ``O(n^3)`` candidate maps, each verified in ``O(n)`` — instead of ``n!``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
from typing import Any, Mapping, Sequence

from ..geometry.affine_frame import signed_double_area
from .validation import ExactPoints, _as_exact_points

__all__ = [
    "GeometricWitness",
    "AffineEquivalenceVerdict",
    "find_exact_affine_witness",
    "check_geometric_equivalence",
    "solve_affine_from_three",
]


@dataclass(frozen=True)
class GeometricWitness:
    """Explicit constructive evidence: a map and the bijection it induces.

    Evidence *kind A* of the specification.  A witness is a proof: it can be verified by
    applying the map, which is why a witness and a "yes" answer are not the same thing.
    """

    matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    translation: tuple[Fraction, Fraction]
    point_mapping: Mapping[int, int]
    control_triples: tuple[tuple[int, int, int], tuple[int, int, int]]
    n_points: int

    @property
    def determinant(self) -> Fraction:
        return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]

    def apply(self, point: Sequence[Fraction]) -> tuple[Fraction, Fraction]:
        return (
            self.matrix[0][0] * point[0] + self.matrix[0][1] * point[1] + self.translation[0],
            self.matrix[1][0] * point[0] + self.matrix[1][1] * point[1] + self.translation[1],
        )

    def verify(self, x: ExactPoints, y: ExactPoints) -> bool:
        """Re-apply the map and re-check the correspondence.  O(n), exact."""
        if len(x) != len(y):
            return False
        for source_index, target_index in self.point_mapping.items():
            if self.apply(x[source_index]) != y[target_index]:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": [[str(value) for value in row] for row in self.matrix],
            "translation": [str(value) for value in self.translation],
            "determinant": str(self.determinant),
            "point_mapping": dict(sorted(self.point_mapping.items())),
            "control_triples": [list(self.control_triples[0]), list(self.control_triples[1])],
            "n_points": self.n_points,
        }

    def describe(self, lang: str = "zh") -> str:
        rows = "  ".join(f"({row[0]}, {row[1]})" for row in self.matrix)
        head = {
            "zh": "找到显式仿射映射（构造性证据）",
            "en": "explicit affine map found (constructive evidence)",
        }.get(lang, "witness")
        return f"{head}\n  M = {rows}\n  t = ({self.translation[0]}, {self.translation[1]})\n  det M = {self.determinant}"


@dataclass(frozen=True)
class AffineEquivalenceVerdict:
    """Outcome of the geometric route, including what it did *not* manage to decide."""

    equivalent: bool
    exhaustive: bool
    undecided: bool
    witness: GeometricWitness | None
    candidates_tried: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "exhaustive": self.exhaustive,
            "undecided": self.undecided,
            "candidates_tried": self.candidates_tried,
            "reason": self.reason,
            "witness": self.witness.as_dict() if self.witness else None,
        }

    def describe(self, lang: str = "zh") -> str:
        if self.undecided:
            verdict = {"zh": "无法判定", "en": "undecided"}.get(lang, "undecided")
        elif self.equivalent:
            verdict = {"zh": "存在仿射映射", "en": "an affine map exists"}.get(lang, "yes")
        else:
            verdict = {
                "zh": "不存在仿射映射" if self.exhaustive else "未找到仿射映射（搜索受限）",
                "en": "no affine map exists" if self.exhaustive else "none found (search bounded)",
            }.get(lang, "no")
        return f"{verdict}\n  reason: {self.reason}\n  candidates tried: {self.candidates_tried}"


def solve_affine_from_three(
    source: Sequence[Sequence[Fraction]], target: Sequence[Sequence[Fraction]]
) -> tuple[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], tuple[Fraction, Fraction]] | None:
    """Solve ``M u = du`` and ``M v = dv`` exactly, or ``None`` if degenerate.

    With ``u = s1 - s0``, ``v = s2 - s0`` (columns of ``U``) and ``du``, ``dv`` the matching
    target differences (columns of ``D``), the map is ``M = D U^{-1}``:

        M = (1/det U) [[du0, dv0], [du1, dv1]] [[v1, -v0], [-u1, u0]]

    Written out entry by entry.  Getting the transpose/companion pattern wrong is easy and
    silent, so the function **verifies its own output** on the two spanning vectors before
    returning — an O(1) check that converts a whole class of sign errors into a loud
    assertion.
    """
    u = (source[1][0] - source[0][0], source[1][1] - source[0][1])
    v = (source[2][0] - source[0][0], source[2][1] - source[0][1])
    du = (target[1][0] - target[0][0], target[1][1] - target[0][1])
    dv = (target[2][0] - target[0][0], target[2][1] - target[0][1])

    det = u[0] * v[1] - u[1] * v[0]
    if det == 0:
        return None

    matrix = (
        (Fraction(du[0] * v[1] - dv[0] * u[1], det), Fraction(dv[0] * u[0] - du[0] * v[0], det)),
        (Fraction(du[1] * v[1] - dv[1] * u[1], det), Fraction(dv[1] * u[0] - du[1] * v[0], det)),
    )
    assert (matrix[0][0] * u[0] + matrix[0][1] * u[1], matrix[1][0] * u[0] + matrix[1][1] * u[1]) == du  # noqa: S101
    assert (matrix[0][0] * v[0] + matrix[0][1] * v[1], matrix[1][0] * v[0] + matrix[1][1] * v[1]) == dv  # noqa: S101

    translation = (
        target[0][0] - (matrix[0][0] * source[0][0] + matrix[0][1] * source[0][1]),
        target[0][1] - (matrix[1][0] * source[0][0] + matrix[1][1] * source[0][1]),
    )
    return matrix, translation


def _non_degenerate_triples(points: ExactPoints) -> list[tuple[int, int, int]]:
    return [
        triple
        for triple in itertools.combinations(range(len(points)), 3)
        if signed_double_area(points[triple[0]], points[triple[1]], points[triple[2]]) != 0
    ]


def _recover_mapping(
    x: ExactPoints, y: ExactPoints, matrix: Any, translation: Any
) -> dict[int, int] | None:
    """Recover the bijection induced by the map, or ``None`` when it is not unique."""
    remaining = list(range(len(y)))
    mapping: dict[int, int] = {}
    for index, point in enumerate(x):
        image = (
            matrix[0][0] * point[0] + matrix[0][1] * point[1] + translation[0],
            matrix[1][0] * point[0] + matrix[1][1] * point[1] + translation[1],
        )
        hits = [candidate for candidate in remaining if y[candidate] == image]
        if len(hits) != 1:
            return None  # zero hits cannot happen here; two hits means duplicate coordinates
        mapping[index] = hits[0]
        remaining.remove(hits[0])
    return mapping


def find_exact_affine_witness(x: Any, y: Any) -> GeometricWitness | None:
    """Return an explicit affine map carrying ``X`` onto ``Y``, or ``None``.

    Exact rational arithmetic throughout: no tolerance, no ``allclose``.  The final
    comparison is equality of two sorted lists of rational pairs, which is an exact
    statement about the two point *sets* and needs no correspondence guess.
    """
    witness, _ = _search_witness_with_count(x, y)
    return witness


def _search_witness_with_count(x: Any, y: Any) -> tuple[GeometricWitness | None, int]:
    """The search itself, also reporting how many candidate maps were examined.

    The count is not decoration: ``check_geometric_equivalence`` needs it to state that a
    negative answer came from an *exhaustive* enumeration rather than a bounded one.
    """
    exact_x = _as_exact_points(x)
    exact_y = _as_exact_points(y)
    if len(exact_x) != len(exact_y) or not exact_x:
        return None

    control_x = _non_degenerate_triples(exact_x)
    if not control_x:
        return None
    base = control_x[0]
    base_points = (exact_x[base[0]], exact_x[base[1]], exact_x[base[2]])
    target_multiset = sorted(exact_y)
    tried = 0

    for ordered in itertools.permutations(range(len(exact_y)), 3):
        target_points = (exact_y[ordered[0]], exact_y[ordered[1]], exact_y[ordered[2]])
        solved = solve_affine_from_three(base_points, target_points)
        tried += 1
        if solved is None:
            continue
        matrix, translation = solved
        images = sorted(
            (
                matrix[0][0] * point[0] + matrix[0][1] * point[1] + translation[0],
                matrix[1][0] * point[0] + matrix[1][1] * point[1] + translation[1],
            )
            for point in exact_x
        )
        if images != target_multiset:
            continue
        mapping = _recover_mapping(exact_x, exact_y, matrix, translation)
        if mapping is None:
            continue
        return (
            GeometricWitness(
                matrix=matrix,
                translation=translation,
                point_mapping=mapping,
                control_triples=(base, ordered),
                n_points=len(exact_x),
            ),
            tried,
        )
    return None, tried


def check_geometric_equivalence(x: Any, y: Any) -> AffineEquivalenceVerdict:
    """Decide affine equivalence **without** any reference to the Kozlov code.

    The verdict records whether the search was *exhaustive*, because
    "no map was found" and "I proved no map exists" are different answers:

    * when ``X`` has a non-degenerate triple, every candidate map is enumerated, so a
      negative answer is a proof — ``exhaustive=True``;
    * when all points are collinear there is no control triple to anchor the search, and the
      route reports ``undecided`` rather than guessing.
    """
    exact_x = _as_exact_points(x)
    exact_y = _as_exact_points(y)
    if len(exact_x) != len(exact_y):
        return AffineEquivalenceVerdict(
            equivalent=False,
            exhaustive=True,
            undecided=False,
            witness=None,
            candidates_tried=0,
            reason=f"cardinality differs: {len(exact_x)} vs {len(exact_y)}",
        )
    controls = _non_degenerate_triples(exact_x)
    if not controls:
        return AffineEquivalenceVerdict(
            equivalent=True,
            exhaustive=False,
            undecided=True,
            witness=None,
            candidates_tried=0,
            reason=(
                "every triple of X is degenerate, so there is no control triple to anchor the "
                "search; this route does not cover 1-dimensional images"
            ),
        )

    witness, tried = _search_witness_with_count(exact_x, exact_y)
    if witness is not None:
        return AffineEquivalenceVerdict(
            equivalent=True,
            exhaustive=True,
            undecided=False,
            witness=witness,
            candidates_tried=tried,
            reason="an affine map reproducing Y as a set was found; it can be re-verified in O(n)",
        )
    return AffineEquivalenceVerdict(
        equivalent=False,
        exhaustive=True,
        undecided=False,
        witness=None,
        candidates_tried=tried,
        reason=(
            f"all {tried} ordered non-degenerate triples of Y were tried; none yields a map "
            "reproducing X onto Y — a proof, not a search timeout"
        ),
    )
