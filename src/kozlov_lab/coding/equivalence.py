"""Code equivalence, with two independent decision routes that cross-check each other.

The relation being implemented
------------------------------
Two codes are equivalent when there is a **bijection**

.. math::

    \\psi : M_A \\to M_B

that preserves every corresponding area ratio.  Note what the source does *not* say:
it does not require the codes to be literally identical as data structures.  Two
images whose points are numbered differently have different ``M_A`` and different
index sets inside every :class:`~kozlov_lab.coding.model.RatioEntry`, and they are
equivalent anyway — that is precisely the renumbering the definition is about.

This is why the model keeps the indexing: collapsing ``T_A`` into a bag of numbers
would make the *definition* unimplementable, not merely less efficient.

Two routes, deliberately
------------------------
1. **Code route** (:func:`codes_equivalent_exact` / :func:`codes_equivalent_numeric`)
   — search for :math:`\\psi` directly.  This is the route the theory describes.
2. **Geometric route** (:func:`affine_equivalent_exact`) — solve for an affine map
   from one control triple and check that the images coincide as *sets*, using exact
   rational arithmetic.  It knows nothing about codes.

Theorem 1 says the two must agree on the domain where it applies, so running both and
comparing is a genuine test of the implementation rather than of itself.  That is what
:func:`cross_check_exact` does, and it is the discipline the project applies
everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import itertools
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .model import (
    KozlovImageCode,
    RatioEntry,
    Triple,
)
from .exact import build_kozlov_image_code_exact
from .validation import ExactPoints, _as_exact_points

__all__ = [
    "KozlovCodeEquivalence",
    "AffineEquivalenceResult",
    "find_bijection",
    "codes_equivalent_exact",
    "codes_equivalent_numeric",
    "images_equivalent_by_code_exact",
    "affine_equivalent_exact",
    "cross_check_exact",
    "DEFAULT_SEARCH_BUDGET",
]

#: Upper bound on the number of candidate bijections verified before giving up.
#: Exceeding it yields ``undecided=True`` — *not* ``False*.  "I could not decide" and
#: "they are not equivalent" are different answers, and conflating them would turn a
#: budget limit into a false negative.
DEFAULT_SEARCH_BUDGET = 20_000


@dataclass(frozen=True)
class KozlovCodeEquivalence:
    """Outcome of the code route."""

    equivalent: bool
    undecided: bool
    reason: str
    bijection: Mapping[int, int] | None
    candidates_examined: int
    n_entries_a: int
    n_entries_b: int
    tolerance: float | None
    cross_check: Mapping[str, Any] | None = field(default=None)

    def describe(self, lang: str = "zh") -> str:
        if self.undecided:
            verdict = {"zh": "无法判定（搜索预算耗尽）", "en": "undecided (search budget exhausted)"}
        elif self.equivalent:
            verdict = {"zh": "代码等价", "en": "codes equivalent"}
        else:
            verdict = {"zh": "代码不等价", "en": "codes not equivalent"}
        text = verdict.get(lang, verdict["en"])
        lines = [
            f"{text}",
            f"reason: {self.reason}",
            f"entries: A={self.n_entries_a}, B={self.n_entries_b}; "
            f"candidates examined={self.candidates_examined}",
            f"tolerance: {self.tolerance if self.tolerance is not None else 'exact'}",
        ]
        if self.bijection:
            lines.append(f"psi: {dict(sorted(self.bijection.items()))}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "undecided": self.undecided,
            "reason": self.reason,
            "bijection": dict(sorted(self.bijection.items())) if self.bijection else None,
            "candidates_examined": self.candidates_examined,
            "n_entries_a": self.n_entries_a,
            "n_entries_b": self.n_entries_b,
            "tolerance": self.tolerance,
            "cross_check": dict(self.cross_check) if self.cross_check else None,
        }


@dataclass(frozen=True)
class AffineEquivalenceResult:
    """Outcome of the geometric route."""

    equivalent: bool
    undecided: bool
    reason: str
    matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] | None
    translation: tuple[Fraction, Fraction] | None
    control_triple: tuple[Triple, Triple] | None
    point_mapping: Mapping[int, int] | None

    def describe(self, lang: str = "zh") -> str:
        if self.undecided:
            verdict = {"zh": "无法判定", "en": "undecided"}
        elif self.equivalent:
            verdict = {"zh": "存在仿射变换", "en": "an affine map exists"}
        else:
            verdict = {"zh": "不存在仿射变换", "en": "no affine map exists"}
        return f"{verdict.get(lang, verdict['en'])}\nreason: {self.reason}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "undecided": self.undecided,
            "reason": self.reason,
            "matrix": [[str(v) for v in row] for row in self.matrix] if self.matrix else None,
            "translation": [str(v) for v in self.translation] if self.translation else None,
            "point_mapping": dict(sorted(self.point_mapping.items())) if self.point_mapping else None,
        }


# --------------------------------------------------------------------------- #
# route 1: search for the bijection psi
# --------------------------------------------------------------------------- #
def _verify_bijection(
    a: KozlovImageCode,
    b: KozlovImageCode,
    psi: Mapping[int, int],
    tolerance: float | None,
) -> bool:
    """Whether ``psi`` maps every entry of ``a`` onto an entry of ``b``.

    The check is done by *lookup on the index pair*, not by comparing values in bulk:
    an implementation that compared the two value multisets instead would accept
    bijections that scramble which triples produced which ratio — exactly the
    correlation structure the source's indexing encodes.
    """
    index_b = {(entry.numerator, entry.denominator): entry for entry in b.entries}
    for entry in a.entries:
        mapped_numerator = frozenset(psi[label] for label in entry.numerator)
        mapped_denominator = frozenset(psi[label] for label in entry.denominator)
        target = index_b.get((mapped_numerator, mapped_denominator))
        if target is None:
            return False
        if tolerance is None:
            if entry.value != target.value:
                return False
        elif abs(float(entry.value) - float(target.value)) > tolerance:
            return False
    return True


def find_bijection(
    a: KozlovImageCode,
    b: KozlovImageCode,
    *,
    tolerance: float | None = None,
    budget: int = DEFAULT_SEARCH_BUDGET,
) -> tuple[Mapping[int, int] | None, str, int, bool]:
    """Search for the renumbering :math:`\\psi` making the two codes correspond.

    Returns ``(bijection_or_None, reason, candidates_examined, undecided)``.

    The search is pruned by :meth:`KozlovImageCode.signature_classes`: a label's
    role/value fingerprint is invariant under any :math:`\\psi` that preserves the
    code, so labels can only correspond within equal fingerprint classes.  For
    general-position images those classes are singletons and the search is immediate;
    for highly symmetric ones the class sizes grow, which is exactly when the budget
    cap matters.
    """
    if len(a.labels) != len(b.labels):
        return None, f"label set sizes differ: {len(a.labels)} vs {len(b.labels)}", 0, False
    if a.domain != b.domain:
        # An affine map preserves collinearity and the two-parallel-lines property, so the
        # domain class is itself an affine invariant — this is a legitimate necessary
        # condition, not a shortcut.
        return (
            None,
            f"the two images are in different domain classes ({a.domain} vs {b.domain}); "
            "the domain class is an affine invariant, so they cannot correspond",
            0,
            False,
        )
    if len(a.entries) != len(b.entries):
        return (
            None,
            f"|T_A| differs: {len(a.entries)} vs {len(b.entries)} — a bijection must "
            "preserve every indexed ratio, so this alone refutes equivalence",
            0,
            False,
        )

    # Cheap necessary condition — *only* admissible for exact values.  With floats the
    # last-bit differences would reject genuinely equivalent images, i.e. the prune
    # itself would become the bug.
    if a.exact and b.exact:
        if a.value_multiset() != b.value_multiset():
            return None, "the multisets of rho values differ (necessary condition fails)", 0, False

    classes_a = a.signature_classes()
    classes_b = b.signature_classes()
    if sorted(len(v) for v in classes_a.values()) != sorted(len(v) for v in classes_b.values()):
        return (
            None,
            "the label signature classes have different size profiles; no bijection can "
            "preserve the indexed ratios",
            0,
            False,
        )

    # label -> its signature class key, so the enumeration can be organised per class
    key_of_a = {label: key for key, labels in classes_a.items() for label in labels}
    groups: list[tuple[list[int], list[int]]] = []
    for key, labels_a in classes_a.items():
        labels_b = classes_b.get(key)
        if labels_b is None:
            return None, f"signature class missing in B: {key[:2]}...", 0, False
        groups.append((sorted(labels_a), sorted(labels_b)))

    candidate_count = 1
    for labels_a, labels_b in groups:
        candidate_count *= _factorial(len(labels_a))
    if candidate_count > budget:
        return (
            None,
            f"{candidate_count} candidate bijections exceed the budget {budget} "
            "(the image has large symmetry classes)",
            0,
            True,
        )

    examined = 0
    for combination in _product_of_permutations(groups):
        examined += 1
        psi = dict(combination)
        if _verify_bijection(a, b, psi, tolerance):
            return psi, "a bijection preserving every indexed ratio was found", examined, False

    return (
        None,
        "no bijection preserves the indexed ratios (all candidates were verified)",
        examined,
        False,
    )


def codes_equivalent_exact(
    a: KozlovImageCode, b: KozlovImageCode, *, budget: int = DEFAULT_SEARCH_BUDGET
) -> KozlovCodeEquivalence:
    """Decide code equivalence for two exact codes."""
    if not (a.exact and b.exact):
        raise TypeError(
            "codes_equivalent_exact needs exact codes; use codes_equivalent_numeric for "
            "float codes (the tolerance is the difference between the two)"
        )
    psi, reason, examined, undecided = find_bijection(a, b, tolerance=None, budget=budget)
    return KozlovCodeEquivalence(
        equivalent=psi is not None,
        undecided=undecided,
        reason=reason,
        bijection=psi,
        candidates_examined=examined,
        n_entries_a=len(a.entries),
        n_entries_b=len(b.entries),
        tolerance=None,
    )


def codes_equivalent_numeric(
    a: KozlovImageCode,
    b: KozlovImageCode,
    *,
    tolerance: float,
    budget: int = DEFAULT_SEARCH_BUDGET,
) -> KozlovCodeEquivalence:
    """Decide code equivalence for two float codes with a stated tolerance."""
    if a.exact or b.exact:
        raise TypeError("codes_equivalent_numeric expects float codes")
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    psi, reason, examined, undecided = find_bijection(a, b, tolerance=tolerance, budget=budget)
    return KozlovCodeEquivalence(
        equivalent=psi is not None,
        undecided=undecided,
        reason=reason,
        bijection=psi,
        candidates_examined=examined,
        n_entries_a=len(a.entries),
        n_entries_b=len(b.entries),
        tolerance=tolerance,
    )


def images_equivalent_by_code_exact(x: Any, y: Any, **kwargs: Any) -> KozlovCodeEquivalence:
    """Build both exact codes and decide equivalence (route 1 end to end)."""
    return codes_equivalent_exact(
        build_kozlov_image_code_exact(x), build_kozlov_image_code_exact(y), **kwargs
    )


# --------------------------------------------------------------------------- #
# route 2: solve for an affine map (independent of the code)
# --------------------------------------------------------------------------- #
def _first_non_degenerate_triple(points: ExactPoints) -> tuple[int, int, int] | None:
    from .exact import double_area

    for triple in itertools.combinations(range(len(points)), 3):
        i, j, k = triple
        if double_area(points[i], points[j], points[k]) != 0:
            return triple
    return None


def affine_equivalent_exact(x: Any, y: Any) -> AffineEquivalenceResult:
    """Decide affine equivalence of two point **sets** by exact rational arithmetic.

    Method: an affine map is determined by the image of any non-degenerate triple, so
    fix one from ``x``, enumerate every ordered non-degenerate triple of ``y``, build
    the map, and test whether the images of all of ``x`` reproduce ``y`` *as a
    multiset*.  Working with :class:`~fractions.Fraction` throughout means the final
    comparison is an exact equality, with no tolerance to argue about.

    Scope, stated honestly: the method needs a non-degenerate triple, so an image whose
    points are all collinear yields ``undecided`` rather than a guess.  (Such images
    also have no code, so neither route covers them, and the specification says so.)
    """
    exact_x = _as_exact_points(x)
    exact_y = _as_exact_points(y)
    if len(exact_x) != len(exact_y):
        return AffineEquivalenceResult(
            False, False, f"cardinality differs: {len(exact_x)} vs {len(exact_y)}", None, None, None, None
        )

    control_x = _first_non_degenerate_triple(exact_x)
    control_y = _first_non_degenerate_triple(exact_y)
    if control_x is None or control_y is None:
        return AffineEquivalenceResult(
            True,
            True,
            "at least one image is fully collinear, so this geometric route (which needs a "
            "non-degenerate control triple) does not apply",
            None,
            None,
            None,
            None,
        )

    i, j, k = control_x
    base = (exact_x[i], exact_x[j], exact_x[k])
    from .exact import double_area

    target_multiset = sorted(exact_y)
    for ordered in itertools.permutations(range(len(exact_y)), 3):
        p, q, r = (exact_y[t] for t in ordered)
        if double_area(p, q, r) == 0:
            continue
        matrix, translation = _affine_from_three(base, (p, q, r))
        if matrix is None:
            continue
        images = sorted(_apply_exact(matrix, translation, point) for point in exact_x)
        if images == target_multiset:
            mapping = _match_exact(exact_x, exact_y, matrix, translation)
            return AffineEquivalenceResult(
                True,
                False,
                "an affine map reproducing Y as a set was found",
                matrix,
                translation,
                (frozenset({i, j, k}), frozenset(ordered)),
                mapping,
            )

    return AffineEquivalenceResult(
        False,
        False,
        f"no affine map reproduces Y (all {len(exact_y) * (len(exact_y) - 1) * (len(exact_y) - 2)} "
        "ordered triples of Y were tried)",
        None,
        None,
        None,
        None,
    )


def _affine_from_three(
    source: Sequence[Sequence[Fraction]], target: Sequence[Sequence[Fraction]]
) -> tuple[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] | None, tuple[Fraction, Fraction] | None]:
    """Solve ``M u = du`` and ``M v = dv`` for the 2x2 matrix ``M``, exactly.

    With ``u = s_1 - s_0``, ``v = s_2 - s_0`` (columns of ``U``) and ``du``, ``dv`` the
    corresponding target differences (columns of ``D``), the map is ``M = D U^{-1}``:

        M = D U^{-1} = (1/det U) [[du0, dv0], [du1, dv1]] [[v1, -v0], [-u1, u0]]

    giving the four entries written out below.  Getting this transpose/companion
    pattern wrong is easy and, historically, silent — so the function **verifies its own
    output** by applying it to ``u`` and ``v`` before returning.  That check is O(1) and
    turns a whole class of quiet sign/transpose errors into a loud assertion.
    """
    u = (source[1][0] - source[0][0], source[1][1] - source[0][1])
    v = (source[2][0] - source[0][0], source[2][1] - source[0][1])
    du = (target[1][0] - target[0][0], target[1][1] - target[0][1])
    dv = (target[2][0] - target[0][0], target[2][1] - target[0][1])

    det = u[0] * v[1] - u[1] * v[0]
    if det == 0:
        return None, None

    m00 = Fraction(du[0] * v[1] - dv[0] * u[1], det)
    m01 = Fraction(dv[0] * u[0] - du[0] * v[0], det)
    m10 = Fraction(du[1] * v[1] - dv[1] * u[1], det)
    m11 = Fraction(dv[1] * u[0] - du[1] * v[0], det)
    matrix = ((m00, m01), (m10, m11))

    # self-check: the solved matrix must reproduce both spanning vectors exactly
    assert (m00 * u[0] + m01 * u[1], m10 * u[0] + m11 * u[1]) == du, (  # noqa: S101
        "affine solve failed to reproduce the first spanning vector"
    )
    assert (m00 * v[0] + m01 * v[1], m10 * v[0] + m11 * v[1]) == dv, (  # noqa: S101
        "affine solve failed to reproduce the second spanning vector"
    )

    translation = (
        target[0][0] - (m00 * source[0][0] + m01 * source[0][1]),
        target[0][1] - (m10 * source[0][0] + m11 * source[0][1]),
    )
    return matrix, translation


def _apply_exact(
    matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    translation: tuple[Fraction, Fraction],
    point: Sequence[Fraction],
) -> tuple[Fraction, Fraction]:
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + translation[0],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + translation[1],
    )


def _match_exact(
    exact_x: ExactPoints,
    exact_y: ExactPoints,
    matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    translation: tuple[Fraction, Fraction],
) -> dict[int, int] | None:
    """Recover the induced bijection (``None`` when it is ambiguous)."""
    remaining = list(range(len(exact_y)))
    mapping: dict[int, int] = {}
    for index, point in enumerate(exact_x):
        image = _apply_exact(matrix, translation, point)
        hits = [t for t in remaining if exact_y[t] == image]
        if not hits:
            return None
        if len(hits) > 1:
            return None  # duplicate coordinates: the correspondence is not unique
        mapping[index] = hits[0]
        remaining.remove(hits[0])
    return mapping


# --------------------------------------------------------------------------- #
# cross-check
# --------------------------------------------------------------------------- #
def cross_check_exact(x: Any, y: Any, **kwargs: Any) -> dict[str, Any]:
    """Run **both** routes and report whether they agree.

    The agreement is not a tautology: the two routes share no code path beyond the
    input coordinates, so a disagreement means one of them is wrong.  Any result of
    this function carries ``routes_agree``, and theorem 1 predicts it is ``True`` on
    the general-position domain.
    """
    code_result = images_equivalent_by_code_exact(x, y, **kwargs)
    geometry_result = affine_equivalent_exact(x, y)

    agree: bool | None
    if code_result.undecided or geometry_result.undecided:
        agree = None
    else:
        agree = code_result.equivalent == geometry_result.equivalent

    return {
        "code_route": code_result.as_dict(),
        "geometry_route": geometry_result.as_dict(),
        "routes_agree": agree,
        "note": (
            "the code route implements the definition; the geometric route knows nothing "
            "about codes. Theorem 1 predicts agreement in general position."
        ),
    }


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _factorial(n: int) -> int:
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result


def _product_of_permutations(
    groups: Iterable[tuple[list[int], list[int]]]
) -> Iterator[list[tuple[int, int]]]:
    """Yield every assignment of ``a``-labels to ``b``-labels, class by class."""
    group_list = list(groups)

    def recurse(position: int, accumulated: list[tuple[int, int]]) -> Iterator[list[tuple[int, int]]]:
        if position == len(group_list):
            yield accumulated
            return
        labels_a, labels_b = group_list[position]
        for permutation in itertools.permutations(labels_b):
            yield from recurse(position + 1, accumulated + list(zip(labels_a, permutation)))

    yield from recurse(0, [])
