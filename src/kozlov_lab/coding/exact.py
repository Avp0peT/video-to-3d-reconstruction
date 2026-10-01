"""Exact implementation of Kozlov's image code — the mathematical oracle.

This module computes :math:`K(A)=\\langle M_A, T_A\\rangle` with **no floating point
anywhere**: coordinates are :class:`int` or :class:`~fractions.Fraction`, areas are
kept as doubled integers, and every :math:`\\rho` is a :class:`~fractions.Fraction`.

Why exactness is the whole point
--------------------------------
The companion module :mod:`kozlov_lab.coding.numeric` has to decide "is this triple
degenerate?" with a tolerance, because a mathematically collinear triple has a
computed determinant of ``~1e-16`` rather than ``0``.  That is a *numerical strategy*,
and a strategy can be wrong.  Without an exact implementation there is nothing to
check the strategy against.

Here, ``D == 0`` is an exact statement and :math:`S_1/S_2 = D_1/D_2` is an exact
identity, so this module serves as the reference against which the numeric layer is
validated — and as the arbiter for the hand-worked examples demanded by gate G3.6.

The doubled-area trick
----------------------
``S = D/2`` where

.. math::

    D(a,b,c) = \\left|(x_b-x_a)(y_c-y_a) - (y_b-y_a)(x_c-x_a)\\right|

so ratios of *areas* equal ratios of *doubled areas* and the ``1/2`` never appears:

.. math::

    \\frac{S_1}{S_2} = \\frac{D_1/2}{D_2/2} = \\frac{D_1}{D_2}.

With integer coordinates, ``D`` is an integer and every :math:`\\rho` is
``Fraction(D_num, D_den)`` — exactly the construction the specification prescribes.
"""

from __future__ import annotations

from fractions import Fraction
import itertools
from typing import Any, Iterable, Sequence

from .model import (
    DOMAIN_CODE_UNDEFINED,
    KozlovImageCode,
    RatioEntry,
    Triple,
)
from .validation import (
    ExactPoints,
    _exact_double_area,
    _as_exact_points,
    validate_kozlov_image,
)

__all__ = [
    "double_area",
    "triple_double_areas",
    "exact_ratio",
    "build_kozlov_image_code_exact",
    "max_entries_for",
]

#: Safety cap on the number of ratio entries.  The construction is
#: ``O(C(n,3)^2)``: 220 triples at ``n = 12`` already give 48 400 entries, and
#: ``n = 20`` would give 1.3e6.  Exceeding the cap raises rather than truncating —
#: a truncated ``T_A`` is **not** the code, and silently returning one would
#: produce a "complete invariant" that is neither complete nor an invariant.
DEFAULT_MAX_ENTRIES = 2_000_000


def max_entries_for(n_points: int) -> int:
    """Exact number of entries ``|T_A|`` for an image of ``n`` points with no degeneracies.

    Every ordered pair of *distinct* triples contributes one ratio, and the numerator
    may be degenerate while the denominator may not, so the count is

        C(n,3) * |admissible denominators| - |admissible denominators|

    in the fully general case, and ``C(n,3)*(C(n,3)-1)`` when nothing is degenerate.
    """
    triples = n_points * (n_points - 1) * (n_points - 2) // 6
    return triples * (triples - 1)


def double_area(a: Sequence[Any], b: Sequence[Any], c: Sequence[Any]) -> Any:
    """Twice the area of ``abc``, as an exact integer or ``Fraction``."""
    return _exact_double_area(a, b, c)


def triple_double_areas(points: Any) -> dict[Triple, Any]:
    """``D`` for every unordered triple, degenerate ones included (``D == 0``)."""
    exact = _as_exact_points(points)
    n = len(exact)
    return {
        frozenset(triple): _exact_double_area(exact[triple[0]], exact[triple[1]], exact[triple[2]])
        for triple in itertools.combinations(range(n), 3)
    }


def exact_ratio(d_numerator: Any, d_denominator: Any) -> Fraction:
    """``rho = D_num / D_den`` as an exact :class:`~fractions.Fraction`.

    Raises
    ------
    ZeroDivisionError
        When the denominator triple is degenerate.  The source states the ratio is
        undefined there, and an exception is the honest encoding of "undefined" — a
        sentinel like ``nan`` or ``0`` would be silently usable in downstream
        arithmetic.
    """
    if d_denominator == 0:
        raise ZeroDivisionError(
            "the ratio S_ksp/S_ksp is undefined when S_ksp = 0: the source excludes "
            "degenerate denominator triples by definition"
        )
    return Fraction(d_numerator, d_denominator)


def build_kozlov_image_code_exact(
    points: Any,
    *,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    require_theorem_domain: bool = False,
) -> KozlovImageCode:
    """Build :math:`K(A)=\\langle M_A,T_A\\rangle` exactly.

    Parameters
    ----------
    points:
        Integer or :class:`~fractions.Fraction` coordinates.  Floats are refused by
        the validation layer (use :mod:`kozlov_lab.coding.numeric` for those).
    max_entries:
        Refuse to build a code larger than this; raising beats truncating (see
        :data:`DEFAULT_MAX_ENTRIES`).
    require_theorem_domain:
        When ``True``, a two-parallel-lines image raises instead of being accepted
        with a recorded note.  Use it whenever the code is about to be used as a
        *complete recognition invariant*, because that is exactly what theorem 1
        guarantees and what the two-parallel-lines case does not.

    Returns
    -------
    KozlovImageCode
        With ``exact=True``, ``Fraction`` values, and the domain verdict attached.
    """
    validation = validate_kozlov_image(points, require_theorem_domain=require_theorem_domain)
    exact = _as_exact_points(points)
    labels = frozenset(range(len(exact)))

    areas = triple_double_areas(exact)
    admissible = sorted((t for t, d in areas.items() if d != 0), key=lambda t: sorted(t))
    degenerate = frozenset(t for t, d in areas.items() if d == 0)

    n_pairs = len(areas) * len(admissible) - len(admissible)
    if n_pairs > max_entries:
        raise ValueError(
            f"the code would contain {n_pairs} ratio entries, above the cap {max_entries}. "
            "Refusing to truncate: a truncated T_A is not the code of the image. "
            "Raise `max_entries` deliberately, or restrict the image size."
        )

    entries: set[RatioEntry] = set()
    for numerator, d_num in areas.items():
        for denominator in admissible:
            if numerator == denominator:
                continue  # the source requires two *distinct* triples
            entries.add(
                RatioEntry(
                    numerator=numerator,
                    denominator=denominator,
                    value=exact_ratio(d_num, areas[denominator]),
                )
            )

    warnings = list(validation.notes)
    code = KozlovImageCode(
        labels=labels,
        entries=frozenset(entries),
        exact=True,
        n_points=len(exact),
        n_triples=len(areas),
        n_admissible_denominators=len(admissible),
        degenerate_triples=degenerate,
        domain=validation.domain,
        warnings=tuple(warnings),
    )

    if not code.is_complete:
        raise AssertionError(  # pragma: no cover - internal consistency
            f"entry count {len(code.entries)} != expected {n_pairs}: the enumeration "
            "missed or duplicated a pair"
        )
    return code
