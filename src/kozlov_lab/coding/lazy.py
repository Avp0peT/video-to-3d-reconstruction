"""Lazy representation of the formal code.

Why this exists
---------------
The formal object is

.. math::

    K(A) = \\langle M_A,\\ T_A \\rangle,
    \\qquad T_A = \\{\\, \\rho_{mnu,ksp} \\,\\},

with ``q`` admissible (non-degenerate) denominators and ``N = C(n,3)`` triples in total.
Every denominator pairs with every *different* numerator, so

.. math::

    |T_A| = q\\,(N-1) \\;\\le\\; \\binom{n}{3}\\left(\\binom{n}{3}-1\\right)
          = O(n^6).

The exponent is the problem: ``n = 16`` already allows 313 040 entries and ``n = 20``
allows 1.3 million, each an object holding two frozensets and a rational.  Materialising
``T_A`` is therefore a *choice with a cost*, not a property of the definition.

:class:`IndexedRatioView` keeps exactly what the definition needs and nothing more — the
label set ``M_A``, the triple list, the doubled areas ``D``, and the mask of admissible
denominators — and forms :math:`\\rho = D_a/D_b` on demand.  The mathematical object is
identical (``tests/test_kozlov_lazy.py`` checks the two representations entry by entry)
while memory becomes ``O(n^3)`` instead of ``O(n^6)``.

What laziness does *not* buy
----------------------------
Equivalence still has to look at the relations, so an equivalence query materialises what
it needs.  The lazy view is a memory optimisation and a cost model, **not** a way to avoid
the ``O(n^6)`` work when that work is genuinely required.  Claiming otherwise would be the
kind of optimism this project avoids elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterator, Sequence

from .exact import DEFAULT_MAX_ENTRIES, triple_double_areas
from .model import ArithmeticMode, KozlovImageCode, RatioEntry, Triple
from .validation import ExactPoints, _as_exact_points

__all__ = [
    "IndexedRatioView",
    "KozlovImageCodeLazy",
    "build_kozlov_image_code_lazy_exact",
    "build_kozlov_image_code_lazy_numeric",
    "entry_count_formula",
]


def entry_count_formula(n_triples: int, n_admissible: int) -> int:
    """``|T_A| = q(N-1)``: each admissible denominator pairs with every other triple.

    A degenerate triple may be a *numerator* (giving ``rho = 0``) but never a denominator,
    so the count is ``q`` times the ``N-1`` triples different from that denominator.
    """
    return n_admissible * (n_triples - 1)


class IndexedRatioView:
    """Read-only view answering "what is ``rho`` for these two triples?".

    Not a dataclass on purpose: the label→position index must be built **once**.  A
    ``@property`` that rebuilt it per call would make :meth:`iter_entries` cost
    ``O(N^2)`` dict constructions — an ``O(n^6)`` algorithm with an ``O(n^3)`` inner
    constant, which is exactly the kind of accidental complexity the lazy representation is
    meant to remove.
    """

    __slots__ = (
        "labels",
        "triples",
        "double_areas",
        "admissible",
        "mode",
        "degenerate",
        "n_entries",
        "_index",
    )

    def __init__(
        self,
        *,
        labels: frozenset[int],
        triples: Sequence[Triple],
        double_areas: Sequence[Any],
        admissible: Sequence[bool],
        mode: ArithmeticMode,
        degenerate: Sequence[Triple] = (),
        n_entries: int = 0,
    ) -> None:
        if not (len(triples) == len(double_areas) == len(admissible)):
            raise ValueError("triples, double_areas and admissible must have equal length")
        self.labels = labels
        self.triples: tuple[Triple, ...] = tuple(triples)
        self.double_areas: tuple[Any, ...] = tuple(double_areas)
        self.admissible: tuple[bool, ...] = tuple(admissible)
        self.mode = mode
        self.degenerate: tuple[Triple, ...] = tuple(degenerate)
        self.n_entries = n_entries or entry_count_formula(len(self.triples), sum(self.admissible))
        self._index: dict[Triple, int] = {
            triple: position for position, triple in enumerate(self.triples)
        }

    # -- counts ----------------------------------------------------------- #
    @property
    def n_points(self) -> int:
        return len(self.labels)

    @property
    def n_triples(self) -> int:
        return len(self.triples)

    @property
    def n_admissible_denominators(self) -> int:
        return sum(self.admissible)

    @property
    def admissible_triples(self) -> tuple[Triple, ...]:
        return tuple(t for t, ok in zip(self.triples, self.admissible) if ok)

    def __len__(self) -> int:
        """``|T_A|`` from the formula — never by building the entries."""
        return self.n_entries

    # -- the definition --------------------------------------------------- #
    def double_area(self, triple: Triple) -> Any:
        return self.double_areas[self._index[triple]]

    def is_admissible(self, triple: Triple) -> bool:
        return self.admissible[self._index[triple]]

    def ratio(self, numerator: Triple, denominator: Triple) -> Fraction | float:
        """``rho_{numerator,denominator} = D_num / D_den``.

        Raises
        ------
        ValueError
            If a triple is unknown, if the two triples are equal (the definition requires
            *distinct* triples — ``rho = 1`` is not part of ``T_A``), or if the denominator
            is degenerate (the ratio is undefined there).  Refusals rather than sentinel
            values, so an undefined ratio cannot flow into arithmetic unnoticed.
        """
        index = self._index
        if numerator not in index or denominator not in index:
            raise ValueError(f"unknown triple: {sorted(numerator)} / {sorted(denominator)}")
        if numerator == denominator:
            raise ValueError(
                "the two triples of a ratio must be distinct; rho = S/S = 1 is not part of T_A"
            )
        position = index[denominator]
        if not self.admissible[position]:
            raise ValueError(
                f"the denominator triple {sorted(denominator)} is degenerate (D = 0): the ratio "
                "is undefined by definition"
            )
        if self.mode is ArithmeticMode.EXACT:
            return Fraction(self.double_areas[index[numerator]], self.double_areas[position])
        return float(self.double_areas[index[numerator]]) / float(self.double_areas[position])

    # -- iteration -------------------------------------------------------- #
    def iter_entries(self, *, limit: int | None = None) -> Iterator[RatioEntry]:
        """Yield the entries of ``T_A``.

        Turning the view back into the ``O(n^6)`` object; deliberately explicit, so callers
        who need only a few ratios use :meth:`ratio` instead.
        """
        produced = 0
        denominators = self.admissible_triples
        for denominator in denominators:
            denominator_position = self._index[denominator]
            denominator_value = self.double_areas[denominator_position]
            for numerator in self.triples:
                if numerator == denominator:
                    continue
                if limit is not None and produced >= limit:
                    return
                numerator_value = self.double_areas[self._index[numerator]]
                value: Fraction | float
                if self.mode is ArithmeticMode.EXACT:
                    value = Fraction(numerator_value, denominator_value)
                else:
                    value = float(numerator_value) / float(denominator_value)
                yield RatioEntry(numerator=numerator, denominator=denominator, value=value)
                produced += 1

    def materialize(self) -> frozenset[RatioEntry]:
        return frozenset(self.iter_entries())

    def to_code(self, *, domain: str = "general_position", n_points: int | None = None) -> KozlovImageCode:
        return KozlovImageCode(
            labels=self.labels,
            entries=self.materialize(),
            exact=self.mode is ArithmeticMode.EXACT,
            n_points=n_points if n_points is not None else self.n_points,
            n_triples=self.n_triples,
            n_admissible_denominators=self.n_admissible_denominators,
            degenerate_triples=frozenset(self.degenerate),
            domain=domain,  # type: ignore[arg-type]
            warnings=("MATERIALIZED_FROM_LAZY_VIEW",),
        )

    def memory_footprint(self) -> dict[str, int]:
        """What the view holds — for the scaling experiment, not for the definition."""
        return {
            "n_labels": len(self.labels),
            "n_triples": self.n_triples,
            "n_admissible": self.n_admissible_denominators,
            "n_entries_if_materialized": self.n_entries,
            "entries_per_triple": self.n_triples - 1,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"IndexedRatioView(n={self.n_points}, triples={self.n_triples}, "
            f"admissible={self.n_admissible_denominators}, |T_A|={self.n_entries}, {self.mode})"
        )


@dataclass(frozen=True)
class KozlovImageCodeLazy:
    """``<M_A, T_A>`` held in ``O(n^3)`` instead of ``O(n^6)``.

    Carries the validation record too, so the three domain questions are answered without
    materialising anything.
    """

    view: IndexedRatioView
    validation: Any

    @property
    def labels(self) -> frozenset[int]:
        return self.view.labels

    @property
    def mode(self) -> ArithmeticMode:
        return self.view.mode

    @property
    def exact(self) -> bool:
        return self.view.mode is ArithmeticMode.EXACT

    @property
    def domain(self) -> str:
        return self.validation.domain

    @property
    def theorem_applies(self) -> bool:
        return bool(self.validation.theorem_applicable)

    @property
    def numerically_well_conditioned(self) -> bool:
        return bool(self.validation.numerically_well_conditioned)

    def __len__(self) -> int:
        return self.view.n_entries

    def ratio(self, numerator: Triple, denominator: Triple) -> Fraction | float:
        return self.view.ratio(numerator, denominator)

    def materialize(self) -> KozlovImageCode:
        return self.view.to_code(domain=self.validation.domain, n_points=self.validation.n_points)

    def as_dict(self) -> dict[str, Any]:
        return {
            "representation": "lazy",
            "mode": str(self.view.mode),
            "n_points": self.validation.n_points,
            "n_triples": self.view.n_triples,
            "n_admissible_denominators": self.view.n_admissible_denominators,
            "n_entries": len(self.view),
            "domain": self.validation.domain,
            "theorem_applicable": self.validation.theorem_applicable,
            "numerically_well_conditioned": self.validation.numerically_well_conditioned,
            "degenerate_triples": [sorted(t) for t in self.view.degenerate],
            "memory": self.view.memory_footprint(),
        }


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_kozlov_image_code_lazy_exact(
    points: Any,
    *,
    require_theorem_domain: bool = False,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> KozlovImageCodeLazy:
    """Exact lazy code.

    Validation is delegated to the strict adapter — the same one the materialised builder
    uses — so the two representations cannot disagree about the domain.
    """
    from .validation import validate_kozlov_image

    validation = validate_kozlov_image(points, require_theorem_domain=require_theorem_domain)
    exact_points: ExactPoints = _as_exact_points(points)
    areas_map = triple_double_areas(exact_points)
    triples = tuple(sorted(areas_map, key=sorted))
    areas = tuple(areas_map[triple] for triple in triples)
    admissible = tuple(value != 0 for value in areas)
    degenerate = tuple(triple for triple, ok in zip(triples, admissible) if not ok)

    n_entries = entry_count_formula(len(triples), sum(admissible))
    if n_entries > max_entries:
        raise ValueError(
            f"the code would contain {n_entries} ratio entries, above the cap {max_entries}. "
            "Refusing to truncate: a truncated T_A is not the code of the image. (The lazy "
            "view holds no entries, so raising the cap costs memory only when materialised.)"
        )
    view = IndexedRatioView(
        labels=frozenset(range(len(exact_points))),
        triples=triples,
        double_areas=areas,
        admissible=admissible,
        mode=ArithmeticMode.EXACT,
        degenerate=degenerate,
        n_entries=n_entries,
    )
    return KozlovImageCodeLazy(view=view, validation=validation)


def build_kozlov_image_code_lazy_numeric(
    points: Any,
    *,
    flatness_tol: float | None = None,
    require_theorem_domain: bool = False,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> KozlovImageCodeLazy:
    """Numeric lazy code.

    The degeneracy decision is delegated to the numeric validator, i.e. it is the same
    *numerical strategy* the materialised numeric builder uses — one place, not two.
    """
    import numpy as np

    from ..config import COLLINEAR_TOL
    from ..geometry.affine_frame import triangle_flatness
    from ..geometry.points import as_point_matrix
    from .numeric import numeric_double_areas
    from .validation import validate_kozlov_image_numeric

    tol = COLLINEAR_TOL if flatness_tol is None else flatness_tol
    matrix = np.asarray(as_point_matrix(points, name="points", min_points=1), dtype=float)
    validation = validate_kozlov_image_numeric(
        matrix, tol=tol, require_theorem_domain=require_theorem_domain
    )

    areas_map = numeric_double_areas(matrix)
    triples = tuple(sorted(areas_map, key=sorted))
    areas = tuple(areas_map[triple] for triple in triples)
    admissible = []
    for triple in triples:
        i, j, k = sorted(triple)
        admissible.append(triangle_flatness(matrix[i], matrix[j], matrix[k]) > tol)
    if not any(admissible):
        raise ValueError(f"no admissible denominator: every triple is degenerate at tol = {tol:g}")

    degenerate = tuple(triple for triple, ok in zip(triples, admissible) if not ok)
    n_entries = entry_count_formula(len(triples), sum(admissible))
    if n_entries > max_entries:
        raise ValueError(f"the code would contain {n_entries} ratio entries, above {max_entries}")

    view = IndexedRatioView(
        labels=frozenset(range(matrix.shape[0])),
        triples=triples,
        double_areas=areas,
        admissible=tuple(admissible),
        mode=ArithmeticMode.NUMERIC,
        degenerate=degenerate,
        n_entries=n_entries,
    )
    return KozlovImageCodeLazy(view=view, validation=validation)
