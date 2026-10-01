"""Strict validation adapter for Kozlov's formal image model (Phase 3).

Why a separate adapter
----------------------
Phase 1 deliberately accepts a **multiset** of points (repeated points are legal
input there, and the README's stability matrix demands that).  Kozlov's formal
theory is about a **finite set**: the object ``A`` cannot contain "the same
coordinate twice", and the numbering requirement is «разные точки – разные номера»
(different points get different numbers).

Those two contracts are incompatible, so the transition is made explicit here
instead of being papered over:

* this module **refuses** an image with repeated points (it does not silently
  deduplicate — doing so would quietly change what the user supplied);
* the UI may offer *deduplication as a named preprocessing step*, but then it must
  show the point count before and after (see ``app/pages/05_Kozlov_Official_Code.py``).

The three-valued domain — the part that is easy to conflate
-----------------------------------------------------------
"Non-collinear" is **not** the full hypothesis of the theorem.  The source first
excludes images whose points all lie on one line, because then the code cannot be
defined at all; but theorem 1 is stated for a "плоское изображение", whose defining
condition is stronger: the points must not lie on *one line or on two parallel
lines*.  Three distinct states therefore exist, and the specification requires them
to be kept apart:

===========================  ============================  ====================
verdict                      code computable?              theorem 1's iff in force?
===========================  ============================  ====================
``code_undefined``           no (no admissible S ≠ 0)      no
``two_parallel_lines``       **yes**                       **no**
``general_position``         yes                           **yes**
===========================  ============================  ====================

Conflating the middle row with the last is the specific error this module exists to
prevent: a code computed on a two-parallel-lines image is numerically fine and
*semantically weaker*, and the UI must say so rather than let "the code was
computed" look like "the theorem guarantees complete recognition".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
import itertools
import math
from typing import Any, Iterable, Sequence

import numpy as np

from ..geometry.points import as_point_matrix
from .model import (
    DOMAIN_CODE_UNDEFINED,
    DOMAIN_GENERAL_POSITION,
    DOMAIN_TWO_PARALLEL_LINES,
    DomainVerdict,
    Triple,
)

__all__ = [
    "KozlovImageError",
    "NonExactCoordinateError",
    "DuplicatePointError",
    "TooFewPointsError",
    "CollinearImageError",
    "TheoremDomainError",
    "KozlovImageValidation",
    "validate_kozlov_image",
    "validate_kozlov_image_numeric",
    "classify_domain_exact",
    "classify_domain_numeric",
    "noise_floor_for",
    "ExactPoints",
]


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class KozlovImageError(ValueError):
    """Base class for every refusal of the formal (Kozlov) image model."""


class NonExactCoordinateError(KozlovImageError, TypeError):
    """Raised when the exact layer cannot represent the input exactly.

    **Integer-valued** floats are fine (the conversion is lossless).  What is refused is
    a *non-integer* float: ``Fraction(0.1)`` is the binary rational
    ``3602879701896397/36028797018963968``, not one tenth, so converting automatically
    would make the "exact" layer silently inexact and destroy its value as an oracle.
    Use :class:`~fractions.Fraction` for exact rationals, or
    :mod:`kozlov_lab.coding.numeric` for floating point.
    """


class DuplicatePointError(KozlovImageError):
    """Raised when two points of the image share the same coordinate.

    Not silently deduplicated: the caller must decide, and must be able to show the
    user that the point count changed.
    """

    def __init__(self, duplicates: Sequence[tuple[Any, int]]) -> None:
        self.duplicates = tuple(duplicates)
        detail = ", ".join(f"{coord} x{count}" for coord, count in self.duplicates[:5])
        super().__init__(
            "the formal image is a finite *set*: every point must have a distinct "
            f"coordinate, but {len(self.duplicates)} coordinate(s) repeat ({detail}). "
            "Deduplicate explicitly (and report the change) instead of relying on a "
            "silent merge."
        )


class TooFewPointsError(KozlovImageError):
    """Raised when the image has fewer than three points.

    Kept distinct from :class:`CollinearImageError` because the two are different
    facts and conflating them produces a misleading message ("all 2 points are
    collinear" is not what is wrong).
    """

    def __init__(self, n_points: int) -> None:
        self.n_points = n_points
        super().__init__(
            f"the code needs at least three points (a triple to form a ratio); got {n_points}"
        )


class CollinearImageError(KozlovImageError):
    """Raised when every triple is degenerate, so no denominator is defined."""

    def __init__(self, n_points: int) -> None:
        self.n_points = n_points
        super().__init__(
            f"all {n_points} points are collinear, so no triple has S != 0 and every "
            "denominator is undefined: the code <M_A, T_A> does not exist for this image"
        )


class TheoremDomainError(KozlovImageError):
    """Raised when the caller demands the theorem's hypothesis and it does not hold.

    Only raised when ``require_theorem_domain=True``.  A ``two_parallel_lines`` image
    is computable, so refusing it unconditionally would be wrong; refusing it when the
    caller intends to use theorem 1's iff statement is right.
    """

    def __init__(self, domain: DomainVerdict, evidence: dict[str, Any]) -> None:
        self.domain = domain
        self.evidence = evidence
        super().__init__(
            f"theorem 1 is stated for images whose points do not lie on one line or on "
            f"two parallel lines; this image is classified '{domain}' "
            f"(evidence: {evidence}). The code can still be computed, but the "
            "iff-guarantee does not apply."
        )


# --------------------------------------------------------------------------- #
# exact point handling
# --------------------------------------------------------------------------- #
#: A tuple of exact coordinate pairs (``Fraction`` or ``int``).
ExactPoints = tuple[tuple[Fraction, Fraction], ...]


def _raw_pairs(points: Any) -> list[tuple[Any, Any]]:
    """Materialise coordinate pairs **without** coercion to ``float64``.

    The library's shared :func:`~kozlov_lab.geometry.points.as_point_matrix` converts
    everything to ``float64``, which is right for the geometry layer and fatal here: a
    :class:`~fractions.Fraction` would be flattened to a float before this layer could
    see it.  Hence a small object-preserving companion.
    """
    from collections.abc import Mapping as _Mapping

    from ..geometry.points import Point2D, PointSet2D

    if isinstance(points, PointSet2D):
        # a PointSet2D already stored float64; treat it as float input
        return [(float(a), float(b)) for a, b in points.as_array()]
    if isinstance(points, _Mapping):
        if "points" not in points:
            raise ValueError("mapping input needs a 'points' key")
        return _raw_pairs(points["points"])
    if isinstance(points, np.ndarray):
        arr = np.asarray(points)
        if arr.ndim == 1 and arr.shape == (2,):
            arr = arr.reshape(1, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"expected shape (n, 2), got {arr.shape}")
        return [(arr[i, 0], arr[i, 1]) for i in range(arr.shape[0])]

    try:
        items = list(points)
    except TypeError as exc:
        raise TypeError(f"cannot interpret {type(points).__name__} as points") from exc
    if not items:
        raise ValueError("an image must be non-empty")
    if isinstance(items[0], Point2D):
        return [(item.x, item.y) for item in items]
    pairs: list[tuple[Any, Any]] = []
    for row in items:
        try:
            sequence = list(row)
        except TypeError as exc:
            raise TypeError(f"cannot interpret {row!r} as a point") from exc
        if len(sequence) != 2:
            raise ValueError(f"each point needs exactly 2 coordinates, got {sequence!r}")
        pairs.append((sequence[0], sequence[1]))
    return pairs


def _exact_component(value: Any) -> Fraction:
    """One coordinate as an exact :class:`~fractions.Fraction`."""
    if isinstance(value, bool):
        raise TypeError("boolean coordinates are almost certainly a mistake")
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)  # a Decimal is an exact decimal, unlike a float
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"coordinates must be finite, got {number}")
        if number.is_integer():
            return Fraction(int(number))
        raise NonExactCoordinateError(
            f"the coordinate {number!r} is not integer-valued. The exact layer cannot "
            "represent it without inventing precision: Fraction(0.1) is the binary "
            "rational 3602879701896397/36028797018963968, not one tenth. Pass "
            "fractions.Fraction(1, 10) to mean exactly one tenth, or use "
            "kozlov_lab.coding.build_kozlov_image_code_numeric for floating point."
        )
    if isinstance(value, str):
        raise TypeError(f"coordinates must be numbers, got the string {value!r}")
    raise TypeError(f"unsupported coordinate type {type(value).__name__}")


def _as_exact_points(points: Any) -> ExactPoints:
    """Coerce to exact rational coordinates.

    Accepted:

    * integer coordinates (``int``, numpy integer arrays);
    * **integer-valued** floats — ``np.array([[0, 0], [2, 0]])`` is ``float64`` yet every
      entry is exactly an integer, so the conversion is lossless.  Refusing this would be
      obstructive rather than rigorous, so the integrality is *verified* entry by entry
      before being trusted;
    * :class:`~fractions.Fraction` and :class:`~decimal.Decimal` coordinates, for
      genuinely non-integer exact input such as ``Fraction(1, 10)``.

    **Non-integer floats are refused** — see :class:`NonExactCoordinateError`.
    """
    pairs = _raw_pairs(points)
    return tuple((_exact_component(x), _exact_component(y)) for x, y in pairs)


def _exact_double_area(a: Sequence[Fraction], b: Sequence[Fraction], c: Sequence[Fraction]) -> int | Fraction:
    r"""``D(a,b,c) = |(x_b-x_a)(y_c-y_a) - (y_b-y_a)(x_c-x_a)|``.

    Twice the triangle area.  Working with ``D`` avoids the ``1/2`` entirely, and
    since :math:`S = D/2` we have :math:`S_1/S_2 = D_1/D_2` — so the exact ratios can
    be formed as :class:`~fractions.Fraction` values of two **integers** when the
    input coordinates are integers.  That is what makes this layer an oracle: no
    division happens until the ratio itself, and ``D == 0`` is an exact test for
    degeneracy.
    """
    return abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    )


def _exact_offsets(points: ExactPoints, direction: tuple[Fraction, Fraction]) -> list[Fraction]:
    """Offsets of every point from the line through ``points[0]`` along ``direction``.

    Returns the **scaled** signed distance ``(p - o) x direction``, i.e. the signed
    parallelogram area rather than the Euclidean distance.  Scaling by the constant
    ``|direction|`` is irrelevant here because the only thing the caller asks is *how
    many distinct offsets* exist — and equality of the scaled values is equivalent to
    equality of the true distances.

    Keeping the quantity rational means the subsequent count is an **exact** statement,
    not a tolerance judgement: no square root is ever taken.
    """
    dx, dy = direction
    origin = points[0]
    return [(p[0] - origin[0]) * dy - (p[1] - origin[1]) * dx for p in points]


def classify_domain_exact(points: Any) -> tuple[DomainVerdict, dict[str, Any]]:
    """Exact three-valued domain classification (rational arithmetic only).

    Returns ``(verdict, evidence)`` where ``evidence`` carries the witness that
    decided the verdict, so the UI can show *why* rather than asserting a label.
    """
    exact = _as_exact_points(points)
    n = len(exact)
    if n < 3:
        return DOMAIN_CODE_UNDEFINED, {"reason": "fewer than three points", "n_points": n}

    areas: dict[Triple, Any] = {}
    for triple in itertools.combinations(range(n), 3):
        areas[frozenset(triple)] = _exact_double_area(
            exact[triple[0]], exact[triple[1]], exact[triple[2]]
        )
    non_degenerate = [t for t, d in areas.items() if d != 0]
    if not non_degenerate:
        return DOMAIN_CODE_UNDEFINED, {
            "reason": "every triple is degenerate (all points collinear)",
            "n_points": n,
            "n_degenerate_triples": len(areas),
        }

    # "no three collinear" was already ruled out above; now test the stronger
    # hypothesis: do all points lie on at most two parallel lines?
    best: dict[str, Any] | None = None
    for triple in non_degenerate:
        i, j, k = sorted(triple)
        dx, dy = exact[j][0] - exact[i][0], exact[j][1] - exact[i][1]
        offsets = _exact_offsets(exact, (dx, dy))
        distinct = sorted(set(offsets))
        if len(distinct) <= 2:
            best = {
                "direction_from_triple": [i, j, k],
                "n_distinct_offsets": len(distinct),
                "witness_lines": len(distinct),
            }
            break

    if best is not None:
        if best["n_distinct_offsets"] == 1:
            # all on one line — but that would have been caught above; keep the branch
            # explicit so the two cases can never be silently merged
            return DOMAIN_CODE_UNDEFINED, {
                "reason": "all points lie on a single line",
                **best,
            }
        return DOMAIN_TWO_PARALLEL_LINES, {
            "reason": "all points lie on two parallel lines",
            "n_points": n,
            **best,
        }

    return DOMAIN_GENERAL_POSITION, {
        "reason": "no direction puts all points on at most two parallel lines",
        "n_points": n,
        "n_triples_examined": len(non_degenerate),
    }


def classify_domain_numeric(
    points: Any, *, tol: float = 1e-9
) -> tuple[DomainVerdict, dict[str, Any]]:
    """Float classification with a **scale-free** tolerance.

    Offsets are divided by the point-set diameter before clustering, so the same
    tolerance works for an image of size ``1e-3`` and one of size ``1e3``.  This is
    the same lesson as Phase 2's flatness criterion: a fixed threshold on a
    dimensioned quantity is wrong at some scale.

    Degeneracy of a triple uses ``triangle_flatness`` (dimensionless) rather than
    ``D == 0``, because ``D == 0`` is not attainable in floating point for a
    mathematically collinear triple.
    """
    from ..geometry.affine_frame import triangle_flatness

    matrix = np.asarray(as_point_matrix(points, name="points"), dtype=float)
    n = matrix.shape[0]
    if n < 3:
        return DOMAIN_CODE_UNDEFINED, {"reason": "fewer than three points", "n_points": n}

    diameter = float(np.abs(matrix).max()) or 1.0
    loosest = 0.0
    for i, j, k in itertools.combinations(range(n), 3):
        loosest = max(loosest, triangle_flatness(matrix[i], matrix[j], matrix[k]))
    if loosest <= 0.0:
        return DOMAIN_CODE_UNDEFINED, {"reason": "every triple is degenerate", "n_points": n}

    best: dict[str, Any] | None = None
    for triple in itertools.combinations(range(n), 3):
        i, j, k = triple
        if triangle_flatness(matrix[i], matrix[j], matrix[k]) <= tol:
            continue
        direction = matrix[j] - matrix[i]
        norm = float(math.hypot(direction[0], direction[1]))
        if norm <= 0:
            continue
        offsets = [
            float((p[0] - matrix[i][0]) * direction[1] - (p[1] - matrix[i][1]) * direction[0])
            / (norm * diameter)
            for p in matrix
        ]
        clustered = [offsets[0]]
        for value in sorted(offsets):
            if all(abs(value - existing) > tol for existing in clustered):
                clustered.append(value)
        if len(clustered) <= 2:
            best = {
                "direction_from_triple": [int(i), int(j), int(k)],
                "n_distinct_offsets": len(clustered),
                "tolerance": tol,
                "scale": diameter,
            }
            break

    if best is None:
        return DOMAIN_GENERAL_POSITION, {
            "reason": "no direction puts all points on at most two parallel lines",
            "n_points": n,
            "tolerance": tol,
        }
    if best["n_distinct_offsets"] == 1:
        return DOMAIN_CODE_UNDEFINED, {"reason": "all points lie on a single line", **best}
    return DOMAIN_TWO_PARALLEL_LINES, {"reason": "all points lie on two parallel lines", **best}


# --------------------------------------------------------------------------- #
# the validation report
# --------------------------------------------------------------------------- #
def noise_floor_for(points: ExactPoints) -> Fraction:
    """Determinant magnitude below which a computed area carries no information.

    ``64 * eps * diameter**2``: an *area*, so the comparison with ``D`` is
    dimensionally consistent (``D`` is also an area) and scale-free in the sense that
    both sides scale as length squared.

    Why this is needed at all: the exact layer's exactness protects the *test* (``D == 0``
    is exact), **not the input**.  A triple can be exactly non-degenerate with a
    determinantal area of ``1`` while the image diameter is ``1e7`` — a legitimate integer
    example is ``(0,0), (10**7, 3), (3333333, 1)``, whose ``D`` is exactly ``1`` and whose
    floor is ``~1.42``.  The definition admits it, and then ``rho`` reaches
    ``S_max / 1 ~ 1e13``, a magnitude that says nothing about the image.  This floor is
    what separates "admissible by definition" from "informative".
    """
    if not points:
        return Fraction(0)
    diameter = max(
        max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p in points for q in points
    )
    if diameter == 0:
        return Fraction(0)
    eps = Fraction(1, 2**52)
    return 64 * eps * diameter * diameter


@dataclass(frozen=True)
class KozlovImageValidation:
    """Everything the caller must know before building or trusting a code.

    The specification insists on three *independent* questions, and this object answers
    all three separately rather than merging them into one "ok":

    1. :attr:`code_definable` — does ``<M_A, T_A>`` exist?
    2. :attr:`theorem_applicable` — is theorem 1's iff guarantee in force?
    3. :attr:`numerically_well_conditioned` — do the numbers mean anything?

    Answering only the first and calling the result "validated" is the specific mistake
    this design exists to prevent.
    """

    n_points: int
    exact: bool
    duplicate_count: int
    duplicates: tuple[tuple[Any, int], ...]
    domain: DomainVerdict
    evidence: dict[str, Any]
    degenerate_triples: tuple[Triple, ...]
    #: Triples that are non-degenerate *exactly* but whose determinantal area is at
    #: rounding scale.  They are admissible by the definition and will dominate ``T_A``.
    noise_dominated_triples: tuple[Triple, ...] = field(default=())
    noise_floor: Fraction | None = None
    notes: tuple[str, ...] = field(default=())

    @property
    def code_definable(self) -> bool:
        """Whether ``<M_A, T_A>`` exists at all."""
        return self.domain != DOMAIN_CODE_UNDEFINED

    @property
    def theorem_applicable(self) -> bool:
        """Whether theorem 1's iff statement is in force."""
        return self.domain == DOMAIN_GENERAL_POSITION

    @property
    def numerically_well_conditioned(self) -> bool:
        """Whether the exact code's ratios will carry information.

        ``False`` when some triple has ``0 < D < noise_floor``: the definition admits it,
        but ``rho = S_num / S_den`` for such a denominator is a ratio of a rounding
        artefact to a real area, and its magnitude says nothing about the image.
        """
        return not self.noise_dominated_triples

    @property
    def conditioning_verdict(self) -> str:
        return "well_conditioned" if self.numerically_well_conditioned else "noise_dominated"

    def describe(self, lang: str = "zh") -> str:
        domain_text = {
            DOMAIN_CODE_UNDEFINED: {
                "zh": "A. 全部共线 ⇒ 正式代码不可定义",
                "en": "A. all collinear => the formal code does not exist",
            },
            DOMAIN_TWO_PARALLEL_LINES: {
                "zh": "B. 落在两条平行线上 ⇒ 代码可计算，但**不**适用定理 1 的 iff 保证",
                "en": "B. two parallel lines => the code exists, but theorem 1's iff does NOT apply",
            },
            DOMAIN_GENERAL_POSITION: {
                "zh": "C. 一般位置 ⇒ 进入定理 1 的正式适用域",
                "en": "C. general position => theorem 1 applies",
            },
        }[self.domain]
        head = {
            "zh": f"点数 {self.n_points}｜重复点 {self.duplicate_count}",
            "en": f"n_points {self.n_points} | duplicates {self.duplicate_count}",
        }
        lines = [head.get(lang, head["en"]), domain_text.get(lang, domain_text["en"])]
        lines.append(f"evidence: {self.evidence}")
        conditioning = {
            "zh": "数值良态" if self.numerically_well_conditioned else
                  f"数值被噪声主导：{len(self.noise_dominated_triples)} 个三元组的 |D| 低于舍入底噪",
            "en": "numerically well conditioned" if self.numerically_well_conditioned else
                  f"noise dominated: {len(self.noise_dominated_triples)} triple(s) below the rounding floor",
        }
        lines.append(conditioning.get(lang, conditioning["en"]))
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "exact": self.exact,
            "duplicate_count": self.duplicate_count,
            "duplicates": [[str(coord), count] for coord, count in self.duplicates],
            "domain": self.domain,
            "code_definable": self.code_definable,
            "theorem_applicable": self.theorem_applicable,
            "evidence": {k: (v if not isinstance(v, Fraction) else str(v)) for k, v in self.evidence.items()},
            "n_degenerate_triples": len(self.degenerate_triples),
            "conditioning_verdict": self.conditioning_verdict,
            "numerically_well_conditioned": self.numerically_well_conditioned,
            "noise_dominated_triples": [sorted(t) for t in self.noise_dominated_triples],
            "noise_floor": None if self.noise_floor is None else str(self.noise_floor),
            "notes": list(self.notes),
        }


def _duplicate_report(points: ExactPoints) -> tuple[tuple[tuple[Any, int], ...], int]:
    counts: dict[tuple[Fraction, Fraction], int] = {}
    for point in points:
        counts[point] = counts.get(point, 0) + 1
    duplicates = tuple(
        (coord, count) for coord, count in counts.items() if count > 1
    )
    surplus = sum(count - 1 for _, count in duplicates)
    return duplicates, surplus


def validate_kozlov_image(
    points: Any,
    *,
    require_theorem_domain: bool = False,
    allow_duplicates: bool = False,
) -> KozlovImageValidation:
    """Validate an image against the formal model and classify its domain.

    Raises
    ------
    NonExactCoordinateError
        Float coordinates handed to the exact layer.
    DuplicatePointError
        Unless ``allow_duplicates=True``; deduplication is the caller's explicit,
        visible decision, never this function's.
    CollinearImageError
        When every triple is degenerate, so no denominator exists.
    TheoremDomainError
        Only when ``require_theorem_domain=True`` and the image is *not* in general
        position.  Without that flag a two-parallel-lines image is accepted with a
        recorded note, because the code genuinely is computable there.
    """
    exact_points = _as_exact_points(points)
    duplicates, surplus = _duplicate_report(exact_points)
    if duplicates and not allow_duplicates:
        raise DuplicatePointError([(f"({c[0]}, {c[1]})", count) for c, count in duplicates])

    domain, evidence = classify_domain_exact(exact_points)

    degenerate: tuple[Triple, ...] = ()
    if domain != DOMAIN_CODE_UNDEFINED:
        areas = {
            frozenset(triple): _exact_double_area(
                exact_points[triple[0]], exact_points[triple[1]], exact_points[triple[2]]
            )
            for triple in itertools.combinations(range(len(exact_points)), 3)
        }
        degenerate = tuple(sorted((t for t, d in areas.items() if d == 0), key=sorted))

    noise_dominated: tuple[Triple, ...] = ()
    noise_floor = noise_floor_for(exact_points) if domain != DOMAIN_CODE_UNDEFINED else None
    if noise_floor is not None and noise_floor > 0:
        areas_all = {
            frozenset(triple): _exact_double_area(
                exact_points[triple[0]], exact_points[triple[1]], exact_points[triple[2]]
            )
            for triple in itertools.combinations(range(len(exact_points)), 3)
        }
        noise_dominated = tuple(
            sorted((t for t, d in areas_all.items() if d != 0 and d < noise_floor), key=sorted)
        )

    notes: list[str] = []
    if duplicates:
        notes.append("DUPLICATES_ALLOWED: duplicates were tolerated by request; the code's "
                     "label set no longer corresponds to a set of points")
    if noise_dominated:
        notes.append(
            f"NOISE_DOMINATED_TRIPLES: {len(noise_dominated)} triple(s) are non-degenerate "
            f"exactly but have |D| below the rounding floor ({noise_floor}); the definition "
            "admits them and they will dominate T_A, so this code must not be used as a "
            "recognition invariant. The usual cause is coordinates that are not exactly "
            "representable (e.g. 0.1) making an *intended* collinearity inexact."
        )
    if domain == DOMAIN_TWO_PARALLEL_LINES:
        notes.append(
            "THEOREM_DOMAIN_NOT_MET: the code can be computed, but theorem 1's "
            "iff-statement is stated for images that do not lie on one line or on two "
            "parallel lines; do not use this code as a complete recognition invariant"
        )

    validation = KozlovImageValidation(
        n_points=len(exact_points),
        exact=True,
        duplicate_count=surplus,
        duplicates=tuple((f"({c[0]}, {c[1]})", count) for c, count in duplicates),
        domain=domain,
        evidence=evidence,
        degenerate_triples=degenerate,
        noise_dominated_triples=noise_dominated,
        noise_floor=noise_floor,
        notes=tuple(notes),
    )

    if domain == DOMAIN_CODE_UNDEFINED:
        raise TooFewPointsError(len(exact_points)) if len(exact_points) < 3 else CollinearImageError(
            len(exact_points)
        )
    if require_theorem_domain and not validation.theorem_applicable:
        raise TheoremDomainError(domain, evidence)
    return validation


def validate_kozlov_image_numeric(
    points: Any, *, tol: float = 1e-9, require_theorem_domain: bool = False
) -> KozlovImageValidation:
    """Float counterpart of :func:`validate_kozlov_image`.

    Same three-valued domain, but degeneracy and the parallel-line test use a
    scale-free tolerance instead of exact zero.  Kept separate on purpose: the
    tolerance is a **numerical strategy**, not part of the mathematical definition.
    """
    matrix = np.asarray(as_point_matrix(points, name="points"), dtype=float)
    n = matrix.shape[0]

    counts: dict[tuple[float, float], int] = {}
    for point in matrix:
        key = (float(point[0]), float(point[1]))
        counts[key] = counts.get(key, 0) + 1
    duplicates = [(key, count) for key, count in counts.items() if count > 1]
    surplus = sum(count - 1 for _, count in duplicates)
    if duplicates:
        raise DuplicatePointError([(f"({c[0]}, {c[1]})", count) for c, count in duplicates])

    domain, evidence = classify_domain_numeric(matrix, tol=tol)

    from ..geometry.affine_frame import triangle_flatness

    degenerate = tuple(
        sorted(
            (frozenset(triple) for triple in itertools.combinations(range(n), 3)
             if triangle_flatness(matrix[triple[0]], matrix[triple[1]], matrix[triple[2]]) <= tol),
            key=sorted,
        )
    )

    notes: list[str] = []
    if domain == DOMAIN_TWO_PARALLEL_LINES:
        notes.append(
            "THEOREM_DOMAIN_NOT_MET: the code can be computed, but theorem 1's "
            "iff-statement is stated for images that do not lie on one line or on two "
            "parallel lines."
        )

    validation = KozlovImageValidation(
        n_points=n,
        exact=False,
        duplicate_count=surplus,
        duplicates=tuple((f"({c[0]}, {c[1]})", count) for c, count in duplicates),
        domain=domain,
        evidence=evidence,
        degenerate_triples=degenerate,
        notes=tuple(notes),
    )

    if domain == DOMAIN_CODE_UNDEFINED:
        raise TooFewPointsError(n) if n < 3 else CollinearImageError(n)
    if require_theorem_domain and not validation.theorem_applicable:
        raise TheoremDomainError(domain, evidence)
    return validation
