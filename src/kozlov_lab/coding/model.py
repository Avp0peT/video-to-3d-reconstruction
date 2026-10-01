"""Shared data model for Kozlov's formal image code (Phase 3).

The mathematical object
-----------------------
Kozlov's formal code of a point image ``A`` is a pair

.. math::

    K(A) = \\langle M_A,\\ T_A \\rangle

where

* :math:`M_A` is the **set of labels** assigned to the points of ``A`` (the source
  insists on an explicit numbering: "перенумеруют точки изображения А");
* :math:`T_A` is the set of **indexed** ratios

.. math::

    \\rho_{mnu,ksp} = \\frac{S_{mnu}}{S_{ksp}},

  with :math:`S_{mnu}` the area of the triangle spanned by the points with labels
  :math:`m,n,u`, the **order inside a triple irrelevant**, the two triples
  **distinct**, and the ratio undefined when :math:`S_{ksp}=0`.

Two consequences of that wording shape the whole module, and both are easy to get
wrong:

1. **A triple is an unordered set of three labels**, hence a ``frozenset`` of size 3
   — not an ordered tuple. The source says the order inside a triple does not
   matter, so storing an ordered triple would create spurious distinctions.
2. **The ratios stay indexed.** ``T_A`` is a set of
   :class:`RatioEntry` objects carrying *which* triples produced each value, not a
   bag of numbers. Collapsing it to ``sorted([rho1, rho2, ...])`` throws away the
   correlation structure that the equivalence relation is *defined on* (the
   renumbering bijection :math:`\\psi` acts on the indices). The helper
   :meth:`KozlovImageCode.value_multiset` exists for quick rejects, and is
   documented as **not** the code.

Exactness is a property of the *instance*, not of the class
-----------------------------------------------------------
:class:`RatioEntry.value` is a :class:`~fractions.Fraction` in the exact layer and a
``float`` in the numeric layer.  Keeping one model with an ``exact`` flag — rather
than two parallel class hierarchies — means a single comparison routine can be
written once, and the *only* place the mathematical/engineering split surfaces is
the tolerance used in :mod:`kozlov_lab.coding.equivalence`.  That is exactly where
the split belongs.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import json
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Triple",
    "RatioEntry",
    "KozlovImageCode",
    "DomainVerdict",
    "CODE_FORMAT_VERSION",
    "triple_key",
    "ArithmeticMode",
    "ModeMismatchError",
]

#: Which arithmetic a code was built with.  An ``Enum`` rather than a bare ``bool`` so
#: that mixing the two is a *type-level* mistake rather than a silent coercion: a
#: ``Fraction`` silently becoming a ``float`` is exactly how an "exact" verdict turns
#: into an approximate one without anyone noticing.  (``KozlovImageCode.exact`` is kept
#: for compatibility; it is derived from this.)
class ArithmeticMode(str, Enum):
    EXACT = "exact"
    NUMERIC = "numeric"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class ModeMismatchError(TypeError):
    """Raised when an exact code and a numeric code are compared against each other.

    Deliberately an error instead of a coercion.  ``codes_equivalent_*`` cannot answer
    "are these equivalent" across modes without either inventing precision or discarding
    it, and either choice would make the verdict's meaning depend on an implicit
    conversion.  The caller must convert explicitly, and thereby state the tolerance.
    """

    def __init__(self, left: "ArithmeticMode", right: "ArithmeticMode", operation: str) -> None:
        self.left = left
        self.right = right
        self.operation = operation
        super().__init__(
            f"{operation}: refusing to mix arithmetic modes ({left.value} vs {right.value}). "
            "An exact code and a numeric code cannot be compared without an explicit "
            "conversion, because either direction silently changes what the verdict means."
        )

#: An unordered triple of labels — order inside a triple is irrelevant by definition.
Triple = frozenset[int]

#: Bump when the canonical text changes, so codes from different implementations
#: can never compare equal by accident.
CODE_FORMAT_VERSION = "kozlov-image-code/1"

#: Three-valued domain classification.  The specification insists on separating
#: "the code can be computed" from "theorem 1's iff guarantee applies" from
#: "the numbers are trustworthy"; these are the first two.
DomainVerdict = str  # one of: "code_undefined" | "two_parallel_lines" | "general_position"

DOMAIN_CODE_UNDEFINED = "code_undefined"
DOMAIN_TWO_PARALLEL_LINES = "two_parallel_lines"
DOMAIN_GENERAL_POSITION = "general_position"


def triple_key(triple: Iterable[int]) -> Triple:
    """Coerce an iterable of labels to an unordered triple, validating its size."""
    frozen = frozenset(int(i) for i in triple)
    if len(frozen) != 3:
        raise ValueError(f"a triple must have exactly 3 distinct labels, got {sorted(frozen)}")
    return frozen


@dataclass(frozen=True)
class RatioEntry:
    """One indexed ratio :math:`\\rho_{mnu,ksp} = S_{mnu}/S_{ksp}`.

    ``numerator`` and ``denominator`` are unordered triples of labels and must be
    **distinct** — the source defines the ratio only for two *different* triples, so
    the trivially-one entries are not part of :math:`T_A`.
    """

    numerator: Triple
    denominator: Triple
    #: ``Fraction`` in the exact layer, ``float`` in the numeric layer.
    value: Fraction | float

    def __post_init__(self) -> None:
        if len(self.numerator) != 3 or len(self.denominator) != 3:
            raise ValueError("both triples of a ratio entry must have exactly 3 labels")
        if self.numerator == self.denominator:
            raise ValueError(
                "the two triples of a ratio must be distinct (S_ksp/S_ksp = 1 carries no "
                "information and is not part of T_A)"
            )

    # -- rendering -------------------------------------------------------- #
    def sort_key(self) -> tuple[Any, ...]:
        """Deterministic total order, independent of label values where possible."""
        return (
            tuple(sorted(self.numerator)),
            tuple(sorted(self.denominator)),
        )

    def describe(self) -> str:
        num = ",".join(str(i) for i in sorted(self.numerator))
        den = ",".join(str(i) for i in sorted(self.denominator))
        return f"rho({num};{den}) = {self.value}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "numerator": sorted(self.numerator),
            "denominator": sorted(self.denominator),
            "value": str(self.value),
        }

    def compared_to(self, other: "RatioEntry", tolerance: float | None) -> bool:
        """Whether this entry equals ``other`` (up to ``tolerance`` when numeric).

        ``tolerance=None`` means exact comparison, which is the only admissible mode
        for :class:`~fractions.Fraction` values.
        """
        if self.numerator != other.numerator or self.denominator != other.denominator:
            return False
        if tolerance is None:
            return self.value == other.value
        return abs(float(self.value) - float(other.value)) <= tolerance


@dataclass(frozen=True)
class KozlovImageCode:
    """``K(A) = <M_A, T_A>`` — the formal image code, with its trust diagnostics.

    Produced by exactly one of two builders, and the ``exact`` flag says which:

    * :func:`~kozlov_lab.coding.exact.build_kozlov_image_code_exact` — values are
      :class:`~fractions.Fraction`, degeneracy is an exact test, and this is the
      reference implementation;
    * :func:`~kozlov_lab.coding.numeric.build_kozlov_image_code_numeric` — values are
      ``float`` and degeneracy is decided by a *numerical strategy*.

    ``labels`` is :math:`M_A` and ``entries`` is :math:`T_A`, which stays **indexed**:
    each :class:`RatioEntry` records which two triples produced its value.  That
    indexing is not decoration — the equivalence relation of the theory is defined
    through a bijection on the labels, so a bag of numbers would not be implementable
    against the source.

    Code equality is **not** decided by this object.  :meth:`canonical_text` and
    :meth:`digest` serialise one *presentation*, and two equivalent images whose points
    are numbered differently have different presentations; use
    :func:`~kozlov_lab.coding.equivalence.codes_equivalent_exact` (or the numeric
    variant) for the real test.
    """

    labels: frozenset[int]
    entries: frozenset[RatioEntry]
    #: ``True`` for the exact (``Fraction``) layer, ``False`` for the numeric one.
    exact: bool
    n_points: int
    n_triples: int
    n_admissible_denominators: int
    degenerate_triples: frozenset[Triple]
    domain: DomainVerdict
    warnings: tuple[str, ...] = field(default=())

    # -- basic ----------------------------------------------------------- #
    @property
    def mode(self) -> ArithmeticMode:
        """Which arithmetic produced this code."""
        return ArithmeticMode.EXACT if self.exact else ArithmeticMode.NUMERIC

    def require_same_mode(self, other: "KozlovImageCode", operation: str) -> None:
        """Raise :class:`ModeMismatchError` unless both codes use the same arithmetic."""
        if self.mode != other.mode:
            raise ModeMismatchError(self.mode, other.mode, operation)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def theorem_applies(self) -> bool:
        """Whether theorem 1's iff statement is in force for this image.

        Only ``general_position`` qualifies.  A ``two_parallel_lines`` image yields a
        perfectly computable code, but the theorem explicitly excludes it — so any
        *identification* claim from such a code is not backed by the theorem.
        """
        return self.domain == DOMAIN_GENERAL_POSITION

    @property
    def is_complete(self) -> bool:
        """Whether every admissible ``(numerator, denominator)`` pair is present."""
        expected = self.n_triples * self.n_admissible_denominators - self.n_admissible_denominators
        return len(self.entries) == expected

    # -- quick structure-preserving summaries ---------------------------- #
    def value_multiset(self) -> Counter:
        """Multiset of ``rho`` values.

        **Not the code.** Useful as a cheap necessary condition, but it discards the
        indexing, so equality of multisets does *not* imply equality of codes (and
        therefore must never be used as the equivalence test).
        """
        return Counter(entry.value for entry in self.entries)

    def label_signature(self, label: int) -> tuple[Any, ...]:
        """Role/value fingerprint of one label, used to prune the equivalence search.

        The two triples of an entry are required to be *distinct*, not disjoint, so a
        label may appear in both — hence both roles are recorded explicitly rather
        than assumed.
        """
        terms: list[tuple[str, Any]] = []
        for entry in self.entries:
            if label in entry.numerator:
                terms.append(("num", entry.value))
            if label in entry.denominator:
                terms.append(("den", entry.value))
        terms.sort(key=lambda item: (item[0], str(item[1])))
        return tuple(terms)

    def signature_profile(self) -> tuple[Any, ...]:
        """Multiset of label fingerprints — **relabelling-invariant**, but not the code.

        A cheap necessary condition: two codes can only be equivalent if this profile
        matches.  It is *not* sufficient, because it forgets which fingerprint sits at
        which label index — see :func:`~kozlov_lab.coding.equivalence.find_bijection` for
        the real test.
        """
        return tuple(sorted((str(key), len(labels)) for key, labels in self.signature_classes().items()))

    def signature_classes(self) -> dict[tuple[Any, ...], list[int]]:
        """Group labels by :meth:`label_signature`.

        Labels in different classes can never correspond under a renumbering that
        preserves the code, so this partitions the search space for the bijection.
        """
        classes: dict[tuple[Any, ...], list[int]] = {}
        for label in sorted(self.labels):
            classes.setdefault(self.label_signature(label), []).append(label)
        return classes

    # -- canonical form --------------------------------------------------- #
    def canonical_text(self) -> str:
        """Deterministic text form.

        Entries are emitted with labels **relabelled to a canonical 0..n-1 order**
        derived from the labels themselves, so that the text of a code and of the
        same code with points renumbered identically is byte-identical *after* the
        renumbering is undone.  (Code equality is decided by
        :mod:`kozlov_lab.coding.equivalence`, not by this text.)
        """
        ordered_labels = sorted(self.labels)
        remap = {label: position for position, label in enumerate(ordered_labels)}
        lines = [
            f"{CODE_FORMAT_VERSION} exact={int(self.exact)} n_points={self.n_points} "
            f"n_entries={len(self.entries)} n_triples={self.n_triples} "
            f"n_admissible={self.n_admissible_denominators} domain={self.domain}"
        ]
        for entry in sorted(self.entries, key=_canonical_entry_key):
            num = ",".join(str(remap[i]) for i in sorted(entry.numerator))
            den = ",".join(str(remap[i]) for i in sorted(entry.denominator))
            lines.append(f"{num};{den}={entry.value}")
        return "\n".join(lines)

    def canonical_bytes(self) -> bytes:
        return self.canonical_text().encode("utf-8")

    def digest(self, algorithm: str = "sha256") -> str:
        """Hash of the canonical text — an index, never the invariant itself.

        A hash is discontinuous: an infinitesimal perturbation changes it while the
        images remain almost the same.  It cannot answer "how much do these differ",
        so it must not stand in for the code (README sections 7 and 23).
        """
        return hashlib.new(algorithm, self.canonical_bytes()).hexdigest()

    # -- serialisation ---------------------------------------------------- #
    def as_dict(self, *, include_entries: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": CODE_FORMAT_VERSION,
            "exact": self.exact,
            "labels": sorted(self.labels),
            "n_points": self.n_points,
            "n_entries": len(self.entries),
            "n_triples": self.n_triples,
            "n_admissible_denominators": self.n_admissible_denominators,
            "n_degenerate_triples": len(self.degenerate_triples),
            "degenerate_triples": [sorted(t) for t in sorted(self.degenerate_triples, key=sorted)],
            "domain": self.domain,
            "theorem_applies": self.theorem_applies,
            "complete": self.is_complete,
            "warnings": list(self.warnings),
            "digest": self.digest(),
        }
        if include_entries:
            payload["entries"] = [entry.as_dict() for entry in sorted(self.entries, key=_canonical_entry_key)]
        return payload

    def to_json(self, *, include_entries: bool = True, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(include_entries=include_entries), indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"KozlovImageCode(n={self.n_points}, |T_A|={len(self.entries)}, "
            f"exact={self.exact}, domain={self.domain})"
        )


def _canonical_entry_key(entry: RatioEntry) -> tuple[Any, ...]:
    """Order entries by their (already relabelled) index sets, not by their values."""
    return (tuple(sorted(entry.numerator)), tuple(sorted(entry.denominator)))
