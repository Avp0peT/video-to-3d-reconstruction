"""Kozlov's formal image code — **Phase 3 implementation** (unlocked 2026-10-01).

Status
------
``Phase 3`` is **unlocked**.  The permission gate was open question Q1, now resolved
from a **primary full text**: V. N. Kozlov,
«Распознавание изображений, представляемых конечным множеством точек»,
Фундаментальная и прикладная математика 15:5 (2009) 95–110.  See
``docs/specs/official_image_code.md`` for the verbatim definition, the equivalence
relation, the theorem and its domain, and the gate table (G3.1–G3.7 all met).

The object
----------
.. math::

    K(A) = \\langle M_A,\\ T_A \\rangle,
    \\qquad T_A = \\{\\, \\rho_{mnu,ksp} = S_{mnu}/S_{ksp} \\,\\}

with the numerator and denominator triples **unordered and distinct**, and the ratio
undefined when :math:`S_{ksp} = 0`.  ``T_A`` stays **indexed**: a bag of numbers is not
the code, and the equivalence relation is defined through a bijection on the indices.

Layering — the point of the package
-----------------------------------
=========================================  ==========================================
module                                     role
=========================================  ==========================================
:mod:`~kozlov_lab.coding.model`            the shared object ``<M_A, T_A>``
:mod:`~kozlov_lab.coding.exact`            ``Fraction`` implementation — the **oracle**
:mod:`~kozlov_lab.coding.numeric`          ``float`` implementation + a *strategy* for ``S = 0``
:mod:`~kozlov_lab.coding.validation`       the strict adapter and the domain classification
:mod:`~kozlov_lab.coding.equivalence`      the bijection psi, plus an independent geometric check
=========================================  ==========================================

The mathematical definition and its numerical approximation are kept in **different
modules on purpose**.  A project that computes a formal invariant with floats and then
describes the result as "the definition" has lost the ability to say when it is wrong;
here the exact layer is the arbiter and the numeric layer is tested against it.

Two facts that are easy to conflate, and that the API refuses to conflate
------------------------------------------------------------------------
1. **"The code could be computed" is not "the theorem applies".**
   :func:`~kozlov_lab.coding.validation.validate_kozlov_image` returns a three-valued
   domain: ``code_undefined`` (all points collinear), ``two_parallel_lines``
   (computable, but theorem 1's iff is *not* in force), and ``general_position`` (the
   theorem's actual hypothesis).  :attr:`KozlovImageCode.theorem_applies` exposes it.
2. **Repeated points are not part of the model.**  The formal object is a finite
   *set*; duplicates are refused by the validator, never merged silently.
   Deduplication, if wanted, is an explicit preprocessing step whose point-count
   change must be shown.

The pedagogical construction in :mod:`kozlov_lab.teaching` is a **different**
construction (frame-relative internal coordinates) and remains labelled as such; it is
not re-exported here and must not be confused with the formal code.
"""

from __future__ import annotations

from .equivalence import (
    DEFAULT_SEARCH_BUDGET,
    AffineEquivalenceResult,
    KozlovCodeEquivalence,
    affine_equivalent_exact,
    codes_equivalent_exact,
    codes_equivalent_numeric,
    cross_check_exact,
    find_bijection,
    images_equivalent_by_code_exact,
)
from .exact import (
    DEFAULT_MAX_ENTRIES,
    build_kozlov_image_code_exact,
    double_area,
    exact_ratio,
    max_entries_for,
    triple_double_areas,
)
from .model import (
    CODE_FORMAT_VERSION,
    DOMAIN_CODE_UNDEFINED,
    DOMAIN_GENERAL_POSITION,
    DOMAIN_TWO_PARALLEL_LINES,
    DomainVerdict,
    KozlovImageCode,
    RatioEntry,
    Triple,
    triple_key,
)
from .numeric import (
    NUMERICAL_STRATEGY_NOTE,
    build_kozlov_image_code_numeric,
    numeric_double_areas,
)
from .validation import (
    CollinearImageError,
    DuplicatePointError,
    KozlovImageError,
    KozlovImageValidation,
    NonExactCoordinateError,
    TheoremDomainError,
    TooFewPointsError,
    classify_domain_exact,
    classify_domain_numeric,
    validate_kozlov_image,
    validate_kozlov_image_numeric,
)

__all__ = [
    # model
    "CODE_FORMAT_VERSION",
    "Triple",
    "RatioEntry",
    "KozlovImageCode",
    "DomainVerdict",
    "triple_key",
    "DOMAIN_CODE_UNDEFINED",
    "DOMAIN_TWO_PARALLEL_LINES",
    "DOMAIN_GENERAL_POSITION",
    # exact layer (the oracle)
    "build_kozlov_image_code_exact",
    "triple_double_areas",
    "double_area",
    "exact_ratio",
    "max_entries_for",
    "DEFAULT_MAX_ENTRIES",
    # numeric layer
    "build_kozlov_image_code_numeric",
    "numeric_double_areas",
    "NUMERICAL_STRATEGY_NOTE",
    # validation / domain
    "validate_kozlov_image",
    "validate_kozlov_image_numeric",
    "classify_domain_exact",
    "classify_domain_numeric",
    "KozlovImageValidation",
    "KozlovImageError",
    "NonExactCoordinateError",
    "DuplicatePointError",
    "TooFewPointsError",
    "CollinearImageError",
    "TheoremDomainError",
    # equivalence
    "KozlovCodeEquivalence",
    "AffineEquivalenceResult",
    "codes_equivalent_exact",
    "codes_equivalent_numeric",
    "images_equivalent_by_code_exact",
    "find_bijection",
    "affine_equivalent_exact",
    "cross_check_exact",
    "DEFAULT_SEARCH_BUDGET",
]
