"""Floating-point implementation of Kozlov's image code.

Relationship to the mathematical definition — read this before using the module
------------------------------------------------------------------------------
This module computes the **same** object as :mod:`kozlov_lab.coding.exact`,

.. math::

    K(A) = \\langle M_A,\\ T_A\\rangle,
    \\qquad T_A = \\{\\, \\rho_{mnu,ksp} = S_{mnu}/S_{ksp} \\,\\},

but with ``float`` coordinates and, crucially, with a **substitute test for a
substitute question**.  The definition says the ratio is undefined when
:math:`S_{ksp}=0`; in floating point a mathematically collinear triple has a computed
determinant of ``~eps*scale^2`` rather than ``0``, so the literal test is unusable.

> **The degeneracy test used here is a numerical strategy, not part of the original
> mathematical definition.**  Concretely: a triple is treated as degenerate when its
> dimensionless flatness ``2*Area / max_side^2`` falls below ``flatness_tol``.  The
> tolerance, and the flatness criterion itself, belong to this module and to nothing
> else.

Where the two layers genuinely differ — measured, not assumed
-------------------------------------------------------------
An earlier draft of this docstring claimed that a *mathematically* collinear triple has a
computed determinant of ``~1e-16`` instead of ``0``.  A search over 400 000 exactly
collinear integer triples (including large ones whose products exceed ``2**53``) found
**no** such case: for exactly collinear stored values the float determinant is exactly
zero.  The claim was wrong and is retracted here.

The real mechanism is subtler and matters more:

* coordinates such as ``0.1`` are not representable, so a triple that is collinear **in
  the user's intent** is generally *not* exactly collinear in the stored values;
* the exact layer faithfully reports this: ``D != 0``, so the triple is admissible and
  ``rho`` reaches ``S_max / D`` — for a ``D`` at rounding scale that is astronomically
  large, and it is a faithful consequence of the input rather than a bug;
* the float computation may then round the same determinant to exactly ``0`` (dataset A
  of this project: exact ``D = 1.11e-18`` versus float flatness ``0.0``).

So the two layers **agree on the arithmetic** and disagree on the *question*:

============================  =====================================================
layer                         question it answers
============================  =====================================================
exact                         "is the stored configuration *exactly* degenerate?"
numeric (this module)         "is it degenerate *up to tolerance*?"
============================  =====================================================

Both answers are correct, and the second is a **strategy** — which is exactly why it lives
here and not in :mod:`kozlov_lab.coding.exact`.  The third layer,
:attr:`~kozlov_lab.coding.validation.KozlovImageValidation.numerically_well_conditioned`,
is what surfaces the resulting ill-conditioning to the caller.

Measured motivation for the tolerance (dataset B, a trapezoid with sampled bases):

======================================  ====================
quantity                                value
======================================  ====================
smallest **non-zero** doubled area      ``1.11e-16``
largest doubled area                    ``6.83``
so ``max rho`` under the literal test    ``3.07e16``
same quantity after a **pure translate** ``1.23e17``
======================================  ====================

The literal test makes ``T_A``'s largest element a function of rounding, and one that
changes under the weakest transform.  With the flatness test the same measurements are
stable to ``9.5e-14``.

That is exactly why the two layers are kept apart: the exact layer decides what the
definition *means*, and this layer decides what a computer can *do* — and then gets
validated against the exact layer (``tests/test_kozlov_exact_oracle.py``).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..config import COLLINEAR_TOL
from ..geometry.affine_frame import triangle_flatness
from ..geometry.points import as_point_matrix
from .model import (
    KozlovImageCode,
    RatioEntry,
    Triple,
)
from .validation import validate_kozlov_image_numeric
from .exact import DEFAULT_MAX_ENTRIES

__all__ = [
    "build_kozlov_image_code_numeric",
    "numeric_double_areas",
    "NUMERICAL_STRATEGY_NOTE",
]

#: Quoted verbatim wherever the numeric layer is documented or displayed, so the
#: boundary between definition and strategy cannot be lost in a later refactor.
NUMERICAL_STRATEGY_NOTE = (
    "这是 Kozlov 数学代码的浮点数值实现，而 flatness 是用于近似判断 S=0 的数值策略，"
    "不属于原始数学定义。 / This is a floating-point numerical implementation of Kozlov's "
    "mathematical code; the flatness criterion is a numerical strategy for approximating "
    "S = 0 and is NOT part of the original mathematical definition."
)


def numeric_double_areas(points: Any) -> dict[Triple, float]:
    """``D = 2*Area`` for every triple, as floats (display/diagnostics only)."""
    matrix = np.asarray(as_point_matrix(points, name="points", min_points=1), dtype=float)
    areas: dict[Triple, float] = {}
    for triple in itertools.combinations(range(matrix.shape[0]), 3):
        i, j, k = triple
        u = matrix[j] - matrix[i]
        v = matrix[k] - matrix[i]
        areas[frozenset(triple)] = float(abs(u[0] * v[1] - u[1] * v[0]))
    return areas


def build_kozlov_image_code_numeric(
    points: Any,
    *,
    flatness_tol: float = COLLINEAR_TOL,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    require_theorem_domain: bool = False,
) -> KozlovImageCode:
    """Float counterpart of :func:`~kozlov_lab.coding.exact.build_kozlov_image_code_exact`.

    Same ``<M_A, T_A>`` structure and the same :class:`~kozlov_lab.coding.model.RatioEntry`
    indexing — only the value type and the degeneracy test differ.
    """
    matrix = np.asarray(as_point_matrix(points, name="points", min_points=1), dtype=float)
    validation = validate_kozlov_image_numeric(
        matrix, tol=flatness_tol, require_theorem_domain=require_theorem_domain
    )
    n = matrix.shape[0]
    labels = frozenset(range(n))

    areas = numeric_double_areas(matrix)

    admissible: list[Triple] = []
    degenerate: set[Triple] = set()
    for triple in areas:
        i, j, k = sorted(triple)
        if triangle_flatness(matrix[i], matrix[j], matrix[k]) > flatness_tol:
            admissible.append(triple)
        else:
            degenerate.add(triple)
    admissible.sort(key=sorted)
    if not admissible:
        raise ValueError(
            "no admissible denominator: every triple is degenerate at "
            f"flatness_tol = {flatness_tol:g}"
        )

    n_pairs = len(areas) * len(admissible) - len(admissible)
    if n_pairs > max_entries:
        raise ValueError(
            f"the code would contain {n_pairs} ratio entries, above the cap {max_entries}. "
            "Refusing to truncate: a truncated T_A is not the code of the image."
        )

    entries: set[RatioEntry] = set()
    for numerator, d_num in areas.items():
        for denominator in admissible:
            if numerator == denominator:
                continue
            denominator_value = areas[denominator]
            if denominator_value == 0.0:  # pragma: no cover - flatness already excludes it
                continue
            entries.add(
                RatioEntry(
                    numerator=numerator,
                    denominator=denominator,
                    value=float(d_num / denominator_value),
                )
            )

    warnings = [*validation.notes]
    if any(degenerate):
        warnings.append(
            f"NUMERIC_DEGENERACY_STRATEGY: {len(degenerate)} triple(s) were excluded as "
            f"degenerate by the flatness test (tol = {flatness_tol:g}), not by an exact "
            "S = 0 test. See the module docstring."
        )

    code = KozlovImageCode(
        labels=labels,
        entries=frozenset(entries),
        exact=False,
        n_points=n,
        n_triples=len(areas),
        n_admissible_denominators=len(admissible),
        degenerate_triples=frozenset(degenerate),
        domain=validation.domain,
        warnings=tuple(warnings),
    )
    if not code.is_complete:
        raise AssertionError(  # pragma: no cover - internal consistency
            f"entry count {len(code.entries)} != expected {n_pairs}"
        )
    return code
