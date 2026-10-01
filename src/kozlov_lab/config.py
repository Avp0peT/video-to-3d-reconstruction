"""Unified numerical tolerances.

The project README (section 19, "数值稳定性") requires a *single* place that
defines every tolerance used by the mathematical layer.  Nothing in
``kozlov_lab`` may hard-code a magic epsilon: import from here instead.
"""

from __future__ import annotations

__all__ = [
    "FLOAT_TOL",
    "COLLINEAR_TOL",
    "MATCH_TOL",
    "SIGNATURE_DECIMALS",
    "MAX_CONDITION_NUMBER",
    "UNUSABLE_CONDITION_NUMBER",
    "UNUSABLE_ACCURACY",
    "TEACHING_CODE_DECIMALS",
    "EPS",
    "tolerance_for_condition",
]

import math

import numpy as np

#: Machine epsilon of ``float64``.
EPS: float = float(np.finfo(np.float64).eps)

#: Absolute tolerance for "these two floats are the same number".
#: Used for exact-arithmetic-style comparisons (identity of matrices,
#: zero translations, ...).
FLOAT_TOL: float = 1e-12

#: Scale-invariant collinearity threshold, expressed as the *sine* of the
#: angle between two spanning vectors:
#:
#:     sin(angle BAC) = |det[B-A, C-A]| / (|B-A| * |C-A|)
#:
#: Using the sine (a dimensionless quantity) instead of the raw determinant is
#: essential: ``det[B-A, C-A]`` is an *area* and therefore scales with the
#: image size.  A raw determinant threshold would call a huge triangle
#: "collinear" and a tiny one "generic".  ``COLLINEAR_TOL`` is the smallest
#: sine we still accept when building an affine frame.
COLLINEAR_TOL: float = 1e-9

#: Tolerance used by recognition ("do these two point sets match?").
MATCH_TOL: float = 1e-9

#: Decimal places used when quantising points into an order-independent
#: canonical signature.  See ``PointSet2D.signature``.
SIGNATURE_DECIMALS: int = 9

#: Above this condition number of the 2x2 frame matrix we refuse to make
#: numerical claims at the ``MATCH_TOL`` level.  See
#: :func:`tolerance_for_condition`.
#:
#: The value is not arbitrary: it is exactly where ``cond * eps`` reaches
#: ``MATCH_TOL`` (``1e-9 / 2.22e-16 ≈ 1e6``), i.e. the point at which the
#: project README's ``error < 1e-9`` requirement stops being attainable by any
#: float64 algorithm.  It is therefore "the 1e-9 budget spent", not a taste
#: parameter.
MAX_CONDITION_NUMBER: float = 1e6

#: Beyond this, the internal coordinates carry no significant digits at all and
#: results must be reported as unusable rather than merely imprecise.
UNUSABLE_CONDITION_NUMBER: float = 1e12

#: Accuracy bound above which a result is reported as unusable (fewer than
#: three decimal digits of ``(alpha, beta)`` survive).
UNUSABLE_ACCURACY: float = 1e-3

#: Default quantisation resolution of the *teaching* affine code
#: (``kozlov_lab.teaching``).  Must stay coarser than the attainable precision
#: of a well-conditioned frame, otherwise the canonical form would encode
#: floating-point noise as if it were structure.
TEACHING_CODE_DECIMALS: int = 9


def tolerance_for_condition(condition_number: float, base: float = FLOAT_TOL) -> float:
    """Return the error level actually achievable for a linear solve.

    Why this exists
    ---------------
    The project README demands ``error < 1e-9`` for internal affine
    coordinates.  That claim is *only* meaningful together with the
    conditioning of the frame, because solving ``M z = r`` with
    ``M = [B-A, C-A]`` satisfies the classical bound

        ||dz|| / ||z||  <=  cond(M) * eps_machine

    A very flat triangle can have ``cond(M) = 1e12``; no algorithm can then
    reproduce ``alpha``/``beta`` to 1e-9, and pretending otherwise would be a
    silent lie.  This helper makes the achievable tolerance explicit so that
    tests can assert the *right* thing:

    * well-conditioned frame  -> :data:`FLOAT_TOL` (and reality is ~1e-15);
    * ill-conditioned frame   -> ``1e-9``-level demands are downgraded to the
      condition-relative bound instead of being reported as a failure.

    Parameters
    ----------
    condition_number:
        Condition number of the frame matrix (2-norm).
    base:
        Floor tolerance for well-conditioned problems.
    """
    cond = float(condition_number)
    if not math.isfinite(cond) or cond <= 1.0:
        return base
    return max(base, cond * EPS * 64.0)
