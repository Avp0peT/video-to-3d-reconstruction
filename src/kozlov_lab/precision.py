"""Attainable-precision reporting — replacing the unconditional ``1e-9`` claim.

Why this module exists
----------------------
The project README asks for ``error < 1e-9`` on the internal affine coordinates.
That sentence is only meaningful together with the conditions under which it can
hold, because two *independent* mechanisms degrade accuracy:

**M1 — conditioning of the frame.**  Solving ``M z = P - A`` with
``M = [B-A, C-A]`` obeys

.. math:: \\frac{\\|\\delta z\\|}{\\|z\\|} \\le \\mathrm{cond}(M)\\,\\varepsilon .

A very flat triangle has ``cond ~ 1e8`` and simply cannot deliver ``1e-9`` in
float64.

**M2 — magnitude of the coordinates.**  ``alpha`` divides a difference of
coordinates by the frame size, and a coordinate of magnitude ``|t|`` is only
resolved to ``ulp(|t|) = eps*|t|``, hence

.. math:: \\max|\\Delta\\alpha| \\approx \\frac{\\varepsilon\\,\\max\\|P\\|}{\\|B-A\\|} .

This loss happens in the *input* (the coordinates were already rounded), so it is
**not** covered by the conditioning bound.  Beyond
``eps*|t| >= |B-A|`` the transformed image collapses onto a single float and no
accuracy statement is possible at all.

Therefore every result that depends on these numbers carries a
:class:`PrecisionReport`: the two bounds, the total, a severity level, and
machine-readable warnings that the UI can render verbatim.  Nothing in this
project may assert a bare ``< 1e-9``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import numpy as np

from .config import (
    EPS,
    FLOAT_TOL,
    MATCH_TOL,
    MAX_CONDITION_NUMBER,
    UNUSABLE_ACCURACY,
    UNUSABLE_CONDITION_NUMBER,
    tolerance_for_condition,
)

__all__ = [
    "Severity",
    "Level",
    "PrecisionWarning",
    "PrecisionReport",
    "precision_report",
    "precision_of_frame",
    "precision_of_comparison",
    "precision_of_invariance",
    "frame_scale_of",
    "coordinate_scale_of",
]

Severity = Literal["info", "warning", "critical"]
Level = Literal["ok", "caution", "unusable", "unknown"]

#: Ratio of frame size to image size below which we stay silent about M2.
#: Chosen as ``MATCH_TOL`` because that is the accuracy the README asks for:
#: if the magnitude-induced error already exceeds it, the decimal digits the
#: README cares about are gone.
_MAGNITUDE_WARN_RATIO = MATCH_TOL


# --------------------------------------------------------------------------- #
# warnings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrecisionWarning:
    """A machine-readable accuracy warning, translatable for the UI."""

    code: str
    severity: Severity
    message_zh: str
    message_ru: str
    message_en: str
    detail: Mapping[str, float]

    def text(self, lang: str = "zh") -> str:
        return {
            "zh": self.message_zh,
            "ru": self.message_ru,
            "en": self.message_en,
        }.get(lang, self.message_en)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "zh": self.message_zh,
            "ru": self.message_ru,
            "en": self.message_en,
            "detail": dict(self.detail),
        }


def _warn_near_collinear(condition_number: float) -> PrecisionWarning:
    return PrecisionWarning(
        code="NEAR_COLLINEAR_FRAME",
        severity="warning" if condition_number <= UNUSABLE_CONDITION_NUMBER else "critical",
        message_zh=(
            f"标架接近共线：cond = {condition_number:.3e}。"
            f"该标架无法达到 1e-9，可达精度见 total_bound。"
        ),
        message_ru=(
            f"Базисный треугольник почти вырожден: cond = {condition_number:.3e}. "
            f"Точность 1e-9 недостижима; см. total_bound."
        ),
        message_en=(
            f"Frame is nearly collinear: cond = {condition_number:.3e}. "
            f"1e-9 is not attainable; see total_bound."
        ),
        detail={"condition_number": condition_number, "threshold": MAX_CONDITION_NUMBER},
    )


def _warn_magnitude_ratio(ratio: float, coordinate_scale: float, frame_scale: float) -> PrecisionWarning:
    return PrecisionWarning(
        code="TRANSLATION_MAGNITUDE_DOMINATES",
        severity="warning" if ratio < 1.0 else "critical",
        message_zh=(
            f"坐标量级远大于图像尺度：ε·|P| / |B-A| = {ratio:.3e}。"
            f"该误差发生在输入舍入阶段，条件数界看不到它。"
        ),
        message_ru=(
            f"Порядок координат много больше масштаба изображения: "
            f"eps*|P| / |B-A| = {ratio:.3e}. Эта погрешность возникает на входе."
        ),
        message_en=(
            f"Coordinate magnitude dominates the image scale: "
            f"eps*|P| / |B-A| = {ratio:.3e}. This loss is not covered by the conditioning bound."
        ),
        detail={
            "ratio": ratio,
            "coordinate_scale": coordinate_scale,
            "frame_scale": frame_scale,
        },
    )


def _warn_structure_destroyed(ratio: float) -> PrecisionWarning:
    return PrecisionWarning(
        code="IMAGE_STRUCTURE_DESTROYED",
        severity="critical",
        message_zh=(
            f"图像结构已被摧毁：ulp(|P|) ≥ |B-A|（比值 {ratio:.3e}）。"
            f"变换后的点会塌缩到同一个浮点数，任何精度声明都不成立。"
        ),
        message_ru=(
            f"Структура изображения разрушена: ulp(|P|) >= |B-A| (отношение {ratio:.3e}). "
            f"Точки после преобразования сливаются в одно число."
        ),
        message_en=(
            f"Image structure destroyed: ulp(|P|) >= |B-A| (ratio {ratio:.3e}). "
            f"Transformed points collapse onto one float; no accuracy statement is possible."
        ),
        detail={"ratio": ratio, "threshold": 1.0},
    )


def _warn_unusable_condition(condition_number: float) -> PrecisionWarning:
    return PrecisionWarning(
        code="FRAME_UNUSABLE",
        severity="critical",
        message_zh=(
            f"标架条件数过大：cond = {condition_number:.3e} > {UNUSABLE_CONDITION_NUMBER:.0e}。"
            f"内部坐标已无有效数字。"
        ),
        message_ru=(
            f"Число обусловленности слишком велико: cond = {condition_number:.3e}. "
            f"Внутренние координаты не имеют значащих цифр."
        ),
        message_en=(
            f"Condition number too large: cond = {condition_number:.3e} > "
            f"{UNUSABLE_CONDITION_NUMBER:.0e}. The internal coordinates carry no significant digits."
        ),
        detail={"condition_number": condition_number, "threshold": UNUSABLE_CONDITION_NUMBER},
    )


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrecisionReport:
    """Attainable precision for one comparison, with its decomposition."""

    condition_number: float
    bound_from_condition: float
    frame_scale: float | None
    coordinate_scale: float | None
    bound_from_magnitude: float | None
    total_bound: float
    level: Level
    warnings: tuple[PrecisionWarning, ...]

    # -- convenience ------------------------------------------------------ #
    @property
    def meets_readme_requirement(self) -> bool:
        """Whether the project README's ``1e-9`` is attainable *here*."""
        return self.total_bound <= MATCH_TOL

    @property
    def significant_decimal_digits(self) -> float:
        """Roughly how many decimal digits of ``(alpha, beta)`` survive."""
        if self.total_bound <= 0:
            return 16.0
        return max(0.0, -math.log10(self.total_bound))

    def is_acceptable(self, observed: float) -> bool:
        """Whether an observed deviation is within the reported bound."""
        return observed <= max(self.total_bound, FLOAT_TOL)

    # -- rendering -------------------------------------------------------- #
    def describe(self, lang: str = "zh") -> str:
        head = {
            "zh": "可达精度",
            "ru": "Достижимая точность",
            "en": "Attainable precision",
        }.get(lang, "Attainable precision")
        parts = [f"{head}: {self.total_bound:.3e}  (level={self.level})"]
        parts.append(f"  cond-bound = {self.bound_from_condition:.3e}  (cond = {self.condition_number:.3e})")
        if self.bound_from_magnitude is not None:
            parts.append(
                f"  magnitude-bound = {self.bound_from_magnitude:.3e}"
                f"  (eps*|P|/|B-A|, |P| = {self.coordinate_scale:.3e}, |B-A| = {self.frame_scale:.3e})"
            )
        for warning in self.warnings:
            parts.append(f"  [{warning.severity.upper()}] {warning.code}: {warning.text(lang)}")
        return "\n".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_number": self.condition_number,
            "bound_from_condition": self.bound_from_condition,
            "frame_scale": self.frame_scale,
            "coordinate_scale": self.coordinate_scale,
            "bound_from_magnitude": self.bound_from_magnitude,
            "total_bound": self.total_bound,
            "level": self.level,
            "meets_readme_requirement": self.meets_readme_requirement,
            "significant_decimal_digits": self.significant_decimal_digits,
            "warnings": [w.as_dict() for w in self.warnings],
        }

    def row(self, lang: str = "zh") -> dict[str, Any]:
        """Flat row for the UI tables required by the M3 page spec."""
        return {
            "predicted_attainable_precision": self.total_bound,
            "bound_from_condition": self.bound_from_condition,
            "bound_from_magnitude": self.bound_from_magnitude,
            "condition_number": self.condition_number,
            "level": self.level,
            "warning": self.warnings[0].text(lang) if self.warnings else "",
        }


# --------------------------------------------------------------------------- #
# constructors
# --------------------------------------------------------------------------- #
def frame_scale_of(frame: Any) -> float:
    """``|B - A|`` of a frame-like object (``AffineFrame`` or a sequence of points).

    Written with an explicit branch rather than ``getattr(frame, "a", frame[0])``:
    the default argument of ``getattr`` is evaluated **eagerly**, so the ``getattr``
    form raises on an ``AffineFrame`` (which is iterable but not subscriptable).
    """
    if hasattr(frame, "a") and hasattr(frame, "b"):
        a, b = frame.a, frame.b
    else:
        points = list(frame)
        a, b = points[0], points[1]
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    return float(math.hypot(float(second[0] - first[0]), float(second[1] - first[1])))


def coordinate_scale_of(points: Any) -> float:
    """Largest absolute coordinate component (the quantity that sets ``ulp``)."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.abs(arr).max())


def precision_report(
    condition_number: float,
    *,
    frame_scale: float | None = None,
    coordinate_scale: float | None = None,
) -> PrecisionReport:
    """Decompose the attainable precision into its two mechanisms.

    Parameters
    ----------
    condition_number:
        Worst condition number involved — for an invariance comparison, use
        ``max(cond_before, cond_after)``, because the answer is wrong if either
        solve is wrong.
    frame_scale:
        ``|B - A|``.  Required for the magnitude term.
    coordinate_scale:
        Largest absolute coordinate.  Required for the magnitude term.

    Notes
    -----
    The level is *not* a judgement about whether the result is "good": it is a
    statement about whether the README's ``1e-9`` is attainable.  ``total_bound``
    is the number that must be compared against any observed deviation.
    """
    cond = float(condition_number)
    warnings: list[PrecisionWarning] = []

    if not math.isfinite(cond):
        bound_cond = math.inf
        warnings.append(_warn_unusable_condition(cond))
    else:
        bound_cond = tolerance_for_condition(cond)
        if cond > UNUSABLE_CONDITION_NUMBER:
            warnings.append(_warn_unusable_condition(cond))
        elif cond > MAX_CONDITION_NUMBER:
            warnings.append(_warn_near_collinear(cond))

    bound_mag: float | None = None
    ratio: float | None = None
    if frame_scale is not None and coordinate_scale is not None:
        fs = float(frame_scale)
        cs = float(coordinate_scale)
        if fs > 0.0 and cs >= 0.0:
            bound_mag = EPS * cs / fs
            ratio = bound_mag
            if ratio >= 1.0:
                warnings.append(_warn_structure_destroyed(ratio))
            elif ratio > _MAGNITUDE_WARN_RATIO:
                warnings.append(_warn_magnitude_ratio(ratio, cs, fs))
        elif fs <= 0.0:
            bound_mag = math.inf
            ratio = math.inf
            warnings.append(_warn_structure_destroyed(ratio))

    total = bound_cond if bound_mag is None else max(bound_cond, bound_mag)

    if not math.isfinite(total):
        level: Level = "unusable"
    elif total > UNUSABLE_ACCURACY:
        level = "unusable"
    elif total > MATCH_TOL:
        level = "caution"
    else:
        level = "ok"

    # an explicit critical warning always wins over the numeric level
    if any(w.severity == "critical" for w in warnings):
        level = "unusable"

    return PrecisionReport(
        condition_number=cond,
        bound_from_condition=bound_cond,
        frame_scale=None if frame_scale is None else float(frame_scale),
        coordinate_scale=None if coordinate_scale is None else float(coordinate_scale),
        bound_from_magnitude=bound_mag,
        total_bound=total,
        level=level,
        warnings=tuple(warnings),
    )


def precision_of_frame(
    frame: Any,
    probe_points: Any | None = None,
    *,
    coordinate_points: Any | None = None,
) -> PrecisionReport:
    """Precision report for internal coordinates computed in ``frame``.

    ``probe_points`` supplies the frame's own condition number if given as a
    ``(n,2)`` array; otherwise the frame's own ``condition_number`` attribute is
    used when available.
    """
    fs = frame_scale_of(frame)
    cond = getattr(frame, "condition_number", None)
    if cond is None:  # pragma: no cover - defensive
        from .geometry.affine_frame import _condition_number  # local import: avoids a cycle

        cond = _condition_number(np.asarray(getattr(frame, "matrix"), dtype=float))

    candidates = [coordinate_points, probe_points]
    coordinate = None
    for candidate in candidates:
        if candidate is not None:
            coordinate = coordinate_scale_of(candidate)
            break
    return precision_report(
        float(cond),
        frame_scale=fs,
        coordinate_scale=coordinate,
    )


def precision_of_invariance(
    delta: Any,
    *,
    frame_scale: float,
    coordinate_scale: float,
) -> PrecisionReport:
    """Precision report for an ``InvarianceDelta`` produced by the geometry layer.

    ``delta`` is duck-typed: any object exposing ``condition_number_before`` and
    ``condition_number_after`` works, which keeps this module free of an import
    cycle with :mod:`kozlov_lab.geometry`.

    The point of the helper is that the *observed* deviation and the *predicted*
    attainable precision are computed in the same call site, so a report cannot
    quote one without the other.
    """
    return precision_of_comparison(
        float(delta.condition_number_before),
        float(delta.condition_number_after),
        frame_scale=frame_scale,
        coordinate_scale=coordinate_scale,
    )


def precision_of_comparison(
    condition_before: float,
    condition_after: float,
    *,
    frame_scale: float | None = None,
    coordinate_scale: float | None = None,
) -> PrecisionReport:
    """Precision report for a before/after invariance comparison.

    Uses ``max(cond_before, cond_after)``: the comparison is only as good as its
    worse solve.
    """
    return precision_report(
        max(float(condition_before), float(condition_after)),
        frame_scale=frame_scale,
        coordinate_scale=coordinate_scale,
    )
