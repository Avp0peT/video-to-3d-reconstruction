"""Invariant measurements — the engine behind the "Invariant Explorer" (M6).

Two fundamentally different kinds of property
---------------------------------------------
The table in README section 9 mixes two things, and measuring them the same way
would be a methodological mistake:

**Continuous quantities** (absolute coordinates, distances, angles, length
ratios, internal affine coordinates)
    A property says "this *number* survives".  The verdict is a comparison of
    two numbers, so we report the largest violation and compare it against a
    tolerance.

**Relations** (collinearity, parallelism)
    An affine map does *not* preserve the angle between two lines — it
    preserves the *fact of being parallel*.  So the right object is the
    **set** of index triples / index quadruples that stand in the relation, and
    the verdict is "is this set unchanged?".

Three-state verdicts, because "preserved" must carry information
---------------------------------------------------------------
A naive relation test returns "preserved" whenever both sets are empty, which
happens for almost every random point set (no three random points are exactly
collinear).  Such a verdict is *vacuous* — it would pass even for a completely
broken implementation.  This module therefore returns one of::

    "preserved"   the relation/quantity survived, and there was something to survive
    "violated"    a witness of failure exists (with its index tuple)
    "vacuous"     nothing was tested (e.g. no collinear triple exists in X)

The GUI displays vacuous rows in grey with the text "无见证，此结论无信息量",
and the test suite contains at least one *negative* test per property: a
transform under which the property is demonstrably violated.  Without that
negative control a green table would be meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from ..config import COLLINEAR_TOL, FLOAT_TOL, MATCH_TOL
from .affine_frame import AffineFrame, collinearity_sine, triangle_flatness
from .points import as_point_matrix
from .transforms import AffineTransform

__all__ = [
    "Verdict",
    "TheoryVerdict",
    "InvariantEntry",
    "InvariantReport",
    "TheoryRow",
    "TheoryCheck",
    "Discrepancy",
    "PROPERTY_KEYS",
    "THEORY_EXPECTATION",
    "FAMILY_LABELS_ZH",
    "theory_verdict",
    "check_against_theory",
    "absolute_coordinate_change",
    "distance_change",
    "angle_change",
    "length_ratio_change",
    "collinearity_relation_change",
    "parallelism_relation_change",
    "affine_coordinate_change",
    "invariant_report",
]

Verdict = Literal["preserved", "violated", "vacuous"]

#: What the theory (README section 9) predicts, per transformation *kind*.
#:
#: ``"preserved_or_vacuous"`` is the honest value for the two relation-valued
#: properties: the relation is preserved, but if the image contains no instance
#: of it there is nothing to preserve and the verdict must be ``vacuous``.
TheoryVerdict = Literal["preserved", "violated", "preserved_or_vacuous", "unknown"]

#: Canonical property order, matching README section 9 top-to-bottom.
PROPERTY_KEYS: tuple[str, ...] = (
    "absolute_coordinates",
    "distances",
    "angles",
    "length_ratios",
    "collinearity",
    "parallelism",
    "affine_coordinates",
)

_ISO = {
    "absolute_coordinates": "violated",
    "distances": "preserved",
    "angles": "preserved",
    "length_ratios": "preserved",
    "collinearity": "preserved_or_vacuous",
    "parallelism": "preserved_or_vacuous",
    "affine_coordinates": "preserved",
}

#: The theory table as data.  This is the single source of truth used by
#: :func:`check_against_theory`, by the tests, and by the Invariant Explorer
#: page — so a disagreement between "what the theory says" and "what we measure"
#: is always a *visible* discrepancy rather than a stale copy in a docstring.
THEORY_EXPECTATION: dict[str, dict[str, TheoryVerdict]] = {
    "identity": {
        "absolute_coordinates": "preserved",
        "distances": "preserved",
        "angles": "preserved",
        "length_ratios": "preserved",
        "collinearity": "preserved_or_vacuous",
        "parallelism": "preserved_or_vacuous",
        "affine_coordinates": "preserved",
    },
    "translation": dict(_ISO),
    "rotation": dict(_ISO),
    "reflection": dict(_ISO),
    "similarity": {
        "absolute_coordinates": "violated",
        "distances": "violated",  # unless s == 1, in which case it is an isometry
        "angles": "preserved",
        "length_ratios": "preserved",
        "collinearity": "preserved_or_vacuous",
        "parallelism": "preserved_or_vacuous",
        "affine_coordinates": "preserved",
    },
    "affine": {
        "absolute_coordinates": "violated",
        "distances": "violated",
        "angles": "violated",
        "length_ratios": "violated",
        "collinearity": "preserved_or_vacuous",
        "parallelism": "preserved_or_vacuous",
        "affine_coordinates": "preserved",
    },
    "singular": {
        "absolute_coordinates": "unknown",
        "distances": "unknown",
        "angles": "unknown",
        "length_ratios": "unknown",
        "collinearity": "unknown",
        "parallelism": "unknown",
        "affine_coordinates": "unknown",
    },
    "unknown": {key: "unknown" for key in PROPERTY_KEYS},
}

#: Human labels for the theory table rows (readme section 9 wording).
FAMILY_LABELS_ZH: dict[str, str] = {
    "identity": "恒等",
    "translation": "平移",
    "rotation": "旋转",
    "reflection": "反射",
    "similarity": "相似",
    "affine": "仿射",
    "singular": "退化",
    "unknown": "未知",
}


def theory_verdict(
    kind: str,
    property_key: str,
    *,
    transform: "AffineTransform | None" = None,
) -> TheoryVerdict:
    """Predicted verdict for ``property_key`` under a transform of ``kind``.

    Most of the table is static, but two entries are genuinely *conditional* and
    are refined when the transform is supplied:

    * ``similarity`` + ``distances`` — preserved iff ``|s - 1| <= MATCH_TOL``
      (a similarity with ``s = 1 + 5e-10`` does change distances, but only at a
      level the measurement cannot resolve; reporting a "discrepancy" there
      would be a tolerance artefact, not a finding).

    Without a transform the static table is used, which is the right answer for
    documentation and for the rendered theory table.
    """
    if kind == "similarity" and property_key == "distances" and transform is not None:
        return "violated" if abs(transform.scale_factor - 1.0) > MATCH_TOL else "preserved"
    table = THEORY_EXPECTATION.get(kind)
    if table is None:
        return "unknown"
    return table.get(property_key, "unknown")


@dataclass(frozen=True)
class TheoryRow:
    """One row of the theory-vs-measurement comparison."""

    property_key: str
    label_zh: str
    expected: TheoryVerdict
    measured: Verdict
    agreed: bool
    delta: float
    tolerance: float
    witness: str | None
    note: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "property": self.property_key,
            "label_zh": self.label_zh,
            "theory": self.expected,
            "measured": self.measured,
            "agreed": self.agreed,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "witness": self.witness,
            "note": self.note,
        }


@dataclass(frozen=True)
class Discrepancy:
    """Theory and measurement disagree — must never be hidden."""

    property_key: str
    label_zh: str
    expected: TheoryVerdict
    measured: Verdict
    delta: float
    tolerance: float
    severity: Literal["warning", "critical"]
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "property": self.property_key,
            "label_zh": self.label_zh,
            "expected": self.expected,
            "measured": self.measured,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "severity": self.severity,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class TheoryCheck:
    """Result of comparing a measurement against the theory table."""

    kind: str
    rows: tuple[TheoryRow, ...]
    discrepancies: tuple[Discrepancy, ...]

    @property
    def has_discrepancy(self) -> bool:
        return bool(self.discrepancies)

    @property
    def critical(self) -> tuple[Discrepancy, ...]:
        return tuple(d for d in self.discrepancies if d.severity == "critical")

    def headline(self, lang: str = "zh") -> str:
        if self.discrepancies:
            return {
                "zh": f"发现 {len(self.discrepancies)} 处理论/实测不一致",
                "ru": f"Обнаружено расхождений: {len(self.discrepancies)}",
                "en": f"{len(self.discrepancies)} theory/measurement discrepancy(ies)",
            }.get(lang, f"{len(self.discrepancies)} discrepancy(ies)")
        n_checked = sum(1 for r in self.rows if r.expected != "unknown")
        if n_checked == 0:
            # An all-"unknown" table must NOT read as a pass: nothing was compared.
            return {
                "zh": f"无可比对项：变换类型 `{self.kind}` 没有理论预测（对照实验 / 未知变换）",
                "ru": f"Нет сравнимых пунктов: для типа `{self.kind}` нет теоретического прогноза",
                "en": f"Nothing comparable: no theory prediction for kind `{self.kind}`",
            }.get(lang, f"Nothing comparable for kind `{self.kind}`")
        return {
            "zh": f"理论预测与实测一致（{n_checked} 项可比对）",
            "ru": f"Теория и измерение согласуются ({n_checked} сравнимых пунктов)",
            "en": f"Theory and measurement agree ({n_checked} comparable items)",
        }.get(lang, f"Theory and measurement agree ({n_checked} items)")

    def banner(self, lang: str = "zh") -> str | None:
        """The literal discrepancy banner required by the Invariant Explorer spec."""
        if not self.discrepancies:
            return None
        worst = max(self.discrepancies, key=lambda d: (d.severity == "critical", abs(d.delta)))
        lines = [
            "NUMERICAL / IMPLEMENTATION DISCREPANCY",
            f"property = {worst.property_key} ({worst.label_zh})",
            f"theory  = {worst.expected}",
            f"measured= {worst.measured}",
            f"delta   = {worst.delta:.3e}   tolerance = {worst.tolerance:.3e}",
            f"severity= {worst.severity}",
            worst.explanation,
        ]
        if len(self.discrepancies) > 1:
            lines.append(f"(+{len(self.discrepancies) - 1} more)")
        return "\n".join(lines)

    def as_rows(self) -> list[dict[str, Any]]:
        return [row.as_row() for row in self.rows]

    def markdown_table(self) -> str:
        symbol = {"preserved": "✓", "violated": "×", "vacuous": "–"}
        lines = [
            "| 性质 | 理论预测 | 实测 | 一致 | max \\|Δ\\| | 容差 | 见证 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            mark = "✓" if row.agreed else "**✗**"
            lines.append(
                f"| {row.label_zh} | {row.expected} | {symbol[row.measured]} {row.measured} | {mark} | "
                f"{row.delta:.3e} | {row.tolerance:.1e} | {row.witness or '—'} |"
            )
        return "\n".join(lines)


def check_against_theory(
    report: "InvariantReport",
    *,
    transform: "AffineTransform | None" = None,
) -> TheoryCheck:
    """Compare a measurement against :data:`THEORY_EXPECTATION`.

    Agreement rule:

    * ``preserved`` / ``violated`` must match exactly;
    * ``preserved_or_vacuous`` accepts either ``preserved`` or ``vacuous``;
    * ``unknown`` (degenerate transform, or a control pair) never counts as a
      discrepancy — but it is also excluded from the "comparable items" count,
      so a table full of ``unknown`` cannot look like a pass.
    """
    rows: list[TheoryRow] = []
    discrepancies: list[Discrepancy] = []

    for entry in report.entries:
        expected = theory_verdict(report.transform_kind, entry.key, transform=transform)
        measured = entry.verdict

        if expected == "unknown":
            agreed = True
        elif expected == "preserved_or_vacuous":
            agreed = measured in ("preserved", "vacuous")
        else:
            agreed = measured == expected

        rows.append(
            TheoryRow(
                property_key=entry.key,
                label_zh=entry.label_zh,
                expected=expected,
                measured=measured,
                agreed=agreed,
                delta=entry.delta,
                tolerance=entry.tolerance,
                witness=entry.witness,
                note=entry.note,
            )
        )
        if agreed:
            continue

        severity, explanation = _explain_discrepancy(expected, measured)
        discrepancies.append(
            Discrepancy(
                property_key=entry.key,
                label_zh=entry.label_zh,
                expected=expected,
                measured=measured,
                delta=entry.delta,
                tolerance=entry.tolerance,
                severity=severity,
                explanation=explanation,
            )
        )

    return TheoryCheck(report.transform_kind, tuple(rows), tuple(discrepancies))


def _explain_discrepancy(
    expected: TheoryVerdict, measured: Verdict
) -> tuple[Literal["warning", "critical"], str]:
    """Classify a theory/measurement disagreement and explain its likely cause.

    The distinction that matters: a ``vacuous`` measurement is **untested**, not
    falsified, so it is a warning; anything else is a genuine conflict and gets
    the critical banner.
    """
    if measured == "vacuous":
        return (
            "warning",
            "该性质在图像中不存在可检验的见证：结论**未被检验**，不含信息。"
            "不是反例，但也不能算作通过（见 docs/specs/invariants.md §2）。",
        )
    if expected == "preserved":
        return (
            "critical",
            "理论要求保持，实测报破坏。检查三处：① 容差是否与该性质的量纲/尺度匹配；"
            "② 逐点对应关系是否正确；③ 是否落入平移幅度导致的输入精度失效区间"
            "（见 docs/theory/numerical_limits.md §2）。",
        )
    if measured == "preserved":
        return (
            "critical",
            "理论要求破坏，实测报保持。这通常指向**测量方法错误**，而不是数学错误："
            "例如把关系型性质（平行性）当连续量（夹角）来测，"
            "或对已排序的坐标做逐点比较（见 docs/specs/invariants.md §1）。",
        )
    return ("critical", "理论预测与实测结论不一致，需要人工核查。")


# --------------------------------------------------------------------------- #
# report containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InvariantEntry:
    """One row of the invariant table."""

    key: str
    label_zh: str
    label_ru: str
    measure: str
    value_before: float
    value_after: float
    delta: float
    tolerance: float
    verdict: Verdict
    witness_count: int
    witness: str | None = None
    note: str | None = None

    @property
    def preserved(self) -> bool:
        """``True`` only for a *non-vacuous* preservation."""
        return self.verdict == "preserved"

    def as_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label_zh": self.label_zh,
            "label_ru": self.label_ru,
            "measure": self.measure,
            "before": self.value_before,
            "after": self.value_after,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "verdict": self.verdict,
            "witness_count": self.witness_count,
            "witness": self.witness,
        }


@dataclass(frozen=True)
class InvariantReport:
    """Result of running the whole invariant table on one (X, Y) pair."""

    transform_kind: str
    invariant_group: str
    n_points: int
    entries: tuple[InvariantEntry, ...]

    def get(self, key: str) -> InvariantEntry:
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(key)

    def verdicts(self) -> dict[str, Verdict]:
        return {e.key: e.verdict for e in self.entries}

    def as_rows(self) -> list[dict[str, Any]]:
        return [e.as_row() for e in self.entries]

    def to_json(self) -> dict[str, Any]:
        return {
            "transform_kind": self.transform_kind,
            "invariant_group": self.invariant_group,
            "n_points": self.n_points,
            "rows": self.as_rows(),
        }

    def markdown_table(self) -> str:
        """Render the table exactly as in README section 9, but *measured*."""
        symbol = {"preserved": "✓", "violated": "×", "vacuous": "–"}
        head = "| 性质 | 结果 | max |Δ| | 容差 | 见证数 |"
        sep = "|---|---|---|---|---|"
        lines = [head, sep]
        for e in self.entries:
            lines.append(
                f"| {e.label_zh} | {symbol[e.verdict]} | {e.delta:.3e} | "
                f"{e.tolerance:.1e} | {e.witness_count} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # theory comparison (used by the Invariant Explorer and by the tests)
    # ------------------------------------------------------------------ #
    def theory_check(self, transform: "AffineTransform | None" = None) -> "TheoryCheck":
        """Compare this measurement against the theory table.

        Pass the transform whenever it is known: two entries of the table are
        conditional on it (see :func:`theory_verdict`).
        """
        return check_against_theory(self, transform=transform)

    def theory_table(self, transform: "AffineTransform | None" = None) -> str:
        """Markdown rendering of theory vs. measurement."""
        return self.theory_check(transform).markdown_table()


# --------------------------------------------------------------------------- #
# continuous quantities
# --------------------------------------------------------------------------- #
def absolute_coordinate_change(x: Any, y: Any) -> tuple[float, int, str]:
    """``max_i ||x_i - y_i||`` under the given index correspondence.

    Uses the *correspondence* supplied by the caller (the i-th transformed point
    is the image of the i-th point).  This is legitimate here because the
    laboratory applies the transform itself; the correspondence-free version is
    the *recognition* problem of Phase 4 and is a strictly harder question.
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    d = np.linalg.norm(Y - X, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), 1, f"点 #{i}"


def distance_change(x: Any, y: Any) -> tuple[float, list[tuple[int, int]], float]:
    """``max_{i<j} |d_ij - d'_ij|`` together with its witnesses and the scale.

    The comparison tolerance is *relative* to the image diameter, so the same
    threshold works for an image of size 1e-3 and one of size 1e3.
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    dx = _pairwise_distance_matrix(X)
    dy = _pairwise_distance_matrix(Y)
    iu = np.triu_indices(X.shape[0], k=1)
    delta = np.abs(dy[iu] - dx[iu])
    order = np.argsort(delta)[::-1]
    worst = [(int(iu[0][k]), int(iu[1][k])) for k in order[:3] if delta[k] > 0]
    scale = max(float(dx.max()) if dx.size else 0.0, 1.0)
    return (float(delta.max()) if delta.size else 0.0), worst, scale


def angle_change(x: Any, y: Any, *, max_points: int = 30) -> tuple[float, list[tuple[int, int, int]], int]:
    """``max`` change of the interior angles of all point triples, in radians.

    Every triple ``(i, j, k)`` contributes its three interior angles.  The
    tolerance is absolute (radians) because an angle is already dimensionless;
    ``1e-9`` rad is far above the ~1e-16 level of floating-point noise.
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    if X.shape[0] > max_points:
        raise ValueError(
            f"angle_change: {X.shape[0]} points would need C(n,3) triples; "
            f"raise max_points explicitly (currently {max_points})"
        )
    worst_delta = 0.0
    witnesses: list[tuple[int, int, int]] = []
    n_tested = 0
    for (i, j, k) in itertools.combinations(range(X.shape[0]), 3):
        for vertex, p, q in ((i, j, k), (j, i, k), (k, i, j)):
            a1 = _angle_at(X[vertex], X[p], X[q])
            a2 = _angle_at(Y[vertex], Y[p], Y[q])
            if math.isnan(a1) or math.isnan(a2):
                continue  # degenerate triple in one of the images
            n_tested += 1
            delta = abs(a1 - a2)
            if delta > worst_delta:
                worst_delta = delta
                witnesses = [(vertex, p, q)]
            elif delta > 0 and delta == worst_delta and len(witnesses) < 3:
                witnesses.append((vertex, p, q))
    return worst_delta, witnesses[:3], n_tested


def length_ratio_change(x: Any, y: Any) -> tuple[float, list[tuple[int, int]], int]:
    """``max`` change of normalised length ratios ``d_ij / d_max``.

    Ratios are normalised by the largest distance of the respective image, so
    the quantity is scale-free by construction.  Witnesses are index *pairs*
    ``(i, j)``: the ratio attached to the pair ``ij``.
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    rx, wxs = _normalised_ratio_vector(X)
    ry, _ = _normalised_ratio_vector(Y)
    if rx.size == 0:
        return 0.0, [], 0
    delta = np.abs(ry - rx)
    order = np.argsort(delta)[::-1]
    worst = [wxs[k] for k in order[:3] if delta[k] > 0]
    return float(delta.max()), worst, int(rx.size)


def affine_coordinate_change(
    x: Any,
    y: Any,
    frame_indices: Sequence[int],
) -> tuple[float, float, float, np.ndarray]:
    """``(max|Δα|, max|Δβ|, accuracy bound, delta vector)`` for one frame.

    The frame is *moved with the image*: ``(A, B, C)`` are the points at
    ``frame_indices`` in ``X`` and their counterparts in ``Y``.  This is the
    definition the theory gives, and it is what makes the answer exactly zero.
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    idx = tuple(int(i) for i in frame_indices)
    if len(idx) != 3:
        raise ValueError("affine_coordinate_change: frame_indices must have exactly 3 entries")
    fx, fy = AffineFrame.from_indices(X, idx), AffineFrame.from_indices(Y, idx)  # type: ignore[arg-type]
    if not fx.is_valid():
        raise ValueError("affine_coordinate_change: the reference triple is collinear in X")
    if not fy.is_valid():
        return math.inf, math.inf, math.inf, np.array([[math.inf], [math.inf]])
    before = fx.coordinates_batch(X)
    after = fy.coordinates_batch(Y)
    da = np.abs(after["alpha"] - before["alpha"])
    db = np.abs(after["beta"] - before["beta"])
    bound = max(
        math.sqrt(max(float(before["condition_number"] * after["condition_number"]), 1.0))
        * np.finfo(np.float64).eps
        * 64.0,
        FLOAT_TOL,
    )
    return float(da.max()), float(db.max()), float(bound), np.stack([da, db])


# --------------------------------------------------------------------------- #
# relations
# --------------------------------------------------------------------------- #
def collinearity_relation_change(
    x: Any, y: Any, *, tol: float = COLLINEAR_TOL
) -> tuple[frozenset[tuple[int, int, int]], frozenset[tuple[int, int, int]], int]:
    """Compare the *sets of collinear triples* of ``X`` and ``Y``.

    Returns ``(only_in_x, only_in_y, n_relation_in_x)``.  ``n_relation_in_x``
    is what lets the caller distinguish "preserved" from "vacuous".
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    rx = _collinear_triples(X, tol)
    ry = _collinear_triples(Y, tol)
    return frozenset(rx - ry), frozenset(ry - rx), len(rx)


def parallelism_relation_change(
    x: Any, y: Any, *, tol: float = COLLINEAR_TOL
) -> tuple[frozenset[tuple[int, int, int, int]], frozenset[tuple[int, int, int, int]], int]:
    """Compare the *sets of parallel segment pairs* of ``X`` and ``Y``.

    A segment is an unordered pair ``(i, j)``; a witness is therefore a
    4-tuple ``(i, j, k, l)`` meaning "segment ``ij`` is parallel to segment
    ``kl``".  Self-comparisons and shared-endpoint pairs are skipped (two
    segments sharing a point are parallel only in degenerate configurations,
    which would make the relation trivially vacuous).
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    for mat, label in ((X, "X"), (Y, "Y")):
        if mat.shape[0] > 24:
            raise ValueError(
                f"parallelism_relation_change: {mat.shape[0]} points in {label} would "
                "create O(n^4) segment pairs; pass at most 24 points"
            )
    rx = _parallel_segment_pairs(X, tol)
    ry = _parallel_segment_pairs(Y, tol)
    return frozenset(rx - ry), frozenset(ry - rx), len(rx)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
_LABELS: dict[str, tuple[str, str, str]] = {
    "absolute_coordinates": ("绝对坐标", "абсолютные координаты", "max_i ‖x_i − x'_i‖"),
    "distances": ("距离", "расстояния", "max_{i<j} |d_ij − d'_ij| / scale"),
    "angles": ("角度", "углы", "max |θ − θ'| (rad)"),
    "length_ratios": ("长度比例", "отношения длин", "max |r − r'|, r = d_ij / d_max"),
    "collinearity": ("共线性", "коллинеарность", "共线三元组集合的对称差"),
    "parallelism": ("平行性", "параллельность", "平行线段对集合的对称差"),
    "affine_coordinates": ("仿射坐标", "аффинные координаты", "max |Δα|, |Δβ|"),
}


def invariant_report(
    x: Any,
    y: Any,
    *,
    kind: str | None = None,
    invariant_group: str | None = None,
    frame_indices: Sequence[int] | None = None,
    rel_tol: float = MATCH_TOL,
    angle_tol: float = 1e-9,
    count_tol: float = 2.0,
) -> InvariantReport:
    """Measure the full invariant table for the pair ``(X, Y)``.

    Parameters
    ----------
    x, y:
        Images with the *same number of points*; point ``i`` of ``Y`` is
        assumed to be the image of point ``i`` of ``X``.
    kind, invariant_group:
        Transform classification.  ``kind=None`` triggers a best-effort
        inference from the pair itself (see :func:`_infer_kind`); pass it
        explicitly whenever the transform is known — inference is a
        convenience for the GUI's manual edit mode, never a substitute for the
        real label in an experiment.
    frame_indices:
        Which three points to use for the internal-coordinate row.  When
        ``None`` a well-conditioned frame is chosen automatically.  When the
        image admits no frame the row is reported as ``"vacuous"`` instead of
        crashing — an important distinction from "preserved".
    rel_tol:
        Relative tolerance for metric comparisons (distance, ratios).
    angle_tol:
        Absolute tolerance in radians for angles.
    count_tol:
        A relation verdict is ``"violated"`` when the symmetric difference has
        at least this many witnesses.  A single flipping triple right at the
        tolerance boundary is floating-point noise, not a broken invariant; set
        to ``1.0`` for a strict reading.
    """
    X = as_point_matrix(x, name="x")
    Y = as_point_matrix(y, name="y")
    _require_same_size(X, Y)
    scale = max(float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))), 1.0)

    if kind is None or invariant_group is None:
        inferred_kind, inferred_group = _infer_kind(X, Y)
        kind = kind or inferred_kind
        invariant_group = invariant_group or inferred_group

    entries: list[InvariantEntry] = []

    # 1. absolute coordinates -------------------------------------------------
    delta, _, witness = absolute_coordinate_change(X, Y)
    tol_abs = rel_tol * scale
    entries.append(
        _entry("absolute_coordinates",
               float(np.abs(X).max()), float(np.abs(Y).max()),
               delta, tol_abs,
               "violated" if delta > tol_abs else "preserved", 1, witness,
               note="只有在恒等变换下保持")
    )

    # 2. distances -----------------------------------------------------------
    delta, witnesses, dist_scale = distance_change(X, Y)
    tol_dist = rel_tol * dist_scale
    entries.append(
        _entry("distances", float(_pairwise_distance_matrix(X).max()) if X.shape[0] > 1 else 0.0,
               float(_pairwise_distance_matrix(Y).max()) if Y.shape[0] > 1 else 0.0,
               delta, tol_dist,
               "violated" if delta > tol_dist else "preserved",
               len(witnesses), _fmt_pairs(witnesses), note="平移/旋转/反射下保持；缩放改变")
    )

    # 3. angles --------------------------------------------------------------
    try:
        delta, witnesses, n_tested = angle_change(X, Y)
        if n_tested == 0:
            verdict: Verdict = "vacuous"
        else:
            verdict = "violated" if delta > angle_tol else "preserved"
        entries.append(
            _entry("angles", math.nan, math.nan, delta, angle_tol, verdict,
                   n_tested, _fmt_triples(witnesses),
                   note="等距/相似下保持；仿射（剪切）下改变")
        )
    except ValueError as exc:
        entries.append(_vacuous("angles", str(exc)))

    # 4. length ratios -------------------------------------------------------
    delta, witnesses, n_tested = length_ratio_change(X, Y)
    if n_tested == 0:
        entries.append(_vacuous("length_ratios", "点数不足，无法构成长度比"))
    else:
        entries.append(
            _entry("length_ratios", math.nan, math.nan, delta, rel_tol,
                   "violated" if delta > rel_tol else "preserved",
                   n_tested, _fmt_pairs(witnesses, "比值对"),
                   note="相似下保持；非等比缩放下改变")
        )

    # 5. collinearity --------------------------------------------------------
    only_x, only_y, n_rel = collinearity_relation_change(X, Y)
    n_wit = len(only_x) + len(only_y)
    verdict = "vacuous" if n_rel == 0 and not only_y else (
        "violated" if n_wit >= count_tol else "preserved"
    )
    entries.append(
        _entry("collinearity", float(n_rel), float(len(only_x) + len(only_y)),
               float(n_wit), 0.0, verdict, n_wit,
               _fmt_triples(sorted(only_x | only_y)[:3]),
               note="仿射变换下保持（关系，而非角度数值）")
    )

    # 6. parallelism ---------------------------------------------------------
    try:
        only_x, only_y, n_rel = parallelism_relation_change(X, Y)
        n_wit = len(only_x) + len(only_y)
        verdict = "vacuous" if n_rel == 0 and not only_y else (
            "violated" if n_wit >= count_tol else "preserved"
        )
        entries.append(
            _entry("parallelism", float(n_rel), float(len(only_x) + len(only_y)),
                   float(n_wit), 0.0, verdict, n_wit,
                   _fmt_quads(sorted(only_x | only_y)[:3]),
                   note="仿射变换下保持平行关系（两条线段之间的夹角会变）")
        )
    except ValueError as exc:
        entries.append(_vacuous("parallelism", str(exc)))

    # 7. affine coordinates --------------------------------------------------
    try:
        idx = tuple(frame_indices) if frame_indices is not None else None
        if idx is None:
            fx = AffineFrame.auto(X)
            idx = fx.indices
        assert idx is not None
        da, db, bound, _ = affine_coordinate_change(X, Y, idx)
        delta = max(da, db)
        tol_ab = max(bound, FLOAT_TOL)
        entries.append(
            _entry("affine_coordinates", math.nan, math.nan, delta, tol_ab,
                   "violated" if delta > tol_ab else "preserved", 1,
                   f"frame={idx}", note="一切可逆仿射变换下保持——项目核心结论")
        )
    except ValueError as exc:
        entries.append(_vacuous("affine_coordinates", str(exc)))

    return InvariantReport(
        transform_kind=str(kind),
        invariant_group=str(invariant_group),
        n_points=int(X.shape[0]),
        entries=tuple(entries),
    )


def report_for_transform(
    x: Any,
    transform: AffineTransform,
    *,
    frame_indices: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, InvariantReport]:
    """Apply ``transform`` to ``x`` in the *point-index preserving* way and report.

    Returns ``(Y, report)``.  The transform classification is taken from the
    transform object itself, so the report's verdicts can be checked against
    the theoretical group rather than guessed from the numbers.
    """
    X = as_point_matrix(x, name="x")
    Y = transform.apply(X)
    report = invariant_report(
        X,
        Y,
        kind=transform.classify(),
        invariant_group=transform.invariant_group(),
        frame_indices=frame_indices,
        **kwargs,
    )
    return Y, report


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _entry(
    key: str,
    before: float,
    after: float,
    delta: float,
    tolerance: float,
    verdict: Verdict,
    witness_count: int,
    witness: str | None,
    *,
    note: str | None = None,
) -> InvariantEntry:
    label_zh, label_ru, measure = _LABELS[key]
    return InvariantEntry(
        key=key,
        label_zh=label_zh,
        label_ru=label_ru,
        measure=measure,
        value_before=before,
        value_after=after,
        delta=delta,
        tolerance=tolerance,
        verdict=verdict,
        witness_count=witness_count,
        witness=witness,
        note=note,
    )


def _vacuous(key: str, reason: str) -> InvariantEntry:
    label_zh, label_ru, measure = _LABELS[key]
    return InvariantEntry(
        key=key,
        label_zh=label_zh,
        label_ru=label_ru,
        measure=measure,
        value_before=math.nan,
        value_after=math.nan,
        delta=math.nan,
        tolerance=math.nan,
        verdict="vacuous",
        witness_count=0,
        witness=None,
        note=reason,
    )


def _require_same_size(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"invariant comparison needs equally many points, got {x.shape[0]} and {y.shape[0]}"
        )


def _pairwise_distance_matrix(pts: np.ndarray) -> np.ndarray:
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _angle_at(vertex: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    r"""Angle ``angle(p, vertex, q)`` in ``[0, pi]``, computed stably.

    Why not ``arccos(dot / (|u||v|))``
    ----------------------------------
    ``arccos`` is numerically catastrophic exactly where point sets are most
    interesting: when the two vectors are nearly parallel or nearly
    anti-parallel (i.e. nearly collinear triples), ``cos`` is ``1 - O(eps)`` and
    the derivative ``-1/sqrt(1-cos^2)`` blows up, so a 1e-16 error in ``cos``
    becomes a ~1e-8 error in the angle.  A naive implementation would then report
    "angles are not preserved" for a pure *translation*, purely because the
    translated points are no longer axis-aligned and the collinear triples moved
    off the exact ``cos = 1``.

    ``atan2(|u x v|, u . v)`` has no such problem: near parallel vectors the
    first argument is ``O(eps)`` while the second is ``O(1)``, and ``atan2`` is
    accurate for small angles by construction.  It is also well defined for the
    degenerate (zero-length) vector case, which we map to NaN so callers can
    skip it.
    """
    u = p - vertex
    v = q - vertex
    nu = float(math.hypot(u[0], u[1]))
    nv = float(math.hypot(v[0], v[1]))
    if nu <= 0 or nv <= 0:
        return math.nan
    cross = abs(float(u[0] * v[1] - u[1] * v[0]))
    dot = float(u[0] * v[0] + u[1] * v[1])
    return math.atan2(cross, dot)


def _normalised_ratio_vector(
    pts: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """All ``d_ij / d_max`` ratios as a vector, plus their index witnesses."""
    n = pts.shape[0]
    d = _pairwise_distance_matrix(pts)
    iu = np.triu_indices(n, k=1)
    values = d[iu]
    if values.size == 0:
        return np.zeros(0), []
    scale = float(values.max())
    if scale <= 0:
        return np.zeros(0), []
    pairs = list(zip(iu[0].tolist(), iu[1].tolist()))
    return values / scale, [(int(i), int(j)) for i, j in pairs]


def _collinear_triples(pts: np.ndarray, tol: float) -> frozenset[tuple[int, int, int]]:
    """Index triples standing in the collinearity *relation*.

    Uses :func:`~kozlov_lab.geometry.affine_frame.triangle_flatness`, which is
    symmetric in the three points.  Thresholding the vertex sine
    ``|sin(angle BAC)|`` would make the relation depend on the index order — the
    same triple could count as collinear or not depending on which point has the
    smallest index.
    """
    n = pts.shape[0]
    found: set[tuple[int, int, int]] = set()
    for (i, j, k) in itertools.combinations(range(n), 3):
        if triangle_flatness(pts[i], pts[j], pts[k]) <= tol:
            found.add((i, j, k))
    return frozenset(found)


def _parallel_segment_pairs(
    pts: np.ndarray, tol: float
) -> frozenset[tuple[int, int, int, int]]:
    """Segment pairs ``(i, j, k, l)`` whose direction vectors are parallel.

    Segments sharing an endpoint are skipped: for them "parallel" would mean
    "collinear", which is tested by the collinearity row instead.
    """
    n = pts.shape[0]
    segments: list[tuple[tuple[int, int], np.ndarray, float]] = []
    for i, j in itertools.combinations(range(n), 2):
        vec = pts[j] - pts[i]
        norm = float(math.hypot(vec[0], vec[1]))
        if norm > 0:
            segments.append(((i, j), vec, norm))

    found: set[tuple[int, int, int, int]] = set()
    for m in range(len(segments)):
        (p, vp, np_) = segments[m]
        for l in range(m + 1, len(segments)):
            (q, vq, nq) = segments[l]
            if set(p) & set(q):
                continue
            cross = abs(float(vp[0] * vq[1] - vp[1] * vq[0]))
            if cross / (np_ * nq) <= tol:
                found.add((p[0], p[1], q[0], q[1]))
    return frozenset(found)


def _infer_kind(x: np.ndarray, y: np.ndarray) -> tuple[str, str]:
    """Best-effort transform classification from the point pair alone.

    Used ONLY when the caller does not know the transform (interactive editing
    in the GUI).  It tries to fit an affine map from three control points and
    classifies it; a poor fit means "not an affine image of X at all".

    Deliberately never returns ``"identity"``/``"similarity"`` by guessing:
    an inference error here would silently corrupt the invariant table, so the
    caller is told which path was used via ``transform_kind``.
    """
    n = x.shape[0]
    if n < 3:
        return "unknown", "unknown"
    for (i, j, k) in itertools.combinations(range(min(n, 8)), 3):
        if collinearity_sine(x[i], x[j], x[k]) <= COLLINEAR_TOL:
            continue
        try:
            t = AffineTransform.from_control_points(x[[i, j, k]], y[[i, j, k]])
        except ValueError:
            continue
        if not t.is_invertible:
            continue
        residual = float(np.abs(t.apply(x) - y).max())
        if residual <= MATCH_TOL * max(1.0, float(np.abs(x).max())):
            return t.classify(), t.invariant_group()
    return "unknown", "unknown"


def _fmt_pairs(pairs: Iterable[tuple[int, int]], label: str = "距离对") -> str | None:
    items = [f"({i},{j})" for i, j in pairs]
    return f"{label} " + ", ".join(items) if items else None


def _fmt_triples(triples: Iterable[tuple[int, int, int]]) -> str | None:
    items = [f"({i},{j},{k})" for i, j, k in triples]
    return "三元组 " + ", ".join(items) if items else None


def _fmt_quads(quads: Iterable[tuple[int, int, int, int]]) -> str | None:
    items = [f"({i},{j})‖({k},{l})" for i, j, k, l in quads]
    return "线段对 " + ", ".join(items) if items else None
