"""M2 — Transformation Laboratory (README section 5).

Four families, one unified representation.  The important design choice on display
here: the page never claims "preserved / not preserved" as a label.  It drags
parameters, measures the deviation, and compares it against a stated tolerance —
so the invariant table is *measured* rather than asserted.
"""

from __future__ import annotations

import pathlib
import sys

_APP = pathlib.Path(__file__).resolve().parent
_ROOT = _APP.parent
for _path in (_APP, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import _shared as ui  # noqa: E402
from kozlov_lab.config import tolerance_for_condition  # noqa: E402
from kozlov_lab.geometry import (  # noqa: E402
    AffineFrame,
    AffineTransform,
    report_for_transform,
)

ui.setup_page("M2 变换实验室")
ui.title(
    "M2 — 几何变换实验室",
    "平移 ⊂ 等距 ⊂ 相似 ⊂ 仿射；参数可调，效果实时可见",
)

image = ui.source_selector(key="m2")
x = image.as_array()
centre = x.mean(axis=0)
span = float(np.abs(x).max()) or 1.0

# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
st.sidebar.header("变换参数")
family = st.sidebar.selectbox(
    "变换族",
    ["translation", "rotation", "reflection", "similarity", "scaling", "shear", "general affine"],
)

translation_x = st.sidebar.slider("translation x", -3.0 * span, 3.0 * span, 0.0, 0.01 * span)
translation_y = st.sidebar.slider("translation y", -3.0 * span, 3.0 * span, 0.0, 0.01 * span)
rotation_deg = st.sidebar.slider("rotation angle (deg)", -180.0, 180.0, 0.0, 1.0)
uniform_scale = st.sidebar.slider("uniform scale", 0.1, 4.0, 1.0, 0.01)
scale_x = st.sidebar.slider("x scale", 0.1, 4.0, 1.0, 0.01)
scale_y = st.sidebar.slider("y scale", 0.1, 4.0, 1.0, 0.01)
shear_x = st.sidebar.slider("shear x", -1.5, 1.5, 0.0, 0.01)
shear_y = st.sidebar.slider("shear y", -1.5, 1.5, 0.0, 0.01)
about_centroid = st.sidebar.checkbox("围绕质心（否则围绕原点）", value=True)


def build() -> AffineTransform:
    """Assemble the transform from the sliders, honouring the chosen family."""
    pivot = centre if about_centroid else None
    if family == "translation":
        return AffineTransform.translation_of(translation_x, translation_y)
    if family == "rotation":
        return AffineTransform.rotation(rotation_deg, center=pivot) @ AffineTransform.translation_of(
            translation_x, translation_y
        )
    if family == "reflection":
        angle = np.radians(rotation_deg)
        matrix = np.array(
            [[np.cos(2 * angle), np.sin(2 * angle)], [np.sin(2 * angle), -np.cos(2 * angle)]]
        )
        return AffineTransform(matrix, np.zeros(2)) @ AffineTransform.translation_of(
            translation_x, translation_y
        )
    if family == "similarity":
        return AffineTransform.similarity(
            uniform_scale, rotation_deg, center=pivot
        ) @ AffineTransform.translation_of(translation_x, translation_y)
    if family == "scaling":
        return AffineTransform.scaling(scale_x, scale_y, center=pivot) @ AffineTransform.translation_of(
            translation_x, translation_y
        )
    if family == "shear":
        return AffineTransform.shear(shear_x, shear_y) @ AffineTransform.translation_of(
            translation_x, translation_y
        )
    matrix = np.array(
        [[scale_x * np.cos(np.radians(rotation_deg)), shear_x], [shear_y, scale_y]]
    )
    return AffineTransform(matrix, np.array([translation_x, translation_y]))


transform = build()
y = transform.apply(x)

# --------------------------------------------------------------------------- #
# display
# --------------------------------------------------------------------------- #
ui.side_by_side(x, y, left_title="原始图像 X", right_title="变换后 T(X)")

ui.explain(
    "同一个视觉对象在变换前后的**外部坐标**完全不同；下面的表格测量的是"
    "「哪些性质改变了、改变了多少」。",
    "Внешние координаты объекта до и после преобразования полностью различны; "
    "таблица ниже измеряет, какие свойства изменились и насколько.",
)

# --------------------------------------------------------------------------- #
# transform diagnostics
# --------------------------------------------------------------------------- #
st.subheader("变换本身的诊断")
parameters = transform.parameters()
st.dataframe(
    {
        "量": ["矩阵 A", "平移 t", "det A", "cond(A)", "类型", "不变量族"],
        "值": [
            f"[[{parameters['a11']:+.4f}, {parameters['a12']:+.4f}], "
            f"[{parameters['a21']:+.4f}, {parameters['a22']:+.4f}]]",
            f"({parameters['tx']:+.4f}, {parameters['ty']:+.4f})",
            f"{transform.det:+.6f}",
            f"{transform.condition_number:.4f}",
            transform.classify(),
            transform.invariant_group(),
        ],
    },
    width="stretch",
    hide_index=True,
)
ui.formula(r"T(x) = A\,x + t,\qquad \det A \neq 0")
if not transform.is_invertible:
    ui.note("det A ≈ 0：该变换不可逆，不是合法的图像变换（README 第 5 节要求 det A ≠ 0）。", "error")

# --------------------------------------------------------------------------- #
# flatness / conditioning of an automatically chosen frame, before and after
# --------------------------------------------------------------------------- #
st.subheader("标架随变换一起变化：flatness 与 cond 的前后对比")
ui.explain(
    "仿射变换会改变标架的**形状**，因而改变它的平坦度与条件数——"
    "这决定了内部坐标（M3 页面）能达到的精度。",
    "Аффинное преобразование меняет форму базисного треугольника и тем самым его "
    "плоскостность и число обусловленности.",
)

try:
    frame_before = AffineFrame.auto(x)
    frame_after = frame_before.transformed(transform)
    st.dataframe(
        {
            "量": [
                "标架点编号",
                "flatness = 2·Area / max_side²",
                "cond([B−A, C−A])",
                "可达精度（cond·ε·64）",
                "标架是否合法",
            ],
            "变换前": [
                f"{frame_before.indices}",
                f"{frame_before.collinearity_sine:.6f}",
                f"{frame_before.condition_number:.6f}",
                f"{tolerance_for_condition(frame_before.condition_number):.3e}",
                "✓" if frame_before.is_valid() else "✗",
            ],
            "变换后": [
                f"{frame_after.indices}",
                f"{frame_after.collinearity_sine:.6f}",
                f"{frame_after.condition_number:.6f}",
                f"{tolerance_for_condition(frame_after.condition_number):.3e}",
                "✓" if frame_after.is_valid() else "✗",
            ],
            "比值（后 / 前）": [
                "—",
                f"{frame_after.collinearity_sine / max(frame_before.collinearity_sine, 1e-300):.4f}",
                f"{frame_after.condition_number / max(frame_before.condition_number, 1e-300):.4f}",
                "—",
                "—",
            ],
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "flatness 是**置换不变**的共线性判据（$2\\,\\mathrm{Area}/\\max(\\text{边})^2$，"
        "即最短高 ÷ 最长边），与顶点选择无关；它同时也是标架的合法性判据。"
        "若变换后标架不再合法，M3 页面会明确拒绝而不是给出一个看似合理的坐标系。"
    )
except ValueError as exc:
    st.warning(f"该图像无法自动选出标架：{exc}")

# --------------------------------------------------------------------------- #
# measured invariants
# --------------------------------------------------------------------------- #
st.subheader("实测不变量")

_, report = report_for_transform(x, transform)
check = report.theory_check(transform)

ui.discrepancy_banner(check)

ui.verdict_table(check.as_rows(), theory_column=True)
st.caption(
    "「理论预测」来自 `THEORY_EXPECTATION`（单一事实来源，与测试共用）；"
    "「一致」列若出现 ✗，页面会给出 NUMERICAL / IMPLEMENTATION DISCREPANCY 横幅，"
    "不做任何自动隐藏。"
)

st.subheader("真实测量值（不是「保持 / 不保持」）")
st.dataframe(
    {
        "性质": [entry.label_zh for entry in report.entries],
        "度量": [entry.measure for entry in report.entries],
        "max|Δ|": [f"{entry.delta:.3e}" for entry in report.entries],
        "容差": [f"{entry.tolerance:.3e}" for entry in report.entries],
        "见证数": [entry.witness_count for entry in report.entries],
        "见证": [entry.witness or "—" for entry in report.entries],
    },
    width="stretch",
    hide_index=True,
)

# --------------------------------------------------------------------------- #
# the canonical demonstration
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("特别展示：正方形 → 仿射 → 平行四边形")
ui.explain(
    "改变：绝对坐标、距离、角度。保持：共线性、平行性、仿射组合系数。",
    "Меняются: абсолютные координаты, расстояния, углы. "
    "Сохраняются: коллинеарность, параллельность, коэффициенты аффинной комбинации.",
)

columns = st.columns(3)
with columns[0]:
    st.plotly_chart(
        ui.point_figure(x, title="原始 X", x_range=None),
        width="stretch",
    )
with columns[1]:
    st.plotly_chart(
        ui.point_figure(y, title="T(X)", colour=ui.TARGET),
        width="stretch",
    )
with columns[2]:
    ratio = {
        "共线性": report.get("collinearity"),
        "平行性": report.get("parallelism"),
        "角度": report.get("angles"),
        "距离": report.get("distances"),
    }
    st.markdown("**关系 vs 连续量**")
    st.dataframe(
        {
            "性质": list(ratio),
            "类型": ["关系", "关系", "连续量", "连续量"],
            "结论": [
                {"preserved": "✓ 保持", "violated": "× 破坏", "vacuous": "– 无见证"}[
                    entry.verdict
                ]
                for entry in ratio.values()
            ],
            "max|Δ|": [f"{entry.delta:.3e}" for entry in ratio.values()],
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "共线性与平行性是**关系**：仿射变换不保持线段夹角，但保持「平行」这个事实。"
        "把平行性当成夹角来测会得到与理论相反的结论——那是测法错，不是数学错。"
    )

ui.why(
    "为什么分类只看线性部分？",
    "因为决定不变量的是线性部分 $A$，不是平移 $t$。绕很远的中心旋转会产生很大的平移向量，"
    "但它仍然是等距变换；反之，一个「看起来很小」的剪切可以彻底破坏角度。",
    r"A=\begin{pmatrix}s\cos\theta & -s\sin\theta\\ s\sin\theta & s\cos\theta\end{pmatrix}"
    r"\ \Longrightarrow\ \text{similarity}",
)

ui.footer()
