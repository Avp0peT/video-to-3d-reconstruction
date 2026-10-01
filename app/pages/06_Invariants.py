"""M6 — Invariant Explorer (README section 9).

The table in README section 9 is a *claim* about which properties survive which
transformation family.  This page turns it into a measurement and shows four things
side by side, as the spec requires:

1. the theoretical prediction (from ``THEORY_EXPECTATION``, the same table the tests
   use — one source of truth);
2. the measured result, with the deviation and its tolerance;
3. the numerical error, in a form where "small" means small *relative to the stated
   tolerance*;
4. a control experiment against an unrelated point set of the same size, which must
   **break every property** — without it, a broken measurement would still be green.

If theory and measurement disagree, the page shows the literal banner
``NUMERICAL / IMPLEMENTATION DISCREPANCY`` and never hides it.
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
from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.geometry import (  # noqa: E402
    FAMILY_LABELS_ZH,
    THEORY_EXPECTATION,
    AffineTransform,
    invariant_report,
    report_for_transform,
)

ui.setup_page("M6 不变量实验中心")
ui.title(
    "M6 — 不变量实验中心",
    "Invariant Explorer ｜ 每一项都必须能通过实际数值实验验证，而不仅仅展示文字（README 第 9 节）",
)

image = ui.source_selector(key="m6")
x = image.as_array()
centre = x.mean(axis=0)
span = float(np.abs(x).max()) or 1.0
reflection = np.array([[-0.6, 0.8], [0.8, 0.6]])

# --------------------------------------------------------------------------- #
# family selection
# --------------------------------------------------------------------------- #
st.sidebar.header("变换族")
family = st.sidebar.selectbox(
    "族",
    ["translation", "rotation", "similarity", "affine"],
    format_func=lambda name: f"{FAMILY_LABELS_ZH.get(name, name)}（{name}）",
)
advanced = st.sidebar.checkbox("显示子类型（反射 / 非等比缩放 / 剪切 / 一般仿射）", value=False)

CANDIDATES: dict[str, AffineTransform] = {
    "translation": AffineTransform.translation_of(0.37 * span, -0.21 * span),
    "rotation": AffineTransform.rotation(37.0, center=centre),
    "similarity": AffineTransform.similarity(2.5, 41.0, center=centre),
    "affine": AffineTransform(np.array([[1.3, 0.45], [-0.2, 0.9]]), np.array([-1.0, 2.0])),
    # sub-types, shown when `advanced` is on
    "reflection": AffineTransform(reflection, centre - reflection @ centre),
    "scale_nonuniform": AffineTransform.scaling(2.0, 0.5, center=centre),
    "shear": AffineTransform.shear(0.8, -0.2),
}

LABELS = {
    "reflection": "反射（等距）",
    "scale_nonuniform": "非等比缩放",
    "shear": "剪切",
}

selected: list[str] = [family]
if advanced:
    selected = st.sidebar.multiselect(
        "对比多个族", list(CANDIDATES), default=[family], format_func=lambda n: LABELS.get(n, FAMILY_LABELS_ZH.get(n, n))
    )
    if not selected:
        selected = [family]

# --------------------------------------------------------------------------- #
# theory table overview
# --------------------------------------------------------------------------- #
st.subheader("① 理论预测")
symbol = {"preserved": "✓", "violated": "×", "preserved_or_vacuous": "✓/–", "unknown": "?"}
st.dataframe(
    {
        "性质": list(THEORY_EXPECTATION["translation"]),
        **{
            FAMILY_LABELS_ZH[kind]: [symbol[value] for value in THEORY_EXPECTATION[kind].values()]
            for kind in ("translation", "rotation", "similarity", "affine")
        },
    },
    width="stretch",
    hide_index=True,
)
st.caption(
    "✓ 保持 ｜ × 破坏 ｜ ✓/– 保持，但若图像中没有该关系的见证则判为 vacuous（无信息），"
    "**不算保持** ｜ ? 无理论预测。这张表就是 `THEORY_EXPECTATION`，与测试共用同一份数据。"
)

# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("②–③ 实测结果与数值误差")

all_ok = True
for name in selected:
    transform = CANDIDATES[name]
    y, report = report_for_transform(x, transform)
    check = report.theory_check(transform)

    st.markdown(f"#### {FAMILY_LABELS_ZH.get(name, name)} — `{name}`（{report.transform_kind}）")
    ui.side_by_side(
        x,
        y,
        left_title="X",
        right_title="T(X)",
        height=300,
    )
    ui.discrepancy_banner(check)
    all_ok = all_ok and not check.has_discrepancy
    ui.verdict_table(check.as_rows(), theory_column=True)

    with st.expander("见证（证明 × 不是舍入噪声）"):
        witnesses = {
            entry.label_zh: (entry.witness, entry.delta)
            for entry in report.entries
            if entry.witness and entry.verdict == "violated"
        }
        if not witnesses:
            st.caption("该族下没有被破坏的性质，因此没有破坏见证。")
        else:
            st.dataframe(
                {
                    "性质": list(witnesses),
                    "见证": [value[0] for value in witnesses.values()],
                    "幅度": [f"{value[1]:.3e}" for value in witnesses.values()],
                },
                width="stretch",
                hide_index=True,
            )

    with st.expander("为何容差是这些数值（而不是一个固定的 1e-9）"):
        st.markdown(
            "容差必须与该性质的量纲／尺度匹配：\n\n"
            "* **距离**：相对图像尺度（`rel_tol × max(d, 1)`）；\n"
            "* **角度**：弧度制绝对容差（角度本身无量纲）；\n"
            "* **长度比例**：已按 $d_{\\max}$ 归一化，因此可用绝对容差；\n"
            "* **仿射坐标**：由标架条件数决定的可达精度 $\\mathrm{cond}\\,M\\cdot\\varepsilon$；\n"
            "* **关系（共线性 / 平行性）**：判据是集合的对称差是否为空，容差为判据阈值本身。\n\n"
            "把所有性质都塞进一个全局 `1e-9` 会在小尺度上过松、在大尺度上过严。"
        )

# --------------------------------------------------------------------------- #
# control experiment
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("④ 对照实验：无关图像")

ui.explain(
    "把该图像与一个**同基数的随机点集**比较。凡是对仿射变换成立的性质，在此必须全部被破坏。",
    "Сравнение с несвязанным множеством той же мощности: все свойства, справедливые "
    "для аффинных преобразований, должны быть нарушены.",
)

control_seed = st.number_input("对照随机种子", 0, 10**9, 20261001)
unrelated = shapes.random_points(x.shape[0], seed=int(control_seed)).as_array()
control = invariant_report(x, unrelated, kind="unknown", invariant_group="unknown")
control_check = control.theory_check()

ui.side_by_side(x, unrelated, left_title="X", right_title="无关点集 Z", right_colour=ui.NEUTRAL, height=300)

st.dataframe(
    {
        "性质": [entry.label_zh for entry in control.entries],
        "对照组结论": [
            {"preserved": "✓ 保持", "violated": "× 破坏", "vacuous": "– 无见证"}[entry.verdict]
            for entry in control.entries
        ],
        "max|Δ|": [f"{entry.delta:.3e}" for entry in control.entries],
        "见证": [entry.witness or "—" for entry in control.entries],
    },
    width="stretch",
    hide_index=True,
)

violated_count = sum(1 for entry in control.entries if entry.verdict == "violated")
if violated_count == len(control.entries):
    st.success(
        f"对照组成立：{violated_count}/{len(control.entries)} 项全部被破坏。"
        "这一组的存在使得上面那些「保持」具有信息量——否则一个恒返回「保持」的实现也能全绿。"
    )
else:
    st.error(
        f"对照组只破坏了 {violated_count}/{len(control.entries)} 项。"
        "这意味着对应性质的测量可能没有真正在测量，需要人工核查。"
    )
st.info(control_check.headline())
st.caption(
    "对照组的变换类型是 `unknown`，因此没有理论预测可言——"
    "页面明确写出「无可比对项」，而不是让它看起来像一次通过。"
)

# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
st.divider()
if all_ok:
    st.success("理论预测与实测在所选各族上完全一致（一致性由 THEORY_EXPECTATION 逐项判定）。")
else:
    st.error("存在 NUMERICAL / IMPLEMENTATION DISCREPANCY，见上方横幅。")

with st.expander("为什么又要有「vacuous」这一态？"):
    st.markdown(
        "关系型性质（共线性、平行性）的判据是「两个集合的对称差为空」。当两边集合**都是空的**时"
        "这个判据恒为真：\n\n"
        "* 随机点集里几乎没有共线三元组，于是「共线性保持」永远成立；\n"
        "* 一个把关系集合恒返回空的损坏实现，也能全绿。\n\n"
        "因此本页面把结论分成三态：`preserved`（保持且原本有见证）、"
        "`violated`（有失败见证）、`vacuous`（**无见证，结论不含信息**）。\n\n"
        "想现场看到 `vacuous`：把图像来源切到**数据集 C / D / E** 或内置图形 `circle_12`，"
        "看「平行性」那一行。"
    )

ui.footer()
