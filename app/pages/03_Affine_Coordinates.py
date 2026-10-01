"""M3 — Internal Affine Coordinates (README section 6).

The most important page of the project.  It performs the seven-step walkthrough and
shows, for every point:

* external coordinates before and after;
* internal coordinates before and after;
* the observed deviation ``Δα``, ``Δβ``;
* the **predicted attainable precision**.

The requested presentation rule is explicit: no bare PASS / FAIL.  A number without
its error budget is not a result, so the page always shows both.
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
from kozlov_lab.config import MATCH_TOL  # noqa: E402
from kozlov_lab.geometry import AffineTransform  # noqa: E402
from kozlov_lab.precision import frame_scale_of, precision_of_invariance  # noqa: E402

ui.setup_page("M3 内部仿射坐标")
ui.title(
    "M3 — 内部仿射坐标",
    "项目核心：P − A = α(B − A) + β(C − A)，且 (α, β) 在一切可逆仿射变换下不变",
)

image = ui.source_selector(key="m3")
x = image.as_array()

# --------------------------------------------------------------------------- #
# step 1-3: choose the frame
# --------------------------------------------------------------------------- #
st.sidebar.header("基准点 A, B, C")
frame = ui.frame_selector(image, key="m3")

if frame is None:
    st.error(
        "该图像无法建立仿射坐标系（点数不足 3，或所选三点共线）。"
        "这是**明确拒绝**，不是返回一个看似合理的坐标系。"
    )
    ui.why(
        "为什么要求三个点不共线？",
        "因为 $B-A$ 与 $C-A$ 必须线性无关，否则不能形成二维仿射坐标系："
        "垂直于该直线的分量无法表示，分解不存在或不唯一。",
        r"\det[\,B-A\quad C-A\,]\neq 0",
    )
    ui.footer()
    st.stop()

st.subheader("①–③ 标架与外部 / 内部坐标")
columns = st.columns([3, 2])
with columns[0]:
    st.plotly_chart(
        ui.point_figure(x, title="图像与标架", frame=frame),
        width="stretch",
    )
with columns[1]:
    st.dataframe(
        {
            "项": ["A = x[i]", "B = x[j]", "C = x[k]", "flatness = h_min/max_side", "cond([B−A, C−A])"],
            "值": [
                f"#{frame.indices[0]} = {np.array2string(frame.a, precision=4)}",
                f"#{frame.indices[1]} = {np.array2string(frame.b, precision=4)}",
                f"#{frame.indices[2]} = {np.array2string(frame.c, precision=4)}",
                f"{frame.collinearity_sine:.6f}",
                f"{frame.condition_number:.6f}",
            ],
        },
        width="stretch",
        hide_index=True,
    )

ui.formula(r"P - A = \alpha\,(B - A) + \beta\,(C - A)")
ui.formula(
    r"\begin{pmatrix}\alpha\\ \beta\end{pmatrix}"
    r"= [\,B-A\quad C-A\,]^{-1}(P-A)"
)
ui.explain(
    "点 P 的位置可以完全由它相对于 A、B、C 的关系描述。",
    "Положение точки P задаётся её внутренними аффинными координатами "
    "относительно A, B и C.",
)

before = frame.coordinates_batch(x)

# --------------------------------------------------------------------------- #
# step 4-6: transform and recompute
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("④–⑥ 施加仿射变换（标架随图像一起变换）")

st.sidebar.header("变换")
mode = st.sidebar.radio("方式", ["随机仿射变换", "手动参数"])
span = float(np.abs(x).max()) or 1.0

if mode == "随机仿射变换":
    seed = st.sidebar.number_input("随机种子", 0, 10**9, 20261001)
    budget = st.sidebar.slider("条件数预算 max cond(A)", 1.0, 200.0, 20.0, 1.0)
    transform = AffineTransform.random_invertible(int(seed), max_condition=budget)
    st.sidebar.caption(f"cond(A) = {transform.condition_number:.4f}｜类型 {transform.classify()}")
else:
    tx = st.sidebar.slider("tx", -3.0 * span, 3.0 * span, 0.4 * span)
    ty = st.sidebar.slider("ty", -3.0 * span, 3.0 * span, -0.2 * span)
    rotation = st.sidebar.slider("rotation (deg)", -180.0, 180.0, 25.0)
    sx = st.sidebar.slider("sx", 0.2, 3.0, 1.4, 0.01)
    sy = st.sidebar.slider("sy", 0.2, 3.0, 0.8, 0.01)
    kx = st.sidebar.slider("shear kx", -1.0, 1.0, 0.35, 0.01)
    matrix = np.array([[sx * np.cos(np.radians(rotation)), kx], [0.0, sy]])
    transform = AffineTransform(matrix, np.array([tx, ty]))

y = transform.apply(x)
moved_frame = frame.transformed(transform)
after = moved_frame.coordinates_batch(y)

ui.side_by_side(
    x,
    y,
    left_title="变换前 X",
    right_title="变换后 T(X)",
    left_frame=frame,
    right_frame=moved_frame,
)

if not transform.is_invertible:
    ui.note("该变换不可逆，内部坐标不变性不成立。", "error")

# --------------------------------------------------------------------------- #
# step 7: the table the spec requires
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑦ 外部坐标 / 内部坐标：前后对照")
st.caption(
    "页面必须同时给出：变换前后的外部坐标、变换前后的内部坐标、实测数值误差、可达精度预测。"
    "下表逐点给出这五项。"
)

st.dataframe(
    {
        "#": list(range(image.n)),
        "外部 (x, y)": [f"({px:+.5f}, {py:+.5f})" for px, py in x],
        "外部 T(x, y)": [f"({px:+.5f}, {py:+.5f})" for px, py in y],
        "内部 (α, β)": [
            f"({a:+.9f}, {b:+.9f})" for a, b in zip(before["alpha"], before["beta"])
        ],
        "内部 (α′, β′)": [
            f"({a:+.9f}, {b:+.9f})" for a, b in zip(after["alpha"], after["beta"])
        ],
        "Δα": [f"{v:+.3e}" for v in (after["alpha"] - before["alpha"])],
        "Δβ": [f"{v:+.3e}" for v in (after["beta"] - before["beta"])],
    },
    width="stretch",
    hide_index=True,
)

delta_alpha = np.abs(after["alpha"] - before["alpha"])
delta_beta = np.abs(after["beta"] - before["beta"])
observed = float(max(delta_alpha.max(), delta_beta.max()))

frame_scale = frame_scale_of(frame)
coordinate_scale = float(np.abs(y).max())
report = precision_of_invariance(
    frame.invariance(transform, x),
    frame_scale=frame_scale,
    coordinate_scale=coordinate_scale,
)

columns = st.columns(3)
columns[0].metric("实测 max|Δα|", f"{delta_alpha.max():.3e}")
columns[1].metric("实测 max|Δβ|", f"{delta_beta.max():.3e}")
columns[2].metric(
    "README 要求 < 1e-9",
    "✓ 满足" if observed < MATCH_TOL else "✗ 未达到",
    delta=f"余量 {MATCH_TOL / observed:.1e}x" if observed > 0 else None,
)

ui.precision_block(report, observed=observed)

st.markdown(
    "**说明（必须与结论一起读）**：`1e-9` 是**良态标架**的结论；上面的「可达精度」是"
    "本标架下真正可达的界，由两项构成——标架条件数 $\\mathrm{cond}\\,M\\cdot\\varepsilon$，"
    "以及坐标量级导致的输入侧损失 $\\varepsilon\\|P\\|/\\|B-A\\|$。"
    "后者**不在**条件数界内，它发生在求解之前。"
)

# --------------------------------------------------------------------------- #
# the formula for one point
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("单点公式（README 第 20 节：图形 + 公式 + 数值 + 解释）")
index = st.slider("选择点", 0, image.n - 1, 3 if image.n > 3 else 0)
sample = frame.coordinates(x[index])
st.code(sample.formula(3, f"P_{index}"), language="text")
st.latex(r"P_{%d} = A + %.3f\,(B-A) + %.3f\,(C-A)" % (index, sample.alpha, sample.beta))
st.dataframe(
    {
        "量": ["外部坐标 P", "内部坐标 (α, β)", "残差 ‖P − 重建‖", "该标架的可达精度"],
        "值": [
            f"({x[index, 0]:+.6f}, {x[index, 1]:+.6f})",
            f"({sample.alpha:+.9f}, {sample.beta:+.9f})",
            f"{sample.residual:.3e}",
            f"{sample.accuracy_bound:.3e}",
        ],
    },
    width="stretch",
    hide_index=True,
)

ui.why(
    "为什么 (α, β) 一定不变？",
    "设 $T(x)=Mx+t$ 且 $M$ 可逆，$u=B-A$、$v=C-A$。由 $P-A=\\alpha u+\\beta v$ 得 "
    "$TP-TA=M(P-A)=\\alpha(Mu)+\\beta(Mv)$，而 $Mu=TB-TA$、$Mv=TC-TA$。"
    "因为 $M$ 可逆，$Mu$ 与 $Mv$ 仍线性无关，所以 $(TA,TB,TC)$ 仍是合法标架；"
    "再由分解的**唯一性**即得 $(\\alpha',\\beta')=(\\alpha,\\beta)$。",
    r"P-A=\alpha u+\beta v\ \Longrightarrow\ T P - T A=\alpha (Mu)+\beta (Mv)",
)

with st.expander("标架换顺序会怎样？（这是最容易误判的一点）"):
    st.markdown(
        "把 $(A,B,C)$ 换成另一种顺序，$\\alpha,\\beta$ **会变**——因为坐标是相对于"
        "「哪个点是原点」定义的。正确的规律是：\n\n"
        "$$[q-p,\\;r-p] = [B-A,\\;C-A]\\,T,\\qquad z' = T^{-1}(z+\\delta)$$\n\n"
        "其中 $T$ 是一个整数幺模矩阵（$|\\det T|=1$，符号等于置换的奇偶性），"
        "$\\delta$ 是原点位移。**转置无关的是重心坐标** "
        "$(1-\\alpha-\\beta,\\ \\alpha,\\ \\beta)$ 随标架一起做同样的置换。\n\n"
        "教材层依赖的正是这条关系：教学型编码对三元组的 6 种顺序取最小值，"
        "从而把「顺序」这个自由度消掉。"
    )

ui.footer()
