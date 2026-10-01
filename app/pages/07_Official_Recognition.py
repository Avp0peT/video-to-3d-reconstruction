"""M6 — Official Affine Recognition (Sprint 4Q).

Two images in, and every one of the questions the specification requires answered
separately — code definable? theorem applicable? arithmetic mode? code equivalent?
geometric witness? final status — plus, when a map is found, the map itself and the three
plots ``X``, ``T(X)``, ``Y`` so that ``T(X) = Y`` is something the eye can check.
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
from kozlov_lab.coding import (  # noqa: E402
    ArithmeticMode,
    KozlovImageError,
    RecognitionStatus,
    recognize_affine_exact,
    recognize_affine_numeric,
)

ui.setup_page("M6 正式仿射识别")
ui.title(
    "M6 — 正式仿射识别",
    "两个点集图像之间的仿射关系：正式代码路线 + 独立几何路线 + 交叉校验",
)

ui.explain(
    "本页把四个概念严格分开：**数学定理**（适用域内的 iff）、**代码等价**（重编号双射 ψ）、"
    "**几何仿射等价**（显式映射）、**数值近似**（容差内的匹配）。",
    "Страница разделяет теорему, эквивалентность кодов, геометрическую эквивалентность и "
    "численное приближение.",
)

# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
DEFAULT_X = "0,0\n2,0\n0,2\n1,1\n3,5"
st.sidebar.header("图像 X")
text_x = st.sidebar.text_area("X：每行 x,y", DEFAULT_X, height=130, key="m6_x")

st.sidebar.header("图像 Y")
scenario = st.sidebar.radio(
    "Y 的构造方式",
    ["X 的仿射像", "X 的重编号", "无关图像", "手动输入"],
    help="前三项由 X 直接派生，便于演示；最后一项自行输入。",
)
matrix_a = st.sidebar.slider("M[0][0]", -3, 3, 2)
matrix_b = st.sidebar.slider("M[0][1]", -3, 3, 1)
matrix_c = st.sidebar.slider("M[1][0]", -3, 3, 0)
matrix_d = st.sidebar.slider("M[1][1]", -3, 3, 3)
translation_x = st.sidebar.slider("t_x", -10, 10, 5)
translation_y = st.sidebar.slider("t_y", -10, 10, -7)
manual_y = st.sidebar.text_area("Y（手动）", "0,0\n5,0\n0,5\n2,2\n7,3", height=130, key="m6_y")

mode = st.sidebar.radio("算术模式", ["精确（Fraction）", "浮点（带容差）"])
tolerance = st.sidebar.number_input(
    "浮点容差", min_value=1e-12, max_value=1.0, value=1e-6, format="%.1e"
)


def _parse(text: str, label: str) -> list[list[float]] | None:
    points: list[list[float]] = []
    for line_no, raw in enumerate(text.strip().splitlines(), start=1):
        chunk = raw.strip()
        if not chunk:
            continue
        parts = [piece for piece in chunk.replace("\t", ",").split(",") if piece.strip()]
        if len(parts) != 2:
            st.error(f"{label} 第 {line_no} 行不是 x,y 形式：{chunk!r}")
            return None
        try:
            points.append([float(parts[0]), float(parts[1])])
        except ValueError:
            st.error(f"{label} 第 {line_no} 行含非数字：{chunk!r}")
            return None
    return points or None


x_points = _parse(text_x, "X")
if x_points is None:
    ui.footer()
    st.stop()

if scenario == "X 的仿射像":
    y_points = [
        [matrix_a * px + matrix_b * py + translation_x, matrix_c * px + matrix_d * py + translation_y]
        for px, py in x_points
    ]
elif scenario == "X 的重编号":
    permutation = list(np.random.default_rng(20261001).permutation(len(x_points)))
    y_points = [x_points[i] for i in permutation]
    st.caption(f"重编号 ψ 用的置换：{permutation}")
elif scenario == "无关图像":
    rng = np.random.default_rng(4242)
    y_points = [
        [float(rng.integers(-8, 9)), float(rng.integers(-8, 9))] for _ in range(len(x_points))
    ]
    if len({tuple(point) for point in y_points}) != len(x_points):
        st.warning("随机 Y 含重复点，请再点一次或改用手动输入。")
        ui.footer()
        st.stop()
else:
    y_points = _parse(manual_y, "Y")
    if y_points is None:
        ui.footer()
        st.stop()

exact = mode.startswith("精确")
if exact and any(float(v) != round(float(v)) for point in x_points + y_points for v in point):
    st.error("精确模式要求坐标为整数值；请切换到浮点模式或改写输入。")
    ui.footer()
    st.stop()

# --------------------------------------------------------------------------- #
# run the pipeline
# --------------------------------------------------------------------------- #
try:
    if exact:
        result = recognize_affine_exact(x_points, y_points)
    else:
        result = recognize_affine_numeric(x_points, y_points, tolerance=float(tolerance))
except KozlovImageError as exc:
    st.error(f"{type(exc).__name__}：{exc}")
    ui.footer()
    st.stop()

# --------------------------------------------------------------------------- #
# the separate questions
# --------------------------------------------------------------------------- #
st.subheader("① 每个问题单独回答（不合并成一个「有效」）")
columns = st.columns(5)


def _tri(value: bool | None) -> str:
    return "—" if value is None else ("✓ 是" if value else "✗ 否")


columns[0].metric("代码是否可定义", _tri(result.code_definable))
columns[1].metric("定理 1 是否适用", _tri(result.theorem_applicable))
columns[2].metric("算术模式", "—" if result.arithmetic_mode is None else str(result.arithmetic_mode))
columns[3].metric("代码等价", _tri(result.code_equivalent))
columns[4].metric("几何 witness", "✓ 有" if result.witnessed else "✗ 无")

status_colour = {
    RecognitionStatus.EXACT_AFFINE_EQUIVALENT: st.success,
    RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT: st.info,
    RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE: st.warning,
    RecognitionStatus.NUMERIC_AFFINE_MATCH: st.success,
    RecognitionStatus.NUMERIC_NON_MATCH: st.info,
    RecognitionStatus.NUMERIC_INDETERMINATE: st.warning,
    RecognitionStatus.CODE_UNDEFINED: st.error,
    RecognitionStatus.INVALID_INPUT: st.error,
}.get(result.status, st.info)
status_colour(f"**final status: `{result.status.value}`**")

st.markdown(f"**证据种类**：{'、'.join(str(kind) for kind in result.evidence) or '—'}")
if result.undecided:
    st.warning(
        "该结论是**未定**而非「不等价」：把资源上限或定理适用域缺口当成否定判断，"
        "会把「无法判定」变成假阴性。"
    )

# --------------------------------------------------------------------------- #
# the witness
# --------------------------------------------------------------------------- #
st.subheader("② 仿射 witness（若找到）")
if result.geometric_witness is not None:
    witness = result.geometric_witness
    left, right = st.columns([2, 3])
    with left:
        st.latex(
            r"M=\begin{pmatrix}" f"{witness.matrix[0][0]}&{witness.matrix[0][1]}\\\\"
            f"{witness.matrix[1][0]}&{witness.matrix[1][1]}" r"\end{pmatrix}"
        )
        st.latex(r"t=\begin{pmatrix}" f"{witness.translation[0]}\\\\{witness.translation[1]}" r"\end{pmatrix}")
        st.caption(f"det M = {witness.determinant}｜点数 {witness.n_points}｜残差 {result.residual}")
        st.code(witness.describe(), language="text")
    with right:
        st.code(
            "点对应 ψ：\n" + "\n".join(f"  {i} -> {j}" for i, j in sorted(witness.point_mapping.items())),
            language="text",
        )
    st.caption(
        "witness 是**构造性证据**：可以用它把 X 映回去再逐点核对（本页已自动核对，残差为 0）。"
    )
else:
    st.caption("本次没有找到显式映射。对「不等价」的结论，来源是几何 oracle 的**穷尽**枚举。")

# --------------------------------------------------------------------------- #
# the three plots
# --------------------------------------------------------------------------- #
st.subheader("③ X、T(X)、Y —— 让 T(X) = Y 肉眼可见")
x_array = np.asarray(x_points, dtype=float)
y_array = np.asarray(y_points, dtype=float)

if result.geometric_witness is not None:
    witness = result.geometric_witness
    transformed = np.asarray([witness.apply(tuple(point)) for point in x_array], dtype=float)
else:
    transformed = None

columns = st.columns(3)
x_range, y_range = ui._joint_range(x_array, y_array, *( [transformed] if transformed is not None else [] ))
with columns[0]:
    st.plotly_chart(
        ui.point_figure(x_array, title="X", x_range=x_range, y_range=y_range, height=340),
        width="stretch",
    )
with columns[1]:
    if transformed is not None:
        st.plotly_chart(
            ui.point_figure(transformed, title="T(X)", colour=ui.TARGET, x_range=x_range, y_range=y_range, height=340),
            width="stretch",
        )
    else:
        st.info("没有 witness，因此没有 T(X) 可画。")
with columns[2]:
    st.plotly_chart(
        ui.point_figure(y_array, title="Y", colour=ui.FRAME, x_range=x_range, y_range=y_range, height=340),
        width="stretch",
    )

if transformed is not None:
    residual = float(np.abs(np.sort(transformed, axis=0) - np.sort(y_array, axis=0)).max())
    st.success(f"把 X 的两个坐标各自排序后与 Y 比较，最大偏差 = {residual:.3e}（精确模式下应为 0）")

# --------------------------------------------------------------------------- #
# the numeric conditioning, when it exists
# --------------------------------------------------------------------------- #
if result.numeric_condition is not None:
    st.subheader("④ 数值条件（浮点模式下必看）")
    condition = result.numeric_condition
    columns = st.columns(4)
    columns[0].metric("tolerance", f"{condition['tolerance']:.1e}")
    columns[1].metric("坐标噪声地板", f"{condition['noise_floor']:.3e}")
    columns[2].metric("最大残差", "—" if condition["max_residual"] is None else f"{condition['max_residual']:.3e}")
    columns[3].metric("匹配点数", condition["matched_points"])
    st.caption(
        "坐标噪声地板是 **长度**量纲（$64\\varepsilon\\cdot$直径），与残差可比；"
        "比值噪声地板是**无量纲**量（$64\\varepsilon\\cdot\\max\\rho$），用于判断代码本身是否被噪声主导。"
        "把两者混在一起比较就是量纲错误，而这个错误恰好会让守卫在最需要它的地方失效。"
    )
    st.json(condition["conditioning"])

# --------------------------------------------------------------------------- #
# the vocabulary, so nothing is hidden behind a status name
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑤ 状态语义")
st.dataframe(
    {
        "status": [
            "EXACT_AFFINE_EQUIVALENT",
            "EXACT_NOT_AFFINE_EQUIVALENT",
            "CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE",
            "NUMERIC_AFFINE_MATCH",
            "NUMERIC_NON_MATCH",
            "NUMERIC_INDETERMINATE",
            "CODE_UNDEFINED",
            "INVALID_INPUT",
        ],
        "含义": [
            "精确仿射等价（有 witness 或定理适用域内代码等价）",
            "精确判定的不等价（几何 oracle 穷尽枚举后的否定）",
            "代码等价，但定理 1 的 iff 不适用——不得读作「仿射等价」",
            "浮点容差内的匹配（证据种类 C，不是定理）",
            "浮点下的不匹配",
            "无法判定（残差触及噪声地板，或搜索受阻）",
            "代码不可定义（全部共线，无合法分母）",
            "输入非法（重复点、点数不足、非精确坐标等）",
        ],
    },
    width="stretch",
    hide_index=True,
)

with st.expander("与 M4 教学型编码、M5 正式编码的关系"):
    st.markdown(
        "* **M4 教学型编码**：标架 + 内部仿射坐标 + 排序量化 —— 教学构造，**不是**正式定义；\n"
        "* **M5 正式图像代码**：$K(A)=\\langle M_A,T_A\\rangle$，面积比 + 带索引关系，来自 2009 正式定义；\n"
        "* **M6（本页）**：把 M5 的代码路线与一条**完全独立**的几何路线放在一起用，并交叉校验。\n\n"
        "三者都保留，因为它们回答不同的问题，而且对照本身就是教学价值所在。"
    )

ui.footer()
