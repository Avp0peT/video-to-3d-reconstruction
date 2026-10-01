"""M4 — Teaching Affine Code (Pedagogical Extension).

**This page is a pedagogical extension, not the textbook's image code.**  The
construction is derived from the internal-coordinate idea; whether it coincides with
Kozlov's formal ``код изображения`` is open question Q1 and is *not* claimed here.

The page therefore leads with the disclaimer and then does the one thing the
construction is for: show that ``C_teach(X) = C_teach(T(X))`` within tolerance, and
show a control in which it must differ.
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
from kozlov_lab.geometry import AffineTransform  # noqa: E402
from kozlov_lab.config import TEACHING_CODE_DECIMALS  # noqa: E402
from kozlov_lab.teaching import (  # noqa: E402
    CODE_FORMAT_VERSION,
    TeachingCodingParameters,
    build_teaching_image_code,
    compare_teaching_codes,
    recommended_decimals,
)

ui.setup_page("M4 教学型编码")
ui.title(
    "M4 — 教学型仿射不变编码（Pedagogical Extension）",
    "C_teach(X)：对一切可逆仿射变换不变的构造 —— 但它不是教材的正式图像代码",
)

ui.teaching_disclaimer()

ui.formula(
    r"C_{\text{teach}}(X) = \mathrm{sort}\Big\{\ \mathrm{sort}\big[\,"
    r"\mathrm{quantize}\big((\alpha_i,\beta_i)_i\big)\big]\ \Big|\ "
    r"\text{all admissible frames}\ \Big\}"
)
ui.explain(
    "对每个合法标架，把全部点的内部坐标量化后排序（消掉点编号），"
    "再对所有标架描述符排序（消掉标架选择），得到规范形式。",
    "Для каждого допустимого базиса внутренние координаты квантуются и сортируются, "
    "затем сортируются сами дескрипторы базисов.",
)

image = ui.source_selector(key="m4")
x = image.as_array()

# --------------------------------------------------------------------------- #
# quantisation choice
# --------------------------------------------------------------------------- #
st.sidebar.header("量化参数")
feasible = recommended_decimals(x)
st.sidebar.caption(
    f"该图像的最差标架条件数下的推荐精度：**{feasible}** 位小数"
    f"（由 `recommended_decimals` 依据 cond·ε 推出，不是固定常数）"
)
decimals = st.sidebar.slider("decimals", 3, 12, min(feasible, TEACHING_CODE_DECIMALS))
max_frames_label = st.sidebar.selectbox(
    "max_frames（安全性上限）", ["不限制", "50", "200", "1000", "5000"], index=0
)
max_frames = None if max_frames_label == "不限制" else int(max_frames_label)
params = TeachingCodingParameters(decimals=int(decimals), max_frames=max_frames)

try:
    code = build_teaching_image_code(x, params)
except ValueError as exc:
    st.error(f"该图像无法编码：{exc}")
    st.caption(
        "这是**明确拒绝**而不是返回空编码——空编码会与任何其它空编码相等，"
        "在识别阶段就是一次静默的误报。"
    )
    ui.why(
        "为什么一维图像必须被拒绝？",
        "教学型编码的完备性依赖于「存在至少一个合法标架」。"
        "若所有三元组共线，则不存在二维仿射坐标系，编码无从定义。"
        "返回空编码看似温和，实际会让任意两幅退化图像被判为等价。",
        r"\det[\,B-A\quad C-A\,]\neq 0\ \text{for at least one triple}",
    )
    ui.footer()
    st.stop()

# --------------------------------------------------------------------------- #
# code quality
# --------------------------------------------------------------------------- #
st.subheader("① 编码质量（先看这个，再看结论）")
quality = code.quality()
columns = st.columns(4)
columns[0].metric("点数", quality["n_points"])
columns[1].metric("参与编码的标架数", quality["n_frames"])
columns[2].metric("被排除的共线三元组", quality["n_collinear_triples"])
columns[3].metric("量化是否安全", "✓ 是" if quality["quantization_is_safe"] else "✗ 否")

st.dataframe(
    {
        "项": [
            "规范形式版本",
            "quantum = 10^−decimals",
            "最差标架条件数",
            "量化余量（十进制位）",
            "整数格点量级",
            "格点可精确表示（≤ 2^53）",
            "完备不变量（未被截断）",
            "C_teach(X) 的 SHA-256",
        ],
        "值": [
            CODE_FORMAT_VERSION,
            f"{quality['quantum']:.1e}",
            f"{quality['worst_condition_number']:.3e}",
            f"{quality['quantization_digits_margin']:.2f}",
            f"{quality['quantization_magnitude']:.4e}",
            "✓" if quality["quantization_representable"] else "✗",
            "✓" if quality["complete_invariant"] else "✗（已截断，不再是完备不变量）",
            quality["digest"],
        ],
    },
    width="stretch",
    hide_index=True,
)

for warning in code.warnings:
    ui.note(warning, "warning")

st.info(
    "**哈希不是数学图像代码本身，只能是代码的计算机索引。** "
    "坐标上 1e-16 的扰动会让哈希完全改变，而数学上两幅图像几乎相同——"
    "哈希无法回答「差多少」，因此不能代替编码（README 第 7、23 节）。"
)

# --------------------------------------------------------------------------- #
# invariance demonstration
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("② 仿射不变性：C_teach(X) = C_teach(T(X))")

st.sidebar.header("变换")
seed = st.sidebar.number_input("随机种子", 0, 10**9, 20261001)
budget = st.sidebar.slider("条件数预算", 1.0, 200.0, 20.0, 1.0)
transform = AffineTransform.random_invertible(int(seed), max_condition=budget)
st.sidebar.caption(f"cond(T) = {transform.condition_number:.4f}｜类型 {transform.classify()}")

y = transform.apply(x)
comparison = compare_teaching_codes(x, y, params)

ui.side_by_side(x, y, left_title="X", right_title="T(X)", height=320)

columns = st.columns(2)
columns[0].metric("codes equal", "✓ 相等" if comparison.equivalent else "✗ 不相等")
columns[1].metric("证据强度", "弱（见 quality）" if comparison.weak_evidence else "正常")

st.code(comparison.describe(), language="text")

# --------------------------------------------------------------------------- #
# control
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("③ 对照：无关图像必须不相等")
control_seed = st.number_input("对照随机种子", 0, 10**9, 424242)
other = shapes.random_points(x.shape[0], seed=int(control_seed)).as_array()
control = compare_teaching_codes(x, other, params)

columns = st.columns(2)
columns[0].metric("与无关图像 codes equal", "✗ 不相等（正确）" if not control.equivalent else "✓ 相等（异常）")
columns[1].metric("原因", control.reason.split(":")[0])
with st.expander("对照比较的完整说明"):
    st.code(control.describe(), language="text")

if control.equivalent:
    st.error("对照失败：一个恒返回常量的实现也能通过上一节的不变性检验。")
else:
    st.success(
        "对照组成立。上一节的「相等」因此具有信息量：编码确实在区分内容，"
        "而不是对任何输入都返回同一个值。"
    )

# --------------------------------------------------------------------------- #
# canonical form
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("④ 规范形式（前几个标架描述符）")
st.caption(
    "规范形式中**不含任何点编号**，也不含条件数——点编号是任意标签，"
    "条件数在两个等价图像的末位比特上可能不同。这些量只作为诊断信息出现在下面的表里。"
)
st.code("\n".join(code.canonical_bodies()[:4]), language="text")
st.dataframe(
    {
        "标架（诊断）": [frame.describe() for frame in code.frames[:8]],
    },
    width="stretch",
    hide_index=True,
)

with st.expander("为什么规范形式里不能出现点编号？（这条被测试盯着）"):
    st.markdown(
        "如果规范文本里含有点编号，那么同一幅图像换一种书写顺序就会得到**不同的编码**——"
        "而「顺序不影响数学意义」正是 Phase 1 的核心要求。\n\n"
        "对应测试：`test_canonical_text_contains_no_point_indices`，"
        "以及 `tests/test_teaching_code_isolation.py` 中对命名与目录的隔离规则。"
    )

ui.footer()
