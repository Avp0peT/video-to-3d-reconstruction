"""M5 — Kozlov 2009 official image code (the formal construction).

This page is the **formal** counterpart of ``04_Teaching_Code.py``, and the two must not
be confused.  They are different constructions:

=========================  =================================  ==========================
                           this page (M5)                      M4 (teaching)
=========================  =================================  ==========================
object                     ``<M_A, T_A>``: ratios of areas      frame-relative coordinates
source                     Kozlov 2009, verbatim                this project, pedagogical
value type                 ``Fraction`` (exact) or ``float``    float, quantised
=========================  =================================  ==========================

The page therefore leads with the source, and with the **three independent questions**
the specification insists on keeping apart: is the code definable, does theorem 1 apply,
and are the numbers trustworthy.  Collapsing them into one "valid" badge is the mistake
the design exists to prevent.
"""

from __future__ import annotations

import pathlib
import sys

_APP = pathlib.Path(__file__).resolve().parent
_ROOT = _APP.parent
for _path in (_APP, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import _shared as ui  # noqa: E402
from kozlov_lab.coding import (  # noqa: E402
    NUMERICAL_STRATEGY_NOTE,
    DuplicatePointError,
    KozlovImageError,
    affine_equivalent_exact,
    build_kozlov_image_code_exact,
    build_kozlov_image_code_numeric,
    codes_equivalent_exact,
    cross_check_exact,
    validate_kozlov_image,
)

ui.setup_page("M5 正式图像代码")
ui.title(
    "M5 — Козлов 2009 正式图像代码",
    "K(A) = ⟨M_A, T_A⟩，T_A = { S_mnu / S_ksp }（遍历所有三元组对，两三元组不同，分母非退化）",
)

# --------------------------------------------------------------------------- #
# separation banner: this page is NOT the pedagogical construction
# --------------------------------------------------------------------------- #
st.info(
    "**这是正式的（正本）实现**，来源为 Козлов 2009 论文全文"
    "（Фундам. прикл. матем. 15:5, 95–110）正文第一页给出的定义。\n\n"
    "它与 **M4 教学型编码** 是**不同的构造**：M4 用「标架 + 内部仿射坐标 + 排序量化」，"
    "本页用「两个三元组的面积之比」。两者都仿射不变，但机制不同，**不可互相替代**。"
)

ui.formula(r"\rho_{mnu,ksp}=\frac{S_{mnu}}{S_{ksp}},\qquad K(A)=\langle M_A,\ T_A\rangle")
ui.explain(
    "三元组内部顺序不重要（无序集合）；两个三元组必须不同；S_ksp = 0 时该比值不定义。",
    "Порядок внутри тройки не важен; тройки различны; при S_ksp = 0 отношение не определено.",
)

# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
st.sidebar.header("输入图像")
mode = st.sidebar.radio("坐标类型", ["整数（精确层）", "浮点（数值层）"])

DEFAULT_TEXT = "0,0\n2,0\n0,2\n1,1\n3,5"
text = st.sidebar.text_area("每行一个点：x,y", DEFAULT_TEXT, height=150)
parsed: list[list[float]] = []
parse_error: str | None = None
for line_no, raw in enumerate(text.strip().splitlines(), start=1):
    chunk = raw.strip()
    if not chunk:
        continue
    parts = [piece for piece in chunk.replace("\t", ",").split(",") if piece.strip()]
    if len(parts) != 2:
        parse_error = f"第 {line_no} 行不是 x,y 形式：{chunk!r}"
        break
    try:
        parsed.append([float(parts[0]), float(parts[1])])
    except ValueError:
        parse_error = f"第 {line_no} 行含非数字：{chunk!r}"
        break

if parse_error:
    st.error(parse_error)
    ui.footer()
    st.stop()
if len(parsed) < 3:
    st.error(f"正式代码至少需要 3 个点（构成一个三元组）；当前 {len(parsed)} 个。")
    ui.footer()
    st.stop()

points = parsed

# --------------------------------------------------------------------------- #
# the strict adapter: duplicates are refused, deduplication is explicit
# --------------------------------------------------------------------------- #
st.subheader("① 入口校验：重复点必须显式处理（A8 裁定）")
st.caption(
    "原文的数学对象是**有限集合**，集合不含「同一坐标出现两次」。Phase 1 的通用基础设施"
    "接受多重集，两者契约不兼容，因此这里**拒绝**而不是静默合并——静默合并会改变用户的输入。"
)

array = np.array(points, dtype=float)
unique_rows = np.unique(array, axis=0)
n_duplicates = array.shape[0] - unique_rows.shape[0]
use_deduplicated = False
if n_duplicates:
    st.warning(
        f"检测到 {n_duplicates} 个重复点：点数 **{array.shape[0]} → {unique_rows.shape[0]}**（去重后）。"
    )
    use_deduplicated = st.checkbox(
        "Deduplicate as preprocessing（显式预处理：去掉重复点后继续）", value=False
    )
    if not use_deduplicated:
        st.error("已拒绝。请显式勾选上面的选项，或修改输入。")
        ui.footer()
        st.stop()
    points = [[float(x), float(y)] for x, y in unique_rows]
    st.success(f"已按显式预处理去重：现在 {len(points)} 个点。")

if any(float(v) != round(float(v)) for row in points for v in row) and mode.startswith("整数"):
    st.error(
        "整数模式下坐标必须为整数值（或改用 `Fraction` 输入精确有理数）。"
        "请把坐标类型切换为「浮点（数值层）」。"
    )
    ui.footer()
    st.stop()

# --------------------------------------------------------------------------- #
# the three layers
# --------------------------------------------------------------------------- #
st.subheader("② 三个互相独立的问题（不可合并为一个「有效」）")

try:
    if mode.startswith("整数"):
        verification = validate_kozlov_image(points)
        code = build_kozlov_image_code_exact(points)
        layer_used = "exact"
    else:
        from kozlov_lab.coding import validate_kozlov_image_numeric

        verification = validate_kozlov_image_numeric(points)
        code = build_kozlov_image_code_numeric(points)
        layer_used = "numeric"
except DuplicatePointError as exc:
    st.error(str(exc))
    ui.footer()
    st.stop()
except KozlovImageError as exc:
    st.error(f"{type(exc).__name__}：{exc}")
    ui.footer()
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    ui.footer()
    st.stop()

columns = st.columns(3)
columns[0].metric(
    "1. 代码是否可定义",
    "✓ 是" if verification.code_definable else "✗ 否",
    help="全部共线时不存在任何非退化分母，代码不可定义",
)
columns[1].metric(
    "2. 定理 1 的 iff 是否成立",
    "✓ 是" if verification.theorem_applicable else "✗ 否",
    help="定理针对「点不落在一条直线或两条平行线上」的图像",
)
columns[2].metric(
    "3. 数值是否良态",
    "✓ 是" if verification.numerically_well_conditioned else "✗ 否",
    help="存在 |D| 低于舍入底噪的三元组时，比值由噪声主导",
)

st.code(verification.describe(), language="text")
if not verification.theorem_applicable:
    st.warning(
        "**代码可以计算，但定理 1 的 iff 保证不适用。** "
        "这只影响「用它作完整识别不变量」的宣称，不影响代码本身的存在性。"
    )
if not verification.numerically_well_conditioned:
    st.error(
        f"有 {len(verification.noise_dominated_triples)} 个三元组的 |D| 低于舍入底噪 "
        f"（{verification.noise_floor}）。定义允许它们作分母，但由此得到的 ρ 量级不说明图像的任何性质。"
        "常见原因是坐标不可精确表示（如十进制 0.1）使**意图中的共线**不再精确共线。"
    )

st.caption(f"使用的层：`{layer_used}`" + ("（Fraction，精确）" if layer_used == "exact" else "（float + flatness 策略）"))
if layer_used == "numeric":
    st.info(NUMERICAL_STRATEGY_NOTE)

# --------------------------------------------------------------------------- #
# the code itself
# --------------------------------------------------------------------------- #
st.subheader("③ 代码 K(A)")
st.caption(
    "T_A 的元素是**带索引**的比值——记录是哪两个三元组产生了它。这不是装饰："
    "等价关系正是通过重编号双射 ψ 作用在这些索引上的。"
)

summary_columns = st.columns(4)
summary_columns[0].metric("点数 n", code.n_points)
summary_columns[1].metric("三元组数", code.n_triples)
summary_columns[2].metric("可作分母的三元组", code.n_admissible_denominators)
summary_columns[3].metric("|T_A|", len(code))

ui.why(
    "为什么 n = 3 时 |T_A| = 0？",
    "因为定义要求两个三元组**不同**：比值 S_mnu / S_mnu 恒等于 1，不携带任何信息，"
    "因此不属于 T_A。三个点只有一个三元组，构不成「不同的一对」，所以 T_A 为空集。\n\n"
    "这**不是**编码失败：仿射群在非共线三点上是传递的，所以所有三点的平面图像"
    "本来就彼此仿射等价——代码在这里「没有区分能力」恰好是正确的。",
    r"#\{m,n,u\}=1,\quad #\{k,s,p\}=1,\quad "
    r"#\{\text{pairs}: \{m,n,u\}\ne\{k,s,p\}\}=0",
)

if len(code) == 0:
    st.warning(
        "**|T_A| = 0**：只有一个三元组，而定义要求两个三元组**不同**，因此没有任何比值。"
        "这不是错误——仿射群在非共线三点上是传递的，所以所有三点的平面图像本就彼此等价，"
        "代码「没有区分能力」恰好是正确的。但界面必须说明这一点。"
    )
else:
    table_rows = [
        {
            "分子三元组": ",".join(str(i) for i in sorted(entry.numerator)),
            "分母三元组": ",".join(str(i) for i in sorted(entry.denominator)),
            "ρ": str(entry.value) if layer_used == "exact" else f"{entry.value:.6e}",
        }
        for entry in sorted(code.entries, key=lambda e: (sorted(e.numerator), sorted(e.denominator)))
    ]
    shown = st.slider("显示条目数（前 N 条）", 5, min(len(table_rows), 60), min(20, len(table_rows)))
    st.dataframe(table_rows[:shown], width="stretch", hide_index=True)
    st.caption(f"共 {len(table_rows)} 条；上表显示前 {shown} 条。")

if code.degenerate_triples:
    st.markdown("**退化三元组（D = 0，只能作分子，给出 ρ = 0）**")
    st.code(", ".join("{" + ",".join(str(i) for i in sorted(t)) + "}" for t in code.degenerate_triples),
            language="text")

st.caption(
    "**哈希只是索引，不是数学图像代码本身**（README 第 7、23 节）。"
    f"本呈现的 SHA-256：`{code.digest()[:32]}…`；重编号后哈希会改变，但代码仍然等价。"
)

# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #
st.subheader("④ 等价判定：重编号双射 ψ（两种答复不能合并）")
st.caption(
    "形式理论**不**要求代码逐字相同，而要求存在双射 ψ: M_A → M_B 保持所有对应的面积比。"
    "下面用「把点顺序打乱」来构造同一图像的另一种编号。"
)

seed = st.number_input("打乱用的随机种子", 0, 10**9, 20261001)
rng = np.random.default_rng(int(seed))
permutation = list(rng.permutation(len(points)))
permuted = [points[i] for i in permutation]

try:
    if layer_used == "exact":
        reference = code
        other = build_kozlov_image_code_exact(permuted)
        comparison = codes_equivalent_exact(reference, other)
    else:
        reference = code
        other = build_kozlov_image_code_numeric(permuted)
        from kozlov_lab.coding import codes_equivalent_numeric

        comparison = codes_equivalent_numeric(reference, other, tolerance=1e-9)
except (KozlovImageError, ValueError) as exc:
    st.warning(f"打乱后的图像无法编码：{exc}")
    comparison = None

if comparison is not None:
    st.code(comparison.describe(), language="text")
    if comparison.undecided:
        st.warning(
            "**无法判定**：搜索预算耗尽。这与「不等价」是**不同**的答复——"
            "把预算上限当作「不等价」会把资源限制变成假阴性。"
        )
    elif comparison.equivalent:
        st.success("代码等价：重编号后代码不变（在 ψ 的意义下）。")
    else:
        st.error("代码不等价。")

if layer_used == "exact" and len(points) >= 3:
    with st.expander("交叉校验：与「几何路线」比对（定理 1 预测两者一致）"):
        st.caption(
            "几何路线**不使用代码**：它用有理数解出仿射映射，再验证点集相等。"
            "两条路线除输入坐标外不共享任何代码路径，因此一致与否是一个真正的检验。"
        )
        report = cross_check_exact(points, permuted)
        st.code(
            f"code_route.equivalent     = {report['code_route']['equivalent']}\n"
            f"geometry_route.equivalent = {report['geometry_route']['equivalent']}\n"
            f"routes_agree              = {report['routes_agree']}",
            language="text",
        )
        if report["routes_agree"] is True:
            st.success("两条路线一致。")
        else:
            st.error("两条路线不一致——其中一条有错，需人工核查。")

# --------------------------------------------------------------------------- #
# contrast with the teaching construction
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑤ 与 M4 教学型编码的区别（不要混用）")
st.dataframe(
    {
        "维度": ["基本单元", "是否需要标架", "不变性来源", "消编号手段", "主要风险", "来源"],
        "M5 正式（本页）": [
            "两个三元组的面积之比",
            "不需要（无所谓 frame 是否有序）",
            "面积按 |det L| 等比例缩放，比值抵消",
            "取遍所有三元组对 + 不计编号（ψ）",
            "退化/近退化分母使 ρ 由噪声主导",
            "Козлов 2009 论文（一手）",
        ],
        "M4 教学型": [
            "标架下的内部仿射坐标",
            "需要，且要对 6 种顺序取规范代表",
            "内部坐标在 M 下恒等不变",
            "排序 + 量化",
            "量化步长 vs 可达精度的冲突",
            "本项目构造（教学用）",
        ],
    },
    width="stretch",
    hide_index=True,
)
st.info(
    "**本编码器是根据内部仿射坐标思想构造的教学型仿射不变编码，"
    "不声明其等同于 Козлов 教材中的正式图像代码。**"
    "经过 Phase 3 的取证，这句话的理由更强了：两者被证实是**不同的构造**，"
    "而不只是「未经验证的相似物」。"
)

ui.footer()
