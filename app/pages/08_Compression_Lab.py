"""5J — Compression Laboratory.

The page's core is the **two-route comparison**, because that is where compression either
commutes with a transform or does not:

    Route A:   X  →  C(X)  →  T(C(X))
    Route B:   X  →  T(X)  →  C(T(X))

Everything else on the page exists to make one of those two routes auditable: which compressor
and family (official / teaching / baseline / control), what it promised, how big the sketch is,
how far it is from the original, and what the recogniser makes of the two sketches.

The corpus is kept to **exact rationals** and the transforms are **integer maps**, so the exact
recogniser applies.  That is not a convenience: the first version of the Phase 5 experiment fed
floats to the exact pipeline and produced a headline that was purely an artefact of that mistake.
"""

from __future__ import annotations

from fractions import Fraction
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
from kozlov_lab.coding import recognize_affine_exact  # noqa: E402
from kozlov_lab.compression import (  # noqa: E402
    code_size_reduction,
    coverage_report,
    get_compressor,
    list_compressors,
)

ui.setup_page("5J 压缩实验室")
ui.title(
    "5J — 压缩实验室",
    "压缩了什么 · 保留了多少识别结构 · 换来了多少计算收益",
)

# --------------------------------------------------------------------------- #
# sidebar — compressor and parameters
# --------------------------------------------------------------------------- #
FAMILY_LABEL = {
    "official": "OFFICIAL（来自一手文本）",
    "teaching": "TEACHING（通用基线，非论文内容）",
    "experimental": "EXPERIMENTAL（本项目自研）",
    "control": "CONTROL（恒等对照）",
}
ORDER = ["IdentityCompressor", "KozlovSketch", "KozlovSkeleton", "TeachingEpsilonCover",
         "TeachingFarthestPoint", "TeachingGridReduce", "AffineAwareSampler"]

st.sidebar.header("压缩器")
contracts = {c["name"]: c for c in list_compressors()}
name = st.sidebar.selectbox(
    "选择", [n for n in ORDER if n in contracts],
    format_func=lambda n: f"{n} — {contracts[n]['family']}",
)
contract = contracts[name]
st.sidebar.markdown(f"**{FAMILY_LABEL[contract['family']]}**")
st.sidebar.caption(
    f"覆盖保证：{contract['cover_guarantee']}｜基数保证：{contract['cardinality_guarantee']}｜"
    f"结果 ⊆ 输入：{contract['surjective_on']}"
)
st.sidebar.caption(f"来源：{contract['source']}")

st.sidebar.header("参数")
parameter = {"-": None, "r": "r", "epsilon": "epsilon", "k": "k", "cell": "cell"}.get(
    "r" if name.startswith("Kozlov") or name == "AffineAwareSampler" else
    ("epsilon" if name == "TeachingEpsilonCover" else ("k" if name == "TeachingFarthestPoint"
     else ("cell" if name == "TeachingGridReduce" else "-")))
)
strength = 0.0
if parameter is not None:
    strength = st.sidebar.slider(parameter, 0.0, 2.0, 0.5 if parameter != "k" else 5.0, 0.05)
    strength = int(strength) if parameter == "k" else strength

st.sidebar.header("变换 T")
mtx = st.sidebar.selectbox("类型", ["平移", "旋转 90°", "相似 ×2", "一般仿射 det=6", "剪切 det=2"])
MAPS = {
    "平移": ((1, 0, 0, 1), (3, -2)),
    "旋转 90°": ((0, -1, 1, 0), (2, 1)),
    "相似 ×2": ((2, 0, 0, 2), (-1, 1)),
    "一般仿射 det=6": ((2, 1, 0, 3), (1, -1)),
    "剪切 det=2": ((1, 1, 0, 2), (0, 0)),
}
matrix, translation = MAPS[mtx]

# --------------------------------------------------------------------------- #
# region 1 — original image
# --------------------------------------------------------------------------- #
SCALES = {
    "triangle_A": [[0, 0], [31, 2], [6, 24], [9, 5], [19, 12], [4, 7], [11, 18], [13, 22], [18, 15]],
    "quad_B": [[0, 0], [40, 5], [28, 21], [8, 18], [13, 19], [20, 6], [30, 11], [5, 9], [35, 3], [10, 15]],
    "star_D": [[17, 0], [29, 22], [0, 25], [36, 30], [7, 8], [32, 9], [22, 36], [12, 33]],
}
st.sidebar.header("图像 X")
shape_name = st.sidebar.selectbox("图形", list(SCALES))
raw = SCALES[shape_name]
points = [(Fraction(x), Fraction(y)) for x, y in raw]

st.subheader("① 原始图像")
columns = st.columns(4)
columns[0].metric("点数 n", len(points))
try:
    reference_result = recognize_affine_exact(points, points)
    columns[1].metric("代码可定义", "✓" if reference_result.code_definable else "✗")
    columns[2].metric("定理 1 适用", "✓" if reference_result.theorem_applicable else "✗")
except Exception as exc:  # noqa: BLE001
    columns[1].metric("代码可定义", "✗")
    columns[2].metric("定理 1 适用", "✗")
    st.caption(f"{type(exc).__name__}: {exc}")
sizes = code_size_reduction(len(points), len(points))
columns[3].metric("正式代码规模 |T_X|（上界）", f"{sizes['n_entries_before']:,}")
st.caption(f"三元组数 N_X = {sizes['n_triples_before']}｜代码规模按 $q(N-1)$ 的上界给出")

st.plotly_chart(
    ui.point_figure(np.asarray(points, dtype=float), title=f"X（n = {len(points)}）", height=300),
    width="stretch",
)

# --------------------------------------------------------------------------- #
# region 2/3/4 — compressed image
# --------------------------------------------------------------------------- #
st.subheader("②–④ 压缩")
compressor = get_compressor(name)()
kwargs = {} if parameter is None else {parameter: int(strength) if parameter == "k" else float(strength)}
try:
    sketch = compressor.fit(points, **kwargs)
except Exception as exc:  # noqa: BLE001
    st.error(f"压缩失败：{type(exc).__name__}: {exc}")
    ui.footer()
    st.stop()

coverage = coverage_report(np.asarray(points, dtype=float), np.asarray(sketch.points, dtype=float))
columns = st.columns(5)
columns[0].metric("点数 before", sketch.n_before)
columns[1].metric("点数 after", sketch.n_after)
columns[2].metric("压缩率 CR", f"{sketch.compression_ratio:.3f}")
columns[3].metric("覆盖误差 E(X→S)", f"{coverage['directed_original_to_sketch']:.3e}")
columns[4].metric("Hausdorff d_H", f"{coverage['hausdorff']:.3e}")

if contract["surjective_on"]:
    pool = {tuple(map(float, q)) for q in points}
    inside = all(tuple(map(float, p)) in pool for p in sketch.points)
    state = "全部在 X 内" if inside else "有点落在 X 外（与契约矛盾）"
    extra = "（эскиз 的定义要求 A_r 是 A 的片段）" if contract["family"] == "official" else ""
    st.caption("契约声明 结果 ⊆ 输入；实测：" + state + extra)
else:
    st.caption("契约声明 结果可以不在 X 内（真 epsilon-网允许任意中心）")

ui.side_by_side(
    np.asarray(points, dtype=float),
    np.asarray(sketch.points, dtype=float),
    left_title=f"X（{len(points)} 点）",
    right_title=f"S(X)（{sketch.n_after} 点）",
    height=320,
)

# --------------------------------------------------------------------------- #
# two-route comparison
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑤ 两条路线：T(C(X)) 与 C(T(X))")


def apply_map(mapping, offset, source):
    a, b, c, d = mapping
    tx, ty = offset
    return [(a * x + b * y + tx, c * x + d * y + ty) for x, y in source]


moved = apply_map(matrix, translation, points)
route_a = apply_map(matrix, translation, sketch.points)      # T(C(X))
route_b = compressor.fit(moved, **kwargs).points             # C(T(X))

left = np.asarray(route_a, dtype=float)
right = np.asarray(route_b, dtype=float)
same_cardinality = left.shape[0] == right.shape[0]
if same_cardinality:
    difference = float(np.abs(np.array(sorted(map(tuple, left))) - np.array(sorted(map(tuple, right)))).max())
    set_equal = difference <= 1e-12
else:
    difference = float("inf")
    set_equal = False

scale = float(np.ptp(np.asarray(points, dtype=float), axis=0).max()) or 1.0
verdict = (
    "Equivariant（精确等变）" if set_equal
    else "Approximately equivariant（近似等变）" if difference <= 0.05 * scale
    else "Not equivariant（不等变）"
)
columns = st.columns(4)
columns[0].metric("基数差", f"{left.shape[0] - right.shape[0]:+d}")
columns[1].metric("最大坐标差", "∞" if difference == float("inf") else f"{difference:.3e}")
columns[2].metric("集合相等", "✓" if set_equal else "✗")
columns[3].metric("结论", verdict)
st.caption(
    "欧氏距离在一般仿射下不保持，所以**不等变是预期结果**，不是缺陷。"
    "这里的价值在于把它**测出来**并对每一族分别报告（G5.4）。"
)

st.markdown(
    "**Route A**：X → C(X) → **T(C(X))**　　vs　　"
    "**Route B**：X → T(X) → **C(T(X))**"
)
ui.side_by_side(
    left, right,
    left_title="Route A：T(C(X))", right_title="Route B：C(T(X))",
    left_colour=ui.TARGET, right_colour=ui.FRAME, height=320,
)

# --------------------------------------------------------------------------- #
# recognition panel
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑥ 压缩后的识别")
try:
    outcome = recognize_affine_exact(sketch.points, compressor.fit(moved, **kwargs).points)
    columns = st.columns(5)
    columns[0].metric("代码可定义", "✓" if outcome.code_definable else "✗")
    columns[1].metric("定理适用", "—" if outcome.theorem_applicable is None else
                      ("✓" if outcome.theorem_applicable else "✗"))
    columns[2].metric("基数相同", "✓" if same_cardinality else "✗")
    columns[3].metric("代码等价", "—" if outcome.code_equivalent is None else
                      ("✓" if outcome.code_equivalent else "✗"))
    columns[4].metric("几何 witness", "✓ 有" if outcome.witnessed else "✗ 无")
    st.info(f"**final status**: `{outcome.status.value}`")
    if not same_cardinality:
        st.warning(
            "**STRUCTURE_LOST_CARDINALITY**：两条路线给出的点数不同。"
            "这必须在谈「识别失败」之前单独记录——它比识别失败更早发生。"
        )
except Exception as exc:  # noqa: BLE001
    st.error(f"{type(exc).__name__}: {exc}")

st.markdown("**失败分类法（F5.1）——每种结论保留原因，不合并为一个布尔值**")
st.dataframe(
    {
        "类别": [
            "EXACTLY_PRESERVED",
            "AFFINE_EQUIVALENCE_LOST",
            "STRUCTURE_LOST_CARDINALITY",
            "CODE_UNDEFINED",
            "OUTSIDE_THEOREM_DOMAIN",
            "CODE_COLLISION_OUTSIDE_DOMAIN",
            "NUMERIC_INDETERMINATE",
            "RECOGNIZER_DISCREPANCY",
        ],
        "含义": [
            "压缩后仍仿射等价",
            "压缩前等价、压缩后不等价（Q2 的核心结果）",
            "两条路线的点数不同——比识别失败更早发生",
            "压缩后无法构成非退化三元组，代码不可定义",
            "代码可定义，但定理 1 的 iff 不适用",
            "代码等价而几何不等价，且位于定理域外",
            "数值层无法稳定判定",
            "出现 Phase 4 意义上的内部不一致（不该出现的缺陷）",
        ],
    },
    width="stretch",
    hide_index=True,
)
st.caption(
    "**不允许只写「识别失败」**：基数不同、代码不可定义、定理域之外、数值未定，"
    "是四种不同的结论，把它们合并成一个布尔值正是本阶段要避免的事。"
)

# --------------------------------------------------------------------------- #
# complexity panel
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑦ 复杂度与收益")
sizes = code_size_reduction(len(points), sketch.n_after)
columns = st.columns(4)
columns[0].metric("N_X = C(n,3)", f"{sizes['n_triples_before']:,}")
columns[1].metric("N_S = C(m,3)", f"{sizes['n_triples_after']:,}")
columns[2].metric("|T_S| / |T_X|（实测公式）", f"{sizes['code_size_ratio']:.4f}")
columns[3].metric("代码规模加速比", f"{sizes['code_size_speedup']:.1f}×")

theoretical = (sketch.n_after / len(points)) ** 6 if points else 0.0
st.latex(r"\frac{|T_S|}{|T_X|}\approx\left(\frac{m}{n}\right)^6")
st.dataframe(
    {
        "量": ["点数比 m/n", "理论代码规模比 (m/n)^6", "实测代码规模比", "实测加速比",
               "压缩耗时", "该压缩器覆盖半径"],
        "值": [
            f"{sketch.n_after / len(points):.4f}", f"{theoretical:.4f}",
            f"{sizes['code_size_ratio']:.4f}", f"{sizes['code_size_speedup']:.1f}×",
            f"{sketch.build_ms:.2f} ms", f"{sketch.achieved_radius:.4e}",
        ],
    },
    width="stretch",
    hide_index=True,
)
st.caption(
    "点数只降一点，正式代码的关系数按 $O(n^6)$ 大幅下降——所以「压缩」同时是**降本**手段。"
    "但上表的**实测**加速比才是可引用的数字；理论值只是标度。"
)

with st.expander("本页用到的契约与来源（G5.9 命名隔离）"):
    rows = list_compressors()
    st.dataframe(
        {
            "压缩器": [c["name"] for c in rows],
            # the upper-case label is what marks the family, so it belongs in the table rather
            # than only in the sidebar (which can show just one selection at a time)
            "家族": [FAMILY_LABEL[c["family"]] for c in rows],
            "覆盖保证": [c["cover_guarantee"] for c in rows],
            "基数保证": [c["cardinality_guarantee"] for c in rows],
            "结果 ⊆ 输入": [c["surjective_on"] for c in rows],
        },
        width="stretch",
        hide_index=True,
    )

ui.footer()
