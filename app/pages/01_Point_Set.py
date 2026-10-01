"""M1 — Point Set Laboratory (README section 4).

An image is a finite non-empty subset of ``R^2``.  This page is the entry point of
the pipeline: it creates the object every other page operates on.

The storage order of the point list must not change the mathematical meaning of
the point set — that is stated on the page and then **demonstrated** by the shuffle
button, which recomputes the canonical signature before and after.
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
from kozlov_lab.geometry import PointSet2D  # noqa: E402

ui.setup_page("M1 点集实验室")
ui.title(
    "M1 — 有限点集图像实验室",
    "图像的数学定义：X = {x₁, …, xₙ} ⊂ ℝ²（有限、非空）",
)

ui.formula(r"X=\{x_1,x_2,\ldots,x_n\}\subset\mathbb{R}^2,\qquad n\ge 1")
ui.explain(
    "点列表的存储顺序不应该改变点集本身的数学意义。",
    "Порядок хранения списка точек не должен менять математический смысл "
    "самого множества точек.",
    "The storage order of the point list must not change the mathematical meaning "
    "of the point set itself.",
)

image = ui.get_image()

# --------------------------------------------------------------------------- #
# sidebar: construction of the image
# --------------------------------------------------------------------------- #
st.sidebar.header("构造图像")

internal = shapes.shape_names()
choice = st.sidebar.selectbox(
    "内置图形",
    ["（保持当前图像）", *internal],
    help="可选的示例图形。选择后会立即替换当前图像。",
)
if choice != "（保持当前图像）":
    if st.sidebar.button(f"载入 {choice}", type="primary"):
        ui.set_image(shapes.build_shape(choice))
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("随机点集")
random_n = st.sidebar.slider("点数", 3, 30, 8)
random_seed = st.sidebar.number_input("随机种子", 0, 10**9, 20261001)
if st.sidebar.button("生成随机点集（非共线 ）"):
    ui.set_image(shapes.random_points(random_n, seed=int(random_seed)))
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("JSON 导入 / 导出")
uploaded = st.sidebar.file_uploader("导入 JSON", type=["json"])
if uploaded is not None:
    try:
        loaded = PointSet2D.from_json(uploaded.getvalue().decode("utf-8"))
        ui.set_image(loaded)
        st.sidebar.success(f"已导入 {loaded.n} 个点。")
    except (ValueError, UnicodeDecodeError) as exc:
        st.sidebar.error(f"导入失败：{exc}")

st.sidebar.download_button(
    "导出当前图像",
    data=image.to_json_text() + "\n",
    file_name=f"{image.name or 'image'}.json",
    mime="application/json",
)

# --------------------------------------------------------------------------- #
# main: table editing + shuffle
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    st.subheader("点表（可直接增删改）")
    st.caption(
        "表格方式：双击单元格修改；用行首的多选框删除；在最后一行之后输入新坐标即可添加。"
        "鼠标选择（点选图中点）见下方说明。"
    )
    edited, messages = ui.point_editor(image, key="m1")
    for message in messages:
        st.info(message)
    if edited is not image:
        ui.set_image(edited)
        image = edited

    with st.expander("关于鼠标输入的限制（如实说明）"):
        st.markdown(
            "Streamlit 的原生图表无法把「在空白处点击的坐标」回传给服务端，因此**鼠标直接点击"
            "画布添加点**需要一个自定义前端组件。本页面采用的是：\n\n"
            "1. **表格**方式完成添加 / 删除 / 修改（满足「鼠标或表格方式输入」中的表格一路）；\n"
            "2. 图形支持 plotly 的**框选/点选**，用于在图上选中并核对某个点的坐标。\n\n"
            "我们没有用第三方组件绕过这一限制——引入一个未经审计的前端依赖，代价高于它带来的便利。"
        )

    st.subheader("点编号与坐标")
    indices = np.arange(image.n)
    st.dataframe(
        {
            "#": indices,
            "x": image.as_array()[:, 0],
            "y": image.as_array()[:, 1],
        },
        width="stretch",
        hide_index=True,
    )

with right:
    st.subheader("图形")
    st.plotly_chart(
        ui.point_figure(image.as_array(), title=f"{image.name or 'image'}（n = {image.n}）"),
        width="stretch",
    )

    st.subheader("度量")
    xmin, ymin, xmax, ymax = image.bounds()
    centroid = image.centroid()
    st.dataframe(
        {
            "量": ["点数", "质心", "包围盒", "直径", "一维？", "重复点"],
            "值": [
                f"{image.n}",
                f"({centroid.x:.4f}, {centroid.y:.4f})",
                f"[{xmin:.3f}, {xmax:.3f}] × [{ymin:.3f}, {ymax:.3f}]",
                f"{image.diameter():.4f}",
                "是（退化）" if image.is_one_dimensional() else "否",
                f"{image.duplicate_count()} 个",
            ],
        },
        width="stretch",
        hide_index=True,
    )

# --------------------------------------------------------------------------- #
# the order-independence demonstration
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("顺序无关性：打乱点顺序")

ui.explain(
    "点列表的存储顺序不应该改变点集本身的数学意义。打乱后下列每一项都必须不变——"
    "因为它们描述的是**集合**，而不是列表。",
    "После перемешивания ни одна из перечисленных величин не должна измениться, "
    "поскольку они описывают множество, а не список.",
)

before_array = image.as_array().copy()
columns = st.columns([1, 1, 2])
with columns[1]:
    shuffle = st.button("打乱点顺序", type="primary")

if shuffle:
    rng = np.random.default_rng()
    permutation = rng.permutation(image.n)
    shuffled = PointSet2D(before_array[permutation], name=image.name)
    ui.set_image(shuffled)
    image = shuffled
    st.success(f"已打乱（置换 = {list(permutation)}）")

after_array = image.as_array()
reference = PointSet2D(before_array)

st.markdown("打乱前 → 打乱后：")
ui.side_by_side(
    before_array,
    after_array,
    left_title="打乱前 X",
    right_title="打乱后 X′",
    height=320,
)

checks = [
    ("集合相等 `X == X′`", reference == image, "顺序无关的集合语义"),
    ("`signature()` 一致", reference.signature() == image.signature(), "量化指纹（集合）"),
    (
        "`multiset_signature()` 一致",
        reference.multiset_signature() == image.multiset_signature(),
        "量化指纹（含重数）",
    ),
    (
        "`sorted_points()` 一致",
        bool(np.array_equal(reference.sorted_points(), image.sorted_points())),
        "规范顺序（字典序）",
    ),
    (
        "坐标逐点对应（同一多点集）",
        sorted(map(tuple, reference.as_array().tolist())) == sorted(map(tuple, after_array.tolist())),
        "集合的内容没有变，只是书写顺序变了",
    ),
]
st.dataframe(
    {
        "检查": [name for name, _, _ in checks],
        "结果": ["✓" if ok else "✗" for _, ok, _ in checks],
        "说明": [why for _, _, why in checks],
    },
    width="stretch",
    hide_index=True,
)

st.caption(
    "注意最后一行：坐标**列表**确实变了（`X == X′` 不比较列表），"
    "而作为集合的描述完全没有变。这正是 README 第 4 节的验收条件。"
)

# --------------------------------------------------------------------------- #
# teaching panel
# --------------------------------------------------------------------------- #
st.divider()
with st.expander("为什么？为什么必须有重复点与共线点的处理"):
    st.markdown(
        "数学上点集不允许重复元素，但工程上有两个理由必须接受它们：\n\n"
        "1. **量化与浮点**产生重复点；\n"
        "2. README 第 19 节的测试矩阵明确要求覆盖「重复点」和「共线点」。\n\n"
        "这里的处理方式是**显式报告**而不是静默丢弃：`has_duplicates` 与 "
        "`duplicate_count()` 会告诉你重复了多少个，`deduplicated()` 才真正合并。\n\n"
        "共线同样如此：全共线的点集**没有**仿射标架，下游页面会明确说「无法建立仿射坐标系」，"
        "而不是给出一个看似合理的坐标系。"
    )
ui.formula(r"\det[\,B-A\quad C-A\,]\neq 0")
st.markdown("——只有满足上式，三个点才能构成二维仿射坐标系。")

ui.footer()
