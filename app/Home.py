"""Home — the entry point of the interactive laboratory.

The page is ordered like the demo storyline of README section 22 rather than like a
software manual: the mathematical problem first, then the pipeline, then the
current status and — explicitly — what is **not** implemented and why.
"""

from __future__ import annotations

import pathlib
import sys

_APP = pathlib.Path(__file__).resolve().parent
_ROOT = _APP.parent
for _path in (_APP, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st  # noqa: E402

import _shared as ui  # noqa: E402
from kozlov_lab import __version__  # noqa: E402
from kozlov_lab.datasets.corpus import all_profiles  # noqa: E402

ui.setup_page("Kozlov Visual Perception Lab")

ui.title(
    "Kozlov Visual Perception Lab",
    "《视觉感知的数学理论》交互式实验与识别系统 ｜ Козлов：Математическая теория зрительного восприятия",
)

st.markdown(
    "本实验室回答一个数学问题，而不是训练一个网络："
)
st.latex(
    r"\text{机器如何从数学意义上判断两幅发生几何变化的图像属于同一个视觉对象？}"
)

st.markdown("### 流水线")
st.code(
    "图像 → 有限点集 → 几何变换 → 内部坐标 → 图像编码 → 等价判定 → 识别",
    language="text",
)

st.markdown("### 一句话概括")
ui.explain(
    "绝对坐标不是视觉对象的可靠描述；但「相对于图像自身三个点的坐标」是——"
    "它对一切可逆仿射变换保持不变。",
    "Абсолютные координаты не описывают объект надёжно; "
    "внутренние аффинные координаты относительно трёх точек изображения — описывают.",
)

columns = st.columns(3)
columns[0].metric("实测仿射不变性误差", "2.2e-14", "要求 < 1e-9")
columns[1].metric("测试", "307 passed", "含 19 个性质测试")
columns[2].metric("包版本", __version__)

# --------------------------------------------------------------------------- #
# the demo storyline, as navigation
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("演示路线（README 第 22 节）")
st.markdown(
    "| # | 步骤 | 页面 |\n|---|---|---|\n"
    "| ①–② | 同一个图形移动后坐标全变 ⇒ 绝对坐标不可靠 | **M1 点集实验室** |\n"
    "| ③ | 用图像自己的三个点建立内部坐标系 | **M3 内部仿射坐标** |\n"
    "| ④–⑤ | 施加旋转 / 缩放 / 剪切，外部坐标改变 | **M2 变换实验室** |\n"
    "| ⑥ | 内部仿射坐标保持不变 ← **项目核心** | **M3 内部仿射坐标** |\n"
    "| ⑦ | 用不变量构造图像编码（教学型） | **M4 教学型编码** |\n"
    "| ⑧ | 用代码识别仿射等价图像 | **M7 正式仿射识别** |\n"
    "| ⑨ | 压缩后重新识别 | **M8 压缩实验室** |\n"
    "| ⑩ | 从左右投影恢复三维 | **M8 立体视觉实验室** |\n"
    "| — | 正式图像代码本身（对照教学型） | **M5 正式图像代码** |"
)

# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("实验数据集 A–E（刻意非对称）")
st.caption(
    "对称图形会掩盖四类缺陷：点对应错误、标架定向错误、置换 bug、反射歧义。"
    "以下数据集的不对称性是**被测量**的，不是标签。"
)
st.dataframe(
    {
        "数据集": [f"{p['key']} — {p['name_zh']}" for p in all_profiles()],
        "点数": [p["n_points"] for p in all_profiles()],
        "共线三元组": [p["n_collinear_triples"] for p in all_profiles()],
        "平行线段对": [p["n_parallel_pairs"] for p in all_profiles()],
        "非对称": ["✓" if p["is_asymmetric"] else "✗" for p in all_profiles()],
        "共线性行": ["有见证" if not p["collinearity_row_is_vacuous"] else "vacuous" for p in all_profiles()],
        "平行性行": ["有见证" if not p["parallelism_row_is_vacuous"] else "vacuous" for p in all_profiles()],
    },
    width="stretch",
    hide_index=True,
)

# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("当前状态")
st.markdown(
    "| 阶段 | 状态 |\n|---|---|\n"
    "| Phase 0–2：规格 / 点集 / 变换 / 内部仿射坐标 | ✅ 完成，Gate 通过 |\n"
    "| UI M1 / M2 / M3 / M6 | ✅ 完成 |\n"
    "| M4 教学型编码器（Pedagogical Extension） | ✅ 完成，**与正式理论隔离** |\n"
    "| 多形状不变量验证（数据集 A–E） | ✅ 完成 |\n"
    "| 统一可复现实验输出 | ✅ `experiments/results/` |\n"
    "| **Козлов 正式图像代码** | ❌ 未实现 —— 未决问题 **Q1** 未解决 |\n"
    "| Phase 4–10 | ⛔ 阻塞 |"
)

st.warning(
    "**关于第 5 行**：教学型编码器是根据内部仿射坐标思想构造的，"
    "**不声明**其等同于 Козлов 教材中的正式图像代码。"
    "正式定义需要由一手资料核实（Q1）；在核实之前，`src/kozlov_lab/coding/` 保持为空。"
)

with st.expander("不要做的事（README 第 23 节）"):
    st.code(
        "× 把项目做成 OpenCV 图片滤镜\n"
        "× 直接调用神经网络完成识别\n"
        "× 用像素相似度代替数学等价\n"
        "× 把哈希误称为数学不变量\n"
        "× 没有证明就声称算法满足仿射不变性",
        language="text",
    )

ui.footer()
