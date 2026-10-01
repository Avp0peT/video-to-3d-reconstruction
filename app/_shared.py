"""Shared UI helpers for the Kozlov Visual Perception Lab Streamlit app.

Layering rule (README section 2): this module may call the mathematical layer but
must never re-implement it (README section 18, "UI 不允许重新实现数学算法").  Every
number shown on a page comes from ``kozlov_lab``; nothing is recomputed here except
pure presentation (axis ranges, colours, formatting).

Presentation rules enforced by the helpers below:

* every panel shows **figure + formula + numbers + explanation** (README section 20);
* verdicts are three-state (``preserved`` / ``violated`` / ``vacuous``) and the
  ``vacuous`` case is styled as "no information", never as a pass;
* the pedagogical coding construction always renders its disclaimer.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Iterable, Mapping, Sequence

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _path in (_HERE, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from kozlov_lab import __version__  # noqa: E402
from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.datasets.corpus import DATASETS, dataset_names  # noqa: E402
from kozlov_lab.geometry import (  # noqa: E402
    AffineFrame,
    AffineTransform,
    PointSet2D,
)
from kozlov_lab.precision import PrecisionReport  # noqa: E402

# --------------------------------------------------------------------------- #
# palette (light theme)
# --------------------------------------------------------------------------- #
INK = "#1c1f24"
INK_SOFT = "#545b66"
INK_FAINT = "#8a929e"
LINE = "#e3e6ea"
ACCENT = "#1f6feb"
OK = "#14663a"
BAD = "#a8271c"
WARN = "#8a5a00"
NEUTRAL = "#5a626d"
TARGET = "#c2410c"  # transformed image
FRAME = "#7c3aed"  # reference frame points

STATE_KEY = "point_set"
STATE_NAME_KEY = "point_set_name"

COLORS = {
    "preserved": OK,
    "violated": BAD,
    "vacuous": NEUTRAL,
}
SYMBOLS = {"preserved": "✓", "violated": "×", "vacuous": "–"}


# --------------------------------------------------------------------------- #
# page furniture
# --------------------------------------------------------------------------- #
def setup_page(title: str, icon: str = "◆") -> None:
    st.set_page_config(page_title=f"{title} · Kozlov Lab", page_icon=icon, layout="wide")


def title(text: str, subtitle: str | None = None) -> None:
    st.title(text)
    if subtitle:
        st.caption(subtitle)


def formula(latex: str) -> None:
    """Display a formula (README section 20 requires it next to the numbers)."""
    st.latex(latex)


def explain(zh: str, ru: str | None = None, en: str | None = None) -> None:
    """Bilingual explanation block — the project shows Chinese and Russian."""
    st.markdown(f"> {zh}")
    if ru:
        st.caption(ru)
    if en:
        st.caption(en)


def why(question: str, answer: str, latex: str | None = None) -> None:
    """The per-page "为什么？" panel required by README section 21."""
    with st.expander(f"为什么？ {question}"):
        st.markdown(answer)
        if latex:
            st.latex(latex)


def note(text: str, kind: str = "info") -> None:
    (st.info if kind == "info" else st.warning if kind == "warning" else st.error)(text)


def teaching_disclaimer() -> None:
    """The mandatory three-language disclaimer for the pedagogical code."""
    from kozlov_lab.teaching import DISCLAIMER_EN, DISCLAIMER_RU, DISCLAIMER_ZH

    st.warning(
        "**这不是 Козлов 教材的正式图像代码。**\n\n"
        f"{DISCLAIMER_ZH}\n\n{DISCLAIMER_EN}\n\n{DISCLAIMER_RU}"
    )


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _joint_range(*arrays: np.ndarray, pad: float = 0.08) -> tuple[list[float], list[float]]:
    stacked = np.vstack([np.asarray(a, dtype=float).reshape(-1, 2) for a in arrays])
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    lo = lo - pad * span
    hi = hi + pad * span
    return [float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]


def point_figure(
    points: Any,
    *,
    title: str,
    colour: str = ACCENT,
    show_indices: bool = True,
    frame: AffineFrame | None = None,
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
    height: int = 380,
    connect: Sequence[tuple[int, int]] | None = None,
) -> go.Figure:
    """Scatter plot of a point set, with optional frame highlighting and segments."""
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    figure = go.Figure()

    if connect:
        for i, j in connect:
            figure.add_trace(
                go.Scatter(
                    x=[array[i, 0], array[j, 0]],
                    y=[array[i, 1], array[j, 1]],
                    mode="lines",
                    line=dict(color=LINE, width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    figure.add_trace(
        go.Scatter(
            x=array[:, 0],
            y=array[:, 1],
            mode="markers+text" if show_indices else "markers",
            text=[str(i) for i in range(len(array))] if show_indices else None,
            textposition="top center",
            textfont=dict(size=11, color=INK_SOFT),
            marker=dict(size=11, color=colour, line=dict(width=1, color="white")),
            name=title,
            hovertemplate="%{text}<br>(%{x:.4f}, %{y:.4f})<extra></extra>"
            if show_indices
            else "(%{x:.4f}, %{y:.4f})<extra></extra>",
        )
    )

    if frame is not None and frame.indices is not None:
        labels = ("A", "B", "C")
        for corner, label in zip(frame, labels):
            figure.add_trace(
                go.Scatter(
                    x=[corner[0]],
                    y=[corner[1]],
                    mode="markers+text",
                    text=[label],
                    textposition="bottom right",
                    textfont=dict(size=14, color=FRAME),
                    marker=dict(size=17, color=FRAME, symbol="circle-open", line=dict(width=3)),
                    name=f"frame {label}",
                    hoverinfo="skip",
                )
            )

    figure.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK_SOFT)),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(color=INK),
    )
    figure.update_xaxes(
        gridcolor=LINE,
        zerolinecolor=LINE,
        showline=True,
        linecolor=LINE,
        range=x_range,
        title=dict(text="x", font=dict(size=12, color=INK_FAINT)),
    )
    figure.update_yaxes(
        gridcolor=LINE,
        zerolinecolor=LINE,
        showline=True,
        linecolor=LINE,
        range=y_range,
        scaleanchor="x",
        scaleratio=1,
        title=dict(text="y", font=dict(size=12, color=INK_FAINT)),
    )
    return figure


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_title: str = "X",
    right_title: str = "T(X)",
    left_colour: str = ACCENT,
    right_colour: str = TARGET,
    left_frame: AffineFrame | None = None,
    right_frame: AffineFrame | None = None,
    height: int = 380,
) -> None:
    """Two panels on the **same** axes range, so the difference is visible."""
    x_range, y_range = _joint_range(left, right)
    columns = st.columns(2)
    with columns[0]:
        st.plotly_chart(
            point_figure(
                left,
                title=left_title,
                colour=left_colour,
                frame=left_frame,
                x_range=x_range,
                y_range=y_range,
                height=height,
            ),
            width="stretch",
        )
    with columns[1]:
        st.plotly_chart(
            point_figure(
                right,
                title=right_title,
                colour=right_colour,
                frame=right_frame,
                x_range=x_range,
                y_range=y_range,
                height=height,
            ),
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# verdict rendering
# --------------------------------------------------------------------------- #
def verdict_chip(verdict: str) -> str:
    """Markdown chip for a three-state verdict."""
    return {
        "preserved": f":green[**✓ preserved**]",
        "violated": f":red[**× violated**]",
        "vacuous": f":gray[**– vacuous**]",
    }.get(verdict, verdict)


def verdict_table(rows: Iterable[Mapping[str, Any]], *, theory_column: bool = False) -> None:
    """Render measurement rows as a dataframe with honest three-state verdicts.

    ``vacuous`` is rendered as "无见证（结论不含信息）", never as a pass — that is the
    single most important presentation rule in this app.
    """
    import pandas as pd

    records = []
    for row in rows:
        record: dict[str, Any] = {
            "性质": row.get("label_zh") or row.get("property"),
            "结论": {"preserved": "✓ 保持", "violated": "× 破坏", "vacuous": "– 无见证"}.get(
                row.get("measured", row.get("verdict")), row.get("measured", row.get("verdict"))
            ),
            "max|Δ|": _fmt(row.get("delta")),
            "容差": _fmt(row.get("tolerance")),
            "见证": row.get("witness") or "—",
        }
        if theory_column:
            record["理论预测"] = row.get("theory", "")
            record["一致"] = "✓" if row.get("agreed") else "✗"
        if "measure" in row:
            record["度量"] = row["measure"]
        records.append(record)
    st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—" if np.isnan(number) else "∞"
    return f"{number:.3e}"


def discrepancy_banner(check: Any) -> None:
    """Show the mandatory banner when theory and measurement disagree."""
    if not check.has_discrepancy:
        st.success(check.headline())
        return
    st.error("```\n" + (check.banner() or "") + "\n```")


def precision_block(report: PrecisionReport, *, observed: float | None = None) -> None:
    """Render the attainable-precision report (never a bare PASS/FAIL)."""
    st.markdown("**Predicted attainable precision**")
    columns = st.columns(4)
    columns[0].metric("可达精度", f"{report.total_bound:.3e}")
    columns[1].metric("cond 项", f"{report.bound_from_condition:.3e}")
    columns[2].metric(
        "量级项 ε|P|/|B−A|",
        "—" if report.bound_from_magnitude is None else f"{report.bound_from_magnitude:.3e}",
    )
    columns[3].metric("有效十进制位", f"{report.significant_decimal_digits:.1f}")
    if observed is not None:
        st.markdown(
            f"实测偏差 **{observed:.3e}** ｜ 是否在界内："
            f"{'✓ 是' if report.is_acceptable(observed) else '✗ 否'}"
        )
    for warning in report.warnings:
        note(warning.text("zh"), kind="error" if warning.severity == "critical" else "warning")


# --------------------------------------------------------------------------- #
# image source (shared by M2 / M3 / M6)
# --------------------------------------------------------------------------- #
def set_image(image: PointSet2D) -> None:
    st.session_state[STATE_KEY] = image
    st.session_state[STATE_NAME_KEY] = image.name


def get_image() -> PointSet2D:
    """The current working image, defaulting to the README's 4-point example."""
    if STATE_KEY not in st.session_state:
        set_image(shapes.triangle_demo())
    return st.session_state[STATE_KEY]


def source_selector(*, key: str, allow_datasets: bool = True) -> PointSet2D:
    """Pick the image this page operates on, and return it.

    Three sources, in order of preference:

    1. the image built on the **M1** page (kept in session state — this is how the
       pages compose into a pipeline);
    2. a built-in shape from the registry;
    3. an experiment dataset A-E.

    An empty selection would make every page meaningless, so the widget never
    produces an empty set: the builders all yield at least three points and a
    degenerate image is reported rather than swallowed.
    """
    st.sidebar.subheader("图像来源")
    options = ["M1 页面的点集", "内置图形", *(["实验数据集 A–E"] if allow_datasets else [])]
    choice = st.sidebar.radio("来源", options, key=f"{key}_source", label_visibility="collapsed")

    if choice == "M1 页面的点集":
        image = get_image()
        st.sidebar.caption(f"来自 M1：`{image.name}`（{image.n} 点）")
        return image

    if choice == "内置图形":
        name = st.sidebar.selectbox("图形", shapes.shape_names(), key=f"{key}_shape")
        image = shapes.build_shape(name)
        st.sidebar.caption(f"{image.n} 点")
        return image

    key_name = st.sidebar.selectbox("数据集", dataset_names(), key=f"{key}_dataset")
    dataset = DATASETS[key_name]
    image = dataset.image
    profile = dataset.profile()
    st.sidebar.caption(
        f"{dataset.name_zh}｜{profile['n_points']} 点｜非对称：{'是' if profile['is_asymmetric'] else '否'}"
    )
    st.sidebar.caption(
        f"共线三元组 {profile['n_collinear_triples']}｜平行线段对 {profile['n_parallel_pairs']}"
    )
    return image


def frame_selector(
    image: PointSet2D, *, key: str, default_indices: tuple[int, int, int] | None = None
) -> AffineFrame | None:
    """Choose the three reference points A, B, C.

    Returns ``None`` when the image admits no frame (fewer than three points or all
    points collinear) so that the caller can say so explicitly instead of crashing.
    """
    x = image.as_array()
    if x.shape[0] < 3:
        st.sidebar.warning("点数不足 3，无法建立仿射坐标系。")
        return None

    auto = AffineFrame.auto(x) if x.shape[0] >= 3 else None
    if auto is None:
        st.sidebar.warning("该图像的所有三元组均共线，无法建立仿射坐标系。")
        return None

    mode = st.sidebar.radio(
        "标架选择", ["自动（最小 cond/flatness）", "手动指定"], key=f"{key}_frame_mode"
    )
    if mode.startswith("自动"):
        st.sidebar.caption(
            f"自动选中 {auto.indices}｜cond = {auto.condition_number:.3f}｜"
            f"flatness = {auto.collinearity_sine:.3f}"
        )
        return auto

    indices = list(range(x.shape[0]))
    default = list(default_indices or auto.indices or (0, 1, 2))
    picked = st.sidebar.multiselect(
        "A, B, C 的编号（按顺序）", indices, default=default, max_selections=3, key=f"{key}_frame"
    )
    if len(picked) != 3:
        st.sidebar.warning("请恰好选择三个点。")
        return None
    frame = AffineFrame.from_indices(x, tuple(picked))  # type: ignore[arg-type]
    if not frame.is_valid():
        st.sidebar.error("所选三点共线，不能作为仿射标架。请另选。")
        return None
    st.sidebar.caption(
        f"cond = {frame.condition_number:.3f}｜flatness = {frame.collinearity_sine:.3f}"
    )
    return frame


# --------------------------------------------------------------------------- #
# point editing (M1)
# --------------------------------------------------------------------------- #
def point_editor(image: PointSet2D, *, key: str) -> tuple[PointSet2D, list[str]]:
    """Table-based add/delete/edit of points.

    Returns the (possibly new) image and a list of messages describing what changed,
    so the caller can report it instead of silently mutating the session state.
    """
    import pandas as pd

    messages: list[str] = []
    frame = pd.DataFrame(image.as_array(), columns=["x", "y"])
    frame.index.name = "#"
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        width="stretch",
        key=f"{key}_editor",
        column_config={
            "x": st.column_config.NumberColumn("x", format="%.6f"),
            "y": st.column_config.NumberColumn("y", format="%.6f"),
        },
    )

    if edited.equals(frame):
        return image, messages

    values = edited.to_numpy(dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] == 0:
        messages.append("编辑结果为空集：已被拒绝，图像保持不变（空集不是图像）。")
        return image, messages

    if values.shape != image.as_array().shape:
        messages.append(f"点数由 {image.n} 变为 {values.shape[0]}。")
    duplicates = values.shape[0] - PointSet2D(values).deduplicated().n
    if duplicates:
        messages.append(f"检测到 {duplicates} 个重复点（点集语义下会被视为同一集合）。")
    messages.append("图像已更新。")
    return PointSet2D(values, name=image.name or "edited"), messages


# --------------------------------------------------------------------------- #
# footer
# --------------------------------------------------------------------------- #
def footer() -> None:
    st.divider()
    st.caption(
        f"kozlov-lab {__version__} ｜ 数学层与界面层严格分离：本页面调用的所有数值均来自 "
        "`kozlov_lab`，界面不重新实现任何数学算法（README 第 18 节）。"
    )
