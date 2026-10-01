"""6R — Stereo Laboratory.

The page is built around one ordering, because the phase's sanity rule is an ordering:

    ① the body  →  ② its two projections  →  ③ **can it be recovered at all?**  →  ④ who is who?

Step ③ must be read and pass before step ④ means anything.  A page that jumped to the
correspondence search would be studying "who corresponds to whom" while the answer to "can it be
recovered" was still unknown — the exact inversion the directive forbids.

Everything shown is computed by ``kozlov_lab.stereo``; the page only presents.  Three layers are
labelled explicitly on screen because they must never be confused:

* ``SOURCE-EXACT`` — the body, the 3D code, **parallel** projection, ``a'``-equivalence (Козлов 1996);
* ``BASELINE`` — pinhole camera, DLT triangulation, reprojection cost (standard multi-view geometry,
  *not* the author's content);
* ``EXPERIMENTAL`` — the correspondence search in this project.

``AlgT'``/``AlgSo`` are **not** implemented: the 6A gate found their step-by-step description was
never extracted, so nothing here is allowed to carry that name.
"""

from __future__ import annotations

import dataclasses
import itertools
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
from kozlov_lab.stereo import (  # noqa: E402
    AffineTransform3D,
    DegeneracyKind,
    RecoveryOutcome,
    build_scene,
    classify_geometry,
    engines_agree,
    is_a_equivalent,
    kozlov_body_code,
    match_correspondences,
    oracle_correspondence,
    parallel_project,
    profile_relative_difference,
    reconstruction_outcome,
    relative_3d_error,
    reprojection_error,
    rmse_3d,
    scene_names,
    triangulate_many,
    volume_ratio_profile,
)
from kozlov_lab.stereo.baseline.camera import look_at_camera  # noqa: E402

ui.setup_page("6R 立体视觉实验室")
ui.title(
    "6R — 立体视觉实验室",
    "先证「会恢复」（③），再研究「谁对应谁」（④）· 三层来源在页面上逐处标注",
)

# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.header("三维体（тело）")
scene = st.sidebar.selectbox(
    "图形", scene_names(), index=scene_names().index("cube"),
    help="默认选 cube：它正好展示本阶段的核心现象（多个零代价解，其中多数是错的）。"
         "选 irregular 则是最良态的对照。",
)
orientation = st.sidebar.radio(
    "姿态", ["axis_aligned", "generic"],
    help="generic = 绕 z 转 30° 再绕 y 转 37°，脱离与基线/asymmetric 的对齐",
)
GENERIC = (
    AffineTransform3D.rotation("z", 30.0).matrix @ AffineTransform3D.rotation("y", 37.0).matrix
)
body = build_scene(scene).as_array()
if orientation == "generic":
    body = body @ GENERIC.T

st.sidebar.header("相机（BASELINE）")
baseline = st.sidebar.slider("基线长度", 0.001, 8.0, 4.0, 0.001, format="%.3f")
noise = st.sidebar.slider("像素噪声 σ（相对图像尺度）", 0.0, 1.0, 0.0, 0.01)
st.sidebar.caption("焦距 500 px｜左相机 (0,0,12) 看向原点")

left = look_at_camera([0, 0, 12], [0, 0, 0], [0, 1, 0], focal=500.0, name="L")
right = look_at_camera([baseline, 0, 12], [0, 0, 0], [0, 1, 0], focal=500.0, name="R")
centres = [left.centre.tolist(), right.centre.tolist()]

image_left = left.project(body)
image_right = right.project(body)
rng = np.random.default_rng(20261007)
if noise > 0:
    scale = float(np.ptp(np.vstack([image_left, image_right]), axis=0).max()) or 1.0
    image_left = image_left + rng.normal(scale=noise * scale, size=image_left.shape)
    image_right = image_right + rng.normal(scale=noise * scale, size=image_right.shape)

n = body.shape[0]
perm = list(rng.permutation(n))
truth = tuple(perm.index(i) for i in range(n))
shuffled = image_right[perm]

kind, evidence = classify_geometry(centres, body)

# --------------------------------------------------------------------------- #
# ① the body
# --------------------------------------------------------------------------- #
st.subheader("① 三维体 A —— те́ло = R³ 中的有限点集")
st.caption(
    "**SOURCE-EXACT**：«Назовем трехмерным изображением А или телом конечное множество точек "
    "в трехмерном евклидовом пространстве.»（Козлов 1996, §2）"
)
columns = st.columns(4)
columns[0].metric("点数 n", n)
columns[1].metric("四元组数 C(n,4)", f"{len(list(itertools.combinations(range(n), 4))):,}")
code: dict | None = None
try:
    code = kozlov_body_code(body)
    columns[2].metric("|T_A| 上界（3D 代码）", f"{code['n_entries']:,}")
    columns[3].metric("共面（退化）四元组", code["n_degenerate_coplanar_quadruples"])
except Exception as exc:  # noqa: BLE001
    columns[2].metric("|T_A|", "—")
    columns[3].metric("共面四元组", "—")
    st.caption(f"{type(exc).__name__}: {exc}")
st.caption(
    "3D 代码是用**四面体体积比** $\\rho = V_{mnuv}/V_{kspq}$ 构造的 2D 构造的逐字升维，"
    "规模 $|T_A| = q(N-1) = O(n^8)$——比 2D 的 $O(n^6)$ 高两阶。共面四元组体积为 0，不能作分母。"
)

st.plotly_chart(
    ui.body_figure(body, title=f"A（n = {n}，{orientation}）", cameras=centres, height=440),
    width="stretch",
)

# --------------------------------------------------------------------------- #
# ② the two projections
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("② 两个投影 —— 平行投影（SOURCE-EXACT）与中心投影（BASELINE）")
st.caption(
    "原文只有 «проекции по разным **направлениям**»（沿不同方向的投影）：全文 «Центральн…» 出现 "
    "**0** 次、«параллельн…» 出现 **45** 次。所以作者的投影是**平行投影**；"
    "pinhole 相机 $P=K[R|t]$ 属于标准多视图几何，标 **BASELINE**，不得冒充作者内容。"
)

parallel_left = parallel_project(body, [0.0, 0.0, -1.0])
parallel_right = parallel_project(body, [0.6, 0.0, -1.0])
columns = st.columns(2)
with columns[0]:
    st.plotly_chart(
        ui.point_figure(parallel_left, title="平行投影 S₁（SOURCE-EXACT）", height=300),
        width="stretch",
    )
with columns[1]:
    st.plotly_chart(
        ui.point_figure(parallel_right, title="平行投影 S₂（方向不同）", height=300),
        width="stretch",
    )
ui.side_by_side(
    image_left, shuffled,
    left_title="左视图（pinhole, BASELINE）", right_title="右视图（已打乱对应）",
    height=300,
)

# --------------------------------------------------------------------------- #
# ③ the sanity rule
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("③ 能恢复吗？—— 已知对应 + 零噪声必须机器精度重建（本阶段的 sanity rule）")
st.caption(
    "**这是必须先读的一节。** 如果 known correspondence + zero noise + well-conditioned cameras "
    "都无法稳定重建，就不允许继续开发 correspondence algorithm。"
)

known_points, known_failed = triangulate_many(
    left.projection, image_left, right.projection, image_right
)
if known_failed or noise > 0:
    st.warning(
        f"当前设置下 {len(known_failed)} 个点三角化失败；噪声 σ={noise:.2f} 时下面的数字"
        "包含噪声，不是纯粹的 sanity 检查——把 σ 调回 0。"
    )
else:
    rel = relative_3d_error(known_points, body)
    reprojection = max(
        reprojection_error(left, known_points, image_left),
        reprojection_error(right, known_points, image_right),
    )
    columns = st.columns(4)
    columns[0].metric("相对 3D 误差", f"{rel:.2e}")
    columns[1].metric("重投影误差 (px)", f"{reprojection:.2e}")
    columns[2].metric("min 视线夹角 (°)", f"{evidence.get('min_ray_angle_degrees', float('nan')):.3f}")
    columns[3].metric("条件数", f"{evidence.get('condition_number', float('nan')):.3f}")
    st.success(
        f"**PASS** —— 相对 3D 误差 {rel:.2e} < 1e-10，"
        "«会恢复» 成立，第 ④ 节关于 correspondence 的结论才可解释。"
    )

# --------------------------------------------------------------------------- #
# ④ correspondence
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("④ 谁对应谁？—— oracle（n! 参考）与剪枝引擎（EXPERIMENTAL）")
if n > 8:
    st.error(f"oracle 要枚举 n! 个双射，n={n} 超出上限 8。请选点数 ≤ 8 的图形。")
    ui.footer()
    st.stop()

oracle = oracle_correspondence(left, right, image_left, shuffled)
pruned = match_correspondences(left, right, image_left, shuffled, candidates_per_point=3)

breakdown: dict[str, int] = {}
for candidate in oracle.good_mappings:
    swapped = dataclasses.replace(
        oracle, mapping={i: int(c) for i, c in enumerate(candidate)}
    )
    outcome = reconstruction_outcome(
        swapped, left, right, image_left, shuffled, body, true_mapping=truth
    )
    breakdown[outcome.value] = breakdown.get(outcome.value, 0) + 1

valid = breakdown.get(RecoveryOutcome.EXACT_MATCH.value, 0) + breakdown.get(
    RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY.value, 0
)
invalid = breakdown.get(RecoveryOutcome.INVALID_RECONSTRUCTION.value, 0)

columns = st.columns(5)
columns[0].metric("oracle 判定", str(oracle.verdict))
columns[1].metric("零代价解个数", len(oracle.good_mappings))
columns[2].metric("其中正确（仿射等价）", valid)
columns[3].metric("其中**错误**", invalid)
columns[4].metric("两引擎一致性", str(engines_agree(oracle, pruned)))
st.caption(
    f"n! 参考枚举了 {oracle.diagnostics['n_bijections_examined']:,} 个双射；"
    f"剪枝引擎只访问 {pruned.diagnostics.get('nodes', 0):,} 个节点，"
    "但**用同一个打分函数**，所以任何差异都只能归因于剪枝。"
)
st.caption(
    "**一致性判定必须区分「并列」与「缺陷」**：对称体或共线情形下存在**多个零代价解**，"
    "两个都正确的引擎会返回不同的代表元。把 `mapping == mapping` 当判据会把并列报成缺陷——"
    "所以判据是「答案是否属于最优集」。"
)

st.markdown("**最优集的成分（而不是引擎抽中的那一个）**")
st.dataframe(
    {
        "类别": [
            RecoveryOutcome.EXACT_MATCH.value,
            RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY.value,
            RecoveryOutcome.INVALID_RECONSTRUCTION.value,
            RecoveryOutcome.UNDECIDED.value,
            RecoveryOutcome.NO_CORRESPONDENCE.value,
        ],
        "含义": [
            "就是真对应——3D 重建与 A 仿射等价",
            "换了一个对应，但重建体仍与 A 仿射等价（对称性允许的另一种正确答案）",
            "换了一个对应，重建体**与 A 不等价**——代价仍为 0 但答案是错的",
            "等价判定**弃权**（体全部共面 ⇒ 没有可用的控制四元组）——弃权不是「正确」",
            "引擎没给出任何对应",
        ],
        "本页计数": [
            breakdown.get(RecoveryOutcome.EXACT_MATCH.value, 0),
            breakdown.get(RecoveryOutcome.EQUIVALENT_VIA_SYMMETRY.value, 0),
            breakdown.get(RecoveryOutcome.INVALID_RECONSTRUCTION.value, 0),
            breakdown.get(RecoveryOutcome.UNDECIDED.value, 0),
            breakdown.get(RecoveryOutcome.NO_CORRESPONDENCE.value, 0),
        ],
    },
    width="stretch",
    hide_index=True,
)

# --------------------------------------------------------------------------- #
# ⑤ degeneracy taxonomy
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑤ 退化分类（6J）—— 绝不写一个「failed」")
st.caption(
    "**分类法必须能表达它自己最常见的结果。** 本阶段实测发现指令列出的 D1–D6 少了两个"
    "真实且常见的类别（D7/D8），若不补名，分类法就会把真实原因误归因。"
)
st.info(f"当前配置判定：**{kind.value}**")
st.dataframe(
    {
        "代码": [k.value for k in DegeneracyKind],
        "含义": [
            "无退化",
            "D1 基线为零——其余判据全部失去意义",
            "D2 基线相对物体尺度过小",
            "D3 有点落在基线上（投影落在对极点上）",
            "D4 两条视线近平行（深度病态）",
            "D5 点云共面/过薄——3D 代码没有合法分母（**内在**属性）",
            "D6 物体对称——**良性**：多个答案都正确，不是不知道答案",
            "D7 两个点在某一视图里落在**同一条射线**上（同一像素）",
            "D8 多个点共处一个**对极平面**——左像素有多个零代价右候选",
        ],
    },
    width="stretch",
    hide_index=True,
)
columns = st.columns(4)
columns[0].metric("同射线点对 = 0？", evidence.get("n_coincident_ray_pairs", "—"))
columns[1].metric("对极平面点对 = 0？", evidence.get("n_epipolar_plane_pairs", "—"))
columns[2].metric("对称可互换点对", evidence.get("n_symmetric_pairs", "—"))
columns[3].metric("baseline/scale", f"{evidence['baseline_over_scale']:.3f}")
st.caption(
    "**D3 必须排在 D8 之前**：落在基线上的点必然处在过基线的**每一个**对极平面里，"
    "若先查 D8，D3 永远不可能被报告（实测：D8 先查时整个语料没有一次 D3）。"
    "**D6 排在最后是刻意的**：实测一个绕轴旋转 30° 的四棱锥和未旋转的一样对称（都是 6 对可互换点），"
    "但它的对应是**唯一**的——歧义来自与对极几何的**对齐**（D8），不是来自对称。"
)

# --------------------------------------------------------------------------- #
# ⑥ tie-breaking
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑥ 重投影代价**无法**拆开并列，仿射不变量可以（6O/6P）")
ideal_left, ideal_right = left.project(body), right.project(body)
ideal_shuffled = ideal_right[perm]
ideal_oracle = oracle_correspondence(left, right, ideal_left, ideal_shuffled)

residuals: list[float] = []
separated = missed = false_alarm = wrong = total = 0
reference_profile = volume_ratio_profile(body) if n >= 4 else ()
for candidate in ideal_oracle.good_mappings:
    total += 1
    points, failed = triangulate_many(
        left.projection, ideal_left, right.projection, ideal_shuffled[list(candidate)]
    )
    if failed:
        wrong += 1  # a reconstruction that did not happen is not a correct answer
        continue
    residuals.append(max(
        reprojection_error(left, points, ideal_left),
        reprojection_error(right, points, ideal_shuffled[list(candidate)]),
    ))
    equivalent = is_a_equivalent(points, body)["equivalent"]
    try:
        difference = profile_relative_difference(volume_ratio_profile(points), reference_profile)
    except ValueError:
        difference = float("inf")
    flagged = difference > 1e-6
    if equivalent:
        false_alarm += 1 if flagged else 0
    else:
        wrong += 1
        separated += 1 if flagged else 0
        missed += 0 if flagged else 1

columns = st.columns(5)
columns[0].metric("零代价解", total)
columns[1].metric("最大重投影误差 (px)", f"{max(residuals):.2e}" if residuals else "—")
columns[2].metric("其中错误", wrong)
columns[3].metric("不变量识别出", separated)
columns[4].metric("不变量漏掉", missed)
if total > 1:
    st.warning(
        f"**{total} 个解的重投影代价都 ≈ 0（最大 {max(residuals):.2e} px），其中 {wrong} 个是错的。**"
    )
st.caption(
    "⇒ **零重投影误差不蕴含对应正确。** 只报残差的管线会在一大批错答案上报告成功，"
    "因为所有零代价解的残差都一样小，残差里没有任何信息可以用来区分它们。"
    "这和 Phase 3 的「代码可定义 ≠ 定理适用 ≠ 数值良态」是同一种三分问题："
    "**「残差吻合」与「体被恢复」是两个不同的问题**。"
    "选一个退化姿态的对称体（例如默认的 cube）就能看到并列，选 irregular 则只有一个零代价解。"
)
st.caption(
    f"不变量（按序的四面体体积比摘要）识别出 {separated} 个错误重建、漏掉 {missed} 个，"
    f"对正确重建误报 {false_alarm} 次。**它是有漏检率的检测器，不是判据**——"
    "判据是 `is_a_equivalent`（使用**带索引**的代码）。这正是本项目「有序袋不得作判据」那条规则。"
)

# --------------------------------------------------------------------------- #
# ⑦ link to the earlier phases
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("⑦ 与前面各阶段的连接（6Q）")
recovered, failed_rec = triangulate_many(
    left.projection, ideal_left, right.projection,
    ideal_shuffled[list(ideal_oracle.permutation)] if ideal_oracle.permutation else ideal_shuffled,
)
if not failed_rec:
    equivalent = is_a_equivalent(recovered, body)
    columns = st.columns(4)
    columns[0].metric("坐标 RMSE", f"{rmse_3d(recovered, body):.3f}")
    columns[1].metric("相对 3D 误差", f"{relative_3d_error(recovered, body):.2e}")
    columns[2].metric("3D 仿射等价？", "✓" if equivalent["equivalent"] else "✗")
    columns[3].metric("3D 代码 |T_A|", f"{code['n_entries']:,}" if isinstance(code, dict) else "—")
st.caption(
    "**坐标本身不是目标。** Тhеорема 3 说 3D 代码在**三维仿射等价**意义下决定体，"
    "所以「重建正确」只能表述为 $\\hat X \\sim_{aff,3D} X$，不能表述为 $\\hat X = X$。"
    "这与 2D 阶段只能比较 $a$-等价、以及 Phase 5 只能比较代码等价，是同一条纪律。"
)

with st.expander("来源与命名隔离（G6 命名规则）"):
    st.dataframe(
        {
            "内容": [
                "те́ло = R³ 有限点集", "3D 代码 ρ = V/V（四元组）", "平行（方向）投影",
                "a'-等价（三维仿射等价）", "四面体（三角+面外点）确定对应",
                "AlgT' / AlgSo", "pinhole P=K[R|t]、DLT 三角化、重投影代价",
                "基于重投影代价的 correspondence 搜索",
            ],
            "标签": [
                "SOURCE-EXACT", "SOURCE-EXACT", "SOURCE-EXACT", "SOURCE-EXACT", "SOURCE-EXACT",
                "**未实现**（步骤未提取）", "BASELINE", "EXPERIMENTAL",
            ],
            "落点": [
                "`stereo/model.py`", "`stereo/official/kozlov_code3d.py`",
                "`stereo/official/kozlov_code3d.py`", "`stereo/official/kozlov_code3d.py`",
                "`stereo/official/kozlov_code3d.py`", "—（6A Gate 未通过）",
                "`stereo/baseline/`", "`stereo/correspondence.py`",
            ],
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "6A Gate：模型层通过，**算法层未通过**。`AlgT'`/`AlgSo` 的逐步描述未从一手文献中提取，"
        "因此在读清之前**不得**以 «Kozlov stereo reconstruction» 命名任何重建过程。"
        "本页的重建一律标注为 BASELINE 的 DLT 三角化。"
    )

ui.footer()
