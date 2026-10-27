# 集成架构：最小侵入地接入真实数据与 2025 理论

> 建立日期：2026-10-02（T00）
> 原则：**在现有项目上增量扩展**，不从空目录重新实现。
> 2026-10-02 复核：本文件给出架构边界，不表示模块已实施。[裁决](delegation/DECISIONS_T00_REVIEW.md) 为设计依据；[photo-io-v1](contracts.md) 已冻结照片 IO 子集，几何与数学接口仍待后续裁决。

> 2026-10-03 后续裁决：用户已选择 COLMAP/pycolmap 路线。T02A 按 [sfm-native-v1](contracts_sfm_native_v1.md) 接入 pycolmap 4.2.1 CPU 并保留原生模型；仅该子集放行。下文建议目录是架构方向，mask 过滤、去畸变及旧 Camera 适配仍待分阶段放行。

> 2026-10-03 交付目标补充：用户希望 CAD 实体，允许较大尺寸误差，目标软件待定。首版默认 FreeCAD 可打开的有效近似实体与 STEP；[交付目标](delegation/FINAL_GOAL_CAD_V1.txt) 规定真实照片→稠密表面→实体→重新导入检查的产品通路，并单列尺度/精度状态。下文理论与数据适配架构不覆盖全部 CAD 工作，后续接口与写入范围逐阶段冻结；本补充不改变当前 T02A 契约。

---

## 1. 实际现状（不要按需求 §15 的建议目录重排）

需求 §15 建议 `reconstruction/ segmentation/ kozlov/ kozlov3d/ configs/ ...` 顶层平铺，
但项目**实际**采用 `src/` layout + `src/kozlov_lab/` 单包：

```
kozlov-visual-lab/
├── app/                      # Streamlit UI：Home + 9 页（3 327 行）
├── docs/                     # 规格/理论/报告（本次新增 6 份 T00 文档）
├── examples/                 # 9 个 JSON 点集样例（合成，非图像）
├── experiments/              # 既有实验与 results/（独立复现尚未核验，见证据索引）
├── src/kozlov_lab/
│   ├── config.py             # ★ 现有"配置"方式：数值容差的单一来源（111 行）
│   ├── precision.py          # 可达精度报告（450 行）
│   ├── geometry/             # 点集 / 仿射变换 / 内部仿射坐标 / 不变量（2 856 行）
│   ├── coding/               # 2009 图像码：model/exact/numeric/equivalence/recognition/lazy/geometric/validation（4 017 行）
│   ├── compression/          # 2002：official/(kozlov_sketch) teaching/ control/ experimental/（1 540 行）
│   ├── stereo/               # 1996：baseline/(camera,triangulate) official/(kozlov_code3d) correspondence/degeneracy/metrics/model（1 748 行）
│   ├── datasets/             # 合成数据集 A–E 与形状库（748 行）
│   ├── teaching/             # 教学型仿射码（922 行）
│   └── recognition/          # ⚠️ 陈旧占位包（33 行），见 §5
└── tests/                    # 19 个文件 / 562 用例（7 358 行）
```

★ **不要**为了满足需求 §15 的示例结构而重排现有目录 —— 那会造成大范围无谓 diff 并毁掉 git 历史可追溯性。

---

## 2. 建议的最小侵入增量

沿用项目已有的**子包 + `official/` 命名隔离**惯例（`compression/official/`、`stereo/official/` 已是先例）：

```
src/kozlov_lab/
├── io/                # 【新】真实数据接入：frame 清单、图像/视频读取、mask 读写
│   ├── frames.py      #   frame_id 清单（相对路径/来源/尺寸/时间戳/处理状态）
│   ├── images.py      #   Pillow 起步的图像读取（OpenCV 为后续按需可选依赖）
│   └── masks.py       #   mask 读写 + 尺寸/方向一致性校验
├── segmentation/      # 【新】可替换分割接口（含"人工 mask"实现）
│   ├── protocol.py    #   Protocol/ABC：RGB -> binary mask
│   └── manual.py      #   人工/文件 mask
├── reconstruction/    # 【新】外部几何基线适配（COLMAP/pycolmap）
│   ├── colmap_adapter.py   # 命令构建 + 退出码/日志 + 缓存失效
│   ├── export.py           # 内参/畸变/位姿/稀疏点/观测 → 约定 schema
│   └── object_points.py    # 依 mask + 可见性规则过滤物体点
├── kozlov2025/        # 【新】严格对应 S-2025，禁止掺入工程发明
│   ├── ellipse.py     #   最小面积外接椭圆 → центр / размер（引 Загускин 1958）
│   ├── covering.py    #   покрытие / кусок / независимые / правильное
│   ├── ostov.py       #   m-остов（★ 与 2002 r-остов 命名隔离）
│   ├── futlyar.py     #   primary 使用最小中心点距 R；generalized 的 r 及约束另行区分
│   ├── containment.py #   вместимость / m-разбиение / вложимость
│   ├── group.py       #   отдельная / полная / замкнутая группа，S+ / S++
│   └── invariants.py  #   дробность M / эталон / признак / метка
├── kozlov3d/          # 【新】只放我们明确声明的三维推广
│   └── ...            #   每个模块 docstring 必须写明 OUR_3D_EXTENSION
└── viz/               # 【新】命令行调试脚本（非 GUI）
```

★ 关键：**`kozlov2025/` 与 `compression/official/` 必须命名隔离**。
`остов` 在 2002 与 2025 是**不同概念**（见 `docs/theory_evolution.md` §2）。
新增 2025 类型可命名 `Ostov2025M`；旧类型保留公开 API，不为命名统一做大规模重命名。定义关系由原文上下文裁定，不能只看词频。

---

## 3. 三层区分与验收

需求 §6 要求保持 `ORIGINAL_KOZLOV` / `ENGINEERING_INTERPRETATION` / `OUR_3D_EXTENSION` 三层区分。
项目已有**可复用的强制手段**：`tests/test_teaching_code_isolation.py`（15 个用例）
用**导入扫描**禁止禁忌名出现在错误层。

建议同法增加：

| 强制项 | 手段 | 参考先例 |
|---|---|---|
| `kozlov2025/` 严格定义与工程近似 API 区分 | 来源和接口审查，必要的依赖检查 | 扫描只作架构回归参考 |
| `kozlov3d/` 每个模块 docstring 必须含 `OUR_3D_EXTENSION` | docstring 扫描测试 | 同上 |
| 三维结论不得由 `kozlov2025/` 产出 | 依赖方向测试（2025 只依赖 geometry） | `coding/geometric.py` 的"仅依赖 geometry 层"先例 |
| 每个数学函数 docstring 必须给出**原始出处 id + 页码** | docstring 扫描（要求出现 `S-2025 p.N` 之类） | README §19 已要求"注明对应论文定义" |

扫描只能发现部分架构回归，不能证明数学语义或算法正确。S-2025 详细展开二维并提及高维推广；本项目三维算法须单独标明文献依据与新增假设。

T03 验收要求固定坐标约定下的逐点坐标、ID、观测、位姿和投影一致性；仿射等价只作可选附加检查。椭圆须检查包含性、已知解面积/中心、退化输入和收敛误差，中心仿射协变性本身不足以证明最小面积。严格 fits 的稳定性条件未落实前，不将近似结果传播为严格理论已验证。

---

## 4. 配置问题（需求 §11：先查现状，不要机械创建 `configs/default.yaml`）

**现状（实测）**：
- `src/kozlov_lab/config.py`（111 行）是**唯一容差来源**，模块 docstring 明说
  「Nothing in `kozlov_lab` may hard-code a magic epsilon: import from here instead.」
  导出 `FLOAT_TOL` / `COLLINEAR_TOL` / `MATCH_TOL` / `MAX_CONDITION_NUMBER` / `tolerance_for_condition()` 等。
- 其余"配置"以**函数默认参数**形式存在（如 `triangulate` 无参数、`match_correspondences(candidates_per_point=3, budget=200_000)`、
  `oracle_correspondence(tie_relative_pixels=1e-6, max_n=8)`）。
- `pyproject.toml` **没有** `pyyaml` 依赖；仓库内 **0 个 YAML 文件**。

**建议**：
1. **本阶段不引入 `configs/default.yaml`**，也不引入 YAML 依赖。
   理由：① 会新增依赖且与"容差单一来源"的现有纪律重复；② T00 无任何运行级参数需求；
   ③ 需求 §11 明确"避免单纯为了任务书形式要求引入架构债务"。
2. 需要**运行级参数**时（T01/T02 会需要：抽帧规则、mask 阈值、COLMAP 参数），
   用**带类型标注与显式默认值的 frozen dataclass**，放在对应子包内，
   并让容差继续从 `kozlov_lab.config` 取 —— 保持"数学容差"与"运行参数"不混。
3. 只有当 T01/T02 真的需要跨阶段复用的运行配置时，再单独提 ADR 评估 YAML。

---

## 5. ⚠️ 必须先处理的既有隐患（否则 T03 会踩）

**`src/kozlov_lab/recognition/` 是陈旧占位包**：
- 内容仅 33 行，docstring 写「Affine-equivalence recognition — **not implemented yet** (Phase 4) …
  Blocked by the Phase 3 gate: "Phase 3 无法保证点排列无关性，不允许进入 Phase 4。"」
- 但 **Phase 3 已于 2026-10-01 解锁、Phase 4 已 ✅ 完成**（Gate G4.1–G4.10 全通过），
  真实实现位于 `src/kozlov_lab/coding/recognition.py`。

⇒ **风险**：外部智能体（T03/T05）按包名直觉会往 `recognition/` 里写"新实现"，
制造与 `coding/recognition.py` 的**第二套真相**。
⇒ **已裁决**：保留路径，查调用者后将陈旧说明改为指向 `coding/recognition.py` 的指针，不新增实现。本轮未改源码，后续作为独立小改动。

**`app/`（Streamlit UI，3 327 行，10 页）**：
已存在且是 Phase 7 的成果，**保留**。
调试可视化归 T06-V；T03 为外部点云适配任务，两者均**不得**扩展 `app/`；
若需复用其中的绘图逻辑，应提取为可被 CLI 调用的纯函数，而不是加页面。

---

## 6. 分阶段冻结接口

照片 IO 已按 [contracts.md](contracts.md) 的 `photo-io-v1` 冻结。相机/可见性/几何门槛不是 T01A 的计算职责，其字段保留 null；下列几何事项在 T02/T03 前分别冻结，不阻塞 fixture 下的照片 IO 实现：

| # | 项 | 现状 |
|---|---|---|
| 1 | `frame_id` 构成与稳定性 | photo-io-v1 已定 |
| 2 | 图像通道/位深/EXIF 与 mask 同步 | photo-io-v1 已定 |
| 3 | **畸变模型** | 保留后端原生模型/参数；真实数据不得默认无畸变；去畸变适配待几何契约 |
| 4 | 内参单位与 `K` 的约定（像素 vs 归一化） | `look_at_camera` 用像素、principal 默认 `(0,0)` |
| 5 | 位姿方向约定（world→cam 还是 cam→world） | `stereo` 用 `P = K[R\|t]`，`t = -R·eye`（world→cam） |
| 6 | 坐标系与**尺度**：无绝对尺度来源时**不得**宣称米 | `triangulate.py` docstring 已保留"尺度未定"纪律 |
| 7 | mask 语义 | photo-io-v1 已定人工二值 mask；几何边界采样待 T02 |
| 8 | 可见性/遮挡规则（多视图投影中哪些点算可见） | 无 |
| 9 | 缓存失效条件 | IO 见 photo-io-v1；几何另定，不以文件存在代表成功 |
| 10 | 指标公式（重投影误差定义、注册帧数、完整度） | `stereo/metrics.py` 有局部先例 |
| 11 | 验收门槛（数值化） | 无 |

T01A 可用明确标记的 fixture 验收为 TESTED_SYNTHETIC；真实数据缺失只阻塞真实输入验收。几何契约未冻结时，不得自行定义重建门槛并宣称成功。

---

## 7. 单一写入者与并行规则（沿用任务包纪律）

- 公共资产（`config.py`、`pyproject.toml`、公共 schema）**只由主智能体**改；
  外部智能体只提交补丁建议。
- 同一文件同一时间**只有一个写入者**；外部智能体在独立副本上作业。
- 每个外部任务给：**单份任务文本 + 必要文件 + contracts 版本 + 可写范围 + 验收命令**。
- 模块归属（避免冲突）：
  - T01A → `io/`、`segmentation/` 和指定测试；视频另开 T01B
  - T02 → `reconstruction/colmap_adapter.py`、`export.py`、`object_points.py` 与对应测试
  - T03 → `reconstruction/to_kozlov.py` 与适配测试，不改 T02 文件
  - T04 → 逐个数学模块分派，主线冻结定义，不一次放行整个包
  - T05 → 另行批准的跨视模块，依赖 T03 及二维/三维定义验收
  - T06-V → `viz/` 命令行展示，T06-E → 实验/报告，分别验收
  - 现有 `geometry/ coding/ compression/ stereo/` 对以上外包任务只读；发现缺陷交最小复现，由主线另行裁决

共享 `__init__.py`、配置和依赖由主线协调；复用类前核对输入和规模约束，不要求小规模精确 oracle 处理整个真实点云。
