# 当前系统基线（T00 — Existing System Baseline & Real-World Reconstruction Gap Audit）

> 建立日期：2026-10-02（T00）
> 权威项目目录：当前 Git 仓库根目录
> 审计方式：T00 曾读取代码/测试/实验/UI 并运行测试；T00-R 仅复核、归档已有证据和修正文档，未重跑测试/实验。
> 本文件回答需求 §12 的 10 项。
> T00-R 修订：2026-10-02。证据入口：[证据索引](evidence/t00_review_20261002/INDEX.md)；后续 P1–P11 边界以 [裁决记录](delegation/DECISIONS_T00_REVIEW.md) 为准。本次文档快照不表示整体 Gate PASS。

---

## 后续状态更新 2026-10-03

T01A R1 照片与人工 mask IO 已在既有环境接收为 **TESTED_SYNTHETIC**：213 项定向测试和 11 行 CLI 产物已复核，首次审查的五项缺口关闭，见 [R1 主线记录](delegation/T01A_R1_REVIEW.md)。新增 `src/kozlov_lab/io/` 与 `src/kozlov_lab/segmentation/`，主线已声明可选 `io` extra 的 `Pillow==12.3.0` 并补安装说明；该接收时点尚未执行干净安装验证，也尚未获准真实照片。

以下 §1–§10 保留 T00 时点的目录、统计及证据快照，其中“segmentation 不存在”“依赖仅 numpy”等不能作为本次新增 IO 后的完整状态；核心必装依赖仍是 numpy。T01A 定向测试与原系统测试分别记录，不相加为新的全套成绩，不改变 T00 未决项或宣称三维系统已完成。

同日用户采用 COLMAP/pycolmap 路线，[原生模型契约](contracts_sfm_native_v1.md)继续冻结。T02A经外包R1与主线3处补修后已[接收](delegation/T02A_R1_REVIEW.md)：81个唯一用例、12个往返子集及预期NO_MODEL smoke通过，新增reconstruction指定5个文件。在该接收时点真实照片SfM与CAD仍未验证。T01安装目录已有交付候选，主线尚未单独验收它；此状态不因T02A专属环境测试而自动改变。

同日后续 [T02R 主线复核](delegation/T02R_REVIEW.md)已确认固定12张真实照片经T01产生12条成功输入，T02为PARTIAL/exit1、10图注册、3份局部模型（4图312点/3图44点/3图130点）。主模型4图未达8图工程门槛；独立像素、身份、投影和PLY核查支持局部重建事实，不能宣告完整物体/精度/CAD成功。执行及留证有缺口，不能标全项合规PASS；原证据保留、不重跑。该记录另解释4096配置值与实际4407关键点行数的主线措辞勘误，冻结契约与接受实现未改。

[CAD-01调查与CAD-00库存复核](delegation/CAD01_REVIEW.md)已收尾：同设备/焦距标签支持一次共享相机实验；时间跨度252秒，未证实完整环绕/固定角步进。指定Windows wheel不带CLI，但现有pycolmap具有去畸变API，转换尚未测。[CAD-02A候选阻塞复核](delegation/CAD02A_BLOCKED_REVIEW.md)现确认源码已变为8f422909…未接受候选，监督器失败、profile缺陷和未做回归尚待处理，真实运行0次。原11471…旧五文件由主线事后精确恢复至独立证据目录，未覆盖工作树。下一项分[监督器](delegation/CAD02A_PRE_SUPERVISOR.txt)与[profile修复回归](delegation/CAD02A_PRE_PROFILE.txt)，均不跑照片；分别复核后再派真实运行。新profile契约保持，稠密/实体/尺度均未新增验证。

---

## §1 当前真实目录结构

```
kozlov-visual-lab/                      # Git 仓库，master
├── .sources/                           # 一手来源（不入 git，.gitignore）
│   ├── SOURCES.md                      # 指向 docs/sources/SOURCES.md 的短指针
│   ├── kozlov1996_mvk321.{pdf,txt}
│   ├── kozlov2002_zvmmf1236.pdf + kozlov2002.txt
│   ├── kozlov2009_fpm1248.{pdf,txt}    # ★ 本次取得（此前仅有摘要）
│   └── kozlov2025_three_approximations.{pdf,txt}  # ★ 本次取得（此前不存在）
├── app/                                # Streamlit UI：Home.py + _shared.py + 9 页
├── docs/
│   ├── sources/SOURCES.md              # 来源登记唯一维护点（URL/日期/页数/SHA-256）
│   ├── evidence/t00_review_20261002/    # T00-R 证据索引、清单与必要附件
│   ├── theory_evolution.md             # ★ 本次新建（需求 §3）
│   ├── current_baseline.md             # ★ 本次新建（本文件）
│   ├── real_data_gap_analysis.md       # ★ 本次新建
│   ├── integration_architecture.md     # ★ 本次新建
│   ├── kozlov2025_mapping.md           # ★ 本次新建
│   ├── open_questions.md               # ★ 本次新建
│   ├── delegation/NEXT_TASKS_T01_T06.md# ★ 本次新建（新的外部任务书）
│   ├── specs/      (13 份规格, 1 805 行)
│   ├── theory/     (complexity.md, numerical_limits.md, pipeline.md)
│   ├── reports/    (phase5_gate.md, phase6_gate.md)
│   ├── experiments/(EXP-A.md, M6-invariant-table.md, README.md, sprint3-multishape.md)
│   ├── demo_script.md, final_report.md, status_report.html, terminology_ru_zh.md
├── examples/                           # 9 个 JSON 点集样例（合成，非图像）
├── experiments/                        # 12 个 exp_*.py + _common.py + run_all.py(4 263 行)
│   └── results/                        # 34 个产物 + MANIFEST.json
├── src/kozlov_lab/                     # 46 个 .py / 12 521 行
│   ├── config.py (111)  precision.py (450)
│   ├── geometry/ (5 文件, 2 856)   coding/ (9 文件, 4 017)
│   ├── compression/ (13 文件, 1 540)  stereo/ (10 文件, 1 748)
│   ├── datasets/ (3 文件, 748)   teaching/ (2 文件, 922)
│   └── recognition/ (1 文件, 33)   # ⚠️ 陈旧占位包，见 §10
├── tests/                              # 19 文件 / 7 358 行
├── pyproject.toml    # setuptools；requires-python >=3.12；依赖仅 numpy
├── .gitattributes    # * text=auto eol=lf（LF 强制）
└── .gitignore        # 含 .sources/
```

★ **不存在**：`data/`、`configs/`、`papers/`、`reconstruction/`、`segmentation/`、`kozlov/`、`kozlov3d/`、`visualization/`。

---

## §2 当前 Git 状态

| 项 | 值 |
|---|---|
| 分支 | `master`（无远程跟踪） |
| HEAD | `4a76c0aae70a4079b55c193171608e314c8d6759` |
| 提交标题 | `Phase 6: stereo vision and 3D reconstruction (6A-6R) + G6 gate audit -> PASS` |
| 工作树 | T00-R 开始时既有代码无改动，但 T00 审计文档为未跟踪文件；不能把整个工作树记为干净。本节 HEAD 为审计基点，后续文档提交另行记录。 |
| 提交数 | 10（全部在 **2026-10-01 ~ 10-02**） |

提交时间线（最新 5 条）：

```
4a76c0a 2026-10-02 Phase 6: stereo vision and 3D reconstruction (6A-6R) + G6 gate audit -> PASS
f2ed5e7 2026-10-01 Phase 5 finalization: F5.0 sanity audit, 5J UI, G5.1-G5.10 audit -> PASS
ad85715 2026-10-01 Phase 4 finalization (F4.1-F4.4) + Phase 5 compression (5A-5I)
17bdb04 2026-10-01 Sprint 4: official affine recognition (G4.1-G4.10)
055542e 2026-10-01 Phase 3 UNLOCKED: formal Kozlov image code (exact oracle + numeric strategy)
```

★ T00 新增文档与来源文件；T00-R 仅修订审计文档、来源登记和归档证据，未改动既有源码、依赖及原实验产物（见 §9 的复用清单）。

---

## §3 测试实际运行结果（本次实跑，非引用 README）

- 解释器：项目专属 Python 环境（记录版本为 Python **3.13.14**）
- 原 T00 记录的命令形式：`python -m pytest -q --tb=short --junitxml=...`；完整命令、执行器状态和子进程状态的留存边界见证据索引，不能由简写补造缺失项。
- **run3 JUnit 记录的用例结果**（归档：[junit_run3.xml](evidence/t00_review_20261002/junit_run3.xml)；完整性清单：[manifest.json](evidence/t00_review_20261002/manifest.json)）：

| 指标 | 值 |
|---|---|
| XML 中的 testcase | **562** |
| `failure` | **0** |
| `error` | **0** |
| `skipped` | **0** |

- 收集分布（19 个文件）：`test_app_pages` 64、`test_transforms` 56、`test_affine_frame` 42、`test_stereo` 40、
  `test_invariant_datasets` 38、`test_numerical_stability` 37、`test_teaching_code` 37、`test_points` 32、
  `test_kozlov_equivalence` 28、`test_kozlov_equivalence_engine` 28、`test_invariants` 26、`test_compression` 22、
  `test_kozlov_exact_oracle` 21、`test_kozlov_domain_and_validation` 19、`test_properties` 19、
  `test_kozlov_lazy` 16、`test_kozlov_recognition` 15、`test_teaching_code_isolation` 15、`test_gate_phase2` 7。
- ★ 与 README 声明（561 passed）**不一致**：实际为 **562**。README 需回改。

### ⚠️ 测试会话证据缺口与未决异常

1. **必须按运行配对**：run3 XML 有 562 个 testcase、0 failure/error/skipped；[run3_stdout.txt](evidence/t00_review_20261002/run3_stdout.txt) 到达 `[100%]`，但没有汇总行，**该次退出码未存证**。明确的 `EXIT=1` 在前一次 [run2_stdout.txt](evidence/t00_review_20261002/run2_stdout.txt)。不得把 run2 退出码与 run3 XML 拼成同一次运行，也不能据此断言“进程退出码不反映测试结果”。
2. run3 输出含 `[safe-delete][SAFE_DELETE_FAIL_CLOSED] … trash-failed` 与 `SAFE_DELETE_BULK_CONFIRM_REQUIRED`。这些是相关诊断线索，尚不能证明缺汇总或退出异常的根因。受影响的 T00 运行缺少 terminal summary；[historical_pytest_stdout.txt](evidence/t00_review_20261002/historical_pytest_stdout.txt) 则留有完整 `562 passed`，所以不写“从不写出汇总”。保留安全删除保护，不为取得 exit 0 绕过保护或修改退出状态。
3. [historical_lastfailed.json](evidence/t00_review_20261002/historical_lastfailed.json) 留有 10 个历史 nodeid；这些 nodeid **全部不在 run3 XML 中**，当前相关测试已改名。mtime 早于 run3，只能支持“历史缓存”；不能称为假记录，也不与本次 0 失败直接矛盾。
4. [run1_stdout.txt](evidence/t00_review_20261002/run1_stdout.txt) 在约 12% 处有一个 `F`，但缺少失败摘要，**失败项身份尚未确认**。保留该失败证据，撤回“顺序相关”的推断；原 T00 记录 `pytest-randomly` / `pytest-random-order` / `pytest-xdist` 均未安装，`-p no:randomly` 不能提供随机顺序对照。
5. 原 T00 叙述曾做单用例与 60 个临时目录探针。当前归档未提供足以核验其完整调用和状态的配对证据，故不据此排除环境或删除保护相关原因；根因继续记为 **UNKNOWN**。

**Gate 判据**：同时核对同次收集/实际执行清单、完整 XML、Python 子进程退出码和会话结束情况，并单独保存外层执行器状态。XML 是用例结果证据，不能作为唯一签收判据。本轮只修正与归档，不重新签收整体 Gate；后续有边界的诊断遵循 P9。

---

## §4 Phase 0–8 实际能力矩阵

证据等级：**E1** = T00 用例结果证据（run3 会话退出状态缺口见 §3）；**E2** = T00 代码审阅；**E3** = 既有产物+manifest（未重跑）。下表的 `PROVEN` 沿用原表作为局部能力标签，不表示完整测试会话成功或一般性数学证明。

| 能力 | 结论 | 证据 | 说明 |
|---|---|---|---|
| 二维点集（集合语义、重编号不变） | **PROVEN** | E1+E2 | `geometry/points.py`；`test_points` 32 用例 |
| 仿射变换族（平移/旋转/缩放/剪切/反射/奇异） | **PROVEN** | E1+E2 | `geometry/transforms.py`；`test_transforms` 56 用例 |
| **内部仿射坐标 (α,β) + 可达精度报告** | **PROVEN** | E1+E2 | `geometry/affine_frame.py` + `precision.py`；★ 项目核心 |
| 不变量表（含 `vacuous` 三值语义） | **PROVEN** | E1+E2 | `geometry/invariants.py`；`test_invariant_datasets` 38 用例 |
| **2009 图像码 `K(A)=⟨M_A,T_A⟩` + `Fraction` 精确 oracle** | **PROVEN** | E1+E2 | `coding/{model,exact,numeric,geometric,lazy}`；`test_kozlov_exact_oracle` 21 用例 |
| 仿射等价判定 + 识别 + 适用域 | **PROVEN** | E1+E2 | `coding/{equivalence,recognition,validation}.py`；Phase 4 Gate G4.1–G4.10 |
| **2002 压缩：эскиз / остов / Теорема 4** | **PROVEN** | E1+E2 | `compression/official/kozlov_sketch.py`；原文引用本次已复核 |
| 教学型仿射码（**与正式码隔离**） | **PROVEN** | E1+E2 | `teaching/affine_code.py`；`test_teaching_code_isolation` 15 用例以导入/命名扫描防部分架构回归；扫描不能证明无重复算法或数学语义正确 |
| 合成数据集 A–E（**非对称性经测量**） | **PROVEN** | E1+E2 | `datasets/{corpus,shapes}.py` |
| 三维点集 / 3D 仿射 / 体的码 / `a'`-等价 | **PROVEN（数学层）** | E1+E2 | `stereo/model.py` + `stereo/official/kozlov_code3d.py`（SOURCE-EXACT，引 S-1996 Теорема 3/4） |
| 双视图 **DLT 三角化**（含退化拒绝） | **PROVEN（合成）** | E1+E2 | `stereo/baseline/triangulate.py`；★ 秩亏时**主动抛错**，拒绝伪造深度 |
| 对应求解（oracle 全排列 + 剪枝 B&B）与**裁决感知**一致性 | **EXPERIMENTAL**（论文自标） | E2 | `stereo/correspondence.py` docstring 自标 *EXPERIMENTAL, ours*；输入是**已抽取的点**，非像素 |
| 立体视觉实验扫描（scenes × sigma × baseline × orientation） | **TESTED_SYNTHETIC** | E3 | `experiments/results/phase6_stereo.csv`（66 965 B）；场景为 cube/cuboid/tetrahedron/pyramid/irregular |
| 相机模型 `P=K[R\|t]` + 解析构造 | **IMPLEMENTED_BUT_LIMITED** | E2 | `stereo/baseline/camera.py`；★ **无畸变模型**、`look_at_camera` 为解析构造 |
| `AlgT`/`AlgSo`（1996 的构造过程与 Algorithm 层） | **NOT_IMPLEMENTED** | E2 | `docs/reports/phase6_gate.md` §5：«**AlgT' / AlgSo 未实现**（步骤未提取）…… 模型层通过，算法层未通过»。§6 只报告来源定位进展，未放行算法实现 |
| 多视图 SfM（增量注册/全局优化） | **NOT_IMPLEMENTED** | E2 | 全仓库无 BA/PnP/essential/RANSAC |
| 图像特征检测/描述（SIFT/ORB/…） | **NOT_IMPLEMENTED** | E2 | 无检测器；`teaching` 里的 "descriptor" 是**代数描述子**，与图像特征无关 |
| 稠密重建 / MVS / 深度图 | **NOT_IMPLEMENTED** | E2 | 无 |
| Mesh / 纹理模型 | **NOT_IMPLEMENTED** | E2 | 无 |
| 真实图像/视频 I/O | **NOT_IMPLEMENTED** | E2 | `src/` 内**零** `imread`/`Image.open`/`cv2`/`VideoCapture` 命中 |
| 分割 / mask | **NOT_IMPLEMENTED** | E2 | 无模块、无 mask 文件 |
| Streamlit 交互式实验室（10 页） | **PROVEN** | E1+E2 | `app/`；`test_app_pages` 64 用例 |
| 实验记录框架（SHA-256 + git provenance manifest） | **IMPLEMENTED_BUT_LIMITED** | E2+E3 | 11 个必需产物的当前哈希与 manifest 匹配；生成时完整代码状态绑定不足，干净版本上的独立复现尚未核验，见 §12 |

### ★ 需求 §5A 的专项确认：所谓"3D reconstruction"到底是哪一层？

按需求给出的 6 个候选层级，**实测定位**：

| 候选 | 是否具备 | 证据 |
|---|---|---|
| 1. 数学/合成 stereo reconstruction | ✅ **是** | `stereo/model.SCENES`（cube/cuboid/tetrahedron/pyramid/irregular）；`phase6_stereo.csv` 的 scene 列全是合成体 |
| 2. 已知对应点三角化 | ✅ **是**（主路径） | `triangulate_dlt` / `triangulate_many`；CSV 中 `correspondence=known` 的 S1 行 |
| 3. 多视图几何 | ⚠️ **部分** | 仅有**双视图**；`correspondence` 引擎支持双视；**无**三视图以上 |
| 4. 真实照片 SfM | ❌ **否** | 无真实图像、无特征、无 BA、无 COLMAP |
| 5. 稠密 MVS | ❌ **否** | 无 |
| 6. Mesh reconstruction | ❌ **否** | 无 |

★ **结论：当前"3D reconstruction"= 第 1 层 + 第 2 层（合成场景 + 已知对应三角化），
外加第 3 层的一个双视图特例。 绝不能统称为已完成的"3D reconstruction"，
更与真实照片多视图重建无关。**
（注：`test_stereo.py` 有 40 个用例、`exp_phase6_stereo.py` 有完整扫描 —— 质量不低，
但**验证的是合成设置下的数学行为**，不是真实重建能力。）

---

## §5 真实照片/视频链路缺口（摘要，详见 `real_data_gap_analysis.md`）

目标链 10 段中：**2 段 IMPLEMENTED_BUT_LIMITED（合成侧）、1 段分层不齐、7 段为空**。

| 链段 | 状态 |
|---|---|
| real photos/video 读取 | ⛔ 无 |
| frames（抽帧/清单） | ⛔ 无 |
| object masks | ⛔ 无 |
| camera intrinsics | ⚠️ 可存 `K`，**无畸变模型**、无标定、无 EXIF |
| feature matching | ⚠️ 仅"已给点集"的对应求解，**无图像特征** |
| camera poses | ⚠️ 仅解析构造，**无 PnP/BA/RANSAC** |
| sparse 3D points | ⚠️ 双视图 DLT（已知相机+已知对应） |
| dense object point cloud | ⛔ 无 |
| Kozlov representation | ⚠️ 有 2009/2002 层；2025 可复用基础点集操作，专门结构尚缺，见 §6 |
| mesh / 纹理 | ⛔ 无 |

**★ 当前代码在链条上的终止点**：给定「已构造相机 + 已给定对应点」→ 双视图 DLT 三角化。
上游（像素→点→相机）与下游（稠密→Mesh）**全空**。

---

## §6 2025 论文映射状态

**来源**：`S-2025` 已取得俄文原文（17 页 / 期刊 146–162 / SHA-256 `82fc78d0…31b8f9`），
登记见 [`docs/sources/SOURCES.md`](sources/SOURCES.md)。T00-R 复核关键定义页并修正结论强度，不宣称已经完成四篇论文的全面数学审计。

**映射结果**（详见 `docs/kozlov2025_mapping.md`）：基础点集操作可部分复用，`подизображение` 等 2025 专门结构仍缺实现，详见逐项表。
本表与 mapping 的定义分组口径不同，不计算未经统一口径的覆盖率；已有 2002 同名类型也不计作 2025 已实现。半径须区分 primary/generalized футляр，prototype 单值假定按 P5/P6 修正。

| 概念 | 状态 |
|---|---|
| изображение / часть / подизображение | `PARTIAL_REUSE`：基础点集操作可复用；`подизображение` 的仿射等价类等 2025 语义仍 `MISSING` |
| среда S / S′ / S⋆ | `MISSING`（且 §Q5 组合爆炸） |
| отдельная группа（第一逼近） | `MISSING` |
| центр / размер（最小面积外接椭圆） | `MISSING`（★ 全仓库无椭圆/凸包代码） |
| покрытие / кусок / правильное | `MISSING` |
| остов（m-остов） | `MISSING`；与 `compression` 的 2002 版 **`CONFLICT`（同名不同义）** |
| футляр ⟨A,O⟩ | `MISSING`：primary футляр 的 R 固定为最小中心点距；generalized футляр 使用受约束的给定 r（r≤最小中心点距），详见 mapping 与 P6 |
| вместимость / m-вместимое / вложимость | `MISSING` |
| дробность M / m-совместимая | `MISSING` |
| эталон（**可为多个**） | `MISSING`（★ 与 brief 的单一 prototype 假定冲突） |
| полная группа / полный образ（第二逼近） | `MISSING`（★ 最具工程价值：可作识别判别器） |
| существенная часть / признак / метка | `MISSING` |
| замкнутый образ / шлейф / спектр / подобраз | `MISSING` |

**原文范围与检索证据边界**

1. 原 T00 的词频结果只用于定位文本。抽取文本零命中不能证明概念不存在、术语首次出现或唯一出处；同名词出现也不能单独证明同义或不同义。定义、前提和跨版本关系须回到原文上下文及公式逐条核对。
2. 2025 PDF p.2 / 期刊 147 提到三维图像和高维推广，并把后文详细讨论限制于二维。准确表述为：**本文详细展开二维结构；提及高维推广，但没有给出本项目完整三维实现及真实照片重建流程。** 本项目新增的三维定义仍须标 `OUR_3D_EXTENSION`；引用 1996 时也必须逐条满足其投影模型和定理前提，不能整体替现代 SfM 背书。
3. 2002 与 2025 的 `остов` 按已核对定义及来源隔离，不能因同名直接合并；`дробность` 等尚待核对的上下文保留未决状态。命名边界见 P4。

### ★ 附带发现的**来源定位进展**

`docs/reports/phase6_gate.md` §5 登记 «**`AlgT'`/`AlgSo` 未实现**（一手文献的逐步描述**未提取**）»
并称「这是本阶段最大的开口」。**但 `S-1996` 原文中确实存在**：
> «Опишем процедуру **AlgT** построения некоторого тела Т1 по S1 и S2 …»
> «**Теорема 4.** Если S1 и S2 суть a'-разные проекции тела Т … и тело Т' построено по S1 и S2
> использованием **AlgT'**, то тела Т и Т' a'-эквивалентны.»

⇒ 这只解除“来源未定位”的阻塞（原文在 `.sources/kozlov1996_mvk321.txt` 内）。还须分别提取 `AlgT'` 与 `AlgSo` 的逐步过程、前提、目标和出处，再裁决算法是否放行；搜到名称/定理不等于步骤已复现或 Gate 已通过，见 P11。

---

## §7 外部依赖建议

| 需求 | 建议 | 理由 |
|---|---|---|
| 图像 I/O（T01） | **先用已装的 Pillow 12.3.0 + NumPy**；OpenCV 为后续视频/图像处理可选依赖 | P2 已明确：照片输入开发不被 OpenCV 缺失阻塞 |
| 最小面积外接椭圆（T04） | P2 允许 SciPy 作为可选数值依赖，算法仍须单独实现与验证；`cv2.fitEllipse` 不是最小面积外接椭圆的替代 | 验收需检查包含性、已知解面积/中心、退化输入、收敛与误差界，不能只测中心仿射协变 |
| SfM/MVS 基线（T02） | 引入 **COLMAP**（CLI 或 `pycolmap`） | 需求 §9 定位为外部基线；**不得**声称实现了 SfM |
| 三维可视化（后续独立子任务） | 后续分配时再评估 `open3d` 或 matplotlib 导出 | 不属于 T03 外部点云适配的交付范围 |
| 点云/Mesh（T06 之后） | `trimesh` | 谨慎：需求说稠密/mesh **仅在单独分配时推进** |
| 运行级配置 | **不引入 YAML**；用带类型默认值的 frozen dataclass；容差继续取 `kozlov_lab.config` | 见 `integration_architecture.md` §4；`pyproject.toml` 无 pyyaml 依赖 |
| GPU | RTX 5060 Laptop 8 GB / driver 582.05 **存在**；MVS 可用性记 `UNKNOWN` | ★ "识别到 GPU" ≠ "MVS 可运行" |

★ 本 T00/T00-R **未安装任何包、未下载权重**。P2 的可选依赖许可不表示本轮安装；具体任务需要时使用项目隔离环境并锁定实测版本。

---

## §8 推荐的新 T01–T06

完整任务书见 **`docs/delegation/NEXT_TASKS_T01_T06.md`**（取代旧 `03_T01`/`04_T02`/`05_T03`）。

| # | 任务 | 与旧包的差别 |
|---|---|---|
| T01 | Real Image / Video Ingestion | 在 `io/`+`segmentation/` 增量接入现有核心；无真实数据只阻塞真实验收，契约冻结后可用显式 fixture 开发 |
| T02 | External SfM Adapter | 明确定位为**外部基线**；**不得**声称实现 SfM |
| T03 | **3D Point Set → Existing Kozlov Core Adapter** | ★ **不是**重写 stereo，而是"外部点云 → 现有数据结构"的适配层 |
| T04 | Kozlov 2025 Formalization | 先做形式化和椭圆单模块；其他理论模块按定义/来源条件另行放行 |
| T05 | 2D/3D Cross-View Structural Consistency | 新增；★ 必须用裁决感知比较，不能只看重投影误差 |
| T06 | Object Reconstruction Experiment | A–D；★ 只有 D 才检验 Kozlov representation |

**后续放行顺序**：T04 的形式化/椭圆单模块可与 T01 并行；T02 真实运行必须等 T01 真实验收；T03 外部点云适配可用 fixture 提前开发；T05 依赖必须包含 T03。本轮仅 T00-R，不同时展开上述实施。

**★ T01 开发与真实验收的前置分开登记**：

1. **开发**：冻结 `integration_architecture.md` §6 的契约（frame_id、原生相机模型/畸变、位姿方向、尺度、mask、可见性、缓存失效、指标及门槛等）后，可使用明确标注的 fixture 开发和单测。
2. **真实验收**：需用户提供或明确选定的有许可数据（如 1 个物体 15–30 张照片 + 可选 1 段视频），输入放 `data/real/<dataset_id>/` 且原图只读，输出进 `outputs/`。真实照片不得默认“无畸变”；保留后端原生 `camera_model/params`，复用针孔 Camera 前明确去畸变及对应 K/观测/mask，不支持的模型显式拒绝。

---

## §9 可直接复用的代码清单（★ 禁止重复实现）

| 模块 | 用途 | 复用方式 |
|---|---|---|
| `kozlov_lab.config` | **全部数值容差** | 直接 import；★ 不得硬编码 epsilon |
| `geometry/points.PointSet2D` | 二维点集（集合语义） | 直接复用 |
| `geometry/transforms.AffineTransform` | 仿射变换族 | 直接复用 |
| `geometry/affine_frame.AffineFrame` | 内部仿射坐标 + 条件数 | 直接复用 |
| `geometry/invariants` | 不变量/关系判定（含 `vacuous`） | 直接复用 |
| `coding/exact.py` | **精确 oracle 模式**（`Fraction`） | ★ T04 应照此做"精确/近似分层" |
| `coding/{model,equivalence,recognition,validation}` | 2009 码与等价判定 | 直接复用；T03 可作补充等价诊断，导入验收仍须逐点坐标/ID/观测/位姿/投影一致 |
| `coding/lazy.py`, `coding/geometric.py` | 延迟实现 / 仅依赖 geometry 的独立 oracle | 直接复用 |
| `compression/official/kozlov_sketch.py` | 2002 版 эскиз/остов | 直接复用；★ 与 2025 остов **命名隔离** |
| `compression/metrics.py` | 压缩度量 | 直接复用 |
| `stereo/model.{PointSet3D,AffineTransform3D,build_scene}` | 三维点集/变换/**合成场景** | ★ T03/T06-A 直接复用 |
| `stereo/baseline/camera.{Camera,look_at_camera}` | 相机 `P=K[R\|t]` | T03 在相机模型已兼容或已去畸变时用 `from_parts`；原生畸变参数保留在适配契约，不伪装为零畸变 |
| `stereo/baseline/triangulate` | DLT 三角化 + 退化拒绝 | 直接复用 |
| `stereo/correspondence.{engines_agree,RecoveryOutcome}` | **裁决感知**的结果比较 | ★ T05 必须用它 |
| `stereo/official/kozlov_code3d` | 体的码 / 平行投影 / `a'`-等价 | 直接复用 |
| `stereo/metrics.py`, `stereo/degeneracy.py` | 度量 / 退化分类法（D1…D7） | 直接复用 |
| `precision.py` | 可达精度报告 | 直接复用（报误差时用） |
| `experiments/_common.py` | provenance/stamp 工具 | ★ 新实验应复用其 manifest 机制 |
| `tests/conftest.py` + 既有 fixture | 测试夹具 | 直接复用 |

---

## §10 绝对不要重复实现的地方（★ 硬约束）

1. **不要**建立 `new_affine.py` / `new_geometry.py` / `new_stereo.py` 之类重复实现
   （需求 §5C）。`geometry/`、`coding/`、`compression/`、`stereo/` 四包**只读**。
2. **不要**重写 SfM/MVS/三角化（需求 §9、T02）。
3. **不要**重新实现 2009 图像码或仿射等价判定 —— 它们已是 `Fraction` 精确的 oracle 实现。
4. ⚠️ **不要把新实现写进 `src/kozlov_lab/recognition/`**：
   该包仅 33 行、docstring 仍写"not implemented yet (Phase 4)"，与**已完成的 Phase 4** 矛盾；
   真实实现在 `coding/recognition.py`。P8 已选“先查调用者、保留路径、把陈旧说明改为指针”，
   放后续独立提交，不删包、不新增实现，本轮不改源码。
5. **不要**把 2002 的 остов 与 2025 的 остов 合并为同一实现（详见 `theory_evolution.md` §2）。
6. **不要**混淆两种 `футляр`：primary（PDF p.10）固定 R 为骨架中心的最小点距；generalized（PDF p.13）允许给定 r，但须满足 r≤最小中心点距及其余定义条件。按 `kozlov2025_mapping.md` 与 P6 分别实现，不能把任一种简化为任意半径的区域。
7. **不要**用质心/包围盒中心替代 2025 的 `центр`（Теорема 1 依赖"最小面积"性质）。
8. **不要**把 2025 详细展开的二维定义直接"抬"成三维而不标注 `OUR_3D_EXTENSION`（论文提及高维推广，不等于已给出本项目的三维定义和实现）。
9. **不要**用 `cv2.fitEllipse` 冒充最小面积外接椭圆而不标注（二者语义不同）。
10. **不要**扩展 `app/`（Streamlit）来做 T03 —— 需求要求命令行脚本，不做 GUI。
11. **不要**为了满足旧任务书形式而创建 `configs/default.yaml`（见 `integration_architecture.md` §4）。
12. **不要**把 fixture / mock / 合成结果标为 `TESTED_REAL` 或"Kozlov 理论验证成功"。

---

## §11 P1–P11 裁决摘要

权威边界见 [`delegation/DECISIONS_T00_REVIEW.md`](delegation/DECISIONS_T00_REVIEW.md)。下表是导航摘要，不另立待裁决状态；本轮不实施这些后续任务。

| # | 事项 | 出处 |
|---|---|---|
| P1 | 输入 `data/real/<dataset_id>/` 原图只读，输出 `outputs/`；fixture 可开发，真实验收等待有许可数据 | `real_data_gap_analysis.md` §3.A.1 |
| P2 | 照片先 Pillow+NumPy；SciPy/OpenCV 按需作为可选依赖，不预装 | 本文件 §7 |
| P3 | 保留原生相机模型/参数；真实照片不得默认无畸变，不支持模型显式拒绝 | `integration_architecture.md` §6 |
| P4 | 2002/2025 类型按来源和语义隔离，保留旧公开 API | `open_questions.md` Q1 |
| P5 | 严格层返回集合；不完整搜索标 `complete=false`，单一代表另设工程选择 | `open_questions.md` Q2 |
| P6 | 区分 primary 的固定 R 与 generalized 的受约束给定 r；其余定义条件及单中心/重复中心/边界须核对，不擅设默认值 | `kozlov2025_mapping.md`、`open_questions.md` Q4 |
| P7 | 严格 fits 保留稳定性；未证实时返回 UNKNOWN/UNSUPPORTED，近似解另标 | `open_questions.md` Q6 |
| P8 | 保留 recognition 包路径，查调用者后另提交说明指针修正 | `open_questions.md` Q8 |
| P9 | 先归档同次运行证据；有边界诊断，保留删除保护，不盲跑全量 | 本文件 §3 |
| P10 | 后续冻结代码/配置并隔离输出逐脚本复现；不覆盖旧产物、不阻塞无关 IO 开发 | 本文件 §12 |
| P11 | 仅解除来源定位阻塞；分别提取 AlgT'/AlgSo 过程和前提后再裁决实现 | 本文件 §6 |

---

## §12 未在本 T00 中验证的事项（诚实边界）

1. **实验未重跑**：§4 中标 `E3` 的条目来自**既有产物 + MANIFEST**，本次**未**重跑 `run_all.py`。
2. ⚠️ **实验快照完整性已核对，独立复现尚未核验**：
   - `provenance.git_commit = f2ed5e7…`（**Phase 5** finalization）、`git_dirty = true`；
   - `required` 包含 `phase6_stereo.csv`；`docs/reports/phase6_gate.md` 第 5 行明确记录“f2ed5e7 + 未提交改动”，因此它与后来 Phase6 提交 `4a76c0a` 并不矛盾；
   - `experiment_config = {"quick": false, "verify_only": true}`
     —— `run_all.py` 第 101–143 行表明该次汇总跳过实验，只检查必需文件存在/非空并重算哈希；不核对旧哈希或 sidecar 内容，也不重演实验。
   - `docs/reports/phase6_gate.md` §7.1 记载 10 个脚本逐个运行后再汇总；verify-only **不表示此前没有运行实验**。T00-R 对 11 个必需产物只读重算哈希，均匹配原 manifest，支持当前快照完整性。
   **限定结论**：生成时的完整代码状态绑定不足，干净版本上的独立复现尚未核验；旧实验未被上述字段推翻。
   **后续 P10**：先冻结实际脚本清单、配置、代码快照和输出隔离；若程序写死 `experiments/results`，在单独验证副本运行，不覆盖原始产物。可以逐脚本运行后汇总，每项保留自身命令、退出码、输入/配置/代码与产物哈希。本轮不重跑，长期复现不阻塞无关照片 IO 开发。
3. **GPU 的 MVS 可用性** 未探测（COLMAP 未安装）⇒ `UNKNOWN`。
4. **教材 `Козлов 2007 [1]`**（2025 的参考文献 [1]，"最优互位可由有限过程求得"的依据）
   未见全文落盘 ⇒ 登记 `SOURCE_REQUIRED`。
5. **2009 文本层**抽取时 pypdf 报 CFF Type1 字体编码警告（未装 fontTools）：
   正文西里尔字母正常，但**数学符号需逐处目视复核**后才能写入公式。
6. **2002 中 `дробность` 的原文上下文**仍待核对（`open_questions.md` Q9）；不从抽取文本的命中次数推出概念唯一性。
