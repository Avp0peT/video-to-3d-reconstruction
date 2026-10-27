# Kozlov Visual Perception Lab

**《视觉感知的数学理论》交互式实验与识别系统**
**Козлов：Математическая теория зрительного восприятия — 可视化数学实验室**

项目包含两个分别验收的部分：照片/视频到三维几何的工程流程（SAM3、COLMAP、OpenMVS、Blender/FreeCAD），以及 В. Н. Козлов 理论的数学实验。工程流程生成模型，并不自动证明论文算法完成或对重建产生收益。研究部分讨论：

$$\text{机器如何从数学意义上判断两幅发生几何变化的图像属于同一个视觉对象？}$$

$$
\text{图像}\to\text{有限点集}\to\text{几何变换}\to\text{内部坐标}
\to\text{图像编码}\to\text{等价判定}\to\text{识别}
$$

---

## 当前状态

**2026-10-11 当前状态：照片到 Blender 的系统流程已有两组真实整链成果，项目仍在实施中。**
[统一入口](docs/PHOTO_DEMO_USAGE.md) 已将颅骨 66 张照片中的 42 个有效视图、骰子 33 张照片中的 31 个有效视图分别生成表面和 `.blend`，Blender 独立回读通过；两次总状态均为 PARTIAL，未补齐缺失掩膜。骰子可辨认，颅骨仍有粗糙和跨缺口大三角片；视频目前只有外形不足的局部结果。

骰子真实照片网格另经固定 quadric 简化和 FreeCAD 转换，得到 2000 面有效闭合单实体，FCStd 重开与 STEP 重导入通过。它是**任意尺度的分面实体**，没有实测尺寸、照片精度或参数化设计验收。证据：[照片整链](outputs/delegation/GENERAL_PIPELINE/20261011_dice_e2e/ACCEPTANCE_EVIDENCE.json)、[CAD 对照结果](outputs/delegation/GENERAL_PIPELINE/20261011_dice_cad_quadric/CONTRAST_RESULT.json)。完整范围仍以[项目完成清单](docs/PROJECT_COMPLETION_CHECKLIST.md)为准；下方历史数学测试不代表整个项目完成。

下表与历史测试数保留原阶段范围。立体视觉的数值仿射对应判定已修正点重编号错误，
最新说明见[研究纠错记录](outputs/delegation/RESEARCH_CORRESPONDENCE/20261006T151710Z/CORRECTIONS.md)。
`is_a_equivalent` 是几何仿射双射数值判定器；当前三维比例袋不是原文带索引代码的完整实现。
2026-10-07新增独立[小规模精确三维索引代码](docs/INDEXED_CODE3D_USAGE.md)：完整四元组比值、全局标签双射核验及有界搜索已有50项测试和真实来源六点受控对照；旧比例袋与几何oracle保持原有职责，真实立体恢复与AlgT'/AlgSo仍未完成。

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 数学规格（Theory Agent 输出） | ✅ 完成 |
| 1 | 有限点集 + 几何变换 | ✅ 完成 |
| 2 | 内部仿射坐标（**项目核心**） | ✅ 完成，**Gate 通过** |
| 3 | 图像代码（正式，`coding/`） | ✅ **完成**（Phase 3 已解锁 2026-10-01） |
| 4 | 仿射等价识别（正式，`recognition.py`） | ✅ **完成**（Gate G4.1–G4.10 全通过） |
| 5 | 信息压缩（`compression/`） | ✅ **完成**（G5.1–G5.10 全通过；见 `docs/reports/phase5_gate.md`） |
| 6 | 立体视觉（`stereo/`） | 部分完成：几何模型、数值实验及有界精确索引代码已有实现；`AlgT'`/`AlgSo` 与真实对应恢复尚未完成，历史 G6 结论受后续纠错限制 |
| 7 | Streamlit Web UI（M1–M8） | ✅ 完成 9 页（含编码、识别、压缩实验室、立体视觉实验室） |
| 8 | 多形状不变量验证 + 统一可复现产物 | ✅ 完成 |
| 9 | 完整报告 + 答辩 Demo | ⏳ 阶段报告 |
| 10 | 2026 扩展模块 E1–E5 | ⛔ 未开始 |

**历史阶段记录**：561 passed、10 个实验通过；这些数字不是当前代码全套复验结果，也不包含照片/视频到 CAD 的最终验收。

### ★ Phase 4 结果

| Gate | 内容 | 结果 |
|---|---|---|
| G4.1 | 代码等价与数学定义一致 | ✅ 逐条比对 `brute_force_bijection` |
| G4.2 | 重编号测试全通过 | ✅ 含**适用域分类的重编号不变性**（一个真实缺陷由此发现，见规格 §13.7） |
| G4.3 | 几何 oracle 独立实现 | ✅ `geometric.py` 仅依赖 geometry 层，由导入扫描测试强制 |
| G4.4 | 适用域差分测试 0 处未解释分歧 | ✅ **9 796 例，0 处未解释分歧** |
| G4.5 | ≥10 000 个精确随机用例 | ✅ 请求 10 000，实际 9 796（204 例因退化被跳过） |
| G4.6 | $n=3$ / 共线 / 两平行线 / 重复点都有明确语义 | ✅ |
| G4.7 | 精确/数值不发生静默混合 | ✅ `ModeMismatchError` |
| G4.8 | lazy 与 materialized 逻辑一致 | ✅ $n=4,5,6,7,8$ 逐条核对 |
| G4.9 | 性能报告完成并记录 $O(n^6)$ | ✅ [`docs/theory/complexity.md`](docs/theory/complexity.md) |
| G4.10 | UI 可展示实际 affine witness | ✅ `07_Official_Recognition.py`：`M`、`t`、`det M`、点对应 ψ、X / T(X) / Y 三联图 |

Phase 4/5 的历史 Gate 报告在 `docs/reports/`。Phase 6 的算法层尚未完成；
其原报告及数字保留作历史证据，涉及点对应和比例袋判别的现行解释以上述研究纠错记录为准。

### ★ Q1 已结案：Phase 3 解锁

未决问题 **Q1** 已由 **Kozlov 2009 论文全文**（一手来源）解决：

$$\langle M_A,\ T_A\rangle,\qquad T_A=\{\,S_{mnu}/S_{ksp}\,\}$$

三个约束来自论文正文，且都影响实现：三元组**内部顺序不重要**（无序集合）；
两个三元组必须**不同**（故 $n=3$ 时 $T_A=\varnothing$）；$S_{ksp}=0$ 时比值**不定义**。
等价关系是「存在重编号双射 $\psi:M_A\to M_B$ 保持所有面积比」——
**不是**要求代码逐字相同。

来源：В. Н. Козлов，《Распознавание изображений, представляемых конечным множеством точек》，
Фундам. прикл. матем. **15**:5 (2009) 95–110（英译 J. Math. Sci. 172:5, 690–699）。
专利 RU2779379C1 **降为交叉验证来源**。

G3.1–G3.7 **七个 Gate 全部满足**：

```text
Phase 3 = UNLOCKED
```

`src/kozlov_lab/coding/` 实现五个模块，**数学定义与数值近似分开放**：

| 模块 | 角色 |
|---|---|
| `model.py` | `K(A) = <M_A, T_A>`；`RatioEntry` **保持索引**（等价关系作用在索引上） |
| `exact.py` | **`Fraction` 精确层**：`D = 0` 是精确判据 —— 数学正确性的 **oracle** |
| `numeric.py` | 浮点层 + flatness 退化**策略**（明确标注：不属于原始数学定义） |
| `validation.py` | 严格适配层：**拒绝重复点**（A8 裁定）；三层适用域判定 |
| `equivalence.py` | 重编号双射 ψ 的搜索 + **独立的几何路线** + 交叉校验 |

`KozlovImageValidation` 分别回答三个**不可合并**的问题：

$$\text{代码是否可定义}\ \ne\ \text{定理 1 的 iff 是否成立}\ \ne\ \text{数值是否良态}$$

**详见 [`docs/specs/official_image_code.md`](docs/specs/official_image_code.md)。**

---

## 快速开始

### 照片或视频一键建模

Windows PowerShell 统一入口为 `scripts/run_photo_demo.ps1`。输入可以是照片目录或视频文件；
入口会依次执行输入整理、SAM3 对象分割、SfM、视图与几何诊断、表面生成、
Blender 导出、研究侧支和报告。建议先执行计划检查，确认依赖和阶段计划后再运行真实任务。

```powershell
Set-Location '<project-root>'

# 只检查输入、依赖和执行计划，不运行模型
.\scripts\run_photo_demo.ps1 `
  -Source '<path-to-video>' `
  -Prompt 'glove' `
  -Frames 24 `
  -Target blender `
  -PlanOnly
```

视频到 Blender 的一键执行示例：

```powershell
.\scripts\run_photo_demo.ps1 `
  -Source '<path-to-video>' `
  -Prompt 'glove' `
  -Frames 24 `
  -Target blender `
  -Output '.\outputs\photo_demo\glove_demo01'
```

照片目录到 Blender 的一键执行示例：

```powershell
.\scripts\run_photo_demo.ps1 `
  -Source 'C:\data\object_photos' `
  -Prompt 'beverage can' `
  -Target blender `
  -Output '.\outputs\photo_demo\can_demo01'
```

继续生成 CAD 时，将目标改为 `cad`，并绑定已有 FreeCAD Python：

```powershell
.\scripts\run_photo_demo.ps1 `
  -Source 'C:\data\object_photos' `
  -Prompt 'object' `
  -Target cad `
  -FreecadPython 'C:\path\to\FreeCAD\bin\python.exe' `
  -Output '.\outputs\photo_demo\cad_demo01'
```

任务结束后可查询状态或只重新生成报告，不重新运行模型：

```powershell
$python='.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe'

& $python -B -X utf8 .\scripts\run_photo_demo.py status `
  --run '.\outputs\photo_demo\glove_demo01'

& $python -B -X utf8 .\scripts\run_photo_demo.py report `
  --run '.\outputs\photo_demo\glove_demo01'
```

主要参数：`Source` 为照片目录或视频，`Prompt` 为目标物体描述，`Frames` 为视频抽帧数
（2–96），`Target` 可选 `blender`、`report` 或 `cad`。`Output` 必须是尚不存在的新目录。
最终结果查看 `<输出目录>/report/index.html`，机器可读状态查看 `<输出目录>/pipeline.json`。

退出码：`0` 表示所选目标阶段完成；`1` 表示保留了部分成果；`2` 表示输入或准备错误；
`3` 表示分割环境或前置条件不满足；`4` 表示运行、证据、回收或报告错误；
`5` 表示资源限制拒绝；`130` 表示用户取消。

### 系统生成 Blender 文件

交互界面的侧栏新增 **Blender Export / 生成 Blender 文件**。选择本地 PLY，或点击
“使用当前已验收样例”，再点击“生成 Blender 文件”。系统在后台创建 `.blend`，
用独立 Blender 进程回读核对几何，通过后提供下载；无需手工打开 Blender 导入。
孔洞和面顺序保留，尺度未知的模型继续标注尺度未知。

也可通过 `scripts/export_blender.py --spec <场景清单> --output <新输出目录>` 调用同一后端。
需要本机已安装 Blender；不会自动安装或下载。具体参数和范围见
[Blender 导出说明](docs/BLENDER_EXPORT_USAGE.md)。这一功能消费已有 PLY，
不代表照片/视频的稠密重建已全部接入通用演示入口。

```bash
# 环境（已建好的隔离 venv）
PY=C:/Users/13944/.workbuddy/binaries/python/envs/kozlov/Scripts/python.exe

# 跑测试（约 10 秒）
$PY -m pytest -q

# 只跑 Gate
$PY -m pytest -q -m gate -s

# 现场演示（README 第 25 节第 1–7 步）
$PY experiments/mvp_walkthrough.py

# 一次性跑完所有可复现实验，并校验产物（写入 experiments/results/）
$PY experiments/run_all.py            # --quick 可缩短样本量

# 交互式界面（5 个页面）
$PY -m streamlit run app/Home.py
```

安装为包（可选）：

```bash
pip install -e .[dev]
```

照片与人工 mask CLI 使用独立的 `io` 可选依赖。需要照片 IO 时，在隔离环境安装：

```bash
python -m pip install -e ".[io]"
# 同时开发和运行 IO 定向测试时使用 ".[dev,io]"
python -m kozlov_lab.io --help
```

Pillow 暂固定为已验证的 `12.3.0`；未承诺其他版本兼容。R1 已在既有环境通过 213 项定向测试与合成 CLI，干净安装验证另列待办。运行照片任务前请读 [photo-io-v1 契约](docs/contracts.md)；当前接收状态和证据见 [R1 主线复核](docs/delegation/T01A_R1_REVIEW.md)。合成 IO 验收不等于真实照片或三维重建验收。

---

## 数学核心：一屏看懂

取图像三个不共线点 $A,B,C$，则任意点 $P$ 有唯一分解

$$P-A=\alpha(B-A)+\beta(C-A)
\quad\Longleftrightarrow\quad
\begin{pmatrix}\alpha\\\beta\end{pmatrix}=[\,B-A\ \ C-A\,]^{-1}(P-A).$$

对**任意**可逆仿射变换 $T(x)=Mx+t$：

$$\alpha(TP)=\alpha(P),\qquad\beta(TP)=\beta(P)$$

（标架随图像一起变换）。这就是可以充当图像编码的不变量。

**实测**（1000 次随机仿射变换，标架随图像变换）：

```text
max |alpha - alpha'| = 2.243e-14        required < 1e-9
max |beta  - beta' | = 2.109e-14        bound    = 1.655e-12
median |alpha - alpha'| = 6.661e-16     (machine precision)
```

详细推导与「为什么必须三点不共线」等教学解释：
[`docs/theory/pipeline.md`](docs/theory/pipeline.md)。

---

## 仓库结构

```text
kozlov-visual-lab/
├── src/kozlov_lab/
│   ├── config.py                 统一容差（FLOAT_TOL / COLLINEAR_TOL / MATCH_TOL）
│   ├── geometry/
│   │   ├── points.py             Point2D, PointSet2D（集合语义）
│   │   ├── transforms.py         AffineTransform（平移/等距/相似/仿射）
│   │   ├── affine_frame.py       AffineFrame, affine_coordinates  ← 项目核心
│   │   └── invariants.py         不变量测量（Invariant Explorer 引擎）
│   ├── precision.py              可达精度报告（条件数项 + 量级项）
│   ├── datasets/
│   │   ├── shapes.py             图形注册表（triangle/star/grid/…）
│   │   └── corpus.py             实验数据集 A–E（非对称性**实测**）
│   ├── compression/              ✅ 压缩（official эскиз/остов · teaching 基线 · experimental · control）
│   ├── teaching/                 ✅ 教学型编码（Pedagogical Extension，与正式实现隔离）
│   ├── coding/                   ✅ 正式编码 + 识别
│   │   ├── model.py exact.py numeric.py validation.py equivalence.py
│   │   ├── geometric.py          独立的精确几何 oracle（不依赖代码层）
│   │   ├── lazy.py               O(n^3) 内存的 lazy 表示（IndexedRatioView）
│   │   └── recognition.py        RecognitionResult + 两条识别流水线
│   └── stereo/                   ✅ 立体视觉（official 模型层 · baseline 多视图几何 · experimental 对应搜索）
├── app/                          Streamlit 界面
│   ├── Home.py                   演示故事线 + 状态
│   ├── _shared.py                共享 UI 组件（figures / 三态结论 / 精度块）
│   └── pages/
│       ├── 01_Point_Set.py       M1 点集实验室
│       ├── 02_Transformations.py M2 变换实验室
│       ├── 03_Affine_Coordinates.py  M3 内部仿射坐标（核心）
│       ├── 04_Teaching_Code.py   M4 教学型编码（Pedagogical construction）
│       ├── 05_Kozlov_Official_Code.py  M5 正式图像代码（Козлов 2009）
│       ├── 06_Invariants.py      M6 不变量实验中心
│       ├── 07_Official_Recognition.py  M7 正式仿射识别
│       ├── 08_Compression_Lab.py  M8 压缩实验室
│       └── 09_Stereo_Lab.py      M8 立体视觉实验室（Phase 6）
├── tests/                        561 个测试
├── experiments/                  可执行实验与演示
│   └── results/                  统一可复现产物 + MANIFEST.json（含 SHA-256）
├── examples/                     9 个教材示例 JSON
├── docs/
│   ├── specs/                    数学规格（带 status 与来源等级）
│   ├── theory/                   数学原理 + 数值极限
│   ├── experiments/              策展版实验报告
│   ├── terminology_ru_zh.md      中俄术语对照（唯一译法来源）
│   ├── status_report.html        可交互阶段报告
│   ├── demo_script.md            课堂演示流程
│   └── final_report.md           阶段报告
└── .sources/                     一手文献（不入库；见 docs/specs/official_image_code.md §1）
```

---

## 核心 API

```python
from kozlov_lab import (
    PointSet2D, AffineTransform, AffineFrame,
    affine_coordinates, invariant_report,
)

ps = PointSet2D.load("examples/square_3x.json")
t  = AffineTransform.shear(0.8, 0.0)
moved = t.apply_set(ps)

frame = AffineFrame.auto(ps.as_array())          # 自动选最不病态的标架
before = frame.coordinates_batch(ps.as_array())
after  = frame.transformed(t).coordinates_batch(moved.as_array())

print((after["alpha"] - before["alpha"]).max())  # ~1e-16
print(invariant_report(ps.as_array(), moved.as_array(),
                       kind=t.classify(), invariant_group=t.invariant_group()).markdown_table())
```

UI 层**不得**重新实现任何数学算法（README 第 18 节）。

---

## 已实测的关键数字

| 结论 | 数字 | 出处 |
|---|---|---|
| 内部坐标在仿射变换下不变 | $2.2\times10^{-14}$（要求 $<10^{-9}$） | [`docs/experiments/EXP-A.md`](docs/experiments/EXP-A.md) |
| 不变量表与 README 理论表一致 | 8 族 × 7 性质，逐格一致 | [`docs/experiments/M6-invariant-table.md`](docs/experiments/M6-invariant-table.md) |
| 控制组（无关图像）破坏全部性质 | 7/7 被破坏 | 同上 |
| $10^{-9}$ 的可达条件 | $\mathrm{cond}\lesssim10^{6}$ | [`docs/theory/numerical_limits.md`](docs/theory/numerical_limits.md) |
| 平移幅度导致的独立失效 | $\max\|\Delta\alpha\|\approx\varepsilon\|t\|/\|B-A\|$ | 同上 |
| 教学型编码在 5 数据集 × 8 变换族下不变 | 0 失败（含对照组） | `experiments/results/shape_generalization.csv` |
| 正式定义的字面 $S\ne0$ 判据**不可用** | $\max\rho=3.07\times10^{16}$，平移后变 $1.23\times10^{17}$ | [`docs/specs/official_image_code.md`](docs/specs/official_image_code.md) §13 |

---

## 方法论纪律

本仓库刻意执行以下约定；改动它们需要先改 `docs/specs/`：

| 约定 | 原因 |
|---|---|
| 规格先于实现，带 `status: verified / unverified / planned` | README 第 16 节：不允许猜数学定义 |
| 不变量表有 `preserved / violated / vacuous` 三态 | 「空集合相等」不能算"性质保持" |
| 每个"保持"断言配一个"破坏"断言 | 否则测试无法区分"实现正确"与"实现撒谎" |
| 容差集中在 `config.py`，判据无量纲化 | 固定绝对阈值在大/小坐标下必然出错 |
| 每个结果携带条件数与可达精度界 | README 第 6 节的 `1e-9` 只在良态标架成立 |
| 哈希只是索引，**不是**数学图像代码 | README 第 7、23 节 |
| 每次运行写 `provenance.json` | 让报告里的数字可追溯 |
| 每个"保持"断言必须配一个"破坏"断言 | 否则测试无法区分"实现正确"与"实现撒谎" |
| 哈希只是索引，**不是**数学图像代码 | README 第 7、23 节 |
| 教学型编码与正式实现**互不冒充**，由 AST 扫描测试强制 | 两种构造不同，命名混淆会让读者误以为是同一个对象 |

---

## 文档索引

| 想了解 | 看 |
|---|---|
| 数学主线与"为什么" | [`docs/theory/pipeline.md`](docs/theory/pipeline.md) |
| 形式代码的规模与实测增长（$O(n^6)$） | [`docs/theory/complexity.md`](docs/theory/complexity.md) |
| 精度边界与已知失效 | [`docs/theory/numerical_limits.md`](docs/theory/numerical_limits.md) |
| 各模块的数学规格与验收条件 | [`docs/specs/`](docs/specs/README.md) |
| 实验证据 | [`docs/experiments/`](docs/experiments/README.md) |
| 中俄术语 | [`docs/terminology_ru_zh.md`](docs/terminology_ru_zh.md) |
| 课堂演示流程 | [`docs/demo_script.md`](docs/demo_script.md) |
| 阶段报告与剩余路线 | [`docs/final_report.md`](docs/final_report.md) |

---

## 技术栈

Python 3.13 ｜ NumPy ｜ pytest ｜ Hypothesis
（Phase 7 将引入 Streamlit + Plotly）

第一版**不使用**深度学习框架——这是刻意的：
CNN 会把本项目要展示的数学机制藏起来。

---

## 不要做的事（README 第 23 节）

```text
× 把项目做成 OpenCV 图片滤镜
× 直接调用神经网络完成识别
× 用像素相似度代替数学等价
× 把哈希误称为数学不变量
× 没有证明就声称算法满足仿射不变性
```
