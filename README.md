# Kozlov Visual Perception Lab

**《视觉感知的数学理论》交互式实验与识别系统**
**Козлов：Математическая теория зрительного восприятия — 可视化数学实验室**

一个实现 В. Н. Козлов 教材数学思想的交互式实验环境。
它**不是** CNN/YOLO 项目，而是回答一个数学问题：

$$\text{机器如何从数学意义上判断两幅发生几何变化的图像属于同一个视觉对象？}$$

$$
\text{图像}\to\text{有限点集}\to\text{几何变换}\to\text{内部坐标}
\to\text{图像编码}\to\text{等价判定}\to\text{识别}
$$

---

## 当前状态

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 数学规格（Theory Agent 输出） | ✅ 完成 |
| 1 | 有限点集 + 几何变换 | ✅ 完成 |
| 2 | 内部仿射坐标（**项目核心**） | ✅ 完成，**Gate 通过** |
| 3 | 图像代码（正式，`coding/`） | ✅ **完成**（Phase 3 已解锁 2026-10-01） |
| 4 | 仿射等价识别（正式，`recognition.py`） | ✅ **完成**（Gate G4.1–G4.10 全通过） |
| 5 | 信息压缩（`compression/`） | ✅ **完成**（G5.1–G5.10 全通过；见 `docs/reports/phase5_gate.md`） |
| 6 | 立体视觉 | ⛔ 阻塞（未决问题 Q4） |
| 7 | Streamlit Web UI（M1–M8） | ✅ 完成 8 页（含编码、识别、压缩实验室） |
| 8 | 多形状不变量验证 + 统一可复现产物 | ✅ 完成 |
| 9 | 完整报告 + 答辩 Demo | ⏳ 阶段报告 |
| 10 | 2026 扩展模块 E1–E5 | ⛔ 未开始 |

**测试**：528 passed ｜ **实验**：7 个可复现实验全部通过 ｜ **教学型编码 / 正式编码 / 正式识别**：✅✅✅

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

**Phase 4 ✅ ⇒ Phase 5 解锁条件满足**（是否进入 Phase 5 待你确认）。

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
│   ├── recognition/              ⛔ Phase 4
│   ├── compression/              ⛔ Phase 5
│   └── stereo/                   ⛔ Phase 6
├── app/                          Streamlit 界面
│   ├── Home.py                   演示故事线 + 状态
│   ├── _shared.py                共享 UI 组件（figures / 三态结论 / 精度块）
│   └── pages/
│       ├── 01_Point_Set.py       M1 点集实验室
│       ├── 02_Transformations.py M2 变换实验室
│       ├── 03_Affine_Coordinates.py  M3 内部仿射坐标（核心）
│       ├── 04_Teaching_Code.py   M4 教学型编码（Pedagogical construction）
│       ├── 05_Kozlov_Official_Code.py  M5 正式图像代码（Козлов 2009）
│       ├── 05_Kozlov_Official_Code.py  M5 正式图像代码
│       ├── 06_Invariants.py      M6 不变量实验中心
│       ├── 07_Official_Recognition.py  M7 正式仿射识别
│       └── 08_Compression_Lab.py  M8 压缩实验室
├── tests/                        343 个测试
├── experiments/                  可执行实验与演示
│   └── results/                  5 个统一可复现产物 + MANIFEST.json
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
