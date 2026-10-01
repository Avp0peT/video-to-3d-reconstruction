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
| 3 | 图像代码 | ⛔ 阻塞（未决问题 Q1） |
| 4 | 仿射等价识别 | ⛔ 阻塞（Phase 3） |
| 5 | 信息压缩 | ⛔ 阻塞（Phase 4） |
| 6 | 立体视觉 | ⛔ 阻塞（未决问题 Q4） |
| 7 | Streamlit Web UI | ⛔ 未开始 |
| 8 | 实验 B–E + 测试矩阵 | ⛔ 部分 |
| 9 | 完整报告 + 答辩 Demo | ⛔ 部分 |
| 10 | 2026 扩展模块 E1–E5 | ⛔ 未开始 |

**测试**：219 passed ｜ **实验**：Experiment A ✅、M6 不变量表 ✅

> Phase 3 的阻塞有两层，且必须分开看：
> ① **工程阻塞**：Phase 2 的性质测试（Phase 2 Gate）——**已解除**；
> ② **数学阻塞**：教材正式编码定义尚未由一手资料确认（未决问题 Q1）——**未解除**。
> 只解除第一层就动手实现编码，等于「凭猜测实现教材算法」，被 README 第 23 节禁止。

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

# 实验（结果写入 results/<EXP-ID>_<日期>/）
$PY experiments/exp_a_affine_invariance.py
$PY experiments/exp_invariant_table.py
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
│   ├── datasets/shapes.py        图形注册表（triangle/star/grid/…）
│   ├── coding/                   ⛔ Phase 3（未实现，见 docs/specs/image_code.md）
│   ├── recognition/              ⛔ Phase 4
│   ├── compression/              ⛔ Phase 5
│   └── stereo/                   ⛔ Phase 6
├── tests/                        219 个测试
├── experiments/                  可执行实验与演示
├── examples/                     9 个教材示例 JSON
├── docs/
│   ├── specs/                    数学规格（带 status 与来源等级）
│   ├── theory/                   数学原理 + 数值极限
│   ├── experiments/              策展版实验报告
│   ├── terminology_ru_zh.md      中俄术语对照（唯一译法来源）
│   ├── demo_script.md            课堂演示流程
│   └── final_report.md           阶段报告
└── results/                      每次运行的 JSON 产物（含 provenance）
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
| `results/` 不入库，策展报告入库 | 原始产物可重生成，结论需要人工解读 |

---

## 文档索引

| 想了解 | 看 |
|---|---|
| 数学主线与"为什么" | [`docs/theory/pipeline.md`](docs/theory/pipeline.md) |
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
