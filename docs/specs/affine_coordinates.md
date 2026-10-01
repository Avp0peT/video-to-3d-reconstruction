---
spec: affine_coordinates
module: M3 内部仿射坐标
status: verified
source_tier: S0
phase: 2
implemented_in: src/kozlov_lab/geometry/affine_frame.py
tests: tests/test_affine_frame.py, tests/test_numerical_stability.py, tests/test_gate_phase2.py
gate: 本规格的验收通过是进入 Phase 3 的**硬性前提**
---

# 规格 M3：内部仿射坐标（项目核心）

README 第 6 节称本模块为「整个项目最重要的模块」。

## 1. 定义

取三个**不共线**点 $A,B,C$，则平面上任意点 $P$ 存在唯一分解

$$P-A=\alpha\,(B-A)+\beta\,(C-A),$$

等价地

$$\begin{pmatrix}\alpha\\\beta\end{pmatrix}
=[\,B-A\quad C-A\,]^{-1}(P-A).$$

$(\alpha,\beta)$ 称为 $P$ 关于标架 $(A,B,C)$ 的**内部仿射坐标**。
来源：README 第 6 节（S0）。

## 2. 存在唯一性条件

$$\det[\,B-A\quad C-A\,]\neq 0
\quad\Longleftrightarrow\quad A,B,C\ \text{不共线}.$$

### 2.1 判据必须是无量纲的

$$\det[B-A,\ C-A]\ \text{是}\ \textbf{面积} \Rightarrow \text{随图像尺度平方变化}.$$

因此**不能**使用固定阈值。本项目使用

$$\text{flatness}(A,B,C)=\frac{2\,\mathrm{Area}}{\max(\text{边长})^2}
=\frac{h_{\min}}{\max(\text{边长})},$$

即「最短高 ÷ 最长边」，无量纲，且对三个点**对称**。

判据：`flatness > COLLINEAR_TOL`（默认 $10^{-9}$）。

### 2.2 ⚠️ 为什么不能用 $|\sin(\angle BAC)|$ 作为关系判据

$|\sin(\angle BAC)|$ **依赖于哪个点被命名为 $A$**：同一个三角形的三个顶点正弦值不相等
（90-45-45 三角形为 $1,\ 0.707,\ 0.707$）。对近乎共线的三角形，三者可以相差许多数量级。

具体反例（由 Hypothesis 在 Phase 2 发现）：

$$a=(0,0),\quad b=(3.47370303\times10^{-152},\,0),\quad c=(0,1)$$

$$|\sin(\angle BAC)|=1.0,\qquad |\sin(\angle BCA)|\approx3.5\times10^{-152}.$$

若用 $|\sin(\angle BAC)|>10^{-9}$ 定义"三点不共线"，则同一组点在换名后会得到**相反**的判定。
对依赖"枚举所有共线三元组"的图像代码层（Phase 3）而言这是致命的。

结论：

| 量 | 用途 | 是否置换不变 |
|---|---|---|
| `collinearity_sine(a,b,c)` | 诊断**具体标架** $(A,B,C)$ 的条件数 | ✗ |
| `triangle_flatness(a,b,c)` | 判定**共线关系**、标架合法性 | ✓ |
| `min_vertex_sine(a,b,c)` | 对称的正弦下界 | ✓ |

回归测试：`test_vertex_sine_is_not_permutation_invariant_flatness_is`。

## 3. 核心不变性

对任意可逆仿射变换 $T(x)=Mx+t$：

$$\alpha(TP)=\alpha(P),\qquad \beta(TP)=\beta(P).$$

**标架随图像一起变换**，即计算 $T(X)$ 的内部坐标时使用 $(TA,TB,TC)$。
这是定理成立的形式；若固定原标架去算 $T(X)$，结论不成立（那已不是同一个标架）。

### 3.1 为什么成立（README 第 21 节要求给出「为什么」）

设 $u=B-A,\ v=C-A$，则 $P-A=\alpha u+\beta v$，于是

$$TP-TA=M(P-A)=\alpha\,(Mu)+\beta\,(Mv)=\alpha\,(TB-TA)+\beta\,(TC-TA).$$

由于 $M$ 可逆，$Mu,Mv$ 仍线性无关，故 $(TA,TB,TC)$ 仍是合法标架，
且由唯一性知 $(\alpha',\beta')=(\alpha,\beta)$。∎

### 3.2 为什么要求三点不共线（README 第 21 节原问题）

若 $B-A$ 与 $C-A$ 线性相关，则它们只张成一条直线，无法构成二维仿射坐标系，
$(P-A)$ 中垂直于该直线的分量无法表示，解不唯一或不存在。

## 4. 数值精度：必须与条件数一起报告

线性方程组的经典误差界为

$$\frac{\|\delta z\|}{\|z\|}\le \mathrm{cond}(M)\,\varepsilon_{\text{machine}},\qquad M=[\,B-A\quad C-A\,].$$

因此 README 第 6 节的「误差 < 1e-9」**只有在标架良态时才是有意义的断言**。
一个 flatness $=10^{-8}$ 的标架有 $\mathrm{cond}\approx2\times10^{8}$，
float64 下**不可能**达到 $1e-9$。

因此实现规定：

1. 每个返回值都携带 `condition_number`、`residual` 与 `accuracy_bound`；
2. `tolerance_for_condition(cond) = max(FLOAT_TOL, 64·cond·ε)` 给出**该标架实际可达**的精度；
3. 所有测试断言 `measured <= reported_bound`，而不是硬编码 `1e-9`。

实测（Experiment A，见 `docs/experiments/EXP-A.md`）：

| flatness | cond | 实测 max\|Δ\| | 报告界 | 满足 1e-9？ |
|---|---|---|---|---|
| $10^{-2}$ | $2.0\times10^{2}$ | $1.71\times10^{-14}$ | $2.84\times10^{-12}$ | 是 |
| $10^{-4}$ | $2.0\times10^{4}$ | $1.57\times10^{-12}$ | $2.84\times10^{-10}$ | 是 |
| $10^{-6}$ | $2.0\times10^{6}$ | $1.74\times10^{-10}$ | $2.84\times10^{-8}$ | 是 |
| $10^{-8}$ | $2.0\times10^{8}$ | $1.35\times10^{-8}$ | $2.84\times10^{-6}$ | **否** |
| $5\times10^{-9}$ | $4.0\times10^{8}$ | $1.39\times10^{-8}$ | $5.68\times10^{-6}$ | **否** |

**结论**：`1e-9` 是良态标架的结论，且库永远不会在病态标架上**声称**它。
详细分析见 [`../theory/numerical_limits.md`](../theory/numerical_limits.md)。

## 5. 求解实现约束

- 必须用 `np.linalg.solve`，**不得**显式求逆（README 第 19 节）；
- 近共线检查必须在求解**之前**完成，否则把"解不出"伪装成"解出来了"；
- 批量接口 `affine_coordinates_batch` 一次校验、一次分解，因为 M3/M4 会
  对图像的每个三元组重复调用它。

## 6. 标架选择

`AffineFrame.auto(points)` 不取"第一个合法三元组"，而是取使

$$\text{score}=\frac{\mathrm{cond}(M)}{\text{flatness}}$$

最小的三元组（最多扫描 2000 个候选以保持交互响应）。
理由：所有下游数值精度由 $\mathrm{cond}(M)$ 决定，而 $\mathrm{cond}$ 完全由选哪三个点决定。
对细长的图形（如 `letter_L`）这一选择会带来数量级的差别。

## 7. 验收条件

| # | 条件 | 测试 |
|---|---|---|
| 1 | $\det[B-A,C-A]\neq0$ 检测正确 | `test_is_non_collinear_thresholding`（6 个阈值点） |
| 2 | 标架三点映射为 $(0,0),(1,0),(0,1)$ | `test_frame_points_have_unit_coordinates` |
| 3 | 变换前后 $(\alpha,\beta)$ 不变，误差 < 1e-9 | `test_internal_coordinates_are_affine_invariant`（9 类变换）、Experiment A（1000 次） |
| 4 | 共线标架必须**抛错**，不得返回数值 | `test_collinear_frame_is_rejected_not_silently_answered` |
| 5 | 批量与单点结果一致 | `test_batch_and_single_agree` |
| 6 | 与重建互逆 | `test_reconstruct_is_the_inverse_of_coordinates` |

### 7.1 Phase 2 Gate（README 第 24 节）

> Phase 2 没有通过性质测试，不允许进入 Phase 3。

Gate 的可执行形式是 `tests/test_gate_phase2.py`，包含：

- Experiment A 主臂（1000 次随机仿射，实测 worst $=2.24\times10^{-14}$）；
- **非平凡性检查**：worst 必须 $>0$（否则说明根本没在测量）；
- **可证伪性检查**：病态标架下必须**超过** 1e-9，证明该阈值具有区分力；
- 逐行理论界比较 `Δ ≤ cond·ε·64`；
- 点顺序无关性（25 次随机置乱，逐个索引对应）。

## 8. 已知局限（必须报告，不得隐藏）

见 `docs/theory/numerical_limits.md`。摘要：

1. **标架条件数界**：$1e-9$ 需 $\mathrm{cond}\lesssim 10^{6}$；
2. **平移幅度界**：$\max|\Delta\alpha|\approx \varepsilon\,|t|/|B-A|$，
   该失效模式**不在**条件数界内（信息在输入阶段就丢了）；
3. **图像结构摧毁阈值**：当 $\varepsilon|t|>|B-A|$ 时，平移后的图像所有点塌缩到同一个
   float，标架退化，库会**抛错**而不是返回看似合理的噪声。
