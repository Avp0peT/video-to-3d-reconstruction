---
spec: transforms
module: M2 几何变换实验室
status: verified
source_tier: S0
phase: 1
implemented_in: src/kozlov_lab/geometry/transforms.py
tests: tests/test_transforms.py
---

# 规格 M2：几何变换

## 1. 四类变换（README 第 5 节，S0）

| 名称 | 作用 | 参数 | 保不变量 |
|---|---|---|---|
| 平移 | $x'=x+t$ | $t\in\mathbb R^2$ | 距离、角度、长度比例、共线性、平行性、仿射坐标 |
| 等距变换 | $x'=Rx+t$，$R$ 正交，$\det R=\pm1$ | 旋转角、反射 | 同上 |
| 相似变换 | $x'=sRx+t$，$s>0$ | $s$、旋转角、$t$ | 角度、长度比例、共线性、平行性、仿射坐标 |
| 仿射变换 | $x'=Ax+t$，$\det A\neq0$ | $A\in GL_2$、$t$ | 共线性、平行性、仿射坐标 |

**统一实现**：所有四类都是 `AffineTransform` 的特例，
`classify()` 判定一个具体实例属于哪一类，`invariant_group()` 给出粗分类。

这样做的理由：README 第 9 节要求不变量表必须由**实际数值实验**验证。
若为每一类写一个独立的类，就只能在界面上"声明"类别；
统一表示之后，界面拖动滑块产生的是一个 `AffineTransform`，
实验室**测量**它到底保持了哪些性质，而不是相信一个标签。

## 2. 分类规则

分类只看**线性部分**（因为决定不变量的是它），按从强到弱判定：

```
det A = 0                      -> "singular"
A A^T = I  且 A = I, t = 0     -> "identity"
A A^T = I  且 A = I            -> "translation"
A A^T = I  且 det A = +1       -> "rotation"
A A^T = I  且 det A = -1       -> "reflection"
A A^T = s^2 I（s = sqrt|det A|）-> "similarity"
否则                            -> "affine"
```

判据中的 $A A^T=I$ 使用绝对容差 `1e-9`。
注意**非等比缩放不是相似变换**：`AffineTransform.scaling(2.0, 0.5)` 分类为 `"affine"`，
这一点由 `test_nonuniform_scaling_is_not_a_similarity` 固定。

> 为什么分类必须基于线性部分：绕远点旋转的平移向量可以任意大，
> 但它仍然是等距变换。反之，一个"看起来很小"的剪切可以彻底破坏角度。

## 3. 复合与逆

- 复合约定为矩阵式：$(f\otimes g)(x)=f(g(x))$，即 `g` 先作用；
- 逆变换必须用 `np.linalg.solve`，**禁止**直接用 `np.linalg.inv`（README 第 19 节）：

```python
A_inv = np.linalg.solve(A, np.eye(2))
```

- `is_invertible` 的判据是相对化的：$|\det A|>\text{FLOAT\_TOL}\cdot\sigma_{\max}^2$。
  绝对阈值在这里是错的——大坐标图像的"接近奇异"与小坐标图像的
  同一个几何形态，其行列式可以相差 $10^{12}$ 倍。

## 4. 形状约定

`AffineTransform.apply` 的输出形状**跟随输入形状**：

| 输入 | 输出 |
|---|---|
| `Point2D`、`[x, y]`、`(2,)` 数组 | `(2,)` |
| 点集（`(n,2)`、`PointSet2D`、`[[x,y],...]`） | `(n,2)` |

向量化核心 API `apply_affine` 则**恒为** `(n,2)`。
两条约定并存是刻意的：批量接口需要形状稳定，而逐点接口若把单点变成 `(1,2)`，
会让 `frame.transformed(...)` 这类代码在形状上出错。

## 5. 随机变换生成器（Experiment A/B 的基底）

`AffineTransform.random_invertible` 由**奇异值**构造，而不是抽取四个独立矩阵元素：

$$A=R_2\,\mathrm{diag}(s_1,s_2)\,R_1$$

理由：$\mathrm{cond}(A)$ 正是决定内部坐标可达精度的量。
若直接抽元素，相当比例的随机矩阵接近奇异，Experiment A 会悄悄变成
一个关于病态条件的实验。

### 5.1 条件数预算必须靠钳制奇异值实现

第一版实现用「把整个矩阵除以标量」来控制条件数——**这是错的，而且不报错**：

$$\mathrm{cond}(cA)=\frac{c\,\sigma_{\max}}{c\,\sigma_{\min}}=\mathrm{cond}(A).$$

标量对所有奇异值等比例缩放，条件数完全不变。
正确的做法是钳制较小奇异值：

$$\sigma_2\leftarrow\max\Big(\sigma_2,\ \frac{\sigma_1}{\text{max\_condition}}\Big).$$

该缺陷由 `test_random_invertible_respects_condition_budget` 发现
（实测条件数 55.4 > 预算 50），
`test_condition_budget_holds_exactly_for_every_budget` 对
$\{1, 1.5, 3, 10, 1000\}$ 五个预算逐一固定。

## 6. 由三对控制点确定仿射变换

给定源三点 $(A,B,C)$ 与目标三点 $(A',B',C')$，唯一仿射变换为

$$M=[B'-A'\ \ C'-A']\,[B-A\ \ C-A]^{-1},\qquad t=A'-MA .$$

实现使用 `np.linalg.solve(S^T, D^T)^T` 而不是显式求逆。
源三点共线时无唯一解，必须**抛错**而不是返回数值垃圾。

这是 README 第 8 节「方法 B」的基础，也是 `invariants._infer_kind()`
（GUI 手动编辑模式下推断变换类型）的依据。

## 7. 验收条件与对应测试

| README 要求 | 测试 |
|---|---|
| 等距变换：$\|x_i-x_j\|=\|x'_i-x'_j\|$ | `test_rotation_preserves_all_distances` |
| 相似变换：长度比例保持 | `test_similarity_preserves_angles`、`test_length_ratio_change_is_scale_free` |
| 仿射：$\det A\neq0$，共线性、平行性保持 | `test_shear_breaks_angles_but_keeps_collinearity_and_parallelism` |
| 正方形 → 仿射 → 平行四边形 | 同上（`square_3x` 夹具） |
| 界面参数可调 | `parameters()` 返回 UI 需要的 11 个字段 |

## 8. 反例测试

README 只要求验证"保持"，但只验证保持的测试无法区分「实现正确」与「实现撒谎」。
因此每个"保持"断言都配一个"破坏"断言：

| 性质 | 保持于 | 破坏于（反例测试） |
|---|---|---|
| 距离 | 旋转 | `test_nonuniform_scaling_breaks_distances...`（缩放 3×1） |
| 角度 | 相似 | `test_shear_breaks_angles_...`（剪切，实测变化 > 0.1 rad） |
| 共线性 | 剪切 | `test_affine_map_does_not_preserve_collinearity_when_it_is_not_affine` |
| 逆变换 | 9 类变换 | `test_singular_transform_cannot_be_inverted` |
| 控制点求解 | 一般仿射 | `test_from_control_points_rejects_collinear_source` |
