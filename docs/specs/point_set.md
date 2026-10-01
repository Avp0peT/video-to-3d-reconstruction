---
spec: point_set
module: M1 有限点集图像实验室
status: verified
source_tier: S0
phase: 1
implemented_in: src/kozlov_lab/geometry/points.py
tests: tests/test_points.py
---

# 规格 M1：有限点集模型

## 1. 数学对象

二维图像定义为平面上的**有限非空点集**：

$$X=\{x_1,x_2,\ldots,x_n\},\qquad x_i\in\mathbb R^2,\quad n\ge 1 .$$

来源：README 第 4 节（S0）。

### 1.1 必须坚持的三条语义

1. **有限**：$n<\infty$，且 $n\ge 1$（空集不是图像）。
2. **集合**：$X$ 由它的元素决定，**与元素书写顺序无关**。
   README 验收条件原文：「点的排列顺序改变时，作为数学集合应视作同一输入对象」。
3. **可重复元素**：数学上 $X$ 是集合，因此原则上不允许重复点；但工程上必须
   接受重复点输入（README 第 19 节测试矩阵明确列出「重复点」），
   并提供显式的去重操作。

## 2. 实现约定

| 数学概念 | 实现 | 说明 |
|---|---|---|
| 点集 $X$ | `PointSet2D` | 不可变值对象；每个「修改」都返回新实例 |
| 点 $x_i$ | `Point2D` | 冻结 dataclass，含向量运算 |
| 集合相等 | `__eq__` → 量化后的 `frozenset` | 顺序无关、容差感知 |
| 集合指纹 | `signature()` | **忽略重数**（集合语义） |
| 重数指纹 | `multiset_signature()` | **保留重数**（识别阶段需要） |
| 规范顺序 | `sorted_points()` | 按 $(x,y)$ 字典序；供下游顺序相关算法使用 |

### 2.1 为什么需要两个指纹

- `signature()` 实现**集合**语义：$\{a,b,b\}=\{a,b\}$，满足 README 第 4 节要求；
- `multiset_signature()` 保留重数，因为 README 第 13 节要求研究
  「missing points / extra points」——若两者混同，多一个点的图像会被判为同一对象，
  该实验就无法测量。

两者**都必须存在**；只保留任一个都会违反 README 的某一条要求。

### 2.2 量化容差

`signature()` 把坐标吸附到 $10^{-\text{SIGNATURE\_DECIMALS}}$（默认 9）的整数格点上。
后果：跨越格点边界的两个"同一个点"会被判为不同。

已知局限，必须记录：**基于量化的相等在格点边界附近不可靠**。
需要精确几何比较时使用 `hausdorff_distance()`（顺序无关、与量化无关）。

### 2.3 顺序无关的几何比较

`hausdorff_distance(Y)` 定义为双向 Hausdorff 距离

$$d_H(X,Y)=\max\Big(\max_{x\in X}\min_{y\in Y}\|x-y\|,\ \max_{y\in Y}\min_{x\in X}\|x-y\|\Big).$$

当 $|X|=|Y|$ 且两集合的最小点间距大于该距离时，$d_H$ 小等价于存在双射——这正是
识别阶段需要的判据。

## 3. 序列化格式

```json
{
  "name": "triangle_demo",
  "points": [
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.5]
  ]
}
```

- `name`：字符串或 `null`（保留，便于 GUI 显示）；
- `points`：形如 `(n,2)` 的有限实数数组，**每行一个点**（便于人工编辑）；
- `meta`：可选，自由字典（GUI 保存变换信息时使用）。

**往返必须逐位精确**：`load(save(X))` 与 $X$ 的坐标逐位相等（不是"近似相等"）。
这一点由 `tests/test_points.py::test_json_round_trip_is_exact` 强制。

## 4. 验收条件（README 第 4 节）

| # | 条件 | 测试 |
|---|---|---|
| 1 | 任意有限非空点集可正确保存和恢复 | `test_json_round_trip_is_exact`（9 个图形参数化） |
| 2 | 改变点顺序后仍视作同一输入对象 | `test_reordering_points_preserves_identity` |
| 3 | GUI 与数学对象完全同步 | `test_mutating_helpers_never_touch_the_original`、`test_as_array_returns_a_copy` |

第 3 条通过**不可变值语义**实现：任何操作返回新对象，旧对象永不改变，
因此界面持有的引用与数学对象之间不可能出现"界面改了、模型忘了"的漂移。

## 5. 反例测试（必须有，否则验收可能是空的）

| 反例 | 目的 | 测试 |
|---|---|---|
| 平移 $10^{-3}$ 后的集合必须**不相等** | 防止 `__eq__` 退化为恒 `True` | `test_equal_size_is_not_enough_shifted_set_must_differ` |
| 与 `list` 比较返回 `False` | 防止类型混淆 | `test_equality_across_types_returns_notimplemented` |
| 空点集必须被拒绝 | 防止 $n=0$ 静默流入下游 | `test_single_point_is_legal_but_empty_is_not` |
| 非有限坐标必须被拒绝 | 防止 NaN 污染 | `test_invalid_inputs_are_rejected` |

## 6. 度量定义

| 量 | 定义 | 边界情况 |
|---|---|---|
| 质心 | $\frac1n\sum x_i$ | — |
| 包围盒 | $(\min x,\min y,\max x,\max y)$ | — |
| 直径 | $\max_{i,j}\|x_i-x_j\|$ | $n=1$ 时 $0$ |
| 特征尺度 | 直径，若 $\le$ `FLOAT_TOL` 则取 $1$ | 防止下游除零 |
| 一维性 | 中心化坐标矩阵第二奇异值与第一奇异值之比 $\le 10^{-9}$ | $n\le2$ 视为一维 |

「特征尺度」的存在是刻意的：README 第 19 节要求处理大坐标 / 小坐标，
所有相对判据都必须除以某个尺度，而该尺度在退化输入上必须良定义。
