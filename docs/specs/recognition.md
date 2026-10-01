---
spec: recognition
module: M5 仿射等价识别器
status: planned
source_tier: S0
phase: 4
implemented_in: null
blocked_by:
  - Phase 3 Gate（点排列无关性）
  - 未决问题 Q1
---

# 规格 M5：仿射等价识别（**尚未实现**）

## 1. 问题陈述

给定两个有限点集 $X,Y$，判定是否存在可逆仿射变换 $T$ 使 $T(X)=Y$。

README 第 8 节给出的 Demo：

```text
图像 X
↓ 随机 affine transform
图像 Y
系统：X ≠ Y（坐标不同）但 X ~aff Y

随机无关图像 Z： X ≁aff Z
```

## 2. 接口（冻结，供 GUI 与实验提前对接）

```python
result = affine_equivalent(X, Y)
```

```python
{
    "equivalent": true,
    "matrix": A,
    "translation": t,
    "point_mapping": ...,
    "residual": ...
}
```

实现时须补充两个字段（README 未要求，但实验 B 的方法 A/B 对比需要）：

```python
    "method": "code" | "control_points",
    "cost": {"triples_tested": int, "points_verified": int},
```

## 3. 必须实现两种方法

### 方法 A —— 利用图像代码比较

- 依赖 Phase 3 的 canonical code；
- 优点：一次编码可复用于多次比较；
- 缺点：编码本身的复杂度是 $O(n^4)$ 量级。

### 方法 B —— 枚举控制点三元组

- 选三对非共线对应点求解 $(A,t)$，再验证其余全部点；
- 朴素复杂度 $O\!\left(\binom{n}{3}^2\cdot n\right)$；
- 必须先做**必要条件剪枝**（例如按点数、按距离谱），否则不可用。

### 必须对比的三项（README 第 8 节）

| 指标 | 说明 |
|---|---|
| 正确率 | 同对象 → equivalent；异对象 → non-equivalent |
| 运行时间 | 按 $n$ 分层报告，不能只给单点值 |
| 点数增长带来的复杂度 | 拟合出指数并给出测量区间 |

## 4. 前置条件与风险

1. **点排列无关性**：方法与结果不得依赖点的书写顺序。
   这是 Phase 3 的 Gate，也是本模块的直接前提。
2. **精确 vs 近似必须分开**：README 第 13 节要求研究
   「完全等价识别和近似识别有什么区别？」，
   二者必须作为**两个不同接口**（`affine_equivalent` /
   `affine_approximately_equivalent`）并在报告中分列。
   混在一个函数里会让 Experiment D 的结论无法归因。
3. **方法的适用边界**：$n<3$ 或全共线时不存在唯一标架，
   必须显式返回"不可判定"，而不是 `False`。
4. **对应关系是未知的**：这是本模块与 `invariants.md` 的根本差别。
   不变量表的逐点比较用的是**已知**对应（实验室自己施加了变换）；
   识别问题的对应未知，正是难度所在。两者不得混用结论。
