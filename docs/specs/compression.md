---
spec: compression
module: M7 图像信息压缩
status: planned
source_tier: S0（算法层）/ S1（教材算子，待确认）
phase: 5
implemented_in: null
blocked_by:
  - 未决问题 Q3（教材「сжатие информации」使用的具体算子）
---

# 规格 M7：图像信息压缩（**尚未实现**）

## 1. 目标

README 第 10 节：生成 $X\to S(X)$ 的教学型简化，并研究

> 到什么程度的信息压缩仍能保留足够的识别结构？

## 2. 计划实现的算子（优先 $\varepsilon$-cover）

| 算子 | 定义 | 备注 |
|---|---|---|
| $\varepsilon$-cover | 最小/近最小子集 $S\subseteq X$，使 $\max_{x\in X}\min_{s\in S}\|x-s\|\le\varepsilon$ | **优先** |
| greedy representative points | 贪心最远点 | — |
| grid reduction | 按网格合并 | — |
| farthest-point sampling | 最远点采样 | — |

接口草案：

```python
compress_point_set(points, epsilon) -> {
    "sketch": PointSet2D,
    "compression_ratio": 1 - |S(X)| / |X|,
    "covering_radius": float,
    "method": str,
}
```

## 3. ⚠️ 必须验证的陷阱：覆盖半径

**压缩率本身没有意义**，除非同时验证 $S(X)$ 真的在 $\varepsilon$ 内覆盖了 $X$。
一个"把点删掉 90%"的实现总能报告很漂亮的压缩率，
但若剩余点根本没覆盖原集合，Experiment D 的全部结论都是假的。

因此实现必须：

1. 返回 `covering_radius`；
2. 断言 `covering_radius <= epsilon`（数值上留 $1+10^{-9}$ 的松弛）；
3. 该断言必须在测试中**存在反例**：故意构造一个已知超出 $\varepsilon$ 的 sketch，
   确认测量会报"不覆盖"。没有反例的覆盖断言是恒定值假通过。

## 4. 计划实验（Experiment D）

改变 $\varepsilon$，记录：

```text
剩余点数 / 压缩率 / 识别成功率 / 匹配误差
```

输出曲线：

- $\varepsilon$ vs 点数；
- $\varepsilon$ vs 识别质量。

**方法论要求**：识别质量的分母必须显式声明是 measurement 分母还是 sampling 分母，
且每个 $\varepsilon$ 点至少 3 次重复（README 未明写，与本项目已建立的口径一致）。

## 5. 与 Phase 4 的依赖

Experiment D 需要 Phase 4 的识别器。压缩后的点集与压缩后的变换版本比较，
其结论的可解释性完全依赖识别器本身是否可靠。
因此 **Phase 5 不得早于 Phase 4 完成**——README 的开发顺序已经体现了这一点。
