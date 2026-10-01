---
spec: image_code
module: M4 图像代码
status: unverified
source_tier: S0 (仅"教学型编码器"部分)
phase: 3
implemented_in: null  ← 尚无实现；被 Phase 2 Gate 与未决问题 Q1 双重阻塞
tests: null
blocked_by:
  - Phase 2 Gate（README 第 24 节）
  - 未决问题 Q1：教材正式编码定义的精确形式
---

# 规格 M4：图像代码（**尚未实现**）

## 1. 当前状态：明确不做的事

本规格**故意**只写到这里就停。理由有两层，二者独立且都足以阻塞实现。

### 1.1 硬阻塞：Phase 2 Gate

README 第 24 节：

> Phase 3 无法保证点排列无关性，不允许进入 Phase 4。
> （前置条件）Phase 2 没有通过性质测试，不允许进入 Phase 3。

Phase 2 的 Gate 现已通过（见 `docs/experiments/EXP-A.md`），
但 Gate 通过只解除**工程**阻塞，不解除下面的**数学**阻塞。

### 1.2 数学阻塞：未决问题 Q1

README 第 26 节：

> 任何涉及 Козлов 理论具体定义、定理或编码形式的内容，在编码之前必须由
> Theory Agent 从可靠的一手资料验证。

一手资料（S1：Козлов 2026；S2：Козлов 2009）**尚未在本仓库中核实**。
在 Q1 解决前，实现教材的正式编码形式就是「凭猜测实现教材算法」，
被 README 第 23 节明确禁止。

## 2. 已确定的部分（可以实现的"教学型编码器"）

README 第 7 节本身已经给出了足够的形式化描述，这一部分**不是猜测**，
而是 README 的明文规定，可以照此实现（前提是 Q1 未解决时必须在 UI 上标注
「教学型，非教材正式定义」）：

### 2.1 编码流程

输入 $X=\{x_1,\ldots,x_n\}$：

1. 枚举所有**有序**非共线三元组 $(A,B,C)$ 作为基准标架；
   （实现层用 `iter_non_collinear_triples`，$i<j<k$ 顺序，
   合法性判据用 `triangle_flatness` 而非顶点正弦——见 `affine_coordinates.md` §2.2）
2. 对每个标架，计算全部点的内部坐标 $(\alpha_i,\beta_i)$；
3. 构造 canonical descriptor：

```python
AffineFrameCode:
    frame
    normalized_points
```

4. 汇总：

```python
ImageCode:
    frame_codes[]
```

### 2.2 消除点编号影响（README 第 7 节的五步）

1. 对内部坐标排序；
2. 对浮点数按 tolerance 量化；
3. 对所有合法 frame code 做规范排序；
4. 构造 canonical representation；
5. 可进一步计算 SHA-256 **仅用于快速索引**。

### 2.3 ⚠️ 必须写入用户文档的一句话

> **哈希不是数学图像代码本身，只能是代码的计算机索引。**

README 第 7 节与第 23 节都点名了这一点（「把哈希误称为数学不变量」被列为
"必须避免的问题"）。这句话必须同时出现在：
① 本规格；② 用户文档；③ 界面上哈希值旁边。

## 3. 必须由 Theory Agent 确认的问题（Q1 细化）

| # | 问题 | 影响 |
|---|---|---|
| Q1.1 | 教材的 $C(X)$ 是否基于"全部三元组"？是否存在选择的公理/判据？ | 决定算法复杂度与是否 NP-hard |
| Q1.2 | 编码是否要求**有序**三元组，还是无序？ | 决定 frame code 的数量因子（$\times6$） |
| Q1.3 | 编码中的量化步长在教材中是否有定义？ | 决定 $C(X)$ 的等价类粗细 |
| Q1.4 | 教材的等价判定用的是编码相等，还是编码在某距离下相近？ | 决定 Phase 4 方法 A 的实现 |
| Q1.5 | 是否存在"用三个以上点确定标架"的变体？ | 影响对噪声的稳健性 |

## 4. 待确认后才能写入的实现接口（草案，非最终）

```python
build_frame_code(points, frame) -> AffineFrameCode
build_image_code(points) -> ImageCode
canonical_form(image_code, decimals) -> CanonicalCode      # 顺序无关
code_hash(canonical_code) -> str                           # 索引，不是不变量
codes_equivalent(code_a, code_b, tol) -> bool
```

复杂度预警：$\binom{n}{3}$ 个标架，每个标架 $O(n)$ 计算 `solve`，
故朴素实现为 $O(n^4)$。$n=20$ 时约 $1.1\times10^5$ 次 $2\times2$ 求解，
可接受；$n=100$ 时已不可接受。**这一点必须在实现前与 Theory Agent 确认**
是否存在教材规定的剪枝，而不是由工程侧自行发明剪枝规则。

## 5. 完成标准（README 第 26 节）

```text
implementation  +  tests  +  demo  +  documentation
```

四者齐备才可标记 Done。当前：**0 / 4**。
