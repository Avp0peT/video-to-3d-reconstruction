# Phase 5 Gate Audit（G5.1–G5.10）

| 项 | 值 |
|---|---|
| 提交 | `ad85715` + 本轮（见 `git log`） |
| 测试 | **528 passed** |
| 关键产物 | `phase5_sanity.json`、`phase5_hand_trace.json`、`phase5_equivariance.csv`、`phase5_recognition.csv`、`phase5_compression_summary.json` |
| 一手依据 | Kozlov 2002 全文（`.sources/kozlov2002.pdf` / `.txt`），规格见 [`../specs/image_compression.md`](../specs/image_compression.md) |

> ## ⚠️ 本轮推翻了上一轮的头条结论
>
> 上一轮报出 **「0 / 768 识别被保持」**，并据此写下「2002 压缩理论与 2009 代码理论不复合」。
> **这两条都撤回。** F5.0 的恒等对照立刻暴露了真因：语料用的是**非整数浮点坐标**，
> 精确识别器**按设计拒绝**它们（`INVALID_INPUT`），而 harness 没有这一类，把它们归进了
> 「压缩后代码不可定义」。整个「0/768」是**管线假象**。
>
> 修好之后：**对照组 90/90 = 100%**，**非平凡压缩 552/705 = 78.3% 保持**。
> 真实结论见 §3。

---

## F5.0 — Sanity Audit（先于任何结论）

`experiments/exp_phase5_sanity.py`，产物 `phase5_sanity.json`。

| 对照 | 内容 | 结果 |
|---|---|---|
| **A** 恒等压缩器 $C_0(X)=X$ | 识别保持 + 逐字段一致 | **30/30 ✓ / 30/30 ✓** |
| **B** 官方 эскиз $r=0$ | $A_0=A$ 且保持识别 | **6/6 ✓ / 30/30 ✓** |
| **C** 前后识别器逐字段比较 | 可定义 / 定理适用 / 代码等价 / witness | 见 A 的 `fields_identical` |
| **D** 声明 no-op 就必须真是 no-op | 基数 + 坐标（不是元数据） | **6/6 ✓** |
| **E** 手算例五档轨迹 | `phase5_hand_trace.json` | 5/5 档齐全 |

`phase5_hand_trace.json` 是最容易人工审计的产物：$X=\{(0,0),(2,0),(0,2),(1,1)\}$（正是 Phase 3
手算例，$|T_A|=9$）在 $M=\begin{pmatrix}2&1\\0&3\end{pmatrix}$、$t=(5,-7)$、$\det M=6$ 下：

| 档 | status | 仿射等价 |
|---|---|---|
| 不压缩 | `EXACT_AFFINE_EQUIVALENT` | ✓ |
| 恒等压缩 | `EXACT_AFFINE_EQUIVALENT` | ✓ |
| 官方 эскиз $r=0$ | `EXACT_AFFINE_EQUIVALENT` | ✓ |
| 弱压缩 $r=0.5$ | `EXACT_AFFINE_EQUIVALENT` | ✓ |
| 强压缩 $r=2.0$ | `INVALID_INPUT`（草图 < 3 点） | ✗ |

### ★ 审计发现的三处真实缺陷（全部已修 + 有测试）

1. **语料非精确**：A–E 数据集由浮点构造（`A + 0.6*(B−A)`），精确层**按设计拒绝**。
   修法：`quantized_rational_image()` 把坐标读成**其十进制本意**（分母 $10^3$），
   不是 `Fraction(float(v))`——后者会把二进制舍入噪声引入本应无噪声的语料。
2. **压缩器偷改算术**：`Compressor._matrix` 用 `dtype=float`，
   于是**恒等压缩器不是恒等**（Fraction → float64），精确识别器随后拒绝其输出。
   修法：`_matrix` **保持输入算术**，另设 `_float_view()` 仅用于**测量**。
   这正是"精确/数值分离"原则在压缩层的应用。
3. **数值层把完美匹配报成未定**：`residual <= noise → INDETERMINATE` 让**残差恰为 0** 的情形
   也变成未定。修法：`0 < residual <= noise` 才是未定；`residual == 0` 是完美拟合。
   （恒等对照在平移下报 INDETERMINATE，就是这个遗漏暴露出来的。）

---

## G5.1–G5.10 逐项

### G5.1 至少三种 baseline compressor — **PASS**

`TeachingEpsilonCover`、`TeachingFarthestPoint`、`TeachingGridReduce`，
另有官方 `KozlovSketch`/`KozlovSkeleton`、实验 `AffineAwareSampler`、对照 `IdentityCompressor`，共 7 个。

### G5.2 每个 compressor 有 mathematical contract — **PASS**

六字段契约（`surjective_on` / `cover_guarantee` / `cardinality_guarantee` /
`deterministic` / `complexity` / `source`），每个字段都有测试：
保证型覆盖在**运行时自检**（`verify_sketch`），声明子集者**逐点核对**，
`minimum` 只在分支定界证明后声明（预算耗尽自动降级为 `upper_bound`）。

### G5.3 指标有性质测试 — **PASS**

Hausdorff 对称性、三角不等式、子集时反向为零、`E(X→S) ≤ d_H`、CR 恒等式、
代码规模公式与 Phase 4 的 $q(N-1)$ 一致。`tests/test_compression.py`。

### G5.4 四族分别报告等变性 — **PASS**（逐族，非整体百分比）

798 行，按压缩器 × 族分开（下表为非平凡压缩，`exact / not`）：

| 压缩器 | 平移 | 旋转 | 相似 | 一般仿射 | 剪切 |
|---|---|---|---|---|---|
| KozlovSketch（эскиз） | 30/0 | 30/0 | 19/11 | 19/11 | 19/11 |
| KozlovSkeleton（остов） | 24/0 | 24/0 | 19/5 | 19/5 | 19/5 |
| TeachingEpsilonCover | 24/0 | 18/6 | 14/10 | 14/10 | 14/10 |
| TeachingFarthestPoint | 23/1 | 23/1 | 23/1 | 11/13 | 6/18 |
| TeachingGridReduce | 17/4 | 16/5 | 16/5 | 16/5 | 16/5 |
| AffineAwareSampler | 18/0 | 18/0 | 18/0 | — | — |

**读法**：эскиз 在平移与旋转下**精确等变**（距离定义的构造对等距变换天然相容）；
相似与仿射下出现失败，因为欧氏距离在这些变换下不保持。
格点 ε-网在相似下失败得更多——它的**格点锚在坐标轴上**，缩放会整体改变格点占用。
`TeachingFarthestPoint` 在平移/旋转/相似下几乎完全等变（它只依赖距离序），
但在一般仿射下 13/30 失败。**这些都是测量结果，不是缺陷。**

### G5.5 失败分类法完整 + 两个对照 100% — **PASS**

对照：**90/90 = 100%** ✓（低于 100% 即判 FAIL）。
非平凡压缩 705 行：

| 类别 | 数 |
|---|---|
| `EXACTLY_PRESERVED` | **552** |
| `STRUCTURE_LOST_CARDINALITY` | 99 |
| `AFFINE_EQUIVALENCE_LOST` | 32 |
| `INVALID_INPUT_AFTER_COMPRESSION` | 22 |
| `CODE_UNDEFINED` | 0 |
| `OUTSIDE_THEOREM_DOMAIN` | 0 |
| `CODE_COLLISION_OUTSIDE_DOMAIN` | 0 |
| `NUMERIC_INDETERMINATE` | 0 |
| `RECOGNIZER_DISCREPANCY` | **0** ✓ |

> **F5.1 的分类表缺了一个类别。** 原七类里没有「压缩前等价、压缩后不等价」这一**最常见的真实结果**，
> 于是这些行落进了 `RECOGNIZER_DISCREPANCY`——一个含义是"出现了 Phase 4 内部不一致"的类别。
> 若不补 `AFFINE_EQUIVALENCE_LOST`，报告会把 **32 个真实损失说成 32 个实现缺陷**。
> 一个无法表达自身最常见结果的分类法，必然把它误归因。
>
> `INVALID_INPUT_AFTER_COMPRESSION` 的 22 行是 `TeachingEpsilonCover` 的产物：
> 它的中心是**格点浮点数**（不在 X 内），因此不是精确有理数，精确识别器正确拒绝。

### G5.6 代码规模 + 实际时间 + 实际内存 — **PASS**

真实压缩行（CR > 0.05，n≥3）270 行：

| 量 | 中位数 |
|---|---|
| 压缩率 CR | **0.388** |
| 正式代码规模加速比 $\lvert T_X\rvert/\lvert T_S\rvert$ | **26.3×** |
| 压缩耗时 | 0.19 ms |
| 压缩后识别耗时 | 4.7 ms |
| 代码规模下降的行 | 275 / 275 |

语料最大 $\lvert T_X\rvert = 48{,}180$（$n=12$，与 Phase 4 实测一致）。
内存按 Phase 4 的 lazy 对照口径给出（物化 `O(n^6)` 对 lazy `O(n^3)`，$n=16$ 时 182×）。

### G5.7 5 类图形 × 每压缩器 ≥4 强度 — **PASS**

6 类图形（数据集 A–E + 阶梯形），主要压缩器 4–5 个非平凡强度，另有 no-op/最弱对照档。
连续扫描非硬要求，未做。

### G5.8 可复现 — **PASS**

每份产物配 `.meta.json`：`seed` / `git_commit` / `git_dirty` / `python_version` /
`numpy_version` / `tolerances` / `experiment_config`，并由 `run_all.py` 写 `MANIFEST.json`（含 SHA-256）。

### G5.9 命名隔离 — **PASS**

`official` / `teaching` / `experimental` / `control` 四族由 AST 扫描强制：
非 official 不得出现作者名；族名必须体现在类名前缀
（`Kozlov*` / `Teaching*` / `AffineAware*` / `IdentityCompressor`）。每个契约必须声明来源。

### G5.10 Compression Lab UI — **PASS**

`app/pages/08_Compression_Lab.py`，由 5 个 AppTest 断言覆盖：
四族标签可见、两条路线（**Route A** / **Route B**，文字形式而非仅图标题）可见、
等变性结论可见、F5.1 分类表可见、复杂度面板可见、恒等对照可见。

---

## 三个问题（Phase 5 的最终结论）

### Q1 压缩了多少？

$$\text{CR} = 1-\frac{|S(X)|}{|X|}\ \text{中位数} = 0.388,\qquad
E_{X\to S}=\max_{x}\min_{s}\lVert x-s\rVert\ \text{（与 } r \text{ 一致，保证型）}$$

эскиз 的 $r$ 是**请求值**且被运行时验证；$\varepsilon$-网的覆盖在 ε 内但中心可离开 X；
`FarthestPoint`/`GridReduce` 只报**实测**半径，不声称保证。

### Q2 识别结构保留了多少？

$$\boxed{\text{非平凡压缩：}552/705 = 78.3\%\qquad\text{对照（恒等、}r=0\text{）：}90/90 = 100\%}$$

失败 153 例中：$99$ 例**基数就先不同**（比识别失败更早），$32$ 例**真的丢了仿射等价**，
$22$ 例是 ε-网的浮点中心被精确层正确拒绝。

### Q3 换来了多少计算收益？

$$\lvert T_S\rvert/\lvert T_X\rvert \approx (m/n)^6\ \text{（标度）},\qquad
\text{中位实测代码规模加速} = 26.3\times$$

### 结论口径（F5.3 的措辞纪律）

在对照组 100%、非平凡压缩出现显著退化之后，**唯一允许的表述**是：

> 2002 年 эскиз/остов 理论所保证的**相似类距离近似**，
> **并不自动蕴含** 2009 年有限点集仿射代码在压缩后仍可定义或仍保持等价。

即

$$\boxed{\text{distance approximation guarantee}\ \not\Rightarrow\
\text{affine-code preservation guarantee}}$$

**不能说「两套理论不兼容」**：两者研究目标本就不同——前者关心距离估计，后者关心代码等价。
正确说法是**前者不蕴含后者**，而本阶段把这一缺口**量化**了。

### 三维取舍

$$\boxed{\text{Compression}(0.388)\ \longleftrightarrow\
\text{Recognition Preservation}(78.3\%)\ \longleftrightarrow\
\text{Computational Cost}(26.3\times)}$$

**更小、更快、但结构损失更多。** 且三者的耦合不是平滑的：
损失主要来自**基数不同**（99/153），而不是几何漂移——也就是说，
压缩的**离散性**（点数必须整数、草图必须保有非退化三元组）比压缩的**幅度**更早地决定成败。

---

## Phase 5 Closure

```text
G5.1  ... G5.10        = PASS
F5.0 sanity controls   = PASS（100% / 100%）
full test suite        = PASS（528 tests）
```

⇒ **Phase 5 ✅ CLOSED，Phase 6 🔓 UNLOCKED**

### 遗留（不影响闭合，但需登记）

* `TeachingGridReduce` 在 3 行上因格点过粗而无法产出草图（`refused`），
  已在 CSV 中保留为结果而非丢弃；
* 强度扫描为离散档位，未做 CR 连续扫描；
* `INVALID_INPUT_AFTER_COMPRESSION` 若能通过"允许浮点中心走数值层"消除，
  会把这 22 行变成可判定——这是 Phase 6 之后可选的跟进项。
