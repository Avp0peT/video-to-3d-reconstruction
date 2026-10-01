---
spec: teaching_affine_code
module: M4 教学型编码（Pedagogical Extension）
status: pedagogical
source_tier: 无（本项目构造，不是教材内容）
phase: Sprint 3C
implemented_in: src/kozlov_lab/teaching/affine_code.py
tests: tests/test_teaching_code.py, tests/test_teaching_code_isolation.py
ui: app/pages/04_Teaching_Code.py
---

# 规格 M4：教学型仿射不变编码 `C_teach(X)`

> ## ⚠️ 本编码器不是教材的正式图像代码
>
> 本编码器是根据内部仿射坐标思想构造的教学型仿射不变编码，不声明其等同于
> Козлов 教材中的正式图像代码。
>
> **This is a pedagogical affine-invariant construction derived from the
> internal-coordinate idea. It is not claimed to be Kozlov's formal image code
> unless verified against a primary source.**
>
> **Данный код является учебной конструкцией на основе внутренних аффинных
> координат и не отождествляется с формальным кодом изображения Козлова
> без проверки по первоисточнику.**
>
> 三语声明必须与任何输出一起显示（`kozlov_lab.teaching.DISCLAIMERS`，
> 界面由 `_shared.teaching_disclaimer()` 渲染）。

Sprint 3C 之后这条声明的理由**更强**了：`docs/specs/official_image_code.md` 表明
正式定义 $\langle M_A,T_A\rangle$ 用的是**两个三元组面积之比**，与本编码器
（标架 + 内部仿射坐标）是**不同的构造**，不只是"未经验证的相似物"。

---

## 1. 隔离规则（由测试强制）

| 规则 | 强制方式 |
|---|---|
| 代码位于 `src/kozlov_lab/teaching/`，**不得**进入 `src/kozlov_lab/coding/` | `test_pedagogical_code_lives_in_teaching_not_coding` |
| `src/kozlov_lab/coding/` 必须为空（Q1 未裁定） | `test_coding_public_api_is_empty` |
| 类名只能是 `TeachingAffineFrameCode` / `TeachingAffineImageCode` | `test_only_teaching_defines_an_image_code_class` |
| 禁止使用 `KozlovImageCode` / `OfficialImageCode` 及其近似名 | `test_forbidden_names_are_absent_from_every_identifier`、`test_no_near_miss_names_outside_the_disclaimer` |
| 三语声明必须存在且含"教学型 / pedagogical / учебной"与"不声明 / not claimed / не отождествляется" | `test_disclaimer_says_pedagogical_and_not_official` |
| 教学层不得反向依赖 `coding`，也不得依赖 UI | `test_teaching_imports_are_within_the_math_layer` |

> 名称扫描基于 **AST**（标识符 + 非文档字符串字面量），不是原文检索：
> 声明的正文必须能写出"这不是 `KozlovImageCode`"，原文检索会把它自己判为违规。

---

## 2. 定义

对每一个**无序**非共线三元组 $\{i,j,k\}$：

1. 在该三元组的 6 种顺序中取使描述符最小的那一种（步骤 3 的规范化）；
2. 计算**全部点**在该标架下的内部仿射坐标 $(\alpha_t,\beta_t)$；
3. 按 `decimals` 量化到整数格点；
4. 字典序排序（消掉点编号）。

得到该三元组的 frame descriptor。全体三元组的 descriptor 排序后即为

$$C_{\text{teach}}(X).$$

### 2.1 规范形式中**只有**量化后的坐标

`triple`、`ordering`、`condition_number` 都是**诊断字段，不进入规范形式**：

* 点编号是任意标签，放进去会让"换一种书写顺序"产生不同编码，
  直接违反 Phase 1 的核心要求；
* 条件数在两个等价图像的末位比特上可能不同，会制造伪差异。

对应测试：`test_canonical_text_contains_no_point_indices`。

### 2.2 完备性（非正式论证）

固定标架后，$z\mapsto A+\alpha(B-A)+\beta(C-A)$ 是单射，因此坐标的多重集
**唯一确定点对应**；可逆仿射变换把标架映到标架并保持每个 $z$ 不变，故描述符多重集不变。
反之，共享描述符给出一个标架对，二者之间的仿射映射把一幅图像映到另一幅。
**在量化精度内**，编码判定仿射等价。

---

## 3. 参数与量化（"tolerance-aware" 的真实含义）

`TeachingCodingParameters`：

| 字段 | 默认 | 说明 |
|---|---|---|
| `decimals` | 9（`TEACHING_CODE_DECIMALS`） | quantum $=10^{-\text{decimals}}$ |
| `collinear_tol` | `COLLINEAR_TOL` | 标架合法性判据（用**置换不变**的 flatness） |
| `max_frames` | `None` | 安全阀；**触发时会标记 `truncated`**，因为此时不再是完备不变量 |
| `include_frame_points` | `True` | 含/不含标架三点都是完备不变量 |

### 3.1 `recommended_decimals`：量化步长必须由条件数推出

规范形式的质量取决于

$$\text{quantum} \gg \mathrm{cond}_{\text{worst}}\cdot\varepsilon\cdot64\cdot 100 .$$

`recommended_decimals(points)` 由图像**最差标架**的条件数推出合适的位数，
而不是把 9 写死。数据集 A 的推荐值是 7（最差 cond ≈ 133）。

### 3.2 三条必须报告的质量项

| 项 | 含义 | 报告字段 |
|---|---|---|
| 量化是否安全 | quantum 是否比可达精度粗至少 100 倍 | `quantization_is_safe` |
| 整数格点是否可精确表示 | $\max\lvert\alpha\cdot10^{\text{decimals}}\rvert \le 2^{53}$ | `quantization_representable` |
| 是否仍是完备不变量 | 是否被 `max_frames` 截断 | `complete_invariant` |

⚠️ 格点溢出判据针对的是**内部坐标**（$\alpha,\beta$），**不是输入坐标**：
一个"标架极小、图像极大"的点集会给出巨大的 $\alpha$。
对应测试 `test_quantization_overflow_is_reported` 与它的镜像控制
`test_large_absolute_scale_alone_does_not_trigger_the_overflow_warning`。

### 3.3 哈希只是索引

> **哈希不是数学图像代码本身，只能是代码的计算机索引。**

$C_{\text{teach}}(X)$ 的 SHA-256 由 `digest()` 给出，且 `equivalent_to()` **不**使用哈希
（用规范字节串比较）。理由：哈希在 1e-16 扰动下完全改变，无法回答"差多少"。

---

## 4. 拒绝而不是返回空编码

点数 $<3$ 或全部三元组共线时，`build_teaching_image_code` **抛错**。

空编码会与任何其它空编码相等 ⇒ 识别阶段的一次静默误报。
对应测试：`test_one_dimensional_images_are_refused_not_given_an_empty_code`。

---

## 5. 边界三元组的处理

恰好落在共线阈值上的三元组，可能在一种顺序下合法、在另一种顺序下被拒
（`triangle_flatness` 只在末位比特上对称）。处理方式：**跳过被拒的顺序**，
描述符取"可接受顺序中的最小者"；若 6 种顺序全被拒，才抛错。

同时三元组账目必须闭合：

$$n_{\text{collinear}}+n_{\text{truncated}}+\lvert\text{frames}\rvert=n_{\text{total}} .$$

该恒等式在 `build_teaching_image_code` 内以断言强制（`n_truncated_triples` 字段），
使"截断"与"退化"两种排除不会被混为一谈。

---

## 6. 验收

| # | 条件 | 测试 |
|---|---|---|
| 1 | 8 个变换族 × 5 个数据集下 `C_teach(X) = C_teach(T(X))` | `test_teaching_code_is_affine_invariant`、`test_dataset_round_trip_and_coding` |
| 2 | 可证伪：单点扰动后编码必须不同 | `test_teaching_code_distinguishes_a_valid_affine_map_from_a_general_distortion` |
| 3 | 重标号不变 | `test_teaching_code_is_invariant_under_relabelling` |
| 4 | 规范形式不含点编号/条件数 | `test_canonical_text_contains_no_point_indices` |
| 5 | 点数不同 → 报 `cardinality mismatch` | `test_cardinality_mismatch_is_reported_as_such` |
| 6 | $n=3$ 时所有非共线图像同码（数学上正确） | `test_three_point_images_are_all_equivalent` + `test_four_point_quadrilaterals_differ` |
| 7 | 隔离规则 | `tests/test_teaching_code_isolation.py`（13 项） |

**实验证据**：`experiments/results/shape_generalization.csv`
（5 数据集 × 8 族 × 3 次重复 + 对照，0 失败）。
