# 数学规格目录 / Каталог математических спецификаций

本目录是项目中**唯一的数学定义来源**。README 第 16 条明确规定：

> Theory Agent **先输出数学规格**。其他 Agent 不允许自己猜数学定义。
>
> 如果论文与当前实现思路不一致：**停止猜测 → 建 issue → 交给 Theory Agent 确认**。

因此每个规格文件顶部都有 `status:` 字段，任何实现都必须以它为准：

| status | 含义 | 允许的实现范围 |
|---|---|---|
| `verified` | 定义已由权威来源确认，可直接实现 | 实现必须与规格逐条对应 |
| `unverified` | 定义**尚未**由一手资料确认 | **只允许实现标注为「教学型」的近似版本**，且 UI 必须显式声明 |
| `planned` | 尚在规划，本阶段不实现 | 不允许实现 |

## 来源等级（source tier）

每条数学断言都必须标注它来自哪一层来源，避免出现第二个"真相来源"：

| tier | 来源 | 说明 |
|---|---|---|
| **S0** | 本项目 README（`readme.txt`） | 本阶段的最高权威：它本身转述了教材的定义 |
| **S1** | В. Н. Козлов《Математическая теория зрительного восприятия》，МГУ，2026 | 教材正文 |
| **S2** | В. Н. Козлов，「Распознавание изображений, представляемых конечным множеством точек」，2009 | 论文 |
| **S3** | МГУ 官方课程材料 | 辅助 |
| **S4** | 本项目推导 | 必须标注为推导，**不得**冒充教材内容 |

> ⚠️ **当前状态（Phase 0–2）**：S1–S3 的一手资料**尚未**在本仓库中核实。
> 因此本目录中所有标记为 S0 的规格都可以实现（它们直接来自 README 的数学陈述），
> 而任何需要 S1–S2 才能确认的内容一律标记为 `unverified`，并且在代码中**没有实现**。
> 这不是保守，而是 README 第 23 条「不要根据二手博客自行创造教材算法」的执行。

## 文件清单

| 文件 | 模块 | status | 覆盖阶段 |
|---|---|---|---|
| [`point_set.md`](point_set.md) | M1 有限点集图像 | `verified` (S0) | Phase 1 |
| [`transforms.md`](transforms.md) | M2 几何变换实验室 | `verified` (S0) | Phase 1 |
| [`affine_coordinates.md`](affine_coordinates.md) | M3 内部仿射坐标 | `verified` (S0) | Phase 2 |
| [`invariants.md`](invariants.md) | M6 不变量实验中心 | `verified` (S0) | Phase 2 |
| [`image_code.md`](image_code.md) | M4 图像代码（旧规格，已被 official_image_code.md 取代） | `superseded` | — |
| [`official_image_code.md`](official_image_code.md) | **Q1：正式「код изображения」定义** | `verified`（2009 全文，一手） | Phase 3（**已解锁**） |
| [`teaching_affine_code.md`](teaching_affine_code.md) | M4 教学型编码（Pedagogical Extension） | `pedagogical` | Sprint 3C（已完成） |
| [`recognition.md`](recognition.md) | M5 仿射等价识别 | `planned` | Phase 4（**被 Gate 阻塞**） |
| [`compression.md`](compression.md) | M7 图像信息压缩 | `planned` | Phase 5 |
| [`stereo.md`](stereo.md) | M8 立体视觉 | `planned` | Phase 6 |

## 未决问题清单（必须在实现前解决）

| ID | 问题 | 阻塞 | 需要的来源 |
|---|---|---|---|
| Q1 | 教材正式定义的「图像代码」`C(X)` 的精确形式是什么？ | Phase 3 | ✅ **已取得定义**（见 [`official_image_code.md`](official_image_code.md)）；仅剩 G3.1"专利能否替代教材"待裁定 |
| Q2 | `k-эквивалентность` 的严格定义 | Phase 10 (E1) | S1 / S3 |
| Q3 | 教材中「сжатие информации」使用的具体简化算子 | Phase 5 | S1 |
| Q4 | 教材立体视觉模型所用相机模型与对应点求解方法 | Phase 6 | S1 |
| Q5 | 「判别函数」的教材形式（线性 / 二次 / 其他） | Phase 10 (E2) | S1 / S3 |

Q1 现已取得**逐字可读的正式定义**（来源为 Kozlov 本人专利说明书 + 2009 摘要，
见 [`official_image_code.md`](official_image_code.md)）。但该文件记录的 Gate 表中有
一项（G3.1 的来源层级）需项目负责人裁定，因此：

```text
Phase 3 = LOCKED（待 G3.1 裁定）
```

在裁定之前，`src/kozlov_lab/coding/` 保持为空，教学型编码继续保留其
"非教材正式定义"的声明（而且现在理由更强：两者被证实是**不同的构造**）。

## 规格变更流程

1. 修改规格 → 更新 `status` 与 `source tier`；
2. 由 Theory Agent 确认（Phase 0–2 阶段由项目负责人确认）；
3. 实现与测试同步更新（测试必须能因规格变化而失败，否则测试与规格脱钩）；
4. 在 `docs/experiments/` 中记录受影响的结果是否需要重跑。
