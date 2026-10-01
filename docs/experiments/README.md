# 实验结果文档

本目录存放**策展版**（curated）实验报告：经过解读、给出结论、并指明适用边界的版本。

原始产物（每次运行的 JSON + 时间戳 + provenance）不放在这里，而在
仓库根目录的 `results/<EXP-ID>_<日期>/`，可用下列命令重新生成：

```bash
python experiments/run_all.py            # 全部 5 个实验 + 产物校验
python experiments/run_all.py --quick    # 缩短样本量
```

Sprint 3E 之后，权威产物是 `experiments/results/`：

| 文件 | 内容 |
|---|---|
| `phase2_gate.json` | Phase 2 Gate（主臂 / 可证伪性 / 非平凡性 / 顺序无关性） |
| `invariant_matrix.csv` | 5 数据集 × 7 变换族 × 7 性质 = 245 行，含理论预测与实测 |
| `precision_stress.csv` | 条件数臂 + 量级臂 + 结构摧毁臂 |
| `shape_generalization.csv` | 教学型编码 5 数据集 × 8 族 × 3 次重复 + 对照 |
| `official_code_definition.json` | Q1 正式定义的验证（手算例子 + 5 数据集） |
| `MANIFEST.json` | 每个产物的大小、行数、SHA-256 与 provenance |

每个产物都配 `<name>.meta.json`：`timestamp` / `git_commit` / `git_dirty` /
`python_version` / `numpy_version` / `seed` / `tolerances` / `experiment_config`。

## 约定

每个报告必须包含以下五节，缺一不可：

| 节 | 内容 | 为什么必须有 |
|---|---|---|
| 协议 | 图像、标架、变换族、样本数、随机种子 | 没有协议的数字无法复核 |
| 结果 | 数值表格（含实测值 **与** 理论界） | 只有实测值的结果无法解释 |
| 对照 | 至少一个预期"失败"的配置 | 全绿的表可能什么都没测 |
| 含义 | 能得出什么结论 / 不能得出什么结论 | 防止过度解读 |
| 边界 | 该结论在什么条件下失效 | 见 `../theory/numerical_limits.md` |

## 已完成

| 报告 | 对应 README 实验 | 状态 |
|---|---|---|
| [`EXP-A.md`](EXP-A.md) | Experiment A — 仿射不变量 | ✅ 完成 |
| [`M6-invariant-table.md`](M6-invariant-table.md) | M6 不变量实验中心（README 第 9 节） | ✅ 完成 |
| [`sprint3-multishape.md`](sprint3-multishape.md) | Sprint 3B/3C/3D：多形状验证 + 教学型编码 + Q1 | ✅ 完成 |

## 未开始（被前置阶段阻塞）

| 实验 | 阻塞原因 |
|---|---|
| Experiment B — 仿射识别（100 图 × 50 变体） | 需要 Phase 3 图像代码 + Phase 4 识别器 |
| Experiment C — 点顺序（1000 次打乱，canonical code 不变） | 需要 Phase 3 的 canonical code；Phase 1/2 级别的顺序无关性已由 `test_gate_phase2.py` 覆盖 |
| Experiment D — 压缩 | 需要 Phase 4 识别器 |
| Experiment E — Stereo | 需要 Phase 6 |
