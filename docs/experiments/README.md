# 实验结果文档

本目录存放**策展版**（curated）实验报告：经过解读、给出结论、并指明适用边界的版本。

原始产物（每次运行的 JSON + 时间戳 + provenance）不放在这里，而在
仓库根目录的 `results/<EXP-ID>_<日期>/`，可用下列命令重新生成：

```bash
python experiments/exp_a_affine_invariance.py --n 1000
python experiments/exp_invariant_table.py
```

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

## 未开始（被前置阶段阻塞）

| 实验 | 阻塞原因 |
|---|---|
| Experiment B — 仿射识别（100 图 × 50 变体） | 需要 Phase 3 图像代码 + Phase 4 识别器 |
| Experiment C — 点顺序（1000 次打乱，canonical code 不变） | 需要 Phase 3 的 canonical code；Phase 1/2 级别的顺序无关性已由 `test_gate_phase2.py` 覆盖 |
| Experiment D — 压缩 | 需要 Phase 4 识别器 |
| Experiment E — Stereo | 需要 Phase 6 |
