# 严格审核及推算前后自动对照

当前处理路径把原深度重建、宽容轮廓推算和严格轮廓推算分别保留。点审核和轮廓推算是两个独立实验，不把轮廓的改变归因于点阈值，也不把推算顶点改称实测点。

## 随建模自动执行

在项目目录运行 `scripts/run_depth_fusion_model.py`，同时指定 `--full-depth-parent` 与 `--allow-inferred-completion`。完成融合、审核及补全模型后，自动执行 `strict_comparison/`：严格点/面审核、三个版本的同视角图、每个标定视角的投影对照、HTML 和请求的 Blender 导出。

```powershell
C:\Python313\python.exe -B -X utf8 scripts/run_depth_fusion_model.py --source-model outputs/delegation/GLOBAL_LOCAL/20261025_model07/real01 --full-depth-parent outputs/delegation/DENSITY_MODEL/20261023_identity_surface02 --output outputs/delegation/GLOBAL_LOCAL/my_fresh_completion --allow-inferred-completion --blender --seconds 720
```

该命令仅展示使用方式。本轮没有重跑完整融合链；完整串联由 CLI 测试验证，真实运行验证了从现有补全输入到严格审核/图像/Blender的路径。全部阶段共享总时限。新输出必须不存在，不覆盖旧模型。没有该补全选项时，不进入推算和对照阶段。

## 复用现有模型，仅执行对照

```powershell
C:\Python313\python.exe -B -X utf8 scripts/compare_object_completion.py --inferred-run outputs/delegation/GLOBAL_LOCAL/20261026_inferred_completion16/real01 --baseline-delivery outputs/delegation/GLOBAL_LOCAL/20261026_closed_model17/real01 --output outputs/delegation/GLOBAL_LOCAL/my_fresh_comparison --blender --seconds 600
```

新输出也必须不存在。该路径不执行 SAM3、SfM、MVS、重拍或重新计算轮廓场。验明输入哈希、相机顺序、同坐标网格、同网格间距的 q=0/q=0.1 候选，再处理。可省略 `--blender`。

## 固定审核语义

- 原深度点：2来源/1°/2% 的原支持集合进一步收紧到3来源/2°/1%。遮挡、未知、缺少深度不当作反票。所有证据共享重建输入，不能称独立真值。
- 原深度面：三个顶点通过，并有3个共同深度支持视角和足够视差，再检查完整投影三角形是否跨出掩膜。保存原索引和每项拒绝原因，不移动点、不增加面。
- 轮廓对照：只比较已冻结的 q=0 与 q=0.1 包络；不存在严格包络时展示空列和缺失，不捏造网格。无可导几何的 Blender 记 `NOT_APPLICABLE/strict_envelope_absent`，不写 SUCCESS。
- 更严格可以减少背景投影，同时降低前景覆盖。共享输入的覆盖率和 IoU 都不是物理准确率；严格版不会自动替换当前版。

`POINT_AUDIT.npz` 保存逐点逐面理由及原索引；`POINT_AUDIT.json` 保存计数和判据；`PROJECTIONS.json` 与 `projections/*.png` 保存每视角结果；`RESULT.json` 分列拓扑、覆盖、质量及 Blender 状态。

2026-10-26 真实对照：67视角，64项最终测试通过，1次有界CPU对照，Blender创建/独立回读通过；主线完整视频入口未在本轮重跑。模型仍无物理尺度、独立形状真值和 CAD 验收。
