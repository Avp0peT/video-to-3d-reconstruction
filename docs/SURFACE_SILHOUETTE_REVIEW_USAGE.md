# 已交付网格的全视图轮廓检查

`scripts/review_surface_silhouettes.py` 对统一照片流程中**全部选定视图**投影原始三角网格，并与同一去畸变域的 SAM 掩膜比较。它不启动 SAM3、SfM、OpenMVS 或 Blender，不修改原任务、网格、掩膜或 QUALITY 状态。

```powershell
& '<已有项目Python>' -B -X utf8 .\scripts\review_surface_silhouettes.py `
  --run '<已完成pipeline目录>' `
  --output '<项目outputs内的全新、独立目录>' --seconds 180
```

需要已有 NumPy、Pillow；不安装任何依赖。输入限统一 `photo-demo-pipeline-v1` 的成功表面及完整选定视图掩膜，最多96视图、500000顶点、1000000三角形，单文件64MiB、图像每边2048；共享合作时间限1–300秒、输出256MiB。命令本身是合作截止，正式验收运行另受既有Windows Job监督，240秒/2GiB/512MiB。不能把合作检查称为硬实时上限。

逐项校验实际PLY、MVSI7场景、图像、mask/validity的身份与哈希、场景相机K/R/C、原T01 index/frame_id/canonical路径及输入末尾稳定性。只检查原任务选定的全部视图，不挑表现最好的角度；被分割排除的帧另列，不补为成功。无效或缺失输入在创建输出前拒绝，后期异常保留 `INCOMPLETE` 及已完成视图。

方法：`R(X-C)` 转相机坐标，使用实际OpenMVS整数像素中心K（相对COLMAP主点减0.5）；正深度三角形投影并集形成不透明轮廓。当前不实现相机平面裁剪，任何网格顶点非正深度即拒绝。二倍采样、BOX下采样及128门限，并报告一倍采样差异；不对未知validity域计分。空分母返回null，不记完美分数。

产物为 `RESULT.json`、各视图 `*_overlay.png` / `*_silhouette.png` 和离线 `index.html`。绿色轮廓来自SAM，紫色来自网格；全部图像有输出摘要。正常完成退出0；输入、预算或后期错误非零。查看新报告不改变原pipeline的交付/质量状态。

**含义边界**：这些数值是与同源推断掩膜的内部一致性；不是独立人工真值，不是可见纹理/凹陷质量、完整物体、底面或尺寸精度。没有设定或宣称物理精度阈值。

2026-10-12真实cad05运行：31/31选定视图完成，另2张NO_MATCH仍排除；IoU最小0.9861368、中位0.9881095、最大0.9897210。耗时61.15秒（外层61.73秒），输出19.00MiB，Job峰值257957888字节，132项输入绑定不变；29项定向用例通过。[报告](../outputs/delegation/GENERAL_PIPELINE/20261012_cad05_allview_review/review/index.html)、[机器结果](../outputs/delegation/GENERAL_PIPELINE/20261012_cad05_allview_review/review/RESULT.json)。最差idx4/最好idx14叠图已AI辅助查看，边缘基本贴合；没有人工真值。普通浏览器操作未验收。
# Explicit experimental surface source

Use `--experimental-surface <completed surface directory>` instead of `--run` for an
authenticated experimental native surface. The two source flags are mutually exclusive.
This reopens the producer identity and native bridge files and recomputes mask warping;
it starts no native process or model. Every selected scene view is joined by authenticated
native name, global ID, K/R/C, photo bytes, mask and validity bytes. No pipeline receipt is
created. Results carry `source_kind: experimental-source` and `pipeline_acceptance: false`.

```powershell
python -B -X utf8 scripts/review_surface_silhouettes.py --experimental-surface <surface-dir> --output <new-project-output-dir> --seconds 180
```

The same 64 MiB per source file, 256 MiB output and 1..300 second limits apply.
The reader follows only the roles actually consumed by the existing experimental
validator: identity/derived-parent records, producer source bindings and selected-arm
inputs, accepted mask manifests and their photoIO, native models, bridge records,
InterfaceCOLMAP tool, and warped outputs. Bound JSON contents are not recursively
followed: checkpoint, environment, historical argv and unrelated tool strings are ignored.
Plan origin is `experimental-real`; the real surface result declares
`caller_declared_experimental-real`. The fixture tests preserve this distinction.
All consumed source files must resolve under the project ROOT, including bound native tools;
an oversized tool or an external source is refused. Selected-view SAM consistency
is the only quality evidence. Unregistered producer views and frames outside the
producer subset remain unevaluated, as do unseen surfaces and physical accuracy.
Original-frame exclusions come only from explicit `input_index` values in a bound
photoIO `frames.jsonl`; an input-row count alone cannot establish an index universe.

2026-10-12，两种入口均实际完成全选定视图检查：

| 来源 | 已查视图 | IoU 最小 / 中位 / 最大 | 证据 |
|---|---:|---|---|
| 公共入口新视频 video03 | 14/14 | 0.778555 / 0.848180 / 0.893015 | [报告](../outputs/delegation/GENERAL_PIPELINE/20261012_video03_allview_review/HANDOFF.md) |
| 先前实验来源视频表面 | 15/15 | 0.798235 / 0.842714 / 0.893900 | [报告](../outputs/delegation/GENERAL_PIPELINE/20261012_video_experimental_allview_review/HANDOFF.md) |

两者最差视图均可见网格伸入背景。视图集合、相机和重建不同，不能将两列当作单变量效果比较；文件可用不代表物体外形验收。所有原网格、pipeline及QUALITY记录不变。
