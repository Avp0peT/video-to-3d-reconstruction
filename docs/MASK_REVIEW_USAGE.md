# 物体 mask 复核材料

从项目目录运行现有含 NumPy/Pillow 的环境，无需 SAM3/Torch：

```powershell
& '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe' -B '.\scripts\review_object_masks.py' --mask-manifest 'C:\path\mask_run\manifest.json' --output 'C:\path\new_review'
```

多个批次可重复 --mask-manifest；它们须同 T01 run、同物体且不重复 input_index。输出目录必须全新，父目录预先存在，且位于输入目录之外。

可选同时添加 `--sfm-run 'C:\path\sfm_run' --model-id '0'`，检查所选分量中保存的稀疏观测标签。它只读取既有 description sidecar，不运行 COLMAP、不重载 native binary。全量 per-pair 统计在 report.json，HTML 展示摘要与未知原因。

产物：report.html、report.json、contact_sheet.png、panels/*.png。首先查看最终状态、输入稳定性、上游证据声明和错误，再看原图/mask/叠加图及候选选择。没有选中的候选仍标 UNSELECTED；没有有效照片绑定的失败帧不猜图。提示词、选择方法和来源保留。

人工检查：是否同一个物体；背景或别的实例是否误入；轮廓、孔洞、凹槽是否明显误删；跨视不一致来自错误分割、遮挡、边缘还是不良 tracks。统计没有通用自动通过阈值，稀疏一致不证明整个表面一致。

退出 0 只表示复核材料准备好，不表示分割质量合格。退出 1 表示部分材料或没有可复核的已选 mask；退出 2 为输入错误，4 为内部/输出失败。始终 quality_status/cross_view_status=NOT_EVALUATED；人审结论另存任务证据，不回写原 SAM3 manifest。失败时现存图片不单独构成成功证据。

本轮演示中的合成标签即使覆盖真实照片也不是真实 SAM3 输出，更不是物体真值。完整约束见 contracts_mask_review_v1.md；后续真实 mask 经检查后，仍需 B 的域转换/文件接入及 OpenMVS 实际消费验证。
