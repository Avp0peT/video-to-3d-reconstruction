# 真实照片对应点到 Kozlov 精确构造的适用性检查

入口 `scripts/audit_photo_parallel.py` 读取现有 COLMAP 原生二进制模型，不运行 SfM、SAM3、MVS 或 GPU。对应关系由旧 SfM 提供，本模块没有解决未知对应问题，也不修改已有网格。

```powershell
& 'C:/Users/13944/.workbuddy/binaries/python/envs/kozlov/Scripts/python.exe' -B -X utf8 `
  scripts/audit_photo_parallel.py `
  --sfm-run '<已有 sfm-native-object-v1 run 绝对路径>' `
  --model-id 0 --output '<新的输出目录>'
```

输出目录必须不存在且不在原 SfM run 内。退出0仅表示诊断证据完成，不能作为“新照片模型生成”或输入适用的成功码；读取 `RESULT.exact_status`。发生输入/身份/证据错误退出2并保留已产生文件。

## 输入与确定性选择

完整有界解析 images.bin / points3D.bin，双向核对每个 image/point2D_idx/point3D_id。允许合法的同一 track 在同图出现多个不同观测；完整保留这些记录，不随意选择其中一个。

逐图对统计在两幅图中各恰有一个观测的 track。按此数量降序、image_id升序打破并列，选定一对；其合法共同track按point3D_id升序取六个，不依据数学结果重新选点。INPUT.json保留全部图对计数、全模型多观测账本、所选图对排除项，以及六点的原始binary64十六进制表示、实际照片SHA256和frames_map身份。

这些是明确的场景六点子集，不称完整物体；未核定前景归属。原图像观测直接取自images.bin，绝不使用3D重投影制造精确相容数据。

用Fraction.from_float精确保留储存值。不能四舍五入或limit_denominator以通过论文域检查；现有AlgT′有理输入分子/分母界均为10^12，超界返回INVALID_INPUT并保留原因。

## 数学判定

公开API位于 `kozlov_lab.stereo.official.given_parallel_correspondence`：

```python
assess_given_parallel_input(source_xy, target_xy, mapping, max_work=..., seconds=...)
assess_given_parallel_certificate(source_xy, target_xy, mapping, certificate, ...)
```

mapping是完整整数标签双射dict；每图完整4–7有理二维点。照片桥接固定六点，避免四点联合仿射秩天然≤3所造成的缺乏额外一致性约束。

- 联合 `(x,y,u,v)` 仿射秩4：`INCOMPATIBLE_EXACT_PARALLEL`，提供精确行列式/张成证书。仅否定该对应和原坐标的精确平行投影模型。
- 秩≤2：`UNSUPPORTED_PLANAR`，保留平面/同核等歧义，不将它称为错误匹配。
- 秩3：实际从两侧运行AlgT′，验证全部点的两幅投影和代表体间的仿射关系，成功为`CONSTRUCTED`。它是未知尺度的仿射点集代表，不是度量网格或CAD。
- 预算耗尽或分支未解决：`UNKNOWN`；输入无效：`INVALID_INPUT`。不能当成“不存在”。

独立验证器核输入身份、精确秩及全部投影几何，不重新调用构造或rank搜索。运行branch trace只核结构，不能用VERIFIED证明历史执行路线真实；返回`branch_trace_validation=STRUCTURE_ONLY_NOT_EXECUTION_REPLAY`。真实调用由运行路径及调用测试单独证明。

来源为S-1996 AlgT′及项目已有[原文映射](specs/stereo_algorithms_1996.md)§4/6/8。joint-rank判据、native输入桥接和以下TLS诊断属于项目推导/工程补充，不冒充论文原文完整照片算法。

## 近似残差与合成控制

对所选图对全部唯一观测共同track，在原始四维像素坐标中中心化，SVD给出到最优三维仿射平面的正交距离。RMS/中位数/最大值只描述这组数据的近似性；不是每图重投影误差或物理尺寸误差。没有噪声标定或接受阈值，故不判近似平行投影可用，也不把拟合坐标传入精确AlgT′。

固定六点非平面有理体及两组平行投影另作合成正控制，必须实际构造并独立回读通过。合成控制不能充当照片重建成果。

## 2026-10-14 首次真实输入结果

[完整交接](../outputs/delegation/KOZLOV_LEGACY/20261014_real_parallel_applicability/RUN/HANDOFF.md)：骰子33照片既有model0；image31/32对应原index30/31，1195共同track中11项同图多观测，1184项可作唯一对应。六点固定为1/7/8/12/13/15，精确联合rank4，独立拒绝证据VERIFIED。全1184点TLS联合距离RMS0.499275像素、最大2.949080像素，无接受阈值；合成正控制CONSTRUCTED。

这个结果不区分透视、镜头畸变、测量噪声、量化或错误track各自的影响。下一步应单独定义可审查的近似适配/鲁棒对应实验，并与原始未改动观测作对照；不能因为精确条件不成立就人工改点、补面或声明作者算法无效。
