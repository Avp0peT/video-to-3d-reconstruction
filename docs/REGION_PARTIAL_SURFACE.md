# 区域重建的局部表面

已有 `input/PLAN.json` 和 `geometry/RESULT.json` 的区域重建可直接继续：

```powershell
& <项目Python> -B -X utf8 scripts/run_photo_demo.py surface-regions --run <区域重建目录> --output <新的输出目录>
```

默认选择注册视角最多的分量；也可显式 `--model-id 0`。不合并分量，不重新分割或运行 SfM。输出目录必须全新，与输入目录互不包含。

自动步骤：

1. 冻结输入与代码。重复视图或少于两个独立视图的三维点，最多占点数 1%，按整点生成显式排除清单；其他结构错误拒绝。原文件只读。
2. 导出独立原生模型与 PNG 图像。JPEG 实际解码为 PNG，逐像素核对；保存名称映射。原始点与派生点的关系在 `export/DERIVATION.json`。
3. 检查相机数值域；COLMAP 去畸变；OpenMVS 格式往返核对。
4. 将每张已绑定掩膜映射到对应去畸变相机，逐像素回读 mask/validity。缺图、空掩膜或身份不一致即停。
5. 冻结 OpenMVS 计划，只尝试一次稠密和网格计算。线程 2；共享 1200 秒、每阶段最多 900 秒、作业提交量 8 GiB。不开启补洞、平滑或分量合并。
6. 若得到有限非空网格，保存原始 PLY、离线 HTML、Blender 文件并独立回读。超出显示导出器容量时保留原 PLY，不擅自简化。
7. 自动输出 `point_density/index.html`，比较物体内旧/新采样及实际三角化观测的局部密度。未达标区域保留明确缺口；新候选不会自动伪装成三维点。详见 `OBJECT_POINT_DENSITY.md`。

每步单独保存 argv、日志、退出码和清理状态。外层整目录 2 GiB 采样阈值包含导出、桥接、掩膜和日志，可能早于内层表面预算触发；时间与磁盘限额均含采样延迟，作业内存由 Windows Job 限制。

`SURFACE_GENERATED_NOT_ACCEPTED` 只代表真实生成且回读通过。区域来源由独立 `REGION_SOURCE_CHAIN.json` 验证；现有通用表面运行器仍标记 `input_origin=unverified`，不冒充旧的照片生产契约。未有正向原生消费证据时，掩膜消费保持 `NOT_VERIFIED`。物体完整性、尺度、精度与 CAD 均不随文件生成而通过。
