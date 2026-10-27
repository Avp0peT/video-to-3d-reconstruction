# 自动区域对应与 SIFT 对照

`scripts/run_mask_region_experiment.py execute` 从现有完整视频联合掩膜和同源 SfM 出发，自动完成：

1. 冻结完整解码时间轴，64 张原始 canonical 端点照片使用原 PNG；中间帧使用已有同网格 JPEG。
2. 掩膜内分层补种，21×21 区域的金字塔 LK、反向检查、原始灰度 ZNCC 与掩膜支持率校验。
3. 中断立即结束该 ID，新显露区域补新 ID，不用预测或插值充当观测。
4. 独立复制旧 SIFT 数据库，保留原特征行及原始匹配，新增区域观测；删除副本描述子和旧几何验证结果，全部重新验证及建模。
5. 另做同参数、无新增区域观测的 SIFT 重新验证对照。
6. 读取原生模型，检查数值相机域；失败分量仅留诊断，不计合格投影观测。生成二维覆盖、共享候选、各独立三维分量和身份分裂统计及 HTML 查看器。

```powershell
& '<light-python>' -B -X utf8 scripts/run_mask_region_experiment.py execute `
  --source-mask-run '<full-union-run>' `
  --baseline-sfm '<same-source-native-sfm-run>' `
  --baseline-report '<baseline-report-run>' `
  --run '<new-output-directory>'
```

也可用 `prepare` 后接 `run`；`report` 只绘图，不启动求解器。已运行的目录拒绝再次执行。当前版本上限为 64 个相同网格端点，完整连续解码时间轴。每个跟踪或原生求解进程均用现有 Windows Job 监督，600 秒、4 GiB 提交量、数值线程 2；不用 GPU、不安装新依赖。

这不是 LoFTR/TAPNext 推理，也不是 Kozlov 几何码的替代实现。区域仍以图像可见纹理为依据，不能直接测出不可辨识的凹凸。相邻区域相似不保证整条轨迹身份；几何内点和低重投影误差不等于物理正确。独立分量不能自动拼接，真实尺度未知。

SIFT 对照的验证随机种子沿用既有配置（可能为 -1）；单次差值不能当统计或严格因果结论。源 mask 是联合掩膜而非人工真值。执行失败保留日志，不自动换参数重试。

依据：先核对项目书籍摘录 `book_code_audit_20261015/SOURCE_REVIEW.md`（图像点集的几何关系与已知对应）；再查 OpenCV 4.13 的 PyrLK/ECC 官方接口与 LoFTR 原作者资料。本轮采用已安装 CPU 区域路径，LoFTR 保留为后续独立方案。

- https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html
- https://zju3dv.github.io/loftr/
