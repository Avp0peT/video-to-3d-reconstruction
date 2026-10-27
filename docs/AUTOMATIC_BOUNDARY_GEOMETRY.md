# 自动边界几何诊断

区域表面流程在 `face_local` 的契约和产物检查通过后，自动启动一次 `boundary_geometry` 诊断。该步骤从同一表面运行中已注册且有深度的视图选择最多 16 帧，结合原分辨率 RGB 边界证据与三角形自由空间检查。未选视图、缺失深度或不可验证证据保持未评估，不能借邻帧相机补成有效观测。

监督命令如下，工作进程最多运行 600 秒，内存上限 4 GiB：

```text
python -B -X utf8 scripts/build_region_surface.py boundary-geometry-worker
  --source <surface_run> --gate <surface_run>/face_local
  --output <surface_run>/boundary_geometry
```

工作进程调用 `boundary_geometry.run(surface_run, local_run, output, max_frames=16)`。它只生成诊断，不删除三角面，不修改 mask，不重新运行 SAM、SfM 或 MVS。RGB 支持和自由空间候选也不是分割或几何质量验收。工程阈值不是 Kozlov 书中算法。

主流程读取 `boundary_geometry/RESULT.json`，要求：

- `status` 为 `BOUNDARY_GEOMETRY_DIAGNOSTIC_NOT_ACCEPTED`；不适用时为 `NOT_APPLICABLE` 并带非空 `reason`。
- `source_unchanged=true`、`geometry_accepted=false`、`geometry_changed=false`。
- `boundary_geometry/index.html` 存在。

通过后，主结果写入 `boundary_geometry_status`，报告链接诊断首页和结果记录，并绑定新诊断产物。没有可核验的同源 literal-union 来源时，明确展示 `NOT_APPLICABLE`，不伪造诊断结果。

该步骤始终保留 `face_local/retained_viewer.html` 为现有 preferred 查看对象，不读取诊断输出中的替代模型建议。工作进程失败、契约不符或报告缺失时，`boundary_geometry_status=FAILED`，整体状态为 `PARTIAL_OR_FAILED`；已经生成的表面和局部候选仍保留，但不能以表面生成状态冒称整个流程完成。

源码与测试通过小型 mock 验证派发顺序、600 秒/4 GiB 限制、最多 16 帧、只读契约及 preferred 不变。mock 测试不替代实际 RGB、深度和几何诊断的验收。
