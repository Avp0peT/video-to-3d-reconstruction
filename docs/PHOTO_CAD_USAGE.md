# 已完成照片任务的 CAD 续接

也可在统一主命令直接选 `--target cad --freecad-python <已有FreeCAD的python.exe>`：输入、SAM3、SfM、表面与Blender验证后，保留独立研究侧支，再以固定quadric方案继续CAD。默认仍是 `blender`；`report` 和 `blender` 不启动CAD。

```powershell
python -B -X utf8 scripts/run_photo_demo.py run --source <照片目录或视频> --prompt <物体描述> --target cad --output outputs/photo_demo/<全新任务> --freecad-python <已有FreeCAD的python.exe>
```

主命令在当前任务写不可覆盖的 `CAD_SOURCE.json`，其 `GEOMETRY_READY_FOR_CAD` 表示已可交付表面/Blender、CAD尚未完成，不是假定整链成功。续接以 `--geometry-record` 核验该固定快照与同任务当前几何身份，输出在本任务 `cad/`。CAD失败或开放网格拒绝时，整体CAD目标明确未达成；`geometry_deliverable_ready=true` 表示原Blender成果保留，`deliverable_ready=false` 表示目标CAD未交付。不补洞或自动重试。

对已完成、`deliverable_ready=true` 的统一照片任务运行：

```powershell
python -B -X utf8 scripts/continue_photo_cad.py --run outputs/photo_demo/<已完成任务> --output outputs/photo_cad/<全新目录>
```

`--plan-only` 只核对文件、来源和依赖，打印冻结计划，不创建输出或运行原生程序。可用 `--python` 与 `--freecad-python` 指定已安装解释器。系统不安装依赖。

入口核对原任务表面、Blender创建与独立回读证据和当前文件摘要，明确选择同一成功MVS生产者的 `mesh` 输出角色。不会修改原任务或原Blender文件。固定使用 quadric、目标2000面、输入上限400000面；双向顶点及面心采样偏差限为原包围盒对角线1%，体积变化限2%。采样检查不是完整Hausdorff界，也不是物理精度证明。

准备限60秒/2GiB，通过现有拓扑和原生几何门才运行CAD交付，其内部共享预算120秒/2GiB。外层Job用于监督、回收及记录；CAD外层留135秒处理内部回收。输出阈值256MiB，内部CAD仍采用既有128MiB阈值。任何失败保留 `RESULT.json` 和日志，不重试、不换参数、不补洞、不提高额度。开放或不合格网格拒绝转换。

成功文件在新目录的 `cad/cad/model.FCStd` 与 `model.step`。现有交付验证器核对原生文件重开/重导入、有效闭合单实体与数值一致性，结果附实际SHA256和字节数。模型是三角面组成的 faceted solid，不是恢复参数化设计；尺度未知，单位任意。照片外形精度、物理尺寸及GUI操作仍需另行验收。

页面提供下载前调用 `integrations.demo_pipeline.cad.verify_cad_output(output, expected_source_run)`。它只读重核当前结果、指定照片任务、冻结来源和代码、既有交付验证器与FCStd/STEP摘要，返回 `artifacts`（path/bytes/sha256）；失效或来源不一致时抛出异常，不启动原生程序。
