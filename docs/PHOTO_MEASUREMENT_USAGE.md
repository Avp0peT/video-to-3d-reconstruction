# 用实物参考长度续接已有CAD

本入口读取已验证的统一CAD任务，或通过 `--cad-output` 明确指定其成功CAD续接输出。只使用当前prepared.ply，不重建、不重简化、不修改旧任务。它复用现有参考尺度账本与CAD交付，生成新的毫米单位分面实体。

## 在照片页面操作

1. 在“照片到3D”页载入已有的成功CAD任务，展开“按实测参考长度生成 CAD”。
2. 点击“载入可定尺度网格”，再生成并下载端点索引查看器。用准备网格的顶点编号确认实物端点位置。
3. 填写参考段两个编号、实测长度、单位、名称、端点对应说明和已有测量记录文件路径。
4. 可启用“增加独立校验尺寸”，选择另一段端点并填写其实际测量值与允许绝对误差（毫米）。它不参与定尺度，不能复用参考段。
5. 点击“生成带参考尺度的 FCStd / STEP”。原任务保留，新结果单独写入 `outputs/photo_measurement_ui/`；只有来源、账本、原生回读和当前文件摘要均通过才显示下载。

没有参考测量时保留任意尺度结果即可。独立校验启用后若数据缺失、端点相同或超过声明误差，界面不给出成功交付。页面重新渲染和下载不会重新运行原生任务。

当前验证：31项后端/界面定向检查通过；真实骰子几何经过一次人为指定倍率50的数值控制，生成新FCStd/STEP并通过实际页面函数与AppTest只读下载核验。该控制所有记录均为 `CONTROL_ASSIGNED_LENGTH_NOT_PHYSICAL`，绝非实物测量。浏览器工具拒绝 `file://` URL，因此真实浏览器交互未验收；离线HTML已生成。

```powershell
python -B -X utf8 scripts/measure_photo_cad.py --run <原照片任务> --reference <新任务专属reference.json> --output outputs/measurement/<新目录> --plan-only
```

移除 `--plan-only` 才执行一次交付。内预算120秒/2GiB，外Job135秒/2GiB，256MiB采样输出阈值；无自动重试。实际定尺度不得填入猜测值或示例数值；开发数值控制必须独立标为人为指定长度，不得当成实物尺度验收。

UI接口：`inspect_source(run,cad_output=None)`返回当前prepared网格path/hash、顶点数和最多20000个原序顶点；`make_preview(source,output)`生成独立HTML及现有顶点索引工具。索引必须来自该prepared网格，而非原始高面数网格、简化前selection map或照片像素。

`create_request(source,vertex_indices,length,unit,label,correspondence,evidence_path,checks=None)`校验并返回mesh-scale-request-v1，无写入。单位为mm/cm/m/in，真实正长度必须有实物测量证据文件、物理端点与网格索引对应说明；证据以绝对路径和SHA绑定。UI应将reference保存到本次新任务专属文件，再调用 `plan`，且交付output使用未存在的子目录。独立checks不能复用拟合端点对，可附实际允许误差；没有阈值仍是未知。

`verify_result(output)`只读核验原CAD来源、prepared绑定、参考及证据、尺度账本、原生独立回读与当前FCStd/STEP摘要后提供下载绑定。文件位于新output的 `delivery/cad/`。标定只依赖用户声明长度与对应关系；证据hash不能证明测量真实，单段拟合回算误差不能冒充完整物体精度或最终CAD端点实测验收。模型仍为faceted solid，并非参数化设计恢复。

独立 checks 若超过用户声明阈值，纯数据预检及交付账本复核均拒绝成功交付。结果返回 independent_check_status 与 independent_checks；无检查或未声明阈值为 NOT_EVALUATED，全部明确阈值通过为 PASS。这只验证声明长度之间的数值一致性，physical_accuracy 始终为 NOT_EVALUATED。
