# 封闭网格到 FreeCAD / STEP

`scripts/mesh_to_cad.py` 是独立入口，使用已准备的 **FreeCAD 1.1.4 自带 Python 3.11**，不导入 `kozlov_lab`，不依赖项目 Python >=3.12 环境。输入是已完成清理的 `.ply`、`.stl` 或 `.obj` 网格；通过结果只说明几何转换与 CAD 文件往返，不说明照片外形准确、真实尺度正确或参数化设计已恢复。

## 调用

在项目根目录打开独立 PowerShell 进程，设置本进程环境后调用。`-I -B` 放在脚本路径前；不要在 T02A 的 Python 3.13 环境运行，也不要 `pip install -e`。

```powershell
$fc = Join-Path $PWD 'outputs\toolchains\freecad-1.1.4\FreeCAD_1.1.4-Windows-x86_64-py311'
$env:PATH = (Join-Path $fc 'bin') + ';' + $env:PATH
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
& (Join-Path $fc 'bin\python.exe') -I -B .\scripts\mesh_to_cad.py `
  --mesh 'C:\path\to\closed_object.ply' `
  --output 'C:\path\to\new_cad_output' `
  --source-run 'C:\path\to\source_run' `
  --scale-status unknown --length-unit arbitrary
$LASTEXITCODE
```

这是进程内环境设置，不修改系统 PATH。不要把示例占位路径原样运行。批处理应使用可记录真实 argv/退出码、限制墙钟/输出、只回收该子进程树的监督器。脚本将数值线程环境变量设置为 2；Windows 上还把本进程限制到当前允许的最多两个 CPU，因为 OCCT 的详细检查会请求原生并行。

参数：

| 参数 | 含义 |
|---|---|
| `--mesh` | 必须存在的网格，最大 64 MiB；读取后检查实际三角面数 |
| `--output` | 全新的目录；任何已存在路径均拒绝，绝不覆盖旧结果 |
| `--source-run` | 必须存在的来源文件或运行目录；文件会记录 SHA256，目录仅作为来源引用 |
| `--scale-status unknown --length-unit arbitrary` | 没有可靠尺度的默认使用方式；保留坐标数值，CAD 毫米仅是文件单位约定 |
| `--scale-status reference_scaled --length-unit mm` | 调用者声明输入坐标已按参考定为毫米；另需 `--scale-reference` 指定现存证据文件 |
| `--scale-reference` | 参考尺度证据文件路径与摘要；转换器不核实测量事实，不执行缩放 |
| `--max-faces` | 默认 10000；超限立即拒绝 BRep 转换，不自动简化。提高上限需另行评估资源 |
| `--synthetic` | 明确将合成输入标成合成验证，成功状态不会写成照片结果 |

所有模式的坐标因子固定为 1；脚本不会根据文件扩展名、包围盒或物体类别猜测真实毫米。`physical_accuracy_status` 始终为 `NOT_EVALUATED`。不提供厘米/米自动换算；先在独立、有变换记录的预处理步骤中定尺度。

## 实际门禁

先检查有限坐标、非退化三角形、重复面、边恰有两个反向邻面、顶点邻面形成单一闭环、正有向体积，再调用 FreeCAD 原生网格检查。仅接受单组件、闭合、法向一致、无原生检出的自交或损坏邻接。开放网格直接拒绝；不补洞、封底、生成凸包、反转法向或代换几何。

FreeCAD 网格自交检测会跳过某些共享顶点的面组合，因此还对生成实体、FCStd 回读实体和 STEP 回读实体运行 `shape.check(True)`。它检查 OCCT 自身容差下的 BRep 自交与其他几何错误；这属于软件验证，并非数学意义下对任意输入的完备证明。

转换要求整个形状恰含一个封闭壳，不丢弃游离面/边，三角面数保持不变；实体必须有效、封闭、恰含一个 solid、体积有限且为正。网格→BRep、FCStd 回读、STEP 新文档导入均做数值检查。包围盒使用 `source_extent_v1`：六个坐标差分别不超过 `max(1e-9, 1e-7 × 源包围盒最长边)`，不再按绝对世界坐标放宽容差。体积单独采用 `math.isclose(rtol=1e-7, atol=1e-9)`；逐次报告两类实际阈值。网格缝合容差独立固定为 `1e-7` 输入坐标单位。没有对文件存在性直接判通过的路径。

这些容差是转换数值门槛，不是对真实物体的尺寸准确度承诺。复杂、小尺度或病态网格可能被拒绝，应保留原始失败证据后另行诊断，不在同一输出目录自动调整容差重跑。

## 输出和退出码

- `model.FCStd`、`model.step`：面片化近似 BRep；没有恢复参数化草图/孔径/圆角特征树。
- `conversion_report.json`：源网格摘要、来源引用、尺度声明、真实软件版本、拓扑/几何检查、往返差值及产物摘要。
- `conversion.log`：实际完成的阶段；失败时还有 `error.log`。
- `runtime/`：本次 FreeCAD 配置、数据和临时目录；系统配置不修改。

退出 0 且报告状态为 `PASS_GEOMETRIC_CAD_CONVERSION` 或 `PASS_SYNTHETIC_CAD_CONVERSION` 才是转换通过。退出 2 表示输入/输出条件拒绝，3 表示运行时/后端错误，4 表示 CAD 构造或往返门槛失败。早期参数错误或输出已存在时，只向 stderr 报错，保留既有目录；其他失败保留本次目录及未完成文件，文件存在不代表成功。原生进程被外部终止时，报告可能停在 `STARTED`，监督器的真实退出/超时记录优先。

## 验证边界

2026-10-07数值修复：已复现旧坐标比较会放过“尺寸1、原点1e9、导出错位1”的错误。新比较按物体尺寸约束线性误差，加入远原点/形状变化/体积独立门槛和坏指标反例。此次只验证纯数值逻辑，不导入FreeCAD或重跑旧原生用例；新版本的完整原生转换仍待后续真实网格验收。来源目录仍仅为声明引用，单纯几何PASS不能用作已验证照片来源链。

历史首轮仅运行两个真实 FreeCAD 子进程：合成封闭四面体成功，删除一面的同一四面体以 `OPEN_MESH` 拒绝。结果证据在 `outputs/delegation/CAD_CONVERTER/`。本次未重复此前 FreeCAD 探针或运行照片、GUI。

成功的 `.ply` 用例也不能代表所有 PLY 方言、STL/OBJ 输入、万面网格、照片噪声、自交负例或极端尺度均已验证。真实网格仍必须逐件经过门禁；真实外形检查与最终 GUI 打开截图由后续交付负责。

原生检查的语义依据：[FreeCAD 1.1.4 Mesh 检查](https://github.com/FreeCAD/FreeCAD/blob/1.1.4/src/Mod/Mesh/App/Core/Evaluation.cpp)、[TopoShape Python 绑定](https://github.com/FreeCAD/FreeCAD/blob/1.1.4/src/Mod/Part/App/TopoShapePyImp.cpp)、[OCCT BOP 调用](https://github.com/FreeCAD/FreeCAD/blob/1.1.4/src/Mod/Part/App/TopoShape.cpp)。
