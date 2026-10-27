# 从已准备网格生成 CAD 交付包

`scripts/build_cad_delivery.py` 将已有 `mesh-prep-v1` 成功产物接到可选定尺度、FreeCAD/STEP 转换与离线 PLY 预览。它是照片/视频通路的**交付后半程**，不做分割、SfM、稠密重建、补洞或网格准备。

当前证据：新来源契约下的合成网格准备与未知尺度 CAD 交付各执行一次并退出 0，得到一个有效闭合的 400 面实体，FCStd 重开和 STEP 重导入通过；pure02 XML 为 111 项纯数据测试全部通过。历史参考尺度路径单独保留，不能据此宣称本轮重新执行了定尺度。实际照片 CAD、物理精度、GUI 与浏览器交互仍未验收。

## 一条命令生成

项目目录内，使用项目 Python 启动监督器、FreeCAD 自带 Python 执行原生转换：

```powershell
& '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe' -B '.\scripts\build_cad_delivery.py' `
  --prepared-run '.\outputs\delegation\MESH_SOURCE_LINK\20261007T150200Z_main\prepared' `
  --freecad-python '.\outputs\toolchains\freecad-1.1.4\FreeCAD_1.1.4-Windows-x86_64-py311\bin\python.exe' `
  --output '.\outputs\cad_delivery_demo01' --wall-seconds 120
```

这条示例使用**合成球面网格**。目录必须全新，不会覆盖旧失败或成功产物。没有参考长度时保留任意尺度；CAD文件虽然采用mm存储约定，不能当作物理毫米。

如已对当前 `prepared.ply` 完成参考端点对应，可加 `--reference <mesh-scale-request-v1.json>`。请求必须绑定**简化后**网格的实际SHA256和顶点索引；参考长度、单位、测量证据与可选独立检查按 `MESH_SCALE_USAGE.md`。不自动从原始网格或照片中猜测尺寸。

输入目录必须含 `mesh_prep.json`、`prepared.ply`、`selected_before.ply`、`selection_map.json`。入口核对成功状态、严格整数退出码、synthetic 标志、原始输入和全部准备产物 hash，以及五个准备实现文件的当前 hash：`prepare_mesh.py`、`mesh_to_cad.py`、`geometry.py`、`ply_io.py`、新增 `integrations/meshprep/provenance.py`。它还重算 `inputs.mesh/source_record` 的生产者输出关联，与报告 `source_validation` 完整比较。合成/真实声明继承上游，不由新开关改写。

准备阶段只接受两种来源：合成 `mesh-prep-source-v1` 必须有 `data_kind=synthetic`、`evidence_scope=TEST_ONLY`、严格布尔 `photo_reconstruction=false`、非空 `generation` 和唯一匹配的 `fixture_mesh` 文件绑定；surface 来源须为成功声明的 `mvs-surface-run-v1` 与同目录匹配的 `plan.json`。surface 准备时必须传 `--source-role mesh` 或 `--source-role pre_cleanup_surface`；合成来源使用 `--synthetic`，角色默认或显式为 `fixture_mesh`。这些参数属于 `prepare_mesh.py`，不是本交付 CLI。完整 JSON 示例、路径/hash/阶段状态检查范围见 [MESH_PREP_USAGE.md](MESH_PREP_USAGE.md)。

`source_validation` 的版本为 `mesh-source-link-v1`，关联状态 `PRODUCER_OUTPUT_BOUND`，记录 mesh/source_record/可选 plan 的实际路径、字节数和 SHA256。它只证明声明的生产者输出与输入文件相连，`photo_lineage_status/native_execution_status` 仍为 `NOT_VERIFIED`；不认证 surface 的 native 执行、完整照片链或 mask 消费效果。

旧任意来源 `SOURCE.json`、仅绑定四个准备实现文件、或缺少 `source_validation` 的报告保留为历史证据，当前入口不能直接复用。不要回填旧报告或改写旧基线；需要新准备产物时使用受支持来源和全新目录。

## 结果与失败定位

输出根目录 `RESULT.json` 是当前交付阶段的终态；成功状态 `COMPLETED_GEOMETRIC_DELIVERY` 只表示下面各项完成：

- 可选 `scale/`：定尺度结果和可复算 `scale_ledger.json`。
- `cad/model.FCStd`、`cad/model.step`：原生有效实体及往返检查。定尺度路径另要求 `scale_verification.json` 最终通过，不能只看内层CAD报告。
- `viewer.html`：CAD输入PLY的单文件离线预览；不是STEP/FCStd渲染，状态仍为 `EXPORTED_NOT_UI_VERIFIED`。
- `request.json`、`stages.json`、逐阶段stdout/stderr、`supervisor_result.json`：冻结输入、实际argv/环境、进程退出、执行顺序与回收结果。

任何子阶段失败都不启动后续阶段。已写出的中间产物不会被删除或升级为成功；阶段失败详情见 `stages.json`。监督器失败、超时或未确认回收时，不能因目录内已有CAD文件而接收交付。

固定监督边界：Windows Job拥有协调进程及其后代，提交量限制2GiB，共享运行预算默认/最高120秒，回收最多5秒；RUN输出阈值128MiB按采样检查，不是磁盘硬配额。主程序的前置文件哈希与末尾核验在此Job计时之外，记录总墙钟。监督器沿用已运行版本的原字节，没有另写进程树终止逻辑；旧raw peak内存字段不作为已验证峰值引用。

本轮没有内存压力或强制超时的新实测，不借用正常CAD完成声称覆盖所有失败时序。原生阶段当前绑定FreeCAD Python、指定pyd及核心DLL；不是完整工具链/系统镜像的部署证明。

## 只读复核现有交付

```powershell
& '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe' -B '.\scripts\build_cad_delivery.py' --verify-run '<已有交付目录>'
```

通过返回 `BINDINGS_AND_REPORTS_VALID`，只表示当前输入、实现、指定工具、完整输出集合与既有报告仍一致；尺度路径会重新计算尺度账本。它不重跑FreeCAD、不打开GUI、不评价实物精度，也不从失败位置续跑。源码/输入/产物漂移或缺项会拒绝。完整自动缓存、失败续跑与照片/视频统一调度仍未实现。

## 当前来源契约下的执行证据

[本轮 HANDOFF](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/HANDOFF.md)；[pure02.xml](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/pure02.xml) 为 111 项、0 失败/错误/跳过。`prepared/mesh_prep.json` 为 `PASS_PREPARED_TOPOLOGY/0`，typed 合成球体从 528 简化到 400 面。

[cad_delivery/RESULT.json](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/cad_delivery/RESULT.json) 为 `COMPLETED_GEOMETRIC_DELIVERY/0`，`synthetic_input=true`、`scale_status=unknown`。转换报告记录一个有效闭合的 400 面实体，FCStd 重开与 STEP 重导入通过。prepare 和 CAD 两个入口的监督记录均确认退出 0 和回收；没有本轮参考尺度执行。PLY 预览仍为 `EXPORTED_NOT_UI_VERIFIED`，GUI、完整照片来源及物理精度仍未验收。

## 历史合成交付证据（20261007T100714Z_main）

`outputs/delegation/CAD_DELIVERY/20261007T100714Z_main/`：

- 该批次 32 项纯数据测试通过；以下是当时实现的执行记录，旧 prepared 的四文件绑定不满足当前入口。
- `arbitrary01/`：未知尺度，一次CAD子命令成功。
- `scaled01/`：一次定尺度＋一次CAD子命令成功。
- 两者均在Job内执行，外层退出0、回收确认；逻辑CAD执行共2次，Job统计OS成员分别4/6（含解释器启动器），不把逻辑命令数当OS进程数。
- 两次生成总外层墙钟约4.81秒；两个独立verify命令通过。没有照片、SAM3、GPU、下载或安装。

浏览器工具此前因file协议安全政策拒绝访问，本轮没有绕过或尝试其他自动化浏览器。可由用户在本地浏览器打开HTML，并在FreeCAD中打开FCStd；这两项GUI检查仍待完成。

## 来源连接修正与历史验证（2026-10-07）

`validate_cad` 现在必须接收预期来源文件，核对转换报告的 `input.source_run/source_run_sha256`。未知尺度不得带尺度参考；定尺度时来源和参考须为同一份已校验 ledger。wrapper 的 ledger、data_kind、实际启动声明、转换器身份、有效参数和记录的 wrapper argv 分别核对。

注意：定尺度转换通过 `mesh_to_cad.main(arguments)` 在 wrapper 内调用；历史 conversion_report.argv 记录的是 scaled_mesh_to_cad 的启动参数，不是传入的 converter_arguments。新校验分别解释二者，不修改或伪造历史命令。

协调器在启动 CAD 之前，以及 `--verify-run` 的复核路径，都会把已独立校验的 ledger.source_mesh/source_record/request 分别连接到本次 prepared.mesh/prepared.record/reference。两套各自有效的文件不能因此被拼为同一交付。文件路径、字节数和 hash 均须一致；同内容异路径也不冒称同一来源。

证据：`outputs/delegation/CAD_LINEAGE_REPAIR/20261007T144200Z_main/`。67 项纯数据定向测试通过，既有任意尺度/定尺度合成交付的这几条来源边只读检查通过；内存替换报告的两项反例在旧校验被放过、新校验被拒绝。没有重新执行 FreeCAD 或完整交付命令。旧交付仍绑定旧实现，不能把本次局部读回说成新版本完整原生验收；原 `--verify-run` 对当前实现漂移的拒绝规则保留。

最新 `MESH_SOURCE_LINK/20261007T150200Z_main` 已补充 meshprep.inputs.mesh 与 source_record 的指定输出角色绑定，在准备前和 CAD 复用时都重新核验。尚未闭合的是 surface 与 bridge/所选模型/SfM/T01 的来源连接，以及真实 mask 消费效果；生产者记录本身也不是原生执行认证。`photo_lineage_status=NOT_VERIFIED` 保持，不能把这些连接修正当成真实照片 CAD 成果。
