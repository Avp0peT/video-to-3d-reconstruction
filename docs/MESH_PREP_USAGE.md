# 网格检查与保拓扑简化

入口 `scripts/prepare_mesh.py`，使用已准备的 FreeCAD 1.1.4 自带 Python 3.11 / VTK 9.3.1。面向单个待转换物体的闭合三角网格；它不做照片重建、自动物体识别、补洞、法向反转或凸包替代。真实照片网格尚未验收。

在项目根目录的独立 PowerShell 中使用已知 FreeCAD 环境，清除进程内 PYTHONPATH/PYTHONHOME，以 `-I -B` 调用。参数为：

```text
<FreeCAD bin/python.exe> -I -B scripts/prepare_mesh.py
  --mesh <existing triangle.ply>
  --source-run <existing source record.json>
  --output <new output directory>
  --target-faces 10000
  --max-relative-deviation 0.01
  --max-relative-volume-change 0.02
  [--component <explicit ID>]
  [--source-role fixture_mesh|mesh|pre_cleanup_surface] [--synthetic]
```

这是参数说明，尖括号必须替换。原生过滤应由外层监督器以60秒墙钟限制运行；入口自身只有阶段间/采样中的60秒协作检查，不能中断卡住的原生调用，不声称硬RAM限额。进程内线程变量为2，Windows亲和性限制至最多两个可用CPU。本轮有界监督示例与真实调用记录在下面证据目录，不能重复运行其已用输出目录。

## 输入与检查

`--source-run` 必须是受支持的来源记录文件，不能只提供一个存在的 JSON 或目录。入口在读取 PLY 几何、配置原生环境之前调用来源校验；完成前再次计算并要求结果一致。来源 JSON 上限 1MiB，拒绝非对象根、重复键和非有限数值；选定网格文件上限 64MiB。选定产物的规范绝对路径、实际字节数和小写完整 SHA256 必须同时匹配，同字节但不同路径不视为同一产物。

### 合成来源

合成输入必须传 `--synthetic`，使用 `schema_version="mesh-prep-source-v1"`，而非任意旧 `SOURCE.json`。示例是字段模板，路径、字节数和摘要须替换成实际值：

```json
{
  "schema_version": "mesh-prep-source-v1",
  "data_kind": "synthetic",
  "evidence_scope": "TEST_ONLY",
  "photo_reconstruction": false,
  "generation": "明确说明合成网格的生成方式；不使用照片重建声明",
  "artifacts": [
    {
      "role": "fixture_mesh",
      "path": "C:/absolute/path/fixture.ply",
      "bytes": 1234,
      "sha256": "<实际文件的小写64位SHA256>"
    }
  ]
}
```

`generation` 必须非空；`photo_reconstruction` 必须是布尔 `false`。artifact 必须使用 `fixture_mesh`，禁止重复产物路径，输入网格须唯一匹配其中一项。合成路径可省略 `--source-role`，此时固定为 `fixture_mesh`；也可显式传 `--source-role fixture_mesh`。本轮可核对的实例见 [SOURCE.json](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/SOURCE.json)。

### surface 来源

真实声明路径不传 `--synthetic`，必须显式指定 `--source-role mesh` 或 `--source-role pre_cleanup_surface`。前者选择 surface 输出网格，后者选择清理前 raw 网格；不根据文件名、面积或文件存在情况猜角色。

来源须为 `contract_version="mvs-surface-run-v1"`、`profile="openmvs-2.4.0-cpu-preserve-raw-v1"` 的 `result.json`：要求 `SURFACE_GENERATED_NOT_ACCEPTED`、严格整数 `exit_code=0`、`input_drift=[]`。同目录 `plan.json` 的契约和 profile 须一致；两份记录的 `input_origin` 均须为 `caller_declared_real`。阶段须按 dense、mesh 排序，均为 `GEOMETRY_READBACK_PASSED` 且严格整数 `returncode=0`。mesh 输出记录须恰含 `pre_cleanup_surface` 与 `mesh` 两角色、不同路径和正整数顶点/三角面计数；路径须位于该 run 的 `work/` 下，所选角色的实际网格身份须匹配。

这些检查只绑定声明的生产者输出，不认证原生执行、不验证 surface 的上游照片链，也不证明 mask 排背景生效。`caller_declared_real` 是声明，不是实拍真值认证；失败 surface 的残留文件不能接入。

仅接收PLY1.0的ASCII、binary little-endian、binary big-endian三角面；64MiB、40万顶点、20万面上限。x/y/z声明须为float/double，精度由自有受限读入器保留。其他标量颜色/法向属性解析后丢弃并登记；纹理不是此CAD几何入口的输出。其他元素、非三角多边形、损坏或超限输入拒绝，不静默三角化或修复。

组件按共享边划分，同一顶点接触不算同组件。多组件必须先读失败报告中的组件摘要，再在新目录显式指定--component；不自动取最大组件，不推断背景。ID按当前输入最小面索引确定，并绑定整个输入文件hash。选定面/顶点原始索引存selection_map.json；无关组件与孤立顶点移除单独记录。

选定网格与最终落盘网格均检查：有限坐标、三角面、重复/退化面、边成对反向邻面、顶点link单环、单一边组件、正有向体积。保存前后Euler与体积变化。自交在本阶段仍NOT_EVALUATED，随后必须经过mesh_to_cad.py的FreeCAD/OCCT检查，不能将本入口PASS当实体有效。

## 简化与偏差

仅当面数超过目标时使用vtkDecimatePro，开启PreserveTopology/AccumulateError，关闭Splitting/PreSplitMesh/BoundaryVertexDeletion，使用double输出坐标。将源包围盒对角线乘指定比例算为绝对误差参数并显式调用SetErrorIsAbsolute(1)/SetAbsoluteError。保拓扑约束可能阻止达到目标；此时保存candidate.ply并拒绝，不改变阈值重跑。

双向采样每个顶点与每个三角面中心到另一网格三角表面的距离，保存两方向样本数及最大值。默认最大采样距离为源对角线1%，体积变化不超过2%；这些是可配置的处理变更预算，不是实物精度、完整Hausdorff界或外形正确性承诺。真实孔洞/凹槽仍需对照照片检查。

自有double ASCII PLY写入后重新读回，坐标和面索引须完全一致；所有后置拓扑与距离检查针对实际落盘数据。输入网格与来源记录运行前后hash核对。原生VTK PLY读写器不用于权威数据：固定9.3.1实现会降低坐标精度，见[Reader源码](https://github.com/Kitware/VTK/blob/v9.3.1/IO/PLY/vtkPLYReader.cxx)与[Writer源码](https://github.com/Kitware/VTK/blob/v9.3.1/IO/PLY/vtkPLYWriter.cxx)。保拓扑简化约束见[固定DecimatePro接口](https://github.com/Kitware/VTK/blob/v9.3.1/Filters/Core/vtkDecimatePro.h)。

## 输出与后续转换

成功状态PASS_PREPARED_TOPOLOGY/exit0；输入拒绝2、后端版本/环境问题3、处理约束或后端失败4。输出独占，失败保留报告及实际产生的候选，不将残留文件视为成功结果。

- mesh_prep.json：输入/代码身份、`source_validation`、组件选择、前后拓扑、参数、采样偏差与限制。
- selected_before.ply：选定组件，简化前。
- prepared.ply：所有本阶段检查通过后才命名；失败候选叫candidate.ply。
- selection_map.json：原始面和顶点对应。

`source_validation.schema_version="mesh-source-link-v1"`，成功关联状态为 `PRODUCER_OUTPUT_BOUND`，保存 `source_schema/source_role/synthetic_input`、网格/来源记录/可选 plan 的完整文件绑定。合成来源的 plan 为 null；`photo_lineage_status` 与 `native_execution_status` 均保持 `NOT_VERIFIED`，不能把来源关联状态当作执行或照片认证。

准备报告现在绑定五个当前实现文件：`scripts/prepare_mesh.py`、`scripts/mesh_to_cad.py`、`integrations/meshprep/geometry.py`、`integrations/meshprep/ply_io.py`、`integrations/meshprep/provenance.py`。CAD 交付入口还会重算 `inputs.mesh/source_record` 的来源关联，并与已保存的 `source_validation` 完整比较。

后续优先使用 [CAD 交付入口](CAD_DELIVERY_USAGE.md)，其中示例已改为本轮新 `prepared/`。独立 `mesh_to_cad.py` 只负责几何转换，不能代替交付入口的来源复核。未定尺度时保留任意坐标；物理尺度/准确性及完整照片链仍未评估。

旧任意 `SOURCE.json`、仅绑定四个实现文件或缺少 `source_validation` 的 `mesh-prep-v1` 报告保留为历史证据，不能直接作为当前交付入口的输入。不得补写旧报告或覆盖历史基线使其看似通过新检查；新准备操作必须使用受支持的来源记录与全新输出目录。

## 当前验证

本轮：[MESH_SOURCE_LINK / HANDOFF](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/HANDOFF.md)。[pure02.xml](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/pure02.xml) 记录 111 项纯数据测试、0 失败、0 错误、0 跳过；这些用例不证明原生或照片重建能力。

本轮各执行一次 prepare 与 CAD 交付入口，监督记录均退出 0 并确认回收。[prepared/mesh_prep.json](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/prepared/mesh_prep.json) 记录合成球体 528→400 面、体积相对变化 0.6892%，双向最大采样距离 0.02740/0.01804，阈值 0.03464；typed `fixture_mesh` 来源关联和五文件实现绑定已保存。[CAD RESULT](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/cad_delivery/RESULT.json) 为 `COMPLETED_GEOMETRIC_DELIVERY`，未知尺度；原生转换报告记录一个有效闭合的 400 面实体，FCStd 重开、STEP 重导入及数值检查通过。未执行本轮定尺度路径，GUI/浏览器仍未验收，照片链和物理精度无结论。

## 历史验证（20261007T081724Z_main）

历史证据：[MESH_PREP / HANDOFF](../outputs/delegation/MESH_PREP/20261007T081724Z_main/HANDOFF.md)。该批次的旧来源记录和四文件报告不直接复用于当前入口。

最终95个纯数据测试通过；原生合成球528→400面、体积变化0.6892%，双向最大采样距离0.02740/0.01804（阈值0.03464）。开放球拒绝。400面结果真实转成单一有效封闭FreeCAD实体，FCStd重新打开与STEP重新导入均通过详细OCCT检查及source_extent_v1数值检查。没有GUI截图，没有真实照片网格或CAD精度结论。首次测试名过长及VTK属性调用失败保留，实际2次pytest、4个原生子进程；原生累计约5.73秒。
