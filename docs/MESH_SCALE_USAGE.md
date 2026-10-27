# 参考尺寸定尺度与 CAD 接入

当前状态：TESTED_SYNTHETIC。只证明参考计算、网格变换和CAD几何转换，尚无真实实物测量、照片外形或最终CAD端点精度验收。

流程：prepare_mesh.py的prepared.ply/mesh_prep.json → scale_mesh.py → scaled.ply/scale_ledger.json → scaled_mesh_to_cad.py → 原mesh_to_cad.py → FCStd/STEP。原转换器未修改；没有参考时继续使用它的unknown/arbitrary方式，不生成虚构参考。

## 参考记录

参考必须绑定当前最终PLY的SHA256及零基顶点编号。简化前selection_map.json不能代替简化后顶点身份；不支持无归属XYZ或照片像素直接作端点。

~~~json
{
  "schema_version": "mesh-scale-request-v1",
  "source_mesh_sha256": "<prepared.ply完整小写SHA256>",
  "data_kind": "real",
  "reference": {
    "vertex_indices": [12, 87],
    "length": 25.0,
    "unit": "mm",
    "label": "两个可辨认标记之间的卡尺尺寸",
    "correspondence": "说明实物标记与当前网格12/87顶点的对应方式",
    "evidence": {"path": "measurement_record.json", "sha256": "<证据文件SHA256>"}
  },
  "checks": []
}
~~~

示例占位值不能作为实测数据。证据相对路径以参考文件为基准；证据≤16MiB，参考JSON≤1MiB，拒绝重复键/非有限值。单位支持mm/cm/m/in；长度有限且为正。

默认正均匀缩放、不平移；可显式增加origin_vertex以该源顶点为新原点。实际计算x_mm=s*(x_source-origin)，记录比例、原点和正反4×4矩阵。

checks最多64条，与reference字段相同，可加allowed_absolute_error_mm（有限且≥0）。禁止复用拟合端点对、反向同一对或重复检查对。没有阈值只报告差值，within_threshold=null。所有差值只针对所列长度，不能代表整个物体精度。

真实输入必须来自成功的mesh-prep-v1记录，核对PASS/exit0、产物path/hash/bytes及synthetic_input一致。合成算术夹具还允许mesh-scale-fixture-source-v1/data_kind=synthetic/mesh绑定。未知来源格式拒绝。这只是相邻处理步骤一致性，不认证上游照片链或外部测量真实性。

## 调用

在项目根目录使用现有项目Python执行（替换占位路径；全部output必须在项目outputs内且全新）：

~~~powershell
& '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe' -B .\scripts\scale_mesh.py --mesh 'C:\path\prepared.ply' --reference 'C:\path\reference.json' --source-run 'C:\path\mesh_prep.json' --output '.\outputs\my_object_scale'
~~~

CAD阶段用独立PowerShell进程，只调整该进程环境：

~~~powershell
$fc = Join-Path $PWD 'outputs\toolchains\freecad-1.1.4\FreeCAD_1.1.4-Windows-x86_64-py311\bin'
$env:PATH = $fc + ';' + $env:PATH
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
& (Join-Path $fc 'python.exe') -I -B .\scripts\scaled_mesh_to_cad.py --ledger '.\outputs\my_object_scale\scale_ledger.json' --output '.\outputs\my_object_cad'
$LASTEXITCODE
~~~

CAD默认最多10000面，不自动简化。合成标记从核对后的来源继承，不能另传flag改变。阶段预算需由外部监督器施加；本次scale≤30秒、CAD≤60秒，数值线程2，FreeCAD限制自身最多两个CPU。没有硬RAM上限。

## 验证与结果语义

- 输入使用既有严格PLY解析：≤64MiB、≤400000点/200000面。保留点序与面索引/绕序，不删点填洞；被丢弃的附加标量属性写入台账。
- double PLY落盘回读，坐标等于实际计算结果；拒绝完整XYZ坍缩、三角退化或溢出。每条三角边必须符合统一比例，固定相对数值门槛1e-12、绝对容差0。
- 回逆策略per_component_origin_relative_ulp_v2：逐分量8 ULP，并将绝对回逆误差限制在实际参考段源长度的1e-12内；不按世界坐标大数放宽。原始参考距离在落盘坐标中重新检查，相对容差1e-12。均为浮点数值门槛，不是物理精度。
- CAD包装前重算源网格、参考、全部输出顶点/面、变换和检查，不只相信status或hash。严格区分布尔和数字，拒绝以0替换false。任意声明检查阈值失败时不启动CAD。
- 独立长度检查对象是scaled PLY。CAD仍使用原转换器的几何/体积/包围盒容差，未逐个对应FCStd/STEP顶点重测端点。不能把PLY的微小回算残差冒充最终CAD实测精度。
- physical_accuracy_status始终NOT_EVALUATED，上游变换历史/照片链NOT_VERIFIED。来源、测量证据及实现按hash绑定；移动或修改它们会使复核失败，尚未做可搬运部署。

尺度入口RESULT.json需exit0/PASS_REFERENCE_SCALING；拒绝exit2，内部错误exit4。失败产物保留、不覆盖、不自动重试。CAD最终以scale_verification.json及真实退出码为准；若后置身份检查失败，旧几何报告PASS不能代表整个包装通过。

## 当前证据

outputs/delegation/MESH_SCALE/20261007T090514Z_main/HANDOFF.md：最终117项通过。202点/400面既有合成球以50倍定尺度，600条边最大相对变化8.17e-16；真实FreeCAD获得有效闭合单实体，FCStd/STEP回读通过，包围盒[-50,-50,-100]至[50,50,0]毫米约定坐标。

原double PLY到CAD体积相对差3.079e-8，在转换器1e-7数值阈值内。参考和检查数字都是合成夹具约定值，不是实物测量。真实物体、GUI、CAD端点复测尚未完成。
