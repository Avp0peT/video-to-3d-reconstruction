# 开放曲面交付

2026-10-11 的实测样例已从照片重建网格得到有效开放 CAD 曲面。用户允许未识别区域留孔，因此开放曲面可以独立交付；闭合实体验收规则保持独立。

## 当前可用产物

- 交付首页：`outputs/delegation/PROJECT_FINISH/20261011_delivery_C/index.html`
- 网格：`outputs/delegation/PROJECT_FINISH/20261011_open_repair_A/candidate01/open_surface.ply`
- FreeCAD：`outputs/delegation/PROJECT_FINISH/20261011_open_cad_A/cad01/model.FCStd`
- STEP：同目录 `model.step`
- 原生检查和回读证据：同目录 `conversion_report.json`
- 总包复核：`outputs/delegation/PROJECT_FINISH/20261011_open_surface_main/FINAL_ACCEPTANCE.json`

保留 5000 个顶点、9996 个三角面。仅删除已定位的 4 个相交面，生成 4 个三角缺口、12 条边界边；删除面积占原简化网格约 0.030745%。原始网格和简化前后记录均保留。这些缺口是修复产生的，不能据此认定物体本身有相同孔洞。

FreeCAD 1.1.4 实际检查了网格自交、非流形、朝向和邻域；BRep 以及重新读取的 FCStd/STEP 均有效，详细 `check(True)` 通过。三个 CAD 检查对象均为 9996 面、12 条边界边、0 个实体；面积和包围盒在固定容差内保持。

## 两个独立入口

`scripts/repair_open_mesh.py` 接收原网格、绑定的原生交叉面诊断、上游网格准备报告及三个 SHA256。它在新目录生成候选，保留其余坐标、面序和绕向，并输出 `repair_report.json`、`face_map.json`。仅写出候选不等于通过原生几何检查。

`scripts/open_mesh_to_cad.py` 用 FreeCAD 自带 Python 执行，接收 `--mesh`、`--mesh-sha256`、`--source-report`、`--source-report-sha256`、`--output`。它核对修复来源链，允许合法开放边界，拒绝自交及非流形，不补洞、不构造 Solid。输出目录必须是新的。实测命令和既有 Job 监督记录位于 `20261011_open_cad_A/entry01/SUPERVISOR_RESULT.json`；原实跑额度已消费，不把旧命令重复执行称为同一次运行。

新入口不修改旧 `scripts/mesh_to_cad.py` 的闭合实体验收要求。新样例的成功也不改写早先被拒绝的 CAD 记录。

## 实用边界

- 可以在支持这些格式的软件里加载该多面曲面；FreeCAD 原生 API 已实际读取，交互 GUI 的查看仍未验收。
- 坐标来自任意尺度重建。CAD 文件的毫米约定不提供真实尺寸，不能据此作精密测量。
- 此交付没有恢复参数化设计历史、解析工程曲面或闭合实体。
- 照片外形精度、背景排除效果和跨视一致性尚无完整验收。单一样例成功不等于任意照片或视频的一键产品验收。
- Kozlov 理论的研究实现与当前几何工程链单独验收；本次不增加该理论在真实照片上的证明。
