# 将已有原生模型接入研究结构

本入口把**已去畸变的针孔 SfM 分量**接到项目的 `PointSet3D` 和 `Camera`。它不会运行照片重建、分割、Kozlov 编码或 CAD 转换。精确定义见 [契约](contracts_research_native_v1.md)。

当前验证：155项纯数据测试通过；一个手算真值原生fixture和已有44图真实派生分量通过；实际CLI另跑一次退出0。真实侧录保持2393个原生点ID、9904条已三角化观测、57430个全部features。集合视图为2341个精确唯一坐标，52组重复坐标的全部原生记录仍在，不能把去重集合当作完整SfM记录。

## 已验证命令

在项目根目录执行，替换输出名为不存在的新目录：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:BLIS_NUM_THREADS = '2'
$env:NUMEXPR_NUM_THREADS = '2'
& '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe' -B `
  -m kozlov_lab.reconstruction.to_kozlov `
  --model '.\outputs\delegation\MVS_B\20261006T163538Z_r2\run_44img\undistorted\sparse' `
  --output '.\outputs\research_scene_demo01'
```

`research_scene.json`含完整描述侧录、point ID与坐标数组、ID→集合行映射、研究相机K/R/t/P、单位矩阵正逆变换、未知尺度和来源hash。`RESULT.json`绑定场景文件及指定实现文件。已存在输出目录会被拒绝。

这条命令只证明所指原生模型的适配。T01原frame_id、native frame ID与源分量model_id不相同，普通CLI不会按文件名猜关联。本轮较完整的上游来源核对、44个T01关联以及筛点/去畸变比较，另在`outputs/delegation/T03/20261007T102949Z_main/native01/real44_scene/`保存；`cli01/`只包含通用CLI证明。

## Python 调用

```python
from kozlov_lab.reconstruction.to_kozlov import load_native_scene
scene = load_native_scene(model_directory)
body = scene.body                         # PointSet3D：唯一坐标集合
native_point_ids = scene.point_ids         # 全部原生ID，不是body行号
body_row = scene.point_id_to_body_row[int(native_point_ids[0])]
camera = scene.cameras[native_image_id]
pixels = camera.project(scene.native_xyz) # 仅投影可见/有效点时使用
center, world_rays = scene.rays(native_image_id, [[u, v]])
```

`scene.rays`直接按K逆和R构造世界射线，不调用旧`Camera.ray`的SVD实现；本轮不验证或修改那个旧方法。需要序列化时调用`write_scene(scene, new_output)`。返回对象可供计算，调用方应避免修改原始侧录后仍引用原验收状态。

不要直接把大点集送入组合枚举、全配对diameter或小规模仿射oracle。适配模块按线性规模保存/检查数据，但这些研究算法各自仍有规模与定义边界。

## 结果含义

- 原生和研究投影最大差约6.82e-13像素；它表示数值兼容，不是物体准确率。
- 已有原始畸变模型与去畸变模型的57430条射线另行比较，最大差约4.16e-17；没有重跑去畸变。
- 真实输入已在上游从2413点删去20点，当前适配的是2393点派生分量；不声称原始分量全部点均被保留。
- 52组“仅xyz相同”与此前MVS的48组“xyz、RGB、视图成员均相同”定义不同，不能混用这两个计数。
- 坐标和尺度保持原样，`scale_status=unknown,length_unit=null`。残差、投影兼容、物理精度和Kozlov研究收益分别陈述；后两者尚未验证。

本轮未运行SAM3、GPU、SfM、MVS、稠密网格或CAD。证据入口：[HANDOFF](../outputs/delegation/T03/20261007T102949Z_main/HANDOFF.md)。
