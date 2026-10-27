# 原生针孔模型到研究结构的契约

版本 `research-native-pinhole-v1`，2026-10-07。对应 T03 / RC08 的受限工程适配，不是 Kozlov 算法、三维骨架或精度结论。它不改变 photo-io-v1 或 sfm-native-v1；原生文件仍是权威来源。

## 输入域与拒绝条件

输入是 pycolmap 4.2.1 的一个已经去畸变的原生 binary 分量，必须有 cameras、images、points3D、rigs、frames 五个 bin 文件。混有 text 模型时拒绝。只接受 PINHOLE 和 SIMPLE_PINHOLE，包括参数、尺寸、全部已注册图像与完整 features；径向模型即使畸变系数为零也拒绝，不在此模块偷偷去畸变。至少两图一点，每点至少两个不同视图，track 与 image feature 双向一致。不把最低结构要求当作物体覆盖或质量要求。

显式上限：256 图、100000 原生点、1000000 个全部 features、原生五文件合计64MiB、导出JSON128MiB。拒绝非有限坐标/位姿/观测、非正焦距、非单位四元数、重复 ID、计数不符、未注册图、断裂引用或相机前方条件不满足的已三角化观测。ID 保持整数；未三角化 sentinel 为 uint64 max，不改为0或猜测连续编号。上限控制输入规模，不是操作系统硬内存或运行时间看守。

## 坐标、单位与身份

- 原世界坐标、轴向和数值逐点保留。`native_to_research` 与 `research_to_native` 都是4×4单位矩阵，不居中、不归一、不翻轴。
- 使用 `Xc = R Xw + t`；R来自原xyzw四元数，K来自已去畸变针孔参数。Camera保持原像素坐标（COLMAP左上像素中心0.5,0.5），不与T01的0,0约定混用，不额外加减0.5。
- 全部原生点按整数ID排序，`native_xyz`和`point_ids`逐行对应。原生RGB、tracks、全部points2D（含未三角化点）、图像名、相机/原生frame/rig关系保存在`native_description`；原生point.error另存`native_point_errors`。该侧录来自既有describe_model，不声称包含原生格式的全部未来字段。
- 既有PointSet3D名义上是集合，但构造器未强制唯一/有限；适配器先校验并按**精确xyz相等**构造集合视图。不同原生ID即使坐标相等仍保留全部侧录，以`point_id_to_body_row`和`duplicate_groups`说明多对一关系。不能用容差去重或把同坐标不同轨迹混为一个原生点。
- `scale_status=unknown`、`length_unit=null`始终保持；数值坐标不成为物理毫米。原生frame ID不是T01 frame_id。上游frames_map和筛点/去畸变来源链由调用方另行绑定验证，不能从文件名推断。

## 返回、导出与数值证据

`adapt_description()` 是仅需NumPy的纯数据入口，返回ResearchScene；其body为既有PointSet3D，cameras按原生image ID索引，使用既有Camera.from_parts。`load_native_scene()`延迟导入pycolmap、核对版本/源文件前后hash，并以原生projection和inverse-intrinsics对照研究Camera。只做读取，不执行重建、特征或优化。

数值兼容门：已三角化观测点的原生/研究投影绝对差≤1e-8像素；所有features逆K射线差≤1e-12。这是双精度实现兼容阈值，**不是**重投影残差、照片覆盖或物理准确性要求。原观测残差单独报告，不以低残差声称精度。

需要世界射线时调用`ResearchScene.rays(image_id, pixel_pairs)`，返回中心`-Rᵀt`及方向`RᵀK⁻¹[u,v,1]`，不归一为单位长度。适配器不使用既有Camera.ray的SVD实现，也不凭本轮投影检查宣称那个旧方法已验证。原生rig/frame的sensor成员、data_ids及位姿组合另作一致性检查。

`write_scene()`只写项目outputs下全新目录：research_scene.json与RESULT.json，包含坐标侧录、集合映射、研究相机、来源与指定实现hash。产物可以被普通JSON工具读取；直接使用Python返回对象可进入研究计算。未实现通用native模型逆写或原生重建器，也不运行全点云的组合算法、diameter或仿射oracle。

命令：

```powershell
& '<项目Python绝对路径>' -B -m kozlov_lab.reconstruction.to_kozlov --model '<已去畸变binary模型目录>' --output '<项目outputs下新目录>'
```

正常导出exit0。已识别输入/文件/导入错误exit2；其他未预期原生错误以非零退出并保留日志，不生成成功报告。调用方应设置线程和外层超时。本轮未授权读取不可信超大模型或运行昂贵研究算法。

## 结论边界

已知真值fixture和已有真实44图分量分别验收。真实模型为2393点的保守派生物，原44图分量的2413点已有20点在上游被删除，本适配器不把它们恢复或隐去。集合视图不能代替原生多重记录；适配成功不能证明Kozlov收益、表面、实体或照片/视频到CAD完成。
