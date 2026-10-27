# 时间窗深度诊断与局部 MVS 实验

此流程用于检查全视频重建的局部缺失是否与跨时间段证据不一致有关。它不按物体类别写死形状、不补造缺失部分，不把时间窗筛选称作新的测量。

## 可重复入口

1. `scripts/audit_temporal_depth.py --source SOURCE --ledger-run LEDGER --region-json ROI --mask-run MASKS --output NEW_ANALYSIS`
2. `scripts/prepare_temporal_mvs_subset.py --source SOURCE --analysis NEW_ANALYSIS --window-seconds 2 --output NEW_SUBSET`
3. 使用 `NEW_SUBSET/DERIVATION.json` 中的 InterfaceCOLMAP 参数导入；随后原生导回并调用 `scripts/prepare_mvs.py --internal verify`。导入成功不替代回读校验。
4. 以 `scripts/build_mvs_surface.py plan/run` 在新目录运行既有低纹理配置，再用 `scripts/compare_temporal_surface.py` 比较。

这些入口都拒绝覆盖旧结果。新输出和旧数据应位于互不包含的目录。步骤2需已有的 pycolmap 4.2.1 环境；步骤1只读现有数据，不加载 SAM3、不运行 MVS。当前属于显式诊断流程，尚未设为默认建模策略。

## 固定比较规则

- 时间使用已绑定 `decode_index → canonical mask path/hash → pts × time_base` 的精确分数；不用 `input_index` 或文件顺序猜时间。
- ±1、±2、±4秒全部报告，不根据结果隐去窗口。候选总分母相同；生成视角必须在窗内，目标与生成视角均不投支持票。
- 延续已有的2%深度一致、至少3个其它视角、两两8°方向门。unknown、普通遮挡 behind 和位于已知深度前方的 front conflict 分列。
- 每窗的局部/全视角拒绝都使用同一支持集合。近远视角机会数不同，且近处支持已条件化，因此拒绝率差不是形变或相机错误的因果证明。
- 旧深度本身是全视频 MVS 计算的。仅筛其来源视角不会撤销窗外邻居对深度估计的影响；新的局部 MVS 才实际限制估计输入。
- 首个原生实验固定±2秒作为中间尺度，不称最佳窗口。派生时保留原PINHOLE相机、XYZ/RGB和观察，只截断轨迹；至少两个保留观察的点才进入子集。通过独立 `general_subset.compare` 和序列化回读。
- InterfaceCOLMAP 的半像素主点约定仅用于相机域比较，不再次移动模型或图像。
- 表面比较看同相机前景、背景和已绑定缝隙占用，以及固定缺失区。投影覆盖增加不能自动证明三维深度正确；不能通过桥接空白区域来冒充修复。

## 本轮证据

2026-10-25：`outputs/delegation/SHAPE_REPAIR/20261025_temporal01/` 保存两次只读诊断（第一版与输入契约补强后的终态）、独立子集、原生导入/导出/回读、一次稠密和网格实验。所有运行都有输出、退出和回收记录。未修改既有默认模型。

## 依据与限制

先核对本地书籍审查 `outputs/delegation/PPP_VIDEO/20261014_object_first01/book_code_audit_20261015/SOURCE_REVIEW.md`：平行/仿射投影的精确相容性不能直接代替任意透视视频的相机与深度估计。

COLMAP 官方说明相机优化后应重跑稠密重建，不能沿用旧坐标假设：[COLMAP FAQ](https://colmap.github.io/faq.html)。非刚体场景可能需要显式形变建模：[NRMVS, WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/papers/Innmann_NRMVS_Non-Rigid_Multi-view_Stereo_WACV_2020_paper.pdf)。本实验未实现 NRMVS，也未仅凭冲突就确证手套发生了导致失败的形变。
