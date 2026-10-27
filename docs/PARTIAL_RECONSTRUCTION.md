# 局部重建与缺测保留

用户允许覆盖不足的部位留孔。`scripts/reconstruct_partial.py` 因此从已有前景 SfM 数据库自动安排局部重建，而不要求完整覆盖或把独立分量强行拼接。它是独立执行入口，旧 `post_review` 的只读性质不变。

## 命令

在项目目录，使用已有 pycolmap 4.2.1 环境执行：

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/reconstruct_partial.py --sfm-run '<原始run.json绝对路径>' --output '<新目录绝对路径>' --execute --blender
```

省略 `--execute` 只生成计划；计划和真实执行必须分别使用新目录。省略 `--blender` 仍生成 PLY、查看器、逐点账本及图像叠点。已有 `.blend` 导出器另启两个有界 Blender 进程，分别创建、独立回读，不操作用户当前 Blender 场景。

## 固定版本 local-three-view-k0-v1

- 输入最多32张已分割、身份绑定的图，使用原SQLite几何验证匹配；不重跑SAM3、特征提取或匹配，不改原数据库。
- 连通图分组，匹配边至少15对；同一连通轨迹若包含同图两个feature row即记冲突。每组至少3个视图、15条无冲突三视图轨迹才安排。所有孤立图、冲突及未安排组均记录。
- 每组一次 `incremental_mapping`，45秒原生时间约束、单模型；固定原DB的零径向畸变与主点，共享焦距允许精炼。禁止结构不足时的无结构注册回退。随后一次100迭代/15秒最终BA，固定两相机规范自由度。不是在原历史模型上修改坐标，也不是旧运行的严格消融重放。
- 主进程冻结输入与实现，Job外层480秒、2GiB、2线程；Job回收和来源不漂移均须确认。不会自动重试或改变阈值。
- 最终BA必须为CONVERGENCE。每点至少3个不同视图，原DB feature row/像素及原生track双向一致、无来源冲突、全部在前景内、正深度、每观测误差≤2px、原像素反解世界射线最大夹角≥2度、XYZ Jacobian条件数≤1e4。固定相机逐个留一观测DLT预测误差≤3px。
- 相机本身由同一数据拟合，原几何内点也经过选择，故留一是条件内部一致性检查，不能称独立精度验证。相机拟合仍可使用组内所有原生匹配，最终点门不意味着相机只用通过门的点估计。
- 每模型超过1000点则整项拒绝，不静默采样。输出保留所有候选的接受/拒绝账本。
- 面仅在保留点上作局部线性插值：三顶点有至少3个共同视图、各视图边长≤60px、3D边≤点集最近邻距离中位数3倍、三角形非退化、不重复、不产生超过两面的边；投影三角区域不能跨前景mask孔洞。未支持的区域不封口。不把插值面内部称作测量细节。
- 每组独立坐标，不拼接；允许无面的点云交付；模型、PLY及Blender都独立回读。

## 结论边界

零畸变与共享内参是本轮工作假设，不是物理标定。`PARTIAL_CONDITIONAL_MODEL` 仅指通过固定内部检查的局部候选，不代表外形真值、毫米精度、闭合实体或CAD验收。`geometry_accepted=false`、尺度unknown保持。`NO_SUPPORTED_LOCAL_MODEL` 与软件/导出错误 `FAILED` 分开；组错误不得冒充几何拒绝。

书籍《Сборка макета》印刷页41–42说明点标签与不同投影方向的前提，47–48页AlgT′只保证其平行投影域下的仿射等价。本入口是工程透视几何补充，不能冒称Kozlov原算法。参考 [COLMAP内参约束](https://colmap.github.io/faq.html#fix-intrinsics)；实际API以本地4.2.1为准。

本轮结果：`outputs/delegation/PPP_VIDEO/20261014_object_first01/partial_reconstruction_20261016/`。旧模型与报告保留，不因新候选生成而改写旧验收。
