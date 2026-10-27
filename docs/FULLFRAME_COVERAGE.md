# 物体内候选覆盖与分级三维

该入口默认先使用已有 SAM 物体掩膜，再在物体内部均匀补点和进行较宽松的初始匹配。背景不播种，跟踪中心离开物体即结束；物体掩膜缺失时不选点。候选点数量不是完整外形的证明。

## 命令

在项目目录使用已有 pycolmap 4.2.1 Python 环境：

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/expand_video_coverage.py --pipeline '<已有视频pipeline目录>' --local-models '<reconstruct_partial输出的local目录>' --output '<新输出目录>' --execute --blender
```

省略 `--execute` 仅保存命令和实现绑定，计划与真实执行须用不同目录。省略 `--blender` 仍有候选叠图、逐点记录、PLY 与三维查看器。这是可重复运行的独立脚本，尚未改写原 `run_photo_demo.py` 默认行为。

`--selection-scope object` 是默认值；CLI、子进程、INPUTS/PLAN/RESULT 和逐点来源均记录该范围。只有显式传入 `--selection-scope whole-frame` 才启用全图诊断模式。历史全图结果保留，不回写。

## 选点与匹配

- 处理已有 2–96 帧、长边不超过1024的规范图像，复用已有视频时间和文件身份；不重新解码、分割、下载或安装。
- 每帧在非空物体 mask 的外接框内划分8列×12行，且角点中心必须在实际 mask 内（不是只检查外接框）；孔洞不选点。每格每轮最多6个真实Shi–Tomasi角点。`coverage-grid-lk-v3` 每帧重新补种，分格轮询防止前部格子占满预算；每格最多8条活跃轨迹，单帧最多768点，序列最多40000条轨迹。无纹理格保留为空，不生成装饰点。外接框使用整数半开区间，播种、活跃预算和计数使用同一格边界。
- 存活轨迹迁移后，拥挤格按轨迹长度、往返误差、LK误差、ID确定性保留8条，超额记 `GRID_BUDGET_RETIRED`。这是预算退休，不是几何失败或遮挡；逐帧记录原因，给欠覆盖格保留补种位置。第一轮实跑曾暴露背景轨迹占满总名额的问题，v1证据保留，v2没有改变匹配阈值。
- 同一帧一次排斥mask，按格ROI加角点窗口边界计算；活跃点周围8px不补新种子。角点阈值为格内0.005。
- 相邻已抽取帧执行金字塔LK和反向检查，往返≤3px、LK误差≤40保留；其中往返≤1px且误差≤25标为strong。其余仍留在候选账本，并非全部删除。LK误差是OpenCV亮度匹配量，不是像素位移。
- 失效轨迹结束；新出现角点使用新ID。终点碰撞按误差与ID确定性消解，不自动合并。没有实现被遮挡物理点的可靠重识别；光流连续不证明物理身份。
- 默认物体模式下所有保留观测都为 foreground。当前帧没有 mask 时，先结束活跃轨迹并记 `MASK_UNAVAILABLE`；空 mask 则记 `EMPTY_FOREGROUND`。两种情况都不播种、不做光流，也不使用相邻 mask 冒充当前帧识别。再次识别时新角点使用新 ID。光流中心移出 mask 记 `LEFT_FOREGROUND`，在终点冲突和格预算之前剔除。原 SAM 的 NO_MATCH/NEEDS_SELECTION 等状态保持；它们不证明物体实际不存在。
- 显式 whole-frame 诊断模式仍分别记录 foreground/background/unknown，可在未知 mask 帧跟踪；该模式不会由物体模式自动回退启用。

## 三维分层

只在已接受局部运行中注册的相机上求新点；相机与焦距不重新拟合，局部之间不拼接。原图、mask、帧ID、数据库image ID、原生相机和点、已绑定PLY互相核对。旧已保留点重新通过原门后继续保留。

1. `STRICT_SUPPORTED`：至少3视图，重投影≤2px、留一观测≤3px、射线夹角≥2°。
2. `RELAXED_SUPPORTED`：至少3视图，重投影≤4px、留一观测≤6px、射线夹角≥1°。放宽门槛带来更大误匹配和位置误差风险；这类点在图中用橙色区分。
3. `TWO_VIEW_PROVISIONAL`：恰2视图，正深度、重投影≤4px、射线夹角≥1°、Jacobian条件检查；没有留一验证，独立点层展示，不参与建面。

全部观测参与DLT，不删除误差较大的观测追求通过；背景和未知mask上的观测不进入物体三维。物体模式没有跨未知mask的连续轨迹。全图诊断模式若两次前景观测之间有已知背景观测，则整条局部候选拒绝；跨未知mask记录 `unknown_bridge_steps`，不据此宣称身份已确认。

按至少两个共同视图、所有共同观测距离≤2px做局部去重，重复原因保留；该规则不证明物理点相同。三视图候选优先于双视图暂定点，避免临时点占掉更强支持。每组最多3000点；超过1000支持点则保留点但不生成新面，不静默截断。

局部面仍需至少3个共同视图、投影与三维边长门、前景三角区域核对，不补孔。面内部是线性插值，不是已测量的稠密细节。

## 证据与限度

`INPUTS.json` 在光流和三角化前落盘；`TRACKS.json` 保留候选及失效原因，`CANDIDATE_LEDGER.json` 保留各层决策和去重；`COVERAGE.json` 记录每帧网格域、外接范围、格计数和mask状态。物体模式的候选与三维覆盖统计使用物体网格；另保留 `comparison_fixed_image_grid`（全图8×12坐标分区内与物体相交的格），供旧运行按同分区比较。这两种网格不能直接比较覆盖率。格子占用不是表面积或建模完成百分比。

外层监督300秒、2GiB、2线程，仅终止本任务进程树。输入与实现执行前后核hash。PLY逐数值回读；Blender另走已有创建与独立重开核验。

相机未独立标定、尺度未知，`geometry_accepted=false`、`quality_status=NOT_EVALUATED`。可见范围扩大不等于真实物体完整，尤其手持纸层可能遮挡、变化或变形。

书籍《Сборка макета》41–42页要求同一物理点对应及不同投影方向；本模块是工程透视补充，不能称为Kozlov原平行投影算法。参考[OpenCV角点](https://docs.opencv.org/4.13.0/d4/d8c/tutorial_py_shi_tomasi.html)与[光流](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html)。
