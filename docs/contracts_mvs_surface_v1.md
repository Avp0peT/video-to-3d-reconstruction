# mvs-surface-run-v1

这是固定 OpenMVS 2.4.0 CPU 的计划/执行核心，将已给定去畸变图像、二值mask/validity与稀疏场景推进到可读回的稠密点云、原始表面。**不是整套照片→CAD入口**；不认证SAM3/B来源链、mask质量或物体完整度。真实执行须在独立任务额度下显式调用run，本模块实现不消耗A/B待执行额度。

## 场景与输入

只读MVSI v7 little-endian，接受固定Interface格式，不接受Boost MVS或旧版本。完整解析到EOF，字符串/vector先验长度界限；文件≤128MiB、图像≤256、累计相机≤256、poses≤4096、点/线各≤100万、观测≤1000万、字符串≤4096 bytes。所有浮点有限，旋转为proper R(1e-8)，K非归一化、宽高为正。图像依platform/camera/pose关联，绝对R=camera.R×pose.R、C=pose.Rᵀ×camera.C+pose.C。保留原K，不能当实际深度分辨率K。

本执行profile要求至少2图、非空稀疏点且每点至少2视图、transform为identity、所有maskName为空。global IDs必须唯一有效；point-view是images数组局部索引，不能错当global ID。图像路径必须相对、无父路径逃逸/Windows设备名/大小写等价冲突；保留Unicode原拼写，只归一化路径分隔符。

mask与validity分别由显式目录提供：图像stem.mask.png及stem.validity.png。所有scene图像逐项存在，不按文件名数字匹配ID；同stem冲突拒绝。图像为opaque L/RGB PNG，标签为原编码8-bit L PNG、0/255、单帧、无alpha/方向变换；尺寸与scene相机一致。前景非空，validity=0不能是前景。labels已在去畸变域，不能直接用原canonical mask。该核心按目录消费显式标签，不替代外包B的object-mask→mvs-mask身份链consumer；来源链始终NOT_VERIFIED，调用者须另完成B和形状/跨视检查。

计划包含scene、每图/mask/validity、指定EXE与全部同目录DLL、实现依赖的实际SHA-256/大小。固定CPU的两EXE哈希必须匹配；DLL按实际集合冻结，不宣称查明系统DLL/原生构建源码身份。input_origin仅为caller_declared，不能以参数自证真实数据。

## 计划、运行、目录与参数

`plan`只读验证并独占写一个JSON，不启动程序、不复制照片。`run`需要计划文件精确SHA与全新output，重算当前计划确保代码/工具/输入/选项未漂移；无resume、覆盖或自动重试。输出不能进入任何输入/工具根或原scene目录。

新output/work中的scene和全部相对图像树按字节复制，不重编码、不改scene字符串、不硬链接。工作文件前缀__mvs_保留，拒绝输入占用。图片、mask、validity和scene均复核复制hash。命令配置文件独立为空，dense配置在新工作目录从固定二进制默认生成；源码的其他隐藏默认仍属于固定binary行为，并非逐项实测。

阶段固定为Densify→Reconstruct，无重试，最多2个真实启动。两者cwd和-w均为work；线程参数/环境2，archive=1，verbosity=3。Dense固定1024最大边、640最小边、level1、5邻居、3估计/2几何迭代、保留dmap；postprocess/tower/ROI/坐标归一化等关闭。忽略标签只能0（masked）或-1（独立control）；两者须不同新目录，不复用缓存。全部固定参数见PLAN.options及实际argv。

Mesh关闭可选清理、补洞、平滑与裁剪。但[固定源码](https://github.com/cdcseacave/openMVS/blob/v2.4.0/apps/ReconstructMesh/ReconstructMesh.cpp#L424-L459)仍调用最终内部Clean。verbosity3在清理前保存__mvs_surface_raw.ply，因此两份PLY都必须存在、非空且可读；后续PC09以pre_cleanup_surface为原始输入。__mvs_surface.ply单独登记，不能冒充完全未清理产物。原生`.mvs`本轮检查非空并绑定，未独立解析Boost档案，不能称原生往返。

## 预算与回收

明确整数预算：总秒数10..1800；单阶段5..总秒数；Job提交内存128MiB..12GiB；work软输出阈值16MiB..8GiB。暂存固定成本+1MiB元数据余量必须小于输出阈值。整体deadline在执行前重核起点前建立；staging复制逐块检查、每阶段仅使用共享剩余时间；最终读回/输入复核后再次检查总耗时与目录字节，不能借第二阶段重置。

复用哈希固定的CAD supervisor，私有加载副本适配accounting、目录统计与日志失败登记：accounting失败绕过旧的继续处理分支，进入回收；目录不可测/链接等失败即停止。Windows目录项缓存可能返回零链接数，故用完整path.lstat读取身份；不把0当单链接放行。旧监督源码完全不修改。append失败原样重抛，同时私有记录其类型/事件；supervisor返回后将此前可能吞掉的reclaim/cleanup失败补入log_errors，阻止普通成功，**不保证即时停止**。Windows Job限制本阶段进程树的committed memory；0.1s采样时间/磁盘为软停止阈值，不是硬CPU/磁盘配额。父协调器的解析/读回通过尺寸界限约束，不在子Job内。每段停止后回收最多5s；失败回收/记录动作可能超出运行截止，总时限不是操作系统硬实时保证。work阈值含暂存/原生/阶段日志，父output下plan/result是额外小型元数据。

不得用“kill调用成功”代替退出确认。成功条件为status=completed、正整数PID、实际argv/cwd匹配、真实整型exit0、reclaim_confirmed=true、errors/log_errors均空。启动计数在收到有效PID后立即更新；监督调用未返回或创建身份不合法时native_launches=null并保留已确认创建数与attempts，不把未知记0。

## 输出验收与边界

Dense非空有限PLY；mesh的raw与final两份非空有限triangle PLY，拒绝索引异常/重复三角顶点/零面积等。单份PLY≤128MiB、vertices≤50万、faces≤100万，超界显式失败，不截取/简化。原始输入与暂存副本前后hash检查，任何漂移都失败；启动后的失败不继续mesh。

最高状态仅`SURFACE_GENERATED_NOT_ACCEPTED`，表示程序监督和非空几何读回通过。mask_consumption/mask_source_chain依然NOT_VERIFIED、quality NOT_EVALUATED、scale unknown/unit null、CAD NOT_RUN。还需真实mask消费对照、照片外形和未观察区审查、PC09网格准备、CAD及软件查看验收。

本轮没有运行Densify/Reconstruct或真实照片重建。合成控制流测试、微型监督进程与既有scene只读核对分列，不相互冒用证据。
