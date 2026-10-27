# sfm-native-profile-v1：CAD-02A 配置扩展

日期：2026-10-03。这是主线冻结的扩展契约；不回写 `contracts_sfm_native_v1.md` 或 `contracts.md`。仅定义本次共享相机实验，运行授权与预算见 `delegation/CAD02A_SHARED_CAMERA_EXPERIMENT.txt`。

## 版本与兼容

CLI 新增 `--profile`，仅两个合法值：`t02a-v1`（默认）和 `cad02a-shared-v1`。无参数与显式 `t02a-v1` 保持旧配置、输入验证、状态/退出码、坐标与导出语义；默认 run 的 `contract_version` 仍为 `sfm-native-v1`，不因增加 selector 改写原 effective_options。执行日志可增补。

显式共享模式使用 `contract_version=sfm-native-profile-v1`，新增顶层 `profile_id=cad02a-shared-v1`；`effective_options.profile_id` 同值，`effective_options.camera_sharing=single`，实际提取选项记录 `camera_mode=SINGLE`。其他字段沿用旧契约。不得让共享 run 冒充符合旧契约的独立相机配置。

实现摘要继续对 `src/kozlov_lab/reconstruction/` 顶层全部 Python 文件，按项目相对路径排序的 `[路径,文件SHA256]` 列表做规范 JSON SHA256。新增文件必须纳入；禁止在该目录新增未被算法覆盖的子包。外部监督器/驱动另入运行证据绑定。新旧代码内容不同，新摘要必须不同；旧摘要只属于保留的旧快照。

input_fingerprint 沿用旧字段和规范 JSON 算法，但 `contract_version` 必须取本次实际版本，effective_options 必含所选共享 profile。最终摘要绑定完成后的实际选项。旧模式不新增 profile 字段到指纹载荷，仍会因新版实现摘要而得到新指纹；不能要求新版复制历史指纹。不用缓存。

## 唯一配置变化

共享模式仅将提取阶段的 `pycolmap.CameraMode.PER_IMAGE` 改为 `SINGLE`。不新增用户任意内参参数，不改初始焦距/主点/畸变、匹配/验证选项、BA、seed 或数据。保留版本自身的下游行为；共享 ID 触发的后端分支属于干预效果，不能强行恢复旧匹配图。

同一批 canonical 图必须尺寸相同，且有适用共享相机的输入依据。本任务固定十二张 1024×683；尺寸混合时共享模式在启动原生阶段前 INPUT_ERROR，禁止重采样迁就。输入身份/去重继续遵循 photo-io-v1。

数据库提取后必须验证：恰有十二个唯一图像记录、一台相机、每个图像引用该 camera_id；相机为 SIMPLE_RADIAL、1024×683、初始 `[1228.8,512,341.5,0]`，无真实焦距标定声明。以真实 names→IDs→camera_id 校验，不假定 ID 连续。原生各模型内也必须验证所有注册图像引用同一有效相机并保存原有 rig/frame/track 关联；不同模型优化出的相机参数可不同，绝不据同 ID 拼接坐标。

固定 CPU、线程配置 2、长边上限 1024、唯一图上限 12、SIFT max_num_features 配置 4096；实际 keypoints.rows 独立报告，不将配置值解释为硬行数上限。mapper seed=0、min_model_size=2、multiple_models=True、max_num_models=12、优化焦距与径向项、不自由优化主点；全部其他默认从同一 pycolmap 4.2.1 读取并与固定对照比较。

## 输出与失败

输出仍限制在项目 `outputs/delegation/T02A/`；本任务使用其 `CAD02A/<task_id>/` 新父目录。全部原生组件 binary/text 导出和父子进程往返比对沿用旧语义。共享验证失败为 BACKEND_ERROR，不能吞掉后改用独立相机。部分注册仍按旧 SUCCESS/PARTIAL/NO_MODEL 规则报告；实验预测另列，不混入 CLI status。

`quality_status=NOT_EVALUATED`、`scale_status=unknown`、`length_unit=null`、`reconstruction_scope=scene_sparse`、`mask_usage=not_applied`。本契约不承诺真实形状、精度、连通、稠密或 CAD 成体。
