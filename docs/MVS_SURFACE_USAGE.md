# 稠密与表面执行核心

入口`scripts/build_mvs_surface.py`，规则见[契约](contracts_mvs_surface_v1.md)。使用当前项目环境的Python/NumPy/Pillow与已固定的OpenMVS CPU工具，无自动安装或下载。默认只有明确的plan子命令；真实运行必须另选run。

1. 准备MVSI7场景及其相对图像树。以原scene的原生image全列表为准。准备去畸变mask、validity目录，分别命名`图像stem.mask.png`、`图像stem.validity.png`，每张图必须完整提供。
2. 用`plan`传入`--scene-path --image-root --mask-root --validity-root --tool-root`，再给出整数`--total-seconds --stage-seconds --memory-bytes --output-bytes`及全新`--plan-output`。`--input-origin real`必须同时传`--photo-source ABS_SPEC_JSON`；`synthetic`或默认`unverified`不得传`--photo-source`。默认ignore-label0；独立对照才设-1。
3. 复核计划的所有文件、SHA、参数、预算与来源。脚本输出该计划的实际SHA-256。plan不启动MVS重建进程；真实照片来源核验会用pycolmap 4.2.1读取既有模型的图像身份元数据，不重新求解SfM。
4. 获得相应任务执行额度后，以`run --plan ABS_PLAN --plan-sha256 EXACT_HASH --output ABS_NEW_DIRECTORY`执行一次。输出父目录须先存在；失败保留结果，不覆盖或自动再跑。

以上为用法，**不是本轮真实运行指令**。外包A pilot03/B consumer仍按各自任务书进行。新增照片输入核验只连接图像身份与当前文件字节；此核心仍不负责读取B完整manifest或认证从T01/SAM3到mask的来源链，在自动产品路径中仍需将B产物核验后接入。

## 真实照片输入说明文件

`--photo-source`指向JSON文件，`schema_version`固定为`surface-photo-input-v1`。`bridge_run`、`sfm_run`、`io_run`、`dataset_manifest`均须为绝对路径；`model_id`须明确给出，不能根据最大模型或目录顺序猜选。例如，既有44图桥接对应：

```json
{
  "schema_version": "surface-photo-input-v1",
  "bridge_run": "C:/Users/13944/Desktop/train/kozlov-visual-lab/outputs/delegation/MVS_B/20261006T163538Z_r2/run_44img",
  "sfm_run": "C:/Users/13944/Desktop/train/kozlov-visual-lab/outputs/delegation/T02A/CAD_FULL66/20261006T153820Z_full66/20261006T153915Z_88271f72",
  "io_run": "C:/Users/13944/Desktop/train/kozlov-visual-lab/outputs/delegation/T01A/CAD_FULL66/20261006T153820Z_full66/20261006T153857Z_5c3795b195b6",
  "dataset_manifest": "C:/Users/13944/Desktop/train/kozlov-visual-lab/data/real/skull_flash_66_0f31c229_1024/manifest.json",
  "model_id": "0"
}
```

本例`--scene-path`为`bridge_run/scene.mvs`，**`--image-root`必须是`bridge_run`根目录，不是其`undistorted`目录**。场景原生名称为`undistorted/images/images/000003.png`，由image root解析后指向实际去畸变图片。核验采用已核对的固定桥接布局`bridge_run/undistorted/images`，拼接完整native名称`images/000003.png`；该目录约定与既有PLAN的`mvs_import --image-folder`相符，但核验器不泛化解释任意历史argv。两条规范物理路径须一致。不能通过basename或`image_id-1`建立关联；scene局部index、global_id与native image_id分别记录并显式核对。

所选视图通过SfM `frames_map.jsonl`与T01 `frames.jsonl`连接到dataset输入和canonical输出，实际文件哈希分别核验；同一native view的aliases只能共享相同canonical字节。T01 `image_sha256`属于其输入文件，不能直接替代canonical输出哈希。当前dataset输入是预处理PNG，因此这一步也不等于认证原始相机照片。

照片连接结果保存在计划`photo_inputs`中。`PHOTO_INPUT_IDENTITIES_BOUND`仅表示当前视图、路径及文件字节的关联成立；`geometry_lineage`、`mask_source_chain`、`raw_camera_capture`仍为`NOT_VERIFIED`。所选点集的派生过程、去畸变像素的生成过程、scene几何的历史来源均未认证，不能据此宣称真实几何、完整照片重建链、分割质量或CAD交付通过。

本轮来源连接的证据入口为[交接记录](../outputs/delegation/SURFACE_PHOTO_LINK/20261007T153400Z_main/HANDOFF.md)，最终统计与结论以该任务实际落盘记录为准。

运行目录包含plan.json、result.json及work。work保留字节复制的场景/图像、mask/validity、dmap、dense.mvs/PLY、surface_raw.ply、surface.ply/MVS、生成配置及两个阶段日志（内部实际文件带__mvs_前缀）。**PC09优先使用__mvs_surface_raw.ply**；final PLY经过上游不可完全关闭的内部清理，单独留存。

exit0仅为`SURFACE_GENERATED_NOT_ACCEPTED`：非空有限几何，不是质量、消费或CAD通过。exit2通常为启动前输入拒绝，exit4为执行/读回/输入漂移失败，exit5为本入口可识别的总预算/末尾阈值失败；原生监督超时等详情保留在原始supervisor结果，不能只看外层exit4猜原因。

最终145项深度读取测试属于上一模块，本模块另有自己的scene/runner测试和证据，不借用历史绿灯。真实表面首次运行尚未执行。

## dense 后的 mask 加载失败日志守卫

2026-10-19 视频接入补充：通用子集派生与真实照片身份核验现接收既有 `sfm-native-object-video-v1` 契约，支持视频 profile 的192行输入账本和部分注册视图；仍核对源哈希、行身份与计数。仅扩展该已定义契约的白名单，不放宽几何或资源门限，也不更改下文固定43视图历史入口。此次三组定向测试85例通过；PPP实际续接已生成可由Blender独立重开的表面，但外形仍明显失真，不属于几何质量验收通过。证据：[PPP本轮交接](../outputs/delegation/PPP_FULL/20261019_full01/HANDOFF.md)。

当前生产 recipe/分辨率不变。dense 原生监督成功后、任何 mesh 启动前，入口核对真实 `mvs_surface_adapter.output` 日志目录，读取该目录的 stdout.txt/stderr.txt 与本次 work 根目录全部 DensifyPointCloud-*.log；原生日志至少一份且非空，缺失、不可读、超过16MiB、含NUL或读时漂移均拒绝，exit4且不启动mesh。

按冻结OpenMVS2.4源码捕获 `warning: can not load the segmentation mask` 和 `error: cannot load mask image`，在任一流/原生日志出现即失败，`stages[].mask_log_guard`保留实际文件hash与诊断位置。无法取得日志时记录LOG_EVIDENCE_UNAVAILABLE。日志按字节扫描以兼容Windows ANSI。

NO_MASK_LOAD_DIAGNOSTIC_FOUND只是未发现这两条已知诊断，不证明逐图加载成功、深度标签排除或降采样对齐；mask_consumption仍NOT_VERIFIED。新plan须重新生成并绑定更新后的runner；本守卫不更改冻结参数、不运行额外对照、不放宽DMAP verifier。

## 显式43视图派生（缺59）

独立入口 `scripts/derive_native_subset.py --clean-parent ABS_CLEAN44_SPARSE --sfm-run ABS_EXISTING_SFM --io-run ABS_T01 --output ABS_NEW_OUTPUT`。不得覆写历史目录；本任务实际派生另行批准。固定PyCOLMAP4.2.1，仅在新副本deregister_frame(60)，输入59必须对应单图native image60/frame60，禁BA/重三角化/renumber。入口协作预算30秒，原生读取/写入的硬停止仍需外层已有监督器；失败保留结果且不重试。

输出 `sparse/` 五个native bin及 `subset_certificate.json`。证书不是信任PASS：验证者重新读原SfM model0、既有clean44和derived43，逐项比较所有注册视图身份/相机/位姿、XYZ/RGB/点ID、保留2D坐标/双向tracks；clean44必须恰为原模型删除20个重复image-track整点的结果（2413→2393）。派生仅除59，2393点保留、185观测移除、每轨至少2唯一视图，导出回读必须相符。文件/源码/映射/canonical图像全部绑定并复核稳定性。

重新用prepare_mvs.py生成43图桥接后，surface-photo-input-v1可增加 `subset_certificate: {"path":"ABS_CERT","sha256":"ACTUAL_SHA256"}`；无该字段时默认selected/original相等门不变。显式模式仍独立核original44注册IDs、selected/scene精确43，证书必须属于当前桥接source_sparse及同SfM/T01。返回完整证书/输入/实现绑定供plan漂移复核。它只证明当前限制关系，不证明照片真实性、原始几何来源、mask消费、完整物体或绝对尺度。
