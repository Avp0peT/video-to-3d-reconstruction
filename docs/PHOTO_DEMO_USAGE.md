# 照片 / 视频 → 开放三维表面、Blender 与演示报告

2026-10-22：公共 Python/PowerShell 入口默认启用有界自动恢复（`object-recovery-v1`）。正常完成的 SAM 结果仍有 `NO_MATCH` 或 `NEEDS_SELECTION` 时，只对这些帧追加一次描述提示词尝试（最多96帧），仅接收唯一候选，原来成功的掩膜不替换。可用 `--recovery-prompt` / `-RecoveryPrompt` 提供同一目标的备用描述；未提供时使用 `the <原提示词>`。保留原始 manifest 和 `PRIMARY_PRODUCT_OUTCOME.json`，报告读取带来源账本的派生结果；没有隐式人工选框或填补未识别帧。

表面计算前自动检查所选相机的全部二维观测（含未三角化点）及9×9图像网格是否有有限、可往返的原生逆投影。数值域失效时，最多独立计算一次 `cad-object-pinhole-v1`（SIMPLE_PINHOLE，去掉径向畸变自由度）；只有域检查通过且所选分量注册视图数不减少，才用于后续表面。零畸变是对照模型假设，不是已知标定。显式 `--model-id` 与192帧档不自动换相机。源漂移、资源停止或回收未确认会阻止继续；不同分量不拼接，洞仍保留。记录在 `camera_recovery/RESULT.json`。`--no-auto-recover` / `-NoAutoRecover` 可关闭这两条分支。

例：` .\scripts\run_photo_demo.ps1 -Source '<视频路径>' -Prompt 'glove' -RecoveryPrompt 'black and red glove' -Frames 64 `。恢复仍使用原有单阶段资源限制：额外 SAM 最多950秒/外层14GiB，受约束 SfM 最多720秒/2GiB；不是重复尝试直到成功。聚合 SAM 结果可在同一次流程继续到表面，但旧 `resume` 契约尚不支持这类多来源聚合，会明确拒绝续跑；原始结果继续保留。测试通过、相机域有效、生成Blender均不代替外形验收。

2026-10-17 起，默认后期复核自动包含 **时序点跟踪 → 候选掩膜关联 → 局部运动与边界检查 → 点级支持 → 已存相机几何诊断**。使用原来的建模命令即可，不需要手填帧号、历史目录或封印摘要；每次读取当前任务自己的数据，输出 `point_review/index.html`，并链接到总报告。无证据、无相机、未识别或超预算会记录具体状态，不自动覆盖掩膜、不补洞。它检查本次实际抽取的全部帧（最多192），不会把默认24帧抽样悄悄改为全视频。既有 `--no-post-review` 会同时禁用这条自动分支。详见[自动点证据复核](AUTOMATIC_POINT_REVIEW.md)。

2026-10-14 起，新建单物体任务改为 **输入准备 → SAM3 目标分割 → 前景特征提取与 SfM → 几何检查与后续表面**。只使用生产清单中已选中的非空 `OK` 掩膜；`NO_MATCH`、多候选 `NEEDS_SELECTION`、失败和未请求行保留排除原因，不自动改为整图选点。原 RGB 图像和坐标不裁剪、不涂黑，COLMAP 使用独立二值 mask；提取后逐个回查数据库特征中心。少于两张可用前景图即停止 SfM，报告仍生成。历史任务保持其原始整图流程语义，不追认已使用掩膜。细节见 [前景选点](OBJECT_FIRST_SFM.md)。

2026-10-13功能补充：[显式表面续跑](PHOTO_DEMO_RESUME.md)和[从续跑表面继续CAD](PHOTO_CAD_FROM_CONTINUATION.md)已接入CLI和页面，并用新33图生产者、独立六阶段表面续跑及CAD续接真实验证：Blender重开、FCStd重开及STEP重导入通过。上游两张NO_MATCH仍保留为PARTIAL。页面的四份实际下载字节已只读核验，普通浏览器交互未因此验收。[部署配置](PHOTO3D_DEPLOYMENT.md)与[只读迁移规划](PHOTO3D_RELOCATION.md)已有功能，新根实际迁移仍未执行；[精确研究页面](EXACT_RESEARCH_UI.md)保持独立研究范围。功能总清单以[最新优先级](PROJECT_COMPLETION_CHECKLIST.md)为准。

2026-10-12新增[统一视频入口实跑](../outputs/delegation/GENERAL_PIPELINE/20261012_unified_video03/HANDOFF.md)：公共命令从视频重新抽取36帧，最终生成可独立重开的Blender开放表面，11297顶点/22472面。仅选中一个分量的14视图，其他22视图未用于表面，总体PARTIAL；[全选定视图检查](../outputs/delegation/GENERAL_PIPELINE/20261012_video03_allview_review/HANDOFF.md)仍发现伸入背景的多余网格。文件可以查看，不代表外形已验收。视频结果的只读UI载入及下载字节已核对，实际浏览器交互、CAD与物理尺度未因此通过。

截至 2026-10-12：统一 `--target cad` 已经由公共 PowerShell 命令完成一次原始照片到 Blender、FCStd、STEP 的真实整链：[cad05 终态](../outputs/delegation/GENERAL_PIPELINE/20261012_unified_cad05/FINAL.json)。33 张原图全部注册为一个 SfM 分量，31 张分割成功、2 张 NO_MATCH，表面、Blender、研究侧支和 CAD 均成功；总体 PARTIAL/exit 1，`target_achieved=true`。CAD 为有效闭合的 2000 面单实体，FCStd 重开与 STEP 重导入通过；这不证明照片外形、真实尺度或任意新输入的成功率。页面载入同一任务与四个下载已通过 [AppTest 实际回读](../outputs/delegation/GENERAL_PIPELINE/20261012_cad05_ui_readback/RESULT.json)，真实浏览器交互仍未验收。默认目标仍为 `blender`，`report` 仅分割与稀疏报告。原生 mask 有固定三个视图的 depth 阶段实效证据，完整 dense 对照仍为 INCONCLUSIVE。

同次 `QUALITY.json` 独立标记 `object_shape_status=NOT_EVALUATED`、`object_goal_achieved=false`。文件交付与物体外形分别显示；允许孔洞不会自动证明物体覆盖充分，也不会放宽 CAD 闭合实体检查。当前模型为未知尺度的分面实体，尚无参数化设计特征或实物精度验收。

已完成CAD的页面中新增“按实测参考长度生成 CAD”：选择准备网格端点、填写自己的测量记录，可加另一段独立校验；新文件单独生成并通过回读后下载。[操作说明](PHOTO_MEASUREMENT_USAGE.md)。软件通路已用明确标注的人为指定长度完成原生数值控制，尚无用户实物测量验收；浏览器交互仍未验证。

成功表面可用独立命令生成全部选定视图的网格/SAM轮廓叠图与一致性报告：[全视图检查](SURFACE_SILHOUETTE_REVIEW_USAGE.md)。cad05的31个有效视图已全部检查；这不修改原质量状态，不将同源掩膜当作独立真值。

本入口串联输入准备、SAM3、前景 COLMAP/PyCOLMAP、OpenMVS表面、Blender导出与独立回读，以及所选CAD目标和离线报告。每个原生阶段最多执行一次，没有自动重试、安装或下载。开放表面允许孔洞；CAD仅接收通过既有闭合实体门的网格，不自动填洞。真实尺度未知，几何格式通过不等于照片外形或物理精度通过。

Streamlit页面 `11_Photo_to_3D.py` 提供照片目录/视频路径、目标描述、抽帧数量及可选SfM模型ID。“同时生成 CAD 实体（FCStd / STEP）”默认未勾选；勾选后运行统一 `cad` 目标，先查看计划和依赖。每次执行建立 `outputs/photo_ui/<UTC唯一ID>/pipeline/` 新任务，整个CLI受Job监督，现有外层限1800秒、12GiB、3GiB目录采样阈值。CAD失败时原Blender成果保留，可依据 `geometry_deliverable_ready` 及几何身份核验继续下载；整体CAD目标的 `deliverable_ready` 为false。FCStd/STEP下载需当前来源、独立回读、监督记录和实际发送字节摘要都通过。

2026-10-11 已完成两次新的真实照片整链：`outputs/photo_demo/20261011_general_e2e02`（颅骨，66 输入、42 有效视图）与 `outputs/photo_demo/20261011_dice_e2e01`（骰子，33 输入、31 有效视图）。两次均 PARTIAL、表面及 Blender 成功、独立重开通过；未补齐未识别视图。骰子预览可辨认，颅骨仍有粗糙和跨缺口伪影。视频局部结果已有，但通用视频入口外形质量、物理尺度和精度仍待验收。

页面的“载入已有任务”可读取上述目录，不启动推理/重建。UI 产生的任务检查外层监督；CLI 产生的任务逐项检查各原生阶段记录，并明确没有全程 UI 外层记录。仅阶段记录、产物身份及 Blender 独立回读都通过才显示下载。现有颅骨 CLI 结果已通过 AppTest 只读载入，浏览器交互仍未验收。

真实UI→CAD续接已通过AppTest驱动页面执行现有CLI、核验CAD文件及下载身份，rerun未再次启动native：[独立证据](../outputs/delegation/GENERAL_PIPELINE/20261011_cad_ui_fixed/FINAL.json)。此前准备schema比较导致的UI失败保留在旧任务，不回写为成功。该续接使用已完成骰子Blender任务，不是新增统一 `--target cad` 照片整链实跑，也不代表真实浏览器操作全部验收。

## 一条命令

在项目根目录运行：

首次使用或迁移目录时，可先运行[只读环境预检](PHOTO3D_ENVIRONMENT.md)，明确缺少的工具、环境路径及旧目录绑定。它不会安装、启动模型或查询GPU，文件齐全也不代表运行兼容性通过。

```powershell
.\scripts\run_photo_demo.ps1 -Source '<path-to-photo-folder>' -Prompt 'beverage can'
```

默认自动创建 `outputs/photo_demo/<UTC时间_唯一ID>/`。先检查计划，不运行模型：

```powershell
.\scripts\run_photo_demo.ps1 -Source '<path-to-photo-folder>' -Prompt 'beverage can' -PlanOnly
```

统一CAD目标（仅使用已有FreeCAD）：

```powershell
.\scripts\run_photo_demo.ps1 -Source '<照片目录或视频>' -Prompt '<目标物体>' -Target cad -FreecadPython '<FreeCAD的python.exe>'
# Python等效参数：run_photo_demo.py run --source <素材> --prompt <物体> --target cad --freecad-python <已有解释器>
```

成功几何检查后，系统写不可覆盖的 `CAD_SOURCE.json`：`GEOMETRY_READY_FOR_CAD` 仅表示几何前置达成，CAD尚未完成；固定快照绑定同任务表面/Blender身份，避免把后续变化的 `pipeline.json` 全文当作不可变来源。续接固定quadric、目标2000面、1%双向顶点/面心采样偏差、2%体积变化，准备60秒/2GiB、CAD内部120秒/2GiB。不补洞、不换参重试；细节见 [CAD用法](PHOTO_CAD_USAGE.md)。

研究默认开启，选定原生模型的point_id前六点进入既有精确索引代码。默认同时输出固定仿射正例和单点扰动负例，经独立oracle核对；预算不足或对照未建立如实报告。`--no-research-code6` 可关闭侧支，研究失败不替代或伪造几何结果。固定六点验证不代表完整物体或照片未知对应恢复：[真实控制证据](../outputs/delegation/GENERAL_PIPELINE/20261011_research_controls_01/HANDOFF.md)。

视频入口相同，默认最多均匀选取 24 帧，保留原视频时间戳：

```powershell
.\scripts\run_photo_demo.ps1 -Source 'D:\capture\object.mp4' -Prompt 'toy car' -Frames 24
```

未能从 PATH 找到 FFmpeg 时，传 `-Ffmpeg '<本地ffmpeg.exe>' -Ffprobe '<本地ffprobe.exe>'`。不修改系统执行策略；若终端不允许运行 PS1，直接使用已有 Python：

```powershell
python -B .\scripts\run_photo_demo.py run --source '<path-to-photo-folder>' --prompt 'beverage can'
```

`--output` / `-Output` 可指定项目 `outputs` 内的新目录。已有输出一律拒绝覆盖。照片目录非递归，支持 2–96 张 JPEG/PNG；视频沿用现有 512 MiB / 120 秒等约束。准备图最长边 1024，原素材只读。共享相机假设要求拍摄时保持同一相机/焦距；没有 EXIF 时不能证明该假设。

## 输出和状态

- `report/index.html`：离线展示照片、分割叠图和掩膜；如有可用点云，嵌入可旋转查看器。没有点云时显示真实失败状态。
- `pipeline.json`：全过程状态、阶段退出码、日志位置与错误。
- `PLAN.json`：本轮输入摘要、环境路径、配置与预算。
- `input/RESULT.json`：照片准备或视频抽帧与 T01 绑定。
- `segmentation/PRODUCT_OUTCOME.json`、其 `run_dir/manifest.json`：真实分割结果。只认已提交的选中 mask，磁盘孤立文件不计成功。
- `pointcloud/`：通过既有模型往返检查后的 PLY 和查看器；多个分量分别展示，不自动拼接，不补面。
- `geometry/surface_pipeline/`：默认 Blender 目标的表面、视图子集证据和 Blender 导出 / 回读结果；报告只链接真实存在且 SHA-256 匹配的成功产物。选中和缺失视图明确登记，不自动补齐缺失掩膜或补洞。
- `CAD_SOURCE.json`、`cad/`：仅统一CAD目标；固定几何快照、准备、原生交付及独立回读证据。FCStd/STEP是任意尺度的分面实体，不是恢复参数化设计。
- `research/`：默认固定六点表示、仿射正负对照与各自真实状态，不将主表示成功冒充对照通过。
- `logs/<stage>/`：实际 argv、退出码、资源监督和 stdout/stderr。

SfM 原始结果仍写入契约指定的 `outputs/delegation/T02A/PIPELINE_<run_id>/`，其绝对路径记录在 `pipeline.json`。准备图还会写入新的 `data/real/demo_<run_id>/`。这些目录保留用于复核，不自动清理。

退出码：`0` 所选目标各阶段成功；`1` 部分结果（例如分割成功但 NO_MODEL）；`2` 输入/准备错误；`3` 分割环境或前置条件不满足；`4` 运行、证据、回收或报告错误；`5` 对外保留的资源拒绝；`130` 用户取消。默认 Blender 目标另以 `pipeline.json` 的 `deliverable_ready` 核实表面与 Blender 文件证据；页面再次检查下载文件。报告可生成并不等于重建成功；即使都成功，也不代表 CAD 或精度完成。

统一CAD成功且上游分割PARTIAL时仍可 `deliverable_ready=true`、整体PARTIAL/1；CAD失败则目标未达成且非零，原几何成果另由 `geometry_deliverable_ready` 表达。固定三视图mask对照观察到label0背景深度被排除，前景双方保留；两份完整dense执行均触及原600MiB输出阈值而停止，完整densepair仍INCONCLUSIVE，不能推广到所有视图、最终网格或通用外形质量：[depth实效范围](../outputs/delegation/GENERAL_PIPELINE/20261011_native_mask_control/DEPTH_STAGE_BEHAVIOR_EVIDENCE.json)。

## 查看进度和重新生成报告

以下操作不会重新推理或重建：

```powershell
& '<项目Python>' -B .\scripts\run_photo_demo.py status --run '<本轮输出目录>'
& '<项目Python>' -B .\scripts\run_photo_demo.py report --run '<本轮输出目录>'
```

失败后保留结果供检查。上游三阶段已完成并有正式 `SURFACE_RESUME_SOURCE.json` 时，可用 `plan-resume` / `resume` 在新目录继续表面与Blender，复用已有输入、SfM和SAM3，详见 [续跑用法](PHOTO_DEMO_RESUME.md)。没有检查点的历史任务不能补造生产证据。系统不自动重试；再次 `run` 仍会建立新任务并重新处理全部阶段。

运行中按 Ctrl+C 会记录取消请求，在监督器已持有进程身份的安全位置停止并回收本任务进程；确认不了退出就如实记录，不启动后续阶段。强制关闭终端不等同于这条正常取消路径。

## 现有环境和预算

可先用 [部署配置入口](PHOTO3D_DEPLOYMENT.md) 把本机已有路径及身份保存为独立配置，再传给统一Python入口：

```powershell
& '<项目Python>' -B .\scripts\run_photo_demo.py plan --source '<素材>' --prompt '<物体>' --deployment-config '<配置.json>' --deployment-config-sha256 '<创建时返回的SHA256>'
# 将 plan 改为 run 才执行。实际输入种类必须匹配配置，显式参数与配置冲突时拒绝。
```

读取配置会重新核对依赖字节，不继承历史运行PASS。旧封印绑定到其他根目录的配置仍被阻止；本功能尚不代表可以直接搬到新机器执行。独立精确研究入口见 [有限研究用法](KOZLOV_EXACT_RESEARCH_USAGE.md)。

仅使用现有本地环境：项目 Python 3.13 环境、SAM3 隔离 Python 3.12 环境、`C:\sam3\sam3.pt`，以及当前 `outputs/delegation/DEMO_RUNTIME/20261011_text_v2` 运行包中绑定的 worker 模块。新的执行不调用旧 `run_once.py` 或 `execute_pilot.ps1`，也不复用旧 pilot 的执行次数。该固定运行包及其绑定的来源文件当前仍是依赖，不能当作临时缓存删除；跨机器迁移需要重新建立绑定，不是随意改路径就能运行。

运行时验证本次输入以及实际模型/代码来源。父进程不加载模型，SAM3 子进程重型加载一次服务本批。12 GiB worker 提交量、7 GiB 整卡显存软阈值；外层 Job 14 GiB 覆盖整批进程。线程 2；输入/SfM/SAM3 外层限时分别 180/720/950 秒，内部预算仍生效。超时或回收不明时停止继续启动原生工作；保留可复核结果。不自动关其他程序、清缓存、改驱动或提高阈值。

演示入口使用 `integrations/demo_pipeline/supervision.py`：它保留现有 Job 监督器机制并增加协作取消查询。共享 CAD 监督器及其哈希绑定没有修改。

## 历史真实素材的结论（2026-10-08，仅旧报告目标）

本地 4 张饮料罐照片已完成分割；修正竖拍相机尺寸校验后，真实 SfM 结果仍为 `NO_MODEL`。原图、掩膜与处理证据可以展示，尚没有该物体的点云。俯视图的模糊罐身可能漏分，分割数量不是人工质量合格数。

建议补拍连续重叠视角，保持物体固定、相机焦距与照明稳定，降低强反光。根据 [COLMAP 官方采集建议](https://colmap.github.io/tutorial.html#structure-from-motion)，相邻画面应有充分重叠，并从不同相机位置拍摄。无保证成功的固定照片张数。

## 历史脚本验证（2026-10-08，不代表当前整链验收）

统一 PowerShell 入口已实跑一次：`outputs/photo_demo/20261008_can_scripted/`。整理成功、SfM `NO_MODEL`、SAM3 4/4 已提交掩膜、报告生成成功，整体退出码 1（部分成果）。输入、SfM、分割各执行一次，无自动重试；三阶段回收均确认。之后补齐的取消和失败处理只做了定向验证，没有再运行模型。

流程与运行适配的 43 项检查、监督器 6 项检查、最终入口接线 3 项检查均通过；后两组各含一个有界的真实标准库 Python 子进程。报告另检查了错误可见性、链接和离线资源依赖。使用历史原生 SfM 结果只读验证了两个独立分量、4176 点的导出与报告内嵌，没有把它们当作饮料罐成果。视频分支完成参数与抽帧元数据接线检查，本轮没有新的实拍视频端到端证据。浏览器预览受本地文件 URL 策略限制，未冒称完成按钮或截图视觉验收。

入口与证据摘要见本轮 `AUTOMATION_VALIDATION.json`。运行状态成功仅说明相应工程步骤完成；真实尺度、分割轮廓质量、跨视一致性及 CAD 质量仍需独立检查。


## 自动离线复核

照片/视频建模默认追加对应与外观复核，无需手动研究步骤；不会修改模型、点或孔洞，也不代表建模验收。`--no-post-review`可关闭；已有任务可用独立单命令生成新复核报告。参数、证据边界和资源说明见[自动建模后复核](AUTOMATIC_POST_REVIEW.md)。

## SAM3 区域内选点再建模（实验）

统一入口支持 `--point-selection rcover-v1`。执行顺序是 SAM3 已提交选中掩膜 → 掩膜内原生 SIFT → 覆盖选点 → 描述子匹配/几何验证 → SfM → 原生导出。`NO_MATCH`、待选目标和未提交中间文件不进入建模，不回退到整幅图。SAM 掩膜可能误含手或支撑，掩膜身份通过不代表外形质量通过。

```powershell
& '<项目Python>' -B -X utf8 .\scripts\run_photo_demo.py plan --source '<视频或照片目录>' --prompt '<目标名称>' --point-selection rcover-v1
# 查看计划后，将 plan 换成 run 执行；沿用部署配置、环境和其他参数。
```

已有 T01 与已提交 SAM3 掩膜可直接复用，免于重新推理：

```powershell
& '<固定重建Python>' -B -X utf8 .\scripts\reconstruct_object.py --io-run '<T01目录>' --object-mask-manifest '<manifest.json>' --point-selection rcover-v1 --output '<outputs/delegation/T02A/全新目录>'
```

`rcover-v1` 使用书中覆盖思想启发的贪心 r-cover（长边1024时半径16像素，按尺寸缩放），不是最小覆盖求解，也不是 Kozlov 平行投影恢复算法的完整实现。它从真实特征行选取代表点，保持原坐标、完整描述子、原行到新行映射和覆盖见证；在匹配前同步更新 keypoints/descriptors 并回读。选点独立进行，可能降低跨视图对应的数量或稳定性，不能由覆盖均匀直接推断几何精度提高。默认 `native` 保留原有特征集合，显式选择实验策略才生效。

原生目录保存 `point_selection.json` 及只读核对用的选点前数据库备份。统一入口在证据 VERIFIED 后自动生成 `selected_points/index.html`，即使无模型也保留诊断页；主页证据链接可打开它。红/青线显示 SAM 内/外边界，灰点显示原特征，黄点显示送入匹配的选点。它核对照片、掩膜、数据库行与描述子身份，并不认证表面质量。选点中的“同序号”不代表同一个物理点；对应仍由描述子、几何验证与跨视图轨迹建立。遮挡轮廓随视角改变，不能直接按轮廓顺序强行连点。多分量不得拼成假完整物体；未观测部分保持缺失。


## 整段视频逐帧模式（whole-video-192-v1）

`--all-frames` 显式处理完整视频的全部解码帧，保留原始解码索引和 PTS，不插值；实际帧数必须为 2–192。超过 192 帧明确拒绝，不自动抽样、不截断，也不自动分段。该模式忽略 `--frames` 的抽样数量；默认不加此开关时，仍按时间均匀抽取 2–96 帧。

以下 PowerShell 示例在仓库根目录执行。先生成计划，确认后使用完全相同参数执行；`$output` 必须是尚不存在的新目录。

```powershell
$python = '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe'
$source = '.\outputs\delegation\PPP_VIDEO\20261014_object_first01\normalized\ppp_square_pixels_assumed.mp4'
$output = '.\outputs\photo_demo\ppp_all_frames_new'
& $python -B -X utf8 .\scripts\run_photo_demo.py plan --source $source --prompt 'toilet paper roll' --object-id 'ppp-whole-video' --all-frames --point-selection rcover-v1 --target blender --output $output
& $python -B -X utf8 .\scripts\run_photo_demo.py run --source $source --prompt 'toilet paper roll' --object-id 'ppp-whole-video' --all-frames --point-selection rcover-v1 --target blender --output $output
```

需要显式部署配置或工具路径时，在两条命令中使用相同的已有环境、`--deployment-config`、`--deployment-config-sha256`、`--ffmpeg` 和 `--ffprobe` 参数；该开关不安装工具或下载权重。

192 帧视频的 SAM3 按两个顺序批次执行，每批最多 96 帧；较短视频按需使用一个或两个批次。合并记录保留每帧身份和状态，`NO_MATCH` 与 `NEEDS_SELECTION` 不被自动改成可用目标，也不按最大面积或最高分消除歧义。建模仅消费已提交选中物体掩膜，`rcover-v1` 在掩膜内选点，不回退整图特征；允许保留缺失部分和孔洞，不人工补点或填洞。

全帧覆盖只说明时间上使用全部解码画面，不保证完整几何。物体分割、遮挡、形变、模糊、重复纹理和跨视角对应仍可能使模型局部、分量分离或无法生成。错误物体掩膜不会因逐帧模式自动变正确；流程完成、分割状态和几何质量分别报告。
