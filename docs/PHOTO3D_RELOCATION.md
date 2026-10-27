# 显式迁移规划器（只规划、不执行）

`scripts/plan_photo3d_relocation.py` 实现真实只读规划。它核对当前固定父 BUNDLE 与完整包文件、实际 active 运行依赖、AUTHORITY/BASELINE 根和 checkout 元数据、PROFILE worker/approved_rebindings 关联、环境实际静态包清单、worker 的 ROOT/BOUND_CHECKPOINT/BOUND_SHA256 常量以及源 checkpoint stat。拒绝封印错误、源码漂移、路径逃逸、重复目标映射、根重叠、目标角色目录碰撞、缺少必要键和不符合当前布局的路径。

```powershell
python -B -X utf8 scripts/plan_photo3d_relocation.py --source-root 'C:\source-project' --source-package 'C:\source-project\outputs\delegation\DEMO_RUNTIME\20261011_text_v2' --expected-source-bundle-sha256 6704d9a2331bb6c82500bd0fa58d276e21188cc407c84296539b60c76d36f843 --target-root 'D:\new-project' --target-paths 'C:\target-paths.json' --output 'C:\source-project\outputs\new-relocation-plan\PLAN.json'
```

target-paths JSON 必须包含 project_python、sam3_python、sam3_base_python、sam3_source、sam3_package、checkpoint、checkpoint_sha256、git、input_kind、target。路径显式绝对；checkpoint 摘要必须保持已批准模型身份。视频另须 ffmpeg/ffprobe；blender 或 cad 须 mvs_tool_root/blender；cad 另须 freecad_python。SAM 环境、基础 CPython、源码和运行包按目标根下现行相对布局固定。目标目录可以不存在，规划器不创建目标。输出必须是源 outputs 下一个尚不存在的新子目录，拒绝覆盖。

API：`plan_relocation(source_root, source_package, expected_source_bundle_sha256, target_root, target_paths, *, max_metadata_bytes=8388608)` 返回计划；`write_plan(plan, output)` 仅新建计划输出。CLI 返回 0 仅表示计划文件已写入；其状态始终 `PLANNED_REVIEW_REQUIRED`、run_argv=null、execution_authorized=false、target_verification=NOT_PERFORMED。拒绝时 stdout 返回 PLANNING_REJECTED，退出 2。

copy_map 逐条给出源/目标、用途、实际源摘要/字节或未验证标识，包括封印、运行依赖、当前产品源码、环境解释器及静态元数据。源 checkpoint 仅 stat，记录 declared_sha256 和 bytes_rehashed=false；不读取权重字节。完整环境/CPython/源码 checkout/工具分发仍列为 directory_requirements：元数据和单 exe 的哈希不等于完整环境/DLL 可搬运。目标工具没有源工具参数，因此不会虚构它们的源复制路径。复制、安装、包生成、候选 authority/BUNDLE 和审批均未执行。

新 profile 仍需固定父封印和精确 AST/JSON 受限变换核验，保持模型和源码身份，重新绑定目标实际 stat 与运行来源。规划列出 mmap worker 两个常量的 previous/proposed 和 review gates，但不改文件，不追加 digest 白名单授权。旧 PROFILE/current evidence 不混用，历史 PASS 不复制；当前 runtime loader 不能直接消费迁移规划。

历史证据：`RUN20261013_relocation_planner` 两批 pytest 各 10 项通过，总约 4.3 秒，涵盖 4 MiB 元数据边界、真实 fixture 封印与源码审核、哈希漂移、路径冲突、缺键、逃逸、计划独占写入、无网络/进程和权重不重哈。该轮唯一真实只读规划保留失败：旧 authority 使用 Windows 反斜杠，初版将其拒绝为 noncanonical；随后按原 runtime 的规则先统一反斜杠再做重复/逃逸检查。目标 `D:\PHOTO3D_PLANNER_TARGET_NOT_CREATED_20261013` 在该次失败后仍不存在，失败输出保留。

主线最终验证已完成：[10 项测试通过](../outputs/delegation/GENERAL_PIPELINE/20261013_resume_cad_round2/relocation.txt)及 [XML 测试记录](../outputs/delegation/GENERAL_PIPELINE/20261013_resume_cad_round2/relocation.xml)覆盖修正后的终态。[实际只读规划执行记录](../outputs/delegation/GENERAL_PIPELINE/20261013_resume_cad_round2/READONLY_RELOCATION_EXECUTION.json)状态为 `PLANNED_REVIEW_REQUIRED`，生成 281 条文件映射，复制为 0、native_calls=0、target_created=false、checkpoint_bytes_rehashed=false、execution_authorized=false、current_runtime_evidence=[]；约 0.59 秒。[完整规划](../outputs/delegation/GENERAL_PIPELINE/20261013_resume_cad_round2/relocation_plan/PLAN.json)摘要为 `f597400dfadcdcfe82ab09b71a27aca02dcf4e8fb325ca06c8628af81ecbbc67`。规划功能已经有真实只读证据；实际文件迁移、新目标运行授权和目标机器整链验收仍未执行。
