# 表面阶段的显式续跑

`integrations.demo_pipeline.resume` 提供三项 API；不会自动恢复整个任务，也不启动 SAM/SfM、安装依赖、改变分割参数或调整表面质量。

1. `write_resume_checkpoint(run, root=...)`：仅供当前统一生产者调用，在 input/SfM/segmentation 全部成功或可用 PARTIAL、确认回收并 persist 后，surface 启动前，pipeline.status 仍为 RUNNING / exit_code=None。input 必须 SUCCESS。首次 exclusive 写 `SURFACE_RESUME_SOURCE.json`，禁止覆盖。
2. `plan(run, output, python=..., tool_root=..., blender_exe=..., model_id=None, root=...)`：只读核验已有任务和生产时 checkpoint，生成 `photo-surface-continuation-v1` 计划。output 必须是项目 outputs 内的新目录，并与原任务、原素材及旧SfM结果不重叠。
3. `execute(frozen)`：重新计算并严格比较计划，再建立独立输出，调用已有 `complete_surface` 和 Windows Job 阶段监督接口。executor / surface_builder 参数仅用于无原生工作的单元测试。

生产者已接线：三阶段状态持久化之后、表面目标分支之前；以三阶段成功/可用 PARTIAL且回收已确认为条件。报告目标也可留下将来的表面起点。checkpoint 失败会写 `pipeline.json.resume_checkpoint.status=UNAVAILABLE` 和原因，不冒领续跑资格，也不阻断原有单次生成流程。

公共Python入口：

```powershell
& '<项目Python>' -B -X utf8 scripts/run_photo_demo.py plan-resume --run '<原任务>' --output '<outputs内的新目录>'
& '<项目Python>' -B -X utf8 scripts/run_photo_demo.py resume --run '<同一原任务>' --output '<同一新目录>'
& '<项目Python>' -B -X utf8 scripts/run_photo_demo.py status --run '<续跑目录>'
```

`plan-resume`只核验并打印计划，不创建输出、不启动原生任务；`resume`会再次核验后执行。工具可显式传 `--python`、`--mvs-tool-root`、`--blender-exe`；`--model-id`选定分量仍须有原生产checkpoint中的完整文件绑定。

照片/视频页面已新增“继续表面 / Blender（复用上游）”：载入原任务后，只有终态、生产时checkpoint和只读计划核验通过才启用。点击后在独立新目录调用同一Python入口；UI rerun只读，不重新调度。也可载入含`CONTINUATION.json`的目录，在来源、阶段回收、实际网格与Blender独立重开记录逐项核验后下载。40项新旧UI相关测试通过，均为AppTest/桩证据，未启动普通浏览器或原生续跑。

主线复核修正了两项实际缺口：上游checkpoint之后产生的下游ENTRY单独做终态/回收核验并冻结到新计划，不要求它倒签进上游checkpoint；首次超时5或取消130传递到最终状态，不被表面包装替换成通用错误4。复用上游有PARTIAL时，成功交付表面仍记PARTIAL/1，同时 `deliverable_ready=true`。同次主线定向及受影响回归62项通过，仍不是原生续跑证据。

checkpoint 保存稳定的三阶段记录、PLAN、原素材/输入清单、输入receipt和已有验证依赖、T01 run/frames、选定SfM完整组件文件、SfM frame map与实际图像、正式SAM PRODUCT/CONFIG/manifest及所有声明候选mask、所有已启动源stage ENTRY。它不绑定仍会变化的pipeline.json全文；续跑计划另绑定终态pipeline全文，执行前后检查不变。输入准备的既有 `verify_input` 会重新核验原图到准备图转换和T01身份；其解释器版本/实现绑定约束仍适用，不能因换环境绕过。

旧历史run没有生产时checkpoint时拒绝续跑。终态旧run不能调用write_resume_checkpoint补造生产时证明；规划时第一次哈希不等于旧生产时哈希。checkpoint是本地证据链和字节承诺，非数字签名或恶意改写防护。外部删除或篡改证据需报拒绝，不能据某个status字符串升级。

所有源原生阶段须有对应argv/cwd/pid/exit/reclaim记录；RUNNING、未确认回收、声明与ENTRY不一致均拒绝。上游状态、counts、完整frame标签、IO run/hash、原素材清单、组件文件集、逐图像身份、每个正式候选mask摘要及SAM请求对象/提示词均检查。只复用已提交manifest-selected mask；孤立或中间文件不是结果。PARTIAL不会补齐缺失mask，已有surface consumer仍负责标签像素及native子集验证。

计划最多2048个绑定文件，总计3GiB；每文件至多512MiB，控制记录8MiB、几何`descriptions.json`单独128MiB、mask64MiB、选定模型组件128个文件、输入至多96张。几何描述使用同一严格JSON解析，读取前后核验字节身份；不是把所有控制记录上限一并放宽。保留现有完整输入验证的更严格限制。每个下游stage最多一次，名称和上限固定为 native_subset 180s/2GiB、mvs_bridge 210s/2GiB、mvs_masks 240s/2GiB、surface_plan 180s/2GiB、surface_reconstruction 1230s/9GiB、blender_export 660s/8GiB；失败或回收不明后拒绝再次启动。无后台自动重试。

新输出保存 `PLAN.json`、`CONTINUATION.json`、逐stage ENTRY和已有 `geometry/surface_pipeline/` 正式产物。CONTINUATION包含原run、完整冻结source_bindings、标准surface/blender记录、geometry_steps、真实exit及deliverable_ready。此API只执行表面与Blender；研究和CAD仍记NOT_RUN。后续可用[独立CAD继续命令或页面按钮](PHOTO_CAD_FROM_CONTINUATION.md)消费正式CONTINUATION，在新目录生成CAD并验证，不改写续跑任务的历史状态。

本次证据：`docs/evidence/RUN20261013_surface_resume/`。最终13个定向测试通过，使用可失败stub且独立输入验证在fixture内替换；它们验证checkpoint创建、旧run拒绝、模型/帧/mask/原图/状态/回收漂移拒绝、只调用下游、不覆盖源run与失败停止。没有新增GPU、模型、MVS或真实Blender执行，不代表真实续跑或全部产品阶段验收。

历史首次真实尝试：`outputs/delegation/GENERAL_PIPELINE/20261013_resume_production01/FINAL.json`记录33图全部注册、SAM31成功/2无匹配；创建checkpoint时45,323,944字节的真实几何描述超过原8MiB控制记录限额，状态UNAVAILABLE，控制器未启动续跑阶段。此失败保留。128MiB分类修复后，最终49项离线测试通过，包含9MiB几何描述反例；第一轮漏改实际读取器的失败也保留。该任务未补造checkpoint，见`20261013_feature_integration/`。

当前真实验收：`outputs/delegation/GENERAL_PIPELINE/20261013_resume_cad_real02/HANDOFF.md`记录新的完整生产者正式创建checkpoint，然后只复用上游执行六阶段表面续跑，再单独继续CAD。六阶段全部SUCCESS，续跑PARTIAL/1并deliverable_ready=true（上游31/33分割），CAD SUCCESS/0且有效闭合单实体、独立回读通过。三入口各一次、合计378.98秒；源照片与各级来源文件未变，没有在resume中重算SAM/SfM。普通浏览器操作、任意阶段恢复、研究恢复、物理尺度与外形精度不在此结论内。
