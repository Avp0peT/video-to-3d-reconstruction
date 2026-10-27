# 从表面续跑结果显式继续 CAD

正式 `photo-surface-continuation-v1` 来源目录只有 `CONTINUATION.json` 与自己的 `PLAN.json`，不需要补造 `pipeline.json` 或 `CAD_SOURCE.json`。

调用既有API即可：

```python
from integrations.demo_pipeline.cad import plan, execute, verify_cad_output

frozen = plan(continuation_directory, new_cad_directory,
              python=existing_project_python, freecad_python=existing_freecad_python)
# plan只读；显式执行才进入既有受监督CAD步骤。
result = execute(frozen)
# 下载前只读重核，expected_source_run填写continuation_directory。
verified = verify_cad_output(new_cad_directory, continuation_directory)
```

`plan` 自动识别正式CONTINUATION，返回 `source_kind=surface_continuation`；输出必须为项目 outputs 内的新独立目录。pipeline 与 CONTINUATION 同时存在视为来源歧义并拒绝。旧 pipeline 来源、原本同任务 CAD_SOURCE geometry_checkpoint 模式保持兼容。

公共CLI已接入相同API（先计划只读，随后用户明确执行）：

```powershell
& '<项目Python>' -B -X utf8 scripts/run_photo_demo.py plan-cad --run '<CONTINUATION目录>' --output '<outputs内新CAD目录>'
& '<项目Python>' -B -X utf8 scripts/run_photo_demo.py continue-cad --run '<同一CONTINUATION目录>' --output '<同一新CAD目录>'
```

可显式传入`--python`与`--freecad-python`。照片/视频页面载入已验证CONTINUATION后提供“从续跑表面生成 CAD 实体”按钮；仅点击执行，rerun只读校验，下载前核对实际发送字节。失败保留原表面与Blender，不自动重试。

来源须为SUCCESS/0或PARTIAL/1且deliverable_ready=true；续跑PLAN必须READY并绑定本目录、项目根、原生产run及完全一致的source_bindings。全部来源文件重核path/bytes/sha256，缺绑定、重复路径、漂移或超过2048文件/3GiB拒绝。源生产者须有生产时正式 `SURFACE_RESUME_SOURCE.json`，绑定稳定三阶段投影、完整依赖及原pipeline/PLAN；三上游和六下游阶段都检查实际监督身份、argv/cwd/pid/exit与回收。六下游阶段再核对flat ENTRY与logs ENTRY一致，脚本/动作及关键输入输出路径属于冻结来源和该续跑。

CONTINUATION中的表面、Blender和geometry_steps须与 `geometry/surface_pipeline/RESULT.json` 一致。之后复用原CAD严格检查：唯一表面生产者、最终mesh和raw角色、Blender同源表面创建与独立重开、payload/场景记录、文件摘要及 `validate_source` 的mesh生产关联。只读state/PLAN声明不能单独作为证据。

执行在native准备前、准备后、结束时重复核验来源；`verify_cad_output` 在下载核验中再次检查。改source_kind试图将CONTINUATION降级成旧pipeline模式也拒绝。原来源只读，失败保留新CAD输出和非零状态，不重试、不回填历史checkpoint、不自动补洞。

配方保持原有quadric、输入最多400000面、目标2000面、相对采样偏差1%、体积变化2%；准备60秒、CAD内部120秒、外层135秒、2GiB及既有输出额度。开放或无效网格仍由原门拒绝，不能因续跑来源放宽闭合实体检查。

CAD生成和独立回读成功也只表示未知尺度的分面实体文件交付，不代表参数化特征、照片外形、真实尺寸、研究完成或全部阶段恢复。`outputs/delegation/GENERAL_PIPELINE/RUN20261013_cad_from_resume/`的最终67项定向测试包含22项新增来源/漂移/失败/身份检查及原CAD兼容回归；该组正例为synthetic fixture/stub。主线另有23项CLI/UI回归通过。

真实证据单独位于`outputs/delegation/GENERAL_PIPELINE/20261013_resume_cad_real02/HANDOFF.md`：新33图生产者提交正式checkpoint后，独立表面续跑生成Blender，再调用上述continue-cad一次，exit0、有效闭合2000面单实体，FCStd重开和STEP重导入通过。上游两张NO_MATCH不消失，表面任务仍PARTIAL。`20261013_resume_cad_round2/UI_READBACK.json`通过页面同一验证函数核对四份实际下载字节、原生调用0；这不是普通浏览器交互验收。
