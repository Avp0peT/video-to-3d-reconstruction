# object-mask-v1 — 自动目标分割侧文件契约

主线冻结日期：2026-10-07。首个生产后端为用户选定的SAM3。该契约独立于photo-io-v1：自动预测不能冒充人工真值，不修改既有T01格式、数据或历史证据。

## 输入和图像域

输入是一个成功的photo-io-v1运行、明确input_index列表、object_id与目标提示。按frames.jsonl显式取得frame_id、image_output、image_sha256、canonical尺寸，禁止靠目录排序重新推断身份。只接受对应canonical RGB8图像，图像域固定为`photo_io_canonical`，不修改或重编码输入。`source_image_sha256`须对实际读取的image_output文件计算，另存`photo_io_image_sha256`原样引用上游行的image_sha256；两者不能假定相等，frame_id仍沿用上游。将来去畸变消费者须另行登记变换，不能把本mask直接标成去畸变域。

每次执行写新的输出目录；拒绝覆盖。源运行及选定输入图片运行前后摘要应一致；输入路径需保持在指定IO运行目录内。未知/失败项用null和reason，不伪造值。

## manifest.json必需字段

- `contract_version`: `object-mask-v1`。
- `run_id`、`started_utc`、`finished_utc`（UTC ISO8601时间）、`status`：SUCCESS/PARTIAL/INPUT_ERROR/BACKEND_UNAVAILABLE/BACKEND_ERROR/AUTH_REQUIRED/RESOURCE_LIMIT。
- `source_io`: `{path, run_id, run_json_sha256, frames_jsonl_sha256}`；path为实际本机路径，其余绑定读取的字节。
- `backend`: `{name, code_commit, checkpoint_path, checkpoint_sha256, config_sha256, python_version, torch_version, cuda_runtime, device}`。name为`sam3`，模型/配置hash未取得时为null并单列`backend_unavailable_reason`，不能报SUCCESS。
- `object`: `{object_id, text_prompt, visual_prompt, selection_policy}`；object_id是任务拟重建的实例标识，不能仅凭同一类别文字宣称跨视已验证。未用的提示为null。首版selection_policy为`unique_candidate_or_explicit`。
- `frames`: 每项含`input_index, frame_id, source_image, source_image_sha256, photo_io_image_sha256, image_domain, width, height, status, candidates, selected_candidate_id, selected_mask, selection, errors, not_run_reason`；保持所请求的每项输入，失败项也保留。正常执行项not_run_reason为null。
- `counts`: requested/success/no_match/needs_selection/error/not_run，非负整数且与frames可重算一致。success对应OK；error为三类逐帧错误状态之和；requested等于其余五项之和及frames长度。
- `quality_status`: 默认`NOT_EVALUATED`；`cross_view_status`: 默认`NOT_EVALUATED`；不以推理成功提升几何/物理精度结论。

逐帧status为OK/NO_MATCH/NEEDS_SELECTION/INPUT_ERROR/BACKEND_ERROR/RESOURCE_LIMIT/NOT_RUN。候选为空→NO_MATCH；多个候选且没有显式选择→NEEDS_SELECTION，不取最大面积作为隐含规则；唯一非空候选可选取，selection记录`unique_candidate`。手工指定记录`explicit_candidate`及来源。实际未调用的帧用NOT_RUN及非空not_run_reason，不冒充无匹配或失败推理。全部请求帧OK且所有文件/hash/绑定检查通过才SUCCESS；正常完成但有NO_MATCH/NEEDS_SELECTION为PARTIAL。实际后端错误或资源超限不得降成NO_MATCH。全局前置失败使用相应run错误状态；途中错误保留已完成帧并使用实际停止原因对应的run错误状态，未启动余项为NOT_RUN。

## mask与候选

每候选记录`candidate_id, mask_path, mask_sha256, foreground_pixels, score, score_semantics`。score保留有限原始数值及其含义，未提供为null，不把logit假装概率。只导出实际后端返回的候选，不制造候选。

mask为单帧L模式8位PNG、尺寸与canonical图像完全一致、Orientation为1或不存在，仅0/255，0背景、255前景。mask_path为相对本次输出的安全路径，建议`masks/000017_c00.png`等短名；frame_id完整保存在JSON。选中mask不得为空。保留候选概率到二值转换的实际规则和阈值，不以Pillow静默转换规避校验。

`selected_mask`为所选候选的路径与hash，未选为null；`selection`含method/source与`mask_source=automatic_sam3`。未来人工修改另存文件及原mask hash，来源为`automatic_sam3_human_edited`，不覆盖原预测。原分数不因人工修改被重写。

SAM3内部resize的预测须由其接口映射回输入图像尺寸，并以独立尺寸/像素域检查验证；硬标签外部变换用最近邻或等效离散映射。叠图可另存，仅供目视，不能作为权威mask。

## 接口与失败

建议CLI：`segment_sam3.py --io-run ABS --indices 17,34,59 --object-id skull-01 --text-prompt skull --checkpoint ABS --output NEW`。配置/代码路径可增设明确参数。参数数组落盘；不使用shell字符串，不记录完整环境或认证凭证。

进程退出码：SUCCESS=0、PARTIAL=1、INPUT_ERROR=2、BACKEND_UNAVAILABLE/AUTH_REQUIRED=3、BACKEND_ERROR=4、RESOURCE_LIMIT=5。输出目录不可写或已存在可退出2且写stderr；不得声称报告已保存。

真实运行报告须记录模型实际加载、逐帧实际调用、CPU/RAM/VRAM与耗时的可获得值及未测原因。mock单测、GPU探针、真实单图/三图推理、人工外形检查是不同证据层级。无需跑SFM或证明CAD完成才能验收本接口。


## Positive-box request extension (2026-10-11)

### Temporal union adoption (2026-10-22)

`video_temporal_union` is a derived mask source, not a new SAM3 inference.
Its backend name is `video_temporal_union` with `model_executed=false`.
All consumers must validate `video-temporal-union-adoption-v1` derivation:
original T01 rows, sampled decode index/PTS/time base, identity canonical grid,
original status ledger, source producer plan/manifest/fusion hashes, copied mask
hashes, and nonempty binary PNG readback. The historical SAM3 failures remain in
`historical_status` and the original manifest; derived OK means availability only.
Tracking JPEG and canonical PNG pixels are not claimed to be equal.
Existing automatic SAM3 sources retain their original validation rules.

`RunRequest.visual_prompt: dict | None` accepts only `positive_boxes` in
`photo_io_canonical`, with `annotation_source=ai_assisted`. Box-only requires
`text_prompt=None`. Each requested frame has exactly one frame record containing
`input_index`, `frame_id`, `image_sha256`, integer `width`/`height`, and four plain
integer pixels `box_xyxy=[x0,y0,x1,y1]` (bool excluded), with
`0 <= x0 < x1 <= width` and `0 <= y0 < y1 <= height`. No negative boxes, points,
multiple boxes per frame, mixed text or implicit text fallback are accepted.

`image_sha256` binds actual canonical `image_output` bytes; it is not the T01
original-input `image_sha256` reference. The parent and worker rebind frame ID,
canonical dimensions, actual hash and worker image path before model loading.
The backend checks current file hash/dimensions again before forward.

The manifest preserves the full validated `object.visual_prompt`, writes
`object.text_prompt=null` and submitted visual option `positive_boxes`; box
`effective_options` adds `prompt_mode=positive_boxes`, `visual_prompt_sha256`
(canonical sorted-key compact UTF-8 JSON SHA256), and
`autocast_scope=set_image+add_geometric_prompt`. Text-only options and digest
semantics are unchanged. The actual sequence is `set_image` then one positive
`add_geometric_prompt`, in the same inference-mode/CUDA BF16 scope, with no
`set_text_prompt`. Upstream may internally encode the dummy word `visual`; that
is a geometric prompt dependency, not a submitted text-only inference.

AI-assisted boxes are prompts, not human truth masks. All output masks remain
SAM3 predictions; unique/explicit selection, NO_MATCH/NEEDS_SELECTION, resource
limits, ownership and frame commit policies remain unchanged. Box correctness
and real-device execution require separate evidence; offline mocks do not prove
them. Existing historical runs/NO_MATCH evidence must be kept and consumption
selection must bind source manifests explicitly.
