# mask-review-v1

2026-10-07。用途：把已经存在的 object-mask-v1 结果与原照片绑定，形成可供人检查的图片、统计和可选的稀疏观测诊断。工具不推理、不选择候选、不修改 mask、不批准分割质量，也不调用重建或 CAD。

## 输入与身份

`scripts/review_object_masks.py --mask-manifest ABS [重复] --output NEW [--sfm-run ABS --model-id ID]`。

各 manifest 必须为严格有限 UTF-8 JSON，同一 photo-io-v1 SUCCESS run、同一 object_id，input_index 不重复。依次核对 source_io 的 run/frames 文件 hash、run_id、显式 input_index/frame_id、canonical 域与实际照片路径。photo_io_image_sha256 是 T01 的原输入身份；source_image_sha256 是实际 canonical PNG 文件 hash，两者不能强迫相等。不同 input_index 的合法原图别名不被假设为不同 native view。

照片只接受 RGB8 单帧 PNG；mask 只接受 L8 单帧 PNG、原尺寸、0/255 标签、无透明和非恒等方向。候选文件身份、前景数、selected_mask 与候选 ID/hash/path 逐一核对。非 OK 不得带选择，NEEDS_SELECTION 不自动选最大分数。选择方法、来源、提示词和原错误保留；上游 backend/evidence 标签仅为声明，不是对模型运行的独立认证。

合法 INPUT_ERROR/NOT_RUN 可能缺少图像元数据；只在无候选、无选择且已知字段不矛盾时保留为 binding_status=UNAVAILABLE。不会为其猜图像或补造身份，面板明确显示没有可绑定图像。完整身份的失败帧仍可显示原照片。上游整个 IO 无法绑定或输入自相矛盾时返回 INPUT_ERROR，不冒充可复核结果。

输入均只读，读取时检查文件状态并记录实际字节/hash，完成前复算稳定性。输出目录须不存在、父目录存在，不能位于任一 manifest、T01 或 SfM 输入根内。解析相对文件名必须保留在对应输入根，拒绝路径越界。

## 形态与展示

报告每个候选的前景像素数/比例、半开 bbox、触边情况、8 邻接前景组件和最大组件、4 邻接背景封闭区数。此处“孔”只是二维标签性质，不推断物理通孔。连通分析超过 200000 行程节点时统计写 null 和 NOT_COMPUTED_RUN_LIMIT，不写零或通过。

面板按显式 input_index 与候选 ordinal 命名，展示原图、mask 和着色轮廓；缩放仅用于展示，不输出新的权威 mask。选中与未选中明确标注；联系表对未选择帧可展示首候选但仍标 UNSELECTED。HTML 对来源文本转义，显示最终状态、原错误、输入稳定性、提示/选择来源和诊断摘要；详细逐对数据在 JSON。

## 稀疏跨视诊断

仅在同时给出 sfm-run/model-id 时执行。读取该 run 的 models[*].descriptions_path 中 binary 对象以及 frames_map.jsonl，明确标 SOURCE_SIDECAR_NOT_NATIVE_RELOADED；不声称当前 native binary 被重载或相机几何被重新验收。

要求 upstream_io_run_id、COLMAP 原生像素约定、显式注册/相机/图像映射、尺寸与 staged 照片实际 hash 一致。图像索引与原生 ID 不按文件名或 index+1 猜测。点与 feature 的关联须双向一致；同一点有同 view 的不同观测时整点排除并记录，其他损坏关联拒绝。同 native view 的请求别名仅在已校验 mask 像素完全相同时合并；像素 hash 单列为 selected_mask_pixels_sha256，不冒充 PNG 文件 hash。

采样已记录 feature 的 floor(x),floor(y)，仅 [0,W)×[0,H) 有效；无半像素平移或夹紧。无 mask、越界、未注册、未绑定均显式记未知，不算背景。逐点与逐 view-pair 统计分开；报告 FF/FB/BB、未知、支持数和覆盖率。前景条件不一致率为 FB/(FF+FB)，不以大量 BB 稀释；分母为零写 null。即使一致率高，也不能证明物体身份、轮廓完整、遮挡处理或真实精度。诊断永远不提升 cross_view_status。

## 状态与边界

| 状态 | 退出码 | 意义 |
|---|---|---|
| REPORT_READY | 0 | 全部帧有合法选择，来源均 SUCCESS，复核材料形成；不是质量 PASS |
| PARTIAL_REVIEW | 1 | 部分帧可检查，失败/缺选帧保留 |
| NO_REVIEWABLE_MASKS | 1 | 无已选择 mask，仍可形成失败/候选复核材料 |
| INPUT_ERROR | 2 | 身份、输入、配置或完成前稳定性失败 |
| INTERNAL_ERROR | 4 | 渲染、写盘、输出上限或内部故障 |

quality_status/cross_view_status 始终 NOT_EVALUATED，human_review_status=PENDING。无法创建输出目录时 CLI 拒绝；写盘失败只能尽力留下失败报告，不能保证故障磁盘上仍有文件。部分产物不得脱离最终 report.json/HTML 当作验收。

上限：128 请求帧/manifest，单图 4194304 像素，累计源图与 mask 各 64Mi 像素，256 面板，单 JSON 8MiB、输入文件 64MiB、累计唯一输入 256MiB；展示文件上限 60MiB，另预留 4MiB 元数据。tracks 独立限制 256 views、100000 points、1000000 features/observations、2000000 次共享点对工作。它们是输入/输出和工作规模约束，不是硬 RAM/CPU 保证；受控批次另加外层时限。

不提供 mask 人工编辑/批准工作流，不替代 MVS mask consumer、真实跨视审核、重建、CAD、尺度或物理误差验收。
