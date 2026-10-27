# 连续视频掩膜传播

现有 SAM3 图像分割不保留帧间记忆。本入口复用本地 SAM3 视频 tracker，以两个已存在的自动掩膜作种子，在原视频连续帧上分别正向、反向传播。种子不是人工真值，双向一致也不是轮廓准确度证明。

## 一条命令

在项目目录运行已有轻量 Python：

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/run_photo_demo.py propagate-masks --pipeline '<原任务的pipeline目录>' --output '<新的输出目录>' --max-frames 48 --execute
```

省略 `--execute` 只验证输入并生成计划，不运行模型。也可使用独立入口 `scripts/propagate_video_masks.py`，参数相同。输出目录必须不存在；不会覆盖、重试或续写旧实验。

脚本自动完成：

1. 核对原视频、photo_io、采样索引、原视频解码索引、时间戳、图像和原掩膜哈希。
2. 在相邻的两个原始 OK 掩膜之间，选择含 NO_MATCH 且解码帧跨度最短的区间；最多48帧，显式参数最多96帧。没有合适区间时报告 NOT_APPLICABLE。
3. 按原解码索引抽出连续帧，缩放到原掩膜坐标域，再转 JPEG 以适配上游 tracker；记录编码差异，不冒称与原 canonical PNG 字节相同。
4. 本地权重 mmap 严格加载一次；禁 HF 下载；正向与反向分别创建状态；视频和状态存CPU，推理CUDA BF16。直接使用 tracker，不启动分布式/NCCL包装。
5. Windows Job 限制14 GiB/600秒，独立采样实际worker提交量（12 GiB）与整卡显存（7 GiB），首次触发即锁存并回收本任务进程树。软采样不保证捕获任意短峰值。
6. 每张掩膜保存、像素回读、哈希核对后才提交。原掩膜、正向、反向结果生成并排叠图与动画。

## 裁定与输出

双向 IoU ≥ 0.8、面积比 ≤ 1.5、非空且尺寸一致的缺失帧仅标 `PROPAGATED_CANDIDATE`。原有 OK 掩膜保留；不取并集、不膨胀、不自动补洞，也不把候选改写成原始 SAM3 识别成功。两个方向一致仍可能一致地漏掉手指或包含背景。

- `PLAN.json`：输入和区间绑定。
- `START_BINDINGS.json`：执行源码和输入绑定。
- `DECODE.json`：连续帧派生与种子像素差异。
- `worker/events.jsonl`：真实加载、传播和提交记录。
- `worker/forward/`、`worker/reverse/`：分别传播的PNG与结果。
- `resources.jsonl`、`logs/propagate/ENTRY.json`：独立资源通道与原生回收。
- `REVIEW.json`、`index.html`、`motion.gif`：逐帧核查结果。
- `RESULT.json`：执行状态、输入/源码漂移、回收状态。

## 多区间自动覆盖

在同一命令后增加 `--coverage`，脚本自动选择多个原始 OK 掩膜之间的缺口，包含 NO_MATCH 与 NEEDS_SELECTION；优先覆盖缺失采样数量最多的区间，再按时间排序执行。默认且最大为6段、每段96帧、合计384帧。共享端点在处理量中重复计数，报告同时给出唯一解码帧数量。原先48帧短片模式保持不变。

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/run_photo_demo.py propagate-masks --pipeline '<原任务的pipeline目录>' --output '<新的批处理目录>' --coverage --execute
```

所有区间共用一次严格 mmap 模型加载；每个区间和传播方向独立创建跟踪状态，种子只取原始 OK 掩膜，不使用前一区间的传播结果作新种子。模型全批次仍共享600秒、12GiB软提交量、7GiB软整卡显存和14GiB Job限制；首次失败停止后续区间，保留已提交记录，不自动重试。

`COVERAGE.json` 与首页逐项列出原采样的来源和去向：原掩膜保留、双向一致候选、传播分歧、因缺少两端种子/区间过长/预算不足而未运行。`clips/clip_NNN/` 各含独立计划、双向传播、对比图与动画；首页展示每段一致性最差的帧。未处理段仍记未覆盖，不把采样缺失数与新增视频帧数混为一谈。

当前入口是自动连续帧验证与复核步骤，结果尚不自动写入 SfM/MVS 掩膜清单。无种子的全视频区间、参考掩膜质量筛选、传播冲突处理和重建消费需独立验证后再放行。不会因为能生成传播掩膜就宣称手套几何完整。

## 完整掩膜视频

`--full-video` 覆盖源时间轴的每个解码帧，并可用 `--reuse-run` 复用此前已完整提交、输入绑定与像素/hash核验通过的多区间结果。二者不是重跑原任务；所有新产物进入新的输出目录。

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/run_photo_demo.py propagate-masks --pipeline '<pipeline目录>' --output '<新的全片目录>' --full-video --reuse-run '<已有多区间输出>' --execute
```

自动流程：完整解码 → 原始OK锚点分段 → 每段正反传播 → 超长段拆成最多96帧、重叠16帧的窗口 → 在新窗口处读取前一窗口已经提交的种子 → 合并逐帧来源 → 导出并回读视频。一个新worker共享一次严格mmap加载，任务最多64个、累计新方向帧2600，模型墙钟1200秒；12GiB软提交量、7GiB软整卡显存、14GiB Job保持不变，输出目录上限3GiB。失败不自动重试。

从同一原始种子延伸的窗口不算新的独立证据。两端有不同原始种子才可标双向候选；头尾缺少对侧种子时标单向候选。融合前的选择保存在`SOURCE_ASSEMBLY.json`：双向分歧仍选已有预测并保留`DISAGREEMENT_CANDIDATE`。缺少真实提交掩膜不能声称覆盖完成。原始OK掩膜文件保持原字节。

完整模式现在默认 `--mask-fusion union`：在同一解码帧、同一坐标域对原始选择与全部已提交的正反跟踪掩膜逐像素取并集，再导出。它会保留任一输入中的误分，不代表多方一致；有变化的帧标`FUSED_UNION_CANDIDATE`并用紫色边框，新增区域及来源见`FUSION.json`和对比图。不存在原始图像识别掩膜的帧只合并跟踪预测，原`NO_MATCH`原样保留。用`--mask-fusion none`保留之前的只选择行为。

- `masks_lossless.mkv`：FFV1灰度无损视频；每帧解码像素与PNG逐一哈希核对。
- `masks_preview.mp4`：可播放黑白预览；有损格式，不用作二值标签数据。
- `overlay.mp4`：原图域绿色叠加，逐帧状态和风险颜色可见。
- `final_masks/`：逐帧二值PNG；`MANIFEST.json` 记录来源、状态、链深度和哈希。
- `ASSEMBLY.json`：原识别状态、正反证据、选择规则及面积/质心变化提示。
- `EXPORT.json`：三个视频的实际帧数、尺寸、PTS误差和无损像素回读证据。

输出时间戳须与源时间轴核验；能够证明为28fps舍入时间轴时使用28fps，否则用明确标注的毫秒量化VFR。全片有掩膜不表示所有轮廓准确，单向、分歧、空预测、占位分别统计；当前仍不自动进入几何重建。

参考：本地冻结SAM3源码 `examples/sam3_for_sam2_video_task_example.ipynb`、`sam3/model/sam3_tracking_predictor.py`；上游 https://github.com/facebookresearch/sam3 。

## 已完成视频的融合与碎孔对比

不重跑模型，在新目录融合并导出完整视频：

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/run_photo_demo.py fuse-masks --run '<全片推理目录或派生交付目录>' --output '<新的融合目录>' --mode union
```

也可调用 `scripts/fuse_video_masks.py`。所有输入PNG的身份、坐标、二值标签和哈希必须一致；并集不会移动掩膜、膨胀或绘制不存在的输入像素。`SOURCE_ASSEMBLY.json`保留原选择，`FUSION.json`逐帧记录输入、独立根来源、增加面积、孔洞数量变化及误分风险；PNG回读和无损视频完整解码像素校验沿用原入口。

另提供 `--mode supported-holes`（完整模式对应`--mask-fusion supported-holes`）作保守对照：背景8连通且不接触外部的小孔，面积至多1024像素且不超过该帧前景0.5%，不同原始根预测的整图IoU≥0.8且孔内支持≥90%才补其明确前景像素；总增量超过前景5%则整帧撤回修孔。阈值是工程规则，不是书中定理或精度保证。

书籍既有复核讨论点集对应与仿射不变量，不提供像素孔洞填补规则。本地SAM3 `sam3_tracker_utils.fill_holes_in_mask_scores`也提供按连通域面积处理小孔的方法（[上游代码](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_tracker_utils.py)）；本次并集是独立的派生后处理，并未改上游或宣称运行了该函数。

## 并集掩膜自动建模与逐步诊断

完整传播/融合入口自身不启动几何求解。需要建模时使用以下入口，自动接入经过身份校验的并集掩膜、重新计算前景SfM，并在相机域检查通过后继续MVS与Blender：

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/run_photo_demo.py reconstruct-masks --run '<FULL_TIMELINE_FUSED_EXPORTED的union目录>' --output '<新的建模目录>'
```

当前复用原始T01的全部采样照片（2–96张），由decode索引、PTS、时间基和固定像素坐标网格连接全片掩膜；不会将1089张掩膜说成1089视角都参与了SfM。原PNG与跟踪用JPEG不声称像素相同。来源显式标为`video_temporal_union`，保留旧SAM3的`NO_MATCH`/`NEEDS_SELECTION`，不重跑SAM3，不冒充新的图像推理。

执行顺序为：派生输入验证 → 前景特征/匹配/SfM → 数值相机域检查 → 几何诊断 → 注册视角子集证书 → 去畸变/桥接 → 完整掩膜消费 → 稠密 → 网格 → Blender创建及独立重开 → 中文逐阶段报告。每原生阶段至多执行一次，有固定时间/内存和进程树回收；不会自动放宽匹配或填补缺失几何。报告包含全部采样视角、原图与掩膜叠图、特征中心、原始及几何验证对应、注册状态、原生深度/置信图、稠密/网格查看器和Blender链接。

若已产生可用SfM且相机数值域通过，但后续接入修复后需继续，可显式在新目录续跑表面阶段；旧失败证据保留，分割和SfM不重跑。子集/桥接/掩膜转换在新表面任务内重新核验。

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/run_photo_demo.py continue-mask-surface --run '<先前建模目录>' --output '<新的表面续跑目录>'
```

仅更新展示时用 `report-mask-reconstruction --run '<建模或续跑目录>'`，不启动求解器。生成且可打开的网格仍标`SURFACE_GENERATED_NOT_ACCEPTED`；物理尺度未知，完整度/形状精度未验收。MVS掩膜文件的完整覆盖和正确坐标映射，也不等同于已正面验证原生MVS实际使用了掩膜。
