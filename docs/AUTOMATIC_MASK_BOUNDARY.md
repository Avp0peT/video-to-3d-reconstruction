# 自动窄缝保护与原分辨率边缘复核

完整视频 union 融合完成后，现在默认追加有界边界复核。它自动选择帧、核对原生图像身份、检测窄缝、记录局部 RGB 证据、保留失败原因并生成报告。复核完成不等于精准分割；候选不会自动替代默认掩膜或发布到 MVS。

## 调用链与输出

`full_pipeline.execute`（完整视频，包括已有预测复用）与 `fusion.execute`（已有完整掩膜的再次融合）在导出和来源漂移检查结束后调用 `auto_boundary.attach_auto_review`。

- 主入口 `scripts/run_photo_demo.py propagate-masks --full-video --execute` 和 `fuse-masks` 自动接入。
- 独立 `scripts/propagate_video_masks.py` / `scripts/fuse_video_masks.py` 与 Python API 同样覆盖。
- `prepare-only` 不启动复核；失败来源、`none`、`supported-holes` 不进入 literal-union 分析。
- 新结果目录的同级 `<run_name>__boundary_review/` 保存复核产物；源目录新增 `AUTOMATIC_BOUNDARY.json`。已经冻结的源 `RESULT.json` 不重写。
- 原报告增加自动复核状态和链接。成功 union 缺复核结果、复核 FAILED/NOT_EVALUATED 时 CLI 返回非零，保留融合阶段的原状态。
- 输出或 sidecar 已存在时拒绝覆盖/重试；目前没有不经校验直接复用旧自动复核的功能。

新模块不是 `run_photo_demo.py run` 的所有照片路线通用分割器；只有存在完整且可验证的全视频 union 来源时才适用。

## 对已有融合结果执行

无需重新运行 SAM3、视频传播或三维重建；以下入口只做本轮 CPU 边界复核，输出必须是新目录：

```powershell
$repo = '<project-root>'
$py = Join-Path $repo 'outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe'
& $py -B -X utf8 (Join-Path $repo 'scripts\run_photo_demo.py') review-mask-boundaries `
  --run (Join-Path $repo 'outputs\delegation\HAND_VIDEO\20261022_union_masks02') `
  --output (Join-Path $repo 'outputs\delegation\AUTO_BOUNDARY\new_review')
```

默认自动选至多 16 帧；`--indices` 可以显式指定原 decode index，仍限制数量和身份。自动入口带外层 Windows Job 420 秒/2 GiB 看守；原生解码单独受 120 秒/2 GiB 看守。直接调用 Python 函数不是额外的外层硬看守，内部阶段和总时间检查仍存在。

## 自动选择与原生图像绑定

从整个输入时间线按四个时间区间选择声明融合扩张比例最高的帧，加入相邻帧、头尾及覆盖帧。并列按 index 确定性选取。扩张比例只表示风险，不是误分率。未选帧逐数记录 `NOT_EVALUATED`，不把 16 帧抽检称作全视频验证。

从绑定的视频按解码序号 `select`，同次解码后 split 两路：原分辨率 RGB PNG 与原配方（Lanczos、JPEG q2、4:4:4）的 canonical JPEG。后者必须逐字节 hash 等于旧 DECODE 中该帧，才能进行原生证据评估。不按秒 seek 猜帧；不一致时记关联未证实，不直接断定视频时序错误。

`showinfo` 原日志保留但本轮未独立解析/比较 PTS，不报告“PTS已验证”。源视频已有的旋转/SAR/归一化假设继续有效；坐标比例与图像身份通过不等于相机标定。

## 边缘证据与回退

沿源掩膜检测出的双岸，在原 RGB 上取局部前景/背景灰度块。两岸对比度与方向梯度都达到固定门槛且超过局部 MAD 噪声才标为 SUPPORTED；弱对比标 WEAK；过细、邻域混合、越界或不可采样标 UNKNOWN。

所有阈值见 `native_gap_evidence.RULES`。每帧最多4096组去重双岸假设；未采样者保留UNKNOWN。像素统计是同一双岸证据展开回 canonical 网格的数量，不是独立物理缝数、真值像素或分割准确率。阴影、图案、反光和未标注遮挡都可能影响证据；本方法不识别这些成因。

最近邻放大旧 mask 只转移坐标，不产生原生高分辨率分割。源掩膜遗漏的整条指缝仍可能未被列为检测目标。局部 patch 标签混合意味着本判定器无法隔离两侧证据，不证明整片物体不可建模。

窄缝候选保持 `base ⊆ candidate ⊆ union`；无法保住通路时退回 base 候选。超过既有1%改动上限的帧仍为 `REFUSED_CHANGE_CAP`，继续展示 RGB 诊断但不提升为可用修正。所有默认 mask 仍逐字节等于旧 union。背景保留和RGB支持均不能代替多视图可见性或三角面边缘/内部投影检查。

## 本次验证

报告：`outputs/delegation/AUTO_BOUNDARY/20261024_auto01/index.html`。

- 联合测试144项通过；真实公开CLI一次完成16帧自动流程，约22.4秒，外层Job提交峰值约252.2MiB。
- 原生720×1280与canonical576×1024的16帧身份全部匹配；未选1073/1089帧明确未评估。
- 46,672个目标像素对应RGB证据：支持348、弱6,844、未知39,480；大部分未知源于局部标签块混合，不可据此修改为背景或前景。
- 14帧使用保守base候选回退，7帧被1%改动上限拒绝；本轮没有覆盖默认mask、没有新SAM3/MVS/CAD执行。
- 独立读回检查136项去重内层绑定、默认PNG、集合约束、像素/样本计数全部通过。
- 真实运行后仅补强fusion报告缺证据提示与manifest读取；该报告回归单测通过。真实运行用字节与报告后改动分开记录于 `POSTRUN_DELTA.json`，未重跑真实过程。

先查书籍摘录中的对应与证据限制，再参考 [FFmpeg select/showinfo](https://ffmpeg.org/ffmpeg-filters.html)、[OpenCV梯度](https://docs.opencv.org/4.12.0/d5/d0f/tutorial_py_gradients.html)。本模块的具体RGB阈值为工程启发式，不能冒称Kozlov书籍中的定理或原算法。
