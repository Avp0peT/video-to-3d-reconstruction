# 本地视频输入：video-photo-v1

入口：scripts/prepare_video.py。将本地视频按真实呈现时间选为 RGB8 PNG，再调用原有 photo-io-v1；不调用分割、SfM、稠密重建或 CAD。首版为合成验证状态，尚无真实物体视频验收。

## 调用

在项目根目录使用已有项目 Python（非 SAM3 环境）：

~~~powershell
& '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe' -B .\scripts\prepare_video.py --video 'D:\capture\object.mp4' --dataset-id object_video01 --data-kind real --count 48 --output '.\outputs\video\object_video01'
~~~

FFmpeg/ffprobe 须可从 PATH 解析，或用 --ffmpeg 和 --ffprobe 指定绝对路径。二进制来源、文件 hash 和完整版本输出进入证据。当前实测 FFmpeg 8.1 Gyan Windows 构建；未验证其他构建。项目 Python 使用现有 NumPy/Pillow，不安装新依赖。

--data-kind 必须显式为 real 或 synthetic。real 图片固定写入 data/real/<dataset-id>/frames；synthetic 固定写入 outputs/delegation/T01A/fixtures/<dataset-id>/frames，符合既有 T01 契约。dataset-id 最长 64 字符；数据目录与输出目录必须新建、彼此分离。已有目录拒绝覆盖。

## 时间与图像语义

- ffprobe 的整数 best_effort_timestamp 与 time_base 共同定义时间轴。保留解码序号、原始 PTS、精确有理数时基与视频 hash；浮点秒仅供显示，不参与身份比较。
- 在首帧至末帧之间等时间选最近的真实帧，平局取较早帧；重复选择去重，最多 --count 帧（2–96）。不按名义帧率估算，不补帧。
- ffmpeg 使用 copyts、fps_mode passthrough、enc_time_base filter；输出 showinfo 的有理数时间逐帧与原时间核对。无 setpts、fps 滤镜或生产入口 seek。
- 仅接受单一被选视频流、静态宽高/像素格式、方形像素及纯直角旋转矩阵。默认取 V:0（排除封面），可用 --stream-index 明确指定绝对流号。不接受镜像、剪切、非直角、动态矩阵或不完整的 Display Matrix。
- 使用 FFmpeg 的一次静态 autorotate；保持比例、不放大，最长边 1024，输出 RGB8 PNG。原图形尺寸与已解码 PNG/showinfo 对比。T01 再读 PNG 时无 EXIF 二次旋转。
- 流色彩字段已声明而逐帧出现矛盾时拒绝；元数据缺失不冒称色彩准确。没有做相机标定或测量精度校准。
- decoded_rgb_sha256 按宽高、模式、真实像素计算，重复画面拒绝；文件 hash 单独保留。--min-sharpness 默认为 0（不启用），可显式设置拉普拉斯方差阈值；它只是图像质量代理，不是通用清晰度/几何合格标准。
- 拒绝的帧保留原因，至少两帧留下才调用 T01。输出映射明确关联 T01 input_index 与 frame_id；文件名序号是输出序号，不冒充视频原始帧号。

协议及容器限定 file + mov/matroska/avi，阻止播放列表和隐式网络输入；没有下载视频或联网取数。容器/编解码器是否可读仍由已绑定 FFmpeg 判定，不保证任意视频支持。

官方说明：[ffprobe](https://ffmpeg.org/ffprobe.html)、[时间戳和旋转选项](https://ffmpeg.org/ffmpeg.html)、[showinfo](https://ffmpeg.org/ffmpeg-filters.html#showinfo)。旋转测试夹具用 display_rotation 输入选项与 streamcopy，实际矩阵经独立 probe 核对，不能只凭 rotate 字符串认定已写入。

## 资源与结果

输入文件≤512 MiB；接受条件为≤20000解码帧、呈现时间跨度≤120秒、输入≤16777216像素。后面三个条件在有界 ffprobe 解码结束后核验，**不是解码前内存保护**。

所有工作子进程共享120秒截止；数值/编解码器线程设2。输出及数据目录合计512 MiB软阈值，单进程stdout+stderr32 MiB软阈值，轮询0.05秒，超限终止本任务持有的Popen句柄，最多5秒确认退出。无硬RAM上限、无通用后代进程树监督；这些工具的当前命令不启动另一个外部程序。输出量为采样软限，可能短暂超过阈值。大文件哈希和有限的PNG检查不属于子进程实时采样。

每个命令保存 argv、cwd、必要环境覆盖项、PID、真实退出码、起止UTC、墙钟与日志。不得输出完整环境。输入视频、工具、实现和派生PNG在结束时复核；T01本身有独立实现摘要。

RESULT.json 是入口，包含完整帧台账、选择/拒绝原因、视频hash及 photo_io.run 路径。timeline.json 保存全解码时间轴，photo_manifest.json 只包含接受帧，photo_io/ 下为既有契约产物。退出码：0 SUCCESS；2 输入/接受条件错误；4 BACKEND_ERROR；5 RESOURCE_LIMIT。若输出目录本身不能安全新建，返回2且没有虚构的结果文件。失败残留保留，不自动重试。

## 当前证据与缺口

outputs/delegation/VIDEO_INPUT/20261007T084217Z_main/HANDOFF.md：
100项一次定向测试通过。两次同一最终实现的真实FFmpeg→T01执行各6/6成功，12帧合成VFR视频含6个B帧、起始PTS 24000/12000=2秒、间隔1000/2000。第二次包含90°矩阵，96×64变为64×96。全部输入/PTS/PNG/T01身份经过独立回读。

首次旋转夹具的 metadata rotate 写法没有生成矩阵；保留其记录，只计未旋转VFR证据。新增夹具采用 display_rotation 并先核查矩阵，产品实现未变、未借用无效旋转证据。

真实视频、物体外形/视角覆盖、SAM3、SfM、稠密、网格、CAD均未在本任务执行。均匀时间采样不证明视角覆盖；需后续有许可或用户授权的静态物体环绕视频与已验收照片下游通路。
