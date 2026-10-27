# 照片与人工 mask 输入输出契约

版本：`photo-io-v1`；状态：**仅低决策照片 IO 已冻结**；日期：2026-10-02。

本文件是 T01A 的唯一 schema 权威。任务文本只限定工作范围，不另建字段定义。
本契约是工程约定，不证明论文数学定义或重建算法成立。T01A 不需要等待完整几何接口、二维理论验证或 T00 测试诊断。

## 1. 放行范围与待定边界

- 输入：单帧 JPEG/PNG 照片与显式配对的人工二值 PNG mask；实现使用 Pillow、NumPy。
- 输出：统一方向的 `uint8 RGB` 照片、可选二值 mask、逐输入记录与运行证据。
- 不包含视频、抽帧、自动分割、SfM/MVS、去畸变、相机估计、点云、骨架或 GUI。
- `camera`、`distortion`、`pose`、`geometry_scale` 均固定写 JSON `null`，意为尚未定义/估计；不得用零值、单位矩阵或“无畸变”替代未知值。
- 相机模型、内参、位姿方向、几何坐标系/尺度、遮挡、重投影指标、几何验收门槛，以及 Kozlov 二维/三维数学接口：**PENDING**。像素坐标约定只服务本阶段 IO，不冻结后续三维约定；不宣称单位为米。
- 已有公共容差仍由 `src/kozlov_lab/config.py` 管理；不新增 YAML、数学容差或图像阈值。
- 记录实际 Python/Pillow/NumPy 版本，不在本契约绑定新的版本下限。环境存在 Pillow 不等于 `pyproject.toml` 已声明完整依赖；依赖变更交主线补丁，T01A 不安装或修改公共依赖。

## 2. 路径、输入索引与身份

在项目根目录运行 CLI。`dataset_id` 区分大小写，只接受 `[A-Za-z0-9][A-Za-z0-9_-]*`。所有 schema 路径为 UTF-8 字符串。

- 真实输入根目录固定为 `data/real/<dataset_id>/`；原图与原始 mask 只读。
- 合成输入位于 `outputs/delegation/T01A/fixtures/<dataset_id>/`，明确标为 `synthetic`，不得冒充真实数据。
- 输出根目录必须在项目 `outputs/` 内，且不能位于当前输入根目录内。T01A 的所有产物限于 `outputs/delegation/T01A/`。
- `dataset_root` 相对项目根目录；每行 `image`、非 null 的 `mask` 相对 `dataset_root`。解析后验证真实路径仍在所声明根目录内，拒绝越界符号链接/重解析点。
- 相对路径规范化：反斜杠转 `/`，折叠重复分隔符与 `.`，Unicode 转 NFC，保留大小写。拒绝空路径、任何 `..`、绝对路径、盘符、UNC、NUL；不按机器路径或随意大小写折叠生成身份。
- 只读取显式索引；不扫描目录补图，不猜 mask 文件名、不按排序自动配对。目录中的未列出文件不产生记录。

输入 manifest 是 UTF-8 JSON，所有下列顶层字段必填；`frames` 非空，每行必须含 `image` 和 `mask`，其中 `mask` 可以为 null：

```json
{
  "contract_version": "photo-io-v1",
  "dataset_id": "demo01",
  "data_kind": "real",
  "dataset_root": "data/real/demo01",
  "frames": [
    {"image": "images/a.jpg", "mask": "masks/a.png"},
    {"image": "images/b.png", "mask": null}
  ]
}
```

`data_kind` 仅为 `synthetic` 或 `real`，必须与 CLI 参数一致。真实运行还需用户提供或明确选定的有许可数据；本任务包不授权下载/选用真实数据。

每个 `frames` 输入行均保留一条输出，`input_index` 为原 manifest 中从 0 开始的行序号；不可去重删除。字段错误也产生该行错误记录；无法解析 JSON 或顶层结构时只产生运行级错误记录。

原文件 SHA-256 对未经 EXIF 处理、未经解码的完整文件字节计算，使用小写 64 位十六进制。设 `p` 为规范化图像相对路径、`h` 为原图 SHA-256，则：

```text
identity_bytes = UTF8(JSON([dataset_id, p, h], ensure_ascii=false, separators=(",", ":")))
frame_id = "f_" + SHA256(identity_bytes).hexdigest()
```

身份不含绝对根目录、列表行号、mtime 或 mask。整体迁移项目/数据根而保留 dataset_id、相对路径和文件字节时，frame_id 不变；更改 dataset_id、图像相对路径或文件字节时产生新 frame_id，不自动推断旧新对应关系。仅 mask 变化不改变 frame_id，但必须改变缓存指纹。未取得合法路径或原图哈希时 `frame_id=null`。

输出按 `(image_sort_key, mask_sort_key, input_index)` 排序，不依赖目录枚举/地区设置。合法路径使用规范字符串；没有合法路径但原字段为字符串时使用原字符串；缺失/null 用空串；其他非法类型使用该 JSON 值的规范序列化（键排序、ensure_ascii=false、无多余空格）作为排序键。字符串按 Unicode 码点顺序比较。

仅对已可靠取得的非 null `image_sha256` 标记内容重复，以排序后首条的 `input_index` 作为 `duplicate_of_input_index`；首条为 null。`image_sha256=null` 的失败记录永远不据此归为重复。每行仍独立处理其 mask，重复行可有相同 frame_id。产物文件以唯一的 `input_index` 命名，避免重复身份互相覆盖。

## 3. 像素、颜色和 EXIF

图像仅接受可解码的单帧 JPEG/PNG，源每通道位深为 8，解码模式仅为 `RGB` 或 `L`。`L` 通过三通道复制无损扩展为 RGB；不做色彩空间校正，不能据此称为标定的 sRGB。拒绝多帧/动画、透明度（含 tRNS）、调色板、RGBA/LA/CMYK、16 位、浮点及其他模式；不静默丢 alpha、量化高位深或强制 `astype(uint8)`。内部数组必须为 `uint8`，API 输入不满足范围/类型时显式报错。JPEG 解码本身不被描述为无损重建原拍摄像素。

读取原 EXIF Orientation；缺失时记录 `exif_orientation_original=null`、`orientation_applied=1`；存在时仅接受整数 1–8，否则报错。规范输出不保留可再次触发旋转的 Orientation。v1 不 resize、不 crop、不插值。

raw 与 canonical 坐标都以左上像素中心 `(0,0)` 为原点，x 向右、y 向下。原宽高为 `W,H`，以下变换同时用于图像和 raw 坐标 mask；输出还必须保存对应整数 3×3 齐次矩阵 `raw_to_canonical`，满足 `[x',y',1]^T = T [x,y,1]^T`：

| Orientation | `(x', y')` | canonical 宽高 |
|---|---|---|
| 1 | `(x, y)` | `(W,H)` |
| 2 | `(W-1-x, y)` | `(W,H)` |
| 3 | `(W-1-x, H-1-y)` | `(W,H)` |
| 4 | `(x, H-1-y)` | `(W,H)` |
| 5 | `(y, x)` | `(H,W)` |
| 6 | `(H-1-y, x)` | `(H,W)` |
| 7 | `(H-1-y, W-1-x)` | `(H,W)` |
| 8 | `(y, W-1-x)` | `(H,W)` |

## 4. 人工 mask

`mask=null` 表示未知/未提供，输出仍为 null；绝不能替换为全前景。非 null 路径不存在、无法解码或不合规范时为错误，不退化为“缺 mask”。

人工 mask 为单帧 PNG，模式 `L`、位深 8、无透明度，raw 宽高必须严格等于原图 raw 宽高。mask 文件自身 Orientation 只能缺失或为 1；其他值显式拒绝，避免二次旋转。人工绘制坐标必须对应原图 raw 像素；v1 不接收已按 EXIF 转正的 mask。

- 所有像素只可属于 `{0,1}` 或 `{0,255}`；同一 mask 同时出现 1 和 255 时拒绝，其他值拒绝，不做阈值化。
- 0 是背景，非零是用户给定的物体前景；边界由人工 mask 自身定义。
- `mask_encoding` 为 `zero_one`、`zero_255`；全零同时兼容两种编码，记录为 `all_zero`。
- 读取后提供 `bool` 数组；保存为规范方向 `L` PNG，像素严格为 0 或 255。应用与图像完全相同的上表变换。
- 空前景正常输出并记录 `EMPTY_MASK` warning；全前景合法，不推断其质量。

## 5. CLI 与最小输出 schema

入口（使用既有项目环境；源布局按需设置进程级 `PYTHONPATH=src`，无需安装）：

```text
python -m kozlov_lab.io --manifest <manifest.json> --output outputs/delegation/T01A/runs --data-kind synthetic|real
```

`--output` 是产物基目录。每次建立唯一 `<output>/<run_id>/`；`run_id` 可用 UTC 时间与随机 UUID 组合。成功或失败都不得覆盖历史目录。运行保存 `run.json` 和 `frames.jsonl`；成功行产物为 `images/<input_index:06d>.png` 和可选 `masks/<input_index:06d>.png`，路径相对本次 run 目录。

`run.json` 必填字段：

```json
{
  "contract_version": "photo-io-v1",
  "run_id": "<unique>",
  "dataset_id": "demo01",
  "data_kind": "real",
  "manifest_sha256": "<sha256>",
  "status": "SUCCESS",
  "input_count": 2,
  "ok_count": 2,
  "error_count": 0,
  "warning_count": 0,
  "runtime": {"python": "<actual>", "pillow": "<actual>", "numpy": "<actual>"},
  "implementation_sha256": "<sha256>",
  "errors": []
}
```

`status` 仅为 `SUCCESS`、`PARTIAL_FAILURE`、`INPUT_ERROR`。顶层无法解析时 dataset_id 可为 null，计数为 0，errors 给出原因；无法读取输入 manifest 时 manifest_sha256 为 null。warning_count 为逐行 warnings 数量总和。

`frames.jsonl` 每行包含以下全部字段，图示为 Orientation 1 的成功行：

```json
{
  "input_index": 0,
  "input_image": "images/a.jpg",
  "input_mask": "masks/a.png",
  "frame_id": "f_<sha256>",
  "image_path": "images/a.jpg",
  "mask_path": "masks/a.png",
  "image_sha256": "<sha256>",
  "mask_sha256": "<sha256>",
  "duplicate_of_input_index": null,
  "status": "OK",
  "raw_width": 640,
  "raw_height": 480,
  "canonical_width": 640,
  "canonical_height": 480,
  "exif_orientation_original": 1,
  "orientation_applied": 1,
  "raw_to_canonical": [[1,0,0],[0,1,0],[0,0,1]],
  "image_dtype": "uint8",
  "image_mode": "RGB",
  "mask_encoding": "zero_255",
  "mask_foreground_pixels": 120,
  "image_output": "images/000000.png",
  "mask_output": "masks/000000.png",
  "cache_fingerprint": "<sha256>",
  "cache_hit": false,
  "camera": null,
  "distortion": null,
  "pose": null,
  "geometry_scale": null,
  "warnings": [],
  "errors": []
}
```

mask 未提供时，其路径/哈希/编码/前景数/产物均为 null。`input_image/input_mask` 保留原行值供审计，字段缺失时为 null。行 `status` 仅为 `OK` 或 `ERROR`；错误行保留已可靠取得的元数据，未知字段为 null，`image_output/mask_output` 必须为 null，不能指向旧结果；cache_hit=false。warnings/errors 为 `{"code":"...","message":"..."}` 数组，message 不嵌入整份日志或多余本机路径。

最少错误代码：`INVALID_MANIFEST`、`INVALID_PATH`、`MISSING_FILE`、`DECODE_ERROR`、`UNSUPPORTED_IMAGE`、`INVALID_ORIENTATION`、`INVALID_MASK`、`MASK_SIZE_MISMATCH`、`INPUT_CHANGED`、`OUTPUT_ERROR`。空 mask 使用 warning `EMPTY_MASK`。意外异常记录为 `INTERNAL_ERROR`，不得改写成成功。

CLI：所有输入行成功且证据/产物完整才退出 0；存在行错误退出 1；顶层参数/manifest 错误退出 2。输出目录本身不可写时，向 stderr 输出错误并非零退出，不假称证据已保存。此处仅定义新 CLI；不得篡改 pytest 或外层执行器的实际退出码。

## 6. 指纹、确定性与失败隔离

`implementation_sha256` 对本次 `io/` 与 `segmentation/` 下 Python 源文件按相对路径排序后的 `[路径, 文件SHA256]` 列表进行规范 JSON 哈希（UTF-8、无空格、ensure_ascii=false）。

`cache_fingerprint` 对下列对象的规范 JSON 哈希（键排序、UTF-8、无空格、ensure_ascii=false）：contract_version、dataset_id、data_kind、规范 image_path/mask_path、原 image_sha256/mask_sha256（缺 mask 为 null）、implementation_sha256、runtime。不得只用路径、mtime、frame_id 或图像哈希；mask 字节或版本变化必须失效。读取时确认用于哈希的文件字节与解码输入相同，检测到运行中变化即报错。

v1 允许尚不启用缓存复用并始终记录 `cache_hit=false`。若实现复用，只能使用完全匹配指纹、此前状态 OK 且规范产物仍完整可校验的记录；当前输入仍须存在并重新核验哈希。缓存失败、部分产物、当前验证失败都不能回退旧成功结果。输出图像及逐行业务字段须可重复；run_id、时间与日志位置不要求相同。

## 7. T01A 验收与证据门槛

仅运行 `tests/test_io_frames.py`、`tests/test_manual_masks.py`，不重跑全套。用可检查的非对称小图覆盖八种 EXIF 方向及 raw→canonical 坐标、mask 同步、尺寸交换；还要覆盖 RGB/L、非法模式/位深/值域、损坏/缺失文件、路径越界、显式配对、缺 mask、两种 mask 编码/混值拒绝/空 mask、每行保留重复标记、身份迁移规则、确定排序，以及更改 mask/图像/contract 后指纹变化与失败不复用。

一次小型合成 manifest 经 CLI 生成完整产物；人工/自动检查 manifest 行数与逐输入记录数一致，归一化图像/mask 像素和变换符合已知 fixture。合成 fixture 要明确标注 TEST_DATA。

每次测试 run 使用独立证据目录，保存原始 stdout、stderr、JUnit XML、命令、起止时间、实际 Python 子进程退出码；外层工具状态单独记录。XML、输出和退出码必须来自同一 run_id，并核对收集/执行清单、failure/error/skipped 与会话是否正常结束。未保存的退出码为 null/UNKNOWN；禁止拿其他运行补齐，禁止修改原始日志。

任务状态限 `BLOCKED / IMPLEMENTED / TESTED_SYNTHETIC / TESTED_REAL`。`TESTED_SYNTHETIC` 必须同时满足定向 pytest 实际退出 0、必要用例执行且无失败/错误/跳过、CLI 验收退出 0 和证据完整；非零即如实报告未通过，不凭 XML 的零 failure 宣称通过。`TESTED_REAL` 还必须有获准真实数据和本阶段真实 IO 验收；不等于重建或 Kozlov 理论验证。目前没有获准真实照片，因此真实验收仍待数据。
