# T02A 原生稀疏重建接口

版本：`sfm-native-v1`。主线裁决日期：2026-10-03。状态：**仅 T01 到 pycolmap 调用、原生模型保存和运行证据已冻结**。本文件是 T02A 的接口权威；照片 IO 继续遵守 [photo-io-v1](contracts.md)，不修改其字段与已接收实现。

用户已采用 COLMAP/pycolmap 与独立 Kozlov 模块的路线。本轮选定 **pycolmap 4.2.1 的 CPU 稀疏重建**，不同时维护 COLMAP CLI 后端，不引入 OpenMVS、Open3D 或神经重建。真实重建质量门槛、物体点筛选、去畸变后导入旧 Camera 及 Kozlov 理论仍单独裁决。

## 1 环境和已知边界

2026-10-03 核查：既有 Python 为 3.13.14；PATH 未找到 colmap，既有环境 site-packages 未找到 pycolmap。该发现仅描述检查范围，不代表整台机器不存在其他安装。

PyPI 已提供 `pycolmap-4.2.1-cp313-cp313-win_amd64.whl`，SHA-256 为 `2aa586acd21154ed92c1d674d5da9b1b05e1dd9e2156627caf0ae42278423f02`；包元数据要求 Python >=3.10。文件存在不等于本机安装或运行已经通过。执行者在 T02A 专属新 venv 内安装二进制包并验证 import，锁定实际环境；不得修改既有环境、系统驱动或编译 COLMAP。

目前没有已授权的真实输入集。T01 的 11 张标记图只证明 IO，不能拿其重建失败判定 SfM 不可用，也不能伪造其成功重建。T01 安装验证目录已出现交付候选，本轮不重复派发或据此自动宣布安装验收通过。

## 2 输入和身份

CLI：`python -m kozlov_lab.reconstruction --io-run <T01 run目录> --output outputs/delegation/T02A/runs`。从项目根运行，output 位于项目 outputs/delegation/T02A/ 内、不能落入输入目录；每次创建独立 run_id。

输入为一个完整的 T01 `run.json`、`frames.jsonl` 及其 canonical PNG 产物。本轮只接收 `photo-io-v1` 且上游 status=SUCCESS 的运行；含 ERROR 行或计数不符的输入整次拒绝，不自行跳过。data_kind 从上游继承。执行者当前只获准生成/运行 synthetic fixture，不下载或选取真实照片。

- 每行校验 input_index 唯一、frame_id 非空、image_output 位于上游 run 内且为现有 canonical RGB8 PNG，宽高与记录一致。禁止路径越界或越界重解析链接。
- 读取 canonical 产物字节，计算独立 SHA-256；T01 的 image_sha256 是原始输入字节哈希，不能当作 canonical PNG 哈希。mask 非 null 时核实路径、尺寸及字节哈希，原文件只读。
- 只将 canonical 图像复制进本次 `images/`，不再应用 EXIF、不移动像素、不缩放裁剪。不回读原 JPEG 猜焦距，T01 PNG 不保证保留相机元数据。
- 按 canonical 图像完整字节哈希合并完全相同的输入；代表行为该组最小 input_index，文件名为 `<代表input_index:06d>.png`。每条原行仍有映射记录，mask 差异保留但本轮不参与 SfM；不把重复图当成不同观察视角。不同字节不得按 frame_id 或文件名猜测合并。
- 至少两张不同 canonical 图像才能尝试 SfM。COLMAP 的 image_id/camera_id 必须从实际数据库/模型读取，不等于 input_index，也不保证连续。T01 frame_id 与 COLMAP 原生 FRAME_ID 是不同概念。

## 3 固定调用范围和初始配置

调用顺序：SIFT 特征提取、穷举匹配、增量 SfM、原生模型保存与回读。复用 pycolmap API，不重写特征匹配、三角化或束调整。

固定第一版工程配置：CPU；小样例最多 12 张不同输入；每张输入最长边不超过 1024，超限拒绝而非静默 resize；运行线程上限 2；SIFT 最多 4096 个特征/图；随机种子在版本支持时显式设为 0，不能据此保证 SfM 位级确定性。运行参数不作为数学容差。

相机 profile 为 `SIMPLE_RADIAL`、每图独立相机（不假设同一设备/焦距共享），初始焦距为 `1.2 * max(width,height)`，主点为 COLMAP 原生坐标中的 `(width/2,height/2)`，径向参数初值为 0。全部是估计起点，必须记录 `calibration_status=estimated` 和 `intrinsics_initialization=heuristic`，不得把初值当成已标定参数或“无畸变”。允许后端正常优化焦距和径向参数，主点不另行自由优化；保存全部实际生效选项。今后接真实照片时重新核对该 profile 的适用性。

官方在线文档可能对应开发版；以批准的 4.2.1 运行时签名/选项为准。上述能力若在该版本无法实现，交具体不兼容项，不能静默换模型、版本、后端或 GPU。使用受控子进程承载原生后端，保存真实退出码、异常和超时；单次后端调用链最多 10 分钟，避免 native 异常使外层证据完全丢失。

## 4 几何语义

- 本轮只保留 **COLMAP 原生坐标**。COLMAP 左上像素中心为 `(0.5,0.5)`；T01 坐标同一像素中心为 `(0,0)`。同一 canonical 点的关系为 `p_colmap=p_t01+(0.5,0.5)`，仅是坐标约定，不平移/重采样 PNG。
- 原生姿态为 world→camera，列向量形式 `Xc=R*Xw+t`，相机中心 `C=-R^T*t`；相机轴向右、向下、向前。原生文本四元数为 Hamilton wxyz；不假定 Python API 数组顺序与文本相同，优先让原生 writer 保存。
- 原生 camera model、完整参数、共享关系、点 ID、观测和 tracks 全部保留；实际版本产生的 rigs/frames 同样保留。含畸变相机不能只导出 K 后塞入旧的 `stereo.baseline.Camera`。
- 各重建分量有各自坐标与尺度，本轮不拼接、不相互比较绝对 XYZ、不做尺度归一化。`scale_status=unknown`、`length_unit=null`，不宣称单位为米。
- 本轮输出整个场景的稀疏模型，`reconstruction_scope=scene_sparse`、`mask_usage=not_applied`。人工 mask 仅随身份映射留存，尚未得到“物体点云”。

## 5 最小产物

每次新目录保留 `run.json`、`frames_map.jsonl`、`images/`、实际数据库 `database.db`（若该阶段已创建）、`models/<model_id>/`、`logs/` 和选项/文件哈希清单。原生模型同时保存 binary 和 text，分别放 `binary/`、`text/`，避免读取器优先选择 binary 掩盖 text 错误。全部分量保留，不只取 maps[0] 或最大分量。

`run.json` 必填：contract_version、run_id、upstream_io_run_id、data_kind、status、backend（name/version/runtime）、effective_options、input_fingerprint、implementation_sha256、counts、models、stages、errors、cache_hit、coordinate_convention、scale_status、length_unit、reconstruction_scope、mask_usage、quality_status。其中：

- `coordinate_convention="colmap_native_pixel_center_0.5_world_to_camera"`；`quality_status="NOT_EVALUATED"`；cache_hit 固定 false。
- counts：input_rows、unique_images、registered_unique_images、unregistered_unique_images、model_count；点数按每个模型独立记录，跨分量不声称去重总数。
- 每个 models 项含 model_id、binary_path、text_path、registered_image_ids、num_registered_images、num_points3D、roundtrip_verified。model_id 为本次运行内标识，不作为跨运行稳定身份。
- stages 记录实际阶段名、起止时间、状态及日志位置；实际子进程退出码与 Python 异常分开，不给普通 API 调用编造进程 exit。失败/超时的部分文件为诊断，不得标成有效模型。
- errors 为 code/message 数组；未知量用 null 和原因，严格 JSON 禁止 NaN/Infinity。

`frames_map.jsonl` 按 input_index 升序，每条上游输入保留一行，必填：upstream_io_run_id、input_index、frame_id、upstream_image_output、canonical_image_sha256、upstream_mask_output、canonical_mask_sha256、representative_input_index、staged_image_name、database_image_id、registrations、status、reason。registrations 为 `{model_id,image_id,camera_id}` 数组；失败/未注册为空数组，缺失 ID 为 null。status 为 REGISTERED/UNREGISTERED/INPUT_ERROR；重复别名继承代表的注册结果，不能覆盖原身份。不能从排序反推后端 ID。

指纹对上游 run.json/frames.jsonl 原字节哈希、有序逐行身份与 canonical image/mask 哈希、全部生效选项、backend/runtime、契约版本与实现哈希作规范 JSON SHA-256（键排序、UTF-8、无空格、ensure_ascii=false）。implementation_sha256 对 reconstruction/ 下 Python 文件按项目相对路径排序的 `[路径,文件SHA256]` 列表同法哈希。输入处理前后确认文件未变化。本轮完全不复用缓存，每次使用新数据库和模型目录。

## 6 运行结果与测试层级

“可用模型”仅是结构有效：原生回读成功、至少 2 张注册图像、至少 1 个三维点、引用与数值有效。这个下限不证明模型精度或物体完整性。

| run status | 条件 | CLI exit |
|---|---|---|
| SUCCESS | 恰一个可用分量，全部不同输入都注册，导出回读与证据完整 | 0 |
| PARTIAL | 至少一个可用分量，但有未注册图像或多个分量 | 1 |
| NO_MODEL | 后端正常结束但没有可用模型；不得称成功重建 | 1 |
| INPUT_ERROR | 上游/参数/路径/数量不合约 | 2 |
| BACKEND_UNAVAILABLE | 指定后端不能加载或不支持所需能力 | 3 |
| BACKEND_ERROR | 后端异常/实际失败退出/超时，或导出回读失败 | 4 |

输出目录无法写入时 stderr+非零退出，不假称已写证据；其他失败尽量保存 run 与映射。任何非成功结果都不能引用旧运行的成功模型。

T02A 分开记录 unit_tests、native_roundtrip、backend_smoke、sfm_reconstruction、real_data 的 PASS/FAIL/NOT_RUN 及原因。工程交付可以在没有真实数据时完成，但必须真实调用批准的 pycolmap 做原生回读和后端 smoke；mock 只能测试控制流，不计真实后端执行。

最低验收：

1. 两份新定向测试通过。覆盖坏上游/路径、不同 mask 的重复行、非连续 ID、多个分量/未注册、后端失败和超时、新运行不复用旧模型。
2. 用真实 pycolmap 构造明确标 TEST_DATA 的小型已知模型（至少 3 相机、12 非共面点和有效观测），包括 PINHOLE 与非零径向参数的 SIMPLE_RADIAL。分别 binary/text 保存并读回，逐项检查模型参数、姿态、点、tracks、原生像素坐标；独立计算少量已知投影，不拿同一个导出函数生成期望。数值比较使用 config.py 合适容差，不把其值当 SfM 质量阈值。
3. 至少一次真实后端 smoke：可在 T02A 下的临时模拟项目中，用既有 T01 IO 生成 3 张不同常量灰度值的 RGB8 PNG canonical 输入，实际调用 CPU 特征/匹配/映射链路并记录 NO_MODEL。它验证真实依赖和失败传播，不证明成功 SfM。若 API 因无特征明确提前结束，记录实际停在哪一阶段，不伪造后续调用。已知原生模型 fixture 的 roundtrip PASS 不能填入 sfm_reconstruction=PASS。
4. sfm_reconstruction 只有对实际图像执行 SfM 得到可用模型时才能 PASS，并标 synthetic/real。当前 real_data=NOT_RUN。没有照片不要求不断生成更复杂的场景追求成功重建；真实样例与质量门槛另轮安排。

## 7 来源与后续

本契约的字段、默认配置和分阶段门槛是本项目工程裁决，不是 Kozlov 原文结论。后续 T02B 才定义 mask/观测的物体点筛选；T03 才处理去畸变、坐标转换及旧 Camera 接口。

- [COLMAP Python 接口](https://colmap.github.io/pycolmap/index.html)：后端能力与 API，版本签名须以实际 4.2.1 核实。
- [COLMAP 原生输出格式](https://colmap.github.io/format.html)：ID、相机、姿态、点和 tracks。
- [COLMAP 数据库坐标约定](https://colmap.github.io/database.html#keypoints-and-descriptors)：左上像素中心 0.5。
- [pycolmap 4.2.1 PyPI 元数据](https://pypi.org/pypi/pycolmap/4.2.1/json)：固定版本及 wheel 信息；本机运行兼容性待执行者实测。
