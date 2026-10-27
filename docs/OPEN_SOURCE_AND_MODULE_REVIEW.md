# 可复用开源项目与模块缺口复核

复核日期：2026-10-06。范围：官方项目资料、当前仓库源码与实际交付证据；本轮未安装软件、下载权重、运行推理、测试或重建。硬件只读查询确认 NVIDIA GeForce RTX 5060 Laptop GPU、8151 MiB 显存、驱动 582.05；这不代表 PyTorch/CUDA 或任何分割模型已兼容验证。

2026-10-07 用户选型更新：**目标检测、分割及视频跟踪采用 SAM 3**。这取代此前“先GrabCut/SAM2”的主选建议。SAM3接入尚未完成；代码、权重、显存和跨视角效果仍须实测。以下“采用建议”已按该决定更新；历史候选资料保留作为参考。

## 1. 结论与最小组合

保留现有 pycolmap/COLMAP → OpenMVS → FreeCAD 主线。在独立环境接入SAM3，使用文字或视觉提示选取目标并保留人工修正；OpenCV负责预处理、标定及mask几何变换。SAM3已提供文字定位能力，首版不默认再叠加Grounding DINO、SAM2或rembg。网格处理选PyMeshLab，独立几何检查按需接Open3D，CAD继续用已有FreeCAD/OCCT，界面延用现有Streamlit/Plotly；无需重新实现这些项目的底层算法。

首版链路：照片/真实视频 → 输入质量与采集覆盖 → 指定目标与 mask → 相机/稀疏几何 → 去畸变及 mask 同步 → 物体稠密表面 → 有记录的网格清理 → FreeCAD/STEP → 真实外形、实体、尺度分别验收。

自动类别命名不是生成几何的必需条件；目标边界、跨视角实例一致性及坐标变换是必需接口。Kozlov 点集等价识别、图像语义识别和 SfM 局部特征对应是三种不同能力。

## 2. 分割与识别候选

| 项目 | 可复用能力 | 许可/环境核对 | 采用建议 |
|---|---|---|---|
| [OpenCV GrabCut](https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html) | 框选、前景/背景标注后的交互分割；图像质量与标定工具可共用 | [OpenCV Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE)；无需模型权重；Windows CPU 路线。当前重建 venv 尚未安装 cv2 | GrabCut保留备选；OpenCV的预处理、标定、坐标变换仍可复用，不替代已选SAM3 |
| [SAM 2 / 2.1](https://github.com/facebookresearch/sam2) | 点/框提示分割；有序视频的 mask 传播与修正 | 官方说明代码及 checkpoints 为 Apache-2.0。Python≥3.10、torch≥2.5.1、torchvision≥0.20.1；Windows 官方建议 WSL；CUDA 扩展可选。最低版本不是本机新 GPU 的兼容证明 | 仅历史备选；用户已选SAM3，不自动退回SAM2 |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | 文字描述 → 目标框；结合 SAM2 得到 mask | 仓库 Apache-2.0；[官方 tiny 模型卡](https://huggingface.co/IDEA-Research/grounding-dino-tiny) 标相同许可。有 CPU 路线；原生算子/框架兼容仍需探针 | 非默认依赖；仅在SAM3的实际定位缺口需要独立对照时评估 |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | 上述组件组合与视频工作流参考 | 顶层 Apache-2.0 不替代每个组件/权重许可；部分示例接商业云 API | 只参考本地 GDINO 1.0 + SAM2 接口，不整体照搬所有后端 |
| [SAM 3 / 3.1](https://github.com/facebookresearch/sam3) | 文字/视觉概念提示下检测、分割和跟踪 | [自定义 SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)，不能因包 classifier 写 MIT 就当 MIT。[依赖](https://github.com/facebookresearch/sam3/blob/main/pyproject.toml)含 NumPy<2，与当前2.5.3冲突；README 要 Python≥3.12、PyTorch≥2.7、CUDA GPU 环境，权重有访问流程 | **用户已选主方案**。建立独立环境、固定实际commit/checkpoint，先验证图像分割再验证视频；具体3/3.1权重身份不得混写 |
| [rembg](https://github.com/danielgatis/rembg) | 批量去背景、生成单图 mask 草稿，支持 ONNX CPU | 工具 MIT，但权重分别授权。当前 README 默认 BRIA RMBG-2.0，商业使用有另行协议；可选后端也包含云服务 | 仅作候选初稿；必须显式固定本地模型、权重许可和 hash，不能裸用默认或把其当指定实例跟踪 |

视频传播能力不证明无序多视角照片的一致性，这是本项目对接口边界的工程判断。语义标签正确也不能证明轮廓、孔洞或 CAD 尺寸正确。

## 3. 几何、视频与 CAD 可复用项目

| 项目 | 接入用途 | 当前选择与边界 |
|---|---|---|
| [COLMAP/pycolmap](https://github.com/colmap/colmap) | 相机姿态、特征对应、稀疏模型与去畸变 | 已有真实运行，继续复用；多模型不能仅按点数相加或任意拼接 |
| [OpenMVS 2.4.0](https://github.com/cdcseacave/openMVS/tree/v2.4.0) | 深度、融合点云、表面重建 | 工具已准备；固定版本[源码](https://github.com/cdcseacave/openMVS/blob/v2.4.0/apps/DensifyPointCloud/DensifyPointCloud.cpp)有 ignore-mask-label，默认-1表示禁用。需实证 mask 文件名/域/标签与消费行为。[许可 AGPL-3.0](https://github.com/cdcseacave/openMVS/blob/v2.4.0/LICENSE) |
| [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) | 连通片过滤、简化、有限补洞 | GPL-3.0；[过滤器参数](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)含 preserveboundary/preservetopology 等，默认值不可直接当安全策略。补洞阈值按边数，不等于物理尺寸；需前后几何检查 |
| [Open3D](https://github.com/isl-org/Open3D) | 点云/网格分析、距离、预览、简化 | MIT；[TriangleMesh API](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html)可检查自交、流形与水密。闭合或简化成功不证明外形真实；按实际缺口再引入 |
| [FFmpeg/ffprobe](https://ffmpeg.org/ffprobe.html) | 视频解码、逐帧时间信息与抽帧 | 系统工具存在，但产品视频入口未落地。保留真实 PTS、旋转和变帧率信息；不能用名义 fps 倒推全部时间戳。具体发行二进制许可依构建配置登记 |
| [OpenCV 标定](https://docs.opencv.org/4.13.0/d4/d94/tutorial_camera_calibration.html) | 相机标定、畸变处理；可另接标记板定尺度 | 标定/已知长度是可选增强，真实尺度声明必做；物体类别不能作为参考尺寸 |
| [FreeCAD/OCCT](https://github.com/FreeCAD/FreeCAD) | 网格到面片 B-rep、实体检查、FCStd/STEP 与实际查看 | 已有便携1.1.4及合成转换验证；FreeCAD LGPL-2.1。继续复用，无需另接一套 CAD 内核。真实网格/查看验收尚未完成 |

选择 Meshroom/AliceVision 等替代整条前端只适合作独立对照，不作为当前增加依赖的默认方向。单图生成式3D可用于另外的外观演示，但未观测面的生成结果不能当实测恢复表面。

## 4. 当前缺口：区别“没有模块”和“已有但未闭环”

| 优先级 | 模块 | 当前证据与实际缺口 | 最小验收 |
|---|---|---|---|
| P0 | 采集质量/覆盖与输入通用化 | 有固定66图数据与身份链，缺面向任意用户输入的模糊、重叠、底面/遮挡、静态性提示；新profile仍需审查非1024×683等尺寸行为 | 明确拒绝/提示；未观察区登记；不靠注册数声称覆盖充分 |
| P0 | 目标选择/实例分割/人工修正 | 仅人工 mask 解码校验；当前SFM记录 mask_usage=not_applied | 同一object_id、提示与修正记录、真实轮廓/孔洞检查 |
| P0 | mask坐标与跨视一致性适配 | 缺EXIF/resize/crop/undistort完整变换链与视频身份检查 | mask与实际图像/相机同域；硬标签使用离散重采样；遮挡/空mask/身份切换不静默通过 |
| P0 | COLMAP↔MVS几何/身份适配 | prepare_mvs.py存在；局部去畸变已通过，MVS回读匹配在同坐标点对应处失败 | 原始点重数/身份、相机和几何核对；不能简单放宽容差或抹去重复点冒充无损 |
| P1 | 真实稠密物体表面 | 尚无实际深度/稠密点云/网格任务证据 | mask实际排除背景；表面有多视图支持；轮廓对照与缺失登记 |
| P1 | 网格清理/简化/有限修补 | 只有工具/库探针，没有产品处理通路 | 保留原网格、修改日志、边界/自交/连通性及前后偏差；不以凸包替代缺失物体 |
| P1 | 真实CAD、单位、查看与质量 | 转换器已有合成正负例；没有真实照片CAD。scale unknown | FCStd/STEP有效实体并回读；照片对照；未知尺度如实声明，参考长度与独立误差分开 |
| P2 | 视频入口与帧质量 | 未见prepare_video.py或真实视频产物；前次会话中“准备实现”不计代码完成 | 原视频hash/PTS/帧映射/过滤原因；真实视频复用照片通路得到CAD |
| P2 | 统一CLI/UI、缓存恢复、部署 | 三个独立脚本，现有Streamlit是数学实验室；缺全链入口 | 单入口、阶段状态、同身份才恢复、可搬运依赖与人工修正/模型查看 |
| P2 | 独立真实验收与研究对照 | 无真实照片或视频CAD整体验收；Kozlov研究仍有独立缺口 | 独立真实照片及视频；几何/尺度/实体分项结论；研究定义和收益另行验证 |

同图多方向SIFT观测形成的重复view tracks与不同point ID具有相同XYZ是两类不同问题，不能使用同一种自动“去重”掩盖。现有派生只删除20条重复view track对应的完整3D点，不代表解决MVS的同坐标点身份问题。

## 5. 当前结果的准确入口

- FULL66：[ATTEMPT](../outputs/delegation/CAD_FULL66/20261006T153820Z_full66/ATTEMPT.json)、[GEOMETRY](../outputs/delegation/CAD_FULL66/20261006T153820Z_full66/GEOMETRY.json)：63/66唯一注册、两模型44和20张（04重叠）；产品PARTIAL。分析COMPLETE不等于几何完成。
- 选择主分量：[保守派生RESULT](../outputs/delegation/CAD_MODEL_SELECTION/20261006T154420Z_main44/RESULT.json)：44图、2393点，原文件未变；不证明完整外形或尺度。
- 桥接：[R1 RESULT](../outputs/delegation/MVS_BRIDGE/20261006T154353Z_r1/bridge/RESULT.json)：FAIL，verify退出1；去畸变部分通过。不能按旧使用说明只称首次序列化失败，也不能称桥接PASS。
- CAD：[合成转换HANDOFF](../outputs/delegation/CAD_CONVERTER/20261006T153810Z_6afe7ea4/HANDOFF.md)：闭四面体通过、开四面体正确拒绝；真实网格及实际GUI未测。
- 研究：[对应纠错](../outputs/delegation/RESEARCH_CORRESPONDENCE/20261006T151710Z/CORRECTIONS.md)：无序点集双射已修；带索引三维码、AlgT'/AlgSo、2025结构及真实数据适配仍需独立工作。接入开源分割不等于完成Kozlov理论。

## 6. 实施顺序与边界

1. 同时完成最小目标分割接入和现有MVS身份对应检查，冻结mask域/标签/对象身份契约。先少量代表视图验证孔洞和轮廓，再扩展到全部选定视图。
2. 在固定预算下生成第一份真实稠密表面；检查背景、覆盖和拓扑后决定必要清理，保存每一步原始及派生几何。
3. 用现有FreeCAD转换器完成真实模型与STEP回读/查看，建立第一份照片CAD证据。
4. 完成真实视频入口与统一产品入口，再用未参与调参的素材验收。

SAM3的接入分工和前置条件见[SAM3_INTEGRATION_PLAN.md](SAM3_INTEGRATION_PLAN.md)。其单独环境使用模型所需NumPy/PyTorch版本；通过显式图片、mask和JSON契约与现有Python3.13/NumPy2.5.3重建环境交换，不原地降级旧环境。8GB显存只是硬件容量，不是图像或长视频可运行的保证；新推理预算须实测制定，不能套用旧SFM的2GiB Job上限。

每个新候选记录精确commit/版本、二进制及权重hash、许可、输入/输出schema、实际命令、时间/内存和真实退出。此次是选型与缺口审计，不是对上述所有依赖的安装或运行验收；保留既有SFM环境和历史试验，不机械重跑旧任务。
