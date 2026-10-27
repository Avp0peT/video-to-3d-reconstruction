# SAM3接入与开源模块分工

日期：2026-10-07（Asia/Shanghai）。用户明确选择SAM3，替代此前SAM2/GrabCut主选建议。状态：选型与接入计划已确定，模型安装、权重获取和真实推理尚未验证。本文件不是成功报告。

本轮已形成可并行派发的[任务入口](delegation/20261007_PRODUCT_ROUND1/00_README.txt)：A负责SAM3独立环境及三帧分割接口，B负责MVS桥接修复。主线拥有[object-mask-v1契约](contracts_object_masks_v1.md)、集成和验收裁决；耗token实现及定向运行由用户转发任务书给外包会话。文件准备完成不表示任务已发送或实现已通过。

## 选定的组件职责

| 环节 | 组件 | 本项目负责 |
|---|---|---|
| 视频解码/抽帧 | [FFmpeg/ffprobe](https://ffmpeg.org/ffprobe.html) | 原视频身份、真实PTS/旋转、帧选择和图像清单 |
| 图像预处理/标定 | [OpenCV](https://docs.opencv.org/4.13.0/d4/d94/tutorial_camera_calibration.html) | 模糊/重复/覆盖提示，mask变换，已知参考长度的登记 |
| 目标检测/实例分割/视频跟踪 | [SAM3](https://github.com/facebookresearch/sam3) | 提示/目标实例选择、object_id、人工修正、置信度与跨视一致性；不默认叠加Grounding DINO |
| 相机/稀疏重建 | [COLMAP/pycolmap](https://github.com/colmap/colmap) | 复用现有4.2.1；身份/相机/坐标协议，资源和失败处理 |
| 稠密点云/表面 | [OpenMVS](https://github.com/cdcseacave/openMVS) | 复用已准备2.4.0；修复桥接验证，去畸变mask实际接入 |
| 网格清理/简化/有限补洞 | [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/filter_list.html) | 原/派生几何、修改区域、保拓扑参数、前后偏差；真实孔洞不自动填死 |
| 独立几何检查 | [Open3D](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html)及已有FreeCAD/VTK | 按需选接口；封闭/流形/自交/距离等检查与失败分类，避免重复造算法 |
| CAD实体/交换 | [FreeCAD/OCCT](https://github.com/FreeCAD/FreeCAD) | 复用现有1.1.4及转换器；真实网格、实体有效性、STEP回读与单位说明 |
| 交互/预览 | [Streamlit](https://github.com/streamlit/streamlit)、[Plotly](https://github.com/plotly/plotly.py) | 沿用现有界面技术，补目标修正、状态、下载和模型查看；预览不替代CAD检查 |

Kozlov理论、正式代码与算法验证仍由研究模块承担；成熟工程组件只能提供重建基线和数据，不能替代理论定义或证明研究收益。

## SAM3前置核验

官方[README](https://github.com/facebookresearch/sam3)列出Python≥3.12、PyTorch≥2.7、CUDA≥12.6，并有CUDA12.8轮子的安装示例。实际选择须适配RTX5060 Laptop并用最小CUDA算子验证；最低版本/驱动存在都不能代替兼容证明。

采用独立Python3.12环境，不原地修改已有重建venv。SAM3当前[pyproject](https://github.com/facebookresearch/sam3/blob/main/pyproject.toml)要求NumPy<2；现有SFM为NumPy2.5.3，必须隔离。没有必要安装训练/开发全部依赖或可选加速扩展来完成第一张图推理。

代码和权重遵循[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)。官方权重通过[Hugging Face模型仓库](https://huggingface.co/facebook/sam3)申请访问并认证下载；实际接入时先检查已有授权状态，不输出token、不通过第三方镜像绕过访问限制，也不把“需要访问”写成“本账号已被拒绝”。具体SAM3/SAM3.1配置、commit和checkpoint需成组冻结，不依赖滚动main默默变化。

## 接口边界

1. 输入明确列出图片hash、图像域（原始/EXIF规范化/缩放/去畸变）、宽高、文字或视觉提示、待选实例及object_id。多实例匹配不能自动取最大实例当用户目标。
2. 模型输出是候选掩码，不是真值。保存原始分数、候选实例、选取规则及人工修改；无匹配、多候选不确定、遮挡/出画和空mask有明确状态。
3. 现有photo-io-v1只接人工mask；自动mask不得静默伪装为人工真值。另行定义自动分割来源/模型/权重/提示的版本化记录，必要时增加独立consumer，保留旧契约和证据。
4. 输出为每图单实例二值PNG及JSON清单，明确0为背景、255为前景、路径/hash和变换链。resize/去畸变时硬标签用离散重采样，并验证有效像素域；不能重编码照片替代独立mask。
5. 有序视频允许调用SAM3跟踪，但离散环拍照片仍需跨视实例与投影检查。所有同类实例的分割不等于指定同一物体。
6. 稠密侧以所用OpenMVS2.4.0实测证明忽略背景标签生效；不能仅检查mask文件存在。照片曝光/纹理保持原始或已记录的既有预处理，避免引入不必要的外观修改。

## 验证次序

- 环境和合法权重可读 → 单图实际推理与显存/耗时 → 少量包含细孔/边界的代表视图 → 人工核查/修正 → 目标一致性的全组mask → mask域转换/消费反例 → 有界真实稠密重建。
- 每项正式运行前冻结输入、工具/权重hash、墙钟/RAM/VRAM/落盘预算。SAM3预算独立于旧2GiB SFM实验；不把资源失败解释为物体不存在或几何失败。
- 检查内容包括主要轮廓、真实孔洞、细结构、背景残留、身份切换与空mask。模型分数高或mask IoU高不等于CAD毫米精度。
- 本阶段与COLMAP/OpenMVS桥接修复可以并行；桥接通过且目标分离实测生效后，才开始第一份真实物体表面与CAD交付。

当前桥接在点对应歧义处FAIL；CAD转换器只有合成案例。选用SAM3不改变这些状态，也不把未观测的底面自动变为已恢复表面。
