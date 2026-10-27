# mvs-mask-v1 — canonical mask到既有去畸变图像的消费契约

范围：消费object-mask-v1侧文件及已接收的COLMAP→OpenMVS桥接产物，不修改photo-io-v1、object-mask-v1或任何旧运行。此契约只规定域转换和文件绑定，不证明SAM3分割质量、实例跨视一致性或MVS实际排除背景。

## 输入与对应

输入：一个既有桥接目录（PLAN/RESULT/undistorted/scene.mvs）、明确的源SFM run（run.json、frames_map.jsonl）、其原生源模型及一个或多个object-mask-v1 manifest。源模型路径须与桥接PLAN绑定一致；只消费桥接中同一选定分量，禁止拼接另一个分量。

从frames_map中的input_index、frame_id、staged_image_name建立到源与去畸变模型image.name的映射；检查source_io run_id、原T01行身份、实际canonical文件hash、相机尺寸和mask尺寸。T01目录取object-mask-v1的source_io.path，核验其run.json/frames.jsonl实际hash与source_io声明，再将run_id连接SFM的upstream_io_run_id；SFM只有run_id，不能据此扫描目录猜路径。多个manifest须指向同一已绑定T01与同一object_id，引用图像与该T01行及SFM实际暂存图像hash均须一致。不得按目录排序、image_id相等、文件名数字或basename猜身份。所有路径归一化后须处于对应输入根内；重复输入/路径别名/目标名碰撞拒绝，不静默去重。

同一图在多个manifest中出现时拒绝重复，即使内容相同；不取第一个/最后一个。仅消费逐帧OK且selected_candidate_id确实指向候选、selected_mask与该候选路径/hash一致的非空mask。来源run可PARTIAL，但NO_MATCH/NEEDS_SELECTION/错误/未跑帧不能伪造为全前景或全背景。绑定实际读取文件hash与上游声明，保留原有来源/质量/跨视状态。

## 几何

选定图像的原生源与目标模型必须同名且姿态一致：R/t、相机中心在既有桥接1e-8容差内。全套相机参数及模型名留证；目标只允许PINHOLE/SIMPLE_PINHOLE。源mask严格为L模式8位PNG、单帧、无透明、EXIF方向1或无、0/255且尺寸等于canonical/源相机。

复用 integrations/mvs_masks/mask_geometry.py，不重新实现畸变近似，不将mask简单resize到目标尺寸。输出尺寸来自实际去畸变相机（当前44图案例1013×675，不能硬写1024×683）。
每个目标像素使用COLMAP中心(x+.5,y+.5)，经目标反投影和完整源相机投影，最近邻离散采样；不再施加InterfaceCOLMAP自己的半像素变换。未知/越界/非有限点输出mask=0且validity=0；有效背景为mask=0且validity=255。分别计数，不把未知当已知背景。转换后空前景不得算可用目标mask。

## OpenMVS 2.4.0 文件布局

按[DensifyPointCloud源码](https://raw.githubusercontent.com/cdcseacave/openMVS/v2.4.0/apps/DensifyPointCloud/DensifyPointCloud.cpp)及[Util::getFileName](https://raw.githubusercontent.com/cdcseacave/openMVS/v2.4.0/libs/Common/Util.h)，mask-path目录使用去掉路径和最后扩展名的basename加.mask.png，例如 images/000017.png → masks/000017.mask.png；不是000017.png.mask.png。Windows不区分大小写，同basename不同路径/扩展名也碰撞，必须拒绝。

输出mask为0背景/未知、255前景；候选消费配置是 --mask-path ABS/masks --ignore-mask-label 0。默认-1会关闭忽略，不能照默认声称排背景。这里只产出建议参数，尚未证明原生消费者实际生效；--estimate-segmentation也不能代替深度忽略的验证。

## 输出与状态

新目录独占创建，拒绝覆盖。仅输出新mask、单独validity PNG、可选overlay及manifest.json；不复制/重编码原照片，不改原MVS场景/模型/历史清单。

manifest：
- contract_version=mvs-mask-v1，run_id、UTC起止、status=SUCCESS/PARTIAL/INPUT_ERROR/BACKEND_ERROR。
- inputs：全部输入manifest/模型/frames_map/桥接文件的路径、字节数和实际sha256；输入前后稳定性。
- implementation：核心及consumer代码hash、pycolmap/numpy/Pillow版本、算法id、tile_rows、max_pixels。
- object_id；image_domain=colmap_undistorted；source_image_domain=photo_io_canonical。
- frames：按全部桥接图像列出input_index/frame_id/image_name、源candidate与mask绑定、两相机/姿态、目标尺寸、status=OK/MISSING/INPUT_ERROR/EMPTY_AFTER_WARP/NOT_RUN、输出mask/validity路径hash及核心counts，失败reason。
- counts由frames重算：expected/ok/missing/error/not_run；error为INPUT_ERROR与EMPTY_AFTER_WARP之和，expected等于其余四项之和及frames长度。
- coverage_status=COMPLETE/PARTIAL；quality_status与cross_view_status保留来源，mvs_consumption_status=NOT_RUN；不得自行升级为已可重建。
- mvs_mask_arguments：仅完整且无冲突时可给出待验的参数数组；实际未调用计0。部分覆盖不得发可直接稠密运行的命令。
- evidence_level必须反映实际输入；测试夹具和合成标签永不写成真实SAM3分割或物体重建结果。

默认要求全部桥接图像有可用mask，缺失exit2。显式--allow-partial允许输出已有结果，缺失项逐项保留、status=PARTIAL/exit1且不提供稠密命令；输入矛盾/损坏仍exit2，内部/产物错误exit4；完整格式转换exit0。全部实际无可用mask不能称SUCCESS。
目录不能创建时允许stderr/exit2；其他失败尽力保留manifest，不补造未产生文件/hash。路径安全、整数计数、有限JSON、输入前后hash及输出PNG回读均须检查。

## 验证边界

主线核心已用真实pycolmap camera对象的15个小型用例及三组既有相机/合成棋盘格验证。后续consumer须验证identity/hash/标签/尺寸/缺失/冲突与PNG落盘，仍不算MVS消费通过。OpenMVS实际mask加载、忽略标签、深度/融合对照另派有界原生任务；真实mask质量与跨视审查独立。不得将旧两个桥接PASS或本格式SUCCESS升级为稠密/网格/CAD成功。
