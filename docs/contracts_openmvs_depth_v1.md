# openmvs-depth-outcome-v1

范围：独立读取 OpenMVS v2.4.0 Windows x86/x64 little-endian `.dmap`，比较同一 reference image 的两份深度产物。纯 Python + NumPy + Pillow，不启动原生程序、不改输入。不替代 `mvs-mask-v1` consumer，也不证明 SAM3 外形或 CAD 完成。

## 原生格式与严格读取

来源为固定 tag 的 [Interface.h](https://github.com/cdcseacave/openMVS/blob/v2.4.0/libs/MVS/Interface.h#L773-L792) 和 [DepthMap.cpp](https://github.com/cdcseacave/openMVS/blob/v2.4.0/libs/MVS/DepthMap.cpp)。原生文件没有版本号或运行配置；本接口支持指定 ABI，不推断其他构建兼容性。

头为 28 bytes `<2sBBIIIIff`：DR、flags、reserved、image W/H、depth W/H、dMin/dMax。flags 为 depth=1 / normal=2 / confidence=4 / views=8；必须含 depth。随后为 uint16 文件名字节数、原文件名字节、uint32 视图数、uint32 全局 IDs、21 个 float64 的 row-major K/R/C、H×W float32 depth，及依 flags 存在的 normal×3 float32、confidence float32、views×4 uint8。第一个 ID 为 reference，其余为 neighbors，不按文件名数字猜 ID。

`read_dmap` 拒绝未知 flags、截断/尾随字节、零/超限尺寸、depth 尺寸超过 image、空/含 NUL 的文件名、视图数不在 1..255、重复/NO_ID、非有限浮点、负深度/置信度、非正或逆序深度区间。长度算完后才建 NumPy 数组。reserved 原样保存；上游构造函数初始化为 0，本读取器不将非零字节解释成新协议。无强制深度严格落在 dMin/dMax 内、法线单位化或 views 局部索引语义验收。

文件名仅作 bytes 保存；显式 UTF-8 属性可失败，不自动开它指向的文件。返回数组共享不可写 bytes，不能靠 `setflags(write=True)` 修改。元数据为冻结 dataclass，但用户自行构造对象不构成已读取凭据。实际读取 bytes 的 SHA-256 与文件大小保留，读前/读后检查文件身份、大小、mtime、ctime；不能防止刻意恢复全部元数据的对抗性并发写入。

硬上界：每文件 512 MiB；image 与 depth 各不超过 16,777,216 像素，调用方只能降低。此为输入/分配规模上界，非操作系统 RAM 限额。

## 双产物核验

`verify_depth_pair` 只从路径重新读取，所需全部文件须处于一个显式 input_root 内。禁止 control 与 masked 是同一文件；原生内部路径须相对且最终匹配各自传入的真实图像路径。两图实际文件 SHA-256 必须等于同一外部声明值。mask/validity 各有显式 SHA-256，文件要求源编码 8-bit L PNG、0/255、单帧、无透明、无方向变换，且尺寸等于实际去畸变图像。未知像素 validity=0 必须 mask=0。

两份 dmap 须同 depth 尺寸、相同有序 reference/neighbor IDs；reference 须等于外部提供的全局 ID。专门面向 `DepthData::Save` 路径，因此要求 header.image_size == depth_size。**不能用该 image_size 替代源去畸变图像尺寸**，因为上游保存时将 depthMap.size() 传作 imageSize。

调用方提供实际 depth 分辨率上的 K/R/C；复制前核对固定形状与数值类型，最多复制9个标量。K 的绝对容差 1e-7，R/C 1e-8，rtol=0；R 必须正交且 det≈1，K 焦距为正、底行为 [0,0,1]、K[1,0]在1e-7内为0且行列式符号为正（拒奇异）。此接口不从原图 K 简单猜缩放，也不证明外部提供的相机本身正确。

声明 options 必须为 masked ignore_label=0、control=-1。只有 postprocess_dmaps=0 进入排除结果裁定。其他合法 bitset 一律 INCONCLUSIVE，避免填洞等后处理掩盖或重建原来被置零的位置。输入在完成核验前再次以大小和 SHA-256 验稳定；无结果文件覆盖行为。

## 尺寸变换与裁定

等尺寸直接按数组位置核验。尺寸不等时，仅提供 [OpenCV 4.10 标量 legacy INTER_NEAREST](https://github.com/opencv/opencv/blob/4.10.0/modules/imgproc/src/resize.cpp#L1068-L1078) 的参考采样诊断：每轴 `floor(dst_index * (1 / (dst_size/src_size)))`，上界 clamp。它不是 PIL nearest。OpenMVS tag 没固定 OpenCV；尚未证明本机二进制的版本/SIMD 路径及内存 mask 是否串级缩放。因此尺寸变化总是 `INCONCLUSIVE / NATIVE_RESIZE_MAPPING_UNVERIFIED`；参考计数不能升级为原生采样验收。

按 mask=0 统计 ignored，进一步区分已知背景(mask=0,validity=255)与未知(validity=0)。只有同尺寸、无后处理时：

- 忽略像素有非零深度：`VIOLATION`，只说明保存的结果违背声明规则，不直接断言没加载 mask。
- 忽略像素全零且 masked/control 均有前景有效深度、control 有已知背景有效深度：`SUPPORTED`，说明两份保存结果具有声明的排除效果。
- 全零、无已知背景、control 背景无深度等：`INCONCLUSIVE`，不得空洞通过。

结果只作用于一张 reference，不扩展到整场景、邻居匹配窗口、最终融合、网格或实物精度。`native_consumption_status`、`runtime_provenance_status` **始终 NOT_VERIFIED**；configuration 为 DECLARED_NOT_NATIVE_VERIFIED，mask_alignment 为 CALLER_SUPPLIED，quality 为 NOT_EVALUATED。即使 outcome_status=SUPPORTED 也不能写整体 MVS 消费 PASS。

完整原生验收仍须另绑定：可执行文件/依赖版本、真实 argv 与配置、scene 和全部图片/掩膜的身份、mask 加载证据、无旧 dmap 缓存、真实退出码与资源记录，以及实际尺寸变换/空间对齐。合成 DR 夹具不算原生 roundtrip 或真实 MVS。

## 已知上游风险

[ImportIgnoreMask](https://github.com/cdcseacave/openMVS/blob/v2.4.0/libs/MVS/DepthMap.cpp#L335-L359) 对坏图像仅警告并返回 false；CPU 空 mask 可继续估计全图。每级 `INTER_NEAREST` resize 不检查源长宽比。CPU 排除估计坐标，CUDA 在估计结束后清零，不能宣称 GPU 节省了背景计算。[SceneDensify 后处理](https://github.com/cdcseacave/openMVS/blob/v2.4.0/libs/MVS/SceneDensify.cpp#L2291-L2345) 可填洞，几何迭代会替换深度，已有 `.dmap` 可能复用。后续原生任务须使用新目录并冻结实际选项。
