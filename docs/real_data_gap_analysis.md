# 真实世界输入链缺口分析

> 建立日期：2026-10-02（T00）
> 方法：实现状态依据 T00 代码审阅，环境表为 T00 探测记录；未执行的运行能力标 `UNKNOWN`。T00-R 只修正文档并复核已有证据，没有重跑测试、实验或环境探测。
> 测试与 Git 证据边界见 [`current_baseline.md`](current_baseline.md) §2–3；归档见 [`evidence/t00_review_20261002/INDEX.md`](evidence/t00_review_20261002/INDEX.md)。
> 后续 P1–P11 边界以 [`delegation/DECISIONS_T00_REVIEW.md`](delegation/DECISIONS_T00_REVIEW.md) 为准；本轮不展开 T01–T06 实施。

---

## 1. 目标链与现状逐段标注

需求 §6 的目标链，逐段核对 `kozlov-visual-lab` 的实际支撑：

| # | 链段 | 现状 | 证据 |
|---|---|---|---|
| 1 | **real photos / video 读取** | ⛔ `NOT_IMPLEMENTED` | `find . -iname "*.jpg" -o -iname "*.png" -o -iname "*.mp4"`（排除 `.git`）→ **0 个结果**；`grep -rn "imread\|Image.open\|cv2\|VideoCapture" src/` → **0 命中**。仓库内**没有任何真实图像 I/O 代码**。 |
| 2 | **frames（抽帧/清单）** | ⛔ `NOT_IMPLEMENTED` | 无视频读取 ⇒ 无抽帧。无 `frame_id` 清单结构。 |
| 3 | **object masks** | ⛔ `NOT_IMPLEMENTED` | 无 mask 文件、无分割模块、无 mask 读写。`data/masks/` 目录不存在。 |
| 4 | **camera intrinsics** | ⚠️ `IMPLEMENTED_BUT_LIMITED`（合成侧） | `stereo/baseline/camera.py` 的 `Camera` 可存 `K,[R\|t]`；但 `look_at_camera()` 是**解析构造**（`focal=500.0`、`principal=(0,0)` 为默认参数）。★ **无畸变模型**：`grep -rn "distort\|radial\|k1\|EXIF" src/` → 0 命中。无 EXIF 读取、无标定流程。 |
| 5 | **feature matching** | ⚠️ `EXPERIMENTAL`（且非图像特征） | `stereo/correspondence.py` 在**已给定的两组二维点**上求解双射（oracle 全排列 + 剪枝 branch-and-bound）。**没有** SIFT/ORB/SURF 等检测器/描述子：`grep -rn "SIFT\|ORB\|SURF\|detector" src/` → 仅命中 `teaching/affine_code.py` 的**代数描述子**（affine frame descriptor），与图像特征无关。 |
| 6 | **camera poses** | ⚠️ `IMPLEMENTED_BUT_LIMITED`（合成侧） | 位姿只由 `look_at_camera(eye,target,up)` 解析给出。**无** PnP、无 essential/fundamental matrix、无 bundle adjustment、无 RANSAC：`grep -rn "PnP\|essential\|fundamental\|ransac\|bundle" src/` → 0 个真实命中（仅 `config.py:41` 的英文单词 "essential" 与 `invariants.py` 的注释）。 |
| 7 | **sparse 3D points** | ⚠️ `IMPLEMENTED_BUT_LIMITED`（双视图、已知对应） | `stereo/baseline/triangulate.py` 实现标准 **DLT 双视图三角化**，且在不满足秩条件时**主动抛错**（拒绝伪造深度）。★ 仅 **2 视图**，且输入是"已知相机 + 已知对应点"。无多视图 SfM、无增量注册、无全局优化。 |
| 8 | **dense object point cloud** | ⛔ `NOT_IMPLEMENTED` | 无 MVS、无深度图、无稠密匹配。`open3d`/`trimesh` 未安装（见 §2）。 |
| 9 | **Kozlov representation** | ⚠️ **分层不齐** | 已有：2009 图像码（`coding/`，精确 `Fraction` oracle）、2002 压缩/эскиз/остов（`compression/`）。2025 可以复用基础点集操作，但 `футляр`/`дробность`/`эталон`/`признак`/`существенная часть`/最小面积外接椭圆等专门结构未实现；同名 2002 类型不计作 2025 已实现，见 `docs/kozlov2025_mapping.md`。 |
| 10 | **mesh / 纹理模型** | ⛔ `NOT_IMPLEMENTED` | 无 mesh 生成、无纹理映射、无导出（`.obj`/`.ply`）。 |

### ★ 一句话结论：当前代码在链条上的**终止点**

> 给定「已构造的相机」+「已给定的二维点对应」，可以做出**双视图 DLT 三角化**，并对结果做仿射等价判别。
> 该点**上游**（像素→点→相机）与**下游**（稠密→Mesh）**全部为空**。

---

## 2. T00 环境能力探测记录（`kozlov` venv = `.workbuddy/binaries/python/envs/kozlov`）

平台：Windows 11（10.0.26200），Python **3.13.14**。本表保留原 T00 记录，不表示 T00-R 重新探测了当前安装状态；后续实际安装或执行前应记录当时环境。

| 组件 | 状态 | 检测方式 |
|---|---|---|
| NumPy | ✅ `2.5.3` | `importlib.metadata.version('numpy')` + 实测 import |
| Pillow | ✅ `12.3.0` | 同上。★ T00 已确认可读写图像，足以启动照片 I/O；OpenCV 缺失不是照片输入的硬阻塞 |
| jsonschema | ✅ `4.26.0` | 可用于 T03 外部点云适配的数据契约校验；不替代坐标/对应/投影一致性检查 |
| streamlit / plotly | ✅ `1.64.0` / `7.1.0` | 已有 10 页 UI（`app/`） |
| hypothesis / pytest | ✅ `6.168.3` / `9.1.1` | 测试栈 |
| **SciPy** | ❌ 未安装 | `ModuleNotFoundError: scipy` |
| **OpenCV (`cv2`)** | ❌ 未安装 | `opencv-python` / `opencv-contrib-python` 均不在 dist 列表 |
| **Open3D** | ❌ 未安装 | `ModuleNotFoundError: open3d` |
| **pycolmap** | ❌ 未安装 | `ModuleNotFoundError: pycolmap` |
| **trimesh** | ❌ 未安装 | `ModuleNotFoundError: trimesh` |
| **scikit-image** | ❌ 未安装 | 同上 |
| **torch** | ❌ 未安装 | 同上（`tests/test_validate_fp16_path.py` 之类依赖 torch 的测试在本仓库**不存在**；本仓库测试不依赖 torch） |
| **COLMAP 可执行** | ❌ 未找到 | `command -v colmap` → 无；`find /c -maxdepth 4 -iname "colmap*.exe"` → 无 |
| **GPU** | ✅ 存在 | `nvidia-smi`：`NVIDIA GeForce RTX 5060 Laptop GPU`，driver `582.05`，**8151 MiB** |
| GPU 是否可用于 MVS | ⚠️ `UNKNOWN` | ★ **"识别到 GPU" ≠ "MVS 可运行"**。COLMAP 未安装、其 CUDA 构建状况未探测 ⇒ 按 T00 纪律记 `UNKNOWN` |

### 缺失目录（需求 §15 建议的结构 vs 实际）

| 需求建议 | 实际 |
|---|---|
| `data/{images,videos,masks}` | ❌ 不存在 |
| `papers/kozlov.pdf` | ❌ 不存在（PDF/抽取文本保留在 `.sources/`；来源登记唯一维护点为 [`docs/sources/SOURCES.md`](sources/SOURCES.md)） |
| `configs/default.yaml` | ❌ 不存在（现有配置方式见 `docs/integration_architecture.md` §4） |
| `reconstruction/` `segmentation/` `kozlov/` `kozlov3d/` `visualization/` | ❌ 不存在（包都在 `src/kozlov_lab/` 下） |
| `experiments/` `tests/` `docs/` | ✅ 存在且有实质内容 |

---

## 3. 缺口按"是否阻塞下一阶段"分类

### A. 真实验收前置与可提前开发的接口

1. **真实图像/视频落盘位置与授权**：T00 检查时仓库内没有真实照片，也没有 `data/` 目录。
   P1 已指定 `data/real/<dataset_id>/` 为输入位置，原图只读，输出进 `outputs/`。无真实数据阻塞 **TESTED_REAL 验收**；接口冻结后可用明确标注的 fixture 开发和单测，不把全部 T01 编码卡死。真实验收等待用户提供或明确选定的有许可数据，本轮不下载大型数据集。
2. **图像 I/O 层**：先用已安装的 `Pillow + NumPy` 开发照片路径；P2 允许 OpenCV 在后续视频/图像处理实际需要时作为可选依赖，使用项目隔离环境并锁定实测版本，不改系统 Python。OpenCV 不阻塞照片输入起步。
3. **物体 mask 来源**：需求说"可替换的 segmentation 接口 + 支持人工 mask"。
   无自动分割方案时**不下载/不训练大网络** ⇒ 需要人工 mask 或提示式分割。
4. **相机模型与畸变**：现有针孔 `Camera` 没有原生畸变参数位。P3 禁止真实照片默认“无畸变”：适配契约必须保留后端原生 `camera_model/params`；复用针孔 Camera 前，明确去畸变处理及对应 K、观测坐标与 mask。不支持的模型显式拒绝；零畸变只用于已知标定/已去畸变数据或明确标注的合成测试。

### B. 需要外部基线的缺口（T02 范围）

5. **COLMAP/pycolmap 接入**：全无。定位明确为"现实照片 → 位姿 + 三维点云"的**外部基线**，
   不得声称"实现了 SfM/MVS"（需求 §9）。
6. **MVS/稠密**：无。需求说稠密/mesh **仅在单独分配时推进** ⇒ 本阶段不启动。

### C. 理论层缺口（T04 范围）

7. **2025 三近似**：基础点集操作可部分复用，2025 专门结构仍缺实现，详见 `kozlov2025_mapping.md` 的逐项定义表；不计算未经统一分组口径的覆盖率。2025 PDF p.2 / 期刊 147 提及三维图像和高维推广，后文详细展开二维结构；未给出本项目完整三维实现和真实照片重建流程。词频只能辅助定位，不能裁定概念不存在或唯一出处；新增三维定义仍标 `OUR_3D_EXTENSION`。футляр 还须区分 primary 的固定 R 与 generalized 的受约束给定 r，不能笼统声称所有半径都不可给定，具体条件见 mapping 与 P6。
8. **最小面积外接椭圆**：已有库中尚无该模块。P2 允许 SciPy 作为可选数值依赖，其存在不替代算法本身；`cv2.fitEllipse` 也不是最小面积外接椭圆。T04 先做形式化和椭圆单模块，验收涵盖所有点的包含性、已知解面积/中心、退化输入、收敛与误差界；中心仿射协变性不足以独立验收，数值测试不等于一般性定理证明。

### D. 非缺口（明确已具备，禁止重复实现）

- 二维点集 / 仿射变换 / 内部仿射坐标 / 不变量表 —— `geometry/`（T00 统计 2 856 行；复用时遵守各接口的精确/数值模式，不能把整个包统称为精确实现）
- 2009 图像码：精确 `Fraction` oracle + 数值策略 + 等价判定 + 识别 —— `coding/`（4 017 行）
- 2002 压缩：эскиз / остов / Теорема 4 —— `compression/`（1 540 行，原文逐字引用已复核）
- 三维点集 / 3D 仿射 / 体的码 / `a'`-等价 —— `stereo/`（1 748 行）
- 双视图 DLT 三角化 + 退化拒绝 —— `stereo/baseline/triangulate.py`
- 对应求解（oracle + 剪枝）与**裁决感知**的引擎一致性比较 —— `stereo/correspondence.py`
- 实验记录框架（含 SHA-256 清单与 git provenance）—— `experiments/run_all.py`；11 个必需产物的当前哈希已与原 manifest 匹配，但生成时完整代码状态绑定不足、干净版本上的独立复现尚未核验。verify-only 只表示汇总不重跑，不能否定此前逐脚本执行记录。

---

## 4. 对本阶段范围的直接影响

- T00 阶段不安装包、不下载权重、不建 COLMAP pipeline；T00-R 仅修订文档和归档已有证据，不改源码、依赖或原始实验产物。
- T01 照片输入可在接口冻结后用显式 fixture 开发；真实验收必须落实 §3.A.1 的有许可数据与 §3.A.3 的 mask 来源。
- T02 的真实运行必须等 T01 真实验收；外部基线适配不得重写 SfM/MVS。
- T03 统一为“外部点云 → 现有 Kozlov 数据结构适配”，可提前用 fixture 开发；验收核对约定坐标下逐点坐标、点/帧 ID、观测对应、位姿和投影一致性。有意坐标变换须保存变换及逆变换；仿射等价只是补充诊断，因为错误缩放或剪切也可能通过等价检查。可视化/报告另列后续子任务，T05 必须依赖 T03。
- 独立实验复现按 P10 冻结代码/配置并隔离输出，可以逐脚本记录后汇总；不覆盖旧产物，也不阻塞无关照片 IO 开发。
- fixture / mock / 合成样例**不得**记为真实重建成功，也不得记为 Kozlov 理论验证成功。
