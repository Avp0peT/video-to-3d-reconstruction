# 通用运动候选掩膜

本模块针对同一目标实例的图像序列，不读取物体类别、解剖部位名、指定帧编号、三维稀疏点或固定部件数。它把相对目标主运动具有一致运动证据的局部区域输出为独立候选 mask，整体 mask 仍逐字节保留。它不是完整的语义部件分割，也不以局部仿射残差证明三维刚体或物理关节。

## 自动入口

```powershell
C:\Python313\python.exe -B -X utf8 scripts/run_region_mask_review.py --motion-manifest C:\data\motion_input.json --output C:\data\fresh_review
```

该模式自动按最多六帧、相邻窗口重叠两帧处理。它不启动 SAM3、SfM、OpenMVS、CAD，也不修改默认模型。历史 calibrated-world 入口的原行为保留；两种模式不可混用。

可直接调用 `scripts/prepare_motion_part_masks.py --input-manifest ... --output ...`，另提供 `--window-size 3..6`、`--max-frames`、`--seconds`。默认最多120帧、300秒合作式检查、256MiB产物、2线程；没有硬内存/单算子超时看守。单幅最多4194304像素，不静默缩放。

## 输入

```json
{"schema":"motion-part-input-v1","object_id":"one-instance","frames":[
 {"frame_id":"decode_000020","time_seconds":0.8,
  "image":{"path":"C:/data/20.jpg","sha256":"实际文件SHA256"},
  "mask":{"path":"C:/data/20.png","sha256":"实际文件SHA256"}}
]}
```

帧ID唯一、时间严格递增；必须是同一实例、同一图像域的RGB及单通道0/255掩膜。可加同格式 `validity`；未知域上不得有前景。拒绝缺文件、hash漂移、重复key、非有限时间、不同尺寸，绝不把 input_index 猜作 decode_index。生成绑定清单的上游必须保留来源、时间与实例语义；hash不能证明选中了同一个物体。

## 处理和输出

1. 仅在大掩膜内按格网提取角点；光流正反检查、两端掩膜约束。
2. 拟合目标的主图像运动，随后检验相对运动候选。遮挡或丢失不造点；窗口只跟踪起始角点，不实现长时间重识别。
3. 使用共享track ID和固定anchor组确认窗内多帧证据；歧义不强行关联。窗口内ID稳定，跨窗口另用共享真实帧和坐标映射验证候选关联，不能直接比较局部track ID。
4. 沿原mask内四邻域有限传播，不能越过已有背景孔洞。该域是运动证据附近的候选处理范围，不是像素准确部件边界。
5. 输出 `whole`、每个 `motion_*`、`fallback`、`unknown` 及逐帧图；核验 `parts ∪ fallback = whole`，不增加原背景像素。

### 默认空间连通与时序检查（2026-10-25）

自动入口现在默认在运动候选后调用 `motion_part_coherence.refine_motion_parts`。
同一运动标签下的四邻接连通块先分开，再按同一raw候选的真实track成员建立跨帧关系。
至少4个非共线共同点、双向局部仿射残差、双向mask warp IoU（默认0.25）、互为唯一关联、整段共同ID交集及固定起点直接检查共同决定是否保持ID。
至少3连续帧确认一个短段；遮挡、分裂/合并歧义会截断，不跨缺失帧强行连接。阈值均是可记录的工程筛选条件，不是物理精度证明。

保留的mask只能是原候选的单个连通块，不使用闭运算、填孔或扩大选区。默认每帧每组最多128组件、最多12个输出短段，超限也记录。被拒区域退回整体fallback，同时标记part身份未知；全局mask不变。

`RAW_MOTION_EVIDENCE.json`及`*_raw_*.png`保留同次运行的原候选；`COHERENCE_EVIDENCE.json`记录所有检查与拒绝；页面三列是原图、原候选、检查后候选。每帧报告候选面积减少，不能把删去小块冒称恢复更多形状。

新`evidence_domain_masks`与refined unknown互补；原图像运动证据另存`raw_evidence_domain_masks`。这不表示被拒像素没有发生运动，而是其部件归属缺少足够证据。

`--raw-motion-only`是显式关闭细化及跨窗关联的对照选项，两个CLI都支持。此层的“轮廓检查”仅指时序mask重合，并未实现RGB边缘重新分割或SAM局部重推理，仍不能将输出直接用于自动几何替换。

### 稳健对应核心 v2

`motion-coherence-robust-v2`将“全部共享点的最大误差”改为双向RANSAC的一致核心，再对同一核心做确定性双向最小二乘、秩/条件、最大误差及原mask重合检查。误差阈仍为1.5像素，min_frames、IoU和组件配置不变。每条边至少75%的当前共享点、且至少4点同时被双向拟合接受；被拒点永久退出该段共同身份核心。固定首帧检查继续启用。

75%分母为每条边的当前共享点，不代表整窗口保持初始点的75%。账本逐步及最终记录`initial_support_count`、`accepted_core_count`和`start_support_retention`；`whole_window_ratio_gate=false`。少量异常点不再一票否决整个区域，但不能用逐步比例掩盖累计支持下降。

这属于抗异常对应的工程改动，并不保证保留的局部区域具有正确物理身份。参考OpenCV官方estimateAffine2D说明：https://docs.opencv.org/4.5.1/d9/d0c/group__calib3d.html 。本机实际OpenCV版本与运行代码由各次产物记录；不宣称4.5.1就是当前安装版本。

### 默认跨窗口候选关联（motion-window-links-v1）

同一自动入口在所有窗口写出后运行`motion_window_review.build_review`，输出`window_links/index.html`、`RESULT.json`及输入绑定。只新增候选关联账本，不重命名或改写原mask、不取两窗mask并集、不修补缝隙。

共享帧须frame_id、时间、RGB SHA、whole mask SHA和像素网格一致。至少两张共同活动的共享帧，每张mask IoU至少0.25；各帧2像素半径内用**双向相互最近邻、等距歧义拒绝**建立窗口局部track ID之间的映射。该规则允许半径内有更远的其他候选，并非“半径内只有一个点”。映射在全部共享帧取交集，至少4对非共线点，左右轨迹的位移向量差不超过1.5像素。候选级一对多、多对一全部拒绝；不用最佳分数强行消歧。

相邻窗口的合格候选加入`sequence_*`。若同组已有其他窗口也与新窗口共享至少两帧，必须通过直接配对复核；冲突则截断。没有长程重叠的传递链接只叫候选轨迹，不声称物理身份或已解决累计漂移/遮挡后重识别。所有成员用`(window_id, local_part)`限定。没有足够链接时自动输出`NO_CROSS_WINDOW_LINK`，原候选与整体完整保留。

同时生成同一RGB画面的前窗/后窗/交集三列图，逐对报告IoU、持续坐标映射与拒绝原因。直接比较同一共享帧不需要把相机运动当作零运动；这里对照的是两个窗口对**同一画面**的解释。保留首次ledger hash，任何后续读数漂移都拒绝，不能用新的hash覆盖最初证据。加载最多两个6帧窗口，单窗part数组有128MiB准入上限；仍不是进程级硬内存限制。

这些阈值是可审计的工程规则，精度与完整性仍须实测。RGB边缘可能是纹理或阴影，不能仅凭边缘把局部运动种子扩成完整物理部件。本阶段没有实施RGB轮廓重分割。

`fallback`只代表保留原分割，`unknown`代表缺少运动证据；二者可重叠。共运动的物理部件无法仅凭本算法分开，保持整体是正常结果。相机透视、不同深度、物体形变和错误大掩膜都可能形成假分组。因此当前所有候选 `geometry_accepted=false`，不可直接发布给几何重建替代全局输入。

阈值为版本化的像素阈值（见各次MOTION_EVIDENCE.policy），依赖原图分辨率；不是对所有物体的精度承诺。下一阶段需要基于RGB轮廓/时序可见性验证候选边缘，再在共同相机下验证深度，不能用裸并集或自动补洞填平缝隙。

## 依据与范围

先核对本地 `book_code_audit_20261015/SOURCE_REVIEW.md`：精确仿射/平行投影结论不可直接外推一般非共面透视视频。本实现属于工程运动候选，不冒称完整Kozlov算法。

- Keuper等，ICCV2015，Motion Trajectory Segmentation via Minimum Cost Multicuts：https://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Motion_Trajectory_Segmentation_ICCV_2015_paper.html 。轨迹一致性方向参考，未实现multicut。
- Yang与Ramanan，CVPR2021，Learning to Segment Rigid Motions from Two Frames：https://arxiv.org/abs/2101.03694 。三维刚体运动与二维图像残差应区分；未引入该网络或新增权重。
