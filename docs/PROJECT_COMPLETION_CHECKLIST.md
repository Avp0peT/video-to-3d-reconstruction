# 项目完整完成清单：照片/视频到 CAD 与 Kozlov 研究

## 2026-10-19 邻域候选自动复核v17

[候选与实际图像对照](../outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_neighborhood_candidates_20261019/index.html)已接入默认流程。冻结实际forward图像拟合池，排除自身，以同源同目标近邻的鲁棒仿射、留一、空间支持和歧义检查给候选分级。严格计数和覆盖不变；仅符合原双向证据与cycle(2,4]px的候选允许最多8区×2目标真实续跟，新增目标仍用2px回环，不替换测量坐标或注入三维。

真实复核一次114.446s、Job峰468.34MiB、退出0且回收确认；最终1214项全绿。363条区域观测仍107严格通过，另保留70图像候选：5邻域一致、27邻域不足、38未形成合格一致模型。5条都因反向有效像素覆盖不足而缺合格回环，故本次续跟0、新3D点0。独立核源/目标发现其33px局部灰度逐像素相同（两幅完整图不相同），不能当新增视差或三维收益；未知目标mask与源已知mask导致正反支持口径不同。后续需修复支持域语义并实测，不以降低阈值冒领恢复。旧737+32封存文件、897轨迹/230候选/22条件XYZ/4PLY保持，项目与研究总验收未完成。

## 2026-10-19 空间均衡与连接区域v16

[自动图像报告与新旧对照](../outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_connected_regions_20261019/index.html)已接入默认自动复核。3×3空间种子、真实前景图桥接及均衡前沿，允许复用已选节点，最大256区/每批8、仍共享180s/2GiB/2线程。全部选择先冻结，旧配准门不变；首64计划另存，复用同次独立测量只在前缀内评边，不重跑优化。

实测256区，局部连续11、重叠支持5；首64分别3/2。逐源旧64/新64/新256去重覆盖、计划分量与失败原因见报告。图连通与实际跟踪通过分开，不按新增区域数声称建模完整。完整复核108.374s，Job峰470.21MiB；最终1178测试通过、独立读回通过、退出0且回收确认。旧648封印文件/897轨迹/230候选/22条件XYZ/4PLY保持；未新增相机、三维点或表面，质量NOT_EVALUATED、尺度unknown。更多视角、遮挡恢复、区域深度与建模验收仍未完成。

## 2026-10-19 原SAM前景铺区与重叠一致性

[新旧覆盖对照、区域原图和逐边核验](../outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_overlap_regions_20261019/index.html)已接入默认自动点复核v15。两个既有源视角的原mask生成2500个候选，1775个符合窗口条件，预算内先冻结64个新区域，不读取旧点位移；同源兼容路线前缀取最长已声明路径并记录来源。区域整体配准及像素/形变/回环门不变；不以运行结果补选。

64区中6区局部连续通过至少两个目标，4区同时获重叠邻区支持。源input52/decode140的局部连续前景并集由旧8区的2178像素（0.73%）增至5956（1.99%），重叠支持3778（1.26%）；源input27/decode73仍为0。源mask分母分别299750和190735，不跨源合成三维覆盖率。计划覆盖分别10.08%/15.85%，不是全物体；每源16个互不相连的二元区域组，尚未形成连续覆盖物体的网络。32条冻结边的96次目标检查中19一致、0冲突、77未测；未测不能当一致。

两次真实入口：real01因规划器误拒兼容前缀退出2，overlap原生配准0次，失败保留；修正后real02完整复核73.489秒、Job峰394.5MiB、退出0且回收确认。新分支测量9.895秒、含图表15.166秒，165次模板搜索/165次ECC；旧8区同次测量3.225秒。选区及工作量不同，不声称提速或三维精度提高。最终1162项测试通过（首轮1157通过），独立审计24076项、562绑定通过，旧365文件、897条轨迹账本/230候选/22条件XYZ/4PLY保持。

仍无新增SAM/SfM/MVS/CAD执行、相机、三维点或表面。覆盖是源图mask像素并集，重叠估计共享纹理，派生控制坐标不是独立物理对应。源视角与路线仍取已有诊断，未扩展全视频视角；遮挡重识别、不同尺度区域、连通覆盖与三维接入仍待实现。尺度unknown、质量NOT_EVALUATED，项目与研究总验收未完成。

## 2026-10-18 整块区域配准与像素容差

[区域变形框、实际纹理与差异图](../outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_area_alignment_20261018/index.html)随默认自动点复核v14生成。纠正上一轮“区域内逐点LK”的理解：新分支以整块33×33图像共同估计一个六参数仿射变换，每个目标均配准到固定源模板，运动估计不读原点位移。8个源ROI沿用原选区作受控对照，选区本身仍来自候选短轨迹；尚未实现完全由区域驱动的全物体发现。

允许区域像素误差：先用训练像素估计亮度增益与偏置，再在固定留出像素上检查NCC≥0.80、归一化RMSE≤0.7、有效覆盖≥85%。阈值是工程默认值，非标定精度或允许15%任意遮挡的保证；未知中间帧SAM仍为UNKNOWN。整体正反向变换的四角/中心回环误差≤2px单独检查，这些位置由变换推导，不当独立匹配点。

一次实际视频复核62.388秒，Job观测峰376299520B，退出0并确认回收。整体区域测量3.049秒、含报告4.250秒：8区域中2个形成源帧140至141/142/143的四帧连续区域轨迹；14个目标尝试中8通过、6拒绝。26次模板搜索和26次ECC，整体区域分支LK调用0。6次拒绝为4次回环不一致、1次形变越界、1次超出搜索域；本次没有因像素相关/残差阈值拒绝。高纹理相关不保证位置正确，未为追求通过率改阈值；2/8区域与旧点数统计单位不同，不宣称优于点方法。

最终61文件1122测试全过，无失败/错误/跳过；独立回读466个绑定、26组真实模板与搜索裁块、27225个双线性像素重算、50组像素指标与60个派生回环控制位置。旧354份交付保留，原897条轨迹账本/230候选及22条件XYZ/4PLY保持。无新增SAM/SfM/MVS/CAD执行、查询相机、三维点或表面。区域配准是结合书籍整体一致性思想的工程实现，不是Kozlov几何代码算法的直接复现；物理尺度unknown、质量NOT_EVALUATED，项目和研究总验收仍未完成。

## 2026-10-18 多起点批量区域跟踪与有界扩展

[本轮图像与证据](../outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_region_growth_20261018/index.html)把区域计算接入默认自动点复核v13。每个中心展开9×9网格，各点独立测量双向LK；通过固定区域留出检查后向四邻域提出新区域，最多3层/24区域。源图21像素窗口须位于原SAM前景内；查询掩膜未知保持未知，子区域不复制父位移或XYZ。相邻patch和重叠网格可能相关，点记录不能当独立物理点。

一次真实复核57.139秒、Job观测峰373927936B。区域部分8→5→3共16区域，1296条计划记录中194源窗口不合格、667未形成两腿前缀、435形成至少三帧二维短轨迹；按(source_input_index,raw_xy)去重为386个位置。8个根中心7个连续，周围网格另有213条连续；递归后相对根区域新增166个唯一源坐标。区域调用90次批量LK、实际4488个方向测量；区域测量/逐轮记录1.746秒，含报告3.682秒。没有逐点同条件计时对照，不主张加速倍率或三维精度改善。

最终57文件1082测试全过；首轮空区间接线KeyError已修，失败保留。独立核451个绑定、4488条raw记录、2093个FB及1438个仿射投影；旧468份文件不变，原897条轨迹账本/230候选及22条件XYZ/4PLY完整保留。当前只覆盖两个源视角的局部区域，仍无新查询相机、新三维点或表面。物理尺度unknown、外形精度NOT_EVALUATED；项目与研究总验收仍未完成。

## 2026-10-18 查询相机自动验证与第18个研究界面模式

[本轮证据](../outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_query_cameras_20261018/index.html)将查询相机及条件扩展接入默认自动点复核v12。训练/留出身份预先固定，查询帧分别估计焦距，检查SIMPLE_PINHOLE和SIMPLE_RADIAL两种假设；中心主点、方形像素和原条件XYZ是明确假设，不当作已标定真值。训练、初始化及负对照不借用留出观测；扩展排除所有相机训练和留出锚点，保留原掩膜未知状态。

一次真实自动复核48.471秒、Job观测峰369123328B，退出0并确认回收。12组记录的24个相机假设中16个因锚点不足BLOCKED，另8个REJECTED，主要是重投影误差，部分还有负深度、径向域或焦距界限问题。因此本轮新增查询相机与条件三维点均0；原22个条件XYZ及230条短轨迹保留。已有自由投影的内参分解与中心/方形假设存在差异，但不足以区分相机假设、上游XYZ、对应误差或优化器的原因。下一步应有控制地核对这些因素，不能把误差阈值放宽当作建模成功。

最终53个定向文件1032测试全过，首次2个错误标签断言失败已修并留存；独立研究UI76项全过。研究页面新增第18模式：2019椭圆代码未知对应搜索、独立核验和无损下载。完整8点正例及4点负例经实际UI计算封装、写盘、读回与独立核验均VERIFIED；它们是构造的精确数学数据，不是本视频的对应或精度证据。整个项目和研究总验收仍未完成，物理尺度unknown，外形精度NOT_EVALUATED。

## 2026-10-18 自动补测三视锚点与未知对应椭圆代码

[本轮可查看证据](../outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_anchor_routes_20261018/index.html)将新增动作接入统一自动点复核v11：从短轨迹源点沿连续原视频帧，补测至另外两个同模型已保存相机；候选与路线在测量前冻结，保留原230条短轨迹。96个补测候选中36个形成三视观测，15个严格条件XYZ、7个放宽档条件XYZ，14个来源身份此前未连接到XYZ。不同模型与等级分别导出点云，不把它们并成物理模型，不改原生SfM点或相机。查询帧71/72各有14个严格条件锚点并完成留出投影诊断；另一段仍锚点不足。现有相机与径向分支仍是条件假设，尺度与外形精度未验收。

2019研究新增`coding/ellipse_equivalence2019.py`：完整3–8点、任意三点非共线、有理输入下的未知双射搜索；正例核全部索引比值，负例逐排列保存反证，预算不足返回UNKNOWN。专用分块精确JSON保持原通用解码限额，解码与数学核验分开。完整8点正例及4点负例实际写盘、读回、独立核验均VERIFIED；仅是构造的精确数学数据，不声称照片识别改善或论文整体完成。原2026-10-14“未知对应椭圆搜索未完成”在此严格有界域内已被本增量取代；完整退化域、噪声透视图像应用仍未完成。研究UI未新增模式。

最终50个定向测试文件979例全过，先前两批失败记录保留；一次真实自动复核49.971秒，Job观测峰369287168B，退出0并确认回收。独立读回原始LK、身份、固定相机投影、LOO、PLY和查询投影；旧354份交付逐字节保留。新相机/新原生模型点/新表面均0，整个项目仍未完成。下一步是有条件的查询相机及镜头模型验证，再扩展三角化；不以连续性或点数增长替代几何与质量验收。

## 2026-10-14 理解2019论文并落地代码（最新）

当前用户要求按论文思想继续实现。已实际核对S-2019-CODES打印58/59页，形成[思想—代码说明](KOZLOV_2019_IMPLEMENTATION.md)：保留全局标签和完整面积比结构；子代码从完整环境作字面限制；给定双射的跨图比例按原文单向谓词、退化保持及加强全三元组条件分别报告；椭圆比值由引理常数抵消，并输出逐三角形精确中心/矩阵及最小性证书。

新增`coding/environment_code.py`、`environment_proportionality.py`、`ellipse_code2019.py`，均有独立验算。环境完整1–8点，椭圆严格3–8点且任意三点不共线；int64全局标签及256位有理输入，不采样。n8完整56三元组/3080比例，椭圆另有56几何实体。平行四边形不满足原文平面定理域，重叠self-pair不属于代码索引，均有反例覆盖。

[本轮证据](../outputs/delegation/KOZLOV_LEGACY/20261014_environment_codes/HANDOFF.md)：最终58个明确测试文件，**1061个唯一用例+11个子用例**（JUnit1072），0失败/错误/跳过，exit0；界面独立**61项**通过。新增3入口，研究页面合计**17模式**。5个实际数学例写盘、无损读回及独立核验均VERIFIED；正/负/域外语义分开，源文件测试前后无漂移。

2019本批为有界且严格域实现，未知对应椭圆等价搜索、完整共线退化域、2019分割及关联形象系统仍未完成。未运行真实模型、GPU或照片重建，未声称图像失败已改善。以下14模式/951用例及更早内容保留历史时点，不累计为当前总数。

## 2026-10-14 作者既往论文实现增量（最新）

用户明确要求实现Kozlov以往论文算法。本轮以[逐篇实现地图](KOZLOV_PAPER_IMPLEMENTATION_MAP.md)管理原文、构造、定义域和缺口；从官方正常TLS取得6篇既往原文，1999/2000全文失败保留SOURCE_REQUIRED。来源登记纠正S-2002唯一作者，旧PDF不改。

新增S-2002页258/S-2009页102固定朝向与给定对应最优平移，及S-2009页106–107给定位置普通/狭义接受与瓶颈匹配。均为完整1–32个有理点、精确平方距离、可独立核验的正/负证书和有界UNKNOWN。最小圆支撑、增广路径/Hall和有限阈值搜索为项目对作者构造/定义的实现；未声称完整角度搜索或候选互位U/U₀已完成。研究页面增至**14种模式**，新3模式实际AppTest点击、无损下载、独立回读与旧模式全部通过。

纠正旧`certified_nearest`：当前只计算固定摆放Hausdorff，不能认证作者的等距商空间最近模板；现在明确`certified=false`、`nearest_index=null`，保留独立的摆放候选字段及原数值。未把归一化旋转/尺度网格认作精确作者距离。

[本轮证据](../outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/HANDOFF.md)：最终54个明确测试文件，**951个唯一用例+11个子用例**（JUnit962），0失败/错误/跳过，exit0；UI独立**43项**全过，运行前后代码绑定无漂移。5个实际算法样例无损JSON回读后全部VERIFIED。第一次样例的最终记录含Fraction未能JSON序列化，真实exit1，旧产物保留；已修记录脚本并以新目录真实重跑exit0，不借失败运行宣称绿色。

这批是作者既往算法的**部分实现**，整篇集、一般识别、2014/2019环境分割、高层形象、照片真实收益及整个研究预验收仍未完成。2019椭圆代码可在非退化三角形上解析复用面积比，但退化/环境子代码尚未形成完整新接口，不能算全篇完成。下方早前11模式/745用例等保留为历史时点，不与本轮累加。

## 2026-10-14 网络参考后的多中心充分分支

[本轮证据](../outputs/delegation/GENERAL_PIPELINE/20261014_network_followup/HANDOFF.md)登记代码绑定、实际运行与最终统计；本轮`final01`为745个唯一测试项+11个子用例（JUnit756），44个明确测试文件，0失败/错误/跳过，exit0，12.47秒；`ui01`为36项AppTest，exit0，8.54秒。历史测试不重复累计。[官方网络参考清单](../outputs/delegation/GENERAL_PIPELINE/20261014_network_followup/OPEN_REFERENCES.md)区分论文定义、精确证书架构、实代数量词工具与数值候选原型；本轮未安装或运行这些外部求解工具。

`circuit_center_alignment.py`对完整4–7个有理中心采用四点仿射关系下界，完整覆盖全部n!对应，并仅接受可逆候选达到全局有效下界的分支。5/6中心实际运行分别完整访问120/720对应，均取得非零误差E=1/100证书。另一次七中心有界实例已完整成功：5040对应、50400合法四点关系计算；该七点多数位于两条平行线上，仅10个合法四点关系，构造work=1301004在默认2M内、耗时1.45秒，独立无损回读VERIFIED、核验work=292532（见本轮`seven_center02/ENTRY.json`）。该实例不推广为一般七点通过：一般位置七点可能有35个合法四点关系，默认2M预算仍可能UNKNOWN；预算不足不能称穷尽或不存在。多中心强共同H稳定性与原文点位移半径桥接属于此充分域；`circuit_fits.py`另验完整块严格包含。`given_m_compatibility.py`及有限兼容搜索的m=5–7已接入新schema，仍核同一reference、同一F、全部成员。研究UI增至十一种数学模式。

这是由S-2025定义与项目证明得到的`SOURCE_DERIVED`充分证书，不是Kozlov或Boyd直接提供的现成算法，也不是一般strict fits完备求解器。对每个对应取有效四点关系下界的最大值，再取全部对应的最小值；最佳下界不紧、没有达到者或稳定性充分条件失败仍UNKNOWN。已有非共线多中心强充分分支；一般非共线最优互位、原定义局部fits与连续容纳量词仍未解决。未达基数上界的最大粒度、一般完整群、闭形象、全部etalons、三维跨视结构及真实照片/视频收益仍未完成。2007教材合法完整公开版本次未确认。

## 2026-10-14 当前算法增量与未完成边界

本节更新当前状态；2026-10-13及更早证据原样保留，不累计旧测试数。[本轮主线证据](../outputs/delegation/GENERAL_PIPELINE/20261014_algorithm_completion/HANDOFF.md)登记最终整合统计与实现绑定。整个项目、整个研究预验收仍未完成。

* **RC06当前IN_PROGRESS**：`exact_four_center_alignment.py`新增源四中心任意三点非共线、目标rank2的全部24双射精确全局minimax；残差下界与可逆最优区分已达到/仅下确界未达到；平行最优tie域有强共同仿射稳定性和原文点位移稳定半径桥接。`four_center_fits.py`再检查给定完整块、同primary以及所属截断圆域严格包含。一般非共线多中心（已有4–7中心强充分分支）、非平行最优tie下原定义局部fits、一般连续容纳与多Root复合域仍未决。
* **RC07当前IN_PROGRESS**：`finite_compatibility_search.py`自动遍历完整≤7环境中实际reference成员、全部独立reference覆盖、全部完整source标签分划；source凸包不独立也保留。零误差后共线充分求解，成功独立核同一reference、同一F、全部成员。达到u=min成员基数才有精确M；较小见证仅LOWER_BOUND，候选失败/预算/不支持均UNKNOWN。它没有宣称一般存在性或一般最大粒度已完成。
* **完整群新增充分证明及否证**：`full_group_certificate.py`对已证精确M的G，从完整环境减去成员标签并集得到r；r<M时所有严格超群至少新增一个点数≤r的图像，故其M'<M，G可证完整。r≥M只返回UNKNOWN；给定同环境、同精确M的严格超群则NOT_FULL_GIVEN_EXTENSION。超群添加成员图像，不扩大每个原成员。不推出闭形象、全部etalons或特征。
* **当前研究UI**：`app/pages/12_Exact_Research.py`现有十一种数学模式，完整JSON输入、独立核验与无损下载。界面、证书运行、研究总体接受及照片噪声/遮挡条件下的真实收益分开判定。

剩余一般RC06/RC07、RC09三维与跨视结构、RC10/RC11独立整体验证和真实照片/视频对照、RC12报告闭合均保留；2007来源缺失不以本轮推导补造。下方原RC06 BLOCKED、RC07 NOT_STARTED已更新为当前IN_PROGRESS，不勾选整项接受。

## 2026-10-13 算法优先更新（晚于本页下方历史记录）

当前用户要求继续以算法为主、推进到验收前。PC01–PC16与RC01–RC12原需求全部保留，不以新的有限域改写总体目标。下方“四接触仅推导”“407/26”“primary未实现”等文字是早前时点，应以本节及[本轮算法证据](../outputs/delegation/GENERAL_PIPELINE/20261013_algorithm_core/HANDOFF.md)读取当前状态。

* **RC03精确未知标记扩展**：`bounded_parallel_correspondence.py`对完整4–7点理论平行图枚举双射，实际调用两侧AlgT′并独立核全点投影，附CLI。五/六点全枚举可恢复已知真映射；四点例24个合法解全部保留。联合rank判据明确为SOURCE-DERIVED，不同核方向/非平面有限域、平面歧义、未知与多解分别报告；未扩展普通透视照片。
* **RC04/RC05**：一般四接触有理pencil、至多三次实根算术、独立正权重最优证书和统一精确中心API/CLI已实现。覆盖中心族升级v2。固定完整七标签输入全部877分划中433合法覆盖现为**433精确、0未知**；完整证书与无损读回通过。它仍是有限给定点集，不是整图/语义部件或任意规模完备性。
* **RC06**：给定覆盖primary已有有理版与单实代数域版；完整≤7点、m≥2时最多一个四接触块，真三次中心可参与精确半径和全点归属比较。零中心误差strict fits有完整块充分证书及有限候选搜索；共线中心有双方向精确LP原/对偶全局最优和共同仿射稳定性证书，包括非零最优误差。一般非共线非零误差、连续仿射容纳和大集合多Root复合域仍未完成。
* **RC07**：有限环境S'、S*基数、S+标签互斥组完整枚举，以及保持“同一群成员reference、同一primary F、全部成员”量词的给定m兼容充分见证已实现。零误差与共线非零fits证书均可被同群核验；若已证m达到最小成员点数u，则`certified_granularity.py`以同一见证与基数上界证明精确M=u，其他情形只给上下界。一般兼容存在性、一般粒度搜索、完整群/闭形象和etalons全流程未完成；未达上界的充分见证不能给最大粒度结论。
* **可操作研究入口**：`app/pages/12_Exact_Research.py`提供完整精确JSON输入、独立核验、无损结果下载与输入变更失效；工程照片/CAD链和研究数学链仍分开。

新的独立审查已纠正证书bool与0/1等值的类型漏洞。测试证据只证明其声明域。研究总验收还需要一般RC06/RC07算法、RC09三维与跨视结构、RC10/RC11独立复现/真实对照、RC12报告闭合；2007原教材缺失仍不以目录说明替代。不得把本节改写为整个项目已完成或研究已验收。

## 2026-10-13 优先级：先补齐需求，再改善质量

用户最新指示为“先把需求都做完，再慢慢改”。PC01–PC16、RC01–RC12全部保留；当前把功能缺项与质量验收分开推进，不再扩展视频伪影或SAM调参实验。允许开放表面和孔洞，仍不把开放网格写成闭合CAD实体；实测参考长度待定。

**功能清点**：[当前28项审计](../outputs/delegation/GENERAL_PIPELINE/20261013_requirements_priority/REPORT.md)区分缺实现、缺验收、质量改进及外部数据。统一照片/视频入口、Blender文件、照片分面CAD/STEP、参考段定尺度和结果回读已有功能与各自证据，不重新实现。先补有限覆盖中心族、futlyar给定互位判定、阶段续跑、部署配置，然后补其余严格数学域和高层研究功能。普通浏览器操作、干净机器部署、独立数据总验收仍未完成。

**RC04/RC05新增有限中心族和五接触域**：[中心族交付](../outputs/delegation/GENERAL_PIPELINE/RUN20261013_cover_center_family/HANDOFF.md)及[五接触交付](../outputs/delegation/GENERAL_PIPELINE/RUN20261013_contact5_ellipse/HANDOFF.md)已接通：点/线/三角、contact3，再回退到严格正权重有理rank5的contact5证书，保留完整块及未知项。新增集成与五接触87项通过；完整固定7点再独立执行一次，877分划/433合法覆盖，仍为407精确/26未知，没有虚报提升，精确JSON往返与独立证书核验通过。一般四接触代数域、一般ostov及RC04/RC05整体仍未完成。[集成证据](../outputs/delegation/GENERAL_PIPELINE/20261013_family_contact5_integration/HANDOFF.md)。

**RC06新增给定互位谓词**：[合同](contracts_generalized_futlyar_v1.md)按S-2025 PDF13定义，精确核验半径、唯一最近中心、严格点距及非空块，并有独立证书核验；35项最终测试通过。它不搜索仿射嵌入，不求最优fits或primary futlyar，不代表完整RC06。另有JSON输入/输出和独立回读入口 `scripts/verify_generalized_futlyar.py`，8项接口测试通过。

**PC13显式表面及CAD续跑已真实通过**：[本轮交付](../outputs/delegation/GENERAL_PIPELINE/20261013_resume_cad_real02/HANDOFF.md)。新33图生产者完成SfM33/33、SAM31成功/2无匹配并正式提交checkpoint；随后`resume`只运行六个下游阶段，全部成功，Blender独立重开通过；再由`continue-cad`生成有效闭合2000面单实体，FCStd重开及STEP重导入通过。三次明确入口合计378.98秒，上游模型不在续跑时重算，源照片、源任务、续跑任务均保持绑定不变。生产者和表面续跑仍PARTIAL/1，CAD续接SUCCESS/0；不把两张NO_MATCH升级。新增CAD来源/兼容回归67项、主线UI/CLI23项通过，四份真实下载字节只读核验通过。旧45,323,944字节描述触发8MiB限制的失败保留，新生产者验证了分类128MiB修复；未补造旧checkpoint。一般全阶段恢复与研究续接仍未完成。

**PC14新增显式部署配置**：[说明](PHOTO3D_DEPLOYMENT.md)。创建/复审/消费配置已接统一CLI，本机实际41包及权重/封印工具绑定完成，公共`plan`消费成功、未创建执行目录、原生调用0。当前根可用；新根authority/runtime封印的正规迁移和干净机器验收未完成，明确拒绝冒用旧根。配置参数与相关接口24项通过，不等于新机器通过。

**PC14新增迁移规划器**：[说明](PHOTO3D_RELOCATION.md)。终态10项测试通过，本机实际只读规划生成281条文件映射，目标未创建、复制/原生调用0、权重仅stat，状态始终PLANNED_REVIEW_REQUIRED、不授权执行。它补齐迁移前的显式映射与阻塞说明，实际复制、目标环境重建、重新封印与干净机器运行仍未做。先前部署配置绑定旧代码，后续代码变化后须重新创建/复审，不能沿用旧记录声称当前有效。

**研究功能界面**：[使用说明](EXACT_RESEARCH_UI.md)。新增独立精确点集页面，两个显式按钮入口支持完整有限覆盖族及给定中心/半径futlyar，精确结果/证书可下载，输入改变使旧结果失效，rerun不重算。最终代码6项AppTest通过，无模型/原生执行；数学TRUE不等于照片/CAD完成。

**接下来的功能缺项**：先做一般四接触精确域及其中心族集成，再推进有明确原文定义的affine containment/fits、primary futlyar与高层结构；工程侧补一般阶段恢复、正规部署迁移。全部功能再按最终版本做照片/视频总验收。物理尺寸仍需实测参考，2007原始来源及独立验证数据仍欠缺；这些不以数值近似或历史单例冒充完成。

四接触域已有[精确代数推导](../outputs/delegation/GENERAL_PIPELINE/RUN20261013_contact4_derivation/REPORT.md)：有理conic pencil、至多三次驻点多项式、实根隔离与独立正权重证书。当前仅为实现依据，未写入产品、未通过候选求解测试，不将推导算作RC04/RC05完成。

**已做诊断在此收口**：[SAM过滤分数](../outputs/delegation/GENERAL_PIPELINE/20261013_video03_score_diagnostic/HANDOFF.md)只运行一次，36帧状态与既有结果一致，29张候选mask逐字节一致；8个NO_MATCH各有200个query，其实际combined score均未过0.5，未证明模糊/反光/prompt的最终因果。[网格生成对照](../outputs/delegation/GENERAL_PIPELINE/20261013_geometry_generation_bound_images/REPORT.md)未改善结果：SAM外区域总数182691→226874，中位IoU约0.848→0.837；不采用该改动。首次路径失败与第二个独立运行均留存，不继续调参。

下面2026-10-12及更早的“下一步/优先级”只作历史记录，不能覆盖本节；整个项目仍未完成。

## 2026-10-12 当前进展（覆盖下面历史段落的“当前/下一步”）

原 PC01–PC16 与 RC01–RC12 范围保留，整个项目尚未完成。用户最近允许孔洞，因此 Blender 开放表面与闭合 CAD 实体分别验收。

**最新用户优先级：先查失败原因。** 用户明确要求追查图像失败原因，不以手工修改错误结果作为解决方案。下面删面派生仅是未执行提案，现已暂停。当前分别定位SAM识别失败、SfM未注册/多分量、稠密与网格背景伪影的首次出现阶段；原图/mask/网格不变。证据不足时先做能区分原因的对照，再修改生成流程，不以清理后外观代替根因。

**首轮原因定位已完成**：[综合证据](../outputs/delegation/GENERAL_PIPELINE/20261012_video03_failure_cause_review/REPORT.md)确认idx10为两候选未选、8帧NO_MATCH缺过滤前分数；几何验证图分0–14和15–35，14→15仅20原始匹配/0内点；13份保存depth中884719正深度均在自身前景mask，dense却已出现跨视图SAM轮廓矛盾，mesh连面扩大可见片状伪影。背景主导SfM是风险而非相机错误证明。模糊、反光、取景变化仍为相关条件，下一步需过滤前分数及生成机制对照，未宣称唯一根因。

**统一视频入口已真实完成文件交付**：[video03](../outputs/delegation/GENERAL_PIPELINE/20261012_unified_video03/HANDOFF.md)经公共PowerShell入口重新完成36帧准备、SfM、SAM3、表面、Blender和默认研究侧支，未复用先前15视图网格。27/36视图注册、4分量；SAM为27成功/8无匹配/1待选择。选定分量中14视图进入表面，生成11297顶点/22472面/36边界边的开放网格，Blender独立重开通过。总体PARTIAL/exit1，geometry/deliverable_ready=true，object_goal_achieved=false；CAD未运行。[实际UI只读回读](../outputs/delegation/GENERAL_PIPELINE/20261012_video03_ui_readback/RESULT.json)核对下载字节并确认rerun无原生调用。

**视频外形的具体缺口已量化**：[video03全14视图检查](../outputs/delegation/GENERAL_PIPELINE/20261012_video03_allview_review/HANDOFF.md)得到轮廓IoU最小/中位/最大0.7785545791/0.8481801770/0.8930153134。AI辅助查看最差14、最好4及首帧0：14视图网格明显伸入背景，其余也有边界差异。仅为同源SAM一致性，不能作为独立精度或根因证明；剩余22视图仍未知。允许孔洞不等于接受背景伪影。旧pipeline/QUALITY未回改。

**部署预检已产品化**：[使用说明](PHOTO3D_ENVIRONMENT.md)提供不导入模型、不查询GPU、不安装的目录/包元数据/工具存在性检查；7项定向测试通过，本机video/cad静态预检无缺项，但运行兼容性和干净机器部署仍未验收。

**研究新增精确三角凸包与给定覆盖ostov**：[三角椭圆](../outputs/delegation/GENERAL_PIPELINE/RUN20261012_exact_triangle_ellipse/RESULT.json)和[给定覆盖](../outputs/delegation/GENERAL_PIPELINE/RUN20261012_supported_block_ostov/RESULT.json)支持点、段及三角凸包块，并独立核验证书。完整真实7点的预定覆盖33为一三点块加两段，m=3，精确结果及JSON往返通过；无抽样或丢标签。整个7点凸包有5顶点，仍被三角分支明确拒绝。它不是全部433覆盖的中心族、一般RC05或完整照片物体理论验收。

**上述7点新增一般三接触证书域**：[contact3独立API](../outputs/delegation/GENERAL_PIPELINE/RUN20261012_contact3_ellipse/RESULT.json)保持完整7点为输入，只把13/24/26作为椭圆接触支撑。全部7点精确包含及支撑最优性共同证明完整输入最优，独立核验和JSON往返通过；中心为(16500457/32768,35284499/196608)。26项新测试及102项回归通过，旧三角域和given-cover行为不变。完整433覆盖只读分类显示旧域支持363；contact3筛查预测407，但尚未构造并核验该族，一般4/5接触、futlyar、fits及完整研究仍未完成。

**视频背景伪影下一步已收窄**：[只读支持诊断及派发文本](../outputs/delegation/GENERAL_PIPELINE/20261012_video03_artifact_diagnostic/REPORT.md)标出跨视背景候选，同时保留前景冲突；不能直接当作真实背景。拟用明确的共同三视图背景、三像素边界余量作一次独立派生，保留原PLY字节和原面/顶点映射，逐一比较全部14视图，不填洞、不改原pipeline/CAD来源。该派生尚未执行。先前实验15视图的[独立轮廓检查](../outputs/delegation/GENERAL_PIPELINE/20261012_video_experimental_allview_review/HANDOFF.md)也发现背景凸出，IoU中位0.842714；两个模型不是单变量对照。

**最新补充：视频已生成可重开的开放表面**：[新任务终态](../outputs/delegation/GENERAL_PIPELINE/20261012_video_unmasked_disk_budget/FINAL.json)复用真实视频28帧中的15视图原生分量；显式派生仅移除15个含同view重复观测的完整点，725→710点，保留相机/姿态/剩余几何。桥接和mask warp来自独立已完成任务。旧256MiB任务因目录阈值停止；新任务只把磁盘输出上限改为1GiB，重建选项、图像、mask、2GiB内存和时间限制保持一致，真实dense与mesh均exit0，得到21974稠密点及13536顶点/26947三角面。一次[Blender导出和独立重开](../outputs/delegation/GENERAL_PIPELINE/20261012_video_blender_delivery/FINAL.json)通过，53条边界边保留，源码和输入绑定未漂移。它是实验来源的部分视角表面；质量、完整物体、当前mask原生消费实效、物理尺度仍未验收，CAD未运行，不升级为统一视频产品整链验收。下面6视图0融合结果继续作为历史失败保留。

**全选定视图外形检查已实现并实跑**：[可复用CLI及证据](../outputs/delegation/GENERAL_PIPELINE/20261012_silhouette_product/HANDOFF.md)对cad05全部31个选定视图执行投影轮廓检查，IoU最小/中位/最大0.986136808/0.988109452/0.989720975；同次132项输入绑定无漂移。29项定向检查通过。它仍仅证明同源SAM标签内部一致性；idx8/11未恢复，没有独立人工真值或物理精度验收。未回写旧QUALITY或旧pipeline终态。

**研究共线分支新增**：[m-ostov交付](../outputs/delegation/GENERAL_PIPELINE/20261012_ostov_impl/HANDOFF.md)实现完整1–7个互异精确共线二维点集合的独立覆盖中心及精确根尺寸表示；48项定向检查通过，7点合成全877候选得到64覆盖，独立切缝oracle一致。缓存真实非共线7点正确拒绝，未宣称真实物体ostov；一般非共线精确中心、m-ostov、futlyar和fits仍未完成。

**实物尺度决定**：用户本轮明确回复实测长度“待定”。继续任意尺度模型及系统功能；不自行指定毫米真值或精度门槛，不把人为50倍控制作为实物验收。

**2026-10-12 视频局部表面实际结果**：[单次执行](../outputs/delegation/GENERAL_PIPELINE/20261012_video_surface_execute_control/COMPACT_RESULT.json) 对先前masked28的6视图/144点/477观测模型0，新增明确experimental-real来源适配，未冒充原T01/SfM run。真实bridge、去畸变mask/validity与surface plan都通过；Densify实际exit0，2张depth map/113742个深度融合为0点，无dense PLY，mesh未启动，整体FAILED/exit4。全部回收确认、223项当前绑定无漂移，无重试或阈值放宽。这个负结果不能用成功注册6视图替代；旧surface结果的dense_status NOT_RUN仅为旧输出状态，实际原生命令已执行，需另行改善状态表达。

**2026-10-12 RC05小集合新增实现**：`kozlov2025/cover_enumeration.py` 对完整给定1–7点集合枚举全部Bell(N)分划，复用精确独立覆盖求解与独立证书核验，共享时间/工作预算；未知或中断不报完整。新增及原覆盖合计90定向用例通过；共线4点8覆盖、正方形14覆盖、共线7点877候选/64覆盖。[交付](../outputs/delegation/GENERAL_PIPELINE/20261012_cover_enumeration_impl/HANDOFF.md)。仅完成小集合独立覆盖族，不是一般exactcenter、m-ostov、futlyar、fits或语义部件发现；下面RC05旧NOT_STARTED条目保留历史，当前应读作局部实现/验证。

**2026-10-12 新增尺度交互证据**：[measurement UI](../outputs/delegation/GENERAL_PIPELINE/20261012_measurement_ui_check/ACTUAL_UI_READBACK.json) 已接入参考段、记录文件、可选独立尺寸及下载；31项定向检查通过。一次真实CAD原生控制用人为指定倍率50，新FCStd/STEP有效并独立回读；原始模型不变，页面rerun原生调用0。控制明确为 `CONTROL_ASSIGNED_LENGTH_NOT_PHYSICAL`，物理尺度/误差仍未验收。超用户声明阈值的独立检查会阻止成功；无检查为NOT_EVALUATED。浏览器 `file://` 被工具安全策略拒绝，未绕过，普通浏览器交互仍缺。

**2026-10-12 新增外形证据**：[三视图SAM一致性](../outputs/delegation/GENERAL_PIPELINE/20261012_cad05_silhouette_review/HANDOFF.md) 用当次真实去畸变K/R/C投影raw表面，固定首/中/末视图0/17/32的轮廓IoU为0.986493/0.987142/0.988653，叠图位置吻合。这是与同源SAM标签的内部一致性，不是独立真值、完整外形或物理精度；未回写旧QUALITY和旧pipeline终态。

**最新整链已完成文件交付**：[cad05](../outputs/delegation/GENERAL_PIPELINE/20261012_unified_cad05/FINAL.json) 经公共 PowerShell 命令一次执行，原始 33 张骰子照片 → 单分量 SfM（33 注册）→ SAM3（31 成功、2 NO_MATCH）→ OpenMVS 表面 → Blender → 固定六点研究正负对照 → CAD。总体 PARTIAL/exit 1，`target_achieved=true`，所有原生进程确认回收、源和代码前后不变。Blender 独立重开、有效闭合 2000 面单实体、FCStd 重开和 STEP 重导入均通过。[页面实际回读](../outputs/delegation/GENERAL_PIPELINE/20261012_cad05_ui_readback/RESULT.json) 显示四个正确下载，rerun 不启动原生任务。此记录替代下面“新版统一整链未通过”的当前口径；cad01–04 原始失败均保留。

**仍未完成的终点**：同次 `QUALITY.json` 为外形 NOT_EVALUATED、完整物体目标 false；尺度 unknown，精度未评估，CAD 是分面实体。视频同 28 视图的前景特征对照虽真正排除了背景特征，却使注册视图 21→9、最大分量 15→6，未改善外形覆盖：[对照结论](../outputs/delegation/GENERAL_PIPELINE/20261012_video_mask_sfm_pair/FINAL.json)。新的15视图开放表面另见顶部，不改写该对照结论。普通浏览器使用、干净环境部署、尺度交互的实物验收、独立外形/尺寸误差与研究全部要求仍需分别关闭，不以单例成功推广。

**研究新增的有限结果**：2025 独立覆盖已有精确 Fraction 求解及独立见证校验，固定真实图像的 1123 个已跟踪观测经保留全部别名的坐标集合映射为 1010 点，两个给定组的严格凸包分离证书通过。它不是语义部件发现、完整 ostov/futlyar、未知对应求解或照片重建收益；独立复核还指出零预算空工作路径及完整元数据验证的契约边界，尚未宣告研究线完成。2007 原始教材依赖仍未取得可核定理全文。

接下来优先：外形质量与视频覆盖、真实尺度产品接线及参考实测、普通用户入口/部署复现与完整研究验证。以下以 cad04 为最新的表述均属于保留的上一轮状态。

统一主命令已接通 `--target cad`/`--freecad-python`，UI以“同时生成 CAD 实体（FCStd / STEP）”勾选进入该目标，默认仍为Blender。CAD使用不可变 `CAD_SOURCE.json` 的明确几何中间状态，再以既有固定quadric方案和原生门执行；失败保留已存在的Blender，整体CAD目标未达成。研究默认自动固定六点正负对照。新版统一CAD照片整链已实际尝试但未通过：cad04的SAM3资源观察器在加载前因 `nvidia-smi rc=255` 停止，33帧均未开始；本次没有Blender或CAD成果。既有骰子交付和UI→CAD续接仍是各自旧任务的成功证据，不能代替本次完整目标验收。

- **最新统一CAD失败边界**：[cad04终态](../outputs/delegation/GENERAL_PIPELINE/20261012_unified_cad04/FINAL.json)确认外层completed/exit5、回收及源/代码稳定；输入成功，SfM注册33图但生成3分量，SAM3为RESOURCE_LIMIT、0成功/33 NOT_RUN。首因是设备查询失败，不是测得RAM/VRAM越限或超时；设备用量为null、加载未开始，剩余load/batch预算约288.922/888.922秒。Git绝对路径、41包、权重及HEAD/clean前置检查已通过。后续表面、Blender、研究及CAD未启动，`geometry_deliverable_ready=false`、`deliverable_ready=false`、`target_achieved=false`。旧失败与既有骰子模型都保留，不自动重试或借旧产物宣称新目标完成。

- **照片系统流程已实跑到 Blender**：颅骨 66 输入/42 有效视图、骰子 33 输入/31 有效视图，两次 PARTIAL 且 `deliverable_ready=true`，独立 Blender 回读通过。对应 [颅骨证据](../outputs/delegation/GENERAL_PIPELINE/20261011T151000Z_main/execution02/FINAL_SUMMARY.json) 与 [骰子证据](../outputs/delegation/GENERAL_PIPELINE/20261011_dice_e2e/ACCEPTANCE_EVIDENCE.json)。输入缺失、分量未选和 NO_MATCH 不伪装为成功。
- **真实照片几何 CAD 增量**：骰子 quadric 简化得到 2000 面，采样偏差约包围盒对角线 0.362%，体积变化 0.151%；FreeCAD/OCCT 有效闭合单实体、FCStd 重开、STEP 重导入和数值检查通过。[单变量对照](../outputs/delegation/GENERAL_PIPELINE/20261011_dice_cad_quadric/CONTRAST_RESULT.json)。之前 decimate-pro 产生无效拓扑的失败保留，未放松门限。PC10 的几何转换取得实际进展；物理尺度、照片精度、参数化设计与 GUI 操作不因此通过。
- **外形与掩膜**：AI辅助预览能辨认骰子，颅骨有伪影，视频局部模型质量不足。固定三个骰子视图的label0/-1单变量depth对照已观察到背景正深度被排除、前景双方保留：[实效记录](../outputs/delegation/GENERAL_PIPELINE/20261011_native_mask_control/DEPTH_STAGE_BEHAVIOR_EVIDENCE.json)。两份完整dense执行均触及原600MiB输出阈值停止，完整densepair为INCONCLUSIVE；三视图实效不推广为全视图/最终网格保证。真人真值、量化外形精度仍未完成。
- **界面恢复**：已有任务载入、阶段证据校验和经哈希验证的下载已实现；实际已完成 CLI 结果通过 Streamlit AppTest 只读载入，未启动原生任务。[实际载入记录](../outputs/delegation/GENERAL_PIPELINE/20261011T151000Z_main/UI_REAL_READBACK.json)。浏览器服务启动被自动审批拒绝，未绕过，真实浏览器交互未验收。
- **UI→CAD实际闭环**：既有骰子任务通过真实页面函数触发现有CAD CLI、准备、有效单实体、FCStd重开、STEP重导入及下载身份核验；rerun原生调用0次。[独立新任务](../outputs/delegation/GENERAL_PIPELINE/20261011_cad_ui_fixed/FINAL.json)。之前默认decimate-pro拓扑失败及UI准备schema误拒均保留。新增统一CAD目标与immutable几何快照已做离线接线验证；其最新真实整链cad04已尝试失败，不能用旧续接成功替代本次输入→CAD通过。
- **研究侧支**：骰子已成功 SfM 模型按 point_id 固定前六点，调用既有精确索引代码，产生 15 四元组/210 记录。[真实调用](../outputs/delegation/GENERAL_PIPELINE/20261011_research_code6_run01/dice_real/RESULT.json)。精确性针对存储的 binary64 坐标；不是完整物体、未知对应恢复、AlgT′/AlgS0 透视重建或完整 2025 理论。
- **研究自动正负对照**：默认侧支已接固定仿射正例/单点扰动负例，真实骰子六点正例EQUIVALENT、负例穷尽720双射NOT_EQUIVALENT，独立Fraction行列式oracle同意：[独立证据](../outputs/delegation/GENERAL_PIPELINE/20261011_research_controls_01/HANDOFF.md)。这次仅研究子进程，无重建/GPU；没有把该新功能写成新版照片整链验收。对照失败或预算不足有独立状态，不借主表示SUCCESS冒充PASS。

接下来优先关闭：cad04设备观察查询失败的准确原因、新版统一CAD目标的真实整链、全dense掩膜对照、双入口外形质量、用户入口与部署复现、尺度/实物误差及未完成研究要求。以下旧RUN、失败次数、局部验收和原始条目仍按其原范围解释，不回写成当前成功。

## 历史进展

2026-10-08 pilot06复核（覆盖较早下一步）：[主线报告](delegation/20261008_PILOT06_REVIEW/REVIEW.md)确认父层5秒freshness接线真实生效；segment返回2次，但产品只提交idx17，idx34是未提交中间mask、idx59未开始。RAM越限读数早于第二帧，原观察者等待设备查询后才裁决；主线已修正RAM即时锁存和开始帧状态，分别16项/12项离线验证通过，失败脚手架记录保留。现有可消费mask仍1张，内存分配根因和峰值收益未证明。下一项为[A内存诊断包离线准备](delegation/20261008_PILOT06_REVIEW/A_MEMORY_DIAGNOSTIC_PREP.txt)，不执行真实模型、不提限；新起点38项，B/C本轮不接新任务。

2026-10-08 pilot05最新复核（覆盖较早下一步）：[主线报告](delegation/20261008_PILOT05_REVIEW/REVIEW.md)确认一次运行在加载前因设备freshness接线失配停止：策略声明5秒、底层漏参按2秒判断，3.969秒被拒。实际worker身份已登记但未被唯一资源样本覆盖，0完成帧/0新mask，不能比较内存峰值；空stderr也不证明torch未导入。主线已直接修正接线及时间/命令记录，36项离线行为验证通过，保留一次脚手架启动失败；资源容量限额未提高。下一项为[A-FRESHNESS-REAL3](delegation/20261008_PILOT05_REVIEW/A_FRESHNESS_REAL3_EXECUTE.txt)，仅用户转发后固定pilot06一次真实尝试，主线未执行。B/C继续停止，真实分割仍只有pilot04的1张，照片到CAD终点未变。

2026-10-08 A数组/B-R3主线收口（本段覆盖较早下一步）：[本轮复核](delegation/20261008_AB_R3_REVIEW/REVIEW.md)已关闭A的bool/object/BF16转换边界，最终CPU torch验证84项全过；B关闭完整SUCCESS路径异常、子集和scene几何绑定缺口，56项定向测试全过，并实际将idx17真实mask消费到44图桥接，结果仍为1张OK/43张缺失的PARTIAL，不可发布MVS参数。新A执行包已独立104项离线通过，冻结35文件/41包；[下一份外包任务](delegation/20261008_AB_R3_REVIEW/A_ARRAY_REAL3_EXECUTE.txt)只授权转发后pilot05一次三帧验证，原12GiB限额不变，尚未执行。B/C本轮停止。内存收益、真实多帧质量/跨视、完整mask覆盖、实际稠密排背景及照片CAD仍待验收。

2026-10-08最新进展（覆盖下列较早“下一步”）：[本轮复核](delegation/20261008_ABC_R2_REVIEW/REVIEW.md)已核实A pilot04第17帧首张可接收的真实SAM3掩膜，1/3完成、第二帧RAM软阈值停止，实际前向兼容性走通但无BF16数值对照/跨视验收；B R2虽48合成测试通过，仍有原始/canonical摘要误用、66→44集合和scene身份等阻塞，下一项是真实单mask的PARTIAL消费；C由主线补齐换名共同点分支，一次31测试通过，数学源码不变，当前PASS_SCOPED仅限已知必需分支与指定有理对照。[A离线转换优化](delegation/20261008_ABC_R2_REVIEW/A_ARRAY_CONVERSION_OFFLINE.txt)与[B实物consumer返工](delegation/20261008_ABC_R2_REVIEW/B_REAL_MASK_CONSUMER_R3.txt)已准备供用户转发，C本轮结束。真实稠密表面与照片CAD闭环尚未完成，旧超额和留证问题不被新测试抹去。

2026-10-08最新A/B/C复核见[联合审查](delegation/20261008_ABC_REVIEW/JOINT_REVIEW.md)：A pilot03为真实RAM软阈值停止，原生权重加载返回1次而backend.load未完成，仍无真实mask；B真实selected_mask字段不兼容、身份绑定与失败终态有缺口，尚不可接收；C的合法退化换面反例已实际误拒，BED成功分支已实际证明可达，撤回required_branch_gate全通过。主线本次只做一次独立轻量反例入口，未运行照片/GPU/MVS/CAD。下一轮B/C任务与各自当前起点绑定位于同目录；旧RUN和历史证据保持原样。以下记录按各自时点阅读，不以历史“下一步”覆盖此更新。

A加载审计小修已由主线完成：[新包](../outputs/delegation/SAM3_REAL3_LOAD_AUDIT/20261008_main/HANDOFF.md)最终104项纯数据通过，新BUNDLE 11项核对；旧产品34项/旧包7项未变，真实模型执行0。新增loader返回/设备迁移阶段记录，完整键审计在loader返回后、迁移前阻断；尚未证明内存峰值改善。[A一次新诊断](delegation/20261008_ABC_REVIEW/A_LOAD_STAGE_EXECUTE.txt)、[B consumer返工](delegation/20261008_ABC_REVIEW/B_MASK_CONSUMER_R2.txt)、[C算法分支返工](delegation/20261008_ABC_REVIEW/C_LABELLED_STEREO_R2.txt)均已准备，等待用户转发，不由主线自动联系外部会话。

登记日期：2026-10-06；已补入当日新照片运行结果。项目根目录以当前 Git 仓库根目录为准。

当日补充复核与开源选型见 [OPEN_SOURCE_AND_MODULE_REVIEW.md](OPEN_SOURCE_AND_MODULE_REVIEW.md)。该报告区分已实现、合成验证、真实部分结果和未实现模块；下列新增说明更新旧快照，不改写历史验收文件。

2026-10-07分工更新：用户选定SAM3，并要求耗token任务交由其转发给外包会话。[本轮任务包](delegation/20261007_PRODUCT_ROUND1/00_README.txt)允许A（SAM3接口）与B（MVS桥接）并行；主线负责协议、整合和接收。该更新仅登记可派发工作，不提升以下功能的完成状态。

同日回报复核：[A/B联合复核](delegation/20261007_PRODUCT_ROUND1/REVIEW_AB_20261007.md)已接收B的4图/44图桥接功能与几何结果；RAM监控要求未满足，不能称全项合规。A有27项假后端测试和AUTH_REQUIRED产物，但主线另复现五类接口缺陷，改派A-R1离线修复；torch/CUDA/真实SAM3仍未就绪。以下历史快照保留原时点状态；真实稠密表面与CAD尚未完成。

A-R1后续复核：两环境66项历史测试存在，主线9项原反例回归通过；[新的产品路径探针](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_R1_20261007.md)仍发现看守未接通、短暂越限丢失、采样失败后继续运行及收尾缺口，新增3模块亦未纳入原完整交付。当前仅派A-R2运行控制修复；没有启动安装/真实推理，也未因用户询问替代方案而自动切换模型。

A-R2最新复核：用户明确保持SAM3；[主线复核](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_R2_20261007.md)核实76项历史测试及16文件终态，并恢复完整16文件真实起点。最小离线探针仍发现活跃worker回收回调无对象、工作期间资源观察缺口、close/末尾事件未完整进入裁决。当前派A-R3集中修复，未放行真实模型；B保持收口。后续为本地环境恢复/CUDA验证→三帧SAM3→mask外形与跨视检查→稠密/网格/CAD。当前仍无真实SAM3成功或新的网格/CAD证据。

A-R3最新复核：[主线报告](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_R3_20261007.md)核实85项历史通过、17文件终态和7个可写起点。活跃load/frame停止和close错误降级已有证据，但持续观察仍读启动器、空通道漏拒、GPU查询未接线；后代身份/异常回收和未知事件裁决仍有反例。下一项为A-R4定点修复，保持SAM3；不重复整份混有真实进程的测试。真实CUDA/SAM3仍NOT_RUN，B无需重跑，产品终点尚未改变。

A-R4最新复核：[主线报告](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_R4_20261007.md)核实15项纯数据通过与17文件终态。真实批次原始2过1败，按目标行为复核为场景1 INVALID_TEST、场景2 FAIL、场景3 observer.start切点有限PASS；不能按pytest总数接收运行控制。A-R4原像目录为空、冻结记录缺失，主线已恢复全部8份起点并核对旧哈希。当前派[A-R5剩余缺口](delegation/20261007_PRODUCT_ROUND1/A_R5_REMAINING_GAPS.txt)：统一CPU/CUDA设备要求、保护完整post-Popen生命周期、回收所有已拥有成员，再验证两个剩余真实假后端场景。B保持功能收口；CUDA/真实SAM3仍NOT_RUN，尚无新的稠密/网格/CAD结果。

A-R5最新复核：[主线报告](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_R5_20261007.md)确认INCOMPLETE/NOT_ACCEPTED：8份原像和17份终态有效，保存JUnit为13败9过；默认auto已经改好，但策略字段/成员类型及控制路径仍未接通。旧补丁把9份既有文件当新增，不能按其报告反向恢复；主线纠正delta在scratch重建17/17通过。预算总批数/40分钟耗尽缺完整证据，不按自述当实测。下一项[A-R6](delegation/20261007_PRODUCT_ROUND1/A_R6_OFFLINE_CLOSURE.txt)只收口离线代码，真实进程两场景另派；真实CUDA/SAM3依然NOT_RUN，B保持收口，稠密/网格/CAD无新增完成结果。

A-R6主线直接实施：用户改为由主线收口当前修复，减少交接。[交付记录](../outputs/delegation/SAM3_A_R6/20261007T041108Z_main/HANDOFF.md)为OFFLINE_TARGETED_PASS；最终字节47例全部通过（含旧22例），两批pytest累计14.329秒，首批43过3败完整保留。设备策略传递、成员句柄回收、统一截止、异常收尾及相关文档已修正；17文件补丁前向重建一致，51只读项未变。仅完成离线范围，两个CPU真实进程场景、CUDA、SAM3均NOT_RUN；下一步先做这两个场景，B不重跑。

A-R7主线真实CPU验证：[交付记录](../outputs/delegation/SAM3_A_R7/20261007T042737Z_main/HANDOFF.md)为两场景首次2/2通过，外层2.488秒；实际worker进入segment后RAM注入触发终止、root正常结束后回收已登记且仍活跃后代，均由真实OS身份和独立句柄确认。共6个产品成员，17个绑定文件与A-R6最终字节一致。未做真实RAM压力、CUDA或SAM3前向，亦不覆盖未登记迟生后代。下一步可转发[A-ENV环境恢复](delegation/20261007_PRODUCT_ROUND1/A_ENV_RECOVERY.txt)，仅环境与小型CUDA/驱动验证；尚未派发或执行，三帧真实SAM3另行处理，B不重跑。

A-ENV主线复核：[报告](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_ENV_20261007.md)接收torch2.11.0+cu128/torchvision0.26.0+cu128在专属环境安装、保存的真实CUDA算子对照和产品显存查询；17项绑定未变。整体FUNCTIONAL_ACCEPTED_WITH_DEVIATIONS：探针实际启动3次、失败后继续，第2次rc0仍含步骤错误；GPU执行不能记1。依赖仍未闭合，NumPy2.5.2与固定SAM3源码<2冲突，缺口还包括Windows Triton及其他传递导入。下一项[A-DEPS](delegation/20261007_PRODUCT_ROUND1/A_DEPS_IMPORT_CLOSURE.txt)只完成依赖和真实CPU导入/分词，不重装torch或重跑A-ENV。模型加载/前向仍NOT_RUN，稠密/网格/CAD无新增完成。

A-DEPS最新主线复核：[报告](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_DEPS_20261007.md)接收依赖安装、原始21模块CPU导入与真实tokenizer字段；当前41顶层包的声明依赖独立检查无冲突，26项绑定和13项旧清单不变。原探针FAIL/exit1保留；其版本冲突可由setuptools内置元数据后覆盖复现，原marker因果解释不成立。整体FUNCTIONAL_ACCEPTED_WITH_DEVIATIONS。下一份[A-REAL3](delegation/20261007_PRODUCT_ROUND1/A_REAL3_LOCAL_PILOT.txt)在新额度内尝试当前环境CUDA检查、本地权重和最多三帧，尚未发送或执行；模型/真实mask/稠密/网格/CAD仍无新增完成证据。

A-REAL3主线复核与修复：[报告](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_REAL3_20261007.md)确认原产品运行BACKEND_ERROR、三帧NOT_RUN；原包装还有重复读取权重、未捕获真实加载返回、A/D不阻断等缺陷，不能只补PYTHONPATH重跑。主线在[新RUN](../outputs/delegation/SAM3_REAL3_REPAIR/20261007T062459Z_main/HANDOFF.md)直接修复，最终39个离线用例和真实轻量模块导入通过；33文件基线、旧RUN均未改。主线共2次轻量进程尝试（首次自身驱动属性错误已留证修正），0次模型/GPU。下一项[固定入口执行](delegation/20261007_PRODUCT_ROUND1/A_REAL3_EXECUTE_FIXED.txt)尚未派发/执行，SAM3真实加载与mask仍未达成，三维/CAD无新增完成。

A-REAL3 pilot01最新复核：[报告](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_REAL3_PILOT01_20261007.md)接收一次执行失败即停及当前环境CUDA证据。原生load_state_dict返回1次，1134主模型键全覆盖；22个unexpected经固定源码证实为关闭的SAM2交互neck。主线旧规则过严，新建精确兼容规则并完成98项离线测试、1次轻量导入，本轮模型/GPU0。旧pilot01保持BACKEND_ERROR，backend.load完成0、segment0、mask0；采样提交内存峰值11.93GiB接近12GiB阈值。下一项[固定新入口](delegation/20261007_PRODUCT_ROUND1/A_REAL3_OPTIONAL_BRANCH_EXECUTE.txt)准备完成但未发送/执行，只给pilot02一次新额度。外形、跨视及三维/CAD没有新增完成证据。

A-REAL3 pilot02最新进展：[主线复核](delegation/20261007_PRODUCT_ROUND1/REVIEW_A_REAL3_PILOT02_20261007.md)接收GPU模型加载、22键规则及gate D；首帧真实segment已尝试但dtype失败，完成帧0、mask0。主线补CUDA BF16 autocast、严格OOM类型和worker资源状态传递，123离线用例与一次轻量导入通过，5文件补丁精确重建；本轮模型/GPU0。旧报告资源峰值漏父层OS样本，真实worker提交观测峰值仍约11.93GiB。新34文件基线显式登记4改1增，旧33基线及旧RUN不回写。下一份[固定dtype修复入口](delegation/20261007_PRODUCT_ROUND1/A_REAL3_DTYPE_EXECUTE.txt)仅授权转发后pilot03一次尝试，目前未发送/执行。分割质量、跨视与三维/CAD尚无新增完成证据。

MVS mask主线增量：[交付](../outputs/delegation/MVS_MASK_GEOMETRY/20261007T074731Z_main/HANDOFF.md)新增canonical→既有去畸变相机的离散标签核心；15个native-camera小用例及3对既有相机/合成棋盘格通过，实际尺寸1024×683→1013×675。未实现manifest consumer，未运行SAM3/OpenMVS；A的34绑定未变。可并行转发[B文件接入任务](delegation/20261007_PRODUCT_ROUND1/B_MASK_BRIDGE_IMPLEMENT.txt)与A pilot03；两份均未由主线发送或执行。真实mask质量、跨视、MVS消费及CAD仍未完成。

CAD数值主线修复：[交付](../outputs/delegation/CAD_NUMERIC_REPAIR/20261007T080927Z_main/HANDOFF.md)复现并修正远离原点时包围盒错误被放过的问题。线性容差改按源物体最长边计算，体积独立；23项纯数值测试一次通过。未运行FreeCAD/GPU，旧四面体原生证据不自动覆盖新字节；A34/B214绑定未变。真实网格处理、来源链及最终CAD往返仍待验证。

网格准备主线增量：[证据](../outputs/delegation/MESH_PREP/20261007T081724Z_main/HANDOFF.md)新增保double精度PLY读写、显式边组件选择和保拓扑简化入口。95纯数据用例通过，合成闭球528→400面后真实转换为有效单实体并完成FCStd/STEP回读；开球拒绝。实际2次pytest、4个原生子进程（含失败），原生累计5.73秒；失败与修正均留证。完整新通路只达TESTED_SYNTHETIC，不代表照片物体、GUI或精度已完成。A34/B214未变，A/B仍待用户转发结果。

本清单用于持续跟踪完整项目目标、实施依赖和验收证据。它是工作与验收清单，不是完成报告。初次登记依据现有源码、项目文档和下列运行记录；初版清单编写未运行测试、照片重建或下载软件，后续审查另有上述证据记录。

2026-10-07本地素材核实：`C:\sam3\sam3.pt` 的3450062241字节及完整SHA256与官方facebook/sam3仓库revision `3c879f39826c281e95690f02c7821c4de09afae7` 的LFS元数据匹配；项目词表与用户目录副本一致且gzip/UTF-8读取通过。证据见[LOCAL_ASSETS.json](../outputs/delegation/SAM3_LOCAL_ASSETS/20261007T050120Z_main/LOCAL_ASSETS.json)。不再以“必须登录下载权重”为当前本地运行前置项；历史AUTH_REQUIRED记录保留。现场读取专属环境包元数据仍无torch/torchvision，模型加载/CUDA/真实前向尚未验证，A-ENV仍为下一项外包任务。

## 1. 完整目标与范围

项目完成必须同时满足两条线：

1. **产品线**：输入真实静态物体的多角度照片或环绕视频，经可追溯处理得到可查看的近似三维外形，并交付 FreeCAD 可实际打开、检查有效的实体 `.FCStd` 和可重新导入的 `.step`。普通使用者能够按统一入口、依赖说明和中文使用说明完成同一通路。
2. **研究线**：保留并完成 Kozlov 数学表征、相应来源核对、适用域与数值语义、真实数据适配和分项实验；区分原文定义、本项目工程扩展、成熟 SfM/MVS 基线与实测贡献。普通重建得到 CAD 不能代替 Kozlov 理论或收益验证。

首版允许面片化 B-rep 近似实体，不要求恢复原设计的草图、尺寸约束、圆柱/圆角参数或特征树。可接受较大尺寸误差，但当前没有用户给定的物理误差上限，不能自行编造准确率或毫米精度门槛。无可靠参考长度时，可完成明确标注的任意尺度 CAD 演示。

用户当前完整目标是持续推进整个项目。旧 T01A、T02A、CAD-02A 等外包任务的单轮可写范围和交付范围，只约束对应历史任务，不能据此把本项目目标缩小为“接入完成”“监督器完成”或“最后一次实验已回报”。新工作应按依赖拆成独立、可审查、有预算和停止条件的任务。

历史任务的禁止重试规则仍属于历史证据边界：已关闭的共享相机照片实验保持 `1/1`，不得改写为未运行、覆盖失败产物或在旧任务名下续跑。后续独立实验须采用新任务和运行 ID、预先冻结输入/配置/预算，并保留与历史结果的关系。新任务不是改写旧实验结果的许可。

完整目标的既有定义见 [FINAL_GOAL_CAD_V1.txt](delegation/FINAL_GOAL_CAD_V1.txt)。工程依赖见 [PRODUCT_ROADMAP_CAD_V1.txt](delegation/PRODUCT_ROADMAP_CAD_V1.txt)，研究编号含义见 [NEXT_TASKS_T01_T06.md](delegation/NEXT_TASKS_T01_T06.md)。历史文档保留其原有时间和状态，本清单不回写旧验收。

## 2. 状态与证据规则

| 状态 | 含义 |
|---|---|
| `NOT_STARTED` | 本条完整工作或真实验收未启动；已有草案不改变此状态。 |
| `IN_PROGRESS` | 正在调查、实施或准备，仍有必要工作未完成。 |
| `IMPLEMENTED` | 指定功能已存在，尚未取得本条要求的运行或验收证据。 |
| `TESTED_SYNTHETIC` | 已通过明确标注的 fixture、合成数据或接口检查，不能据此声称真实产品成功。 |
| `TESTED_REAL_PARTIAL` | 有真实输入证据，但覆盖、几何质量或后续交付门槛尚未满足。 |
| `ACCEPTED` | 当前指定版本及数据的全部条款已有可复核证据；须同时登记验收人/过程、运行 ID 和证据路径。 |
| `BLOCKED` | 存在明确阻塞及下一解除条件；不把它渲染成完成。 |

此前任何 `PASS`、测试总数、文件存在、实现哈希或阶段“完成”，均不能代替当前完整验收。历史报告中的 561/562 项、T02A 定向测试、82/88 项工程检查或 223 项绑定检查，只能按其实际输入、版本和证明范围引用，不相加为新的项目通过数。不得为了更新文字机械重跑全部旧测试；新增或受影响功能做必要定向检查，冻结最终版本后再完成有理由的整体验收。

每项接受记录至少包含：当前状态；输入/实现/配置/工具身份；实际命令与真实退出码；适用时的 stdout/stderr 和同次测试 XML；产物路径；实际通过的门槛；失败/未知项；后续依赖。源码审阅、合成测试、真实运行、几何验证、视觉检查和研究结论分别记录。`null + reason` 优于虚构数值。

## 3. 当前证据快照与可复用组件

| 对象 | 初次登记事实 | 可支持的结论与限制 | 证据入口 |
|---|---|---|---|
| 照片与人工 mask IO | T01A/R1 已接收；有 canonical 图像、mask、身份和运行记录代码 | 可复用 JPEG/PNG 解码、EXIF 同步、hash 与逐帧失败定位；不含视频、自动分割、相机或重建 | [T01A_R1_REVIEW.md](delegation/T01A_R1_REVIEW.md)、`src/kozlov_lab/io/`、`src/kozlov_lab/segmentation/manual.py` |
| 原生 SfM 工程接入 | T02A/R1 已接收；保留 binary/text 模型、相机/rig/frame/观测映射和阶段退出 | 接口与往返证据不等于充分覆盖的真实物体重建 | [T02A_R1_REVIEW.md](delegation/T02A_R1_REVIEW.md)、`src/kozlov_lab/reconstruction/` |
| 历史真实局部 SfM | 10 图、3 个模型；主模型 4 图、312 点；pilot `NOT_MET` | 证明存在局部真实稀疏结构；不足以声明完整表面或 CAD | [T02R_REVIEW.md](delegation/T02R_REVIEW.md) |
| 旧共享相机照片实验 | `1/1` 已消耗；提取阶段原生 abort，`BACKEND_ERROR / INCONCLUSIVE`，无后续匹配/建模/导出 | 不判定共享相机假设成立或不成立；不产生新模型 | [CAD02A_SINGLE_RUN_REVIEW.md](delegation/CAD02A_SINGLE_RUN_REVIEW.md) |
| Job 内存计量与回收 | 历史诊断及 v2 双进程功能验证已登记，旧硬限额与失败保留 | 工程监督证据不能证明重建可在该限额内完成 | [CAD02A_MEM_V2/HANDOFF.md](../outputs/delegation/CAD02A_MEM_V2/20261006T143330Z_v2/HANDOFF.md) |
| 新独立运行时诊断 | 主线已做无图像 OpenBLAS 对照；`ATTEMPT.json` 记录 `PASS`、`bindings_stable=true`；`PLAN.json` 记录 `photo_inputs=0`、`inherited/threads2` 两组 | 只证明该运行时探针及对照的实际结果；不直接证明旧 abort 根因，也不等于照片提取或 SfM 成功 | [CAD_RUNTIME_DIAG/ATTEMPT.json](../outputs/delegation/CAD_RUNTIME_DIAG/20261006T150711Z_runtime/ATTEMPT.json)、[PLAN.json](../outputs/delegation/CAD_RUNTIME_DIAG/20261006T150711Z_runtime/PLAN.json) |
| 新独立照片实验 | `CAD_RUNTIME_PHOTO/20261006T151031Z_threads2` 已完成，24.39 秒，产品状态 `PARTIAL`；12/12 注册但仍为 3 个模型：01–04/312 点、05–09/238 点、10–12/127 点。Job 限额仍为 2 GiB；最大采样 current 约 639 MiB、原始 peak 约 642 MiB | 本次通过提取/匹配/建模/导出，不能声明整体恢复；05–12 后 8 图单组件目标未满足。中间模型存在 1 个 track 含同 view 重复观测，尚未通过完整结构闸门；比较报告标 `artifact_run_technically_valid=false`、假设 `INCONCLUSIVE`。旧 1/1 失败不改变 | [CAD_RUNTIME_PHOTO/ATTEMPT.json](../outputs/delegation/CAD_RUNTIME_PHOTO/20261006T151031Z_threads2/ATTEMPT.json)、[COMPARISON.json](../outputs/delegation/CAD_RUNTIME_PHOTO/20261006T151031Z_threads2/COMPARISON.json) |
| 稠密表面、CAD、视频与产品入口 | 当前源码未见对应完整通路；现有九页应用为数学实验室 | 当前不能声明 M2/M3/M4/M5 达成 | `src/kozlov_lab/reconstruction/__init__.py`、`src/kozlov_lab/io/__init__.py`、`app/pages/` |
| 数学基础 | 有 geometry、coding、compression、stereo 及历史实验/UI | 可作为后续复用基础；历史通过不自动解除本清单 RC 项的定义、语义和真实验证缺口 | [README.md](../README.md)、[phase6_gate.md](reports/phase6_gate.md)、[kozlov2025_mapping.md](kozlov2025_mapping.md) |

后续实际增量：完整66图已取得并完成新的独立SFM，63/66唯一图注册为两个模型（44与20图，04重复注册），产品状态仍为PARTIAL。主模型20条同view重复观测track对应的完整点被另存派生时移除，得到44图/2393点，原始产物不变。OpenMVS2.4.0/FreeCAD1.1.4已准备；FreeCAD转换器仅完成合成闭四面体和开放四面体两例。MVS桥接R1的点对应失败保留；后续B-R2已修复合法重复点的双射验证，4图/44图格式与几何功能接收，RAM监控缺口单列。真实表面及CAD尚无。准确证据入口统一见上述补充复核第5节。

可直接复用的工程能力包括版本化 profile、原生阶段子进程、输入与配置指纹、真实退出码、日志、模型回读、相机中心与位姿描述、严格人工 mask IO。新增 MVS/CAD 格式转换应保持原生几何与身份链，不为复用旧数学结构而丢失畸变、rig/frame 或尺度语义。

### 3.1 当前数据源的在线元数据核查

2026-10-06 只读核查固定 GitLab 项目 `8724498`、commit `0f31c229ac289a615b3ae981a5f584856bb24c47`：目录包含 `01.JPG`–`66.JPG` 共 **66 张采集照片**，另有 `Preview.jpg` 和 `README.md`；不是只有当前使用的 12 张。尚未使用的 13–66 共 54 张可作为新输入准备的来源。证据：[固定版本目录 API](https://gitlab.com/api/v4/projects/8724498/repository/tree?ref=0f31c229ac289a615b3ae981a5f584856bb24c47&recursive=true&per_page=100)。

固定版本 README 标明作者 `alansartlog`、许可 **CC BY 4.0**，并提供归属模板；派生图像需保留来源、许可及处理说明。README 与文件树没有逐图视角、尺度参考或完整底面覆盖说明。更多图像可能改善跨分量连通性是下一可检验假设，不能仅凭数量保证恢复完整；本轮没有下载这些照片、检查其像素内容或运行重建。[固定版本 README](https://gitlab.com/api/v4/projects/8724498/repository/files/README.md/raw?ref=0f31c229ac289a615b3ae981a5f584856bb24c47)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

同组当前 12 个图像集的文件树仅见照片/预览和 README，未见原始视频文件；主索引的 YouTube 教程链接不能当作已核对许可的环绕采集视频。**视频素材来源仍缺，留后续独立任务解决**；不得把现有照片合成幻灯片来充当真实视频验收。[项目组目录](https://gitlab.com/api/v4/groups/photogrammetry-test-sets/projects?simple=true&per_page=100)、[作者主索引](https://github.com/AlansCodeLog/photogrammetry-test-sets)。

更新：本小节上述“13–66尚未使用／本轮没有下载”只描述初次元数据核查。随后已准备完整66张，并以独立`CAD_FULL66/20261006T153820Z_full66`运行；62张首轮成功与4张独立传输恢复均保留记录。全图联系表展示侧面及上方环绕视角，仍不能据此声称底面完整或尺度已知。

## 4. 产品线完成项

下列复选框均表示整条验收是否完成，不能因其中某个子模块已有代码就勾选。

### PC01 — 冻结当前实施基线和新任务边界

- [ ] 状态：`IN_PROGRESS`。把当前源码、已接受模块、工具、输入和配置绑定到可追溯版本；历史快照与新实现分开。
- [ ] 每项独立执行任务记录目标、前置条件、可写范围、输入身份、图数/分辨率、线程、内存、磁盘、墙钟和运行次数上限。版本化扩展保留旧默认行为，不偷偷增大旧常量再引用旧哈希。
- [ ] 新失败记录明确阶段和下一可验证假设；范围或预算变化产生新任务记录。旧实验失败及一次运行上限不回写。

交付：当前基线清单、新任务记录和产物索引。依赖：无；贯穿其余 PC/RC 项。

### PC02 — 解除原生提取的实际运行阻塞

- [ ] 状态：`IN_PROGRESS`。结合既有崩溃/WER、原生分配路径、进程树/Job 内存与运行时线程池证据，区分已观察事实和根因假设。
- [ ] 将无图像 OpenBLAS 对照用于设计下一独立实验，不把探针成功升级为已修好照片通路。
- [ ] 若需改配置或最小实现，先做对应定向验证；新照片试验在明确预算和次数上限内真实完成，并留下完整阶段证据。不得以关闭资源监督或伪造成功状态通过。

当前增量证据：`20261006T151031Z_threads2` 已在原 2 GiB Job 限额下走完四个原生阶段，24.39 秒，内存 current/peak 约 639/642 MiB；该具体配置与输入不再在提取阶段 abort。本次结果只解除该次运行的提取阻塞，不构成旧 abort 唯一根因的证明，且不解除 PC05 的连通和结构问题。

交付：诊断裁决、必要补丁的验证、新运行结果与资源记录。依赖：PC01；成功标准包含真实提取结果，而非仅导入 NumPy/pycolmap。

### PC03 — 确认实际工具、兼容格式和最小依赖

- [ ] 状态：`IN_PROGRESS`。pycolmap4.2.1、OpenMVS2.4.0、FreeCAD1.1.4已有工具身份与探针；视频工具只读可发现，完整组合与可搬运部署未验收。继续锁定实际版本、来源、hash、命令及运行环境。
- [ ] 逐项验证需要的 CLI/API；pycolmap wheel 的存在不能当作完整 COLMAP CLI 已存在。MVS 的“支持 COLMAP”说明不能代替本项目所选版本的 rig/frame、相机和格式兼容验证。
- [ ] 工具获取和环境变更采用独立有界步骤，保留既有 T02A 环境。仅引入当前必需工具，不预装所有候选、改驱动或改系统共享依赖。
- [ ] 小型合成封闭网格可用于 FreeCAD 接口探针，明确只证明工具能力，不计为照片到 CAD 成功。

交付：最小依赖与兼容矩阵、复现命令、阶段资源建议。依赖：PC01；对应 CAD-00。

### PC04 — 锁定可恢复的真实输入、覆盖和对象定义

- [ ] 状态：`TESTED_REAL_PARTIAL`。冻结获准使用的真实素材与 hash，检查静态/刚性、纹理、模糊、反光/透明和运动情况，说明适用范围。
- [ ] 逐面登记正/侧/背/底面及孔洞、凹槽、遮挡的可见性；明确对象数量、背景处理、预处理和允许补全部位。
- [ ] 既有局部模型覆盖不足时，冻结补充视角或合适的新首版对象；不能无依据拼接独立分量，不能把旧 12 图/1024 profile 当作所有对象的采集标准。
- [ ] 记录参考长度是否存在；缺失尺度与缺失视角分别处理，避免用类别、EXIF 或想象补造尺寸。

交付：输入锁、联系表/覆盖清单、采集或选择说明、尺度初记。依赖：PC01；可与 PC02/PC03 并行准备；对应 CAD-01。

### PC05 — 获得覆盖适用、可复用的真实 SfM

- [ ] 状态：`TESTED_REAL_PARTIAL`。按新任务实际运行或有理由复用充分的既有模型，保存原生相机、rig/frame、位姿、观测、tracks 和身份映射。
- [ ] 独立核对实际退出、注册图、分量、点/track、重投影、相机配置和可见外形，说明选择哪个分量及其原因。
- [ ] 注册图数、低残差、点数多或模型可读不能单独放行稠密/CAD；需结合 PC04 覆盖、连通性和几何合理性判断。

当前增量证据：新独立照片运行已注册 12/12，但分为 01–04、05–09、10–12 三个模型，分别 312、238、127 点；不得合计点数或拼接分量声称完整物体。05–12 单组件目标未满足；中间模型 1 个 track 的同 view 重复观测需要独立定位/处理，本项仍未通过完整结构闸门，不放行将局部模型冒充最终对象。

其后的FULL66增量：63/66唯一图注册为44图/2413点和20图/1763点两模型。44图保守派生保留2393点；全图验证匹配图连通，但SfM仍分裂，匹配图连通不等于单一有效表面。允许继续检查选定分量的表面潜力，不把派生结构检查或相机方位统计升级为完整物体验收。

交付：原生模型、稀疏 PLY、图像/模型绑定、几何与覆盖审查。依赖：PC02、PC04；对应 CAD-02A。通过后才解除当前稠密通路的主要前置阻塞。

### PC06 — 去畸变与 MVS 场景转换保持几何对应

- [ ] 状态：TESTED_REAL_PARTIAL。B-R2的4图/312点与44图/2393点格式桥接和几何往返功能已接收；合法重复几何由完美双射验证，R1失败产物保留。该次RAM未监控，不能称全部资源条款合规。证据见[A/B联合复核](delegation/20261007_PRODUCT_ROUND1/REVIEW_AB_20261007.md)。
- [ ] 核对完整畸变模型、有效像素域、分辨率与 K 的变化、R/t、相机中心、坐标约定及 rig/frame 解析后的逐图姿态。
- [ ] 转换前冻结投影对应容差并实测核对；输出文件能加载不能代替几何一致性。支持不足时留下最小反例，选择明确兼容方式。

交付：可加载的 MVS 场景、去畸变相机/图片、映射与投影检查报告。依赖：PC03、PC05；对应 CAD-02B；不得以旧 Camera/Kozlov 适配作为工程转换绕行。

### PC07 — 物体分离确实作用于稠密通路

- [ ] 2026-10-07用户已选择SAM3作为检测/分割/跟踪主方案，按[SAM3接入计划](SAM3_INTEGRATION_PLAN.md)独立环境实施。选型完成不表示模型已安装、权重已可访问或真实mask已验收。
- [ ] 状态：TESTED_REAL_PARTIAL。人工mask IO已接收；SAM3已产生idx17一张真实mask，pilot04为1/3完成、第二帧RAM软阈值停止。数组转换最终离线验证84项通过，尚未实测降低提交量。去畸变consumer主线56项定向测试通过，真实44图桥接仅1张OK/43张缺失，目标mask与validity已回读；原生OpenMVS实际加载mask和排背景仍未运行。各图像域保持离散标签与明确validity，完整分离与跨视质量尚未接受。
- [ ] 复核入口增量：[MASK_REVIEW交付](../outputs/delegation/MASK_REVIEW/20261007T140600Z_main/HANDOFF.md)新增原图/mask/叠图、提示与选择来源、形态统计和可选稀疏观测诊断。最终85项离线测试通过；旧pilot02三帧失败记录真实CLI读回为NO_REVIEWABLE_MASKS，另一次真实照片上的合成几何标签形成报告。所选44图sidecar的20个重复view点整点排除并披露，3张mask只有6个可比较共享点且仅1个前景支持；不得据一致率判跨视通过。仅达到复核工具接口/既有数据读回验证，非真实SAM3输出、人审、MVS消费或分割质量验收。用法见[MASK_REVIEW_USAGE.md](MASK_REVIEW_USAGE.md)。
- [ ] 2026-10-06 源码复核：`segmentation/manual.py` 只解码、校验和变换已有人工 mask；重建记录仍是 `reconstruction_scope=scene_sparse`、`mask_usage=not_applied`。不能把已保存 mask 当作已对重建生效。优先完成目标选择（允许用户框选）→逐图目标 mask→跨视角目标一致性→去畸变域传递→稠密阶段实际排除背景的通路。
- [ ] 能力命名保持明确：现有 `coding/recognition.py` 判断有限点集的仿射关系，SfM 的特征对应关联局部观测；二者均不是照片中的语义分类或实例分割。自动类别命名不作为首版 CAD 的必要前置，但目标边界和实例一致性必须验收。
- [ ] 检查主要轮廓、真实孔洞和凹槽未被明显误删；未知/越界域不能当已知前景。
- [ ] 用所选 MVS 版本确认 mask 的实际接入；若采用深度或网格后过滤，冻结多视图支持与失败规则并标明限制。简单背景只有在实际检查后才可说明省略分离的理由。

2026-10-07主线补充：[原生深度读取与结果核验](../outputs/delegation/MVS_DEPTH_READBACK/20261007T115500Z_main/HANDOFF.md)已实现，达到TESTED_SYNTHETIC（最终145项用例通过）。固定v2.4.0源码7份绑定；DR布局、尺寸/身份/hash、已知背景与未知域、非空对照信号均有检查。尺寸缩放仅为参考映射，尚未核实本机OpenCV；即使结果SUPPORTED，也不认证实际mask加载、原生程序/缓存/配置。SAM3、B consumer、完整真实mask及原生对照仍待验，本项未完成。

2026-10-07执行入口补充：[MVS_SURFACE_RUNNER](../outputs/delegation/MVS_SURFACE_RUNNER/20261007T122600Z_main/HANDOFF.md)已接受显式去畸变mask/validity目录并绑定全scene输入，来源链和实际消费仍NOT_VERIFIED。它不读取或替代B的manifest consumer，不将文件存在当作背景被排除的证据；本轮没有运行真实MVS。本项保持未完成。

交付：mask/ROI、叠图、变换记录及后续产物中的作用证据。依赖：PC04、PC06；原图标注可提前；对应 CAD-03。稀疏点过滤不代替本项。

### PC08 — 从真实输入得到可检查的物体表面

- [ ] 状态：`NOT_STARTED`。在固定预算下完成稠密深度/融合与网格重建，优先复用已选成熟 MVS；精化仅针对粗网格的具体问题单独安排。
- [ ] 保留原始深度/稠密场景、点云、表面网格、配置、耗时/资源、缺失和伪影清单。
- [ ] 多角度对照真实照片，确认非空表面与主要可见外形一致，识别背景、漂浮片和明显错层。照片未观察区域不得算作已恢复。

执行前只读核查已保存于 [MVS_SURFACE_PRE](../outputs/delegation/MVS_SURFACE_PRE/20261007T112500Z_main/HANDOFF.md)：工具/帮助与44张图的60项绑定前后无变。已确认场景含相对图像路径、Densify默认ROI/tower行为、Reconstruct默认补洞/平滑；新执行器需独立目录与显式原始表面策略。后续MVS_DEPTH_READBACK已冻结深度相关源码并实现独立读取，但源码与本机二进制/OpenCV对应仍未验证。完整真实mask、B consumer、原生消费对照仍缺，当前没有运行稠密或mesh；本项保持NOT_STARTED。

执行核心已实现：[契约](contracts_mvs_surface_v1.md)、[用法](MVS_SURFACE_USAGE.md)、[证据](../outputs/delegation/MVS_SURFACE_RUNNER/20261007T122600Z_main/HANDOFF.md)。最终178项合成/控制流测试通过；首次171项中9败已保留，Windows目录缓存属性误判经修正。两份既有真实MVSI场景只读解析得到4图/312点、44图/2393点，48张图片尺寸/身份核对通过；两个真实微型Python进程验证正常退出及活动后监控失败回收。以上均不是OpenMVS原生执行。新入口按字节复制相对图像树，固定Densify→Reconstruct与共享预算，要求保存清理前_raw.ply；上游即使关闭选项仍有内部清理，final PLY不冒充raw。真实表面仍NOT_STARTED，未宣告PC08完成。

照片身份接线：[SURFACE_PHOTO_LINK](../outputs/delegation/SURFACE_PHOTO_LINK/20261007T153400Z_main/HANDOFF.md)为真实来源计划增加必需的photo-source记录，按完整桥接路径及scene global ID连接选定/源原生模型、frames_map、T01和数据集文件，并纳入运行前后稳定性检查。最终229项测试通过；既有44图一次只读核对成功（两个已有COLMAP模型读入，重建/推理0次）。这是当前图像身份与字节关联；点集派生、去畸变生成、scene几何历史来源、原始拍摄链和mask消费仍未认证，不将本项升级为真实稠密完成。

交付：原始点云/网格与实际三维预览。依赖：PC06、PC07。对应 CAD-04；本项通过仅代表 M2，不代表有效实体。

### PC09 — 有记录地简化、有限修补并检查拓扑

- [ ] 状态：TESTED_SYNTHETIC（受限闭合三角网格）。scripts/prepare_mesh.py已有double PLY、显式边组件选择、保拓扑简化、前后拓扑/体积及双向采样偏差证据；95纯数据用例和合成原生通路通过。开放网格拒绝，自动修补/真实网格处理尚未验收；不以Euler或有限采样代替自交与真实外形检查。
- [ ] 来源连接增量：[MESH_SOURCE_LINK](../outputs/delegation/MESH_SOURCE_LINK/20261007T150200Z_main/HANDOFF.md)使网格输入必须匹配来源记录声明的产物路径/字节数/hash。支持显式TEST_ONLY合成来源与固定surface协议的raw/final角色，网格准备前、结束时及CAD复用时重核；旧任意JSON与缺新绑定的历史prepared不自动升级。最终111项定向测试通过；同轮一次合成球528→400面准备及一次CAD交付均退出0，FCStd/STEP回读为有效封闭单实体。surface协议目前只用手写接口夹具验证；真实MVS执行、原始照片来源、mask消费与真实物体外形仍未验证，不提升PC08或真实CAD状态。
- [ ] 按 FreeCAD 实测能力冻结面数/运行上限，保存简化/修补前后网格、逐项修改和几何偏差。
- [ ] 目标边界封闭、可定向、无已知自交，且实体数量与对象定义相符；真实贯通孔不能为封闭而填死。
- [ ] 大面积缺失不能用凸包、任意 CAD primitive 或几何体代替。有限补全须有明确位置与理由，不能标作照片观测；无法成体时保留网格并记录本项未通过。

交付：前后网格、拓扑报告、修改/补全清单和关键轮廓对照。依赖：PC04、PC08；对应 CAD-05。前后网格距离仅表示处理变化，不表示实物准确性。

### PC10 — 有效 FreeCAD 实体与 STEP 往返

- [ ] 状态：TESTED_SYNTHETIC。除历史四面体外，新400面简化合成球也已由当前scripts/mesh_to_cad.py转换为有效封闭单实体；FCStd回读、STEP回读、OCCT详细检查及source_extent_v1数值比较通过。真实照片网格、GUI打开旋转/截图仍未测；不能当作产品CAD交付完成。
- [ ] CAD来源连接增量：[CAD_LINEAGE_REPAIR](../outputs/delegation/CAD_LINEAGE_REPAIR/20261007T144200Z_main/HANDOFF.md)修正报告未核来源文件、有效尺度ledger可与异源prepared/reference拼接的缺口。67项离线测试通过；既有两组合成CAD的来源边只读通过，内存报告替换的反例旧实现放过、新实现拒绝。38份历史文件未改，没有重跑FreeCAD；meshprep→surface→照片完整链仍未验证，不提升真实CAD状态。
- [ ] 实际 FreeCAD 检查几何有效、边界封闭、预期实体数正确、体积有限且为正；不能只看类型名或文件存在。
- [ ] 在新文档重新导入 STEP，重复上述检查，并核对包围盒/体积在事前说明的转换数值容差内一致。
- [ ] 实际打开、旋转和缩放查看，保存截图；再与照片检查主要轮廓、孔洞和凹槽。清楚标注面片化近似实体及不具备原设计特征树的事实。

交付：`model.FCStd`、`model.step`、原生与重导入检查、截图和模型来源链。依赖：PC03、PC09、PC11 中的单位/变换登记；对应 CAD-06，通过后才可报告 M3。

### PC11 — 尺度、单位、偏差与未知项分别验收

- [ ] 状态：`TESTED_SYNTHETIC`（定尺度子模块）。新增scale_mesh.py及CAD前置核验：网格hash/最终顶点端点/证据绑定、单位换算、正逆变换、局部数值与逐边长度检查；最终117项测试及400面合成网格50倍缩放→FCStd/STEP原生往返通过。真实物体参考、完整上游链及最终CAD端点复测仍未验收。[用法](MESH_SCALE_USAGE.md)、[证据](../outputs/delegation/MESH_SCALE/20261007T090514Z_main/HANDOFF.md)。
- [ ] 有未参与缩放/拟合的独立实测尺寸时才报告相应误差。只有定尺度参考时写“按参考定尺度，精度未验证”。
- [ ] 无可靠参考长度时明确“任意尺度演示”；CAD 文件的 mm 单位约定不能充当真实毫米证据。
- [ ] 分开记录实体是否有效、外形是否合理、尺度是否已知和物理精度是否评估。可接受较大误差不允许编造测量结果。

交付：尺度/单位与变换清单、误差证据或未知原因。依赖：PC04，贯穿 PC06–PC10；未知真实尺度不阻止任意尺度首版。

### PC12 — 真实视频入口复用已验证照片通路

- [ ] 状态：`TESTED_SYNTHETIC`（入口子模块）。新增本地视频到既有T01照片通路，100项定向测试与两次合成VFR原生运行通过；保留真实PTS/时基/原始解码序号、方向与PNG身份，拒重复画面并提供显式清晰度代理阈值。真实视频、视角覆盖和下游CAD仍未验收。[用法](VIDEO_INPUT_USAGE.md)、[证据](../outputs/delegation/VIDEO_INPUT/20261007T084217Z_main/HANDOFF.md)。
- [ ] 保留视频 hash、实际帧号/时间戳和派生图片映射；变帧率不能以帧号/名义 fps 冒充精确时间戳。
- [ ] 控制最终图数并复用照片通路，不另造第二套重建；至少一个真实视频产生通过 PC08–PC11 的 CAD 文件。

交付：视频与抽帧清单、选择/剔除原因、入口与真实视频到 CAD 证据。依赖：照片 M3；对应 VID-01/T01B。动态物体及任意视频普适恢复不属于首版保证。

### PC13 — 统一入口、进度、失败定位与恢复

- [ ] 状态：`IN_PROGRESS`（离线 PLY 查看器与已准备网格到CAD交付入口已实现；完整照片/视频重建入口仍未实现）。完成照片目录/视频输入、输出目录、配置、进度、明确错误与产物索引的一致 CLI；复用已有 IO/后端模块。
- [ ] 缓存/恢复按输入、配置、工具及实现 hash 判断兼容；失败残留不能当作成功结果复用。
- [ ] 覆盖无效输入、后端失败、预算超限和不满足几何条件时的用户可理解状态；不得显示成功或生成伪模型。
- [ ] 提供最小模型查看方式；如接现有应用只加必要导入/查看入口，不把大型 GUI 当完成前置。

新增 `scripts/export_3d_viewer.py` 与 `docs/VIEWER_USAGE.md`：真实 2393 点 RGB 点云和合成 400 面网格已导出为单文件 HTML，原坐标与来源/单位声明分别保留。最终 64 项输入/导出测试及 JS 控制测试替身通过；自动化浏览器因 file 协议政策拒绝访问，实际 WebGL/交互仍 `NOT_RUN`，所以本项不勾选。证据：`outputs/delegation/VIEWER/20261007T094322Z_main/`。不将此预览计作真实稠密表面、CAD 文件导入或全通路完成。

新增 `scripts/build_cad_delivery.py` 与 `docs/CAD_DELIVERY_USAGE.md`：以成功且身份/实现仍匹配的mesh-prep目录为输入，统一可选参考定尺度、原生CAD/STEP及PLY预览。32项定向检查通过；未知尺度/参考尺度两条合成路径首次执行均成功，两次真实CAD转换得到有效闭合400面单实体并完成FCStd/STEP往返；Job回收确认。只读verify只核对绑定与既有报告，不声称重新原生验收或实现失败续跑。证据：`outputs/delegation/CAD_DELIVERY/20261007T100714Z_main/`。该后半程为`TESTED_SYNTHETIC`，真实照片/视频统一调度、GUI、物理精度及完整缓存恢复仍未完成。

新增输入前端 `scripts/prepare_object.py`：[用法](OBJECT_INPUT_USAGE.md)、[契约](contracts_object_input_v1.md)、[证据](../outputs/delegation/OBJECT_INPUT/20261007T131000Z_main/HANDOFF.md)。照片目录显式生成输入清单，原始JPEG/PNG经一次EXIF与有记录的1024缩放后交既有T01；视频包装调用原入口，已有成功视频可只读接纳。69项离线测试通过；两张旋转合成大图和三张既有许可原始照片两次真实CLI准备成功，3张4272×2848缩至1024×683后的像素与旧准备结果一致；旧六帧合成视频只读接纳与inspect通过，没有重解码。原图变换、T01身份、像素、视频索引/PTS和hash绑定逐项核对。该输入子通路达到TESTED_REAL_PARTIAL；新video包装的原生成功调用、真实物体视频及后续SAM3/SfM/MVS/CAD串联仍未验证，PC13不勾选完成。

交付：可复现入口、结果目录结构、状态说明和最小使用流程。依赖：照片通路后先接照片版，PC12 后接视频版；对应 PIPE-01。

### PC14 — 最小可复现安装与版本部署验收

- [ ] 状态：`NOT_STARTED`。核对 `pyproject.toml`、外部二进制和隔离环境之间的实际依赖说明，不能把当前机器偶然可用当作部署完成。
- [ ] 在独立干净环境/快照完成必要安装或可搬运工具配置检查，证明 CLI 和必需工具能被找到并调用；保存版本、命令和退出码。
- [ ] 必须复用的外部环境或手工步骤写清；未验证平台不作兼容承诺。保留已有工作环境和历史证据。

交付：最小依赖清单、安装/配置步骤、干净环境验证记录。依赖：PC03、PC13 的稳定接口。

### PC15 — 冻结版本后的独立真实整体验收

- [ ] 状态：`NOT_STARTED`。冻结版本后至少覆盖一个真实照片案例和一个真实视频案例；至少一组素材未用于前期反复调参，条件允许时采用用户自己的物体。
- [ ] 两条实际通路均通过输入追溯、资源预算、表面检查、CAD/STEP 往返、实际查看及尺度披露。检查失败输入不能被显示成成功。
- [ ] 必要回归检查与当前实现版本绑定；既有定向 PASS、合成闭合网格、同一旧案例反复成功都不能替代本项。
- [ ] 若最终只有同一案例，只能报告单例演示，本项仍未达到完整双入口验收；不据此宣布所有材质、所有物体或任意视频可用。

交付：冻结版本号/身份、验收案例锁、完整证据和独立核查记录。依赖：PC01–PC14。

### PC16 — 产品交付说明和最终状态一致

- [ ] 状态：`NOT_STARTED`。交付可直接找到的 CAD 成果、原始/清理网格、复现命令、中文使用说明、错误处理和已知限制。
- [ ] 用当前验收证据更新产品状态说明与入口索引，区分历史阶段报告和当前交付，避免 README 的旧测试数/旧状态造成完整成功印象。
- [ ] 分别报告 M1 稀疏、M2 表面、M3 照片 CAD、M4 视频 CAD、M5 普通用户入口；记录适用范围和未测项，不用一个总 PASS 覆盖差异。

交付：可审查交付包与当前状态说明。依赖：PC15；产品线至此方可独立验收。

## 5. Kozlov 研究线完成项

研究线不阻塞基础照片 CAD 的成熟工程通路，但“整个项目完成”不能省略以下工作。定义未决时保留 `UNKNOWN/UNSUPPORTED`；需要限制适用域时写清边界，不能把尚未解决的一般问题或未实施功能改写为已完成。

### RC01 — 原文、定义、命题与实现逐条绑定

- [ ] 状态：`IN_PROGRESS`。核对已取得原文、出处页码、定义/命题的假设、目标与算法步骤，建立来源→规格→函数→测试→结论映射。
- [ ] 分开 1996 三维/立体视觉、2002 压缩/ostov、2009 图像码、2025 形象定义；名称近似或词干命中不等于定义相同。
- [ ] 明确二维/平行投影/中心投影/三维扩展的适用域，区分 `SOURCE-EXACT`、数值策略、`BASELINE`、`EXPERIMENTAL` 和本项目新增定义。

交付：已核对来源映射与待解问题清单。依据：[kozlov2025_mapping.md](kozlov2025_mapping.md)、[open_questions.md](open_questions.md)、[stereo_kozlov.md](specs/stereo_kozlov.md)。

### RC02 — 核查现有三维代码与等价判据的真实语义

- [ ] 状态：`TESTED_REAL_PARTIAL`（完整小规模精确代码及真实来源子集对照）。旧`kozlov_body_code`保持数值摘要接口；新`official/indexed_code3d.py`保存1–12点全部合法四元组索引比，提供给定映射逐条核验、最多8点有界搜索、完整性/预算未知与精确定理域检查。50项测试和固定6个真实SfM点的受控正/负对照通过；不代表2393点整云编码或真实对应恢复。[契约](contracts_indexed_code3d_v1.md)、[用法](INDEXED_CODE3D_USAGE.md)、[证据](../outputs/delegation/CODE3D_INDEXED/20261007T112500Z_main/HANDOFF.md)。
- [x] `is_a_equivalent`按行比较的无序集合错误已修为完美匹配的仿射双射数值判定，返回对应见证并处理弃权；19项新增定向用例通过，另2项旧错误断言更正后定向通过。证据见[CORRECTIONS.md](../outputs/delegation/RESEARCH_CORRESPONDENCE/20261006T151710Z/CORRECTIONS.md)，不等于正式带索引代码或一般定理完成。
- [x] 当前UI、实验打印、函数文档中“is_a_equivalent使用带索引代码”已改正；摘要、浮点几何oracle、新精确索引代码分开。历史phase6_gate原字节保留并由新用法与既有勘误连接；本轮没有重新运行UI或历史实验。
- [ ] 修复前后的定向复现和受影响实验另存新证据，不改历史结果。精确新模块有独立证据，旧数值对应有此前定向证据；真实对应、全部受影响历史实验及UI运行仍未重验，不自动沿用旧结论。

交付：明确语义、最小反例、定向修复/测试及受影响结论清单。依据：上述源码、`app/pages/09_Stereo_Lab.py`、[phase6_gate.md](reports/phase6_gate.md)。依赖：RC01。

### RC03 — AlgT' 与 AlgSo 分别完成抽取、实现和验证

- [ ] 状态：`IN_PROGRESS`（来源抽取与纠错完成；实现及算法 Gate 未完成）。[S-1996算法规格](specs/stereo_algorithms_1996.md)已目视核对PDF12–19页/印刷331–338页，纠正给定标记、a′-不同、跨视共面谓词及不完整投影含义；六个精确合成解释检查通过。交线逐点映射和输出退化需要显式SOURCE-DERIVED补足；[证据](../outputs/delegation/STEREO_SOURCE/20261007T140000Z_main/HANDOFF.md)。下一份[C数学实现任务](delegation/20261007_PRODUCT_ROUND1/C_LABELLED_STEREO_IMPLEMENT.txt)已准备，未发送/执行；它不代替未知对应、真实照片研究或A/B。
- [ ] 以合适的小型精确/真值用例验证，实现与来源建立逐步对应；不把针孔 DLT 或重投影搜索命名为作者算法。
- [ ] 明确算法所得等价类与实际度量深度的差别、退化与歧义限制；理论证明、有限测试和数值实现分别陈述。

交付：两算法独立规格、实现、必要测试与算法 Gate 报告。依赖：RC01、相关 RC02 语义；对应历史 Phase 6 最大未完成项。

### RC04 — 2025 最小面积外接椭圆的完整数值契约

- [ ] 状态：`TESTED_REAL_PARTIAL`（数值有限点域及真实 SfM 特征点单例已测试，非 RC04 全项完成）。新增 `kozlov2025/ellipse.py` 与[冻结契约](contracts_ellipse_v1.md)：1–4096 个有限二维 binary64 兼容点、Khachiyan 浮点候选、Fraction 精确包含与面积比证书、点/线退化、迭代/work 预算及明确失败状态。近退化可以 `NUMERICAL_FAILURE` 结束。
- [x] [pure02.xml](../outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/pure02.xml) 的 45 例全部通过（0 failure、0 error、0 skipped），覆盖包含性、已知解中心/面积、仿射中心/形状/面积协变、退化、预算耗尽、非法输入、不可变结果及独立证书重算。有限测试不代替一般定理证明。
- [ ] `APPROXIMATE_CERTIFIED` 只保证实际 binary64 输入的 `area(output)/area(optimum) ≤ 1 + area_tolerance` 且精确包含；不是精确 Kozlov 中心，不提供中心误差限、长轴尺寸精度或物理测量精度，不自动放行严格 fits、覆盖或高层形象结论。
- [x] 真实 SfM 特征像素 [demo01](../outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/DEMO_RESULT.json) 唯一一次运行退出码 0；image 4 `images/000003.png` 的全部 3571 点仅精确去重为 3179 点，395 次更新返回 `APPROXIMATE_CERTIFIED`。独立 Fraction 重算包含全部原始点，面积比上界 `1.0049380785009892 ≤ 1.01`，输入来源未改。
- [ ] 真实单例只对上述特征点集成立，不代表物体 mask、图像边界、背景移除或完整研究通路；真实物体中心/尺寸精度仍未评估。

交付：椭圆规格/实现、真值与数值验证及[使用说明](ELLIPSE_USAGE.md)。证据目录：`outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/`。依赖：RC01；对应 T04B。其余研究项和项目总状态保持未完成。

### RC05 — 环境、覆盖及 2025 ostov

- [ ] 状态：`NOT_STARTED`。实现经核对的图像/部分/子图像、环境 S/S'/S⋆、独立群、覆盖、块、覆盖尺寸与 2025 m-ostov。
- [ ] 对有限枚举明确规模/预算与完备性；部分搜索记录 `complete=false`，不能把候选集当完整结果。
- [ ] 2002 r-ostov 与 2025 m-ostov 按来源/语义隔离，保留旧 API；基础点集可复用但不能据此将高层结构标为实现。

交付：来源一致的类型/算法、真值用例、边界与规模限制。依赖：RC01、RC04。

### RC06 — futlyar、containment 与严格 fits

- [ ] 状态：`IN_PROGRESS`（2026-10-14，已有可独立验证的严格有限域）。primary、generalized、零误差、共线非零与四中心非共线強稳定fits已实现；一般求解及m=1、重复中心等未覆盖边界仍保留，见本页最新增量。
- [ ] 严格 fits 保留最优互位与稳定性要求；不得删除条件以取得成功。无法判定时返回未知及原因，近似优化采用不同名称。
- [ ] 数值近似及有限测试不升级为严格 containment/fits 成功，未知不得向下游传播为完整群/粒度/etalons 的理论结论。

交付：可实施的严格适用域/算法、边界裁决、近似与严格结果隔离及验证。依赖：RC01、RC04、RC05。剩余问题：[open_questions.md](open_questions.md) Q4–Q6；本轮仅解决上述明确有限域，不宣称一般问题已解决。

### RC07 — 三层形象、群、粒度、etalons 与特征

- [ ] 状态：`IN_PROGRESS`（2026-10-14，有限兼容自动搜索、tight上界精确M、完整群充分证书及给定超群否证已有实现和独立核验）。一般m-compatible存在性、未达上界的最大粒度、一般完整群、全部etalons、闭形象、谱、重要部分、特征/标记关系仍未完成。
- [ ] etalon 的严格结果为满足条件的集合；单个工程代表另存 `selected_prototype`。核对 2002 与 2025 granularity 的语义，不提前合并。
- [ ] 定义输出 schema、依赖未知的传播、搜索完备性与资源边界。有限样例频率不能替代理论定义或类别结论。

交付：逐定义实现、依赖状态、测试与结构输出。依赖：RC04–RC06；对应 `kozlov2025_mapping.md` 的剩余 `MISSING` 项。

### RC08 — 真实原生数据到研究结构的无损语义适配

- [ ] 状态：`TESTED_REAL_PARTIAL`（已选分量、已去畸变针孔模型适配）。新增`reconstruction/to_kozlov.py`及[冻结契约](contracts_research_native_v1.md)：原生ID/全features/tracks/rig/frame与point.error保留，PointSet3D精确集合视图另带多对一映射，Camera保持原K/位姿/像素约定，世界变换为identity、尺度unknown。155项纯数据测试、原生真值fixture、现有44图2393点/57430features/9904观测对照及实际CLI通过；原生投影差最大6.82e-13像素。旧20点筛除保持披露，T01关联/原始畸变射线在同次证据核对；未运行研究编码或把几何保真当收益。[用法](RESEARCH_NATIVE_USAGE.md)、[证据](../outputs/delegation/T03/20261007T102949Z_main/HANDOFF.md)。
- [ ] 在已知真值 fixture 和真实原生模型上分别核对坐标、位姿、对应及投影往返；明确非有限值、重复点和大点云规模处理。
- [ ] 仿射等价检查仅作附加验证，不能代替数值/身份保持；不强迫既有小规模 oracle 处理整个真实点云。

交付：T03 适配器、fixture/真实分别验收、变换与身份映射。依赖：PC05、PC06、RC02；工程 MVS 转换和本项是不同任务。

### RC09 — 三维结构定义与跨视一致性

- [ ] 状态：`NOT_STARTED`。明确本项目三维扩展、3D skeleton 构造和跨视对应定义；无 skeleton 时不能以均值/体素中心伪装输出。
- [ ] 使用相容相机/观测投影，记录可见性、遮挡/未知、对应歧义和重投影误差。零残差不等于正确或唯一对应。
- [ ] 实现与验证结构一致性时明确真值、观测和推断的区别；继承 `RecoveryOutcome` 等状态时先核对其语义和适用规模。

交付：三维扩展规格、跨视结构算法与分项检查。依赖：RC05–RC08；对应 T05/T06-V。

### RC10 — 数学验证与历史实验的独立复现

- [ ] 状态：`NOT_STARTED`（当前完整验收）。对新增/修正定义完成必要性质、反例、退化和独立 oracle 检查；适用域与样本跳过原因必须明确。
- [ ] 在代码/配置/种子/工具绑定的独立输出目录复现要引用的历史核心结论，保留旧产物；仅 `verify-only` 或 hash 一致不算重算。
- [ ] 复核当前三维对应规模边界：已有 n≤8 oracle 对照不能声明大规模正确；按需冻结更大规模可证明/可验证范围。
- [ ] 区分定义、证明、数值实现、有限实验与未决问题，不为统一绿色状态隐去弃权或失败。

交付：当前可引用实验集、独立复现记录、适用域与差异解释。依赖：对应 RC 实现稳定；不要求机械重跑与变化无关的全部旧任务。

### RC11 — 真实照片/视频与独立对照实验

- [ ] 状态：`NOT_STARTED`。分别冻结合成几何、静态照片、视频、同类多个实例所需数据、指标公式、阈值与资源；不以 A–C 的重建表现证明 D 的类别结构性质。
- [ ] 如主张 Kozlov 方法带来收益，使用相同输入、相容语义和匹配预算的普通 SfM/MVS 基线、必要消融与独立对照，报告成本、失败/未知、适用范围和实际贡献。
- [ ] 对真实无真值数据不编造完整度/物理精度；负结果与未确认收益同样保留。工程成功、研究功能可运行、研究收益成立分别判定。

交付：分开的 metrics/report、原始结果、对照与结论边界。依赖：PC 的相关真实通路、RC08–RC10；对应 T06-E。

### RC12 — 当前研究报告、演示与证据闭合

- [ ] 状态：`NOT_STARTED`。完成反映当前实现和实验的研究报告/演示，逐条标注原文理论、成熟基线、本项目方法、实测观察与解释。
- [ ] 清理当前展示中的过度完成措辞和实现/报告语义不一致；保留历史报告与勘误关系，未决数学项不能静默删除。
- [ ] 每个主要研究结论能追溯到当前有效来源、规格、实现和运行证据；未测/受限部分明确列出。若研究范围需改变，应作显式范围裁决，不能以产品完成自动缩减研究目标。

交付：最终研究报告、演示、结论—证据索引和明确限制。依赖：RC01–RC11；研究线至此方可独立验收。

## 6. 推进顺序和结案规则

当前产品关键路径为 PC02 → PC05 → PC06/PC07 → PC08 → PC09 → PC10 → PC12 → PC13–PC16。PC01 持续执行；PC03/PC04 可并行准备；PC11 的尺度登记贯穿几何处理。不得在上游输入不满足时启动无界下游计算。

研究可并行进行 RC01/RC02/RC03 的来源与语义工作，再按 RC04–RC07 的数学依赖推进；RC08 接已验证原生几何，RC09–RC12 接已定义结构和相应真实数据。工程与研究可独立取得阶段结果，但都需要自己的完成证据。

每次更新本清单只勾选有当前证据的条目，并登记新增运行 ID、输入/实现版本、通过门槛、未测项和证据路径。计划已写、接口已定义、脚本已实现、探针已通过、真实通路成功和结论通过是不同状态。

项目总状态初次登记为 **未完成**。只有产品线 PC01–PC16 与研究线 RC01–RC12 的必要工作均按各自范围验收，且没有以旧外包范围、旧 PASS、合成替代、任意几何体替代或未经证明的研究结论掩盖缺口，才能报告整个项目完成。
