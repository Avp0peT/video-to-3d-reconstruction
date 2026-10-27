# 独立精确研究页面

2026-10-18当前增量：追加第18模式“2019椭圆代码未知对应等价”，其构造与下载证书独立核验采用两个按钮，详见文末。旧17模式及各自历史证据保留。新增UI测试尚待主线统一执行。

Streamlit 多页应用新增 `12_Exact_Research.py`，中文名称“精确点集研究（独立数学侧支）”。仅用户点击“计算并独立核验证书”时运行有限 CPU 数学构造；普通 rerun 不重复计算。输入、功能或预算改变后旧结果失效，下载隐藏；非法输入会清除旧结果。页面与照片三维、SAM3、原生工具及 CAD 产线隔离，不从照片偷偷提取或抽样点集。

第一个入口接受完整 1–7 个有唯一整数 label 的二维点，schema 为 `cover-center-family-input-v1`，仅含 schema 和 points。每点为 `{"label":0,"xy":[0,0]}`。调用 `build_cover_center_family`，再对下载载荷精确往返后的对象调用 `verify_cover_center_family`。枚举最多 877 个候选，工作预算最多 2,000,000，时间最多 30 秒。所有已注册覆盖与 UNKNOWN 原样保留；独立核验 True 证明所声明部分/完整结果的证据有效，不保证全部覆盖为 EXACT。超过 7 点直接拒绝，不采样。默认人工正方形示例当前预期为 14 个覆盖、0 个 UNKNOWN，完整精确完成；这是该有限示例的结果，不是所有无界输入完成。

第二个入口沿用 `scripts/verify_generalized_futlyar.py` 的 `generalized-futlyar-input-v1` schema、`input_values`、精确 rational/parser/encode 和 certificate_from_json：完整 points、完整 centers 和正有理 radius。只验证给定中心配置，不搜索仿射环境。调用 `verify_generalized_futlyar_2d`；存在证书时，将序列化载荷重读后调用 `verify_generalized_futlyar_certificate`。FALSE/UNKNOWN/UNSUPPORTED 保留状态、原因和见证；没有证书时显示 NOT_AVAILABLE。工作预算最多 262,144。默认人工例包含 1/2 坐标，预期 TRUE 和独立核验 TRUE。

坐标仅接受整数或 `{"numerator":"1","denominator":"2"}`，要求约分、正分母与现有 bounded parser 规范。拒绝 float、重复 JSON 键与非有限 JSON 常量，文本最多 4 MiB。结果带输入摘要；两个入口均可下载完整结果及证书/见证。

覆盖族下载使用既有 `RUN20261013_cover_center_family/run_once.py` 的精确 tagged serializer：Fraction 编码为 fraction，tuple 编码为 tuple，字典编码为 mapping 键值对，从而保留整数键与元组。原运行脚本含顶层执行，页面复用其同一函数实现，不导入或执行该脚本。Futlyar 下载继续使用已有 CLI JSON 证书格式，可按相同输入字节绑定通过 CLI 独立核验。

物理尺度始终 unknown，照片质量未评估。数学 TRUE、部分证据核验 True、完整有限数学环境、照片几何重建和 CAD 完成是不同结论；页面不把数学成功标为 CAD 完成。

证据在 `outputs/delegation/GENERAL_PIPELINE/RUN20261013_exact_research_ui/`。使用现有 kozlov Streamlit 环境的 AppTest，两批共不到 10 秒；第二批 6 项通过，包括两个 UI 按钮路径、rerun 不重算、旧结果失效、预算 UNKNOWN、精确往返、完整点输入边界、无模型/native/network 调用。第一批 5 通过/1 失败：测试错误地以零半径作合法反例；既有 API 正确拒绝零半径，测试改用正半径 4 后通过，原失败输出保留。未启动浏览器或真实原生流程。


## 新增有限精确入口

原有覆盖族和给定中心半径 Futlyar 功能、控件键与载荷保持兼容；新增四个可实际操作入口，仍只在点击时运行。输入或预算改变后旧结果与下载失效。算法状态和序列化后的独立核验分开显示；FALSE、UNKNOWN、PARTIAL_UNKNOWN、UNSUPPORTED 不替换为成功。

- 单个完整点集精确中心：`exact-center-input-v1`，完整 points。调用统一 `solve_exact_center`，展示精确中心/长轴符号及数域，再从无损 JSON 往返对象调用 `assess_exact_center_result`。支持完整 1–7 点；点/线是规范点/线段中心，不能称非退化椭圆唯一。默认人工四点例的最优参数为真三次实代数数。
- 精确仿射子像：`affine-subimage-input-v1`，完整 source 和 target。source 最多 12 点、target 最多 32 点。调用有限仿射锚枚举，存在证书时重建 `AffineSubimageCertificate` 并独立验证完整注入与可逆仿射式。不是几何包围或通用 fits。
- 有限环境与标签互斥 S+：`finite-environment-input-v1`，完整 points 和显式布尔 include_empty。最多 7 点；全部非空部分、按标签互斥的 S+ 组独立核验。S* 仅给出基数，不枚举幂集。标签互斥不等于闭凸包互斥、独立覆盖或兼容。三点示例有 7 个非空部分、14 个不含空组的 S+ 记录；七点为 127 部分与 4139 记录。
- 给定覆盖 primary 验证：`primary-futlyar-input-v1`，完整 points 与 parts 标签列表。仅给定覆盖，不搜索覆盖；调用 `algebraic_primary_futlyar.assess_algebraic_primary_futlyar`，仅 TRUE 时从载荷调用 `assess_algebraic_primary_certificate`。默认两单点例为 TRUE。完整最多 7 点，可核验最多一个四接触实代数根，包括真实三次中心；严格边界、域外与预算未完成原样展示。FALSE 保留算法反例，TRUE-only 独立证书入口不接收 FALSE，显示 NOT_AVAILABLE 并解释。

新入口采用 `exact_center.encode_exact/decode_exact` 的 `$exact` rational/tuple/integer-map 标签，下载 envelope 为 `exact-research-tagged-v1`，encoding 为 `exact-center-tags-v1`。它与原有 family 的 fraction/tuple/mapping 标签是两个显式格式，不能混用解码器；原 family 下载继续使用原标签。所有新入口都保留 Fraction、元组及整数标签身份，拒绝 JSON 默认把整数键偷偷变成字符串。

较大的环境 S+ 下载使用 `exact-research-chunked-v1`：document 保存其余新标签 envelope，group_chunks 每块最多 128 条，含连续 offset 与精确 tuple groups。页面的 `decode_new_payload` 对每个块调用新的有界解码器，再按偏移恢复完整证书；最多 40 块/4140 记录。这样保留解码器每树 20000 节点上限，不在算法模块放宽门。普通新入口同一 helper 直接使用新解码器。往返后再独立核验全部分组。

新入口预算最多 2000000 work/30s；独立核验使用构造余下时间/工作。同一 UI 仍不导入模型或执行原生流程、shell、网络；物理尺度 unknown、照片质量未评估。单块数学验证、完整环境枚举、整图几何与 CAD 是不同证据层级。

新增验收证据在 `RUN20261013_exact_research_ui_extensions`。最终 UI 测试 17 项通过，包含原两模式、四个新增 AppTest 实际按钮路径、下载/rerun/陈旧结果、预算未知、独立核验、布尔/重复键拒绝，以及七点环境分块无损恢复。历史 RUN 原样保留。


## 最后接线：实代数 primary 与零中心误差搜索

新增“零中心误差 fits 搜索”入口，控件键为 `exact_fit_search_input`、`fit_search_work`、`fit_search_seconds`。JSON schema 为 `zero-error-fit-search-input-v1`，完整字段为 schema、reference、reference_parts、source、source_parts。两侧点集各完整最多 7 点；各 parts 为标签组。用户不提供 M/t，有限搜索自行枚举对应中心锚并构造候选可逆仿射变换。

调用 `search_zero_error_fits`，仅 TRUE 时对无损标签 JSON 往返后的结果调用 `assess_zero_error_fit_search_result`。独立核验只检查充分见证，不重新搜索。预算与时间剩余传递；下载沿用新 tagged envelope。结果分别显示中心秩、尝试候选、算法状态、独立核验及作用域。

TRUE 表示找到且可核验的零中心误差 fits 充分证据。`NO_SUPPORTED_WITNESS` 仅表示本有限支持域没有找到见证，不证明通用 fits 不存在；秩 1 横向扩展仍受限，`general_fits_decided` 始终 false。界面明确保留此提示，不声称一般 fits 已解决。UNKNOWN/UNSUPPORTED 同样保留，不交给仅 TRUE 的 verifier 后误显示 INVALID。

primary 的完整五点例 `(0,0),(1,0),(0,1),(2,3/2),(100,100)`，parts `[[0,1,2,3],[4]]`，现在实际得到 TRUE/VERIFIED，四点块中心使用三次实代数证书。原两单点默认例和既有功能/键保持兼容。数学 FALSE 不调用 TRUE-only verifier，显示 NOT_AVAILABLE 与反例尚未另核验说明。

最终接线证据位于 `RUN20261013_exact_research_ui_final_connections`。21 项 UI 测试通过（6.09 秒），AppTest 实际操作真实三次 primary、秩 2 自动仿射正例、秩 1 未找到及不宣称通用不存在、点击才计算、rerun 不重算、输入失效与下载隐藏。只使用纯 CPU 数学与 AppTest，没有原生/GPU/模型/网络调用；算法模块未修改。


## 共线中心严格 fits

新增第八个入口“共线中心严格 fits”，保留先前七个功能与控件键。输入 `collinear-fits-input-v1` 包含完整 reference、reference_parts、source、source_parts，与零误差入口的字段形式相同；每侧完整最多 7 点，用户不给定矩阵/平移。控件键 `exact_collinear_fits_input`、`collinear_fits_work`、`collinear_fits_seconds`。

调用 `collinear_fits` 构造给定分划下的共线中心全局最优稳定互位，并在固定横向仿射延拓后核验所有完整块进入 reference primary 的严格截断域。仅 `COLLINEAR_FITS_SUFFICIENT_WITNESS` 时，对新标签无损下载往返后的完整证书调用 `assess_collinear_fits_certificate`。算法结果和独立核验分别显示；剩余时间/工作传给核验。

界面展示精确 `error_squared`，保留 Fraction，允许非零误差。默认 reference 三单点 x=0,1,3 与 source 三单点 x=0,1,2 的最大中心误差平方为 1/16，实际充分证据和独立核验成功。外部 source 标签是 10/11/12；稳定互位的对应标签使用块索引 0/1/2，下载中不混淆这两类身份。

该入口只覆盖共线中心、给定分划和固定横向延拓域。成功是有限充分证据，不能声称通用 fits 已解决。`NO_SUPPORTED_WITNESS` 表示固定延拓未得到完整严格包含见证，明确不证明通用 fits 不存在；UNKNOWN/UNSUPPORTED 保留，不调用仅成功证书的 verifier。源分划无需闭凸包独立，但完整点逐一验证。

证据位于 `RUN20261013_exact_research_ui_collinear`。24 项 UI 测试通过（5.55 秒），包括新增 AppTest 非零误差实际按钮、1/16 显示与完整证书下载、工作预算 0 UNKNOWN、点击才计算/rerun 不重算、输入改变下载失效以及大块严格包含失败的非通用不存在说明。保留七模式既有测试与证据。未启动 GUI server、GPU、模型、原生或网络；算法模块未修改。

## 四中心非共线严格 fits 与自动群搜索

新增两个入口，原八个入口保留。页面仍仅点击按钮时执行；输入、功能或预算变化会隐藏旧下载，普通 rerun 使用缓存。

“四中心非共线严格 fits”使用 `four-center-fits-input-v1`，完整字段 schema、reference、reference_parts、source、source_parts。每侧完整最多七点、各四个非空块；用户不提供变换。调用 `four_center_fits`，只在 `FOUR_CENTER_FITS_SUFFICIENT_WITNESS` 时对无损下载回读的完整证书调用 `assess_four_center_fits_certificate`。默认 reference 四单点 `(0,0),(3,0),(0,2),(4,2)` 与 source 四单点正方形的精确最大中心误差平方为 1/16，实际证书独立核验 VERIFIED。展示非零 Fraction，不转换浮点；证书保留全部24排列、全点严格包含与共同逐点位移稳定半径。`NO_SUPPORTED_WITNESS` 仅说明该充分域没有见证，不能推出原论文局部 fits 不存在。控件键 `exact_four_center_fits_input`、`four_center_fits_work`、`four_center_fits_seconds`。

“自动群兼容、粒度与完整群”使用 `finite-compatibility-search-input-v1`，仅 schema、environment、group；environment 是完整最多七点，group 为标签互斥的非空图像标签组。用户不给手工分划、reference 或矩阵。调用 `search_finite_compatibility`；存在证书时先无损回读完整搜索结果（含所有已访问候选及尝试），再调用 `assess_finite_compatibility_certificate`。精确粒度且兼容证书 VERIFIED 时，继续 `certify_full_group`；存在完整群证书时再次无损回读并调用 `assess_full_group_certificate`。搜索、兼容核验、完整群构造和独立核验统一累计消耗剩余 work/时间。数学搜索状态、兼容独立核验、完整群状态和完整群独立核验分别展示。LOWER_BOUND 只表示下界，UNKNOWN 或支持域耗尽不表示一般不兼容。完整群这里只由精确粒度和未用标签数界证明，不自动宣布 closed/etalon。

默认完整环境为 `(0,0),(1,0),(3,0),(0,10),(1,10),(2,10),(99,99)`，群 `[[0,1,2],[3,4,5]]`。完整保留第6标签，不能用群成员并集替代环境。该例自动构造非零中心误差充分证据，精确粒度 M=3；只剩1个未用标签，小于M，完整群状态 FULL_GROUP_CERTIFIED，两证书独立核验均 VERIFIED。控件键 `exact_compatibility_input`、`compatibility_work`、`compatibility_seconds`。

自动搜索的大候选账本若超过既有精确解码器每树20000节点门，使用显式 `exact-research-proof-chunked-v1` 传输。document 的 piece 类型为 exact / mapping / tuple / list；exact piece 仍分别调用原 `decode_exact`，大容器拆成带精确键的独立 bounded pieces。恢复完整原 dict/list/tuple 身份、Fraction、整数标签、result、全部 visited_candidates、attempts、counter 和证书，没有裁剪或省略。piece 结构自身限20000节点和32层，拒绝未知类型、重复键、错误shape；既有 codec 的节点界完全不修改。普通载荷继续使用原 tagged 格式；环境 S+ 的旧 group_chunks 格式保持兼容。

本轮证据在 `outputs/delegation/GENERAL_PIPELINE/RUN20261014_algorithm_ui/`。实际 AppTest 两按钮路径、两种证书独立回读、普通 rerun 不重算、输入/预算失效、工作预算0 UNKNOWN、部分候选账本原样保留、大账本合法分块回读均有测试。全部旧模式测试保留。仅 CPU 数学与 AppTest，不表示照片/CAD 产线验收。

本轮两批完整 UI 测试均通过：第一批30项、补充非平行并列与外部调用隔离后的第二批32项。两个新增入口均实际点击验证；第二批保留全部24项旧模式测试。本轮没有失败或重试修复记录；算法文件未修改。

## 多中心下界证书 fits

新增第十一个入口“多中心下界证书 fits”，保留原十种模式及控件键。新键为 `exact_circuit_fits_input`、`circuit_fits_work`、`circuit_fits_seconds`。输入 `circuit-fits-input-v1` 包含 schema、reference、reference_parts、source、source_parts，完整两侧各4–7有理点及给定完整分划，不采样或合并。中心需符合秩2和合法四点关系的有界充分算法域。

调用 `circuit_fits`。只有 `CIRCUIT_FITS_SUFFICIENT_WITNESS` 时，将完整下载JSON无损回读后调用 `assess_circuit_fits_certificate`，独立核验使用构造余下work/时间。算法状态和独立回读核验分别展示；显示精确 `alignment_certificate.error_squared`，保留所有n!下界账本、全点严格包含和稳定半径。默认五个singleton reference为 `(-1/10,0),(101/10,0),(1/10,20),(99/10,20),(31,47)`，标签0..4；source为 `(0,0),(10,0),(0,20),(10,20),(31,47)`，外部标签10..14。该非零实例误差平方1/100、全部120对应和五点包含通过独立核验。

四点关系提供多中心精确下界充分证书，不能替代一般 fits 判定。UNKNOWN、UNSUPPORTED、NO_SUPPORTED_WITNESS 保留；所选中心变换全块包含失败时，不声称试过所有稳定变换，也不证明通用 fits 不存在。输入/预算变化隐藏旧下载，普通rerun不重算，只有点击按钮执行。

六/七中心的账本超过512条时，下载使用既有 `exact-research-proof-chunked-v1`，显式递归拆出完整ledger逐行独立bounded exact pieces。保持Fraction、整数标签、元组和完整容器身份，沿用每piece的20000节点解码界和整体piece结构界；不放宽算法或codec门。真实六点例在默认五点例加reference `(3,5)` 标签5、source `(3,5)` 标签15，全部720对应与六点包含均通过实际AppTest按钮、下载分块回读和独立证书核验。

本轮完整UI测试36项通过（包含原32项），使用现有kozlov Python，`-B -X utf8`、禁pytest缓存，新basetemp叶子。新增测试覆盖真实5点非零证书、11模式选择、点击才计算/rerun不重算、下载及陈旧结果失效、预算0 UNKNOWN、真实6点720条分块回读和未知保持完整输入且不作一般否定。算法模块未修改；没有浏览器server、模型、原生或网络运行。证据位于 `outputs/delegation/GENERAL_PIPELINE/20261014_network_followup/ui/`。

## 既往论文的固定朝向平移与给定位置匹配

新增三个入口，合计14种模式，保留历史11种及控件键、原JSON传输与字节插入顺序。全部仅显式点击计算；普通rerun不重算，输入或预算改变隐藏旧下载。每侧完整1–32点，不抽样。调用registration公开API；构造、内部核验与下载无损回读后的独立核验共用剩余工作/时间预算。算法状态和独立核验分别展示，UNKNOWN/UNSUPPORTED保持原状态。

“固定朝向与给定对应最优平移”：`fixed-orientation-translation-input-v1`，字段schema/source/target/mapping/orientation。source和target为既有label/xy点数组；mapping为明确的`{"source":整数标签,"target":整数标签}`数组，必须完整双射；orientation为2x2整数或numerator/denominator矩阵，必须精确有理正交，可反射。调用`solve_fixed_orientation_translation`，变换方向Q(source)+t向target；完整输入及证书无损回读后调用`assess_fixed_orientation_certificate`。默认source `(0,0),(10,0),(0,10)`、target `(0,0),(6,0),(-2,7)`，对应0→10/1→11/2→12，Q=I；平移 `(-2,-5/6)`、误差平方169/36。只优化固定朝向和给定对应下的平移，不优化全局转角或对应。键exact_translation_input/translation_work/translation_seconds。

“给定位置接受性”：`placed-candidate-input-v1`，字段schema/source/target/radius_squared。调用`candidate_acceptance_exact`，回读完整证书后调用`assess_candidate_acceptance_certificate`。普通接受需两侧每点有阈值内邻居；窄接受还需双射。默认source x=0,1,10，target x=0,9,10，均y=0，半径平方1。所有点都有邻居，普通接受True；前两个source只有target x=0一个邻居，Hall缺陷证明窄接受False。展示缺陷证书。仅给定位置判定；键exact_candidate_acceptance_input/candidate_acceptance_work/candidate_acceptance_seconds。

“给定位置瓶颈匹配”：`placed-bottleneck-input-v1`，字段schema/source/target。调用`placed_bottleneck_exact`，回读完整证书后调用`assess_placed_bottleneck_certificate`。默认同上三点集，最小瓶颈距离平方64，含达到阈值的匹配及较低阈值不能完美匹配的证书。要求两侧等长，只搜索对应与有限阈值，不搜索旋转、平移或全局配准。键exact_placed_bottleneck_input/placed_bottleneck_work/placed_bottleneck_seconds。

下载沿用exact-research-tagged-v1与exact-center-tags-v1，保存完整输入、全部证书、Fraction/tuple/int标签及输入文字sha256；不转换浮点距离，不计算近似平方根。证据目录为`outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/ui/`。验证仅CPU数学与Streamlit AppTest，不代表实际GUI/照片/CAD/模型验收。

定向完整UI测试43项通过，9.81秒，包含历史36项与新增三模式实际AppTest点击、下载、rerun不重算、陈旧结果隐藏、预算0 UNKNOWN、非零最优平移、Hall缺陷和瓶颈最小性、完整输入往返、确定性JSON字节顺序及外部执行隔离。证据pytest01.log/junit01.xml使用新叶子，不覆盖历史运行；没有安装依赖或执行模型。


## 2019 环境代码与椭圆代码

新增“环境代码与子图像限制”和“2019椭圆代码”，合计16种模式，保留既有14模式和控件键。仅显式点击按钮执行，普通rerun使用原结果；改输入、预算或功能时隐藏旧下载。新功能依然是完整给定二维精确点集的数学侧支，不判仿射等价，不代表照片/CAD质量或物理尺度验收。

环境输入 schema 为 `environment-code-input-v1`，字段仅 schema、environment、subset。environment 为完整1–8个label/xy点；subset为非空整数标签数组，必须为环境标签无重复子集。调用 `restrict_environment_code`，完整精确下载往返后使用 `assess_restricted_code` 独立核验。环境证书和子代码保留原始global标签、全部三元组双倍面积、退化零分母账本、零分子且非零分母的0比例、全部有序异三元组索引项。默认环境标签11/22/33/44/55，三角形11/22/33与33/44/55共享33；subset为11/22/33/44，没有重编号，环境标签55仍保留在完整输入与嵌套环境证书中。展示全环境/子代码比例数、全局标签、子代码定义状态、退化分母数和双方平面定理域。代码允许单线/两平行线域外输入，code_defined=False不伪称仿射等价。

椭圆输入 schema 为 `ellipse-code2019-input-v1`，字段仅 schema、points；完整3–8个精确点，所有三元组非共线。调用 `build_ellipse_code2019`；下载完整回读后调用 `assess_ellipse_code2019`。默认标签11/22/33/44对应 `(0,0),(2,0),(0,2),(3,3)`，有4个精确三角形最小外接椭圆及12项带索引面积比。展示全局标签、比例数、精确几何椭圆数、平面定理域。非共线三元组构造前提与平面定理域分开，三元组共线返回 UNSUPPORTED，不能伪造退化椭圆实体；所有原输入仍在下载中保存。

新控件键分别为 exact_environment_code_input/environment_code_work/environment_code_seconds 与 exact_ellipse_code2019_input/ellipse_code2019_work/ellipse_code2019_seconds。构造含内部核验、序列化回读后的独立核验共用剩余 work/time 预算。UNKNOWN保留且不运行无证书的独立验算；数学状态与 independent_check 分开展示。

下载沿用 `$exact` tagged传输。若完整新结果超过既有每树20000节点界，采用既有 `exact-research-proof-chunked-v1` 显式bounded pieces；不改变codec，不删字段或截断比例。n=8抛物线点的环境完整限制与椭圆代码均实测完整下载往返：56三元组/3080比例；椭圆模式另含56实体。旧环境S+、fits候选账本及registration传输保持兼容。

本轮定向UI测试51项全部通过，14.27秒，包含原43项与新增两模式的真实AppTest按钮、下载、rerun缓存、陈旧输入/预算失效、预算0 UNKNOWN、完整输入绑定、非共线拒绝、域外代码、n=8完整下载及外部执行隔离。使用既有kozlov Python，`-B -X utf8 -m pytest -p no:cacheprovider`，新basetemp叶子；没有安装、网络、模型或原生运行。证据在 `outputs/delegation/KOZLOV_LEGACY/20261014_environment_codes/RUN/ui/` 的 test01.execution.json 和stdout/stderr。算法源文件未修改。


## 环境内给定对应比例性

追加第17模式“环境内给定对应比例性”，保留上述16模式及51项测试。输入 schema 为 `environment-proportionality-input-v1`，严格字段 schema、env、source_labels、target_labels、mapping。env 是完整最多8点label/xy数组；source_labels/target_labels 是非空整数全局标签数组，允许两部分重叠；mapping 是 source/target 行数组，要求给定完整双射，拒bool及重复source/target。只输入对应，不自动搜索另一双射。

调用 `paper_proportionality_given_mapping`，完整输入/全环境证书/三元组账本无损下载回读后，使用剩余工作/时间调用 `assess_proportionality_certificate`。`PROPORTIONAL_GIVEN_MAPPING` 与 `NOT_PROPORTIONAL_GIVEN_MAPPING` 都是完整的给定对应谓词结论，因此界面 complete=True；不能以状态是否 EXACT_ 开头判断这两个结论。UNKNOWN/UNSUPPORTED及 independent_check 另行保留。

四项分开展示：`paper_proportionality_given_mapping`（原文只量化source非共线三元组）、`degeneracy_preserved`、`full_area_proportionality`、`theorem_eligible`。默认环境五点 `(0,0),(1,0),(2,0),(0,1),(1,1)`，标签0..4；A为0/1/2/3，B为0/4/2/3，对应0→0、1→4、2→2、3→3。原文给定对应比例性True、退化保持False、全部面积比例False、平面定理适用False；source三元组0/1/2共线而mapped三元组0/2/4面积非零，标 NOT_QUANTIFIED 并保留账本，不删去该事实。三项不能混为仿射见证。

重叠三元组可能映射到相同三元组；同一非零面积之比1可导出，但原文环境代码排除self-pair。因此账本明确 SELF_PAIR_EXCLUDED，界面说明此1不是环境索引项，不向环境ratios补项。显示全环境/双方全局标签、比例常数、三元组账本数和self-pair数。给定对应反例只否定此对应的比例谓词，不证明不存在其他对应、不判全局非仿射。

控件键 exact_proportionality_input/proportionality_work/proportionality_seconds。完整8点identity输入实测保留56条self-pair账本和3080环境比例索引项，下载独立回读通过，未伪造self-pair环境比例。仅点击计算；rerun缓存、陈旧输入隐藏下载及预算0 UNKNOWN均经过真实AppTest按钮核验。

终轮UI测试61项通过，15.46秒，保留原51项，并增加比例性语义区分、合法负证书完整性、严格bool/重复对应拒绝、8点完整identity回读、真实按钮与外部执行隔离。实际argv、stdout/stderr保存为本轮 `RUN/ui/test02.*`，test01证据保留。仅修改页面/UI测试/本说明；算法源只读，未安装依赖、执行模型、网络、原生产管线或GUI server。

## 2019椭圆代码未知对应等价：分开的构造与独立核验

第18模式将已验证的 `ellipse_equivalence2019` 算法及专用codec接入页面。输入 schema 为 `ellipse-equivalence2019-input-v1`，字段严格仅 schema/source/target；两侧各为完整3–8个 label/xy 点数组。坐标沿用整数或约分后的 numerator/denominator 对象；拒绝float、bool标签、重复标签、多余字段、过多/过少点，不抽样。所有三元组必须非共线，共线时算法返回 UNSUPPORTED，完整输入仍保留供下载。用户不提供双射，算法自行按完整编号枚举候选。

控件键 `exact_ellipse_equivalence2019_input`、`ellipse_equivalence2019_work`、`ellipse_equivalence2019_seconds`。原有 `exact_compute` 在本模式显示“搜索并构造椭圆等价证书”，仅调用搜索与专用序列化，不自动调用下载回读的外部assess。结果先显示 `independent_check=NOT_RUN`。算法自身内部核验与用户另行触发的独立下载核验明确区分。

结果与两个无损下载出现后，再点击“独立核验下载证书”（键 `exact_ellipse_equivalence2019_verify`）才回读原下载JSON并调用 `assess_ellipse_equivalence2019`；不重新搜索。普通rerun不会调用搜索或独立核验，保留原结果。输入、模式、工作或时间预算变化后旧结果及下载失效，须显式重新构造；非法输入清除旧结果。旧17模式按钮行为、控件键与传输格式均保留。

`ELLIPSE_EQUIVALENT` 与 `NOT_ELLIPSE_EQUIVALENT` 都是代码定义级完整结论；后者只在全部允许双射被排除或基数不同后成立。UNKNOWN/UNSUPPORTED 下载保存完整输入，`complete=False`、`has_certificate=False`、`independent_check=NOT_AVAILABLE`，独立核验按钮禁用；不能把无证书核验成 VERIFIED。工作预算0实际进入算法 UNKNOWN 分支；序列化剩余预算耗尽同样转为 UNKNOWN，不发布部分证书或保留肯定结论。

构造工作预算最多2,000,000、时间最多30秒，搜索及专用序列化使用同一剩余额度。独立核验是另一显式动作，使用当前相同预算上限，保守预留一半work给专用解码、一半给assess，两者共享总时间；任一不足保留 UNKNOWN，不能当作证书无效或不存在映射。codec仅表示精确结构解析，数学 VERIFIED 必须来自assess。下载摘要的代码谓词、定理域、定理结论与仿射见证状态也须与独立核验严格类型一致，否则 INVALID。

本模式下载外层为 `exact-research-ellipse-equivalence2019-v1`，metadata/input使用原 exact tags，证书专门调用 `encode_ellipse_equivalence2019` / `decode_ellipse_equivalence2019`。完整8点仍保存两侧各56个实际数学椭圆及3080项索引比例，比例256行分块共13块；不借用通用codec放宽全局20,000节点界，不删字段。输入JSON沿用4MiB限额；下载证书回读另设64MiB限额并拒重复JSON对象键，避免将完整负证书误按输入小树截断。

页面分开展示原文椭圆代码等价、作者平面定理域、定理层结论及**同一编号映射**仿射见证。所有三元组非共线不自动等于作者平面域；三角形或平行四边形代码定义结果可成立，但定理结论为 NOT_APPLICABLE。同一映射 NOT_ESTABLISHED 不否定其他仿射编号对应。默认两侧4点为重新编号的有理非奇异仿射例，完整比例12项；8点正例与4点全24排列负例进入新增测试。

本轮工作目录为 `outputs/delegation/PPP_VIDEO/20261014_object_first01/automatic_query_cameras_20261018`，修改前页面、旧测试和本说明与 START_BINDINGS.json 及 original 快照三方SHA256一致。新增测试设计包含两按钮实际AppTest路径、缓存与输入/预算失效、完整8点专用codec下载回读、小负例、域外编号歧义、无证书/预算UNKNOWN、非法输入及元数据篡改、禁止外部执行。未在接线阶段运行测试，等待主线统一验证。算法与上一轮封印文件未修改；没有运行模型、GPU、真实照片/视频、原生重建或浏览器server。数学入口可操作不表示整篇论文、照片收益或CAD验收完成。
