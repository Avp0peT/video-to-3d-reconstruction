# independent-cover-v1：给定二维有限集合的独立覆盖

依据：Kozlov 2025 PDF4／期刊149 §4：W(X)为凸包，A、B独立当且仅当W(A)、W(B)没有公共点；P_X为由X的两两独立非空部分组成的覆盖。PDF10／期刊155 §6的正式m-ostov构造沿用独立块覆盖前提。此接口只核验给定cover，不搜索最优cover，也不计算中心、覆盖尺寸、m-ostov、futlyar、fits、完整/闭形象或重要部分。

模块 `kozlov_lab.kozlov2025.independent_cover`，仅标准库。公开接口：

- `verify_independent_cover_2d(points, parts, max_work=2000000, seconds=60)`：返回状态、完整性、缺失labels、凸包labels和成对证书。points是非空dict，互异整数label→二维integer/Fraction/有限binary64坐标。初始坐标必须互异，禁止bool/string/NaN/inf，禁止静默去重。parts为显式非空label列表，每label只出现一次，恰好覆盖X；最多4096点、32块。
- `verify_cover_certificate(points, parts, result)`：不调用hull或SAT求解器，用所有原点的投影不等式及凸组合恒等式独立复核；返回bool。result必须dict。字段、part索引、坐标、axis、weights和labels错误均拒绝，不泄漏常见结构或数值异常。

算术对实际输入值精确；binary64逐值转Fraction，不表示物理测量误差为零。输入分子/分母各至多2048bit；证书axis8192bit，交点/权重32768bit；超限拒绝。证书数至多32选2=496，每个凸组合最多3个支持点。操作预算0..2000000、时间0..60秒，不能自动提高；这是协作停止阈值，不是额外操作系统硬限额。

独立性针对闭凸包：严格分离证书 `max(n·A)<min(n·B)` 从每对完整原点集合复算；相等不算分离。相切、穿越、包含均为NOT_INDEPENDENT。负证书给公共有理点z及其在两块中的非负凸组合，系数和为1，精确复算坐标等于z。点与共线线段退化正常处理；m=1成对独立性为空量词，合法覆盖不解决primary futlyar的m=1半径问题。

状态：VALID_INDEPENDENT_COVER／NOT_INDEPENDENT给证书及decision_complete=true；INCOMPLETE_COVER列missing_labels；UNSUPPORTED_INPUT为类型、重复坐标/labels、归属或规模错误；UNKNOWN_BUDGET与UNKNOWN_UNCERTIFIED保持decision_complete=false。UNKNOWN不能向高层形象传成PASS。

真实来源：隔离原型在固定骰子registered image1的全部1123条声明tracked observations上先诚实拒绝重复坐标。主线显式批准工程集合投影后，用完整point2D_idx／point3D_id aliases保留所有观测，将完全相等存储坐标归为1010个集合点，固定x<width/2分组512／498，并独立通过真实严格分离证书。证据在 `outputs/delegation/GENERAL_PIPELINE/20261012_independent_cover_proto/set_adapter/`。这是该完整声明二维集合的给定工程分组核验，不是语义部件、三维结构或整个研究完成。

原始observations→coordinate set adapter仍留隔离outputs：后续建议新增 `integrations/demo_pipeline/research_cover.py` 和 `scripts/export_research_cover.py`，读取冻结来源及显式全aliases ledger，先核身份再调用此模块。该产品接线需主线另行批准；本轮未修改pipeline/report或旧研究算法，未重跑真实重建。

证书边界修订：结果必须带 `schema=independent-cover-certificate-v1`、正确contract，且完整覆盖证明必须有 `coverage_complete=true`、`decision_complete=true`、无missing_labels、status与布尔pairwise_independence一致；缺少该schema的历史证书不能直接按新契约通过，必须从冻结输入重新生成。`hull_labels`仅为求解器诊断展示，不属于独立验证证书字段，其内容未获验证，调用方不得将其作为已认证凸包；独立验证只依赖完整原点集合与严格分离/交集见证。

时间预算从接口开始约束所有数学决策：seconds=0立即UNKNOWN_BUDGET，即使单点空量词也不返回PASS或不存在结论；规范化后和最终决策前再次检查截止时间。max_work=0在正时间预算下允许不需要计数操作的结论（例如单点单块空量词覆盖）；需要凸包/成对求解工作时返回UNKNOWN_BUDGET。输入规范化仍受既有4096点/32块/有理数位数上界限制，时间阈值为协作检查而非硬抢占。
