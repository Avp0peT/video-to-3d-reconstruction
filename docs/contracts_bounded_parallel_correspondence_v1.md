# bounded-parallel-correspondence-v1

来源S-1996印刷p338，本地PDF/抽文page19（page18是p337）。docs/specs/stereo_algorithms_1996.md§6/8：原文未知标记程序考虑n!双射，解可以不唯一，两侧实际AlgT′构造并四锚对齐，核验完整射线对应。这里joint4D affine-rank判据是SOURCE_DERIVED补足，不冒充原文逐字数值算法或n⁴总时间。

域：两个完整4..7有理互异二维点dict，数量相同，各图affine rank2，标签为64位int（拒bool/string），两图标签可同值但不推定对应。坐标int/Fraction，约分分子绝对值及正分母≤10¹²，与现有AlgT′输入一致；不抽样、不合并。仅精确单射平行投影、非平面代表、两个kernel方向不同，不含透视、噪声、遮挡或米制尺度。

给定mapping，joint=(x,y,u,v) affine rank3是上述域相容充要条件。证明：若非平面体通过两rank2投影且kernel不同，联合线性映射从R³到R⁴为单射，joint rank3。反之rank3 joint affine hull同构R³，坐标投影各rank2（完整图满2D），kernel为两条不同线，否则joint rank降2；有限各图互异保证点射线单射。rank4不可能来自R³两仿射投影，REJECTED_RANK4。rank≤2是UNSUPPORTED_PLANAR，不是错误对应：例如同kernel投影可来自非平面体，但不属于本不同方向模型。不能泛化此判据到原文特定объемное тело定理6。

每个rank证书给原点、r个差向量基、非零r×r minor列以及完整点的基系数。独立核验只det和全部span代入，证明rank上下界，不调用rank搜索。rank3找两图公共非共线abc以及joint面外e；转换标签使用完整显式`int -> 'L:'+str(int)`映射，原integer身份保留。两方向实际调用prepare_views/construct_representative，保存两body、全标签branch traces及AlgT′状态。四非共面anchors解第一body到完整原始两图的2×4 affine projection、第二body到第一body的3×4可逆alignment；逐全点独立代入。

accepted数学证书重验body rank3（四anchors非零det）、两projection linear rank2、两完整图投影等式、第二body四锚可逆alignment及全点相同、完整双射和string转换身份。分支trace是实际执行记录；独立几何证书证明该完整输出相容，不替代逐分支形式正确性证明。不能只joint rank通过而写AlgT′已执行。

`search_bounded_parallel_correspondence(source_xy,target_xy,max_candidates=5040,max_work=20000000,seconds=30)`枚举全部n!双射，逐mapping ledger保存ACCEPTED/REJECTED_RANK4/UNSUPPORTED_PLANAR/UNKNOWN_BRANCH及rank证据，accepted保存所有已证成功候选。SEARCH_COMPLETE且complete=True仅在n!全部独立可验分类，无UNKNOWN_BRANCH。planar仍按unsupported_count单列，不声称全部属于支持域。找到一或多候选均不证明真实唯一。预算/候选截断PARTIAL_UNKNOWN，保留已有候选，无抽样、删除或失败转换FALSE；n7可以因30秒或20M预算partial。

单一Budget含rank搜索、两实际分支构造（prepare budget.spent）、有限projection求解及独立核验。现有AlgT′协作operation预算，不硬实时中断；deadline在候选/分支边界检查。独立`assess_bounded_parallel_correspondence_result`不调用rank search/prepare/constructors，重验全部ledger、accepted缓存、身份、rank、projection；完整性依n!不同合法双射计数，与逐状态证据一起证明。候选证书树先作16384位/结构门再数学运算；未知预算UNKNOWN，不成为INVALID对应。

若prepare_views在返回Prepared/counter之前抛operation预算异常，无法取得实际spent。此时保守扣整个剩余allocation，逐候选work_accounting明确标conservative_unreturned_prepare_allocation、charged_upper_bound、actual_counter_available=False；work_used为保守逻辑计费而非实际指令数，不伪造测量。保留该mapping的rank3与UNKNOWN_BRANCH记录，搜索PARTIAL_UNKNOWN绝不complete。
