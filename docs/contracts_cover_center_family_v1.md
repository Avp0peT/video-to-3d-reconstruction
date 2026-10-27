# 有限给定环境的覆盖中心族 v1

> 历史v1合同保留。2026-10-13当前生产接口已升级为[v2](contracts_cover_center_family_v2.md)，新增一般四接触精确代数证书；下文“未实现四接触”仅描述v1时点。验证器仍支持合法历史v1。

模块`kozlov2025/cover_center_family.py`，入口`build_cover_center_family(points,max_candidates=877,total_max_work=2000000,seconds=30)`及`verify_cover_center_family(points,result)`。仅完整给定1–7个互异二维点；整数非bool标签及有限int/binary64/Fraction坐标，沿用现有2048位规范。无采样、合并或丢标签；原照片全部观测并不属于此有限输入。

复用`cover_enumeration`的完整RGS独立覆盖枚举、`ostov_supported_blocks`的点/共线/三角凸包精确构造，以及`exact_contact3_ellipse`的完整块三接触证书。三接触域明确不支持且完整块至少5点时，回退到`exact_contact5_ellipse`的有理rank5、严格正接触权重证书；新增kind=`contact5`，独立验证调用其证书验证器。支撑点只证明最优性，不替换完整块。五接触失败也保留UNKNOWN，不推断不存在椭圆。未实现一般四接触代数精确求解、全部ostov/futlyar、strict fits或三维理论。

枚举所有已证合法cover后立即为每cover登记UNKNOWN；每块从完整标签子集求中心/长轴全长及独立来源证书。相同标签块缓存只避免重复数学构造。每cover的`centers`保留按parts顺序的中心列表（包括可能重复中心），`cover_size`为精确非负根表达式的最大值；不伪装成有理数或物理尺寸。不能处理的块保留UNKNOWN所在cover及原因，其余cover继续处理。

共享work账本与从入口开始的单一monotonic截止时间，子调用只接收剩余额度，work_used返回后累计，包含枚举、补构拒绝证书、supported调用与contact3/contact5回退，不为每块重置。工作计量沿用子求解器操作定义，不涵盖Python指令或独立验证指令；墙钟覆盖这些操作并在边界检查，协作阈值不是OS硬抢占。零时间/零工作均不报完成。

旧枚举未保存拒绝cover证书，本模块补构已判非独立分划的证书并计入同一预算，产生按candidate_index连续排列的`decision_ledger`。独立验证器复用RGS仅列举分划身份，不调用求解器、hull或中心构造器；逐项验分离证书并对照合法cover族，故可识别遗漏合法cover、误标签、证书污染、中心/尺寸漂移。三角/点线证书与contact3证书分别调用已有独立验证器。两个源摘要编码不同：family/块采用exact_triangle的整数label编码，enumeration采用其原hex-label编码，分别重算不混用。

`enumeration_complete`说明全部Bell(N)候选已判；`decision_ledger_complete`说明全部分划证书已保存；`complete`只有两者成立且全部合法cover中心均EXACT才true/COMPLETE。不支持的接触域或预算失败保留PARTIAL_UNKNOWN，不能过滤未知cover后声称complete。验证器true表示声明的部分/完整结构与证书一致，并不单独表示完成；消费方须同时检查complete。无enumeration的早期输入/预算失败不能通过结构证书验证。

输入、配置与输出由调用方冻结，不自动读写文件。JSON必须无损保留Fraction、tuple、整数dict键，禁止浮点转换证书。来源语义延续已有三个合同及S-2025 PDF4/149中心/尺寸与PDF10/155独立覆盖块中心；这是有限给定环境的有界中心族，不宣称RC04/RC05一般理论或完整物体研究完成。
