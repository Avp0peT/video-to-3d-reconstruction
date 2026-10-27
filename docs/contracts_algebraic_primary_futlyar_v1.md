# 给定覆盖的单实代数域 primary futlyar

2026-10-13；入口`algebraic_primary_futlyar.assess_algebraic_primary_futlyar(points,parts)`，独立核验`assess_algebraic_primary_certificate(points,certificate)`。延续S-2025 PDF10与有理版合同，覆盖完整1–7个互异有理二维点、至少两个非空块，必须凸包独立。不搜索覆盖或未知仿射互位。

完整分划最多只有一个不少于4点的块，因此现有三/四/五接触求解器至多引入一个四接触Root。其余有理中心和尺寸嵌入同一域，无需不同Root间的复合域。全有理输入使用P=X的唯一实根统一编码。各块均保留完整`exact-center-v1`证书，不单独取接触点计算块几何。

`R²=min(i<j)||O_i−O_j||²`，R为最小中心距全值，不是半距。每点须严格唯一最近所属中心且距离平方<R²。共线块尺寸L的筛查为4R²>=L²；椭圆全长为sqrt(2(trace+sqrt(discriminant)))，筛查等价于u=2R²−trace>=0且u²>=discriminant。所有比较在指定实根上精确执行，无平方根近似、浮点或跨通道相减。

独立核验先重查独立覆盖、各完整块的最优中心证书及Root绑定，再重算所有中心距离、全局最小半径、大小筛查与逐点归属。它不调用覆盖求解器或中心构造器。序列化只保留精确数对及原Root语境；类型敏感比较拒绝bool冒充整数。root内部为比较而细化隔离区间不改变输出绑定的原隔离区间。

TRUE证明此给定覆盖在严格内部子域形成primary；FALSE只否定此覆盖。m=1、中心重合、原文未裁定的圆周相等情形UNSUPPORTED；预算或代数构造未完成UNKNOWN。两百万工作/30秒为合作阈值，子调用只使用剩余预算。未实现大输入上的多Root复合域、一般连续仿射搜索或一般strict fits。
