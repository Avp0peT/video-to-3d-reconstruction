# 完整输入三接触最优外接椭圆证书

`complete-input-contact3-ellipse-v1`，2026-10-12。独立新profile，不改exact_triangle及已有given-cover适用域，不构造407覆盖族，不标RC04/05/06完成。

## 来源与证明

Kozlov S-2025 PDF4/149定义中心为最小面积外接椭圆中心、尺寸为长轴全长。权威数学依据：[Michael J. Todd作者讲义](https://people.orie.cornell.edu/miketodd/ublndetz.pdf)，§7–9/PDF10–12的外接椭圆弱对偶、接触及矩阵等号、自由中心。

三非共线支撑点均匀概率权重1/3，c为其平均，C=Σ(v-c)(v-c)ᵀ/3，Q=2C。既有独立三角证书证明该三点的唯一最优椭圆E(c,Q)。任意包含完整输入的椭圆必包含三个支撑点，其面积不小于E；若E以精确二次型检查包含完整输入，则E本身可行且达到同一下界。故它是完整输入唯一最优椭圆，精确中心相同。此证明不要求额外点位于支撑三角的凸包；三元组是接触支撑，并非替换完整输入。独立验证器嵌入支撑三角最优性证书再单独核验完整输入绑定和所有点包含。

## 接口与边界

`complete_input_contact3_ellipse(points, max_work=2000000, seconds=30)`：完整1..7互异二维点dict，原生int标签；原生int/有限binary64 float/Fraction坐标，2048位分子/分母限制。bool/字符串/complex/非有限值/重复坐标/8点以上拒绝，不采样。float按实际存储值解释。排序枚举最多35三元组；共线三元组跳过。支撑证书构造与逐个完整点二次型均精确有理。

`EXACT_OPTIMUM_COMPLETE_INPUT`：完整输入绑定、支撑唯一最优和全点包含均独立通过；complete/exact_optimum=true。`UNSUPPORTED_CONTACT_DOMAIN`：有限三接触profile未找到可核验证书，包括点/线输入；不是证明无椭圆/无ostov，也不是一般4/5接触分类定理。`PARTIAL_UNKNOWN`：合作预算耗尽、支撑计算未完成；不能附成功。`UNSUPPORTED_INPUT`：参数/输入错误。退化点/线沿已有独立API可处理，本profile只返回三非共线接触证书。

work原生int0..2M，seconds有限原生int/float0..30（bool拒绝），共享deadline/剩余work，三角子调用只收到余额，不能每个候选重置30秒。三元组检查、完整点二次型、末端独立重验收费，排序/normalize/有理位运算非硬抢占，调用方需外层墙钟约束。结果visited_triples计已访问（含共线）候选数；早成功无需枚举其余三元组，因为最优性已有证明。

证书含完整标签/坐标SHA、三支撑标签、独立支撑三角最优证书、全输入精确q值、c/Q/Q^-1、面积平方除π²、半轴平方与长轴全长根表达式。Q特征值为半轴平方；长轴全长=`sqrt(2*(trace+sqrt(discriminant)))`，不是有理平方量。没有浮点尺寸当精确结果。

`verify_complete_input_contact3_ellipse(points, certificate)`不调用求解器、组合枚举或hull，独立核验完整绑定/严格标签类型、支撑最优证书、输出字段一致以及每点q≤1。证书是可变Python dict，消费前重验；JSON证据需保留Fraction分子分母及标签类型，禁止浮点反解。真实7点证明只针对固定完整子集，不针对1010点图像、语义物体、futlyar或严格fits。

测试用四极点但三接触最优的独立参考/仿射真值、外点绑定、square不支持、支撑与权重/身份/矩阵篡改、禁止验证器求解路径、near-collinear binary64、非法类型及共享预算。尚未实现一般4/5接触代数椭圆与独立根隔离证书。
