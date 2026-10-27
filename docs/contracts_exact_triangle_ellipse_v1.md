# 完整三角凸包输入的精确最小面积外接椭圆

`exact-triangle-ellipse-v1`，2026-10-12。独立研究基础，不修改ellipse/ostov/cover接口，不完成RC04/RC05一般域。

## 权威依据与独立证明

已查作者Cornell托管的[Michael J. Todd, ICIAM 2007, Minimum-Volume Ellipsoids: Applications, Duality, and Algorithms](https://people.orie.cornell.edu/miketodd/ublndetz.pdf)，PDF10-12页（§7-9）：包含椭圆的迹/行列式弱对偶、接触与矩阵等号条件、自由中心提升。其讨论对象是最小外接椭圆。本分支不是最大内接Steiner椭圆。

以下为本项目独立推导，而非从提示接受公式。三角顶点v1,v2,v3，c=(v1+v2+v3)/3，d_i=v_i-c，C=Σd_i d_iᵀ/3，Q=2C。Q正定，椭圆定义为(x-c)ᵀQ^-1(x-c)≤1。以参考顶点(0,0),(3,0),(0,3)独立计算c=(1,1)，Q=[[4,-2],[-2,4]]，三个顶点二次型均为1；任意非退化三角形是参考三角形的可逆仿射像，Q及接触等式协变。因此候选包住三角形及内部全部点。

任意竞争外接椭圆(x-b)ᵀH(x-b)≤1，对顶点取正权重1/3得tr(HC)+(c-b)ᵀH(c-b)≤1。HC与正定对称矩阵C^(1/2)HC^(1/2)相似，两个正特征值乘积≤其迹平方/4，故det(H)det(C)≤1/4。竞争面积≥2π√det(C)=π√det(Q)，候选恰达到下界。等号必须b=c、HC=I/2，故H=Q^-1；唯一最优，中心精确。可白化写成接触权重2/3及Σ(2/3)u_i u_iᵀ=I；概率权重1/3的协方差则是I/2，不混淆两种归一化。

## 接口与边界

`exact_triangle_enclosing_ellipse(points, max_work=2000000, seconds=30)`。输入完整dict[int, pair]，1..4096互异点；不采样、不容差合并。沿independent_cover规范：仅原生int、有限原生float(binary64实际值)、Fraction，拒绝bool、字符串、complex、NumPy标量；分子/分母各≤2048bits。重复坐标拒绝。全凸包恰三极点才EXACT_OPTIMUM；单点、线段、四个或更多极点UNSUPPORTED_DOMAIN；非法输入/预算UNSUPPORTED_INPUT。显式工作/时间耗尽PARTIAL_UNKNOWN，不能传播成功。

max_work原生整数0..2000000；seconds原生int/float、有限且0..30，bool拒绝。复用现有Budget/hull，工作按预处理、hull操作、逐点几何与证书核验保守收费；排序、整数位运算和normalize不是硬抢占，外层调用方仍须看守墙钟。无迭代或自动扩预算。

结果c/Q/Q^-1及证书全部有理精确。shape_matrix是Q（特征值是半轴长度平方），不是二次型矩阵；inverse_shape_matrix才是二次型。trace=t，discriminant=D=(Qxx-Qyy)^2+4Qxy²。半轴平方λ±=(t±√D)/2，长轴全长=√(2(t+√D))；两者以结构化精确根表达式保存，不把平方当长度。面积平方除π²=det(Q)。没有浮点主轴显示冒充精确值。

证书绑定全部标签/坐标SHA256，保存每个点相对于三个输入顶点的非负重心坐标、c/Q/逆矩阵、正接触概率权重1/3、行列式和精确根表达式。`verify_exact_triangle_ellipse(points, certificate)`不调用hull/solver，独立核验全点重心恒等式（证明凸包恰该三角）、包含性、正定性、逆矩阵、接触边界与协方差等式（证明最优性及唯一性）。结果是普通可修改dict，必须在消费时重验；不是不可变认证对象。当前证书使用Fraction Python对象；JSON导出应显式编码分子/分母后按类型严格解码，不能用float恢复。

测试包含独立参考真值及有理仿射变换、内部点、完整7点四极点反例、binary64/极薄三角形、非法类型、重复与预算、绑定与证书篡改。只覆盖数学纯数据；未执行模型/GPU/SfM/native，未证明真实照片完整产品验收，未接入一般非共线ostov。
