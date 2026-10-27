# 给定独立覆盖的点/线/三角块精确m-ostov

`supported-block-given-ostov-v1`，2026-10-12。新独立接口，不改既有枚举/ostov/视频链代码。

已读`contracts_ostov_v1.md`和原S-2025提取文本：PDF4/期刊149以唯一最小面积外接椭圆中心定义图像中心、长轴全长定义尺寸，共线退化为线段中点；PDF10/期刊155将完整独立覆盖各块中心构成图像命名为m-ostov。故给定独立覆盖中心构造无需枚举全部覆盖，也无需futlyar/fits条件。覆盖尺寸为各块尺寸最大值，不能用平方量充当尺寸。

`given_cover_supported_block_ostov(points, parts, max_work=2000000, seconds=30)`接收完整声明的互异1..4096二维点、1..32非空块。标签必须原生int，完整输入每个标签恰覆盖一次，无抽样；原生int/float/Fraction坐标沿independent_cover2048位限制，bool/NaN/Inf/字符串拒绝，float解释为实际binary64值。先验证给定覆盖并独立重查每对凸包的严格分离证书；不独立时NOT_INDEPENDENT，缺点INCOMPLETE_COVER。

每个块允许点、共线线段、恰三极点凸包（可含内部点）。点尺寸0；线段取真正极端点中点与端点距离。三角块调用exact_triangle_ellipse的精确最优中心与证明。任一块超过三极点则整个给定覆盖UNSUPPORTED_DOMAIN，不附完整ostov，不据此说该覆盖不存在。EXACT_GIVEN_COVER_OSTOV仅意味着这个给定覆盖全部中心精确且覆盖独立，不意味着所有覆盖中心族已完成，更不是整物体骨架。

结构化尺寸：点/线使用`sqrt(squared)`；三角长轴全长使用`sqrt(2*(trace+sqrt(discriminant)))`。`cover_size={expression: max(nonnegative_lengths), lengths: [...]}`是各精确根表达式最大值，未伪装成有理数，不承诺指定哪个块取得最大值。表达式根都取非负实根。

work原生int0..2000000、seconds有限原生int/float0..30，bool不合法。共享单一全局deadline及工作账本；覆盖与三角子调用只收到剩余预算，返回work再记入全局，每块不重置30秒。逐块和末端独立重验也收费。合作检查不是硬实时监督；排序、normalize与有理位运算仍需外部墙钟看守。耗尽PARTIAL_UNKNOWN/complete=false/exact_centers=false。

`verify_supported_block_ostov(points,result)`从全部输入重新规范身份与唯一覆盖，重查独立cover证书，逐块独立重查点/线真正端点、共线性、中点、长度平方或三角证书；校验中心列表与最大根表达式。验证器不调用cover求解器、hull、中心构造器。普通dict可变，消费时须重验；JSON要精确保留分子/分母与标签类型，禁止浮点恢复证书。

真实调用预先固定规则：从既有433独立覆盖按candidate_index升序选择首个恰含一个非共线三点块且其余块最多两点的覆盖；独立重查既存cover证书，保留原固定7标签，不重新选点。仅进行一次给定覆盖中心构造，再独立核验；没有模型/GPU/SfM/native或下载。
