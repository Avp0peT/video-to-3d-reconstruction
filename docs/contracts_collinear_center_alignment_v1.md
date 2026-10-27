# collinear-center-alignment-v1

S-2025 PDF p9–10 / 期刊154–155定义最优互位与共同仿射扰动后稳定性。本模块仅完整有理共线中心集合全局对齐，不证明任何wholeblock fits。两集合各2..7互异二维点、点数相同，标签64位int，坐标int/Fraction约分256位；bool/float/重复坐标/非共线拒绝。输入全点保留。

证明一：任意可逆仿射像的source中心仍共线。到target直线的正交投影不增加每对欧氏距离，其投影是source线参数的仿射函数cy+b（允许c=0）。一维绝对距离瓶颈匹配可以uncross：若x1≤x2,z1≤z2，交叉匹配的最大距离不小于顺序匹配；反复消交叉即排序匹配。故只需source两方向排序和a≥0的LP。负方向按−y重新升序，不能仍按原y升序。

证明二：target归一化坐标端点0,1，source按任一方向排序的参数跨度1。collapsed a=0最小误差1/2。取足够小a>0及b=(1−a(u_first+u_last))/2，两端误差降为(1−a)/2；每个内部target点原误差严格<1/2，有限多个点有正margin，故小a仍严格改善1/2。没有内部点的2点情形直接成立。因此最优a>0、投影下界由全平面可逆延拓实际达到，而非仅奇异极限。若构造得a=0或未证完备则UNKNOWN。

LP变量(a,b,e)：每点±(a*u_i+b−x_i)≤e，加a≥0/e≥0，rows·primal≤rhs。枚举三个活动约束，以Fraction解3×3；找可行primal与λ≥0满足Σλrow=(0,0,−1)、−Σλrhs=e，取得完全精确上下界。LP有有限最优顶点；退化最优也具有至多3支持约束的dual，可补0权重到独立3活动约束。两个方向均保存primal/dual才可宣布全局最优。

构造M=[a*s*uX,J(uX)][uY,J(uY)]^-1，t=x0+b*uX−M*y0，a>0使M可逆；全部source中心精确映到target直线上。最大欧氏距离平方=e²||uX||²。横向固定系数1只是一个合法中心延拓，不声称wholeblock入域。

稳定性证明：对(X,TY)作为整体同时作用任意可逆H后，两图像仍在共同直线上；仿射线参数关系不变。任意新的source仿射像再次投影和排序仍归同一个LP，误差统一乘||HuX||，原对应仍全局最小。故不仅局部成立，epsilon1=epsilon2=1也满足原文定义。不把独立变换X与TY误称共同H，不采样邻域。

`align_collinear_centers(target,source)`成功GLOBAL_OPTIMAL_STABLE，预算不足UNKNOWN、域外UNSUPPORTED。独立`assess_collinear_alignment_certificate`只核完整点线参数、两方向所有primal不等式、dual非负和恒等式、可逆M/t、全点映像/双射和误差平方；不调用枚举器或3×3求解器。严格int/Fraction类型拒bool污染；在任何证书算术前使用有界树检查，所有储存有理数至多16384位，包含在源直线上不影响映像的横向系数。超界UNKNOWN，不先执行超大数运算。默认单一2M工作/30秒，构造+独立证书共同计费，协作式时间检查不是硬墙钟中断。
