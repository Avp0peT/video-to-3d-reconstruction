# 完整输入的严格五接触有理精确椭圆证书

公开独立模块`kozlov2025/exact_contact5_ellipse.py`，入口`complete_input_contact5_ellipse(points,max_work=2000000,seconds=30)`及`verify_complete_input_contact5_ellipse(points,certificate)`。不改contact3/triangle/family/init。完整1–7个互异二维点，原生非bool整数标签，有限int/binary64/Fraction坐标沿既有2048位规则；不采样、不静默去重，float按实际存储值解释。

来源和证明：Kozlov S-2025 PDF4/149以最小面积外接椭圆中心/长轴全长定义center/size；[Todd作者讲义](https://people.orie.cornell.edu/miketodd/ublndetz.pdf) §7–9/PDF10–12弱对偶、接触与矩阵等号，自由中心处理。新rank5有理conic构造与精确线性权重是本项目实现策略，具体静态推导见`outputs/delegation/GENERAL_PIPELINE/20261013_contact5_feasibility/REPORT.md`。

每个五标签支撑的5×6评价矩阵按行`[x²,2xy,y²,2x,2y,1]`。精确RREF rank5才取唯一非零nullspace系数(A,B,C,D,E,F)，对应`A*x²+2B*x*y+C*y²+2D*x+2E*y+F=0`。M=[[A,B],[B,C]]须正/负定，整体规范至A=1、detM>0。d=(D,E)，c=-M^-1*d，rho=d^T*M^-1*d-F>0，S=rho*M^-1，H=M/rho。完整输入均须(p-c)^T H(p-c)<=1，五支撑均精确等于1。

λ的6×5系统列`[1,zx,zy,zx²,zx*zy,zy²]`，右侧`[1,0,0,S00/2,S01/2,S11/2]`，z=p-c。rank5与边界恒等式给唯一有理解；实现仍检查一致性。五权重必须严格正，和1、均值c、协方差S/2。零权重不作为本profile成功，不声称解决较小支撑问题。

最优性独立推导：Cλ=S/2。任意包含完整输入的椭圆中心b与precision G满足tr(G*Cλ)+(c-b)^T G(c-b)<=1，由二维迹/行列式不等式得面积>=2πsqrt(det(Cλ))=πsqrt(detS)。候选完整包含且达到下界，故是完整输入唯一最小面积椭圆。此域与既有三角/contact3的正权重协方差等号一致。

支撑最多C(7,5)=21个，按排序标签枚举，成功即可停止（证书足够证明最优，不需其余搜索）。N<5无候选，返回UNSUPPORTED_CONTACT_DOMAIN。候选失败原因计数包含rank_not_five/not_definite_conic/nonpositive_radius/moment_system_inconsistent/nonpositive_weight/complete_input_outside；未找到仅表示严格五接触有理rank5域不支持，不证明无椭圆或一定四接触。

工作max_work原生整数0..2M；seconds有限原生int/float0..30（拒bool）；一个Budget/单一deadline，全部候选消元、包含、minor与末端验签计入。work是项目操作计数非CPU指令；协作边界非OS硬抢占。内部有理分子/分母最大16384位；超过立即PARTIAL_UNKNOWN，不能因中间位数超限返回“无解”。零工作/时间PARTIAL_UNKNOWN；不重置候选预算。非法输入/预算UNSUPPORTED_INPUT；已规范输入之后内部构造/验签异常PARTIAL_UNKNOWN，不冒充输入错误。

EXACT_OPTIMUM_COMPLETE_INPUT仅完整绑定/五正权重/独立核验均通过，complete/exact_optimum=true。其他状态没有成功证书。证书包含support/full labels、坐标摘要、conic_coefficients、rank5_minor_columns/determinant、rho、λ、c/S/H、每点q、detS、det(Cλ)=detS/4、area_squared_over_pi_squared=detS、精确根轴长与proof。

验证器不调用RREF、求解器、组合搜索或hull；从原完整输入重算5阶minor直接置换行列式与conic接触，检查rank5，M*c=-d、M*S=rhoI、S*H=I、正定、五权重和/均值/协方差、完整每点包含、支撑边界及所有证书输出字段。数字证书只接受int/Fraction，严格拒bool/float（包括与正确有理值数值相等的float）。标签严格int。消费前重验可变dict；JSON须无损保留Fraction、tuple、整数键。

根表达式沿旧接口：半轴平方`(trace +/- sqrt(discriminant))/2`，长轴全长`sqrt(2*(trace+sqrt(discriminant)))`，trace=S00+S11，disc=(S00-S11)²+4S01²。不是精确有理长度，也不是实物毫米。

手算真值五点(1,0),(-3/5,±4/5),(0,±1)，c=0、S=I，权重依次5/16,25/96,25/96,1/12,1/12；λ二阶矩I/2，可独立验证仿射变换。完整点包含/外点、类型与字段污染、rank退化、非椭圆、负权重、共享预算、位数超限、禁止验证器求解是本次必要测试范围。未实现一般四接触代数域、一般RC04或完整物体理论；任何真实证书只对声明完整输入成立。
