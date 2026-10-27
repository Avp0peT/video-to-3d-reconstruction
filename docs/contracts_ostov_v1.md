# 精确2025 m-ostov首个分支契约 v1

此分支推进一般m-ostov研究；一般目标不缩小，RC05不标完成。

S-2025 原文件 `.sources/kozlov2025_three_approximations.pdf` SHA256
`82fc78d0c0f8fa3de99ede3fd227adaeddc8f8328a1ef99d06755f1ec231b8f9`。
PDF p.4/期刊149 §4以最小面积外接椭圆中心定义图像中心，以长轴长定义尺寸；共线退化为线段，中心“центр — середина отрезка”。同页独立性为凸包无公共点、覆盖尺寸为最大块尺寸。PDF p.10/期刊155 §6将独立完整覆盖的m个块中心组成的图像命名为m-ostov。未使用§4非正式等半径圆假设，未引入同页后续futlyar条件。

`enumerate_collinear_m_ostovs_2025`只接受完整声明的1–7个互异二维共线点，整数身份标签，有限int/float/Fraction坐标；拒绝bool、非有限值、重复坐标与超范围，沿用独立覆盖2048位坐标限制。float按实际binary64精确有理值解释。共线用精确叉积判定；非共线为UNSUPPORTED_DOMAIN，不采样、不吸附。单点作为端点重合的退化线段，中心自身、尺寸0，是显式SOURCE-DERIVED退化延伸。

每块端点u,v，中心=(u+v)/2，尺寸²=Σ(u_k−v_k)²。`piece_size`为原文块尺寸，`cover_size`为原文覆盖尺寸（最大块尺寸）的精确描述：`{squared: Fraction, nonnegative_root: true}`表示非负sqrt(squared)，不能把squared字段称为尺寸值。保留每个覆盖的全部标签与中心见证；不按中心集合去重。

复用既有RGS分划枚举和独立覆盖证书；最多877候选、2000000工作量、30秒，共享预算不重置。工作量是约定的验证、求解、构造与独立oracle操作计数，不是Python指令数；时间采用合作边界检查，并非硬实时截止。中断保留已构造见证，complete=false/PARTIAL_UNKNOWN。只有全部Bell(n)候选判定、全部中心证书与独立切缝oracle相符才COMPLETE。独立覆盖必须为共线排序的连续区间，全部切缝给出2^(n−1)覆盖、每m为C(n−1,m−1)，这是本项目SOURCE-DERIVED证明，不冒充原文算法。

独立验证器从原始点重查标签、覆盖证书、端点极值、精确中心与尺寸，并用切缝集合检查完整结果；未调用枚举或中心构造函数。候选index/输入hash保留绑定。Python结果含Fraction，JSON证据将其写为numerator/denominator对象；独立验证器接受原生Python结果，不承诺直接接受编码JSON。

一般非共线域还缺最小面积外接椭圆**精确中心**的可证明求解/证书（既有APPROXIMATE_CERTIFIED仅包含与面积比，未保证中心误差）；不能拿近似中心放行此严格分支。futlyar与最优互位/fits另有边界和缺失2007教材程序依赖。本实现没有这些结论、物体语义、三维骨架或研究收益。
