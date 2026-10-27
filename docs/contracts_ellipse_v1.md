# 2025 最小面积外接椭圆：数值契约

版本`ellipse-area-v1`，2026-10-07。主线依据完整目标和T04B边界冻结本契约，新增`kozlov_lab.kozlov2025.ellipse`，不修改旧数学API或A/B绑定文件。

## 定义与实现层次

已目视核对S-2025 PDF第4页／期刊149：有限二维图像以包含其凸包的唯一最小面积椭圆定义中心，以长轴**全长**定义size；共线时退化为线段，中心为中点。原PDF的SHA256仍为`82fc78d0c0f8fa3de99ede3fd227adaeddc8f8328a1ef99d06755f1ec231b8f9`。单点作为零长线段处理，是本项目对退化端点的显式约定。

本实现返回面积有证书的**数值近似**，不称精确最优椭圆或精确Kozlov中心。算法采用[Todd–Yıldırım作者版](https://people.orie.cornell.edu/miketodd/TYKhach.pdf) §2、Algorithm3.1所述的Khachiyan对偶更新。求解为NumPy float64；最后证书使用Python标准库Fraction。没有SciPy/OpenCV安装，也不把拟合椭圆当最小包围椭圆。

## 接口、域与预算

`minimum_enclosing_ellipse(points, area_tolerance=1e-3, max_iterations=10000, max_work=10000000)`。

输入为1至4096个有限二维binary64兼容坐标，拒绝bool、complex、string、空/不规则维数、NaN/Inf及不能无损转为binary64的整型坐标。输入float32等数值以其实际表示值提升；不恢复输入测量精度。精确重复坐标去重，保留排序后的`unique_points`，不做容差聚类或丢弃极点。

`area_tolerance`须在[1e-8,0.25]；`max_iterations`为0至100000整数，`max_work`为1至100000000整数。work定义为迭代更新次数×唯一点数；排序、初始化/终止检查、预处理和末端证书不包含在该计数中，均另受输入规模限制。它不是墙钟/内存硬看守，调用方需有外层预算。

实际迭代上限=min(max_iterations,floor(max_work/n))。允许零更新：对称初值可能已满足证书；不能因为预算为0就否认已成立的证书。随机数据或未收敛问题达到上限时明确报告，不扩预算、不擅自放宽tolerance。

## 求解与包含性

先对存储float坐标作精确有理数仿射秩判断。真点/真共线直接退化；近共线但精确秩2不得压成线段。秩2输入先平移、缩放、SVD白化以改善条件。不能稳定白化时返回NUMERICAL_FAILURE，不添加人为厚度。

在工作坐标取q_i=(x_i,y_i,1)，u从均匀simplex开始；每次计算X=Σu_i q_i q_iᵀ、M_i=q_iᵀX⁻¹q_i，以最大M更新相应权重，步长(M−3)/(3(M−1))。权重归一，保留实际浮点权重。

候选c来自加权中心，C为加权协方差；B满足BᵀB=C⁻¹/2，然后映回世界坐标。结果以`||B(x-c)||²≤1`为权威表示，避免病态A=BᵀB显式相乘损失正定性。白化和反变换的全部舍入都受最终世界坐标证书检查。

浮点停止采用Mmax≤3+τ，为最终膨胀留出余量；这不是最终成功标准。末端先精确计算全部输入点q值；如有越界，最多3次向外缩小B，每次写回float后重新精确验包含。膨胀失败明确返回数值失败。

## 精确末端面积证书

下面是本项目对证书的独立推导。将**实际存储的世界输入、输出c/B、权重与τ**直接转Fraction；不能先算浮点协方差/行列式再转Fraction。非负权重按其有理数总和精确归一。令μ=Σu_i p_i，C_u=Σu_i(p_i−μ)(p_i−μ)ᵀ，要求det C_u>0。

任一包含所有点的椭圆(x−b)ᵀH(x−b)≤1满足

`tr(H C_u)+(μ−b)ᵀH(μ−b)≤1`。

由2×2正定矩阵的迹/行列式不等式，`det(H) det(C_u)≤1/4`，所以最优面积至少为`L=2π√det(C_u)`。这对任意simplex权重成立，**不要求输出中心等于μ**。

对已精确验证包含、det B≠0的输出，面积为π/|det B|；因此

`R² = 1 / (4 (det B)² det(C_u))`。

只有精确有理数比较`R²≤(1+τ)²`成立，才返回`APPROXIMATE_CERTIFIED`。结合包含性，它保证存储浮点输入的`1≤area(output)/area(optimum)≤R≤1+τ`。展示用的`area_ratio_upper_bound`也以精确平方检查向上舍入；不把近似sqrt当严格上界。

这不提供中心逐坐标或轴长误差≤τ的承诺，不证明实物/图像测量精度。π和半轴/长轴的展示值是浮点近似；面积比例证书中π相消。若展示量溢出/下溢或数值算法失败，保留明确失败，不填假数值。

## 状态与退化

- `APPROXIMATE_CERTIFIED`：精确包含与指定面积比均成立，converged=true；仍非精确最优。
- `DEGENERATE_POINT`：零面积、零尺寸、原点自身为中心。
- `DEGENERATE_SEGMENT`：精确共线，端点为真正极端点，中心用有理数精确保存；float中心是舍入显示，长度平方亦有精确字段。极端长度溢出或半轴下溢时相应显示字段为null并说明，不伪造有限长度或把非零半轴写成0。
- `MAX_ITERATIONS`：达到迭代/work上限且面积要求未证明；可附带精确包含的候选，但converged=false。
- `NUMERICAL_FAILURE`：数值/证书/范围问题；converged=false，原因保留。即使候选包含点，也不能冒充面积通过。
- 不合法参数抛`EllipseInputError`。

非退化结果给出center、factor、area、semiaxes（长/短半轴）、major_axis_length（长轴全长）、原始浮点权重及证书。结果字段冻结，数组由不可写bytes支持；`certificate`和`to_dict()`提供独立副本，避免普通调用修改几何后仍携带旧证书。`contains()`用实际保存值的Fraction精确判定，不通过任意epsilon把外点变成内点。未收敛不能传播为严格fits、覆盖或完整形象结论。

验收至少覆盖包含性、已知三角形/矩形/圆对称解的中心与面积、仿射中心/形状/面积协变、点/线/重复/近线退化、迭代耗尽、非法输入，以及独立重算证书。有限测试不替代一般定理证明。
