# complete-small-set-covers-v1

模块 `kozlov_lab.kozlov2025.cover_enumeration`，公开 `enumerate_independent_covers_2d(points,max_candidates=877,total_max_work=2000000,seconds=30)`；仅标准库，不改现有独立覆盖求解器。依据2025 PDF4/期刊149闭凸包独立覆盖，PDF10/期刊155有限覆盖族；RGS为项目实现的完整枚举方法，不是作者指定算法。

完整输入1..7个互异二维点，label为非bool整数；坐标沿用independent-cover-v1有限Fraction规则及2048bit上界。内部复用现有normalize进行输入验证；不采样、不静默去重，N>7明确UNSUPPORTED_SIZE。原观测aliases ledger仍由适配层维护。

RGS按排序labels枚举每个未命名集合分划恰一次。独立部分不能共享点，因此任一完整独立覆盖必是分划；全部分划经精确求解及独立原点证书校验后，得到完整覆盖族。Bell计数单独递推，N1..7为1,2,5,15,52,203,877。

输出complete/visited_candidates/judged_candidates/rejected_candidates/uncertain_candidates/independent_covers；有效覆盖每项含parts、certificate、certificate_verified、源坐标SHA256。source_coordinate_sha256按排序label的hex及Fraction分子/分母十进制编码（紧凑JSON）计算，绑定实际binary64转Fraction值，非物理真值。

只有全部Bell(N)分划已判且每个数学证书独立通过才COMPLETE。max_candidates不足、solver UNKNOWN、证书污染/失败或中断均PARTIAL_UNKNOWN、complete=false；可保留已证覆盖，不把空结果当不存在。工作预算所有候选共享递减，计数为底层solver的work_used，非Python指令或独立验证指令；时间从接口开始含输入验证/独立验证，单调截止时间在各边界检查，属于协作阈值非OS硬抢占。seconds=0即UNKNOWN；正时间零工作单点空量词可COMPLETE。

没有覆盖尺寸、一般exactcenter、m-ostov/futlyar/fits或视觉语义结论。不是求最小覆盖：无其它约束单块覆盖恒满足独立性空量词。真实1010点给定cover验证与此小集枚举分开；不能取七点宣告完整1010点搜索。

用法：`result=enumerate_independent_covers_2d({0:(0,0),1:(1,0),2:(2,0),3:(3,0)})`，完整15候选判定、8个独立覆盖。不自动写文件；含Fraction证书时调用方必须显式无损序列化，不能转float后声称证书精确。
