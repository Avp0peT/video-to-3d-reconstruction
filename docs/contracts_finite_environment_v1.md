# finite-environment-v1

来源S-2025 PDF p.3 / 期刊148：S为N≥1点环境，S'为全部非空部分（2^N−1），S*为全部群。PDF p.3–4 / 期刊148–149：S+群内成员两两点编号不相交。不是闭凸包独立，不需要覆盖全部环境。变换后不同身份的点即使坐标重合仍分别保留标签；本模块只接原始完整环境，仍拒原始重复坐标，不按变换后坐标合并身份。

输入为完整1..7标签二维点dict，坐标int/Fraction，约分256位，标签64位int，拒bool/float与重复坐标。无抽样。`enumerate_environment_parts`返回冻结EnvironmentParts，parts按排序标签的非零subsetmask递增，保留全部外部标签，star_cardinality=2^(2^N−1)仅为数量，不枚举S*。`verify_environment_parts`以独立包含/排除递推检查所有子集完整性。

`verify_separate_group`接显式成员标签列表，每部分非空、内部标签无重复且属于环境；群内标签交集非空则FALSE并给overlap身份。群无需覆盖环境；若成员凸包相交但身份互斥仍TRUE。不得拿本谓词替代独立覆盖。重复群成员也共享身份所以FALSE。空群在原文未明文单列；数学幂集解释包含空群，但后续u=min图像点数对它无定义。本模块默认include_empty=False（返回EXCLUDED_BY_POLICY），调用者可显式True采用空量词成立。该policy不作为原文新定理，不进入m-compatible。

`enumerate_separate_groups`用哨兵所属block0表示unused，其他非空块为群图像，枚举N+1身份的canonical restricted-growth-string分划。每个标签可unused；没有要求覆盖环境。每个S+群唯一对应这一分划。include_empty=True时数量Bell(N+1)，N=7为4140；False排除全unused唯一空群，4139。输出groups按字典序规范排序，各部分内部及群内排序。结果冻结，COMPLETE必须通过共享预算下独立完整证书核验。

`verify_separate_groups_certificate`不调用RGS枚举器：递归取最小剩余身份，分unused分支与包含该身份的所有块子集分支，独立得到完整集合；核验外部完整环境绑定、policy、规范排序、无重复和精确集合一致。缺群、重复群、错误policy、身份变动均拒绝。只有完整结果有证书；截断组列表是候选部分，PARTIAL_UNKNOWN或UNKNOWN且complete=False，绝不作为完整S+。

共享Budget覆盖RGS节点、输入工作登记和独立验证递推，默认2000000工作/30秒；输入验证受固定profile限制，先规范化后登记预算，协作时间检查不保证硬墙钟中断。环境全部部分最多127个，固定有限输入无需搜索预算；S*双指数集合永不生成。

m-compatible仅记录依赖：PDF p.11–12，∃一个参考图像、∃其同一个m-futlyar、∀群成员m-fits；依赖合法primary、整块包含、最优互位及稳定性，不执行。granularity/etalons/完整群/第三近似仍未实现；本模块仅给第一近似基础，无family/core或产品入口改动。
