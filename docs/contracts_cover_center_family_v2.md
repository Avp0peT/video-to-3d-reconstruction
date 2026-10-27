# 有限完整覆盖中心族 v2

2026-10-13。继承[v1](contracts_cover_center_family_v1.md)的完整输入、全部Bell分划、合法/拒绝覆盖账本、单预算与独立完整性核验；新生产schema为`cover-center-family-v2`。

每个完整块依次尝试supported、contact3、contact5与一般contact4。只有域不支持才转下一求解器；耗尽预算不冒充域不支持。四接触采用[精确实代数证书](contracts_exact_contact4_v1.md)。相同完整标签块可以缓存；未解决块和其覆盖均保留UNKNOWN，不删除后再报完成。

contact4中心是`algebraic_vector`，长轴是`algebraic_length`，两者各自带Root定义，不能把不同覆盖的生成元视为同一数。覆盖尺寸仍是精确非负长度表达式的max；此接口不返回跨代数域比较后的赢家。独立验证兼容历史v1，但v1不接受contact4行；v2逐项重验四接触正权重最优证书。

固定完整七标签输入[13,16,21,24,26,27,31]本次独立实际运行：877分划、433合法覆盖、433精确、0未知；旧结果407精确/26未知保留历史。逐项证书核验与无损JSON读回通过。它是这个固定有限输入的全部覆盖中心，不是整张照片、物体语义或三维模型验收。

证据：`outputs/delegation/GENERAL_PIPELINE/20261013_algorithm_core/REAL7_RESULT.json`及同目录输入绑定、原始结果、执行记录。
