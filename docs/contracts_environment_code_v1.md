# 完整环境代码与子代码限制 v1

来源：`S-2019-CODES`，打印 pp57–58，论文《Коды, определяющие изображения с точностью до аффинных преобразований》。本轮封存全文为 `outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/2019_codes.txt`；先前逐页核对见同目录 `CODES_2019_AUDIT.md`。定义为带编号三角形面积比代码；p58规定环境 S 的子代码按全部索引属于子标签集合筛选。有限枚举、存储格式及证书验算是项目实现，不是作者逐字给出的程序。

## API 和完整输入

模块：`kozlov_lab.coding.environment_code`。

- `build_environment_code(points, *, max_work=2000000, seconds=30)`。
- `restrict_environment_code(points, labels, *, max_work=2000000, seconds=30)`。
- `assess_environment_code(points, certificate, *, max_work=2000000, seconds=30)`。
- `assess_restricted_code(points, labels, certificate, *, max_work=2000000, seconds=30)`。

`points` 必须是完整的 1..8 个不同二维点的 dict，键为 int64 全局标签。坐标只能是精确 int/Fraction，其分子绝对值与分母分别最多 256 bit。拒绝 bool、float、重复坐标、不完整形状及超界输入。子标签使用非空 tuple/list，必须是环境标签的无重复子集；顺序规范化，但不重新编号。空环境或空子图不是原文的非空图像，返回 UNSUPPORTED。

成功返回 EXACT_ENVIRONMENT_CODE / EXACT_RESTRICTED_CODE；验算成功 VERIFIED。返回包含 status、certificate、work_used、code_defined、plane_domain、affine_equivalence_decided=False。非法构造输入返回 UNSUPPORTED，非法证书返回 INVALID；工作或时间预算耗尽返回 UNKNOWN 且 certificate=None，不发布截断代码或假否定。

## 索引和证书

环境 schema 为 `indexed-environment-code-v1`，source 精确为 `S-2019-CODES:57-58`。

- `environment_points` 是按全局标签排序的 `(label,(Fraction x,Fraction y))` tuple，绑定全部输入。
- `labels` 保留全局编号。
- `double_areas` 为所有无序三元组的绝对双倍面积，包括共线三元组的 0。
- `zero_denominators` 列出全部面积为 0 的三元组索引。
- `ratios` 列出全部不同三元组的有序分子/分母对，且仅排除零分母。零分子保留 Fraction(0)；自对排除。比值为双倍面积之比，与原文面积比严格相同。
- `code_defined`、`domain`、`plane_domain` 描述代码与定理适用域，不作等价结论。

索引键有唯一语法：排序三元组用 canonical 十进制 int64 标签和逗号组成 `a,b,c`；面积比用 `a,b,c/d,e,f`。无前导零、加号、空白或重新编号。负号、逗号和斜杠互不混淆。验算不只解析：完整重算 canonical 键和值，拒绝漏项、额外项、不规范键、bool 冒充数字或标签。比值是带标签键的映射，不是值的 multiset。

子 schema 为 `restricted-environment-code-v1`，具有 source、environment（完整嵌套环境证书）、subset_labels 和 restricted。restricted 含 labels、double_areas、zero_denominators、ratios、code_defined、domain、plane_domain。构造从完整环境所有索引项筛选，验算重新枚举子标签所有三元组/索引对，并查取已经独立验算的环境项。不能用仅对子图重新生成的环境证书冒充全环境绑定。

n=8 最多 56 个三元组、3080 个 ratio 项，没有截断。字符串索引避免重复嵌套三元组带来的树规模：环境和全标签限制证书均可使用既有 `encode_exact` / `decode_exact` 完整 JSON 往返，新增测试实测 n=8 全限制回读。存储数字与证书树另设 16384 bit、深度16、单容器4096界限。

## 域、验算与限制

不足3点或没有非共线三元组时 code_defined=False；完整面积/零分母账本仍保存。恰有一个非退化三元组时 code_defined=True，即使异三元组比值集为空。没有把空的 T 自动解释为完整识别不变量。

plane_domain 仅当全部点不能被一条直线或两条平行直线覆盖时为 True。精确枚举所有点对方向，计算所有点的方向行列式 offset；至多两个 offset 即相应域外见证。对于不超过3点的情况自然不能进入该域。代码允许域外输入，但没有判仿射等价或使用2019定理1的充分必要结论。

独立验算器不调用构造器、投影器；用展开行列式重算全部双倍面积，从分母方向独立重算全部比例，再核验输入绑定和域。预算复用 registration.fixed_orientation 的有界工作/时间实现；验算与构造同受界。已有 exact codec 只用于运输，不代替几何/索引验算。

该模块没有椭圆几何、椭圆代码别名、跨图比例搜索、点编号双射搜索、仿射等价判定、模型或生产管线接入。退化零分母的比值不定义，不补成 0；零分子且非零分母的比例为 0。
