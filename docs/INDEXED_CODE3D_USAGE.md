# 完整三维索引代码：小规模精确接口

当前状态为 `TESTED_REAL_PARTIAL`：50项纯数据测试通过，并对已有SfM模型的固定6点做了精确变换正/负对照。它补上完整索引记录及定义式比较，不实现 AlgT'/AlgSo，不恢复尺度或照片几何，不是2393点全模型的验收。

## 三个接口的区别

| 接口 | 实际含义 |
|---|---|
| 旧 `kozlov_body_code` | 数值策略计数和去重量化比值摘要，不保存完整索引；保留兼容性 |
| 旧 `is_a_equivalent` | 浮点仿射拟合及点双射几何oracle，不比较索引代码 |
| 新 `indexed_code3d.indexed_body_code` | 完整保留每一合法四元组对、标签及精确比值；可验证全局标签双射 |

来源为 [S-1996](sources/SOURCES.md) PDF p.6/7、期刊325/326；本轮核对的PDF SHA256为 `1b831bcf92abe1cd638fd3855766e34eafb2846a8a6095ad81b409a40488e3ca`。完整定义、算术及资源规则见 [indexed-body-code-v1](contracts_indexed_code3d_v1.md)。新有限枚举和域检测是项目实现，不是作者命名的重建算法。

## 使用

在既有项目环境、项目根目录设置 `PYTHONPATH=src` 后使用：

```python
from kozlov_lab.stereo.official.indexed_code3d import (
    indexed_body_code, code_equivalence_for_mapping, find_code_equivalence,
)

points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,3,5)]
left = indexed_body_code(points, labels=[10,20,30,40,50])
order = [4,0,3,1,2]
right = indexed_body_code([points[i] for i in order], labels=[101,102,103,104,105])

known = {left.labels[source]: right.labels[row] for row, source in enumerate(order)}
checked = code_equivalence_for_mapping(left, right, known)
searched = find_code_equivalence(left, right, max_permutations=120)
print(checked['comparison_scope'], checked['status'])
print(searched['status'], searched['mapping'], searched['permutation'])
```

`code_equivalence_for_mapping` 的否定只针对给定映射，不能推断不存在其他对应；字段 `comparison_scope=supplied_bijection` 明确这一点。`find_code_equivalence` 才回答是否存在任一全局双射，范围字段为 `existential_bijection`。找到一个见证即可证明存在；必要条件不符可直接否定，否则只有穷尽才可否定。耗尽预算输出 `UNKNOWN`、`equivalent=None`，不能当 false。

mapping是标签对，按左侧输入标签次序返回；`permutation[i]`是右侧输入行索引。四元组字段则始终使用标签值的升序tuple，不是输入行号。每条比值使用同一个全局映射，不能逐条独立重排，也不能把比值排序后直接判等。

## 定义域、算术和规模

- 1–12个不同三维点；重复坐标或标签拒绝，不静默合并。整数/Fraction精确保留，至多binary64浮点按实际存储值转Fraction；不通过NumPy预先降低整数精度。坐标分子/分母各最多2048bit，标签为有符号64位整数。
- 完整记录上限默认10000，可显式设为1..250000；建码前计算实际需求，超限抛 `CodeBudgetError`，不输出“完整”的截断码。8个一般位置点有4830条，12点最多244530条。有限预算不是墙钟或RAM硬看守。
- 搜索最多8点，默认10000个排列、硬上限100000；9–12点只支持显式给定映射核验。改变预算须由调用方明确作出，不能失败后自动扩大。
- 分母精确为0的四元组禁用；这些四元组仍是合法零分子。分子、分母四元组必须不同，彼此可以相交。
- `DEFINED/complete=True` 表示记录完整。不共面4点只有一个四元组，合法记录为空。全体共面或不足4点为 `UNDEFINED_PLANAR/complete=False`，等价比较返回未知，不能冒充同一空码。
- `theorem_domain` 独立检查是否可放在一平面或两平行平面上；满秩3本身不够，立方体和四面体也是两平行平面的可覆盖情况。负域结论保留精确法向量/偏移/标签组见证。即使域通过，本API也只报告代码等价，`affine_equivalence=None`；没有自动升级成测量或几何重建结果。

通过工厂创建的结果不可变，禁止直接构造/`dataclasses.replace` 后冒充完整对象。内部域结构递归冻结，公开属性与 `to_dict()` 返回独立副本。

JSON中有理数用 `rational-hex-v1` 保存，以避免大整数十进制字数上限；还原方式：

```python
from fractions import Fraction
value = Fraction(int(record['numerator_hex'], 16), int(record['denominator_hex'], 16))
```

序列化hash识别当前编号呈现，不是重编号不变量。尚未提供任意外部JSON反序列化为可信代码对象的接口；应从已校验点坐标重新建码。

## 当前证据

[交付入口](../outputs/delegation/CODE3D_INDEXED/20261007T112500Z_main/HANDOFF.md)保存同次代码绑定、JUnit、命令、起止/退出码、源码快照和输入hash。唯一纯数据批次为50/0/0/0、exit0、外层约1.797秒；包含独立4×4有理行列式oracle、完整索引与零项、反射/有理仿射变换、重编号、预算、定理域、不可变性和超过14300bit分母的hex序列化用例。测试并未将几何oracle当真值。

唯一真实子集演示固定使用既有T03导出场景的原生point_id 1–6，得到15个四元组、210条索引记录。对坐标施加预先固定的可逆整数仿射矩阵和平移，再重排标签；402个候选后找到见证，并逐条核对完整记录。将一个坐标改动精确1/7的负对照穷尽720个双射后被拒绝，另一个直接索引枚举oracle同样拒绝。输入场景前后不变，外层约0.558秒。

这是“真实来源坐标上的受控算术对照”，不是从照片重新恢复了对应。没有对2393点整云建码，没有实测物理精度、Kozlov重建算法、研究收益或 CAD。

## 历史更正

当前UI、实验打印和函数说明已删除“is_a_equivalent使用带索引代码”的误称。旧 [phase6_gate.md](reports/phase6_gate.md) 保留历史字节：其中第187/386行一类说法不再作为当前证据；已有 [CORRECTIONS.md](../outputs/delegation/RESEARCH_CORRESPONDENCE/20261006T151710Z/CORRECTIONS.md) 记录早先对应判定修复。本轮未重跑历史实验，也未启动Streamlit页面，只有源码语法核验和标签修正；不把文字修正当UI运行验收。
