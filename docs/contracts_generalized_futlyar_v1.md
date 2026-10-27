# generalized-futlyar-v1

来源：唯一登记表 `docs/sources/SOURCES.md` 的 **S-2025**，俄文原文 PDF p.13 / 期刊 p.158；本地抽文 `.sources/kozlov2025_three_approximations.txt` 的 page 13。核对 `docs/open_questions.md` Q4/Q6。原文给定 A、O、互位与 r，要求中心数不超过点数，r 不大于最小中心距，每点唯一最近中心且距离严格小于 r，以及所有归属块非空。

## API 与精确域

直接从 `kozlov_lab.kozlov2025.generalized_futlyar` 导入：

* `verify_generalized_futlyar_2d(points, centers, radius, max_work=262144)` 返回冻结 `FutlyarResult`。
* `verify_generalized_futlyar_certificate(points, centers, radius, certificate, max_work=262144)` 独立验证成功证书，返回状态字符串。

points 与 centers 是调用者声明完整输入的非空 dict；标签为至多64位的 int（拒绝 bool），坐标为恰好两个 int/Fraction。标签只在各自字典内唯一；允许点标签与中心标签相同。点集1..4096点，中心集1..32点且不多于点数。拒绝点坐标重复，不合并、不抽样。中心重合保留为 `UNSUPPORTED`。坐标与半径约分后分子/分母至多256位；拒绝 float、字符串、bool、非二维。r 的受支持输入域为正有理数，r<=0 为输入错误；这是显式 API 域限制，不给原文补定义。输入域错误抛 ValueError。

全部几何运算为 Fraction，比较平方距离，不开平方、不引入容差。两中心距恰为 r 可接受；最近点距恰为 r 不可接受；等距最近不可接受。中心不必为 A 的子集，本模块不计算或验证椭圆中心。

## 状态和预算

* `TRUE`：全部条件通过，给出完整分块和证书。
* `FALSE`：给定互位失败，reason 与 witness 给出精确反例（中心距、点距/等距或空块）。不等于其他互位不存在。
* `UNKNOWN`：全部点到中心的 n*m 次及 m*(m-1)/2 次中心间平方距离所需预算超过 max_work；不执行部分判断，不报 FALSE。
* `UNSUPPORTED`：m=1 或中心重合，保持 Q4 未定边界。

max_work 为0..262144的 int；先完整规范化和验证输入域，再识别未定边界，再预检距离预算。固定数量及位数限制同时约束规范化、算术与证书大小；此为逻辑工作量上限，不是硬墙钟超时保证。证书独立核验使用同一预算规则。

## 证书独立核验

冻结 `FutlyarCertificate` 保存规范排序的全部点/中心、精确 r、按中心标签排序的所有非空块、每点到所有中心的平方距离、每对中心平方距离及版本号。核验器接收完整外部输入，逐项重算平方距离、严格唯一归属、半径条件和非空块，校验全部标签/成员/缓存距离，拒绝缺点、重复成员、错归属、改距离、改半径、改版本与非规范类型。核验器不调用判定器；测试禁用判定器后仍可核验。证书内容可人工按有理算术独立检查。模块本身不执行文件IO；另已提供严格JSON CLI及研究页面，见`KOZLOV_EXACT_RESEARCH_USAGE.md`和`EXACT_RESEARCH_UI.md`。

输入完整性依赖调用者确实提供整个 A/O；本模块无法证明上游未丢失像素或观测。证书核验只是数学见证核验，不是数字签名或来源认证。

## 旁支范围

现有 `independent_cover.py` 检查给定独立覆盖，明确排除 futlyar；此模块实现其尚缺的另一谓词。未改 public init 或旧算法，未接入照片到 CAD 主调用链。此功能不定义 primary 半径，不搜索仿射互位，不解决 Q6 稳定性/最优 fits，不证明完整群、粒度、эталон，也不构成完整 RC06 通过。来源已落盘及小例通过不替代全产品验收。
