# affine-subimage-v1

来源S-2025 PDF p.2 / 期刊147：часть为非空子集；подизображение为与某часть仿射等价的图像。本模块判定完整给定B是否存在可逆全平面仿射T使T(B)⊆A；它不判generalized futlyar嵌入、strict fits、粒度或完整群。

`find_affine_subimage_2d(source, target, budget=None)`接收完整非空dict。source至多12点，target至多32点；标签为64位int（拒bool），互异二维坐标只接受int/Fraction（拒float/bool），约分分子分母至多256位；重复坐标拒绝，无抽样或合并。非法输入抛ValueError。输出冻结AffineSubimageResult和TRUE时冻结AffineSubimageCertificate。

固定source仿射基。rank2枚举target全部有序非共线三点，Fraction构造唯一可逆仿射映射，复算全部source点的target归属。rank1固定两点，枚举target所有不同有序两点，使用`M=[v,Jv][u,Ju]^-1`构造可逆全平面扩展；完整共线source上任意真正嵌入与该扩展相同，所以不会漏解，target可以非共线。rank0用单位线性变换与平移。每个真实嵌入必命中固定source基对应的一组枚举anchors，因而穷尽后FALSE是完整否定。rank2最多32·31·30候选，rank1最多32·31候选。source点数超过target立即完整FALSE。

TRUE给完整外部输入的规范元组、2×2有理matrix、translation及完整标签单射mapping。`verify_affine_subimage_certificate`重新检查输入绑定、严格规范类型、非零行列式、全部点映像和完整单射，不调用search、solve或搜索映射辅助函数；伪造证书FALSE，预算耗尽UNKNOWN。它是数学见证而非来源签名。

共享`independent_cover.Budget`覆盖输入工作登记、基选择、候选、全部点复算及独立证书核验；默认2000000工作量/30秒。输入域完整验证在预算登记之前执行，且有固定大小/位数上限。每个实际算术步骤在固定profile内有限，但预算是逻辑工作与协作式墙钟检查，不是硬实时中断。最大见证有理位数8192，超界UNKNOWN。调用者传递共享Budget时不得重置它。TRUE/FALSE的complete表示数学谓词已完整决定；TRUE不枚举所有映射。UNKNOWN无证书且complete=False，不能解释为不存在嵌入。

验收独立oracle枚举所有完整单射，用仿射坐标恒等式检查全部点，覆盖rank0/1/2及正负例；证书核验在禁用search/solve/map后仍成功。验证仅涉及声明完整的二维有理输入，不证明上游数据完整，不构成RC06/RC07全项通过。
