# 自动分区与原生掩膜消费诊断

入口 `scripts/run_region_mask_review.py` 自动串联候选域准备、三臂mask消费实测、四臂同相机分区深度对照。当前实验档限定已标定的12视图子集和目标840，不宣称是任意物体自动解剖分割。RGB、相机、稀疏模型、遮挡物保持不变；区域由既有世界坐标point ID及真实观测驱动，不按固定屏幕区域移动，也不借完整度强造未知位置。

```powershell
& C:\Python313\python.exe -B -X utf8 scripts/run_region_mask_review.py `
  --source-plan outputs/delegation/SHAPE_REPAIR/20261025_temporal01/surface01/plan.json `
  --subset-root outputs/delegation/SHAPE_REPAIR/20261025_temporal01/subset01 `
  --output <从未存在的新目录>
```

历史plan不修改。重新绑定只允许实现代码变化，图像、相机、工具、预算和选项不符即拒绝。每个native工作目录全新、不复用depth缓存、不重试；2线程，单进程树8GiB提交量，逐臂180秒和1GiB输出采样停止阈值。三臂诊断600秒、四臂900秒，回收有界。原生消费未被正向反例验证则不启动四臂。

三臂：DISABLED和RECT使用相同局部禁止mask，仅ignore标签-1/0不同；BLACK为目标全黑。必须有可比较的正深度观测，并且RECT禁止内部0、外部仍有深度，BLACK有认证深度图且全0；缺图或监督失败保留UNKNOWN。仅证明此CPU/reference深度路径的mask响应，不证明target NCC patch掩膜、部位身份、几何质量或历史运行。

四臂：WHOLE原物体mask与G0/G1/G2候选域，均保留12视图和原RGB。无种子的全黑区域是UNKNOWN，不是物体不存在。区域重叠只统计2%深度冲突，不平均，不选胜者，不拼mesh。每视图分别记录共同有效、只在whole有效、只在region有效及双方未知；同源估计相符不是独立精度。

首次probe01因运行期文件链接信息异常停止，保留UNKNOWN。Windows真实删除/替换短fixture复现零链接数；监督器新增一次metadata刷新，只有重新得到普通文件、单链接且有效身份，或明确ENOENT，才允许继续。持续未知、真实多链接、目录/权限/reparse异常仍停止；strict前后检查不变。该机制修复不能追认probe01，也不说明其历史异常的具体链接数。

参考：本地 `book_code_audit_20261015/SOURCE_REVIEW.md` 仅用于限制Kozlov对应/投影条件；实际工程掩膜行为按[固定OpenMVS v2.4.0](https://github.com/cdcseacave/openMVS/tree/v2.4.0)源代码与本机正负对照判断，分区不是论文已证明的完整重建定理。
