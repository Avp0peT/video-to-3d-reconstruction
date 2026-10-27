# 窄缝自动定位与保护

当前状态：`NARROW_GAP_CANDIDATES_NOT_ACCEPTED`。规则保护已提供掩膜中的背景证据，不等同于精确分割或三维验收。

## 已实现的路径

- `integrations/video_masks/narrow_gap.py`：逐帧重新检测窄背景区域、封闭背景及未知邻域；无需手指坐标或物体类别模板。
- `integrations/video_masks/narrow_gap_review.py` + `scripts/protect_narrow_gaps.py`：读取已完成的 union 融合来源，重新核对原图、基础掩膜、支持掩膜与逐像素 OR，生成独立候选、宽度记录、对照图及邻帧慢放。
- `integrations/video_masks/boundary_refine.py`：现有边界细化候选评估已自动调用保护器。证据缺失、不一致或检查异常均拒绝候选，结果记录在 `gap_guard`。

这不是全时间线融合脚本的新默认模式，也没有自动向 MVS 发布候选。`default_masks/` 始终保留输入 union 原文件字节，候选只写 `proposals/`。后续几何验证应显式绑定候选清单，不能把目录存在当作可消费授权。

2026-10-24 后续接线：完整视频 union 与独立 union 融合现在默认追加有界的窄缝及原分辨率RGB复核，详见 `AUTOMATIC_MASK_BOUNDARY.md`。这改变了自动复核调用链，仍未把候选改为默认分割结果。

## 规则与适用边界

背景按 4 连通分类：已知像素通路到达图像边缘为 EXTERIOR；没有已知外连且邻接未知像素为 UNRESOLVED；其余为 ENCLOSED。本轮没有以 8 连通结论替代 4 连通，斜向细缝仍有像素尺度敏感性。

在水平、垂直与两个对角方向寻找前景两岸，要求两岸之间全部为有效背景。记录的是四方向中最小的前景像素中心跨度上界，并非真实最窄距离或三维物理宽度。最大窄缝跨度为物体图像范围的 4%，限制在 6–48 像素。小于 3 像素采样尺度的缝仍保留，但标为分辨率不足，不能作为已验证的三维空隙。

局部不确定余量为 `min(3, 0.2 * bank_span)`，本轮只输出诊断/操作上限，不自动改写 trimap，不把 UNKNOWN 当成背景。全 1 validity 仅表示本轮二值掩膜的整幅网格有效，不表示所有标签正确。

候选必须满足 `base <= candidate <= literal_union`。先阻止 union 新增前景覆盖受保护背景；若恢复这些像素后仍不能保住原背景通路或组件，整帧候选退回基础掩膜。每帧改动上限仍为 union 前景的 1%。保护器不挖除 base 前景来人为造缝。

因此，这一规则可以防止原有可见缝隙在融合中被填入；无法发现已经被 base 错标为前景的指缝，也可能保留 base 中错误的背景孔洞。源模型的 NO_MATCH / NOT_SAMPLED 等状态继续保留。

## 自动运行

在仓库根目录使用已有轻量环境，输出必须是新目录。单批最多 256 帧，算法墙钟上限 240 秒；本次只验证了下面 16 帧。批量拆分需保留同一源绑定与每批独立记录，不能无记录地重复执行旧输出。

```powershell
$py = '.\outputs\delegation\T02A\envs\20261002T190239Z_9b9b5ddb\venv\Scripts\python.exe'
& $py -B -X utf8 .\scripts\protect_narrow_gaps.py run `
  --source .\outputs\delegation\HAND_VIDEO\20261022_union_masks02 `
  --output .\outputs\delegation\NARROW_GAPS\new_review `
  --indices 84 85 86 87 88 325 326 327 328 329 330 510 511 512 513 514
```

`prepare` 可只冻结 PLAN；`execute --output <同一目录>` 执行一次。CPU 入口返回 0 仅表示候选生成及约束检查完成且没有改动上限拒绝，不表示精准分割或建模成功。真实任务可复用本轮 `execute_entry.py` 的 Windows Job 监督方式；上述直接 CLI 的内置时间检查不是原生调用的硬超时。

## 本次真实读回

入口：`outputs/delegation/NARROW_GAPS/20261024_gap01/index.html`。

- 16 帧，三个邻帧区间，CPU 重放一次；53 项联合测试通过。随后仅补强一条既有测试的统计断言，该例单独通过；没有再次执行终态 53 项整批，产品字节未因补充断言改变。
- 原 union 填入基础掩膜受保护背景 2,014 像素，候选为 0；被抹掉的基础背景组件 5 个，候选为 0。
- 总计阻止 5,578 个 union 新增前景像素，其中包含 8 帧整帧基础掩膜回退。不能将全部 5,578 像素称为已确认的指缝误分。
- 基础前景损失 0，候选新增于 union 之外的前景 0；默认 PNG 仍逐字节等于原 union。
- 独立验证重算了 OR、连通性、全部有限两岸宽度记录和源 hash。
- 现有网格为 576×1024，原视频编码分辨率为 720×1280。本轮未重新解码原分辨率、未重新分割、未验证跨帧缝隙身份、未重建新网格。

## 下一层验证

优先在原分辨率 RGB 上对这些自动定位区域检查边缘与遮挡，保留不确定状态；再用有正确相机位姿和可见性支持的多视图验证候选，最后将可靠背景约束传给三角面边缘/内部投影检查。掩膜通路保留不能直接证明三维手指已经分离。

参考顺序遵循书籍 → 官方资料 → 工程假设。已查项目内书籍摘录 `track_recovery_20261015/book_review/BOOK_REVIEW.md`：对应关系和联合证据思想可借鉴，但没有现成的窄缝精分割保证。形态学闭运算会填小孔，不能作为这类区域的无条件修复，参见 [OpenCV 形态学文档](https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html)。现有局部细化的背景参考为 [GrabCut 原论文](https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/)，本轮保护器本身没有调用新的分割模型。
