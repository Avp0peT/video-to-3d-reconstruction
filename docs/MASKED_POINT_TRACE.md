# 逐帧真实掩膜约束的单点跟踪诊断

入口 `scripts/trace_masked_object_point.py` 对已冻结的单点实验生成独立的新结果。
当前输入适配器针对 NEW_VIDEO 的 raw555–618 区间：复用9张已有真实SAM掩膜，
只补分割54张未分割图像，保留raw576的历史NO_MATCH。旧结果不改写。
它是该区间的自动诊断入口，尚未替换通用建模流水线。

## 自动执行

在仓库根使用项目轻量Python调用（output必须尚不存在）：

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 `
  scripts/trace_masked_object_point.py run `
  --old-run outputs/delegation/NEW_VIDEO/20261019_new01/single_point_selection/real01 `
  --sam-config outputs/delegation/NEW_VIDEO/20261019_new01/pipeline/segmentation/CONFIG.json `
  --output outputs/delegation/NEW_VIDEO/20261019_new01/masked_point_motion/new_run
```

`prepare`、`segment`、`finish`可分步调用；segment已留下启动记录时拒绝重复执行。
SAM仅一个批次、一个worker加载，不自动改提示词、选择候选、放宽资源阈值或重试。
沿用现有mmap SAM运行入口：12GiB worker提交量、7GiB整卡显存门，外层14GiB Job及950秒看守。
IO、跟踪、报告同样受Job监督。照片IO产生的新index与原decode_index以FRAME_MAPPING逐项绑定。

## 三路结果

- A：只读复制历史无逐帧掩膜的诊断。失联后的红色shadow是未经确认的位置。
- B：原光流及光度检验不变，加实际当前掩膜中心门。
- C：B基础上最多多一次LK初值候选。只用前两次连续通过的观测与真实PTS估计速度；
  预测位移上限96像素。原21×21窗口、3级金字塔和FB/光度阈值不变。

候选若相距超过2像素且都通过，记AMBIGUOUS；不择优隐藏歧义。
掩膜中心采用floor(x+0.5)和图像范围检查，patch15前景比例仅供诊断。
灰度计算不把掩膜外改黑。拒绝点不吸附到掩膜、不补点、不插值。
没有当前真实掩膜时停止该链。首次检验失败后不把后续shadow恢复为同一点。

## 解释边界

SAM掩膜也是估计，可能漏分或错分；位于掩膜内不能证明是原物理点。
运动预测初值可能改善搜索，不等于放宽匹配正确性，也不保证快速运动都可恢复。
逐帧结果记录所有原始候选、拒绝原因、初值和mask来源；网页只把accepted_xy画作新有效点。
不将本诊断的有效2D观测冒称三维、尺度或CAD验收。

方法参考：OpenCV 4.13 LK光流文档的`OPTFLOW_USE_INITIAL_FLOW`。
https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html
书籍的近似图形对应与整体一致性是研究边界参考；本次光流工程改动不是Kozlov定理实现。
