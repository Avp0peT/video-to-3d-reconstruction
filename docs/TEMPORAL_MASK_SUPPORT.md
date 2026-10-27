# 短时SAM缺失：完整轮廓与局部搜索范围分开

本轮针对已冻结NEW_VIDEO raw576–579缺口。原SAM保持NO_MATCH。
脚本自动选前后相邻的实际mask作为锚，并选择此前等长、全有mask的区间做隐藏中间掩膜对照。
原始图像、锚mask、代码、OpenCV来源先绑定，全部计算及测试由现有Windows Job监督。
不重复运行SAM，不手绘mask，不改原manifest或跟踪身份。

## 两种输出权限

1. `bridge_temporal_mask_gaps.py`：在前后锚之间逐相邻帧双向稠密光流；长边最多512，
   固定参数，无调参重试。target→source逆采样，source→target循环误差约束，valid随链传播。
   未知边界不作为背景真值。双向端点全图IoU、实际锚有效覆盖、逐帧一致性共同决定完整临时mask门。
2. `export_temporal_search_support.py`：只读同次保存的两方向mask与valid。
   完整门未通过时，仍可输出双向都支持的局部搜索候选，但其`full_mask_accepted`恒为false。
   不确定部分保留孔洞。仅用于后续搜点候选范围，禁止冒充SAM mask、MVS mask或3D观测接受证据。

当前局部导出只是生成独立结果；尚未接入通用建模消费者。
两端都有图像时才可离线双向复核，因此不是零延迟实时能力。

## 可重复自动命令

使用新的输出目录；不覆盖本次real01。两条命令没有人工mask修改步骤。

```powershell
$python = 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe'
& $python -B -X utf8 scripts/bridge_temporal_mask_gaps.py run `
  --source outputs/delegation/NEW_VIDEO/20261019_new01/masked_point_motion/real01 `
  --output outputs/delegation/NEW_VIDEO/20261019_new01/temporal_mask_support/new_run
if ($LASTEXITCODE -ne 0) { throw 'Temporal review failed' }
& $python -B -X utf8 scripts/export_temporal_search_support.py run `
  --full-run outputs/delegation/NEW_VIDEO/20261019_new01/temporal_mask_support/new_run `
  --output outputs/delegation/NEW_VIDEO/20261019_new01/temporal_mask_search/new_run
```

当前适配器仅接受恰好一个1–6帧、两侧有实际mask的缺口，并要求更早存在等长控制区间。
不满足就明确拒绝，没有跨长时间外推或未知区域播种。

## 本次结果

- 对照：570–575，仅两端mask参与传播，中间571–574原mask只用于评估；IoU 0.873–0.881。
- 实际：575→580端点全图支持IoU 0.577，反向0.542；实际端点前景valid约0.652/0.605。
  四帧完整mask全部未通过，不放宽门限。
- 局部：576–579分别有62196/62432/61980/61536像素满足双向一致和valid交集。
  占预测并集55.4%–56.6%；该分母不是人工真实物体面积，不称真实覆盖率。
- 原SAM仍60/64，不新增SAM成功数；原失联点没有被重新认证。
- 9项传播测试+8项局部语义测试全部通过，包含实际CPU合成平移和错误循环/小有效域反例。

书籍的ε近似对应与整体一致性是思想参考，不提供SAM漏检恢复保证。
本轮使用的是工程光流传播；OpenCV官方接口：
https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html

SAM3官方同时提供图像与视频入口，当前生产图像路径逐帧独立。视频对象记忆是后续方向，
本轮没有加载或验证该视频入口：https://github.com/facebookresearch/sam3
