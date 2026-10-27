# 单点连续视频诊断

入口 `scripts/trace_single_object_point.py` 从已绑定的 photometric run 中选择一条实际保存的观测，复用连续原始帧缓存追踪。它用于检查跨 ID 外观候选，不生成三维点，也不改变旧匹配结果。

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/trace_single_object_point.py run `
  --source-run '<photometric_selection/real01 的绝对路径>' `
  --supplement '<crossid_region_selection/real01/regions/SUPPLEMENT.json 的绝对路径>' `
  --seed-index 81 --seed-track 39285 --target-index 85 --target-track 41532 `
  --frames 64 --output '<尚不存在的新输出目录>'
```

先确定起点、比较目标和帧数，再执行一次。输出包括 INPUTS、PLAN、TRACE、RESULT、逐阶段进程记录、交互网页、运动 GIF、联系表与首次报警局部图。原始图像哈希、时间戳和采样帧的 RGB 身份均核对。比较目标只用于显示和事后距离计算，绝不传入光流搜索。

沿用原有 `photometric_step`：21 像素窗口、3 级金字塔、1 像素往返门、15 像素原图块及原光度门限。每一步使用相邻原始帧。没有重新播种、插值、邻点恢复或丢帧后直接跳到目标图。

- 绿色：指定起点，或截至当前连续图像检查通过。不是物理身份真值。
- 橙色：第一次失败后的有效前向提议，仅供观察漂移。即使之后局部检查通过，仍不会恢复认证。
- NOT_TRACKED：没有可用前向提议，后续不补造坐标。
- 红叉：旧跨 ID 候选，只在它的原始帧显示。

细线是历次图像坐标历史，不是向当前物体表面重投影。GIF 每帧120毫秒用于慢放，原始采集时间逐帧保留。中间原始帧没有新的 SAM 分割，不会声称每帧点都在已验证物体内。失联后的提议与旧目标距离不能单独证明哪个对应正确。

方法依据：OpenCV 官方 [Lucas–Kanade 光流教程](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)。Kozlov 书籍所讨论的近似对应与整体一致性不等价于“局部光流连续便证明身份”；此诊断不作为论文算法或几何重建验收。
