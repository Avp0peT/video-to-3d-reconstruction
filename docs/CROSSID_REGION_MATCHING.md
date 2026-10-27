# 跨轨迹编号的区域对应

入口 `scripts/match_object_regions.py`。本实验复用原视频、原 SAM mask 和保存的真实观测，自动完成局部区域匹配、原生几何验证、相机估计及图像对照。

前置是已经完成的 `connect_object_tracks.py` 对照目录。控制结果固定为其中的 `all_shared_tracks_zero_radial/native`；不重跑控制。

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/match_object_regions.py run --control 'outputs/delegation/NEW_VIDEO/20261019_new01/connectivity_selection/real01' --output '<一个不存在的新目录>' --offline-evidence '<本轮监督测试的ENTRY.json>'
```

保持双方相同的 ≥2 个真实已保存视图观测资格、全部 feature row 顺序、坐标与原 track ID。只对旧共有 ID 图中不连通、且样本序列间距 1..3 的画面对，用原始灰度区域寻找跨编号对应。真实采样间隔另记，1..3 并非连续原始视频帧。

区域描述使用现有 OpenCV SIFT，在保存的中心、固定尺寸与实测局部梯度方向计算；不重新检测点，不把 mask 外像素涂黑再提特征。完整支持域需在原 mask 内。该条件会损失轮廓附近的观测，逐帧明确记录“没有可计算描述”与“描述后未匹配”的区别。双向最近/次近比与互为最近邻消除明显歧义，但不能保证物理身份。原 same-ID 行优先，显式跨 ID 行仅追加，不合并原始编号；总行数达 15 的画面对才进入原生两视图验证。

固定求解器、零径向系数及自由焦距、原数值门，与控制一致。输出仍分为 raw 对应、原生内点、匹配图组件、独立重建模型；低残差与更多注册视图不代表正确外形。零径向不是标定，报告按模型列出实际焦距，防止注册数掩盖退化解。

每个阶段由 Windows Job Object 监督，匹配/求解各 240 秒，报告 120 秒；进程提交量上限 8 GiB、输出 1 GiB。新真实阶段各一次，无自动重试。失败或未确认回收即停止。源文件和实现前后 hash 复核。不会把独立模型拼为一个物体，不自动补洞，不改变生产默认标定方案。

依据层次：本地书稿印刷 61–62 页要求从局部候选继续检查整组一致性；这里提供工程区域描述与透视验证，不能声称已经实现书中全部投影前提。

- OpenCV 官方 SIFT 描述与匹配：https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
- COLMAP 相机内参固定及自标定问题：https://colmap.github.io/faq.html#fix-intrinsics

输出：`PLAN.json` / `INPUTS.json` / `regions/SUPPLEMENT.json` / `regions/FRAME_DIAGNOSTICS.json` / `native/PLAN.json` / `native/RESULT.json` / `native/GEOMETRY_EVIDENCE.json` / `GRAPH_AUDIT.json` / `SUMMARY.json` / `index.html`。原生 PLY 仅为分量级候选，仍未外形与真实尺度验收。
