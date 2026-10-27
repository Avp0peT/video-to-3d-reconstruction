# 按观测与几何证据准入短轨迹

此实验入口复用已完成光照跟踪的原始观测。旧规则要求轨迹至少出现在三个**不同的保存样本**，并非三个连续原始帧。新入口不重新跟踪、分割、解码或补造观测。

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/select_geometry_tracks.py run --control <photometric_selection/real01绝对路径> --output <全新输出绝对路径> --offline-evidence <测试记录绝对路径>
```

命令自动核对来源与旧实现绑定，保存计划，对旧模型只读核查，执行两组原生重建并生成三列同帧报告。每个新原生组只运行一次，失败不调参重试；Windows Job Object 限每组240秒、提交量8GiB，任务输出限1GiB。输出存在即拒绝。

| 方案 | 每条轨迹保存样本准入 | ignore_two_view_tracks |
|---|---:|---|
| 已保存光照跟踪对照 | 至少3 | true |
| two_sample_candidates | 至少2 | true |
| geometry_evidence | 至少2 | false |

两次真实观测是进入几何求解的最低条件，不是正确三维的保证。保持相同图像、mask、测量坐标、ID、+0.5像素转换、共享SIMPLE_RADIAL相机及数值阈值。配对仍限相邻3个保存样本；不是全视图图，也没有宣称所有两观测轨迹都实际提交匹配。原生选项逐项落盘，独立分量不拼接。

每个模型保存原始binary/text/PLY及全部原生点，另行核查所有真实模型观测：有限XYZ、严格正深度、最大重投影误差≤4px、最大有效三角化角≥1.5度。有效角为min(theta,180−theta)，拒绝同心及近共线退化。通过的两视点标`two_view_provisional`，多视点标`multiview_consistent_not_accepted`，未通过者留存原因，不删除原点。检查使用拟合该模型的相同观测，不是留出测试；重复纹理、镜面高光和错误匹配仍可能通过。

报告逐帧显示：实际入库候选、验证内点、原始模型观测、冻结模型一致性。模型两层使用同一个最强原始分量，不跨分量拼凑。8×8占格率表示mask内图像支持范围，不代表物体三维表面覆盖率。公共默认建模入口尚未更换，用户可复用此自动比较入口。

依据：书籍印刷61–62页的点对应及整体一致性思想用于解释校验层次，未将它称为照片透视下的Kozlov算法。COLMAP官方FAQ说明默认忽略两视轨迹，允许它们可增加约束、少数场景可能改善稳定性；不保证精度或完整模型：[官方说明](https://colmap.github.io/faq.html#increase-number-of-matches-sparse-3d-points)。运行时以固定pycolmap4.2.1的实际选项为准。
