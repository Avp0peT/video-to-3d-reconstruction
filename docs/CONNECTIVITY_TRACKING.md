# 自动检查局部模型连接与径向参数

输入为已完成 `select_geometry_tracks.py` 的独立比较目录。入口复用相同的真实TRACKS、图像和mask，不重新跟踪、分割、解码，不把跨ID点强行合并。输出目录必须全新。

```powershell
& 'outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe' -B -X utf8 scripts/connect_object_tracks.py run --control <evidence_selection/real01绝对路径> --output <全新输出绝对路径> --offline-evidence <验证记录绝对路径>
```

程序自动完成来源绑定→候选图审计→旧模型只读审核→两组原生求解→逐帧三列报告。每组原生只运行一次；不自动调参重试。使用Windows Job Object，单组240秒/8GiB提交量，输出1GiB；完成与回收状态分别记录。

| 臂 | 候选图 | 径向k | 焦距 |
|---|---|---|---|
| 保存的geometry_evidence控制 | 保存样本间距≤3，≥15共同真实ID | 自由估计 | 自由估计 |
| all_shared_tracks | 全部图像对，≥15共同真实ID | 自由估计 | 自由估计 |
| all_shared_tracks_zero_radial | 相同全对图 | 冻结为0 | 自由估计 |

三个臂均允许至少两次实际观测和两视三角化；输入坐标、track身份、RANSAC/BA数值门、4px与1.5°一致性门不变。零径向臂同时关闭BA及绝对位姿的extra参数优化；有效局部/全局BA与mapper选项一并核对，输出binary的k必须仍0。共享camera ID仅表示一个分量内共享，独立分量各有自己的相机估计。

**k=0是诊断假设，不是标定真值。** 焦距初始值仍为1.2×图像最长边，不将其当已知焦距。新增实际观测域诊断：r²=(Xc/Zc)²+(Yc/Zc)²，径向scale=1+k·r²、径向导数=1+3k·r²；报告有限/无效值、分布、非正次数。只记录，不增加剔除门、不因|k|大就断言画面折叠，不将局部导数正常当全图几何有效。

全对候选图只能连接已经有共同候选ID的图像。缺失共同ID的断裂不会因此补齐；图审计包含孤立帧并区分候选边、验证边和原生模型分量。报告逐帧保持同一原始分量用于原生点与一致性点的对照，不跨分量累加成完整物体。

来源：书印刷62页强调候选对应之后整体复核，但不提供本视频内参；本脚本是工程几何对照，不称Kozlov原始算法。[COLMAP固定内参说明](https://colmap.github.io/faq.html#fix-intrinsics)指出自标定可能退化且可按参数组冻结；[相机模型说明](https://colmap.github.io/cameras.html)用于理解模型自由度。当前运行接口以固定pycolmap4.2.1实测选项为准。
