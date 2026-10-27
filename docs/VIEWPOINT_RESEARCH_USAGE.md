# 原始照片的跨视角研究诊断

入口 `scripts/audit_viewpoint_research.py` 消费 `audit_photo_parallel.py` 生成的 `INPUT.json`，只读已注册模型的原始二维观测。当前至多 12 个视角、每对至多 4096 条轨迹；不重跑 SAM3、COLMAP 或 OpenMVS，不改变轨迹、相机、像素或网格。

```powershell
python -B -X utf8 scripts/audit_viewpoint_research.py --prior-input <原始像素审查目录>/INPUT.json --output <新的输出目录>
```

两项研究结果分别保存：

- **论文精确域**：1996 年 AlgT′ 所用的给定对应平行投影输入条件，复用 `given_parallel_correspondence` 的精确秩证书与独立核验。名义最前六点的拒绝保留。另按 ID 升序取两幅图各自坐标互异的首六点，记录跳过 ID、碰撞侧和坐标。精确不相容时不调用 AlgT′ 构造，更不生成代理照片。合成精确输入上实际调用 AlgT′，并以预定错误对应作负控制。
- **工程近似诊断**：原始四维像素的仿射超平面，与使用全部训练点的归一化八点法 F 模型，均只拟合一次。留出点、循环错配、反序错配共用冻结模型。模型自由度、残差定义不同，不以两个 RMS 的大小宣布算法优劣。原图畸变没有消除，F 也只是工程对照。

共享任一侧完全相同坐标的轨迹用连通组划分，整组进入训练或留出。所有组按最小 point ID 排序，交替分组；不按残差挑点。原始轨迹可能是同一空间点的别名，也可能有错误，坐标重合本身不决定原因。分组仅防止精确重复穿过划分，没有删除原模型点。

`PLAN.json` 在拟合前保存所有图对、像素及 binary64 hex、point2D/point3D 身份、分组、六点选择、源字节哈希和实现绑定。各图对的 `SCORES.json` 保存模型参数、逐点残差、右侧来源 ID、负控制实际位移、固定点及相同像素别名数量。全槽计数与真正改变坐标槽的配对计数分开，不称检测率。

输出 `DIAGNOSTICS_NOT_GEOMETRY_ACCEPTED` 表示诊断完成。原始对应来自已有 COLMAP，视角与轨迹相互相关；留出不是独立真值。精确拒绝不能单独归因为角度过大、错配、噪声或透视；工程残差变大只能证明对指定扰动敏感，不能认证原对应。未建立自动删点阈值、局部错配召回率或三维精度。

PPP 首次运行：`outputs/delegation/PPP_VIDEO/20261014_object_first01/research_viewpoints_20261014/`。其中 `exact_raw` 保留首次名义六点被重复坐标拒绝的证据，`all_pairs` 为全部 10 组视角的冻结实验。旧 PPP 网格和报告不改写。
