# 点级候选的相机几何接入

`scripts/evaluate_candidate_anchor_geometry.py` 消费已封印的点支持账本，不修改原SAM状态、掩膜、追踪身份段或模型。它用于回答：同身份段内已经定位的原始前景观测能否支持一个条件性三维锚点？没有相机的query仍未获几何验证。

参数：`--source` 指 `candidate_gate_20261017/real01/point_support`；`--seal-sha256` 由外部权威给定，不接受来源自行声明；`--coverage` 为原始逐帧覆盖输入；`--local` 为已保存局部原生模型；`--tracks-review` 为已验证的时序跟踪比较结果；`--output` 必须全新目录。

入口先检查前序交付及绑定，写PLAN，再用固定pycolmap4.2.1只读原生模型。图像ID、名称、frame ID、相机尺寸、参数及姿态逐项与绑定账本比较。原始掩膜像素实际回读；未确认/非strong/缺相机/跨身份段均保留排除理由。原COLMAP像素不重复减0.5。

`candidate_geometry_inputs.build_anchor_groups` 负责逐观测筛选与分组，`candidate_anchor_geometry.evaluate_anchor_group` 负责固定相机下的数值检查。两视图仅可输出TWO_VIEW_PROVISIONAL；三视图及以上必须满足严格重投影、角度和固定相机留一门。它们不证明物理身份、真实精度或相机标定。

输出包含PLAN、CAMERAS、GROUPS、RESULT、FINAL_INPUT_BINDINGS。已计算坐标仅存在于诊断JSON，不自动并入原点云、导出表面或授权MVS。缺相机可单独推进3D–2D注册，不能靠插值位姿或跨段合并补齐。本轮query全部缺相机；注册本身未执行。

实际封印执行与独立复核见 `outputs/delegation/PPP_VIDEO/20261014_object_first01/candidate_geometry_20261017/`。`execute_once.py` 的real01存在时拒绝再次运行；用于将来新输入时必须新建并冻结执行计划，不覆盖旧证据。
