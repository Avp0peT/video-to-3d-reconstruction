# 自动候选排除与点级支持

入口为 `scripts/apply_candidate_point_support.py`。它消费已封印的边界/局部运动证据和相同来源的实际 TRACKS，产生独立点级账本；不需要逐帧手选候选、不改原始SAM manifest。

```text
python -B -X utf8 scripts/apply_candidate_point_support.py --boundary <boundary_task/real01/boundary_support> --boundary-seal-sha256 <经复核的SHA256SUMS.txt摘要> --tracks-review <temporal_review/real01/comparison> --output <不存在的新目录>
python -B -X utf8 scripts/report_candidate_point_support.py --source <上述结果目录> --output <不存在的报告目录>
```

先验证边界包完整封印、依赖hash、帧ID、原mask字节及逐点包含关系。来源限定当前双片段协议49..62和77..90，不能宣传为任意视频生产入口。内核策略不按具体帧号或c01名称写规则。

只有原NEEDS_SELECTION参与派生选择。遗漏可信正点的候选记冲突。参考轮廓外见证不是背景真值：任何仍兼容正点的候选出现外见证冲突，整帧保持REFERENCE_CONFLICT_HOLD。正点/外点缺失、候选并列时不选择。唯一兼容者只得到PROVISIONAL_POINT_SUPPORT_ONLY，不提升整张mask。边缘、面积和SAM分数不决胜。

`POINT_SUPPORT_LEDGER.json` 保存派生选择、精确支持点及原候选binding；`MASK_SELECTION_LEDGER.json`保持空。每个片段的`*_POINT_ANNOTATIONS.json`逐行保留来源标签、原身份段、COLMAP实际坐标和额外点支持；不会替换TRACKS。只允许原有强观测且OpenCV查询坐标等于float32(COLMAP xy-.5)。未重新跟踪、不插值、不重建身份段、不新增观测、不把target_verified设true。没有被支持的行仍为UNCHANGED。

原几何入口要求原始mask、已验证前景和相机；此旁路没有改变它。所有新点支持的geometry_eligible仍为false，待独立点级几何接入检验相机、视差、重投影与同身份段。source_declared_geometry_eligible仅保存来源显式字段（可能null）；source_original_foreground不是最终几何通过。

可见图像边缘与参考外点只能提供条件性检查，不能证明整张轮廓正确，也不能断言镜头/焦距是错误的唯一原因。三维、孔洞补全、CAD和实际尺度未由此入口完成。
