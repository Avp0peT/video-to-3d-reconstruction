# given-m-compatibility-v1

S-2025 PDF p12 / 期刊157原文“существует хотя бы одно из изображений группы”明确reference必须群成员。量词为∃同一reference A∃同一个primary F∀群成员B。此模块验证给定充分见证，不实施一般兼容性、granularity、etalons或完整群搜索。

完整环境1..7互异有理标签点。group为非空、成员标签互斥的非空部分列表，可以不覆盖环境；原始重复坐标拒绝，不按仿射后坐标合并身份。空group无法定义u=min成员点数，本profile明确拒绝。reference_index指定实际群成员；reference_parts为它的完整给定独立覆盖，m≥2；每个群成员给一个完整zero-error-fits证书（search TRUE的certificate字段亦可）或collinear-fits非零稳定互位与完整块包含证书。按schema调用相应独立核验器，未知schema拒绝。必须包括reference自身，不得遗漏、重复或交换成员。输入顺序保留为证书身份。

先独立核验共享primary，再核全部member exact fits证书。共享数学F由完整A外部坐标绑定/source摘要与标签、给定parts、精确centers及固定radius_squared确定。每份源几何证书独立复验；可以使用不同合法证明表示，但不能换A/parts/中心/R。成功M_COMPATIBLE_SUFFICIENT_WITNESS仅证明给定m和这个F容纳所有成员。INVALID_WITNESS不是一般不兼容，UNKNOWN不得向粒度传播。

`given_m_compatibility`只组装及调用独立`assess_given_m_compatibility_certificate`，不搜group或F。独立核验不调用search、fits/primary producer或center constructor。所有子核验用单一剩余2M工作/30秒预算并计费。需全环境外部输入，绑定群身份和所有成员source点；改m、漏成员、重复成员、单项有效但各自reference不同均拒绝。

2026-10-14：另接收`four-center-fits-v1`，调用其独立核验器；仍保留相同reference/F/全成员量词及完整环境≤7的边界。四中心非零示例属于分别给定reference与source的独立fits API；不能据此声称在≤7点环境中已形成两个互斥四点成员。自动搜索入口另见`contracts_finite_compatibility_search_v1.md`。

网络参考推进：另接收`circuit-fits-v1`的多中心强充分证书，仍按对应独立核验器裁决。该schema不改变以上完整环境基数界或量词，也不把下界未达到的候选升级为fits。
