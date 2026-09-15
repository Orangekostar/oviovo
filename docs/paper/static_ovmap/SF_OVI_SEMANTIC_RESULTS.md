# Cached SF → OVI semantics

科学状态：**COMPLETE_NO_NET_GAIN**；六条件完成，未建立跨场景有效性。

固定 O-only 几何、owner、注册表与原分数字符串；仅改OVI类别。所有数值为百分数。

## A：配对性能

| Run | Condition | Role | uAP / AP50 / AP25 | mIoU / mAcc | Subset entry / exit |
|---|---|---|---|---|---|
| fp32_primary | AT_O_AREA | REUSED_REFERENCE | 20.644/39.323/40.990 | 36.556/41.980 | 0/0 |
| fp32_primary | OD_E1_SEMANTIC | REUSED_REFERENCE | 20.644/39.323/40.990 | 36.556/41.980 | 0/0 |
| fp32_primary | ST_A_NATIVE_TOP1 | NEW_PAYLOAD_EVALUATION | 15.367/30.064/31.731 | 31.544/34.723 | 0/0 |
| fp32_primary | ST_B_SF_TRANSFER | NEW_PAYLOAD_EVALUATION | 18.213/29.758/31.425 | 31.718/36.486 | 2/1 |
| fp32_primary | ST_C_SF_SUPPORT_GATE | NEW_PAYLOAD_EVALUATION | 18.213/29.758/31.425 | 31.718/36.486 | 2/1 |
| fp32_repeat | AT_O_AREA | REUSED_REFERENCE | 20.644/39.323/40.990 | 36.556/41.980 | 0/0 |
| fp32_repeat | OD_E1_SEMANTIC | REUSED_REFERENCE | 20.644/39.323/40.990 | 36.556/41.980 | 0/0 |
| fp32_repeat | ST_A_NATIVE_TOP1 | REUSED_IDENTICAL_PAYLOAD | 15.367/30.064/31.731 | 31.544/34.723 | 0/0 |
| fp32_repeat | ST_B_SF_TRANSFER | NEW_PAYLOAD_EVALUATION | 18.213/29.758/31.425 | 31.718/36.486 | 2/1 |
| fp32_repeat | ST_C_SF_SUPPORT_GATE | NEW_PAYLOAD_EVALUATION | 18.213/29.758/31.425 | 31.720/36.487 | 2/1 |

全部条件的独立类别无关 Canonical AP75 保持16.842%，不宣称几何改善。活动OVI重叠掩码与unique掩码逐项相等，只评估一次；完整身份键相同的payload明确复用。新增图像/文本/2D/3D推理均为0。

## B：建议与采用漏斗

| Run | Method | Receivers / cohort | Valid / same-label | Proposed / adopted changes | No-suggestion abstentions |
|---|---|---|---|---|---|
| fp32_primary | ST_A_NATIVE_TOP1 | 64/26 | 26/18 | 8/8 | 38 |
| fp32_primary | ST_B_SF_TRANSFER | 64/50 | 50/26 | 24/24 | 14 |
| fp32_primary | ST_C_SF_SUPPORT_GATE | 64/50 | 50/26 | 24/23 | 14 |
| fp32_repeat | ST_A_NATIVE_TOP1 | 64/26 | 26/18 | 8/8 | 38 |
| fp32_repeat | ST_B_SF_TRANSFER | 64/51 | 51/26 | 25/25 | 13 |
| fp32_repeat | ST_C_SF_SUPPORT_GATE | 64/51 | 51/26 | 25/23 | 13 |

S-A移除整组旧采用门槛，不是margin-only消融。S-C仅对S-B同一供体建议应用2/3支持门控，不另选供体；单个正权重供体可通过，不构成独立确认。全部拒绝原因、支持分布、重复合并、单供体通过及成本见机器漏斗。

## C：固定几何下的正确性（strict >.5）

| Run | Method | Changed transitions | Useful adopted / rejected | Harmful adopted / prevented |
|---|---|---|---|---|
| fp32_primary | ST_A_NATIVE_TOP1 | {'wrong_to_wrong': 1, 'right_to_wrong': 4, 'unresolved_geometry': 3} | 0/0 | 4/0 |
| fp32_primary | ST_B_SF_TRANSFER | {'right_to_wrong': 10, 'wrong_to_right': 4, 'unresolved_geometry': 6, 'wrong_to_wrong': 4} | 4/0 | 10/0 |
| fp32_primary | ST_C_SF_SUPPORT_GATE | {'right_to_wrong': 10, 'wrong_to_right': 4, 'unresolved_geometry': 5, 'wrong_to_wrong': 4} | 4/0 | 10/0 |
| fp32_repeat | ST_A_NATIVE_TOP1 | {'wrong_to_wrong': 1, 'right_to_wrong': 4, 'unresolved_geometry': 3} | 0/0 | 4/0 |
| fp32_repeat | ST_B_SF_TRANSFER | {'right_to_wrong': 10, 'wrong_to_right': 4, 'unresolved_geometry': 7, 'wrong_to_wrong': 4} | 4/0 | 10/0 |
| fp32_repeat | ST_C_SF_SUPPORT_GATE | {'right_to_wrong': 10, 'wrong_to_right': 4, 'unresolved_geometry': 5, 'wrong_to_wrong': 4} | 4/0 | 10/0 |

仅唯一几何对应可判对错；ambiguous/unmatched保持显式，不采用多数像素定义对象正确性。全部接收者与.25/.5/.75阈值均在账本中，含原本正确者。

## D：真实发布版对象结果（strict >.5）

| Run | Method | Gained | Lost | U00 added recovered | Not recovered |
|---|---|---|---|---|---|
| fp32_primary | ST_A_NATIVE_TOP1 | [] | [19002, 19005, 25000, 48000] | [] | [5000, 5005, 10004, 15000, 17000, 24000, 33000] |
| fp32_primary | ST_B_SF_TRANSFER | [10004, 15000, 17000, 33000] | [4000, 6000, 9006, 19002, 19005, 19006, 25000, 39005, 48000, 51000] | [10004, 15000, 17000, 33000] | [5000, 5005, 24000] |
| fp32_primary | ST_C_SF_SUPPORT_GATE | [10004, 15000, 17000, 33000] | [4000, 6000, 9006, 19002, 19005, 19006, 25000, 39005, 48000, 51000] | [10004, 15000, 17000, 33000] | [5000, 5005, 24000] |
| fp32_repeat | ST_A_NATIVE_TOP1 | [] | [19002, 19005, 25000, 48000] | [] | [5000, 5005, 10004, 15000, 17000, 33000] |
| fp32_repeat | ST_B_SF_TRANSFER | [10004, 15000, 17000, 33000] | [4000, 6000, 9006, 19002, 19005, 19006, 25000, 39005, 48000, 51000] | [10004, 15000, 17000, 33000] | [5000, 5005] |
| fp32_repeat | ST_C_SF_SUPPORT_GATE | [10004, 15000, 17000, 33000] | [4000, 6000, 9006, 19002, 19005, 19006, 25000, 39005, 48000, 51000] | [10004, 15000, 17000, 33000] | [5000, 5005] |

历史7/6新增、5/4最佳OVI形状足够但类别错、2/2没有足够OVI形状均已复现；完整headroom检查同时覆盖所有baseline misses。GT-aware计数只是诊断，不是输出方法或AP上界。

S-B与S-A证据源和目标覆盖均不同；common_target_comparison给共同原生owner逐对象对照，不报告子集AP为全景结果。两个缓存是同一开发场景敏感性检查，不是独立场景、盲验证或显著性证据。

证据：[performance](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/performance.json)、[funnel](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/intervention_funnel.json)、[correctness](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/label_correctness_and_gate.json.gz)、[routing](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/donor_routing.json.gz)、[objects](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/object_outcomes.json.gz)、[headroom](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/diagnostic_headroom.json)。
