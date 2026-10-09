# Disagreement query results

固定筛选完成：10 条方法路径、4 个已曝光场景、40 条完整输出结果、20 个有序双场景池。三个预定扩展对比均失败，按协议保留 DQ01_G1，部署 N0_UNCHANGED。完整 8/18 场景回归 NOT_RUN_NO_SCREEN_SIGNAL；严格与材料目标未在完整回归中验证。

## 表 1：四场景开发筛选（%，不是 Replica8/CF18）

| method | replica_probe2_apall_percent | replica_probe2_ap50_percent | replica_probe2_ap25_percent | replica_probe2_miou_percent | replica_probe2_macc_percent | cf_probe2_apall_percent | cf_probe2_ap50_percent | cf_probe2_ap25_percent | cf_probe2_miou_percent | cf_probe2_macc_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQ00_D2 | 14.6297 | 32.7658 | 34.6176 | 26.4052 | 33.5234 | 13.1069 | 30.9147 | 31.7879 | 26.2255 | 33.8615 |
| DQ01_G1 | 14.6297 | 32.7658 | 34.6176 | 26.3945 | 33.5238 | 13.1069 | 30.9147 | 31.7879 | 26.2621 | 33.9024 |
| AREA_MEAN | 14.3190 | 32.8668 | 34.7186 | 27.2357 | 36.1190 | 11.6732 | 27.6889 | 31.7431 | 24.1612 | 32.9453 |
| COVERAGE_MEAN | 15.2414 | 35.6445 | 37.4964 | 28.6156 | 37.5772 | 11.6732 | 27.6889 | 28.5173 | 23.7318 | 31.4365 |
| VERIFY_MEAN | 13.1181 | 30.7456 | 35.6277 | 27.2952 | 35.9318 | 11.6732 | 27.6889 | 33.3560 | 24.5266 | 33.3415 |
| DISAGREEMENT_MEAN | 13.1181 | 30.7456 | 35.6277 | 27.2951 | 35.9318 | 11.6732 | 27.6889 | 33.3560 | 24.5266 | 33.3415 |
| QS_AREA | 14.3190 | 32.8668 | 34.7186 | 27.2356 | 36.1190 | 11.6732 | 27.6889 | 28.5173 | 23.7441 | 31.4247 |
| QS_SUPPORT | 13.4772 | 31.3516 | 33.2035 | 26.4508 | 35.2859 | 11.6732 | 27.6889 | 28.5173 | 23.7441 | 31.4247 |
| QD_AREA | 13.1181 | 30.7456 | 35.6277 | 27.2951 | 35.9318 | 11.6732 | 27.6889 | 30.1302 | 24.0400 | 31.6963 |
| QD_SUPPORT | 13.1181 | 30.7456 | 35.6277 | 27.2951 | 35.9318 | 11.6732 | 27.6889 | 30.1302 | 24.0400 | 31.6963 |

## 表 2：匹配机制对照（百分点）

| candidate | reference | replica_probe2_apall_delta_pp | cf_probe2_apall_delta_pp | second_view_divergence | selected_owner_class_differences | wrong_to_right | right_to_wrong | net_unique_GT50 | net_unique_GT75 | extension_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DISAGREEMENT_MEAN | COVERAGE_MEAN | -2.1233 | 0.0000 | 54 | 15 | 3 | 6 | -3 | -1 | False |
| QS_SUPPORT | COVERAGE_MEAN | -1.7642 | 0.0000 | 0 | 6 | 0 | 2 | -2 | 0 | False |
| QD_SUPPORT | DISAGREEMENT_MEAN | 0.0000 | 0.0000 | 0 | 5 | 1 | 1 | 0 | 0 | False |
| QD_SUPPORT | QS_SUPPORT | -0.3591 | 0.0000 | 54 | 9 | 2 | 3 | -1 | -1 | — |

## 表 3：覆盖与成本

| method | selected | query_eligible | required_FULL_success | updated | logical_FULL_reads | logical_probe_heads | QD_fallback_count | sign_conflict_count | duplicate_records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQ00_D2 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DQ01_G1 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| AREA_MEAN | 64 | 64 | 64 | 14 | 128 | 0 | 0 | 0 | 0 |
| COVERAGE_MEAN | 64 | 64 | 64 | 12 | 128 | 0 | 0 | 0 | 0 |
| VERIFY_MEAN | 64 | 64 | 64 | 20 | 128 | 0 | 0 | 0 | 0 |
| DISAGREEMENT_MEAN | 64 | 64 | 64 | 20 | 128 | 247 | 0 | 34 | 0 |
| QS_AREA | 64 | 64 | 64 | 15 | 128 | 0 | 0 | 0 | 0 |
| QS_SUPPORT | 64 | 64 | 64 | 16 | 128 | 0 | 0 | 0 | 0 |
| QD_AREA | 64 | 64 | 64 | 18 | 128 | 247 | 0 | 34 | 0 |
| QD_SUPPORT | 64 | 64 | 64 | 18 | 128 | 247 | 0 | 34 | 0 |

| phase | category | FC_encoding_attempts | FULL_pool_attempts | PROBE_pool_attempts | pool_heads | worker_wall_seconds_including_load | model_load_seconds | peak_allocated_GiB | peak_reserved_GiB | standalone_policy_cold_timing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| engineering_anchor | ENGINEERING | 0 | 4 | 7 | 11 | 13.2332 | 7.1780 | 1.3566 | 1.4746 | False |
| screen_anchor | SCIENCE | 12 | 62 | 240 | 302 | 33.2182 | 9.0200 | 1.8930 | 2.1992 | False |
| screen_second | SCIENCE | 23 | 148 | 0 | 148 | 35.4655 | 7.1539 | 1.8930 | 2.2227 | False |

| method | support_equal_mean_weights_atol1e12 | support_equal_area_weights_atol1e12 | support_exact_mean_pnew | support_exact_area_pnew | support_exact_mean_pfinal | support_exact_area_pfinal | whole_scene_output_equal_mean | whole_scene_output_equal_area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QS_SUPPORT | 0 | 12 | 0 | 12 | 0 | 12 | 1 | 3 |
| QD_SUPPORT | 0 | 1 | 0 | 1 | 0 | 1 | 2 | 3 |

权重相等以归一化权重绝对容差 1e-12 判断；分布与整场输出相等为严格身份/逐元素相等。权重/分布分母为 64 对象，整场输出分母为 4 场景。

科研联合采集新增 FC 图像编码 35 次、区域池化/head 450 次；工程对象额外 4 FULL + 7 子区域 = 11 次池化/head，新增图像编码 0。筛选 B 完全复用已获取证据。采集阶段耗时包含缓存读取和模型加载，不能作为各独立策略的冷耗时或在线 FPS。条件冷测未触发。

DISAGREEMENT 对 VERIFY 的第二视角选择分歧 4/64；回退 0/64；真实 tile 符号冲突 34/64。词表诊断 40509 条嵌套记录中有 3484 条概率差符号相对二类词表翻转，固定缩放分数差翻转 0；这些记录并非独立对象。几何分区、类无关匹配和恢复区域保持不变，无法据此宣称实例几何改善。

全部五项逐场景/逐类、GT50/75 唯一匹配、原始评分事件、逐对象权重、概率和诊断见 [结果包](../../../artifacts/static_ovmap/disagreement_query_v1/result_store.json) 与 [bundles](../../../artifacts/static_ovmap/disagreement_query_v1/bundles/manifest.json)。固定 G1 参考中有 8/64 类别映射未定义；纠错统计不把它们记成错误或从选择分母剔除。

原始评分器一致性检查实际在筛选之后补齐；所有预测、查询和数值算子保持此前冻结版本，未根据该检查改变方法。两处真实 G1 评分和四个基线有序池一致性均通过。
