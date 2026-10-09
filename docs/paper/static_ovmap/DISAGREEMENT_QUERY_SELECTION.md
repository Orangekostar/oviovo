# Disagreement query selection

固定筛选完成：10 条方法路径、4 个已曝光场景、40 条完整输出结果、20 个有序双场景池。三个预定扩展对比均失败，按协议保留 DQ01_G1，部署 N0_UNCHANGED。完整 8/18 场景回归 NOT_RUN_NO_SCREEN_SIGNAL；严格与材料目标未在完整回归中验证。

QS=COVERAGE；US=MEAN。选择使用未舍入 fraction 与固定字典序；未按数据集切换。

扩展条件同时要求真实输出变化、两个池各自 APall 损失不超过 0.10 pp、平均 APall 增益至少 0.05 pp、相对 G1 的 APall/AP50/mIoU 保护，以及净固定参考纠错或净唯一 GT50 至少 +1。三个对比均未通过。

| candidate | reference | replica_probe2_apall_delta_pp | cf_probe2_apall_delta_pp | second_view_divergence | selected_owner_class_differences | wrong_to_right | right_to_wrong | net_unique_GT50 | net_unique_GT75 | extension_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DISAGREEMENT_MEAN | COVERAGE_MEAN | -2.1233 | 0.0000 | 54 | 15 | 3 | 6 | -3 | -1 | False |
| QS_SUPPORT | COVERAGE_MEAN | -1.7642 | 0.0000 | 0 | 6 | 0 | 2 | -2 | 0 | False |
| QD_SUPPORT | DISAGREEMENT_MEAN | 0.0000 | 0.0000 | 0 | 5 | 1 | 1 | 0 | 0 | False |
| QD_SUPPORT | QS_SUPPORT | -0.3591 | 0.0000 | 54 | 9 | 2 | 3 | -1 | -1 | — |

QS/US 的完整排序元组与门槛证据见 [selection.json](../../../artifacts/static_ovmap/disagreement_query_v1/selection.json)。
严格目标与材料目标：NOT_EVALUATED_FULL_REGRESSION_NOT_TRIGGERED。完整目标未验证，不将筛选数值当作完整 8/18 场景收益。
