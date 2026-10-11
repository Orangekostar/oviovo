# Frozen FC routing results

完成状态：**COMPLETE_NO_TARGET_GAIN**。科学优化 16,000 步；选定小型头 8 个；新整图评分 286 行 / 22 个池。

已记录的重复优化步数：3，与科学更新分开统计。

DEV 固定提名：`FG2_SURFACE`。通过 DEV 门槛：FG1_VIEW (seed17, step500), FG2_SURFACE (seed17, step1000), FG3_AUX (seed17, step1000), FG4_ROUTE (seed17, step1500), FG1_VIEW (seed29, step250), FG2_SURFACE (seed29, step2000), FG3_AUX (seed29, step1000), FG4_ROUTE (seed29, step2000)。部署保持 `N0_UNCHANGED`。

DEV 数字以百分比显示，CE/NLL 保留原单位；69 个原始对象的四种条件相关，不能视作276个独立样本。

| method | seed | step | status | A_pct | M_pct | C_pct | CE_base160 | NLL_full200 | corrections | harm | net_corrections | original_objects | variant_records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FC8 | NA | NA | FROZEN_REFERENCE | 37.288510 | 50.612374 | 42.058081 | 4.066619 | 4.290697 | 0 | 0 | 0 | 69 | 276 |
| FG1_VIEW | 17 | 500 | QUALIFIED | 42.535774 | 58.429834 | 46.363636 | 3.911982 | 4.139906 | 22 | 15 | 7 | 69 | 276 |
| FG2_SURFACE | 17 | 1000 | QUALIFIED | 46.071128 | 58.866342 | 48.011364 | 3.949017 | 4.176042 | 24 | 9 | 15 | 69 | 276 |
| FG3_AUX | 17 | 1000 | QUALIFIED | 45.315657 | 58.072691 | 45.233586 | 3.951118 | 4.178064 | 23 | 9 | 14 | 69 | 276 |
| FG4_ROUTE | 17 | 1500 | QUALIFIED | 44.476010 | 57.318723 | 42.455808 | 3.936418 | 4.163035 | 23 | 10 | 13 | 69 | 276 |
| FG1_VIEW | 29 | 250 | QUALIFIED | 40.435606 | 56.098485 | 42.960859 | 3.945075 | 4.172773 | 16 | 13 | 3 | 69 | 276 |
| FG2_SURFACE | 29 | 2000 | QUALIFIED | 46.890783 | 59.322691 | 48.516414 | 3.945984 | 4.172726 | 26 | 9 | 17 | 69 | 276 |
| FG3_AUX | 29 | 1000 | QUALIFIED | 46.828704 | 59.580628 | 48.011364 | 3.942229 | 4.169163 | 26 | 9 | 17 | 69 | 276 |
| FG4_ROUTE | 29 | 2000 | QUALIFIED | 46.724537 | 58.519120 | 48.636364 | 3.934708 | 4.161238 | 21 | 7 | 14 | 69 | 276 |


官方整图有序池化结果（%，不平均场景 AP）：

| cohort | method | seed | status | APall_pct | AP50_pct | AP25_pct | mIoU_pct | mAcc_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replica8 | REF_D2 | NA | MEASURED_OFFICIAL_ORDERED_POOL | 11.741338 | 24.495210 | 37.974586 | 29.691085 | 37.427869 |
| replica8 | REF_G1 | NA | MEASURED_OFFICIAL_ORDERED_POOL | 12.386369 | 26.057090 | 39.752550 | 30.272532 | 38.403595 |
| replica8 | FG0_FC8 | NA | MEASURED_OFFICIAL_ORDERED_POOL | 11.729921 | 26.570994 | 41.470906 | 31.771445 | 39.298482 |
| replica8 | FG1_VIEW | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 11.112718 | 24.321497 | 39.830103 | 30.125539 | 37.258976 |
| replica8 | FG2_SURFACE | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 11.769134 | 26.155712 | 42.648419 | 32.165940 | 39.967293 |
| replica8 | FG3_AUX | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 11.898163 | 26.828868 | 43.324317 | 32.483479 | 40.317523 |
| replica8 | FG4_ROUTE | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 10.872204 | 24.467550 | 40.549073 | 31.508603 | 38.581687 |
| scannet_cf18 | REF_D2 | NA | MEASURED_OFFICIAL_ORDERED_POOL | 8.309185 | 17.961360 | 24.982824 | 18.997180 | 28.377972 |
| scannet_cf18 | REF_G1 | NA | MEASURED_OFFICIAL_ORDERED_POOL | 8.232195 | 17.833990 | 24.755755 | 19.021902 | 28.492872 |
| scannet_cf18 | FG0_FC8 | NA | MEASURED_OFFICIAL_ORDERED_POOL | 7.994702 | 17.577712 | 26.022644 | 20.096838 | 29.250947 |
| scannet_cf18 | FG1_VIEW | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 8.158292 | 18.093779 | 25.998911 | 20.510896 | 29.707394 |
| scannet_cf18 | FG2_SURFACE | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 8.187554 | 17.924653 | 26.262072 | 20.488766 | 29.555397 |
| scannet_cf18 | FG3_AUX | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 8.220479 | 17.795085 | 26.158863 | 20.372377 | 29.530084 |
| scannet_cf18 | FG4_ROUTE | 17 | MEASURED_OFFICIAL_ORDERED_POOL | 8.208977 | 17.941555 | 26.435670 | 20.586187 | 29.588805 |

整图研究候选：`NONE`。固定 DEV 提名与曝光整图候选分开记录；第二种子的完整结果见 `table1_whole_map_repeat`。

H2：`COMPLETE`；原始对象 115，novel 对象 20，novel 家族 4。这是提案识别诊断，不是独立整图 AP。

真实提案：`COMPLETE`；生成 72、保留 68、多视图支持 67、有效 GT 匹配 7。这是提案识别诊断，不是独立整图 AP。

主代理核对：固定提名 FG2 在主轮 Replica 的 mIoU 为 32.166%，高于 G1 的 30.273%，但 APall 为 11.769%，低于 G1 的 12.386%；CF18 的 APall 为 8.188%，低于 D2 的 8.309%。普通 FC8 已有部分语义收益，FG2 相对 FC8 的 mIoU 增量在 Replica/CF18 仅为 0.394/0.392 个百分点。两种子所有训练方法均未达到整图升级门槛，研究基线保留 REF_G1。

H2 普通 ALL Top-1：FC8 为 26.087%，固定 FG2 两种子为 31.304% / 30.652%；20 个 heldout 对象上的四条件普通 Top-1 为 31.25% → 40.00% / 41.25%。这是有限对象集上的识别改善。真实提案七个 GT 匹配对象中，九种方法均答对五个（71.429%）；该小样本未区分方法优劣。

FG4 的两种子 DEV A 均低于 FG2；现有结果不足以支持“复杂路由稳定优于简单表面分组”的结论。完整验收见 [FINAL_REVIEW](FROZEN_FC_ROUTING_FINAL_REVIEW.md)，三类表格见 [artifacts](../../../artifacts/static_ovmap/frozen_fc_routing_v1/tables)。
