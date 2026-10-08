# SAM-V 固定局部探针结果

固定完整场景 office1、room0、scene0011_00、scene0050_00；前两者为 Replica-probe2，后两者为 CF-probe2。每组由两个场景按固定顺序调用 released evaluator 池化，非逐场景 AP 平均。APall 使用 .50:.05:.90，最小实例100点。全部场景此前已曝光。

| Method | Replica APall | Replica AP50 | Replica mIoU | CF APall | CF AP50 | CF mIoU |
| --- | --- | --- | --- | --- | --- | --- |
| D2 reference | 14.630 | 32.766 | 26.405 | 13.107 | 30.915 | 26.226 |
| G1 | 14.630 | 32.766 | 26.395 | 13.107 | 30.915 | 26.262 |
| SAM2 geometry | 14.630 | 32.766 | 26.396 | 13.107 | 30.915 | 26.262 |
| SAM-V geometry | 14.630 | 32.766 | 26.398 | 13.107 | 30.915 | 26.262 |
| OLD-mask FC | 13.788 | 31.251 | 25.438 | 13.107 | 30.915 | 27.584 |
| SAM-V-mask FC | 13.788 | 31.251 | 25.437 | 13.462 | 31.489 | 27.949 |
| SAM-V + fixed FC | 13.788 | 31.251 | 25.444 | 13.462 | 31.489 | 27.949 |

研究选择：SV00_G1；状态 COMPLETE_NO_PILOT_GAIN；MATERIAL=False。部署 N0_UNCHANGED。

固定词典序下的最佳简单对照：SV01_SAM2_GEOM。该对照单独报告，不重命名为候选新方法。

原始、允许修改与最终独占支持分别保留；class-agnostic GT50/75 不等于官方 AP，也不以查询数作为 precision 分母。

| Cohort | Model | Targets | Anchor OK | Raw rows | Admitted rows | Blocked rows | GT50 | GT75 | Best IoU | Fixed-GT IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cf_probe2 | SAM2 | 16 | 15 | 8838 | 1538 | 7300 | 30 | 11 | 0.172 | 0.183 |
| cf_probe2 | SAMV | 16 | 15 | 3432 | 2714 | 718 | 30 | 11 | 0.172 | 0.183 |
| replica_probe2 | SAM2 | 16 | 13 | 82269 | 45606 | 36663 | 45 | 21 | 0.219 | 0.319 |
| replica_probe2 | SAMV | 16 | 10 | 61378 | 30701 | 30677 | 45 | 21 | 0.216 | 0.315 |

| Cohort | Joint eligible | Wrong to right | Right to wrong | No fixed GT | APall delta (pp) | AP50 delta (pp) | AP25 delta (pp) | mIoU delta (pp) | mAcc delta (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cf_probe2 | 8 | 1 | 0 | 1 | 0.355 | 0.574 | 0.605 | 0.365 | 0.029 |
| replica_probe2 | 10 | 0 | 0 | 5 | 0.000 | 0.000 | 0.000 | -0.001 | 0.000 |

窗口计时：相同预锁定 JPEG/提示、BF16、模型驻留、无特征/结果缓存；读图到原尺寸二值 mask，加载/lifting/FC/评分/保存在计时外。第二轮反序；OS 页缓存未控制。每个输出与科学结果逐像素一致。

| Scene | Owner | Model | Calls | Frames | Dtype | Mean (s) | Median (s) | Allocated (GiB) | Reserved (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| office1 | 8 | SAM2 | 2 | 6 | BF16 | 0.796 | 0.796 | 1.822 | 2.102 |
| scene0011_00 | 14 | SAM2 | 2 | 6 | BF16 | 0.721 | 0.721 | 1.822 | 2.102 |
| office1 | 8 | SAMV | 2 | 6 | BF16 | 3.407 | 3.407 | 12.575 | 13.498 |
| scene0011_00 | 14 | SAMV | 2 | 6 | BF16 | 3.220 | 3.220 | 12.575 | 13.498 |

逻辑记录28、两场景池14；新场景 scorer 调用 18。独立的工程检查、失败、模型加载和缓存成本见机器数据。

在上述两个相同六帧窗口中，SAM-V耗时分别为SAM2的4.28倍和4.47倍，峰值allocated显存为6.90倍。设备NVIDIA A40，作者各自要求的torch/CUDA环境不同；每窗每模型仅两次测量，保留全部原始值，不作显著性结论。此组数值只比较分割窗口，不能与G1全场景19.89秒或OVIMap在线模块时间直接求速度比。

SAM-V-mask FC相对G1在CF-probe2提高APall约0.355个百分点、mIoU约1.687个百分点，但在Replica-probe2降低APall约0.842个百分点、mIoU约0.957个百分点。结构分割未增加相对SAM2的全地图GT50/75，目标best-IoU均值略低；语义收益主要来自CF的一次wrong→right，未形成两队列一致改善。

[完整五项指标](../../../artifacts/static_ovmap/samv_local_probe_v1/tables/table1_all_five_metrics.md) · [阶段成本](../../../artifacts/static_ovmap/samv_local_probe_v1/tables/table3_actual_stage_costs.md) · [真实 mask 全目标展示](../../../artifacts/static_ovmap/samv_local_probe_v1/visuals) · [规范结果库](../../../artifacts/static_ovmap/samv_local_probe_v1/result_store.json)

完整地图独立延迟为 NOT_MEASURED_THIS_PROBE；不声称在线30 FPS。未自动扩大到 Replica8/CF18。
