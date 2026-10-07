# 固定证据探索实验结果

固定九方法 × 26 场景已完成，覆盖 234/234 场景行及 18/18 官方完整池化结果。目标门槛：未达到；统一研究推荐 `EV01_G1_V2`，部署 `N0_UNCHANGED`。

Replica-8 与 ScanNet-CF18 均为已曝光队列；CF18 包含 18 次采集、7 个物理场景家族。结果属于回顾性研究选择，不是独立验证。

指标单位为百分数；差值为百分点，参考 B1。AP 使用全队列官方池化。显示保留两位小数，选模使用未舍入数值。

| Cohort | Method | apall | ap50 | miou | Δapall_pp | Δap50_pp | Δmiou_pp | TARGET_MET |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replica8 | EV00_D2 | 11.74 | 24.50 | 29.69 | -0.65 | -1.56 | -0.58 | False |
| replica8 | EV01_G1_V2 | 12.39 | 26.06 | 30.27 | 0.00 | 0.00 | 0.00 | False |
| replica8 | EV02_MARGIN | 12.39 | 26.06 | 30.21 | 0.00 | 0.00 | -0.06 | False |
| replica8 | EV03_CONTRAST | 12.39 | 26.06 | 30.21 | 0.00 | 0.00 | -0.06 | False |
| replica8 | EV04_BILINEAR | 12.10 | 25.54 | 30.13 | -0.29 | -0.52 | -0.14 | False |
| replica8 | EV05_ANYUP | 12.39 | 26.06 | 30.34 | 0.00 | 0.00 | 0.07 | False |
| replica8 | EV06_OWNER_ANYUP | 12.10 | 25.54 | 30.15 | -0.29 | -0.52 | -0.12 | False |
| replica8 | EV07_COMBINATION | 12.39 | 26.06 | 30.26 | 0.00 | 0.00 | -0.01 | False |
| replica8 | EV08_NO_DEPTH | 12.10 | 25.54 | 30.14 | -0.29 | -0.52 | -0.13 | False |
| scannet_cf18 | EV00_D2 | 8.31 | 17.96 | 19.00 | 0.08 | 0.13 | -0.02 | False |
| scannet_cf18 | EV01_G1_V2 | 8.23 | 17.83 | 19.02 | 0.00 | 0.00 | 0.00 | False |
| scannet_cf18 | EV02_MARGIN | 8.29 | 17.91 | 19.05 | 0.06 | 0.08 | 0.03 | False |
| scannet_cf18 | EV03_CONTRAST | 8.26 | 17.88 | 19.04 | 0.03 | 0.05 | 0.02 | False |
| scannet_cf18 | EV04_BILINEAR | 8.26 | 17.87 | 19.05 | 0.03 | 0.03 | 0.03 | False |
| scannet_cf18 | EV05_ANYUP | 8.29 | 17.91 | 19.06 | 0.06 | 0.08 | 0.04 | False |
| scannet_cf18 | EV06_OWNER_ANYUP | 8.26 | 17.87 | 19.06 | 0.03 | 0.04 | 0.04 | False |
| scannet_cf18 | EV07_COMBINATION | 8.29 | 17.91 | 19.07 | 0.06 | 0.08 | 0.04 | False |
| scannet_cf18 | EV08_NO_DEPTH | 8.26 | 17.87 | 19.06 | 0.03 | 0.04 | 0.04 | False |

方法机制的配对比较如下。TP/FP 列是 IoU=.5 的确定性评分条目变化；唯一 GT 匹配变化、歧义条目和全部类级差值见 companion JSON。

| Cohort | Comparison | ΔAPall_pp | ΔAP50_pp | ΔmIoU_pp | Accepted | Deferred | Relabelled | Lost_reference_TP50 | Removed_reference_FP50 | Lost_B1_TP50 | Removed_B1_FP50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replica8 | EV03_CONTRAST / EV02_MARGIN | 0.00 | 0.00 | 0.00 | 22 | 20 | 0 | 0 | 0 | 0 | 1 |
| replica8 | EV05_ANYUP / EV04_BILINEAR | 0.29 | 0.52 | 0.21 | 42 | 0 | 15 | 0 | 2 | 0 | 1 |
| replica8 | EV06_OWNER_ANYUP / EV05_ANYUP | -0.29 | -0.52 | -0.19 | 42 | 0 | 12 | 0 | 1 | 0 | 1 |
| replica8 | EV06_OWNER_ANYUP / EV08_NO_DEPTH | 0.00 | 0.00 | 0.01 | 42 | 0 | 12 | 0 | 1 | 0 | 1 |
| replica8 | EV07_COMBINATION / EV03_CONTRAST | 0.00 | 0.00 | 0.05 | 20 | 22 | 1 | 0 | 1 | 0 | 2 |
| replica8 | EV07_COMBINATION / EV06_OWNER_ANYUP | 0.29 | 0.52 | 0.11 | 20 | 22 | 1 | 0 | 4 | 0 | 2 |
| scannet_cf18 | EV03_CONTRAST / EV02_MARGIN | -0.03 | -0.03 | -0.01 | 77 | 55 | 0 | 0 | 3 | 0 | 17 |
| scannet_cf18 | EV05_ANYUP / EV04_BILINEAR | 0.03 | 0.04 | 0.01 | 132 | 0 | 50 | 0 | 8 | 0 | 17 |
| scannet_cf18 | EV06_OWNER_ANYUP / EV05_ANYUP | -0.03 | -0.04 | -0.00 | 132 | 0 | 44 | 0 | 2 | 0 | 15 |
| scannet_cf18 | EV06_OWNER_ANYUP / EV08_NO_DEPTH | 0.00 | 0.00 | 0.00 | 132 | 0 | 44 | 0 | 0 | 0 | 15 |
| scannet_cf18 | EV07_COMBINATION / EV03_CONTRAST | 0.03 | 0.03 | 0.02 | 81 | 51 | 15 | 0 | 7 | 0 | 23 |
| scannet_cf18 | EV07_COMBINATION / EV06_OWNER_ANYUP | 0.03 | 0.04 | 0.00 | 81 | 51 | 15 | 0 | 18 | 0 | 23 |

新配对冷计时：模型与公共几何/NQF 驻留，G1 搜索、额外几何、FC、AnyUp 与输出均在边界内；8 场景各测两轮，第二轮反序。系统页缓存及共享主机后台 CPU 负载未控制，初期另有只读配对诊断进程；未验证在线 30 FPS。

| Method | Calls | Seconds_scene | Allocated_GiB | Reserved_GiB | FC_inputs_total | AnyUp_QK_total | Added_projection_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EV01_G1_V2 | 16 | 22.71 | 1.90 | 2.21 | 72 | 0 | 0 |
| EV07_COMBINATION | 16 | 24.20 | 4.35 | 6.65 | 72 | 70 | 72 |
| EV03_CONTRAST | 16 | 22.69 | 1.90 | 2.21 | 72 | 0 | 72 |
| EV06_OWNER_ANYUP | 16 | 25.49 | 4.33 | 6.65 | 72 | 70 | 72 |

科学获取新增 FC 编码 0 次、AnyUp Q/K 123 次。官方完整帧一致性验证额外 1 次 AnyUp Q/K，计时阶段的新推理另见 table3 与 timing_records.json。基线固定旧 rank 的两次诊断池化不属于官方性能。

显存为 PyTorch allocator allocated/reserved 峰值，包含驻留模型；完整建图系统显存未测。模型加载时间、五阶段耗时、非可加 CUDA events、失败成本和全部重复保留在机器数据中。

成功的共享几何准备完成 160 次完整选中帧投射；首次字段修复前另有 1 次 CPU 投射，隔离保留。共享盘 ENOSPC 中断的预测/评分叶没有新增 GPU 推理，但失败墙钟未在当时记录，记为 null，不补零。新冷计时的输出边界止于完整预测 payload 构造；预测哈希、一致性检查及收据写入在计时外，不能与旧版包含文件导出的计时直接作加速比。
