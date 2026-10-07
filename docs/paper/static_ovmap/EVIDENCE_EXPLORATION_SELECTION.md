# 研究选模

固定九方法 × 26 场景已完成，覆盖 234/234 场景行及 18/18 官方完整池化结果。目标门槛：未达到；统一研究推荐 `EV01_G1_V2`，部署 `N0_UNCHANGED`。

Replica-8 与 ScanNet-CF18 均为已曝光队列；CF18 包含 18 次采集、7 个物理场景家族。结果属于回顾性研究选择，不是独立验证。

规则使用未舍入 fraction 与 1e-10 数值容差：两队列各五项指标均不低于 B1，CF18 APall 严格高于 D2，且 AP50 不低于 D2；0.10 pp 为另列的工程幅度标记。

- `EV00_D2`：Replica 五项 False；CF18 五项 False；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV01_G1_V2`：Replica 五项 True；CF18 五项 True；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV02_MARGIN`：Replica 五项 False；CF18 五项 False；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV03_CONTRAST`：Replica 五项 False；CF18 五项 False；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV04_BILINEAR`：Replica 五项 False；CF18 五项 True；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV05_ANYUP`：Replica 五项 False；CF18 五项 True；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV06_OWNER_ANYUP`：Replica 五项 False；CF18 五项 True；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV07_COMBINATION`：Replica 五项 False；CF18 五项 True；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。
- `EV08_NO_DEPTH`：Replica 五项 False；CF18 五项 True；CF AP 条件 False；TARGET_MET False；MATERIAL_TARGET_MET False。

通过者按 CF APall、CF AP50、Replica APall、CF mIoU、固定方法索引决胜；无通过者保留 B1。没有逐场景或逐队列切换，也未追加阈值搜索。

冷计时探索对象 `EV07_COMBINATION`，固定比较集：EV01_G1_V2, EV07_COMBINATION, EV03_CONTRAST, EV06_OWNER_ANYUP。未新计时的精度对比不作速度结论。
