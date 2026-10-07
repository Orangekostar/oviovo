# 证据与结论边界

固定九方法 × 26 场景已完成，覆盖 234/234 场景行及 18/18 官方完整池化结果。目标门槛：未达到；统一研究推荐 `EV01_G1_V2`，部署 `N0_UNCHANGED`。

Replica-8 与 ScanNet-CF18 均为已曝光队列；CF18 包含 18 次采集、7 个物理场景家族。结果属于回顾性研究选择，不是独立验证。

- replica8，`EV03_CONTRAST` 对 `EV02_MARGIN`：APall +0.0000 pp、AP50 +0.0000 pp、mIoU +0.0000 pp。
- replica8，`EV05_ANYUP` 对 `EV04_BILINEAR`：APall +0.2894 pp、AP50 +0.5208 pp、mIoU +0.2087 pp。
- replica8，`EV06_OWNER_ANYUP` 对 `EV05_ANYUP`：APall -0.2894 pp、AP50 -0.5208 pp、mIoU -0.1898 pp。
- replica8，`EV06_OWNER_ANYUP` 对 `EV08_NO_DEPTH`：APall +0.0000 pp、AP50 +0.0000 pp、mIoU +0.0136 pp。
- replica8，`EV07_COMBINATION` 对 `EV03_CONTRAST`：APall +0.0000 pp、AP50 +0.0000 pp、mIoU +0.0458 pp。
- replica8，`EV07_COMBINATION` 对 `EV06_OWNER_ANYUP`：APall +0.2894 pp、AP50 +0.5208 pp、mIoU +0.1082 pp。
- scannet_cf18，`EV03_CONTRAST` 对 `EV02_MARGIN`：APall -0.0258 pp、AP50 -0.0291 pp、mIoU -0.0120 pp。
- scannet_cf18，`EV05_ANYUP` 对 `EV04_BILINEAR`：APall +0.0287 pp、AP50 +0.0444 pp、mIoU +0.0148 pp。
- scannet_cf18，`EV06_OWNER_ANYUP` 对 `EV05_ANYUP`：APall -0.0277 pp、AP50 -0.0415 pp、mIoU -0.0006 pp。
- scannet_cf18，`EV06_OWNER_ANYUP` 对 `EV08_NO_DEPTH`：APall +0.0000 pp、AP50 +0.0000 pp、mIoU +0.0000 pp。
- scannet_cf18，`EV07_COMBINATION` 对 `EV03_CONTRAST`：APall +0.0258 pp、AP50 +0.0291 pp、mIoU +0.0248 pp。
- scannet_cf18，`EV07_COMBINATION` 对 `EV06_OWNER_ANYUP`：APall +0.0282 pp、AP50 +0.0426 pp、mIoU +0.0044 pp。

只有完整指标和匹配诊断支持的方法差异可作为研究结论。接受数量不是真实召回；owner/depth 约束不能保证恢复粗特征中已丢失的信息，竞争区域可来自同类邻居。几何不足与类别错误为非互斥诊断。

replica8：42/53 个候选有 B1 表征，其中 37 个在 IoU=.5 下几何不足、31 个目标投影少于 100 点；released scorer 实际忽略 2 个。三项非互斥，不相加。

scannet_cf18：132/175 个候选有 B1 表征，其中 131 个在 IoU=.5 下几何不足、69 个目标投影少于 100 点；released scorer 实际忽略 16 个。三项非互斥，不相加。

粗对比相对单纯 margin 未增加确定 TP50，CF18 APall 还下降约 0.026 pp，不能支持其额外有效性。普通 AnyUp 相对 bilinear 改善两队列的五项指标，但 Replica AP25 低于 B1，未达到固定目标。owner/depth 约束相对普通 AnyUp 使两队列 APall/AP50 下降；CF18 去深度后的五项池化指标完全相同，未显示深度的增益。组合在 CF18 相对其两个组件有所改善，但仍低于 D2 的 APall，且 Replica mIoU/mAcc 低于 B1。

所有直接机制比较的唯一 GT50 匹配增减均为 0；AP 改变同时涉及类别、FP 与官方当前类别面积 rank，不能将其描述成新增真实召回，也不能仅用 TP50 解释 APall 的九阈值池化差异。

FC 与 AnyUp 均为冻结预训练模型；继承 Q/温度包含既有拟合，因此不宣称全系统无训练。没有新建图、前端、Native N/Q 推理、训练或拟合，也不宣称泛化、显著性、在线实时性能或 OVI-MAP 全方法复现。
