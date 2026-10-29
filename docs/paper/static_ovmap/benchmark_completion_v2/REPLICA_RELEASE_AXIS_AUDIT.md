# Replica 发布包类别轴核验

2026-10-26。原生产协议、GT、模型与活动队列保持不变。这里只核对作者源码定义，没有打开 GT 文本数值。

OpenYOLO3D（commit `24f344d001d28d1cdd4087d3769c0d53c34d64cd`）与 OpenIns3D（commit `56ccc220dc003e38c644116d7d2078057b6136ad`）的 Replica 名称和原始 ID 两轴逐项相同。名称又与本任务冻结的 48 类逐项相同；作者原始 ID 和任务的连续 1–48 ID 不同。以下对应关系来自源码中的完整名单，不依赖任何测试标签分布。

[OpenYOLO3D 配套评测器](https://github.com/aminebdj/OpenYOLO3D/blob/24f344d001d28d1cdd4087d3769c0d53c34d64cd/evaluate/replica/eval_semantic_instance.py#L63)；[OpenIns3D 类别定义](https://github.com/Pointcept/OpenIns3D/blob/56ccc220dc003e38c644116d7d2078057b6136ad/openins3d/utils.py#L34)。

|名称|发布包配套评测器 ID|当前冻结 ID|
|---|---:|---:|
|basket|3|1|
|bed|7|2|
|bench|8|3|
|bin|10|4|
|blanket|11|5|
|blinds|12|6|
|book|13|7|
|bottle|14|8|
|box|15|9|
|bowl|16|10|
|camera|17|11|
|cabinet|18|12|
|candle|19|13|
|chair|20|14|
|clock|22|15|
|cloth|23|16|
|comforter|26|17|
|cushion|29|18|
|desk|34|19|
|desk-organizer|35|20|
|door|37|21|
|indoor-plant|44|22|
|lamp|47|23|
|monitor|52|24|
|nightstand|54|25|
|panel|56|26|
|picture|59|27|
|pillar|60|28|
|pillow|61|29|
|pipe|62|30|
|plant-stand|63|31|
|plate|64|32|
|pot|65|33|
|sculpture|70|34|
|shelf|71|35|
|sofa|76|36|
|stool|78|37|
|switch|79|38|
|table|80|39|
|tablet|82|40|
|tissue-paper|83|41|
|tv-screen|87|42|
|tv-stand|88|43|
|vase|91|44|
|vent|92|45|
|wall-plug|95|46|
|window|97|47|
|rug|98|48|

作者配套评测器将 GT 解释为 `semantic_id*1000+instance_component`。因此公开 GT 文件不能直接作为当前连续轴的 GT；若未来获准接入同一点域，必须以这份完整双射显式改写语义编码，同时保留全部点和原实例 mask。无效原类别必须继续作为 void，不能因 ID 恰好落在 1–48 内而变成有效类别。实际 GT 编码仍须在真实预测封存后核验。

Replica 原始 ID、当前 Replica 连续 ID 和 ScanNet 训练语义 ID 是不同体系。不能用它们的整数相等来推导训练 base 角色。14 个未解析训练角色仍未解决；本次没有把未匹配类别判为 novel。

已有两份公开包的八个 PLY/GT 文件字节相同；八个 PLY 与本机 OpenScene 三字段 tuple 的 FP32 坐标逐点原序相同。该证据仍未证明与冻结 Open3DIS 四字段 tuple 的 half-point 域一致，不能把本次类别轴核验当作点序证明或正式 G3。

接入前仍需：冻结原目标点域的可验证连接、GT 与当前协议的显式重编码、先封存实际完整预测、再完成真实首场作者/共同评测器一致性。此次未改写 GT，未生成标准 Replica AP，未运行模型。

源码均按固定 commit 下载，并以 Git tree 的 blob SHA1 和本地 SHA256 双重核对。机器记录：`assets/acquisition_20261026/Replica_source_audit_20261026/class_axis_audit.json`，身份 `bce3d52e89480b4a2adeeba987ca12288016b82a4873c48191dd63afa78bcb46`。核验程序实际退出 0，日志 `recovery/takeover_20261026/replica_release_axis.log`。原 93 份冻结文件 SHA/mtime 核验通过。
