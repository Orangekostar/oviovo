# 180场景完整分类分数实测

2026-10-25：前180个ScanNet场景已完成1440份方法/种子分类矩阵，共111968个owner行，其中103012行有定义、8956行未定义。数量跨方法和种子累计，不表示111968个独立实例。当前汇总身份为`6b94f6ae10f774342524c2c36bfc89cb1d02c6df989c2b4915e0d5ece7afe5d6`。

| 范围 | 矩阵 | owner行 | 有定义 | 未定义 |
|---|---:|---:|---:|---:|
| 原170场景 | 1360 | 102848 | 94628 | 8220 |
| 新增171–180项 | 80 | 9120 | 8384 | 736 |
| 当前180场景 | 1440 | 111968 | 103012 | 8956 |

新增场景为`scene0246_00`、`scene0353_01`、`scene0353_02`、`scene0559_00`、`scene0559_01`、`scene0559_02`、`scene0593_00`、`scene0593_01`、`scene0653_00`、`scene0653_01`。各场景四项科学作业、三项有序元数据核验及合并主审均实际退出0，准备阶段随后固定前180份预测及评分收据。后台第181项及以后的新收据不进入本次汇总。

| 单次执行步骤 | 实际退出 | 进程墙钟（秒） | 实际退出收据身份 |
|---|---:|---:|---|
| 准备及归档 | 0 | 41.9768 | `cff0069b8dcf54bb16a674707c267ddd9946ed1603526cb2e3e13507ea983bb2` |
| 分类数值校验 | 0 | 91.8182 | `df987599251d3f120d822d1ce04954baad46b1a467a5835773a5a7d50a47ed84` |
| 主审及新基线 | 0 | 20.2774 | `ea89ef21570702f33c70f80de7c2a2f1cc08b0f592c8ea6d77e40328d8462a1f` |

新增9120行的1824000个类别位置经过独立数值校验，绝对和相对容差均为`1e-12`；既有标签、置信度和导出标志保持一致，原owner及未定义行保留。数值循环、稳定softmax和访问限制共三个AST单元与既有入口一致。数值进程未访问GT、几何或模型worker，未导入torch/tensorflow，新增模型推理、训练更新和AP重算均为0。主审在数值进程之外盲哈希50份历史GT元数据收据，不解码GT标签。

旧1360条矩阵描述及2720份矩阵文件保持原SHA/修改时间，2881份只读复用输出通过校验。旧170汇总复制到`review/history/class_scores_170scenes_576608f8c1b1a133115de4363d39923454530f3a3ccbfab9afe13d3fdd3b7518.json`，SHA为`97430b071b143a7111b88da9831d4962efe707588e434bae31e7b08e0c61fcf6`，修改时间为`1792935714718779040`纳秒，与原件相同且inode独立。原3080份基线中仅分类汇总路径授权替换，其余3079份旧路径保持SHA/修改时间；新增160份矩阵文件后，新稳定基线共3240份。93份原生产/测试/元数据冻结文件保持不变。

可核验记录位于本轮输出根目录：

- `review/G4_180scene_class_score_preparation_primary_review.json`：`eddbbec8172854d96c56c5e90808de01b706c830fbb1b5c0cf994d33f7bf336e`。
- `review/G4_full_scores_scenes171_to180_actual.json`：`c941b81eb8476977e550de946841aecf1452f0e69a71e8dff8ee1b3e571096cb`。
- `review/G4_full_scores_scenes171_to180_primary_review.json`：`2c1ed9daca855eb5f7692e1bb4db49a29bb14d18baad969b5812b1e7c01be41b`。
- `bootstrap/before180_report_primary_review.json`：`aa8cf0c4ca14aced92486beef0b5d9cd85b9491640d3ade2195a5698fcea16a7`，作为后续逐场景主审基线。
- 三步实际退出收据及完整日志分别保存在`bootstrap/prepare_class_scores180_actual*`、`bootstrap/full_scores_scenes171_to180_actual*`和`bootstrap/full_scores_scenes171_to180_primary_actual*`。

[准备入口](CLASSIFICATION_SCORES180_ENTRY.md)和交接身份`4cda20ddac53918c0c8918924596cdb25eb93ef37fea4fa10429f76fad1d4174`保留创建时的待执行状态；其17份支持文件均保持原SHA/修改时间，不改写历史。本页记录随后实际完成的执行。最终有限来源清单仍需纳入本批脚本、差异、入口、准备/数值/主审记录、实际日志与退出收据、新基线、旧汇总归档和本文。

本检查点不扩大已有配对池82场景、条件诊断84场景或公开47场景范围，Replica仍为0/8。双卡继续后续科学队列；完整任务、正式统计、最终来源清单/17章审核和C1/C2尚未完成。
