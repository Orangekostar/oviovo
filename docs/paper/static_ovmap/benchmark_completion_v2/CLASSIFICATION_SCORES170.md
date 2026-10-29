# 170场景完整分类分数实测

2026-10-25：前170个ScanNet场景已完成1360份方法/种子分类矩阵，共102848个owner行，其中94628行有定义、8220行未定义。数量跨方法和种子累计，不表示102848个独立实例。当前汇总身份为`576608f8c1b1a133115de4363d39923454530f3a3ccbfab9afe13d3fdd3b7518`。

| 范围 | 矩阵 | owner行 | 有定义 | 未定义 |
|---|---:|---:|---:|---:|
| 原160场景 | 1280 | 96040 | 88296 | 7744 |
| 新增161–170项 | 80 | 6808 | 6332 | 476 |
| 当前170场景 | 1360 | 102848 | 94628 | 8220 |

新增场景为`scene0081_00`、`scene0081_01`、`scene0081_02`、`scene0353_00`、`scene0535_00`、`scene0645_00`、`scene0645_01`、`scene0645_02`、`scene0647_00`、`scene0647_01`。各场景四项科学作业、三项有序元数据核验及合并主审均实际退出0，准备阶段随后固定前170份预测及评分收据。后台第171项及以后的新收据不进入本次汇总。

| 单次执行步骤 | 实际退出 | 进程墙钟（秒） | 实际退出收据身份 |
|---|---:|---:|---|
| 准备及归档 | 0 | 34.2238 | `3037bb14e0ad3ca1696bbef5aa71adb9fafe4ae0ef99dadb30d96d29f3f0f483` |
| 分类数值校验 | 0 | 115.3837 | `afb0ae12c9804044dcdc51cfe73ca6dad29f82b8830043349913f0998cf40ddd` |
| 主审及新基线 | 0 | 23.8843 | `b6cccfdbc1ddccabc1446cd292470d62da90e5ebd3ff85605c09fdd4d15bfb02` |

新增6808行的1361600个类别位置经过独立数值校验，绝对和相对容差均为`1e-12`；既有标签、置信度和导出标志保持一致，原owner及未定义行保留。数值循环、稳定softmax和访问限制共三个AST单元与既有入口一致。数值进程未访问GT、几何或模型worker，未导入torch/tensorflow，新增模型推理、训练更新和AP重算均为0。主审在数值进程之外盲哈希50份历史GT元数据收据，不解码GT标签。

旧1280条矩阵描述及2560份矩阵文件保持原SHA/修改时间，2721份只读复用输出通过校验。旧160汇总复制到`review/history/class_scores_160scenes_679c87e019c5147783371558249657b9510be6e5d74c88c03bb1ad77196ad053.json`，SHA为`e4b06b3aeec56d07e20c01fb6dea10de38611833eacc266ae0cfd168bded86bf`，修改时间为`1792921129299169492`纳秒，与原件相同且inode独立。原2920份基线中仅分类汇总路径授权替换，其余2919份旧路径保持SHA/修改时间；新增160份矩阵文件后，新稳定基线共3080份。93份原生产/测试/元数据冻结文件保持不变。

可核验记录位于本轮输出根目录：

- `review/G4_170scene_class_score_preparation_primary_review.json`：`19c9610db1e08a8c2819981eecc6c7049d318313bb6619306a466c1bf45052ad`。
- `review/G4_full_scores_scenes161_to170_actual.json`：`21aad8cbc1e79baa683d0101cf5ff52ec4eeebe6a782886d25141ebd10a362b1`。
- `review/G4_full_scores_scenes161_to170_primary_review.json`：`5475c96540a42c30ebab7e37f86bc591338adb37349fa7adacb7b9f023b031b8`。
- `bootstrap/before170_report_primary_review.json`：`b48f4b555cac517ad3be0242a8eecc555612801fe196ff4d1bbe42dd78c40c81`，作为后续逐场景主审基线。
- 三步实际退出收据及完整日志分别保存在`bootstrap/prepare_class_scores170_actual*`、`bootstrap/full_scores_scenes161_to170_actual*`和`bootstrap/full_scores_scenes161_to170_primary_actual*`。

[准备入口](CLASSIFICATION_SCORES170_ENTRY.md)和交接身份`2882ec11080fa8aeb97daa65bd27bbe89ea279021b6c36258c5828631a8950af`保留创建时的待执行状态；其17份支持文件均保持原SHA/修改时间，不改写历史。本页记录随后实际完成的执行。最终有限来源清单仍需纳入本批脚本、差异、入口、准备/数值/主审记录、实际日志与退出收据、新基线、旧汇总归档和本文。

本检查点不扩大已有配对池82场景、条件诊断84场景或公开47场景范围，Replica仍为0/8。双卡继续后续科学队列；完整任务、正式统计、最终来源清单/17章审核和C1/C2尚未完成。
