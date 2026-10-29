# node7 Replica progress V8

复用 node7 原有 Replica 八场景；本次没有重新下载 Replica 数据集。

原生 RAW 四组实验已完成全部八场景，共 32/64 标准单元及四个全场景 RAW 汇总。U01/U11 和 Replica bootstrap 仍缺 14 类训练词表归属依据。

完成评分：OpenMask3D 3/8；MaskClustering 8/8；OVI-MAP 8/8；Open3DIS 3/8。

已完成全八场景汇总：OVI-MAP、MaskClustering。

| Replica 全八场景汇总 | AP (%) | AP50 (%) | AP25 (%) |
| --- | ---: | ---: | ---: |
| OVI-MAP | 8.7893 | 19.9662 | 32.6683 |
| MaskClustering | 12.0052 | 24.7013 | 34.4891 |

以上结果均由原官方 evaluate_matches 对全部八场景共同计算，未平均各场景 AP。

Open3DIS：node7 GPU1/2 执行 room0/room1，node1 GPU0 执行 office3，node3 GPU0 执行 office4；office2 释放 GPU1 后提前启动 room2。原队列以后复用 room2 同一结果，不重复执行。GPU 利用率快照不等于任务进度。

node2 的 room1 仅完成 CPU 预检，发现 GPU 被占后未启动 GPU 推理；六个旧 CPU 等待调度进程的退出记录保留，未停止任何模型或评分工作进程。

room0、room1 的 Grounded SAM-HQ 阶段均已成功。原控制器在下一阶段单次 GPU 空闲检查退出；V6 续跑复用四项成功结果，三维融合也已通过，现进入特征细化。新增检查至多等待 30 秒，保留每次观测，要求所选卡零占用、零利用率且无计算进程；8 项测试通过。

已完成场景均通过作者与统一评测器 480 项 AP 数组逐项一致检查。完整目标尚未完成，发布仍关闭。

评审标识：498fa8e7e329c1c6c5c82169df9e4af26f7b5b453197e541d24dc2237c3aa495
