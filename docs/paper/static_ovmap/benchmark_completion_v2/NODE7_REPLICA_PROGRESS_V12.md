# node7 Replica progress V12

复用 node7 原有 Replica 八场景；本次没有重新下载 Replica 数据集。

原生 RAW 四组实验已完成全部八场景，共 32/64 标准单元及四个全场景 RAW 汇总。U01/U11 和 Replica bootstrap 仍缺 14 类训练词表归属依据。

完成评分：OpenMask3D 7/8；MaskClustering 8/8；OVI-MAP 8/8；Open3DIS 8/8。

已完成全八场景汇总：OVI-MAP、MaskClustering。

| Replica 全八场景汇总 | AP (%) | AP50 (%) | AP25 (%) |
| --- | ---: | ---: | ---: |
| OVI-MAP | 8.7893 | 19.9662 | 32.6683 |
| MaskClustering | 12.0052 | 24.7013 | 34.4891 |

以上结果均由原官方 evaluate_matches 对全部八场景共同计算，未平均各场景 AP。

Open3DIS 完整推理已完成 8/8 场景。当前仍运行的推理控制器为 0 个。

| 场景 | 节点 | GPU | 当前状态 |
| --- | --- | ---: | --- |
| office1 | node3 | 0 | 评分完成 |
| office2 | node3 | 1 | 评分完成 |
| office3 | node1 | 0 | 评分完成 |
| office4 | node3 | 0 | 评分完成 |
| room0 | node107 | 1 | 评分完成 |
| room1 | node107 | 2 | 评分完成 |
| room2 | node3 | 1 | 评分完成 |

三个 room 场景的提前评分队列等待各自完整推理真实退出 0 后再启动；主队列复用同一评分结果。

Open3DIS 已导入 41 个已结束 GPU 工作进程的耗时核算，累计 5.4827 小时（含工作进程内的 CPU 开销）；控制器等待时间未计入；本次 Open3DIS 所选 GPU 工作进程的核算已全部完成。核算导入没有执行模型。

node2 的 room1 仅完成 CPU 预检，发现 GPU 被占后未启动 GPU 推理；七个旧 CPU 等待调度进程的退出记录保留，未停止任何模型或评分工作进程。

room0、room1 的 Grounded SAM-HQ 阶段均已成功。原控制器在下一阶段单次 GPU 空闲检查退出；V6 续跑复用四项成功结果，三维融合也已通过；room1 的完整推理及评分也已成功。新增检查至多等待 30 秒，保留每次观测，要求所选卡零占用、零利用率且无计算进程；8 项测试通过。

输入核验索引优化的 19 项测试和真实私有副本复核已通过；返回条目、identity、磁盘字节与修改时间一致。仅给尚未启动的 OpenMask3D 全场景汇总更换索引缓存 context；原锁、合并及异常在真实变化时仍完整执行。当前评分与 Open3DIS 汇总保持原进程。

已完成场景均通过作者与统一评测器 480 项 AP 数组逐项一致检查。完整目标尚未完成，发布仍关闭。

评审标识：5a448aab791d647e5a4a6c20d4072d4b40e26e328e9537356ed54ddc153184a6
