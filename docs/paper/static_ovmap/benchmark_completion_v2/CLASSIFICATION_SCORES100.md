# 100场景完整分类分数验收

当前本地完整分类分数覆盖100个ScanNet场景、800份200类矩阵，共58296个owner-method行：53836行有定义、4460行无定义。分类缓存、自有/OVI逐场景主审100、已审核配对池化82、诊断84、公开包47/表格45分别记录；完整312个ScanNet场景及8个Replica场景尚未完成。

本批新增协议第91–100项的80份矩阵、4416行和883200个类别分数位置，其中4124行有定义、292行无定义。输入严格固定为协议前100份预测收据，后续运行场景不自动纳入。

| 场景 | owner | F可用 | MF关闭unknown | MF开启unknown |
| --- | ---: | ---: | ---: | ---: |
| scene0357_00 | 87 | 79 | 8 | 4 |
| scene0357_01 | 98 | 90 | 8 | 3 |
| scene0430_00 | 61 | 57 | 4 | 1 |
| scene0430_01 | 43 | 39 | 4 | 2 |
| scene0574_00 | 46 | 43 | 3 | 0 |
| scene0608_02 | 117 | 100 | 17 | 9 |
| scene0651_00 | 15 | 14 | 1 | 0 |
| scene0651_01 | 16 | 15 | 1 | 0 |
| scene0651_02 | 30 | 27 | 3 | 1 |
| scene0689_00 | 39 | 36 | 3 | 1 |

每场景包含U00/U10/U01/U11与种子17/29的8份矩阵。数值复算沿用原稳定softmax、共享angle_cap及独立逐owner循环，三段数值/访问守卫AST与90场景版本及原恢复入口一致；新矩阵在1e-12绝对/相对容差内一致。标签、confidence、exported、owner与类别轴均与原冻结预测一致；无定义行保留NaN分数、defined=false及label=0。

数值进程未访问GT标签或原始几何，未启动模型、训练或评分worker。50份既有GT收据仅在准备与主审进程盲核对SHA/mtime。3797份输入/既有证据及1440份原矩阵文件不变；重复准备1601份当前文件的SHA/mtime保持不变。源码预审`f58cc15199dabb2e8509297b93694d41583540ecc3926d673c062841b9f01f85`当时的99/100待验收状态保留为历史，实际准备已要求第91–100项主审均actual0。

| 实际步骤 | 子进程退出 | 收据identity |
| --- | ---: | --- |
| 准备 | 0 | `c22728837cd76e7534fa42e381619ac404d9025b5af6677f28ee92ba5f90bcbe` |
| 数值复算 | 0 | `fbe9b035f8a496dc5be9cd243e4269eb49f6fa4d374f775142fb74c8ee4efd6c` |
| 主审 | 0 | `44858c3f46f0d102632785d4c98240f531aa7e245dc22df57e76b50a184736e3` |

准备证明：`review/G4_100scene_class_score_preparation_primary_review.json`，`c743e74e58559a225aeae276495d0fbaa85343abd548b39a1b89905a79e14331`。

数值证明：`review/G4_full_scores_scenes91_to100_actual.json`，`3372e1f638b2afed44cb668d24c930f6cfab4160a70e4e350c4e350448e12d84`。

主审证明：`review/G4_full_scores_scenes91_to100_primary_review.json`，`10eb1f6033cad218b1b3aeb7fc9b77d8fbbd608a959dc1518277f0fa59a01fa9`。主审核对全部800份manifest与数组SHA/字节数、原预测来源、完整owner域及方法/种子键；旧720条汇总记录原样保留，未重算AP。

当前汇总：`publication/class_scores.json`，`b5e67b8d1af1d5ad806319378ab0e4065b70b96d2bf774afce4544be9f8772e2`。

旧90场景汇总`60f905115d22de97c636b10f76a553157a5d6ddc892857da8e9e8dfdd940f7b2`已普通复制至`review/history/class_scores_90scenes_60f905115d22de97c636b10f76a553157a5d6ddc892857da8e9e8dfdd940f7b2.json`，SHA/mtime与旧1800份基线一致，归档使用独立inode。旧1800份基线保留为历史；唯一授权变化的既有输出是分类汇总90→100，其余1799份旧路径及93份原冻结文件保持SHA/mtime。新增160份矩阵文件后建立新1960份稳定基线：`bootstrap/before100_report_primary_review.json`，`0e43dc07862d4b8b930ea4fa4d81cb51eee27f9f6a1903f77387b1946eb904f9`。后续接续检查使用此新基线，历史一次性入口不重跑。

执行源码为`bootstrap/prepare_class_scores100.py`、`validate_full_scores_scenes91_to100.py`、`review_full_scores91_to100.py`、`execute_class_scores100.py`，沿用已通过3项控制的`snapshot_score_scope.py`。精确源码SHA、实际命令及日志SHA见准备和真实退出收据；四文件通过预审后保持不变。

双卡科学队列继续，本次矩阵物化新增GPU推理0。本次证据不证明效率、统计显著性、完整benchmark优势或在线30FPS；完整任务的正式统计、最终来源包、提示词逐项审核及C1/C2仍未完成。
