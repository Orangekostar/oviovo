# 第111–120场景完整分类分数交付

2026-10-24：当前为120场景、960份完整矩阵；68416个owner×方法行，其中63032行有分数、5384行无可用源。主审`19d5dc182dbf91637983b8703a2ce05f6338b6e8acd424eb2be9da2ff0af7bae`，真实主审退出0`08740f7341dde69eeb8673317f8de22a287b9849600f581b124a45d5d324e156`。汇总身份`3438a989b15728089936f37316cacb20acac7698506a6c12bfb569c7495d64c0`。

新增80份矩阵对应4984个对象行、996800个类别位置，其中4500行有分数、484行无可用源。独立复算使用原CAP算子、稳定softmax和N/Q–F分组，atol/rtol均为1e-12；新标签、confidence和export标记与冻结预测一致。MFoff不读取N/Q；无源行保留未定义标记。

| 协议项 | 场景 | owner | F可用 | MF关闭unknown | MF开启unknown |
|---|---|---:|---:|---:|---:|
| 111 | scene0671_01 | 46 | 43 | 3 | 1 |
| 112 | scene0618_00 | 25 | 23 | 2 | 1 |
| 113 | scene0382_00 | 16 | 15 | 1 | 0 |
| 114 | scene0382_01 | 18 | 16 | 2 | 0 |
| 115 | scene0490_00 | 55 | 46 | 9 | 4 |
| 116 | scene0621_00 | 94 | 83 | 11 | 3 |
| 117 | scene0607_00 | 59 | 51 | 8 | 4 |
| 118 | scene0607_01 | 49 | 45 | 4 | 2 |
| 119 | scene0149_00 | 97 | 82 | 15 | 9 |
| 120 | scene0695_00 | 164 | 133 | 31 | 11 |

旧110场景汇总以独立inode普通文件保留原SHA/mtime：`review/history/class_scores_110scenes_4d7932a5c76a0bb53f9379657d4ac8adc9c1e46f9b3c678d8697ea4533797057.json`。仅原`publication/class_scores.json`路径获准更新；其余2119份旧路径及93份冻结文件不变，新增加160份manifest/数组文件，形成2280份新基线`6b4b9a968d4392cf69b7670cf958b0acb5d42327f8335465c13d928a01a59b91`。旧880条描述及1760份矩阵文件保持不变。

准备、数值生成及主审三个步骤均实际退出0。数值过程耗时39.66秒，1921份输出在第二次读取后保持SHA/mtime；50份历史GT收据只在主代理流程做盲哈希，数值进程禁止访问GT、几何和worker，不导入torch/tensorflow。本批新增模型调用、训练及AP计算均为0。

第112项元数据链明确保留中断前native实际退出0，恢复后新增两项实际退出0；原中断N/Q阶段的实际退出和完整耗时仍未知，不补零。其他九项按各自已审核完成链纳入；第113项使用恢复后的完整重建结果。

已执行入口为`bootstrap/execute_class_scores120.py`的prepare、numeric、review三步；这些是保留历史的单次执行，不要重复运行。准备主审`1ef418d2b9111971ef332b0e9f63a8518811ed0ec6de4f19f082d40d5cbb5edf`、实际退出0`ab43620d714395bd87bc7c168caddb85ada409c1c347f593fb30ad7c1e308839`；数值验证`76ad5d5f8fa55c0842e853c41e5ef4948b159160e10bc0137d51cfd3eaf90c46`、实际退出0`33a92995f00398c0c831c0ae96ce285ec115f0af4a23cac72e62e58cdfedb2ba`。原始证据为`review/G4_120scene_class_score_preparation_primary_review.json`、`review/G4_full_scores_scenes111_to120_actual.json`、`review/G4_full_scores_scenes111_to120_primary_review.json`及对应bootstrap日志。

逐场景自有/OVI主审为120/312、960组。配对池化表仍为共同82场景、条件诊断84场景、公开47场景/45场景表格；本批未刷新这些快照。GPU0/GPU2场景队列继续，完整312/8场景、正式统计、最终来源清单和17章审核、C1/C2及发布后实际resume尚待完成。
