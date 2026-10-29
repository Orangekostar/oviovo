# 第101–110场景完整分类分数交付

2026-10-24：当前为110场景、880份完整矩阵；63432个owner×方法行，其中58532行有分数、4900行无可用源。主审 `1f5e12f68edc6ad326d231ea8660eda670ab86aa20f0eec218d81ad63c7a4ef5`，真实主审退出0 `2c02e5898fb36459f46cfc0b54600089e89cfe182f1e2b1c0337efe56bd6b43a`。汇总身份 `4d7932a5c76a0bb53f9379657d4ac8adc9c1e46f9b3c678d8697ea4533797057`。

新增80份矩阵对应5136个对象行、1027200个类别位置，其中4696行有分数、440行无可用源。独立复算使用原CAP算子、稳定softmax和N/Q–F分组，容差atol/rtol均为1e-12；全部新标签、confidence和export标记与冻结预测一致。MFoff不读取N/Q；无源行保留未定义标记，不补造分类证据。

| 协议项 | 场景 | owner | F可用 | MF关闭unknown | MF开启unknown |
|---|---|---:|---:|---:|---:|
| 101 | scene0574_01 | 39 | 36 | 3 | 1 |
| 102 | scene0574_02 | 48 | 45 | 3 | 1 |
| 103 | scene0329_00 | 104 | 92 | 12 | 6 |
| 104 | scene0329_01 | 100 | 86 | 14 | 9 |
| 105 | scene0329_02 | 116 | 99 | 17 | 12 |
| 106 | scene0153_00 | 21 | 19 | 2 | 1 |
| 107 | scene0153_01 | 26 | 23 | 3 | 3 |
| 108 | scene0616_00 | 64 | 52 | 12 | 4 |
| 109 | scene0616_01 | 66 | 62 | 4 | 1 |
| 110 | scene0671_00 | 58 | 56 | 2 | 0 |

旧100场景汇总以独立inode普通文件保留原SHA/mtime：`review/history/class_scores_100scenes_b5e67b8d1af1d5ad806319378ab0e4065b70b96d2bf774afce4544be9f8772e2.json`。仅原`publication/class_scores.json`路径获准更新；其余1959份旧路径及93份冻结文件不变，新增加160份manifest/数组文件，形成2120份新基线 `a4873e13b0545a89734758d8ad72aaa4a7f5f16b73cbc2d0bdd292fcac90b766`。原100场景的800条描述和1600份矩阵文件保持不变。

预检查、数值生成及主审三个原子步骤均实际退出0。数值过程耗时45.14秒，1761份输出在第二次读取后保持SHA/mtime；50份历史GT收据只在主代理流程做盲哈希，数值进程禁止访问GT、几何和worker，不导入torch/tensorflow。本批新增模型调用、训练及AP计算均为0。

已执行入口为`bootstrap/execute_class_scores110.py`的prepare、numeric、review三步；这些是保留历史的单次执行，后续不要重复运行。原始证据为`review/G4_110scene_class_score_preparation_primary_review.json`、`review/G4_full_scores_scenes101_to110_actual.json`、`review/G4_full_scores_scenes101_to110_primary_review.json`及对应bootstrap实际退出和日志。

逐场景自有/OVI主审同为110/312、880组。配对池化表仍是共同82场景，条件诊断84场景，公开索引47场景/表格45场景；本批没有刷新这些较早快照。完整312/8场景、正式统计、最终来源清单和17章审核、C1/C2及发布后实际resume仍待完成。
