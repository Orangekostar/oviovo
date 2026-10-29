# 190场景分类分数入口（待执行）

2026-10-25：六份入口脚本及完整差异已复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`，第181–190项合并主审均尚未完成。当前分类分数保持180场景/1440矩阵，汇总身份`6b94f6ae10f774342524c2c36bfc89cb1d02c6df989c2b4915e0d5ece7afe5d6`。本次没有归档旧180汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

固定新增批次为`scene0064_00`、`scene0064_01`、`scene0356_00`、`scene0356_01`、`scene0356_02`、`scene0030_00`、`scene0030_01`、`scene0030_02`、`scene0222_00`、`scene0222_01`。每项四个科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧180汇总并固定前190份收据。随后新增80矩阵，复核1520矩阵；旧1440条描述和2880份矩阵文件保留，3041份输出只读复用。只允许替换原分类汇总，其余3239份旧路径保持SHA/mtime；新增160文件后，稳定基线预期由3240增至3400份。新增owner/defined/undefined数量由实际收据决定。

数值循环、稳定softmax和访问限制三个AST单元与171–180、161–170及81–87入口一致。固定收据范围工具及既有三项实测通过收据继续复用，后台第191项及以后不得进入本批。生产源码和冻结配置保持不变；本次没有新增模型推理、训练或AP重算。

入口计划`bootstrap/class_scores190_entry_plan.json`身份`9c9f77cd4d7ecc8f3ae895497a7f64f984b75e265d4cd343ef0c7cd2c99af802`。只读预检`review/G4_class_scores190_entry_preflight.json`身份`6a28bbb983868810bdb525db254acf5c49cd5fd1447ba51bbf3ef4108868271d`，实际退出20收据`bootstrap/class_scores190_entry_preflight_actual_exit.json`身份`e635117aa9ff3cf706d1d470ff6105075ae74275474d79198ae9de3a907ab7eb`。3240份稳定文件及93份冻结文件的SHA/mtime通过核验。该预检为单次执行，不重复运行。

当前父检查点详见[180场景实测](CLASSIFICATION_SCORES180.md)：分类主审`2c1ed9daca855eb5f7692e1bb4db49a29bb14d18baad969b5812b1e7c01be41b`、实际退出0`ea89ef21570702f33c70f80de7c2a2f1cc08b0f592c8ea6d77e40328d8462a1f`，3240份稳定基线`aa8cf0c4ca14aced92486beef0b5d9cd85b9491640d3ade2195a5698fcea16a7`。

待第181–190项全部合并主审实际退出0，重新核对当前180汇总和固定源码SHA后，在worktree按顺序执行尚未运行的三个步骤；wrapper将子进程固定为CPU和单线程BLAS。任何步骤非零，保留原尝试并核对同一日志和实际退出记录。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores190.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores190.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores190.py review
```

当前逐场景主审180/312、1440组，完整分类180/1440；配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续后续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出记录、本文及交接收据；未来数值执行证据完成后再追加。
