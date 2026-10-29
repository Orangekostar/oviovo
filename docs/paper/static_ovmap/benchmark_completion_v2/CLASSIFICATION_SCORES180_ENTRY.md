# 180场景分类分数入口（待执行）

2026-10-25：六份入口脚本及完整差异已复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`，第171–180项合并主审均尚未完成。当前分类分数保持170场景/1360矩阵，汇总身份`576608f8c1b1a133115de4363d39923454530f3a3ccbfab9afe13d3fdd3b7518`。本次没有归档旧170汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

固定新增批次为`scene0353_01`、`scene0353_02`、`scene0559_00`、`scene0559_01`、`scene0559_02`、`scene0593_00`、`scene0593_01`、`scene0246_00`、`scene0653_00`、`scene0653_01`。每项四个科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧170汇总并固定前180份收据。随后新增80矩阵，复核1440矩阵；旧1360条描述和2720份矩阵文件保留，2881份输出只读复用。只允许替换原分类汇总，其余3079份旧路径保持SHA/mtime；新增160文件后，稳定基线预期由3080增至3240份。新增owner/defined/undefined数量由实际收据决定。

数值循环、稳定softmax和访问限制三个AST单元与161–170、151–160及81–87入口一致。固定收据范围工具及既有三项实测通过收据继续复用，后台第181项及以后不得进入本批。生产源码和冻结配置保持不变；本次没有新增模型推理、训练或AP重算。

入口计划`bootstrap/class_scores180_entry_plan.json`身份`ba7e8281094dbdf41683aceaf1d7a3e9488d08e0815641dbeb0f47f5a0c9f357`。只读预检`review/G4_class_scores180_entry_preflight.json`身份`48ce1e315cb8f4d56a080195f01d4fd7a5284c390a4c3cae491215f4373dc8c2`，实际退出20收据`bootstrap/class_scores180_entry_preflight_actual_exit.json`身份`662b8157153931a6f7d9eb5194d106e796d82b86e67b20ae4245b709df979236`。3080份稳定文件及93份冻结文件的SHA/mtime通过核验。该预检为单次执行，不重复运行。

当前父检查点详见[170场景实测](CLASSIFICATION_SCORES170.md)：分类主审`5475c96540a42c30ebab7e37f86bc591338adb37349fa7adacb7b9f023b031b8`、实际退出0`b6cccfdbc1ddccabc1446cd292470d62da90e5ebd3ff85605c09fdd4d15bfb02`，3080份稳定基线`b48f4b555cac517ad3be0242a8eecc555612801fe196ff4d1bbe42dd78c40c81`。

待第171–180项全部合并主审实际退出0，重新核对当前170汇总和固定源码SHA后，在worktree按顺序执行尚未运行的三个步骤；wrapper将子进程固定为CPU和单线程BLAS。任何步骤非零，保留原尝试并核对同一日志和实际退出记录。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores180.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores180.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores180.py review
```

当前逐场景主审170/312、1360组，完整分类170/1360；配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续后续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出记录、本文及交接收据；未来数值执行证据完成后再追加。
