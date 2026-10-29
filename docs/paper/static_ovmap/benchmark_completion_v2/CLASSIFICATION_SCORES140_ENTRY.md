# 140场景分类分数入口（待执行）

2026-10-24：六份入口脚本及完整差异已复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`，第131–140项合并主审均尚未完成。当前分类分数保持130场景/1040矩阵，汇总身份`5430ebbfa303e8e27e71c09c5f08cabe6c5275fe7034aa4652b6a8193d7e025c`。本次没有归档旧130汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

固定新增批次为`scene0629_01`、`scene0629_02`、`scene0496_00`、`scene0633_00`、`scene0633_01`、`scene0518_00`、`scene0652_00`、`scene0406_00`、`scene0406_01`、`scene0406_02`。每项四个科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧130汇总并固定前140份收据。随后新增80矩阵，复核1120矩阵；旧1040条描述和2080份矩阵文件保留，2241份输出只读复用。只允许替换原分类汇总，其余2439份旧路径保持SHA/mtime；新增160文件后，稳定基线预期由2440增至2600份。新增owner/defined/undefined数量由实际收据决定。

数值循环、稳定softmax和访问限制三个AST单元与121–130、111–120及81–87入口一致。固定收据范围工具及既有三项实测通过收据继续复用，后台第141项及以后不得进入本批。生产源码和冻结配置保持不变；本次没有新增模型推理、训练或AP重算。

入口计划`bootstrap/class_scores140_entry_plan.json`身份`0792478604f2ba9b02d435b316a358e9460fb470a7cbc0c8e64237893e76d2a8`。只读预检`review/G4_class_scores140_entry_preflight.json`身份`eae600adb5dbea00fb6d6e984291873350ee28e5aff10475fc285dfcd91cd728`，实际退出20收据`bootstrap/class_scores140_entry_preflight_actual_exit.json`身份`387bf3b77ced3d8a068afdfdd85e88b0cdbd56fb940209fd9ebcf88181781286`。2440份稳定文件及93份冻结文件的SHA/mtime通过核验。该预检为单次执行，不重复运行。

当前父检查点详见[130场景实测](CLASSIFICATION_SCORES130.md)：分类主审`f58f0b6ca26b37064ac8ee3ea65b39e5cf2be379eb0323f1f12b2af77a7a2870`、实际退出0`c0054e7e1346c1a86dad4a1e50abf6e10d4a38404528b9f490ddf327e8ed75a9`，2440份稳定基线`b61dc58427a1914c934cb863f9018a4f8c42b93e63a36de3a634364f6d6948b5`。

待第131–140项全部合并主审实际退出0，重新核对当前130汇总和固定源码SHA后，在worktree按顺序执行尚未运行的三个步骤；wrapper将子进程固定为CPU和单线程BLAS。任何步骤非零，保留原尝试并核对同一日志和实际退出记录。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores140.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores140.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores140.py review
```

当前逐场景主审130/312、1040组，完整分类130/1040；配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续第131、132项，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出记录、本文及交接收据；未来数值执行证据完成后再追加。
