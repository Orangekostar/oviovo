# 200场景分类分数入口（待执行）

2026-10-25：六份入口脚本及完整差异已复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`，第191–200项合并主审均尚未完成。当前分类分数保持190场景/1520矩阵，汇总身份`4ea374b5c6d51864621128a20c9036f2125967dc05b40457c25d74cf1d722892`。本次没有归档旧190汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

固定新增批次为`scene0338_00`、`scene0338_01`、`scene0338_02`、`scene0378_00`、`scene0378_01`、`scene0378_02`、`scene0660_00`、`scene0553_00`、`scene0553_01`、`scene0553_02`。每项四个科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧190汇总并固定前200份收据。随后新增80矩阵，复核1600矩阵；旧1520条描述和3040份矩阵文件保留，3201份输出只读复用。只允许替换原分类汇总，其余3399份旧路径保持SHA/mtime；新增160文件后，稳定基线预期由3400增至3560份。新增owner/defined/undefined数量由实际收据决定。

数值循环、稳定softmax和访问限制三个AST单元与181–190、171–180及81–87入口一致。固定收据范围工具及既有三项实测通过收据继续复用，后台第201项及以后不得进入本批。生产源码和冻结配置保持不变；本次没有新增模型推理、训练或AP重算。

入口计划`bootstrap/class_scores200_entry_plan.json`身份`8eb69258c98f5548b213166dbd9cd209ada2c3015d731c0ebacff18170c2c703`。只读预检`review/G4_class_scores200_entry_preflight.json`身份`cf6aea7bfa7282665ffda57f6f0f7850d86da6af3d94c2442e296fc12abe18c2`，实际退出20收据`bootstrap/class_scores200_entry_preflight_actual_exit.json`身份`2df80061a3199f8752abe015954b4f1f082808a3229c5d165ea48a77cc7fd9d1`。3400份稳定文件及93份冻结文件的SHA/mtime通过核验。该预检为单次执行，不重复运行。

当前父检查点详见[190场景实测](CLASSIFICATION_SCORES190.md)：分类主审`0f9186d073915a2caf839c8058e94cf0558602f241d1459c6751e3c23c6fb19d`、实际退出0`cd59f4dc1cc7bc9269a6bf14665fda5b6e7ddb2493988a3d15258774ae2c73d7`，3400份稳定基线`e8edf51294b2d884e9dccabfe904f1f059d4f378a29d3e3b2e5983e14ba72fee`。

待第191–200项全部合并主审实际退出0，重新核对当前190汇总和固定源码SHA后，在worktree按顺序执行尚未运行的三个步骤；wrapper将子进程固定为CPU和单线程BLAS。任何步骤非零，保留原尝试并核对同一日志和实际退出记录。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores200.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores200.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores200.py review
```

当前逐场景主审190/312、1520组，完整分类190/1520；配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续后续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出记录、本文及交接收据；未来数值执行证据完成后再追加。
