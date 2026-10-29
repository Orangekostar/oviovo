# 170场景分类分数入口（待执行）

2026-10-25：六份入口脚本及完整差异已复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`，第161–170项合并主审均尚未完成。当前分类分数保持160场景/1280矩阵，汇总身份`679c87e019c5147783371558249657b9510be6e5d74c88c03bb1ad77196ad053`。本次没有归档旧160汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

固定新增批次为`scene0645_00`、`scene0645_01`、`scene0645_02`、`scene0081_00`、`scene0081_01`、`scene0081_02`、`scene0647_00`、`scene0647_01`、`scene0535_00`、`scene0353_00`。每项四个科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧160汇总并固定前170份收据。随后新增80矩阵，复核1360矩阵；旧1280条描述和2560份矩阵文件保留，2721份输出只读复用。只允许替换原分类汇总，其余2919份旧路径保持SHA/mtime；新增160文件后，稳定基线预期由2920增至3080份。新增owner/defined/undefined数量由实际收据决定。

数值循环、稳定softmax和访问限制三个AST单元与151–160、141–150及81–87入口一致。固定收据范围工具及既有三项实测通过收据继续复用，后台第171项及以后不得进入本批。生产源码和冻结配置保持不变；本次没有新增模型推理、训练或AP重算。

入口计划`bootstrap/class_scores170_entry_plan.json`身份`262d1da1b18ec3a4f812c00f32ae6064e7be6703281ed20747ca911ac029844b`。只读预检`review/G4_class_scores170_entry_preflight.json`身份`fb8de0cdd177f0110c3425e3fb8426490914c1177480b17fe675359c1fe0aec0`，实际退出20收据`bootstrap/class_scores170_entry_preflight_actual_exit.json`身份`39bbe10376a6bb69bb251f7a225076e1b3d5d019f7229b7b1ab6f2a0fbfdcb29`。2920份稳定文件及93份冻结文件的SHA/mtime通过核验。该预检为单次执行，不重复运行。

当前父检查点详见[160场景实测](CLASSIFICATION_SCORES160.md)：分类主审`5a2a2e42deb4ef19987cc89dba6d9a48c1122c97fa3cb0f27f33cc176e0bdba3`、实际退出0`59f3b4c40604f386f9a16b7f3bb2e38c0fefe420508815b89b1e65ead59aa9f5`，2920份稳定基线`718468c91b58448948693eca9cdc5508cd95964e3030121314e4c514919f94bf`。

待第161–170项全部合并主审实际退出0，重新核对当前160汇总和固定源码SHA后，在worktree按顺序执行尚未运行的三个步骤；wrapper将子进程固定为CPU和单线程BLAS。任何步骤非零，保留原尝试并核对同一日志和实际退出记录。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores170.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores170.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores170.py review
```

当前逐场景主审160/312、1280组，完整分类160/1280；配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续后续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出记录、本文及交接收据；未来数值执行证据完成后再追加。
