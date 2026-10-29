# 150场景分类分数入口（待执行）

2026-10-25：六份入口脚本及完整差异已复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`，第141–150项合并主审均尚未完成。当前分类分数保持140场景/1120矩阵，汇总身份`dff0264f0320f6107965111ff6a6051a7257d7e12694cf186f698757d663dacb`。本次没有归档旧140汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

固定新增批次为`scene0144_00`、`scene0144_01`、`scene0494_00`、`scene0278_00`、`scene0278_01`、`scene0316_00`、`scene0609_00`、`scene0609_01`、`scene0609_02`、`scene0609_03`。每项四个科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧140汇总并固定前150份收据。随后新增80矩阵，复核1200矩阵；旧1120条描述和2240份矩阵文件保留，2401份输出只读复用。只允许替换原分类汇总，其余2599份旧路径保持SHA/mtime；新增160文件后，稳定基线预期由2600增至2760份。新增owner/defined/undefined数量由实际收据决定。

数值循环、稳定softmax和访问限制三个AST单元与131–140、121–130及81–87入口一致。固定收据范围工具及既有三项实测通过收据继续复用，后台第151项及以后不得进入本批。生产源码和冻结配置保持不变；本次没有新增模型推理、训练或AP重算。

入口计划`bootstrap/class_scores150_entry_plan.json`身份`2d539cb13a4445681daec5c65aeb52be82843e9c4013bfe465ca472b533f79ba`。只读预检`review/G4_class_scores150_entry_preflight.json`身份`e932f3587d4c43ed5ca7b2916eadd5a6798648baf241b73673776d65876e0c2b`，实际退出20收据`bootstrap/class_scores150_entry_preflight_actual_exit.json`身份`9dc8f5f996a3cf736db60904a68786b5c8dfa3e80eb1ae5336c6f23545727d4e`。2600份稳定文件及93份冻结文件的SHA/mtime通过核验。该预检为单次执行，不重复运行。

当前父检查点详见[140场景实测](CLASSIFICATION_SCORES140.md)：分类主审`9877754015ba294cd5152a015d547ebe0622287df4f70719ee351fe209d845bf`、实际退出0`022261756bf7048e644c384b1f3e966293abf06fae911e427349a5b29c9a3bce`，2600份稳定基线`e39796f3156be2acc6e529087690734e9fe1a0cfae7388e8480ccf2c57f43828`。

待第141–150项全部合并主审实际退出0，重新核对当前140汇总和固定源码SHA后，在worktree按顺序执行尚未运行的三个步骤；wrapper将子进程固定为CPU和单线程BLAS。任何步骤非零，保留原尝试并核对同一日志和实际退出记录。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores150.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores150.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores150.py review
```

当前逐场景主审140/312、1120组，完整分类140/1120；配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续后续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出记录、本文及交接收据；未来数值执行证据完成后再追加。
