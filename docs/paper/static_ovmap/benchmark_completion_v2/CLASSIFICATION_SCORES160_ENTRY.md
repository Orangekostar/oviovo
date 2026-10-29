# 160场景分类分数入口（待执行）

2026-10-25：六份入口脚本及完整差异已复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`，第151–160项合并主审均尚未完成。当前分类分数保持150场景/1200矩阵，汇总身份`67c738d237d8ce249cab436182bd34a939877542eed8f9696609cc92ff69a231`。本次没有归档旧150汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

固定新增批次为`scene0084_00`、`scene0084_01`、`scene0084_02`、`scene0696_00`、`scene0696_01`、`scene0696_02`、`scene0351_00`、`scene0351_01`、`scene0643_00`、`scene0644_00`。每项四个科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧150汇总并固定前160份收据。随后新增80矩阵，复核1280矩阵；旧1200条描述和2400份矩阵文件保留，2561份输出只读复用。只允许替换原分类汇总，其余2759份旧路径保持SHA/mtime；新增160文件后，稳定基线预期由2760增至2920份。新增owner/defined/undefined数量由实际收据决定。

数值循环、稳定softmax和访问限制三个AST单元与141–150、131–140及81–87入口一致。固定收据范围工具及既有三项实测通过收据继续复用，后台第161项及以后不得进入本批。生产源码和冻结配置保持不变；本次没有新增模型推理、训练或AP重算。

入口计划`bootstrap/class_scores160_entry_plan.json`身份`083df85a54dd7c26834e35a5e81cd547b152a1a0d673f64764623c0161e32b8e`。只读预检`review/G4_class_scores160_entry_preflight.json`身份`dbd2e38a0828069f4218cba6e05d7cfb8d7962fe527b0a8bbcbec4af1a143887`，实际退出20收据`bootstrap/class_scores160_entry_preflight_actual_exit.json`身份`c360fa8ab269a63792a12e8a9074566ea6717f893143e6cb2534f435c4051fdc`。2760份稳定文件及93份冻结文件的SHA/mtime通过核验。该预检为单次执行，不重复运行。

当前父检查点详见[150场景实测](CLASSIFICATION_SCORES150.md)：分类主审`17b331316eebbd7d38729628a22422bdeccb6bc9289908be64ea097106e5590f`、实际退出0`68e820cc94da2b4af68702714287684120f27ccc543c33a34a702d3679dccc3a`，2760份稳定基线`f61b1f5aa8e3bf1a52360e345235622e738655ee22deff6f7294f59c7175b3ee`。

待第151–160项全部合并主审实际退出0，重新核对当前150汇总和固定源码SHA后，在worktree按顺序执行尚未运行的三个步骤；wrapper将子进程固定为CPU和单线程BLAS。任何步骤非零，保留原尝试并核对同一日志和实际退出记录。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores160.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores160.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores160.py review
```

当前逐场景主审150/312、1200组，完整分类150/1200；配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续后续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出记录、本文及交接收据；未来数值执行证据完成后再追加。
