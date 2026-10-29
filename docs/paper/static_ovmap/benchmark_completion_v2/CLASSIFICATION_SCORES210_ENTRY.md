# 210场景分类分数入口（待执行）

2026-10-26：六份入口脚本及完整差异已由主审复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`；第201–210项合并主审创建时为0项就绪、10项等待。当前分类分数保持200场景/1600矩阵，汇总身份`c542552de344a825e845d793a773b682d00ccda6bd76c5e19551acff9a53ee3e`。本次未归档旧200汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

新增批次固定为`scene0527_00`、`scene0663_00`、`scene0663_01`、`scene0663_02`、`scene0664_00`、`scene0664_01`、`scene0664_02`、`scene0334_00`、`scene0334_01`、`scene0334_02`。各场景四项科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧200汇总并固定前210份收据。随后新增80矩阵，核验1680矩阵；旧1600条描述及3200份矩阵文件保持SHA/修改时间，3361份输出只读复用。原分类汇总为唯一可替换旧路径，其余3559份旧路径不变；新增160份文件后，稳定基线预期由3560增至3720份。owner/defined/undefined行数由实际收据确定。

数值循环、稳定softmax和访问限制三个AST单元与191–200、181–190和81–87入口一致。类别轴固定为ScanNet200的200类，场景数与类别数分别核对。固定收据工具及其既有三项实测通过证据保持原样，第211项及以后的收据不得进入本批。未改生产源码或冻结配置，本次新增模型推理、训练及AP重算均为0。

入口计划`bootstrap/class_scores210_entry_plan.json`身份`c0d2c7b496e1038e60e17e3f8f3efa7c3c6bed18a053113be1adfe8f1845ce73`；预检`review/G4_class_scores210_entry_preflight.json`身份`e1ad649cb0b86705a60abc89928f275169d261c62ed3358e86b90da02d16cb13`；实际退出20收据`bootstrap/class_scores210_entry_preflight_actual_exit.json`身份`6ebdccb627a3123dab6bb084ffc1f83d090c698d8ba49d4cda5e86d26d1b3d8a`。3560份稳定文件及93份冻结文件通过SHA/修改时间校验。该预检只执行一次。

父检查点见[200场景实测](CLASSIFICATION_SCORES200.md)：分类主审`74f47fc3b8117c401ee4fde8c0ff494d1ce0822226df2c66ec030ca71a26daeb`、实际退出0`ebced36ce57c30b3248274039ac2a82787e7f8b121510feae568ff6f63cdf76e`、3560份稳定基线`acb2fae03644ee38386feaddcba556151439d2c935e6f03c3061327f2b0c6817`。

第201–210项全部合并主审退出0后，重新核对当前汇总及固定源码SHA，在worktree顺序执行尚未运行的三个步骤。wrapper固定CPU及单线程BLAS；任何非零均保留原尝试，核对同一实际退出记录及日志。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores210.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores210.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores210.py review
```

当前逐场景主审200/312、1600组；完整分类200/1600；配对82、诊断84、公开47及Replica0/8各保留原范围。GPU0/GPU2继续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出收据、本文及交接收据，后续实际执行证据再追加。
