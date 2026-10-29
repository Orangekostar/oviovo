# 220场景分类分数入口（待执行）

2026-10-26：六份入口脚本及完整差异已由主审复核，只读预检实际退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`；第211–220项合并主审创建时为1项就绪、9项等待。当前分类分数保持210场景/1680矩阵，汇总身份`22065e8dc9c8524c8acf615fd7efefdb298a4e1145d54911ce6613a3c2341646`。本次未归档旧210汇总或生成新矩阵。本文固定记录入口创建时状态，后续实际执行另页记录。

新增批次固定为`scene0046_00`、`scene0046_01`、`scene0046_02`、`scene0203_00`、`scene0203_01`、`scene0203_02`、`scene0088_00`、`scene0088_01`、`scene0088_02`、`scene0088_03`。各场景四项科学阶段、三项有序元数据核验及合并主审均实际退出0后，才允许归档旧210汇总并固定前220份收据。随后新增80矩阵，核验1760矩阵；旧1680条描述及3360份矩阵文件保持SHA/修改时间，3521份输出只读复用。原分类汇总为唯一可替换旧路径，其余3719份旧路径不变；新增160份文件后，稳定基线预期由3720增至3880份。owner/defined/undefined行数由实际收据确定。

数值循环、稳定softmax和访问限制三个AST单元与201–210、191–200和81–87入口一致。类别轴固定为ScanNet200的200类，场景数与类别数分别核对。固定收据工具及其既有三项实测通过证据保持原样，第221项及以后的收据不进入本批。生产源码和冻结配置保持原样，本次新增模型推理、训练及AP重算均为0。跳帧恢复入口904份历史文件/7份源码、gate64份和有限来源198份支持路径继续按SHA/mtime保护；旧audit真实退出1保留，放行尚待所有生产者和原五阶段终态。

入口计划`bootstrap/class_scores220_entry_plan.json`身份`f7f1520f5f78bc6ba1f6047cc2723f04db53bff67cf7f9a9838c72baa573b341`；预检`review/G4_class_scores220_entry_preflight.json`身份`c39b5527508fba4a4a2466760de539a86a4feb332102fbda0edcc538c1c6ab6c`；实际退出20收据`bootstrap/class_scores220_entry_preflight_actual_exit.json`身份`1cb7b016b24c2c4765c2fb5d5ebabf4eac7ef2a9d917d2099028c3f66be9f494`。3720份稳定文件及93份冻结文件通过SHA/修改时间校验。该预检只执行一次，其日志SHA为`870e4f52517c273db7d43e781fad12099c77368ab322f3660f7cbebdbdb59d81`。

父检查点见[210场景实测](CLASSIFICATION_SCORES210.md)：分类主审`6b19131eab84950e4721c4f7116e8c47efdf9e85da302a7249b24ec6072a2226`、实际退出0`c88525f22a0d55014cf87dc533542bef04060147cbaae087a3e6905299438b14`、3720份稳定基线`b465bdea0a9cb609f7431d65d2791030a969f4f1f100b2e60228fd10244f8169`。

第211–220项全部合并主审退出0后，重新核对当前汇总及固定源码SHA，在worktree顺序执行尚未运行的三个步骤。wrapper固定CPU及单线程BLAS；任何非零均保留原尝试，核对同一实际退出记录及日志。

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores220.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores220.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores220.py review
```

当前逐场景主审211/312、1688组；完整分类210/1680；配对82、诊断84、公开47及Replica0/8各保留原范围。GPU0/GPU2继续科学队列，完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。最终有限来源清单须纳入本批六份脚本、六份差异、计划、预检、实际日志/退出收据、本文及交接收据，后续实际执行证据再追加。
