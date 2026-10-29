# 130场景分类分数入口（待执行）

2026-10-24：四份入口脚本已生成并完成完整差异复核。只读预检查真实退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`：第121–130项主审均仍待完成。当前分类分数为120场景/960矩阵，汇总身份`3438a989b15728089936f37316cacb20acac7698506a6c12bfb569c7495d64c0`。本次只读预检未生成新矩阵或归档旧120汇总。

固定新增批次为`scene0695_01`、`scene0695_02`、`scene0695_03`、`scene0389_00`、`scene0377_00`、`scene0377_01`、`scene0377_02`、`scene0342_00`、`scene0139_00`、`scene0629_00`。全部逐场景主审真实退出0后，准备步骤才归档旧120汇总并固定前130份预测收据；随后生成80份新矩阵，主审1040份完整矩阵。旧960条描述及1920份矩阵文件保留，目标2081份输出只读复用，稳定基线将由2280增至2440份，只有原分类汇总路径允许替换，其余2279份旧路径保留SHA/mtime。新owner/defined/undefined数量由实测收据决定，不预填。

数值循环、稳定softmax和访问限制共3个AST单元与111–120、101–110和81–87入口逐字等价。固定收据范围工具及其既有3项实测通过收据复用，后台新场景不能越过130范围。第121–130项为普通完成链，要求每项4个科学阶段及3项元数据实际退出0；第112项中断与恢复的原始证据保留在父检查点，不作为本批新执行记录。

入口计划`bootstrap/class_scores130_entry_plan.json`身份`1b904c489ceeba1b78074c286d5858d30af00e9542cfa912ac590e8e7ca2f3d1`。只读预检查`review/G4_class_scores130_entry_preflight.json`身份`b03fe8fcf31bdb9f871bfd55560507c0ab7ed59995e8550b0826847f11917660`，实际退出20收据`bootstrap/class_scores130_entry_preflight_actual_exit.json`身份`681f658f174506e926ccad1bdc37bce7217d430e71d28afe52cacbe9200c5d6f`。2280份稳定结果及93份冻结文件SHA/mtime通过；预检为保留历史的单次执行，不重复运行。

待第121–130项全部真实主审退出0，重新核对当前120汇总及四份新脚本SHA后，在worktree顺序执行以下尚未运行的步骤。任何一步非零，先检查同一日志和实际退出记录，保留原尝试：

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores130.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores130.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores130.py review
```

通过后保留旧120汇总的原SHA/mtime，并更新分类分数范围与后续逐场景审阅基线。当前逐场景主审120/312、960组，配对池82、条件诊断84、公开47及Replica0/8分别维持原范围。GPU0/GPU2继续第121/122项；完整科学任务、正式统计、最终来源清单/17章审核与C1/C2未结项。最终有限来源须纳入本批六份脚本、差异、计划、预检和真实日志/退出记录、本文及交接收据；未来数值运行证据完成后再追加。
