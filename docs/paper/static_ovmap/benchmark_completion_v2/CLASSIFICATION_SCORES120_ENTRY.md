# 120场景分类分数入口（已执行；保留入口历史）

2026-10-24后续：prepare、numeric、review三步已真实退出0，120场景/960矩阵通过主审，见[CLASSIFICATION_SCORES120.md](CLASSIFICATION_SCORES120.md)。不要重跑同名one-shot。以下保留此前等待和入口准备时的记录；其中较早范围为历史。

2026-10-24：四份入口脚本已生成并完成完整差异复核。只读预检查真实退出20，状态为`WAITING_FOR_LIVE_SCENE_QUEUE`：第111–114项主审及实际退出链已验证，第115–120项仍等待双卡队列完成。分类分数仍为110场景/880矩阵，汇总身份`4d7932a5c76a0bb53f9379657d4ac8adc9c1e46f9b3c678d8697ea4533797057`；本轮未执行数值生成、归档旧110汇总或形成120场景结果。

新增批次固定为`scene0671_01`、`scene0618_00`、`scene0382_00`、`scene0382_01`、`scene0490_00`、`scene0621_00`、`scene0607_00`、`scene0607_01`、`scene0149_00`、`scene0695_00`。全部通过逐场景主审后，原子准备步骤才归档旧110汇总，固定纳入前120份预测收据；再生成80份新矩阵，并主审960份完整矩阵。既有880条描述及1760份矩阵文件保留，目标1921份输出只读复用，最终基线由2120增至2280份。新增owner/defined/undefined数待实测，不预填。

数值进程的原CAP、稳定softmax、N/Q与F等权分组、1e-12校验及访问限制共3个AST单元与原101–110及81–87入口逐字等价。原固定收据范围工具及3项通过收据复用，防止后台新增场景越过120范围。第112项的3项元数据成功明确包含中断前已观测的1项native实际0和恢复后2项新实际0，不改写原中断阶段退出未知的事实。数值进程继续禁止GT/几何访问、模型导入及worker启动。

入口计划`bootstrap/class_scores120_entry_plan.json`身份`e279b0c9f2b68513b907e1659f403d5aa9ffb5c4f929b5056b23dd7bb66f7302`。只读预检查`review/G4_class_scores120_entry_preflight.json`身份`34c033a9ee73c2c586bd51000a0f111ca68dfb67fa5dc42f721ff3e2ad8ace48`，实际退出20收据`bootstrap/class_scores120_entry_preflight_actual_exit.json`身份`53465a9e5881b5085a45f0de97b3d1e0698f5897de0f509d50dc7b60a8bda709`。93份冻结文件及2120份稳定结果SHA/mtime通过。该预检为已保留的单次执行，后续不要重跑同名one-shot。

待第111–120项全部真实主审退出0、当前110汇总及四份新脚本SHA重新核对后，在worktree依次执行以下三个尚未运行的步骤。任何一步非零都先检查相同日志和实际退出记录，不重新执行同名one-shot：

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores120.py prepare
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores120.py numeric
PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/execute_class_scores120.py review
```

通过后保留旧110汇总的原SHA/mtime，更新分类分数范围及后续逐场景审阅基线；配对池82、条件诊断84和公开47不得随之自动升级。当前主审范围仍114/312、912组，完整任务、正式统计、最终来源清单/17章审核及C1/C2未结项。最终有限来源须显式纳入本批新脚本、计划、预检及未来真实退出记录。
