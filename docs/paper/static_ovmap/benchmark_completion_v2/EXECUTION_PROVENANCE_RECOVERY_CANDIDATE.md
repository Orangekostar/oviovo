# 恢复证据来源构建器候选

2026-10-24完成隔离候选及实文件主审，支持执行包第16章要求的命令、失败历史和交接来源。当前逐场景主审仍116/312；本轮没有新增模型调用或AP计算。

旧`build_execution_provenance82.py`只允许bootstrap内的终态日志路径。恢复监督器的回执引用本地日志，而共享副本保留原JSON身份；候选通过显式、带SHA/大小/mtime的清单建立路径映射，读取共享副本，不改写原回执。拒绝未声明引用、SHA不符、重复映射、路径穿越及父目录符号链接；输出为普通副本，重复构建不改变既有文件mtime。

8项真实文件系统回归在旧入口全部失败，v1/v2候选均通过。v1首次真实构建因原79项清单遗漏`audit_preflight_actual.log`而退出1：原预检退出0回执及日志SHA一致，构建器按约束拒绝继续。原清单与失败候选源码/回执/日志保留，新清单以原79项为完整前缀，增加该日志和3项失败证据，共83项。没有重写旧记录。

证据均位于`$OUT/bootstrap/`，其中`$OUT=$OUT`：

|步骤|真实退出|回执身份|
|---|---:|---|
|旧入口8项回归|1|`fec5bd2a91899b19c53cfd85cd800fa0f99178f98213b8678a39c0266da9a6b1`|
|v1候选8项回归|0|`02894bbab10f3e12c574aba57759f2b83454e8af4acb4b9be55742c2ebb82179`|
|v1真实构建|1|`e214bcdfbc10f178a53cff7649e0644d17e55173b88b02e9f7d9f36b995304bf`|
|补充清单准备|0|`8a303b4ee4616fb1b1e2530040e683e75485e08dfb058965c7be50bbfa413cff`|
|v2候选8项回归|0|`a4a8fd0fd04c5d6ab6405284c84a37ce6bc2be33f13d08bf08a16a65497f5eba`|
|v2真实构建|0|`12ad8049e49d0ceb04e79d76569addabda656ab8c874ccbcf212c85c1d7a7b9b`|
|真实文件主审|0|`b003ae157b1ec10e36cbc4653cd0c72813508db21e919c46f301d64de574e7f4`|

最终候选`execution_provenance_recovery_candidate_v2.json`身份`470e2113258f0fdea70ec3b7a98eaf44f4f3cce5101f3d28e869111ccc6f8d26`，258个显式输入去重为231文件：67份Python源码、103份密封JSON、61份日志。83项恢复输入及44项本地原路径映射逐一验证；14份引用本地日志的终态回执原字节/身份保留，包括真实失败退出。231个真实副本另在临时输出根交给发布v8读取器，全部接受，未替换当前发布清单。

主审`review/G4_execution_provenance_recovery_candidate_primary_review.json`身份`6b1baf2f6fe2990bdf605dea3241493400c59246cd6d4597ac05ae73bb2672d9`。当前发布清单、公开47索引、旧构建器及旧恢复清单SHA/mtime保持不变；93份冻结文件及2120份稳定文件同样通过。原控制器和原后处理仍保持原有暂停状态。

候选沿用旧82场景报告与class80来源范围，新增已完成的恢复证据及scene112主审支持；class90/100/110/120后续检查点、scene113及以后的主审、5个恢复角色终态文件、原5阶段后处理和最终17章审核，仍须在最终显式清单中追加。当前`publication/execution_provenance.json`保持原134文件范围；候选未发布，未执行C1/C2。

主要新增构建器`bootstrap/build_execution_provenance82_recovery_v2.py`，测试`bootstrap/test_execution_provenance_recovery_v2.py`。可重复运行以下命令核验文件系统行为；它只使用临时测试目录：

```bash
cd $REPO
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src:. $LOCAL_ASSET/python $OUT/bootstrap/test_execution_provenance_recovery_v2.py $OUT/bootstrap/build_execution_provenance82_recovery_v2.py
```

已记录真实退出的prepare/build/primary监督入口为一次性执行，不重复启动；核对其现有回执和日志。
