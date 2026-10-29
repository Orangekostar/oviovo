# 双卡资源记录修复候选（未应用）

2026-10-24：真实资源账本仍只关联初始单卡硬件快照，而后续场景已经使用GPU0/GPU2。隔离候选保留原快照身份和字节，另以`runtime_allocation`关联封存的双卡入口；校验binding、protocol、model和调度源码身份。该字段表示配置上限，不是同时利用率、独占模块耗时或在线FPS实测。

候选只涉及`src/static_ovmap/benchmark_completion_v2/resources.py`及新增`tests/static_ovmap/test_benchmark_completion_resources.py`，位于`$OUT/bootstrap/resource_runtime_candidate_v1/`；补丁`resource_runtime.patch`的SHA256为`65cdf68120210b38eac4585ed56fc81c4297ca56b69b61d97d3bda1a6f60d4be`。当前生产源码、资源账本和原测试均未改写。

先在原实现上得到3项测试、6处预期断言失败，真实退出1；修复候选3项测试全部通过，真实退出0。覆盖双卡入口关联、旧硬件保持、单卡回退、4种外来身份拒绝，以及重复构建的字节和mtime保持。这些是临时账本回归测试，不计入科学场景或模型失败。16条原构建语句及9个原账本数值字段AST保持一致；计数、顺序耗时和规划区间的计算未改变。

主代理另外只读核对真实入口得到GPU[0,2]、场景worker上限2、科学CPU评分上限3、BLAS1；原硬件快照仍为GPU worker上限1，SHA/mtime保持。隔离临时副本的`git apply --check`、`git apply`均退出0，补丁后的两个文件与候选逐字节一致。1960份稳定文件及93份冻结文件SHA/mtime保持，新模型调用和训练均为0。

证据：

- 旧实现回归：`bootstrap/resource_runtime_candidate_v1/red_actual_exit.json`，`712703f16b20ea9cf66d169a917f0f7cc652f514b89d9dfab812cf784c26de6a`。
- 候选回归：`bootstrap/resource_runtime_candidate_v1/green_actual_exit.json`，`c9bd520747aec7af4974a95e2d619e29f1df665e6d4768561192b2dd64e1deac`。
- 主审：`review/G4_resource_runtime_candidate_primary_review.json`，`daf9c3d4ff110d780a1d05f8b6a527628aa2109d8cffe89746ac79ea3d1f4684`。
- 主审真实退出0：`bootstrap/resource_runtime_candidate_v1/primary_actual_exit.json`，`683118a14ad6a75ce69c26c0e0019fdb418eaec38c0ad2593a38a725af2594ac`。

原队列、双卡队列及原后处理全部终止前保持候选隔离；安全边界后与发布v8候选一起重新审查，再应用和重建最终资源账本。最终有限来源清单须纳入该候选源码、测试、日志及实际退出证据；目前未执行C1/C2，不升级完整任务完成标志。

修复前源码已有原字节副本`bootstrap/publication_delivery_candidate_v6/producer_boundary/modules/resources.py`，SHA256为`d5f059ba57ac3a6c881cef8e829dac09719422ccb0673893dbcb18e189b99877`，与当前原实现的字节及mtime均一致；本轮未复制或改写它。引用证据`bootstrap/resource_runtime_candidate_v1/original_source_reference.json`（`0a6e158a308445ab880ace1589f7570b0119a52be2fb29788a2aec3f16e0f616`）关联原RED回执。最终有限执行来源清单须显式纳入此原源码和引用证据，保留正式应用后对旧测试实现的追溯。

2026-10-24终态集成顺序复核：`lifecycle.producer_identity()`绑定本namespace全部Python模块；资源修复及发布v8都会改变该整体身份。`publication.publish()`要求最终主审绑定当前producer及result_store；`publication.finish()`在C1/C2完成后才封存当前producer的terminal，再以实际CLI运行`--resume`并核对退出码、科学收据SHA/mtime及零worker/科学导入/写入标记。当前terminal、transaction、publication/final与primary_final均不存在，未执行发布后实测。

因此，所有原源码使用者及原后处理结束后，先复核并应用两个候选补丁，再重建最终资源账本/表格/报告和有限执行来源清单，再做绑定新producer与最终result_store的逐项主审，随后执行C1/C2及实际终态恢复。已有校准/模型组件源码复核仍需检查其具体来源未变；它们不代替应用补丁后的整体身份主审。原`postprocess.PHASES`仅为benchmark_score/statistics/baselines/tables/review，明确不尝试publication；其阶段调用完成与科学覆盖/七个完成标志分别验收。

2026-10-24资源让行：GPU0与外部训练显存冲突时，仅暂停本项目scene0329_02无子进程的worker，GPU2继续运行；双场景是并发上限，不能据此宣称任意时刻两卡均忙。自动恢复条件及真实进程记录见[MULTI_GPU.md](MULTI_GPU.md)。该worker整体墙钟包含等待，后续资源报告必须保留该区别；已完成阶段的独立作业收据不改写。

2026-10-24 04:34:06 UTC：GPU0临时资源等待门禁已真实退出0并恢复原worker，固定支持清单与主审见MULTI_GPU.md。记录等待375.19秒仅为入口后下界，包含暂停的整worker耗时仍不可当独占模块延迟；独立科学作业原收据保持。

2026-10-24计时口径核对：scene0329_02的bundle构造收据`0875da38e042f65c79936dc4822c97bdb9edb6c3a2b1ea26abed974ef6932673`记录`elapsed_seconds=666.129222546937`，计时跨过资源暂停；门禁入口后已记录的等待下界为375.1911528110504秒，完整暂停时长未精确计量，不能据此相减生成“纯计算”结果。`resources.build`原样将bundle阶段墙钟写入历史字段`bundle_CPU_seconds`，再与四阶段作业墙钟求和形成规划用`sequential_pipeline_seconds`；该字段不是CPU活跃秒，亦不是并行吞吐或独占模块延迟。原始收据与计时值保持不变，最终资源/成本/交接审核须保留这一具体注释，不能用该值宣称算子性能。
