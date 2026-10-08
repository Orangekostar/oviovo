# 主代理最终提示词核对

原包：`OVIMAP_SAMV_LOCAL_PROBE_CODEX_2026-10-08.zip`，SHA256 `a3d1d3c9e60b5bd48adeaf402cef8aa433ce7fe1514e43081a49e1c6c4241727`。本核对覆盖执行指令0–11、实现契约A–I及评测规则1–6；科学和交付核验通过。发布的最终门槛在提交后独立实测，证据为共享盘 `attempt_001/publication/final.json`，避免提交SHA自引用。

机器核验见 [FINAL_REQUIREMENT_REVIEW.json](FINAL_REQUIREMENT_REVIEW.json)，以下是主代理对实现语义、实际运行和主张边界的核对，不以测试数量代替实验结果。

| 原包条款 | 实现与实际证据 | 核对结论 |
| --- | --- | --- |
| 指令0–1：任务、基座、范围、存储 | source_binding、initial_state、父实验 publication；四个场景，基座 a95c24d990cea57b14bb537e9acca95b95659bda；独立研究 worktree，持久扩展盘 | 全部小实验完成；未训练、未扩大场景、未删除旧数据。负结果状态为 COMPLETE_NO_PILOT_GAIN。 |
| 指令2–3：先读源码、窄实现、原包原样复制、完整CLI | 下面源码映射；spec_copy_verification；22个冻结源/配置/测试文件；runner.PHASES/main | 15份作者文件及JSON逐字一致；13种phase含all和规定参数均实现。未调用作者GT采样比较CLI或mock_predict。 |
| 指令4、契约A–B：实际G1/D2、完整表面、观察和资源 | binding.bind/load_scene；observations receipts；model_assets/environment_setup；实际worker model_load | 四场景D2重建误差为0；128个观察有完整身份复用证明、0新射线；SAM/VGGT严格完整加载594/1797张量，stage2精确142张量；SAM2.1 Hiera-L及明确postprocessing profile。 |
| 指令5、契约C–E：GT隔离、查询、窗口、域和冻结 | query_plan完整inventory/queries/domain；geometry；pilots；freeze_initial/freeze | 32目标为26 incumbent/6 recovered，不按模型成功替换；同一6帧、JPEG95、正点和固定语义帧对；原始图像坐标→1024方形；真实两窗pilot后冻结、无OOM切换，原始科学冻结发生在新AP前。 |
| 指令6、契约D/I：真实分割和缓存 | 64个segmentation叶子；samv_worker/sam2_worker/worker_io；实际方向覆盖和decoded RGB digest | 各模型32查询完整运行；SAM-V unchanged joint forward、先拆panorama再resize；SAM2同anchor正反向独立状态；空输出/锚点失败均保留真实mask；模型异常未替成空mask。 |
| 指令7、契约F–G：共同lifting、原始与准入支持、同时仲裁 | lifting.count_view_votes/arbitrate；八个lifted receipt；edits NPZ和full diagnostics | 一视图一票，n≥2，2/3加入、1/3删除，整数比值仲裁、同票保留旧owner；非域/保护核不变、正点owner保留、无新ID。未把mask裁到旧owner、未用VGGT预测深度；raw-zero在合法域可改并逐项记录。 |
| 指令7、契约H：同帧OLD/NEW FC与输出 | readout.acquire/paired_labels、fc_worker.run、outputs.build/predict；decisions/predictions | 双源双视图共成功域；原FC FP32 signed_mask/v2 area fallback，完整词表等权float64余弦和.01 margin；15新图、31父dense命中、121新pool。SV05严格SV02 owners×SV04 class-map；真实native_ranks和单一独占分区供AP/mIoU。 |
| 指令8、评测1–3：锁定、baseline、完整地图/两场景池 | 全四预测先锁定；no_evidence_builder_check；baseline_parity；scoring/per_class/traces | 四张真实无编辑构造严格G1一致；每队列一张新G1官方view/confusion/五指标/trace parity通过。28记录/14固定顺序pool，18新scorer调用；内容相同别名有身份依据，无旧Replica8/CF18总池或AP均值填表；APall .50:.05:.90、min100。 |
| 指令8、评测4：机制诊断 | 四场景完整gzip诊断、released trace index、table2及semantic supplement | raw/admitted/final和物理面积、保护抑制、n<2、冲突、donor、消失、固定原支持最大交集GT及tie规则、全地图严格IoU>.50/.75一对一最大匹配、WR/RW和rank/trace重复项均保留。2D GT指标未测并明确null；按原indexed mesh诊断、不暗中weld拓扑。 |
| 指令8、评测5：原定晋升规则和停机 | selection.select及完整未舍入fraction；best_simple_control/claims | 三候选均未通过双队列五指标不降要求；uniform研究选择G1，简单对照另报SAM2 geometry。SEGMENTOR_SUPPORT_SIGNAL和SEMANTIC_MASK_SIGNAL均false；未声称协同或新分割器；部署N0，未全量扩展。 |
| 指令9、评测6：实际成本及唯一额外计时块 | costs、window_timings和所有8个原始计时receipt | 同A40、相同锁定6JPEG/正点、BF16、模型驻留、feature/result cache off；CUDA同步/reset peak，读图→原尺寸mask，2轮反序；8/8输出原尺寸和canonical mask精确一致。加载、mapping/lifting/FC/评分/导出在计时外；OS页缓存未控。新推理如实计数，2次导入失败/加载/未知null分开；独立完整地图延迟未测、不算在线30FPS。 |
| 指令10：窄测试/真实pilot | tests_red、tests_before_freeze、focused_tests、pilot receipts、两次baseline preflight | 12个边界测试通过、变更模块编译通过；未做老仓库全量suite或26场景重放。模型导入/CUDA库路径两个实际问题已修复，并保留失败耗时。 |
| 指令11：公开交付、真实终态和发布门槛 | 四份报告；三表族MD/CSV/JSON/TeX；64紧凑mask、8变动owner-row数组、四场景全目标图、增益/损失对；visual_qa及表格PDF | 全部图表来自canonical store；紧凑文件约18MiB，无RGB/depth、稠密地图、权重；主代理已看渲染图。两次完整指定all --resume真实exit0。最终正常push及完整本地/远端SHA由外部publication/final.json核验。 |

## 实际阅读来源到实现函数的映射

| 原包来源 | 实际实现 |
| --- | --- |
| S03–05，作者README/requirements/checkpoint.load_partial_checkpoint | binding.assets、samv_worker.load_model，独立env_setup/receipt；严格bases及作者partial-loader校验，未改作者模型 |
| S06–09，compare_baseline_sam2的infer_samvggt/infer_sam2、SamVGGT.forward、坐标转换 | samv_worker.infer、sam2_worker.infer、worker_io.restore_tiles；只调用真实推理原语，不运行GT prompts/dataset CLI |
| S10–12，SAM2 setup/build_sam2_video_predictor/propagate_in_video/作者checkpoint | sam2_worker.load_model/infer；torch2.5.1隔离进程、显式Hiera-L、明确overrides、独立正反向state |
| S13，source_preserving_update.binding.load_scene及decisions、父结果与交接 | binding.bind/load_scene；只复用四场景窄输入，不启动父bind/all；父memo重定向到新任务 |
| S14，SourceRowProjector/representative_rows/pose_banks/support_mask | query_plan.observations/load_observation/plan_scene、geometry.choose_window；只读取source_rows/valid而非panoptic标签 |
| S15，旧construct_partition及source_preserving_update.outputs的语义限制 | outputs.build/predict和lifting.arbitrate替换旧whole-unit约束；固定表面上实际行级改动 |
| S16，cvpr_compact.area_fallback.region_vector和region_worker | fc_worker.run、readout.acquire/paired_labels；原FP32 visual-head、signed mask、v2 fallback和相同text |
| S17，minimal_instance_repair.evaluation.partition_evaluator、m2_reviewer_study/源更新scorer | evaluation.partition_registry_key/ordered_rows/preflight_parity/evaluate；实际owners digest注册，native_ranks及完整顺序池 |
| S18–19，父实验结果和范围诊断 | 固定OLD-mask简单对照、独立support/semantic诊断；不挪用历史总体分数填新池、不按GT换目标 |

## 冻结和交付边界

原始科学冻结为 `1d9eca0a5ac600270bdfb812dd1c15aceb0eba95`。在科学全部测完后，`6d3280531150afc0ce0b2c9d87dcb2f55abc6b82` 仅修订costs/reporting/runner：图标题、简单对照和数字溯源、CPU/head计数、CLI无证据构造校验。模型、规划、lifting/readout/output/scorer、配置、资产和窗口未变；没有新增模型调用或AP评分。两条冻结证据均保留，未把事后报告修订伪称为事前科学冻结。

最强结构信号不成立：两队列SAM-V相对SAM2未增加GT50/75且目标best-IoU均值略低。NEW-mask FC在CF有一次wrong→right，但Replica无语义修正、mIoU相对OLD-mask FC微降；不能称均匀语义提升。相对G1，CF的语义改善伴随Replica下降，故没有晋升。

这四个场景此前已曝光，两个场景的小池仅是开发证据。每窗只有两次正式计时，两个环境使用作者各自要求的PyTorch/CUDA版本；结果刻画当前可运行配置，不能推导完整地图更新延迟、统计显著性或在线FPS。原indexed triangle-soup的components不代表焊接后连续表面连通分量。环境导入生成的资产checkout未跟踪bytecode已披露；tracked源码和gitlink未改。
