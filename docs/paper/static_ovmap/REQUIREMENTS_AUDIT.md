# 0913_1：A–I 要求核对

这是逐项证据索引，不是“全部完成”声明。Replica8 最终汇总及交接尚在进行。
任务分支 `research/ovimap-static-benchmark-v1`；基底
`d5c0688bc662f8e65455cb9c62909de87c941b43`。

| 原文要求 | 当前结论 | 证据与边界 |
|---|---|---|
| A：隔离开发、保护已有修改 | COMPLETE | 独立工作树 `/home/ww/crove/ovimap-static`；旧 dirty checkout 未修改。外部两个版本写入 `protocol_manifest.json`；历史二进制身份仍不完整，不能把 B0/B1 差异算作方法收益。 |
| B：实际调用链与复用 | COMPLETE | `IMPLEMENTATION_PLAN.md` 的 Verified source trace；动态 anchor、semantic memory 和 src.v2 主链不作为静态 runner。原发布版投影和评测源文件通过独立 package loader 加载。 |
| C/R0：协议与资产 | COMPLETE（Replica）；BLOCKED（ScanNet 输入） | `protocol_manifest.json`、`baseline_asset_inventory.json`、`frame_input_inventory.json`、`replica8_gt_inventory.json`。ScanNet18 清单保留全部 18 scans / 7 physical scenes；没有输入就不编造帧表。论文类别无关 AP 来源未验证，该列保持 null。 |
| C/R1：原始读出精确对照 | COMPLETE | `room0_enriched/native_readout_audit.json`、`projection_audit.json`、`room0_b1/parity.json`。源顺序 last8、面积融合、min-two；语义类别、mask、score、GT 与原实现配对核验。 |
| C/R2：bank、选择和拆分 | COMPLETE（开发实验） | `S1_RESULTS.md`；last8/random8/quality8/coverage8/all_views 和 S1a/b/c 独立；不同空间拒绝混合。质量来自预测输入；缺失不填满分。属于 STATIC_OFFLINE_READOUT，不声明严格在线。 |
| C：完整历史 | COMPLETE（新运行捕获）；BLOCKED（历史丢弃记录恢复） | `QUERY_HISTORY_CAPTURE.md`；新 8 场景原生运行捕获实际已执行查询。保留原 top10 对照和完整池独立条件。旧 Room0 丢弃记录的 owner/area 未恢复；新运行不是旧向量恢复，也不是零成本补录。 |
| D/R3：单查询 fallback | COMPLETE | `R3_RESULTS.md`；独立多帧几何支持与一次语义查询分开，新增 TP/FP/ignored 有记录；无有效召回增益，C1 关闭。 |
| D/R4：merge / split | COMPLETE | `R4_RESULTS.md`；预测多视图证据、分量级 cannot-link、固定层级规则、原始 partition 和可逆映射、宽整数导出；G1 负收益，G2 混合收益，均关闭。 |
| E/R5：一个稠密分支 | COMPLETE | `R5_RESULTS.md`；GLA-CLIP 分类前特征、ROI 对照、AnyUp 原论文路径、同 owner 稀疏净化；三种 lambda 标签均等于 D2。报告实际存储及缺失观测；没有大规模训练。 |
| E/R6：独立 3D 轨道 | COMPLETE（Room0） | `R6_RESULTS.md`；完整 SpaCeFormer checkpoint、同 RGB-D 预测点云、1152 维独立文本空间、T0/T1；预训练排除与重复运行差异未解决，不声明未见场景泛化或确定性。 |
| E/R7：预算难例复核 | NOT_RUN | 原文可选；Q0/Q1 明确未实现、关闭，没有额外类别条件 mask 查询。 |
| F：独立条件与冻结组合 | COMPLETE（路由与开发实验） | `ovimap_static_conditions_v1.json`、`C1_FREEZE.md`；C1 仅 S1a，未按测试场景 GT 改配置；无收益分支不要求全部跑满。 |
| F：Replica8 / 额外7场景 | NOT_RUN（最终汇总尚未验证） | 8 个完整查询 native 均 exit 0；评测正在完成。最终须调用跨场景 released AP，另列 pooled semantic confusion；额外7场景不叫独立新测试集。 |
| F：耗时与资源 | COMPLETE（可测部分）；BLOCKED（历史分项） | R5/R6 实测成本及各 native stage 日志；前端复用与新增 mapping 分开。未独立测量的 VLM、导出、显存列 null；共享机器场景并行结果不作隔离硬件端到端 FPS 比较。 |
| G：相关测试和真实试跑 | COMPLETE（已执行单元） | 各阶段报告列出相关测试、真实 Room0 和 office0 pipeline；包括语义空间、选择、互斥传递、小图 owner 边界、投影/格式回归。无全动态测试或无关扫描。Replica8 新聚合 CLI 仍须真实运行。 |
| H：阶段提交 / 最终交接 | COMPLETE（阶段提交）；NOT_RUN（最终交接） | 代码、小结果和阶段文档已分阶段提交；大 mesh/权重留在外部路径。最终主表、逐场景、成本及 SHA 核验仍需完成。 |
| I：提示包参考测试 | BLOCKED | `/home/ww/crove/docs/0913_1` 实际只有三份 Markdown，没有 `reference/test_static_core.py`。不能用自写测试冒充原包 8 项测试。 |

原始小结果位于仓库 `artifacts/static_ovmap/`；表内相对 JSON 名称均以该目录为根。
阶段测试记录在对应结果文档中，重复运行和失败尝试没有被删去或替换成最好分数。

ScanNet 扩展文件名搜索覆盖 `/mnt/shared/ww/archive` 和
`/mnt/shared/ww/node107-root-relief-20260912`，结果与错误文件均为空。
这只说明所查路径未匹配文件名，不证明所有存储或压缩包内没有数据。
更早的精确场景目录搜索范围及限制见 `scannet18_asset_inventory.json`。
