# CROVE multimethod readout V1 — revised V2 handoff

交付分支：`research/crove-multimethod-readout-v1`，仓库 `Orangekostar/oviovo`。
任务依据：`CODEX_CROVE_MULTIMETHOD_READOUT_V1_REVISED_V2_ALL_IN_ONE.md`。
后端筛选与静态确认已完成；协议偏差和代码来源限制保留在验收记录中。

`DEV_SCREENING_STATUS=COMPLETE`；`STATIC_CONFIRMATION_STATUS=COMPLETE`；
`DYNAMIC_CONFIRMATION_STATUS=NOT_RUN_RAW_MISSING`。

已完成 85 项 DEV 全图评分（room0 27 项、Apartment B3/H2 各 29 项），
以及未参与选型的 room1 四类配置与直接对照共 11 项确认评分。
配置在确认前冻结；不根据 room1 得分替换代表或继续调参。
冻结选择使用原有 83 项证据；验收时发现缺少的 TOPK 边界图 B3/H2
两项在确认后补跑，明确标为 POST_FREEZE_SUPPLEMENT，不追溯纳入选择。
两项分别为 0.139538／0.139455，与几何图同分，均未通过地图质量门槛。
静态跨场景确认为 COMPLETE；Office 外部动态确认因原始资产缺失为
NOT_RUN_RAW_MISSING，不能声称动态跨环境泛化。

| 家族 | DEV 主要证据 | room1 冻结确认 | 结论 |
|---|---|---|---|
| M1 | QUALITY room0 mIoU 0.400847；低于 S2 0.447889 | QUALITY 0.355117，原生 0.399250 | 新语义头增益不稳定；局部背景真实运行但未改善合格读出 |
| M2 | S2 边界图 room0 0.443967，低于 S2 | 边界图 0.446672，S2 0.444857 | 小幅确认增益；边界项本身不优于几何图 0.446689 |
| M3 | 共识 CA-AP50 0.192182，原生 0.470711 | 共识 0.227223，原生 0.209150 | 优于 pairwise；相对原生的收益有明显场景依赖 |
| M4 | 官方学习头 room0 0.396887，匹配均值池化 0.364617 | 学习头 0.345663，均值池化 0.411792 | 训练后权重确实参与全图推理，但 DEV 增益未转移 |

静态语义指标赢家仍是 S2，静态实例 DEV 赢家仍是原生 OVI。
动态最终合格配置为 `MV_QUALITY_GRAPH_BOUNDARY@H2`，current mIoU
0.137699；更高分的 S2 dense replay 系列未通过原 Ghost/地图质量门槛。
这不是新增几何：全 current free-conflict 与源行集合固定，语义条件化
Ghost 的零值必须结合分子/分母读取。

## 实际结果与配置

统一紧凑根目录：`configs/evaluation/results/crove_multimethod_readout_v1/`。

- `all_method_results.json/.csv`、`family_results.md`：全部 DEV 行及直接增量。
- `selected_configs.json`、`selection_source_results.json`：冻结选择与不可变 DEV 证据。
- `room1_confirmation_results.md`、逐方法 JSON、确认 status/invariants：完整静态确认。
- `apartment_readout_geometry_audit.json`、`apartment_per_class_audit.json`：58 项动态几何与逐类检查。
- `apartment_H2_recovered_semantic_attribution.json`：2,939 源行/666 物理样本的恢复归因。
- `*_registry.json`：实际角色、unary、图参数、投票裁决及输入预算。
- `model_manifest.json`：官方来源、代码版本、checkpoint SHA256、严格加载证据。
- `compact_artifact_index.json`：已跟踪紧凑制品的大小与 SHA256；索引不包含自身，以索引内实际条目数为准。

完整讨论与局限见同目录的结果报告；计划文件保留输入和实现约束。
各任务使用独立协议，不把动态 mIoU 与 Replica 静态 mIoU 混合排名。
表中 `feature_coverage_source` 区分原评分记录与完整预测掩码的源行比例；
该比例不是唯一物理体素比例，也不等于 RGB 有效覆盖率。
crop 数、独立视图数和 seed 未记录时保留 null，不从总前向数猜测。
`code_provenance.json` 单独记录代码版本限制：启动时的完整 Git SHA 未写入
原评分记录，因此不以当前交付提交替代。已验证列出的两个末期 runner
和两个公共评分器自 `d995ae0` 后没有改动，并记录其实际文件 SHA256；
该检查不能扩大解释为所有历史 trial 的完整依赖版本证明。

## 已执行的主要命令

在研究 worktree 根目录运行，Python 使用既有环境：

```bash
python scripts/evaluation/run_crove_local_background_readouts.py
python scripts/evaluation/run_crove_graph_apartment.py --unary topk --boundary
python scripts/evaluation/audit_crove_apartment_readout_geometry.py
python scripts/evaluation/audit_crove_recovered_semantics.py
python scripts/evaluation/audit_crove_dynamic_per_class.py
python scripts/evaluation/summarize_crove_multimethod_readouts.py
python scripts/evaluation/select_crove_multimethod_readouts.py
python scripts/evaluation/run_crove_room1_confirmation.py
python scripts/evaluation/plot_crove_room0_semantic_delta.py
python scripts/evaluation/export_crove_selected_apartment_views.py
python scripts/evaluation/audit_crove_selected_map_exports.py
```

选择脚本用于首次冻结；汇总刷新后不要重写已有冻结文件。
大型预测、实际特征与四份全地图保存在
`$HOME/oviovo_baseline_runs/20260912_crove_multimethod_readout_v1/`。
输入不变时复用缓存；不要删除旧实验来重新获取成功状态。

## 上传边界与限制

CODE、紧凑 RESULTS、确认结果和代表地图预览已推送。
代码/结果交付提交为 `f1f1c129aa8941877588bd3eb3e4b79984f6b2b8`：
本地与远端一致，GitHub API 回读总表和该提交交接文件的字节哈希一致。
其后的收尾提交仅更新交付记录与文档，不替代实验启动版本。
MODEL 为 REUSED_EXTERNAL：本轮新增训练更新为 0，官方约 1.67 GB 权重
未再分发，来源与一次校验记录已提供。
MAP 为 LOCAL_ONLY_POLICY：最终 H2 四份 PLY 共约 2.08 GB，每份包含
15,625,540 行与 5,198,109 个三角形；完整源行/标签/连接审计通过。
800,000 行的固定视角预览仅用于展示，不能代替全图数值。

局部成败图覆盖全部 2,939 个恢复源行：112 行改善、589 行保持正确、
506 行剩余错误、1,732 行无邻近 GT 支持，未按表现筛选 ROI。
源 CSV、预测哈希、SVG/PDF/PNG 及字体/碰撞 QA 一并交付。

总表字段审查、任务书逐项核对和制品索引检查已完成，证据及范围见
`crove_multimethod_readout_v1_acceptance.md`。
启动时 Git SHA 未记录的限制见代码来源记录；不能用交付 SHA 代替。
TOPK 边界对照晚于确认补跑的协议偏差不影响已保存的选型或确认数值，
但不能称为完全按预定时序完成。新预算优先验证 M3 的场景依赖和 M2
几何图；不保留全部组合，也不据本轮结果启动 M4 新训练。
