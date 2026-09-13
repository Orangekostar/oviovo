# Replica8 完整查询配对结果

8 场景 × 200 帧（0,10,…,1990），12 个独立条件。各场景 native、评测及8/7场景聚合均 COMPLETE，聚合 CLI exit 0。

内部条件名 B0 对应本次重建原生 **B1**；历史 B0 Room0 结果不混入主表。
以下数值为百分数。语义指标先累计顶点混淆矩阵，实例 AP 使用发布版 evaluator 一次跨场景计算；不是逐场景 AP 的平均值。

| 条件 | mIoU | mAcc | semantic AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| B0 | 27.25568 | 32.65928 | 8.56337 | 21.47686 | 34.58538 |
| S1a | 27.26642 | 32.70895 | 8.57567 | 21.47353 | 34.61914 |
| S1b | 27.29476 | 34.60619 | 8.64713 | 21.58096 | 34.71511 |
| S1c | 26.59465 | 34.49284 | 8.67263 | 21.61118 | 34.81237 |
| C1 | 27.26642 | 32.70895 | 8.57567 | 21.47353 | 34.61914 |
| RANDOM8 | 25.94855 | 31.44048 | 8.26837 | 19.25018 | 32.41352 |
| QUALITY8 | 26.03858 | 32.38609 | 8.64713 | 21.58096 | 34.37367 |
| ALL_VIEWS | 26.10011 | 31.44048 | 8.33189 | 19.39353 | 32.55687 |
| S1a_FULL | 27.26500 | 32.70895 | 8.57567 | 21.47353 | 34.61914 |
| RANDOM8_FULL | 24.54378 | 30.42742 | 8.10707 | 18.47920 | 31.64559 |
| QUALITY8_FULL | 26.00553 | 32.38609 | 8.64765 | 21.58333 | 34.30340 |
| ALL_VIEWS_FULL | 26.30217 | 31.59722 | 8.32705 | 19.46431 | 32.66562 |

C1 保持扩展场景前冻结的 S1a（retained-top10 池、K=8、原面积融合）。
S1b/S1c 和 `_FULL` 仅为消融；没有根据本表选择新组合。ALL_VIEWS 的读出预算不同。

C1 相对本次原生 B1：mIoU +0.01074 个百分点，AP +0.01230 个百分点；AP50 略降。
这不支持稳定、显著或全面超越的主张。完整池 S1a_FULL 没有额外 AP 收益。
历史 Room0 的较大改进没有在完整主表重现，不能把历史开发提升外推到所有场景。

原始机器可读结果：共享存储 `aggregate_full_replica8/replica8.json`。
逐场景原始评测和来源见仓库 `artifacts/static_ovmap/replica8_full_history/<scene>/`。
canonical 几何只报告独立诊断口径；论文类别无关 AP 来源未核实，保持 null。

## 额外7场景诊断

Room0 已用于开发；本表不声明新的独立测试集。

| 条件 | mIoU (%) | mAcc (%) | AP (%) | AP50 (%) | AP25 (%) |
|---|---:|---:|---:|---:|---:|
| B0 | 28.10870 | 33.37516 | 8.06338 | 20.06032 | 34.99509 |
| S1a | 28.06904 | 33.35767 | 8.05867 | 20.03558 | 34.99509 |
| S1b | 28.17120 | 35.52765 | 8.22936 | 20.26377 | 35.19853 |
| S1c | 27.37141 | 35.31385 | 8.22936 | 20.26377 | 35.25892 |
| C1 | 28.06904 | 33.35767 | 8.05867 | 20.03558 | 34.99509 |
| RANDOM8 | 26.90501 | 32.28673 | 7.73885 | 17.70693 | 32.70208 |
| QUALITY8 | 26.80228 | 33.15064 | 8.22936 | 20.26377 | 34.84225 |
| ALL_VIEWS | 27.07081 | 32.28673 | 7.82183 | 17.88641 | 32.88156 |
| S1a_FULL | 28.06707 | 33.35767 | 8.05867 | 20.03558 | 34.99509 |
| RANDOM8_FULL | 25.14780 | 30.96816 | 7.54275 | 17.05774 | 32.06497 |
| QUALITY8_FULL | 26.76367 | 33.15064 | 8.22991 | 20.26624 | 34.76892 |
| ALL_VIEWS_FULL | 27.33696 | 32.47845 | 7.81587 | 18.02445 | 33.06852 |

C1 在额外7场景的 mIoU 与 AP 均略降；不支持稳定泛化收益。

## 逐场景配对

| 场景 | B1 mIoU (%) | C1 mIoU (%) | B1 AP (%) | C1 AP (%) | S1a_FULL AP (%) |
|---|---:|---:|---:|---:|---:|
| office0 | 17.01785 | 17.01785 | 9.09722 | 9.09722 | 9.09722 |
| office1 | 14.30350 | 14.30350 | 8.33333 | 8.33333 | 8.33333 |
| office2 | 35.08245 | 35.08244 | 9.23203 | 9.23203 | 9.23203 |
| office3 | 28.78204 | 28.48648 | 6.64251 | 6.54589 | 6.54589 |
| office4 | 38.70085 | 38.87503 | 16.66667 | 16.66667 | 16.66667 |
| room0 | 33.35612 | 33.57635 | 15.36137 | 15.57999 | 15.57999 |
| room1 | 28.89066 | 28.89066 | 9.13395 | 9.13395 | 9.13395 |
| room2 | 34.60077 | 34.60077 | 17.85301 | 17.85301 | 17.85301 |

全部12条件的逐场景主指标、发布版 mP/mR 与全原生几何诊断见 `aggregate/per_scene.json`。

## 成本（秒）

| 场景 | 前端原记录 | 几何原记录 | 新 mapping+VLM+export | 本次 native 增量合计 | 前端/几何复用 |
|---|---:|---:|---:|---:|---|
| office0 | 359.494 | 1174.017 | 1397.677 | 1397.677 | True |
| office1 | 340.252 | 1182.022 | 1265.379 | 1265.379 | True |
| office2 | 346.552 | 1149.746 | 2011.856 | 3508.154 | False |
| office3 | 343.380 | 1167.727 | 2105.347 | 3616.454 | False |
| office4 | 344.353 | 1174.782 | 1380.537 | 2899.671 | False |
| room0 | 339.051 | 1175.390 | 1498.304 | 1498.304 | True |
| room1 | 347.445 | 1175.260 | 1344.515 | 2867.220 | False |
| room2 | 341.833 | 1186.131 | 1418.497 | 2946.462 | False |

这是各阶段记录之和，非隔离端到端测速；不包含初始失败尝试或所有历史构建费用。
单独 VLM、导出耗时及 peak GPU 为 null，原因是原生 runner 未分项测量。
并发共享机器不能推导可比 E2E FPS。逐阶段后处理/评测成本见 `aggregate/costs.json`。

## 选择诊断

| 场景 | C1 选择集合变化 | C1 类别变化 | FULL 选择集合变化 | FULL 类别变化 |
|---|---:|---:|---:|---:|
| office0 | 11 | 0 | 15 | 0 |
| office1 | 6 | 0 | 7 | 0 |
| office2 | 8 | 1 | 13 | 0 |
| office3 | 18 | 2 | 21 | 4 |
| office4 | 12 | 1 | 13 | 1 |
| room0 | 7 | 1 | 8 | 1 |
| room1 | 7 | 0 | 9 | 0 |
| room2 | 12 | 0 | 12 | 0 |

所选 query ID、前后类别、margin、读出时间与预算见 `aggregate/selection_diagnostics.json`。
完整查询共4255条、原保留3422条；完整池新增833条已有查询，没有额外图像推理。
但三个场景为补捕获重新运行了 mapping，重跑成本已单列。

## 已执行聚合命令

在工作树运行，使用原发布版依赖环境：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/aggregate_static_replica_readouts.py --scene-result office0=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/office0 --scene-result office1=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/office1 --scene-result office2=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/office2 --scene-result office3=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/office3 --scene-result office4=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/office4 --scene-result room0=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/room0 --scene-result room1=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/room1 --scene-result room2=/mnt/shared/ww/ovimap-static-20260913/evaluated_full_replica8/room2 --evaluator-root /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP --output /mnt/shared/ww/ovimap-static-20260913/aggregate_full_replica8
```

机器可读主表、7场景诊断、逐场景、成本、选择诊断及原文件 SHA 位于
`artifacts/static_ovmap/replica8_full_history/aggregate/`。
