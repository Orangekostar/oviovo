# OVIOVO AAAI Benchmark Tables: Current Status Snapshot

- Generated at: `2026-07-20T07:20:12Z` (UTC)
- Source commit: `959d57cedb79b96d4c8e0daf09a2785a579a9d13`
- Source table: `docs/paper/benchmark_tables_baselines.md`
- Token registry: `docs/paper/benchmark_tokens.tsv`
- Cell status: `65 VERIFIED / 5 N/A / 310 UNFILLED` (380 total)

`UNFILLED` means no paper-eligible value is registered. `N/A` means the metric is not natively supported. OVI-MAP released metrics are listed separately as verified diagnostics because the paper's Table 2 AP contract is not present in the released evaluator.

## OVI-MAP Paper Diagnostics

| Contract | Metric | Paper | Ubuntu 24 native rerun | Status |
| --- | --- | ---: | ---: | --- |
| Table 3 semantic | mIoU | 0.265 | 0.2727 | VERIFIED_DIAGNOSTIC |
| Table 3 semantic | mAcc | 0.322 | 0.3268 | VERIFIED_DIAGNOSTIC |
| Table 3 semantic-instance | AP25 | 0.345 | 0.345854 | VERIFIED_DIAGNOSTIC |
| Table 3 semantic-instance | AP50 | 0.212 | 0.214769 | VERIFIED_DIAGNOSTIC |
| Table 3 semantic-instance | APall | 0.085 | 0.085127 | VERIFIED_DIAGNOSTIC |
| Table 2 instance | mIoU | 0.363 | 0.360613 | VERIFIED_DIAGNOSTIC |
| Table 2 instance | AP25 vs released mP@25 | 0.767 | 0.767162 | BLOCKED_NAME_MISMATCH |
| Table 2 instance | AP50 vs released mP@50 | 0.508 | 0.514950 | BLOCKED_NAME_MISMATCH |
| Table 2 instance | AP75 vs released mP@75 | 0.220 | 0.216363 | BLOCKED_NAME_MISMATCH |

Native mapping status: `COMPLETE_NATIVE_MAPPING`, 8 scenes x 200 frames. Released evaluation status: `COMPLETE_RELEASED_EVALUATION`, 18/18 commands PASS. Paper audit has exactly one failure: missing `class_agnostic_ap_manifest`.

## Table 1: Static Mapping Quality

### Semantic quality

| Method | Mode | Replica-8 mIoU | Replica-8 mAcc | Replica-8 f-mIoU | Replica-7 mIoU | ScanNet200-5 mIoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenFusion | native | 0.271 | 0.345 | 0.479 | 0.270 | UNFILLED |
| OVI-MAP | native | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| ConceptGraphs | native | 0.118 | 0.212 | 0.106 | 0.118 | UNFILLED |
| DualMap | native | 0.156 | 0.204 | 0.129 | 0.155 | UNFILLED |
| OVIV2 | online | 0.173 | 0.223 | 0.441 | 0.178 | UNFILLED |

### Instance and geometry quality

| Method | Mode | Replica-8 AP25 | Replica-8 AP50 | Replica-8 F@5cm | Replica-7 AP50 | ScanNet200-5 AP25 | ScanNet200-5 AP50 | ScanNet200-5 F@5cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFusion | native | N/A | N/A | 0.895 | N/A | N/A | N/A | UNFILLED |
| OVI-MAP | native | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| ConceptGraphs | native | 0.401 | 0.238 | 0.897 | 0.220 | UNFILLED | UNFILLED | UNFILLED |
| DualMap | native | 0.384 | 0.164 | 0.887 | 0.147 | UNFILLED | UNFILLED | UNFILLED |
| OVIV2 | online | 0.119 | 0.025 | 0.882 | 0.025 | UNFILLED | UNFILLED | UNFILLED |

Table 1 registry status: `29 VERIFIED / 5 N/A / 26 UNFILLED`. ScanNet200-5 remains unavailable; OVI-MAP Replica tokens additionally wait for AP-contract resolution.

## Table 2: Dynamic Current-Map Quality

### Official TESSE-CD metrics

| Method | Mode | Apartment object F1 | Apartment dynamic F1 | Apartment change F1 | Office object F1 | Office dynamic F1 | Office change F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVI-MAP (frozen) | frozen | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| ConceptGraphs (frozen) | frozen | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| DualMap | native | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Panoptic Mapping + shared masks | composed | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Khronos (open-set) | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Khronos (GT semantics) | oracle | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| OVIOVO | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

### Common current-map metrics

| Method | Mode | Current mIoU | Ghost rate | Background F@5cm | Recovery frames |
| --- | --- | ---: | ---: | ---: | ---: |
| OVI-MAP (frozen) | frozen | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| ConceptGraphs (frozen) | frozen | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| DualMap | native | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Panoptic Mapping + shared masks | composed | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Khronos (open-set) | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Khronos (GT semantics) | oracle | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| OVIOVO | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

Table 2 registry status: `70 UNFILLED`. Official TESSE-CD Apartment and Office bags remain unavailable.

## Table 3: Causal Component Ablation

| Method | Mode | Static mIoU | Change F1 | Stale FP | Ghost rate | Background F@5cm | ID switches | Reactivation R@1 | NOT_FOUND F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Positive-only base | geometry-first | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| + Signed visibility | visibility | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| + Reversible ownership | ownership | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| + Background reclaim | reclaim | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| + Dormant re-ID | re-identification | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| + Calibrated NOT_FOUND | calibration | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

Table 3 registry status: `48 UNFILLED`.

## Table 4: Online Efficiency and Memory

### Latency

| Method | Mode | Frontend s/frame | Backend s/frame | Maintenance s/frame | Total s/frame | Processed Hz | Finalization s | Query p50 ms | Query p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OVI-MAP | native | 1.63 | 6.23 | 0.00 | 7.86 | 0.13 | 0.00 | 7.70 | 7.74 |
| ConceptGraphs | native | 3.52 | 0.50 | 0.00 | 4.02 | 0.25 | 22.98 | 8.77 | 8.81 |
| DualMap | native | 1.66 | 0.07 | 0.00 | 1.73 | 0.58 | 12.52 | 6.37 | 6.45 |
| Khronos | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| OVIOVO (maintenance off) | maintenance-off | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| OVIOVO | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

### Resources

| Method | Mode | Peak GPU GB | Peak RAM GB | Final map MB | Evaluation/I/O s |
| --- | --- | ---: | ---: | ---: | ---: |
| OVI-MAP | native | 26.14 | 20.71 | 699.43 | 194.89 |
| ConceptGraphs | native | 24.01 | 9.09 | 5.80 | 32.01 |
| DualMap | native | 23.80 | 6.73 | 17.97 | 37.52 |
| Khronos | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| OVIOVO (maintenance off) | maintenance-off | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| OVIOVO | online | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

Table 4 registry status: `36 VERIFIED / 36 UNFILLED`.

## Table S1: Open-Vocabulary Current-State Localization

| Split | Methods | Metrics per method | Status |
| --- | ---: | ---: | --- |
| Validation | 5 | 7 | 35 UNFILLED |
| Held-out test | 5 | 7 | 35 UNFILLED |

Table S1 registry status: `70 UNFILLED`.

## Table S2: Temporal Identity on 3RScan

| Methods | Metrics per method | Status |
| ---: | ---: | --- |
| 5 | 6 | 30 UNFILLED |

Table S2 registry status: `30 UNFILLED`.

## Table S3: Absence Reliability and Calibration

| Methods | Metrics per method | Status |
| ---: | ---: | --- |
| 5 | 6 | 30 UNFILLED |

Table S3 registry status: `30 UNFILLED`.

## Main-Table Completion

| Table | Method rows | VERIFIED | BLOCKED | Primary blocker |
| --- | ---: | ---: | ---: | --- |
| T1 | 5 | 0 | 5 | ScanNet200-5 unavailable; OVI-MAP AP naming unresolved |
| T2 | 7 | 0 | 7 | TESSE-CD official bags unavailable |
| T4 | 6 | 3 | 3 | Khronos input and OVIOVO runs pending |
