# OVIOVO AAAI Benchmark Tables

Numerical cells are provenance tokens. `--` denotes a protocol-level N/A.

## Table 1: Static Mapping Quality

**Caption.** Static open-vocabulary mapping under native predictions. Replica-8 is a compatibility split; Replica-7 and ScanNet200-5 are held out and carry the generalization claim.

**Claim.** Lifecycle maintenance must not trade away the static mapping foundation.

### Semantic quality

| Method | Mode | Replica-8 mIoU $\uparrow$ | Replica-8 mAcc $\uparrow$ | Replica-8 f-mIoU $\uparrow$ | Replica-7 mIoU $\uparrow$ | ScanNet200-5 mIoU $\uparrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenFusion | native | {{T1_OPENFUSION_REPLICA8_MIOU}} | {{T1_OPENFUSION_REPLICA8_MACC}} | {{T1_OPENFUSION_REPLICA8_FMIOU}} | {{T1_OPENFUSION_REPLICA7_MIOU}} | {{T1_OPENFUSION_SCANNET5_MIOU}} |
| OVI-MAP | native | {{T1_OVIMAP_REPLICA8_MIOU}} | {{T1_OVIMAP_REPLICA8_MACC}} | {{T1_OVIMAP_REPLICA8_FMIOU}} | {{T1_OVIMAP_REPLICA7_MIOU}} | {{T1_OVIMAP_SCANNET5_MIOU}} |
| ConceptGraphs | native | 0.070 | 0.101 | 0.053 | 0.070 | {{T1_CONCEPTGRAPHS_SCANNET5_MIOU}} |
| DualMap | native | 0.156 | 0.204 | 0.129 | 0.155 | {{T1_DUALMAP_SCANNET5_MIOU}} |
| OVIOVO | online | {{T1_OVIOVO_REPLICA8_MIOU}} | {{T1_OVIOVO_REPLICA8_MACC}} | {{T1_OVIOVO_REPLICA8_FMIOU}} | {{T1_OVIOVO_REPLICA7_MIOU}} | {{T1_OVIOVO_SCANNET5_MIOU}} |

### Instance and geometry quality

| Method | Mode | Replica-8 AP25 $\uparrow$ | Replica-8 AP50 $\uparrow$ | Replica-8 F@5cm $\uparrow$ | Replica-7 AP50 $\uparrow$ | ScanNet200-5 AP25 $\uparrow$ | ScanNet200-5 AP50 $\uparrow$ | ScanNet200-5 F@5cm $\uparrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFusion | native | -- | -- | {{T1_OPENFUSION_REPLICA8_F5}} | -- | -- | -- | {{T1_OPENFUSION_SCANNET5_F5}} |
| OVI-MAP | native | {{T1_OVIMAP_REPLICA8_AP25}} | {{T1_OVIMAP_REPLICA8_AP50}} | {{T1_OVIMAP_REPLICA8_F5}} | {{T1_OVIMAP_REPLICA7_AP50}} | {{T1_OVIMAP_SCANNET5_AP25}} | {{T1_OVIMAP_SCANNET5_AP50}} | {{T1_OVIMAP_SCANNET5_F5}} |
| ConceptGraphs | native | 0.399 | 0.214 | 0.376 | 0.213 | {{T1_CONCEPTGRAPHS_SCANNET5_AP25}} | {{T1_CONCEPTGRAPHS_SCANNET5_AP50}} | {{T1_CONCEPTGRAPHS_SCANNET5_F5}} |
| DualMap | native | 0.384 | 0.164 | 0.887 | 0.147 | {{T1_DUALMAP_SCANNET5_AP25}} | {{T1_DUALMAP_SCANNET5_AP50}} | {{T1_DUALMAP_SCANNET5_F5}} |
| OVIOVO | online | {{T1_OVIOVO_REPLICA8_AP25}} | {{T1_OVIOVO_REPLICA8_AP50}} | {{T1_OVIOVO_REPLICA8_F5}} | {{T1_OVIOVO_REPLICA7_AP50}} | {{T1_OVIOVO_SCANNET5_AP25}} | {{T1_OVIOVO_SCANNET5_AP50}} | {{T1_OVIOVO_SCANNET5_F5}} |

## Table 2: Dynamic Current-Map Quality

**Caption.** Current-map quality on TESSE-CD. Frozen rows cannot update after intervention; Panoptic Mapping is composed with shared masks; Khronos GT semantics is an oracle and is excluded from ranking.

**Claim.** Signed visibility and reversible maintenance remove stale geometry and recover revealed space.

### Official TESSE-CD metrics

| Method | Mode | Apartment object F1 $\uparrow$ | Apartment dynamic F1 $\uparrow$ | Apartment change F1 $\uparrow$ | Office object F1 $\uparrow$ | Office dynamic F1 $\uparrow$ | Office change F1 $\uparrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVI-MAP (frozen) | frozen | {{T2_OVIMAP_FROZEN_APARTMENT_OBJECT_F1}} | {{T2_OVIMAP_FROZEN_APARTMENT_DYNAMIC_F1}} | {{T2_OVIMAP_FROZEN_APARTMENT_CHANGE_F1}} | {{T2_OVIMAP_FROZEN_OFFICE_OBJECT_F1}} | {{T2_OVIMAP_FROZEN_OFFICE_DYNAMIC_F1}} | {{T2_OVIMAP_FROZEN_OFFICE_CHANGE_F1}} |
| ConceptGraphs (frozen) | frozen | {{T2_CONCEPTGRAPHS_FROZEN_APARTMENT_OBJECT_F1}} | {{T2_CONCEPTGRAPHS_FROZEN_APARTMENT_DYNAMIC_F1}} | {{T2_CONCEPTGRAPHS_FROZEN_APARTMENT_CHANGE_F1}} | {{T2_CONCEPTGRAPHS_FROZEN_OFFICE_OBJECT_F1}} | {{T2_CONCEPTGRAPHS_FROZEN_OFFICE_DYNAMIC_F1}} | {{T2_CONCEPTGRAPHS_FROZEN_OFFICE_CHANGE_F1}} |
| DualMap | native | {{T2_DUALMAP_APARTMENT_OBJECT_F1}} | {{T2_DUALMAP_APARTMENT_DYNAMIC_F1}} | {{T2_DUALMAP_APARTMENT_CHANGE_F1}} | {{T2_DUALMAP_OFFICE_OBJECT_F1}} | {{T2_DUALMAP_OFFICE_DYNAMIC_F1}} | {{T2_DUALMAP_OFFICE_CHANGE_F1}} |
| Panoptic Mapping + shared masks | composed | {{T2_PANOPTIC_SHARED_APARTMENT_OBJECT_F1}} | {{T2_PANOPTIC_SHARED_APARTMENT_DYNAMIC_F1}} | {{T2_PANOPTIC_SHARED_APARTMENT_CHANGE_F1}} | {{T2_PANOPTIC_SHARED_OFFICE_OBJECT_F1}} | {{T2_PANOPTIC_SHARED_OFFICE_DYNAMIC_F1}} | {{T2_PANOPTIC_SHARED_OFFICE_CHANGE_F1}} |
| Khronos (open-set) | online | {{T2_KHRONOS_OPEN_APARTMENT_OBJECT_F1}} | {{T2_KHRONOS_OPEN_APARTMENT_DYNAMIC_F1}} | {{T2_KHRONOS_OPEN_APARTMENT_CHANGE_F1}} | {{T2_KHRONOS_OPEN_OFFICE_OBJECT_F1}} | {{T2_KHRONOS_OPEN_OFFICE_DYNAMIC_F1}} | {{T2_KHRONOS_OPEN_OFFICE_CHANGE_F1}} |
| Khronos (GT semantics) | oracle | {{T2_KHRONOS_ORACLE_APARTMENT_OBJECT_F1}} | {{T2_KHRONOS_ORACLE_APARTMENT_DYNAMIC_F1}} | {{T2_KHRONOS_ORACLE_APARTMENT_CHANGE_F1}} | {{T2_KHRONOS_ORACLE_OFFICE_OBJECT_F1}} | {{T2_KHRONOS_ORACLE_OFFICE_DYNAMIC_F1}} | {{T2_KHRONOS_ORACLE_OFFICE_CHANGE_F1}} |
| OVIOVO | online | {{T2_OVIOVO_APARTMENT_OBJECT_F1}} | {{T2_OVIOVO_APARTMENT_DYNAMIC_F1}} | {{T2_OVIOVO_APARTMENT_CHANGE_F1}} | {{T2_OVIOVO_OFFICE_OBJECT_F1}} | {{T2_OVIOVO_OFFICE_DYNAMIC_F1}} | {{T2_OVIOVO_OFFICE_CHANGE_F1}} |

### Common current-map metrics

| Method | Mode | Current mIoU $\uparrow$ | Ghost rate $\downarrow$ | Background F@5cm $\uparrow$ | Recovery frames $\downarrow$ |
| --- | --- | ---: | ---: | ---: | ---: |
| OVI-MAP (frozen) | frozen | {{T2_OVIMAP_FROZEN_CURRENT_MIOU}} | {{T2_OVIMAP_FROZEN_GHOST_RATE}} | {{T2_OVIMAP_FROZEN_BG_F5}} | {{T2_OVIMAP_FROZEN_RECOVERY_FRAMES}} |
| ConceptGraphs (frozen) | frozen | {{T2_CONCEPTGRAPHS_FROZEN_CURRENT_MIOU}} | {{T2_CONCEPTGRAPHS_FROZEN_GHOST_RATE}} | {{T2_CONCEPTGRAPHS_FROZEN_BG_F5}} | {{T2_CONCEPTGRAPHS_FROZEN_RECOVERY_FRAMES}} |
| DualMap | native | {{T2_DUALMAP_CURRENT_MIOU}} | {{T2_DUALMAP_GHOST_RATE}} | {{T2_DUALMAP_BG_F5}} | {{T2_DUALMAP_RECOVERY_FRAMES}} |
| Panoptic Mapping + shared masks | composed | {{T2_PANOPTIC_SHARED_CURRENT_MIOU}} | {{T2_PANOPTIC_SHARED_GHOST_RATE}} | {{T2_PANOPTIC_SHARED_BG_F5}} | {{T2_PANOPTIC_SHARED_RECOVERY_FRAMES}} |
| Khronos (open-set) | online | {{T2_KHRONOS_OPEN_CURRENT_MIOU}} | {{T2_KHRONOS_OPEN_GHOST_RATE}} | {{T2_KHRONOS_OPEN_BG_F5}} | {{T2_KHRONOS_OPEN_RECOVERY_FRAMES}} |
| Khronos (GT semantics) | oracle | {{T2_KHRONOS_ORACLE_CURRENT_MIOU}} | {{T2_KHRONOS_ORACLE_GHOST_RATE}} | {{T2_KHRONOS_ORACLE_BG_F5}} | {{T2_KHRONOS_ORACLE_RECOVERY_FRAMES}} |
| OVIOVO | online | {{T2_OVIOVO_CURRENT_MIOU}} | {{T2_OVIOVO_GHOST_RATE}} | {{T2_OVIOVO_BG_F5}} | {{T2_OVIOVO_RECOVERY_FRAMES}} |

## Table 3: Causal Component Ablation

**Caption.** Cumulative ablation across the held-out static, dynamic, identity, and current-query splits. Each row adds exactly one mechanism.

**Claim.** Visibility, ownership, reclaim, re-identification, and rejection target distinct failures.

### Cumulative mechanisms

| Method | Mode | Static mIoU $\uparrow$ | Change F1 $\uparrow$ | Stale FP $\downarrow$ | Ghost rate $\downarrow$ | Background F@5cm $\uparrow$ | ID switches $\downarrow$ | Reactivation R@1 $\uparrow$ | NOT_FOUND F1 $\uparrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Positive-only base | geometry-first | {{T3_BASE_STATIC_MIOU}} | {{T3_BASE_CHANGE_F1}} | {{T3_BASE_STALE_FP}} | {{T3_BASE_GHOST_RATE}} | {{T3_BASE_BG_F5}} | {{T3_BASE_IDSW}} | {{T3_BASE_REACT_R1}} | {{T3_BASE_NOT_FOUND_F1}} |
| + Signed visibility | visibility | {{T3_VIS_STATIC_MIOU}} | {{T3_VIS_CHANGE_F1}} | {{T3_VIS_STALE_FP}} | {{T3_VIS_GHOST_RATE}} | {{T3_VIS_BG_F5}} | {{T3_VIS_IDSW}} | {{T3_VIS_REACT_R1}} | {{T3_VIS_NOT_FOUND_F1}} |
| + Reversible ownership | ownership | {{T3_OWNER_STATIC_MIOU}} | {{T3_OWNER_CHANGE_F1}} | {{T3_OWNER_STALE_FP}} | {{T3_OWNER_GHOST_RATE}} | {{T3_OWNER_BG_F5}} | {{T3_OWNER_IDSW}} | {{T3_OWNER_REACT_R1}} | {{T3_OWNER_NOT_FOUND_F1}} |
| + Background reclaim | reclaim | {{T3_RECLAIM_STATIC_MIOU}} | {{T3_RECLAIM_CHANGE_F1}} | {{T3_RECLAIM_STALE_FP}} | {{T3_RECLAIM_GHOST_RATE}} | {{T3_RECLAIM_BG_F5}} | {{T3_RECLAIM_IDSW}} | {{T3_RECLAIM_REACT_R1}} | {{T3_RECLAIM_NOT_FOUND_F1}} |
| + Dormant re-ID | re-identification | {{T3_REID_STATIC_MIOU}} | {{T3_REID_CHANGE_F1}} | {{T3_REID_STALE_FP}} | {{T3_REID_GHOST_RATE}} | {{T3_REID_BG_F5}} | {{T3_REID_IDSW}} | {{T3_REID_REACT_R1}} | {{T3_REID_NOT_FOUND_F1}} |
| + Calibrated NOT_FOUND | calibration | {{T3_FULL_STATIC_MIOU}} | {{T3_FULL_CHANGE_F1}} | {{T3_FULL_STALE_FP}} | {{T3_FULL_GHOST_RATE}} | {{T3_FULL_BG_F5}} | {{T3_FULL_IDSW}} | {{T3_FULL_REACT_R1}} | {{T3_FULL_NOT_FOUND_F1}} |

## Table 4: Online Efficiency and Memory

**Caption.** Same-hardware measurements with initialization, online processing, finalization, and evaluation I/O reported separately. Paper-reported hardware numbers are not mixed into ranked columns.

**Claim.** Current-state maintenance must have bounded online latency and memory cost.

### Latency

| Method | Mode | Frontend s/frame $\downarrow$ | Backend s/frame $\downarrow$ | Maintenance s/frame $\downarrow$ | Total s/frame $\downarrow$ | Processed Hz $\uparrow$ | Finalization s $\downarrow$ | Query p50 ms $\downarrow$ | Query p95 ms $\downarrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OVI-MAP | native | {{T4_OVIMAP_FRONTEND_SPF}} | {{T4_OVIMAP_BACKEND_SPF}} | {{T4_OVIMAP_MAINT_SPF}} | {{T4_OVIMAP_TOTAL_SPF}} | {{T4_OVIMAP_HZ}} | {{T4_OVIMAP_FINAL_S}} | {{T4_OVIMAP_QUERY_P50_MS}} | {{T4_OVIMAP_QUERY_P95_MS}} |
| ConceptGraphs | native | 3.52 | 0.50 | 0.00 | 4.02 | 0.25 | 22.98 | 8.77 | 8.81 |
| DualMap | native | 1.66 | 0.07 | 0.00 | 1.73 | 0.58 | 12.52 | 6.37 | 6.45 |
| Khronos | online | {{T4_KHRONOS_FRONTEND_SPF}} | {{T4_KHRONOS_BACKEND_SPF}} | {{T4_KHRONOS_MAINT_SPF}} | {{T4_KHRONOS_TOTAL_SPF}} | {{T4_KHRONOS_HZ}} | {{T4_KHRONOS_FINAL_S}} | {{T4_KHRONOS_QUERY_P50_MS}} | {{T4_KHRONOS_QUERY_P95_MS}} |
| OVIOVO (maintenance off) | maintenance-off | {{T4_OVIOVO_STATIC_FRONTEND_SPF}} | {{T4_OVIOVO_STATIC_BACKEND_SPF}} | {{T4_OVIOVO_STATIC_MAINT_SPF}} | {{T4_OVIOVO_STATIC_TOTAL_SPF}} | {{T4_OVIOVO_STATIC_HZ}} | {{T4_OVIOVO_STATIC_FINAL_S}} | {{T4_OVIOVO_STATIC_QUERY_P50_MS}} | {{T4_OVIOVO_STATIC_QUERY_P95_MS}} |
| OVIOVO | online | {{T4_OVIOVO_FRONTEND_SPF}} | {{T4_OVIOVO_BACKEND_SPF}} | {{T4_OVIOVO_MAINT_SPF}} | {{T4_OVIOVO_TOTAL_SPF}} | {{T4_OVIOVO_HZ}} | {{T4_OVIOVO_FINAL_S}} | {{T4_OVIOVO_QUERY_P50_MS}} | {{T4_OVIOVO_QUERY_P95_MS}} |

### Resources

| Method | Mode | Peak GPU GB $\downarrow$ | Peak RAM GB $\downarrow$ | Final map MB $\downarrow$ | Evaluation/I/O s $\downarrow$ |
| --- | --- | ---: | ---: | ---: | ---: |
| OVI-MAP | native | {{T4_OVIMAP_GPU_GB}} | {{T4_OVIMAP_RAM_GB}} | {{T4_OVIMAP_MAP_MB}} | {{T4_OVIMAP_EVAL_IO_S}} |
| ConceptGraphs | native | 24.01 | 9.09 | 5.80 | 32.01 |
| DualMap | native | 23.80 | 6.73 | 17.97 | 37.52 |
| Khronos | online | {{T4_KHRONOS_GPU_GB}} | {{T4_KHRONOS_RAM_GB}} | {{T4_KHRONOS_MAP_MB}} | {{T4_KHRONOS_EVAL_IO_S}} |
| OVIOVO (maintenance off) | maintenance-off | {{T4_OVIOVO_STATIC_GPU_GB}} | {{T4_OVIOVO_STATIC_RAM_GB}} | {{T4_OVIOVO_STATIC_MAP_MB}} | {{T4_OVIOVO_STATIC_EVAL_IO_S}} |
| OVIOVO | online | {{T4_OVIOVO_GPU_GB}} | {{T4_OVIOVO_RAM_GB}} | {{T4_OVIOVO_MAP_MB}} | {{T4_OVIOVO_EVAL_IO_S}} |

## Table S1: Open-Vocabulary Current-State Localization

**Caption.** Validation and held-out current-state query macro averages. Frozen maps and the composed Khronos text head use the same query normalization and cannot exploit manipulation or navigation signals.

**Claim.** Bring me the new book is normalized to action=locate, category=book, temporal_predicate=added_since_previous_visit, scope=current.

### Validation split

| Method | Mode | Present R@1 $\uparrow$ | Moved R@1 $\uparrow$ | New R@1 $\uparrow$ | NOT_FOUND F1 $\uparrow$ | Stale FP $\downarrow$ | Median error m $\downarrow$ | Recovery frames $\downarrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OVI-MAP (frozen) | frozen | {{S1_OVIMAP_FROZEN_VALIDATION_PRESENT_R1}} | {{S1_OVIMAP_FROZEN_VALIDATION_MOVED_R1}} | {{S1_OVIMAP_FROZEN_VALIDATION_NEW_R1}} | {{S1_OVIMAP_FROZEN_VALIDATION_NOT_FOUND_F1}} | {{S1_OVIMAP_FROZEN_VALIDATION_STALE_FP}} | {{S1_OVIMAP_FROZEN_VALIDATION_LOC_ERROR_M}} | {{S1_OVIMAP_FROZEN_VALIDATION_RECOVERY_FRAMES}} |
| ConceptGraphs (frozen) | frozen | {{S1_CONCEPTGRAPHS_FROZEN_VALIDATION_PRESENT_R1}} | {{S1_CONCEPTGRAPHS_FROZEN_VALIDATION_MOVED_R1}} | {{S1_CONCEPTGRAPHS_FROZEN_VALIDATION_NEW_R1}} | {{S1_CONCEPTGRAPHS_FROZEN_VALIDATION_NOT_FOUND_F1}} | {{S1_CONCEPTGRAPHS_FROZEN_VALIDATION_STALE_FP}} | {{S1_CONCEPTGRAPHS_FROZEN_VALIDATION_LOC_ERROR_M}} | {{S1_CONCEPTGRAPHS_FROZEN_VALIDATION_RECOVERY_FRAMES}} |
| DualMap | native | {{S1_DUALMAP_VALIDATION_PRESENT_R1}} | {{S1_DUALMAP_VALIDATION_MOVED_R1}} | {{S1_DUALMAP_VALIDATION_NEW_R1}} | {{S1_DUALMAP_VALIDATION_NOT_FOUND_F1}} | {{S1_DUALMAP_VALIDATION_STALE_FP}} | {{S1_DUALMAP_VALIDATION_LOC_ERROR_M}} | {{S1_DUALMAP_VALIDATION_RECOVERY_FRAMES}} |
| Khronos + shared text head | composed | {{S1_KHRONOS_SHARED_VALIDATION_PRESENT_R1}} | {{S1_KHRONOS_SHARED_VALIDATION_MOVED_R1}} | {{S1_KHRONOS_SHARED_VALIDATION_NEW_R1}} | {{S1_KHRONOS_SHARED_VALIDATION_NOT_FOUND_F1}} | {{S1_KHRONOS_SHARED_VALIDATION_STALE_FP}} | {{S1_KHRONOS_SHARED_VALIDATION_LOC_ERROR_M}} | {{S1_KHRONOS_SHARED_VALIDATION_RECOVERY_FRAMES}} |
| OVIOVO | online | {{S1_OVIOVO_VALIDATION_PRESENT_R1}} | {{S1_OVIOVO_VALIDATION_MOVED_R1}} | {{S1_OVIOVO_VALIDATION_NEW_R1}} | {{S1_OVIOVO_VALIDATION_NOT_FOUND_F1}} | {{S1_OVIOVO_VALIDATION_STALE_FP}} | {{S1_OVIOVO_VALIDATION_LOC_ERROR_M}} | {{S1_OVIOVO_VALIDATION_RECOVERY_FRAMES}} |

### Held-out test split

| Method | Mode | Present R@1 $\uparrow$ | Moved R@1 $\uparrow$ | New R@1 $\uparrow$ | NOT_FOUND F1 $\uparrow$ | Stale FP $\downarrow$ | Median error m $\downarrow$ | Recovery frames $\downarrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OVI-MAP (frozen) | frozen | {{S1_OVIMAP_FROZEN_TEST_PRESENT_R1}} | {{S1_OVIMAP_FROZEN_TEST_MOVED_R1}} | {{S1_OVIMAP_FROZEN_TEST_NEW_R1}} | {{S1_OVIMAP_FROZEN_TEST_NOT_FOUND_F1}} | {{S1_OVIMAP_FROZEN_TEST_STALE_FP}} | {{S1_OVIMAP_FROZEN_TEST_LOC_ERROR_M}} | {{S1_OVIMAP_FROZEN_TEST_RECOVERY_FRAMES}} |
| ConceptGraphs (frozen) | frozen | {{S1_CONCEPTGRAPHS_FROZEN_TEST_PRESENT_R1}} | {{S1_CONCEPTGRAPHS_FROZEN_TEST_MOVED_R1}} | {{S1_CONCEPTGRAPHS_FROZEN_TEST_NEW_R1}} | {{S1_CONCEPTGRAPHS_FROZEN_TEST_NOT_FOUND_F1}} | {{S1_CONCEPTGRAPHS_FROZEN_TEST_STALE_FP}} | {{S1_CONCEPTGRAPHS_FROZEN_TEST_LOC_ERROR_M}} | {{S1_CONCEPTGRAPHS_FROZEN_TEST_RECOVERY_FRAMES}} |
| DualMap | native | {{S1_DUALMAP_TEST_PRESENT_R1}} | {{S1_DUALMAP_TEST_MOVED_R1}} | {{S1_DUALMAP_TEST_NEW_R1}} | {{S1_DUALMAP_TEST_NOT_FOUND_F1}} | {{S1_DUALMAP_TEST_STALE_FP}} | {{S1_DUALMAP_TEST_LOC_ERROR_M}} | {{S1_DUALMAP_TEST_RECOVERY_FRAMES}} |
| Khronos + shared text head | composed | {{S1_KHRONOS_SHARED_TEST_PRESENT_R1}} | {{S1_KHRONOS_SHARED_TEST_MOVED_R1}} | {{S1_KHRONOS_SHARED_TEST_NEW_R1}} | {{S1_KHRONOS_SHARED_TEST_NOT_FOUND_F1}} | {{S1_KHRONOS_SHARED_TEST_STALE_FP}} | {{S1_KHRONOS_SHARED_TEST_LOC_ERROR_M}} | {{S1_KHRONOS_SHARED_TEST_RECOVERY_FRAMES}} |
| OVIOVO | online | {{S1_OVIOVO_TEST_PRESENT_R1}} | {{S1_OVIOVO_TEST_MOVED_R1}} | {{S1_OVIOVO_TEST_NEW_R1}} | {{S1_OVIOVO_TEST_NOT_FOUND_F1}} | {{S1_OVIOVO_TEST_STALE_FP}} | {{S1_OVIOVO_TEST_LOC_ERROR_M}} | {{S1_OVIOVO_TEST_RECOVERY_FRAMES}} |

## Table S2: Temporal Identity on 3RScan

**Caption.** Streaming 3RScan identity evaluation without future scans. ReScene4D is an offline upper bound and is not ranked against online methods.

**Claim.** Dormancy and re-identification should preserve identity without false reactivation.

### Cross-visit identity

| Method | Mode | Per-stage AP50 $\uparrow$ | t-AP $\uparrow$ | t-REC $\uparrow$ | ID switches $\downarrow$ | Reactivation R@1 $\uparrow$ | False ReID $\downarrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ESAM (per-visit) | per-visit | {{S2_ESAM_VISIT_STAGE_AP50}} | {{S2_ESAM_VISIT_T_AP}} | {{S2_ESAM_VISIT_T_REC}} | {{S2_ESAM_VISIT_IDSW}} | {{S2_ESAM_VISIT_REACT_R1}} | {{S2_ESAM_VISIT_FALSE_REID}} |
| Khronos (adapted stream) | online-adapted | {{S2_KHRONOS_ADAPTED_STAGE_AP50}} | {{S2_KHRONOS_ADAPTED_T_AP}} | {{S2_KHRONOS_ADAPTED_T_REC}} | {{S2_KHRONOS_ADAPTED_IDSW}} | {{S2_KHRONOS_ADAPTED_REACT_R1}} | {{S2_KHRONOS_ADAPTED_FALSE_REID}} |
| ReScene4D | offline | {{S2_RESCENE4D_STAGE_AP50}} | {{S2_RESCENE4D_T_AP}} | {{S2_RESCENE4D_T_REC}} | {{S2_RESCENE4D_IDSW}} | {{S2_RESCENE4D_REACT_R1}} | {{S2_RESCENE4D_FALSE_REID}} |
| OVIOVO (no re-ID) | online | {{S2_OVIOVO_NO_REID_STAGE_AP50}} | {{S2_OVIOVO_NO_REID_T_AP}} | {{S2_OVIOVO_NO_REID_T_REC}} | {{S2_OVIOVO_NO_REID_IDSW}} | {{S2_OVIOVO_NO_REID_REACT_R1}} | {{S2_OVIOVO_NO_REID_FALSE_REID}} |
| OVIOVO | online | {{S2_OVIOVO_STAGE_AP50}} | {{S2_OVIOVO_T_AP}} | {{S2_OVIOVO_T_REC}} | {{S2_OVIOVO_IDSW}} | {{S2_OVIOVO_REACT_R1}} | {{S2_OVIOVO_FALSE_REID}} |

## Table S3: Absence Reliability and Calibration

**Caption.** Presence and absence reliability on held-out queries. Composed baseline rejectors and OVIOVO use one validation-only calibration policy serialized before test execution.

**Claim.** A current-state map must reject absent targets instead of always returning an embedding match.

### Presence and rejection reliability

| Method | Mode | Presence AUROC $\uparrow$ | NOT_FOUND precision $\uparrow$ | NOT_FOUND recall $\uparrow$ | NOT_FOUND F1 $\uparrow$ | Binary ECE $\downarrow$ | Risk-coverage AUC $\downarrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DualMap + calibrated rejector | composed | {{S3_DUALMAP_CAL_PRESENCE_AUROC}} | {{S3_DUALMAP_CAL_NOT_FOUND_PREC}} | {{S3_DUALMAP_CAL_NOT_FOUND_REC}} | {{S3_DUALMAP_CAL_NOT_FOUND_F1}} | {{S3_DUALMAP_CAL_BINARY_ECE}} | {{S3_DUALMAP_CAL_RISK_COVERAGE_AUC}} |
| ConceptGraphs + calibrated rejector | composed | {{S3_CONCEPTGRAPHS_CAL_PRESENCE_AUROC}} | {{S3_CONCEPTGRAPHS_CAL_NOT_FOUND_PREC}} | {{S3_CONCEPTGRAPHS_CAL_NOT_FOUND_REC}} | {{S3_CONCEPTGRAPHS_CAL_NOT_FOUND_F1}} | {{S3_CONCEPTGRAPHS_CAL_BINARY_ECE}} | {{S3_CONCEPTGRAPHS_CAL_RISK_COVERAGE_AUC}} |
| Khronos shared head + rejector | composed | {{S3_KHRONOS_SHARED_CAL_PRESENCE_AUROC}} | {{S3_KHRONOS_SHARED_CAL_NOT_FOUND_PREC}} | {{S3_KHRONOS_SHARED_CAL_NOT_FOUND_REC}} | {{S3_KHRONOS_SHARED_CAL_NOT_FOUND_F1}} | {{S3_KHRONOS_SHARED_CAL_BINARY_ECE}} | {{S3_KHRONOS_SHARED_CAL_RISK_COVERAGE_AUC}} |
| OVIOVO (uncalibrated) | online | {{S3_OVIOVO_UNCAL_PRESENCE_AUROC}} | {{S3_OVIOVO_UNCAL_NOT_FOUND_PREC}} | {{S3_OVIOVO_UNCAL_NOT_FOUND_REC}} | {{S3_OVIOVO_UNCAL_NOT_FOUND_F1}} | {{S3_OVIOVO_UNCAL_BINARY_ECE}} | {{S3_OVIOVO_UNCAL_RISK_COVERAGE_AUC}} |
| OVIOVO | calibrated | {{S3_OVIOVO_PRESENCE_AUROC}} | {{S3_OVIOVO_NOT_FOUND_PREC}} | {{S3_OVIOVO_NOT_FOUND_REC}} | {{S3_OVIOVO_NOT_FOUND_F1}} | {{S3_OVIOVO_BINARY_ECE}} | {{S3_OVIOVO_RISK_COVERAGE_AUC}} |
