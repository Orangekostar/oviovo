# OVIOVO AAAI Experiment Framework Design

**Date:** 2026-07-13
**Status:** Framework frozen; numerical cells intentionally unfilled

## 1. Purpose

This document fixes the experiment structure before final implementation and benchmark runs. Numerical results use machine-searchable `{{TOKEN}}` placeholders and must only be filled from frozen, reproducible final runs.

The paper must answer five research questions:

1. **RQ1, static quality:** Does OVIOVO preserve competitive static semantic, instance, and geometric mapping quality?
2. **RQ2, dynamic quality:** Does visibility-driven reversible maintenance produce a more accurate current map under object disappearance, motion, occlusion, and reappearance?
3. **RQ3, identity:** Does fallback re-identification preserve object identity without corrupting normal geometry-first association?
4. **RQ4, causality:** Which gains come from visibility evidence, reversible ownership, background reclaim, and re-identification?
5. **RQ5, practicality:** What are the online latency, memory, and long-sequence stability costs?

## 2. Claim-Evidence Contract

| Claim | Required evidence | Primary artifact |
|---|---|---|
| Geometry-first mapping remains competitive in static scenes | Runtime semantic metrics, class-agnostic instance AP, and geometry F-score | Table 1 |
| Signed visibility evidence distinguishes disappearance from occlusion | Current-map quality under disappearance and occlusion; focused ablation | Tables 2 and 3, Figure 4 |
| Reversible voxel ownership reduces stale geometry without damaging valid surfaces | Ghost Rate, background recovery F-score, static-quality retention | Tables 2 and 3 |
| Fallback re-identification preserves identity after relocation | t-AP, ID switches, reactivation recall, false re-ID rate | Tables 2 and 3 |
| The system is viable online | Per-stage latency, peak memory, map size, finalization cost | Table 4 |

No abstract or introduction claim may be stronger than the corresponding completed evidence row.

## 3. Frozen Evaluation Protocol

### 3.1 Dataset Roles

| Role | Dataset | Use |
|---|---|---|
| Development only | Replica `room0`, 200 frames, stride 10 | Debugging and regression gates; never the sole paper result |
| Primary static benchmark | Replica eight locally available scenes | Static semantic, instance, and geometry quality |
| Static generalization | ScanNet evaluation manifest | Generalization beyond synthetic Replica scenes |
| Primary continuous-dynamics benchmark | Khronos-compatible dynamic sequences | Current-map accuracy, ghost removal, background recovery |
| Cross-session change and identity benchmark | 3RScan/ReScene4D protocol with `stmetrics` | Temporal AP, change recall, and re-identification |

Before final runs, each benchmark must have a committed manifest containing scene IDs, frame ranges, stride, vocabulary, class aliases, depth scale, and pose source. Scene-specific prompt tuning is prohibited.

### 3.2 Fairness Rules

1. All methods receive the same RGB-D frames, poses, resolution, frame stride, and canonical vocabulary whenever their interfaces permit it.
2. Runtime semantic prediction must use only model outputs available at inference time. Ground truth may only enter the evaluator.
3. Replica `room0` is a development scene. Hyperparameters selected on `room0` are frozen before the other scenes are evaluated.
4. Geometry metrics use fixed thresholds of 2 cm, 5 cm, and 10 cm; the 5 cm result is the primary value.
5. Per-scene values and macro averages are retained. Frequency-weighted metrics never replace macro metrics.
6. Stochastic configurations use three fixed seeds and report mean and standard deviation. Deterministic configurations report one run and are marked deterministic.
7. Initialization, online mapping, finalization, and evaluation/I/O times are reported separately on one fixed hardware configuration.
8. External repositories run in isolated environments and exchange predictions through files; their source code is not imported into OVIOVO.
9. Any baseline adaptation, unavailable official component, or fallback implementation is disclosed directly in the table caption.

### 3.3 Metric Names and Definitions

| Metric | Definition and role |
|---|---|
| `Runtime-Sem mIoU/mAcc` | Semantic labels produced by YOLO-World or the configured VLM backend; no GT-derived object labels. This is the primary semantic result. |
| `Oracle-Inst mIoU/mAcc` | Each predicted object receives its majority GT class. This diagnoses geometric partition and instance contamination only and must never be called open-vocabulary semantic accuracy. |
| `AP25/AP50` | Class-agnostic 3D instance AP under the OVI-MAP-compatible mask protocol. |
| `Geometry F@5cm` | Harmonic mean of surface precision and completeness at 5 cm. |
| `Coverage@5cm` | Fraction of GT surface points within 5 cm of the predicted current surface. |
| `Current mIoU` | Runtime semantic mIoU evaluated at each annotated time and macro-averaged over time and sequences. |
| `Ghost Rate` | Fraction of predicted object voxels in changed regions that lie in GT-confirmed free space at the current time. Lower is better. |
| `Change F1` | Voxel-level F1 for changed versus unchanged regions between annotated times. |
| `Background Recovery F@5cm` | Surface F-score inside regions revealed after foreground removal. |
| `t-AP / t-REC` | Temporal instance and change metrics produced through the `stmetrics` contract. |
| `IDSW` | Number of predicted identity switches along matched GT object trajectories. Lower is better. |
| `ReID R@1` | Fraction of reappearing or relocated GT objects reactivated with the correct historical ID. |
| `False ReID` | Fraction of re-ID decisions that merge distinct GT identities. Lower is better. |

The legacy evaluator in `run_room0_full_eval.py` is retained only as the implementation source for `Oracle-Inst` diagnostics. A separate runtime-semantic evaluator is required for all primary semantic claims.

## 4. Compared Methods

### 4.1 Static Table

- Panoptic Mapping: classical closed-vocabulary geometric baseline.
- OpenFusion: dense open-vocabulary baseline.
- ConceptGraphs: object-centric open-vocabulary baseline.
- OpenVox: probabilistic instance-voxel baseline.
- OVI-MAP: closest static instance-semantic mapping baseline and evaluator reference.
- OVIOVO: proposed method.

### 4.2 Dynamic Table

- Static OVIOVO: no lifecycle removal, representing ordinary cumulative mapping.
- Timeout OVIOVO: current frame-count lifecycle baseline.
- DualMap: dynamic open-vocabulary mapping baseline.
- Khronos: spatio-temporal metric and dynamic mapping baseline where compatible outputs are available.
- Full OVIOVO: signed visibility, reversible ownership, background reclaim, and fallback re-ID.

Methods that cannot produce a metric are marked `N/A`, never assigned zero. The caption must explain interface incompatibilities.

## 5. Main Paper Tables

All tokens below are fill-only fields. A final result importer should replace them from benchmark JSON rather than by manual transcription.

### Table 1. Static Mapping Quality

**Message:** The dynamic design does not depend on a weak or degraded static backbone.

| Method | Dataset | Runtime-Sem mIoU (higher) | mAcc (higher) | AP25 (higher) | AP50 (higher) | Geometry F@5cm (higher) | Coverage@5cm (higher) |
|---|---|---:|---:|---:|---:|---:|---:|
| Panoptic Mapping | Replica | `{{PANOPTIC_REPLICA_MIOU}}` | `{{PANOPTIC_REPLICA_MACC}}` | `{{PANOPTIC_REPLICA_AP25}}` | `{{PANOPTIC_REPLICA_AP50}}` | `{{PANOPTIC_REPLICA_F5}}` | `{{PANOPTIC_REPLICA_COV5}}` |
| OpenFusion | Replica | `{{OPENFUSION_REPLICA_MIOU}}` | `{{OPENFUSION_REPLICA_MACC}}` | `N/A` | `N/A` | `{{OPENFUSION_REPLICA_F5}}` | `{{OPENFUSION_REPLICA_COV5}}` |
| ConceptGraphs | Replica | `{{CONCEPTGRAPHS_REPLICA_MIOU}}` | `{{CONCEPTGRAPHS_REPLICA_MACC}}` | `{{CONCEPTGRAPHS_REPLICA_AP25}}` | `{{CONCEPTGRAPHS_REPLICA_AP50}}` | `{{CONCEPTGRAPHS_REPLICA_F5}}` | `{{CONCEPTGRAPHS_REPLICA_COV5}}` |
| OpenVox | Replica | `{{OPENVOX_REPLICA_MIOU}}` | `{{OPENVOX_REPLICA_MACC}}` | `{{OPENVOX_REPLICA_AP25}}` | `{{OPENVOX_REPLICA_AP50}}` | `{{OPENVOX_REPLICA_F5}}` | `{{OPENVOX_REPLICA_COV5}}` |
| OVI-MAP | Replica | `{{OVIMAP_REPLICA_MIOU}}` | `{{OVIMAP_REPLICA_MACC}}` | `{{OVIMAP_REPLICA_AP25}}` | `{{OVIMAP_REPLICA_AP50}}` | `{{OVIMAP_REPLICA_F5}}` | `{{OVIMAP_REPLICA_COV5}}` |
| **OVIOVO** | Replica | `{{OURS_REPLICA_MIOU}}` | `{{OURS_REPLICA_MACC}}` | `{{OURS_REPLICA_AP25}}` | `{{OURS_REPLICA_AP50}}` | `{{OURS_REPLICA_F5}}` | `{{OURS_REPLICA_COV5}}` |
| OVI-MAP | ScanNet | `{{OVIMAP_SCANNET_MIOU}}` | `{{OVIMAP_SCANNET_MACC}}` | `{{OVIMAP_SCANNET_AP25}}` | `{{OVIMAP_SCANNET_AP50}}` | `{{OVIMAP_SCANNET_F5}}` | `{{OVIMAP_SCANNET_COV5}}` |
| **OVIOVO** | ScanNet | `{{OURS_SCANNET_MIOU}}` | `{{OURS_SCANNET_MACC}}` | `{{OURS_SCANNET_AP25}}` | `{{OURS_SCANNET_AP50}}` | `{{OURS_SCANNET_F5}}` | `{{OURS_SCANNET_COV5}}` |

`Oracle-Inst mIoU/mAcc` is reported in the supplement as a geometry/partition diagnostic, not in the primary semantic columns.

### Table 2. Dynamic Current-Map and Identity Quality

**Message:** OVIOVO removes stale geometry, recovers revealed background, and preserves identity through occlusion or relocation.

| Method | Current mIoU (higher) | Ghost Rate (lower) | Change F1 (higher) | BG Recovery F@5cm (higher) | t-AP (higher) | IDSW (lower) | ReID R@1 (higher) | False ReID (lower) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static OVIOVO | `{{STATIC_CURRENT_MIOU}}` | `{{STATIC_GHOST_RATE}}` | `{{STATIC_CHANGE_F1}}` | `{{STATIC_BG_F5}}` | `{{STATIC_TAP}}` | `{{STATIC_IDSW}}` | `{{STATIC_REID_R1}}` | `{{STATIC_FALSE_REID}}` |
| Timeout OVIOVO | `{{TIMEOUT_CURRENT_MIOU}}` | `{{TIMEOUT_GHOST_RATE}}` | `{{TIMEOUT_CHANGE_F1}}` | `{{TIMEOUT_BG_F5}}` | `{{TIMEOUT_TAP}}` | `{{TIMEOUT_IDSW}}` | `{{TIMEOUT_REID_R1}}` | `{{TIMEOUT_FALSE_REID}}` |
| DualMap | `{{DUALMAP_CURRENT_MIOU}}` | `{{DUALMAP_GHOST_RATE}}` | `{{DUALMAP_CHANGE_F1}}` | `{{DUALMAP_BG_F5}}` | `{{DUALMAP_TAP}}` | `{{DUALMAP_IDSW}}` | `{{DUALMAP_REID_R1}}` | `{{DUALMAP_FALSE_REID}}` |
| Khronos | `{{KHRONOS_CURRENT_MIOU}}` | `{{KHRONOS_GHOST_RATE}}` | `{{KHRONOS_CHANGE_F1}}` | `{{KHRONOS_BG_F5}}` | `{{KHRONOS_TAP}}` | `{{KHRONOS_IDSW}}` | `{{KHRONOS_REID_R1}}` | `{{KHRONOS_FALSE_REID}}` |
| **Full OVIOVO** | `{{OURS_CURRENT_MIOU}}` | `{{OURS_GHOST_RATE}}` | `{{OURS_CHANGE_F1}}` | `{{OURS_BG_F5}}` | `{{OURS_TAP}}` | `{{OURS_IDSW}}` | `{{OURS_REID_R1}}` | `{{OURS_FALSE_REID}}` |

If Khronos or another baseline cannot export semantic or historical identities, only compatible cells are filled and the remaining cells stay `N/A`.

### Table 3. Component Ablation

**Message:** Each proposed component has a distinct, measurable effect and the full method retains static quality.

| Geometry-first association | Signed visibility | Reversible ownership | BG reclaim | Fallback re-ID | Static mIoU (higher) | Ghost Rate (lower) | Change F1 (higher) | BG F@5cm (higher) | IDSW (lower) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Y |  |  |  |  | `{{ABL_BASE_MIOU}}` | `{{ABL_BASE_GHOST}}` | `{{ABL_BASE_CHANGE}}` | `{{ABL_BASE_BG}}` | `{{ABL_BASE_IDSW}}` |
| Y | Y |  |  |  | `{{ABL_VIS_MIOU}}` | `{{ABL_VIS_GHOST}}` | `{{ABL_VIS_CHANGE}}` | `{{ABL_VIS_BG}}` | `{{ABL_VIS_IDSW}}` |
| Y | Y | Y |  |  | `{{ABL_OWNER_MIOU}}` | `{{ABL_OWNER_GHOST}}` | `{{ABL_OWNER_CHANGE}}` | `{{ABL_OWNER_BG}}` | `{{ABL_OWNER_IDSW}}` |
| Y | Y | Y | Y |  | `{{ABL_RECLAIM_MIOU}}` | `{{ABL_RECLAIM_GHOST}}` | `{{ABL_RECLAIM_CHANGE}}` | `{{ABL_RECLAIM_BG}}` | `{{ABL_RECLAIM_IDSW}}` |
| Y | Y | Y | Y | Y | `{{ABL_FULL_MIOU}}` | `{{ABL_FULL_GHOST}}` | `{{ABL_FULL_CHANGE}}` | `{{ABL_FULL_BG}}` | `{{ABL_FULL_IDSW}}` |

Required focused controls:

- Absolute depth consistency versus signed visibility evidence.
- Free-space removal enabled versus treating out-of-view and occlusion as negative evidence.
- Hard owner overwrite versus soft owner support and reversible release.
- Re-ID disabled, label-only re-ID, and descriptor-assisted re-ID.
- Geometry-first association versus semantics injected into normal low-level association.

### Table 4. Runtime and Memory

**Message:** Dynamic maintenance adds bounded overhead and avoids hidden deferred work.

| Method | Frontend s/frame (lower) | Mapping backend s/frame (lower) | Maintenance s/frame (lower) | Total online s/frame (lower) | Finalization s (lower) | Peak GPU GB (lower) | Peak RAM GB (lower) | Final map MB (lower) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DualMap | `{{DUALMAP_FRONTEND_S}}` | `{{DUALMAP_BACKEND_S}}` | `{{DUALMAP_MAINT_S}}` | `{{DUALMAP_TOTAL_S}}` | `{{DUALMAP_FINAL_S}}` | `{{DUALMAP_GPU_GB}}` | `{{DUALMAP_RAM_GB}}` | `{{DUALMAP_MAP_MB}}` |
| OVI-MAP | `{{OVIMAP_FRONTEND_S}}` | `{{OVIMAP_BACKEND_S}}` | `N/A` | `{{OVIMAP_TOTAL_S}}` | `{{OVIMAP_FINAL_S}}` | `{{OVIMAP_GPU_GB}}` | `{{OVIMAP_RAM_GB}}` | `{{OVIMAP_MAP_MB}}` |
| OVIOVO without maintenance | `{{OURS_STATIC_FRONTEND_S}}` | `{{OURS_STATIC_BACKEND_S}}` | `0` | `{{OURS_STATIC_TOTAL_S}}` | `{{OURS_STATIC_FINAL_S}}` | `{{OURS_STATIC_GPU_GB}}` | `{{OURS_STATIC_RAM_GB}}` | `{{OURS_STATIC_MAP_MB}}` |
| **Full OVIOVO** | `{{OURS_FRONTEND_S}}` | `{{OURS_BACKEND_S}}` | `{{OURS_MAINT_S}}` | `{{OURS_TOTAL_S}}` | `{{OURS_FINAL_S}}` | `{{OURS_GPU_GB}}` | `{{OURS_RAM_GB}}` | `{{OURS_MAP_MB}}` |

## 6. Supplementary Tables and Figures

1. Per-scene Replica and ScanNet results with macro averages.
2. `Oracle-Inst mIoU/mAcc`, fragmentation, over-merge rate, and object count.
3. Robustness to depth noise, dropped detections, occlusion duration, and frame stride.
4. Semantic backend comparison: YOLO-World labels only, descriptor only, and labels plus descriptor.
5. Visibility thresholds and lifecycle hysteresis sensitivity.
6. Qualitative timelines for disappearance, occlusion, relocation, and reappearance.
7. Failure cases: same-class distractors, reflective depth, incomplete background exposure, and long out-of-view intervals.

The principal qualitative figure uses one sequence with four synchronized rows: RGB observation, cumulative/static map, OVIOVO current map, and lifecycle/history events. It must show both successful removal and successful reactivation, not only favorable final maps.

## 7. Experiment Section Narrative

The Experiments section follows this order:

1. **Setup and protocol.** Define datasets, fixed vocabulary, runtime-only semantics, geometry thresholds, dynamic annotations, and isolated baseline environments.
2. **Static mapping quality.** Establish that the geometry-first foundation remains competitive before claiming dynamic benefits.
3. **Dynamic current-map quality.** Show that the full method improves current-world accuracy, ghost removal, and background recovery.
4. **Identity persistence.** Analyze occlusion, relocation, and reactivation with temporal metrics and failure cases.
5. **Ablations.** Tie every proposed module to a distinct metric and verify static-quality retention.
6. **Efficiency and robustness.** Report online cost, memory, long-sequence behavior, and sensitivity.

The central narrative is:

> Anchor-guided masks and geometry-first TSDF association provide stable observations, but cumulative fusion cannot determine whether old geometry remains valid. OVIOVO closes this loop with signed visibility evidence, reversible ownership, and fallback re-identification, producing an accurate current map while retaining historical object identity.

## 8. Result-Filling Rules

1. Placeholder tokens are replaced only from committed benchmark JSON files generated under frozen manifests.
2. Every table row records the repository commit, model weights, command, environment, hardware, and output directory in a result manifest.
3. Failed or incompatible runs remain `N/A`; no number may be copied from a paper under a different protocol into a locally reproduced comparison row.
4. Best and second-best formatting is applied only after all compatible cells are filled.
5. The paper reports regressions and failure modes; low or negative results are not silently removed after the experiment matrix is frozen.

## 9. Acceptance Gates Before Final Number Filling

1. Runtime semantic predictions are invariant to GT-label permutation.
2. `Oracle-Inst` and runtime-semantic metrics are generated by separate evaluator entry points.
3. Static OVIOVO recovers the validated static floor under the frozen development protocol before dynamic modules are enabled.
4. Occlusion and out-of-view frames do not create absence evidence in synthetic unit sequences.
5. Releasing one instance preserves competing voxel support and valid static geometry.
6. Background recovery is evaluated only in GT-confirmed revealed regions.
7. Re-ID reports both recall and false merge rate.
8. Final claims use multi-scene or multi-sequence macro averages, never Replica `room0` alone.
