# Complementary composition results

Status: **PARTIAL**. Confirmation: **NOT_RUN**.

Historical CAL and regression scenes are previously exposed; Q_GAIN checkpoint selection already used historical CAL. Cross-fitted temperatures do not create a fresh holdout. Two confirmation scenes cannot establish generalization. No deployment was changed.

## Table A — available measured performance

Metrics are percentages. Logical N/S2 counts are conservative required source operations; the common native map is listed separately in each numerical row. Physical shared work is counted once in Table D.

| Role | Scene | Method | uAP | AP50 | AP25 | mIoU | mAcc | Changed/all owners | Positive/evaluated | Logical N/S2/crops | Physical work / shared dependencies | Source reuse | Evaluation first method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| compose_cal | scene0056_00 | N0 | 1.818855 | 9.853574 | 21.813028 | 15.376647 | 18.601010 | 0/99 | 99/61 | 1079/0/6474 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| compose_cal | scene0056_00 | Q_COMBINE | 2.117977 | 11.391723 | 23.176729 | 18.866548 | 20.950140 | 68/99 | 43/33 | 200/0/1200 | `{"cache_hits": 200, "charged_once": true, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_loads": 0, "reused_integration_check": "/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/technical/native_parity/scene0056_00/receipt.json", "tiles": 0}` | REUSE_REQUIRED_NATIVE_PARITY | Q_COMBINE |
| compose_cal | scene0056_00 | S_SIGLIP2_AREA | 1.566830 | 7.939342 | 27.544407 | 16.875080 | 18.884515 | 43/99 | 99/61 | 1079/280/8154 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |
| compose_cal | scene0534_00 | N0 | 5.068226 | 17.543860 | 35.619096 | 25.936995 | 35.668579 | 0/93 | 93/78 | 902/0/5412 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| compose_cal | scene0534_00 | S_SIGLIP2_AREA | 5.896686 | 19.736842 | 26.864035 | 27.521707 | 38.932137 | 45/93 | 93/78 | 902/264/6996 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |

| Role | Method | Mean uAP (defined/total) | Mean mIoU (defined/total) | Status |
|---|---|---:|---:|---|
| compose_cal | N0 | 3.443540 (2/2) | 20.656821 (2/2) | COMPLETE |
| compose_cal | Q_COMBINE | 2.117977 (1/2) | 18.866548 (1/2) | MISSING_ROWS |
| compose_cal | S_SIGLIP2_AREA | 3.731758 (2/2) | 22.198394 (2/2) | COMPLETE |

Missing required control/composition records: 35. Missing rows are not zero-valued measurements.

## Table B — complementarity and routing

| Scene | Status | Owners | Correctness categories | Source unavailable counts |
|---|---|---:|---|---|
| scene0056_00 | SOURCE_EVIDENCE_INCOMPLETE | — | — | — |
| scene0534_00 | SOURCE_EVIDENCE_INCOMPLETE | — | — | — |

For scenes with complete source evidence, the numerical tables preserve all-owner correctness, source GT-class ranks, unavailable/unmatched/ambiguous categories and available M1/M2 routing counts. SOURCE_EVIDENCE_INCOMPLETE means those analyses remain pending. Oracle-correctable object counts are diagnostic, not an AP upper bound.

## Table C — controlled contrasts and trajectory mechanics

| Role | Contrast | ΔuAP (pp) | ΔmIoU (pp) |
|---|---|---:|---:|

Lane preference/winner/fallback counts, request Jaccards, per-owner paid budgets and retained/dropped evidence are in `trajectory_diagnostics`. A changed trajectory alone is not a gain.

## Table D — frozen nomination, confirmation and operations

Nominee: **NOT_FROZEN**.
Experiment commit A: `NOT_FROZEN`.
Query physical totals: `{'model_loads': 0, 'model_forwards': 0, 'crop_inputs': 0, 'cache_hits': 200, 'inference_seconds': 0.0}`; new native captures: 0.

| Role | Nominee versus | Status | Worst ΔuAP / ΔmIoU (pp) | Every scene nonnegative |
|---|---|---|---:|---|

Actual released TP/FN gains/losses and FP events at 0.5/0.75 are linked in `released_transitions`, separately from geometric class correctness. AP increments are not added across objects.

Complete numerical tables: `/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/reports/61b4a177915d6bcdde8079d6c2da46efc62bdd608c862affd217d1b1025b8651/tables.json`.
Publication receipt (written only after verified ordinary push): `/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/publication_receipt.json`.
