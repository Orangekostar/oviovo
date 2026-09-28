# Complementary composition results

Status: **PARTIAL**. Confirmation: **NOT_RUN**.

Historical CAL and regression scenes are previously exposed; Q_GAIN checkpoint selection already used historical CAL. Cross-fitted temperatures do not create a fresh holdout. Two confirmation scenes cannot establish generalization. No deployment was changed.

## Table A — available measured performance

Metrics are percentages. Logical N/S2 counts are conservative required source operations; the common native map is listed separately in each numerical row. Physical shared work is counted once in Table D.

| Role | Scene | Method | uAP | AP50 | AP25 | mIoU | mAcc | Changed/all owners | Positive/evaluated | Logical N/S2/crops | Physical work / shared dependencies | Source reuse | Evaluation first method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| compose_cal | scene0056_00 | CP_M1_AGREE_KEEP | 1.805973 | 9.809328 | 22.707654 | 15.871237 | 19.490412 | 7/99 | 99/61 | 1279/280/9354 | `{"Q_GAIN": {"cache_hits": 109, "crop_inputs": 546, "inference_seconds": 18.731678278069012, "model_forwards": 91, "model_load_seconds": 2.5390652009518817, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M1_AGREE_KEEP |
| compose_cal | scene0056_00 | CP_M2_EQUAL_CAL | 1.628271 | 8.265606 | 28.359711 | 19.207170 | 21.898430 | 28/99 | 99/61 | 1279/280/9354 | `{"Q_GAIN": {"cache_hits": 109, "crop_inputs": 546, "inference_seconds": 18.731678278069012, "model_forwards": 91, "model_load_seconds": 2.5390652009518817, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_CAL |
| compose_cal | scene0056_00 | CP_M2_EQUAL_RAW | 1.628271 | 8.265606 | 26.573996 | 18.902699 | 23.085861 | 26/99 | 99/61 | 1279/280/9354 | `{"Q_GAIN": {"cache_hits": 109, "crop_inputs": 546, "inference_seconds": 18.731678278069012, "model_forwards": 91, "model_load_seconds": 2.5390652009518817, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_RAW |
| compose_cal | scene0056_00 | CP_M3_COMBINE_S2 | 1.710128 | 9.506803 | 20.508787 | 16.205523 | 17.188145 | 76/99 | 43/33 | 200/200/2400 | `{"controller": {"cache_hits": 200, "charged_once": true, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_loads": 0, "reused_integration_check": "/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/technical/native_parity/scene0056_00/receipt.json", "tiles": 0}, "reader": {"cache_hits": 29, "crop_inputs": 1026, "inference_seconds": 36.24356622714549, "model_forwards": 171, "model_load_seconds": 11.818606934975833, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M3_COMBINE_S2 |
| compose_cal | scene0056_00 | CP_M4_GAIN_S2 | 1.775990 | 9.897605 | 22.173328 | 18.081824 | 20.292123 | 53/99 | 96/60 | 200/200/2400 | `{"controller": {"cache_hits": 109, "crop_inputs": 546, "inference_seconds": 18.731678278069012, "model_forwards": 91, "model_load_seconds": 2.5390652009518817, "model_loads": 1, "tiles": 0}, "reader": {"cache_hits": 63, "crop_inputs": 822, "inference_seconds": 29.381081162486225, "model_forwards": 137, "model_load_seconds": 19.84571957902517, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M4_GAIN_S2 |
| compose_cal | scene0056_00 | CP_M5_MIX50_NATIVE | 1.789966 | 9.991497 | 22.303198 | 16.108331 | 18.303756 | 38/99 | 84/57 | 200/0/1200 | `{"cache_hits": 159, "crop_inputs": 246, "inference_seconds": 8.467599669238552, "model_forwards": 41, "model_load_seconds": 1.8454811939736828, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | CP_M5_MIX50_NATIVE |
| compose_cal | scene0056_00 | N0 | 1.818855 | 9.853574 | 21.813028 | 15.376647 | 18.601010 | 0/99 | 99/61 | 1079/0/6474 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| compose_cal | scene0056_00 | Q_COMBINE | 2.117977 | 11.391723 | 23.176729 | 18.866548 | 20.950140 | 68/99 | 43/33 | 200/0/1200 | `{"cache_hits": 200, "charged_once": true, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_loads": 0, "reused_integration_check": "/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/technical/native_parity/scene0056_00/receipt.json", "tiles": 0}` | REUSE_REQUIRED_NATIVE_PARITY | Q_COMBINE |
| compose_cal | scene0056_00 | Q_GAIN | 1.815739 | 10.000000 | 23.161617 | 16.617066 | 20.017795 | 33/99 | 96/60 | 200/0/1200 | `{"cache_hits": 109, "crop_inputs": 546, "inference_seconds": 18.731678278069012, "model_forwards": 91, "model_load_seconds": 2.5390652009518817, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | Q_GAIN |
| compose_cal | scene0056_00 | S_SIGLIP2_AREA | 1.566830 | 7.939342 | 27.544407 | 16.875080 | 18.884515 | 43/99 | 99/61 | 1079/280/8154 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |
| compose_cal | scene0534_00 | CP_M1_AGREE_KEEP | 5.068226 | 17.543860 | 30.263158 | 24.830146 | 31.914726 | 9/93 | 93/78 | 1102/264/8196 | `{"Q_GAIN": {"cache_hits": 125, "crop_inputs": 450, "inference_seconds": 15.578504843171686, "model_forwards": 75, "model_load_seconds": 2.5346053789835423, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M1_AGREE_KEEP |
| compose_cal | scene0534_00 | CP_M2_EQUAL_CAL | 5.701754 | 19.736842 | 26.023392 | 27.682762 | 35.899954 | 38/93 | 93/78 | 1102/264/8196 | `{"Q_GAIN": {"cache_hits": 125, "crop_inputs": 450, "inference_seconds": 15.578504843171686, "model_forwards": 75, "model_load_seconds": 2.5346053789835423, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_CAL |
| compose_cal | scene0534_00 | CP_M2_EQUAL_RAW | 5.701754 | 19.736842 | 27.339181 | 28.002512 | 36.283478 | 39/93 | 93/78 | 1102/264/8196 | `{"Q_GAIN": {"cache_hits": 125, "crop_inputs": 450, "inference_seconds": 15.578504843171686, "model_forwards": 75, "model_load_seconds": 2.5346053789835423, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_RAW |
| compose_cal | scene0534_00 | CP_M3_COMBINE_S2 | 4.483431 | 12.280702 | 24.232456 | 21.858774 | 31.479416 | 70/93 | 46/41 | 200/200/2400 | `{"controller": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "reader": {"cache_hits": 35, "crop_inputs": 990, "inference_seconds": 34.00042107549962, "model_forwards": 165, "model_load_seconds": 10.671198635012843, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M3_COMBINE_S2 |
| compose_cal | scene0534_00 | CP_M4_GAIN_S2 | 4.483431 | 14.035088 | 32.163743 | 26.840345 | 39.350823 | 56/93 | 83/70 | 200/200/2400 | `{"controller": {"cache_hits": 125, "crop_inputs": 450, "inference_seconds": 15.578504843171686, "model_forwards": 75, "model_load_seconds": 2.5346053789835423, "model_loads": 1, "tiles": 0}, "reader": {"cache_hits": 56, "crop_inputs": 864, "inference_seconds": 31.039804264204577, "model_forwards": 144, "model_load_seconds": 11.797017112956382, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M4_GAIN_S2 |
| compose_cal | scene0534_00 | CP_M5_MIX50_NATIVE | 4.483431 | 12.280702 | 26.578947 | 24.166663 | 33.807048 | 47/93 | 69/63 | 200/0/1200 | `{"cache_hits": 150, "crop_inputs": 300, "inference_seconds": 10.387859101290815, "model_forwards": 50, "model_load_seconds": 1.8412782330997288, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | CP_M5_MIX50_NATIVE |
| compose_cal | scene0534_00 | N0 | 5.068226 | 17.543860 | 35.619096 | 25.936995 | 35.668579 | 0/93 | 93/78 | 902/0/5412 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| compose_cal | scene0534_00 | Q_COMBINE | 4.483431 | 12.280702 | 25.877193 | 23.389570 | 29.420872 | 61/93 | 46/41 | 200/0/1200 | `{"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}` | FORCED_TRAJECTORY | Q_COMBINE |
| compose_cal | scene0534_00 | Q_GAIN | 4.824561 | 17.105263 | 23.391813 | 21.429484 | 26.859862 | 42/93 | 83/70 | 200/0/1200 | `{"cache_hits": 125, "crop_inputs": 450, "inference_seconds": 15.578504843171686, "model_forwards": 75, "model_load_seconds": 2.5346053789835423, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | Q_GAIN |
| compose_cal | scene0534_00 | S_SIGLIP2_AREA | 5.896686 | 19.736842 | 26.864035 | 27.521707 | 38.932137 | 45/93 | 93/78 | 902/264/6996 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |

| Role | Method | Mean uAP (defined/total) | Mean mIoU (defined/total) | Status |
|---|---|---:|---:|---|
| compose_cal | N0 | 3.443540 (2/2) | 20.656821 (2/2) | COMPLETE |
| compose_cal | Q_COMBINE | 3.300704 (2/2) | 21.128059 (2/2) | COMPLETE |
| compose_cal | Q_GAIN | 3.320150 (2/2) | 19.023275 (2/2) | COMPLETE |
| compose_cal | S_SIGLIP2_AREA | 3.731758 (2/2) | 22.198394 (2/2) | COMPLETE |
| compose_cal | CP_M1_AGREE_KEEP | 3.437100 (2/2) | 20.350692 (2/2) | COMPLETE |
| compose_cal | CP_M2_EQUAL_RAW | 3.665013 (2/2) | 23.452605 (2/2) | COMPLETE |
| compose_cal | CP_M2_EQUAL_CAL | 3.665013 (2/2) | 23.444966 (2/2) | COMPLETE |
| compose_cal | CP_M3_COMBINE_S2 | 3.096780 (2/2) | 19.032149 (2/2) | COMPLETE |
| compose_cal | CP_M4_GAIN_S2 | 3.129711 (2/2) | 22.461084 (2/2) | COMPLETE |
| compose_cal | CP_M5_MIX50_NATIVE | 3.136698 (2/2) | 20.137497 (2/2) | COMPLETE |

Missing required control/composition records: 20. Missing rows are not zero-valued measurements.

## Table B — complementarity and routing

| Scene | Status | Owners | Correctness categories | Source unavailable counts |
|---|---|---:|---|---|
| scene0056_00 | COMPLETE | 99 | {'SHAPE_UNMATCHED': 86, 'Q_ONLY_CORRECT': 1, 'ALL_CORRECT': 7, 'ALL_WRONG': 4, 'N0_Q_CORRECT': 1} | {'N0': 0, 'S_SIGLIP2_AREA': 2, 'Q_GAIN': 3} |
| scene0534_00 | COMPLETE | 93 | {'SHAPE_UNMATCHED': 84, 'ALL_WRONG': 4, 'ALL_CORRECT': 4, 'S2_ONLY_CORRECT': 1} | {'N0': 0, 'S_SIGLIP2_AREA': 2, 'Q_GAIN': 10} |

For scenes with complete source evidence, the numerical tables preserve all-owner correctness, source GT-class ranks, unavailable/unmatched/ambiguous categories and available M1/M2 routing counts. SOURCE_EVIDENCE_INCOMPLETE means those analyses remain pending. Oracle-correctable object counts are diagnostic, not an AP upper bound.

## Table C — controlled contrasts and trajectory mechanics

| Role | Contrast | ΔuAP (pp) | ΔmIoU (pp) |
|---|---|---:|---:|
| compose_cal | CP_M3_COMBINE_S2 − Q_COMBINE | -0.203924 | -2.095910 |
| compose_cal | CP_M4_GAIN_S2 − Q_GAIN | -0.190439 | 3.437810 |
| compose_cal | CP_M4_GAIN_S2 − CP_M3_COMBINE_S2 | 0.032931 | 3.428936 |
| compose_cal | CP_M2_EQUAL_CAL − CP_M2_EQUAL_RAW | 0.000000 | -0.007639 |
| compose_cal | CP_M5_MIX50_NATIVE − Q_COMBINE | -0.164005 | -0.990562 |
| compose_cal | CP_M5_MIX50_NATIVE − Q_GAIN | -0.183452 | 1.114222 |
| compose_cal | 2x2_INTERACTION − (M4-Q_GAIN)-(M3-Q_COMBINE) | 0.013485 | 5.533720 |

Lane preference/winner/fallback counts, request Jaccards, per-owner paid budgets and retained/dropped evidence are in `trajectory_diagnostics`. A changed trajectory alone is not a gain.

## Table D — frozen nomination, confirmation and operations

Nominee: **CP_M2_EQUAL_RAW**.
Experiment commit A: `7a9bd5a212e1f85619235c15a06b405eece2be39`.
Query physical totals: `{'model_loads': 8, 'model_forwards': 874, 'crop_inputs': 5244, 'cache_hits': 1325, 'inference_seconds': 183.83051462110598}`; new native captures: 0.
Historical static S2 reused requests: 584. CPU calibration: 9 completed scalar fits, 168 recorded optimizer evaluations; initial NLL evaluations are listed separately in the numerical ledger. Calibration elapsed time was not instrumented.

| Role | Nominee versus | Status | Worst ΔuAP / ΔmIoU (pp) | Every scene nonnegative |
|---|---|---|---:|---|
| compose_cal | N0 | MEAN_GAIN_WITH_SCENE_TRADEOFF | -0.190584 / 2.065517 | False |
| compose_cal | S_SIGLIP2_AREA | NO_MEAN_GAIN | -0.194932 / 0.480804 | False |
| regression_only | N0 | INCONCLUSIVE_INCOMPLETE_ROWS | — / — | — |
| regression_only | S_SIGLIP2_AREA | INCONCLUSIVE_INCOMPLETE_ROWS | — / — | — |
| confirmation | N0 | INCONCLUSIVE_INCOMPLETE_ROWS | — / — | — |
| confirmation | S_SIGLIP2_AREA | INCONCLUSIVE_INCOMPLETE_ROWS | — / — | — |

Actual released TP/FN gains/losses and FP events at 0.5/0.75 are linked in `released_transitions`, separately from geometric class correctness. AP increments are not added across objects.
The observed M2_RAW/M2_CAL equal-uAP case is examined in [UNCHANGED_AP_DIAGNOSTICS.md](complementary_composition_v1/UNCHANGED_AP_DIAGNOSTICS.md), including actual eligibility, ignored predictions, class AP states and the tied-score FP responsible for its AP25 change.

Complete numerical tables: `/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/reports/a522b8e0584e5ac73094ae2785f63304756a18f38a56783dfbaec0190e881d34/tables.json`.
Publication receipt (written only after verified ordinary push): `/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/publication_receipt.json`.
