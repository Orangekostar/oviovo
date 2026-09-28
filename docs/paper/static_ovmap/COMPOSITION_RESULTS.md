# Complementary composition results

Status: **COMPLETE**. Confirmation: **COMPLETE**.

Historical CAL and regression scenes are previously exposed; Q_GAIN checkpoint selection already used historical CAL. Cross-fitted temperatures do not create a fresh holdout. Two confirmation scenes cannot establish generalization. No deployment was changed.

## Table A — available measured performance

Metrics are percentages. Logical N/S2 counts are conservative required source operations; the common native map is listed separately in each numerical row. Physical shared work is counted once in Table D.
Physical cells describe query/readout work. Confirmation source rows reuse N0/S2 generated earlier in the same authorized attempt; their zero additional readout forwards do not make the new capture or static S2 free. New frontend, mapping and static S2 jobs are listed separately in Table D's numerical ledger.

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
| confirmation | scene0064_00 | CP_M2_EQUAL_RAW | 7.916667 | 19.166667 | 36.085859 | 19.524671 | 25.715520 | 22/63 | 63/48 | 943/179/6732 | `{"Q_GAIN": {"cache_hits": 31, "crop_inputs": 1014, "inference_seconds": 35.84443709149491, "model_forwards": 169, "model_load_seconds": 2.448856968083419, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_RAW |
| confirmation | scene0064_00 | N0 | 2.361111 | 8.055556 | 24.974747 | 11.278576 | 17.412030 | 0/63 | 63/48 | 743/0/4458 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| confirmation | scene0064_00 | Q_COMBINE | 5.967078 | 13.148148 | 29.814815 | 14.461502 | 20.422631 | 43/63 | 31/26 | 200/0/1200 | `{"cache_hits": 0, "crop_inputs": 1200, "inference_seconds": 43.30909981718287, "model_forwards": 200, "model_load_seconds": 1.8742711500963196, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | Q_COMBINE |
| confirmation | scene0064_00 | Q_GAIN | 7.752792 | 18.386243 | 35.305435 | 18.187115 | 24.184154 | 25/63 | 59/46 | 200/0/1200 | `{"cache_hits": 31, "crop_inputs": 1014, "inference_seconds": 35.84443709149491, "model_forwards": 169, "model_load_seconds": 2.448856968083419, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | Q_GAIN |
| confirmation | scene0064_00 | S_SIGLIP2_AREA | 4.989712 | 19.907407 | 31.776094 | 16.950823 | 26.666569 | 30/63 | 63/48 | 743/179/5532 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |
| confirmation | scene0553_00 | CP_M2_EQUAL_RAW | 25.000000 | 33.333333 | 33.333333 | 30.714752 | 31.025902 | 12/30 | 30/22 | 559/87/3876 | `{"Q_GAIN": {"cache_hits": 58, "crop_inputs": 852, "inference_seconds": 30.556340308277868, "model_forwards": 142, "model_load_seconds": 2.4230722869979218, "model_loads": 1, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_RAW |
| confirmation | scene0553_00 | N0 | 19.444444 | 25.000000 | 25.000000 | 23.743545 | 24.003368 | 0/30 | 30/22 | 359/0/2154 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| confirmation | scene0553_00 | Q_COMBINE | 19.444444 | 25.000000 | 25.000000 | 23.743545 | 24.003368 | 11/30 | 26/22 | 200/0/1200 | `{"cache_hits": 0, "crop_inputs": 1200, "inference_seconds": 42.4114926608745, "model_forwards": 200, "model_load_seconds": 1.8474325330462307, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | Q_COMBINE |
| confirmation | scene0553_00 | Q_GAIN | 19.444444 | 25.000000 | 25.000000 | 21.696035 | 21.955075 | 9/30 | 30/22 | 200/0/1200 | `{"cache_hits": 58, "crop_inputs": 852, "inference_seconds": 30.556340308277868, "model_forwards": 142, "model_load_seconds": 2.4230722869979218, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | Q_GAIN |
| confirmation | scene0553_00 | S_SIGLIP2_AREA | 17.592593 | 25.000000 | 31.250000 | 31.491277 | 32.343547 | 18/30 | 30/22 | 359/87/2676 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |
| regression_only | scene0445_00 | CP_M1_AGREE_KEEP | 42.222222 | 60.000000 | 60.000000 | 51.705610 | 53.403678 | 0/16 | 16/15 | 411/40/2706 | `{"Q_GAIN": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | N0 |
| regression_only | scene0445_00 | CP_M2_EQUAL_CAL | 46.666667 | 60.000000 | 62.000000 | 58.617111 | 60.753663 | 5/16 | 16/15 | 411/40/2706 | `{"Q_GAIN": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_CAL |
| regression_only | scene0445_00 | CP_M2_EQUAL_RAW | 38.888889 | 50.000000 | 50.000000 | 46.004286 | 47.662628 | 2/16 | 16/15 | 411/40/2706 | `{"Q_GAIN": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_RAW |
| regression_only | scene0445_00 | CP_M3_COMBINE_S2 | 28.888889 | 40.000000 | 40.000000 | 37.549641 | 39.172769 | 9/16 | 15/14 | 199/199/2388 | `{"controller": {"cache_hits": 199, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "reader": {"cache_hits": 9, "crop_inputs": 1140, "inference_seconds": 40.439086253754795, "model_forwards": 190, "model_load_seconds": 21.6342200540239, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M3_COMBINE_S2 |
| regression_only | scene0445_00 | CP_M4_GAIN_S2 | 38.888889 | 50.000000 | 52.000000 | 50.906770 | 52.858575 | 7/16 | 16/15 | 200/200/2400 | `{"controller": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "reader": {"cache_hits": 60, "crop_inputs": 840, "inference_seconds": 30.679402786074206, "model_forwards": 140, "model_load_seconds": 21.386539375060238, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M4_GAIN_S2 |
| regression_only | scene0445_00 | CP_M5_MIX50_NATIVE | 38.888889 | 50.000000 | 50.000000 | 46.004286 | 47.662628 | 3/16 | 16/15 | 200/0/1200 | `{"cache_hits": 176, "crop_inputs": 144, "inference_seconds": 5.060971725964919, "model_forwards": 24, "model_load_seconds": 1.861219516955316, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | CP_M5_MIX50_NATIVE |
| regression_only | scene0445_00 | N0 | 42.222222 | 60.000000 | 60.000000 | 51.705610 | 53.403678 | 0/16 | 16/15 | 211/0/1266 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| regression_only | scene0445_00 | Q_COMBINE | 42.222222 | 60.000000 | 60.000000 | 37.611862 | 44.877657 | 5/16 | 15/14 | 199/0/1194 | `{"cache_hits": 199, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}` | FORCED_TRAJECTORY | Q_COMBINE |
| regression_only | scene0445_00 | Q_GAIN | 38.888889 | 50.000000 | 52.000000 | 50.906770 | 52.858575 | 4/16 | 16/15 | 200/0/1200 | `{"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}` | FORCED_TRAJECTORY | Q_GAIN |
| regression_only | scene0445_00 | S_SIGLIP2_AREA | 38.888889 | 57.500000 | 57.500000 | 47.017123 | 53.553357 | 7/16 | 16/15 | 211/40/1506 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |
| regression_only | scene0626_00 | CP_M1_AGREE_KEEP | 16.077441 | 29.545455 | 43.181818 | 27.948487 | 38.284255 | 2/29 | 29/21 | 562/84/3876 | `{"Q_GAIN": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M1_AGREE_KEEP |
| regression_only | scene0626_00 | CP_M2_EQUAL_CAL | 25.168350 | 38.636364 | 50.000000 | 36.333988 | 45.483721 | 10/29 | 29/21 | 562/84/3876 | `{"Q_GAIN": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_CAL |
| regression_only | scene0626_00 | CP_M2_EQUAL_RAW | 18.350168 | 31.818182 | 45.454545 | 35.951993 | 47.313855 | 9/29 | 29/21 | 562/84/3876 | `{"Q_GAIN": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": true, "static_sources_reused": true}` | NEW_STATIC_LABEL_FUSION | CP_M2_EQUAL_RAW |
| regression_only | scene0626_00 | CP_M3_COMBINE_S2 | 14.393939 | 26.515152 | 35.606061 | 28.271473 | 40.990773 | 19/29 | 21/18 | 198/198/2376 | `{"controller": {"cache_hits": 198, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "reader": {"cache_hits": 14, "crop_inputs": 1104, "inference_seconds": 40.79048930981662, "model_forwards": 184, "model_load_seconds": 21.19097989902366, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M3_COMBINE_S2 |
| regression_only | scene0626_00 | CP_M4_GAIN_S2 | 25.168350 | 38.636364 | 47.727273 | 39.099217 | 50.082677 | 18/29 | 28/21 | 200/200/2400 | `{"controller": {"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}, "reader": {"cache_hits": 53, "crop_inputs": 882, "inference_seconds": 31.875438970630057, "model_forwards": 147, "model_load_seconds": 10.961402582004666, "model_loads": 1, "tiles": 0}, "shared_dependency_count_once_globally": true}` | FORCED_TRAJECTORY | CP_M4_GAIN_S2 |
| regression_only | scene0626_00 | CP_M5_MIX50_NATIVE | 16.077441 | 29.545455 | 40.909091 | 26.126778 | 36.658763 | 8/29 | 27/21 | 200/0/1200 | `{"cache_hits": 151, "crop_inputs": 294, "inference_seconds": 10.262783821905032, "model_forwards": 49, "model_load_seconds": 1.7885622319299728, "model_loads": 1, "tiles": 0}` | NEW_CAUSAL_CONTROLLER | CP_M5_MIX50_NATIVE |
| regression_only | scene0626_00 | N0 | 9.259259 | 22.727273 | 25.000000 | 18.963276 | 33.405230 | 0/29 | 29/21 | 362/0/2172 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | N0 |
| regression_only | scene0626_00 | Q_COMBINE | 13.131313 | 24.242424 | 35.606061 | 23.726057 | 32.452698 | 16/29 | 21/18 | 198/0/1188 | `{"cache_hits": 198, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}` | FORCED_TRAJECTORY | Q_COMBINE |
| regression_only | scene0626_00 | Q_GAIN | 25.168350 | 38.636364 | 50.000000 | 36.335854 | 45.481273 | 11/29 | 28/21 | 200/0/1200 | `{"cache_hits": 200, "crop_inputs": 0, "inference_seconds": 0.0, "model_forwards": 0, "model_load_seconds": 0.0, "model_loads": 0, "tiles": 0}` | FORCED_TRAJECTORY | Q_GAIN |
| regression_only | scene0626_00 | S_SIGLIP2_AREA | 16.077441 | 29.545455 | 43.181818 | 25.828997 | 37.122139 | 16/29 | 29/21 | 362/84/2676 | `{"crop_inputs": 0, "model_forwards": 0, "model_loads": 0, "reused_source": true}` | EXACT_FROZEN_SOURCE_PREDICTION | S_SIGLIP2_AREA |

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
| regression_only | N0 | 25.740741 (2/2) | 35.334443 (2/2) | COMPLETE |
| regression_only | Q_COMBINE | 27.676768 (2/2) | 30.668959 (2/2) | COMPLETE |
| regression_only | Q_GAIN | 32.028620 (2/2) | 43.621312 (2/2) | COMPLETE |
| regression_only | S_SIGLIP2_AREA | 27.483165 (2/2) | 36.423060 (2/2) | COMPLETE |
| regression_only | CP_M1_AGREE_KEEP | 29.149832 (2/2) | 39.827049 (2/2) | COMPLETE |
| regression_only | CP_M2_EQUAL_RAW | 28.619529 (2/2) | 40.978139 (2/2) | COMPLETE |
| regression_only | CP_M2_EQUAL_CAL | 35.917508 (2/2) | 47.475550 (2/2) | COMPLETE |
| regression_only | CP_M3_COMBINE_S2 | 21.641414 (2/2) | 32.910557 (2/2) | COMPLETE |
| regression_only | CP_M4_GAIN_S2 | 32.028620 (2/2) | 45.002994 (2/2) | COMPLETE |
| regression_only | CP_M5_MIX50_NATIVE | 27.483165 (2/2) | 36.065532 (2/2) | COMPLETE |
| confirmation | N0 | 10.902778 (2/2) | 17.511061 (2/2) | COMPLETE |
| confirmation | Q_COMBINE | 12.705761 (2/2) | 19.102524 (2/2) | COMPLETE |
| confirmation | Q_GAIN | 13.598618 (2/2) | 19.941575 (2/2) | COMPLETE |
| confirmation | S_SIGLIP2_AREA | 11.291152 (2/2) | 24.221050 (2/2) | COMPLETE |
| confirmation | CP_M2_EQUAL_RAW | 16.458333 (2/2) | 25.119712 (2/2) | COMPLETE |

Missing required control/composition records: 0. Missing rows are not zero-valued measurements.
Missing frozen confirmation records: 0. Confirmation coverage is checked separately from the development experiment.

## Table B — complementarity and routing

| Scene | Status | Owners | Correctness categories | Source unavailable counts |
|---|---|---:|---|---|
| scene0056_00 | COMPLETE | 99 | {'SHAPE_UNMATCHED': 86, 'Q_ONLY_CORRECT': 1, 'ALL_CORRECT': 7, 'ALL_WRONG': 4, 'N0_Q_CORRECT': 1} | {'N0': 0, 'S_SIGLIP2_AREA': 2, 'Q_GAIN': 3} |
| scene0064_00 | COMPLETE | 63 | {'ALL_WRONG': 6, 'SHAPE_UNMATCHED': 51, 'ALL_CORRECT': 3, 'Q_ONLY_CORRECT': 2, 'S2_ONLY_CORRECT': 1} | {'N0': 0, 'S_SIGLIP2_AREA': 1, 'Q_GAIN': 4} |
| scene0445_00 | COMPLETE | 16 | {'ALL_CORRECT': 4, 'SOURCE_UNAVAILABLE': 1, 'SHAPE_UNMATCHED': 9, 'N0_Q_CORRECT': 1, 'S2_ONLY_CORRECT': 1} | {'N0': 0, 'S_SIGLIP2_AREA': 2, 'Q_GAIN': 0} |
| scene0534_00 | COMPLETE | 93 | {'SHAPE_UNMATCHED': 84, 'ALL_WRONG': 4, 'ALL_CORRECT': 4, 'S2_ONLY_CORRECT': 1} | {'N0': 0, 'S_SIGLIP2_AREA': 2, 'Q_GAIN': 10} |
| scene0553_00 | COMPLETE | 30 | {'SHAPE_UNMATCHED': 23, 'N0_Q_CORRECT': 1, 'ALL_WRONG': 3, 'S2_ONLY_CORRECT': 1, 'ALL_CORRECT': 2} | {'N0': 0, 'S_SIGLIP2_AREA': 1, 'Q_GAIN': 0} |
| scene0626_00 | COMPLETE | 29 | {'SHAPE_UNMATCHED': 20, 'Q_ONLY_CORRECT': 1, 'ALL_CORRECT': 6, 'ALL_WRONG': 2} | {'N0': 0, 'S_SIGLIP2_AREA': 1, 'Q_GAIN': 1} |

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
| regression_only | CP_M3_COMBINE_S2 − Q_COMBINE | -6.035354 | 2.241598 |
| regression_only | CP_M4_GAIN_S2 − Q_GAIN | 0.000000 | 1.381682 |
| regression_only | CP_M4_GAIN_S2 − CP_M3_COMBINE_S2 | 10.387205 | 12.092437 |
| regression_only | CP_M2_EQUAL_CAL − CP_M2_EQUAL_RAW | 7.297980 | 6.497411 |
| regression_only | CP_M5_MIX50_NATIVE − Q_COMBINE | -0.193603 | 5.396572 |
| regression_only | CP_M5_MIX50_NATIVE − Q_GAIN | -4.545455 | -7.555780 |
| regression_only | 2x2_INTERACTION − (M4-Q_GAIN)-(M3-Q_COMBINE) | 6.035354 | -0.859916 |

Lane preference/winner/fallback counts, request Jaccards, per-owner paid budgets and retained/dropped evidence are in `trajectory_diagnostics`. A changed trajectory alone is not a gain.

## Table D — frozen nomination, confirmation and operations

Nominee: **CP_M2_EQUAL_RAW**.
Frozen best single: **S_SIGLIP2_AREA**.
M6 CAL gate: `NOT_REQUIRED_BY_COMPOSITION_CAL_GATE`; checks: `{'M3_vs_COMBINE': False, 'M4_vs_GAIN': False, 'M5_vs_COMBINE': False}`. Exact finite CAL inputs and tolerance are preserved in `m6_gate`.
Experiment commit A: `7a9bd5a212e1f85619235c15a06b405eece2be39`.
Query physical totals: `{'model_loads': 18, 'model_forwards': 2319, 'crop_inputs': 13914, 'cache_hits': 2475, 'inference_seconds': 495.06005736708175, 'static_siglip2_forwards': 266, 'static_siglip2_crops': 1596}`; new native captures: 2.
Completed new frontend jobs: 2; processed frames: 400. Frontend and mapping elapsed times remain separate in the numerical ledger.
Native capture SigLIP forwards/crops are recorded separately per capture from the worker shutdown log and checked against captured requests; they are additional to query totals. Failed or missing worker counts are not estimated.
Historical static S2 reused requests: 668. CPU calibration: 9 completed scalar fits, 168 recorded optimizer evaluations; initial NLL evaluations are listed separately in the numerical ledger. Calibration elapsed time was not instrumented.

| Role | Nominee versus | Status | Worst ΔuAP / ΔmIoU (pp) | Every scene nonnegative |
|---|---|---|---:|---|
| compose_cal | N0 | MEAN_GAIN_WITH_SCENE_TRADEOFF | -0.190584 / 2.065517 | False |
| compose_cal | S_SIGLIP2_AREA | NO_MEAN_GAIN | -0.194932 / 0.480804 | False |
| regression_only | N0 | MEAN_GAIN_WITH_SCENE_TRADEOFF | -3.333333 / -5.701325 | False |
| regression_only | S_SIGLIP2_AREA | MEAN_GAIN_WITH_SCENE_TRADEOFF | 0.000000 / -1.012838 | False |
| confirmation | N0 | MEAN_GAIN_NO_OBSERVED_SCENE_LOSS | 5.555556 / 6.971207 | True |
| confirmation | S_SIGLIP2_AREA | MEAN_GAIN_WITH_SCENE_TRADEOFF | 2.926955 / -0.776525 | False |

Nominee versus N0: actual released matching changes at IoU 0.5. GT IDs identify instances; event counts are not additive AP contributions.

| Role | Scene | Newly matched GT IDs | Lost matched GT IDs | FP events N0 → nominee |
|---|---|---|---|---:|
| compose_cal | scene0056_00 | 2007 | — | 43 → 42 |
| compose_cal | scene0534_00 | 7000 | — | 57 → 56 |
| confirmation | scene0064_00 | 1000, 24001 | — | 31 → 29 |
| confirmation | scene0553_00 | 48000 | — | 13 → 12 |
| regression_only | scene0445_00 | — | 11000 | 7 → 8 |
| regression_only | scene0626_00 | 24000 | — | 14 → 13 |

Actual released TP/FN gains/losses and FP events at 0.5/0.75 are linked in `released_transitions`, separately from geometric class correctness. AP increments are not added across objects.
The observed M2_RAW/M2_CAL equal-uAP case is examined in [UNCHANGED_AP_DIAGNOSTICS.md](complementary_composition_v1/UNCHANGED_AP_DIAGNOSTICS.md), including actual eligibility, ignored predictions, class AP states and the tied-score FP responsible for its AP25 change.

Complete numerical tables: `/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/reports/b9d08fc432947f1ccc721a11ebd7c34829b89c7fd442ca42566a7af8dd97f0d2/tables.json`.
Publication receipt (written only after verified ordinary push): `/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/publication_receipt.json`.
