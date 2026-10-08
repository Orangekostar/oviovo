# Source-preserving semantic update results

Execution: SCIENCE_COMPLETE; reporting: TABLES_COMPLETE. Scene-method rows 234/234; complete pools 18/18.

Selection: COMPLETE_NO_TARGET_GAIN; selected SU01_G1; passing candidates []; material target False.

Nine fixed conditions; matched controls explicitly marked. Released ordered pools, percent; gates use unrounded fractions.

| Method | Rule | Replica_apall_pct | Replica_ap50_pct | Replica_miou_pct | CF18_apall_pct | CF18_ap50_pct | CF18_miou_pct | Target_met | Material_met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SU00_D2 | Inherited D2 | 11.741 | 24.495 | 29.691 | 8.309 | 17.961 | 18.997 | — | — |
| SU01_G1 | Inherited G1 | 12.386 | 26.057 | 30.273 | 8.232 | 17.834 | 19.022 | — | — |
| SU02_HARD_MATCHED | Hard overwrite (matched) | 9.945 | 21.917 | 27.444 | 8.399 | 18.557 | 20.618 | no | no |
| SU03_STABLE_MATCHED | Stable overwrite (matched) | 10.634 | 23.289 | 28.281 | 8.308 | 18.243 | 20.151 | no | no |
| SU04_F_REPLACE | Replace F | 10.955 | 23.467 | 28.606 | 8.295 | 18.023 | 19.948 | no | no |
| SU05_F_BLEND | Blend F+A | 10.847 | 23.335 | 28.507 | 8.183 | 17.947 | 19.853 | no | no |
| SU06_F_COARSE | Blend F+C | 11.879 | 25.201 | 30.195 | 8.186 | 17.962 | 19.843 | no | no |
| SU07_GLOBAL_BLEND | Global same-A-mass blend | 10.534 | 23.102 | 28.378 | 8.379 | 18.526 | 20.498 | no | no |
| SU08_PAIRED_DELTA | Paired A-C delta | 11.375 | 24.123 | 28.642 | 8.292 | 18.034 | 19.140 | no | no |


Six fixed contrasts in both complete cohorts. Deltas in percentage points; fixed original P semantic outcomes at strict IoU > .50.

| Cohort | Contrast | Delta_apall_pp | Delta_ap50_pp | Delta_miou_pp | GT50_gained | GT50_lost | Wrong_to_right | Right_to_wrong | Output_identity_tie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replica8 | SU04_F_REPLACE - SU02_HARD_MATCHED | 1.010 | 1.549 | 1.162 | 1 | 0 | 1 | 0 | no |
| scannet_cf18 | SU04_F_REPLACE - SU02_HARD_MATCHED | -0.104 | -0.534 | -0.670 | 1 | 4 | 1 | 4 | no |
| replica8 | SU05_F_BLEND - SU04_F_REPLACE | -0.108 | -0.132 | -0.099 | 0 | 0 | 0 | 0 | no |
| scannet_cf18 | SU05_F_BLEND - SU04_F_REPLACE | -0.112 | -0.076 | -0.095 | 0 | 1 | 0 | 1 | no |
| replica8 | SU05_F_BLEND - SU06_F_COARSE | -1.032 | -1.866 | -1.688 | 0 | 2 | 0 | 2 | no |
| scannet_cf18 | SU05_F_BLEND - SU06_F_COARSE | -0.003 | -0.016 | 0.011 | 0 | 1 | 0 | 1 | no |
| replica8 | SU05_F_BLEND - SU07_GLOBAL_BLEND | 0.313 | 0.233 | 0.129 | 1 | 1 | 1 | 1 | no |
| scannet_cf18 | SU05_F_BLEND - SU07_GLOBAL_BLEND | -0.196 | -0.579 | -0.644 | 0 | 3 | 0 | 3 | no |
| replica8 | SU08_PAIRED_DELTA - SU05_F_BLEND | 0.528 | 0.788 | 0.135 | 2 | 0 | 2 | 0 | no |
| scannet_cf18 | SU08_PAIRED_DELTA - SU05_F_BLEND | 0.109 | 0.087 | -0.713 | 3 | 3 | 3 | 3 | no |
| replica8 | SU03_STABLE_MATCHED - SU02_HARD_MATCHED | 0.690 | 1.372 | 0.837 | 1 | 0 | 1 | 0 | no |
| scannet_cf18 | SU03_STABLE_MATCHED - SU02_HARD_MATCHED | -0.091 | -0.313 | -0.466 | 0 | 1 | 0 | 1 | no |


Common-domain and actual class-change coverage. CPU cost is conditional resident decision time on Replica8 only; shared acquisition is separate.

| Cohort | Method | Selected | Eligible | Applied_changes | Paired_FULL | CPU_ms_scene | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| replica8 | SU00_D2 | — | — | 0 | — | 0.003 | Inherited output; literal pass-through |
| scannet_cf18 | SU00_D2 | — | — | 0 | — | — | Inherited output; literal pass-through |
| replica8 | SU01_G1 | — | — | 0 | — | 0.003 | Inherited output; literal pass-through |
| scannet_cf18 | SU01_G1 | — | — | 0 | — | — | Inherited output; literal pass-through |
| replica8 | SU02_HARD_MATCHED | 128 | 111 | 62 | 218 | 2.628 | N/Q/F + all same-view A/C; historical class replay |
| scannet_cf18 | SU02_HARD_MATCHED | 288 | 209 | 127 | 366 | — | N/Q/F + all same-view A/C; historical class replay |
| replica8 | SU03_STABLE_MATCHED | 128 | 111 | 23 | 218 | 2.630 | N/Q/F + all same-view A/C; historical class replay |
| scannet_cf18 | SU03_STABLE_MATCHED | 288 | 209 | 26 | 366 | — | N/Q/F + all same-view A/C; historical class replay |
| replica8 | SU04_F_REPLACE | 128 | 111 | 34 | 218 | 5.632 | N/Q/F + A; C establishes shared availability |
| scannet_cf18 | SU04_F_REPLACE | 288 | 209 | 111 | 366 | — | N/Q/F + A; C establishes shared availability |
| replica8 | SU05_F_BLEND | 128 | 111 | 19 | 218 | 5.636 | N/Q/F + A; C establishes shared availability |
| scannet_cf18 | SU05_F_BLEND | 288 | 209 | 86 | 366 | — | N/Q/F + A; C establishes shared availability |
| replica8 | SU06_F_COARSE | 128 | 111 | 20 | 218 | 5.690 | N/Q/F + same-view C; A establishes shared availability |
| scannet_cf18 | SU06_F_COARSE | 288 | 209 | 86 | 366 | — | N/Q/F + same-view C; A establishes shared availability |
| replica8 | SU07_GLOBAL_BLEND | 128 | 111 | 32 | 218 | 5.614 | N/Q/F + A; C establishes shared availability |
| scannet_cf18 | SU07_GLOBAL_BLEND | 288 | 209 | 90 | 366 | — | N/Q/F + A; C establishes shared availability |
| replica8 | SU08_PAIRED_DELTA | 128 | 111 | 14 | 218 | 6.287 | N/Q/F + paired A-C |
| scannet_cf18 | SU08_PAIRED_DELTA | 288 | 209 | 51 | 366 | — | N/Q/F + paired A-C |


New successful FC image encodings: 0; successful coarse FULL pools: 584; unavailable coarse pools: 0; failed attempts: 0. Failed-attempt FC encodings: 0; failed-attempt encoder calls: 0; failed-attempt coarse pools: 0. AnyUp QK, new geometric projections, N/Q/frontend inference and end-to-end cold calls: 0. Coarse worker wall (including resume invocations): 85.93630635295995 s; model-load wall: 14.332648084964603 s. Model-load time is included in worker/first-leaf wall, not additive. A and historical FC sources are reused, not free at deployment. CPU timing excludes loading, GPU models, payload/ranks, evaluation and serialization.

Replica8 and CF18 were previously exposed. CF18 comprises 18 captures from seven physical families; this is neither untouched confirmation nor full ScanNet200 validation. No significance or monotone-accuracy claim is supported. Deployment: N0_UNCHANGED.

AP25/mAcc and exact gates: supplementary_metrics.csv/json. Full per-scene, GT50/75, tied TP/FP multiplicity, rank changes and class deltas are in the compact result/diagnostic bundles. Probability equality is separate from label/output equality.

Historical unrestricted IR06/IR07 context (not matched main-table results; percent):

| Cohort | Historical method | APall | AP50 | AP25 | mIoU | mAcc | Source pool identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| replica8 | IR06_ANYUP_REREAD | 9.945 | 21.917 | 36.173 | 27.444 | 35.887 | e8d5fa429bce8771ac9ca9691fc31a44fd526800193bec3c5aa936a24a4633d2 |
| replica8 | IR07_BOUNDARY_STABLE | 10.634 | 23.289 | 37.076 | 28.281 | 36.312 | 71e8d26634368695153b3b57032f1aeb256bcfa977634b3600302a781e873ff9 |
| scannet_cf18 | IR06_ANYUP_REREAD | 8.620 | 19.018 | 26.642 | 20.917 | 29.464 | e7a05aa9c0cac6ec0b21b6610701094e1700d209c39e0ce8cdc401e495272858 |
| scannet_cf18 | IR07_BOUNDARY_STABLE | 8.531 | 18.716 | 26.538 | 20.477 | 29.536 | 441e9a5087a6bc586546d18231126f9ed471c35a5299b9358ff7e3afdd3e1365 |
