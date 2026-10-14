# FG2 branch utility Wave2 measured results

Execution: **COMPLETE**. Science: **NO_BOTH_SEED_PROMOTION_SIGNAL**. Deployment: **N0_UNCHANGED**. New controller updates: **8000**. New RGB/SAM: **0/0**.

The task tests branch admission over preselected cached view banks, not online camera acquisition. Frozen FC/text/FG2 weights, parent splits and scene order are retained.

Table1. Exact parent support and pooled recognition baselines (%). Full200 competition; TRAIN/DEV base-positive originals only.

| group | originals | families | categories | A_percent | M_percent | C_percent |
| --- | --- | --- | --- | --- | --- | --- |
| dev:FC_BASE:0:p2 | 69 | 4 | 28 | 36.092172 | 48.796898 | 35.877525 |
| dev:FC_BASE:0:p4 | 69 | 4 | 28 | 33.310185 | 46.709055 | 34.229798 |
| dev:FC_BASE:0:p8 | 69 | 4 | 28 | 37.288510 | 50.612374 | 42.058081 |
| dev:FG2:17:p2 | 69 | 4 | 28 | 44.722222 | 57.050866 | 42.960859 |
| dev:FG2:17:p4 | 69 | 4 | 28 | 38.703704 | 50.652056 | 35.637626 |
| dev:FG2:17:p8 | 69 | 4 | 28 | 46.071128 | 58.866342 | 48.011364 |
| dev:FG2:29:p2 | 69 | 4 | 28 | 44.884259 | 57.943723 | 44.210859 |
| dev:FG2:29:p4 | 69 | 4 | 28 | 39.773779 | 52.338564 | 38.535354 |
| dev:FG2:29:p8 | 69 | 4 | 28 | 46.890783 | 59.322691 | 48.516414 |
| train:FC_BASE:0:p2 | 622 | 24 | 74 | 26.164277 | 37.716360 | 27.258807 |
| train:FC_BASE:0:p4 | 622 | 24 | 74 | 27.165182 | 37.853569 | 27.493854 |
| train:FC_BASE:0:p8 | 622 | 24 | 74 | 28.051998 | 38.992957 | 28.645156 |
| train:FG2:17:p2 | 622 | 24 | 74 | 30.535988 | 44.426487 | 30.911697 |
| train:FG2:17:p4 | 622 | 24 | 74 | 31.282542 | 45.699520 | 31.088784 |
| train:FG2:17:p8 | 622 | 24 | 74 | 32.485936 | 48.238009 | 32.995672 |
| train:FG2:29:p2 | 622 | 24 | 74 | 29.617715 | 44.145974 | 30.238305 |
| train:FG2:29:p4 | 622 | 24 | 74 | 31.641550 | 46.252522 | 31.833054 |
| train:FG2:29:p8 | 622 | 24 | 74 | 32.195462 | 48.482381 | 32.308742 |

Table2. Full200 GT-margin software interventions. Base+local=total; interaction is reported separately, never added again. Conditions are repeated measurements, not independent originals.

| exact_support_aliases | families | group | mean_base | mean_interaction | mean_local | mean_total | originals | records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 4 | dev:s17:2-4 | 0.000059 | -0.000448 | -0.000029 | 0.000030 | 69 | 276 |
| 44 | 4 | dev:s17:4-8 | 0.000056 | -0.000191 | 0.000064 | 0.000120 | 69 | 276 |
| 8 | 4 | dev:s29:2-4 | -0.000007 | -0.000500 | -0.000193 | -0.000200 | 69 | 276 |
| 44 | 4 | dev:s29:4-8 | 0.000060 | -0.000173 | 0.000175 | 0.000235 | 69 | 276 |
| 80 | 24 | train:s17:2-4 | 0.000856 | -0.000829 | 0.000537 | 0.001393 | 622 | 2488 |
| 292 | 24 | train:s17:4-8 | -0.000082 | -0.000247 | 0.000000 | -0.000082 | 622 | 2488 |
| 80 | 24 | train:s29:2-4 | 0.000930 | -0.000771 | 0.000535 | 0.001466 | 622 | 2488 |
| 292 | 24 | train:s29:4-8 | -0.000132 | -0.000221 | -0.000046 | -0.000178 | 622 | 2488 |

Table3. TRAIN prefix8 only, original deduplication over4 conditions. Positive≥.01; negative≤−.01; same-view branch preference>coupled/KEEP+.01. GT-best is an offline TRAIN diagnostic, not a method or AP bound.

| seed | criterion | originals | families | minimum_originals | minimum_families | seed_passed | both_seed_training_activated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | positive | 602 | 24 | 40 | 8 | True | True |
| 17 | negative | 602 | 24 | 40 | 8 | True | True |
| 17 | branch_preference | 436 | 24 | 20 | 6 | True | True |
| 29 | positive | 602 | 24 | 40 | 8 | True | True |
| 29 | negative | 602 | 24 | 40 | 8 | True | True |
| 29 | branch_preference | 430 | 24 | 20 | 6 | True | True |

Table4. Seed17 finite DEV selection and seed29 same-profile repeat. Gate: ∆A8≥+.5pp, ∆M8/∆C8/∆A4≥−.5pp, ordinary net corrections8>0, both seeds. No seed29 reselection.

| seed | method | base | gamma | step | tau | A8_percent | M8_percent | C8_percent | A4_percent | mean_edits | net_corrections8 | delta_A8_pp | delta_M8_pp | delta_C8_pp | delta_A4_pp | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | COUPLED | — | — | 1000 | 0.000000 | 46.345749 | 59.114358 | 48.011364 | 43.610059 | 0.847826 | 0 | 0.274621 | 0.248016 | 0.000000 | 4.906355 | False |
| 17 | NO_NEW_CORRESPONDENCE | — | — | 500 | 0.100000 | 46.071128 | 58.866342 | 48.011364 | 39.166667 | 0.030797 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.462963 | False |
| 17 | STATIC | MASK_AREA | 0.500000 | 0 | 0.000000 | 46.507786 | 58.102453 | 46.483586 | 42.471591 | 0.000000 | 3 | 0.436658 | -0.763889 | -1.527778 | 3.767887 | False |
| 17 | WITH_NEW_CORRESPONDENCE | — | — | 1000 | 0.010000 | 46.345749 | 59.114358 | 48.011364 | 43.818392 | 0.664855 | 0 | 0.274621 | 0.248016 | 0.000000 | 5.114689 | False |
| 29 | COUPLED | — | — | 1000 | 0.000000 | 45.692340 | 57.993326 | 48.011364 | 42.281145 | 0.706522 | -3 | -1.198443 | -1.329365 | -0.505051 | 2.507365 | False |
| 29 | NO_NEW_CORRESPONDENCE | — | — | 500 | 0.100000 | 46.890783 | 59.322691 | 48.516414 | 39.773779 | 0.019928 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |
| 29 | STATIC | MASK_AREA | 0.500000 | 0 | 0.000000 | 46.800295 | 57.527056 | 49.261364 | 41.537247 | 0.000000 | 1 | -0.090488 | -1.795635 | 0.744949 | 1.763468 | False |
| 29 | WITH_NEW_CORRESPONDENCE | — | — | 1000 | 0.010000 | 45.692340 | 57.993326 | 48.011364 | 42.744108 | 0.617754 | -3 | -1.198443 | -1.329365 | -0.505051 | 2.970328 | False |

Table5. Official ordered scene-pool metrics (%), or exact fixed activation reason. No averaging scene AP; absent classes remain null. Fixed G1 geometry and original N/Q/F ranking path.

| status | reason | C_STAR | both_seed_passes | new_map_rows |
| --- | --- | --- | --- | --- |
| NOT_ACTIVATED_BY_FIXED_RULE | WITH_NEW_CORRESPONDENCE_OR_LOCKED_C_STAR_DID_NOT_PASS_BOTH_FIXED_SEED_GATES | STATIC | {"COUPLED": false, "NO_NEW_CORRESPONDENCE": false, "STATIC": false, "WITH_NEW_CORRESPONDENCE": false} | 0 |

Table6. Measured cached inference/training boundaries include all enumerated subset heads, phi and transfers. Native-head cap85000; cache≤32GiB; conditional optimizer budget8000. These are not RGB stream30fps or skipped-frame measurements; nested timers are not additive.

| phase | status | boundary | wall_seconds | native_head_attempts | phi_calls | phi_vectors | logical_action_candidates | phi_seconds | optimizer_updates | RGB_encodings | host_peak_RSS_GiB | cache_budget_GiB | peak_allocated_GiB | peak_reserved_GiB | online_FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | COMPLETE | CACHED_FP32_GRIDS_INCLUDES_MODEL_LOAD_LOCK_WAIT_ALL_SUBSET_HEADS | 1221.838656 | 45700 | 68460 | 3737586 | 192976 | 18.236374 | 0 | 0 | 10.740982 | 32 | 1.519604 | 1.640625 | — |
| baseline | FAILED | CACHED_FP32_GRIDS_INCLUDES_MODEL_LOAD_LOCK_WAIT_ALL_SUBSET_HEADS | 934.449457 | 28321 | 41775 | 2302925 | — | 16.012478 | 0 | 0 | 8.673496 | 32 | 1.519527 | 1.646484 | — |
| repeat | COMPLETE | CACHED24_FEATURES_ONLY_MLP_NO_FC_OR_FG2 | 24.244959 | 0 | 0 | 0 | — | 0.000000 | 0 | 0 | 0.719543 | 32 | 0.011310 | 0.023438 | — |
| select | COMPLETE | CACHED24_FEATURES_ONLY_MLP_NO_FC_OR_FG2 | 216.558670 | 0 | 0 | 0 | — | 0.000000 | 0 | 0 | 0.729496 | 32 | 0.011310 | 0.023438 | — |
| train | COMPLETE | CACHED24_FEATURES_ONLY_MLP_NO_FC_OR_FG2 | 109.020778 | 0 | 0 | 0 | — | 0.000000 | 8000 | 0 | 1.640736 | 32 | 0.030869 | 0.041016 | — |
| preflight | OBSERVED_PHASE_EXIT | OBSERVED_PHASE_SUBPROCESS_WALL_INCLUDES_STARTUP_CACHED_IO | 3.585625 | 0 | 0 | — | — | — | 0 | 0 | — | 32 | — | — | — |
| factorial | OBSERVED_PHASE_EXIT | OBSERVED_PHASE_SUBPROCESS_WALL_INCLUDES_STARTUP_CACHED_IO | 304.111256 | 0 | 0 | — | — | — | 0 | 0 | — | 32 | — | — | — |
| action_bank | OBSERVED_PHASE_EXIT | OBSERVED_PHASE_SUBPROCESS_WALL_INCLUDES_STARTUP_CACHED_IO | 7.758927 | 0 | 0 | — | — | — | 0 | 0 | — | 32 | — | — | — |
| training_gate | OBSERVED_PHASE_EXIT | OBSERVED_PHASE_SUBPROCESS_WALL_INCLUDES_STARTUP_CACHED_IO | 8.734466 | 0 | 0 | — | — | — | 0 | 0 | — | 32 | — | — | — |
| map_predict | OBSERVED_PHASE_EXIT | OBSERVED_PHASE_SUBPROCESS_WALL_INCLUDES_STARTUP_CACHED_IO | 3.386245 | — | 0 | — | — | — | 0 | 0 | — | 32 | — | — | — |
| map_evaluate | OBSERVED_PHASE_EXIT | OBSERVED_PHASE_SUBPROCESS_WALL_INCLUDES_STARTUP_CACHED_IO | 1.982577 | 0 | 0 | — | — | — | 0 | 0 | — | 32 | — | — | — |


All finite candidates, family/condition strata and fixed examples are supplied in tables/. All action utilities, labels, margins and24 features are in lossless action chunks; raw embeddings/full cosines remain in OUT with bytes/hash/regeneration command. No local group U banks are persisted.

WITH_NEW_CORRESPONDENCE improves A4 by 5.11/2.97pp (seeds17/29), but A8 changes by 0.27/-1.20pp. A8 matches the coupled control at both seeds; branch independence is not established.

No profile satisfied the specified promotion route across both seeds. New map scoring is therefore intentionally not activated. Existing26-scene conveyor analysis remains separate.
