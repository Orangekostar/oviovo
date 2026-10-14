
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
