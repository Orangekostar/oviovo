# Runtime Parity Results

Cache-cold incremental recovery with model/common geometry/NQF resident; OS page cache uncontrolled.

Source binding: `aab44cdbe2fa6fb5aefcdf0296baa1daceb50491e122983de63069412c507f29`. Actual v2: `639edf0708a14d7b3100826effacbf34a8369457b08313c220ff6a57fb0d7bdd`. Reference commit: `7fad2c70eb5754417b1bc9a2e61f8d6782f19c35`.

Frozen nominee: `R2_ROI_EXACT`; recommended future execution: `R2_ROI_EXACT`. Deployment: `N0_UNCHANGED`.

New paired G1 speedup: 2.3576x; mean reduction: 57.584%. Status: `MATERIAL_REDUCTION`. This is not a significance test.

| Arm | n/N | TP50 | FP50 | AP (%) | mIoU (%) | Seconds/scene |
| --- | --- | --- | --- | --- | --- | --- |
| NONE | 0/53 | 0 | 0 | 11.74 | 29.69 | 0 |
| U2 | 9/53 | 3 | 2 | 12.39 | 30.17 | 5.96 |
| G1 | 42/53 | 3 | 3 | 12.39 | 30.27 | 18.85 |
| G3 | 42/53 | 3 | 4 | 12.10 | 30.18 | 20.65 |

## Supplementary S8

| Stage (seconds) | U2 control | G1 reference | G1 selected/retained |
| --- | --- | --- | --- |
| registry geometry | 0.677 | 2.426 | 2.060 |
| view search | 0.498 | 34.922 | 9.780 |
| rgb preprocess | 0.120 | 0.445 | 0.357 |
| recognition | 0.153 | 0.560 | 0.561 |
| export bookkeeping | 4.508 | 6.070 | 6.071 |
| other sync | 0.007 | 0.024 | 0.024 |
| complete incremental total | 5.962 | 44.448 | 18.853 |
| encoder | 0.132 | 0.476 | 0.476 |
| region pool head | 0.005 | 0.022 | 0.022 |

GPU event rows are non-additive annotations. Host stages include all required output writes.

Coverage: 16 pilot + 64 final cold calls; all 26 CPU view/cached-export checks passed. Accuracy inherits 172 rows/14 pools as `PREDICTION_IDENTICAL_PARENT_METRIC`.

Cold-scene image encodings: 412; normal pool/head calls: 256; fallback pool/head calls: 236. Shape-only warm-ups: 2 image inputs and 2 pool/head calls, counted separately. CF18 new GPU inference: 0.

GPU: {'compute_capability': '8.6', 'driver_version': '595.71.05', 'name': 'NVIDIA A40', 'uuid': 'GPU-41ba4cb6-edfe-ed30-c047-a19d9f23d0a8'}; CPU: Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz; PyTorch 2.4.0; CUDA 12.1; Open3D 0.19.0. Raycast threads=4, encoder batch=1; original precision/TF32/determinism unchanged.

U2 recomputes FC features for its genuine archived Native observation policy; G1/G3 reconstruct views/BVH from the full sequence in every call. Existing grouping, model residency and inference_mode are inherited. The common new evidence writer replaces legacy duplicate receipts for both R0 and candidates; historical timings are not the speedup denominator.

Two repeats use all valid observations, including slow runs. Per-call CSV, per-scene means, round means, repetition differences, memory peaks, ray/IO/transfer/output counters and all constituent measurement IDs are in the compact artifacts.

Unchanged accuracy is expected. Runtime optimization does not repair CF18's recovery tradeoff: A3 vs D2 remains approximately AP -0.08pp, mIoU +0.02pp. All scenes are exposed; CF18 comprises 18 captures from seven physical families. No online, end-to-end FPS, unseen generalization or global-SOTA claim follows.
