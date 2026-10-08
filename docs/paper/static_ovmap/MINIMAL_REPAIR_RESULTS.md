# Minimal instance repair results

Actual coverage: 234/234 scene-method records, 18/18 released ordered pools. Fixed surface; one exclusive instance/semantic output. All cohorts were previously exposed.

Research recommendation: `IR01_G1`. TARGET_MET=False; MATERIAL_TARGET_MET=False. Deployment: N0_UNCHANGED.

## Table 1

| Method | R APall | R AP50 | R mIoU | CF APall | CF AP50 | CF mIoU | CF ΔAP G1 | CF ΔAP D2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR00 D2 | 11.74 | 24.50 | 29.69 | 8.31 | 17.96 | 19.00 | 0.08 | 0.00 |
| IR01 G1 | 12.39 | 26.06 | 30.27 | 8.23 | 17.83 | 19.02 | 0.00 | -0.08 |
| IR02 Nearest attach | 12.20 | 25.45 | 30.23 | 8.23 | 17.84 | 19.02 | 0.00 | -0.07 |
| IR03 Evidence attach | 12.39 | 26.06 | 30.27 | 8.23 | 17.83 | 19.02 | 0.00 | -0.08 |
| IR04 Direct group | 12.39 | 26.06 | 30.27 | 8.23 | 17.83 | 19.02 | 0.00 | -0.08 |
| IR05 Verified repair | 12.39 | 26.06 | 30.27 | 8.23 | 17.83 | 19.02 | 0.00 | -0.08 |
| IR06 AnyUp reread | 9.94 | 21.92 | 27.44 | 8.62 | 19.02 | 20.92 | 0.39 | 0.31 |
| IR07 Boundary stable | 10.63 | 23.29 | 28.28 | 8.53 | 18.72 | 20.48 | 0.30 | 0.22 |
| IR08 Combination | 10.63 | 23.29 | 28.28 | 8.53 | 18.72 | 20.48 | 0.30 | 0.22 |

Official dataset pools (%), fixed surface, one exclusive partition. Deltas are percentage points; all five metrics in source data.

## Table 2

| Pair | Cohort | ΔAPall | ΔAP50 | ΔmIoU | Ops A/B | Geom50 +/− | Geom75 +/− | GT50 +/− | GT75 +/− | W→R | R→W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR03−IR02 | Replica | 0.19 | 0.61 | 0.05 | 0/25 | 1/0 | 1/0 | 1/0 | 1/0 | 0 | 0 |
| IR05−IR04 | Replica | 0.00 | 0.00 | 0.00 | 1/1 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0 |
| IR07−IR06 | Replica | 0.69 | 1.37 | 0.84 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 1 | 0 |
| IR08−IR05 | Replica | -1.75 | -2.77 | -1.99 | 1/1 | 0/0 | 0/0 | 0/3 | 0/1 | 0 | 3 |
| IR08−IR07 | Replica | 0.00 | 0.00 | 0.00 | 1/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0 |
| IR03−IR02 | CF18 | -0.00 | -0.01 | -0.00 | 6/93 | 1/0 | 0/0 | 0/0 | 0/0 | 0 | 0 |
| IR05−IR04 | CF18 | 0.00 | 0.00 | -0.00 | 1/6 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0 |
| IR07−IR06 | CF18 | -0.09 | -0.30 | -0.44 | 0/0 | 0/0 | 0/0 | 0/1 | 0/0 | 0 | 1 |
| IR08−IR05 | CF18 | 0.30 | 0.88 | 1.45 | 1/1 | 0/0 | 0/0 | 5/0 | 2/0 | 5 | 0 |
| IR08−IR07 | CF18 | 0.00 | 0.00 | 0.00 | 1/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0 |

Prespecified pairs. Geom: class-agnostic matching; GT: released class-aware matching. W/R: geometrically matchable original P supports. Full thresholds and label changes in supplement.

## Table 3

| Method | R APall | R mIoU | CF APall | CF mIoU | s/scene | Alloc GiB | Reserv GiB | FC/call | QK/call | Regions/call | Obs/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR00 D2 | 11.74 | 29.69 | 8.31 | 19.00 | — | — | — | — | — | — | — |
| IR01 G1 | 12.39 | 30.27 | 8.23 | 19.02 | 16.23 | 1.89 | 2.21 | 4.50 | 0.00 | 0.00 | 0.00 |
| IR02 Nearest attach | 12.20 | 30.23 | 8.23 | 19.02 | — | — | — | — | — | — | — |
| IR03 Evidence attach | 12.39 | 30.27 | 8.23 | 19.02 | — | — | — | — | — | — | — |
| IR04 Direct group | 12.39 | 30.27 | 8.23 | 19.02 | — | — | — | — | — | — | — |
| IR05 Verified repair | 12.39 | 30.27 | 8.23 | 19.02 | 85.57 | 1.89 | 2.21 | 4.50 | 0.00 | 0.00 | 32.00 |
| IR06 AnyUp reread | 9.94 | 27.44 | 8.62 | 20.92 | — | — | — | — | — | — | — |
| IR07 Boundary stable | 10.63 | 28.28 | 8.53 | 20.48 | 81.10 | 4.40 | 6.65 | 20.12 | 16.75 | 80.50 | 32.00 |
| IR08 Combination | 10.63 | 28.28 | 8.53 | 20.48 | 106.15 | 4.40 | 6.65 | 20.12 | 16.75 | 80.50 | 32.00 |

Fresh paired cold calls, 8 Replica scenes × 2 rounds per measured arm. Required models resident; peaks are maxima, time is the mean. Unmeasured: —.


Hardware and model load: see [timing provenance](../../../artifacts/static_ovmap/minimal_instance_repair_v1/provenance/timing/summary.json). [OviMAP Table 8](https://arxiv.org/html/2603.26541v1#S8) reports ms per processed keyframe on RTX3090 + i7-12700K; this experiment measures seconds per scene on A40 + Xeon Silver 4314. Tables 7/8 do not report peak VRAM. Different hardware and pipeline boundaries prevent an external speedup ratio. Scene latency cannot be relabeled as per-frame skipped frames or validated online 30 FPS. The earlier module comparison remains available in [RESOURCE_COMPARISON.md](RESOURCE_COMPARISON.md).


Within this new series (16 calls per arm), Combination: wall time 6.542× G1, allocated peak 2.329× G1; Verified repair: wall time 5.273× G1, allocated peak 1.000× G1; Boundary stable: wall time 4.998× G1, allocated peak 2.329× G1. Reserved peaks describe allocator reservations, not model-only memory. Common Native/SigLIP/segmentation construction is outside this scene-repair boundary.
