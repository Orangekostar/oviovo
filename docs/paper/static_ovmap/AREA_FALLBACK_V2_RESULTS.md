# Area Fallback V2 Results

Protocol: `PROJECTED_FC_EMPTY_AREA_FALLBACK_V2`.

Post-hoc diagnostic on already exposed scenes. Only empty projected G1/G3 FC masks use area-weighted pooling. The incumbent N/Q/F evidence, archived U2, fixed views, full-resolution masks, geometry, vocabulary, frozen head and released scoring are unchanged.

Coverage: 172/172 scene-method scores; 14/14 complete official pools. 74/74 empty masks repaired; 175 original successful vectors reused bit-for-bit. Accuracy probe: new GPU inference 0; new full-image encoding 0; CPU projection-head calls 74.

Methods: A0 = Native; A1 = D2 refinement; A2 = Native + repaired G1-FC recovery; A3 = D2 + repaired G1-FC recovery; A4 = D2 + original G1-Native recovery; A5 = existing FC-only readout + repaired G1-FC recovery; U2 = unchanged archived recovery; G3 = D2 + repaired three-view FC recovery.

| Cohort | Method | APall (%) | AP50 (%) | AP25 (%) | mIoU (%) | mAcc (%) |
| --- | --- | --- | --- | --- | --- | --- |
| replica8 | A0 | 8.63 | 21.48 | 34.59 | 27.24 | 32.67 |
| replica8 | A1 | 11.74 | 24.50 | 37.97 | 29.69 | 37.43 |
| replica8 | A2 | 9.44 | 23.69 | 36.93 | 27.96 | 33.65 |
| replica8 | A3 | 12.39 | 26.06 | 39.75 | 30.27 | 38.40 |
| replica8 | A4 | 11.74 | 24.50 | 38.15 | 29.82 | 37.86 |
| replica8 | A5 | 10.94 | 21.88 | 32.29 | 23.72 | 33.28 |
| replica8 | U2 | 12.39 | 26.06 | 39.84 | 30.17 | 38.15 |
| replica8 | G3 | 12.10 | 25.54 | 39.23 | 30.18 | 38.47 |
| scannet_cf18 | A0 | 7.18 | 15.72 | 21.21 | 16.38 | 25.36 |
| scannet_cf18 | A1 | 8.31 | 17.96 | 24.98 | 19.00 | 28.38 |
| scannet_cf18 | A2 | 7.11 | 15.61 | 21.09 | 16.44 | 25.47 |
| scannet_cf18 | A3 | 8.23 | 17.83 | 24.76 | 19.02 | 28.49 |
| scannet_cf18 | A4 | 8.30 | 17.93 | 24.92 | 18.99 | 28.44 |
| scannet_cf18 | A5 | 8.32 | 17.39 | 23.76 | 17.94 | 24.02 |

## A3 Compared With A1

| Cohort | APall (pp) | AP50 (pp) | AP25 (pp) | mIoU (pp) | mAcc (pp) |
| --- | --- | --- | --- | --- | --- |
| replica8 | +0.65 | +1.56 | +1.78 | +0.58 | +0.98 |
| scannet_cf18 | -0.08 | -0.13 | -0.23 | +0.02 | +0.11 |

## V1 To V2 On The Identical Complete Cohort

Replica v1 recovery pools were unavailable because office1 was blocked; no v1-to-v2 full-Replica gain can be claimed. The following CF18 changes use the same full 18-scene cohort.

| Method | APall (pp) | AP50 (pp) | AP25 (pp) | mIoU (pp) | mAcc (pp) |
| --- | --- | --- | --- | --- | --- |
| A2 | -0.0019 | -0.0075 | -0.0148 | +0.0119 | +0.0149 |
| A3 | -0.0004 | -0.0034 | -0.0144 | -0.0037 | +0.0149 |
| A5 | -0.0160 | -0.0711 | -0.1227 | -0.0336 | +0.0149 |

## Replica Recovery

| Arm | Source-available n/N | Added TP50 | Added FP50 | Ambiguous TP/FP | Cold Seconds/Scene |
| --- | --- | --- | --- | --- | --- |
| NONE | 0/53 | 0 | 0 | 0/0 | 0 (by definition) |
| U2 | 9/53 | 3 | 2 | 0/0 | 6.61 (unchanged control) |
| G1 | 42/53 | 3 | 3 | 0/0 | 45.58 |
| G3 | 42/53 | 3 | 4 | 0/0 | 45.63 |

## V2 Cold Remeasurement

| Scene | G1 (s) | G3 (s) | Discrete Output Parity |
| --- | --- | --- | --- |
| office0 | 39.828 | 39.698 | exact / exact |
| office1 | 36.475 | 34.753 | exact / exact |
| office2 | 45.891 | 46.480 | exact / exact |
| office3 | 53.151 | 55.181 | exact / exact |
| office4 | 44.768 | 44.779 | exact / exact |
| room0 | 53.067 | 52.438 | exact / exact |
| room1 | 43.875 | 42.826 | exact / exact |
| room2 | 47.614 | 48.909 | exact / exact |

16/16 serial calls completed on NVIDIA A40 (GPU-41ba4cb6-edfe-ed30-c047-a19d9f23d0a8). New GPU full-image encodings: 128; GPU region-pool/head calls: 159; area-fallback poolings: 74. Model loading: 7.966s, excluded from the means.

Each mean is the sum of all eight scene costs divided by eight (one sample per scene/arm). Timing includes fresh projection, RGB decode/preprocessing, image encoding, pooling/head, classification and output/rank export. Model loading, geometry mapping and benchmark evaluation are excluded. Feature, view and result caches are disabled; same-frame sharing is allowed only within one call. Feature-cache-cold does not mean the OS page cache was cleared.

All fixed masks, view selections, source labels/availability, full instance support, semantic labels and current-class ranks match the v2 scientific outputs. Maximum FP32 feature absolute difference: 1.1920929e-07; inherited tolerance: atol=rtol=1e-5. Accuracy metrics are unchanged.

TP/FP are definite entries from the unchanged released matcher, not unique recovered objects. Source availability is not correct detection. Semantic unknown errors remain in whole-scene confusion matrices.

## Artifacts

The three filled LaTeX tables, typed cell provenance, per-scene CSV, complete result store, official per-class pool receipts and preview PDF are under `tables/`. Full predictions, vectors and scoring traces remain in this independent attempt.

G1/G3 latency comes from the independent v2 GPU cold remeasurement; archived U2 retains its unchanged v1 control measurement. Deployment is unchanged.

Parent: `/mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001`. V2 attempt: `/home/ww/ovimap-area-fallback-v2/attempt_001`.
