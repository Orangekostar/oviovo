# Room0 R6/T1 cached attribution results

Decision: **DIAGNOSIS_COMPLETE_OWNERSHIP_LIMITED**. Additional repair: **NO_REPAIR_JUSTIFIED**.
All 34 required scene/run/condition rows are COMPLETE: 14 main conditions plus 3 fixed-registry geometry bridge cells on each of two saved independent FP32 predictions. New model inference/training/mapping counts are zero. All numbers below are actual cached evaluations, not target values.

The core candidate banks contain 64 qualified OVI candidates plus 141/143 SpaCeFormer candidates, on 1,862,429 returned points. Native B0 and S1a readouts bind to the same source generation. Candidate IDs are run-local; repeat comparison uses GT IDs. Extra 3D pretraining is present; exclusion of these scenes from training is UNVERIFIED.

## 1. Paired performance

Percentages; P=fp32_primary, R=fp32_repeat. AP/AP50/AP25 are overlapping released class-aware AP. uAP is released AP on disjoint unique masks. cAP75 is the separately named class-agnostic canonical greedy IoU>=.75 diagnostic, not released AP75. Candidate/kept/eligible counts, mAcc, all per-class values, exact readout SHA, geometry, filters/ignored counts, inference counts and costs are in `artifacts/static_ovmap/t1_attribution_v1/analysis/paired_performance.json`.

| Run | Condition | Pool/kept/eligible | AP % | AP50 % | AP25 % | mIoU % | mAcc % | uAP % | cAP75 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P | AT_O_AREA | 205/64/51 | 20.644 | 39.323 | 40.990 | 36.556 | 41.980 | 20.644 | 16.842 |
| P | AT_NATIVE_U11 | 205/160/114 | 17.275 | 36.999 | 46.153 | 18.915 | 23.788 | 9.158 | 2.531 |
| P | AT_U00 | 205/205/154 | 23.153 | 50.356 | 55.144 | 16.054 | 20.974 | 6.005 | 1.630 |
| P | AT_ASSIGN_OVI_FILL | 205/160/114 | 20.516 | 41.166 | 50.319 | 36.570 | 42.006 | 20.644 | 15.726 |
| P | AT_RANK_CLASS_NORM | 205/160/114 | 20.516 | 41.166 | 50.319 | 21.877 | 27.272 | 12.399 | 5.059 |
| P | AT_U11 | 205/160/114 | 20.516 | 41.166 | 50.319 | 21.877 | 27.272 | 12.399 | 2.531 |
| P | AT_NMS_SF_FIRST | 205/160/114 | 18.257 | 39.465 | 48.988 | 21.920 | 26.860 | 11.813 | 2.067 |
| P | AT_U10 | 205/205/152 | 19.640 | 38.863 | 47.606 | 21.866 | 27.025 | 11.213 | 1.630 |
| P | AT_U01 | 205/160/114 | 21.131 | 46.953 | 52.275 | 20.985 | 25.741 | 11.241 | 2.531 |
| P | AT_NATIVE_O_AREA | 205/64/51 | 17.404 | 35.157 | 36.823 | 33.631 | 38.496 | 17.404 | 16.842 |
| P | AT_S_AREA | 205/141/103 | 12.656 | 31.303 | 37.518 | 15.873 | 20.777 | 5.816 | 2.044 |
| P | AT_S_RELEASED | 205/141/103 | 14.096 | 31.008 | 36.165 | 28.331 | 33.708 | 12.366 | 8.226 |
| P | AT_ASSIGN_RAW | 205/160/114 | 20.516 | 41.166 | 50.319 | 21.877 | 27.272 | 12.399 | 2.531 |
| P | AT_ASSIGN_CLASS_NORM | 205/160/114 | 20.516 | 41.166 | 50.319 | 35.115 | 41.246 | 19.142 | 4.145 |
| R | AT_O_AREA | 207/64/51 | 20.644 | 39.323 | 40.990 | 36.556 | 41.980 | 20.644 | 16.842 |
| R | AT_NATIVE_U11 | 207/163/114 | 16.569 | 32.019 | 40.829 | 15.548 | 18.533 | 5.112 | 2.391 |
| R | AT_U00 | 207/207/153 | 22.081 | 43.755 | 51.788 | 13.939 | 18.688 | 4.067 | 1.351 |
| R | AT_ASSIGN_OVI_FILL | 207/163/114 | 19.810 | 36.185 | 44.995 | 36.575 | 42.008 | 20.201 | 15.426 |
| R | AT_RANK_CLASS_NORM | 207/163/114 | 19.810 | 36.185 | 44.995 | 18.579 | 22.017 | 8.352 | 5.086 |
| R | AT_U11 | 207/163/114 | 19.810 | 36.185 | 44.995 | 18.579 | 22.017 | 8.352 | 2.391 |
| R | AT_NMS_SF_FIRST | 207/163/114 | 17.135 | 30.838 | 44.127 | 17.867 | 21.530 | 7.700 | 1.807 |
| R | AT_U10 | 207/207/151 | 18.391 | 31.278 | 42.109 | 18.063 | 21.752 | 7.527 | 1.351 |
| R | AT_U01 | 207/163/114 | 20.714 | 42.783 | 48.630 | 19.406 | 23.617 | 9.510 | 2.391 |
| R | AT_NATIVE_O_AREA | 207/64/51 | 17.404 | 35.157 | 36.823 | 33.631 | 38.496 | 17.404 | 16.842 |
| R | AT_S_AREA | 207/143/102 | 11.220 | 25.053 | 34.161 | 14.759 | 19.590 | 4.644 | 1.765 |
| R | AT_S_RELEASED | 207/143/102 | 13.053 | 27.883 | 35.123 | 28.216 | 33.517 | 11.903 | 7.982 |
| R | AT_ASSIGN_RAW | 207/163/114 | 19.810 | 36.185 | 44.995 | 18.579 | 22.017 | 8.352 | 2.391 |
| R | AT_ASSIGN_CLASS_NORM | 207/163/114 | 19.810 | 36.185 | 44.995 | 33.529 | 39.073 | 17.927 | 4.355 |
| P | AT_GEO_NATIVE | 61/61/51 | 18.672 | 38.490 | 40.990 | 36.218 | 41.645 | 18.672 | 14.654 |
| P | AT_GEO_TRANSFER | 61/61/51 | 20.567 | 39.323 | 40.990 | 36.556 | 41.980 | 20.567 | 14.770 |
| P | AT_GEO_AREA | 61/61/51 | 20.644 | 39.323 | 40.990 | 36.556 | 41.980 | 20.644 | 16.842 |
| R | AT_GEO_NATIVE | 61/61/51 | 18.672 | 38.490 | 40.990 | 36.218 | 41.645 | 18.672 | 14.654 |
| R | AT_GEO_TRANSFER | 61/61/51 | 20.567 | 39.323 | 40.990 | 36.556 | 41.980 | 20.567 | 14.770 |
| R | AT_GEO_AREA | 61/61/51 | 20.644 | 39.323 | 40.990 | 36.556 | 41.980 | 20.644 | 16.842 |

The released runtime all-AP array has nine thresholds from 0.5 through approximately 0.9 (not 0.95), with AP25 separately appended. Match comparison is strict `>`; e.g. the runtime 0.75 entry is 0.7500000000000002. Scores are serialized to six decimals before parsing. Full float/hex thresholds, 51 semantic and 48 instance IDs, predicted min-region 100, no-GT/null/FN and void behavior are locked in `evaluation/metric_protocol.json`. cAP75 instead uses its own recorded 84-object canonical denominator and >= comparison; the released evaluable GT object denominator is 68.

The geometry bridge freezes the historical 61-candidate emitted registry and full-precision original scores reconstructed from its exact saved masks; serialized scores match the original manifest. Native historical full-semantic mIoU is 36.21754699025487%, while the fixed emitted-registry value is 36.21760223426797%: 72 positive semantic vertices lie outside that emitted registry. Core eligibility includes 64 candidates. These effects are reported separately, never credited as better reconstruction or fusion. The historical T0 `native_area_control` used within-class normalized area assignment; raw `AT_S_AREA` is intentionally different. Both released T0 rows and historical normalized T1 semantics match their receipts exactly.

## 2. Intervention effects

Differences in percentage points (after minus before). Four factorial contrasts and the nonadditive interaction are separate. Full AP50/AP25, unique metrics and readout interactions remain in `analysis/intervention_effects.json`.

| Run | Effect | Δ AP pp | Δ mIoU pp | Δ uAP pp |
| --- | --- | --- | --- | --- |
| P | candidate_addition | 2.509 | -20.503 | -14.640 |
| P | borrow_without_nms | -3.513 | 5.812 | 5.208 |
| P | borrow_with_nms | -0.615 | 0.893 | 1.157 |
| P | nms_without_borrow | -2.022 | 4.931 | 5.237 |
| P | nms_with_borrow | 0.876 | 0.012 | 1.186 |
| P | fusion_increment_S1a | -0.128 | -14.679 | -8.246 |
| P | assignment_class_norm | 0.000 | 13.238 | 6.744 |
| P | assignment_ovi_fill | 0.000 | 14.692 | 8.246 |
| P | rank_class_norm | 0.000 | 0.000 | 0.000 |
| P | nms_order | -2.259 | 0.043 | -0.586 |
| P | readout_native_O | 3.241 | 2.925 | 3.241 |
| P | readout_native_U11 | 3.241 | 2.963 | 3.241 |
| P | fusion_increment_native | -0.128 | -14.717 | -8.246 |
| P | factorial_interaction | 2.898 | -4.919 | -4.051 |
| P | readout_fusion_interaction | -0.000 | 0.038 | 0.000 |
| R | candidate_addition | 1.437 | -22.617 | -16.577 |
| R | borrow_without_nms | -3.690 | 4.124 | 3.459 |
| R | borrow_with_nms | -0.904 | -0.827 | -1.157 |
| R | nms_without_borrow | -1.366 | 5.467 | 5.442 |
| R | nms_with_borrow | 1.419 | 0.516 | 0.826 |
| R | fusion_increment_S1a | -0.834 | -17.977 | -12.292 |
| R | assignment_class_norm | 0.000 | 14.950 | 9.575 |
| R | assignment_ovi_fill | 0.000 | 17.996 | 11.848 |
| R | rank_class_norm | 0.000 | 0.000 | 0.000 |
| R | nms_order | -2.675 | -0.712 | -0.652 |
| R | readout_native_O | 3.241 | 2.925 | 3.241 |
| R | readout_native_U11 | 3.241 | 3.031 | 3.241 |
| R | fusion_increment_native | -0.834 | -18.083 | -12.292 |
| R | factorial_interaction | 2.785 | -4.951 | -4.617 |
| R | readout_fusion_interaction | 0.000 | 0.106 | 0.000 |
| P | geometry_transfer | 1.895 | 0.339 | 1.895 |
| P | geometry_rank_area | 0.077 | 0.000 | 0.077 |
| P | core_registry_eligibility | 0.000 | -0.000 | 0.000 |
| R | geometry_transfer | 1.895 | 0.339 | 1.895 |
| R | geometry_rank_area | 0.077 | 0.000 | 0.077 |
| R | core_registry_eligibility | 0.000 | -0.000 | 0.000 |

U00 improves overlapping candidate AP over O-only in both runs, while its unique map deteriorates strongly. Borrowing and NMS have conditional, nonadditive effects. Final U11 loses AP and unique AP against paired common-domain O-only in both runs; the apparent historical native-S1a→T1 gain mostly precedes fusion through geometry transfer/export and score changes. Native-vs-S1a gains also occur in O-only and must not be credited to fusion.

OVI_FILL recovers mIoU to 36.570/36.575%, but unique AP is 20.644/20.201% versus O-only 20.644% in both runs. Hence semantic preservation does not establish repeat-consistent unique-instance improvement. Rank-only normalization leaves released AP unchanged in these scene-local runs and leaves owner/semantic arrays exactly identical; its canonical class-agnostic score ordering can change cAP75. This is not a claim of pooled multi-scene AP invariance.

## 3. Object gains and losses

Below are actual released matched-GT set transitions at strict IoU>0.5, by intervention and representation. Gain/loss lists are GT IDs, not model queries; surviving counts refer to each overlapping gain that remains matched after unique ownership. Empty lists mean no set change, not equal precision or AP. All .25/.5/runtime-.75 events, exact first-match and duplicate score-owner evidence, borrowing, suppression, extent and TP rank tie intervals are in `analysis/object_gains_losses.json` and `analysis/object_diagnostics.json.gz`.

| Run | Effect | Representation | Gained GT IDs | Lost GT IDs | Gains surviving unique |
| --- | --- | --- | --- | --- | --- |
| P | candidate_addition | overlapping | 5000,5005,10004,15000,17000,24000,33000 | — | 2 |
| P | candidate_addition | unique | 5000,17000 | 4000,4001,5004,6001,7002,9006,10005,19002,19005,19006,21001,23001,25000,39006,48000,51000 | n/a |
| P | borrow_without_nms | overlapping | — | 10004,15000,17000,33000 | 0 |
| P | borrow_without_nms | unique | 9006,19002,25000,48000 | 17000 | n/a |
| P | borrow_with_nms | overlapping | — | 17000 | 0 |
| P | borrow_with_nms | unique | 48000 | 17000 | n/a |
| P | nms_without_borrow | overlapping | — | 10004,15000,33000 | 0 |
| P | nms_without_borrow | unique | 7002,9006,19002,25000 | — | n/a |
| P | nms_with_borrow | overlapping | — | — | 0 |
| P | nms_with_borrow | unique | 7002 | — | n/a |
| P | fusion_increment_S1a | overlapping | 5000,5005,24000 | — | 1 |
| P | fusion_increment_S1a | unique | 5000 | 4000,4001,5004,6001,10005,19005,19006,21001,23001,39006,51000 | n/a |
| P | assignment_class_norm | overlapping | — | — | 0 |
| P | assignment_class_norm | unique | 4000,4001,6001,10005,19005,19006,24000,39006,51000 | 5000 | n/a |
| P | assignment_ovi_fill | overlapping | — | — | 0 |
| P | assignment_ovi_fill | unique | 4000,4001,5004,6001,10005,19005,19006,21001,23001,39006,51000 | 5000 | n/a |
| P | nms_order | overlapping | — | 7002 | 0 |
| P | nms_order | unique | — | 7002 | n/a |
| R | candidate_addition | overlapping | 5000,5005,10004,15000,17000,33000 | — | 2 |
| R | candidate_addition | unique | 5000,5006,17000 | 4000,4001,5004,6000,6001,7002,9006,10000,10005,19002,19005,19006,21000,21001,23000,23001,25000,39005,39006,48000,51000 | n/a |
| R | borrow_without_nms | overlapping | — | 10004,15000,17000,33000 | 0 |
| R | borrow_without_nms | unique | 9006,19002,25000 | 17000 | n/a |
| R | borrow_with_nms | overlapping | — | 17000 | 0 |
| R | borrow_with_nms | unique | — | 17000 | n/a |
| R | nms_without_borrow | overlapping | — | 10004,15000,33000 | 0 |
| R | nms_without_borrow | unique | 7002,9006,19002,25000 | — | n/a |
| R | nms_with_borrow | overlapping | — | — | 0 |
| R | nms_with_borrow | unique | 7002 | — | n/a |
| R | fusion_increment_S1a | overlapping | 5000,5005 | — | 1 |
| R | fusion_increment_S1a | unique | 5000,5006 | 4000,4001,5004,6000,6001,10000,10005,19005,19006,21000,21001,23000,23001,39005,39006,48000,51000 | n/a |
| R | assignment_class_norm | overlapping | — | — | 0 |
| R | assignment_class_norm | unique | 4000,4001,10005,19005,19006,21000,23000,39005,39006,48000,51000 | 5000,5006 | n/a |
| R | assignment_ovi_fill | overlapping | — | — | 0 |
| R | assignment_ovi_fill | unique | 4000,4001,5004,6000,6001,10000,10005,19005,19006,21000,21001,23000,23001,39005,39006,48000,51000 | 5000,5006 | n/a |
| R | nms_order | overlapping | — | 4001,7002 | 0 |
| R | nms_order | unique | 23000 | 5006,7002 | n/a |

Borrowing classifications use geometry-only correspondence to exactly one evaluable GT at the stated strict threshold; ambiguous/unmatched cases remain explicit. They are not inferred from the final borrowed class. At 0.5:

- fp32_primary, all borrowed candidates: {'same_label': 32, 'wrong_to_right': 12, 'right_to_wrong': 5, 'unmatched': 10, 'wrong_to_wrong': 5}.
- fp32_primary, retained only: {'same_label': 9, 'right_to_wrong': 1, 'wrong_to_right': 2, 'unmatched': 5, 'wrong_to_wrong': 2}.
- fp32_repeat, all borrowed candidates: {'same_label': 33, 'wrong_to_right': 12, 'right_to_wrong': 5, 'unmatched': 11, 'wrong_to_wrong': 4}.
- fp32_repeat, retained only: {'same_label': 10, 'right_to_wrong': 1, 'wrong_to_right': 3, 'unmatched': 6, 'wrong_to_wrong': 1}.

Deterministic examples (selection rule archived before interpreting examples): recovered GT5000 (blinds) at .25 first matches candidate81, while duplicate score replacement ultimately credits candidate166; these identities are deliberately separate. GT4000 is the first loss example. SpaCeFormer query105 has a corrupted borrowed label and is suppressed by OVI owner3 in U01; for primary GT15000, candidate–GT IoU is .691534 and suppressor–GT IoU .774811, but the suppressor class is wrong, no other retained correct candidate covers it, and the actual released match is lost. The same mechanism appears in the repeat. The first same-class ownership example contains 219,335 primary valid-GT vertices: unchanged semantic class does not imply preserved instance extent. Full evidence cards are in `analysis/examples.json`; no favorable-only example selection or rendered-image claim is made.

Duplicate, background, poor localization, ignored and mixed-GT flags are linked to actual evaluator events and are explicitly nonexclusive. Do not sum them into an error partition. AP traces preserve released first-match, duplicate min/max confidence, hard FN and ignored-prediction behavior; no generic matcher substitutes for released events.

Repeat sensitivity is recorded by GT objects in `analysis/repeat_sensitivity.json`. At .5, primary-only U11 overlapping matches include GT24000; unique-map differences and higher-threshold disagreements are retained with both score-owner traces. The analysis uses neither a better-run selection nor confidence intervals.

## 4. Regional losses

Frozen pre-borrowing support × multiplicity × original-class agreement strata. The displayed primary candidate-addition and both assignment controls include every region, including unmatched and no-support. Full 26 paired comparisons and indexed integer confusion deltas are in `analysis/regional_losses.json` and `analysis/regional_confusion_deltas.npz`. C/W refer to semantic correctness on valid GT, owner IDs remain canonical+1, and prediction0 contributes FN.

| Effect | Region | Support/multiplicity/agreement | Valid GT | C→W | W→C | W→W | C→C | Owner Δ | Same-class owner Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_addition | -1 | PROJECTION_UNMATCHED/None/None | 111180 | 0 | 0 | 111180 | 0 | 0 | 0 |
| candidate_addition | 0 | neither/zero/none | 436 | 0 | 0 | 436 | 0 | 0 | 0 |
| candidate_addition | 13 | OVI-only/one/unanimous | 6366 | 0 | 0 | 3475 | 2891 | 0 | 0 |
| candidate_addition | 22 | SpaCeFormer-only/one/unanimous | 226 | 0 | 111 | 115 | 0 | 226 | 0 |
| candidate_addition | 25 | SpaCeFormer-only/multiple/unanimous | 412 | 0 | 248 | 164 | 0 | 412 | 0 |
| candidate_addition | 26 | SpaCeFormer-only/multiple/conflicting | 727 | 0 | 384 | 343 | 0 | 727 | 0 |
| candidate_addition | 34 | both/multiple/unanimous | 244558 | 0 | 0 | 22611 | 221947 | 219335 | 219335 |
| candidate_addition | 35 | both/multiple/conflicting | 582202 | 224190 | 12789 | 145996 | 199227 | 528155 | 168444 |
| assignment_class_norm | -1 | PROJECTION_UNMATCHED/None/None | 111180 | 0 | 0 | 111180 | 0 | 0 | 0 |
| assignment_class_norm | 0 | neither/zero/none | 436 | 0 | 0 | 436 | 0 | 0 | 0 |
| assignment_class_norm | 13 | OVI-only/one/unanimous | 6366 | 0 | 0 | 3475 | 2891 | 0 | 0 |
| assignment_class_norm | 22 | SpaCeFormer-only/one/unanimous | 226 | 0 | 0 | 212 | 14 | 0 | 0 |
| assignment_class_norm | 25 | SpaCeFormer-only/multiple/unanimous | 412 | 0 | 0 | 164 | 248 | 0 | 0 |
| assignment_class_norm | 26 | SpaCeFormer-only/multiple/conflicting | 727 | 102 | 30 | 430 | 165 | 196 | 0 |
| assignment_class_norm | 34 | both/multiple/unanimous | 244558 | 0 | 0 | 22113 | 222445 | 0 | 0 |
| assignment_class_norm | 35 | both/multiple/conflicting | 582202 | 52603 | 196225 | 161598 | 171776 | 316691 | 0 |
| assignment_ovi_fill | -1 | PROJECTION_UNMATCHED/None/None | 111180 | 0 | 0 | 111180 | 0 | 0 | 0 |
| assignment_ovi_fill | 0 | neither/zero/none | 436 | 0 | 0 | 436 | 0 | 0 | 0 |
| assignment_ovi_fill | 13 | OVI-only/one/unanimous | 6366 | 0 | 0 | 3475 | 2891 | 0 | 0 |
| assignment_ovi_fill | 22 | SpaCeFormer-only/one/unanimous | 226 | 0 | 0 | 212 | 14 | 0 | 0 |
| assignment_ovi_fill | 25 | SpaCeFormer-only/multiple/unanimous | 412 | 0 | 0 | 164 | 248 | 0 | 0 |
| assignment_ovi_fill | 26 | SpaCeFormer-only/multiple/conflicting | 727 | 0 | 0 | 460 | 267 | 0 | 0 |
| assignment_ovi_fill | 34 | both/multiple/unanimous | 244558 | 498 | 0 | 22113 | 221947 | 145183 | 144362 |
| assignment_ovi_fill | 35 | both/multiple/conflicting | 582202 | 8617 | 207655 | 150168 | 215762 | 409438 | 81876 |

All regional confusion matrices and deltas sum exactly to their corresponding global matrices. Projection-unmatched vertices are a distinct stratum: 111,427 total, 111,180 with valid semantic GT; they remain prediction0 across controls. Assignment-only owner changes occur only at retained-U11 multiplicity>=2; rank-only owner/class changes are zero. OVI_FILL fills 2,413/2,398 source points without a qualified OVI candidate, but 2,379/2,363 already have native geometry owners. Only 34/35 lack matched native geometry owners. “No qualified OVI semantic candidate” is not “unobserved physical space.”

## Costs, execution, and handoff

Measured timings are CPU warm-cache costs; no end-to-end FPS claim. `costs.json` separately records candidate/IoU preparation, checked projection reuse, label/NMS profiles, assignment including first writes, condition evaluation/export, bridge and analysis. Fusion-only profiles exclude preparation/assignment/evaluation: borrowing off+on took about 0.0010 s per run; NMS off+on 0.0172/0.0180 s. Cached projection load/write is not a fresh search. Per-condition costs include cache-hit flags; resume wall times exclude already completed cells. Intermediate runs are not extra model repeats.

Environment: `/home/ww/miniconda3/envs/ovimap-map/bin/python`, Python 3.11.15, NumPy 1.26.4, original receipt-bound OVI-MAP evaluator. Native mapping and model environments were not merged. Exact input and evaluator hashes are in prediction/evaluation inventories; final code hashes and paths are in `source_handoff.json`. The reviewed base is `8cefe6b4bca464e0478f82e6ef1e6d0e3783e347`; the task branch is `research/ovimap-t1-attribution-v1`.

Native-readout metadata was corrected after the first evaluation: all 28 condition masks/labels/kept/rank/assignment/owner arrays matched regenerated `predictions_verified` exactly. Only five provenance fields in evaluator ledgers were refreshed; old ledgers are preserved externally. Metrics were not rerun because evaluator inputs were unchanged. `evaluation/provenance_refresh.json` and `final_verification.json` document this. The final source-only empty-SF compatibility fix was red/green tested and affects none of the nonempty real banks; execution-time source hashes remain preserved.

Recorded script argv below exited 0 in this session; receipts use sys.argv and do not capture the interpreter. The verified evaluator environment is stated above. Run from the task worktree with an appropriate Python interpreter. Output directories already exist, so do not rerun generation into them. Initial core evaluation completed before the full resume; the resume command is the recorded final evaluator invocation. Provenance refresh and final asset verification were executed Python heredocs, not nonexistent standalone scripts.

```bash
scripts/evaluation/run_static_t1_attribution.py --config configs/evaluation/ovimap_t1_attribution_v1.json --output /mnt/shared/ww/ovimap-t1-attribution-v1/predictions
```

prediction_initial: exit 0.

```bash
scripts/evaluation/run_static_t1_attribution.py --config configs/evaluation/ovimap_t1_attribution_v1.json --output /mnt/shared/ww/ovimap-t1-attribution-v1/predictions_verified
```

prediction_metadata_corrected: exit 0.

```bash
scripts/evaluation/diagnose_static_t1_attribution.py --config configs/evaluation/ovimap_t1_attribution_v1.json --predictions /mnt/shared/ww/ovimap-t1-attribution-v1/predictions --output /mnt/shared/ww/ovimap-t1-attribution-v1/evaluation --resume
```

evaluation_full_resume: exit 0.

```bash
scripts/evaluation/evaluate_static_t1_geometry_bridge.py --config configs/evaluation/ovimap_t1_attribution_v1.json --predictions /mnt/shared/ww/ovimap-t1-attribution-v1/predictions --evaluation /mnt/shared/ww/ovimap-t1-attribution-v1/evaluation --output /mnt/shared/ww/ovimap-t1-attribution-v1/geometry_bridge
```

geometry_bridge: exit 0.

```bash
scripts/evaluation/summarize_static_t1_attribution.py --config configs/evaluation/ovimap_t1_attribution_v1.json --predictions /mnt/shared/ww/ovimap-t1-attribution-v1/predictions_verified --evaluation /mnt/shared/ww/ovimap-t1-attribution-v1/evaluation --geometry-bridge /mnt/shared/ww/ovimap-t1-attribution-v1/geometry_bridge --output /mnt/shared/ww/ovimap-t1-attribution-v1/analysis_final
```

analysis_final: exit 0.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/evaluation/test_static_t1_attribution.py tests/evaluation/test_static_t1_evaluator_trace.py tests/evaluation/test_static_t1_object_attribution.py tests/evaluation/test_static_t1_geometry_bridge.py tests/evaluation/test_static_proposal_fusion.py tests/evaluation/test_static_projected_masks.py tests/evaluation/test_static_projected_instance_metrics.py tests/evaluation/test_static_released_loader.py
```

final_regression: exit 0.

Final regression: **26 passed**. Actual legacy fusion arrays/ledger, both T0 released metrics, normalized T1 semantics, native same-generation semantics, all observed/unobserved released AP and PR/FN, assignment manifest bytes, rank-only map invariance, projection commutation and regional accounting pass. See `T1_ATTRIBUTION_AUDIT.md` for spec §§1–10 review.

Small deliverables: `artifacts/static_ovmap/t1_attribution_v1/` (four tables, evidence cards, protocol, counts, ledgers, decisions, commands, costs and source receipts). Large prediction masks, owners, projections and complete evaluator traces remain at `/mnt/shared/ww/ovimap-t1-attribution-v1/`; original inputs remain at their receipt-bound paths. No model weights, raw logits, masks, meshes, credentials or restricted datasets are included in the Git handoff.

No additional repair or oracle was run (optional NOT_RUN, not a blocked required condition). The diagnosis is complete, but Room0 development evidence cannot establish publication readiness, statistical significance or cross-scene generalization. One next experiment is predeclared: confirm O_AREA/U11/OVI_FILL on Room1 and Office1 with unchanged parameters and both saved seeds where available, using the complete current metric set and GT-object accounting; pool released evaluation across scenes and semantic confusion matrices. Acquire missing inputs only under that future task; launch no new mapping or inference here. Do not tune on those confirmation scenes.
