# Local ownership results — two cached Room0 FP32 runs

**COMPLETE_NO_NET_UNIQUE_GAIN.** Six new cells COMPLETE; model inference/training/mapping counts remain zero. U00 masks, original classes, canonical IDs and raw source-area AP ranks are unchanged. S1a, geometry transfer and old candidate-union gains are fixed inputs, not new contributions. This concerns the static fusion/output layer only.

## Table A — paired performance

All metrics below are percentages. P/R are the original primary/repeat network predictions. AT_* rows are REUSED_REFERENCE, not rerun experiments; LO_* rows are NEW. cAP75 is the separate class-agnostic canonical >=.75 diagnostic, not released semantic AP75. The released evaluator uses its actual float thresholds (strict >; runtime .75 is .7500000000000002), 48 instance versus 51 semantic classes, and minimum predicted region100. Candidate masks are identical within the new U00 family; unique masks are evaluated separately with unchanged rank.

| Run | Condition | Role | Candidate AP | Unique AP | Unique AP50 | Unique AP25 | cAP75 | mIoU | mAcc | Unique eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P | AT_O_AREA | reused | 20.644 | 20.644 | 39.323 | 40.990 | 16.842 | 36.556 | 41.980 | 51 |
| P | AT_U00 | reused | 23.153 | 6.005 | 16.468 | 18.545 | 1.630 | 16.054 | 20.974 | 63 |
| P | AT_U11 | reused | 20.516 | 12.399 | 21.005 | 27.406 | 2.531 | 21.877 | 27.272 | 62 |
| R | AT_O_AREA | reused | 20.644 | 20.644 | 39.323 | 40.990 | 16.842 | 36.556 | 41.980 | 51 |
| R | AT_U00 | reused | 22.081 | 4.067 | 9.970 | 14.387 | 1.351 | 13.939 | 18.688 | 61 |
| R | AT_U11 | reused | 19.810 | 8.352 | 11.382 | 17.667 | 2.391 | 18.579 | 22.017 | 55 |
| P | LO_U00_OVI_FILL | new | 23.153 | 20.644 | 39.323 | 40.990 | 15.136 | 36.592 | 42.025 | 53 |
| P | LO_U00_LOCAL | new | 23.153 | 16.940 | 33.638 | 35.408 | 9.975 | 37.356 | 42.599 | 67 |
| P | LO_U00_SPATIAL | new | 23.153 | 16.940 | 33.638 | 35.408 | 9.975 | 37.362 | 42.600 | 66 |
| R | LO_U00_OVI_FILL | new | 22.081 | 20.644 | 39.323 | 40.990 | 14.857 | 36.586 | 42.018 | 54 |
| R | LO_U00_LOCAL | new | 22.081 | 17.480 | 36.068 | 38.082 | 10.584 | 37.203 | 42.444 | 64 |
| R | LO_U00_SPATIAL | new | 22.081 | 16.940 | 33.638 | 35.651 | 10.584 | 37.207 | 42.439 | 65 |

Counts, source composition and evidence coverage for new rows:

| Run | Method | Candidates OVI/SF | Nonempty unique OVI/SF | Empty | Small nonempty | Observed source % | Challenger-qualified source % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | LO_U00_OVI_FILL | 64/141 | 64/26 | 115 | 24 | 98.085 | 93.664 |
| P | LO_U00_LOCAL | 64/141 | 63/117 | 25 | 86 | 98.085 | 93.664 |
| P | LO_U00_SPATIAL | 64/141 | 63/114 | 28 | 83 | 98.085 | 93.664 |
| R | LO_U00_OVI_FILL | 64/143 | 64/23 | 120 | 20 | 98.081 | 93.660 |
| R | LO_U00_LOCAL | 64/143 | 63/115 | 29 | 89 | 98.081 | 93.660 |
| R | LO_U00_SPATIAL | 64/143 | 63/114 | 30 | 88 | 98.081 | 93.660 |

Full precision, per-class AP, ignored event counts and costs: [performance](../../../artifacts/static_ovmap/local_ownership_v1/analysis/performance.json), [source counts](../../../artifacts/static_ovmap/local_ownership_v1/source_counts.json), [protocol](../../../artifacts/static_ovmap/local_ownership_v1/evaluation/metric_protocol.json). The 454,188/454,699 atoms and 2,437,909/2,443,916 candidate incidences retain same-class and SF-only conflicts. Empty candidates remain represented; no redistribution or AP rescoring occurs.

## Table B — controlled effects

After minus before, **percentage points**. Candidate AP changes are zero for protection/local/spatial ownership comparisons. Comparisons to O-only have a different candidate pool: their already-known candidate AP difference is explicitly excluded from the new ownership contribution.

| Run | Effect | Δ candidate AP | Δ unique AP | Δ unique AP50 | Δ cAP75 | Δ mIoU | Δ mAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | protection | 0.000 | 14.640 | 22.855 | 13.507 | 20.539 | 21.051 |
| P | local_evidence | 0.000 | -3.704 | -5.686 | -5.161 | 0.764 | 0.573 |
| P | spatial_term | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.001 |
| P | local_net | 2.509 | -3.704 | -5.686 | -6.867 | 0.800 | 0.619 |
| P | spatial_net | 2.509 | -3.704 | -5.686 | -6.867 | 0.806 | 0.620 |
| R | protection | 0.000 | 16.577 | 29.353 | 13.506 | 22.647 | 23.330 |
| R | local_evidence | 0.000 | -3.164 | -3.255 | -4.274 | 0.618 | 0.427 |
| R | spatial_term | 0.000 | -0.540 | -2.431 | 0.000 | 0.004 | -0.006 |
| R | local_net | 1.437 | -3.164 | -3.255 | -6.258 | 0.647 | 0.464 |
| R | spatial_net | 1.437 | -3.704 | -5.686 | -6.258 | 0.651 | 0.459 |

Protection=U00_FILL−raw U00; local_evidence=LOCAL−U00_FILL; spatial_term=SPATIAL−LOCAL; local_net/spatial_net compare to paired common-domain O-only. U00_FILL ties O-only unique AP on both runs. LOCAL increases macro mIoU but decreases unique AP and canonical high-IoU quality versus both references. SPATIAL adds no unique AP on primary and loses 0.540 pp on repeat. Improvement over very poor raw-U00 unique maps is insufficient evidence of net method value.

## Table C — object retention and losses

Actual released matching at strict runtime thresholds; GT IDs are the comparison identity. Lists below include positive and negative outcomes, not a favorable subset. Candidate/score owners, exact trace references and all conditional object transitions accompany the machine table.

| Run | Method | Runtime threshold | U00-added IDs | Survive unique | Previously correct OVI lost | New unique over OVI |
| --- | --- | --- | --- | --- | --- | --- |
| P | LO_U00_OVI_FILL | 0.25 | 5000, 5001, 5003, 5005, 7003, 7004, 10004, 15000, 17000, 24000, 33000 | — | — | — |
| P | LO_U00_OVI_FILL | 0.5 | 5000, 5005, 10004, 15000, 17000, 24000, 33000 | — | — | — |
| P | LO_U00_OVI_FILL | 0.7500000000000002 | 5000, 10004, 17000, 33000 | — | — | — |
| P | LO_U00_LOCAL | 0.25 | 5000, 5001, 5003, 5005, 7003, 7004, 10004, 15000, 17000, 24000, 33000 | — | — | 5006 |
| P | LO_U00_LOCAL | 0.5 | 5000, 5005, 10004, 15000, 17000, 24000, 33000 | — | 5004 | — |
| P | LO_U00_LOCAL | 0.7500000000000002 | 5000, 10004, 17000, 33000 | — | — | — |
| P | LO_U00_SPATIAL | 0.25 | 5000, 5001, 5003, 5005, 7003, 7004, 10004, 15000, 17000, 24000, 33000 | — | — | 5006 |
| P | LO_U00_SPATIAL | 0.5 | 5000, 5005, 10004, 15000, 17000, 24000, 33000 | — | 5004 | — |
| P | LO_U00_SPATIAL | 0.7500000000000002 | 5000, 10004, 17000, 33000 | — | — | — |
| R | LO_U00_OVI_FILL | 0.25 | 5000, 5001, 5003, 5005, 7003, 7004, 10004, 15000, 17000, 33000 | — | — | — |
| R | LO_U00_OVI_FILL | 0.5 | 5000, 5005, 10004, 15000, 17000, 33000 | — | — | — |
| R | LO_U00_OVI_FILL | 0.7500000000000002 | 5000, 10004, 33000 | — | — | — |
| R | LO_U00_LOCAL | 0.25 | 5000, 5001, 5003, 5005, 7003, 7004, 10004, 15000, 17000, 33000 | — | — | 5006 |
| R | LO_U00_LOCAL | 0.5 | 5000, 5005, 10004, 15000, 17000, 33000 | — | 5004 | — |
| R | LO_U00_LOCAL | 0.7500000000000002 | 5000, 10004, 33000 | — | — | — |
| R | LO_U00_SPATIAL | 0.25 | 5000, 5001, 5003, 5005, 7003, 7004, 10004, 15000, 17000, 33000 | — | — | 5006 |
| R | LO_U00_SPATIAL | 0.5 | 5000, 5005, 10004, 15000, 17000, 33000 | — | 5004 | — |
| R | LO_U00_SPATIAL | 0.7500000000000002 | 5000, 10004, 33000 | — | — | — |

At >.5, none of the original U00-added 7/6 GT objects survives any new unique map; raw U00 previously retained two in each run. LOCAL and SPATIAL lose previously correct blinds GT5004 in both runs. Its OVI owner40 mask had IoU .5131137407 under protection. LOCAL reduces owned projected size from 20,520 to 12,848/12,885 (62.612/62.792%), chiefly transferring original source points to SF query9. Its fixed geometric-atom-graph component count rises 32→56 / 31→67. These are graph structure proxies, not mesh topology or a postprocessing split. Same-class reassignment can destroy object extent without changing semantic class.

The repeat SPATIAL AP loss does **not** require another matched-GT loss at .25/.5/.75. SF query137 (chair, canonical105) grows from 99 to 101 projected points, crossing the fixed evaluator min-region100. It is then a counted unmatched FP with unchanged area rank25082 above two chair TPs ranked24085 and23879. Chair AP50 falls from1 to.416667; all-class unique AP50 falls 2.431 pp. SF query198 (nightstand) grows94→110, but its no-GT class AP remains null. The rules were not adjusted to suppress these unfavorable results.

See [object transitions](../../../artifacts/static_ovmap/local_ownership_v1/analysis/objects.json), [extent/competing candidates/events](../../../artifacts/static_ovmap/local_ownership_v1/analysis/candidate_extent_and_events.json.gz), [size-boundary events](../../../artifacts/static_ovmap/local_ownership_v1/event_transitions.json), and [deterministic examples](../../../artifacts/static_ovmap/local_ownership_v1/analysis/examples.json). Positive multi-GT intersections, background/localization flags and empty duplicates are nonexclusive diagnostics, not an additive error taxonomy. Low owned fraction is not itself a false negative: actual evaluator matches determine that.

## Table D — evidence and regional diagnosis

Regions are frozen original U00 support×multiplicity×class agreement plus projection-unmatched. All integer regional confusions and paired deltas reconstruct the global valid-GT confusion exactly, including pred0 false negatives. Below are all regions for LOCAL−FILL and SPATIAL−LOCAL, both runs; C/W denote semantic correctness.

| Run | Effect | Region/support/multiplicity/agreement | Valid GT | C→W | W→C | W→W | C→C | Owner changes | Class changes | Same-class owner changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P | local_evidence | -1/PROJECTION_UNMATCHED/None/None | 111180 | 0 | 0 | 111180 | 0 | 0 | 0 | 0 |
| P | local_evidence | 0/neither/zero/none | 436 | 0 | 0 | 436 | 0 | 0 | 0 | 0 |
| P | local_evidence | 13/OVI-only/one/unanimous | 6366 | 0 | 0 | 3475 | 2891 | 0 | 0 | 0 |
| P | local_evidence | 22/SpaCeFormer-only/one/unanimous | 226 | 0 | 0 | 115 | 111 | 0 | 0 | 0 |
| P | local_evidence | 25/SpaCeFormer-only/multiple/unanimous | 412 | 0 | 0 | 164 | 248 | 97 | 0 | 97 |
| P | local_evidence | 26/SpaCeFormer-only/multiple/conflicting | 727 | 84 | 71 | 272 | 300 | 270 | 241 | 29 |
| P | local_evidence | 34/both/multiple/unanimous | 244558 | 0 | 0 | 22611 | 221947 | 1147 | 0 | 1147 |
| P | local_evidence | 35/both/multiple/conflicting | 582202 | 2327 | 2031 | 156754 | 421090 | 35267 | 6552 | 28715 |
| P | spatial_term | -1/PROJECTION_UNMATCHED/None/None | 111180 | 0 | 0 | 111180 | 0 | 0 | 0 | 0 |
| P | spatial_term | 0/neither/zero/none | 436 | 0 | 0 | 436 | 0 | 0 | 0 | 0 |
| P | spatial_term | 13/OVI-only/one/unanimous | 6366 | 0 | 0 | 3475 | 2891 | 0 | 0 | 0 |
| P | spatial_term | 22/SpaCeFormer-only/one/unanimous | 226 | 0 | 0 | 115 | 111 | 0 | 0 | 0 |
| P | spatial_term | 25/SpaCeFormer-only/multiple/unanimous | 412 | 0 | 0 | 164 | 248 | 18 | 0 | 18 |
| P | spatial_term | 26/SpaCeFormer-only/multiple/conflicting | 727 | 4 | 1 | 355 | 367 | 20 | 17 | 3 |
| P | spatial_term | 34/both/multiple/unanimous | 244558 | 0 | 0 | 22611 | 221947 | 269 | 0 | 269 |
| P | spatial_term | 35/both/multiple/conflicting | 582202 | 214 | 245 | 158836 | 422907 | 1997 | 696 | 1301 |
| R | local_evidence | -1/PROJECTION_UNMATCHED/None/None | 111180 | 0 | 0 | 111180 | 0 | 0 | 0 | 0 |
| R | local_evidence | 0/neither/zero/none | 491 | 0 | 0 | 491 | 0 | 0 | 0 | 0 |
| R | local_evidence | 13/OVI-only/one/unanimous | 6064 | 0 | 0 | 3489 | 2575 | 0 | 0 | 0 |
| R | local_evidence | 22/SpaCeFormer-only/one/unanimous | 135 | 0 | 0 | 32 | 103 | 0 | 0 | 0 |
| R | local_evidence | 25/SpaCeFormer-only/multiple/unanimous | 386 | 0 | 0 | 165 | 221 | 92 | 0 | 92 |
| R | local_evidence | 26/SpaCeFormer-only/multiple/conflicting | 789 | 6 | 67 | 437 | 279 | 282 | 263 | 19 |
| R | local_evidence | 34/both/multiple/unanimous | 244838 | 0 | 0 | 23025 | 221813 | 1268 | 0 | 1268 |
| R | local_evidence | 35/both/multiple/conflicting | 582224 | 2408 | 2013 | 156344 | 421459 | 34744 | 6230 | 28514 |
| R | spatial_term | -1/PROJECTION_UNMATCHED/None/None | 111180 | 0 | 0 | 111180 | 0 | 0 | 0 | 0 |
| R | spatial_term | 0/neither/zero/none | 491 | 0 | 0 | 491 | 0 | 0 | 0 | 0 |
| R | spatial_term | 13/OVI-only/one/unanimous | 6064 | 0 | 0 | 3489 | 2575 | 0 | 0 | 0 |
| R | spatial_term | 22/SpaCeFormer-only/one/unanimous | 135 | 0 | 0 | 32 | 103 | 0 | 0 | 0 |
| R | spatial_term | 25/SpaCeFormer-only/multiple/unanimous | 386 | 0 | 0 | 165 | 221 | 12 | 0 | 12 |
| R | spatial_term | 26/SpaCeFormer-only/multiple/conflicting | 789 | 0 | 1 | 442 | 346 | 6 | 6 | 0 |
| R | spatial_term | 34/both/multiple/unanimous | 244838 | 0 | 0 | 23025 | 221813 | 290 | 0 | 290 |
| R | spatial_term | 35/both/multiple/conflicting | 582224 | 220 | 248 | 158504 | 423252 | 1993 | 667 | 1326 |

Evidence and graph coverage (source domain, percentages):

| Run | Region | Observed points % | Challenger atoms | No-challenger fallback % | LOCAL changed points | SPATIAL changed points | Supported edges touching region |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | 0 | 79.730 | 0 | 100.000 | 0 | 0 | 276 |
| P | 13 | 74.021 | 0 | 100.000 | 0 | 0 | 2814 |
| P | 22 | 83.784 | 0 | 100.000 | 0 | 0 | 238 |
| P | 25 | 90.840 | 115 | 33.206 | 270 | 27 | 439 |
| P | 26 | 89.465 | 334 | 42.717 | 830 | 58 | 1134 |
| P | 34 | 97.788 | 126122 | 6.029 | 3105 | 673 | 414896 |
| P | 35 | 98.347 | 290803 | 5.961 | 79215 | 4671 | 924771 |
| R | 0 | 73.128 | 0 | 100.000 | 0 | 0 | 276 |
| R | 13 | 74.941 | 0 | 100.000 | 0 | 0 | 2933 |
| R | 22 | 81.600 | 0 | 100.000 | 0 | 0 | 186 |
| R | 25 | 91.051 | 122 | 30.156 | 270 | 28 | 450 |
| R | 26 | 89.925 | 350 | 45.469 | 847 | 19 | 1205 |
| R | 34 | 98.110 | 127006 | 5.615 | 3498 | 766 | 418322 |
| R | 35 | 98.190 | 290333 | 6.168 | 78386 | 4682 | 923110 |

No-challenger fallback is 6.336/6.340% overall; entirely unobserved points are only about1.92%. Shared evidence exists over most points, so blanket observation unavailability is not the explanation. Usable correspondence rejects small remainders, low IoU and ambiguous matches explicitly; per-incidence frame IDs/counts/masses/agreements are stored externally. [Evidence regions](../../../artifacts/static_ovmap/local_ownership_v1/analysis/evidence_regions.json) include score-margin quantiles and distinguish missing observations from insufficient challengers.

There are 1,322,619/1,323,903 positive graph edges, so SPATIAL is not an empty-graph control. Its energies strictly decrease across five recorded sweeps. Final atom/point change counts are in [verification](../../../artifacts/static_ovmap/local_ownership_v1/verification.json). LOCAL changes 83,420/83,001 source points relative to protection; on valid GT, 29,988/29,893 ownership changes retain the semantic class. LOCAL causes 2,411/2,414 correct→wrong and2,102/2,080 wrong→correct semantic vertices, despite improving macro mIoU: macro IoU does not equal total vertex accuracy. SPATIAL changes another2,304/2,301 valid-GT owners and slightly improves mIoU while failing to improve unique instances.

Projection-unmatched remains111,427 vertices (111,180 valid semantic GT), all pred0. Changes occur only at U00 multiplicity>=2; neither single-candidate nor unsupported regions are altered. Raw depth agreement is visibility, not instance truth. Frame predictions share frontend provenance with OVI and are not independent teachers. Exact whole-mask/class-conflict pairs were absent in both banks; nevertheless geometry-only evidence has no semantic-class correctness signal, and unresolved class-conflict regions remain. No GT-based class choice is introduced.

## Cost and scientific decision

CPU threads are explicitly8. The complete prediction pass took 456.53s (asset binding, shared full-cloud projection, atom/evidence/graph construction, solvers and writes), peak RSS 1.261GiB. Shared selected-frame projection took 21.89s, counted once because returned coordinates/order are exactly equal. The measured real smoke used65,536 source points, frame0 and38,609 positive-depth-consistent pixels; it is I/O evidence, not a benchmark subset result.

| Run | Atoms s | Evidence s | Graph s | FILL solver s | LOCAL solver s | SPATIAL solver s |
| --- | --- | --- | --- | --- | --- | --- |
| P | 24.493 | 96.127 | 9.640 | 0.352 | 0.106 | 61.346 |
| R | 26.404 | 104.048 | 8.973 | 0.494 | 0.124 | 59.584 |

All six unique-map evaluations plus checks/export took 77.29s, peak RSS 0.933GiB; post-hoc diagnosis took 45.09s. Per-cell unique-mask export and released evaluation times are in Table A's machine rows. New prediction caches were cold; overlapping AP/trace was reused only after canonical mask/class/.6f parity. These nested CPU timers are not end-to-end FPS and exclude original network/mapping cost. [Costs](../../../artifacts/static_ovmap/local_ownership_v1/costs.json).

H1/H2/H3 are not supported for the required net unique-instance objective. Local evidence is present and moves ownership, but fails to preserve gains or correct OVI extent. Spatial smoothness decreases its own energy without repeat-consistent instance benefit and exposes threshold-sensitive residual FPs. The final status is **COMPLETE_NO_NET_UNIQUE_GAIN**, not an implementation failure or a coverage-blocked experiment.

Only Room0 development evidence and two saved runs are available; no confidence interval, significance, generalization, or unseen-scene claim follows. Extra3D pretraining is disclosed; training-scene exclusion remains UNVERIFIED. Raw model logits were not opened. No parameter sweep or seventh method was run.

Single next experiment: a separately authorized, predeclared Room1 confirmation using the unchanged O-only/U00_FILL/LOCAL/SPATIAL package and two saved runs, with all metrics and object accounting. Do not select a favorable scene, add a repair or launch new mapping/inference in this task. Implementation and reproduction details: [handoff](LOCAL_OWNERSHIP_HANDOFF.md).
