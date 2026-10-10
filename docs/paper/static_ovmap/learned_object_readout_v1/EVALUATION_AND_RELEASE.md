# Evaluation, nomination, resource accounting and release

## 1. Three different evaluations, never merge their meaning

| Set | What the inputs are | What may be concluded |
|---|---|---|
| DEV (4 new train-split families) | GT-derived clean/corrupted proposals, unseen by the optimizer | Checkpoint and architecture selection; supervised development only |
| H (4 separately withheld families) | Same declared proposal generator, labels hidden until DEV nominations | New-family **proposal recognition** under controlled perturbation; NOT independent whole-map AP |
| Replica8 and CF18 | Existing predicted G1 geometry/owners and real RGB-D captures | Same-protocol whole-map regression on repeatedly exposed benchmarks; not untouched confirmation |

Do not generate GT proposals on the 26 existing scenes for primary inference, do not project target GT colors into model features, and do not silently turn H recognition into a map-level comparison with published OVI-MAP scores.

## 2. Checkpoints and architecture nomination before H/regression

For DEV checkpoint selection, the exact criterion is:

`mean_over_families(mean_over_original_base_objects(0.5*CE(clean) + (CE(truncate)+CE(append)+CE(truncate_append))/6))`.

CE uses the base-category text denominator and fixed loss temperature0.07. Missing numerical prediction is an execution error; a method is not rewarded by dropping hard examples. Fixed engineering eligibility applies equally. Store per-family values, examples and corrupted no-op counts. Only the four planned checkpoint steps are compared; ties <=1e-12 choose the earlier step.

Select a proposed architecture from LR06/07/08 using the same DEV criterion; tie LR06,LR07,LR08. LR05 is the trained simple comparator. Train repeat seed29 for these two, without changing architecture or schedule. Write `dev_nomination.json` including the entire DEV ranking, checkpoint hashes, effective data and class lists before H is read. Do not nominate from H, Replica, CF18, latency or memory.

Main results are seed17. Repeat results are seed29 and stay separate; never select the better seed per dataset/object/class. All four scientific branches remain in the main table even when DEV dislikes them, provided training is technically valid. Do not hide a simple baseline that beats the proposed models.

## 3. H and representation metrics

On H, evaluate FC2/4/8 and the four learned models, seven methods, on clean and three corrupt proposal conditions. Report full-vocabulary top1, macro class recall and NLL at the same fixed0.07; base/adapter-heldout/all-class subsets with counts; drop from clean to corrupt; complete/missing view distribution. Report an original-object denominator as well as the four correlated variant records.

Provide the following fixed paired analyses: MA8 minus FC8 (learning a region pool); MV_VIEW minus MA8 (hierarchical multiview readout); MV_SURFACE minus MV_VIEW (physical correspondence at matched parameter shapes); MV_AUX minus MV_SURFACE (supervised membership/correspondence losses). All use the same view bank and image budget. Run the corresponding repeat pair on H as a separately marked seed check.

These tests use clean/GT-derived supports deliberately to measure representation capacity. They do not prove the GT-derived corruption distribution matches actual predicted-map errors. The whole-map regression below is required precisely because the gap may be substantial.

## 4. Whole-map output construction and original official evaluator

Resolve D2/G1 baselines from the full source-update parent's compact store and external real predictions. Compare frozen geometry, source projection and ordered vocabularies to the current base commit. The four-scene disagreement result store is NOT the 26-scene parent.

All new outputs share the full G1 owner partition and recovered labels. Only eligible D2 incumbent labels can change. Rebuild current-class area ranks using `native_ranks`/`official_view` under the existing projection. A change of class can change ranks of other instances; never carry old ranks forward. Do not replace official ranks with learned confidence to manufacture AP gains.

Use a registry enumerating every positive owner in the **actual full G1 output**, not just the selected/trained owners. The same prediction supplies AP and mIoU. Unknown, small or unselected surface regions remain in semantic confusion counts. No new geometric support or mask pruning.

Reuse `SceneEvaluator`, `validate_overlaps`, `fraction_metrics`, `released_pool_with_classes`, and trace diagnostics through a new adapter without old branch/method-count hardcoding. APall here is the inherited `.50,.55,...,.90` protocol, AP50 strict matching per released code and minimum100 points; do not silently use COCO .95 or a different ScanNet release. Verify overlaps from the actual runtime evaluator, not only a JSON constant.

Generate all required predictors' complete scene payloads before loading that stage's evaluation labels. Main seed17:9*26=234 logical rows,9*2=18 pools. Repeat:2*26=52 rows,2*2=4 pools. Total286/22 if complete. Baseline aliases and identical score-input aliases require exact semantic/owner/rank/scorer-context identity; a rounded metric tie is insufficient. The logical record still names its method/seed and reuse proof.

Pool all ordered scenes through the released evaluator. Never average per-scene AP. Save full5 metrics in fractions plus display-percent tables. Per-class and per-scene values must be recoverable. Existing parent results are permitted only for exact baseline or identical scorer-input aliases, not missing trained predictions.

## 5. Selection and interpretation

For each main nonparent method, the strict research upgrade gate is:

1. Replica8's APall,AP50,AP25,mIoU,mAcc each >= the complete G1 baseline minus1e-10.
2. CF18 same five comparisons >= G1 minus1e-10.
3. CF18 APall > D2 APall +1e-10; CF18 AP50 >= D2 minus1e-10.
4. One method/checkpoint/algorithm across both cohorts.

Material flag adds CF18 APall >= D2 +0.001 fraction (0.10pp), within the comparison tolerance. Runtime is not in either gate. A legitimate negative result still counts as completed science. Do not reweight the criteria after results.

Keep distinct fields:

- `dev_nomination`: trained architecture/checkpoint chosen without benchmark outcomes.
- `best_regression_candidate`: among main methods satisfying the strict gate, lexicographically maximize CF APall, Replica APall, CF mIoU, Replica mIoU, then prefer untrained/fewer-parameter/lower method ID ties. This is explicitly **exploratory selection on exposed benchmarks**.
- `repeat_consistency`: report the seed29 gate and paired deltas for the pre-nominated pair. Do not train extra seeds because a non-nominated model happens to win regression.
- `research_retained`: best passing main candidate, or G1 if none. A learned result without a second-seed check is labeled `SINGLE_SEED_CANDIDATE`, not robustly replicated.
- `deployment`: always `N0_UNCHANGED` in this research task.

A novel-mechanism claim needs more than passing a baseline gate: show the pre-nominated proposed branch improves over the trained MA control / matched VIEW control under the appropriate paired comparison, and disclose any heldout or seed inconsistency. Do not present a post-hoc regression winner as preselected independent validation.

## 6. Per-object explanation, not only aggregate AP

For every eligible existing owner retain old label, old full N/Q/F cosine/probability records, each selected view/mask identity, new embedding/scores, learned pooling maps or compact summaries, proposed/applied label, output rank changes and availability. Report ordinary F top1 vs learned F top1 separately from final D2+newF labels.

After outputs lock, associate original predicted supports to GT using the existing fixed-reference matching diagnostics. Label wrong->right/right->wrong only when the GT reference is defined under the stated geometry criterion; undefined instances stay in the denominator and get an undefined category, not automatically “wrong.” Report class-aware unique GT50/75 gained/lost, tied matcher-entry attribution where applicable, and complete metric deltas.

Class-agnostic matches must be unchanged because geometry is fixed. If they change, debug the output/evaluation path before scientific conclusions. Do not call a category correction an improvement in reconstructed surface geometry.

For auxiliary supervision report valid physical-site pairs, positive/negative membership token counts, fraction of unavailable local tokens, and explicit MA-fallback count. A zero auxiliary loss with no valid pairs is not successful correspondence learning. Report different 2/4/8 prefixes actually available; do not call repeated padding “eight views.”

## 7. Three table families

**Table1 — Whole-map regression:** the nine seed17 methods, Replica8 and CF18 APall/AP50/mIoU (full5 in adjacent machine supplement). Denote trained heads clearly; give training families/positive classes. Show two seed29 rows/pair in a small supplement, not mixed into best-of-seed values. Published external literature values, if retained from old work, stay separately sourced and are not same-protocol new replications.

**Table2 — What learning contributed:** fixed paired differences MA-FC8, VIEW-MA, SURFACE-VIEW, AUX-SURFACE; DEV/H proposal recognition, heldout-category denominator, whole-map correction/damage counts. Training budget and exact parameter counts accompany this table. Do not claim 3D necessity if VIEW or MA is best.

**Table3 — Robustness and actual compute:** clean/truncate/append/both recognition changes, missing-view/local-support coverage, actual scientific encoder/step counts, wall time and allocated/reserved peaks. These are stage costs, not online FPS or a promised latency advantage. Model resident inference with cached features must be labeled as conditional, not cold end-to-end.

Make Markdown, CSV, JSON and LaTeX table fragments. PDF rendering is optional and not a scientific completion dependency; do not spend a new round solely restyling the runtime table. Include readable README/reproduction commands and avoid narrative clutter about hundreds of hashes in the main paper table.

## 8. Resource and failure policy

A slower model can win. Do not veto learned heads using G1 latency. Reuse frozen raw image features, stream objects, preserve FP32 and limit CPU workers to4 for data and3 for evaluation. One physical A40 worker by default; multiple devices only for independent branches with recorded allocations and identical schedules, not changed minibatch math.

Report feature capture, supervised data generation, training, DEV/H inference, whole-map inference, evaluation and packaging separately. Physical encode count differs from logical object-view requests. Cache hits do not make those inputs free at deployment. 3904 distinct scientific FC-image upper bound comes from the fixed new-family and regression frame pools; repeats reuse those features. Engineering/retry work is separately counted, with null when a failed leaf was not timed.

If disk is full, preserve parents and completed shards, move only task-owned outputs using explicit path map/storage root, resume from identity-valid state. Failed mandatory inference is not a model-empty prediction. A model returning a finite but wrong label is a valid negative result. Do not replace wrong outputs with G1 based on evaluation labels.

## 9. Minimum release content

Under `artifacts/static_ovmap/learned_object_readout_v1/` publish compact split/class manifests, execution config/versions, method registry, DEV selection, H metrics, all complete regression metrics, per-object decisions/compact embeddings or compressed score arrays, learned-head model configs and selected checkpoint hashes. Save small learned weights for all selected main branches and the two repeat branches; only trainable heads, never the bound FC/text backbone. Retain original upstream license/notices for the reused numerical MA head. Respect data license terms when sharing derived weights; if redistribution is not permitted, publish the exact checkpoint identity and authorized retrieval instructions rather than raw restricted training data.

Required reports under `docs/paper/static_ovmap/`:

- `LEARNED_OBJECT_RESULTS.md`
- `LEARNED_OBJECT_HANDOFF.md`
- `LEARNED_OBJECT_SELECTION.md`
- `LEARNED_OBJECT_CLAIMS.md`

The handoff includes one exact successful full CLI (or exact failure and uncompleted stages), all resolved data/model/cache paths, environment command, how to load selected head and reproduce every table, train/base/heldout vocabulary scope and remaining limitations. Report no-target-gain honestly; do not write “not allowed to use this result” boilerplate in an otherwise completed results table.

Use normal Git commits and push on the task branch. After push get local `git rev-parse HEAD` and `git ls-remote origin refs/heads/research/ovimap-learned-object-readout-v1`; record exact equality in external `publication/final.json` with observed CLI exit/status/result identity. The release commit cannot contain its own full SHA. Worktree cleanliness is a separate check, not accuracy evidence. If push fails, report `PUSH_FAILED` and retained local artifacts; do not say published.
