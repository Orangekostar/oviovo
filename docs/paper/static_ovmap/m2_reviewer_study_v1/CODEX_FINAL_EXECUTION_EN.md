# Final execution directive: M2_CAL reviewer-evidence study

**Specification date:** 2026-09-28  
**Repository:** `Orangekostar/oviovo`  
**Audited implementation anchor:** `8fee8294c1a3e83feeae28782f6ef4700f08d35c`  
**Upstream OVI-MAP anchor:** `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`  
**New branch:** `research/ovimap-m2-reviewer-evidence-v1`

This is an implementation, execution, analysis, and publication instruction, not a request for another plan. Read this document and `PROTOCOL_SPEC.json` completely. `SOURCES.md` binds the factual premises to existing code. Proposed `RV_*` methods, new modules, and the new runner below do **not** exist merely because this specification names them; implement them.

## 0. Objective and non-negotiable outcome

Determine which parts of the existing M2_CAL result are necessary and defensible: source complementarity, soft versus hard decisions, probability sharpness, source-specific calibration, query selection, and evaluation conventions. A simpler method is an acceptable scientific winner. Do not assume that three branches or three temperatures must survive.

The frozen target is `CP_M2_EQUAL_CAL`, not a newly tuned replacement. Keep the original maps, results, checkpoints, temperatures, data locks, and upstream evaluator unchanged. Deliver real predictions, metrics, object-level explanations, a claim ledger, and a verified GitHub push. Negative findings are complete scientific results. A functioning smoke, a large passing test count, or a table of null metrics is not a completed experiment.

No new foundation models; no SF/WOW integration; no new geometry, segmentation, owner allocation, or Q-head training. The core source/temperature ablations require **zero new image inference**. Missing CPU score reconstruction must not be replaced by invented logits, one-hot vectors, or a different model. Query-control experiments are allowed to execute missing native six-crop requests. Vocabulary tests are allowed to encode text only. At most four genuinely new, locally authorized ScanNet scenes may be captured for confirmation under Section 12.

Do not stop all work because one optional data-dependent stage is unavailable. Implement its actual success path, record the concrete limitation, and finish every executable stage. Do not modify the study to manufacture a positive result.

## 1. What was actually established, and what remains a hypothesis

The published Replica table contains eight scenes and six methods. Its APall field is an alias of `uap`, derived from released `all_ap`; it is not an additional metric. The displayed results use **equal scene means**. M2_CAL improves those means over N0, while Q_GAIN and static SigLIP2 individually do not. That establishes a promising result, not the necessity of every component. [E12,E14,E17]

Source code establishes the following:

| Existing behavior | Required consequence for this task |
|---|---|
| `label_fusion.fuse_labels()` requires exactly three source keys. | Implement a small separate ablation fuser. Do not mutate the historical source-set check to make old experiments accept new methods. [E01] |
| A source must have genuine evidence, a complete score vector, and a score argmax reproducing its label. | Preserve availability masks and class-ID alignment; a fallback to N0 is not another source vote. [E01,E03] |
| N0 scores are canonical-relative; Q/S2 scores are cosines. | Add a **score-representation** ablation, not just different temperatures. [E03] |
| Temperature fitting uses scene-balanced source NLL, log bounds `[0.01,2]`, and two ScanNet CAL scenes. | Keep the loss, fitting data, support checks, and bounds explicit. A parameter near the lower bound is not proof of fine-grained reliability learning. [E02,E04] |
| Q decisions use current captured geometry and previously paid evidence; lineage can irreversibly discard features. | New query controls must replay causally, not select from final-map/GT knowledge or resurrect evicted observations. [E05–E08] |
| `same_source_surface()` and S/Q validation require original ranks. | Reranking belongs in a separate **evaluation view**, not a relaxation of the frozen-prediction contract. [E03,E10,E11] |
| Official `eval_sem_seg.py` gathers all scene files before calling `evaluate()` and computes semantic confusion across scenes. | Add official **dataset-pooled** evaluation. It is not the mean of scene AP or scene mIoU. [E14,E15] |
| Replica access wrappers whitelist the old six methods and B200. | Add a new study access contract and dataset adapter; do not disguise RV methods as authorized old methods. [E12,E13] |

For N0, the code computes

`r(c) = min_k exp(s(c))/(exp(s(c))+exp(a_k)) = sigmoid(s(c)-max_k a_k)`.

This monotone transform normally preserves top-1 but changes score gaps; floating-point ties must still be audited. This is a mathematical observation about the code, not an experimentally established explanation for the gains. [E03]

## 2. Workspace, resources, and immutable inputs

Start from the anchor in a clean dedicated worktree. Preserve unrelated work and do not reset, force-push, change default branches, or overwrite old artifacts. If the requested task branch already exists, inspect and resume it only when it belongs to this study. Record actual HEAD and the base relationship. A newer unrelated remote commit is not authorization to silently change the experimental anchor.

Default new output root:

`/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001`

Resolve resources from actual files in this order:

1. `/mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001/transfer.json` and each scene's `resolved_config.json`, `source_manifest.json`, source bundles, query receipts, predictions, and evaluation records.
2. Its `source_selection` and the matching ScanNet composition root, normally `/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001`.
3. The runtime/model/checkpoint identities recorded by those files, including the module study root `/mnt/shared/ww/ovimap-module-validation-v1/scannet_study_v1`.
4. The repository's `artifacts/static_ovmap/replica_composition_transfer_v1/transfer.json` is a locator and identity reference, **not** proof that the large arrays exist locally.

Use the existing native/SigLIP2 environments, explicit HWC image backend, FP32 visual/text models, text ordering, crop convention, Q checkpoint and scaler. Resolve actual Python executables from the bound runtime rather than installing a new environment. Do not mix features from the two model spaces. Hash consumed assets once per run, memoizing unchanged file identities; do not recursively rehash every historical publication on every stage or method.

Bind the original full 48-row CSV and legacy prediction hashes. Source files are a compatibility reference, not numbers to paste into new output. Reconstruct the five core controls and compare their frozen-rank predictions/metrics against actual historical rows. Any discrepancy gets a precise difference report before dependent comparisons proceed. Repair an adapter defect and rerun its affected descendants; do not retune the algorithm.

Resource failure policy: retry one clear path/identity resolution using actual adjacent receipts and mounted roots. Do not recursively search the whole machine. If large source scores or predictions cannot be recovered, identify the missing asset and mark affected methods blocked; finish independently available analyses and publication. Never claim core experimental completion without core metrics.

## 3. Dataset roles and selection permissions

### 3.1 Fixed existing-scene study

- **CAL:** `scene0056_00`, `scene0534_00`.
- **Replica historical transfer:** `room0`, `room1`, `room2`, `office0`, `office1`, `office2`, `office3`, `office4`.
- The old ScanNet FIT/SELECT/CONFIRM families remain exposed history. In particular, neither `scene0553_00` nor `scene0064_00` is assumed fresh again.
- Q's existing checkpoint already used historical FIT/CAL information. Leave-one-CAL temperature fitting controls temperature reuse, but does not turn CAL into an independent test of the entire Q system.

Use CAL only for the explicitly authorized scalar fits, simple-comparator nomination, and query-curve trigger. Replica is a **historically exposed, frozen-parameter mechanism/transfer evaluation**, not a blind selection set. Freeze all new fitting rules and method definitions before opening their new Replica results. Do not choose temperatures, seeds, prompts, source weights, or the final comparator from Replica.

### 3.2 Main and comparator nomination

The primary method stays **the published M2_CAL**. Select one simpler comparator from A1–A6 using CAL leave-one-scene-out predictions, in this order: mean frozen-rank uAP descending; mean mIoU descending; required unique visual operations ascending; fixed table order. Require at least one actual label change for nomination. If every candidate has no effective intervention, choose A5 as a declared no-intervention diagnostic comparator rather than claiming it is a learned winner.

Select the strongest single alternative between `Q_GAIN` and `S_SIGLIP2_AREA` on the same CAL lexicographic metric rule. N0 remains a mandatory baseline regardless of this choice. Freeze these identities before Replica analysis. These choices define confirmation controls, not automatic deployment. Do not require all scenes to improve merely to run the study; report scene losses separately. Deployment stays `N0_UNCHANGED` unless a later explicit decision authorizes replacement.

## 4. Source contract and exact reconstruction

Create a reusable `SourceEvidence` representation with:

- dataset, scene, native geometry/projection identity, owner registry;
- source ID, query-policy ID when applicable, model/processor/text identity;
- `valid_ids` and exact class-name order;
- actual availability per owner, label, full score vector, score-representation ID;
- used/retained/attempted request IDs, fallback reason;
- source receipts and required logical operations.

Reuse `native_readout()`, `native_scores()`, `static_objects()`, `query_objects()` and saved aggregates. Preserve their original precision, saved observation order, min-view requirements, top10/last8 conventions and irreversible lineage rules. Do not average hard labels into fabricated probabilities. Do not construct a score vector by copying the old class's one-hot indicator. [E03,E05–E10,E21]

All available source scores must reproduce that source's frozen top-1 after ID alignment. Unavailable sources have no score vector. All owners in the fixed source map remain in whole-map evaluation, including small, uncertain, or semantically unassigned regions; attribution eligibility must not filter predictions.

For the N0 cosine probe, recompute `cosine(native_saved_aggregate, native_text)` from the same saved FP32 aggregate/text, leaving Q and S2 untouched. Do not rerender, recrop, re-encode, or change N0's geometry/ranking. Record every top-1 disagreement with canonical-relative N0, including numeric ties. If non-tie mismatches reveal a reconstruction defect, fix it; do not hide them behind a tolerance.

The original M2 source set includes actual N0 evidence. A3 removes N0 **from fusion**, not from the mapping infrastructure: masks, active registry, and fixed ranks still originate from N0. Do not advertise A3 as a fully N0-free deployable pipeline.

## 5. Required core matrix: eight new rows for seven questions

Five controls: `N0`, `Q_GAIN`, `S_SIGLIP2_AREA`, `CP_M2_EQUAL_RAW`, `CP_M2_EQUAL_CAL`. Reuse exact original predictions where possible. M4's historical result may be displayed for context but is not another required core control.

| ID | Source set and decision | Temperatures | Question |
|---|---|---|---|
| `RV_A1_NQ` | N0 + Q probability mean | Corresponding historical source temperatures | Is observation diversity sufficient? |
| `RV_A2_NS2` | N0 + static S2 probability mean | Corresponding historical source temperatures | Is Q necessary? |
| `RV_A3_QS2` | Q + static S2 probability mean | Corresponding historical source temperatures | Does N0 evidence add value? |
| `RV_A4_HARD` | Three-source plurality vote | None | Does the full distribution matter? |
| `RV_A5_T001` | Original three-source probability mean | Every source `0.01` | Is common sharpening sufficient? |
| `RV_A6_SHARED_T` | Original three-source probability mean | One shared fitted scalar | Are three scalar fits necessary? |
| `RV_A7_COS_FIXED` | Replace only N0 scores by native cosine | Original three source temperatures | What is the effect of score representation at fixed settings? |
| `RV_A7_COS_REFIT` | Same cosine representation as preceding row | Refit N0 scalar on CAL; Q/S2 keep their matching historical fits | Does the representation effect persist after matched scalar calibration? |

A7 has two rows intentionally: changing representation without a recalibrated control cannot establish that one representation is intrinsically better.

Soft fusion: align class IDs, apply stable softmax separately, and take the arithmetic mean over actually available selected sources. No adaptive weights, class-specific rules, margin gate, or source selection by confidence. Excluded/failed sources get no votes or weight. Re-normalize only over the remaining selected sources.

Empty-source behavior: A3 emits semantic class `0` when both selected sources are unavailable; it must not secretly consult N0. All other variants retain the native label only when no selected source is available. Report the A3 abstention cost and also compare methods on the common-source-available object subset; never remove abstentions from whole-map AP/mIoU.

Hard vote: count only genuine available top-1 labels. With a tie, prefer N0 **only if N0 is available and its label is tied**; otherwise choose the tied class occurring first in the frozen `valid_ids` order. No epsilon or scene-dependent tie rule. Do not add an extra tie-breaking vote.

Per-source temperature reuse on CAL means the historical opposite-scene fold fit, not the final fit on both scenes. On Replica use the final source-only ScanNet fit. A1–A3 do not refit weights or temperatures after deleting a branch. The main question is removal from the fixed system, not exhaustive optimization of every competitor.

Counts: `10 scenes × (5 controls + 8 new methods) = 130` scene-method prediction records. Two ranking views yield `260` scene-method-ranking records. This is not 130 independent trials and does not imply 130 image runs. On Replica there are 64 new ablation predictions. Equal prediction/label identities may share actual evaluator work while retaining distinct method/provenance records.

## 6. Scalar fitting and probability evaluation

Reuse source examples with unique strict geometry IoU `>0.5`, actual evidence, and an allowed GT semantic ID. Supervision is evaluation/calibration-side only. Build examples after prediction sources have been locked. Never use semantic correctness to select test objects or their source inputs.

For source `m`, fit temperature by the historical scene-balanced mean source NLL. For a **shared** temperature minimize the equal average of those per-source scene-balanced NLL objectives, using each source's same valid examples. Do not switch A6 to a mixture-NLL objective: that would confound parameter sharing with a different loss. Fit in log space, bounds `[0.01,2.0]`, bounded scalar minimization, `maxiter=64`, `xatol=1e-4`. Insufficient support (`<5` unique objects or `<2` classes), nonfinite inputs, or optimizer failure uses the explicit `0.07` fallback and preserves a failure/status record.

Fit a shared temperature only when every participating source passes the stated object/class support checks on that training fold; otherwise record a shared `0.07` fallback for the entire shared-T condition. Do not silently remove a difficult source from the shared objective.

A7_COS_REFIT refits **only N0** on the same examples and loss; Q/S2 use the corresponding original fits. Run two leave-one-scene-out CAL folds for newly fitted settings and one final CAL refit before Replica evaluation. Keep the published M2_CAL temperatures unchanged even if refitting a new representation helps. Record NLL before/after, example IDs, object/class counts, per-scene counts, optimizer status and bound hits. Do not enlarge the range because a temperature reaches its boundary.

Report probability quality, not only AP:

- Primary paired population: unique GT correspondence, positive GT semantic class, and all three genuine sources available. Use one fixed set per scene for every method.
- Secondary: each method's available union, reporting coverage and missing-source patterns so smaller/easier sets cannot masquerade as better calibration.
- NLL using a `1e-12` numerical clip only at scoring; multiclass Brier `sum_c (p_c-y_c)^2` without division by class count; top-label ECE with 15 equal-width confidence bins; accuracy and object count.
- For hard vote use normalized vote counts as its declared distribution. Do not infer calibrated confidence from vote proportions. For a single source compare its `T=.07` and corresponding final/fold CAL distribution; top-1 must be unchanged except declared numeric ties.
- Apply identical populations to RAW/CAL/shared/cosine comparisons. Record GT-ambiguous/unmatched instances separately; they still remain in map metrics. This estimates calibration conditional on the measured object population, not calibration of every false positive in the world.

No probability-quality fit on Replica, regression, or new confirmation scenes. Additional CAL fits are small models with supervision and must be disclosed.

## 7. Two rankings and two aggregation conventions

### 7.1 Frozen-rank primary view

Preserve the exact N0 mask/owner/rank vectors and `.6f` scores, as in the historical experiment. Class decisions alone change. Continue using the pinned released matching, ignore regions, min-region threshold, class IDs and floating-point IoU thresholds. Read these from the evaluator at runtime, save their exact values, and do not assume a generic COCO range ending at 0.95. In the published Replica record APall/uAP covers 0.50 through 0.90; AP25 is separate. [E12–E17]

Use the existing `PredictionPayload`/`relabel_prediction()` for fixed-rank predictions. Do not add a method to the `G` or unrestricted `COMBO` branch merely to bypass an invariant. Keep diagnostic supervision out of prediction metadata.

### 7.2 Official-current-class ranking view

The purpose is to test whether the claimed improvement survives the author's full export convention. Starting from **the same already frozen owners and final labels**, project with the same frozen projection and call the pinned `map_pred_mesh()` (or a demonstrated exact reproduction) to export each method's masks/ranks under its **current predicted classes**. Do not use fusion confidence as the ranking score.

Create separate evaluation-only manifests under `official_current_class/`. Do not mutate frozen source `instance_ranks` or relax `same_source_surface()`. Compare exported masks/classes/serialized ranks once against the official function on a real scene. Preserve exclusions and zero-class behavior exactly. Report any class-eligibility change, not just the count of exported masks.

### 7.3 Dataset-pooled evaluation is mandatory

The historical CSV is a scene macro-average. The official OVI-MAP entry point instead gathers all scene GT and prediction files and calls its released evaluator once, and pools semantic confusion before averaging class IoUs. Implement both for every core method/ranking view:

1. `SCENE_MACRO`: arithmetic mean of defined complete scene metrics; retain all per-scene deltas.
2. `RELEASED_DATASET_POOL`: one released evaluation over all eight Replica files, with the author's scene order `office0..office4, room0..room2`; semantic mIoU/mAcc from summed confusion, ignoring the same invalid GT rows. Never average AP arrays or class IoUs from separate scenes and label that pooled.

Do not overwrite `APall` with a new meaning. Include explicit `dataset`, `rank_mode`, `aggregation`, scene and class counts. Expected Replica pooled records: `13 methods × 2 rank modes = 26`. Read all runtime class/threshold identities. Keep ScanNet and Replica pools separate. CAL folds can be summarized but are not an independent official benchmark.

Validate pooling by calling the original evaluate path on the same already exported real masks and original ordered files. No new visual inference is needed. For per-scene evaluation cache identity, include dataset, complete labels/ranks/masks, source/projection identity, GT content identity, ordered vocabulary, evaluator source and rank mode. A pooled key also includes the ordered set of all scene inputs. Identical new predictions may reuse exact results but never stale probabilities or probabilities from another temperature.

**Any paper-table superiority statement must use the appropriate official pooled/current-class result and disclose remaining configuration/compute differences.** A positive macro-average alone does not establish it.

## 8. Object and source-mechanism diagnostics

Process all frozen positive owners. Reuse true released traces for AP and a separate class-independent unique IoU correspondence for semantic reasoning. Do not replace the released duplicate/ignore handling with a new greedy matcher.

Produce one full ledger with source availability, source labels, full source/mixture probabilities, correct-class rank (evaluation only), final label, native-to-final change, exact used/retained request sets and source operation identities. Include:

1. N0 wrong → mixture right; N0 right → mixture wrong; preserved N0 correctness; still wrong.
2. All available source top-1 predictions wrong but mixture right; at least one source right but mixture wrong. Give counts and denominators, not just examples.
3. Results on all-three-available, two-source, and missing-source strata; true abstentions separately.
4. N0/Q used-request Jaccard and overlap coefficient, plus model-and-actual-input operation overlap. Repeated observations from one model are correlated evidence, not independent votes. Calculate co-error and exclusive-correction rates on the same identifiable population; do not interpret correlation as causation.
5. Leave-one-source changes from A1–A3 and changed-object margins from A5/A6/A7. Do not assign a unique causal contributor merely because its top-1 equals the mixture's label.
6. Released added/lost/duplicate/ignored matches at every recorded overlap, with a compact display at AP25 and strict IoU>0.5. Preserve trace parity.

Explicitly explain office1 AP50 loss using actual object records, and also report all other scenes. Select qualitative examples deterministically after evaluation: first up to two corrected and two harmed owners in sorted owner order per scene; no hand-picking only flattering examples. Existing image projections may be used for overlays; no new segmentation.

An oracle based on GT may estimate *object classification* headroom on a fixed identifiable population. Label it diagnostic, never a candidate, never a deployable AP row, and never use its decisions to build predictions.

Report scene-level paired deltas and a descriptive scene bootstrap of macro deltas (2,000 draws, seed17). Eight related indoor scenes provide limited assurance. Do not bootstrap vertices as independent samples, and do not pool calibration and evaluation populations to inflate N. Pooled metrics remain point estimates unless an exact, separately budgeted pooled bootstrap is implemented; a macro CI is not a pooled CI.

## 9. Query-necessity controls under a matched budget allowance

This is a mandatory B200 stage, not a speculative recommendation. It asks whether Q_GAIN contributes more than extra or simply different observations. Keep N0, static S2, geometry, ranks, source availability rules and fusion form fixed. Replace only the middle query source.

Policies:

- existing Q_GAIN (locked checkpoint/scaler);
- original Q_COMBINE;
- `RV_Q_RANDOM`, three fixed seeds `17,23,41`.

Run on the two CAL and eight Replica scenes. Keep same schedule, current candidate universe, cumulative B200 allowance, feature-retention count, native FP32 encoder, overlap weights, frame barriers and lineage. The new random policy uses `random_exploration_ranking()` with `(seed,scene_id,frame_index)` hashing. It ranks current eligible unattempted candidates, never future frames or GT. It is **not** the old supervised exploration mode, which has hard-coded FIT/CAL budgets. Add a study-local replay adapter or a tested generic kernel hook; do not fake its name/role to bypass old constraints.

Q_COMBINE must execute its coverage side effects once per valid frame, including quota-zero frames. Failed attempts consume quota. Features are read only after debit; aliases/splits/evictions follow the original implementation. Final owners may be reconciled only after the last frame, not used to make queries. State must start empty; no preloading N0 full-map semantic evidence into the new policy.

For every replacement policy/seed export its standalone labels and full source scores, then two three-source fusions:

- `RV_B_<policy>_RAW`: all temperatures0.07, directly matched with published M2_RAW.
- `RV_B_<policy>_CAL`: N0/S2 retain original matching source fits; fit only the replacement Q-source scalar on the same CAL recipe, opposite-scene folds for CAL and both scenes for Replica. Do not handicap a different query policy by calling the old Q_GAIN temperature its own calibration.

Use policy IDs and seeds in evidence identities even though the middle schema slot is the query source. Do not call a random source Q_GAIN in reports. Report all three random runs and their metric mean/spread; they are three randomized replicates, **not** a three-run ensemble or three independent scenes. Never retain the best random seed.

At most `10 scenes × 5 policy/seed paths × 200 = 10,000` logical native requests at B200, including existing Q_GAIN/Q_COMBINE controls. Reuse exact cached operations wherever possible; only missing operations get new forwards. Static S2 remains unchanged; do not run it again on the new Q requests. Each fusion still includes the original native baseline and the static S2 cost; it is not a standalone B200 system.

### Query curve trigger (CAL only)

Trigger B100/B400 curves only if cross-fitted CAL M2_GAIN has strictly higher mean uAP than **both** the Q_COMBINE fusion and the three-seed mean random fusion, and mean mIoU no lower than either (tolerance1e-10). Freeze the trigger before new Replica control results. Otherwise write a real `NOT_TRIGGERED` reason and continue every other stage.

If triggered, on all eight Replica scenes run Q_GAIN and Q_COMBINE at B100 and B400, with their B200 final CAL temperatures frozen. Do not refit temperatures per budget. This adds at most8,000 logical native requests. Recompute state from the beginning: B100 is not obtained by truncating a B400 final state. Standalone and fused metrics are both reported. The unchanged Q predictor sees different budget features; report this as budget transfer, not a separately trained optimum.

B200 is an equal **allowance**, not a promise of equal actual successful queries. Report actual attempts, success/failure, crop count, and total costs. Where realized counts differ, retain the data and avoid a strict equal-cost claim.

## 10. Cost and implementation-efficiency evidence

Separate four quantities:

1. Shared original mapping/front-end cost and native semantic-capture cost.
2. Each method's required logical visual operations, by model and request.
3. Shared-operation union for a feasible cached deployment, deduplicating only identical model/processor/precision/RGB/target/union/bbox/crop inputs with proved identities.
4. This study's physical work: new forwards, cache hits, model loads, wall time, text encoding, evaluator time and scalar fitting.

Use exact operation hashes and keep source-attributed counts alongside union counts. N0 and Q may share operations; never blindly add them and call that unavoidable cost, but never subtract merely overlapping or similar views either. Preserve the expense needed to create N0's active map/rank registry even when its score source is ablated. A source deletion does not by itself prove end-to-end cost savings.

Measure peak GPU memory on the already required inference processes; serialize per-model jobs in their existing environments. If source-only historical jobs did not record memory, report it missing rather than rerunning full scenes for a number. A bounded timing sample of up to32 already-bound requests per model may be run only when necessary for an otherwise missing timing estimate, separately charged and explicitly labelled a microbenchmark, never end-to-end latency.

Plots/tables compare quality versus total declared request/crop counts and measured incremental inference time. Do not equate one native six-crop request, one S2 request, and a free CPU fusion. Do not claim online real-time operation from offline cached evaluation.

## 11. Vocabulary and prompting robustness without new image inference

Use frozen image aggregates from the same objects. Recover missing aggregates from the exact final retained request IDs and stored raw six-crop means/visible-area weights; top-1 labels or probabilities cannot be inverted into image embeddings. Preserve original N0 last8 order and Q final retention/lineage decisions. Do not rerun a Q controller under the new text vocabulary: that would change observation selection and confound this text-only test. If aggregate recovery is impossible, mark that source/scene vocabulary test unavailable rather than re-encoding images or inventing embeddings. For N0 preserve its canonical reference phrases. Encode each vocabulary in its model's own text space. Every tensor explicitly carries model and ordered label/text identity.

Methods: N0, Q_GAIN, static S2, original M2_RAW, original M2_CAL, and the CAL-selected simple comparator. No new fitting. For each, evaluate:

- Original names.
- `a photo of {name}` for every existing class.
- `a close-up photo of {name}` for every existing class.
- Original names unchanged plus the16 frozen distractors in `PROTOCOL_SPEC.json`.

Do not alter the distractor list after inspecting results. Prompt tests preserve the original label IDs. The expansion test retains all original true labels and adds only distractor columns; if a distractor wins, count it as an incorrect object classification, not an ignored success. Keep expansion results in an explicit vocabulary-stress evaluator, **not** the official AP table that would discard invalid label IDs. Report original-label accuracy, original-label-vs-original-label argmax changes, distractor-selection rate, NLL, and coverage on the same identifiable objects. State that this is a controlled word-set test, not proof that all added names are truly absent from every possible region.

Prompted or expanded vocabulary scores intentionally differ from frozen historical source top1, so construct a new vocabulary-evidence type. Do not misuse the old top1-parity check or alter original source bundles. Temperatures remain frozen from the original ScanNet task. Save raw text, text-feature identity, class mapping and per-object score/probability records. Batch CPU text encoding or use the existing GPU lock; both are counted. No image re-encoding.

Add a focused mathematical/fixture check that softmax over a single queried text is identically1. The current M2 formula is not by itself a meaningful arbitrary single-text retrieval score. Do not invent a new retrieval head or claim this study has validated one. Record this limitation in the claim ledger; a future retrieval formulation is outside this execution scope.

## 12. Genuinely new confirmation, if local data supports it

Core, query controls, ranking, pooling, attribution, and vocabulary work may not be blocked by fresh-scene availability.

During binding, inspect only actual authorized local ScanNet inventory and prior experiment scene/exposure manifests. Exclude every original FIT/CAL/SELECT/CONFIRM physical family, every family appearing in repository experiment results or runtime logs, and any unknown-exposure family. Do not infer freshness merely from absence in one short list. The explicit14-family exclusion in JSON is a minimum, not the whole exposure audit. Inspect lightweight metadata/receipts; do not rescan all heavy arrays or download data.

If at least four complete, locally authorized, previously unexposed physical scene families remain, sort by SHA256 of UTF-8 `M2_REVIEW_FRESH_V1\0<family_id>` and take the first four. Record chosen IDs, completeness criteria, exposure sources, and known limits before RGB export, annotation conversion or model inference. Otherwise write `FRESH_BLOCKED_INSUFFICIENT_LOCAL_UNEXPOSED_FAMILIES`, with counts and missing requirements. Do not substitute the old holdout pair, duplicate captures, or renamed frames. No new ScanNet download is authorized by this pack.

Freeze one confirmation plan containing N0, **unchanged published M2_CAL**, the CAL-selected simple comparator, and the CAL-selected strongest single alternative. At most four methods. Confirm these same identities on all four scenes regardless of what new results show. The workflow must include real export/capture, native/S2 source formation, frozen Q execution, both rankings and pooled/per-scene evaluation; a manifest-only placeholder is not implemented confirmation.

Reuse the upstream native capture implementation under a **new study-local authorization contract**. Old composition/replica contracts hard-code their own scenes and methods; do not bypass them or overwrite their locks. Keep the same200-slot ScanNet schedule recipe and model precision. Finish all fixed predictions before examining labels/metrics to make adjustments. A runtime defect may be fixed with an explicit repair receipt and all affected frozen methods recomputed; then disclose any resulting exposure. No confirmation-based temperature, source, seed or comparator changes.

Do not require positive historical gains to open this one fixed confirmation: its purpose is to test the target, not to select favourable scenes. If confirmation fails, retain and publish it. Do not start another round within this task.

## 13. Implementation mapping and runnable workflow

Use a small task-local package `src/static_ovmap/m2_reviewer_study/` and runner `scripts/evaluation/run_ovimap_m2_reviewer_study.py`. Suggested separation is binding/scores, ablations/calibration, query controls, evaluation/diagnostics, vocabulary, confirmation and reporting; do not build a new general experiment platform.

| New responsibility | Existing code to reuse | Essential addition |
|---|---|---|
| Bind frozen sources and score vectors | composition `object_evidence.py`, `calibration_jobs.py`; replica `jobs.py` | Runtime cache locators and raw-N0-cosine reconstruction |
| Source/temperature variants | `label_fusion.py`, `temperature.py` | Subset fuser, hard vote, shared-temperature objective and fold handling |
| Causal query comparisons | `CapturedFrames`, `CurrentLineage`, `FeatureStore`, query ranking, `VisualRequestLoader` | Study-local B200 random policy and dataset-generic budget runner |
| Whole-map predictions | `load_prediction`, `relabel_prediction`, immutable `PredictionPayload` | New method IDs, unchanged source masks/ranks |
| Evaluator views | Released adapter, `trace_released_matches`, official `map_pred_mesh`/`evaluate` | Evaluation-only reranking; truly pooled dataset calls |
| Object/probability evidence | correspondence helper and true matching traces | Full-score diagnostics and fixed calibration populations |
| New-scene stage | native capture/export functions | New exact scene/access contract; no bypass of old contracts |
| Reports and publishing | existing lightweight JSON/NPZ/CSV writers | Real phase outputs; no recursive old-publication copy or rehash |

Phases: `bind`, `core`, `query-controls`, `diagnostics`, `robustness`, `fresh`, `report`, `publish`, and `all`. Every phase must have a real callable implementation. A blocked prerequisite affects descendants only. `all` must run required work, evaluate, report and publish; it must not spend its final time repeatedly refreshing progress or waiting indefinitely on a lock.

Internal sequencing:

1. Bind all source identities and fresh inventory; run only the focused interface checks.
2. In `core`, reconstruct CAL, fit folds, evaluate CAL variants, freeze comparator choice, fit final scalars; then evaluate eight Replica variants and controls with both rankings/pooling.
3. In `query-controls`, complete CAL control trajectories/scalars and freeze the curve trigger before new Replica control evaluations; then execute required Replica rows and gated curves.
4. Complete object, probability, overlap and cost analyses; vocabulary tests.
5. Freeze and execute the one fresh plan when its asset requirement is met; otherwise document its concrete block.
6. Validate report arithmetic/identities once, publish, and return final state.

Example command **after implementation** (the runner is proposed, not already present):

```bash
PY=/home/ww/miniconda3/envs/ovimap-map/bin/python
"$PY" scripts/evaluation/run_ovimap_m2_reviewer_study.py \
  --spec docs/paper/static_ovmap/m2_reviewer_study_v1/PROTOCOL_SPEC.json \
  --source-transfer /mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001/transfer.json \
  --output-root /mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001 \
  --phase all --resume
```

Resolve different actual environment paths from the input runtime when necessary; do not run all models inside the coordinator's environment. Emit stage elapsed time and completed/expected real rows. Resume compares exact consumed dependencies, not path names. An interrupted export is resumed or repaired in the new output root; it does not rerun every visual stage.

## 14. Focused validation, not excessive safety work

No test-count target and no full repository regression. Use only these ten compact fixture groups:

1. Exact legacy three-source fusion parity; available-label argmax and ID permutation.
2. Source dropout, fallback exclusion, A3 all-unavailable class0, hard-vote ties.
3. Shared and per-source objectives/folds/support fallbacks; no Replica fitting.
4. Canonical-to-cosine reconstruction, positive-temperature top1, boundary/tie disclosure.
5. Immutable owner/ranks versus separate evaluation-only native reranking.
6. Dataset pooling versus scene mean; class absence and duplicate/ignore parity.
7. Causal query prefix, quota-zero combine side effects, no feature access before debit, irreversible lineage drops.
8. Exact model/crop cache identity, logical versus physical/union costs, random seeds and no duplicate attempts.
9. Vocabulary extension/ID isolation, technical abstention, singleton-softmax limitation.
10. Selection/fresh authorization and stage status/report dependency handling.

Use one existing real source sample/scene for no-new-image legacy parity, and one real official-export/pool parity check. The actual experiments are not tests to be replaced by fixtures. A fixture failing due to an actual interface defect must be fixed; do not waive it or expand into unrelated fuzzing. Run lint/compilation only on changed paths. Reuse previous environment verification.

Audit the final measured row matrix, objective settings, ranks, aggregation identities, calibrated/example populations, and sums once. Do not repeat all old77/199 tests, inspect every old16,430 artifact, or run broad CROVE dynamic regressions. No repeated GPU stress tests, environment reinstalls, or mandatory multiple identical full experimental passes.

## 15. Reports and scientific decisions

Mandatory tables:

- **A — Core ablations:** five controls/eight variants; both rankings, scene means and official pools; per-scene changes and availability.
- **B — Probability/score mechanism:** RAW/.01/shared/per-source/cosine, NLL/Brier/ECE, fit counts/bounds and representation effects.
- **C — Actual object events:** corrections, harm, source overlap and unused complementarity; office1 loss and other failures.
- **D — Query and cost:** standalone/fused B200 policies and random replicates; triggered curves; realized costs, dependency union and physical reuse.
- **E — Vocabulary/freshness:** prompt/word-set stress and the four fresh-scene results or exact unavailability status.
- **F — Claim ledger:** each proposed contribution, its matched control, evidence, limitations and retain/simplify/unsupported decision.

Do not automatically declare three-source necessity when gains are within observed scene variability. Report effect sizes, loss distribution and limited uncertainty. No forced p-value threshold, no post-hoc choice of a favourable aggregation. An AP-positive/mIoU-negative result is a tradeoff, not failure of execution. A null metric has a reason and denominator; never fill it with0 or a historical number.

The claim ledger must separately decide:

- whether N0, Q and S2 each add useful evidence beyond the best CAL-selected subset;
- whether soft distributions outperform hard voting;
- whether source-specific fitting improves over common sharpening/sharedT;
- whether native score representation creates a material hidden weighting effect;
- whether learned query selection improves its matched nonlearned alternatives;
- whether gains survive official reranking and **dataset pooling**;
- whether probability calibration improves under transfer, rather than only top1 decisions;
- whether the current formulation supports more than fixed-vocabulary recognition;
- whether new-scene evidence and cost justify broader claims.

Conclusions may be `SUPPORTED_IN_THIS_STUDY`, `MIXED`, `NOT_SUPPORTED`, or `NOT_TESTED_PREREQUISITE`; never convert the last into a negative method result. M2 may be demoted in favour of a simpler scientific explanation, but no new deployment model is silently installed.

## 16. GitHub publication is part of completion

Commit scoped code, task spec, real resolved configuration with no credentials, scalar parameter files, all small metric tables, object/cost ledgers, and:

- `docs/paper/static_ovmap/M2_REVIEWER_RESULTS.md`
- `docs/paper/static_ovmap/M2_REVIEWER_HANDOFF.md`
- `docs/paper/static_ovmap/M2_CLAIM_LEDGER.md`

Small result destination:

`artifacts/static_ovmap/m2_reviewer_study_v1/<run_id>/`

Keep raw images, model weights, full surface arrays and large per-point probabilities on shared storage. Store real paths, sizes, SHA256 identities and exact rebuild commands. A path manifest is not a claim that large bytes were uploaded. Include compressed object ledgers, compact per-method source references, scalar fits and metrics; do not copy the entire old release. Use gzip/NPZ for repetitive arrays; target the new tracked artifact bundle below100MiB. Preserve principal metric/selection JSON bytes. Verify only the newly created package and consumed dependencies once.

Create a normal commit and push this task branch. Resolve the remote URL from the authenticated repository. Never force-push or alter unrelated files. After pushing, compare full `git rev-parse HEAD` with `git ls-remote origin refs/heads/research/ovimap-m2-reviewer-evidence-v1`. Only equality is `PUSH_VERIFIED`. Record the final SHA in an external publication receipt to avoid recursive self-reference; a previous commit may contain a pending receipt. If authentication truly fails, preserve local commits and report the actual error and exact push command; do not claim publication or create another remote.

Final response must report implementation, measured core row count, query-control/curve status, probability/aggregation conclusions, fresh status, scientific recommendation, actual inference/scalar-fit/evaluation costs, changed-file tests, report paths, branch and full verified remote SHA. Distinguish study completion from a positive novelty or performance conclusion.

## 17. Final pre-delivery self-check for Codex

Before the final report, verify these concrete facts once:

- All executable required rows contain actual measurements, not smoke receipts.
- CAL nomination and any curve gate were fixed before new evaluation results; legacy CAL exposure is disclosed.
- The A3 fallback, A7 two-row distinction, RAW/CAL controls and shared loss match this specification.
- Every positive benchmark statement identifies fixed/current-class ranks and macro/pooled aggregation.
- Probabilities, source availability and full owner ledger explain identical outputs and gains/losses.
- Query controls run true current-frame selection under their own history; three random replicates are not combined into one lucky result.
- Fresh scenes are genuinely separate or explicitly unavailable; old confirmation is not recycled.
- Artifacts, MD reports and code are committed; the actual remote SHA was checked.

Do not execute another broad audit after these checks pass. Deliver the result.
