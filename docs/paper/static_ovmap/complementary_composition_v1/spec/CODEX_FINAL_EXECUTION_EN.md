# FINAL CODEX EXECUTION CONTRACT — Complementary composition on OVI-MAP

**Version:** 2026-09-26-reviewed-v1  
**Repository:** `Orangekostar/oviovo`  
**Reviewed starting commit:** `498f2c5a2fa511c919a83ff9c27946e37cb46a0f`  
**Task branch:** `research/ovimap-complementary-composition-v1`

Read this entire file and `PROTOCOL_SPEC.json`. They define one bounded implementation and experiment task. `SOURCES.md` supplies pinned evidence; `CODE_REVIEW_AND_AUDIT_ZH.md` explains design decisions. A JSON field marked `specification_only` is not a runnable path configuration: resolve it into `resolved_config.json` first. Do not interpret past `NO_NET_GAIN` or a past standalone retention gate as a prohibition on this newly authorized composition experiment.

## 1. Objective, scope and what constitutes completion

Implement and measure whether complementary **predicted evidence**, rather than per-scene or ground-truth-based cherry-picking, improves one final OVI map. Keep the native surface, instance owners, projection and serialized instance ranking fixed. Study three types of composition:

1. Parallel object-level fusion of N0, native-model Q_GAIN, and static SigLIP2 AREA evidence.
2. A fixed native query trajectory re-read by SigLIP2, without feeding SigLIP2 back into query selection.
3. One genuinely replayed, shared-state, B200 mixed query policy; optionally re-read its fixed trajectory with SigLIP2.

Deliver actual predictions, released-evaluator results, object-level explanations, a frozen nomination, a bounded confirmation, and a normal verified GitHub push. Do not stop at another plan, adapters-only smoke, all-null table, or an implementation-complete declaration based only on tests.

Authorized: reuse existing native-v10 captures, model snapshots, Q_GAIN checkpoint/standardizer and result caches; acquire missing native/SigLIP2 region features for the listed requests; fit only the specified scalar temperatures; execute the listed composition trials. A new, independent confirmation contract authorizes processing the **two already acquired raw confirmation scenes**, after nomination is frozen. No new raw-data downloads or foundation-model downloads are required.

Not authorized: new SF/WOW/SAM/other model trials; backbone or selector/Q head retraining; G geometry changes; NMS/rank changes; global hyperparameter, prompt, mixture-weight, ratio or seed searches; replaying all 12 development mappings; relabeling old scene roles to evade guards; changing the old frozen selection; reconstructing results from GT; reading confirmation outcomes to choose a method; whole-repository dynamic-CROVE tests; repeated large release audits.

Keep old results and branches intact. All new choices below are **v1 engineering hypotheses**, not established model specialities or guaranteed gains. In particular, Q and N0 share a visual model, and Q/SigLIP2 errors may be correlated.

## 2. Evidence that determines the task

At the reviewed revision:

- SELECT N0 averages are uAP 25.740741% / mIoU 35.334443%. Q_GAIN averages 32.028620% / 43.621312%, but loses 3.333333 uAP points on one SELECT scene against Q_COMBINE. Static SigLIP2 AREA averages 27.483165% / 36.423060%. These are descriptive two-scene results, not a population estimate. [SRC02–03]
- Old combinations require independent module retention. The S-only hybrid also performs a fresh native static reread before applying a semantic refinement. It therefore does **not** implement preserving Q's existing evidence and fusing it with a second source. Do not call the old combo and relabel it M4. [SRC04]
- Q_GAIN is a native-model state-dependent ranker; its later requests depend on earlier paid features. Simply replacing `NativeQueryLoader` with SigLIP2 changes the controller's input distribution and often its trajectory. [SRC05,08–10]
- Current lineage follows segment memberships, permanently drops ambiguous/evicted features, and does not treat matching integer owner numbers as sufficient identity. [SRC06]
- Q retains up to ten features, fuses their **unnormalized six-crop means** with visible-overlap weights, and exports class 0 for no evidence. N0 instead uses its original native last-eight-of-retained/top10 readout and minimum-observation rule. Current static S2 AREA also preserves raw view-mean magnitudes (this was already repaired at the reviewed revision), but it uses a different top-three view pool. Preserve each audited single-source control and its view pool; do not reapply an obsolete normalization fix. [SRC07,14–15]
- Static SigLIP2 can fall back to N0 when no real request succeeds. That fallback is **not** a second independent vote. The repaired image backend is `rgb_siglip.FrozenSiglipBackend`, explicitly HWC. [SRC11–13]

No new benchmark was run to prepare this contract. Runtime filesystem availability must be verified from actual receipts; the source paths below are not invented proof of readability.

## 3. Workspace, source binding and execution interface

Create a new worktree beside the existing project. Base the new task branch on the reviewed commit. If an existing branch is its descendant with compatible task receipts, resume it; otherwise make a clearly named sibling with the starting HEAD suffix. Never reset, stash, delete or force-push unrelated user work.

Default output root: `/mnt/shared/ww/ovimap-complementary-composition-v1`. Use compatible `attempt_001`, etc.; incompatible scientific inputs require a new attempt, not overwritten old results. Store this instruction bundle under `docs/paper/static_ovmap/complementary_composition_v1/spec/`.

Create a real public entry point:

```bash
python scripts/evaluation/run_ovimap_composition_study.py \
  --spec docs/paper/static_ovmap/complementary_composition_v1/spec/PROTOCOL_SPEC.json \
  --phase all \
  --output-root /mnt/shared/ww/ovimap-complementary-composition-v1
```

Phases: `bind`, `prepare-cal`, `calibrate`, `compose-cal`, `freeze`, `regression`, `confirm`, `report`, `publish`, `all`. Leaf phases accept `--resolved-config` and resume compatible artifacts. `all` must invoke executable jobs and finish with an exit code reflecting task status. A receipt loader is not an inference/evaluation handler. `report` derives state and tables from actual completed leaf outputs, never `measured={}` or hard-coded COMPLETE/BLOCKED constants.

Known binding sources:

- `configs/evaluation/ovimap_module_scannet_runtime.json`
- `configs/evaluation/ovimap_module_scannet_study.json`
- `docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md`
- `artifacts/static_ovmap/module_validation_v1/finalization-20260923-native-v10/selection/selection.json`
- source study root `/mnt/shared/ww/ovimap-module-validation-v1/scannet_study_v1`
- source native root `/mnt/shared/ww/ovimap-module-validation-v1/scannet_runtime_v1`
- raw-data lock `/mnt/shared/ww/ovimap-module-validation-v1/data/scannet/acquisition_lock.json`
- source Q checkpoint and standardizer from `query/calibration_receipt.json`; follow its exact paths and verify hashes, never select another epoch.

Bind model/processor/tokenizer, text `class_names` and non-contiguous `valid_ids`, dataset vocabulary, original native pickle, N0 manifest/arrays, frozen source-to-GT projection, per-scene captures, static S2 request manifests/records, Q decisions/cache receipts, evaluator files and the existing Q checkpoint. Resolve explicit overrides first, then real configs and linked receipts, then one bounded search under their discovered roots. No machine-wide crawl.

Create a source-asset index once. Reuse **read-only** source prediction/record readers; do not run old high-level pipelines merely to read results or change their `study_root` in place. A new worktree path is not a reason to overwrite old receipts or rehash every old artifact on every phase. Store new outputs separately even when sources point into the old workspace.

Use the existing native-v10 extension, pinned upstream `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`, repaired RGB backend and isolated Python environments. Resolve the actual GPU from the source runtime (recorded as GPU 2); use **the source study's existing global GPU lock path**, not a different lock with the same basename under the new root. Do not launch overlapping visual workers or interfere with another task.

## 4. Data roles and a new confirmation authorization

Use this fixed, small composition study, not another 12-scene training run:

| Role | Scenes | Use |
|---|---|---|
| COMPOSE_CAL | `scene0056_00`, `scene0534_00` | Temperature calibration, M6 trigger, combination nomination |
| REGRESSION_ONLY | `scene0445_00`, `scene0626_00` | Run every configured composition **after freeze**, report previously exposed scene behavior; no model choice |
| CONFIRM | `scene0553_00`, `scene0064_00` | One pre-nominated composition and declared controls, after the new lock |
| Historical FIT | existing eight FIT families | Q-head provenance and existing records only; no new visual sweep or head fitting |

The CAL and regression scenes have already been exposed in earlier studies, and the fixed Q checkpoint was itself selected using historical CAL. Even leave-one-scene-out temperature calibration does not undo that exposure. Call these development comparisons, not fresh generalization estimates.

The earlier report said the two CONFIRM raw scenes were acquired but not exported/evaluated. Check the current known receipts before opening their images or labels. If they have since been used for selection elsewhere in this project, mark the confirmation evidence as exposed, do not invent a replacement scene or call it untouched, and continue CAL/regression. If either scene has such exposure, set `BLOCKED_CONFIRMATION_EXPOSURE` and do not open either scene for this confirmation; finish CAL/regression and publication. A specific exposure/asset block affects confirmation, not every runnable trial.

This task explicitly creates a **new** confirmation contract, separate from `module_validation.confirmation_access`. Preserve the old gate and old `N0/NO_NET_GAIN` result. Do not fake a non-N0 old selection, change old role labels, monkeypatch its guard, or use a `--force` bypass. Implement a new contract validating exact nominated method(s), sources, settings, code identity and two scene IDs.

After freeze, authorize the nominated composition plus `N0`, `Q_COMBINE`, `Q_GAIN`, `S_SIGLIP2_AREA` on both confirmation scenes; include M5 only if M6 is nominated. This is at most six output methods per scene. Supporting controller features can be generated for the nominated method but are not extra selectable candidates. No per-scene nomination is permitted.

Unlike the old standalone rule, **a technically complete, effective-change CAL nominee is permitted one confirmation even if CAL shows a tradeoff**. This is a bounded transfer/falsification check, not automatic deployment. If there is no effective candidate or a genuine technical blocker, record the exact reason. Do not consume confirmation merely to search all combinations.

## 5. Shared immutable prediction and source-evidence contracts

### 5.1 Geometry and evaluation identity

For every scene, use exactly its stored N0 surface coordinates/faces, native-painted `owner_ids`, projection, non-contiguous ScanNet200 class list and frozen `.6f` rank values. Use payload branch `S` for M1/M2 and `Q` for M3/M4/M5/M6 so the existing fixed-owner/fixed-rank checks remain active; do not weaken them by routing through an unchecked generic branch. Reuse `PredictionPayload`, `load_prediction`, `relabel_prediction`, and invariant checks.

**Do not substitute raw voxel owner labels or G_ORIGINAL for N0.** N0's original mesh paint and readout/export conventions are already reconciled. Preserve every native owner, including technical-fallback or unmodified objects. Class 0 is not an extra vote or a background-removal strategy. Do not re-rank after relabeling; do not remove small masks to repair AP.

Before composing different sources, verify geometry, owners, source rows, vocabulary and ranks agree. A mismatch is an affected-source binding error, not permission for an unreported nearest-neighbor bridge. AP may legitimately change when labels enter/leave the evaluated class set despite fixed shapes and scores.

### 5.2 Per-object evidence table

Create one schema with scene ID, stable owner, source identity, label, real-evidence availability, fallback reason, used request IDs, retained request IDs, complete class-score vector where defined, text-space identity, readout identity and source receipts.

- **N0:** recover the exact saved-native readout for the stored owner, including original ordering/minimum observations. For soft fusion, reproduce its actual FP32 canonical-relative vector from the saved native aggregate and text/canonical embeddings, not an invented one-hot vector. Its argmax must reproduce the frozen N0 label with original tie handling.
- **Q_GAIN / Q_COMBINE:** recover the **post-final-reconciliation** feature set and raw cosine vector using the existing query readout. Nonempty real retained features and the actual nonzero export determine availability. Do not use an intermediate `after_scores` vector as the final object score.
- **Static SigLIP2 AREA:** recover the existing `S_SIGLIP2_AREA` readout from its verified top-three request manifest and own text cache. `technical_fallback=True` or zero successful views means unavailable to fusion, even when the final label equals N0. Preserve its repaired raw-mean AREA fusion and its static top-three view pool; do not substitute Q's top10 request pool.

Reconstructing scores from cached features is allowed. Never reconstruct full scores from a top1 label or margin. Source score scale differs between N0 canonical-relative values and cosine values; calibration is source-specific. A source reconstruction that changes its frozen top1 must be explained/fixed from the actual implementation before that source participates in M2; do not tweak scores to force a match.

Native and SigLIP2 image features never share a vector space, model cache namespace or query-state feature list. Being 1024-dimensional does not establish compatibility. `FeatureStore` caches by request within a store; create separate stores per visual model.

### 5.3 Object complementarity diagnosis

Once the source predictions are locked, use the existing evaluation-only strict-IoU>0.5 object correspondence helper to tabulate all available patterns of N0/Q/S2 correct/incorrect. Include unavailable sources and unmatched/ambiguous native objects as separate categories. Keep all those objects in full-map evaluation.

Distinguish: N0-only correct; Q-only correct; S2-only correct; Q and S2 correct against N0; Q and S2 jointly wrong against a correct N0; all sources wrong; and an adequate shape not available. Record prediction agreement even where GT correspondence is unavailable.

An oracle-correctable **object count** is diagnostic, not an AP upper bound or a deployable oracle prediction. Do not choose subjects, weights, routes or new prompts from GT cases. This table does not gate the mandatory trials.

## 6. Mandatory methods: five mechanisms, six output configurations

| Name | Method ID | Definition |
|---|---|---|
| M1 | `CP_M1_AGREE_KEEP` | Real Q_GAIN and static S2 AREA agree; otherwise keep N0 |
| M2_RAW | `CP_M2_EQUAL_RAW` | Equal probability mixture, fixed T=0.07 for each source |
| M2_CAL | `CP_M2_EQUAL_CAL` | Identical sources/weights; separate scalar temperatures |
| M3 | `CP_M3_COMBINE_S2` | Frozen Q_COMBINE attempted trajectory, SigLIP2 reread |
| M4 | `CP_M4_GAIN_S2` | Frozen Q_GAIN attempted trajectory, SigLIP2 reread |
| M5 | `CP_M5_MIX50_NATIVE` | Shared-state native B200 mixed query replay |
| M6 (conditional) | `CP_M6_MIX50_S2` | Frozen M5 attempted trajectory, SigLIP2 reread |

M2_RAW is the single inexpensive added ablation that tests whether temperature calibration adds anything. Do not add more weights, confidence thresholds, majority rules or ratios.

### 6.1 M1: agreement with an incumbent fallback

For each native owner `i`:

```text
if Q_GAIN and S2_AREA both have genuine usable evidence
   and q_label > 0 and q_label == s2_label:
       final_label = q_label
else:
       final_label = N0_label
```

Log same-label agreement separately from an actual replacement. A source fallback cannot trigger agreement. Disagreement cannot trigger a search for a preferred alternative. This is cross-source agreement, **not** a calibrated safety guarantee or statistically independent votes.

### 6.2 M2_RAW and M2_CAL: class-evidence fusion

Align source score columns by the full official `valid_ids`, never by an assumed contiguous ID or a predicted class subset. For available source `m`, use

`p_m(c) = softmax(z_m(c) / T_m)`.

Use the arithmetic mean of the available distributions. All available sources have equal weight; absent sources receive no fabricated score row, and the remaining weights sum to one. If no source is available, retain N0. Stable `valid_ids` order breaks exact final ties. No KEEP bias or acceptance threshold is added.

For M2_RAW, every `T_m=0.07`. It is an explicitly **uncalibrated mixture control**, not an assertion that this scale is fair or optimal. For M2_CAL use section 7. Feature aggregation and all availability bits are identical between the two rows; only temperatures differ.

M1 and M2 are **static, multi-branch** methods. Their required source inference is not free merely because it already exists on disk. They are not B200-only methods.

### 6.3 M3/M4: fixed trajectory, replacement recognizer

Reuse or generate the native `Q_COMBINE` and native `Q_GAIN` B200 controllers, retaining their existing encoder, frozen Q_GAIN checkpoint/scaler, current-state features, tie rules, frame schedule, successful/failed attempts and end-of-frame barriers.

Freeze each controller's complete ordered **paid-attempt** trajectory. Save request IDs, order, frame indices, mask/union/bbox/image identities, controller outcomes and the trace hash. Do not rebuild a trajectory from its final top10 IDs, sort it by final owners, choose three views, or skip a native attempted request simply because it was later evicted.

Run a second, forced-choice replay over that exact attempted sequence with a separate SigLIP2 feature state:

1. Load the matching current frame and the same region/bbox/union with `CapturedFrames`.
2. Advance the second stream with the same temporal segment-membership snapshots using its own `CurrentLineage` and its own text space. It is a readout stream, not a query ranker.
3. Attempt all controller-selected requests in the same per-frame order. SigLIP2 uses the fixed local snapshot, explicit HWC backend, same six crop **images**, its own processor/text embeddings and FP32. No new target mask construction, background crops, prompt or name mapping.
4. Admit that frame's results after one barrier. Apply the existing top10 overlap/frame/request retention and irreversible split/eviction behavior. A SigLIP2 technical failure does not cause a substitute request or a fallback native feature.
5. After the final frame only, reconcile to the actual final numerical membership as the original Q exporter does. Restrict output to the fixed N0 owners; fuse raw six-crop means with overlap weights and final L2 normalization, as Q does. With no usable S2 evidence export class0, **not N0 fallback**. This maintains the existing query-readout experiment semantics.
6. Export the final label vector on exactly N0 geometry and ranks. Record successful/failing reread differences and final retained request IDs.

First validate the forced replay with native cached features on one real CAL trajectory: exported labels and retained IDs must reproduce the original native trajectory result. This is an integration check, not another model condition. Reuse that work in formal execution.

The second stream must never change the controller's later selections. Native Q_GAIN was not trained on SigLIP2 scores. These are **native-controller trajectory + separately paid S2 reread** experiments, not an already validated online S2-controller system. Count up to B200 native controller requests **plus** B200 S2 attempts per scene.

Report the 2x2 contrasts using consistent readouts:

- M3 minus native Q_COMBINE: recognizer effect on the combine trajectory;
- M4 minus native Q_GAIN: recognizer effect on the gain trajectory;
- M4 minus M3: controller-trajectory effect with S2;
- `(M4 - Q_GAIN) - (M3 - Q_COMBINE)`: interaction in each metric, in percentage points.

Do not add AP increments as if AP were an additive per-object loss.

### 6.4 M5: one shared-state, work-conserving mixed controller

Implement a new replay that uses one `QueryPolicyState`, one `CurrentLineage`, one native-model `FeatureStore`, one `NativeCombineState`, and a total budget of 200 attempted requests. Do not concatenate two independently produced old trajectories or grant each lane B200.

At each scheduled frame, using only state available before this frame's queries:

1. Load/advance/register exactly as `replay_captured` does. Exclude already paid request IDs.
2. Call `native_combine_candidates` **once** on all current not-yet-paid candidates. Preserve its original coverage-before-area mutation, even when frame quota is zero. It is not a pure predicate to call repeatedly while filling slots.
3. Build `R_C` by existing Q_COMBINE ranking of that admitted subset; build `R_G` by existing Q_GAIN ranking of the full candidate set using the **mixed policy's own** native paid history and the frozen checkpoint/scaler. Both rankings are computed once from the same pre-batch state.
4. Quota is the existing `max(0, floor(B*(f+1)/F)-spent)`. For the kth actual attempt slot, starting at 1 globally, odd slots prefer `R_C`, even slots prefer `R_G`. Choose the highest-ranked unused request in the preferred list; if exhausted use the other list. Remove a selected ID from both lists. If neither has a request, end the batch; no cost is charged for an empty slot.
5. Debit the complete selected batch before any feature access; acquire only selected requests; admit all results at the frame barrier. No same-frame result changes a pending slot's ranking.
6. Every successful mixed query updates the shared object evidence and the successful-overlap history used by subsequent combine gates, regardless of which lane selected it. Call lineage recording once. No duplicate appends for a single success, and no refunds after later feature eviction.
7. Export the existing native Q readout at the end; no N0 class fallback where Q has no evidence.

The rule is **50:50 preferred slots, not a promise of exactly 100 realized selections per lane**. Log preferred lane, actual winning lane, fallback reason, admitted subset, rank position, frame quota, used budget and owner. If combine frequently lacks candidates, the actual mix may be gain-heavy; report it, do not silently impose another ratio.

This is a new policy with frozen weights under a changed history distribution. Its value must be measured, not inferred from the two parent scores. Training a new gain head or fixing the old combine side effect is outside this round.

### 6.5 M6 trigger and definition

Only COMPOSE_CAL may trigger M6. Define `weak_dom(A,B)` as mean uAP and mean mIoU each no smaller within `1e-10`, with at least one strict increase greater than `1e-10`.

Run M6 iff:

```text
(weak_dom(M3, Q_COMBINE) OR weak_dom(M4, Q_GAIN))
AND weak_dom(M5, Q_COMBINE)
```

All referenced metrics must be finite on both CAL scenes. Persist the exact inputs/decision. This does not require old standalone retention or every-scene dominance. If not triggered, M6 is `NOT_REQUIRED_BY_COMPOSITION_CAL_GATE`, with the measured reason.

When triggered, M6 performs the same forced S2 reread as section 6.3 on the fixed M5 trajectory. It does not add an online expert router. M6 is also run on regression scenes; on confirmation it runs only if nominated.

## 7. Scalar calibration without using the regression/confirmation outcomes

No new Q or semantic selector head is trained. Fit at most three temperatures per fit invocation: one for N0, one for native Q_GAIN, one for static S2 AREA.

Build supervised calibration targets **only after source predictions are locked**, in an evaluation-side file. Use the existing unique strict-IoU>0.5 geometric object correspondence with a valid semantic GT class, independent of predicted label. Include both initially right and wrong objects. Each source uses only its genuinely available examples. Unmatched objects are omitted only from temperature fitting, not from inference or full-map evaluation.

Use leave-one-COMPOSE_CAL-scene-out predictions for selecting M2_CAL: when evaluating one CAL scene, fit each source's T on the other scene. Fit a scalar `log(T)` by minimizing scene-balanced mean negative log likelihood with stable logsumexp, using `scipy.optimize.minimize_scalar(method='bounded')`, T bounds `[0.01,2.0]`, `xatol=1e-4`, `maxiter=64`. A fit needs >=5 objects and >=2 GT classes for that source. Failed/unsupported fits use **T=0.07**, explicitly `UNCALIBRATED_DEFAULT_T`; never fabricate labels or silently omit the M2 row. Temperature bound hits must be recorded, not expanded by a search.

Select/nominating metrics for M2_CAL are those cross-fitted CAL predictions. After nomination, fit temperatures once using both CAL scenes, with each scene contributing equally, and freeze them for regression and confirmation. Do not update temperatures after either phase. Store both fold fit sets, fitted values, NLL before/after, counts, source score definition and final values. The final refit may change M2 labels; disclose this normal calibration step, do not retroactively replace cross-fitted selection scores with in-sample ones.

The previous Q model already used historical CAL for checkpoint selection. Disclose this; this procedure reduces temperature fitting reuse but does not turn CAL into a new holdout. Do not claim temperature scaling guarantees calibrated probabilities on these scenes. [EXT01]

## 8. Ordered execution, nomination and confirmation

### P0 / bind

Resolve source revisions, assets, model snapshots, repaired extension/runtime, existing source results, actual current confirmation exposure, split names and request budgets. Record a compact `binding.json`, `source_manifest.json`, `resolved_config.json` and availability table. Do not copy the entire 16,430-file old release. Missing one source blocks only methods needing it. A published checkpoint path is not accepted without its recorded content identity.

### P1 / prepare-cal

For both CAL scenes, collect N0 and S2_AREA with availability/complete scores. Restore or execute native B200 Q_COMBINE and Q_GAIN with the existing frozen checkpoint, **not random training traces**. A Q_GAIN CAL trajectory may be missing from the previous release; generate it in the new root. Collect the complementarity table after all source vectors are locked. Reuse exact old valid outputs; do not claim these are new independent model runs.

### P2 / calibrate and compose-cal

Fit fold temperatures. Implement and run M1, M2_RAW, M2_CAL, M3, M4 and M5 on both CAL scenes. Evaluate source controls in the same domain. Run M6 only under section 6.5. Finish actual metrics and effect/availability rows even for no-change outputs or negative results.

### P3 / freeze

A nomination-eligible composition has complete finite uAP/mIoU on both CAL scenes and changes at least one owner label from N0 somewhere on CAL. An all-fallback or exact N0 no-intervention method is still measured but not a substantive nomination.

Nominate one eligible composition by: highest CAL mean uAP, then mean mIoU, then smaller mean required logical six-crop **request operation** count, then order `M1, M2_RAW, M2_CAL, M3, M4, M5, M6`. Use `1e-10` only for numeric tie comparisons. Do not change weighting by observing regression. A tradeoff candidate may be nominated; preserve its actual CAL losses.

Also freeze `best_single_CAL` using the same ordering over N0, Q_COMBINE, Q_GAIN, S2_AREA (fixed tie order as printed). This comparator is not a per-scene choice. Save all source means and a `calibration_recommendation` that distinguishes gain over N0, gain over best single, and tradeoff. Do not call every method above N0 a successful composition if the best single is better.

Persist M6 gate, nominee, final temperatures, required confirmation methods, source/model/checkpoint/vocabulary identities, code file hashes, schedules, cost definitions and settings into one immutable `selection.json`. Make an implementation commit A and record its full SHA in the freeze. Store the lock outside the tracked file whose own commit would otherwise self-reference. Later report-only commits do not invalidate identical science; cache keys depend on actual scoped source/input content, not merely a new HEAD.

### P4 / regression

Run all mandatory configurations, plus M6 iff its CAL gate opened, on the two previously exposed SELECT scenes. Use final frozen temperatures. Report those scenes as regression, not a new SELECT set and not evidence used for nomination. Do not change the nominee if another combination looks better here.

Mandatory new composition rows before confirmation: **6 methods x (2 CAL +2 regression) =24**. Conditional M6 adds four. Source control rows are additional and may be identity-reused. Different method records with exactly equal final predictions may reuse evaluation; log both records and their shared prediction identity.

### P5 / confirmation

Validate the new composition contract before exporting either held-out scene's RGB-D or labels. Under this new authorization, reuse the old sensor export, annotation conversion, native commands and six-crop workers on their already acquired raw data with **exact preselected schedules**. At most two new native captures; do not remap the four CAL/regression scenes.

The old capture wrapper only permits old development/confirmation contracts. Implement a narrow `composition_study/capture_bridge.py` that validates the new contract and executes the same per-scene job recipe from `scannet_runtime` for the two explicitly authorized rows. Reuse helpers/command construction where possible. Do not invoke the old high-level guard with falsified roles. Keep old guard behavior unchanged. Store new exports/captures under the new root (an immutable raw-data view may reference existing scans/locks); retain original role/schedule provenance.

Build each held-out N0 via its real saved native readout and native-paint/rank path; do not fill in old N0 averages. Compute its projection once, use it for all methods. Pay for required controllers and sources. Evaluate only the frozen nominee and declared controls. Run and report one confirmation even if CAL nomination was a tradeoff, provided it had a real intervention and is technically executable. No alternative trial/ratio/teacher after reading these outcomes.

Report separately:

- mean metric gain/tradeoff versus N0;
- mean gain/tradeoff versus the frozen best single and the method's matched parent;
- worst-scene deltas and every-scene nonnegative/strictly positive flags;
- actual logical cost and additional model use;
- two-scene evidence only, not generalization assurance.

Do not silently relax the old study's gate or claim it was passed. The new study measures mean and scene-level outcomes separately. Leave deployment unchanged; a research nomination is not an automatic replacement of N0.

### P6 / report and publish

Finish reports from actual payloads, then commit and push as section 13. Genuine partial availability produces per-method technical statuses and partial tables, not fabricated averages or a universal success banner.

## 9. Implementation map: reuse versus new code

Create a compact package `src/static_ovmap/composition_study/`; proposed files below are **new** interfaces, not claims that these already exist:

| New file | Responsibility | Audited reuse |
|---|---|---|
| `binding.py` | Read-only source bundles, data roles, source/model identity | source configs, saved receipts, `load_prediction`, `CapturedFrames` |
| `object_evidence.py` | Reconstruct actual N0/Q/S2 complete score vectors and availability | `native_readout`, `direct_readouts`, `update_cached_class_scores`, model record readers |
| `label_fusion.py` | M1, M2_RAW, M2_CAL | `relabel_prediction`, vocabulary/order checks |
| `temperature.py` | Evaluation-side target construction, fold fitting, frozen scalar files | `study_targets.semantic_events`; SciPy scalar optimizer |
| `trajectory_reread.py` | Full fixed-attempt replay and S2-specific feature/cache stream | `CapturedFrames`, `CurrentLineage`, query-state classes, repaired `rgb_siglip` |
| `mixed_query.py` | M5 shared-state scheduling and lane accounting | combine gates, `rank_query_candidates`, dispatch and native query loader |
| `execution.py` | Actual leaf jobs, dependency/resume logic and bounded costs | existing immutable IO utilities, evaluator adapter |
| `selection.py` | CAL-only M6/nomination, frozen contract | simple pure comparisons; NOT old standalone gate |
| `capture_bridge.py` | New-contract-only two-scene native recipe | sensor export, GT conversion, existing per-scene native command recipe |
| `reporting.py` | Four tables, compact exports and publication record | released trace/object helpers, prediction/evaluation manifests |

Add the public runner named in section 3 and a small resolved config. Keep implementation modular but do not build a generic plugin/experiment service or a new database. Small additive helper extraction is acceptable in the new worktree only when it preserves old behavior and provenance; do not edit the previous running workspace.

## 10. Cost accounting and cache reuse

Use two distinct ledgers:

1. **Physical work in this task:** model loads/forwards, processed crop inputs, elapsed inference, native capture/front-end work, cache hits and CPU calibration.
2. **Method-required logical work:** all source/controller/reader operations required to produce that method, even when cached. Charge failed attempts and features later dropped. Separate common native-map construction from incremental readout/composition work.

M1/M2 need N0 evidence, Q_GAIN-controller evidence and static S2 evidence. M3/M4 need the native controller plus its S2 attempted trajectory. M5 alone spends <=200 native query attempts; M6 spends <=200 native controller attempts plus <=200 S2 rereads. These are not interchangeable equal-cost results.

Do not sum N0's prior logical ledger twice merely because it is copied into an S2 record. Normalize inherited versus `added_*` fields into primitive operations. Deduplicate a logical visual operation only with proof of identical model weights, processor, dtype, crop/union mask, RGB, bbox and actual request inputs. Same scene/frame/bbox or equal output labels are insufficient. If such proof is unavailable, use conservative additive accounting and label it as such. N0 historical visual work must be listed when required by a parallel fusion; do not hide it as a free fallback.

Keep distinct model cache namespaces. Cross-directory import of an old S2 record is allowed only after canonical input identity reconciliation; record the import map. Never rename a native feature file as an S2 hit. Confirmation cannot peek at feature caches for unselected requests; only the fixed attempted sequence or new mixed policy authorizes access.

Bounds for four CAL/regression scenes: at most three B200 native trajectories per scene (COMBINE, GAIN, MIX), hence <=2400 native request operations; S2 rereads of those trajectories <=2400 with M6; static S2 <=128x3 per scene (<=1536 requests total), usually already cached. Every native/S2 visual request is six crops. No new background crops, WOW, model search or repeated FIT sweep. The two confirmation scenes add only their declared sources/controllers and at most one nominated reread stream; their original N0 capture cost is separate. Do not present these caps as predicted execution time.

For hashes: once bind each large source/model content identity and use one invocation-wide memo keyed by canonical path plus device/inode/size/mtime_ns/ctime_ns. Re-read modified files. Do not recursively hash the full source release inside every crop, owner or report row. A final manifest check should cover **new output files and actual dependencies**, not repeat previous 16,430-file publication verification.

## 11. Evaluation and required evidence tables

Use the existing released ScanNet200 evaluator, GT conversion, strict projection rule and match tracing. Freeze predictions before GT-based evaluation. Persist full-precision metrics and percentage display separately. Do not rewrite match assignment, AP thresholds, minimum regions, semantic averaging, or export eligibility to manufacture gains.

Cache evaluation by the complete prediction identity **and** evaluator/projection/GT-content/vocabulary identity. Method name alone is neither sufficient nor necessary for cache equivalence. Same owners but different classes are different predictions. If using an existing in-memory adapter whose key omits protocol content, isolate an adapter per fixed scene/protocol and extend the new persistent key; never reuse across different target files because a path string matches.

Required reports:

**Table A — complete method performance:** each scene/role/variant, source control, uAP/AP50/AP25/mIoU/mAcc, exact owner/label changes, positive-owner and evaluated-prediction counts, required logical cost, new physical cost, cache-reuse status. Show means with defined/total denominators. No comparing CAL means with regression means as a method delta.

**Table B — complementarity and actual routing:** three-source correctness/availability patterns, M1 accepted/kept reasons, M2 available source counts, teacher fallback rates, changed-object correctness and source ranks. Map masks with ambiguous GT correspondence to an explicit category. Separate class correctness from actual released TP/FP transitions.

**Table C — fixed-trajectory interaction and mixed-policy mechanism:** 2x2 contrasts, M2_CAL minus M2_RAW, M5 minus each native parent, M6 minus M5 if run; lane counts, fallback share, request overlap/Jaccard, per-owner budget, successful evidence/unknown owners, retained/dropped feature counts. A changed trajectory is not by itself a better trajectory.

**Table D — nomination, holdout and operational summary:** exact CAL nomination/gate values, frozen best single, regression-only results, holdout result and worst-scene deltas, major released-object gains/losses, scope limitations, actual costs and publication evidence. No new candidate selection after holdout.

Save all-owner decision ledgers, scored source availability, retained-request ledgers and exact released match/trace references. For unchanged AP with changed labels, inspect real eligibility, ignored/small predictions, duplicate/ranking interactions and affected classes. Do not conclude "nothing happened" from one scalar or equate object gains minus losses with AP.

## 12. Focused testing and truthful completion

Tests address only new failure modes:

- real versus fallback votes, missing-source renormalization and non-contiguous class IDs;
- temperature split isolation, positive finite T and no test-target input to prediction;
- same-model fixed-trajectory parity, correct model-cache separation and no resurrected dropped features;
- mixed unique requests, one shared budget/state, once-per-frame combine mutation and pre-batch decisions;
- fixed geometry/ranks, persistent evaluation identity, M6/CAL-only freeze rules;
- one real leaf-to-evaluator-to-report path and normal resume without empty fabricated result rows.

Use small parameterized fixtures and one real native forced-replay check; reuse it in the formal job. Run targeted tests, scoped lint and compile once after changes, then affected tests only for an observed defect. There is no test-count target. Do not rerun all old 199 tests, all old matrices, broad hardware/safety tests or full dependency audits.

Engineering fixes that change actual inputs/predictions invalidate affected artifacts with an explicit new identity. Preserve failed results and label repairs. Do not silently swap models, lower resolution, change precision, prompt, temperature range, ratio or scene membership after outcomes. A repaired confirmed method must not be claimed to have used untouched data if its earlier outcomes were inspected.

Implementation COMPLETE requires the new real runner and report path to work. Experiment COMPLETE requires every mandatory CAL/regression method to have actual complete predictions/evaluations (or explicit exact identity reuse), with only M6 allowed to be skipped by its predeclared gate. A technically unavailable input is a genuine per-method partial result, not a successful row. Scientific states include `MEAN_GAIN_WITH_SCENE_TRADEOFF`, `MEAN_GAIN_NO_OBSERVED_SCENE_LOSS`, `NO_MEAN_GAIN`, `NO_EFFECTIVE_INTERVENTION`, or `INCONCLUSIVE_<reason>`. Keep deployment unchanged and expose confirmation separately.

## 13. Mandatory GitHub delivery

Tracked reports:

```text
docs/paper/static_ovmap/COMPOSITION_RESULTS.md
docs/paper/static_ovmap/COMPOSITION_HANDOFF.md
```

Also commit the scoped source, runnable config, copied specification, focused tests, source/input manifest, scalar calibration files, selection lock snapshot, all small metric/decision/trajectory tables, and exact reconstruction commands. Store new compact evidence under `artifacts/static_ovmap/complementary_composition_v1/`.

Do **not** copy the 16,430 files of the previous release. Reference their immutable commit and input manifest. Keep old/new large model weights, RGB-D, dense owner arrays and maps in external storage with path/size/content-hash manifests. For new large ledgers, gzip once or shard by scene/method; retain readable summaries. Per-owner final label vectors plus referenced immutable N0 owners allow deterministic dense prediction reconstruction. Preserve complete new numeric results, not just favorable rows.

Prefer a compact new-only release; an oversized log does not justify rerunning science, repeatedly repacking the old repository, or postponing every result. Publish new metrics and code first with truthful external references for large data. Do not expose credentials or dump environment secrets in logs.

Record actual prediction commit A in the tracked report. After adding results/handoff make release commit B. The external `publication_receipt.json` records B, branch and remote SHA so no self-referential commit promise is needed. Tracked handoff points to that external receipt and distinguishes experiment code identity from release identity.

Execute a normal push to the task branch, then compare:

```bash
TASK_BRANCH="$(git branch --show-current)"  # must equal the task branch in resolved_config
git rev-parse HEAD
git ls-remote --heads origin "refs/heads/${TASK_BRANCH}"
```

Only matching full SHAs permit `PUSH_VERIFIED`. On failure preserve local commits and report the actual error; do not claim remote delivery and do not force-push. Do not commit directly to the prior module-validation branch.

The final Codex response must contain the measured composition table, nominated method and CAL/confirmation reasoning, inference/crop/cache counts, engineering/science/confirmation/publication statuses, exact report paths, and verified remote SHA. Never present a package audit or test pass count as a benchmark gain.
