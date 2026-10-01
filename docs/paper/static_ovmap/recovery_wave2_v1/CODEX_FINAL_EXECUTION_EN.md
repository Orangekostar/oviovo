# Final execution instruction — OVI-MAP recovery wave 2

## 0. Deliver a measured improvement attempt, not only an implementation

Develop, run, select, document and publish the study in this package. Do not stop after a
smoke test, prototype implementation, audit report, or a list of proposed next steps.
The scientific objective is better released-evaluator **APall**, while preserving AP50 and
mIoU as specified, relative to the same-input **BB00_NATIVE + D2** baseline. A complete study
can still find no net gain; report that honestly rather than reinterpreting an intermediate
metric as success. Do not turn a failed new mechanism into an alleged contribution because
an old simple control performs better.

Use repository `Orangekostar/oviovo`, starting from full commit
`c21297413954ecb7d706d1050a8e07822938ea6f`. Create a new worktree and branch
`research/ovimap-recovery-wave2-v1`. Preserve the historical branch, code, measurements,
nomination and deployment. Do not force-push, rewrite history, delete user work, or commit
unrelated changes. Resume the new branch only when its manifest agrees with this task.

Read this instruction, PROTOCOL_SPEC.json, IMPLEMENTATION_CONTRACTS.md and SOURCE_EVIDENCE.md.
The specification fixes numbers/IDs, the contracts fix algorithm semantics, and this file
fixes orchestration. Correct a genuine contradiction before the affected experiment, record
it, and apply it to the whole affected cohort. Never choose a result-dependent interpretation.
The former wave-1 prohibition on changing semantic eligibility does NOT prohibit the new,
explicitly named U experiments. All other inherited baselines remain unchanged.

## 1. Questions and authorized scope

Answer these questions with real predictions:

1. Can already reconstructed but native-ineligible instances become useful outputs using
   an existing single native observation, already paid Q evidence, or FC region recognition?
2. Can a small interpolation between FC_EQ and D2 improve the current accuracy tradeoff?
3. Does treating an uncertain association as USE_NATIVE instead of CREATE_NEW avoid the
   forced-fragmentation chain? Does a compatible many-to-one association improve further?
4. Does preserving current CropFormer candidates while using cached SAM additions avoid
   loss from partial SAM overlap? Does explicit cross-object conflict evidence help?
5. Does one pre-frozen combination outperform its components on the unchanged evaluator?

Do NOT run a new model search, new neural training, temperature refit, Q retraining, E06
text prototype search, full OVRCOAT front end, SAM3, new SAM2 predictions, depth completion,
Gaussian reconstruction, or a replay of all historical studies. This wave is not a claim
that the proposed rules are globally novel. FC/OVI/SAM operators retain their attribution.

Implement the complete conditional S2 path even if its experimental invocation is skipped
for a documented no-op. An access flag or placeholder returning baseline is not that path.

## 2. Source and asset binding before execution

Read the inspected source list. In particular inspect:

- `backbone_wave1/{readouts,semantic_readout,region_readout,features,maps,mapping_hooks,
  association,frontend_sam2,diagnostics}.py`;
- `module_validation/{scannet_study,semantic_study,native_capture,query_pipeline,
  query_study,evaluation}.py`;
- `paired_evidence_study/residuals.py`, `m2_reviewer_study/evaluation.py`;
- the full wave-1 native patch, especially beginBackboneAssociation,
  backboneCandidateAllowed, tagBackboneFreshLabel, mode4 count updates and alias handling;
- the frozen result/selection/bridge records, raw geometry, availability-change and
  per-object/intervention records. Do not infer object identity from equal numeric IDs.

Default parent attempt:
`/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001`.
Default new root:
`/mnt/shared/ww/ovimap-recovery-wave2-v1/attempt_001`.
Default Python:
`/home/ww/miniconda3/envs/ovimap-map/bin/python`.

Use actual `resolved_inputs.json`, per-map/per-readout receipts and the external manifest
rather than guessed cache names. Resolve native SigLIP, FC assets/text, Q checkpoint/scaler,
original poses/intrinsics/depth/CropFormer and cached SAM tracks/winner rasters from them.
No asset has been claimed newly loaded by the author of this instruction.

Support `--parent-root` and a read-only path map. Rewrite paths only in the new binding;
never silently rewrite a historical receipt or create a replacement with the same identity.
Parent caches are read-only: lookup exact parent entries, but write all new arrays/receipts
under the new attempt's content cache. Do not append new job receipts into an old attempt.
Hash only consumed objects, memoize validations, and avoid repeatedly traversing the entire
28-MB provenance file or all old arrays. Preserve weights and model operator revisions.

Record before measurement:

- source and runtime commits, actual loaded native `.so` and patch stack;
- scene schedules and original completed-frame policy;
- actual N/Q/F score order, vocabulary, per-scene inherited temperatures;
- raw numeric vs native-painted owners, native eligible registry, original source availability;
- native request-to-feature keys, FC region/dense/text cache keys;
- cached SAM per-track bit masks, RAW raster-to-seed mapping and chunk frames;
- which data are available, and which requests genuinely need new FC encodings.

The inherited native ratio threshold is **0.0**. Do not repeat a full RATIO_GATE arm or
replace the threshold with an unrequested value. Its mathematical redundancy is already known.

## 3. Cohorts and immutable comparisons

Development, in this order:
`scene0056_00`, `scene0534_00`, `scene0445_00`, `scene0626_00`.
Replica transfer/regression, in this order:
`office0`, `office1`, `office2`, `office3`, `office4`, `room0`, `room1`, `room2`.

All these scenes are exposed. Keep each original scheduled/completed frame list, poses,
intrinsics, RGB/depth bytes, invalid-frame treatment and category set. Do not inspect new
labels outside this cohort, add a convenient scene, or replace a difficult scene.

Use the existing parent **BB00_NATIVE** map and complete source distributions for the cheap
screen. Use exact measured values from receipts, not the rounded 11.74/24.50/29.69 table.
The earlier 11.763 paired-evidence result is historical context, not the new paired baseline.
Do not regenerate the 12 complete baseline maps merely because a task name changed.

Parent SYNC/RAW and failed FORWARD/BIDIR remain historical diagnostics; new recovery,
weights and association experiments start from BB00_NATIVE, not from a different winner per
scene. A tiny deterministic native trace checks instrumentation, not the benchmark score.

## 4. Execution phases and command

Implement `scripts/evaluation/run_ovimap_recovery_wave2.py` and the package
`src/static_ovmap/recovery_wave2/` with phases:

`bind -> cache-screen -> association-screen -> sam-screen -> compose -> freeze -> transfer
 -> report -> publish`.

`all` must execute the entire authorized dependency graph. `report` and `publish` must never
launch models or mapping. `--resume` reuses only matching complete identities and preserves
failed attempts. Mapping interruption restarts that scene from frame zero; do not invent a
partial TSDF resume capability. Successful upstream prerequisites are reused.

Required CLI after implementation:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_recovery_wave2.py \
  --phase all \
  --spec docs/paper/static_ovmap/recovery_wave2_v1/PROTOCOL_SPEC.json \
  --parent-root /mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-recovery-wave2-v1/attempt_001 \
  --gpu 2 --mapping-workers 2 --mapping-threads 8 \
  --evaluation-workers 3 --resume
```

GPU 2 is a historical example, not authority to occupy an in-use GPU. Resolve an idle allowed
GPU and the existing lock; record the actual command. Lower concurrency for memory without
changing frames, precision, evidence, or scientific rules. Do not kill other jobs.

## 5. Phase CACHE-SCREEN: weights and omitted-instance recovery

### 5.1 W controls: exact endpoints, including missing sources

Keep original instance masks, N/Q/F evidence and temperatures unchanged. Reuse FC_EQ and D2.
Test only gamma=.40 and .45. When all three sources exist:

`p_gamma=(1-gamma)*(pN+pQ)/2+gamma*pF`.

When a source is absent, use the inherited missing-source behavior. A reliable equivalent
implementation is interpolation between the actual FC_EQ and D2 probabilities with
`t=6*gamma-2`. Copy exact endpoints rather than perturbing them through a second softmax.
Both old methods assign .5/.5 when only one native source and F exist; a naive gamma formula
would incorrectly change that case. No fabricated probability for a missing source.

W never changes geometry or acquires visual evidence. Evaluate actual new labels and official
area ranks. If a probability changes but the labels and ranks do not, record this and reuse
the identical evaluator payload. Do not relabel it a failed run.

### 5.2 Build a geometry-derived recovery registry, not a replacement base map

For each parent BB00 map, bind raw owners R and the native-painted owner array O on the same
source vertices. Let active base owners be positive IDs in O. A recovery candidate is a
positive raw owner not active in O, with at least 100 source rows in
`(R==owner) & (O==0)`. Source-row count is not the evaluator's target-point minimum and is not
GT-derived. Keep that distinction in the report.

Sort by descending residual source-row count, then a canonical world-coordinate support
hash, and take at most128 candidates per scene BEFORE any new inference or GT diagnosis.
Do not use evaluator target coordinates, gt IoU, known classes, failed-object lists or result
labels to choose candidates. Report every reason for exclusion and every remaining no-view
candidate. `raw_owner>0` alone does not mean a correct physical object.

Lock all old native-painted masks. Only fill previously unowned source rows. Do not replace
the entire owner field by the raw numerical owner field; do not let new objects steal rows
from old objects. No partial repair of already active owners is authorized here. Preserve
all old D2 classes in a pure U arm. A W+U combination changes those old classes only through W.

This is a deliberate new output-eligibility method, not a bug fix to the official evaluator
and not evidence of a stronger TSDF reconstruction.

### 5.3 Restore actual requests through lineage

Use `reconcile_semantic_request` against ALL active final raw owners, not only native-painted
owners. Preserve request/mask/image identity. All segments in a request must reach one final
raw owner through the captured alias graph. A numerical ID or similar colored surface is
not sufficient. Dropped/ambiguous observations remain unavailable.

Captured requests include more technical candidates than the native selected subset. Use
that difference; do not synthesize a final-mesh view and call it a recorded causal request.
For U3 choose up to three genuine captured requests by descending visible area, frame ID,
then physical request identity. Do not rescan for a nicer view after a failure.

For an U1 candidate, require exactly one genuinely successful retained native observation
and bind its original request/feature. Recompute its canonical-relative single-view native
classifier with the original text/canonical embeddings in FP32, without modifying the old
`len(frames)>=2` function. A request whose lineage no longer matches the final candidate does
not qualify. Reconstruct six-crop vectors from the stored raw-vector cache only by applying
the exact original crop normalization/mean. Prefer the stored native feature when available.

### 5.4 Run four recovery arms

| ID | Additions on baseline-unowned support |
|---|---|
| RW_U1_NATIVE_SINGLE | Eligible single successful native observations, classified by the inherited native classifier |
| RW_UQ_PAID_QUERY | Already acquired/final-reconciled Q scores for the capped candidates; no new Q acquisition |
| RW_U2_FC_MATCHED_SINGLE | The exact U1 candidate and request set, using one FC region view; no extra view |
| RW_U3_FC_CAPTURED | All capped candidates with up to3 legal captured requests, FC area aggregation |

Do not combine U1 and UQ automatically, nor union all recovery outputs. They are separate
controls. U2 versus U1 gets both a common-successful-support comparison and a full-pipeline
comparison; technical failure does not count as model error on the paired subset.

New candidates without a genuine usable source stay unexported. In U2/U3, native or query
fallback cannot impersonate FC success. Existing base owners are never dropped because a
new candidate has no evidence. No zero-as-score placeholder may win a class.

### 5.5 Reuse FC dense caches correctly; cap genuinely new frame encodings

The wave-1 region_readout NOW stores `dense/*.npz`, region features, text prototypes and their
receipts. Reuse them only with identical actual model, image tensor, normalization/padding,
precision and region operator. A fresh mask requires fresh pooling/projection even when the
full-frame encoding is reusable. A source aggregate requires the exact used view IDs and weights.
A pooled vector is not a dense feature map. CPU/GPU loading and pooling still cost resources.

Reuse original FC text vectors/templates. No new text forward or new checkpoint is authorized.
There is no new SigLIP2/OVR branch. A matching original checkpoint is already expected in the
parent binding; its present server accessibility must be verified, not assumed.

For recovery only, permit at most32 previously uncached FC image encoder inputs per scene
application. Assign this frame allowance before inference using the candidate/view order
above; cached frames do not consume this NEW allowance but do count toward standalone method
cost. Selected requests on extra uncached frames are unavailable with an explicit budget
reason. Do not replace them with lower-ranked views. Reuse the union of required U2/U3 inputs.
Technical retries consume physical cost, not an enlarged scientific request budget.

Publish a same-covered-subset result to reveal effects of this compute cap. Treat a method
that relies on pre-existing dense evidence as a warm-cache refinement system unless cold
standalone obligations have been reported. Do not claim zero-cost recognition.

## 6. Phase ASSOCIATION-SCREEN: change failed-match semantics first

Keep CropFormer, native depth fusion, measured depth/poses, mode4/data_association2, and the
native ratio value unchanged. Build an isolated new extension on the exact prior patch stack.
Do not switch to mode3. Proposed changes must reach candidate filtering, fresh-label logic,
mode4 count updates and aliases; a Python association table alone is not an intervention.

### 6.1 Explicit per-local-group actions

Implement USE_NATIVE / ASSIGN_EXISTING / CREATE_NEW. The three scientific A arms never issue
CREATE_NEW explicitly. A non-match is USE_NATIVE, and the original mapper may still create
an object when its own evidence requires it. Do not preallocate a new owner for every non-match.

An all-USE_NATIVE trace must reproduce the original native path in serial mode. Fallback
must apply beyond candidate generation: no missing planned-owner assertion, forced mode4
owner, fresh-group-only veto, or inherited blanket alias veto may remain for an ordinary
unconstrained fallback group. Protection of labels actually touched by an accepted plan is
allowed; log those mixed interactions explicitly. They are not an exact native route on
those protected labels. Do not preload every historical owner as a hard cannot-merge token.

### 6.2 Actual A arms

- **RW_A1_NATIVE_FALLBACK:** use exactly the prior forward one-to-one eligible/benefit solver:
  intersection>=100, forward coverage>.2, dummy unmatched nodes. Accepted pairs are explicit;
  every other local group takes USE_NATIVE.
- **RW_A2_MULTI_FREE:** keep those eligible edges but choose each row's greatest forward
  coverage independently. Multiple local groups may select the same existing global owner.
  This isolates relaxing capacity, without the later union/specificity gate.
- **RW_A3_MULTI_UNION:** start from A2 proposals. Require per-local-group dominance>=.8 among
  all known-owner intersection pixels and forward best-vs-second margin>=.1 (missing second=0).
  Group retained local parts by owner and require their union reverse coverage>.2 of that
  owner's current depth-consistent visible support. Failed members/groups take USE_NATIVE.

Do not merge two existing global owners, use semantic GT or final FC probabilities to make
these association decisions, or reuse another experimental map's future owner assignments.
Multiple local groups may update a common existing owner, but a local group cannot update
multiple owners. Keep original count-update weights; do not multiply a whole-object vote
once per constituent fragment without the original per-fragment meaning.

A2 fresh-label candidate logic must check compatible planned existing owner, not merely
identical current 2D group. The inherited fresh-group-only rule would otherwise silently
reintroduce one-to-one behavior. Assigned-instance sets and current-to-global maps must also
permit the explicit many-to-one plan. Preserve native behavior for unplanned groups.

Run the three A arms on all four development scenes. Compare against BB00, and compare A1
against the historical failed FORWARD only as a secondary repair diagnostic. Reaching that
failed control's score does not count as a new improvement.

## 7. Phase SAM-SCREEN: candidate preservation from actual cached outputs

The parent saved **per-track binary masks and the final RAW winner raster**, not full logits.
Do not reconstruct probabilities from masks, rerun SAM silently, or pretend a rejected
winner can be replaced by the true second-highest logit. All proposed S rules intentionally
use the recorded winner support. Bind each raster label to its seed/track via parent remaps,
chunk first-frame ordering and stored track IDs, with a round-trip check.

### 7.1 RW_S1_CROP_PRIORITY

This is a conservative mask-union/completion test, NOT complete replacement by SAM. Preserve
every currently positive CropFormer pixel and its separate original group. Use SAM only on
current CropFormer-background pixels:

1. Compute actual binary mask IoU and per-CropFormer coverage. Attach a SAM track to a current
   CropFormer group only for a mutual unambiguous best-IoU pair with coverage>=.8 and IoU>=.5.
2. Add that track's RECORDED RAW winning pixels on background to the matched CropFormer group.
   All original pixels of that CropFormer object remain in that same group.
3. An unattached track may form an independent completion only when its overlap with the
   union of current CropFormer positives is strictly below20% of its binary mask area.
   Use its recorded winner pixels on background; otherwise add nothing. This avoids creating
   duplicate residual objects beside a partially overlapping protected CropFormer instance.
4. Use one deterministic spatial label remapping; compare physical supports, not numeric IDs.
   Chunk-first and no-seed frames are exactly the original CropFormer input.

This differs from the old rule that could drop a full current candidate after only20% SAM
coverage. It cannot guarantee better geometry: a CropFormer error is also preserved.

Construct S1 rasters across all development frames without new SAM inference. If every raster
is canonically identical to CropFormer, record EQUIVALENT_INPUT and reuse the parent baseline
rather than rebuilding four identical inputs. Otherwise run S1 on the complete development
cohort. Per-scene exact-equivalence aliases are permitted with input evidence.

### 7.2 RW_S2_CONFLICT

Use S1, but reject misleading SAM-origin additions when an OWN-map, pre-insertion prior and
past snapshots provide specific contrary ownership evidence. Never damage current CropFormer
pixels to execute this rule. The complete causal definition is in the contracts.

Use only existing positive, depth-consistent prior owners whose superpoint-to-owner membership
is stable in both preceding completed snapshots. Resolve a track's self owner from its previous
visible support, with >=100 supported pixels and >=.8 purity. If that cannot be resolved, keep S1.
For enough known current support, compute other-owner conflict over known self/other support.
If conflict>.2, drop that track's proposed additions, not the original CropFormer object.
Unknown/newly exposed space is not other-owner evidence. Preserve the same unmodified SAM
track history; S2 must not feed corrections back into SAM or use a final map for a past decision.

Run S2 only if S1 has a real intervention and the required saved track assets are readable.
Lack of real conflict evidence is an abstention, not technical failure. Report the numbers
of evaluated, abstained and suppressed tracks/pixels, and whether final input/masks changed.
S2 can end up equivalent; publish that fact instead of creating another fake success.

## 8. Geometry-first cost control without confusing it with the final goal

Every newly mapped development arm first exports its raw geometry and a prediction-only
frozen map receipt, then runs post-lock geometry diagnosis. This is authorized before neural
semantic inference; old wave-1 diagnose_map's all-readouts assertion is not the contract for
this new stage. Never feed diagnostic GT to the mapping process.

Compare against the same four parent baseline maps. An arm is catastrophic when either:

- mean best-GT IoU falls by at least3.0pp AND R50 loses at least2 GT; OR
- substantial fragments/GT is at least1.75 times baseline AND R50 loses at least1 GT.

For a catastrophic arm, retain the complete maps/diagnostics, mark SCREENED_OUT_GEOMETRY,
and stop expensive N/Q/FC evaluation for this arm. Report semantics as NOT_RUN_RESOURCE_SCREEN,
not zero, not COMPLETE. This is a predeclared resource decision, not a proof that no semantic
readout could help it. Do not screen out a mildly mixed geometry result using this exception.

For every noncatastrophic arm generate NATIVE_READOUT, FC_EQ and fixed D2 with fresh anchors,
actual new view requests and the original frozen models/temperatures. Do not reuse old owner
scores just because the IDs or surfaces look similar. Reuse only exact physical encoder
content and validated feature aliases. New map projections must be recalculated when source
coordinates/order change. Query replay sees only its own current frames and paid evidence.

## 9. Selection, one composition and frozen transfer

Use four-scene official development pools only. Raw units in JSON are fractions; .2pp=.002.
For selection, a configuration is feasible against BB00+D2 if delta APall>=-.05pp,
delta AP50>=-.10pp and delta mIoU>=-.10pp. Keep the baseline in the candidate set.
Select with APall .05pp band, then mIoU .10pp band, then AP50 .10pp band; prefer fewer standalone
new encoder inputs, fewer changed blocks, a parameter closer to baseline, then fixed method ID.
Bands are decision preferences, not statistical equivalence claims.

For W pick one gamma using its four controls. For U pick one recovery rule or NONE. Evaluate
exactly one W-selected x U-selected combination on the original four maps (components already
exist). Select the final light package among baseline, individually measured W/U conditions and
that composition. Do not exhaust a W-by-U Cartesian grid. An UQ or U1 control may win.

A new map can be transferred only if fixed-D2 development APall improves by>=.20pp while
AP50 and mIoU fall by no more than.10pp and it passes the geometry resource screen. At most
one such map is selected by the same band rule. A map with improved mIoU but lower AP remains
a tradeoff record; do not spend eight new maps just because a family requires a representative.
If no map qualifies, complete the light transfer; do not enlarge a parameter sweep.

Apply the frozen light package on that selected new map for the four-scene composition check.
Do not transfer per-owner scores from the old map. The two-by-two is:
old map/fixed D2, old map/light, new map/fixed D2, new map/light.
No A-plus-S architecture combination is in scope this wave.

Freeze all scalar choices, recipes, technical caps, candidate methods and research nomination,
commit them and record the full pre-Replica commit BEFORE any new Replica prediction or metric.
The original parent results may be read for context. After freeze:

- execute the complete eight fixed light controls/arms on all eight Replica scenes, plus the
  one frozen light package where distinct; all these use the original BB00 maps;
- reconstruct at most one qualifying new map on all eight Replica scenes;
- run its three standard readouts and the one frozen light package, where distinct.

A resource-unavailable U arm gets honest per-leaf status; do not omit a hard scene from its
pooled metric or replace it with a partial cohort average. A defined scientific no-op gets a
complete identical prediction and reuse record. Replica cannot change gamma, recovery choices,
association thresholds, SAM rules or the nomination. A descriptive best regression result is
not the frozen nominee. Deployment stays N0_UNCHANGED.

Provide a leave-one-development-scene-out sensitivity table for the already measured fixed
methods. Do not fit four new algorithms or reinterpret those four correlated pools as new trials.

## 10. Official evaluation and proof of the actual intervention

Keep the released evaluator, category lists, ignore rules, minimum region size, triangle
conventions on old support, nearest-neighbor projection and current-class area ranking unchanged.
Read the actual overlap vector; APall in this release uses .50 through .90, not an assumed .95.
Pool ordered complete scene files with the released evaluator; sum semantic confusion matrices.
A scene average is diagnostic, never a substitute for the official pool.

For U output construct a NEW payload with the same coordinates and expanded positive owner
universe. Do not use relabel_prediction to pretend the old owner array changed. The inherited
SceneEvaluator enumerates mask files through its input registry: extend that registry for ALL
actual output owners, including recovered owners with N unavailable, without fabricating N
scores. Confirm added owners appear in the emitted manifest and masks. Recalculate official
ranks for ALL owners, including existing ones. Adding one large same-class instance can alter
old ranks, so unchanged old masks/classes does not guarantee unchanged AP.

Report primary official metrics. For W alone the old fixed-rank diagnostic can be reused.
For U report old-owner rank changes and actual match losses; do not label a mixed new/old rank
table as the original FROZEN_N0 protocol. For each new map optional secondary ranks use that
map's own legitimate native table, never a historical different-geometry rank table.

Required small scientific tables:

1. Full method x cohort APall/AP50/AP25/mIoU/mAcc and paired deltas; actual coverage/status.
2. Raw registry -> native eligible -> lawful captured requests -> used evidence -> exported
   recovered owner funnel, separately by U arm and source availability.
3. Added TP/FP, displaced matches, class/rank changes and point-confusion changes; official
   matcher output, not guessed from label correctness. Diagnostic source sets must correspond
   to the actual method, not all sources that happened to be saved.
4. Association actions -> candidate/alias effects -> realized owner -> fresh-owner rate and
   raw geometry. Distinguish planned and realized membership.
5. Crop pixels preserved, SAM additions, stable self/foreign evidence, suppression/abstention
   and actual geometry. A new raster ID alone is not an intervention.
6. Physical and standalone logical cost by operation/model, including cache dependence.

All predictions lock before GT is read for diagnostics. Use GT only in the evaluator and
separate development/analysis processes. Do not pick individual predictions with GT, assign
GT-specific synonyms or silently export an oracle combination.

## 11. Required implementation layout (new, not asserted to exist)

Create minimal task-local modules with clear boundaries:

- `binding.py`, `workflow.py`: parent artifacts, isolated paths, task DAG and resume;
- `recovery_registry.py`: raw-to-painted gap, candidate caps, strict request reconciliation;
- `recovery_sources.py`: cached native/Q restoration and real FC region requests;
- `weight_readout.py`, `export.py`: exact available-source interpolation and expanded payloads;
- `association.py`, `mapping_hooks.py`, `native_patch.py`: per-group actions and actual C++ path;
- `sam_completion.py`: binary/winner-only candidate composition and causal conflict filter;
- `evaluation.py`, `selection.py`, `reporting.py`: unchanged scorer adapters, frozen decisions,
  dynamic summaries. Reuse imported numerical operators instead of copying entire old workflows.

A new wrapper is preferable to weakening old assertions. Add the local native patch and exact
apply/build instructions; do not replace the original extension in place. Keep original licenses.
The actual Python/C++ function signatures may be designed during implementation, but their
semantics must implement the contracts. This is not permission to silently omit a required layer.

## 12. Bounded checks, compute and failure handling

Do only checks tied to these changes: exact weight endpoints/missing-source behavior;
old-mask preservation and recovered-owner export; single/paid/FC lineage; fallback-native
and compatible-many-to-one native paths; binary-mask crop priority/unknown-space handling;
selection units and one real evaluated payload. Reuse the supplied synthetic kernels as
examples, not evidence the production code works. Run targeted tests once after repairs.

Use one serial3-valid-frame native off/all-fallback/accepted/follower trace. Confirm unchanged
state off, no preallocated owner for USE_NATIVE, and a real pair of local groups reaching one
existing owner. If current real data lacks such a pair, use a small synthetic native fixture
in addition; never mark the path validated solely because a Python unit test passed.

Do not run all repository tests, rebuild unrelated packages, repeatedly hash weights, rerun
old64 maps, or create an audit-count target. Fix real problems without approval loops; retain
logs and count failed work. A leaf blocked by missing source data must not block cheap unrelated
arms. A changed scientific recipe requires a new attempt and rerunning affected cohort outputs.

Caps: at most20 new development map-scene successes plus8 new Replica map-scene successes;
no new full baseline maps. At most32 new recovery FC image inputs per scene application,
128 candidate targets,3 views/target. No new SAM/CropFormer/text forwards and no training.
Map semantic regeneration uses the inherited N/Q/FC rules and is separately costed; it is
NOT governed by the small recovery-only FC cap. Do not claim zero GPU for the whole task.

Default mapping2 jobs x8 threads, evaluation3 jobs x4 BLAS threads, one model worker per GPU.
Track actual encoder inputs and calls for N/Q/FC, region pooling, model loads, CPU/wall time,
cache hits and missing timings. CUDA event timing is useful if added without changing inference;
otherwise label worker wall time honestly. Concurrent duration sums are not elapsed wall time.

## 13. Publish code, results and handoff, then verify remote

Commit the executable code, patch, frozen spec, commands, selected parameters, compact numeric
results and all four reports:

- `docs/paper/static_ovmap/RECOVERY_WAVE2_RESULTS.md`
- `docs/paper/static_ovmap/RECOVERY_WAVE2_HANDOFF.md`
- `docs/paper/static_ovmap/RECOVERY_WAVE2_SELECTION.md`
- `docs/paper/static_ovmap/RECOVERY_WAVE2_CLAIMS.md`

Put compact artifacts under `artifacts/static_ovmap/recovery_wave2_v1/`, including:
`resolved_inputs.json`, `experiment_matrix.json`, `recovery_funnel.json`, `source_scores.json.gz`,
`locked_decisions.json.gz`, `official_scene_rows.json`, `official_pools.json`, `geometry_summary.json`,
`mechanism_comparisons.json`, `selection.json`, `failure_and_costs.json`, `external_artifacts.json`,
`completion.json`, and targeted validation receipts. Include actual small evidence, not only
external path names. Large RGB-D, weights, dense features, full maps/TSDF and verbose native
logs remain on shared storage with truthful content manifests and reconstruction commands.
Aim for <=50 MiB compact publication, not a hard scientific truncation rule.

Normal push to the named branch is required. Then compare full local HEAD with
`git ls-remote origin refs/heads/research/ovimap-recovery-wave2-v1`.
Report PUSH_VERIFIED only if equal. If push/auth fails, keep the local commit and report the
actual error; do not claim publication. Store the final post-push receipt externally under
`publication/final.json` to avoid a self-referential commit hash. No recursive verification commits.

The handoff must identify what was implemented, measured, aliased as a no-op, resource-screened,
or blocked; exact runtime/models; how to restore external data; which frozen candidate wins;
what happened on Replica; and whether the NET_GAIN criterion actually passed. Completion,
scientific outcome and publication are separate statuses.

Do not conclude merely “all tests passed.” Conclude with the full paired score table and a
specific decision: retain baseline, retain a tradeoff candidate, or retain a measured gain
candidate. Do not automatically replace production. A task with no gains must publish the
negative measurements, not broaden the search without a new specification.
