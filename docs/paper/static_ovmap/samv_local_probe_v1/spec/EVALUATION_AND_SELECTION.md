# Evaluation, selection and cost rules

## 1. What is held fixed, and what is newly measured?

Four full-scene maps are used. Source xyz, faces, TSDF and frozen target projection
remain unchanged. Owner partitions may change only in structural arms; semantic
arms preserve the original G1 owner array. Main AP/mIoU always come from the SAME
actual full-map prediction, including all untouched objects and unknown labels.

The six main methods × four scenes yield 24 main records. D2 adds four reference
records. Pool each method over the exact two scenes of each probe cohort: 12 main
pools + two D2 pools. Thus 28 total logical records /14 complete subset pools.
Logical records, new prediction payloads, physical scorer calls, cache aliases and
model forwards are different counts and must be reported separately.

No existing full Replica8/CF18 number can populate these subset pools. Compute
subset pools from the exact parent per-scene D2/G1 predictions/scorer receipts. A
scene with no legal query remains as unchanged G1, with reason. Missing required
inference/scoring cannot be removed from a pool or replaced by a mean.

## 2. Released evaluation integration

Use `PredictionPayload` with branch COMBO and inherited GeometryIdentity, plus new
owner_ids / semantic_labels and `native_ranks` from actual target-space areas.
Unowned rows have class 0; one positive owner has exactly one class. Re-enumerate
all positive owners, including G1-recovered and nonselected donor owners.

Reuse `partition_evaluator`'s approach: instantiate a source registry for all actual
owners and place mask files under a path keyed by the actual owner-array digest.
Never reuse the old `owner_17.npy` merely because the ID still equals17. Reuse an
assessment only when actual owner/classes/ranks and complete scorer context match.

Retain OFFICIAL_CURRENT_CLASS ranking and the original minimum region size100.
APall averages .50,.55,...,.90, AP50 and AP25 follow the original released code.
Do not replace it with a .50:.95 implementation. AP25/mAcc remain in the canonical
store even if the compact main table shows APall/AP50/mIoU only.

Freeze all predictions before annotation access. Per-scene AP means do NOT replace
complete ordered pooled AP. Pooled semantic metrics come from summed confusion
matrices, including small/unknown/predicted-error regions according to the original
scorer. Score and trace all full-scene predictions, not just query masks.

## 3. Baseline parity and model/interface validity

Before final scoring, verify one G1 baseline scene per cohort against its actual
parent released view, confusion matrix, AP and trace. The new local builder must
reproduce G1 exactly when no model evidence or no edits/relabels are supplied.
Empty masks are valid neural outcomes; missing checkpoints, nonfinite logits or
unreturned frames due to exceptions are failures. Do not label resource failure
as scientific abstention. An old unknown/raw-zero point restriction may not be
silently reinstated, as it can suppress the very new support under test.

The paper's prompting/evaluation code reads GT for sample construction. Our actual
prediction worker receives only locked query manifests and RGB; our lifter receives
measured depth/cameras/mesh. Neither reads `*_label.npy`, GT instance images, per-GT
opportunity ledgers or future evaluations. Tests target this data separation, not
an expensive general sandbox/security audit.

## 4. Mechanism diagnostics from the same predictions

Report the following, with scoped denominators:

A. Query coverage: planned/selected targets, pool (incumbent/recovered), available
representative views, anchor success, raw empty/nonempty masks, qualified OLD/NEW
semantic masks, joint semantic eligibility. Never label queried owners as TP count.

B. Three-dimensional support:
- independent raw voted target supports, admitted candidate supports, actual final
  exclusive supports; don't conflate them;
- class-agnostic maximum one-to-one GT matching at strict IoU>.50 and >.75 for each
  actual whole-map partition, gained/lost unique GT IDs versus SV00;
- per-target best valid-GT IoU before/after, plus a fixed-reference GT identity chosen
  from old-support maximum overlap AFTER prediction locking (no best GT re-selection
  in each view to hide identity drift); ties smallest GT ID;
- if no valid old association exists, mark fixed-reference diagnosis undefined;
  still include the whole-scene official evaluation and raw support inventory;
- number/area of raw proposals suppressed by domain/core safeguards, anchor failures,
  rows with fewer than two visible views, collisions and protected interiors;
- overmerge/fragmentation indicators and donors harmed. Geometric gain may be
  constrained by the lifter, so separate raw-mask promise from final-map performance.

C. Semantics: OLD-vs-NEW wrong→right/right→wrong, matching the same original owner
supports; paired success exclusions; unique released GT50/75 gains/losses and class
rank changes. Matching entries, unique GTs, ignored predictions and unresolved
mixed tied-score entries remain separate. Do not compute precision by dividing
unique GT counts by query owners.

D. 2D identity/quality: no new annotation acquisition is needed. When the original
capture has genuine aligned 2D GT with verified dataset identity, use it only for
post-lock evaluation. Otherwise do not pretend pre-insertion panoptic predictions
are GT. Projected 3D-GT-derived 2D diagnostics, if generated for inspection, must be
called proxy measurements and never counted as direct annotated 2D benchmark scores.
It is acceptable to report only 3D matching and raw model masks for this first pilot.

E. Novelty boundary: SAM-V is an external pretrained segmentor. A positive plug-in
result validates an integration hypothesis, not a new geometry-aware decoder or
proof of superiority over every multi-view method. The SAME SAM2/SAM-V prompts and
lifter isolate practical module choice, not identical architecture/training budget.

## 5. Positive paths, fixed selection and stop rules

Always finish the fixed small comparison unless a real execution dependency fails.
A negative arm is not a reason to stop other arms, and a positive arm is not permission
to expand or tune. Study statuses are separate:
IMPLEMENTATION, ASSETS, INFERENCE, EVALUATION, PILOT_TARGET, MECHANISM and PUBLICATION.

Research candidates: SV02, SV04, SV05. A candidate passes the pilot target only if:
1) all five fraction metrics in EACH two-scene cohort >= its SV00 counterpart−1e-10;
2) CF-probe APall > corresponding REF_D2 APall+1e-10;
3) CF-probe AP50 >= corresponding REF_D2 AP50−1e-10.
The extra MATERIAL flag requires CF-probe APall−REF_D2 APall>=.001 (0.10pp).
It is not statistical significance. Select among passers lexicographically by
CF APall, CF AP50, Replica APall, CF mIoU, treating differences<=1e-10 as ties;
remaining ties prefer SV02, then SV04, then SV05. No pass -> SV00. Report the best
simple control separately; don't rename SV01/SV03 as an original method.

Mechanism flags, independent of this target:
- SEGMENTOR_SUPPORT_SIGNAL: raw/admitted/final evidence shows where SAM-V improves
  support over same-query SAM2; require a strict positive net unique class-agnostic
  GT50 OR GT75 difference and nonnegative best-IoU mean in both probe cohorts for
  the strongest positive flag. Otherwise describe the narrower actual outcome.
- SEMANTIC_MASK_SIGNAL: SV04 versus SV03, report every metric and actual corrections;
  call it a uniform positive only when all five per cohort are nondecreasing and
  some metric or unique correct match increases. Failures/coverage changes remain visible.
- COMBINATION_SIGNAL: SV05 versus SV02 and versus SV04; no cooperative-gain claim
  when the combination merely equals one component.
- REPRESENTATION_ONLY_PROMISE: raw masks improve but the fixed local output adapter
  erases gains. This is diagnostic, not a passing complete method.

An exact tie with G1 is not an improvement. An attractive screenshot alone does not
justify expansion. The engineering OOM profile was chosen before GT evaluation,
not to maximize a score. Never use different checkpoints/methods on the two datasets.

Regardless of outcome, deployment stays N0_UNCHANGED. Report whether a separately
approved full Replica8/CF18 study is justified; do not execute it in this task.
These previously exposed four scenes provide development evidence, not untouched
confirmation, and their small pools can have unstable category coverage.

## 6. Cost accounting and the only extra timing block

First scientific acquisition: separate planning/raycast cost, SAM model loading,
VGGT model loading, SAM encoder image inputs, VGGT joint groups and image slots,
SAM-V target decodes, SAM2 initialization/images/forward+reverse outputs, FC frame
encoding and region/head work, source-row lifting, output/rank construction,
evaluator/export costs. Cache hits do not imply zero standalone compute. Scientific
model work is capped at32 target queries/model, not at32 physical retries.
All failed calls and engineering checks are counted separately; unknown time is null.

Additional timing is narrowly defined: two prelocked pilot query windows, two models,
when both cohorts have queryable targets (otherwise mark the absent block undefined
with NO_QUERIABLE_TARGET, without replacing a scene). The ordinary scope is
two repetitions =8 calls. Model resident; no feature/result cache; same canonical
files/prompts/window/profile; begin before reading canonical RGB, end after raw masks
are restored to original image dimensions. Required model initialization, canonical
file generation, mapping, query planning, lifting, FC and scoring are excluded and
reported elsewhere. Synchronize CUDA at boundary and reset peak memory stats.
Verify raw-mask parity to that method’s locked scientific result outside timing.
Report allocated and reserved separately, actual device/software, per-window values
and all measurements. Do not call this seconds/full-scene update or online FPS.
This timing intentionally keeps model-specific input normalization, while raw decoded
RGB and prompt information are identical. SAM2 has forward/reverse propagation;
SAM-V has joint inference; that computational difference is part of the comparison.

No timing sweep, no 64-call full-method replay, no fixed speedup target. If one model
cannot run, no invented comparable time. If single inference time already reveals
unreasonable cost, still finish the small fixed study where feasible and report it.
