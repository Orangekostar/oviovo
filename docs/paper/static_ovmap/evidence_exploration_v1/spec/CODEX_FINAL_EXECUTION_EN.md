# Final Codex execution instruction — evidence-driven recovery exploration

## 0. Mandate, objective, scope and authority

Implement, execute, analyze and publish this bounded study. Do not stop after a
plan, mock table, synthetic test or asset installation. The objective is to find
one uniform recovery rule that improves ScanNet-CF18 instance AP while preserving
all five current Replica-8 metrics. This objective is **not a guaranteed result**.
A complete negative study is preferable to a fabricated gain or a selectively
reported subset. Finish all executable branches even if another branch is blocked.

Use repository `Orangekostar/oviovo`, base commit
`30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f`, new branch
`research/ovimap-evidence-exploration-v1`. Use a new worktree and output root
`/mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001`.
The read-only parent runtime attempt is
`/mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001`.
Its `reference_binding.json` resolves the actual v2 evidence; a known v2 location
is `/home/ww/ovimap-area-fallback-v2/attempt_001`, but use its recorded identities
rather than trusting the path name. Preserve the previous working trees, their
indexes, model assets, published receipts and deployment. Do not force-push.

This document is authoritative. Copy the accompanying `PROTOCOL_SPEC.json` to
`configs/static_ovmap/evidence_exploration_v1.json`. All constants below are NEW,
prespecified experimental choices unless described as inherited. They are not
claimed to be empirically optimal or to reproduce another paper's full method.
Do not conduct a hidden threshold sweep or change constants after seeing scores.

There are two proposed mechanisms:
1. geometry-weighted target-versus-competing-support semantic contrast, followed
   by an explicit accept/defer decision;
2. owner/depth-conditioned resampling of frozen FC features using frozen AnyUp
   attention, compared with ordinary AnyUp and bilinear interpolation.

Do NOT add active second-view acquisition, preference training, LoRA, factor-graph
libraries, an LLM/agent, SAM, a new segmentation front-end, or a new mapper. FunFact
and GeoGuide motivate questions, not dependencies or claimed reproductions. Keep
D2, N/Q, all text prototypes, calibrated temperatures, geometry and instance ranks'
DEFINITION unchanged. This task may change recovery classes and accepted recovery
owners, but never the existing D2 instance masks or labels.

## 1. Required source reading and exact task binding

Before implementation, read these real paths at the base commit and write a short
`source_binding.json` / `SOURCE_BINDING.md` mapping of symbol -> input -> output ->
planned change. Do not just copy this table without reading the local files.

| Source | Existing responsibility | Use in this task |
|---|---|---|
| `runtime_parity/binding.py`: `import_metrics`, `bind_reference` | Actual v2/source/metric lineage | Consume the verified reference and existing 172/14 store; do not replay its old freeze process |
| `runtime_parity/runner.py`: `execution_config`, `load_common`, `CommonInputs` | Actual geometry, Native-painted owners, N/Q/F, class order and evaluator projection | Reuse data loading behind a new task adapter |
| `runtime_parity/views.py`: `build_views`, `scientific_key`, `CallLoader` | R2 geometry-only G1/G3 selection and content identity | Keep the exact G1 selected frame/mask; reuse saved views for acquisition |
| `runtime_parity/kernels.py`: `ExactProjector` | Conservatively reduced rays, full-mesh occlusion | Retain R2 for standalone G1 search, not a new visibility heuristic |
| `cvpr_compact/projected_views.py`: `FullSceneProjector.project_frame` | Dense depth-consistent owner raster from the full predicted mesh | Obtain additional owner/depth evidence on the already selected G1 frames |
| `cvpr_compact/area_fallback.py`: `pool_region`, `region_vector` | Exact original pooling when supported; area-weighted empty-mask fallback | Freeze as B1, use for coarse target/control features |
| `runtime_parity/session.py`: `RuntimeSession.encode_exact` | Actual frozen FC forward, v2 or U2 path | Reference feature layer, precision and cost accounting; use a NEW acquisition worker |
| `a7_evidence_upgrade/region_adapter.py` | RGB preprocessing, signed mask, frozen visual projection and pinned OVRCOAT operators | Preserve pixel geometry, feature space and final projection head |
| `cvpr_compact/region_worker.py`: `classify_regions` | Cosine scores followed by unconditional argmax when available | Do not reuse its unconditional export semantics for the new verifier |
| `cvpr_compact/outputs.py`: `construct_output`, `build_method_outputs` | Fixed incumbent support, accepted additions, recomputed official ranks | Call `construct_output` with accepted labels; do not silently weaken the old source contracts |
| `cvpr_compact/evaluation.py` and `recovery_wave2/evaluation.py` | Expanded registry, released full-scene scoring, per-class and full-cohort pools | Narrow new task adapter, unchanged numerical scoring |
| `released_trace.py` | Actual scored TP/FP entries and ties | Diagnosis after prediction lock, not a predictor input |

All repository-relative source paths in this table start with `src/static_ovmap/`.
Also read parent `RUNTIME_PARITY_RESULTS.md`, `RUNTIME_PARITY_SELECTION.md`,
`RUNTIME_PARITY_CLAIMS.md`, `reference_binding.json`, `imported_metrics.json` and
actual v2 region/prediction receipts. Read only consumed dependencies. A previous
result filename is not proof that the file exists or that its contents match.

Create one narrow task binder. The parent binder includes historical source aliases
and old freeze guards; do not impersonate the parent task or disable its assertions.
Read the recorded immutable inputs, resolve relocation through an explicit path map,
and create a new task binding. If a historic worktree moved, a matching published
source snapshot can satisfy producer provenance; never demand an unneeded historic
worktree just to run current code. Missing actual RGB/depth, model weights, geometry
or feature evidence is a real leaf prerequisite failure and must be recorded.

Use parent GPU/interpreter settings unless explicitly overridden. The known main
interpreter is `/home/ww/miniconda3/envs/ovimap-map/bin/python`. Reuse the parent's
working FC environment. Install AnyUp into a separate, minimally compatible worker
environment if needed; do not upgrade the working mapper/FC environment in place.

New controller (to IMPLEMENT, not claimed to exist now):
```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_evidence_exploration.py \
  --spec configs/static_ovmap/evidence_exploration_v1.json \
  --parent-runtime-root /mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001 \
  --phase all --resume
```
Support phases `bind`, `diagnose`, `assets`, `prepare`, `encode`, `predict`,
`evaluate`, `select`, `time`, `tables`, `publish`, `all`, plus `--gpu`,
`--path-map`, and `--parent-reference`. `all --resume` must run missing authorized
work and preserve successful immutable work. Existing files must not trigger
"complete" unless their science/operator identities match.

## 2. Freeze shared inputs; establish why the experiment can be informative

Use EXACTLY the parent Replica-8 and CF18 lists from the JSON. There are 26 shared
native geometries. No new map construction is authorized. Do not change depth,
pose, original frame schedule, image resolution, candidate cap or class dictionary.
CF18 is 18 captures of 7 physical families; it is not the full ScanNet validation set.
All current cohorts are exposed. No unseen/generalization or significance claim.

The parent candidate registry is shared across ALL arms: raw owner absent from the
Native-painted positive owner set; residual source support at least 100 rows; the
same area/support-digest order and cap of 128. Use `build_registry` and verify its
identity against the parent. Recognition uses the full visible owner mask; export
uses only `K_i = (raw==i) & (native_painted_owner==0)`. Do not confuse these masks.
G1 uses the single originally selected geometry-ranked view. No model-score-based
view replacement, no new frame search policy, no per-dataset rules.

Bind B0 to the parent `CT_A1_E` (D2 with no recovery) and B1 to `CT_A3_ER` (G1 v2).
Read full-precision scores from the parent, not the rounded numbers in the chat.
Their Replica display values approximately include APall 11.74 vs 12.39 and mIoU
29.69 vs 30.27; CF18 approximately includes APall 8.31 vs 8.23. A mismatch requires
explanation of source/scorer identity, not editing expected metrics to pass.

Run a CACHE-ONLY baseline diagnosis first, using the actual v2 masks and locked
predictions. For every recovered candidate record evaluation size, assigned class,
class-agnostic best IoU to eligible GT, assigned-class overlap, void/group/small-GT
ignore behavior, newly scored TP/FP/tied entries, and whether its predicted class
has eligible GT in this scan and/or other scans. Also record old rank changes.
Classify errors in a separate post-prediction diagnostic: target-small/ignored,
geometric insufficiency, category error, duplicate, TP, tie/ambiguous. Keep a
nonexclusive raw ledger as well; do not force ambiguous cases into one explanation.

At most TWO cache-only counterfactual full pools are allowed, one per cohort:
keep B1 predictions but hold old B0 scores fixed (new prediction scores remain B1).
Label this diagnostic, NOT official performance. This isolates old-rank changes;
it does not establish that all remaining loss is background leakage. The official
rank remains current-class area. Do not tune the new rules using individual GT
labels, construct a protected-TP list, or discard unfavorable candidate types.

Diagnostics must not become a prerequisite requiring unavailable old logs forever.
If final masks and GT exist, regenerate only the necessary matching diagnostics.
Proceed with the fixed slate even when the hypothesized error is rare; report
coverage/no-op outcomes. The diagnosis is not permission to select only bad CF18
objects or only positive Replica scenes.

## 3. Predeclared local intervention set and competing-support regions

### 3.1 Shared low-support/mixed-support trigger

Compute this BEFORE new semantic decisions from the B1 G1 mask and unchanged FC
preprocessing. Let `M` be its resized, padded binary mask on the FC input grid.
Let `w = area_resize(M, dense_hw)` with the actual cached/encoded `clip_vis_dense`
shape. Let `h` be the original signed-bilinear positive support count from v2.
Define `purity = sum(w*w)/(sum(w)+1e-8)`. This is a geometric feature-cell occupancy
proxy, not predicted accuracy. Require valid positive mass for any observable M.

The intervention trigger is `h < 4 OR purity < 0.5`. Freeze it for all new arms.
Outside this set, return EXACT B1 recovery scores, labels, acceptance and vectors;
existing D2 instances are always unchanged. Record trigger counts by cohort and
baseline normal/fallback path. Triggering is not allowed to depend on dataset name,
GT, B1 error status or the new model's answer. If no objects trigger, mark the
corresponding methods as factual no-ops, not as successful innovation.

### 3.2 Additional geometric evidence, not additional recognition views

For the union of already selected G1 frames, obtain the full predicted-scene
first-hit owner raster and original measured depth using
`FullSceneProjector.project_frame`. Non-candidate faces and owner-zero faces remain
occluders. Homogeneous positive raw-owner triangles produce owner IDs; ambiguous
triangles and invalid/occluded depth evidence are unknown. Preserve camera-z rays,
original intrinsics/poses and tolerance max(0.02 m, 0.02*measured_depth).

This is extra geometry computation on SELECTED frames, not another entire 200-frame
search. Verify each target mask equals the original selected mask. Do not use the
partial queried ROI as if it were a complete owner raster. Store one compressed
owner/depth-evidence map per selected frame if useful; record its dependency/cost.
All methods can share the same acquired evidence in scientific runs. Standalone
method costs include the evidence they actually require.

### 3.3 Competing-support controls

For each triggered target, expand its ORIGINAL half-open bbox on every side by
`max(8, ceil(0.1*max(bbox_width,bbox_height)))`, clipping to the image. For each
other positive depth-consistent projected raw owner j, define a control region:
`C_j = expanded_box & ~target_mask & (visible_owner==j)`.
Keep regions with at least 100 original pixels. Select at most two distinct owners
by decreasing region area, tie by numeric owner ID. Do not combine all background
into a single region or use unknown/invalid pixels as negative evidence. A control
is a competing spatial explanation, NOT ground-truth proof of a different object;
it may be another part or a same-class neighbor. Hence its influence is soft and
must be ablated. Do not use existing class labels to choose which controls to keep.

No control qualifies -> `controls=[]`, not an error. Failed control representation
-> no valid control from that region, with a reason. Target view/mask never changes.
Pool control regions from the SAME selected RGB and the SAME representation family
as the target. For owner-conditioned resampling, key occupancy uses the control
owner's full visible support, while final control pooling uses only C_j.

## 4. Model assets and aligned representations

### 4.1 Locked FC and original v2 reference

Retain the exact original FC weights, text vectors, 14 templates, class ID order,
FP32 precision and runtime settings. Reuse `image_tensor` (short side 800, long
side max 1333, PIL bilinear RGB, pad to multiples of 32); the architecture suffix
`320` is NOT its current input resolution. The feature input is
`extract_features_convnext(...)["clip_vis_dense"]` AFTER norm_pre and BEFORE the
visual prediction head. Its channel count is read from the tensor, never assumed
identical to the text embedding dimension.

Normal and area-fallback v2 paths remain exact for B1 and coarse controls. The v2
fallback is already implemented, not a proposed novelty. Do not replace normal
pooling with area pooling globally. Cache keys must include actual feature layer,
model, preprocessing, image, mask, operator revision and numeric settings.
The inherited `scientific_key` hardcodes v2 pooling: use it only for B1 equality,
not as an AnyUp cache key. New keys also include output size, guidance normalization,
geometry evidence, resampling factor and control-region identities. If a separate
AnyUp environment is required, return raw pooled FP32 visual vectors to the original
FC environment for the frozen visual head; do not silently change the FC framework
or preprocessing by migrating the whole pipeline.

### 4.2 AnyUp asset, not a moving default

Use `wimmerth/anyup` at `351807a9c4287368732cc247f26c7c81c9139af4`, entry `anyup`,
`use_natten=False`, frozen/eval, FP32, original paper checkpoint:
`https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth`.
Verified release bytes: 3,540,612. SHA256:
`9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2`.
Download once and verify; load state_dict strictly. Do not substitute the newer
multi-backbone checkpoint, NATTEN, a random model, or a different upsampler. Preserve
CC-BY-4.0 attribution and upstream modification notices; do not republish weights.

Read the pinned `hubconf.py`, `anyup/model.py`,
`anyup/layers/attention/chunked_attention.py`, and `attention_masking.py`.
The implementation uses image/feature encoders for Q/K and an averaged-head
attention matrix to weight UNPROJECTED low-resolution V. It is not valid to
upsample projected class probabilities and call that AnyUp feature upsampling.

### 4.3 Aligned guidance, masks and depth

Build a separate AnyUp guidance tensor from the identical RGB resize used by FC,
convert to [0,1], normalize with ImageNet mean (.485,.456,.406) and std
(.229,.224,.225), then pad to FC's exact padded H/W with normalized zero.
Do not feed the FC-normalized tensor as AnyUp RGB guidance. Keep all low-resolution
FC features on their original padded grid; do not crop away feature cells and
silently shift registration.

Resize categorical visible-owner maps with nearest neighbor to the identical
unpadded RGB size and pad unknown zero. Resize measured depth with nearest neighbor,
pad invalid zero, retain an explicit validity map. Pool occupancies and valid-depth
moments by area to each needed grid. No bilinear interpolation of numeric owner IDs.
The target recognition mask follows original nearest mask resizing/padding exactly.
These fixed transformations are experimental choices, shared by ordinary and
geometry-conditioned AnyUp.

Output grid is `(padded_H//4,padded_W//4)`. Use this SAME grid and area-weighted final
mask pooling for bilinear, AnyUp, owner/depth AnyUp and no-depth arms. Final region
feature is `sum(H_q*w_q)/(sum(w_q)+1e-8)`, shaped Bx1xC, then the original frozen
`visual_prediction_forward_convnext` head and L2 normalization. Do not normalize
individual low-resolution value cells before averaging; use original raw V. The
head expects pooled visual channels, not text-space vectors. The current ConvNeXt
head ignores its masks argument, but pass the valid aligned mask for interface
consistency. Score against the exact original text matrix.

Bilinear control: `F.interpolate(dense,size=output_hw,mode='bilinear',align_corners=False)`.
AnyUp control: official forward with the specified guidance, dense features and
output size. All nontriggered targets preserve B1; controls are computed only for
triggered target decision arms requiring them.

### 4.4 Bounded memory and query reuse

Use batch=1, query chunks of 256. On genuine OOM, allow only 128 then 64 query chunks
with the same tensors, output grid and precision; record attempts. Do not reduce
resolution or precision. Keep same successful chunk setting across comparable arms.
A patched stream may compute original attention once and obtain ordinary,
owner/depth and owner-only outputs from it, saving repeated Q/K work. It must first
match the pinned ordinary AnyUp on one real frame with atol=rtol=1e-5 and matching
region classes. One full-frame control validation is enough, not a large test suite.

Never persist a full QxK attention tensor for every frame. Accumulate region sums
per query chunk and discard attention; do not retain all high-resolution C-channel
maps across scenes. Query-subset execution is optional only if it preserves the
full Q convolution/positions and exact original window mask. No change to window
ratio, Q/K encoders, feature normalization, head averaging or pretrained weights.

If checkpoint/dependency/true hardware execution is unavailable, mark only dependent
AnyUp arms BLOCKED and finish coarse contrast/margin and diagnostic branches. A
numeric invariant violation is a bug to fix, not a per-object silent baseline fallback.
Zero area mass for a nonempty, correctly aligned bound mask is an invariant failure
and must be fixed. A finite zero projected feature vector may fall back to B1 ONLY
under an explicit `REPRESENTATION_UNAVAILABLE_KEEP_B1` outcome and must be counted.
Do not report a whole method as successful AnyUp execution if every target fell back.

## 5. Proposed owner/depth-conditioned resampling (N2)

This is our prespecified adaptation, NOT an existing AnyUp or GeoGuide method.
Let A[q,k] be original AnyUp averaged-head attention within its unchanged local
window. Let V[k] be original FC values. For each owner i whose target or control
region is being read, obtain:
- p_i[k]: area occupancy of visible owner i at FC key resolution;
- p_unknown[k]: occupancy of unknown/invalid/ambiguous owner pixels, including pad;
- d_i[k]: area-weighted measured depth of owner i pixels at the key, defined only
  when owner-i depth mass is positive;
- d_i[q]: the analogous owner-i depth at query resolution, used only for positive
  target/control query mass. Unknown depth is neutral, not contradictory.

Define
```
g_owner(i,k) = 0.05 + 0.95 * clamp(p_i[k] + p_unknown[k],0,1)
z = min(abs(d_i[q]-d_i[k]) / (0.02 + 0.02*d_i[q]), 3)
g_depth(i,q,k) = exp(-z) if both owner-i depths exist, otherwise 1
g = g_owner * g_depth
A_i[q,k] = A[q,k]*g / sum_k(A[q,k]*g)
H_i[q] = sum_k A_i[q,k]*V[k]
```
Only target/control queries with positive pooling mass are required. Since g_owner
has a positive floor and g_depth >= exp(-3), nonempty original attention remains
normalizable. If original AnyUp has an all-masked query, record representation
unavailability; do not create a fictitious uniform feature. Preserve A's zero mask
entries. Unknown-owner keys are NOT treated as definite other-object evidence.

The NO_DEPTH ablation sets `g_depth=1` and changes nothing else. Neutral geometry
(g=1) must reproduce original AnyUp within declared FP32 tolerance. Apply the gate
AFTER original averaged-head attention, before the einsum with unprojected V. Do
not describe it as a pre-softmax logit gate or a multi-head Bayesian posterior.
Do not alter learned parameters. A compact new wrapper/patch may expose attention
chunks; keep the pinned upstream checkout unmodified and publish the patch/wrapper.

This tests whether projected 3D ownership and measured depth restrict feature
borrowing to more relevant support. It does NOT guarantee recovery of information
already destroyed by coarse features, and it does not learn object boundaries.

## 6. Proposed geometric contrast + selective recovery (N1)

Let s_T(c) be the full-vocabulary cosine scores from the target representation.
For valid competing-support controls in the same representation family, let
`b(c) = max_j s_Cj(c)` and `b_centered = b - mean_c(b)`. If none exists, b_centered=0.
Let eta=1-purity from the common ORIGINAL coarse occupancy (Section 3). Define:
```
r(c) = s_T(c) - 0.25 * eta * b_centered(c)
c_star = first argmax in the inherited class order
margin = largest(r) - second_largest(r)
accept if margin >= 0.01, else defer
```
On a nontriggered candidate return B1 verbatim, bypassing both contrast and gate.
On a triggered candidate the MARGIN control uses `r=s_T` with the same threshold.
Missing controls remove the contrast term; they do not count as negative evidence.
The low margin may still cause deferral as in the simple control. Do not require
Native/FC category agreement, since this would reject known opportunities where
only FC recognizes the object correctly. Do not treat a same-class neighbor as a
proven negative: it can influence scores only through the explicit soft rule.

N1 uses v2 coarse target AND v2 coarse control vectors. The COMBINATION uses the
owner/depth-conditioned high-resolution target AND control vectors, with the same
trigger, eta, lambda and margin threshold. Bilinear/ordinary AnyUp/N2 alone always
accept valid representations as B1 does; do not add a hidden margin threshold.

This is a relative evidence score, not a calibrated posterior, uncertainty bound,
causal effect, or guarantee of improved AP. Centering removes an irrelevant scalar;
the intended intervention is contrast between different class responses. The
reference checks must include lambda=0, empty controls, duplicate class ties,
nontriggered identity and a correctly classified same-class-neighbor example that
could be harmed. Failures of the hypothesis are scientific results, not bugs.

Store separately `feature_available`, `proposed_class`, `accepted`,
`defer_reason`, `raw_cosines`, `decision_scores`, `margin`, control identities,
purity and representation-fallback reason. Do NOT set old `available=False` to
conceal a semantic deferral or write corrected scores as ORIGINAL_FC_COSINE.

## 7. Exactly nine fixed outputs; no hidden search

| ID | Definition | Question |
|---|---|---|
| EV00_D2 | inherited D2, no recovery | Strong no-recovery baseline |
| EV01_G1_V2 | inherited complete A3 v2 with optimized R2 execution | Current method to preserve |
| EV02_MARGIN | B1 + triggered-target margin gate only | Is a simple threshold sufficient? |
| EV03_CONTRAST | B1 + Section 6 coarse contrast and same margin | Does competing-support evidence add value? |
| EV04_BILINEAR | triggered target upsampled bilinearly, same area pool/head | Is interpolation alone sufficient? |
| EV05_ANYUP | triggered target with original AnyUp, same pool/head | Is importing AnyUp alone sufficient? |
| EV06_OWNER_ANYUP | Section 5 owner/depth-conditioned AnyUp | Does the geometry condition add value? |
| EV07_COMBINATION | EV06 representation + Section 6 decision | Are better representation and verification complementary? |
| EV08_NO_DEPTH | EV06 with depth factor set to 1 | Does depth contribute beyond owner support? |

Every row covers all 26 scenes, even where it is a no-op. There are 234 required
scene/method records and 18 ordered full-cohort pools. The 52 B0/B1 rows can be
identity-proven imports; the seven new methods require 182 scene/method outcomes.
Do not rerun a scorer for truly identical prediction inputs; link the alias and
report unique evaluated inputs separately. An aliased/no-op condition is not an
independent successful mechanism. Do not add U2 as a tenth experiment; keep its
published values only as explicitly attributed context when discussing previous
work, never mix old U2 timings with new methods' timing as a new paired comparison.

The two fixed pilot scenes are office1 and scene0011_00. Use them only to check real
inputs, padding, normal/fallback paths, tensor dimensions, operator neutrality and
output interfaces. Do not choose thresholds, resolution or model by their metrics.
Their final successful evidence can be reused in the full frozen run by content
identity. Do not add a second exploratory slate after inspecting final results.

## 8. Output, scoring and diagnosis interfaces

Use a new `evidence_exploration` module, not an edit that corrupts previous source
records. Suggested modules: binding.py, evidence.py, anyup_adapter.py, verifier.py,
outputs.py, evaluation.py, analysis.py, workflow.py, publication.py. Consolidation
is allowed; responsibilities and executable entrypoints are not optional.

Separate predictor inputs (RGB/depth/pose/predicted geometry/ordered text) from
post-prediction GT/evaluation inputs. `CommonInputs` contains evaluator projections;
expose a reduced read-only view to scoring rules. The release export may use the
standard target projection to rank/output predictions, but that mapping must not
select, prune, classify or protect candidates. No training on evaluator GT.

Compute incumbent D2 labels once with original `fuse_readout`, or verify imported
D2 against those exact sources/scalars. Pass accepted recovery labels and immutable
incumbent D2 labels to `construct_output`. A deferred owner simply is not appended;
raw geometry is not deleted. Keep old metadata, add new method/source identities.
The old `build_method_outputs` requires raw cosine argmax and unconditional available
sources; do not disguise new verifier outputs to satisfy it. Use the lower-level
construct function in a correctly named adapter.

Do not change source coordinates, faces, original owner/painted-owner arrays,
nearest-neighbor projection, minimum target-instance size, category order or official
current-class area score. Recompute official ranks after every class/acceptance
change. Feature reliability scores MUST NOT replace official AP confidence.

Use expanded native evaluation registry containing actual accepted owners so newly
recovered masks are scored. Evaluate all GT regions, including missed objects and
semantic-unknown errors. Reuse `SceneEvaluator`, `expanded_native_registry`,
`fraction_metrics`, `trace_class_metrics` and `released_pool_with_classes` through
a new protocol adapter. Old task/phase freeze guards are not an API to spoof.

APall uses the actual released nine mathematical thresholds .50,.55,...,.90; AP25
separate. Preserve the loaded floating values and strict comparisons. Machine
metrics are fractions, displayed metrics percent, deltas percentage points. Full
cohort pools use all ordered scans, not means of scene AP. Keep ties and ignore
rules, and record added definite/ambiguous TP/FP score entries AND unique GT match
changes as distinct diagnostics. Do not call recovered-owner count true recall.

Required paired analyses: contrast vs margin; AnyUp vs bilinear; owner/depth AnyUp
vs AnyUp; owner/depth vs no-depth; combination vs its two components. Report all
five metrics, real accepted/relabelled/deferred counts, removed baseline FP, lost
baseline TP, corrected category errors, geometry-insufficient cases and absent-
class cross-scan FP. Do not infer which category caused a pooled loss from a scene
mean alone. Publish actual class-wise pooled deltas.

## 9. Implementation freeze, research selection and non-regression target

Before new full-cohort scoring, commit the implementation/source/weights/constant
freeze. Prediction records must lock before their GT scoring. Any real bug fix after
freeze produces a new explicit patch identity and invalidates only dependent outputs.
Do not rewrite old successful results, hide a failure, or silently redefine a metric.
There are no new parameter fits in this wave. Pretrained AnyUp and inherited fitted
Q/temperatures mean the entire system is not "all-components-training-free".

All nine full outputs are run even if early cases look negative, subject to genuine
resource failures. Selection is a **retrospective research recommendation on exposed
cohorts**, NOT independent validation. It is acceptable to identify a promising
candidate; it is not acceptable to relabel the same cohort as a held-out confirmation.

Let B1 be current complete A3 v2, and B0 be D2. Read unrounded fractions. Numerical
comparison tolerance is 1e-10 (not an allowed scientific regression). Candidate i
satisfies `TARGET_MET` only if:
- all five Replica metrics >= B1 minus numerical tolerance;
- all five CF18 metrics >= B1 minus numerical tolerance;
- CF18 APall > B0 APall + numerical tolerance;
- CF18 AP50 >= B0 AP50 minus numerical tolerance.

Additionally label `MATERIAL_TARGET_MET` when CF18 APall exceeds B0 by at least
0.10 percentage points. This is an engineering magnitude flag, not significance.
A method which improves CF18 but lowers any Replica metric is a disclosed tradeoff,
not a successful "no-regression" result. Check all five metrics, not just rounded AP.

Among passing methods choose lexicographically larger CF18 APall, CF18 AP50,
Replica APall, CF18 mIoU, then the lower method index as the simplicity tie-break.
Ties use the same numerical tolerance. Simple controls can win. If none passes,
retain B1 as recommendation, publish the full Pareto/tradeoff table, and do not
launch an unbounded rescue sweep. Always use ONE method ID across both datasets;
no per-dataset switch, oracle scene routing or protected object lists.

## 10. Compute, practical tests and bounded execution

No new TSDF/CropFormer/SAM/Native N/Q inference, new dataset or new training.
Load at most one scene's active geometric evidence and one frame's dense/attention
state at a time. Scientific acquisition can reuse exact G1 views, dense FC tensors,
normal target vectors and text. Changed target/control pools need new operator keys.
AnyUp is new inference even when its FC input was cached. Count FC image inputs,
AnyUp image-encoder/QK computations, query chunks, original vs geometry-weighted
attention applications, target/control pools, head projections and actual timings.

Structural limits: at most 128 fixed candidates per scene, 3,328 in 26 scenes;
at most two controls each, 6,656 control records; no more than the selected-G1 frame
union for new RGB processing. Its absolute upper bound is 3,328, usually much less.
Only triggered frames require AnyUp. FC is encoded at most once per required unique
frame for acquisition, and AnyUp Q/K once per triggered frame. The three AnyUp
resampling variants share original attention chunks. Optional full-frame AnyUp
validation on the two pilots is extra and separately counted; no other duplicate
model experiments. CPU-only shared evidence preparation is allowed and counted.

Use one GPU worker on the bound device, three CPU evaluation workers with at most
four BLAS threads each, and inherited 4-thread raycasting. Do not assume free remote
GPUs or kill unrelated processes. Reuse valid caches to avoid repeated experiments,
not to change the definition of cold latency. One asset download verification, one
focused unit-test invocation, and the two real integration pilots suffice. Add
regression tests only for actual new failure modes; no full dynamic/CROVE test suite,
security fuzzing, all-history hashing, enormous release-file revalidation, or quotas
on test count. The tests must hit actual production adapters, not only toy kernels.

Required focused properties: neutral geometry equals original AnyUp; unknown-depth
factor is neutral; window mask zeros stay zero; same-class controls are not assumed
false labels; empty control is neutral contrast; trigger-off returns B1 exactly;
raw score availability differs from deferral; incumbent support/labels unchanged;
new accepted registry reaches the scorer; same images/masks/feature channels across
controls; no-op aliases are honest; all units and cohort denominators are correct.

Allow at most two retries after a genuine failed execution leaf, preserving costs
and failure logs. OOM chunk changes use only the specified sequence. An all-failed
model installation is BLOCKED, never fake baseline success. Conversely, a valid
algorithm that accepts zero objects is a valid zero-recovery prediction and MUST
be evaluated; do not repeat the earlier mistake of treating every deterministic
method failure as a permanently missing benchmark cell.

## 11. Timing: secondary to discovery, but not optional for a promoted method

After metric selection freeze, choose at most FOUR unique methods for paired timing.
Begin with B1 and the chosen target-passing method. If no method passes, the
exploratory timing candidate is the first executable method in this fixed order:
EV07_COMBINATION, EV03_CONTRAST, EV02_MARGIN (or none if all are blocked).
Then append the candidate's direct comparators from the fixed map below, remove
duplicates preserving order, and take at most four total IDs. No padding is needed.

| Candidate | Ordered direct timing comparators |
|---|---|
| EV02_MARGIN | EV01_G1_V2 |
| EV03_CONTRAST | EV02_MARGIN |
| EV04_BILINEAR | EV01_G1_V2 |
| EV05_ANYUP | EV04_BILINEAR |
| EV06_OWNER_ANYUP | EV05_ANYUP, EV08_NO_DEPTH |
| EV07_COMBINATION | EV03_CONTRAST, EV06_OWNER_ANYUP |
| EV08_NO_DEPTH | EV05_ANYUP, EV06_OWNER_ANYUP |

Report any accuracy comparison not newly timed. This is not a model selection by
latency. Do not choose the baseline using historical speed.

Eight Replica scenes, two repeats, at most 64 cold scene calls. Repeat 2 reverses
scene and method order. A complete no-op method uses the actual fast path but the
inherited fixed-view setup cost still applies if required by its deployed code.
No benchmark-wide repeated model search. If all candidate branches are blocked,
report timing incomplete rather than timing substitute code under their names.

Use same physical GPU/CPU, model(s) resident and common geometry/NQF resident. No
previous recovery views, feature tensors, output or posterior caches in a timed call.
Clear these between methods/repeats; within-call same-frame reuse is allowed. Keep
OS page cache explicitly uncontrolled, as before. Compute G1 search with R2, then
any added selected-frame owner/depth evidence, FC, AnyUp and accepted outputs within
the timer. Model loading separately measured; pretrained AnyUp is not free. Record
end-to-end incremental wall time with CUDA synchronized at boundaries, nonadditive
GPU event times, peak memory and five stage times. Do not move evidence generation
outside the timer. Keep all valid repeats, not the fastest one. Record correctness
parity between acquisition and timed prediction after timing; do not count parity
checking as inference latency. No online or 30 FPS claim.

## 12. Tables, handoff and GitHub publication

Create three compact research tables (MD/CSV/JSON; LaTeX supported, PDF optional):
1. all nine method rows x two cohorts, APall/AP50/mIoU, delta vs B1 and target flag;
2. mechanism comparisons: margin/contrast, bilinear/AnyUp/geometry/no-depth, and
   combination; accepted/deferred/relabelled, lost TP/removed FP and class deltas;
3. selected same-boundary timing rows, including runtime and added computation.
All five metrics and all per-scene/class results go into companion machine files.
Do not overwrite the previously published manuscript tables as if a research winner
were a prospectively confirmed final method. A later paper can choose the smallest
supported story; the present report must keep all controls and negative results.

One result store must generate every table and cell provenance. Write four reports
under `docs/paper/static_ovmap/`:
`EVIDENCE_EXPLORATION_RESULTS.md`, `EVIDENCE_EXPLORATION_HANDOFF.md`,
`EVIDENCE_EXPLORATION_SELECTION.md`, `EVIDENCE_EXPLORATION_CLAIMS.md`.
Publish actual modules, wrapper/patch, narrow tests, configuration, upstream and
weight identity manifest, cached-feature dependency references, candidate decisions,
234-row/18-pool coverage, diagnostics, timing records, costs, failures and handoff
under `artifacts/static_ovmap/evidence_exploration_v1/`. Aim for <=50 MiB in Git;
large tensors/maps/weights stay on shared storage with exact reconstruction commands.
Do not publish credentials, licensed ScanNet RGB/meshes or model weight files.

Before release, perform ONE requirement review mapping each deliverable to an
actual artifact. Keep implementation, asset availability, scientific coverage,
performance target, timing, research selection and publication as separate fields.
If some branch is blocked, publish everything completed plus a precise dependency
list and resumable command; do not call the entire study COMPLETE. Do not stop at
"tests passed" while main outputs or push are missing.

Commit and normal-push the target branch. Verify local full HEAD against
`git ls-remote origin refs/heads/research/ovimap-evidence-exploration-v1`.
Only identical full SHAs permit `PUSH_VERIFIED`. Put the final publication receipt
outside the commit it attests to, avoiding a self-referential commit loop. A push
failure needs the actual error and retry command, not a request to reveal tokens.
Keep deployment `N0_UNCHANGED` regardless of retrospective research selection.

Final response must link the four GitHub reports and commit; show all nine rows,
exact target gate, baseline/simple-control comparisons, actual inference and costs,
what the mechanism evidence supports, and remaining blocks. Never claim a new
probabilistic theory, universal novelty, unseen confirmation or improved geometry
from this fixed-geometry exploratory study alone.
