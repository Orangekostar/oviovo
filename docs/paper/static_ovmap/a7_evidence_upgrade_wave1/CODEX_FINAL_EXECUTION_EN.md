# Final execution directive — A7 evidence-upgrade wave 1

**Date:** 2026-09-29  
**Repository:** `Orangekostar/oviovo`  
**Base commit:** `1074eb746808ca6882b30ccd827167474bb7eb8f`  
**Upstream OVI-MAP:** `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`  
**Task branch:** `research/ovimap-a7-evidence-upgrade-wave1`

Read this file, `PROTOCOL_SPEC.json`, and `SOURCES.md`. This is an instruction to implement, measure, select research candidates, and publish—not to return another plan. The named `AW_*` methods and new runner are proposed interfaces to implement. They are not already available merely because this document names them.

## 0. Scope and the result we need

Execute the first wave of the supplied literature roadmap: **E01, E02, C0, E03, E04**, in that order. These are independent families, not a growing chain where each experiment silently inherits the preceding winner. Compare every family to the same frozen A7 base. Only Section 11 permits one explicitly selected two-slot composition.

Keep `RV_A7_COS_REFIT` as the primary base, `RV_A7_COS_FIXED` as a secondary reference, and N0 as the mandatory original-system reference. These A7 variants were discovered on already exposed data: describe this study as development and historical regression, not new blind confirmation.

The research questions are:

1. Does a robust final readout use the **same** paid Q observations better?
2. Does a cleaner 2D recognition mask improve an **unchanged** 3D instance?
3. Does a larger static encoder or genuinely region-tuned encoder provide better evidence?
4. Can localized concept evidence resolve disagreements that score voting cannot?

Do not assume any module improves scores, do not preserve an elaborate method when a simpler control is as good, and do not equate a working adapter with a positive scientific result.

**Explicitly deferred:** E05–E12, new 3D proposals, CropFormer replacement in mapping, TSDF changes, owner reassignment, Gaussian mapping, test-time prompt optimization, new Q/selection-head training, new datasets, and new paper-writing tasks. This wave uses SAM only to construct recognition evidence; it does not change the geometric map. A completed wave is not completion of the entire 24-paper roadmap.

All currently executable wave-1 families must be measured, even if an earlier family has no gain. External model access can block its own leaf, not E01 or unrelated families. There is no fresh-scene prerequisite in this wave.

## 1. Evidence-based corrections to the roadmap

The attached research roadmap motivates experiments but is not a runnable specification. This directive resolves the following implementation details. Source IDs refer to `SOURCES.md`.

| Verified behavior | Required implementation consequence |
|---|---|
| Historical Q aggregates are recovered from final retained requests, raw six-crop means, and original area weights. [L01] | E01 must recover those actual vectors and retention IDs, not invert class probabilities or replay the controller with a changed aggregator. |
| `native_crops()` rejects a union mask that fails to contain every target pixel. [L02] | A repaired SAM mask can legitimately shrink. Add a separate recognition-mask crop function; do not OR the repair back into the original target to satisfy this assertion. |
| `_load_request()` verifies original local/global union identity. [L03] | Verify and load the original request first; attach a separate semantic-mask override. Never relabel a modified mask as the original union. |
| `direct_readouts()` whitelists native/SigLIP2/WOW and can return an incumbent label after no successful view. [L04] | Add real task-local adapters. A fallback label is not genuine new evidence and must not contribute another fusion vote. |
| A7 changes N0 scores to cosine while preserving geometry and frozen numerical ranks. [L05] | Preserve the exact A7 fold/final temperature and score definitions in untouched slots. Do not restart from old M2 canonical-relative N0. |
| OVR MaskPooling uses `mask > 0` after interpolation; its public classification path outputs probabilities. [X02,X03] | Supply signed mask values, preserve padding, and export unscaled region/text cosine before softmax. Do not treat returned probabilities as cosine logits. |
| OVR's public config selects a ConvNeXt-L OpenCLIP backbone; its region branch can run without panoptic heads. [X01,X04] | Use a matched original frozen backbone control. Do not build/train an unnecessary segmentor or compare only OVR against a different SigLIP architecture. |
| SAM3 derives array dimensions using trailing axes and stores post-sigmoid masks under `masks_logits`. [X07] | Pass PIL RGB, not ambiguous HWC arrays. Record the actual value domain and do not sigmoid it a second time. Reset prompt state between classes. |
| The main benchmark is current-class official export plus released dataset pooling. [L06] | Keep scene means and frozen-rank diagnostics, but do not select a favorable aggregation to claim superiority. |

The previous 3-sigma outlier idea is not directly applicable to our short view lists. With population standardization on n observations, the maximum absolute standardized deviation is at most sqrt(n−1); a strict deviation beyond 3 cannot occur for n≤10. E01 is therefore an explicitly specified small-sample adaptation, **not** a claimed ReLaGS reproduction. [R01]

## 2. Workspace, input binding, and dependencies

### 2.1 Workspace

Create or resume a dedicated worktree from the audited base. Do not reset unrelated work, switch the base to a newer unrelated commit, force-push, or overwrite historical outputs. Record the actual source HEAD and relationship to the base. A task branch that already contains a compatible partial run may be resumed.

Default external output root:

`/mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001`

Bind existing assets in this order:

1. `/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001/source_binding.json` and the resolved scene configurations it references.
2. That root's `calibration/new_folds.json`, `calibration/new_final.json`, locked predictions, source score files, recovered aggregates, metrics and official-pool receipts.
3. Its bound Replica transfer and ScanNet composition roots, original semantic-request manifests, source predictions, Q paid/retained request records, RGB, depth, and captured request arrays.
4. Repository publications are locators and identity references; an external path in a published JSON is not proof that the bytes exist locally.

Reuse `m2_reviewer_study.scores.read_scene()` to obtain `SourceEvidence` and `cosine_native`. Resolve environment paths, local native/S2 models and text caches from actual receipts. Do not infer them from a remembered home directory. Retain the original image orientation, coordinate frame, precision, tokenizer, vocabulary IDs and projection.

Read each consumed heavy asset identity once and memoize it; do not recursively rehash the prior 90 MiB publication or all old files per method. A missing path permits one focused recovery through adjacent receipts or known mounted roots, not a recursive machine-wide search. Record real missing assets and proceed with independent leaves.

### 2.2 Authorized external models

Official model/code downloads are authorized **for this table only**, subject to existing licenses and account access. No dataset download is authorized. Do not bypass a gated model using an unverified mirror. Do not accept credentials in reports or upload tokens. Freeze each actual checkpoint revision and SHA256 before the first model experiment. Weight loading was not performed by the author of this directive.

| Purpose | Exact identity / priority |
|---|---|
| Existing native/S2 | Reuse bound local models, processors, text caches and original FP32 environments. |
| E02 SAM2 | `facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4`; `sam2.1_hiera_large.pt`; `configs/sam2.1/sam2.1_hiera_l.yaml`; official URL in JSON. |
| C0 capacity | `google/siglip2-so400m-patch14-384@f3b7a187cd133857ff43c0dccfe88f8268372549`; use its own tokenizer and processor, not S2-L's text vectors. |
| E03 OVR | `nickormushev/OVRCOAT@9fd9450d22852d269d426b521663a127f3983a4b`; author's checkpoint link in JSON; matched original OpenCLIP `convnext_large_d_320 / laion2b_s29b_b131k_ft_soup`. |
| E04 SAM3 | `facebookresearch/sam3@2345a4ad109ac29c569da749c91d84f10dc08c40`; official `facebook/sam3`, `sam3.pt`. Resolve and freeze the original SAM3 checkpoint revision; **not** SAM3.1. |

Total new model download allowance is 25 GiB. Reuse exact verified local copies first. An oversized/inaccessible artifact is a leaf prerequisite failure, not permission to choose a smaller unrelated model or silently remove its paired control. Record licenses, upstream training datasets as documented, known benchmark overlap, unresolved overlap, parameter counts and checkpoint keys. Absence of public training details is recorded as unknown, not assumed clean.

Reuse working native and S2 environments without upgrading them. For a new worker, first reuse a compatible existing environment. Otherwise create one isolated environment and resolve dependencies against the pinned upstream code. Save `pip freeze`/Python/Torch/CUDA after the real smoke. Do not run a speculative environment-version grid. OVR region-only execution can avoid the segmentation pixel-decoder CUDA build; do not blindly follow the legacy INSTALL path `fcclip/...` when the pinned tree is `ovrcoat/...`.

Old/new environments communicate via plain manifests and NPZ outputs. A worker's actual shape and dtype are explicit; there is no fixed 1024-dimensional assumption for every encoder. Run one visual model per GPU at a time. Sequential workers may reuse the same GPU.

## 3. Data roles, primary controls, and experimental matrix

Use the already captured full schedules of exactly these scenes:

- CAL: `scene0056_00`, `scene0534_00`.
- Replica: `office0`, `office1`, `office2`, `office3`, `office4`, `room0`, `room1`, `room2`.

CAL has prior Q/model-selection exposure; it is development data, not an independent validation of the whole system. Replica is historically exposed regression data. No ground-truth-dependent selection of regions, views, prompts, model branches or runtime actions is allowed. Existing labels may be opened only by calibration, evaluation and post-prediction diagnostics.

Keep five controls: `N0`, `Q_GAIN`, `S_SIGLIP2_AREA`, `RV_A7_COS_FIXED`, `RV_A7_COS_REFIT`. Reuse exact artifacts where dependencies match; validate the reconstructed A7 label/owner/rank identity once, and compare any re-evaluated control against its same-mode historic result. Published rounded values are references, not data to paste into fresh rows.

For every new source variant below create two outputs:

- `<variant>_DIRECT`: this source's top-1 on all genuinely available owners; the **N0** label on owners without that source. Mark technical and target-cap fallback. This is a source-plus-explicit-incumbent diagnostic, not a claim that the source classified fallback owners.
- `<variant>_A7`: replace exactly the indicated slot of **A7_REFIT**, retain the other two real sources, and use equal probability fusion with one separately calibrated scalar for the changed slot (Section 5). Do not append a fourth source.

| Variant | Slot replaced | One change |
|---|---|---|
| `AW_E01_RAW_EQ` | Q_GAIN | Equal mean of the same raw per-view six-crop means. |
| `AW_E01_UNIT_EQ` | Q_GAIN | Equal mean after normalizing each per-view vector. |
| `AW_E01_GMED` | Q_GAIN | Smoothed geometric median of those same unit vectors. |
| `AW_E02_GLOBAL` | S_SIGLIP2_AREA | Same bbox/views/model; black-background mask is captured global target only. |
| `AW_E02_SAM2` | S_SIGLIP2_AREA | Same bbox/views/model; black-background mask is one SAM2 box repair. |
| `AW_C0_SO400M` | S_SIGLIP2_AREA | Same union six-crop inputs; larger matching image/text encoder. |
| `AW_E03_FC_FROZEN` | S_SIGLIP2_AREA | Matched original frozen ConvNeXt-L region readout on captured global masks. |
| `AW_E03_OVR` | S_SIGLIP2_AREA | Region-tuned checkpoint, otherwise identical to the preceding row. |

Add E04's three complete-map decisions: `AW_E04_SHORTLIST`, `AW_E04_SPATIAL`, `AW_E04_MIX50`.

Thus, if all model and input prerequisites are available: **19 new methods + 5 controls = 24 methods × 10 scenes = 240 scene-method records; two rank views = 480 scene-rank records; 24 × 2 = 48 Replica pool records**. These are reporting records, not independent trials or image runs. Identical predictions may share evaluator work while retaining method provenance. Inaccessible external leaves stay visibly blocked with actual reasons; no null-filled report may call them measured.

All original positive owners, including excluded/unsupported targets, remain in whole-map evaluation. Do not filter the map to the diagnostic correspondence set or to objects where a new model succeeds.

## 4. Shared request and source contracts

### 4.1 Reuse—not reselect—the original request manifest

For E02, C0 and E03, reuse the existing `semantic_requests.json` for each scene. This fixes the historical label-free cap of up to 128 final targets, up to 3 views each, original visibility order, original bbox and segment-lineage proof. Do not call a modified view selector and accidentally create a different target/view study. If a bound source manifest is genuinely missing, reconstruct it once from the same capture using the pinned `prepare_semantic_manifest()` and require identity/target/view parity where an old record exists.

Load RGB as HWC `uint8` in RGB order, and masks as full original-resolution booleans. Use captured `{request_id}_target` and `{request_id}_union` arrays. Preserve the original bbox's min/max coordinates and historical exclusive upper image-slice convention. New model preprocessing may resize inputs, but never mutates the capture or projection.

### 4.2 Source evidence

Every variant produces all-owner rows with: original request IDs; derivative request identity; actual model/checkpoint/tokenizer/processor identity; mask-mode identity; view success status; full class score vector; exact ordered `valid_ids`; genuine availability; source top-1; aggregate feature when applicable; original visibility weights; fallback reason; timing and operation references.

Unavailable sources have no score vector. Do not fabricate one-hot probabilities, duplicate an N0 fallback as evidence, or invert probabilities to pretend an embedding exists. For available cosine sources, explicitly verify that the vector argmax reproduces the stated label. The fusion *slot* can remain `S_SIGLIP2_AREA` for compatibility, but its metadata and report must say `OVR`, `FC_FROZEN`, or `SO400M` when that is the actual source.

Missing historical bytes for an allegedly available source are an input error, not a new source unavailability event. Resolve or mark the dependent leaf blocked; do not improve scores by silently withholding difficult cached evidence.

Derivative cache identity includes actual model/processor/precision, source RGB SHA, original request identity, semantic-mask SHA, bbox/crop geometry, score representation, text vocabulary identity, and adapter source revision. Model-changing and mask-changing cache entries must not collide with the original request cache.

### 4.3 Crops and paid evidence

For S2-style branches use exactly six views of each region: three original-image crops alternating with three black-background crops at expansion factors 0, 0.1 and 0.2. Normalize each crop feature, average the six **without normalizing that view mean**, then use original `visible_target_pixels` weights across views and normalize once at the final aggregate. Use each model's own unit text features to compute cosine scores. No three extra background-only crops are requested in this wave.

For E02 the three raw-image crops are byte-identical to the old crops. Reuse their exact saved crop vectors when identity matches; encode only changed foreground crops. If those individual vectors are unavailable, encode all six and report the extra physical cost. Never reconstruct individual crops from a saved average.

Changing a foreground mask does not change the area weight in this controlled experiment. This is deliberate isolation, not a claim that original area weights are optimal for the new mask.

## 5. Scalar calibration and research nomination

Freeze N0's A7 cosine representation. On CAL use the original opposite-scene A7 fold settings for untouched slots; on Replica use the final A7 source settings. N0 uses the new-final `cosine_N0` temperature, not the old canonical-relative N0 temperature. Q and S2 use their matching historical fits unless their slot is the one being replaced.

For every changed source, fit exactly one temperature using the existing scene-balanced source NLL recipe, the same strict class-independent unique geometry correspondence at IoU > 0.5, valid semantic IDs, and genuine evidence. Fit two opposite-scene CAL folds and one final fit on both CAL scenes. Bounds `[0.01,2]`, log parameterization, `maxiter=64`, `xatol=1e-4`; fewer than 5 unique objects or 2 classes gives the explicit 0.07 fallback. Record bound hits and counts. No target-correctness filtering other than the predefined geometric correspondence; no new thresholds, weights or prompt search on Replica.

This compares complete readout pipelines under equal scalar-calibration budgets. A change in calibrated outcome is not attributed solely to an uncalibrated pooling formula. Save pre-temperature scores and fitted scalars so the score/readout/calibration chain remains inspectable. There is no obligation to create an extra exhaustive fixed-temperature matrix.

E04 has no fitted temperature or learnable gate: its restricted probability and spatial normalization are fully specified in Section 10.

After CAL experiments, nominate one research candidate from the measured `_A7` variants, `AW_E04_MIX50`, and the optional `AW_COMBO_QR` after its CAL evaluation, with A7_REFIT and N0 allowed as incumbents. Rank by **CAL out-of-fold official-pooled APall**, then mIoU, then AP50; differences within 1e-10 are ties. Break remaining ties by fewer exact unique required visual operations, then registry order. Report the full vector and worst-scene changes, not just a winner label. No universal non-degradation-per-scene condition is imposed.

Freeze the nomination and optional composition choices before evaluating new Replica results. Finish all executable prescribed Replica rows even when CAL is negative, because this is a module study, not selective reporting. A later retrospective Replica best row is explicitly exploratory and cannot replace the frozen nomination. Deployment remains `N0_UNCHANGED` in this task.

## 6. E01 — robust final aggregation on frozen query evidence

Recover actual `Q_GAIN` final retained per-view features from original query receipts and decisions, following `recover_aggregates()`. Require retained request order and membership parity, including irreversible alias/split/eviction handling. The original area aggregate must reproduce frozen Q scores. Do not initialize or rerun Q, change a request, read an evicted vector, or feed new scores back into an earlier query state.

For owner i let `f_j` be each retained raw six-crop mean, `a_j` its original visibility area, and `v_j=f_j/||f_j||`. Use float64 readout arithmetic.

- Historical control: `unit(sum_j a_j f_j / sum_j a_j)`.
- RAW_EQ: `unit(mean_j f_j)`.
- UNIT_EQ: `unit(mean_j v_j)`.
- GMED: compute a smoothed Euclidean geometric median of the `v_j`, then normalize once.

The required GMED algorithm is deterministic:

```text
x = mean(v_j)
eps = 1e-4
repeat at most 64 times:
    d_j = sqrt(||x - v_j||^2 + eps^2)
    x_new = sum(v_j / d_j) / sum(1 / d_j)
    stop after accepting x_new when ||x_new-x|| <= 1e-8
    x = x_new
result = unit(x)
```

No projection to the unit sphere inside the iterations; that would be a different optimizer. No learned outlier threshold or hard view deletion. For n=1 use the original direction. For n=2 use the unit-vector midpoint (do not claim it can detect the bad observation). If a final direction has norm≤1e-12, use the original area aggregate and record a numerical fallback. Nonfinite or missing input is not a numerical fallback—it is a failed input dependency. Record convergence, n, original norm spread, final normalized influence weights and effective sample size as diagnostics, not correctness probabilities.

RAW_EQ versus original isolates area weighting. UNIT_EQ versus RAW_EQ exposes per-view norm changes. GMED versus UNIT_EQ isolates the robust direction operation, subject to their identical calibration recipe. This normalization control is required; otherwise normalizing views could be mistaken for robustness.

**New visual inference: zero.** A Q readout change cannot fix incorrect instance shape, and a same-score result is not proof of robust rejection unless an actual influence change occurred.

## 7. E02 — recognition-mask changes only

### 7.1 Shared mask override interface

First run the original `_load_request()` checks on the captured request. Create a separate `semantic_mask` and derivative manifest. Keep `target_mask_sha256`, original union and 3D owner identity untouched.

Implement `recognition_crops(rgb, original_bbox, semantic_mask)` using exactly the old crop geometries and raw RGB crops, but blacken pixels outside the supplied recognition mask. It may shrink or extend the old global target; it is not required to contain it. Do not weaken `native_crops()`'s original union assertion globally.

Validate a baseline union call reproduces the old six crop pixels exactly. For all modes, each raw crop must be nonempty and each foreground crop must retain at least one semantic-mask pixel. An empty foreground after the prescribed transforms makes that request unavailable with a reason; do not replace it with a whole-image embedding or secretly add a point prompt.

### 7.2 GLOBAL

Set `semantic_mask = captured_global_target`. No new segmentation. Same S2-L, same three views, same bbox and weights. Do not alter RGB for the raw-background branch.

### 7.3 SAM2

Use the original SAM2.1 Hiera large checkpoint and pinned image predictor. Cache image encodings per actual frame. Call once per original region with **only** its unchanged float XYXY box, no points, previous mask, text, expansion or iterative correction. Set `multimask_output=True`, `return_logits=False`, `normalize_coords=True` according to the official predictor interface.

Select the returned full-resolution mask with the highest finite predicted quality; tie by original returned index. Preserve all quality values and selected index. This is a fixed prediction-side choice, not a GT or source-label choice. Use the returned binary mask at original resolution; do not union it with the old target, clip it to an original instance, select a connected component by GT, or re-estimate bbox.

Use bfloat16 autocast as in the official image example. Save actual model/precision settings. A global compatibility repair to FP32 may be made during the real smoke **before** any batch results are evaluated, with the whole leaf subsequently using that fixed mode. No per-sample precision or prompt search.

Report original-vs-repaired area, IoU, foreground crop occupancy, mask rejection/technical failures, and actual source changes. These masks are only semantic input masks. No SAM mask is ever projected into the final owner map in this wave.

## 8. C0 — frozen capacity control

Replace only the static S2 slot by `google/siglip2-so400m-patch14-384` at the pinned HF revision in JSON. Use all six **original union** crops and the exact same manifest. No SAM mask, robust view pooling or new text description is used here.

Use FP32, the model's official tokenizer/processor, the same raw class strings and ordered semantic IDs as S2-L, and the same raw-six-mean/area readout. Dynamically read the embedding dimension; do not reuse the old text matrix or force a shape. Verify preprocessing receives HWC RGB explicitly, as the fixed `rgb_siglip.py` does. Do not request normalized sigmoid classifier outputs from a convenience pipeline; save raw image/text features and compute the defined cosine evidence.

The larger encoder is a capacity comparison, not an original 2026 algorithm contribution. Report model load memory and physical inference separately from the other methods.

## 9. E03 — region-tuned recognition with a matched control

### 9.1 What this experiment is and is not

Test **only** OVR's region encoding and region/text alignment. It is an adaptation of the author's region branch to externally supplied OVI masks, not a full OVRCOAT panoptic reproduction. Do not run its objectness correction, learned mask proposals, void probability ensemble, seen/unseen category weighting, or segmentation-head inference.

The matching control uses original frozen OpenCLIP ConvNeXt-L weights with the exact same region computation. It need not run or download a trained FC-CLIP segmentation head: that head is not used. Label it `FC_FROZEN_REGION`, not a reproduced full FC-CLIP benchmark.

### 9.2 Checkpoint and architecture binding

Pinned config: `configs/coco/panoptic-segmentation/fcclip/ovrcoat_convnext_large_eval_ade20k.yaml`, with its listed bases. Backbone: `convnext_large_d_320`, original tag `laion2b_s29b_b131k_ft_soup`.

Load the official OVR checkpoint's **learned `backbone` region weights**, not its `frozen_backbone`. A task-local loader may extract `backbone.clip_model.*` into the matching OpenCLIP module; record exact checkpoint-key-to-module-key mapping and all excluded keys. Every weight used by image trunk, pooling projection and text encoder must be either loaded from that branch or proven to be the corresponding unchanged pretrained weight. Missing active weights, random initialization, an all-frozen mistakenly selected branch or dimension mismatch are blocking defects, not acceptable partial loads.

For the control, use the original pretrained weights. If the checkpoint includes `frozen_backbone`, it may be used only after establishing that it matches the original declared pretrained model; do not assume naming alone proves that. Record learned/control active-parameter differences. If the two active branches are identical, stop the claimed tuning comparison and identify the load problem.

A small OpenCLIP-only region adapter can implement the verified chain directly:

```text
normalised full RGB -> ConvNeXt stem -> stages[0..3] -> norm_pre
    -> signed-mask pooling at dense feature resolution
    -> visual.trunk.head -> visual.head -> unit region vector
```

This avoids importing an unused CUDA deformable-attention segmentor. Copy or adapt only the required licensed subroutines with source attribution. When an original-wrapper execution is feasible in the existing environment, compare one actual region's feature/classification with it; otherwise validate each defined operator with the pinned code and disclose that no complete upstream-model runtime parity was executed. A random/synthetic weight smoke is not an OVR loading check.

### 9.3 Image, mask and text definition

Use the same original static target/view manifest and **captured global target mask** for both OVR and control. Use full-frame dense features, not the S2 six-crop sequence. Different systems may have different native preprocessing; this is why the FC/OVR matched pair is essential. OVR versus S2 alone cannot attribute a difference solely to region tuning.

For both members, set RGB, resize shortest edge to 800 subject to longest edge≤1333 (round output dimensions with `int(x+0.5)`), normalize 0–255 pixels by the mean/std in JSON, and pad the normalized image with zeros to a multiple of 32. This is the **explicit external-region adapter convention**, not a claim that every author's evaluation path has identical resize defaults. Do not tune this resolution after seeing scores.

Resize the original boolean global mask with nearest-neighbor to the same unpadded dimensions. Convert to **+1 inside / −1 outside**, pad with −1, bilinearly resize with `align_corners=False` to the dense map, then use the author's `>0` mask pooling. This avoids spreading a positive 0/1 interpolation halo. Record the resulting dense support. A mask with zero support is unavailable; do not fill it with background tokens.

Pool the dense feature with the binary mask divided by its support, use the matching visual projection, and unit-normalize the result. The same signed mask and operations must be used in both controls. Record masks at original, resized and feature-grid resolutions for input diagnostics.

For text, use the **exact 14 `VILD_PROMPT` templates in the pinned OVR code** once per raw benchmark class name; no extra comma-separated synonyms or manual aliases. Encode in each loaded branch's own text tower, unit-normalize template vectors, average within each class, and unit-normalize the class prototype. Preserve benchmark ID order, with no foreground label from training metadata used at prediction time.

Export cosine `unit(region) @ unit(class_prototype).T` **before** the learned logit scale, class softmax or void term. The official `get_classification_logits()` multiplies by a common positive scale; recover/verify its pre-scale cosine on a sample, not by taking a log of returned probabilities. Do not call `out_of_vocab_classification()` and softmax its returned probabilities again.

For each owner use the original area-weighted mean of unit per-view region vectors, then unit-normalize. Fit each branch's changed-slot temperature using Section 5. Text templates, image geometry and pooling are identical within the FC/OVR pair. No training is run in this task.

## 10. E04 — localized concept evidence, not another free vote

### 10.1 Frozen target and class selection

Use only the unchanged A7_REFIT predictions/probabilities and genuine original N0/Q/S2 evidence to form the E04 target list. A target must belong to the original static subset, have at least two genuine available source labels and at least two distinct positive labels, and have an original selected view. Select at most 64 such owners per scene by SHA256 of `scene_id + '|AW_E04|' + str(owner_id)`, tie by owner ID. Do not rank by GT, previous corrected/harmed lists, or new model success.

For each target use its original up-to-three static views. Candidate class IDs are the union of genuine positive N0/Q/S2 top-1 IDs and A7_REFIT's positive winner: at most four. Sort by frozen vocabulary order. Include A7's winner even when it is not a source top-1. Freeze target/view/candidate manifests before any SAM3 calls. Never use E01–E03 outputs to regenerate this first-wave shortlist.

### 10.2 SAM3 execution

Use the pinned **SAM3 image model**, not the newer video/SAM3.1 release. Resolve original weights through authorized official access. The model may be gate-limited; that blocks E04 only.

Pass `PIL.Image` RGB to `Sam3Processor`, resolution1008 and `confidence_threshold=0.0`; binary mask threshold remains its0.5. This threshold choice deliberately avoids the demo's extra0.5 detection prefilter; it is not a reliability claim. Record the presence/object score product returned by the author.

For each frame, encode the image once. For each distinct candidate raw class string, reset all text/geometric prompt state while retaining the image backbone, then call only `set_text_prompt`. No box prompt, label description, prompt ensemble or scene-specific spelling is used. Cache a frame/class result and share it across target regions requesting that identical operation. Copy outputs before another prompt mutates the state.

The processor's `masks_logits` field is actually post-sigmoid at this pin. Treat it as probabilities; use its binary `masks` or threshold once. Never sigmoid it again or assume its shape is the original model-logit shape. SAM3 receives no test GT, source confidence or object name beyond each explicit candidate string.

### 10.3 Spatial evidence and complete-map controls

For candidate c, view v and returned mask k compute

`support(i,v,c) = max_k[ SAM_score(v,c,k) × IoU(SAM_mask(v,c,k), original_global_target(i,v)) ]`.

A valid model call with no detections or no intersection gives0 support, not a technical failure and not a proof of real-world absence. A technical failure for any candidate at view v excludes that view for **all** candidate classes of this owner. This prevents missing output from one candidate becoming artificial negative evidence. All remaining views use the same original-area denominator:

`e_i(c) = sum_v a_v support(i,v,c) / sum_v a_v`.

If no common successful view remains, or `sum_c e_i(c)≤1e-12`, keep A7_REFIT and record `NO_LOCALIZED_EVIDENCE`; do not claim a successful verification. Otherwise set `q_i(c)=e_i(c)/sum_c e_i(c)` on the fixed candidate set. Obtain `p_i(c)` by restricting and renormalizing A7_REFIT's **actual** mixture probabilities to the same set.

Produce three complete-map outputs, keeping A7_REFIT for all unselected owners:

1. **SHORTLIST:** argmax of restricted p. It must reproduce the base decision since the base winner is included. This is an intentionally identical coverage control, not another positive result.
2. **SPATIAL:** argmax q. With an exact tie, prefer the A7 winner if tied, otherwise the first tied class in frozen vocabulary order.
3. **MIX50:** argmax `(p+q)/2`, using the same tie rule. No fitted coefficient or confidence gate.

This evaluates localization evidence directly before proposing a complex acceptance network. Its scores are not calibrated class probabilities. Report shortlist recall on the identifiable diagnostic subset, but never use that recall or GT to revise the shortlist. A right answer outside the candidate set is an explicit mechanism limitation.

Maximum target-view-class pairs: `10×64×3×4=7,680`; actual shared frame/class executions will be lower. Record both. Process a frame at a time; do not retain all image backbones or full-resolution masks in GPU memory. Save sufficient selected-mask/overlap evidence (bit-packed masks or equivalent small reproducible records), not a huge uncompressed tensor dump.

## 11. One bounded interaction experiment

A richer source can help a combination even when its standalone row is not best. Do **not** require every single module to pass an all-scene no-loss rule before allowing the one composition.

Using CAL only, select the best measured E01 `_A7` member and the best measured replacement-region `_A7` member among E02/C0/E03 under Section 5. Each must have some genuine finite source-score change relative to its old slot; a pure technical fallback or numerically identical source is not a module to compose. If both exist, create `AW_COMBO_QR`: change only Q readout and static region source simultaneously; preserve original cosine N0 and each selected source's own frozen fold/final temperature. No new weights or scalar refit for the combination. This costs no new image inference beyond the chosen leaves.

Evaluate `base / Q-only / region-only / Q+region` on both CAL scenes and the eight Replica scenes. Compute interaction deltas in the same metric/aggregation, and preserve adverse interactions. This adds one method at most (20 scene-rank rows and2 Replica pools). If a family is inaccessible, select from the actually measured eligible members; if no eligible pair exists, record exactly why no combination was run.

Do not compose E04, new masks into geometry, or every pair of methods. Candidate selection is frozen before Replica evaluation and not changed to match whichever Replica row looks best.

## 12. Evaluation and scientific acceptance

Reuse the released evaluator, original projected masks, semantic GT, class definitions and min-size rules. `SceneEvaluator` and `pool()` already provide the two ranking views. A new study-local wrapper should supply new output paths and method identities without bypassing old method/access guards or editing old study artifacts.

- **Primary:** official current-predicted-class area ranking plus one released dataset-pooled call over all eight Replica scenes in official order.
- **Secondary:** frozen N0 numerical ranks; per-scene results and scene macro averages.
- CAL: report both out-of-fold per-scene and pooled diagnostics, clearly marked development.

Never use a semantic model's probability as the AP ranking score in this wave. Fusion or direct readout changes class labels only. Official export may subsequently change ranks by the author's rule; keep that evaluation-only view separate from immutable source ranks.

Report APall/uAP alias, AP50, AP25, mIoU and mAcc. Get actual overlap thresholds from the pinned evaluator (the current release's APall covers0.50–0.90, not an assumed COCO0.95). AP/IoU must come from the same complete-map prediction. Never combine the best columns from different variants.

Reuse actual released matching traces for added/lost/duplicate/ignored predictions. Use the separate strict unique geometric correspondence only for semantic diagnostics; no filtering of map evaluation based on that correspondence. Retain source cap/failure/availability strata and matched common-available comparisons.

For each family distinguish:

- `MEASURED_NET_GAIN`: official pooled APall and mIoU both nonnegative relative to A7_REFIT, at least one positive, with AP50 and worst-scene changes disclosed.
- `MEASURED_TRADEOFF`: mixed target-metric changes; still a completed scientific experiment.
- `MEASURED_NO_GAIN`: no favorable target effect under the declared comparison.
- `NO_EFFECTIVE_INTERVENTION`: executed correctly but changed evidence/decisions insufficiently; explain the stage.
- `BLOCKED_ASSET_OR_INTERFACE`: actual required bytes/runtime unavailable; not a method-negative result.

These descriptive statuses are not p-values, automatic deployment authorization or proof of universality. Compare against N0 and A7_FIXED as well. A source may be poorer directly yet useful inside A7; show both rows. Eight historically exposed Replica scenes cannot establish new blind generalization.

## 13. Task-to-code binding and runnable phases

Create a small task-local package `src/static_ovmap/a7_evidence_upgrade/` and `scripts/evaluation/run_ovimap_a7_evidence_upgrade.py`. Do not build a universal experiment platform or copy entire historical pipelines.

| Responsibility | Existing interface | Required addition |
|---|---|---|
| Bind base and requests | `m2_reviewer_study.scores.read_scene`; original scene configs and semantic manifests | New scope/model lock, unchanged input identities. |
| Recover Q views | `aggregates.recover_aggregates`; query receipts/retention | Export actual per-view vectors; robust readout without replay. |
| Recognition masks | `semantic_models._load_request`; `region_evidence.native_crops` | Independent semantic-mask crop helper; original target validation remains. |
| New visual models | `rgb_siglip.FrozenSiglipBackend`; worker pattern | SO400M, minimal ConvNeXt region adapter, isolated SAM workers. |
| Semantic source output | `semantic_study.direct_readouts` conventions | Generic task-local source records with real model/slot identity. |
| Scalars | `composition_study.temperature.fit_temperature` | Per-leaf changed-source examples/fold/final records only. |
| Fusion and map output | `m2_reviewer_study.fusion.fuse`; `relabel_prediction`; `save_prediction` | New method registry and E04 localized decision rules. |
| Evaluation | `m2_reviewer_study.evaluation.SceneEvaluator,pool`; released traces | New-root evaluation wrappers and both aggregation modes. |
| Reporting | Existing JSON/NPZ/CSV writers | Actual per-family outputs, selection, four evidence tables, publication. |

Implement real phases: `bind`, `e01`, `e02`, `c0`, `e03`, `e04`, `compose`, `evaluate`, `report`, `publish`, `all`, with `--resume`. Provide `--split cal|replica|all` and `--scene` only as controlled leaf/resume options. A top-level `all` must schedule dependencies itself, not print a plan, read smoke receipts as if metrics existed, or finish with an empty measured dictionary.

`all` order:

1. Bind common input and model prerequisites; snapshot allowed revisions and freeze all method definitions.
2. Complete CAL family leaves in E01→E02→C0→E03→E04 order; independently continue after an external leaf is truly blocked.
3. Fit CAL scalars, produce genuine CAL predictions/metrics, choose the one compatible pair and evaluate its CAL combination when eligible; then freeze both the pair identity and final research nomination.
4. Run every executable prescribed Replica family using those choices and final scalars, one family at a time. Do not CAL-prune unfavorable defined methods.
5. Run the frozen allowed composition on Replica, then finish official pools, object/cost reports, and the final contribution decision.
6. Commit and push the actual result; verify remote SHA. Exit with a complete status summary, not a progress-refresh loop.

Example command **after implementing the runner**:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_a7_evidence_upgrade.py \
  --spec docs/paper/static_ovmap/a7_evidence_upgrade_wave1/PROTOCOL_SPEC.json \
  --reviewer-binding /mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001/source_binding.json \
  --output-root /mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001 \
  --phase all --resume
```

The coordinator Python above is an existing locator from receipts, not authority to install every model in that environment; resolve actual differences. Model workers use their locked environments.

For mid-run failure, resume missing descendants and repair only the affected artifacts. Record repaired interface/input changes and recompute every affected condition before comparing them. A report status cannot be hard-coded COMPLETE. `implementation`, `assets`, `measured_rows`, `scientific_result`, `selection`, and `publication` are separate fields.

## 14. Budgets, costs and limited testing

### 14.1 Resource boundaries

Reuse all original mappings/captures and original Q trajectories. E01 has0 image forwards. No new CropFormer inference or native TSDF capture is allowed.

For each static visual variant there are at most `10×128×3=3,840` target-view requests. An S2-style request logically contains6 crops. E02 may physically reuse3 original crops and compute only3 foreground crops. The FC/OVR region models use dense full-frame encodings shared by multiple requested masks; count their image encoding, region pooling and text encodings separately, not as6-crop calls. At most one dense frame encoding per distinct selected frame/model; no recomputing the frame for every target.

SAM2 shares image features per frame and counts each box decode; SAM3 shares image features and identical frame/class prompt outputs. Record complete logical dependencies as well as this run's physical cache reuse. Excluding a source from fusion does not remove the cost of forming the original map and active registry.

Use existing FP32 native/S2/capacity/OVR visual modes; record peak GPU memory on required worker runs. SAM bfloat16 settings are fixed as described. No model quantization, gradient fine-tuning or inference-resolution search. On OOM reduce a batch deterministically down to1; this may change throughput but not input geometry. If a single input still fails, record the technical failure. Do not silently lower resolution. Count failed-attempt time and model-load time when observable.

Keep full images, models, dense maps and large arrays on shared storage. Compress masks and source logs. The new tracked publication target is≤40MiB; no old release duplication. Principal metrics and selection JSON are preserved byte-for-byte.

### 14.2 Only checks directly needed for validity

No test-count target, no full-repository safety regression, no repeated GPU stress test. Implement a compact set of checks for:

1. Baseline cosine-A7 reconstruction, immutable geometry/owner/frozen ranks and class-ID alignment.
2. Actual availability/fallback exclusion and complete-map inclusion of excluded targets.
3. Exact original union crop parity; new semantic-mask shrinkage without falsifying original identity.
4. E01 raw/unit/median controls, n=1/n=2/zero direction, deterministic convergence and unchanged retention.
5. Model/mask/text/precision cache separation; SO400M non-hardcoded feature dimension.
6. OVR checkpoint branch, signed mask threshold/padding, text prototypes and pre-scale score export.
7. SAM box/PIL coordinate correctness, prompt reset, probability-vs-logit field domain and shared image cache.
8. E04 shortlist inclusion, common successful-view denominator, all-zero fallback and bounded deterministic target selection.
9. Opposite-scene scalar fits, CAL-only selection, official rank/pool outputs and state/report dependencies.

Run one real input smoke per newly loaded adapter, using deterministic CAL requests without inspecting GT first. For OVR use two spatially distinct masks on the same frame to show that changing the mask changes the pooled region representation when content differs; this is an interface check, not a semantic accuracy test. A smoke's successful generation/feature shape does not replace full-scene evaluation.

Run lint/compile on changed paths and the compact relevant tests once after repairs. Validate the final result matrix and arithmetic once. Do not rerun old34/8/6-condition studies,560 reviewer rows, or inspect16,430 old artifact files. Do not generate a large test suite simply to report a high pass count.

## 15. Required outputs and interpretation

For each family, immediately persist its real CAL/Replica leaf results as they complete. Produce one final set of four main tables:

**A. Performance and comparisons:** five controls, all new direct and A7 rows, E04 three rows, optional one composition; official pooled main results, fixed-rank and per-scene supplementary results; no missing-as-zero metrics. Add the FC→OVR matched contrast, E01 normalization/robustness contrasts, and base/Q/region/Q+region interaction.

**B. Effective intervention:** all owners; capped targets; requested/usable views; nonempty final region support; genuine source output; changed source suggestions; changed final labels; failed/fallback reasons. An identical SHORTLIST output is expected, not a discovery.

**C. Corrections and damage:** original versus final labels, actual released added/lost matches and duplicates/ignore events, common-available correctness, correct-class source rank (evaluation-only), unused correct evidence; show failures as well as successes. Mask-only E02 claims remain about recognition, never reconstructed 3D shape.

**D. Costs and decisions:** per-model images/crops/prompt decodes, source-required operation unions, physical reuse, model downloads/load time, measured inference/fitting/evaluation time, peak memory, incomplete timing; CAL nomination and reasons; deferred next candidate based on actual error type.

Use deterministic qualitative examples: up to two corrected and two harmed owners per scene in owner-ID order. Existing mask projections may be rendered; no extra segmentation for prettier figures. Publish data for every prescribed condition, not only the final nominated one.

Interpretation rules:

- E01 GMED below UNIT_EQ does not validate robust aggregation merely because GMED beats some other weak baseline.
- E02 SAM2 not beating GLOBAL does not establish a benefit from its additional segmentation cost.
- OVR beating S2 but not FC_FROZEN does not establish an advantage from region tuning.
- OVR checkpoint inaccessible means the tuned comparison is untested, not that OVR failed scientifically.
- A larger encoder winning is a useful capacity result, not proof of a new algorithm.
- A replacement source weaker directly but helpful in fusion remains a valid finding; quantify its harm and extra cost.
- SAM3 can confirm a wrong concept at the wrong region; use localized overlap, but do not call that ground truth.
- No net gain permits a complete negative report. Never repair labels by object ID, scene ID or known GT class.

Write a next-action recommendation choosing **one** unresolved capability from E05–E12 if first-wave data supports it. Do not execute that next-wave research in this task. Do not repeatedly search for fresh scenes that the parent study has already shown unavailable; new confirmation needs a later authorized data task.

## 16. GitHub publication is a completion requirement

Track actual implementation, new spec, minimal upstream adapter patches with license attribution, real resolved configs without secrets, scalar fits, metrics, compact object/cost ledgers and:

- `docs/paper/static_ovmap/A7_WAVE1_RESULTS.md`
- `docs/paper/static_ovmap/A7_WAVE1_HANDOFF.md`
- `docs/paper/static_ovmap/A7_WAVE1_SELECTION.md`

Place new small results under `artifacts/static_ovmap/a7_evidence_upgrade_wave1/<run_id>/`. External models, full RGB-D and large arrays stay in shared storage with actual path/size/SHA256/rebuild references. A path manifest is not an upload of the external bytes.

The handoff must state exact source/checkpoint/environment identities, measured and blocked families, restore/resume commands, comparison definitions, known training-data overlap, selection provenance, cost omissions and the fact that the source map was not changed. Make measured evidence—not a fixed status literal—drive the report.

Commit and push normally to `research/ovimap-a7-evidence-upgrade-wave1`. Resolve origin from the authenticated repository. Do not force-push, alter the default branch, or publish unrelated files. After the final push compare complete `git rev-parse HEAD` with `git ls-remote origin refs/heads/research/ovimap-a7-evidence-upgrade-wave1`. Only exact equality is `PUSH_VERIFIED`.

Store the final publication receipt outside the commit to avoid a recursive self-SHA. If authentication genuinely fails, keep local commits and return the actual error and exact push command; do not fabricate remote verification or create a replacement remote.

## 17. Final audit before returning to the user

Check once that:

- Real wave-1 predictions and evaluator records exist; external blocks are separated from completed leaves.
- All executed methods use the same native geometry, owner and projection; no mask refinement leaked into mapping.
- E01 uses actual fixed retention; E02 retains original bbox and view choices; E03 changes the correct loaded branch; E04 uses frozen prediction-only shortlists.
- All new sources have own score/text/model identities and never count fallback as evidence.
- CAL fitting and nomination preceded new Replica selection; all prescribed executable results remain reported.
- Official pool/current-class results and auxiliary macro/frozen results are unambiguously labeled.
- The optional composition is only one preselected compatible pair; no hidden model or temperature search was added.
- Code, reports and real small results were pushed, and the full remote SHA was checked.

Return a Chinese summary with one table per family, main A7/N0 deltas, effective intervention, actual costs, selected research candidate, remaining asset restrictions and report links. State clearly that this completes **wave1**, not all24 papers or all future geometry research. Do not conclude with only a plan, a smoke count, or an unmeasured COMPLETE banner.
