# 02 — S: frozen region evidence and paired semantic correction

## S0. What this branch tests

Two separate hypotheses must be measured: an alternative frozen region recognizer contains useful corrections; a learned KEEP/REPLACE decision preserves those corrections while avoiding damage. Neither a teacher swap nor a gate that rejects everything establishes the proposed mechanism.

Use native domain N. The incumbent is the original native OVI semantic output for that scene, including its original observation aggregation. Preserve the complete surface coordinates, owner vector, active registry and serialized numeric ranks for every S condition. Background owner 0 stays 0. Changing class may change instance-class eligibility; let the released evaluator handle that normally.

The historical H64 study may also be replayed as a retrospective diagnostic if its assets exist, but cannot select teachers, thresholds or checkpoints. Do not write the historical 20.644% number into a native-scene expectation.

## S1. Target/view manifest and input controls

All incumbent instances remain in every final map, including those the teacher cannot process. Process at most 128 targets per scene: when there are more, sort by `sha256(scene + '|S|' + stable_target_id)` and take the first 128, without labels, scores or GT-based filtering. Record excluded targets and retain their incumbent labels. This cap controls computation, not the evaluator's target universe. The historical 64 targets all fit the cap.

For each target, use at most three of its **actual native captured query regions**, ordered by visible overlap area descending, frame ID ascending, request ID ascending. Deduplicate exact request identities. Keep native global bbox and `local_mask OR global_mask`; do not substitute the sparse positive-PNG point projections from `ownership_evidence.py`. Retain the source map version and lineage. A capture-time region whose current owner cannot be reconciled unambiguously with the final target is unavailable, not silently assigned by array position. Define usable RGB, bbox and region-mask validity without GT; do not use recognition outcomes to choose views.

Prepare the same ordered requests for all models. Save original image/mask, actual crop geometry, foreground fraction, depth-valid fraction and model-preprocessed mask support. This is a same-target/same-request study, not a claim that model internals and all preprocessing are identical. Invalid requests have explicit reasons. No new masks, dilation, prompt search or per-example alternate input mode is permitted.

The native incumbent N0 uses its original full native observations. Matched three-view controls below distinguish the input/aggregation change from the alternative recognizer. A teacher beating a degraded three-view control is not automatically better than N0.

## S2. Frozen model adapters, and fixed readouts

Implement these five direct readouts; N0 is an additional measured/reused reference:

| ID | Visual encoding | Readout |
|---|---|---|
| S_NATIVE_AREA | native SigLIP, original six crops | visible-area-weighted feature mean, final L2 normalization, native text cosine argmax |
| S_NATIVE_VOTE | reuse exactly the same native features | per-view cosine argmax; plurality vote |
| S_SIGLIP2_AREA | frozen SigLIP2-L, same six crop images, its own processor/text space | the same area fusion rule |
| S_SIGLIP2_VOTE | reuse SigLIP2 features | the same vote rule |
| S_WOW_VOTE | frozen WOW, actual region-conditioned inputs | generated name -> fixed vocabulary mapping -> the same vote rule |

For any vote: largest vote count wins; a tie is broken by the highest-ranked successful request supporting a tied class; a remaining exact tie follows original `valid_ids` order. No reliable-view threshold, incumbent-label tie preference, or post-hoc view replacement. One successful view is one vote, not three copied votes. If no valid result exists, retain the incumbent and mark `TECHNICAL_FALLBACK`, not teacher success.

Native encoding must preserve `native_crops()` exactly, including existing exclusive upper-bound slicing. Add a new encoder that can return the six individual unit crop vectors and their arithmetic mean; do not change the old encoder's return contract. Check that its first-six mean matches the old calculation on a real request. Bind text templates, class order and canonical readout implementation from the real native text-cache receipt. Do not infer that different feature spaces share a 1024-D representation. The SigLIP2 adapter uses its matching image/text model; the vocabulary and prompt templates are the same strings, but the embeddings are recomputed in its own space.

### WOW binding (no silent plain-image fallback)

Pin author code to `AAwcAA/WOW-Seg-Meta@bfc6f2424c47097e70a641c6a8016319cac192cb`; bind the actual `AAwcAA/WOW-Seg` weight snapshot separately. Use force image size 448, at most 12 dynamic full-image patches, thumbnail enabled, region crop scale 2.5, greedy decoding, `do_sample=False`, `num_beams=1`, `max_new_tokens=32`.

Use exactly:

```text
This is the original image: <image>. Please classify the specified target area. Reply with only the category name of the target area, without explanation. Specified target area:<image>
```

Inspect the loaded checkpoint's actual `chat` and visual-token code. The adapter must pass the nonempty 16x16 region mask into the actual mask-conditioned computation. Merely finding `pixel_masks` in a signature or obtaining fluent output is insufficient. Record one hook/shape trace at that computation boundary on a real request. Use `official_combined`; if the actual checkpoint requires `num_patches_list`, lock `combined_with_patch_list` during the same smoke. These are allowed only when both preserve the same full-image/region inputs and fixed two-image prompt. Do not call the author's per-example `_chat_with_fallbacks()` and do not use `full_image_mask_only` as a fallback. A missing region interface is `BLOCKED_WOW_MASK_INTERFACE`.

If the region becomes empty in the official 16x16 representation, do not generate an ordinary-image answer and count it as region recognition. Mark that request unavailable. Record the original and final support sizes for every request. With fewer than half the targets having any valid teacher request, still report the bounded completed direct study, but classify teacher evidence as `INPUT_COVERAGE_INCONCLUSIVE`; do not fit a WOW-specific selector or claim WOW's general ability has been refuted. No automatic input repair is authorized in v1.

Preserve complete raw generation. For mapping, apply the author's `clean_category_response()`; compare lowercased, whitespace-collapsed, hyphen/underscore-to-space strings against the complete official dataset class-name list. Exact unique match maps directly. Otherwise embed the cleaned answer and unmodified class names using frozen `sentence-transformers/all-MiniLM-L6-v2`, L2 normalize and take highest cosine, ties in original class order. Save all similarities and the first/second gap. No hand-added synonyms, class menu in the prompt, incumbent/SF class in the prompt, or GT-assisted correction. Empty/technical failure is unavailable. Nonempty uncertain/multi-object prose is mapped by this same rule but separately flagged for interpretation; mapping cosine is not correctness confidence.

### Additional object/context evidence

For the native encoder only, compute three background-only crops at the same scales, using original RGB with target-mask pixels blacked out. The original six crop vectors remain untouched. Save the nine vectors with explicit `raw`, `foreground`, `background` and scale identities. These extra three crops are for selector features, not a change to S_NATIVE_AREA/VOTE. If background support is empty, mark those features unavailable. No generative inpainting.

Upper bounds per scene: 128 targets x3 requests =384 target-view requests per available visual model; native six-crop control 2304 crop inputs plus at most1152 background inputs; SigLIP2 at most2304 crop inputs; WOW at most384 generations. Count WOW tiles/region tiles/tokens separately. Do not encode duplicate requests twice. FIT/CAL/SELECT run before confirmation; the two raw alternative teachers are not both rerun on CONFIRM unless selected by the locked rule in file05.

## S3. Alternative teacher selection — explicit and finite

Alternative candidates are `S_SIGLIP2_AREA`, `S_SIGLIP2_VOTE`, `S_WOW_VOTE`; native direct controls are not alternative teachers. On CAL only, rank available candidates by mean scene uAP descending, mean scene mIoU descending, median measured model cost ascending, then the ID order printed above. Freeze one alternative and its model/processor/prompt/mapping settings in `teacher_selection.json` before SELECT. Do not choose per scene, per class or per object.

If no alternative is available, the S learned branch is blocked but all adapters and direct native controls are delivered. If the best alternative is worse overall than N0, it may still contain useful corrections: proceed with learning only if FIT contains the event support specified below. Do not require direct global net gain to permit a selective-correction test.

## S4. A small, completely specified paired selector

Prediction actions: KEEP incumbent label, or REPLACE by the fixed selected-teacher suggestion. The selector must not change that suggestion or search another donor. Same-label suggestions are recorded and bypassed as unchanged. Technical fallback cannot be a replacement event.

### FIT labels, separate from prediction

Use the existing independent geometric correspondence helper to find a unique GT object with strict IoU>0.5 for an incumbent mask. Ignore class while making this geometric correspondence. Ambiguous or unmatched masks are excluded only from supervised head targets, not from full-scene predictions or released evaluation. Define three labels for differing, technically available suggestions:

- GAIN: incumbent wrong, suggestion correct;
- HARM: incumbent correct, suggestion wrong;
- OTHER: neither of the above.

Require at least five GAIN and five HARM objects across FIT and representation of each event in at least two physical FIT scenes. If absent, mark learned S `BLOCKED_EVENT_SUPPORT` and report direct methods. Never satisfy this count by copying views of one object. Each object contributes one aggregate training example, weighted inversely by the number of usable FIT examples in its scene. The label represents a geometric object's class correctness, not a per-object decomposition of AP.

### Features (32 values +32 availability bits)

No IDs, class index vectors, fixed class-slot logits, dataset indicator, GT IoU, historical error codes or scene names enter the predictor. All cosine/statistics are taken **within** their own model space. Use the following ordered values:

0–7: native aggregate cosine for KEEP; for REPLACE; their difference; native top1-top2 gap; entropy of softmax(native cosine/0.07)/log(C); native-view KEEP vote fraction; native-view REPLACE vote fraction; mean(1-cosine) over distinct native view-feature pairs.

8–15: native valid-view fraction; teacher valid-view fraction; teacher REPLACE vote fraction; teacher view-label entropy/log(C); teacher aggregate top1-top2 strength gap; teacher KEEP strength; teacher REPLACE strength; mean name-mapping top1-top2 gap. For SigLIP2, strength is its own cosine; for WOW it is the mean mapped-name cosine over valid answers. Name-mapping gap is unavailable for SigLIP2, not a fabricated calibrated value.

16–23: log1p(source mask point count); mean log1p(request mask pixels); mean target-mask/bbox-pixel ratio; minimum valid-depth fraction inside target; mean local/global mask IoU; mean fraction of target requests with surviving region representation (WOW16x16, otherwise original representation); teacher technical-failure fraction; fraction of requested regions unavailable before inference.

24–31: mean(raw KEEP cosine - foreground KEEP cosine); corresponding REPLACE difference; mean background(REPLACE-KEEP) margin; mean foreground(REPLACE-KEEP) margin; within-view standard deviation across the three foreground-scale KEEP cosines averaged across views; corresponding REPLACE deviation; standard deviation across views of foreground(REPLACE-KEEP) margin; fraction of requests with usable background evidence.

If KEEP is class0 or otherwise lacks a text entry, its text-dependent features are unavailable; never index the last text row with a missing index. For item24 onward, compute comparisons using the same native text vectors; first average the three appropriately normalized crop score values per view unless a scale deviation is specified. A statistic with no defined inputs is value0/availability0. Otherwise availability1. For paired feature agreement with only one view, mark the pair statistic unavailable. Standardize each continuous value using FIT-only mean/std over available entries, std floor1e-6; fill missing standardized entries with0; append the32 availability bits. Clip standardized values to[-8,8] using the fixed rule, not test percentiles.

### Heads and ablations

All three heads have the SAME 64->32(ReLU)->3(softmax) architecture and training budget:

| ID | Available features |
|---|---|
| S_SIMPLE | only indices0,1,2,3,8,9,12,19, and their availability bits; all other columns zero |
| S_NO_CONTEXT | indices0–23; context indices24–31 and corresponding bits zero |
| S_PAIRED | all32 features and bits |

Train seed17, Adam lr0.001, weight_decay0.001, batch128, at most100epochs, weighted cross entropy. Evaluate CAL NLL every5epochs; save minimum-NLL checkpoint, tie earlier epoch. Stop after three evaluations without improvement. No class rebalancing that changes priors without correction; use the scene weights only. One fit per head, no architecture/seed grid. If CAL has no differing technically available supervised examples, select the final FIT epoch checkpoint, mark `UNCALIBRATED`, and do not promote the learned module; do not invent CAL labels.

Adoption value `v=P(GAIN)-P(HARM)`. For each head select threshold from `[0,0.10,0.20,0.30,0.50,+infinity]` on CAL full-map results, maximizing mean uAP, then mean mIoU, then fewer harmful geometric relabelings, then fewer total replacements, then larger threshold. Use `v>threshold`, not >=. Infinity means KEEP all and is an explicit no-intervention control. Save all six CAL rows; do not tune lambda, fit another score, or select on SELECT. Probabilities are head estimates, not guaranteed calibrated risks.

## S5. Outputs and branch conclusions

SELECT rows: N0, all available five direct readouts, and the three learned heads if supported. Show direct selected teacher versus S_SIMPLE/S_NO_CONTEXT/S_PAIRED on the identical suggestion set. A zero-replacement winner is `NO_INTERVENTION`, not proof of reliable correction. An improved S_PAIRED over N0 but not over the best frozen direct readout is not evidence the new mechanism is necessary.

Save per-object request/suggestion/accepted/final labels, raw responses, within-model scores, region validity, scalar features, learned decisions, correction/damage and actual evaluator match events in separated prediction/evaluation ledgers. GT-based errors are appended only by the evaluator after all predictions are frozen. Report names that look plausible but are mapped differently as mapping diagnostics, never manually fix a test prediction.

Implement S fully even when fitting is blocked. Source code, feature schemas and actual direct results are publishable with the correct status. Do not launch an unlisted new teacher when neither listed alternative works.

Record full and active parameter counts for every small head. Optimizer/scaler states and class-order-independent feature schemas are checkpoint provenance. A `KEEP_ALL` CAL winner must remain KEEP_ALL on SELECT, not be replaced with a smaller threshold to force changes.
