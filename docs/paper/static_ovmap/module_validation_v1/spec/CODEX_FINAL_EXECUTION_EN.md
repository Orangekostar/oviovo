# FINAL EXECUTION INSTRUCTION — OVI-MAP module validation

This file concatenates the six authoritative English specification sections without changing their content. Companion PROTOCOL_SPEC.json and SOURCES.md are included in the same bundle.

# OVI-MAP module validation — final Codex execution contract

Version: 2026-09-20-v1. This is an execution instruction, not another proposal.

## 1. Objective and authority

Work in `Orangekostar/oviovo`, starting from `b6455520c758a3413988e0827b3e1f34667bdfd1`. Implement and experimentally evaluate three isolated hypotheses:

1. **S — selective semantic correction:** preserve OVI geometry, obtain alternative region evidence, and learn whether replacing the incumbent label is beneficial.
2. **G — complete-partition selection:** preserve TSDF geometry, construct a bounded set of complete local instance partitions from native segment evidence, and compare learned versus old agreement-based selection.
3. **Q — pre-query utility:** decide whether to pay for a new native visual query from information available before that query, under a common causal budget.

The deliverable is working scoped code, actual experiments where prerequisites are present, a deterministic model/module selection decision, and a verified GitHub push. Do not stop after writing another plan. Do not claim that a method is useful because code, tests, or experiment rows completed.

The delivered `CODEX_FINAL_EXECUTION_EN.md` is the exact concatenation of this file and files `01`–`05`. Read that single file, or read this file and files `01`–`05` in order; they contain the same instruction. They jointly define one contract. `PROTOCOL_SPEC.json` encodes its constants; its `specification_only` marker means you must resolve local paths and versions into a runnable configuration, not feed unresolved metadata into the old runners. `CODE_REVIEW_AND_AUDIT_ZH.md` and `SOURCES.md` explain provenance; they do not add executable experiments. The earlier `THREE_INNOVATIONS_REVIEW_ZH.md` is background only. If an old task says zero training, zero new image inference, or no mapping replay, those restrictions are superseded **only for the scoped operations explicitly authorized here**. Preserve old source, configurations and result directories.

## 2. Authorization and limits

Authorized: frozen native SigLIP; frozen SigLIP2-L/16-384; one official WOW-Seg region recognizer and its fixed name mapper; bounded native OVI replay with instrumentation; CropFormer inference only to create missing front-end caches for the preselected legally available development/confirmation scenes; fitting the small predictors specified here; main experiments, listed ablations, selection and one frozen confirmation; commit and ordinary push to the task branch.

Not authorized: retraining segmentation/VLM backbones; launching new SF, SAM, Open3DIS, LEGO, OVRCOAT, GeoGuide, APPLE or NBV-Gym pipelines; changing the benchmark trajectory, evaluator, class subset or projection threshold; a broad parameter/prompt/model search; using test GT in prediction, adoption, hypothesis proposal or acquisition; silently filling missing assets with ground truth; repeating the old 34/8/6-condition studies; whole-repository dynamic-CROVE regression or repeated safety audits. The referenced papers motivate mechanisms, not extra workloads.

Do not download restricted datasets or accept new restricted-data terms on the user's behalf. Use authorized local datasets and existing permitted credentials. Official public model downloads are authorized. A missing dependency is a specific status, not permission to invent a substitute.

## 3. Workspace and startup

Create/resume `research/ovimap-module-validation-v1` from the reviewed commit. Prefer a new worktree beside the existing project. Do not reset, stash, delete, or force-push unrelated user work. If the task branch already exists, verify its ancestry and resume only its own receipts. If it is unrelated, create `research/ovimap-module-validation-v1-<first8-of-start-HEAD>` and record why.

Default output root: `/mnt/shared/ww/ovimap-module-validation-v1`; use a new `attempt_001`, `attempt_002`, ... child when an existing root has incompatible configuration. An exact compatible interrupted run resumes atomically at its first incomplete artifact. Never overwrite completed old studies.

Copy this task bundle into `docs/paper/static_ovmap/module_validation_v1/spec/`. Implement a single public orchestration entry point:

```bash
python scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --phase all \
  --output-root /mnt/shared/ww/ovimap-module-validation-v1
```

Supported phases, with explicit receipts: `bind`, `capture`, `semantic`, `geometry`, `query`, `select`, `confirm`, `report`, `all`. `all` executes missing prerequisites in this order and finishes reporting/release preparation even when a scientifically dependent phase cannot be run. It must not substitute zero arrays or toy measurements for a blocked phase. Each leaf job also accepts `--resolved-config` and can be resumed independently.

## 4. Required implementation surface

Create a small package `src/static_ovmap/module_validation/` rather than changing historical algorithm behavior:

- `contracts.py`, `assets.py`: typed records, provenance, path resolution and scene splits.
- `native_capture.py`: immutable native image/segment/state capture and replay access.
- `region_evidence.py`, `semantic_selector.py`: S.
- `entity_hypotheses.py`, `partition_quality.py`: G.
- `query_state.py`, `query_gain_policy.py`: Q.
- `evaluation.py`, `selection.py`, `reporting.py`: common metrics, model choice and reports.

Keep adapters for incompatible model environments separate. Use compact NPZ/JSONL/PNG interfaces. Do not build a service, dashboard, generic workflow engine or new experiment database.

Store official OVI patches in `third_party_patches/ovimap/module_validation_v1/`, with a script that applies them to a **separate clean checkout** at `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`. Never modify the original baseline installation in place. The checked-out C++ extension, loaded `.so`, patch digest and compiler/runtime environment belong in receipts. Patching a file without loading its rebuilt extension is not integration.

## 5. Data and experimental precedence

There are three distinct evidence levels; keep them distinct everywhere:

- **HISTORICAL:** the Room0 `AT_O_AREA` common-point-domain study. Its approximately 20.644% uAP is not the universal native OVI baseline.
- **DEVELOPMENT:** physically disjoint FIT/CAL/SELECT scenes chosen before model outcomes; FIT trains, CAL chooses teacher/head checkpoint/adoption settings, SELECT decides which modules are retained.
- **CONFIRMATION:** untouched scenes for this study, run once after `selection.json` is frozen. Known Replica8 and old benchmark scene results are exposed, not blind validation.

All three modules are coded. Their empirical prerequisites are different. A blocked training-data dependency cannot be replaced with a random crop split of Room0. Continue independent runnable work and publish exact incompleteness; never report overall empirical completion when an indispensable branch was not measured. This requirement is not a request for endless preflight checking.

## 6. Execute in stages, not a Cartesian product

1. Resolve assets and splits once; capture a faithful native baseline and input/state records.
2. Run the five direct region readouts in file 02, plus the unchanged native reference. Select the **alternative** teacher on CAL, not Room0 or confirmation; the native model remains the incumbent.
3. Fit and evaluate the paired semantic selectors if the independent FIT labels contain identifiable beneficial and harmful replacement events. An inferior teacher overall can still supply useful selective corrections; do not require a direct net win as a training gate.
4. Construct native G partitions, run the old and learned scorers on exactly the same hypotheses, and evaluate final non-overlapping partitions.
5. Collect bounded randomized **training-only** acquisition logs, fit Q, then evaluate causal equal-budget policies; no offline final-map state in the causal result.
6. Select modules using file 05. Test only the eligible combination path, not all eight combinations of three modules.
7. Freeze selection, confirm once, produce final reports and push.

## 7. Tests: necessary, small, and linked to failure modes

No test-count target. Write a compact suite for: identity/geometry invariants; crop-feature compatibility; real region-mask consumption; partition completeness and no parent/child duplication; split isolation; pre-query information exclusion; budget debit and pending-result handling; evaluator trace parity on one payload; resume invalidation. Prefer a few parametrized tests. Run one real-input smoke per genuinely distinct integration boundary (visual region adapter, C++ snapshot, causal acquisition), not repeated full-scene smoke studies.

When a check fails, fix the observed fault and rerun the affected check/downstream artifacts. Do not rerun all historical research. Do not turn negative scientific results into implementation bugs without evidence.

## 8. Completion and communication

Keep a compact `progress.md`: implemented/running/completed/blocked and the concrete next action. Finish with actual method rows, decisions, measured costs and exact file paths. Separate:

- implementation: COMPLETE / PARTIAL;
- experiments per branch: MEASURED / BLOCKED_<reason> / NOT_REQUIRED_BY_FROZEN_GATE;
- scientific result: NET_GAIN / TRADEOFF / NO_NET_GAIN / INCONCLUSIVE;
- confirmation: CONFIRMED_ON_STUDY_HOLDOUT / NOT_CONFIRMED / NOT_RUN_<reason>;
- publication: PUSH_VERIFIED / PUSH_FAILED_<actual_error>.

The final release rules in file 05 are mandatory even for a negative result. No background-work promises. No claim of end-to-end real-time speed from cached processing. No claim that these engineering adaptations reproduce entire 2026 methods or establish novelty merely by combining them.

All implementation constants below are frozen v1 engineering choices for a bounded test, not literature-derived optimal values. Fix a verified implementation error when found, but do not change scientific constants after SELECT/CONFIRM outcomes to obtain a win. Document this scope as a minimum module-validation study; it does not establish the entire proposed online research system.

---

# 01 — Assets, native integration and shared contracts

## A. Verified starting points; resolve, do not invent

Read these real files at the reviewed project commit:

| File | Actual reuse |
|---|---|
| `configs/evaluation/ovimap_t1_attribution_v1.json` | two T0 receipts, native mesh/readout, evaluator root, fixed old projections and text cache |
| `configs/evaluation/ovimap_object_decoupling_v1.json` | local native checkpoint/processor, RGB receipt, old OD and LOCAL roots |
| `configs/evaluation/ovimap_sf_ovi_semantics_v1.json` | frozen AT pool and last semantic-study output root |
| `docs/paper/static_ovmap/SF_OVI_SEMANTIC_HANDOFF.md` | real commands/environments, changed-object ledger, five evaluated payloads and one identity reuse |
| `src/static_ovmap/candidate_semantics.py` | `native_crops`, cache identity, original encoder; do NOT use `score_views`' gated final class as a direct-teacher output |
| `src/static_ovmap/ownership_evidence.py` | registered projection utilities; existing positive-instance restriction must be explicit |
| `scripts/evaluation/evaluate_static_sf_ovi_semantics.py` | fixed-geometry adapter pattern; remove copied hard-coded zero-inference fields in the NEW adapter |
| `src/static_ovmap/attribution_objects.py` and existing evaluator helpers | actual released matching/duplicates/ignore behavior, not a replacement greedy matcher |

Known recorded roots, to verify rather than assume accessible:

```text
/mnt/shared/ww/ovimap-t1-attribution-v1/predictions_verified
/mnt/shared/ww/ovimap-t1-attribution-v1/evaluation
/mnt/shared/ww/ovimap-object-semantic-decoupling-v1
/mnt/shared/ww/ovimap-sf-to-ovi-semantics-v1
/home/ww/oviovo_baseline_runs/20260913_static_ovmap
/home/ww/vv/paper2/OVI-MAP
/home/ww/vv/paper2/model_cache/google_siglip-large-patch16-384
```

Resolution order: explicit environment overrides `OVIMAP_DATA_ROOTS`, `OVIMAP_MODEL_ROOTS`, `OVIMAP_REPLAY_ROOT`; then these configurations; then receipts linked by them; then at most one bounded directory discovery (depth <=4) below the discovered data/model roots. Do not search the entire machine. Match by manifest identity, not similar basenames. Two differing plausible assets without disambiguating metadata produce `BLOCKED_ASSET_IDENTITY`, not an arbitrary selection. Resolve all paths into `resolved_config.json` and retain source receipts.

## B. Pin models and code once

| Role | Fixed candidate |
|---|---|
| Incumbent | bound local native `google/siglip-large-patch16-384`; preserve its actual processor/tokenizer/text settings |
| Alternative encoder | `google/siglip2-large-patch16-384` with its OWN text embeddings and processor |
| Alternative region recognizer | `AAwcAA/WOW-Seg`, official author weights only |
| WOW code | `AAwcAA/WOW-Seg-Meta@bfc6f2424c47097e70a641c6a8016319cac192cb` |
| Name mapper | `sentence-transformers/all-MiniLM-L6-v2` |
| Mapping code | `OVI-MAP/OVI-MAP@f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424` plus tracked scoped patch |
| 2D front end | existing native CropFormer checkpoint/config from real receipts; no guessed replacement |

For HF weights resolve the actual repository revision to an immutable commit before downloading, save config/tokenizer/processor/code/weight identities, and reuse that snapshot throughout. The WOW weight endpoint was not successfully loaded during planning; availability and model-code compatibility are an execution prerequisite, not a claimed success. An inaccessible WOW model marks only that candidate unavailable and does not erase the SigLIP2 comparison. Do not replace WOW with a plain InternVL model.

Use the existing native/evaluation environment where compatible; create an isolated WOW environment from the pinned author's requirements. Do not globally upgrade the old mapping environment. Capture package versions after successful load. Try the documented ordinary loader and one specifically justified compatibility fix; no open-ended dependency search. Eager attention and BF16 on CUDA are permitted for WOW; native and SigLIP2 use FP32, batch/chunking allowed. Microbatch reduction is permitted for OOM, but changing precision, resolution, maximum tiles or checkpoint after seeing accuracy is not. If a model remains unavailable, disclose it and continue independent modules.

## C. Bind independent supervision before fitting

Preferred development source: already authorized local ScanNet training captures with RGB/depth/poses, instance annotations and compatible semantic names. Use native loader/registration and the released scene ontology. Do not interpret the 51-class Replica label indices as ScanNet indices. Text encoders get the official scene-dataset vocabulary; learned heads receive class-permutation-invariant statistics, never class ID slots.

Build `scene_inventory.json` from native manifests/receipts and official train/val split files. Group physical scenes, including alternate ScanNet captures (`sceneNNNN_*`) and other known revisits. Exclude every physical scene in the old Replica8, old ScanNet18, prior training/selection logs and intended confirmation from new confirmation. Exclude those exposed physical families from FIT, CAL and SELECT as well; they may only be retrospective diagnostics. Querying an already evaluated scene under another capture name does not make it independent.

Deterministic selection, before image-model results:

1. For each eligible physical family, choose `_00` when complete, otherwise the lexicographically first complete capture. Completeness is modality/annotation availability, not score.
2. Sort family IDs by `sha256('ovimap-module-v1|' + family_id)` and take exactly the first12 complete training families for v1; do not expand this set based on outcomes.
3. Require12 development families for learned-module claims: first8 FIT, next2 CAL, final2 SELECT. With fewer than12, mark supervised branches BLOCKED_INDEPENDENT_SCENES; do not repurpose exposed scenes or shrink the split silently.
4. Take two distinct eligible official-validation families with the same hash ordering as CONFIRM, excluding all exposed/overlapping families. At least two are required for any study-holdout confirmation statement; record the actual count. Do not claim formal statistical generalization from two scenes.
5. Lock splits and exclusions in `splits.json` before evaluating teachers. Scene shortages have explicit statuses. Still implement all modules, complete Room0/frozen-model diagnostics and any independent available work. Do not manufacture FIT/CAL/SELECT by splitting one room's frames.

Per new capture construct the native scheduled range using the recorded start/end and native step rule, then retain its first200scheduled frame slots and set the replay end accordingly. Require at least200scheduled slots; missing actual frames are logged and not backfilled. Historical controls retain their original exact schedule. This is a paired native-protocol subset, not an assertion that an arbitrary sequence exactly reproduces the paper table. Thus each new capture uses200scheduled slots; if the native negative-step implementation would give zero for a short sequence, exclude that capture at inventory time rather than modifying the baseline silently. Preserve registration, intrinsics, units and RGB/BGR conversion. New 2D front-end cache generation is allowed only for these preselected captures and uses the same fixed CropFormer weights/config. At most14 new independent captures (12 development +2 confirmation), 200 selected frames each; reuse existing caches first. Restricted raw datasets are never downloaded automatically.

All predictor training labels are computed in a separate training-target builder from FIT annotations. Prediction packages do not contain GT paths or cause-ledger columns. CAL/SELECT annotations go only to their evaluators and selection routines. Unknown pretraining overlap is disclosed; this split does not certify foundation-model pretraining exclusion.

## D. Three geometry/input domains; never merge their scores

`H`: historical `AT_O_AREA` common SF-point domain, old 64 receiver registry, exact old projection/rank strings. Use as a retrospective control and existing-error ledger.

`N`: native OVI mesh and native ROI construction, instrumented replay with unchanged segmentation/TSDF/association. This is the authoritative new-module baseline. Active instance counts and baseline metrics are measured, not hard-coded as 64 or 20.644%.

`R`: a new replay whose baseline drifts from the historical/native receipt. If a no-op replay does not match, record the difference, investigate one concrete cause and preserve a paired replay baseline. Results may be reported against R, but never label R=N/H or count a projection/mesh change as a module gain. A materially unresolved baseline drift prevents deployment promotion, not code delivery.

For S and Q comparisons the geometry/instance registry is fixed within their chosen native study. G changes instance labels only on the SAME native surface coordinates. Compute the native-source-to-GT projection once per scene/domain using the existing evaluator convention and freeze it for every G condition. Do not compare a G voxel-center map with an N mesh baseline.

## E. Native capture hooks — required active call sites

In the pinned `scripts/panoptic_mapping_.py` the active path is:

```text
frameToSegmentsCropFormer -> insertSegmentsOpen -> integrateFrame
-> raycastInstancePredictions -> select_views_for_frame
-> global bbox + (local mask OR global mask) -> worker request
-> clearTemporaryMemory
```

Capture at three points:

1. Before insertion: frame-local refined depth/entity segments and their pixel support/provenance.
2. After `integrateFrame`, before `clearTemporaryMemory`: registered segment labels and alias/merge events, plus label-to-current-instance mapping.
3. After raycast, before expensive visual dispatch: current global instance raster, local PNG, native bbox, union mask, visible area, pose/time and selection metadata. Capture **all technically admissible candidates**, not just the original chosen queries, for Q. Store selected native queries separately for baseline fidelity.

New C++ read-only accessors (these are required NEW interfaces, not existing methods):

```text
exportStudyFrameState() -> frame segment registration + aliases + active association configuration
exportStudySurfaceLabels(native_xyz) -> numerical segment/instance labels aligned with the exact input surface rows
```

Implement them beside `GlobalSegmentMap_py`, bind them in its actual pybind registration, and return copies under the existing appropriate locks. `segments_to_integrate_` contains registered `segment->label_` after integration and is deleted by `clearTemporaryMemory`; late export loses the evidence. `generateMesh` already has independent label/instance mesh flags; this is useful for checking alignment, but do not infer numerical labels from colors without the actual color map.

The wrapper initializes SegGraph only for association modes 3/4/6/7. Read actual run arguments/receipts; do not assume these modes are active or switch the mode to make an export convenient. Trace `getInstanceLabel` and the real mesh color/label lookup. Implement surface-label lookup through the SAME mapping/alias semantics used by the native mesh, and compare its incumbent-instance output to the existing native mesh registry. If a segment graph is inactive, export registered segments/current mapping from the active fusion path; don't patch a dormant graph and claim an effect. Do not assert API names beyond these new contracts without inspecting their definitions.

G v1 reads these structures into a Python hypothesis sidecar and writes a separate **static export**. No live TSDF owner mutation or semantic feedback to association is required. This is deliberately a bounded first test of complete-partition quality; it is not yet an online multi-hypothesis mapper.

## F. Captured schemas

`NativeScenePack`: scene/family/split/domain; source config/revisions; ordered surface XYZ/normals; original per-point owner; segment labels and alias table; original instance registry; untouched TSDF digest; frozen projection identity; frame list; class vocabulary/text-space identity. Missing normals use an explicit validity bit and the same fixed local PCA estimate for every condition, not hidden GT normals.

`FrameObservation`: frame ID/pose/registered image size/intrinsics; RGB/depth/PNG paths+hashes; native current map-state ID; refined segments and registered labels; current global owner raster; complete admissible ROI table; native selected ROI IDs; request completion boundary. Full data arrays are in NPZ or PNG, not megabyte JSON lists.

`RegionRequest`: scene, frame, persistent target/lineage, source-map version, exact target mask hash, bbox, native union mask hash, visible target pixel count, crop convention, requested view rank, image hash. Any newly shaped G target gets a NEW request key.

`PredictionPayload`: domain; ordered source coordinates; owner vector; active IDs; class vector; rank values and exact serialized strings; projection/evaluator identity. Semantic conditions may change classes, never masks/owner/ranks; Q changes paid observations/readout but not geometry; G changes owner/registry and keeps source geometry.

Every cache key includes its full consumed data/model/processor/prompt/feature/schema/config identity. Actual evaluator reuse is keyed on final masks+classes+ranks+projection+evaluator, not method name. Identical payloads may reuse evaluation, but receipts must distinguish reused computation from new independent evidence.

## G. Serialization and small runtime limits

Serialize a disabled adoption threshold as the string `KEEP_ALL`, never JSON Infinity. Undefined metrics are null; no NaN/Infinity in committed JSON. Arrays may contain masked missing values only when accompanied by explicit validity and excluded from numerical fitting as specified. Run at most one visual model per GPU at a time; default one CUDA device, CPU worker limit8, no automatic multi-node training. The native CPU mapper and its existing separate visual worker are permitted. Training heads use CPU or that one available GPU. This is a workload cap, not a promised latency or memory bound. Historical Room0 capture, when required to bind the known baseline, is at most one additional already-authorized200-frame replay; count it separately from the14 independent captures.

Initial `capture` processes FIT/CAL/SELECT only (plus the optional bound historical replay). CONFIRM mapping/visual processing is deferred to the `confirm` phase after selection is locked. The bound data inventory may check confirmation file availability without reading its outcomes. If the annotation ontology, native sampling or pose registration cannot be resolved from actual assets, record that concrete blocker rather than invent a universal ScanNet configuration.

---

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

---

# 03 — G: native-leaf complete partition hypotheses

## G0. Precise scope

This is a STATIC refinement/export experiment on unchanged native TSDF surface geometry. It is not a new segmentation backbone, a LIVE multi-hypothesis mapper, or a reproduction of LEGO/GeoGuide. Native geometric leaves and frame evidence are captured through file01's active C++ integration path. Keep all surface positions, faces, projection, depth and TSDF values identical across G conditions; only instance ownership changes. No SF candidates enter G v1. Historical LOCAL/object-arbitration outputs remain negative controls, not implementation starting masks.

The experiment separates hypothesis coverage from hypothesis scoring. Never report an oracle-selected partition as a predicted method. Never prefilter hypotheses using GT or the old acceptance gates. A new learned scorer must see the SAME complete hypothesis set as its old-agreement control.

## G1. Numerical native leaves and adjacency

Export registered numerical segment labels, alias resolution and incumbent owner on the exact native surface rows. The no-op export must reproduce the native owner assignment. Do not recover IDs from arbitrary RGB colors. If this is unavailable, implement the C++ read-only API and an affected real smoke; do not synthesize labels from final SF/OVI binary masks.

A leaf is a connected component of `(resolved numerical segment label, incumbent owner)` on native surface adjacency. This ensures original ownership can always be represented by unions of leaves. Use existing mesh edges when valid faces are exported. Otherwise use mutual 8-nearest-neighbor links whose Euclidean length is<=0.03m, generated in chunks, with stable source-row tie breaking. This graph construction is a declared fixed source-domain approximation, not true mesh topology. Surface normals use native values, otherwise the same PCA16-neighbor estimate with an explicit validity flag. Do not use GT mesh topology or normals in prediction.

Keep owner0 geometry unchanged outside candidate groups. A leaf cannot be divided by a downstream pointwise argmax. If native leaves are too coarse to express a needed split, report `COARSE_LEAF_LIMIT`; such a result does not validate all possible multi-granularity methods.

Construct the leaf contact graph from the same surface adjacency. Count adjacency edges; aggregate mean absolute normal dot product, spatial distances and per-frame visible entity evidence. For each of the frozen at-most32 pose-distinct/evenly sampled frames, project the current native surface using the shared camera convention and depth tolerance0.05m. Retain depth-consistent observations with local label0 as UNKNOWN, not contrary evidence. Do not make old `positive_instance_pixels` the denominator for all visible evidence.

For a leaf, assign a frame entity ID only when at least16 of its projected pixels are observed and a positive local entity covers at least0.6 of all depth-consistent observed leaf pixels. Otherwise that frame is unknown. A pair has known same/different evidence only when both leaf labels are known in that frame. Count frame observations once; do not count every surface vertex as an independent vote. This evidence supplies features/proposals; it is not a ground-truth correctness score.

## G2. Bounded disjoint conflict groups and hypotheses

Build adjacent incumbent-owner pairs with at least one leaf contact edge. Define a prediction-side conflict strength as the fraction of known cross-owner leaf/frame observations having the same 2D entity, plus the fraction of within-owner adjacent leaf/frame observations having different entities (each fraction0 if denominator0). Sort adjacent pairs by descending strength then ascending pair IDs. Greedily accept a pair only if neither owner was assigned to an earlier group. Remaining owners form singleton groups. Take at most64 groups/scene, ordering by descending conflict strength then owner tuple; untouched owners retain N0. Pair groups allow changing TWO parents together, unlike old single-parent protection. This is bounded, not arbitrary global set partitioning.

Each group covers its original owner's entire source support, not merely its intersection with a new mask. Max1024 leaves/group for spectral candidates; oversized groups still permit original and merge candidates and have a documented spectral omission. No downsampling that silently deletes leaves.

Generate complete partitions of all group leaves:

1. ORIGINAL: exact original owner partition, always present.
2. MERGE: one group containing all leaves, only for a two-owner group.
3. FRAME_ENTITY: for each captured frame with two or three known positive entities in the group, assign known leaves by their dominant frame entity. Assign unknown leaves to the nearest centroid of known leaves of an entity, weighted by leaf surface-point count; Euclidean ties use the entity's minimum source-row ID. If no known support exists, no candidate is generated. Thus each proposal partitions all leaves; the uncertain-fill fraction is recorded as a quality feature, not hidden.
4. GRAPH_SPLIT2 / GRAPH_SPLIT3: use the leaf contact affinity `exp(-distance/0.03)*(0.1+0.9*abs_normal_dot)*(0.25+0.75*same_fraction)`. When normals are invalid use normal factor1; when there is no known pair evidence use same_fraction0.5 and a missing bit. For disconnected contact components set intercomponent affinity0. Compute the normalized graph Laplacian's smallest k eigenvectors, deterministic eigenvector sign (largest-absolute coordinate positive), row-normalize, then k-means k=2/3 with random_state17,n_init1. No alternative clustering grid. Skip k when fewer than k leaves; empty clusters invalidate that candidate. The original and all other valid proposals remain.

Canonicalize a partition by sorting output components by their minimum source-row index and encoding leaf->component integers. Deduplicate exact partitions. Keep ORIGINAL, then MERGE, GRAPH_SPLIT2/3, then FRAME_ENTITY proposals in descending known-pixel support and ascending frame ID, up to8 hypotheses/group. Do not remove a hypothesis because its old teacher agreement does not improve. Do not apply benchmark100-point filtering or the old85% support-retention rule here.

Every candidate completely partitions the same group geometry; output components cannot overlap. Empty components are invalid. Combining selected disjoint groups preserves whole-scene unique ownership, except preexisting owner0. New component IDs derive deterministically from scene/group/partition/component identity; original IDs are retained when the component support is unchanged. Keep ancestry, not just new ordinal IDs.

## G3. Old agreement and learned complete-partition score

### G_AGREEMENT

For every candidate and valid frame, rasterize its component IDs at depth-consistent visible surface samples. Match candidate components to positive 2D entities with a maximum-intersection Hungarian assignment. Frame agreement is sum assigned intersections / all depth-consistent group samples, INCLUDING unknown/unmatched samples in the denominator. Frame agreement with no denominator is missing; average over defined frames. Select the maximum mean score, tie ORIGINAL then canonical hypothesis order. There is no inherited `gain>=0.03` adoption gate. This isolates old information's ranking ability from the old conservative threshold. Name it `G_AGREEMENT`, not an exact reproduction of the earlier OD method.

### G_QUALITY features and target

For each complete partition record this 20-dimensional, permutation-invariant vector, plus20 availability bits:

1 log1p(group source size); 2 log1p(leaf count); 3 component count; 4 minimum component area fraction; 5 maximum area fraction; 6 normalized entropy of component area fractions (0 for one component); 7 count of disconnected surface components / output component count; 8 fraction of contact edges crossing output components; 9 mean absolute normal dot for within-component edges; 10 corresponding cross-component mean; 11 same-entity fraction on within-component known leaf/frame pairs; 12 same fraction across components; 13 different-entity fraction within components; 14 different fraction across components; 15 fraction of depth-consistent pixels with unknown2D labels; 16 fraction of leaves with any known view; 17 mean fraction of leaves assigned by the FRAME_ENTITY nearest-centroid completion (0 for other proposal types, but no proposal-type ID is provided); 18 visible-pixel-weighted old agreement; 19 mean component bounding-box diagonal / group diagonal; 20 mean within-component normal dispersion `1-norm(mean(unit normals))`.

Undefined continuous values have value0/availability0; standardize using FIT as in S. Feature17 is a measured proposal construction property, not GT. All sources use the same definitions. Do not pass semantic classes or GT-derived shape descriptors.

A separate FIT target builder matches predicted components to **whole** GT objects by IoU on the frozen projection. For a group include every GT object with any valid intersection with group support; use the full GT object's denominator, not its clipped group fragment. Hungarian-match pairs with strictIoU>0.5; let

`quality = sum(matched IoU) / (TP + 0.5*FP + 0.5*FN)`;

zero denominator produces0 and an explicit flag. GT objects outside the group with no intersection do not enter this local target. Label this a local PQ-like training target, not the released AP. It penalizes damage to both old objects and newly proposed components. Original is scored by exactly the same target. There are no test-oracle choices in prediction.

Train a40->32(ReLU)->1 sigmoid head, seed17, Adam0.001, weight_decay0.001, batch64 groups, max100epochs. Loss: per-group mean squared quality error plus0.1 times mean `softplus(-(score_better-score_worse))` for hypothesis pairs whose target gap>=0.05. Average groups equally within scenes, then scenes equally. Use the same CAL-every5/three-nonimprovement-check early stopping, selecting lowest CAL loss. Require at least20 FIT groups with differing quality targets across at least4 FIT scenes; otherwise code/datasets/G_AGREEMENT are delivered and learned G is `BLOCKED_PARTITION_TARGET_SUPPORT`.

Choose a CAL adoption margin from `[0,0.02,0.05,+infinity]`: accept a non-original maximum-scoring hypothesis only when its score exceeds ORIGINAL by strictly more than the margin. Pick by mean canonical class-agnostic AP50, then mean canonicalAP75, then fewer changed source points, then larger margin. Freeze before SELECT. An all-original winner is NO_INTERVENTION, not an effective geometry module.

## G4. Required experiments, same geometry and semantics conventions

| ID | Hypothesis pool | Selection |
|---|---|---|
| G_ORIGINAL | same stored pool | ORIGINAL |
| G_AGREEMENT | same stored pool | mean 2D agreement |
| G_QUALITY | same stored pool | learned complete-partition quality, CAL margin |

First evaluate class-agnostic geometric correspondence/coverage and the existing canonical AP50/AP75 diagnostic. Do not rename canonical diagnostic AP as paper-released class-agnostic AP unless an actual protocol parity test establishes this. Keep normal released semantic-instance and semantic metrics too.

For all G rows, use source component point count as the same deterministic numeric ranking rule, and re-read semantics of their ACTUAL final masks with the SAME frozen native SigLIP, max3 views and native six-crop area fusion. For G_ORIGINAL this is the matched re-read control. Do not inherit parent categories for new children. Define their ROI by final native component projection with depth visibility, bbox from its own projection, and union with the maximum-overlap original local entity, retaining full construction identity. This differs from online native capture for newly shaped masks and is explicitly a static post-map re-read. N0 is separately reported; G gains must not be attributed to semantic-input or rank bridges.

Unchanged masks can reuse exactly matching S/native requests; changed masks require new keys. At most128 final target masks/scene/condition receive re-encoding, ordered by the same prediction-only hash convention. Unchanged unavailable targets retain their original category; a new unavailable component has semantic0, not an inherited or GT label. All components remain in geometric diagnostics. Report counts of such unknown components and include their missed semantic matches; do not hide them from evaluation.

Oracle best-in-pool quality/coverage is optional only as a compulsory **diagnostic table computed from the already evaluated finite pool**, with name `ORACLE_DIAGNOSTIC_NOT_METHOD`. It does not incur new inference or select a runtime configuration. Show whether failure is inadequate leaves, no useful complete hypothesis, scorer choice, or semantic reread. The released predicted rows remain those above.

## G5. Integration output

Deliver the numerical native snapshot adapter, leaf graph, finite partition manifest, feature/target builders, old and learned selectors, final unique-map exporter and traces. No live owner override is claimed or required. Do not expand to a new TSDF mapper if this finite experiment is negative. Put implementation status and actual data limitation in the handoff separately.

---

# 04 — Q: causal, pre-encoding query utility under a common budget

## Q0. Scope and honest name

Implement a causal replay driven by CONTEMPORANEOUS native mapping states captured during the unchanged OVI trajectory. This is not final-map retrospective top-K selection, robot trajectory planning, or end-to-end real-time deployment. Geometry formation in the inspected native main loop does not consume the VLM output; therefore its time-indexed native geometry snapshots can be shared across query policies without replaying the TSDF for every policy. Verify this property in the actually patched execution path and record it.

A Q policy sees current RGB-D geometry/region metadata and previously acquired successful features only. It cannot read the final map/registry, final incumbent labels, future frames, GT, or unacquired per-view VLM features. Final registry/projection may be used only at end-of-sequence export/evaluation. A file path to a precomputed feature is NOT permission to access it before acquisition.

Q v1 uses only frozen native SigLIP with the original six crops. It tests budgeted query RANKING. It does not learn expert routing or a separate optimal stopping rule. Finite per-frame budget availability, technical eligibility and sequence termination determine skipping. This smaller formulation prevents a negative utility threshold from silently disabling the whole experiment.

## Q1. Split native candidate generation from native decisions

The current `select_views_for_frame()` both filters candidates and mutates coverage. Do not merely call it and rank the already filtered output; that cannot test acquisition outside native choices. Create a pure candidate builder preserving native technical requirements and metadata: current nonzero global owner, global area>=500, majority-local-entity area>=1000, usable current depth and a nondegenerate native crop. Preserve the native handling of majority entity0 and document it rather than silently changing the baseline. Expose local/global masks and union; retain selection-independent geometric coverage features.

Keep an unmodified native combine implementation as a control with its original side effects. In particular, the current coverage routine mutates its array before the later visible-area check; a seemingly harmless refactor must not retroactively rewrite historical control semantics. New policies use their own explicitly separate state and the same raw candidate universe. Two states are distinct: all geometrically observed surface coverage (no visual encoding needed), and coverage/history of actually acquired visual evidence. Name both; never count an unqueried view as paid semantic evidence.

Capture genuine frame-local IDs, registered segments and alias events BEFORE clearing native temporary memory. Maintain query-state aliases only as they become known at that time; merge past feature lists and spend counters by known ancestry. Do not discover final aliases early. A split whose retained feature provenance cannot be assigned to the current child is marked unavailable for that child, not blindly copied. Unresolved lineage blocks causal claims with `BLOCKED_CAUSAL_LINEAGE`, but offline S/G may continue.

For deterministic module isolation, use a per-frame result barrier for ALL Q policies: select and debit the frame's whole batch using only the pre-dispatch state, run/retrieve those paid queries, then admit successful features before the next frame. Record this controlled schedule. It is not a throughput claim about the author's asynchronous worker. Do not let another policy's cache fill the current policy state.

## Q2. Budget, readout and four policies

Main allowance B=200 target-view query ATTEMPTS per scene. One attempt costs six native crop inputs; failures cost their actual attempted work and consume their one-attempt allowance. A repeated exact request is skipped rather than repaid. For planned frame index t=0..F-1, the cumulative allowance is `floor(B*(t+1)/F)`; unspent allowance carries forward. Invalid input frames still advance planned time. The known native frame schedule F is allowed metadata; future model outputs and final geometry are not.

At each frame, quota is cumulative allowance minus attempts already spent. Rank currently technically eligible requests not previously attempted; select up to the quota. There is no per-object acquisition cap in Q v1. Keep at most the10 largest-visible-area SUCCESSFUL features for the common current semantic readout, tie by frame/request ID; all paid attempts remain in the cost ledger. Retention is not a refund. The first query of an unseen object has no cached label: its prior is uniform over the dataset vocabulary. After queries, area-weighted feature fusion followed by cosine classification is identical for all four policies. At export, an instance with no usable acquired evidence has class0; do not give it the final native label for free.

| ID | Selection from currently available requests |
|---|---|
| Q_COMBINE | requests that the native combine control would admit from its OWN paid/query state; descending overlap area then owner/request ID under the allowance |
| Q_AREA | all technical candidates, descending overlap area, deterministic IDs |
| Q_UNCERTAINTY | all technical candidates, descending normalized entropy of past acquired class scores; unseen=1; ties overlap area then IDs |
| Q_GAIN | all technical candidates, descending predicted one-step loss reduction; ties overlap area then IDs |

Q_GAIN fills its available quota even when predicted scores are negative; primary v1 measures relative acquisition priority, not stopping. This removes a confounded reject-everything gate. Q_COMBINE can use fewer requests because its native novelty gates exclude candidates; report this. Q_AREA/UNCERTAINTY/GAIN have the same raw eligibility but may differ in successful cost. Describe the main result as an equal **allowance**, not necessarily equal realized FLOPs; actual debit/cost curves decide stronger cost-dominance claims.

No teacher calls, background-only crops, S selector, G refinement or re-ranking enter these four standalone Q rows. The unbudgeted original native N0 is separately shown for context, not called an equal-budget control.

## Q3. Training-only bounded acquisition exploration

Use FIT native contemporaneous snapshots, a fresh initially empty semantic state, and seed17 random request ranking within each frame; maximum512 six-crop attempts/scene, paced as in Q2 with B=512. No acquisition target uses final-map geometry. Log candidate metadata and prefix state before selection, then the acquired feature and readout change. All image encoding for these training labels is counted as training cost, not hidden in inference speed.

On CAL collect an independent seed23 random trace with B=256 per scene for head checkpoint assessment. Do not create exploratory inference traces on SELECT/CONFIRM. Existing compatible randomized traces may be reused exactly; native-selected-only logs cannot be claimed to be unbiased exploration data.

A separate supervised target builder projects authorized annotations onto the CURRENT visible region. A target is identifiable if it has at least64 valid annotated region pixels and one allowed class accounts for>=0.8 of them. This is a training-label rule only, not query eligibility. The GT category never appears in policy features. If no identifiable target, retain the paid trace but exclude its loss target.

Before/after distributions use softmax(current native cosine/0.07), or uniform prior before any feature. Target `y=clip(NLL_before-NLL_after,-5,5)` for the same identifiable class. New queries can increase loss; keep negative labels. This is a semantic acquisition proxy, not AP decomposed into additive rewards. Do not train only on positive queries or label an unobserved action with fabricated gain.

Require>=100 identifiable FIT acquisition events from>=4 physical FIT scenes, including at least10 positive (>1e-6) and10 nonpositive events. Otherwise Q_GAIN fitting is `BLOCKED_QUERY_TARGET_SUPPORT`; the other three policies are still measured if causal capture works.

## Q4. Fixed utility head

Twenty scalar features plus20 availability bits, in this order:

1 log1p(current global mask pixels); 2 log1p(local-majority mask pixels); 3 intersection/global fraction; 4 intersection/local fraction; 5 target/bbox fraction; 6 valid-depth fraction; 7 median visible depth; 8 depth interquartile range; 9 new geometrically observed spherical-cell fraction; 10 new acquired-evidence spherical-cell fraction; 11 log1p(past successful queries); 12 log1p(past attempts); 13 frames since last successful query / F; 14 past retained-feature normalized class entropy; 15 past top1-top2 cosine gap; 16 mean pairwise retained-feature 1-cosine; 17 current overlap area / max(1,best previous paid overlap area); 18 camera translation from last paid pose in meters; 19 camera rotation from last paid pose in radians; 20 remaining attempt fraction.

All unavailable history features use0+availability0; no vectors of class IDs or scene IDs. Normalize with FIT-only statistics as in S. No image encoder, current-crop text score, teacher answer or hidden-state computation may be used to rank an unacquired request. Cheap geometry-based quality computation is allowed and timed.

Head40->32(ReLU)->1(linear), seed17, Adam0.001, weight_decay0.001, Huber(delta1) loss, batch128, max100epochs, CAL MSE every5epochs, stop after3 nonimproving evaluations, choose lowest CAL MSE/earlier epoch. Weight scenes equally. One fit, no hyperparameter grid. Clip predicted gain to[-5,5] only for numeric stability; ranking unchanged except ties. Negative predictions remain valid.

Save a prefix-sentinel test: replacing any unacquired or future cached feature with an impossible sentinel must leave all decisions prior to its actual acquisition unchanged. This is a targeted causal-integration check, not a generic security audit.

## Q5. Evaluation, optional budget curve and cache accounting

Run the four policies at B=200 on SELECT. Evaluate their final maps using the same geometry, vocabulary, ranks and released projection. Then, ONLY when Q_GAIN is eligible under file05, run Q_GAIN and the frozen best nonlearned Q comparator at B=100 and400. These four additional curve rows are prescribed, not another model sweep. Do not refit the head or change readout for the curve.

Maintain two ledgers:

- physical execution cost: all actual forwards/cache hits, model loads, tiles, inference times;
- policy logical acquisition cost: selected request attempts, successful requests and crop-equivalent cost, debited even when a feature is physically cached by another method.

A shared feature store sits behind `acquire(request_id, budget_token)`. The policy sees only admitted returned features; it cannot inspect the store. Acquired cached data can save physical computation but not a method's logical budget. Do not precompute all SELECT/CONFIRM request features for convenience. Document retained/evicted features and failed-query costs.

Report uAP, AP50, mIoU, classes/masks without observations, actual requests/crops, realized cost, and optional curves. If gain is only lower compute at similar accuracy, label it efficiency evidence. If current-map snapshots are missing and only final-map reprojected requests exist, label an offline diagnostic separately and BLOCK the causal Q experiment; never rename it online acquisition.

---

# 05 — Evaluation, selection, confirmation and GitHub publication

## E1. Preserve the evaluator, separate input domains

Reuse the existing released semantic-instance evaluator and its trace instrumentation (`bind_protocol`, `evaluate_set`, `semantic_metrics`, `attribution_objects`) through a NEW adapter. Inspect their actual signatures at the pinned project ref. Do not pass new methods through historical hard-coded METHODS lists. Do not change native GT filtering, strict IoU comparisons, minimum region size, instance/semantic subsets, ignored/duplicate handling, nearest-neighbor projection tolerance or class-name order.

The old SF evaluator hard-codes new inference=0 and compares against a fixed64-row H pool. Neither is correct for this study. Replace those fields in the new adapter with actual branch/model/capture ledgers and a scene-specific registry. Preserve the old adapter and tests.

Distinguish released semantic-instance uAP and AP50/AP25; semantic mIoU/mAcc; class-agnostic canonical geometric diagnostics; overlapping-candidate AP; and actual runtime/cost. Do not substitute one for another or call canonical AP the paper metric without parity evidence. A class-only change may alter instance eligibility even with fixed shape; this is not an invariant violation. Report full precision values plus percentages, and differences in percentage POINTS, not percentages. Scene averaging is unweighted mean of scene metric values where defined, with denominators shown. Undefined values stay null, not0. A required selection metric undefined on any selection scene makes that comparison INCONCLUSIVE rather than silently dropping the scene.

Calculate full input/prediction identity, not method-name identity. Identical final payloads may reuse evaluations once checked, but all logical method rows remain. A copied row is not an independent validation. Every S/Q row must pass fixed-geometry and fixed-rank equality. G rows must share source XYZ/faces/TSDF/projection and satisfy unique ownership and partition completeness. GT-dependent diagnostics only append AFTER prediction manifests are locked.

## E2. Fixed selection hierarchy — no cherry-picking

FIT fits model weights; CAL selects teacher/head checkpoint/adoption margin; SELECT evaluates frozen options and chooses retained modules. CONFIRM is opened only after selection is locked. This study defaults to12 development families (8FIT/2CAL/2SELECT) and2CONFIRM families, not24+4; do not expand v1 after seeing outcomes. The small numbers justify exploratory conclusions, not universal guarantees.

Use tolerance1e-10 on metrics in their [0,1] scale for tie/no-difference classification. A positive result should additionally be reported with per-scene changes and a descriptive scene-bootstrap interval (seed17,2000 resamples, no training reruns). The interval with two scenes is not robust population inference. No required arbitrary AP gain is promised.

### S selection

On SELECT, identify the best available DIRECT alternative by the CAL-frozen teacher ID; do not reselect the teacher using SELECT. Also show all direct controls for interpretation. Compare S_PAIRED to N0, to its same-teacher S_SIMPLE and S_NO_CONTEXT, and to the CAL-selected direct teacher.

S_PAIRED is mechanism-eligible only if: it makes>=1 actual nontechnical replacement; mean uAP improves over N0 AND S_SIMPLE beyond tolerance; mean mIoU is not below N0 or S_SIMPLE; and the uAP gain over N0 occurs in both SELECT scenes. If its only win is over a degraded matched-input native control, it is not eligible. If S_NO_CONTEXT matches/beats it within these rules, retain the simpler NO_CONTEXT as an engineering choice, but the object/background feature contribution is UNSUPPORTED. When S_SIMPLE matches/beats the complex heads, report useful supervision/selection baseline if applicable, not validation of the proposed contextual mechanism.

Select an S deployment candidate among N0, the CAL-selected raw teacher and the3 learned heads using SELECT mean uAP, then mIoU, then lower added cost, then complexity order N0 < raw teacher < SIMPLE < NO_CONTEXT < PAIRED. It is usable only if mean uAP strictly exceeds N0, mean mIoU is noninferior, and both SELECT scenes have nonnegative uAP deltas. Otherwise retain N0; label other points TRADEOFF rather than hide them. The no-context and simple options remain scientifically distinct.

### G selection

G_QUALITY is eligible when it changes>=1 entire partition; mean canonical class-agnostic AP50 exceeds both G_ORIGINAL and G_AGREEMENT; mean canonical AP75 does not decrease versus G_ORIGINAL; its matched native semantic readout has mean released uAP and mIoU not below G_ORIGINAL; and canonical AP50 improves in both SELECT scenes. If the native N0 bridge is materially different, expose it and require a paired-domain interpretation. A gain solely from fresh semantic reread does not qualify as a geometry contribution.

Select between ORIGINAL/AGREEMENT/QUALITY by canonical AP50, thenAP75, then semantic uAP, then fewer changed points, with originality as final tie preference. Retain a changed G candidate for combination only if it passes the same geometric+semantic noninferiority rule against ORIGINAL; choose ORIGINAL otherwise. A useful G_AGREEMENT is an engineering result, not evidence for learned quality. Oracle rows never enter selection.

### Q selection

At B=200 choose the strongest nonlearned Q comparator on CAL by mean uAP, then mIoU, then lower logical attempted cost, then ID order COMBINE/AREA/UNCERTAINTY. Run those three CAL policies once; reuse compatible acquired requests and debit each policy separately. Q_GAIN training checkpoint still uses CAL MSE, not another policy-driven hyperparameter search.

Q_GAIN is eligible if its SELECT mean uAP exceeds that locked comparator, mIoU does not decrease, and both SELECT scenes have nonnegative uAP deltas with at least one strictly positive. All rows use B=200. If Q_GAIN spent more actual logical cost, call it an allowance-constrained accuracy result, not equal-realized-cost dominance; the B100/400 curve tests efficiency separately. A cheaper noninferior result without strict AP gain is EFFICIENCY_ONLY and may be retained as an engineering option, not a third precision contribution. Unknown-label output and failed costs cannot be omitted. No final-label warm start.

## E3. Combination gate — at most two new variants, no giant factorial

The three standalone branches are mandatory when their prerequisites exist. Combinations are conditional:

1. If an S candidate and a changed G candidate are independently retained, run `COMBO_GS` on SELECT. Apply G to native geometry, produce fresh semantic evidence on each actual final component, then apply the frozen S decision to the corresponding native-versus-teacher alternatives. Regenerate evidence for changed masks; do not copy parent features. The S head weights and threshold remain frozen; failure on changed masks is a real distribution-shift result. Unchanged requests may be reused. Use the128-target cap, count unknown/new components and all extra encodings.
2. If COMBO_GS (or retained S-only/G-only when only one is retained) passes its noninferiority check, and Q_GAIN is independently retained, run `COMBO_Q_REFINEMENT`. First complete Q's causal mapping/readout. Then apply the retained static G/S refinement on actually observed RGB-D, with all refinement queries counted separately. Its incumbent labels come from Q or explicitly paid fresh-mask native reread, never free final N0 labels; all model/head settings remain frozen. This is a **causal-query plus static-refinement hybrid**, not fully online mapping and not a B200-total-cost method. Provide a matched baseline with the same static-refinement allowance. Do not claim the three modules add linearly. If the required fresh-mask input or feature provenance is unsupported, block this combination specifically.

Select a final candidate only if mean released uAP exceeds its matched native baseline, mIoU is noninferior, and every SELECT scene has nonnegative uAP delta. Otherwise prefer the best independently supported simpler candidate. Never combine the best AP of one map and best mIoU of another. Save every measured negative combination too. At most two combination variants; their necessary paired controls are not new model choices.

No retained modules means final selection=N0 and scientific result=NO_NET_GAIN. This is a legitimate conclusion and still requires code, results and publication.

## E4. Confirmation without redesign

Write `selection.json`, frozen resolved configuration, teacher/heads, split and prediction-code commit identity BEFORE opening CONFIRM labels. On the2preselected confirmation scenes run N0, the selected final candidate, and its nearest required same-model/simple comparison. Maximum4semantic/pipeline rows per confirmation scene. For a selected Q candidate include the locked B200 comparator; budget curves are SELECT-only. Do not evaluate every teacher and head on confirmation and select another winner there.

If candidate=N0 or prerequisites for a real retained candidate are absent, confirmation is `NOT_REQUIRED_NO_RETAINED_CANDIDATE` or the actual asset blocker. If a candidate fails confirmation, publish NOT_CONFIRMED without changing thresholds, retraining or choosing a different model on those results. Record study holdout separately from unknown foundation-model pretraining overlap and prior benchmark exposure. Reusing existing exposed Replica8/ScanNet18 scenes is retrospective validation, never the same as this confirmation split.

## E5. Required reports and machine artifacts

Commit small artifacts to `artifacts/static_ovmap/module_validation_v1/`:

- resolved config, environment/model/source lock, splits/exclusions and asset inventory;
- native-domain bridge/parity and active C++ path evidence;
- complete method matrix with required/run/reused/blocked/not-required rows;
- target-view/input validity and teacher generation/name-mapping ledgers;
- S feature schemas, FIT/CAL sizes, checkpoint and all CAL threshold decisions;
- G hypothesis/partition summaries, provenance, scoring and completeness records;
- Q prefix/acquisition/cost summaries, training exploration and causal checks;
- per-scene metrics, all actual object-match gains/losses and eligibility transitions;
- teacher/module/final `selection.json`, confirmation results;
- compact tests, exact commands, stage timings, errors and reproducibility entrypoints.

Large raw surfaces, raster sequences, crops, all logits and backbone weights stay outside ordinary Git. Commit their actual paths, byte sizes, hashes and regeneration commands; a manifest is not a file upload. Save small learned weights/scalers as NPZ/JSON or safe framework weights under the experiment directory if each<=10MiB and total release additions<=100MiB. Do not serialize arbitrary Python model objects when simple weights suffice. Retain checkpoint choice and training provenance.

Required Markdown:

`docs/paper/static_ovmap/MODULE_VALIDATION_RESULTS.md`
`docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md`

RESULTS has exactly five principal tables (supporting appendices allowed):
A. all branches/per-scene metrics and cost; B. semantic suggestion->adoption->correction/damage; C. geometry hypotheses->selected complete partitions->object preservation; D. query availability->paid acquisition->readout improvement/cost; E. module selection/combination/confirmation with evidence paths and status.

HANDOFF specifies actual source and extension identities, worktree, branch, data/model paths, completed/blocked scope, how to reproduce each leaf job, remaining technical dependencies and the one next evidence-based research action. Do not repeat the full prior historical audit or manufacture an attractive positive narrative.

## E6. Audit once, then publish actual work

Before the final push, perform ONE scoped final audit:

- every required row has a measured payload or an explicit evidenced prerequisite/gate status;
- every positive claim matches the actual domain/control and a changed effective prediction;
- model/threshold selection uses CAL and module selection uses SELECT only;
- no GT label/error ledger reached prediction or query features;
- G exports complete unique partitions on the same source geometry;
- Q decisions use prefix state and pay for every acquired query;
- numeric summary reconciles to machine metrics; no hidden fallback, wrong-inference=0 field, duplicated independent run or best-column splice;
- all runnable code/configs and small result/head files are in the publication set, not only MDs.

Fix concrete faults and rerun ONLY invalidated artifacts. This is not permission for a second broad audit campaign or global regression suite.

Record a code commit BEFORE experiments (after executable configuration/code is ready). Experiment receipts refer to that exact code commit plus any dirty diff digest; final corrected predictions must reference the actual generating code. The final result/handoff commit may be later. Do not attempt to embed a final Git commit's own SHA inside a file belonging to that same commit.

Stage explicit scoped paths, inspect `git diff --cached --stat`, commit actual code/config/results/head weights/docs. Run ordinary push of the task branch; no force push and no automatic merge into main. Resolve a normal authentication/network failure only through existing permitted credentials; never print tokens or upload credentials. Fetch the remote task branch, compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/<actual-branch>`; equality after successful push is required for `PUSH_VERIFIED`.

Write a local publication receipt AFTER final commit containing the compared SHAs and commands; this receipt can remain a delivery artifact outside that self-referential commit. The remote branch contains the source/results/handoff, not a promise to upload them. A GitHub Release/tag is optional and not required; branch publication satisfies this task. If push fails, deliver the commit, actual error and a local scoped archive, report PUSH_FAILED, and do not claim remote success.

The final Codex response must state implementation/experiment/science/confirmation/publication statuses separately; list final candidate and measured headline metrics, actual inference/training/capture costs, report paths, branch and full verified SHA. Complete negative results and accurately published partial results are preferable to fake completion or another plan.
