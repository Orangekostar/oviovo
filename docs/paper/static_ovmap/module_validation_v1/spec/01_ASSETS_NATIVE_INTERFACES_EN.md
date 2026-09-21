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
