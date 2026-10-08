# FINAL EXECUTION INSTRUCTION — SAM-V local map probe v1

## 0. Mission, authority and completion

Implement, run, analyze and publish the fixed four-scene experiment in this package.
The question is whether geometry-conditioned joint multi-view segmentation supplies
useful object support that our prior fixed-mask semantic experiments could not supply.
Use the actual G1 v2 map and current source-preserving-update lineage as inputs.

**This is a local probe, not a full benchmark campaign and not a paper-SAM-V reproduction.**
Finish all prescribed small arms even when an early result is negative. Do not expand
beyond four scenes or train any model. A scientifically negative but fully measured
study is COMPLETE_NO_PILOT_GAIN, not an implementation failure. An inaccessible
required checkpoint or failed inference is a real dependency/execution block, not
a zero mask. Publish a truthful partial handoff in that case.

Authority order: user request -> this package's explicit scope -> PROTOCOL_SPEC.json
and IMPLEMENTATION_CONTRACTS.md / EVALUATION_AND_SELECTION.md. They are intended
to agree. Do not import the previous experiment's numeric protocol or 26-scene
completion guards. Do not relax the selection target after seeing scores.

Existing helpers are reusable; old results, captures, specs, model environments,
partitions and hashes are read-only. New source and small results must be pushed to
GitHub. Never use a paper score, a mock mask, a previous owner's feature or an oracle
label to fill a missing cell.

## 1. Exact repository and worktree

Repository: `Orangekostar/oviovo`.
Source commit: `a95c24d990cea57b14bb537e9acca95b95659bda`.
New branch: `research/ovimap-samv-local-probe-v1`.
Suggested worktree: `/mnt/shared/ww/ovimap-samv-local-probe-v1/worktree`.
Parent: `/mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001`.
Output: `/mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001`.

Locate the existing clone, confirm its origin, fetch the exact source commit and
create a new worktree. Do not reset or clean another worktree. If this task branch
already exists, verify its ancestry/spec and resume its own artifacts. A valid
`--storage-root` and explicit `--path-map` may relocate files without changing bytes.
Keep at least 30 GiB available before downloading assets; never delete old runs to
make space. Read actual parent publication and stores rather than assuming the
server paths still resolve. If the publication proof is absent but the committed
producer/artifact identities and actual files can establish the same lineage,
record that route explicitly; do not fabricate a receipt.

## 2. Required reading before implementation decisions

Read SOURCE_EVIDENCE.md and these files at the pinned revisions:

Our repo:
- source_preserving_update/binding.py: narrow scene and actual N/Q/F/D2/G1 binding.
- source_preserving_update/outputs.py: why the previous semantic-only adapter is
  inappropriate for changing owner partitions.
- minimal_instance_repair/observations.py: SourceRowProjector, pose_banks,
  representative_rows, support_mask; actual depth units and source-row mapping.
- minimal_instance_repair/outputs.py: whole-unit/raw-zero restrictions NOT carried
  over as assertions in the new local-refinement protocol.
- minimal_instance_repair/evaluation.py and m2_reviewer_study/evaluation.py:
  partition-specific masks, official class-area ranks and ordered pooling.
- cvpr_compact/area_fallback.py and region_worker.py: original frozen FC model,
  full-resolution signed masks and v2 fallback.
- source_preserving_update/decisions.py, source_preserving_update/evaluation.py,
  SOURCE_UPDATE_RESULTS.md and SOURCE_UPDATE_HANDOFF.md.

SAM-V @ `33fab24c1d0d21ac56f4e471abf30c6de3e7018b`:
- README.md, requirements.txt, .gitmodules, LICENSE/NOTICE as present.
- model/sam_vggt_model.py, utils/checkpoint.py.
- benchmarks/compare_baseline_sam2.py, especially infer_samvggt/infer_sam2.
- demos/web/inference.py for original-image coordinate conversion only.
- pinned sam2/build_sam.py and sam2_video_predictor.py.

Record an evidence-to-code mapping with actual function names in the handoff.
**Do not run the paper comparison CLI:** it samples prompts and visibility pools
from GT and clamps frame_count to 16–32. We use 6 frames and map-generated points.
Never invoke mock_predict or the every-object proposal/NMS pipeline in this task.

## 3. Implement the new study surface

Create `src/static_ovmap/samv_local_probe/` with equivalent responsibilities:
- binding.py: narrow four-scene immutable input and asset binding.
- query_plan.py: target pools, automatic prompts, fixed view groups and edit domains.
- samv_worker.py / sam2_worker.py: isolated real pretrained model inference.
- lifting.py: visibility-normalized source-row evidence and simultaneous arbitration.
- readout.py: two fixed paired FC views and matched semantic decisions.
- outputs.py: actual row-wise partition builder, not the old whole-unit builder.
- evaluation.py / diagnostics.py: released full-map/subset evaluation and attribution.
- costs.py: separate scientific-cost records and the eight segmentation microtimings.
- reporting.py / runner.py: resumable dependency graph, tables and handoff.

Implement `scripts/evaluation/run_ovimap_samv_local_probe.py` with:
`--spec --parent-root --output-root --storage-root --path-map --gpu --phase --resume`.
Optional explicit `--samv-python --sam2-python --assets-root` are allowed.
Valid phases: bind, assets, plan, pilot, segment, lift, readout, predict, evaluate,
diagnose, cost, report, all. `all` executes every scientific/reporting phase in order.
Publication is a final explicit action after a successful full CLI or a recorded
partial failure. `all --resume` must not re-infer successful unchanged leaves.

Copy PROTOCOL_SPEC.json verbatim to
`configs/static_ovmap/samv_local_probe_v1.json` and this package's spec documents to
`docs/paper/static_ovmap/samv_local_probe_v1/spec/`. Implement efficient functions,
not a large generic experiment framework. Reuse loaders/scorers without invoking
their old full-cohort bind, freeze, pilot or run orchestration.

## 4. Input binding and model assets

Bind four scenes only: office1, room0, scene0011_00, scene0050_00. Read parent SU00_D2
and SU01_G1 payloads plus per-scene released scorer receipts. Verify actual surface
xyz/faces/TSDF, G1 ownership/classes, frozen target correspondence, N/Q/F sources,
text class order, D2 probabilities and capture RGB/depth/cameras. Reconstruct the
D2 labels once as an integration check. Hash consumed files once, memoize immutable
identities, and invalidate only descendants of changed new-task inputs.

The parent's source binding points to the minimal-instance-repair parent. Reuse its
32-view source-row observation rasters only when geometry, camera, depth and observer
operator are identical. If absent, recompute at most these four scenes × 32 selected
poses with SourceRowProjector. Do not rebuild maps or rerun CropFormer. Do not use
GT visibility, GT masks or target labels to plan queries. Frozen evaluation nearest/
matched arrays are for released evaluation/ranking only, never for query selection.

SAM-V requires three asset groups: frozen SAM ViT-H, frozen VGGT-1B and the released
stage-2 trained modules. Reuse matching local author weights first. Otherwise fetch
only the specified files, not the training datasets or all submodules/baselines.
Resolve the author HF revision and download hash once before freeze. The assistant
verified the README release pointer but did NOT download/verify the HF stage-2 file.
If inaccessible, attempt the stated author endpoint and local exact-file search;
report the actual failure, do not substitute stage-1, another repository or random
weights. Inspect and record licenses without republishing weights.

SAM-V repo and required gitlinks are pinned in the spec. Only initialize vggt,
patched sam-hq and sam2. Do not recursively install ODIN/Point-SAM/PanSt3R/apex.
Use the author's partial-checkpoint verifier; every trainable fusion/decoder tensor
must be supplied with correct shape. Verify frozen encoder key coverage separately;
do not accept a shape-filtered non-strict load as success.

Use separate Python processes/environments:
- SAM-V: Python 3.10, author requirements (torch 2.3.1, torchvision 0.18.1,
  timm 1.0.22). Only its submodule paths are added before imports.
- SAM2: Python 3.10, torch 2.5.1 / torchvision 0.20.1 (matching CUDA wheels),
  numpy 1.26.4, hydra-core 1.3.2, iopath 0.1.10; install the pinned SAM2 locally.
  Never upgrade the SAM-V/controller/FC environment to satisfy SAM2's setup.py.
- Geometry/released scoring: inherited controller.
- FC: inherited FC worker with unchanged model/text/preprocessing.

SAM2 baseline is **SAM2.1 Hiera-L**, not the compare script's mismatched B+ default.
Use the config/checkpoint pair specified in JSON. Set SAM2_BUILD_CUDA=0 only in its
isolated environment; the exact hydra_overrides_extra strings are in JSON. For reproducibility without a
CUDA connected-components build, construct with apply_postprocessing=False and
explicit overrides enabling dynamic_multimask_via_stability=true,
dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98,
binarize_mask_from_pts_for_mem_enc=true, fill_hole_area=0. Use vos_optimized=False.
This is an explicitly stated SAM2 baseline profile, NOT an exact paper Table-1
reproduction. Do not silently change it or claim an official published score.

## 5. Query planning and freeze

Follow IMPLEMENTATION_CONTRACTS.md exactly. Up to 8 targets/scene are selected from
actual G1 outputs: nominally 4 uncertain D2 incumbents and 4 G1-recovered owners.
Shortages are filled deterministically from the other pool. Old predicted masks
must be promptable in at least two of the fixed representative views. No model
success, GT match, AP contribution or dataset-specific class whitelist is allowed
in target selection. Save the full eligible/rejected inventory and exclusion reason.
A scene with fewer targets remains in evaluation; a model failure never triggers
replacement by the next easier object.

Each target receives up to 3 positive points in ONE old-mask anchor frame, and the
same 6 chronological RGB views and prompts go to both segmentors. The two mandatory
semantic frames are chosen before segmentation. Keep absent/occluded context frames
when selected by the fixed pose-diversity rule; do not consult GT to balance them.

Plan the source-row edit domain before segmentation. This protocol permits local
row-wise ownership correction on a fixed surface, including selected old support
and visible unowned surface; it is not restricted to unions of old raw owners.
Unselected interiors and prompt rows are protected. Full rules are in the contracts.

Run a real engineering pilot on the first locked target in office1 and scene0011_00.
If one has no queryable target, use the other prelocked scene in that same cohort
for the pilot only; do not change its evaluated targets or replace a scene. If an
entire cohort has no queryable target, record NO_QUERIABLE_TARGET, keep its map outputs
and leave its segmentation timing undefined. Do not fabricate a model pilot. The
two-window/eight-timing count applies when both prescribed cohort pilots exist.
Check weights, coordinate transforms, mask shapes, finite logits, real SAM2 forward/
reverse behavior and lifter/output invariants. Do NOT calculate AP to choose the
resource profile. Default is 6 frames, bf16 autocast, SAM encoder microbatch 1, group
batch 1. Only a genuine CUDA OOM before scientific evaluation can trigger a single
GLOBAL switch to 4 frames. Replan ALL targets/arms by the same rule, archive the
failed 6-frame pilot and freeze the 4-frame profile. No per-scene frame count,
resolution or dtype adaptation. If 4 also cannot execute, record a resource block.
Other I/O problems may be fixed without changing the algorithm; no endless retries.

Commit the complete implementation/spec after narrow tests and successful pilot,
record the exact commit and effective resource profile, then acquire the remaining
science. Successful pilot leaves may be reused if their full identity matches.
Real numerical/bug fixes get a corrective freeze and scoped rerun, not a result-
conditioned threshold change.

## 6. Segmentation execution

One GPU worker at a time. Do not hold SAM-V, SAM2 and FC simultaneously in VRAM.
Use released inference primitives, not a rewritten attention/decoder. For SAM-V,
call its forward with raw canonical RGB floats 0..255, point coordinates in the
1024-square frame, correct frame indices, multimask_output=False, visualize=False.
Split low-resolution panorama logits by view BEFORE interpolation. Restore each
view to 1024 then original captured H/W with bilinear align_corners=False, then
threshold logit >= 0. See the coordinate contract.

For SAM2, add the identical anchor-frame points, propagate forward then reverse
from that anchor, and reset/discard the independent target state. The window order
is the same original chronological order as SAM-V's input. Use bf16 autocast and
inference_mode; no extra points, reverse-time input reorder, box prompts, best-mask
search, or object interaction across independent queries. Missing emitted frames
are recorded explicitly; unexpected execution failures are not zero predictions.

Save raw per-view masks, available logits/scores, prompt adherence and all identities.
A successful empty prediction is a valid model result. When anchor prompts fail,
record NO_ANCHOR_SUPPORT and preserve that target's old output for map intervention;
retain the failed masks in raw-mask diagnostics. Do not call mock_predict.

SV02, SV04 and SV05 consume the exact SAME SAM-V predictions. Do not run SAM-V three
times. Exact task-local mask cache reuse is allowed for resume. If image embeddings
are cached, SAM's key includes actual canonical pixels/preprocessing/precision;
VGGT/fused keys include the ENTIRE ordered view group and model settings. A changed
view group cannot reuse per-image geometry features. Do not add a new accelerated
encode/decode path in this first probe unless its equivalence is actually checked;
the default authoritative path is unchanged full forward per target.

## 7. Lifting, FC controls and actual outputs

Construct SV01/SV02 independently using one shared visibility-normalized lifter.
Do not crop new masks to old owner supports. Do not use VGGT-predicted depths in
place of the bound measured depth. Do not make unobserved rows negative. Use the
same simultaneous conflict policy for both segmentors. Structural labels inherit
G1 for all retained owners. No SAM predicted-IoU score is substituted for released
class-area ranking. Write all changes, donors, retained prompts and untouched rows.

Construct SV03/SV04 on the unmodified G1 partition using the same two fixed frames,
ordinary frozen FC and the same update rule. Their common success domain requires
both mask sources to be usable in both frames. Failure is KEEP for BOTH semantic
controls on that object, not a hidden one-sided fallback; raw availability is still
reported. Run the existing v2 area fallback for empty dense support, not AnyUp.

Construct SV05 with exact SV02 ownership and exact SV04 owner-to-class decisions.
Do not choose a new label after viewing the repaired mask. This is the controlled
geometry × semantic combination, not an optimized combined re-estimation.
Build `PredictionPayload(branch='COMBO')` with unchanged GeometryIdentity, actual
owner_ids/semantic_labels and recomputed native_ranks. Use a new local output builder;
calling the old construct_partition() would cancel or forbid the intended experiment.

## 8. Scoring, diagnoses and research decision

Generate all 24 main scene-method rows and the 4 D2 reference rows. Reuse actual
baseline predictions, not full-dataset pooled numbers. Compute 12 main two-scene
pools plus 2 D2 reference pools using the original released evaluator. Keep the
actual complete map for each selected scene; never evaluate only repaired objects
as the main metric. One exclusive partition produces both AP and mIoU.

Follow EVALUATION_AND_SELECTION.md for masks, ranks, class-aware traces,
class-agnostic matching, localized diagnostics and promotion. Annotation reading
begins only after predictions and candidate selection are locked. No two-scene
result can be labeled Replica8 or CF18, and no partial set may fill a complete pool.

Research choice is uniform across the four scenes. Primary candidates SV02/SV04/SV05
must preserve all five metrics against SV00 in each probe cohort, cross the exact
D2 CF-probe APall/AP50 conditions, and beat their relevant simple control before
receiving a positive mechanism claim. Differentiate PILOT_TARGET_MET from the
individual segmentation/semantic evidence flags. Never change deployment from N0.
No automatic 26-scene expansion even if the pilot is positive; publish the concrete
next-stage recommendation and stop.

## 9. Costs without another large timing campaign

Log all actual elapsed work by stage, including planning/observation, model loading,
SAM/VGGT/SAM2 encoder calls, target decoding, source-row lifting, FC, payload creation,
scoring and exports. Track scientific first computations and resume hits separately.
New segmentation IS new inference; do not report the task as zero new inference.
FC may reuse exact parent dense tensors, but new masks still require new pooling.

Run exactly eight extra paired segmentation-window timing calls: the two fixed
pilot windows × SAM2/SAM-V × 2 repetitions. Models are resident, feature/result
caches are off, canonical images are read in the timed call, and restored raw masks
mark the end. Synchronize CUDA, record peak allocated/reserved and loading separately.
Round 2 reverses method/window order. Check each timed raw binary mask against its
locked scientific counterpart outside the timer. A parity mismatch is investigated
with a narrow regression; do not publish a comparable timing for a different output.
OS page cache is uncontrolled and disclosed.
These timings measure segmentation windows, not full map update latency. Do NOT
compare them numerically with G1's 19.89 seconds/scene or the paper's timing table.
No new standalone end-to-end benchmark is authorized in this probe.

## 10. Narrow validation, not excessive testing

Implement at most roughly 12 focused test cases for new boundaries: auto prompts,
no GT planning, view ordering, panorama split/coordinate restore, visibility
normalization, no-evidence KEEP, simultaneous conflict/core protection, uniform
classes, per-partition mask registry, paired semantic availability, complete subset
pooling and resume identity. Synthetic helpers in reference/ can be reused.

Run the targeted test file and compile changed modules. Perform the two genuine
engineering pilots and one baseline scoring parity check per cohort. That is enough
unless a specific failure justifies a regression test. Do not run old full-repo
suites, repeated global hashes, adversarial fuzzing, deployment audits, generic
security testing or 26-scene old pipelines. A count of passed checks is not the
scientific result. Do not create thousands of trivial assertions as a success KPI.

## 11. Deliverables and publication

Generate from one canonical result store:
- `SAMV_PROBE_RESULTS.md`, `SAMV_PROBE_HANDOFF.md`, `SAMV_PROBE_SELECTION.md`,
  `SAMV_PROBE_CLAIMS.md` under `docs/paper/static_ovmap/`.
- Three table families in MD/CSV/JSON/LaTeX: complete probe metrics; mechanism
  diagnostics; observed cost and clearly scoped eight-call timings.
- Per-scene locked target/query manifests, model/preprocess identities, compact
  masks or references, actual edits/class decisions, scorer identities and traces.
- At least one fixed-case visual contact sheet per scene (all selected objects in
  miniature, not only winners), plus an explicitly labeled success/failure pair
  when such cases exist. Render masks directly, not generated illustrations.
- Actual status counts, thresholds/resource profile, asset manifest, run commands,
  failed-attempt costs (null where unmeasured), and unresolved prerequisites.

Small code, JSON/NPZ masks, decisions, metrics and tables go into
`artifacts/static_ovmap/samv_local_probe_v1/` (target under 30 MiB; compact masks rather
than source images where permissions are uncertain). Large RGB/depth, dense maps,
weights and full raster tensors remain on shared storage with identities. Do not
upload credentials, licensed scans or model weights. Publication is not complete
if GitHub contains only absolute server links and no actual compact results.

Run the exact reproduction CLI to completion once:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_samv_local_probe.py \
  --spec configs/static_ovmap/samv_local_probe_v1.json \
  --parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001 \
  --phase all --resume
```

Record its real exit code. A complete negative science run exits 0; incomplete
required model/evaluation execution exits nonzero and is reported as partial.
Stage and commit only this task's intended files; use normal `git push -u origin
research/ovimap-samv-local-probe-v1`. Verify `git rev-parse HEAD` equals the exact
branch SHA returned by `git ls-remote origin refs/heads/research/ovimap-samv-local-probe-v1`.
Write the observed hashes and PUSH_VERIFIED to an external
`<output_root>/publication/final.json`; avoid a self-referential commit-hash loop.
If an auth/network error prevents pushing, say PUSH_FAILED with the real error,
not that publication succeeded. Preserve dirty unrelated files, do not force-push.

Final response to the user: compact six-arm accuracy table, D2 reference, true target
coverage, evidence flags, measured cost scope, final uniform research choice,
remaining limitations, and the actual final commit/report links. Distinguish
implementation completion, inference completion, pilot result and publication.
