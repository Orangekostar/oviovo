# Final Codex execution instruction — minimal instance repair and semantic rereading

## 0. Mandate, source pin and deliverable

Implement, run, evaluate, audit once, render the prescribed tables, and publish the
actual code and results. Do not stop after another plan or synthetic smoke test.
Use this file with IMPLEMENTATION_CONTRACTS.md, EVALUATION_AND_SELECTION.md and
PROTOCOL_SPEC.json. Numeric settings in the JSON are authoritative. They are new
fixed exploratory choices unless explicitly called inherited; none is claimed to
be an empirically optimal value.

Repository: `Orangekostar/oviovo`.
Base commit: `247e1e9782d43e882589bd9ab0015d513c200e49`.
New branch: `research/ovimap-minimal-instance-repair-v1`.
Parent root: `/mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001`.
Its known physical storage hint is
`/home/ww/ovimap-evidence-exploration-v1-storage/attempt_001`.
New logical root: `/mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001`.

Create an isolated worktree from the pinned commit. An already-existing matching
new-task worktree may be resumed. Preserve the original worktree, old results,
parent symlinks, model assets and successful measurements. Never force-push,
reset an unrelated branch, delete another task, or copy scan/weight data into Git.
Use an explicit `--storage-root` for this task when the shared filesystem is full;
keep a recorded path mapping or a new-task-only symlink. Validate only consumed
files and new outputs, not every file in the historical project.

The research objective is one unified method that preserves all five Replica
metrics of G1 v2 and all five CF18 metrics of G1 v2, while CF18 APall exceeds D2 and
AP50 reaches D2. This is a selection target, not a promised outcome. All scenes
have prior exposure. Do not claim independent confirmation, online operation or
field-wide novelty solely from these experiments. Deployment is N0_UNCHANGED.

## 1. Read and bind the actual data before modifying the code

Read SOURCE_EVIDENCE.md and inspect the corresponding production functions. At
minimum inspect:

| File under src/static_ovmap/ | Required understanding |
|---|---|
| recovery_wave2/recovery_registry.py | K versus full raw R; original active-owner exclusion |
| module_validation/native_capture.py | capture type, pre-insertion panoptic, camera/depth and surface arrays |
| module_validation/entity_hypotheses.py | existing mesh edges, leaf data and old projection/hypothesis machinery |
| runtime_parity/runner.py, views.py, kernels.py | actual common-input binding, R2 view construction and full-mesh ray geometry |
| cvpr_compact/area_fallback.py, region_worker.py | exact successful/fallback pooling and cosine classification |
| backbone_wave1/readouts.py | unchanged D2 probabilities and missing-source behavior |
| evidence_exploration/acquisition.py, anyup_adapter.py, timing.py | ordinary AnyUp operator, chunking and current timer boundaries |
| module_validation/evaluation.py | GeometryIdentity vs instance partition; valid PredictionPayload branches |
| cvpr_compact/outputs.py | why old append-only helper cannot implement new assignments |
| m2_reviewer_study/evaluation.py, composition_study/evaluation.py | owner registry, partition-root mask cache, area ranks and target loading |
| evidence_exploration/analysis.py, diagnostic_details.py | class-agnostic IoU, scorer ignores, score-entry ties and unique-GT diagnostics |

Read the parent study_result_store/result_store, source_binding, v2 source receipts,
per-scene contexts, G1 source and manifest, original D2 and G1 payloads, and the
published 234/18 exploration report. Locate consumed files through those manifests;
server path strings are clues, not existence proofs. Trace the parent reference
back to actual v2. Do not call a parent binder using a spoofed task ID or mutate
its authoritative spec/freezes. Implement a narrow new binder.

Import exactly EV00_D2→IR00_D2 and EV01_G1_V2→IR01_G1. Verify both payload and
scoring identities, not only rounded table numbers. Read frozen N/Q/F scores,
D2 temperatures, FC weights/text/projection head, current AnyUp source/checkpoint,
projection arrays and camera conventions. The current key source pin for AnyUp
is in PROTOCOL_SPEC.json; use the same paper checkpoint, not multi_backbone.

For each scene establish two separate objects:
1. Prediction inputs: xyz/faces/raw, painted Native owner map b, D2 labels and
   probabilities, original G1 payload and recovered evidence, RGB-D and 2D predictions.
2. Evaluation inputs: frozen point projection, labels and GT eligibility. These
   may be used only after predictions/proposals are locked, except the required
   official-rank export adapter. They must not drive view or operation selection.

The current capture type is OVIMAP_NATIVE_CAPTURE, not necessarily a synthetic
NativeScenePack. Resolve paths relative to capture_manifest.parent. Read stored
panoptic PNG with exact integer IDs; never treat a visualization palette as IDs.
Check the actual capture producer/caller once: the evidence must be the saved
pre-insertion 2D frontend raster, not global_owner_path or a projection of the
final 3D partition. Its labels are local to each frame. No cross-frame equality
of numerical 2D IDs is permitted. RGB/depth are already aligned; do not repeat
warping, depth unit conversion, image flipping or axis transformations.

If a required consumed raster is genuinely missing, try its bound original source
path and identity-preserving relocation. Do not regenerate segmentation, fill it
with zero, or skip the frame to obtain a complete pool. Continue independent arms
where possible and publish a precise dependency block. A present all-zero raster
is valid unknown evidence and may produce a real no-op. These cases are distinct.

## 2. Establish the diagnostic opportunity before expensive recognition

On all 26 scenes, first bind the original candidate universe and generate the
GT-free geometry/observation proposal library as specified below. Then execute the
evaluation-only support diagnostic from EVALUATION_AND_SELECTION.md:
- actual residual K_i, full raw R_i, and unchanged incumbent supports;
- candidate attachment/group supports from the GT-free library;
- optional relaxed maximum matching of that fixed candidate set to eligible GT.

Record current residual insufficiency separately from full-raw insufficiency,
clipping, duplicate/absorbed objects, and incumbent semantic errors. Do not infer
one from another. The preceding 131/132 CF18 figure concerns actual recovery
supports, not all original raw objects or the whole map.

The diagnostic can guide the written interpretation and work scheduling, but not
per-scene activation, candidate filtering or threshold selection. This task has a
fixed nine-arm slate. Even a no-op gets a genuine content-equal output/alias;
negative opportunity does not authorize inventing a different method halfway
through. Expensive union recognition runs only for union hypotheses selected by
at least one of IR04/IR05, using GT-free scores.

## 3. Build a reusable observation table on the same surface

Implement the 32-frame, pose-diverse, proposal/verification split and label-independent
pixel-to-surface observation map in IMPLEMENTATION_CONTRACTS.md. Reuse camera-ray
math, original mesh validation and nondegenerate-triangle indexing. Do not reuse
an old raw-owner-only raster as the new pixel-to-source-row map.

Each selected frame uses the full predicted mesh for nearest-hit occlusion and
the original measured depth tolerance. Save compact representative source-row IDs,
validity and integer panoptic IDs, or stream them into per-unit counts. Do not store
huge dense AnyUp outputs or attention matrices. One shared table serves all arms.
The query itself must not depend on proposed owner assignments.

The frame-role split is fixed from poses before reading mask categories or scores.
Use different pose bins for proposal and verification. This is disjoint-view
verification within a known sequence, not independent data or online evidence.
The same bank may subsequently be used for semantic classification AFTER structural
accept/reject decisions are locked; it may not regenerate the proposal library.

## 4. Implement the three controlled method pairs

### Pair A: fragment role / attachment
- IR02_NEAREST_ATTACH is a real nearest-surface incumbent attachment control.
- IR03_EVIDENCE_ATTACH uses the same eligible target universe and adds proposal-bank
  same-object and separation evidence plus the fixed edit penalty.
- Neither merges two incumbents, moves an incumbent source row, or changes the
  incumbent's D2 class. They may add residual rows to its support.

### Pair B: full-object groups / held-view verification
- Generate one bounded positive-consensus library from proposal views.
- IR04_DIRECT_GROUP applies it without the new held-view verification.
- IR05_VERIFIED_REPAIR tests the SAME candidates on the verification bank with
  separation evidence and the specified edit penalty. No validation-guided new
  proposals or parameter adjustment are allowed.
- Both may group 2–4 units, with at most one incumbent. Candidate conflicts are
  resolved by the same frozen proposal order and a single disjoint application pass.
- Attachment-containing groups inherit the host D2 class. Residual-only unions
  obtain one FC classification from a newly identified union observation, shared
  between the arms whenever support/view/model identities match.

### Pair C: incumbent semantics / boundary stability
- Select at most 16 original incumbents by original D2 margin, not GT or outcome.
- IR06_ANYUP_REREAD applies ordinary frozen AnyUp to full-region views with a simple
  aggregate margin-over-old-class rule.
- IR07_BOUNDARY_STABLE uses the same full views and candidate class, plus the core
  and frontend-intersection masks and the exact stability tests. It does not alter
  the 3D support. There is no new owner/depth attention factor.
- This is a test on INCUMBENTS, not a relabeling of the previous recovery-only
  EV05/EV06 experiments. New region vectors must be computed for the true masks.

### Fixed combination
IR08 uses IR05's structure and IR07's decisions for original incumbents. Decisions
are based on the original incumbent support even if it receives added fragments;
this is an explicit core-anchored classification rule. Apply the host's accepted
class uniformly to the enlarged host. Novel residual-only unions keep their own
union FC class. No rematching by integer ID, no fresh combination parameter sweep.

## 5. Recognition and model work

All model weights, FC resolutions/normalization, text prototypes, label ordering,
original FP32 settings, D2 weights and temperatures remain fixed. Do not run new
CropFormer, SAM, N/Q extraction, training, calibration or text-template search.
Use one FC/AnyUp worker and the inherited environment; create no duplicate model
copy per object. Ensure the actual parent model keys match before using caches.

New FC content work is limited to selected observer frames and masks needed by the
new unions or selected incumbents. The theoretical scientific ceiling is 26×32
unique new FC frame inputs and 26×32 AnyUp Q/K computations; reuse should normally
lower this. Pilots may add at most two FC inputs and three AnyUp Q/K computations,
with the extra original AnyUp invocation used only if wrapper neutrality must be
re-established. Cold calls have separate counters, not this scientific cache cap.

Use new content keys for geometry support, selected frame, target mask, representation,
model/text, output grid and aggregation rule. Changing grouping/semantic support
cannot reuse an old object's vector solely because its ID is preserved. Full-frame
FC dense tensors are reusable by content; new regions need new pooling. Record
which old caches were consumed and actual new computation. Release transient fine
features and attention by frame/chunk. Use the inherited 256→128→64 query chunk
sequence only on a true OOM; preserve precision and grid.

A new region with no qualifying view or an intrinsically unusable representation
causes that specific proposed operation/relabel to KEEP the parent result with a
reason. Do not shrink/dilate masks or search a better-scoring view. Missing weights,
corrupt input, incomplete computation or exhausted OOM are resource/execution
blocks, not valid negative recognition. Preserve failures and costs.

## 6. New partition export and exact evaluation

Implement a separate structural output builder; do not weaken cvpr_compact's old
construct_output() assertions globally. Reuse PredictionPayload with branch COMBO
and the unchanged GeometryIdentity. XYZ, faces, TSDF and nearest/matched projection
remain fixed; owner membership is new and has its own digest. One positive owner
has exactly one class. Unknown/unowned rows obey the parent contract.

Do not use a different prediction for AP and mIoU. Build a single mutually exclusive
partition, then recompute official current-class area ranks. No confidence rank,
no duplicated union plus its constituent instances, no GT-based final arbitration.

SceneEvaluator fixes its owner masks at construction and writes owner-ID filenames
under shared_masks/scene. Therefore instantiate it against each ACTUAL candidate
payload and an owner registry covering every positive output owner. Use a
partition-specific output root keyed by the full owner-array content, not just
scene/method or old integer owner names. An old root must not overwrite or silently
reuse a different mask. The original target coordinates/projection and label tables
are unchanged. Reuse locked scoring only if owner arrays, semantic arrays, ranks,
GT/evaluator protocol and target projection all match.

Run the released whole-dataset evaluator on the exact ordered cohorts, including
all valid GT. Do not average scene AP. Keep APall's loaded .50 through .90 vector,
AP25, mIoU and mAcc. Collect new/lost unique GT matches separately from definite and
ambiguous score entries. A support can improve without introducing a new TP50;
report all nine threshold effects, not just one count.

## 7. Execution order and CLI to deliver

Create these interfaces under src/static_ovmap/minimal_instance_repair/ (files may
be consolidated, responsibilities may not):
`binding`, `observations`, `support_units`, `proposals`, `verification`, `recognition`,
`reread`, `outputs`, `evaluation`, `diagnostics`, `selection`, `timing`, `reporting`.
Reuse existing primitives and IO/evaluator utilities after checking their contracts.
Do not invoke the old large hypothesis pipeline unchanged: it uses different
leaves, frame selection and 0.05-m point-projection settings.

Copy this specification verbatim into
`configs/static_ovmap/minimal_instance_repair_v1.json` and archive this instruction
package under `docs/paper/static_ovmap/minimal_instance_repair_v1/spec/`.
Deliver `scripts/evaluation/run_ovimap_minimal_instance_repair.py` accepting:
`--spec`, `--parent-root`, `--parent-reference`, `--output-root`, `--storage-root`,
`--gpu`, `--path-map` (JSON path), `--phase`, and `--resume`.

Phases in `all`: bind → observe → propose → diagnose → assets → pilot → freeze →
recognize → predict → evaluate → select → time → tables → publish.
`diagnose` is evaluation-only; no function importing its labels is allowed in
observe/propose/recognize/predict. `freeze` commits the implementation and complete
numeric spec before main recognition/results. Pilot fixes may address implementation
errors, not change thresholds according to measured AP. Existing successful pilot
leaves are reusable in the scientific pass when identities match.

Required final command, from the new worktree:
```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_minimal_instance_repair.py \
  --spec configs/static_ovmap/minimal_instance_repair_v1.json \
  --parent-root /mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001 \
  --phase all --resume
```
This command is to be implemented, not claimed to exist on the parent commit.
Resume must execute missing authorized work, not merely reprint status. Changed
leaf inputs invalidate that leaf and descendants only, never all historical maps.

## 8. Resource and validation discipline

Before the first persistent observation write, check output free space and give an
explicit storage relocation when necessary. Do not delete old runs to free space.
Use at most two CPU observation workers, three evaluation workers with at most four
BLAS threads each, four raycast threads, and one designated GPU worker. Reuse the
existing GPU lock. During paired cold timing stop this task's competing CPU/GPU
workers; observe other users' work without killing it and record remaining load.

Tests are limited to new production boundaries and one integrated pilot on office1
and scene0011_00. Cover: pose-bank separation; 2D-ID locality/unknowns; source-row
hit mapping and occlusion; KEEP identity; attachment/union disjointness; fresh union
IDs; single-label output; partition-specific evaluator; ordinary AnyUp parity;
unchanged-semantic conditions and the all-five target. Reuse prior checks where
identities match. No full-repository test suite, security fuzzing, test-count quota,
unrelated renderer builds or repeated global hashes. Tests in this package are
specification illustrations; they do not replace production integration tests.

No more than two identical automatic retries. A code failure requires a real fix
before another attempt; an OOM follows the fixed chunk policy. Record failed work
as observed; unrecorded durations remain null, not zero. Technical failures in one
method must not prevent publication of independent completed methods, but all
234/18 completion claims require full expected scientific coverage.

## 9. Research choice and time measurement

Implement the exact all-five criteria and simplicity tie-break in
EVALUATION_AND_SELECTION.md. Ordinary controls can win. Do not lower the target,
change a cohort, or enable different methods per dataset after seeing results.
Record promising tradeoffs separately; deployment remains unchanged.

Cold timing is bounded to four distinct arms: IR01, the selected nonbaseline arm
if any, then IR08/IR05/IR07/... in JSON priority until four. Eight Replica scenes,
two repeats, second order reversed. Required models/common map/NQF are resident;
G1 views/features/results, new observation tables, selected hypotheses and new
region vectors are not imported into a timed call. Recompute their required work.
Time from before G1 recovery construction to the completed final payload; stop
before hashing/parity/receipt writes. Any unavoidable view-manifest writes inside
the production call count in both baseline and candidate. Do not compare this new
series to old 18.85/19.89/22.71 seconds as a speedup denominator.

## 10. Reports, publication, and final response

Produce exactly three main research tables from one canonical result store:
1. nine complete-method rows on both cohorts (main three metrics, all five in data);
2. structural and semantic pairwise mechanism differences including geometry and
   unique matching diagnostics, not only object counts;
3. same-boundary time/memory/work and comparison provenance for measured arms.
Write compact MD/CSV/JSON and LaTeX. A preview PDF may reuse the existing renderer;
render once and inspect it, but do not rebuild an article. Keep extra per-scene,
per-class and source-row change ledgers in supplementary artifacts.

Publish:
- `docs/paper/static_ovmap/MINIMAL_REPAIR_RESULTS.md`;
- `MINIMAL_REPAIR_HANDOFF.md`, `MINIMAL_REPAIR_SELECTION.md`, `MINIMAL_REPAIR_CLAIMS.md`;
- actual code/tests/config and source attribution;
- `artifacts/static_ovmap/minimal_instance_repair_v1/` with the result store,
  source binding, pose roles, compact hypotheses/decisions, support and evaluation
  identities, pooled/scene metrics, trace summaries, costs/times, table provenance
  and ONE requirement review.

Target <50 MiB of compact Git artifacts. Large observation rasters, scans, model
weights, dense tensors and TSDF remain in shared/local storage with exact hashes
and reconstruction commands. Do not publish GT-bearing diagnostic coordinates or
licensed scan imagery unnecessarily; aggregate/private-reference diagnostics are
sufficient. Retain AnyUp and reused code licenses/attribution.

Audit requirements against actual files once, then commit and normal-push the new
branch. Compare full local HEAD with `git ls-remote origin
refs/heads/research/ovimap-minimal-instance-repair-v1`. Only exact equality authorizes
PUSH_VERIFIED in an external publication/final.json receipt. Do not recursively
commit the receipt that describes its own commit. Authentication failure must be
reported with the retry command, not a request to expose a token.

The final server response must provide the four GitHub report links, full verified
SHA, scientific coverage, exact target flags, principal positive/negative mechanism
comparisons, actual new computation and remaining blocks. Distinguish implementation
completion, scientific result, target achievement, timing and publication.
