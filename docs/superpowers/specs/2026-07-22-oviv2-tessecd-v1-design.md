# OVIV2 TESSE-CD v1 Design

## Objective

Produce the protocol-valid TESSE-CD results needed to replace the numerical
placeholders in the OVIV2 abstract. The evaluated method is the OVIV2 Stage3
online mapper derived from commit `47962fbd9f363c0696cc5016f8ab42f83a3bf7e5`.
The Replica route3 surface-observation post-processing result and ScanNet200
Stage4 configuration are outside this experiment.

The final run is named `oviv2-tessecd-v1` and is accepted only when its source
commit is clean and its configuration, model weights, frontend caches, dense
semantic caches, TESSE-CD inputs, schedule, target package, and evaluator are
hash-bound in the result provenance.

## Evidence Boundary

The main TESSE-CD result supports these claims:

- OVIV2 reduces stale current-map state, measured by current mIoU and ghost
  rate against protocol-valid non-oracle baselines.
- OVIV2 recovers revealed background sooner, measured by background F@5 cm and
  recovery frames.
- Official TESSE-CD object, dynamic-object, and change F1 are reported per
  scene when the official evaluator path is complete.

The statement that occlusion does not cause false removal requires a separate
prediction-independent occlusion stress test. It is not inferred from the main
T2 metrics alone.

Unknown results remain unfilled until strict finalization completes. No metric
or baseline improvement is estimated from smoke tests, partial runs, or unit
tests.

## Selected Approach

Generate OVIV2-native, frame-local YOLO-World plus MobileSAM object caches for
the official Apartment and Office vocabularies, and generate RADSeg dense
probability caches for the same frozen vocabularies. This preserves the Stage3
two-branch design: object observations supply persistent entity evidence while
dense probabilities preserve surface semantics and background recovery.

The existing TESSE-CD `shared_full_v1` caches are not used as the OVIV2
semantic frontend because their frozen class list contains only `item`. They
may remain inputs to protocol-valid baselines.

## Repository Isolation

Implementation and runs use an isolated worktree based on the approved Stage3
base. The current route3 and Stage4 working-tree changes are not copied into the
implementation branch. Protocol code currently living as untracked files in
the baseline worktree is ported as reviewed source and tests, not inherited via
the dirty working tree.

The design document is the only file committed before the isolated
implementation worktree is created. All subsequent commits must keep the
method lineage traceable to `47962fb`.

## Components

### TESSE-CD Dataset Adapter

Add a read-only dataset adapter for the exported TESSE-CD RGB-D views. It
validates:

- scene is exactly `apartment` or `office`;
- image counts are 1745 and 4346 respectively;
- RGB is 720 by 480 and depth is uint16 millimeters;
- intrinsics match the frozen camera manifest;
- every trajectory row is a finite 4 by 4 camera-to-world transform;
- timestamps are strictly increasing and match the causal schedule;
- the RGB-D export manifest and source database hashes are unchanged.

Frames are processed at stride 1. Dataset index, source frame index, and cache
index are identical. Runtime timestamps come from `timestamps.csv`; they are
not replaced by frame numbers.

### Frozen Scene Vocabularies

Create one ordered vocabulary per scene from the official Hydra label-space
files and the common-v2 alias map. Semantic ID 0 remains unknown. Object and
structure names are kept in official ID order, with aliases used only for
normalizing frontend labels during evaluation.

The vocabulary files record the official label-space hash and alias-map hash.
Apartment and Office may have different class lists, but all branches within a
scene consume the same ordered list.

### Object Frontend Cache

Extend the existing Stage3 frontend orchestration pattern to TESSE-CD. For each
frame, YOLO-World uses the frozen scene vocabulary and MobileSAM produces the
object masks. The cache retains the fields consumed by
`CachedFrontendAdapter`, including masks, boxes, confidences, class IDs, class
names, and image features.

The frontend manifest binds:

- scene, frame count, source frame IDs, and image shape;
- ordered vocabulary and aliases;
- frontend code commit and algorithm configuration;
- YOLO, MobileSAM, and CLIP model hashes;
- every per-frame cache hash and their prefix digest.

A completed manifest is published atomically only after all frames validate.

### RADSeg Dense Cache

Generalize the Stage3 dense-cache precomputation path to accept the TESSE-CD
dataset adapter. RADSeg processes each RGB frame independently with the frozen
scene vocabulary. The cache keeps the existing Stage3 top-k probability,
entropy, margin, model provenance, and per-frame checksum contracts.

Dense inference never receives change annotations, target arrays, future
frames, or evaluator labels. A cache is reusable across threshold sweeps
because the tuned lifecycle parameters affect only online map maintenance.

### Online Runner

Add a TESSE-CD runner that constructs `Oviv2Runtime` using the Stage3 runtime,
signed visibility, reversible ownership, and dense semantic integration. It
does not call Replica route3 composition or any ScanNet200 Stage4 code.

For every source frame, the runner performs this order:

1. load RGB-D, pose, intrinsics, and true timestamp;
2. load frame-local object and dense semantic caches;
3. call `Oviv2Runtime.process_frame` once;
4. if the frame is a scheduled checkpoint, commit an immutable runtime
   snapshot immediately;
5. export the checkpoint into the neutral temporal artifact format;
6. continue to the next frame.

For checkpoint frame `t`, provenance records
`consumed_through_frame_exclusive = t + 1`. Snapshot timestamp and event IDs
must exactly match the common-v2 schedule. The runner rejects missing,
duplicate, out-of-order, or extra checkpoints.

### OVIV2 Checkpoint Export

The exporter derives the current semantic surface from the immutable Stage3
snapshot. Current ownership is authoritative for supported entity voxels;
uncertainty-aware dense semantics supply labels where ownership has been
released or is absent. Historical entity records do not automatically appear
in the current map.

Each neutral snapshot and its entity metadata are written atomically and
hash-bound in a scene temporal index. Evaluation reads only these frozen
artifacts and cannot inspect the mutable runtime.

### Common-v2 Evaluation and Finalization

Port the complete common-v2 target, semantic crosswalk, evaluation, and strict
finalization contracts into the implementation branch. Extend them with an
`OVIV2` online method whose summary mode is `causal_checkpoints` and whose T2
tokens use `T2_OVIV2_*`.

The finalizer accepts one Apartment summary and one Office summary together
with byte-identical repeat summaries. It rejects:

- mismatched method, scene, mode, schedule, targets, evaluator, or source
  hashes;
- summaries containing future-state checkpoint metadata;
- dirty or inconsistent frozen-run provenance;
- a primary/repeat byte mismatch;
- missing, non-finite, or protocol-ineligible metrics.

The result binds current mIoU, ghost rate, background F@5 cm, and recovery
frames. Official TESSE-CD object, dynamic-object, and change F1 use a separate
official-evaluator finalization path but share the same frozen run identity.

## Configuration and Tuning Protocol

Start from `configs/oviv2_replica8_stage3.json` and its Stage3 base runner
configuration, replacing only dataset-, cache-, vocabulary-, schedule-, and
scene-specific paths before tuning.

Apartment is the validation scene. Only these maintenance parameters may be
selected on Apartment:

- `visibility_depth_tolerance_m`;
- `absence_negative_support`;
- `ownership_min_net_support`.

The runner must expose `ownership_min_net_support`, which already exists in
`Oviv2RuntimeConfig` but is not currently read from JSON. Detection,
association, tracking, TSDF, dense semantic, and fusion parameters remain at
their approved Stage3 values.

After Apartment selection, one configuration is frozen. Office is evaluated
held out without threshold changes. The Office repeat is a determinism audit,
not a second opportunity to select parameters.

## Frozen Run Identity

`oviv2-tessecd-v1` contains one machine-readable freeze manifest with:

- clean repository commit and parent Stage3 commit;
- normalized algorithm configuration hash;
- Apartment and Office scene configuration hashes;
- frontend and dense cache manifest hashes;
- model IDs and weight hashes;
- TESSE-CD RGB-D export, source database, trajectory, timestamp, and camera
  hashes;
- schedule, common-v2 target, label-space, alias-map, evaluator, and finalizer
  hashes;
- Python, CUDA, library, GPU, and host information;
- exact run commands and output roots.

Only path fields, scene identifiers, source manifests, and scene vocabularies
may differ between Apartment and Office. Algorithm parameters must yield the
same normalized algorithm hash.

## Determinism

Each scene is run twice from empty output directories with the same frozen
inputs. Evaluator summaries must be byte-identical. Large checkpoint artifacts
may use separate paths, but their content hashes and temporal-index records
must match exactly.

All JSON uses sorted keys, finite values, stable separators, and a trailing
newline. Checkpoint order, entity order, voxel order, and token order are
deterministic. Wall-clock timings are stored outside the byte-compared metric
summary.

## Occlusion Safety Evidence

Derive an occlusion target package without using method predictions. A target
sample is an object that is active in official ground truth, projects into the
camera frustum, and is hidden by a closer observed surface. For those samples,
measure false ownership release and retained-object recall.

Compare the full signed-visibility method against an ablation that treats every
missing object observation as absence. The abstract may claim occlusion safety
only if the full method lowers false release without invalidating its stale
state and recovery results.

## Error Handling

All preflight failures occur before an output directory is considered
complete. Partial cache or run directories lack a completed manifest and
cannot be resumed unless every existing artifact matches the requested frozen
prefix. Atomic publication and checksum validation follow the established
Stage3 patterns.

The pipeline never silently substitutes a baseline cache, a Replica cache, a
route3 output, a Stage4 configuration, an oracle semantic stream, or a partial
TESSE-CD sequence.

## Test Strategy

Development follows test-driven changes. Tests cover:

- TESSE-CD image, depth, pose, timestamp, intrinsics, and manifest validation;
- vocabulary ordering and semantic alias normalization;
- object and dense cache manifest completion and hash rejection;
- full-frame chronological processing and true timestamp propagation;
- exact causal checkpoint boundaries and future-frame exclusion;
- Stage3 snapshot-to-neutral-export semantics;
- JSON exposure of `ownership_min_net_support`;
- `OVIV2` finalizer support and `T2_OVIV2_*` token bindings;
- rejection of legacy `T2_OVIOVO_*` entries in the T2 table;
- deterministic summary generation and byte mismatch rejection;
- a small synthetic occlusion case where only visible absence reduces support.

After unit and integration tests, run a short real-data smoke prefix that
crosses the first Apartment intervention. Smoke success proves only pipeline
operation. Full metrics require complete Apartment and Office double runs.

## Acceptance Criteria

The abstract placeholders may be replaced only after all of these conditions
hold:

1. A clean `oviv2-tessecd-v1` commit and freeze manifest exist.
2. Full Apartment and Office sequences complete twice using Stage3 online
   processing and the frozen configuration.
3. Every scheduled checkpoint is causal and hash-bound.
4. Per-scene metric summaries are byte-identical across repeats.
5. Strict common-v2 and official TESSE-CD finalizers pass.
6. T2 OVIV2 tokens are verified and imported into the paper tables.
7. Improvements are computed against the best protocol-valid,
   ranking-eligible non-oracle baseline with metric direction preserved.
8. The occlusion clause is either supported by the dedicated stress test or
   removed from the abstract.
