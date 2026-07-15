# OVIV2 Voxel-First Replica Table 1 Design

**Date:** 2026-07-15
**Status:** Approved design; implementation not started

## 1. Goal

Build a runnable method named **OVIV2** whose authoritative map is a sparse voxel map and use it to produce the OVIV2 row of Table 1. The first milestone is Replica `room0` with 200 sampled frames at stride 10. After the room0 protocol is stable, the same executable and evaluator will run Replica-8 and aggregate Replica-7 by excluding room0.

The room0 run must produce all six primary measurements from one immutable voxel snapshot:

- semantic `mIoU`, `mAcc`, and `f-mIoU`;
- semantic instance `AP25` and `AP50` using the OVI-MAP/ScanNet-style protocol;
- geometry `F@5cm`.

## 2. Scope

### Included

- A voxel-first OVIV2 static mapping pipeline.
- Reuse of the existing Replica loader and validated RGB-D observation frontend.
- New OVIV2 temporal, entity, voxel fusion, ownership, snapshot, and evaluation modules.
- An OVI-MAP-aligned voxel-to-mesh evaluation path.
- Two-frame and 20-frame smoke runs before the 200-frame room0 run.
- Frozen run manifests and machine-readable result artifacts.

### Excluded From This Milestone

- ScanNet200-5, because licensed ScanNet data is not present on this machine.
- TESSE-CD dynamic experiments and 3RScan identity experiments.
- Query calibration and `NOT_FOUND` evaluation.
- Navigation and manipulation evaluation.
- Dense point-cloud state, dense semantic caches, and dense-map fusion.

ScanNet200-5 remains the second Table 1 milestone after Replica-8/7 is complete.

## 3. Hard Architectural Decisions

1. `OVIV2` is the method and artifact name. Old OVIOVO results cannot fill the OVIV2 row.
2. The sparse voxel map is authoritative. Dense point clouds are not maintained, checkpointed, queried, or fused.
3. Evaluation meshes and GT-aligned PLY files are derived artifacts. They never feed back into mapping.
4. New OVIV2 modules cannot read or mutate legacy `SystemState`, `ObjectMap`, `local_pcd`, `DenseSurfaceMap`, or legacy TSDF owner state.
5. Legacy `Patch3D` values may cross into OVIV2 only through a one-way observation adapter.
6. Geometry, semantic evidence, entity evidence, and current ownership are separate voxel layers sharing the same integer voxel address space.
7. Current ownership is derived and reversible. Releasing an owner cannot erase TSDF geometry or competing evidence.
8. Runtime mapping cannot read Replica semantic labels, instance labels, GT meshes, or future frames.

## 4. Reference Behavior From OVI-MAP

OVI-MAP uses a label-TSDF voxel map as its mapping substrate. Its evaluation does not compare a runtime dense point cloud with GT. The relevant reference path is:

1. Generate an instance mesh from the voxel/label-TSDF map.
2. Attach global instance IDs to mesh vertices.
3. Aggregate per-instance visual features and match them to a text vocabulary.
4. Attach semantic IDs to mesh vertices.
5. Project the predicted mesh onto Replica GT mesh vertices with nearest-neighbor distance below 5 cm.
6. Evaluate semantic confusion and semantic instance masks on the common GT vertex domain.

The local reference implementation is pinned at OVI-MAP commit `58a804e2d7c82ba05a489eb071aba3367301fed8`:

- mesh generation: `scripts/panoptic_mapping_.py`;
- mesh semantic post-processing: `scripts/utils/mesh_postprocess_utils.py`;
- semantic evaluation: `scripts/eval_sem_seg.py`;
- instance AP: `scripts/eval_utils.py`.

OVIV2 reproduces this evaluation definition with testable native code. It does not import OVI-MAP's ROS runtime or copy its mapping implementation.

## 5. Runtime Architecture

```text
Replica RGB-D + camera-to-world pose
                 |
        Observation Frontend
    YOLO-World + SAM2 + depth refinement
                 |
        Legacy Patch Adapter
                 |
         FrameObservation
                 |
       Local Temporal Module
                 |
            LocalTrack
                 |
        Visibility Module
                 |
       Entity Association
                 |
         Entity Registry
                 |
       Voxel Evidence Fusion
                 |
       Reversible Ownership
                 |
        Lifecycle Management
                 |
       Immutable Voxel Snapshot
                 |
       OVI-MAP-Style Evaluator
```

The module order is fixed. Visibility runs before association and fusion. Only the committer publishes a new snapshot revision.

## 6. Reused and New Modules

### Reused Through Narrow Interfaces

- `ReplicaRoom0Dataset` for RGB-D, intrinsics, and poses.
- YOLO-World and SAM2 proposal generation.
- Depth refinement and 3D patch lifting.
- Existing label aliases and the frozen Replica-41 vocabulary manifest.
- Mathematical metric definitions where they agree with the OVI-MAP protocol.

### Reimplemented for OVIV2

- observation kind and provenance normalization;
- 3-5 frame local tracking;
- voxel visibility evidence;
- active association and entity registry resolution;
- sparse TSDF block storage and projective fusion;
- semantic voxel evidence;
- multi-entity voxel evidence;
- reversible ownership;
- persistent entity state and lifecycle;
- immutable voxel snapshots and checkpoint serialization;
- voxel-to-mesh extraction and OVI-MAP-style evaluation;
- room0 and Replica multi-scene runners.

Legacy `AssociationModule`, `ObjectUpdateModule`, and stateful `SemanticMemoryModule` are not reused because their public operations depend on `ObjectMap` or `SystemState`.

## 7. Sparse Voxel Backbone

### Addressing

- Default voxel size: `0.05 m`.
- Default block side: `8` voxels.
- Global voxel key: integer `(x, y, z)`.
- Block key: floor division of the global key by 8.
- Local voxel index: modulo-8 coordinate inside the block.

Blocks are allocated on demand. All layers use the same global keys and block layout.

### GeometryVoxelLayer

Each allocated geometry voxel stores:

- normalized TSDF value;
- accumulated TSDF weight;
- accumulated RGB color and color weight;
- first and last observation frame;
- update revision.

Geometry integrates valid full-frame depth independently of instance stability. Projective fusion samples the truncation band around each valid depth measurement, computes signed projective distance, clamps it by the truncation distance, and applies a bounded weighted average. Invalid, non-finite, non-positive, or out-of-range depth is ignored.

### SemanticVoxelLayer

Surface voxels store a bounded top-k set of runtime semantic label IDs and support values. The first implementation uses `k=4`. Semantic input comes only from runtime anchors and masks. The frozen Replica-41 vocabulary and aliases normalize labels before fusion.

Structural classes such as `wall`, `floor`, and `ceiling` update this layer without creating entities. Object observations update both semantic and entity evidence after temporal acceptance and entity resolution.

### EntityEvidenceLayer

Each surface voxel stores a bounded top-k set of entity IDs with positive support, negative support, last timestamp, and evidence revision. The first implementation uses `k=4`. Competing candidates remain available after a current owner is selected.

### OwnershipLayer

Ownership stores the current entity ID, confidence, ownership epoch, and source evidence revision for each surface voxel. It is recomputed from entity evidence. Owner release changes only this layer and the entity's inverted voxel index; it cannot delete geometry, semantics, or competing entity evidence.

## 8. Observation and Entity Rules

`FrameObservation` gains an explicit kind:

- `OBJECT`: eligible for LocalTrack, entity association, entity evidence, and ownership;
- `STRUCTURE`: eligible for semantic voxel fusion but never entity creation;
- `UNKNOWN`: geometry-supporting observation with no semantic commitment.

Local tracks retain a bounded 3-5 frame observation window, voxel keys, centroid, bounds, semantic vote, and optional appearance embedding. They do not contain persistent entity IDs. A tentative track is not allowed to create entity evidence until it reaches the configured hit threshold.

Association is geometry-first. It uses voxel overlap, centroid/bounds consistency, and visibility compatibility. Runtime semantic agreement can break close ties but cannot override a geometry rejection. Association emits decisions; the registry alone allocates new entity IDs.

Persistent entities store semantic and identity features, lifecycle intervals, and a voxel-submap handle. They do not store point clouds. Per-entity geometry is obtained by resolving owned or supported voxel keys.

## 9. Static Visibility and Lifecycle Behavior

The visibility module projects previously supported surface voxels into the current camera and compares projected depth with the current depth image:

- `PRESENT`: current depth agrees within tolerance;
- `ABSENT`: valid current depth demonstrates free space behind the former surface;
- `OCCLUDED`: current depth lies in front of the former surface;
- `UNOBSERVED`: outside the image, invalid depth, or otherwise not testable.

Only `ABSENT` provides negative evidence. `OCCLUDED` and `UNOBSERVED` cannot remove ownership. Static Replica is expected to exercise mainly PRESENT/OCCLUDED/UNOBSERVED behavior, but the same contract remains valid for later dynamic experiments.

## 10. Immutable Snapshot and Persistence

The committed OVIV2 snapshot records:

- scene ID, timestamp, frame ID, schema version, and global revision;
- geometry, semantic, entity-evidence, ownership, registry, and lifecycle revisions;
- sparse block arrays for each voxel layer;
- persistent entity records and inverted voxel indexes;
- frozen configuration and vocabulary hashes.

Snapshots are written atomically. A reader rejects inconsistent schema versions, voxel sizes, array shapes, duplicate entity IDs, invalid revisions, and ownership references to absent entities. Checkpoint restore must reproduce the same evaluator output as the original in-memory snapshot.

## 11. OVI-MAP-Aligned Replica Evaluation

### Frozen Vocabulary

The evaluator uses the frozen Replica-41 vocabulary and aliases from `configs/evaluation/manifests/replica8.json`, not OVI-MAP's original Replica-51 list. OVI-MAP alignment refers to voxel-mesh extraction, 5 cm GT projection, semantic confusion, and semantic instance AP mechanics. A common frozen vocabulary is required for fair Table 1 comparison.

`ceiling`, `floor`, and `wall` participate in semantic metrics but are excluded from semantic instance AP.

### Voxel-to-Mesh Extraction

Marching Cubes extracts the zero-crossing triangle mesh from the GeometryVoxelLayer. Every mesh vertex receives:

- interpolated position and color;
- semantic ID from neighboring semantic surface voxels;
- entity ID from neighboring ownership voxels;
- ownership and semantic confidence for auditing.

The raw mesh is the only input to geometric and GT-projection evaluation. Voxel centers are not substituted for mesh vertices in the headline metrics.

### Projection to the GT Vertex Domain

For each GT mesh vertex, find the nearest predicted mesh vertex. A prediction is transferred only when squared Euclidean distance is strictly below `0.05^2 m^2`. Otherwise semantic and instance IDs remain zero. The transfer produces GT-aligned semantic and instance arrays with exactly the GT vertex count.

The optional `semantic_map_gt.ply` and `instance_map_gt.ply` files are audit renderings of these arrays on the GT mesh topology.

### Semantic Metrics

Semantic evaluation ignores GT vertices outside the frozen vocabulary. Predicted ID zero is unmatched/background. A confusion matrix is accumulated on GT vertices and yields:

- per-class IoU and accuracy;
- macro mIoU;
- macro mAcc;
- frequency-weighted mIoU.

No GT information is used to choose runtime entity labels.

### Semantic Instance AP25 and AP50

Predicted masks are groups of GT-aligned vertices sharing one nonzero entity ID. A prediction participates only when:

- its semantic class belongs to the frozen instance vocabulary;
- it contains at least 100 mapped vertices;
- the corresponding persistent entity was supported by at least two accepted views.

Prediction confidence follows the OVI-MAP official post-processing definition: mapped vertex count divided by the maximum mapped vertex count among predictions of the same semantic class. GT instances smaller than 100 vertices are ignored. AP is computed per semantic class with ScanNet-style greedy matching and then averaged over classes with valid GT. Table 1 reports AP at IoU thresholds 0.25 and 0.50. Class-agnostic AP is a supplementary diagnostic and cannot replace these columns.

### Geometry F@5cm

Geometry precision is the fraction of raw predicted TSDF mesh vertices within 5 cm of the GT mesh. Geometry recall is the fraction of GT mesh vertices within 5 cm of the raw predicted mesh. `F@5cm` is their harmonic mean. It is computed before GT-domain projection so missing geometry remains penalized.

## 12. Room0 Execution Protocol

The frozen room0 run uses:

- scene: `room0`;
- source frames: `[0, 2000)`;
- stride: `10`;
- sampled frames: `200`;
- camera and depth scaling from `replica8_static_v1`;
- voxel size: `0.05 m`;
- no runtime GT access;
- no future-frame access.

Execution proceeds through three gates:

1. Two-frame contract smoke: allocation, fusion, snapshot, restore, and evaluator completion.
2. Twenty-frame system smoke: real frontend, multiple entities, mesh extraction, and finite metrics.
3. Two-hundred-frame room0 validation: complete artifacts, timing, memory, and all Table 1 metrics.

## 13. Output Contract

Every room0 run writes:

```text
<run_root>/
  run_manifest.json
  timing.json
  checkpoints/
    latest_voxel_snapshot.npz
    latest_entities.jsonl
  final/
    oviv2_voxel_snapshot.npz
    oviv2_entities.jsonl
    oviv2_instance_mesh.ply
  evaluation/
    gt_aligned_semantic_ids.npy
    gt_aligned_instance_ids.npy
    semantic_map_gt.ply
    instance_map_gt.ply
    metrics.json
    per_class_semantic.json
    per_class_instance_ap.json
```

`run_manifest.json` records repository commit, dirty-state digest, command, config hash, manifest hash, vocabulary hash, model weights, hardware, frame selection, voxel settings, and artifact checksums.

`metrics.json` is the only source used to fill Table 1 tokens. PLY files are audit outputs, not numeric sources.

## 14. Error Handling

- Missing RGB, depth, pose, model weights, or GT evaluator inputs fail before the run starts.
- Empty proposal frames still integrate geometry and commit an empty observation batch.
- Frontend failures identify frame and stage and do not publish a partial snapshot.
- Non-finite voxel values, non-monotonic revisions, invalid owner references, and incompatible checkpoints fail fast.
- Mesh extraction failure preserves the final voxel snapshot and marks evaluation failed without mutating the map.
- Metric output rejects NaN headline values. Classes absent from GT are excluded from the macro rather than replaced with zero.
- Interrupted runs resume only from an atomically completed checkpoint whose manifest matches the requested configuration.

## 15. Test Strategy

### Unit Tests

- voxel/block key conversion, including negative coordinates;
- on-demand sparse block allocation;
- projective TSDF weighted fusion;
- bounded semantic top-k support;
- competing entity evidence retention;
- reversible ownership release preserving geometry;
- visibility PRESENT/ABSENT/OCCLUDED/UNOBSERVED classification;
- local-track promotion and expiry;
- immutable snapshot round trip;
- Marching Cubes label and owner projection;
- strict 5 cm GT transfer behavior;
- semantic confusion, frequency weighting, AP25/AP50, and F@5cm fixtures.

### Integration Tests

- synthetic plane geometry reconstructs a nonempty zero-crossing mesh;
- two labeled cubes remain separate instances after fusion;
- an occluder does not create absence evidence;
- releasing one owner retains the mesh and competing evidence;
- restored snapshots produce byte-equivalent aligned label arrays and equal metrics;
- the OVIV2 runtime imports no legacy mutable map state.

### System Tests

- real Replica room0 two-frame run;
- real Replica room0 20-frame run;
- real Replica room0 200-frame run;
- repeated evaluation of one final snapshot produces identical JSON metrics;
- evaluation with permuted GT semantic IDs does not change runtime artifacts.

## 16. Acceptance Criteria

The room0 milestone is complete only when:

1. OVIV2 processes all 200 sampled room0 frames without using legacy map state.
2. The authoritative final artifact contains voxel blocks and entities but no dense point cloud.
3. The raw TSDF mesh is nonempty and every headline metric is finite.
4. mIoU, mAcc, f-mIoU, semantic AP25, semantic AP50, and F@5cm are generated by one evaluator invocation.
5. Semantic and instance metrics use the OVI-MAP-style GT vertex projection at 5 cm.
6. AP25/AP50 use semantic class constraints and the frozen Replica-41 instance vocabulary.
7. Re-evaluating the immutable snapshot yields identical metrics.
8. The run manifest and metric artifacts pass provenance validation.
9. Dense visualization products, when requested, are derived after snapshot commit and cannot affect metrics.

After room0 acceptance, the same runner is applied without algorithm changes to the remaining seven Replica scenes. Replica-8 reports macro averages over eight scenes; Replica-7 reports macro averages over the same results excluding room0.
