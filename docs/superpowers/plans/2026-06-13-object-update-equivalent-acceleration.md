# Object Update Equivalent Acceleration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for implementation and review. This plan targets equivalent backend acceleration first. Do not introduce new accuracy tradeoffs until the existing accuracy regression is isolated.

**Goal:** Reduce 200f stride=10 runtime below `1000s` by accelerating `object_update` and nearby backend stages without further reducing accuracy.

**Current best runtime baseline:** `20260613_backend_fast_lazy_voxel_s10_200f`

- `mIoU`: `0.4051`
- `f-mIoU`: `0.3617`
- mapping time: `1050.7s`
- mean frame time: `5.25s/frame`
- `object_update`: `3.0399s/frame`
- `association`: `0.7636s/frame`
- `runtime_vis`: `0.4361s/frame`
- `active_set`: `0.2390s/frame`
- `patch_lifting`: `0.0440s/frame`
- `dense_surface`: near zero

## Current Diagnosis

The previous lazy voxel fix succeeded:

- `patch_lifting` recovered from `0.6936s/frame` to `0.0440s/frame`.
- mapping improved from `1194.1s` to `1050.7s`.
- metrics stayed identical to the previous backend-fast run.

The remaining bottleneck is now clear:

- `object_update` still costs about `3.04s/frame`.
- It correlates strongly with:
  - `surface_owner_gate.source_point_count`: `0.883`
  - `surface_owner_gate.accepted_point_count`: `0.884`
  - `surface_owner_gate.decision_voxel_count`: `0.782`
  - `local_memory_point_count_total`: `0.680`
  - `matched_patch_count`: `0.573`

Scale in the best current run:

- mean surface source points: `526654/frame`
- mean decision voxels: `8840/frame`
- mean matched patches: `22.625/frame`
- max local memory points: `738468`
- mean association candidates: `604/frame`

Conclusion:

- The frontend and `patch_lifting` are not the next target.
- `dense_surface` online cost is already solved.
- `object_update` is still doing too much point-level and global debug work.
- Next changes must be equivalent acceleration, not more aggressive pruning.

## Accuracy Constraint

The current backend-fast family has an accuracy regression compared with `20260613_anchor_first_sam_fast_s10_200f`:

- `mIoU`: `0.4252 -> 0.4051`
- `f-mIoU`: `0.3709 -> 0.3617`
- final objects: `76 -> 57`
- `plant-stand`: `0.5025 -> 0.0`
- `indoor-plant`: `0.9657 -> 0.8064`

The lazy voxel fix did not cause this; eager and lazy backend-fast metrics are identical.

Therefore:

- Do not make association caps more aggressive yet.
- Do not reduce `max_points_per_object` further yet.
- Do not add more lossy sampling yet.
- First recover runtime through equivalent computation changes.

## Phase 1: Add Object Update Internal Timings

Files:

- `src/modules/object_update.py`
- `src/pipelines/main_pipeline.py` if needed for exposing nested timings
- `run_room0_full_eval.py` / `scripts/run_room0_checkpointed_eval.py` only if frame metrics need extra fields
- tests

Problem:

`object_update` is currently a black box. We need timing by substage before larger rewrites.

Add timing fields:

- `object_update_current_frame_visibility_gate`
- `object_update_surface_owner_gate`
- `object_update_tsdf_integrate`
- `object_update_local_pcd_update`
- `object_update_association_geometry_update`
- `object_update_refresh_object_debug`
- `object_update_semantic_vote`
- `object_update_provisional_pool`
- `object_update_surface_gate_summary`

Implementation:

- Add a lightweight internal timing accumulator in `ObjectUpdateModule`.
- Reset it at the start of `process`.
- Expose it as `self.last_stage_timings`.
- Merge into `pipeline.last_frame_debug["object_update_stage_timings"]`.
- Write into each frame metric under `object_update_stage_timings`.

Tests:

- `ObjectUpdateModule.process` populates timing keys.
- Existing frame metrics still contain normal top-level `stage_timings`.

Acceptance:

- 20f run includes all object update substage timing keys.
- Total substage timings should roughly explain `object_update`.

## Phase 2: Make `_refresh_object_debug` Incremental Or Low-Frequency

Files:

- `src/modules/object_update.py`
- `src/modules/tsdf_instance_map.py`
- tests

Problem:

`_refresh_object_debug(obj, state.tsdf_volume)` calls:

```python
summarize_instance_support(volume, obj.object_id)
```

`summarize_instance_support` scans all `volume.owner_support.values()` for one object. Because this is called after many object updates per frame, it can become:

```text
updated_objects_per_frame * global_owner_voxels
```

This is likely expensive debug/stat work, not core mapping logic.

Implementation option A, lowest risk:

- Add config:

```yaml
object_update:
  refresh_object_debug_interval: 10
  refresh_object_debug_on_create: true
```

- Refresh full TSDF support debug only:
  - when object is created;
  - every N source frames;
  - at export/final audit if needed.
- Between refreshes, update cheap fields only:
  - local point count
  - association geometry count
  - whole evidence
  - anchor semantic compact state

Implementation option B, better:

- Maintain per-instance TSDF support summary incrementally during `integrate_patch`.
- Store summary in `TSDFInstanceVolume` or in a side map:
  - owned voxel count
  - support mass
  - mean/max support
  - competing support mass if feasible
- `_refresh_object_debug` reads cached summary.

Recommended order:

1. Implement option A first for quick runtime proof.
2. If it helps materially, implement option B to preserve fresh debug every frame without scan.

Tests:

- Debug fields still exist.
- Low-frequency mode does not remove `global_instance_substrate`.
- Created objects get initial debug.
- Final export/report still runs.

Acceptance:

- `object_update_refresh_object_debug` is no longer a major substage.
- `object_update` improves without metric change.
- 20f mIoU/object counts unchanged.

## Phase 3: Surface Owner Gate Fast Path

Files:

- `src/modules/object_update.py`
- tests

Problem:

Surface owner gate now uses voxel decisions, but still materializes point masks and copies patches for every passing patch.

Current expensive behavior:

- build `accepted_mask` for every point;
- build `background_reject_mask`;
- build `foreign_owner_mask`;
- build `same_owner_mask`;
- call `_copy_patch_with_filtered_points` even when the patch passes unchanged;
- recompute bbox/centroid for copied patch.

Implementation:

Add a true pass-through fast path:

```text
if representative mode is enabled
and decision_foreign_count == 0
and decision_background_count == 0
and no rejection_reasons
and no structural_reject is needed:
    attach debug to original patch metadata if needed
    return original patch, None, debug
```

Important:

- Avoid point-mask materialization on the fast path.
- Avoid point copy on the fast path.
- Preserve debug output.
- Do not mutate shared patch metadata in a way that breaks observation history; if needed, copy metadata shallowly only.

Add debug counters:

- `surface_gate_fast_path_patch_count`
- `surface_gate_point_mask_materialized_count`
- `surface_gate_patch_copy_count`

Tests:

- clean pass-through patch returns same point array or same patch object.
- rejected patch still filters correctly.
- structural reject still works.
- debug counters are correct.

Acceptance:

- `object_update_surface_owner_gate` drops.
- `object_update` improves.
- no change in patch/object counts.

## Phase 4: Pass `PatchVoxelView` Through Object Update

Files:

- `src/modules/object_update.py`
- `src/modules/tsdf_instance_map.py`
- tests

Problem:

Lazy voxel view exists, but view reuse is incomplete. A patch may build a view for surface gate and then build another view for TSDF integrate.

Implementation:

- In `ObjectUpdateModule.process`, maintain:

```python
voxel_views: dict[int, PatchVoxelView]
```

or use `id(patch)` if patch ids can collide.

- `_filter_patch_by_surface_owner` should accept optional `voxel_view`.
- Return:

```python
filtered_patch, structural_reject, gate_debug, filtered_voxel_view
```

or introduce a small result dataclass.

- Pass `filtered_voxel_view` into:

```python
self.tsdf_module.integrate_patch(..., voxel_view=filtered_voxel_view)
```

- If the patch passes unchanged, pass the same view.
- If the patch is filtered, derive filtered view via `filter_patch_voxel_view`.

Tests:

- unchanged patch reuses same view.
- filtered patch view matches recomputation.
- TSDF integrate receives view and results match old behavior.

Acceptance:

- fewer lazy view rebuilds.
- object_update timing improves.
- no metric change.

## Phase 5: Association Equal-Quality Speedup

Files:

- `src/modules/association.py`
- tests

Problem:

`association` is improved but still `0.7636s/frame`.
Current candidate caps already caused some accuracy risk, so the next change should speed the same candidate set rather than prune more.

Implementation:

- Keep current `max_scored_candidates=24` and `max_geometry_candidates=8`.
- Add KDTree backend for geometry consistency:
  - use `scipy.spatial.cKDTree` if available;
  - fallback to current NumPy broadcast.
- Cache per-object KDTree and invalidate when `association_pcd` changes.
- Keep deterministic point subsampling.

Debug:

- `geometry_backend`
- `kdtree_cache_hit_count`
- `kdtree_cache_miss_count`

Tests:

- KDTree and NumPy geometry scores are close.
- fallback path works.
- selected match unchanged on fixtures.

Acceptance:

- `association <= 0.6s/frame`.
- no further mIoU drop.

## Phase 6: Active Set Visibility Cache

Files:

- `src/modules/active_set.py`
- `src/modules/tsdf_instance_map.py`
- tests

Problem:

`active_set` is `0.239s/frame` and rises to about `0.32s/frame` later in the sequence. It scans TSDF owner voxels for visible instances.

Implementation:

- Add object-level visibility first:
  - object bbox/frustum check;
  - nearby radius;
  - whole-prior ids.
- Cache full voxel visibility result:

```yaml
active_set:
  object_level_visibility_enabled: true
  full_voxel_visibility_interval: 10
  pose_delta_translation_threshold: 0.25
  pose_delta_rotation_threshold_deg: 10
```

- Fall back to full scan conservatively.

Acceptance:

- `active_set <= 0.1s/frame`.
- association candidate count does not grow materially.
- no object count explosion.

## Phase 7: Accuracy Regression Ablation

Purpose:

Find which backend-fast knob caused the `mIoU -0.0201` and object-count drop.

Do this after equivalent acceleration improves runtime, not before.

Ablation configs:

1. `lazy_voxel + dense_surface_lazy only`
2. `lazy_voxel + dense_surface_lazy + association caps`
3. `lazy_voxel + dense_surface_lazy + max_points_per_object=40000`
4. `lazy_voxel + representative_voxel_mode=false`
5. `lazy_voxel + max_scored_candidates disabled`

Run at least 20f first. If object count/plant classes diverge, run 200f for the likely culprit.

Track:

- final object count
- `plant-stand` IoU
- `indoor-plant` IoU
- `mIoU`
- `f-mIoU`
- association candidate counts
- local memory max

Expected suspects:

- `max_points_per_object=20000`
- `max_scored_candidates=24`
- `representative_voxel_mode=true`

Acceptance:

- Identify the culprit or show that multiple knobs interact.
- Restore mIoU toward `0.42` if possible without losing the speed wins.

## Validation Protocol

For each phase:

1. Unit tests.
2. 20f smoke.
3. 200f fast run if 20f passes and timing improves.

20f command:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_object_update_accel_smoke_s10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-version 2.1 \
  --sam-ckpt-path data/input/sam_ckpts \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --fast-eval \
  --quiet
'
```

200f command:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_object_update_accel_s10_200f \
  --num-frames 200 \
  --frame-stride 10 \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-version 2.1 \
  --sam-ckpt-path data/input/sam_ckpts \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --fast-eval \
  --quiet
'
```

## Acceptance Gates

Short-term required:

- mapping time `< 1000s`
- `object_update < 2.7s/frame`
- no mIoU drop below `0.405`
- `patch_lifting <= 0.08s/frame`
- `dense_surface <= 0.05s/frame`

Preferred after phases 2-4:

- `object_update < 2.3s/frame`
- mapping time `< 900s`

Preferred after phases 5-6:

- `association < 0.6s/frame`
- `active_set < 0.1s/frame`
- mapping time `< 800s`

Accuracy recovery target after ablation:

- `mIoU >= 0.42`
- `f-mIoU >= 0.37`
- final object count not below `70` unless IoU improves

Reject if:

- stage time only moves elsewhere;
- object count drops further;
- `plant-stand` or `indoor-plant` degrade further;
- final exports become incomplete.

## Recommended Next Coding Batch

Do this exact first batch:

1. Add object_update internal substage timings.
2. Add low-frequency `_refresh_object_debug` config.
3. Add surface gate pass-through fast path.
4. Pass `PatchVoxelView` into TSDF integrate when available.
5. Run tests and 20f smoke.
6. If `object_update` improves, run 200f fast.

This batch is the highest-value low-risk path because it avoids additional pruning and targets the measured hot path directly.
