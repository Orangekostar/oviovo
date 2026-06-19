# Object Update Voxel-First Structural Acceleration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for implementation and review. This plan is a follow-up to `2026-06-13-object-update-equivalent-acceleration.md`. Keep the first batch equivalent: no new proposal pruning, no more aggressive association caps, and no lossy threshold changes until the accuracy regression is isolated.

## Goal

Move the 200f stride=10 backend-fast run from:

- run: `20260613_object_update_accel_s10_200f`
- `mIoU`: `0.4050997576`
- `f-mIoU`: `0.3616707051`
- `final_object_count`: `57`
- `mapping_loop_sec`: `1006.009s`
- `process_frame_sec_total`: `976.227s`
- `sec/frame`: `5.0300`

to:

- required: `mapping_loop_sec < 1000s`
- preferred: `mapping_loop_sec < 900s`
- required: no metric drop below current `mIoU=0.4051`, `f-mIoU=0.3617`, object count `57`
- preferred: recover toward `mIoU >= 0.42` after accuracy ablation, without losing the new speed wins

## Current Evidence

The last optimization worked but did not fully solve the target:

- `object_update`: `3.0399 -> 2.7426s/frame`
- mapping: `1050.7 -> 1006.0s`
- precision/objects unchanged
- `_refresh_object_debug`: now only `0.0048s/frame`

Current main timings:

- `object_update_surface_owner_gate`: `1.5753s/frame`
- `object_update_local_pcd_update`: `0.5492s/frame`
- `object_update_association_geometry_update`: `0.4397s/frame`
- `association`: `0.8186s/frame`
- `runtime_vis`: `0.4432s/frame`
- `active_set`: `0.2415s/frame`
- `proposal_generation`: `0.3271s/frame`
- `patch_lifting`: `0.0416s/frame`
- `dense_surface`: near zero

Conclusion:

- `refresh_object_debug` is no longer the bottleneck.
- TSDF integrate is not the bottleneck.
- The frontend is not the bottleneck.
- The current bottleneck is point-cloud lifecycle cost in object update:
  - voxel-owner decisions expanded back into point masks;
  - patch copies for large partially rejected patches;
  - repeated `local_pcd` concatenation/downsample/cap;
  - repeated association geometry concatenation/cap.

## What The Last Run Proved

The original hidden-global-scan hypothesis was partially correct:

- `_refresh_object_debug` had a real full-scan risk.
- Replacing it with indexed TSDF support made the cost negligible.
- But total runtime only improved by about `44.7s`, so it was not the dominant remaining problem.

The stronger finding is:

- `surface_owner_gate` alone costs `315s` over 200 frames.
- local object geometry maintenance costs about `198s` over 200 frames:
  - `local_pcd_update`: `109.8s`
  - `association_geometry_update`: `87.9s`
- These are structural costs from moving and filtering large point arrays.

## Non-Solutions Or Secondary Solutions

### Association KDTree

Useful, but it does not solve object_update directly.

It can target:

- `association`: `0.8186s/frame`
- total potential: maybe `30-60s` if geometry scoring is improved

It will not reduce:

- `object_update_surface_owner_gate`
- `local_pcd_update`
- `association_geometry_update` inside object update

Therefore it is a good secondary track, not the main object_update fix.

### Active Set Visibility Cache

Useful for crossing the immediate `<1000s` line because only about `6s` are missing.

It can target:

- `active_set`: `0.2415s/frame`, about `48.3s` total

It will not solve object_update. It should be implemented as a quick equivalent win, but not mistaken for the root fix.

### More Threshold Pruning

Not allowed in the first batch.

Reason:

- backend-fast already has an accuracy regression:
  - `mIoU`: `0.4252 -> 0.4051`
  - objects: `76 -> 57`
  - `plant-stand`: `0.5025 -> 0`
- more caps/pruning may make the same problem worse.

## Phase 1: Surface Gate Instrumentation Deepening

Files:

- `src/modules/object_update.py`
- `run_room0_full_eval.py`
- `scripts/run_room0_checkpointed_eval.py`
- tests

Add frame-level counters:

- `surface_gate_decision_lookup_count`
- `surface_gate_unique_decision_voxel_count`
- `surface_gate_owner_lookup_sec`
- `surface_gate_mask_expand_sec`
- `surface_gate_patch_copy_sec`
- `surface_gate_structural_copy_sec`
- `surface_gate_filtered_point_count`
- `surface_gate_rejected_point_count`
- `surface_gate_partial_filter_patch_count`
- `surface_gate_reject_without_copy_patch_count`

Acceptance:

- 20f smoke writes all counters.
- Existing top-level metrics remain serializable.
- Subtimings roughly explain `object_update_surface_owner_gate`.

Purpose:

- Confirm exactly how much time is owner lookup versus mask expansion versus patch copy.
- Avoid rewriting the wrong part.

## Phase 2: Voxel-First Surface Gate

Files:

- `src/modules/object_update.py`
- `src/modules/tsdf_instance_map.py` if helper utilities are useful
- tests

Current issue:

Representative voxel mode already makes decisions on voxels, but when any voxel is rejected it expands decisions back to per-point masks:

```text
decision voxels -> rejected_voxel_keys set -> accepted_mask over all points -> copy accepted points
```

This is expensive for patches with hundreds of thousands of points.

New design:

Introduce a `SurfaceGateResult` dataclass:

```python
@dataclass
class SurfaceGateResult:
    patch: Patch3D | None
    structural_reject: Patch3D | None
    debug: dict[str, Any]
    voxel_view: PatchVoxelView | None
    accepted_voxel_view: PatchVoxelView | None
    rejected_voxel_view: PatchVoxelView | None
    accepted_voxel_keys: set[tuple[int, int, int]] | None
    rejected_voxel_keys: set[tuple[int, int, int]] | None
```

Implement decision flow:

1. Build decision arrays only on representative voxels.
2. If clean:
   - return metadata-only patch wrapper and original voxel view.
3. If full reject:
   - return `None` without point mask materialization.
4. If partial accept:
   - decide whether point materialization is actually needed:
     - needed for `local_pcd`;
     - needed for observation export;
     - not needed for TSDF integrate if `accepted_voxel_view` can drive integration.

First safe implementation:

- Keep partial accepted patch materialization, but optimize mask construction:
  - use `voxel_view.inverse` and a boolean per-unique-voxel keep array instead of Python tuple membership per point.
  - avoid building separate `foreign_owner_mask`, `same_owner_mask`, `background_reject_mask` point arrays unless debug explicitly requests them.
- Structural reject patch is only materialized when `background_count > 0` and background update needs it.

Expected impact:

- Reduce `surface_owner_gate` from `1.575s/frame` toward `0.9-1.1s/frame`.

Tests:

- clean pass-through does not mutate source patch.
- full reject does not materialize point masks.
- partial accept produces the same points as old tuple-membership path.
- filtered voxel view integrates only accepted voxels.
- debug counters remain accurate.

## Phase 3: Owner Lookup Acceleration

Files:

- `src/modules/tsdf_instance_map.py`
- `src/modules/object_update.py`
- tests

Current issue:

Surface gate loops in Python over decision voxels and calls:

```python
support = state.tsdf_volume.owner_support.get(key)
owner_id = support.owner_id if support is not None else -1
```

This is repeated for thousands of voxels per frame.

New design:

Maintain an owner index in `TSDFInstanceVolume`:

```python
voxel_owner_id: dict[tuple[int, int, int], int]
```

Update it inside the same TSDF support index sync path:

- after support changes, compute current owner;
- store owner if `>=0`;
- delete if unowned.

Surface gate uses:

```python
owner_id = volume.voxel_owner_id.get(key, -1)
```

This avoids per-lookup `max(support.support)` over support dicts.

Compatibility:

- If loaded checkpoint lacks `voxel_owner_id`, lazily rebuild it from `owner_support`.
- Manual test-created `owner_support` entries must still work.

Expected impact:

- Reduce owner lookup part of `surface_owner_gate`.
- Also speed `vote_patch_to_instance` and `query_visible_instances` if reused carefully.

Tests:

- index matches `owner_support.owner_id` after integrate/remove.
- stale/manual owner_support rebuilds index.
- surface gate output identical before/after.

## Phase 4: Local Geometry Memory Reservoir

Files:

- `src/core/data_structures.py`
- `src/modules/object_update.py`
- export code only if needed
- tests

Current issue:

`_update_object` does:

```python
obj.local_pcd = np.concatenate([obj.local_pcd, patch.points], axis=0)
if obj.update_count % downsample_interval == 0:
    obj.local_pcd = voxel_downsample(...)
if len(obj.local_pcd) > max_points:
    cap(...)
obj.centroid = obj.local_pcd.mean(...)
obj.bbox_min, obj.bbox_max = compute_bbox(...)
```

This repeatedly copies growing arrays.

New design:

Add optional bounded geometry accumulator:

```python
object_update:
  local_geometry_reservoir_enabled: true
  local_geometry_reservoir_voxel: 0.01
  local_geometry_reservoir_max_voxels: 20000
  local_geometry_reservoir_flush_interval: 5
```

Maintain per-object debug/private structure:

- voxel key -> representative point or running mean/count
- current centroid accumulator
- bbox min/max incrementally
- dirty flag for `local_pcd`

Safe first version:

- Keep `obj.local_pcd` as the public/export source.
- Accumulate incoming patch into a temporary reservoir.
- Flush to `obj.local_pcd` only every N updates or before export/final audit.
- For association geometry, use patch-level updates directly; do not require full local_pcd flush every frame.

Acceptance:

- Export still sees a real `obj.local_pcd`.
- `local_geometry_memory.point_count` remains meaningful.
- mIoU/object count unchanged on 20f; expected unchanged on 200f.

Expected impact:

- Reduce `local_pcd_update` from `0.549s/frame` toward `0.2-0.3s/frame`.

Risk:

- If flush timing changes object bbox/centroid used by active_set/association, association can change.
- First implementation should keep centroid/bbox exact from local_pcd unless a test proves incremental values match.

Recommended first cut:

- Do not enable in 200f until 20f and a 50f trace prove object counts and association outcomes are stable.

## Phase 5: Association Geometry Incremental Voxel Set

Files:

- `src/core/data_structures.py`
- `src/modules/object_update.py`
- `src/modules/association.py`
- tests

Current issue:

`_update_association_geometry_from_patch` does:

```python
patch_rep = voxel_downsample(patch_points, association_geometry_voxel)
combined = concatenate(existing_points, patch_rep)
if len(combined) > max:
    deterministic_spatial_cap(combined)
```

This copies and caps repeatedly.

New design:

Maintain bounded association geometry as voxel-keyed representatives:

- `obj.association_pcd` remains public array.
- private/debug cache:
  - `association_voxel_keys`
  - `association_voxel_points`
  - dirty flag

Patch update:

- voxelize patch points;
- add only new voxel representatives;
- if over cap, deterministic cap on voxel keys, not full point array;
- rebuild `association_pcd` only when changed.

Expected impact:

- Reduce `association_geometry_update` from `0.4397s/frame` toward `0.15-0.25s/frame`.
- Also helps association because object geometry arrays stay cleaner and bounded.

Tests:

- same final `association_pcd` or same voxel coverage as old method for deterministic examples.
- cap is deterministic.
- association debug remains valid.

## Phase 6: Association KDTree As Secondary Track

Files:

- `src/modules/association.py`
- tests

Current issue:

`association` remains `0.8186s/frame`.

But it is secondary to object_update. Do this after phases 2-3 or in a disjoint subagent.

Implementation:

- Cache `cKDTree` per object for `association_pcd`.
- Invalidate when `association_pcd` identity, shape, or generation changes.
- Keep NumPy fallback.
- Do not change candidate caps.

Acceptance:

- identical match decisions on unit tests.
- 20f metrics unchanged.
- `association < 0.65s/frame` preferred.

## Phase 7: Active Set Visibility Cache As Quick Total-Runtime Win

Files:

- `src/modules/active_set.py`
- `src/modules/tsdf_instance_map.py`
- tests

Current issue:

`active_set` costs `0.2415s/frame`, about `48.3s` total.

Implementation:

- Cache visible ids for nearby poses or fixed frame intervals:
  - key by quantized camera translation and forward direction;
  - invalidate when TSDF owner index revision changes.
- Add `TSDFInstanceVolume.owner_index_revision`, increment when voxel owner changes.

Conservative config:

```yaml
active_set:
  visibility_cache_enabled: true
  visibility_cache_translation_voxel: 0.05
  visibility_cache_rotation_cosine: 0.995
  visibility_cache_max_entries: 64
```

Risk:

- Active set affects association candidates.
- Cache must be exact for same quantized pose or must include fallback conservative union.

Safer first version:

- Cache projected visible ids by exact frame pose hash only inside a run. This helps only repeated calls, so likely little benefit.
- Better version: use quantized pose but union cached visible ids with nearby and whole-prior ids, and never remove new-object candidates.

Acceptance:

- 20f association outcomes unchanged.
- 200f mIoU/object count unchanged.
- total runtime crosses `<1000s` even before deeper local geometry changes.

## Phase 8: Accuracy Regression Ablation

This is not a speed optimization but must happen before threshold changes.

Compare:

- `20260613_anchor_first_sam_fast_s10_200f`
- `20260613_backend_fast_lazy_voxel_s10_200f`
- `20260613_object_update_accel_s10_200f`

Known regression:

- `mIoU`: `0.4252 -> 0.4051`
- objects: `76 -> 57`
- `plant-stand`: `0.5025 -> 0`
- `indoor-plant`: `0.9657 -> 0.8064`

Ablations:

1. backend-fast with surface gate relaxed to baseline values.
2. backend-fast with provisional behavior matched to baseline.
3. backend-fast with association caps matched to baseline, but keep lazy dense/export fixes.
4. surface gate disabled only for tiny attached object classes:
   - `plant-stand`
   - `switch`
   - `wall-plug`
   - `vent`

Acceptance:

- Identify which single knob causes most of `76 -> 57` object drop.
- Do not merge speed changes that make this worse.

## Execution Order

### Batch A: Safe Runtime Win

1. Phase 1 instrumentation.
2. Phase 2 voxel-first mask expansion.
3. Phase 3 owner index.
4. 20f smoke.
5. 200f run if 20f passes.

Expected:

- `surface_owner_gate < 1.2s/frame`
- `object_update < 2.4s/frame`
- `mapping_loop_sec < 950s`
- no metric/object count change

### Batch B: Total Runtime Crossing And Secondary Wins

1. Phase 7 active_set visibility cache.
2. Phase 6 association KDTree.
3. 20f smoke.
4. 200f run.

Expected:

- `active_set < 0.12s/frame`
- `association < 0.65s/frame`
- `mapping_loop_sec < 850-900s`

### Batch C: Structural Object Memory

1. Phase 4 local geometry reservoir behind config flag.
2. Phase 5 association geometry incremental voxel set behind config flag.
3. 20f and 50f association outcome trace.
4. 200f only if object count and metrics remain stable.

Expected:

- `local_pcd_update < 0.3s/frame`
- `association_geometry_update < 0.25s/frame`
- `object_update < 1.8-2.0s/frame`

### Batch D: Accuracy Recovery

Run ablations from Phase 8.

Expected:

- recover part of `mIoU 0.405 -> 0.425` without reverting core speed wins.

## Validation Commands

Unit tests:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python -m pytest tests/test_backend_runtime_fast.py tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled tests/test_provisional_pool.py -q
'
```

20f smoke:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_voxel_first_surface_gate_smoke_s10_20f \
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

200f run:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_voxel_first_surface_gate_s10_200f \
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

## Stop Conditions

Stop and report if:

- mIoU drops below `0.4050`;
- object count changes from `57` before an intentional accuracy ablation;
- `plant-stand` or `indoor-plant` worsens further;
- `surface_owner_gate` does not improve after Phase 2/3, because that means the bottleneck is patch copy/local memory rather than owner lookup/mask expansion;
- local geometry reservoir changes association outcomes in 20f/50f traces.

## Recommended Next Coding Batch

Use subagent-driven coding/review with disjoint ownership:

- Worker A:
  - `src/modules/object_update.py`
  - implement Phase 1 and Phase 2.
- Worker B:
  - `src/modules/tsdf_instance_map.py`
  - implement Phase 3 owner index and tests.
- Reviewer:
  - check equivalence, stale index handling, old checkpoint compatibility, JSON serializability.

Do not start with local geometry reservoir. It is high reward but higher risk. First prove the surface gate can be sped up while preserving exact outputs.
