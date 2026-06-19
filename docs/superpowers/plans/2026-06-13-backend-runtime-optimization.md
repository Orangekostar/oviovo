# Backend Runtime Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for implementation and review. Keep changes scoped, benchmark every step, and preserve the latest anchor-first SAM frontend path as the timing baseline.

**Goal:** Reduce OViOVO 200f stride=10 online mapping runtime after the frontend bottleneck has been removed.

**Current baseline:** `20260613_anchor_first_sam_fast_s10_200f`

- `mIoU`: `0.4252`
- `f-mIoU`: `0.3709`
- mapping time: `1280.7s`
- mean frame time: `6.40s/frame`
- `proposal_generation`: `0.3215s/frame`
- `anchor_guided_sam_fusion`: absent

This plan is about backend runtime. Accuracy improvements for structural classes should be handled separately unless they directly interact with runtime.

## Current Bottleneck Evidence

Fast run stage timings:

- `object_update`: `3.2755s/frame`, `655.1s total`, `51.2%`
- `association`: `1.2280s/frame`, `245.6s total`, `19.2%`
- `runtime_vis`: `0.4421s/frame`, `88.4s total`, `6.9%`
- `dense_surface`: `0.4298s/frame`, `86.0s total`, `6.7%`
- `proposal_generation`: `0.3215s/frame`, `64.3s total`, `5.0%`
- `active_set`: `0.2522s/frame`, `50.4s total`, `3.9%`
- `depth_refinement`: `0.2160s/frame`, `43.2s total`, `3.4%`

Scale signals from `frame_metrics.jsonl`:

- mean raw proposals/frame: `17.35`
- mean refined patches/frame: `26.24`
- total refined patches: `5248`
- mean association candidate scores/frame: `1418.24`
- total association candidate scores: `283648`
- total geometry NN scores: `52188`
- mean surface-owner checked points/frame: `526654`
- total surface-owner checked points: `105330832`
- max local memory points: `1387041`

Conclusion:

- The frontend is no longer the main bottleneck.
- The backend is dominated by point/voxel memory updates and patch-to-object association.
- The current implementation repeatedly touches large point arrays and sparse Python dict voxel maps.

## Backend Workflow To Preserve

Current per-frame backend flow:

```text
anchor-first proposals
  -> runtime_vis grouping
  -> depth_refinement
  -> patch_lifting
  -> active_set
  -> bg_obj_split
  -> association
  -> object_update
  -> semantic_memory
  -> dense_surface
  -> background_update / dynamic_maintenance
```

Do not bypass the backend semantics casually. Optimizations must preserve:

- TSDF owner-support as the global instance substrate.
- object local memory as bounded object geometry.
- semantic vote/observation records needed for export.
- final `run_report.json`, `frame_metrics.jsonl`, and timing outputs.

## Phase 1: Low-Risk Config Experiments

Create one or more config variants derived from:

- `configs/replica_yoloworld_anchor_first_sam_4090.yaml`

Suggested first config:

- `configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml`

Changes to test:

```yaml
association:
  max_scored_candidates: 24
  max_geometry_candidates: 8
  score_parallel_enabled: true
  score_parallel_workers: 4
  score_parallel_min_candidates: 16

object_update:
  downsample_interval: 20
  max_points_per_object: 20000
  surface_owner_gate:
    representative_voxel_mode: true

dual_map:
  dense_surface_update_interval: 5
  dense_surface_cap_per_object: 2000
```

Run order:

1. 20f stride=10 smoke.
2. 200f stride=10 fast run.
3. Compare runtime and IoU against `20260613_anchor_first_sam_fast_s10_200f`.

Acceptance:

- mapping time improves by at least `20%`.
- `mIoU` drop is less than `0.02`.
- object count does not explode by more than `25%`.
- no stage timing field disappears.

Rollback:

- If object count explodes or mIoU drops sharply, revert candidate cap first.
- If structural/object boundaries degrade, revert `representative_voxel_mode`.
- If export becomes sparse, revert dense surface caps/interval.

## Phase 2: Association Algorithm Acceleration

Files:

- `src/modules/association.py`
- tests around association behavior

Problem:

- Current association scores about `1418` candidates/frame.
- Geometry consistency uses patch points against object points with broadcast nearest-neighbor distance.
- Even with `max_geometry_candidates`, candidate scoring remains broad.

Implement two-stage scoring:

1. Cheap stage:
   - TSDF owner vote
   - centroid score
   - 3D bbox IoU
   - optional same semantic label boost only for candidate pruning, not final low-level association

2. Expensive stage:
   - geometry NN only for top-k candidates.
   - default `top_k_geometry=3` or `5`.

Add config:

```yaml
association:
  two_stage_enabled: true
  top_k_geometry: 5
  top_k_final: 24
```

Potential acceleration algorithms:

- `scipy.spatial.cKDTree` for patch-to-object NN.
- cached per-object KDTree rebuilt only when `association_pcd` changes.
- torch `cdist` batched on GPU if CPU KDTree is slower.
- FAISS for batched nearest-neighbor if object count grows.

First implementation preference:

- Use `cKDTree` if SciPy is available in the container.
- Fall back to current NumPy broadcast if SciPy is unavailable.
- Keep deterministic subsampling unchanged.

Tests:

- Association result remains unchanged for a small deterministic fixture.
- Two-stage mode only computes geometry for top-k candidates.
- Fallback path works when SciPy is unavailable.

Metrics to report:

- `candidate_score_count`
- `geometry_score_count`
- `score_parallel_used_count`
- `association` mean/p95/total
- selected object match agreement vs baseline on a small fixture

## Phase 3: Object Update Voxel-First Rewrite

Files:

- `src/core/data_structures.py`
- `src/modules/patch_lifting.py`
- `src/modules/object_update.py`
- `src/modules/tsdf_instance_map.py`

Problem:

- `object_update` checks about `526k` points/frame.
- `TSDFInstanceMapModule.integrate_patch` converts points to voxels and loops Python sets/dicts per patch.
- surface owner gate and visibility gate also walk patch points.

Add cached voxel representation to `Patch3D`:

```python
metadata["voxel_indices"] or patch.voxels
metadata["representative_point_indices"]
```

Implementation direction:

- During `patch_lifting`, compute patch voxel indices once.
- Store unique voxels and representative point indices.
- Use unique voxels for:
  - TSDF vote
  - TSDF integrate
  - surface owner gate decision
- Use representative points only when depth projection is required.

Potential acceleration algorithms:

- NumPy unique/inverse grouping for voxel aggregation.
- Numba for voxel hash update loops if available.
- C++/pybind sparse hash map only if Python/Numba remains too slow.
- Torch sparse tensors are lower priority because Python dict owner-support semantics need careful porting.

Acceptance:

- `object_update` mean below `1.8s/frame`.
- `surface_owner_gate.source_point_count` replaced or supplemented by `decision_voxel_count`.
- TSDF owned voxel counts remain within `5%` on a 20f comparison.
- no material IoU drop on 200f.

## Phase 4: Incremental Local Geometry Memory

Files:

- `src/modules/object_update.py`
- `src/core/data_structures.py`

Problem:

- `obj.local_pcd = concatenate(old, patch.points)` grows large.
- Periodic `voxel_downsample` and deterministic cap operate on full object point clouds.
- max local memory reached `1.39M` points in 200f.

Implementation direction:

- Maintain object-level voxel reservoir instead of repeated full point concat/downsample.
- Update only voxels touched by the new patch.
- Store:
  - compact `local_pcd` for export/debug
  - bounded `association_pcd` for association
  - optional dense points moved to `dense_surface_map`

Potential algorithms:

- voxel hash map keyed by integer voxel, storing running mean/count.
- reservoir sampling per object.
- per-object capped spatial grid.

Acceptance:

- max local memory points reduced by at least `50%`.
- `object_update` mean decreases.
- association quality does not collapse.
- final export still has enough points for evaluation.

## Phase 5: Dense Surface Decoupling

Files:

- `src/modules/dense_surface.py`
- export path in `run_room0_full_eval.py`

Problem:

- `dense_surface` costs about `0.43s/frame`.
- It refreshes online surfaces even though fast evaluation mostly needs final export.

Implementation direction:

- Increase update interval by config first.
- Then add lazy/export-time dense refresh mode.
- Optionally run dense surface refresh in a background worker.

Config:

```yaml
dual_map:
  dense_surface_update_interval: 5
  dense_surface_lazy_export: true
```

Acceptance:

- online `dense_surface` below `0.1s/frame`.
- export time may increase, but total build-export time should still improve.
- final evaluation output remains valid.

## Phase 6: Active Set Acceleration

Files:

- `src/modules/active_set.py`
- `src/modules/tsdf_instance_map.py`

Problem:

- `active_set` costs about `0.25s/frame`.
- Visible set calculation scans TSDF owner voxels.

Implementation direction:

- Maintain per-object owned voxel bbox / centroid / radius.
- Use object-level frustum test before voxel-level visibility.
- Recompute full visible set every `N` frames, reuse between frames.

Potential algorithms:

- object bbox frustum culling.
- spatial grid over object centroids.
- cached visible ids with pose delta invalidation.

Acceptance:

- `active_set` below `0.1s/frame`.
- association candidate count does not increase materially.

## Validation Protocol

For every phase:

1. Unit tests for changed module.
2. 20f stride=10 smoke:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_backend_fast_smoke_s10_20f \
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
  --fast-eval
'
```

3. 200f stride=10 fast run.
4. Optional runtime-vis run only if metrics change unexpectedly.

Required report fields:

- `mIoU`, `mAcc`, `f-mIoU`, `f-mAcc`
- mapping wall time
- `object_update`, `association`, `dense_surface`, `active_set` mean/p95/total
- candidate score count
- geometry score count
- surface-owner point or voxel count
- final object count
- max local memory points

## Target Milestones

Milestone A, config-only:

- mapping time below `1000s`
- frame time below `5.0s/frame`
- no more than `0.02` mIoU drop

Milestone B, association acceleration:

- `association < 0.7s/frame`
- geometry score count reduced by at least `50%`

Milestone C, object update rewrite:

- `object_update < 1.8s/frame`
- total mapping below `700s`

Milestone D, full backend optimized:

- total mapping below `10min` for 200f stride=10
- frontend remains below `0.5s/frame`
- no accuracy regression beyond accepted frontend baseline

## Expected Best Path

The fastest useful path is likely:

1. Config-only cap/downsample/interval experiment.
2. Association two-stage scoring with KDTree fallback.
3. Patch voxel cache and TSDF/surface gate voxel-first update.
4. Dense surface lazy export.

Do not spend time optimizing YOLOWorld/SAM2 until backend frame time is below `3s/frame`.
