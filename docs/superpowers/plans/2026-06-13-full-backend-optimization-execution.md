# Full Backend Optimization Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for implementation and review. This is an execution plan, not a brainstorming note. Do every phase or explicitly document why a phase is blocked by evidence.

**Goal:** Move the anchor-first OViOVO pipeline toward the target 200f stride=10 runtime without hiding work, dropping required backend semantics, or moving cost into another stage.

**Current reference runs:**

- Frontend baseline: `20260613_anchor_first_sam_fast_s10_200f`
- Backend-fast attempt: `20260613_backend_fast_s10_200f`
- Smoke final2: `20260613_backend_fast_smoke_s10_20f_final2`

## Non-Negotiable Requirements

- Do not optimize by disabling evaluation, semantic outputs, TSDF support, or required object observations.
- Do not count config-only tuning as complete unless 200f validation confirms runtime and accuracy.
- Do not move work from one timed stage to another and call it a win.
- Keep `frame_metrics.jsonl`, `run_report.json`, `mapping_timer_result.json`, and `export_eval_timer_result.json` valid.
- Keep `anchor_guided_sam_fusion` out of the critical path.
- Every optimization must have:
  - unit tests or focused module tests;
  - 20f stride=10 smoke;
  - 200f stride=10 fast run;
  - timing comparison against both reference runs when applicable.

## Evidence And Diagnosis

### Anchor-First Frontend Baseline

`20260613_anchor_first_sam_fast_s10_200f`:

- `mIoU`: `0.4252`
- `f-mIoU`: `0.3709`
- mapping time: `1280.7s`
- mean frame time: `6.4036s/frame`
- `proposal_generation`: `0.3215s/frame`
- `patch_lifting`: `0.0422s/frame`
- `association`: `1.2280s/frame`
- `object_update`: `3.2755s/frame`
- `dense_surface`: `0.4298s/frame`
- max local memory points: `1,387,041`

### Backend-Fast Attempt

`20260613_backend_fast_s10_200f`:

- `mIoU`: `0.4051`
- `f-mIoU`: `0.3617`
- mapping time: `1194.1s`
- mean frame time: `5.9704s/frame`
- `proposal_generation`: `0.3257s/frame`
- `patch_lifting`: `0.6936s/frame`
- `association`: `0.7769s/frame`
- `object_update`: `3.0685s/frame`
- `dense_surface`: near zero
- max local memory points: `738,468`

What worked:

- association candidate cap reduced candidate scores from `1418/frame` to `604/frame`.
- geometry scores dropped from `261/frame` to `124/frame`.
- dense surface online cost was removed.
- local memory max dropped by about `47%`.

What failed:

- `patch_lifting` regressed from `0.042s/frame` to `0.694s/frame`.
- `object_update` only improved slightly.
- 200f mIoU dropped by `0.0201`.
- `plant-stand` collapsed and `indoor-plant` degraded.

Root cause:

- eager voxel cache in `patch_lifting` computes `np.unique(voxel_indices, axis=0)` for every patch.
- this shifts voxel work into `patch_lifting` before the backend knows whether each patch needs it.
- surface/visibility gates still process point-level masks, so object update remains heavy.
- the optimization is partial: cache exists, but backend is not yet truly voxel-first.

## Required Final Architecture

The backend must use a shared, lazy, voxel-first representation:

```text
patch_lifting
  -> light RGB-D lift only
  -> Patch3D(points, bbox, centroid)

backend lazy voxel view
  -> built once per patch per TSDF voxel size
  -> shared by TSDF vote, surface gate, TSDF integrate
  -> filtered cheaply when gate removes points

association
  -> cheap candidate pruning
  -> top-k geometry only

object_update
  -> voxel-first surface/owner decisions
  -> minimal point copies
  -> bounded incremental object memory

dense_surface
  -> lazy/export-time or low-frequency online refresh

active_set
  -> object-level visibility first, voxel scan only when needed
```

## Phase 0: Guardrails And Baseline Tests

Create or update tests that lock down the current intended behavior before rewriting internals.

Files:

- `tests/test_backend_runtime_fast.py`
- `tests/test_pipeline.py`
- any existing association/object update tests

Required tests:

- anchor-first frontend still emits `anchor_prompted_sam`.
- `anchor_guided_sam_fusion` is absent in anchor-first fast configs.
- backend-fast config still loads.
- dense surface lazy/export path writes final surfaces.
- stage timings include:
  - `patch_lifting`
  - `association`
  - `object_update`
  - `dense_surface`
  - `active_set`
  - `proposal_generation`

Acceptance:

- Tests pass before code rewrite.
- 1f smoke still runs.

## Phase 1: Fix Patch Lifting Regression

Problem:

- `patch_lifting` now eagerly builds voxel cache and does expensive `np.unique(axis=0)`.
- This caused a `0.65s/frame` regression.

Files:

- `src/modules/patch_lifting.py`
- `configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml`
- `tests/test_backend_runtime_fast.py`

Implementation:

1. Add config:

```yaml
patch_lifting:
  voxel_cache_enabled: false
```

2. Change `PatchLiftingModule`:

- keep `_build_voxel_metadata` available;
- only call it when `voxel_cache_enabled=true`;
- default false for backend-fast config;
- default should preserve old behavior in configs that do not request eager cache.

3. Keep metadata compatibility:

- downstream code must tolerate missing `voxel_indices`;
- no required path should assume eager cache exists.

Tests:

- `voxel_cache_enabled=false` produces no voxel metadata.
- `voxel_cache_enabled=true` still produces valid cache.
- lifted points are identical in both modes.

Acceptance:

- 20f `patch_lifting <= 0.08s/frame`.
- no increase in `object_update`.
- no change in proposal/patch counts.

Rollback:

- If downstream code requires eager cache, fix downstream lazy view first; do not keep eager cache as default.

## Phase 2: Shared Lazy Patch Voxel View

Problem:

- TSDF vote, TSDF integrate, surface owner gate, and filtered patches need voxel data.
- Recomputing or eagerly computing voxel data is wasteful.

Files:

- `src/modules/tsdf_instance_map.py`
- `src/modules/object_update.py`
- possibly `src/core/data_structures.py`
- tests in `tests/test_backend_runtime_fast.py`

Implementation:

Add a lightweight helper:

```python
@dataclass(frozen=True)
class PatchVoxelView:
    voxel_size: float
    voxel_indices: np.ndarray
    unique_voxels: np.ndarray
    representative_point_indices: np.ndarray
    inverse: np.ndarray | None = None
    cache_used: bool = False
```

Required API:

```python
get_patch_voxel_view(patch, voxel_size, *, need_inverse=False) -> PatchVoxelView
filter_patch_voxel_view(view, keep_mask) -> PatchVoxelView | None
```

Rules:

- Build view lazily in backend stages, not in `patch_lifting`.
- If metadata cache exists and matches, use it.
- If no cache exists, compute once per patch per frame and pass the view explicitly.
- Store temporary per-frame views in a local dict keyed by `patch_id` or `id(patch)`.
- Do not persist large voxel arrays in observations unless explicitly needed.

Update TSDF methods:

```python
vote_patch_to_instance(volume, patch, voxel_view=None)
integrate_patch(volume, patch, instance_id, voxel_view=None)
remove_patch_support(volume, patch, instance_id, voxel_view=None)
```

Tests:

- cached and uncached views produce identical unique voxels.
- TSDF vote/integrate results match old implementation.
- filtered voxel view matches recomputation from filtered points.
- no eager `patch_lifting` cache is required.

Acceptance:

- no `patch_lifting` regression.
- TSDF stages can consume explicit voxel view.
- no accuracy change in 20f.

## Phase 3: True Voxel-First Object Update

Problem:

- `object_update` still checks about `526k` points/frame.
- representative mode alone does not eliminate point-level work.

Files:

- `src/modules/object_update.py`
- `src/modules/tsdf_instance_map.py`
- tests

Implementation:

1. In `ObjectUpdateModule.process`, create and reuse `PatchVoxelView` for each patch.

2. Rewrite `_filter_patch_by_surface_owner` to:

- decide owner conflicts on `unique_voxels` first;
- use representative points only for metadata and minimal point filtering;
- avoid full point copy when the patch passes unchanged;
- only build full accepted mask if some voxels/points are rejected.

3. Add fast path:

```text
if no foreign owner and no background owner:
    return original patch, same voxel view
```

4. When filtering is needed:

- use `representative_point_indices` and optional `inverse`;
- construct accepted mask only once;
- return filtered patch plus filtered voxel view.

5. Pass voxel views to:

- `TSDFInstanceMapModule.integrate_patch`
- `TSDFInstanceMapModule.vote_patch_to_instance` where applicable
- surface gate debug summaries

6. Add new debug fields:

- `decision_voxel_count`
- `unique_voxel_count`
- `accepted_decision_voxel_count`
- `fast_path_patch_count`
- `point_mask_materialized_count`
- `cached_voxel_view_used_count`

Tests:

- unchanged pass-through patch returns same points and no unnecessary copy.
- rejected voxel filters points correctly.
- surface gate decisions match old point-level mode on small fixtures.
- `structural_reject` behavior remains correct.

Acceptance:

- `object_update <= 2.3s/frame` after Phase 3.
- `surface_owner_gate.source_point_count` may remain for reporting, but new `decision_voxel_count` must be much lower.
- no 20f object count explosion.

## Phase 4: Association Two-Stage Scoring

Problem:

- backend-fast candidate caps helped, but association is still `0.777s/frame`.
- geometry score count remains nontrivial.

Files:

- `src/modules/association.py`
- tests

Implementation:

1. Keep candidate cap from backend-fast config.

2. Implement real two-stage scoring:

- cheap stage computes voxel vote, centroid, bbox for candidates;
- geometry NN only runs for top `top_k_geometry`;
- final selection only considers top `top_k_final` unless TSDF owner vote requires inclusion.

3. Add optional KDTree acceleration:

- Use `scipy.spatial.cKDTree` if available.
- Fallback to current NumPy broadcast.
- Cache KDTree per object association geometry version.

4. Add debug:

- `cheap_candidate_count`
- `top_k_geometry_count`
- `geometry_backend`
- `kdtree_cache_hit_count`
- `kdtree_cache_miss_count`

Tests:

- two-stage mode chooses same match as old mode on deterministic fixtures.
- geometry computation count is capped.
- KDTree fallback works if SciPy is missing.

Acceptance:

- `association <= 0.6s/frame`.
- geometry score count reduced by at least `50%` from backend-fast.
- mIoU does not drop more than `0.01` from backend-fast.

## Phase 5: Incremental Object Local Memory

Problem:

- local memory was capped from `1.39M` to `0.74M`, but object update is still heavy.
- `local_pcd` concat/downsample/cap still touches large arrays.

Files:

- `src/core/data_structures.py`
- `src/modules/object_update.py`
- export code if needed

Implementation:

1. Add object-level voxel reservoir:

```python
local_voxel_centroids: dict[voxel_key, running_mean/count]  # or compact arrays
```

2. Update only voxels touched by new patch.

3. Rebuild compact `local_pcd` only:

- every N updates;
- before export;
- when association geometry refresh needs it.

4. Keep `association_pcd` bounded and incremental:

- merge downsampled patch representative points;
- cap without full `local_pcd` scan.

5. Preserve observations:

- observation records should keep patch references or compact filtered points needed for audit/export.
- do not silently remove semantic evidence.

Tests:

- object centroid/bbox update remains close to old result.
- local memory cap is respected.
- export still sees valid object geometry.
- semantic votes from observations are unchanged.

Acceptance:

- max local memory points below `500k` on 200f.
- `object_update <= 1.8s/frame`.
- final export remains valid.

## Phase 6: Dense Surface Lazy Export Completion

Problem:

- dense surface online cost is solved, but lazy export must be verified as a complete feature.

Files:

- `src/modules/dense_surface.py`
- `run_room0_full_eval.py`
- `scripts/run_room0_checkpointed_eval.py`

Implementation:

- keep online lazy path;
- ensure export calls `refresh_all_for_export` exactly once when lazy mode is enabled;
- report online dense time and export dense refresh time separately.

Tests:

- lazy online path keeps entries valid but empty/lightweight.
- export refresh fills surface points.
- metrics output files still exist.

Acceptance:

- online `dense_surface <= 0.05s/frame`.
- export time increase is reported.
- total build-export time still improves.

## Phase 7: Active Set Acceleration

Problem:

- `active_set` costs about `0.25s/frame` in 200f runs.
- it scans TSDF owner voxels to find visible instances.

Files:

- `src/modules/active_set.py`
- `src/modules/tsdf_instance_map.py`
- `src/modules/object_update.py`

Implementation:

1. Maintain per-object TSDF support summary:

- owned voxel bbox
- owned voxel centroid
- owned voxel count

2. Active set first uses object-level frustum/bbox test.

3. Full voxel visibility scan only runs:

- every `N` frames;
- or when pose delta exceeds threshold;
- or when object support summary changed significantly.

4. Add config:

```yaml
active_set:
  object_level_visibility_enabled: true
  full_voxel_visibility_interval: 10
  pose_delta_translation_threshold: 0.25
  pose_delta_rotation_threshold_deg: 10
```

Tests:

- object-level visibility includes known visible objects.
- periodic full scan refreshes cache.
- candidate count does not grow unbounded.

Acceptance:

- `active_set <= 0.1s/frame`.
- association candidate count does not increase materially.

## Phase 8: RuntimeVis Fast Path For Anchor-Prompted SAM

Problem:

- `runtime_vis` remains about `0.44s/frame`.
- anchor-first SAM proposals are already anchor-labeled and mostly one mask per anchor.

Files:

- `src/modules/runtime_vis.py`
- `src/pipelines/main_pipeline.py`

Implementation:

- Add fast path for `proposal_source == "anchor_prompted_sam"`.
- Skip pairwise merge scoring when:
  - every proposal has strong anchor label;
  - no overlapping same-label fragments exceed threshold;
  - proposal count below configured small threshold.
- Preserve debug output and overlay compatibility.

Config:

```yaml
runtime_vis:
  anchor_prompt_fast_path_enabled: true
  anchor_prompt_fast_path_max_proposals: 64
```

Tests:

- fast path returns same proposal count for non-overlapping anchor-prompted masks.
- fallback triggers on overlapping fragments.
- runtime-vis debug JSON remains valid.

Acceptance:

- `runtime_vis <= 0.25s/frame`.
- no obvious overlay degradation on runtime-vis sample.

## Phase 9: Final Config Assembly

Create:

- `configs/replica_yoloworld_anchor_first_sam_backend_optimized_4090.yaml`

It should include all validated changes:

- eager patch voxel cache disabled;
- lazy voxel view enabled;
- voxel-first object update;
- association two-stage scoring;
- dense surface lazy export;
- active set object-level visibility;
- runtime-vis anchor fast path if validated.

Do not overwrite:

- `configs/replica_yoloworld_anchor_first_sam_4090.yaml`
- `configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml`

## Validation Commands

20f smoke:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_optimized_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_backend_optimized_smoke_s10_20f \
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

200f fast:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_optimized_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_backend_optimized_s10_200f \
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

Optional runtime-vis:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_optimized_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260613_backend_optimized_runtimevis_s10_200f \
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
  --runtime-vis-debug \
  --runtime-vis-debug-every 10 \
  --runtime-vis-debug-max-decisions 120 \
  --quiet
'
```

## Final Acceptance Gates

A phase is complete only when 200f data supports it.

Minimum acceptable optimized result:

- mapping time `< 1000s`
- mean frame time `< 5.0s/frame`
- `patch_lifting <= 0.08s/frame`
- `association <= 0.6s/frame`
- `object_update <= 2.0s/frame`
- `dense_surface <= 0.05s/frame`
- `active_set <= 0.1s/frame`
- `mIoU >= 0.405`
- `f-mIoU >= 0.36`

Preferred target:

- mapping time `< 700s`
- mean frame time `< 3.5s/frame`
- `mIoU >= 0.42`
- `f-mIoU >= 0.37`

Reject or revise if:

- mIoU drops below `0.40`;
- final object count collapses or explodes by more than `25%`;
- `plant-stand`, `indoor-plant`, or major object classes collapse without explanation;
- a stage improves only because its cost moves to another stage;
- final exports are incomplete.

## Reporting Template

Every implementation pass must report:

```text
Experiment:
Config:
Commit/worktree state:

Metrics:
- mIoU:
- f-mIoU:
- final_object_count:
- dense_geometry_point_count:

Runtime:
- mapping_sec:
- sec_per_frame:
- proposal_generation:
- patch_lifting:
- association:
- object_update:
- dense_surface:
- active_set:
- runtime_vis:

Scale:
- mean patch_count:
- mean candidate_score_count:
- mean geometry_score_count:
- mean surface decision voxel count:
- mean surface source point count:
- max local_memory_point_count_total:

Decision:
- accept / reject / continue tuning
- reason:
```

## Expected Path To Target

The practical order is:

1. Fix patch lifting regression.
2. Add shared lazy voxel view.
3. Convert object update to true voxel-first fast path.
4. Finish association two-stage/KDTree.
5. Keep dense surface lazy export.
6. Accelerate active set.
7. Add runtime-vis anchor fast path.

The current backend-fast attempt should not be treated as the endpoint. It proved which knobs are useful, but the real runtime target requires completing the voxel-first rewrite instead of only adding eager voxel cache.
