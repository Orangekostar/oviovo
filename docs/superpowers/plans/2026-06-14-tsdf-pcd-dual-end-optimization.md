# TSDF + PCD Dual-End Runtime Optimization Plan

> For agentic workers: use subagent-driven development for implementation and review. This plan is based on the latest 200f stride=10 run `20260613_voxel_first_surface_gate_s10_200f`. Do not treat config-only pruning as completion. Every speedup must preserve or explicitly report the metric/object-count effect.

## Goal

Build toward a dual-end open-vocabulary mapping backend:

- TSDF voxel map: online owner, support, visibility, stability, and spatial vote substrate.
- PCD pool: bounded object memory for semantic evidence, association sketch, debug, and export.

Next runtime target on Replica room0 200f stride=10:

- required: mapping loop `< 800s`
- preferred: mapping loop `< 700s`
- required: no drop below current `mIoU=0.4051`, `f-mIoU=0.3617`, `mAcc=0.4734`, object count `57`
- preferred: recover toward anchor-first quality after speed path is stable

## Current Baseline

Run: `outputs/tmp_validation/20260613_voxel_first_surface_gate_s10_200f`

- mapping loop: `860.67s`
- process frames: `829.28s`
- sec/frame: `4.3034`
- `mIoU`: `0.4050997576`
- `mAcc`: `0.4734024045`
- `f-mIoU`: `0.3616707051`
- `f-mAcc`: `0.4522131218`
- final objects: `57`

Major timed stages:

- `object_update`: `391.03s`, `1.955s/frame`
- `association`: `157.87s`, `0.789s/frame`
- `runtime_vis`: `93.94s`, `0.470s/frame`
- `proposal_generation`: `71.33s`, `0.357s/frame`
- `active_set`: `49.14s`, `0.246s/frame`
- `depth_refinement`: `46.06s`, `0.230s/frame`

Object update internals:

- `surface_owner_gate`: `154.26s`, `0.771s/frame`
- `local_pcd_update`: `114.01s`, `0.570s/frame`
- `association_geometry_update`: `84.89s`, `0.424s/frame`
- `tsdf_integrate`: `11.78s`, `0.059s/frame`
- `refresh_object_debug`: `0.99s`, solved

Conclusion:

- TSDF owner-index acceleration worked.
- Dense surface online cost is solved.
- The current bottleneck is now the PCD lifecycle plus association/active-set visibility.
- Surface gate is partly fixed, but still pays expensive voxel-view work before knowing whether a patch is a clean pass.

## Concrete Code Diagnosis

`src/modules/object_update.py`

- `_filter_patch_by_surface_owner` calls `get_patch_voxel_view(... need_inverse=self.surface_gate_representative_voxel_mode)` before the clean fast path.
- Clean patches do not need `inverse`, but currently still pay for it when representative mode is enabled.
- `_update_object` still updates `local_pcd` by `np.concatenate([obj.local_pcd, patch.points])`, periodic `voxel_downsample`, cap, then recomputes centroid and bbox from the full array.
- `_update_association_geometry_from_patch` still downsamples patch points, concatenates with existing `association_pcd`, then caps/copies.

`src/modules/association.py`

- Association already has `cKDTree` fallback support, but it builds the KDTree per score call.
- `association_pcd` is bounded, but not revision-cached as a reusable geometry index.
- Geometry scoring is top-k limited, but still repeats cheap candidate scoring and NN setup across many patches.

`src/modules/active_set.py`

- Visible ids are still queried from TSDF every frame.
- Nearby and whole-prior ids scan all objects every frame.
- Cost is not dominant, but `49s` total is enough to matter after object-update wins.

## Optimization Direction

The next round should not add more thresholds first. It should change data ownership:

1. TSDF owns online spatial truth.
2. PCD pool becomes a bounded, incremental, lazy materialized memory.
3. Association uses a geometry sketch/index, not repeated full point-array rebuilds.
4. Runtime visualization is an analysis profile, not part of the fastest timing profile.

## Phase A: Lazy Surface-Gate Inverse

Files:

- `src/modules/object_update.py`
- `src/modules/tsdf_instance_map.py`
- `tests/test_backend_runtime_fast.py`

Implementation:

- Change surface gate initial voxel view to `need_inverse=False`.
- Add helper `ensure_patch_voxel_inverse(voxel_view, patch)` or equivalent local helper.
- Compute inverse only for:
  - partial accept;
  - structural reject materialization;
  - exact point-count debug on non-clean paths.
- Clean fast path should return without inverse computation and without point mask materialization.
- Add counters:
  - `surface_gate_inverse_requested_count`
  - `surface_gate_inverse_build_sec`
  - `surface_gate_clean_inverse_skipped_count`

Acceptance:

- Existing surface-gate tests pass.
- 20f output shows clean fast-path inverse skipped.
- 200f metrics unchanged.
- Target: `object_update_surface_owner_gate < 0.65s/frame`.

Stop condition:

- If metrics/object count change, rollback except instrumentation.

## Phase B: Incremental PCD Pool

Files:

- `src/core/data_structures.py`
- `src/modules/object_update.py`
- `src/modules/dense_surface.py`
- tests

Problem:

- `local_pcd_update` costs `114s`.
- The current implementation repeatedly concatenates large arrays and recomputes centroid/bbox from full `local_pcd`.

Implementation:

- Add an internal object geometry pool state, either as dataclass fields or under `obj.debug["local_geometry_pool"]`:
  - pending point chunks;
  - materialized `local_pcd`;
  - dirty flag;
  - total point estimate;
  - running sum/count;
  - bbox min/max;
  - last compaction update id.
- Update centroid and bbox incrementally from accepted patches.
- Materialize or compact `local_pcd` only when:
  - point cap is exceeded;
  - export/dense surface needs a concrete array;
  - debug/audit explicitly requests full pool;
  - configured compaction interval is reached.
- Keep current exact path behind config:

```yaml
object_update:
  local_pcd_incremental_pool_enabled: true
  local_pcd_materialize_interval: 20
  local_pcd_export_materialize: true
```

Shadow mode first:

- Run new incremental stats beside existing `local_pcd`.
- Compare centroid, bbox, point count, and final exported metrics.
- Only enable replacement after shadow deltas are acceptable.

Acceptance:

- `local_pcd_update < 0.25s/frame`.
- No missing export artifacts.
- Metrics unchanged in 200f equivalent run.

Stop condition:

- If dense-surface export or semantic instance export depends on stale `local_pcd`, add explicit `materialize_object_local_pcd(obj)` before export instead of reverting the pool design.

## Phase C: Association Geometry Sketch

Files:

- `src/core/data_structures.py`
- `src/modules/object_update.py`
- `src/modules/association.py`
- tests

Problem:

- `association_geometry_update` costs `84.89s`.
- It still repeatedly downsamples, concatenates, caps, and copies arrays.

Implementation:

- Add per-object association geometry state:
  - voxel-key set or quantized representative table;
  - bounded representative points;
  - `association_geometry_revision`;
  - dirty flag.
- Update this state from patch voxel representatives, not full patch points when possible.
- Rebuild `obj.association_pcd` lazily only when association needs the array or at fixed revision intervals.
- Preserve deterministic cap ordering so association outcomes remain stable.

Acceptance:

- 20f shadow mode confirms association geometry point count and bbox are close to current path.
- 200f metrics/object count unchanged.
- Target: `association_geometry_update < 0.18s/frame`.

Stop condition:

- If association match ids diverge heavily in shadow mode, keep geometry sketch only for KDTree/index generation and continue using exact `association_pcd` for scoring until differences are isolated.

## Phase D: Association KDTree Cache

Files:

- `src/modules/association.py`
- `src/core/data_structures.py`
- tests

Problem:

- `association` costs `157.87s`.
- `_patch_to_object_nn_distances` can use `cKDTree`, but currently the tree is built per object-score call.

Implementation:

- Cache one KDTree per object association geometry revision.
- Store cache outside serialized object state if needed:
  - key: `(object_id, association_geometry_revision)`
  - value: tree plus source array id/shape.
- Reuse patch sample across candidate objects within one patch score call.
- Keep numpy fallback when scipy is unavailable.
- Add debug:
  - `association_kdtree_cache_hit_count`
  - `association_kdtree_cache_miss_count`
  - `association_geometry_score_sec`

Acceptance:

- Unit test proves KDTree reused until object revision changes.
- Association scores match current path within tolerance.
- Target: `association < 0.60s/frame`.

Stop condition:

- If KDTree cache increases memory or serialization risk, keep cache runtime-only inside `AssociationModule`, not inside `ObjectMap`.

## Phase E: Active-Set Visibility Cache

Files:

- `src/modules/active_set.py`
- `src/modules/tsdf_instance_map.py`
- tests

Problem:

- `active_set` costs `49.14s`.
- It is not the root bottleneck, but becomes important after object-update reduction.

Implementation:

- Cache visible ids using:
  - `tsdf_volume.owner_index_revision`;
  - quantized camera pose;
  - intrinsics signature.
- Cache nearby ids using:
  - object centroid revision or object update count sum;
  - quantized camera position.
- Cache whole-prior ids until object whole-evidence revision changes.
- Use conservative union on cache uncertainty.

Acceptance:

- Active-set candidate ids are never smaller than exact mode in safety tests.
- Target: `active_set < 0.12s/frame`.

Stop condition:

- If exact equality is hard, allow conservative superset only; never use an undersized candidate set.

## Phase F: Runtime Profiles

Files:

- configs
- `src/modules/runtime_vis.py`
- run scripts
- tests

Problem:

- `runtime_vis` costs `93.94s`, about `0.47s/frame`.
- It is useful for diagnosis but should not be counted in the fastest backend timing.

Implementation:

- Define two validated profiles:
  - analysis profile: runtime overlay enabled, writes per-frame debug and overlay.
  - fast profile: runtime overlay disabled or decimated, metrics and JSON preserved.
- Make run names explicit:
  - `*_analysis_runtimevis_*`
  - `*_fast_*`

Acceptance:

- Analysis run writes readable overlays.
- Fast run still writes `frame_metrics.jsonl`, `run_report.json`, `mapping_timer_result.json`, `export_eval_timer_result.json`.
- Fast runtime report clearly records `runtime_vis.enabled=false` or interval value.

## Phase G: Precision Regression Ablation

Files:

- configs
- analysis scripts
- tests only if code paths change

Reason:

- Current fastest family is still below `20260613_anchor_first_sam_fast_s10_200f`:
  - `mIoU`: `0.4252 -> 0.4051`
  - final objects: `76 -> 57`
  - known fragile classes include `plant-stand` and `indoor-plant`.

Experiments:

- Compare exact same frontend with:
  - surface gate on/off;
  - provisional promotion thresholds unchanged vs relaxed;
  - association geometry exact vs sketch;
  - active-set exact vs conservative cache.
- Add per-object lifecycle audit:
  - proposal seen;
  - provisional created;
  - promoted;
  - matched/merged;
  - rejected by surface owner;
  - lost at export.

Acceptance:

- Identify which stage loses the missing objects.
- Any speed optimization must have an equivalent-mode run before tuning thresholds.

## Execution Order

Batch 1: Low-risk equivalent speed

- Phase A lazy inverse.
- Phase D KDTree cache if isolated enough.
- 20f smoke, then 200f fast.
- Expected: mapping `820s` range, no metric change.

Batch 2: PCD lifecycle restructure

- Phase B incremental local PCD shadow.
- Phase C association geometry sketch shadow.
- 20f/50f shadow comparison, then enabled 200f.
- Expected: mapping `<760s`, no metric change.

Batch 3: Profile and active-set cleanup

- Phase E active-set cache.
- Phase F fast/analysis profiles.
- 200f fast and 200f analysis-runtimevis.
- Expected fast mapping `<700-730s`; analysis run slower but visually inspectable.

Batch 4: Accuracy recovery

- Phase G ablation after equivalent runtime path is stable.
- Goal: recover toward `mIoU >= 0.42` without giving back the structural speedups.

## Required Validation Commands

Unit/focused tests:

```bash
docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python -m pytest tests/test_backend_runtime_fast.py tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled tests/test_provisional_pool.py tests/test_association.py -q'
```

20f smoke:

```bash
docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python scripts/run_room0_checkpointed_eval.py --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml --output-root outputs/tmp_validation/20260614_dual_end_smoke_s10_20f --num-frames 20 --frame-stride 10'
```

200f fast:

```bash
nohup docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python scripts/run_room0_checkpointed_eval.py --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml --output-root outputs/tmp_validation/20260614_dual_end_fast_s10_200f --num-frames 200 --frame-stride 10' > /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/20260614_dual_end_fast_s10_200f.nohup.log 2>&1 &
```

200f analysis runtime-vis:

```bash
nohup docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python scripts/run_room0_checkpointed_eval.py --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml --output-root outputs/tmp_validation/20260614_dual_end_analysis_runtimevis_s10_200f --num-frames 200 --frame-stride 10' > /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/20260614_dual_end_analysis_runtimevis_s10_200f.nohup.log 2>&1 &
```

## Review Checklist

- No threshold/cap change is counted as a structural optimization.
- No required output file is disabled.
- 20f smoke must include nested timings and new counters.
- 200f fast must compare against `20260613_voxel_first_surface_gate_s10_200f`.
- Analysis-runtimevis must include overlays sufficient to inspect object identity and surface ownership failures.
- If object count, mIoU, or class-level IoU changes, report the first divergent phase before continuing.
