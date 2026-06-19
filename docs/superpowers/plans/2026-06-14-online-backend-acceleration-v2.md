# Online Backend Acceleration V2 Plan

> For agentic workers: use subagent-driven development for implementation and review. This plan is based on the completed 200f stride=10 run `20260614_bounded_pool_s10_200f/20260614153952_room0_checkpointed_s10_200f`. Do not merge all algorithmic changes into one untraceable patch. Implement as independent experiment lines with one shared benchmark contract.

## Goal

Move the system from "bounded but still expensive" to a genuinely online TSDF voxel map + object PCD pool open-vocabulary mapping backend.

System story:

- TSDF voxel map is the online spatial substrate for ownership, visibility, support, and fast candidate retrieval.
- Object PCD pool is a bounded semantic geometry memory, materialized only when needed.
- Association uses indexed object memory, not repeated dense point-array scans.
- Fast benchmark and debug visualization are separate profiles.

Replica room0, 200 frames, stride 10 target:

- required: `mIoU >= 0.396`
- preferred: `mIoU >= 0.406`
- required: final objects within `57 +/- 3`
- required: `final_local_memory_point_count_total <= 1,200,000`
- required: `online_mapping_sec < 700s`
- preferred: `online_mapping_sec < 620s`
- required: `end_to_end_sec < 840s`
- preferred: `end_to_end_sec < 760s`

## Current Evidence

Latest bounded run:

`outputs/tmp_validation/20260614_bounded_pool_s10_200f/20260614153952_room0_checkpointed_s10_200f`

- `mIoU`: `0.406403`
- `mAcc`: `0.474338`
- `f-mIoU`: `0.365161`
- `f-mAcc`: `0.457040`
- final objects: `57`
- `online_mapping_sec`: `800.74s`
- `finalization_sec`: `3.45s`
- `eval_io_sec`: `135.42s`
- `end_to_end_sec`: `939.61s`
- `final_local_memory_point_count_total`: `370,723`
- `max_local_memory_point_count_total`: `1,241,120`
- `mapping_state_size_bytes`: `1,077,766,097`

Comparison:

- Previous fast bounded/eager-ish run `20260614_structural_fast_s10_200f_fasteval`:
  - `online_mapping_sec`: `875.53s`
  - `export_eval_sec`: `129.27s`
  - `mIoU`: `0.404964`
- Naive lazy diagnostic `20260614_structural_v2_fast_s10_200f`:
  - `online_mapping_sec`: `702.64s`
  - `export_eval_sec`: `671.35s`
  - `mIoU`: `0.408156`
  - `final_local_memory_point_count_total`: `85,371,124`

Interpretation:

- Bounded local memory fixed the invalid export explosion.
- Quality is stable.
- The remaining problem is online backend work, especially repeated compaction, surface ownership checks, and association retrieval.

## Bottleneck Facts

From the latest bounded 200f run:

- `object_update`: `362.24s`, `45.2%` of online time
- `object_update_surface_owner_gate`: `156.17s`
- `object_update_local_pcd_update`: `125.72s`
- `local_pcd_compaction_sec`: `124.74s`
- `association`: `143.36s`
- `object_update_association_geometry_update`: `44.12s`
- `runtime_vis`: `88.26s`
- `proposal_generation`: `65.85s`
- `active_set`: `48.55s`
- `depth_refinement`: `42.90s`

Correlations:

- `association` correlates with `local_memory_point_count_total` at `0.68`.
- `object_update_surface_owner_gate` correlates with `local_memory_point_count_total` at `0.61`.
- `object_update_local_pcd_update` correlates with `local_memory_point_count_total` at `0.57`.
- `runtime_vis` correlates with `raw_proposal_count` at `0.91`.
- `active_set` correlates with `total_object_count` at `0.91`.

Conclusion:

- The next speedup must reduce data scanned per frame.
- Config-only frequency tuning is not enough.
- The local pool must become indexed memory, not an array that is repeatedly concatenated and downsampled.

## Shared Benchmark Contract

Every experiment line must write and compare:

- `online_mapping_sec`
- `finalization_sec`
- `eval_io_sec`
- `end_to_end_sec`
- `mIoU`, `mAcc`, `f-mIoU`, `f-mAcc`
- `final_object_count`
- `final_local_memory_point_count_total`
- `max_local_memory_point_count_total`
- `local_pcd_pending_point_peak`
- `local_pcd_compaction_count`
- `local_pcd_compaction_sec`
- `association_geometry_sketch_update_count`
- `association_geometry_sketch_new_key_count`
- `association_geometry_sketch_dropped_key_count`
- `association_geometry_sketch_materialize_count`
- `mapping_state_size_bytes`
- `largest_export_size_bytes`
- `stage_timing_summary`
- `association_pruning_summary`

Required run profiles:

1. `fast`
   - no runtime visualization except minimal JSON counters;
   - used for headline runtime.
2. `debug_overlay`
   - runtime visualization enabled at a fixed interval or selected frames;
   - used for human inspection only.

Do not compare a debug-overlay runtime against a fast runtime as the headline speed result.

## Experiment Line A: Incremental Voxel PCD Pool

Files:

- `src/modules/object_update.py`
- `src/core/data_structures.py` if persistent fields are needed
- `src/modules/dense_surface.py`
- `run_room0_full_eval.py`
- `tests/test_pipeline.py`
- `tests/test_dual_map.py`
- `configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml`

Observed problem:

- `object_update_local_pcd_update = 125.72s`.
- `local_pcd_compaction_sec = 124.74s`.
- Current bounded pool solved export size by doing repeated online materialize + voxel downsample + cap.
- This is bounded, but it is not the right online data structure.

Hypothesis:

If object PCD memory is maintained as a voxel-indexed representative table, online updates become proportional to new patch voxels instead of current object point count.

Implementation:

Add an internal per-object pool state:

- `voxel_size`
- `voxel_key -> representative_point`
- `voxel_key -> count`
- `running_point_count`
- `running_sum`
- `bbox_min`
- `bbox_max`
- `revision`
- `dirty_materialized_local_pcd`
- `last_materialized_revision`
- `debug counters`

Online update:

1. Quantize accepted patch points with `downsample_voxel_size`.
2. Insert or update only touched voxel keys.
3. Update centroid and bbox incrementally.
4. Do not concatenate previous `obj.local_pcd`.
5. Keep `obj.local_pcd` as a materialized snapshot, not the primary online store.

Materialization:

- only for export;
- dense surface refresh if needed;
- debug/audit if explicitly enabled;
- optional low-frequency sanity snapshot.

Cap rule:

- If voxel table exceeds `max_points_per_object`, use deterministic spatial cap on keys or representative points.
- Cap must be stable across runs.

Debug counters:

- `local_pcd_voxel_pool_enabled`
- `local_pcd_voxel_key_count`
- `local_pcd_voxel_insert_count`
- `local_pcd_voxel_update_count`
- `local_pcd_materialize_count`
- `local_pcd_materialize_sec`
- `local_pcd_cap_applied_count`
- `local_pcd_snapshot_stale`

Acceptance:

- 20f smoke passes.
- 200f fast run has `mIoU >= 0.396`.
- `object_update_local_pcd_update < 45s`.
- `local_pcd_compaction_sec < 30s`.
- `final_local_memory_point_count_total <= 1,200,000`.
- Export artifacts are present and valid.

Stop condition:

- If object count changes by more than `3`, run a shadow-mode diff:
  - centroid delta;
  - bbox IoU;
  - local point count;
  - per-object semantic label.
- If dense surface or export uses stale `local_pcd`, add explicit materialization at export boundary instead of reverting the voxel pool.

## Experiment Line B: Association KDTree And Indexed Retrieval

Files:

- `src/modules/association.py`
- `src/modules/object_update.py`
- `src/core/data_structures.py` if revision fields are needed
- `tests/test_association.py`
- `tests/test_backend_runtime_fast.py`

Observed problem:

- `association = 143.36s`.
- Mean candidate score count: `411.545` per frame.
- Mean geometry score count: `76.04` per frame.
- Geometry pruning works, but object candidate retrieval and geometry NN setup still repeat too much work.

Hypothesis:

Association should query indexed active object memory. Rebuilding or scanning per patch is avoidable when object geometry revisions are unchanged.

Implementation:

1. Add object-level geometry revision:
   - association sketch update increments revision only when representative set changes.
2. Add cached geometry index:
   - `object_id -> revision -> cKDTree`
   - cache stores point count and bbox.
3. Add centroid/spatial candidate index:
   - cKDTree over active object centroids or bbox centers;
   - rebuilt once per frame or only when active object set changes.
4. Candidate flow:
   - semantic/anchor gate;
   - spatial radius or top-k nearest object gate;
   - only then geometry NN score.
5. Keep exact old scoring behind config for ablation:

```yaml
association:
  indexed_retrieval_enabled: true
  centroid_kdtree_enabled: true
  geometry_kdtree_cache_enabled: true
  geometry_kdtree_cache_min_points: 64
  max_spatial_candidates: 12
```

Debug counters:

- `association_centroid_index_build_sec`
- `association_centroid_index_query_sec`
- `association_geometry_kdtree_cache_hit_count`
- `association_geometry_kdtree_cache_miss_count`
- `association_geometry_kdtree_build_sec`
- `association_spatial_pruned_candidate_count`
- `association_geometry_query_sec`

Acceptance:

- 20f tests and smoke pass.
- 200f `association < 90s`.
- `mIoU >= 0.396`.
- `final_object_count` within `57 +/- 3`.
- `association_pruning_summary` shows reduced geometry score count without major raw match collapse.

Stop condition:

- If quality drops, log disagreement audit:
  - patch id;
  - old best object id;
  - indexed best object id;
  - semantic labels;
  - distances;
  - whether the correct candidate was pruned by spatial gate.

## Experiment Line C: Surface Owner Gate Representative Cache

Files:

- `src/modules/object_update.py`
- `src/modules/tsdf_instance_map.py`
- `tests/test_pipeline.py`
- `tests/test_backend_runtime_fast.py`

Observed problem:

- `object_update_surface_owner_gate = 156.17s`.
- It remains strongly tied to large object/point state.
- This stage is accuracy-sensitive because it controls contamination and structural rejection.

Hypothesis:

Most patches can be classified with a small representative voxel view and cached owner-support data. Full inverse/point mask materialization should be reserved for conflicted patches.

Implementation:

1. Representative patch voxel view:
   - cap patch representative voxels;
   - use TSDF voxel keys before point masks.
2. Owner-support cache:
   - key: `(frame_id bucket, patch voxel hash, tsdf revision)` or object/voxel revision;
   - value: owner histogram and background support summary.
3. Quick accept path:
   - high self-owner ratio;
   - low foreign/background ratio;
   - sufficient representative voxel count.
4. Escalation path:
   - ambiguous ownership;
   - structural class;
   - attached surface class;
   - low point count;
   - semantic conflict.
5. Keep existing full gate behavior for escalated patches.

Debug counters:

- `surface_gate_representative_voxel_count`
- `surface_gate_quick_accept_count`
- `surface_gate_escalated_count`
- `surface_gate_owner_cache_hit_count`
- `surface_gate_owner_cache_miss_count`
- `surface_gate_cache_query_sec`
- `surface_gate_full_inverse_sec`
- `surface_gate_accuracy_guard_reject_count`

Acceptance:

- 20f overlay/audit shows no obvious contamination regression.
- 200f `surface_owner_gate < 90s`.
- `mIoU >= 0.396`.
- Object count within `57 +/- 3`.
- Structural overlay and contamination audit counters remain interpretable.

Stop condition:

- If mIoU/object count drops, compare quick-accepted patches against old full gate on a sampled shadow path.
- Do not keep quick accept if it causes large object merges.

## Experiment Line D: Fast/Debug Profile Split

Files:

- `scripts/run_room0_checkpointed_eval.py`
- `run_room0_full_eval.py`
- configs
- tests

Observed problem:

- `runtime_vis = 88.26s`.
- It is useful for inspection, but it should not be included in the headline fast runtime.
- Its cost correlates with `raw_proposal_count` at `0.91`.

Implementation:

Add explicit profiles:

```yaml
benchmark_profile:
  name: fast
  runtime_vis_enabled: false
  runtime_vis_interval: 0
```

```yaml
benchmark_profile:
  name: debug_overlay
  runtime_vis_enabled: true
  runtime_vis_interval: 10
  runtime_vis_max_frames: 20
```

Runner rules:

- `fast` writes timing JSON, reports, final eval artifacts.
- `debug_overlay` writes overlay JSON/images for selected frames.
- Reports must include `benchmark_profile`.
- The plan and final analysis must not mix profile timings.

Acceptance:

- Fast 200f run has `runtime_vis <= 5s`.
- Debug overlay run produces human-readable overlays.
- Both runs record profile metadata.

Stop condition:

- If disabling runtime vis removes data needed for audit, keep minimal JSON counters in fast and reserve images for debug.

## Experiment Line E: Checkpoint Size Reduction

Files:

- `run_room0_full_eval.py`
- `scripts/run_room0_checkpointed_eval.py`
- `src/core/data_structures.py`
- tests

Observed problem:

- `mapping_state_size_bytes = 1.08GB`.
- This is too large for repeatable experiments and slows save/export.

Hypothesis:

The checkpoint stores more than the minimal recoverable mapping state, likely including dense `geometry_accum`, full observations, debug structures, or patch-level point arrays.

Implementation:

1. Add checkpoint size audit:
   - pickle-size estimate by major payload key;
   - object observations count and approximate bytes;
   - geometry accumulator point count;
   - debug payload size estimate.
2. Split checkpoint:
   - `mapping_state.pkl`: minimal state for export/resume;
   - `geometry_accum.npz`: dense fused geometry;
   - `frame_metrics.jsonl`: already separate;
   - optional `debug_state.pkl` for full audit.
3. Observation compaction:
   - remove or compress full patch points from stored observations unless required for export;
   - keep frame id, semantic metadata, centroid, bbox, point count, representative geometry id.

Acceptance:

- `mapping_state_size_bytes < 500MB`.
- Export from checkpoint still works.
- `mIoU` and artifacts unchanged.

Stop condition:

- If exact export depends on full stored observations, split them to a separate optional artifact instead of deleting.

## Execution Order

Implement in this order:

1. Line D: fast/debug split.
   - This gives clean timing before deeper algorithm changes.
2. Line A: incremental voxel PCD pool.
   - Largest direct online object-update win.
3. Line B: association KDTree/indexed retrieval.
   - Independent major win after object memory has stable revisions.
4. Line C: surface owner gate representative cache.
   - Accuracy-sensitive, do after A/B instrumentation is stable.
5. Line E: checkpoint size reduction.
   - Can be partly parallel, but should not block runtime algorithm validation.

Do not merge A+B+C in one code pass. Each line needs:

- focused unit tests;
- 5f or 20f smoke;
- 200f stride=10 fast run;
- result comparison against `20260614_bounded_pool_s10_200f`.

## Required Validation Commands

Targeted tests:

```bash
docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python -m pytest tests/test_association.py tests/test_backend_runtime_fast.py tests/test_dual_map.py tests/test_pipeline.py::TestObjectUpdateModule -q'
```

Fast smoke:

```bash
docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python scripts/run_room0_checkpointed_eval.py --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml --output-root outputs/tmp_validation/<run_name> --num-frames 20 --frame-stride 10 --proposal-backend sam2 --proposal-device cuda --sam-version 2.1 --sam-ckpt-path data/input/sam_ckpts --fast-eval --quiet'
```

Fast 200f:

```bash
nohup docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python scripts/run_room0_checkpointed_eval.py --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml --output-root outputs/tmp_validation/<run_name> --num-frames 200 --frame-stride 10 --proposal-backend sam2 --proposal-device cuda --sam-version 2.1 --sam-ckpt-path data/input/sam_ckpts --fast-eval --quiet' > outputs/tmp_validation/<run_name>.nohup.log 2>&1 &
```

## Final Decision Rule

Accept the next system version only if:

- online runtime decreases for the right reason, visible in stage timings;
- no work is secretly shifted into unbounded finalization/export;
- memory stays bounded;
- quality remains within the accepted band;
- fast/debug profiles are not mixed in the headline comparison.

The paper claim should be:

> We accelerate open-vocabulary mapping by replacing repeated dense object-memory scans with bounded indexed dual memory: TSDF for online spatial ownership and object voxel-PCD pools for semantic geometry and retrieval.

