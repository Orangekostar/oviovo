# Bounded Dual-Memory Runtime And Benchmark Plan

> For agentic workers: use subagent-driven development for implementation and review. This plan follows the 200f stride=10 diagnostic run `20260614_structural_v2_fast_s10_200f`. Do not present unbounded lazy local-pcd as the final real-time method. It is an ablation that proves the bottleneck and motivates bounded object memory.

## Goal

Build the final backend story for a TSDF voxel map + object-level PCD pool open-vocabulary mapping system:

- TSDF voxel map is the online spatial substrate for ownership, visibility, support, and association votes.
- Object PCD pool is a bounded auxiliary memory for semantic geometry, inspection, export, and evaluation.
- Online mapping must be fast without shifting unbounded work to final export/evaluation.

Target for Replica room0, 200 frames, stride 10:

- `mapping_loop_sec <= 730s`
- `export_eval_sec <= 180s`
- `final_local_memory_point_count_total <= 1,200,000`
- `mIoU >= 0.403`
- final object count within `57 +/- 3`

Preferred:

- `association_geometry_update <= 40s`
- `association <= 110s`
- end-to-end time lower than `prev_fast`

## Current Evidence

### Previous stable fast baseline

Run:

`outputs/tmp_validation/20260614_structural_fast_s10_200f_fasteval/20260614063946_room0_checkpointed_s10_200f`

- `mapping_loop_sec`: `875.53s`
- `process_frame_sec_total`: `842.91s`
- `export_eval_sec`: `129.27s`
- `mIoU`: `0.404964`
- `f-mIoU`: `0.361247`
- final objects: `57`
- `final_local_memory_point_count_total`: `691,517`
- `local_pcd_update`: `114.40s`
- `association_geometry_update`: `83.62s`
- `association`: `164.78s`
- `object_update`: `391.27s`

### Naive lazy local-pcd diagnostic run

Run:

`outputs/tmp_validation/20260614_structural_v2_fast_s10_200f/20260614073450_room0_checkpointed_s10_200f`

- `mapping_loop_sec`: `702.64s`
- `process_frame_sec_total`: `671.54s`
- `export_eval_sec`: `671.35s`
- `mIoU`: `0.408156`
- `f-mIoU`: `0.373338`
- final objects: `57`
- `final_local_memory_point_count_total`: `85,371,124`
- `local_pcd_update`: `0.34s`
- `association_geometry_update`: `85.92s`
- `association`: `134.58s`
- `object_update`: `261.94s`

Interpretation:

- Removing full local-pcd maintenance from the online critical path works.
- Unbounded lazy accumulation is not valid as the final method because it shifts cost to export and creates a huge object memory.
- The final design must be lazy but bounded.

## Benchmark Contract

Report four timing buckets, not a single runtime number:

1. `online_mapping_sec`
   - Per-frame online critical path.
   - Includes frontend, runtime grouping, depth refinement, patch lifting, active set, association, TSDF update, online object state, semantic memory.
   - Excludes final map export, GT matching, and metric computation.

2. `finalization_sec`
   - Stream-end algorithmic work needed to make the map complete and bounded.
   - Includes deferred local-pcd compaction, sketch materialization, checkpoint preparation.
   - Must be bounded and reported.

3. `eval_io_sec`
   - PLY writing, GT mesh matching, metric script work.
   - Report separately because this is evaluator/IO cost, not online mapping.

4. `end_to_end_sec`
   - `online_mapping_sec + finalization_sec`.
   - Use this for fair system-level comparison.

Also report memory/backlog:

- `final_local_memory_point_count_total`
- `max_local_memory_point_count_total`
- `local_pcd_pending_point_peak`
- `local_pcd_compaction_count`
- `local_pcd_compaction_sec`
- `association_geometry_sketch_update_count`
- `association_geometry_sketch_materialize_count`
- `mapping_state_size_bytes`
- largest exported PLY size

Paper rule:

- The final method may claim online speedup only if memory/backlog is bounded and finalization does not explode.
- The naive lazy run should be shown as a diagnostic ablation, not the proposed method.

## Phase 1: Bounded Local PCD Pool

Files:

- `src/modules/object_update.py`
- `tests/test_pipeline.py`
- `configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml`

Problem:

- Naive lazy local-pcd reduces online `local_pcd_update` from `114.40s` to `0.34s`.
- But local memory grows from `691,517` to `85,371,124` points.
- Export/eval grows from `129.27s` to `671.35s`.

Implementation:

Add bounded compaction settings:

```yaml
object_update:
  local_pcd_chunk_pool:
    enabled: true
    bounded_compaction_enabled: true
    materialize_interval: 20
    max_pending_points: 50000
    compact_to_max_points: true
```

Online behavior:

1. Append accepted patch points to pending chunks.
2. Keep centroid and bbox incremental.
3. Do not concatenate every frame.
4. When `pending_point_count >= max_pending_points` or `update_count % materialize_interval == 0`:
   - materialize current object once;
   - voxel downsample;
   - deterministic spatial cap to `max_points_per_object`;
   - reset pending state.
5. Before checkpoint/export, flush each object and guarantee `len(obj.local_pcd) <= max_points_per_object`.

Required debug counters:

- `local_pcd_pending_point_count`
- `local_pcd_pending_point_peak`
- `local_pcd_compaction_count`
- `local_pcd_compaction_sec`
- `local_pcd_compacted_point_count_before`
- `local_pcd_compacted_point_count_after`
- `local_pcd_cap_applied_count`

Acceptance:

- `final_local_memory_point_count_total <= 1,200,000`
- `export_eval_sec <= 180s`
- `local_pcd_update` remains far below the eager baseline, target `< 30s`
- no NaN centroid/bbox
- final objects within `57 +/- 3`

Stop condition:

- If online `local_pcd_update` returns near `114s`, compaction is too frequent or too expensive.
- If export still writes multi-GB `room0_instance_map.ply`, the pool is not actually bounded.

## Phase 2: True Association Geometry Sketch

Files:

- `src/modules/object_update.py`
- `tests/test_pipeline.py`

Problem:

- Current sketch path did not reduce `association_geometry_update`:
  - previous: `83.62s`
  - latest: `85.92s`
- It still does too much patch representative/downsample/key work.

Implementation:

Add budgeted sketch settings:

```yaml
object_update:
  association_geometry:
    enabled: true
    sketch_enabled: true
    materialize_interval: 1000000
    patch_budget_points: 256
    update_budget_new_keys: 128
    max_points_per_object: 2048
```

Rules:

1. Association geometry must not rebuild from full `local_pcd` during online mapping.
2. Patch points are sampled before voxel representative extraction.
3. Each patch update processes at most `patch_budget_points`.
4. Each object update inserts at most `update_budget_new_keys`.
5. `association_pcd` remains `<= 2048`.
6. Flush materializes from the sketch table, not from full local-pcd.

Required debug counters:

- `association_geometry_sketch_update_count`
- `association_geometry_sketch_new_key_count`
- `association_geometry_sketch_dropped_key_count`
- `association_geometry_sketch_materialize_count`
- `association_geometry_sketch_update_sec`
- `association_geometry_sketch_materialize_sec`

Acceptance:

- `association_geometry_update <= 40s`
- `association_pcd` point count remains bounded
- association outcomes do not visibly drift
- `mIoU >= 0.403`

Stop condition:

- If `mIoU` drops more than `0.005`, revert budget first, not the whole sketch.

## Phase 3: Association Candidate Ablation

Files:

- `src/modules/association.py`
- `tests/test_association.py`
- fast config

Current latest:

- `association`: `134.58s`
- `candidate_score_count`: `82,309`
- `geometry_score_count`: `15,206`
- KDTree cache: `0 hit / 0 miss`

Candidate set A:

```yaml
association:
  top_k_final: 16
  top_k_geometry: 3
  max_scored_candidates: 16
  max_geometry_candidates: 4
```

Candidate set B:

```yaml
association:
  top_k_final: 12
  top_k_geometry: 2
  max_scored_candidates: 12
  max_geometry_candidates: 3
```

Rules:

- Keep TSDF owner-vote candidates strongly prioritized.
- Keep observation identity gate unchanged.
- Do not prune owner-vote candidates below the cheap-score threshold without explicit ablation.

Acceptance:

- `association <= 110s`
- `mIoU` drop <= `0.005`
- final object count within `57 +/- 3`

Stop condition:

- If object count drifts or overlay shows obvious association errors, revert to candidate set A.

## Phase 4: Runtime Reporting

Files:

- `scripts/run_room0_checkpointed_eval.py`
- `run_room0_full_eval.py`
- `tests/test_dual_map.py` or `tests/test_backend_runtime_fast.py`

Implementation:

Split the existing timing payload into:

```json
{
  "online_mapping_sec": 0.0,
  "finalization_sec": 0.0,
  "eval_io_sec": 0.0,
  "end_to_end_sec": 0.0,
  "final_local_memory_point_count_total": 0,
  "max_local_memory_point_count_total": 0,
  "local_pcd_pending_point_peak": 0,
  "local_pcd_compaction_count": 0,
  "local_pcd_compaction_sec": 0.0,
  "association_geometry_sketch_update_count": 0,
  "association_geometry_sketch_materialize_count": 0,
  "mapping_state_size_bytes": 0,
  "largest_export_size_bytes": 0
}
```

Acceptance:

- `mapping_timer_result.json`, `export_eval_timer_result.json`, and `run_report.json` expose the new fields.
- Previous timing fields remain available for compatibility.
- 20f smoke writes all new fields.

## Experiment Matrix

1. `bounded_pool_smoke_s10_20f`
   - Phase 1 only.
   - Validate export does not explode.

2. `bounded_pool_s10_200f`
   - Phase 1 only.
   - Main check: local-pcd bounded, export back near baseline.

3. `bounded_pool_assoc_sketch_s10_200f`
   - Phase 1 + Phase 2.
   - Main check: `association_geometry_update` drops.

4. `bounded_pool_assoc_cap_s10_200f`
   - Phase 1 + Phase 2 + candidate set B.
   - Main check: association speed/accuracy tradeoff.

5. `bounded_pool_overlay_check_s10_50f`
   - Final config, runtime-vis debug enabled.
   - Inspect overlay for obvious association/semantic drift.

## Suggested Commands

Focused tests:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates &&
python -m pytest \
  tests/test_association.py \
  tests/test_backend_runtime_fast.py \
  tests/test_pipeline.py::TestObjectUpdateModule::test_object_update_chunk_pool_defers_local_pcd_concat_until_flush \
  tests/test_pipeline.py::TestObjectUpdateModule::test_object_update_chunk_pool_does_not_materialize_when_total_exceeds_max_points \
  tests/test_pipeline.py::TestObjectUpdateModule::test_association_geometry_sketch_materializes_on_interval_and_flush \
  -q
'
```

20f smoke:

```bash
docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates &&
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml \
  --output-root outputs/tmp_validation/20260614_bounded_pool_smoke_s10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-version 2.1 \
  --sam-ckpt-path data/input/sam_ckpts \
  --fast-eval \
  --quiet
'
```

200f run:

```bash
nohup docker exec ww-ai bash -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates &&
python scripts/run_room0_checkpointed_eval.py \
  --config-path configs/replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml \
  --output-root outputs/tmp_validation/20260614_bounded_pool_s10_200f \
  --num-frames 200 \
  --frame-stride 10 \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-version 2.1 \
  --sam-ckpt-path data/input/sam_ckpts \
  --fast-eval \
  --quiet
' > outputs/tmp_validation/20260614_bounded_pool_s10_200f.nohup.log 2>&1 &
```

## Expected Paper Story

Do not claim:

- "We defer local-pcd update to eval, therefore real-time improves."

Claim:

- "Full object-level point-cloud maintenance is not required on the online critical path."
- "Naive unbounded lazy accumulation only shifts cost to export."
- "The proposed bounded dual-memory design keeps online state queryable while bounding object memory and finalization cost."

Benchmark table snapshot:

Full table and LaTeX draft: `docs/superpowers/reports/2026-06-15-paper-benchmark-table.md`

| Method | Online s/frame | Online total s | Finalization s | Eval/IO s | mIoU | f-mIoU | Final local points | Peak local points | Backlog bounded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eager local-PCD baseline | 4.378 | 875.53 | n/a | 129.27 | 0.405 | 0.361 | 691,517 | n/a | Yes |
| Naive lazy local-PCD diagnostic | 3.513 | 702.64 | n/a | 671.35 | 0.408 | 0.373 | 85,371,124 | n/a | No |
| Bounded local-PCD pool | 4.004 | 800.74 | 3.45 | 135.42 | 0.406 | 0.365 | 370,723 | 1,241,120 | Yes |
| Bounded pool + association index | 3.389 | 677.76 | 4.05 | 124.40 | 0.406 | 0.365 | 370,723 | 1,241,120 | Yes |

Existing-method local baselines:

| Method | Scene | Frames | Runtime s/frame | mIoU | f-mIoU | Status |
|---|---|---:|---:|---:|---:|---|
| DualMap | Replica room0 | 200, stride 10 | 0.242 | 0.276 | 0.716 | Available, different runtime contract |
| HOV-SG | Replica office0 | n/a | n/a | 0.222 | 0.298 | Existing result is office0 only |
| OVI-MAP official | Replica room0 | target 200, stride 10 | pending | pending | pending | Environment imports; missing CropFormer panoptic masks |
| OVO / OVOMap | Replica room0 | target 200, stride 10 | pending | pending | pending | No completed local result found |
| ConceptGraphs | Replica room0 | target 200, stride 10 | pending | pending | pending | No completed local result found |

Final method is acceptable only if it wins online time without unbounded memory or finalization explosion.
