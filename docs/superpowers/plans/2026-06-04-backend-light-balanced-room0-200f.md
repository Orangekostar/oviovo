# Backend Light Balanced Room0 200f Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backend-light Balanced profile that reduces `association`, `object_update`, and `dense_surface` cost, then run room0 200f to measure speed and accuracy against the current Balanced online baseline.

**Architecture:** Keep the current Balanced frontend unchanged: YOLOWorld anchor-box fast path on most frames and full SAM2 every 10 frames. Add bounded backend work in three local modules: cap active-set candidates before association, cap scored association candidates before expensive geometry checks, and throttle dense surface refreshes. Use a separate config for the experiment so the current baseline remains unchanged.

**Tech Stack:** Python, YAML, pytest, existing `Pipeline`, `ActiveSetModule`, `AssociationModule`, `ObjectUpdateModule`, `DenseSurfaceModule`, and `scripts/run_room0_checkpointed_eval.py`.

---

## Baseline To Beat

Use the completed run:

- Output: `outputs/tmp_validation/room0_balanced_yoloworld_sam_s10_200f_postfix`
- Frames: `200`, stride `10`
- Mapping: `845.039s`
- Time/frame: `4.225s`
- Accuracy: `mIoU=0.2785`, `f-mIoU=0.3850`, `mAcc=0.3733`, `f-mAcc=0.5095`
- Main stage means:
  - `object_update`: `1.9980s`
  - `association`: `1.1726s`
  - `dense_surface`: `0.1939s`
  - `active_set`: `0.1757s`
  - `proposal_generation`: `0.1523s`

Target for this plan:

- Reduce total time/frame from `4.225s` to below `2.5s` in room0 200f.
- Keep `f-mIoU` above `0.365` and `mIoU` above `0.260`.
- Confirm SAM2 still runs exactly `20/200` frames.

## File Structure

- Modify: `src/modules/active_set.py`
  - Add deterministic caps for active-set candidate IDs.
  - Preserve visible instances first, then new candidates, nearby instances, and whole-prior instances.

- Modify: `src/modules/association.py`
  - Add `max_scored_candidates`.
  - Score only the best cheap candidates plus vote owners.
  - Keep existing `max_geometry_candidates` as the geometry-consistency sub-cap.

- Modify: `src/modules/dense_surface.py`
  - Add `update_interval`.
  - Refresh dense point surfaces only for current-frame objects on interval frames.
  - Keep semantic-label refresh cheap on skipped frames.

- Add: `configs/replica_yoloworld_sam_backend_light_balanced_4090.yaml`
  - Derived from `configs/replica_yoloworld_sam_balanced_online_4090.yaml`.
  - Enables backend caps and lower memory bounds.

- Modify: `tests/test_pipeline.py`
  - Unit tests for active-set caps, association scored-candidate caps, and dense-surface interval behavior.

## Task 1: Active Set Candidate Caps

**Files:**
- Modify: `src/modules/active_set.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing active-set cap test**

Add this test to `tests/test_pipeline.py` near active-set or association tests:

```python
def test_active_set_caps_candidates_with_visible_priority():
    module = ActiveSetModule(
        {
            "nearby_radius": 100.0,
            "whole_prior_threshold": 0.1,
            "max_candidate_ids": 5,
            "max_nearby_ids": 3,
            "max_whole_prior_ids": 2,
        }
    )
    state = SystemState()
    for object_id in range(10):
        obj = ObjectMap(
            object_id=object_id,
            local_pcd=np.array([[float(object_id), 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([float(object_id) + 0.1, 0.1, 1.1], dtype=np.float32),
            whole_evidence=WholeEvidenceScores(whole_evidence_score=0.9),
        )
        state.objects[object_id] = obj

    module.tsdf_module.query_visible_instances = lambda volume, pose, intrinsics: {8, 9}

    active = module.process(
        state,
        np.eye(4, dtype=np.float64),
        CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8),
    )

    assert active.visible_ids == {8, 9}
    assert len(active.nearby_ids) <= 3
    assert len(active.whole_prior_ids) <= 2
    assert len(active.all_candidate_ids) <= 5
    assert {8, 9}.issubset(active.all_candidate_ids)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::test_active_set_caps_candidates_with_visible_priority -q
```

Expected: FAIL because `ActiveSetModule` ignores `max_candidate_ids`, `max_nearby_ids`, and `max_whole_prior_ids`.

- [ ] **Step 3: Implement active-set caps**

In `src/modules/active_set.py`, add config reads in `__init__`:

```python
self.max_candidate_ids = int(config.get("max_candidate_ids", 0))
self.max_nearby_ids = int(config.get("max_nearby_ids", 0))
self.max_whole_prior_ids = int(config.get("max_whole_prior_ids", 0))
```

Add helper methods:

```python
@staticmethod
def _take_sorted(ids: Set[int], limit: int) -> Set[int]:
    if limit <= 0 or len(ids) <= limit:
        return set(ids)
    return set(sorted(int(item) for item in ids)[: int(limit)])

def _cap_active_set(self, active_set: ActiveSet) -> ActiveSet:
    visible_ids = set(active_set.visible_ids)
    nearby_ids = self._take_sorted(set(active_set.nearby_ids), self.max_nearby_ids)
    whole_prior_ids = self._take_sorted(set(active_set.whole_prior_ids), self.max_whole_prior_ids)
    new_ids = set(active_set.new_object_candidate_ids)
    if self.max_candidate_ids <= 0:
        return ActiveSet(
            visible_ids=visible_ids,
            nearby_ids=nearby_ids,
            whole_prior_ids=whole_prior_ids,
            new_object_candidate_ids=new_ids,
        )

    selected: list[int] = []
    for group in (visible_ids, new_ids, nearby_ids, whole_prior_ids):
        for object_id in sorted(int(item) for item in group):
            if object_id not in selected:
                selected.append(object_id)
            if len(selected) >= self.max_candidate_ids:
                break
        if len(selected) >= self.max_candidate_ids:
            break
    selected_set = set(selected)
    return ActiveSet(
        visible_ids=visible_ids & selected_set,
        nearby_ids=nearby_ids & selected_set,
        whole_prior_ids=whole_prior_ids & selected_set,
        new_object_candidate_ids=new_ids & selected_set,
    )
```

Then wrap the return in `process()`:

```python
active_set = self._cap_active_set(active_set)
```

- [ ] **Step 4: Run active-set test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::test_active_set_caps_candidates_with_visible_priority -q
```

Expected: PASS.

## Task 2: Association Scored-Candidate Cap

**Files:**
- Modify: `src/modules/association.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing association cap test**

Add this test near existing association pruning tests:

```python
def test_association_caps_scored_candidates_before_expensive_scoring():
    module = AssociationModule(
        {
            "match_threshold": 0.1,
            "max_scored_candidates": 4,
            "max_geometry_candidates": 2,
            "geometry_candidate_min_cheap_score": 0.0,
            "score_parallel_enabled": False,
        }
    )
    patch = Patch3D(
        patch_id=1,
        points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
    )
    objects = {}
    for object_id in range(20):
        objects[object_id] = ObjectMap(
            object_id=object_id,
            local_pcd=np.array([[float(object_id), 0.0, 1.0]], dtype=np.float32),
            association_pcd=np.array([[float(object_id), 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([float(object_id), 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([float(object_id) + 0.1, 0.1, 1.1], dtype=np.float32),
        )
    active = ActiveSet(nearby_ids=set(objects))

    result = module.process([patch], objects, TSDFInstanceVolume(), active_set=active)
    debug = result.debug["per_patch"][1]

    assert debug["candidate_count_before_scored_cap"] == 20
    assert debug["candidate_count_after_scored_cap"] == 4
    assert len(debug["candidate_object_ids"]) == 4
    assert len(result.scores) == 4
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::test_association_caps_scored_candidates_before_expensive_scoring -q
```

Expected: FAIL because association currently scores all active candidate IDs.

- [ ] **Step 3: Implement scored-candidate cap**

In `AssociationModule.__init__`, add:

```python
self.max_scored_candidates = int(config.get("max_scored_candidates", 0))
```

Add helper:

```python
def _select_scored_candidate_ids(
    self,
    candidate_ids: List[int],
    cheap_by_id: dict[int, tuple[float, float, float, float]],
    vote: VoxelVoteResult,
) -> List[int]:
    valid_ids = [int(obj_id) for obj_id in candidate_ids if int(obj_id) in cheap_by_id]
    if self.max_scored_candidates <= 0 or len(valid_ids) <= self.max_scored_candidates:
        return valid_ids
    vote_owner_ids = {int(obj_id) for obj_id in vote.owner_votes}
    ranked = sorted(
        valid_ids,
        key=lambda obj_id: (
            1 if obj_id in vote_owner_ids else 0,
            cheap_by_id[obj_id][3],
            cheap_by_id[obj_id][0],
            -obj_id,
        ),
        reverse=True,
    )
    return ranked[: self.max_scored_candidates]
```

In `process()`, after `_select_geometry_candidate_ids(...)`, add:

```python
scored_candidate_ids = self._select_scored_candidate_ids(list(candidate_ids), cheap_by_id, vote)
patch_debug["candidate_count_before_scored_cap"] = int(len(candidate_ids))
patch_debug["candidate_count_after_scored_cap"] = int(len(scored_candidate_ids))
patch_debug["candidate_object_ids"] = list(scored_candidate_ids)
```

Then call `_score_candidate_objects()` with `scored_candidate_ids` instead of `candidate_ids`.

Update summary:

```python
"max_scored_candidates": int(self.max_scored_candidates),
```

- [ ] **Step 4: Run association cap test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::test_association_caps_scored_candidates_before_expensive_scoring -q
```

Expected: PASS.

## Task 3: Dense Surface Refresh Interval

**Files:**
- Modify: `src/modules/dense_surface.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing dense-surface interval test**

Add this test near dense surface or pipeline timing tests:

```python
def test_dense_surface_skips_point_refresh_on_non_interval_frames():
    module = DenseSurfaceModule({"dense_surface_voxel": 0.02, "dense_surface_cap_per_object": 4000, "update_interval": 5})
    state = SystemState()
    obj = ObjectMap(
        object_id=7,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
        last_seen_frame=1,
    )
    patch = Patch3D(
        patch_id=3,
        points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
        centroid=np.array([0.05, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_max=np.array([0.1, 0.0, 1.0], dtype=np.float32),
        source_frame_id=1,
    )
    obj.observations.append(ObservationRecord(frame_id=1, patch=patch))
    state.objects[7] = obj

    module.process(state, frame_id=1)
    assert 7 not in state.dense_surface_map.entries

    module.process(state, frame_id=5)
    assert 7 in state.dense_surface_map.entries
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::test_dense_surface_skips_point_refresh_on_non_interval_frames -q
```

Expected: FAIL because `DenseSurfaceModule` refreshes point surfaces every frame.

- [ ] **Step 3: Implement dense-surface interval**

In `DenseSurfaceModule.__init__`, add:

```python
self.update_interval = max(1, int(config.get("dense_surface_update_interval", config.get("update_interval", 1)) or 1))
```

At the top of `process()` add:

```python
refresh_points_this_frame = int(frame_id) % int(self.update_interval) == 0
```

Change the loop:

```python
for object_id, obj in state.objects.items():
    if not self._has_current_frame_observation(obj, frame_id):
        self._refresh_semantic_label(state.dense_surface_map, obj)
        continue
    if not refresh_points_this_frame:
        self._refresh_semantic_label(state.dense_surface_map, obj)
        continue
```

- [ ] **Step 4: Run dense-surface test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::test_dense_surface_skips_point_refresh_on_non_interval_frames -q
```

Expected: PASS.

## Task 4: Add Backend-Light Balanced Config

**Files:**
- Add: `configs/replica_yoloworld_sam_backend_light_balanced_4090.yaml`

- [ ] **Step 1: Create config from Balanced online config**

Run:

```bash
cp configs/replica_yoloworld_sam_balanced_online_4090.yaml configs/replica_yoloworld_sam_backend_light_balanced_4090.yaml
```

Then edit these sections:

```yaml
active_set:
  nearby_radius: 3.0
  whole_prior_threshold: 0.45
  max_candidate_ids: 64
  max_nearby_ids: 48
  max_whole_prior_ids: 16

association:
  centroid_distance_weight: 0.4
  bbox_overlap_weight: 0.3
  geometry_overlap_weight: 0.2
  pose_proximity_weight: 0.1
  match_threshold: 0.5
  max_scored_candidates: 32
  max_geometry_candidates: 6
  geometry_candidate_min_cheap_score: 0.04
  geometry_vote_owner_priority: true
  geometry_nn_patch_sample: 48
  geometry_nn_object_sample: 48
  score_parallel_enabled: true
  score_parallel_workers: 4
  score_parallel_min_candidates: 16
  observation_identity_gate_enabled: true
  observation_identity_min_patch_confidence: 0.35
  semantic_conflict_gate_enabled: true
  semantic_conflict_min_patch_confidence: 0.35
  semantic_conflict_min_object_confidence: 0.35
  semantic_conflict_subregion_point_ratio: 0.35
  semantic_conflict_subregion_volume_ratio: 0.35

object_update:
  contested_residual_pool:
    enabled: true
  downsample_voxel_size: 0.02
  downsample_interval: 5
  max_points_per_object: 12000
  association_geometry:
    enabled: true
    voxel_size: 0.04
    max_points_per_object: 768
  surface_owner_gate:
    enabled: true
    representative_voxel_mode: false
    background_enabled: true
    min_accept_points: 20
    min_update_accept_ratio: 0.30
    min_new_object_accept_ratio: 0.55
    max_foreign_owner_ratio: 0.25
    max_background_owner_ratio: 0.35
    attached_surface_classes: [switch, wall-plug, vent]
    attached_max_voxels: 40
    self_background_min_score: 0.65
    self_background_margin: 0.15
  current_frame_visibility_gate:
    enabled: true
    distance_threshold: 0.05
    min_accept_points: 20
    min_accept_ratio: 0.30
  provisional_pool:
    enabled: true
    match_distance: 0.35
    promotion_hits: 3
    max_idle_frames: 30
    downsample_voxel_size: 0.03
    max_points_per_object: 3000
    unanchored_objectness_min: 0.75
    unanchored_min_points: 128

dual_map:
  enabled: false
  active_radius: 3.0
  warm_ttl_frames: 60
  cold_ttl_frames: 180
  dense_surface_voxel: 0.03
  dense_surface_cap_per_object: 2000
  dense_surface_update_interval: 5
```

- [ ] **Step 2: Validate YAML config**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import yaml
from pathlib import Path
path = Path("configs/replica_yoloworld_sam_backend_light_balanced_4090.yaml")
payload = yaml.safe_load(path.read_text())
assert payload["pipeline"]["yoloworld_sam_balanced_frontend_enabled"] is True
assert payload["active_set"]["max_candidate_ids"] == 64
assert payload["association"]["max_scored_candidates"] == 32
assert payload["association"]["max_geometry_candidates"] == 6
assert payload["object_update"]["association_geometry"]["max_points_per_object"] == 768
assert payload["dual_map"]["dense_surface_update_interval"] == 5
print("backend-light config ok")
PY
```

Expected: prints `backend-light config ok`.

## Task 5: Focused Regression Tests

**Files:**
- No code edits expected.

- [ ] **Step 1: Run backend-light targeted tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::test_active_set_caps_candidates_with_visible_priority \
  tests/test_pipeline.py::test_association_caps_scored_candidates_before_expensive_scoring \
  tests/test_pipeline.py::test_dense_surface_skips_point_refresh_on_non_interval_frames \
  -q
```

Expected: `3 passed`.

- [ ] **Step 2: Run existing relevant regression tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_association_limits_geometry_consistency_to_top_k_candidates \
  tests/test_pipeline.py::TestPipeline::test_association_geometry_top_k_prioritizes_tsdf_vote_owner \
  tests/test_pipeline.py::TestPipeline::test_association_restricts_candidates_to_active_set \
  tests/test_pipeline.py::TestPipeline::test_object_update_refreshes_bounded_association_geometry_memory \
  tests/test_pipeline.py::TestPipeline::test_pipeline_balanced_yoloworld_sam_runs_sam_only_on_interval_frames \
  -q
```

Expected: all pass.

## Task 6: Run Room0 200f Backend-Light Benchmark

**Files:**
- No code edits expected.

- [ ] **Step 1: Run 20f smoke**

Run:

```bash
rm -rf outputs/tmp_validation/room0_backend_light_balanced_s10_20f_smoke
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_room0_checkpointed_eval.py \
  --mode build-export \
  --scene-name room0 \
  --experiment-name room0_backend_light_balanced_s10_20f_smoke \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/replica_yoloworld_sam_backend_light_balanced_4090.yaml \
  --output-root outputs/tmp_validation \
  --num-frames 20 \
  --frame-stride 10 \
  --fast-eval \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-version 2.1 \
  --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --points-per-side 16 \
  --max-proposals 50 \
  --confidence-threshold 0.5 \
  --min-mask-area 100 \
  --stability-score-th 0.95 \
  --nms-iou-th 0.8 \
  --min-mask-region-area 100 \
  --quiet
```

Expected: `outputs/tmp_validation/room0_backend_light_balanced_s10_20f_smoke/status.json` reports `complete`, with SAM2 `2/20` frames.

- [ ] **Step 2: Run 200f benchmark**

Run:

```bash
rm -rf outputs/tmp_validation/room0_backend_light_balanced_s10_200f
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_room0_checkpointed_eval.py \
  --mode build-export \
  --scene-name room0 \
  --experiment-name room0_backend_light_balanced_s10_200f \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/replica_yoloworld_sam_backend_light_balanced_4090.yaml \
  --output-root outputs/tmp_validation \
  --num-frames 200 \
  --frame-stride 10 \
  --fast-eval \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-version 2.1 \
  --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --points-per-side 16 \
  --max-proposals 50 \
  --confidence-threshold 0.5 \
  --min-mask-area 100 \
  --stability-score-th 0.95 \
  --nms-iou-th 0.8 \
  --min-mask-region-area 100 \
  --quiet
```

Expected: `outputs/tmp_validation/room0_backend_light_balanced_s10_200f/status.json` reports `complete`.

- [ ] **Step 3: Summarize benchmark**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

def load(name):
    root = Path("outputs/tmp_validation") / name
    scene = root / "room0"
    mapping = json.loads((root / "mapping_timer_result.json").read_text())
    export = json.loads((root / "export_eval_timer_result.json").read_text())
    report = json.loads((scene / "run_report.json").read_text())
    rows = [json.loads(line) for line in (scene / "frame_metrics.jsonl").read_text().splitlines() if line.strip()]
    return root, mapping, export, report, rows

baseline_root, baseline_mapping, baseline_export, baseline_report, baseline_rows = load(
    "room0_balanced_yoloworld_sam_s10_200f_postfix"
)
light_root, light_mapping, light_export, light_report, light_rows = load("room0_backend_light_balanced_s10_200f")
for label, mapping, export, report, rows in [
    ("baseline", baseline_mapping, baseline_export, baseline_report, baseline_rows),
    ("backend_light", light_mapping, light_export, light_report, light_rows),
]:
    timings = report.get("stage_timing_summary", {})
    print(label)
    print("sec_per_frame", mapping["sec_per_frame"])
    print("mapping_sec", mapping["mapping_loop_sec_excluding_init_and_final_outputs"])
    print("miou", export["miou"])
    print("fmiou", export["fmiou"])
    print("macc", export["macc"])
    print("fmacc", export["fmacc"])
    print("sam_ran_count", sum(float(r.get("stage_timings", {}).get("sam2_full_frame_ran", 0) or 0) > 0.5 for r in rows))
    for key in ["proposal_generation", "active_set", "association", "object_update", "dense_surface"]:
        if key in timings:
            item = timings[key]
            print(key, item["mean_sec"], item["median_sec"], item["p95_sec"])
    print()
PY
```

Expected: prints baseline and backend-light metrics.

## Task 7: Decide Next Backend Direction

**Files:**
- No code edits expected.

- [ ] **Step 1: Apply success criteria**

Use this decision table:

```text
If sec_per_frame <= 2.5 and fmiou >= 0.365:
  Keep backend-light config as the next default experimental profile.

If sec_per_frame <= 2.5 and fmiou < 0.365:
  Relax caps in this order:
    max_scored_candidates: 32 -> 48
    max_geometry_candidates: 6 -> 8
    association_geometry.max_points_per_object: 768 -> 1024

If sec_per_frame > 2.5 and association remains > 0.7s:
  Add spatial index / frustum-projected candidate prefilter before scoring.

If sec_per_frame > 2.5 and object_update remains > 1.0s:
  Plan dirty-object update and background support caching as the next implementation.

If dense_surface remains > 0.12s:
  Move dense_surface to every 10 frames or export-only.
```

- [ ] **Step 2: Final report**

Report in Chinese:

- New backend-light logic.
- Files changed.
- Commands run.
- Unit/smoke/200f results.
- Comparison against `room0_balanced_yoloworld_sam_s10_200f_postfix`.
- Recommendation for the next optimization.

## Self-Review

- Spec coverage: active-set cap, association scored-candidate cap, dense-surface interval, backend-light config, tests, smoke run, 200f run, and comparison are covered.
- Placeholder scan: no placeholders or open-ended implementation instructions remain.
- Type consistency: config keys match planned code fields: `max_candidate_ids`, `max_nearby_ids`, `max_whole_prior_ids`, `max_scored_candidates`, `dense_surface_update_interval`.
- Scope check: this plan is one bounded backend-light experiment. It intentionally does not implement full async backend or dirty-object update; those are follow-up decisions after 200f evidence.
