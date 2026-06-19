# Fast High-IoU Association and Delayed Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the 0526 high-coverage room0 behavior while reducing runtime by bounding expensive association geometry and preventing semantically blocked residuals from staying unlabeled forever when later direct anchor evidence appears.

**Architecture:** Keep the 0526 `room0_surface_gate_4090.yaml` style frontend as the accuracy baseline: full SAM proposals plus YOLO/YOLOE semantic voting, not the current over-conservative `anchor_guided_sam` main path. Add a two-stage association scorer that uses cheap TSDF/centroid/bbox evidence to select a small top-k set before nearest-neighbor geometry. Maintain a bounded per-object `association_pcd` for matching while preserving `local_pcd` as the higher-detail export pool, and gate residual semantics so blocked labels are delayed rather than committed or permanently lost.

**Tech Stack:** Python 3, NumPy, pytest, YAML configs, existing OVIOVO modules (`AssociationModule`, `ObjectUpdateModule`, `PatchLiftingModule`, `SemanticMemoryModule`, `Pipeline`, `run_room0_full_eval.py`).

---

## Evidence This Plan Responds To

- Latest fast/no-debug 2000f run: `outputs/tmp_validation/20260531_room0_anchor_guided_sam_fast_stride1_2000f_gpu`
- Metrics: mIoU `0.3682`, f-mIoU `0.4432`, FPS `0.1354`, wall time about `14765.7s`
- 0526 high-coverage baseline: `outputs/tmp_validation/20260526_room0_observation_first_2b830e1_stride1_2000f`
- 0526 metrics: mIoU `0.6050`, f-mIoU `0.7531`
- Current bottlenecks: association `5714.0s`, object_update `2082.9s`, runtime_vis `2204.7s`, proposal_generation `3109.8s`
- Current precision failure: final audit has `29` committed objects and `105` unlabeled objects; blanket/candle/plant-stand/pillar/wall-plug are at IoU `0.0`

## File Structure

- Modify `src/modules/association.py`
  - Add cheap scoring and `max_geometry_candidates` top-k geometry pruning.
  - Use `ObjectMap.association_pcd` when available for nearest-neighbor geometry.
  - Add per-frame debug summary counts for candidate pruning.
- Modify `src/core/data_structures.py`
  - Add `ObjectMap.association_pcd` as bounded geometry for association only.
- Modify `src/modules/object_update.py`
  - Maintain `association_pcd` after object create/update/provisional promotion/refinement replacement.
  - Keep `local_pcd` semantics unchanged for exports.
  - Preserve delayed semantic evidence in `_refresh_object_debug()`.
- Modify `src/modules/patch_lifting.py`
  - Preserve `semantic_commit_allowed`, `residual_semantic_policy`, and `mask_anchor_relation` metadata into `Patch3D`.
- Modify `src/modules/semantic_memory.py`
  - Treat `semantic_commit_allowed=False` as delayed semantic evidence, not direct commit evidence.
  - Allow later direct strong anchor observations to commit/export normally.
- Modify `run_room0_full_eval.py`
  - Record compact association pruning summaries in `frame_metrics.jsonl`.
  - Include aggregate association pruning totals in `run_report.json`/`run_report.md`.
- Create `configs/room0_surface_gate_fast_high_iou_4090.yaml`
  - Base on the 0526 surface-gate configuration.
  - Enable association top-k pruning and association representative geometry.
  - Keep `anchor_guided_sam` disabled so the run preserves 0526-style semantic coverage.
- Modify `tests/test_pipeline.py`
  - Add association pruning tests.
  - Add association representative geometry tests.
  - Add delayed semantic commit and metadata preservation tests.
- Modify `tests/test_provisional_pool.py`
  - Add one end-to-end object-update test for blocked residual first, later strong direct anchor.

## Task 1: Association Two-Stage Top-K Geometry

**Files:**
- Modify: `src/modules/association.py`
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for geometry pruning**

Add these methods inside the association test area in `tests/test_pipeline.py`, near `test_association_restricts_candidates_to_active_set`:

```python
    def test_association_limits_geometry_consistency_to_top_k_candidates(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "voxel_vote_weight": 0.0,
                "centroid_distance_weight": 0.7,
                "bbox_overlap_weight": 0.1,
                "geometry_overlap_weight": 0.2,
                "max_geometry_candidates": 2,
            }
        )
        patch_points = np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=10,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        objects = {}
        for obj_id, offset in [(1, 0.01), (2, 0.20), (3, 1.20), (4, 1.60), (5, 2.00)]:
            pts = patch_points + np.array([offset, 0.0, 0.0], dtype=np.float32)
            objects[obj_id] = ObjectMap(
                object_id=obj_id,
                local_pcd=pts.copy(),
                centroid=pts.mean(axis=0),
                bbox_min=pts.min(axis=0),
                bbox_max=pts.max(axis=0),
            )

        called_object_ids: list[int] = []

        def fake_geometry(_patch, obj):
            called_object_ids.append(int(obj.object_id))
            return 1.0 if int(obj.object_id) == 1 else 0.25

        module._geometry_consistency = fake_geometry

        result = module.process([patch], objects, SystemState().tsdf_volume)

        assert called_object_ids == [1, 2]
        assert len(result.matched) == 1
        assert result.matched[0][1] == 1
        patch_debug = result.debug["per_patch"][10]
        assert patch_debug["candidate_count_before_geometry"] == 5
        assert patch_debug["geometry_candidate_object_ids"] == [1, 2]
        assert patch_debug["geometry_candidate_count"] == 2
        assert result.debug["summary"]["candidate_score_count"] == 5
        assert result.debug["summary"]["geometry_score_count"] == 2
        assert result.debug["summary"]["geometry_pruned_candidate_count"] == 3

    def test_association_geometry_top_k_prioritizes_tsdf_vote_owner(self):
        module = AssociationModule(
            {
                "match_threshold": 0.1,
                "voxel_vote_weight": 0.7,
                "centroid_distance_weight": 0.3,
                "bbox_overlap_weight": 0.0,
                "geometry_overlap_weight": 0.0,
                "max_geometry_candidates": 1,
            }
        )
        patch_points = np.array(
            [[0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [0.10, 0.0, 0.0], [0.11, 0.0, 0.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=20,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        near_points = patch_points.copy()
        far_owner_points = patch_points + np.array([2.0, 0.0, 0.0], dtype=np.float32)
        objects = {
            1: ObjectMap(
                object_id=1,
                local_pcd=near_points.copy(),
                centroid=near_points.mean(axis=0),
                bbox_min=near_points.min(axis=0),
                bbox_max=near_points.max(axis=0),
            ),
            9: ObjectMap(
                object_id=9,
                local_pcd=far_owner_points.copy(),
                centroid=far_owner_points.mean(axis=0),
                bbox_min=far_owner_points.min(axis=0),
                bbox_max=far_owner_points.max(axis=0),
            ),
        }
        volume = TSDFInstanceVolume(voxel_size=0.05)
        volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 2.0})
        volume.owner_support[(2, 0, 0)] = VoxelOwnerSupport({9: 2.0})
        called_object_ids: list[int] = []

        def fake_geometry(_patch, obj):
            called_object_ids.append(int(obj.object_id))
            return 0.0

        module._geometry_consistency = fake_geometry

        result = module.process([patch], objects, volume)

        assert called_object_ids == [9]
        assert result.debug["per_patch"][20]["geometry_candidate_object_ids"] == [9]
```

- [ ] **Step 2: Run the pruning tests to verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_association_limits_geometry_consistency_to_top_k_candidates tests/test_pipeline.py::TestPipeline::test_association_geometry_top_k_prioritizes_tsdf_vote_owner -q
```

Expected: FAIL because `AssociationModule` does not define `max_geometry_candidates` behavior or the debug fields yet, and `_geometry_consistency()` is called for every candidate.

- [ ] **Step 3: Add association pruning config and helper methods**

In `src/modules/association.py`, add these fields in `AssociationModule.__init__` after the weight fields:

```python
        self.max_geometry_candidates = int(config.get("max_geometry_candidates", 0))
        self.geometry_candidate_min_cheap_score = float(config.get("geometry_candidate_min_cheap_score", 0.0))
        self.geometry_vote_owner_priority = bool(config.get("geometry_vote_owner_priority", True))
        self.geometry_nn_patch_sample = int(config.get("geometry_nn_patch_sample", 200))
        self.geometry_nn_object_sample = int(config.get("geometry_nn_object_sample", 200))
```

Add these helper methods before `_compute_score()`:

```python
    def _voxel_score_for_object(self, obj_id: int, vote: VoxelVoteResult) -> float:
        vote_denominator = max(vote.supported_voxel_count, 1)
        if vote.owner_votes and int(obj_id) in vote.owner_votes:
            return float(vote.owner_votes[int(obj_id)] / vote_denominator)
        return 0.0

    def _cheap_score_components(
        self,
        patch: Patch3D,
        obj: ObjectMap,
        vote: VoxelVoteResult,
    ) -> tuple[float, float, float, float]:
        voxel_score = self._voxel_score_for_object(int(obj.object_id), vote)
        dist = float(np.linalg.norm(patch.centroid - obj.centroid))
        centroid_score = max(0.0, 1.0 - dist / 2.0)
        bbox_score = bbox_iou_3d(patch.bbox_min, patch.bbox_max, obj.bbox_min, obj.bbox_max)
        cheap_total = (
            self.w_voxel_vote * voxel_score
            + self.w_centroid * centroid_score
            + self.w_bbox * bbox_score
        )
        return voxel_score, centroid_score, bbox_score, cheap_total

    def _select_geometry_candidate_ids(
        self,
        patch: Patch3D,
        objects: Dict[int, ObjectMap],
        candidate_ids: List[int],
        vote: VoxelVoteResult,
    ) -> tuple[set[int], dict[int, tuple[float, float, float, float]]]:
        cheap_by_id: dict[int, tuple[float, float, float, float]] = {}
        scored: list[tuple[tuple[int, float, float, int], int]] = []
        for obj_id in candidate_ids:
            obj = objects.get(obj_id)
            if obj is None or obj.state.value == "removed":
                continue
            components = self._cheap_score_components(patch, obj, vote)
            cheap_by_id[int(obj_id)] = components
            voxel_score, _centroid_score, _bbox_score, cheap_total = components
            if cheap_total < self.geometry_candidate_min_cheap_score and voxel_score <= 0.0:
                continue
            vote_owner_rank = 1 if self.geometry_vote_owner_priority and voxel_score > 0.0 else 0
            scored.append(((vote_owner_rank, cheap_total, voxel_score, -int(obj_id)), int(obj_id)))

        if self.max_geometry_candidates <= 0:
            return set(cheap_by_id), cheap_by_id

        scored.sort(reverse=True)
        selected = [obj_id for _key, obj_id in scored[: self.max_geometry_candidates]]
        return set(selected), cheap_by_id
```

- [ ] **Step 4: Replace `_compute_score()` with optional geometry**

Replace the current `_compute_score()` body in `src/modules/association.py` with this version:

```python
    def _compute_score(
        self,
        patch: Patch3D,
        obj: ObjectMap,
        vote: VoxelVoteResult,
        *,
        cheap_components: tuple[float, float, float, float] | None = None,
        compute_geometry: bool = True,
    ) -> AssociationScore:
        """Compute association score combining voxel voting + geometric fallback.

        Scoring is spatial-first. Semantics are NOT used here (Rule C).
        """
        if cheap_components is None:
            voxel_score, centroid_score, bbox_score, _cheap_total = self._cheap_score_components(patch, obj, vote)
        else:
            voxel_score, centroid_score, bbox_score, _cheap_total = cheap_components

        geometry_score = self._geometry_consistency(patch, obj) if compute_geometry else 0.0
        total = (
            self.w_voxel_vote * voxel_score
            + self.w_centroid * centroid_score
            + self.w_bbox * bbox_score
            + self.w_geometry * geometry_score
        )

        return AssociationScore(
            centroid_distance=centroid_score,
            bbox_overlap=bbox_score,
            geometry_overlap=geometry_score,
            voxel_vote_score=voxel_score,
            total_score=total,
            vote_result=VoxelVoteResult(
                touched_voxel_count=vote.touched_voxel_count,
                supported_voxel_count=vote.supported_voxel_count,
                owner_votes=dict(vote.owner_votes),
                normalized_vote_score=voxel_score,
                best_instance_id=vote.best_instance_id,
                geometry_consistency_score=geometry_score,
                final_association_score=total,
                new_instance_created=False,
            ),
        )
```

- [ ] **Step 5: Use top-k selection in `process()`**

Inside `AssociationModule.process()`, immediately after `patch_debug` is created, add:

```python
            geometry_candidate_ids, cheap_by_id = self._select_geometry_candidate_ids(
                patch,
                objects,
                list(candidate_ids),
                vote,
            )
            patch_debug["candidate_count_before_geometry"] = int(len(candidate_ids))
            patch_debug["geometry_candidate_object_ids"] = [
                int(obj_id) for obj_id in candidate_ids if int(obj_id) in geometry_candidate_ids
            ]
            patch_debug["geometry_candidate_count"] = int(len(geometry_candidate_ids))
```

Then replace the score call in the candidate loop:

```python
                score = self._compute_score(patch, obj, vote)
```

with:

```python
                score = self._compute_score(
                    patch,
                    obj,
                    vote,
                    cheap_components=cheap_by_id.get(int(obj_id)),
                    compute_geometry=int(obj_id) in geometry_candidate_ids,
                )
```

Before `logger.debug(...)` at the end of `process()`, add:

```python
        per_patch_debug = result.debug.get("per_patch", {}) or {}
        candidate_score_count = int(
            sum(len(item.get("candidate_object_ids", []) or []) for item in per_patch_debug.values())
        )
        geometry_score_count = int(
            sum(int(item.get("geometry_candidate_count", 0)) for item in per_patch_debug.values())
        )
        result.debug["summary"] = {
            "patch_count": int(len(object_patches)),
            "candidate_score_count": candidate_score_count,
            "geometry_score_count": geometry_score_count,
            "geometry_pruned_candidate_count": int(max(0, candidate_score_count - geometry_score_count)),
            "max_geometry_candidates": int(self.max_geometry_candidates),
        }
```

- [ ] **Step 6: Add compact association summary to frame metrics**

In `run_room0_full_eval.py`, inside the `record = { ... }` block after `"stage_timings": ...`, add:

```python
                    "association_summary": dict(
                        (pipeline.last_frame_debug.get("association_debug", {}) or {}).get("summary", {}) or {}
                    ),
```

Add this helper near `build_stage_timing_summary()`:

```python
def build_association_pruning_summary(frame_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [
        dict(metrics.get("association_summary", {}) or {})
        for metrics in frame_metrics
        if metrics.get("association_summary")
    ]
    if not summaries:
        return {}
    candidate_score_count = int(sum(int(item.get("candidate_score_count", 0)) for item in summaries))
    geometry_score_count = int(sum(int(item.get("geometry_score_count", 0)) for item in summaries))
    return {
        "frame_count": int(len(summaries)),
        "candidate_score_count_total": candidate_score_count,
        "geometry_score_count_total": geometry_score_count,
        "geometry_pruned_candidate_count_total": int(max(0, candidate_score_count - geometry_score_count)),
        "mean_candidate_score_count": float(candidate_score_count / max(len(summaries), 1)),
        "mean_geometry_score_count": float(geometry_score_count / max(len(summaries), 1)),
    }
```

In `write_run_report()`, after `stage_timing_summary = build_stage_timing_summary(frame_metrics)`, add:

```python
    association_pruning_summary = build_association_pruning_summary(frame_metrics)
```

Add a Markdown section after the stage timing section:

```python
    if association_pruning_summary:
        lines.append("")
        lines.append("## Association Pruning Summary")
        for key, value in association_pruning_summary.items():
            lines.append(f"- `{key}`: `{value}`")
```

In the `sidecar = { ... }` dict, add:

```python
        "association_pruning_summary": association_pruning_summary,
```

- [ ] **Step 7: Run pruning tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_association_limits_geometry_consistency_to_top_k_candidates tests/test_pipeline.py::TestPipeline::test_association_geometry_top_k_prioritizes_tsdf_vote_owner -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/association.py run_room0_full_eval.py tests/test_pipeline.py
git commit -m "perf: prune association geometry candidates"
```

## Task 2: Bounded Association Geometry Memory

**Files:**
- Modify: `src/core/data_structures.py`
- Modify: `src/modules/association.py`
- Modify: `src/modules/object_update.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for association representative geometry**

Add these methods to `tests/test_pipeline.py` near the association tests:

```python
    def test_association_geometry_uses_association_pcd_when_present(self):
        module = AssociationModule(
            {
                "match_threshold": 0.8,
                "voxel_vote_weight": 0.0,
                "centroid_distance_weight": 0.0,
                "bbox_overlap_weight": 0.0,
                "geometry_overlap_weight": 1.0,
            }
        )
        patch_points = np.array(
            [[0.0, 0.0, 1.0], [0.02, 0.0, 1.0], [0.0, 0.02, 1.0]],
            dtype=np.float32,
        )
        far_local_points = patch_points + np.array([5.0, 0.0, 0.0], dtype=np.float32)
        obj = ObjectMap(
            object_id=7,
            local_pcd=far_local_points.copy(),
            centroid=far_local_points.mean(axis=0),
            bbox_min=far_local_points.min(axis=0),
            bbox_max=far_local_points.max(axis=0),
        )
        obj.association_pcd = patch_points.copy()
        patch = Patch3D(
            patch_id=70,
            points=patch_points.copy(),
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )

        result = module.process([patch], {7: obj}, SystemState().tsdf_volume)

        assert len(result.matched) == 1
        assert result.matched[0][1] == 7
        assert result.scores[(70, 7)].geometry_overlap > 0.9

    def test_object_update_refreshes_bounded_association_geometry_memory(self):
        xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 80), np.linspace(0.0, 1.0, 80))
        points = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3).astype(np.float32)
        patch = Patch3D(
            patch_id=80,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=5,
            metadata={"anchor_class_name": "rug", "anchor_confidence": 0.93, "anchor_label_strength": "strong"},
        )
        module = ObjectUpdateModule(
            {
                "downsample_interval": 999,
                "max_points_per_object": 10000,
                "association_geometry": {
                    "enabled": True,
                    "voxel_size": 0.03,
                    "max_points_per_object": 128,
                },
                "surface_owner_gate": {"enabled": False},
                "tsdf": {"voxel_size": 0.05},
            }
        )

        state = module.process(AssociationResult(new_object_patches=[80]), [patch], SystemState())

        obj = next(iter(state.objects.values()))
        assert len(obj.local_pcd) == len(points)
        assert 0 < len(obj.association_pcd) <= 128
        assert obj.debug["association_geometry"]["point_count"] == len(obj.association_pcd)
        assert obj.debug["association_geometry"]["max_points_per_object"] == 128
```

- [ ] **Step 2: Run representative geometry tests to verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_association_geometry_uses_association_pcd_when_present tests/test_pipeline.py::TestPipeline::test_object_update_refreshes_bounded_association_geometry_memory -q
```

Expected: FAIL because `ObjectMap` has no `association_pcd` field and association still uses `local_pcd` only.

- [ ] **Step 3: Add `association_pcd` to `ObjectMap`**

In `src/core/data_structures.py`, add this field after `local_pcd` in `ObjectMap`:

```python
    association_pcd: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
```

- [ ] **Step 4: Make association use representative points**

In `src/modules/association.py`, add this helper before `_geometry_consistency()`:

```python
    @staticmethod
    def _object_geometry_points(obj: ObjectMap) -> np.ndarray:
        association_pcd = getattr(obj, "association_pcd", None)
        if association_pcd is not None and len(association_pcd) > 0:
            return np.asarray(association_pcd, dtype=np.float32)
        return np.asarray(obj.local_pcd, dtype=np.float32)
```

In `_geometry_consistency()`, replace:

```python
        o_pts = obj.local_pcd
```

with:

```python
        o_pts = self._object_geometry_points(obj)
```

Also replace random sampling with deterministic sampling:

```python
        max_patch_pts = self.geometry_nn_patch_sample if self.geometry_nn_patch_sample > 0 else 200
        max_object_pts = self.geometry_nn_object_sample if self.geometry_nn_object_sample > 0 else 200
        p_pts = patch.points
        o_pts = self._object_geometry_points(obj)
        if len(p_pts) > max_patch_pts:
            idx = np.linspace(0, len(p_pts) - 1, num=max_patch_pts, dtype=np.int64)
            p_pts = p_pts[idx]
        if len(o_pts) > max_object_pts:
            idx = np.linspace(0, len(o_pts) - 1, num=max_object_pts, dtype=np.int64)
            o_pts = o_pts[idx]
```

- [ ] **Step 5: Maintain `association_pcd` in object update**

In `src/modules/object_update.py`, add these config fields in `ObjectUpdateModule.__init__` after `self.max_points`:

```python
        association_geometry_cfg = config.get("association_geometry", {})
        self.association_geometry_enabled = bool(association_geometry_cfg.get("enabled", False))
        self.association_geometry_voxel = float(
            association_geometry_cfg.get("voxel_size", max(float(self.downsample_voxel), 0.02))
        )
        self.association_geometry_max_points = int(
            association_geometry_cfg.get("max_points_per_object", min(int(self.max_points), 2048))
        )
```

Add this method near `_update_object()`:

```python
    def _refresh_association_geometry(self, obj: ObjectMap) -> None:
        if not self.association_geometry_enabled or len(obj.local_pcd) == 0:
            obj.association_pcd = np.empty((0, 3), dtype=np.float32)
            obj.debug["association_geometry"] = {
                "enabled": bool(self.association_geometry_enabled),
                "point_count": 0,
                "voxel_size": float(self.association_geometry_voxel),
                "max_points_per_object": int(self.association_geometry_max_points),
            }
            return
        points = voxel_downsample(np.asarray(obj.local_pcd, dtype=np.float32), self.association_geometry_voxel)
        if self.association_geometry_max_points > 0 and len(points) > self.association_geometry_max_points:
            points = self._deterministic_spatial_cap(points, self.association_geometry_max_points)
        obj.association_pcd = np.asarray(points, dtype=np.float32)
        obj.debug["association_geometry"] = {
            "enabled": True,
            "point_count": int(len(obj.association_pcd)),
            "voxel_size": float(self.association_geometry_voxel),
            "max_points_per_object": int(self.association_geometry_max_points),
        }
```

Call `self._refresh_association_geometry(obj)`:

- at the end of `_update_object()` before `_update_whole_evidence(obj, patch)`
- at the end of `_create_object()` before `return obj`
- at the end of `_promote_provisional_object()` before `return obj`
- in `replace_observations()` after recomputing `target_obj.local_pcd`, before `rebuild_anchor_semantic_votes_from_observations(target_obj)`

In `_refresh_object_debug()`, keep the existing `local_geometry_memory` block and do not overwrite `association_geometry` if it already exists.

- [ ] **Step 6: Run representative geometry tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_association_geometry_uses_association_pcd_when_present tests/test_pipeline.py::TestPipeline::test_object_update_refreshes_bounded_association_geometry_memory -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/core/data_structures.py src/modules/association.py src/modules/object_update.py tests/test_pipeline.py
git commit -m "perf: maintain bounded association geometry"
```

## Task 3: Delayed Residual Semantics Without Parent-Anchor Contamination

**Files:**
- Modify: `src/modules/patch_lifting.py`
- Modify: `src/modules/semantic_memory.py`
- Modify: `src/modules/object_update.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_provisional_pool.py`

- [ ] **Step 1: Write failing tests for metadata preservation and delayed commit**

Add this assertion block to the existing `test_patch_lifting_preserves_anchor_label_strength_and_blocked_candidates` in `tests/test_pipeline.py`:

```python
        proposal.metadata["semantic_commit_allowed"] = False
        proposal.metadata["residual_semantic_policy"] = "unknown"
        proposal.metadata["mask_anchor_relation"] = "contained_residual"
```

Add these assertions at the end of that test:

```python
        assert patches[0].metadata["semantic_commit_allowed"] is False
        assert patches[0].metadata["residual_semantic_policy"] == "unknown"
        assert patches[0].metadata["mask_anchor_relation"] == "contained_residual"
```

Add these methods near the semantic memory tests in `tests/test_pipeline.py`:

```python
    def test_semantic_commit_blocked_anchor_observation_is_delayed_not_exported(self):
        obj = ObjectMap(object_id=91)
        patch = Patch3D(
            patch_id=91,
            points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            source_frame_id=4,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.96,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
            },
        )

        accumulate_anchor_semantic_vote(obj, patch)

        assert object_export_semantic_label(obj) == ""
        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["ignored_observation_reasons"] == {"semantic_commit_blocked": 1}
        assert anchor_state["delayed_evidence"][0]["label"] == "sofa"
        assert anchor_state["delayed_evidence"][0]["commit_eligible"] is False

    def test_delayed_residual_can_later_export_from_direct_strong_anchor(self):
        obj = ObjectMap(object_id=92)
        blocked = Patch3D(
            patch_id=92,
            points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            source_frame_id=4,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.96,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
            },
        )
        direct = Patch3D(
            patch_id=93,
            points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            source_frame_id=5,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.95,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
                "semantic_commit_allowed": True,
            },
        )

        accumulate_anchor_semantic_vote(obj, blocked)
        accumulate_anchor_semantic_vote(obj, direct)

        assert object_export_semantic_label(obj) == "blanket"
        state = object_export_semantic_state(obj)
        assert state["export_state"] == "safe_provisional"
        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["label_weighted_score"]["blanket"] > 0.75
        assert "sofa" not in anchor_state["label_weighted_score"]
```

Add this end-to-end test to `tests/test_provisional_pool.py`:

```python
def test_blocked_residual_object_exports_later_direct_anchor_label() -> None:
    module = ObjectUpdateModule(
        {
            "downsample_interval": 99,
            "surface_owner_gate": {"enabled": False},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    residual = _make_patch(patch_id=300, frame_id=1)
    residual.metadata.update(
        {
            "anchor_class_name": "sofa",
            "anchor_confidence": 0.97,
            "anchor_view_quality": 0.9,
            "anchor_label_strength": "strong",
            "semantic_commit_allowed": False,
            "residual_semantic_policy": "unknown",
        }
    )
    state = module.process(AssociationResult(new_object_patches=[300]), [residual], SystemState())
    obj_id = next(iter(state.objects))
    assert object_export_semantic_label(state.objects[obj_id]) == ""

    direct = _make_patch(patch_id=301, frame_id=2)
    direct.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.95,
            "anchor_view_quality": 0.9,
            "anchor_label_strength": "strong",
            "semantic_commit_allowed": True,
        }
    )
    state = module.process(AssociationResult(matched=[(301, obj_id, None)]), [direct], state)

    assert object_export_semantic_label(state.objects[obj_id]) == "blanket"
    assert state.objects[obj_id].debug["anchor_semantics"]["delayed_evidence"][0]["label"] == "sofa"
```

- [ ] **Step 2: Run delayed semantic tests to verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_patch_lifting_preserves_anchor_label_strength_and_blocked_candidates tests/test_pipeline.py::TestPipeline::test_semantic_commit_blocked_anchor_observation_is_delayed_not_exported tests/test_pipeline.py::TestPipeline::test_delayed_residual_can_later_export_from_direct_strong_anchor tests/test_provisional_pool.py::test_blocked_residual_object_exports_later_direct_anchor_label -q
```

Expected: FAIL because patch lifting drops the residual semantic keys and `accumulate_anchor_semantic_vote()` currently does not honor `semantic_commit_allowed=False`.

- [ ] **Step 3: Preserve residual semantic metadata through patch lifting**

In `src/modules/patch_lifting.py`, add these entries to the `Patch3D.metadata` dict in `_lift_single_proposal()`:

```python
                "semantic_commit_allowed": bool(proposal.metadata.get("semantic_commit_allowed", True)),
                "residual_semantic_policy": str(proposal.metadata.get("residual_semantic_policy", "")),
                "mask_anchor_relation": str(proposal.metadata.get("mask_anchor_relation", "")),
```

- [ ] **Step 4: Record blocked anchor labels as delayed evidence**

In `src/modules/semantic_memory.py`, add this check inside `accumulate_anchor_semantic_vote()` after `label` and `confidence` are computed and before the `strength != "strong"` branch:

```python
    if patch.metadata.get("semantic_commit_allowed") is False:
        _record_delayed_anchor_observation(
            obj,
            label=label,
            confidence=confidence,
            frame_id=frame_id,
            strength=strength,
            reason="semantic_commit_blocked",
        )
        _record_ignored_anchor_observation(obj, "semantic_commit_blocked")
        return
```

Add this helper near `_record_contextual_anchor_observation()`:

```python
def _record_delayed_anchor_observation(
    obj: ObjectMap,
    *,
    label: str,
    confidence: float,
    frame_id: int,
    strength: str,
    reason: str,
) -> None:
    state = _anchor_semantic_state(obj)
    state["delayed_evidence"].append(
        {
            "label": str(label),
            "confidence": float(confidence),
            "frame_id": int(frame_id),
            "anchor_label_strength": str(strength),
            "reason": str(reason),
            "commit_eligible": False,
        }
    )
```

In `_anchor_semantic_state()`, add:

```python
    if not isinstance(state.get("delayed_evidence"), list):
        state["delayed_evidence"] = []
```

- [ ] **Step 5: Preserve delayed evidence in object debug refresh**

In `src/modules/object_update.py`, inside `_refresh_object_debug()`, build `delayed_evidence` next to `contextual_evidence`:

```python
            delayed_evidence = []
            for item in anchor_state.get("delayed_evidence", []) or []:
                if not isinstance(item, dict):
                    continue
                delayed_evidence.append(
                    {
                        "label": str(item.get("label", "")),
                        "confidence": float(item.get("confidence", 0.0)),
                        "frame_id": int(item.get("frame_id", 0)),
                        "anchor_label_strength": str(item.get("anchor_label_strength", "")),
                        "reason": str(item.get("reason", "")),
                        "commit_eligible": bool(item.get("commit_eligible", False)),
                    }
                )
```

Then include these keys in the reconstructed `obj.debug["anchor_semantics"]` dict:

```python
                "delayed_evidence": delayed_evidence,
                "ignored_observation_count": int(anchor_state.get("ignored_observation_count", 0)),
                "ignored_observation_reasons": {
                    str(reason): int(count)
                    for reason, count in (anchor_state.get("ignored_observation_reasons", {}) or {}).items()
                },
```

- [ ] **Step 6: Run delayed semantic tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_patch_lifting_preserves_anchor_label_strength_and_blocked_candidates tests/test_pipeline.py::TestPipeline::test_semantic_commit_blocked_anchor_observation_is_delayed_not_exported tests/test_pipeline.py::TestPipeline::test_delayed_residual_can_later_export_from_direct_strong_anchor tests/test_provisional_pool.py::test_blocked_residual_object_exports_later_direct_anchor_label -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/patch_lifting.py src/modules/semantic_memory.py src/modules/object_update.py tests/test_pipeline.py tests/test_provisional_pool.py
git commit -m "fix: delay blocked residual semantics until direct evidence"
```

## Task 4: Fast High-IoU Room0 Config

**Files:**
- Create: `configs/room0_surface_gate_fast_high_iou_4090.yaml`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing config test**

Add this test near the config tests in `tests/test_pipeline.py`:

```python
    def test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend(self):
        import yaml

        path = Path("configs/room0_surface_gate_fast_high_iou_4090.yaml")
        assert path.exists()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["anchor_frontend"]["enabled"] is True
        assert config["anchor_frontend"]["use_sam_intersection_proposals"] is True
        assert config["anchor_frontend"].get("anchor_primary_mode", False) is False
        assert config.get("anchor_guided_sam", {}).get("enabled", False) is False
        assert config["association"]["max_geometry_candidates"] == 12
        assert config["association"]["geometry_nn_patch_sample"] == 96
        assert config["association"]["geometry_nn_object_sample"] == 96
        assert config["object_update"]["association_geometry"]["enabled"] is True
        assert config["object_update"]["association_geometry"]["max_points_per_object"] == 2048
        assert config["semantic_memory"]["anchor_export"]["enabled"] is True
```

- [ ] **Step 2: Run config test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend -q
```

Expected: FAIL because the config file does not exist.

- [ ] **Step 3: Create the config from the 0526 surface-gate baseline**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
cp configs/room0_surface_gate_4090.yaml configs/room0_surface_gate_fast_high_iou_4090.yaml
```

Edit `configs/room0_surface_gate_fast_high_iou_4090.yaml` so the following keys are present with these values:

```yaml
anchor_frontend:
  enabled: true
  use_boxes_as_primary_proposals: true
  use_sam_intersection_proposals: true
  anchor_primary_mode: false
  assignment_policy: semantic_vote
  supplemental_enabled: true

anchor_guided_sam:
  enabled: false

association:
  centroid_distance_weight: 0.4
  bbox_overlap_weight: 0.3
  geometry_overlap_weight: 0.2
  pose_proximity_weight: 0.1
  match_threshold: 0.5
  max_geometry_candidates: 12
  geometry_candidate_min_cheap_score: 0.02
  geometry_vote_owner_priority: true
  geometry_nn_patch_sample: 96
  geometry_nn_object_sample: 96
  observation_identity_gate_enabled: true
  observation_identity_min_patch_confidence: 0.35
  semantic_conflict_gate_enabled: true
  semantic_conflict_min_patch_confidence: 0.35
  semantic_conflict_min_object_confidence: 0.35
  semantic_conflict_subregion_point_ratio: 0.35
  semantic_conflict_subregion_volume_ratio: 0.35

object_update:
  association_geometry:
    enabled: true
    voxel_size: 0.03
    max_points_per_object: 2048
```

Keep these 0526 coverage-oriented values unchanged from `configs/room0_surface_gate_4090.yaml`:

```yaml
patch_lifting:
  point_sample_ratio: 1.0
  max_points_per_patch: 0

object_update:
  downsample_voxel_size: 0.01
  downsample_interval: 20
  max_points_per_object: 40000

semantic_memory:
  anchor_export:
    enabled: true
```

Set:

```yaml
pipeline:
  log_interval: 1
  verbose: false
  collect_stage_timings: true
```

- [ ] **Step 4: Run config test**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add configs/room0_surface_gate_fast_high_iou_4090.yaml tests/test_pipeline.py
git commit -m "config: add fast high-iou room0 mode"
```

## Task 5: Regression Suite

**Files:**
- No code changes unless tests fail.

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_anchor_guided_sam.py tests/test_runtime_vis.py tests/test_provisional_pool.py tests/test_dual_map.py tests/test_async_refinement.py tests/test_pipeline.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: If tests fail, fix only root-cause regressions**

Use `superpowers:systematic-debugging`. Do not loosen semantic safety assertions to make tests pass.

- [ ] **Step 3: Commit test fixes if any**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git status --short
git add src/core/data_structures.py src/modules/association.py src/modules/object_update.py src/modules/patch_lifting.py src/modules/semantic_memory.py run_room0_full_eval.py tests/test_anchor_guided_sam.py tests/test_runtime_vis.py tests/test_provisional_pool.py tests/test_dual_map.py tests/test_async_refinement.py tests/test_pipeline.py configs/room0_surface_gate_fast_high_iou_4090.yaml
git commit -m "test: stabilize fast high-iou regression suite"
```

Skip this commit if there are no changes after Step 1.

## Task 6: Smoke and 200f Evaluation Gates

**Files:**
- No code changes unless metrics contradict the design.

- [ ] **Step 1: Run 20f stride10 smoke**

Run on GPU outside the sandbox if CUDA is hidden:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260531_room0_surface_gate_fast_high_iou_stride10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected:
- Command exits `0`.
- `run_report.json` exists.
- `association_pruning_summary.geometry_score_count_total < association_pruning_summary.candidate_score_count_total`.
- No new classes that were strong in 0526 become obviously all-zero in the 20f smoke report.

- [ ] **Step 2: Run 200f stride10 evaluation**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260531_room0_surface_gate_fast_high_iou_stride10_200f \
  --num-frames 200 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected gate:
- mIoU must be at least `0.52`.
- f-mIoU must be at least `0.65`.
- It must beat the failed 20260531 anchor-guided full-run mIoU `0.3682` by a wide margin.
- `association` mean time should be lower than the current failed full-run mean `2.857s/frame`; target for 200f is under `1.5s/frame`.
- `final_object_semantic_audit.json` should not show the same failure shape as the failed run: committed/exported objects should be much higher than `29`, and unlabeled objects should be much lower than `105`.

- [ ] **Step 3: If 200f gate fails, do not run 2000f**

If either quality or speed gate fails, inspect:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/tmp_validation/20260531_room0_surface_gate_fast_high_iou_stride10_200f")
report = json.load(open(root / "room0/run_report.json"))
audit = json.load(open(root / "room0/final_object_semantic_audit.json"))
print("metrics", report["miou"], report["macc"], report["fmiou"], report["fmacc"])
print("timing", report["stage_timing_summary"])
print("association_pruning", report.get("association_pruning_summary", {}))
print("semantic_state_counts", audit.get("semantic_state_counts", {}))
print("export_state_counts", audit.get("export_state_counts", {}))
PY
```

Then adjust only the narrow cause:
- If mIoU drops while unlabeled count is high, inspect delayed semantic evidence and anchor export policy.
- If mIoU drops while unlabeled count is normal, loosen only `max_geometry_candidates` from `12` to `20`.
- If association is still slow, reduce `geometry_nn_patch_sample` and `geometry_nn_object_sample` from `96` to `64` before lowering proposal coverage.

- [ ] **Step 4: Commit metric-driven config adjustment if needed**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add configs/room0_surface_gate_fast_high_iou_4090.yaml
git commit -m "tune: adjust fast high-iou association gates"
```

Skip if no config change was needed.

## Task 7: 2000f Full Fast Evaluation

**Files:**
- No code changes unless the 2000f run reveals a regression.

- [ ] **Step 1: Run full 2000f stride1 fast eval only after Task 6 passes**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260531_room0_surface_gate_fast_high_iou_stride1_2000f \
  --num-frames 2000 \
  --frame-stride 1 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected gate:
- mIoU should stay close to the 0526 baseline direction. Minimum acceptable first gate: `>= 0.56`; target: `>= 0.59`.
- f-mIoU minimum acceptable first gate: `>= 0.70`; target: `>= 0.74`.
- FPS must beat the failed 20260531 run `0.1354`; minimum acceptable first gate: `>= 0.22`; target: `>= 0.35`.
- Association total time must fall materially below `5714s`; target first gate: `< 3500s`.
- The final audit must not repeat `29 committed / 105 unlabeled`.

- [ ] **Step 2: Summarize final metrics**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/tmp_validation/20260531_room0_surface_gate_fast_high_iou_stride1_2000f")
report = json.load(open(root / "room0/run_report.json"))
audit = json.load(open(root / "room0/final_object_semantic_audit.json"))
print("metrics", report["miou"], report["macc"], report["fmiou"], report["fmacc"])
print("objects", report["final_object_count"], report["final_local_memory_point_count_total"])
print("timing")
for stage, values in sorted(report["stage_timing_summary"].items(), key=lambda kv: kv[1]["total_sec"], reverse=True)[:10]:
    print(stage, values)
print("association_pruning", report.get("association_pruning_summary", {}))
print("semantic_state_counts", audit.get("semantic_state_counts", {}))
print("export_state_counts", audit.get("export_state_counts", {}))
PY
```

- [ ] **Step 3: Commit final config if 2000f gate passes**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git status --short
git add configs/room0_surface_gate_fast_high_iou_4090.yaml
git commit -m "tune: validate fast high-iou room0 full config"
```

Skip this commit if `git status --short` shows no changes after Task 7.

## Self-Review Checklist

- Spec coverage: This plan directly addresses the measured failures: association runtime growth, object-update matching cost, and unlabeled residual semantics.
- Placeholder scan: No `TBD`, `TODO`, or vague “write tests” steps remain; every code task includes concrete test and implementation snippets.
- Type consistency: New `ObjectMap.association_pcd` is an `np.ndarray`; association consumes it read-only, object update maintains it, and export still uses `local_pcd`.
- Safety invariant: `semantic_commit_allowed=False` never commits a parent/overlap label; later direct strong anchor evidence can still commit/export a different label.
- Experiment gate: 2000f is explicitly blocked until smoke and 200f gates pass.
