# DualMap-Inspired Fast Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast hybrid room0 mode that borrows DualMap's detector-constrained workload while preserving OVIOVO's semantic safety gates and geometry-first object update path.

**Architecture:** Keep the existing full SAM2-driven path as the default. Add opt-in controls for per-stage timing, deterministic patch point sampling, a YOLO-anchor-primary proposal mode, and a fast config that combines these pieces for smoke/full experiments. The fast path should reduce per-frame proposal count and lifted point count without reintroducing unsafe blanket/rug-to-sofa semantic inheritance.

**Tech Stack:** Python 3, NumPy, pytest, YAML configs, existing OVIOVO modules (`Pipeline`, `ObjectAnchorModule`, `PatchLiftingModule`, `ObjectUpdateModule`), existing YOLOWorld anchor backend, precomputed SAM2 cache for optional discovery.

---

## File Structure

- Modify `src/pipelines/main_pipeline.py`
  - Add a small per-stage timing helper.
  - Add an opt-in anchor-primary proposal branch.
  - Include timing and fast-mode provenance in `last_frame_debug`.
- Modify `src/modules/object_anchor.py`
  - Add `generate_anchor_box_proposals()` to create detector-box proposals without needing full-frame SAM proposals.
  - Preserve existing semantic-vote and contained-subproposal behavior for the default path.
- Modify `src/modules/patch_lifting.py`
  - Add deterministic point sampling after foreground-depth filtering and before world-frame lifting.
  - Record sampling diagnostics in patch metadata.
- Modify `src/modules/object_update.py`
  - Add an opt-in representative-point mode for surface-owner gate statistics and accept/reject masks.
  - Keep default point-level behavior unchanged.
- Modify `configs/room0_surface_gate_4090.yaml`
  - Add disabled-by-default config keys for the new fast controls.
- Create `configs/room0_fast_hybrid_4090.yaml`
  - New experiment config enabling anchor-primary proposals, patch sampling, representative gate mode, and fast output-compatible settings.
- Modify `run_room0_full_eval.py`
  - Copy `pipeline.last_frame_debug["stage_timings"]` into `frame_metrics.jsonl`.
  - Add report summary for timing totals and fast controls.
- Modify `tests/test_object_anchor.py`
  - Add anchor-primary proposal tests.
- Modify `tests/test_pipeline.py`
  - Add patch sampling tests, pipeline timing tests, and report timing payload tests.

## Task 1: Per-Stage Timing Payload

**Files:**
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for pipeline timing**

Append this test method inside `class TestPipeline` in `tests/test_pipeline.py`:

```python
    def test_pipeline_records_stage_timings_when_enabled(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 4
  image_width: 4
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
pipeline:
  verbose: false
  collect_stage_timings: true
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        timings = pipe.last_frame_debug["stage_timings"]
        expected_stages = {
            "frame_input",
            "proposal_generation",
            "runtime_vis",
            "depth_refinement",
            "patch_lifting",
            "active_set",
            "bg_obj_split",
            "association",
            "object_update",
            "semantic_memory",
            "dense_surface",
            "map_tiering",
            "background_update",
            "dynamic_maintenance",
        }
        assert expected_stages.issubset(set(timings))
        assert all(value >= 0.0 for value in timings.values())
```

- [ ] **Step 2: Run timing test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled -q
```

Expected: FAIL with `KeyError: 'stage_timings'` or missing timing keys.

- [ ] **Step 3: Implement timing helper**

In `src/pipelines/main_pipeline.py`, add this import near the existing imports:

```python
import time
from contextlib import contextmanager
```

In `Pipeline.__init__`, after `self.verbose = ...`, add:

```python
        pipeline_cfg = self.config.get("pipeline", {})
        self.collect_stage_timings = bool(pipeline_cfg.get("collect_stage_timings", False))
        self._stage_timings: dict[str, float] = {}
```

Add this method to `Pipeline` before `process_frame()`:

```python
    @contextmanager
    def _timed_stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.collect_stage_timings:
                self._stage_timings[name] = self._stage_timings.get(name, 0.0) + float(
                    time.perf_counter() - start
                )
```

- [ ] **Step 4: Wrap process_frame stages**

In `src/pipelines/main_pipeline.py`, at the top of `process_frame()` before Step 1, add:

```python
        self._stage_timings = {}
```

Wrap each existing stage body with `with self._timed_stage("<stage>"):` while keeping code order unchanged. Use exactly these stage names:

```python
"frame_input"
"proposal_generation"
"runtime_vis"
"depth_refinement"
"patch_lifting"
"active_set"
"bg_obj_split"
"association"
"object_update"
"semantic_memory"
"dense_surface"
"map_tiering"
"background_update"
"dynamic_maintenance"
```

At the end of `last_frame_debug`, add:

```python
            "stage_timings": dict(self._stage_timings),
```

- [ ] **Step 5: Add timing to frame metrics**

In `run_room0_full_eval.py`, inside the `record = { ... }` block where `surface_owner_gate` is written, add:

```python
                    "stage_timings": dict(pipeline.last_frame_debug.get("stage_timings", {}) or {}),
```

- [ ] **Step 6: Run timing test to verify it passes**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/pipelines/main_pipeline.py run_room0_full_eval.py tests/test_pipeline.py
git commit -m "feat: record per-stage pipeline timings"
```

## Task 2: Deterministic Patch Point Sampling

**Files:**
- Modify: `src/modules/patch_lifting.py`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for patch sampling**

Append this test method inside `class TestPatchLiftingModule` in `tests/test_pipeline.py`:

```python
    def test_patch_lifting_deterministically_samples_points_after_depth_filter(self):
        depth = np.ones((10, 10), dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=21,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": 1,
                "point_sample_ratio": 0.25,
                "max_points_per_patch": 20,
            }
        )
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=10, height=10)

        first = module.process([proposal], depth, np.eye(4, dtype=np.float64), intrinsics, frame_id=0)
        second = module.process([proposal], depth, np.eye(4, dtype=np.float64), intrinsics, frame_id=0)

        assert len(first) == 1
        assert len(second) == 1
        assert len(first[0].points) == 20
        np.testing.assert_allclose(first[0].points, second[0].points)
        sampling = first[0].metadata["point_sampling"]
        assert sampling["enabled"] is True
        assert sampling["input_point_count"] == 100
        assert sampling["sampled_point_count"] == 20
        assert sampling["point_sample_ratio"] == 0.25
        assert sampling["max_points_per_patch"] == 20
```

- [ ] **Step 2: Run sampling test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestPatchLiftingModule::test_patch_lifting_deterministically_samples_points_after_depth_filter -q
```

Expected: FAIL because `point_sampling` metadata does not exist and point count remains 100.

- [ ] **Step 3: Add sampling config fields**

In `src/modules/patch_lifting.py`, inside `PatchLiftingModule.__init__`, after the foreground-depth config fields, add:

```python
        self.point_sample_ratio = float(config.get("point_sample_ratio", 1.0))
        self.max_points_per_patch = int(config.get("max_points_per_patch", 0))
```

- [ ] **Step 4: Add deterministic sampling helper**

In `src/modules/patch_lifting.py`, add this method to `PatchLiftingModule` before `_foreground_depth_mask()`:

```python
    def _sample_pixel_vectors(
        self,
        u: np.ndarray,
        v: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        input_count = int(z.size)
        ratio = float(np.clip(self.point_sample_ratio, 0.0, 1.0))
        ratio_limit = input_count if ratio <= 0.0 or ratio >= 1.0 else max(self.min_points, int(np.ceil(input_count * ratio)))
        cap_limit = input_count if self.max_points_per_patch <= 0 else max(self.min_points, int(self.max_points_per_patch))
        target = min(input_count, ratio_limit, cap_limit)
        diagnostics = {
            "enabled": bool(target < input_count),
            "input_point_count": input_count,
            "sampled_point_count": int(target),
            "point_sample_ratio": float(self.point_sample_ratio),
            "max_points_per_patch": int(self.max_points_per_patch),
        }
        if target >= input_count:
            return u, v, z, diagnostics
        indices = np.linspace(0, input_count - 1, num=target, dtype=np.int64)
        return u[indices], v[indices], z[indices], diagnostics
```

- [ ] **Step 5: Apply sampling in `_lift_single_proposal()`**

In `_lift_single_proposal()`, after foreground depth filtering and the second `if z.size < self.min_points:` check, add:

```python
        u, v, z, sampling_debug = self._sample_pixel_vectors(u, v, z)
        if z.size < self.min_points:
            return None
```

In the returned `Patch3D.metadata`, add:

```python
                "point_sampling": sampling_debug,
```

- [ ] **Step 6: Add default disabled config**

In `configs/room0_surface_gate_4090.yaml`, under `patch_lifting:`, add:

```yaml
  point_sample_ratio: 1.0
  max_points_per_patch: 0
```

- [ ] **Step 7: Run sampling tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestPatchLiftingModule -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/patch_lifting.py configs/room0_surface_gate_4090.yaml tests/test_pipeline.py
git commit -m "feat: add deterministic patch point sampling"
```

## Task 3: Anchor-Primary Proposal Mode

**Files:**
- Modify: `src/modules/object_anchor.py`
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Test: `tests/test_object_anchor.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for anchor box proposals**

Append this test to `tests/test_object_anchor.py`:

```python
def test_object_anchor_generates_anchor_box_primary_proposals() -> None:
    module = ObjectAnchorModule({"enabled": False, "anchor_primary_mode": True})
    module.enabled = True
    module.anchor_primary_mode = True
    module.proposal_min_area = 1

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=4,
                    bbox_xyxy=np.array([1, 1, 4, 5], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.82,
                )
            ]

    module.backend = _FakeBackend()
    rgb = np.zeros((6, 7, 3), dtype=np.uint8)

    anchors, proposals, assignments = module.generate_anchor_box_proposals(rgb)

    assert len(anchors) == 1
    assert len(proposals) == 1
    assert len(assignments) == 1
    proposal = proposals[0]
    assert proposal.proposal_id == 4
    assert proposal.backend_name == "anchor_box_primary"
    assert proposal.area == 12
    assert proposal.metadata["anchor_class_name"] == "sofa"
    assert proposal.metadata["anchor_label_strength"] == "strong"
    assert proposal.metadata["mask_source"] == "anchor_box"
    assert assignments[0].anchor_id == 4
    assert assignments[0].keepalive is True
```

- [ ] **Step 2: Run anchor-primary test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_object_anchor.py::test_object_anchor_generates_anchor_box_primary_proposals -q
```

Expected: FAIL because `generate_anchor_box_proposals` does not exist.

- [ ] **Step 3: Add anchor-primary config field**

In `src/modules/object_anchor.py`, inside `ObjectAnchorModule.__init__` after `self.use_sam_intersection_proposals = ...`, add:

```python
        self.anchor_primary_mode = bool(config.get("anchor_primary_mode", False))
```

- [ ] **Step 4: Implement `generate_anchor_box_proposals()`**

Add this method to `ObjectAnchorModule` after `generate_proposals()`:

```python
    def generate_anchor_box_proposals(
        self,
        rgb: np.ndarray,
    ) -> Tuple[List[Anchor2D], List[Proposal2D], List[AnchorAssignment]]:
        """Generate DualMap-style detector-box primary proposals.

        This fast path intentionally avoids assigning labels to unrelated SAM
        masks. Each detector anchor becomes one rectangular proposal, and SAM2
        high-recall discovery can be layered on separately by the pipeline.
        """
        self.last_anchors = self._generate_merged_anchors(rgb)
        proposals: list[Proposal2D] = []
        assignments: list[AnchorAssignment] = []
        height, width = rgb.shape[:2]
        for anchor in self.last_anchors:
            mask = self._bbox_mask((height, width), np.asarray(anchor.bbox_xyxy, dtype=np.float32))
            area = int(mask.sum())
            if area < self.proposal_min_area:
                continue
            proposal_id = int(anchor.anchor_id)
            proposal = Proposal2D(
                proposal_id=proposal_id,
                mask=mask,
                bbox_xyxy=self._mask_bbox(mask),
                area=area,
                confidence=float(anchor.confidence),
                backend_name="anchor_box_primary",
                metadata={
                    "source": "anchor_box_primary",
                    "requested_backend": self.requested_backend_name,
                    "actual_backend": self.active_backend_name,
                    "anchor_primary_proposal": True,
                    "anchor_id": int(anchor.anchor_id),
                    "anchor_class_name": str(anchor.class_name),
                    "anchor_confidence": float(anchor.confidence),
                    "anchor_bbox_iou": 1.0,
                    "anchor_center_inside": True,
                    "anchor_keepalive": True,
                    "anchor_label_strength": "strong",
                    "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
                    "anchor_source_bbox_xyxy": np.asarray(anchor.bbox_xyxy, dtype=np.float32).copy(),
                    "anchor_source_class_name": str(anchor.class_name),
                    "anchor_source_confidence": float(anchor.confidence),
                    "mask_source": "anchor_box",
                    "anchor_assignment_strategy": "anchor_box_primary",
                    "source_raw_proposal_ids": [],
                },
            )
            assignment = AnchorAssignment(
                proposal_id=proposal_id,
                anchor_id=int(anchor.anchor_id),
                class_name=str(anchor.class_name),
                confidence=float(anchor.confidence),
                bbox_iou=1.0,
                center_inside=True,
                keepalive=True,
            )
            proposals.append(proposal)
            assignments.append(assignment)
        self.last_assignments = assignments
        return self.last_anchors, proposals, assignments
```

- [ ] **Step 5: Run anchor-primary test**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_object_anchor.py::test_object_anchor_generates_anchor_box_primary_proposals -q
```

Expected: PASS.

- [ ] **Step 6: Write failing pipeline branch test**

Append this test method inside `class TestPipeline` in `tests/test_pipeline.py`:

```python
    def test_pipeline_anchor_primary_mode_skips_sam_source_proposals(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
  anchor_primary_mode: true
  proposal_min_area: 1
pipeline:
  verbose: false
patch_lifting:
  min_points: 1
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe.object_anchor.enabled = True
        pipe.object_anchor.anchor_primary_mode = True

        class _FakeAnchorBackend:
            def generate_anchors(self, rgb):
                return [
                    Anchor2D(
                        anchor_id=2,
                        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                        class_name="chair",
                        confidence=0.9,
                    )
                ]

        class _ExplodingProposalBackend:
            def generate_proposals(self, rgb, depth, frame=None):
                raise AssertionError("SAM proposal backend should not run in anchor-primary mode")

        pipe.object_anchor.backend = _FakeAnchorBackend()
        pipe.proposal.backend = _ExplodingProposalBackend()
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["proposal_source"] == "anchor_box_primary"
        assert len(pipe.last_source_proposals) == 0
        assert len(pipe.last_raw_proposals) == 1
        assert pipe.last_raw_proposals[0].metadata["anchor_class_name"] == "chair"
```

- [ ] **Step 7: Run pipeline branch test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_anchor_primary_mode_skips_sam_source_proposals -q
```

Expected: FAIL because `process_frame()` still calls `self.proposal.process()`.

- [ ] **Step 8: Route anchor-primary mode in pipeline**

In `src/pipelines/main_pipeline.py`, replace the start of Step 2 with this branch order:

```python
        proposal_source = "proposal_backend"
        source_proposals: list[Proposal2D] = []
        if self.object_anchor.enabled and getattr(self.object_anchor, "anchor_primary_mode", False):
            anchors, proposals, anchor_assignments = self.object_anchor.generate_anchor_box_proposals(frame.rgb)
            proposal_source = "anchor_box_primary"
        elif self.object_anchor.enabled and self.object_anchor.use_sam_intersection_proposals:
            sam_proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
            source_proposals = list(sam_proposals)
            anchors, proposals, anchor_assignments = self.object_anchor.generate_proposals(frame.rgb, sam_proposals)
            vote_policies = {"semantic_vote", "vote_only", "sam_mask_semantic_vote"}
            assignment_policy = str(getattr(self.object_anchor, "assignment_policy", "") or "")
            proposal_source = "anchor_sam_union"
            if assignment_policy in vote_policies or (
                proposals
                and all(
                    proposal.backend_name == "sam2_anchor_vote"
                    or proposal.metadata.get("source") == "sam2_anchor_vote"
                    for proposal in proposals
                )
            ):
                proposal_source = "sam2_anchor_vote"
        else:
            proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
            source_proposals = list(proposals)
            anchors, anchor_assignments = self.object_anchor.process(frame.rgb, proposals)
```

Ensure `Proposal2D` is imported at the top of `main_pipeline.py` if type checking complains:

```python
    Proposal2D,
```

- [ ] **Step 9: Add default disabled config**

In `configs/room0_surface_gate_4090.yaml`, under `anchor_frontend:`, add:

```yaml
  anchor_primary_mode: false
```

- [ ] **Step 10: Run anchor-primary tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_object_anchor.py::test_object_anchor_generates_anchor_box_primary_proposals tests/test_pipeline.py::TestPipeline::test_pipeline_anchor_primary_mode_skips_sam_source_proposals -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/object_anchor.py src/pipelines/main_pipeline.py configs/room0_surface_gate_4090.yaml tests/test_object_anchor.py tests/test_pipeline.py
git commit -m "feat: add anchor-primary fast proposal mode"
```

## Task 4: Representative Surface Gate Mode

**Files:**
- Modify: `src/modules/object_update.py`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing surface gate representative test**

Append this test method inside the existing `ObjectUpdateModule` test class in `tests/test_pipeline.py`. If there is no such class, append it as a top-level test function:

```python
def test_surface_owner_gate_representative_mode_uses_unique_voxel_counts():
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.5,
                "max_foreign_owner_ratio": 0.25,
                "max_background_owner_ratio": 1.0,
            },
            "tsdf": {"voxel_size": 1.0},
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    state.tsdf_volume.owner_support[(0, 0, 0)] = VoxelOwnerSupport({7: 1.0})
    state.tsdf_volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({3: 1.0})
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1],
            [1.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=1,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
    )

    filtered, structural, debug = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert filtered is not None
    assert structural is None
    assert debug["representative_voxel_mode"] is True
    assert debug["source_point_count"] == 4
    assert debug["representative_point_count"] == 2
    assert debug["foreign_owner_point_count"] == 1
    assert debug["accepted_point_count"] == 3
```

- [ ] **Step 2: Run representative test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::test_surface_owner_gate_representative_mode_uses_unique_voxel_counts -q
```

Expected: FAIL because `representative_voxel_mode` is not implemented.

- [ ] **Step 3: Add config field**

In `src/modules/object_update.py`, inside `ObjectUpdateModule.__init__` after `self.surface_owner_gate_enabled = ...`, add:

```python
        self.surface_gate_representative_voxel_mode = bool(gate_cfg.get("representative_voxel_mode", False))
```

- [ ] **Step 4: Add representative voxel lookup helper**

In `src/modules/object_update.py`, add this static method near `_points_to_voxels()`:

```python
    @staticmethod
    def _first_index_per_voxel(voxels: np.ndarray) -> np.ndarray:
        if len(voxels) == 0:
            return np.zeros(0, dtype=np.int64)
        _unique, first_indices = np.unique(voxels, axis=0, return_index=True)
        return np.sort(first_indices.astype(np.int64))
```

- [ ] **Step 5: Use representatives for gate decisions and expand back to points**

In `_filter_patch_by_surface_owner()`, after:

```python
        voxels = self._points_to_voxels(points, state.tsdf_volume.voxel_size)
```

add:

```python
        representative_indices = (
            self._first_index_per_voxel(voxels)
            if self.surface_gate_representative_voxel_mode
            else np.arange(point_count, dtype=np.int64)
        )
        decision_voxels = voxels[representative_indices]
```

Change the loop from `for idx, voxel in enumerate(voxels):` to:

```python
        decision_same_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_foreign_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_background_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        for decision_idx, voxel in enumerate(decision_voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            support = state.tsdf_volume.owner_support.get(key)
            owner_id = support.owner_id if support is not None else -1
            if allowed_owner_id is not None and owner_id == int(allowed_owner_id):
                decision_same_owner_mask[decision_idx] = True
            elif owner_id >= 0:
                decision_foreign_owner_mask[decision_idx] = True

            if self.surface_gate_background_enabled:
                decision_background_owner_mask[decision_idx] = self_structural_background or key in background_support
```

Then replace the existing `background_reject_mask` / `accepted_mask` block with:

```python
        decision_background_reject_mask = (
            np.zeros(len(decision_voxels), dtype=bool)
            if attached_override
            else decision_background_owner_mask.copy()
        )
        decision_accepted_mask = ~(decision_foreign_owner_mask | decision_background_reject_mask)

        representative_count = int(len(decision_voxels))
        decision_accepted_count = int(decision_accepted_mask.sum())
        decision_foreign_count = int(decision_foreign_owner_mask.sum())
        decision_background_count = int(decision_background_reject_mask.sum())
        decision_same_owner_count = int(decision_same_owner_mask.sum())

        if self.surface_gate_representative_voxel_mode:
            rejected_voxel_keys = {
                (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                for voxel, keep in zip(decision_voxels, decision_accepted_mask)
                if not bool(keep)
            }
            accepted_mask = np.array(
                [
                    (int(voxel[0]), int(voxel[1]), int(voxel[2])) not in rejected_voxel_keys
                    for voxel in voxels
                ],
                dtype=bool,
            )
            foreign_count = decision_foreign_count
            background_count = decision_background_count
            same_owner_count = decision_same_owner_count
            denominator = max(representative_count, 1)
        else:
            background_reject_mask = decision_background_reject_mask
            accepted_mask = decision_accepted_mask
            foreign_count = int(decision_foreign_owner_mask.sum())
            background_count = int(decision_background_reject_mask.sum())
            same_owner_count = int(decision_same_owner_mask.sum())
            denominator = max(point_count, 1)
```

Ensure `background_reject_mask` exists before structural split:

```python
        if self.surface_gate_representative_voxel_mode:
            background_voxel_keys = {
                (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                for voxel, rejected in zip(decision_voxels, decision_background_reject_mask)
                if bool(rejected)
            }
            background_reject_mask = np.array(
                [
                    (int(voxel[0]), int(voxel[1]), int(voxel[2])) in background_voxel_keys
                    for voxel in voxels
                ],
                dtype=bool,
            )
```

Compute ratios as:

```python
        accepted_count = int(accepted_mask.sum())
        accepted_ratio = float(decision_accepted_count / denominator)
        foreign_ratio = float(foreign_count / denominator)
        background_ratio = float(background_count / denominator)
```

Add to `debug`:

```python
            "representative_voxel_mode": bool(self.surface_gate_representative_voxel_mode),
            "representative_point_count": representative_count,
```

- [ ] **Step 6: Add default disabled config**

In `configs/room0_surface_gate_4090.yaml`, under `object_update.surface_owner_gate:`, add:

```yaml
    representative_voxel_mode: false
```

- [ ] **Step 7: Run representative gate tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::test_surface_owner_gate_representative_mode_uses_unique_voxel_counts -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/object_update.py configs/room0_surface_gate_4090.yaml tests/test_pipeline.py
git commit -m "feat: add representative surface gate mode"
```

## Task 5: Fast Hybrid Config and Report Summary

**Files:**
- Create: `configs/room0_fast_hybrid_4090.yaml`
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_dual_map.py`

- [ ] **Step 1: Write failing report test for timing summary**

Append this test to `tests/test_dual_map.py`:

```python
def test_write_run_report_includes_stage_timing_summary(tmp_path: Path) -> None:
    output_profile = build_output_profile(
        SimpleNamespace(
            fast_eval=True,
            full_debug_output=False,
            no_primary_exports=False,
            no_final_audit=False,
            no_reports=False,
        )
    )
    frame_metrics = [
        {
            "raw_proposal_count": 2,
            "matched_patch_count": 1,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 10,
            "frontend_stage": {
                "source_sam_proposal_count": 0,
                "anchor_voted_proposal_count": 2,
                "runtime_merged_proposal_count": 2,
                "runtime_merge_count": 0,
                "anchor_voted_unanchored_count": 0,
                "runtime_anchor_label_missing_edge_count": 0,
                "runtime_anchor_label_mismatch_edge_count": 0,
                "runtime_anchor_identity_mismatch_edge_count": 0,
            },
            "surface_owner_gate": {},
            "current_frame_visibility_gate": {},
            "stage_timings": {"proposal_generation": 0.2, "patch_lifting": 0.4},
        },
        {
            "raw_proposal_count": 4,
            "matched_patch_count": 2,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 20,
            "frontend_stage": {
                "source_sam_proposal_count": 0,
                "anchor_voted_proposal_count": 4,
                "runtime_merged_proposal_count": 4,
                "runtime_merge_count": 0,
                "anchor_voted_unanchored_count": 0,
                "runtime_anchor_label_missing_edge_count": 0,
                "runtime_anchor_label_mismatch_edge_count": 0,
                "runtime_anchor_identity_mismatch_edge_count": 0,
            },
            "surface_owner_gate": {},
            "current_frame_visibility_gate": {},
            "stage_timings": {"proposal_generation": 0.4, "patch_lifting": 0.8},
        },
    ]

    write_run_report(
        scene_dir=tmp_path,
        experiment_name="timing_report",
        config_path="configs/room0_fast_hybrid_4090.yaml",
        output_root=str(tmp_path),
        frame_limit=2,
        frame_stride=10,
        proposal_backend="precomputed",
        proposal_device="cuda",
        benchmark_audit_enabled=False,
        frame_metrics=frame_metrics,
        eval_payload={"dummy": True},
        final_audit={"objects": []},
        output_profile=output_profile,
    )

    report_md = (tmp_path / "run_report.md").read_text(encoding="utf-8")
    report_json = (tmp_path / "run_report.json").read_text(encoding="utf-8")
    assert "Stage Timing Summary" in report_md
    assert "`proposal_generation`: mean `0.3000s`" in report_md
    assert "`patch_lifting`: mean `0.6000s`" in report_md
    assert '"stage_timing_summary"' in report_json
```

- [ ] **Step 2: Run report test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_dual_map.py::test_write_run_report_includes_stage_timing_summary -q
```

Expected: FAIL because timing summary is absent.

- [ ] **Step 3: Add timing summary builder**

In `run_room0_full_eval.py`, add this function near `build_frontend_stage_report_payload()`:

```python
def build_stage_timing_summary(frame_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stage_values: dict[str, list[float]] = {}
    for metrics in frame_metrics:
        timings = dict(metrics.get("stage_timings", {}) or {})
        for stage, value in timings.items():
            stage_values.setdefault(str(stage), []).append(float(value))
    summary: dict[str, dict[str, float]] = {}
    for stage, values in sorted(stage_values.items()):
        if not values:
            continue
        summary[stage] = {
            "mean_sec": float(np.mean(values)),
            "max_sec": float(np.max(values)),
            "total_sec": float(np.sum(values)),
        }
    return summary
```

- [ ] **Step 4: Include timing summary in report**

In `write_run_report()`, after `frontend_stage_totals = ...`, add:

```python
    stage_timing_summary = build_stage_timing_summary(frame_metrics)
```

In the Markdown report lines, add a section:

```python
    if stage_timing_summary:
        lines.extend(["", "## Stage Timing Summary"])
        for stage, values in stage_timing_summary.items():
            lines.append(
                f"- `{stage}`: mean `{values['mean_sec']:.4f}s`, "
                f"max `{values['max_sec']:.4f}s`, total `{values['total_sec']:.4f}s`"
            )
```

In the JSON payload, add:

```python
        "stage_timing_summary": stage_timing_summary,
```

- [ ] **Step 5: Create fast hybrid config**

Create `configs/room0_fast_hybrid_4090.yaml` with:

```yaml
frame_input:
  image_height: 480
  image_width: 640

proposal:
  backend: precomputed
  min_mask_area: 100
  max_proposals: 50
  confidence_threshold: 0.5
  precomputed:
    cache_dir: /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2
    manifest_path: /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json
    strict: true

anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_boxes_as_primary_proposals: true
  use_sam_intersection_proposals: false
  assignment_policy: semantic_vote
  contained_subproposal_gate_enabled: true
  contained_max_anchor_coverage: 0.20
  contained_min_proposal_coverage: 0.85
  backend: yoloworld
  device: cuda:0
  model_path: /home/ww/vv/DualMapV1/model/yolov8l-world.pt
  python_executable: /home/ww/miniconda3/envs/oviovo/bin/python
  confidence_threshold: 0.2
  max_detections: 64
  proposal_min_area: 25
  classes:
  - basket
  - blanket
  - blinds
  - book
  - cabinet
  - candle
  - chair
  - cushion
  - ceiling
  - door
  - floor
  - indoor-plant
  - lamp
  - picture
  - pillar
  - plant-stand
  - plate
  - pot
  - sofa
  - stool
  - switch
  - table
  - vase
  - vent
  - wall
  - wall-plug
  - window

pipeline:
  verbose: false
  collect_stage_timings: true

runtime_vis:
  enabled: true

depth_refinement:
  min_mask_area_after_refine: 25

patch_lifting:
  min_points: 10
  proposal_parallel_enabled: true
  proposal_parallel_workers: 0
  proposal_parallel_min_tasks: 8
  point_sample_ratio: 0.08
  max_points_per_patch: 2048
  foreground_depth_filter:
    enabled: true
    front_quantile: 0.05
    depth_band: 0.08
    min_component_points: 10
    min_depth_gap: 0.15
    max_removed_ratio: 0.50

object_update:
  downsample_voxel_size: 0.02
  downsample_interval: 20
  max_points_per_object: 10000
  surface_owner_gate:
    enabled: true
    representative_voxel_mode: true
    background_enabled: true
    min_accept_points: 20
    min_update_accept_ratio: 0.30
    min_new_object_accept_ratio: 0.55
    max_foreign_owner_ratio: 0.25
    max_background_owner_ratio: 0.35
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
    downsample_voxel_size: 0.02
    max_points_per_object: 6000
    unanchored_objectness_min: 0.75
    unanchored_min_points: 128

semantic_memory:
  backend: placeholder

dual_map:
  dense_surface_cap_per_object: 4000
  dense_surface_update_interval: 5
  active_radius: 2.5
  warm_radius: 5.0

background_update:
  voxel_size: 0.05

tsdf:
  voxel_size: 0.05
  truncation_distance: 0.15
  support_increment: 1.0
  support_decay: 0.95

logging:
  level: INFO
  log_file: ""
```

- [ ] **Step 6: Run report test**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_dual_map.py::test_write_run_report_includes_stage_timing_summary -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add run_room0_full_eval.py configs/room0_fast_hybrid_4090.yaml tests/test_dual_map.py
git commit -m "feat: add fast hybrid room0 config"
```

## Task 6: Focused Verification and Smoke Experiment

**Files:**
- No code files unless tests reveal defects.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_object_anchor.py tests/test_pipeline.py::TestPatchLiftingModule tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled tests/test_pipeline.py::TestPipeline::test_pipeline_anchor_primary_mode_skips_sam_source_proposals tests/test_pipeline.py::test_surface_owner_gate_representative_mode_uses_unique_voxel_counts tests/test_dual_map.py::test_write_run_report_includes_stage_timing_summary -q
```

Expected: PASS.

- [ ] **Step 2: Run full lightweight test suite**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_object_anchor.py tests/test_dual_map.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 3: Run fast hybrid 20-frame smoke**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_fast_hybrid_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260529_room0_fast_hybrid_stride10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected:
- Exit status 0.
- `outputs/tmp_validation/20260529_room0_fast_hybrid_stride10_20f/room0/frame_metrics.jsonl` exists.
- `run_report.md` contains `Stage Timing Summary`.

- [ ] **Step 4: Summarize speed and workload**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python3 -c 'import json,statistics; p="outputs/tmp_validation/20260529_room0_fast_hybrid_stride10_20f/room0/frame_metrics.jsonl"; rows=[json.loads(l) for l in open(p) if l.strip()]; avg=lambda k: sum(r.get(k,0) for r in rows)/len(rows); print({"frames":len(rows),"avg_raw":avg("raw_proposal_count"),"avg_patch":avg("patch_count"),"max_local_memory":max(r.get("local_memory_point_count_total",0) for r in rows),"timing_keys":sorted(rows[-1].get("stage_timings",{}))})'
```

Expected:
- `avg_raw` materially lower than the previous SAM-vote smoke average of about 78.
- `avg_patch` materially lower than the previous smoke average of about 84.
- `timing_keys` includes `proposal_generation`, `patch_lifting`, and `object_update`.

- [ ] **Step 5: Commit smoke notes if any config tuning was required**

Only run this if Step 3 required a config/code fix:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add configs/room0_fast_hybrid_4090.yaml src run_room0_full_eval.py tests
git commit -m "fix: tune fast hybrid smoke behavior"
```

## Self-Review

- Spec coverage: The plan covers the requested DualMap-inspired acceleration path: detector-constrained proposals, sampled lifting, representative gate decisions, timing instrumentation, and a runnable room0 fast config.
- Placeholder scan: No `TBD`, `TODO`, or "implement later" placeholders remain. Each task has exact files, tests, implementation snippets, commands, and expected outcomes.
- Type consistency: New fields are consistently named `anchor_primary_mode`, `point_sample_ratio`, `max_points_per_patch`, `representative_voxel_mode`, and `stage_timings` across config, modules, tests, metrics, and reports.
- Risk note: `anchor_box_primary` uses rectangular detector masks as the first fast implementation. This intentionally prioritizes speed and workload reduction. A later refinement can add true box-prompt SAM/MobileSAM masks if rectangular masks are too coarse in visual evaluation.
