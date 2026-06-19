# TSDF Contamination Visibility Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent object semantic contamination caused by mask back-projection of background/structural surfaces into `local_pcd` and TSDF owner support.

**Architecture:** Add a foreground-depth core before `PatchLiftingModule` writes 3D patch points, then add a visibility/depth-consistency gate before `ObjectUpdateModule` commits patch points into `local_pcd` and TSDF. Keep DBSCAN-style cleanup out of the first pass; this plan focuses on root-cause prevention rather than post-hoc denoising.

**Tech Stack:** Python 3, NumPy, pytest, existing OVIOVO dataclasses and modules under `src/`.

---

## File Structure

- Create `src/modules/visibility_projector.py`: small NumPy-only utilities for projecting world points into a frame depth map and filtering by depth consistency.
- Modify `src/modules/patch_lifting.py`: add configurable foreground-depth filtering inside `_lift_single_proposal()` before back-projecting mask pixels.
- Modify `src/modules/object_update.py`: add optional current-frame visibility filtering before TSDF integration and `local_pcd` updates.
- Modify `src/pipelines/main_pipeline.py`: pass `frame.depth`, `frame.pose`, and `frame.intrinsics` into `ObjectUpdateModule.process()`.
- Modify `configs/default.yaml`, `configs/room0_surface_gate_4090.yaml`, `configs/midrecall_local_memory_boost.yaml`: add explicit config defaults.
- Modify `run_room0_full_eval.py`: expose new per-patch debug counters in local-memory audit records.
- Add `tests/test_visibility_projector.py`: unit tests for projection and depth consistency.
- Add focused tests to `tests/test_pipeline.py`: foreground-depth core behavior in patch lifting.
- Add focused tests to `tests/test_provisional_pool.py`: object-update commit gate behavior.

---

### Task 1: Visibility Projector Utility

**Files:**
- Create: `src/modules/visibility_projector.py`
- Test: `tests/test_visibility_projector.py`

- [ ] **Step 1: Write failing projection tests**

Create `tests/test_visibility_projector.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import CameraIntrinsics
from src.modules.visibility_projector import (
    filter_points_by_depth_consistency,
    project_world_points_to_depth,
)


def test_project_world_points_to_depth_returns_pixels_and_camera_depths() -> None:
    intr = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5)
    pose = np.eye(4, dtype=np.float64)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 2.0],
            [10.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    projection = project_world_points_to_depth(points, pose, intr)

    assert projection.valid_mask.tolist() == [True, True, True, False, False]
    assert projection.pixel_u.tolist()[:3] == [2, 3, 2]
    assert projection.pixel_v.tolist()[:3] == [2, 2, 2]
    np.testing.assert_allclose(projection.camera_depth[:3], np.array([1.0, 1.0, 2.0]), atol=1e-6)


def test_filter_points_by_depth_consistency_rejects_occluded_and_invalid_depth_points() -> None:
    intr = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5)
    pose = np.eye(4, dtype=np.float64)
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0
    depth[2, 3] = 0.7
    points = np.array(
        [
            [0.0, 0.0, 1.02],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 2.0],
        ],
        dtype=np.float32,
    )

    mask, diagnostics = filter_points_by_depth_consistency(
        points,
        depth,
        pose,
        intr,
        distance_threshold=0.05,
    )

    assert mask.tolist() == [True, False, False]
    assert diagnostics["input_point_count"] == 3
    assert diagnostics["projected_in_bounds_count"] == 3
    assert diagnostics["accepted_point_count"] == 1
    assert diagnostics["depth_rejected_point_count"] == 1
    assert diagnostics["invalid_depth_point_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_visibility_projector.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.modules.visibility_projector'`.

- [ ] **Step 3: Implement visibility projector**

Create `src/modules/visibility_projector.py` with:

```python
"""Frame-depth visibility utilities for 3D map points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.data_structures import CameraIntrinsics


@dataclass(frozen=True)
class ProjectionResult:
    pixel_u: np.ndarray
    pixel_v: np.ndarray
    camera_depth: np.ndarray
    valid_mask: np.ndarray


def project_world_points_to_depth(
    points: np.ndarray,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> ProjectionResult:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid_mask = np.zeros(len(points), dtype=bool)
    pixel_u = np.zeros(len(points), dtype=np.int32)
    pixel_v = np.zeros(len(points), dtype=np.int32)
    camera_depth = np.zeros(len(points), dtype=np.float32)
    if len(points) == 0:
        return ProjectionResult(pixel_u, pixel_v, camera_depth, valid_mask)

    world_to_camera = np.linalg.inv(np.asarray(pose, dtype=np.float64))
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    z = camera_points[:, 2]
    forward = z > 1e-6

    u_float = intrinsics.fx * (camera_points[:, 0] / np.maximum(z, 1e-6)) + intrinsics.cx
    v_float = intrinsics.fy * (camera_points[:, 1] / np.maximum(z, 1e-6)) + intrinsics.cy
    u = np.rint(u_float).astype(np.int32)
    v = np.rint(v_float).astype(np.int32)
    in_bounds = (
        forward
        & (u >= 0)
        & (u < int(intrinsics.width))
        & (v >= 0)
        & (v < int(intrinsics.height))
    )

    pixel_u[:] = u
    pixel_v[:] = v
    camera_depth[:] = z.astype(np.float32)
    valid_mask[:] = in_bounds
    return ProjectionResult(pixel_u, pixel_v, camera_depth, valid_mask)


def filter_points_by_depth_consistency(
    points: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    distance_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    depth = np.asarray(depth, dtype=np.float32)
    projection = project_world_points_to_depth(points, pose, intrinsics)
    keep = np.zeros(len(points), dtype=bool)
    if len(points) == 0:
        return keep, {
            "input_point_count": 0,
            "projected_in_bounds_count": 0,
            "accepted_point_count": 0,
            "depth_rejected_point_count": 0,
            "invalid_depth_point_count": 0,
        }

    valid_indices = np.flatnonzero(projection.valid_mask)
    measured = depth[projection.pixel_v[valid_indices], projection.pixel_u[valid_indices]]
    valid_depth = np.isfinite(measured) & (measured > 0.0)
    depth_delta = np.abs(projection.camera_depth[valid_indices] - measured)
    accepted_local = valid_depth & (depth_delta <= float(distance_threshold))
    keep[valid_indices[accepted_local]] = True

    diagnostics = {
        "input_point_count": int(len(points)),
        "projected_in_bounds_count": int(len(valid_indices)),
        "accepted_point_count": int(keep.sum()),
        "depth_rejected_point_count": int((valid_depth & ~accepted_local).sum()),
        "invalid_depth_point_count": int((~valid_depth).sum()),
    }
    return keep, diagnostics
```

- [ ] **Step 4: Run projection tests**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_visibility_projector.py -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
cd /home/ww/vv/oviovo
git add src/modules/visibility_projector.py tests/test_visibility_projector.py
git commit -m "feat: add depth visibility projector"
```

---

### Task 2: Foreground-Depth Core in Patch Lifting

**Files:**
- Modify: `src/modules/patch_lifting.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing patch-lifting tests**

Add `RefinedProposal2D` to the dataclass import list near the top of `tests/test_pipeline.py`:

```python
    RefinedProposal2D,
```

Add `PatchLiftingModule` to the module import list near the top of `tests/test_pipeline.py`:

```python
from src.modules.patch_lifting import PatchLiftingModule
```

Append these tests inside `class TestRoadmapRefactor`:

```python
    def test_patch_lifting_foreground_depth_core_discards_background_tail(self):
        depth = np.full((6, 6), 2.0, dtype=np.float32)
        depth[2:4, 2:4] = 1.0
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:5, 1:5] = True
        proposal = RefinedProposal2D(
            proposal_id=12,
            mask=mask,
            bbox_xyxy=np.array([1, 1, 5, 5], dtype=np.float32),
            area=int(mask.sum()),
            metadata={"anchor_class_name": "lamp", "anchor_confidence": 0.9},
        )
        module = PatchLiftingModule(
            {
                "min_points": 1,
                "foreground_depth_filter": {
                    "enabled": True,
                    "front_quantile": 0.05,
                    "depth_band": 0.08,
                    "min_component_points": 1,
                },
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=3.0, cy=3.0, width=6, height=6),
            frame_id=0,
        )

        assert len(patches) == 1
        np.testing.assert_allclose(patches[0].points[:, 2], np.ones(4, dtype=np.float32), atol=1e-6)
        assert patches[0].metadata["foreground_depth_filter"]["enabled"] is True
        assert patches[0].metadata["foreground_depth_filter"]["removed_point_count"] == 12
        assert patches[0].metadata["lifted_point_count"] == 4

    def test_patch_lifting_foreground_depth_core_falls_back_when_too_few_points_remain(self):
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        depth[1, 1] = 1.0
        mask = np.ones((4, 4), dtype=bool)
        proposal = RefinedProposal2D(
            proposal_id=13,
            mask=mask,
            bbox_xyxy=np.array([0, 0, 4, 4], dtype=np.float32),
            area=int(mask.sum()),
        )
        module = PatchLiftingModule(
            {
                "min_points": 8,
                "foreground_depth_filter": {
                    "enabled": True,
                    "front_quantile": 0.05,
                    "depth_band": 0.05,
                    "min_component_points": 8,
                },
            }
        )

        patches = module.process(
            [proposal],
            depth,
            np.eye(4, dtype=np.float64),
            CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4),
            frame_id=0,
        )

        assert len(patches) == 1
        assert len(patches[0].points) == 16
        assert patches[0].metadata["foreground_depth_filter"]["fallback_reason"] == "insufficient_foreground_points"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_patch_lifting_foreground_depth_core_discards_background_tail tests/test_pipeline.py::TestRoadmapRefactor::test_patch_lifting_foreground_depth_core_falls_back_when_too_few_points_remain -q
```

Expected: FAIL because `PatchLiftingModule` does not yet read `foreground_depth_filter` metadata.

- [ ] **Step 3: Add config fields in `PatchLiftingModule.__init__`**

In `src/modules/patch_lifting.py`, extend `__init__` after `self.proposal_parallel_min_tasks`:

```python
        fg_cfg = config.get("foreground_depth_filter", {})
        self.foreground_depth_filter_enabled = bool(fg_cfg.get("enabled", False))
        self.foreground_front_quantile = float(fg_cfg.get("front_quantile", 0.05))
        self.foreground_depth_band = float(fg_cfg.get("depth_band", 0.08))
        self.foreground_min_component_points = int(fg_cfg.get("min_component_points", self.min_points))
```

- [ ] **Step 4: Add foreground mask helper**

In `src/modules/patch_lifting.py`, add this method before `_get_pixel_grid()`:

```python
    def _foreground_depth_mask(self, z: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        z = np.asarray(z, dtype=np.float64)
        diagnostics = {
            "enabled": bool(self.foreground_depth_filter_enabled),
            "input_point_count": int(z.size),
            "kept_point_count": int(z.size),
            "removed_point_count": 0,
            "front_depth": 0.0,
            "depth_band": float(self.foreground_depth_band),
            "fallback_reason": "",
        }
        if not self.foreground_depth_filter_enabled or z.size == 0:
            return np.ones(z.shape, dtype=bool), diagnostics

        front_quantile = float(np.clip(self.foreground_front_quantile, 0.0, 1.0))
        front_depth = float(np.quantile(z, front_quantile))
        keep = z <= front_depth + float(self.foreground_depth_band)
        kept_count = int(keep.sum())
        diagnostics["front_depth"] = front_depth
        diagnostics["kept_point_count"] = kept_count
        diagnostics["removed_point_count"] = int(z.size - kept_count)

        min_points = max(int(self.min_points), int(self.foreground_min_component_points))
        if kept_count < min_points:
            diagnostics["kept_point_count"] = int(z.size)
            diagnostics["removed_point_count"] = 0
            diagnostics["fallback_reason"] = "insufficient_foreground_points"
            return np.ones(z.shape, dtype=bool), diagnostics

        return keep, diagnostics
```

- [ ] **Step 5: Apply foreground mask before back-projection**

In `_lift_single_proposal()` after:

```python
        z = depth[valid].astype(np.float64)
        if z.size < self.min_points:
            return None
```

insert:

```python
        foreground_mask, foreground_debug = self._foreground_depth_mask(z)
        if not np.all(foreground_mask):
            u = u[foreground_mask]
            v = v[foreground_mask]
            z = z[foreground_mask]
        else:
            foreground_debug["kept_point_count"] = int(z.size)
            foreground_debug["removed_point_count"] = int(foreground_debug.get("input_point_count", z.size) - z.size)
        if z.size < self.min_points:
            return None
```

Add this key in the returned patch metadata near `"lifted_point_count"`:

```python
                "foreground_depth_filter": foreground_debug,
```

- [ ] **Step 6: Run patch-lifting tests**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_patch_lifting_foreground_depth_core_discards_background_tail tests/test_pipeline.py::TestRoadmapRefactor::test_patch_lifting_foreground_depth_core_falls_back_when_too_few_points_remain -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 7: Commit Task 2**

```bash
cd /home/ww/vv/oviovo
git add src/modules/patch_lifting.py tests/test_pipeline.py
git commit -m "feat: filter foreground depth before patch lifting"
```

---

### Task 3: Current-Frame Visibility Gate Before Object Commit

**Files:**
- Modify: `src/modules/object_update.py`
- Test: `tests/test_provisional_pool.py`

- [ ] **Step 1: Add failing object-update visibility tests**

Add `CameraIntrinsics` to the import list in `tests/test_provisional_pool.py`:

```python
    CameraIntrinsics,
```

Append this test after the existing surface owner gate tests:

```python
def test_current_frame_visibility_gate_writes_only_depth_consistent_points() -> None:
    module = ObjectUpdateModule(
        {
            "downsample_interval": 99,
            "surface_owner_gate": {"enabled": False},
            "current_frame_visibility_gate": {
                "enabled": True,
                "distance_threshold": 0.05,
                "min_accept_points": 1,
                "min_accept_ratio": 0.25,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 2.0],
            [0.0, -0.1, 1.0],
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=33,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=3,
    )
    existing_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    existing = ObjectMap(
        object_id=4,
        local_pcd=existing_points.copy(),
        centroid=existing_points[0],
        bbox_min=existing_points[0],
        bbox_max=existing_points[0],
    )
    state = SystemState(objects={4: existing}, next_object_id=5)
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0
    depth[2, 3] = 0.5
    depth[3, 3] = 1.0

    updated = module.process(
        AssociationResult(matched=[(33, 4, None)]),
        [patch],
        state,
        current_depth=depth,
        current_pose=np.eye(4, dtype=np.float64),
        current_intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5),
    )

    obj = updated.objects[4]
    assert len(obj.local_pcd) == 3
    np.testing.assert_allclose(obj.local_pcd[1:], np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], dtype=np.float32))
    gate = obj.debug["last_current_frame_visibility_gate"]
    assert gate["accepted_point_count"] == 2
    assert gate["depth_rejected_point_count"] == 1
    assert gate["invalid_depth_point_count"] == 1


def test_current_frame_visibility_gate_rejects_patch_when_too_few_points_remain() -> None:
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {"enabled": False},
            "current_frame_visibility_gate": {
                "enabled": True,
                "distance_threshold": 0.05,
                "min_accept_points": 2,
                "min_accept_ratio": 0.75,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=34, frame_id=3)
    existing_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    existing = ObjectMap(
        object_id=4,
        local_pcd=existing_points.copy(),
        centroid=existing_points[0],
        bbox_min=existing_points[0],
        bbox_max=existing_points[0],
    )
    state = SystemState(objects={4: existing}, next_object_id=5)
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0

    updated = module.process(
        AssociationResult(matched=[(34, 4, None)]),
        [patch],
        state,
        current_depth=depth,
        current_pose=np.eye(4, dtype=np.float64),
        current_intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5),
    )

    assert np.array_equal(updated.objects[4].local_pcd, existing_points)
    assert module.last_current_frame_visibility_gate_stats["rejected_patch_count"] == 1
    assert "low_accept_ratio" in updated.objects[4].debug["last_current_frame_visibility_gate"]["rejection_reasons"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_provisional_pool.py::test_current_frame_visibility_gate_writes_only_depth_consistent_points tests/test_provisional_pool.py::test_current_frame_visibility_gate_rejects_patch_when_too_few_points_remain -q
```

Expected: FAIL with `TypeError: ObjectUpdateModule.process() got an unexpected keyword argument 'current_depth'`.

- [ ] **Step 3: Add object-update config fields and stats**

In `src/modules/object_update.py`, import the shared types and utility:

```python
from src.core.data_structures import CameraIntrinsics
from src.modules.visibility_projector import filter_points_by_depth_consistency
```

In `ObjectUpdateModule.__init__`, after surface-owner config setup, add:

```python
        visibility_cfg = config.get("current_frame_visibility_gate", {})
        self.current_frame_visibility_gate_enabled = bool(visibility_cfg.get("enabled", False))
        self.current_frame_visibility_distance_threshold = float(visibility_cfg.get("distance_threshold", 0.05))
        self.current_frame_visibility_min_accept_points = int(visibility_cfg.get("min_accept_points", 20))
        self.current_frame_visibility_min_accept_ratio = float(visibility_cfg.get("min_accept_ratio", 0.30))
        self.last_current_frame_visibility_gate_records: list[dict[str, Any]] = []
        self.last_current_frame_visibility_gate_stats: dict[str, Any] = {}
```

- [ ] **Step 4: Extend `process()` signature**

Change `ObjectUpdateModule.process()` signature to:

```python
    def process(
        self,
        association: AssociationResult,
        patches: List[Patch3D],
        state: SystemState,
        background_patches: List[Patch3D] | None = None,
        *,
        current_depth: np.ndarray | None = None,
        current_pose: np.ndarray | None = None,
        current_intrinsics: CameraIntrinsics | None = None,
    ) -> SystemState:
```

At the start of `process()`, after `self.last_surface_gate_records = []`, add:

```python
        self.last_current_frame_visibility_gate_records = []
```

- [ ] **Step 5: Add visibility gate helper methods**

Add these methods before `_build_background_voxel_support()`:

```python
    def _filter_patch_by_current_frame_visibility(
        self,
        patch: Patch3D,
        *,
        current_depth: np.ndarray | None,
        current_pose: np.ndarray | None,
        current_intrinsics: CameraIntrinsics | None,
    ) -> tuple[Patch3D | None, dict[str, Any]]:
        point_count = int(len(patch.points))
        if (
            not self.current_frame_visibility_gate_enabled
            or current_depth is None
            or current_pose is None
            or current_intrinsics is None
            or point_count == 0
        ):
            debug = {
                "enabled": bool(self.current_frame_visibility_gate_enabled),
                "source_patch_id": int(patch.patch_id),
                "source_point_count": point_count,
                "projected_in_bounds_count": point_count,
                "depth_rejected_point_count": 0,
                "invalid_depth_point_count": 0,
                "accepted_point_count": point_count,
                "accepted_ratio": 1.0 if point_count > 0 else 0.0,
                "passed": True,
                "rejection_reasons": [],
            }
            return patch, debug

        keep_mask, diagnostics = filter_points_by_depth_consistency(
            patch.points,
            current_depth,
            current_pose,
            current_intrinsics,
            distance_threshold=self.current_frame_visibility_distance_threshold,
        )
        accepted_count = int(keep_mask.sum())
        accepted_ratio = float(accepted_count / max(point_count, 1))
        rejection_reasons: list[str] = []
        if accepted_count < self.current_frame_visibility_min_accept_points:
            rejection_reasons.append("insufficient_accepted_points")
        if accepted_ratio < self.current_frame_visibility_min_accept_ratio:
            rejection_reasons.append("low_accept_ratio")

        debug = {
            "enabled": True,
            "source_patch_id": int(patch.patch_id),
            "source_point_count": point_count,
            "accepted_point_count": accepted_count,
            "accepted_ratio": accepted_ratio,
            "distance_threshold": float(self.current_frame_visibility_distance_threshold),
            "min_accept_points": int(self.current_frame_visibility_min_accept_points),
            "min_accept_ratio": float(self.current_frame_visibility_min_accept_ratio),
            "passed": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            **diagnostics,
        }
        if rejection_reasons:
            return None, debug
        return self._copy_patch_with_filtered_points(
            patch,
            keep_mask,
            debug,
            metadata_key="current_frame_visibility_gate",
        ), debug

    def _record_current_frame_visibility_gate_debug(self, debug: dict[str, Any]) -> None:
        self.last_current_frame_visibility_gate_records.append(dict(debug))

    def _summarize_current_frame_visibility_gate_records(self) -> dict[str, Any]:
        records = list(self.last_current_frame_visibility_gate_records)
        if not records:
            return {"enabled": bool(self.current_frame_visibility_gate_enabled), "record_count": 0}
        return {
            "enabled": bool(self.current_frame_visibility_gate_enabled),
            "record_count": int(len(records)),
            "rejected_patch_count": int(sum(1 for record in records if not record.get("passed", True))),
            "source_point_count": int(sum(int(record.get("source_point_count", 0)) for record in records)),
            "accepted_point_count": int(sum(int(record.get("accepted_point_count", 0)) for record in records)),
            "depth_rejected_point_count": int(sum(int(record.get("depth_rejected_point_count", 0)) for record in records)),
            "invalid_depth_point_count": int(sum(int(record.get("invalid_depth_point_count", 0)) for record in records)),
        }
```

- [ ] **Step 6: Keep gate metadata separate, then call visibility gate before surface owner gate**

Before wiring the call sites, update `_copy_patch_with_filtered_points()` so visibility and surface-owner gate metadata do not overwrite each other. Change the signature to:

```python
    def _copy_patch_with_filtered_points(
        self,
        patch: Patch3D,
        mask: np.ndarray,
        gate_debug: dict[str, Any],
        *,
        metadata_key: str = "surface_owner_gate",
    ) -> Patch3D | None:
```

Inside that helper, replace:

```python
        metadata["surface_owner_gate"] = gate_debug
```

with:

```python
        metadata[metadata_key] = gate_debug
```

In all three `process()` lanes that currently call `_filter_patch_by_surface_owner()`:

1. matched object lane,
2. provisional new object lane,
3. non-provisional new object lane,

insert this before `_filter_patch_by_surface_owner()`:

```python
            patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                patch,
                current_depth=current_depth,
                current_pose=current_pose,
                current_intrinsics=current_intrinsics,
            )
            self._record_current_frame_visibility_gate_debug(visibility_debug)
            if patch is None:
                obj.debug["last_current_frame_visibility_gate"] = visibility_debug
                continue
```

For new-object lanes where no `obj` exists, use:

```python
            patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                patch,
                current_depth=current_depth,
                current_pose=current_pose,
                current_intrinsics=current_intrinsics,
            )
            self._record_current_frame_visibility_gate_debug(visibility_debug)
            if patch is None:
                continue
```

After `self.last_surface_gate_stats = self._summarize_surface_gate_records()`, add:

```python
        self.last_current_frame_visibility_gate_stats = self._summarize_current_frame_visibility_gate_records()
```

After successful matched-object update, set:

```python
            obj.debug["last_current_frame_visibility_gate"] = visibility_debug
```

After successful new-object creation, set:

```python
                new_obj.debug["last_current_frame_visibility_gate"] = visibility_debug
```

- [ ] **Step 7: Run object-update tests**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_provisional_pool.py::test_current_frame_visibility_gate_writes_only_depth_consistent_points tests/test_provisional_pool.py::test_current_frame_visibility_gate_rejects_patch_when_too_few_points_remain -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 8: Commit Task 3**

```bash
cd /home/ww/vv/oviovo
git add src/modules/object_update.py tests/test_provisional_pool.py
git commit -m "feat: gate object commits by current frame depth"
```

---

### Task 4: Pipeline Wiring and Config Defaults

**Files:**
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `configs/default.yaml`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Modify: `configs/midrecall_local_memory_boost.yaml`

- [ ] **Step 1: Wire current frame into object update**

In `src/pipelines/main_pipeline.py`, change the call at Step 9 to:

```python
        self.state = self.object_update.process(
            association,
            association_patches,
            self.state,
            background_patches=bg_patches,
            current_depth=frame.depth,
            current_pose=frame.pose,
            current_intrinsics=frame.intrinsics,
        )
```

- [ ] **Step 2: Add default patch-lifting config**

In each config file under `patch_lifting:`, add:

```yaml
  foreground_depth_filter:
    enabled: true
    front_quantile: 0.05
    depth_band: 0.08
    min_component_points: 10
```

- [ ] **Step 3: Add object-update visibility gate config**

In each config file under `object_update:`, next to `surface_owner_gate:`, add:

```yaml
  current_frame_visibility_gate:
    enabled: true
    distance_threshold: 0.05
    min_accept_points: 20
    min_accept_ratio: 0.30
```

- [ ] **Step 4: Run regression tests covering touched modules**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_visibility_projector.py tests/test_pipeline.py::TestRoadmapRefactor tests/test_provisional_pool.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
cd /home/ww/vv/oviovo
git add src/pipelines/main_pipeline.py configs/default.yaml configs/room0_surface_gate_4090.yaml configs/midrecall_local_memory_boost.yaml
git commit -m "config: enable foreground and visibility contamination gates"
```

---

### Task 5: Audit Counters for Contamination Gates

**Files:**
- Modify: `run_room0_full_eval.py`
- Test: existing unit tests plus a smoke run if compute is available

- [ ] **Step 1: Add patch metadata fields to local-memory audit**

In `build_local_memory_frame_audit()`, inside the `patch_record` dictionary built for each related patch, add:

```python
                    "foreground_depth_filter": dict(patch.metadata.get("foreground_depth_filter", {}) or {}),
                    "surface_owner_gate": dict(patch.metadata.get("surface_owner_gate", {}) or {}),
                    "current_frame_visibility_gate": dict(patch.metadata.get("current_frame_visibility_gate", {}) or {}),
```

Use the existing `patch.metadata` source. Do not add new dependencies.

- [ ] **Step 2: Add frame-level visibility gate stats**

In the returned dictionary from `build_local_memory_frame_audit()`, add:

```python
        "current_frame_visibility_gate": dict(
            getattr(pipeline.object_update, "last_current_frame_visibility_gate_stats", {}) or {}
        ),
```

- [ ] **Step 3: Add report-level totals**

In `write_run_report()`, compute these values from `frame_metrics` or audit summaries where structural diagnostics are currently assembled:

```python
    current_frame_visibility_rejected_patch_total = int(
        sum(
            int(metrics.get("current_frame_visibility_gate", {}).get("rejected_patch_count", 0))
            for metrics in frame_metrics
        )
    )
    current_frame_visibility_depth_rejected_point_total = int(
        sum(
            int(metrics.get("current_frame_visibility_gate", {}).get("depth_rejected_point_count", 0))
            for metrics in frame_metrics
        )
    )
```

Add both fields to the markdown report and JSON report near the existing structural purity diagnostics:

```python
        f"- `current_frame_visibility_rejected_patch_total`: `{current_frame_visibility_rejected_patch_total}`",
        f"- `current_frame_visibility_depth_rejected_point_total`: `{current_frame_visibility_depth_rejected_point_total}`",
```

and:

```python
        "current_frame_visibility_rejected_patch_total": current_frame_visibility_rejected_patch_total,
        "current_frame_visibility_depth_rejected_point_total": current_frame_visibility_depth_rejected_point_total,
```

- [ ] **Step 4: Run import and existing audit tests**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor tests/test_provisional_pool.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
cd /home/ww/vv/oviovo
git add run_room0_full_eval.py
git commit -m "chore: audit contamination gate activity"
```

---

### Task 6: Validation Run and Decision Criteria

**Files:**
- No source files required.
- Output: new run under `outputs/tmp_validation/`.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_visibility_projector.py tests/test_pipeline.py::TestRoadmapRefactor tests/test_provisional_pool.py -q
```

Expected: PASS.

- [ ] **Step 2: Run room0 200-frame validation**

Run:

```bash
cd /home/ww/vv/oviovo
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config configs/room0_surface_gate_4090.yaml \
  --experiment-name 20260522_room0_stride10_200f_foreground_visibility_gate \
  --frame-limit 200 \
  --stride 10
```

Expected: run completes and writes:

```text
outputs/tmp_validation/20260522_room0_stride10_200f_foreground_visibility_gate/room0/run_report.md
outputs/tmp_validation/20260522_room0_stride10_200f_foreground_visibility_gate/room0/final_object_semantic_audit.md
```

- [ ] **Step 3: Compare target diagnostics**

Inspect these values in the new `run_report.md`:

```text
mIoU
f-mIoU
surface_gate_rejected_patch_total
current_frame_visibility_rejected_patch_total
current_frame_visibility_depth_rejected_point_total
pool_point_count
projected_dense_labeled_point_count
```

Acceptance criteria for the first pass:

```text
current_frame_visibility_depth_rejected_point_total > 0
mIoU >= 0.4733 - 0.02
f-mIoU >= 0.6382 - 0.02
projected_dense_labeled_point_count does not drop by more than 20%
```

Inspect these rows in `final_object_semantic_audit.md`:

```text
object 10: floor majority should decrease or object should lose cabinet dominance
object 20: floor majority should decrease or object should be rejected/split
object 119: lamp absorbed by ceiling should decrease
class lamp: absorbed_by_other_majority should decrease from 14872
class wall-plug/switch: should not regress by losing all attached tiny objects
```

- [ ] **Step 4: Commit validation notes**

Create `outputs/tmp_validation/20260522_room0_stride10_200f_foreground_visibility_gate/room0/validation_notes.md` with a short summary of the above values, then commit only if project policy allows committing output notes:

```bash
cd /home/ww/vv/oviovo
git add outputs/tmp_validation/20260522_room0_stride10_200f_foreground_visibility_gate/room0/validation_notes.md
git commit -m "docs: record foreground visibility gate validation"
```

If output artifacts are not committed in this repository, leave the notes uncommitted and include the numbers in the final handoff.

---

## Self-Review

- Spec coverage: foreground-depth prevention is covered by Task 2; visibility/depth-consistency gate is covered by Tasks 1, 3, and 4; auditability and validation are covered by Tasks 5 and 6.
- Placeholder scan: the plan contains no placeholder markers, no open-ended test requests, and no undefined function names. Every new function referenced in tests is defined in a task.
- Type consistency: `CameraIntrinsics`, `Patch3D`, `RefinedProposal2D`, `SystemState`, and `AssociationResult` match existing dataclasses. New `ObjectUpdateModule.process()` arguments are keyword-only and optional, so existing tests and callers remain compatible.
