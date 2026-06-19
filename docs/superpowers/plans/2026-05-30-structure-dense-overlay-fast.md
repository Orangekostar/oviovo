# Structure Dense Overlay Fast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast structural overlay path that recovers `wall`, `window`, `blinds`, `ceiling`, `floor`, and `door` IoU while preserving the current fast YOLO/YOLOE anchor-primary object map.

**Architecture:** Keep anchor-box proposals as the only proposals that flow through RuntimeVis, depth refinement, association, and object update. In parallel, consume only existing precomputed SAM proposals plus detector anchors to accumulate a separate voxel-level structural overlay, then conservatively fuse that overlay into dense projection and evaluation. Synthetic negative ids represent structure labels during export/evaluation so real object ids keep the existing GT-majority mapping.

**Tech Stack:** Python 3, NumPy, SciPy KDTree, pytest, YAML configs, existing OVIOVO pipeline modules, precomputed proposal cache.

---

## File Structure

- Modify `src/core/data_structures.py`
  - Add `StructuralOverlayVoxel` and `StructuralOverlayMap`.
  - Add `SystemState.structural_overlay_map`.
- Create `src/modules/structural_overlay.py`
  - Build structure voxel votes from `Frame`, `Anchor2D`, and `Proposal2D`.
  - Clip SAM masks to structure anchor boxes before lifting.
  - Track per-frame skip and vote counts.
- Modify `src/pipelines/main_pipeline.py`
  - Instantiate `StructuralOverlayModule`.
  - Add a `structural_overlay` timing stage.
  - In anchor-primary fast mode, load precomputed SAM proposals for overlay only.
  - Keep overlay proposals out of RuntimeVis, association, and object update.
- Modify `run_room0_full_eval.py`
  - Add synthetic structural id helpers.
  - Project overlay voxels to dense geometry points.
  - Conservatively fuse overlay labels with object projection.
  - Evaluate synthetic overlay ids by direct semantic class mapping.
  - Export `room0_structural_overlay.ply` and report overlay/fusion counts.
- Create `configs/room0_fast_structural_overlay_4090.yaml`
  - Copy the fast-hybrid config and enable structural overlay with conservative coverage.
- Create `tests/test_structural_overlay.py`
  - Unit tests for mask clipping, filtering, skip counts, and voxel voting.
- Modify `tests/test_dual_map.py`
  - Tests for conservative fusion, synthetic id color/eval behavior, and report payload.
- Modify `tests/test_pipeline.py`
  - Pipeline wiring test proving overlay consumes precomputed SAM proposals but does not feed association.

## Task 1: Structural Overlay State

**Files:**
- Modify: `src/core/data_structures.py`
- Test: `tests/test_structural_overlay.py`

- [ ] **Step 1: Write failing tests for overlay state**

Create `tests/test_structural_overlay.py` with this initial content:

```python
"""Tests for the structural dense overlay layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import StructuralOverlayMap, StructuralOverlayVoxel, SystemState


def test_structural_overlay_voxel_returns_top_label_and_support() -> None:
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("wall", 0.25, frame_id=3)
    voxel.add_vote("window", 0.75, frame_id=4)
    voxel.add_vote("wall", 0.75, frame_id=5)

    assert voxel.top_label == "wall"
    assert voxel.top_support == 1.0
    assert voxel.observation_count == 3
    assert voxel.last_seen_frame == 5


def test_system_state_owns_empty_structural_overlay_map() -> None:
    state = SystemState()

    assert isinstance(state.structural_overlay_map, StructuralOverlayMap)
    assert state.structural_overlay_map.voxel_size == 0.05
    assert state.structural_overlay_map.voxels == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_structural_overlay.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` because structural overlay dataclasses do not exist.

- [ ] **Step 3: Add structural overlay dataclasses**

In `src/core/data_structures.py`, after `DenseSurfaceMap`, add:

```python
@dataclass
class StructuralOverlayVoxel:
    """Voxel-level semantic votes for stuff-like structural classes."""
    label_votes: Dict[str, float] = field(default_factory=dict)
    observation_count: int = 0
    last_seen_frame: int = -1

    def add_vote(self, label: str, weight: float, frame_id: int) -> None:
        normalized = str(label).strip().lower()
        if not normalized:
            return
        self.label_votes[normalized] = float(self.label_votes.get(normalized, 0.0) + float(weight))
        self.observation_count += 1
        self.last_seen_frame = max(int(self.last_seen_frame), int(frame_id))

    @property
    def top_label(self) -> str:
        if not self.label_votes:
            return ""
        return max(self.label_votes, key=self.label_votes.get)

    @property
    def top_support(self) -> float:
        if not self.label_votes:
            return 0.0
        return float(max(self.label_votes.values()))


@dataclass
class StructuralOverlayMap:
    """Sparse dense overlay for planar or stuff-like structure labels."""
    voxel_size: float = 0.05
    voxels: Dict[Tuple[int, int, int], StructuralOverlayVoxel] = field(default_factory=dict)
    update_count: int = 0
```

In `SystemState`, add this field after `dense_surface_map`:

```python
    structural_overlay_map: StructuralOverlayMap = field(default_factory=StructuralOverlayMap)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_structural_overlay.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/core/data_structures.py tests/test_structural_overlay.py
git commit -m "feat: add structural overlay state"
```

## Task 2: Structural Overlay Module

**Files:**
- Create: `src/modules/structural_overlay.py`
- Modify: `tests/test_structural_overlay.py`

- [ ] **Step 1: Add failing module tests**

Append these tests to `tests/test_structural_overlay.py`:

```python
from src.core.data_structures import Anchor2D, CameraIntrinsics, Frame, Proposal2D
from src.modules.structural_overlay import StructuralOverlayModule


def _frame(height: int = 8, width: int = 8) -> Frame:
    return Frame(
        frame_id=7,
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth=np.ones((height, width), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=0.0, cy=0.0, width=width, height=height),
        source_frame_id=70,
    )


def test_structural_overlay_clips_proposal_to_anchor_box() -> None:
    frame = _frame()
    anchor = Anchor2D(
        anchor_id=1,
        bbox_xyxy=np.array([2, 2, 5, 5], dtype=np.float32),
        class_name="window",
        confidence=0.8,
    )
    mask = np.ones((8, 8), dtype=bool)
    proposal = Proposal2D(
        proposal_id=10,
        mask=mask,
        bbox_xyxy=np.array([0, 0, 8, 8], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        backend_name="precomputed",
    )
    overlay = StructuralOverlayMap(voxel_size=0.001)
    module = StructuralOverlayModule(
        {
            "enabled": True,
            "classes": ["window"],
            "voxel_size": 0.001,
            "min_overlap_area": 1,
            "min_proposal_anchor_coverage": 0.01,
            "min_anchor_proposal_coverage": 0.01,
            "clip_to_anchor_box": True,
        }
    )

    summary = module.process(frame, [anchor], [proposal], overlay)

    assert summary["enabled"] is True
    assert summary["structure_anchor_count"] == 1
    assert summary["accepted_pair_count"] == 1
    assert summary["voted_pixel_count"] == 9
    assert len(overlay.voxels) == 9
    assert {voxel.top_label for voxel in overlay.voxels.values()} == {"window"}


def test_structural_overlay_ignores_non_structure_anchors() -> None:
    frame = _frame()
    anchor = Anchor2D(
        anchor_id=2,
        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
        class_name="chair",
        confidence=0.9,
    )
    mask = np.ones((8, 8), dtype=bool)
    proposal = Proposal2D(3, mask, np.array([0, 0, 8, 8], dtype=np.float32), int(mask.sum()))
    overlay = StructuralOverlayMap(voxel_size=0.01)
    module = StructuralOverlayModule({"enabled": True, "classes": ["wall"], "min_overlap_area": 1})

    summary = module.process(frame, [anchor], [proposal], overlay)

    assert summary["structure_anchor_count"] == 0
    assert summary["skip_reason"] == "no_structure_anchors"
    assert overlay.voxels == {}


def test_structural_overlay_skips_mismatched_proposal_masks() -> None:
    frame = _frame(height=8, width=8)
    anchor = Anchor2D(1, np.array([0, 0, 8, 8], dtype=np.float32), "wall", 0.7)
    bad_mask = np.ones((4, 4), dtype=bool)
    proposal = Proposal2D(5, bad_mask, np.array([0, 0, 4, 4], dtype=np.float32), int(bad_mask.sum()))
    overlay = StructuralOverlayMap(voxel_size=0.01)
    module = StructuralOverlayModule({"enabled": True, "classes": ["wall"], "min_overlap_area": 1})

    summary = module.process(frame, [anchor], [proposal], overlay)

    assert summary["mismatched_mask_count"] == 1
    assert summary["accepted_pair_count"] == 0
    assert overlay.voxels == {}
```

- [ ] **Step 2: Run module tests to verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_structural_overlay.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.modules.structural_overlay'`.

- [ ] **Step 3: Implement `StructuralOverlayModule`**

Create `src/modules/structural_overlay.py`:

```python
"""Dense structural overlay from existing anchors and precomputed SAM proposals."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.data_structures import Anchor2D, Frame, Proposal2D, StructuralOverlayMap, StructuralOverlayVoxel


class StructuralOverlayModule:
    """Accumulates voxel votes for structure labels without creating object instances."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.classes = {str(label).strip().lower() for label in self.config.get("classes", []) if str(label).strip()}
        self.voxel_size = float(self.config.get("voxel_size", 0.05))
        self.min_overlap_area = int(self.config.get("min_overlap_area", 25))
        self.min_proposal_anchor_coverage = float(self.config.get("min_proposal_anchor_coverage", 0.20))
        self.min_anchor_proposal_coverage = float(self.config.get("min_anchor_proposal_coverage", 0.03))
        self.clip_to_anchor_box = bool(self.config.get("clip_to_anchor_box", True))
        self.pixel_sample_stride = max(1, int(self.config.get("pixel_sample_stride", 1)))
        self.max_pixels_per_pair = max(0, int(self.config.get("max_pixels_per_pair", 0)))
        self.last_summary: dict[str, Any] = {"enabled": self.enabled}

    def process(
        self,
        frame: Frame,
        anchors: list[Anchor2D],
        proposals: list[Proposal2D],
        overlay_map: StructuralOverlayMap,
    ) -> dict[str, Any]:
        summary = {
            "enabled": bool(self.enabled),
            "structure_anchor_count": 0,
            "sam_proposal_count": int(len(proposals)),
            "accepted_pair_count": 0,
            "mismatched_mask_count": 0,
            "voted_pixel_count": 0,
            "new_voxel_count": 0,
            "updated_voxel_count": 0,
            "skip_reason": "",
        }
        if not self.enabled:
            summary["skip_reason"] = "disabled"
            self.last_summary = summary
            return summary

        overlay_map.voxel_size = float(self.voxel_size)
        structure_anchors = [
            anchor for anchor in anchors if str(anchor.class_name).strip().lower() in self.classes
        ]
        summary["structure_anchor_count"] = int(len(structure_anchors))
        if not structure_anchors:
            summary["skip_reason"] = "no_structure_anchors"
            self.last_summary = summary
            return summary
        if not proposals:
            summary["skip_reason"] = "no_sam_proposals"
            self.last_summary = summary
            return summary

        valid_depth = np.isfinite(frame.depth) & (frame.depth > 0)
        initial_voxel_count = len(overlay_map.voxels)
        for anchor in structure_anchors:
            anchor_mask = self._anchor_box_mask(anchor.bbox_xyxy, frame.depth.shape)
            anchor_area = int(np.count_nonzero(anchor_mask))
            if anchor_area <= 0:
                continue
            label = str(anchor.class_name).strip().lower()
            for proposal in proposals:
                if proposal.mask.shape != frame.depth.shape:
                    summary["mismatched_mask_count"] += 1
                    continue
                proposal_mask = np.asarray(proposal.mask, dtype=bool)
                overlap = proposal_mask & anchor_mask
                overlap_area = int(np.count_nonzero(overlap))
                if overlap_area < self.min_overlap_area:
                    continue
                proposal_area = max(int(np.count_nonzero(proposal_mask)), 1)
                proposal_anchor_coverage = overlap_area / float(proposal_area)
                anchor_proposal_coverage = overlap_area / float(anchor_area)
                if proposal_anchor_coverage < self.min_proposal_anchor_coverage:
                    continue
                if anchor_proposal_coverage < self.min_anchor_proposal_coverage:
                    continue

                mask = overlap if self.clip_to_anchor_box else proposal_mask
                voted = self._vote_mask(frame, mask & valid_depth, overlay_map, label, anchor, proposal)
                if voted <= 0:
                    continue
                summary["accepted_pair_count"] += 1
                summary["voted_pixel_count"] += int(voted)

        summary["updated_voxel_count"] = int(len(overlay_map.voxels))
        summary["new_voxel_count"] = int(max(0, len(overlay_map.voxels) - initial_voxel_count))
        overlay_map.update_count += 1
        if summary["accepted_pair_count"] == 0 and not summary["skip_reason"]:
            summary["skip_reason"] = "no_matching_structure_pairs"
        self.last_summary = summary
        return summary

    def _vote_mask(
        self,
        frame: Frame,
        mask: np.ndarray,
        overlay_map: StructuralOverlayMap,
        label: str,
        anchor: Anchor2D,
        proposal: Proposal2D,
    ) -> int:
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return 0
        if self.pixel_sample_stride > 1:
            rows = rows[:: self.pixel_sample_stride]
            cols = cols[:: self.pixel_sample_stride]
        if self.max_pixels_per_pair > 0 and len(rows) > self.max_pixels_per_pair:
            indices = np.linspace(0, len(rows) - 1, num=self.max_pixels_per_pair, dtype=np.int64)
            rows = rows[indices]
            cols = cols[indices]

        z = frame.depth[rows, cols].astype(np.float64)
        x = (cols.astype(np.float64) - frame.intrinsics.cx) * z / frame.intrinsics.fx
        y = (rows.astype(np.float64) - frame.intrinsics.cy) * z / frame.intrinsics.fy
        points_cam = np.stack([x, y, z], axis=1)
        points = ((frame.pose[:3, :3] @ points_cam.T).T + frame.pose[:3, 3]).astype(np.float32)
        voxel_indices = np.floor(points / float(overlay_map.voxel_size)).astype(np.int32)
        if len(voxel_indices) == 0:
            return 0
        unique_voxels = np.unique(voxel_indices, axis=0)
        weight = max(float(anchor.confidence), 0.0) * max(float(proposal.confidence), 0.0)
        if weight <= 0.0:
            weight = 1.0
        for voxel in unique_voxels:
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            entry = overlay_map.voxels.setdefault(key, StructuralOverlayVoxel())
            entry.add_vote(label, weight, frame.frame_id)
        return int(len(rows))

    @staticmethod
    def _anchor_box_mask(bbox_xyxy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        left = max(0, min(width, int(np.floor(x1))))
        top = max(0, min(height, int(np.floor(y1))))
        right = max(0, min(width, int(np.ceil(x2))))
        bottom = max(0, min(height, int(np.ceil(y2))))
        mask = np.zeros((height, width), dtype=bool)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
        return mask
```

- [ ] **Step 4: Run module tests to verify they pass**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_structural_overlay.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/structural_overlay.py tests/test_structural_overlay.py
git commit -m "feat: build structural overlay votes"
```

## Task 3: Pipeline Wiring Without Association Leakage

**Files:**
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing pipeline wiring test**

Append this test inside `class TestPipeline` in `tests/test_pipeline.py`:

```python
    def test_structural_overlay_uses_precomputed_sam_without_changing_object_proposals(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: precomputed
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
structural_overlay:
  enabled: true
  classes: [wall]
  voxel_size: 0.01
  min_overlap_area: 1
  min_proposal_anchor_coverage: 0.01
  min_anchor_proposal_coverage: 0.01
depth_refinement:
  min_mask_area_after_refine: 1
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

        class AnchorPrimary:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            collect_generation_timings = False

            def generate_anchor_box_proposals(self, rgb):
                anchor = Anchor2D(4, np.array([1, 1, 7, 7], dtype=np.float32), "wall", 0.9)
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                proposal = Proposal2D(
                    proposal_id=4,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.9,
                    metadata={"anchor_id": 4, "anchor_class_name": "wall", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(4, 4, "wall", 0.9, keepalive=True)
                return [anchor], [proposal], [assignment]

        class PrecomputedProposal:
            active_backend_name = "precomputed"

            def __init__(self):
                self.calls = 0

            def process(self, rgb, depth, frame=None):
                self.calls += 1
                mask = np.zeros((8, 8), dtype=bool)
                mask[2:6, 2:6] = True
                return [
                    Proposal2D(
                        proposal_id=90,
                        mask=mask,
                        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="precomputed",
                    )
                ]

        precomputed = PrecomputedProposal()
        pipe.object_anchor = AnchorPrimary()
        pipe.proposal = precomputed

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=12)

        assert precomputed.calls == 1
        assert len(pipe.last_raw_proposals) == 1
        assert pipe.last_raw_proposals[0].metadata["anchor_class_name"] == "wall"
        assert len(pipe.last_source_proposals) == 1
        assert pipe.last_source_proposals[0].proposal_id == 90
        assert pipe.last_frame_debug["structural_overlay"]["accepted_pair_count"] == 1
        assert pipe.last_frame_debug["stage_timings"]["structural_overlay"] >= 0.0
        assert len(pipe.state.structural_overlay_map.voxels) > 0
```

- [ ] **Step 2: Run wiring test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_structural_overlay_uses_precomputed_sam_without_changing_object_proposals -q
```

Expected: FAIL because `Pipeline` does not instantiate or call `StructuralOverlayModule`.

- [ ] **Step 3: Import and instantiate overlay module**

In `src/pipelines/main_pipeline.py`, add the import:

```python
from src.modules.structural_overlay import StructuralOverlayModule
```

In `_TOP_LEVEL_STAGE_NAMES`, add:

```python
        "structural_overlay",
```

In `Pipeline.__init__`, after `self.async_refinement_backend = None`, add:

```python
        self.structural_overlay = StructuralOverlayModule(self.config.get("structural_overlay", {}))
```

- [ ] **Step 4: Add overlay proposal loader**

Add this method to `Pipeline` near `_async_refinement_fallback_allowed()`:

```python
    def _load_structural_overlay_proposals(
        self,
        frame: Frame,
        proposal_source: str,
        source_proposals: list[Proposal2D],
    ) -> tuple[list[Proposal2D], dict[str, Any]]:
        if not getattr(self.structural_overlay, "enabled", False):
            return [], {"proposal_source": "disabled", "skip_reason": "structural_overlay_disabled"}
        if source_proposals:
            return list(source_proposals), {"proposal_source": proposal_source, "skip_reason": ""}
        active_backend = str(getattr(self.proposal, "active_backend_name", "") or "")
        if proposal_source == "anchor_box_primary" and active_backend == "precomputed":
            proposals = list(self.proposal.process(frame.rgb, frame.depth, frame=frame))
            return proposals, {"proposal_source": "precomputed_overlay_only", "skip_reason": ""}
        return [], {
            "proposal_source": "unavailable",
            "active_backend": active_backend,
            "skip_reason": "structural_overlay_requires_precomputed_sam_proposals",
        }
```

- [ ] **Step 5: Add structural overlay stage after proposal generation**

In `process_frame()`, immediately after the proposal generation stage and before RuntimeVis, add:

```python
        structural_overlay_summary: dict[str, Any] = {"enabled": bool(self.structural_overlay.enabled)}
        with self._timed_stage("structural_overlay"):
            overlay_source_proposals, overlay_source_debug = self._load_structural_overlay_proposals(
                frame=frame,
                proposal_source=proposal_source,
                source_proposals=source_proposals,
            )
            if overlay_source_proposals:
                source_proposals = list(overlay_source_proposals)
                self.last_source_proposals = source_proposals
            structural_overlay_summary = self.structural_overlay.process(
                frame=frame,
                anchors=list(anchors),
                proposals=list(overlay_source_proposals),
                overlay_map=self.state.structural_overlay_map,
            )
            structural_overlay_summary.update(overlay_source_debug)
```

In `last_frame_debug`, add:

```python
            "structural_overlay": dict(structural_overlay_summary),
```

- [ ] **Step 6: Run wiring test to verify it passes**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_structural_overlay_uses_precomputed_sam_without_changing_object_proposals -q
```

Expected: PASS.

- [ ] **Step 7: Run focused pipeline regression tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_pipeline_skips_async_refinement_fallback_for_non_sam_backend tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled -q
```

Expected: PASS. If `test_pipeline_records_stage_timings_when_enabled` is not present in this branch, run only the async refinement fallback test and the new structural overlay test.

- [ ] **Step 8: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/pipelines/main_pipeline.py tests/test_pipeline.py
git commit -m "feat: wire structural overlay into fast pipeline"
```

## Task 4: Dense Projection Fusion Helpers

**Files:**
- Modify: `run_room0_full_eval.py`
- Modify: `tests/test_dual_map.py`

- [ ] **Step 1: Add failing tests for synthetic ids and conservative fusion**

Add these imports to the `from run_room0_full_eval import (...)` block in `tests/test_dual_map.py`:

```python
    apply_conservative_structural_overlay,
    colors_for_object_labels,
    structural_overlay_direct_label_maps,
    structural_overlay_object_id,
```

Append these tests to `tests/test_dual_map.py`:

```python
def test_structural_overlay_object_id_is_stable_negative_id() -> None:
    assert structural_overlay_object_id(17) == -10017
    assert structural_overlay_object_id(0) == -10000


def test_conservative_structural_overlay_does_not_overwrite_protected_object() -> None:
    sofa = ObjectMap(
        object_id=4,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("sofa", 0.9)]),
    )
    state = SystemState(objects={4: sofa})
    labels = np.array([4, -1], dtype=np.int32)
    state_ids = np.array([1, 0], dtype=np.uint8)
    supports = np.array([0.8, 0.0], dtype=np.float32)
    overlay_ids = np.array([-10003, -10003], dtype=np.int32)
    overlay_supports = np.array([2.0, 2.0], dtype=np.float32)

    fused_labels, fused_state_ids, fused_supports, summary = apply_conservative_structural_overlay(
        labels=labels,
        state_ids=state_ids,
        supports=supports,
        overlay_ids=overlay_ids,
        overlay_supports=overlay_supports,
        state=state,
        structure_labels={"wall"},
        protected_labels={"sofa"},
    )

    assert fused_labels.tolist() == [4, -10003]
    assert fused_state_ids.tolist() == [1, 0]
    assert fused_supports.tolist() == [0.8, 2.0]
    assert summary["overlay_candidate_point_count"] == 2
    assert summary["overlay_replaced_point_count"] == 1
    assert summary["overlay_protected_point_count"] == 1


def test_conservative_structural_overlay_may_replace_existing_structure_object() -> None:
    wall = ObjectMap(
        object_id=8,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("wall", 0.9)]),
    )
    state = SystemState(objects={8: wall})
    labels = np.array([8], dtype=np.int32)
    state_ids = np.array([1], dtype=np.uint8)
    supports = np.array([0.5], dtype=np.float32)
    overlay_ids = np.array([-10005], dtype=np.int32)
    overlay_supports = np.array([3.0], dtype=np.float32)

    fused_labels, _state_ids, fused_supports, summary = apply_conservative_structural_overlay(
        labels=labels,
        state_ids=state_ids,
        supports=supports,
        overlay_ids=overlay_ids,
        overlay_supports=overlay_supports,
        state=state,
        structure_labels={"wall", "window"},
        protected_labels={"chair", "sofa", "rug"},
    )

    assert fused_labels.tolist() == [-10005]
    assert fused_supports.tolist() == [3.0]
    assert summary["overlay_replaced_point_count"] == 1


def test_colors_for_object_labels_supports_direct_negative_structure_labels() -> None:
    object_ids = np.array([-10003, -1], dtype=np.int32)
    colors = colors_for_object_labels(
        object_ids,
        SystemState(),
        direct_object_label_names={-10003: "wall"},
    )

    assert tuple(int(value) for value in colors[0]) == semantic_surface_color("wall", -10003)
    assert tuple(int(value) for value in colors[1]) == (180, 180, 180)


def test_structural_overlay_direct_label_maps_use_class_names() -> None:
    direct_ids_to_class, direct_ids_to_name = structural_overlay_direct_label_maps(
        class_names={3: "wall", 5: "window", 9: "chair"},
        structure_labels={"wall", "window"},
    )

    assert direct_ids_to_class == {-10003: 3, -10005: 5}
    assert direct_ids_to_name == {-10003: "wall", -10005: "window"}
```

- [ ] **Step 2: Run fusion tests to verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_structural_overlay_object_id_is_stable_negative_id tests/test_dual_map.py::test_conservative_structural_overlay_does_not_overwrite_protected_object tests/test_dual_map.py::test_conservative_structural_overlay_may_replace_existing_structure_object tests/test_dual_map.py::test_colors_for_object_labels_supports_direct_negative_structure_labels tests/test_dual_map.py::test_structural_overlay_direct_label_maps_use_class_names -q
```

Expected: FAIL with import errors.

- [ ] **Step 3: Add synthetic id and fusion helpers**

In `run_room0_full_eval.py`, after `PLY_DTYPE`, add:

```python
STRUCTURAL_OVERLAY_ID_OFFSET = 10000


def structural_overlay_object_id(class_id: int) -> int:
    return -int(STRUCTURAL_OVERLAY_ID_OFFSET + int(class_id))


def structural_overlay_direct_label_maps(
    class_names: dict[int, str],
    structure_labels: set[str],
) -> tuple[dict[int, int], dict[int, str]]:
    normalized = {str(label).strip().lower() for label in structure_labels}
    direct_ids_to_class: dict[int, int] = {}
    direct_ids_to_name: dict[int, str] = {}
    for class_id, class_name in sorted(class_names.items()):
        label = str(class_name).strip().lower()
        if label not in normalized:
            continue
        object_id = structural_overlay_object_id(int(class_id))
        direct_ids_to_class[object_id] = int(class_id)
        direct_ids_to_name[object_id] = label
    return direct_ids_to_class, direct_ids_to_name
```

After `project_instances_to_dense_points()`, add:

```python
def apply_conservative_structural_overlay(
    labels: np.ndarray,
    state_ids: np.ndarray,
    supports: np.ndarray,
    overlay_ids: np.ndarray,
    overlay_supports: np.ndarray,
    state: SystemState,
    structure_labels: set[str],
    protected_labels: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    fused_labels = np.asarray(labels, dtype=np.int32).copy()
    fused_state_ids = np.asarray(state_ids, dtype=np.uint8).copy()
    fused_supports = np.asarray(supports, dtype=np.float32).copy()
    overlay_ids = np.asarray(overlay_ids, dtype=np.int32)
    overlay_supports = np.asarray(overlay_supports, dtype=np.float32)
    normalized_structure = {str(label).strip().lower() for label in structure_labels}
    normalized_protected = {str(label).strip().lower() for label in protected_labels}
    summary = {
        "overlay_candidate_point_count": int(np.count_nonzero(overlay_ids < 0)),
        "overlay_replaced_point_count": 0,
        "overlay_protected_point_count": 0,
    }
    for index, overlay_id in enumerate(overlay_ids):
        if int(overlay_id) >= 0:
            continue
        current_id = int(fused_labels[index])
        can_replace = current_id < 0
        if current_id >= 0:
            obj = state.objects.get(current_id)
            current_label = object_semantic_label(obj).strip().lower() if obj is not None else ""
            if current_label in normalized_protected:
                summary["overlay_protected_point_count"] += 1
                continue
            can_replace = (not current_label) or current_label in normalized_structure
        if not can_replace:
            summary["overlay_protected_point_count"] += 1
            continue
        fused_labels[index] = int(overlay_id)
        fused_state_ids[index] = 0
        fused_supports[index] = float(overlay_supports[index])
        summary["overlay_replaced_point_count"] += 1
    return fused_labels, fused_state_ids, fused_supports, summary
```

- [ ] **Step 4: Extend colors for negative direct ids**

Change the signature of `colors_for_object_labels()` in `run_room0_full_eval.py` to:

```python
def colors_for_object_labels(
    object_ids: np.ndarray,
    state: SystemState,
    default_color: tuple[int, int, int] = (180, 180, 180),
    direct_object_label_names: dict[int, str] | None = None,
) -> np.ndarray:
```

Inside the function, before the loop over positive object ids, add:

```python
    direct_object_label_names = direct_object_label_names or {}
    for object_id, semantic_label in direct_object_label_names.items():
        color = semantic_surface_color(str(semantic_label), int(object_id))
        colors[object_ids == int(object_id)] = np.asarray(color, dtype=np.uint8)
```

- [ ] **Step 5: Run fusion tests to verify they pass**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_structural_overlay_object_id_is_stable_negative_id tests/test_dual_map.py::test_conservative_structural_overlay_does_not_overwrite_protected_object tests/test_dual_map.py::test_conservative_structural_overlay_may_replace_existing_structure_object tests/test_dual_map.py::test_colors_for_object_labels_supports_direct_negative_structure_labels tests/test_dual_map.py::test_structural_overlay_direct_label_maps_use_class_names -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add run_room0_full_eval.py tests/test_dual_map.py
git commit -m "feat: add conservative structural overlay fusion"
```

## Task 5: Overlay Projection and Direct Evaluation

**Files:**
- Modify: `run_room0_full_eval.py`
- Modify: `tests/test_dual_map.py`

- [ ] **Step 1: Add failing tests for overlay projection and direct evaluation**

Add these imports to the `from run_room0_full_eval import (...)` block in `tests/test_dual_map.py`:

```python
    evaluate_semantics,
    project_structural_overlay_to_dense_points,
```

Append these tests to `tests/test_dual_map.py`:

```python
def test_project_structural_overlay_to_dense_points_returns_synthetic_ids() -> None:
    state = SystemState()
    state.structural_overlay_map.voxel_size = 0.1
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("wall", 2.5, frame_id=1)
    state.structural_overlay_map.voxels[(0, 0, 10)] = voxel
    class_names = {3: "wall"}
    direct_ids_to_class, _direct_ids_to_name = structural_overlay_direct_label_maps(class_names, {"wall"})

    overlay_ids, overlay_supports = project_structural_overlay_to_dense_points(
        dense_points=np.array([[0.02, 0.02, 1.02], [5.0, 5.0, 5.0]], dtype=np.float32),
        structural_overlay_map=state.structural_overlay_map,
        direct_ids_to_class=direct_ids_to_class,
        class_names=class_names,
        neighbor_radius=0,
    )

    assert overlay_ids.tolist() == [-10003, -1]
    assert overlay_supports.tolist() == [2.5, 0.0]


def test_evaluate_semantics_uses_direct_labels_for_negative_structure_ids() -> None:
    gt_vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    gt_labels = np.array([3, 4], dtype=np.int32)
    dense_points = gt_vertices.copy()
    object_ids = np.array([-10003, 7], dtype=np.int32)

    evaluation = evaluate_semantics(
        gt_vertices=gt_vertices,
        gt_labels=gt_labels,
        dense_points=dense_points,
        object_ids=object_ids,
        class_names={3: "wall", 4: "chair"},
        direct_object_label_ids={-10003: 3},
    )

    assert evaluation["pred_labels"].tolist() == [3, 4]
    assert evaluation["object_to_class"][-10003] == 3
    assert evaluation["object_to_class"][7] == 4
```

- [ ] **Step 2: Run projection/evaluation tests to verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_project_structural_overlay_to_dense_points_returns_synthetic_ids tests/test_dual_map.py::test_evaluate_semantics_uses_direct_labels_for_negative_structure_ids -q
```

Expected: FAIL because projection helper and direct evaluation argument do not exist.

- [ ] **Step 3: Implement overlay projection**

In `run_room0_full_eval.py`, after `project_instances_to_dense_points()`, add:

```python
def project_structural_overlay_to_dense_points(
    dense_points: np.ndarray,
    structural_overlay_map,
    direct_ids_to_class: dict[int, int],
    class_names: dict[int, str],
    neighbor_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    overlay_ids = np.full(len(dense_points), -1, dtype=np.int32)
    overlay_supports = np.zeros(len(dense_points), dtype=np.float32)
    if len(dense_points) == 0 or not getattr(structural_overlay_map, "voxels", None):
        return overlay_ids, overlay_supports

    label_to_direct_id = {
        str(class_names[class_id]).strip().lower(): int(object_id)
        for object_id, class_id in direct_ids_to_class.items()
        if int(class_id) in class_names
    }
    dense_indices = np.floor(dense_points / float(structural_overlay_map.voxel_size)).astype(np.int32)
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in range(-int(neighbor_radius), int(neighbor_radius) + 1)
        for dy in range(-int(neighbor_radius), int(neighbor_radius) + 1)
        for dz in range(-int(neighbor_radius), int(neighbor_radius) + 1)
    ]
    for index, voxel in enumerate(dense_indices):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        entry = structural_overlay_map.voxels.get(key)
        if entry is None and neighbor_radius > 0:
            best_entry = None
            best_support = -1.0
            for dx, dy, dz in neighbor_offsets:
                candidate = structural_overlay_map.voxels.get((key[0] + dx, key[1] + dy, key[2] + dz))
                if candidate is None:
                    continue
                if candidate.top_support > best_support:
                    best_support = candidate.top_support
                    best_entry = candidate
            entry = best_entry
        if entry is None:
            continue
        direct_id = label_to_direct_id.get(str(entry.top_label).strip().lower())
        if direct_id is None:
            continue
        overlay_ids[index] = int(direct_id)
        overlay_supports[index] = float(entry.top_support)
    return overlay_ids, overlay_supports
```

- [ ] **Step 4: Add direct id support to evaluator**

Change `evaluate_semantics()` signature to:

```python
def evaluate_semantics(
    gt_vertices: np.ndarray,
    gt_labels: np.ndarray,
    dense_points: np.ndarray,
    object_ids: np.ndarray,
    class_names: dict[int, str],
    direct_object_label_ids: dict[int, int] | None = None,
) -> dict[str, Any]:
```

Inside the function, after `nearest_object_ids = object_ids[nn_indices]`, add:

```python
    direct_object_label_ids = {int(key): int(value) for key, value in (direct_object_label_ids or {}).items()}
```

Replace the vote loop with:

```python
    per_object_votes: dict[int, Counter] = defaultdict(Counter)
    valid_gt_mask = gt_labels >= 0
    for object_id, gt_label in zip(nearest_object_ids[valid_gt_mask], gt_labels[valid_gt_mask]):
        object_id = int(object_id)
        if object_id < 0 and object_id not in direct_object_label_ids:
            continue
        if object_id in direct_object_label_ids:
            continue
        per_object_votes[object_id][int(gt_label)] += 1
```

After creating `object_to_class`, add:

```python
    object_to_class.update(direct_object_label_ids)
```

Replace the prediction loop condition with:

```python
        object_id = int(object_id)
        if object_id < 0 and object_id not in direct_object_label_ids:
            continue
        pred_labels[index] = object_to_class.get(object_id, -1)
```

- [ ] **Step 5: Run projection/evaluation tests to verify they pass**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_project_structural_overlay_to_dense_points_returns_synthetic_ids tests/test_dual_map.py::test_evaluate_semantics_uses_direct_labels_for_negative_structure_ids -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add run_room0_full_eval.py tests/test_dual_map.py
git commit -m "feat: evaluate structural overlay direct labels"
```

## Task 6: Export Integration and Reporting

**Files:**
- Modify: `run_room0_full_eval.py`
- Modify: `tests/test_dual_map.py`

- [ ] **Step 1: Add failing export/report tests**

Add this import to the `from run_room0_full_eval import (...)` block in `tests/test_dual_map.py`:

```python
    build_structural_overlay_records,
```

Append this test to `tests/test_dual_map.py`:

```python
def test_build_structural_overlay_records_exports_negative_ids_with_semantic_colors() -> None:
    state = SystemState()
    state.structural_overlay_map.voxel_size = 0.1
    voxel = StructuralOverlayVoxel()
    voxel.add_vote("window", 1.25, frame_id=2)
    state.structural_overlay_map.voxels[(1, 2, 3)] = voxel
    direct_ids_to_class, direct_ids_to_name = structural_overlay_direct_label_maps(
        {5: "window"},
        {"window"},
    )

    records = build_structural_overlay_records(
        state.structural_overlay_map,
        direct_ids_to_class=direct_ids_to_class,
        direct_ids_to_name=direct_ids_to_name,
    )

    assert len(records) == 1
    assert int(records[0]["object_id"]) == -10005
    assert float(records[0]["support"]) == 1.25
    assert tuple(int(value) for value in records[0][["red", "green", "blue"]]) == semantic_surface_color("window", -10005)
    assert np.allclose([records[0]["x"], records[0]["y"], records[0]["z"]], [0.15, 0.25, 0.35])
```

- [ ] **Step 2: Run export test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_build_structural_overlay_records_exports_negative_ids_with_semantic_colors -q
```

Expected: FAIL because `build_structural_overlay_records` does not exist.

- [ ] **Step 3: Add overlay PLY record builder**

In `run_room0_full_eval.py`, after `build_dense_surface_records()`, add:

```python
def build_structural_overlay_records(
    structural_overlay_map,
    direct_ids_to_class: dict[int, int],
    direct_ids_to_name: dict[int, str],
) -> np.ndarray:
    rows = []
    label_to_direct_id = {str(label).strip().lower(): int(object_id) for object_id, label in direct_ids_to_name.items()}
    voxel_size = float(structural_overlay_map.voxel_size)
    for voxel_key, voxel in sorted(structural_overlay_map.voxels.items()):
        label = str(voxel.top_label).strip().lower()
        object_id = label_to_direct_id.get(label)
        if object_id is None or object_id not in direct_ids_to_class:
            continue
        center = (np.asarray(voxel_key, dtype=np.float32) + 0.5) * voxel_size
        color = semantic_surface_color(label, int(object_id))
        rows.append(
            (
                float(center[0]),
                float(center[1]),
                float(center[2]),
                color[0],
                color[1],
                color[2],
                int(object_id),
                0,
                float(voxel.top_support),
            )
        )
    return np.asarray(rows, dtype=PLY_DTYPE)
```

- [ ] **Step 4: Integrate overlay into `main()` export path**

In `main()`, move these existing lines so they run immediately after `dense_points, dense_rgb = finalize_geometry_accum(geometry_accum)` and before `projected_colors` or any dense instance PLY is written:

```python
    gt_vertices = read_gt_vertices(args.gt_mesh_ply)
    gt_labels = read_gt_labels(args.gt_labels)
    class_names = read_class_names(args.gt_info_json)
```

Remove the later duplicate copy of those three lines that currently appears immediately before `evaluation = evaluate_semantics(...)`.

After the `labels, state_ids, supports = project_instances_to_dense_points(...)` call and before `projected_colors` is computed, add:

```python
    structural_cfg = dict(pipeline.config.get("structural_overlay", {}) or {})
    structure_labels = {
        str(label).strip().lower()
        for label in structural_cfg.get("classes", [])
        if str(label).strip()
    }
    protected_labels = {
        str(label).strip().lower()
        for label in structural_cfg.get("protected_labels", [])
        if str(label).strip()
    }
    direct_object_label_ids, direct_object_label_names = structural_overlay_direct_label_maps(
        class_names=class_names,
        structure_labels=structure_labels,
    )
    overlay_records = build_structural_overlay_records(
        state.structural_overlay_map,
        direct_ids_to_class=direct_object_label_ids,
        direct_ids_to_name=direct_object_label_names,
    )
    overlay_path = exports_dir / "room0_structural_overlay.ply"
    write_binary_ply(overlay_path, overlay_records)
    overlay_ids, overlay_supports = project_structural_overlay_to_dense_points(
        dense_points=dense_points,
        structural_overlay_map=state.structural_overlay_map,
        direct_ids_to_class=direct_object_label_ids,
        class_names=class_names,
        neighbor_radius=max(0, int(structural_cfg.get("projection_neighbor_radius", 0))),
    )
    if bool(structural_cfg.get("conservative_fusion", False)):
        labels, state_ids, supports, structural_overlay_fusion = apply_conservative_structural_overlay(
            labels=labels,
            state_ids=state_ids,
            supports=supports,
            overlay_ids=overlay_ids,
            overlay_supports=overlay_supports,
            state=state,
            structure_labels=structure_labels,
            protected_labels=protected_labels,
        )
    else:
        structural_overlay_fusion = {
            "overlay_candidate_point_count": int(np.count_nonzero(overlay_ids < 0)),
            "overlay_replaced_point_count": 0,
            "overlay_protected_point_count": 0,
        }
```

Move the existing `projected_colors = colors_for_object_labels(labels, state)` line to after fusion and change it to:

```python
    projected_colors = colors_for_object_labels(
        labels,
        state,
        direct_object_label_names=direct_object_label_names,
    )
```

Change the `evaluate_semantics()` call to pass:

```python
        direct_object_label_ids=direct_object_label_ids,
```

- [ ] **Step 5: Report overlay totals**

Change `write_run_report()` signature to include:

```python
    structural_overlay_fusion: dict[str, int] | None = None,
```

Inside `write_run_report()`, after `stage_timing_summary`, add:

```python
    structural_overlay_fusion = structural_overlay_fusion or {}
    structural_overlay_frame_totals = Counter()
    for metrics in frame_metrics:
        overlay_metrics = dict(metrics.get("structural_overlay", {}) or {})
        for key in (
            "structure_anchor_count",
            "sam_proposal_count",
            "accepted_pair_count",
            "mismatched_mask_count",
            "voted_pixel_count",
            "new_voxel_count",
        ):
            structural_overlay_frame_totals[key] += int(overlay_metrics.get(key, 0))
```

In the metadata lines, after `projected_dense_object_count`, add:

```python
        f"- `structural_overlay_voxel_count`: `{len(state.structural_overlay_map.voxels)}`",
        f"- `structural_overlay_accepted_pair_count_total`: `{int(structural_overlay_frame_totals['accepted_pair_count'])}`",
        f"- `structural_overlay_voted_pixel_count_total`: `{int(structural_overlay_frame_totals['voted_pixel_count'])}`",
        f"- `structural_overlay_replaced_point_count`: `{int(structural_overlay_fusion.get('overlay_replaced_point_count', 0))}`",
        f"- `structural_overlay_protected_point_count`: `{int(structural_overlay_fusion.get('overlay_protected_point_count', 0))}`",
```

In `export_lines`, add:

```python
        f"- `room0_structural_overlay.ply` (synthetic negative-id structure overlay): `{scene_dir / 'exports' / 'room0_structural_overlay.ply'}`",
```

In the JSON sidecar, add:

```python
        "structural_overlay_voxel_count": int(len(state.structural_overlay_map.voxels)),
        "structural_overlay_frame_totals": {
            str(key): int(value) for key, value in structural_overlay_frame_totals.items()
        },
        "structural_overlay_fusion": {
            str(key): int(value) for key, value in structural_overlay_fusion.items()
        },
```

In the `write_run_report()` call in `main()`, add:

```python
        structural_overlay_fusion=structural_overlay_fusion,
```

- [ ] **Step 6: Copy structural overlay metrics into frame metrics**

In `main()`, inside the per-frame `record = { ... }` block, add:

```python
                    "structural_overlay": dict(pipeline.last_frame_debug.get("structural_overlay", {}) or {}),
```

- [ ] **Step 7: Run export/report tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_build_structural_overlay_records_exports_negative_ids_with_semantic_colors tests/test_dual_map.py::test_write_run_report_omits_fast_eval_debug_artifacts -q
```

Expected: PASS. If the report test needs new required argument defaults, fix `write_run_report()` defaults so existing callers still pass.

- [ ] **Step 8: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add run_room0_full_eval.py tests/test_dual_map.py
git commit -m "feat: export structural overlay projection"
```

## Task 7: Room0 Fast Structural Overlay Config

**Files:**
- Create: `configs/room0_fast_structural_overlay_4090.yaml`
- Modify: `tests/test_dual_map.py`

- [ ] **Step 1: Add failing config test**

Append this test to `tests/test_dual_map.py`:

```python
def test_room0_fast_structural_overlay_config_enables_conservative_overlay() -> None:
    config_path = Path("configs/room0_fast_structural_overlay_4090.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "precomputed"
    assert config["anchor_frontend"]["anchor_primary_mode"] is True
    assert config["anchor_frontend"]["use_sam_intersection_proposals"] is False
    assert config["structural_overlay"]["enabled"] is True
    assert config["structural_overlay"]["conservative_fusion"] is True
    assert config["structural_overlay"]["classes"] == ["wall", "window", "blinds", "ceiling", "floor", "door"]
    assert "sofa" in config["structural_overlay"]["protected_labels"]
    assert "rug" in config["structural_overlay"]["protected_labels"]
    assert "chair" in config["structural_overlay"]["protected_labels"]
    assert "table" in config["structural_overlay"]["protected_labels"]
```

- [ ] **Step 2: Run config test to verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_room0_fast_structural_overlay_config_enables_conservative_overlay -q
```

Expected: FAIL because the config file does not exist.

- [ ] **Step 3: Create config from fast-hybrid baseline**

Copy `configs/room0_fast_hybrid_4090.yaml` to `configs/room0_fast_structural_overlay_4090.yaml`, then append this top-level block:

```yaml
structural_overlay:
  enabled: true
  classes: [wall, window, blinds, ceiling, floor, door]
  voxel_size: 0.05
  min_overlap_area: 25
  min_proposal_anchor_coverage: 0.20
  min_anchor_proposal_coverage: 0.03
  clip_to_anchor_box: true
  conservative_fusion: true
  projection_neighbor_radius: 0
  pixel_sample_stride: 2
  max_pixels_per_pair: 4096
  protected_labels:
  - basket
  - blanket
  - book
  - cabinet
  - candle
  - chair
  - cushion
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
  - wall-plug
  - rug
```

Keep these existing fast-hybrid settings unchanged:

```yaml
proposal:
  backend: precomputed
anchor_frontend:
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
pipeline:
  collect_stage_timings: true
```

- [ ] **Step 4: Run config test to verify it passes**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_room0_fast_structural_overlay_config_enables_conservative_overlay -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add configs/room0_fast_structural_overlay_4090.yaml tests/test_dual_map.py
git commit -m "config: add room0 fast structural overlay"
```

## Task 8: Focused Regression Suite

**Files:**
- No source edits expected.

- [ ] **Step 1: Run focused tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_structural_overlay.py tests/test_dual_map.py tests/test_pipeline.py::TestPipeline::test_structural_overlay_uses_precomputed_sam_without_changing_object_proposals tests/test_object_anchor.py -q
```

Expected: PASS.

- [ ] **Step 2: Run previously stable core tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_async_refinement.py tests/test_object_anchor.py tests/test_pipeline.py tests/test_dual_map.py -q
```

Expected: PASS. Previous baseline was 178 passing tests; the exact count may increase because this plan adds new tests.

- [ ] **Step 3: Commit verification note if any test fixtures needed adjustment**

If tests required a small fixture-only adjustment, commit it:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add tests src configs run_room0_full_eval.py
git commit -m "test: cover structural overlay integration"
```

If no files changed after Step 2, do not create an empty commit.

## Task 9: Room0 Smoke and 200f Experiment

**Files:**
- No source edits expected unless the smoke reveals a bug.

- [ ] **Step 1: Run 20f stride=10 smoke**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_fast_structural_overlay_4090.yaml \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --output-root outputs/tmp_validation \
  --experiment-name 20260530_room0_structure_dense_overlay_stride10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected:

- Process exits with status 0.
- `room0/run_report.json` exists.
- `room0/exports/room0_structural_overlay.ply` exists.
- `stage_timing_summary.structural_overlay.total_sec` is present.
- `structural_overlay_frame_totals.accepted_pair_count` is greater than 0.

- [ ] **Step 2: Inspect smoke metrics**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("outputs/tmp_validation/20260530_room0_structure_dense_overlay_stride10_20f/room0/run_report.json")
report = json.loads(path.read_text())
print("mIoU", report["miou"])
print("window", {item["class_name"]: item["iou"] for item in report["per_class"]}.get("window"))
print("wall", {item["class_name"]: item["iou"] for item in report["per_class"]}.get("wall"))
print("blinds", {item["class_name"]: item["iou"] for item in report["per_class"]}.get("blinds"))
print("overlay", report["structural_overlay_frame_totals"])
print("fusion", report["structural_overlay_fusion"])
print("timing", report["stage_timing_summary"].get("structural_overlay", {}))
PY
```

Expected:

- `window` appears with a numeric IoU.
- Overlay accepted pairs are nonzero.
- Overlay replaced point count is nonzero unless no structural class is visible in the 20 sampled frames.

- [ ] **Step 3: Run 200f stride=10 experiment in tmux**

Run:

```bash
tmux new-session -d -s room0_structure_overlay_s10_200f "/bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && experiment=20260530_room0_structure_dense_overlay_stride10_200f && log=outputs/tmp_validation/${experiment}.log && status=outputs/tmp_validation/${experiment}.status && { echo experiment=${experiment}; echo started_at=$(date --iso-8601=seconds); /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_fast_structural_overlay_4090.yaml --dataset-root /home/ww/vv/dataset/Replica/room0 --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json --output-root outputs/tmp_validation --experiment-name ${experiment} --num-frames 200 --frame-stride 10 --proposal-backend precomputed --proposal-device cuda --fast-eval; rc=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=${rc}; echo exit_status=${rc} > ${status}; } > ${log} 2>&1'"
```

Expected: tmux session starts and writes `outputs/tmp_validation/20260530_room0_structure_dense_overlay_stride10_200f.log`.

- [ ] **Step 4: Monitor 200f experiment**

Run:

```bash
tmux capture-pane -pt room0_structure_overlay_s10_200f -S -80
```

Expected while running: frame progress lines increase. Expected after completion: `exit_status=0` in the log and status file.

- [ ] **Step 5: Compare against fast baselines**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
runs = [
    "outputs/tmp_validation/20260529_room0_fast_hybrid_worker_depth_stride10_200f/room0/run_report.json",
    "outputs/tmp_validation/20260529_room0_coarse_to_fine_async_delayguard_stride10_200f/room0/run_report.json",
    "outputs/tmp_validation/20260530_room0_structure_dense_overlay_stride10_200f/room0/run_report.json",
]
classes = ["wall", "window", "blinds", "chair", "table", "sofa", "rug"]
for run in runs:
    path = Path(run)
    report = json.loads(path.read_text())
    per = {item["class_name"]: item["iou"] for item in report["per_class"]}
    print(path.parts[-3], "mIoU", round(report["miou"], 4))
    for name in classes:
        print(" ", name, None if name not in per else round(per[name], 4))
    print(" overlay", report.get("structural_overlay_fusion", {}))
PY
```

Expected acceptance bar:

- `window` IoU is greater than `0.0`.
- `wall` IoU is greater than `0.347`.
- `blinds` IoU is greater than `0.384`.
- `chair`, `table`, `sofa`, and `rug` do not drop more than `0.05` absolute IoU from `20260529_room0_fast_hybrid_worker_depth_stride10_200f`.

- [ ] **Step 6: Commit experiment notes if source is unchanged**

Create or update `docs/superpowers/experiments/2026-05-30-structure-dense-overlay-fast.md` with the compared metrics and runtime, then commit:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add docs/superpowers/experiments/2026-05-30-structure-dense-overlay-fast.md
git commit -m "docs: record structural overlay experiment"
```

If smoke or 200f exposes a bug, fix the smallest failing component with a focused test before rerunning the experiment.

## Self-Review

- Spec coverage: Tasks 1-3 implement separate overlay state, SAM-proposal/anchor evidence collection, precomputed-only fast pipeline wiring, and no association/object-update leakage. Tasks 4-6 implement conservative dense fusion, synthetic negative ids, direct-label evaluation, and reporting/export. Tasks 7-9 implement config and experiment verification.
- Placeholder scan: The plan contains no unspecified implementation slots. Every code-changing task includes concrete tests, concrete code snippets, commands, expected outcomes, and commit messages.
- Type consistency: `StructuralOverlayMap`, `StructuralOverlayVoxel`, `structural_overlay_object_id()`, `structural_overlay_direct_label_maps()`, `project_structural_overlay_to_dense_points()`, `apply_conservative_structural_overlay()`, and `direct_object_label_ids` use the same names across tests and implementation steps.
