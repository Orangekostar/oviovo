# Coarse-To-Fine Async Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine the 2026-05-26 SAM/anchor accuracy path with the current fast anchor-primary path by letting anchor/depth/voxel mapping run first and applying delayed SAM2 mask refinement to replace coarse observations.

**Architecture:** Keep the current fast hybrid path as the online/coarse layer: YOLOWorld/YOLOE anchors produce fast rectangular proposals, depth refinement and voxel ownership create a usable map immediately. Add an opt-in refinement layer that records coarse observations, runs SAM/precomputed-SAM proposal matching after the coarse frame update, and replaces matching coarse local-memory observations with fine SAM/anchor-union patches. First implementation is offline-deterministic and uses the existing precomputed SAM cache; the same queue interface can later be backed by a true asynchronous SAM2 worker.

**Tech Stack:** Python 3, NumPy, pytest, YAML configs, existing OVIOVO modules (`Pipeline`, `ObjectAnchorModule`, `DepthRefinementModule`, `PatchLiftingModule`, `AssociationModule`, `ObjectUpdateModule`), existing precomputed SAM2 proposal backend.

---

## File Structure

- Create `src/modules/async_refinement.py`
  - Owns refinement job records, anchor-to-SAM matching, delayed fine proposal generation, and replacement patch preparation.
  - Depends only on existing modules and core dataclasses.
- Modify `src/core/data_structures.py`
  - Add lightweight observation provenance fields to `ObservationRecord` so coarse observations can be replaced safely.
- Modify `src/modules/object_update.py`
  - Stamp observation provenance on newly created/updated observations.
  - Add an opt-in `replace_observations()` helper that removes coarse local-memory points/evidence for a frame+anchor and inserts fine patches.
  - Rebuild anchor semantic votes after replacement using the existing `rebuild_anchor_semantic_votes_from_observations()`.
- Modify `src/modules/tsdf_instance_map.py`
  - Add `remove_patch_support()` for optional future support rollback.
  - First pipeline integration should leave TSDF rollback disabled by default and record that in debug metadata.
- Modify `src/pipelines/main_pipeline.py`
  - Instantiate `AsyncRefinementModule`.
  - After the coarse `object_update`, request and apply ready refinement jobs.
  - Expose refinement timing and counts in `last_frame_debug`.
- Modify `run_room0_full_eval.py`
  - Add refinement fields to `frame_metrics.jsonl`.
  - Add aggregate refinement summary to `run_report.md` and `run_report.json`.
- Create `configs/room0_coarse_to_fine_async_4090.yaml`
  - Clone fast-hybrid speed controls, but enable delayed SAM refinement from the precomputed cache.
- Modify `tests/test_pipeline.py`
  - Add replacement and pipeline integration tests.
- Create `tests/test_async_refinement.py`
  - Add focused tests for job scheduling, SAM/anchor matching, and fine proposal provenance.
- Modify `tests/test_dual_map.py`
  - Add report-summary tests for refinement fields.

## Task 1: Add Observation Provenance

**Files:**
- Modify: `src/core/data_structures.py`
- Modify: `src/modules/object_update.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing provenance test**

Append this test inside `class TestObjectUpdateModule` in `tests/test_pipeline.py`:

```python
    def test_object_update_records_observation_provenance(self):
        module = ObjectUpdateModule(
            {
                "surface_owner_gate": {"enabled": False},
                "provisional": {"enabled": False},
            }
        )
        state = SystemState()
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        patch = Patch3D(
            patch_id=7,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=12,
            metadata={
                "source_proposal_id": 7,
                "source_backend_name": "anchor_box_primary",
                "anchor_id": 3,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.91,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "12:3",
            },
        )
        association = AssociationResult(new_object_patches=[7])

        state = module.process(association, [patch], state)

        obs = state.objects[0].observations[0]
        assert obs.source_frame_id == 12
        assert obs.source_proposal_id == 7
        assert obs.anchor_id == 3
        assert obs.observation_layer == "coarse"
        assert obs.refinement_key == "12:3"
        assert obs.replaced_by_refinement is False
```

- [ ] **Step 2: Run the provenance test and verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestObjectUpdateModule::test_object_update_records_observation_provenance -q
```

Expected: FAIL with `AttributeError: 'ObservationRecord' object has no attribute 'source_frame_id'`.

- [ ] **Step 3: Add provenance fields**

In `src/core/data_structures.py`, replace `ObservationRecord` with this definition:

```python
@dataclass
class ObservationRecord:
    """Record of a single observation of an object.

    Attributes:
        frame_id: Frame in which the observation was made.
        patch: The 3D patch observed.
        crop_bbox: 2D bounding box of the crop used for semantic encoding.
        timestamp: Observation time.
        source_frame_id: Original frame id from the observation patch.
        source_proposal_id: Proposal id that produced this observation.
        anchor_id: Detector anchor id used for semantic evidence, or -1.
        observation_layer: "coarse" for fast anchor-primary observations, "fine" for SAM-refined observations.
        refinement_key: Stable key used to replace one coarse observation with a later fine observation.
        replaced_by_refinement: True when this record was superseded by a finer observation.
    """
    frame_id: int
    patch: Patch3D
    crop_bbox: Optional[np.ndarray] = None
    timestamp: float = 0.0
    source_frame_id: int = 0
    source_proposal_id: int = -1
    anchor_id: int = -1
    observation_layer: str = ""
    refinement_key: str = ""
    replaced_by_refinement: bool = False
```

- [ ] **Step 4: Add an observation factory helper**

In `src/modules/object_update.py`, add this private method before `_update_object()`:

```python
    def _observation_from_patch(self, patch: Patch3D) -> ObservationRecord:
        return ObservationRecord(
            frame_id=patch.source_frame_id,
            patch=patch,
            crop_bbox=self._patch_crop_bbox(patch),
            timestamp=patch.timestamp,
            source_frame_id=int(patch.source_frame_id),
            source_proposal_id=int(patch.metadata.get("source_proposal_id", patch.patch_id)),
            anchor_id=int(patch.metadata.get("anchor_id", -1)),
            observation_layer=str(patch.metadata.get("observation_layer", "")),
            refinement_key=str(patch.metadata.get("refinement_key", "")),
            replaced_by_refinement=False,
        )
```

Replace every direct construction of `ObservationRecord(...)` in `src/modules/object_update.py` with:

```python
self._observation_from_patch(patch)
```

The three locations are `_upsert_provisional_object()`, `_update_object()`, and `_create_object()`.

- [ ] **Step 5: Run the provenance test and full object-update subset**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestObjectUpdateModule::test_object_update_records_observation_provenance tests/test_pipeline.py::TestObjectUpdateModule -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/core/data_structures.py src/modules/object_update.py tests/test_pipeline.py
git commit -m "feat: track observation refinement provenance"
```

## Task 2: Add Refinement Job Matching

**Files:**
- Create: `src/modules/async_refinement.py`
- Test: `tests/test_async_refinement.py`

- [ ] **Step 1: Write failing job-matching tests**

Create `tests/test_async_refinement.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Anchor2D, CameraIntrinsics, Frame, Proposal2D
from src.modules.async_refinement import AsyncRefinementModule


def _frame() -> Frame:
    return Frame(
        frame_id=4,
        source_frame_id=40,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.ones((10, 10), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=10, height=10),
    )


def _proposal(proposal_id: int, y1: int, y2: int, x1: int, x2: int) -> Proposal2D:
    mask = np.zeros((10, 10), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return Proposal2D(
        proposal_id=proposal_id,
        mask=mask,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.8,
        backend_name="sam2",
        metadata={"source": "sam2"},
    )


def test_async_refinement_selects_sam_masks_by_anchor_overlap() -> None:
    module = AsyncRefinementModule(
        {
            "enabled": True,
            "min_proposal_anchor_coverage": 0.20,
            "min_anchor_proposal_coverage": 0.20,
            "proposal_min_area": 1,
        }
    )
    anchor = Anchor2D(
        anchor_id=2,
        bbox_xyxy=np.array([2, 2, 8, 8], dtype=np.float32),
        class_name="sofa",
        confidence=0.9,
    )
    fine, debug = module.build_fine_proposals(
        frame=_frame(),
        anchors=[anchor],
        sam_proposals=[
            _proposal(10, 2, 8, 2, 8),
            _proposal(11, 0, 2, 0, 2),
        ],
    )

    assert len(fine) == 1
    assert fine[0].metadata["observation_layer"] == "fine"
    assert fine[0].metadata["refinement_key"] == "40:2"
    assert fine[0].metadata["anchor_class_name"] == "sofa"
    assert fine[0].metadata["source_raw_proposal_ids"] == [10]
    assert debug["matched_anchor_count"] == 1
    assert debug["fine_proposal_count"] == 1


def test_async_refinement_preserves_unanchored_sam_masks_as_context_only() -> None:
    module = AsyncRefinementModule({"enabled": True, "proposal_min_area": 1})
    fine, debug = module.build_fine_proposals(
        frame=_frame(),
        anchors=[],
        sam_proposals=[_proposal(10, 2, 8, 2, 8)],
    )

    assert fine == []
    assert debug["unmatched_sam_proposal_count"] == 1
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_async_refinement.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.modules.async_refinement'`.

- [ ] **Step 3: Implement `AsyncRefinementModule`**

Create `src/modules/async_refinement.py` with:

```python
"""Delayed SAM refinement for coarse-to-fine mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.data_structures import Anchor2D, Frame, Proposal2D


@dataclass(frozen=True)
class RefinementResult:
    fine_proposals: list[Proposal2D]
    debug: dict[str, Any]


class AsyncRefinementModule:
    """Build fine SAM/anchor proposals that can replace coarse anchor-box observations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.delay_frames = int(self.config.get("delay_frames", 0))
        self.proposal_min_area = int(self.config.get("proposal_min_area", 25))
        self.min_proposal_anchor_coverage = float(
            np.clip(self.config.get("min_proposal_anchor_coverage", 0.20), 0.0, 1.0)
        )
        self.min_anchor_proposal_coverage = float(
            np.clip(self.config.get("min_anchor_proposal_coverage", 0.05), 0.0, 1.0)
        )
        self.clip_to_anchor_box = bool(self.config.get("clip_to_anchor_box", False))
        self.last_debug: dict[str, Any] = {}

    def build_fine_proposals(
        self,
        *,
        frame: Frame,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
    ) -> tuple[list[Proposal2D], dict[str, Any]]:
        fine: list[Proposal2D] = []
        matched_sam_ids: set[int] = set()
        matched_anchor_count = 0

        for anchor in anchors:
            selected: list[Proposal2D] = []
            anchor_mask = self._bbox_mask(frame.rgb.shape[:2], np.asarray(anchor.bbox_xyxy, dtype=np.float32))
            anchor_area = max(int(anchor_mask.sum()), 1)
            for proposal in sam_proposals:
                proposal_mask = np.asarray(proposal.mask, dtype=bool)
                proposal_area = max(int(proposal_mask.sum()), 1)
                overlap_mask = proposal_mask & anchor_mask
                overlap_area = int(overlap_mask.sum())
                if overlap_area < self.proposal_min_area:
                    continue
                proposal_anchor_coverage = overlap_area / proposal_area
                anchor_proposal_coverage = overlap_area / anchor_area
                if (
                    proposal_anchor_coverage < self.min_proposal_anchor_coverage
                    and anchor_proposal_coverage < self.min_anchor_proposal_coverage
                ):
                    continue
                selected.append(proposal)
                matched_sam_ids.add(int(proposal.proposal_id))

            if not selected:
                continue

            union_mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
            for proposal in selected:
                source_mask = np.asarray(proposal.mask, dtype=bool)
                union_mask |= (source_mask & anchor_mask) if self.clip_to_anchor_box else source_mask

            area = int(union_mask.sum())
            if area < self.proposal_min_area:
                continue

            matched_anchor_count += 1
            source_ids = [int(item.proposal_id) for item in selected]
            proposal_id = len(fine)
            fine.append(
                Proposal2D(
                    proposal_id=proposal_id,
                    mask=union_mask,
                    bbox_xyxy=self._mask_bbox(union_mask),
                    area=area,
                    confidence=max(float(anchor.confidence), max(float(item.confidence) for item in selected)),
                    backend_name="async_sam_refinement",
                    metadata={
                        "source": "async_sam_refinement",
                        "observation_layer": "fine",
                        "refinement_key": self.refinement_key(frame, anchor),
                        "anchor_id": int(anchor.anchor_id),
                        "anchor_class_name": str(anchor.class_name),
                        "anchor_confidence": float(anchor.confidence),
                        "anchor_label_strength": "strong",
                        "anchor_keepalive": True,
                        "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
                        "source_raw_proposal_ids": source_ids,
                        "source_raw_proposal_confidences": [float(item.confidence) for item in selected],
                        "mask_source": "delayed_sam_union",
                    },
                )
            )

        debug = {
            "enabled": bool(self.enabled),
            "anchor_count": int(len(anchors)),
            "sam_proposal_count": int(len(sam_proposals)),
            "matched_anchor_count": int(matched_anchor_count),
            "fine_proposal_count": int(len(fine)),
            "matched_sam_proposal_count": int(len(matched_sam_ids)),
            "unmatched_sam_proposal_count": int(
                len({int(item.proposal_id) for item in sam_proposals} - matched_sam_ids)
            ),
        }
        self.last_debug = debug
        return fine, debug

    @staticmethod
    def refinement_key(frame: Frame, anchor: Anchor2D) -> str:
        frame_id = int(frame.source_frame_id if frame.source_frame_id is not None else frame.frame_id)
        return f"{frame_id}:{int(anchor.anchor_id)}"

    @staticmethod
    def _bbox_mask(shape: tuple[int, int], bbox_xyxy: np.ndarray) -> np.ndarray:
        height, width = shape
        x1 = int(np.clip(np.floor(float(bbox_xyxy[0])), 0, width))
        y1 = int(np.clip(np.floor(float(bbox_xyxy[1])), 0, height))
        x2 = int(np.clip(np.ceil(float(bbox_xyxy[2])), 0, width))
        y2 = int(np.clip(np.ceil(float(bbox_xyxy[3])), 0, height))
        mask = np.zeros((height, width), dtype=bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if len(xs) == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
```

- [ ] **Step 4: Run async-refinement tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_async_refinement.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/async_refinement.py tests/test_async_refinement.py
git commit -m "feat: build delayed sam refinement proposals"
```

## Task 3: Replace Coarse Local-Memory Observations

**Files:**
- Modify: `src/modules/object_update.py`
- Modify: `src/modules/tsdf_instance_map.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing replacement test**

Append this test inside `class TestObjectUpdateModule` in `tests/test_pipeline.py`:

```python
    def test_replace_observations_removes_coarse_points_and_rebuilds_anchor_votes(self):
        module = ObjectUpdateModule(
            {
                "surface_owner_gate": {"enabled": False},
                "provisional": {"enabled": False},
            }
        )
        state = SystemState()
        coarse_points = np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            dtype=np.float32,
        )
        coarse_patch = Patch3D(
            patch_id=1,
            points=coarse_points,
            centroid=coarse_points.mean(axis=0),
            bbox_min=coarse_points.min(axis=0),
            bbox_max=coarse_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 1,
                "source_backend_name": "anchor_box_primary",
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.9,
                "anchor_label_strength": "strong",
                "observation_layer": "coarse",
                "refinement_key": "5:9",
            },
        )
        state = module.process(AssociationResult(new_object_patches=[1]), [coarse_patch], state)

        fine_points = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
        fine_patch = Patch3D(
            patch_id=2,
            points=fine_points,
            centroid=fine_points.mean(axis=0),
            bbox_min=fine_points.min(axis=0),
            bbox_max=fine_points.max(axis=0),
            source_frame_id=5,
            metadata={
                "source_proposal_id": 2,
                "source_backend_name": "async_sam_refinement",
                "anchor_id": 9,
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.95,
                "anchor_label_strength": "strong",
                "observation_layer": "fine",
                "refinement_key": "5:9",
            },
        )

        summary = module.replace_observations(state, [fine_patch])

        obj = state.objects[0]
        assert summary["replaced_observation_count"] == 1
        assert summary["inserted_observation_count"] == 1
        assert len(obj.observations) == 1
        assert obj.observations[0].observation_layer == "fine"
        assert len(obj.local_pcd) == 2
        assert obj.debug["anchor_semantics"]["label_frame_hits"] == {"sofa": 1}
        assert obj.debug["last_async_refinement"]["replaced_observation_count"] == 1
```

- [ ] **Step 2: Run replacement test and verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestObjectUpdateModule::test_replace_observations_removes_coarse_points_and_rebuilds_anchor_votes -q
```

Expected: FAIL with `AttributeError: 'ObjectUpdateModule' object has no attribute 'replace_observations'`.

- [ ] **Step 3: Add TSDF rollback primitive**

In `src/modules/tsdf_instance_map.py`, add this method after `integrate_patch()`:

```python
    def remove_patch_support(
        self,
        volume: TSDFInstanceVolume,
        patch: Patch3D,
        instance_id: int,
    ) -> None:
        """Remove one patch worth of owner support for an instance.

        This is intentionally conservative: it only subtracts support from the
        requested instance and leaves occupancy/weight intact. The first
        coarse-to-fine integration keeps this disabled by default, but tests and
        future configs can enable support rollback when replacement artifacts
        are visible in the TSDF backbone.
        """
        voxel_indices = _world_to_voxel(patch.points, volume.voxel_size)
        unique_voxels = set(map(tuple, voxel_indices.tolist()))
        for vk in unique_voxels:
            support = volume.owner_support.get(vk)
            if support is None or instance_id not in support.support:
                continue
            support.support[instance_id] = float(support.support[instance_id]) - float(volume.support_increment)
            if support.support[instance_id] <= 0.01:
                del support.support[instance_id]
            if not support.support:
                del volume.owner_support[vk]
```

- [ ] **Step 4: Implement replacement helper**

In `src/modules/object_update.py`, add this import near semantic-memory imports:

```python
from src.modules.semantic_memory import rebuild_anchor_semantic_votes_from_observations
```

In `ObjectUpdateModule.__init__`, add:

```python
        refinement_cfg = config.get("async_refinement", {})
        self.refinement_replace_enabled = bool(refinement_cfg.get("replace_coarse_observations", True))
        self.refinement_rebuild_tsdf_support = bool(refinement_cfg.get("rebuild_tsdf_support", False))
        self.last_async_refinement_summary: dict[str, Any] = {}
```

Add this method before `_filter_patch_by_current_frame_visibility()`:

```python
    def replace_observations(self, state: SystemState, fine_patches: list[Patch3D]) -> dict[str, Any]:
        summary = {
            "enabled": bool(self.refinement_replace_enabled),
            "fine_patch_count": int(len(fine_patches)),
            "replaced_observation_count": 0,
            "inserted_observation_count": 0,
            "updated_object_count": 0,
            "rebuild_tsdf_support": bool(self.refinement_rebuild_tsdf_support),
        }
        if not self.refinement_replace_enabled or not fine_patches:
            self.last_async_refinement_summary = summary
            return summary

        updated_ids: set[int] = set()
        for fine_patch in fine_patches:
            key = str(fine_patch.metadata.get("refinement_key", ""))
            if not key:
                continue
            target_obj: ObjectMap | None = None
            target_id = -1
            replaced_records: list[ObservationRecord] = []
            for obj_id, obj in state.objects.items():
                matches = [
                    obs
                    for obs in obj.observations
                    if str(obs.refinement_key) == key
                    and str(obs.observation_layer) == "coarse"
                    and not bool(obs.replaced_by_refinement)
                ]
                if matches:
                    target_obj = obj
                    target_id = int(obj_id)
                    replaced_records = matches
                    break
            if target_obj is None:
                continue

            for obs in replaced_records:
                obs.replaced_by_refinement = True
                if self.refinement_rebuild_tsdf_support:
                    self.tsdf_module.remove_patch_support(state.tsdf_volume, obs.patch, target_id)
            target_obj.observations = [
                obs
                for obs in target_obj.observations
                if not (str(obs.refinement_key) == key and str(obs.observation_layer) == "coarse")
            ]
            target_obj.observations.append(self._observation_from_patch(fine_patch))
            target_obj.local_pcd = (
                np.concatenate([obs.patch.points for obs in target_obj.observations], axis=0)
                if target_obj.observations
                else np.empty((0, 3), dtype=np.float32)
            )
            if len(target_obj.local_pcd) > 0:
                if len(target_obj.local_pcd) > self.max_points:
                    target_obj.local_pcd = self._deterministic_spatial_cap(target_obj.local_pcd, self.max_points)
                target_obj.centroid = target_obj.local_pcd.mean(axis=0)
                target_obj.bbox_min, target_obj.bbox_max = compute_bbox(target_obj.local_pcd)
            target_obj.update_count = int(len(target_obj.observations))
            target_obj.last_seen_frame = max(int(obs.frame_id) for obs in target_obj.observations)
            if self.refinement_rebuild_tsdf_support:
                self.tsdf_module.integrate_patch(state.tsdf_volume, fine_patch, target_id)
            rebuild_anchor_semantic_votes_from_observations(target_obj)
            target_obj.debug["last_async_refinement"] = {
                "refinement_key": key,
                "replaced_observation_count": int(len(replaced_records)),
                "inserted_observation_count": 1,
                "fine_patch_point_count": int(len(fine_patch.points)),
                "rebuild_tsdf_support": bool(self.refinement_rebuild_tsdf_support),
            }
            updated_ids.add(target_id)
            summary["replaced_observation_count"] += int(len(replaced_records))
            summary["inserted_observation_count"] += 1

        summary["updated_object_count"] = int(len(updated_ids))
        self.last_async_refinement_summary = summary
        return summary
```

- [ ] **Step 5: Run replacement test**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestObjectUpdateModule::test_replace_observations_removes_coarse_points_and_rebuilds_anchor_votes -q
```

Expected: PASS.

- [ ] **Step 6: Run affected tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py tests/test_object_anchor.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/object_update.py src/modules/tsdf_instance_map.py tests/test_pipeline.py
git commit -m "feat: replace coarse observations with refined patches"
```

## Task 4: Integrate Delayed Refinement Into Pipeline

**Files:**
- Modify: `src/modules/async_refinement.py`
- Modify: `src/pipelines/main_pipeline.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_async_refinement.py`

- [ ] **Step 1: Write failing pipeline integration test**

Append this test inside `class TestPipeline` in `tests/test_pipeline.py`:

```python
    def test_pipeline_applies_ready_async_refinement_after_coarse_update(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
async_refinement:
  enabled: true
  delay_frames: 0
  proposal_min_area: 1
  replace_coarse_observations: true
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe.object_anchor.generate_anchor_box_proposals = lambda rgb: (
            [
                Anchor2D(
                    anchor_id=4,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                )
            ],
            [
                Proposal2D(
                    proposal_id=0,
                    mask=np.pad(np.ones((6, 6), dtype=bool), ((1, 1), (1, 1))),
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=36,
                    confidence=0.9,
                    backend_name="anchor_box_primary",
                    metadata={
                        "source": "anchor_box_primary",
                        "anchor_id": 4,
                        "anchor_class_name": "sofa",
                        "anchor_confidence": 0.9,
                        "anchor_label_strength": "strong",
                        "anchor_keepalive": True,
                        "observation_layer": "coarse",
                        "refinement_key": "100:4",
                    },
                )
            ],
            [AnchorAssignment(proposal_id=0, anchor_id=4, class_name="sofa", confidence=0.9, keepalive=True)],
        )

        sam_mask = np.zeros((8, 8), dtype=bool)
        sam_mask[2:6, 2:6] = True
        pipe.async_refinement_backend = lambda frame: [
            Proposal2D(
                proposal_id=20,
                mask=sam_mask,
                bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                area=16,
                confidence=0.8,
                backend_name="sam2",
            )
        ]

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=100)

        obj = pipe.state.objects[0]
        assert pipe.last_frame_debug["async_refinement"]["fine_proposal_count"] == 1
        assert pipe.last_frame_debug["async_refinement"]["replaced_observation_count"] == 1
        assert len(obj.observations) == 1
        assert obj.observations[0].observation_layer == "fine"
        assert len(obj.local_pcd) == 16
        assert pipe.last_frame_debug["stage_timings"]["async_refinement"] >= 0.0
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_applies_ready_async_refinement_after_coarse_update -q
```

Expected: FAIL with missing `async_refinement` attribute or missing debug key.

- [ ] **Step 3: Extend async refinement module with backend hook**

In `src/modules/async_refinement.py`, add this method inside `AsyncRefinementModule`:

```python
    def process_ready(
        self,
        *,
        frame: Frame,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
    ) -> RefinementResult:
        if not self.enabled:
            debug = {
                "enabled": False,
                "fine_proposal_count": 0,
                "replaced_observation_count": 0,
                "inserted_observation_count": 0,
                "updated_object_count": 0,
            }
            self.last_debug = debug
            return RefinementResult([], debug)
        fine, debug = self.build_fine_proposals(frame=frame, anchors=anchors, sam_proposals=sam_proposals)
        return RefinementResult(fine, debug)
```

- [ ] **Step 4: Integrate into `Pipeline`**

In `src/pipelines/main_pipeline.py`, add this import:

```python
from src.modules.async_refinement import AsyncRefinementModule
```

In `Pipeline.__init__`, after `self.object_update = ...`, add:

```python
        self.async_refinement = AsyncRefinementModule(self.config.get("async_refinement", {}))
        self.async_refinement_backend = None
```

After the `object_update` stage and before semantic memory, add:

```python
        async_refinement_summary = {
            "enabled": bool(self.async_refinement.enabled),
            "sam_proposal_count": 0,
            "fine_proposal_count": 0,
            "replaced_observation_count": 0,
            "inserted_observation_count": 0,
            "updated_object_count": 0,
        }
        if self.async_refinement.enabled:
            with self._timed_stage("async_refinement"):
                if self.async_refinement_backend is not None:
                    refinement_sam_proposals = list(self.async_refinement_backend(frame))
                else:
                    refinement_sam_proposals = list(self.proposal.process(frame.rgb, frame.depth, frame=frame))
                refinement_result = self.async_refinement.process_ready(
                    frame=frame,
                    anchors=list(anchors),
                    sam_proposals=refinement_sam_proposals,
                )
                fine_refined = self.depth_refinement.process(frame.depth, refinement_result.fine_proposals)
                fine_patches = self.patch_lifting.process(
                    fine_refined,
                    frame.depth,
                    frame.pose,
                    frame.intrinsics,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                )
                replacement_summary = self.object_update.replace_observations(self.state, fine_patches)
                async_refinement_summary = {
                    **refinement_result.debug,
                    **replacement_summary,
                    "sam_proposal_count": int(len(refinement_sam_proposals)),
                    "fine_refined_proposal_count": int(len(fine_refined)),
                    "fine_patch_count": int(len(fine_patches)),
                }
```

In `last_frame_debug`, add:

```python
            "async_refinement": dict(async_refinement_summary),
```

- [ ] **Step 5: Run pipeline integration test**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_applies_ready_async_refinement_after_coarse_update -q
```

Expected: PASS.

- [ ] **Step 6: Run focused test set**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_async_refinement.py tests/test_pipeline.py::TestPipeline tests/test_pipeline.py::TestObjectUpdateModule -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add src/modules/async_refinement.py src/pipelines/main_pipeline.py tests/test_async_refinement.py tests/test_pipeline.py
git commit -m "feat: apply delayed sam refinement in pipeline"
```

## Task 5: Report Refinement Metrics

**Files:**
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_dual_map.py`

- [ ] **Step 1: Write failing report tests**

Append this test in `tests/test_dual_map.py` after `test_build_stage_timing_summary_skips_bad_values()`:

```python
def test_build_async_refinement_summary_totals_counts() -> None:
    from run_room0_full_eval import build_async_refinement_summary

    frame_metrics = [
        {
            "async_refinement": {
                "sam_proposal_count": 10,
                "fine_proposal_count": 3,
                "fine_patch_count": 2,
                "replaced_observation_count": 2,
                "inserted_observation_count": 2,
                "updated_object_count": 1,
            }
        },
        {
            "async_refinement": {
                "sam_proposal_count": 8,
                "fine_proposal_count": 1,
                "fine_patch_count": 1,
                "replaced_observation_count": 1,
                "inserted_observation_count": 1,
                "updated_object_count": 1,
            }
        },
    ]

    assert build_async_refinement_summary(frame_metrics) == {
        "async_refinement_sam_proposal_count_total": 18,
        "async_refinement_fine_proposal_count_total": 4,
        "async_refinement_fine_patch_count_total": 3,
        "async_refinement_replaced_observation_count_total": 3,
        "async_refinement_inserted_observation_count_total": 3,
        "async_refinement_updated_object_count_total": 2,
    }
```

In `test_write_run_report_includes_stage_timing_summary()`, add `"async_refinement": {...}` to both frame metric dictionaries:

```python
            "async_refinement": {
                "sam_proposal_count": 10,
                "fine_proposal_count": 3,
                "fine_patch_count": 2,
                "replaced_observation_count": 2,
                "inserted_observation_count": 2,
                "updated_object_count": 1,
            },
```

and:

```python
            "async_refinement": {
                "sam_proposal_count": 8,
                "fine_proposal_count": 1,
                "fine_patch_count": 1,
                "replaced_observation_count": 1,
                "inserted_observation_count": 1,
                "updated_object_count": 1,
            },
```

Then add assertions after `report_json = ...`:

```python
    assert "- `async_refinement_fine_proposal_count_total`: `4`" in report_md
    assert "- `async_refinement_replaced_observation_count_total`: `3`" in report_md
    assert report_json["async_refinement_fine_patch_count_total"] == 3
    assert report_json["async_refinement_updated_object_count_total"] == 2
```

- [ ] **Step 2: Run report tests and verify they fail**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_dual_map.py::test_build_async_refinement_summary_totals_counts tests/test_dual_map.py::test_write_run_report_includes_stage_timing_summary -q
```

Expected: FAIL with `ImportError` for `build_async_refinement_summary` or missing report fields.

- [ ] **Step 3: Add summary builder**

In `run_room0_full_eval.py`, after `build_frontend_stage_report_payload()`, add:

```python
def build_async_refinement_summary(frame_metrics: list[dict[str, Any]]) -> dict[str, int]:
    keys = {
        "sam_proposal_count": "async_refinement_sam_proposal_count_total",
        "fine_proposal_count": "async_refinement_fine_proposal_count_total",
        "fine_patch_count": "async_refinement_fine_patch_count_total",
        "replaced_observation_count": "async_refinement_replaced_observation_count_total",
        "inserted_observation_count": "async_refinement_inserted_observation_count_total",
        "updated_object_count": "async_refinement_updated_object_count_total",
    }
    totals = {output_key: 0 for output_key in keys.values()}
    for metrics in frame_metrics:
        refinement = dict(metrics.get("async_refinement", {}) or {})
        for input_key, output_key in keys.items():
            totals[output_key] += int(refinement.get(input_key, 0) or 0)
    return totals
```

- [ ] **Step 4: Write per-frame async field**

In the frame `record = { ... }` block in `run_room0_full_eval.py`, add:

```python
                    "async_refinement": dict(pipeline.last_frame_debug.get("async_refinement", {}) or {}),
```

- [ ] **Step 5: Add report fields**

In `write_run_report()`, after:

```python
    frontend_stage_totals = build_frontend_stage_report_payload(frame_metrics)
```

add:

```python
    async_refinement_totals = build_async_refinement_summary(frame_metrics)
```

In the Evaluation Metadata `lines` list, after `runtime_anchor_identity_mismatch_edge_count_total`, add:

```python
        f"- `async_refinement_sam_proposal_count_total`: `{async_refinement_totals['async_refinement_sam_proposal_count_total']}`",
        f"- `async_refinement_fine_proposal_count_total`: `{async_refinement_totals['async_refinement_fine_proposal_count_total']}`",
        f"- `async_refinement_fine_patch_count_total`: `{async_refinement_totals['async_refinement_fine_patch_count_total']}`",
        f"- `async_refinement_replaced_observation_count_total`: `{async_refinement_totals['async_refinement_replaced_observation_count_total']}`",
        f"- `async_refinement_inserted_observation_count_total`: `{async_refinement_totals['async_refinement_inserted_observation_count_total']}`",
        f"- `async_refinement_updated_object_count_total`: `{async_refinement_totals['async_refinement_updated_object_count_total']}`",
```

In the `report_json` dict, add:

```python
        **async_refinement_totals,
```

- [ ] **Step 6: Run report tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_dual_map.py::test_build_async_refinement_summary_totals_counts tests/test_dual_map.py::test_write_run_report_includes_stage_timing_summary -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add run_room0_full_eval.py tests/test_dual_map.py
git commit -m "feat: report async refinement metrics"
```

## Task 6: Add Coarse-To-Fine Room0 Config

**Files:**
- Create: `configs/room0_coarse_to_fine_async_4090.yaml`
- Test: `tests/test_dual_map.py`

- [ ] **Step 1: Write failing config test**

Append this test in `tests/test_dual_map.py`:

```python
def test_room0_coarse_to_fine_async_config_enables_fast_coarse_and_refinement() -> None:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "room0_coarse_to_fine_async_4090.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["anchor_frontend"]["anchor_primary_mode"] is True
    assert config["anchor_frontend"]["use_sam_intersection_proposals"] is False
    assert config["async_refinement"]["enabled"] is True
    assert config["async_refinement"]["delay_frames"] == 0
    assert config["async_refinement"]["replace_coarse_observations"] is True
    assert config["async_refinement"]["rebuild_tsdf_support"] is False
    assert config["pipeline"]["collect_stage_timings"] is True
    assert config["patch_lifting"]["point_sample_ratio"] == 0.08
    assert config["depth_refinement"]["connected_components_backend"] == "opencv"
    assert config["depth_refinement"]["use_bbox_crop"] is True
```

- [ ] **Step 2: Run config test and verify it fails**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_dual_map.py::test_room0_coarse_to_fine_async_config_enables_fast_coarse_and_refinement -q
```

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create config**

Copy `configs/room0_fast_hybrid_4090.yaml` to `configs/room0_coarse_to_fine_async_4090.yaml`, then add this top-level section:

```yaml
async_refinement:
  enabled: true
  delay_frames: 0
  proposal_min_area: 25
  min_proposal_anchor_coverage: 0.20
  min_anchor_proposal_coverage: 0.05
  clip_to_anchor_box: false
  replace_coarse_observations: true
  rebuild_tsdf_support: false
```

Inside `object_update`, add:

```yaml
  async_refinement:
    replace_coarse_observations: true
    rebuild_tsdf_support: false
```

Keep these fast settings from `room0_fast_hybrid_4090.yaml` unchanged:

```yaml
anchor_frontend:
  use_boxes_as_primary_proposals: true
  use_sam_intersection_proposals: false
  anchor_primary_mode: true
patch_lifting:
  point_sample_ratio: 0.08
  max_points_per_patch: 2048
depth_refinement:
  connected_components_backend: opencv
  use_bbox_crop: true
pipeline:
  collect_stage_timings: true
```

- [ ] **Step 4: Run config test**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_dual_map.py::test_room0_coarse_to_fine_async_config_enables_fast_coarse_and_refinement -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git add configs/room0_coarse_to_fine_async_4090.yaml tests/test_dual_map.py
git commit -m "config: add coarse-to-fine async room0 mode"
```

## Task 7: Verification And Smoke Experiment

**Files:**
- No code files unless a test exposes a defect.

- [ ] **Step 1: Run core unit tests**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
pytest tests/test_async_refinement.py tests/test_object_anchor.py tests/test_pipeline.py tests/test_dual_map.py -q
```

Expected: PASS.

- [ ] **Step 2: Run 20f smoke**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_coarse_to_fine_async_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260529_room0_coarse_to_fine_async_stride10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected:
- Command exits 0.
- `outputs/tmp_validation/20260529_room0_coarse_to_fine_async_stride10_20f/room0/run_report.json` exists.
- Report JSON contains `async_refinement_fine_proposal_count_total > 0`.
- Report JSON contains `async_refinement_replaced_observation_count_total > 0`.
- mIoU should be at least the old fast-hybrid 20f value `0.1135`.

- [ ] **Step 3: Compare smoke timings**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
paths = [
    "outputs/tmp_validation/20260529_room0_fast_hybrid_worker_depth_stride10_20f/room0/run_report.json",
    "outputs/tmp_validation/20260529_room0_coarse_to_fine_async_stride10_20f/room0/run_report.json",
]
for path in paths:
    d = json.load(open(path))
    st = d.get("stage_timing_summary", {})
    total = sum(v.get("total_sec", 0.0) for k, v in st.items() if k not in {"yoloworld_primary", "yoloe_supplemental", "anchor_merge"})
    print(path)
    print("miou", d.get("miou"), "fmiou", d.get("fmiou"), "objects", d.get("final_object_count"))
    print("pipeline_total_sec", round(total, 3), "per_frame", round(total / max(d.get("frame_count", 20), 1), 3))
    print("async_fine", d.get("async_refinement_fine_proposal_count_total"), "replaced", d.get("async_refinement_replaced_observation_count_total"))
PY
```

Expected:
- Coarse-to-fine mIoU is not below fast-hybrid smoke.
- Per-frame time may increase versus fast-hybrid, but should remain below the synchronous 0526 high-accuracy path.

- [ ] **Step 4: Commit any test-only fixes**

If Step 1-3 required code changes, run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
git status --short
git add src/modules/async_refinement.py src/core/data_structures.py src/modules/object_update.py src/modules/tsdf_instance_map.py src/pipelines/main_pipeline.py run_room0_full_eval.py configs/room0_coarse_to_fine_async_4090.yaml tests/test_async_refinement.py tests/test_pipeline.py tests/test_dual_map.py
git commit -m "fix: stabilize coarse-to-fine async refinement smoke"
```

If no fixes were needed, do not create an empty commit.

## Task 8: Run 200f Comparison Experiment

**Files:**
- No source changes.

- [ ] **Step 1: Start stride=10 200f experiment**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
tmux new-session -d -s room0_c2f_async_s10_200f "/bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_coarse_to_fine_async_4090.yaml --output-root outputs/tmp_validation --experiment-name 20260529_room0_coarse_to_fine_async_stride10_200f --num-frames 200 --frame-stride 10 --proposal-backend precomputed --proposal-device cuda --fast-eval > outputs/tmp_validation/20260529_room0_coarse_to_fine_async_stride10_200f.log 2>&1'"
```

Expected: tmux session starts.

- [ ] **Step 2: Monitor completion**

Run:

```bash
tmux capture-pane -pt room0_c2f_async_s10_200f -S -80
```

Expected during run: progress lines show frames processed. Expected after completion: report and eval paths printed.

- [ ] **Step 3: Compare against 0526 and fast-hybrid 200f**

Run after completion:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
paths = [
    "outputs/tmp_validation/20260526_room0_observation_first_fix52e1714_200f/room0/run_report.json",
    "outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_200f/room0/run_report.json",
    "outputs/tmp_validation/20260529_room0_fast_hybrid_worker_depth_stride10_200f/room0/run_report.json",
    "outputs/tmp_validation/20260529_room0_coarse_to_fine_async_stride10_200f/room0/run_report.json",
]
for path in paths:
    d = json.load(open(path))
    st = d.get("stage_timing_summary", {})
    total = sum(v.get("total_sec", 0.0) for k, v in st.items() if k not in {"yoloworld_primary", "yoloe_supplemental", "anchor_merge"})
    print(path)
    print("miou", d.get("miou"), "fmiou", d.get("fmiou"), "macc", d.get("macc"), "fmacc", d.get("fmacc"))
    print("objects", d.get("final_object_count"), "pool", d.get("pool_point_count"), "dense", d.get("dense_surface_point_count"))
    print("pipeline_total_sec", round(total, 3), "per_frame", round(total / max(d.get("frame_count", 200), 1), 3))
    print("source_sam", d.get("source_sam_proposal_count_total"), "anchor_voted", d.get("anchor_voted_proposal_count_total"))
    print("async_fine", d.get("async_refinement_fine_proposal_count_total"), "replaced", d.get("async_refinement_replaced_observation_count_total"))
PY
```

Expected:
- Coarse-to-fine mIoU should be materially above fast-hybrid `0.2790`.
- Coarse-to-fine time should be below or close to the 0526 synchronous high-accuracy path.
- `async_refinement_replaced_observation_count_total` should be nonzero; if zero, refinement integration is not working and execution should stop for debugging.

## Self-Review Notes

- This plan intentionally does not reimplement already completed fast-hybrid pieces: persistent YOLOE worker, OpenCV depth connected components, anchor-primary proposals, patch point sampling, and stage timing.
- First implementation uses delayed in-process refinement from the existing precomputed SAM cache. It validates the map logic without adding GPU thread-safety risk.
- The first version replaces `ObjectMap.local_pcd` and anchor semantic evidence, because current evaluation is based on pool/dense projection from object local memory. TSDF support rollback is implemented but disabled by config until visual artifacts prove it is needed.
- The plan keeps unanchored SAM masks out of semantic inheritance. They are counted in debug but not inserted as labeled objects in the first version, preserving the safety rule that a proposal cannot inherit sofa/chair just because it overlaps a parent anchor.
