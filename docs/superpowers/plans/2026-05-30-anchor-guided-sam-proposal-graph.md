# Anchor-Guided SAM Proposal Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DualMap-like anchor-guided SAM path where YOLO/YOLOWorld/YOLOE anchors define candidate objects and SAM only refines local mask boundaries, while weakly related SAM masks stay unknown/residual instead of inheriting a parent semantic label.

**Architecture:** Add a pure `AnchorGuidedSAMModule` that takes detector anchors plus SAM masks and emits a proposal graph: strong anchored masks, anchor-box fallbacks, and unknown residual masks. Wire this as a new pipeline proposal source for anchor-primary mode, using the existing precomputed SAM cache for the first experiment and preserving the option to add a real promptable SAM backend after the behavior is validated.

**Tech Stack:** Python 3.10, NumPy mask operations, PyYAML config, existing `Proposal2D`/`Anchor2D` dataclasses, existing `Pipeline`, existing Replica room0 evaluation runner, pytest.

---

## File Structure

- Create `src/modules/anchor_guided_sam.py`
  - Owns all anchor/SAM relation logic.
  - Exposes `AnchorGuidedSAMModule.build_proposals(frame, anchors, sam_proposals)`.
  - Does not call YOLO, SAM, depth refinement, object update, or semantic memory.
  - Produces only `Proposal2D` plus a debug dictionary.

- Modify `src/pipelines/main_pipeline.py`
  - Imports and constructs `AnchorGuidedSAMModule`.
  - Adds a proposal-generation branch after anchor detection in anchor-primary mode.
  - Loads SAM proposals once through the existing `ProposalModule` when `anchor_guided_sam.enabled` is true.
  - Records `last_frame_debug["anchor_guided_sam"]`.

- Modify `src/modules/proposal.py`
  - Adds `process_for_anchors(...)` as a non-breaking wrapper.
  - For this first implementation it delegates to `process(...)`; a later promptable SAM backend can override the lower-level backend interface.

- Modify `src/models/proposal_backend.py`
  - Adds optional `generate_proposals_for_anchors(...)` on `ProposalBackend`.
  - Default implementation delegates to `generate_proposals(...)`.
  - No existing backend is required to change for the first experiment.

- Create `configs/room0_anchor_guided_sam_4090.yaml`
  - Copies the current fast hybrid setup from `configs/room0_fast_hybrid_4090.yaml`.
  - Enables `anchor_guided_sam`.
  - Keeps `proposal.backend: precomputed` for reproducible 20f/200f validation.
  - Disables `structural_overlay`.

- Create `tests/test_anchor_guided_sam.py`
  - Unit tests for mask relation classification and metadata.
  - Covers the sofa/blanket/rug failure mode: a small SAM mask inside a sofa anchor must not become sofa when the relation is contained-but-not-scale-compatible.

- Modify `tests/test_pipeline.py`
  - Adds pipeline wiring tests for anchor-guided mode.
  - Verifies the pipeline uses SAM candidates in anchor-primary mode only when the new mode is enabled.

---

## Invariants

The implementation must preserve these rules:

1. Detector anchors are the only source of concrete class labels in this path.
2. A SAM mask may receive an anchor label only when the mask and anchor have a strong, scale-compatible relation.
3. A SAM mask mostly inside a larger anchor but covering only a small fraction of that anchor is an unknown residual unless another matching child anchor exists.
4. Unknown residuals must carry no class vote and must set `semantic_commit_allowed: false`.
5. SAM masks outside anchors are discarded in this fast path unless `include_unanchored_residuals` is explicitly enabled.
6. Structure classes and object classes are evaluated separately; structure anchors must not overwrite object-neighborhood residuals.
7. The initial validated path uses precomputed SAM cache to simulate anchor-guided selection. A real box-prompt SAM backend is a follow-up interface task, not a requirement for the first 200f test.

---

## Task 1: Add Pure Anchor-Guided SAM Module

**Files:**
- Create: `src/modules/anchor_guided_sam.py`
- Test: `tests/test_anchor_guided_sam.py`

- [ ] **Step 1: Write failing tests for strong anchored refinement**

Add the following helper and test to `tests/test_anchor_guided_sam.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Anchor2D, CameraIntrinsics, Frame, Proposal2D
from src.modules.anchor_guided_sam import AnchorGuidedSAMModule


def _frame() -> Frame:
    return Frame(
        frame_id=1,
        source_frame_id=10,
        rgb=np.zeros((12, 12, 3), dtype=np.uint8),
        depth=np.ones((12, 12), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=6.0, cy=6.0, width=12, height=12),
    )


def _anchor(anchor_id: int, x1: int, y1: int, x2: int, y2: int, class_name: str = "sofa") -> Anchor2D:
    return Anchor2D(
        anchor_id=anchor_id,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        class_name=class_name,
        confidence=0.91,
    )


def _proposal(proposal_id: int, x1: int, y1: int, x2: int, y2: int) -> Proposal2D:
    mask = np.zeros((12, 12), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return Proposal2D(
        proposal_id=proposal_id,
        mask=mask,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.87,
        backend_name="sam2",
        metadata={"source": "sam2"},
    )


def test_anchor_guided_sam_labels_scale_compatible_mask() -> None:
    module = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "min_proposal_anchor_coverage_for_label": 0.70,
            "min_anchor_proposal_coverage_for_label": 0.20,
            "contained_residual_enabled": True,
            "contained_min_proposal_coverage": 0.85,
            "contained_max_anchor_coverage": 0.35,
            "clip_to_anchor_box": True,
            "include_unknown_residuals": True,
            "max_sam_proposals_per_anchor": 4,
        }
    )

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[_anchor(3, 2, 2, 8, 8, "sofa")],
        sam_proposals=[_proposal(40, 2, 2, 8, 8)],
    )

    assert len(proposals) == 1
    assert len(assignments) == 1
    assert proposals[0].proposal_id == 3
    assert proposals[0].metadata["source"] == "anchor_guided_sam"
    assert proposals[0].metadata["anchor_id"] == 3
    assert proposals[0].metadata["anchor_class_name"] == "sofa"
    assert proposals[0].metadata["anchor_label_strength"] == "strong"
    assert proposals[0].metadata["semantic_commit_allowed"] is True
    assert proposals[0].metadata["mask_anchor_relation"] == "scale_compatible"
    assert proposals[0].metadata["source_raw_proposal_ids"] == [40]
    assert assignments[0].anchor_id == 3
    assert assignments[0].class_name == "sofa"
    assert debug["anchored_proposal_count"] == 1
    assert debug["unknown_residual_count"] == 0
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
pytest tests/test_anchor_guided_sam.py::test_anchor_guided_sam_labels_scale_compatible_mask -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.modules.anchor_guided_sam'`.

- [ ] **Step 3: Add the module skeleton and strong-match implementation**

Create `src/modules/anchor_guided_sam.py` with:

```python
"""Anchor-guided SAM proposal graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.data_structures import Anchor2D, AnchorAssignment, Frame, Proposal2D


@dataclass(frozen=True)
class MaskAnchorRelation:
    proposal: Proposal2D
    anchor: Anchor2D
    overlap_area: int
    proposal_anchor_coverage: float
    anchor_proposal_coverage: float
    bbox_iou: float
    relation: str


class AnchorGuidedSAMModule:
    """Turn detector anchors plus SAM masks into safe local proposals."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.proposal_min_area = int(self.config.get("proposal_min_area", 25))
        self.min_proposal_anchor_coverage_for_label = float(
            np.clip(self.config.get("min_proposal_anchor_coverage_for_label", 0.70), 0.0, 1.0)
        )
        self.min_anchor_proposal_coverage_for_label = float(
            np.clip(self.config.get("min_anchor_proposal_coverage_for_label", 0.20), 0.0, 1.0)
        )
        self.contained_residual_enabled = bool(self.config.get("contained_residual_enabled", True))
        self.contained_min_proposal_coverage = float(
            np.clip(self.config.get("contained_min_proposal_coverage", 0.85), 0.0, 1.0)
        )
        self.contained_max_anchor_coverage = float(
            np.clip(self.config.get("contained_max_anchor_coverage", 0.35), 0.0, 1.0)
        )
        self.clip_to_anchor_box = bool(self.config.get("clip_to_anchor_box", True))
        self.include_unknown_residuals = bool(self.config.get("include_unknown_residuals", True))
        self.include_anchor_box_fallbacks = bool(self.config.get("include_anchor_box_fallbacks", True))
        self.max_sam_proposals_per_anchor = int(self.config.get("max_sam_proposals_per_anchor", 4))
        self.structure_classes = {
            str(item).strip()
            for item in self.config.get("structure_classes", ["wall", "window", "blinds", "ceiling", "floor", "door"])
            if str(item).strip()
        }
        self.last_debug: dict[str, Any] = {}

    def build_proposals(
        self,
        *,
        frame: Frame,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
    ) -> tuple[list[Proposal2D], list[AnchorAssignment], dict[str, Any]]:
        if not self.enabled:
            debug = self._debug(
                anchors=anchors,
                sam_proposals=sam_proposals,
                output=[],
                assignments=[],
                matched_sam_ids=set(),
                dropped_count=0,
            )
            self.last_debug = debug
            return [], [], debug

        image_shape = frame.rgb.shape[:2]
        output: list[Proposal2D] = []
        assignments: list[AnchorAssignment] = []
        matched_sam_ids: set[int] = set()
        dropped_count = 0

        for anchor in anchors:
            relations = self._relations_for_anchor(image_shape, anchor, sam_proposals)
            strong = [relation for relation in relations if relation.relation == "scale_compatible"]
            strong = sorted(
                strong,
                key=lambda item: (-item.anchor_proposal_coverage, -item.proposal_anchor_coverage, int(item.proposal.proposal_id)),
            )[: self.max_sam_proposals_per_anchor]

            if strong:
                proposal = self._build_anchored_proposal(image_shape, anchor, strong, proposal_id=int(anchor.anchor_id))
                output.append(proposal)
                assignments.append(
                    AnchorAssignment(
                        proposal_id=int(proposal.proposal_id),
                        anchor_id=int(anchor.anchor_id),
                        class_name=str(anchor.class_name),
                        confidence=float(anchor.confidence),
                        bbox_iou=max(float(item.bbox_iou) for item in strong),
                        center_inside=True,
                        keepalive=True,
                    )
                )
                matched_sam_ids.update(int(item.proposal.proposal_id) for item in strong)
            elif self.include_anchor_box_fallbacks:
                proposal = self._build_anchor_box_fallback(image_shape, anchor, proposal_id=int(anchor.anchor_id))
                if proposal is not None:
                    output.append(proposal)
                    assignments.append(
                        AnchorAssignment(
                            proposal_id=int(proposal.proposal_id),
                            anchor_id=int(anchor.anchor_id),
                            class_name=str(anchor.class_name),
                            confidence=float(anchor.confidence),
                            bbox_iou=1.0,
                            center_inside=True,
                            keepalive=True,
                        )
                    )

            for relation in relations:
                if relation.relation != "contained_residual":
                    continue
                if int(relation.proposal.proposal_id) in matched_sam_ids:
                    continue
                residual = self._build_unknown_residual(relation.proposal, relation, proposal_id=self._residual_id(anchor, relation.proposal))
                if residual is None:
                    dropped_count += 1
                    continue
                output.append(residual)
                assignments.append(
                    AnchorAssignment(
                        proposal_id=int(residual.proposal_id),
                        anchor_id=-1,
                        class_name="",
                        confidence=0.0,
                        bbox_iou=float(relation.bbox_iou),
                        center_inside=True,
                        keepalive=False,
                    )
                )
                matched_sam_ids.add(int(relation.proposal.proposal_id))

        debug = self._debug(
            anchors=anchors,
            sam_proposals=sam_proposals,
            output=output,
            assignments=assignments,
            matched_sam_ids=matched_sam_ids,
            dropped_count=dropped_count,
        )
        self.last_debug = debug
        return output, assignments, debug

    def _relations_for_anchor(
        self,
        image_shape: tuple[int, int],
        anchor: Anchor2D,
        sam_proposals: list[Proposal2D],
    ) -> list[MaskAnchorRelation]:
        anchor_mask = self._bbox_mask(image_shape, np.asarray(anchor.bbox_xyxy, dtype=np.float32))
        anchor_area = int(anchor_mask.sum())
        if anchor_area <= 0:
            return []

        relations: list[MaskAnchorRelation] = []
        anchor_bbox = np.asarray(anchor.bbox_xyxy, dtype=np.float32)
        for proposal in sam_proposals:
            proposal_mask = np.asarray(proposal.mask, dtype=bool)
            if proposal_mask.shape != image_shape:
                continue
            proposal_area = int(proposal_mask.sum())
            if proposal_area < self.proposal_min_area:
                continue
            overlap_area = int((proposal_mask & anchor_mask).sum())
            if overlap_area <= 0:
                continue
            proposal_anchor_coverage = overlap_area / max(proposal_area, 1)
            anchor_proposal_coverage = overlap_area / max(anchor_area, 1)
            bbox_iou = self._bbox_iou(anchor_bbox, np.asarray(proposal.bbox_xyxy, dtype=np.float32))
            relation = self._classify_relation(
                proposal_anchor_coverage=proposal_anchor_coverage,
                anchor_proposal_coverage=anchor_proposal_coverage,
            )
            relations.append(
                MaskAnchorRelation(
                    proposal=proposal,
                    anchor=anchor,
                    overlap_area=overlap_area,
                    proposal_anchor_coverage=proposal_anchor_coverage,
                    anchor_proposal_coverage=anchor_proposal_coverage,
                    bbox_iou=bbox_iou,
                    relation=relation,
                )
            )
        return relations

    def _classify_relation(self, *, proposal_anchor_coverage: float, anchor_proposal_coverage: float) -> str:
        if (
            proposal_anchor_coverage >= self.min_proposal_anchor_coverage_for_label
            and anchor_proposal_coverage >= self.min_anchor_proposal_coverage_for_label
        ):
            return "scale_compatible"
        if (
            self.contained_residual_enabled
            and proposal_anchor_coverage >= self.contained_min_proposal_coverage
            and anchor_proposal_coverage <= self.contained_max_anchor_coverage
        ):
            return "contained_residual"
        return "weak_overlap"

    def _build_anchored_proposal(
        self,
        image_shape: tuple[int, int],
        anchor: Anchor2D,
        relations: list[MaskAnchorRelation],
        *,
        proposal_id: int,
    ) -> Proposal2D:
        anchor_mask = self._bbox_mask(image_shape, np.asarray(anchor.bbox_xyxy, dtype=np.float32))
        mask = np.zeros(image_shape, dtype=bool)
        for relation in relations:
            source_mask = np.asarray(relation.proposal.mask, dtype=bool)
            mask |= source_mask & anchor_mask if self.clip_to_anchor_box else source_mask
        area = int(mask.sum())
        return Proposal2D(
            proposal_id=int(proposal_id),
            mask=mask,
            bbox_xyxy=self._mask_bbox(mask),
            area=area,
            confidence=max(float(anchor.confidence), max(float(item.proposal.confidence) for item in relations)),
            backend_name="anchor_guided_sam",
            metadata={
                "source": "anchor_guided_sam",
                "observation_layer": "fine",
                "anchor_id": int(anchor.anchor_id),
                "anchor_class_name": str(anchor.class_name),
                "anchor_confidence": float(anchor.confidence),
                "anchor_label_strength": "strong",
                "anchor_keepalive": True,
                "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
                "semantic_commit_allowed": True,
                "source_raw_proposal_ids": [int(item.proposal.proposal_id) for item in relations],
                "mask_anchor_relation": "scale_compatible",
                "proposal_anchor_coverage": max(float(item.proposal_anchor_coverage) for item in relations),
                "anchor_proposal_coverage": max(float(item.anchor_proposal_coverage) for item in relations),
                "mask_source": "anchor_guided_sam_selected",
            },
        )

    def _build_anchor_box_fallback(
        self,
        image_shape: tuple[int, int],
        anchor: Anchor2D,
        *,
        proposal_id: int,
    ) -> Proposal2D | None:
        mask = self._bbox_mask(image_shape, np.asarray(anchor.bbox_xyxy, dtype=np.float32))
        area = int(mask.sum())
        if area < self.proposal_min_area:
            return None
        return Proposal2D(
            proposal_id=int(proposal_id),
            mask=mask,
            bbox_xyxy=self._mask_bbox(mask),
            area=area,
            confidence=float(anchor.confidence),
            backend_name="anchor_guided_sam",
            metadata={
                "source": "anchor_guided_sam",
                "observation_layer": "coarse",
                "anchor_id": int(anchor.anchor_id),
                "anchor_class_name": str(anchor.class_name),
                "anchor_confidence": float(anchor.confidence),
                "anchor_label_strength": "strong",
                "anchor_keepalive": True,
                "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
                "semantic_commit_allowed": True,
                "source_raw_proposal_ids": [],
                "mask_anchor_relation": "anchor_box_fallback",
                "mask_source": "detector_anchor_box",
                "force_object_candidate": True,
            },
        )

    def _build_unknown_residual(
        self,
        proposal: Proposal2D,
        relation: MaskAnchorRelation,
        *,
        proposal_id: int,
    ) -> Proposal2D | None:
        if not self.include_unknown_residuals:
            return None
        mask = np.asarray(proposal.mask, dtype=bool).copy()
        area = int(mask.sum())
        if area < self.proposal_min_area:
            return None
        return Proposal2D(
            proposal_id=int(proposal_id),
            mask=mask,
            bbox_xyxy=self._mask_bbox(mask),
            area=area,
            confidence=float(proposal.confidence),
            backend_name="anchor_guided_sam",
            metadata={
                "source": "anchor_guided_sam",
                "observation_layer": "residual",
                "anchor_id": -1,
                "anchor_class_name": "",
                "anchor_confidence": 0.0,
                "anchor_label_strength": "none",
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
                "source_raw_proposal_ids": [int(proposal.proposal_id)],
                "nearby_anchor_id": int(relation.anchor.anchor_id),
                "nearby_anchor_class_name": str(relation.anchor.class_name),
                "mask_anchor_relation": "contained_residual",
                "proposal_anchor_coverage": float(relation.proposal_anchor_coverage),
                "anchor_proposal_coverage": float(relation.anchor_proposal_coverage),
                "mask_source": "sam_unknown_residual",
            },
        )

    @staticmethod
    def _residual_id(anchor: Anchor2D, proposal: Proposal2D) -> int:
        return 1_000_000 + int(anchor.anchor_id) * 10_000 + int(proposal.proposal_id)

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

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
        area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        denom = area_a + area_b - inter
        return 0.0 if denom <= 0.0 else float(inter / denom)

    def _debug(
        self,
        *,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
        output: list[Proposal2D],
        assignments: list[AnchorAssignment],
        matched_sam_ids: set[int],
        dropped_count: int,
    ) -> dict[str, Any]:
        anchored = [proposal for proposal in output if proposal.metadata.get("anchor_label_strength") == "strong"]
        residuals = [proposal for proposal in output if proposal.metadata.get("anchor_label_strength") == "none"]
        candidate_counts = [
            len(proposal.metadata.get("source_raw_proposal_ids", []) or [])
            for proposal in anchored
        ]
        semantic_blocked = [
            proposal for proposal in residuals if proposal.metadata.get("semantic_commit_allowed") is False
        ]
        return {
            "enabled": bool(self.enabled),
            "anchor_count": int(len(anchors)),
            "source_sam_proposal_count": int(len(sam_proposals)),
            "output_proposal_count": int(len(output)),
            "anchored_proposal_count": int(len(anchored)),
            "unknown_residual_count": int(len(residuals)),
            "dropped_proposal_count": int(dropped_count),
            "matched_sam_proposal_count": int(len(matched_sam_ids)),
            "mean_sam_candidates_per_anchor": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            "semantic_blocked_residual_count": int(len(semantic_blocked)),
            "assignment_count": int(len(assignments)),
        }
```

- [ ] **Step 4: Run the strong-match test and verify it passes**

Run:

```bash
pytest tests/test_anchor_guided_sam.py::test_anchor_guided_sam_labels_scale_compatible_mask -v
```

Expected: PASS.

- [ ] **Step 5: Commit the pure module base**

Run:

```bash
git add src/modules/anchor_guided_sam.py tests/test_anchor_guided_sam.py
git commit -m "feat: add anchor-guided SAM proposal module"
```

---

## Task 2: Protect Contained Child Masks From Parent Semantics

**Files:**
- Modify: `tests/test_anchor_guided_sam.py`
- Modify: `src/modules/anchor_guided_sam.py`

- [ ] **Step 1: Add failing test for blanket/rug-inside-sofa residual behavior**

Append to `tests/test_anchor_guided_sam.py`:

```python
def test_anchor_guided_sam_does_not_label_small_contained_mask_as_parent() -> None:
    module = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "min_proposal_anchor_coverage_for_label": 0.70,
            "min_anchor_proposal_coverage_for_label": 0.30,
            "contained_residual_enabled": True,
            "contained_min_proposal_coverage": 0.85,
            "contained_max_anchor_coverage": 0.20,
            "clip_to_anchor_box": True,
            "include_unknown_residuals": True,
            "include_anchor_box_fallbacks": True,
        }
    )

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[_anchor(4, 1, 1, 11, 11, "sofa")],
        sam_proposals=[_proposal(99, 4, 4, 7, 7)],
    )

    residuals = [proposal for proposal in proposals if proposal.metadata["anchor_label_strength"] == "none"]
    anchored = [proposal for proposal in proposals if proposal.metadata["anchor_label_strength"] == "strong"]

    assert len(anchored) == 1
    assert anchored[0].metadata["mask_anchor_relation"] == "anchor_box_fallback"
    assert len(residuals) == 1
    assert residuals[0].metadata["anchor_id"] == -1
    assert residuals[0].metadata["anchor_class_name"] == ""
    assert residuals[0].metadata["anchor_label_strength"] == "none"
    assert residuals[0].metadata["semantic_commit_allowed"] is False
    assert residuals[0].metadata["residual_semantic_policy"] == "unknown"
    assert residuals[0].metadata["nearby_anchor_class_name"] == "sofa"
    assert residuals[0].metadata["source_raw_proposal_ids"] == [99]
    assert all(assignment.class_name != "sofa" for assignment in assignments if assignment.proposal_id == residuals[0].proposal_id)
    assert debug["unknown_residual_count"] == 1
    assert debug["semantic_blocked_residual_count"] == 1
```

- [ ] **Step 2: Run the residual test and verify it fails if Task 1 did not already cover it**

Run:

```bash
pytest tests/test_anchor_guided_sam.py::test_anchor_guided_sam_does_not_label_small_contained_mask_as_parent -v
```

Expected before fixing gaps: FAIL if residual metadata or fallback emission is incomplete.

- [ ] **Step 3: Fix residual and fallback behavior**

If the Task 1 implementation already passes, make no code change in this step. If it fails, update `src/modules/anchor_guided_sam.py` so:

```python
def _classify_relation(self, *, proposal_anchor_coverage: float, anchor_proposal_coverage: float) -> str:
    if (
        proposal_anchor_coverage >= self.min_proposal_anchor_coverage_for_label
        and anchor_proposal_coverage >= self.min_anchor_proposal_coverage_for_label
    ):
        return "scale_compatible"
    if (
        self.contained_residual_enabled
        and proposal_anchor_coverage >= self.contained_min_proposal_coverage
        and anchor_proposal_coverage <= self.contained_max_anchor_coverage
    ):
        return "contained_residual"
    return "weak_overlap"
```

Also ensure `_build_unknown_residual(...)` contains:

```python
"anchor_id": -1,
"anchor_class_name": "",
"anchor_confidence": 0.0,
"anchor_label_strength": "none",
"semantic_commit_allowed": False,
"residual_semantic_policy": "unknown",
"nearby_anchor_id": int(relation.anchor.anchor_id),
"nearby_anchor_class_name": str(relation.anchor.class_name),
```

- [ ] **Step 4: Run the anchor-guided unit tests**

Run:

```bash
pytest tests/test_anchor_guided_sam.py -v
```

Expected: all tests in `tests/test_anchor_guided_sam.py` PASS.

- [ ] **Step 5: Commit contained-residual protection**

Run:

```bash
git add src/modules/anchor_guided_sam.py tests/test_anchor_guided_sam.py
git commit -m "fix: keep contained SAM residuals semantic-unknown"
```

---

## Task 3: Add Anchor-Aware Proposal Backend Interface

**Files:**
- Modify: `src/models/proposal_backend.py`
- Modify: `src/modules/proposal.py`
- Test: `tests/test_anchor_guided_sam.py`

- [ ] **Step 1: Add failing interface test**

Append to `tests/test_anchor_guided_sam.py`:

```python
def test_proposal_module_process_for_anchors_delegates_to_backend() -> None:
    from src.modules.proposal import ProposalModule

    calls: list[tuple[int, int]] = []

    class Backend:
        def generate_proposals_for_anchors(self, rgb, depth, anchors, frame=None):
            calls.append((len(anchors), int(frame.frame_id if frame is not None else -1)))
            return [_proposal(7, 2, 2, 5, 5)]

    module = object.__new__(ProposalModule)
    module.config = {"min_mask_area": 1}
    module.requested_backend_name = "fake"
    module.active_backend_name = "fake"
    module.backend_init_error = None
    module.backend = Backend()

    frame = _frame()
    proposals = module.process_for_anchors(
        frame.rgb,
        frame.depth,
        anchors=[_anchor(1, 1, 1, 6, 6, "chair")],
        frame=frame,
    )

    assert calls == [(1, 1)]
    assert len(proposals) == 1
    assert proposals[0].metadata["requested_backend"] == "fake"
    assert proposals[0].metadata["actual_backend"] == "fake"
```

- [ ] **Step 2: Run the interface test and verify it fails**

Run:

```bash
pytest tests/test_anchor_guided_sam.py::test_proposal_module_process_for_anchors_delegates_to_backend -v
```

Expected: FAIL with `AttributeError: 'ProposalModule' object has no attribute 'process_for_anchors'`.

- [ ] **Step 3: Add default backend method**

Modify `src/models/proposal_backend.py`:

```python
from typing import List

import numpy as np

from src.core.data_structures import Anchor2D, Frame, Proposal2D
```

Add this non-abstract method to `ProposalBackend` under `generate_proposals(...)`:

```python
    def generate_proposals_for_anchors(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        anchors: List[Anchor2D],
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        """Produce proposals for a known anchor set.

        Backends that support box-prompt SAM can override this. Backends that
        replay cached masks or run automatic masks use the normal proposal path.
        """
        return self.generate_proposals(rgb, depth, frame=frame)
```

- [ ] **Step 4: Add module wrapper with shared metadata stamping**

Modify `src/modules/proposal.py` so `process(...)` and `process_for_anchors(...)` share a helper:

```python
    def _finalize_proposals(self, proposals: List[Proposal2D]) -> List[Proposal2D]:
        min_area = self.config.get("min_mask_area", 100)
        proposals = [p for p in proposals if p.area >= min_area]
        for proposal in proposals:
            proposal.metadata = dict(proposal.metadata)
            proposal.metadata["requested_backend"] = self.requested_backend_name
            proposal.metadata["actual_backend"] = self.active_backend_name
            if self.backend_init_error:
                proposal.metadata["backend_fallback_reason"] = self.backend_init_error
        logger.debug("Generated %d proposals (after area filter).", len(proposals))
        return proposals
```

Replace the final block of `process(...)` with:

```python
        proposals = self.backend.generate_proposals(rgb, depth, frame=frame)
        return self._finalize_proposals(proposals)
```

Add:

```python
    def process_for_anchors(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        anchors: List[Anchor2D],
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        """Generate proposals using anchor context when the backend supports it."""
        generate_for_anchors = getattr(self.backend, "generate_proposals_for_anchors", None)
        if callable(generate_for_anchors):
            proposals = generate_for_anchors(rgb, depth, anchors, frame=frame)
        else:
            proposals = self.backend.generate_proposals(rgb, depth, frame=frame)
        return self._finalize_proposals(proposals)
```

Also add `Anchor2D` to the imports in `src/modules/proposal.py`:

```python
from src.core.data_structures import Anchor2D, Frame, Proposal2D
```

- [ ] **Step 5: Run interface and existing proposal tests**

Run:

```bash
pytest tests/test_anchor_guided_sam.py::test_proposal_module_process_for_anchors_delegates_to_backend tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit interface wrapper**

Run:

```bash
git add src/models/proposal_backend.py src/modules/proposal.py tests/test_anchor_guided_sam.py
git commit -m "feat: add anchor-aware proposal backend hook"
```

---

## Task 4: Wire Anchor-Guided Mode Into Pipeline Stage 2

**Files:**
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing pipeline wiring test**

Append to the existing pipeline test class in `tests/test_pipeline.py`:

```python
    def test_pipeline_uses_anchor_guided_sam_in_anchor_primary_mode(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: true
  proposal_min_area: 1
  min_proposal_anchor_coverage_for_label: 0.70
  min_anchor_proposal_coverage_for_label: 0.20
  contained_residual_enabled: true
  contained_min_proposal_coverage: 0.85
  contained_max_anchor_coverage: 0.20
  include_unknown_residuals: true
  include_anchor_box_fallbacks: true
pipeline:
  verbose: false
  collect_stage_timings: true
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
bg_obj_split:
  background_threshold: 0.95
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

        class AnchorPrimary:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            collect_generation_timings = False

            def generate_anchor_box_proposals(self, rgb):
                anchor = Anchor2D(4, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                proposal = Proposal2D(
                    proposal_id=4,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.9,
                    metadata={"anchor_id": 4, "anchor_class_name": "sofa", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(4, 4, "sofa", 0.9, keepalive=True)
                return [anchor], [proposal], [assignment]

        class PrecomputedProposal:
            active_backend_name = "precomputed"

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:7, 1:7] = True
                return [
                    Proposal2D(
                        proposal_id=90,
                        mask=mask,
                        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                        area=int(mask.sum()),
                        confidence=1.0,
                        backend_name="precomputed",
                    )
                ]

        pipe.object_anchor = AnchorPrimary()
        pipe.proposal = PrecomputedProposal()

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        assert pipe.last_frame_debug["proposal_source"] == "anchor_guided_sam"
        assert pipe.last_frame_debug["raw_proposal_count"] == 1
        assert pipe.last_frame_debug["anchor_guided_sam"]["source_sam_proposal_count"] == 1
        assert pipe.last_raw_proposals[0].metadata["source"] == "anchor_guided_sam"
        assert pipe.last_raw_proposals[0].metadata["source_raw_proposal_ids"] == [90]
```

- [ ] **Step 2: Run the pipeline wiring test and verify it fails**

Run:

```bash
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_uses_anchor_guided_sam_in_anchor_primary_mode -v
```

Expected: FAIL because `Pipeline` does not construct or use `AnchorGuidedSAMModule`.

- [ ] **Step 3: Import and construct the module**

Modify `src/pipelines/main_pipeline.py` imports:

```python
from src.modules.anchor_guided_sam import AnchorGuidedSAMModule
```

In `Pipeline.__init__`, after `self.object_anchor = ...` and before proposal-stage use, add:

```python
        self.anchor_guided_sam = AnchorGuidedSAMModule(self.config.get("anchor_guided_sam", {}))
```

- [ ] **Step 4: Replace Stage 2 anchor-primary branch with anchor-guided branch**

In `src/pipelines/main_pipeline.py`, replace:

```python
            if self.object_anchor.enabled and getattr(self.object_anchor, "anchor_primary_mode", False):
                anchors, proposals, anchor_assignments = self.object_anchor.generate_anchor_box_proposals(frame.rgb)
                self._stamp_anchor_primary_coarse_metadata(frame, proposals, anchor_assignments)
                proposal_source = "anchor_box_primary"
```

with:

```python
            anchor_guided_sam_summary: dict[str, Any] = {"enabled": bool(getattr(self.anchor_guided_sam, "enabled", False))}
            if self.object_anchor.enabled and getattr(self.object_anchor, "anchor_primary_mode", False):
                anchors, anchor_box_proposals, anchor_assignments = self.object_anchor.generate_anchor_box_proposals(frame.rgb)
                if getattr(self.anchor_guided_sam, "enabled", False):
                    sam_proposals = self.proposal.process_for_anchors(
                        frame.rgb,
                        frame.depth,
                        anchors,
                        frame=frame,
                    )
                    source_proposals = list(sam_proposals)
                    proposals, anchor_assignments, anchor_guided_sam_summary = self.anchor_guided_sam.build_proposals(
                        frame=frame,
                        anchors=list(anchors),
                        sam_proposals=sam_proposals,
                    )
                    proposal_source = "anchor_guided_sam"
                else:
                    proposals = anchor_box_proposals
                    self._stamp_anchor_primary_coarse_metadata(frame, proposals, anchor_assignments)
                    proposal_source = "anchor_box_primary"
```

If `anchor_guided_sam_summary` must be visible outside the `with` block, initialize it before the block:

```python
        anchor_guided_sam_summary: dict[str, Any] = {"enabled": bool(getattr(self.anchor_guided_sam, "enabled", False))}
```

- [ ] **Step 5: Preserve RuntimeVis metadata for the new source**

In the RuntimeVis metadata preservation block, change:

```python
            if proposal_source == "anchor_box_primary":
```

to:

```python
            if proposal_source in {"anchor_box_primary", "anchor_guided_sam"}:
```

Inside the copied metadata keys tuple, include:

```python
                        "semantic_commit_allowed",
                        "residual_semantic_policy",
                        "mask_anchor_relation",
                        "source_raw_proposal_ids",
```

- [ ] **Step 6: Add debug output**

In `self.last_frame_debug = {...}`, add:

```python
            "anchor_guided_sam": dict(anchor_guided_sam_summary),
```

Keep the existing `"async_refinement"` and `"structural_overlay"` entries unchanged.

- [ ] **Step 7: Run pipeline wiring test**

Run:

```bash
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_uses_anchor_guided_sam_in_anchor_primary_mode -v
```

Expected: PASS.

- [ ] **Step 8: Run focused pipeline tests**

Run:

```bash
pytest tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit pipeline wiring**

Run:

```bash
git add src/pipelines/main_pipeline.py tests/test_pipeline.py
git commit -m "feat: wire anchor-guided SAM into pipeline"
```

---

## Task 5: Add Child-Anchor Override for True Nested Objects

**Files:**
- Modify: `src/modules/anchor_guided_sam.py`
- Modify: `tests/test_anchor_guided_sam.py`

- [ ] **Step 1: Add failing test for child anchor ownership**

Append to `tests/test_anchor_guided_sam.py`:

```python
def test_anchor_guided_sam_labels_contained_mask_when_matching_child_anchor_exists() -> None:
    module = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "min_proposal_anchor_coverage_for_label": 0.70,
            "min_anchor_proposal_coverage_for_label": 0.30,
            "contained_residual_enabled": True,
            "contained_min_proposal_coverage": 0.85,
            "contained_max_anchor_coverage": 0.20,
            "include_unknown_residuals": True,
            "include_anchor_box_fallbacks": True,
        }
    )

    proposals, assignments, debug = module.build_proposals(
        frame=_frame(),
        anchors=[
            _anchor(4, 1, 1, 11, 11, "sofa"),
            _anchor(5, 4, 4, 7, 7, "blanket"),
        ],
        sam_proposals=[_proposal(99, 4, 4, 7, 7)],
    )

    blanket = [proposal for proposal in proposals if proposal.metadata.get("anchor_class_name") == "blanket"]
    residuals = [proposal for proposal in proposals if proposal.metadata.get("anchor_label_strength") == "none"]

    assert len(blanket) == 1
    assert blanket[0].metadata["semantic_commit_allowed"] is True
    assert blanket[0].metadata["source_raw_proposal_ids"] == [99]
    assert blanket[0].metadata["mask_anchor_relation"] == "scale_compatible"
    assert residuals == []
    assert any(assignment.anchor_id == 5 and assignment.class_name == "blanket" for assignment in assignments)
    assert debug["unknown_residual_count"] == 0
```

- [ ] **Step 2: Run the child-anchor test and verify current behavior**

Run:

```bash
pytest tests/test_anchor_guided_sam.py::test_anchor_guided_sam_labels_contained_mask_when_matching_child_anchor_exists -v
```

Expected before ownership fix: FAIL if the parent anchor consumes the SAM mask as an unknown residual before the child anchor can label it.

- [ ] **Step 3: Assign each SAM mask to its best strong anchor before residual creation**

Modify `build_proposals(...)` in `src/modules/anchor_guided_sam.py` to compute strong ownership first:

```python
        relations_by_anchor: dict[int, list[MaskAnchorRelation]] = {
            int(anchor.anchor_id): self._relations_for_anchor(image_shape, anchor, sam_proposals)
            for anchor in anchors
        }
        strong_owner_by_sam: dict[int, int] = {}
        for anchor in anchors:
            for relation in relations_by_anchor.get(int(anchor.anchor_id), []):
                if relation.relation != "scale_compatible":
                    continue
                sam_id = int(relation.proposal.proposal_id)
                current_anchor_id = strong_owner_by_sam.get(sam_id)
                if current_anchor_id is None:
                    strong_owner_by_sam[sam_id] = int(anchor.anchor_id)
                    continue
                current_relation = next(
                    item
                    for item in relations_by_anchor[current_anchor_id]
                    if int(item.proposal.proposal_id) == sam_id
                )
                if (
                    relation.anchor_proposal_coverage,
                    relation.proposal_anchor_coverage,
                    -int(anchor.anchor_id),
                ) > (
                    current_relation.anchor_proposal_coverage,
                    current_relation.proposal_anchor_coverage,
                    -int(current_anchor_id),
                ):
                    strong_owner_by_sam[sam_id] = int(anchor.anchor_id)
```

Then replace each call to `_relations_for_anchor(...)` inside the anchor loop with:

```python
            relations = relations_by_anchor.get(int(anchor.anchor_id), [])
```

Filter strong relations for the current anchor:

```python
            strong = [
                relation
                for relation in relations
                if relation.relation == "scale_compatible"
                and strong_owner_by_sam.get(int(relation.proposal.proposal_id)) == int(anchor.anchor_id)
            ]
```

When creating contained residuals, skip SAM masks with any strong owner:

```python
                if int(relation.proposal.proposal_id) in strong_owner_by_sam:
                    continue
```

- [ ] **Step 4: Run anchor-guided tests**

Run:

```bash
pytest tests/test_anchor_guided_sam.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit child-anchor ownership**

Run:

```bash
git add src/modules/anchor_guided_sam.py tests/test_anchor_guided_sam.py
git commit -m "fix: prefer matching child anchors for contained SAM masks"
```

---

## Task 6: Add Room0 Anchor-Guided Config

**Files:**
- Create: `configs/room0_anchor_guided_sam_4090.yaml`

- [ ] **Step 1: Copy the fast hybrid config**

Run:

```bash
cp configs/room0_fast_hybrid_4090.yaml configs/room0_anchor_guided_sam_4090.yaml
```

- [ ] **Step 2: Add anchor-guided block and disable structural overlay**

Edit `configs/room0_anchor_guided_sam_4090.yaml` so these keys are present:

```yaml
proposal:
  backend: precomputed
  min_mask_area: 100
  max_proposals: 50
  precomputed:
    cache_dir: /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2
    manifest_path: /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json
    strict: true

anchor_frontend:
  enabled: true
  use_boxes_as_primary_proposals: true
  use_sam_intersection_proposals: false
  anchor_primary_mode: true
  unanchored_fallback_enabled: false
  proposal_min_area: 25
  contained_subproposal_gate_enabled: true
  contained_max_anchor_coverage: 0.20
  contained_min_proposal_coverage: 0.85

anchor_guided_sam:
  enabled: true
  proposal_min_area: 25
  min_proposal_anchor_coverage_for_label: 0.70
  min_anchor_proposal_coverage_for_label: 0.20
  contained_residual_enabled: true
  contained_min_proposal_coverage: 0.85
  contained_max_anchor_coverage: 0.35
  clip_to_anchor_box: true
  include_unknown_residuals: true
  include_anchor_box_fallbacks: true
  include_unanchored_residuals: false
  max_sam_proposals_per_anchor: 4
  structure_classes:
  - wall
  - window
  - blinds
  - ceiling
  - floor
  - door
  object_classes:
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

structural_overlay:
  enabled: false
```

Preserve the current YOLOWorld and YOLOE supplemental class list from `configs/room0_fast_hybrid_4090.yaml`.

- [ ] **Step 3: Validate YAML load**

Run:

```bash
python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/room0_anchor_guided_sam_4090.yaml").read_text())
assert cfg["proposal"]["backend"] == "precomputed"
assert cfg["anchor_frontend"]["anchor_primary_mode"] is True
assert cfg["anchor_guided_sam"]["enabled"] is True
assert cfg.get("structural_overlay", {}).get("enabled") is False
print("room0_anchor_guided_sam_4090.yaml ok")
PY
```

Expected output:

```text
room0_anchor_guided_sam_4090.yaml ok
```

- [ ] **Step 4: Commit config**

Run:

```bash
git add configs/room0_anchor_guided_sam_4090.yaml
git commit -m "config: add room0 anchor-guided SAM setup"
```

---

## Task 7: Add Reportable Debug Metrics

**Files:**
- Modify: `src/pipelines/main_pipeline.py`
- Modify: any report builder that already serializes `last_frame_debug`, if present
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Locate report builder**

Run:

```bash
rg -n "build_frontend_stage_report_payload|last_frame_debug|stage_timings|anchor_guided" .
```

Expected: find `src/pipelines/main_pipeline.py`; if a report helper exists, note the exact path from the search output.

- [ ] **Step 2: Add failing assertion to the pipeline wiring test**

In `tests/test_pipeline.py::TestPipeline::test_pipeline_uses_anchor_guided_sam_in_anchor_primary_mode`, add:

```python
        expected_debug_keys = {
            "source_sam_proposal_count",
            "anchored_proposal_count",
            "unknown_residual_count",
            "dropped_proposal_count",
            "mean_sam_candidates_per_anchor",
            "semantic_blocked_residual_count",
        }
        assert expected_debug_keys.issubset(pipe.last_frame_debug["anchor_guided_sam"])
```

- [ ] **Step 3: Run the debug-key test**

Run:

```bash
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_uses_anchor_guided_sam_in_anchor_primary_mode -v
```

Expected: FAIL if `last_frame_debug["anchor_guided_sam"]` is absent or missing one of the required keys.

- [ ] **Step 4: Ensure debug keys are emitted even when disabled**

Modify the disabled summary initialization in `src/pipelines/main_pipeline.py` to:

```python
        anchor_guided_sam_summary: dict[str, Any] = {
            "enabled": bool(getattr(self.anchor_guided_sam, "enabled", False)),
            "anchor_count": 0,
            "source_sam_proposal_count": 0,
            "output_proposal_count": 0,
            "anchored_proposal_count": 0,
            "unknown_residual_count": 0,
            "dropped_proposal_count": 0,
            "matched_sam_proposal_count": 0,
            "mean_sam_candidates_per_anchor": 0.0,
            "semantic_blocked_residual_count": 0,
            "assignment_count": 0,
        }
```

The enabled path should replace this with the module debug dictionary.

- [ ] **Step 5: Add report payload pass-through only if a separate report builder exists**

If Step 1 finds a helper named `build_frontend_stage_report_payload`, add this key to its payload:

```python
"anchor_guided_sam": dict(last_frame_debug.get("anchor_guided_sam", {})),
```

If no separate report helper exists, keep the change limited to `Pipeline.last_frame_debug`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_anchor_guided_sam.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit metrics**

Run:

```bash
git add src/pipelines/main_pipeline.py tests/test_pipeline.py
git commit -m "chore: report anchor-guided SAM metrics"
```

---

## Task 8: Run Smoke Test and 200f Experiment

**Files:**
- Read: `run_room0_full_eval.py`
- Read: `configs/room0_anchor_guided_sam_4090.yaml`
- Output: `outputs/tmp_validation/20260530_room0_anchor_guided_sam_stride10_20f`
- Output: `outputs/tmp_validation/20260530_room0_anchor_guided_sam_stride10_200f`

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest tests/test_anchor_guided_sam.py tests/test_async_refinement.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 2: Start 20f stride=10 smoke**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_anchor_guided_sam_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260530_room0_anchor_guided_sam_stride10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected:

```text
evaluation completed
```

The exact final line may differ; the command must exit with status 0 and create `outputs/tmp_validation/20260530_room0_anchor_guided_sam_stride10_20f`.

- [ ] **Step 3: Inspect smoke metrics**

Run:

```bash
find outputs/tmp_validation/20260530_room0_anchor_guided_sam_stride10_20f -maxdepth 3 -type f | sort
```

Then inspect the metrics file printed by the command. If the runner writes `metrics.json`, run:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/tmp_validation/20260530_room0_anchor_guided_sam_stride10_20f")
metrics = sorted(root.rglob("*metrics*.json"))
assert metrics, "no metrics json found"
data = json.loads(metrics[0].read_text())
print(metrics[0])
print(json.dumps(data, indent=2, sort_keys=True)[:4000])
PY
```

Expected: a readable metrics summary with per-class IoU and stage timings.

- [ ] **Step 4: Compare smoke against the previous fast hybrid baseline**

Use the existing best 0526/fast-hybrid 200f or stride10 run as the reference available under `outputs/tmp_validation`. Record:

```text
chair IoU
table IoU
sofa IoU
rug IoU
wall IoU
window IoU
blinds IoU
mean proposal_generation time
mean depth_refinement time
mean total frame time
```

Proceed to 200f only if:

```text
chair/table/sofa/rug each drop by no more than 0.05 absolute from the selected fast-hybrid reference
wall/blinds/window do not collapse to zero
window improves above the previous anchor-box-only value when visible
mean frame time is closer to fast hybrid than to full SAM automatic proposal flow
```

- [ ] **Step 5: Start 200f stride=10 experiment**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_anchor_guided_sam_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260530_room0_anchor_guided_sam_stride10_200f \
  --num-frames 200 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected: command exits with status 0 and creates `outputs/tmp_validation/20260530_room0_anchor_guided_sam_stride10_200f/room0/exports`.

- [ ] **Step 6: Commit experiment notes**

Create or update a dated notes file only after the experiment finishes:

```bash
git add docs/superpowers/plans/2026-05-30-anchor-guided-sam-proposal-graph.md
git commit -m "docs: record anchor-guided SAM validation results"
```

If no notes are added to the plan document, skip this commit.

---

## Task 9: Add Real Promptable SAM Backend After Precomputed Validation

**Files:**
- Modify: `src/models/sam2_proposal_backend.py`
- Modify: `src/models/proposal_backend.py`
- Test: `tests/test_anchor_guided_sam.py`

This task runs only after Task 8 shows that the proposal graph behavior improves accuracy or preserves object IoU while reducing proposal volume.

- [ ] **Step 1: Add a backend-level contract test with a fake backend**

Append to `tests/test_anchor_guided_sam.py`:

```python
def test_backend_anchor_prompt_contract_returns_anchor_local_masks() -> None:
    from src.models.proposal_backend import ProposalBackend

    class FakePromptBackend(ProposalBackend):
        def initialize(self, config):
            self.initialized = True

        def generate_proposals(self, rgb, depth, frame=None):
            raise AssertionError("anchor prompt path should not call automatic generation")

        def generate_proposals_for_anchors(self, rgb, depth, anchors, frame=None):
            return [_proposal(int(anchor.anchor_id), 1, 1, 4, 4) for anchor in anchors]

    backend = FakePromptBackend()
    backend.initialize({})
    proposals = backend.generate_proposals_for_anchors(
        np.zeros((12, 12, 3), dtype=np.uint8),
        np.ones((12, 12), dtype=np.float32),
        [_anchor(2, 1, 1, 4, 4, "chair")],
        frame=_frame(),
    )

    assert len(proposals) == 1
    assert proposals[0].proposal_id == 2
```

- [ ] **Step 2: Run the contract test**

Run:

```bash
pytest tests/test_anchor_guided_sam.py::test_backend_anchor_prompt_contract_returns_anchor_local_masks -v
```

Expected: PASS once Task 3 has added the optional backend method.

- [ ] **Step 3: Add SAM2 box-prompt method**

Modify `src/models/sam2_proposal_backend.py` to implement:

```python
    def generate_proposals_for_anchors(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        anchors: list[Anchor2D],
        frame: Frame | None = None,
    ) -> list[Proposal2D]:
        """Run SAM2 with detector boxes as prompts and return one or more masks per anchor."""
        proposals: list[Proposal2D] = []
        for anchor in anchors:
            masks, scores = self._predict_masks_for_box(rgb, np.asarray(anchor.bbox_xyxy, dtype=np.float32))
            for mask, score in zip(masks, scores):
                mask_bool = np.asarray(mask, dtype=bool)
                area = int(mask_bool.sum())
                if area <= 0:
                    continue
                proposals.append(
                    Proposal2D(
                        proposal_id=len(proposals),
                        mask=mask_bool,
                        bbox_xyxy=self._mask_bbox(mask_bool),
                        area=area,
                        confidence=float(score),
                        backend_name="sam2_box_prompt",
                        metadata={
                            "source": "sam2_box_prompt",
                            "prompt_anchor_id": int(anchor.anchor_id),
                            "prompt_anchor_class_name": str(anchor.class_name),
                        },
                    )
                )
        return proposals
```

If the current SAM2 wrapper does not expose a predictor object that supports box prompts, add `_predict_masks_for_box(...)` around the existing loaded model API in the same file. The method must return a list of boolean masks and float scores.

- [ ] **Step 4: Add a guarded smoke command for real prompted SAM**

Use a separate config name such as `configs/room0_anchor_prompted_sam2_4090.yaml` and set:

```yaml
proposal:
  backend: sam2
anchor_guided_sam:
  enabled: true
```

Run a 5f smoke before 20f:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_anchor_prompted_sam2_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name 20260530_room0_anchor_prompted_sam2_stride10_5f \
  --num-frames 5 \
  --frame-stride 10 \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --fast-eval
```

Expected: status 0, proposal-generation time lower than automatic full-image SAM for the same 5 frames.

- [ ] **Step 5: Commit prompt backend**

Run:

```bash
git add src/models/sam2_proposal_backend.py configs/room0_anchor_prompted_sam2_4090.yaml tests/test_anchor_guided_sam.py
git commit -m "feat: add SAM2 anchor-prompt proposal path"
```

---

## Success Criteria

The implementation is acceptable when:

1. `pytest tests/test_anchor_guided_sam.py tests/test_async_refinement.py tests/test_pipeline.py -q` passes.
2. `outputs/tmp_validation/20260530_room0_anchor_guided_sam_stride10_20f` completes with status 0.
3. A 200f stride=10 room0 experiment completes with exports.
4. Sofa/chair/table/rug IoU does not collapse relative to the accepted 0526 fast baseline.
5. Wall/window/blinds are not produced by overwriting object-neighborhood proposals.
6. Residual SAM masks inside parent anchors carry `semantic_commit_allowed: false`.
7. Runtime is closer to anchor-box fast hybrid than to full automatic SAM proposal flow.

---

## Design Rationale

This design keeps the 0526 version's main advantage: detector anchors provide stable object identity and class priors. It adds SAM only where SAM helps most, which is local boundary recovery around a detected anchor. The critical change is that overlap is no longer treated as semantic permission. A SAM mask inside a sofa anchor is not a sofa unless it is scale-compatible with that sofa anchor; otherwise it becomes unknown residual evidence and must earn semantics through a child anchor or multi-frame 3D posterior.

This also matches the DualMap-style speed intuition. The detector path gives a small set of candidate boxes quickly. SAM work is then bounded by anchor count or by selected cached masks around anchors, rather than by blindly admitting every full-image proposal into mapping.
