# Semantic Vote Weak Structure Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Keep strict whole-proposal anchor semantics for object proposals while recovering wall/floor/ceiling-style structure coverage through local clipped weak proposals.

**Architecture:** `ObjectAnchorModule` keeps SAM masks unchanged for strong semantic vote proposals. Partial object anchors remain non-assigning for the whole SAM proposal, but partial structure anchors can produce additional `SAM ∩ anchor_box` clipped proposals with weak structure labels. Downstream modules continue consuming `Proposal2D`; weak clipped proposals are distinguished by metadata and are generated only when explicitly enabled.

**Tech Stack:** Python, NumPy masks, pytest, existing OVIOVO `Proposal2D` / `Anchor2D` / `AnchorAssignment` data structures.

---

## File Structure

- Modify `src/modules/object_anchor.py`
  - Add config for weak structure overlap proposal generation.
  - Split semantic-vote candidate assignment into strong whole-mask assignment and weak clipped structure proposal generation.
  - Add helper methods with narrow responsibilities:
    - `_is_weak_structure_class(class_name)`
    - `_semantic_vote_overlap_metrics(proposal, anchor)`
    - `_weak_structure_overlap_proposal(proposal, anchor, clipped_mask, metrics, weak_index)`
    - `_generate_weak_structure_overlap_proposals(proposal, anchors, existing_assignments)`
- Modify `tests/test_object_anchor.py`
  - Add regression tests for weak structure clipped proposals.
  - Keep existing object contamination regression intact.
- Modify `configs/room0_surface_gate_4090.yaml`
  - Enable weak structure overlap proposals for room0 validation.
  - Declare the class list and coverage thresholds explicitly.
- Optional modify `configs/default.yaml`
  - Add disabled defaults so the config surface is documented without changing other experiments.
- Optional modify `configs/midrecall_local_memory_boost.yaml`
  - Add the same disabled defaults if this config is used for parity experiments.
- Optional modify `run_room0_full_eval.py`
  - Serialize `anchor_label_strength` and `mask_source` in local-memory audit proposal stage records if missing from existing output.

## Behavior Contract

- A SAM proposal gets a whole-proposal strong semantic label only when the anchor covers enough of that SAM proposal:
  - `proposal_coverage >= covering_min_proposal_coverage`
- A small object anchor contained in a large SAM proposal must not label the whole SAM proposal.
- A partial structure anchor may create a new local weak proposal whose mask is exactly `SAM mask ∩ anchor bbox mask`.
- Weak clipped structure proposals must not replace or relabel the original SAM proposal.
- Weak clipped proposals must carry metadata:
  - `anchor_label_strength: "weak_overlap"`
  - `source: "sam2_anchor_vote"`
  - `geometry_source: "sam2_anchor_overlap"`
  - `mask_source: "weak_structure_anchor_box_clip"`
  - `source_raw_proposal_id`
  - `source_raw_proposal_ids`
  - `anchor_id`
  - `anchor_class_name`
  - `anchor_confidence`
  - `anchor_proposal_coverage`
  - `anchor_anchor_coverage`
- Strong whole-mask proposals must carry `anchor_label_strength: "strong"` when assigned and `"none"` when unassigned.

## Task 1: Add Weak Structure Config Defaults

**Files:**
- Modify: `src/modules/object_anchor.py`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Optional modify: `configs/default.yaml`
- Optional modify: `configs/midrecall_local_memory_boost.yaml`
- Test: `tests/test_object_anchor.py`

- [x] **Step 1: Write failing config/default assertions**

Add this test near the existing semantic-vote tests in `tests/test_object_anchor.py`:

```python
def test_object_anchor_weak_structure_overlap_config_defaults() -> None:
    module = ObjectAnchorModule({"enabled": False, "assignment_policy": "semantic_vote"})

    assert module.weak_structure_overlap_enabled is False
    assert module.weak_structure_classes == {"wall", "floor", "ceiling", "blinds", "window"}
    assert module.weak_structure_min_proposal_coverage == 0.05
    assert module.weak_structure_min_anchor_coverage == 0.20
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_object_anchor_weak_structure_overlap_config_defaults -q
```

Expected: FAIL with `AttributeError` for `weak_structure_overlap_enabled`.

- [x] **Step 3: Implement config parsing**

In `src/modules/object_anchor.py`, add these fields in `ObjectAnchorModule.__init__` after `covering_min_proposal_coverage`:

```python
        self.weak_structure_overlap_enabled = bool(config.get("weak_structure_overlap_enabled", False))
        self.weak_structure_classes = {
            str(item).strip()
            for item in config.get(
                "weak_structure_classes",
                ["wall", "floor", "ceiling", "blinds", "window"],
            )
            if str(item).strip()
        }
        self.weak_structure_min_proposal_coverage = float(
            np.clip(config.get("weak_structure_min_proposal_coverage", 0.05), 0.0, 1.0)
        )
        self.weak_structure_min_anchor_coverage = float(
            np.clip(config.get("weak_structure_min_anchor_coverage", 0.20), 0.0, 1.0)
        )
```

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_object_anchor_weak_structure_overlap_config_defaults -q
```

Expected: PASS.

- [x] **Step 5: Add room0 config values**

In `configs/room0_surface_gate_4090.yaml`, under `anchor_frontend`, after `covering_min_proposal_coverage` if present or after `proposal_min_area`, add:

```yaml
  weak_structure_overlap_enabled: true
  weak_structure_classes:
  - wall
  - floor
  - ceiling
  - blinds
  - window
  weak_structure_min_proposal_coverage: 0.05
  weak_structure_min_anchor_coverage: 0.20
```

If `covering_min_proposal_coverage` is not explicit in this file, add:

```yaml
  covering_min_proposal_coverage: 0.85
```

- [x] **Step 6: Optionally document disabled defaults in shared configs**

In `configs/default.yaml` and `configs/midrecall_local_memory_boost.yaml`, add the same keys with:

```yaml
  weak_structure_overlap_enabled: false
```

Keep the class list and thresholds identical to room0 if the surrounding config style favors explicit defaults.

- [x] **Step 7: Run focused tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py -q
```

Expected: all tests in `tests/test_object_anchor.py` pass.

## Task 2: Mark Strong/None Semantic Vote Label Strength

**Files:**
- Modify: `src/modules/object_anchor.py`
- Test: `tests/test_object_anchor.py`

- [x] **Step 1: Write failing metadata test**

Add this assertion to `test_anchor_vote_proposals_keep_sam_mask_pixels_unchanged` after the existing `anchor_class_name` assertions:

```python
    assert voted[0].metadata["anchor_label_strength"] == "strong"
```

Add this assertion to `test_anchor_vote_does_not_label_large_sam_mask_from_small_contained_anchor` after `anchor_class_name == ""`:

```python
    assert voted[0].metadata["anchor_label_strength"] == "none"
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py -k "keep_sam_mask_pixels_unchanged or small_contained_anchor" -q
```

Expected: FAIL with `KeyError: 'anchor_label_strength'`.

- [x] **Step 3: Implement metadata**

In `_clone_proposal_with_anchor_vote`, add `anchor_label_strength` in both branches.

In the `assignment is None` metadata update:

```python
                    "anchor_label_strength": "none",
```

In the assigned metadata update:

```python
                    "anchor_label_strength": "strong",
```

- [x] **Step 4: Run tests to verify pass**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py -k "keep_sam_mask_pixels_unchanged or small_contained_anchor" -q
```

Expected: PASS.

## Task 3: Add Weak Structure Clipped Proposal Generation

**Files:**
- Modify: `src/modules/object_anchor.py`
- Test: `tests/test_object_anchor.py`

- [x] **Step 1: Write failing test for structure clipped proposal**

Add this test after `test_anchor_vote_does_not_label_large_sam_mask_from_small_contained_anchor`:

```python
def test_anchor_vote_adds_weak_clipped_structure_proposal_for_partial_structure_anchor() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.85,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=3,
                    bbox_xyxy=np.array([5, 5, 15, 15], dtype=np.float32),
                    class_name="wall",
                    confidence=0.8,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.ones((20, 20), dtype=bool)
    proposal = Proposal2D(
        proposal_id=9,
        mask=mask.copy(),
        bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.95,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 2
    whole = next(item for item in voted if item.proposal_id == 9)
    weak = next(item for item in voted if item.proposal_id != 9)

    np.testing.assert_array_equal(whole.mask, mask)
    assert whole.metadata["anchor_id"] == -1
    assert whole.metadata["anchor_class_name"] == ""
    assert whole.metadata["anchor_label_strength"] == "none"

    expected_clip = np.zeros((20, 20), dtype=bool)
    expected_clip[5:15, 5:15] = True
    np.testing.assert_array_equal(weak.mask, expected_clip)
    assert weak.bbox_xyxy.tolist() == [5.0, 5.0, 15.0, 15.0]
    assert weak.area == 100
    assert weak.backend_name == "sam2_anchor_vote"
    assert weak.metadata["anchor_id"] == 3
    assert weak.metadata["anchor_class_name"] == "wall"
    assert weak.metadata["anchor_label_strength"] == "weak_overlap"
    assert weak.metadata["geometry_source"] == "sam2_anchor_overlap"
    assert weak.metadata["mask_source"] == "weak_structure_anchor_box_clip"
    assert weak.metadata["source_raw_proposal_id"] == 9
    assert weak.metadata["source_raw_proposal_ids"] == [9]
    assert weak.metadata["anchor_proposal_coverage"] == 0.25
    assert weak.metadata["anchor_anchor_coverage"] == 1.0

    assert len(assignments) == 2
    assert any(int(item.anchor_id) == -1 for item in assignments)
    assert any(int(item.anchor_id) == 3 and item.class_name == "wall" for item in assignments)
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_anchor_vote_adds_weak_clipped_structure_proposal_for_partial_structure_anchor -q
```

Expected: FAIL because only one proposal is returned.

- [x] **Step 3: Add overlap metrics helper**

In `src/modules/object_anchor.py`, add this helper above `_assign_semantic_vote_candidates`:

```python
    def _semantic_vote_overlap_metrics(
        self,
        proposal: Proposal2D,
        anchor: Anchor2D,
    ) -> dict[str, Any]:
        proposal_bbox = np.asarray(proposal.bbox_xyxy, dtype=np.float32)
        proposal_mask = np.asarray(proposal.mask, dtype=bool)
        proposal_area = max(int(proposal_mask.sum()), 1)
        anchor_box = np.asarray(anchor.bbox_xyxy, dtype=np.float32)
        box_mask = self._bbox_mask(proposal_mask.shape, anchor_box)
        clipped_mask = proposal_mask & box_mask
        overlap_area = int(clipped_mask.sum())
        anchor_area_px = max(int(box_mask.sum()), 1)
        center = np.asarray(
            [(proposal_bbox[0] + proposal_bbox[2]) * 0.5, (proposal_bbox[1] + proposal_bbox[3]) * 0.5],
            dtype=np.float32,
        )
        center_inside = self._center_inside(center, anchor_box) if self.center_assign_enabled else False
        bbox_iou = self._bbox_iou(proposal_bbox, anchor_box)
        return {
            "anchor_box": anchor_box,
            "box_mask": box_mask,
            "clipped_mask": clipped_mask,
            "overlap_area": overlap_area,
            "anchor_coverage": overlap_area / anchor_area_px,
            "proposal_coverage": overlap_area / proposal_area,
            "center_inside": bool(center_inside),
            "bbox_iou": float(bbox_iou),
        }
```

- [x] **Step 4: Refactor `_assign_semantic_vote_candidates` to use helper**

Replace the duplicated overlap computation inside `_assign_semantic_vote_candidates` with:

```python
        for anchor in anchors:
            metrics = self._semantic_vote_overlap_metrics(proposal, anchor)
            overlap_area = int(metrics["overlap_area"])
            if overlap_area < self.proposal_min_area:
                continue

            anchor_coverage = float(metrics["anchor_coverage"])
            proposal_coverage = float(metrics["proposal_coverage"])
            center_inside = bool(metrics["center_inside"])
            bbox_iou = float(metrics["bbox_iou"])
            assignable = proposal_coverage >= self.covering_min_proposal_coverage
            if not assignable:
                continue
```

Keep the existing `keepalive` and `AnchorAssignment` creation unchanged.

- [x] **Step 5: Add structure helpers**

Add these methods below `_assign_semantic_vote_candidates`:

```python
    def _is_weak_structure_class(self, class_name: str) -> bool:
        return str(class_name).strip() in self.weak_structure_classes

    def _weak_structure_overlap_proposal(
        self,
        proposal: Proposal2D,
        anchor: Anchor2D,
        clipped_mask: np.ndarray,
        metrics: dict[str, Any],
        weak_index: int,
    ) -> tuple[Proposal2D, AnchorAssignment] | None:
        clipped_mask = np.asarray(clipped_mask, dtype=bool)
        area = int(clipped_mask.sum())
        if area < self.proposal_min_area:
            return None

        bbox_iou = float(metrics["bbox_iou"])
        center_inside = bool(metrics["center_inside"])
        anchor_coverage = float(metrics["anchor_coverage"])
        keepalive = bool(
            bbox_iou >= self.keepalive_iou_threshold
            or (self.keepalive_center_inside and center_inside)
            or anchor_coverage >= self.min_anchor_mask_coverage
        )
        assignment = AnchorAssignment(
            proposal_id=-(int(proposal.proposal_id) * 1000 + int(anchor.anchor_id) + weak_index + 1),
            anchor_id=int(anchor.anchor_id),
            class_name=str(anchor.class_name),
            confidence=float(anchor.confidence),
            bbox_iou=bbox_iou,
            center_inside=center_inside,
            keepalive=keepalive,
        )
        metadata = {
            "source": "sam2_anchor_vote",
            "geometry_source": "sam2_anchor_overlap",
            "mask_source": "weak_structure_anchor_box_clip",
            "source_raw_proposal_id": int(proposal.proposal_id),
            "source_raw_proposal_ids": [int(proposal.proposal_id)],
            "anchor_id": int(anchor.anchor_id),
            "anchor_class_name": str(anchor.class_name),
            "anchor_confidence": float(anchor.confidence),
            "anchor_bbox_iou": bbox_iou,
            "anchor_center_inside": center_inside,
            "anchor_keepalive": keepalive,
            "anchor_vote_score": float(self._assignment_score(assignment)),
            "anchor_candidate_classes": [str(anchor.class_name)],
            "anchor_candidate_ids": [int(anchor.anchor_id)],
            "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
            "anchor_label_strength": "weak_overlap",
            "anchor_proposal_coverage": float(metrics["proposal_coverage"]),
            "anchor_anchor_coverage": anchor_coverage,
        }
        weak_proposal = Proposal2D(
            proposal_id=int(assignment.proposal_id),
            mask=clipped_mask.copy(),
            bbox_xyxy=self._mask_bbox(clipped_mask),
            area=area,
            confidence=max(float(proposal.confidence), float(anchor.confidence)),
            backend_name="sam2_anchor_vote",
            metadata=metadata,
        )
        return weak_proposal, assignment
```

- [x] **Step 6: Add weak proposal generator**

Add this method below `_weak_structure_overlap_proposal`:

```python
    def _generate_weak_structure_overlap_proposals(
        self,
        proposal: Proposal2D,
        anchors: list[Anchor2D],
        existing_assignments: list[AnchorAssignment],
    ) -> tuple[list[Proposal2D], list[AnchorAssignment]]:
        if not self.weak_structure_overlap_enabled:
            return [], []

        assigned_anchor_ids = {int(assignment.anchor_id) for assignment in existing_assignments}
        weak_proposals: list[Proposal2D] = []
        weak_assignments: list[AnchorAssignment] = []
        for anchor in anchors:
            if int(anchor.anchor_id) in assigned_anchor_ids:
                continue
            if not self._is_weak_structure_class(str(anchor.class_name)):
                continue
            metrics = self._semantic_vote_overlap_metrics(proposal, anchor)
            if int(metrics["overlap_area"]) < self.proposal_min_area:
                continue
            if float(metrics["proposal_coverage"]) >= self.covering_min_proposal_coverage:
                continue
            if float(metrics["proposal_coverage"]) < self.weak_structure_min_proposal_coverage:
                continue
            if float(metrics["anchor_coverage"]) < self.weak_structure_min_anchor_coverage:
                continue
            result = self._weak_structure_overlap_proposal(
                proposal,
                anchor,
                np.asarray(metrics["clipped_mask"], dtype=bool),
                metrics,
                weak_index=len(weak_proposals),
            )
            if result is None:
                continue
            weak_proposal, weak_assignment = result
            weak_proposals.append(weak_proposal)
            weak_assignments.append(weak_assignment)

        return weak_proposals, weak_assignments
```

- [x] **Step 7: Wire weak proposals into semantic vote generation**

Modify `_generate_semantic_vote_proposals` so each source proposal always appends the whole SAM-vote proposal first, then appends weak structure clipped proposals.

Replace the method body loop with:

```python
        for proposal in proposals:
            assignments = self._assign_semantic_vote_candidates(proposal, self.last_anchors)
            if not assignments:
                voted_proposals.append(self._clone_proposal_with_anchor_vote(proposal, None, []))
                voted_assignments.append(AnchorAssignment(proposal_id=int(proposal.proposal_id)))
            else:
                best_assignment = max(assignments, key=self._assignment_score)
                voted_proposals.append(
                    self._clone_proposal_with_anchor_vote(
                        proposal,
                        best_assignment,
                        assignments,
                    )
                )
                voted_assignments.append(best_assignment)

            weak_proposals, weak_assignments = self._generate_weak_structure_overlap_proposals(
                proposal,
                self.last_anchors,
                assignments,
            )
            voted_proposals.extend(weak_proposals)
            voted_assignments.extend(weak_assignments)
```

- [x] **Step 8: Run structure clipped proposal test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_anchor_vote_adds_weak_clipped_structure_proposal_for_partial_structure_anchor -q
```

Expected: PASS.

## Task 4: Prevent Weak Object Overlap Proposals

**Files:**
- Modify: `tests/test_object_anchor.py`
- Verify: `src/modules/object_anchor.py`

- [x] **Step 1: Extend object contamination regression**

Update `test_anchor_vote_does_not_label_large_sam_mask_from_small_contained_anchor` config to include:

```python
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
```

Add this assertion after `generate_proposals`:

```python
    assert len(voted) == 1
```

This ensures a partial `sofa` anchor neither labels the whole mask nor creates a weak clipped object proposal.

- [x] **Step 2: Run test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_anchor_vote_does_not_label_large_sam_mask_from_small_contained_anchor -q
```

Expected: PASS.

## Task 5: Avoid Duplicate Weak Proposal When Strong Assignment Exists

**Files:**
- Modify: `tests/test_object_anchor.py`
- Verify: `src/modules/object_anchor.py`

- [x] **Step 1: Add test**

Add this test after the weak clipped structure test:

```python
def test_anchor_vote_does_not_duplicate_weak_structure_when_strong_assignment_exists() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.8,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=2,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    class_name="ceiling",
                    confidence=0.7,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True
    proposal = Proposal2D(
        proposal_id=4,
        mask=mask.copy(),
        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    assert voted[0].metadata["anchor_class_name"] == "ceiling"
    assert voted[0].metadata["anchor_label_strength"] == "strong"
```

- [x] **Step 2: Run test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_anchor_vote_does_not_duplicate_weak_structure_when_strong_assignment_exists -q
```

Expected: PASS if Task 3 skipped assigned anchors correctly.

## Task 6: Serialize Weak Label Metadata in Audit

**Files:**
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_pipeline.py` if an existing local-memory audit serializer test is easy to extend; otherwise verify with a small direct command.

- [x] **Step 1: Extend serializer output**

In `run_room0_full_eval.py`, update `serialize_proposal_stage_record` return dict with:

```python
        "anchor_label_strength": str(metadata.get("anchor_label_strength", "")),
        "mask_source": str(metadata.get("mask_source", "")),
        "anchor_proposal_coverage": float(metadata.get("anchor_proposal_coverage", 0.0)),
        "anchor_anchor_coverage": float(metadata.get("anchor_anchor_coverage", 0.0)),
```

- [x] **Step 2: Run relevant tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py -k "audit or anchor_voted" -q
```

Expected: PASS.

If no tests are selected, run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipelineDebugOutputs::test_local_memory_audit_records_anchor_voted_proposals -q
```

If that test name does not exist, use:

```bash
rg -n "anchor_voted_proposals|serialize_proposal_stage_record|local_memory_audit" tests/test_pipeline.py
```

Then run the matching test shown by `rg`.

## Task 7: Full Focused Test Pass

**Files:**
- Verify only.

- [x] **Step 1: Run object anchor tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py -q
```

Expected: all tests pass.

- [x] **Step 2: Run runtime grouping tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_runtime_vis.py -q
```

Expected: all tests pass. Weak clipped proposals are still normal `Proposal2D` records, so grouping should not crash.

- [x] **Step 3: Run pipeline/audit tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py -k "semantic_vote or anchor_voted or local_memory_audit" -q
```

Expected: all selected tests pass.

## Task 8: Validation Experiments

**Files:**
- Output only under `outputs/tmp_validation/`.

- [x] **Step 1: Run 20-frame smoke**

Run in tmux:

```bash
tmux new-session -d -s weakstruct20f \
'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && LOG=/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_20f.log && STATUS=/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_20f.status && { echo experiment=20260526_room0_semantic_vote_weak_structure_stride10_20f; echo started_at=$(date --iso-8601=seconds); echo running > "$STATUS"; /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_4090.yaml --experiment-name 20260526_room0_semantic_vote_weak_structure_stride10_20f --num-frames 20 --frame-stride 10 --dataset-root /home/ww/vv/dataset/Replica/room0 --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json --output-root /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation --proposal-backend precomputed --proposal-cache-dir /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2 --proposal-cache-manifest /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json; status=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=$status; echo exit_status=$status > "$STATUS"; exit $status; } > "$LOG" 2>&1'
```

- [x] **Step 2: Verify smoke finished successfully**

Run:

```bash
cat outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_20f.status
tail -n 30 outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_20f.log
wc -l outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_20f/room0/frame_metrics.jsonl
```

Expected:

```text
exit_status=0
20 .../frame_metrics.jsonl
```

- [x] **Step 3: Run 200-frame comparison**

Run in tmux:

```bash
tmux new-session -d -s weakstruct200f \
'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && LOG=/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_200f.log && STATUS=/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_200f.status && { echo experiment=20260526_room0_semantic_vote_weak_structure_stride10_200f; echo started_at=$(date --iso-8601=seconds); echo running > "$STATUS"; /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_4090.yaml --experiment-name 20260526_room0_semantic_vote_weak_structure_stride10_200f --num-frames 200 --frame-stride 10 --dataset-root /home/ww/vv/dataset/Replica/room0 --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json --output-root /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation --proposal-backend precomputed --proposal-cache-dir /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2 --proposal-cache-manifest /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json; status=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=$status; echo exit_status=$status > "$STATUS"; exit $status; } > "$LOG" 2>&1'
```

- [x] **Step 4: Compare against strict anchorcover and fix52 baselines**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

runs = {
    "fix52": Path("outputs/tmp_validation/20260526_room0_observation_first_fix52e1714_200f"),
    "strict_anchorcover": Path("outputs/tmp_validation/20260526_room0_observation_first_anchorcover_stride10_200f_rerun2"),
    "weak_structure": Path("outputs/tmp_validation/20260526_room0_semantic_vote_weak_structure_stride10_200f"),
}
for name, root in runs.items():
    report = json.loads((root / "room0/run_report.json").read_text())
    iou = json.loads((root / "replica/classes_iou.json").read_text())
    acc = json.loads((root / "replica/classes_acc.json").read_text())
    print(name)
    print("  miou", report["miou"], "macc", report["macc"], "fmiou", report["fmiou"], "fmacc", report["fmacc"])
    print("  unanchored", report["anchor_voted_unanchored_count_total"])
    print("  missing/mismatch/id", report["runtime_anchor_label_missing_edge_count_total"], report["runtime_anchor_label_mismatch_edge_count_total"], report["runtime_anchor_identity_mismatch_edge_count_total"])
    for cls in ["rug", "sofa", "floor", "ceiling", "wall", "window", "lamp", "book"]:
        print(" ", cls, "IoU", iou.get(cls), "Acc", acc.get(cls))
PY
```

Expected interpretation:

- `runtime_anchor_label_mismatch_edge_count_total` stays closer to strict anchorcover than fix52.
- `runtime_anchor_label_missing_edge_count_total` drops substantially from strict anchorcover.
- `floor`, `ceiling`, and `wall` recover relative to strict anchorcover.
- `rug` and `sofa` should not fall back to the fix52 pollution pattern.

## Self-Review Notes

- Spec coverage: The plan covers strict strong assignment, weak clipped structure proposals, no weak object proposals, metadata distinction, audit visibility, tests, and validation runs.
- Placeholder scan: No task uses unbounded "TODO" work; each code step includes concrete snippets and commands.
- Type consistency: All new helpers use existing `Proposal2D`, `Anchor2D`, and `AnchorAssignment`. Metadata keys are plain dict entries consumed by existing downstream code.
- Scope check: This is one coherent frontend anchor-assignment change. It does not require changes to runtime association or semantic memory beyond optional audit serialization.
