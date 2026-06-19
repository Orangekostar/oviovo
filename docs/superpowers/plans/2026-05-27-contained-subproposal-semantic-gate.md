# Contained Subproposal Semantic Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent SAM subproposals fully contained inside a larger YOLO anchor from inheriting the parent's semantic label, and make object-level semantic labels order-insensitive across frames.

**Architecture:** Add a frontend containment ambiguity gate in `ObjectAnchorModule` so strong anchor labels require both proposal coverage and anchor/proposal scale compatibility. Preserve blocked proposals as unlabeled SAM geometry with explicit debug metadata. Replace the current recency-biased object anchor vote with an evidence ledger that appends every valid observation and recomputes canonical labels from all evidence, making replay order irrelevant.

**Tech Stack:** Python, NumPy, pytest, existing `Proposal2D` / `Anchor2D` / `Patch3D` metadata flow.

---

## File Structure

- Modify `src/modules/object_anchor.py`
  - Add containment-gate config.
  - Add helper methods for scale compatibility and blocked-candidate metadata.
  - Attach `anchor_blocked_candidates` metadata to unassigned proposals.
- Modify `src/modules/semantic_memory.py`
  - Add semantic evidence ledger helpers.
  - Make `accumulate_anchor_semantic_vote()` ignore weak/blocked/no-label observations.
  - Recompute canonical label from evidence in a deterministic, order-insensitive way.
- Modify `configs/default.yaml`
  - Document containment gate and thresholds.
- Modify `configs/room0_surface_gate_4090.yaml`
  - Enable containment gate for the room0 experiments.
- Modify `tests/test_object_anchor.py`
  - Add failing coverage for small contained SAM masks inside large sofa anchors.
  - Add positive coverage for scale-compatible SAM masks.
  - Assert weak-structure behavior remains unchanged.
- Modify `tests/test_pipeline.py`
  - Add order-invariance tests for anchor semantic evidence.
  - Add tests that contained/blocked frontend metadata does not enter object semantic memory.

---

### Task 1: Frontend Containment Gate Configuration

**Files:**
- Modify: `src/modules/object_anchor.py`
- Modify: `configs/default.yaml`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Test: `tests/test_object_anchor.py`

- [ ] **Step 1: Write failing config/default test**

Add this test after `test_object_anchor_weak_structure_overlap_config_defaults()` in `tests/test_object_anchor.py`:

```python
def test_object_anchor_contained_subproposal_gate_config_defaults() -> None:
    module = ObjectAnchorModule({"enabled": False, "assignment_policy": "semantic_vote"})

    assert module.contained_subproposal_gate_enabled is True
    assert module.contained_max_anchor_coverage == 0.20
    assert module.contained_min_proposal_coverage == 0.85
    assert module.scale_compatible_min_anchor_coverage == 0.20
    assert module.scale_compatible_min_bbox_iou == 0.10
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_object_anchor_contained_subproposal_gate_config_defaults -q
```

Expected: FAIL with `AttributeError: 'ObjectAnchorModule' object has no attribute 'contained_subproposal_gate_enabled'`.

- [ ] **Step 3: Add config fields to `ObjectAnchorModule.__init__`**

In `src/modules/object_anchor.py`, directly after `self.covering_min_proposal_coverage = ...`, add:

```python
        self.contained_subproposal_gate_enabled = bool(
            config.get("contained_subproposal_gate_enabled", self.semantic_vote_policy)
        )
        self.contained_max_anchor_coverage = float(
            np.clip(config.get("contained_max_anchor_coverage", self.min_anchor_mask_coverage), 0.0, 1.0)
        )
        self.contained_min_proposal_coverage = float(
            np.clip(config.get("contained_min_proposal_coverage", self.covering_min_proposal_coverage), 0.0, 1.0)
        )
        self.scale_compatible_min_anchor_coverage = float(
            np.clip(config.get("scale_compatible_min_anchor_coverage", self.min_anchor_mask_coverage), 0.0, 1.0)
        )
        self.scale_compatible_min_bbox_iou = float(
            np.clip(config.get("scale_compatible_min_bbox_iou", 0.10), 0.0, 1.0)
        )
```

- [ ] **Step 4: Document config in `configs/default.yaml`**

Add these keys under `anchor_frontend`, directly after `covering_min_proposal_coverage: 0.85`:

```yaml
  # Prevent small SAM masks fully inside a much larger YOLO box from inheriting the parent label.
  contained_subproposal_gate_enabled: true
  contained_max_anchor_coverage: 0.20
  contained_min_proposal_coverage: 0.85
  scale_compatible_min_anchor_coverage: 0.20
  scale_compatible_min_bbox_iou: 0.10
```

- [ ] **Step 5: Enable config in `configs/room0_surface_gate_4090.yaml`**

Add these keys under `anchor_frontend`, directly after `covering_min_proposal_coverage: 0.85`:

```yaml
  contained_subproposal_gate_enabled: true
  contained_max_anchor_coverage: 0.20
  contained_min_proposal_coverage: 0.85
  scale_compatible_min_anchor_coverage: 0.20
  scale_compatible_min_bbox_iou: 0.10
```

- [ ] **Step 6: Run test and verify it passes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_object_anchor_contained_subproposal_gate_config_defaults -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/modules/object_anchor.py configs/default.yaml configs/room0_surface_gate_4090.yaml tests/test_object_anchor.py
git commit -m "feat: configure contained subproposal semantic gate"
```

---

### Task 2: Block Parent-Label Inheritance For Contained SAM Subproposals

**Files:**
- Modify: `src/modules/object_anchor.py`
- Test: `tests/test_object_anchor.py`

- [ ] **Step 1: Write failing contained-subproposal regression test**

Add this test before `test_anchor_vote_does_not_label_large_sam_mask_from_small_contained_anchor()` in `tests/test_object_anchor.py`:

```python
def test_anchor_vote_does_not_label_small_contained_sam_mask_from_large_parent_anchor() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.20,
            "covering_min_proposal_coverage": 0.85,
            "contained_subproposal_gate_enabled": True,
            "contained_max_anchor_coverage": 0.20,
            "contained_min_proposal_coverage": 0.85,
            "scale_compatible_min_anchor_coverage": 0.20,
            "scale_compatible_min_bbox_iou": 0.10,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    proposal = Proposal2D(
        proposal_id=40,
        mask=mask.copy(),
        bbox_xyxy=np.array([8, 8, 12, 12], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.99,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    assert assignments[0].anchor_id == -1
    assert voted[0].metadata["anchor_id"] == -1
    assert voted[0].metadata["anchor_class_name"] == ""
    assert voted[0].metadata["anchor_label_strength"] == "none"
    assert voted[0].metadata["anchor_candidate_classes"] == []
    assert voted[0].metadata["anchor_label_votes"] == {}
    assert voted[0].metadata["anchor_blocked_candidates"][0]["anchor_id"] == 0
    assert voted[0].metadata["anchor_blocked_candidates"][0]["class_name"] == "sofa"
    assert voted[0].metadata["anchor_blocked_candidates"][0]["reason"] == "contained_subproposal_without_child_anchor"
    assert voted[0].metadata["anchor_blocked_candidates"][0]["proposal_coverage"] == 1.0
    assert voted[0].metadata["anchor_blocked_candidates"][0]["anchor_coverage"] == 0.04
```

- [ ] **Step 2: Write positive scale-compatible test**

Add this test immediately after the contained-subproposal regression test:

```python
def test_anchor_vote_keeps_strong_label_for_scale_compatible_sam_mask() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.20,
            "covering_min_proposal_coverage": 0.85,
            "contained_subproposal_gate_enabled": True,
            "contained_max_anchor_coverage": 0.20,
            "contained_min_proposal_coverage": 0.85,
            "scale_compatible_min_anchor_coverage": 0.20,
            "scale_compatible_min_bbox_iou": 0.10,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((20, 20), dtype=bool)
    mask[1:19, 1:19] = True
    proposal = Proposal2D(
        proposal_id=41,
        mask=mask.copy(),
        bbox_xyxy=np.array([1, 1, 19, 19], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.99,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    assert assignments[0].anchor_id == 0
    assert assignments[0].class_name == "sofa"
    assert voted[0].metadata["anchor_id"] == 0
    assert voted[0].metadata["anchor_class_name"] == "sofa"
    assert voted[0].metadata["anchor_label_strength"] == "strong"
    assert voted[0].metadata["anchor_candidate_classes"] == ["sofa"]
    assert voted[0].metadata["anchor_blocked_candidates"] == []
```

- [ ] **Step 3: Run the two tests and verify one fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_object_anchor.py::test_anchor_vote_does_not_label_small_contained_sam_mask_from_large_parent_anchor \
  tests/test_object_anchor.py::test_anchor_vote_keeps_strong_label_for_scale_compatible_sam_mask \
  -q
```

Expected: the contained-subproposal test FAILS because the current code assigns `sofa`.

- [ ] **Step 4: Add helper methods in `src/modules/object_anchor.py`**

Add these methods directly after `_semantic_vote_overlap_metrics()`:

```python
    def _is_scale_compatible_semantic_vote(self, metrics: dict[str, Any]) -> bool:
        anchor_coverage = float(metrics["anchor_coverage"])
        bbox_iou = float(metrics["bbox_iou"])
        return bool(
            anchor_coverage >= self.scale_compatible_min_anchor_coverage
            or bbox_iou >= self.scale_compatible_min_bbox_iou
        )

    def _is_contained_subproposal_without_child_anchor(self, metrics: dict[str, Any]) -> bool:
        if not self.contained_subproposal_gate_enabled:
            return False
        proposal_coverage = float(metrics["proposal_coverage"])
        anchor_coverage = float(metrics["anchor_coverage"])
        if proposal_coverage < self.contained_min_proposal_coverage:
            return False
        if anchor_coverage > self.contained_max_anchor_coverage:
            return False
        return not self._is_scale_compatible_semantic_vote(metrics)

    def _blocked_semantic_vote_candidate(
        self,
        anchor: Anchor2D,
        metrics: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "anchor_id": int(anchor.anchor_id),
            "class_name": str(anchor.class_name),
            "confidence": float(anchor.confidence),
            "reason": str(reason),
            "bbox_iou": float(metrics["bbox_iou"]),
            "center_inside": bool(metrics["center_inside"]),
            "proposal_coverage": float(metrics["proposal_coverage"]),
            "anchor_coverage": float(metrics["anchor_coverage"]),
            "overlap_area": int(metrics["overlap_area"]),
        }
```

- [ ] **Step 5: Change `_assign_semantic_vote_candidates()` to return blocked candidates**

Replace the whole `_assign_semantic_vote_candidates()` method with this implementation:

```python
    def _assign_semantic_vote_candidates(
        self,
        proposal: Proposal2D,
        anchors: List[Anchor2D],
    ) -> tuple[list[AnchorAssignment], list[dict[str, Any]]]:
        assignments: list[AnchorAssignment] = []
        blocked_candidates: list[dict[str, Any]] = []
        if not anchors:
            return assignments, blocked_candidates

        for anchor in anchors:
            metrics = self._semantic_vote_overlap_metrics(proposal, anchor)
            if int(metrics["overlap_area"]) < self.proposal_min_area:
                continue

            assignable = float(metrics["proposal_coverage"]) >= self.covering_min_proposal_coverage
            if not assignable:
                continue

            if self._is_contained_subproposal_without_child_anchor(metrics):
                blocked_candidates.append(
                    self._blocked_semantic_vote_candidate(
                        anchor,
                        metrics,
                        reason="contained_subproposal_without_child_anchor",
                    )
                )
                continue

            bbox_iou = float(metrics["bbox_iou"])
            center_inside = bool(metrics["center_inside"])
            anchor_coverage = float(metrics["anchor_coverage"])
            keepalive = bool(
                bbox_iou >= self.keepalive_iou_threshold
                or (self.keepalive_center_inside and center_inside)
                or anchor_coverage >= self.min_anchor_mask_coverage
            )
            assignments.append(
                AnchorAssignment(
                    proposal_id=int(proposal.proposal_id),
                    anchor_id=int(anchor.anchor_id),
                    class_name=str(anchor.class_name),
                    confidence=float(anchor.confidence),
                    bbox_iou=float(bbox_iou),
                    center_inside=bool(center_inside),
                    keepalive=keepalive,
                )
            )

        return assignments, blocked_candidates
```

- [ ] **Step 6: Update `_generate_semantic_vote_proposals()` call site**

In `_generate_semantic_vote_proposals()`, replace:

```python
            assignments = self._assign_semantic_vote_candidates(proposal, self.last_anchors)
            if not assignments:
                voted_proposals.append(self._clone_proposal_with_anchor_vote(proposal, None, []))
                voted_assignments.append(AnchorAssignment(proposal_id=int(proposal.proposal_id)))
            else:
```

with:

```python
            assignments, blocked_candidates = self._assign_semantic_vote_candidates(proposal, self.last_anchors)
            if not assignments:
                voted_proposals.append(
                    self._clone_proposal_with_anchor_vote(proposal, None, [], blocked_candidates)
                )
                voted_assignments.append(AnchorAssignment(proposal_id=int(proposal.proposal_id)))
            else:
```

Then replace:

```python
                        assignments,
```

inside the `else` branch with:

```python
                        assignments,
                        blocked_candidates,
```

- [ ] **Step 7: Update `_clone_proposal_with_anchor_vote()` signature and metadata**

Change the method signature from:

```python
    def _clone_proposal_with_anchor_vote(
        self,
        proposal: Proposal2D,
        assignment: AnchorAssignment | None,
        candidates: list[AnchorAssignment],
    ) -> Proposal2D:
```

to:

```python
    def _clone_proposal_with_anchor_vote(
        self,
        proposal: Proposal2D,
        assignment: AnchorAssignment | None,
        candidates: list[AnchorAssignment],
        blocked_candidates: list[dict[str, Any]] | None = None,
    ) -> Proposal2D:
```

Inside `metadata.update({ ... })`, add:

```python
                "anchor_blocked_candidates": list(blocked_candidates or []),
```

- [ ] **Step 8: Update other `_assign_semantic_vote_candidates()` call sites**

Run:

```bash
rg -n "_assign_semantic_vote_candidates" src tests
```

For any direct test or implementation call expecting a list, change:

```python
assignments = module._assign_semantic_vote_candidates(proposal, anchors)
```

to:

```python
assignments, blocked_candidates = module._assign_semantic_vote_candidates(proposal, anchors)
assert blocked_candidates == []
```

- [ ] **Step 9: Run object anchor tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add src/modules/object_anchor.py tests/test_object_anchor.py
git commit -m "fix: block contained sam proposals from parent semantic labels"
```

---

### Task 3: Keep Weak Structure Recovery Compatible

**Files:**
- Modify: `src/modules/object_anchor.py`
- Test: `tests/test_object_anchor.py`

- [ ] **Step 1: Add regression for weak structure unaffected by containment gate**

Add this test after `test_anchor_vote_adds_weak_clipped_structure_proposal_for_partial_structure_anchor()`:

```python
def test_containment_gate_does_not_block_weak_structure_overlap_recovery() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.85,
            "contained_subproposal_gate_enabled": True,
            "contained_max_anchor_coverage": 0.20,
            "contained_min_proposal_coverage": 0.85,
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

    whole = next(item for item in voted if item.proposal_id == 9)
    weak = next(item for item in voted if item.proposal_id != 9)

    assert whole.metadata["anchor_id"] == -1
    assert whole.metadata["anchor_blocked_candidates"] == []
    assert weak.metadata["anchor_class_name"] == "wall"
    assert weak.metadata["anchor_label_strength"] == "weak_overlap"
    assert any(item.anchor_id == 3 and item.class_name == "wall" for item in assignments)
```

- [ ] **Step 2: Run the weak-structure regression**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_containment_gate_does_not_block_weak_structure_overlap_recovery -q
```

Expected: PASS.

- [ ] **Step 3: Run object anchor suite**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add src/modules/object_anchor.py tests/test_object_anchor.py
git commit -m "test: preserve weak structure recovery with containment gate"
```

---

### Task 4: Deterministic Semantic Evidence Ledger

**Files:**
- Modify: `src/modules/semantic_memory.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write order-invariance test**

Add this test after `test_anchor_semantic_vote_allows_later_high_confidence_relabel()` in `tests/test_pipeline.py`:

```python
    def test_anchor_semantic_vote_is_order_invariant_for_same_evidence(self):
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )

        def make_patch(patch_id: int, label: str, confidence: float, view_quality: float) -> Patch3D:
            return Patch3D(
                patch_id=patch_id,
                points=points,
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=patch_id,
                metadata={
                    "anchor_class_name": label,
                    "anchor_confidence": confidence,
                    "anchor_view_quality": view_quality,
                    "anchor_label_strength": "strong",
                },
            )

        evidence = [
            make_patch(1, "sofa", 0.60, 0.70),
            make_patch(2, "sofa", 0.60, 0.70),
            make_patch(3, "blanket", 0.95, 0.90),
            make_patch(4, "blanket", 0.95, 0.90),
        ]

        forward = ObjectMap(object_id=21)
        reverse = ObjectMap(object_id=22)
        for patch in evidence:
            accumulate_anchor_semantic_vote(forward, patch)
        for patch in reversed(evidence):
            accumulate_anchor_semantic_vote(reverse, patch)

        assert forward.debug["anchor_semantics"]["label_weighted_score"] == reverse.debug["anchor_semantics"]["label_weighted_score"]
        assert forward.debug["anchor_semantics"]["canonical_label"] == reverse.debug["anchor_semantics"]["canonical_label"]
        assert preferred_object_semantic_label(forward) == "blanket"
        assert preferred_object_semantic_label(reverse) == "blanket"
```

- [ ] **Step 2: Write blocked/weak evidence exclusion test**

Add this test immediately after the order-invariance test:

```python
    def test_anchor_semantic_vote_ignores_blocked_and_weak_overlap_labels(self):
        obj = ObjectMap(object_id=23)
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)

        weak_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "wall",
                "anchor_confidence": 0.9,
                "anchor_label_strength": "weak_overlap",
            },
        )
        blocked_patch = Patch3D(
            patch_id=2,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "",
                "anchor_confidence": 0.0,
                "anchor_label_strength": "none",
                "anchor_blocked_candidates": [
                    {
                        "anchor_id": 0,
                        "class_name": "sofa",
                        "confidence": 0.9,
                        "reason": "contained_subproposal_without_child_anchor",
                    }
                ],
            },
        )
        strong_patch = Patch3D(
            patch_id=3,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=3,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.85,
                "anchor_view_quality": 0.8,
                "anchor_label_strength": "strong",
            },
        )

        accumulate_anchor_semantic_vote(obj, weak_patch)
        accumulate_anchor_semantic_vote(obj, blocked_patch)
        accumulate_anchor_semantic_vote(obj, strong_patch)

        anchor_state = obj.debug["anchor_semantics"]
        assert preferred_object_semantic_label(obj) == "blanket"
        assert anchor_state["label_weighted_score"] == {"blanket": 0.7225}
        assert anchor_state["ignored_observation_count"] == 2
        assert anchor_state["ignored_observation_reasons"] == {
            "non_strong_anchor_label": 1,
            "missing_anchor_label": 1,
        }
```

- [ ] **Step 3: Run tests and verify the order-invariance test fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestSemanticMemory::test_anchor_semantic_vote_is_order_invariant_for_same_evidence \
  tests/test_pipeline.py::TestSemanticMemory::test_anchor_semantic_vote_ignores_blocked_and_weak_overlap_labels \
  -q
```

Expected: the order-invariance test FAILS because the current implementation applies recency decay using observation order.

- [ ] **Step 4: Add evidence ledger helper in `src/modules/semantic_memory.py`**

Replace `_accumulate_anchor_vote()` with this deterministic implementation:

```python
def _accumulate_anchor_vote(
    obj: ObjectMap,
    *,
    label: str,
    confidence: float,
    frame_id: int,
    view_quality: float,
) -> None:
    state = _anchor_semantic_state(obj)
    previous_label = str(state.get("canonical_label", ""))
    evidence = state["evidence"]
    vote_weight = float(confidence) if confidence > 0.0 else 1.0
    view_weight = float(np.clip(view_quality, 0.0, 1.0))
    weighted_vote = vote_weight * (0.25 + 0.75 * view_weight)
    evidence.append(
        {
            "label": str(label),
            "confidence": float(confidence),
            "frame_id": int(frame_id),
            "view_quality": float(view_weight),
            "vote_weight": float(vote_weight),
            "weighted_vote": float(weighted_vote),
        }
    )
    _recompute_anchor_semantic_state(state)
    best_label = str(state.get("canonical_label", ""))
    if previous_label and best_label and previous_label != best_label:
        state.setdefault("relabel_events", []).append(
            {
                "frame_id": int(frame_id),
                "old_label": previous_label,
                "new_label": best_label,
                "new_confidence": float(confidence),
                "new_view_quality": float(view_weight),
                "reason": "stronger_anchor_evidence_ledger",
            }
        )
```

Add this helper directly after `_accumulate_anchor_vote()`:

```python
def _recompute_anchor_semantic_state(state: dict[str, Any]) -> None:
    score_sum: dict[str, float] = {}
    weighted_score: dict[str, float] = {}
    max_confidence: dict[str, float] = {}
    best_view_quality: dict[str, float] = {}
    seen_frames: dict[str, list[int]] = {}
    high_quality_hits: dict[str, int] = {}

    for item in state.get("evidence", []):
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        confidence = float(item.get("confidence", 0.0))
        frame_id = int(item.get("frame_id", 0))
        view_quality = float(np.clip(float(item.get("view_quality", 0.0)), 0.0, 1.0))
        vote_weight = float(item.get("vote_weight", confidence if confidence > 0.0 else 1.0))
        weighted_vote = float(item.get("weighted_vote", vote_weight * (0.25 + 0.75 * view_quality)))

        score_sum[label] = float(score_sum.get(label, 0.0) + vote_weight)
        weighted_score[label] = float(weighted_score.get(label, 0.0) + weighted_vote)
        max_confidence[label] = max(float(max_confidence.get(label, 0.0)), confidence)
        best_view_quality[label] = max(float(best_view_quality.get(label, 0.0)), view_quality)
        frames = seen_frames.setdefault(label, [])
        if frame_id not in frames:
            frames.append(frame_id)
        if confidence >= ANCHOR_HIGH_CONFIDENCE_THRESHOLD and view_quality >= ANCHOR_HIGH_VIEW_QUALITY_THRESHOLD:
            high_quality_hits[label] = int(high_quality_hits.get(label, 0) + 1)

    frame_hits = {label: len(frames) for label, frames in seen_frames.items()}
    best_label = ""
    best_key: tuple[int, int, float, float, float, str] | None = None
    for candidate_label in sorted(weighted_score):
        candidate_key = (
            int(high_quality_hits.get(candidate_label, 0)),
            int(frame_hits.get(candidate_label, 0)),
            float(weighted_score.get(candidate_label, 0.0)),
            float(max_confidence.get(candidate_label, 0.0)),
            float(best_view_quality.get(candidate_label, 0.0)),
            candidate_label,
        )
        if best_key is None or candidate_key > best_key:
            best_label = candidate_label
            best_key = candidate_key

    state["label_score_sum"] = score_sum
    state["label_weighted_score"] = weighted_score
    state["label_recent_score"] = dict(weighted_score)
    state["label_max_confidence"] = max_confidence
    state["label_best_view_quality"] = best_view_quality
    state["label_seen_frames"] = {label: sorted(frames) for label, frames in seen_frames.items()}
    state["label_frame_hits"] = frame_hits
    state["label_high_quality_hits"] = high_quality_hits
    state["canonical_label"] = best_label
    state["canonical_score"] = float(weighted_score.get(best_label, 0.0)) if best_label else 0.0
    state["canonical_frame_hits"] = int(frame_hits.get(best_label, 0)) if best_label else 0
    state["canonical_confidence"] = float(max_confidence.get(best_label, 0.0)) if best_label else 0.0
    state["canonical_best_view_quality"] = float(best_view_quality.get(best_label, 0.0)) if best_label else 0.0
    state["source"] = "anchor_vote" if best_label else ""
```

- [ ] **Step 5: Update `_anchor_semantic_state()` for evidence ledger**

Inside `_anchor_semantic_state()`, after `if not isinstance(state.get("relabel_events"), list):`, add:

```python
    if not isinstance(state.get("evidence"), list):
        state["evidence"] = []
    if not isinstance(state.get("ignored_observation_reasons"), dict):
        state["ignored_observation_reasons"] = {}
    state.setdefault("ignored_observation_count", 0)
```

- [ ] **Step 6: Ignore weak/blocked/non-strong labels in `accumulate_anchor_semantic_vote()`**

Replace the top of `accumulate_anchor_semantic_vote()` with:

```python
def accumulate_anchor_semantic_vote(obj: ObjectMap, patch: Patch3D) -> None:
    """Accumulate an anchor-class vote into the object's semantic state."""
    label = str(patch.metadata.get("anchor_class_name", "")).strip()
    strength = str(patch.metadata.get("anchor_label_strength", "strong")).strip()
    if not label:
        _record_ignored_anchor_observation(obj, "missing_anchor_label")
        return
    if strength != "strong":
        _record_ignored_anchor_observation(obj, "non_strong_anchor_label")
        return
    confidence = float(patch.metadata.get("anchor_confidence", 0.0))
    frame_id = int(patch.source_frame_id)
    _accumulate_anchor_vote(
        obj,
        label=label,
        confidence=confidence,
        frame_id=frame_id,
        view_quality=_patch_anchor_view_quality(patch),
    )
```

Add this helper directly after `accumulate_anchor_semantic_vote()`:

```python
def _record_ignored_anchor_observation(obj: ObjectMap, reason: str) -> None:
    state = _anchor_semantic_state(obj)
    state["ignored_observation_count"] = int(state.get("ignored_observation_count", 0)) + 1
    reasons = state["ignored_observation_reasons"]
    reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1
```

- [ ] **Step 7: Run semantic memory tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestSemanticMemory::test_anchor_semantic_vote_allows_later_high_confidence_relabel \
  tests/test_pipeline.py::TestSemanticMemory::test_anchor_semantic_vote_is_order_invariant_for_same_evidence \
  tests/test_pipeline.py::TestSemanticMemory::test_anchor_semantic_vote_ignores_blocked_and_weak_overlap_labels \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/modules/semantic_memory.py tests/test_pipeline.py
git commit -m "feat: make anchor semantic evidence order invariant"
```

---

### Task 5: Audit Visibility For Blocked Semantic Candidates

**Files:**
- Modify: `src/pipelines/main_pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write audit regression**

In `tests/test_pipeline.py`, update `test_local_memory_frame_audit_records_weak_structure_anchor_metadata()` by adding `anchor_blocked_candidates` to `voted_proposal.metadata`:

```python
                "anchor_blocked_candidates": [
                    {
                        "anchor_id": 0,
                        "class_name": "sofa",
                        "confidence": 0.9,
                        "reason": "contained_subproposal_without_child_anchor",
                        "proposal_coverage": 1.0,
                        "anchor_coverage": 0.04,
                    }
                ],
```

Then add these assertions after the existing `anchor_anchor_coverage` assertion:

```python
        assert audit["anchor_voted_proposals"][0]["anchor_blocked_candidates"][0]["class_name"] == "sofa"
        assert audit["anchor_voted_proposals"][0]["anchor_blocked_candidates"][0]["reason"] == "contained_subproposal_without_child_anchor"
```

- [ ] **Step 2: Run audit test and verify it fails if audit omits metadata**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestLocalMemoryAudit::test_local_memory_frame_audit_records_weak_structure_anchor_metadata -q
```

Expected: FAIL if `anchor_blocked_candidates` is not exported by the audit record.

- [ ] **Step 3: Add audit field in `build_local_memory_frame_audit()`**

In `src/pipelines/main_pipeline.py`, find the helper block that serializes `anchor_voted_proposals`. Add this key to each proposal record:

```python
                "anchor_blocked_candidates": list(proposal.metadata.get("anchor_blocked_candidates", [])),
```

- [ ] **Step 4: Run audit test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestLocalMemoryAudit::test_local_memory_frame_audit_records_weak_structure_anchor_metadata -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/pipelines/main_pipeline.py tests/test_pipeline.py
git commit -m "feat: expose blocked anchor candidates in frame audit"
```

---

### Task 6: Full Verification And Smoke Experiment

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 2: Run prior export/pipeline regression set**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py tests/test_dual_map.py tests/test_dualmap_baseline_export.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 3: Start a room0 stride=10 200f smoke experiment**

Run:

```bash
tmux new-session -d -s room0_contained_gate_stride10_200f "/bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && experiment=20260527_room0_contained_gate_stride10_200f && log=outputs/tmp_validation/${experiment}.log && status=outputs/tmp_validation/${experiment}.status && { echo experiment=${experiment}; echo started_at=$(date --iso-8601=seconds); echo running > ${status}; /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_4090.yaml --experiment-name ${experiment} --num-frames 200 --frame-stride 10 --dataset-root /home/ww/vv/dataset/Replica/room0 --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json --output-root /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation --proposal-backend precomputed --proposal-cache-dir /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2 --proposal-cache-manifest /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json; rc=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=${rc}; echo exit_status=${rc} > ${status}; exit ${rc}; } > ${log} 2>&1'"
```

Expected:
- tmux session starts.
- `outputs/tmp_validation/20260527_room0_contained_gate_stride10_200f.status` contains `running`.
- `outputs/tmp_validation/20260527_room0_contained_gate_stride10_200f/room0/frame_metrics.jsonl` begins growing after the first processed frames.

- [ ] **Step 4: Inspect target failure signatures**

After the smoke experiment produces exports/audits, inspect these signals:

```bash
rg -n "contained_subproposal_without_child_anchor|anchor_blocked_candidates" outputs/tmp_validation/20260527_room0_contained_gate_stride10_200f/room0
```

Expected:
- At least some frame audit records contain `contained_subproposal_without_child_anchor` when a small SAM proposal was fully inside a large parent anchor.
- Those same proposal records have `anchor_class_name` as an empty string and `anchor_label_strength` as `none`.

- [ ] **Step 5: Commit smoke-test metadata only if new expected files are intentionally tracked**

If no tracked files changed, do not commit. If tracked benchmark fixtures were intentionally updated, run:

```bash
git status --short
git add <tracked-fixture-paths>
git commit -m "test: update contained gate smoke fixtures"
```

---

## Design Rationale

The immediate bug is not simply "SAM bad" or "YOLO bad"; it is an unsafe semantic transfer rule. A YOLO sofa anchor can be correct at the parent-object level while a contained SAM mask is a different child object, for example blanket/rug/cushion. If the SAM proposal has high `proposal_coverage` but very low `anchor_coverage`, the overlap says "the proposal is inside the anchor box", not "the proposal is the anchor class".

The frontend rule should therefore be:

```text
Strong semantic assignment = high proposal coverage AND scale-compatible anchor/proposal geometry.
Contained ambiguity = high proposal coverage AND low anchor coverage AND low bbox IoU.
Contained ambiguity keeps geometry but blocks semantic inheritance.
```

The multi-frame rule should be:

```text
Object semantic label = deterministic function of all strong semantic evidence.
Observation order must not change the final label.
Weak overlap and blocked candidates are debug evidence, not semantic votes.
```

This keeps the core paper story clean: SAM gives geometry proposals, YOLO/YOLO-World gives lexical evidence, and the system explicitly refuses parent-label spillover when geometry says the proposal is a contained subregion rather than the detected object itself.

## Self-Review

Spec coverage:
- Sofa/blanket failure is covered by Task 2.
- Existing vase/indoor-plant style parent-child ambiguity is covered by the same containment rule.
- Weak structure recovery is preserved in Task 3.
- Different frame orders producing different labels is addressed by Task 4.
- Debug visibility for visual validation is addressed by Task 5.
- Experiment validation is addressed by Task 6.

Placeholder scan:
- No steps use unspecified code or deferred behavior.
- Every code change step includes concrete code.
- Every test step includes exact commands and expected results.

Type consistency:
- `anchor_blocked_candidates` is a list of dictionaries stored in `Proposal2D.metadata`.
- `AnchorAssignment` remains unchanged; blocked candidates are not assignments.
- Existing `anchor_label_strength` values remain `strong`, `weak_overlap`, and `none`.
- `semantic_memory.py` keeps existing `anchor_semantics` keys while adding `evidence`, `ignored_observation_count`, and `ignored_observation_reasons`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-27-contained-subproposal-semantic-gate.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
