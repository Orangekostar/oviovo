# Geometry-First Semantic-Late Object Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the 0526 observation-first recall while keeping the 0527 contained/weak semantic safeguards by separating 3D geometry ownership from delayed object semantic commitment.

**Architecture:** Add an object-level semantic posterior with explicit `unlabeled`, `provisional`, and `committed` states. Strong direct anchor evidence can commit labels after enough multi-frame support; weak/contextual/contained evidence is retained for audit but cannot determine export labels or association identity. Association uses committed labels when available and falls back to provisional labels only after confidence gates, so cross-label evidence becomes residual geometry instead of contaminating parent objects.

**Tech Stack:** Python dataclasses, NumPy, pytest, existing `Patch3D` metadata flow, existing `ObjectMap.debug["anchor_semantics"]` ledger, existing contested residual/provisional pool.

---

## File Structure

- Modify `src/modules/semantic_memory.py`
  - Add semantic posterior configuration constants and helpers.
  - Extend anchor evidence records with `evidence_kind`.
  - Track semantic commit state and committed label separately from the current best posterior label.
  - Make `preferred_object_semantic_label()` return committed labels by default, with a controlled provisional fallback.

- Modify `src/modules/observation_identity.py`
  - Use committed object identity for cross-label blocking.
  - Treat uncommitted objects as updateable geometry targets but do not let them block child/residual promotion.

- Modify `src/modules/object_update.py`
  - Preserve semantic posterior fields during debug refresh.
  - Ensure provisional promotion rebuilds posterior evidence and keeps uncommitted objects unlabeled until commit gates pass.
  - Add config plumbing for delayed semantic commit thresholds.

- Modify `run_room0_full_eval.py`
  - Export/audit committed/provisional semantic state.
  - Add counts for unlabeled/provisional/committed objects and weak/contextual evidence.

- Modify `configs/default.yaml`
  - Add `semantic_memory.anchor_commit` thresholds.

- Modify `configs/room0_surface_gate_4090.yaml`
  - Enable delayed commit for the room0 full experiment profile.

- Modify `tests/test_pipeline.py`
  - Add semantic posterior unit tests.
  - Add observation identity tests proving provisional parent labels do not swallow confident child labels.
  - Add audit/export tests for semantic state fields.

---

## Design Rules

1. Geometry may enter `ObjectMap` or `ProvisionalObject` before semantic label commitment.
2. Only strong direct evidence contributes to semantic commitment.
3. Weak/contextual/contained evidence is recorded but cannot become `committed_label`.
4. Association identity blocking uses committed object labels. If the object is uncommitted, geometry update is allowed but the object remains semantically provisional.
5. A confident child patch with a strong label different from a committed parent still goes to contested residual/provisional tracking.
6. Export labels use committed labels. If an object has no committed label, export may fall back to provisional only when explicitly enabled.
7. The implementation must be order-insensitive: replaying the same evidence in different frame order produces the same committed label.

---

### Task 1: Add Semantic Posterior State And Commit Gates

**Files:**
- Modify: `src/modules/semantic_memory.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing posterior state test**

Add this test after `test_anchor_semantic_vote_is_order_invariant_for_same_evidence` in `tests/test_pipeline.py`:

```python
    def test_anchor_semantic_posterior_requires_multiframe_commit(self):
        from src.modules.semantic_memory import object_semantic_commit_state

        obj = ObjectMap(object_id=31)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )

        first_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=10,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.92,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
        )
        accumulate_anchor_semantic_vote(obj, first_patch)

        state = object_semantic_commit_state(obj)
        assert state["semantic_state"] == "provisional"
        assert state["posterior_label"] == "blanket"
        assert state["committed_label"] == ""
        assert preferred_object_semantic_label(obj) == ""

        second_patch = Patch3D(
            patch_id=2,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=12,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.88,
                "anchor_view_quality": 0.85,
                "anchor_label_strength": "strong",
            },
        )
        accumulate_anchor_semantic_vote(obj, second_patch)

        state = object_semantic_commit_state(obj)
        assert state["semantic_state"] == "committed"
        assert state["posterior_label"] == "blanket"
        assert state["committed_label"] == "blanket"
        assert preferred_object_semantic_label(obj) == "blanket"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_anchor_semantic_posterior_requires_multiframe_commit -q
```

Expected: FAIL with `ImportError` for `object_semantic_commit_state` or assertion that `preferred_object_semantic_label()` returns `blanket` too early.

- [ ] **Step 3: Add commit constants and state helper**

In `src/modules/semantic_memory.py`, add these constants below `ANCHOR_HIGH_VIEW_QUALITY_THRESHOLD`:

```python
ANCHOR_COMMIT_MIN_FRAME_HITS = 2
ANCHOR_COMMIT_MIN_HIGH_QUALITY_HITS = 1
ANCHOR_COMMIT_MIN_WEIGHTED_SCORE = 1.0
ANCHOR_COMMIT_MIN_SCORE_MARGIN = 0.15
ANCHOR_PROVISIONAL_EXPORT_FALLBACK = False
```

Then add this helper after `preferred_object_semantic_label()`:

```python
def object_semantic_commit_state(obj: ObjectMap) -> dict[str, Any]:
    """Return export-facing semantic state for an object."""
    anchor_state = obj.debug.get("anchor_semantics", {})
    if not isinstance(anchor_state, dict):
        return {
            "semantic_state": "unlabeled",
            "posterior_label": "",
            "committed_label": "",
            "posterior_score": 0.0,
            "commit_reason": "no_anchor_semantics",
        }
    posterior_label = str(anchor_state.get("canonical_label", "")).strip()
    committed_label = str(anchor_state.get("committed_label", "")).strip()
    semantic_state = str(anchor_state.get("semantic_state", "")).strip()
    if not semantic_state:
        semantic_state = "committed" if committed_label else ("provisional" if posterior_label else "unlabeled")
    return {
        "semantic_state": semantic_state,
        "posterior_label": posterior_label,
        "committed_label": committed_label,
        "posterior_score": float(anchor_state.get("canonical_score", 0.0)),
        "commit_reason": str(anchor_state.get("commit_reason", "")),
    }
```

- [ ] **Step 4: Update `preferred_object_semantic_label()`**

Replace the anchor-state block in `preferred_object_semantic_label()` with:

```python
    anchor_state = obj.debug.get("anchor_semantics", {})
    if isinstance(anchor_state, dict):
        committed_label = str(anchor_state.get("committed_label", "")).strip()
        if committed_label:
            return committed_label
        if ANCHOR_PROVISIONAL_EXPORT_FALLBACK:
            canonical_label = str(anchor_state.get("canonical_label", "")).strip()
            if canonical_label:
                return canonical_label
```

- [ ] **Step 5: Add commit recomputation to `_recompute_anchor_semantic_state()`**

At the end of `_recompute_anchor_semantic_state()`, immediately after setting `state["source"]`, add:

```python
    if not best_label:
        state["semantic_state"] = "unlabeled"
        state["committed_label"] = ""
        state["commit_reason"] = "no_valid_evidence"
        return

    sorted_scores = sorted(
        ((label, float(weighted_score.get(label, 0.0))) for label in weighted_score),
        key=lambda item: (-item[1], item[0]),
    )
    runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = float(weighted_score.get(best_label, 0.0) - runner_up_score)
    frame_hit_count = int(frame_hits.get(best_label, 0))
    high_quality_count = int(high_quality_hits.get(best_label, 0))
    enough_frames = frame_hit_count >= ANCHOR_COMMIT_MIN_FRAME_HITS
    enough_quality = high_quality_count >= ANCHOR_COMMIT_MIN_HIGH_QUALITY_HITS
    enough_score = float(weighted_score.get(best_label, 0.0)) >= ANCHOR_COMMIT_MIN_WEIGHTED_SCORE
    enough_margin = margin >= ANCHOR_COMMIT_MIN_SCORE_MARGIN
    if enough_frames and enough_quality and enough_score and enough_margin:
        previous_committed = str(state.get("committed_label", ""))
        state["semantic_state"] = "committed"
        state["committed_label"] = best_label
        state["commit_reason"] = "multiframe_strong_anchor_evidence"
        if previous_committed and previous_committed != best_label:
            state.setdefault("relabel_events", []).append(
                {
                    "frame_id": int(max((item.get("frame_id", 0) for item in state.get("evidence", []) if isinstance(item, dict)), default=0)),
                    "old_label": previous_committed,
                    "new_label": best_label,
                    "reason": "committed_label_changed_by_stronger_posterior",
                }
            )
    else:
        state["semantic_state"] = "provisional"
        state["committed_label"] = ""
        state["commit_reason"] = (
            f"waiting_for_commit:frames={frame_hit_count},high_quality={high_quality_count},"
            f"score={float(weighted_score.get(best_label, 0.0)):.3f},margin={margin:.3f}"
        )
```

- [ ] **Step 6: Run the posterior test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_anchor_semantic_posterior_requires_multiframe_commit -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/modules/semantic_memory.py tests/test_pipeline.py
git commit -m "feat: delay semantic label commitment"
```

---

### Task 2: Record Weak And Contextual Evidence Without Committing It

**Files:**
- Modify: `src/modules/semantic_memory.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing weak evidence ledger test**

Add this test after `test_anchor_semantic_vote_ignores_blocked_and_weak_overlap_labels`:

```python
    def test_weak_contextual_anchor_evidence_is_recorded_but_not_committed(self):
        from src.modules.semantic_memory import object_semantic_commit_state

        obj = ObjectMap(object_id=40)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        weak_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.99,
                "anchor_label_strength": "weak_overlap",
            },
        )
        strong_patch = Patch3D(
            patch_id=2,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "blanket",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
        )

        accumulate_anchor_semantic_vote(obj, weak_patch)
        accumulate_anchor_semantic_vote(obj, strong_patch)

        anchor_state = obj.debug["anchor_semantics"]
        assert anchor_state["contextual_label_score_sum"] == {"sofa": 0.99}
        assert anchor_state["ignored_observation_reasons"] == {"non_strong_anchor_label": 1}
        assert anchor_state["label_weighted_score"] == {"blanket": 0.84175}
        assert object_semantic_commit_state(obj)["semantic_state"] == "provisional"
        assert preferred_object_semantic_label(obj) == ""
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_weak_contextual_anchor_evidence_is_recorded_but_not_committed -q
```

Expected: FAIL because `contextual_label_score_sum` is missing.

- [ ] **Step 3: Add contextual ledger initialization**

In `_anchor_semantic_state()`, after `label_high_quality_hits` initialization, add:

```python
    if not isinstance(state.get("contextual_label_score_sum"), dict):
        state["contextual_label_score_sum"] = {}
    if not isinstance(state.get("contextual_evidence"), list):
        state["contextual_evidence"] = []
```

- [ ] **Step 4: Add contextual evidence recorder**

Add this helper before `_record_ignored_anchor_observation()`:

```python
def _record_contextual_anchor_observation(
    obj: ObjectMap,
    *,
    label: str,
    confidence: float,
    frame_id: int,
    strength: str,
) -> None:
    state = _anchor_semantic_state(obj)
    score_sum = state["contextual_label_score_sum"]
    score_sum[label] = float(score_sum.get(label, 0.0) + confidence)
    state["contextual_evidence"].append(
        {
            "label": str(label),
            "confidence": float(confidence),
            "frame_id": int(frame_id),
            "strength": str(strength),
            "commit_eligible": False,
        }
    )
```

- [ ] **Step 5: Update non-strong handling in `accumulate_anchor_semantic_vote()`**

Replace:

```python
    if strength != "strong":
        _record_ignored_anchor_observation(obj, "non_strong_anchor_label")
        return
    confidence = float(patch.metadata.get("anchor_confidence", 0.0))
```

with:

```python
    confidence = float(patch.metadata.get("anchor_confidence", 0.0))
    if strength != "strong":
        _record_contextual_anchor_observation(
            obj,
            label=label,
            confidence=confidence,
            frame_id=int(patch.source_frame_id),
            strength=strength,
        )
        _record_ignored_anchor_observation(obj, "non_strong_anchor_label")
        return
```

- [ ] **Step 6: Preserve contextual fields in `object_update._refresh_object_debug()`**

In `src/modules/object_update.py`, inside the rebuilt `obj.debug["anchor_semantics"]` dict, add:

```python
                "semantic_state": str(anchor_state.get("semantic_state", "")),
                "committed_label": str(anchor_state.get("committed_label", "")),
                "commit_reason": str(anchor_state.get("commit_reason", "")),
                "contextual_label_score_sum": {
                    str(label): float(score)
                    for label, score in (anchor_state.get("contextual_label_score_sum", {}) or {}).items()
                },
                "contextual_evidence": list(anchor_state.get("contextual_evidence", []) or []),
```

Place these next to the existing `canonical_label` and score fields.

- [ ] **Step 7: Run focused tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_weak_contextual_anchor_evidence_is_recorded_but_not_committed tests/test_pipeline.py::TestRoadmapRefactor::test_anchor_semantic_vote_ignores_blocked_and_weak_overlap_labels -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/modules/semantic_memory.py src/modules/object_update.py tests/test_pipeline.py
git commit -m "feat: retain contextual semantic evidence separately"
```

---

### Task 3: Make Observation Identity Use Committed Labels Only

**Files:**
- Modify: `src/modules/observation_identity.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing uncommitted-parent identity test**

Add this test near the existing observation identity tests:

```python
    def test_observation_identity_does_not_block_against_uncommitted_parent_label(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=7,
            points=np.zeros((4, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.90},
        )
        obj = ObjectMap(object_id=2)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "semantic_state": "provisional",
            "committed_label": "",
            "label_max_confidence": {"sofa": 0.95},
        }

        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

        assert decision.can_update is True
        assert decision.relation == "uncommitted_object"
        assert decision.patch_label == "blanket"
        assert decision.object_label == ""
```

- [ ] **Step 2: Write committed-parent blocking test**

Add this test immediately after the uncommitted-parent test:

```python
    def test_observation_identity_blocks_against_committed_parent_label(self):
        from src.modules.observation_identity import classify_observation_identity

        patch = Patch3D(
            patch_id=8,
            points=np.zeros((4, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox_min=np.zeros(3, dtype=np.float32),
            bbox_max=np.ones(3, dtype=np.float32),
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.90},
        )
        obj = ObjectMap(object_id=3)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "sofa",
            "semantic_state": "committed",
            "committed_label": "sofa",
            "label_max_confidence": {"sofa": 0.95},
        }

        decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

        assert decision.can_update is False
        assert decision.relation == "cross_label_contested"
        assert decision.patch_label == "blanket"
        assert decision.object_label == "sofa"
        assert decision.reason == "cross_label_observation_identity"
```

- [ ] **Step 3: Run the failing tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_does_not_block_against_uncommitted_parent_label tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_blocks_against_committed_parent_label -q
```

Expected: first test FAILS because the current helper uses `preferred_object_semantic_label()` and sees provisional labels too early or returns a relation other than `uncommitted_object`.

- [ ] **Step 4: Update `object_identity_label()`**

In `src/modules/observation_identity.py`, change imports to:

```python
from src.modules.semantic_memory import object_semantic_commit_state
```

Replace `object_identity_label()` with:

```python
def object_identity_label(obj: ObjectMap) -> str:
    """Return committed semantic identity used for association blocking."""
    commit_state = object_semantic_commit_state(obj)
    if str(commit_state.get("semantic_state", "")) != "committed":
        return ""
    return normalize_label(commit_state.get("committed_label", ""))
```

- [ ] **Step 5: Update uncommitted relation in `classify_observation_identity()`**

Replace the `if not object_label:` return branch with:

```python
    if not object_label:
        relation = "uncommitted_object" if object_semantic_commit_state(obj)["posterior_label"] else "unlabeled_object"
        return ObservationIdentityDecision(
            can_update=True,
            relation=relation,
            patch_label=patch_label,
            object_label="",
            patch_confidence=patch_confidence,
        )
```

- [ ] **Step 6: Run focused association identity tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_does_not_block_against_uncommitted_parent_label tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_blocks_against_committed_parent_label -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/modules/observation_identity.py tests/test_pipeline.py
git commit -m "feat: use committed semantics for identity blocking"
```

---

### Task 4: Preserve Delayed Semantics Through Provisional Promotion

**Files:**
- Modify: `src/modules/object_update.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing provisional promotion test**

Add this test near the existing provisional/object update tests:

```python
    def test_promoted_provisional_requires_semantic_commit_before_export_label(self):
        updater = ObjectUpdateModule(
            {
                "provisional_pool": {
                    "enabled": True,
                    "promotion_hits": 2,
                    "match_distance": 1.0,
                    "downsample_voxel_size": 0.01,
                    "max_points_per_object": 1000,
                }
            }
        )
        state = SystemState()
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        patch1 = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.90,
                "anchor_view_quality": 0.80,
                "anchor_label_strength": "strong",
            },
        )
        patch2 = Patch3D(
            patch_id=2,
            points=points + np.array([0.02, 0.0, 0.0], dtype=np.float32),
            centroid=(points + np.array([0.02, 0.0, 0.0], dtype=np.float32)).mean(axis=0),
            bbox_min=(points + np.array([0.02, 0.0, 0.0], dtype=np.float32)).min(axis=0),
            bbox_max=(points + np.array([0.02, 0.0, 0.0], dtype=np.float32)).max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.85,
                "anchor_label_strength": "strong",
            },
        )

        updater._upsert_provisional_object(state, patch1)
        updater._upsert_provisional_object(state, patch2)
        updater._promote_stable_provisionals(state)

        assert len(state.objects) == 1
        obj = next(iter(state.objects.values()))
        assert obj.debug["anchor_semantics"]["semantic_state"] == "committed"
        assert obj.debug["anchor_semantics"]["committed_label"] == "rug"
        assert preferred_object_semantic_label(obj) == "rug"
```

- [ ] **Step 2: Run the test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_promoted_provisional_requires_semantic_commit_before_export_label -q
```

Expected: PASS if Task 1 rebuild logic already works. If it fails because debug refresh drops semantic state, continue with implementation.

- [ ] **Step 3: Write contested residual promotion label test**

Add this test immediately after the prior test:

```python
    def test_promoted_contested_residual_keeps_parent_context_out_of_commit(self):
        updater = ObjectUpdateModule(
            {
                "provisional_pool": {
                    "enabled": True,
                    "promotion_hits": 2,
                    "match_distance": 1.0,
                    "downsample_voxel_size": 0.01,
                    "max_points_per_object": 1000,
                },
                "contested_residual_pool": {"enabled": True},
            }
        )
        state = SystemState()
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        for frame_id in [1, 2]:
            patch = Patch3D(
                patch_id=frame_id,
                points=points + np.array([0.01 * frame_id, 0.0, 0.0], dtype=np.float32),
                centroid=(points + np.array([0.01 * frame_id, 0.0, 0.0], dtype=np.float32)).mean(axis=0),
                bbox_min=(points + np.array([0.01 * frame_id, 0.0, 0.0], dtype=np.float32)).min(axis=0),
                bbox_max=(points + np.array([0.01 * frame_id, 0.0, 0.0], dtype=np.float32)).max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "blanket",
                    "anchor_confidence": 0.92,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                    "contested_parent_object_id": 15,
                    "contested_parent_label": "sofa",
                    "contested_patch_label": "blanket",
                    "contested_reason": "cross_label_observation_identity",
                },
            )
            updater._upsert_provisional_object(state, patch, contested=True)

        updater._promote_stable_provisionals(state)

        obj = next(iter(state.objects.values()))
        assert obj.debug["promoted_contested_residual"]["parent_label"] == "sofa"
        assert obj.debug["anchor_semantics"]["contextual_label_score_sum"] == {}
        assert obj.debug["anchor_semantics"]["committed_label"] == "blanket"
        assert preferred_object_semantic_label(obj) == "blanket"
```

- [ ] **Step 4: Run promotion tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_promoted_provisional_requires_semantic_commit_before_export_label tests/test_pipeline.py::TestRoadmapRefactor::test_promoted_contested_residual_keeps_parent_context_out_of_commit -q
```

Expected: PASS after Tasks 1-2. If contextual debug fields are dropped, update `_refresh_object_debug()` as described in Task 2.

- [ ] **Step 5: Commit**

```bash
git add src/modules/object_update.py tests/test_pipeline.py
git commit -m "test: cover delayed semantics in provisional promotion"
```

---

### Task 5: Add Configurable Commit Thresholds

**Files:**
- Modify: `src/modules/semantic_memory.py`
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `configs/default.yaml`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing config update test**

Add this test near semantic memory tests:

```python
    def test_semantic_memory_module_applies_anchor_commit_config(self):
        module = SemanticMemoryModule(
            {
                "backend": "placeholder",
                "anchor_commit": {
                    "min_frame_hits": 3,
                    "min_high_quality_hits": 2,
                    "min_weighted_score": 2.5,
                    "min_score_margin": 0.4,
                    "provisional_export_fallback": True,
                },
            }
        )

        assert module.anchor_commit_min_frame_hits == 3
        assert module.anchor_commit_min_high_quality_hits == 2
        assert module.anchor_commit_min_weighted_score == 2.5
        assert module.anchor_commit_min_score_margin == 0.4
        assert module.anchor_provisional_export_fallback is True
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_semantic_memory_module_applies_anchor_commit_config -q
```

Expected: FAIL because module attributes do not exist.

- [ ] **Step 3: Add module attributes**

In `SemanticMemoryModule.__init__`, after `self.min_stability_score = ...`, add:

```python
        anchor_commit_cfg = config.get("anchor_commit", {})
        self.anchor_commit_min_frame_hits = int(
            anchor_commit_cfg.get("min_frame_hits", ANCHOR_COMMIT_MIN_FRAME_HITS)
        )
        self.anchor_commit_min_high_quality_hits = int(
            anchor_commit_cfg.get("min_high_quality_hits", ANCHOR_COMMIT_MIN_HIGH_QUALITY_HITS)
        )
        self.anchor_commit_min_weighted_score = float(
            anchor_commit_cfg.get("min_weighted_score", ANCHOR_COMMIT_MIN_WEIGHTED_SCORE)
        )
        self.anchor_commit_min_score_margin = float(
            anchor_commit_cfg.get("min_score_margin", ANCHOR_COMMIT_MIN_SCORE_MARGIN)
        )
        self.anchor_provisional_export_fallback = bool(
            anchor_commit_cfg.get("provisional_export_fallback", ANCHOR_PROVISIONAL_EXPORT_FALLBACK)
        )
```

This test only verifies config parsing. Do not change global runtime behavior in this task.

- [ ] **Step 4: Add config keys**

In both `configs/default.yaml` and `configs/room0_surface_gate_4090.yaml`, under `semantic_memory`, add:

```yaml
  anchor_commit:
    min_frame_hits: 2
    min_high_quality_hits: 1
    min_weighted_score: 1.0
    min_score_margin: 0.15
    provisional_export_fallback: false
```

- [ ] **Step 5: Run config test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_semantic_memory_module_applies_anchor_commit_config -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/modules/semantic_memory.py configs/default.yaml configs/room0_surface_gate_4090.yaml tests/test_pipeline.py
git commit -m "feat: configure semantic commit thresholds"
```

---

### Task 6: Surface Semantic Commit State In Audit And Reports

**Files:**
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing final audit field test**

Add this test near existing final object semantic audit tests:

```python
    def test_final_object_semantic_audit_includes_semantic_commit_state(self):
        obj = ObjectMap(object_id=5)
        obj.local_pcd = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        obj.debug["anchor_semantics"] = {
            "canonical_label": "rug",
            "canonical_score": 1.25,
            "semantic_state": "committed",
            "committed_label": "rug",
            "commit_reason": "multiframe_strong_anchor_evidence",
            "label_weighted_score": {"rug": 1.25},
            "contextual_label_score_sum": {"sofa": 0.8},
        }
        state = SystemState(objects={5: obj})

        audit = build_final_object_semantic_audit(
            state=state,
            pred_to_gt_vertex_indices=np.array([], dtype=np.int64),
            pred_object_ids=np.array([], dtype=np.int64),
            gt_vertex_class_ids=np.array([], dtype=np.int64),
            class_id_to_name={98: "rug"},
            per_class_metrics=[],
        )

        record = audit["objects"][0]
        assert record["semantic_state"] == "committed"
        assert record["committed_label"] == "rug"
        assert record["posterior_label"] == "rug"
        assert record["commit_reason"] == "multiframe_strong_anchor_evidence"
        assert record["contextual_label_score_sum"] == {"sofa": 0.8}
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_final_object_semantic_audit_includes_semantic_commit_state -q
```

Expected: FAIL because those fields are missing.

- [ ] **Step 3: Import commit helper in `run_room0_full_eval.py`**

Change the import:

```python
from src.modules.semantic_memory import preferred_object_semantic_label
```

to:

```python
from src.modules.semantic_memory import object_semantic_commit_state, preferred_object_semantic_label
```

- [ ] **Step 4: Add fields in final object audit**

Inside `build_final_object_semantic_audit()`, where each object record is built, compute:

```python
        semantic_commit = object_semantic_commit_state(obj)
```

Then add these fields to the object record:

```python
            "semantic_state": str(semantic_commit["semantic_state"]),
            "committed_label": str(semantic_commit["committed_label"]),
            "posterior_label": str(semantic_commit["posterior_label"]),
            "commit_reason": str(semantic_commit["commit_reason"]),
            "contextual_label_score_sum": {
                str(label): float(score)
                for label, score in (anchor_semantics.get("contextual_label_score_sum", {}) or {}).items()
            },
```

- [ ] **Step 5: Add audit summary counts**

After object records are built and before returning the audit dict, add:

```python
    semantic_state_counts: dict[str, int] = {}
    for record in object_records:
        semantic_state = str(record.get("semantic_state", "unlabeled"))
        semantic_state_counts[semantic_state] = int(semantic_state_counts.get(semantic_state, 0) + 1)
```

Then include it at the top level:

```python
        "semantic_state_counts": semantic_state_counts,
```

- [ ] **Step 6: Run audit test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_final_object_semantic_audit_includes_semantic_commit_state -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add run_room0_full_eval.py tests/test_pipeline.py
git commit -m "feat: audit semantic commit state"
```

---

### Task 7: Regression Scenario For Sofa-Blanket And Rug-Floor Separation

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write association regression test for committed sofa vs blanket patch**

Add this test near association tests:

```python
    def test_committed_parent_does_not_absorb_confident_child_patch(self):
        sofa = ObjectMap(
            object_id=15,
            local_pcd=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            centroid=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.array([0.0, -0.1, -0.1], dtype=np.float32),
            bbox_max=np.array([0.3, 0.1, 0.1], dtype=np.float32),
        )
        sofa.debug["anchor_semantics"] = {
            "semantic_state": "committed",
            "committed_label": "sofa",
            "canonical_label": "sofa",
            "label_max_confidence": {"sofa": 0.95},
        }
        patch = Patch3D(
            patch_id=8,
            points=np.array([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0]], dtype=np.float32),
            centroid=np.array([0.055, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.array([0.05, -0.01, -0.01], dtype=np.float32),
            bbox_max=np.array([0.06, 0.01, 0.01], dtype=np.float32),
            source_frame_id=20,
            metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.91},
        )
        assoc = AssociationModule(
            {
                "match_threshold": 0.1,
                "observation_identity_gate_enabled": True,
                "observation_identity_min_patch_confidence": 0.35,
            }
        )

        result = assoc.process([patch], {15: sofa}, TSDFInstanceVolume(voxel_size=0.05))

        assert result.matched == []
        assert result.contested_object_patches == [8]
        assert result.contested_matches[0].blocked_object_id == 15
        assert result.contested_matches[0].patch_label == "blanket"
        assert result.contested_matches[0].object_label == "sofa"
```

- [ ] **Step 2: Write uncommitted rug/floor geometry recall test**

Add this test immediately after the sofa-blanket test:

```python
    def test_uncommitted_floor_like_parent_does_not_block_rug_geometry_recall(self):
        floor_like = ObjectMap(
            object_id=41,
            local_pcd=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            centroid=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.array([0.0, -0.1, -0.1], dtype=np.float32),
            bbox_max=np.array([0.3, 0.1, 0.1], dtype=np.float32),
        )
        floor_like.debug["anchor_semantics"] = {
            "semantic_state": "provisional",
            "committed_label": "",
            "canonical_label": "floor",
            "label_max_confidence": {"floor": 0.80},
        }
        patch = Patch3D(
            patch_id=9,
            points=np.array([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0]], dtype=np.float32),
            centroid=np.array([0.055, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.array([0.05, -0.01, -0.01], dtype=np.float32),
            bbox_max=np.array([0.06, 0.01, 0.01], dtype=np.float32),
            source_frame_id=21,
            metadata={"anchor_class_name": "rug", "anchor_confidence": 0.88},
        )
        assoc = AssociationModule(
            {
                "match_threshold": 0.1,
                "observation_identity_gate_enabled": True,
                "observation_identity_min_patch_confidence": 0.35,
            }
        )

        result = assoc.process([patch], {41: floor_like}, TSDFInstanceVolume(voxel_size=0.05))

        assert len(result.matched) == 1
        assert result.matched[0][1] == 41
        assert result.contested_object_patches == []
```

- [ ] **Step 3: Run regression tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_committed_parent_does_not_absorb_confident_child_patch tests/test_pipeline.py::TestRoadmapRefactor::test_uncommitted_floor_like_parent_does_not_block_rug_geometry_recall -q
```

Expected: PASS after Task 3.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: cover semantic-late association regressions"
```

---

### Task 8: Verification And Smoke Experiment

**Files:**
- No source files expected.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor -q
```

Expected: PASS.

- [ ] **Step 2: Run frontend/object anchor tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 3: Run a stride=10 200f smoke experiment**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
experiment=20260528_room0_geometry_first_semantic_late_stride10_200f
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_surface_gate_4090.yaml \
  --experiment-name "${experiment}" \
  --num-frames 200 \
  --frame-stride 10 \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --output-root /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation \
  --proposal-backend precomputed \
  --proposal-cache-dir /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2 \
  --proposal-cache-manifest /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json
```

Expected:

- Exit status 0.
- `run_report.json` exists.
- `final_object_semantic_audit.json` contains `semantic_state_counts`.
- `rug` IoU should not collapse below the 0527 stride10 200f baseline by more than 0.03.
- `final_object_count` should remain in a controlled range, initially expected between 120 and 230 for stride10 200f.

- [ ] **Step 4: Compare against previous 200f baselines**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
base = Path('/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation')
names = [
    '20260526_observation_first_fix52e1714_200f',
    '20260527_room0_contained_gate_stride10_200f',
    '20260528_room0_geometry_first_semantic_late_stride10_200f',
]
for name in names:
    candidates = list(base.glob(f'{name}/room0/run_report.json'))
    if not candidates:
        candidates = list(base.glob(f'*{name}*/room0/run_report.json'))
    if not candidates:
        print(name, 'missing')
        continue
    report = json.loads(candidates[0].read_text())
    per_class = {item['class_name']: item for item in report['per_class']}
    print(name)
    print('  miou', round(report['miou'], 4), 'fmiou', round(report['fmiou'], 4), 'objects', report['final_object_count'])
    for cls in ['rug', 'sofa', 'floor', 'wall', 'blinds', 'window', 'blanket', 'book']:
        item = per_class.get(cls)
        if item:
            print(' ', cls, 'iou', round(item['iou'], 4), 'acc', round(item['acc'], 4))
PY
```

Expected: New smoke result should preserve 0527's reduced object count tendency while recovering some 0526 recall, especially for `rug`, `floor`, and `wall`.

- [ ] **Step 5: Commit verification note**

If the smoke experiment is acceptable, update this plan with a short result note under this task and commit:

```bash
git add docs/superpowers/plans/2026-05-28-geometry-first-semantic-late.md
git commit -m "docs: record semantic-late smoke results"
```

---

## Self-Review Checklist

- Spec coverage:
  - Geometry-first recall is preserved through existing observation-first/provisional paths and Task 3's uncommitted-object update rule.
  - Semantic-late commitment is implemented by Tasks 1, 2, and 5.
  - Cross-label contamination prevention is preserved by Task 3 and Task 7.
  - Audit/evaluation visibility is covered by Task 6.

- Placeholder scan:
  - No `TBD`, `TODO`, or unspecified implementation placeholders remain.
  - Each task has exact file paths, code snippets, commands, and expected outcomes.

- Type consistency:
  - New helper `object_semantic_commit_state(obj)` returns a plain dict used by `observation_identity.py` and `run_room0_full_eval.py`.
  - Existing `ObjectMap.debug["anchor_semantics"]` remains the storage location.
  - `preferred_object_semantic_label(obj)` remains the export-facing API.

