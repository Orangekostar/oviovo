# Observation-First Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent high-confidence frontend observations such as blanket, cushion, rug, cabinet, and window from being absorbed into geometrically overlapping parent objects when their SAM2 mask and YOLO/anchor semantic identity are already explicit.

**Architecture:** Make the frontend observation an immutable identity fact: SAM2 mask geometry defines the patch extent, while YOLO/anchor vote defines the patch semantic identity. Association may use TSDF and geometry to find spatial support, but a confident cross-label observation must not directly update an existing object's `local_pcd`, TSDF ownership, or anchor semantic votes; it becomes a contested residual/provisional hypothesis until repeated observations promote it. TSDF remains a geometry and visibility substrate, not an authority that can overwrite the frontend label.

**Tech Stack:** Python dataclasses, NumPy point clouds, existing `AssociationModule`, `ObjectUpdateModule`, `TSDFInstanceMapModule`, pytest.

---

## Problem Summary

The 200f room0 run at `outputs/tmp_validation/20260523_room0_frontend_semantic_vote_anchorid_200f` shows that contamination is much lower, but several observed objects disappear or become partially absorbed:

- `blanket` is repeatedly observed but becomes mostly sofa-owned.
- `cushion` on chair/sofa is observed but often becomes chair/sofa-owned.
- `rug` survives mostly only under stool, with wider rug evidence missing.
- `cabinet` is partly swallowed by wall.
- `window` is absorbed by blinds/door/wall.

The frame-1800 local audit pattern is decisive:

- `r23 blanket A8` matched object 31 `sofa` with high voxel/geometry scores.
- `r14 cushion A13` matched object 31 `sofa` with high voxel/geometry scores.
- The frontend had already observed `blanket` and `cushion`; the failure was not "not observed", it was "observed but committed to the wrong identity".

Depth gap is not a reliable separator for attached or same-surface objects. A blanket and sofa, cushion and chair, rug and floor, cabinet and wall, or window and blinds can be near-coplanar or physically touching. The correct invariant is therefore identity-based, not depth-gap-based:

> If the current frame confidently observes a patch as label `X`, and the best geometric candidate is an existing object whose stable label is `Y != X`, the patch must not update object `Y`. It must enter a contested residual/provisional path for label `X`.

## File Structure

- Modify: `src/core/data_structures.py`
  - Add explicit contested-observation data carried by `AssociationResult`.
  - Extend `ProvisionalObject` with optional contested-parent metadata without changing existing behavior for normal new objects.

- Create: `src/modules/observation_identity.py`
  - Centralize label extraction, normalization, confidence handling, and the "may this patch update this object?" decision.
  - Keep thresholds small and semantic, not scattered through association and update code.

- Modify: `src/modules/association.py`
  - Replace the current subregion-only semantic conflict gate with an observation identity gate.
  - Same-label matches keep current geometry behavior.
  - Confident cross-label candidates become `contested_matches`, never `matched`.
  - Low-confidence/unlabeled patches keep conservative legacy behavior.

- Modify: `src/modules/object_update.py`
  - Add a contested residual update path that reuses the provisional pool mechanics while preserving the parent object reference.
  - Ensure contested patches do not update the parent object's `local_pcd`, TSDF owner support, observations, whole evidence, or anchor semantic votes.
  - Promote repeated contested residuals into real objects using the existing provisional promotion path.

- Modify: `src/pipelines/main_pipeline.py`
  - Add per-frame counters for contested observations and residual promotions to `last_frame_debug`.

- Modify: `run_room0_full_eval.py`
  - Add visible outcome fields so local audit can show "observed blanket/cushion was blocked from sofa and stored as residual".

- Modify: `configs/room0_surface_gate_4090.yaml`, `configs/default.yaml`, `configs/midrecall_local_memory_boost.yaml`
  - Add an `observation_identity` association section and a `contested_residual_pool` object-update section.
  - Keep defaults conservative and enabled for the room0 config.

- Modify: `tests/test_pipeline.py`
  - Association and audit tests already live here; add focused regression coverage near existing association/audit tests.

- Modify: `tests/test_provisional_pool.py`
  - Add object-update tests for contested residual accumulation and promotion.

## Core Invariants

1. A matched association may update an existing object's `local_pcd` only when the patch identity is same-label or genuinely unlabeled.
2. A confident cross-label patch may not update the parent object's `local_pcd`.
3. A confident cross-label patch may not call `accumulate_anchor_semantic_vote(parent, patch)`.
4. A confident cross-label patch may not integrate TSDF support under the parent object id.
5. A confident cross-label patch should not be discarded only because it overlaps parent geometry; it enters a residual/provisional identity path.
6. Repeated observations of the same residual identity promote it to a normal object.
7. Absorption is allowed only through negative visibility evidence later, not while the frontend is currently observing the object.

## Task 1: Add Observation Identity Data Structures

**Files:**
- Modify: `src/core/data_structures.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing dataclass test**

Add this near `TestDataStructures.test_association_result` in `tests/test_pipeline.py`:

```python
def test_association_result_records_contested_matches(self):
    from src.core.data_structures import AssociationScore, ContestedAssociation

    score = AssociationScore(total_score=0.91)
    contested = ContestedAssociation(
        patch_id=8,
        blocked_object_id=3,
        patch_label="cushion",
        object_label="sofa",
        reason="cross_label_observation_identity",
        score=score,
    )
    result = AssociationResult(contested_matches=[contested], contested_object_patches=[8])

    assert result.matched == []
    assert result.new_object_patches == []
    assert result.contested_object_patches == [8]
    assert result.contested_matches[0].blocked_object_id == 3
    assert result.contested_matches[0].patch_label == "cushion"
    assert result.contested_matches[0].object_label == "sofa"
    assert result.contested_matches[0].score.total_score == 0.91
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestDataStructures::test_association_result_records_contested_matches -q
```

Expected: FAIL with `ImportError` or `NameError` because `ContestedAssociation` does not exist.

- [ ] **Step 3: Add the dataclasses**

In `src/core/data_structures.py`, add this dataclass immediately after `AssociationScore` and before `AssociationResult`:

```python
@dataclass
class ContestedAssociation:
    """A confident frontend observation blocked from updating a cross-label object.

    The spatial candidate is useful context, but it is not an update target.
    ObjectUpdateModule should route the patch into a residual/provisional identity path.
    """
    patch_id: int
    blocked_object_id: int
    patch_label: str = ""
    object_label: str = ""
    reason: str = "cross_label_observation_identity"
    score: Optional[AssociationScore] = None
    debug: Dict[str, Any] = field(default_factory=dict)
```

Then replace `AssociationResult` with this compatible extension:

```python
@dataclass
class AssociationResult:
    """Result of associating object patches to existing objects.

    Attributes:
        matched: List of (patch_id, object_id, score) for successful same-identity matches.
        new_object_patches: Patch IDs that should create normal new objects.
        contested_object_patches: Patch IDs that were observed as a different confident label
            from their best spatial candidate and must enter residual/provisional tracking.
        contested_matches: Detailed records for blocked cross-label spatial candidates.
        scores: Full scoring details per candidate pair.
    """
    matched: List[tuple] = field(default_factory=list)
    new_object_patches: List[int] = field(default_factory=list)
    contested_object_patches: List[int] = field(default_factory=list)
    contested_matches: List[ContestedAssociation] = field(default_factory=list)
    scores: Dict[tuple, AssociationScore] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run the dataclass test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestDataStructures::test_association_result_records_contested_matches -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/data_structures.py tests/test_pipeline.py
git commit -m "feat: add contested association records"
```

## Task 2: Centralize Observation Identity Rules

**Files:**
- Create: `src/modules/observation_identity.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing helper tests**

Add these tests near the association tests in `TestRoadmapRefactor` in `tests/test_pipeline.py`:

```python
def test_observation_identity_allows_same_label_update(self):
    from src.modules.observation_identity import classify_observation_identity

    patch = Patch3D(
        patch_id=1,
        points=np.zeros((1, 3), dtype=np.float32),
        centroid=np.zeros(3, dtype=np.float32),
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=np.zeros(3, dtype=np.float32),
        metadata={"anchor_class_name": " Cushion ", "anchor_confidence": 0.80},
    )
    obj = ObjectMap(object_id=2)
    obj.debug["anchor_semantics"] = {
        "canonical_label": "cushion",
        "canonical_score": 0.90,
        "label_max_confidence": {"cushion": 0.90},
    }

    decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

    assert decision.can_update is True
    assert decision.relation == "same_label"
    assert decision.patch_label == "cushion"
    assert decision.object_label == "cushion"
```

```python
def test_observation_identity_blocks_confident_cross_label_update(self):
    from src.modules.observation_identity import classify_observation_identity

    patch = Patch3D(
        patch_id=8,
        points=np.zeros((1, 3), dtype=np.float32),
        centroid=np.zeros(3, dtype=np.float32),
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=np.zeros(3, dtype=np.float32),
        metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.82},
    )
    obj = ObjectMap(object_id=31)
    obj.debug["anchor_semantics"] = {
        "canonical_label": "sofa",
        "canonical_score": 0.95,
        "label_max_confidence": {"sofa": 0.95},
    }

    decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

    assert decision.can_update is False
    assert decision.relation == "cross_label_contested"
    assert decision.reason == "cross_label_observation_identity"
    assert decision.patch_label == "blanket"
    assert decision.object_label == "sofa"
```

```python
def test_observation_identity_allows_unlabeled_patch_legacy_update(self):
    from src.modules.observation_identity import classify_observation_identity

    patch = Patch3D(
        patch_id=9,
        points=np.zeros((1, 3), dtype=np.float32),
        centroid=np.zeros(3, dtype=np.float32),
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=np.zeros(3, dtype=np.float32),
        metadata={},
    )
    obj = ObjectMap(object_id=31)
    obj.debug["anchor_semantics"] = {
        "canonical_label": "sofa",
        "canonical_score": 0.95,
        "label_max_confidence": {"sofa": 0.95},
    }

    decision = classify_observation_identity(patch, obj, min_patch_confidence=0.35)

    assert decision.can_update is True
    assert decision.relation == "unlabeled_patch"
```

- [ ] **Step 2: Run the failing helper tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_allows_same_label_update \
  tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_blocks_confident_cross_label_update \
  tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_allows_unlabeled_patch_legacy_update \
  -q
```

Expected: FAIL because `src.modules.observation_identity` does not exist.

- [ ] **Step 3: Implement the helper module**

Create `src/modules/observation_identity.py`:

```python
"""Observation identity rules for frontend-first object tracking."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.data_structures import ObjectMap, Patch3D
from src.modules.semantic_memory import preferred_object_semantic_label


@dataclass(frozen=True)
class ObservationIdentityDecision:
    """Decision for whether a patch may update an existing object."""
    can_update: bool
    relation: str
    patch_label: str = ""
    object_label: str = ""
    patch_confidence: float = 0.0
    reason: str = ""


def normalize_label(label: object) -> str:
    """Normalize frontend/object labels for exact identity comparison."""
    return str(label or "").strip().lower()


def patch_anchor_label(patch: Patch3D) -> str:
    """Return the frontend semantic identity assigned to a patch."""
    return normalize_label(patch.metadata.get("anchor_class_name", ""))


def patch_anchor_confidence(patch: Patch3D) -> float:
    """Return patch anchor confidence with malformed metadata treated as zero."""
    try:
        return float(patch.metadata.get("anchor_confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def object_identity_label(obj: ObjectMap) -> str:
    """Return the stable semantic identity of an object."""
    return normalize_label(preferred_object_semantic_label(obj))


def classify_observation_identity(
    patch: Patch3D,
    obj: ObjectMap,
    *,
    min_patch_confidence: float,
) -> ObservationIdentityDecision:
    """Decide if a patch is allowed to update an object identity.

    Confident cross-label observations are not association updates; they are
    contested residual observations that need their own identity track.
    """
    patch_label = patch_anchor_label(patch)
    patch_confidence = patch_anchor_confidence(patch)
    object_label = object_identity_label(obj)

    if not patch_label:
        return ObservationIdentityDecision(
            can_update=True,
            relation="unlabeled_patch",
            patch_label="",
            object_label=object_label,
            patch_confidence=patch_confidence,
        )
    if patch_confidence < float(min_patch_confidence):
        return ObservationIdentityDecision(
            can_update=True,
            relation="low_confidence_patch",
            patch_label=patch_label,
            object_label=object_label,
            patch_confidence=patch_confidence,
        )
    if not object_label:
        return ObservationIdentityDecision(
            can_update=True,
            relation="unlabeled_object",
            patch_label=patch_label,
            object_label="",
            patch_confidence=patch_confidence,
        )
    if patch_label == object_label:
        return ObservationIdentityDecision(
            can_update=True,
            relation="same_label",
            patch_label=patch_label,
            object_label=object_label,
            patch_confidence=patch_confidence,
        )
    return ObservationIdentityDecision(
        can_update=False,
        relation="cross_label_contested",
        patch_label=patch_label,
        object_label=object_label,
        patch_confidence=patch_confidence,
        reason="cross_label_observation_identity",
    )
```

- [ ] **Step 4: Run the helper tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_allows_same_label_update \
  tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_blocks_confident_cross_label_update \
  tests/test_pipeline.py::TestRoadmapRefactor::test_observation_identity_allows_unlabeled_patch_legacy_update \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/observation_identity.py tests/test_pipeline.py
git commit -m "feat: centralize observation identity rules"
```

## Task 3: Route Cross-Label Association to Contested Residuals

**Files:**
- Modify: `src/modules/association.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing association regression test**

Add this near `TestRoadmapRefactor.test_association_blocks_semantic_conflict_subregion_matches` in `tests/test_pipeline.py`:

```python
def test_association_routes_confident_cross_label_overlap_to_contested_not_matched(self):
    module = AssociationModule(
        {
            "match_threshold": 0.1,
            "observation_identity_gate_enabled": True,
            "observation_identity_min_patch_confidence": 0.35,
        }
    )
    obj_points = np.array(
        [[x * 0.05, y * 0.05, 1.0] for x in range(6) for y in range(6)],
        dtype=np.float32,
    )
    patch_points = obj_points[:12].copy()
    patch = Patch3D(
        patch_id=22,
        points=patch_points,
        centroid=patch_points.mean(axis=0),
        bbox_min=patch_points.min(axis=0),
        bbox_max=patch_points.max(axis=0),
        metadata={"anchor_class_name": "blanket", "anchor_confidence": 0.88},
    )
    sofa = ObjectMap(
        object_id=31,
        local_pcd=obj_points,
        centroid=obj_points.mean(axis=0),
        bbox_min=obj_points.min(axis=0),
        bbox_max=obj_points.max(axis=0),
    )
    sofa.debug["anchor_semantics"] = {
        "canonical_label": "sofa",
        "canonical_score": 0.94,
        "label_max_confidence": {"sofa": 0.94},
    }

    result = module.process([patch], {31: sofa}, SystemState().tsdf_volume)

    assert result.matched == []
    assert result.new_object_patches == []
    assert result.contested_object_patches == [22]
    assert len(result.contested_matches) == 1
    contested = result.contested_matches[0]
    assert contested.patch_id == 22
    assert contested.blocked_object_id == 31
    assert contested.patch_label == "blanket"
    assert contested.object_label == "sofa"
    assert contested.reason == "cross_label_observation_identity"
    assert patch.metadata["contested_parent_object_id"] == 31
    assert patch.metadata["contested_parent_label"] == "sofa"
    assert patch.metadata["contested_patch_label"] == "blanket"
    assert result.debug["per_patch"][22]["final_outcome"] == "contested_residual"
```

- [ ] **Step 2: Run the failing association regression test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_association_routes_confident_cross_label_overlap_to_contested_not_matched -q
```

Expected: FAIL because `observation_identity_gate_enabled` is not implemented and `contested_object_patches` is not populated.

- [ ] **Step 3: Import the new types and helper**

In `src/modules/association.py`, extend imports:

```python
from src.core.data_structures import (
    ActiveSet,
    AssociationResult,
    AssociationScore,
    ContestedAssociation,
    ObjectMap,
    Patch3D,
    TSDFInstanceVolume,
    VoxelVoteResult,
)
from src.modules.observation_identity import classify_observation_identity
```

- [ ] **Step 4: Add config knobs**

In `AssociationModule.__init__()`, after `self.match_threshold`:

```python
self.observation_identity_gate_enabled = bool(
    config.get("observation_identity_gate_enabled", config.get("semantic_conflict_gate_enabled", False))
)
self.observation_identity_min_patch_confidence = float(
    config.get(
        "observation_identity_min_patch_confidence",
        config.get("semantic_conflict_min_patch_confidence", 0.35),
    )
)
```

- [ ] **Step 5: Replace candidate blocking logic with identity-first contested routing**

In `AssociationModule.process()`, replace the block beginning with:

```python
if score.total_score > self.match_threshold:
    conflict_reason = self._semantic_conflict_reason(patch, obj)
```

with:

```python
if score.total_score > self.match_threshold:
    identity_decision = classify_observation_identity(
        patch,
        obj,
        min_patch_confidence=self.observation_identity_min_patch_confidence,
    )
    if self.observation_identity_gate_enabled and not identity_decision.can_update:
        blocked_candidates.append(
            {
                "object_id": int(obj_id),
                "score": float(score.total_score),
                "reason": identity_decision.reason,
                "patch_label": identity_decision.patch_label,
                "object_label": identity_decision.object_label,
                "relation": identity_decision.relation,
                "patch_confidence": float(identity_decision.patch_confidence),
            }
        )
        continue
    if best_score is None or score.total_score > best_score.total_score:
        best_score = score
        best_obj_id = obj_id
```

Keep `_semantic_conflict_reason()` for now so older tests can be updated or removed later, but do not call it in the main path after this step.

- [ ] **Step 6: Emit contested results instead of normal new-object requests**

Replace the `else:` branch that currently appends `result.new_object_patches` when `blocked_candidates` exists with:

```python
else:
    if blocked_candidates:
        best_blocked = max(blocked_candidates, key=lambda item: float(item["score"]))
        parent_id = int(best_blocked["object_id"])
        patch_label = str(best_blocked["patch_label"])
        object_label = str(best_blocked["object_label"])
        reason = str(best_blocked["reason"])
        patch.metadata["contested_parent_object_id"] = parent_id
        patch.metadata["contested_parent_label"] = object_label
        patch.metadata["contested_patch_label"] = patch_label
        patch.metadata["contested_reason"] = reason
        patch.metadata["semantic_split_candidate_from_object_id"] = parent_id
        patch.metadata["semantic_split_candidate_parent_label"] = object_label
        patch.metadata["semantic_split_candidate_new_label"] = patch_label
        patch.metadata["semantic_split_candidate_reason"] = reason
        result.contested_object_patches.append(patch.patch_id)
        result.contested_matches.append(
            ContestedAssociation(
                patch_id=int(patch.patch_id),
                blocked_object_id=parent_id,
                patch_label=patch_label,
                object_label=object_label,
                reason=reason,
                score=result.scores.get((patch.patch_id, parent_id)),
                debug=dict(best_blocked),
            )
        )
        patch_debug["selected_object_id"] = None
        patch_debug["blocked_object_id"] = parent_id
        patch_debug["final_association_score"] = float(best_blocked["score"])
        patch_debug["new_instance_created"] = False
        patch_debug["final_outcome"] = "contested_residual"
    else:
        result.new_object_patches.append(patch.patch_id)
        vote.new_instance_created = True
        patch_debug["selected_object_id"] = None
        patch_debug["final_association_score"] = 0.0
        patch_debug["new_instance_created"] = True
        patch_debug["final_outcome"] = "new_object"
```

In the successful match branch, add:

```python
patch_debug["final_outcome"] = "matched"
```

- [ ] **Step 7: Update the old semantic conflict test expectation**

In `test_association_blocks_semantic_conflict_subregion_matches`, update the module config to include:

```python
"observation_identity_gate_enabled": True,
"observation_identity_min_patch_confidence": 0.35,
```

Then change the assertions from:

```python
assert result.new_object_patches == [8]
...
blocked = result.debug["per_patch"][8]["semantic_conflict_blocked_candidates"]
assert blocked[0]["reason"] == "semantic_conflict_subregion"
```

to:

```python
assert result.new_object_patches == []
assert result.contested_object_patches == [8]
assert patch.metadata["semantic_split_candidate_from_object_id"] == 3
assert patch.metadata["semantic_split_candidate_parent_label"] == "sofa"
assert patch.metadata["semantic_split_candidate_new_label"] == "cushion"
blocked = result.debug["per_patch"][8]["semantic_conflict_blocked_candidates"]
assert blocked[0]["reason"] == "cross_label_observation_identity"
assert result.debug["per_patch"][8]["final_outcome"] == "contested_residual"
```

- [ ] **Step 8: Run focused association tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_routes_confident_cross_label_overlap_to_contested_not_matched \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_blocks_semantic_conflict_subregion_matches \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_restricts_candidates_to_active_set \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_with_empty_active_set_does_not_fallback_to_global_objects \
  -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/modules/association.py tests/test_pipeline.py
git commit -m "feat: route cross-label observations to residuals"
```

## Task 4: Prevent Contested Patches from Updating Parent Objects

**Files:**
- Modify: `src/modules/object_update.py`
- Test: `tests/test_provisional_pool.py`

- [ ] **Step 1: Write the failing no-parent-update test**

Add this to `tests/test_provisional_pool.py` after `test_provisional_pool_records_semantic_split_candidates`:

```python
def test_contested_patch_does_not_update_parent_object_memory_or_votes() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {"enabled": True, "promotion_hits": 3, "max_idle_frames": 30},
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    sofa_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    sofa = ObjectMap(
        object_id=31,
        local_pcd=sofa_points.copy(),
        centroid=sofa_points.mean(axis=0),
        bbox_min=sofa_points.min(axis=0),
        bbox_max=sofa_points.max(axis=0),
    )
    sofa.debug["anchor_semantics"] = {
        "canonical_label": "sofa",
        "canonical_score": 1.0,
        "label_votes": {"sofa": 1.0},
        "label_max_confidence": {"sofa": 0.95},
    }
    blanket_patch = _make_patch(patch_id=22, frame_id=8)
    blanket_patch.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.88,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "blanket",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    state = SystemState(objects={31: sofa}, next_object_id=32)
    association = AssociationResult(contested_object_patches=[22])

    updated = module.process(association, [blanket_patch], state)

    assert np.array_equal(updated.objects[31].local_pcd, sofa_points)
    assert updated.objects[31].update_count == 0
    assert updated.objects[31].observations == []
    assert updated.objects[31].debug["anchor_semantics"]["label_votes"] == {"sofa": 1.0}
    assert len(updated.provisional_objects) == 1
    residual = next(iter(updated.provisional_objects.values()))
    assert residual.anchor_class_name == "blanket"
    assert residual.debug["contested_parent_object_id"] == 31
    assert residual.debug["contested_parent_label"] == "sofa"
    assert residual.debug["contested_patch_label"] == "blanket"
```

- [ ] **Step 2: Run the failing no-parent-update test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_provisional_pool.py::test_contested_patch_does_not_update_parent_object_memory_or_votes -q
```

Expected: FAIL because `contested_object_patches` are ignored by `ObjectUpdateModule.process()`.

- [ ] **Step 3: Add object-update config and debug counters**

In `ObjectUpdateModule.__init__()`, after provisional config:

```python
contested_cfg = config.get("contested_residual_pool", {})
self.contested_residual_enabled = bool(contested_cfg.get("enabled", self.provisional_enabled))
self.last_contested_residual_patch_ids: list[int] = []
self.last_contested_residual_promoted_object_ids: list[int] = []
```

At the beginning of `process()`, after resetting `self.last_created_object_ids`:

```python
self.last_contested_residual_patch_ids = []
self.last_contested_residual_promoted_object_ids = []
```

- [ ] **Step 4: Process contested patches before normal new-object patches**

In `ObjectUpdateModule.process()`, after the matched-object loop and before `# Create new objects or accumulate them in the provisional local pool.`, add:

```python
if self.contested_residual_enabled:
    for patch_id in association.contested_object_patches:
        patch = patch_map.get(patch_id)
        if patch is None:
            continue
        patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
            patch,
            current_depth=current_depth,
            current_pose=current_pose,
            current_intrinsics=current_intrinsics,
        )
        self._record_current_frame_visibility_gate_debug(visibility_debug)
        if patch is None:
            continue
        patch, structural_reject, gate_debug = self._filter_patch_by_surface_owner(
            patch,
            target_object_id=None,
            state=state,
            background_support=background_support,
            new_object=True,
        )
        self._record_surface_gate_debug(gate_debug)
        if structural_reject is not None:
            self.last_structural_reject_patches.append(structural_reject)
        if patch is None:
            continue
        if not self._should_enter_provisional_pool(patch):
            continue
        self._upsert_provisional_object(state, patch, contested=True)
        self.last_contested_residual_patch_ids.append(int(patch_id))
```

Do not integrate the patch into `state.tsdf_volume` here. Integration happens only after promotion under the promoted object's own id.

- [ ] **Step 5: Extend provisional upsert with contested metadata**

Change the signature:

```python
def _upsert_provisional_object(self, state: SystemState, patch: Patch3D, *, contested: bool = False) -> None:
```

When setting `provisional.debug`, add these keys:

```python
"is_contested_residual": bool(contested or patch.metadata.get("contested_parent_object_id", -1) != -1),
"contested_parent_object_id": int(patch.metadata.get("contested_parent_object_id", -1)),
"contested_parent_label": str(patch.metadata.get("contested_parent_label", "")),
"contested_patch_label": str(patch.metadata.get("contested_patch_label", "")),
"contested_reason": str(patch.metadata.get("contested_reason", "")),
```

Keep the existing `semantic_split_candidate_*` keys for backwards-compatible audit/debug output.

- [ ] **Step 6: Preserve contested promotion debug**

In `_promote_provisional_object()`, after `promoted_split_candidate` handling:

```python
if bool(provisional.debug.get("is_contested_residual", False)):
    obj.debug["promoted_contested_residual"] = {
        "parent_object_id": int(provisional.debug.get("contested_parent_object_id", -1)),
        "parent_label": str(provisional.debug.get("contested_parent_label", "")),
        "patch_label": str(provisional.debug.get("contested_patch_label", "")),
        "reason": str(provisional.debug.get("contested_reason", "")),
    }
```

In `_promote_stable_provisionals()`, after `_refresh_object_debug(obj, state.tsdf_volume)` and before appending to `last_created_object_ids`, record promoted contested residuals with this exact key-presence check:

```python
if "promoted_contested_residual" in obj.debug:
    self.last_contested_residual_promoted_object_ids.append(int(obj.object_id))
```

- [ ] **Step 7: Run the no-parent-update test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_provisional_pool.py::test_contested_patch_does_not_update_parent_object_memory_or_votes -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/modules/object_update.py tests/test_provisional_pool.py
git commit -m "feat: keep contested observations out of parent memory"
```

## Task 5: Promote Repeated Contested Residuals into Objects

**Files:**
- Modify: `src/modules/object_update.py`
- Test: `tests/test_provisional_pool.py`

- [ ] **Step 1: Write the failing promotion test**

Add this to `tests/test_provisional_pool.py`:

```python
def test_contested_residual_promotes_after_repeated_observations() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {
                "enabled": True,
                "promotion_hits": 2,
                "match_distance": 0.40,
                "max_idle_frames": 30,
            },
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    state = SystemState(
        objects={
            31: ObjectMap(
                object_id=31,
                local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
        },
        next_object_id=32,
    )

    for frame_id, patch_id in [(10, 101), (11, 102)]:
        patch = _make_patch(patch_id=patch_id, frame_id=frame_id)
        patch.metadata.update(
            {
                "anchor_class_name": "cushion",
                "anchor_confidence": 0.86,
                "contested_parent_object_id": 31,
                "contested_parent_label": "sofa",
                "contested_patch_label": "cushion",
                "contested_reason": "cross_label_observation_identity",
            }
        )
        state = module.process(AssociationResult(contested_object_patches=[patch_id]), [patch], state)

    assert 32 in state.objects
    promoted = state.objects[32]
    assert promoted.debug["anchor_semantics"]["canonical_label"] == "cushion"
    assert promoted.debug["promoted_contested_residual"]["parent_object_id"] == 31
    assert promoted.debug["promoted_contested_residual"]["parent_label"] == "sofa"
    assert promoted.debug["promoted_contested_residual"]["patch_label"] == "cushion"
    assert len(state.provisional_objects) == 0
    assert module.last_contested_residual_promoted_object_ids == [32]
    assert len(state.tsdf_volume.owner_support) > 0
```

- [ ] **Step 2: Run the failing promotion test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_provisional_pool.py::test_contested_residual_promotes_after_repeated_observations -q
```

Expected before Task 4: FAIL. Expected after Task 4 if promotion is already correct: PASS. If it fails, continue with Step 3.

- [ ] **Step 3: Fix provisional matching for contested residuals**

If the test fails by creating multiple provisionals for the same residual, update `_match_provisional_object()` so contested residuals match only within the same parent and same label:

```python
patch_parent_id = int(patch.metadata.get("contested_parent_object_id", -1))
patch_is_contested = patch_parent_id >= 0
patch_anchor_class = str(patch.metadata.get("anchor_class_name", ""))
for provisional in state.provisional_objects.values():
    provisional_parent_id = int(provisional.debug.get("contested_parent_object_id", -1))
    provisional_is_contested = provisional_parent_id >= 0
    if patch_is_contested != provisional_is_contested:
        continue
    if patch_is_contested and provisional_parent_id != patch_parent_id:
        continue
    if patch_anchor_class:
        if provisional.anchor_class_name and provisional.anchor_class_name != patch_anchor_class:
            continue
    elif provisional.anchor_class_name:
        continue
    distance = float(np.linalg.norm(provisional.centroid - patch.centroid))
    if distance > self.provisional_match_distance:
        continue
    if distance < best_distance:
        best_distance = distance
        best = provisional
```

- [ ] **Step 4: Run promotion and existing provisional tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_provisional_pool.py::test_contested_residual_promotes_after_repeated_observations \
  tests/test_provisional_pool.py::test_provisional_pool_records_semantic_split_candidates \
  tests/test_provisional_pool.py::test_ambiguous_patch_matches_existing_object_and_updates_local_memory \
  tests/test_provisional_pool.py::test_ambiguous_patch_only_enters_provisional_pool_when_unmatched \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/object_update.py tests/test_provisional_pool.py
git commit -m "feat: promote repeated contested residuals"
```

## Task 6: Expose Contested Outcomes in Local Memory Audit

**Files:**
- Modify: `run_room0_full_eval.py`
- Modify: `src/pipelines/main_pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Confirm audit source**

Run:

```bash
rg -n "def build_local_memory_frame_audit|new_instance_events|surface_owner_gate" run_room0_full_eval.py
```

Expected: `build_local_memory_frame_audit()` appears in `run_room0_full_eval.py`.

- [ ] **Step 2: Write the failing local audit test**

Add this near `TestPipeline.test_build_local_memory_frame_audit_includes_anchor_metadata` in `tests/test_pipeline.py`:

```python
def test_local_memory_audit_exposes_contested_residual_outcome(self):
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:5, 2:6] = True
    proposal = Proposal2D(
        proposal_id=22,
        mask=mask,
        bbox_xyxy=np.array([2, 1, 6, 5], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
    )
    refined = RefinedProposal2D(
        proposal_id=22,
        mask=mask,
        bbox_xyxy=np.array([2, 1, 6, 5], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        metadata={"source_raw_proposal_id": 22},
    )
    points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    patch = Patch3D(
        patch_id=22,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        metadata={
            "source_proposal_id": 22,
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.88,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "blanket",
            "contested_reason": "cross_label_observation_identity",
        },
    )
    association = AssociationResult(
        contested_object_patches=[22],
        contested_matches=[
            ContestedAssociation(
                patch_id=22,
                blocked_object_id=31,
                patch_label="blanket",
                object_label="sofa",
                reason="cross_label_observation_identity",
                score=AssociationScore(total_score=1.06),
            )
        ],
    )
    pipeline = SimpleNamespace(
        last_runtime_vis_output=SimpleNamespace(groups=[], proposal_profiles=[], raw_to_group={}),
        last_association=association,
        state=SystemState(),
        last_raw_proposals=[proposal],
        last_refined_proposals=[refined],
        last_patches=[patch],
        last_anchors=[],
        last_anchor_assignments=[],
        object_update=SimpleNamespace(
            last_current_frame_visibility_gate_stats={},
            last_contested_residual_patch_ids=[22],
            last_contested_residual_promoted_object_ids=[],
        ),
        last_frame_debug={"bg_patch_ids": [], "obj_patch_ids": [22], "amb_patch_ids": []},
        association=SimpleNamespace(match_threshold=0.45),
        patch_lifting=SimpleNamespace(min_points=12),
        depth_refinement=SimpleNamespace(min_area=16),
    )
    frame = SimpleNamespace(frame_id=1800, depth=np.ones((8, 8), dtype=np.float32))

    audit = build_local_memory_frame_audit(
        frame=frame,
        pipeline=pipeline,
        prev_object_ids=set(),
        prev_next_object_id=0,
    )

    assert audit["contested_residual_count"] == 1
    assert audit["contested_residual_events"] == [
        {
            "patch_id": 22,
            "blocked_object_id": 31,
            "patch_label": "blanket",
            "object_label": "sofa",
            "reason": "cross_label_observation_identity",
            "score": 1.06,
        }
    ]
    patch_record = audit["raw_proposals"][0]["patches"][0]
    assert patch_record["association_outcome"] == "contested_residual"
    assert patch_record["contested_parent_object_id"] == 31
    assert patch_record["contested_parent_label"] == "sofa"
    assert patch_record["contested_patch_label"] == "blanket"
```

Ensure `ContestedAssociation` is imported at the top of `tests/test_pipeline.py` beside `AssociationResult`.

- [ ] **Step 3: Run the failing audit test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_local_memory_audit_exposes_contested_residual_outcome -q
```

Expected: FAIL because audit does not expose contested residual fields.

- [ ] **Step 4: Add pipeline per-frame counters**

In `src/pipelines/main_pipeline.py`, where `last_frame_debug` is assembled near existing association counts, add:

```python
"contested_residual_patch_count": int(len(association.contested_object_patches)),
"contested_residual_patch_ids": [int(patch_id) for patch_id in association.contested_object_patches],
"contested_residual_promoted_object_ids": list(self.object_update.last_contested_residual_promoted_object_ids),
```

- [ ] **Step 5: Add local audit summary fields**

In `build_local_memory_frame_audit()`, derive events from `pipeline.last_association.contested_matches`:

```python
contested_matches = list(getattr(last_association, "contested_matches", []))
contested_residual_events = []
for contested in contested_matches:
    score = getattr(contested, "score", None)
    contested_residual_events.append(
        {
            "patch_id": int(contested.patch_id),
            "blocked_object_id": int(contested.blocked_object_id),
            "patch_label": str(contested.patch_label),
            "object_label": str(contested.object_label),
            "reason": str(contested.reason),
            "score": float(getattr(score, "total_score", 0.0) if score is not None else 0.0),
        }
    )
```

Add to the returned audit dict:

```python
"contested_residual_count": int(len(contested_residual_events)),
"contested_residual_events": contested_residual_events,
"contested_residual_patch_ids": [
    int(patch_id) for patch_id in getattr(last_association, "contested_object_patches", [])
],
"contested_residual_promoted_object_ids": [
    int(object_id)
    for object_id in getattr(getattr(pipeline, "object_update", None), "last_contested_residual_promoted_object_ids", [])
],
```

- [ ] **Step 6: Add per-patch audit outcome fields**

Where each patch record is constructed, add:

```python
contested_patch_ids = set(int(patch_id) for patch_id in getattr(last_association, "contested_object_patches", []))
matched_patch_ids = {int(patch_id) for patch_id, _obj_id, _score in getattr(last_association, "matched", [])}
new_patch_ids = set(int(patch_id) for patch_id in getattr(last_association, "new_object_patches", []))
```

Then for each patch record:

```python
patch_id_int = int(patch.patch_id)
if patch_id_int in contested_patch_ids:
    association_outcome = "contested_residual"
elif patch_id_int in matched_patch_ids:
    association_outcome = "matched"
elif patch_id_int in new_patch_ids:
    association_outcome = "new_object"
else:
    association_outcome = "unassociated"

patch_record["association_outcome"] = association_outcome
patch_record["contested_parent_object_id"] = int(patch.metadata.get("contested_parent_object_id", -1))
patch_record["contested_parent_label"] = str(patch.metadata.get("contested_parent_label", ""))
patch_record["contested_patch_label"] = str(patch.metadata.get("contested_patch_label", ""))
patch_record["contested_reason"] = str(patch.metadata.get("contested_reason", ""))
```

- [ ] **Step 7: Run audit tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_local_memory_audit_exposes_contested_residual_outcome \
  tests/test_pipeline.py::TestPipeline::test_build_local_memory_frame_audit_includes_anchor_metadata \
  tests/test_pipeline.py::TestRoadmapRefactor::test_local_memory_audit_exposes_frontend_stage_records \
  tests/test_pipeline.py::TestRoadmapRefactor::test_local_memory_audit_attributes_runtime_group_patches_to_all_members \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add run_room0_full_eval.py src/pipelines/main_pipeline.py tests/test_pipeline.py
git commit -m "feat: expose contested residuals in local memory audit"
```

## Task 7: Add Configuration for Observation-First Tracking

**Files:**
- Modify: `configs/default.yaml`
- Modify: `configs/room0_surface_gate_4090.yaml`
- Modify: `configs/midrecall_local_memory_boost.yaml`
- Test: existing focused tests

- [ ] **Step 1: Add association config**

In each config under the existing `association:` section, add:

```yaml
  observation_identity_gate_enabled: true
  observation_identity_min_patch_confidence: 0.35
```

Keep the old `semantic_conflict_*` keys for compatibility during this transition.

- [ ] **Step 2: Add object-update config**

In each config under the existing `object_update:` section, add:

```yaml
  contested_residual_pool:
    enabled: true
```

Do not add class-specific exceptions. Blanket/cushion/rug/cabinet/window should be handled by the same identity rule.

- [ ] **Step 3: Run config smoke tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline -q
```

If `TestPipeline` is not a valid selector in this repo version, run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add configs/default.yaml configs/room0_surface_gate_4090.yaml configs/midrecall_local_memory_boost.yaml
git commit -m "config: enable observation-first identity gate"
```

## Task 8: Integration Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py \
  tests/test_provisional_pool.py \
  tests/test_dual_map.py \
  tests/test_runtime_vis.py \
  tests/test_object_anchor.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run formatting/diff check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Run a short validation experiment**

Use the same evaluation entrypoint and config family as the previous 200f run. If the current repo still uses `run_room0_full_eval.py`, run a 200f validation with a fresh output name:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config configs/room0_surface_gate_4090.yaml \
  --room room0 \
  --max-frames 200 \
  --stride 10 \
  --output-dir outputs/tmp_validation/20260525_room0_observation_first_200f
```

Expected:

- Local audit around previously failing frames shows `blanket`/`cushion` cross-label sofa matches as `contested_residual`, not `matched`.
- Parent sofa/chair objects do not receive blanket/cushion points in `local_pcd`.
- Residual/provisional debug shows `contested_parent_object_id` and correct `contested_patch_label`.
- Final object semantic audit reduces `absorbed_by` counts for `blanket`, `cushion`, `rug`, `cabinet`, and `window`.

- [ ] **Step 4: Inspect local audit for the known failure mode**

Open the audit JSON/Markdown for the same frame pattern as the earlier frame-1800 observation. Verify the equivalent of:

```text
r23 blanket A8 -> final_outcome: contested_residual, blocked_object_id: sofa object
r14 cushion A13 -> final_outcome: contested_residual, blocked_object_id: sofa object
```

Expected: these records should no longer appear as direct matches to sofa.

- [ ] **Step 5: Commit final verification notes if needed**

If the validation command produces tracked docs updates, commit only those tracked docs. Do not commit generated validation output directories by default.

```bash
git status --short
git commit -m "docs: record observation-first validation notes"
```

Expected: create this commit only after `git add` has staged an intentional tracked docs change. If no tracked docs changed, do not create an empty commit.

## Expected Behavioral Change

Before this plan, the backend could see high TSDF/geometry overlap and commit a `blanket` or `cushion` observation into a `sofa` object. That polluted the parent pool and erased the child identity even though the frontend had observed it.

After this plan:

- Same-label observations still strengthen existing objects.
- Low-confidence or unlabeled geometry can still use conservative spatial tracking.
- Confident cross-label observations become residual identity tracks.
- Residual tracks promote after repeated hits.
- Parent object memory stays clean because cross-label points and votes never enter it.
- The local audit explains the decision path directly, so future failures can be diagnosed without guessing from final PLY colors.

## Risks and Guardrails

- This may increase object count for repeated false-positive labels. Guardrail: only confident anchored labels are blocked; low-confidence/unlabeled patches keep legacy behavior.
- This may delay TSDF ownership for true attached small objects until promotion. Guardrail: provisional promotion already requires repeated frame evidence, and after promotion the object's own TSDF support is integrated.
- This does not solve objects that are never observed by the frontend. It specifically solves "observed but absorbed".
- This does not rely on class-specific patches. Blanket, cushion, rug, cabinet, and window use the same identity invariant.

## Rollback Plan

Set these config keys to false:

```yaml
association:
  observation_identity_gate_enabled: false

object_update:
  contested_residual_pool:
    enabled: false
```

This returns association/update behavior to the previous geometry-first path while leaving dataclasses and audit fields harmlessly unused.
