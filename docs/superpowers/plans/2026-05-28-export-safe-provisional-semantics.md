# Export Safe Provisional Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover semantic coverage in PLY/eval/dense exports by allowing safe provisional anchor labels at export time while keeping identity and association committed-only.

**Architecture:** Split object semantics into two explicit surfaces: committed-only identity labels for association blocking, and export-only labels for visualization/evaluation/audit. Export fallback reads only direct strong anchor evidence from `anchor_semantics.evidence`; weak/contextual overlap evidence remains diagnostic and never becomes an exported label.

**Tech Stack:** Python, dataclasses, NumPy, pytest, existing OVIOVO modules under `src/modules`, room0 validation runner `run_room0_full_eval.py`.

---

## File Structure

- Modify: `src/modules/semantic_memory.py`
  - Add `AnchorExportPolicy`, global policy setters/getters, export-safe decision helpers, and explicit public APIs:
    - `object_identity_semantic_label(obj)`
    - `object_export_semantic_state(obj)`
    - `object_export_semantic_label(obj)`
  - Keep `preferred_object_semantic_label(obj)` as a compatibility wrapper returning committed/identity semantics, not unsafe provisional fallback.

- Modify: `src/modules/observation_identity.py`
  - Use `object_identity_semantic_label(obj)` for association identity labels.

- Modify: `src/modules/association.py`
  - Remove export-label dependency from semantic conflict helper and ensure any remaining semantic gate uses identity semantics only.

- Modify: `src/modules/dense_surface.py`
  - Use `object_export_semantic_label(obj)` for dense surface export records.

- Modify: `run_room0_full_eval.py`
  - Import export semantic APIs.
  - Make `object_semantic_label(obj)` export-facing.
  - Add `export_state`, `export_reason`, `export_source`, and `export_state_counts` to final semantic audit.

- Modify: `configs/default.yaml`
  - Add `semantic_memory.anchor_export` defaults separate from `anchor_commit`.

- Modify: `configs/room0_surface_gate_4090.yaml`
  - Add the same `semantic_memory.anchor_export` defaults for room0 experiments.

- Modify: `tests/test_pipeline.py`
  - Add unit tests for export-safe provisional semantics, contextual evidence rejection, conflict margins, and identity/export separation.

- Modify: `tests/test_dual_map.py`
  - Add tests proving pool, dense, and projected exports use export semantics.

- Modify: `tests/test_provisional_pool.py`
  - Update imports/assertions where the old preferred label name was used as an export label.

---

### Task 1: Add Failing Tests for Identity vs Export Semantics

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Update semantic-memory imports in the test file**

At the existing import block from `src.modules.semantic_memory`, change it to:

```python
from src.modules.semantic_memory import (
    SemanticMemoryModule,
    accumulate_anchor_semantic_vote,
    object_export_semantic_label,
    object_export_semantic_state,
    object_identity_semantic_label,
    object_semantic_commit_state,
    preferred_object_semantic_label,
)
```

- [ ] **Step 2: Add tests for safe provisional export and committed-only identity**

Append these tests inside `class TestRoadmapRefactor:` after `test_anchor_semantic_posterior_requires_multiframe_commit`:

```python
    def test_safe_provisional_anchor_exports_without_identity_commit(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        obj = ObjectMap(object_id=131)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            128,
            axis=0,
        )
        patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=10,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.92,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
        )

        try:
            set_anchor_commit_policy(
                {
                    "min_frame_hits": 2,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 1.0,
                    "min_score_margin": 0.15,
                    "provisional_export_fallback": False,
                }
            )
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 1,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 0.75,
                    "min_score_margin": 0.05,
                    "min_confidence": 0.75,
                    "min_view_quality": 0.25,
                    "allow_single_frame_high_confidence": True,
                    "single_frame_min_confidence": 0.90,
                    "single_frame_min_view_quality": 0.50,
                }
            )

            accumulate_anchor_semantic_vote(obj, patch)

            commit_state = object_semantic_commit_state(obj)
            export_state = object_export_semantic_state(obj)

            assert commit_state["semantic_state"] == "provisional"
            assert commit_state["committed_label"] == ""
            assert object_identity_semantic_label(obj) == ""
            assert preferred_object_semantic_label(obj) == ""
            assert object_export_semantic_label(obj) == "rug"
            assert export_state["export_label"] == "rug"
            assert export_state["export_state"] == "safe_provisional"
            assert export_state["export_source"] == "anchor_safe_provisional"
            assert export_state["semantic_state"] == "provisional"
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_contextual_overlap_evidence_never_exports_as_safe_provisional(self):
        from src.modules.semantic_memory import set_anchor_export_policy

        obj = ObjectMap(object_id=132)
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

        try:
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 1,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 0.50,
                    "min_score_margin": 0.0,
                    "min_confidence": 0.50,
                    "min_view_quality": 0.0,
                    "allow_single_frame_high_confidence": True,
                    "single_frame_min_confidence": 0.50,
                    "single_frame_min_view_quality": 0.0,
                }
            )

            accumulate_anchor_semantic_vote(obj, weak_patch)

            export_state = object_export_semantic_state(obj)
            assert obj.debug["anchor_semantics"]["contextual_label_score_sum"] == {"sofa": 0.99}
            assert object_export_semantic_label(obj) == ""
            assert export_state["export_state"] == "unlabeled"
            assert export_state["export_source"] == "none"
            assert export_state["export_reason"] == "no_direct_strong_anchor_evidence"
        finally:
            set_anchor_export_policy(None)

    def test_close_provisional_posterior_margin_is_not_exported(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        obj = ObjectMap(object_id=133)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )

        def make_patch(patch_id: int, label: str, confidence: float) -> Patch3D:
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
                    "anchor_view_quality": 0.80,
                    "anchor_label_strength": "strong",
                },
            )

        try:
            set_anchor_commit_policy(
                {
                    "min_frame_hits": 2,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 1.0,
                    "min_score_margin": 0.15,
                    "provisional_export_fallback": False,
                }
            )
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 1,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 0.50,
                    "min_score_margin": 0.20,
                    "min_confidence": 0.50,
                    "min_view_quality": 0.25,
                    "allow_single_frame_high_confidence": True,
                    "single_frame_min_confidence": 0.50,
                    "single_frame_min_view_quality": 0.25,
                }
            )

            accumulate_anchor_semantic_vote(obj, make_patch(1, "sofa", 0.90))
            accumulate_anchor_semantic_vote(obj, make_patch(2, "blanket", 0.86))

            export_state = object_export_semantic_state(obj)
            assert object_semantic_commit_state(obj)["semantic_state"] == "provisional"
            assert export_state["posterior_label"] == "sofa"
            assert object_export_semantic_label(obj) == ""
            assert export_state["export_state"] == "unlabeled"
            assert export_state["export_reason"].startswith("waiting_for_export:")
            assert "margin=" in export_state["export_reason"]
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)

    def test_committed_anchor_export_still_wins_over_safe_provisional_policy(self):
        from src.modules.semantic_memory import set_anchor_export_policy

        obj = ObjectMap(object_id=134)
        obj.debug["anchor_semantics"] = {
            "semantic_state": "committed",
            "committed_label": "chair",
            "canonical_label": "chair",
            "canonical_score": 1.25,
            "canonical_frame_hits": 2,
            "canonical_confidence": 0.88,
            "canonical_best_view_quality": 0.80,
            "commit_reason": "multiframe_strong_anchor_evidence",
            "label_weighted_score": {"chair": 1.25},
            "label_frame_hits": {"chair": 2},
            "label_high_quality_hits": {"chair": 2},
            "label_max_confidence": {"chair": 0.88},
            "label_best_view_quality": {"chair": 0.80},
            "evidence": [],
        }

        try:
            set_anchor_export_policy({"enabled": False})

            export_state = object_export_semantic_state(obj)
            assert object_identity_semantic_label(obj) == "chair"
            assert preferred_object_semantic_label(obj) == "chair"
            assert object_export_semantic_label(obj) == "chair"
            assert export_state["export_label"] == "chair"
            assert export_state["export_state"] == "committed"
            assert export_state["export_source"] == "anchor_committed"
            assert export_state["export_reason"] == "committed_anchor_semantic"
        finally:
            set_anchor_export_policy(None)
```

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestRoadmapRefactor::test_safe_provisional_anchor_exports_without_identity_commit \
  tests/test_pipeline.py::TestRoadmapRefactor::test_contextual_overlap_evidence_never_exports_as_safe_provisional \
  tests/test_pipeline.py::TestRoadmapRefactor::test_close_provisional_posterior_margin_is_not_exported \
  tests/test_pipeline.py::TestRoadmapRefactor::test_committed_anchor_export_still_wins_over_safe_provisional_policy \
  -q
```

Expected: FAIL during import with `cannot import name 'object_export_semantic_label'` or `cannot import name 'set_anchor_export_policy'`.

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/test_pipeline.py
git commit -m "test: define export-safe provisional semantic contract"
```

---

### Task 2: Implement Export-Safe Semantic Policy APIs

**Files:**
- Modify: `src/modules/semantic_memory.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add export policy constants and dataclass**

In `src/modules/semantic_memory.py`, after `ANCHOR_PROVISIONAL_EXPORT_FALLBACK = False`, add:

```python
ANCHOR_EXPORT_ENABLED = True
ANCHOR_EXPORT_MIN_FRAME_HITS = 1
ANCHOR_EXPORT_MIN_HIGH_QUALITY_HITS = 1
ANCHOR_EXPORT_MIN_WEIGHTED_SCORE = 0.75
ANCHOR_EXPORT_MIN_SCORE_MARGIN = 0.05
ANCHOR_EXPORT_MIN_CONFIDENCE = 0.75
ANCHOR_EXPORT_MIN_VIEW_QUALITY = 0.25
ANCHOR_EXPORT_ALLOW_SINGLE_FRAME_HIGH_CONFIDENCE = True
ANCHOR_EXPORT_SINGLE_FRAME_MIN_CONFIDENCE = 0.90
ANCHOR_EXPORT_SINGLE_FRAME_MIN_VIEW_QUALITY = 0.50
```

After `AnchorCommitPolicy`, add:

```python
@dataclass(frozen=True)
class AnchorExportPolicy:
    enabled: bool = ANCHOR_EXPORT_ENABLED
    min_frame_hits: int = ANCHOR_EXPORT_MIN_FRAME_HITS
    min_high_quality_hits: int = ANCHOR_EXPORT_MIN_HIGH_QUALITY_HITS
    min_weighted_score: float = ANCHOR_EXPORT_MIN_WEIGHTED_SCORE
    min_score_margin: float = ANCHOR_EXPORT_MIN_SCORE_MARGIN
    min_confidence: float = ANCHOR_EXPORT_MIN_CONFIDENCE
    min_view_quality: float = ANCHOR_EXPORT_MIN_VIEW_QUALITY
    allow_single_frame_high_confidence: bool = ANCHOR_EXPORT_ALLOW_SINGLE_FRAME_HIGH_CONFIDENCE
    single_frame_min_confidence: float = ANCHOR_EXPORT_SINGLE_FRAME_MIN_CONFIDENCE
    single_frame_min_view_quality: float = ANCHOR_EXPORT_SINGLE_FRAME_MIN_VIEW_QUALITY
```

- [ ] **Step 2: Add global export policy getter/setter**

After `_CURRENT_ANCHOR_COMMIT_POLICY = _DEFAULT_ANCHOR_COMMIT_POLICY`, add:

```python
_DEFAULT_ANCHOR_EXPORT_POLICY = AnchorExportPolicy()
_CURRENT_ANCHOR_EXPORT_POLICY = _DEFAULT_ANCHOR_EXPORT_POLICY
```

After `set_anchor_commit_policy`, add:

```python
def current_anchor_export_policy() -> AnchorExportPolicy:
    return _CURRENT_ANCHOR_EXPORT_POLICY


def set_anchor_export_policy(policy: AnchorExportPolicy | dict[str, Any] | None) -> None:
    global _CURRENT_ANCHOR_EXPORT_POLICY
    if policy is None:
        _CURRENT_ANCHOR_EXPORT_POLICY = _DEFAULT_ANCHOR_EXPORT_POLICY
        return
    if isinstance(policy, AnchorExportPolicy):
        _CURRENT_ANCHOR_EXPORT_POLICY = policy
        return
    _CURRENT_ANCHOR_EXPORT_POLICY = AnchorExportPolicy(
        enabled=bool(policy.get("enabled", ANCHOR_EXPORT_ENABLED)),
        min_frame_hits=int(policy.get("min_frame_hits", ANCHOR_EXPORT_MIN_FRAME_HITS)),
        min_high_quality_hits=int(policy.get("min_high_quality_hits", ANCHOR_EXPORT_MIN_HIGH_QUALITY_HITS)),
        min_weighted_score=float(policy.get("min_weighted_score", ANCHOR_EXPORT_MIN_WEIGHTED_SCORE)),
        min_score_margin=float(policy.get("min_score_margin", ANCHOR_EXPORT_MIN_SCORE_MARGIN)),
        min_confidence=float(policy.get("min_confidence", ANCHOR_EXPORT_MIN_CONFIDENCE)),
        min_view_quality=float(policy.get("min_view_quality", ANCHOR_EXPORT_MIN_VIEW_QUALITY)),
        allow_single_frame_high_confidence=bool(
            policy.get("allow_single_frame_high_confidence", ANCHOR_EXPORT_ALLOW_SINGLE_FRAME_HIGH_CONFIDENCE)
        ),
        single_frame_min_confidence=float(
            policy.get("single_frame_min_confidence", ANCHOR_EXPORT_SINGLE_FRAME_MIN_CONFIDENCE)
        ),
        single_frame_min_view_quality=float(
            policy.get("single_frame_min_view_quality", ANCHOR_EXPORT_SINGLE_FRAME_MIN_VIEW_QUALITY)
        ),
    )
```

- [ ] **Step 3: Replace `preferred_object_semantic_label` and add explicit identity/export APIs**

Replace the current `preferred_object_semantic_label` function with:

```python
def preferred_object_semantic_label(obj: ObjectMap) -> str:
    """Compatibility wrapper for committed object semantics.

    This function is intentionally committed-only when anchor semantic state is
    present. Use object_export_semantic_label() for PLY/eval visualization.
    """
    return object_identity_semantic_label(obj)


def object_identity_semantic_label(obj: ObjectMap) -> str:
    """Return the committed semantic label used for object identity decisions."""
    commit_state = object_semantic_commit_state(obj)
    if str(commit_state.get("semantic_state", "")) == "committed":
        return str(commit_state.get("committed_label", "")).strip()

    anchor_state = obj.debug.get("anchor_semantics")
    if isinstance(anchor_state, dict):
        return ""

    hypotheses = getattr(obj.semantic_memory, "label_hypotheses", [])
    if not hypotheses:
        return ""
    return str(hypotheses[0][0]).strip()


def object_export_semantic_label(obj: ObjectMap) -> str:
    """Return the semantic label used only for exports and evaluation."""
    return str(object_export_semantic_state(obj).get("export_label", "")).strip()


def object_export_semantic_state(obj: ObjectMap) -> dict[str, Any]:
    """Return export-facing semantic state without weakening identity semantics."""
    commit_state = object_semantic_commit_state(obj)
    anchor_state = obj.debug.get("anchor_semantics")

    if str(commit_state.get("semantic_state", "")) == "committed":
        committed_label = str(commit_state.get("committed_label", "")).strip()
        if committed_label:
            return {
                **commit_state,
                "export_label": committed_label,
                "export_state": "committed",
                "export_source": "anchor_committed",
                "export_reason": "committed_anchor_semantic",
            }

    if isinstance(anchor_state, dict):
        safe_label, reason = _safe_provisional_export_label(anchor_state)
        if safe_label:
            return {
                **commit_state,
                "export_label": safe_label,
                "export_state": "safe_provisional",
                "export_source": "anchor_safe_provisional",
                "export_reason": reason,
            }
        return {
            **commit_state,
            "export_label": "",
            "export_state": "unlabeled",
            "export_source": "none",
            "export_reason": reason,
        }

    hypotheses = getattr(obj.semantic_memory, "label_hypotheses", [])
    if hypotheses:
        label = str(hypotheses[0][0]).strip()
        if label:
            return {
                **commit_state,
                "export_label": label,
                "export_state": "semantic_memory",
                "export_source": "semantic_memory_no_anchor",
                "export_reason": "no_anchor_semantics_fallback",
            }

    return {
        **commit_state,
        "export_label": "",
        "export_state": "unlabeled",
        "export_source": "none",
        "export_reason": "no_export_semantic_evidence",
    }
```

- [ ] **Step 4: Add the safe provisional decision helper**

Add this helper before `object_semantic_commit_state`:

```python
def _safe_provisional_export_label(anchor_state: dict[str, Any]) -> tuple[str, str]:
    policy = current_anchor_export_policy()
    if not policy.enabled:
        return "", "export_provisional_disabled"

    candidate = str(anchor_state.get("canonical_label", "")).strip()
    if not candidate:
        evidence = anchor_state.get("evidence", [])
        if isinstance(evidence, list) and len(evidence) == 0:
            return "", "no_direct_strong_anchor_evidence"
        return "", "no_posterior_label"

    evidence = anchor_state.get("evidence", [])
    direct_evidence = [
        item
        for item in evidence
        if isinstance(item, dict) and str(item.get("label", "")).strip() == candidate
    ]
    if not direct_evidence:
        return "", "no_direct_strong_anchor_evidence"

    weighted_scores = anchor_state.get("label_weighted_score", {}) or {}
    frame_hits = anchor_state.get("label_frame_hits", {}) or {}
    high_quality_hits = anchor_state.get("label_high_quality_hits", {}) or {}
    max_confidence = anchor_state.get("label_max_confidence", {}) or {}
    best_view_quality = anchor_state.get("label_best_view_quality", {}) or {}

    candidate_weighted_score = float(weighted_scores.get(candidate, 0.0))
    runner_up_score = max(
        (float(score) for label, score in weighted_scores.items() if str(label) != candidate),
        default=0.0,
    )
    score_margin = float(candidate_weighted_score - runner_up_score)
    candidate_frame_hits = int(frame_hits.get(candidate, 0))
    candidate_high_quality_hits = int(high_quality_hits.get(candidate, 0))
    candidate_confidence = float(max_confidence.get(candidate, 0.0))
    candidate_view_quality = float(best_view_quality.get(candidate, 0.0))

    enough_frames = candidate_frame_hits >= policy.min_frame_hits
    enough_quality = candidate_high_quality_hits >= policy.min_high_quality_hits
    enough_score = candidate_weighted_score >= policy.min_weighted_score
    enough_margin = score_margin >= policy.min_score_margin
    enough_confidence = candidate_confidence >= policy.min_confidence
    enough_view = candidate_view_quality >= policy.min_view_quality
    single_frame_override = bool(
        policy.allow_single_frame_high_confidence
        and candidate_frame_hits == 1
        and candidate_confidence >= policy.single_frame_min_confidence
        and candidate_view_quality >= policy.single_frame_min_view_quality
        and enough_score
        and enough_margin
    )

    if (
        (enough_frames or single_frame_override)
        and enough_quality
        and enough_score
        and enough_margin
        and enough_confidence
        and enough_view
    ):
        return (
            candidate,
            "safe_provisional_anchor_evidence:"
            f"frames={candidate_frame_hits},"
            f"high_quality={candidate_high_quality_hits},"
            f"score={candidate_weighted_score:.6f},"
            f"margin={score_margin:.6f},"
            f"confidence={candidate_confidence:.6f},"
            f"view_quality={candidate_view_quality:.6f}",
        )

    return (
        "",
        "waiting_for_export:"
        f"frames={candidate_frame_hits},"
        f"high_quality={candidate_high_quality_hits},"
        f"score={candidate_weighted_score:.6f},"
        f"margin={score_margin:.6f},"
        f"confidence={candidate_confidence:.6f},"
        f"view_quality={candidate_view_quality:.6f}",
    )
```

- [ ] **Step 5: Load `anchor_export` config in `SemanticMemoryModule.__init__`**

In `SemanticMemoryModule.__init__`, after:

```python
        anchor_commit_cfg = config.get("anchor_commit", {})
        set_anchor_commit_policy(anchor_commit_cfg)
```

add:

```python
        anchor_export_cfg = config.get("anchor_export", {})
        set_anchor_export_policy(anchor_export_cfg)
```

After the existing `self.anchor_provisional_export_fallback = ...`, add:

```python
        anchor_export_policy = current_anchor_export_policy()
        self.anchor_export_enabled = anchor_export_policy.enabled
        self.anchor_export_min_frame_hits = anchor_export_policy.min_frame_hits
        self.anchor_export_min_high_quality_hits = anchor_export_policy.min_high_quality_hits
        self.anchor_export_min_weighted_score = anchor_export_policy.min_weighted_score
        self.anchor_export_min_score_margin = anchor_export_policy.min_score_margin
        self.anchor_export_min_confidence = anchor_export_policy.min_confidence
        self.anchor_export_min_view_quality = anchor_export_policy.min_view_quality
        self.anchor_export_allow_single_frame_high_confidence = (
            anchor_export_policy.allow_single_frame_high_confidence
        )
        self.anchor_export_single_frame_min_confidence = anchor_export_policy.single_frame_min_confidence
        self.anchor_export_single_frame_min_view_quality = anchor_export_policy.single_frame_min_view_quality
```

- [ ] **Step 6: Add a config-loading assertion to the existing config test**

In `test_semantic_memory_module_applies_anchor_commit_config`, add this `anchor_export` block to the module config:

```python
                    "anchor_export": {
                        "enabled": True,
                        "min_frame_hits": 1,
                        "min_high_quality_hits": 1,
                        "min_weighted_score": 0.8,
                        "min_score_margin": 0.06,
                        "min_confidence": 0.76,
                        "min_view_quality": 0.30,
                        "allow_single_frame_high_confidence": True,
                        "single_frame_min_confidence": 0.91,
                        "single_frame_min_view_quality": 0.55,
                    },
```

After `assert module.anchor_provisional_export_fallback is True`, add:

```python
            assert module.anchor_export_enabled is True
            assert module.anchor_export_min_frame_hits == 1
            assert module.anchor_export_min_high_quality_hits == 1
            assert module.anchor_export_min_weighted_score == 0.8
            assert module.anchor_export_min_score_margin == 0.06
            assert module.anchor_export_min_confidence == 0.76
            assert module.anchor_export_min_view_quality == 0.30
            assert module.anchor_export_allow_single_frame_high_confidence is True
            assert module.anchor_export_single_frame_min_confidence == 0.91
            assert module.anchor_export_single_frame_min_view_quality == 0.55
```

- [ ] **Step 7: Run the semantic-memory tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestDataStructures::test_semantic_memory_module_applies_anchor_commit_config \
  tests/test_pipeline.py::TestDataStructures::test_semantic_memory_anchor_commit_config_controls_runtime_policy \
  tests/test_pipeline.py::TestRoadmapRefactor::test_safe_provisional_anchor_exports_without_identity_commit \
  tests/test_pipeline.py::TestRoadmapRefactor::test_contextual_overlap_evidence_never_exports_as_safe_provisional \
  tests/test_pipeline.py::TestRoadmapRefactor::test_close_provisional_posterior_margin_is_not_exported \
  tests/test_pipeline.py::TestRoadmapRefactor::test_committed_anchor_export_still_wins_over_safe_provisional_policy \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit the semantic-memory implementation**

```bash
git add src/modules/semantic_memory.py tests/test_pipeline.py
git commit -m "feat: add export-safe provisional semantic labels"
```

---

### Task 3: Keep Association on Identity Semantics Only

**Files:**
- Modify: `src/modules/observation_identity.py`
- Modify: `src/modules/association.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add a regression test for association not seeing export fallback**

Append this test inside `class TestRoadmapRefactor:` in `tests/test_pipeline.py`, after `test_uncommitted_floor_like_parent_does_not_block_rug_geometry_recall`:

```python
    def test_safe_provisional_export_label_does_not_create_association_conflict(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        obj = ObjectMap(
            object_id=71,
            local_pcd=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
        )
        source_patch = Patch3D(
            patch_id=1,
            points=points,
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=1,
            metadata={
                "anchor_class_name": "rug",
                "anchor_confidence": 0.93,
                "anchor_view_quality": 0.90,
                "anchor_label_strength": "strong",
            },
        )
        incoming_patch = Patch3D(
            patch_id=2,
            points=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=2,
            metadata={
                "anchor_class_name": "sofa",
                "anchor_confidence": 0.91,
                "anchor_view_quality": 0.85,
                "anchor_label_strength": "strong",
            },
        )

        try:
            set_anchor_commit_policy(
                {
                    "min_frame_hits": 2,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 1.0,
                    "min_score_margin": 0.15,
                    "provisional_export_fallback": False,
                }
            )
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 1,
                    "min_high_quality_hits": 1,
                    "min_weighted_score": 0.75,
                    "min_score_margin": 0.05,
                    "min_confidence": 0.75,
                    "min_view_quality": 0.25,
                    "allow_single_frame_high_confidence": True,
                    "single_frame_min_confidence": 0.90,
                    "single_frame_min_view_quality": 0.50,
                }
            )
            accumulate_anchor_semantic_vote(obj, source_patch)

            module = AssociationModule(
                {
                    "min_score": 0.1,
                    "semantic_conflict_gate": {
                        "enabled": True,
                        "min_patch_confidence": 0.5,
                        "min_object_confidence": 0.5,
                    },
                }
            )
            state = SystemState(objects={71: obj})
            result = module.process([incoming_patch], state)

            assert object_export_semantic_label(obj) == "rug"
            assert object_identity_semantic_label(obj) == ""
            assert result.contested_matches == []
            assert result.matched[0][0] == 2
            assert result.matched[0][1] == 71
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)
```

- [ ] **Step 2: Run the regression and verify the current behavior**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestRoadmapRefactor::test_safe_provisional_export_label_does_not_create_association_conflict \
  -q
```

Expected before implementation: FAIL if `AssociationModule` or its legacy helper sees export/provisional labels, or PASS if the main path is already identity-only. Keep the test either way.

- [ ] **Step 3: Update `observation_identity.py` to call the new identity API**

Change the import at the top of `src/modules/observation_identity.py` to:

```python
from src.modules.semantic_memory import object_identity_semantic_label, object_semantic_commit_state
```

Replace `object_identity_label` with:

```python
def object_identity_label(obj: ObjectMap) -> str:
    """Return committed semantic identity used for association blocking."""
    return normalize_label(object_identity_semantic_label(obj))
```

- [ ] **Step 4: Update `association.py` to remove export-label semantics from conflict checking**

Change the import near the top of `src/modules/association.py` from:

```python
from src.modules.semantic_memory import preferred_object_semantic_label
```

to:

```python
from src.modules.semantic_memory import object_identity_semantic_label
```

In `_semantic_conflict_reason`, replace:

```python
        object_label = preferred_object_semantic_label(obj).strip()
```

with:

```python
        object_label = object_identity_semantic_label(obj).strip()
```

- [ ] **Step 5: Run association identity tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestDataStructures::test_association_result_records_contested_matches \
  tests/test_pipeline.py::TestRoadmapRefactor::test_safe_provisional_export_label_does_not_create_association_conflict \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_routes_confident_cross_label_overlap_to_contested_not_matched \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_blocks_semantic_conflict_subregion_matches \
  tests/test_pipeline.py::TestRoadmapRefactor::test_uncommitted_floor_like_parent_does_not_block_rug_geometry_recall \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit the association separation**

```bash
git add src/modules/observation_identity.py src/modules/association.py tests/test_pipeline.py
git commit -m "fix: keep association on committed identity semantics"
```

---

### Task 4: Route PLY, Dense Surface, and Audit Through Export Semantics

**Files:**
- Modify: `src/modules/dense_surface.py`
- Modify: `run_room0_full_eval.py`
- Modify: `tests/test_dual_map.py`
- Modify: `tests/test_provisional_pool.py`

- [ ] **Step 1: Update dual-map test imports**

In `tests/test_dual_map.py`, change:

```python
from src.modules.semantic_memory import preferred_object_semantic_label
```

to:

```python
from src.modules.semantic_memory import object_export_semantic_label, preferred_object_semantic_label
```

- [ ] **Step 2: Add an export-label regression for pool and dense records**

Append this test after `test_anchor_voted_labels_drive_dense_surface_and_pool_exports` in `tests/test_dual_map.py`:

```python
def test_safe_provisional_labels_drive_dense_surface_and_pool_exports() -> None:
    from src.modules.semantic_memory import accumulate_anchor_semantic_vote, set_anchor_commit_policy, set_anchor_export_policy

    patch_points = np.repeat(
        np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0], [0.05, 0.05, 1.0]], dtype=np.float32),
        64,
        axis=0,
    )
    patch = Patch3D(
        patch_id=77,
        points=patch_points,
        centroid=patch_points.mean(axis=0),
        bbox_min=patch_points.min(axis=0),
        bbox_max=patch_points.max(axis=0),
        source_frame_id=11,
        metadata={
            "anchor_class_name": "rug",
            "anchor_confidence": 0.94,
            "anchor_view_quality": 0.90,
            "anchor_label_strength": "strong",
        },
    )
    obj = ObjectMap(
        object_id=77,
        state=ObjectState.ACTIVE,
        local_pcd=patch_points.copy(),
        centroid=patch_points.mean(axis=0),
        bbox_min=patch_points.min(axis=0),
        bbox_max=patch_points.max(axis=0),
        observations=[ObservationRecord(frame_id=11, patch=patch)],
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.99)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}

    try:
        set_anchor_commit_policy(
            {
                "min_frame_hits": 2,
                "min_high_quality_hits": 1,
                "min_weighted_score": 1.0,
                "min_score_margin": 0.15,
                "provisional_export_fallback": False,
            }
        )
        set_anchor_export_policy(
            {
                "enabled": True,
                "min_frame_hits": 1,
                "min_high_quality_hits": 1,
                "min_weighted_score": 0.75,
                "min_score_margin": 0.05,
                "min_confidence": 0.75,
                "min_view_quality": 0.25,
                "allow_single_frame_high_confidence": True,
                "single_frame_min_confidence": 0.90,
                "single_frame_min_view_quality": 0.50,
            }
        )
        accumulate_anchor_semantic_vote(obj, patch)
        state = SystemState(objects={77: obj})
        state = DenseSurfaceModule({"dense_surface_voxel": 0.02, "dense_surface_cap_per_object": 8}).process(
            state,
            frame_id=11,
        )

        assert preferred_object_semantic_label(obj) == ""
        assert object_export_semantic_label(obj) == "rug"
        assert state.dense_surface_map.entries[77].semantic_label == "rug"

        pool_records = build_pool_semantic_records(state)
        debug_records = build_pool_debug_records(state)
        dense_records = build_dense_surface_records(state)
        expected_color = semantic_surface_color("rug", 77)
        assert tuple(int(value) for value in pool_records[0][["red", "green", "blue"]]) == expected_color
        assert tuple(int(value) for value in debug_records[0][["red", "green", "blue"]]) == expected_color
        assert tuple(int(value) for value in dense_records[0][["red", "green", "blue"]]) == expected_color
    finally:
        set_anchor_commit_policy(None)
        set_anchor_export_policy(None)
```

- [ ] **Step 3: Add an audit regression for export state counts**

Append this test near `test_write_run_report_describes_pool_and_tsdf_exports` in `tests/test_dual_map.py`:

```python
def test_final_semantic_audit_reports_export_state_counts() -> None:
    from run_room0_full_eval import build_final_object_semantic_audit
    from src.modules.semantic_memory import accumulate_anchor_semantic_vote, set_anchor_commit_policy, set_anchor_export_policy

    points = np.repeat(
        np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32),
        64,
        axis=0,
    )
    patch = Patch3D(
        patch_id=1,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=1,
        metadata={
            "anchor_class_name": "rug",
            "anchor_confidence": 0.93,
            "anchor_view_quality": 0.90,
            "anchor_label_strength": "strong",
        },
    )
    obj = ObjectMap(
        object_id=5,
        state=ObjectState.ACTIVE,
        local_pcd=points.copy(),
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        observations=[ObservationRecord(frame_id=1, patch=patch)],
    )

    try:
        set_anchor_commit_policy(
            {
                "min_frame_hits": 2,
                "min_high_quality_hits": 1,
                "min_weighted_score": 1.0,
                "min_score_margin": 0.15,
                "provisional_export_fallback": False,
            }
        )
        set_anchor_export_policy(
            {
                "enabled": True,
                "min_frame_hits": 1,
                "min_high_quality_hits": 1,
                "min_weighted_score": 0.75,
                "min_score_margin": 0.05,
                "min_confidence": 0.75,
                "min_view_quality": 0.25,
                "allow_single_frame_high_confidence": True,
                "single_frame_min_confidence": 0.90,
                "single_frame_min_view_quality": 0.50,
            }
        )
        accumulate_anchor_semantic_vote(obj, patch)
        audit = build_final_object_semantic_audit(
            SystemState(objects={5: obj}),
            {
                "gt_labels": np.array([], dtype=np.int32),
                "per_class": [],
                "object_to_class": {},
            },
            {},
        )

        assert audit["semantic_state_counts"] == {"provisional": 1}
        assert audit["export_state_counts"] == {"safe_provisional": 1}
        assert audit["objects"][0]["exported_label"] == "rug"
        assert audit["objects"][0]["export_state"] == "safe_provisional"
        assert audit["objects"][0]["export_source"] == "anchor_safe_provisional"
        assert audit["objects"][0]["export_reason"].startswith("safe_provisional_anchor_evidence:")
    finally:
        set_anchor_commit_policy(None)
        set_anchor_export_policy(None)
```

- [ ] **Step 4: Run export tests and verify failure before routing**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_dual_map.py::test_safe_provisional_labels_drive_dense_surface_and_pool_exports \
  tests/test_dual_map.py::test_final_semantic_audit_reports_export_state_counts \
  -q
```

Expected before routing: FAIL because dense/pool/audit still use committed-only labels or do not emit `export_state_counts`.

- [ ] **Step 5: Route `dense_surface.py` through export labels**

Change the import in `src/modules/dense_surface.py` from:

```python
from src.modules.semantic_memory import preferred_object_semantic_label
```

to:

```python
from src.modules.semantic_memory import object_export_semantic_label
```

Replace `_semantic_label` with:

```python
    def _semantic_label(self, obj: ObjectMap) -> str:
        return object_export_semantic_label(obj)
```

- [ ] **Step 6: Route `run_room0_full_eval.py` through export labels**

Change the import near the top of `run_room0_full_eval.py` from:

```python
from src.modules.semantic_memory import object_semantic_commit_state, preferred_object_semantic_label
```

to:

```python
from src.modules.semantic_memory import object_export_semantic_label, object_export_semantic_state, object_semantic_commit_state
```

Replace:

```python
def object_semantic_label(obj) -> str:
    return preferred_object_semantic_label(obj)
```

with:

```python
def object_semantic_label(obj) -> str:
    return object_export_semantic_label(obj)
```

- [ ] **Step 7: Add export state fields to final audit records**

In `build_final_object_semantic_audit`, after:

```python
        semantic_commit = object_semantic_commit_state(obj)
```

add:

```python
        semantic_export = object_export_semantic_state(obj)
```

In the `record = { ... }` dictionary, replace:

```python
            "exported_label": object_semantic_label(obj),
```

with:

```python
            "exported_label": str(semantic_export["export_label"]),
            "export_state": str(semantic_export["export_state"]),
            "export_source": str(semantic_export["export_source"]),
            "export_reason": str(semantic_export["export_reason"]),
```

After the existing `semantic_state_counts` loop, add:

```python
    export_state_counts = {}
    for record in object_records:
        export_state = str(record.get("export_state", "unlabeled"))
        export_state_counts[export_state] = int(export_state_counts.get(export_state, 0) + 1)
```

In the returned dictionary, add:

```python
        "export_state_counts": export_state_counts,
```

immediately after `"semantic_state_counts": semantic_state_counts,`.

- [ ] **Step 8: Show export state counts in the Markdown audit**

In `render_final_object_semantic_audit`, after:

```python
        f"- provisional_object_count: `{int(audit.get('provisional_object_count', 0))}`",
```

add:

```python
        f"- semantic_state_counts: `{json.dumps(audit.get('semantic_state_counts', {}), sort_keys=True)}`",
        f"- export_state_counts: `{json.dumps(audit.get('export_state_counts', {}), sort_keys=True)}`",
```

The file already imports `json`; do not add a duplicate import.

- [ ] **Step 9: Update provisional pool tests that use preferred label as export label**

In `tests/test_provisional_pool.py`, change the import:

```python
from src.modules.semantic_memory import preferred_object_semantic_label
```

to:

```python
from src.modules.semantic_memory import object_export_semantic_label
```

Replace the existing export assertion:

```python
assert preferred_object_semantic_label(obj) == "rug"
```

with:

```python
assert object_export_semantic_label(obj) == "rug"
```

- [ ] **Step 10: Run export routing tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_dual_map.py::test_anchor_voted_labels_drive_dense_surface_and_pool_exports \
  tests/test_dual_map.py::test_safe_provisional_labels_drive_dense_surface_and_pool_exports \
  tests/test_dual_map.py::test_final_semantic_audit_reports_export_state_counts \
  tests/test_dual_map.py::test_write_run_report_describes_pool_and_tsdf_exports \
  tests/test_provisional_pool.py::test_provisional_pool_buffers_then_promotes_new_objects \
  -q
```

Expected: PASS.

- [ ] **Step 11: Commit export routing**

```bash
git add src/modules/dense_surface.py run_room0_full_eval.py tests/test_dual_map.py tests/test_provisional_pool.py
git commit -m "feat: route exports through safe semantic labels"
```

---

### Task 5: Add Config Defaults and Run Full Verification

**Files:**
- Modify: `configs/default.yaml`
- Modify: `configs/room0_surface_gate_4090.yaml`

- [ ] **Step 1: Add `anchor_export` defaults to `configs/default.yaml`**

Under `semantic_memory.anchor_commit`, keep `provisional_export_fallback: false`, then add:

```yaml
  anchor_export:
    enabled: true
    min_frame_hits: 1
    min_high_quality_hits: 1
    min_weighted_score: 0.75
    min_score_margin: 0.05
    min_confidence: 0.75
    min_view_quality: 0.25
    allow_single_frame_high_confidence: true
    single_frame_min_confidence: 0.90
    single_frame_min_view_quality: 0.50
```

- [ ] **Step 2: Add `anchor_export` defaults to `configs/room0_surface_gate_4090.yaml`**

Under `semantic_memory.anchor_commit`, keep `provisional_export_fallback: false`, then add:

```yaml
  anchor_export:
    enabled: true
    min_frame_hits: 1
    min_high_quality_hits: 1
    min_weighted_score: 0.75
    min_score_margin: 0.05
    min_confidence: 0.75
    min_view_quality: 0.25
    allow_single_frame_high_confidence: true
    single_frame_min_confidence: 0.90
    single_frame_min_view_quality: 0.50
```

- [ ] **Step 3: Run the targeted regression suite**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestDataStructures::test_semantic_memory_module_applies_anchor_commit_config \
  tests/test_pipeline.py::TestDataStructures::test_semantic_memory_anchor_commit_config_controls_runtime_policy \
  tests/test_pipeline.py::TestRoadmapRefactor \
  tests/test_dual_map.py \
  tests/test_provisional_pool.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Run a broader non-GPU verification set**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py \
  tests/test_dual_map.py \
  tests/test_provisional_pool.py \
  tests/test_object_anchor.py \
  tests/test_dualmap_baseline_export.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit config and verification-ready changes**

```bash
git add configs/default.yaml configs/room0_surface_gate_4090.yaml
git commit -m "config: enable export-safe provisional semantics"
```

---

### Task 6: Run room0 Stride-10 Smoke Experiment and Compare

**Files:**
- No code changes expected.
- Runtime output: `outputs/tmp_validation/20260528_room0_export_safe_provisional_stride10_200f`

- [ ] **Step 1: Start the room0 200f stride-10 experiment in tmux**

Run this from the repository root:

```bash
tmux new-session -d -s room0_export_safe_provisional_stride10_200f "/bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && experiment=20260528_room0_export_safe_provisional_stride10_200f && log=outputs/tmp_validation/${experiment}.log && status=outputs/tmp_validation/${experiment}.status && { echo experiment=${experiment}; echo started_at=$(date --iso-8601=seconds); echo running > ${status}; /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_4090.yaml --experiment-name ${experiment} --num-frames 200 --frame-stride 10 --dataset-root /home/ww/vv/dataset/Replica/room0 --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json --output-root /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation --proposal-backend precomputed --proposal-cache-dir /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2 --proposal-cache-manifest /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json; rc=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=${rc}; echo exit_status=${rc} > ${status}; exit ${rc}; } > ${log} 2>&1'"
```

Expected immediately:

```bash
tmux list-sessions
```

shows `room0_export_safe_provisional_stride10_200f`.

- [ ] **Step 2: Monitor completion**

Run:

```bash
tail -n 80 outputs/tmp_validation/20260528_room0_export_safe_provisional_stride10_200f.log
cat outputs/tmp_validation/20260528_room0_export_safe_provisional_stride10_200f.status
```

Expected at completion:

```text
exit_status=0
```

- [ ] **Step 3: Inspect export coverage and metrics**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/tmp_validation/20260528_room0_export_safe_provisional_stride10_200f/room0")
audit = json.loads((root / "final_object_semantic_audit.json").read_text())
results = json.loads((root / "eval" / "results.json").read_text())
empty_labels = sum(1 for item in audit["objects"] if not item.get("exported_label"))
top_labels = {}
for item in audit["objects"]:
    label = item.get("exported_label", "")
    top_labels[label] = top_labels.get(label, 0) + 1
print("results", results)
print("object_count", audit["object_count"])
print("semantic_state_counts", audit["semantic_state_counts"])
print("export_state_counts", audit["export_state_counts"])
print("empty_exported_labels", empty_labels)
print("top_exported_labels", sorted(top_labels.items(), key=lambda kv: kv[1], reverse=True)[:12])
PY
```

Expected:

```text
exit_status=0
semantic_state_counts still includes committed/provisional/unlabeled
export_state_counts includes safe_provisional
empty_exported_labels is substantially lower than 49 out of 76 from 20260528_room0_geometry_first_semantic_late_d39853c_stride10_200f_gpu
miou is higher than 0.4831 or fmiou is higher than 0.4886
```

- [ ] **Step 4: Compare against the previous committed-only experiment**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

experiments = [
    "20260528_room0_geometry_first_semantic_late_d39853c_stride10_200f_gpu",
    "20260528_room0_export_safe_provisional_stride10_200f",
]
for exp in experiments:
    root = Path("outputs/tmp_validation") / exp / "room0"
    results = json.loads((root / "eval" / "results.json").read_text())
    audit = json.loads((root / "final_object_semantic_audit.json").read_text())
    empty = sum(1 for item in audit["objects"] if not item.get("exported_label"))
    print(exp)
    print("  miou", results["miou"])
    print("  fmiou", results["fmiou"])
    print("  object_count", audit["object_count"])
    print("  semantic_state_counts", audit.get("semantic_state_counts", {}))
    print("  export_state_counts", audit.get("export_state_counts", {}))
    print("  empty_exported_labels", empty)
PY
```

Expected: The new run has fewer empty exported labels and better or equal semantic metrics while retaining committed-only identity behavior from unit tests.

- [ ] **Step 5: Capture experiment comparison for the final handoff**

Do not edit repository files in this step. Copy these exact fields from Step 4 into the implementation handoff:

```text
old_miou:
old_fmiou:
old_object_count:
old_semantic_state_counts:
old_export_state_counts:
old_empty_exported_labels:
new_miou:
new_fmiou:
new_object_count:
new_semantic_state_counts:
new_export_state_counts:
new_empty_exported_labels:
```

---

## Expected Behavior After Implementation

- Association and observation identity remain committed-only:
  - provisional posterior labels do not block cross-label updates;
  - export fallback cannot create contested residuals;
  - weak/contextual overlap labels never become object identity.

- PLY/eval/dense exports can use safe provisional labels:
  - committed labels still win;
  - direct strong single-frame high-confidence labels can be exported;
  - contextual labels such as blocked sofa-overlap labels remain unexported;
  - close posterior conflicts remain unexported until additional evidence improves margin.

- Final audit becomes diagnosable:
  - `semantic_state_counts` continues to describe committed/provisional/unlabeled identity state;
  - `export_state_counts` describes committed/safe_provisional/semantic_memory/unlabeled export state;
  - each object records `export_state`, `export_source`, and `export_reason`.

## Self-Review Checklist

- Spec coverage:
  - The plan addresses the failure mode where committed-only export leaves many objects without labels.
  - The plan does not weaken association identity; Task 3 adds an explicit regression.
  - Weak overlap/contextual labels are rejected by tests and by implementation.
  - Dense surface, pool records, TSDF colors, and final audit route through export semantics.

- Placeholder scan:
  - The plan contains no `TBD`, no unresolved function names, and no open-ended implementation steps.
  - The only conditional instruction is the validation-note commit, which is intentionally optional because the repository may not track experiment notes.

- Type consistency:
  - `object_export_semantic_state(obj)` returns keys used later by `run_room0_full_eval.py`.
  - `object_identity_semantic_label(obj)` is the only label API used by association.
  - `preferred_object_semantic_label(obj)` remains available for existing imports and is committed-only.
