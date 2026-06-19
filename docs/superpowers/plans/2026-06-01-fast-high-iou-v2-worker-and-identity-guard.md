# Fast High-IoU V2 Worker And Identity Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the 0526 high-recall semantic quality while removing the avoidable YOLOE per-frame process overhead and preventing provisional objects from absorbing cross-class observations.

**Architecture:** Keep the 0526-style SAM high-recall frontend path in `configs/room0_surface_gate_fast_high_iou_4090.yaml`; do not switch to anchor-primary fast hybrid because prior smoke runs were fast but low-IoU. Add a separate association-only identity guard label so repeated provisional anchor evidence can block cross-class merges without becoming an export/commit label. Relax safe provisional export only for repeated, stable, direct anchor evidence so large structures and repeated objects do not remain unlabeled.

**Tech Stack:** Python, pytest, NumPy, YAML config, existing OVIOVO modules under `src/modules`, existing room0 evaluator `run_room0_full_eval.py`, tmux for long experiments.

---

## Evidence And Constraints

- Latest full run: `outputs/tmp_validation/20260531_room0_surface_gate_fast_high_iou_stride1_2000f_fast`
  - `exit_status=0`
  - mIoU `0.4945`, f-mIoU `0.5052`
  - wall time `30624s`, FPS `0.0653`
  - `yoloe_supplemental` mean `4.7618s/frame`, total `9523.5334s`
  - `final_object_count=126`
  - `anchor_voted_unanchored_count_total=63195`
  - final audit: `42 committed`, `83 provisional`, `84 export unlabeled`
- Best baseline: `outputs/tmp_validation/20260526_room0_observation_first_2b830e1_stride1_2000f`
  - mIoU `0.6050`, f-mIoU `0.7531`
  - `final_object_count=475`
  - `anchor_voted_unanchored_count_total=6222`
- The fix must preserve the user's semantic safety rule:
  - A SAM proposal that is not sufficiently covered by its YOLO anchor must not inherit the overlapping parent anchor label just because it overlaps a sofa/chair/wall.
  - This plan only uses provisional labels to block unsafe cross-label association, not to assign labels to unanchored proposals.

## File Structure

- Modify `configs/room0_surface_gate_fast_high_iou_4090.yaml`
  - Enable the existing YOLOE persistent worker in the fast high-IoU config.
  - Add repeated-evidence safe export thresholds under `semantic_memory.anchor_export`.
- Modify `src/modules/semantic_memory.py`
  - Extend `AnchorExportPolicy` with repeated-evidence fallback thresholds.
  - Add `object_association_identity_semantic_label(obj)` for association-only identity guarding.
  - Keep `object_identity_semantic_label(obj)` and `preferred_object_semantic_label(obj)` behavior unchanged.
- Modify `src/modules/observation_identity.py`
  - Use `object_association_identity_semantic_label(obj)` in `classify_observation_identity`.
  - Keep unlabeled patches allowed to update; the guard only acts when the incoming patch has a confident anchor label.
- Modify `src/modules/association.py`
  - Replace the legacy semantic-conflict helper's direct committed-label lookup with the association identity helper for consistency.
- Modify `tests/test_pipeline.py`
  - Add tests for config worker enablement, association identity guard behavior, repeated-evidence export, and legacy identity/export separation.
- Optional modify `tests/test_dual_map.py`
  - Only if a focused config test belongs better beside existing fast-hybrid config tests; prefer keeping this plan in `tests/test_pipeline.py` to minimize touched files.

## Task 1: Enable YOLOE Worker In Fast High-IoU Config

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `configs/room0_surface_gate_fast_high_iou_4090.yaml`

- [ ] **Step 1: Write the failing config test**

In `tests/test_pipeline.py`, update `TestPipeline.test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend` by adding these assertions after the existing `assert config["semantic_memory"]["anchor_export"]["enabled"] is True` line:

```python
        supplemental = config["anchor_frontend"]["supplemental"]
        assert supplemental["worker_enabled"] is True
        assert supplemental["worker_fallback_on_error"] is True
        assert supplemental["worker_request_timeout_sec"] == 120.0
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend -q
```

Expected: FAIL with `KeyError: 'worker_enabled'` or an assertion showing the config does not enable the worker.

- [ ] **Step 3: Add worker keys to the config**

In `configs/room0_surface_gate_fast_high_iou_4090.yaml`, under `anchor_frontend.supplemental`, immediately after `python_executable`, add:

```yaml
    worker_enabled: true
    worker_fallback_on_error: true
    worker_request_timeout_sec: 120.0
```

The resulting block should look like:

```yaml
  supplemental:
    backend: yoloe_seg_pf
    device: cuda:0
    python_executable: /home/ww/miniconda3/envs/oviovo/bin/python
    worker_enabled: true
    worker_fallback_on_error: true
    worker_request_timeout_sec: 120.0
    repo_root: /home/ww/vv/yoloe_repo_probe
    checkpoint_path: /home/ww/vv/yoloe_repo_probe/pretrain/yoloe-v8s-seg-pf.pt
    vocab_source_checkpoint_path: /home/ww/vv/yoloe_repo_probe/pretrain/yoloe-v8s-seg.pt
    confidence_threshold: 0.01
    max_detections: 128
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add configs/room0_surface_gate_fast_high_iou_4090.yaml tests/test_pipeline.py
git commit -m "config: enable yoloe worker for fast high-iou mode"
```

## Task 2: Add Association-Only Provisional Identity Guard

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/modules/semantic_memory.py`
- Modify: `src/modules/observation_identity.py`
- Modify: `src/modules/association.py`

- [ ] **Step 1: Write a failing unit test for association identity labels**

In `tests/test_pipeline.py`, add `object_association_identity_semantic_label` to the semantic memory import block:

```python
    object_association_identity_semantic_label,
```

Add this test near the existing safe provisional export tests in `TestPipeline`, before `test_safe_provisional_anchor_exports_without_identity_commit`:

```python
    def test_repeated_provisional_anchor_label_becomes_association_identity_guard_only(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        obj = ObjectMap(object_id=230)
        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )

        def make_patch(frame_id: int) -> Patch3D:
            return Patch3D(
                patch_id=frame_id,
                points=points.copy(),
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "rug",
                    "anchor_confidence": 0.55,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                },
            )

        try:
            set_anchor_commit_policy(
                {
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "provisional_export_fallback": False,
                }
            )
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "min_confidence": 0.95,
                    "min_view_quality": 0.95,
                    "allow_single_frame_high_confidence": False,
                    "repeated_evidence_enabled": True,
                    "repeated_min_frame_hits": 3,
                    "repeated_min_weighted_score": 1.0,
                    "repeated_min_score_margin": 0.25,
                    "repeated_min_view_quality": 0.50,
                    "repeated_min_confidence": 0.25,
                }
            )

            accumulate_anchor_semantic_vote(obj, make_patch(1))
            accumulate_anchor_semantic_vote(obj, make_patch(2))
            accumulate_anchor_semantic_vote(obj, make_patch(3))

            assert object_semantic_commit_state(obj)["semantic_state"] == "provisional"
            assert object_identity_semantic_label(obj) == ""
            assert preferred_object_semantic_label(obj) == ""
            assert object_association_identity_semantic_label(obj) == "rug"
            assert object_export_semantic_label(obj) == "rug"
            assert object_export_semantic_state(obj)["export_state"] == "safe_provisional"
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_repeated_provisional_anchor_label_becomes_association_identity_guard_only -q
```

Expected: FAIL with import error for `object_association_identity_semantic_label` or assertion failure because the new behavior does not exist.

- [ ] **Step 3: Add policy fields to `AnchorExportPolicy`**

In `src/modules/semantic_memory.py`, add these constants after `ANCHOR_EXPORT_SINGLE_FRAME_MIN_VIEW_QUALITY`:

```python
ANCHOR_EXPORT_REPEATED_EVIDENCE_ENABLED = False
ANCHOR_EXPORT_REPEATED_MIN_FRAME_HITS = 3
ANCHOR_EXPORT_REPEATED_MIN_WEIGHTED_SCORE = 1.0
ANCHOR_EXPORT_REPEATED_MIN_SCORE_MARGIN = 0.25
ANCHOR_EXPORT_REPEATED_MIN_VIEW_QUALITY = 0.50
ANCHOR_EXPORT_REPEATED_MIN_CONFIDENCE = 0.25
```

Extend `AnchorExportPolicy`:

```python
    repeated_evidence_enabled: bool = ANCHOR_EXPORT_REPEATED_EVIDENCE_ENABLED
    repeated_min_frame_hits: int = ANCHOR_EXPORT_REPEATED_MIN_FRAME_HITS
    repeated_min_weighted_score: float = ANCHOR_EXPORT_REPEATED_MIN_WEIGHTED_SCORE
    repeated_min_score_margin: float = ANCHOR_EXPORT_REPEATED_MIN_SCORE_MARGIN
    repeated_min_view_quality: float = ANCHOR_EXPORT_REPEATED_MIN_VIEW_QUALITY
    repeated_min_confidence: float = ANCHOR_EXPORT_REPEATED_MIN_CONFIDENCE
```

Extend `set_anchor_export_policy()` when building `AnchorExportPolicy`:

```python
        repeated_evidence_enabled=bool(
            policy.get("repeated_evidence_enabled", ANCHOR_EXPORT_REPEATED_EVIDENCE_ENABLED)
        ),
        repeated_min_frame_hits=int(
            policy.get("repeated_min_frame_hits", ANCHOR_EXPORT_REPEATED_MIN_FRAME_HITS)
        ),
        repeated_min_weighted_score=float(
            policy.get("repeated_min_weighted_score", ANCHOR_EXPORT_REPEATED_MIN_WEIGHTED_SCORE)
        ),
        repeated_min_score_margin=float(
            policy.get("repeated_min_score_margin", ANCHOR_EXPORT_REPEATED_MIN_SCORE_MARGIN)
        ),
        repeated_min_view_quality=float(
            policy.get("repeated_min_view_quality", ANCHOR_EXPORT_REPEATED_MIN_VIEW_QUALITY)
        ),
        repeated_min_confidence=float(
            policy.get("repeated_min_confidence", ANCHOR_EXPORT_REPEATED_MIN_CONFIDENCE)
        ),
```

- [ ] **Step 4: Add helper functions for repeated direct anchor evidence**

In `src/modules/semantic_memory.py`, add this helper above `_safe_provisional_export_label()`:

```python
def _anchor_label_stats(anchor_state: dict[str, Any], candidate: str) -> dict[str, float | int]:
    weighted_scores = anchor_state.get("label_weighted_score", {})
    frame_hits_by_label = anchor_state.get("label_frame_hits", {})
    high_quality_hits_by_label = anchor_state.get("label_high_quality_hits", {})
    max_confidence_by_label = anchor_state.get("label_max_confidence", {})
    best_view_quality_by_label = anchor_state.get("label_best_view_quality", {})

    weighted_score = _safe_float(weighted_scores.get(candidate, 0.0)) if isinstance(weighted_scores, dict) else 0.0
    frame_hits = _safe_int(frame_hits_by_label.get(candidate, 0)) if isinstance(frame_hits_by_label, dict) else 0
    high_quality_hits = (
        _safe_int(high_quality_hits_by_label.get(candidate, 0)) if isinstance(high_quality_hits_by_label, dict) else 0
    )
    max_confidence = (
        _safe_float(max_confidence_by_label.get(candidate, 0.0)) if isinstance(max_confidence_by_label, dict) else 0.0
    )
    best_view_quality = (
        _safe_float(best_view_quality_by_label.get(candidate, 0.0))
        if isinstance(best_view_quality_by_label, dict)
        else 0.0
    )
    runner_up_score = (
        max((_safe_float(score) for label, score in weighted_scores.items() if str(label) != candidate), default=0.0)
        if isinstance(weighted_scores, dict)
        else 0.0
    )
    margin = _safe_float(weighted_score - runner_up_score)
    return {
        "weighted_score": weighted_score,
        "frame_hits": frame_hits,
        "high_quality_hits": high_quality_hits,
        "max_confidence": max_confidence,
        "best_view_quality": best_view_quality,
        "margin": margin,
    }
```

Then add this helper below `_anchor_label_stats()`:

```python
def _has_direct_anchor_evidence(anchor_state: dict[str, Any], candidate: str) -> bool:
    evidence = anchor_state.get("evidence", [])
    evidence_items = evidence if isinstance(evidence, list) else []
    return any(
        isinstance(item, dict) and str(item.get("label", "")).strip() == candidate
        for item in evidence_items
    )
```

- [ ] **Step 5: Implement `object_association_identity_semantic_label()`**

In `src/modules/semantic_memory.py`, add this function below `object_identity_semantic_label()`:

```python
def object_association_identity_semantic_label(obj: ObjectMap) -> str:
    """Return the semantic identity used only to prevent cross-class association absorption."""
    committed = object_identity_semantic_label(obj).strip()
    if committed:
        return committed

    anchor_state = obj.debug.get("anchor_semantics")
    if not isinstance(anchor_state, dict):
        return ""

    candidate = str(anchor_state.get("canonical_label", "")).strip()
    if not candidate or not _has_direct_anchor_evidence(anchor_state, candidate):
        return ""

    stats = _anchor_label_stats(anchor_state, candidate)
    policy = current_anchor_export_policy()
    if not policy.repeated_evidence_enabled:
        return ""

    if (
        int(stats["frame_hits"]) >= policy.repeated_min_frame_hits
        and float(stats["weighted_score"]) >= policy.repeated_min_weighted_score
        and float(stats["margin"]) >= policy.repeated_min_score_margin
        and float(stats["best_view_quality"]) >= policy.repeated_min_view_quality
        and float(stats["max_confidence"]) >= policy.repeated_min_confidence
    ):
        return candidate
    return ""
```

- [ ] **Step 6: Rework `_safe_provisional_export_label()` to use helpers and repeated fallback**

In `src/modules/semantic_memory.py`, replace the duplicated stats block inside `_safe_provisional_export_label()` with:

```python
    evidence = anchor_state.get("evidence", [])
    evidence_items = evidence if isinstance(evidence, list) else []
    if not candidate:
        if not evidence_items:
            return "", "no_direct_strong_anchor_evidence"
        return "", "no_posterior_label"

    if not _has_direct_anchor_evidence(anchor_state, candidate):
        return "", "no_direct_strong_anchor_evidence"

    stats = _anchor_label_stats(anchor_state, candidate)
    weighted_score = float(stats["weighted_score"])
    frame_hits = int(stats["frame_hits"])
    high_quality_hits = int(stats["high_quality_hits"])
    max_confidence = float(stats["max_confidence"])
    best_view_quality = float(stats["best_view_quality"])
    margin = float(stats["margin"])
```

Then, after `single_frame_override`, add:

```python
    repeated_evidence_override = (
        policy.repeated_evidence_enabled
        and frame_hits >= policy.repeated_min_frame_hits
        and weighted_score >= policy.repeated_min_weighted_score
        and margin >= policy.repeated_min_score_margin
        and best_view_quality >= policy.repeated_min_view_quality
        and max_confidence >= policy.repeated_min_confidence
    )
```

Change the final success condition to:

```python
    if (
        (
            (
                (enough_frames or single_frame_override)
                and enough_quality
                and enough_score
                and enough_margin
                and enough_confidence
                and enough_view
            )
            or repeated_evidence_override
        )
    ):
        return candidate, f"safe_provisional_anchor_evidence:{reason_fields}"
```

- [ ] **Step 7: Run the focused test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_repeated_provisional_anchor_label_becomes_association_identity_guard_only -q
```

Expected: PASS.

- [ ] **Step 8: Write a failing association test for cross-label provisional guard**

Add this test in `tests/test_pipeline.py` near `test_safe_provisional_export_label_does_not_create_association_conflict`:

```python
    def test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity(self):
        from src.modules.semantic_memory import set_anchor_commit_policy, set_anchor_export_policy

        points = np.repeat(
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            64,
            axis=0,
        )
        obj = ObjectMap(
            object_id=231,
            local_pcd=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
        )

        def rug_patch(frame_id: int) -> Patch3D:
            return Patch3D(
                patch_id=frame_id,
                points=points.copy(),
                centroid=points.mean(axis=0),
                bbox_min=points.min(axis=0),
                bbox_max=points.max(axis=0),
                source_frame_id=frame_id,
                metadata={
                    "anchor_class_name": "rug",
                    "anchor_confidence": 0.55,
                    "anchor_view_quality": 0.90,
                    "anchor_label_strength": "strong",
                },
            )

        incoming_patch = Patch3D(
            patch_id=99,
            points=points.copy(),
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            source_frame_id=99,
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
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "provisional_export_fallback": False,
                }
            )
            set_anchor_export_policy(
                {
                    "enabled": True,
                    "min_frame_hits": 100,
                    "min_high_quality_hits": 100,
                    "min_weighted_score": 100.0,
                    "min_score_margin": 100.0,
                    "min_confidence": 0.95,
                    "min_view_quality": 0.95,
                    "allow_single_frame_high_confidence": False,
                    "repeated_evidence_enabled": True,
                    "repeated_min_frame_hits": 3,
                    "repeated_min_weighted_score": 1.0,
                    "repeated_min_score_margin": 0.25,
                    "repeated_min_view_quality": 0.50,
                    "repeated_min_confidence": 0.25,
                }
            )
            accumulate_anchor_semantic_vote(obj, rug_patch(1))
            accumulate_anchor_semantic_vote(obj, rug_patch(2))
            accumulate_anchor_semantic_vote(obj, rug_patch(3))

            module = AssociationModule(
                {
                    "match_threshold": 0.1,
                    "observation_identity_gate_enabled": True,
                    "observation_identity_min_patch_confidence": 0.5,
                    "semantic_conflict_gate_enabled": True,
                    "semantic_conflict_min_patch_confidence": 0.5,
                    "semantic_conflict_min_object_confidence": 0.5,
                }
            )
            result = module.process([incoming_patch], {231: obj}, SystemState().tsdf_volume)

            assert result.matched == []
            assert result.contested_object_patches == [99]
            assert result.contested_matches[0].patch_label == "sofa"
            assert result.contested_matches[0].object_label == "rug"
            assert result.contested_matches[0].reason == "cross_label_observation_identity"
        finally:
            set_anchor_commit_policy(None)
            set_anchor_export_policy(None)
```

- [ ] **Step 9: Run the association test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity -q
```

Expected: FAIL because `classify_observation_identity()` still ignores provisional association labels and lets the sofa patch match the rug object.

- [ ] **Step 10: Use association identity helper in observation identity**

In `src/modules/observation_identity.py`, change the import:

```python
from src.modules.semantic_memory import object_association_identity_semantic_label, object_semantic_commit_state
```

Change `object_identity_label()` to:

```python
def object_identity_label(obj: ObjectMap) -> str:
    """Return semantic identity used for association blocking."""
    return normalize_label(object_association_identity_semantic_label(obj))
```

Inside `classify_observation_identity()`, replace:

```python
    commit_state = object_semantic_commit_state(obj)
    if str(commit_state.get("semantic_state", "")) == "committed":
        object_label = normalize_label(commit_state.get("committed_label", ""))
    else:
        object_label = ""
```

with:

```python
    commit_state = object_semantic_commit_state(obj)
    object_label = object_identity_label(obj)
```

Keep this existing relation logic unchanged:

```python
    if not object_label:
        relation = "uncommitted_object" if str(commit_state.get("posterior_label", "")).strip() else "unlabeled_object"
```

- [ ] **Step 11: Use association identity helper in legacy semantic conflict helper**

In `src/modules/association.py`, change:

```python
from src.modules.semantic_memory import object_identity_semantic_label
```

to:

```python
from src.modules.semantic_memory import object_association_identity_semantic_label
```

In `_semantic_conflict_reason()`, replace:

```python
        object_label = object_identity_semantic_label(obj).strip()
```

with:

```python
        object_label = object_association_identity_semantic_label(obj).strip()
```

- [ ] **Step 12: Run the association test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity -q
```

Expected: PASS.

- [ ] **Step 13: Run legacy safety test to verify single-frame safe provisional still does not block association**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_safe_provisional_export_label_does_not_create_association_conflict -q
```

Expected: PASS. This preserves the existing behavior that one high-confidence safe-provisional export alone does not become an association identity guard.

- [ ] **Step 14: Commit Task 2**

Run:

```bash
git add src/modules/semantic_memory.py src/modules/observation_identity.py src/modules/association.py tests/test_pipeline.py
git commit -m "fix: guard association with repeated provisional semantics"
```

## Task 3: Configure Repeated-Evidence Export For Fast High-IoU Mode

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `configs/room0_surface_gate_fast_high_iou_4090.yaml`

- [ ] **Step 1: Extend semantic memory config test**

In `tests/test_pipeline.py`, update `TestPipeline.test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend` by adding these assertions after the existing semantic export enabled assertion:

```python
        anchor_export = config["semantic_memory"]["anchor_export"]
        assert anchor_export["repeated_evidence_enabled"] is True
        assert anchor_export["repeated_min_frame_hits"] == 3
        assert anchor_export["repeated_min_weighted_score"] == 1.0
        assert anchor_export["repeated_min_score_margin"] == 0.25
        assert anchor_export["repeated_min_view_quality"] == 0.50
        assert anchor_export["repeated_min_confidence"] == 0.25
```

- [ ] **Step 2: Run config test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend -q
```

Expected: FAIL with `KeyError: 'repeated_evidence_enabled'` or an assertion failure.

- [ ] **Step 3: Add repeated evidence thresholds to fast high-IoU config**

In `configs/room0_surface_gate_fast_high_iou_4090.yaml`, under `semantic_memory.anchor_export`, add:

```yaml
    repeated_evidence_enabled: true
    repeated_min_frame_hits: 3
    repeated_min_weighted_score: 1.0
    repeated_min_score_margin: 0.25
    repeated_min_view_quality: 0.50
    repeated_min_confidence: 0.25
```

The complete block should include both the existing strict path and the repeated evidence path:

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
    repeated_evidence_enabled: true
    repeated_min_frame_hits: 3
    repeated_min_weighted_score: 1.0
    repeated_min_score_margin: 0.25
    repeated_min_view_quality: 0.50
    repeated_min_confidence: 0.25
```

- [ ] **Step 4: Run config test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend -q
```

Expected: PASS.

- [ ] **Step 5: Run focused semantic/association regression tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_repeated_provisional_anchor_label_becomes_association_identity_guard_only \
  tests/test_pipeline.py::TestPipeline::test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity \
  tests/test_pipeline.py::TestPipeline::test_safe_provisional_export_label_does_not_create_association_conflict \
  tests/test_pipeline.py::TestPipeline::test_safe_provisional_anchor_exports_without_identity_commit \
  tests/test_pipeline.py::TestPipeline::test_contextual_overlap_evidence_never_exports_as_safe_provisional \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add configs/room0_surface_gate_fast_high_iou_4090.yaml tests/test_pipeline.py
git commit -m "config: export repeated provisional anchor evidence"
```

## Task 4: Run Broader Unit Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Run focused module test suite**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_anchor_guided_sam.py \
  tests/test_runtime_vis.py \
  tests/test_provisional_pool.py \
  tests/test_dual_map.py \
  tests/test_async_refinement.py \
  tests/test_pipeline.py \
  tests/test_object_anchor.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: If tests fail, isolate and fix only the failed contract**

For each failing test, run the single test:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity -q
```

Expected: PASS after a targeted fix. If a different test failed in Step 1, run that exact failing node id from pytest output instead of the example above; do not loosen tests without confirming the new behavior matches this plan.

- [ ] **Step 3: Commit verification fixes if any**

If Task 4 required source/test edits, run:

```bash
git add src tests configs
git commit -m "test: cover fast high-iou identity guard regressions"
```

If Task 4 required no edits, do not create an empty commit.

## Task 5: Run 20f Fast Smoke To Verify Worker Speed And Semantic State

**Files:**
- No source edits expected.
- Output: `outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_20f_smoke`

- [ ] **Step 1: Start the 20f smoke run in tmux**

Run:

```bash
tmux new-session -d -s room0_fhiou_v2_s10_20f "/bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && experiment=20260601_room0_fast_high_iou_v2_stride10_20f_smoke && log=outputs/tmp_validation/${experiment}.log && status=outputs/tmp_validation/${experiment}.status && { echo experiment=${experiment}; echo started_at=$(date --iso-8601=seconds); echo worktree=$(pwd); /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml --output-root outputs/tmp_validation --experiment-name ${experiment} --num-frames 20 --frame-stride 10 --proposal-backend precomputed --proposal-device cuda --fast-eval; rc=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=${rc}; echo exit_status=${rc} > ${status}; } > ${log} 2>&1'"
```

- [ ] **Step 2: Poll the smoke run until it exits**

Run:

```bash
tmux capture-pane -pt room0_fhiou_v2_s10_20f
```

Expected while running: frame progress lines.

When it finishes, run:

```bash
cat outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_20f_smoke.status
```

Expected: `exit_status=0`.

- [ ] **Step 3: Extract smoke timing and semantic diagnostics**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_20f_smoke")
rep = json.loads((root / "room0/run_report.json").read_text())
rows = [json.loads(line) for line in (root / "room0/frame_metrics.jsonl").read_text().splitlines() if line.strip()]

def mean_stage(name):
    vals = [float(row.get("stage_timings", {}).get(name, 0.0)) for row in rows]
    return sum(vals) / max(len(vals), 1)

print({
    "frames": rep["frame_count"],
    "miou": round(float(rep["miou"]), 4),
    "fmiou": round(float(rep["fmiou"]), 4),
    "objects": rep["final_object_count"],
    "unanchored": rep["anchor_voted_unanchored_count_total"],
    "mean_yoloe_supplemental_s": round(mean_stage("yoloe_supplemental"), 4),
    "mean_proposal_generation_s": round(mean_stage("proposal_generation"), 4),
    "audit_summary": rep.get("final_object_semantic_audit_summary", {}),
})
PY
```

Expected:
- `mean_yoloe_supplemental_s` is close to prior worker runs, ideally below `0.60s/frame` after worker warmup.
- `audit_summary.semantic_state_counts` still contains provisional objects, but export unlabeled count should not dominate strong repeated evidence after longer runs.

- [ ] **Step 4: If YOLOE worker falls back, inspect logs before continuing**

Run:

```bash
rg -n "worker failed|worker_fallback|YOLOE-Seg worker|Traceback|error" outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_20f_smoke.log
```

Expected: no worker failure lines. If worker failure appears, fix the worker issue before running longer experiments.

## Task 6: Run 200f Stride10 Validation

**Files:**
- No source edits expected.
- Output: `outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f`

- [ ] **Step 1: Start the 200f validation run**

Run:

```bash
tmux new-session -d -s room0_fhiou_v2_s10_200f "/bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && experiment=20260601_room0_fast_high_iou_v2_stride10_200f && log=outputs/tmp_validation/${experiment}.log && status=outputs/tmp_validation/${experiment}.status && { echo experiment=${experiment}; echo started_at=$(date --iso-8601=seconds); echo worktree=$(pwd); /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml --output-root outputs/tmp_validation --experiment-name ${experiment} --num-frames 200 --frame-stride 10 --proposal-backend precomputed --proposal-device cuda --fast-eval; rc=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=${rc}; echo exit_status=${rc} > ${status}; } > ${log} 2>&1'"
```

- [ ] **Step 2: Poll until complete**

Run:

```bash
tmux capture-pane -pt room0_fhiou_v2_s10_200f
```

Then:

```bash
cat outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f.status
```

Expected: `exit_status=0`.

- [ ] **Step 3: Compare against 200f/stride10 baselines**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("outputs/tmp_validation")
experiments = [
    "20260526_room0_semantic_vote_weak_structure_stride10_200f",
    "20260527_room0_contained_gate_stride10_200f",
    "20260531_room0_surface_gate_fast_high_iou_stride1_2000f_fast",
    "20260601_room0_fast_high_iou_v2_stride10_200f",
]
classes = ["wall", "ceiling", "blinds", "window", "rug", "sofa", "blanket", "pillar", "plant-stand", "wall-plug", "floor", "door"]
for exp in experiments:
    root = base / exp
    if not root.exists():
        print(exp, "MISSING")
        continue
    rep = json.loads((root / "room0/run_report.json").read_text())
    rows = [json.loads(line) for line in (root / "room0/frame_metrics.jsonl").read_text().splitlines() if line.strip()] if (root / "room0/frame_metrics.jsonl").exists() else []
    def mean_stage(name):
        vals = [float(row.get("stage_timings", {}).get(name, 0.0)) for row in rows]
        return sum(vals) / max(len(vals), 1)
    per = {item["class_name"]: item for item in rep.get("per_class", [])}
    print("\\n", exp)
    print({
        "miou": round(float(rep["miou"]), 4),
        "fmiou": round(float(rep["fmiou"]), 4),
        "objects": rep["final_object_count"],
        "unanchored": rep.get("anchor_voted_unanchored_count_total"),
        "yoloe_s": round(mean_stage("yoloe_supplemental"), 4),
        "proposal_s": round(mean_stage("proposal_generation"), 4),
        "object_update_s": round(mean_stage("object_update"), 4),
    })
    print({cls: round(float(per.get(cls, {}).get("iou", 0.0)), 4) for cls in classes})
PY
```

Expected:
- YOLOE mean should be near worker baseline rather than `4.7s/frame`.
- mIoU should not regress below the previous fast high-IoU smoke trajectory.
- Structural classes should improve relative to the latest full fast run if unlabeled provisional export was the primary cause.

- [ ] **Step 4: Decide whether to run 2000f**

Proceed to Task 7 only if:
- `exit_status=0`
- YOLOE worker is active and mean `yoloe_supplemental` is below `0.60s/frame`
- mIoU and structural IoUs do not show a clear new regression versus 0526 stride10 200f and 0527 contained-gate 200f baselines

If this gate fails, return to Task 2 or Task 3 with evidence from the report; do not start a long 2000f run.

## Task 7: Run 2000f Full Fast Evaluation

**Files:**
- No source edits expected.
- Output: `outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride1_2000f_fast`

- [ ] **Step 1: Start the 2000f fast run**

Run:

```bash
tmux new-session -d -s room0_fhiou_v2_s1_2000f "/bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && experiment=20260601_room0_fast_high_iou_v2_stride1_2000f_fast && log=outputs/tmp_validation/${experiment}.log && status=outputs/tmp_validation/${experiment}.status && { echo experiment=${experiment}; echo started_at=$(date --iso-8601=seconds); echo worktree=$(pwd); /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml --output-root outputs/tmp_validation --experiment-name ${experiment} --num-frames 2000 --frame-stride 1 --proposal-backend precomputed --proposal-device cuda --fast-eval; rc=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=${rc}; echo exit_status=${rc} > ${status}; } > ${log} 2>&1'"
```

- [ ] **Step 2: Poll until complete**

Run periodically:

```bash
tmux capture-pane -pt room0_fhiou_v2_s1_2000f
```

After completion:

```bash
cat outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride1_2000f_fast.status
```

Expected: `exit_status=0`.

- [ ] **Step 3: Extract final FPS, metrics, and key class IoUs**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
from datetime import datetime

root = Path("outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride1_2000f_fast")
rep = json.loads((root / "room0/run_report.json").read_text())
res = json.loads((root / "replica/results.json").read_text())
rows = [json.loads(line) for line in (root / "room0/frame_metrics.jsonl").read_text().splitlines() if line.strip()]
per = {item["class_name"]: item for item in rep.get("per_class", [])}

log = (root.with_suffix(".log")).read_text().splitlines()
started = next((line.split("=", 1)[1] for line in log if line.startswith("started_at=")), "")
finished = next((line.split("=", 1)[1] for line in log if line.startswith("finished_at=")), "")
wall_s = None
if started and finished:
    wall_s = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()

def mean_stage(name):
    vals = [float(row.get("stage_timings", {}).get(name, 0.0)) for row in rows]
    return sum(vals) / max(len(vals), 1)

print({
    "run_root": str(root),
    "wall_s": wall_s,
    "wall_fps": round(rep["frame_count"] / wall_s, 4) if wall_s else None,
    "miou": round(float(res["miou"]), 4),
    "macc": round(float(res["macc"]), 4),
    "fmiou": round(float(res["fmiou"]), 4),
    "fmacc": round(float(res["fmacc"]), 4),
    "objects": rep["final_object_count"],
    "pool_points": rep["pool_point_count"],
    "dense_surface_points": rep["dense_surface_point_count"],
    "unanchored": rep["anchor_voted_unanchored_count_total"],
    "mean_yoloe_s": round(mean_stage("yoloe_supplemental"), 4),
    "mean_proposal_s": round(mean_stage("proposal_generation"), 4),
    "mean_object_update_s": round(mean_stage("object_update"), 4),
    "mean_association_s": round(mean_stage("association"), 4),
    "audit_summary": rep.get("final_object_semantic_audit_summary", {}),
})
print({cls: round(float(per.get(cls, {}).get("iou", 0.0)), 4) for cls in ["wall", "ceiling", "blinds", "window", "rug", "sofa", "blanket", "pillar", "plant-stand", "wall-plug", "floor", "door"]})
PY
```

Expected:
- FPS is materially better than the latest `0.0653 FPS`.
- `mean_yoloe_s` is materially below `4.7618s/frame`.
- mIoU moves toward the 0526 baseline rather than remaining near `0.4945`.
- Structural class IoUs no longer collapse because of unlabeled provisional exports.

- [ ] **Step 4: Compare final run against full baselines**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("outputs/tmp_validation")
experiments = [
    "20260526_room0_observation_first_2b830e1_stride1_2000f",
    "20260527_room0_semantic_vote_weak_structure_stride1_2000f",
    "20260531_room0_surface_gate_fast_high_iou_stride1_2000f_fast",
    "20260601_room0_fast_high_iou_v2_stride1_2000f_fast",
]
classes = ["wall", "ceiling", "blinds", "window", "rug", "sofa", "blanket", "pillar", "plant-stand", "wall-plug", "floor", "door"]
for exp in experiments:
    root = base / exp
    if not root.exists():
        print(exp, "MISSING")
        continue
    rep = json.loads((root / "room0/run_report.json").read_text())
    res = json.loads((root / "replica/results.json").read_text())
    per = {item["class_name"]: item for item in rep.get("per_class", [])}
    print("\\n", exp)
    print({
        "miou": round(float(res["miou"]), 4),
        "macc": round(float(res["macc"]), 4),
        "fmiou": round(float(res["fmiou"]), 4),
        "fmacc": round(float(res["fmacc"]), 4),
        "objects": rep["final_object_count"],
        "pool": rep["pool_point_count"],
        "dense": rep["dense_surface_point_count"],
        "unanchored": rep.get("anchor_voted_unanchored_count_total"),
    })
    print({cls: round(float(per.get(cls, {}).get("iou", 0.0)), 4) for cls in classes})
PY
```

Expected:
- Report exact metric deltas; do not claim success unless both speed and IoU moved in the intended direction.

## Self-Review Checklist

- [ ] Spec coverage: worker speed issue is covered by Tasks 1, 5, 6, and 7.
- [ ] Spec coverage: provisional identity absorption issue is covered by Task 2.
- [ ] Spec coverage: unlabeled repeated provisional export issue is covered by Tasks 2 and 3.
- [ ] Spec coverage: unanchored SAM proposals must not inherit parent anchor labels; preserved by keeping `object_identity_semantic_label()` unchanged and only guarding association when the incoming patch has its own confident `anchor_class_name`.
- [ ] Placeholder scan: no `TBD`, `TODO`, "similar to", or vague test instructions remain.
- [ ] Type consistency: all new functions and policy fields use names shown consistently across tests and implementation.
- [ ] Verification: unit tests precede implementation, smoke precedes long run, 200f gate precedes 2000f.
