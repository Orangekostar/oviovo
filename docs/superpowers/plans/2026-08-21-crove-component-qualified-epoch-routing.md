# CROVE Component-Qualified Epoch Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve identity-qualified dynamic-state evidence while preventing that evidence from resetting geometry epochs, then compare the corrected method with a fixed no-prototype-isolation ablation on Apartment.

**Architecture:** Extend the pure temporal evidence router to report geometry and identity threshold qualification separately. `TemporalCurrentRuntime` continues to use their disjunction for dynamic-state hysteresis, but geometry integration and epoch reset consume only the geometry-qualified token. Keep the approved main-method prototype isolation unchanged; test legacy prototype adaptation only on a separate ablation branch.

**Tech Stack:** Python 3.10+, NumPy, frozen dataclasses/enums, pytest, TESSE-CD common-v2 evaluator, Khronos official evaluator, Git worktrees.

---

## File Map

- Modify `src/oviv2/temporal_evidence_router.py`: expose exact per-source motion qualification.
- Modify `tests/oviv2/test_temporal_evidence_router.py`: prove aggregate and per-source qualifications agree.
- Modify `src/oviv2/temporal_runtime.py`: restrict geometry epoch transitions to geometry-qualified evidence for A4.
- Modify `tests/oviv2/test_temporal_runtime.py`: prove identity-only motion updates `D/R` without updating `G`.
- Create a separate ablation commit modifying only `src/oviv2/temporal_runtime.py` and its focused test to restore legacy active prototype adaptation.
- Update `docs/superpowers/reports/2026-08-21-crove-evidence-routing-apartment-result.md` only after measured results exist.

## Task 1: Report Per-Source Motion Qualification

**Files:**

- Modify: `tests/oviv2/test_temporal_evidence_router.py`
- Modify: `src/oviv2/temporal_evidence_router.py`

- [ ] **Step 1: Add failing router tests**

Extend the existing source-combination test so every result asserts these exact
fields:

```python
assert result.geometry_qualifies_as_motion is geometry_qualifies
assert result.identity_qualifies_as_motion is identity_qualifies
assert result.qualifies_as_motion is (
    geometry_qualifies or identity_qualifies
)
```

Add a mixed case where admitted geometry is below threshold and qualified
identity is above threshold:

```python
def test_identity_qualification_does_not_qualify_geometry() -> None:
    result = route_active_motion_evidence(
        geometry_accepted=True,
        geometry_confidence=0.2,
        geometry_displacement_m=0.05,
        identity_qualified=True,
        appearance_similarity=0.9,
        identity_displacement_m=0.2,
        config=dynamic_config(),
    )

    assert result.source is MotionEvidenceSource.COMBINED
    assert result.geometry_qualifies_as_motion is False
    assert result.identity_qualifies_as_motion is True
    assert result.qualifies_as_motion is True
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_evidence_router.py::test_identity_qualification_does_not_qualify_geometry
```

Expected: FAIL because `RoutedMotionEvidence` has no per-source qualification
fields.

- [ ] **Step 3: Add exact fields and invariants**

Add these frozen result fields:

```python
geometry_qualifies_as_motion: bool
identity_qualifies_as_motion: bool
```

In `__post_init__`, require exact booleans and enforce:

```python
if self.qualifies_as_motion is not (
    self.geometry_qualifies_as_motion
    or self.identity_qualifies_as_motion
):
    raise ValueError("aggregate qualification must match source qualifications")
if self.geometry_qualifies_as_motion and self.source not in {
    MotionEvidenceSource.GEOMETRY,
    MotionEvidenceSource.COMBINED,
}:
    raise ValueError("geometry qualification requires admitted geometry")
if self.identity_qualifies_as_motion and self.source not in {
    MotionEvidenceSource.IDENTITY,
    MotionEvidenceSource.COMBINED,
}:
    raise ValueError("identity qualification requires admitted identity")
```

Compute qualifications from each source-bound `(displacement, confidence)`
pair before selecting the aggregate token:

```python
geometry_qualifies = bool(
    geometry_admitted
    and geometry_displacement >= normalized_config.displacement_floor_m
    and normalized_geometry_confidence
    >= normalized_config.minimum_motion_confidence
)
identity_qualifies = bool(
    identity_admitted
    and identity_displacement >= normalized_config.displacement_floor_m
    and identity_confidence >= normalized_config.minimum_motion_confidence
)
```

Pass both fields to `RoutedMotionEvidence` and keep the existing aggregate
selection and tie-breaking unchanged.

- [ ] **Step 4: Run router tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/oviv2/test_temporal_evidence_router.py
```

Expected: all router tests PASS.

## Task 2: Enforce Component-Qualified Geometry Updates

**Files:**

- Modify: `tests/oviv2/test_temporal_runtime.py`
- Modify: `src/oviv2/temporal_runtime.py`

- [ ] **Step 1: Add the failing A4 state-factorization test**

Extend
`test_a4_consecutive_active_identity_motion_recovers_dynamic_without_geometry_confidence`
to preserve the initial geometry epoch and assert after both identity-only
motion frames:

```python
before = runtime.state.geometry.current(entity_id)

assert runtime.state.geometry.current(entity_id) == before
assert first.diagnostics.epoch_reset_trigger_count == 0
assert second.diagnostics.epoch_reset_trigger_count == 0
assert first.export.samples[0].dynamic_state is DynamicState.STATIC
assert second.export.samples[0].dynamic_state is DynamicState.DYNAMIC
```

Add a geometry-accepted mixed-token test whose identity token qualifies but
whose geometry displacement is below the floor. Assert that the observation is
integrated into the current epoch and that no new epoch is created.

```python
def test_a4_identity_motion_does_not_reset_stationary_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oviv2.temporal_runtime as module
    from src.oviv2.temporal_geometry import MotionDecision, ObjectMotionEstimate

    runtime = _runtime(_config(ExecutionProfile.A4))
    entity_id = _confirm(runtime)
    before = runtime.state.geometry.current(entity_id)

    def weak_stationary_geometry(*args, **kwargs):
        pose = np.array(kwargs["previous_object_to_world"], copy=True)
        pose[0, 3] += 0.05
        return ObjectMotionEstimate(
            pose,
            MotionDecision.ICP_ACCEPTED,
            0.2,
            0.01,
        )

    monkeypatch.setattr(module, "estimate_object_motion", weak_stationary_geometry)
    frame = _frame(2, depth=1.2)
    result = runtime.process_frame(
        frame,
        (_observation(frame, centroid_z=1.2),),
    )
    after = runtime.state.geometry.current(entity_id)

    assert after.epoch_id == before.epoch_id
    assert float(after.submap.weights.sum()) > float(before.submap.weights.sum())
    assert result.diagnostics.epoch_reset_trigger_count == 0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_runtime.py::test_a4_consecutive_active_identity_motion_recovers_dynamic_without_geometry_confidence
```

Expected: FAIL because the current A4 rejected-geometry branch starts a new
epoch from high-confidence identity.

- [ ] **Step 3: Separate dynamic and geometry transitions**

Keep dynamic hysteresis unchanged:

```python
dynamic = advance_dynamic_state(
    previous_dynamic,
    accepted_motion=routed_motion.accepted,
    displacement_m=routed_motion.displacement_m,
    confidence=routed_motion.confidence,
    config=self.config.dynamic_state,
)
```

For `ExecutionProfile.A4`, change only geometry ownership:

```python
geometry_qualifies_as_motion = (
    routed_motion.geometry_qualifies_as_motion
)
```

Apply these exact rules:

1. Rejected geometry plus unqualified identity preserves the existing forced-new behavior.
2. Rejected geometry plus qualified identity keeps the identity assignment and dynamic evidence but leaves the current geometry epoch unchanged.
3. Accepted geometry resets or integrates a moving epoch only when `geometry_qualifies_as_motion` is true.
4. Stationary integration uses `geometry_displacement`, never the selected identity displacement.
5. A2/A3 retain their existing rejected-motion and geometry-only behavior.

Do not change lifecycle, association, appearance prototype, current-center
readout, cumulative state, checkpoint schema, or configuration schema.

- [ ] **Step 4: Run focused runtime tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_runtime.py::test_a4_consecutive_active_identity_motion_recovers_dynamic_without_geometry_confidence \
  tests/oviv2/test_temporal_runtime.py::test_translation_rejection_with_high_identity_confidence_starts_new_epoch
```

Expected: A4 factorization and legacy A2/A3 behavior all PASS.

- [ ] **Step 5: Run the temporal unit suite**

Run:

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_evidence_router.py \
  tests/oviv2/test_temporal_runtime.py
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the main-method correction**

```bash
git add \
  src/oviv2/temporal_evidence_router.py \
  src/oviv2/temporal_runtime.py \
  tests/oviv2/test_temporal_evidence_router.py \
  tests/oviv2/test_temporal_runtime.py
git commit -m "fix: isolate identity motion from geometry epochs"
```

## Task 3: Verify Protected Boundaries

**Files:** none

- [ ] **Step 1: Run focused and integration contracts**

Run:

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_evidence_router.py \
  tests/oviv2/test_temporal_runtime.py \
  tests/evaluation/test_run_oviv2_tesse_cd_v2.py \
  tests/evaluation/test_build_oviv2_tesse_search_preflight.py
```

Expected: all tests PASS.

- [ ] **Step 2: Run the frozen T1 source gate**

Run:

```bash
python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py \
  --verify-source-manifest \
  configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json
```

Expected: exit status 0.

- [ ] **Step 3: Verify syntax and diff integrity**

Run:

```bash
python -m py_compile \
  src/oviv2/temporal_evidence_router.py \
  src/oviv2/temporal_runtime.py
git diff --check
git status --short
```

Expected: compilation and diff checks succeed; only planned files differ from
the pre-task commit.

## Task 4: Create The Fixed Prototype-Ablation Branch

**Files:**

- Modify on ablation branch: `src/oviv2/temporal_runtime.py`
- Modify on ablation branch: `tests/oviv2/test_temporal_runtime.py`

- [ ] **Step 1: Create an isolated ablation worktree**

Create branch `codex/crove-no-prototype-isolation-ablation-20260821` from the
main-method correction commit in a sibling worktree.

- [ ] **Step 2: Write the failing ablation test**

On the ablation branch, change the existing unqualified same-model test to
`test_same_model_weak_identity_match_updates_prototype` and assert that legacy
active adaptation updates both the entity prototype and the identity-bank
prototype.

- [ ] **Step 3: Verify RED**

Run:

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_runtime.py::test_same_model_weak_identity_match_updates_prototype
```

Expected: FAIL because the approved main method isolates the prototype.

- [ ] **Step 4: Restore only legacy active prototype adaptation**

Replace the A4 qualification block with the pre-router behavior:

```python
prototype, feature_model_id = _prototype_update(
    old.image_prototype,
    old.feature_model_id,
    observation,
)
```

Do not alter motion routing, epoch qualification, lifecycle, association,
readout centers, cumulative state, or schemas.

- [ ] **Step 5: Verify and commit the ablation**

Run the same temporal unit suite and frozen T1 source gate as Tasks 2-3.
Commit with:

```bash
git commit -am "ablation: restore active prototype adaptation"
```

## Task 5: Run Apartment Selection

**Files:** none

- [ ] **Step 1: Deploy source-bound variants**

Deploy the main-method correction to node106 and the fixed prototype ablation
to node101. Verify each remote repository tree, algorithm hash, configuration
hash, and focused tests before launch.

- [ ] **Step 2: Run both causal captures**

Use `--causal-table-metrics-only` with the checked Apartment A4 configuration.
Do not stop or overwrite any existing run. Give each output a unique run ID.

- [ ] **Step 3: Postprocess source-bound metrics**

For every `PASS` capture, export the temporal artifact, run common-v2, prepare
the Khronos bridge, import it, and run the repeated official evaluator. Record
all artifact hashes and reject any non-finite or non-repeatable metric.

- [ ] **Step 4: Apply the existing promotion gates**

Promote a variant only when all conditions hold against A6:

- Dyn. F1 or Chg. F1 improves by at least 0.01 absolute.
- Obj. F1 and current mIoU each regress by at most 0.01 absolute.
- Ghost increases by at most 0.02 absolute.
- Fragmentation diagnostics do not regress.
- T1 source and exactness gates pass.

If neither variant passes, retain A6 and do not run Office. If one passes,
freeze that exact source/config identity before the single Office run.

- [ ] **Step 5: Update the diagnostic report**

Append only measured values and hashes to
`docs/superpowers/reports/2026-08-21-crove-evidence-routing-apartment-result.md`.
Do not update main paper tables until formal promotion succeeds.
