# OVIV2 T2 Baseline Dominance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the frozen cumulative T1 readout exactly while making the temporal readout causal, correctly evaluated, and capable of exceeding every strongest non-oracle T2 baseline on the frozen protocol.

**Architecture:** Keep `Oviv2Runtime` and all cumulative artifacts unchanged. Route the current-map branch through profile-locked readouts and isolate six temporal concerns: causal per-frame export, proposal recovery, persistent identity memory, disposable geometry epochs, explicit motion decisions, and reversible background ownership. Promotion is fail-closed: exact T1 equality, Apartment-only development, T4 budgets, then a one-shot frozen Office transfer.

**Tech Stack:** Python 3.11, NumPy, Open3D, pytest, JSON/JSONL artifacts, existing TESSE-CD/common-v2/Khronos evaluators, three CUDA devices for independent candidate runs.

---

## Scope And Success Contract

- Frozen and never edited by this plan: GT artifacts, causal schedule, label space, official Khronos evaluator/configuration, common-v2 evaluator, checkpoint set, metric formulas, non-OVIV2 baselines, and cumulative T1 algorithm/configuration.
- T1 contract: v1 direct cumulative artifacts equal the A0 cumulative audit byte-for-byte; A0-A4 cumulative audits equal A0 at every checkpoint and final export.
- Apartment development gates: object F1 does not fall below `0.469`; current-map mIoU does not fall below the frozen Apartment A0 value; all five target metrics improve over the same-protocol Apartment strongest baseline.
- Final two-scene gates: Dynamic F1 `> 0.822`, Change F1 `> 0.302`, ghost rate `< 0.218`, background F@5cm `> 0.163`, recovery latency `< 450` frames, current-map mIoU `>= 0.200`, object F1 `>= 0.469`.
- T4 upper bounds: total `<= 6.42 s/frame`, query mean `<= 11.92 ms`, query p95 `<= 12.12 ms`, GPU `<= 12.76 GB`, RAM `<= 9.36 GB`, map `<= 46.77 MB`.
- No weighted score may compensate for a failed hard gate. An ineligible metric is reported as `N/A`, never as an improvement.

## Dependency And Parallelization Map

```text
Batch A (parallel, disjoint files)
  Task 1 profile/config contract
  Task 3 cumulative exactness verifier

Batch B (parallel after Task 1, disjoint files)
  Task 2 causal export model
  Task 4 reversible background ledger
  Task 5 motion decisions + geometry epochs
  Task 6 proposal recovery + identity memory

Batch C (after Tasks 2, 4, 5, 6)
  Task 7 temporal runtime integration

Batch D (after Task 7)
  Task 8 snapshot/readout integration

Batch E (sequential after Task 8; exporter files preserve existing dirty edits)
  Task 9 per-frame runner export
  Task 10 artifact and Khronos bridge correction

Batch F (sequential because Tasks 11-12 both harden packaging)
  Task 11 strengthened T1 transaction and exact artifact gates
  Task 12 search/tuning/package fail-closed gates
  Task 13 trusted T4 collection and verification
  Task 14 focused and full verification
  Task 15 Apartment experiments on three GPUs
  Task 16 T1/T4 freeze and one-shot Office transfer
```

Tasks that touch any currently modified exporter, bridge, packager, or occlusion file must first inspect and preserve the existing working-tree diff. Each subagent commits only its declared files; the integrator reviews every commit before continuing.

### Task 1: Freeze Profile And Configuration Semantics

**Files:**
- Modify: `src/oviv2/temporal_config.py`
- Modify: `configs/oviv2_tesse_cd_apartment_v2.json`
- Modify: `configs/oviv2_tesse_cd_office_v2.json`
- Test: `tests/oviv2/test_temporal_config.py`

- [ ] Add failing tests that require exact keys for proposal, identity, dynamic-state, motion, geometry-epoch, and background-ledger configuration, reject unknown keys, and expand every profile to a canonical component map.

```python
def test_a4_components_are_profile_locked() -> None:
    config = parse_temporal_readout_config(_valid_config("a4"))
    assert config.execution_profile.components == {
        "lifecycle": "probabilistic_hysteresis",
        "proposal": "temporal_recovery",
        "identity": "dormant_reid",
        "geometry": "object_submap_epoch",
        "background": "reversible_ledger",
        "motion": "gated_icp",
    }


def test_profile_component_override_is_rejected() -> None:
    raw = _valid_config("a2")
    raw["components"]["identity"] = "dormant_reid"
    with pytest.raises(ValueError, match="profile components"):
        parse_temporal_readout_config(raw)
```

- [ ] Run the focused tests and confirm they fail because the new keys and component map do not exist.

Run: `python -m pytest -q tests/oviv2/test_temporal_config.py`

- [ ] Implement frozen dataclasses `TemporalProposalConfig`, `TemporalIdentityConfig`, `TemporalDynamicConfig`, `TemporalMotionConfig`, `TemporalGeometryEpochConfig`, and `TemporalBackgroundLedgerConfig`; validate finite numeric bounds, exact booleans, capacity relations, pre-registered search grids, and exact JSON keys. Restrict dynamic search to consecutive-motion `{2,3,4}`, displacement `{0.05,0.10,0.15}` m, confidence `{0.6,0.7,0.8}`, and static-off `{5,10,20}` frames; restrict ledger commit support to `{2,3,4}` frames and `{2,3}` distinct view bins.

- [ ] Change `ExecutionProfile` so A0-A4 uniquely derive all component axes. Keep A0 as v1 reference, A1 as lifecycle-only overlay, A2 as proposal/active identity/translation, A3 as A2 plus reversible background, and A4 as dormant re-ID plus gated ICP with explicit motion rejection.

- [ ] Update both v2 configs with identical mechanism parameters; only scene/input bindings may differ. Recompute their canonical `algorithm_hash` through the existing config loader, never by hand.

- [ ] Run focused tests and deterministic serialization tests.

Run: `python -m pytest -q tests/oviv2/test_temporal_config.py tests/oviv2/test_reference_readout.py`

- [ ] Commit only the four declared files.

```bash
git add src/oviv2/temporal_config.py tests/oviv2/test_temporal_config.py configs/oviv2_tesse_cd_apartment_v2.json configs/oviv2_tesse_cd_office_v2.json
git commit -m "feat: freeze temporal profile contracts"
```

### Task 2: Define Causal Per-Frame Temporal Export Records

**Files:**
- Create: `src/oviv2/temporal_export.py`
- Create: `tests/oviv2/test_temporal_export.py`
- Modify: `src/oviv2/reference_readout.py`
- Test: `tests/oviv2/test_reference_readout.py`

- [ ] Add failing tests for strict immutable observation samples plus independent lifecycle/readout events, both bound to a native frame and timestamp.

```python
def test_export_sample_requires_explicit_dynamic_state() -> None:
    with pytest.raises(TypeError, match="dynamic_state"):
        TemporalExportSample(
            frame_index=7,
            timestamp_ns=700,
            entity_id=1,
            centroid_xyz=(0.0, 0.0, 1.0),
            observation_count=2,
            dynamic_state=None,
            motion_confidence=0.8,
            geometry_epoch=0,
            readout_valid=True,
        )


def test_export_batch_is_complete_for_processed_frame() -> None:
    batch = TemporalExportBatch(frame_index=7, timestamp_ns=700, samples=(), events=())
    assert batch.to_json_records() == ()


def test_visible_absent_event_does_not_require_an_observation_sample() -> None:
    event = TemporalLifecycleEvent(
        frame_index=7,
        timestamp_ns=700,
        entity_id=1,
        before=LifecycleState.PRESENT,
        after=LifecycleState.UNCERTAIN,
        evidence=EvidenceKind.VISIBLE_ABSENT,
        geometry_epoch=2,
        readout_valid=False,
    )
    batch = TemporalExportBatch(frame_index=7, timestamp_ns=700, samples=(), events=(event,))
    assert batch.events == (event,)


def test_dynamic_state_requires_consecutive_accepted_motion() -> None:
    state = DynamicEvidenceState.static()
    state = advance_dynamic_state(
        state, accepted_motion=True, displacement_m=0.2, confidence=0.9, config=_config()
    )
    assert state.state is DynamicState.STATIC
    state = advance_dynamic_state(
        state, accepted_motion=True, displacement_m=0.2, confidence=0.9, config=_config()
    )
    assert state.state is DynamicState.DYNAMIC


def test_rejected_motion_never_marks_entity_dynamic() -> None:
    state = advance_dynamic_state(
        DynamicEvidenceState.static(),
        accepted_motion=False,
        displacement_m=4.0,
        confidence=1.0,
        config=_config(),
    )
    assert state.state is DynamicState.STATIC
```

- [ ] Run the new tests and confirm import failure.

Run: `python -m pytest -q tests/oviv2/test_temporal_export.py`

- [ ] Implement `DynamicState` (`STATIC`, `DYNAMIC`, `UNKNOWN`), `DynamicEvidenceState`, the pure causal `advance_dynamic_state` hysteresis function, `TemporalLifecycleEvent`, `TemporalExportSample`, and `TemporalExportBatch(samples, events)`. Events are independent of observation samples so `VISIBLE_ABSENT` can invalidate an unobserved entity on the exact frame. Reject NaN, duplicate IDs per collection, non-monotonic observation counts, and any inferred default for `dynamic_state`. The bridge later treats only `DYNAMIC` as eligible; rejected motion passes `accepted_motion=False`.

- [ ] Add a read-only A0/A1 exporter tracker to `ReferenceReadout`. It derives stable tracks causally from frozen cumulative entity IDs and frame-to-frame centroids; it does not mutate cumulative entities and never uses semantic class or future frames to decide motion.

- [ ] Require the reference readout transaction snapshot/restore to include the exporter tracker and prove byte-identical retry output after an injected exception.

- [ ] Run focused tests.

Run: `python -m pytest -q tests/oviv2/test_temporal_export.py tests/oviv2/test_reference_readout.py`

- [ ] Commit the declared files.

```bash
git add src/oviv2/temporal_export.py src/oviv2/reference_readout.py tests/oviv2/test_temporal_export.py tests/oviv2/test_reference_readout.py
git commit -m "feat: add causal temporal export records"
```

### Task 3: Freeze The Cumulative Transitive Dependency Inventory

**Files:**
- Create: `configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json`
- Modify: `scripts/evaluation/verify_oviv2_dual_readout_development_gates.py`
- Test: `tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py`

- [ ] Add failing tests that reject missing, extra, symlinked, current-tree-only, or hash-mismatched cumulative dependencies and require hashes to be read from the frozen cumulative commit `8034e79d9cb853166222610981a7e6893f6cca70`. This is the last committed tree before the approved design work and includes the current exact sparse-assignment implementation.

```python
def test_source_manifest_must_match_trusted_git_objects(tmp_path: Path) -> None:
    manifest = _valid_source_manifest()
    manifest["files"]["src/oviv2/geometry.py"] = "0" * 64
    with pytest.raises(GateVerificationError, match="trusted source hash"):
        verify_source_manifest(manifest, repo=tmp_path, git=_fake_git)
```

- [ ] Run the verifier tests and confirm the new manifest contract fails.

Run: `python -m pytest -q tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py`

- [ ] Expand the allowlist to all cumulative imports used by `Oviv2Runtime`, including geometry, evidence, ownership, entities, tracking, association, visibility, dense projection, snapshot, and their local transitive imports. Fail closed on an unresolved local import.

- [ ] Add `--write-source-manifest` mode that reads bytes with `git show <base>:<path>`, writes canonical JSON atomically, and refuses a dirty destination. Generate the checked-in manifest from the trusted base.

Run: `python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py --repo . --base-commit 8034e79d9cb853166222610981a7e6893f6cca70 --write-source-manifest configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json`

- [ ] Make normal evidence generation verify both trusted-base hashes and current-worktree hashes before and after tests.

- [ ] Add `--verify-source-manifest` mode for a read-only full-closure check. It must compare paths and hashes, not only the six legacy protected files.

- [ ] Run focused tests and inspect the manifest for absolute paths, usernames, and unstable timestamps.

Run: `python -m pytest -q tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py && ! rg -n '/home/|generated_at|TODO|TBD' configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json`

- [ ] Commit the declared files.

```bash
git add configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json scripts/evaluation/verify_oviv2_dual_readout_development_gates.py tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py
git commit -m "test: freeze cumulative dependency hashes"
```

### Task 4: Implement A Reversible Background Ownership Ledger

**Files:**
- Create: `src/oviv2/temporal_background_ledger.py`
- Create: `tests/oviv2/test_temporal_background_ledger.py`
- Modify: `src/oviv2/temporal_background.py`
- Test: `tests/oviv2/test_temporal_background.py`

- [ ] Add failing tests for provisional evidence, commit thresholds, cancellation on present/occluded evidence, deterministic block rebuild, and capacity overflow that rejects evidence without modifying committed background.

```python
def test_overflow_never_changes_committed_blocks(ledger: ReversibleBackgroundLedger) -> None:
    before = ledger.committed_digest()
    result = ledger.stage(_evidence(block_count=ledger.config.maximum_journal_blocks + 1))
    assert result is LedgerDecision.REJECTED_CAPACITY
    assert ledger.committed_digest() == before


def test_visible_absent_stages_then_commits_revealed_surface(ledger: ReversibleBackgroundLedger) -> None:
    ledger.stage(_absent_evidence(entity_id=1, epoch=2, frame_id=10, view_id="v0"))
    assert ledger.provisional_count == 1
    ledger.stage(_absent_evidence(entity_id=1, epoch=2, frame_id=12, view_id="v1"))
    assert ledger.committed_generation == 1
```

- [ ] Run the tests and confirm the ledger module is missing.

Run: `python -m pytest -q tests/oviv2/test_temporal_background_ledger.py`

- [ ] Implement immutable contribution records keyed by `(entity_id, geometry_epoch, frame_id, block_key)`, per-block committed observations, and deterministic rebuild from retained records. Commit only after configured distinct-view and frame-gap thresholds.

- [ ] On `PRESENT` or `OCCLUDED`, discard matching provisional contributions. On overflow or rebuild error, return an explicit rejection and leave both committed volume and journal digest unchanged.

- [ ] Keep `TemporalBackgroundVolume` as the low-level TSDF holder; expose clone/rebuild primitives needed by the ledger without changing cumulative TSDF code.

- [ ] Run focused tests including deterministic clone/canonical-state checks.

Run: `python -m pytest -q tests/oviv2/test_temporal_background.py tests/oviv2/test_temporal_background_ledger.py`

- [ ] Commit the declared files.

```bash
git add src/oviv2/temporal_background.py src/oviv2/temporal_background_ledger.py tests/oviv2/test_temporal_background.py tests/oviv2/test_temporal_background_ledger.py
git commit -m "feat: add reversible background ledger"
```

### Task 5: Make Motion Rejection And Geometry Epochs Explicit

**Files:**
- Modify: `src/oviv2/temporal_geometry.py`
- Create: `src/oviv2/temporal_epoch.py`
- Modify: `tests/oviv2/test_temporal_geometry.py`
- Create: `tests/oviv2/test_temporal_epoch.py`

- [ ] Replace `used_icp: bool` tests with the three-state decision contract and add a regression test for over-limit displacement.

```python
def test_translation_over_motion_limit_is_rejected() -> None:
    result = estimate_object_translation(
        _submap_at(0.0), _points_at(2.0), (2.0, 0.0, 0.0), _config(maximum_motion_m=0.5)
    )
    assert result.decision is MotionDecision.REJECTED
    assert np.array_equal(result.object_to_world, np.eye(4))


def test_rejected_motion_cannot_integrate_into_existing_epoch() -> None:
    epoch = _epoch(epoch_id=3)
    with pytest.raises(ValueError, match="rejected motion"):
        epoch.integrate(_rejected_estimate(), _points_at(2.0), frame_id=8)
```

- [ ] Run the tests and confirm the old silent fallback behavior fails them.

Run: `python -m pytest -q tests/oviv2/test_temporal_geometry.py tests/oviv2/test_temporal_epoch.py`

- [ ] Add `MotionDecision.ICP_ACCEPTED`, `TRANSLATION_ACCEPTED`, and `REJECTED`; preserve fitness/rmse diagnostics but remove all behavioral inference from `used_icp`.

- [ ] Implement `GeometryEpoch(entity_id, epoch_id, object_to_world, submap, readout_valid)`. A trustworthy `VISIBLE_ABSENT` invalidates readout immediately; `OCCLUDED`, `OUT_OF_VIEW`, and `DEPTH_UNKNOWN` preserve it. A rejected motion cannot integrate into the old epoch.

- [ ] Add deterministic `start_new_epoch()` behavior. High-confidence re-ID retains entity ID but increments the epoch and initializes geometry only from the current causal observation.

- [ ] Run focused tests.

Run: `python -m pytest -q tests/oviv2/test_temporal_geometry.py tests/oviv2/test_temporal_epoch.py`

- [ ] Commit the declared files.

```bash
git add src/oviv2/temporal_geometry.py src/oviv2/temporal_epoch.py tests/oviv2/test_temporal_geometry.py tests/oviv2/test_temporal_epoch.py
git commit -m "fix: reject unsafe motion across geometry epochs"
```

### Task 6: Add Proposal Recovery And Persistent Identity Memory

**Files:**
- Create: `src/oviv2/temporal_proposals.py`
- Create: `src/oviv2/temporal_identity.py`
- Create: `tests/oviv2/test_temporal_proposals.py`
- Create: `tests/oviv2/test_temporal_identity.py`
- Modify: `src/oviv2/temporal_association.py`
- Test: `tests/oviv2/test_temporal_association.py`

- [ ] Add failing tests for causal residual proposals, deterministic proposal ordering, active/uncertain-only matching in A2/A3, dormant-bank matching only in A4, and identity retention after geometry eviction.

```python
def test_dormant_identity_survives_geometry_eviction() -> None:
    memory = IdentityMemory.from_entity(_entity(7))
    bank = IdentityMemoryBank(maximum_identities=4)
    bank.put(memory)
    memory.release_geometry()
    assert bank.get(7).semantic_probabilities == memory.semantic_probabilities


def test_recovered_proposal_uses_only_current_and_past_frames() -> None:
    first = recover_proposals(_prefix(through_frame=8), frame_id=8, config=_config())
    second = recover_proposals(_prefix(through_frame=9), frame_id=8, config=_config())
    assert first == second
```

- [ ] Run tests and confirm missing modules/behavior.

Run: `python -m pytest -q tests/oviv2/test_temporal_proposals.py tests/oviv2/test_temporal_identity.py tests/oviv2/test_temporal_association.py`

- [ ] Implement proposal recovery from current-frame segmentation plus causal depth/appearance residuals inside projected identity search regions. Do not use GT category, GT trajectory, future frames, or evaluator thresholds.

- [ ] Implement a bounded lightweight identity bank holding stable ID, semantic/appearance prototype, lifecycle belief, last causal observation, and motion prior separately from geometry epochs.

- [ ] Replace two-pass active-then-dormant assignment with one deterministic global candidate assignment in A4. Apply a wider dormant spatial gate only after semantic/appearance qualification; motion/ICP runs after identity selection.

- [ ] Emit diagnostic opportunity/trigger counts for proposal recovery and re-ID. Zero opportunity must serialize as an eligible-count of zero, not a successful metric.

- [ ] Run focused tests.

Run: `python -m pytest -q tests/oviv2/test_temporal_proposals.py tests/oviv2/test_temporal_identity.py tests/oviv2/test_temporal_association.py`

- [ ] Commit the declared files.

```bash
git add src/oviv2/temporal_proposals.py src/oviv2/temporal_identity.py src/oviv2/temporal_association.py tests/oviv2/test_temporal_proposals.py tests/oviv2/test_temporal_identity.py tests/oviv2/test_temporal_association.py
git commit -m "feat: recover proposals and decouple identity"
```

### Task 7: Integrate The Temporal Core Without Touching Cumulative State

**Files:**
- Modify: `src/oviv2/temporal_state.py`
- Modify: `src/oviv2/temporal_runtime.py`
- Modify: `tests/oviv2/test_temporal_runtime.py`
- Create: `tests/oviv2/test_temporal_state.py`

- [ ] Add failing end-to-end unit tests for the approved state transitions.

```python
def test_visible_absent_hides_geometry_before_identity_becomes_dormant() -> None:
    runtime = _confirmed_runtime(profile="a4")
    result = runtime.process_frame(_visible_absent_frame(), ())
    entity = runtime.state.identities.get(1)
    assert entity.lifecycle is LifecycleState.UNCERTAIN
    assert runtime.state.geometry.current(1).readout_valid is False
    assert result.export.samples == ()
    assert result.export.events[0].readout_valid is False
    assert result.export.events[0].evidence is EvidenceKind.VISIBLE_ABSENT


def test_occlusion_retains_geometry_and_cancels_provisional_background() -> None:
    runtime = _confirmed_runtime(profile="a4")
    runtime.process_frame(_occluded_frame(), ())
    assert runtime.state.geometry.current(1).readout_valid is True
    assert runtime.state.background_ledger.provisional_count == 0


def test_occlusion_does_not_revive_an_invalid_epoch() -> None:
    runtime = _confirmed_runtime(profile="a4")
    runtime.process_frame(_visible_absent_frame(), ())
    runtime.process_frame(_occluded_frame(), ())
    assert runtime.state.geometry.current(1).readout_valid is False
    assert runtime.state.background_ledger.provisional_count == 0
```

- [ ] Run focused tests and confirm failures against the coupled entity/submap model.

Run: `python -m pytest -q tests/oviv2/test_temporal_state.py tests/oviv2/test_temporal_runtime.py`

- [ ] Store identity memory, geometry epochs, lifecycle beliefs, background ledger, export tracker, and diagnostics as separately snapshot-able state. Preserve deterministic canonical dump order.

- [ ] Integrate profiles exactly: A0 reference only; A1 lifecycle overlay with no submap/motion/background calls; A2 proposal + active identity + translation; A3 adds ledger; A4 adds dormant re-ID and gated ICP with explicit rejection.

- [ ] Enforce the motion invariant: `REJECTED` either starts a fresh epoch after high-confidence identity match or creates a new identity; it never calls `integrate_object_submap` on the rejected epoch.

- [ ] Generate one `TemporalExportBatch` after every processed frame, including empty batches. Emit events for lifecycle/readout changes even when an entity has no observation sample on that frame. Add mechanism opportunity/trigger counters to `TemporalFrameResult`.

- [ ] Test deep rollback for failures in proposal, motion, ledger rebuild, and export generation.

Run: `python -m pytest -q tests/oviv2/test_temporal_state.py tests/oviv2/test_temporal_runtime.py tests/oviv2/test_temporal_export.py`

- [ ] Commit the declared files.

```bash
git add src/oviv2/temporal_state.py src/oviv2/temporal_runtime.py tests/oviv2/test_temporal_state.py tests/oviv2/test_temporal_runtime.py
git commit -m "feat: integrate decoupled temporal state"
```

### Task 8: Export Only Valid Geometry Epochs And Committed Background

**Files:**
- Create: `src/oviv2/t1_exactness.py`
- Modify: `src/oviv2/temporal_snapshot.py`
- Modify: `src/oviv2/dual_readout.py`
- Modify: `src/evaluation/oviv2_temporal_tesse.py`
- Create: `tests/oviv2/test_t1_exactness.py`
- Modify: `tests/oviv2/test_temporal_snapshot.py`
- Modify: `tests/oviv2/test_dual_readout.py`
- Modify: `tests/oviv2/test_t1_noninterference.py`
- Modify: `tests/evaluation/test_oviv2_temporal_tesse.py`

- [ ] Add failing tests that invalid epochs never appear in current snapshots, occluded valid epochs remain, committed background is included once, and A0 direct/dual artifacts remain exact.

```python
def test_snapshot_excludes_invalid_geometry_epoch() -> None:
    snapshot = build_temporal_snapshot(_state_with_epochs(valid=(1,), invalid=(2,)))
    assert [entity.geometry_epoch for entity in snapshot.entities] == [1]


def test_temporal_failure_restores_nested_state_and_inputs() -> None:
    runtime, frame, observations = _runtime_with_injected_ledger_failure()
    before = deep_fingerprint(runtime, frame, observations)
    with pytest.raises(RuntimeError):
        runtime.process_frame(frame, observations)
    assert deep_fingerprint(runtime, frame, observations) == before
```

- [ ] Run focused tests and confirm the old lifecycle-only filter fails the epoch test.

Run: `python -m pytest -q tests/oviv2/test_temporal_snapshot.py tests/oviv2/test_dual_readout.py tests/oviv2/test_t1_noninterference.py`

- [ ] Update snapshot serialization and the TESSE temporal adapter with `geometry_epoch` and `readout_valid`; reject duplicate current epochs. Serialize only committed ledger background.

- [ ] In `t1_exactness.py`, implement `shared_input_sha256`, `cumulative_state_sha256`, `temporal_state_sha256`, and an explicit `DualTransactionSnapshot`. Replace shallow `__dict__` rollback with deep transaction snapshots for both readouts. Hash RGB, depth, pose, intrinsics, observations, and dense semantics before and after each branch; make array inputs read-only during calls.

- [ ] Extend `DualFrameResult` with the causal export batch while keeping cumulative result and cumulative checkpoint serializers unchanged.

- [ ] Run focused tests twice to expose order dependence.

Run: `python -m pytest -q tests/oviv2/test_t1_exactness.py tests/oviv2/test_temporal_snapshot.py tests/oviv2/test_dual_readout.py tests/oviv2/test_t1_noninterference.py tests/evaluation/test_oviv2_temporal_tesse.py && python -m pytest -q tests/oviv2/test_dual_readout.py tests/oviv2/test_t1_noninterference.py`

- [ ] Commit the declared files.

```bash
git add src/oviv2/t1_exactness.py src/oviv2/temporal_snapshot.py src/oviv2/dual_readout.py src/evaluation/oviv2_temporal_tesse.py tests/oviv2/test_t1_exactness.py tests/oviv2/test_temporal_snapshot.py tests/oviv2/test_dual_readout.py tests/oviv2/test_t1_noninterference.py tests/evaluation/test_oviv2_temporal_tesse.py
git commit -m "fix: isolate transactional temporal snapshots"
```

### Task 9: Record Every Processed Frame In The TESSE Runner

**Files:**
- Modify: `scripts/evaluation/run_oviv2_tesse_cd_v2.py`
- Modify: `tests/evaluation/test_run_oviv2_tesse_cd_v2.py`

- [ ] Add failing tests proving a three-frame run emits three frame records even with one checkpoint, includes explicit static/dynamic/unknown state, and records transitions at their true frame timestamp.

```python
def test_runner_writes_frame_coverage_for_every_frame(tmp_path: Path) -> None:
    result = _run_fixture(tmp_path, frame_count=3, checkpoint_frames=(2,))
    rows = _read_jsonl(result / "temporal_frame_coverage.jsonl")
    assert sorted({row["frame_index"] for row in rows}) == [0, 1, 2]
    assert all("record_count" in row for row in rows)
```

- [ ] Run the runner test and confirm only checkpoint frames are currently present.

Run: `python -m pytest -q tests/evaluation/test_run_oviv2_tesse_cd_v2.py`

- [ ] Append `DualFrameResult.export.samples` immediately after each successful `process_frame`, outside the checkpoint branch, and write `DualFrameResult.export.events` to `lifecycle_transitions.jsonl`. Write `temporal_frame_coverage.jsonl` with one row per input frame, including zero-sample frames. Do not create fake entity rows for empty frames.

- [ ] Hash-bind both sidecars in `capture_status.json`, `source_index.json`, and the artifact inventory. Add manifest fields `processed_frame_count`, `covered_frame_count`, `trajectory_frame_count`, `first_frame_index`, `last_frame_index`, and export-schema version. Fail unless coverage is exact and timestamps are strictly causal.

- [ ] Generate cumulative audit artifacts for A0-A4 at every checkpoint; keep them outside the temporal result selected for T2.

- [ ] Run the complete runner tests.

Run: `python -m pytest -q tests/evaluation/test_run_oviv2_tesse_cd_v2.py`

- [ ] Commit only the runner and its test after preserving any pre-existing diff.

```bash
git add scripts/evaluation/run_oviv2_tesse_cd_v2.py tests/evaluation/test_run_oviv2_tesse_cd_v2.py
git commit -m "fix: export causal records on every frame"
```

### Task 10: Remove Sparse-Trajectory Inference From Export And Khronos Bridge

**Files:**
- Modify: `scripts/evaluation/export_tesse_temporal_artifact.py`
- Modify: `scripts/evaluation/prepare_temporal_khronos_bridge.py`
- Modify: `scripts/evaluation/compat/khronos_temporal_bridge/import_temporal_baseline.cpp`
- Modify: `tests/evaluation/test_export_tesse_temporal_artifact.py`
- Modify: `tests/evaluation/test_prepare_temporal_khronos_bridge.py`
- Modify: `tests/evaluation/test_run_temporal_khronos_bridge.py`

- [ ] First inspect the current uncommitted diff in all four files and preserve its behavior with tests before editing.

Run: `git diff -- scripts/evaluation/export_tesse_temporal_artifact.py scripts/evaluation/prepare_temporal_khronos_bridge.py tests/evaluation/test_export_tesse_temporal_artifact.py tests/evaluation/test_prepare_temporal_khronos_bridge.py`

- [ ] Add failing tests that reject missing frame coverage, reject implicit dynamic inference, and derive presence intervals only from explicit lifecycle/readout transitions.

```python
def test_bridge_never_promotes_track_from_sample_count(tmp_path: Path) -> None:
    rows = [_trajectory_row(frame=i, entity_id=1, dynamic_state="static") for i in range(3)]
    artifact = _prepare(tmp_path, rows)
    assert artifact["assignments"][0]["dynamic_track_eligible"] is False


def test_visible_absent_closes_presence_at_exact_frame(tmp_path: Path) -> None:
    rows = [
        _trajectory_row(frame=0, transition="new_to_present", readout_valid=True),
        _trajectory_row(frame=1, transition="present_to_uncertain", readout_valid=False),
    ]
    assert _intervals(tmp_path, rows) == [{"start_ns": 0, "end_ns_exclusive": 1}]
```

- [ ] Run focused tests and confirm the current `len(prefix) >= 2` and checkpoint-gap logic fail.

Run: `python -m pytest -q tests/evaluation/test_export_tesse_temporal_artifact.py tests/evaluation/test_prepare_temporal_khronos_bridge.py`

- [ ] Make exporter normalization require the exact temporal export schema, lifecycle-transition sidecar, and complete frame-coverage sidecar. Keep current normalization/security/atomic-write guarantees already present in the dirty worktree.

- [ ] Delete `_presence_intervals`/`_presence_runs` use for prediction inference. Build intervals from explicit events and `readout_valid`. At each query, dynamic eligibility is exactly `prefix[-1].dynamic_state == "dynamic"`, where `prefix` is the entity's native samples at or before the query; confidence is already consumed by the frozen dynamic-state machine and is not thresholded again. Never interpolate missing tracks or use GT semantics.

- [ ] Add a dynamic-to-static regression: a query before the static-off transition exports the causal dynamic prefix, while a later query whose latest sample is static exports no dynamic trajectory.

- [ ] Keep the C++ importer as a strict consumer of bridge-provided `dynamic_track_eligible`; remove any sample-count fallback and reject Python/C++ eligibility disagreement.

- [ ] Add audit counts: static, dynamic, unknown, missing-frame, lifecycle transition, geometry epoch, and invalid-readout samples. Reject a nonzero missing-frame count.

- [ ] Run focused tests and `git diff --check`.

Run: `python -m pytest -q tests/evaluation/test_export_tesse_temporal_artifact.py tests/evaluation/test_prepare_temporal_khronos_bridge.py tests/evaluation/test_run_temporal_khronos_bridge.py && git diff --check`

- [ ] Commit the declared files only.

```bash
git add scripts/evaluation/export_tesse_temporal_artifact.py scripts/evaluation/prepare_temporal_khronos_bridge.py scripts/evaluation/compat/khronos_temporal_bridge/import_temporal_baseline.cpp tests/evaluation/test_export_tesse_temporal_artifact.py tests/evaluation/test_prepare_temporal_khronos_bridge.py tests/evaluation/test_run_temporal_khronos_bridge.py
git commit -m "fix: use explicit causal temporal tracks"
```

### Task 11: Enforce Real T1 Exactness And Cross-Profile Noninterference

**Files:**
- Modify: `scripts/evaluation/verify_oviv2_dual_readout_development_gates.py`
- Modify: `tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py`
- Modify: `scripts/evaluation/package_oviv2_tesse_dual_readout_result.py`
- Modify: `tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py`
- Create: `scripts/evaluation/run_oviv2_t1_reference.py`
- Create: `scripts/evaluation/compare_oviv2_cumulative_artifacts.py`
- Create: `tests/evaluation/test_compare_oviv2_cumulative_artifacts.py`

- [ ] Preserve the existing dirty packager changes by reading their diff and adding regression coverage before editing.

- [ ] Add failing tests for exact inventory and SHA comparison, missing/extra checkpoints, metadata normalization attempts, A0-A4 disagreement, wrong base commit, wrong pytest argv, and incomplete source allowlists.

```python
def test_comparator_rejects_same_final_hash_with_missing_checkpoint(tmp_path: Path) -> None:
    left, right = _artifact_pair(tmp_path)
    (right / "checkpoints" / "00000007.npz").unlink()
    with pytest.raises(ArtifactMismatch, match="inventory"):
        compare_cumulative_artifacts(left, right)


def test_packager_rejects_a4_cumulative_hash_drift(tmp_path: Path) -> None:
    evidence = _valid_evidence()
    evidence["profiles"]["a4"]["cumulative_root_sha256"] = "f" * 64
    with pytest.raises(PackageError, match="cumulative"):
        package_result(evidence, tmp_path / "result.json")
```

- [ ] Run tests and confirm the current fixture-only gate accepts at least one invalid case.

Run: `python -m pytest -q tests/evaluation/test_compare_oviv2_cumulative_artifacts.py tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py`

- [ ] Implement canonical recursive inventory comparison over every checkpoint, final checkpoint, neutral entity export, ownership array, mesh/point artifact, and manifest. Compare raw bytes and SHA-256; no numeric tolerance.

- [ ] Implement `run_oviv2_t1_reference.py` as a narrow worker that runs the frozen v1 path in a separate PID from the dual run. Add a separate-process v1-versus-A0 gate and interleaved A0-versus-A1-A4 gate. Record exact command argv, PID, commit, source manifest digest, input fingerprints, artifact inventory, and root digest.

- [ ] Make the packager validate an exact schema and exact command/test lists. Reject extra/missing evidence fields, any profile without cumulative audit, or any mismatch with the trusted source manifest.

- [ ] Add temporal-axis mutation tests: every temporal parameter must change `algorithm_hash`, preserve `non_temporal_config_sha256`, and preserve cumulative artifact root SHA.

- [ ] Run all three focused test files.

Run: `python -m pytest -q tests/evaluation/test_compare_oviv2_cumulative_artifacts.py tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py`

- [ ] Commit only the declared files after reviewing the pre-existing packager diff.

```bash
git add scripts/evaluation/run_oviv2_t1_reference.py scripts/evaluation/compare_oviv2_cumulative_artifacts.py scripts/evaluation/verify_oviv2_dual_readout_development_gates.py scripts/evaluation/package_oviv2_tesse_dual_readout_result.py tests/evaluation/test_compare_oviv2_cumulative_artifacts.py tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py
git commit -m "test: enforce exact cumulative noninterference"
```

### Task 12: Make Search, Tuning, And Packaging Fail Closed

**Files:**
- Modify: `configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json`
- Modify: `scripts/evaluation/run_oviv2_tesse_dual_readout_search.py`
- Modify: `scripts/evaluation/tune_oviv2_tesse_dual_readout.py`
- Modify: `scripts/evaluation/package_oviv2_tesse_dual_readout_result.py`
- Modify: `src/evaluation/oviv2_temporal_occlusion.py`
- Modify: `scripts/evaluation/evaluate_oviv2_tesse_temporal_occlusion.py`
- Modify: `scripts/evaluation/freeze_oviv2_tesse_cd_v2.py`
- Modify: `scripts/evaluation/run_oviv2_tesse_cd_v2.py`
- Modify: `tests/evaluation/test_run_oviv2_tesse_dual_readout_search.py`
- Modify: `tests/evaluation/test_tune_oviv2_tesse_dual_readout.py`
- Modify: `tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py`
- Modify: `tests/evaluation/test_oviv2_temporal_occlusion.py`
- Modify: `tests/evaluation/test_evaluate_oviv2_tesse_temporal_occlusion.py`
- Modify: `tests/evaluation/test_freeze_oviv2_tesse_cd_v2.py`
- Modify: `tests/evaluation/test_run_oviv2_tesse_cd_v2.py`

- [ ] Add failing tests requiring canonical profile components, bounded parameter spaces, mechanism opportunity counts, the 80% Apartment anchor-coverage gate, per-metric promotion, T1/T4 gates, and an Office lock token.

```python
def test_tuner_rejects_compensating_weighted_score() -> None:
    candidate = _candidate(dynamic_f1=.90, change_f1=.40, ghost=.30, bg_f5=.20)
    assert promote(candidate, _frozen_baselines()).status == "REJECTED_GHOST"


def test_office_requires_frozen_unlock_manifest(tmp_path: Path) -> None:
    with pytest.raises(SearchError, match="Office is locked"):
        run_search(_office_request(), freeze_manifest=None)


def test_direct_office_runner_cannot_bypass_freeze(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Office requires frozen authorization"):
        run(_office_config(tmp_path), tmp_path / "office")
```

- [ ] Run focused tests and confirm the existing rank-only selection accepts compensating candidates.

Run: `python -m pytest -q tests/evaluation/test_run_oviv2_tesse_dual_readout_search.py tests/evaluation/test_tune_oviv2_tesse_dual_readout.py tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py`

- [ ] Register A0-A4 component maps and narrow pre-registered bounds for proposal confidence, dynamic motion confidence, absence invalidation, identity gates, epoch reset, and ledger commit. Unknown or profile-incompatible parameters fail manifest validation.

- [ ] Register the four permitted micro-ablations as separate diagnostic candidates: A2 without proposal recovery, A3 masking-only without the ledger, A4 without dormant candidates, and A4 translation-only without ICP. They cannot replace or rename the A0-A4 main rows.

- [ ] Implement two explicit tuner phases to avoid a T4/selection cycle. `--phase shortlist` applies protocol validity, T1 exactness, anchor coverage `>= 80%`, object/current-map non-inferiority, and all five T2 gates, then retains at least the best eligible candidate from each of A4, A3, and A2 in that fallback order. Pareto pruning is allowed only within one profile; it cannot remove the cross-profile fallback closure. `--phase final` requires that shortlist plus a source-bound T4 matrix and writes `selection.json` only for the first candidate passing all six T4 bounds. Never optimize Office results.

- [ ] Add a regression where A4 dominates A3 on T2 but fails T4: A3 must remain in the shortlist, receive T4 evidence, and be selectable. Add the analogous A3-ledger-fails to A2 case.

- [ ] Enforce whole-profile fallback: an A4 gate failure falls back to A3; a reversible-ledger failure falls back to A2. Never combine best cells from different profiles into one result row.

- [ ] Add mechanism telemetry gates: A1 absence, A2 proposal/epoch, A3 release/reclaim, A4 re-ID/ICP/rejection each report opportunities and triggers. A zero-opportunity mechanism cannot support a positive claim.

- [ ] Implement `AnchorCoverageGate` in the occlusion evaluator and recompute it from source-backed anchor mappings. `eligible_count == 0` is unavailable; Apartment requires at least `53/66` uniquely mapped anchors before full-stack packaging or search promotion.

- [ ] Pin deterministic GPU lanes: A0/A1 on lane 0, A2 on lane 1, A3/A4 serialized on lane 2. Run short-prefix/anchor gates before full Apartment jobs and continue enforcing the existing RAM limit.

- [ ] Make the freeze manifest include code/config/evaluator/GT/schedule/seed hashes and one immutable `office_seed_<seed>` authorization per pre-registered seed. The current deterministic implementation uses seed list `[0]`; a future manifest that declares stochastic behavior must use exactly `[17,29,43,71,101]`. Add required `--t1-evidence` and `--t4-evidence` inputs to the freeze command and bind their root hashes.

- [ ] In `run_oviv2_tesse_cd_v2.py`, reject every Office config unless the matching freeze manifest, config hash, seed slot, and output root are supplied; reject a second published run for the same slot. Permit an identical-hash infrastructure retry only when staging was never published and no metric artifact exists.

- [ ] Run focused tests.

Run: `python -m pytest -q tests/evaluation/test_oviv2_temporal_occlusion.py tests/evaluation/test_evaluate_oviv2_tesse_temporal_occlusion.py tests/evaluation/test_run_oviv2_tesse_dual_readout_search.py tests/evaluation/test_tune_oviv2_tesse_dual_readout.py tests/evaluation/test_freeze_oviv2_tesse_cd_v2.py tests/evaluation/test_run_oviv2_tesse_cd_v2.py tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py`

- [ ] Commit the declared files after preserving all pre-existing packager edits.

```bash
git add configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json src/evaluation/oviv2_temporal_occlusion.py scripts/evaluation/evaluate_oviv2_tesse_temporal_occlusion.py scripts/evaluation/run_oviv2_tesse_dual_readout_search.py scripts/evaluation/tune_oviv2_tesse_dual_readout.py scripts/evaluation/freeze_oviv2_tesse_cd_v2.py scripts/evaluation/run_oviv2_tesse_cd_v2.py scripts/evaluation/package_oviv2_tesse_dual_readout_result.py tests/evaluation/test_oviv2_temporal_occlusion.py tests/evaluation/test_evaluate_oviv2_tesse_temporal_occlusion.py tests/evaluation/test_run_oviv2_tesse_dual_readout_search.py tests/evaluation/test_tune_oviv2_tesse_dual_readout.py tests/evaluation/test_freeze_oviv2_tesse_cd_v2.py tests/evaluation/test_run_oviv2_tesse_cd_v2.py tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py
git commit -m "feat: gate temporal candidate promotion"
```

### Task 13: Collect And Verify Trusted T4 Evidence

**Files:**
- Create: `configs/evaluation/manifests/oviv2_tesse_t4_v1.json`
- Create: `scripts/evaluation/measure_oviv2_tesse_t4.py`
- Create: `scripts/evaluation/verify_oviv2_tesse_t4_gate.py`
- Modify: `scripts/evaluation/measure_baseline_queries.py`
- Modify: `scripts/evaluation/run_oviv2_tesse_cd_v2.py`
- Create: `tests/evaluation/test_measure_oviv2_tesse_t4.py`
- Create: `tests/evaluation/test_verify_oviv2_tesse_t4_gate.py`
- Modify: `tests/evaluation/test_measure_baseline_queries.py`
- Modify: `tests/evaluation/test_run_oviv2_tesse_cd_v2.py`

- [ ] Add failing tests for PID-bound external collection, full-tokenization OVIV2 queries, exact final-map inventory, source hashes, units, and every upper bound.

```python
def test_measurement_binds_all_metrics_to_one_run(tmp_path: Path) -> None:
    evidence = measure_t4(_fixture(tmp_path))
    assert set(evidence["metrics"]) == {
        "total_runtime_s_per_frame",
        "query_mean_ms",
        "query_p95_ms",
        "peak_gpu_gb",
        "peak_ram_gb",
        "final_map_mb",
    }
    assert evidence["sources"]["run_manifest_sha256"] == _sha256(_run_manifest(tmp_path))


def test_verifier_rejects_metric_without_raw_source(tmp_path: Path) -> None:
    evidence = _valid_t4_evidence(tmp_path)
    evidence["sources"].pop("gpu_samples_sha256")
    with pytest.raises(T4GateError, match="gpu samples"):
        verify_t4_gate(evidence)
```

- [ ] Run the tests and confirm that no trusted OVIV2 T4 collection chain exists.

Run: `python -m pytest -q tests/evaluation/test_measure_oviv2_tesse_t4.py tests/evaluation/test_verify_oviv2_tesse_t4_gate.py tests/evaluation/test_measure_baseline_queries.py tests/evaluation/test_run_oviv2_tesse_cd_v2.py`

- [ ] Freeze `oviv2_tesse_t4_v1.json` with the query vocabulary/hash, ViT-H-14 text-model ID/checkpoint hash, full tokenization/text-encoding/entity-ranking protocol, ten warmups, five measured repeats, 200 ms resource sampling, final-map inventory rules, exact units, and the six upper bounds.

- [ ] Extend `measure_baseline_queries.py` with `baseline="oviv2"`. Load the selected current-map entity embeddings and the frozen ViT-H-14 text encoder, synchronize CUDA around every query, and emit raw latencies, mean, p50, and p95. Do not use precomputed query embeddings.

- [ ] Make `run_oviv2_tesse_cd_v2.py` emit hash-bound monotonic phase events for initialization, every frame, finalization, and artifact publication. These events are diagnostics only and must not enter cumulative state or alter cumulative serialization.

- [ ] Implement `measure_oviv2_tesse_t4.py` as the only collector entry point with required CLI arguments `--protocol`, `--shortlist`, `--gpu`, and `--output`. It verifies every shortlist candidate/config hash and measures each candidate in listed fallback order under `/usr/bin/time -v`, samples child-process-group GPU memory by PID, executes the frozen query protocol on each final current-map artifact, and sums only each candidate's final snapshot/entities/background inventory for map MB. Compute total seconds/frame from raw elapsed seconds and the run manifest's processed-frame count; take peak GPU over mapping and queries.

- [ ] Implement `verify_oviv2_tesse_t4_gate.py` to recompute all six values for every shortlisted candidate from raw time/GPU/query/inventory sources and verify their SHA-256 bindings, shortlist hash, protocol hash, units, finite values, config hash, and run-manifest hash. Write an immutable candidate-keyed T4 matrix; a summary field is never trusted as its own source.

- [ ] Run focused tests and a fake-collector integration test; no real GPU is required for unit tests.

Run: `python -m pytest -q tests/evaluation/test_measure_oviv2_tesse_t4.py tests/evaluation/test_verify_oviv2_tesse_t4_gate.py tests/evaluation/test_measure_baseline_queries.py tests/evaluation/test_run_oviv2_tesse_cd_v2.py`

- [ ] Commit the declared files.

```bash
git add configs/evaluation/manifests/oviv2_tesse_t4_v1.json scripts/evaluation/measure_oviv2_tesse_t4.py scripts/evaluation/verify_oviv2_tesse_t4_gate.py scripts/evaluation/measure_baseline_queries.py scripts/evaluation/run_oviv2_tesse_cd_v2.py tests/evaluation/test_measure_oviv2_tesse_t4.py tests/evaluation/test_verify_oviv2_tesse_t4_gate.py tests/evaluation/test_measure_baseline_queries.py tests/evaluation/test_run_oviv2_tesse_cd_v2.py
git commit -m "feat: collect source-bound OVIV2 T4 evidence"
```

### Task 14: Run Focused, Cross-Module, And Repository Verification

**Files:**
- Modify only test files needed to correct a verified test defect; do not weaken assertions.

- [ ] Run all temporal unit tests with plugin autoload disabled.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/oviv2/test_temporal_config.py tests/oviv2/test_temporal_export.py tests/oviv2/test_temporal_proposals.py tests/oviv2/test_temporal_identity.py tests/oviv2/test_temporal_geometry.py tests/oviv2/test_temporal_epoch.py tests/oviv2/test_temporal_background.py tests/oviv2/test_temporal_background_ledger.py tests/oviv2/test_temporal_state.py tests/oviv2/test_temporal_runtime.py tests/oviv2/test_temporal_snapshot.py tests/oviv2/test_reference_readout.py tests/oviv2/test_dual_readout.py tests/oviv2/test_t1_noninterference.py`

- [ ] Run all evaluation pipeline tests.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/evaluation/test_oviv2_temporal_tesse.py tests/evaluation/test_run_oviv2_tesse_cd_v2.py tests/evaluation/test_export_tesse_temporal_artifact.py tests/evaluation/test_prepare_temporal_khronos_bridge.py tests/evaluation/test_run_temporal_khronos_bridge.py tests/evaluation/test_compare_oviv2_cumulative_artifacts.py tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py tests/evaluation/test_evaluate_oviv2_tesse_temporal_occlusion.py tests/evaluation/test_measure_oviv2_tesse_t4.py tests/evaluation/test_verify_oviv2_tesse_t4_gate.py tests/evaluation/test_measure_baseline_queries.py tests/evaluation/test_run_oviv2_tesse_dual_readout_search.py tests/evaluation/test_tune_oviv2_tesse_dual_readout.py tests/evaluation/test_freeze_oviv2_tesse_cd_v2.py tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py`

- [ ] Run the complete OVIV2/evaluation suite.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/oviv2 tests/evaluation`

- [ ] Run static repository checks and placeholder/privacy scans on changed source and manifests.

Run: `git diff --check && ! git diff -U0 -- . ':(exclude)docs/superpowers/plans/**' | rg -n '^\+.*(TO[D]O|TB[D]|PLACEHOLDER|/home/)'`

- [ ] Verify the complete cumulative transitive source manifest against the frozen commit and current worktree; do not substitute the legacy six-file list.

Run: `python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py --repo . --base-commit 8034e79d9cb853166222610981a7e6893f6cca70 --verify-source-manifest configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json`

- [ ] Use `superpowers:requesting-code-review` for an independent spec and quality review. Fix findings with new failing tests, rerun the affected suites, and commit each fix separately.

### Task 15: Run Apartment Gates And Candidate Search On Three GPUs

**Files:**
- Generated only under an ignored experiment output root outside tracked source.

- [ ] Generate development-gate evidence and stop immediately on a cumulative/source mismatch.

Run: `python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py --repo . --base-commit 8034e79d9cb853166222610981a7e6893f6cca70 --output "$OVIV2_RESULTS_ROOT/t2_v16/development_gates.json"`

- [ ] Run the deterministic short-prefix A0-A4 mechanism gate. Require exact frame coverage, zero future leakage, and nonzero trigger counts where opportunities exist.

- [ ] Run the 66-anchor Apartment diagnostic and require at least 53 uniquely mapped anchors before any full search. Report zero-overlap and ambiguous anchors separately.

- [ ] Launch only independent full candidates in parallel across the three devices; keep CPU/RAM concurrency at the manifest limit.

Run: `CUDA_VISIBLE_DEVICES=0,1,2 python scripts/evaluation/run_oviv2_tesse_dual_readout_search.py --manifest configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json --apartment-config configs/oviv2_tesse_cd_apartment_v2.json --office-config configs/oviv2_tesse_cd_office_v2.json --output "$OVIV2_RESULTS_ROOT/t2_v16/search" --gpu 0 --gpu 1 --gpu 2 --max-parallel 3`

- [ ] Package common-v2, occlusion, official Khronos, query, and performance outputs for every complete Apartment candidate. Partial official output is a hard failure.

- [ ] Generate the immutable Apartment shortlist only. First retain candidates passing object/current-map guards, then require strict improvement in dynamic/change/ghost/BG/recovery; use non-gating cost diagnostics only to order Pareto ties.

Run: `python scripts/evaluation/tune_oviv2_tesse_dual_readout.py --phase shortlist --manifest configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json --results-root "$OVIV2_RESULTS_ROOT/t2_v16/search" --output "$OVIV2_RESULTS_ROOT/t2_v16/shortlist.json"`

- [ ] For surviving stochastic candidates, run the five pre-registered Apartment seeds and paired entity/event-cluster bootstrap. If no candidate passes every hard gate, stop before Office and return to the first failed mechanism task; do not relax metrics.

- [ ] Collect and verify T4 for every shortlisted whole-profile candidate before final selection.

Run: `CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/measure_oviv2_tesse_t4.py --protocol configs/evaluation/manifests/oviv2_tesse_t4_v1.json --shortlist "$OVIV2_RESULTS_ROOT/t2_v16/shortlist.json" --gpu 0 --output "$OVIV2_RESULTS_ROOT/t2_v16/t4"`

Run: `python scripts/evaluation/verify_oviv2_tesse_t4_gate.py --evidence-root "$OVIV2_RESULTS_ROOT/t2_v16/t4" --shortlist "$OVIV2_RESULTS_ROOT/t2_v16/shortlist.json" --output "$OVIV2_RESULTS_ROOT/t2_v16/t4_matrix.json"`

- [ ] Run final selection from the frozen shortlist and T4 matrix. If A4 fails T4, evaluate A3 next; if the ledger profile fails, evaluate A2 next. If no whole profile passes, stop before freeze.

Run: `python scripts/evaluation/tune_oviv2_tesse_dual_readout.py --phase final --manifest configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json --results-root "$OVIV2_RESULTS_ROOT/t2_v16/search" --shortlist "$OVIV2_RESULTS_ROOT/t2_v16/shortlist.json" --t4-matrix "$OVIV2_RESULTS_ROOT/t2_v16/t4_matrix.json" --output "$OVIV2_RESULTS_ROOT/t2_v16/selection.json"`

### Task 16: Freeze T1/T4 Evidence And Run Office Once

**Files:**
- Generated only under the experiment output root and final paper result registry.
- Update after successful evidence only: `docs/paper/benchmark_tables_baselines.md`

- [ ] Run production v1 cumulative in a separate process and compare it byte-for-byte with A0. Compare A0-A4 cumulative audits at all checkpoints and final output.

- [ ] Run full T1 static benchmarks for the selected configuration and require every registered T1 non-inferiority gate. Do not replace exact artifact gates with metric tolerance.

- [ ] Confirm the selected candidate is bound to a `PASS` row in the frozen T4 matrix and meets every upper bound in the success contract, including p95 query latency.

- [ ] Freeze the code/config/evaluator/GT/schedule/seed manifest only after Apartment, T1, and T4 all pass. Verify the freeze manifest twice and require byte-identical output.

Run: `python scripts/evaluation/freeze_oviv2_tesse_cd_v2.py --apartment-config configs/oviv2_tesse_cd_apartment_v2.json --office-config configs/oviv2_tesse_cd_office_v2.json --selection "$OVIV2_RESULTS_ROOT/t2_v16/selection.json" --t1-evidence "$OVIV2_RESULTS_ROOT/t2_v16/development_gates.json" --t4-evidence "$OVIV2_RESULTS_ROOT/t2_v16/t4_matrix.json" --output "$OVIV2_RESULTS_ROOT/t2_v16/freeze.json"`

- [ ] Execute the frozen Office transfer once for each pre-registered seed. A retry is allowed only for an infrastructure failure and only with identical hashes; results cannot change configuration or thresholds.

Run: `CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/run_oviv2_tesse_cd_v2.py --config configs/oviv2_tesse_cd_office_v2.json --output "$OVIV2_RESULTS_ROOT/t2_v16/office_seed_0" --freeze-manifest "$OVIV2_RESULTS_ROOT/t2_v16/freeze.json" --run-slot office_seed_0`

- [ ] Re-run both evaluators twice from frozen artifacts and require byte-identical summaries. Compute paired 95% intervals at entity/trajectory and scene/event cluster levels; label censored recovery events and zero-opportunity re-ID as exploratory or `N/A`.

- [ ] Update `docs/paper/benchmark_tables_baselines.md` only from validated package fields. Include provenance paths/hashes for every new OVIV2 value; leave no hand-entered metric.

- [ ] Run final consistency checks.

Run: `git diff --check && ! rg -n '\{\{[^}]+\}\}|待补充|TODO|TBD' docs/paper/benchmark_tables_baselines.md`

- [ ] Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Report exact passed gates, any `N/A` mechanisms, experiment artifact root, and commit IDs.

## Failure Routing

| First failed gate | Return to | Required evidence before rerun |
| --- | --- | --- |
| Per-frame coverage/dynamic eligibility | Tasks 2, 9, 10 | Three-frame fixture and official bridge audit pass |
| Anchor coverage below 80% | Task 6 | Proposal opportunity, zero-overlap, ambiguity breakdown |
| Ghost/change failure with object quality intact | Tasks 5, 7 | Epoch invalidation and rejected-motion tests pass |
| BG F5/recovery failure | Task 4 | Ledger stage/commit/rebuild event trace |
| Identity/re-ID failure | Task 6 | Eligible return count and global assignment trace |
| T1 mismatch | Tasks 3, 8, 11 | First differing source/artifact byte and input fingerprint |
| T4 over budget | Tasks 4, 6, 7 | Per-module CPU/GPU/RAM/profile trace |
| Office failure after freeze | No retuning | Report negative transfer and narrow the paper claim |

## Completion Definition

The implementation is complete only when all tests pass, T1 artifacts are exact, Apartment promotion and T4 gates pass, the Office one-shot is reported without post-hoc tuning, and every table value is traceable to a frozen result package. Passing unit tests alone is not completion; exceeding selected T2 metrics while regressing T1 or another hard T2 metric is also not completion.
