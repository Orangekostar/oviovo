# CROVE Unbound-Anchor Visibility Implementation Plan

**Goal:** Publish and evaluate one causal visibility-gated OVI-MAP backbone
candidate that targets the measured unbound-anchor Ghost failure.

**Architecture:** Add a pure visibility state machine, expose suppression as an
optional default-preserving composition input, then integrate it into the
existing hash-bound composition runner and temporal artifact chain.

### Task 1: Pure causal visibility policy

**Files:**
- Create: `src/oviv2/anchor_visibility.py`
- Create: `tests/oviv2/test_anchor_visibility.py`
- Create: `configs/evaluation/crove_ovimap_unbound_visibility_v1.json`

- [x] Write RED tests for deterministic voxel sampling, strict absence and
  presence evidence, three-baseline/six-observation dormancy, neutral
  occlusion handling, presence reset, and two-frame reactivation.
- [x] Implement the exact frozen policy and validation.
- [x] Run the new module tests and lint checks.
- [x] Commit the pure policy separately.

### Task 2: Default-preserving composition suppression

**Files:**
- Modify: `src/oviv2/ovimap_static_anchor.py`
- Modify: `tests/oviv2/test_ovimap_static_anchor.py`

- [x] Write RED tests proving only declared unbound anchors are omitted and
  bound/unknown IDs fail closed.
- [x] Add an empty-by-default suppressed-ID input to checkpoint composition.
- [x] Verify the existing default behavior and protected T1 tests.
- [x] Commit the core integration separately.

### Task 3: Runner, provenance, and export integration

**Files:**
- Modify: `scripts/evaluation/run_crove_ovimap_static_anchor.py`
- Modify: `tests/evaluation/test_run_crove_ovimap_static_anchor.py`

- [x] Write RED tests for role/policy pairing, causal frame processing,
  trajectory validity, lifecycle transitions, checkpoint suppression,
  diagnostics binding, and no-output-on-failure behavior.
- [x] Add the `causal_visibility_candidate` role and optional policy input.
- [x] Load the anchor-bound RGB-D dataset, compute the chronological visibility
  timeline, and publish hash-bound runtime diagnostics.
- [x] Export the result through the existing temporal artifact API and verify
  bridge consistency.
- [x] Commit the runner integration separately.

### Task 4: Apartment decision and frozen transfer

- [x] Compose the single Apartment visibility candidate using the better of
  the compact and P4A moved-geometry modes according to official P4A metrics.
- [x] Run temporal export, Khronos bridge import, official evaluation, and the
  common-v2 gate.
- [x] Record the Apartment rejection; no configuration is frozen because the
  Object F1 promotion gate failed.
- [x] Keep Office held out because no Apartment candidate passed every gate.
- [ ] Update the result ledger/report, run final verification, push the branch,
  and verify the remote SHA.
