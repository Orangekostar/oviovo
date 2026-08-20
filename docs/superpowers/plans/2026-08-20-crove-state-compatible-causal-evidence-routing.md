# CROVE State-Compatible Causal Evidence Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement state-compatible causal evidence routing for CROVE's A4 temporal branch so qualified identity evidence can support active motion, weak identity matches cannot contaminate persistent appearance, and current-state exports use current observations without changing T1 cumulative outputs.

**Architecture:** Add one pure routing module that owns the closed evidence-to-state matrix and active motion combination. Keep `TemporalCurrentRuntime` as the transition owner, integrating the router only at active motion, appearance-prototype update, and temporal-center readout. Reuse the existing export tracker for the last observed center, so checkpoint and artifact schemas remain unchanged.

**Tech Stack:** Python 3.12, NumPy, frozen dataclasses/enums, pytest, existing CROVE temporal runtime and TESSE-CD v2 verification gates.

---

## File Map

- Create `src/oviv2/temporal_evidence_router.py`: pure evidence matrix, routed motion decision, identity-prototype qualification, and readout-center selection.
- Create `tests/oviv2/test_temporal_evidence_router.py`: exhaustive unit contract for the pure router.
- Modify `src/oviv2/temporal_runtime.py`: integrate routed evidence into A4 without changing A2/A3 or cumulative-state ownership.
- Modify `tests/oviv2/test_temporal_runtime.py`: behavior-level regressions for active motion, prototype isolation, dual centers, replay, and legacy profiles.
- Do not modify any file listed by `configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json`.
- Do not change `TemporalEntityState`, `TemporalExportTrackerEntry`, checkpoint schemas, result schemas, or configuration schemas.

### Task 1: Add The Pure Evidence Router

**Files:**

- Create: `tests/oviv2/test_temporal_evidence_router.py`
- Create: `src/oviv2/temporal_evidence_router.py`

- [ ] **Step 1: Write the failing admissibility-matrix tests**

Create the test module with the exact closed matrix:

```python
from __future__ import annotations

from dataclasses import replace

import pytest

from src.oviv2.temporal_config import ExecutionProfile, TemporalDynamicConfig
from src.oviv2.temporal_evidence_router import (
    CausalEvidenceType,
    MotionEvidenceSource,
    TemporalStateComponent,
    admissible_components,
    admits_identity_prototype_update,
    route_active_motion_evidence,
    select_temporal_readout_center,
)


EXPECTED = {
    CausalEvidenceType.PRESENT_SUPPORT: {
        TemporalStateComponent.EXISTENCE,
    },
    CausalEvidenceType.VISIBLE_FREE_SPACE: {
        TemporalStateComponent.EXISTENCE,
        TemporalStateComponent.BACKGROUND_OWNERSHIP,
    },
    CausalEvidenceType.OCCLUDED: set(),
    CausalEvidenceType.OUT_OF_VIEW: set(),
    CausalEvidenceType.DEPTH_UNKNOWN: set(),
    CausalEvidenceType.LOCAL_GEOMETRY: {
        TemporalStateComponent.CURRENT_POSE,
        TemporalStateComponent.DYNAMIC_STATE,
        TemporalStateComponent.GEOMETRY_EPOCH,
        TemporalStateComponent.CURRENT_READOUT,
    },
    CausalEvidenceType.QUALIFIED_IDENTITY: {
        TemporalStateComponent.IDENTITY_PROTOTYPE,
        TemporalStateComponent.DYNAMIC_STATE,
    },
    CausalEvidenceType.CURRENT_OBSERVATION_CENTER: {
        TemporalStateComponent.CURRENT_POSE,
        TemporalStateComponent.CURRENT_READOUT,
    },
    CausalEvidenceType.FROZEN_STATIC_FUSION: {
        TemporalStateComponent.CUMULATIVE_MAP,
    },
}


@pytest.mark.parametrize("kind", tuple(CausalEvidenceType))
def test_admissibility_matrix_is_closed(kind: CausalEvidenceType) -> None:
    assert admissible_components(kind) == frozenset(EXPECTED[kind])


def test_admissibility_rejects_non_evidence_values() -> None:
    with pytest.raises(TypeError, match="CausalEvidenceType"):
        admissible_components("occluded")  # type: ignore[arg-type]
```

- [ ] **Step 2: Write failing routed-motion tests**

Append tests that cover all source combinations and thresholds:

```python
def dynamic_config() -> TemporalDynamicConfig:
    return TemporalDynamicConfig(
        minimum_consecutive_motion_frames=2,
        displacement_floor_m=0.15,
        minimum_motion_confidence=0.7,
        static_off_streak_frames=2,
    )


@pytest.mark.parametrize(
    ("geometry", "identity", "similarity", "source", "confidence"),
    [
        (False, False, None, MotionEvidenceSource.NONE, 0.0),
        (True, False, None, MotionEvidenceSource.GEOMETRY, 0.8),
        (False, True, 0.9, MotionEvidenceSource.IDENTITY, 0.9),
        (True, True, 0.9, MotionEvidenceSource.COMBINED, 0.9),
    ],
)
def test_active_motion_routes_only_admissible_sources(
    geometry: bool,
    identity: bool,
    similarity: float | None,
    source: MotionEvidenceSource,
    confidence: float,
) -> None:
    result = route_active_motion_evidence(
        geometry_accepted=geometry,
        geometry_confidence=0.8 if geometry else 0.0,
        identity_qualified=identity,
        appearance_similarity=similarity,
        displacement_m=0.2,
        config=dynamic_config(),
    )
    assert result.source is source
    assert result.confidence == pytest.approx(confidence)
    assert result.accepted is (source is not MotionEvidenceSource.NONE)
    assert result.qualifies_as_motion is (
        source is not MotionEvidenceSource.NONE
    )


def test_active_motion_retains_displacement_and_confidence_thresholds() -> None:
    low_displacement = route_active_motion_evidence(
        geometry_accepted=False,
        geometry_confidence=0.0,
        identity_qualified=True,
        appearance_similarity=0.9,
        displacement_m=0.149,
        config=dynamic_config(),
    )
    low_confidence = route_active_motion_evidence(
        geometry_accepted=False,
        geometry_confidence=0.0,
        identity_qualified=True,
        appearance_similarity=0.69,
        displacement_m=0.2,
        config=dynamic_config(),
    )
    assert low_displacement.accepted is True
    assert low_displacement.qualifies_as_motion is False
    assert low_confidence.accepted is True
    assert low_confidence.qualifies_as_motion is False


@pytest.mark.parametrize(
    "changes",
    [
        {"geometry_accepted": 1},
        {"geometry_confidence": float("nan")},
        {"identity_qualified": 1},
        {"appearance_similarity": float("inf")},
        {"appearance_similarity": 1.01},
        {"displacement_m": -0.01},
    ],
)
def test_active_motion_rejects_malformed_inputs(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "geometry_accepted": True,
        "geometry_confidence": 0.8,
        "identity_qualified": True,
        "appearance_similarity": 0.9,
        "displacement_m": 0.2,
        "config": dynamic_config(),
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        route_active_motion_evidence(**values)  # type: ignore[arg-type]
```

- [ ] **Step 3: Write failing identity and readout tests**

```python
def test_identity_prototype_requires_qualified_finite_appearance() -> None:
    assert admits_identity_prototype_update(
        identity_qualified=True, appearance_similarity=0.8
    )
    assert not admits_identity_prototype_update(
        identity_qualified=False, appearance_similarity=0.8
    )
    assert not admits_identity_prototype_update(
        identity_qualified=False, appearance_similarity=None
    )


@pytest.mark.parametrize("profile", (ExecutionProfile.A2, ExecutionProfile.A3))
def test_legacy_profiles_select_cumulative_center(
    profile: ExecutionProfile,
) -> None:
    assert select_temporal_readout_center(
        execution_profile=profile,
        cumulative_center=(1.0, 2.0, 3.0),
        current_observation_center=(4.0, 5.0, 6.0),
    ) == (1.0, 2.0, 3.0)


def test_a4_selects_current_observation_center() -> None:
    assert select_temporal_readout_center(
        execution_profile=ExecutionProfile.A4,
        cumulative_center=(1.0, 2.0, 3.0),
        current_observation_center=(4.0, 5.0, 6.0),
    ) == (4.0, 5.0, 6.0)
```

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
python -m pytest -q tests/oviv2/test_temporal_evidence_router.py
```

Expected: collection fails because `src.oviv2.temporal_evidence_router` does not exist.

- [ ] **Step 5: Implement the pure router**

Create `src/oviv2/temporal_evidence_router.py` with exact enums and a frozen
result type:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType

from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalDynamicConfig,
)


class TemporalStateComponent(str, Enum):
    IDENTITY_PROTOTYPE = "identity_prototype"
    EXISTENCE = "existence"
    CURRENT_POSE = "current_pose"
    DYNAMIC_STATE = "dynamic_state"
    GEOMETRY_EPOCH = "geometry_epoch"
    BACKGROUND_OWNERSHIP = "background_ownership"
    CURRENT_READOUT = "current_readout"
    CUMULATIVE_MAP = "cumulative_map"


class CausalEvidenceType(str, Enum):
    PRESENT_SUPPORT = "present_support"
    VISIBLE_FREE_SPACE = "visible_free_space"
    OCCLUDED = "occluded"
    OUT_OF_VIEW = "out_of_view"
    DEPTH_UNKNOWN = "depth_unknown"
    LOCAL_GEOMETRY = "local_geometry"
    QUALIFIED_IDENTITY = "qualified_identity"
    CURRENT_OBSERVATION_CENTER = "current_observation_center"
    FROZEN_STATIC_FUSION = "frozen_static_fusion"


class MotionEvidenceSource(str, Enum):
    NONE = "none"
    GEOMETRY = "geometry"
    IDENTITY = "identity"
    COMBINED = "combined"


_ADMISSIBLE_COMPONENTS = MappingProxyType({
    CausalEvidenceType.PRESENT_SUPPORT: frozenset({
        TemporalStateComponent.EXISTENCE,
    }),
    CausalEvidenceType.VISIBLE_FREE_SPACE: frozenset({
        TemporalStateComponent.EXISTENCE,
        TemporalStateComponent.BACKGROUND_OWNERSHIP,
    }),
    CausalEvidenceType.OCCLUDED: frozenset(),
    CausalEvidenceType.OUT_OF_VIEW: frozenset(),
    CausalEvidenceType.DEPTH_UNKNOWN: frozenset(),
    CausalEvidenceType.LOCAL_GEOMETRY: frozenset({
        TemporalStateComponent.CURRENT_POSE,
        TemporalStateComponent.DYNAMIC_STATE,
        TemporalStateComponent.GEOMETRY_EPOCH,
        TemporalStateComponent.CURRENT_READOUT,
    }),
    CausalEvidenceType.QUALIFIED_IDENTITY: frozenset({
        TemporalStateComponent.IDENTITY_PROTOTYPE,
        TemporalStateComponent.DYNAMIC_STATE,
    }),
    CausalEvidenceType.CURRENT_OBSERVATION_CENTER: frozenset({
        TemporalStateComponent.CURRENT_POSE,
        TemporalStateComponent.CURRENT_READOUT,
    }),
    CausalEvidenceType.FROZEN_STATIC_FUSION: frozenset({
        TemporalStateComponent.CUMULATIVE_MAP,
    }),
})


@dataclass(frozen=True)
class RoutedMotionEvidence:
    source: MotionEvidenceSource
    accepted: bool
    displacement_m: float
    confidence: float
    qualifies_as_motion: bool
```

Implement strict helpers that reject booleans as numbers, require finite
confidence/displacement, constrain confidence to `[0, 1]`, constrain appearance
similarity to `[-1, 1]`, and clip a qualified negative cosine similarity to
zero. `route_active_motion_evidence` must select `COMBINED` whenever both
sources are admissible, use the maximum confidence, and compute
`qualifies_as_motion` only from the existing dynamic config thresholds.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/oviv2/test_temporal_evidence_router.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the pure contract**

```bash
git add src/oviv2/temporal_evidence_router.py tests/oviv2/test_temporal_evidence_router.py
git commit -m "feat: add causal temporal evidence router"
```

### Task 2: Route Active Motion Through Geometry And Qualified Identity

**Files:**

- Modify: `tests/oviv2/test_temporal_runtime.py`
- Modify: `src/oviv2/temporal_runtime.py`

- [ ] **Step 1: Add a failing identity-only active-motion regression**

Add a test that confirms an entity, forces geometry rejection, supplies the
same qualified appearance feature at two displaced observations, and asserts:

```python
assert first.export.samples[0].dynamic_state is DynamicState.STATIC
assert second.export.samples[0].dynamic_state is DynamicState.DYNAMIC
assert second.export.samples[0].motion_confidence >= 0.7
```

The fake estimator must return `MotionDecision.REJECTED` with the supplied
`previous_object_to_world`; the observations must use centers `1.2` and `1.4`
meters so each accepted identity displacement exceeds the configured floor.

- [ ] **Step 2: Add failing negative controls**

Use the existing association wrapper pattern from
`test_rejected_low_confidence_match_creates_new_identity` to set
`high_confidence_identity_match=False`. Assert the old identity never receives
identity motion evidence and the rejected assignment follows the existing
forced-new-identity behavior. Add a one-displaced-frame test asserting the
existing consecutive-motion requirement keeps the sample static.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
python -m pytest -q tests/oviv2/test_temporal_runtime.py \
  -k 'active_identity_motion or one_frame_identity_motion or rejected_low_confidence'
```

Expected: the identity-only positive test fails because the current runtime
passes only geometric confidence to `advance_dynamic_state`.

- [ ] **Step 4: Integrate the routed motion result**

In `temporal_runtime.py`:

1. import `route_active_motion_evidence`;
2. create `current_center_by_entity: dict[int, tuple[float, float, float]]` next
   to `motion_by_entity`;
3. record every successfully backprojected assigned/new observation's finite
   `observation.centroid_xyz` in that map;
4. for active assignments, choose the previous endpoint from
   `old_export[entity_id].last_centroid_xyz`, falling back to `_centroid(old)`;
5. compute displacement between current and previous observed endpoints;
6. call the router with geometric acceptance, geometric confidence, assignment
   identity qualification, appearance similarity, displacement, and the
   existing dynamic config;
7. pass `routed.accepted`, `routed.displacement_m`, and `routed.confidence` to
   `advance_dynamic_state`;
8. use `routed.qualifies_as_motion` for the existing epoch-transition branch;
9. record routed displacement/confidence in `motion_by_entity`.

Do not relax association, motion-distance, or epoch-reset gates. Keep the
rejected low-confidence branch unchanged.

- [ ] **Step 5: Prefer the last observed endpoint for dormant re-ID**

In the bank-only re-identification branch, compute displacement against
`previous_export.last_centroid_xyz` when available, otherwise retain
`record.last_centroid_xyz`. Keep immediate dormant reappearance qualification
and thresholds unchanged.

- [ ] **Step 6: Run runtime and router tests**

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_evidence_router.py \
  tests/oviv2/test_temporal_runtime.py
```

Expected: both files pass, including existing rejected-motion, dormant re-ID,
and epoch-transition tests.

- [ ] **Step 7: Commit routed active motion**

```bash
git add src/oviv2/temporal_runtime.py tests/oviv2/test_temporal_runtime.py
git commit -m "feat: route qualified identity motion evidence"
```

### Task 3: Isolate Persistent Appearance Identity

**Files:**

- Modify: `tests/oviv2/test_temporal_runtime.py`
- Modify: `src/oviv2/temporal_runtime.py`

- [ ] **Step 1: Add a failing weak-identity prototype test**

Confirm an entity with prototype `[1, 0]`. Wrap the accepted association result
so the assignment remains accepted but its diagnostic has
`high_confidence_identity_match=False`. Process an observation with feature
`[0, 1]`, then assert both the entity prototype and identity-bank prototype are
exactly unchanged.

- [ ] **Step 2: Add a qualified-identity positive control**

Process a qualified accepted observation with a non-identical finite feature
whose similarity passes the existing gate. Assert the prototype changes and
remains normalized. This proves the fix is selective rather than freezing all
identity adaptation.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
python -m pytest -q tests/oviv2/test_temporal_runtime.py \
  -k 'prototype and identity'
```

Expected: the weak-identity test fails because the active branch currently
calls `_prototype_update` unconditionally.

- [ ] **Step 4: Gate only the persistent appearance update**

Import `admits_identity_prototype_update`. In the active assignment branch:

```python
if admits_identity_prototype_update(
    identity_qualified=diagnostic.high_confidence_identity_match,
    appearance_similarity=diagnostic.appearance_similarity,
):
    prototype, feature_model_id = _prototype_update(
        old.image_prototype, old.feature_model_id, observation
    )
else:
    prototype = old.image_prototype
    feature_model_id = old.feature_model_id
```

Keep semantic-probability, lifecycle, pose, extent, geometry, and new-identity
updates unchanged. Bank-only dormant re-ID already requires qualified identity
evidence and retains its existing update.

- [ ] **Step 5: Run focused and full runtime tests**

```bash
python -m pytest -q tests/oviv2/test_temporal_evidence_router.py
python -m pytest -q tests/oviv2/test_temporal_runtime.py
```

Expected: all pass.

- [ ] **Step 6: Commit prototype isolation**

```bash
git add src/oviv2/temporal_runtime.py tests/oviv2/test_temporal_runtime.py
git commit -m "fix: isolate temporal identity prototypes"
```

### Task 4: Use Current Observation Centers Only For A4 Temporal Readout

**Files:**

- Modify: `tests/oviv2/test_temporal_runtime.py`
- Modify: `src/oviv2/temporal_runtime.py`

- [ ] **Step 1: Add a failing A4 dual-center test**

Construct an A4 frame where retained submap geometry stays at the previous
center while the current observation center changes. Capture before/after
submap bytes and assert:

```python
assert result.export.samples[0].centroid_xyz == pytest.approx(
    current_observation.centroid_xyz
)
assert runtime.state.export_tracker.entries[0].last_centroid_xyz == pytest.approx(
    current_observation.centroid_xyz
)
assert runtime.state.entities[0].submap == previous_submap
```

Also assert `_centroid(runtime.state.entities[0])` differs from the exported
center, proving the test actually exercises dual centers.

- [ ] **Step 2: Add A2/A3 legacy-center controls**

Parameterize A2 and A3 with the same geometry setup and assert their export
sample center remains `_centroid(entity)`. This locks the change to A4.

- [ ] **Step 3: Run the tests and verify RED**

```bash
python -m pytest -q tests/oviv2/test_temporal_runtime.py \
  -k 'dual_center or legacy_center'
```

Expected: the A4 test fails because the current export loop always calls
`_centroid(entity)`.

- [ ] **Step 4: Select the temporal readout center**

Import `select_temporal_readout_center`. In the export loop:

```python
cumulative_center = _centroid(entity)
current_center = current_center_by_entity.get(entity_id)
centroid = (
    select_temporal_readout_center(
        execution_profile=self.config.execution_profile,
        cumulative_center=cumulative_center,
        current_observation_center=current_center,
    )
    if observed
    else cumulative_center
)
```

Require a current center for every observed A4 entity and fail closed if it is
missing. Store this selected center in the observed export tracker entry and
sample. Keep the earlier identity-bank loop on its existing `_centroid(entity)`
value. Do not alter entity transforms, submaps, geometry epochs, snapshots, or
background ownership.

- [ ] **Step 5: Add deterministic prefix-replay coverage**

Run an identical frame prefix through two fresh A4 runtimes and assert equality
of export batches, tracker canonical dumps, entity submaps, geometry states,
and diagnostics. Then append one extra frame to only one runtime and confirm the
prefix artifacts remain unchanged.

- [ ] **Step 6: Run all temporal behavior suites**

```bash
python -m pytest -q \
  tests/oviv2/test_temporal_evidence_router.py \
  tests/oviv2/test_temporal_runtime.py \
  tests/oviv2/test_temporal_export.py \
  tests/oviv2/test_temporal_state.py \
  tests/oviv2/test_temporal_epoch.py \
  tests/oviv2/test_temporal_background_ledger.py
```

Expected: all pass.

- [ ] **Step 7: Commit dual-center readout**

```bash
git add src/oviv2/temporal_runtime.py tests/oviv2/test_temporal_runtime.py
git commit -m "feat: export causal current observation centers"
```

### Task 5: Prove T1 Non-Interference And Package Compatibility

**Files:**

- No production changes expected.
- Modify the implementation only if a focused regression identifies a real
  violation; do not update trusted manifests to hide a source mismatch.

- [ ] **Step 1: Verify the protected source manifest**

```bash
python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py \
  --verify-source-manifest \
  configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json
```

Expected: exit 0 and no protected-source mismatch.

- [ ] **Step 2: Verify exact cumulative behavior**

```bash
python -m pytest -q \
  tests/oviv2/test_t1_exactness.py \
  tests/oviv2/test_t1_noninterference.py \
  tests/oviv2/test_dual_readout.py
```

Expected: all tests pass; `test_t1_noninterference.py` remains 8/8.

- [ ] **Step 3: Verify v2 runner, provenance, and package contracts**

```bash
python -m pytest -q \
  tests/evaluation/test_run_oviv2_tesse_cd_v2.py \
  tests/evaluation/test_compare_oviv2_cumulative_artifacts.py \
  tests/evaluation/test_build_oviv2_tesse_search_preflight.py \
  tests/evaluation/test_package_oviv2_tesse_dual_readout_result.py \
  tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py
```

Expected: all locally asset-independent tests pass; tests requiring declared
external TESSE assets may skip only through their existing path-presence guard.

- [ ] **Step 4: Verify syntax and diff hygiene**

```bash
python -m py_compile \
  src/oviv2/temporal_evidence_router.py \
  src/oviv2/temporal_runtime.py
git diff --check
git status --short
```

Expected: compile and diff checks exit 0; only intended source, test, design,
plan, literature, and review files are changed.

### Task 6: Freeze The Apartment Evidence Gate Before Any Paper Claim

**Files:**

- Do not update paper result tables in this task.
- Outputs: `outputs/tesse_cd/crove-evidence-routing/apartment_run1/`
- Outputs: `outputs/tesse_cd/crove-evidence-routing/apartment_run2/`

- [ ] **Step 1: Record the exact implementation identity**

```bash
git rev-parse HEAD
git status --porcelain=v1
sha256sum configs/oviv2_tesse_cd_apartment_v2.json
```

Expected: the worktree is clean before a formal launch; save the printed commit
and config digest with the run receipts.

- [ ] **Step 2: Run the Apartment causal sequence twice**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/run_oviv2_tesse_cd_v2.py \
  --config configs/oviv2_tesse_cd_apartment_v2.json \
  --output outputs/tesse_cd/crove-evidence-routing/apartment_run1 \
  --run-slot apartment_run1

CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/run_oviv2_tesse_cd_v2.py \
  --config configs/oviv2_tesse_cd_apartment_v2.json \
  --output outputs/tesse_cd/crove-evidence-routing/apartment_run2 \
  --run-slot apartment_run2
```

Expected: both runs finish with source-bound manifests and identical algorithm
identity. The commands may run concurrently only after frontend caches are
complete and each uses an independent output root.

- [ ] **Step 3: Apply the preregistered promotion decision**

Compare the two runs with the checked A6 Apartment artifact using official
Obj./Dyn./Chg. F1, common-v2 current mIoU, ghost rate, background F@5cm,
recovery latency, identity fragmentation, and epoch-reset diagnostics. Promote
only if the exact gates in the approved design pass. Do not substitute visual
quality, a diagnostic proxy, or one favorable checkpoint for the formal result.

- [ ] **Step 4: Stop before Office unless Apartment passes**

Do not run Office, modify T2/T3/T4 tables, or enable superiority language until
the Apartment result is source-bound, repeat-consistent, and frozen. After a
passing Apartment decision, write a separate freeze/held-out execution plan
using the actual implementation commit and artifact hashes; no placeholder hash
is permitted in that plan.

## Final Review Checklist

- [ ] The evidence matrix has one exact test per evidence type.
- [ ] Runtime calls the router only in A4-sensitive temporal paths.
- [ ] A2/A3 exports and motion behavior are unchanged.
- [ ] Identity-bank and cumulative centers remain fused-map centers.
- [ ] A4 observed export/tracker centers are current observation centers.
- [ ] Unqualified appearance cannot modify the persistent prototype.
- [ ] Geometry or identity evidence alone can support active hysteresis, but
      thresholds and consecutive-frame requirements remain authoritative.
- [ ] Occlusion neutrality, visible-absence ownership release, epoch purity,
      dormant re-ID, and rejected low-confidence behavior remain covered.
- [ ] No T1 protected source, schema, or trusted manifest is changed.
- [ ] Paper tables remain untouched until verified formal artifacts exist.
