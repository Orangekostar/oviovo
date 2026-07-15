# OVIOVO v2 Contracts and Shadow Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the immutable domain contracts, explicit module interfaces, and dependency-injected shadow orchestrator for the OVIOVO v2 pipeline without changing the behavior or default entry point of the current pipeline.

**Architecture:** The existing `src.pipelines.main_pipeline.Pipeline` remains the production path. A parallel contract-first path introduces immutable observations, local-track snapshots, visibility evidence, persistent entities, voxel evidence, reversible ownership, lifecycle deltas, and map snapshots; the shadow orchestrator passes typed batches between modules but contains no mapping algorithm. A legacy adapter converts current `Patch3D` outputs into the new observation contract so later plans can replace one layer at a time.

**Tech Stack:** Python 3.13, dataclasses, enums, typing Protocols, NumPy, pytest.

---

## Frozen Architecture Decisions

The implementation in this plan must preserve these decisions:

1. `FrameObservation` is immutable and never receives an entity ID by mutation.
2. `LocalTrack` has a local ID only; promotion and entity binding are separate messages.
3. Visibility produces `PRESENT`, `ABSENT`, `OCCLUDED`, or `UNOBSERVED` before association and fusion.
4. Geometry, voxel evidence, current ownership, entity identity, and lifecycle are separate state domains.
5. Voxel ownership is derived state. Evidence is not deleted when ownership is released.
6. Active association and dormant re-identification are distinct decision kinds.
7. State-changing modules return deltas. Only the future map committer may atomically publish a new revision.
8. `MapSnapshot` is immutable and is the only input accepted by query/evaluation adapters in the completed architecture.
9. The current `SystemState`, `ObjectMap`, `TSDFInstanceVolume`, and `Pipeline` remain untouched during this plan.

## Upstream and Downstream Contract

```text
Frame
  -> ObservationFrontend.observe
  -> ObservationBatch
  -> LocalTemporalModule.update
  -> LocalTrackBatch
  -> VisibilityModule.evaluate(previous MapSnapshot)
  -> VisibilityEvidenceBatch
  -> EntityAssociation.associate
  -> AssociationDecisionBatch
  -> EntityRegistry.resolve
  -> EntityResolutionBatch
  -> VoxelFusion.fuse
  -> FusionDelta
  -> OwnershipManager.update
  -> OwnershipDelta
  -> LifecycleManager.update
  -> LifecycleDelta
  -> MapCommitter.commit
  -> immutable MapSnapshot
```

## Plan Series Boundaries

This is the first independently testable plan in a five-plan migration series:

1. **Contracts and shadow pipeline:** this document.
2. **Local temporal tracking and pre-fusion visibility:** implements `LocalTemporalModule` and `VisibilityModule` behind these contracts.
3. **Entity association and dormant re-identification:** implements active association, dormant candidate retrieval, and `EntityRegistry`.
4. **Voxel evidence and reversible ownership:** implements shared geometry, `VoxelEvidenceStore`, sparse entity submaps, and atomic ownership transitions.
5. **Lifecycle, current/history snapshots, and query adapter:** implements signed-evidence lifecycle, event history, snapshot export, and evaluation compatibility.

Each later plan must keep the contracts introduced here or explicitly version them before implementation.

## File Structure

### New domain files

- `src/domain/arrays.py`: defensive immutable NumPy conversion.
- `src/domain/observations.py`: `ObservationQuality`, `FrameObservation`, and `ObservationBatch`.
- `src/domain/tracking.py`: `LocalTrackState`, `LocalTrack`, and `LocalTrackBatch`.
- `src/domain/visibility.py`: signed visibility kinds and batch records.
- `src/domain/entities.py`: persistent entity and lifecycle event contracts.
- `src/domain/association.py`: association decisions and resolved entity bindings.
- `src/domain/mapping.py`: geometry, voxel-evidence, fusion, ownership-layer, and ownership-delta contracts.
- `src/domain/snapshots.py`: immutable current/history `MapSnapshot`.
- `src/domain/__init__.py`: stable public exports for the new domain.

### New pipeline files

- `src/pipelines/entity_mapping_interfaces.py`: Protocols for every v2 module boundary.
- `src/pipelines/entity_mapping_pipeline.py`: algorithm-free orchestration in the frozen stage order.
- `src/pipelines/legacy_observation_adapter.py`: transitional `Patch3D -> FrameObservation` adapter.

### New tests

- `tests/architecture/test_domain_contracts.py`: immutability, validation, and identity invariants.
- `tests/architecture/test_entity_mapping_pipeline.py`: exact upstream/downstream call ordering.
- `tests/architecture/test_legacy_observation_adapter.py`: legacy conversion without state mutation.

No existing production file is modified until the final export task, which only adds public imports to the new `src/domain/__init__.py`.

### Task 1: Immutable Observation Contracts

**Files:**
- Create: `src/domain/__init__.py`
- Create: `src/domain/arrays.py`
- Create: `src/domain/observations.py`
- Create: `tests/architecture/__init__.py`
- Create: `tests/architecture/test_domain_contracts.py`

- [ ] **Step 1: Write failing observation-contract tests**

Create `tests/architecture/__init__.py` as an empty file and create `tests/architecture/test_domain_contracts.py` with:

```python
from __future__ import annotations

import numpy as np
import pytest

from src.domain.observations import FrameObservation, ObservationBatch, ObservationQuality


def make_observation(observation_id: str = "7:11") -> FrameObservation:
    return FrameObservation(
        observation_id=observation_id,
        frame_id=7,
        timestamp=1.25,
        source_proposal_id=11,
        source_backend="sam2",
        mask=np.array([[False, True], [False, False]], dtype=bool),
        bbox_xyxy=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        points_world=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
        voxel_keys=np.array([[0, 0, 20], [2, 0, 20]], dtype=np.int64),
        detector_label="book",
        detector_confidence=0.8,
        visual_embedding=np.array([1.0, 0.0], dtype=np.float32),
        quality=ObservationQuality(
            mask_confidence=0.9,
            valid_depth_ratio=1.0,
            visible_point_ratio=0.75,
            view_quality=0.8,
        ),
        refinement_key="7:11",
        parent_observation_ids=("7:3",),
    )


def test_frame_observation_defensively_freezes_arrays() -> None:
    source_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    observation = FrameObservation(
        observation_id="0:1",
        frame_id=0,
        timestamp=0.0,
        source_proposal_id=1,
        source_backend="sam2",
        mask=None,
        bbox_xyxy=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        points_world=source_points,
        voxel_keys=np.array([[0, 0, 20]], dtype=np.int64),
        quality=ObservationQuality(),
    )

    source_points[:] = 9.0
    assert np.array_equal(observation.points_world, [[0.0, 0.0, 1.0]])
    assert observation.points_world.flags.writeable is False
    with pytest.raises(ValueError):
        observation.points_world[0, 0] = 2.0


def test_observation_batch_rejects_mixed_frames_and_duplicate_ids() -> None:
    first = make_observation("7:11")
    duplicate = make_observation("7:11")
    with pytest.raises(ValueError, match="unique"):
        ObservationBatch(frame_id=7, observations=(first, duplicate))

    other_frame = make_observation("7:12")
    object.__setattr__(other_frame, "frame_id", 8)
    with pytest.raises(ValueError, match="frame_id"):
        ObservationBatch(frame_id=7, observations=(first, other_frame))


def test_observation_quality_is_bounded() -> None:
    with pytest.raises(ValueError, match="view_quality"):
        ObservationQuality(view_quality=1.1)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.domain'`.

- [ ] **Step 3: Implement immutable array and observation contracts**

Create `src/domain/__init__.py` as an empty module.

Create `src/domain/arrays.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np


def readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got shape={array.shape}")
    if trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(f"expected trailing shape {trailing_shape}, got shape={array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("array must contain only finite values")
    array.setflags(write=False)
    return array
```

Create `src/domain/observations.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.domain.arrays import readonly_array


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class ObservationQuality:
    mask_confidence: float = 0.0
    valid_depth_ratio: float = 0.0
    visible_point_ratio: float = 0.0
    view_quality: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "mask_confidence",
            "valid_depth_ratio",
            "visible_point_ratio",
            "view_quality",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))


@dataclass(frozen=True)
class FrameObservation:
    observation_id: str
    frame_id: int
    timestamp: float
    source_proposal_id: int
    source_backend: str
    mask: np.ndarray | None
    bbox_xyxy: np.ndarray
    points_world: np.ndarray
    voxel_keys: np.ndarray
    detector_label: str = ""
    detector_confidence: float = 0.0
    visual_embedding: np.ndarray | None = None
    quality: ObservationQuality = field(default_factory=ObservationQuality)
    refinement_key: str = ""
    parent_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.observation_id):
            raise ValueError("observation_id must be non-empty")
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if int(self.source_proposal_id) < 0:
            raise ValueError("source_proposal_id must be non-negative")
        if not str(self.source_backend):
            raise ValueError("source_backend must be non-empty")
        mask = None
        if self.mask is not None:
            mask = readonly_array(self.mask, dtype=bool, ndim=2)
        bbox = readonly_array(self.bbox_xyxy, dtype=np.float32, ndim=1)
        if bbox.shape != (4,) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox_xyxy must be an ordered vector with shape (4,)")
        points = readonly_array(
            self.points_world,
            dtype=np.float32,
            ndim=2,
            trailing_shape=(3,),
        )
        voxel_keys = readonly_array(
            self.voxel_keys,
            dtype=np.int64,
            ndim=2,
            trailing_shape=(3,),
        )
        embedding = None
        if self.visual_embedding is not None:
            embedding = readonly_array(self.visual_embedding, dtype=np.float32, ndim=1)
        parent_ids = tuple(str(value) for value in self.parent_observation_ids)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("parent_observation_ids must be unique")

        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "source_proposal_id", int(self.source_proposal_id))
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "voxel_keys", voxel_keys)
        object.__setattr__(self, "detector_label", str(self.detector_label).strip())
        object.__setattr__(
            self,
            "detector_confidence",
            _probability(self.detector_confidence, "detector_confidence"),
        )
        object.__setattr__(self, "visual_embedding", embedding)
        object.__setattr__(self, "refinement_key", str(self.refinement_key))
        object.__setattr__(self, "parent_observation_ids", parent_ids)


@dataclass(frozen=True)
class ObservationBatch:
    frame_id: int
    observations: tuple[FrameObservation, ...] = ()

    def __post_init__(self) -> None:
        frame_id = int(self.frame_id)
        observations = tuple(self.observations)
        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if any(item.frame_id != frame_id for item in observations):
            raise ValueError("all observations must match batch frame_id")
        observation_ids = [item.observation_id for item in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_id values must be unique within a batch")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "observations", observations)
```

- [ ] **Step 4: Run the observation tests**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the observation contracts**

```bash
git add src/domain/__init__.py src/domain/arrays.py src/domain/observations.py tests/architecture/__init__.py tests/architecture/test_domain_contracts.py
git commit -m "feat: define immutable frame observations"
```

### Task 2: Local Track and Visibility Contracts

**Files:**
- Create: `src/domain/tracking.py`
- Create: `src/domain/visibility.py`
- Modify: `tests/architecture/test_domain_contracts.py`

- [ ] **Step 1: Add failing track and visibility tests**

Append to `tests/architecture/test_domain_contracts.py`:

```python
from src.domain.tracking import LocalTrack, LocalTrackBatch, LocalTrackState
from src.domain.visibility import VisibilityEvidence, VisibilityEvidenceBatch, VisibilityKind


def test_local_track_has_no_persistent_entity_id() -> None:
    track = LocalTrack(
        local_track_id="track-1",
        state=LocalTrackState.STABLE,
        observation_ids=("7:11",),
        first_frame_id=7,
        last_frame_id=7,
        first_timestamp=1.25,
        last_timestamp=1.25,
        hit_count=1,
        miss_count=0,
        centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        bbox_min=np.array([-0.1, -0.1, 0.9], dtype=np.float32),
        bbox_max=np.array([0.1, 0.1, 1.1], dtype=np.float32),
    )
    assert not hasattr(track, "entity_id")
    assert LocalTrackBatch(frame_id=7, tracks=(track,)).stable_tracks == (track,)


def test_visibility_batch_keeps_occlusion_separate_from_absence() -> None:
    occluded = VisibilityEvidence(
        entity_id="entity-1",
        frame_id=7,
        kind=VisibilityKind.OCCLUDED,
        projected_count=20,
        valid_depth_count=20,
        present_count=0,
        absent_count=0,
        occluded_count=18,
        unobserved_count=2,
    )
    batch = VisibilityEvidenceBatch(frame_id=7, evidence=(occluded,))
    assert batch.by_entity("entity-1").kind is VisibilityKind.OCCLUDED
    assert batch.by_entity("missing") is None
```

- [ ] **Step 2: Verify the imports fail**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: collection fails because `src.domain.tracking` and `src.domain.visibility` do not exist.

- [ ] **Step 3: Implement the track contract**

Create `src/domain/tracking.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.domain.arrays import readonly_array


class LocalTrackState(str, Enum):
    TENTATIVE = "tentative"
    STABLE = "stable"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class LocalTrack:
    local_track_id: str
    state: LocalTrackState
    observation_ids: tuple[str, ...]
    first_frame_id: int
    last_frame_id: int
    first_timestamp: float
    last_timestamp: float
    hit_count: int
    miss_count: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    semantic_label: str = ""
    semantic_confidence: float = 0.0
    appearance_embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not str(self.local_track_id):
            raise ValueError("local_track_id must be non-empty")
        observation_ids = tuple(str(value) for value in self.observation_ids)
        if not observation_ids or len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_ids must be non-empty and unique")
        first_frame_id = int(self.first_frame_id)
        last_frame_id = int(self.last_frame_id)
        if first_frame_id < 0 or last_frame_id < first_frame_id:
            raise ValueError("track frame interval is invalid")
        first_timestamp = float(self.first_timestamp)
        last_timestamp = float(self.last_timestamp)
        if not np.isfinite([first_timestamp, last_timestamp]).all() or last_timestamp < first_timestamp:
            raise ValueError("track timestamp interval is invalid")
        if int(self.hit_count) < 0 or int(self.miss_count) < 0:
            raise ValueError("track counters must be non-negative")
        semantic_confidence = float(self.semantic_confidence)
        if not 0.0 <= semantic_confidence <= 1.0:
            raise ValueError("semantic_confidence must be in [0, 1]")

        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "first_frame_id", first_frame_id)
        object.__setattr__(self, "last_frame_id", last_frame_id)
        object.__setattr__(self, "first_timestamp", first_timestamp)
        object.__setattr__(self, "last_timestamp", last_timestamp)
        object.__setattr__(self, "hit_count", int(self.hit_count))
        object.__setattr__(self, "miss_count", int(self.miss_count))
        object.__setattr__(self, "centroid", readonly_array(self.centroid, dtype=np.float32, ndim=1))
        object.__setattr__(self, "bbox_min", readonly_array(self.bbox_min, dtype=np.float32, ndim=1))
        object.__setattr__(self, "bbox_max", readonly_array(self.bbox_max, dtype=np.float32, ndim=1))
        if self.centroid.shape != (3,) or self.bbox_min.shape != (3,) or self.bbox_max.shape != (3,):
            raise ValueError("track centroid and bounds must have shape (3,)")
        if np.any(self.bbox_max < self.bbox_min):
            raise ValueError("track bbox_max must be >= bbox_min")
        object.__setattr__(self, "semantic_label", str(self.semantic_label).strip())
        object.__setattr__(self, "semantic_confidence", semantic_confidence)
        if self.appearance_embedding is not None:
            object.__setattr__(
                self,
                "appearance_embedding",
                readonly_array(self.appearance_embedding, dtype=np.float32, ndim=1),
            )


@dataclass(frozen=True)
class LocalTrackBatch:
    frame_id: int
    tracks: tuple[LocalTrack, ...] = ()

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        track_ids = [track.local_track_id for track in tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("local_track_id values must be unique within a batch")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "tracks", tracks)

    @property
    def stable_tracks(self) -> tuple[LocalTrack, ...]:
        return tuple(track for track in self.tracks if track.state is LocalTrackState.STABLE)
```

- [ ] **Step 4: Implement signed visibility contracts**

Create `src/domain/visibility.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VisibilityKind(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    OCCLUDED = "occluded"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class VisibilityEvidence:
    entity_id: str
    frame_id: int
    kind: VisibilityKind
    projected_count: int
    valid_depth_count: int
    present_count: int
    absent_count: int
    occluded_count: int
    unobserved_count: int
    reason: str = ""

    def __post_init__(self) -> None:
        if not str(self.entity_id):
            raise ValueError("entity_id must be non-empty")
        counts = (
            self.projected_count,
            self.valid_depth_count,
            self.present_count,
            self.absent_count,
            self.occluded_count,
            self.unobserved_count,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("visibility counts must be non-negative")
        if int(self.valid_depth_count) > int(self.projected_count):
            raise ValueError("valid_depth_count cannot exceed projected_count")


@dataclass(frozen=True)
class VisibilityEvidenceBatch:
    frame_id: int
    evidence: tuple[VisibilityEvidence, ...] = ()

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        if any(item.frame_id != int(self.frame_id) for item in evidence):
            raise ValueError("all visibility evidence must match batch frame_id")
        entity_ids = [item.entity_id for item in evidence]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("visibility batch must contain at most one record per entity")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "evidence", evidence)

    def by_entity(self, entity_id: str) -> VisibilityEvidence | None:
        return next((item for item in self.evidence if item.entity_id == entity_id), None)
```

- [ ] **Step 5: Run the domain tests**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: `5 passed`.

- [ ] **Step 6: Commit track and visibility contracts**

```bash
git add src/domain/tracking.py src/domain/visibility.py tests/architecture/test_domain_contracts.py
git commit -m "feat: define local track and visibility contracts"
```

### Task 3: Persistent Entity and Lifecycle Contracts

**Files:**
- Create: `src/domain/entities.py`
- Modify: `tests/architecture/test_domain_contracts.py`

- [ ] **Step 1: Add failing lifecycle tests**

Append to `tests/architecture/test_domain_contracts.py`:

```python
from src.domain.entities import (
    EntityLifecycleState,
    LifecycleDelta,
    LifecycleInterval,
    LifecycleTransition,
    PersistentEntity,
)


def test_persistent_entity_keeps_closed_lifecycle_intervals() -> None:
    entity = PersistentEntity(
        entity_id="entity-1",
        state=EntityLifecycleState.DORMANT,
        first_seen=1.0,
        last_seen=4.0,
        semantic_label="book",
        semantic_confidence=0.9,
        semantic_embedding=np.array([1.0, 0.0], dtype=np.float32),
        identity_embedding=np.array([0.0, 1.0], dtype=np.float32),
        geometry_handle="geometry/entity-1",
        ownership_revision=5,
        lifecycle_intervals=(
            LifecycleInterval(EntityLifecycleState.ACTIVE, 1.0, 4.0),
            LifecycleInterval(EntityLifecycleState.DORMANT, 4.0, None),
        ),
    )
    assert entity.lifecycle_intervals[-1].end_timestamp is None
    assert entity.semantic_embedding.flags.writeable is False


def test_lifecycle_delta_contains_commands_not_mutable_entities() -> None:
    transition = LifecycleTransition(
        entity_id="entity-1",
        from_state=EntityLifecycleState.ACTIVE,
        to_state=EntityLifecycleState.DORMANT,
        timestamp=4.0,
        reason="signed_absence",
        release_ownership=True,
    )
    delta = LifecycleDelta(base_revision=3, transitions=(transition,))
    assert delta.transitions == (transition,)
```

- [ ] **Step 2: Verify the entity import fails**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: collection fails because `src.domain.entities` does not exist.

- [ ] **Step 3: Implement persistent entity and lifecycle contracts**

Create `src/domain/entities.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.domain.arrays import readonly_array


class EntityLifecycleState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    REMOVED = "removed"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class LifecycleInterval:
    state: EntityLifecycleState
    start_timestamp: float
    end_timestamp: float | None

    def __post_init__(self) -> None:
        start = float(self.start_timestamp)
        end = None if self.end_timestamp is None else float(self.end_timestamp)
        if not np.isfinite(start) or (end is not None and (not np.isfinite(end) or end < start)):
            raise ValueError("invalid lifecycle interval")
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "end_timestamp", end)


@dataclass(frozen=True)
class PersistentEntity:
    entity_id: str
    state: EntityLifecycleState
    first_seen: float
    last_seen: float
    semantic_label: str
    semantic_confidence: float
    semantic_embedding: np.ndarray | None
    identity_embedding: np.ndarray | None
    geometry_handle: str
    ownership_revision: int
    lifecycle_intervals: tuple[LifecycleInterval, ...]

    def __post_init__(self) -> None:
        if not str(self.entity_id) or not str(self.geometry_handle):
            raise ValueError("entity_id and geometry_handle must be non-empty")
        first_seen = float(self.first_seen)
        last_seen = float(self.last_seen)
        if not np.isfinite([first_seen, last_seen]).all() or last_seen < first_seen:
            raise ValueError("invalid entity observation interval")
        confidence = float(self.semantic_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("semantic_confidence must be in [0, 1]")
        intervals = tuple(self.lifecycle_intervals)
        if not intervals or intervals[-1].state is not self.state:
            raise ValueError("last lifecycle interval must represent current entity state")
        if intervals[-1].end_timestamp is not None:
            raise ValueError("current lifecycle interval must be open")
        object.__setattr__(self, "first_seen", first_seen)
        object.__setattr__(self, "last_seen", last_seen)
        object.__setattr__(self, "semantic_label", str(self.semantic_label).strip())
        object.__setattr__(self, "semantic_confidence", confidence)
        object.__setattr__(self, "ownership_revision", int(self.ownership_revision))
        object.__setattr__(self, "lifecycle_intervals", intervals)
        if self.semantic_embedding is not None:
            object.__setattr__(
                self,
                "semantic_embedding",
                readonly_array(self.semantic_embedding, dtype=np.float32, ndim=1),
            )
        if self.identity_embedding is not None:
            object.__setattr__(
                self,
                "identity_embedding",
                readonly_array(self.identity_embedding, dtype=np.float32, ndim=1),
            )


@dataclass(frozen=True)
class LifecycleTransition:
    entity_id: str
    from_state: EntityLifecycleState
    to_state: EntityLifecycleState
    timestamp: float
    reason: str
    release_ownership: bool = False

    def __post_init__(self) -> None:
        if not self.entity_id or self.from_state is self.to_state or not self.reason:
            raise ValueError("lifecycle transition must change one identified entity for a reason")
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("transition timestamp must be finite")


@dataclass(frozen=True)
class LifecycleDelta:
    base_revision: int
    transitions: tuple[LifecycleTransition, ...] = ()

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        entity_ids = [item.entity_id for item in transitions]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("one lifecycle delta may transition each entity at most once")
        object.__setattr__(self, "base_revision", int(self.base_revision))
        object.__setattr__(self, "transitions", transitions)
```

- [ ] **Step 4: Run the domain tests**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit entity contracts**

```bash
git add src/domain/entities.py tests/architecture/test_domain_contracts.py
git commit -m "feat: define persistent entity lifecycle contracts"
```

### Task 4: Association and Entity Resolution Contracts

**Files:**
- Create: `src/domain/association.py`
- Modify: `tests/architecture/test_domain_contracts.py`

- [ ] **Step 1: Add failing association-boundary tests**

Append to `tests/architecture/test_domain_contracts.py`:

```python
from src.domain.association import (
    AssociationDecision,
    AssociationDecisionBatch,
    AssociationKind,
    EntityBinding,
    EntityResolutionBatch,
)


def test_dormant_reid_is_distinct_from_active_match() -> None:
    match = AssociationDecision("track-1", AssociationKind.ACTIVE_MATCH, "entity-1", 0.8)
    reid = AssociationDecision("track-2", AssociationKind.DORMANT_REID, "entity-2", 0.9)
    batch = AssociationDecisionBatch(frame_id=7, decisions=(match, reid))
    assert batch.decisions[0].kind is AssociationKind.ACTIVE_MATCH
    assert batch.decisions[1].kind is AssociationKind.DORMANT_REID


def test_new_entity_id_appears_only_after_registry_resolution() -> None:
    decision = AssociationDecision("track-3", AssociationKind.CREATE, None, 0.7)
    assert decision.entity_id is None
    binding = EntityBinding("track-3", "entity-3", ("7:11",), AssociationKind.CREATE)
    resolution = EntityResolutionBatch(frame_id=7, bindings=(binding,))
    assert resolution.bindings[0].entity_id == "entity-3"
```

- [ ] **Step 2: Verify the association import fails**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: collection fails because `src.domain.association` does not exist.

- [ ] **Step 3: Implement association decisions and resolved bindings**

Create `src/domain/association.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssociationKind(str, Enum):
    ACTIVE_MATCH = "active_match"
    DORMANT_REID = "dormant_reid"
    CREATE = "create"
    DEFER = "defer"
    REJECT = "reject"


@dataclass(frozen=True)
class AssociationDecision:
    local_track_id: str
    kind: AssociationKind
    entity_id: str | None
    score: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.local_track_id:
            raise ValueError("local_track_id must be non-empty")
        score = float(self.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("association score must be in [0, 1]")
        requires_entity = self.kind in {AssociationKind.ACTIVE_MATCH, AssociationKind.DORMANT_REID}
        forbids_entity = self.kind in {AssociationKind.CREATE, AssociationKind.DEFER, AssociationKind.REJECT}
        if requires_entity and not self.entity_id:
            raise ValueError("match and re-id decisions require entity_id")
        if forbids_entity and self.entity_id is not None:
            raise ValueError("create, defer, and reject decisions cannot bind entity_id")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class AssociationDecisionBatch:
    frame_id: int
    decisions: tuple[AssociationDecision, ...] = ()

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        track_ids = [item.local_track_id for item in decisions]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("one association decision is allowed per local track")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "decisions", decisions)


@dataclass(frozen=True)
class EntityBinding:
    local_track_id: str
    entity_id: str
    observation_ids: tuple[str, ...]
    association_kind: AssociationKind

    def __post_init__(self) -> None:
        observation_ids = tuple(str(value) for value in self.observation_ids)
        if not self.local_track_id or not self.entity_id or not observation_ids:
            raise ValueError("entity binding requires track, entity, and observations")
        if self.association_kind in {AssociationKind.DEFER, AssociationKind.REJECT}:
            raise ValueError("deferred or rejected tracks cannot produce entity bindings")
        object.__setattr__(self, "observation_ids", observation_ids)


@dataclass(frozen=True)
class EntityResolutionBatch:
    frame_id: int
    bindings: tuple[EntityBinding, ...] = ()

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        track_ids = [item.local_track_id for item in bindings]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("one resolved binding is allowed per local track")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "bindings", bindings)
```

- [ ] **Step 4: Run the domain tests**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: `9 passed`.

- [ ] **Step 5: Commit association contracts**

```bash
git add src/domain/association.py tests/architecture/test_domain_contracts.py
git commit -m "feat: separate association decisions from entity bindings"
```

### Task 5: Voxel Evidence, Ownership, and Snapshot Contracts

**Files:**
- Create: `src/domain/mapping.py`
- Create: `src/domain/snapshots.py`
- Modify: `tests/architecture/test_domain_contracts.py`

- [ ] **Step 1: Add failing evidence and ownership tests**

Append to `tests/architecture/test_domain_contracts.py`:

```python
from src.domain.mapping import (
    EntityVoxelEvidence,
    FusionDelta,
    GeometryDelta,
    OwnershipDelta,
    OwnershipLayer,
    OwnershipTransition,
    VoxelEvidence,
    VoxelEvidenceDelta,
    VoxelEvidenceUpdate,
    VoxelOwnership,
)
from src.domain.snapshots import MapSnapshot, SnapshotScope


def test_voxel_evidence_retains_competing_entities_independently_of_owner() -> None:
    evidence = VoxelEvidence(
        voxel_key=(1, 2, 3),
        entity_evidence=(
            EntityVoxelEvidence("entity-1", 3.0, 0.0, 1.0),
            EntityVoxelEvidence("entity-2", 1.0, 0.5, 1.0),
        ),
        background_support=0.25,
        source_observation_ids=("7:11",),
        revision=4,
    )
    owner = VoxelOwnership((1, 2, 3), "entity-1", 0.75, epoch=2, evidence_revision=4)
    assert len(evidence.entity_evidence) == 2
    assert owner.owner_entity_id == "entity-1"


def test_ownership_release_does_not_delete_voxel_evidence() -> None:
    transition = OwnershipTransition(
        voxel_key=(1, 2, 3),
        previous_owner_entity_id="entity-1",
        new_owner_entity_id=None,
        confidence=0.0,
        evidence_revision=4,
        reason="entity_dormant",
    )
    delta = OwnershipDelta(base_revision=2, transitions=(transition,))
    assert delta.transitions[0].new_owner_entity_id is None


def test_map_snapshot_is_versioned_and_immutable() -> None:
    ownership = OwnershipLayer(revision=3, assignments=())
    snapshot = MapSnapshot(
        scene_id="room0",
        timestamp=2.0,
        revision=8,
        scope=SnapshotScope.CURRENT,
        entities=(),
        geometry_revision=5,
        voxel_evidence_revision=6,
        ownership=ownership,
        lifecycle_revision=7,
    )
    assert snapshot.schema_version == 2
    assert snapshot.ownership.revision == 3
```

- [ ] **Step 2: Verify the mapping imports fail**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: collection fails because mapping and snapshot modules do not exist.

- [ ] **Step 3: Implement geometry, voxel evidence, and ownership deltas**

Create `src/domain/mapping.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


VoxelKey = tuple[int, int, int]


def _voxel_key(value: VoxelKey) -> VoxelKey:
    if len(value) != 3:
        raise ValueError("voxel key must contain three integers")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class EntityVoxelEvidence:
    entity_id: str
    positive_support: float
    negative_support: float
    last_timestamp: float

    def __post_init__(self) -> None:
        if not self.entity_id or self.positive_support < 0.0 or self.negative_support < 0.0:
            raise ValueError("entity voxel evidence is invalid")


@dataclass(frozen=True)
class VoxelEvidence:
    voxel_key: VoxelKey
    entity_evidence: tuple[EntityVoxelEvidence, ...]
    background_support: float
    source_observation_ids: tuple[str, ...]
    revision: int

    def __post_init__(self) -> None:
        evidence = tuple(self.entity_evidence)
        entity_ids = [item.entity_id for item in evidence]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("voxel evidence may contain one record per entity")
        if float(self.background_support) < 0.0:
            raise ValueError("background_support must be non-negative")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))
        object.__setattr__(self, "entity_evidence", evidence)
        object.__setattr__(self, "source_observation_ids", tuple(self.source_observation_ids))
        object.__setattr__(self, "revision", int(self.revision))


@dataclass(frozen=True)
class VoxelEvidenceUpdate:
    voxel_key: VoxelKey
    entity_id: str
    positive_delta: float
    negative_delta: float
    background_delta: float
    observation_id: str
    timestamp: float

    def __post_init__(self) -> None:
        if not self.entity_id or not self.observation_id:
            raise ValueError("voxel evidence update requires entity and observation")
        if min(self.positive_delta, self.negative_delta, self.background_delta) < 0.0:
            raise ValueError("voxel evidence deltas must be non-negative")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))


@dataclass(frozen=True)
class GeometryDelta:
    base_revision: int
    touched_voxel_keys: tuple[VoxelKey, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "touched_voxel_keys", tuple(_voxel_key(key) for key in self.touched_voxel_keys))


@dataclass(frozen=True)
class VoxelEvidenceDelta:
    base_revision: int
    updates: tuple[VoxelEvidenceUpdate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "updates", tuple(self.updates))


@dataclass(frozen=True)
class FusionDelta:
    geometry: GeometryDelta
    voxel_evidence: VoxelEvidenceDelta


@dataclass(frozen=True)
class VoxelOwnership:
    voxel_key: VoxelKey
    owner_entity_id: str | None
    confidence: float
    epoch: int
    evidence_revision: int

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("ownership confidence must be in [0, 1]")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))


@dataclass(frozen=True)
class OwnershipLayer:
    revision: int
    assignments: tuple[VoxelOwnership, ...]

    def __post_init__(self) -> None:
        assignments = tuple(self.assignments)
        keys = [item.voxel_key for item in assignments]
        if len(keys) != len(set(keys)):
            raise ValueError("ownership layer must contain one assignment per voxel")
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "assignments", assignments)

    def owner_of(self, voxel_key: VoxelKey) -> str | None:
        key = _voxel_key(voxel_key)
        item = next((value for value in self.assignments if value.voxel_key == key), None)
        return None if item is None else item.owner_entity_id


@dataclass(frozen=True)
class OwnershipTransition:
    voxel_key: VoxelKey
    previous_owner_entity_id: str | None
    new_owner_entity_id: str | None
    confidence: float
    evidence_revision: int
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("ownership transition requires a reason and bounded confidence")
        object.__setattr__(self, "voxel_key", _voxel_key(self.voxel_key))


@dataclass(frozen=True)
class OwnershipDelta:
    base_revision: int
    transitions: tuple[OwnershipTransition, ...] = ()

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        keys = [item.voxel_key for item in transitions]
        if len(keys) != len(set(keys)):
            raise ValueError("one ownership transition is allowed per voxel in a delta")
        object.__setattr__(self, "base_revision", int(self.base_revision))
        object.__setattr__(self, "transitions", transitions)
```

- [ ] **Step 4: Implement the immutable map snapshot**

Create `src/domain/snapshots.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.domain.entities import PersistentEntity
from src.domain.mapping import OwnershipLayer


class SnapshotScope(str, Enum):
    CURRENT = "current"
    HISTORY = "history"


@dataclass(frozen=True)
class MapSnapshot:
    scene_id: str
    timestamp: float
    revision: int
    scope: SnapshotScope
    entities: tuple[PersistentEntity, ...]
    geometry_revision: int
    voxel_evidence_revision: int
    ownership: OwnershipLayer
    lifecycle_revision: int
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("snapshot timestamp must be finite")
        entities = tuple(self.entities)
        entity_ids = [entity.entity_id for entity in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique within a snapshot")
        revisions = (
            self.revision,
            self.geometry_revision,
            self.voxel_evidence_revision,
            self.lifecycle_revision,
        )
        if any(int(value) < 0 for value in revisions):
            raise ValueError("snapshot revisions must be non-negative")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "entities", entities)
```

- [ ] **Step 5: Run the domain tests**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py -q
```

Expected: `12 passed`.

- [ ] **Step 6: Commit mapping and snapshot contracts**

```bash
git add src/domain/mapping.py src/domain/snapshots.py tests/architecture/test_domain_contracts.py
git commit -m "feat: define voxel evidence ownership and snapshot contracts"
```

### Task 6: Explicit Module Protocols

**Files:**
- Create: `src/pipelines/entity_mapping_interfaces.py`
- Create: `tests/architecture/test_entity_mapping_pipeline.py`

- [ ] **Step 1: Write a failing protocol import test**

Create `tests/architecture/test_entity_mapping_pipeline.py`:

```python
from __future__ import annotations

from src.pipelines.entity_mapping_interfaces import (
    EntityAssociation,
    EntityRegistry,
    LifecycleManager,
    LocalTemporalModule,
    MapCommitter,
    MapQuery,
    ObservationFrontend,
    OwnershipManager,
    VisibilityModule,
    VoxelFusion,
)


def test_v2_module_protocols_are_importable() -> None:
    protocols = (
        ObservationFrontend,
        LocalTemporalModule,
        VisibilityModule,
        EntityAssociation,
        EntityRegistry,
        VoxelFusion,
        OwnershipManager,
        LifecycleManager,
        MapCommitter,
        MapQuery,
    )
    assert all(getattr(protocol, "_is_protocol", False) for protocol in protocols)
```

- [ ] **Step 2: Verify the protocol module is missing**

Run:

```bash
python -m pytest tests/architecture/test_entity_mapping_pipeline.py -q
```

Expected: collection fails because `entity_mapping_interfaces` does not exist.

- [ ] **Step 3: Implement all module interfaces with exact typed boundaries**

Create `src/pipelines/entity_mapping_interfaces.py`:

```python
from __future__ import annotations

from typing import Protocol

from src.core.data_structures import Frame
from src.domain.association import AssociationDecisionBatch, EntityResolutionBatch
from src.domain.entities import LifecycleDelta
from src.domain.mapping import FusionDelta, OwnershipDelta
from src.domain.observations import ObservationBatch
from src.domain.snapshots import MapSnapshot
from src.domain.tracking import LocalTrackBatch
from src.domain.visibility import VisibilityEvidenceBatch


class ObservationFrontend(Protocol):
    def observe(self, frame: Frame) -> ObservationBatch: ...


class LocalTemporalModule(Protocol):
    def update(self, observations: ObservationBatch) -> LocalTrackBatch: ...


class VisibilityModule(Protocol):
    def evaluate(
        self,
        frame: Frame,
        tracks: LocalTrackBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> VisibilityEvidenceBatch: ...


class EntityAssociation(Protocol):
    def associate(
        self,
        tracks: LocalTrackBatch,
        visibility: VisibilityEvidenceBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> AssociationDecisionBatch: ...


class EntityRegistry(Protocol):
    def resolve(
        self,
        tracks: LocalTrackBatch,
        decisions: AssociationDecisionBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> EntityResolutionBatch: ...


class VoxelFusion(Protocol):
    def fuse(
        self,
        frame: Frame,
        observations: ObservationBatch,
        resolutions: EntityResolutionBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> FusionDelta: ...


class OwnershipManager(Protocol):
    def update(
        self,
        fusion: FusionDelta,
        previous_snapshot: MapSnapshot | None,
    ) -> OwnershipDelta: ...


class LifecycleManager(Protocol):
    def update(
        self,
        visibility: VisibilityEvidenceBatch,
        decisions: AssociationDecisionBatch,
        ownership: OwnershipDelta,
        previous_snapshot: MapSnapshot | None,
    ) -> LifecycleDelta: ...


class MapCommitter(Protocol):
    def commit(
        self,
        frame: Frame,
        resolutions: EntityResolutionBatch,
        fusion: FusionDelta,
        ownership: OwnershipDelta,
        lifecycle: LifecycleDelta,
        previous_snapshot: MapSnapshot | None,
    ) -> MapSnapshot: ...


class MapQuery(Protocol):
    def query(self, snapshot: MapSnapshot, text: str, top_k: int = 5) -> tuple[str, ...]: ...
```

- [ ] **Step 4: Run the protocol test**

Run:

```bash
python -m pytest tests/architecture/test_entity_mapping_pipeline.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit the module interfaces**

```bash
git add src/pipelines/entity_mapping_interfaces.py tests/architecture/test_entity_mapping_pipeline.py
git commit -m "feat: define v2 module boundary protocols"
```

### Task 7: Algorithm-Free Shadow Orchestrator

**Files:**
- Create: `src/pipelines/entity_mapping_pipeline.py`
- Modify: `tests/architecture/test_entity_mapping_pipeline.py`

- [ ] **Step 1: Add a failing exact-order orchestration test**

Append to `tests/architecture/test_entity_mapping_pipeline.py`:

```python
import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.domain.association import AssociationDecisionBatch, EntityResolutionBatch
from src.domain.entities import LifecycleDelta
from src.domain.mapping import (
    FusionDelta,
    GeometryDelta,
    OwnershipDelta,
    OwnershipLayer,
    VoxelEvidenceDelta,
)
from src.domain.observations import ObservationBatch
from src.domain.snapshots import MapSnapshot, SnapshotScope
from src.domain.tracking import LocalTrackBatch
from src.domain.visibility import VisibilityEvidenceBatch
from src.pipelines.entity_mapping_pipeline import EntityMappingModules, EntityMappingPipeline


def make_frame(frame_id: int = 0) -> Frame:
    return Frame(
        frame_id=frame_id,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 2, 2),
        timestamp=float(frame_id),
    )


def test_entity_mapping_pipeline_calls_modules_in_frozen_order() -> None:
    calls: list[str] = []
    observations = ObservationBatch(frame_id=0)
    tracks = LocalTrackBatch(frame_id=0)
    visibility = VisibilityEvidenceBatch(frame_id=0)
    decisions = AssociationDecisionBatch(frame_id=0)
    resolutions = EntityResolutionBatch(frame_id=0)
    fusion = FusionDelta(GeometryDelta(0), VoxelEvidenceDelta(0))
    ownership = OwnershipDelta(base_revision=0)
    lifecycle = LifecycleDelta(base_revision=0)
    snapshot = MapSnapshot(
        scene_id="room0",
        timestamp=0.0,
        revision=1,
        scope=SnapshotScope.CURRENT,
        entities=(),
        geometry_revision=1,
        voxel_evidence_revision=1,
        ownership=OwnershipLayer(1, ()),
        lifecycle_revision=1,
    )

    class Frontend:
        def observe(self, frame):
            calls.append("frontend")
            return observations

    class Temporal:
        def update(self, value):
            assert value is observations
            calls.append("temporal")
            return tracks

    class Visibility:
        def evaluate(self, frame, value, previous):
            assert value is tracks and previous is None
            calls.append("visibility")
            return visibility

    class Association:
        def associate(self, value, evidence, previous):
            assert value is tracks and evidence is visibility and previous is None
            calls.append("association")
            return decisions

    class Registry:
        def resolve(self, value, decision_batch, previous):
            assert value is tracks and decision_batch is decisions and previous is None
            calls.append("registry")
            return resolutions

    class Fusion:
        def fuse(self, frame, observation_batch, resolution_batch, previous):
            assert observation_batch is observations and resolution_batch is resolutions and previous is None
            calls.append("fusion")
            return fusion

    class Ownership:
        def update(self, fusion_delta, previous):
            assert fusion_delta is fusion and previous is None
            calls.append("ownership")
            return ownership

    class Lifecycle:
        def update(self, evidence, decision_batch, ownership_delta, previous):
            assert evidence is visibility
            assert decision_batch is decisions
            assert ownership_delta is ownership
            assert previous is None
            calls.append("lifecycle")
            return lifecycle

    class Committer:
        def commit(self, frame, resolution_batch, fusion_delta, ownership_delta, lifecycle_delta, previous):
            assert resolution_batch is resolutions
            assert fusion_delta is fusion
            assert ownership_delta is ownership
            assert lifecycle_delta is lifecycle
            assert previous is None
            calls.append("commit")
            return snapshot

    pipeline = EntityMappingPipeline(
        EntityMappingModules(
            frontend=Frontend(),
            temporal=Temporal(),
            visibility=Visibility(),
            association=Association(),
            registry=Registry(),
            fusion=Fusion(),
            ownership=Ownership(),
            lifecycle=Lifecycle(),
            committer=Committer(),
        )
    )
    result = pipeline.process_frame(make_frame())

    assert result is snapshot
    assert pipeline.current_snapshot is snapshot
    assert calls == [
        "frontend",
        "temporal",
        "visibility",
        "association",
        "registry",
        "fusion",
        "ownership",
        "lifecycle",
        "commit",
    ]
```

- [ ] **Step 2: Verify the orchestrator import fails**

Run:

```bash
python -m pytest tests/architecture/test_entity_mapping_pipeline.py -q
```

Expected: collection fails because `entity_mapping_pipeline` does not exist.

- [ ] **Step 3: Implement the dependency-injected orchestrator**

Create `src/pipelines/entity_mapping_pipeline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from src.core.data_structures import Frame
from src.domain.snapshots import MapSnapshot
from src.pipelines.entity_mapping_interfaces import (
    EntityAssociation,
    EntityRegistry,
    LifecycleManager,
    LocalTemporalModule,
    MapCommitter,
    ObservationFrontend,
    OwnershipManager,
    VisibilityModule,
    VoxelFusion,
)


@dataclass(frozen=True)
class EntityMappingModules:
    frontend: ObservationFrontend
    temporal: LocalTemporalModule
    visibility: VisibilityModule
    association: EntityAssociation
    registry: EntityRegistry
    fusion: VoxelFusion
    ownership: OwnershipManager
    lifecycle: LifecycleManager
    committer: MapCommitter


class EntityMappingPipeline:
    def __init__(self, modules: EntityMappingModules) -> None:
        self.modules = modules
        self.current_snapshot: MapSnapshot | None = None

    def process_frame(self, frame: Frame) -> MapSnapshot:
        previous = self.current_snapshot
        observations = self.modules.frontend.observe(frame)
        tracks = self.modules.temporal.update(observations)
        visibility = self.modules.visibility.evaluate(frame, tracks, previous)
        decisions = self.modules.association.associate(tracks, visibility, previous)
        resolutions = self.modules.registry.resolve(tracks, decisions, previous)
        fusion = self.modules.fusion.fuse(frame, observations, resolutions, previous)
        ownership = self.modules.ownership.update(fusion, previous)
        lifecycle = self.modules.lifecycle.update(visibility, decisions, ownership, previous)
        snapshot = self.modules.committer.commit(
            frame,
            resolutions,
            fusion,
            ownership,
            lifecycle,
            previous,
        )
        if previous is not None and snapshot.revision <= previous.revision:
            raise ValueError("map snapshot revision must increase monotonically")
        self.current_snapshot = snapshot
        return snapshot
```

- [ ] **Step 4: Add and run a monotonic revision test**

Append to `tests/architecture/test_entity_mapping_pipeline.py`:

```python
from types import SimpleNamespace

import pytest


def test_entity_mapping_pipeline_rejects_non_monotonic_snapshot_revision() -> None:
    previous = MapSnapshot(
        scene_id="room0",
        timestamp=0.0,
        revision=2,
        scope=SnapshotScope.CURRENT,
        entities=(),
        geometry_revision=1,
        voxel_evidence_revision=1,
        ownership=OwnershipLayer(1, ()),
        lifecycle_revision=1,
    )
    modules = EntityMappingModules(
        frontend=SimpleNamespace(observe=lambda frame: ObservationBatch(frame.frame_id)),
        temporal=SimpleNamespace(update=lambda batch: LocalTrackBatch(batch.frame_id)),
        visibility=SimpleNamespace(
            evaluate=lambda frame, tracks, snapshot: VisibilityEvidenceBatch(frame.frame_id)
        ),
        association=SimpleNamespace(
            associate=lambda tracks, evidence, snapshot: AssociationDecisionBatch(tracks.frame_id)
        ),
        registry=SimpleNamespace(
            resolve=lambda tracks, decisions, snapshot: EntityResolutionBatch(tracks.frame_id)
        ),
        fusion=SimpleNamespace(
            fuse=lambda frame, observations, resolutions, snapshot: FusionDelta(
                GeometryDelta(previous.geometry_revision),
                VoxelEvidenceDelta(previous.voxel_evidence_revision),
            )
        ),
        ownership=SimpleNamespace(
            update=lambda fusion, snapshot: OwnershipDelta(previous.ownership.revision)
        ),
        lifecycle=SimpleNamespace(
            update=lambda evidence, decisions, ownership, snapshot: LifecycleDelta(
                previous.lifecycle_revision
            )
        ),
        committer=SimpleNamespace(commit=lambda *args: previous),
    )
    pipeline = EntityMappingPipeline(modules)
    pipeline.current_snapshot = previous

    with pytest.raises(ValueError, match="increase monotonically"):
        pipeline.process_frame(make_frame(frame_id=1))
```

Run:

```bash
python -m pytest tests/architecture/test_entity_mapping_pipeline.py -q
```

Expected: `3 passed`; the third test executes and rejects a non-increasing committed revision.

- [ ] **Step 5: Commit the shadow orchestrator**

```bash
git add src/pipelines/entity_mapping_pipeline.py tests/architecture/test_entity_mapping_pipeline.py
git commit -m "feat: add contract-first entity mapping orchestrator"
```

### Task 8: Legacy Patch Observation Adapter

**Files:**
- Create: `src/pipelines/legacy_observation_adapter.py`
- Create: `tests/architecture/test_legacy_observation_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/architecture/test_legacy_observation_adapter.py`:

```python
from __future__ import annotations

import numpy as np

from src.core.data_structures import Patch3D, ProposalSoftScores
from src.pipelines.legacy_observation_adapter import LegacyPatchObservationAdapter


def make_patch() -> Patch3D:
    return Patch3D(
        patch_id=4,
        points=np.array([[0.01, 0.01, 1.01], [0.06, 0.01, 1.01]], dtype=np.float32),
        centroid=np.array([0.035, 0.01, 1.01], dtype=np.float32),
        bbox_min=np.array([0.01, 0.01, 1.01], dtype=np.float32),
        bbox_max=np.array([0.06, 0.01, 1.01], dtype=np.float32),
        timestamp=2.0,
        source_frame_id=9,
        soft_scores=ProposalSoftScores(objectness_score=0.8),
        metadata={
            "source_proposal_id": 12,
            "source_bbox_xyxy": np.array([1.0, 2.0, 5.0, 6.0], dtype=np.float32),
            "source_backend_name": "sam2",
            "anchor_class_name": "book",
            "anchor_confidence": 0.75,
            "refinement_key": "9:12",
            "geometric_features": {"depth_valid_ratio": 0.9},
            "source_raw_proposal_ids": [2, 3],
        },
    )


def test_legacy_adapter_converts_without_mutating_patch() -> None:
    patch = make_patch()
    original_metadata = dict(patch.metadata)
    mask = np.array([[False, True], [False, False]], dtype=bool)
    batch = LegacyPatchObservationAdapter(voxel_size=0.05).convert(
        [patch],
        masks_by_patch_id={4: mask},
    )

    observation = batch.observations[0]
    assert batch.frame_id == 9
    assert observation.observation_id == "9:12:4"
    assert observation.detector_label == "book"
    assert observation.mask is not mask
    assert observation.mask.flags.writeable is False
    assert observation.voxel_keys.shape == (2, 3)
    assert patch.metadata == original_metadata


def test_legacy_adapter_allows_missing_mask_during_shadow_migration() -> None:
    observation = LegacyPatchObservationAdapter(voxel_size=0.05).convert([make_patch()]).observations[0]
    assert observation.mask is None
```

- [ ] **Step 2: Verify the adapter import fails**

Run:

```bash
python -m pytest tests/architecture/test_legacy_observation_adapter.py -q
```

Expected: collection fails because `legacy_observation_adapter` does not exist.

- [ ] **Step 3: Implement the side-effect-free adapter**

Create `src/pipelines/legacy_observation_adapter.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.core.data_structures import Patch3D
from src.domain.observations import FrameObservation, ObservationBatch, ObservationQuality


class LegacyPatchObservationAdapter:
    def __init__(self, *, voxel_size: float) -> None:
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")

    def convert(
        self,
        patches: Sequence[Patch3D],
        *,
        masks_by_patch_id: Mapping[int, np.ndarray] | None = None,
    ) -> ObservationBatch:
        patches = tuple(patches)
        if not patches:
            raise ValueError("legacy adapter requires at least one patch to determine frame_id")
        frame_ids = {int(patch.source_frame_id) for patch in patches}
        if len(frame_ids) != 1:
            raise ValueError("legacy patches must belong to one frame")
        frame_id = next(iter(frame_ids))
        masks = masks_by_patch_id or {}
        observations = tuple(self._convert_one(patch, masks.get(int(patch.patch_id))) for patch in patches)
        return ObservationBatch(frame_id=frame_id, observations=observations)

    def _convert_one(self, patch: Patch3D, mask: np.ndarray | None) -> FrameObservation:
        metadata = patch.metadata
        points = np.asarray(patch.points, dtype=np.float32)
        voxel_keys = np.unique(np.floor(points / self.voxel_size).astype(np.int64), axis=0)
        source_proposal_id = int(metadata.get("source_proposal_id", patch.patch_id))
        parent_ids = tuple(
            f"{int(patch.source_frame_id)}:{int(raw_id)}"
            for raw_id in metadata.get("source_raw_proposal_ids", ())
        )
        geometry = dict(metadata.get("geometric_features", {}) or {})
        return FrameObservation(
            observation_id=f"{int(patch.source_frame_id)}:{source_proposal_id}:{int(patch.patch_id)}",
            frame_id=int(patch.source_frame_id),
            timestamp=float(patch.timestamp),
            source_proposal_id=source_proposal_id,
            source_backend=str(metadata.get("source_backend_name", "legacy_patch")),
            mask=mask,
            bbox_xyxy=np.asarray(metadata.get("source_bbox_xyxy", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32),
            points_world=points,
            voxel_keys=voxel_keys,
            detector_label=str(metadata.get("anchor_class_name", "")),
            detector_confidence=float(metadata.get("anchor_confidence", 0.0)),
            quality=ObservationQuality(
                mask_confidence=float(metadata.get("anchor_confidence", 0.0)),
                valid_depth_ratio=float(geometry.get("depth_valid_ratio", 0.0)),
                visible_point_ratio=1.0 if len(points) else 0.0,
                view_quality=float(metadata.get("anchor_view_quality", 0.0)),
            ),
            refinement_key=str(metadata.get("refinement_key", "")),
            parent_observation_ids=parent_ids,
        )
```

- [ ] **Step 4: Run adapter and domain tests**

Run:

```bash
python -m pytest tests/architecture/test_legacy_observation_adapter.py tests/architecture/test_domain_contracts.py -q
```

Expected: `14 passed`.

- [ ] **Step 5: Commit the legacy adapter**

```bash
git add src/pipelines/legacy_observation_adapter.py tests/architecture/test_legacy_observation_adapter.py
git commit -m "feat: adapt legacy patches to frame observations"
```

### Task 9: Public Domain Exports and Regression Gate

**Files:**
- Modify: `src/domain/__init__.py`
- Test: `tests/architecture/test_domain_contracts.py`
- Test: `tests/architecture/test_entity_mapping_pipeline.py`
- Test: `tests/architecture/test_legacy_observation_adapter.py`
- Regression: `tests/test_pipeline.py`
- Regression: `tests/evaluation/test_causal_export.py`

- [ ] **Step 1: Add a failing public-export test**

Append to `tests/architecture/test_domain_contracts.py`:

```python
def test_domain_package_exports_six_core_architecture_objects() -> None:
    from src.domain import (
        FrameObservation,
        LocalTrack,
        MapSnapshot,
        OwnershipLayer,
        PersistentEntity,
        VoxelEvidence,
    )

    assert all(
        value is not None
        for value in (
            FrameObservation,
            LocalTrack,
            PersistentEntity,
            VoxelEvidence,
            OwnershipLayer,
            MapSnapshot,
        )
    )
```

- [ ] **Step 2: Verify public imports fail**

Run:

```bash
python -m pytest tests/architecture/test_domain_contracts.py::test_domain_package_exports_six_core_architecture_objects -q
```

Expected: fails because `src.domain` does not export the six names.

- [ ] **Step 3: Export the stable domain surface**

Replace `src/domain/__init__.py` with:

```python
from src.domain.association import (
    AssociationDecision,
    AssociationDecisionBatch,
    AssociationKind,
    EntityBinding,
    EntityResolutionBatch,
)
from src.domain.entities import (
    EntityLifecycleState,
    LifecycleDelta,
    LifecycleInterval,
    LifecycleTransition,
    PersistentEntity,
)
from src.domain.mapping import (
    EntityVoxelEvidence,
    FusionDelta,
    GeometryDelta,
    OwnershipDelta,
    OwnershipLayer,
    OwnershipTransition,
    VoxelEvidence,
    VoxelEvidenceDelta,
    VoxelEvidenceUpdate,
    VoxelOwnership,
)
from src.domain.observations import FrameObservation, ObservationBatch, ObservationQuality
from src.domain.snapshots import MapSnapshot, SnapshotScope
from src.domain.tracking import LocalTrack, LocalTrackBatch, LocalTrackState
from src.domain.visibility import VisibilityEvidence, VisibilityEvidenceBatch, VisibilityKind

__all__ = [
    "AssociationDecision",
    "AssociationDecisionBatch",
    "AssociationKind",
    "EntityBinding",
    "EntityLifecycleState",
    "EntityResolutionBatch",
    "EntityVoxelEvidence",
    "FrameObservation",
    "FusionDelta",
    "GeometryDelta",
    "LifecycleDelta",
    "LifecycleInterval",
    "LifecycleTransition",
    "LocalTrack",
    "LocalTrackBatch",
    "LocalTrackState",
    "MapSnapshot",
    "ObservationBatch",
    "ObservationQuality",
    "OwnershipDelta",
    "OwnershipLayer",
    "OwnershipTransition",
    "PersistentEntity",
    "SnapshotScope",
    "VisibilityEvidence",
    "VisibilityEvidenceBatch",
    "VisibilityKind",
    "VoxelEvidence",
    "VoxelEvidenceDelta",
    "VoxelEvidenceUpdate",
    "VoxelOwnership",
]
```

- [ ] **Step 4: Run the complete new architecture test suite**

Run:

```bash
python -m pytest tests/architecture -q
```

Expected: all architecture tests pass.

- [ ] **Step 5: Run legacy regression tests**

Run:

```bash
python -m pytest tests/test_pipeline.py tests/evaluation/test_causal_export.py -q
```

Expected: all selected legacy tests pass with no changed assertions, proving that this plan did not change the default pipeline or existing snapshot behavior.

- [ ] **Step 6: Commit public exports**

```bash
git add src/domain/__init__.py tests/architecture/test_domain_contracts.py
git commit -m "feat: publish OVIOVO v2 domain contracts"
```

- [ ] **Step 7: Verify no old production file changed in this plan**

Run:

```bash
git diff --name-only HEAD~9..HEAD -- src | sort
```

Expected output contains only:

```text
src/domain/__init__.py
src/domain/arrays.py
src/domain/association.py
src/domain/entities.py
src/domain/mapping.py
src/domain/observations.py
src/domain/snapshots.py
src/domain/tracking.py
src/domain/visibility.py
src/pipelines/entity_mapping_interfaces.py
src/pipelines/entity_mapping_pipeline.py
src/pipelines/legacy_observation_adapter.py
```

## Completion Gate

This plan is complete only when all conditions hold:

1. `python -m pytest tests/architecture -q` passes.
2. `python -m pytest tests/test_pipeline.py tests/evaluation/test_causal_export.py -q` passes.
3. The old `Pipeline` remains the default runtime path.
4. No new algorithm reads or mutates `SystemState` through the v2 interfaces.
5. The call-order test proves visibility precedes association and fusion.
6. `LocalTrack` contains no persistent `entity_id`.
7. `AssociationKind.DORMANT_REID` is distinct from `ACTIVE_MATCH`.
8. `VoxelEvidence` can retain multiple entity candidates while `OwnershipLayer` exposes one current derived owner.
9. Current user-owned uncommitted changes remain untouched and uncommitted by the targeted commits above.

## Benchmark Impact

This plan must produce no benchmark change because it does not alter the default pipeline, state, mapper, exporter, or configuration. The first numerical changes are allowed only in later plans after shadow-mode parity reports identify which layer caused each difference.
