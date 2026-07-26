from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.oviv2.temporal_config import TemporalIdentityConfig
from src.oviv2.temporal_identity import IdentityMemoryBank, IdentityMemoryRecord
from src.oviv2.temporal_lifecycle import TemporalLifecycle


def config(**changes: object) -> TemporalIdentityConfig:
    values: dict[str, object] = {
        "maximum_identities": 2,
        "maximum_dormant_frames": 3,
        "minimum_reid_similarity": 0.8,
        "maximum_reid_distance_m": 4.0,
    }
    values.update(changes)
    return TemporalIdentityConfig(**values)  # type: ignore[arg-type]


def fields(frame_id: int = 1, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "frame_id": frame_id,
        "timestamp": float(frame_id),
        "semantic_probabilities": ((1, 0.75), (2, 0.25)),
        "appearance_prototype": np.array([3.0, 4.0]),
        "feature_model_id": "clip-v1",
        "lifecycle": TemporalLifecycle.ACTIVE,
        "centroid_xyz": (1.0, 2.0, 3.0),
        "extent_xyz": (0.5, 1.0, 1.5),
        "motion_velocity_xyz": (0.1, 0.0, 0.0),
        "motion_uncertainty_m": 0.2,
    }
    values.update(changes)
    return values


def test_insert_get_owns_normalized_data_and_has_stable_canonical_dump() -> None:
    prototype = np.array([3.0, 4.0])
    bank = IdentityMemoryBank(config())
    record = bank.insert(**fields(appearance_prototype=prototype))
    prototype[:] = 0.0

    assert record.identity_id == 1
    assert record.appearance_prototype == pytest.approx(np.array([0.6, 0.8]))
    assert not record.appearance_prototype.flags.writeable
    assert bank.get(1) == record
    assert bank.canonical_dump() == bank.clone().canonical_dump()
    assert deepcopy(record).appearance_prototype is not record.appearance_prototype
    assert not deepcopy(record).appearance_prototype.flags.writeable
    with pytest.raises(ValueError):
        record.appearance_prototype[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        record.identity_id = 9  # type: ignore[misc]


def test_update_is_strictly_causal_and_preserves_first_observation() -> None:
    bank = IdentityMemoryBank(config())
    first = bank.insert(**fields())
    updated = bank.update(
        first.identity_id,
        **fields(2, timestamp=2.5, centroid_xyz=(2.0, 2.0, 3.0)),
    )
    assert (updated.first_frame_id, updated.first_timestamp) == (1, 1.0)
    assert (updated.last_frame_id, updated.last_timestamp) == (2, 2.5)
    assert updated.first_centroid_xyz == (1.0, 2.0, 3.0)
    before = bank.canonical_dump()
    with pytest.raises(ValueError, match="strictly"):
        bank.update(first.identity_id, **fields(2, timestamp=3.0))
    assert bank.canonical_dump() == before


def test_capacity_rejection_is_atomic_and_ids_are_never_reused() -> None:
    bank = IdentityMemoryBank(config(maximum_identities=1))
    bank.insert(**fields())
    before = bank.canonical_dump()
    with pytest.raises(OverflowError, match="maximum_identities"):
        bank.insert(**fields(2))
    assert bank.canonical_dump() == before

    diagnostics = bank.expire_dormant(current_frame_id=5)
    assert diagnostics.expired_identity_ids == ()
    bank.update(1, **fields(2, lifecycle=TemporalLifecycle.DORMANT))
    diagnostics = bank.expire_dormant(current_frame_id=6)
    assert diagnostics.expired_identity_ids == (1,)
    assert bank.get(1) is None
    replacement = bank.insert(**fields(7))
    assert replacement.identity_id == 2


def test_expiry_rejects_a_frame_before_any_retained_identity() -> None:
    bank = IdentityMemoryBank(config())
    bank.insert(**fields(3))
    before = bank.canonical_dump()
    with pytest.raises(ValueError, match="current_frame_id"):
        bank.expire_dormant(current_frame_id=2)
    assert bank.canonical_dump() == before


def test_expiry_is_deterministic_and_geometry_eviction_is_irrelevant() -> None:
    bank = IdentityMemoryBank(config(maximum_identities=3, maximum_dormant_frames=2))
    first = bank.insert(**fields(lifecycle=TemporalLifecycle.DORMANT))
    second = bank.insert(**fields(2, lifecycle=TemporalLifecycle.DORMANT))
    clone = bank.clone()
    assert bank.expire_dormant(current_frame_id=4).expired_identity_ids == (first.identity_id,)
    assert clone.expire_dormant(current_frame_id=4).expired_identity_ids == (first.identity_id,)
    assert bank.get(second.identity_id) is not None
    geometry_cache: dict[int, object] = {second.identity_id: object()}
    geometry_cache.clear()
    assert bank.get(second.identity_id) is not None


def test_expiry_diagnostics_and_record_causal_pairs_fail_closed() -> None:
    bank = IdentityMemoryBank(config())
    record = bank.insert(**fields())
    with pytest.raises(ValueError, match="strictly"):
        replace(record, last_frame_id=2, last_timestamp=1.0)
    diagnostics = bank.expire_dormant(current_frame_id=1)
    with pytest.raises((TypeError, ValueError)):
        replace(diagnostics, current_frame_id=np.int64(1))
    with pytest.raises((TypeError, ValueError)):
        replace(diagnostics, expired_identity_ids=[1])
    with pytest.raises((TypeError, ValueError)):
        replace(diagnostics, expired_identity_ids=(2, 1))


@pytest.mark.parametrize("value", [2**63, -1, True, np.int64(1)])
def test_identity_ids_and_frames_require_signed_int64_exact_values(value: object) -> None:
    bank = IdentityMemoryBank(config())
    record = bank.insert(**fields())
    with pytest.raises((TypeError, ValueError)):
        replace(record, identity_id=value)
    with pytest.raises((TypeError, ValueError)):
        replace(record, last_frame_id=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"identity_id": True},
        {"identity_id": -1},
        {"semantic_probabilities": ((2, 0.5), (1, 0.5))},
        {"semantic_probabilities": ((1, 0.5),)},
        {"appearance_prototype": np.array([0.0, 0.0])},
        {"appearance_prototype": None, "feature_model_id": "clip-v1"},
        {"lifecycle": "active"},
        {"last_frame_id": np.int64(2)},
        {"last_timestamp": np.nan},
        {"extent_xyz": (1.0, 0.0, 1.0)},
        {"motion_uncertainty_m": -1.0},
    ],
)
def test_record_fails_closed(changes: dict[str, object]) -> None:
    bank = IdentityMemoryBank(config())
    record = bank.insert(**fields())
    with pytest.raises((TypeError, ValueError)):
        replace(record, **changes)


def test_bank_requires_valid_config_and_exact_frame_ids() -> None:
    with pytest.raises(TypeError):
        IdentityMemoryBank(object())  # type: ignore[arg-type]
    bank = IdentityMemoryBank(config())
    with pytest.raises(TypeError):
        bank.insert(**fields(frame_id=np.int64(1)))
    assert bank.records == ()


@pytest.mark.parametrize(
    "bad_config",
    [
        config(minimum_reid_similarity=True),
        config(minimum_reid_similarity=np.nan),
        config(minimum_reid_similarity=1.1),
        config(maximum_reid_distance_m=np.inf),
        config(maximum_reid_distance_m=0.0),
    ],
)
def test_bank_fails_closed_on_manually_constructed_config(bad_config: TemporalIdentityConfig) -> None:
    with pytest.raises((TypeError, ValueError)):
        IdentityMemoryBank(bad_config)
