from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_config import TemporalBackgroundLedgerConfig, TemporalGeometryConfig
from src.oviv2.temporal_lifecycle import TemporalEvidenceKind


BlockKey = tuple[int, int, int]
ContributionKey = tuple[int, int, int, BlockKey]
NativeIdentity = tuple[int, int]
PhysicalObservationKey = tuple[int, int, BlockKey]


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _timestamp(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _block_key(value: object) -> BlockKey:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("block_key must be a canonical three-integer tuple")
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
        for item in value
    ):
        raise TypeError("block_key must contain only integers")
    return tuple(int(item) for item in value)


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, copy=True, order="C")
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


class _FrozenCameraIntrinsics(CameraIntrinsics):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        object.__setattr__(self, "_ledger_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_ledger_frozen", False):
            raise FrozenInstanceError(f"cannot assign to field '{name}'")
        super().__setattr__(name, value)

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenCameraIntrinsics:
        memo[id(self)] = self
        return self


class _FrozenFrame(Frame):
    def __init__(self, **values: object) -> None:
        super().__init__(**values)
        object.__setattr__(self, "_ledger_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_ledger_frozen", False):
            raise FrozenInstanceError(f"cannot assign to field '{name}'")
        super().__setattr__(name, value)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenFrame:
        memo[id(self)] = self
        return self


def _frozen_frame(frame: object) -> Frame:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if type(frame) is _FrozenFrame:
        return frame
    frame_id = _integer(frame.frame_id, "frame.frame_id")
    timestamp = _timestamp(frame.timestamp, "frame.timestamp")
    if not isinstance(frame.intrinsics, CameraIntrinsics):
        raise TypeError("frame.intrinsics must be CameraIntrinsics")
    intrinsics = _FrozenCameraIntrinsics(
        _timestamp(frame.intrinsics.fx, "frame.intrinsics.fx"),
        _timestamp(frame.intrinsics.fy, "frame.intrinsics.fy"),
        _timestamp(frame.intrinsics.cx, "frame.intrinsics.cx"),
        _timestamp(frame.intrinsics.cy, "frame.intrinsics.cy"),
        _integer(frame.intrinsics.width, "frame.intrinsics.width"),
        _integer(frame.intrinsics.height, "frame.intrinsics.height"),
    )
    source_frame_id = (
        None
        if frame.source_frame_id is None
        else _integer(frame.source_frame_id, "frame.source_frame_id")
    )
    return _FrozenFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        rgb=_readonly(frame.rgb),
        depth=_readonly(frame.depth),
        pose=_readonly(frame.pose),
        intrinsics=intrinsics,
        source_frame_id=source_frame_id,
    )


@dataclass(frozen=True, eq=False)
class BackgroundContribution:
    block_key: BlockKey

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_key", _block_key(self.block_key))

    def canonical_payload(self) -> tuple[object, ...]:
        return (self.block_key,)

    def __deepcopy__(self, memo: dict[int, object]) -> BackgroundContribution:
        memo[id(self)] = self
        return self


def _native_identity(frame: Frame) -> NativeIdentity:
    if frame.source_frame_id is not None:
        return (1, frame.source_frame_id)
    return (0, frame.frame_id)


def _native_frame_index(frame: Frame) -> int:
    if frame.source_frame_id is not None:
        return frame.source_frame_id
    return frame.frame_id


def _native_frame_payload(frame: Frame) -> tuple[object, ...]:
    if type(frame) is _FrozenFrame:
        cached = getattr(frame, "_ledger_native_payload", None)
        if cached is not None:
            return cached
    payload = (
        frame.source_frame_id,
        float(frame.timestamp),
        _array_digest(frame.rgb),
        _array_digest(frame.depth),
        _array_digest(frame.pose),
        (
            frame.intrinsics.fx,
            frame.intrinsics.fy,
            frame.intrinsics.cx,
            frame.intrinsics.cy,
            frame.intrinsics.width,
            frame.intrinsics.height,
        ),
    )
    if type(frame) is _FrozenFrame:
        object.__setattr__(frame, "_ledger_native_payload", payload)
    return payload


def _processed_frame_payload(frame: Frame) -> tuple[object, ...]:
    return (frame.frame_id, _native_frame_payload(frame))


def _observation_payload(
    frame: Frame, depth_m: np.ndarray
) -> tuple[object, ...]:
    return (_native_frame_payload(frame), _array_digest(depth_m))


@dataclass(frozen=True, eq=False)
class BackgroundLedgerEvidence:
    entity_id: int
    geometry_epoch: int
    frame_id: int
    timestamp: float
    kind: TemporalEvidenceKind
    view_bin: int | None
    contributions: tuple[BackgroundContribution, ...]
    frame: Frame | None = None
    depth_m: np.ndarray | None = None
    _observation_canonical: tuple[object, ...] | None = field(
        init=False, repr=False
    )
    _native_frame_canonical: tuple[object, ...] | None = field(
        init=False, repr=False
    )
    _processed_frame_canonical: tuple[object, ...] | None = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        entity_id = _integer(self.entity_id, "entity_id")
        epoch = _integer(self.geometry_epoch, "geometry_epoch")
        frame_id = _integer(self.frame_id, "frame_id")
        timestamp = _timestamp(self.timestamp, "timestamp")
        if not isinstance(self.kind, TemporalEvidenceKind):
            raise TypeError("kind must be a TemporalEvidenceKind")
        if not isinstance(self.contributions, tuple):
            raise TypeError("contributions must be a tuple")
        if any(not isinstance(item, BackgroundContribution) for item in self.contributions):
            raise TypeError("contributions must contain BackgroundContribution values")
        if self.kind is TemporalEvidenceKind.VISIBLE_ABSENT:
            view_bin = _integer(self.view_bin, "view_bin")
            if not self.contributions:
                raise ValueError("visible-absent evidence requires contributions")
            frame = _frozen_frame(self.frame)
            depth = np.asarray(self.depth_m)
            if depth.dtype.kind != "f":
                raise TypeError("depth_m must have a floating dtype")
            if depth.shape != frame.depth.shape:
                raise ValueError("depth_m shape must match frame.depth")
            if not np.isfinite(depth).all() or np.any(depth < 0.0):
                raise ValueError("depth_m must be finite and nonnegative")
            depth_m = _readonly(depth)
            if frame.frame_id != frame_id:
                raise ValueError("observation frame_id must match evidence frame_id")
            if frame.timestamp != timestamp:
                raise ValueError("observation timestamp must match evidence timestamp")
            positive = depth_m > 0.0
            native_depth = np.asarray(frame.depth)
            if np.any(
                positive
                & (
                    ~np.isfinite(native_depth)
                    | (native_depth <= 0.0)
                    | (depth_m != native_depth)
                )
            ):
                raise ValueError(
                    "nonzero masked depth must equal valid native frame.depth"
                )
            native_frame_canonical = _native_frame_payload(frame)
            processed_frame_canonical = _processed_frame_payload(frame)
            observation_canonical = _observation_payload(frame, depth_m)
        else:
            if self.kind not in (TemporalEvidenceKind.PRESENT, TemporalEvidenceKind.OCCLUDED):
                raise ValueError("ledger evidence must be visible-absent, present, or occluded")
            if self.view_bin is not None:
                raise ValueError("view_bin must be None for present or occluded evidence")
            if self.contributions:
                raise ValueError("present or occluded evidence cannot contain contributions")
            if self.frame is not None or self.depth_m is not None:
                raise ValueError(
                    "present or occluded evidence cannot contain an observation"
                )
            view_bin = None
            frame = None
            depth_m = None
            native_frame_canonical = None
            processed_frame_canonical = None
            observation_canonical = None
        block_keys = tuple(item.block_key for item in self.contributions)
        if len(set(block_keys)) != len(block_keys):
            raise ValueError("contribution block keys must be unique")
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "geometry_epoch", epoch)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "view_bin", view_bin)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "depth_m", depth_m)
        object.__setattr__(self, "_native_frame_canonical", native_frame_canonical)
        object.__setattr__(
            self, "_processed_frame_canonical", processed_frame_canonical
        )
        object.__setattr__(self, "_observation_canonical", observation_canonical)
        object.__setattr__(
            self,
            "contributions",
            tuple(sorted(self.contributions, key=lambda item: item.block_key)),
        )

    def __deepcopy__(self, memo: dict[int, object]) -> BackgroundLedgerEvidence:
        memo[id(self)] = self
        return self


class LedgerDecision(str, Enum):
    STAGED = "staged"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    NO_OP = "no_op"
    REJECTED_CAPACITY = "rejected_capacity"
    REJECTED_INTEGRATION = "rejected_integration"


@dataclass(frozen=True, eq=False)
class _Record:
    entity_id: int
    geometry_epoch: int
    frame_id: int
    timestamp: float
    view_bin: int
    contribution: BackgroundContribution
    frame: Frame
    depth_m: np.ndarray
    observation_canonical: tuple[object, ...]

    @property
    def key(self) -> ContributionKey:
        return (self.entity_id, self.geometry_epoch, self.frame_id, self.contribution.block_key)

    def __deepcopy__(self, memo: dict[int, object]) -> _Record:
        memo[id(self)] = self
        return self


def _validate_config(config: object) -> TemporalBackgroundLedgerConfig:
    if not isinstance(config, TemporalBackgroundLedgerConfig):
        raise TypeError("config must be a TemporalBackgroundLedgerConfig")
    names = (
        "maximum_journal_blocks",
        "commit_support_frames",
        "commit_distinct_view_bins",
        "minimum_commit_frame_gap",
        "maximum_records_per_block",
    )
    normalized: dict[str, int] = {}
    for name in names:
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"config.{name} must be an integer")
        normalized[name] = int(value)
        if normalized[name] <= 0:
            raise ValueError(f"config.{name} must be positive")
    config = replace(config, **normalized)
    if config.maximum_records_per_block < config.commit_support_frames:
        raise ValueError("maximum_records_per_block cannot be below commit_support_frames")
    if config.commit_support_frames not in {2, 3, 4}:
        raise ValueError("commit_support_frames must be one of 2, 3, or 4")
    if config.commit_distinct_view_bins not in {2, 3}:
        raise ValueError("commit_distinct_view_bins must be 2 or 3")
    return config


def _array_digest(value: np.ndarray) -> tuple[str, tuple[int, ...], str]:
    array = np.asarray(value)
    return (array.dtype.str, array.shape, hashlib.sha256(array.tobytes(order="C")).hexdigest())


def _record_payload(record: _Record) -> tuple[object, ...]:
    return (
        record.entity_id,
        record.geometry_epoch,
        record.frame_id,
        record.timestamp,
        record.view_bin,
        record.contribution.canonical_payload(),
        record.observation_canonical,
    )


def _record_group_key(record: _Record) -> tuple[int, int, BlockKey]:
    return (record.entity_id, record.geometry_epoch, record.contribution.block_key)


def _physical_observation_key(record: _Record) -> PhysicalObservationKey:
    native = _native_identity(record.frame)
    return (native[0], native[1], record.contribution.block_key)


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _LedgerState:
    volume: TemporalBackgroundVolume
    generation: int
    provisional: dict[ContributionKey, _Record]
    committed: dict[ContributionKey, _Record]
    event_digests: dict[tuple[int, int, int], str]
    native_frames: dict[NativeIdentity, tuple[object, ...]]
    native_view_bins: dict[NativeIdentity, int]
    processed_frames: dict[int, tuple[object, ...]]
    last_frame_id: int
    last_timestamp: float
    base_volume: TemporalBackgroundVolume | None = None
    combined_volume: TemporalBackgroundVolume | None = None
    derivation_digest: str | None = None
    base_observation_watermark: tuple[object, ...] | None = None


class ReversibleBackgroundLedger:
    __hash__ = None

    def __init__(
        self,
        geometry_config: TemporalGeometryConfig,
        config: TemporalBackgroundLedgerConfig,
    ) -> None:
        self._config = _validate_config(config)
        volume = TemporalBackgroundVolume(geometry_config)
        if self.config.maximum_journal_blocks > volume.config.background_block_count:
            raise ValueError(
                "maximum_journal_blocks cannot exceed geometry background_block_count"
            )
        self._maximum_ownership_records_per_observation = (
            volume.config.maximum_entities
        )
        self._state = _LedgerState(
            volume=volume,
            generation=0,
            provisional={},
            committed={},
            event_digests={},
            native_frames={},
            native_view_bins={},
            processed_frames={},
            last_frame_id=-1,
            last_timestamp=-math.inf,
        )

    @property
    def _volume(self) -> TemporalBackgroundVolume:
        return self._state.volume

    @property
    def _generation(self) -> int:
        return self._state.generation

    @property
    def _provisional(self) -> dict[ContributionKey, _Record]:
        return self._state.provisional

    @property
    def _committed(self) -> dict[ContributionKey, _Record]:
        return self._state.committed

    @property
    def _event_digests(self) -> dict[tuple[int, int, int], str]:
        return self._state.event_digests

    @property
    def _native_frames(self) -> dict[NativeIdentity, tuple[object, ...]]:
        return self._state.native_frames

    @property
    def _native_view_bins(self) -> dict[NativeIdentity, int]:
        return self._state.native_view_bins

    @property
    def _processed_frames(self) -> dict[int, tuple[object, ...]]:
        return self._state.processed_frames

    @property
    def _last_frame_id(self) -> int:
        return self._state.last_frame_id

    @property
    def _last_timestamp(self) -> float:
        return self._state.last_timestamp

    def __deepcopy__(self, memo: dict[int, object]) -> ReversibleBackgroundLedger:
        clone = self.clone()
        memo[id(self)] = clone
        return clone

    def clone(self) -> ReversibleBackgroundLedger:
        clone = object.__new__(type(self))
        clone._config = self._config
        clone._maximum_ownership_records_per_observation = (
            self._maximum_ownership_records_per_observation
        )
        clone._state = _LedgerState(
            volume=self._volume.clone(),
            generation=self._generation,
            provisional=dict(self._provisional),
            committed=dict(self._committed),
            event_digests=dict(self._event_digests),
            native_frames=dict(self._native_frames),
            native_view_bins=dict(self._native_view_bins),
            processed_frames=dict(self._processed_frames),
            last_frame_id=self._last_frame_id,
            last_timestamp=self._last_timestamp,
            base_volume=(
                None
                if self._state.base_volume is None
                else self._state.base_volume.clone()
            ),
            combined_volume=(
                None
                if self._state.combined_volume is None
                else self._state.combined_volume.clone()
            ),
            derivation_digest=self._state.derivation_digest,
            base_observation_watermark=self._state.base_observation_watermark,
        )
        return clone

    @property
    def config(self) -> TemporalBackgroundLedgerConfig:
        return self._config

    @property
    def provisional_count(self) -> int:
        return len(self._provisional)

    @property
    def committed_record_count(self) -> int:
        return len(self._committed)

    @property
    def committed_generation(self) -> int:
        return self._generation

    @property
    def provisional_keys(self) -> tuple[ContributionKey, ...]:
        return tuple(sorted(self._provisional))

    @property
    def committed_volume(self) -> TemporalBackgroundVolume:
        return self._volume.clone()

    @property
    def _published_volume(self) -> TemporalBackgroundVolume:
        value = self._state.combined_volume
        return self._volume if value is None else value

    @property
    def combined_volume(self) -> TemporalBackgroundVolume:
        return self._published_volume.clone()

    def validate_combined_volume(self, *, rebuild: bool = True) -> None:
        del rebuild
        if self._state.combined_volume is not None:
            raise RuntimeError(
                "ledger subclass must validate its combined volume derivation"
            )

    def _evidence_digest(self, evidence: BackgroundLedgerEvidence) -> str:
        return _digest(
            (
                evidence.entity_id,
                evidence.geometry_epoch,
                evidence.frame_id,
                evidence.timestamp,
                evidence.kind.value,
                evidence.view_bin,
                tuple(item.canonical_payload() for item in evidence.contributions),
                evidence._observation_canonical,
            )
        )

    def journal_digest(self) -> str:
        blocks = [
            (dtype, shape, hashlib.sha256(data).hexdigest())
            for dtype, shape, data in self._volume.canonical_block_state()
        ]
        return _digest(
            {
                "geometry_config": asdict(self._volume.config),
                "ledger_config": asdict(self.config),
                "generation": self._generation,
                "last_frame_id": self._last_frame_id,
                "last_timestamp": (
                    None if self._last_frame_id < 0 else self._last_timestamp
                ),
                "events": [
                    (key, self._event_digests[key])
                    for key in sorted(self._event_digests)
                ],
                "native_frames": [
                    (identity, self._native_frames[identity])
                    for identity in sorted(self._native_frames)
                ],
                "native_view_bins": [
                    (identity, self._native_view_bins[identity])
                    for identity in sorted(self._native_view_bins)
                ],
                "processed_frames": [
                    (frame_id, self._processed_frames[frame_id])
                    for frame_id in sorted(self._processed_frames)
                ],
                "provisional": [
                    _record_payload(self._provisional[key])
                    for key in sorted(self._provisional)
                ],
                "committed": [
                    _record_payload(self._committed[key])
                    for key in sorted(self._committed)
                ],
                "blocks": blocks,
                "base_blocks": (
                    None
                    if self._state.base_volume is None
                    else [
                        (dtype, shape, hashlib.sha256(data).hexdigest())
                        for dtype, shape, data
                        in self._state.base_volume.canonical_block_state()
                    ]
                ),
                "combined_blocks": (
                    None
                    if self._state.combined_volume is None
                    else [
                        (dtype, shape, hashlib.sha256(data).hexdigest())
                        for dtype, shape, data
                        in self._state.combined_volume.canonical_block_state()
                    ]
                ),
                "derivation_digest": self._state.derivation_digest,
                "base_observation_watermark": (
                    self._state.base_observation_watermark
                ),
            }
        )

    def committed_digest(self) -> str:
        blocks = [
            (dtype, shape, hashlib.sha256(data).hexdigest())
            for dtype, shape, data in self._volume.canonical_block_state()
        ]
        return _digest(
            {
                "geometry_config": asdict(self._volume.config),
                "ledger_config": asdict(self.config),
                "generation": self._generation,
                "records": [
                    _record_payload(self._committed[key])
                    for key in sorted(self._committed)
                ],
                "blocks": blocks,
            }
        )

    def stage(self, evidence: BackgroundLedgerEvidence) -> LedgerDecision:
        if not isinstance(evidence, BackgroundLedgerEvidence):
            raise TypeError("evidence must be BackgroundLedgerEvidence")
        event_key = (evidence.entity_id, evidence.geometry_epoch, evidence.frame_id)
        evidence_digest = self._evidence_digest(evidence)
        previous_digest = self._event_digests.get(event_key)
        if previous_digest is not None:
            if previous_digest == evidence_digest:
                return LedgerDecision.NO_OP
            raise ValueError("conflicting duplicate ledger evidence")
        if evidence.frame_id < self._last_frame_id:
            raise ValueError("evidence.frame_id cannot move backwards")
        if evidence.timestamp < self._last_timestamp:
            raise ValueError("evidence.timestamp cannot move backwards")
        if (
            evidence.frame_id == self._last_frame_id
            and evidence.timestamp != self._last_timestamp
        ):
            raise ValueError("evidence.timestamp must agree within one frame")
        current_event_count = (
            len(self._event_digests)
            if evidence.frame_id == self._last_frame_id
            else 0
        )
        if current_event_count >= self._volume.config.maximum_entities:
            return LedgerDecision.REJECTED_CAPACITY

        try:
            if evidence.contributions:
                assert evidence.frame is not None
                assert evidence.depth_m is not None
                assert evidence._native_frame_canonical is not None
                assert evidence._processed_frame_canonical is not None
                native_identity = _native_identity(evidence.frame)
                existing_native = self._native_frames.get(native_identity)
                if (
                    existing_native is not None
                    and existing_native != evidence._native_frame_canonical
                ):
                    raise ValueError(
                        "native frame content conflicts for the same identity"
                    )
                existing_view_bin = self._native_view_bins.get(native_identity)
                if (
                    existing_view_bin is not None
                    and existing_view_bin != evidence.view_bin
                ):
                    raise ValueError(
                        "native frame identity is already bound to another view_bin"
                    )
                existing_processed = self._processed_frames.get(evidence.frame_id)
                if (
                    existing_processed is not None
                    and existing_processed != evidence._processed_frame_canonical
                ):
                    raise ValueError(
                        "processed frame_id maps to conflicting native frame content"
                    )
                touched = self._volume.candidate_block_keys(
                    evidence.frame,
                    evidence.depth_m,
                )
                declared = tuple(item.block_key for item in evidence.contributions)
                if declared != touched:
                    raise ValueError(
                        "contribution block_key values must exactly match touched blocks"
                    )
        except (TypeError, ValueError):
            raise
        except Exception:
            return LedgerDecision.REJECTED_INTEGRATION

        provisional = dict(self._provisional)
        committed = dict(self._committed)
        if evidence.kind in (TemporalEvidenceKind.PRESENT, TemporalEvidenceKind.OCCLUDED):
            matching_provisional = tuple(
                key for key in provisional
                if key[:2] == (evidence.entity_id, evidence.geometry_epoch)
            )
            matching_committed = tuple(
                key for key in committed
                if key[:2] == (evidence.entity_id, evidence.geometry_epoch)
            )
            if not matching_provisional and not matching_committed:
                self._publish_event(evidence, evidence_digest, provisional, committed)
                return LedgerDecision.NO_OP
            for key in matching_provisional:
                del provisional[key]
            for key in matching_committed:
                del committed[key]
            rebuilt = None
            generation = None
            if matching_committed:
                try:
                    rebuilt = self._rebuild_committed_volume(committed)
                except (TypeError, ValueError):
                    raise
                except Exception:
                    return LedgerDecision.REJECTED_INTEGRATION
                generation = self._generation + 1
            self._publish_event(
                evidence,
                evidence_digest,
                provisional,
                committed,
                volume=rebuilt,
                generation=generation,
            )
            return LedgerDecision.CANCELLED

        assert evidence.view_bin is not None
        assert evidence.frame is not None
        assert evidence.depth_m is not None
        assert evidence._observation_canonical is not None
        shared_frames: dict[NativeIdentity, Frame] = {}
        shared_depths: dict[tuple[NativeIdentity, tuple[object, ...]], np.ndarray] = {}
        for existing_record in (*provisional.values(), *committed.values()):
            native = _native_identity(existing_record.frame)
            shared_frames.setdefault(native, existing_record.frame)
            depth_key = (native, existing_record.observation_canonical[1])
            shared_depths.setdefault(depth_key, existing_record.depth_m)
        native = _native_identity(evidence.frame)
        record_frame = shared_frames.get(native, evidence.frame)
        depth_key = (native, evidence._observation_canonical[1])
        record_depth = shared_depths.get(depth_key, evidence.depth_m)
        for contribution in evidence.contributions:
            record = _Record(
                evidence.entity_id,
                evidence.geometry_epoch,
                evidence.frame_id,
                evidence.timestamp,
                evidence.view_bin,
                contribution,
                record_frame,
                record_depth,
                evidence._observation_canonical,
            )
            if record.key in provisional or record.key in committed:
                raise ValueError("duplicate contribution key")
            provisional[record.key] = record
        if not self._within_capacity(provisional, committed):
            return LedgerDecision.REJECTED_CAPACITY

        eligible_groups: set[tuple[int, int, BlockKey]] = set()
        grouped: dict[tuple[int, int, BlockKey], list[_Record]] = {}
        for record in provisional.values():
            group_key = _record_group_key(record)
            grouped.setdefault(group_key, []).append(record)
        for group_key, records in grouped.items():
            physical: dict[NativeIdentity, _Record] = {}
            for record in sorted(records, key=lambda item: item.key):
                physical.setdefault(_native_identity(record.frame), record)
            observations = tuple(physical.values())
            view_bins = {record.view_bin for record in observations}
            native_indices = {
                _native_frame_index(record.frame) for record in observations
            }
            if (
                len(observations) >= self.config.commit_support_frames
                and len(view_bins) >= self.config.commit_distinct_view_bins
                and max(native_indices) - min(native_indices)
                >= self.config.minimum_commit_frame_gap
            ):
                eligible_groups.add(group_key)
        committing = {
            key: record
            for key, record in provisional.items()
            if _record_group_key(record) in eligible_groups
        }
        if committing:
            committed.update(committing)
            for key in committing:
                del provisional[key]
            try:
                rebuilt = self._rebuild_committed_volume(committed)
            except Exception:
                return LedgerDecision.REJECTED_INTEGRATION
            self._publish_event(
                evidence,
                evidence_digest,
                provisional,
                committed,
                volume=rebuilt,
                generation=self._generation + 1,
            )
            return LedgerDecision.COMMITTED
        self._publish_event(evidence, evidence_digest, provisional, committed)
        return LedgerDecision.STAGED

    def _within_capacity(
        self,
        provisional: dict[ContributionKey, _Record],
        committed: dict[ContributionKey, _Record],
    ) -> bool:
        records = (*provisional.values(), *committed.values())
        blocks = {record.contribution.block_key for record in records}
        if len(blocks) > self.config.maximum_journal_blocks:
            return False
        observation_counts: dict[BlockKey, set[PhysicalObservationKey]] = {}
        ownership_counts: dict[PhysicalObservationKey, int] = {}
        for record in records:
            observation_key = _physical_observation_key(record)
            block_key = record.contribution.block_key
            observation_counts.setdefault(block_key, set()).add(observation_key)
            ownership_counts[observation_key] = (
                ownership_counts.get(observation_key, 0) + 1
            )
        return (
            all(
                len(observations) <= self.config.maximum_records_per_block
                for observations in observation_counts.values()
            )
            and all(
                count <= self._maximum_ownership_records_per_observation
                for count in ownership_counts.values()
            )
        )

    def _aggregate_observations(
        self,
        committed: dict[ContributionKey, _Record],
    ) -> tuple[
        tuple[PhysicalObservationKey, BlockKey, Frame, np.ndarray], ...
    ]:
        grouped: dict[PhysicalObservationKey, list[_Record]] = {}
        for key in sorted(committed):
            record = committed[key]
            grouped.setdefault(_physical_observation_key(record), []).append(record)
        observations: list[
            tuple[PhysicalObservationKey, BlockKey, Frame, np.ndarray]
        ] = []
        for observation_key in sorted(grouped):
            records = grouped[observation_key]
            first = records[0]
            native_payload = _native_frame_payload(first.frame)
            merged = np.zeros_like(first.frame.depth)
            for record in records:
                if _native_frame_payload(record.frame) != native_payload:
                    raise ValueError("native frame content conflicts during rebuild")
                positive = record.depth_m > 0.0
                overlap = positive & (merged > 0.0)
                if np.any(overlap & (merged != record.depth_m)):
                    raise ValueError("masked depth conflict during rebuild")
                merged[positive] = record.depth_m[positive]
            observations.append(
                (
                    observation_key,
                    first.contribution.block_key,
                    first.frame,
                    _readonly(merged),
                )
            )
        return tuple(observations)

    def _rebuild_committed_volume(
        self, committed: dict[ContributionKey, _Record]
    ) -> TemporalBackgroundVolume:
        return TemporalBackgroundVolume.rebuild_blocks(
            self._volume.config, self._aggregate_observations(committed)
        )

    def _publish_event(
        self,
        evidence: BackgroundLedgerEvidence,
        evidence_digest: str,
        provisional: dict[ContributionKey, _Record],
        committed: dict[ContributionKey, _Record],
        *,
        volume: TemporalBackgroundVolume | None = None,
        generation: int | None = None,
    ) -> None:
        events = (
            dict(self._event_digests)
            if evidence.frame_id == self._last_frame_id
            else {}
        )
        events[(evidence.entity_id, evidence.geometry_epoch, evidence.frame_id)] = (
            evidence_digest
        )
        native_frames, native_view_bins, processed_frames = self._record_indexes(
            provisional, committed, evidence.frame_id
        )
        next_state = _LedgerState(
            volume=self._volume if volume is None else volume,
            generation=self._generation if generation is None else generation,
            provisional=provisional,
            committed=committed,
            event_digests=events,
            native_frames=native_frames,
            native_view_bins=native_view_bins,
            processed_frames=processed_frames,
            last_frame_id=evidence.frame_id,
            last_timestamp=evidence.timestamp,
            base_volume=self._state.base_volume,
            combined_volume=self._state.combined_volume,
            derivation_digest=self._state.derivation_digest,
            base_observation_watermark=(
                self._state.base_observation_watermark
            ),
        )
        next_state = self._finalize_state(next_state)
        self._before_publish(next_state)
        self._state = next_state

    def _finalize_state(self, next_state: _LedgerState) -> _LedgerState:
        return next_state

    @staticmethod
    def _record_indexes(
        provisional: dict[ContributionKey, _Record],
        committed: dict[ContributionKey, _Record],
        current_frame_id: int,
    ) -> tuple[
        dict[NativeIdentity, tuple[object, ...]],
        dict[NativeIdentity, int],
        dict[int, tuple[object, ...]],
    ]:
        native_frames: dict[NativeIdentity, tuple[object, ...]] = {}
        native_view_bins: dict[NativeIdentity, int] = {}
        processed_frames: dict[int, tuple[object, ...]] = {}
        for record in (*provisional.values(), *committed.values()):
            native = _native_identity(record.frame)
            native_payload = _native_frame_payload(record.frame)
            previous_native = native_frames.setdefault(native, native_payload)
            if previous_native != native_payload:
                raise ValueError("native frame content conflicts while indexing records")
            previous_view = native_view_bins.setdefault(native, record.view_bin)
            if previous_view != record.view_bin:
                raise ValueError("native frame view_bin conflicts while indexing records")
            if record.frame_id == current_frame_id:
                processed_payload = (
                    record.frame_id,
                    _native_frame_payload(record.frame),
                )
                previous_processed = processed_frames.setdefault(
                    record.frame_id, processed_payload
                )
                if previous_processed != processed_payload:
                    raise ValueError(
                        "processed frame content conflicts while indexing records"
                    )
        return native_frames, native_view_bins, processed_frames

    def _before_publish(self, next_state: _LedgerState) -> None:
        del next_state
