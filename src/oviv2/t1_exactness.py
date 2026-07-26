from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import math
import struct
from typing import Any, Iterator, Mapping

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.dense_projection import DenseSemanticIntegrator
from src.oviv2.entities import EntityRegistry
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.observation_graph import CausalObservationGraph
from src.oviv2.observations import FrameObservation
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.reference_readout import ReferenceReadoutState
from src.oviv2.runtime import Oviv2Runtime
from src.oviv2.temporal_runtime import TemporalCurrentRuntime
from src.oviv2.temporal_state import TemporalGeometryState, TemporalRuntimeState
from src.oviv2.tracking import LocalTracker
from src.oviv2.visibility import VoxelVisibilityProjector


_TRUSTED_STATE_OBJECT_TYPES = (
    SparseEvidenceStore,
    ReversibleOwnershipStore,
    LocalTracker,
    CausalObservationGraph,
    EntityRegistry,
    VoxelVisibilityProjector,
    DenseSemanticIntegrator,
)


def _pack(tag: bytes, content: bytes) -> bytes:
    return tag + struct.pack(">Q", len(content)) + content


def _canonical_bytes(value: Any) -> bytes:
    if value is None:
        return b"n"
    if type(value) is bool:
        return b"b1" if value else b"b0"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return _pack(b"i", str(int(value)).encode("ascii"))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("state contains a non-finite number")
        return b"f" + struct.pack(">d", number)
    if isinstance(value, str):
        return _pack(b"s", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _pack(b"y", value)
    if isinstance(value, Enum):
        return _pack(b"e", _canonical_bytes(type(value).__qualname__) + _canonical_bytes(value.value))
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be hashed deterministically")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError("array inputs must be finite")
        array = np.ascontiguousarray(value)
        header = _canonical_bytes((array.dtype.str, array.shape))
        return _pack(b"a", header + array.tobytes(order="C"))
    if isinstance(value, Mapping):
        items = [(_canonical_bytes(key), _canonical_bytes(item)) for key, item in value.items()]
        items.sort(key=lambda item: item[0])
        if len({key for key, _ in items}) != len(items):
            raise ValueError("mapping keys have ambiguous canonical encodings")
        return _pack(b"m", b"".join(_pack(b"k", key) + _pack(b"v", item) for key, item in items))
    if isinstance(value, tuple):
        return _pack(b"t", b"".join(_pack(b"x", _canonical_bytes(item)) for item in value))
    if isinstance(value, list):
        return _pack(b"l", b"".join(_pack(b"x", _canonical_bytes(item)) for item in value))
    if isinstance(value, (set, frozenset)):
        items = sorted(_canonical_bytes(item) for item in value)
        if len(set(items)) != len(items):
            raise ValueError("set values have ambiguous canonical encodings")
        return _pack(b"q", b"".join(_pack(b"x", item) for item in items))
    raise TypeError(f"unsupported deterministic hash value: {type(value).__name__}")


def _trusted_dump(value: Any) -> Any:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if isinstance(value, (np.integer, np.floating, np.ndarray, Enum)):
        return value
    if isinstance(value, Mapping):
        return {_trusted_dump(key): _trusted_dump(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_trusted_dump(item) for item in value)
    if isinstance(value, list):
        return [_trusted_dump(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return frozenset(_trusted_dump(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        if not type(value).__module__.startswith("src."):
            raise TypeError(
                f"unsupported deterministic hash value: {type(value).__name__}"
            )
        payload = tuple(
            (field.name, _trusted_dump(getattr(value, field.name)))
            for field in fields(value)
        )
        return ("dataclass", type(value).__module__, type(value).__qualname__, payload)
    if type(value) in _TRUSTED_STATE_OBJECT_TYPES:
        payload = tuple(
            (name, _trusted_dump(item)) for name, item in sorted(vars(value).items())
        )
        return ("state", type(value).__module__, type(value).__qualname__, payload)
    raise TypeError(f"unsupported deterministic hash value: {type(value).__name__}")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(_trusted_dump(value))).hexdigest()


def shared_input_sha256(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None,
) -> str:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if type(observations) is not tuple or any(type(item) is not FrameObservation for item in observations):
        raise TypeError("observations must be an exact tuple of FrameObservation values")
    if dense_semantics is not None and type(dense_semantics) is not DenseSemanticFrame:
        raise TypeError("dense_semantics must be a DenseSemanticFrame or None")
    return _digest((frame, observations, dense_semantics))


def _geometry_dump(volume: SparseTsdfVolume) -> tuple[Any, ...]:
    active = volume._grid.hashmap().active_buf_indices()
    keys = volume._grid.hashmap().key_tensor()[active].numpy()
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0])) if len(keys) else np.empty((0,), dtype=np.int64)
    return (
        volume.config,
        np.asarray(keys)[order],
        tuple((name, volume._grid.attribute(name)[active].numpy()[order]) for name in volume._ATTRIBUTE_NAMES),
    )


def cumulative_state_sha256(runtime: Oviv2Runtime) -> str:
    if not isinstance(runtime, Oviv2Runtime):
        raise TypeError("runtime must be an Oviv2Runtime")
    return _digest((
        runtime.scene_id, runtime.config, runtime.dense_semantic_provenance,
        _geometry_dump(runtime.geometry), runtime.evidence, runtime.ownership,
        runtime.tracker, runtime.registry, runtime.visibility,
        runtime.dense_semantic_integrator,
        runtime.revision, runtime.last_frame_id, runtime.last_timestamp,
    ))


def temporal_state_sha256(runtime: TemporalCurrentRuntime | object) -> str:
    state = getattr(runtime, "state", runtime)
    if isinstance(state, TemporalRuntimeState):
        return _digest(state.canonical_dump())
    if isinstance(state, ReferenceReadoutState):
        return _digest(state)
    raise TypeError("runtime must expose a supported temporal state")


def _clone_geometry(volume: SparseTsdfVolume) -> SparseTsdfVolume:
    clone = SparseTsdfVolume(volume.config)
    active = volume._grid.hashmap().active_buf_indices()
    if int(active.shape[0]):
        keys = volume._grid.hashmap().key_tensor()[active]
        destination, activated = clone._grid.hashmap().activate(keys)
        if not np.all(activated.numpy()):
            raise RuntimeError("failed to snapshot cumulative TSDF blocks")
        for name in volume._ATTRIBUTE_NAMES:
            clone._grid.attribute(name)[destination] = volume._grid.attribute(name)[active]
    return clone


def _clone_temporal_state(state: TemporalRuntimeState) -> TemporalRuntimeState:
    geometry = TemporalGeometryState(
        copy.deepcopy(state.geometry.epochs),
        state.geometry.maximum_epochs_per_identity,
        state.geometry.maximum_retained_epochs,
        state.geometry.next_epoch_ids,
    )
    return TemporalRuntimeState(
        state.scene_id, state.revision, state.last_frame_id, state.last_timestamp,
        state.next_entity_id, state.entities, state.background, state.tracker,
        state.identities, geometry, state.lifecycle_beliefs,
        state.background_ledger, state.export_tracker, copy.deepcopy(state.diagnostics),
    )


def _restore_nested_object(target: object, snapshot: object) -> None:
    if not hasattr(target, "__dict__") or not hasattr(snapshot, "__dict__"):
        raise TypeError("nested transaction state is not restorable")
    clone_method = getattr(snapshot, "clone", None)
    owned = clone_method() if callable(clone_method) else copy.deepcopy(snapshot)
    target.__dict__.clear()
    target.__dict__.update(owned.__dict__)


@dataclass(frozen=True)
class _CumulativeSnapshot:
    attributes: tuple[tuple[str, object], ...]
    geometry: SparseTsdfVolume
    nested: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class _ArraySnapshot:
    array: np.ndarray
    content: np.ndarray
    writeable: bool

    @classmethod
    def capture(cls, array: np.ndarray) -> "_ArraySnapshot":
        return cls(array, np.array(array, copy=True), bool(array.flags.writeable))

    def restore(self) -> None:
        if self.writeable:
            self.array.flags.writeable = True
            np.copyto(self.array, self.content, casting="no")
        elif not np.array_equal(self.array, self.content):
            raise RuntimeError("immutable shared input array changed")
        self.array.flags.writeable = self.writeable


@dataclass(frozen=True)
class _ObjectSnapshot:
    target: object
    fields: tuple[tuple[str, object, _ArraySnapshot | None], ...]

    @classmethod
    def capture(cls, target: object) -> "_ObjectSnapshot":
        if type(target) not in {Frame, CameraIntrinsics, FrameObservation, DenseSemanticFrame}:
            raise TypeError("unsupported shared input object")
        values = []
        for field in fields(target):
            value = getattr(target, field.name)
            array = _ArraySnapshot.capture(value) if isinstance(value, np.ndarray) else None
            values.append((field.name, value, array))
        return cls(target, tuple(values))

    def restore(self) -> None:
        expected = {name for name, _, _ in self.fields}
        if hasattr(self.target, "__dict__"):
            for name in set(vars(self.target)) - expected:
                delattr(self.target, name)
        for name, value, array in self.fields:
            if array is not None:
                array.restore()
                value = array.array
            object.__setattr__(self.target, name, value)


@dataclass(frozen=True)
class _SharedInputSnapshot:
    objects: tuple[_ObjectSnapshot, ...]

    @classmethod
    def capture(
        cls,
        frame: Frame,
        observations: tuple[FrameObservation, ...],
        dense_semantics: DenseSemanticFrame | None,
    ) -> "_SharedInputSnapshot":
        objects = [_ObjectSnapshot.capture(frame), _ObjectSnapshot.capture(frame.intrinsics)]
        objects.extend(_ObjectSnapshot.capture(item) for item in observations)
        if dense_semantics is not None:
            objects.append(_ObjectSnapshot.capture(dense_semantics))
        return cls(tuple(objects))

    def restore(self) -> None:
        for snapshot in self.objects:
            snapshot.restore()


@dataclass(frozen=True)
class DualTransactionSnapshot:
    cumulative: _CumulativeSnapshot
    temporal_attributes: tuple[tuple[str, object], ...]
    temporal_state: TemporalRuntimeState | ReferenceReadoutState
    temporal_public: tuple[tuple[str, object], ...]
    shared_inputs: _SharedInputSnapshot | None

    @classmethod
    def capture(
        cls,
        cumulative: Oviv2Runtime,
        temporal: object,
        frame: Frame | None = None,
        observations: tuple[FrameObservation, ...] = (),
        dense_semantics: DenseSemanticFrame | None = None,
    ) -> "DualTransactionSnapshot":
        if not isinstance(cumulative, Oviv2Runtime):
            raise TypeError("cumulative must be an Oviv2Runtime")
        attributes = tuple(cumulative.__dict__.items())
        nested = tuple(
            (name, copy.deepcopy(value))
            for name, value in attributes
            if name in {
                "evidence", "ownership", "tracker", "registry", "visibility",
                "dense_semantic_integrator",
            }
        )
        state = getattr(temporal, "state", None)
        if isinstance(state, TemporalRuntimeState):
            state_snapshot: TemporalRuntimeState | ReferenceReadoutState = (
                _clone_temporal_state(state)
            )
            temporal_public = tuple(
                (name, getattr(state, name))
                for name in (
                    "geometry", "lifecycle_beliefs", "export_tracker", "diagnostics"
                )
            )
        elif isinstance(state, ReferenceReadoutState):
            state_snapshot = copy.deepcopy(state)
            temporal_public = ()
        else:
            raise TypeError("temporal runtime has an unsupported state")
        return cls(
            _CumulativeSnapshot(attributes, _clone_geometry(cumulative.geometry), nested),
            tuple(temporal.__dict__.items()), state_snapshot,
            temporal_public,
            None if frame is None else _SharedInputSnapshot.capture(
                frame, observations, dense_semantics
            ),
        )

    def restore(self, cumulative: Oviv2Runtime, temporal: object) -> None:
        original = dict(self.cumulative.attributes)
        geometry = original["geometry"]
        assert isinstance(geometry, SparseTsdfVolume)
        geometry.__dict__.clear()
        geometry.__dict__.update(self.cumulative.geometry.__dict__)
        for name, clone in self.cumulative.nested:
            target = original[name]
            if hasattr(target, "__dict__") and hasattr(clone, "__dict__"):
                target.__dict__.clear()
                target.__dict__.update(copy.deepcopy(clone.__dict__))
        cumulative.__dict__.clear()
        cumulative.__dict__.update(original)

        temporal_original = dict(self.temporal_attributes)
        original_state = temporal_original["state"]
        if isinstance(original_state, TemporalRuntimeState) and isinstance(
            self.temporal_state, TemporalRuntimeState
        ):
            for name in original_state.__slots__:
                original_value = object.__getattribute__(original_state, name)
                snapshot_value = object.__getattribute__(self.temporal_state, name)
                if name in {
                    "_background_state", "_tracker_state", "_identities_state",
                    "_ledger_state",
                } and original_value is not None and snapshot_value is not None:
                    _restore_nested_object(original_value, snapshot_value)
                    object.__setattr__(original_state, name, original_value)
                else:
                    object.__setattr__(original_state, name, snapshot_value)
            public = dict(self.temporal_public)
            geometry = public["geometry"]
            geometry.__dict__.clear()
            geometry.__dict__.update(self.temporal_state.geometry.__dict__)
            object.__setattr__(original_state, "geometry", geometry)
            lifecycle = public["lifecycle_beliefs"]
            for target, snapshot in zip(
                lifecycle, self.temporal_state.lifecycle_beliefs, strict=True
            ):
                _restore_nested_object(target, snapshot)
            object.__setattr__(original_state, "lifecycle_beliefs", lifecycle)
            for name in ("export_tracker", "diagnostics"):
                target = public[name]
                _restore_nested_object(target, getattr(self.temporal_state, name))
                object.__setattr__(original_state, name, target)
        elif isinstance(original_state, ReferenceReadoutState) and isinstance(
            self.temporal_state, ReferenceReadoutState
        ):
            for field in fields(original_state):
                object.__setattr__(
                    original_state, field.name,
                    copy.deepcopy(getattr(self.temporal_state, field.name)),
                )
        temporal.__dict__.clear()
        temporal.__dict__.update(temporal_original)
        if self.shared_inputs is not None:
            self.shared_inputs.restore()


def _input_arrays(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None,
) -> tuple[np.ndarray, ...]:
    arrays: list[np.ndarray] = [frame.rgb, frame.depth, frame.pose]
    for observation in observations:
        arrays.append(observation.mask)
        if observation.image_feature is not None:
            arrays.append(observation.image_feature)
        if observation.text_feature is not None:
            arrays.append(observation.text_feature)
    if dense_semantics is not None:
        for field in fields(dense_semantics):
            value = getattr(dense_semantics, field.name)
            if isinstance(value, np.ndarray):
                arrays.append(value)
    unique: dict[int, np.ndarray] = {}
    for array in arrays:
        unique.setdefault(id(array), array)
    return tuple(unique.values())


@contextmanager
def frozen_shared_inputs(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None,
) -> Iterator[None]:
    arrays = _input_arrays(frame, observations, dense_semantics)
    flags = tuple(array.flags.writeable for array in arrays)
    try:
        for array in arrays:
            array.flags.writeable = False
        yield
    finally:
        for array, writeable in zip(reversed(arrays), reversed(flags)):
            array.flags.writeable = writeable


__all__ = [
    "DualTransactionSnapshot", "cumulative_state_sha256", "frozen_shared_inputs",
    "shared_input_sha256", "temporal_state_sha256",
]
