from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
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
from src.oviv2.reference_readout import (
    LifecycleOverlayReadout,
    ReferenceCurrentReadout,
    ReferenceReadoutState,
)
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
_TRUSTED_REFERENCE_METHODS = {
    ReferenceCurrentReadout: ReferenceCurrentReadout.process_cumulative_frame,
    LifecycleOverlayReadout: LifecycleOverlayReadout.process_cumulative_frame,
}
_TRUSTED_CUMULATIVE_METHOD = Oviv2Runtime.process_frame
_TRUSTED_TEMPORAL_METHOD = TemporalCurrentRuntime.process_frame


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
    active = volume._grid.hashmap().active_buf_indices()
    capacity = max(1, int(active.shape[0]))
    config = replace(volume.config, block_count=capacity)
    clone = SparseTsdfVolume(config)
    if int(active.shape[0]):
        keys = volume._grid.hashmap().key_tensor()[active]
        destination, activated = clone._grid.hashmap().activate(keys)
        if not np.all(activated.numpy()):
            raise RuntimeError("failed to snapshot cumulative TSDF blocks")
        for name in volume._ATTRIBUTE_NAMES:
            clone._grid.attribute(name)[destination] = volume._grid.attribute(name)[active]
    clone._config = volume.config
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


@dataclass(frozen=True)
class _ArrayNodeSnapshot:
    array: np.ndarray
    content: bytes
    dtype: np.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    writeable: bool

    @classmethod
    def capture(cls, array: np.ndarray) -> "_ArrayNodeSnapshot":
        return cls(
            array,
            array.tobytes(order="A"),
            array.dtype,
            array.shape,
            array.strides,
            bool(array.flags.writeable),
        )

    def assert_unchanged(self) -> None:
        if (
            self.array.dtype != self.dtype
            or self.array.shape != self.shape
            or self.array.strides != self.strides
            or bool(self.array.flags.writeable) is not self.writeable
            or self.array.tobytes(order="A") != self.content
        ):
            raise RuntimeError("branch mutated isolated ndarray state")


@dataclass(frozen=True)
class _ArraySnapshot:
    array: np.ndarray
    chain: tuple[_ArrayNodeSnapshot, ...]

    @classmethod
    def capture(cls, array: np.ndarray) -> "_ArraySnapshot":
        chain: list[_ArrayNodeSnapshot] = []
        current: object = array
        seen: set[int] = set()
        while isinstance(current, np.ndarray) and id(current) not in seen:
            seen.add(id(current))
            chain.append(_ArrayNodeSnapshot.capture(current))
            current = current.base
        return cls(array, tuple(chain))

    def assert_unchanged(self) -> None:
        current: object = self.array
        for snapshot in self.chain:
            if current is not snapshot.array:
                raise RuntimeError("branch replaced isolated ndarray base chain")
            snapshot.assert_unchanged()
            current = current.base
        if isinstance(current, np.ndarray):
            raise RuntimeError("branch extended isolated ndarray base chain")


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

    def assert_unchanged(self) -> None:
        expected = {name for name, _, _ in self.fields}
        if not hasattr(self.target, "__dict__") or set(vars(self.target)) != expected:
            raise RuntimeError("branch changed isolated input object fields")
        for name, value, array in self.fields:
            current = getattr(self.target, name)
            if array is not None:
                if current is not value:
                    raise RuntimeError("branch replaced isolated input array")
                array.assert_unchanged()
            elif isinstance(value, CameraIntrinsics):
                if current is not value:
                    raise RuntimeError("branch replaced isolated camera intrinsics")
            elif type(current) is not type(value) or current != value:
                raise RuntimeError("branch mutated isolated input field")


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

    def assert_unchanged(self) -> None:
        for snapshot in self.objects:
            snapshot.assert_unchanged()


def clone_shared_inputs(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None,
) -> tuple[Frame, tuple[FrameObservation, ...], DenseSemanticFrame | None]:
    arrays: dict[int, np.ndarray] = {}

    def clone_array(array: np.ndarray) -> np.ndarray:
        cached = arrays.get(id(array))
        if cached is not None:
            return cached
        if isinstance(array.base, np.ndarray):
            base = clone_array(array.base)
            offset = int(array.ctypes.data) - int(array.base.ctypes.data)
            cloned = np.ndarray(
                array.shape,
                dtype=array.dtype,
                buffer=base,
                offset=offset,
                strides=array.strides,
            )
        else:
            cloned = np.array(array, copy=True, order="K")
        cloned.flags.writeable = bool(array.flags.writeable)
        arrays[id(array)] = cloned
        return cloned

    def clone_object(value: object) -> object:
        cloned = copy.copy(value)
        for field in fields(value):
            item = getattr(value, field.name)
            owned = clone_array(item) if isinstance(item, np.ndarray) else copy.deepcopy(item)
            object.__setattr__(cloned, field.name, owned)
        return cloned

    cloned_frame = clone_object(frame)
    assert isinstance(cloned_frame, Frame)
    cloned_observations = tuple(clone_object(item) for item in observations)
    assert all(isinstance(item, FrameObservation) for item in cloned_observations)
    cloned_dense = None if dense_semantics is None else clone_object(dense_semantics)
    assert cloned_dense is None or isinstance(cloned_dense, DenseSemanticFrame)
    return cloned_frame, cloned_observations, cloned_dense


def shared_input_snapshot(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None,
) -> _SharedInputSnapshot:
    return _SharedInputSnapshot.capture(frame, observations, dense_semantics)


def isolated_cumulative_runtime(runtime: Oviv2Runtime) -> Oviv2Runtime:
    trial = copy.copy(runtime)
    trial.__dict__ = dict(runtime.__dict__)
    if type(runtime).process_frame is _TRUSTED_CUMULATIVE_METHOD:
        return trial
    trial.geometry = _clone_geometry(runtime.geometry)
    for name in (
        "evidence", "ownership", "tracker", "registry", "visibility",
        "dense_semantic_integrator",
    ):
        setattr(trial, name, copy.deepcopy(getattr(runtime, name)))
    return trial


def isolated_temporal_runtime(runtime: object) -> object:
    trial = copy.copy(runtime)
    trial.__dict__ = dict(runtime.__dict__)
    state = getattr(runtime, "state", None)
    if isinstance(runtime, TemporalCurrentRuntime):
        if type(runtime).process_frame is not _TRUSTED_TEMPORAL_METHOD:
            trial.state = _clone_temporal_state(state)
        return trial
    expected = _TRUSTED_REFERENCE_METHODS.get(type(runtime))
    if expected is None:
        raise TypeError("temporal runtime has an unsupported type")
    if type(runtime).process_cumulative_frame is not expected:
        trial.state = copy.deepcopy(state)
    return trial


def commit_runtime_state(target: object, trial: object) -> None:
    target.__dict__.clear()
    target.__dict__.update(trial.__dict__)


@dataclass(frozen=True)
class DualTransactionSnapshot:
    cumulative_attributes: tuple[tuple[str, object], ...]
    temporal_attributes: tuple[tuple[str, object], ...]

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
        state = getattr(temporal, "state", None)
        if not isinstance(state, (TemporalRuntimeState, ReferenceReadoutState)):
            raise TypeError("temporal runtime has an unsupported state")
        return cls(
            tuple(cumulative.__dict__.items()),
            tuple(temporal.__dict__.items()),
        )

    def restore(self, cumulative: Oviv2Runtime, temporal: object) -> None:
        cumulative.__dict__.clear()
        cumulative.__dict__.update(self.cumulative_attributes)
        temporal.__dict__.clear()
        temporal.__dict__.update(self.temporal_attributes)


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
            try:
                array.flags.writeable = writeable
            except ValueError:
                pass


__all__ = [
    "DualTransactionSnapshot", "clone_shared_inputs", "commit_runtime_state",
    "cumulative_state_sha256", "frozen_shared_inputs",
    "isolated_cumulative_runtime", "isolated_temporal_runtime",
    "shared_input_sha256", "shared_input_snapshot", "temporal_state_sha256",
]
