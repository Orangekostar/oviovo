from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import struct

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.runtime import Oviv2Runtime, RuntimeFrameResult
from src.oviv2.reference_readout import (
    CumulativeReadoutView,
    LifecycleOverlayReadout,
    ReferenceFrameResult,
    ReferenceCurrentReadout,
    ReferenceReadoutState,
)
from src.oviv2.temporal_runtime import (
    TemporalCurrentRuntime,
    TemporalFrameResult,
    build_proposal_recovery_evidence,
)
from src.oviv2.temporal_config import temporal_config_to_json
from src.oviv2.temporal_snapshot import TemporalCurrentSnapshot, build_temporal_snapshot
from src.oviv2.temporal_export import TemporalExportBatch, timestamp_seconds_to_ns
from src.oviv2.t1_exactness import (
    DualTransactionSnapshot,
    clone_shared_inputs,
    commit_runtime_state,
    frozen_shared_inputs,
    isolated_cumulative_runtime,
    isolated_temporal_runtime,
    shared_input_sha256,
    shared_input_snapshot,
)


ReferenceReadout = ReferenceCurrentReadout | LifecycleOverlayReadout


_STRUCTURAL_WITNESS_OBJECT_TYPES = {
    CameraIntrinsics,
    DenseSemanticFrame,
    Frame,
    FrameObservation,
}


def _supports_structural_witness(value: object) -> bool:
    value_type = type(value)
    if value is None or value_type in {bool, int, float, str, bytes}:
        return True
    if value_type is np.ndarray:
        return not value.dtype.hasobject
    if value_type is ObservationKind:
        return True
    if value_type is tuple:
        return all(_supports_structural_witness(item) for item in value)
    if value_type is frozenset:
        return all(
            type(item) is tuple
            and len(item) == 3
            and all(type(coordinate) is int for coordinate in item)
            for item in frozenset.__iter__(value)
        )
    if value_type in _STRUCTURAL_WITNESS_OBJECT_TYPES:
        expected = tuple(field.name for field in fields(value))
        attributes = vars(value)
        return tuple(attributes) == expected and all(
            _supports_structural_witness(attributes[name]) for name in expected
        )
    return False


def _structural_witness_equal(left: object, right: object) -> bool:
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    value_type = type(left)
    if left is None or value_type in {bool, int, str, bytes}:
        return left == right
    if value_type is float:
        return struct.pack(">d", left) == struct.pack(">d", right)
    if value_type is np.ndarray:
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.strides == right.strides
            and bool(left.flags.writeable) is bool(right.flags.writeable)
            and left.tobytes(order="A") == right.tobytes(order="A")
        )
    if value_type is ObservationKind:
        return left is right
    if value_type is tuple:
        assert isinstance(left, tuple) and isinstance(right, tuple)
        return len(left) == len(right) and all(
            _structural_witness_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    if value_type is frozenset:
        return frozenset.__eq__(left, right) is True
    if value_type in _STRUCTURAL_WITNESS_OBJECT_TYPES:
        left_attributes = vars(left)
        right_attributes = vars(right)
        return tuple(left_attributes) == tuple(right_attributes) and all(
            _structural_witness_equal(
                left_attributes[name], right_attributes[name]
            )
            for name in left_attributes
        )
    return False


def _shared_inputs_support_structural_witness(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None,
) -> bool:
    return _supports_structural_witness((frame, observations, dense_semantics))


def _shared_inputs_structurally_equal(
    expected: tuple[
        Frame,
        tuple[FrameObservation, ...],
        DenseSemanticFrame | None,
    ],
    observed: tuple[
        Frame,
        tuple[FrameObservation, ...],
        DenseSemanticFrame | None,
    ],
) -> bool:
    return _structural_witness_equal(expected, observed)


def _exact_value_signature(value: object) -> tuple[object, ...]:
    value_type = type(value)
    if value is None or value_type in {bool, int, str, bytes}:
        return ("scalar", value_type, value)
    if value_type is float:
        return ("float", value_type, struct.pack(">d", value))
    if isinstance(value, np.generic):
        return ("numpy_scalar", value_type, value.dtype.str, value.tobytes())
    if isinstance(value, np.ndarray):
        return (
            "array",
            value_type,
            value.dtype.str,
            value.shape,
            value.strides,
            bool(value.flags.writeable),
            hashlib.sha256(value.tobytes(order="A")).digest(),
        )
    if isinstance(value, Enum):
        return ("enum", value_type, _exact_value_signature(value.value))
    if isinstance(value, Mapping):
        items = tuple(
            sorted(
                (
                    (_exact_value_signature(key), _exact_value_signature(item))
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
        return ("mapping", value_type, items)
    if isinstance(value, tuple):
        return (
            "tuple",
            value_type,
            tuple(_exact_value_signature(item) for item in tuple.__iter__(value)),
        )
    if isinstance(value, list):
        return (
            "list",
            value_type,
            tuple(_exact_value_signature(item) for item in list.__iter__(value)),
        )
    if isinstance(value, (set, frozenset)):
        iterator = (
            set.__iter__(value) if isinstance(value, set) else frozenset.__iter__(value)
        )
        return (
            "set",
            value_type,
            tuple(sorted((_exact_value_signature(item) for item in iterator), key=repr)),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            value_type,
            tuple(
                (field.name, _exact_value_signature(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    return ("identity", value_type, id(value))


def _shared_input_exact_signature(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None,
) -> tuple[object, ...]:
    return _exact_value_signature((frame, observations, dense_semantics))


def _is_reference_readout(value: object) -> bool:
    return type(value) in {ReferenceCurrentReadout, LifecycleOverlayReadout}


def _validate_reference_readout(value: object) -> ReferenceReadout:
    if not _is_reference_readout(value):
        raise TypeError("reference readout must be an exact built-in readout")
    if set(value.__dict__) != {"config", "state"}:
        raise ValueError("reference readout has unexpected mutable state")
    return value


@dataclass(frozen=True)
class DualFrameResult:
    cumulative: RuntimeFrameResult
    temporal: TemporalFrameResult | ReferenceFrameResult
    export: TemporalExportBatch | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cumulative, RuntimeFrameResult):
            raise TypeError("cumulative must be a RuntimeFrameResult")
        if not isinstance(self.temporal, (TemporalFrameResult, ReferenceFrameResult)):
            raise TypeError("temporal must be a temporal or reference frame result")
        expected = self.temporal.export
        if self.export is None:
            object.__setattr__(self, "export", expected)
        elif type(self.export) is not TemporalExportBatch:
            raise TypeError("export must be a TemporalExportBatch")
        elif self.export != expected:
            raise ValueError("export must be the temporal causal export batch")


class DualReadoutRuntime:
    def __init__(
        self,
        cumulative: Oviv2Runtime,
        temporal: TemporalCurrentRuntime | ReferenceReadout,
    ) -> None:
        isolated_cumulative_runtime(cumulative)
        isolated_temporal_runtime(temporal)
        reference = (
            _validate_reference_readout(temporal)
            if _is_reference_readout(temporal)
            else None
        )
        temporal_state = temporal.state
        if cumulative.scene_id != temporal_state.scene_id:
            raise ValueError("cumulative and temporal scene IDs must match")
        if cumulative.revision != temporal_state.revision:
            raise ValueError("cumulative and temporal revision progress must match")
        if cumulative.last_frame_id != temporal_state.last_frame_id:
            raise ValueError("cumulative and temporal frame progress must match")
        if (
            cumulative.last_frame_id >= 0
            and cumulative.last_timestamp != temporal_state.last_timestamp
        ):
            raise ValueError("cumulative and temporal timestamp progress must match")
        self.cumulative = cumulative
        self.temporal = temporal
        if reference is not None:
            reference.bind_initial_cumulative_view(
                CumulativeReadoutView.capture(cumulative)
            )

    def current_checkpoint(self) -> TemporalCurrentSnapshot | CumulativeReadoutView:
        if _is_reference_readout(self.temporal):
            view = self.temporal.state.cumulative_view
            if view is None:
                raise RuntimeError("reference readout has no cumulative checkpoint")
            return view
        config_bytes = (
            json.dumps(
                temporal_config_to_json(self.temporal.config),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        return build_temporal_snapshot(
            self.temporal.state,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        )

    def process_temporal_only_frame(
        self,
        frame: Frame,
        observations: tuple[FrameObservation, ...],
        dense_semantics: DenseSemanticFrame | None = None,
    ) -> TemporalFrameResult:
        if type(self.temporal) is not TemporalCurrentRuntime:
            raise TypeError("temporal-only processing requires TemporalCurrentRuntime")
        temporal_trial = isolated_temporal_runtime(self.temporal)
        original_snapshot = shared_input_snapshot(
            frame, observations, dense_semantics
        )
        original_exact = _shared_input_exact_signature(
            frame, observations, dense_semantics
        )
        temporal_frame: Frame | None = None
        isolated_temporal_observations: tuple[FrameObservation, ...] | None = None
        temporal_dense: DenseSemanticFrame | None = None
        temporal_clone_exact: tuple[object, ...] | None = None
        try:
            (
                temporal_frame,
                isolated_temporal_observations,
                temporal_dense,
            ) = clone_shared_inputs(frame, observations, dense_semantics)
            temporal_clone_exact = _shared_input_exact_signature(
                temporal_frame, isolated_temporal_observations, temporal_dense
            )
            if temporal_clone_exact != original_exact:
                raise ValueError("shared input clone is not equivalent")
            original_snapshot.assert_unchanged()
            with frozen_shared_inputs(
                temporal_frame, isolated_temporal_observations, temporal_dense
            ):
                temporal_snapshot = shared_input_snapshot(
                    temporal_frame, isolated_temporal_observations, temporal_dense
                )
                temporal_trial_exact = _shared_input_exact_signature(
                    temporal_frame, isolated_temporal_observations, temporal_dense
                )
                try:
                    proposal_evidence = (
                        build_proposal_recovery_evidence(
                            temporal_frame,
                            isolated_temporal_observations,
                            temporal_dense,
                            temporal_trial.config,
                            temporal_trial.state,
                        )
                        if temporal_trial.config.proposal_recovery_enabled
                        and temporal_dense is not None
                        else None
                    )
                    if temporal_dense is None:
                        result = temporal_trial.process_frame(
                            temporal_frame,
                            isolated_temporal_observations,
                            temporal_dense,
                        )
                    else:
                        result = temporal_trial.process_frame(
                            temporal_frame,
                            isolated_temporal_observations,
                            temporal_dense,
                            proposal_evidence=proposal_evidence,
                        )
                finally:
                    temporal_snapshot.assert_unchanged()
                    if _shared_input_exact_signature(
                        temporal_frame, isolated_temporal_observations, temporal_dense
                    ) != temporal_trial_exact:
                        raise RuntimeError(
                            "temporal branch mutated isolated shared inputs"
                        )
        finally:
            caller_changed = False
            try:
                original_snapshot.assert_unchanged()
                caller_changed = (
                    _shared_input_exact_signature(
                        frame, observations, dense_semantics
                    )
                    != original_exact
                )
            except BaseException:
                caller_changed = True
            clone_changed = False
            if (
                temporal_frame is not None
                and isolated_temporal_observations is not None
                and temporal_clone_exact is not None
            ):
                try:
                    clone_changed = (
                        _shared_input_exact_signature(
                            temporal_frame,
                            isolated_temporal_observations,
                            temporal_dense,
                        )
                        != temporal_clone_exact
                    )
                except BaseException:
                    clone_changed = True
            if caller_changed:
                raise RuntimeError(
                    "original caller inputs changed during temporal-only processing"
                )
            if clone_changed:
                raise RuntimeError("temporal branch mutated isolated shared inputs")
        commit_runtime_state(self.temporal, temporal_trial)
        return result

    def advance_temporal_only_frame(self, frame: Frame) -> TemporalFrameResult:
        if type(self.temporal) is not TemporalCurrentRuntime:
            raise TypeError("temporal-only advance requires TemporalCurrentRuntime")
        return self.temporal.advance_frame_without_update(frame)

    @staticmethod
    def _restore(
        cumulative: Oviv2Runtime,
        temporal: TemporalCurrentRuntime | ReferenceReadout,
        snapshot: DualTransactionSnapshot,
        original_error: BaseException,
    ) -> None:
        failures: list[tuple[str, BaseException]] = []
        try:
            snapshot.restore(cumulative, temporal)
        except BaseException as error:
            failures.append(("transaction", error))
        if failures:
            detail = ", ".join(
                f"{name}: {type(error).__name__}: {error}"
                for name, error in failures
            )
            raise RuntimeError(f"dual readout rollback failed ({detail})") from original_error

    def process_frame(
        self,
        frame: Frame,
        observations: tuple[FrameObservation, ...],
        dense_semantics: DenseSemanticFrame | None = None,
        *,
        temporal_observations: tuple[FrameObservation, ...] | None = None,
    ) -> DualFrameResult:
        cumulative_trial = isolated_cumulative_runtime(self.cumulative)
        temporal_trial = isolated_temporal_runtime(self.temporal)
        transaction = DualTransactionSnapshot.capture(
            self.cumulative, self.temporal, frame, observations, dense_semantics
        )
        original_inputs = (frame, observations, dense_semantics)
        temporal_inputs = (
            original_inputs
            if temporal_observations is None
            else (frame, temporal_observations, dense_semantics)
        )
        use_structural_witness = (
            _shared_inputs_support_structural_witness(*original_inputs)
            and _shared_inputs_support_structural_witness(*temporal_inputs)
        )
        original_input_sha256 = (
            None
            if use_structural_witness
            else shared_input_sha256(*original_inputs)
        )
        original_input_exact = (
            None
            if use_structural_witness
            else _shared_input_exact_signature(*original_inputs)
        )
        original_input_snapshot = shared_input_snapshot(
            frame, observations, dense_semantics
        )
        temporal_input_sha256 = (
            original_input_sha256
            if temporal_observations is None
            else (
                None
                if use_structural_witness
                else shared_input_sha256(*temporal_inputs)
            )
        )
        temporal_input_exact = (
            original_input_exact
            if temporal_observations is None
            else (
                None
                if use_structural_witness
                else _shared_input_exact_signature(*temporal_inputs)
            )
        )
        temporal_original_snapshot = (
            None
            if temporal_observations is None
            else shared_input_snapshot(
                frame, temporal_observations, dense_semantics
            )
        )

        original_validation_attempted = False
        original_validation_error: BaseException | None = None

        def assert_original_inputs_unchanged() -> None:
            nonlocal original_validation_attempted, original_validation_error
            if original_validation_attempted:
                if original_validation_error is not None:
                    raise original_validation_error
                return
            original_validation_attempted = True
            failures: list[BaseException] = []
            try:
                original_input_snapshot.assert_unchanged()
            except BaseException as error:
                failures.append(error)
            if not use_structural_witness:
                try:
                    if (
                        shared_input_sha256(*original_inputs)
                        != original_input_sha256
                        or _shared_input_exact_signature(*original_inputs)
                        != original_input_exact
                    ):
                        failures.append(
                            RuntimeError("original shared inputs changed during cloning")
                        )
                except BaseException as error:
                    failures.append(error)
            if temporal_original_snapshot is not None:
                assert temporal_observations is not None
                try:
                    temporal_original_snapshot.assert_unchanged()
                except BaseException as error:
                    failures.append(error)
                if not use_structural_witness:
                    try:
                        if (
                            shared_input_sha256(*temporal_inputs)
                            != temporal_input_sha256
                            or _shared_input_exact_signature(*temporal_inputs)
                            != temporal_input_exact
                        ):
                            failures.append(
                                RuntimeError(
                                    "original temporal inputs changed during cloning"
                                )
                            )
                    except BaseException as error:
                        failures.append(error)
            if failures:
                original_validation_error = RuntimeError(
                    "original caller inputs changed during cloning"
                )
                raise original_validation_error from failures[0]

        def assert_equivalent_clone(
            expected_inputs: tuple[
                Frame,
                tuple[FrameObservation, ...],
                DenseSemanticFrame | None,
            ],
            expected_sha256: str | None,
            expected_exact: tuple[object, ...] | None,
            cloned_frame: Frame,
            cloned_observations: tuple[FrameObservation, ...],
            cloned_dense: DenseSemanticFrame | None,
        ) -> None:
            if use_structural_witness:
                if not _shared_inputs_structurally_equal(
                    expected_inputs,
                    (cloned_frame, cloned_observations, cloned_dense),
                ):
                    raise ValueError("shared input clone is not equivalent")
                return
            assert expected_sha256 is not None and expected_exact is not None
            try:
                observed_sha256 = shared_input_sha256(
                    cloned_frame, cloned_observations, cloned_dense
                )
                observed_exact = _shared_input_exact_signature(
                    cloned_frame, cloned_observations, cloned_dense
                )
            except (TypeError, ValueError) as error:
                raise ValueError("shared input clone is not equivalent") from error
            if (
                observed_sha256 != expected_sha256
                or observed_exact != expected_exact
            ):
                raise ValueError("shared input clone is not equivalent")

        try:
            reference = (
                _validate_reference_readout(temporal_trial)
                if _is_reference_readout(temporal_trial)
                else None
            )
            if (
                reference is not None
                and temporal_observations is not None
                and (
                    not _shared_inputs_structurally_equal(
                        original_inputs, temporal_inputs
                    )
                    if use_structural_witness
                    else (
                        temporal_input_sha256 != original_input_sha256
                        or temporal_input_exact != original_input_exact
                    )
                )
            ):
                raise ValueError(
                    "reference readout cannot consume different temporal observations"
                )
            isolated_frame, isolated_observations, isolated_dense = (
                clone_shared_inputs(frame, observations, dense_semantics)
            )
            assert_equivalent_clone(
                original_inputs,
                original_input_sha256,
                original_input_exact,
                isolated_frame,
                isolated_observations,
                isolated_dense,
            )
            if temporal_observations is not None and reference is None:
                (
                    temporal_frame,
                    isolated_temporal_observations,
                    temporal_dense,
                ) = clone_shared_inputs(
                    frame, temporal_observations, dense_semantics
                )
                assert_equivalent_clone(
                    temporal_inputs,
                    temporal_input_sha256,
                    temporal_input_exact,
                    temporal_frame,
                    isolated_temporal_observations,
                    temporal_dense,
                )
                temporal_freeze = frozen_shared_inputs(
                    temporal_frame,
                    isolated_temporal_observations,
                    temporal_dense,
                )
            else:
                temporal_frame = isolated_frame
                isolated_temporal_observations = isolated_observations
                temporal_dense = isolated_dense
                temporal_freeze = nullcontext()
            with frozen_shared_inputs(
                isolated_frame, isolated_observations, isolated_dense
            ), temporal_freeze:
                input_snapshot = shared_input_snapshot(
                    isolated_frame, isolated_observations, isolated_dense
                )
                temporal_input_snapshot = (
                    input_snapshot
                    if temporal_frame is isolated_frame
                    else shared_input_snapshot(
                        temporal_frame,
                        isolated_temporal_observations,
                        temporal_dense,
                    )
                )
                isolated_trial_exact = (
                    None
                    if use_structural_witness
                    else _shared_input_exact_signature(
                        isolated_frame, isolated_observations, isolated_dense
                    )
                )
                temporal_trial_exact = (
                    isolated_trial_exact
                    if temporal_frame is isolated_frame
                    else (
                        None
                        if use_structural_witness
                        else _shared_input_exact_signature(
                            temporal_frame,
                            isolated_temporal_observations,
                            temporal_dense,
                        )
                    )
                )
                try:
                    cumulative_result = cumulative_trial.process_frame(
                        isolated_frame,
                        isolated_observations,
                        isolated_dense,
                    )
                    input_snapshot.assert_unchanged()
                    if reference is not None:
                        before = reference.state.cumulative_view
                        if before is None:
                            raise RuntimeError(
                                "reference readout has no previous cumulative view"
                            )
                        temporal_result = reference.process_cumulative_frame(
                            isolated_frame,
                            before=before,
                            after=CumulativeReadoutView.capture(cumulative_trial),
                            cumulative_result=cumulative_result,
                        )
                    else:
                        proposal_evidence = (
                            build_proposal_recovery_evidence(
                                temporal_frame,
                                isolated_temporal_observations,
                                temporal_dense,
                                temporal_trial.config,
                                temporal_trial.state,
                            )
                            if temporal_trial.config.proposal_recovery_enabled
                            and temporal_dense is not None
                            else None
                        )
                        temporal_result = temporal_trial.process_frame(
                            temporal_frame,
                            isolated_temporal_observations,
                            temporal_dense,
                            proposal_evidence=proposal_evidence,
                        )
                finally:
                    trial_input_failures: list[BaseException] = []
                    try:
                        input_snapshot.assert_unchanged()
                    except BaseException as error:
                        trial_input_failures.append(error)
                    if not use_structural_witness:
                        try:
                            if _shared_input_exact_signature(
                                isolated_frame, isolated_observations, isolated_dense
                            ) != isolated_trial_exact:
                                trial_input_failures.append(
                                    RuntimeError(
                                        "cumulative branch mutated isolated shared inputs"
                                    )
                                )
                        except BaseException as error:
                            trial_input_failures.append(error)
                    if temporal_input_snapshot is not input_snapshot:
                        try:
                            temporal_input_snapshot.assert_unchanged()
                        except BaseException as error:
                            trial_input_failures.append(error)
                        if not use_structural_witness:
                            try:
                                if _shared_input_exact_signature(
                                    temporal_frame,
                                    isolated_temporal_observations,
                                    temporal_dense,
                                ) != temporal_trial_exact:
                                    trial_input_failures.append(
                                        RuntimeError(
                                            "temporal branch mutated isolated shared inputs"
                                        )
                                    )
                            except BaseException as error:
                                trial_input_failures.append(error)
                    if trial_input_failures:
                        raise RuntimeError(
                            "dual readout branch mutated isolated shared inputs"
                        ) from trial_input_failures[0]
            temporal_state = temporal_trial.state
            expected_progress = (
                cumulative_result.frame_id,
                cumulative_result.revision,
            )
            if (
                temporal_state.scene_id != cumulative_trial.scene_id
                or (temporal_result.frame_id, temporal_result.revision)
                != expected_progress
                or (cumulative_trial.last_frame_id, cumulative_trial.revision)
                != expected_progress
                or (temporal_state.last_frame_id, temporal_state.revision)
                != expected_progress
                or cumulative_trial.last_timestamp != temporal_state.last_timestamp
                or temporal_result.export.timestamp_ns
                != timestamp_seconds_to_ns(frame.timestamp)
            ):
                raise RuntimeError("dual readout frame/revision mismatch")
            assert_original_inputs_unchanged()
            commit_runtime_state(self.cumulative, cumulative_trial)
            commit_runtime_state(self.temporal, temporal_trial)
            return DualFrameResult(cumulative_result, temporal_result, temporal_result.export)
        except BaseException as error:
            try:
                assert_original_inputs_unchanged()
            except BaseException as mutation_error:
                self._restore(
                    self.cumulative, self.temporal, transaction, mutation_error
                )
                raise RuntimeError("original shared inputs were mutated") from error
            self._restore(
                self.cumulative,
                self.temporal,
                transaction,
                error,
            )
            raise
