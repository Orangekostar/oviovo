from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from src.core.data_structures import Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import FrameObservation
from src.oviv2.runtime import Oviv2Runtime, RuntimeFrameResult
from src.oviv2.reference_readout import (
    CumulativeReadoutView,
    LifecycleOverlayReadout,
    ReferenceFrameResult,
    ReferenceCurrentReadout,
    ReferenceReadoutState,
)
from src.oviv2.temporal_runtime import TemporalCurrentRuntime, TemporalFrameResult
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
        if not isinstance(cumulative, Oviv2Runtime):
            raise TypeError("cumulative must be an Oviv2Runtime")
        if not isinstance(temporal, TemporalCurrentRuntime) and not _is_reference_readout(
            temporal
        ):
            raise TypeError("temporal must be a temporal runtime or reference readout")
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
    ) -> DualFrameResult:
        transaction = DualTransactionSnapshot.capture(
            self.cumulative, self.temporal, frame, observations, dense_semantics
        )
        shared_input_sha256(frame, observations, dense_semantics)
        isolated_frame, isolated_observations, isolated_dense = clone_shared_inputs(
            frame, observations, dense_semantics
        )
        cumulative_trial = isolated_cumulative_runtime(self.cumulative)
        temporal_trial = isolated_temporal_runtime(self.temporal)
        reference = _validate_reference_readout(temporal_trial) if _is_reference_readout(
            temporal_trial
        ) else None
        try:
            with frozen_shared_inputs(
                isolated_frame, isolated_observations, isolated_dense
            ):
                input_snapshot = shared_input_snapshot(
                    isolated_frame, isolated_observations, isolated_dense
                )
                cumulative_result = cumulative_trial.process_frame(
                    isolated_frame,
                    isolated_observations,
                    isolated_dense,
                )
                input_snapshot.assert_unchanged()
                if reference is not None:
                    before = reference.state.cumulative_view
                    if before is None:
                        raise RuntimeError("reference readout has no previous cumulative view")
                    temporal_result = reference.process_cumulative_frame(
                        isolated_frame,
                        before=before,
                        after=CumulativeReadoutView.capture(cumulative_trial),
                        cumulative_result=cumulative_result,
                    )
                else:
                    temporal_result = temporal_trial.process_frame(
                        isolated_frame,
                        isolated_observations,
                        isolated_dense,
                    )
                input_snapshot.assert_unchanged()
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
            commit_runtime_state(self.cumulative, cumulative_trial)
            commit_runtime_state(self.temporal, temporal_trial)
            return DualFrameResult(cumulative_result, temporal_result, temporal_result.export)
        except BaseException as error:
            self._restore(
                self.cumulative,
                self.temporal,
                transaction,
                error,
            )
            raise
