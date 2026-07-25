from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if not isinstance(self.cumulative, RuntimeFrameResult):
            raise TypeError("cumulative must be a RuntimeFrameResult")
        if not isinstance(self.temporal, (TemporalFrameResult, ReferenceFrameResult)):
            raise TypeError("temporal must be a temporal or reference frame result")


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

    @staticmethod
    def _restore(
        cumulative: Oviv2Runtime,
        temporal: TemporalCurrentRuntime | ReferenceReadout,
        cumulative_snapshot: dict[str, object],
        temporal_snapshot: dict[str, object] | ReferenceReadoutState,
        original_error: BaseException,
    ) -> None:
        failures: list[tuple[str, BaseException]] = []
        try:
            cumulative.__dict__.clear()
            cumulative.__dict__.update(cumulative_snapshot)
        except BaseException as error:
            failures.append(("cumulative", error))
        try:
            if _is_reference_readout(temporal):
                assert isinstance(temporal_snapshot, ReferenceReadoutState)
                temporal.restore_transaction(temporal_snapshot)
            else:
                assert isinstance(temporal_snapshot, dict)
                temporal.__dict__.clear()
                temporal.__dict__.update(temporal_snapshot)
        except BaseException as error:
            failures.append(("temporal", error))
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
        cumulative_snapshot = dict(self.cumulative.__dict__)
        reference = (
            _validate_reference_readout(self.temporal)
            if _is_reference_readout(self.temporal)
            else None
        )
        temporal_snapshot = (
            reference.transaction_snapshot()
            if reference is not None
            else dict(self.temporal.__dict__)
        )

        try:
            cumulative_result = self.cumulative.process_frame(
                frame,
                observations,
                dense_semantics,
            )
            if reference is not None:
                before = reference.state.cumulative_view
                if before is None:
                    raise RuntimeError("reference readout has no previous cumulative view")
                temporal_result = reference.process_cumulative_frame(
                    frame,
                    before=before,
                    after=CumulativeReadoutView.capture(self.cumulative),
                    cumulative_result=cumulative_result,
                )
            else:
                temporal_result = self.temporal.process_frame(
                    frame,
                    observations,
                    dense_semantics,
                )
            temporal_state = self.temporal.state
            expected_progress = (
                cumulative_result.frame_id,
                cumulative_result.revision,
            )
            if (
                temporal_state.scene_id != self.cumulative.scene_id
                or (temporal_result.frame_id, temporal_result.revision)
                != expected_progress
                or (self.cumulative.last_frame_id, self.cumulative.revision)
                != expected_progress
                or (temporal_state.last_frame_id, temporal_state.revision)
                != expected_progress
                or self.cumulative.last_timestamp != temporal_state.last_timestamp
            ):
                raise RuntimeError("dual readout frame/revision mismatch")
            return DualFrameResult(cumulative_result, temporal_result)
        except BaseException as error:
            self._restore(
                self.cumulative,
                self.temporal,
                cumulative_snapshot,
                temporal_snapshot,
                error,
            )
            raise
