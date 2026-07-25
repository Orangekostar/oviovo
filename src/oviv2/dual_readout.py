from __future__ import annotations

from dataclasses import dataclass

from src.core.data_structures import Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import FrameObservation
from src.oviv2.runtime import Oviv2Runtime, RuntimeFrameResult
from src.oviv2.temporal_runtime import TemporalCurrentRuntime, TemporalFrameResult


@dataclass(frozen=True)
class DualFrameResult:
    cumulative: RuntimeFrameResult
    temporal: TemporalFrameResult

    def __post_init__(self) -> None:
        if not isinstance(self.cumulative, RuntimeFrameResult):
            raise TypeError("cumulative must be a RuntimeFrameResult")
        if not isinstance(self.temporal, TemporalFrameResult):
            raise TypeError("temporal must be a TemporalFrameResult")


class DualReadoutRuntime:
    def __init__(
        self,
        cumulative: Oviv2Runtime,
        temporal: TemporalCurrentRuntime,
    ) -> None:
        if not isinstance(cumulative, Oviv2Runtime):
            raise TypeError("cumulative must be an Oviv2Runtime")
        if not isinstance(temporal, TemporalCurrentRuntime):
            raise TypeError("temporal must be a TemporalCurrentRuntime")
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

    @staticmethod
    def _restore(
        cumulative: Oviv2Runtime,
        temporal: TemporalCurrentRuntime,
        cumulative_snapshot: dict[str, object],
        temporal_snapshot: dict[str, object],
        original_error: BaseException,
    ) -> None:
        failures: list[tuple[str, BaseException]] = []
        for name, runtime, snapshot in (
            ("cumulative", cumulative, cumulative_snapshot),
            ("temporal", temporal, temporal_snapshot),
        ):
            try:
                runtime.__dict__.clear()
                runtime.__dict__.update(snapshot)
            except BaseException as error:
                failures.append((name, error))
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
        temporal_snapshot = dict(self.temporal.__dict__)

        cumulative_result = self.cumulative.process_frame(
            frame,
            observations,
            dense_semantics,
        )
        try:
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
