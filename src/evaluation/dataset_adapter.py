"""Conversions between the online mapper Frame and neutral FramePacket."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence

from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.contracts import FramePacket, GroundTruthSequence


def frame_to_packet(scene_id: str, frame: Frame) -> FramePacket:
    """Copy one runtime frame into the process-neutral evaluation contract."""
    return FramePacket(
        scene_id=scene_id,
        frame_id=int(frame.frame_id),
        timestamp=float(frame.timestamp),
        rgb=frame.rgb,
        depth_m=frame.depth,
        intrinsics=frame.intrinsics.to_matrix(),
        camera_to_world=frame.pose,
    )


def packet_to_frame(packet: FramePacket) -> Frame:
    """Convert a neutral packet back to the mapper's authoritative Frame type."""
    height, width = packet.depth_m.shape
    intrinsics = CameraIntrinsics(
        fx=float(packet.intrinsics[0, 0]),
        fy=float(packet.intrinsics[1, 1]),
        cx=float(packet.intrinsics[0, 2]),
        cy=float(packet.intrinsics[1, 2]),
        width=int(width),
        height=int(height),
    )
    return Frame(
        frame_id=int(packet.frame_id),
        rgb=packet.rgb.copy(),
        depth=packet.depth_m.copy(),
        pose=packet.camera_to_world.copy(),
        intrinsics=intrinsics,
        timestamp=float(packet.timestamp),
        source_frame_id=int(packet.frame_id),
    )


class InMemoryDatasetAdapter:
    """Small deterministic adapter used by fixtures and causal regression tests."""

    def __init__(
        self,
        frames_by_scene: Mapping[str, Iterable[FramePacket]],
        ground_truth_by_scene: Mapping[str, GroundTruthSequence] | None = None,
    ) -> None:
        self._frames: dict[str, tuple[FramePacket, ...]] = {}
        for scene_id, frames in frames_by_scene.items():
            sequence = tuple(frames)
            if any(frame.scene_id != scene_id for frame in sequence):
                raise ValueError(f"frame scene mismatch for {scene_id}")
            timestamps = [frame.timestamp for frame in sequence]
            if timestamps != sorted(timestamps):
                raise ValueError(f"timestamps for {scene_id} must be monotonically non-decreasing")
            self._frames[str(scene_id)] = sequence
        self._ground_truth = dict(ground_truth_by_scene or {})

    def scenes(self) -> Sequence[str]:
        return tuple(sorted(self._frames))

    def frames(self, scene_id: str) -> Iterator[FramePacket]:
        try:
            yield from self._frames[scene_id]
        except KeyError as error:
            raise KeyError(f"unknown scene: {scene_id}") from error

    def ground_truth(self, scene_id: str) -> GroundTruthSequence:
        try:
            return self._ground_truth[scene_id]
        except KeyError as error:
            raise KeyError(f"ground truth unavailable for scene: {scene_id}") from error
