from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.contracts import FramePacket, RunMetadata
from src.evaluation.dataset_adapter import InMemoryDatasetAdapter, frame_to_packet, packet_to_frame


def _frame(frame_id: int, timestamp: float) -> Frame:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, -2.0, 0.5]
    return Frame(
        frame_id=frame_id,
        rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        depth=np.full((3, 4), 2.0, dtype=np.float32),
        pose=pose,
        intrinsics=CameraIntrinsics(4.0, 4.0, 1.5, 1.0, 4, 3),
        timestamp=timestamp,
    )


def test_frame_packet_round_trip_preserves_camera_to_world_convention() -> None:
    source = _frame(frame_id=7, timestamp=1.25)
    packet = frame_to_packet("scene-a", source)
    restored = packet_to_frame(packet)

    assert packet.scene_id == "scene-a"
    assert np.array_equal(restored.rgb, source.rgb)
    assert np.array_equal(restored.depth, source.depth)
    assert np.allclose(restored.pose, source.pose)
    assert np.allclose(restored.intrinsics.to_matrix(), source.intrinsics.to_matrix())

    camera_point = np.array([0.2, -0.1, 2.0, 1.0])
    world_point = packet.camera_to_world @ camera_point
    recovered = np.linalg.inv(packet.camera_to_world) @ world_point
    assert np.allclose(recovered, camera_point)


def test_runtime_frame_contract_cannot_carry_ground_truth_fields() -> None:
    names = {item.name.lower() for item in fields(FramePacket)}
    forbidden = ("ground_truth", "gt_", "semantic", "instance", "change", "visibility")
    assert not any(any(marker in name for marker in forbidden) for name in names)


def test_frame_packet_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="depth_m shape"):
        FramePacket(
            scene_id="scene-a",
            frame_id=0,
            timestamp=0.0,
            rgb=np.zeros((3, 4, 3), dtype=np.uint8),
            depth_m=np.ones((2, 4), dtype=np.float32),
            intrinsics=np.eye(3),
            camera_to_world=np.eye(4),
        )


def test_in_memory_adapter_enforces_monotonic_timestamps() -> None:
    packets = [
        frame_to_packet("scene-a", _frame(0, 2.0)),
        frame_to_packet("scene-a", _frame(1, 1.0)),
    ]
    with pytest.raises(ValueError, match="monotonically"):
        InMemoryDatasetAdapter({"scene-a": packets})


def test_composed_and_offline_run_metadata_are_explicit() -> None:
    with pytest.raises(ValueError, match="composed method label"):
        RunMetadata(method="Khronos", integration="composed", execution_mode="online")

    composed = RunMetadata(
        method="Khronos + shared text head",
        integration="composed",
        execution_mode="online",
    )
    offline = RunMetadata(
        method="ReScene4D (offline)",
        integration="native",
        execution_mode="offline",
    )
    assert composed.eligible_for_online_latency is True
    assert offline.eligible_for_online_latency is False
