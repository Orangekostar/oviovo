from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import CameraIntrinsics, Frame
from src.pipelines.proposal_bundle import FrameProposalBundle
from src.pipelines.proposal_prefetch import FrameProposalPrefetcher, ProposalPrefetchStats


def _frame(frame_id: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        source_frame_id=frame_id * 10,
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=4, fy=4, cx=2, cy=2, width=4, height=4),
    )


def _bundle(frame: Frame) -> FrameProposalBundle:
    return FrameProposalBundle(
        frame_id=frame.frame_id,
        source_frame_id=frame.source_frame_id,
        source_proposals=[],
        raw_proposals=[],
        anchors=[],
        anchor_assignments=[],
        proposal_source="test_prefetch",
        actual_backend="precomputed",
    )


def test_prefetch_stats_as_dict_uses_required_keys_and_conversions() -> None:
    stats = ProposalPrefetchStats(
        submitted_count=np.int64(1),
        hit_count=np.int64(2),
        miss_count=np.int64(3),
        exception_count=np.int64(4),
        wait_sec_total=np.float32(1.5),
        last_exception=RuntimeError("bad"),
    )

    assert stats.as_dict() == {
        "submitted_count": 1,
        "hit_count": 2,
        "miss_count": 3,
        "exception_count": 4,
        "wait_sec_total": 1.5,
        "last_exception": "bad",
    }
    assert isinstance(stats.as_dict()["submitted_count"], int)
    assert isinstance(stats.as_dict()["wait_sec_total"], float)
    assert isinstance(stats.as_dict()["last_exception"], str)


def test_prefetcher_returns_matching_ready_bundle() -> None:
    prefetcher = FrameProposalPrefetcher(_bundle, wait_timeout_sec=-1.0)
    try:
        frame = _frame(3)
        assert prefetcher.submit(frame) is True
        bundle = prefetcher.result_for(frame.frame_id)

        assert bundle is not None
        assert bundle.frame_id == 3
        stats = prefetcher.snapshot_stats()
        assert stats["submitted_count"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 0
        assert stats["exception_count"] == 0
    finally:
        prefetcher.close()


def test_prefetcher_disabled_does_not_submit_or_return_results() -> None:
    prefetcher = FrameProposalPrefetcher(_bundle, enabled=False)
    try:
        assert prefetcher.submit(_frame(8)) is False
        assert prefetcher.result_for(8) is None
        stats = prefetcher.snapshot_stats()
        assert stats["submitted_count"] == 0
        assert stats["hit_count"] == 0
    finally:
        prefetcher.close()


def test_prefetcher_rejects_mismatched_frame_id_without_consuming_future() -> None:
    prefetcher = FrameProposalPrefetcher(_bundle, wait_timeout_sec=-1.0)
    try:
        assert prefetcher.submit(_frame(4)) is True
        assert prefetcher.result_for(5) is None
        stats = prefetcher.snapshot_stats()
        assert stats["submitted_count"] == 1
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 1

        bundle = prefetcher.result_for(4)
        assert bundle is not None
        assert bundle.frame_id == 4
    finally:
        prefetcher.close()


def test_prefetcher_timeout_abandons_pending_future_and_allows_new_submit() -> None:
    ready = threading.Event()

    def _blocking_builder(frame: Frame) -> FrameProposalBundle:
        ready.wait(timeout=2.0)
        return _bundle(frame)

    prefetcher = FrameProposalPrefetcher(_blocking_builder, wait_timeout_sec=0.0)
    try:
        frame = _frame(13)
        assert prefetcher.submit(frame) is True
        assert prefetcher.result_for(13) is None
        stats = prefetcher.snapshot_stats()
        assert stats["miss_count"] == 1
        assert "timed out" in stats["last_exception"]

        assert prefetcher.submit(_frame(14)) is True
    finally:
        ready.set()
        prefetcher.close()


def test_prefetcher_rejects_returned_bundle_frame_id_mismatch_and_clears_pending() -> None:
    def _mismatched_builder(frame: Frame) -> FrameProposalBundle:
        return _bundle(_frame(frame.frame_id + 1))

    prefetcher = FrameProposalPrefetcher(_mismatched_builder, wait_timeout_sec=-1.0)
    try:
        assert prefetcher.submit(_frame(11)) is True
        assert prefetcher.result_for(11) is None
        stats = prefetcher.snapshot_stats()
        assert stats["submitted_count"] == 1
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 1
        assert stats["exception_count"] == 0
        assert "frame_id mismatch" in stats["last_exception"]
        assert "expected 11" in stats["last_exception"]
        assert "got 12" in stats["last_exception"]

        assert prefetcher.submit(_frame(12)) is True
    finally:
        prefetcher.close()


def test_prefetcher_records_exception_and_allows_inline_fallback() -> None:
    def _failing_builder(frame: Frame) -> FrameProposalBundle:
        raise RuntimeError(f"bad frame {frame.frame_id}")

    prefetcher = FrameProposalPrefetcher(_failing_builder, wait_timeout_sec=-1.0)
    try:
        assert prefetcher.submit(_frame(7)) is True
        assert prefetcher.result_for(7) is None
        stats = prefetcher.snapshot_stats()
        assert stats["submitted_count"] == 1
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 1
        assert stats["exception_count"] == 1
        assert "bad frame 7" in stats["last_exception"]
    finally:
        prefetcher.close()


def test_prefetcher_does_not_submit_second_pending_frame() -> None:
    prefetcher = FrameProposalPrefetcher(_bundle, wait_timeout_sec=-1.0)
    try:
        assert prefetcher.submit(_frame(1)) is True
        assert prefetcher.submit(_frame(2)) is False
        bundle = prefetcher.result_for(1)
        assert bundle is not None
        assert bundle.frame_id == 1
    finally:
        prefetcher.close()
