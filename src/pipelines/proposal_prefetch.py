"""Single-frame proposal prefetching utility for ordered pipelines."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from src.core.data_structures import Frame
from src.pipelines.proposal_bundle import FrameProposalBundle


@dataclass
class ProposalPrefetchStats:
    submitted_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    exception_count: int = 0
    wait_sec_total: float = 0.0
    last_exception: Any = ""

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "submitted_count": int(self.submitted_count),
            "hit_count": int(self.hit_count),
            "miss_count": int(self.miss_count),
            "exception_count": int(self.exception_count),
            "wait_sec_total": float(self.wait_sec_total),
            "last_exception": str(self.last_exception),
        }


class FrameProposalPrefetcher:
    """Prefetch at most one frame proposal bundle ahead of ordered consumption."""

    def __init__(
        self,
        builder: Callable[[Frame], FrameProposalBundle],
        *,
        wait_timeout_sec: float = -1.0,
        enabled: bool = True,
    ) -> None:
        self._builder = builder
        self._wait_timeout_sec = wait_timeout_sec
        self.enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="proposal-prefetch")
        self._future: Future[FrameProposalBundle] | None = None
        self._frame_id: int | None = None
        self._stats = ProposalPrefetchStats()
        self._lock = Lock()
        self._closed = False

    def submit(self, frame: Frame) -> bool:
        """Start prefetching a frame if no previous result is pending."""
        with self._lock:
            if not self.enabled or self._closed or self._future is not None:
                return False
            self._frame_id = frame.frame_id
            self._future = self._executor.submit(self._builder, frame)
            self._stats.submitted_count += 1
            return True

    def result_for(self, frame_id: int) -> FrameProposalBundle | None:
        """Return a ready matching bundle, or None so callers can build inline."""
        with self._lock:
            if not self.enabled:
                return None
            if self._future is None:
                self._stats.miss_count += 1
                return None
            if self._frame_id != frame_id:
                self._stats.miss_count += 1
                return None
            future = self._future

        timeout = None if self._wait_timeout_sec < 0 else self._wait_timeout_sec
        start = time.perf_counter()
        try:
            bundle = future.result(timeout=timeout)
        except TimeoutError:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._stats.wait_sec_total += elapsed
                self._stats.miss_count += 1
                self._stats.last_exception = (
                    f"TimeoutError: proposal prefetch timed out for frame {frame_id}"
                )
                if self._future is future and self._frame_id == frame_id:
                    self._future = None
                    self._frame_id = None
            future.cancel()
            return None
        except Exception as exc:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._stats.wait_sec_total += elapsed
                self._stats.miss_count += 1
                self._stats.exception_count += 1
                self._stats.last_exception = str(exc)
                self._future = None
                self._frame_id = None
            return None

        elapsed = time.perf_counter() - start
        with self._lock:
            if self._future is future and self._frame_id == frame_id:
                self._stats.wait_sec_total += elapsed
                self._future = None
                self._frame_id = None
                if bundle.frame_id != frame_id:
                    self._stats.miss_count += 1
                    self._stats.last_exception = (
                        f"frame_id mismatch: expected {frame_id}, got {bundle.frame_id}"
                    )
                    return None
                self._stats.hit_count += 1
                return bundle
            self._stats.wait_sec_total += elapsed
            self._stats.miss_count += 1
            return None

    def snapshot_stats(self) -> dict[str, int | float | str]:
        with self._lock:
            return self._stats.as_dict()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            future = self._future
            self._future = None
            self._frame_id = None
        if future is not None:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> FrameProposalPrefetcher:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
