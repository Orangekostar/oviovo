# Metric-Stable Parallel Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordered frontend prefetching and bounded read-only parallelism so the fast high-IoU room0 path runs faster without meaningful metric regression.

**Architecture:** Keep all map mutation in frame order. Extract frontend outputs into an immutable `FrameProposalBundle`, let the runner prefetch the next frame's bundle while the current frame runs stateful stages, and add deterministic thread-pool scoring in association. Depth refinement uses its existing proposal-level parallel path through explicit config flags.

**Tech Stack:** Python, dataclasses, `ThreadPoolExecutor`, pytest, NumPy, YAML config, existing OVIOVO pipeline modules and `run_room0_full_eval.py`.

---

## Reference Spec

- Design: `docs/superpowers/specs/2026-06-01-metric-stable-parallel-scheduling-design.md`
- Baseline run: `outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f`
- Baseline metrics: `mIoU=0.5223944433435148`, `f-mIoU=0.5433664207872895`
- Acceptance floor for 200-frame validation:
  - `mIoU >= 0.5203944433435148`
  - `f-mIoU >= 0.5413664207872895`

## File Structure

- Create: `src/pipelines/proposal_bundle.py`
  - Owns `FrameProposalBundle`.
  - Keeps frontend data separate from mutable `SystemState`.
- Create: `src/pipelines/proposal_prefetch.py`
  - Owns `FrameProposalPrefetcher` and `ProposalPrefetchStats`.
  - Does not import room0 runner code.
- Modify: `src/pipelines/main_pipeline.py`
  - Extract frontend generation into `_build_proposal_bundle(frame)`.
  - Add optional `proposal_bundle` parameter to `process_frame()`.
  - Preserve downstream stage order and existing `last_*` fields.
- Modify: `src/modules/association.py`
  - Add deterministic candidate scoring parallelism behind config flags.
- Modify: `src/modules/depth_refinement.py`
  - Add debug fields that show whether the existing proposal parallel branch ran.
- Modify: `configs/room0_surface_gate_fast_high_iou_4090.yaml`
  - Enable depth refinement parallelism.
  - Enable association scoring parallelism.
  - Enable proposal prefetch through pipeline config.
- Modify: `run_room0_full_eval.py`
  - Wire the prefetcher into the frame loop.
  - Record scheduling metrics in `frame_metrics.jsonl`, `run_report.md`, and `run_report.json`.
  - Extend stage timing summaries with median, p95, and outlier frame ids.
- Test: `tests/test_pipeline.py`
  - Pipeline bundle injection.
  - Association parallel equivalence.
  - Depth refinement parallel equivalence and config checks.
  - Report summary helpers.
- Create: `tests/test_proposal_prefetch.py`
  - Prefetcher hit/miss/exception/close behavior.

## Task 1: Proposal Bundle Extraction And Injection

**Files:**
- Create: `src/pipelines/proposal_bundle.py`
- Modify: `src/pipelines/main_pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing bundle injection test**

Add these imports near the existing imports in `tests/test_pipeline.py`:

```python
from src.pipelines.main_pipeline import Pipeline
from src.pipelines.proposal_bundle import FrameProposalBundle
```

Add this test method inside `class TestPipeline` after `test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend`:

```python
    def test_process_frame_consumes_proposal_bundle_without_calling_frontend(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "pipeline:",
                    "  verbose: false",
                    "  collect_stage_timings: true",
                    "anchor_frontend:",
                    "  enabled: false",
                    "proposal:",
                    "  backend: placeholder",
                ]
            ),
            encoding="utf-8",
        )
        pipe = Pipeline(config_path=str(config_path))

        class FailingProposal:
            config = {}
            active_backend_name = "failing"

            def process(self, *args, **kwargs):
                raise AssertionError("proposal frontend should not run when bundle is supplied")

        pipe.proposal = FailingProposal()

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4)
        bundle = FrameProposalBundle(
            frame_id=0,
            source_frame_id=123,
            source_proposals=[],
            raw_proposals=[],
            anchors=[],
            anchor_assignments=[],
            proposal_source="prefetched_empty",
            anchor_guided_sam_summary={"enabled": False, "assignment_count": 0},
            generation_timings={"yoloe_supplemental": 0.25},
            actual_backend="precomputed",
        )

        state = pipe.process_frame(
            rgb,
            depth,
            np.eye(4, dtype=np.float64),
            intrinsics,
            timestamp=1.5,
            source_frame_id=123,
            proposal_bundle=bundle,
        )

        assert isinstance(state, SystemState)
        assert pipe.last_raw_proposals == []
        assert pipe.last_source_proposals == []
        assert pipe.last_anchors == []
        assert pipe.last_anchor_assignments == []
        assert pipe.last_frame_debug["proposal_source"] == "prefetched_empty"
        assert pipe.last_frame_debug["frontend_source"] == "prefetch"
        assert pipe.last_frame_debug["frontend_actual_backend"] == "precomputed"
        assert pipe.last_frame_debug["stage_timings"]["yoloe_supplemental"] == pytest.approx(0.25)
```

- [ ] **Step 2: Run the bundle injection test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_process_frame_consumes_proposal_bundle_without_calling_frontend -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipelines.proposal_bundle'` or `TypeError: Pipeline.process_frame() got an unexpected keyword argument 'proposal_bundle'`.

- [ ] **Step 3: Create `FrameProposalBundle`**

Create `src/pipelines/proposal_bundle.py`:

```python
"""Immutable frontend proposal bundle for ordered pipeline scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.data_structures import Anchor2D, AnchorAssignment, Proposal2D


@dataclass(frozen=True)
class FrameProposalBundle:
    """Frontend outputs for one frame, with no map-state ownership."""

    frame_id: int
    source_frame_id: int | None
    source_proposals: list[Proposal2D]
    raw_proposals: list[Proposal2D]
    anchors: list[Anchor2D]
    anchor_assignments: list[AnchorAssignment]
    proposal_source: str
    anchor_guided_sam_summary: dict[str, Any] = field(default_factory=dict)
    generation_timings: dict[str, float] = field(default_factory=dict)
    actual_backend: str = ""
```

- [ ] **Step 4: Extract frontend generation in `main_pipeline.py`**

In `src/pipelines/main_pipeline.py`, add the import:

```python
from src.pipelines.proposal_bundle import FrameProposalBundle
```

Add this method after `_merge_stage_timings()`:

```python
    def _default_anchor_guided_sam_summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.anchor_guided_sam, "enabled", False)),
            "anchor_count": 0,
            "source_sam_proposal_count": 0,
            "output_proposal_count": 0,
            "anchored_proposal_count": 0,
            "unknown_residual_count": 0,
            "dropped_proposal_count": 0,
            "matched_sam_proposal_count": 0,
            "mean_sam_candidates_per_anchor": 0.0,
            "semantic_blocked_residual_count": 0,
            "assignment_count": 0,
        }

    def _build_proposal_bundle(self, frame: Frame) -> FrameProposalBundle:
        anchor_guided_sam_summary = self._default_anchor_guided_sam_summary()
        proposal_source = "proposal_backend"
        source_proposals: list[Proposal2D] = []
        generation_timings: dict[str, float] = {}
        with self._timed_stage("proposal_generation"):
            if self.object_anchor.enabled and getattr(self.object_anchor, "anchor_primary_mode", False):
                anchors, anchor_box_proposals, anchor_assignments = self.object_anchor.generate_anchor_box_proposals(frame.rgb)
                if getattr(self.anchor_guided_sam, "enabled", False):
                    sam_proposals = self.proposal.process_for_anchors(
                        frame.rgb,
                        frame.depth,
                        anchors,
                        frame=frame,
                    )
                    source_proposals = list(sam_proposals)
                    proposals, anchor_assignments, anchor_guided_sam_summary = self.anchor_guided_sam.build_proposals(
                        frame=frame,
                        anchors=list(anchors),
                        sam_proposals=sam_proposals,
                    )
                    proposal_source = "anchor_guided_sam"
                else:
                    proposals = anchor_box_proposals
                    self._stamp_anchor_primary_coarse_metadata(frame, proposals, anchor_assignments)
                    proposal_source = "anchor_box_primary"
            elif self.object_anchor.enabled and self.object_anchor.use_sam_intersection_proposals:
                sam_proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
                source_proposals = list(sam_proposals)
                anchors, proposals, anchor_assignments = self.object_anchor.generate_proposals(frame.rgb, sam_proposals)
                vote_policies = {"semantic_vote", "vote_only", "sam_mask_semantic_vote"}
                assignment_policy = str(getattr(self.object_anchor, "assignment_policy", "") or "")
                proposal_source = "anchor_sam_union"
                if assignment_policy in vote_policies or (
                    proposals
                    and all(
                        proposal.backend_name == "sam2_anchor_vote"
                        or proposal.metadata.get("source") == "sam2_anchor_vote"
                        for proposal in proposals
                    )
                ):
                    proposal_source = "sam2_anchor_vote"
            else:
                proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
                source_proposals = list(proposals)
                anchors, anchor_assignments = self.object_anchor.process(frame.rgb, proposals)
            generation_timings = dict(getattr(self.object_anchor, "last_generation_timings", {}) or {})
        return FrameProposalBundle(
            frame_id=int(frame.frame_id),
            source_frame_id=frame.source_frame_id,
            source_proposals=list(source_proposals),
            raw_proposals=list(proposals),
            anchors=list(anchors),
            anchor_assignments=list(anchor_assignments),
            proposal_source=proposal_source,
            anchor_guided_sam_summary=dict(anchor_guided_sam_summary),
            generation_timings=generation_timings,
            actual_backend=str(getattr(self.proposal, "active_backend_name", "")),
        )

    def _consume_proposal_bundle(self, frame: Frame, bundle: FrameProposalBundle) -> tuple[list[Proposal2D], list[Proposal2D], list[Any], list[Any], str, dict[str, Any]]:
        if int(bundle.frame_id) != int(frame.frame_id):
            raise ValueError(f"Proposal bundle frame_id mismatch: expected {frame.frame_id}, got {bundle.frame_id}")
        self.last_raw_proposals = list(bundle.raw_proposals)
        self.last_source_proposals = list(bundle.source_proposals)
        self.last_anchors = list(bundle.anchors)
        self.last_anchor_assignments = list(bundle.anchor_assignments)
        self._merge_stage_timings(bundle.generation_timings)
        return (
            self.last_raw_proposals,
            self.last_source_proposals,
            self.last_anchors,
            self.last_anchor_assignments,
            str(bundle.proposal_source),
            dict(bundle.anchor_guided_sam_summary),
        )
```

- [ ] **Step 5: Update `process_frame()` signature and frontend block**

Change the `process_frame()` signature:

```python
        source_frame_id: int | None = None,
        proposal_bundle: FrameProposalBundle | None = None,
    ) -> SystemState:
```

Replace the current inline proposal generation block with:

```python
        frontend_source = "inline"
        frontend_actual_backend = str(getattr(self.proposal, "active_backend_name", ""))
        if proposal_bundle is None:
            bundle = self._build_proposal_bundle(frame)
        else:
            bundle = proposal_bundle
            frontend_source = "prefetch"
            frontend_actual_backend = str(bundle.actual_backend)
        proposals, source_proposals, anchors, anchor_assignments, proposal_source, anchor_guided_sam_summary = (
            self._consume_proposal_bundle(frame, bundle)
        )
        if self.verbose:
            logger.info("  Proposals (%s): %d", proposal_source, len(proposals))
            if self.object_anchor.enabled:
                logger.info(f"  Anchors: {len(anchors)}")
```

In `self.last_frame_debug`, add:

```python
            "frontend_source": frontend_source,
            "frontend_actual_backend": frontend_actual_backend,
```

- [ ] **Step 6: Run the bundle injection test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_process_frame_consumes_proposal_bundle_without_calling_frontend -q
```

Expected: PASS.

- [ ] **Step 7: Run existing pipeline smoke tests touched by the signature**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend tests/test_pipeline.py::TestPipeline::test_process_frame_passes_current_frame_geometry_to_object_update -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add src/pipelines/proposal_bundle.py src/pipelines/main_pipeline.py tests/test_pipeline.py
git commit -m "feat: add proposal bundle pipeline injection"
```

## Task 2: Proposal Prefetcher Utility

**Files:**
- Create: `src/pipelines/proposal_prefetch.py`
- Create: `tests/test_proposal_prefetch.py`

- [ ] **Step 1: Write failing prefetcher tests**

Create `tests/test_proposal_prefetch.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import CameraIntrinsics, Frame
from src.pipelines.proposal_bundle import FrameProposalBundle
from src.pipelines.proposal_prefetch import FrameProposalPrefetcher


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
```

- [ ] **Step 2: Run prefetcher tests and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_proposal_prefetch.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipelines.proposal_prefetch'`.

- [ ] **Step 3: Implement `FrameProposalPrefetcher`**

Create `src/pipelines/proposal_prefetch.py`:

```python
"""Single-frame-ahead frontend prefetching for ordered pipeline runs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import time
from typing import Callable

from src.core.data_structures import Frame
from src.pipelines.proposal_bundle import FrameProposalBundle


@dataclass
class ProposalPrefetchStats:
    submitted_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    exception_count: int = 0
    wait_sec_total: float = 0.0
    last_exception: str = ""

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
    """Prefetch exactly one future frame's frontend bundle."""

    def __init__(
        self,
        builder: Callable[[Frame], FrameProposalBundle],
        *,
        wait_timeout_sec: float = -1.0,
        enabled: bool = True,
    ) -> None:
        self.builder = builder
        self.wait_timeout_sec = float(wait_timeout_sec)
        self.enabled = bool(enabled)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="proposal-prefetch")
        self._pending_frame_id: int | None = None
        self._pending_future: Future[FrameProposalBundle] | None = None
        self._stats = ProposalPrefetchStats()

    def submit(self, frame: Frame) -> bool:
        if not self.enabled:
            return False
        if self._pending_future is not None and not self._pending_future.done():
            return False
        self._pending_frame_id = int(frame.frame_id)
        self._pending_future = self._executor.submit(self.builder, frame)
        self._stats.submitted_count += 1
        return True

    def result_for(self, frame_id: int) -> FrameProposalBundle | None:
        if not self.enabled:
            return None
        if self._pending_future is None or self._pending_frame_id is None:
            self._stats.miss_count += 1
            return None
        if int(frame_id) != int(self._pending_frame_id):
            self._stats.miss_count += 1
            return None

        timeout = None if self.wait_timeout_sec < 0.0 else max(0.0, self.wait_timeout_sec)
        start = time.perf_counter()
        try:
            bundle = self._pending_future.result(timeout=timeout)
        except TimeoutError:
            self._stats.miss_count += 1
            return None
        except Exception as exc:
            self._stats.miss_count += 1
            self._stats.exception_count += 1
            self._stats.last_exception = f"{type(exc).__name__}: {exc}"
            self._pending_future = None
            self._pending_frame_id = None
            return None
        finally:
            self._stats.wait_sec_total += float(time.perf_counter() - start)

        self._pending_future = None
        self._pending_frame_id = None
        if int(bundle.frame_id) != int(frame_id):
            self._stats.miss_count += 1
            self._stats.last_exception = f"frame_id mismatch: expected {frame_id}, got {bundle.frame_id}"
            return None
        self._stats.hit_count += 1
        return bundle

    def snapshot_stats(self) -> dict[str, int | float | str]:
        return self._stats.as_dict()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
```

- [ ] **Step 4: Run prefetcher tests and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_proposal_prefetch.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/pipelines/proposal_prefetch.py tests/test_proposal_prefetch.py
git commit -m "feat: add ordered proposal prefetcher"
```

## Task 3: Runner Prefetch Wiring And Scheduling Metrics

**Files:**
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing scheduling summary tests**

Add this import to the `run_room0_full_eval` import block in `tests/test_pipeline.py`:

```python
    build_scheduling_report_payload,
```

Add this test method inside `class TestPipeline` near the config test:

```python
    def test_scheduling_report_payload_sums_prefetch_metrics(self):
        payload = build_scheduling_report_payload(
            [
                {
                    "scheduling": {
                        "prefetch_enabled": True,
                        "frontend_source": "prefetch",
                        "prefetch_submitted": True,
                        "prefetch_hit": True,
                        "prefetch_miss": False,
                        "prefetch_wait_sec": 0.25,
                        "prefetch_exception": "",
                    }
                },
                {
                    "scheduling": {
                        "prefetch_enabled": True,
                        "frontend_source": "inline",
                        "prefetch_submitted": True,
                        "prefetch_hit": False,
                        "prefetch_miss": True,
                        "prefetch_wait_sec": 0.50,
                        "prefetch_exception": "RuntimeError: bad frame",
                    }
                },
            ]
        )

        assert payload["prefetch_enabled"] is True
        assert payload["prefetch_submitted_count"] == 2
        assert payload["prefetch_hit_count"] == 1
        assert payload["prefetch_miss_count"] == 1
        assert payload["prefetch_wait_sec_total"] == pytest.approx(0.75)
        assert payload["frontend_prefetch_frame_count"] == 1
        assert payload["frontend_inline_frame_count"] == 1
        assert payload["frontend_exception_count"] == 1
        assert payload["frontend_last_exception"] == "RuntimeError: bad frame"
```

- [ ] **Step 2: Run scheduling summary test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_scheduling_report_payload_sums_prefetch_metrics -q
```

Expected: FAIL with `ImportError` for `build_scheduling_report_payload`.

- [ ] **Step 3: Add scheduling report helper**

In `run_room0_full_eval.py`, add after `build_frontend_stage_report_payload()`:

```python
def build_scheduling_report_payload(frame_metrics: list[dict[str, Any]]) -> dict[str, int | float | bool | str]:
    submitted_count = 0
    hit_count = 0
    miss_count = 0
    wait_sec_total = 0.0
    prefetch_frames = 0
    inline_frames = 0
    exception_count = 0
    last_exception = ""
    enabled = False
    for metrics in frame_metrics:
        scheduling = dict(metrics.get("scheduling", {}) or {})
        enabled = enabled or bool(scheduling.get("prefetch_enabled", False))
        if bool(scheduling.get("prefetch_submitted", False)):
            submitted_count += 1
        if bool(scheduling.get("prefetch_hit", False)):
            hit_count += 1
        if bool(scheduling.get("prefetch_miss", False)):
            miss_count += 1
        try:
            wait_sec_total += float(scheduling.get("prefetch_wait_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        if str(scheduling.get("frontend_source", "")) == "prefetch":
            prefetch_frames += 1
        if str(scheduling.get("frontend_source", "")) == "inline":
            inline_frames += 1
        exception = str(scheduling.get("prefetch_exception", "") or "")
        if exception:
            exception_count += 1
            last_exception = exception
    return {
        "prefetch_enabled": bool(enabled),
        "prefetch_submitted_count": int(submitted_count),
        "prefetch_hit_count": int(hit_count),
        "prefetch_miss_count": int(miss_count),
        "prefetch_wait_sec_total": float(wait_sec_total),
        "frontend_prefetch_frame_count": int(prefetch_frames),
        "frontend_inline_frame_count": int(inline_frames),
        "frontend_exception_count": int(exception_count),
        "frontend_last_exception": last_exception,
    }
```

- [ ] **Step 4: Run scheduling summary test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_scheduling_report_payload_sums_prefetch_metrics -q
```

Expected: PASS.

- [ ] **Step 5: Wire prefetcher into the room0 frame loop**

In `run_room0_full_eval.py`, update the data-structure import and add the prefetcher import:

```python
from src.core.data_structures import Frame, ObjectState, SystemState
from src.pipelines.proposal_prefetch import FrameProposalPrefetcher
```

After `pipeline.proposal = build_proposal_module(...)`, add:

```python
    pipeline_cfg = dict(pipeline.config.get("pipeline", {}) or {})
    prefetch_enabled = bool(pipeline_cfg.get("proposal_prefetch_enabled", False))
    prefetch_wait_timeout_sec = float(pipeline_cfg.get("proposal_prefetch_wait_timeout_sec", -1.0))
    prefetcher = FrameProposalPrefetcher(
        pipeline._build_proposal_bundle,
        enabled=prefetch_enabled,
        wait_timeout_sec=prefetch_wait_timeout_sec,
    )

    def build_prefetch_frame(dataset_frame: Frame, processed_frame_id: int) -> Frame:
        return Frame(
            frame_id=int(processed_frame_id),
            source_frame_id=int(dataset_frame.frame_id),
            rgb=dataset_frame.rgb,
            depth=dataset_frame.depth,
            pose=dataset_frame.pose,
            intrinsics=dataset_frame.intrinsics,
            timestamp=float(dataset_frame.timestamp),
        )
```

Replace the top of the frame loop with the ordered prefetch pattern:

```python
        try:
            next_prefetch_index = 0
            if prefetch_enabled and selected_dataset_indices:
                first_frame = dataset[int(selected_dataset_indices[0])]
                prefetcher.submit(
                    build_prefetch_frame(first_frame, processed_frame_id=0)
                )
                next_prefetch_index = 1
            for idx, dataset_index in enumerate(selected_dataset_indices, start=1):
                frame = dataset[int(dataset_index)]
                scheduling_debug = {
                    "prefetch_enabled": bool(prefetch_enabled),
                    "prefetch_submitted": False,
                    "prefetch_hit": False,
                    "prefetch_miss": False,
                    "prefetch_wait_sec": 0.0,
                    "prefetch_exception": "",
                    "frontend_source": "inline",
                }
                proposal_bundle = None
                if prefetch_enabled:
                    wait_before = time.perf_counter()
                    proposal_bundle = prefetcher.result_for(idx - 1)
                    scheduling_debug["prefetch_wait_sec"] = float(time.perf_counter() - wait_before)
                    if proposal_bundle is not None:
                        scheduling_debug["prefetch_hit"] = True
                        scheduling_debug["frontend_source"] = "prefetch"
                    else:
                        scheduling_debug["prefetch_miss"] = True
                    stats = prefetcher.snapshot_stats()
                    scheduling_debug["prefetch_exception"] = str(stats.get("last_exception", ""))

                submitted_next_before_processing = False
                if proposal_bundle is not None and prefetch_enabled and next_prefetch_index < len(selected_dataset_indices):
                    next_dataset_index = selected_dataset_indices[next_prefetch_index]
                    next_frame = dataset[int(next_dataset_index)]
                    submitted = prefetcher.submit(
                        build_prefetch_frame(next_frame, processed_frame_id=next_prefetch_index)
                    )
                    scheduling_debug["prefetch_submitted"] = bool(submitted)
                    if submitted:
                        next_prefetch_index += 1
                        submitted_next_before_processing = True
```

Then pass `proposal_bundle=proposal_bundle` into `pipeline.process_frame(...)`, and add this after the call:

```python
                scheduling_debug["frontend_source"] = str(
                    pipeline.last_frame_debug.get("frontend_source", scheduling_debug["frontend_source"])
                )
                pipeline.last_frame_debug["scheduling"] = dict(scheduling_debug)
                if (
                    prefetch_enabled
                    and not submitted_next_before_processing
                    and next_prefetch_index < len(selected_dataset_indices)
                ):
                    next_dataset_index = selected_dataset_indices[next_prefetch_index]
                    next_frame = dataset[int(next_dataset_index)]
                    submitted = prefetcher.submit(
                        build_prefetch_frame(next_frame, processed_frame_id=next_prefetch_index)
                    )
                    if submitted:
                        next_prefetch_index += 1
```

This explicit `Frame(...)` construction is required because `pipeline.process_frame()` owns `pipeline.frame_input.frame_counter`; background prefetch must not increment that counter before the ordered main-thread frame is committed. Submitting the next prefetch before current-frame processing is limited to hit frames, so a miss does not run inline frontend work and background frontend work through the same mutable modules at the same time.

At the end of the existing `try/finally`, ensure the prefetcher closes:

```python
        finally:
            prefetcher.close()
            if audit_handle is not None:
                audit_handle.close()
```

When building `record`, add:

```python
                    "scheduling": dict(pipeline.last_frame_debug.get("scheduling", {}) or {}),
```

- [ ] **Step 6: Add scheduling totals to run report**

In `write_run_report()`, compute:

```python
    scheduling_totals = build_scheduling_report_payload(frame_metrics)
```

Add these lines in the Evaluation Metadata section:

```python
        f"- `prefetch_enabled`: `{scheduling_totals['prefetch_enabled']}`",
        f"- `prefetch_submitted_count`: `{scheduling_totals['prefetch_submitted_count']}`",
        f"- `prefetch_hit_count`: `{scheduling_totals['prefetch_hit_count']}`",
        f"- `prefetch_miss_count`: `{scheduling_totals['prefetch_miss_count']}`",
        f"- `prefetch_wait_sec_total`: `{scheduling_totals['prefetch_wait_sec_total']:.4f}`",
        f"- `frontend_prefetch_frame_count`: `{scheduling_totals['frontend_prefetch_frame_count']}`",
        f"- `frontend_inline_frame_count`: `{scheduling_totals['frontend_inline_frame_count']}`",
        f"- `frontend_exception_count`: `{scheduling_totals['frontend_exception_count']}`",
```

Add to the `sidecar` dict:

```python
        **scheduling_totals,
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_scheduling_report_payload_sums_prefetch_metrics tests/test_proposal_prefetch.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add run_room0_full_eval.py tests/test_pipeline.py
git commit -m "feat: prefetch frontend bundles in room0 runner"
```

## Task 4: Deterministic Association Score Parallelism

**Files:**
- Modify: `src/modules/association.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing association parallel equivalence test**

Add this test method inside `class TestRoadmapRefactor` near the other association tests:

```python
    def test_association_parallel_scoring_matches_serial_result(self):
        patch_points = np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0], [0.05, 0.05, 1.0]],
            dtype=np.float32,
        )
        patch = Patch3D(
            patch_id=10,
            points=patch_points,
            centroid=patch_points.mean(axis=0),
            bbox_min=patch_points.min(axis=0),
            bbox_max=patch_points.max(axis=0),
        )
        objects = {}
        for obj_id, offset in [(1, 0.02), (2, 0.35), (3, 0.70), (4, 1.10)]:
            pts = patch_points + np.array([offset, 0.0, 0.0], dtype=np.float32)
            objects[obj_id] = ObjectMap(
                object_id=obj_id,
                centroid=pts.mean(axis=0),
                bbox_min=pts.min(axis=0),
                bbox_max=pts.max(axis=0),
                local_pcd=pts.copy(),
            )

        base_config = {
            "match_threshold": 0.1,
            "voxel_vote_weight": 0.0,
            "centroid_distance_weight": 0.7,
            "bbox_overlap_weight": 0.1,
            "geometry_overlap_weight": 0.2,
            "max_geometry_candidates": 4,
            "score_parallel_min_candidates": 2,
        }
        serial = AssociationModule({**base_config, "score_parallel_enabled": False})
        parallel = AssociationModule(
            {
                **base_config,
                "score_parallel_enabled": True,
                "score_parallel_workers": 2,
            }
        )

        serial_result = serial.process([patch], objects, SystemState().tsdf_volume)
        parallel_result = parallel.process([patch], objects, SystemState().tsdf_volume)

        assert parallel_result.matched[0][0:2] == serial_result.matched[0][0:2]
        assert parallel_result.new_object_patches == serial_result.new_object_patches
        assert parallel_result.contested_object_patches == serial_result.contested_object_patches
        assert sorted(parallel_result.scores) == sorted(serial_result.scores)
        for key in sorted(serial_result.scores):
            assert parallel_result.scores[key].total_score == pytest.approx(serial_result.scores[key].total_score)
            assert parallel_result.scores[key].geometry_overlap == pytest.approx(serial_result.scores[key].geometry_overlap)
        assert parallel_result.debug["summary"]["score_parallel_used_count"] == 1
        assert parallel_result.debug["summary"]["score_parallel_candidate_count_total"] == 4
```

- [ ] **Step 2: Run association parallel test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_association_parallel_scoring_matches_serial_result -q
```

Expected: FAIL because `score_parallel_used_count` is missing.

- [ ] **Step 3: Add association config fields**

In `src/modules/association.py`, add imports:

```python
from concurrent.futures import ThreadPoolExecutor
import os
```

In `AssociationModule.__init__()`, add:

```python
        self.score_parallel_enabled = bool(config.get("score_parallel_enabled", False))
        self.score_parallel_workers = int(config.get("score_parallel_workers", 0))
        self.score_parallel_min_candidates = int(config.get("score_parallel_min_candidates", 16))
```

Add this helper near `_resolve_candidate_ids()`:

```python
    @staticmethod
    def _resolve_worker_count(configured_workers: int, task_count: int) -> int:
        if task_count <= 0:
            return 1
        if configured_workers > 0:
            return max(1, min(configured_workers, task_count))
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count, task_count))
```

- [ ] **Step 4: Add deterministic score calculation helper**

Add this method near `_compute_score()`:

```python
    def _score_candidate_objects(
        self,
        patch: Patch3D,
        objects: Dict[int, ObjectMap],
        candidate_ids: list[int],
        vote: VoxelVoteResult,
        geometry_candidate_ids: set[int],
        cheap_by_id: dict[int, tuple[float, float, float, float]],
    ) -> tuple[list[tuple[int, AssociationScore]], bool]:
        valid_ids = [
            int(obj_id)
            for obj_id in candidate_ids
            if obj_id in objects and objects[obj_id].state.value != "removed"
        ]
        use_parallel = (
            self.score_parallel_enabled
            and len(valid_ids) >= self.score_parallel_min_candidates
            and self._resolve_worker_count(self.score_parallel_workers, len(valid_ids)) > 1
        )

        def _score(obj_id: int) -> tuple[int, AssociationScore]:
            obj = objects[obj_id]
            return (
                obj_id,
                self._compute_score(
                    patch,
                    obj,
                    vote,
                    cheap_components=cheap_by_id.get(int(obj_id)),
                    compute_geometry=int(obj_id) in geometry_candidate_ids,
                ),
            )

        if not use_parallel:
            return [_score(obj_id) for obj_id in valid_ids], False

        worker_count = self._resolve_worker_count(self.score_parallel_workers, len(valid_ids))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(_score, valid_ids)), True
```

- [ ] **Step 5: Use parallel helper in `process()`**

Inside `process()`, before the patch loop add:

```python
        score_parallel_used_count = 0
        score_parallel_candidate_count_total = 0
```

Replace the inner `for obj_id in candidate_ids:` scoring loop with:

```python
            scored_candidates, score_parallel_used = self._score_candidate_objects(
                patch,
                objects,
                list(candidate_ids),
                vote,
                geometry_candidate_ids,
                cheap_by_id,
            )
            if score_parallel_used:
                score_parallel_used_count += 1
                score_parallel_candidate_count_total += len(scored_candidates)

            for obj_id, score in scored_candidates:
                obj = objects.get(obj_id)
                if obj is None or obj.state.value == "removed":
                    continue
                result.scores[(patch.patch_id, obj_id)] = score
```

Keep the existing identity-gate and best-score logic below that line.

In `result.debug["summary"]`, add:

```python
            "score_parallel_enabled": bool(self.score_parallel_enabled),
            "score_parallel_used_count": int(score_parallel_used_count),
            "score_parallel_candidate_count_total": int(score_parallel_candidate_count_total),
```

- [ ] **Step 6: Run association parallel test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestRoadmapRefactor::test_association_parallel_scoring_matches_serial_result -q
```

Expected: PASS.

- [ ] **Step 7: Run association regression tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_restricts_candidates_to_active_set \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_limits_geometry_consistency_to_top_k_candidates \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_blocks_confident_cross_label_patch_against_repeated_provisional_identity \
  tests/test_pipeline.py::TestRoadmapRefactor::test_safe_provisional_export_label_does_not_create_association_conflict \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

Run:

```bash
git add src/modules/association.py tests/test_pipeline.py
git commit -m "perf: parallelize deterministic association scoring"
```

## Task 5: Depth Refinement Parallel Debug And Fast High-IoU Config

**Files:**
- Modify: `src/modules/depth_refinement.py`
- Modify: `configs/room0_surface_gate_fast_high_iou_4090.yaml`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing depth parallel equivalence test**

Add this test method inside `class TestPipeline` near the existing depth refinement tests:

```python
    def test_depth_refinement_parallel_matches_serial_output(self):
        depth = np.full((16, 16), 2.0, dtype=np.float32)
        proposals = []
        for proposal_id in range(6):
            mask = np.zeros((16, 16), dtype=bool)
            row = 1 + proposal_id * 2
            mask[row : row + 2, 2:8] = True
            proposals.append(
                Proposal2D(
                    proposal_id=proposal_id,
                    mask=mask,
                    bbox_xyxy=np.array([2, row, 8, row + 2], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.9,
                )
            )

        base_config = {
            "depth_edge_threshold": 10.0,
            "min_mask_area_after_refine": 1,
            "connected_components_backend": "python",
            "use_bbox_crop": True,
            "proposal_parallel_min_tasks": 2,
        }
        serial = DepthRefinementModule({**base_config, "proposal_parallel_enabled": False})
        parallel = DepthRefinementModule(
            {
                **base_config,
                "proposal_parallel_enabled": True,
                "proposal_parallel_workers": 2,
            }
        )

        serial_refined = serial.process(depth, proposals)
        parallel_refined = parallel.process(depth, proposals)

        assert parallel.last_parallel_used is True
        assert [item.proposal_id for item in parallel_refined] == [item.proposal_id for item in serial_refined]
        assert [item.area for item in parallel_refined] == [item.area for item in serial_refined]
        assert [item.mask.tolist() for item in parallel_refined] == [item.mask.tolist() for item in serial_refined]
```

Extend `test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend` with:

```python
        assert config["pipeline"]["proposal_prefetch_enabled"] is True
        assert config["pipeline"]["proposal_prefetch_wait_timeout_sec"] == -1.0
        assert config["depth_refinement"]["proposal_parallel_enabled"] is True
        assert config["depth_refinement"]["proposal_parallel_workers"] == 4
        assert config["depth_refinement"]["proposal_parallel_min_tasks"] == 16
        assert config["association"]["score_parallel_enabled"] is True
        assert config["association"]["score_parallel_workers"] == 4
        assert config["association"]["score_parallel_min_candidates"] == 16
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_depth_refinement_parallel_matches_serial_output \
  tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend \
  -q
```

Expected: FAIL because `last_parallel_used` is missing and config keys are absent.

- [ ] **Step 3: Add depth refinement parallel debug fields**

In `DepthRefinementModule.__init__()`, add:

```python
        self.last_parallel_used = False
        self.last_parallel_worker_count = 1
```

At the start of `process()`, after `depth_edges = ...`, add:

```python
        self.last_parallel_used = False
        self.last_parallel_worker_count = 1
```

Inside the `if worker_count > 1:` branch before constructing the executor, add:

```python
                self.last_parallel_used = True
                self.last_parallel_worker_count = int(worker_count)
```

- [ ] **Step 4: Enable config flags**

In `configs/room0_surface_gate_fast_high_iou_4090.yaml`, under `depth_refinement`, add:

```yaml
  proposal_parallel_enabled: true
  proposal_parallel_workers: 4
  proposal_parallel_min_tasks: 16
```

Under `association`, add:

```yaml
  score_parallel_enabled: true
  score_parallel_workers: 4
  score_parallel_min_candidates: 16
```

Under `pipeline`, add:

```yaml
  proposal_prefetch_enabled: true
  proposal_prefetch_wait_timeout_sec: -1.0
```

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_depth_refinement_parallel_matches_serial_output \
  tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add src/modules/depth_refinement.py configs/room0_surface_gate_fast_high_iou_4090.yaml tests/test_pipeline.py
git commit -m "config: enable metric-stable parallel scheduling"
```

## Task 6: Reporting P50/P95/Outlier Timing Summaries

**Files:**
- Modify: `run_room0_full_eval.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing stage timing summary test**

Add this import to the `run_room0_full_eval` import block in `tests/test_pipeline.py`:

```python
    build_stage_timing_summary,
```

Add this test method inside `class TestPipeline`:

```python
    def test_stage_timing_summary_includes_percentiles_and_outliers(self):
        summary = build_stage_timing_summary(
            [
                {"frame_id": 0, "stage_timings": {"association": 1.0}},
                {"frame_id": 10, "stage_timings": {"association": 2.0}},
                {"frame_id": 20, "stage_timings": {"association": 10.0}},
            ]
        )

        assoc = summary["association"]
        assert assoc["mean_sec"] == pytest.approx(13.0 / 3.0)
        assert assoc["median_sec"] == pytest.approx(2.0)
        assert assoc["p95_sec"] == pytest.approx(10.0)
        assert assoc["max_sec"] == pytest.approx(10.0)
        assert assoc["max_frame_id"] == 20
        assert assoc["outlier_frame_ids"] == [20]
```

- [ ] **Step 2: Run timing summary test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_stage_timing_summary_includes_percentiles_and_outliers -q
```

Expected: FAIL because `median_sec`, `p95_sec`, `max_frame_id`, and `outlier_frame_ids` are missing.

- [ ] **Step 3: Extend `build_stage_timing_summary()`**

Replace `build_stage_timing_summary()` in `run_room0_full_eval.py` with:

```python
def build_stage_timing_summary(frame_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stage_values: dict[str, list[tuple[int, float]]] = {}
    for metrics in frame_metrics:
        frame_id = int(metrics.get("frame_id", -1))
        timings = dict(metrics.get("stage_timings", {}) or {})
        for stage, value in timings.items():
            try:
                timing_value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(timing_value):
                continue
            stage_values.setdefault(str(stage), []).append((frame_id, timing_value))
    summary: dict[str, dict[str, Any]] = {}
    for stage, pairs in sorted(stage_values.items()):
        if not pairs:
            continue
        values = np.asarray([value for _frame_id, value in pairs], dtype=np.float64)
        max_index = int(np.argmax(values))
        p95 = float(np.percentile(values, 95.0, method="nearest"))
        outlier_frame_ids = [
            int(frame_id)
            for frame_id, value in pairs
            if float(value) >= p95 and float(value) > float(np.median(values))
        ]
        summary[stage] = {
            "mean_sec": float(np.mean(values)),
            "median_sec": float(np.median(values)),
            "p95_sec": p95,
            "max_sec": float(np.max(values)),
            "total_sec": float(np.sum(values)),
            "max_frame_id": int(pairs[max_index][0]),
            "outlier_frame_ids": outlier_frame_ids[:10],
        }
    return summary
```

- [ ] **Step 4: Update report text for new timing fields**

In `write_run_report()`, replace the stage timing report line with:

```python
                f"- `{stage}`: mean `{values['mean_sec']:.4f}s`, "
                f"median `{values['median_sec']:.4f}s`, "
                f"p95 `{values['p95_sec']:.4f}s`, "
                f"max `{values['max_sec']:.4f}s` at frame `{values['max_frame_id']}`, "
                f"total `{values['total_sec']:.4f}s`"
```

- [ ] **Step 5: Run timing summary test and verify GREEN**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_stage_timing_summary_includes_percentiles_and_outliers -q
```

Expected: PASS.

- [ ] **Step 6: Run report helper tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_stage_timing_summary_includes_percentiles_and_outliers \
  tests/test_pipeline.py::TestPipeline::test_scheduling_report_payload_sums_prefetch_metrics \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add run_room0_full_eval.py tests/test_pipeline.py
git commit -m "perf: report scheduling and timing outliers"
```

## Task 7: Focused Regression And Validation Runs

**Files:**
- No source edits expected.
- Validation outputs under `outputs/tmp_validation/`.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_proposal_prefetch.py \
  tests/test_pipeline.py::TestPipeline::test_process_frame_consumes_proposal_bundle_without_calling_frontend \
  tests/test_pipeline.py::TestPipeline::test_scheduling_report_payload_sums_prefetch_metrics \
  tests/test_pipeline.py::TestRoadmapRefactor::test_association_parallel_scoring_matches_serial_result \
  tests/test_pipeline.py::TestPipeline::test_depth_refinement_parallel_matches_serial_output \
  tests/test_pipeline.py::TestPipeline::test_stage_timing_summary_includes_percentiles_and_outliers \
  tests/test_pipeline.py::TestPipeline::test_room0_surface_gate_fast_high_iou_config_preserves_0526_frontend \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run broader affected tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py \
  tests/test_runtime_vis.py \
  tests/test_object_anchor.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run 20-frame stride-10 smoke validation**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
experiment=20260601_room0_metric_stable_parallel_s10_20f_smoke
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name "${experiment}" \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected:

- exit status `0`
- `outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_20f_smoke/replica/results.json` exists
- `run_report.json` has `prefetch_enabled: true`
- `prefetch_submitted_count > 0`
- `prefetch_hit_count > 0`

- [ ] **Step 4: Inspect 20-frame scheduling metrics**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_20f_smoke")
report = json.loads((root / "room0/run_report.json").read_text())
print("miou", report["miou"])
print("fmiou", report["fmiou"])
print("prefetch_enabled", report["prefetch_enabled"])
print("prefetch_submitted_count", report["prefetch_submitted_count"])
print("prefetch_hit_count", report["prefetch_hit_count"])
print("prefetch_miss_count", report["prefetch_miss_count"])
print("frontend_exception_count", report["frontend_exception_count"])
print("stage_timing_summary_keys", sorted(report["stage_timing_summary"])[:8])
PY
```

Expected:

- `prefetch_enabled True`
- `prefetch_submitted_count` is greater than `0`
- `prefetch_hit_count` is greater than `0`
- `frontend_exception_count 0`

- [ ] **Step 5: Run 200-frame stride-10 validation**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
experiment=20260601_room0_metric_stable_parallel_s10_200f
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml \
  --output-root outputs/tmp_validation \
  --experiment-name "${experiment}" \
  --num-frames 200 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --fast-eval
```

Expected:

- exit status `0`
- `outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_200f/replica/results.json` exists
- `run_report.json` exists
- `frame_metrics.jsonl` contains `scheduling` records

- [ ] **Step 6: Compare against baseline acceptance floor**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
baseline = Path("outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f")
candidate = Path("outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_200f")
base = json.loads((baseline / "replica/results.json").read_text())
cand = json.loads((candidate / "replica/results.json").read_text())
report = json.loads((candidate / "room0/run_report.json").read_text())
print("baseline_miou", base["miou"])
print("candidate_miou", cand["miou"])
print("delta_miou", cand["miou"] - base["miou"])
print("baseline_fmiou", base["fmiou"])
print("candidate_fmiou", cand["fmiou"])
print("delta_fmiou", cand["fmiou"] - base["fmiou"])
print("prefetch_submitted_count", report["prefetch_submitted_count"])
print("prefetch_hit_count", report["prefetch_hit_count"])
print("prefetch_miss_count", report["prefetch_miss_count"])
print("frontend_exception_count", report["frontend_exception_count"])
if cand["miou"] < base["miou"] - 0.002:
    raise SystemExit("mIoU regression exceeds 0.002")
if cand["fmiou"] < base["fmiou"] - 0.002:
    raise SystemExit("f-mIoU regression exceeds 0.002")
if report["frontend_exception_count"] != 0:
    raise SystemExit("prefetch exceptions occurred")
PY
```

Expected: script exits `0`.

- [ ] **Step 7: Check major-class audit regression**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
classes = ["ceiling", "wall", "floor", "sofa", "blinds", "window", "lamp"]
baseline = Path("outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f/replica/classes_iou.json")
candidate = Path("outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_200f/replica/classes_iou.json")
base = json.loads(baseline.read_text())
cand = json.loads(candidate.read_text())
for label in classes:
    delta = cand[label] - base[label]
    print(f"{label}: baseline={base[label]:.4f} candidate={cand[label]:.4f} delta={delta:+.4f}")
    if delta < -0.01:
        raise SystemExit(f"{label} regressed by more than 0.01 IoU")
PY
```

Expected: script exits `0`.

- [ ] **Step 8: Generate concrete validation notes**

If validation passes, run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

classes = ["ceiling", "wall", "floor", "sofa", "blinds", "window", "lamp"]
baseline = Path("outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f")
candidate = Path("outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_200f")
base_results = json.loads((baseline / "replica/results.json").read_text())
cand_results = json.loads((candidate / "replica/results.json").read_text())
report = json.loads((candidate / "room0/run_report.json").read_text())
base_classes = json.loads((baseline / "replica/classes_iou.json").read_text())
cand_classes = json.loads((candidate / "replica/classes_iou.json").read_text())

lines = [
    "# Metric-Stable Parallel Scheduling Validation",
    "",
    "## Runs",
    "",
    f"- Baseline: `{baseline}`",
    f"- Candidate: `{candidate}`",
    "",
    "## Acceptance",
    "",
    f"- baseline mIoU: `{base_results['miou']:.12f}`",
    f"- candidate mIoU: `{cand_results['miou']:.12f}`",
    f"- mIoU delta: `{cand_results['miou'] - base_results['miou']:+.12f}`",
    f"- baseline f-mIoU: `{base_results['fmiou']:.12f}`",
    f"- candidate f-mIoU: `{cand_results['fmiou']:.12f}`",
    f"- f-mIoU delta: `{cand_results['fmiou'] - base_results['fmiou']:+.12f}`",
    f"- prefetch submitted: `{int(report['prefetch_submitted_count'])}`",
    f"- prefetch hit: `{int(report['prefetch_hit_count'])}`",
    f"- prefetch miss: `{int(report['prefetch_miss_count'])}`",
    f"- frontend exceptions: `{int(report['frontend_exception_count'])}`",
    "",
    "## Major Classes",
    "",
]
for label in classes:
    delta = float(cand_classes[label]) - float(base_classes[label])
    lines.append(
        f"- {label}: baseline `{float(base_classes[label]):.6f}`, "
        f"candidate `{float(cand_classes[label]):.6f}`, delta `{delta:+.6f}`"
    )

path = Path("docs/superpowers/experiments/2026-06-01-metric-stable-parallel-scheduling.md")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(path)
PY
```

Expected: `docs/superpowers/experiments/2026-06-01-metric-stable-parallel-scheduling.md` is written with concrete numeric values and no empty metric fields.

- [ ] **Step 9: Commit validation notes**

Run:

```bash
git add docs/superpowers/experiments/2026-06-01-metric-stable-parallel-scheduling.md
git commit -m "docs: record metric-stable scheduling validation"
```

Expected: commit succeeds after the validation note contains concrete numbers.

## Self-Review

- Spec coverage:
  - `FrameProposalBundle`: Task 1.
  - Pipeline bundle injection: Task 1.
  - Frame proposal prefetcher: Task 2.
  - Runner data flow and scheduling metrics: Task 3.
  - Association score parallelism: Task 4.
  - Depth refinement parallel config: Task 5.
  - Instrumentation p50/p95/max/outliers: Task 6.
  - 20-frame and 200-frame validation: Task 7.
- Scope check:
  - No multi-frame state commits.
  - No concurrent writes to TSDF, object maps, semantic memory, or dense surface.
  - No detector threshold or class-list changes.
- Type consistency:
  - `FrameProposalBundle` is imported from `src.pipelines.proposal_bundle`.
  - `FrameProposalPrefetcher` is imported from `src.pipelines.proposal_prefetch`.
  - `proposal_bundle` is the optional `Pipeline.process_frame()` argument.
  - Scheduling records live under `frame_metrics[*]["scheduling"]`.
  - Association parallel summary keys are `score_parallel_used_count` and `score_parallel_candidate_count_total`.
