# Replica YOLOWorld SAM Online Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all available Replica scenes with an online YOLOWorld + SAM2 frontend baseline, without precomputed proposal caches, while preserving the checkpointed fast-eval export and summary flow.

**Architecture:** Keep the existing checkpointed mapping/export runner and all-scene batch runner, but add an online baseline profile that uses `proposal.backend=sam2` and `anchor_frontend.backend=yoloworld`. Add a parallel frontend path for anchor-guided SAM: YOLOWorld runs as the independent anchor line, SAM runs as the dependent SAM line, and the fusion step waits for both; YOLOWorld never waits on SAM. Preserve the existing precomputed fast-eval path as the default.

**Tech Stack:** Python, YAML, `concurrent.futures.ThreadPoolExecutor`, pytest, existing `Pipeline`, `FrameProposalBundle`, `ObjectAnchorModule`, `AnchorGuidedSAMModule`, and checkpointed Replica runners.

---

## Evidence And Constraints

- User requirement: align with the previous YOLOWorld baseline and thresholds, not YOLOE.
- Use all Replica scenes selected in the previous all-scene plan:
  - `room0`, `room1`, `room2`, `office0`, `office1`, `office2`, `office3`, `office4`
- RGB-D data should come directly from `/home/ww/vv/dataset/Replica/<scene>`.
- Do not require precomputed proposal caches for the online baseline.
- Preserve the existing precomputed path and cache safety guard for precomputed experiments.
- YOLOWorld thresholds should match the current validated config:
  - `confidence_threshold: 0.2`
  - `max_detections: 128`
  - `iou_assign_threshold: 0.2`
  - `min_anchor_mask_coverage: 0.2`
  - `min_protected_anchor_confidence: 0.25`
  - `contained_max_anchor_coverage: 0.20`
  - `contained_min_proposal_coverage: 0.85`
  - `scale_compatible_min_anchor_coverage: 0.20`
  - `scale_compatible_min_bbox_iou: 0.10`
  - `weak_structure_min_proposal_coverage: 0.05`
  - `weak_structure_min_anchor_coverage: 0.20`
- Disable `anchor_frontend.supplemental_enabled` for this baseline so YOLOE is not mixed into the YOLOWorld baseline.
- Keep `--fast-eval` exports: reports, metrics, final semantic audit, and primary PLYs; no full debug artifacts.
- Execute with subagent-driven development when implementation starts:
  - Coding subagents: GPT-5.5, reasoning medium.
  - Review subagents: GPT-5.5, reasoning xhigh.

## File Structure

- Add: `src/pipelines/yoloworld_sam_frontend.py`
  - Owns the online YOLOWorld/SAM frontend scheduling helper.
  - Builds a `FrameProposalBundle` from YOLOWorld anchors, SAM2 proposals, and `AnchorGuidedSAMModule` fusion.
  - Records frontend stage timings.

- Modify: `src/pipelines/main_pipeline.py`
  - Reads `pipeline.yoloworld_sam_parallel_frontend_enabled`.
  - Calls the new helper only when the online anchor-guided SAM profile is enabled.
  - Leaves existing `_build_proposal_bundle()` behavior unchanged for precomputed and non-baseline modes.

- Add: `configs/replica_yoloworld_sam_online_baseline_4090.yaml`
  - Online YOLOWorld + SAM2 config for all Replica scenes.
  - Starts from `configs/room0_surface_gate_fast_high_iou_4090.yaml` semantics but changes frontend mode to online SAM2 + YOLOWorld and disables YOLOE supplemental anchors.

- Modify: `scripts/run_replica_all_scenes_fast_eval.py`
  - Adds `--online-yoloworld-sam`.
  - Uses the new config and `--proposal-backend sam2` in online mode.
  - Skips precomputed cache validation in online mode.
  - Keeps default behavior unchanged for existing precomputed fast-eval runs.

- Modify: `tests/test_pipeline.py`
  - Unit tests for YOLOWorld/SAM scheduling and pipeline integration.

- Modify: `tests/test_replica_all_scenes.py`
  - Unit tests for all-scene runner online mode command construction and dry-run behavior.

---

## Task 1: Add Parallel YOLOWorld/SAM Frontend Helper

**Files:**
- Add: `src/pipelines/yoloworld_sam_frontend.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing helper test**

Add this test to `tests/test_pipeline.py` near existing `FrameProposalBundle` tests:

```python
def test_yoloworld_sam_frontend_runs_sam_after_yolo_without_yolo_waiting(self, tmp_path):
    from src.pipelines.yoloworld_sam_frontend import build_yoloworld_sam_bundle

    events: list[str] = []
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
    frame = Frame(
        frame_id=0,
        source_frame_id=100,
        rgb=rgb,
        depth=depth,
        pose=np.eye(4, dtype=np.float64),
        intrinsics=intrinsics,
    )
    anchor = Anchor2D(3, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
    anchor_mask = np.zeros((8, 8), dtype=bool)
    anchor_mask[1:7, 1:7] = True
    anchor_proposal = Proposal2D(
        proposal_id=0,
        mask=anchor_mask,
        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
        area=int(anchor_mask.sum()),
        confidence=0.9,
        backend_name="anchor_box_primary",
        metadata={"anchor_id": 3, "anchor_class_name": "sofa"},
    )
    anchor_assignment = AnchorAssignment(0, 3, "sofa", 0.9, keepalive=True)
    sam_proposal = Proposal2D(
        proposal_id=11,
        mask=anchor_mask.copy(),
        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
        area=int(anchor_mask.sum()),
        confidence=1.0,
        backend_name="sam2",
    )

    class FakeAnchorFrontend:
        active_backend_name = "yoloworld"
        last_generation_timings = {}

        def generate_anchor_box_proposals(self, input_rgb):
            events.append("yolo_start")
            self.last_generation_timings = {"yoloworld_primary": 0.12, "anchor_merge": 0.0}
            events.append("yolo_end")
            return [anchor], [anchor_proposal], [anchor_assignment]

    class FakeProposal:
        active_backend_name = "sam2"

        def process_for_anchors(self, input_rgb, input_depth, anchors, frame=None):
            events.append("sam_start")
            assert events[:2] == ["yolo_start", "yolo_end"]
            assert [int(item.anchor_id) for item in anchors] == [3]
            events.append("sam_end")
            return [sam_proposal]

    anchor_guided_sam = AnchorGuidedSAMModule(
        {
            "enabled": True,
            "proposal_min_area": 1,
            "include_anchor_box_fallbacks": True,
            "include_unknown_residuals": True,
        }
    )

    bundle = build_yoloworld_sam_bundle(
        frame=frame,
        object_anchor=FakeAnchorFrontend(),
        proposal=FakeProposal(),
        anchor_guided_sam=anchor_guided_sam,
        collect_stage_timings=True,
    )

    assert events == ["yolo_start", "yolo_end", "sam_start", "sam_end"]
    assert bundle.frame_id == 0
    assert bundle.source_frame_id == 100
    assert bundle.proposal_source == "anchor_guided_sam"
    assert bundle.actual_backend == "sam2"
    assert len(bundle.anchors) == 1
    assert len(bundle.source_proposals) == 1
    assert len(bundle.raw_proposals) >= 1
    assert bundle.generation_timings["yoloworld_primary"] == pytest.approx(0.12)
    assert bundle.generation_timings["sam2_proposals"] >= 0.0
    assert bundle.generation_timings["anchor_guided_sam_fusion"] >= 0.0
    assert bundle.generation_timings["proposal_generation"] >= 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_yoloworld_sam_frontend_runs_sam_after_yolo_without_yolo_waiting -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipelines.yoloworld_sam_frontend'` or missing `build_yoloworld_sam_bundle`.

- [ ] **Step 3: Implement the helper**

Create `src/pipelines/yoloworld_sam_frontend.py`:

```python
"""Online YOLOWorld + SAM2 frontend scheduling for anchor-guided Replica runs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.core.data_structures import Frame
from src.pipelines.proposal_bundle import FrameProposalBundle


def _finite_timing(value: Any) -> float:
    try:
        timing = float(value)
    except (TypeError, ValueError):
        return 0.0
    if timing < 0.0 or not np.isfinite(timing):
        return 0.0
    return timing


def build_yoloworld_sam_bundle(
    *,
    frame: Frame,
    object_anchor: Any,
    proposal: Any,
    anchor_guided_sam: Any,
    collect_stage_timings: bool = False,
) -> FrameProposalBundle:
    """Build a frontend bundle with independent YOLOWorld and dependent SAM lines.

    YOLOWorld is submitted first and never waits on SAM. The SAM worker waits for
    YOLO anchors before calling `process_for_anchors`, then fusion waits for both.
    """
    start = time.perf_counter()

    def run_yolo():
        return object_anchor.generate_anchor_box_proposals(frame.rgb)

    def run_sam_after_yolo(yolo_future):
        anchors, _anchor_box_proposals, _anchor_assignments = yolo_future.result()
        sam_start = time.perf_counter()
        sam_proposals = proposal.process_for_anchors(
            frame.rgb,
            frame.depth,
            list(anchors),
            frame=frame,
        )
        return list(sam_proposals), max(0.0, time.perf_counter() - sam_start)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="yoloworld-sam-front") as executor:
        yolo_future = executor.submit(run_yolo)
        sam_future = executor.submit(run_sam_after_yolo, yolo_future)
        anchors, anchor_box_proposals, anchor_assignments = yolo_future.result()
        sam_proposals, sam_elapsed = sam_future.result()

    fusion_start = time.perf_counter()
    proposals, anchor_assignments, anchor_guided_sam_summary = anchor_guided_sam.build_proposals(
        frame=frame,
        anchors=list(anchors),
        sam_proposals=list(sam_proposals),
    )
    fusion_elapsed = max(0.0, time.perf_counter() - fusion_start)

    generation_timings = {}
    if collect_stage_timings:
        generation_timings.update(
            {
                str(key): _finite_timing(value)
                for key, value in dict(getattr(object_anchor, "last_generation_timings", {}) or {}).items()
            }
        )
        generation_timings["sam2_proposals"] = float(sam_elapsed)
        generation_timings["anchor_guided_sam_fusion"] = float(fusion_elapsed)
        generation_timings["proposal_generation"] = max(0.0, time.perf_counter() - start)

    if not proposals and anchor_box_proposals:
        proposals = list(anchor_box_proposals)

    return FrameProposalBundle(
        frame_id=int(frame.frame_id),
        source_frame_id=frame.source_frame_id,
        source_proposals=tuple(sam_proposals),
        raw_proposals=tuple(proposals),
        anchors=tuple(anchors),
        anchor_assignments=tuple(anchor_assignments),
        proposal_source="anchor_guided_sam",
        anchor_guided_sam_summary=dict(anchor_guided_sam_summary),
        generation_timings=generation_timings,
        actual_backend=str(getattr(proposal, "active_backend_name", "")),
    )
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_yoloworld_sam_frontend_runs_sam_after_yolo_without_yolo_waiting -q
```

Expected: PASS.

---

## Task 2: Wire Parallel Frontend Into Pipeline

**Files:**
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing pipeline integration test**

Add this test to `tests/test_pipeline.py` near the anchor-guided SAM pipeline tests:

```python
def test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled(self, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: placeholder
  min_mask_area: 1
anchor_frontend:
  enabled: true
  anchor_primary_mode: true
  use_sam_intersection_proposals: false
  proposal_min_area: 1
anchor_guided_sam:
  enabled: true
  proposal_min_area: 1
pipeline:
  verbose: false
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
runtime_vis:
  enabled: false
depth_refinement:
  min_mask_area_after_refine: 1
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
  provisional:
    enabled: false
semantic_memory:
  backend: placeholder
""",
        encoding="utf-8",
    )
    pipe = Pipeline(config_path=str(config_path))
    events: list[str] = []
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True

    class FakeAnchor:
        enabled = True
        anchor_primary_mode = True
        use_sam_intersection_proposals = False
        last_generation_timings = {}
        active_backend_name = "yoloworld"

        def generate_anchor_box_proposals(self, rgb):
            events.append("yolo")
            self.last_generation_timings = {"yoloworld_primary": 0.01, "anchor_merge": 0.0}
            anchor = Anchor2D(1, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.95)
            proposal = Proposal2D(
                proposal_id=1,
                mask=mask,
                bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                area=int(mask.sum()),
                confidence=0.95,
                backend_name="anchor_box_primary",
                metadata={"anchor_id": 1, "anchor_class_name": "sofa"},
            )
            assignment = AnchorAssignment(1, 1, "sofa", 0.95, keepalive=True)
            return [anchor], [proposal], [assignment]

    class FakeSAM:
        active_backend_name = "sam2"

        def process_for_anchors(self, rgb, depth, anchors, frame=None):
            events.append("sam")
            assert [int(anchor.anchor_id) for anchor in anchors] == [1]
            return [
                Proposal2D(
                    proposal_id=5,
                    mask=mask.copy(),
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=1.0,
                    backend_name="sam2",
                )
            ]

    pipe.object_anchor = FakeAnchor()
    pipe.proposal = FakeSAM()

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
    pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=42)

    assert events == ["yolo", "sam"]
    assert pipe.last_frame_debug["proposal_source"] == "anchor_guided_sam"
    assert pipe.last_frame_debug["frontend_actual_backend"] == "sam2"
    assert pipe.last_frame_debug["stage_timings"]["yoloworld_primary"] == pytest.approx(0.01)
    assert "sam2_proposals" in pipe.last_frame_debug["stage_timings"]
    assert "anchor_guided_sam_fusion" in pipe.last_frame_debug["stage_timings"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled -q
```

Expected: FAIL because `pipeline.yoloworld_sam_parallel_frontend_enabled` is ignored.

- [ ] **Step 3: Implement pipeline wiring**

Modify imports in `src/pipelines/main_pipeline.py`:

```python
from src.pipelines.yoloworld_sam_frontend import build_yoloworld_sam_bundle
```

In `Pipeline.__init__`, after `pipeline_cfg = self.config.get("pipeline", {})`, add:

```python
        self.yoloworld_sam_parallel_frontend_enabled = bool(
            pipeline_cfg.get("yoloworld_sam_parallel_frontend_enabled", False)
        )
```

Add this helper method before `_build_proposal_bundle`:

```python
    def _should_use_parallel_yoloworld_sam_frontend(self) -> bool:
        return bool(
            self.yoloworld_sam_parallel_frontend_enabled
            and self.object_anchor.enabled
            and getattr(self.object_anchor, "anchor_primary_mode", False)
            and getattr(self.anchor_guided_sam, "enabled", False)
        )
```

At the start of `_build_proposal_bundle`, before the current `with self._timed_stage("proposal_generation"):` block, add:

```python
        if self._should_use_parallel_yoloworld_sam_frontend():
            return build_yoloworld_sam_bundle(
                frame=frame,
                object_anchor=self.object_anchor,
                proposal=self.proposal,
                anchor_guided_sam=self.anchor_guided_sam,
                collect_stage_timings=self.collect_stage_timings,
            )
```

- [ ] **Step 4: Run the pipeline integration test to verify it passes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled -q
```

Expected: PASS.

- [ ] **Step 5: Run existing bundle tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_process_frame_consumes_proposal_bundle_without_calling_frontend \
  tests/test_pipeline.py::TestPipeline::test_build_proposal_bundle_records_top_level_proposal_generation_timing \
  -q
```

Expected: PASS.

---

## Task 3: Add Online YOLOWorld SAM Baseline Config

**Files:**
- Add: `configs/replica_yoloworld_sam_online_baseline_4090.yaml`
- Modify: `tests/test_replica_all_scenes.py`

- [ ] **Step 1: Write the failing config test**

Add this test to `tests/test_replica_all_scenes.py`:

```python
def test_online_yoloworld_sam_config_uses_sam2_yoloworld_and_no_yoloe() -> None:
    config_path = Path("configs/replica_yoloworld_sam_online_baseline_4090.yaml")
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "sam2"
    assert config["anchor_frontend"]["backend"] == "yoloworld"
    assert config["anchor_frontend"]["supplemental_enabled"] is False
    assert config["anchor_frontend"]["confidence_threshold"] == pytest.approx(0.2)
    assert config["anchor_frontend"]["max_detections"] == 128
    assert config["anchor_frontend"]["iou_assign_threshold"] == pytest.approx(0.2)
    assert config["anchor_frontend"]["min_anchor_mask_coverage"] == pytest.approx(0.2)
    assert config["anchor_frontend"]["min_protected_anchor_confidence"] == pytest.approx(0.25)
    assert config["anchor_frontend"]["contained_max_anchor_coverage"] == pytest.approx(0.20)
    assert config["anchor_frontend"]["contained_min_proposal_coverage"] == pytest.approx(0.85)
    assert config["anchor_frontend"]["scale_compatible_min_anchor_coverage"] == pytest.approx(0.20)
    assert config["anchor_frontend"]["scale_compatible_min_bbox_iou"] == pytest.approx(0.10)
    assert config["anchor_guided_sam"]["enabled"] is True
    assert config["anchor_frontend"]["anchor_primary_mode"] is True
    assert config["anchor_frontend"]["use_sam_intersection_proposals"] is False
    assert config["pipeline"]["yoloworld_sam_parallel_frontend_enabled"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py::test_online_yoloworld_sam_config_uses_sam2_yoloworld_and_no_yoloe -q
```

Expected: FAIL because the config file does not exist.

- [ ] **Step 3: Create the config**

Create `configs/replica_yoloworld_sam_online_baseline_4090.yaml` by copying `configs/room0_surface_gate_fast_high_iou_4090.yaml`, then make these exact changes:

```yaml
proposal:
  backend: sam2
  min_mask_area: 100
  max_proposals: 50
  confidence_threshold: 0.5
  precomputed:
    cache_dir: ''
    manifest_path: ''
    strict: true

anchor_frontend:
  enabled: true
  use_boxes_as_primary_proposals: true
  use_sam_intersection_proposals: false
  anchor_primary_mode: true
  backend: yoloworld
  device: cuda:0
  model_path: /home/ww/vv/DualMapV1/model/yolov8l-world.pt
  python_executable: /home/ww/miniconda3/envs/oviovo/bin/python
  confidence_threshold: 0.2
  max_detections: 128
  iou_assign_threshold: 0.2
  min_anchor_mask_coverage: 0.2
  min_protected_anchor_confidence: 0.25
  contained_max_anchor_coverage: 0.20
  contained_min_proposal_coverage: 0.85
  scale_compatible_min_anchor_coverage: 0.20
  scale_compatible_min_bbox_iou: 0.10
  weak_structure_overlap_enabled: true
  weak_structure_min_proposal_coverage: 0.05
  weak_structure_min_anchor_coverage: 0.20
  supplemental_enabled: false

anchor_guided_sam:
  enabled: true
  proposal_min_area: 25
  min_proposal_anchor_coverage_for_label: 0.70
  min_anchor_proposal_coverage_for_label: 0.20
  contained_residual_enabled: true
  contained_min_proposal_coverage: 0.85
  contained_max_anchor_coverage: 0.35
  clip_to_anchor_box: true
  include_unknown_residuals: true
  include_anchor_box_fallbacks: true
  max_sam_proposals_per_anchor: 4

pipeline:
  yoloworld_sam_parallel_frontend_enabled: true
```

Keep the existing association, object update, semantic memory, structural overlay, fast timing, and export-related settings from `room0_surface_gate_fast_high_iou_4090.yaml`.

- [ ] **Step 4: Run the config test to verify it passes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py::test_online_yoloworld_sam_config_uses_sam2_yoloworld_and_no_yoloe -q
```

Expected: PASS.

---

## Task 4: Add Online Baseline Mode To All-Scene Runner

**Files:**
- Modify: `scripts/run_replica_all_scenes_fast_eval.py`
- Modify: `tests/test_replica_all_scenes.py`

- [ ] **Step 1: Write the failing runner test**

Add this test to `tests/test_replica_all_scenes.py`:

```python
def test_online_yoloworld_sam_mode_uses_sam2_config_and_skips_cache_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "REPO_ROOT", repo_root)
    subprocess_calls = []
    monkeypatch.setattr(
        run_replica_all_scenes_fast_eval.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    def fail_if_precomputed_cache_validation_runs(**kwargs):
        raise AssertionError("online SAM2 mode must not validate precomputed caches")

    monkeypatch.setattr(
        run_replica_all_scenes_fast_eval,
        "validate_precomputed_cache_for_scene",
        fail_if_precomputed_cache_validation_runs,
    )

    _make_executable(repo_root / "bin" / "python")
    for path in [
        repo_root / "scripts" / "run_room0_checkpointed_eval.py",
        repo_root / "configs" / "replica_yoloworld_sam_online_baseline_4090.yaml",
        repo_root / "Replica" / "room1",
        repo_root / "labels" / "room1.txt",
        repo_root / "Original" / "room_1" / "habitat" / "mesh_semantic.ply",
        repo_root / "Original" / "room_1" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "online_batch",
            "--scenes",
            "room1",
            "--dry-run",
            "--online-yoloworld-sam",
            "--python-executable",
            "bin/python",
            "--runner-script",
            "scripts/run_room0_checkpointed_eval.py",
            "--dataset-root-base",
            "Replica",
            "--gt-original-base",
            "Original",
            "--gt-label-dir",
            "labels",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert subprocess_calls == []
    assert "--proposal-backend sam2" in captured.out
    assert "--scene-name room1" in captured.out
    assert "replica_yoloworld_sam_online_baseline_4090.yaml" in captured.out
    assert "--proposal-cache-manifest" not in captured.out
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py::test_online_yoloworld_sam_mode_uses_sam2_config_and_skips_cache_validation -q
```

Expected: FAIL because `--online-yoloworld-sam` does not exist.

- [ ] **Step 3: Implement online mode CLI**

Modify `scripts/run_replica_all_scenes_fast_eval.py`:

Add constant:

```python
DEFAULT_ONLINE_CONFIG_PATH = REPO_ROOT / "configs" / "replica_yoloworld_sam_online_baseline_4090.yaml"
```

Change parser defaults:

```python
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--proposal-backend", default=None)
    parser.add_argument("--online-yoloworld-sam", action="store_true")
```

In `main`, after `args = parse_args(argv)`, resolve config/backend:

```python
    online_yoloworld_sam = bool(args.online_yoloworld_sam)
    default_config = DEFAULT_ONLINE_CONFIG_PATH if online_yoloworld_sam else DEFAULT_CONFIG_PATH
    proposal_backend = args.proposal_backend or ("sam2" if online_yoloworld_sam else DEFAULT_PROPOSAL_BACKEND)
```

Replace the existing `config_path` and command arguments with:

```python
    config_path = _repo_relative(args.config_path or default_config)
```

When building commands, pass:

```python
proposal_backend=proposal_backend
```

When validating precomputed cache, change:

```python
        if proposal_backend == "precomputed":
            errors.extend(
                validate_precomputed_cache_for_scene(
                    scene=scene,
                    config_path=config_path,
                    override=proposal_cache,
                    num_frames=args.num_frames,
                    frame_stride=args.frame_stride,
                )
            )
```

Do not call cache validation for `proposal_backend == "sam2"`.

- [ ] **Step 4: Run the online mode test to verify it passes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py::test_online_yoloworld_sam_mode_uses_sam2_config_and_skips_cache_validation -q
```

Expected: PASS.

- [ ] **Step 5: Run existing precomputed runner tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_replica_all_scenes.py::test_precomputed_cache_manifest_scene_mismatch_fails_validation_for_non_room_scene \
  tests/test_replica_all_scenes.py::test_per_scene_cache_manifest_template_expands_and_is_passed_to_runner \
  tests/test_replica_all_scenes.py::test_per_scene_cache_dir_template_expands_and_passes_derived_manifest_to_runner \
  -q
```

Expected: PASS. This confirms the precomputed cache guard still works.

---

## Task 5: Focused Test Suite And Online Dry-Run

**Files:** no code edits expected unless tests fail.

- [ ] **Step 1: Run focused pipeline tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_yoloworld_sam_frontend_runs_sam_after_yolo_without_yolo_waiting \
  tests/test_pipeline.py::TestPipeline::test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled \
  tests/test_pipeline.py::TestPipeline::test_process_frame_consumes_proposal_bundle_without_calling_frontend \
  tests/test_pipeline.py::TestPipeline::test_build_proposal_bundle_records_top_level_proposal_generation_timing \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run focused all-scene runner tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_replica_all_scenes.py::test_online_yoloworld_sam_config_uses_sam2_yoloworld_and_no_yoloe \
  tests/test_replica_all_scenes.py::test_online_yoloworld_sam_mode_uses_sam2_config_and_skips_cache_validation \
  tests/test_replica_all_scenes.py::test_precomputed_cache_manifest_scene_mismatch_fails_validation_for_non_room_scene \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run full focused file tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py -q
```

Expected: PASS.

- [ ] **Step 4: Run online all-scene dry-run**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_replica_all_scenes_fast_eval.py \
  --batch-name 20260603_replica_yoloworld_sam_online_s10_200f \
  --online-yoloworld-sam \
  --dry-run
```

Expected:

- Exit `0`.
- Eight selected scenes are printed.
- Commands include:
  - `--config-path .../configs/replica_yoloworld_sam_online_baseline_4090.yaml`
  - `--proposal-backend sam2`
  - `--fast-eval`
  - `--num-frames 200`
  - `--frame-stride 10`
- Commands do not include:
  - `--proposal-cache-manifest`
  - `--proposal-cache-dir`

---

## Task 6: One-Scene Online Smoke

**Files:** no code edits expected unless smoke exposes a code bug.

- [ ] **Step 1: Run room1 1-frame online smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_replica_all_scenes_fast_eval.py \
  --batch-name 20260603_replica_yoloworld_sam_online_smoke \
  --online-yoloworld-sam \
  --scenes room1 \
  --num-frames 1 \
  --frame-stride 10
```

Expected:

- Exit `0`.
- `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_smoke_room1_s10_1f_fast_eval/status.json` has `"status": "complete"`.
- `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_smoke_room1_s10_1f_fast_eval/room1/mapping_state.pkl` exists.
- `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_smoke_room1_s10_1f_fast_eval/room1/frame_metrics.jsonl` has 1 line.
- `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_smoke_room1_s10_1f_fast_eval/room1/run_report.md` exists.
- `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_smoke_room1_s10_1f_fast_eval/replica/results.json` exists.
- Primary PLYs exist under the scene exports directory.

- [ ] **Step 2: Inspect frontend scheduling metrics**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/tmp_validation/20260603_replica_yoloworld_sam_online_smoke_room1_s10_1f_fast_eval/room1")
record = json.loads((root / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
print(json.dumps({
    "actual_backend": record.get("actual_backend"),
    "frontend_source": record.get("scheduling", {}).get("frontend_source"),
    "frontend_stage": record.get("frontend_stage", {}),
    "stage_timings": record.get("stage_timings", {}),
}, indent=2, sort_keys=True))
PY
```

Expected:

- `actual_backend` is `sam2`.
- `stage_timings` includes `yoloworld_primary`, `sam2_proposals`, and `anchor_guided_sam_fusion`.
- `frontend_stage.anchor_voted_proposal_count_total` is not required here because this is per-frame JSON, but `frontend_stage.anchor_guided_sam` should show anchor-guided SAM fields.

If the smoke fails due missing local YOLOWorld or SAM2 model files, report the exact missing path. Do not switch to precomputed cache.

---

## Task 7: Full Online All-Scene Run And Summary

**Files:** runtime outputs only unless the full run exposes a code bug.

- [ ] **Step 1: Run all selected scenes online**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_replica_all_scenes_fast_eval.py \
  --batch-name 20260603_replica_yoloworld_sam_online_s10_200f \
  --online-yoloworld-sam \
  --num-frames 200 \
  --frame-stride 10
```

Expected:

- Scenes run sequentially by default.
- No precomputed cache validation errors.
- Each scene writes:
  - `status.json`
  - `<scene>/mapping_state.pkl`
  - `<scene>/frame_metrics.jsonl`
  - `<scene>/run_report.md`
  - `replica/results.json`
  - primary PLY exports

- [ ] **Step 2: Summarize full online batch**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/summarize_replica_all_scenes.py \
  --batch-name 20260603_replica_yoloworld_sam_online_s10_200f
```

Expected:

- Exit `0`.
- Writes:
  - `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_s10_200f/summary.json`
  - `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_s10_200f/summary.md`

- [ ] **Step 3: Report final results**

Report:

- Per-scene metrics table from `summary.md`.
- Macro `mIoU`, `f-mIoU`, `mAcc`, `f-mAcc`.
- Mean TPF.
- Total mapping time.
- Total export/eval time.
- Failed/skipped scenes, if any.
- Output paths for batch summary and scene roots.

---

## Risks And Mitigations

- **Risk: online YOLOWorld + SAM2 is much slower than precomputed fast-eval.**
  - Mitigation: keep sequential scene execution and resumable status checks; run 1-frame smoke first.

- **Risk: running YOLOWorld and SAM2 on the same GPU concurrently can increase memory pressure.**
  - Mitigation: the first implementation uses two frontend workers only for the active frame bundle. If CUDA OOM happens, report it and add a config fallback to disable `pipeline.yoloworld_sam_parallel_frontend_enabled` while preserving online SAM2.

- **Risk: YOLOWorld checkpoint or SAM2 checkpoint path is missing locally.**
  - Mitigation: smoke reports exact missing path. Do not fall back to precomputed caches for this baseline.

- **Risk: current all-scene runner experiment suffix still says `fast_eval`.**
  - Mitigation: keep suffix because export profile remains fast-eval; use a batch name containing `yoloworld_sam_online`.

- **Risk: helper uses SAM2 `process_for_anchors` even though current SAM2 backend ignores anchors.**
  - Mitigation: this preserves the desired dependency contract and future-proofs anchor-prompt backends while keeping current SAM2 behavior valid.

---

## Completion Criteria

- New online baseline config exists and disables YOLOE supplemental anchors.
- Pipeline can build a YOLOWorld/SAM2 anchor-guided bundle with YOLO first and SAM dependent on YOLO anchors.
- Existing precomputed path remains tested and unchanged.
- All-scene runner supports `--online-yoloworld-sam`.
- Online all-scene dry-run passes without precomputed cache errors.
- At least one non-room online smoke completes.
- Full online all-scene run either completes with summary or reports explicit runtime blockers such as missing model files or CUDA OOM.
