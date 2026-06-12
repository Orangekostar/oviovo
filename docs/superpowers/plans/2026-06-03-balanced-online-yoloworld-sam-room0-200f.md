# Balanced Online YOLOWorld SAM Room0 200f Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new Balanced online OVI-OVO profile that runs YOLOWorld anchor-box proposals on most frames, runs full-frame SAM2 + anchor-guided fusion at a fixed low frequency, then benchmark room0 200f for accuracy and timing.

**Architecture:** Keep the existing full online YOLOWorld+SAM2 baseline unchanged. Add a scheduler inside the YOLOWorld/SAM frontend helper so non-SAM frames reuse `ObjectAnchorModule.generate_anchor_box_proposals()` and SAM frames reuse the existing `ProposalModule.process_for_anchors()` plus `AnchorGuidedSAMModule.build_proposals()`. Add a separate config that enables this scheduler and disables prefetch so frame-id-based SAM cadence is deterministic and easy to audit.

**Tech Stack:** Python, YAML, pytest, existing `Pipeline`, `FrameProposalBundle`, `ObjectAnchorModule`, `AnchorGuidedSAMModule`, SAM2 proposal backend, and `scripts/run_room0_checkpointed_eval.py`.

---

## Evidence And Constraints

- Current online baseline runs full SAM2 on every frame through `src/pipelines/yoloworld_sam_frontend.py`.
- `AnchorGuidedSAMModule.build_proposals(frame, anchors, sam_proposals=[])` already falls back to anchor boxes when configured with `include_anchor_box_fallbacks: true`.
- `proposal_source="anchor_box_primary"` is the existing coarse-proposal path and preserves metadata propagation through runtime vis and depth refinement.
- `async_refinement.enabled` can trigger extra SAM2 work later in `Pipeline.process_frame`; the Balanced config must keep it disabled unless a later experiment explicitly measures async refinement.
- `structural_overlay` may lack SAM source proposals on skipped frames; this is acceptable for the speed experiment and is already recorded as skipped by the loader.
- Target room0 test: 200 processed frames, stride 10, fast-eval outputs, stage timing summaries, and final semantic metrics.

## File Structure

- Modify: `src/pipelines/yoloworld_sam_frontend.py`
  - Add Balanced scheduler options and anchor-only bundle builder behavior.
  - Record whether SAM ran in `generation_timings` and `anchor_guided_sam_summary`.

- Modify: `src/pipelines/main_pipeline.py`
  - Read `pipeline.yoloworld_sam_balanced_frontend_enabled` and `pipeline.yoloworld_sam_full_frame_interval`.
  - Pass scheduler options to `build_yoloworld_sam_bundle()`.

- Add: `configs/replica_yoloworld_sam_balanced_online_4090.yaml`
  - New independent Balanced profile derived from the online baseline.
  - Uses `proposal.backend=sam2`, `anchor_frontend.backend=yoloworld`, `anchor_guided_sam.enabled=true`.
  - Enables low-frequency SAM2 with interval 10 and disables proposal prefetch.

- Modify: `tests/test_pipeline.py`
  - Add unit coverage for anchor-only skip frames.
  - Add integration coverage that frame 0 runs SAM and frame 1 skips SAM with interval 10.

## Task 1: Add Balanced Frontend Scheduler Tests

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add helper-level skip-frame test**

Add a test near the existing YOLOWorld/SAM frontend tests:

```python
    def test_yoloworld_sam_frontend_anchor_only_frame_skips_sam_and_uses_anchor_box_primary(self):
        from src.pipelines.yoloworld_sam_frontend import build_yoloworld_sam_bundle

        events: list[str] = []
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        frame = Frame(
            frame_id=1,
            source_frame_id=101,
            rgb=rgb,
            depth=depth,
            pose=np.eye(4, dtype=np.float64),
            intrinsics=intrinsics,
        )
        anchor = Anchor2D(3, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.9)
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        anchor_proposal = Proposal2D(
            proposal_id=0,
            mask=mask,
            bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
            backend_name="anchor_box_primary",
            metadata={"anchor_id": 3, "anchor_class_name": "sofa", "anchor_keepalive": True},
        )
        anchor_assignment = AnchorAssignment(0, 3, "sofa", 0.9, keepalive=True)

        class FakeAnchorFrontend:
            active_backend_name = "yoloworld"
            last_generation_timings = {}

            def generate_anchor_box_proposals(self, input_rgb):
                events.append("yolo")
                self.last_generation_timings = {"yoloworld_primary": 0.05}
                return [anchor], [anchor_proposal], [anchor_assignment]

        class FakeProposal:
            active_backend_name = "sam2"

            def process_for_anchors(self, input_rgb, input_depth, anchors, frame=None):
                raise AssertionError("SAM should not run on anchor-only Balanced frames")

        bundle = build_yoloworld_sam_bundle(
            frame=frame,
            object_anchor=FakeAnchorFrontend(),
            proposal=FakeProposal(),
            anchor_guided_sam=AnchorGuidedSAMModule({"enabled": True}),
            collect_stage_timings=True,
            run_sam=False,
        )

        assert events == ["yolo"]
        assert bundle.proposal_source == "anchor_box_primary"
        assert bundle.actual_backend == "sam2"
        assert bundle.source_proposals == ()
        assert bundle.raw_proposals == (anchor_proposal,)
        assert bundle.anchor_assignments == (anchor_assignment,)
        assert bundle.generation_timings["yoloworld_primary"] == pytest.approx(0.05)
        assert bundle.generation_timings["sam2_full_frame_ran"] == 0.0
        assert bundle.generation_timings["proposal_generation"] >= 0.0
        assert bundle.anchor_guided_sam_summary["skip_reason"] == "balanced_anchor_only_frame"
```

- [ ] **Step 2: Add pipeline interval integration test**

Add a test near `test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled`:

```python
    def test_pipeline_balanced_yoloworld_sam_runs_sam_only_on_interval_frames(self, tmp_path):
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
  yoloworld_sam_balanced_frontend_enabled: true
  yoloworld_sam_full_frame_interval: 10
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
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        sam_calls: list[int] = []

        class FakeAnchor:
            enabled = True
            anchor_primary_mode = True
            use_sam_intersection_proposals = False
            last_generation_timings = {}
            active_backend_name = "yoloworld"

            def generate_anchor_box_proposals(self, rgb):
                self.last_generation_timings = {"yoloworld_primary": 0.01}
                anchor = Anchor2D(1, np.array([1, 1, 7, 7], dtype=np.float32), "sofa", 0.95)
                proposal = Proposal2D(
                    proposal_id=1,
                    mask=mask,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    area=int(mask.sum()),
                    confidence=0.95,
                    backend_name="anchor_box_primary",
                    metadata={"anchor_id": 1, "anchor_class_name": "sofa", "anchor_keepalive": True},
                )
                assignment = AnchorAssignment(1, 1, "sofa", 0.95, keepalive=True)
                return [anchor], [proposal], [assignment]

        class FakeSAM:
            active_backend_name = "sam2"

            def process_for_anchors(self, rgb, depth, anchors, frame=None):
                sam_calls.append(int(frame.frame_id))
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

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=100)
        first_debug = dict(pipe.last_frame_debug)
        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics, source_frame_id=110)
        second_debug = dict(pipe.last_frame_debug)

        assert sam_calls == [0]
        assert first_debug["proposal_source"] == "anchor_guided_sam"
        assert first_debug["stage_timings"]["sam2_full_frame_ran"] == pytest.approx(1.0)
        assert second_debug["proposal_source"] == "anchor_box_primary"
        assert second_debug["stage_timings"]["sam2_full_frame_ran"] == pytest.approx(0.0)
        assert "sam2_proposals" not in second_debug["stage_timings"]
        assert second_debug["anchor_guided_sam"]["skip_reason"] == "balanced_anchor_only_frame"
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_yoloworld_sam_frontend_anchor_only_frame_skips_sam_and_uses_anchor_box_primary tests/test_pipeline.py::TestPipeline::test_pipeline_balanced_yoloworld_sam_runs_sam_only_on_interval_frames -q
```

Expected: fail because `build_yoloworld_sam_bundle()` has no `run_sam` parameter and the pipeline has no Balanced scheduler.

## Task 2: Implement Balanced Frontend Scheduler

**Files:**
- Modify: `src/pipelines/yoloworld_sam_frontend.py`
- Modify: `src/pipelines/main_pipeline.py`

- [ ] **Step 1: Add `run_sam` support to frontend helper**

Change `build_yoloworld_sam_bundle()` signature to include:

```python
    run_sam: bool = True,
```

Then implement:

```python
    anchors, anchor_box_proposals, anchor_assignments = object_anchor.generate_anchor_box_proposals(frame.rgb)
    if not run_sam:
        generation_timings = {}
        if collect_stage_timings:
            generation_timings.update(
                {
                    str(key): _finite_timing(value)
                    for key, value in dict(getattr(object_anchor, "last_generation_timings", {}) or {}).items()
                }
            )
            generation_timings["sam2_full_frame_ran"] = 0.0
            generation_timings["proposal_generation"] = max(0.0, time.perf_counter() - start)
        return FrameProposalBundle(
            frame_id=int(frame.frame_id),
            source_frame_id=frame.source_frame_id,
            source_proposals=tuple(),
            raw_proposals=tuple(anchor_box_proposals),
            anchors=tuple(anchors),
            anchor_assignments=tuple(anchor_assignments),
            proposal_source="anchor_box_primary",
            anchor_guided_sam_summary={
                "enabled": bool(getattr(anchor_guided_sam, "enabled", False)),
                "anchor_count": int(len(anchors)),
                "source_sam_proposal_count": 0,
                "output_proposal_count": int(len(anchor_box_proposals)),
                "anchored_proposal_count": int(len(anchor_assignments)),
                "unknown_residual_count": 0,
                "dropped_proposal_count": 0,
                "matched_sam_proposal_count": 0,
                "mean_sam_candidates_per_anchor": 0.0,
                "semantic_blocked_residual_count": 0,
                "assignment_count": int(len(anchor_assignments)),
                "skip_reason": "balanced_anchor_only_frame",
            },
            generation_timings=generation_timings,
            actual_backend=str(getattr(proposal, "active_backend_name", "")),
        )
```

For the existing SAM path, add:

```python
        generation_timings["sam2_full_frame_ran"] = 1.0
```

- [ ] **Step 2: Add interval scheduling in pipeline**

In `Pipeline.__init__`, read:

```python
        self.yoloworld_sam_balanced_frontend_enabled = bool(
            pipeline_cfg.get("yoloworld_sam_balanced_frontend_enabled", False)
        )
        self.yoloworld_sam_full_frame_interval = max(
            1,
            int(pipeline_cfg.get("yoloworld_sam_full_frame_interval", 1) or 1),
        )
```

Add:

```python
    def _should_run_balanced_sam(self, frame: Frame) -> bool:
        if not self.yoloworld_sam_balanced_frontend_enabled:
            return True
        return int(frame.frame_id) % int(self.yoloworld_sam_full_frame_interval) == 0
```

Pass `run_sam=self._should_run_balanced_sam(frame)` when calling `build_yoloworld_sam_bundle()`.

- [ ] **Step 3: Run targeted tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_yoloworld_sam_frontend_anchor_only_frame_skips_sam_and_uses_anchor_box_primary tests/test_pipeline.py::TestPipeline::test_pipeline_balanced_yoloworld_sam_runs_sam_only_on_interval_frames tests/test_pipeline.py::TestPipeline::test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled -q
```

Expected: all pass.

## Task 3: Add Balanced Config

**Files:**
- Add: `configs/replica_yoloworld_sam_balanced_online_4090.yaml`

- [ ] **Step 1: Create config from online baseline**

Copy `configs/replica_yoloworld_sam_online_baseline_4090.yaml` to `configs/replica_yoloworld_sam_balanced_online_4090.yaml`, then set:

```yaml
pipeline:
  log_interval: 1
  verbose: false
  collect_stage_timings: true
  proposal_prefetch_enabled: false
  proposal_prefetch_wait_timeout_sec: -1.0
  yoloworld_sam_parallel_frontend_enabled: true
  yoloworld_sam_balanced_frontend_enabled: true
  yoloworld_sam_full_frame_interval: 10
```

Confirm `async_refinement` is absent or disabled; if a section exists, set:

```yaml
async_refinement:
  enabled: false
```

- [ ] **Step 2: Validate YAML parses**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import yaml
from pathlib import Path
path = Path("configs/replica_yoloworld_sam_balanced_online_4090.yaml")
payload = yaml.safe_load(path.read_text())
assert payload["pipeline"]["yoloworld_sam_balanced_frontend_enabled"] is True
assert payload["pipeline"]["yoloworld_sam_full_frame_interval"] == 10
assert payload["pipeline"]["proposal_prefetch_enabled"] is False
print("balanced config ok")
PY
```

Expected: prints `balanced config ok`.

## Task 4: Run Tests And Room0 200f Benchmark

**Files:**
- No code edits expected.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_yoloworld_sam_frontend_anchor_only_frame_skips_sam_and_uses_anchor_box_primary tests/test_pipeline.py::TestPipeline::test_pipeline_balanced_yoloworld_sam_runs_sam_only_on_interval_frames tests/test_pipeline.py::TestPipeline::test_pipeline_uses_parallel_yoloworld_sam_frontend_when_enabled -q
```

Expected: all pass.

- [ ] **Step 2: Run room0 200f fast-eval**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_room0_checkpointed_eval.py \
  --mode build-export \
  --scene-name room0 \
  --experiment-name room0_balanced_yoloworld_sam_s10_200f \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/replica_yoloworld_sam_balanced_online_4090.yaml \
  --output-root outputs/tmp_validation \
  --num-frames 200 \
  --frame-stride 10 \
  --fast-eval \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-version 2.1 \
  --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --points-per-side 16 \
  --max-proposals 50 \
  --confidence-threshold 0.5 \
  --min-mask-area 100 \
  --stability-score-th 0.95 \
  --nms-iou-th 0.8 \
  --min-mask-region-area 100 \
  --quiet
```

Expected: `outputs/tmp_validation/room0_balanced_yoloworld_sam_s10_200f/status.json` reports `complete`.

- [ ] **Step 3: Summarize timing and accuracy**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/tmp_validation/room0_balanced_yoloworld_sam_s10_200f")
scene = root / "room0"
mapping = json.loads((root / "mapping_timer_result.json").read_text())
export = json.loads((root / "export_eval_timer_result.json").read_text())
report = json.loads((scene / "run_report.json").read_text())
timings = report.get("stage_timing_summary", {})
print("mapping_sec", mapping["mapping_loop_sec_excluding_init_and_final_outputs"])
print("sec_per_frame", mapping["sec_per_frame"])
print("export_eval_sec", export["export_eval_sec"])
print("miou", export["miou"])
print("fmiou", export["fmiou"])
print("macc", export["macc"])
print("fmacc", export["fmacc"])
print("final_object_count", export["final_object_count"])
for key in [
    "proposal_generation",
    "yoloworld_primary",
    "sam2_proposals",
    "anchor_guided_sam_fusion",
    "runtime_vis",
    "depth_refinement",
    "association",
    "object_update",
]:
    if key in timings:
        print(key, timings[key]["mean_sec"], timings[key]["frame_count"])
PY
```

Expected: prints the headline metrics and per-stage means. `sam2_proposals.frame_count` should be about 20 for 200 frames with interval 10.

## Task 5: Report Comparison

**Files:**
- No code edits expected.

- [ ] **Step 1: Compare against previous online and DualMap measurements**

Use the new summary, plus known reference numbers:

- Previous OVI-OVO online YOLOWorld+SAM2 aggregate: `2.586s/frame`, `mIoU=0.376`, `f-mIoU=0.460`.
- DualMap room0 200f temporary run: `0.2418s/frame`, detector `0.2096s`, local mapping `0.0321s`; caveat CLIP `ViT-B-32` and improvised environment.

- [ ] **Step 2: Final answer in Chinese**

Report:

- Summary of Balanced logic.
- Files changed.
- Commands run.
- Tests/results with exact output paths.
- Remaining issues and caveats.
