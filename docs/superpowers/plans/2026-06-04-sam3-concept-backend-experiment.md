# SAM3 Concept Backend Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use GPT-5.5 medium for coding subagents and GPT-5.5 medium for review subagents unless the caller explicitly overrides it.

**Goal:** Add a contained SAM3 experiment path that can compare SAM3 concept segmentation against the current YOLOWorld+SAM2 online frontend on room0 fast eval without breaking the existing baseline.

**Architecture:** Add `sam3_concept` as an experimental proposal backend that returns semantically tagged `Proposal2D` masks directly from SAM3 concept prompts. Keep YOLOWorld+SAM2 unchanged. First implement a mockable backend and runner/config plumbing, then add a real SAM3 worker adapter guarded by environment probes, then run 20f and 200f room0 comparisons.

**Tech Stack:** Python, pytest, YAML, subprocess JSONL worker protocol, existing `ProposalModule`, `ProposalBackend`, `Pipeline`, checkpointed Replica runners, SAM3 official package in an isolated Python environment.

---

## Evidence And Constraints

- Current fast-eval baseline:
  - Config: `configs/replica_yoloworld_sam_online_baseline_4090.yaml`
  - Proposal backend: `sam2`
  - Anchor backend: `yoloworld`
  - Parallel YOLOWorld+SAM2 frontend is gated by `Pipeline._should_use_parallel_yoloworld_sam_frontend()` requiring active anchor backend `yoloworld` and proposal backend `sam2`.
  - Current strong 200f room0 run: `mIoU=0.5303`, `f-mIoU=0.6062`, `sec/frame=4.0377`.
- The SAM3 experiment must not modify the validated YOLOWorld+SAM2 path.
- Current `/home/ww/miniconda3/envs/oviovo/bin/python` reports Python 3.10.20 and torch 2.6.0+cu124. SAM3 official repo may require a newer Python/PyTorch/CUDA stack, so real SAM3 inference must run through a separate worker env until proven compatible.
- Network/raw GitHub access was unreliable during planning. Treat official SAM3 API names as externally verified during implementation, not assumed from memory.
- Existing `Proposal2D` can carry semantic labels through metadata:
  - `anchor_id`
  - `anchor_class_name`
  - `anchor_confidence`
  - `anchor_label_strength`
  - `semantic_commit_allowed`
  - `anchor_label_votes`
- For the SAM3 experiment, each SAM3 mask should be emitted as an already-labeled proposal. YOLOWorld anchors and `anchor_guided_sam_fusion` should be disabled for the pure SAM3 path.
- RuntimeVis policy:
  - First SAM3 experiment should set `runtime_vis.enabled: false` or an identity path if available.
  - If the pipeline still instantiates disabled RuntimeVis, that is acceptable for the experiment. Do not refactor RuntimeVis in this plan.

## File Structure

- Create: `src/models/sam3_concept_backend.py`
  - Owns the experimental SAM3 concept proposal backend.
  - Supports `mode: mock`, `mode: worker`, and `mode: disabled`.
  - Converts worker JSON results into `Proposal2D` with semantic metadata.

- Modify: `src/modules/proposal.py`
  - Registers `backend: sam3_concept`.

- Modify: `run_replica_room0_analysis.py`
  - Adds runner CLI/config wiring for `sam3_concept` backend.

- Modify: `scripts/run_room0_checkpointed_eval.py`
  - Adds optional SAM3 CLI arguments and forwards them through imported `build_proposal_module()`.

- Modify: `scripts/run_replica_all_scenes_fast_eval.py`
  - Adds `--online-sam3-concept` dry-run/execution path for all-scene script.
  - For first pass, support room0 and keep all-scene selection available but not required.

- Create: `configs/replica_sam3_concept_room0_experiment_4090.yaml`
  - SAM3-only room0 config derived from `replica_yoloworld_sam_online_baseline_4090.yaml`.
  - Disables YOLOWorld/SAM fusion and RuntimeVis.

- Create: `scripts/sam3_concept_worker.py`
  - External worker process for real SAM3 inference.
  - Reads one JSON request per line from stdin and writes one JSON response per line to stdout.
  - Keeps SAM3 model loaded across frames.

- Create: `scripts/probe_sam3_env.py`
  - Verifies SAM3 env import, model construction, and a tiny smoke inference if assets are available.

- Modify: `tests/test_proposal_backends.py`
  - Unit tests for backend config merge, mock proposals, worker result normalization, and fallback errors.

- Modify: `tests/test_pipeline.py`
  - Tests that pure SAM3 backend bypasses YOLOWorld+SAM2 parallel frontend and preserves semantic proposal metadata.

- Modify: `tests/test_replica_all_scenes.py`
  - Tests dry-run command construction for `--online-sam3-concept`.

---

## Task 1: Add Mockable SAM3 Concept Backend

**Files:**
- Create: `src/models/sam3_concept_backend.py`
- Modify: `src/modules/proposal.py`
- Modify: `tests/test_proposal_backends.py`

- [ ] **Step 1: Write backend selection test**

Add this near the existing `ProposalModule` backend selection tests in `tests/test_proposal_backends.py`:

```python
def test_proposal_module_selects_sam3_concept_and_merges_nested_config(monkeypatch) -> None:
    import src.models.sam3_concept_backend as sam3_backend_module

    monkeypatch.setattr(sam3_backend_module, "SAM3ConceptProposalBackend", _FakeBackend)
    module = ProposalModule(
        {
            "backend": "sam3_concept",
            "min_mask_area": 10,
            "sam3_concept": {
                "mode": "mock",
                "classes": ["chair", "table"],
                "confidence_threshold": 0.25,
            },
        }
    )

    assert isinstance(module.backend, _FakeBackend)
    assert module.backend.init_config["backend"] == "sam3_concept"
    assert module.backend.init_config["mode"] == "mock"
    assert module.backend.init_config["classes"] == ["chair", "table"]
    assert module.backend.init_config["confidence_threshold"] == 0.25
```

- [ ] **Step 2: Run selection test and verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_proposal_backends.py::test_proposal_module_selects_sam3_concept_and_merges_nested_config -q
```

Expected: FAIL because `src.models.sam3_concept_backend` or backend registration does not exist.

- [ ] **Step 3: Implement initial backend and registration**

Create `src/models/sam3_concept_backend.py`:

```python
"""Experimental SAM3 concept proposal backend."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.core.data_structures import Frame, Proposal2D
from src.models.proposal_backend import ProposalBackend

logger = logging.getLogger("oviovo.models.sam3_concept_backend")


class SAM3ConceptProposalBackend(ProposalBackend):
    """SAM3 concept segmentation backend with mock and worker modes."""

    def __init__(self) -> None:
        self.mode = "disabled"
        self.classes: list[str] = []
        self.max_proposals = 64
        self.confidence_threshold = 0.0
        self.worker_python = ""
        self.worker_script = ""
        self.worker_env: dict[str, str] = {}
        self.worker: subprocess.Popen[str] | None = None
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: dict) -> None:
        self.mode = str(config.get("mode", "disabled")).strip() or "disabled"
        self.classes = [str(label).strip() for label in config.get("classes", []) if str(label).strip()]
        self.max_proposals = int(config.get("max_proposals", config.get("topk_per_frame", 64)))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
        self.worker_python = str(config.get("worker_python", "")).strip()
        self.worker_script = str(config.get("worker_script", "")).strip()
        self.worker_env = {
            str(key): str(value)
            for key, value in dict(config.get("worker_env", {}) or {}).items()
        }
        if self.mode not in {"mock", "worker", "disabled"}:
            raise ValueError(f"Unsupported sam3_concept mode: {self.mode}")
        if self.mode == "worker":
            self._start_worker()
        self.last_generation_info = {
            "backend": "sam3_concept",
            "mode": self.mode,
            "class_count": len(self.classes),
            "max_proposals": self.max_proposals,
            "confidence_threshold": self.confidence_threshold,
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> list[Proposal2D]:
        if self.mode == "disabled":
            return []
        if self.mode == "mock":
            return self._mock_proposals(rgb)
        response = self._request_worker(rgb, frame)
        proposals = self._proposals_from_response(response, rgb.shape[:2])
        self.last_generation_info = {
            **self.last_generation_info,
            "raw_mask_count": int(response.get("raw_mask_count", len(proposals))),
            "kept_mask_count": len(proposals),
        }
        return proposals

    def close(self) -> None:
        if self.worker is None:
            return
        try:
            self.worker.terminate()
            self.worker.wait(timeout=5)
        except Exception:
            self.worker.kill()
        finally:
            self.worker = None

    def _mock_proposals(self, rgb: np.ndarray) -> list[Proposal2D]:
        height, width = rgb.shape[:2]
        labels = self.classes or ["object"]
        proposals: list[Proposal2D] = []
        for idx, label in enumerate(labels[: min(len(labels), self.max_proposals, 3)]):
            x1 = max(0, int(width * (0.15 + 0.20 * idx)))
            y1 = max(0, int(height * (0.15 + 0.15 * idx)))
            x2 = min(width, max(x1 + 1, int(width * (0.45 + 0.20 * idx))))
            y2 = min(height, max(y1 + 1, int(height * (0.45 + 0.15 * idx))))
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            proposals.append(
                self._proposal_from_mask(
                    proposal_id=len(proposals),
                    mask=mask,
                    label=label,
                    confidence=0.9 - 0.05 * idx,
                    raw_metadata={"mock": True},
                )
            )
        return proposals

    def _start_worker(self) -> None:
        if not self.worker_python:
            raise ValueError("sam3_concept worker mode requires worker_python")
        if not self.worker_script:
            raise ValueError("sam3_concept worker mode requires worker_script")
        script_path = Path(self.worker_script).expanduser()
        if not script_path.exists():
            raise FileNotFoundError(f"SAM3 worker script not found: {script_path}")
        env = None
        if self.worker_env:
            import os

            env = os.environ.copy()
            env.update(self.worker_env)
        self.worker = subprocess.Popen(
            [self.worker_python, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def _request_worker(self, rgb: np.ndarray, frame: Frame | None) -> dict[str, Any]:
        if self.worker is None or self.worker.stdin is None or self.worker.stdout is None:
            raise RuntimeError("SAM3 worker is not running.")
        request = {
            "frame_id": int(frame.frame_id if frame is not None else -1),
            "source_frame_id": int(getattr(frame, "source_frame_id", -1) or -1),
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "classes": list(self.classes),
            "max_proposals": int(self.max_proposals),
            "confidence_threshold": float(self.confidence_threshold),
            "rgb": np.asarray(rgb, dtype=np.uint8).tolist(),
        }
        self.worker.stdin.write(json.dumps(request) + "\n")
        self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("SAM3 worker exited without a response.")
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError(f"SAM3 worker error: {response.get('error', 'unknown error')}")
        return response

    def _proposals_from_response(self, response: dict[str, Any], shape: tuple[int, int]) -> list[Proposal2D]:
        proposals: list[Proposal2D] = []
        for item in response.get("proposals", []):
            confidence = float(item.get("confidence", item.get("score", 0.0)))
            if confidence < self.confidence_threshold:
                continue
            mask = np.asarray(item.get("mask", []), dtype=bool)
            if mask.shape != shape:
                continue
            area = int(mask.sum())
            if area <= 0:
                continue
            label = str(item.get("label", item.get("class_name", ""))).strip()
            proposals.append(
                self._proposal_from_mask(
                    proposal_id=len(proposals),
                    mask=mask,
                    label=label,
                    confidence=confidence,
                    raw_metadata=dict(item.get("metadata", {}) or {}),
                )
            )
        proposals.sort(key=lambda proposal: (proposal.confidence, proposal.area), reverse=True)
        proposals = proposals[: self.max_proposals]
        for proposal_id, proposal in enumerate(proposals):
            proposal.proposal_id = proposal_id
            proposal.metadata["source_raw_proposal_ids"] = [proposal_id]
        return proposals

    def _proposal_from_mask(
        self,
        *,
        proposal_id: int,
        mask: np.ndarray,
        label: str,
        confidence: float,
        raw_metadata: dict[str, Any],
    ) -> Proposal2D:
        mask = np.asarray(mask, dtype=bool)
        bbox = self._mask_to_bbox(mask)
        metadata = {
            "source": "sam3_concept",
            "backend": "sam3_concept",
            "anchor_id": int(proposal_id),
            "anchor_class_name": label,
            "anchor_confidence": float(confidence),
            "anchor_label_strength": "strong" if label else "none",
            "anchor_keepalive": bool(label),
            "anchor_label_votes": {label: float(confidence)} if label else {},
            "semantic_commit_allowed": bool(label),
            "mask_source": "sam3_concept",
            "observation_layer": "fine",
            "sam3_metadata": raw_metadata,
        }
        return Proposal2D(
            proposal_id=int(proposal_id),
            mask=mask,
            bbox_xyxy=bbox,
            area=int(mask.sum()),
            confidence=float(confidence),
            backend_name="sam3_concept",
            metadata=metadata,
        )

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            dtype=np.float32,
        )
```

Modify `src/modules/proposal.py` in `_create_backend()`:

```python
        elif backend_name == "sam3_concept":
            from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

            return SAM3ConceptProposalBackend()
```

- [ ] **Step 4: Run selection test and verify it passes**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_proposal_backends.py::test_proposal_module_selects_sam3_concept_and_merges_nested_config -q
```

Expected: PASS.

- [ ] **Step 5: Write mock proposal metadata test**

Add to `tests/test_proposal_backends.py`:

```python
def test_sam3_concept_mock_backend_returns_semantic_proposals() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()
    backend.initialize(
        {
            "backend": "sam3_concept",
            "mode": "mock",
            "classes": ["chair", "table"],
            "max_proposals": 4,
            "confidence_threshold": 0.0,
        }
    )
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    depth = np.ones((20, 30), dtype=np.float32)

    proposals = backend.generate_proposals(rgb, depth)

    assert [proposal.backend_name for proposal in proposals] == ["sam3_concept", "sam3_concept"]
    assert [proposal.metadata["anchor_class_name"] for proposal in proposals] == ["chair", "table"]
    assert all(proposal.metadata["semantic_commit_allowed"] is True for proposal in proposals)
    assert all(proposal.metadata["anchor_label_strength"] == "strong" for proposal in proposals)
    assert all(proposal.area > 0 for proposal in proposals)
```

- [ ] **Step 6: Run mock metadata test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_proposal_backends.py::test_sam3_concept_mock_backend_returns_semantic_proposals -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/models/sam3_concept_backend.py src/modules/proposal.py tests/test_proposal_backends.py
git commit -m "feat: add experimental sam3 concept proposal backend"
```

---

## Task 2: Route Pure SAM3 Through Pipeline Without YOLOWorld+SAM2 Fusion

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `configs/replica_sam3_concept_room0_experiment_4090.yaml`

- [ ] **Step 1: Write pipeline bypass test**

Add this to `tests/test_pipeline.py` near existing YOLOWorld+SAM frontend tests:

```python
def test_pipeline_skips_yoloworld_sam_frontend_for_sam3_concept_backend(self, tmp_path):
    config_path = tmp_path / "sam3_pipeline.yaml"
    config_path.write_text(
        """
frame_input:
  image_height: 8
  image_width: 8
proposal:
  backend: sam3_concept
  min_mask_area: 1
  sam3_concept:
    mode: mock
    classes: [chair]
    max_proposals: 1
anchor_frontend:
  enabled: false
  backend: placeholder
runtime_vis:
  enabled: false
depth_refinement:
  enabled: false
patch_lifting:
  min_points: 1
association:
  match_threshold: 0.5
object_update:
  provisional_pool:
    enabled: false
pipeline:
  collect_stage_timings: true
  yoloworld_sam_parallel_frontend_enabled: true
""",
        encoding="utf-8",
    )
    pipe = Pipeline(config_path=str(config_path))

    assert pipe.proposal.active_backend_name == "sam3_concept"
    assert pipe._should_use_parallel_yoloworld_sam_frontend() is False

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=5.0, fy=5.0, cx=4.0, cy=4.0, width=8, height=8)
    pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

    assert pipe.last_frame_debug["proposal_source"] == "proposal_backend"
    assert pipe.last_frame_debug["frontend_actual_backend"] == "sam3_concept"
    assert "sam2_proposals" not in pipe.last_frame_debug["stage_timings"]
    assert "anchor_guided_sam_fusion" not in pipe.last_frame_debug["stage_timings"]
    assert pipe.last_raw_proposals[0].metadata["anchor_class_name"] == "chair"
```

- [ ] **Step 2: Run pipeline bypass test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_pipeline_skips_yoloworld_sam_frontend_for_sam3_concept_backend -q
```

Expected: PASS if Task 1 is complete. If it fails because disabled RuntimeVis still creates identity output correctly, inspect the exact assertion and keep the production behavior unchanged unless SAM3 metadata is dropped.

- [ ] **Step 3: Add experiment config**

Create `configs/replica_sam3_concept_room0_experiment_4090.yaml` by copying the validated online baseline shape but with this frontend section:

```yaml
frame_input:
  image_height: 480
  image_width: 640
proposal:
  backend: sam3_concept
  min_mask_area: 100
  max_proposals: 80
  confidence_threshold: 0.20
  sam3_concept:
    mode: mock
    worker_python: /home/ww/miniconda3/envs/sam3/bin/python
    worker_script: scripts/sam3_concept_worker.py
    device: cuda
    max_proposals: 80
    confidence_threshold: 0.20
    classes:
      - basket
      - blanket
      - blinds
      - book
      - cabinet
      - candle
      - chair
      - cushion
      - ceiling
      - door
      - floor
      - indoor-plant
      - lamp
      - picture
      - pillar
      - plant-stand
      - plate
      - pot
      - sofa
      - stool
      - switch
      - table
      - vase
      - vent
      - wall
      - wall-plug
      - window
      - rug
anchor_frontend:
  enabled: false
  backend: placeholder
runtime_vis:
  enabled: false
anchor_guided_sam:
  enabled: false
pipeline:
  log_interval: 1
  verbose: false
  collect_stage_timings: true
  proposal_prefetch_enabled: false
  yoloworld_sam_parallel_frontend_enabled: false
  yoloworld_sam_balanced_frontend_enabled: false
```

Then copy the rest of the validated downstream sections from `configs/replica_yoloworld_sam_online_baseline_4090.yaml` unchanged:

- `depth_refinement`
- `patch_lifting`
- `bg_obj_split`
- `association`
- `object_update`
- `dual_map`
- `semantic_memory`
- `active_set`
- `dense_surface`
- `map_tiering`
- `background_update`
- `dynamic_maintenance`
- `structural_overlay`
- `async_refinement`

Do not enable RuntimeVis in this config.

- [ ] **Step 4: Add config test**

Add to `tests/test_replica_all_scenes.py`:

```python
def test_sam3_concept_experiment_config_disables_yoloworld_sam_and_runtimevis() -> None:
    config_path = Path("configs/replica_sam3_concept_room0_experiment_4090.yaml")
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "sam3_concept"
    assert config["proposal"]["sam3_concept"]["mode"] == "mock"
    assert "chair" in config["proposal"]["sam3_concept"]["classes"]
    assert config["anchor_frontend"]["enabled"] is False
    assert config["anchor_guided_sam"]["enabled"] is False
    assert config["runtime_vis"]["enabled"] is False
    assert config["pipeline"]["yoloworld_sam_parallel_frontend_enabled"] is False
    assert config["pipeline"]["proposal_prefetch_enabled"] is False
```

- [ ] **Step 5: Run Task 2 tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_pipeline.py::TestPipeline::test_pipeline_skips_yoloworld_sam_frontend_for_sam3_concept_backend \
  tests/test_replica_all_scenes.py::test_sam3_concept_experiment_config_disables_yoloworld_sam_and_runtimevis \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add configs/replica_sam3_concept_room0_experiment_4090.yaml tests/test_pipeline.py tests/test_replica_all_scenes.py
git commit -m "test: add pure sam3 concept pipeline experiment config"
```

---

## Task 3: Add Runner CLI Wiring For SAM3 Experiment

**Files:**
- Modify: `run_replica_room0_analysis.py`
- Modify: `scripts/run_room0_checkpointed_eval.py`
- Modify: `scripts/run_replica_all_scenes_fast_eval.py`
- Modify: `tests/test_replica_all_scenes.py`

- [ ] **Step 1: Write command construction test**

Add to `tests/test_replica_all_scenes.py`:

```python
def test_online_sam3_concept_command_passes_backend_without_sam2_flags(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "Replica" / "room0",
        gt_labels=tmp_path / "labels" / "room0.txt",
        gt_mesh_ply=tmp_path / "Replica_original" / "room_0" / "habitat" / "mesh_semantic.ply",
        gt_info_json=tmp_path / "Replica_original" / "room_0" / "habitat" / "info_semantic.json",
    )

    command = run_replica_all_scenes_fast_eval.build_scene_command(
        scene="room0",
        experiment_name="sam3_room0_s10_1f_fast_eval",
        paths=paths,
        python_executable=tmp_path / "python",
        runner_script=tmp_path / "scripts" / "run_room0_checkpointed_eval.py",
        config_path=tmp_path / "sam3.yaml",
        output_root=tmp_path / "outputs",
        num_frames=1,
        frame_stride=10,
        proposal_backend="sam3_concept",
        proposal_device="cuda",
    )

    assert command[command.index("--proposal-backend") + 1] == "sam3_concept"
    assert "--sam-version" not in command
    assert "--proposal-cache-manifest" not in command
```

- [ ] **Step 2: Run command test and verify current behavior**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py::test_online_sam3_concept_command_passes_backend_without_sam2_flags -q
```

Expected: PASS or FAIL only because `--online-sam3-concept` mode is not wired yet. Keep this test as coverage for command shape.

- [ ] **Step 3: Add `--online-sam3-concept` all-scene mode test**

Add to `tests/test_replica_all_scenes.py`:

```python
def test_online_sam3_concept_mode_uses_sam3_config_and_skips_cache_validation(
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
        raise AssertionError("SAM3 mode must not validate precomputed caches")

    monkeypatch.setattr(
        run_replica_all_scenes_fast_eval,
        "validate_precomputed_cache_for_scene",
        fail_if_precomputed_cache_validation_runs,
    )

    _make_executable(repo_root / "bin" / "python")
    for path in [
        repo_root / "scripts" / "run_room0_checkpointed_eval.py",
        repo_root / "configs" / "replica_sam3_concept_room0_experiment_4090.yaml",
        repo_root / "Replica" / "room0",
        repo_root / "labels" / "room0.txt",
        repo_root / "Original" / "room_0" / "habitat" / "mesh_semantic.ply",
        repo_root / "Original" / "room_0" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "sam3_batch",
            "--scenes",
            "room0",
            "--dry-run",
            "--online-sam3-concept",
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
    assert "--proposal-backend sam3_concept" in captured.out
    assert "replica_sam3_concept_room0_experiment_4090.yaml" in captured.out
    assert "--proposal-cache-manifest" not in captured.out
```

- [ ] **Step 4: Run mode test and verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py::test_online_sam3_concept_mode_uses_sam3_config_and_skips_cache_validation -q
```

Expected: FAIL because `--online-sam3-concept` is not defined.

- [ ] **Step 5: Update all-scene runner**

Modify `scripts/run_replica_all_scenes_fast_eval.py`:

Add constant:

```python
DEFAULT_SAM3_CONFIG_PATH = REPO_ROOT / "configs" / "replica_sam3_concept_room0_experiment_4090.yaml"
```

Add arg:

```python
parser.add_argument("--online-sam3-concept", action="store_true")
```

Update mode selection:

```python
online_yoloworld_sam = bool(args.online_yoloworld_sam)
online_sam3_concept = bool(args.online_sam3_concept)
if online_yoloworld_sam and online_sam3_concept:
    raise SystemExit("--online-yoloworld-sam and --online-sam3-concept are mutually exclusive")
if online_sam3_concept:
    default_config = REPO_ROOT / "configs" / DEFAULT_SAM3_CONFIG_PATH.name
elif online_yoloworld_sam:
    default_config = _default_online_config_path()
else:
    default_config = DEFAULT_CONFIG_PATH
proposal_backend = args.proposal_backend or (
    "sam3_concept" if online_sam3_concept else ("sam2" if online_yoloworld_sam else DEFAULT_PROPOSAL_BACKEND)
)
```

Keep cache validation guarded by `proposal_backend == "precomputed"` only.

- [ ] **Step 6: Add runner SAM3 args**

Modify `scripts/run_room0_checkpointed_eval.py` and root `run_room0_full_eval.py` parser args to include optional SAM3 settings:

```python
parser.add_argument("--sam3-worker-python", type=Path, default=None)
parser.add_argument("--sam3-worker-script", type=Path, default=None)
parser.add_argument("--sam3-mode", type=str, default=None)
```

Modify `run_replica_room0_analysis.py::build_proposal_module()`:

```python
    elif args.proposal_backend == "sam3_concept":
        if getattr(args, "sam3_mode", None):
            nested["mode"] = str(args.sam3_mode)
        if getattr(args, "sam3_worker_python", None) is not None:
            nested["worker_python"] = str(args.sam3_worker_python)
        if getattr(args, "sam3_worker_script", None) is not None:
            nested["worker_script"] = str(args.sam3_worker_script)
        nested["max_proposals"] = args.max_proposals
        nested["confidence_threshold"] = args.confidence_threshold
        config.update(nested)
```

- [ ] **Step 7: Run runner tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_replica_all_scenes.py::test_online_sam3_concept_command_passes_backend_without_sam2_flags \
  tests/test_replica_all_scenes.py::test_online_sam3_concept_mode_uses_sam3_config_and_skips_cache_validation \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add run_replica_room0_analysis.py run_room0_full_eval.py scripts/run_room0_checkpointed_eval.py scripts/run_replica_all_scenes_fast_eval.py tests/test_replica_all_scenes.py
git commit -m "feat: wire sam3 concept experiment runner mode"
```

---

## Task 4: Add SAM3 Environment Probe And Worker Skeleton

**Files:**
- Create: `scripts/probe_sam3_env.py`
- Create: `scripts/sam3_concept_worker.py`
- Modify: `tests/test_proposal_backends.py`

- [ ] **Step 1: Create environment probe script**

Create `scripts/probe_sam3_env.py`:

```python
#!/usr/bin/env python3
"""Probe whether the configured Python environment can run SAM3."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sam3_repo_root is not None:
        sys.path.insert(0, str(args.sam3_repo_root.resolve()))
    result: dict[str, object] = {
        "python": sys.version,
        "ok": False,
        "imports": {},
        "errors": [],
    }
    for module_name in ["torch", "sam3"]:
        try:
            module = importlib.import_module(module_name)
            result["imports"][module_name] = str(getattr(module, "__file__", "built-in"))
            if module_name == "torch":
                result["torch_version"] = str(getattr(module, "__version__", "unknown"))
                result["cuda_available"] = bool(module.cuda.is_available())
                result["cuda_version"] = str(getattr(module.version, "cuda", ""))
        except Exception as exc:
            result["imports"][module_name] = None
            result["errors"].append(f"{module_name}: {type(exc).__name__}: {exc}")
    result["ok"] = not result["errors"]
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create worker skeleton with explicit unsupported error**

Create `scripts/sam3_concept_worker.py`:

```python
#!/usr/bin/env python3
"""JSONL worker for experimental SAM3 concept inference."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default="")
    return parser.parse_args()


def _load_sam3(args: argparse.Namespace) -> Any:
    if args.sam3_repo_root is not None:
        sys.path.insert(0, str(args.sam3_repo_root.resolve()))
    try:
        importlib.import_module("sam3")
    except Exception as exc:
        raise RuntimeError(
            "SAM3 package is not importable in this worker environment. "
            "Install the official facebookresearch/sam3 package in an isolated env first."
        ) from exc
    raise RuntimeError(
        "SAM3 worker model binding is intentionally not completed by the skeleton. "
        "Implement this after confirming the official SAM3 API in the installed repo."
    )


def main() -> int:
    args = parse_args()
    try:
        model = _load_sam3(args)
    except Exception as exc:
        for line in sys.stdin:
            _request = json.loads(line)
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return 0

    for line in sys.stdin:
        request = json.loads(line)
        try:
            _ = model
            _ = request
            response = {"ok": True, "raw_mask_count": 0, "proposals": []}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write worker error normalization test**

Add to `tests/test_proposal_backends.py`:

```python
def test_sam3_concept_worker_response_error_is_reported(tmp_path: Path) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    print(json.dumps({'ok': False, 'error': 'boom'}), flush=True)\n",
        encoding="utf-8",
    )
    backend = SAM3ConceptProposalBackend()
    backend.initialize(
        {
            "backend": "sam3_concept",
            "mode": "worker",
            "worker_python": sys.executable,
            "worker_script": str(worker_script),
            "classes": ["chair"],
        }
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)

    with pytest.raises(RuntimeError, match="boom"):
        backend.generate_proposals(rgb, depth)

    backend.close()
```

- [ ] **Step 4: Run worker tests and probe script**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_proposal_backends.py::test_sam3_concept_worker_response_error_is_reported -q
/home/ww/miniconda3/envs/oviovo/bin/python scripts/probe_sam3_env.py --json || true
```

Expected:

- Test: PASS.
- Probe: likely exits 2 in current `oviovo` env if SAM3 is not installed. Record the JSON output in implementation notes.

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/probe_sam3_env.py scripts/sam3_concept_worker.py tests/test_proposal_backends.py
git commit -m "feat: add sam3 worker probe skeleton"
```

---

## Task 5: Implement Real SAM3 Worker After API Confirmation

**Files:**
- Modify: `scripts/sam3_concept_worker.py`
- Modify: `src/models/sam3_concept_backend.py`
- Create or update: `docs/superpowers/reports/2026-06-04-sam3-api-probe.md`

- [ ] **Step 1: Install or point to SAM3 environment outside `oviovo`**

Use an isolated env, not `/home/ww/miniconda3/envs/oviovo`.

Preferred command shape:

```bash
conda create -y -n sam3 python=3.12
conda run -n sam3 python -m pip install --upgrade pip
conda run -n sam3 python -m pip install -e /path/to/facebookresearch/sam3
```

If official SAM3 docs require a different Python/PyTorch/CUDA stack, follow those docs and record exact commands in the report.

- [ ] **Step 2: Probe official API**

Run:

```bash
/home/ww/miniconda3/envs/sam3/bin/python scripts/probe_sam3_env.py --sam3-repo-root /path/to/facebookresearch/sam3 --json
```

Then inspect the installed package:

```bash
/home/ww/miniconda3/envs/sam3/bin/python - <<'PY'
import inspect
import sam3
print("sam3", getattr(sam3, "__file__", None))
for name in [
    "sam3.model_builder",
    "sam3.model.sam3_image_processor",
    "sam3.sam3_image_predictor",
]:
    try:
        module = __import__(name, fromlist=["*"])
    except Exception as exc:
        print(name, "ERR", type(exc).__name__, exc)
        continue
    print("\\n", name, getattr(module, "__file__", None))
    for attr in dir(module):
        if "sam3" in attr.lower() or "processor" in attr.lower() or "predictor" in attr.lower():
            obj = getattr(module, attr)
            print(" ", attr, obj)
            if callable(obj):
                try:
                    print(inspect.signature(obj))
                except Exception:
                    pass
PY
```

- [ ] **Step 3: Write API report**

Create `docs/superpowers/reports/2026-06-04-sam3-api-probe.md`:

```markdown
# SAM3 API Probe

## Environment

- Python:
- torch:
- CUDA:
- SAM3 repo:
- SAM3 package path:

## Confirmed API

- Model build function:
- Image processor / predictor class:
- Text prompt method:
- Output fields:

## Worker Mapping

- Input RGB format:
- Prompt format:
- Returned mask format:
- Score field:
- Label field:

## Risks

- Installation:
- GPU memory:
- Prompt batching:
- Known limitations:
```

Fill every line with actual observed values. Do not leave placeholders.

- [ ] **Step 4: Implement real worker binding**

Replace `_load_sam3()` and the inference loop in `scripts/sam3_concept_worker.py` with the confirmed API. The implementation must:

- Load model once at startup.
- Accept `classes` from each request.
- Run all class prompts for one frame.
- Return:

```json
{
  "ok": true,
  "raw_mask_count": 3,
  "proposals": [
    {
      "label": "chair",
      "confidence": 0.91,
      "mask": [[false, true]],
      "metadata": {
        "sam3_score": 0.91,
        "prompt": "chair"
      }
    }
  ]
}
```

- Sort proposals by confidence before returning.
- Cap to `max_proposals`.
- Filter by `confidence_threshold`.

- [ ] **Step 5: Run real worker smoke**

Use one Replica frame and a small class list:

```bash
/home/ww/miniconda3/envs/sam3/bin/python scripts/sam3_concept_worker.py \
  --sam3-repo-root /path/to/facebookresearch/sam3 \
  --device cuda \
  < /tmp/sam3_one_frame_request.jsonl
```

Expected: JSON response with `ok: true` and `proposals` list. It is acceptable for `proposals` to be empty only if the tested prompt is absent; for `chair` on a visible chair frame it should be nonempty.

- [ ] **Step 6: Commit Task 5**

```bash
git add scripts/sam3_concept_worker.py src/models/sam3_concept_backend.py docs/superpowers/reports/2026-06-04-sam3-api-probe.md
git commit -m "feat: bind sam3 worker to official api"
```

---

## Task 6: Run Room0 20f And 200f Experiments

**Files:**
- No code changes expected.
- Create: `docs/superpowers/reports/2026-06-04-sam3-room0-results.md`

- [ ] **Step 1: Run mock path sanity check**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_room0_checkpointed_eval.py \
  --mode build-export \
  --scene-name room0 \
  --experiment-name 20260604_sam3_mock_room0_s10_20f \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/replica_sam3_concept_room0_experiment_4090.yaml \
  --output-root outputs/tmp_validation \
  --num-frames 20 \
  --frame-stride 10 \
  --fast-eval \
  --proposal-backend sam3_concept \
  --sam3-mode mock \
  --quiet
```

Expected:

- Run completes.
- `stage_timing_summary` has `proposal_generation`.
- `stage_timing_summary` does not have `sam2_proposals` or `anchor_guided_sam_fusion`.
- Metrics are not expected to be good because this is mock.

- [ ] **Step 2: Switch config or CLI to real worker mode**

Use CLI override for real worker:

```bash
--sam3-mode worker \
--sam3-worker-python /home/ww/miniconda3/envs/sam3/bin/python \
--sam3-worker-script scripts/sam3_concept_worker.py
```

If the worker needs additional `--sam3-repo-root` or checkpoint arguments, add them to Task 3 CLI wiring before running.

- [ ] **Step 3: Run real SAM3 20f room0**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_room0_checkpointed_eval.py \
  --mode build-export \
  --scene-name room0 \
  --experiment-name 20260604_sam3_concept_room0_s10_20f \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/replica_sam3_concept_room0_experiment_4090.yaml \
  --output-root outputs/tmp_validation \
  --num-frames 20 \
  --frame-stride 10 \
  --fast-eval \
  --proposal-backend sam3_concept \
  --sam3-mode worker \
  --sam3-worker-python /home/ww/miniconda3/envs/sam3/bin/python \
  --sam3-worker-script scripts/sam3_concept_worker.py \
  --quiet
```

Expected: completes without fallback to placeholder. If it fails due to SAM3 environment, stop and record the blocker.

- [ ] **Step 4: Compare 20f against YOLOWorld+SAM2 baseline**

Extract metrics:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path
for run in [
    "20260604_sam3_concept_room0_s10_20f",
]:
    root = Path("outputs/tmp_validation") / run / "room0"
    report = json.loads((root / "run_report.json").read_text())
    print(run)
    for key in ["miou", "fmiou", "final_object_count"]:
        print(" ", key, report.get(key))
    for stage in ["proposal_generation", "sam2_proposals", "anchor_guided_sam_fusion", "runtime_vis", "object_update"]:
        print(" ", stage, (report.get("stage_timing_summary", {}).get(stage) or {}).get("mean_sec"))
PY
```

- [ ] **Step 5: Run real SAM3 200f only if 20f is credible**

Gate:

- No placeholder fallback.
- Nonzero proposals on most frames.
- `proposal_generation` mean is not worse than YOLOWorld+SAM2 by more than 2x on 20f.
- Qualitative masks are not empty or all-background.

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_room0_checkpointed_eval.py \
  --mode build-export \
  --scene-name room0 \
  --experiment-name 20260604_sam3_concept_room0_s10_200f \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --config-path configs/replica_sam3_concept_room0_experiment_4090.yaml \
  --output-root outputs/tmp_validation \
  --num-frames 200 \
  --frame-stride 10 \
  --fast-eval \
  --proposal-backend sam3_concept \
  --sam3-mode worker \
  --sam3-worker-python /home/ww/miniconda3/envs/sam3/bin/python \
  --sam3-worker-script scripts/sam3_concept_worker.py \
  --quiet
```

- [ ] **Step 6: Write results report**

Create `docs/superpowers/reports/2026-06-04-sam3-room0-results.md`:

```markdown
# SAM3 Room0 Experiment Results

## Baseline

- Run:
- mIoU:
- f-mIoU:
- sec/frame:
- proposal_generation mean:
- sam2_proposals mean:
- anchor_guided_sam_fusion mean:
- runtime_vis mean:
- object_update mean:

## SAM3 20f

- Run:
- mIoU:
- f-mIoU:
- sec/frame:
- proposal_generation mean:
- runtime_vis mean:
- final object count:
- proposal count/frame:
- failure notes:

## SAM3 200f

- Run:
- mIoU:
- f-mIoU:
- sec/frame:
- proposal_generation mean:
- runtime_vis mean:
- final object count:
- proposal count/frame:
- failure notes:

## Decision

- Replace YOLOWorld+SAM2 now: yes/no
- Keep SAM3 as experiment: yes/no
- Next optimization:
```

Fill every field with actual values or `not run: <reason>`.

- [ ] **Step 7: Commit Task 6 report**

```bash
git add docs/superpowers/reports/2026-06-04-sam3-room0-results.md
git commit -m "docs: record sam3 room0 experiment results"
```

---

## Task 7: Review And Decide

**Files:**
- No required code changes.

- [ ] **Step 1: Run focused tests**

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_proposal_backends.py \
  tests/test_pipeline.py::TestPipeline::test_pipeline_skips_yoloworld_sam_frontend_for_sam3_concept_backend \
  tests/test_replica_all_scenes.py::test_sam3_concept_experiment_config_disables_yoloworld_sam_and_runtimevis \
  tests/test_replica_all_scenes.py::test_online_sam3_concept_command_passes_backend_without_sam2_flags \
  tests/test_replica_all_scenes.py::test_online_sam3_concept_mode_uses_sam3_config_and_skips_cache_validation \
  -q
```

Expected: PASS.

- [ ] **Step 2: Dispatch review subagent**

Use GPT-5.5 medium review subagent. Ask it to review:

- `src/models/sam3_concept_backend.py`
- `scripts/sam3_concept_worker.py`
- `run_replica_room0_analysis.py`
- `scripts/run_room0_checkpointed_eval.py`
- `scripts/run_replica_all_scenes_fast_eval.py`
- `configs/replica_sam3_concept_room0_experiment_4090.yaml`
- SAM3 result report

Review prompt:

```text
Review the SAM3 concept backend experiment implementation. Focus on:
1. Whether the YOLOWorld+SAM2 baseline path remains unchanged.
2. Whether SAM3 worker failures are explicit and do not silently fallback to placeholder.
3. Whether proposal metadata preserves semantic labels through downstream pipeline.
4. Whether runner CLI wiring is reproducible and dry-run safe.
5. Whether the experiment report supports or rejects replacing YOLOWorld+SAM2.
Return findings with file/line references and severity.
```

- [ ] **Step 3: Apply accepted review fixes**

Only fix issues that are directly related to the SAM3 experiment. Do not refactor unrelated proposal backends.

- [ ] **Step 4: Final decision**

Use the report to choose one:

- **No replacement:** SAM3 is slower or less accurate. Keep backend for future experiments.
- **Partial replacement candidate:** SAM3 is promising but needs prompt tuning, caching, or tracker integration.
- **Replacement candidate:** SAM3 200f matches or beats baseline f-mIoU and reduces `proposal_generation + anchor_guided_sam_fusion + runtime_vis` total.

Record the decision in the final response and in the results report.

---

## Self-Review Checklist

- No existing YOLOWorld+SAM2 config is edited.
- `sam3_concept` can run in mock mode without SAM3 installed.
- Real SAM3 worker runs outside the current `oviovo` env.
- RuntimeVis is disabled in the SAM3 experiment config.
- Failed SAM3 initialization does not silently look like a valid result.
- 20f run gates the 200f run.
- Results report compares both quality and speed against the known strong baseline.
