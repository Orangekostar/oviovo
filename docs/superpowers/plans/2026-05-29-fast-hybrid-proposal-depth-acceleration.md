# Fast Hybrid Proposal Depth Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce room0 fast-hybrid runtime by removing per-frame YOLOE process/model startup cost and replacing Python mask connected components with a cropped OpenCV path.

**Architecture:** `ObjectAnchorModule` records nested detector timings that `Pipeline` folds into the existing stage timing summary. `YOLOESegAnchorBackend` gains an optional persistent JSON-lines worker that loads YOLOE once and falls back to the current helper path on worker failure. `DepthRefinementModule` keeps the same public output contract while running connected components on proposal bbox crops through OpenCV.

**Tech Stack:** Python 3, NumPy, OpenCV, PIL, subprocess JSON-lines IPC, pytest, existing OVO pipeline/config/reporting code.

---

## File Structure

- Modify `src/modules/object_anchor.py`
  - Add nested detector timing fields and a small timing helper.
  - Record `yoloworld_primary`, `yoloe_supplemental`, and `anchor_merge`.
- Modify `src/pipelines/main_pipeline.py`
  - Merge nested anchor timings into `last_frame_debug["stage_timings"]`.
- Create `src/models/json_line_worker_client.py`
  - Own persistent subprocess lifecycle and request/response timeout handling.
- Create `frontend/yoloe_seg_prompt_free_worker.py`
  - Load YOLOE and vocab once, then serve per-frame inference over stdin/stdout JSON lines.
- Modify `src/models/yoloe_seg_anchor_backend.py`
  - Add `worker_enabled` config path.
  - Encode RGB frames as PNG base64 and call the persistent worker.
  - Preserve current helper path as fallback.
- Modify `src/modules/depth_refinement.py`
  - Add bbox-cropped refinement and OpenCV connected components backend.
  - Keep Python connected components as an explicit fallback.
- Modify `configs/room0_fast_hybrid_4090.yaml`
  - Enable YOLOE worker mode and OpenCV/crop depth refinement for fast hybrid.
- Modify tests:
  - `tests/test_object_anchor.py`
  - `tests/test_pipeline.py`
  - `tests/test_dual_map.py`

---

### Task 1: Add Fine-Grained Anchor Timing

**Files:**
- Modify: `tests/test_object_anchor.py`
- Modify: `tests/test_pipeline.py`
- Modify: `src/modules/object_anchor.py`
- Modify: `src/pipelines/main_pipeline.py`

- [ ] **Step 1: Write failing tests for nested anchor timing**

Add this test near the existing supplemental merge tests in `tests/test_object_anchor.py`:

```python
def test_object_anchor_records_primary_supplemental_and_merge_timings() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "supplemental_enabled": True,
            "supplemental_iou_threshold": 0.5,
            "supplemental_overlap_threshold": 0.6,
        }
    )
    module.enabled = True

    class _PrimaryBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 4, 4], dtype=np.float32),
                    class_name="chair",
                    confidence=0.9,
                )
            ]

    class _SupplementalBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=11,
                    bbox_xyxy=np.array([5, 5, 8, 8], dtype=np.float32),
                    class_name="rug",
                    confidence=0.8,
                )
            ]

    times = iter([10.0, 10.5, 20.0, 20.25, 30.0, 30.75])
    module._clock = lambda: next(times)
    module.backend = _PrimaryBackend()
    module.supplemental_backend = _SupplementalBackend()

    anchors = module._generate_merged_anchors(np.zeros((10, 10, 3), dtype=np.uint8))

    assert [anchor.anchor_id for anchor in anchors] == [0, 11]
    assert module.last_generation_timings == {
        "yoloworld_primary": 0.5,
        "yoloe_supplemental": 0.25,
        "anchor_merge": 0.75,
    }
```

Add this test after `test_pipeline_records_stage_timings_when_enabled` in `tests/test_pipeline.py`:

```python
    def test_pipeline_merges_anchor_frontend_stage_timings(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
frame_input:
  image_height: 4
  image_width: 4
proposal:
  backend: placeholder
anchor_frontend:
  enabled: false
pipeline:
  verbose: false
  collect_stage_timings: true
patch_lifting:
  min_points: 1
object_update:
  surface_owner_gate:
    enabled: false
semantic_memory:
  backend: placeholder
""",
            encoding="utf-8",
        )
        from src.pipelines.main_pipeline import Pipeline

        pipe = Pipeline(config_path=str(config_path))
        pipe.object_anchor.enabled = True
        pipe.object_anchor.anchor_primary_mode = True

        def _fake_generate_anchor_box_proposals(rgb):
            pipe.object_anchor.last_generation_timings = {
                "yoloworld_primary": 0.12,
                "yoloe_supplemental": 0.34,
                "anchor_merge": 0.05,
            }
            return [], [], []

        pipe.object_anchor.generate_anchor_box_proposals = _fake_generate_anchor_box_proposals
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4)

        pipe.process_frame(rgb, depth, np.eye(4, dtype=np.float64), intrinsics)

        timings = pipe.last_frame_debug["stage_timings"]
        assert timings["yoloworld_primary"] == pytest.approx(0.12)
        assert timings["yoloe_supplemental"] == pytest.approx(0.34)
        assert timings["anchor_merge"] == pytest.approx(0.05)
        assert timings["proposal_generation"] >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_object_anchor_records_primary_supplemental_and_merge_timings tests/test_pipeline.py::TestPipeline::test_pipeline_merges_anchor_frontend_stage_timings -q
```

Expected: both tests fail because `last_generation_timings` is not populated and the pipeline does not merge nested timings.

- [ ] **Step 3: Implement nested timing in `ObjectAnchorModule`**

In `src/modules/object_anchor.py`, add this import:

```python
import time
```

In `ObjectAnchorModule.__init__`, after `self.last_supplemental_anchors` is initialized, add:

```python
        self.last_generation_timings: dict[str, float] = {}
        self._clock = time.perf_counter
```

Add this method before `_generate_merged_anchors`:

```python
    def _timed_anchor_generation(self, name: str, callback):
        start = self._clock()
        try:
            return callback()
        finally:
            elapsed = max(0.0, float(self._clock() - start))
            self.last_generation_timings[name] = self.last_generation_timings.get(name, 0.0) + elapsed
```

Replace `_generate_merged_anchors` with:

```python
    def _generate_merged_anchors(self, rgb: np.ndarray) -> List[Anchor2D]:
        self.last_generation_timings = {}
        primary = self._timed_anchor_generation(
            "yoloworld_primary",
            lambda: self.backend.generate_anchors(rgb) if self.enabled else [],
        )
        self.last_primary_anchors = list(primary)
        if not self.supplemental_enabled or self.supplemental_backend is None:
            self.last_supplemental_anchors = []
            self.last_generation_timings.setdefault("yoloe_supplemental", 0.0)
            self.last_generation_timings.setdefault("anchor_merge", 0.0)
            return list(primary)

        supplemental = self._timed_anchor_generation(
            "yoloe_supplemental",
            lambda: self.supplemental_backend.generate_anchors(rgb),
        )
        self.last_supplemental_anchors = list(supplemental)
        return self._timed_anchor_generation(
            "anchor_merge",
            lambda: self._merge_anchor_sets(primary, supplemental),
        )
```

- [ ] **Step 4: Merge nested timings in `Pipeline`**

In `src/pipelines/main_pipeline.py`, add this method after `_timed_stage`:

```python
    def _merge_stage_timings(self, timings: dict[str, float] | None) -> None:
        if not self.collect_stage_timings or not timings:
            return
        for name, value in dict(timings).items():
            try:
                timing_value = float(value)
            except (TypeError, ValueError):
                continue
            if timing_value < 0.0:
                continue
            self._stage_timings[str(name)] = self._stage_timings.get(str(name), 0.0) + timing_value
```

Inside the `proposal_generation` block in `process_frame`, after `self.last_anchor_assignments = anchor_assignments`, add:

```python
            self._merge_stage_timings(getattr(self.object_anchor, "last_generation_timings", {}))
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_object_anchor_records_primary_supplemental_and_merge_timings tests/test_pipeline.py::TestPipeline::test_pipeline_merges_anchor_frontend_stage_timings tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/modules/object_anchor.py src/pipelines/main_pipeline.py tests/test_object_anchor.py tests/test_pipeline.py
git commit -m "perf: record nested anchor generation timings"
```

---

### Task 2: Add A Persistent JSON-Lines Worker Client

**Files:**
- Create: `src/models/json_line_worker_client.py`
- Modify: `tests/test_object_anchor.py`

- [ ] **Step 1: Write failing client tests**

Add these imports near the top of `tests/test_object_anchor.py`:

```python
import json
import subprocess
```

Add this test near the YOLOE backend tests:

```python
def test_json_line_worker_client_sends_payload_and_reads_response(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def __init__(self):
            self.lines = []

        def write(self, value):
            self.lines.append(value)

        def flush(self):
            return None

    class _FakeStdout:
        def __init__(self):
            self.lines = [json.dumps({"id": 1, "anchors": []}) + "\n"]

        def fileno(self):
            return 123

        def readline(self):
            return self.lines.pop(0)

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
            self.stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: (readable, [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={"A": "B"},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )
    response = client.request({"image_png_b64": "abc"})

    assert response == {"id": 1, "anchors": []}
    sent = json.loads(fake_process.stdin.lines[0])
    assert sent["id"] == 1
    assert sent["image_png_b64"] == "abc"
```

Add this second test:

```python
def test_json_line_worker_client_times_out_when_worker_is_silent(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

        def readline(self):
            return ""

    class _FakeProcess:
        stdin = _FakeStdin()
        stdout = _FakeStdout()
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: ([], [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=0.001,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        client.request({"image_png_b64": "abc"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_json_line_worker_client_sends_payload_and_reads_response tests/test_object_anchor.py::test_json_line_worker_client_times_out_when_worker_is_silent -q
```

Expected: import fails because `src.models.json_line_worker_client` does not exist.

- [ ] **Step 3: Implement the worker client**

Create `src/models/json_line_worker_client.py`:

```python
"""Persistent JSON-lines subprocess client for detector workers."""

from __future__ import annotations

import json
import select
import subprocess
import time
from typing import Any


class JsonLineWorkerClient:
    """Send one JSON request per line and read one JSON response per line."""

    def __init__(
        self,
        *,
        command: list[str],
        env: dict[str, str],
        cwd: str,
        request_timeout_sec: float,
    ) -> None:
        self.command = list(command)
        self.env = dict(env)
        self.cwd = str(cwd)
        self.request_timeout_sec = float(request_timeout_sec)
        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self._process is not None
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Worker process was started without stdin/stdout pipes.")

        request_id = self._next_request_id
        self._next_request_id += 1
        request_payload = dict(payload)
        request_payload["id"] = request_id
        self._process.stdin.write(json.dumps(request_payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

        deadline = time.monotonic() + max(0.001, self.request_timeout_sec)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"JSON-line worker request {request_id} timed out.")
            if self._process.poll() is not None:
                raise RuntimeError(f"JSON-line worker exited before response for request {request_id}.")
            readable, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not readable:
                continue
            line = self._process.stdout.readline()
            if line == "":
                raise RuntimeError(f"JSON-line worker closed stdout before response for request {request_id}.")
            response = json.loads(line)
            if int(response.get("id", -1)) != request_id:
                raise RuntimeError(
                    f"JSON-line worker response id mismatch: expected {request_id}, got {response.get('id')!r}."
                )
            return response

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            process.kill()
            process.wait(timeout=2.0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_json_line_worker_client_sends_payload_and_reads_response tests/test_object_anchor.py::test_json_line_worker_client_times_out_when_worker_is_silent -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/models/json_line_worker_client.py tests/test_object_anchor.py
git commit -m "feat: add persistent json-line worker client"
```

---

### Task 3: Add YOLOE Persistent Worker Script

**Files:**
- Create: `frontend/yoloe_seg_prompt_free_worker.py`
- Modify: `tests/test_object_anchor.py`

- [ ] **Step 1: Write failing worker utility test**

Add these imports near the top of `tests/test_object_anchor.py`:

```python
import base64
import importlib.util
from io import BytesIO

from PIL import Image
```

Add this test near the YOLOE backend tests:

```python
def test_yoloe_worker_decode_matches_existing_helper_channel_order() -> None:
    worker_path = Path(__file__).resolve().parent.parent / "frontend" / "yoloe_seg_prompt_free_worker.py"
    spec = importlib.util.spec_from_file_location("yoloe_seg_prompt_free_worker", worker_path)
    assert spec is not None
    assert spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    buffer = BytesIO()
    Image.fromarray(rgb[..., ::-1]).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    decoded = worker.decode_image_png_b64(encoded)

    assert np.array_equal(decoded, rgb)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_yoloe_worker_decode_matches_existing_helper_channel_order -q
```

Expected: test fails because `frontend/yoloe_seg_prompt_free_worker.py` does not exist.

- [ ] **Step 3: Implement the worker script**

Create `frontend/yoloe_seg_prompt_free_worker.py`:

```python
#!/usr/bin/env python3
"""Persistent YOLOE-Seg prompt-free anchor worker."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--classes-json", type=str, default="[]")
    parser.add_argument("--vocab-source-checkpoint-path", type=Path)
    parser.add_argument("--vocab-cache-path", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--max-detections", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def resolve_vocab_source_checkpoint_path(args: argparse.Namespace) -> Path | None:
    if args.vocab_source_checkpoint_path:
        return args.vocab_source_checkpoint_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    if checkpoint_path.stem.endswith("-pf"):
        candidate = checkpoint_path.with_name(f"{checkpoint_path.stem[:-3]}{checkpoint_path.suffix}")
        if candidate.exists():
            return candidate
    return None


def resolve_vocab_cache_path(args: argparse.Namespace, repo_root: Path, classes: list[str]) -> Path:
    if args.vocab_cache_path:
        return args.vocab_cache_path.resolve()
    source_checkpoint_path = resolve_vocab_source_checkpoint_path(args)
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else "",
                "classes": classes,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return repo_root / ".cache" / "oviovo_yoloe_vocab" / f"{digest}.pt"


def decode_image_png_b64(encoded: str) -> np.ndarray:
    payload = base64.b64decode(encoded.encode("ascii"))
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode PNG request image.")
    return image


def anchors_from_result(result: Any) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {})
    if boxes is None:
        return anchors
    for anchor_id, (xyxy, conf, cls_idx) in enumerate(
        zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
    ):
        class_name = names.get(int(cls_idx), str(int(cls_idx))) if isinstance(names, dict) else str(int(cls_idx))
        anchors.append(
            {
                "anchor_id": int(anchor_id),
                "bbox_xyxy": [float(v) for v in xyxy.tolist()],
                "class_name": str(class_name),
                "confidence": float(conf),
                "metadata": {
                    "source": "yoloe_seg_pf_worker",
                    "vocab_applied": True,
                },
            }
        )
    return anchors


def main() -> None:
    args = parse_args()
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if str(args.device).strip().lower().startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    third_party_clip = repo_root / "third_party" / "CLIP"
    if third_party_clip.exists() and str(third_party_clip) not in sys.path:
        sys.path.insert(0, str(third_party_clip))
    third_party_mobileclip = repo_root / "third_party" / "ml-mobileclip"
    if third_party_mobileclip.exists() and str(third_party_mobileclip) not in sys.path:
        sys.path.insert(0, str(third_party_mobileclip))

    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import select_device

    class NoFuseSegmentationPredictor(SegmentationPredictor):
        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        def setup_model(self, model, verbose=True):
            self.model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, verbose=verbose),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
                batch=self.args.batch,
                fuse=False,
                verbose=verbose,
            )
            self.device = self.model.device
            self.args.half = self.model.fp16
            self.model.eval()

    classes = [str(item).strip() for item in json.loads(args.classes_json) if str(item).strip()]
    vocab = None
    if classes:
        vocab_source_checkpoint_path = resolve_vocab_source_checkpoint_path(args)
        if vocab_source_checkpoint_path is None or not vocab_source_checkpoint_path.exists():
            raise FileNotFoundError("YOLOE vocabulary-aware prompt-free inference requires a non-PF source checkpoint.")
        vocab_cache_path = resolve_vocab_cache_path(args, repo_root, classes)
        if vocab_cache_path.exists():
            vocab = torch.load(vocab_cache_path, map_location=str(args.device))
        else:
            unfused_model = YOLOE(str(vocab_source_checkpoint_path))
            unfused_model.to(str(args.device))
            vocab = unfused_model.get_vocab(classes)
            vocab_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vocab, vocab_cache_path)
            del unfused_model

    model = YOLOE(str(args.checkpoint_path))
    model.to(str(args.device))
    if vocab is not None:
        model.set_vocab(vocab, names=classes)
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = float(args.confidence_threshold)
        model.model.model[-1].max_det = int(args.max_detections)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("command") == "shutdown":
            break
        request_id = int(request.get("id", -1))
        try:
            image = decode_image_png_b64(str(request["image_png_b64"]))
            results = model.predict(
                image,
                verbose=False,
                conf=float(args.confidence_threshold),
                max_det=int(args.max_detections),
                device=str(args.device),
                predictor=NoFuseSegmentationPredictor,
            )
            anchors = anchors_from_result(results[0]) if results else []
            response = {"id": request_id, "anchors": anchors}
        except Exception as exc:
            response = {"id": request_id, "error": f"{type(exc).__name__}: {exc}"}
        protocol_stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused test**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_yoloe_worker_decode_matches_existing_helper_channel_order -q
```

Expected: test passes.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add frontend/yoloe_seg_prompt_free_worker.py tests/test_object_anchor.py
git commit -m "feat: add persistent yoloe worker script"
```

---

### Task 4: Wire YOLOE Backend To Persistent Worker With Fallback

**Files:**
- Modify: `src/models/yoloe_seg_anchor_backend.py`
- Modify: `tests/test_object_anchor.py`

- [ ] **Step 1: Write failing backend worker tests**

Add these tests near `test_yoloe_backend_derives_non_pf_vocab_source_checkpoint` in `tests/test_object_anchor.py`:

```python
def test_yoloe_backend_uses_worker_when_enabled(tmp_path: Path) -> None:
    repo_root = tmp_path / "yoloe_repo_probe"
    repo_root.mkdir()
    checkpoint_path = repo_root / "yoloe-v8s-seg-pf.pt"
    checkpoint_path.write_bytes(b"")
    vocab_source_checkpoint_path = repo_root / "yoloe-v8s-seg.pt"
    vocab_source_checkpoint_path.write_bytes(b"")
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")

    class _FakeWorker:
        def __init__(self):
            self.requests = []

        def request(self, payload):
            self.requests.append(dict(payload))
            return {
                "id": 1,
                "anchors": [
                    {
                        "anchor_id": 3,
                        "bbox_xyxy": [1.0, 2.0, 5.0, 6.0],
                        "class_name": "rug",
                        "confidence": 0.77,
                        "metadata": {"source": "yoloe_seg_pf_worker"},
                    }
                ],
            }

    fake_worker = _FakeWorker()
    backend = YOLOESegAnchorBackend()
    backend._create_worker_client = lambda: fake_worker
    backend.initialize(
        {
            "repo_root": str(repo_root),
            "checkpoint_path": str(checkpoint_path),
            "vocab_source_checkpoint_path": str(vocab_source_checkpoint_path),
            "python_executable": str(python_path),
            "classes": ["rug"],
            "worker_enabled": True,
        }
    )

    anchors = backend.generate_anchors(np.zeros((4, 4, 3), dtype=np.uint8))

    assert len(anchors) == 1
    assert anchors[0].anchor_id == 3
    assert anchors[0].class_name == "rug"
    assert anchors[0].confidence == pytest.approx(0.77)
    assert fake_worker.requests
    assert isinstance(fake_worker.requests[0]["image_png_b64"], str)
```

Add this fallback test:

```python
def test_yoloe_backend_falls_back_to_helper_when_worker_fails(tmp_path: Path) -> None:
    repo_root = tmp_path / "yoloe_repo_probe"
    repo_root.mkdir()
    checkpoint_path = repo_root / "yoloe-v8s-seg-pf.pt"
    checkpoint_path.write_bytes(b"")
    vocab_source_checkpoint_path = repo_root / "yoloe-v8s-seg.pt"
    vocab_source_checkpoint_path.write_bytes(b"")
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")

    class _FailingWorker:
        def request(self, payload):
            raise RuntimeError("worker unavailable")

    backend = YOLOESegAnchorBackend()
    backend._create_worker_client = lambda: _FailingWorker()
    backend._generate_anchors_via_helper = lambda rgb: [
        Anchor2D(
            anchor_id=9,
            bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
            class_name="chair",
            confidence=0.5,
        )
    ]
    backend.initialize(
        {
            "repo_root": str(repo_root),
            "checkpoint_path": str(checkpoint_path),
            "vocab_source_checkpoint_path": str(vocab_source_checkpoint_path),
            "python_executable": str(python_path),
            "classes": ["chair"],
            "worker_enabled": True,
            "worker_fallback_on_error": True,
        }
    )

    anchors = backend.generate_anchors(np.zeros((4, 4, 3), dtype=np.uint8))

    assert len(anchors) == 1
    assert anchors[0].anchor_id == 9
    assert anchors[0].class_name == "chair"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_yoloe_backend_uses_worker_when_enabled tests/test_object_anchor.py::test_yoloe_backend_falls_back_to_helper_when_worker_fails -q
```

Expected: tests fail because `worker_enabled` is not implemented.

- [ ] **Step 3: Implement worker mode in `YOLOESegAnchorBackend`**

In `src/models/yoloe_seg_anchor_backend.py`, add imports:

```python
import base64
from io import BytesIO

from src.models.json_line_worker_client import JsonLineWorkerClient
```

In `YOLOESegAnchorBackend.__init__`, add:

```python
        self.worker_enabled = False
        self.worker_script = ""
        self.worker_request_timeout_sec = 120.0
        self.worker_fallback_on_error = True
        self._worker_client = None
```

In `initialize`, after `self.helper_script` is assigned, add:

```python
        worker_default = Path(__file__).resolve().parents[2] / "frontend" / "yoloe_seg_prompt_free_worker.py"
        self.worker_script = str(config.get("worker_script", worker_default)).strip()
```

After helper retry fields are assigned, add:

```python
        self.worker_enabled = bool(config.get("worker_enabled", False))
        self.worker_request_timeout_sec = max(1.0, float(config.get("worker_request_timeout_sec", 120.0)))
        self.worker_fallback_on_error = bool(config.get("worker_fallback_on_error", True))
```

After the helper script path check, add:

```python
        if self.worker_enabled and not Path(self.worker_script).exists():
            raise FileNotFoundError(f"YOLOE worker script not found: {self.worker_script}")
```

Replace `generate_anchors` with:

```python
    def generate_anchors(self, rgb: np.ndarray) -> List[Anchor2D]:
        if self.worker_enabled:
            try:
                anchors = self._generate_anchors_via_worker(rgb)
            except Exception:
                if not self.worker_fallback_on_error:
                    raise
                logger.warning("YOLOE worker inference failed; falling back to helper process.", exc_info=True)
                anchors = self._generate_anchors_via_helper(rgb)
        else:
            anchors = self._generate_anchors_via_helper(rgb)
        self.last_generation_info = {
            **self.last_generation_info,
            "anchor_count": int(len(anchors)),
            "worker_enabled": bool(self.worker_enabled),
        }
        return anchors
```

Add these methods before `_generate_anchors_via_helper`:

```python
    def _worker_command(self) -> list[str]:
        return [
            self.external_python,
            self.worker_script,
            "--checkpoint-path",
            self.checkpoint_path,
            "--repo-root",
            self.repo_root,
            "--classes-json",
            json.dumps(self.classes),
            "--vocab-source-checkpoint-path",
            self.vocab_source_checkpoint_path,
            "--confidence-threshold",
            str(self.confidence_threshold),
            "--max-detections",
            str(self.max_detections),
            "--device",
            self.device,
        ]

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        if str(self.device).strip().lower().startswith("cpu"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def _create_worker_client(self):
        return JsonLineWorkerClient(
            command=self._worker_command(),
            env=self._worker_env(),
            cwd=self.repo_root,
            request_timeout_sec=self.worker_request_timeout_sec,
        )

    def _encode_rgb_png_b64(self, rgb: np.ndarray) -> str:
        buffer = BytesIO()
        Image.fromarray(np.asarray(rgb)[..., ::-1]).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _anchors_from_payload(self, payload: dict[str, Any]) -> List[Anchor2D]:
        anchors: list[Anchor2D] = []
        for item in payload.get("anchors", []):
            anchors.append(
                Anchor2D(
                    anchor_id=int(item["anchor_id"]),
                    bbox_xyxy=np.asarray(item["bbox_xyxy"], dtype=np.float32),
                    class_name=str(item["class_name"]),
                    confidence=float(item["confidence"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return anchors

    def _generate_anchors_via_worker(self, rgb: np.ndarray) -> List[Anchor2D]:
        if self._worker_client is None:
            self._worker_client = self._create_worker_client()
        payload = self._worker_client.request({"image_png_b64": self._encode_rgb_png_b64(rgb)})
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return self._anchors_from_payload(payload)

    def close(self) -> None:
        worker = self._worker_client
        self._worker_client = None
        if worker is not None and hasattr(worker, "close"):
            worker.close()
```

Refactor the anchor parsing at the end of `_generate_anchors_via_helper` to use `_anchors_from_payload`:

```python
        return self._anchors_from_payload(payload)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py::test_yoloe_backend_uses_worker_when_enabled tests/test_object_anchor.py::test_yoloe_backend_falls_back_to_helper_when_worker_fails tests/test_object_anchor.py::test_yoloe_backend_derives_non_pf_vocab_source_checkpoint -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add src/models/yoloe_seg_anchor_backend.py tests/test_object_anchor.py
git commit -m "perf: reuse persistent yoloe worker"
```

---

### Task 5: Accelerate Depth Refinement Connected Components

**Files:**
- Modify: `src/modules/depth_refinement.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing depth refinement tests**

Add this test near the depth refinement tests in `tests/test_pipeline.py`:

```python
    def test_depth_refinement_opencv_components_match_python_backend(self):
        pytest.importorskip("cv2")
        depth = np.full((12, 12), 2.0, dtype=np.float32)
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:4, 1:4] = True
        mask[7:11, 7:11] = True
        proposal = Proposal2D(
            proposal_id=21,
            mask=mask,
            bbox_xyxy=np.array([1, 1, 11, 11], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
        )

        python_module = DepthRefinementModule(
            {
                "depth_edge_threshold": 10.0,
                "min_mask_area_after_refine": 1,
                "connected_components_backend": "python",
                "use_bbox_crop": False,
            }
        )
        opencv_module = DepthRefinementModule(
            {
                "depth_edge_threshold": 10.0,
                "min_mask_area_after_refine": 1,
                "connected_components_backend": "opencv",
                "use_bbox_crop": False,
            }
        )

        python_refined = python_module.process(depth, [proposal])
        opencv_refined = opencv_module.process(depth, [proposal])

        assert [item.area for item in opencv_refined] == [item.area for item in python_refined]
        assert [item.mask.tolist() for item in opencv_refined] == [item.mask.tolist() for item in python_refined]
```

Add this crop test:

```python
    def test_depth_refinement_uses_bbox_crop_for_component_split(self):
        depth = np.full((32, 32), 2.0, dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:15, 20:27] = True
        proposal = Proposal2D(
            proposal_id=22,
            mask=mask,
            bbox_xyxy=np.array([20, 10, 27, 15], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.9,
        )
        module = DepthRefinementModule(
            {
                "depth_edge_threshold": 10.0,
                "min_mask_area_after_refine": 1,
                "connected_components_backend": "python",
                "use_bbox_crop": True,
            }
        )
        seen_shapes = []
        original_split = module._split_connected_components

        def _recording_split(component_mask):
            seen_shapes.append(component_mask.shape)
            return original_split(component_mask)

        module._split_connected_components = _recording_split

        refined = module.process(depth, [proposal])

        assert len(refined) == 1
        assert refined[0].mask.shape == (32, 32)
        assert refined[0].area == int(mask.sum())
        assert seen_shapes == [(5, 7)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_depth_refinement_opencv_components_match_python_backend tests/test_pipeline.py::TestPipeline::test_depth_refinement_uses_bbox_crop_for_component_split -q
```

Expected: tests fail because the backend selector and bbox crop path do not exist.

- [ ] **Step 3: Implement OpenCV and crop paths**

In `src/modules/depth_refinement.py`, add config fields in `__init__` after `self.max_plane_fit_points`:

```python
        self.connected_components_backend = str(config.get("connected_components_backend", "auto")).strip().lower()
        self.use_bbox_crop = bool(config.get("use_bbox_crop", True))
```

Replace the start of `_refine_single_proposal` through component splitting with:

```python
        if self.use_bbox_crop:
            y_slice, x_slice = self._mask_crop_slices(proposal.mask)
            if y_slice.stop <= y_slice.start or x_slice.stop <= x_slice.start:
                component_masks = []
            else:
                cropped_mask = np.asarray(proposal.mask[y_slice, x_slice], dtype=bool)
                cropped_edges = np.asarray(depth_edges[y_slice, x_slice], dtype=bool)
                cropped_new_mask = cropped_mask & (~cropped_edges)
                cropped_components = self._split_connected_components(cropped_new_mask)
                component_masks = [
                    self._restore_cropped_component(component, proposal.mask.shape, y_slice, x_slice)
                    for component in cropped_components
                ]
        else:
            new_mask = proposal.mask & (~depth_edges)
            component_masks = self._split_connected_components(new_mask)
```

Add these methods before `_split_connected_components`:

```python
    def _mask_crop_slices(self, mask: np.ndarray) -> tuple[slice, slice]:
        bbox = self._mask_bbox(np.asarray(mask, dtype=bool))
        x1 = max(0, int(np.floor(float(bbox[0]))))
        y1 = max(0, int(np.floor(float(bbox[1]))))
        x2 = min(mask.shape[1], int(np.ceil(float(bbox[2]))))
        y2 = min(mask.shape[0], int(np.ceil(float(bbox[3]))))
        return slice(y1, y2), slice(x1, x2)

    def _restore_cropped_component(
        self,
        component: np.ndarray,
        full_shape: tuple[int, int],
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        restored = np.zeros(full_shape, dtype=bool)
        restored[y_slice, x_slice] = np.asarray(component, dtype=bool)
        return restored
```

Rename the existing `_split_connected_components` implementation to `_split_connected_components_python`.

Add this dispatcher under the renamed method:

```python
    def _split_connected_components(self, mask: np.ndarray) -> List[np.ndarray]:
        backend = self.connected_components_backend
        if backend in {"opencv", "cv2", "auto"}:
            try:
                return self._split_connected_components_opencv(mask)
            except ImportError:
                if backend in {"opencv", "cv2"}:
                    raise
        return self._split_connected_components_python(mask)
```

Add this OpenCV implementation:

```python
    def _split_connected_components_opencv(self, mask: np.ndarray) -> List[np.ndarray]:
        import cv2

        mask_uint8 = np.asarray(mask, dtype=np.uint8)
        if not np.any(mask_uint8):
            return []
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=4)
        components: List[np.ndarray] = []
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            components.append(labels == label)
        components.sort(key=lambda component: int(component.sum()), reverse=True)
        return components
```

- [ ] **Step 4: Run focused depth tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_pipeline.py::TestPipeline::test_depth_refinement_splits_disconnected_components_and_exposes_scores tests/test_pipeline.py::TestPipeline::test_depth_refinement_keepalive_forces_protected_small_anchor_object_candidate tests/test_pipeline.py::TestPipeline::test_depth_refinement_opencv_components_match_python_backend tests/test_pipeline.py::TestPipeline::test_depth_refinement_uses_bbox_crop_for_component_split -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add src/modules/depth_refinement.py tests/test_pipeline.py
git commit -m "perf: accelerate depth refinement components"
```

---

### Task 6: Enable Fast Mode Config And Verify Reports

**Files:**
- Modify: `configs/room0_fast_hybrid_4090.yaml`
- Modify: `tests/test_dual_map.py`

- [ ] **Step 1: Write failing config assertions**

Find the existing fast-hybrid config test in `tests/test_dual_map.py` that already asserts:

```python
assert config["patch_lifting"]["point_sample_ratio"] == 0.08
assert config["patch_lifting"]["max_points_per_patch"] == 2048
assert config["object_update"]["surface_owner_gate"]["representative_voxel_mode"] is True
assert config["pipeline"]["collect_stage_timings"] is True
```

Add these assertions immediately after them:

```python
    assert config["anchor_frontend"]["supplemental"]["worker_enabled"] is True
    assert config["anchor_frontend"]["supplemental"]["worker_fallback_on_error"] is True
    assert config["depth_refinement"]["connected_components_backend"] == "opencv"
    assert config["depth_refinement"]["use_bbox_crop"] is True
```

- [ ] **Step 2: Run config test to verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_room0_fast_hybrid_config_uses_anchor_primary_fast_path -q
```

Expected: test fails because the config keys are not enabled yet.

- [ ] **Step 3: Update fast hybrid config**

In `configs/room0_fast_hybrid_4090.yaml`, under `anchor_frontend.supplemental`, add:

```yaml
    worker_enabled: true
    worker_fallback_on_error: true
    worker_request_timeout_sec: 120.0
```

Under `depth_refinement`, add:

```yaml
  connected_components_backend: opencv
  use_bbox_crop: true
```

- [ ] **Step 4: Run config and timing report tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_dual_map.py::test_room0_fast_hybrid_config_uses_anchor_primary_fast_path tests/test_dual_map.py::test_write_run_report_includes_stage_timing_summary tests/test_dual_map.py::test_build_stage_timing_summary_skips_bad_values -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add configs/room0_fast_hybrid_4090.yaml tests/test_dual_map.py
git commit -m "config: enable fast hybrid worker acceleration"
```

---

### Task 7: Full Verification And 20f Smoke

**Files:**
- No source edits unless verification exposes a concrete regression.

- [ ] **Step 1: Run focused unit suite**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_object_anchor.py tests/test_dual_map.py tests/test_pipeline.py tests/test_provisional_pool.py -q
```

Expected: all tests pass. Previous baseline was `155 passed`; the new count will be higher because this plan adds tests.

- [ ] **Step 2: Run 20f fast-hybrid smoke**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
/home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
  --config-path configs/room0_fast_hybrid_4090.yaml \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --output-root outputs/tmp_validation \
  --experiment-name 20260529_room0_fast_hybrid_worker_depth_stride10_20f \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-device cuda
```

Expected:

- The run exits with status `0`.
- `outputs/tmp_validation/20260529_room0_fast_hybrid_worker_depth_stride10_20f/room0/run_report.md` exists.
- `Stage Timing Summary` includes `proposal_generation`, `yoloworld_primary`, `yoloe_supplemental`, `anchor_merge`, and `depth_refinement`.

- [ ] **Step 3: Extract timing and metric comparison**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
import json
from pathlib import Path

old_report = Path("outputs/tmp_validation/20260529_room0_fast_hybrid_stride10_20f_final/room0/run_report.json")
new_report = Path("outputs/tmp_validation/20260529_room0_fast_hybrid_worker_depth_stride10_20f/room0/run_report.json")
for label, path in [("old", old_report), ("new", new_report)]:
    data = json.loads(path.read_text(encoding="utf-8"))
    print(label)
    print("  mIoU", data.get("metrics", {}).get("mIoU"))
    print("  f-mIoU", data.get("metrics", {}).get("f-mIoU"))
    timings = data.get("stage_timing_summary", {})
    for key in ["proposal_generation", "yoloworld_primary", "yoloe_supplemental", "anchor_merge", "depth_refinement"]:
        if key in timings:
            print(" ", key, timings[key])
PY
```

Expected:

- New `proposal_generation` mean is lower than the old `4.6988s`.
- New `depth_refinement` mean is lower than the old `1.2354s`.
- `mIoU` and `f-mIoU` remain in the same rough range as the old smoke. A small change is acceptable because runtime and worker path should preserve detector semantics, while connected-component ordering can affect tie cases.

- [ ] **Step 4: Commit verification note if source changes were needed**

If Step 1-3 required source changes, commit those exact files:

```bash
git status --short
git add <changed-source-or-test-files>
git commit -m "fix: stabilize fast hybrid acceleration smoke"
```

If no source changes were needed, do not create an empty commit.
