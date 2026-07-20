"""Tests for detector-anchor assignment and anchor keepalive behavior."""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Anchor2D, AnchorAssignment, Proposal2D
from src.models.yoloe_seg_anchor_backend import YOLOESegAnchorBackend
from src.modules.depth_refinement import DepthRefinementModule
from src.modules.object_anchor import ObjectAnchorModule


def test_object_anchor_assigns_box_and_keepalive() -> None:
    module = ObjectAnchorModule({"enabled": False})
    module.enabled = True
    module.last_anchors = []

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([10, 10, 40, 40], dtype=np.float32),
                    class_name="chair",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((64, 64), dtype=bool)
    mask[12:38, 12:38] = True
    proposal = Proposal2D(
        proposal_id=5,
        mask=mask,
        bbox_xyxy=np.array([12, 12, 38, 38], dtype=np.float32),
        area=int(mask.sum()),
    )
    anchors, assignments = module.process(np.zeros((64, 64, 3), dtype=np.uint8), [proposal])

    assert len(anchors) == 1
    assert len(assignments) == 1
    assignment = assignments[0]
    assert assignment.anchor_id == 0
    assert assignment.class_name == "chair"
    assert assignment.keepalive is True
    assert proposal.metadata["anchor_class_name"] == "chair"
    assert proposal.metadata["anchor_keepalive"] is True


def test_depth_refinement_keeps_anchor_supported_parent_mask() -> None:
    depth = np.full((8, 8), 1.0, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    proposal = Proposal2D(
        proposal_id=3,
        mask=mask,
        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
        area=int(mask.sum()),
        metadata={"anchor_keepalive": True, "anchor_class_name": "chair"},
    )
    module = DepthRefinementModule({"depth_edge_threshold": 0.0, "min_mask_area_after_refine": 1000})
    refined = module.process(depth, [proposal])

    assert len(refined) == 1
    assert int(refined[0].area) == int(mask.sum())


def test_object_anchor_generates_intersection_proposals_from_anchor_and_sam_masks() -> None:
    module = ObjectAnchorModule({"enabled": False})
    module.enabled = True
    module.use_sam_intersection_proposals = True
    module.prefer_largest_covering_box = True
    module.proposal_min_area = 1
    module.covering_min_proposal_coverage = 0.8

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                    class_name="chair",
                    confidence=0.85,
                ),
                Anchor2D(
                    anchor_id=1,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    class_name="chair",
                    confidence=0.95,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    proposal = Proposal2D(
        proposal_id=5,
        mask=mask,
        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
    )
    anchors, proposals, assignments = module.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(anchors) == 2
    assert len(proposals) == 1
    assert len(assignments) == 1
    proposal = proposals[0]
    assert proposal.backend_name == "anchor_sam_union"
    assert proposal.metadata["anchor_primary_proposal"] is True
    assert proposal.metadata["anchor_id"] == 1
    assert proposal.metadata["anchor_class_name"] == "chair"
    assert bool(proposal.mask[3, 3]) is True
    assert bool(proposal.mask[1, 1]) is False
    assert proposal.metadata["mask_source"] == "largest_covering_yolo_box_over_sam_union"
    assert proposal.metadata["anchor_assignment_strategy"] == "largest_covering_box"
    assert proposal.metadata["source_raw_proposal_ids"] == [5]


def test_object_anchor_preserves_nested_different_class_small_boxes() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "preserve_nested_classes",
            "preserve_nested_classes": True,
            "clip_sam_masks_to_anchor_box": True,
            "subtract_cross_class_child_mask_from_parent": True,
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.2,
            "min_protected_anchor_confidence": 0.25,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                ),
                Anchor2D(
                    anchor_id=1,
                    bbox_xyxy=np.array([3, 3, 5, 5], dtype=np.float32),
                    class_name="cushion",
                    confidence=0.8,
                ),
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True
    proposal = Proposal2D(
        proposal_id=5,
        mask=mask,
        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
    )

    _anchors, proposals, _assignments = module.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [proposal],
    )

    by_class = {item.metadata["anchor_class_name"]: item for item in proposals}
    assert sorted(by_class) == ["cushion", "sofa"]
    assert by_class["cushion"].metadata["protected_small_anchor"] is True
    assert by_class["cushion"].metadata["force_object_candidate"] is True
    assert bool(by_class["cushion"].mask[3, 3]) is True
    assert bool(by_class["sofa"].mask[3, 3]) is False
    assert by_class["sofa"].metadata["protected_child_classes"] == ["cushion"]
    assert proposal.metadata["anchor_candidate_classes"] == ["sofa", "cushion"]


def test_object_anchor_merges_supplemental_boxes_with_overlap_filter() -> None:
    module = ObjectAnchorModule({"enabled": False})
    module.supplemental_enabled = True
    module.supplemental_iou_threshold = 0.5
    module.supplemental_overlap_threshold = 0.6

    primary = [
        Anchor2D(
            anchor_id=0,
            bbox_xyxy=np.array([10, 10, 40, 40], dtype=np.float32),
            class_name="chair",
            confidence=0.9,
        )
    ]
    supplemental = [
        Anchor2D(
            anchor_id=10,
            bbox_xyxy=np.array([12, 12, 38, 38], dtype=np.float32),
            class_name="object",
            confidence=0.8,
        ),
        Anchor2D(
            anchor_id=11,
            bbox_xyxy=np.array([45, 45, 70, 70], dtype=np.float32),
            class_name="object",
            confidence=0.7,
        ),
    ]

    merged = module._merge_anchor_sets(primary, supplemental)

    assert [anchor.anchor_id for anchor in merged] == [0, 11]


def test_object_anchor_records_primary_supplemental_and_merge_timings() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "backend": "yoloworld",
            "supplemental_enabled": True,
            "supplemental_iou_threshold": 0.5,
            "supplemental_overlap_threshold": 0.6,
        }
    )
    module.enabled = True
    module.active_backend_name = "yoloworld"
    module.supplemental_backend_name = "yoloe_seg_pf"

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


def test_object_anchor_labels_non_yolo_backend_timings_without_yolo_names() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "backend": "placeholder",
            "supplemental_enabled": True,
            "supplemental": {"backend": "placeholder"},
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
    assert "yoloworld_primary" not in module.last_generation_timings
    assert "yoloe_supplemental" not in module.last_generation_timings
    assert module.last_generation_timings == {
        "placeholder_primary": 0.5,
        "placeholder_supplemental": 0.25,
        "anchor_merge": 0.75,
    }


def test_object_anchor_uses_active_backend_for_primary_timing_label() -> None:
    module = ObjectAnchorModule({"enabled": False, "backend": "yoloworld"})
    module.enabled = True
    module.active_backend_name = "placeholder"

    class _FallbackBackend:
        def generate_anchors(self, rgb):
            return []

    times = iter([10.0, 10.5])
    module._clock = lambda: next(times)
    module.backend = _FallbackBackend()

    anchors = module._generate_merged_anchors(np.zeros((10, 10, 3), dtype=np.uint8))

    assert anchors == []
    assert "yoloworld_primary" not in module.last_generation_timings
    assert module.last_generation_timings["placeholder_primary"] == 0.5


def test_object_anchor_uses_neutral_supplemental_timing_label_when_backend_missing() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "backend": "placeholder",
            "supplemental_enabled": True,
        }
    )
    module.enabled = True
    module.supplemental_backend_name = "yoloe_seg_pf"
    module.supplemental_backend = None

    class _PrimaryBackend:
        def generate_anchors(self, rgb):
            return []

    times = iter([10.0, 10.5])
    module._clock = lambda: next(times)
    module.backend = _PrimaryBackend()

    anchors = module._generate_merged_anchors(np.zeros((10, 10, 3), dtype=np.uint8))

    assert anchors == []
    assert "yoloe_supplemental" not in module.last_generation_timings
    assert module.last_generation_timings["anchor_supplemental"] == 0.0


def test_yoloe_backend_derives_non_pf_vocab_source_checkpoint(tmp_path: Path) -> None:
    repo_root = tmp_path / "yoloe_repo_probe"
    repo_root.mkdir()
    checkpoint_path = repo_root / "yoloe-v8s-seg-pf.pt"
    checkpoint_path.write_bytes(b"")
    vocab_source_checkpoint_path = repo_root / "yoloe-v8s-seg.pt"
    vocab_source_checkpoint_path.write_bytes(b"")
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")

    backend = YOLOESegAnchorBackend()
    backend.initialize(
        {
            "repo_root": str(repo_root),
            "checkpoint_path": str(checkpoint_path),
            "python_executable": str(python_path),
            "classes": ["chair", "table"],
        }
    )

    assert backend.classes == ["chair", "table"]
    assert backend.vocab_source_checkpoint_path == str(vocab_source_checkpoint_path)


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
    assert backend.last_generation_info["worker_enabled"] is True
    assert backend.last_generation_info["worker_fallback_on_error"] is True
    assert backend.last_generation_info["mode"] == "worker"


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
    assert backend.last_generation_info["mode"] == "worker_fallback"


def test_yoloe_backend_closes_worker_after_error_response_fallback(tmp_path: Path) -> None:
    repo_root = tmp_path / "yoloe_repo_probe"
    repo_root.mkdir()
    checkpoint_path = repo_root / "yoloe-v8s-seg-pf.pt"
    checkpoint_path.write_bytes(b"")
    vocab_source_checkpoint_path = repo_root / "yoloe-v8s-seg.pt"
    vocab_source_checkpoint_path.write_bytes(b"")
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")

    class _ErrorWorker:
        def __init__(self):
            self.closed = False

        def request(self, payload):
            return {"id": 1, "error": "RuntimeError: bad"}

        def close(self):
            self.closed = True

    fake_worker = _ErrorWorker()
    backend = YOLOESegAnchorBackend()
    backend._create_worker_client = lambda: fake_worker
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
    assert fake_worker.closed is True
    assert backend._worker_client is None
    assert backend.last_generation_info["mode"] == "worker_fallback"


def test_yoloe_backend_closes_worker_after_error_response_without_fallback(tmp_path: Path) -> None:
    repo_root = tmp_path / "yoloe_repo_probe"
    repo_root.mkdir()
    checkpoint_path = repo_root / "yoloe-v8s-seg-pf.pt"
    checkpoint_path.write_bytes(b"")
    vocab_source_checkpoint_path = repo_root / "yoloe-v8s-seg.pt"
    vocab_source_checkpoint_path.write_bytes(b"")
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")

    class _ErrorWorker:
        def __init__(self):
            self.closed = False

        def request(self, payload):
            return {"id": 1, "error": "RuntimeError: bad"}

        def close(self):
            self.closed = True

    fake_worker = _ErrorWorker()
    backend = YOLOESegAnchorBackend()
    backend._create_worker_client = lambda: fake_worker
    backend.initialize(
        {
            "repo_root": str(repo_root),
            "checkpoint_path": str(checkpoint_path),
            "vocab_source_checkpoint_path": str(vocab_source_checkpoint_path),
            "python_executable": str(python_path),
            "classes": ["chair"],
            "worker_enabled": True,
            "worker_fallback_on_error": False,
        }
    )

    with pytest.raises(RuntimeError, match="RuntimeError: bad"):
        backend.generate_anchors(np.zeros((4, 4, 3), dtype=np.uint8))

    assert fake_worker.closed is True
    assert backend._worker_client is None


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


def test_json_line_worker_client_uses_devnull_stderr(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    popen_kwargs = {}

    class _FakeProcess:
        stdin = None
        stdout = None

        def poll(self):
            return None

    def _fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )
    client.start()

    assert popen_kwargs["stderr"] is subprocess.DEVNULL


def test_json_line_worker_client_closes_worker_after_timeout(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
            self.stopped = False
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.terminated = True
            self.stopped = True

        def kill(self):
            self.killed = True
            self.stopped = True

        def wait(self, timeout=None):
            if not self.stopped:
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: ([], [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=0.001,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        client.request({"image_png_b64": "abc"})

    assert client._process is None
    assert fake_process.terminated or fake_process.killed


def test_json_line_worker_client_closes_worker_after_mismatched_response_id(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

        def readline(self):
            return json.dumps({"id": 2, "anchors": []}) + "\n"

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
            self.stopped = False
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.terminated = True
            self.stopped = True

        def kill(self):
            self.killed = True
            self.stopped = True

        def wait(self, timeout=None):
            if not self.stopped:
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: (readable, [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )

    with pytest.raises(RuntimeError, match="response id mismatch"):
        client.request({"image_png_b64": "abc"})

    assert client._process is None
    assert fake_process.terminated or fake_process.killed


def test_json_line_worker_client_closes_worker_after_malformed_json(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

        def readline(self):
            return "not-json\n"

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
            self.stopped = False
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.terminated = True
            self.stopped = True

        def kill(self):
            self.killed = True
            self.stopped = True

        def wait(self, timeout=None):
            if not self.stopped:
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: (readable, [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )

    with pytest.raises(RuntimeError, match="malformed JSON"):
        client.request({"image_png_b64": "abc"})

    assert client._process is None
    assert fake_process.terminated or fake_process.killed


def test_json_line_worker_client_closes_worker_after_non_dict_json(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

        def readline(self):
            return json.dumps([{"id": 1}]) + "\n"

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
            self.stopped = False
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.terminated = True
            self.stopped = True

        def kill(self):
            self.killed = True
            self.stopped = True

        def wait(self, timeout=None):
            if not self.stopped:
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: (readable, [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )

    with pytest.raises(RuntimeError, match="non-dict JSON"):
        client.request({"image_png_b64": "abc"})

    assert client._process is None
    assert fake_process.terminated or fake_process.killed


def test_json_line_worker_client_times_out_when_readline_hangs(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def __init__(self, stopped):
            self._stopped = stopped

        def fileno(self):
            return 123

        def readline(self):
            self._stopped.wait()
            return ""

    class _FakeProcess:
        def __init__(self):
            self.stopped = threading.Event()
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout(self.stopped)
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped.is_set() else None

        def terminate(self):
            self.terminated = True
            self.stopped.set()

        def kill(self):
            self.killed = True
            self.stopped.set()

        def wait(self, timeout=None):
            if not self.stopped.wait(timeout):
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: (readable, [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=0.001,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        client.request({"image_png_b64": "abc"})

    assert client._process is None
    assert fake_process.terminated or fake_process.killed


def test_json_line_worker_client_times_out_when_write_hangs(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def __init__(self, stopped):
            self._stopped = stopped

        def write(self, value):
            self._stopped.wait()
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

        def readline(self):
            return json.dumps({"id": 1, "anchors": []}) + "\n"

    class _FakeProcess:
        def __init__(self):
            self.stopped = threading.Event()
            self.stdin = _FakeStdin(self.stopped)
            self.stdout = _FakeStdout()
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped.is_set() else None

        def terminate(self):
            self.terminated = True
            self.stopped.set()

        def kill(self):
            self.killed = True
            self.stopped.set()

        def wait(self, timeout=None):
            if not self.stopped.wait(timeout):
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=0.001,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        client.request({"image_png_b64": "abc"})

    assert client._process is None
    assert fake_process.terminated or fake_process.killed


def test_json_line_worker_client_rejects_non_exact_response_id(monkeypatch) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

        def readline(self):
            return json.dumps({"id": "1", "anchors": []}) + "\n"

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
            self.stopped = False
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.terminated = True
            self.stopped = True

        def kill(self):
            self.killed = True
            self.stopped = True

        def wait(self, timeout=None):
            if not self.stopped:
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr("select.select", lambda readable, writable, exceptional, timeout: (readable, [], []))

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )

    with pytest.raises(RuntimeError, match="response id mismatch"):
        client.request({"image_png_b64": "abc"})

    assert client._process is None
    assert fake_process.terminated or fake_process.killed


def test_json_line_worker_client_rejects_bounded_response_without_newline(
    monkeypatch,
) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    requested_sizes = []

    class _FakeStdin:
        def write(self, value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _FakeStdout:
        def fileno(self):
            return 123

        def readline(self, size=-1):
            requested_sizes.append(size)
            if size < 0:
                raise AssertionError("worker response read must be bounded")
            return "x" * size

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
            self.stopped = False
            self.terminated = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.terminated = True
            self.stopped = True

        def wait(self, timeout=None):
            if not self.stopped:
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(
        "select.select",
        lambda readable, writable, exceptional, timeout: (readable, [], []),
    )
    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
        max_response_chars=64,
    )

    with pytest.raises(RuntimeError, match="response.*64.*characters"):
        client.request({"operation": "infer"})

    assert requested_sizes == [65]
    assert client._process is None
    assert fake_process.terminated


def test_json_line_worker_client_close_suppresses_poll_error() -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeProcess:
        def poll(self):
            raise RuntimeError("poll failed")

    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )
    client._process = _FakeProcess()

    client.close()

    assert client._process is None


def test_json_line_worker_client_close_does_not_block_on_stdin_write() -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    class _FakeStdin:
        def __init__(self, stopped):
            self._stopped = stopped
            self.closed = False

        def write(self, value):
            self._stopped.wait()
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class _FakeProcess:
        def __init__(self):
            self.stopped = threading.Event()
            self.stdin = _FakeStdin(self.stopped)
            self.terminated = False
            self.killed = False

        def poll(self):
            return 0 if self.stopped.is_set() else None

        def terminate(self):
            self.terminated = True
            self.stopped.set()

        def kill(self):
            self.killed = True
            self.stopped.set()

        def wait(self, timeout=None):
            if self.stdin.closed:
                self.stopped.set()
            if not self.stopped.is_set():
                raise subprocess.TimeoutExpired(["python", "worker.py"], timeout)
            return 0

    fake_process = _FakeProcess()
    client = JsonLineWorkerClient(
        command=["python", "worker.py"],
        env={},
        cwd="/tmp",
        request_timeout_sec=1.0,
    )
    client._process = fake_process

    close_thread = threading.Thread(target=client.close, daemon=True)
    close_thread.start()
    close_thread.join(0.05)

    assert not close_thread.is_alive()
    assert client._process is None
    assert fake_process.stopped.is_set()


def test_anchor_vote_proposals_keep_sam_mask_pixels_unchanged() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "preserve_nested_classes": False,
            "clip_sam_masks_to_anchor_box": False,
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.2,
            "covering_min_proposal_coverage": 0.2,
            "contained_subproposal_gate_enabled": False,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 8, 8], dtype=np.float32),
                    class_name="chair",
                    confidence=0.95,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True
    proposal = Proposal2D(
        proposal_id=5,
        mask=mask.copy(),
        bbox_xyxy=np.array([3, 3, 5, 5], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    np.testing.assert_array_equal(voted[0].mask, mask)
    assert voted[0].bbox_xyxy.tolist() == [3.0, 3.0, 5.0, 5.0]
    assert voted[0].area == int(mask.sum())
    assert voted[0].proposal_id == 5
    assert voted[0].backend_name == "sam2_anchor_vote"
    assert voted[0].metadata["source"] == "sam2_anchor_vote"
    assert voted[0].metadata["geometry_source"] == "sam2"
    assert voted[0].metadata["anchor_class_name"] == "chair"
    assert voted[0].metadata["anchor_label_strength"] == "strong"
    assert voted[0].metadata["anchor_id"] == 0
    assert voted[0].metadata["anchor_vote_score"] > 0.0
    assert voted[0].metadata["anchor_candidate_classes"] == ["chair"]


def test_object_anchor_weak_structure_overlap_config_defaults() -> None:
    module = ObjectAnchorModule({"enabled": False, "assignment_policy": "semantic_vote"})

    assert module.weak_structure_overlap_enabled is False
    assert module.weak_structure_classes == {"wall", "floor", "ceiling", "blinds", "window"}
    assert module.weak_structure_min_proposal_coverage == 0.05
    assert module.weak_structure_min_anchor_coverage == 0.20


def test_object_anchor_contained_subproposal_gate_config_defaults() -> None:
    module = ObjectAnchorModule({"enabled": False, "assignment_policy": "semantic_vote"})

    assert module.contained_subproposal_gate_enabled is True
    assert module.contained_max_anchor_coverage == 0.20
    assert module.contained_min_proposal_coverage == 0.85
    assert module.scale_compatible_min_anchor_coverage == 0.20
    assert module.scale_compatible_min_bbox_iou == 0.10


@pytest.mark.parametrize("assignment_policy", ["semantic_vote", "vote_only", "sam_mask_semantic_vote"])
def test_anchor_vote_does_not_union_same_anchor_sam_fragments(assignment_policy: str) -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": assignment_policy,
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.1,
            "covering_min_proposal_coverage": 0.1,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
                    class_name="chair",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    left = np.zeros((10, 10), dtype=bool)
    right = np.zeros((10, 10), dtype=bool)
    left[2:5, 2:4] = True
    right[2:5, 6:8] = True
    proposals = [
        Proposal2D(1, left.copy(), np.array([2, 2, 4, 5], dtype=np.float32), int(left.sum()), backend_name="sam2"),
        Proposal2D(2, right.copy(), np.array([6, 2, 8, 5], dtype=np.float32), int(right.sum()), backend_name="sam2"),
    ]

    _anchors, voted, _assignments = module.generate_proposals(
        np.zeros((10, 10, 3), dtype=np.uint8),
        proposals,
    )

    assert [proposal.proposal_id for proposal in voted] == [1, 2]
    np.testing.assert_array_equal(voted[0].mask, left)
    np.testing.assert_array_equal(voted[1].mask, right)
    assert bool(voted[0].mask[3, 7]) is False
    assert bool(voted[1].mask[3, 3]) is False


@pytest.mark.parametrize("assignment_policy", ["semantic_vote", "vote_only", "sam_mask_semantic_vote"])
def test_anchor_vote_preserves_unmatched_sam_masks_when_fallback_disabled(assignment_policy: str) -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": assignment_policy,
            "unanchored_fallback_enabled": False,
            "proposal_min_area": 1,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
                    class_name="chair",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[5:7, 5:7] = True
    proposal = Proposal2D(
        proposal_id=7,
        mask=mask.copy(),
        bbox_xyxy=np.array([5, 5, 7, 7], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.8,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    np.testing.assert_array_equal(voted[0].mask, mask)
    assert voted[0].bbox_xyxy.tolist() == [5.0, 5.0, 7.0, 7.0]
    assert voted[0].area == int(mask.sum())
    assert voted[0].proposal_id == 7
    assert voted[0].metadata["anchor_id"] == -1
    assert voted[0].metadata["anchor_class_name"] == ""
    assert voted[0].metadata["anchor_candidate_classes"] == []
    assert voted[0].metadata["anchor_candidate_ids"] == []
    assert voted[0].metadata["anchor_label_votes"] == {}
    assert assignments[0].anchor_id == -1


def test_anchor_vote_does_not_label_small_contained_sam_mask_from_large_parent_anchor() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.20,
            "covering_min_proposal_coverage": 0.85,
            "contained_subproposal_gate_enabled": True,
            "contained_max_anchor_coverage": 0.20,
            "contained_min_proposal_coverage": 0.85,
            "scale_compatible_min_anchor_coverage": 0.20,
            "scale_compatible_min_bbox_iou": 0.10,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    proposal = Proposal2D(
        proposal_id=40,
        mask=mask.copy(),
        bbox_xyxy=np.array([8, 8, 12, 12], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.99,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    assert assignments[0].anchor_id == -1
    assert voted[0].metadata["anchor_id"] == -1
    assert voted[0].metadata["anchor_class_name"] == ""
    assert voted[0].metadata["anchor_label_strength"] == "none"
    assert voted[0].metadata["semantic_commit_allowed"] is False
    assert voted[0].metadata["residual_semantic_policy"] == "unknown"
    assert voted[0].metadata["mask_anchor_relation"] == "contained_residual"
    assert voted[0].metadata["anchor_candidate_classes"] == []
    assert voted[0].metadata["anchor_label_votes"] == {}
    assert voted[0].metadata["anchor_blocked_candidates"][0]["anchor_id"] == 0
    assert voted[0].metadata["anchor_blocked_candidates"][0]["class_name"] == "sofa"
    assert voted[0].metadata["anchor_blocked_candidates"][0]["reason"] == "contained_subproposal_without_child_anchor"
    assert voted[0].metadata["anchor_blocked_candidates"][0]["proposal_coverage"] == 1.0
    assert voted[0].metadata["anchor_blocked_candidates"][0]["anchor_coverage"] == 0.04


def test_anchor_vote_assigns_child_anchor_and_blocks_parent_for_contained_sam_mask() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.20,
            "covering_min_proposal_coverage": 0.85,
            "contained_subproposal_gate_enabled": True,
            "contained_max_anchor_coverage": 0.20,
            "contained_min_proposal_coverage": 0.85,
            "scale_compatible_min_anchor_coverage": 0.20,
            "scale_compatible_min_bbox_iou": 0.10,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                ),
                Anchor2D(
                    anchor_id=1,
                    bbox_xyxy=np.array([8, 8, 12, 12], dtype=np.float32),
                    class_name="blanket",
                    confidence=0.95,
                ),
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    proposal = Proposal2D(
        proposal_id=42,
        mask=mask.copy(),
        bbox_xyxy=np.array([8, 8, 12, 12], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.99,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    assert assignments[0].anchor_id == 1
    assert assignments[0].class_name == "blanket"
    assert voted[0].metadata["anchor_class_name"] == "blanket"
    assert voted[0].metadata["anchor_label_strength"] == "strong"
    assert voted[0].metadata["anchor_candidate_classes"] == ["blanket"]
    assert voted[0].metadata["anchor_blocked_candidates"][0]["anchor_id"] == 0
    assert voted[0].metadata["anchor_blocked_candidates"][0]["class_name"] == "sofa"
    assert voted[0].metadata["anchor_blocked_candidates"][0]["reason"] == "contained_subproposal_without_child_anchor"


def test_anchor_vote_keeps_strong_label_for_scale_compatible_sam_mask() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.20,
            "covering_min_proposal_coverage": 0.85,
            "contained_subproposal_gate_enabled": True,
            "contained_max_anchor_coverage": 0.20,
            "contained_min_proposal_coverage": 0.85,
            "scale_compatible_min_anchor_coverage": 0.20,
            "scale_compatible_min_bbox_iou": 0.10,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((20, 20), dtype=bool)
    mask[1:19, 1:19] = True
    proposal = Proposal2D(
        proposal_id=41,
        mask=mask.copy(),
        bbox_xyxy=np.array([1, 1, 19, 19], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.99,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    assert assignments[0].anchor_id == 0
    assert assignments[0].class_name == "sofa"
    assert voted[0].metadata["anchor_id"] == 0
    assert voted[0].metadata["anchor_class_name"] == "sofa"
    assert voted[0].metadata["anchor_label_strength"] == "strong"
    assert voted[0].metadata["anchor_candidate_classes"] == ["sofa"]
    assert voted[0].metadata["anchor_blocked_candidates"] == []


def test_anchor_vote_does_not_label_large_sam_mask_from_small_contained_anchor() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.2,
            "covering_min_proposal_coverage": 0.85,
            "iou_assign_threshold": 0.2,
            "center_assign_enabled": True,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([5, 5, 15, 15], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.9,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.ones((20, 20), dtype=bool)
    proposal = Proposal2D(
        proposal_id=40,
        mask=mask.copy(),
        bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.99,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    np.testing.assert_array_equal(voted[0].mask, mask)
    assert voted[0].metadata["anchor_id"] == -1
    assert voted[0].metadata["anchor_class_name"] == ""
    assert voted[0].metadata["anchor_label_strength"] == "none"
    assert voted[0].metadata["anchor_candidate_classes"] == []
    assert voted[0].metadata["anchor_candidate_ids"] == []
    assert voted[0].metadata["anchor_label_votes"] == {}
    assert assignments[0].anchor_id == -1


def test_anchor_vote_adds_weak_clipped_structure_proposal_for_partial_structure_anchor() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.85,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=3,
                    bbox_xyxy=np.array([5, 5, 15, 15], dtype=np.float32),
                    class_name="wall",
                    confidence=0.8,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.ones((20, 20), dtype=bool)
    proposal = Proposal2D(
        proposal_id=9,
        mask=mask.copy(),
        bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.95,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 2
    whole = next(item for item in voted if item.proposal_id == 9)
    weak = next(item for item in voted if item.proposal_id != 9)

    np.testing.assert_array_equal(whole.mask, mask)
    assert whole.metadata["anchor_id"] == -1
    assert whole.metadata["anchor_class_name"] == ""
    assert whole.metadata["anchor_label_strength"] == "none"

    expected_clip = np.zeros((20, 20), dtype=bool)
    expected_clip[5:15, 5:15] = True
    np.testing.assert_array_equal(weak.mask, expected_clip)
    assert weak.bbox_xyxy.tolist() == [5.0, 5.0, 15.0, 15.0]
    assert weak.area == 100
    assert weak.backend_name == "sam2_anchor_vote"
    assert weak.metadata["anchor_id"] == 3
    assert weak.metadata["anchor_class_name"] == "wall"
    assert weak.metadata["anchor_label_strength"] == "weak_overlap"
    assert weak.metadata["geometry_source"] == "sam2_anchor_overlap"
    assert weak.metadata["mask_source"] == "weak_structure_anchor_box_clip"
    assert weak.metadata["source_raw_proposal_id"] == 9
    assert weak.metadata["source_raw_proposal_ids"] == [9]
    assert weak.metadata["anchor_proposal_coverage"] == 0.25
    assert weak.metadata["anchor_anchor_coverage"] == 1.0

    assert len(assignments) == 2
    assert any(int(item.anchor_id) == -1 for item in assignments)
    assert any(int(item.anchor_id) == 3 and item.class_name == "wall" for item in assignments)


def test_containment_gate_does_not_block_weak_structure_overlap_recovery() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.85,
            "contained_subproposal_gate_enabled": True,
            "contained_max_anchor_coverage": 0.20,
            "contained_min_proposal_coverage": 0.85,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=3,
                    bbox_xyxy=np.array([5, 5, 15, 15], dtype=np.float32),
                    class_name="wall",
                    confidence=0.8,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.ones((20, 20), dtype=bool)
    proposal = Proposal2D(
        proposal_id=9,
        mask=mask.copy(),
        bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.95,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    whole = next(item for item in voted if item.proposal_id == 9)
    weak = next(item for item in voted if item.proposal_id != 9)

    assert whole.metadata["anchor_id"] == -1
    assert whole.metadata["anchor_blocked_candidates"] == []
    assert weak.metadata["anchor_class_name"] == "wall"
    assert weak.metadata["anchor_label_strength"] == "weak_overlap"
    assert any(item.anchor_id == 3 and item.class_name == "wall" for item in assignments)


def test_weak_structure_generator_skips_existing_strong_assignment_anchor() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.85,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )

    mask = np.ones((20, 20), dtype=bool)
    proposal = Proposal2D(
        proposal_id=9,
        mask=mask.copy(),
        bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.95,
        backend_name="sam2",
    )
    anchor = Anchor2D(
        anchor_id=3,
        bbox_xyxy=np.array([5, 5, 15, 15], dtype=np.float32),
        class_name="wall",
        confidence=0.8,
    )
    existing_assignments = [
        AnchorAssignment(
            proposal_id=9,
            anchor_id=3,
            class_name="wall",
            confidence=0.8,
        )
    ]

    weak_proposals, weak_assignments = module._generate_weak_structure_overlap_proposals(
        proposal,
        [anchor],
        existing_assignments,
    )

    assert weak_proposals == []
    assert weak_assignments == []


def test_anchor_vote_does_not_duplicate_weak_structure_when_strong_assignment_exists() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.8,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=2,
                    bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
                    class_name="ceiling",
                    confidence=0.7,
                )
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True
    proposal = Proposal2D(
        proposal_id=4,
        mask=mask.copy(),
        bbox_xyxy=np.array([1, 1, 7, 7], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 1
    assert len(assignments) == 1
    assert voted[0].proposal_id == 4
    assert assignments[0].anchor_id == 2
    assert voted[0].metadata["anchor_class_name"] == "ceiling"
    assert voted[0].metadata["anchor_label_strength"] == "strong"


def test_anchor_vote_assigns_unique_ids_to_multiple_weak_structure_proposals() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "covering_min_proposal_coverage": 0.85,
            "weak_structure_overlap_enabled": True,
            "weak_structure_classes": ["wall", "floor", "ceiling", "blinds", "window"],
            "weak_structure_min_proposal_coverage": 0.05,
            "weak_structure_min_anchor_coverage": 0.2,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=3,
                    bbox_xyxy=np.array([0, 0, 5, 5], dtype=np.float32),
                    class_name="wall",
                    confidence=0.8,
                ),
                Anchor2D(
                    anchor_id=2,
                    bbox_xyxy=np.array([5, 0, 10, 5], dtype=np.float32),
                    class_name="floor",
                    confidence=0.7,
                ),
            ]

    module.backend = _FakeBackend()

    mask = np.ones((20, 20), dtype=bool)
    proposal = Proposal2D(
        proposal_id=9,
        mask=mask.copy(),
        bbox_xyxy=np.array([0, 0, 20, 20], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.95,
        backend_name="sam2",
    )

    _anchors, voted, assignments = module.generate_proposals(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [proposal],
    )

    assert len(voted) == 3
    weak_proposals = [item for item in voted if item.proposal_id != 9]
    assert len(weak_proposals) == 2

    weak_proposal_ids = [int(item.proposal_id) for item in weak_proposals]
    assert len(set(weak_proposal_ids)) == len(weak_proposal_ids)
    assert all(proposal_id < 0 for proposal_id in weak_proposal_ids)
    assert {item.metadata["anchor_class_name"] for item in weak_proposals} == {"wall", "floor"}

    weak_assignment_ids = [
        int(item.proposal_id)
        for item in assignments
        if int(item.proposal_id) in weak_proposal_ids
    ]
    assert len(weak_assignment_ids) == 2
    assert len(set(weak_assignment_ids)) == len(weak_assignment_ids)
    assert set(weak_assignment_ids) == set(weak_proposal_ids)


def test_anchor_vote_label_votes_use_max_confidence_per_class() -> None:
    module = ObjectAnchorModule(
        {
            "enabled": False,
            "assignment_policy": "semantic_vote",
            "proposal_min_area": 1,
            "min_anchor_mask_coverage": 0.1,
            "covering_min_proposal_coverage": 0.1,
        }
    )
    module.enabled = True

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([0, 0, 8, 8], dtype=np.float32),
                    class_name="chair",
                    confidence=0.9,
                ),
                Anchor2D(
                    anchor_id=1,
                    bbox_xyxy=np.array([0, 0, 8, 8], dtype=np.float32),
                    class_name="chair",
                    confidence=0.4,
                ),
            ]

    module.backend = _FakeBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    proposal = Proposal2D(
        proposal_id=8,
        mask=mask.copy(),
        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
        area=int(mask.sum()),
        backend_name="sam2",
    )

    _anchors, voted, _assignments = module.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [proposal],
    )

    assert voted[0].metadata["anchor_candidate_classes"] == ["chair", "chair"]
    assert voted[0].metadata["anchor_label_votes"] == {"chair": 0.9}


def test_object_anchor_generates_anchor_box_primary_proposals() -> None:
    module = ObjectAnchorModule({"enabled": False, "anchor_primary_mode": True})
    module.enabled = True
    module.anchor_primary_mode = True
    module.proposal_min_area = 1

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=4,
                    bbox_xyxy=np.array([1, 1, 4, 5], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.82,
                )
            ]

    module.backend = _FakeBackend()
    rgb = np.zeros((6, 7, 3), dtype=np.uint8)

    anchors, proposals, assignments = module.generate_anchor_box_proposals(rgb)

    assert len(anchors) == 1
    assert len(proposals) == 1
    assert len(assignments) == 1
    proposal = proposals[0]
    assert proposal.proposal_id == 0
    assert proposal.backend_name == "anchor_box_primary"
    assert proposal.area == 12
    assert proposal.metadata["anchor_id"] == 4
    assert proposal.metadata["anchor_class_name"] == "sofa"
    assert proposal.metadata["anchor_label_strength"] == "strong"
    assert proposal.metadata["mask_source"] == "anchor_box"
    assert assignments[0].anchor_id == 4
    assert assignments[0].keepalive is True


def test_anchor_box_primary_duplicate_anchor_ids_get_unique_proposal_ids() -> None:
    module = ObjectAnchorModule({"enabled": False, "anchor_primary_mode": True})
    module.enabled = True
    module.anchor_primary_mode = True
    module.proposal_min_area = 1

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([1, 1, 3, 3], dtype=np.float32),
                    class_name="chair",
                    confidence=0.9,
                ),
                Anchor2D(
                    anchor_id=0,
                    bbox_xyxy=np.array([4, 4, 7, 7], dtype=np.float32),
                    class_name="table",
                    confidence=0.8,
                ),
            ]

    module.backend = _FakeBackend()

    _anchors, proposals, assignments = module.generate_anchor_box_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert [proposal.proposal_id for proposal in proposals] == [0, 1]
    assert len({proposal.proposal_id for proposal in proposals}) == 2
    assert [assignment.proposal_id for assignment in assignments] == [0, 1]
    assert [assignment.anchor_id for assignment in assignments] == [0, 0]
    assert [proposal.metadata["anchor_id"] for proposal in proposals] == [0, 0]
    assert [proposal.metadata["anchor_class_name"] for proposal in proposals] == ["chair", "table"]


def test_anchor_box_primary_clips_boxes_and_skips_min_area_failures() -> None:
    module = ObjectAnchorModule({"enabled": False, "anchor_primary_mode": True})
    module.enabled = True
    module.anchor_primary_mode = True
    module.proposal_min_area = 4

    class _FakeBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=5,
                    bbox_xyxy=np.array([-2, -1, 3, 3], dtype=np.float32),
                    class_name="sofa",
                    confidence=0.7,
                ),
                Anchor2D(
                    anchor_id=6,
                    bbox_xyxy=np.array([6, 6, 8, 8], dtype=np.float32),
                    class_name="lamp",
                    confidence=0.6,
                ),
                Anchor2D(
                    anchor_id=7,
                    bbox_xyxy=np.array([10, 10, 12, 12], dtype=np.float32),
                    class_name="plant",
                    confidence=0.5,
                ),
            ]

    module.backend = _FakeBackend()

    _anchors, proposals, assignments = module.generate_anchor_box_proposals(
        np.zeros((6, 6, 3), dtype=np.uint8),
    )

    assert len(proposals) == 1
    assert proposals[0].proposal_id == 0
    assert proposals[0].area == 9
    np.testing.assert_array_equal(proposals[0].bbox_xyxy, np.array([0, 0, 3, 3], dtype=np.float32))
    assert proposals[0].metadata["anchor_id"] == 5
    assert [assignment.anchor_id for assignment in assignments] == [5]
