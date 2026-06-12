"""Tests for proposal backend switching and comparison wiring."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import CameraIntrinsics, Frame, Proposal2D
from src.models.precomputed_proposal_backend import PrecomputedProposalBackend
from src.models.esam_proposal_backend import ESAMProposalBackend
from src.models.entitysam_proposal_backend import EntitySAMProposalBackend
from src.models.sam2_proposal_backend import SAM2ProposalBackend
from src.models.proposal_backend import ProposalBackend
from src.modules.proposal import ProposalModule
from src.utils.visualization import proposal_overlay_image
from frontend.proposal_cache import save_proposals, write_manifest
import run_replica20f_compare_proposals as compare_runner


class _FakeBackend(ProposalBackend):
    def __init__(self) -> None:
        self.init_config = None

    def initialize(self, config: dict) -> None:
        self.init_config = dict(config)

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> list[Proposal2D]:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[1:3, 1:4] = True
        return [
            Proposal2D(
                proposal_id=0,
                mask=mask,
                bbox_xyxy=np.array([1, 1, 4, 3], dtype=np.float32),
                area=int(mask.sum()),
                confidence=0.9,
                backend_name=self.init_config.get("backend", "fake") if self.init_config else "fake",
            )
        ]


class _FakeDataset:
    def __init__(self, frame_count: int = 2) -> None:
        self.frame_count = frame_count

    def iter_frames(self, limit: int | None = None):
        count = self.frame_count if limit is None else min(self.frame_count, limit)
        for frame_id in range(count):
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            depth = np.ones((8, 8), dtype=np.float32)
            yield Frame(
                frame_id=frame_id,
                rgb=rgb,
                depth=depth,
                pose=np.eye(4, dtype=np.float64),
                intrinsics=CameraIntrinsics(fx=5.0, fy=5.0, cx=4.0, cy=4.0, width=8, height=8),
            )

    def summary(self) -> dict:
        return {"frame_count": self.frame_count}


def test_proposal_module_selects_esam_and_merges_nested_config(monkeypatch) -> None:
    import src.models.esam_proposal_backend as esam_backend_module

    monkeypatch.setattr(esam_backend_module, "ESAMProposalBackend", _FakeBackend)
    module = ProposalModule(
        {
            "backend": "esam",
            "min_mask_area": 10,
            "esam": {
                "device": "cpu",
                "repo_root": "/tmp/esam_repo",
                "factory": "local_esam:build_predictor",
            },
        }
    )

    assert isinstance(module.backend, _FakeBackend)
    assert module.backend.init_config["device"] == "cpu"
    assert module.backend.init_config["repo_root"] == "/tmp/esam_repo"
    assert module.backend.init_config["factory"] == "local_esam:build_predictor"
    assert module.backend.init_config["backend"] == "esam"


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


def test_proposal_module_sam3_concept_worker_init_error_does_not_fallback_to_placeholder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProposalModule(
            {
                "backend": "sam3_concept",
                "sam3_concept": {
                    "mode": "worker",
                    "worker_script": str(tmp_path / "missing_worker.py"),
                },
            }
        )


def test_sam3_concept_worker_backend_returns_semantic_proposal(tmp_path: Path) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    assert request["classes"] == ["chair"]
    print(json.dumps({
        "id": request["id"],
        "proposals": [{
            "bbox_xyxy": [1, 2, 5, 6],
            "confidence": 0.82,
            "class_name": "chair",
            "anchor_id": "external-anchor",
            "metadata": {"worker_trace_id": "abc"},
        }],
    }), flush=True)
""",
        encoding="utf-8",
    )

    backend = SAM3ConceptProposalBackend()
    backend.initialize(
        {
            "mode": "worker",
            "classes": ["chair"],
            "worker_python": sys.executable,
            "worker_script": str(worker_script),
            "request_timeout_sec": 0.5,
        }
    )
    proposals = backend.generate_proposals(
        np.zeros((10, 12, 3), dtype=np.uint8),
        np.ones((10, 12), dtype=np.float32),
    )
    backend.close()

    assert len(proposals) == 1
    assert proposals[0].metadata["anchor_id"] == 0
    assert proposals[0].metadata["anchor_class_name"] == "chair"
    assert proposals[0].metadata["anchor_confidence"] == pytest.approx(0.82)
    assert proposals[0].metadata["sam3_metadata"]["anchor_id"] == "external-anchor"
    assert proposals[0].metadata["sam3_metadata"]["worker_trace_id"] == "abc"


def test_sam3_concept_worker_backend_raises_worker_error(tmp_path: Path) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request["id"], "error": "sam3 failed"}), flush=True)
""",
        encoding="utf-8",
    )

    backend = SAM3ConceptProposalBackend()
    backend.initialize({"mode": "worker", "worker_script": str(worker_script)})

    with pytest.raises(RuntimeError, match="sam3 failed"):
        backend.generate_proposals(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4), dtype=np.float32),
        )


def test_sam3_concept_worker_ok_false_without_error_raises(tmp_path: Path) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request["id"], "ok": False}), flush=True)
""",
        encoding="utf-8",
    )

    backend = SAM3ConceptProposalBackend()
    backend.initialize({"mode": "worker", "worker_script": str(worker_script)})

    try:
        with pytest.raises(RuntimeError, match="reported failure"):
            backend.generate_proposals(
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.ones((4, 4), dtype=np.float32),
            )
    finally:
        backend.close()


def test_sam3_concept_worker_response_error_is_reported(tmp_path: Path) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    print(json.dumps({'id': request['id'], 'ok': False, 'error': 'boom'}), flush=True)\n",
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


def test_sam3_concept_rejects_nonpositive_request_timeout() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()

    with pytest.raises(ValueError, match="request_timeout_sec"):
        backend.initialize({"mode": "mock", "request_timeout_sec": 0})


def test_sam3_concept_rejects_invalid_candidate_strategy() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()

    with pytest.raises(ValueError, match="candidate_strategy"):
        backend.initialize({"mode": "mock", "candidate_strategy": "high_recall_everything"})


def test_sam3_concept_rejects_negative_prompt_budget() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()

    with pytest.raises(ValueError, match="max_prompts_per_frame"):
        backend.initialize({"mode": "mock", "max_prompts_per_frame": -1})


def test_sam3_concept_worker_request_limits_prompts_by_round_robin(monkeypatch) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    class RecordingClient:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def request(self, payload: dict) -> dict:
            self.payloads.append(dict(payload))
            return {"id": 1, "ok": True, "proposals": []}

    client = RecordingClient()
    monkeypatch.setattr(SAM3ConceptProposalBackend, "_create_worker_client", lambda self: client)
    backend = SAM3ConceptProposalBackend()
    backend.initialize(
        {
            "mode": "worker",
            "classes": ["wall", "floor", "chair", "table", "sofa"],
            "always_include_classes": ["wall"],
            "candidate_strategy": "round_robin",
            "max_prompts_per_frame": 3,
        }
    )
    frame = Frame(
        frame_id=1,
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4),
        source_frame_id=10,
    )

    proposals = backend.generate_proposals(frame.rgb, frame.depth, frame=frame)

    assert proposals == []
    assert client.payloads[-1]["classes"] == ["wall", "chair", "table"]
    assert client.payloads[-1]["max_prompts_per_frame"] == 3
    assert client.payloads[-1]["prompt_selection"]["candidate_strategy"] == "round_robin"
    assert client.payloads[-1]["prompt_selection"]["full_class_count"] == 5
    assert backend.last_generation_info["selected_classes"] == ["wall", "chair", "table"]
    assert backend.last_generation_info["selected_class_count"] == 3


def test_sam3_concept_mock_backend_uses_same_prompt_budget_selection() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()
    backend.initialize(
        {
            "mode": "mock",
            "classes": ["wall", "floor", "chair", "table", "sofa"],
            "always_include_classes": ["wall"],
            "candidate_strategy": "round_robin",
            "max_prompts_per_frame": 3,
            "max_proposals": 10,
        }
    )
    frame = Frame(
        frame_id=1,
        rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        depth=np.ones((20, 20), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=10.0, cy=10.0, width=20, height=20),
        source_frame_id=10,
    )

    proposals = backend.generate_proposals(frame.rgb, frame.depth, frame=frame)

    assert [proposal.metadata["anchor_class_name"] for proposal in proposals] == [
        "wall",
        "chair",
        "table",
    ]
    assert backend.last_generation_info["selected_classes"] == ["wall", "chair", "table"]


def test_sam3_concept_round_robin_uses_processed_frame_id_for_stride_coverage() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()
    backend.initialize(
        {
            "mode": "mock",
            "classes": ["wall", "c0", "c1", "c2", "c3"],
            "always_include_classes": ["wall"],
            "candidate_strategy": "round_robin",
            "max_prompts_per_frame": 2,
        }
    )
    source_frame_ids = [0, 10, 20, 30]
    selected = []
    for processed_frame_id, source_frame_id in enumerate(source_frame_ids):
        frame = Frame(
            frame_id=processed_frame_id,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.ones((4, 4), dtype=np.float32),
            pose=np.eye(4, dtype=np.float64),
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4),
            source_frame_id=source_frame_id,
        )
        selected.append(backend._select_classes_for_frame(frame))

    assert selected == [["wall", "c0"], ["wall", "c1"], ["wall", "c2"], ["wall", "c3"]]


def test_sam3_concept_worker_reports_missing_official_api_with_request_id(tmp_path: Path) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    sam3_root = tmp_path / "fake_sam3_root"
    package_root = sam3_root / "sam3"
    (package_root / "model").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "model_builder.py").write_text("", encoding="utf-8")
    (package_root / "model" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "model" / "sam3_image_processor.py").write_text("", encoding="utf-8")

    client = JsonLineWorkerClient(
        command=[
            sys.executable,
            "scripts/sam3_concept_worker.py",
            "--sam3-repo-root",
            str(sam3_root),
        ],
        env={},
        cwd=str(Path(__file__).resolve().parent.parent),
        request_timeout_sec=1.0,
    )

    try:
        response = client.request({"classes": ["chair"]})
    finally:
        client.close()

    assert response["id"] == 1
    assert response["ok"] is False
    assert "does not expose official image API" in response["error"]


def test_sam3_concept_worker_runs_fake_official_api_and_filters_sorts_caps(tmp_path: Path) -> None:
    from src.models.json_line_worker_client import JsonLineWorkerClient

    sam3_root = tmp_path / "fake_sam3_root"
    package_root = sam3_root / "sam3"
    (package_root / "model").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "model" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "model_builder.py").write_text(
        """
class FakeModel:
    pass


def build_sam3_image_model(device="cuda", checkpoint_path=None, **kwargs):
    return FakeModel()
""",
        encoding="utf-8",
    )
    (package_root / "model" / "sam3_image_processor.py").write_text(
        """
import numpy as np


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)


class Sam3Processor:
    def __init__(self, model, device="cuda", confidence_threshold=0.0, **kwargs):
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold

    def set_image(self, image):
        return {"image_size": image.size, "prompts": []}

    def set_text_prompt(self, state, prompt):
        state["prompts"].append(prompt)
        if prompt == "chair":
            return {
                "masks": FakeTensor([
                    [[1, 0], [0, 0]],
                    [[1, 1], [0, 0]],
                ]),
                "boxes": FakeTensor([[0, 0, 1, 1], [0, 0, 2, 1]]),
                "scores": FakeTensor([0.2, 0.9]),
            }
        return {
            "masks": FakeTensor([[[0, 1], [0, 1]]]),
            "boxes": FakeTensor([[1, 0, 2, 2]]),
            "scores": FakeTensor([0.8]),
        }
""",
        encoding="utf-8",
    )

    client = JsonLineWorkerClient(
        command=[sys.executable, "scripts/sam3_concept_worker.py", "--sam3-repo-root", str(sam3_root)],
        env={},
        cwd=str(Path(__file__).resolve().parent.parent),
        request_timeout_sec=1.0,
    )

    try:
        response = client.request(
            {
                "image": {
                    "shape": [2, 2, 3],
                    "dtype": "uint8",
                    "encoding": "base64",
                    "data": "AAAAAAAAAAAAAAAA",
                },
                "classes": ["chair", "table"],
                "confidence_threshold": 0.5,
                "max_proposals": 2,
            }
        )
    finally:
        client.close()

    assert response["ok"] is True
    assert response["raw_mask_count"] == 3
    assert [proposal["label"] for proposal in response["proposals"]] == ["chair", "table"]
    assert [proposal["confidence"] for proposal in response["proposals"]] == [0.9, 0.8]
    assert response["proposals"][0]["metadata"] == {"sam3_score": 0.9, "prompt": "chair"}
    assert response["proposals"][0]["mask"] == [[True, True], [False, False]]
    assert response["proposals"][1]["mask"] == [[False, True], [False, True]]


def test_sam3_concept_worker_only_serializes_kept_topk_masks(tmp_path: Path) -> None:
    import scripts.sam3_concept_worker as worker

    mask_conversions = 0
    original_asarray = worker.np.asarray

    def counting_asarray(value, *args, **kwargs):
        nonlocal mask_conversions
        if kwargs.get("dtype") is bool:
            mask_conversions += 1
        return original_asarray(value, *args, **kwargs)

    class FakeProcessor:
        def set_image(self, image):
            return {"image": image}

        def set_text_prompt(self, state, prompt):
            del state, prompt
            masks = np.zeros((5, 2, 2), dtype=np.uint8)
            masks[:, 0, 0] = 1
            return {
                "masks": masks,
                "scores": np.array([0.1, 0.95, 0.2, 0.9, 0.3], dtype=np.float32),
                "boxes": np.tile(np.array([[0, 0, 1, 1]], dtype=np.float32), (5, 1)),
            }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(worker.np, "asarray", counting_asarray)
    try:
        response = worker._run_request(
            FakeProcessor(),
            {
                "image": {
                    "shape": [2, 2, 3],
                    "dtype": "uint8",
                    "encoding": "base64",
                    "data": "AAAAAAAAAAAAAAAA",
                },
                "classes": ["chair"],
                "confidence_threshold": 0.5,
                "max_proposals": 2,
            },
        )
    finally:
        monkeypatch.undo()

    assert response["raw_mask_count"] == 5
    assert [proposal["confidence"] for proposal in response["proposals"]] == pytest.approx([0.95, 0.9])
    assert mask_conversions == 2


def test_sam3_concept_worker_squeezes_single_channel_mask_dimension() -> None:
    import scripts.sam3_concept_worker as worker

    class FakeProcessor:
        def set_image(self, image):
            return {"image": image}

        def set_text_prompt(self, state, prompt):
            del state, prompt
            return {
                "masks": np.array([[[[1, 0], [0, 1]]]], dtype=np.uint8),
                "scores": np.array([0.9], dtype=np.float32),
                "boxes": np.array([[0, 0, 2, 2]], dtype=np.float32),
            }

    response = worker._run_request(
        FakeProcessor(),
        {
            "image": {
                "shape": [2, 2, 3],
                "dtype": "uint8",
                "encoding": "base64",
                "data": "AAAAAAAAAAAAAAAA",
            },
            "classes": ["chair"],
            "confidence_threshold": 0.0,
            "max_proposals": 1,
        },
    )

    assert response["raw_mask_count"] == 1
    assert response["proposals"][0]["mask"] == [[True, False], [False, True]]


def test_sam3_concept_worker_caps_prompt_calls_before_inference() -> None:
    import scripts.sam3_concept_worker as worker

    class FakeProcessor:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def set_image(self, image):
            return {"image": image}

        def set_text_prompt(self, state, prompt):
            del state
            self.prompts.append(prompt)
            return {
                "masks": np.ones((1, 2, 2), dtype=np.uint8),
                "scores": np.array([0.9], dtype=np.float32),
                "boxes": np.array([[0, 0, 2, 2]], dtype=np.float32),
            }

    processor = FakeProcessor()

    response = worker._run_request(
        processor,
        {
            "image": {
                "shape": [2, 2, 3],
                "dtype": "uint8",
                "encoding": "base64",
                "data": "AAAAAAAAAAAAAAAA",
            },
            "classes": ["wall", "floor", "chair", "table"],
            "max_prompts_per_frame": 2,
            "confidence_threshold": 0.0,
            "max_proposals": 10,
        },
    )

    assert processor.prompts == ["wall", "floor"]
    assert response["prompt_count"] == 2
    assert response["raw_mask_count"] == 2
    assert [proposal["label"] for proposal in response["proposals"]] == ["wall", "floor"]


def test_sam3_concept_backend_relative_worker_script_runs_from_repo_root() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()
    backend.initialize(
        {
            "mode": "worker",
            "worker_python": sys.executable,
            "worker_script": "scripts/sam3_concept_worker.py",
            "request_timeout_sec": 1.0,
        }
    )

    try:
        with pytest.raises(RuntimeError, match="SAM3 package is not importable"):
            backend.generate_proposals(
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.ones((4, 4), dtype=np.float32),
            )
    finally:
        backend.close()


def test_sam3_concept_worker_non_numeric_anchor_ids_follow_sorted_proposal_ids(tmp_path: Path) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    worker_script = tmp_path / "worker.py"
    worker_script.write_text(
        """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({
        "id": request["id"],
        "proposals": [
            {
                "bbox_xyxy": [0, 0, 2, 2],
                "confidence": 0.2,
                "class_name": "low",
                "anchor_id": "raw-low",
            },
            {
                "bbox_xyxy": [2, 2, 5, 5],
                "confidence": 0.9,
                "class_name": "high",
                "anchor_id": "raw-high",
            },
        ],
    }), flush=True)
""",
        encoding="utf-8",
    )

    backend = SAM3ConceptProposalBackend()
    backend.initialize({"mode": "worker", "worker_script": str(worker_script)})

    proposals = backend.generate_proposals(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
    )
    backend.close()

    assert [proposal.metadata["anchor_class_name"] for proposal in proposals] == ["high", "low"]
    assert [proposal.proposal_id for proposal in proposals] == [0, 1]
    assert [proposal.metadata["anchor_id"] for proposal in proposals] == [0, 1]
    assert [proposal.metadata["sam3_metadata"]["anchor_id"] for proposal in proposals] == [
        "raw-high",
        "raw-low",
    ]


def test_sam3_concept_worker_mode_requires_existing_script(tmp_path: Path) -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()

    with pytest.raises(FileNotFoundError):
        backend.initialize({"mode": "worker", "worker_script": str(tmp_path / "missing.py")})


def test_sam3_concept_rejects_negative_max_proposals() -> None:
    from src.models.sam3_concept_backend import SAM3ConceptProposalBackend

    backend = SAM3ConceptProposalBackend()

    with pytest.raises(ValueError, match="max_proposals"):
        backend.initialize({"mode": "mock", "max_proposals": -1})


def test_esam_backend_normalizes_sam_annotations_output() -> None:
    backend = ESAMProposalBackend()
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 2:5] = True
    proposals = backend._normalize_outputs(
        {
            "annotations": [
                {
                    "segmentation": mask,
                    "bbox": [2, 1, 3, 3],
                    "predicted_iou": 0.87,
                    "entity_id": 7,
                }
            ]
        }
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.backend_name == "esam"
    assert proposal.confidence == pytest.approx(0.87)
    assert proposal.metadata["entity_id"] == 7
    np.testing.assert_allclose(proposal.bbox_xyxy, np.array([2, 1, 5, 4], dtype=np.float32))


def test_compare_runner_writes_manifest_with_fake_backend(monkeypatch, tmp_path: Path) -> None:
    fake_module = types.SimpleNamespace(ProposalModule=lambda config: _FakeProposalModule(config))
    monkeypatch.setattr(compare_runner, "ProposalModule", fake_module.ProposalModule)

    dataset = _FakeDataset(frame_count=2)
    output_dir = tmp_path / "sam2"
    manifest = compare_runner.run_backend(
        backend_name="sam2",
        backend_config={"backend": "sam2", "sam2": {"device": "cpu"}},
        dataset=dataset,
        num_frames=2,
        output_dir=output_dir,
    )

    manifest_path = output_dir / "manifest.json"
    assert manifest["status"] == "ok"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded["backend"] == "sam2"
    assert loaded["summary"]["num_frames"] == 2


def test_compare_runner_writes_manifest_for_esam_backend(monkeypatch, tmp_path: Path) -> None:
    fake_module = types.SimpleNamespace(ProposalModule=lambda config: _FakeProposalModule(config))
    monkeypatch.setattr(compare_runner, "ProposalModule", fake_module.ProposalModule)

    dataset = _FakeDataset(frame_count=1)
    output_dir = tmp_path / "esam"
    manifest = compare_runner.run_backend(
        backend_name="esam",
        backend_config={"backend": "esam", "esam": {"device": "cpu"}},
        dataset=dataset,
        num_frames=1,
        output_dir=output_dir,
    )

    manifest_path = output_dir / "manifest.json"
    loaded = json.loads(manifest_path.read_text())
    assert manifest["status"] == "ok"
    assert loaded["backend"] == "esam"
    assert loaded["frames"][0]["frontend"] == "esam"


def test_compare_runner_build_esam_config() -> None:
    args = types.SimpleNamespace(
        esam_min_mask_area=64,
        esam_max_proposals=32,
        esam_confidence_threshold=0.12,
        esam_device="cpu",
        esam_repo_root=Path("/tmp/esam_repo"),
        esam_python_path=["/opt/esam"],
        esam_factory="my_adapter:build_predictor",
        esam_predictor="",
        esam_predictor_is_factory=False,
        esam_call_mode="auto",
        esam_bbox_format="auto",
        esam_config_path=None,
        esam_checkpoint_path=None,
    )

    config = compare_runner.build_esam_config(args)
    assert config["backend"] == "esam"
    assert config["min_mask_area"] == 64
    assert config["esam"]["device"] == "cpu"
    assert config["esam"]["repo_root"] == "/tmp/esam_repo"
    assert config["esam"]["factory"] == "my_adapter:build_predictor"


def test_visualization_accepts_backend_provenance_from_both_backends() -> None:
    sam2_mask = np.zeros((8, 8), dtype=bool)
    sam2_mask[1:4, 1:4] = True
    esam_mask = np.zeros((8, 8), dtype=bool)
    esam_mask[4:7, 4:7] = True
    proposals = [
        Proposal2D(
            proposal_id=0,
            mask=sam2_mask,
            bbox_xyxy=np.array([1, 1, 4, 4], dtype=np.float32),
            area=int(sam2_mask.sum()),
            confidence=0.9,
            backend_name="sam2",
        ),
        Proposal2D(
            proposal_id=1,
            mask=esam_mask,
            bbox_xyxy=np.array([4, 4, 7, 7], dtype=np.float32),
            area=int(esam_mask.sum()),
            confidence=0.85,
            backend_name="esam",
        ),
    ]

    image = proposal_overlay_image(np.zeros((8, 8, 3), dtype=np.uint8), proposals)
    arr = np.asarray(image)
    assert arr.shape == (8, 8, 3)
    assert arr.sum() > 0


def test_proposal_module_selects_entitysam_and_merges_nested_config(monkeypatch) -> None:
    import src.models.entitysam_proposal_backend as entitysam_backend_module

    monkeypatch.setattr(entitysam_backend_module, "EntitySAMProposalBackend", _FakeBackend)
    module = ProposalModule(
        {
            "backend": "entitysam",
            "min_mask_area": 10,
            "entitysam": {
                "device": "cpu",
                "repo_root": "/home/phl/vv/paper2/entitysam",
                "sequence_dir": "/tmp/room0/rgb",
                "config_path": "configs/sam2.1_hiera_l.yaml",
                "checkpoint_path": "checkpoints/vit-l/model_0009999.pth",
            },
        }
    )

    assert isinstance(module.backend, _FakeBackend)
    assert module.backend.init_config["device"] == "cpu"
    assert module.backend.init_config["repo_root"] == "/home/phl/vv/paper2/entitysam"
    assert module.backend.init_config["sequence_dir"] == "/tmp/room0/rgb"
    assert module.backend.init_config["config_path"] == "configs/sam2.1_hiera_l.yaml"
    assert module.backend.init_config["backend"] == "entitysam"


def test_compare_runner_writes_manifest_for_entitysam_backend(monkeypatch, tmp_path: Path) -> None:
    fake_module = types.SimpleNamespace(ProposalModule=lambda config: _FakeProposalModule(config))
    monkeypatch.setattr(compare_runner, "ProposalModule", fake_module.ProposalModule)

    dataset = _FakeDataset(frame_count=2)
    output_dir = tmp_path / "entitysam"
    manifest = compare_runner.run_backend(
        backend_name="entitysam",
        backend_config={"backend": "entitysam", "entitysam": {"device": "cpu"}},
        dataset=dataset,
        num_frames=2,
        output_dir=output_dir,
    )

    manifest_path = output_dir / "manifest.json"
    loaded = json.loads(manifest_path.read_text())
    assert manifest["status"] == "ok"
    assert loaded["backend"] == "entitysam"
    assert loaded["frames"][0]["frontend"] == "entitysam"


def test_compare_runner_build_entitysam_config() -> None:
    args = types.SimpleNamespace(
        entitysam_min_mask_area=64,
        entitysam_max_proposals=32,
        entitysam_confidence_threshold=0.12,
        entitysam_device="cpu",
        entitysam_repo_root=Path("/home/phl/vv/paper2/entitysam"),
        entitysam_config_path="configs/sam2.1_hiera_l.yaml",
        entitysam_checkpoint_path="checkpoints/vit-l/model_0009999.pth",
        entitysam_mask_decoder_depth=8,
        entitysam_points_per_side=16,
        entitysam_stability_score_th=0.95,
        entitysam_nms_iou_th=0.8,
        entitysam_min_mask_region_area=100,
        entitysam_mask_binary_threshold=0.5,
        entitysam_object_mask_threshold=0.05,
        entitysam_topk_per_frame=100,
        entitysam_use_m2m=False,
    )

    config = compare_runner.build_entitysam_config(args)
    assert config["backend"] == "entitysam"
    assert config["min_mask_area"] == 64
    assert config["entitysam"]["device"] == "cpu"
    assert config["entitysam"]["repo_root"] == "/home/phl/vv/paper2/entitysam"
    assert config["entitysam"]["sequence_dir"] == ""
    assert config["entitysam"]["config_path"] == "configs/sam2.1_hiera_l.yaml"
    assert config["entitysam"]["mask_decoder_depth"] == 8


def test_visualization_accepts_entitysam_provenance() -> None:
    entitysam_mask = np.zeros((8, 8), dtype=bool)
    entitysam_mask[2:6, 2:6] = True
    proposals = [
        Proposal2D(
            proposal_id=0,
            mask=entitysam_mask,
            bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
            area=int(entitysam_mask.sum()),
            confidence=0.88,
            backend_name="entitysam",
            metadata={"backend": "entitysam"},
        ),
    ]

    image = proposal_overlay_image(np.zeros((8, 8, 3), dtype=np.uint8), proposals)
    arr = np.asarray(image)
    assert arr.shape == (8, 8, 3)
    assert arr.sum() > 0


def test_sam2_import_ignores_inaccessible_fallback_after_preferred_root(monkeypatch, tmp_path: Path) -> None:
    preferred_root = tmp_path / "sam2_repo"
    (preferred_root / "sam2").mkdir(parents=True)
    (preferred_root / "sam2" / "__init__.py").write_text("", encoding="utf-8")
    (preferred_root / "sam2" / "build_sam.py").write_text("build_sam2 = object()\n", encoding="utf-8")
    (preferred_root / "sam2" / "automatic_mask_generator.py").write_text(
        "SAM2AutomaticMaskGenerator = object()\n", encoding="utf-8"
    )

    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if str(path) == "/home/phl/vv/paper2/OVO/thirdParty/segment-anything-2":
            raise PermissionError("blocked fallback")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    build_sam2, mask_generator, origin = SAM2ProposalBackend._import_sam2(preferred_root)

    assert build_sam2 is not None
    assert mask_generator is not None
    assert str(preferred_root) in origin



def test_precomputed_backend_replays_cached_proposals(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:6] = True
    proposals = [
        Proposal2D(
            proposal_id=3,
            mask=mask,
            bbox_xyxy=np.array([2, 2, 6, 5], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.91,
            backend_name="sam2",
            metadata={"source": "cached"},
        )
    ]
    save_proposals(
        cache_dir,
        frame_id=0,
        image_shape=(8, 8),
        proposals=proposals,
        source_backend="sam2",
        generation_info={"mode": "test"},
    )
    write_manifest(
        cache_dir,
        dataset_summary={"frame_count": 1},
        backend_config={"backend": "sam2"},
        frame_entries=[{"frame_id": 0, "file": "frames/frame000000_proposals.npz", "proposal_count": 1}],
    )

    module = ProposalModule(
        {
            "backend": "precomputed",
            "min_mask_area": 1,
            "precomputed": {
                "cache_dir": str(cache_dir),
                "strict": True,
            },
        }
    )
    assert isinstance(module.backend, PrecomputedProposalBackend)

    frame = Frame(
        frame_id=0,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=5.0, fy=5.0, cx=4.0, cy=4.0, width=8, height=8),
    )
    loaded = module.process(frame.rgb, frame.depth, frame=frame)

    assert module.active_backend_name == "precomputed"
    assert len(loaded) == 1
    assert loaded[0].proposal_id == 3
    assert loaded[0].metadata["cache_source"] == "precomputed_npz"
    assert loaded[0].metadata["source_backend"] == "sam2"


def test_precomputed_backend_prefers_source_frame_id_for_cache_lookup(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:4, 2:5] = True
    proposals = [
        Proposal2D(
            proposal_id=7,
            mask=mask,
            bbox_xyxy=np.array([2, 1, 5, 4], dtype=np.float32),
            area=int(mask.sum()),
            confidence=0.87,
            backend_name="sam2",
        )
    ]
    save_proposals(
        cache_dir,
        frame_id=10,
        image_shape=(8, 8),
        proposals=proposals,
        source_backend="sam2",
    )
    write_manifest(
        cache_dir,
        dataset_summary={"frame_count": 1},
        backend_config={"backend": "sam2"},
        frame_entries=[{"frame_id": 10, "file": "frames/frame000010_proposals.npz", "proposal_count": 1}],
    )

    module = ProposalModule(
        {
            "backend": "precomputed",
            "min_mask_area": 1,
            "precomputed": {
                "cache_dir": str(cache_dir),
                "strict": True,
            },
        }
    )
    frame = Frame(
        frame_id=0,
        source_frame_id=10,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=5.0, fy=5.0, cx=4.0, cy=4.0, width=8, height=8),
    )

    loaded = module.process(frame.rgb, frame.depth, frame=frame)

    assert len(loaded) == 1
    assert loaded[0].proposal_id == 7
    assert loaded[0].metadata["cache_frame_id"] == 10
    assert module.backend.last_generation_info["frame_id"] == 0
    assert module.backend.last_generation_info["cache_frame_id"] == 10


def test_precomputed_backend_requires_manifest_when_strict(tmp_path: Path) -> None:
    backend = PrecomputedProposalBackend()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        backend.initialize({"cache_dir": str(cache_dir), "strict": True})


class _FakeProposalModule:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.backend = types.SimpleNamespace(last_generation_info={"loader": "fake"})

    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> list[Proposal2D]:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[1:3, 1:4] = True
        return [
            Proposal2D(
                proposal_id=0,
                mask=mask,
                bbox_xyxy=np.array([1, 1, 4, 3], dtype=np.float32),
                area=int(mask.sum()),
                confidence=0.9,
                backend_name=self.config["backend"],
            )
        ]
