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
from src.models.esam_proposal_backend import ESAMProposalBackend
from src.models.entitysam_proposal_backend import EntitySAMProposalBackend
from src.models.proposal_backend import ProposalBackend
from src.modules.proposal import ProposalModule
from src.utils.visualization import proposal_overlay_image
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
