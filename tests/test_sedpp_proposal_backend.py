from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.models.sedpp_proposal_backend import (
    SEDPPProposalBackend,
    install_openclip_local_weight_patch,
    load_sedpp_class_config,
    resolve_openclip_weight_path,
    semantic_components_to_proposals,
    write_sed_class_list_json,
)
from src.modules.proposal import ProposalModule
from src.pipelines.main_pipeline import Pipeline


def _frame(h: int = 6, w: int = 8) -> Frame:
    return Frame(
        frame_id=0,
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=np.ones((h, w), dtype=np.float32),
        pose=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(fx=4.0, fy=4.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h),
    )


def test_semantic_components_to_proposals_splits_components_and_stamps_metadata() -> None:
    sem_seg = np.zeros((2, 6, 8), dtype=np.float32)
    sem_seg[0, :, :] = 0.05
    sem_seg[1, :, :] = 0.04
    sem_seg[0, 1:3, 1:3] = 0.91
    sem_seg[0, 3:5, 5:7] = 0.84

    proposals = semantic_components_to_proposals(
        sem_seg,
        ["chair", "wall"],
        min_pixel_confidence=0.5,
        sem_seg_output="probabilities",
        min_component_area=2,
    )

    assert [proposal.proposal_id for proposal in proposals] == [0, 1]
    assert [proposal.area for proposal in proposals] == [4, 4]
    assert all(proposal.backend_name == "sedpp" for proposal in proposals)
    assert proposals[0].metadata["source"] == "sedpp_semantic_component"
    assert proposals[0].metadata["mask_source"] == "sedpp_semantic_component"
    assert proposals[0].metadata["anchor_class_name"] == "chair"
    assert proposals[0].metadata["anchor_label_votes"] == {"chair": pytest.approx(0.91)}
    assert proposals[0].metadata["anchor_label_strength"] == "strong"
    assert proposals[0].metadata["semantic_commit_allowed"] is True
    assert proposals[0].metadata["sedpp_class_index"] == 0
    assert proposals[0].metadata["sedpp_class_name"] == "chair"


def test_semantic_components_drop_low_confidence_and_unmapped_classes() -> None:
    sem_seg = np.zeros((2, 5, 5), dtype=np.float32)
    sem_seg[0, :, :] = 0.05
    sem_seg[1, :, :] = 0.04
    sem_seg[0, 1:3, 1:3] = 0.30
    sem_seg[1, 3:5, 3:5] = 0.95

    proposals = semantic_components_to_proposals(
        sem_seg,
        ["chair", "unknown-class"],
        canonical_map={"unknown-class": ""},
        sem_seg_output="probabilities",
        min_pixel_confidence=0.5,
        min_component_area=1,
    )

    assert proposals == []


def test_semantic_components_reassign_ids_after_sorting_and_cap() -> None:
    sem_seg = np.zeros((1, 6, 7), dtype=np.float32)
    sem_seg[0, :, :] = 0.01
    sem_seg[0, 0:1, 0:2] = 0.60
    sem_seg[0, 2:4, 2:4] = 0.95
    sem_seg[0, 4:6, 5:7] = 0.80

    proposals = semantic_components_to_proposals(
        sem_seg,
        ["table"],
        sem_seg_output="probabilities",
        min_pixel_confidence=0.5,
        min_component_area=1,
        max_proposals=2,
    )

    assert [proposal.proposal_id for proposal in proposals] == [0, 1]
    assert [pytest.approx(proposal.confidence) for proposal in proposals] == [
        pytest.approx(0.95),
        pytest.approx(0.80),
    ]


def test_semantic_components_treat_bounded_logits_as_logits_by_default() -> None:
    sem_seg = np.zeros((2, 4, 4), dtype=np.float32)
    sem_seg[0, :, :] = 0.0
    sem_seg[1, :, :] = 0.0
    sem_seg[0, 1:3, 1:3] = 0.8

    proposals = semantic_components_to_proposals(
        sem_seg,
        ["chair", "wall"],
        min_pixel_confidence=0.65,
        min_component_area=1,
    )

    assert len(proposals) == 1
    assert proposals[0].metadata["anchor_class_name"] == "chair"
    assert proposals[0].confidence == pytest.approx(0.6899745)


def test_single_class_logits_use_sigmoid_not_full_frame_softmax() -> None:
    sem_seg = np.full((1, 5, 5), -5.0, dtype=np.float32)
    sem_seg[0, 1:3, 1:3] = 2.0

    proposals = semantic_components_to_proposals(
        sem_seg,
        ["chair"],
        min_pixel_confidence=0.7,
        min_component_area=1,
    )

    assert len(proposals) == 1
    assert proposals[0].area == 4
    assert proposals[0].bbox_xyxy.tolist() == [1.0, 1.0, 3.0, 3.0]
    assert proposals[0].confidence == pytest.approx(0.880797, rel=1e-5)


def test_structural_components_below_structural_threshold_are_not_committed() -> None:
    sem_seg = np.zeros((1, 4, 4), dtype=np.float32)
    sem_seg[0, 1:3, 1:3] = 0.55

    proposals = semantic_components_to_proposals(
        sem_seg,
        ["wall"],
        sem_seg_output="probabilities",
        min_pixel_confidence=0.5,
        strong_confidence_threshold=0.5,
        structural_confidence_threshold=0.6,
        structure_classes=["wall"],
        min_component_area=1,
    )

    assert len(proposals) == 1
    assert proposals[0].metadata["anchor_label_strength"] == "contextual"
    assert proposals[0].metadata["semantic_commit_allowed"] is False
    assert proposals[0].metadata["sedpp_commit_threshold"] == pytest.approx(0.6)


def test_sedpp_class_config_writes_plain_sed_class_list(tmp_path) -> None:
    class_config = tmp_path / "classes.json"
    class_config.write_text(
        '{"classes": ["chair", "sofa"], "aliases": {"couch": "sofa"}}',
        encoding="utf-8",
    )

    classes, aliases = load_sedpp_class_config(class_config)
    sed_class_json = write_sed_class_list_json(classes)

    assert classes == ["chair", "sofa"]
    assert aliases == {"couch": "sofa"}
    assert Path(sed_class_json).read_text(encoding="utf-8") == '["chair", "sofa"]'


def test_resolve_openclip_weight_path_prefers_config_then_sed_defaults(tmp_path) -> None:
    repo = tmp_path / "SED"
    project = tmp_path / "project"
    configured = project / "weights" / "configured.bin"
    default = (
        repo
        / "weights"
        / "CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"
        / "open_clip_pytorch_model.bin"
    )
    configured.parent.mkdir(parents=True)
    configured.write_bytes(b"configured")
    default.parent.mkdir(parents=True)
    default.write_bytes(b"default")

    assert resolve_openclip_weight_path(
        {"openclip_weight_path": "weights/configured.bin"},
        repo_path=repo,
        project_root=project,
        model_name="convnext_large_d_320",
        pretrained_tag="laion2b_s29b_b131k_ft_soup",
    ) == str(configured)

    assert resolve_openclip_weight_path(
        {},
        repo_path=repo,
        project_root=project,
        model_name="convnext_large_d_320",
        pretrained_tag="laion2b_s29b_b131k_ft_soup",
    ) == str(default)


def test_openclip_local_weight_patch_rewrites_matching_alias(tmp_path) -> None:
    class FakeOpenClip:
        def __init__(self) -> None:
            self.calls = []

        def create_model_and_transforms(self, model_name, pretrained=None, *args, **kwargs):
            self.calls.append((model_name, pretrained))
            return "model", None, None

    weight = tmp_path / "open_clip_pytorch_model.bin"
    weight.write_bytes(b"weights")
    fake = FakeOpenClip()

    install_openclip_local_weight_patch(
        fake,
        model_name="convnext_large_d_320",
        pretrained_tag="laion2b_s29b_b131k_ft_soup",
        weight_path=str(weight),
    )

    assert fake.create_model_and_transforms(
        "convnext_large_d_320",
        pretrained="laion2b_s29b_b131k_ft_soup",
    ) == ("model", None, None)
    assert fake.calls == [("convnext_large_d_320", str(weight))]


def test_sedpp_backend_generate_records_inference_and_postprocess_timings() -> None:
    class FakePredictor:
        def __call__(self, image):
            assert image.shape == (5, 5, 3)
            sem_seg = np.zeros((1, 5, 5), dtype=np.float32)
            sem_seg[0, 1:4, 1:4] = 0.9
            return {"sem_seg": sem_seg}

    backend = SEDPPProposalBackend()
    backend.predictor = FakePredictor()
    backend.class_names = ["sofa"]
    backend.min_pixel_confidence = 0.5
    backend.min_component_area = 2
    backend.max_proposals = 10

    proposals = backend.generate_proposals(
        np.zeros((5, 5, 3), dtype=np.uint8),
        np.ones((5, 5), dtype=np.float32),
        frame=_frame(5, 5),
    )

    assert len(proposals) == 1
    assert proposals[0].metadata["anchor_class_name"] == "sofa"
    assert backend.last_generation_timings["sedpp_inference"] >= 0.0
    assert backend.last_generation_timings["sedpp_postprocess"] >= 0.0
    assert backend.last_debug["kept_component_count"] == 1


def test_proposal_module_sedpp_factory_preserves_backend_timings(monkeypatch) -> None:
    class FakePredictor:
        def __call__(self, image):
            sem_seg = np.zeros((1, 4, 4), dtype=np.float32)
            sem_seg[0, 1:3, 1:3] = 0.9
            return {"sem_seg": sem_seg}

    def fake_initialize(self, config):
        self.predictor = FakePredictor()
        self.class_names = ["chair"]
        self.min_pixel_confidence = 0.5
        self.min_component_area = 1
        self.max_proposals = 10
        self.depth_split_enabled = False
        self.depth_split_max_gap = 0.25

    monkeypatch.setattr(SEDPPProposalBackend, "initialize", fake_initialize)

    module = ProposalModule({"backend": "sedpp", "min_mask_area": 0})
    proposals = module.process(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4), dtype=np.float32),
        frame=_frame(4, 4),
    )

    assert module.active_backend_name == "sedpp"
    assert len(proposals) == 1
    assert "sedpp_inference" in module.last_generation_timings
    assert "sedpp_postprocess" in module.last_generation_timings
    assert proposals[0].metadata["actual_backend"] == "sedpp"


def test_proposal_module_final_filter_reassigns_proposal_ids() -> None:
    class FakeBackend:
        def initialize(self, config):
            self.last_generation_timings = {}

        def generate_proposals(self, rgb, depth, frame=None):
            from src.core.data_structures import Proposal2D

            return [
                Proposal2D(
                    proposal_id=0,
                    mask=np.ones((3, 3), dtype=bool),
                    bbox_xyxy=np.array([0, 0, 3, 3], dtype=np.float32),
                    area=9,
                    backend_name="fake",
                ),
                Proposal2D(
                    proposal_id=1,
                    mask=np.ones((1, 1), dtype=bool),
                    bbox_xyxy=np.array([0, 0, 1, 1], dtype=np.float32),
                    area=1,
                    backend_name="fake",
                ),
                Proposal2D(
                    proposal_id=2,
                    mask=np.ones((2, 2), dtype=bool),
                    bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
                    area=4,
                    backend_name="fake",
                ),
            ]

    module = ProposalModule({"backend": "placeholder", "min_mask_area": 4})
    module.backend = FakeBackend()
    module.active_backend_name = "fake"
    proposals = module.process(np.zeros((3, 3, 3), dtype=np.uint8), np.ones((3, 3), dtype=np.float32))

    assert [proposal.proposal_id for proposal in proposals] == [0, 1]
    assert [proposal.area for proposal in proposals] == [9, 4]


def test_pipeline_build_proposal_bundle_merges_proposal_backend_timings(tmp_path) -> None:
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

    class FakeProposal:
        active_backend_name = "sedpp"
        last_generation_timings = {}

        def process(self, *args, **kwargs):
            self.last_generation_timings = {"sedpp_inference": 0.12, "sedpp_postprocess": 0.03}
            return []

    pipe.proposal = FakeProposal()
    bundle = pipe._build_proposal_bundle(_frame())

    assert bundle.generation_timings["sedpp_inference"] == pytest.approx(0.12)
    assert bundle.generation_timings["sedpp_postprocess"] == pytest.approx(0.03)
    assert "proposal_generation" in bundle.generation_timings


def test_process_frame_exposes_sedpp_backend_timings(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "frame_input:",
                "  image_height: 4",
                "  image_width: 4",
                "pipeline:",
                "  verbose: false",
                "  collect_stage_timings: true",
                "anchor_frontend:",
                "  enabled: false",
                "proposal:",
                "  backend: placeholder",
                "patch_lifting:",
                "  min_points: 1",
                "object_update:",
                "  surface_owner_gate:",
                "    enabled: false",
                "semantic_memory:",
                "  backend: placeholder",
            ]
        ),
        encoding="utf-8",
    )
    pipe = Pipeline(config_path=str(config_path))

    class FakeProposal:
        active_backend_name = "sedpp"
        last_generation_timings = {}

        def process(self, *args, **kwargs):
            self.last_generation_timings = {"sedpp_inference": 0.12, "sedpp_postprocess": 0.03}
            return []

    pipe.proposal = FakeProposal()
    intrinsics = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4)

    pipe.process_frame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4), dtype=np.float32),
        np.eye(4, dtype=np.float64),
        intrinsics,
    )

    timings = pipe.last_frame_debug["stage_timings"]
    assert timings["sedpp_inference"] == pytest.approx(0.12)
    assert timings["sedpp_postprocess"] == pytest.approx(0.03)
