from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from src.oviv2.observation_query.contracts import ObservationBank
from src.oviv2.observation_query.model import (
    FrozenReSceneFeatures,
    ObservationQueryBranch,
    ObservationReScene,
    ObservationTensorBank,
)


def _bank(*, empty: bool = False) -> ObservationBank:
    if empty:
        return ObservationBank(
            pair_id="pair",
            model_input_sha256="a" * 64,
            region_keys=(),
            region_visit_ids=np.empty(0, dtype=np.int8),
            region_frame_ids=np.empty(0, dtype=np.int64),
            region_features=np.empty((0, 2), dtype=np.float32),
            region_metadata=np.empty((0, 15), dtype=np.float32),
            csr_indptr=np.asarray([0], dtype=np.int64),
            csr_model_indices=np.empty(0, dtype=np.int64),
            csr_weights=np.empty(0, dtype=np.float32),
            region_reliability=np.empty(0, dtype=np.float32),
            model_visit_ids=np.asarray([0, 0, 1], dtype=np.int8),
            source_manifest={},
        )
    return ObservationBank(
        pair_id="pair",
        model_input_sha256="a" * 64,
        region_keys=("r0", "r1"),
        region_visit_ids=np.asarray([0, 0], dtype=np.int8),
        region_frame_ids=np.asarray([0, 1], dtype=np.int64),
        region_features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        region_metadata=np.zeros((2, 15), dtype=np.float32),
        csr_indptr=np.asarray([0, 2, 3], dtype=np.int64),
        csr_model_indices=np.asarray([0, 1, 1], dtype=np.int64),
        csr_weights=np.asarray([1.0, 1.0, 3.0], dtype=np.float32),
        region_reliability=np.asarray([1.0, 0.5], dtype=np.float32),
        model_visit_ids=np.asarray([0, 0, 1], dtype=np.int8),
        source_manifest={},
    )


def test_sparse_surface_correction_is_normalized_and_leaves_unsupported_rows_zero() -> (
    None
):
    bank = ObservationTensorBank.from_observation_bank(_bank(), device="cpu")
    branch = ObservationQueryBranch(
        observation_feature_dim=2,
        metadata_dim=15,
        hidden_dim=4,
        num_queries=2,
        num_heads=2,
        logical_layer_count=1,
        alpha_initial=0.001,
        beta_initial=0.0,
    )
    region_logits = torch.log(
        torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=torch.float32)
    )

    correction, supported = branch.surface_logit_correction(
        region_logits, bank, layer_index=0
    )

    expected_m1 = (torch.tensor([0.8, 0.1]) + 1.5 * torch.tensor([0.2, 0.7])) / 2.5
    expected_m1 = torch.log(expected_m1) + math.log(3.0)
    torch.testing.assert_close(
        correction[0], torch.log(torch.tensor([0.8, 0.1])) + math.log(3.0)
    )
    torch.testing.assert_close(correction[1], expected_m1)
    torch.testing.assert_close(correction[2], torch.zeros(2))
    torch.testing.assert_close(supported, torch.tensor([True, True, False]))


def test_zero_beta_preserves_3d_logits_and_obs_branch_has_gradients() -> None:
    torch.manual_seed(4)
    branch = ObservationQueryBranch(
        observation_feature_dim=2,
        metadata_dim=15,
        hidden_dim=4,
        num_queries=2,
        num_heads=2,
        logical_layer_count=1,
        alpha_initial=0.001,
        beta_initial=0.0,
    )
    bank = ObservationTensorBank.from_observation_bank(_bank(), device="cpu")
    queries = torch.randn(1, 2, 4, requires_grad=True)
    crossed = torch.randn(1, 2, 4, requires_grad=True)
    base_logits = torch.randn(1, 3, 2, requires_grad=True)
    tokens = branch.encode_regions(bank)

    region_logits = branch.predict_region_logits(queries, tokens)
    corrected, _ = branch.correct_mask_logits(
        base_logits, region_logits, bank, layer_index=0
    )
    updated = branch.attend_observations(crossed, tokens, layer_index=0)

    torch.testing.assert_close(corrected, base_logits)
    (corrected.sum() + region_logits.square().sum() + updated.sum()).backward()
    assert branch.obs_feature_projection.weight.grad is not None
    assert branch.obs_query_projection.weight.grad is not None
    assert branch.obs_cross_attention[0].in_proj_weight.grad is not None
    assert branch.obs_raw_alpha.grad is not None
    assert branch.obs_raw_beta.grad is not None


class _Sparse:
    def __init__(self, features: torch.Tensor, coordinates: torch.Tensor):
        self.F = features
        self.decomposed_features = [features]
        self.decomposed_coordinates = [coordinates]


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.PLANES = [4, 4, 4, 4, 4]
        self.features = nn.Parameter(
            torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        )

    def forward(self, _x):
        coordinates = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 1.0]]
        )
        sparse = _Sparse(self.features, coordinates)
        return sparse, [sparse], [[coordinates]]


class _LoggedLayer(nn.Module):
    def __init__(self, name: str, events: list[str]):
        super().__init__()
        self.name = name
        self.events = events

    def forward(self, value, *_args, **_kwargs):
        self.events.append(self.name)
        return value + 0.01


class _Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_on_segments = False
        self.backbone = _Backbone()
        self.mask_dim = 4
        self.num_queries = 2
        self.num_heads = 2
        self.num_decoders = 1
        self.hlevels = [0]
        self.shared_decoder = True
        self.sample_sizes = [16]
        self.use_level_embed = False
        self.temporal_masking = False
        self.save_segment_info = False
        self.events: list[str] = []
        self.cross_attention = nn.ModuleList(
            [nn.ModuleList([_LoggedLayer("cross", self.events)])]
        )
        self.self_attention = nn.ModuleList(
            [nn.ModuleList([_LoggedLayer("self", self.events)])]
        )
        self.ffn_attention = nn.ModuleList(
            [nn.ModuleList([_LoggedLayer("ffn", self.events)])]
        )
        self.lin_squeeze = nn.ModuleList([nn.ModuleList([nn.Identity()])])
        self.query_seed = nn.Parameter(torch.zeros(1, 2, 4))
        self.mask_embed = nn.Linear(4, 4, bias=False)
        self.class_embed = nn.Linear(4, 3)
        self.attn_seen: list[torch.Tensor] = []
        self.disabled_result = {"native": torch.tensor(7)}

    def forward(self, *_args, **_kwargs):
        return self.disabled_result

    def get_pos_encs(self, _coords):
        return [[[torch.zeros(3, 4)]]]

    def aggregate_features(self, pcd_features, _point2segment):
        return pcd_features.decomposed_features, pcd_features.decomposed_coordinates

    def sample_and_batch_features(self, decomposed, *_args, extra=(), **_kwargs):
        value = torch.stack(decomposed)
        padding = torch.zeros(value.shape[:2], dtype=torch.bool)
        if not extra:
            return value, padding
        return value, torch.stack(extra[0]), torch.stack(extra[1]), padding

    def initialize_queries(self, **_kwargs):
        return self.query_seed, torch.zeros_like(self.query_seed), None

    def mask_module(self, query_feat, features):
        embedded = self.mask_embed(query_feat)
        return (
            self.class_embed(query_feat),
            None,
            torch.bmm(features, embedded.transpose(1, 2)),
        )

    def unstack_batched(self, values, padding):
        return [value[~mask] for value, mask in zip(values, padding)]

    def attn_mask(self, logits, *_args, **_kwargs):
        self.attn_seen.append(logits.detach().clone())
        return [logits[0].detach().sigmoid() < 0.5]

    def _set_aux_loss(self, classes, masks, changes):
        return [
            {"pred_logits": c, "pred_masks": m, "pred_changes": h}
            for c, m, h in zip(classes[:-1], masks[:-1], changes[:-1])
        ]


class _Input:
    pass


def test_wrapper_disabled_and_empty_bank_are_exact_native_fallbacks() -> None:
    native = _Native()
    model = ObservationReScene(native, observation_feature_dim=2, metadata_dim=15)
    x = _Input()

    assert model(x, observations=None) is native.disabled_result
    assert (
        model(x, observations=_bank(), observation_mode="disabled")
        is native.disabled_result
    )
    assert model(x, observations=_bank(empty=True)) is native.disabled_result


def test_enabled_wrapper_inserts_obs_attention_before_self_and_uses_corrected_mask() -> (
    None
):
    torch.manual_seed(8)
    native = _Native()
    model = ObservationReScene(native, observation_feature_dim=2, metadata_dim=15)
    model.obs_branch.obs_cross_attention[0].register_forward_hook(
        lambda *_args: native.events.append("obs")
    )
    with torch.no_grad():
        model.obs_branch.obs_raw_beta.fill_(torch.atanh(torch.tensor(0.5)))

    output = model(
        _Input(), observations=_bank(), observation_mode="full", return_trace=True
    )

    assert native.events == ["cross", "obs", "self", "ffn"]
    assert output["pred_masks"][0].shape == (3, 2)
    assert output["pred_region_logits"].shape == (2, 3)
    assert len(output["region_aux_outputs"]) == 1
    torch.testing.assert_close(
        native.attn_seen[0][0], output["aux_outputs"][0]["pred_masks"][0]
    )
    assert output["observation_stats"] == {
        "region_count": 2,
        "edge_count": 3,
        "supported_model_count": 2,
        "feature_fusion_count": 0,
        "attention_layer_count": 1,
        "feedback_layer_count": 2,
    }
    assert set(output["trace"]) == {
        "layer_ids",
        "region_logits",
        "base_mask_logits",
        "corrected_mask_logits",
    }


def test_observation_parameter_namespace_and_initial_gates_are_bounded() -> None:
    model = ObservationReScene(_Native(), observation_feature_dim=2, metadata_dim=15)
    observation_names = [
        name for name, _ in model.named_parameters() if name.startswith("obs_")
    ]

    assert observation_names
    assert all(name.startswith("obs_branch.obs_") for name in observation_names)
    torch.testing.assert_close(
        torch.tanh(model.obs_branch.obs_raw_alpha), torch.full((1,), 0.001)
    )
    torch.testing.assert_close(
        torch.tanh(model.obs_branch.obs_raw_beta), torch.zeros(1)
    )


def test_wrapper_keeps_the_complete_backbone_frozen_and_in_eval_mode() -> None:
    native = _Native()
    model = ObservationReScene(native, observation_feature_dim=2, metadata_dim=15)

    model.train()

    assert not native.backbone.training
    assert all(
        not parameter.requires_grad for parameter in native.backbone.parameters()
    )
    assert native.cross_attention.training
    assert model.obs_branch.training


def _mode_output(mode: str) -> tuple[_Native, ObservationReScene, dict[str, object]]:
    torch.manual_seed(19)
    native = _Native()
    model = ObservationReScene(native, observation_feature_dim=2, metadata_dim=15)
    model.obs_branch.obs_cross_attention[0].register_forward_hook(
        lambda *_args: native.events.append("obs")
    )
    with torch.no_grad():
        model.obs_branch.obs_raw_alpha.fill_(torch.atanh(torch.tensor(0.5)))
        model.obs_branch.obs_raw_beta.fill_(torch.atanh(torch.tensor(0.5)))
    output = model(
        _Input(), observations=_bank(), observation_mode=mode, return_trace=True
    )
    return native, model, output


def test_late_mode_fuses_regions_only_into_the_final_mask() -> None:
    native, _model, output = _mode_output("late")

    assert native.events == ["cross", "self", "ffn"]
    torch.testing.assert_close(
        native.attn_seen[0][0], output["aux_outputs"][0]["pred_masks"][0]
    )
    assert not torch.equal(
        output["pred_masks"][0], output["aux_outputs"][0]["pred_masks"][0]
    )
    assert output["observation_mode"] == "late"


def test_attention_mode_updates_queries_but_never_feeds_regions_into_masks() -> None:
    native, _model, output = _mode_output("attention")

    assert native.events == ["cross", "obs", "self", "ffn"]
    torch.testing.assert_close(
        native.attn_seen[0][0], output["aux_outputs"][0]["pred_masks"][0]
    )
    assert output["observation_stats"]["feedback_layer_count"] == 0
    assert output["observation_mode"] == "attention"


def test_fuse_mode_injects_surface_features_without_observation_attention() -> None:
    torch.manual_seed(23)
    native = _Native()
    model = ObservationReScene(native, observation_feature_dim=2, metadata_dim=15)
    with torch.no_grad():
        model.obs_branch.obs_raw_fuse.fill_(torch.atanh(torch.tensor(0.5)))
    fused = model(_Input(), observations=_bank(), observation_mode="fuse")

    torch.manual_seed(23)
    comparison_native = _Native()
    comparison = ObservationReScene(
        comparison_native, observation_feature_dim=2, metadata_dim=15
    )
    comparison.load_state_dict(model.state_dict())
    attention = comparison(_Input(), observations=_bank(), observation_mode="attention")

    assert native.events == ["cross", "self", "ffn"]
    assert fused["observation_stats"]["feature_fusion_count"] == 1
    assert not torch.equal(fused["pred_masks"][0], attention["pred_masks"][0])


def test_no_feedback_matches_attention_forward_and_full_reports_feedback() -> None:
    native_attention, model, attention = _mode_output("attention")
    native_attention.events.clear()
    native_attention.attn_seen.clear()
    no_feedback = model(_Input(), observations=_bank(), observation_mode="no_feedback")
    torch.testing.assert_close(no_feedback["pred_masks"][0], attention["pred_masks"][0])
    assert no_feedback["observation_stats"]["feedback_layer_count"] == 0

    _native_full, _full_model, full = _mode_output("full")
    assert full["observation_stats"]["feedback_layer_count"] == 2


def test_unknown_observation_mode_is_rejected() -> None:
    model = ObservationReScene(_Native(), observation_feature_dim=2, metadata_dim=15)

    with pytest.raises(ValueError, match="observation_mode"):
        model(_Input(), observations=_bank(), observation_mode="unknown")


def test_cached_base_tuned_forward_has_no_observation_input_or_outputs() -> None:
    native = _Native()
    model = ObservationReScene(native, observation_feature_dim=2, metadata_dim=15)
    pcd, auxiliary, coordinates = native.backbone(_Input())

    output = model.forward_from_backbone(
        FrozenReSceneFeatures(pcd, auxiliary, coordinates),
        observations=None,
        observation_mode="base_tuned",
    )

    assert native.events == ["cross", "self", "ffn"]
    assert "pred_region_logits" not in output
    assert "region_aux_outputs" not in output
    assert output["observation_mode"] == "base_tuned"
    assert output["observation_stats"] == {
        "region_count": 0,
        "edge_count": 0,
        "supported_model_count": 0,
        "feature_fusion_count": 0,
        "attention_layer_count": 0,
        "feedback_layer_count": 0,
    }
