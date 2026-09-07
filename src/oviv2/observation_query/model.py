"""Observation-query intervention inside the native ReScene decoder loop.

The decoder-loop structure is adapted from ReScene ``models/rescene.py`` at
commit fb2fe42eb8f1e926567c48eea9acb874e608ee10 under the MIT License,
Copyright (c) 2026. Native modules remain owned by the wrapped model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import nn

from src.oviv2.observation_query.contracts import ObservationBank


class ObservationModelError(ValueError):
    """Raised when observation evidence cannot align to a decoder forward."""


def _bounded_raw(initial: float, name: str) -> float:
    if (
        isinstance(initial, bool)
        or not isinstance(initial, Real)
        or not math.isfinite(float(initial))
        or abs(float(initial)) >= 1.0
    ):
        raise ObservationModelError(f"{name} must be finite in (-1, 1)")
    return math.atanh(float(initial))


@dataclass(frozen=True, slots=True)
class ObservationTensorBank:
    """Device-local sparse tensors preserving the ObservationBank index domains."""

    region_features: torch.Tensor
    region_metadata: torch.Tensor
    csr_indptr: torch.Tensor
    csr_model_indices: torch.Tensor
    csr_weights: torch.Tensor
    region_reliability: torch.Tensor
    model_visit_ids: torch.Tensor
    content_sha256: str

    @classmethod
    def from_observation_bank(
        cls,
        bank: ObservationBank,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ObservationTensorBank:
        if not isinstance(bank, ObservationBank):
            raise TypeError("bank must be ObservationBank")
        if not dtype.is_floating_point:
            raise ObservationModelError(
                "observation tensor dtype must be floating point"
            )
        target = torch.device(device)
        return cls(
            region_features=torch.as_tensor(
                np.array(bank.region_features, copy=True), device=target, dtype=dtype
            ),
            region_metadata=torch.as_tensor(
                np.array(bank.region_metadata, copy=True), device=target, dtype=dtype
            ),
            csr_indptr=torch.as_tensor(
                np.array(bank.csr_indptr, copy=True), device=target, dtype=torch.long
            ),
            csr_model_indices=torch.as_tensor(
                np.array(bank.csr_model_indices, copy=True),
                device=target,
                dtype=torch.long,
            ),
            csr_weights=torch.as_tensor(
                np.array(bank.csr_weights, copy=True), device=target, dtype=dtype
            ),
            region_reliability=torch.as_tensor(
                np.array(bank.region_reliability, copy=True), device=target, dtype=dtype
            ),
            model_visit_ids=torch.as_tensor(
                np.array(bank.model_visit_ids, copy=True),
                device=target,
                dtype=torch.int8,
            ),
            content_sha256=bank.content_sha256(),
        )

    @property
    def region_count(self) -> int:
        return int(self.region_features.shape[0])

    @property
    def model_count(self) -> int:
        return int(self.model_visit_ids.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.csr_model_indices.shape[0])


class ObservationQueryBranch(nn.Module):
    """Learn region ownership, sparse surface feedback, and query evidence."""

    def __init__(
        self,
        *,
        observation_feature_dim: int,
        metadata_dim: int,
        hidden_dim: int,
        num_queries: int,
        num_heads: int,
        logical_layer_count: int,
        alpha_initial: float = 0.001,
        beta_initial: float = 0.0,
        fuse_initial: float = 0.001,
        correction_clip: float = 5.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        integer_values = {
            "observation_feature_dim": observation_feature_dim,
            "metadata_dim": metadata_dim,
            "hidden_dim": hidden_dim,
            "num_queries": num_queries,
            "num_heads": num_heads,
            "logical_layer_count": logical_layer_count,
        }
        if any(
            type(value) is not int or value <= 0 for value in integer_values.values()
        ):
            raise ObservationModelError(
                "observation dimensions must be positive integers"
            )
        if hidden_dim % num_heads != 0:
            raise ObservationModelError("hidden_dim must be divisible by num_heads")
        if (
            isinstance(correction_clip, bool)
            or not isinstance(correction_clip, Real)
            or not math.isfinite(float(correction_clip))
            or float(correction_clip) <= 0.0
        ):
            raise ObservationModelError("correction_clip must be finite and positive")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, Real)
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
        ):
            raise ObservationModelError("epsilon must be finite and positive")
        self.num_queries = num_queries
        self.logical_layer_count = logical_layer_count
        self.correction_clip = float(correction_clip)
        self.epsilon = float(epsilon)

        self.obs_feature_projection = nn.Linear(observation_feature_dim, hidden_dim)
        self.obs_metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_region_norm = nn.LayerNorm(hidden_dim)
        self.obs_query_norm = nn.LayerNorm(hidden_dim)
        self.obs_attention_token_norm = nn.LayerNorm(hidden_dim)
        self.obs_query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.obs_region_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.obs_fuse_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.obs_dustbin_head = nn.Linear(hidden_dim, 1)
        self.obs_cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(logical_layer_count)
            ]
        )
        self.obs_raw_alpha = nn.Parameter(
            torch.full(
                (logical_layer_count,), _bounded_raw(alpha_initial, "alpha_initial")
            )
        )
        self.obs_raw_beta = nn.Parameter(
            torch.full(
                (logical_layer_count,), _bounded_raw(beta_initial, "beta_initial")
            )
        )
        self.obs_raw_fuse = nn.Parameter(
            torch.tensor(_bounded_raw(fuse_initial, "fuse_initial"))
        )

    def _layer(self, layer_index: int) -> int:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, Integral)
            or not 0 <= int(layer_index) < self.logical_layer_count
        ):
            raise ObservationModelError("logical observation layer index is invalid")
        return int(layer_index)

    def encode_regions(self, bank: ObservationTensorBank) -> torch.Tensor:
        if not isinstance(bank, ObservationTensorBank):
            raise TypeError("bank must be ObservationTensorBank")
        encoded = self.obs_feature_projection(bank.region_features)
        encoded = encoded + self.obs_metadata_projection(bank.region_metadata)
        return self.obs_region_norm(encoded)

    def predict_region_logits(
        self, queries: torch.Tensor, region_tokens: torch.Tensor
    ) -> torch.Tensor:
        if queries.ndim != 3 or queries.shape[0] != 1:
            raise ObservationModelError(
                "observation prediction supports one pair per batch"
            )
        if queries.shape[1] != self.num_queries or region_tokens.ndim != 2:
            raise ObservationModelError("query or region token shape is invalid")
        projected_queries = self.obs_query_projection(self.obs_query_norm(queries[0]))
        projected_regions = self.obs_region_projection(region_tokens)
        scores = projected_regions @ projected_queries.transpose(0, 1)
        scores = scores / math.sqrt(float(projected_queries.shape[-1]))
        dustbin = self.obs_dustbin_head(region_tokens)
        return torch.cat((scores, dustbin), dim=1)

    def surface_logit_correction(
        self,
        region_logits: torch.Tensor,
        bank: ObservationTensorBank,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._layer(layer_index)
        if region_logits.shape != (bank.region_count, self.num_queries + 1):
            raise ObservationModelError("region logits do not align to R and Q")
        if bank.edge_count == 0:
            return (
                region_logits.new_zeros((bank.model_count, self.num_queries)),
                torch.zeros(
                    bank.model_count, dtype=torch.bool, device=region_logits.device
                ),
            )
        counts = bank.csr_indptr[1:] - bank.csr_indptr[:-1]
        edge_regions = torch.repeat_interleave(
            torch.arange(bank.region_count, device=region_logits.device), counts
        )
        if edge_regions.numel() != bank.edge_count:
            raise ObservationModelError("observation CSR edge count is inconsistent")
        ownership = torch.softmax(region_logits, dim=1)[:, : self.num_queries]
        edge_weights = bank.csr_weights * bank.region_reliability[edge_regions]
        numerator = region_logits.new_zeros((bank.model_count, self.num_queries))
        numerator.index_add_(
            0,
            bank.csr_model_indices,
            ownership[edge_regions] * edge_weights[:, None],
        )
        denominator = region_logits.new_zeros(bank.model_count)
        denominator.index_add_(0, bank.csr_model_indices, edge_weights)
        supported = denominator > 0.0
        correction = region_logits.new_zeros((bank.model_count, self.num_queries))
        if supported.any():
            average = numerator[supported] / denominator[supported, None]
            correction[supported] = torch.clamp(
                torch.log(average + self.epsilon)
                + math.log(float(self.num_queries + 1)),
                min=-self.correction_clip,
                max=self.correction_clip,
            )
        return correction, supported

    def correct_mask_logits(
        self,
        base_logits: torch.Tensor,
        region_logits: torch.Tensor,
        bank: ObservationTensorBank,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self._layer(layer_index)
        if base_logits.shape != (1, bank.model_count, self.num_queries):
            raise ObservationModelError(
                "native mask logits do not align to the M domain"
            )
        correction, supported = self.surface_logit_correction(
            region_logits, bank, layer_index=layer
        )
        beta = torch.tanh(self.obs_raw_beta[layer])
        return base_logits + beta * correction[None, :, :], supported

    def attend_observations(
        self,
        crossed_queries: torch.Tensor,
        region_tokens: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        layer = self._layer(layer_index)
        if region_tokens.shape[0] == 0:
            return crossed_queries
        if crossed_queries.ndim != 3 or crossed_queries.shape[0] != 1:
            raise ObservationModelError(
                "observation attention supports one pair per batch"
            )
        query = self.obs_query_norm(crossed_queries)
        memory = self.obs_attention_token_norm(region_tokens)[None, :, :]
        update = self.obs_cross_attention[layer](
            query, memory, memory, need_weights=False
        )[0]
        alpha = torch.tanh(self.obs_raw_alpha[layer])
        return crossed_queries + alpha * update

    def fuse_regions_into_model(
        self, region_tokens: torch.Tensor, bank: ObservationTensorBank
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project each region once through the CSR relation into the M domain."""

        if region_tokens.shape != (
            bank.region_count,
            self.obs_query_projection.in_features,
        ):
            raise ObservationModelError(
                "region tokens do not align to the observation bank"
            )
        fused = region_tokens.new_zeros((bank.model_count, region_tokens.shape[1]))
        denominator = region_tokens.new_zeros(bank.model_count)
        if bank.edge_count == 0:
            return fused, denominator > 0.0
        counts = bank.csr_indptr[1:] - bank.csr_indptr[:-1]
        edge_regions = torch.repeat_interleave(
            torch.arange(bank.region_count, device=region_tokens.device), counts
        )
        edge_weights = bank.csr_weights * bank.region_reliability[edge_regions]
        fused.index_add_(
            0,
            bank.csr_model_indices,
            self.obs_fuse_projection(region_tokens[edge_regions])
            * edge_weights[:, None],
        )
        denominator.index_add_(0, bank.csr_model_indices, edge_weights)
        supported = denominator > 0.0
        fused[supported] = fused[supported] / denominator[supported, None]
        return torch.tanh(self.obs_raw_fuse) * fused, supported


@dataclass(frozen=True, slots=True)
class FrozenReSceneFeatures:
    """In-memory result of the frozen backbone, before trainable decoder heads."""

    pcd_features: Any
    auxiliary_features: Any
    coordinates: Any


class ObservationReScene(nn.Module):
    """Wrap ReScene and intervene inside, rather than after, its decoder."""

    def __init__(
        self,
        native_model: nn.Module,
        *,
        observation_feature_dim: int,
        metadata_dim: int,
        alpha_initial: float = 0.001,
        beta_initial: float = 0.0,
        fuse_initial: float = 0.001,
        correction_clip: float = 5.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(native_model, nn.Module):
            raise TypeError("native_model must be a torch module")
        required = (
            "mask_dim",
            "num_queries",
            "num_heads",
            "num_decoders",
            "hlevels",
        )
        if any(not hasattr(native_model, name) for name in required):
            raise ObservationModelError(
                "native model lacks its resolved decoder contract"
            )
        self.native = native_model
        logical_layers = int(native_model.num_decoders) * len(native_model.hlevels)
        self.obs_branch = ObservationQueryBranch(
            observation_feature_dim=observation_feature_dim,
            metadata_dim=metadata_dim,
            hidden_dim=int(native_model.mask_dim),
            num_queries=int(native_model.num_queries),
            num_heads=int(native_model.num_heads),
            logical_layer_count=logical_layers,
            alpha_initial=alpha_initial,
            beta_initial=beta_initial,
            fuse_initial=fuse_initial,
            correction_clip=correction_clip,
            epsilon=epsilon,
        )
        self.native.backbone.requires_grad_(False)
        self.native.backbone.eval()

    def train(self, mode: bool = True) -> ObservationReScene:
        super().train(mode)
        self.native.backbone.eval()
        return self

    def forward(
        self,
        x: Any,
        point2segment: Any = None,
        raw_coordinates: Any = None,
        is_eval: bool = False,
        *,
        observations: ObservationBank | None = None,
        observation_mode: str = "full",
        return_trace: bool = False,
    ) -> dict[str, Any]:
        valid_modes = {
            "disabled",
            "base_tuned",
            "late",
            "fuse",
            "attention",
            "full",
            "no_feedback",
            "no_consistency",
        }
        if observation_mode not in valid_modes:
            raise ObservationModelError(
                f"observation_mode must be one of {sorted(valid_modes)}"
            )
        if observation_mode == "disabled" or (
            observation_mode != "base_tuned"
            and (observations is None or len(observations.region_keys) == 0)
        ):
            return self.native(
                x,
                point2segment=point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
            )
        if observation_mode != "base_tuned" and not isinstance(
            observations, ObservationBank
        ):
            raise TypeError("observations must be ObservationBank")
        if bool(getattr(self.native, "save_segment_info", False)):
            raise ObservationModelError(
                "observation inference requires native save_segment_info=false"
            )
        if not bool(getattr(self.native, "train_on_segments", False)):
            point2segment = None
        x.raw_coordinates = raw_coordinates
        x.point2segment = point2segment
        with torch.no_grad():
            pcd_features, auxiliary, coordinates = self.native.backbone(x)
        return self.forward_from_backbone(
            FrozenReSceneFeatures(pcd_features, auxiliary, coordinates),
            point2segment=point2segment,
            is_eval=is_eval,
            observations=observations,
            observation_mode=observation_mode,
            return_trace=return_trace,
        )

    def forward_from_backbone(
        self,
        frozen_features: FrozenReSceneFeatures,
        *,
        point2segment: Any = None,
        is_eval: bool = False,
        observations: ObservationBank | None,
        observation_mode: str = "full",
        return_trace: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(frozen_features, FrozenReSceneFeatures):
            raise TypeError("frozen_features must be FrozenReSceneFeatures")
        enabled_modes = {
            "late",
            "fuse",
            "attention",
            "full",
            "no_feedback",
            "no_consistency",
        }
        if observation_mode == "base_tuned":
            if observations is not None:
                raise ObservationModelError(
                    "cached base_tuned forward cannot consume observations"
                )
        elif observation_mode not in enabled_modes or not isinstance(
            observations, ObservationBank
        ):
            raise ObservationModelError(
                "cached decoder forward requires an enabled mode and observation bank"
            )
        if isinstance(observations, ObservationBank) and not observations.region_keys:
            raise ObservationModelError(
                "cached decoder forward requires observation regions"
            )
        return self._forward_enabled_from_backbone(
            frozen_features,
            point2segment=point2segment,
            is_eval=is_eval,
            observations=observations,
            observation_mode=observation_mode,
            return_trace=return_trace,
        )

    def _forward_enabled_from_backbone(
        self,
        frozen: FrozenReSceneFeatures,
        *,
        point2segment: Any,
        is_eval: bool,
        observations: ObservationBank | None,
        observation_mode: str,
        return_trace: bool,
    ) -> dict[str, Any]:
        pcd_features = frozen.pcd_features
        auxiliary = frozen.auxiliary_features
        coordinates = frozen.coordinates
        positional = self.native.get_pos_encs(coordinates)
        aggregate_features, _ = self.native.aggregate_features(
            pcd_features, point2segment
        )
        batched_features, batch_map = self.native.sample_and_batch_features(
            aggregate_features
        )
        queries, query_pos, sampled_coords = self.native.initialize_queries(
            pcd_features=pcd_features,
            coords=coordinates,
        )
        if batched_features.shape[0] != 1 or queries.shape[0] != 1:
            raise ObservationModelError(
                "first implementation requires one pair per batch"
            )
        uses_regions = observation_mode != "base_tuned"
        tensor_bank: ObservationTensorBank | None = None
        region_tokens: torch.Tensor | None = None
        if uses_regions:
            if not isinstance(observations, ObservationBank):
                raise ObservationModelError("enabled observation mode lacks its bank")
            tensor_bank = ObservationTensorBank.from_observation_bank(
                observations,
                device=queries.device,
                dtype=batched_features.dtype,
            )
            if tensor_bank.model_count != int((~batch_map[0]).sum()):
                raise ObservationModelError(
                    "observation M domain differs from native mask domain"
                )
            region_tokens = self.obs_branch.encode_regions(tensor_bank)
        use_feature_fusion = observation_mode == "fuse"
        use_attention = observation_mode in {
            "attention",
            "full",
            "no_feedback",
            "no_consistency",
        }
        use_layer_feedback = observation_mode in {"full", "no_consistency"}
        use_final_feedback = use_layer_feedback or observation_mode == "late"
        if use_feature_fusion:
            assert region_tokens is not None and tensor_bank is not None
            projected, supported = self.obs_branch.fuse_regions_into_model(
                region_tokens, tensor_bank
            )
            batched_features = batched_features.clone()
            batched_features[0, ~batch_map[0]] = (
                batched_features[0, ~batch_map[0]] + projected
            )

        predictions_class: list[torch.Tensor] = []
        predictions_changes: list[torch.Tensor | None] = []
        predictions_mask: list[list[torch.Tensor]] = []
        predictions_region: list[torch.Tensor] = []
        supported = torch.zeros(
            int((~batch_map[0]).sum()), dtype=torch.bool, device=queries.device
        )
        trace_layers: list[int] = []
        trace_region: list[torch.Tensor] = []
        trace_base: list[torch.Tensor] = []
        trace_corrected: list[torch.Tensor] = []
        logical_layer = 0
        for native_decoder_counter in range(int(self.native.num_decoders)):
            decoder_counter = (
                0 if bool(self.native.shared_decoder) else native_decoder_counter
            )
            for hierarchy_index, hierarchy_level in enumerate(self.native.hlevels):
                output_class, output_change, base_logits = self.native.mask_module(
                    queries, batched_features
                )
                region_logits = (
                    self.obs_branch.predict_region_logits(queries, region_tokens)
                    if region_tokens is not None
                    else None
                )
                if use_layer_feedback:
                    assert region_logits is not None and tensor_bank is not None
                    corrected_logits, supported = self.obs_branch.correct_mask_logits(
                        base_logits,
                        region_logits,
                        tensor_bank,
                        layer_index=logical_layer,
                    )
                else:
                    corrected_logits = base_logits
                output_mask = self.native.unstack_batched(corrected_logits, batch_map)
                attention_masks = self.native.attn_mask(
                    corrected_logits,
                    batch_map,
                    sparse_coords=pcd_features,
                    num_pooling_steps=len(auxiliary) - hierarchy_level - 1,
                    point2segment=point2segment,
                )
                batched_aux, batched_attn, batched_pos, padding_mask = (
                    self.native.sample_and_batch_features(
                        auxiliary[hierarchy_level].decomposed_features,
                        self.native.sample_sizes[hierarchy_level],
                        is_eval,
                        extra=[attention_masks, positional[hierarchy_level][0]],
                    )
                )
                all_masked = batched_attn.sum(1) == batched_attn.shape[1]
                batched_attn.permute((0, 2, 1))[all_masked] = False
                attention_mask = batched_attn.repeat_interleave(
                    int(self.native.num_heads), dim=0
                ).permute((0, 2, 1))
                source = self.native.lin_squeeze[decoder_counter][hierarchy_index](
                    batched_aux
                )
                if bool(self.native.use_level_embed):
                    source = source + self.native.level_embed.weight[hierarchy_index]
                crossed = self.native.cross_attention[decoder_counter][hierarchy_index](
                    queries,
                    source,
                    memory_mask=attention_mask,
                    memory_key_padding_mask=padding_mask,
                    pos=batched_pos,
                    query_pos=query_pos,
                )
                observed = (
                    self.obs_branch.attend_observations(
                        crossed, region_tokens, layer_index=logical_layer
                    )
                    if use_attention and region_tokens is not None
                    else crossed
                )
                self_attended = self.native.self_attention[decoder_counter][
                    hierarchy_index
                ](
                    observed,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )
                queries = self.native.ffn_attention[decoder_counter][hierarchy_index](
                    self_attended
                )
                predictions_class.append(output_class)
                predictions_changes.append(output_change)
                predictions_mask.append(output_mask)
                if region_logits is not None:
                    predictions_region.append(region_logits)
                if return_trace:
                    trace_layers.append(logical_layer)
                    if region_logits is not None:
                        trace_region.append(region_logits[:4, :5])
                    trace_base.append(base_logits[:, :4, :5])
                    trace_corrected.append(corrected_logits[:, :4, :5])
                logical_layer += 1

        output_class, output_change, base_logits = self.native.mask_module(
            queries, batched_features
        )
        region_logits = (
            self.obs_branch.predict_region_logits(queries, region_tokens)
            if region_tokens is not None
            else None
        )
        if use_final_feedback:
            assert region_logits is not None and tensor_bank is not None
            corrected_logits, supported = self.obs_branch.correct_mask_logits(
                base_logits,
                region_logits,
                tensor_bank,
                layer_index=logical_layer - 1,
            )
        else:
            corrected_logits = base_logits
        output_mask = self.native.unstack_batched(corrected_logits, batch_map)
        predictions_class.append(output_class)
        predictions_changes.append(output_change)
        predictions_mask.append(output_mask)
        if region_logits is not None:
            predictions_region.append(region_logits)
        output: dict[str, Any] = {
            "pred_logits": predictions_class[-1],
            "pred_changes": predictions_changes[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self.native._set_aux_loss(
                predictions_class, predictions_mask, predictions_changes
            ),
            "sampled_coords": sampled_coords.detach().cpu().numpy()
            if sampled_coords is not None
            else None,
            "backbone_features": pcd_features,
            "segment_features": [aggregate_features],
            "observation_stats": {
                "region_count": 0 if tensor_bank is None else tensor_bank.region_count,
                "edge_count": 0 if tensor_bank is None else tensor_bank.edge_count,
                "supported_model_count": int(supported.sum().detach().cpu()),
                "feature_fusion_count": int(use_feature_fusion),
                "attention_layer_count": logical_layer if use_attention else 0,
                "feedback_layer_count": (
                    logical_layer + 1
                    if use_layer_feedback
                    else int(observation_mode == "late")
                ),
            },
            "observation_mode": observation_mode,
        }
        if predictions_region:
            output["pred_region_logits"] = predictions_region[-1]
            output["region_aux_outputs"] = [
                {"pred_region_logits": value} for value in predictions_region[:-1]
            ]
        if return_trace:
            output["trace"] = {
                "layer_ids": tuple(trace_layers),
                "region_logits": tuple(trace_region),
                "base_mask_logits": tuple(trace_base),
                "corrected_mask_logits": tuple(trace_corrected),
            }
        return output


__all__ = [
    "FrozenReSceneFeatures",
    "ObservationModelError",
    "ObservationQueryBranch",
    "ObservationReScene",
    "ObservationTensorBank",
]
