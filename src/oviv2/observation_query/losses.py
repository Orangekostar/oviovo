"""Masked pair-level matching and joint surface/region supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.training.ovi_observation_data import ObservationTrainingSample


class ObservationLossError(ValueError):
    """Raised when predictions and criterion-only targets do not align."""


@dataclass(frozen=True, slots=True)
class PairAssignment:
    query_indices: torch.Tensor
    target_indices: torch.Tensor
    unmatched_target_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObservationLossResult:
    components: dict[str, torch.Tensor]
    total: torch.Tensor
    final_assignment: PairAssignment
    aux_assignments: tuple[PairAssignment, ...]
    applied_weights: dict[str, float]
    diagnostics: dict[str, int | float]


def _finite_weight(value: object, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ObservationLossError(f"{name} must be numeric")
    result = float(value)
    minimum = 0.0 if allow_zero else float(np.nextafter(0.0, 1.0))
    if not math.isfinite(result) or result < minimum:
        raise ObservationLossError(f"{name} is outside its legal range")
    return result


def _valid_for_target(label_valid: torch.Tensor, target_index: int) -> torch.Tensor:
    if label_valid.ndim == 1:
        return label_valid
    if label_valid.ndim == 2:
        return label_valid[target_index]
    raise ObservationLossError("label_valid must have shape M or K x M")


class MaskedHungarianMatcher(nn.Module):
    """Apply class/BCE/Dice costs only on each target's trusted M domain."""

    def __init__(
        self,
        *,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = _finite_weight(cost_class, "cost_class")
        self.cost_mask = _finite_weight(cost_mask, "cost_mask")
        self.cost_dice = _finite_weight(cost_dice, "cost_dice")
        if self.cost_class == self.cost_mask == self.cost_dice == 0.0:
            raise ObservationLossError("at least one matching cost must be positive")

    @torch.no_grad()
    def forward(
        self,
        predicted_masks_mq: torch.Tensor,
        predicted_classes_qc: torch.Tensor,
        target_masks_km: torch.Tensor,
        label_valid: torch.Tensor,
        class_targets_k: torch.Tensor,
        class_valid_k: torch.Tensor,
    ) -> PairAssignment:
        if predicted_masks_mq.ndim != 2 or predicted_classes_qc.ndim != 2:
            raise ObservationLossError(
                "matcher predictions must have shape MxQ and QxC"
            )
        model_count, query_count = predicted_masks_mq.shape
        target_count = target_masks_km.shape[0]
        if (
            target_masks_km.shape != (target_count, model_count)
            or predicted_classes_qc.shape[0] != query_count
            or class_targets_k.shape != (target_count,)
            or class_valid_k.shape != (target_count,)
            or label_valid.shape not in {(model_count,), (target_count, model_count)}
        ):
            raise ObservationLossError("matcher K, M, or Q domains are inconsistent")
        tensors = (
            predicted_masks_mq,
            predicted_classes_qc,
            target_masks_km,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ObservationLossError("matcher inputs must be finite")
        device = predicted_masks_mq.device
        if target_count == 0 or query_count == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            unmatched = torch.arange(target_count, dtype=torch.long, device=device)
            return PairAssignment(empty, empty, unmatched)
        probabilities = predicted_classes_qc.float().softmax(dim=1)
        costs = predicted_masks_mq.new_zeros((query_count, target_count)).float()
        query_masks = predicted_masks_mq.transpose(0, 1).float()
        targets = target_masks_km.float()
        for target_index in range(target_count):
            valid = _valid_for_target(label_valid, target_index).to(
                device=device, dtype=torch.bool
            )
            if valid.shape != (model_count,) or not valid.any():
                raise ObservationLossError(
                    "every GT target needs a non-empty trusted M domain"
                )
            prediction = query_masks[:, valid]
            target = targets[target_index, valid][None, :].expand(query_count, -1)
            bce = F.binary_cross_entropy_with_logits(
                prediction, target, reduction="none"
            ).mean(dim=1)
            probability = prediction.sigmoid()
            numerator = 2.0 * (probability * target).sum(dim=1)
            denominator = probability.sum(dim=1) + target.sum(dim=1)
            dice = 1.0 - (numerator + 1.0) / (denominator + 1.0)
            costs[:, target_index] = self.cost_mask * bce + self.cost_dice * dice
            if bool(class_valid_k[target_index]):
                class_id = int(class_targets_k[target_index])
                if not 0 <= class_id < predicted_classes_qc.shape[1] - 1:
                    raise ObservationLossError(
                        "valid class target is outside foreground classes"
                    )
                costs[:, target_index] += self.cost_class * -probabilities[:, class_id]
        if not torch.isfinite(costs).all():
            raise ObservationLossError("matching cost is non-finite")
        query_indices, target_indices = linear_sum_assignment(costs.cpu().numpy())
        order = np.argsort(target_indices, kind="stable")
        query = torch.as_tensor(query_indices[order], dtype=torch.long, device=device)
        target = torch.as_tensor(target_indices[order], dtype=torch.long, device=device)
        matched_targets = {int(value) for value in target_indices}
        unmatched = torch.as_tensor(
            [value for value in range(target_count) if value not in matched_targets],
            dtype=torch.long,
            device=device,
        )
        return PairAssignment(query, target, unmatched)


class ObservationSetCriterion(nn.Module):
    """One assignment per layer shared by native masks/classes and region ownership."""

    def __init__(
        self,
        *,
        foreground_class_count: int,
        num_queries: int,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 2.0,
        no_object_coefficient: float = 0.2,
        native_weight: float = 1.0,
        region_weight: float = 0.2,
        consistency_weight: float = 0.1,
        consistency_ramp_updates: int = 200,
        epsilon: float = 1e-6,
        require_region_supervision: bool = True,
    ) -> None:
        super().__init__()
        if type(foreground_class_count) is not int or foreground_class_count <= 0:
            raise ObservationLossError("foreground_class_count must be positive")
        if type(num_queries) is not int or num_queries <= 0:
            raise ObservationLossError("num_queries must be positive")
        if type(consistency_ramp_updates) is not int or consistency_ramp_updates < 0:
            raise ObservationLossError("consistency_ramp_updates must be nonnegative")
        self.foreground_class_count = foreground_class_count
        self.num_queries = num_queries
        self.cost_class = _finite_weight(cost_class, "cost_class")
        self.cost_mask = _finite_weight(cost_mask, "cost_mask")
        self.cost_dice = _finite_weight(cost_dice, "cost_dice")
        self.no_object_coefficient = _finite_weight(
            no_object_coefficient, "no_object_coefficient", allow_zero=False
        )
        self.native_weight = _finite_weight(native_weight, "native_weight")
        self.region_weight = _finite_weight(region_weight, "region_weight")
        self.consistency_weight = _finite_weight(
            consistency_weight, "consistency_weight"
        )
        self.consistency_ramp_updates = consistency_ramp_updates
        self.epsilon = _finite_weight(epsilon, "epsilon", allow_zero=False)
        if not isinstance(require_region_supervision, bool):
            raise ObservationLossError("require_region_supervision must be boolean")
        if not require_region_supervision and (
            self.region_weight != 0.0 or self.consistency_weight != 0.0
        ):
            raise ObservationLossError(
                "native-only criterion requires zero region and consistency weights"
            )
        self.require_region_supervision = require_region_supervision
        self.matcher = MaskedHungarianMatcher(
            cost_class=self.cost_class,
            cost_mask=self.cost_mask,
            cost_dice=self.cost_dice,
        )
        self.obs_surface_null_head = nn.Linear(num_queries, 1)

    def _prediction_tensors(
        self, output: dict[str, Any], *, require_region: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        masks = output.get("pred_masks")
        classes = output.get("pred_logits")
        regions = output.get("pred_region_logits")
        if (
            not isinstance(masks, (list, tuple))
            or len(masks) != 1
            or not isinstance(masks[0], torch.Tensor)
            or not isinstance(classes, torch.Tensor)
            or classes.ndim != 3
            or classes.shape[0] != 1
        ):
            raise ObservationLossError(
                "prediction output lacks one-pair mask/class tensors"
            )
        if require_region and not isinstance(regions, torch.Tensor):
            raise ObservationLossError("prediction output lacks region logits")
        region_tensor = regions if isinstance(regions, torch.Tensor) else None
        mask_tensor = masks[0]
        class_tensor = classes[0]
        if (
            mask_tensor.ndim != 2
            or mask_tensor.shape[1] != self.num_queries
            or class_tensor.shape != (self.num_queries, self.foreground_class_count + 1)
            or (
                region_tensor is not None
                and region_tensor.shape[1] != self.num_queries + 1
            )
            or not torch.isfinite(mask_tensor).all()
            or not torch.isfinite(class_tensor).all()
            or (region_tensor is not None and not torch.isfinite(region_tensor).all())
        ):
            raise ObservationLossError(
                "prediction tensors have invalid shapes or values"
            )
        return mask_tensor, class_tensor, region_tensor

    def _native_losses(
        self,
        masks: torch.Tensor,
        classes: torch.Tensor,
        sample: ObservationTrainingSample,
        assignment: PairAssignment,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_valid = sample.class_valid.to(device=classes.device, dtype=torch.bool)
        class_targets = sample.class_targets.to(device=classes.device, dtype=torch.long)
        target_masks = sample.instance_masks.to(
            device=masks.device, dtype=torch.float32
        )
        target_classes = torch.full(
            (self.num_queries,),
            self.foreground_class_count,
            dtype=torch.long,
            device=classes.device,
        )
        for query_index, target_index in zip(
            assignment.query_indices.tolist(), assignment.target_indices.tolist()
        ):
            if bool(class_valid[target_index]):
                target_classes[query_index] = class_targets[target_index]
            else:
                target_classes[query_index] = -100
        if torch.all(target_classes == -100):
            class_loss = classes.sum() * 0.0
        else:
            class_weights = torch.ones(
                self.foreground_class_count + 1,
                dtype=classes.dtype,
                device=classes.device,
            )
            class_weights[-1] = self.no_object_coefficient
            class_loss = F.cross_entropy(
                classes,
                target_classes,
                weight=class_weights,
                ignore_index=-100,
            )
        bce_losses: list[torch.Tensor] = []
        dice_losses: list[torch.Tensor] = []
        for query_index, target_index in zip(
            assignment.query_indices.tolist(), assignment.target_indices.tolist()
        ):
            valid = _valid_for_target(sample.label_valid, target_index).to(
                device=masks.device, dtype=torch.bool
            )
            prediction = masks[valid, query_index].float()
            target = target_masks[target_index, valid]
            bce_losses.append(
                F.binary_cross_entropy_with_logits(prediction, target, reduction="mean")
            )
            probability = prediction.sigmoid()
            dice_losses.append(
                1.0
                - (2.0 * (probability * target).sum() + 1.0)
                / (probability.sum() + target.sum() + 1.0)
            )
        zero = masks.sum() * 0.0
        mask_loss = torch.stack(bce_losses).mean() if bce_losses else zero
        dice_loss = torch.stack(dice_losses).mean() if dice_losses else zero
        return class_loss, mask_loss, dice_loss

    def _region_loss(
        self,
        region_logits: torch.Tensor,
        sample: ObservationTrainingSample,
        assignment: PairAssignment,
    ) -> tuple[torch.Tensor, int, float]:
        target_mass = sample.region_instance_mass.to(
            device=region_logits.device, dtype=region_logits.dtype
        )
        if region_logits.shape[0] != target_mass.shape[0]:
            raise ObservationLossError(
                "region predictions and targets have different R"
            )
        converted = torch.zeros_like(region_logits)
        matched_targets: set[int] = set()
        for query_index, target_index in zip(
            assignment.query_indices.tolist(), assignment.target_indices.tolist()
        ):
            converted[:, query_index] = target_mass[:, target_index]
            matched_targets.add(target_index)
        converted[:, -1] = target_mass[:, -1]
        unmatched = sorted(
            set(range(target_mass.shape[1] - 1)).difference(matched_targets)
        )
        excluded = (
            target_mass[:, unmatched].sum(dim=1)
            if unmatched
            else target_mass.new_zeros(target_mass.shape[0])
        )
        retained = converted.sum(dim=1)
        valid = sample.region_label_valid.to(region_logits.device) & (
            retained > self.epsilon
        )
        if valid.any():
            normalized = converted[valid] / retained[valid, None]
            per_region = -(normalized * F.log_softmax(region_logits[valid], dim=1)).sum(
                1
            )
            reliability = sample.observations.region_reliability
            weights = torch.as_tensor(
                np.array(reliability, copy=True),
                dtype=per_region.dtype,
                device=per_region.device,
            )[valid]
            if weights.sum() > 0.0:
                loss = (per_region * weights).sum() / weights.sum()
            else:
                loss = region_logits.sum() * 0.0
        else:
            loss = region_logits.sum() * 0.0
        originally_valid = sample.region_label_valid.to(region_logits.device)
        denominator = target_mass[originally_valid].sum()
        excluded_fraction = (
            float(
                excluded[originally_valid].sum().detach().cpu()
                / denominator.detach().cpu()
            )
            if denominator > 0.0
            else 0.0
        )
        return loss, int(valid.sum().detach().cpu()), excluded_fraction

    def _consistency_loss(
        self,
        masks: torch.Tensor,
        region_logits: torch.Tensor,
        sample: ObservationTrainingSample,
    ) -> torch.Tensor:
        observations = sample.observations
        device = masks.device
        region_count = len(observations.region_keys)
        if masks.shape[0] != len(observations.model_visit_ids):
            raise ObservationLossError("consistency masks differ from observation M")
        null_logits = self.obs_surface_null_head(masks)
        surface_distribution = torch.softmax(
            torch.cat((masks, null_logits), dim=1), dim=1
        )
        indptr = torch.as_tensor(
            np.array(observations.csr_indptr, copy=True),
            dtype=torch.long,
            device=device,
        )
        model_indices = torch.as_tensor(
            np.array(observations.csr_model_indices, copy=True),
            dtype=torch.long,
            device=device,
        )
        weights = torch.as_tensor(
            np.array(observations.csr_weights, copy=True),
            dtype=masks.dtype,
            device=device,
        )
        counts = indptr[1:] - indptr[:-1]
        edge_regions = torch.repeat_interleave(
            torch.arange(region_count, device=device), counts
        )
        projected = masks.new_zeros((region_count, self.num_queries + 1))
        denominators = masks.new_zeros(region_count)
        if len(model_indices):
            projected.index_add_(
                0,
                edge_regions,
                surface_distribution[model_indices] * weights[:, None],
            )
            denominators.index_add_(0, edge_regions, weights)
        valid = sample.region_label_valid.to(device) & (denominators > 0.0)
        if not valid.any():
            return masks.sum() * 0.0 + region_logits.sum() * 0.0
        surface_region = projected[valid] / denominators[valid, None]
        observation_region = torch.softmax(region_logits[valid], dim=1)
        log_surface = torch.log(surface_region.clamp_min(self.epsilon))
        log_observation = torch.log(observation_region.clamp_min(self.epsilon))
        forward = (
            observation_region.detach() * (log_observation.detach() - log_surface)
        ).sum(1)
        reverse = (
            surface_region.detach() * (log_surface.detach() - log_observation)
        ).sum(1)
        reliability = torch.as_tensor(
            np.array(observations.region_reliability, copy=True),
            dtype=masks.dtype,
            device=device,
        )[valid]
        symmetric = 0.5 * (forward + reverse)
        if reliability.sum() <= 0.0:
            return masks.sum() * 0.0 + region_logits.sum() * 0.0
        return (symmetric * reliability).sum() / reliability.sum()

    def forward(
        self,
        outputs: dict[str, Any],
        sample: ObservationTrainingSample,
        *,
        optimizer_update: int,
    ) -> ObservationLossResult:
        if not isinstance(sample, ObservationTrainingSample):
            raise TypeError("sample must be ObservationTrainingSample")
        if type(optimizer_update) is not int or optimizer_update < 0:
            raise ObservationLossError("optimizer_update must be nonnegative")
        final_masks, final_classes, final_regions = self._prediction_tensors(
            outputs, require_region=self.require_region_supervision
        )
        if final_masks.shape[0] != sample.instance_masks.shape[1]:
            raise ObservationLossError("prediction M differs from training sample")
        final_assignment = self.matcher(
            final_masks,
            final_classes,
            sample.instance_masks.to(final_masks.device),
            sample.label_valid.to(final_masks.device),
            sample.class_targets.to(final_masks.device),
            sample.class_valid.to(final_masks.device),
        )
        final_native = self._native_losses(
            final_masks, final_classes, sample, final_assignment
        )
        region_losses: list[torch.Tensor] = []
        consistency_losses: list[torch.Tensor] = []
        if final_regions is not None:
            final_region, supervised_regions, excluded_mass = self._region_loss(
                final_regions, sample, final_assignment
            )
            region_losses.append(final_region)
            consistency_losses.append(
                self._consistency_loss(final_masks, final_regions, sample)
            )
        else:
            supervised_regions = 0
            excluded_mass = 0.0
        components: dict[str, torch.Tensor] = {
            "loss_ce": final_native[0],
            "loss_mask": final_native[1],
            "loss_dice": final_native[2],
        }
        auxiliary_outputs = outputs.get("aux_outputs", [])
        auxiliary_regions = outputs.get("region_aux_outputs", [])
        if not isinstance(auxiliary_outputs, (list, tuple)) or not isinstance(
            auxiliary_regions, (list, tuple)
        ):
            raise ObservationLossError("native and region auxiliary layers must align")
        if self.require_region_supervision and len(auxiliary_outputs) != len(
            auxiliary_regions
        ):
            raise ObservationLossError("native and region auxiliary layers must align")
        if not self.require_region_supervision and auxiliary_regions:
            raise ObservationLossError(
                "native-only criterion received region auxiliaries"
            )
        aux_assignments: list[PairAssignment] = []
        for layer_index, native_output in enumerate(auxiliary_outputs):
            if not isinstance(native_output, dict):
                raise ObservationLossError("auxiliary output must be a mapping")
            merged = dict(native_output)
            if self.require_region_supervision:
                region_output = auxiliary_regions[layer_index]
                if not isinstance(region_output, dict):
                    raise ObservationLossError("auxiliary output must be a mapping")
                merged["pred_region_logits"] = region_output.get("pred_region_logits")
            masks, classes, regions = self._prediction_tensors(
                merged, require_region=self.require_region_supervision
            )
            assignment = self.matcher(
                masks,
                classes,
                sample.instance_masks.to(masks.device),
                sample.label_valid.to(masks.device),
                sample.class_targets.to(masks.device),
                sample.class_valid.to(masks.device),
            )
            aux_assignments.append(assignment)
            native_losses = self._native_losses(masks, classes, sample, assignment)
            components[f"loss_ce_{layer_index}"] = native_losses[0]
            components[f"loss_mask_{layer_index}"] = native_losses[1]
            components[f"loss_dice_{layer_index}"] = native_losses[2]
            if regions is not None:
                region_losses.append(self._region_loss(regions, sample, assignment)[0])
                consistency_losses.append(
                    self._consistency_loss(masks, regions, sample)
                )
        zero = final_masks.sum() * 0.0
        components["loss_region"] = (
            torch.stack(region_losses).mean() if region_losses else zero
        )
        components["loss_consistency"] = (
            torch.stack(consistency_losses).mean() if consistency_losses else zero
        )

        total = final_masks.sum() * 0.0
        for name, value in components.items():
            if name.startswith("loss_ce"):
                total = total + self.native_weight * self.cost_class * value
            elif name.startswith("loss_mask"):
                total = total + self.native_weight * self.cost_mask * value
            elif name.startswith("loss_dice"):
                total = total + self.native_weight * self.cost_dice * value
        total = total + self.region_weight * components["loss_region"]
        if self.consistency_ramp_updates == 0:
            ramp = 1.0
        else:
            ramp = min(float(optimizer_update) / self.consistency_ramp_updates, 1.0)
        consistency_applied = self.consistency_weight * ramp
        total = total + consistency_applied * components["loss_consistency"]
        return ObservationLossResult(
            components=components,
            total=total,
            final_assignment=final_assignment,
            aux_assignments=tuple(aux_assignments),
            applied_weights={
                "loss_ce": self.native_weight * self.cost_class,
                "loss_mask": self.native_weight * self.cost_mask,
                "loss_dice": self.native_weight * self.cost_dice,
                "loss_region": self.region_weight,
                "loss_consistency": consistency_applied,
            },
            diagnostics={
                "target_count": int(sample.instance_masks.shape[0]),
                "matched_target_count": len(final_assignment.target_indices),
                "unmatched_target_count": len(
                    final_assignment.unmatched_target_indices
                ),
                "supervised_region_count": supervised_regions,
                "excluded_unmatched_region_mass": excluded_mass,
            },
        )


__all__ = [
    "MaskedHungarianMatcher",
    "ObservationLossError",
    "ObservationLossResult",
    "ObservationSetCriterion",
    "PairAssignment",
]
