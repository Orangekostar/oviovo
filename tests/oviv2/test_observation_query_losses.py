from __future__ import annotations

import numpy as np
import pytest
import torch

from src.oviv2.observation_query.contracts import ObservationBank
from src.oviv2.observation_query.losses import (
    MaskedHungarianMatcher,
    ObservationSetCriterion,
)
from src.oviv2.rescene_input_bridge import ReSceneModelInput
from src.training.ovi_observation_data import ObservationTrainingSample


def _model_input(model_count: int = 4) -> ReSceneModelInput:
    points = np.column_stack(
        (
            np.arange(model_count, dtype=np.float32),
            np.zeros((model_count, 2), dtype=np.float32),
        )
    )
    split = model_count // 2
    visits = np.asarray([0] * split + [1] * (model_count - split), dtype=np.int8)
    return ReSceneModelInput(
        coordinates_bxyzt=np.column_stack((np.zeros(model_count), points, visits)),
        grid_coordinates_xyz=np.column_stack(
            (np.arange(model_count), np.zeros((model_count, 2), dtype=np.int64))
        ),
        features=np.column_stack(
            (
                points,
                np.full((model_count, 3), 0.5),
                np.tile([0.0, 0.0, 1.0], (model_count, 1)),
            )
        ),
        sparse_batch_offsets=np.asarray([split, model_count]),
        point2segment=np.arange(model_count),
        adapter_to_model=np.arange(model_count),
        model_visit_ids=visits,
        representative_source_point_indices=np.arange(model_count),
        local_frame_indices=np.zeros(model_count, dtype=np.int64),
        global_frame_indices=np.zeros(model_count, dtype=np.int64),
        rows=np.zeros(model_count, dtype=np.int64),
        columns=np.zeros(model_count, dtype=np.int64),
        camera_depth_m=np.ones(model_count),
        observed_depth_m=np.ones(model_count),
        depth_residual_m=np.zeros(model_count),
        neural_voxel_size_m=0.02,
        adapter_geometry_sha256="a" * 64,
        native_sampling_sha256="b" * 64,
        surface_attributes_sha256="c" * 64,
        model_candidates_sha256="d" * 64,
        sampler_source_sha256="e" * 64,
    )


def _sample(
    *,
    masks: np.ndarray,
    class_targets: np.ndarray,
    class_valid: np.ndarray,
    label_valid: np.ndarray,
    region_mass: np.ndarray,
) -> ObservationTrainingSample:
    model = _model_input(masks.shape[1])
    region_count = region_mass.shape[0]
    bank = ObservationBank(
        pair_id="pair",
        model_input_sha256=model.content_sha256(),
        region_keys=tuple(f"r{i}" for i in range(region_count)),
        region_visit_ids=np.zeros(region_count, dtype=np.int8),
        region_frame_ids=np.arange(region_count),
        region_features=np.ones((region_count, 2), dtype=np.float32),
        region_metadata=np.zeros((region_count, 15), dtype=np.float32),
        csr_indptr=np.arange(region_count + 1, dtype=np.int64),
        csr_model_indices=np.arange(region_count, dtype=np.int64),
        csr_weights=np.ones(region_count, dtype=np.float32),
        region_reliability=np.ones(region_count, dtype=np.float32),
        model_visit_ids=model.model_visit_ids,
        source_manifest={},
    )
    return ObservationTrainingSample(
        model_input=model,
        observations=bank,
        instance_masks=torch.tensor(masks, dtype=torch.float32),
        label_valid=torch.tensor(label_valid, dtype=torch.bool),
        class_targets=torch.tensor(class_targets, dtype=torch.long),
        class_valid=torch.tensor(class_valid, dtype=torch.bool),
        temporal_identity_keys=tuple(f"reference:{i + 1}" for i in range(len(masks))),
        ambiguity_metadata={},
        region_instance_mass=torch.tensor(region_mass, dtype=torch.float32),
        region_label_valid=torch.ones(region_count, dtype=torch.bool),
        target_provenance={
            "ground_truth_role": "criterion_only",
            "model_forward_fields": ["model_input", "observations"],
        },
    )


def test_matcher_excludes_unknown_m_rows_before_assignment() -> None:
    matcher = MaskedHungarianMatcher(cost_class=0.0, cost_mask=5.0, cost_dice=2.0)
    sample = _sample(
        masks=np.asarray([[1, 0, 0, 0]], dtype=np.float32),
        class_targets=np.asarray([-100]),
        class_valid=np.asarray([False]),
        label_valid=np.asarray([True, True, False, False]),
        region_mass=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    predicted_masks = torch.tensor(
        [
            [8.0, -8.0],
            [-8.0, 8.0],
            [-100.0, 100.0],
            [-100.0, 100.0],
        ]
    )
    predicted_classes = torch.zeros(2, 3)

    assignment = matcher(
        predicted_masks,
        predicted_classes,
        sample.instance_masks,
        sample.label_valid,
        sample.class_targets,
        sample.class_valid,
    )

    assert assignment.query_indices.tolist() == [0]
    assert assignment.target_indices.tolist() == [0]


def test_region_loss_uses_the_same_query_target_assignment() -> None:
    sample = _sample(
        masks=np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        class_targets=np.asarray([0, 1]),
        class_valid=np.asarray([True, True]),
        label_valid=np.ones(4, dtype=bool),
        region_mass=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    masks = torch.tensor(
        [[-8.0, 8.0], [8.0, -8.0], [-8.0, -8.0], [-8.0, -8.0]],
        requires_grad=True,
    )
    classes = torch.tensor([[[0.0, 8.0, -8.0], [8.0, 0.0, -8.0]]], requires_grad=True)
    correct_regions = torch.tensor(
        [[-8.0, 8.0, -8.0], [8.0, -8.0, -8.0]], requires_grad=True
    )
    wrong_regions = correct_regions.detach()[:, [1, 0, 2]].clone().requires_grad_(True)
    criterion = ObservationSetCriterion(
        foreground_class_count=2,
        num_queries=2,
        cost_class=2.0,
        cost_mask=5.0,
        cost_dice=2.0,
        region_weight=0.2,
        consistency_weight=0.0,
    )

    correct = criterion(
        {
            "pred_masks": [masks],
            "pred_logits": classes,
            "pred_region_logits": correct_regions,
            "aux_outputs": [],
            "region_aux_outputs": [],
        },
        sample,
        optimizer_update=0,
    )
    wrong = criterion(
        {
            "pred_masks": [masks],
            "pred_logits": classes,
            "pred_region_logits": wrong_regions,
            "aux_outputs": [],
            "region_aux_outputs": [],
        },
        sample,
        optimizer_update=0,
    )

    assert correct.final_assignment.query_indices.tolist() == [1, 0]
    assert correct.final_assignment.target_indices.tolist() == [0, 1]
    assert correct.components["loss_region"] < 1e-5
    assert wrong.components["loss_region"] > 10.0


def test_unmatched_gt_mass_is_excluded_not_sent_to_dustbin() -> None:
    sample = _sample(
        masks=np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        class_targets=np.asarray([0, 1]),
        class_valid=np.asarray([True, True]),
        label_valid=np.ones(4, dtype=bool),
        region_mass=np.asarray([[0, 1, 0]], dtype=np.float32),
    )
    criterion = ObservationSetCriterion(
        foreground_class_count=2,
        num_queries=1,
        consistency_weight=0.0,
    )
    output = {
        "pred_masks": [torch.tensor([[8.0], [-8.0], [-8.0], [-8.0]])],
        "pred_logits": torch.tensor([[[8.0, 0.0, -8.0]]]),
        "pred_region_logits": torch.tensor([[0.0, 8.0]]),
        "aux_outputs": [],
        "region_aux_outputs": [],
    }

    result = criterion(output, sample, optimizer_update=0)

    assert result.final_assignment.unmatched_target_indices.tolist() == [1]
    assert result.diagnostics["unmatched_target_count"] == 1
    assert result.diagnostics["excluded_unmatched_region_mass"] == 1.0
    assert result.diagnostics["supervised_region_count"] == 0
    assert result.components["loss_region"] == 0.0


def test_consistency_is_independent_and_sends_gradients_to_both_domains() -> None:
    sample = _sample(
        masks=np.asarray([[1, 0, 0, 0]], dtype=np.float32),
        class_targets=np.asarray([0]),
        class_valid=np.asarray([True]),
        label_valid=np.ones(4, dtype=bool),
        region_mass=np.asarray([[1, 0]], dtype=np.float32),
    )
    mask_logits = torch.tensor([[2.0], [-1.0], [-1.0], [-1.0]], requires_grad=True)
    region_logits = torch.tensor([[-1.0, 2.0]], requires_grad=True)
    criterion = ObservationSetCriterion(
        foreground_class_count=1,
        num_queries=1,
        region_weight=0.0,
        consistency_weight=0.1,
        consistency_ramp_updates=10,
    )
    result = criterion(
        {
            "pred_masks": [mask_logits],
            "pred_logits": torch.tensor([[[2.0, -2.0]]], requires_grad=True),
            "pred_region_logits": region_logits,
            "aux_outputs": [],
            "region_aux_outputs": [],
        },
        sample,
        optimizer_update=10,
    )

    result.total.backward()
    assert result.components["loss_consistency"] > 0
    assert mask_logits.grad is not None and torch.any(mask_logits.grad != 0)
    assert region_logits.grad is not None and torch.any(region_logits.grad != 0)
    assert criterion.obs_surface_null_head.weight.grad is not None
    assert result.applied_weights["loss_consistency"] == 0.1


def test_aux_layer_rematches_and_uses_its_corresponding_region_logits() -> None:
    sample = _sample(
        masks=np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        class_targets=np.asarray([0, 1]),
        class_valid=np.asarray([True, True]),
        label_valid=np.ones(4, dtype=bool),
        region_mass=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    final_masks = torch.tensor([[8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0], [-8.0, -8.0]])
    aux_masks = final_masks.flip(1)
    final_classes = torch.tensor([[[8.0, 0.0, -8.0], [0.0, 8.0, -8.0]]])
    aux_classes = final_classes.flip(1)
    final_regions = torch.tensor([[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]])
    aux_regions = final_regions[:, [1, 0, 2]]
    criterion = ObservationSetCriterion(
        foreground_class_count=2,
        num_queries=2,
        consistency_weight=0.0,
    )

    result = criterion(
        {
            "pred_masks": [final_masks],
            "pred_logits": final_classes,
            "pred_region_logits": final_regions,
            "aux_outputs": [{"pred_masks": [aux_masks], "pred_logits": aux_classes}],
            "region_aux_outputs": [{"pred_region_logits": aux_regions}],
        },
        sample,
        optimizer_update=0,
    )

    assert result.final_assignment.query_indices.tolist() == [0, 1]
    assert result.aux_assignments[0].query_indices.tolist() == [1, 0]
    assert result.components["loss_region"] < 1e-5
    assert result.applied_weights["loss_region"] == 0.2


def test_native_only_criterion_requires_region_weights_to_be_disabled() -> None:
    sample = _sample(
        masks=np.asarray([[1, 0, 0, 0]], dtype=np.float32),
        class_targets=np.asarray([0]),
        class_valid=np.asarray([True]),
        label_valid=np.ones(4, dtype=bool),
        region_mass=np.asarray([[1, 0]], dtype=np.float32),
    )
    outputs = {
        "pred_masks": [torch.tensor([[8.0], [-8.0], [-8.0], [-8.0]])],
        "pred_logits": torch.tensor([[[8.0, -8.0]]]),
        "aux_outputs": [],
    }
    criterion = ObservationSetCriterion(
        foreground_class_count=1,
        num_queries=1,
        region_weight=0.0,
        consistency_weight=0.0,
        require_region_supervision=False,
    )

    result = criterion(outputs, sample, optimizer_update=0)

    assert result.components["loss_region"] == 0.0
    assert result.components["loss_consistency"] == 0.0
    assert result.diagnostics["supervised_region_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_criterion_accepts_cpu_targets_with_cuda_predictions() -> None:
    sample = _sample(
        masks=np.asarray([[1, 0, 0, 0]], dtype=np.float32),
        class_targets=np.asarray([0]),
        class_valid=np.asarray([True]),
        label_valid=np.ones(4, dtype=bool),
        region_mass=np.asarray([[1, 0]], dtype=np.float32),
    )
    device = torch.device("cuda:0")
    criterion = ObservationSetCriterion(
        foreground_class_count=1,
        num_queries=1,
        consistency_weight=0.0,
    ).to(device)

    result = criterion(
        {
            "pred_masks": [
                torch.tensor([[8.0], [-8.0], [-8.0], [-8.0]], device=device)
            ],
            "pred_logits": torch.tensor([[[8.0, -8.0]]], device=device),
            "pred_region_logits": torch.tensor([[8.0, -8.0]], device=device),
            "aux_outputs": [],
            "region_aux_outputs": [],
        },
        sample,
        optimizer_update=0,
    )

    assert result.total.device == device
    assert torch.isfinite(result.total)
