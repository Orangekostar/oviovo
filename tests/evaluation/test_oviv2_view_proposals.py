from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import math

import numpy as np
import pytest

from src.evaluation.oviv2_view_graph import ObservationEdge
from src.evaluation.oviv2_view_proposals import (
    ProposalPyramidConfig,
    VoxelProposal,
    build_proposal_pyramid,
)
from src.oviv2.observations import FrameObservation, ObservationKind


def _voxels(*indices: int) -> set[tuple[int, int, int]]:
    return {(index, 0, 0) for index in indices}


def _observation(
    observation_id: int,
    frame_id: int,
    voxels: set[tuple[int, int, int]],
    **changes: object,
) -> FrameObservation:
    values: dict[str, object] = {
        "observation_id": observation_id,
        "frame_id": frame_id,
        "timestamp": float(frame_id),
        "kind": ObservationKind.OBJECT,
        "label": "chair",
        "semantic_id": 2,
        "confidence": 0.8,
        "mask": np.ones((2, 2), dtype=bool),
        "bbox_xyxy": (0.0, 0.0, 2.0, 2.0),
        "voxel_keys": frozenset(voxels),
        "centroid_xyz": (0.0, 0.0, 1.0),
        "bounds_min_xyz": (0.0, 0.0, 1.0),
        "bounds_max_xyz": (0.0, 0.0, 1.0),
        "border_contact_fraction": 0.2,
    }
    values.update(changes)
    return FrameObservation(**values)


def _edge(
    left_id: int,
    right_id: int,
    *,
    iou: float = 0.2,
    left_coverage: float = 0.5,
    right_coverage: float = 0.5,
    feature_cosine: float | None = None,
) -> ObservationEdge:
    return ObservationEdge(
        left_id=left_id,
        right_id=right_id,
        shared_voxels=2,
        voxel_iou=iou,
        left_coverage=left_coverage,
        right_coverage=right_coverage,
        feature_cosine=feature_cosine,
        view_direction_cosine=None,
    )


def _config(**changes: object) -> ProposalPyramidConfig:
    values: dict[str, object] = {
        "strong_voxel_iou": 0.05,
        "strong_directed_coverage": 0.25,
        "minimum_feature_cosine": 0.2,
        "minimum_distinct_views": 2,
        "core_vote_fraction": 2.0 / 3.0,
        "inclusive_coverage": 0.25,
        "weak_merge_iou": 0.1,
        "minimum_voxels": 3,
        "maximum_proposals": 64,
    }
    values.update(changes)
    return ProposalPyramidConfig(**values)


def _by_kind(proposals: tuple[VoxelProposal, ...]) -> dict[str, VoxelProposal]:
    return {proposal.kind: proposal for proposal in proposals}


def test_builds_consensus_core_and_sparse_inclusive_union_from_three_views() -> None:
    observations = (
        _observation(10, 1, _voxels(0, 1, 2, 3, 10)),
        _observation(21, 2, _voxels(0, 1, 2, 3, 11)),
        _observation(33, 3, _voxels(0, 1, 2, 3, 12)),
        _observation(44, 4, _voxels(0, 20, 21)),
    )
    proposals = build_proposal_pyramid(
        observations,
        (_edge(10, 21), _edge(10, 33), _edge(21, 33)),
        _config(),
    )

    by_kind = _by_kind(proposals)
    assert by_kind["consensus"].voxel_keys == frozenset(_voxels(0, 1, 2, 3, 10, 11, 12))
    assert by_kind["core"].voxel_keys == frozenset(_voxels(0, 1, 2, 3))
    assert by_kind["union"].voxel_keys == frozenset(_voxels(0, 1, 2, 3, 10, 11, 12, 20, 21))
    assert by_kind["union"].observation_ids == (10, 21, 33, 44)
    assert by_kind["union"].supporter_frame_ids == (1, 2, 3, 4)


def test_core_counts_same_frame_observations_once() -> None:
    observations = (
        _observation(10, 1, _voxels(0, 1, 2, 3)),
        _observation(11, 1, _voxels(0, 1, 2, 3)),
        _observation(21, 2, _voxels(0, 1, 2, 3)),
        _observation(33, 3, _voxels(4, 5, 6)),
    )
    proposals = build_proposal_pyramid(
        observations,
        (_edge(10, 21), _edge(11, 21)),
        _config(core_vote_fraction=1.0, minimum_voxels=3),
    )

    assert _by_kind(proposals)["core"].voxel_keys == frozenset(_voxels(0, 1, 2, 3))


def test_weak_bridge_creates_hierarchical_without_merging_strong_components() -> None:
    observations = (
        _observation(10, 1, _voxels(0, 1, 2)),
        _observation(21, 2, _voxels(0, 1, 2)),
        _observation(33, 3, _voxels(3, 4, 5)),
        _observation(44, 4, _voxels(3, 4, 5)),
    )
    evidence = (
        _edge(10, 21, iou=0.5),
        _edge(33, 44, iou=0.5),
        _edge(21, 33, iou=0.15, left_coverage=0.2, right_coverage=0.2, feature_cosine=0.9),
    )
    proposals = build_proposal_pyramid(
        observations,
        evidence,
        _config(strong_voxel_iou=0.2, strong_directed_coverage=0.25),
    )

    assert [proposal.kind for proposal in proposals].count("core") == 2
    hierarchical = _by_kind(proposals)["hierarchical"]
    assert hierarchical.observation_ids == (10, 21, 33, 44)
    assert hierarchical.voxel_keys == frozenset(_voxels(0, 1, 2, 3, 4, 5))


def test_weak_merge_requires_good_available_feature_but_missing_feature_is_allowed() -> None:
    observations = (
        _observation(10, 1, _voxels(0, 1, 2)), _observation(21, 2, _voxels(0, 1, 2)),
        _observation(33, 3, _voxels(3, 4, 5)), _observation(44, 4, _voxels(3, 4, 5)),
    )
    strong = (_edge(10, 21, iou=0.5), _edge(33, 44, iou=0.5))
    bad = strong + (_edge(21, 33, iou=0.15, left_coverage=0.1, right_coverage=0.1, feature_cosine=0.1),)
    missing = strong + (_edge(21, 33, iou=0.15, left_coverage=0.1, right_coverage=0.1),)

    config = _config(strong_voxel_iou=0.2, strong_directed_coverage=0.25)
    assert "hierarchical" not in _by_kind(build_proposal_pyramid(observations, bad, config))
    assert "hierarchical" in _by_kind(build_proposal_pyramid(observations, missing, config))


def test_semantic_mismatch_is_rejected_and_strong_selection_honors_coverage() -> None:
    observations = (
        _observation(10, 1, _voxels(0, 1, 2)),
        _observation(21, 2, _voxels(0, 1, 2), semantic_id=3),
        _observation(33, 3, _voxels(0, 1, 2, 3)),
    )
    with pytest.raises(ValueError, match="semantic"):
        build_proposal_pyramid(observations, (_edge(10, 21, iou=1.0),), _config())
    proposals = build_proposal_pyramid(
        (observations[0], observations[2]),
        (_edge(10, 33, iou=0.01, left_coverage=0.3, right_coverage=0.3, feature_cosine=0.2),),
        _config(),
    )
    assert _by_kind(proposals)["consensus"].observation_ids == (10, 33)


def test_minimum_view_and_voxel_gates_and_empty_cases() -> None:
    pair = (_observation(10, 1, _voxels(0, 1, 2)), _observation(21, 2, _voxels(0, 1, 2)))
    assert build_proposal_pyramid((), (), _config()) == ()
    assert build_proposal_pyramid(pair[:1], (), _config()) == ()
    assert build_proposal_pyramid(pair, (_edge(10, 21),), _config(minimum_distinct_views=3)) == ()
    assert build_proposal_pyramid(pair, (_edge(10, 21),), _config(minimum_voxels=4)) == ()


def test_score_uses_exact_gt_free_components() -> None:
    proposal = VoxelProposal(
        proposal_id="x",
        semantic_id=2,
        kind="core",
        voxel_keys=frozenset(_voxels(1, 2, 3)),
        observation_ids=(10, 21),
        supporter_frame_ids=(1, 2, 3, 4),
        mean_confidence=0.8,
        consensus_density=0.5,
        border_fraction=0.25,
    )
    assert proposal.score == pytest.approx(0.30 * 0.5 + 0.25 * 0.8 + 0.25 * 0.5 + 0.20 * 0.75)
    assert 0.0 <= proposal.score <= 1.0


def test_deduplicates_by_kind_priority_and_assigns_stable_ids() -> None:
    observations = (
        _observation(10, 1, _voxels(0, 1, 2)),
        _observation(21, 2, _voxels(0, 1, 2)),
    )
    proposals = build_proposal_pyramid(observations, (_edge(10, 21),), _config(core_vote_fraction=1.0))

    assert len(proposals) == 1
    assert proposals[0].kind == "core"
    assert proposals[0].proposal_id == "vc:002:core:0000"
    assert build_proposal_pyramid(tuple(reversed(observations)), (_edge(10, 21),), _config(core_vote_fraction=1.0)) == proposals


def test_rejects_over_limit_instead_of_truncating() -> None:
    observations = (
        _observation(10, 1, _voxels(0, 1, 2)), _observation(21, 2, _voxels(0, 1, 2)),
        _observation(33, 3, _voxels(10, 11, 12)), _observation(44, 4, _voxels(10, 11, 12)),
    )
    evidence = (_edge(10, 21), _edge(33, 44))
    with pytest.raises(ValueError, match="maximum_proposals"):
        build_proposal_pyramid(observations, evidence, _config(maximum_proposals=1))


@pytest.mark.parametrize("field", ["strong_voxel_iou", "strong_directed_coverage", "core_vote_fraction", "inclusive_coverage", "weak_merge_iou"])
@pytest.mark.parametrize("value", [True, math.nan, math.inf, -0.1, 1.1])
def test_config_rejects_invalid_unit_fields(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _config(**{field: value})


@pytest.mark.parametrize("field", ["minimum_distinct_views", "minimum_voxels", "maximum_proposals"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_config_rejects_invalid_positive_integer_fields(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _config(**{field: value})


def test_records_freeze_and_validate_values() -> None:
    proposal = VoxelProposal("x", 2, "core", _voxels(1, 2, 3), [1, 2], [3, 4], 0.8, 0.5, 0.2)  # type: ignore[arg-type]
    assert proposal.voxel_keys == frozenset(_voxels(1, 2, 3))
    assert proposal.observation_ids == (1, 2)
    assert proposal.supporter_frame_ids == (3, 4)
    with pytest.raises(FrozenInstanceError):
        proposal.semantic_id = 3  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError)):
        VoxelProposal("", 2, "core", frozenset(_voxels(1)), (2,), (3,), 0.5, 0.5, 0.5)
    with pytest.raises((TypeError, ValueError)):
        VoxelProposal("x", 2, "core", frozenset(_voxels(1)), (2, 1), (3,), 0.5, 0.5, 0.5)


def test_building_rejects_invalid_types_duplicate_ids_and_invalid_evidence_endpoints() -> None:
    observations = (_observation(10, 1, _voxels(0, 1, 2)), _observation(21, 2, _voxels(0, 1, 2)))
    with pytest.raises(TypeError):
        build_proposal_pyramid(object(), (), _config())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate observation_id"):
        build_proposal_pyramid((observations[0], observations[0]), (), _config())
    with pytest.raises(ValueError, match="endpoint"):
        build_proposal_pyramid(observations, (_edge(10, 99),), _config())
    with pytest.raises(ValueError, match="semantic"):
        build_proposal_pyramid((observations[0], _observation(21, 2, _voxels(0, 1, 2), semantic_id=3)), (_edge(10, 21),), _config())
    with pytest.raises(ValueError, match="duplicate"):
        build_proposal_pyramid(observations, (_edge(10, 21), _edge(10, 21)), _config())


def test_public_api_contains_no_ground_truth_or_evaluator_and_uses_sparse_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.evaluation.oviv2_view_proposals as proposal_module

    observations = [
        _observation(10, 1, _voxels(0, 1, 2)),
        _observation(21, 2, _voxels(0, 1, 2)),
    ] + [_observation(index, index, _voxels(index + 100)) for index in range(30, 130)]
    calls = 0
    original = proposal_module._core_overlap_fraction

    def counted(core: frozenset[tuple[int, int, int]], candidate: FrameObservation) -> float:
        nonlocal calls
        calls += 1
        return original(core, candidate)

    monkeypatch.setattr(proposal_module, "_core_overlap_fraction", counted)
    build_proposal_pyramid(observations, (_edge(10, 21),), _config())

    assert calls == 2
    assert "ground_truth" not in inspect.signature(build_proposal_pyramid).parameters
    assert "evaluator" not in inspect.signature(build_proposal_pyramid).parameters
