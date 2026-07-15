from __future__ import annotations

import numpy as np

from src.core.data_structures import Patch3D, ProposalSoftScores
from src.pipelines.legacy_observation_adapter import LegacyPatchObservationAdapter


def make_patch() -> Patch3D:
    return Patch3D(
        patch_id=4,
        points=np.array([[0.01, 0.01, 1.01], [0.06, 0.01, 1.01]], dtype=np.float32),
        centroid=np.array([0.035, 0.01, 1.01], dtype=np.float32),
        bbox_min=np.array([0.01, 0.01, 1.01], dtype=np.float32),
        bbox_max=np.array([0.06, 0.01, 1.01], dtype=np.float32),
        timestamp=2.0,
        source_frame_id=9,
        soft_scores=ProposalSoftScores(objectness_score=0.8),
        metadata={
            "source_proposal_id": 12,
            "source_bbox_xyxy": np.array([1.0, 2.0, 5.0, 6.0], dtype=np.float32),
            "source_backend_name": "sam2",
            "anchor_class_name": "book",
            "anchor_confidence": 0.75,
            "refinement_key": "9:12",
            "geometric_features": {"depth_valid_ratio": 0.9},
            "source_raw_proposal_ids": [2, 3],
        },
    )


def test_legacy_adapter_converts_without_mutating_patch() -> None:
    patch = make_patch()
    original_metadata = dict(patch.metadata)
    mask = np.array([[False, True], [False, False]], dtype=bool)
    batch = LegacyPatchObservationAdapter(voxel_size=0.05).convert(
        [patch],
        masks_by_patch_id={4: mask},
    )

    observation = batch.observations[0]
    assert batch.frame_id == 9
    assert observation.observation_id == "9:12:4"
    assert observation.detector_label == "book"
    assert observation.mask is not mask
    assert observation.mask.flags.writeable is False
    assert observation.voxel_keys.shape == (2, 3)
    assert patch.metadata == original_metadata


def test_legacy_adapter_allows_missing_mask_during_shadow_migration() -> None:
    observation = LegacyPatchObservationAdapter(voxel_size=0.05).convert([make_patch()]).observations[0]
    assert observation.mask is None
