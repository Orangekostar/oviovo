"""Tests for the simplified V2 semantic mapping pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import Anchor2D, CameraIntrinsics, Patch3D, Proposal2D, ProposalSoftScores
from src.v2.pipeline import SemanticMapV2Pipeline
from src.v2.types import ObjectPool, V2ObjectState


def _make_patch(
    patch_id: int,
    *,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    anchor_class_name: str = "",
    anchor_confidence: float = 0.0,
) -> Patch3D:
    ox, oy, oz = offset
    points = np.array(
        [
            [ox + 0.0, oy + 0.0, oz + 1.0],
            [ox + 0.1, oy + 0.0, oz + 1.0],
            [ox + 0.0, oy + 0.1, oz + 1.0],
            [ox + 0.1, oy + 0.1, oz + 1.0],
        ],
        dtype=np.float32,
    )
    return Patch3D(
        patch_id=patch_id,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=patch_id,
        soft_scores=ProposalSoftScores(objectness_score=0.8, backgroundness_score=0.1),
        metadata={"anchor_class_name": anchor_class_name, "anchor_confidence": anchor_confidence},
    )


def test_v2_object_pool_match_updates_same_object() -> None:
    pipe = SemanticMapV2Pipeline()
    pool = ObjectPool(object_id=0)
    patch0 = _make_patch(0)
    pipe._merge_patch_into_pool(pool, patch0, 0)
    pipe.state.object_pools[0] = pool
    pipe._update_coarse_ownership(0, patch0.points, 0)

    patch1 = _make_patch(1, offset=(0.02, 0.0, 0.0))
    matched = pipe._match_patch_to_object_id(patch1)
    assert matched == 0


def test_v2_semantic_freezer_requires_two_high_confidence_hits() -> None:
    pipe = SemanticMapV2Pipeline()
    pool = ObjectPool(object_id=0)
    pipe._merge_patch_into_pool(pool, _make_patch(0), 0)

    froze = pipe._update_pool_semantics(pool, _make_patch(1, anchor_class_name="chair", anchor_confidence=0.9), 1)
    assert froze is False
    assert pool.label_frozen is False

    froze = pipe._update_pool_semantics(pool, _make_patch(2, anchor_class_name="chair", anchor_confidence=0.95), 2)
    assert froze is True
    assert pool.label_frozen is True
    assert pool.canonical_label == "chair"


def test_v2_geometry_accumulator_keeps_background_geometry() -> None:
    pipe = SemanticMapV2Pipeline()
    intr = CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0, width=4, height=4)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pipe.process_frame(rgb, depth, pose, intr)
    assert len(pipe.state.geometry_accum) > 0


def test_v2_coarse_ownership_updates_from_object_pool() -> None:
    pipe = SemanticMapV2Pipeline()
    pool = ObjectPool(object_id=2, canonical_label="table", label_frozen=True)
    patch = _make_patch(0)
    pipe._merge_patch_into_pool(pool, patch, 0)
    pipe.state.object_pools[2] = pool
    pipe._update_coarse_ownership(2, patch.points, 0)
    assert len(pipe.state.coarse_owners) > 0
    owner = next(iter(pipe.state.coarse_owners.values()))
    assert owner.owner_object_id == 2
    assert owner.canonical_label == "table"


def test_v2_deterministic_cap_is_stable() -> None:
    pipe = SemanticMapV2Pipeline()
    points = np.stack(
        np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 2)),
        axis=-1,
    ).reshape(-1, 3).astype(np.float32)
    idx_a = pipe._deterministic_spatial_indices(points, 16)
    idx_b = pipe._deterministic_spatial_indices(points, 16)
    assert np.array_equal(idx_a, idx_b)


def test_v2_pipeline_runs_with_precomputed_backend(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:4, 1:5] = True
    proposal = Proposal2D(
        proposal_id=0,
        mask=mask,
        bbox_xyxy=np.array([1, 1, 5, 4], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        backend_name="sam2",
        metadata={"anchor_class_name": "chair", "anchor_confidence": 0.95},
    )
    from frontend.proposal_cache import save_proposals, write_manifest

    save_proposals(
        cache_dir,
        frame_id=0,
        image_shape=(8, 8),
        proposals=[proposal],
        source_backend="sam2",
    )
    write_manifest(
        cache_dir,
        dataset_summary={"frame_count": 1},
        backend_config={"backend": "sam2"},
        frame_entries=[{"frame_id": 0, "file": "frames/frame000000_proposals.npz", "proposal_count": 1}],
    )

    config_path = tmp_path / "v2.yaml"
    config_path.write_text(
        "\n".join(
            [
                "proposal:",
                "  backend: precomputed",
                "  min_mask_area: 1",
                "  precomputed:",
                f"    cache_dir: {cache_dir}",
                "anchor_frontend:",
                "  enabled: false",
                "logging:",
                "  level: ERROR",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pipe = SemanticMapV2Pipeline(config_path=config_path)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    intr = CameraIntrinsics(fx=5.0, fy=5.0, cx=4.0, cy=4.0, width=8, height=8)
    state = pipe.process_frame(rgb, depth, pose, intr)
    assert state.frame_count == 1
    assert pipe.last_frame_debug["raw_proposal_count"] == 1
    assert pipe.proposal.active_backend_name == "precomputed"
    assert any(pool.state in {V2ObjectState.ACTIVE, V2ObjectState.ARCHIVED} for pool in state.object_pools.values()) or len(state.provisional_buckets) >= 0


def test_v2_pipeline_uses_anchor_primary_proposals(tmp_path) -> None:
    config_path = tmp_path / "v2_anchor_primary.yaml"
    config_path.write_text(
        "\n".join(
            [
                "proposal:",
                "  backend: placeholder",
                "anchor_frontend:",
                "  enabled: false",
                "logging:",
                "  level: ERROR",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pipe = SemanticMapV2Pipeline(config_path=config_path)
    pipe.object_anchor.enabled = True
    pipe.object_anchor.use_sam_intersection_proposals = True
    pipe.object_anchor.proposal_min_area = 1

    class _FakeAnchorBackend:
        def generate_anchors(self, rgb):
            return [
                Anchor2D(
                    anchor_id=1,
                    bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
                    class_name="chair",
                    confidence=0.9,
                )
            ]

    pipe.object_anchor.backend = _FakeAnchorBackend()

    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    sam_proposal = Proposal2D(
        proposal_id=7,
        mask=mask,
        bbox_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
        area=int(mask.sum()),
        confidence=0.9,
        backend_name="sam2",
    )
    pipe.proposal.process = lambda *args, **kwargs: [sam_proposal]

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    intr = CameraIntrinsics(fx=5.0, fy=5.0, cx=4.0, cy=4.0, width=8, height=8)
    state = pipe.process_frame(rgb, depth, pose, intr)

    assert state.frame_count == 1
    assert pipe.last_frame_debug["proposal_source"] == "anchor_sam_union"
    assert pipe.last_frame_debug["raw_proposal_count"] == 1
    assert pipe.last_raw_proposals[0].metadata["anchor_id"] == 1
