from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.ovimap_static_anchor import (
    PrefixIdentitySample,
    StaticAnchorConfig,
    bind_anchor_identities,
)


def _entity(
    entity_id: str,
    center_x: float,
    label: str,
    feature: tuple[float, float],
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(
            (
                (center_x - 0.1, -0.1, -0.1),
                (center_x + 0.1, 0.1, 0.1),
                (center_x, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        semantic_embedding=np.asarray(feature, dtype=np.float32),
        semantic_label=label,
        semantic_score=0.9,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=262.0,
        metadata={"authority": "ovimap_anchor"},
    )


def _anchor() -> MapSnapshot:
    return MapSnapshot(
        method="OVI-MAP causal static anchor",
        scene_id="apartment",
        timestamp=262.0,
        entities=[
            _entity("ovimap:1", 0.0, "Chair", (1.0, 0.0)),
            _entity("ovimap:2", 2.0, "Table", (0.0, 1.0)),
        ],
        background_xyz=np.asarray(((4.0, 0.0, 0.0),), dtype=np.float32),
        scope="current",
    )


def _sample(
    temporal_id: int,
    center_x: float,
    label: str,
    feature: tuple[float, float],
    *,
    frame_index: int = 262,
) -> PrefixIdentitySample:
    return PrefixIdentitySample(
        temporal_entity_id=temporal_id,
        frame_index=frame_index,
        centroid_xyz=(center_x, 0.0, 0.0),
        points_xyz=np.asarray(
            (
                (center_x - 0.1, -0.1, -0.1),
                (center_x + 0.1, 0.1, 0.1),
                (center_x, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        semantic_label=label,
        semantic_embedding=np.asarray(feature, dtype=np.float32),
        geometry_epoch=0,
    )


def _config() -> StaticAnchorConfig:
    return StaticAnchorConfig(
        minimum_spatial_iou=0.01,
        maximum_centroid_distance_m=0.75,
        minimum_semantic_cosine=0.65,
        moved_displacement_m=0.2,
        background_voxel_size_m=0.05,
    )


def test_bind_uses_pre_intervention_geometry_and_is_one_to_one() -> None:
    state = bind_anchor_identities(
        _anchor(),
        (
            _sample(8, 2.0, "Table", (0.0, 1.0)),
            _sample(7, 0.0, "Chair", (1.0, 0.0)),
        ),
        _config(),
        cutoff_frame=262,
    )

    assert state.cutoff_frame == 262
    assert state.last_frame_index == 262
    assert state.bindings == (("ovimap:1", 7), ("ovimap:2", 8))
    assert state.initial_geometry_epochs == ((7, 0), (8, 0))
    assert state.removed_anchor_ids == frozenset()


def test_binding_is_invariant_to_anchor_and_sample_order() -> None:
    anchor = _anchor()
    reversed_anchor = replace(anchor, entities=list(reversed(anchor.entities)))
    samples = (
        _sample(7, 0.0, "Chair", (1.0, 0.0)),
        _sample(8, 2.0, "Table", (0.0, 1.0)),
    )

    expected = bind_anchor_identities(
        anchor, samples, _config(), cutoff_frame=262
    )
    observed = bind_anchor_identities(
        reversed_anchor,
        tuple(reversed(samples)),
        _config(),
        cutoff_frame=262,
    )

    assert observed.bindings == expected.bindings
    assert observed.initial_geometry_epochs == expected.initial_geometry_epochs


def test_binding_rejects_future_sample() -> None:
    with pytest.raises(ValueError, match="after causal cutoff"):
        bind_anchor_identities(
            _anchor(),
            (_sample(7, 0.0, "Chair", (1.0, 0.0), frame_index=263),),
            _config(),
            cutoff_frame=262,
        )


def test_binding_rejects_duplicate_temporal_ids() -> None:
    with pytest.raises(ValueError, match="temporal entity IDs must be unique"):
        bind_anchor_identities(
            _anchor(),
            (
                _sample(7, 0.0, "Chair", (1.0, 0.0)),
                _sample(7, 2.0, "Table", (0.0, 1.0)),
            ),
            _config(),
            cutoff_frame=262,
        )


def test_binding_rejects_semantic_conflict_despite_spatial_overlap() -> None:
    state = bind_anchor_identities(
        _anchor(),
        (_sample(7, 0.0, "Table", (0.0, 1.0)),),
        _config(),
        cutoff_frame=262,
    )

    assert state.bindings == ()


def test_binding_deterministically_assigns_single_candidate_once() -> None:
    anchor = _anchor()
    anchor.entities[1].points_xyz = anchor.entities[0].points_xyz
    anchor.entities[1].semantic_label = "Chair"
    anchor.entities[1].semantic_embedding = anchor.entities[0].semantic_embedding

    state = bind_anchor_identities(
        anchor,
        (_sample(7, 0.0, "Chair", (1.0, 0.0)),),
        _config(),
        cutoff_frame=262,
    )

    assert state.bindings == (("ovimap:1", 7),)
