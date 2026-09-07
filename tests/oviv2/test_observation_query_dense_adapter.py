from __future__ import annotations

from src.oviv2.observation_query.dense_adapter import (
    build_observation_composer_inputs,
    build_observation_dense_layers,
)
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    compose_current_map,
)
from tests.oviv2.test_rescene_dense_instance_readout import (
    _bundle,
    _pair,
    _predictions,
)


def test_adapter_preserves_dense_geometry_and_uses_obs_identity_namespace(
    tmp_path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)

    layers = build_observation_dense_layers(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        method_id="OBS_FULL",
        minimum_query_score=0.3,
        point_chunk_size=2,
    )

    assert layers.method_id == "OBS_FULL"
    assert layers.raw_query_count == 2
    assert layers.retained_query_count == 2
    assert layers.dense_point_count == sum(visit.point_count for visit in pair.visits)
    assert layers.xyz_changed_count == 0
    assert all(
        instance.instance_id.startswith("OBS_FULL:")
        for visit in layers.readout.visits
        for instance in visit.instances
    )
    assert all(
        proposal.query_id.startswith("OBS_FULL:query_")
        for proposals in layers.readout.raw_proposals
        for proposal in proposals
    )
    assert all(
        instance.temporal_identity_id is None
        or instance.temporal_identity_id.startswith("OBS_FULL:query_")
        for visit in layers.readout.visits
        for instance in visit.instances
    )
    assert layers.method_view.method_id == "OBS_FULL"
    assert layers.method_view.source_xyz_sha256


def test_adapter_rejects_historical_or_unknown_method_namespace(tmp_path) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)

    for method_id in ("P2", "FULL", ""):
        try:
            build_observation_dense_layers(
                pair,
                surface=bundle.surface,
                supported_view=bundle.supported_view,
                pred_masks_mq=masks,
                pred_logits_qc=logits,
                method_id=method_id,
            )
        except ValueError as error:
            assert "method_id" in str(error)
        else:
            raise AssertionError("invalid method namespace was accepted")


def test_composer_inputs_reassign_every_dense_row_without_changing_xyz(tmp_path) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)
    layers = build_observation_dense_layers(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        method_id="OBS_FULL",
        minimum_query_score=0.3,
        point_chunk_size=2,
    )

    inputs = build_observation_composer_inputs(pair, layers)

    assert tuple(item.visit_id for item in inputs.visit_maps) == (0, 1)
    assert inputs.source_pair_sha256 == pair.content_sha256()
    for visit_id, (source, visit_map, row_map) in enumerate(
        zip(pair.visits, inputs.visit_maps, inputs.source_d_rows, strict=True)
    ):
        assert visit_map.snapshot.method == f"OBS_FULL dense reassigned t{visit_id}"
        assert [item.entity_id for item in visit_map.snapshot.entities] == [
            item.instance_id for item in layers.readout.visits[visit_id].instances
        ]
        claimed = []
        for entity, (entity_id, rows) in zip(
            visit_map.snapshot.entities, row_map, strict=True
        ):
            assert entity_id == entity.entity_id
            assert entity.metadata["semantic_source"] == "parent_ovi_control"
            assert entity.metadata["source_d_row_count"] == len(rows)
            assert (entity.points_xyz == source.points_xyz[rows]).all()
            claimed.extend(rows.tolist())
        background_rows = inputs.background_source_d_rows[visit_id]
        assert (
            visit_map.snapshot.background_xyz == source.points_xyz[background_rows]
        ).all()
        claimed.extend(background_rows.tolist())
        assert sorted(claimed) == list(range(source.point_count))


def test_shared_queries_are_uncertain_candidates_and_existing_composer_accepts_them(
    tmp_path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)
    layers = build_observation_dense_layers(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        method_id="OBS_FULL",
        minimum_query_score=0.3,
        point_chunk_size=2,
    )

    inputs = build_observation_composer_inputs(pair, layers)

    assert inputs.relations
    assert all(item.state == "uncertain" for item in inputs.relations)
    assert all(item.identity_source == "rescene" for item in inputs.relations)
    assert all(item.evidence["candidate_only"] == 1.0 for item in inputs.relations)
    assert all(len(item.t0_entity_ids) == len(item.t1_entity_ids) == 1 for item in inputs.relations)
    current = compose_current_map(
        *inputs.visit_maps,
        inputs.relations,
        SignedVisibilityGrid.empty(0.05, "f" * 64),
        CompositionConfig(),
    )
    assert current.snapshot.scene_id == pair.pair_id
    assert current.relation_ids == tuple(
        item.temporal_query_id for item in inputs.relations
    )


def test_composer_inputs_do_not_invent_cross_visit_relations(tmp_path) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)
    layers = build_observation_dense_layers(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        method_id="OBS_FULL",
        minimum_query_score=0.3,
        point_chunk_size=2,
    )
    visits = tuple(
        type(visit)(
            visit_id=visit.visit_id,
            scan_id=visit.scan_id,
            owner_instance_indices=visit.owner_instance_indices,
            owner_source_codes=visit.owner_source_codes,
            neural_valid=visit.neural_valid,
            dense_to_model_indices=visit.dense_to_model_indices,
            instances=tuple(
                type(instance)(
                    visit_id=instance.visit_id,
                    instance_index=instance.instance_index,
                    instance_id=instance.instance_id,
                    owner_source=instance.owner_source,
                    point_indices=instance.point_indices,
                    raw_query_index=instance.raw_query_index,
                    temporal_identity_id=None,
                    parent_ovi_entity_point_counts=instance.parent_ovi_entity_point_counts,
                    semantic_embedding=instance.semantic_embedding,
                    semantic_labels=instance.semantic_labels,
                    semantic_provenance=instance.semantic_provenance,
                    component_count=instance.component_count,
                    multi_object_conflict=instance.multi_object_conflict,
                )
                for instance in visit.instances
            ),
        )
        for visit in layers.readout.visits
    )
    no_identity_layers = type(layers)(
        method_id=layers.method_id,
        readout=type(layers.readout)(
            pair_content_sha256=layers.readout.pair_content_sha256,
            minimum_query_score=layers.readout.minimum_query_score,
            raw_query_indices=layers.readout.raw_query_indices,
            query_scores=layers.readout.query_scores,
            visits=visits,
            raw_proposals=layers.readout.raw_proposals,
        ),
        method_view=layers.method_view,
        raw_query_count=layers.raw_query_count,
        retained_query_count=layers.retained_query_count,
        dense_point_count=layers.dense_point_count,
        xyz_changed_count=layers.xyz_changed_count,
    )

    inputs = build_observation_composer_inputs(pair, no_identity_layers)

    assert inputs.relations == ()
