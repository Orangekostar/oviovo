from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.run_ovi_rescene_b7 import (
    B7EvaluationPackage,
    B7RunError,
    atomic_publish_b7_package,
    authorize_b7_scene,
    canonical_method_input_sha256,
    canonical_relations_bytes,
    evaluate_b7_gate,
    execute_geometric_dense_recovery,
    load_and_validate_b7_config,
    load_bound_json,
    load_two_visit_current_map,
    materialize_geometric_relations,
    semantic_labelers_from_frozen_snapshots,
    validate_visit_sha_bindings,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.two_visit_contracts import PairRelation, VisitMap
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    compose_current_map,
    write_two_visit_current_map,
)
from src.oviv2.two_visit_dense_recovery import DenseRecoveryConfig
from src.oviv2.two_visit_execution import SignedVisibilityConfig
from src.oviv2.two_visit_registration import RegistrationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/evaluation/ovi_rescene_b7_g_v1.json"


def _entity(entity_id: str, label: str, offset: float) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(
            [[offset, 0.0, 1.0], [offset + 0.1, 0.1, 1.0]],
            dtype=np.float32,
        ),
        semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        semantic_label=label,
        semantic_score=0.9,
        lifecycle_state="current",
        first_seen=0.0,
        last_seen=1.0,
        metadata={"semantic_authority": "ovi"},
    )


def _visit(visit_id: int, entity: EntityPrediction) -> VisitMap:
    start = 0 if visit_id == 0 else 10
    return VisitMap(
        visit_id=visit_id,
        snapshot=MapSnapshot(
            method=f"OVI t{visit_id}",
            scene_id="apartment",
            timestamp=float(start + 4),
            entities=[entity],
            background_xyz=None,
            scope="current",
        ),
        coordinate_frame_id="world",
        source_manifest_sha256="a" * 64,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def _record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _baseline() -> tuple[VisitMap, VisitMap, object]:
    t0 = _visit(0, _entity("t0:chair", "Chair", 0.0))
    t1 = _visit(1, _entity("t1:chair", "Chair", 1.0))
    baseline = compose_current_map(
        t0,
        t1,
        (),
        SignedVisibilityGrid.empty(0.05, "b" * 64),
        CompositionConfig(),
    )
    return t0, t1, baseline


def test_bound_json_rejects_hash_or_byte_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    record = _record(path)

    assert load_bound_json(record, label="source")["status"] == "PASS"
    with pytest.raises(B7RunError, match="binding mismatch"):
        load_bound_json({**record, "sha256": "0" * 64}, label="source")
    with pytest.raises(B7RunError, match="binding mismatch"):
        load_bound_json(
            {**record, "byte_count": record["byte_count"] + 1}, label="source"
        )


def test_b3_reader_reconstructs_exact_current_map_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    _t0, _t1, baseline = _baseline()
    manifest_path = write_two_visit_current_map(baseline, tmp_path / "b3")
    loaded = load_two_visit_current_map(_record(manifest_path))

    assert loaded.content_sha256() == baseline.content_sha256()
    assert loaded.source_visit_map_sha256 == baseline.source_visit_map_sha256
    provenance = manifest_path.parent / "provenance.jsonl"
    provenance.write_text(
        provenance.read_text(encoding="utf-8").replace(
            '"visibility_score":1.0', '"visibility_score":0.5', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(B7RunError, match="artifact binding mismatch"):
        load_two_visit_current_map(_record(manifest_path))


def test_frozen_semantic_receipts_supply_only_exact_entity_labels() -> None:
    t0, t1, _baseline_value = _baseline()
    t0_labeler, t1_labeler = semantic_labelers_from_frozen_snapshots(
        t0.snapshot, t1.snapshot
    )

    t0_labels = t0_labeler((type("Evidence", (), {"entity_id": "t0:chair"})(),))
    t1_labels = t1_labeler((type("Evidence", (), {"entity_id": "t1:chair"})(),))

    assert t0_labels == {"t0:chair": ("Chair", 0.9)}
    assert t1_labels == {"t1:chair": ("Chair", 0.9)}
    with pytest.raises(B7RunError, match="semantic receipt lacks"):
        t0_labeler((type("Evidence", (), {"entity_id": "unknown"})(),))


def test_visit_hashes_must_equal_every_frozen_variant_binding() -> None:
    t0, t1, baseline = _baseline()
    common = {"source_visit_map_sha256": [t0.snapshot_sha256, t1.snapshot_sha256]}

    validate_visit_sha_bindings(t0, t1, common, common, common, baseline=baseline)
    bad = {"source_visit_map_sha256": ["f" * 64, t1.snapshot_sha256]}
    with pytest.raises(B7RunError, match="VisitMap SHA binding mismatch"):
        validate_visit_sha_bindings(t0, t1, common, bad, common, baseline=baseline)


def test_relation_serialization_is_canonical_and_order_independent() -> None:
    relations = (
        PairRelation(
            temporal_query_id="geom:000002",
            t0_entity_ids=("t0:b",),
            t1_entity_ids=("t1:b",),
            state="persistent_moved",
            query_confidence=0.8,
            evidence={"query_score": 0.8},
            identity_source="geometric_baseline",
        ),
        PairRelation(
            temporal_query_id="geom:000001",
            t0_entity_ids=("t0:a",),
            t1_entity_ids=("t1:a",),
            state="persistent_static",
            query_confidence=0.9,
            evidence={"query_score": 0.9},
            identity_source="geometric_baseline",
        ),
    )

    first = canonical_relations_bytes(relations)
    second = canonical_relations_bytes(tuple(reversed(relations)))

    assert first == second
    payload = json.loads(first)
    assert [item["temporal_query_id"] for item in payload["relations"]] == [
        "geom:000001",
        "geom:000002",
    ]
    assert payload["relations"][0]["evidence"] == {"query_score": 0.9}


def test_atomic_package_runs_evaluator_after_immutable_method_inventory(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def method(stage: Path) -> dict[str, object]:
        calls.append("method")
        path = stage / "method.json"
        path.write_text('{"status":"PASS"}\n', encoding="utf-8")
        return {"method": _record(path)}

    def evaluator(stage: Path, inventory: dict[str, object]) -> dict[str, object]:
        calls.append("evaluator")
        assert (stage / "method.json").is_file()
        assert inventory["method"]["sha256"] == _record(stage / "method.json")["sha256"]
        return {"surface_f1_at_5cm": 0.5}

    output = tmp_path / "run"
    manifest_path = atomic_publish_b7_package(
        output,
        method_writer=method,
        evaluator=evaluator,
        manifest_identity={"config_sha256": "c" * 64},
    )

    assert calls == ["method", "evaluator"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert set(manifest["artifacts"]) == {"method", "metrics"}
    assert manifest["artifacts"]["method"]["path"] == "method.json"
    assert manifest["artifacts"]["metrics"]["path"] == "metrics.json"
    for record in manifest["artifacts"].values():
        artifact = manifest_path.parent / record["path"]
        assert _record(artifact)["sha256"] == record["sha256"]


def test_atomic_package_normalizes_evaluator_artifact_paths(tmp_path: Path) -> None:
    def method(stage: Path) -> dict[str, object]:
        path = stage / "method.json"
        path.write_text("method\n", encoding="utf-8")
        return {"method": _record(path)}

    def evaluator(stage: Path, _inventory: dict[str, object]) -> B7EvaluationPackage:
        path = stage / "attribution" / "summary.json"
        path.parent.mkdir()
        path.write_text("{}\n", encoding="utf-8")
        return B7EvaluationPackage(
            metrics={"surface_f1_at_5cm": 0.5},
            artifacts={"attribution_summary": _record(path)},
        )

    manifest_path = atomic_publish_b7_package(
        tmp_path / "run-with-attribution",
        method_writer=method,
        evaluator=evaluator,
        manifest_identity={"config_sha256": "c" * 64},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["artifacts"]["attribution_summary"]
    assert record["path"] == "attribution/summary.json"
    assert _record(manifest_path.parent / record["path"])["sha256"] == record["sha256"]


def test_method_input_hash_excludes_evaluator_only_frozen_records() -> None:
    config = load_and_validate_b7_config(
        CONFIG_PATH,
        verify_source_bindings=False,
        verify_frozen_inputs=False,
    )
    records = config["frozen_inputs"]["apartment"]
    arguments = {
        "records": records,
        "visit_map_sha256": ("a" * 64, "b" * 64),
        "baseline_current_map_sha256": "c" * 64,
        "relations_sha256": "d" * 64,
        "method": config["method"],
    }
    expected = canonical_method_input_sha256(**arguments)

    for role in (
        "b3_metrics",
        "common_v2_target_manifest",
        "semantic_aliases",
        "semantic_label_space",
        "matrix_summary",
    ):
        changed = deepcopy(records)
        changed[role] = {
            "path": f"/evaluator-only/{role}",
            "sha256": "f" * 64,
            "byte_count": 1,
        }
        assert (
            canonical_method_input_sha256(**{**arguments, "records": changed})
            == expected
        )


def test_atomic_package_removes_staging_on_evaluator_failure(tmp_path: Path) -> None:
    output = tmp_path / "failed"

    def method(stage: Path) -> dict[str, object]:
        path = stage / "method.json"
        path.write_text("method\n", encoding="utf-8")
        return {"method": _record(path)}

    def evaluator(_stage: Path, _inventory: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        atomic_publish_b7_package(
            output,
            method_writer=method,
            evaluator=evaluator,
            manifest_identity={"config_sha256": "c" * 64},
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed.*"))


def test_office_is_blocked_before_any_asset_access() -> None:
    config = {
        "scene_policy": {
            "apartment": {"status": "ENABLED", "attempt_count": 0},
            "office": {"status": "OFFICE_NOT_RUN_HELD_OUT", "attempt_count": 0},
        }
    }

    authorize_b7_scene(config, "apartment")
    with pytest.raises(B7RunError, match="OFFICE_NOT_RUN_HELD_OUT"):
        authorize_b7_scene(config, "office")


def test_checked_in_config_freezes_method_and_gate_before_score(tmp_path: Path) -> None:
    config = load_and_validate_b7_config(
        CONFIG_PATH,
        verify_source_bindings=False,
        verify_frozen_inputs=False,
    )

    assert config["status"] == "FROZEN_BEFORE_FIRST_B7_SCORE"
    assert config["method"]["registration"] == RegistrationConfig().to_json_record()
    assert config["method"]["dense_recovery"] == DenseRecoveryConfig().to_json_record()
    assert config["method"]["signed_visibility"] == asdict(SignedVisibilityConfig())
    assert config["success_gate"] == {
        "maximum_ghost": 0.02,
        "minimum_surface_precision_at_5cm": 0.7244127782,
        "minimum_background_f1_at_5cm": 0.3563600098,
        "minimum_observed_stale_precision": 0.9259634046,
        "minimum_unobserved_recall": 0.0806464685,
        "minimum_surface_f1_at_5cm": 0.4323354117,
        "minimum_accepted_registrations": 1,
        "minimum_recovered_points": 1,
    }
    assert {"rgbd_export_manifest", "rgbd_lock"} <= set(
        config["frozen_inputs"]["apartment"]
    )
    assert config["frozen_inputs"]["apartment"]["causal_schedule"]["path"] == (
        "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
    )
    assert config["frozen_inputs"]["apartment"]["rgbd_lock"]["path"] == (
        "configs/evaluation/manifests/oviv2_tesse_cd_cache.json"
    )
    assert {
        "current_metrics",
        "dataset_loader",
        "evaluation_contracts",
        "map_exporter",
        "semantic_crosswalk",
        "visibility_engine",
    } <= set(config["source_bindings"])

    changed = deepcopy(config)
    changed["method"]["registration"]["maximum_iterations"] = 41
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(B7RunError, match="registration configuration"):
        load_and_validate_b7_config(
            path,
            verify_source_bindings=False,
            verify_frozen_inputs=False,
        )


def test_materialized_geometric_relations_are_deterministic() -> None:
    t0 = _visit(0, _entity("chair", "Chair", 0.0))
    t1 = _visit(1, _entity("chair", "Chair", 0.05))

    first = materialize_geometric_relations(t0, t1)
    second = materialize_geometric_relations(t0, t1)

    assert canonical_relations_bytes(first) == canonical_relations_bytes(second)
    assert len(first) == 1
    assert first[0].identity_source == "geometric_baseline"


def test_runner_executes_registration_visibility_and_dense_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = np.stack(
        np.meshgrid(
            np.linspace(0.0, 0.18, 5),
            np.linspace(0.0, 0.14, 4),
            np.linspace(0.0, 0.10, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    grid[:, 0] += 0.17 * grid[:, 1] ** 2
    source = _entity("chair", "Chair", 0.0)
    source.points_xyz = np.asarray(grid, dtype=np.float32)
    target = _entity("chair", "Chair", 0.0)
    target.points_xyz = np.asarray(grid + np.asarray([0.6, 0.1, 0.0]), dtype=np.float32)
    t0 = _visit(0, source)
    t1 = _visit(1, target)
    relation = PairRelation(
        temporal_query_id="geom:000000",
        t0_entity_ids=("chair",),
        t1_entity_ids=("chair",),
        state="persistent_moved",
        query_confidence=0.95,
        evidence={"query_score": 0.95},
        identity_source="geometric_baseline",
    )
    baseline = compose_current_map(
        t0,
        t1,
        (),
        SignedVisibilityGrid.empty(0.05, "b" * 64),
        CompositionConfig(),
    )

    def visibility(points, _frames, config, *, source_sha256):
        keys = np.unique(
            np.floor(np.asarray(points) / config.voxel_size_m).astype(np.int64), axis=0
        )
        return SignedVisibilityGrid(
            voxel_size_m=config.voxel_size_m,
            voxel_keys=keys,
            statuses=("unobserved",) * len(keys),
            source_sha256=source_sha256,
        )

    monkeypatch.setattr(
        "scripts.evaluation.run_ovi_rescene_b7.derive_signed_visibility_for_points",
        visibility,
    )
    result = execute_geometric_dense_recovery(
        t0=t0,
        t1=t1,
        baseline=baseline,
        relations=(relation,),
        frames=(object(),),
        method_input_sha256="d" * 64,
    )

    assert sum(item.accepted for item in result.registrations) == 1
    assert 0 < result.recovered_point_count < len(source.points_xyz)
    assert result.removed_baseline_point_count == len(source.points_xyz)


def test_gate_requires_every_preregistered_safety_and_gain_predicate() -> None:
    metrics = {
        "ghost": 0.01,
        "surface_precision_at_5cm": 0.73,
        "background_f1_at_5cm": 0.36,
        "t1_observed_region_stale_precision": 0.93,
        "t1_unobserved_region_recall": 0.081,
        "surface_f1_at_5cm": 0.433,
    }

    passed = evaluate_b7_gate(metrics, accepted_registrations=1, recovered_points=1)
    assert passed["decision"] == "GO_B7_MAPPING_EXTENSION"
    assert all(passed["predicates"].values())

    failed = evaluate_b7_gate(
        {**metrics, "ghost": 0.020001},
        accepted_registrations=1,
        recovered_points=1,
    )
    assert failed["decision"] == "STOP_B7_MAPPING_EXTENSION"
    assert failed["predicates"]["ghost"] is False
