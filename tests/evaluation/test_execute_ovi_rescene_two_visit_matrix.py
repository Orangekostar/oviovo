from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import (
    PreparedTwoVisitExecution,
    _enforce_b2_floor,
    _semantic_config,
    execute_prepared_variant,
    load_two_visit_ovi_inputs,
)
from scripts.evaluation.freeze_tesse_two_visit_protocol import protocol_content_sha256
from scripts.evaluation.run_ovi_rescene_two_visit_matrix import (
    execute_required_matrix,
    load_and_validate_matrix,
)
from src.evaluation.baselines.tesse_semantics import (
    SemanticLookup,
    TesseSemanticCrosswalk,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.two_visit_snapshot_metrics import TwoVisitEvaluationContext
from src.oviv2.two_visit_contracts import VisitMap
from src.oviv2.two_visit_current_map import SignedVisibilityGrid

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "configs/evaluation/ovi_rescene_two_visit_matrix.json"


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_semantic_config_binds_vocabulary_and_prompt_rule() -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH, verify_source_bindings=False)

    settings = _semantic_config(
        matrix,
        REPO_ROOT / "configs/evaluation/vocabularies/tesse_cd_apartment.json",
    )

    assert settings == (("object", "things", "stuff", "texture"), 64, 2)
    with np.testing.assert_raises_regex(ValueError, "vocabulary path"):
        _semantic_config(matrix, REPO_ROOT / "different-vocabulary.json")


def test_b2_floor_gate_stops_temporal_variants_above_practical_ghost() -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH, verify_source_bindings=False)
    metrics = {"metric_groups": {"current_state": {"ghost": 0.2}}}

    _enforce_b2_floor(metrics, matrix)

    metrics["metric_groups"]["current_state"]["ghost"] = 0.200001
    with np.testing.assert_raises_regex(ValueError, "B2 Ghost floor"):
        _enforce_b2_floor(metrics, matrix)


def _visit(visit_id: int, point: list[float]) -> VisitMap:
    start = 10 * visit_id
    return VisitMap(
        visit_id=visit_id,
        snapshot=MapSnapshot(
            method=f"OVI-MAP t{visit_id}",
            scene_id="apartment",
            timestamp=float(start + 2),
            entities=[
                EntityPrediction(
                    entity_id=f"ovimap:{visit_id + 1}",
                    points_xyz=np.asarray([point], dtype=np.float32),
                    semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                    semantic_label="chair",
                    semantic_score=0.9,
                    lifecycle_state="current",
                    first_seen=float(start),
                    last_seen=float(start + 2),
                )
            ],
            background_xyz=np.asarray([[1.025, 0.025, 1.025]], dtype=np.float32),
            scope="current",
        ),
        coordinate_frame_id="tesse_cd_world",
        source_manifest_sha256="a" * 64,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 2,
    )


def _prepared() -> PreparedTwoVisitExecution:
    unknown = SemanticLookup(semantic_id=0, native_name="Unknown", matched=False)
    chair = SemanticLookup(semantic_id=1, native_name="Chair", matched=True)
    crosswalk = TesseSemanticCrosswalk(
        scene="apartment",
        unknown=unknown,
        aliases={"chair": chair},
        valid_semantic_ids=frozenset({1}),
        label_space_sha256="b" * 64,
        alias_config_sha256="c" * 64,
    )
    t0 = _visit(0, [5.025, 0.025, 1.025])
    t1 = _visit(1, [0.025, 0.025, 1.025])
    return PreparedTwoVisitExecution(
        t0=t0,
        t1=t1,
        visibility=SignedVisibilityGrid(
            voxel_size_m=0.05,
            voxel_keys=np.asarray([[100, 0, 20]], dtype=np.int64),
            statuses=("unobserved",),
            source_sha256="d" * 64,
        ),
        evaluation=TwoVisitEvaluationContext(
            event_id="apartment_event_04",
            frame_id=12,
            intervention_frame_id=8,
            current_semantic_voxels=np.asarray([[0, 0, 20, 1]], dtype=np.int64),
            changed_region_voxels=np.asarray(
                [[0, 0, 20], [20, 0, 20]], dtype=np.int64
            ),
            confirmed_free_voxels=np.asarray([[100, 0, 20]], dtype=np.int64),
            revealed_background_voxels=np.asarray([[20, 0, 20]], dtype=np.int64),
            current_target_t1_unobserved_mask=np.asarray([False], dtype=np.bool_),
        ),
        crosswalk=crosswalk,
        t0_entity_observed_masks={
            "ovimap:1": np.asarray([False], dtype=np.bool_)
        },
        t0_background_observed_mask=np.asarray([True], dtype=np.bool_),
        ovi_t0_artifact_bytes=100,
        ovi_t1_artifact_bytes=120,
        input_bindings={"two_visit_ovi_manifest": {"sha256": "e" * 64}},
        evaluation_bindings={"target_manifest": {"sha256": "f" * 64}},
    )


def test_executes_b2_with_real_metric_receipt_and_bound_snapshot(tmp_path: Path) -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    row = matrix["rows"][2]
    run_id = "fixture-apartment-B2"
    variant_root = tmp_path / "B2"
    variant_root.mkdir()

    artifacts = execute_prepared_variant(
        _prepared(),
        matrix,
        row,
        run_id,
        variant_root,
        command_line=("execute-two-visit", "--variant", "B2"),
    )

    assert artifacts.output_artifact == variant_root / "current-map-manifest.json"
    assert artifacts.metric_receipt == variant_root / "metrics.json"
    output = json.loads(artifacts.output_artifact.read_text(encoding="utf-8"))
    assert output["status"] == "PASS"
    assert output["variant_id"] == "B2"
    assert output["geometry_sources"] == ["ovi_t1"]
    assert output["artifacts"]["snapshot"]["sha256"]
    metrics = json.loads(artifacts.metric_receipt.read_text(encoding="utf-8"))
    assert metrics["metric_groups"]["current_state"]["current_miou"] == 1.0
    assert metrics["metric_groups"]["current_state"]["ghost"] == 0.0
    assert metrics["metric_groups"]["geometry"]["surface_f1_at_5cm"] == 1.0
    assert metrics["metric_groups"]["systems"]["final_map_point_count"] == 2
    assert metrics["metric_groups"]["current_state"]["object_f1"] is None
    assert metrics["unavailable"]["object_f1"]
    assert set(metrics["method_input_bindings"]) == {"two_visit_ovi_manifest"}
    assert set(metrics["evaluation_only_bindings"]) == {"target_manifest"}


def test_orchestrator_accepts_canonical_actual_metric_receipts(tmp_path: Path) -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    prepared = _prepared()

    def execute(row, run_id: str, variant_root: Path):
        return execute_prepared_variant(
            prepared,
            matrix,
            row,
            run_id,
            variant_root,
            command_line=("fixture", row["id"]),
        )

    summary = execute_required_matrix(
        matrix_path=MATRIX_PATH,
        scene="apartment",
        output_root=tmp_path / "matrix",
        source_commit=_head(),
        executor=execute,
    )

    assert summary.is_file()


def test_b3_retained_t0_provenance_is_used_for_stale_precision(tmp_path: Path) -> None:
    matrix = load_and_validate_matrix(MATRIX_PATH)
    row = matrix["rows"][3]
    variant_root = tmp_path / "B3"
    variant_root.mkdir()

    artifacts = execute_prepared_variant(
        _prepared(),
        matrix,
        row,
        "fixture-apartment-B3",
        variant_root,
        command_line=("execute-two-visit", "--variant", "B3"),
    )

    metrics = json.loads(artifacts.metric_receipt.read_text(encoding="utf-8"))
    assert (
        metrics["metric_groups"]["geometry"]
        ["t1_observed_region_stale_precision"]
        == 1.0
    )
    output = json.loads(artifacts.output_artifact.read_text(encoding="utf-8"))
    assert output["artifacts"]["provenance_manifest"]["sha256"]


def test_loads_and_revalidates_two_visit_ovi_input_manifest(tmp_path: Path) -> None:
    protocol = json.loads(
        (REPO_ROOT / "configs/evaluation/tesse_two_visit_current_v1.json").read_text(
            encoding="utf-8"
        )
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    method_input = tmp_path / "method-input.json"
    method_input.write_text(
        json.dumps(protocol["scenes"]["apartment"]["method_input_manifest"]),
        encoding="utf-8",
    )
    visits: dict[str, dict[str, object]] = {}
    for visit_id in ("t0", "t1"):
        native = tmp_path / f"{visit_id}-native.json"
        materialized = tmp_path / f"{visit_id}-materialized.json"
        native.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        materialized.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        visit = protocol["scenes"]["apartment"]["method_input_manifest"]["visits"][
            visit_id
        ]
        visits[visit_id] = {
            "run_id": f"run-{visit_id}",
            "source_frame_interval": [visit["start_frame"], visit["end_frame"]],
            "source_input_sha256": visit["input_sha256"],
            "materialized_manifest": _record(materialized),
            "native_manifest": _record(native),
        }
    manifest = tmp_path / "two-visit.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "TWO_VISIT_OVI_PASS",
                "protocol_id": "TESSE_TWO_VISIT_CURRENT_V1",
                "protocol_content_sha256": protocol_content_sha256(protocol),
                "scene": "apartment",
                "protocol": _record(protocol_path),
                "method_input_manifest": _record(method_input),
                "visits": visits,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_two_visit_ovi_inputs(
        protocol_path=protocol_path,
        two_visit_ovi_manifest=manifest,
        scene="apartment",
    )

    assert loaded["visits"]["t0"]["native_manifest_path"].name == "t0-native.json"
    assert loaded["visits"]["t1"]["materialized_manifest_path"].name == (
        "t1-materialized.json"
    )

    (tmp_path / "t1-native.json").write_text("changed", encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "binding mismatch"):
        load_two_visit_ovi_inputs(
            protocol_path=protocol_path,
            two_visit_ovi_manifest=manifest,
            scene="apartment",
        )
