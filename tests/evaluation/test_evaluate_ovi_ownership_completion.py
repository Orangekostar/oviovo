from __future__ import annotations

import numpy as np

from scripts.evaluation.evaluate_ovi_ownership_completion import (
    assemble_ownership_completion_report,
    write_instance_readout_csv,
)
from src.evaluation.rscan_method_views import build_method_pair_view
from src.oviv2.two_visit_contracts import PairRelation


def _pair():
    def visit(offset: float) -> np.ndarray:
        return np.asarray(
            [
                [offset, 0.0, 0.0, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 4, 1, 10],
                [offset + 1.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 8, 1, 20],
            ],
            dtype=np.float32,
        )

    return build_method_pair_view(
        pair_id="pair",
        scan_ids=("a", "b"),
        processed_visits=(visit(0.0), visit(0.01)),
        source_manifest_sha256="a" * 64,
        domain_id="D0_NATIVE_PROCESSED",
    ).geometric_sample(neural_voxel_size_m=0.02)


def _relation(source: str) -> PairRelation:
    return PairRelation(
        temporal_query_id=f"{source}:0",
        t0_entity_ids=("segment:000004",),
        t1_entity_ids=("segment:000004",),
        state="persistent_static",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source=source,
    )


def _unsupported_relation() -> PairRelation:
    return PairRelation(
        temporal_query_id="geom:unsupported",
        t0_entity_ids=("missing",),
        t1_entity_ids=(),
        state="removed_candidate",
        query_confidence=1.0,
        evidence={"query_score": 1.0},
        identity_source="geometric_baseline",
    )


def test_report_preserves_geometry_and_marks_unavailable_completion_rows() -> None:
    map_metrics = {
        "current_miou": 0.13,
        "ghost": 0.0,
        "surface_f1_at_5cm": 0.43,
    }
    report = assemble_ownership_completion_report(
        pair=_pair(),
        geometric_relations=(
            _relation("geometric_baseline"),
            _unsupported_relation(),
        ),
        rescene_relations=(_relation("rescene"),),
        b3_payload={
            "schema_version": 1,
            "status": "PASS",
            "variant_id": "B3",
            "metric_groups": {
                "current_state": {"current_miou": 0.13, "ghost": 0.0},
                "geometry": {"surface_f1_at_5cm": 0.43},
            },
        },
        b7_payload={
            "schema_version": 1,
            "status": "PASS",
            "variant_id": "B7-G",
            "metrics": map_metrics,
            "b3_reference": {**map_metrics, "final_map_point_count": 10},
            "diagnostics": {
                "recovered_historical_point_count": 4,
                "registration_attempt_count": 2,
                "registration_accepted_count": 1,
            },
            "attribution": {
                "summary": {
                    "candidate_point_count": 11,
                    "recovered_point_count": 4,
                    "newly_covered_gt_surface_count": 0,
                }
            },
        },
        rscan_matrix_payload={
            "schema_version": 1,
            "artifact_id": "RSCAN_T2_METHOD_MATRIX_AGGREGATE_V1",
            "status": "PASS",
            "methods": {"R_supported": {"status": "PASS"}},
        },
        source_bindings={"fixture": {"path": "/fixture", "sha256": "f" * 64, "byte_count": 1}},
    )

    rows = {row["variant_id"]: row for row in report["ownership"]}
    assert set(rows) == {"A0", "A1", "A2"}
    assert {row["geometry_sha256"] for row in rows.values()} == {
        rows["A0"]["geometry_sha256"]
    }
    assert all(row["geometry_mutation_count"] == 0 for row in rows.values())
    assert all(row["instance_ap"] is None for row in rows.values())
    assert all(row["panoptic_quality"] is None for row in rows.values())
    assert rows["A1"]["source_relation_count"] == 2
    assert rows["A1"]["relation_count"] == 1
    assert rows["A1"]["excluded_out_of_domain_relation_ids"] == [
        "geom:unsupported"
    ]

    completion = {row["variant_id"]: row for row in report["completion"]}
    assert completion["C0"]["status"] == "PASS_MEASURED_B3"
    assert completion["C0"]["metrics"] == map_metrics
    assert completion["C1"]["status"] == "PASS_MEASURED_PRIOR_B7_G"
    assert completion["C1"]["metrics"] == map_metrics
    assert completion["C1"]["completion_surface"] is None
    assert completion["C2"]["status"] == "NOT_COMPUTED_MISSING_R_BOUND_RECOVERY_ARTIFACT"
    assert completion["O1"]["oracle_only"] is True
    assert completion["O2"]["oracle_only"] is True
    assert report["status"] == "PASS_WITH_UNAVAILABLE_ROWS"
    assert report["transfer_boundary"]["D2_OVI_RECONSTRUCTION"] == "MISSING_ASSET"


def test_report_rejects_b7_metric_drift_from_its_b3_reference() -> None:
    pair = _pair()
    relation = (_relation("geometric_baseline"),)
    try:
        assemble_ownership_completion_report(
            pair=pair,
            geometric_relations=relation,
            rescene_relations=(_relation("rescene"),),
            b3_payload={
                "schema_version": 1,
                "status": "PASS",
                "variant_id": "B3",
                "metric_groups": {
                    "current_state": {"current_miou": 0.2},
                    "geometry": {},
                },
            },
            b7_payload={
                "schema_version": 1,
                "status": "PASS",
                "variant_id": "B7-G",
                "metrics": {"current_miou": 0.1},
                "b3_reference": {"current_miou": 0.1},
                "diagnostics": {},
                "attribution": {"summary": {}},
            },
            rscan_matrix_payload={
                "schema_version": 1,
                "artifact_id": "RSCAN_T2_METHOD_MATRIX_AGGREGATE_V1",
                "status": "PASS",
                "methods": {"R_supported": {"status": "PASS"}},
            },
            source_bindings={},
        )
    except ValueError as error:
        assert "B3 reference" in str(error)
    else:
        raise AssertionError("B7/B3 metric drift must fail closed")


def test_writes_dense_instance_rows_as_lf_csv(tmp_path) -> None:
    rows = (
        {
            "pair_id": "pair",
            "variant_id": "U0",
            "iou_threshold": 0.5,
            "precision": None,
            "geometry_unchanged": True,
        },
    )

    output = write_instance_readout_csv(tmp_path / "instance_readout.csv", rows)

    assert b"\r\n" not in output.read_bytes()
    assert output.read_text(encoding="utf-8").splitlines() == [
        "pair_id,variant_id,iou_threshold,precision,geometry_unchanged",
        "pair,U0,0.5,,True",
    ]
