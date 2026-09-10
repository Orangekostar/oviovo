from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.run_crove_entity_epoch import (
    ORDERED_VARIANTS,
    REPO_ROOT,
    _portableize_paths,
    _read_metric_rows,
    adapt_relations_to_backbone,
    cache_identity,
    load_backbone_visits,
    load_backbone_visits_from_surface,
    parse_args,
    run,
    run_requested_variants,
    select_development_pairs,
    select_global_development_config,
)
from src.oviv2.current_surface import (
    CurrentEvidenceState,
    CurrentSurfaceView,
    SemanticSource,
    export_current_surface,
)
from src.oviv2.entity_epoch_update import RelationInferenceState, RelationSupport


def _surface() -> CurrentSurfaceView:
    return CurrentSurfaceView(
        surface_id="backbone",
        vertices_xyz=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
            dtype=np.float32,
        ),
        normals_xyz=np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (6, 1)),
        triangles=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        source_surface_indices=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint16),
        source_vertex_indices=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        source_visit_ids=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int16),
        geometry_epochs=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        observed_rgb_uint8=np.zeros((6, 3), dtype=np.uint8),
        rgb_valid=np.ones(6, dtype=np.bool_),
        current_valid=np.asarray([True, False, True, True, True, True]),
        evidence_state_codes=np.asarray(
            [
                CurrentEvidenceState.HISTORICAL_UNOBSERVED,
                CurrentEvidenceState.REVOKED_VISIBLE_FREE,
                CurrentEvidenceState.HISTORICAL_OCCLUDED,
                CurrentEvidenceState.CURRENT_OBSERVED,
                CurrentEvidenceState.CURRENT_OBSERVED,
                CurrentEvidenceState.CURRENT_OBSERVED,
            ],
            dtype=np.uint8,
        ),
        last_supported_frames=np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32),
        owner_entity_ids=np.asarray([7, 7, 0, 1_000_009, 1_000_009, 0], dtype=np.int64),
        owner_confidences=np.ones(6, dtype=np.float32),
        semantic_ids=np.asarray([1, 1, 0, 2, 2, 0], dtype=np.int32),
        semantic_confidences=np.ones(6, dtype=np.float32),
        semantic_support_reliabilities=np.ones(6, dtype=np.float32),
        semantic_source_codes=np.full(6, SemanticSource.OWNER_ENTITY, dtype=np.uint8),
    )


def test_cli_accepts_three_phases_and_preserves_requested_variant_order() -> None:
    args = parse_args(
        [
            "--config",
            "config.json",
            "--phase",
            "run",
            "--split",
            "dev",
            "--variants",
            "D3_GEOM",
            "D1_B3",
        ]
    )

    assert args.phase == "run"
    assert args.split == "dev"
    assert args.variants == ["D3_GEOM", "D1_B3"]
    for phase in ("prepare", "run", "summarize"):
        assert (
            parse_args(
                ["--config", "config.json", "--phase", phase, "--split", "dev"]
            ).phase
            == phase
        )


def test_cli_requires_frozen_selection_for_confirmation() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--config",
                "config.json",
                "--phase",
                "run",
                "--split",
                "confirm",
            ]
        )

    args = parse_args(
        [
            "--config",
            "config.json",
            "--phase",
            "run",
            "--split",
            "confirm",
            "--selection",
            "selected.json",
        ]
    )
    assert args.selection == Path("selected.json")


def test_cache_identity_binds_pair_configuration_and_checkpoint() -> None:
    first = cache_identity(
        pair_sha256="a" * 64,
        method_config={"minimum_margin": 0.08, "mode": "strict"},
        checkpoint_sha256="b" * 64,
    )
    reordered = cache_identity(
        pair_sha256="a" * 64,
        method_config={"mode": "strict", "minimum_margin": 0.08},
        checkpoint_sha256="b" * 64,
    )
    changed = cache_identity(
        pair_sha256="a" * 64,
        method_config={"mode": "strict", "minimum_margin": 0.09},
        checkpoint_sha256="b" * 64,
    )

    assert first == reordered
    assert first != changed


def test_public_result_paths_hide_local_identity_and_keep_repo_paths_relative() -> None:
    value = {
        "checkpoint": str(Path.home() / "private" / "checkpoint.ckpt"),
        "manifest": str(REPO_ROOT / "configs" / "manifest.json"),
        "nested": [str(Path.home() / "runs" / "result.json")],
    }

    assert _portableize_paths(value) == {
        "checkpoint": "$HOME/private/checkpoint.ckpt",
        "manifest": "configs/manifest.json",
        "nested": ["$HOME/runs/result.json"],
    }


def test_run_isolates_variant_failure_and_preserves_order(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(variant: str, output: Path) -> dict[str, object]:
        calls.append(variant)
        if variant == "D3_GEOM":
            raise RuntimeError("registration failed")
        output.mkdir(parents=True)
        (output / "metrics.json").write_text(
            json.dumps({"status": "PASS", "current_miou": 0.2}),
            encoding="utf-8",
        )
        return {"current_miou": 0.2}

    records = run_requested_variants(
        variants=("D1_B3", "D3_GEOM", "D2_INHERIT"),
        pair_id="apartment-primary",
        run_root=tmp_path,
        execute=execute,
    )

    assert calls == ["D1_B3", "D3_GEOM", "D2_INHERIT"]
    assert [record["status"] for record in records] == ["PASS", "FAILED", "PASS"]
    assert (tmp_path / "apartment-primary" / "D3_GEOM" / "failure.json").is_file()


def test_confirmation_run_records_missing_assets_without_executing_variants(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    selection_path = run_root / "confirm" / "input_selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "selected_pair_ids": ["office-pair"],
                "pairs": [
                    {
                        "pair_id": "office-pair",
                        "asset_status": "RAW_MISSING",
                        "missing_paths": ["/missing/office.db3"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "run_root": str(run_root),
        "compact_output_root": str(tmp_path / "compact"),
        "splits": {"confirm": {"pairs": [{"pair_id": "office-pair"}]}},
        "hypotheses": [],
    }

    records = run(config, split="confirm", variants=None)

    assert records == []
    run_index = json.loads(
        (run_root / "confirm" / "run_index.json").read_text(encoding="utf-8")
    )
    assert run_index["status"] == "RAW_MISSING"
    assert run_index["runs"] == []
    assert run_index["asset_attempts"] == [
        {
            "pair_id": "office-pair",
            "asset_status": "RAW_MISSING",
            "missing_paths": ["/missing/office.db3"],
        }
    ]


def test_metric_reader_excludes_archived_runs_and_normalizes_legacy_d0_receipt(
    tmp_path: Path,
) -> None:
    hypothesis = tmp_path / "pair" / "hypotheses" / "H0" / "D0_T1"
    hypothesis.mkdir(parents=True)
    (hypothesis / "metrics.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "config_id": "H0",
                "variant_id": "D0_T1",
                "deleted_supported_rate": 0.0,
                "retained_history_recall": 1.0,
            }
        ),
        encoding="utf-8",
    )
    archived = tmp_path / "pair" / "hypotheses.invalid" / "H0" / "D0_T1"
    archived.mkdir(parents=True)
    (archived / "metrics.json").write_text(
        json.dumps({"status": "PASS", "variant_id": "D0_T1"}),
        encoding="utf-8",
    )

    rows = _read_metric_rows(tmp_path, ("pair",))

    assert len(rows) == 1
    assert rows[0]["raw_deleted_supported_rate"] == 0.0
    assert rows[0]["raw_retained_history_value"] == 1.0
    assert rows[0]["deleted_supported_rate"] == 1.0
    assert rows[0]["retained_history_recall"] == 0.0


def test_pair_selection_uses_only_frozen_metadata_and_never_replaces_failure() -> None:
    candidates = [
        {
            "pair_id": "fixed",
            "fixed": True,
            "asset_status": "READY",
            "event_type": "object_relocation",
            "visibility_coverage": 0.1,
            "method_score": 0.0,
        },
        {
            "pair_id": "visible",
            "fixed": False,
            "asset_status": "READY",
            "event_type": "object_removal",
            "visibility_coverage": 0.8,
            "method_score": 0.0,
        },
        {
            "pair_id": "high-score",
            "fixed": False,
            "asset_status": "READY",
            "event_type": "object_removal",
            "visibility_coverage": 0.2,
            "method_score": 1.0,
        },
    ]

    selected = select_development_pairs(candidates, maximum_additional_pairs=1)
    candidates[1]["method_score"] = -100.0
    candidates[2]["method_score"] = 100.0

    assert selected == ("fixed", "visible")
    assert select_development_pairs(candidates, maximum_additional_pairs=1) == selected


def test_global_selection_applies_all_pair_gates_then_documented_tiebreaks() -> None:
    rows = [
        {
            "pair_id": pair,
            "config_id": config,
            "variant_id": variant,
            "status": "PASS",
            "current_miou": miou,
            "ghost": ghost,
            "background_f1_at_5cm": bg,
            "surface_precision_at_5cm": precision,
            "deleted_supported_rate": deleted,
            "retained_history_recall": retained,
        }
        for pair, config, variant, miou, ghost, bg, precision, deleted, retained in (
            ("p0", "safe", "D5_RESCENE_EPOCH", 0.30, 0.01, 0.40, 0.75, 0.01, 0.70),
            ("p1", "safe", "D5_RESCENE_EPOCH", 0.20, 0.01, 0.50, 0.76, 0.01, 0.80),
            ("p0", "unsafe", "D6_MEMORY_EPOCH", 0.40, 0.03, 0.60, 0.80, 0.00, 0.90),
            ("p1", "unsafe", "D6_MEMORY_EPOCH", 0.40, 0.01, 0.60, 0.80, 0.00, 0.90),
        )
    ]

    selected, aggregates = select_global_development_config(
        rows,
        expected_pair_ids=("p0", "p1"),
        gates={
            "maximum_ghost": 0.02,
            "minimum_surface_precision_at_5cm": 0.72,
            "maximum_deleted_supported_rate": 0.02,
        },
    )

    assert selected == {"config_id": "safe", "variant_id": "D5_RESCENE_EPOCH"}
    assert {row["config_id"]: row["eligible"] for row in aggregates} == {
        "safe": True,
        "unsafe": False,
    }


def test_published_variant_order_is_the_frozen_matrix() -> None:
    assert ORDERED_VARIANTS == (
        "D0_T1",
        "D1_B3",
        "D2_INHERIT",
        "D3_GEOM",
        "D4_RESCENE_ID_ONLY",
        "D5_RESCENE_EPOCH",
        "D6_MEMORY_EPOCH",
        "DX_ORACLE_REL",
    )


def test_backbone_loader_verifies_export_and_splits_native_triangles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "current_map"
    export_current_surface(
        _surface(),
        root,
        semantic_palette={0: (0, 0, 0), 1: (1, 2, 3), 2: (4, 5, 6)},
        owner_id_table={0: "background", 7: "t0", 1_000_009: "t1"},
        source_surfaces=(
            {"surface_index": 0, "surface_id": "t0"},
            {"surface_index": 1, "surface_id": "t1"},
        ),
    )

    t0, t1, baseline_mask = load_backbone_visits(root)

    assert t0.source_vertex_indices.tolist() == [0, 1, 2]
    assert t1.source_vertex_indices.tolist() == [0, 1, 2]
    assert t0.triangles.tolist() == [[0, 1, 2]]
    assert t1.triangles.tolist() == [[0, 1, 2]]
    assert baseline_mask.tolist() == [True, False, True]

    arrays = root / "current_surface.npz"
    content = bytearray(arrays.read_bytes())
    content[-1] ^= 1
    arrays.write_bytes(content)
    with pytest.raises(ValueError, match="binding"):
        load_backbone_visits(root)


def test_relation_support_is_rebound_to_exact_fine_ovi_rows_and_t1_offset() -> None:
    t0, t1, _ = load_backbone_visits_from_surface(_surface())
    relation = RelationSupport(
        relation_id="r0",
        relation_source="frozen-rescene",
        stable_entity_id="stable:r0",
        t0_owner_entity_id=7,
        t1_owner_entity_id=9,
        t0_source_surface_id="coarse-t0",
        t1_source_surface_id="coarse-t1",
        t0_source_vertex_indices=np.asarray([0], dtype=np.int64),
        t1_source_vertex_indices=np.asarray([0], dtype=np.int64),
        confidence=0.9,
        inference_state=RelationInferenceState.UNRESOLVED,
        accepted=True,
    )

    rebound = adapt_relations_to_backbone(
        (relation,),
        t0=t0,
        t1=t1,
        t0_surface_id="fine-t0",
        t1_surface_id="fine-t1",
        t1_owner_offset=1_000_000,
    )

    assert rebound[0].t0_source_surface_id == "fine-t0"
    assert rebound[0].t1_source_surface_id == "fine-t1"
    assert rebound[0].t0_source_vertex_indices.tolist() == [0, 1]
    assert rebound[0].t1_source_vertex_indices.tolist() == [0, 1]
    assert rebound[0].t1_owner_entity_id == 1_000_009
