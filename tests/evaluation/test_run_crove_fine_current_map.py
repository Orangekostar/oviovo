from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.evaluation.run_crove_fine_current_map import (
    _configured_path,
    _entity_records,
    _static_table_metric,
    _visit_owner_attributes,
    load_ovimap_semantic_mapping,
    select_dynamic_trial,
    select_static_semantic_strategy,
)
from src.oviv2.surface_semantics import SurfaceSemanticStrategy


def test_ovimap_semantic_mapping_translates_source_ids_by_class_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pred_inst_sem_mapping.txt"
    path.write_text(
        "inst-7_label-2.npy 2 0.750000\n"
        "inst-9_label-3.npy 3 0.500000\n",
        encoding="utf-8",
    )

    result = load_ovimap_semantic_mapping(
        path,
        source_vocabulary=("source-only", "chair", "lamp"),
        target_vocabulary=("lamp", "chair"),
    )

    assert result == {7: (2, 0.75), 9: (1, 0.5)}


def test_static_selection_uses_miou_then_fmiou_then_simpler_strategy() -> None:
    metrics = {
        SurfaceSemanticStrategy.S0: {"miou": 0.4, "f_miou": 0.7},
        SurfaceSemanticStrategy.S1: {"miou": 0.45, "f_miou": 0.65},
        SurfaceSemanticStrategy.S2: {"miou": 0.45, "f_miou": 0.66},
    }

    assert select_static_semantic_strategy(metrics) is SurfaceSemanticStrategy.S2

    metrics[SurfaceSemanticStrategy.S1]["f_miou"] = 0.66
    assert select_static_semantic_strategy(metrics) is SurfaceSemanticStrategy.S1


def test_dynamic_selection_applies_non_degradation_gates_before_miou() -> None:
    metrics = {
        "B3_LEGACY": {
            "current_miou": 0.13,
            "ghost": 0.0,
            "background_f1_at_5cm": 0.36,
            "surface_precision_at_5cm": 0.73,
            "t1_observed_region_stale_precision": 0.93,
        },
        "F1_UNSAFE": {
            "current_miou": 0.21,
            "ghost": 0.04,
            "background_f1_at_5cm": 0.38,
            "surface_precision_at_5cm": 0.76,
            "t1_observed_region_stale_precision": 0.94,
        },
        "F2_SAFE": {
            "current_miou": 0.18,
            "ghost": 0.01,
            "background_f1_at_5cm": 0.37,
            "surface_precision_at_5cm": 0.74,
            "t1_observed_region_stale_precision": 0.935,
        },
    }

    selected, eligibility = select_dynamic_trial(
        metrics,
        baseline_trial_id="B3_LEGACY",
        gates={
            "maximum_ghost": 0.02,
            "minimum_background_f1_at_5cm": 0.35,
            "minimum_surface_precision_at_5cm": 0.72,
            "minimum_observed_stale_precision": 0.925,
        },
    )

    assert selected == "F2_SAFE"
    assert eligibility == {
        "B3_LEGACY": True,
        "F1_UNSAFE": False,
        "F2_SAFE": True,
    }


def test_entity_records_preserve_native_unknown_semantics(tmp_path: Path) -> None:
    path = tmp_path / "entities.jsonl"
    path.write_text(
        '{"entity_id":"ovimap:7","first_seen":11.0,"last_seen":12.0,'
        '"semantic_label":null,"semantic_score":1.0,'
        '"metadata":{"source_instance_id":7}}\n',
        encoding="utf-8",
    )

    records = _entity_records(path)

    assert records[7]["semantic_label"] is None


def test_visit_owner_attributes_assign_explicit_semantic_sources() -> None:
    unknown = SimpleNamespace(matched=False, semantic_id=0)
    crosswalk = SimpleNamespace(unknown=unknown, lookup=lambda _label: unknown)

    attributes = _visit_owner_attributes(
        np.asarray([0, 7], dtype=np.int64),
        {
            7: {
                "entity_id": "ovimap:7",
                "semantic_label": None,
                "semantic_score": 1.0,
                "first_seen": 11.0,
                "last_seen": 12.0,
            }
        },
        crosswalk,
        owner_offset=0,
        default_last_supported_frame=12,
        output_prefix="",
    )

    assert attributes[2].tolist() == [0, 0]


def test_configured_path_expands_portable_home_reference(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _configured_path("$HOME/assets/map.ply") == tmp_path / "assets/map.ply"


def test_static_table_metric_preserves_compound_metric_name() -> None:
    assert _static_table_metric("CROVE_FINE_ROOM0_DEV_F_MIOU") == "f_miou"
    assert _static_table_metric("CROVE_FINE_ROOM0_DEV_AP50") == "ap50"
