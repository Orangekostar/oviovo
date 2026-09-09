from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluation import summarize_crove_fine_current_map as summarizer
from scripts.evaluation.summarize_crove_fine_current_map import (
    _merge_csv,
    _portable_payload,
)


def test_portable_payload_removes_machine_specific_home(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = _portable_payload(
        {"path": str(tmp_path / "runs" / "map.ply"), "value": 0.5}
    )

    assert result == {"path": "$HOME/runs/map.ply", "value": 0.5}


def test_portable_payload_sanitizes_path_roots_but_preserves_json_pointers(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = _portable_payload(
        {
            "dataset_root": str(tmp_path / "datasets" / "replica"),
            "json_pointer": "/trials/S2/miou",
        }
    )

    assert result == {
        "dataset_root": "$HOME/datasets/replica",
        "json_pointer": "/trials/S2/miou",
    }


def test_merge_csv_keeps_union_of_trial_columns(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dynamic.csv"
    static.write_text("trial_id,miou\nS0,0.4\n", encoding="utf-8")
    dynamic.write_text("trial_id,ghost\nB3,0.0\n", encoding="utf-8")
    output = tmp_path / "merged.csv"

    _merge_csv(output, (("static", static), ("dynamic", dynamic)))

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"evaluation": "static", "trial_id": "S0", "miou": "0.4", "ghost": ""},
        {"evaluation": "dynamic", "trial_id": "B3", "miou": "", "ghost": "0.0"},
    ]


def test_cost_control_payload_requires_real_source_selection_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "condition": "SOURCE_SURFACE_VOXEL_SELECTION_NOT_RECONSTRUCTION",
                "source_binding": {"path": str(tmp_path / "surface.npz")},
            }
        ),
        encoding="utf-8",
    )

    result = summarizer._cost_control_payload(path)

    assert result["source_binding"]["path"] == "$HOME/surface.npz"
    path.write_text(
        json.dumps(
            {
                "status": "NOT_RUN",
                "condition": "SOURCE_SURFACE_VOXEL_SELECTION_NOT_RECONSTRUCTION",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not PASS"):
        summarizer._cost_control_payload(path)


def test_legacy_figure_requires_display_only_receipt(tmp_path: Path) -> None:
    image = tmp_path / "legacy.png"
    image.write_bytes(b"png")
    receipt = image.with_suffix(".json")
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "display_only": True,
                "geometry_shared_across_views": True,
            }
        ),
        encoding="utf-8",
    )

    assert summarizer._legacy_figure(image) == receipt
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "display_only": False,
                "geometry_shared_across_views": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="display-only"):
        summarizer._legacy_figure(image)
