from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.datasets.rscan import (
    build_selection_manifest,
    select_pilot,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/evaluation/manifests/3rscan_causal_pilot_v1.json"


def _uuid(index: int) -> str:
    return f"00000000-0000-0000-0000-{index:012d}"


def _metadata_fixture(tmp_path: Path) -> tuple[Path, Path]:
    metadata = tmp_path / "3RScan.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "reference": _uuid(20),
                    "type": "validation",
                    "ambiguity": [],
                    "scans": [
                        {
                            "reference": _uuid(21),
                            "rigid": [1],
                            "nonrigid": [],
                            "removed": [],
                            "transform": [1, 0, 0, 0] * 4,
                        }
                    ],
                },
                {
                    "reference": _uuid(10),
                    "type": "validation",
                    "ambiguity": [],
                    "scans": [
                        {
                            "reference": _uuid(11),
                            "rigid": [],
                            "nonrigid": [2],
                            "removed": [3],
                            "transform": [1, 0, 0, 0] * 4,
                        }
                    ],
                },
                {
                    "reference": _uuid(30),
                    "type": "test",
                    "ambiguity": [],
                    "scans": [
                        {"reference": _uuid(31), "rigid": [4]}
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    validation = tmp_path / "val_scans.txt"
    validation.write_text(
        "\n".join((_uuid(10), _uuid(11), _uuid(20), _uuid(21))) + "\n",
        encoding="utf-8",
    )
    return metadata, validation


def test_selection_is_result_independent_and_sorted(tmp_path: Path) -> None:
    metadata, validation = _metadata_fixture(tmp_path)

    selected = select_pilot(metadata, validation, count=2)

    assert len(selected) == 2
    assert [item.reference_id for item in selected] == [_uuid(10), _uuid(20)]
    assert [session.scan_id for session in selected[0].sessions] == [
        _uuid(10),
        _uuid(11),
    ]
    assert {change.change_type for change in selected[0].sessions[1].changes} == {
        "nonrigid",
        "removed",
    }


def test_manifest_binds_selection_and_blocks_missing_runtime_assets(
    tmp_path: Path,
) -> None:
    metadata, validation = _metadata_fixture(tmp_path)
    output = tmp_path / "selection.json"

    manifest = build_selection_manifest(
        metadata,
        validation,
        asset_root=tmp_path / "assets",
        count=2,
        output=output,
    )

    assert manifest["status"] == "BLOCKED_DATASET_ACCESS"
    assert manifest["metadata_status"] == "PASS"
    assert manifest["selected_environment_count"] == 2
    assert manifest["source_bindings"]["metadata"]["sha256"]
    assert output.exists()


def test_frozen_real_manifest_locks_visit_counts_and_selection() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["status"] == "BLOCKED_DATASET_ACCESS"
    assert manifest["environment_count"] == 478
    assert manifest["visit_statistics"]["exact"]["2"] == 194
    assert manifest["visit_statistics"]["at_least"] == {
        "2": 478,
        "3": 284,
        "4": 124,
        "5": 56,
        "6": 25,
    }
    selected = manifest["selected_environments"]
    assert manifest["eligible_environment_count"] == 47
    assert len(selected) == 10
    assert [item["reference_id"] for item in selected] == sorted(
        item["reference_id"] for item in selected
    )
    assert manifest["source_bindings"]["metadata"]["sha256"] == (
        "674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64"
    )
    selected_visit_count = sum(
        len(environment["session_ids"]) for environment in selected
    )
    missing = manifest["runtime_assets"]["missing"]
    assert selected_visit_count == 44
    assert len(missing) == 17
    assert all(record["members"] == ["sequence.zip"] for record in missing)
    assert selected_visit_count - len(missing) == 27
