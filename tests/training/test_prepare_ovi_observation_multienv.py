from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.training.prepare_ovi_observation_multienv import (
    MultiEnvironmentAssetPreparationError,
    build_training_selection_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _sources(tmp_path: Path) -> dict[str, Path]:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    scan_ids = (
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    )
    aliases = ("scene0001_00", "scene0001_01")
    points: list[Path] = []
    instance_gt: list[Path] = []
    for alias in aliases:
        point_path = processed_root / "train" / f"{alias.removeprefix('scene')}.npy"
        label_path = processed_root / "instance_gt" / "train" / f"{alias}.txt"
        point_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        point_path.write_bytes(alias.encode("ascii"))
        label_path.write_text("1\n", encoding="ascii")
        points.append(point_path)
        instance_gt.append(label_path)

    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "artifact_id": "OVI_RESCENE_OBSERVATION_QUERY_SPLITS_MULTIENV_V3",
                "environments": [
                    {
                        "role": "TRAIN",
                        "official_split": "train",
                        "environment_uuid": scan_ids[0],
                        "pair_id": "scene0001_00-scene0001_01",
                        "sessions": [
                            {
                                "visit_id": visit_id,
                                "scene_alias": aliases[visit_id],
                                "scan_uuid": scan_ids[visit_id],
                                "processed_points": _record(points[visit_id]),
                            }
                            for visit_id in (0, 1)
                        ],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    metadata = tmp_path / "3RScan.json"
    transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.2, 0.3, 1.0]
    metadata.write_text(
        json.dumps(
            [
                {
                    "reference": scan_ids[0],
                    "type": "train",
                    "ambiguity": [],
                    "scans": [
                        {
                            "reference": scan_ids[1],
                            "transform": transform,
                            "rigid": [
                                {
                                    "instance_reference": 7,
                                    "instance_rescan": 8,
                                    "transform": transform,
                                }
                            ],
                            "nonrigid": [9],
                            "removed": [10],
                        }
                    ],
                }
            ],
            sort_keys=True,
        ),
        encoding="ascii",
    )
    database = tmp_path / "train_database.yaml"
    database.write_text(
        yaml.safe_dump(
            [
                {
                    "scene": 1,
                    "sub_scene": visit_id,
                    "filepath": str(points[visit_id].relative_to(processed_root)),
                    "instance_gt_filepath": str(
                        instance_gt[visit_id].relative_to(processed_root)
                    ),
                    "raw_filepath": str(raw_root / scan_ids[visit_id] / "mesh.obj"),
                }
                for visit_id in (0, 1)
            ],
            sort_keys=True,
        ),
        encoding="ascii",
    )
    change_gt = processed_root / "change_gt" / "train" / "scene0001_00-scene0001_01.txt"
    change_gt.parent.mkdir(parents=True, exist_ok=True)
    change_gt.write_text("7 8\n", encoding="ascii")
    sequence = tmp_path / "sequence.yaml"
    sequence.write_text(
        yaml.safe_dump(
            {
                "scene0001_00-scene0001_01": {
                    "type": "train",
                    "scene": 1,
                    "sub_scenes": [0, 1],
                    "filepath": str(change_gt.relative_to(processed_root)),
                }
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    return {
        "split": split,
        "metadata": metadata,
        "database": database,
        "sequence": sequence,
        "raw_root": raw_root,
        "processed_root": processed_root,
    }


def test_build_training_selection_manifest_binds_frozen_train_pair(
    tmp_path: Path,
) -> None:
    source = _sources(tmp_path)

    result = build_training_selection_manifest(
        split_manifest=source["split"],
        metadata_path=source["metadata"],
        processed_database_path=source["database"],
        sequence_database_path=source["sequence"],
        raw_root=source["raw_root"],
        processed_root=source["processed_root"],
        pair_ids=("scene0001_00-scene0001_01",),
    )

    assert result["status"] == "PASS"
    assert result["selected_pair_count"] == 1
    pair = result["pairs"][0]
    assert pair["environment_id"] == "00000000-0000-0000-0000-000000000001"
    assert [item["scan_id"] for item in pair["sessions"]] == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    assert pair["common_method_inputs"]["global_alignment"]["matrix"][-4:] == [
        0.1,
        0.2,
        0.3,
        1.0,
    ]
    assert pair["evaluator_only"]["changes"] == {
        "rigid_reference_ids": [7],
        "nonrigid_reference_ids": [9],
        "removed_reference_ids": [10],
    }
    assert pair["sequence"]["change_gt_binding"]["sha256"] == _record(
        source["processed_root"]
        / "change_gt"
        / "train"
        / "scene0001_00-scene0001_01.txt"
    )["sha256"]


def test_build_training_selection_manifest_rejects_nontrain_pair(
    tmp_path: Path,
) -> None:
    source = _sources(tmp_path)
    payload = json.loads(source["split"].read_text(encoding="ascii"))
    payload["environments"][0]["role"] = "DEV"
    source["split"].write_text(json.dumps(payload), encoding="ascii")

    try:
        build_training_selection_manifest(
            split_manifest=source["split"],
            metadata_path=source["metadata"],
            processed_database_path=source["database"],
            sequence_database_path=source["sequence"],
            raw_root=source["raw_root"],
            processed_root=source["processed_root"],
            pair_ids=("scene0001_00-scene0001_01",),
        )
    except MultiEnvironmentAssetPreparationError as error:
        assert "TRAIN" in str(error)
    else:
        raise AssertionError("non-TRAIN split pair was accepted")


def test_cli_can_run_as_a_direct_script() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts/training/prepare_ovi_observation_multienv.py"
            ),
            "--help",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
