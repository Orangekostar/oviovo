from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.rscan_t2_dev import (
    RAW_RUNTIME_MEMBERS,
    RScanT2DevelopmentError,
    audit_t2_development_manifest,
    build_t2_development_manifest,
)


def _uuid(value: int) -> str:
    return f"00000000-0000-0000-0000-{value:012d}"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _environment(
    reference: int,
    rescans: tuple[int, ...],
    *,
    split: str = "validation",
) -> dict[str, object]:
    return {
        "reference": _uuid(reference),
        "type": split,
        "ambiguity": [
            [
                {
                    "instance_source": 7,
                    "instance_target": 8,
                    "transform": [1, 0, 0, 0] * 4,
                }
            ]
        ],
        "scans": [
            {
                "reference": _uuid(rescan),
                "rigid": [
                    {
                        "instance_reference": 1,
                        "instance_rescan": 2,
                        "symmetry": 2,
                        "transform": [1, 0, 0, 0] * 4,
                    }
                ],
                "nonrigid": [3],
                "removed": [4],
                "transform": [1, 0, 0, 0] * 4,
            }
            for rescan in rescans
        ],
    }


def _materialize_scan(root: Path, scan_id: str) -> None:
    scan_root = root / scan_id
    for index, member in enumerate(RAW_RUNTIME_MEMBERS):
        _write(scan_root / member, f"{scan_id}:{member}:{index}\n")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    raw_root = tmp_path / "raw" / "scans"
    processed_root = tmp_path / "processed"
    environments = [
        _environment(40, (41,)),
        _environment(10, (11,)),
        _environment(30, (31,)),
        _environment(20, (21,)),
        _environment(50, (51, 52)),
        _environment(60, (61,), split="train"),
        _environment(70, (71,)),
    ]
    metadata = tmp_path / "3RScan.json"
    metadata.write_text(json.dumps(environments), encoding="utf-8")
    validation = tmp_path / "val_scans.txt"
    validation.write_text(
        "\n".join(
            _uuid(value)
            for value in (10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 52, 70, 71)
        )
        + "\n",
        encoding="utf-8",
    )

    validation_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    sequence_rows: dict[str, dict[str, object]] = {}
    scene_by_reference = {10: 10, 20: 20, 30: 30, 40: 40, 50: 50, 60: 60, 70: 70}
    for environment in environments:
        ids = [environment["reference"]] + [
            scan["reference"] for scan in environment["scans"]
        ]
        if environment["reference"] != _uuid(70):
            for scan_id in ids:
                _materialize_scan(raw_root, scan_id)
        scene = scene_by_reference[int(str(environment["reference"])[-12:])]
        destination = train_rows if environment["type"] == "train" else validation_rows
        for sub_scene, scan_id in enumerate(ids):
            prefix = "train" if environment["type"] == "train" else "validation"
            point_path = Path(prefix) / f"{scene:04d}_{sub_scene:02d}.npy"
            instance_path = Path("instance_gt") / prefix / f"scene{scene:04d}_{sub_scene:02d}.txt"
            _write(processed_root / point_path, f"points:{scan_id}\n")
            _write(processed_root / instance_path, f"instances:{scan_id}\n")
            destination.append(
                {
                    "scene": scene,
                    "sub_scene": sub_scene,
                    "filepath": str(point_path),
                    "instance_gt_filepath": str(instance_path),
                    "raw_filepath": str(raw_root / scan_id / "mesh.refined.v2.obj"),
                    "raw_instance_filepath": str(raw_root / scan_id / "semseg.v2.json"),
                    "raw_label_filepath": str(
                        raw_root / scan_id / "labels.instances.annotated.v2.ply"
                    ),
                    "raw_segmentation_filepath": str(
                        raw_root / scan_id / "mesh.refined.0.010000.segs.v2.json"
                    ),
                }
            )
        if len(ids) == 2:
            key = f"scene{scene:04d}_00-scene{scene:04d}_01"
            change_path = Path("change_gt") / "validation" / f"{key}.txt"
            _write(processed_root / change_path, f"change:{key}\n")
            sequence_rows[key] = {
                "scene": scene,
                "sub_scenes": [0, 1],
                "type": "validation",
                "filepath": str(change_path),
                "rigid": [[1]],
                "nonrigid": [[3]],
                "removed": [[4]],
                "added": [[]],
                "ambiguities": [[7, 8]],
            }

    paths = {
        "metadata": metadata,
        "validation_list": validation,
        "raw_root": raw_root,
        "processed_root": processed_root,
        "validation_database": tmp_path / "validation_database.yaml",
        "train_database": tmp_path / "train_database.yaml",
        "sequence_database": tmp_path / "sequence_database_sliding_2.yaml",
        "checkpoint": tmp_path / "model.ckpt",
        "checkpoint_provenance": tmp_path / "checkpoint_provenance.md",
        "output": tmp_path / "manifest.json",
    }
    paths["validation_database"].write_text(
        yaml.safe_dump(validation_rows), encoding="utf-8"
    )
    paths["train_database"].write_text(yaml.safe_dump(train_rows), encoding="utf-8")
    paths["sequence_database"].write_text(
        yaml.safe_dump(sequence_rows), encoding="utf-8"
    )
    _write(paths["checkpoint"], "checkpoint\n")
    _write(
        paths["checkpoint_provenance"],
        "trained on train; selected on validation; development only\n",
    )
    return paths


def _build(paths: dict[str, Path], *, count: int = 4) -> dict[str, object]:
    return build_t2_development_manifest(
        metadata_path=paths["metadata"],
        validation_list_path=paths["validation_list"],
        raw_root=paths["raw_root"],
        processed_root=paths["processed_root"],
        validation_database_path=paths["validation_database"],
        train_database_path=paths["train_database"],
        sequence_database_path=paths["sequence_database"],
        checkpoint_path=paths["checkpoint"],
        checkpoint_provenance_path=paths["checkpoint_provenance"],
        count=count,
        output_path=paths["output"],
    )


def test_selection_is_complete_exact_t2_validation_only_and_uuid_sorted(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    manifest = _build(paths)

    assert manifest["status"] == "PASS"
    assert manifest["artifact_id"] == "RSCAN_T2_DEV_V1"
    assert manifest["role"] == "DEVELOPMENT_ONLY"
    assert manifest["selected_pair_count"] == 4
    assert [pair["environment_id"] for pair in manifest["pairs"]] == [
        _uuid(10),
        _uuid(20),
        _uuid(30),
        _uuid(40),
    ]
    assert all(len(pair["sessions"]) == 2 for pair in manifest["pairs"])
    assert all(pair["split"] == "validation" for pair in manifest["pairs"])
    assert manifest["checkpoint_provenance"] == {
        "training_environment_overlap": False,
        "validation_selection_overlap": True,
        "claim_boundary": "development_reproduction_not_unseen_generalization",
    }


def test_manifest_separates_method_alignment_from_evaluator_only_gt(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    manifest = _build(paths, count=3)

    pair = manifest["pairs"][0]
    assert pair["common_method_inputs"]["global_alignment"]["direction"] == (
        "rescan_row_vector_to_reference"
    )
    evaluator = pair["evaluator_only"]
    assert evaluator["changes"]["rigid"][0]["symmetry"] == 2
    assert evaluator["ambiguity"][0][0]["instance_source"] == 7
    assert "evaluator_only" not in pair["common_method_inputs"]
    assert pair["sequence"]["key"] == "scene0010_00-scene0010_01"


def test_manifest_hash_binds_every_input_and_selected_asset(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    manifest = _build(paths, count=3)

    assert set(manifest["source_bindings"]) == {
        "metadata",
        "validation_list",
        "validation_database",
        "train_database",
        "sequence_database",
        "checkpoint",
        "checkpoint_provenance",
    }
    for record in manifest["source_bindings"].values():
        assert set(record) == {"path", "sha256", "byte_count"}
        assert len(record["sha256"]) == 64
    for pair in manifest["pairs"]:
        assert set(pair["sequence"]["change_gt_binding"]) == {
            "path",
            "sha256",
            "byte_count",
        }
        for session in pair["sessions"]:
            assert set(session["raw_assets"]) == set(RAW_RUNTIME_MEMBERS)
            assert set(session["processed_assets"]) == {"points", "instance_gt"}


def test_missing_or_incomplete_candidates_are_excluded_before_take_first(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    (paths["raw_root"] / _uuid(20) / "sequence.zip").unlink()

    manifest = _build(paths, count=3)

    assert [pair["environment_id"] for pair in manifest["pairs"]] == [
        _uuid(10),
        _uuid(30),
        _uuid(40),
    ]


def test_pair_without_declared_global_alignment_is_excluded(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    next(item for item in payload if item["reference"] == _uuid(10))["scans"][0].pop(
        "transform"
    )
    paths["metadata"].write_text(json.dumps(payload), encoding="utf-8")

    manifest = _build(paths, count=3)

    assert [pair["environment_id"] for pair in manifest["pairs"]] == [
        _uuid(20),
        _uuid(30),
        _uuid(40),
    ]


def test_selected_scan_cannot_overlap_checkpoint_training_database(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    rows = yaml.safe_load(paths["train_database"].read_text(encoding="utf-8"))
    validation_rows = yaml.safe_load(
        paths["validation_database"].read_text(encoding="utf-8")
    )
    rows.append(
        next(
            record
            for record in validation_rows
            if _uuid(10) in record["raw_filepath"]
        )
    )
    paths["train_database"].write_text(yaml.safe_dump(rows), encoding="utf-8")

    with pytest.raises(RScanT2DevelopmentError, match="training database overlap"):
        _build(paths, count=3)


def test_selection_requires_at_least_three_complete_pairs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(RScanT2DevelopmentError, match="at least three"):
        _build(paths, count=2)


def test_output_is_no_clobber(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].write_text("owned\n", encoding="utf-8")

    with pytest.raises(RScanT2DevelopmentError, match="already exists"):
        _build(paths, count=3)


def test_build_cli_writes_the_same_contract(tmp_path: Path) -> None:
    from scripts.evaluation.build_3rscan_t2_dev import main

    paths = _fixture(tmp_path)

    status = main(
        [
            "--metadata",
            str(paths["metadata"]),
            "--validation-list",
            str(paths["validation_list"]),
            "--raw-root",
            str(paths["raw_root"]),
            "--processed-root",
            str(paths["processed_root"]),
            "--validation-database",
            str(paths["validation_database"]),
            "--train-database",
            str(paths["train_database"]),
            "--sequence-database",
            str(paths["sequence_database"]),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--checkpoint-provenance",
            str(paths["checkpoint_provenance"]),
            "--count",
            "3",
            "--output",
            str(paths["output"]),
        ]
    )

    assert status == 0
    assert json.loads(paths["output"].read_text(encoding="utf-8"))["status"] == "PASS"


def test_audit_revalidates_every_selected_asset_binding(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = _build(paths, count=3)

    audited = audit_t2_development_manifest(paths["output"])

    assert audited["selected_pair_count"] == 3
    selected = manifest["pairs"][0]["sessions"][0]
    Path(selected["raw_assets"]["sequence.zip"]["path"]).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(RScanT2DevelopmentError, match="binding mismatch"):
        audit_t2_development_manifest(paths["output"])
