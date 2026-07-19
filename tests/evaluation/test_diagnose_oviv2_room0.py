from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
from plyfile import PlyData, PlyElement
import pytest


SCRIPT = Path("scripts/evaluation/diagnose_oviv2_room0.py")


@dataclass(frozen=True)
class ReplicaFixture:
    pred_semantic: Path
    pred_entity: Path
    gt_mesh: Path
    gt_info: Path
    manifest: Path


def _load_main() -> Callable[[list[str] | None], int]:
    assert SCRIPT.is_file(), f"missing diagnostic CLI: {SCRIPT}"
    from scripts.evaluation.diagnose_oviv2_room0 import main

    return main


def _replica_fixture(tmp_path: Path) -> ReplicaFixture:
    vertex_data = np.empty(
        4,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    vertex_data["x"] = [0, 1, 1, 0]
    vertex_data["y"] = [0, 0, 1, 1]
    vertex_data["z"] = 0
    face_data = np.empty(
        2,
        dtype=[("vertex_indices", "i4", (3,)), ("object_id", "i4")],
    )
    face_data["vertex_indices"] = [[0, 1, 2], [0, 2, 3]]
    face_data["object_id"] = [0, 1]

    gt_mesh = tmp_path / "mesh_semantic.ply"
    PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        text=False,
    ).write(gt_mesh)
    gt_info = tmp_path / "info_semantic.json"
    gt_info.write_text(
        json.dumps(
            {
                "objects": [
                    {"id": 0, "class_name": "chair"},
                    {"id": 1, "class_name": "table"},
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "diagnostic-fixture",
                "dataset": "Replica",
                "vocabulary": {"classes": ["chair", "table"]},
                "aliases": {},
                "scenes": [{"scene": "room0"}],
            }
        ),
        encoding="utf-8",
    )
    pred_semantic = tmp_path / "pred_semantic.npy"
    pred_entity = tmp_path / "pred_entity.npy"
    np.save(pred_semantic, np.asarray([1, 1, 2, 2], dtype=np.int64))
    np.save(pred_entity, np.asarray([10, 10, 11, 11], dtype=np.int64))
    return ReplicaFixture(
        pred_semantic=pred_semantic,
        pred_entity=pred_entity,
        gt_mesh=gt_mesh,
        gt_info=gt_info,
        manifest=manifest,
    )


def _arguments(fixture: ReplicaFixture, output: Path) -> list[str]:
    return [
        "--pred-semantic",
        str(fixture.pred_semantic),
        "--pred-entity",
        str(fixture.pred_entity),
        "--gt-mesh",
        str(fixture.gt_mesh),
        "--gt-info",
        str(fixture.gt_info),
        "--manifest",
        str(fixture.manifest),
        "--scene",
        "room0",
        "--output",
        str(output),
    ]


def test_main_writes_immutable_room0_diagnostic_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _replica_fixture(tmp_path)
    output = tmp_path / "room0_diagnostics.json"

    exit_code = _load_main()(
        [*_arguments(fixture, output), "--minimum-overlap-vertices", "1"]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["headline_eligible"] is False
    assert report["scene"] == "room0"
    assert report["protocol"] == {
        "manifest_id": "diagnostic-fixture",
        "minimum_overlap_vertices": 1,
        "minimum_overlap_fraction": 0.10,
    }
    assert set(report["input_sha256"]) == {
        "pred_semantic",
        "pred_entity",
        "gt_mesh",
        "gt_info",
        "manifest",
    }
    expected_stdout = {
        "output": str(output),
        "current_miou": report["semantic"]["current_miou"],
        "oracle_miou": report["semantic"]["oracle_miou"],
    }
    assert capsys.readouterr().out == json.dumps(expected_stdout, sort_keys=True) + "\n"


def test_main_refuses_existing_output_unless_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    fixture = _replica_fixture(tmp_path)
    output = tmp_path / "room0_diagnostics.json"
    original = b'{"sentinel": true}\n'
    output.write_bytes(original)
    arguments = _arguments(fixture, output)

    with pytest.raises(FileExistsError):
        _load_main()(arguments)
    assert output.read_bytes() == original

    assert _load_main()([*arguments, "--overwrite"]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["headline_eligible"] is False
