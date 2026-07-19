from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from types import ModuleType
from typing import Any, Callable

import numpy as np
from plyfile import PlyData, PlyElement
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/evaluation/diagnose_oviv2_room0.py"


@dataclass(frozen=True)
class ReplicaFixture:
    pred_semantic: Path
    pred_entity: Path
    gt_mesh: Path
    gt_info: Path
    manifest: Path


def _load_module() -> ModuleType:
    assert SCRIPT.is_file(), f"missing diagnostic CLI: {SCRIPT}"
    from scripts.evaluation import diagnose_oviv2_room0

    return diagnose_oviv2_room0


def _load_main() -> Callable[[list[str] | None], int]:
    main = _load_module().main

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


def _temporary_siblings(output: Path) -> list[Path]:
    return list(output.parent.glob(f".{output.name}.*.tmp"))


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


def test_atomic_json_no_clobber_preserves_competing_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_cli = _load_module()
    output = tmp_path / "room0_diagnostics.json"
    competitor = b'{"publisher": "competitor"}\n'
    real_link = diagnostic_cli.os.link

    def publish_competitor_then_link(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        Path(destination).write_bytes(competitor)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(diagnostic_cli.os, "link", publish_competitor_then_link)

    with pytest.raises(FileExistsError):
        diagnostic_cli._atomic_json(output, {"publisher": "diagnostic"}, overwrite=False)

    assert output.read_bytes() == competitor
    assert _temporary_siblings(output) == []


def test_run_rejects_inputs_changed_during_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_cli = _load_module()
    fixture = _replica_fixture(tmp_path)
    output = tmp_path / "room0_diagnostics.json"
    original_diagnose = diagnostic_cli.diagnose_projected_labels

    def diagnose_then_change_input(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = original_diagnose(*args, **kwargs)
        np.save(fixture.pred_semantic, np.asarray([2, 2, 1, 1], dtype=np.int64))
        return report

    monkeypatch.setattr(
        diagnostic_cli,
        "diagnose_projected_labels",
        diagnose_then_change_input,
    )

    with pytest.raises(RuntimeError, match="inputs changed during computation"):
        diagnostic_cli.run(diagnostic_cli.parse_args(_arguments(fixture, output)))

    assert not output.exists()
    assert _temporary_siblings(output) == []


def test_atomic_json_serialization_failure_cleans_temporary_sibling(
    tmp_path: Path,
) -> None:
    diagnostic_cli = _load_module()
    output = tmp_path / "room0_diagnostics.json"

    with pytest.raises(TypeError):
        diagnostic_cli._atomic_json(
            output,
            {"not_json": object()},
            overwrite=False,
        )

    assert not output.exists()
    assert _temporary_siblings(output) == []


def test_atomic_json_treats_dangling_symlink_as_occupied(tmp_path: Path) -> None:
    diagnostic_cli = _load_module()
    output = tmp_path / "room0_diagnostics.json"
    missing_target = tmp_path / "missing.json"
    try:
        output.symlink_to(missing_target)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(FileExistsError):
        diagnostic_cli._atomic_json(output, {"publisher": "diagnostic"}, overwrite=False)

    assert output.is_symlink()
    assert output.readlink() == missing_target
    assert _temporary_siblings(output) == []


def test_atomic_json_fsyncs_payload_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_cli = _load_module()
    output = tmp_path / "room0_diagnostics.json"
    fsynced_modes: list[int] = []
    real_fsync = diagnostic_cli.os.fsync

    def recording_fsync(file_descriptor: int) -> None:
        fsynced_modes.append(stat.S_IFMT(os.fstat(file_descriptor).st_mode))
        real_fsync(file_descriptor)

    monkeypatch.setattr(diagnostic_cli.os, "fsync", recording_fsync)

    diagnostic_cli._atomic_json(output, {"complete": True}, overwrite=False)

    assert stat.S_IFREG in fsynced_modes
    assert stat.S_IFDIR in fsynced_modes
