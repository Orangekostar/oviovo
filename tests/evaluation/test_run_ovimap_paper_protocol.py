from __future__ import annotations

import sys
import pickle
from pathlib import Path

import pytest

from scripts.evaluation.run_ovimap_paper_protocol import (
    REPLICA8_SCENES,
    RunnerConfig,
    build_command_plan,
    run,
)


def _fixture(tmp_path: Path) -> RunnerConfig:
    evaluator = tmp_path / "ovimap"
    for relative in (
        "scripts/datasets/preprocess_gt_mesh.py",
        "scripts/utils/mesh_postprocess_utils.py",
        "scripts/eval_inst_seg.py",
        "scripts/eval_sem_seg.py",
    ):
        path = evaluator / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    semantic_labels = tmp_path / "semantic_labels"
    instance_labels = tmp_path / "instance_labels"
    semantic_labels.mkdir()
    instance_labels.mkdir()
    siglip = tmp_path / "siglip"
    siglip.mkdir()
    scene_sources: dict[str, Path] = {}
    scene_color_logs: dict[str, Path] = {}
    gt_meshes: dict[str, Path] = {}
    for scene in REPLICA8_SCENES:
        source = tmp_path / "sources" / scene
        source.mkdir(parents=True)
        (source / "instance_mesh_200.ply").write_bytes(b"mesh")
        with (source / "inst_sem_siglip-l-16-384_200_incre_combine.pkl").open("wb") as handle:
            pickle.dump(
                {
                    1: {"color": [1, 1, 1], "frame_id": [0, 10], "box_2d": []},
                    2: {"color": [2, 2, 2], "frame_id": [0], "box_2d": []},
                },
                handle,
            )
        scene_sources[scene] = source
        color_log = tmp_path / "logs" / f"{scene}.log"
        color_log.parent.mkdir(parents=True, exist_ok=True)
        color_log.write_text("Instance: 1 Color: (10,20,30)\n", encoding="utf-8")
        scene_color_logs[scene] = color_log
        gt_mesh = tmp_path / "gt" / f"{scene}.ply"
        gt_mesh.parent.mkdir(parents=True, exist_ok=True)
        gt_mesh.write_bytes(b"gt")
        gt_meshes[scene] = gt_mesh
        (semantic_labels / f"semantic_labels_{scene}.txt").write_text("1\n", encoding="utf-8")
        (instance_labels / f"instance_labels_{scene}.txt").write_text("1001\n", encoding="utf-8")
    return RunnerConfig(
        evaluator_root=evaluator,
        python=Path(sys.executable),
        scene_sources=scene_sources,
        scene_color_logs=scene_color_logs,
        gt_meshes=gt_meshes,
        gt_semantic_folder=semantic_labels,
        gt_instance_folder=instance_labels,
        siglip_model=siglip,
        output=tmp_path / "run",
    )


def test_command_plan_runs_released_steps_without_claiming_paper_ap(tmp_path: Path) -> None:
    config = _fixture(tmp_path)

    commands = build_command_plan(config)

    assert len(commands) == 18
    assert commands[0]["name"] == "preprocess_gt_office0"
    assert commands[0]["argv"][2:4] == ["scripts.datasets.preprocess_gt_mesh", "--scene_num"]
    assert commands[1]["name"] == "mesh_postprocess_office0"
    assert commands[-2]["name"] == "released_instance_diagnostics"
    assert commands[-2]["argv"][2:5] == ["scripts.eval_inst_seg", "--scene_num", "all"]
    assert commands[-1]["name"] == "released_semantic_evaluation"
    assert commands[-1]["argv"][2] == "scripts.eval_sem_seg"


def test_dry_run_stages_replica8_and_preserves_released_metric_names(tmp_path: Path) -> None:
    config = _fixture(tmp_path)

    manifest = run(config, dry_run=True)

    assert manifest["status"] == "DRY_RUN"
    assert manifest["scene_ids"] == list(REPLICA8_SCENES)
    assert manifest["released_metric_contract"]["instance"] == [
        "mIoU",
        "wIoU",
        "mP@75",
        "mR@75",
        "mP@50",
        "mR@50",
        "mP@25",
        "mR@25",
    ]
    assert manifest["paper_metric_availability"]["table_2_class_agnostic_ap"] is False
    assert manifest["paper_metric_availability"]["table_3_semantic"] is True
    for scene in REPLICA8_SCENES:
        staged = config.output / "layout" / scene / "cropformer_inst"
        assert (staged / "instance_mesh_200.ply").is_symlink()
        remapped_features = staged / "inst_sem_siglip-l-16-384_200_incre_combine.pkl"
        assert remapped_features.is_file()
        assert not remapped_features.is_symlink()
        with remapped_features.open("rb") as handle:
            payload = pickle.load(handle)
        assert set(payload) == {1}
        assert payload[1]["color"] == (10, 20, 30)
        assert (config.output / "dataset" / "Replica" / f"{scene}_mesh.ply").is_symlink()
        assert manifest["inputs"][scene]["semantic_features"]["remapped_sha256"]
        assert manifest["inputs"][scene]["semantic_features"]["dropped_stale_ids"] == [2]
        assert manifest["inputs"][scene]["instance_color_log"]["sha256"]
    assert (config.output / "run_manifest.json").is_file()


def test_preflight_fails_before_output_creation_for_missing_native_artifact(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    (config.scene_sources["room2"] / "instance_mesh_200.ply").unlink()

    with pytest.raises(FileNotFoundError, match="room2.*instance_mesh_200"):
        run(config, dry_run=True)

    assert not config.output.exists()
