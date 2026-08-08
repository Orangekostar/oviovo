"""Tests for selected-scene Replica vocabulary generation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import argparse
from types import SimpleNamespace
from pathlib import Path

import pytest
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_room0_full_eval as room0_eval
from src.core.data_structures import ObjectMap, ObjectState, SemanticMemory, SystemState
from scripts import run_room0_checkpointed_eval
from scripts import build_replica_global_vocab
from scripts import run_replica_all_scenes_fast_eval
from scripts import summarize_replica_all_scenes
from scripts.build_replica_global_vocab import BANNED_BROAD_PROMPTS, build_replica_vocab


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _make_executable(path: Path) -> None:
    _touch(path)
    path.chmod(0o755)


def _write_precomputed_config(path: Path, manifest_path: Path | None = None, cache_dir: Path | None = None) -> None:
    lines = ["proposal:", "  precomputed:"]
    if cache_dir is not None:
        lines.append(f"    cache_dir: {cache_dir}")
    if manifest_path is not None:
        lines.append(f"    manifest_path: {manifest_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    scene: str,
    *,
    frame_ids: list[int] | None = None,
    touch_files: bool = True,
) -> None:
    frame_entries = []
    for frame_id in frame_ids if frame_ids is not None else range(0, 2000, 10):
        relpath = f"frames/frame{int(frame_id):06d}_proposals.npz"
        frame_entries.append({"frame_id": int(frame_id), "file": relpath})
        if touch_files:
            _touch(path.parent / relpath)
    _write_json(
        path,
        {
            "dataset_summary": {"root": f"/datasets/Replica/{scene}"},
            "frames": frame_entries,
        },
    )


def _write_fake_scene_outputs(
    output_root: Path,
    batch_name: str,
    scene: str,
    *,
    miou: float,
    macc: float,
    fmiou: float,
    fmacc: float,
    mapping_sec: float,
    export_sec: float,
    sec_per_frame: float,
    objects: int,
    dense_points: int,
    prefetch_hit_count: int = 0,
    prefetch_submitted_count: int = 0,
    prefetch_miss_count: int = 0,
    report_scene: str | None = None,
    status: str = "complete",
    processed_frames: int = 200,
    frame_metric_lines: int = 200,
    include_counts: bool = True,
) -> Path:
    run_root = output_root / run_replica_all_scenes_fast_eval.experiment_name_for_scene(
        batch_name,
        scene,
        frame_stride=10,
        num_frames=200,
    )
    _write_json(
        run_root / "mapping_timer_result.json",
        {
            "mapping_loop_sec_excluding_init_and_final_outputs": mapping_sec,
            "processed_frames": processed_frames,
            "sec_per_frame": sec_per_frame,
            "prefetch_hit_count": prefetch_hit_count,
            "prefetch_submitted_count": prefetch_submitted_count,
            "prefetch_miss_count": prefetch_miss_count,
            "frame_stride": 10,
            "requested_frames": 200,
        },
    )
    _write_json(run_root / "status.json", {"status": status})
    frame_metrics_path = run_root / scene / "frame_metrics.jsonl"
    frame_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    frame_metrics_path.write_text("{}\n" * frame_metric_lines, encoding="utf-8")
    export_payload = {"export_eval_sec": export_sec}
    if include_counts:
        export_payload.update(
            {
                "final_object_count": objects,
                "dense_geometry_point_count": dense_points,
            }
        )
    _write_json(
        run_root / "export_eval_timer_result.json",
        export_payload,
    )
    _write_json(
        run_root / "replica" / "results.json",
        {"miou": miou, "macc": macc, "fmiou": fmiou, "fmacc": fmacc},
    )
    _write_json(
        run_root / (report_scene or scene) / "run_report.json",
        (
            {
                "final_object_count": objects + 100,
                "dense_geometry_point_count": dense_points + 100,
            }
            if include_counts
            else {}
        ),
    )
    return run_root


def test_online_yoloworld_sam_config_uses_sam2_yoloworld_and_no_yoloe() -> None:
    config_path = Path("configs/replica_yoloworld_sam_online_baseline_4090.yaml")
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "sam2"
    assert config["anchor_frontend"]["backend"] == "yoloworld"
    assert config["anchor_frontend"]["supplemental_enabled"] is False
    assert config["anchor_frontend"]["confidence_threshold"] == 0.2
    assert config["anchor_frontend"]["max_detections"] == 128
    assert config["anchor_frontend"]["iou_assign_threshold"] == 0.2
    assert config["anchor_frontend"]["min_anchor_mask_coverage"] == 0.2
    assert config["anchor_frontend"]["min_protected_anchor_confidence"] == 0.25
    assert config["anchor_frontend"]["contained_max_anchor_coverage"] == 0.20
    assert config["anchor_frontend"]["contained_min_proposal_coverage"] == 0.85
    assert config["anchor_frontend"]["scale_compatible_min_anchor_coverage"] == 0.20
    assert config["anchor_frontend"]["scale_compatible_min_bbox_iou"] == 0.10
    assert config["anchor_guided_sam"]["enabled"] is True
    assert config["anchor_frontend"]["anchor_primary_mode"] is True
    assert config["anchor_frontend"]["use_sam_intersection_proposals"] is False
    assert config["pipeline"]["yoloworld_sam_parallel_frontend_enabled"] is True


def test_sam3_concept_experiment_config_disables_yoloworld_sam_and_runtimevis() -> None:
    config_path = Path("configs/replica_sam3_concept_room0_experiment_4090.yaml")
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "sam3_concept"
    assert config["proposal"]["sam3_concept"]["mode"] == "mock"
    assert "chair" in config["proposal"]["sam3_concept"]["classes"]
    assert config["proposal"]["sam3_concept"]["candidate_strategy"] == "round_robin"
    assert config["proposal"]["sam3_concept"]["max_prompts_per_frame"] == 6
    assert config["proposal"]["sam3_concept"]["always_include_classes"] == [
        "wall",
        "floor",
        "ceiling",
    ]
    assert config["proposal"]["sam3_concept"]["max_proposals"] == 24
    assert config["anchor_frontend"]["enabled"] is False
    assert config["anchor_guided_sam"]["enabled"] is False
    assert config["runtime_vis"]["enabled"] is False
    assert config["pipeline"]["yoloworld_sam_parallel_frontend_enabled"] is False
    assert config["pipeline"]["yoloworld_sam_balanced_frontend_enabled"] is False
    assert config["pipeline"]["proposal_prefetch_enabled"] is False


def test_run_replica_room0_analysis_accepts_sam3_worker_cli_args(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_replica_room0_analysis

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_replica_room0_analysis.py",
            "--proposal-backend",
            "sam3_concept",
            "--sam3-mode",
            "worker",
            "--sam3-worker-python",
            "/envs/sam3/bin/python",
            "--sam3-worker-script",
            "scripts/sam3_concept_worker.py",
            "--sam3-repo-root",
            "/src/sam3",
            "--sam3-checkpoint-path",
            "/ckpts/sam3.pt",
        ],
    )

    args = run_replica_room0_analysis.parse_args()

    assert args.sam3_mode == "worker"
    assert args.sam3_worker_python == Path("/envs/sam3/bin/python")
    assert args.sam3_worker_script == Path("scripts/sam3_concept_worker.py")
    assert args.sam3_repo_root == Path("/src/sam3")
    assert args.sam3_checkpoint_path == Path("/ckpts/sam3.pt")


def test_run_replica_room0_analysis_builds_sam3_worker_proposal_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_replica_room0_analysis
    import src.modules.proposal as proposal_module

    captured: dict[str, object] = {}

    class FakeProposalModule:
        def __init__(self, config: dict[str, object]) -> None:
            captured.update(config)

    monkeypatch.setattr(proposal_module, "ProposalModule", FakeProposalModule)
    monkeypatch.setattr(run_replica_room0_analysis, "ProposalModule", FakeProposalModule)

    run_replica_room0_analysis.build_proposal_module(
        {
            "backend": "sam3_concept",
            "min_mask_area": 100,
            "sam3_concept": {"classes": ["chair"]},
        },
        argparse.Namespace(
            proposal_backend="sam3_concept",
            min_mask_area=25,
            proposal_device="cuda",
            sam3_mode="worker",
            sam3_worker_python=Path("/envs/sam3/bin/python"),
            sam3_worker_script=Path("scripts/sam3_concept_worker.py"),
            sam3_repo_root=Path("/src/sam3"),
            sam3_checkpoint_path=Path("/ckpts/sam3.pt"),
            max_proposals=7,
            confidence_threshold=0.33,
        ),
    )

    assert captured["backend"] == "sam3_concept"
    assert captured["min_mask_area"] == 25
    assert captured["mode"] == "worker"
    assert captured["worker_python"] == "/envs/sam3/bin/python"
    assert captured["worker_script"] == "scripts/sam3_concept_worker.py"
    assert captured["device"] == "cuda"
    assert captured["sam3_repo_root"] == "/src/sam3"
    assert captured["checkpoint_path"] == "/ckpts/sam3.pt"
    assert captured["max_proposals"] == 7
    assert captured["confidence_threshold"] == 0.33


def test_sam3_build_proposal_module_keeps_yaml_budget_when_cli_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_replica_room0_analysis
    import src.modules.proposal as proposal_module

    captured: dict[str, object] = {}

    class FakeProposalModule:
        def __init__(self, config: dict[str, object]) -> None:
            captured.update(config)

    monkeypatch.setattr(proposal_module, "ProposalModule", FakeProposalModule)
    monkeypatch.setattr(run_replica_room0_analysis, "ProposalModule", FakeProposalModule)

    run_replica_room0_analysis.build_proposal_module(
        {
            "backend": "sam3_concept",
            "min_mask_area": 100,
            "sam3_concept": {
                "classes": ["chair"],
                "max_proposals": 24,
                "confidence_threshold": 0.35,
            },
        },
        argparse.Namespace(
            proposal_backend="sam3_concept",
            min_mask_area=25,
            proposal_device="cuda",
            sam3_mode="worker",
            sam3_worker_python=None,
            sam3_worker_script=None,
            sam3_repo_root=None,
            sam3_checkpoint_path=None,
            max_proposals=64,
            confidence_threshold=0.0,
        ),
    )

    assert captured["max_proposals"] == 24
    assert captured["confidence_threshold"] == 0.35


def test_build_replica_vocab_unions_aliases_and_scene_coverage(tmp_path: Path) -> None:
    room0_info = tmp_path / "room_0" / "habitat" / "info_semantic.json"
    room1_info = tmp_path / "room_1" / "habitat" / "info_semantic.json"
    _write_json(
        room0_info,
        {
            "classes": [
                {"id": 1, "name": "wall"},
                {"id": 2, "name": "sofa"},
                {"id": 3, "name": "rug"},
                {"id": 4, "name": "wall-plug"},
            ]
        },
    )
    _write_json(
        room1_info,
        {
            "classes": {
                "1": {"name": "floor"},
                "2": {"class_name": "picture"},
                "3": "indoor-plant",
                "4": {"name": "sofa"},
            }
        },
    )

    vocab = build_replica_vocab({"room0": room0_info, "room1": room1_info})

    assert vocab["canonical_eval_vocab"] == [
        "floor",
        "indoor-plant",
        "picture",
        "rug",
        "sofa",
        "wall",
        "wall-plug",
    ]
    assert vocab["prompt_aliases"] == {
        "indoor-plant": ["indoor plant", "plant"],
        "picture": ["picture", "painting", "wall art"],
        "rug": ["rug", "carpet"],
        "sofa": ["sofa", "couch"],
        "wall-plug": ["wall plug", "outlet", "electrical outlet"],
    }
    assert vocab["structural_labels"] == ["floor", "rug", "wall"]
    assert vocab["object_labels"] == ["indoor-plant", "picture", "sofa", "wall-plug"]
    assert vocab["scene_class_coverage"] == {
        "room0": ["rug", "sofa", "wall", "wall-plug"],
        "room1": ["floor", "indoor-plant", "picture", "sofa"],
    }


def test_build_replica_vocab_extracts_names_from_common_info_semantic_shapes(tmp_path: Path) -> None:
    room0_info = tmp_path / "room_0" / "habitat" / "info_semantic.json"
    room1_info = tmp_path / "room_1" / "habitat" / "info_semantic.json"
    _write_json(
        room0_info,
        {
            "objects": [
                {"class_name": "chair"},
                {"category": {"name": "ceiling"}},
                {"semantic_label": "cabinet"},
            ]
        },
    )
    _write_json(
        room1_info,
        {
            "semantic_classes": [
                {"label": "window"},
                {"raw_category": "blinds"},
                {"name": "cabinet"},
            ]
        },
    )

    vocab = build_replica_vocab({"room0": room0_info, "room1": room1_info})

    assert vocab["canonical_eval_vocab"] == ["blinds", "cabinet", "ceiling", "chair", "window"]
    assert vocab["prompt_aliases"] == {
        "blinds": ["blinds", "window blinds"],
        "cabinet": ["cabinet", "cupboard"],
    }
    assert vocab["structural_labels"] == ["blinds", "ceiling", "window"]
    assert vocab["object_labels"] == ["cabinet", "chair"]
    assert vocab["scene_class_coverage"] == {
        "room0": ["cabinet", "ceiling", "chair"],
        "room1": ["blinds", "cabinet", "window"],
    }


def test_build_replica_vocab_prefers_official_classes_over_object_labels(tmp_path: Path) -> None:
    room0_info = tmp_path / "room_0" / "habitat" / "info_semantic.json"
    room1_info = tmp_path / "room_1" / "habitat" / "info_semantic.json"
    _write_json(
        room0_info,
        {
            "classes": [
                {"id": 1, "name": "wall"},
                {"id": 2, "name": "chair"},
            ],
            "objects": [
                {"id": 1, "class_name": "undefined"},
                {"id": 2, "category": {"name": "other-leaf"}},
                {"id": 3, "semantic_label": "anonymize_picture"},
                {"id": 4, "name": "non-plane"},
            ],
            "instances": [{"name": "anonymize_text"}],
            "labels": ["thing"],
        },
    )
    _write_json(
        room1_info,
        {
            "classes": [
                {"id": 1, "name": "floor"},
                {"id": 2, "name": "sofa"},
            ],
            "objects": [{"class_name": "background"}],
        },
    )

    vocab = build_replica_vocab({"room0": room0_info, "room1": room1_info})

    assert vocab["canonical_eval_vocab"] == ["chair", "floor", "sofa", "wall"]
    assert vocab["scene_class_coverage"] == {
        "room0": ["chair", "wall"],
        "room1": ["floor", "sofa"],
    }
    assert set(vocab["canonical_eval_vocab"]).isdisjoint(BANNED_BROAD_PROMPTS)
    assert set(vocab["prompt_aliases"]).isdisjoint(BANNED_BROAD_PROMPTS)


def test_build_replica_vocab_filters_contaminants_without_classes(tmp_path: Path) -> None:
    room0_info = tmp_path / "room_0" / "habitat" / "info_semantic.json"
    _write_json(
        room0_info,
        {
            "objects": [
                {"id": 1, "name": "chair-instance-001", "class_name": "chair"},
                {"id": 2, "name": "cabinet-instance-001", "category": {"name": "cabinet"}},
                {"id": 2, "name": "undefined"},
                {"id": 3, "category": {"name": "other-leaf"}},
                {"id": 4, "semantic_label": "anonymize_picture"},
            ],
            "instances": [
                {"name": "anonymize_text"},
                {"name": "non-plane"},
                {"name": "plant"},
            ],
            "labels": ["floor", "object"],
        },
    )

    vocab = build_replica_vocab({"room0": room0_info})

    assert vocab["canonical_eval_vocab"] == ["cabinet", "chair", "floor", "plant"]
    assert vocab["scene_class_coverage"] == {"room0": ["cabinet", "chair", "floor", "plant"]}


def test_build_replica_vocab_rejects_banned_alias_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    room0_info = tmp_path / "room_0" / "habitat" / "info_semantic.json"
    _write_json(room0_info, {"classes": [{"id": 1, "name": "sofa"}]})
    monkeypatch.setitem(
        build_replica_global_vocab.CONSERVATIVE_PROMPT_ALIASES,
        "sofa",
        ["sofa", " Furniture "],
    )

    with pytest.raises(ValueError, match="Banned broad prompts in prompt_aliases"):
        build_replica_vocab({"room0": room0_info})


def test_generalized_room0_layout_places_scene_artifacts_under_scene_name(tmp_path: Path) -> None:
    layout = room0_eval.build_room0_run_layout(tmp_path / "runs", "office0_eval", scene_name="office0")

    assert layout.run_root == tmp_path / "runs" / "office0_eval"
    assert layout.scene_dir == layout.run_root / "office0"
    assert layout.exports_dir == layout.run_root / "office0" / "exports"
    assert layout.eval_dir == layout.run_root / "replica"
    assert layout.scene_dir != layout.run_root / "room0"


def test_layout_normalizes_blank_scene_names_to_room0(tmp_path: Path) -> None:
    layout = room0_eval.build_room0_run_layout(tmp_path / "runs", "blank_scene", scene_name="  ")

    assert layout.scene_dir == layout.run_root / "room0"


def test_full_eval_parse_args_accepts_scene_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_room0_full_eval.py", "--scene-name", "office0"])

    args = room0_eval.parse_args()

    assert args.scene_name == "office0"


def test_generalized_checkpoint_layout_preserves_scene_name_and_gt_paths(tmp_path: Path) -> None:
    layout = room0_eval.build_room0_run_layout(tmp_path / "runs", "office0_eval", scene_name="office0")
    gt_labels = tmp_path / "gt" / "office0.txt"
    gt_mesh = tmp_path / "gt" / "office0_mesh.ply"
    gt_info = tmp_path / "gt" / "office0_info.json"
    args = SimpleNamespace(
        scene_name="office0",
        output_root=tmp_path / "runs",
        dataset_root=tmp_path / "dataset" / "office0",
        gt_labels=gt_labels,
        gt_mesh_ply=gt_mesh,
        gt_info_json=gt_info,
        config_path=tmp_path / "config.yaml",
        proposal_cache_dir=None,
        proposal_cache_manifest=None,
        sam_repo_root=tmp_path / "sam",
        sam_ckpt_path=tmp_path / "sam_ckpt",
        checkpoint_path=None,
    )
    pipeline = SimpleNamespace(config={}, proposal=SimpleNamespace(active_backend_name="precomputed"))
    output_profile = room0_eval.OutputProfile(name="fast_eval", benchmark_audit_enabled=False)

    payload = room0_eval.build_room0_checkpoint_payload(
        experiment_name="office0_eval",
        layout=layout,
        args=args,
        pipeline=pipeline,
        state=SimpleNamespace(objects={}, provisional_objects={}),
        geometry_accum={},
        frame_metrics=[],
        frame_audits=[],
        frame_limit=0,
        selected_dataset_indices=[],
        output_profile=output_profile,
        build_timing={},
    )
    loaded_args = room0_eval.args_from_room0_checkpoint(payload)
    reconstructed = room0_eval.build_room0_run_layout(
        loaded_args.output_root,
        payload["experiment_name"],
        scene_name=loaded_args.scene_name,
    )

    assert payload["scene_name"] == "office0"
    assert loaded_args.scene_name == "office0"
    assert loaded_args.gt_labels == gt_labels
    assert loaded_args.gt_mesh_ply == gt_mesh
    assert loaded_args.gt_info_json == gt_info
    assert reconstructed.scene_dir == reconstructed.run_root / "office0"


def test_run_report_uses_passed_scene_name(tmp_path: Path) -> None:
    scene_dir = tmp_path / "runs" / "office0_eval" / "office0"
    eval_dir = tmp_path / "runs" / "office0_eval" / "replica"
    scene_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    state = SystemState(objects={1: obj})
    pool_records = room0_eval.build_pool_semantic_records(state)
    args = SimpleNamespace(
        scene_name="office0",
        proposal_device="cuda",
        sam_version="2",
        sam_encoder="hiera_l",
        points_per_side=16,
        max_proposals=64,
    )

    room0_eval.write_run_report(
        scene_dir=scene_dir,
        eval_dir=eval_dir,
        experiment_name="office0_eval",
        dataset=SimpleNamespace(),
        args=args,
        frame_limit=1,
        pipeline=SimpleNamespace(proposal=SimpleNamespace(active_backend_name="precomputed")),
        state=state,
        tsdf_records=np.zeros(0, dtype=room0_eval.build_tsdf_backbone_records(SystemState()).dtype),
        pool_semantic_records=pool_records,
        dense_surface_records=np.zeros(0, dtype=pool_records.dtype),
        dense_points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        evaluation={
            "miou": 0.1,
            "macc": 0.2,
            "fmiou": 0.3,
            "fmacc": 0.4,
            "per_class": [{"class_name": "chair", "iou": 0.5, "acc": 0.6}],
        },
        frame_metrics=[
            {
                "raw_proposal_count": 1,
                "matched_patch_count": 1,
                "ambiguous_patch_count": 0,
                "ambiguous_matched_count": 0,
                "ambiguous_new_patch_count": 0,
                "local_memory_point_count_total": 1,
            }
        ],
        audit_dir=None,
        output_profile=room0_eval.OutputProfile(name="fast_eval", benchmark_audit_enabled=False),
    )

    rendered = (scene_dir / "run_report.md").read_text(encoding="utf-8")
    assert "- Scene: `office0`" in rendered
    assert "- Scene: `room0`" not in rendered


def test_export_from_checkpoint_status_uses_checkpoint_scene_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "runs" / "office0_eval" / "office0" / "mapping_state.pkl"
    payload = {
        "experiment_name": "office0_eval",
        "scene_name": "office0",
        "args": {
            "scene_name": "office0",
            "output_root": tmp_path / "runs",
            "dataset_root": tmp_path / "dataset" / "office0",
        },
    }
    monkeypatch.setattr(
        run_room0_checkpointed_eval,
        "parse_args",
        lambda: SimpleNamespace(
            mode="export",
            checkpoint_path=checkpoint_path,
            experiment_name=None,
            scene_name="room0",
            quiet=False,
        ),
    )
    monkeypatch.setattr(run_room0_checkpointed_eval, "load_room0_checkpoint", lambda _path: payload)
    monkeypatch.setattr(run_room0_checkpointed_eval, "run_selected_modes", lambda *_args, **_kwargs: None)

    run_room0_checkpointed_eval.main()

    status = json.loads((tmp_path / "runs" / "office0_eval" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["scene_name"] == "office0"


def test_export_checkpoint_reconstructs_scene_specific_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "runs" / "office0_eval" / "office0" / "mapping_state.pkl"
    payload = {
        "experiment_name": "office0_eval",
        "scene_name": "office0",
        "args": {
            "scene_name": "office0",
            "output_root": tmp_path / "runs",
            "dataset_root": tmp_path / "dataset" / "office0",
        },
        "output_profile": {"name": "fast_eval", "benchmark_audit_enabled": False},
        "frame_limit": 1,
        "state": SimpleNamespace(objects={}),
        "geometry_accum": {},
        "frame_metrics": [],
        "frame_audits": [],
    }
    captured: dict[str, object] = {}

    class ExportResult:
        report_path = tmp_path / "runs" / "office0_eval" / "office0" / "run_report.md"
        results_path = tmp_path / "runs" / "office0_eval" / "replica" / "results.json"
        instance_map_path = tmp_path / "runs" / "office0_eval" / "office0" / "exports" / "room0_instance_map.ply"
        dense_surface_path = tmp_path / "runs" / "office0_eval" / "office0" / "exports" / "room0_instance_map_dense_surface.ply"
        dense_instance_path = (
            tmp_path / "runs" / "office0_eval" / "office0" / "exports" / "room0_dense_geometry_instance_projected.ply"
        )
        miou = 0.0
        macc = 0.0
        fmiou = 0.0
        fmacc = 0.0
        final_object_count = 0
        dense_geometry_point_count = 0
        finalization_sec = 0.0
        largest_export_size_bytes = 0

    def fake_export_room0_outputs(**kwargs):
        captured["layout"] = kwargs["layout"]
        return ExportResult()

    monkeypatch.setattr(run_room0_checkpointed_eval, "load_room0_checkpoint", lambda _path: payload)
    monkeypatch.setattr(run_room0_checkpointed_eval, "ReplicaRoom0Dataset", lambda _path: SimpleNamespace())
    monkeypatch.setattr(run_room0_checkpointed_eval, "export_room0_outputs", fake_export_room0_outputs)
    monkeypatch.setattr(run_room0_checkpointed_eval, "_write_json", lambda _path, _payload: None)

    run_room0_checkpointed_eval.export_checkpoint(checkpoint_path)

    layout = captured["layout"]
    assert layout.scene_dir == tmp_path / "runs" / "office0_eval" / "office0"
    assert layout.exports_dir == tmp_path / "runs" / "office0_eval" / "office0" / "exports"


def test_all_scenes_default_mapping_and_scene_paths(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.scene_paths(
        "office3",
        dataset_root_base=tmp_path / "Replica",
        gt_original_base=tmp_path / "Replica_original",
        gt_label_dir=tmp_path / "labels",
    )

    assert run_replica_all_scenes_fast_eval.selected_scene_names(None) == [
        "room0",
        "room1",
        "room2",
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
    ]
    assert paths.dataset_root == tmp_path / "Replica" / "office3"
    assert paths.gt_labels == tmp_path / "labels" / "office3.txt"
    assert paths.gt_mesh_ply == tmp_path / "Replica_original" / "office_3" / "habitat" / "mesh_semantic.ply"
    assert paths.gt_info_json == tmp_path / "Replica_original" / "office_3" / "habitat" / "info_semantic.json"


def test_all_scenes_command_includes_fast_eval_scene_name_and_requested_frames(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "Replica" / "office1",
        gt_labels=tmp_path / "labels" / "office1.txt",
        gt_mesh_ply=tmp_path / "Replica_original" / "office_1" / "habitat" / "mesh_semantic.ply",
        gt_info_json=tmp_path / "Replica_original" / "office_1" / "habitat" / "info_semantic.json",
    )

    command = run_replica_all_scenes_fast_eval.build_scene_command(
        scene="office1",
        experiment_name="batch_office1_s5_3f_fast_eval",
        paths=paths,
        python_executable=tmp_path / "python",
        runner_script=tmp_path / "scripts" / "run_room0_checkpointed_eval.py",
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "outputs",
        num_frames=3,
        frame_stride=5,
        proposal_backend="precomputed",
        proposal_device="cuda",
    )

    assert command[:2] == [str(tmp_path / "python"), str(tmp_path / "scripts" / "run_room0_checkpointed_eval.py")]
    assert "--fast-eval" in command
    assert "--quiet" in command
    assert command[command.index("--scene-name") + 1] == "office1"
    assert command[command.index("--num-frames") + 1] == "3"
    assert command[command.index("--frame-stride") + 1] == "5"
    formatted = run_replica_all_scenes_fast_eval.format_scene_command(command)
    assert formatted.startswith(
        "CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 "
    )


def test_cache_dir_override_passes_derived_manifest_to_runner(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "Replica" / "room0",
        gt_labels=tmp_path / "labels" / "room0.txt",
        gt_mesh_ply=tmp_path / "Replica_original" / "room_0" / "habitat" / "mesh_semantic.ply",
        gt_info_json=tmp_path / "Replica_original" / "room_0" / "habitat" / "info_semantic.json",
    )

    command = run_replica_all_scenes_fast_eval.build_scene_command(
        scene="room0",
        experiment_name="batch_room0_s10_2f_fast_eval",
        paths=paths,
        python_executable=tmp_path / "python",
        runner_script=tmp_path / "scripts" / "run_room0_checkpointed_eval.py",
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "outputs",
        num_frames=2,
        frame_stride=10,
        proposal_backend="precomputed",
        proposal_device="cuda",
        proposal_cache=run_replica_all_scenes_fast_eval.ProposalCacheOverride(
            cache_dir=tmp_path / "cache" / "room0",
        ),
    )

    assert command[command.index("--proposal-cache-dir") + 1] == str(tmp_path / "cache" / "room0")
    assert command[command.index("--proposal-cache-manifest") + 1] == str(
        tmp_path / "cache" / "room0" / "manifest.json"
    )


def test_online_yoloworld_sam_command_passes_local_sam2_paths(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "Replica" / "room1",
        gt_labels=tmp_path / "labels" / "room1.txt",
        gt_mesh_ply=tmp_path / "Replica_original" / "room_1" / "habitat" / "mesh_semantic.ply",
        gt_info_json=tmp_path / "Replica_original" / "room_1" / "habitat" / "info_semantic.json",
    )

    command = run_replica_all_scenes_fast_eval.build_scene_command(
        scene="room1",
        experiment_name="online_room1_s10_1f_fast_eval",
        paths=paths,
        python_executable=tmp_path / "python",
        runner_script=tmp_path / "scripts" / "run_room0_checkpointed_eval.py",
        config_path=tmp_path / "online.yaml",
        output_root=tmp_path / "outputs",
        num_frames=1,
        frame_stride=10,
        proposal_backend="sam2",
        proposal_device="cuda",
    )

    assert command[command.index("--sam-version") + 1] == "2.1"
    assert command[command.index("--sam-repo-root") + 1] == str(
        run_replica_all_scenes_fast_eval.DEFAULT_SAM_REPO_ROOT
    )
    assert command[command.index("--sam-ckpt-path") + 1] == str(
        run_replica_all_scenes_fast_eval.DEFAULT_SAM_CKPT_PATH
    )
    assert command[command.index("--max-proposals") + 1] == "50"
    assert command[command.index("--confidence-threshold") + 1] == "0.5"


def test_online_sam3_concept_command_passes_backend_without_sam2_flags(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "Replica" / "room0",
        gt_labels=tmp_path / "labels" / "room0.txt",
        gt_mesh_ply=tmp_path / "Replica_original" / "room_0" / "habitat" / "mesh_semantic.ply",
        gt_info_json=tmp_path / "Replica_original" / "room_0" / "habitat" / "info_semantic.json",
    )

    command = run_replica_all_scenes_fast_eval.build_scene_command(
        scene="room0",
        experiment_name="sam3_room0_s10_1f_fast_eval",
        paths=paths,
        python_executable=tmp_path / "python",
        runner_script=tmp_path / "scripts" / "run_room0_checkpointed_eval.py",
        config_path=tmp_path / "sam3.yaml",
        output_root=tmp_path / "outputs",
        num_frames=1,
        frame_stride=10,
        proposal_backend="sam3_concept",
        proposal_device="cuda",
    )

    assert command[command.index("--proposal-backend") + 1] == "sam3_concept"
    assert "--sam-version" not in command
    assert "--proposal-cache-manifest" not in command


def test_online_sam3_concept_command_passes_runner_sam3_worker_args(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "Replica" / "room0",
        gt_labels=tmp_path / "labels" / "room0.txt",
        gt_mesh_ply=tmp_path / "Replica_original" / "room_0" / "habitat" / "mesh_semantic.ply",
        gt_info_json=tmp_path / "Replica_original" / "room_0" / "habitat" / "info_semantic.json",
    )

    command = run_replica_all_scenes_fast_eval.build_scene_command(
        scene="room0",
        experiment_name="sam3_room0_s10_1f_fast_eval",
        paths=paths,
        python_executable=tmp_path / "python",
        runner_script=tmp_path / "scripts" / "run_room0_checkpointed_eval.py",
        config_path=tmp_path / "sam3.yaml",
        output_root=tmp_path / "outputs",
        num_frames=1,
        frame_stride=10,
        proposal_backend="sam3_concept",
        proposal_device="cuda",
        sam3_mode="worker",
        sam3_worker_python=tmp_path / "envs" / "sam3" / "bin" / "python",
        sam3_worker_script=tmp_path / "scripts" / "sam3_concept_worker.py",
        sam3_repo_root=tmp_path / "third_party" / "sam3",
        sam3_checkpoint_path=tmp_path / "checkpoints" / "sam3.pt",
    )

    assert command[command.index("--sam3-mode") + 1] == "worker"
    assert command[command.index("--sam3-worker-python") + 1] == str(
        tmp_path / "envs" / "sam3" / "bin" / "python"
    )
    assert command[command.index("--sam3-worker-script") + 1] == str(
        tmp_path / "scripts" / "sam3_concept_worker.py"
    )
    assert command[command.index("--sam3-repo-root") + 1] == str(tmp_path / "third_party" / "sam3")
    assert command[command.index("--sam3-checkpoint-path") + 1] == str(
        tmp_path / "checkpoints" / "sam3.pt"
    )



def test_online_sam3_concept_command_ignores_precomputed_cache_overrides(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "Replica" / "room0",
        gt_labels=tmp_path / "labels" / "room0.txt",
        gt_mesh_ply=tmp_path / "Replica_original" / "room_0" / "habitat" / "mesh_semantic.ply",
        gt_info_json=tmp_path / "Replica_original" / "room_0" / "habitat" / "info_semantic.json",
    )

    command = run_replica_all_scenes_fast_eval.build_scene_command(
        scene="room0",
        experiment_name="sam3_room0_s10_1f_fast_eval",
        paths=paths,
        python_executable=tmp_path / "python",
        runner_script=tmp_path / "scripts" / "run_room0_checkpointed_eval.py",
        config_path=tmp_path / "sam3.yaml",
        output_root=tmp_path / "outputs",
        num_frames=1,
        frame_stride=10,
        proposal_backend="sam3_concept",
        proposal_device="cuda",
        proposal_cache=run_replica_all_scenes_fast_eval.ProposalCacheOverride(
            manifest_path=tmp_path / "cache" / "manifest.json",
            cache_dir=tmp_path / "cache",
        ),
    )

    assert command[command.index("--proposal-backend") + 1] == "sam3_concept"
    assert "--proposal-cache-manifest" not in command
    assert "--proposal-cache-dir" not in command


def test_scene_is_complete_requires_status_outputs_report_and_requested_frame_count(tmp_path: Path) -> None:
    run_root = tmp_path / "batch_room1_s10_2f_fast_eval"
    scene_dir = run_root / "room1"
    _write_json(run_root / "status.json", {"status": "complete"})
    _touch(scene_dir / "mapping_state.pkl")
    _write_json(run_root / "replica" / "results.json", {"miou": 1.0})
    (scene_dir / "frame_metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (scene_dir / "frame_metrics.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    _touch(scene_dir / "run_report.md")

    assert run_replica_all_scenes_fast_eval.scene_is_complete(run_root, "room1", expected_frames=2)

    (scene_dir / "frame_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    assert not run_replica_all_scenes_fast_eval.scene_is_complete(run_root, "room1", expected_frames=2)


def test_scene_is_complete_accepts_legacy_room0_report_path(tmp_path: Path) -> None:
    run_root = tmp_path / "batch_office2_s10_1f_fast_eval"
    scene_dir = run_root / "office2"
    _write_json(run_root / "status.json", {"status": "complete"})
    _touch(scene_dir / "mapping_state.pkl")
    _write_json(run_root / "replica" / "results.json", {"miou": 1.0})
    (scene_dir / "frame_metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (scene_dir / "frame_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    _touch(run_root / "room0" / "run_report.md")

    assert run_replica_all_scenes_fast_eval.scene_is_complete(run_root, "office2", expected_frames=1)


def test_validate_scene_inputs_reports_missing_paths(tmp_path: Path) -> None:
    paths = run_replica_all_scenes_fast_eval.ScenePaths(
        dataset_root=tmp_path / "missing_dataset",
        gt_labels=tmp_path / "missing_labels.txt",
        gt_mesh_ply=tmp_path / "missing_mesh.ply",
        gt_info_json=tmp_path / "missing_info.json",
    )

    errors = run_replica_all_scenes_fast_eval.validate_scene_inputs(
        scene="room2",
        paths=paths,
        python_executable=tmp_path / "missing_python",
        config_path=tmp_path / "missing_config.yaml",
        runner_script=tmp_path / "missing_runner.py",
    )

    assert errors == [
        f"room2: missing dataset root: {paths.dataset_root}",
        f"room2: missing GT labels: {paths.gt_labels}",
        f"room2: missing GT mesh PLY: {paths.gt_mesh_ply}",
        f"room2: missing GT info JSON: {paths.gt_info_json}",
        f"room2: missing config path: {tmp_path / 'missing_config.yaml'}",
        f"room2: missing runner script: {tmp_path / 'missing_runner.py'}",
        f"room2: missing python executable: {tmp_path / 'missing_python'}",
    ]


def test_selected_scene_names_deduplicates_preserving_order() -> None:
    assert run_replica_all_scenes_fast_eval.selected_scene_names(["room1", "room1", "office0", "room1"]) == [
        "room1",
        "office0",
    ]


def test_parse_args_rejects_unknown_scenes_and_non_positive_counts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as unknown_scene:
        run_replica_all_scenes_fast_eval.parse_args(["--batch-name", "batch", "--scenes", "missing"])
    assert unknown_scene.value.code == 2
    assert "Unknown scene(s): missing" in capsys.readouterr().err

    with pytest.raises(SystemExit) as bad_frames:
        run_replica_all_scenes_fast_eval.parse_args(["--batch-name", "batch", "--num-frames", "0"])
    assert bad_frames.value.code == 2
    assert "argument --num-frames: must be positive" in capsys.readouterr().err

    with pytest.raises(SystemExit) as bad_stride:
        run_replica_all_scenes_fast_eval.parse_args(["--batch-name", "batch", "--frame-stride", "-1"])
    assert bad_stride.value.code == 2
    assert "argument --frame-stride: must be positive" in capsys.readouterr().err


def test_relative_output_root_is_normalized_for_command_and_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str, int]] = []

    def fake_scene_is_complete(run_root: Path, scene: str, *, expected_frames: int) -> bool:
        calls.append((run_root, scene, expected_frames))
        return True

    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "scene_is_complete", fake_scene_is_complete)
    monkeypatch.setattr(run_replica_all_scenes_fast_eval.subprocess, "run", lambda *_args, **_kwargs: None)
    _make_executable(tmp_path / "python")
    _write_manifest(tmp_path / "cache" / "manifest.json", "room0")
    _write_precomputed_config(tmp_path / "config.yaml", tmp_path / "cache" / "manifest.json")
    for path in [
        tmp_path / "runner.py",
        tmp_path / "Replica" / "room0",
        tmp_path / "labels" / "room0.txt",
        tmp_path / "Original" / "room_0" / "habitat" / "mesh_semantic.ply",
        tmp_path / "Original" / "room_0" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "room0",
            "--output-root",
            "relative_outputs",
            "--python-executable",
            str(tmp_path / "python"),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
        ]
    )

    assert result == 0
    expected_output_root = tmp_path / "relative_outputs"
    assert calls == [(expected_output_root / "batch_room0_s10_200f_fast_eval", "room0", 200)]


def test_relative_paths_are_normalized_against_repo_root_when_invoked_outside_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    _make_executable(repo_root / "bin" / "python")
    _write_manifest(repo_root / "cache" / "office_2" / "manifest.json", "office2")
    _write_precomputed_config(repo_root / "configs" / "config.yaml")
    for path in [
        repo_root / "scripts" / "runner.py",
        repo_root / "Replica" / "office2",
        repo_root / "labels" / "office2.txt",
        repo_root / "Original" / "office_2" / "habitat" / "mesh_semantic.ply",
        repo_root / "Original" / "office_2" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "REPO_ROOT", repo_root)
    monkeypatch.chdir(other_cwd)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "office2",
            "--dry-run",
            "--output-root",
            "outputs",
            "--python-executable",
            "bin/python",
            "--runner-script",
            "scripts/runner.py",
            "--config-path",
            "configs/config.yaml",
            "--dataset-root-base",
            "Replica",
            "--gt-original-base",
            "Original",
            "--gt-label-dir",
            "labels",
            "--proposal-cache-manifest-template",
            "cache/{gt_scene}/manifest.json",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert str(repo_root / "bin" / "python") in captured.out
    assert str(repo_root / "scripts" / "runner.py") in captured.out
    assert str(repo_root / "configs" / "config.yaml") in captured.out
    assert str(repo_root / "Replica" / "office2") in captured.out
    assert str(repo_root / "labels" / "office2.txt") in captured.out
    assert str(repo_root / "Original" / "office_2" / "habitat" / "mesh_semantic.ply") in captured.out
    assert str(repo_root / "outputs") in captured.out
    assert str(repo_root / "cache" / "office_2" / "manifest.json") in captured.out


def test_tilde_paths_are_expanded_before_repo_relative_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    home = tmp_path / "home"
    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "REPO_ROOT", repo_root)
    monkeypatch.setenv("HOME", str(home))

    assert run_replica_all_scenes_fast_eval._repo_relative(Path("~/cache/{scene}")) == home / "cache/{scene}"


def test_main_dry_run_validates_paths_and_does_not_run_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    subprocess_calls = []
    monkeypatch.setattr(run_replica_all_scenes_fast_eval.subprocess, "run", lambda *args, **kwargs: subprocess_calls.append((args, kwargs)))
    _make_executable(tmp_path / "python")
    _write_manifest(tmp_path / "cache" / "manifest.json", "room0")
    _write_precomputed_config(tmp_path / "config.yaml", tmp_path / "cache" / "manifest.json")
    for path in [
        tmp_path / "runner.py",
        tmp_path / "Replica" / "room0",
        tmp_path / "labels" / "room0.txt",
        tmp_path / "Original" / "room_0" / "habitat" / "mesh_semantic.ply",
        tmp_path / "Original" / "room_0" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "room0",
            "--dry-run",
            "--python-executable",
            str(tmp_path / "python"),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert subprocess_calls == []
    assert "[dry-run] room0: batch_room0_s10_200f_fast_eval" in captured.out
    assert "CUDA_VISIBLE_DEVICES=0" in captured.out


def test_online_yoloworld_sam_mode_uses_sam2_config_and_skips_cache_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "REPO_ROOT", repo_root)
    subprocess_calls = []
    monkeypatch.setattr(
        run_replica_all_scenes_fast_eval.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    def fail_if_precomputed_cache_validation_runs(**kwargs):
        raise AssertionError("online SAM2 mode must not validate precomputed caches")

    monkeypatch.setattr(
        run_replica_all_scenes_fast_eval,
        "validate_precomputed_cache_for_scene",
        fail_if_precomputed_cache_validation_runs,
    )

    _make_executable(repo_root / "bin" / "python")
    for path in [
        repo_root / "scripts" / "run_room0_checkpointed_eval.py",
        repo_root / "configs" / "replica_yoloworld_sam_online_baseline_4090.yaml",
        repo_root / "Replica" / "room1",
        repo_root / "labels" / "room1.txt",
        repo_root / "Original" / "room_1" / "habitat" / "mesh_semantic.ply",
        repo_root / "Original" / "room_1" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "online_batch",
            "--scenes",
            "room1",
            "--dry-run",
            "--online-yoloworld-sam",
            "--python-executable",
            "bin/python",
            "--runner-script",
            "scripts/run_room0_checkpointed_eval.py",
            "--dataset-root-base",
            "Replica",
            "--gt-original-base",
            "Original",
            "--gt-label-dir",
            "labels",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert subprocess_calls == []
    assert "--proposal-backend sam2" in captured.out
    assert "--scene-name room1" in captured.out
    assert str(repo_root / "configs" / "replica_yoloworld_sam_online_baseline_4090.yaml") in captured.out
    assert "--proposal-cache-manifest" not in captured.out


def test_online_sam3_concept_mode_uses_sam3_config_and_skips_cache_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "REPO_ROOT", repo_root)
    subprocess_calls = []
    monkeypatch.setattr(
        run_replica_all_scenes_fast_eval.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    def fail_if_precomputed_cache_validation_runs(**kwargs):
        raise AssertionError("SAM3 mode must not validate precomputed caches")

    monkeypatch.setattr(
        run_replica_all_scenes_fast_eval,
        "validate_precomputed_cache_for_scene",
        fail_if_precomputed_cache_validation_runs,
    )

    _make_executable(repo_root / "bin" / "python")
    for path in [
        repo_root / "scripts" / "run_room0_checkpointed_eval.py",
        repo_root / "configs" / "replica_sam3_concept_room0_experiment_4090.yaml",
        repo_root / "Replica" / "room0",
        repo_root / "labels" / "room0.txt",
        repo_root / "Original" / "room_0" / "habitat" / "mesh_semantic.ply",
        repo_root / "Original" / "room_0" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "sam3_batch",
            "--scenes",
            "room0",
            "--dry-run",
            "--online-sam3-concept",
            "--python-executable",
            "bin/python",
            "--runner-script",
            "scripts/run_room0_checkpointed_eval.py",
            "--dataset-root-base",
            "Replica",
            "--gt-original-base",
            "Original",
            "--gt-label-dir",
            "labels",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert subprocess_calls == []
    assert "--proposal-backend sam3_concept" in captured.out
    assert "replica_sam3_concept_room0_experiment_4090.yaml" in captured.out
    assert "--proposal-cache-manifest" not in captured.out


def test_main_returns_nonzero_when_python_executable_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_manifest(tmp_path / "cache" / "manifest.json", "room0")
    _write_precomputed_config(tmp_path / "config.yaml", tmp_path / "cache" / "manifest.json")
    for path in [
        tmp_path / "runner.py",
        tmp_path / "Replica" / "room0",
        tmp_path / "labels" / "room0.txt",
        tmp_path / "Original" / "room_0" / "habitat" / "mesh_semantic.ply",
        tmp_path / "Original" / "room_0" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "room0",
            "--dry-run",
            "--python-executable",
            str(tmp_path / "missing_python"),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
        ]
    )

    assert result == 2
    assert f"room0: missing python executable: {tmp_path / 'missing_python'}" in capsys.readouterr().err


def test_main_requires_python_executable_to_be_file_and_executable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    python_dir = tmp_path / "python_dir"
    python_dir.mkdir()
    python_file = tmp_path / "python_file"
    _touch(python_file)
    _write_manifest(tmp_path / "cache" / "manifest.json", "room0")
    _write_precomputed_config(tmp_path / "config.yaml", tmp_path / "cache" / "manifest.json")
    for path in [
        tmp_path / "runner.py",
        tmp_path / "Replica" / "room0",
        tmp_path / "labels" / "room0.txt",
        tmp_path / "Original" / "room_0" / "habitat" / "mesh_semantic.ply",
        tmp_path / "Original" / "room_0" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result_dir = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "room0",
            "--dry-run",
            "--python-executable",
            str(python_dir),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
        ]
    )
    err_dir = capsys.readouterr().err

    result_file = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "room0",
            "--dry-run",
            "--python-executable",
            str(python_file),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
        ]
    )
    err_file = capsys.readouterr().err

    assert result_dir == 2
    assert result_file == 2
    assert f"room0: python executable is not a file: {python_dir}" in err_dir
    assert f"room0: python executable is not executable: {python_file}" in err_file


def test_precomputed_cache_manifest_scene_mismatch_fails_validation_for_non_room_scene(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_executable(tmp_path / "python")
    _write_manifest(tmp_path / "cache" / "manifest.json", "room0")
    _write_precomputed_config(tmp_path / "config.yaml", tmp_path / "cache" / "manifest.json")
    for path in [
        tmp_path / "runner.py",
        tmp_path / "Replica" / "room1",
        tmp_path / "labels" / "room1.txt",
        tmp_path / "Original" / "room_1" / "habitat" / "mesh_semantic.ply",
        tmp_path / "Original" / "room_1" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "room1",
            "--dry-run",
            "--python-executable",
            str(tmp_path / "python"),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
        ]
    )

    assert result == 2
    assert "room1: precomputed cache manifest scene mismatch" in capsys.readouterr().err


def test_per_scene_cache_manifest_template_expands_and_is_passed_to_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_executable(tmp_path / "python")
    _write_precomputed_config(tmp_path / "config.yaml")
    _write_manifest(tmp_path / "cache" / "office_1" / "manifest.json", "office1")
    for path in [
        tmp_path / "runner.py",
        tmp_path / "Replica" / "office1",
        tmp_path / "labels" / "office1.txt",
        tmp_path / "Original" / "office_1" / "habitat" / "mesh_semantic.ply",
        tmp_path / "Original" / "office_1" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "office1",
            "--dry-run",
            "--python-executable",
            str(tmp_path / "python"),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
            "--proposal-cache-manifest-template",
            str(tmp_path / "cache" / "{gt_scene}" / "manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "--proposal-cache-manifest" in captured.out
    assert str(tmp_path / "cache" / "office_1" / "manifest.json") in captured.out


def test_per_scene_cache_dir_template_expands_and_passes_derived_manifest_to_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_executable(tmp_path / "python")
    stale_manifest = tmp_path / "cache" / "room0" / "manifest.json"
    _write_manifest(stale_manifest, "room0")
    _write_precomputed_config(tmp_path / "config.yaml", stale_manifest)
    _write_manifest(tmp_path / "cache" / "office_1" / "manifest.json", "office1")
    for path in [
        tmp_path / "runner.py",
        tmp_path / "Replica" / "office1",
        tmp_path / "labels" / "office1.txt",
        tmp_path / "Original" / "office_1" / "habitat" / "mesh_semantic.ply",
        tmp_path / "Original" / "office_1" / "habitat" / "info_semantic.json",
    ]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "office1",
            "--dry-run",
            "--python-executable",
            str(tmp_path / "python"),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
            "--proposal-cache-dir-template",
            str(tmp_path / "cache" / "{gt_scene}"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "--proposal-cache-dir" in captured.out
    assert "--proposal-cache-manifest" in captured.out
    assert str(tmp_path / "cache" / "office_1") in captured.out
    assert str(tmp_path / "cache" / "office_1" / "manifest.json") in captured.out


def test_precomputed_cache_validation_requires_selected_frame_entries_and_files(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cache" / "manifest.json"
    _write_manifest(manifest_path, "room0", frame_ids=[0, 10], touch_files=True)

    errors = run_replica_all_scenes_fast_eval.validate_precomputed_cache_for_scene(
        scene="room0",
        config_path=tmp_path / "config.yaml",
        override=run_replica_all_scenes_fast_eval.ProposalCacheOverride(manifest_path=manifest_path),
        num_frames=3,
        frame_stride=10,
    )

    assert errors == [
        f"room0: precomputed cache manifest missing requested frame id 20: {manifest_path}"
    ]

    _write_manifest(manifest_path, "room0", frame_ids=[0, 10, 20], touch_files=False)
    _touch(manifest_path.parent / "frames" / "frame000000_proposals.npz")
    _touch(manifest_path.parent / "frames" / "frame000010_proposals.npz")

    errors = run_replica_all_scenes_fast_eval.validate_precomputed_cache_for_scene(
        scene="room0",
        config_path=tmp_path / "config.yaml",
        override=run_replica_all_scenes_fast_eval.ProposalCacheOverride(manifest_path=manifest_path),
        num_frames=3,
        frame_stride=10,
    )

    assert errors == [
        f"room0: missing precomputed cache frame file for frame 20: "
        f"{manifest_path.parent / 'frames' / 'frame000020_proposals.npz'}"
    ]


def test_main_skips_complete_scene_then_runs_incomplete_scene_with_env_and_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed: list[str] = []
    run_calls: list[dict[str, object]] = []

    def fake_scene_is_complete(_run_root: Path, scene: str, *, expected_frames: int) -> bool:
        completed.append(f"{scene}:{expected_frames}")
        return scene == "room0"

    def fake_run(command, *, cwd, env, check):
        run_calls.append({"command": command, "cwd": cwd, "env": env, "check": check})

    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_replica_all_scenes_fast_eval, "scene_is_complete", fake_scene_is_complete)
    monkeypatch.setattr(run_replica_all_scenes_fast_eval.subprocess, "run", fake_run)
    for scene, gt_scene in {"room0": "room_0", "room1": "room_1"}.items():
        for path in [
            tmp_path / "Replica" / scene,
            tmp_path / "labels" / f"{scene}.txt",
            tmp_path / "Original" / gt_scene / "habitat" / "mesh_semantic.ply",
            tmp_path / "Original" / gt_scene / "habitat" / "info_semantic.json",
        ]:
            _touch(path)
    _make_executable(tmp_path / "python")
    _write_manifest(tmp_path / "cache" / "room0" / "manifest.json", "room0")
    _write_manifest(tmp_path / "cache" / "room1" / "manifest.json", "room1")
    _write_precomputed_config(tmp_path / "config.yaml")
    for path in [tmp_path / "runner.py"]:
        _touch(path)

    result = run_replica_all_scenes_fast_eval.main(
        [
            "--batch-name",
            "batch",
            "--scenes",
            "room0",
            "room1",
            "--num-frames",
            "2",
            "--output-root",
            "outputs",
            "--python-executable",
            str(tmp_path / "python"),
            "--runner-script",
            str(tmp_path / "runner.py"),
            "--config-path",
            str(tmp_path / "config.yaml"),
            "--dataset-root-base",
            str(tmp_path / "Replica"),
            "--gt-original-base",
            str(tmp_path / "Original"),
            "--gt-label-dir",
            str(tmp_path / "labels"),
            "--proposal-cache-manifest-template",
            str(tmp_path / "cache" / "{scene}" / "manifest.json"),
        ]
    )

    assert result == 0
    assert completed == ["room0:2", "room1:2"]
    assert len(run_calls) == 1
    assert run_calls[0]["cwd"] == tmp_path
    assert run_calls[0]["check"] is True
    assert run_calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert run_calls[0]["env"]["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert run_calls[0]["env"]["HF_HUB_OFFLINE"] == "1"
    command = run_calls[0]["command"]
    assert command[command.index("--scene-name") + 1] == "room1"
    assert command[command.index("--output-root") + 1] == str(tmp_path / "outputs")
    assert command[command.index("--proposal-cache-manifest") + 1] == str(tmp_path / "cache" / "room1" / "manifest.json")


def test_summarize_batch_aggregates_fake_scene_outputs_and_records_missing(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room0",
        miou=0.2,
        macc=0.4,
        fmiou=0.6,
        fmacc=0.8,
        mapping_sec=10.0,
        export_sec=2.0,
        sec_per_frame=0.05,
        objects=7,
        dense_points=1000,
        prefetch_hit_count=8,
        prefetch_submitted_count=10,
        prefetch_miss_count=2,
    )
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "office1",
        miou=0.4,
        macc=0.6,
        fmiou=0.8,
        fmacc=1.0,
        mapping_sec=20.0,
        export_sec=4.0,
        sec_per_frame=0.15,
        objects=11,
        dense_points=2000,
        report_scene="room0",
    )

    summary = summarize_replica_all_scenes.summarize_batch(
        batch_name="batch",
        scenes=["room0", "office1", "office2"],
        output_root=output_root,
        frame_stride=10,
        num_frames=200,
    )

    assert [scene["scene"] for scene in summary["scenes"]] == ["room0", "office1"]
    assert summary["scenes"][0]["miou"] == 0.2
    assert summary["scenes"][1]["scene"] == "office1"
    assert summary["scenes"][1]["objects"] == 11
    assert summary["scenes"][1]["dense_points"] == 2000
    assert summary["aggregate"]["macro_average"] == {
        "miou": pytest.approx(0.3),
        "fmiou": pytest.approx(0.7),
        "macc": pytest.approx(0.5),
        "fmacc": pytest.approx(0.9),
    }
    assert summary["aggregate"]["mean_tpf"] == pytest.approx(0.1)
    assert summary["aggregate"]["total_mapping_sec"] == pytest.approx(30.0)
    assert summary["aggregate"]["total_export_eval_sec"] == pytest.approx(6.0)
    assert summary["aggregate"]["object_count_stats"] == {
        "min": 7,
        "max": 11,
        "mean": pytest.approx(9.0),
        "total": 18,
    }
    assert summary["missing_scenes"] == [{"scene": "office2", "reason": "missing run root"}]


def test_write_summary_emits_json_and_markdown_with_requested_columns(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    summary_root = tmp_path / "summaries" / "batch"
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room0",
        miou=0.25,
        macc=0.5,
        fmiou=0.75,
        fmacc=1.0,
        mapping_sec=12.0,
        export_sec=3.0,
        sec_per_frame=0.06,
        objects=5,
        dense_points=900,
        prefetch_hit_count=4,
        prefetch_submitted_count=5,
        prefetch_miss_count=1,
    )
    summary = summarize_replica_all_scenes.summarize_batch(
        batch_name="batch",
        scenes=["room0"],
        output_root=output_root,
        frame_stride=10,
        num_frames=200,
    )

    json_path, markdown_path = summarize_replica_all_scenes.write_summary(summary, summary_root)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert loaded["batch_name"] == "batch"
    assert loaded["scenes"][0]["prefetch"] == "4/5 hits, 1 misses"
    assert "| scene | mIoU | f-mIoU | mAcc | f-mAcc | TPF | mapping_sec | export_sec | objects | dense_points | prefetch |" in markdown
    assert "| room0 | 0.250 | 0.750 | 0.500 | 1.000 | 0.060 | 12.000 | 3.000 | 5 | 900 | 4/5 hits, 1 misses |" in markdown
    assert "## Aggregate" in markdown


def test_summarizer_script_runs_directly_without_pythonpath(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "summarize_replica_all_scenes.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--batch-name" in result.stdout


def test_summarize_batch_excludes_incomplete_scenes(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room0",
        miou=0.2,
        macc=0.4,
        fmiou=0.6,
        fmacc=0.8,
        mapping_sec=10.0,
        export_sec=2.0,
        sec_per_frame=0.05,
        objects=7,
        dense_points=1000,
    )
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room1",
        miou=0.9,
        macc=0.9,
        fmiou=0.9,
        fmacc=0.9,
        mapping_sec=99.0,
        export_sec=99.0,
        sec_per_frame=0.99,
        objects=99,
        dense_points=99,
        status="failed",
    )
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room2",
        miou=0.8,
        macc=0.8,
        fmiou=0.8,
        fmacc=0.8,
        mapping_sec=88.0,
        export_sec=88.0,
        sec_per_frame=0.88,
        objects=88,
        dense_points=88,
        processed_frames=199,
    )
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "office0",
        miou=0.7,
        macc=0.7,
        fmiou=0.7,
        fmacc=0.7,
        mapping_sec=77.0,
        export_sec=77.0,
        sec_per_frame=0.77,
        objects=77,
        dense_points=77,
        frame_metric_lines=199,
    )

    summary = summarize_replica_all_scenes.summarize_batch(
        batch_name="batch",
        scenes=["room0", "room1", "room2", "office0"],
        output_root=output_root,
        frame_stride=10,
        num_frames=200,
    )

    assert [scene["scene"] for scene in summary["scenes"]] == ["room0"]
    assert summary["aggregate"]["total_mapping_sec"] == pytest.approx(10.0)
    reasons = {scene["scene"]: scene["reason"] for scene in summary["failed_scenes"]}
    assert "status is not complete" in reasons["room1"]
    assert "processed_frames 199 != expected 200" in reasons["room2"]
    assert "frame_metrics.jsonl line count 199 != expected 200" in reasons["office0"]


def test_summarize_batch_rejects_non_finite_metrics_and_missing_counts(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room0",
        miou=0.2,
        macc=0.4,
        fmiou=0.6,
        fmacc=0.8,
        mapping_sec=10.0,
        export_sec=2.0,
        sec_per_frame=0.05,
        objects=7,
        dense_points=1000,
    )
    _write_json(
        output_root / "batch_room0_s10_200f_fast_eval" / "replica" / "results.json",
        {"miou": float("nan"), "macc": 0.4, "fmiou": 0.6, "fmacc": 0.8},
    )
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room1",
        miou=0.3,
        macc=0.5,
        fmiou=0.7,
        fmacc=0.9,
        mapping_sec=11.0,
        export_sec=3.0,
        sec_per_frame=0.06,
        objects=0,
        dense_points=0,
        include_counts=False,
    )

    with pytest.raises(RuntimeError, match="No summarizable scenes"):
        summarize_replica_all_scenes.summarize_batch(
            batch_name="batch",
            scenes=["room0", "room1"],
            output_root=output_root,
            frame_stride=10,
            num_frames=200,
        )


def test_summarizer_cli_normalizes_relative_paths_from_outside_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    shutil.copytree(Path(__file__).resolve().parent.parent / "scripts", repo_root / "scripts")
    output_root = repo_root / "relative_outputs"
    _write_fake_scene_outputs(
        output_root,
        "batch",
        "room0",
        miou=0.25,
        macc=0.5,
        fmiou=0.75,
        fmacc=1.0,
        mapping_sec=12.0,
        export_sec=3.0,
        sec_per_frame=0.06,
        objects=5,
        dense_points=900,
    )
    (tmp_path / "outside").mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_replica_all_scenes.py"),
            "--batch-name",
            "batch",
            "--scenes",
            "room0",
            "--output-root",
            "relative_outputs",
            "--summary-root",
            "relative_summary",
        ],
        cwd=tmp_path / "outside",
        text=True,
        capture_output=True,
        check=False,
    )

    summary_path = repo_root / "relative_summary" / "summary.json"
    assert result.returncode == 0
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["output_root"] == str(output_root)
    assert loaded["scenes"][0]["scene"] == "room0"


def test_summarizer_cli_returns_nonzero_when_no_scenes_are_summarizable(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "scripts" / "summarize_replica_all_scenes.py"),
            "--batch-name",
            "missing_batch",
            "--scenes",
            "room0",
            "--output-root",
            str(tmp_path / "outputs"),
            "--summary-root",
            str(tmp_path / "summary"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "No summarizable scenes found" in result.stderr
    assert not (tmp_path / "summary" / "summary.json").exists()
