from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluation.build_tesse_ovimap_static_anchor import (
    AnchorPackagePaths,
    PINNED_OVIMAP_COMMIT,
    TesseNativeEnvironment,
    build_anchor_package,
    build_tesse_native_commands,
    load_causal_prefix_contract,
    main,
    preflight_tesse_native,
    run_tesse_native_mapping,
)
from src.evaluation.exporters.oviovo import read_map_snapshot


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "relative_script",
    (
        "scripts/evaluation/build_tesse_ovimap_static_anchor.py",
        "scripts/evaluation/run_crove_ovimap_static_anchor.py",
        "scripts/evaluation/evaluate_crove_ovimap_static_anchor.py",
    ),
)
def test_static_anchor_clis_bootstrap_repo_imports_from_other_working_directory(
    tmp_path: Path, relative_script: str
) -> None:
    script = REPO_ROOT / relative_script
    code = (
        "import runpy,sys; "
        f"ns=runpy.run_path({str(script)!r}, run_name='static_anchor_cli_test'); "
        "assert str(ns['REPO_ROOT']) in sys.path"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _write_schedule(
    root: Path,
    *,
    scene: str = "apartment",
    interventions: tuple[int, ...] = (3, 7),
    event_ids: tuple[str, ...] | None = None,
) -> Path:
    identifiers = event_ids or tuple(
        f"{scene}_event_{index:02d}"
        for index in range(1, len(interventions) + 1)
    )
    return _write_json(
        root / "schedule.json",
        {
            "dataset": "TESSE-CD",
            "manifest_id": "fixture_schedule",
            "scenes": {
                scene: {
                    "events": [
                        {
                            "event_id": event_id,
                            "intervention_frame_index": frame_index,
                        }
                        for event_id, frame_index in zip(
                            identifiers, interventions, strict=True
                        )
                    ]
                }
            },
            "schema_version": 1,
        },
    )


def _write_rgbd(root: Path, *, scene: str = "apartment", frame_count: int) -> Path:
    rgbd_root = root / "rgbd_v1"
    scene_root = rgbd_root / scene
    results = scene_root / "results"
    results.mkdir(parents=True)
    for frame_index in range(frame_count):
        (results / f"frame{frame_index:06d}.jpg").write_bytes(
            f"rgb:{frame_index}".encode("ascii")
        )
        (results / f"depth{frame_index:06d}.png").write_bytes(
            f"depth:{frame_index}".encode("ascii")
        )
    _write_json(
        scene_root / "export_manifest.json",
        {
            "dataset": "TESSE-CD",
            "frame_count": frame_count,
            "scene": scene,
            "schema_version": 1,
        },
    )
    return rgbd_root


def test_prefix_stops_before_first_intervention(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, interventions=(3, 7))
    rgbd_root = _write_rgbd(tmp_path, frame_count=8)

    contract = load_causal_prefix_contract(
        scene="apartment",
        schedule_path=schedule,
        rgbd_root=rgbd_root,
        configured_cutoff=2,
    )

    assert contract.scene == "apartment"
    assert contract.first_intervention_frame == 3
    assert contract.frame_ids == (0, 1, 2)
    assert contract.maximum_source_frame == 2
    assert contract.schedule_sha256 == _sha256(schedule)
    assert contract.rgbd_export_manifest_sha256 == _sha256(
        rgbd_root / "apartment" / "export_manifest.json"
    )


def test_prefix_rejects_intervention_frame(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, interventions=(3,))
    rgbd_root = _write_rgbd(tmp_path, frame_count=4)

    with pytest.raises(ValueError, match="strictly before first intervention"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=3,
        )


def test_prefix_rejects_scene_not_declared_by_schedule(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, scene="office", interventions=(3,))
    rgbd_root = _write_rgbd(tmp_path, scene="apartment", frame_count=4)

    with pytest.raises(ValueError, match="scene is not declared"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=2,
        )


@pytest.mark.parametrize("missing_name", ("frame000001.jpg", "depth000001.png"))
def test_prefix_rejects_incomplete_rgbd_inventory(
    tmp_path: Path, missing_name: str
) -> None:
    schedule = _write_schedule(tmp_path, interventions=(3,))
    rgbd_root = _write_rgbd(tmp_path, frame_count=4)
    (rgbd_root / "apartment" / "results" / missing_name).unlink()

    with pytest.raises(ValueError, match="causal RGB-D frame is missing"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=2,
        )


def test_prefix_rejects_duplicate_event_ids(tmp_path: Path) -> None:
    schedule = _write_schedule(
        tmp_path,
        interventions=(3, 7),
        event_ids=("duplicate", "duplicate"),
    )
    rgbd_root = _write_rgbd(tmp_path, frame_count=8)

    with pytest.raises(ValueError, match="event IDs must be unique"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=2,
        )


def _touch(path: Path, content: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _native_environment(root: Path) -> TesseNativeEnvironment:
    build_root = root / "native"
    entity_root = build_root / "Entity"
    ovimap_root = build_root / "OVI-MAP"
    frontend_python = _touch(build_root / "cropformer-env" / "bin" / "python")
    mapping_python = _touch(build_root / "map-env" / "bin" / "python")
    _touch(
        entity_root
        / "Entityv2"
        / "CropFormer"
        / "configs"
        / "entityv2"
        / "entity_segmentation"
        / "cropformer_hornet_3x.yaml"
    )
    _touch(
        entity_root
        / "Entityv2"
        / "CropFormer"
        / "demo_cropformer"
        / "demo_from_dirs.py"
    )
    _touch(ovimap_root / "scripts" / "panoptic_mapping_.py")
    _touch(
        ovimap_root
        / "mapping_ros_ws"
        / "devel"
        / "lib"
        / "consistent_gsm.cpython-311-x86_64-linux-gnu.so"
    )
    _touch(
        ovimap_root
        / "mapping_ros_ws"
        / "devel"
        / "lib"
        / "depth_segmentation_py.cpython-311-x86_64-linux-gnu.so"
    )
    cropformer_weights = _touch(build_root / "weights" / "cropformer.pth")
    siglip_model = build_root / "weights" / "siglip"
    siglip_model.mkdir(parents=True)
    _touch(siglip_model / "config.json", b"{}\n")
    source_hashes = _touch(build_root / "environment" / "source-hashes.json")
    return TesseNativeEnvironment(
        build_root=build_root,
        frontend_python=frontend_python,
        mapping_python=mapping_python,
        entity_root=entity_root,
        ovimap_root=ovimap_root,
        cropformer_weights=cropformer_weights,
        siglip_model=siglip_model,
        source_hashes=source_hashes,
    )


def _write_native_rgbd(root: Path, *, frame_count: int = 3) -> Path:
    rgbd_root = _write_rgbd(root, frame_count=frame_count)
    _touch(rgbd_root / "cam_params.json", b"{}\n")
    _touch(rgbd_root / "apartment" / "traj.txt", b"pose\n")
    return rgbd_root


def test_native_commands_use_exact_causal_tesse_frames(tmp_path: Path) -> None:
    rgbd_root = _write_native_rgbd(tmp_path)
    native = _native_environment(tmp_path)

    commands = build_tesse_native_commands(
        scene="apartment",
        rgbd_root=rgbd_root,
        frame_ids=(0, 1, 2),
        native=native,
        attempt_root=tmp_path / "attempt",
    )

    assert commands.frame_ids == (0, 1, 2)
    assert commands.mapping[commands.mapping.index("--scene_num") + 1] == "apartment"
    assert commands.mapping[commands.mapping.index("--data_folder") + 1] == str(
        rgbd_root
    )
    assert commands.mapping[commands.mapping.index("--start") + 1] == "0"
    assert commands.mapping[commands.mapping.index("--end") + 1] == "3"
    assert commands.mapping[commands.mapping.index("--step") + 1] == "1"
    input_index = commands.frontend.index("--input") + 1
    output_index = commands.frontend.index("--output")
    assert commands.frontend[input_index:output_index] == tuple(
        str(rgbd_root / "apartment" / "results" / f"frame{index:06d}.jpg")
        for index in range(3)
    )


def test_native_commands_reject_noncontiguous_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contiguous and start at zero"):
        build_tesse_native_commands(
            scene="apartment",
            rgbd_root=tmp_path / "rgbd",
            frame_ids=(0, 2),
            native=_native_environment(tmp_path),
            attempt_root=tmp_path / "attempt",
        )


def test_native_commands_reject_output_inside_rgbd_source(tmp_path: Path) -> None:
    rgbd_root = _write_native_rgbd(tmp_path)
    with pytest.raises(ValueError, match="outside RGB-D source"):
        build_tesse_native_commands(
            scene="apartment",
            rgbd_root=rgbd_root,
            frame_ids=(0, 1, 2),
            native=_native_environment(tmp_path),
            attempt_root=rgbd_root / "attempt",
        )


def test_native_preflight_binds_inputs_and_pinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_tesse_ovimap_static_anchor as module

    rgbd_root = _write_native_rgbd(tmp_path)
    native = _native_environment(tmp_path)
    commands = build_tesse_native_commands(
        scene="apartment",
        rgbd_root=rgbd_root,
        frame_ids=(0, 1, 2),
        native=native,
        attempt_root=tmp_path / "attempt",
    )
    monkeypatch.setattr(module, "_git_head", lambda path: PINNED_OVIMAP_COMMIT)

    record = preflight_tesse_native(
        scene="apartment",
        rgbd_root=rgbd_root,
        commands=commands,
        native=native,
    )

    assert record["status"] == "PASS"
    assert record["ovimap_commit"] == PINNED_OVIMAP_COMMIT
    assert record["frame_ids"] == [0, 1, 2]
    assert set(record["sources"]) == {
        "camera",
        "cropformer_config",
        "cropformer_demo",
        "cropformer_weights",
        "depth_segmentation_extension",
        "frontend_python",
        "geometry_helper",
        "mapping_python",
        "mapper",
        "siglip_model",
        "source_hashes",
        "consistent_gsm_extension",
        "trajectory",
    }
    assert len(record["frames"]) == 3


def test_native_preflight_rejects_non_pinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_tesse_ovimap_static_anchor as module

    rgbd_root = _write_native_rgbd(tmp_path)
    native = _native_environment(tmp_path)
    commands = build_tesse_native_commands(
        scene="apartment",
        rgbd_root=rgbd_root,
        frame_ids=(0, 1, 2),
        native=native,
        attempt_root=tmp_path / "attempt",
    )
    monkeypatch.setattr(module, "_git_head", lambda path: "f" * 40)

    with pytest.raises(ValueError, match="pinned OVI-MAP commit"):
        preflight_tesse_native(
            scene="apartment",
            rgbd_root=rgbd_root,
            commands=commands,
            native=native,
        )


@pytest.mark.parametrize(
    "missing",
    (
        Path("apartment/traj.txt"),
        Path("apartment/results/depth000001.png"),
    ),
)
def test_native_preflight_rejects_missing_inputs(
    tmp_path: Path, missing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_tesse_ovimap_static_anchor as module

    rgbd_root = _write_native_rgbd(tmp_path)
    native = _native_environment(tmp_path)
    commands = build_tesse_native_commands(
        scene="apartment",
        rgbd_root=rgbd_root,
        frame_ids=(0, 1, 2),
        native=native,
        attempt_root=tmp_path / "attempt",
    )
    (rgbd_root / missing).unlink()
    monkeypatch.setattr(module, "_git_head", lambda path: PINNED_OVIMAP_COMMIT)

    with pytest.raises(ValueError, match="missing"):
        preflight_tesse_native(
            scene="apartment",
            rgbd_root=rgbd_root,
            commands=commands,
            native=native,
        )


def test_native_execution_publishes_only_after_all_stages_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_tesse_ovimap_static_anchor as module

    rgbd_root = _write_native_rgbd(tmp_path)
    native = _native_environment(tmp_path)
    commands = build_tesse_native_commands(
        scene="apartment",
        rgbd_root=rgbd_root,
        frame_ids=(0, 1, 2),
        native=native,
        attempt_root=tmp_path / "attempt",
    )
    monkeypatch.setattr(module, "_git_head", lambda path: PINNED_OVIMAP_COMMIT)
    executed: list[tuple[str, ...]] = []

    def fake_run(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        log: Path,
    ) -> dict[str, object]:
        del cwd, env
        executed.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        content = "stage pass\n"
        if argv == commands.frontend:
            for frame_id in commands.frame_ids:
                _touch(commands.attempt_root / "frontend" / f"frame{frame_id:06d}.png")
        if argv == commands.mapping:
            output = commands.attempt_root / "mapping" / "cropformer_inst"
            _touch(output / "instance_mesh_3.ply", b"ply\n")
            _touch(
                output / "inst_sem_siglip-l-16-384_3_incre_combine.pkl",
                b"pickle-fixture",
            )
            content += "Instance: 1 Color: (10,20,30)\n"
        log.write_text(content, encoding="utf-8")
        return {"argv": list(argv), "exit_code": 0, "log": str(log)}

    monkeypatch.setattr(module, "_run_native_command", fake_run)

    manifest_path = run_tesse_native_mapping(
        scene="apartment",
        rgbd_root=rgbd_root,
        commands=commands,
        native=native,
    )

    assert executed == [commands.geometry, commands.frontend, commands.mapping]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["state"] == "MAPPING_PASS"
    assert manifest["frame_ids"] == [0, 1, 2]
    assert set(manifest["artifacts"]) == {
        "instance_color_log",
        "instance_mesh",
        "semantic_features",
    }
    assert manifest["preflight"] == manifest["postflight"]


def test_native_execution_does_not_publish_after_stage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_tesse_ovimap_static_anchor as module

    rgbd_root = _write_native_rgbd(tmp_path)
    native = _native_environment(tmp_path)
    commands = build_tesse_native_commands(
        scene="apartment",
        rgbd_root=rgbd_root,
        frame_ids=(0, 1, 2),
        native=native,
        attempt_root=tmp_path / "attempt",
    )
    monkeypatch.setattr(module, "_git_head", lambda path: PINNED_OVIMAP_COMMIT)

    def fail_run(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        log: Path,
    ) -> dict[str, object]:
        del argv, cwd, env, log
        raise RuntimeError("frontend failed")

    monkeypatch.setattr(module, "_run_native_command", fail_run)

    with pytest.raises(RuntimeError, match="frontend failed"):
        run_tesse_native_mapping(
            scene="apartment",
            rgbd_root=rgbd_root,
            commands=commands,
            native=native,
        )
    assert not (commands.attempt_root / "native_mapping_manifest.json").exists()


def _write_anchor_native_fixture(root: Path) -> dict[str, Path]:
    attempt = root / "native-attempt"
    mesh = attempt / "mapping" / "cropformer_inst" / "instance_mesh_3.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 5",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 0.0 10 20 30",
                "0.1 0.0 0.0 10 20 30",
                "2.0 0.0 0.0 40 50 60",
                "2.1 0.0 0.0 40 50 60",
                "4.0 0.0 0.0 200 200 200",
            )
        )
        + "\n",
        encoding="ascii",
    )
    semantic_features = (
        attempt
        / "mapping"
        / "cropformer_inst"
        / "inst_sem_siglip-l-16-384_3_incre_combine.pkl"
    )
    with semantic_features.open("wb") as handle:
        pickle.dump(
            {
                1: {
                    "feat": np.asarray([[1.0, 0.0]], dtype=np.float32),
                    "vis_area": [1.0],
                    "frame_id": [0, 1, 2],
                }
            },
            handle,
            protocol=5,
        )
    color_log = _touch(
        attempt / "logs" / "mapping.log",
        (
            "Instance: 1 Color: (10,20,30)\n"
            "Instance: 2 Color: (40,50,60)\n"
        ).encode("ascii"),
    )
    native_manifest = _write_json(
        attempt / "native_mapping_manifest.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "MAPPING_PASS",
            "scene": "apartment",
            "frame_ids": [0, 1, 2],
            "preflight": {"ovimap_commit": PINNED_OVIMAP_COMMIT},
            "artifacts": {
                "instance_mesh": {
                    "path": str(mesh.absolute()),
                    "sha256": _sha256(mesh),
                    "byte_count": mesh.stat().st_size,
                },
                "semantic_features": {
                    "path": str(semantic_features.absolute()),
                    "sha256": _sha256(semantic_features),
                    "byte_count": semantic_features.stat().st_size,
                },
                "instance_color_log": {
                    "path": str(color_log.absolute()),
                    "sha256": _sha256(color_log),
                    "byte_count": color_log.stat().st_size,
                },
            },
        },
    )
    vocabulary = _write_json(
        root / "vocabulary.json",
        {
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "schema_version": 1,
            "classes": ["Chair", "Table"],
        },
    )
    siglip_model = root / "siglip"
    siglip_model.mkdir()
    _touch(siglip_model / "config.json", b"{}\n")
    native_payload = json.loads(native_manifest.read_text(encoding="utf-8"))
    native_payload["preflight"]["sources"] = {
        "siglip_model": {"path": str(siglip_model.absolute())}
    }
    _write_json(native_manifest, native_payload)
    return {
        "native_manifest": native_manifest,
        "mesh": mesh,
        "semantic_features": semantic_features,
        "color_log": color_log,
        "vocabulary": vocabulary,
        "siglip_model": siglip_model,
    }


def _anchor_causal_sources(root: Path) -> dict[str, dict[str, object]]:
    paths = {
        "config": _touch(root / "causal-sources" / "config.json", b"{}\n"),
        "schedule": _touch(root / "causal-sources" / "schedule.json", b"{}\n"),
        "rgbd_export_manifest": _touch(
            root / "causal-sources" / "export_manifest.json", b"{}\n"
        ),
    }
    return {
        role: {
            "path": str(path.absolute()),
            "sha256": _sha256(path),
            "byte_count": path.stat().st_size,
        }
        for role, path in paths.items()
    }


def test_build_anchor_binds_mesh_instances_and_hashes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_tesse_ovimap_static_anchor as module

    inputs = _write_anchor_native_fixture(tmp_path)

    def encode(values: tuple[str, ...], model: Path, device: str) -> np.ndarray:
        assert model == inputs["siglip_model"]
        assert device == "cpu"
        if values == ("Chair", "Table"):
            return np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
        assert values == ("object", "things", "stuff", "texture")
        return np.asarray(
            ((0.7, 0.7), (0.8, 0.6), (0.6, 0.8), (0.5, 0.5)),
            dtype=np.float32,
        )

    monkeypatch.setattr(module, "_encode_text_features", encode)

    result = build_anchor_package(
        scene="apartment",
        cutoff_frame=2,
        native_manifest=inputs["native_manifest"],
        instance_mesh=inputs["mesh"],
        semantic_features=inputs["semantic_features"],
        instance_color_log=inputs["color_log"],
        vocabulary_json=inputs["vocabulary"],
        siglip_model=inputs["siglip_model"],
        device="cpu",
        output_root=tmp_path / "anchor",
        causal_sources=_anchor_causal_sources(tmp_path),
    )

    assert isinstance(result, AnchorPackagePaths)
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    snapshot = read_map_snapshot(result.snapshot, result.entities)
    assert [entity.entity_id for entity in snapshot.entities] == [
        "ovimap:1",
        "ovimap:2",
    ]
    assert snapshot.entities[0].semantic_label == "Chair"
    assert snapshot.entities[0].metadata["authority"] == "ovimap_anchor"
    assert snapshot.entities[1].semantic_embedding is None
    assert snapshot.entities[1].semantic_label is None
    assert np.array_equal(
        snapshot.background_xyz,
        np.asarray(((4.0, 0.0, 0.0),), dtype=np.float32),
    )
    assert manifest["status"] == "PASS"
    assert manifest["causality"] == {
        "first_source_frame": 0,
        "last_source_frame": 2,
        "maximum_source_frame": 2,
        "strictly_pre_intervention": True,
    }
    assert manifest["outputs"]["snapshot"]["sha256"] == _sha256(result.snapshot)
    assert manifest["outputs"]["entities"]["sha256"] == _sha256(result.entities)

    inputs["semantic_features"].unlink()
    loaded_without_pickle = read_map_snapshot(result.snapshot, result.entities)
    assert [entity.entity_id for entity in loaded_without_pickle.entities] == [
        "ovimap:1",
        "ovimap:2",
    ]


def test_build_anchor_rejects_native_artifact_hash_mismatch(tmp_path: Path) -> None:
    inputs = _write_anchor_native_fixture(tmp_path)
    inputs["mesh"].write_bytes(inputs["mesh"].read_bytes() + b"\n")

    with pytest.raises(ValueError, match="native artifact binding mismatch"):
        build_anchor_package(
            scene="apartment",
            cutoff_frame=2,
            native_manifest=inputs["native_manifest"],
            instance_mesh=inputs["mesh"],
            semantic_features=inputs["semantic_features"],
            instance_color_log=inputs["color_log"],
            vocabulary_json=inputs["vocabulary"],
            siglip_model=inputs["siglip_model"],
            device="cpu",
            output_root=tmp_path / "anchor",
            causal_sources=_anchor_causal_sources(tmp_path),
        )


def test_cli_validates_prefix_and_builds_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_tesse_ovimap_static_anchor as module

    inputs = _write_anchor_native_fixture(tmp_path)
    schedule = _write_schedule(tmp_path / "causal", interventions=(3,))
    rgbd_root = _write_rgbd(tmp_path / "causal", frame_count=3)
    config = _write_json(
        tmp_path / "config.json",
        {
            "schema_version": 1,
            "method_id": "crove_ovimap_static_anchor_v1",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "source_role": "causal_pre_intervention_initialization",
            "first_source_frame": 0,
            "last_source_frame": 2,
            "source_stride": 1,
            "ovimap_commit": PINNED_OVIMAP_COMMIT,
            "vocabulary_json": str(inputs["vocabulary"]),
            "minimum_spatial_iou": 0.01,
            "maximum_centroid_distance_m": 0.75,
            "minimum_semantic_cosine": 0.65,
            "moved_displacement_m": 0.2,
            "background_voxel_size_m": 0.05,
        },
    )

    def encode(values: tuple[str, ...], model: Path, device: str) -> np.ndarray:
        del model, device
        if values == ("Chair", "Table"):
            return np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
        return np.asarray(
            ((0.7, 0.7), (0.8, 0.6), (0.6, 0.8), (0.5, 0.5)),
            dtype=np.float32,
        )

    monkeypatch.setattr(module, "_encode_text_features", encode)
    output = tmp_path / "anchor-cli"

    return_code = main(
        [
            "--config",
            str(config),
            "--schedule",
            str(schedule),
            "--rgbd-root",
            str(rgbd_root),
            "--native-manifest",
            str(inputs["native_manifest"]),
            "--output-root",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert return_code == 0
    assert json.loads((output / "anchor_manifest.json").read_text())["status"] == "PASS"
