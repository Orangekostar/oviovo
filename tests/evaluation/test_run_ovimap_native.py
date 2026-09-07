from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.evaluation.run_ovimap_native import (
    GateFailure,
    NativeConfig,
    SceneCommands,
    _frontend_diagnostic_name,
    _frontend_mask_name,
    _mapping_artifacts,
    advance_state,
    audit_scene,
    build_scene_commands,
    default_config,
    preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ENV = REPO_ROOT / "configs/environments/ovimap_cropformer.yaml"
MAPPING_ENV = REPO_ROOT / "configs/environments/ovimap_map.yaml"
BOOTSTRAP = REPO_ROOT / "scripts/reproduction/ovimap/bootstrap_native_envs.sh"
GXX_WRAPPER = REPO_ROOT / "scripts/reproduction/ovimap/conda_gxx12_wrapper.sh"
CROPFORMER_PATCH = REPO_ROOT / "patches/cropformer/ubuntu24-torch21.patch"
ENTITY_ROOT = Path("/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/Entity")
OVIMAP_PATCH = REPO_ROOT / "patches/ovimap/ubuntu24-native.patch"
OVIMAP_ROOT = Path("/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP")


@pytest.fixture
def native_config(tmp_path: Path) -> NativeConfig:
    return NativeConfig(
        build_root=tmp_path / "build",
        run_root=tmp_path / "run",
        data_root=tmp_path / "Replica",
        frontend_python=tmp_path / "ovimap-cropformer" / "bin" / "python",
        mapping_python=tmp_path / "ovimap-map" / "bin" / "python",
        entity_root=tmp_path / "Entity",
        ovimap_root=tmp_path / "OVI-MAP",
        cropformer_weights=tmp_path / "CropFormer_hornet.pth",
        siglip_model=tmp_path / "siglip-large-patch16-384",
    )


def test_default_config_uses_weight_from_frozen_native_build(tmp_path: Path) -> None:
    config = default_config(tmp_path / "run")

    assert config.cropformer_weights == (
        config.entity_root / "checkpoints/CropFormer_hornet_3x_03823a.pth"
    )
    assert config.siglip_model == config.build_root / "siglip-large-patch16-384"


def _dependencies(path: Path) -> tuple[dict, set[str]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    dependencies = {
        value for value in document["dependencies"] if isinstance(value, str)
    }
    return document, dependencies


def test_frontend_environment_pins_official_runtime() -> None:
    document, dependencies = _dependencies(FRONTEND_ENV)

    assert document["name"] == "ovimap-cropformer"
    assert document["channels"] == ["pytorch", "nvidia", "conda-forge", "nodefaults"]
    assert {
        "python=3.8.10",
        "pytorch=2.1.1",
        "torchvision=0.16.1",
        "pytorch-cuda=12.1",
        "cuda-version=12.1",
        "cuda-nvcc=12.1.105",
        "cuda-cudart-dev=12.1.105",
        "cuda-cccl=12.1.109",
        "cuda-libraries-dev=12.1.0",
        "gcc_linux-64=12",
        "gxx_linux-64=12",
        "ninja",
        "pip",
    } <= dependencies
    pip_section = next(
        value["pip"]
        for value in document["dependencies"]
        if isinstance(value, dict) and "pip" in value
    )
    assert {
        "opencv-contrib-python==4.11.*",
        "cython",
        "scipy",
        "shapely",
        "timm",
        "h5py",
        "submitit",
        "scikit-image",
    } <= set(pip_section)


def test_mapping_environment_pins_robostack_abi() -> None:
    document, dependencies = _dependencies(MAPPING_ENV)

    assert document["name"] == "ovimap-map"
    assert document["channels"] == [
        "robostack-noetic",
        "pytorch",
        "nvidia",
        "conda-forge",
        "nodefaults",
    ]
    assert {
        "python=3.11",
        "numpy=1.26",
        "pytorch=2.1.1",
        "torchvision=0.16.1",
        "pytorch-cuda=12.1",
        "ros-noetic-ros-base=1.5.0",
        "ros-noetic-pcl-ros=1.7.4",
        "ros-noetic-eigen-conversions=1.13.2",
        "ros-noetic-tf-conversions=1.13.2",
        "catkin_tools",
        "cmake",
        "make",
        "pybind11",
        "protobuf",
        "glog",
        "gflags",
        "opencv=4.11",
        "pip",
    } <= dependencies
    pip_section = next(
        value["pip"]
        for value in document["dependencies"]
        if isinstance(value, dict) and "pip" in value
    )
    assert {
        "numpy==1.26.4",
        "transformers==4.49.0",
        "accelerate==0.26.*",
        "open3d",
        "plyfile==1.1.3",
        "sentencepiece",
    } <= set(pip_section)
    assert not any(value.startswith("opencv-") for value in pip_section)


def test_bootstrap_freezes_sources_and_environment_provenance() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")

    assert "58a804e2d7c82ba05a489eb071aba3367301fed8" in script
    assert "6e7e13ac91ef508088e1b848167c01f19b00b512" in script
    assert "d1e04565d3bec8719335b88be9e9b961bf3ec464" in script
    assert "https://github.com/OVI-MAP/OVI-MAP.git" in script
    assert "https://github.com/qqlu/Entity.git" in script
    assert "https://github.com/facebookresearch/detectron2.git" in script
    for commit in (
        "b2e87786e77b8c9ce71e56ce2a1146bd32b7c68c",
        "c8ea6e25bc5d98772d57b2ffd7619973952a6839",
        "0e62848b12da76c8cc58a1add42b4f894d1ac21e",
        "3323b388540fa95ec9da6f9cd887f70ead055edb",
        "22a6247a3df11bc285d43d1a030f4e874a413997",
        "fc38fc525f7d48881aebb27a7b9978453556bbd4",
        "40a9edadd15c59f8b57dc947d0135b0a007ea10b",
        "564f12639a8447d4d3e5e7707851424302941056",
        "5528b042124fe056a7cf53f96c8b39e1e32ec2b9",
        "f63b1dfe3b0a1ee21138caa1dcedd32c7f0411d9",
        "721a6cc17e7e937e7accfb88a9967b26134db6f9",
        "6aff4c33a0d79536dd769d176ee5cd1285004c88",
    ):
        assert commit in script
    for filename in (
        "conda-explicit.txt",
        "pip-freeze.txt",
        "source-hashes.json",
        "nvidia-smi.txt",
        "host.json",
    ):
        assert filename in script
    assert "mv \"$temporary_root\" \"$build_root\"" in script


def test_bootstrap_refuses_existing_root_before_conda(tmp_path: Path) -> None:
    root = tmp_path / "existing"
    root.mkdir()

    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), str(root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "already exists" in completed.stderr
    assert list(root.iterdir()) == []


def test_gxx_wrapper_only_removes_broken_python_sysroot(tmp_path: Path) -> None:
    compiler = tmp_path / "compiler"
    compiler.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8"
    )
    compiler.chmod(0o755)

    completed = subprocess.run(
        [str(GXX_WRAPPER), "keep-before", "-Wl,--sysroot=/", "keep-after"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "OVIMAP_CXX_REAL": str(compiler)},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["keep-before", "keep-after"]


def test_cropformer_patch_applies_to_frozen_entity_source() -> None:
    if not ENTITY_ROOT.exists():
        pytest.skip("frozen CropFormer source is unavailable")
    completed = subprocess.run(
        ["git", "-C", str(ENTITY_ROOT), "apply", "--check", str(CROPFORMER_PATCH)],
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode == 0:
        return
    reverse = subprocess.run(
        [
            "git",
            "-C",
            str(ENTITY_ROOT),
            "apply",
            "--reverse",
            "--check",
            str(CROPFORMER_PATCH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reverse.returncode == 0, completed.stderr + reverse.stderr


def test_cropformer_patch_requires_compiled_cuda_without_fallback() -> None:
    patch = CROPFORMER_PATCH.read_text(encoding="utf-8")

    assert patch.count("AT_DISPATCH_FLOATING_TYPES(value.scalar_type()") == 2
    assert "MSDA = None" not in patch
    assert "+        if MSDA is None" not in patch
    assert "ms_deform_attn_core_pytorch(value" not in patch


def test_cropformer_patch_disables_dataset_autoregistration_only() -> None:
    patch = CROPFORMER_PATCH.read_text(encoding="utf-8")

    assert "mask2former/data/datasets/__init__.py" in patch
    assert "Dataset auto-registration is disabled for OVI-MAP inference." in patch
    assert "mask2former/data/__init__.py" not in patch


def test_cropformer_patch_exports_auditable_instance_id_png() -> None:
    patch = CROPFORMER_PATCH.read_text(encoding="utf-8")

    assert "demo_cropformer/demo_from_dirs.py" in patch
    assert "np.argsort(selected_scores, kind=\"stable\")" in patch
    assert "dtype=np.uint8" in patch
    assert "frame_diagnostics" in patch
    assert "foreground_ratio" in patch
    assert "largest_instance_ratio" in patch
    assert "config_sha256" in patch
    assert "weights_sha256" in patch


def test_ovimap_patch_applies_to_frozen_source() -> None:
    if not OVIMAP_ROOT.exists():
        pytest.skip("frozen OVI-MAP source is unavailable")
    completed = subprocess.run(
        ["git", "-C", str(OVIMAP_ROOT), "apply", "--check", str(OVIMAP_PATCH)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return
    reverse = subprocess.run(
        [
            "git",
            "-C",
            str(OVIMAP_ROOT),
            "apply",
            "--reverse",
            "--check",
            str(OVIMAP_PATCH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reverse.returncode == 0, completed.stderr + reverse.stderr


def test_ovimap_patch_is_build_only_plus_native_diagnostics() -> None:
    patch = OVIMAP_PATCH.read_text(encoding="utf-8")
    added = "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )

    for forbidden in (
        "connection_ratio_th_",
        "merging_min_overlap_ratio",
        "vis_area_thres =",
        "ray_cast_max_depth =",
    ):
        assert forbidden not in added
    assert "scripts/eval_inst_seg.py" not in patch
    assert "scripts/eval_sem_seg.py" not in patch
    assert "CMAKE_CXX_STANDARD 17" in patch
    assert "find_package(PCL" in patch
    assert "find_package(OpenGL" in patch
    assert "voxblox_rviz_plugin" in patch
    assert "gsm_node" in patch
    assert "$ENV{CONDA_PREFIX}" in patch


def test_ovimap_patch_adds_local_siglip_and_audit_outputs() -> None:
    patch = OVIMAP_PATCH.read_text(encoding="utf-8")

    assert "siglip_model_path" in patch
    assert "local_files_only" in patch
    assert "/home/ww/vv/paper2" not in patch
    assert "native_audit_dir" in patch
    assert "raycast.npy" in patch
    assert "mapper_rgb" in patch
    assert "logged_rgb" in patch
    assert "instance_colors_cpp.tsv" in patch
    assert "box_2d" in patch


def test_runner_freezes_official_frame_protocol(native_config: NativeConfig) -> None:
    commands = build_scene_commands(native_config, scene="room0")

    assert commands.frame_ids == list(range(0, 2000, 10))
    assert commands.frontend[0] == str(native_config.frontend_python)
    assert commands.mapping[0] == str(native_config.mapping_python)
    combined = " ".join(word for command in commands.commands for word in command)
    for fragment in (
        "--dataset replica",
        "--task Nyu40",
        "--data_association 2",
        "--inst_association 4",
        "--seg_graph_confidence 3",
        "--num_threads 10",
    ):
        assert fragment in combined
    assert "docker" not in combined.lower()
    assert "fallback" not in combined.lower()


def test_runner_builds_explicit_scannet_nyu_commands(
    native_config: NativeConfig,
) -> None:
    commands = build_scene_commands(
        native_config,
        scene="scan-uuid",
        dataset="scannet_nyu",
        start=0,
        end=3,
        step=1,
    )

    assert commands.dataset == "scannet_nyu"
    assert commands.frame_ids == [0, 1, 2]
    assert str(native_config.data_root / "scan-uuid" / "color" / "0.jpg") in commands.frontend
    combined = " ".join(word for command in commands.commands for word in command)
    assert "--dataset scannet_nyu" in combined
    assert "--dataset replica" not in combined
    assert "--scene-data " + str(native_config.data_root / "scan-uuid") in combined


def test_frontend_artifact_names_follow_dataset_image_stems() -> None:
    assert _frontend_mask_name("replica", 7) == "frame000007.png"
    assert _frontend_diagnostic_name("replica", 7) == "frame000007.json"
    assert _frontend_mask_name("scannet_nyu", 7) == "7.png"
    assert _frontend_diagnostic_name("scannet_nyu", 7) == "7.json"


def test_mapping_manifest_binds_every_d2_loader_artifact(tmp_path: Path) -> None:
    output = tmp_path / "mapping/cropformer_inst"
    output.mkdir(parents=True)
    (output / "instance_mesh_2.ply").write_bytes(b"mesh")
    (output / "inst_sem_siglip-l-16-384_2_incre_combine.pkl").write_bytes(b"features")
    color_log = tmp_path / "native_audit/instance_colors_cpp.tsv"
    color_log.parent.mkdir()
    color_log.write_bytes(b"instance_id\tr\tg\tb\n")
    commands = SceneCommands(
        frame_ids=[0, 1],
        geometry=("geometry",),
        frontend=("frontend",),
        mapping=("mapping",),
        attempt_root=tmp_path,
    )

    artifacts = _mapping_artifacts(commands)

    assert set(artifacts) == {
        "instance_mesh",
        "semantic_features",
        "instance_color_log",
    }
    for binding in artifacts.values():
        assert set(binding) == {"path", "sha256", "byte_count"}
    assert artifacts["instance_color_log"]["path"] == str(color_log.resolve())


def test_scannet_nyu_preflight_validates_calibrated_rgbd(
    native_config: NativeConfig,
) -> None:
    required_files = (
        native_config.frontend_python,
        native_config.mapping_python,
        native_config.cropformer_config,
        native_config.cropformer_weights,
        native_config.ovimap_root / "scripts/panoptic_mapping_.py",
        native_config.mapping_workspace
        / "devel/lib/consistent_gsm.cpython-311-x86_64-linux-gnu.so",
        native_config.mapping_workspace
        / "devel/lib/depth_segmentation_py.cpython-311-x86_64-linux-gnu.so",
        native_config.build_root / "environment/source-hashes.json",
    )
    for path in required_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    native_config.siglip_model.mkdir(parents=True)
    scene = native_config.data_root / "scan-uuid"
    for name in ("color", "depth", "pose", "intrinsic"):
        (scene / name).mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(scene / "color/0.jpg"), np.zeros((4, 6, 3), dtype=np.uint8))
    depth = np.asarray([[0, 1000, 0], [0, 0, 0]], dtype=np.uint16)
    assert cv2.imwrite(str(scene / "depth/0.png"), depth)
    np.savetxt(scene / "pose/0.txt", np.eye(4))
    color_intrinsic = np.eye(4)
    color_intrinsic[0, 0] = color_intrinsic[1, 1] = 4.0
    depth_intrinsic = np.eye(4)
    depth_intrinsic[0, 0] = depth_intrinsic[1, 1] = 2.0
    np.savetxt(scene / "intrinsic/intrinsic_color.txt", color_intrinsic)
    np.savetxt(scene / "intrinsic/intrinsic_depth.txt", depth_intrinsic)
    (scene / "materialized_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "RSCAN_OVI_VISIT_V1",
                "status": "MATERIALIZED_INPUT_PASS",
                "scan_id": "scan-uuid",
                "dataset_adapter": "scannet_nyu",
                "depth_shift": 1000.0,
                "frame_count": 1,
                "frame_map": [{"source_frame_id": 7, "target_frame_id": 0}],
                "dimensions": {
                    "color": {"width": 6, "height": 4},
                    "depth": {"width": 3, "height": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    commands = build_scene_commands(
        native_config,
        scene="scan-uuid",
        dataset="scannet_nyu",
        start=0,
        end=1,
        step=1,
    )

    result = preflight(native_config, commands, scene="scan-uuid")

    assert result["status"] == "PASS"
    assert result["dataset"] == "scannet_nyu"
    assert result["depth_shift"] == 1000.0
    assert result["color_shape"] == [4, 6, 3]
    assert result["depth_shape"] == [2, 3]
    assert result["camera_world_round_trip_error_m"] <= 1e-12
    assert result["source_frame_ids"] == [7]

    manifest_path = scene / "materialized_manifest.json"
    malformed = json.loads(manifest_path.read_text(encoding="utf-8"))
    malformed["dimensions"]["color"] = []
    manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(GateFailure, match="materialized dimensions"):
        preflight(native_config, commands, scene="scan-uuid")


def test_runner_uses_only_declared_gate_transitions() -> None:
    state = "ENV_PASS"
    for expected in (
        "FRONTEND_PASS",
        "MAPPING_PASS",
        "ROOM0_PASS",
        "REPLICA8_PASS",
        "EVAL_PASS",
    ):
        state = advance_state(
            state,
            event={"status": "PASS", "next_state": expected, "gate": expected},
        )
    assert state == "EVAL_PASS"


def test_failed_audit_cannot_advance_mapping_gate() -> None:
    with pytest.raises(GateFailure, match="native frame audit"):
        advance_state(
            "FRONTEND_PASS",
            event={
                "status": "FAIL",
                "next_state": "MAPPING_PASS",
                "gate": "native frame audit",
            },
        )


def test_runner_rejects_skipped_or_out_of_order_gates() -> None:
    with pytest.raises(GateFailure, match="invalid transition"):
        advance_state(
            "ENV_PASS",
            event={"status": "PASS", "next_state": "MAPPING_PASS"},
        )


def test_audit_records_raycast_ids_merged_before_final_color_log(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    native = tmp_path / "native_audit"
    frontend.mkdir()
    native.mkdir()
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 1:4] = 1
    assert cv2.imwrite(str(frontend / "frame000000.png"), mask)
    raycast = np.zeros((4, 5), dtype=np.uint16)
    raycast[1, 1:3] = 1
    raycast[2, 3:5] = 2
    np.save(native / "frame000000.raycast.npy", raycast, allow_pickle=False)
    (native / "color_pairs.json").write_text(
        json.dumps(
            [{"instance_id": 1, "mapper_rgb": [10, 20, 30], "logged_rgb": [10, 20, 30]}]
        ),
        encoding="utf-8",
    )
    (native / "frame000000.json").write_text(
        json.dumps(
            {
                "frame_id": 0,
                "shape": [4, 5],
                "box_2d": {"1": [1, 1, 2, 1], "2": [3, 2, 4, 2]},
                "mapper_rgb": {"1": [10, 20, 30], "2": [40, 50, 60]},
            }
        ),
        encoding="utf-8",
    )
    commands = SceneCommands(
        frame_ids=[0],
        geometry=("geometry",),
        frontend=("frontend",),
        mapping=("mapping",),
        attempt_root=tmp_path,
    )

    summary = audit_scene(commands)

    assert summary["status"] == "PASS"
    assert summary["active_color_ids"] == [1]
    assert summary["stale_raycast_ids"] == [2]


def test_scannet_audit_uses_depth_space_raycast_shape(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    native = tmp_path / "native_audit"
    scene_data = tmp_path / "scene"
    frontend.mkdir()
    native.mkdir()
    (scene_data / "depth").mkdir(parents=True)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:2, :3] = 1
    mask[2:, 3:] = 2
    assert cv2.imwrite(str(frontend / "0.png"), mask)
    depth = np.full((2, 3), 1000, dtype=np.uint16)
    assert cv2.imwrite(str(scene_data / "depth/0.png"), depth)
    raycast = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint16)
    np.save(native / "frame000000.raycast.npy", raycast, allow_pickle=False)
    (native / "color_pairs.json").write_text(
        json.dumps(
            [{"instance_id": 1, "mapper_rgb": [10, 20, 30], "logged_rgb": [10, 20, 30]}]
        ),
        encoding="utf-8",
    )
    (native / "frame000000.json").write_text(
        json.dumps(
            {
                "frame_id": 0,
                "shape": [2, 3],
                "box_2d": {"1": [1, 0, 2, 0]},
                "mapper_rgb": {"1": [10, 20, 30]},
            }
        ),
        encoding="utf-8",
    )
    commands = SceneCommands(
        frame_ids=[0],
        geometry=("geometry",),
        frontend=("frontend",),
        mapping=("mapping",),
        attempt_root=tmp_path,
        dataset="scannet_nyu",
        scene_data=scene_data,
    )

    summary = audit_scene(commands)

    assert summary["status"] == "PASS"
    assert summary["raycast_shapes"] == [[2, 3]]


def test_scene_audit_records_mapper_skipped_frames(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    native = tmp_path / "native_audit"
    scene_data = tmp_path / "scene"
    frontend.mkdir()
    native.mkdir()
    (scene_data / "depth").mkdir(parents=True)
    for frame_id in (0, 1):
        assert cv2.imwrite(
            str(frontend / f"{frame_id}.png"), np.ones((4, 6), dtype=np.uint8)
        )
        assert cv2.imwrite(
            str(scene_data / "depth" / f"{frame_id}.png"),
            np.full((2, 3), 1000, dtype=np.uint16),
        )
    raycast = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint16)
    np.save(native / "frame000001.raycast.npy", raycast, allow_pickle=False)
    (native / "color_pairs.json").write_text(
        json.dumps(
            [{"instance_id": 1, "mapper_rgb": [10, 20, 30], "logged_rgb": [10, 20, 30]}]
        ),
        encoding="utf-8",
    )
    (native / "frame000001.json").write_text(
        json.dumps(
            {
                "frame_id": 1,
                "shape": [2, 3],
                "box_2d": {"1": [1, 0, 2, 0]},
                "mapper_rgb": {"1": [10, 20, 30]},
            }
        ),
        encoding="utf-8",
    )
    commands = SceneCommands(
        frame_ids=[0, 1],
        geometry=("geometry",),
        frontend=("frontend",),
        mapping=("mapping",),
        attempt_root=tmp_path,
        dataset="scannet_nyu",
        scene_data=scene_data,
    )

    summary = audit_scene(commands)

    assert summary["frame_count"] == 1
    assert summary["requested_frame_count"] == 2
    assert summary["mapper_skipped_frame_ids"] == [0]
    assert summary["mapper_skipped_frame_count"] == 1


def test_scene_audit_allows_full_frame_box_when_scene_has_partial_box(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    native = tmp_path / "native_audit"
    scene_data = tmp_path / "scene"
    frontend.mkdir()
    native.mkdir()
    (scene_data / "depth").mkdir(parents=True)
    (native / "color_pairs.json").write_text(
        json.dumps(
            [{"instance_id": 1, "mapper_rgb": [10, 20, 30], "logged_rgb": [10, 20, 30]}]
        ),
        encoding="utf-8",
    )
    raycasts = (
        np.ones((2, 3), dtype=np.uint16),
        np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint16),
    )
    boxes = ({"1": [0, 0, 2, 1]}, {"1": [1, 0, 2, 0]})
    for frame_id, (raycast, box) in enumerate(zip(raycasts, boxes, strict=True)):
        assert cv2.imwrite(
            str(frontend / f"{frame_id}.png"), np.ones((4, 6), dtype=np.uint8)
        )
        assert cv2.imwrite(
            str(scene_data / "depth" / f"{frame_id}.png"),
            np.full((2, 3), 1000, dtype=np.uint16),
        )
        np.save(
            native / f"frame{frame_id:06d}.raycast.npy", raycast, allow_pickle=False
        )
        (native / f"frame{frame_id:06d}.json").write_text(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "shape": [2, 3],
                    "box_2d": box,
                    "mapper_rgb": {"1": [10, 20, 30]},
                }
            ),
            encoding="utf-8",
        )
    commands = SceneCommands(
        frame_ids=[0, 1],
        geometry=("geometry",),
        frontend=("frontend",),
        mapping=("mapping",),
        attempt_root=tmp_path,
        dataset="scannet_nyu",
        scene_data=scene_data,
    )

    summary = audit_scene(commands)

    assert summary["raycast_bbox_count"] == 2
    assert summary["full_frame_bbox_count"] == 1
    assert summary["full_frame_bbox_ratio"] == pytest.approx(0.5)
