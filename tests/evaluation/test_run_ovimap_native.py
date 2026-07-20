from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ENV = REPO_ROOT / "configs/environments/ovimap_cropformer.yaml"
MAPPING_ENV = REPO_ROOT / "configs/environments/ovimap_map.yaml"
BOOTSTRAP = REPO_ROOT / "scripts/reproduction/ovimap/bootstrap_native_envs.sh"


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
        "cuda-nvcc=12.1.105",
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
        "catkin_tools",
        "cmake",
        "make",
        "pybind11",
        "protobuf",
        "glog",
        "gflags",
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
        "opencv-contrib-python==4.11.*",
        "open3d",
        "plyfile==1.1.3",
        "sentencepiece",
    } <= set(pip_section)


def test_bootstrap_freezes_sources_and_environment_provenance() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")

    assert "58a804e2d7c82ba05a489eb071aba3367301fed8" in script
    assert "6e7e13ac91ef508088e1b848167c01f19b00b512" in script
    assert "d1e04565d3bec8719335b88be9e9b961bf3ec464" in script
    assert "https://github.com/OVI-MAP/OVI-MAP.git" in script
    assert "https://github.com/qqlu/Entity.git" in script
    assert "https://github.com/facebookresearch/detectron2.git" in script
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
