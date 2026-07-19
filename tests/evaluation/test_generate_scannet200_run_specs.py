from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path("scripts/evaluation/generate_scannet200_run_specs.py")
METHODS = ("CONCEPTGRAPHS", "DUALMAP", "OPENFUSION", "OVIMAP")
SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, ready: bool = True) -> dict[str, Path]:
    methods = {}
    preflight_methods = {}
    for index, method in enumerate(METHODS):
        methods[method] = {
            "upstream_commit": str(index) * 40,
            "mode": "native",
            "runner_kind": method.lower(),
            "environment": f"env-{method.lower()}",
            "required_assets": [],
            "supported_metrics": ["semantic.miou", "geometry.f5"],
            "unsupported_metrics": [],
            "gpu_hint": index,
            "argv_template": [
                "python",
                "runner.py",
                "--manifest",
                "{manifest}",
                "--output",
                "{output}",
            ],
        }
        preflight_methods[method] = {
            "asset_status": "READY" if ready else "BLOCKED",
            "failures": [] if ready else [{"kind": "missing"}],
        }
    config = tmp_path / "methods.json"
    config.write_text(
        json.dumps({"schema_version": 1, "dataset": "ScanNet200", "methods": methods}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "scannet200_5_static_v1",
                "dataset": "ScanNet200",
                "dataset_root": "/official/scannet",
                "scenes": [{"scene_id": scene} for scene in SCENES],
            }
        ),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "READY" if ready else "BLOCKED",
                "reason": None if ready else "METHOD_ASSETS_NOT_READY",
                "method_config": {"path": str(config), "sha256": _sha256(config)},
                "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
                "methods": preflight_methods,
            }
        ),
        encoding="utf-8",
    )
    return {
        "config": config,
        "manifest": manifest,
        "preflight": preflight,
        "input": tmp_path / "input",
        "output": tmp_path / "runs",
        "specs": tmp_path / "specs",
    }


def _run(paths: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--method-config",
            str(paths["config"]),
            "--manifest",
            str(paths["manifest"]),
            "--preflight",
            str(paths["preflight"]),
            "--input-root",
            str(paths["input"]),
            "--output-root",
            str(paths["output"]),
            "--spec-dir",
            str(paths["specs"]),
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_generator_writes_one_ready_spec_per_method(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    assert {path.name for path in paths["specs"].iterdir()} == {
        f"{method.lower()}.json" for method in METHODS
    }
    for method in METHODS:
        spec = json.loads(
            (paths["specs"] / f"{method.lower()}.json").read_text(encoding="utf-8")
        )
        assert spec["status"] == "READY"
        assert spec["method"] == method
        assert spec["scene_order"] == list(SCENES)
        assert spec["manifest"]["sha256"] == _sha256(paths["manifest"])
        assert isinstance(spec["command_argv"], list)
        assert str(paths["manifest"].resolve()) in spec["command_argv"]
        assert spec["required_metric_contract"]["supported"]


def test_generator_refuses_blocked_preflight_without_specs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, ready=False)

    result = _run(paths)

    assert result.returncode != 0
    assert "READY" in result.stderr
    assert not paths["specs"].exists()


def test_generator_never_replaces_nonempty_spec_directory(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["specs"].mkdir()
    (paths["specs"] / "result.json").write_text("{}\n", encoding="utf-8")

    result = _run(paths, "--replace-empty")

    assert result.returncode != 0
    assert "non-empty" in result.stderr
    assert (paths["specs"] / "result.json").is_file()
