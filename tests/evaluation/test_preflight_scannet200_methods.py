from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


CONFIG = Path("configs/evaluation/baselines/scannet200_methods.json")
SCRIPT = Path("scripts/evaluation/preflight_scannet200_methods.py")
METHODS = {"OPENFUSION", "OVIMAP", "CONCEPTGRAPHS", "DUALMAP"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_method_config_declares_exact_capabilities_and_openfusion_na() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert set(config["methods"]) == METHODS
    for method in config["methods"].values():
        assert len(method["upstream_commit"]) == 40
        assert method["mode"]
        assert method["runner_kind"]
        assert method["required_assets"]
        assert method["supported_metrics"]
        assert "unsupported_metrics" in method
    unsupported = config["methods"]["OPENFUSION"]["unsupported_metrics"]
    assert {item["metric"] for item in unsupported} == {"instance.ap25", "instance.ap50"}
    assert all(item["value"] == "N/A" and item["reason"] for item in unsupported)


def _config(tmp_path: Path, *, failures: bool) -> Path:
    methods = {}
    for index, method in enumerate(sorted(METHODS)):
        asset = tmp_path / f"{method.lower()}.bin"
        asset.write_bytes(method.encode("ascii"))
        digest = _sha256(asset)
        if failures and method == "OPENFUSION":
            asset.unlink()
        if failures and method == "OVIMAP":
            digest = "0" * 64
        methods[method] = {
            "upstream_commit": str(index) * 40,
            "mode": "native",
            "runner_kind": method.lower(),
            "environment": f"env-{method.lower()}",
            "required_assets": [
                {"name": "weight", "path": str(asset), "sha256": digest}
            ],
            "supported_metrics": ["semantic.miou", "geometry.f5"],
            "unsupported_metrics": (
                [
                    {
                        "metric": "instance.ap25",
                        "value": "N/A",
                        "reason": "not supported",
                    },
                    {
                        "metric": "instance.ap50",
                        "value": "N/A",
                        "reason": "not supported",
                    },
                ]
                if method == "OPENFUSION"
                else []
            ),
            "gpu_hint": 0,
            "argv_template": ["runner", "--manifest", "{manifest}"],
        }
    path = tmp_path / "methods.json"
    path.write_text(
        json.dumps({"schema_version": 1, "dataset": "ScanNet200", "methods": methods}),
        encoding="utf-8",
    )
    return path


def _run(config: Path, manifest: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--method-config",
            str(config),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_preflight_reports_ready_assets_when_manifest_is_absent(tmp_path: Path) -> None:
    config = _config(tmp_path, failures=False)
    output = tmp_path / "preflight.json"

    result = _run(config, tmp_path / "missing-manifest.json", output)

    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "SCANNET_DATA_NOT_VALIDATED"
    assert all(item["asset_status"] == "READY" for item in report["methods"].values())


def test_preflight_aggregates_missing_and_hash_mismatched_assets(tmp_path: Path) -> None:
    config = _config(tmp_path, failures=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "scannet200_5_static_v1",
                "dataset": "ScanNet200",
                "scenes": [
                    {"scene_id": scene}
                    for scene in (
                        "scene0011_00",
                        "scene0050_00",
                        "scene0231_00",
                        "scene0378_00",
                        "scene0518_00",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "preflight.json"

    result = _run(config, manifest, output)

    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["reason"] == "METHOD_ASSETS_NOT_READY"
    assert report["methods"]["OPENFUSION"]["failures"][0]["kind"] == "missing"
    assert report["methods"]["OVIMAP"]["failures"][0]["kind"] == "hash_mismatch"
