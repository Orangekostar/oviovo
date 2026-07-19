from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/baselines/conceptgraphs_canonical_replica.json"
)
VALIDATOR_PATH = REPOSITORY_ROOT / "scripts/evaluation/validate_conceptgraphs_gate.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_configuration_identity() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["frontend"] == "sam_segment_all"
    assert config["gsa_variant"] == "none"
    assert config["class_agnostic"] is True
    assert config["mask_conf_threshold"] == pytest.approx(0.95)
    assert config["sim_threshold"] == pytest.approx(1.2)
    assert config["dbscan_eps"] == pytest.approx(0.1)
    assert config["merge_interval"] == 20
    assert config["merge_visual_sim_thresh"] == pytest.approx(0.8)
    assert config["merge_text_sim_thresh"] == pytest.approx(0.8)
    assert config["headline_stride"] == 10
    assert config["official_diagnostic_stride"] == 5
    assert config["headline_evaluation"] == {
        "gt_only_class_suppression": False,
        "n_exclude": 0,
    }
    assert "yolo_world" not in json.dumps(config).lower()
    assert "mobile_sam" not in json.dumps(config).lower()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    detections = tmp_path / "detections"
    detections.mkdir()
    (detections / "frame000000.pkl.gz").write_bytes(b"canonical masks")
    map_path = tmp_path / "map.pkl.gz"
    map_path.write_bytes(b"canonical map")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps({"metrics": {"semantic": {"matched_point_ratio": 0.5}}}),
        encoding="utf-8",
    )
    provenance = {}
    for name in (
        "sam_vit_h_checkpoint",
        "groundingdino_checkpoint",
        "ram_checkpoint",
        "mapping_config",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        provenance[name] = {"path": str(path), "sha256": _sha256(path)}
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "source_commits": config["source_commits"],
                "stages": {
                    "frontend": {"exit_status": 0},
                    "mapper": {"exit_status": 0},
                },
                "counts": {"mask_count": 12, "mapped_object_count": 3},
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    return {
        "detections": detections,
        "map": map_path,
        "metrics": metrics,
        "status": status,
        "output": tmp_path / "gate.json",
    }


def _run(
    paths: dict[str, Path],
    *,
    config_path: Path = CONFIG_PATH,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--config",
            str(config_path),
            "--status",
            str(paths["status"]),
            "--detections",
            str(paths["detections"]),
            "--map",
            str(paths["map"]),
            "--neutral-metrics",
            str(paths["metrics"]),
            "--output",
            str(paths["output"]),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_validator_accepts_complete_canonical_gate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    gate = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert gate["status"] == "VERIFIED"
    assert gate["headline_result_eligible"] is True
    assert gate["counts"] == {"mapped_object_count": 3, "mask_count": 12}
    assert gate["neutral_matched_point_ratio"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty_masks", "mask_count"),
        ("wrong_frontend", "frontend"),
        ("gt_suppression", "suppression"),
        ("missing_hash", "provenance"),
    ],
)
def test_validator_rejects_noncanonical_gate(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    if mutation == "empty_masks":
        status["counts"]["mask_count"] = 0
    elif mutation == "wrong_frontend":
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["frontend"] = "yolo_world"
        alternate_config = tmp_path / "config.json"
        alternate_config.write_text(json.dumps(config), encoding="utf-8")
        result = _run(paths, config_path=alternate_config)
        assert result.returncode != 0
        assert message in result.stderr.lower()
        return
    elif mutation == "gt_suppression":
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["headline_evaluation"]["gt_only_class_suppression"] = True
        alternate_config = tmp_path / "config.json"
        alternate_config.write_text(json.dumps(config), encoding="utf-8")
        result = _run(paths, config_path=alternate_config)
        assert result.returncode != 0
        assert message in result.stderr.lower()
        return
    else:
        status["provenance"].pop("sam_vit_h_checkpoint")
    paths["status"].write_text(json.dumps(status), encoding="utf-8")

    result = _run(paths)

    assert result.returncode != 0
    assert message in result.stderr.lower()
