from __future__ import annotations

import hashlib
import json

import pytest

from src.evaluation.baselines.result_manifest import ResultManifestError, finalize_static_result


def _provenance(tmp_path) -> dict:
    manifest = tmp_path / "manifest.json"
    weight = tmp_path / "weight.bin"
    raw_log = tmp_path / "run.log"
    manifest.write_text("{}\n", encoding="utf-8")
    weight.write_bytes(b"weight")
    raw_log.write_text("exit=0\n", encoding="utf-8")
    return {
        "status": "VERIFIED",
        "run_id": "dualmap-replica8",
        "method": {"key": "DUALMAP", "display_label": "DualMap", "mode": "native"},
        "upstream_commit": "abc123",
        "adapter_commit": "def456",
        "environment": {"name": "test"},
        "command": "run baseline",
        "config": {"path": str(manifest)},
        "weights": [{"path": str(weight), "url": "https://example.invalid/weight"}],
        "dataset": {
            "name": "Replica",
            "splits": ["replica_8_compat", "replica_7_heldout"],
            "manifest": {"path": str(manifest)},
            "pose_source": "Replica traj.txt",
        },
        "hardware": {"gpu": "test"},
        "seed": 0,
        "raw_outputs": [{"path": str(raw_log)}],
        "protocol_deviations": [],
        "token_bindings": [
            {
                "token": "T1_DUALMAP_REPLICA8_MIOU",
                "json_pointer": "/metrics/replica_8_compat/semantic/miou",
                "precision": 3,
            }
        ],
    }


def test_finalize_static_result_injects_metrics_and_hashes_files(tmp_path) -> None:
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps({"metrics": {"replica_8_compat": {"semantic": {"miou": 0.25}}}}),
        encoding="utf-8",
    )

    result = finalize_static_result(aggregate, _provenance(tmp_path))

    assert result["metrics"]["replica_8_compat"]["semantic"]["miou"] == 0.25
    assert result["aggregate_source"]["sha256"] == hashlib.sha256(aggregate.read_bytes()).hexdigest()
    assert result["weights"][0]["sha256"] == hashlib.sha256(b"weight").hexdigest()
    assert result["dataset"]["manifest"]["sha256"]
    assert result["raw_outputs"][0]["sha256"]


def test_finalize_static_result_rejects_unverified_oviovo_and_missing_files(tmp_path) -> None:
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text('{"metrics": {}}\n', encoding="utf-8")
    provenance = _provenance(tmp_path)

    provenance["status"] = "BLOCKED"
    with pytest.raises(ResultManifestError, match="VERIFIED"):
        finalize_static_result(aggregate, provenance)

    provenance = _provenance(tmp_path)
    provenance["token_bindings"][0]["token"] = "T1_OVIOVO_REPLICA8_MIOU"
    with pytest.raises(ResultManifestError, match="OVIOVO"):
        finalize_static_result(aggregate, provenance)

    provenance = _provenance(tmp_path)
    provenance["raw_outputs"][0]["path"] = str(tmp_path / "missing.log")
    with pytest.raises(ResultManifestError, match="does not exist"):
        finalize_static_result(aggregate, provenance)
