from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluation.finalize_ovimap_native import (
    FinalizationError,
    aggregate_native_mapping,
    finalize_native_result,
)
from src.evaluation.baselines.ovimap_paper_audit import PAPER_PROTOCOL


SCENES = tuple(PAPER_PROTOCOL["scene_ids"])
FRAMES = list(range(0, 2000, 10))


def test_finalize_cli_imports_repo_modules_outside_repo(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/evaluation/finalize_ovimap_native.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_scene_manifest(tmp_path: Path, scene: str, *, source_hash: str = "1" * 64) -> Path:
    artifact_root = tmp_path / scene
    artifact_root.mkdir(exist_ok=True)
    mesh = artifact_root / "instance_mesh_200.ply"
    features = artifact_root / "features.pkl"
    mesh.write_bytes(f"mesh-{scene}".encode())
    features.write_bytes(f"features-{scene}".encode())
    payload = {
        "status": "PASS",
        "state": "ROOM0_PASS" if scene == "room0" else "MAPPING_PASS",
        "scene": scene,
        "frame_ids": FRAMES,
        "preflight": {
            "sources": {
                "cropformer_config": {"sha256": "2" * 64},
                "cropformer_weights": {"sha256": "3" * 64},
                "mapper": {"sha256": "4" * 64},
                "source_hashes": {"sha256": source_hash},
            }
        },
        "frontend": {"argv": ["/env/frontend/python", "demo.py"]},
        "mapping": {
            "geometry": {"argv": ["/env/mapping/python", "geometry.py"]},
            "mapping": {"argv": ["/env/mapping/python", "mapper.py"]},
        },
        "audit": {"status": "PASS", "frame_count": 200},
        "artifacts": {
            "instance_mesh": {"path": str(mesh), "sha256": _sha256(mesh)},
            "semantic_features": {"path": str(features), "sha256": _sha256(features)},
        },
    }
    path = artifact_root / "native_mapping_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _input_hashes() -> dict[str, str]:
    return {
        "replica51_vocabulary": "5" * 64,
        "siglip_model": "6" * 64,
        "replica_semantic_gt": "7" * 64,
        "replica_instance_gt": "8" * 64,
    }


def test_aggregate_requires_exact_hash_consistent_replica8(tmp_path: Path) -> None:
    manifests = {scene: _write_scene_manifest(tmp_path, scene) for scene in SCENES}

    aggregate = aggregate_native_mapping(manifests, input_hashes=_input_hashes())

    assert aggregate["status"] == "COMPLETE_NATIVE_MAPPING"
    assert aggregate["scene_ids"] == list(SCENES)
    assert aggregate["frame_ids_by_scene"] == {scene: FRAMES for scene in SCENES}
    assert list(aggregate["scenes"]) == list(SCENES)


def test_aggregate_rejects_incomplete_or_mixed_scene_manifests(tmp_path: Path) -> None:
    manifests = {scene: _write_scene_manifest(tmp_path, scene) for scene in SCENES}
    manifests.pop("room2")
    with pytest.raises(FinalizationError, match="Replica-8"):
        aggregate_native_mapping(manifests, input_hashes=_input_hashes())

    manifests["room2"] = _write_scene_manifest(
        tmp_path, "room2", source_hash="9" * 64
    )
    with pytest.raises(FinalizationError, match="source hash mismatch"):
        aggregate_native_mapping(manifests, input_hashes=_input_hashes())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(status="FAIL"), "gate did not pass"),
        (lambda value: value["frame_ids"].pop(), "frame protocol mismatch"),
        (
            lambda value: value["frontend"].update(argv=["/other/python", "demo.py"]),
            "environment mismatch",
        ),
        (
            lambda value: value["artifacts"]["instance_mesh"].update(
                sha256="0" * 64
            ),
            "artifact hash mismatch",
        ),
    ),
)
def test_aggregate_rejects_failed_or_mixed_scene_evidence(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    manifests = {scene: _write_scene_manifest(tmp_path, scene) for scene in SCENES}
    room2_path = manifests["room2"]
    room2 = json.loads(room2_path.read_text(encoding="utf-8"))
    mutation(room2)
    room2_path.write_text(json.dumps(room2), encoding="utf-8")

    with pytest.raises(FinalizationError, match=message):
        aggregate_native_mapping(manifests, input_hashes=_input_hashes())


def _released_manifest(tmp_path: Path) -> Path:
    payload = {
        "status": "COMPLETE_RELEASED_EVALUATION",
        "source_protocol": "released_ovimap_replica51",
        "scene_ids": list(SCENES),
        "frame_count_per_scene": 200,
        "semantic_vocabulary": "Replica-51",
        "paper_metric_availability": {
            "table_2_class_agnostic_ap": False,
            "table_3_semantic": True,
        },
        "output_artifacts": {
            "semantic_vertex_metrics": {"miou": 0.265, "macc": 0.322},
            "semantic_instance_metrics": {"apall": 0.085, "ap50": 0.212, "ap25": 0.345},
            "files": {"placeholder": {"path": "unused", "sha256": "a" * 64}},
        },
    }
    path = tmp_path / "released.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_finalizer_preserves_semantic_diagnostics_without_paper_ap(
    tmp_path: Path,
) -> None:
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            {
                "status": "COMPLETE_NATIVE_MAPPING",
                "protocol_name": PAPER_PROTOCOL["name"],
                "scene_ids": list(SCENES),
                "frame_ids_by_scene": {scene: FRAMES for scene in SCENES},
                "input_hashes": _input_hashes(),
            }
        ),
        encoding="utf-8",
    )

    result = finalize_native_result(
        {
            "native_mapping_manifest": native,
            "released_evaluation_manifest": _released_manifest(tmp_path),
        }
    )

    assert result["availability"]["released_semantic_miou"] == "VERIFIED_DIAGNOSTIC"
    assert result["availability"]["class_agnostic_ap25"] == (
        "UNFILLED_NO_RELEASED_EVALUATOR"
    )
    assert result["metrics"]["released_replica51"]["semantic_miou"] == 0.265
    assert result["token_bindings"] == []


def test_finalizer_rejects_nonfinite_or_renamed_released_metrics(tmp_path: Path) -> None:
    released = json.loads(_released_manifest(tmp_path).read_text(encoding="utf-8"))
    released["output_artifacts"]["semantic_vertex_metrics"] = {
        "mean_iou": 0.265,
        "macc": float("nan"),
    }
    path = tmp_path / "bad-released.json"
    path.write_text(json.dumps(released), encoding="utf-8")

    with pytest.raises(FinalizationError, match="semantic vertex metric"):
        finalize_native_result({"released_evaluation_manifest": path})
