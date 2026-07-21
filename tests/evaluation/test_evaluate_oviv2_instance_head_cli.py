from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.evaluation.evaluate_oviv2_replica import evaluate as evaluate_legacy
from tests.evaluation.test_evaluate_oviv2_replica_cli import _fixture


SCRIPT = Path("scripts/evaluation/evaluate_oviv2_instance_head.py")


def _write_auxiliary_run_manifest(paths: dict[str, Path], path: Path) -> Path:
    from src.oviv2.snapshot import VoxelMapSnapshot

    snapshot = VoxelMapSnapshot.load(paths["snapshot"])
    benchmark_hash = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()
    vocabulary_hash = hashlib.sha256(
        json.dumps(
            {"aliases": {}, "classes": ["wall", "chair"]},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "method": "OVIV2",
                "scene": "fixture",
                "final_revision": snapshot.metadata.revision,
                "algorithm_hash": "a" * 64,
                "benchmark_manifest_hash": benchmark_hash,
                "vocabulary_hash": vocabulary_hash,
                "artifact_checksums": {
                    f"final/oviv2_voxel_snapshot.npz/{name}": digest
                    for name, digest in snapshot.checksums.items()
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_instance_head_cli_preserves_other_heads_and_writes_audit(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        schema_version=3,
        entity_probabilities=((2, 1.0),),
    )
    output = tmp_path / "instance-head"
    command = [
        sys.executable,
        str(SCRIPT),
        "--snapshot",
        str(paths["snapshot"]),
        "--gt-mesh",
        str(paths["gt_mesh"]),
        "--gt-info",
        str(paths["gt_info"]),
        "--manifest",
        str(paths["manifest"]),
        "--scene",
        "fixture",
        "--output",
        str(output),
        "--min-instance-vertices",
        "1",
        "--minimum-component-vertices",
        "1",
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads(completed.stdout)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (output / "instance_head_audit.json").read_text(encoding="utf-8")
    )

    assert summary == {
        name: metrics[name]
        for name in ("ap25", "ap50", "f5", "f_miou", "macc", "miou")
    }
    assert metrics["protocol"]["headline_instance_protocol"] == (
        "independent_gt_projection_hypotheses"
    )
    assert metrics["instance"]["class_agnostic"] == audit["instance_head"]
    assert metrics["instance"]["legacy_class_agnostic"] == audit["legacy"]
    assert audit["config"]["minimum_component_vertices"] == 1
    assert audit["snapshot_checksums"]
    assert set(audit["source_hashes"]) == {
        "cli",
        "geometry_head",
        "instance_head",
        "replica_evaluator",
    }
    assert all(len(value) == 64 for value in audit["source_hashes"].values())

    original_metrics = (output / "metrics.json").read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
    assert (output / "metrics.json").read_bytes() == original_metrics

    # The instance-only head must not recompute or alter semantic/geometry values.
    import argparse

    legacy_output = tmp_path / "legacy"
    legacy = evaluate_legacy(
        argparse.Namespace(
            snapshot=paths["snapshot"],
            entity_info=None,
            gt_mesh=paths["gt_mesh"],
            gt_info=paths["gt_info"],
            manifest=paths["manifest"],
            scene="fixture",
            output=legacy_output,
            min_instance_vertices=1,
            semantic_head="fused_uncertainty",
            fusion_entity_weight_scale=0.49,
        )
    )
    for name in ("miou", "macc", "f_miou", "f5"):
        assert metrics[name] == legacy[name]


def test_instance_head_cli_rejects_non_registry_snapshot(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, schema_version=1)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot",
            str(paths["snapshot"]),
            "--gt-mesh",
            str(paths["gt_mesh"]),
            "--gt-info",
            str(paths["gt_info"]),
            "--manifest",
            str(paths["manifest"]),
            "--scene",
            "fixture",
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "schema v2/v3 registry" in completed.stderr


def test_instance_head_cli_fuses_audited_auxiliary_snapshot(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        schema_version=3,
        entity_probabilities=((2, 1.0),),
    )
    primary_output = tmp_path / "primary"
    ensemble_output = tmp_path / "ensemble"
    run_manifest = _write_auxiliary_run_manifest(paths, tmp_path / "run_manifest.json")
    base_command = [
        sys.executable,
        str(SCRIPT),
        "--snapshot",
        str(paths["snapshot"]),
        "--gt-mesh",
        str(paths["gt_mesh"]),
        "--gt-info",
        str(paths["gt_info"]),
        "--manifest",
        str(paths["manifest"]),
        "--scene",
        "fixture",
        "--min-instance-vertices",
        "1",
        "--minimum-component-vertices",
        "1",
    ]
    primary = subprocess.run(
        [*base_command, "--output", str(primary_output)],
        check=True,
        capture_output=True,
        text=True,
    )
    ensemble = subprocess.run(
        [
            *base_command,
            "--output",
            str(ensemble_output),
            "--auxiliary-instance-snapshot",
            str(paths["snapshot"]),
            "--auxiliary-instance-run-manifest",
            str(run_manifest),
            "--auxiliary-score-weight",
            "6",
            "--auxiliary-support-exponent",
            "1.5",
            "--auxiliary-score-bias",
            "0.05",
            "--deduplication-iou-threshold",
            "0.9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert primary.returncode == ensemble.returncode == 0
    primary_metrics = json.loads((primary_output / "metrics.json").read_text())
    metrics = json.loads((ensemble_output / "metrics.json").read_text())
    audit = json.loads((ensemble_output / "instance_head_audit.json").read_text())

    for name in ("miou", "macc", "f_miou", "f5"):
        assert metrics[name] == primary_metrics[name]
    provenance = metrics["protocol"]["auxiliary_instance_ensemble"]
    assert provenance["upstream_algorithm_hash"] == "a" * 64
    assert len(provenance["algorithm_hash"]) == 64
    assert len(provenance["run_manifest_sha256"]) == 64
    assert provenance["snapshot_checksums"]
    assert provenance["config"] == {
        "score_bias": 0.05,
        "score_weight": 6.0,
        "support_exponent": 1.5,
    }
    assert audit["auxiliary_hypothesis_count"] == 1
    assert audit["instance_head"]["raw_hypothesis_count"] == 2


def test_instance_head_cli_requires_matching_auxiliary_manifest(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        schema_version=3,
        entity_probabilities=((2, 1.0),),
    )
    missing_pair = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot",
            str(paths["snapshot"]),
            "--gt-mesh",
            str(paths["gt_mesh"]),
            "--gt-info",
            str(paths["gt_info"]),
            "--manifest",
            str(paths["manifest"]),
            "--scene",
            "fixture",
            "--output",
            str(tmp_path / "missing-pair"),
            "--auxiliary-instance-snapshot",
            str(paths["snapshot"]),
        ],
        capture_output=True,
        text=True,
    )
    assert missing_pair.returncode != 0
    assert "must be supplied together" in missing_pair.stderr

    run_manifest = _write_auxiliary_run_manifest(paths, tmp_path / "bad_manifest.json")
    payload = json.loads(run_manifest.read_text())
    first_key = sorted(payload["artifact_checksums"])[0]
    payload["artifact_checksums"][first_key] = "f" * 64
    run_manifest.write_text(json.dumps(payload), encoding="utf-8")
    mismatch = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot",
            str(paths["snapshot"]),
            "--gt-mesh",
            str(paths["gt_mesh"]),
            "--gt-info",
            str(paths["gt_info"]),
            "--manifest",
            str(paths["manifest"]),
            "--scene",
            "fixture",
            "--output",
            str(tmp_path / "mismatch"),
            "--auxiliary-instance-snapshot",
            str(paths["snapshot"]),
            "--auxiliary-instance-run-manifest",
            str(run_manifest),
        ],
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0
    assert "checksum mismatch" in mismatch.stderr
