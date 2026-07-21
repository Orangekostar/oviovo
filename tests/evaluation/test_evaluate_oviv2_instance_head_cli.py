from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.evaluation.evaluate_oviv2_replica import evaluate as evaluate_legacy
from tests.evaluation.test_evaluate_oviv2_replica_cli import _fixture


SCRIPT = Path("scripts/evaluation/evaluate_oviv2_instance_head.py")


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
    assert set(audit["source_hashes"]) == {"cli", "instance_head"}
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
