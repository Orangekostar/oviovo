from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.run_oviv2_replica import run as run_mapper
from src.evaluation.oviv2_semantic_replay import (
    SEMANTIC_REPLAY_SOURCE_RELATIVE_PATHS,
)
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.snapshot import VoxelMapSnapshot
from tests.evaluation.test_run_oviv2_replica_cli import (
    _args,
    _write_dense_cache,
    _write_fixture,
)


SCRIPT = Path("scripts/evaluation/replay_oviv2_semantic_evidence.py")


def _semantic_arrays(store) -> dict[tuple[int, int, int], tuple[np.ndarray, ...]]:
    return {
        key: (
            block.semantic_ids.copy(),
            block.semantic_support.copy(),
            block.semantic_revisions.copy(),
        )
        for key, block in sorted(store._blocks.items())
    }


def test_control_replay_matches_mapper_semantic_evidence(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, cache_frames=2)
    _write_dense_cache(config, frame_count=2)
    base_output = tmp_path / "base"
    run_mapper(_args(config, base_output, num_frames=2))
    snapshot_path = base_output / "final" / "oviv2_voxel_snapshot.npz"
    run_manifest_path = base_output / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest.pop("source_frame_ids_hash")
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replay_output = tmp_path / "replay"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--base-snapshot",
            str(snapshot_path),
            "--output",
            str(replay_output),
            "--num-frames",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    manifest = json.loads(
        (replay_output / "replay_manifest.json").read_text(encoding="utf-8")
    )
    base = VoxelMapSnapshot.load(snapshot_path)
    replay = SparseEvidenceStore.load(
        replay_output / "semantic_evidence.npz",
        base.evidence.config,
    )
    expected = _semantic_arrays(base.evidence)
    actual = _semantic_arrays(replay)
    assert actual.keys() == expected.keys()
    for key in actual:
        for actual_array, expected_array in zip(actual[key], expected[key], strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
    assert summary["frame_count"] == 2
    assert manifest["structure_replayed"] is True
    assert manifest["control_parameters_match_base"] is True
    assert manifest["control_semantic_evidence_matches_base"] is True
    assert manifest["overlay_sha256"] == summary["overlay_sha256"]
    assert manifest["base_snapshot_checksums"] == base.checksums
    assert set(manifest["source_hashes"]) == set(
        SEMANTIC_REPLAY_SOURCE_RELATIVE_PATHS
    )
    assert not any(
        replay.entity_candidates(key)
        for key in replay._blocks
    )


def test_replay_cli_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(tmp_path / "missing.json"),
            "--base-snapshot",
            str(tmp_path / "missing-snapshot"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr
