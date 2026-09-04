from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.run_rescene_pair_backend import (
    BackendRunError,
    run_rescene_pair_backend,
)
from src.oviv2.ovi_rescene_adapter import write_neural_sample_artifact
from src.oviv2.rescene_backend import ReSceneBackend, ReSceneBackendError
from src.oviv2.two_visit_contracts import NeuralSampleMap


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "rescene"
    config = checkout / "conf/backbone/concerto.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("name: concerto_base\n", encoding="utf-8")
    (checkout / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    _git(checkout, "config", "user.email", "test@example.invalid")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-q", "-m", "source")
    commit = _git(checkout, "rev-parse", "HEAD")
    required = []
    for relative in ("LICENSE", "conf/backbone/concerto.yaml"):
        content = (checkout / relative).read_bytes()
        required.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "EXTERNAL_SOURCE_PASS",
                "sources": {
                    "rescene": {"commit": commit, "required_files": required}
                },
                "checkpoint": {
                    "selected_backbone": "Concerto",
                    "status": "BOUND_FOR_HERMETIC_TEST",
                },
            }
        ),
        encoding="utf-8",
    )
    return checkout, manifest


def _pair() -> NeuralSampleMap:
    return NeuralSampleMap(
        coordinates_xyzt=np.asarray(
            [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        features=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        visit_ids=np.asarray([0, 1], dtype=np.int8),
        source_visit_ids=np.asarray([0, 1], dtype=np.int8),
        source_entity_ids=("t0:a", "t1:a"),
        source_point_indices=np.asarray([0, 1], dtype=np.int64),
        source_to_token_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        neural_voxel_size_m=0.02,
        feature_schema="rgb",
        coordinate_frame_id="world",
        source_manifest_sha256="a" * 64,
        source_visit_map_sha256=("b" * 64, "c" * 64),
    )


def _fake_executor(tmp_path: Path) -> Path:
    script = tmp_path / "fake_rescene.py"
    script.write_text(
        """\
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pair-arrays', type=Path, required=True)
parser.add_argument('--output-arrays', type=Path, required=True)
parser.add_argument('--output-manifest', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--checkout', type=Path, required=True)
parser.add_argument('--pair-sha256', required=True)
parser.add_argument('--checkpoint-sha256', required=True)
parser.add_argument('--feature-schema', required=True)
parser.add_argument('--neural-voxel-size-m', required=True)
args = parser.parse_args()
with np.load(args.pair_arrays, allow_pickle=False) as source:
    assert source['features'].shape == (2, 3)
args.output_arrays.parent.mkdir(parents=True, exist_ok=True)
with args.output_arrays.open('wb') as stream:
    np.savez_compressed(
        stream,
        token_indices=np.asarray([1, 0], dtype=np.int64),
        query_masks=np.asarray([[True, False], [False, True]], dtype=np.bool_),
        token_scores=np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        query_scores=np.asarray([0.9, 0.8], dtype=np.float32),
    )
content = args.output_arrays.read_bytes()
args.output_manifest.write_text(json.dumps({
    'schema_version': 1,
    'status': 'PASS',
    'backend_name': 'concerto',
    'pair_sha256': args.pair_sha256,
    'checkpoint_sha256': args.checkpoint_sha256,
    'temporal_query_ids': ['q0', 'q1'],
    'runtime_s': 1.25,
    'peak_memory_bytes': 4096,
    'output_arrays': {
        'path': 'query_evidence.npz',
        'sha256': hashlib.sha256(content).hexdigest(),
        'byte_count': len(content),
    },
}), encoding='utf-8')
""",
        encoding="utf-8",
    )
    return script


def _write_backend_config(
    tmp_path: Path,
    *,
    checkout: Path,
    source_manifest: Path,
    checkpoint: Path,
    checkpoint_sha256: str | None,
    command: list[str] | None,
) -> Path:
    source_content = source_manifest.read_bytes()
    status = (
        "PASS"
        if checkpoint_sha256 is not None
        else "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    )
    payload = {
        "schema_version": 1,
        "backend_id": "RESCENE_CONCERTO_TWO_VISIT_V1",
        "status": status,
        "ranking_eligible": status == "PASS",
        "source": {
            "checkout": str(checkout),
            "commit": _git(checkout, "rev-parse", "HEAD"),
            "manifest": {
                "path": str(source_manifest),
                "sha256": hashlib.sha256(source_content).hexdigest(),
                "byte_count": len(source_content),
            },
        },
        "adapter_contract": {
            "feature_schema": "rgb",
            "neural_voxel_size_m": 0.02,
        },
        "checkpoint": {
            "status": status,
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "selected_backbone": "Concerto",
            "selection_policy": "official_or_source_bound_training_only",
        },
        "environment": {
            "status": "HERMETIC_TEST" if status == "PASS" else "NOT_PROVISIONED",
            "python": sys.executable if status == "PASS" else None,
        },
        "executor": {"command": command, "timeout_s": 30.0},
        "random_initialization": {
            "allowed_for_plumbing": True,
            "ranking_eligible": False,
        },
    }
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_subprocess_output_restores_adapter_token_order(tmp_path: Path) -> None:
    checkout, source_manifest = _source(tmp_path)
    checkpoint = tmp_path / "rescene.ckpt"
    checkpoint.write_bytes(b"author-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    backend = ReSceneBackend(
        checkout=checkout,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        source_manifest=source_manifest,
        executor_command=(sys.executable, str(_fake_executor(tmp_path))),
        expected_feature_schema="rgb",
        expected_neural_voxel_size_m=0.02,
    )

    result = backend.infer(_pair())

    assert result.status == "PASS"
    assert result.ranking_eligible is True
    assert result.checkpoint_sha256 == checkpoint_sha256
    assert result.runtime_s == 1.25
    assert result.peak_memory_bytes == 4096
    assert np.array_equal(
        result.query_masks,
        np.asarray([[False, True], [True, False]], dtype=np.bool_),
    )
    assert np.allclose(result.token_scores, [[0.1, 0.9], [0.8, 0.2]])
    assert result.diagnostics["token_order_restored"] == "true"


def test_feature_schema_mismatch_fails_before_subprocess(tmp_path: Path) -> None:
    checkout, source_manifest = _source(tmp_path)
    checkpoint = tmp_path / "rescene.ckpt"
    checkpoint.write_bytes(b"author-checkpoint")
    backend = ReSceneBackend(
        checkout=checkout,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_manifest=source_manifest,
        executor_command=(sys.executable, str(_fake_executor(tmp_path))),
        expected_feature_schema="rgb_normals",
        expected_neural_voxel_size_m=0.02,
    )

    result = backend.infer(_pair())

    assert result.status == "BLOCKED_EXTERNAL_SOURCE"
    assert result.ranking_eligible is False
    assert result.diagnostics["reason"] == "adapter_feature_schema_mismatch"


def test_missing_checkpoint_publishes_only_a_nonranking_blocked_receipt(
    tmp_path: Path,
) -> None:
    checkout, source_manifest = _source(tmp_path)
    pair_root = tmp_path / "pair"
    write_neural_sample_artifact(_pair(), pair_root)
    config = _write_backend_config(
        tmp_path,
        checkout=checkout,
        source_manifest=source_manifest,
        checkpoint=tmp_path / "missing.ckpt",
        checkpoint_sha256=None,
        command=None,
    )

    output = run_rescene_pair_backend(config, pair_root, tmp_path / "output")

    evidence = json.loads(output.evidence_manifest.read_text(encoding="utf-8"))
    receipt = json.loads(output.receipt.read_text(encoding="utf-8"))
    assert evidence["status"] == "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    assert evidence["ranking_eligible"] is False
    assert evidence["arrays"] is None
    assert output.evidence_arrays is None
    assert receipt["status"] == "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    assert receipt["ranking_eligible"] is False
    assert receipt["pair_sha256"] == _pair().content_sha256()
    assert receipt["evidence"]["path"] == "evidence.json"
    assert receipt["evidence"]["sha256"] == hashlib.sha256(
        output.evidence_manifest.read_bytes()
    ).hexdigest()


def test_pass_run_atomically_publishes_hash_bound_restored_arrays(tmp_path: Path) -> None:
    checkout, source_manifest = _source(tmp_path)
    pair_root = tmp_path / "pair"
    write_neural_sample_artifact(_pair(), pair_root)
    checkpoint = tmp_path / "rescene.ckpt"
    checkpoint.write_bytes(b"author-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = _write_backend_config(
        tmp_path,
        checkout=checkout,
        source_manifest=source_manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        command=[sys.executable, str(_fake_executor(tmp_path))],
    )

    output = run_rescene_pair_backend(config, pair_root, tmp_path / "output")

    assert output.evidence_arrays is not None
    manifest = json.loads(output.evidence_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["ranking_eligible"] is True
    with np.load(output.evidence_arrays, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"query_masks", "token_scores", "query_scores"}
        assert np.array_equal(arrays["query_masks"], [[False, True], [True, False]])
    assert manifest["arrays"]["sha256"] == hashlib.sha256(
        output.evidence_arrays.read_bytes()
    ).hexdigest()


def test_backend_run_rejects_overwrite_and_random_ranking_claim(tmp_path: Path) -> None:
    checkout, source_manifest = _source(tmp_path)
    pair_root = tmp_path / "pair"
    write_neural_sample_artifact(_pair(), pair_root)
    config = _write_backend_config(
        tmp_path,
        checkout=checkout,
        source_manifest=source_manifest,
        checkpoint=tmp_path / "missing.ckpt",
        checkpoint_sha256=None,
        command=None,
    )
    output = tmp_path / "output"
    run_rescene_pair_backend(config, pair_root, output)
    with pytest.raises(BackendRunError, match="already exists"):
        run_rescene_pair_backend(config, pair_root, output)

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["random_initialization"]["ranking_eligible"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BackendRunError, match="random initialization"):
        run_rescene_pair_backend(config, pair_root, tmp_path / "second")


def test_unbound_checkout_file_is_rejected_before_inference(tmp_path: Path) -> None:
    checkout, source_manifest = _source(tmp_path)
    (checkout / "unbound.py").write_text("raise RuntimeError\n", encoding="utf-8")
    backend = ReSceneBackend(
        checkout=checkout,
        checkpoint=tmp_path / "missing.ckpt",
        source_manifest=source_manifest,
    )

    result = backend.infer(_pair())

    assert result.status == "BLOCKED_EXTERNAL_SOURCE"
    assert result.diagnostics["reason"] == "source_identity_mismatch"


def test_executor_rejects_nonpermutation_token_order(tmp_path: Path) -> None:
    checkout, source_manifest = _source(tmp_path)
    checkpoint = tmp_path / "rescene.ckpt"
    checkpoint.write_bytes(b"author-checkpoint")
    executor = _fake_executor(tmp_path)
    executor.write_text(
        executor.read_text(encoding="utf-8").replace(
            "token_indices=np.asarray([1, 0]",
            "token_indices=np.asarray([0, 0]",
        ),
        encoding="utf-8",
    )
    backend = ReSceneBackend(
        checkout=checkout,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_manifest=source_manifest,
        executor_command=(sys.executable, str(executor)),
    )

    with pytest.raises(ReSceneBackendError, match="exact adapter-token permutation"):
        backend.infer(_pair())


def test_checked_in_config_records_the_external_checkpoint_blocker() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/evaluation/rescene_two_visit_backend.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = root / config["source"]["manifest"]["path"]
    source_content = source_path.read_bytes()

    assert config["status"] == "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    assert config["ranking_eligible"] is False
    assert config["checkpoint"]["sha256"] is None
    assert config["executor"]["command"] is None
    assert config["random_initialization"]["ranking_eligible"] is False
    assert config["source"]["manifest"]["sha256"] == hashlib.sha256(
        source_content
    ).hexdigest()
    assert config["source"]["manifest"]["byte_count"] == len(source_content)
