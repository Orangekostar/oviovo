from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from scripts.evaluation.audit_persist4d_rescene_checkpoint import (
    CheckpointAuditError,
    audit_checkpoint,
    audit_checkpoint_structure,
    compare_model_topology,
    load_checkpoint_provenance,
    verify_bound_git_source,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> str:
    torch.save(
        {
            "epoch": 4,
            "global_step": 19,
            "state_dict": state_dict,
            "hyper_parameters": {"model": {"num_queries": 100}},
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_checkout(tmp_path: Path, name: str) -> dict[str, object]:
    checkout = tmp_path / name
    checkout.mkdir()
    source = checkout / "model.py"
    source.write_text(f'SOURCE = "{name}"\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    _git(checkout, "config", "user.email", "test@example.invalid")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", "model.py")
    _git(checkout, "commit", "-q", "-m", "source")
    content = source.read_bytes()
    return {
        "checkout_path": str(checkout),
        "commit": _git(checkout, "rev-parse", "HEAD"),
        "required_files": [
            {
                "path": "model.py",
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        ],
    }


def _provenance_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint_sha256 = _checkpoint(
        checkpoint,
        {"model.weight": torch.ones((2, 3), dtype=torch.float32)},
    )
    concerto = tmp_path / "concerto.pth"
    concerto.write_bytes(b"concerto")
    training_config = tmp_path / "train.yaml"
    training_config.write_bytes(b"trainer:\n  max_epochs: 405\n")
    persist4d = _source_checkout(tmp_path, "persist4d")
    upstream = _source_checkout(tmp_path, "upstream")
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": "PERSIST4D_RESCENE4D_C_T2_REPRO_V1",
        "status": "PASS",
        "classification": "SOURCE_BOUND_RESCENE_REPRODUCTION",
        "persist4d_repository": "git@github.com:Orangekostar/Persist4D.git",
        "persist4d_evidence_commit": persist4d["commit"],
        "canonical_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "byte_count": checkpoint.stat().st_size,
            "regular_file": True,
            "symlink": False,
        },
        "training_source": {
            "official_upstream_base_commit": upstream["commit"],
            "persist4d_runtime_commit": persist4d["commit"],
        },
        "training_config": {
            "path": str(training_config),
            "sha256": hashlib.sha256(training_config.read_bytes()).hexdigest(),
            "byte_count": len(training_config.read_bytes()),
        },
        "reproduction_metrics": {
            "validation_sequence_count": 154,
            "t_mAP": 27.939,
            "t_REC": 40.849,
            "overall_mAP": 36.314,
            "paper_target_t_mAP": 34.8,
        },
        "concerto_initialization": {
            "path": str(concerto),
            "revision": "4" * 40,
            "sha256": hashlib.sha256(concerto.read_bytes()).hexdigest(),
            "byte_count": len(concerto.read_bytes()),
            "required_for_runtime": True,
        },
        "official_author_checkpoint": False,
        "source_audit": {
            "persist4d": persist4d,
            "pristine_upstream_rescene4d": upstream,
        },
    }
    config = tmp_path / "provenance.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config, payload


def test_checkpoint_structure_reports_compact_finite_inventory(tmp_path: Path) -> None:
    path = tmp_path / "rescene.ckpt"
    digest = _checkpoint(
        path,
        {
            "model.encoder.weight": torch.ones((2, 3), dtype=torch.float32),
            "model.decoder.bias": torch.zeros((2,), dtype=torch.float16),
            "criterion.empty_weight": torch.ones((4,), dtype=torch.float32),
        },
    )

    result, state_dict = audit_checkpoint_structure(path, digest)

    assert result == {
        "byte_count": path.stat().st_size,
        "epoch": 4,
        "global_step": 19,
        "nonfinite_tensor_count": 0,
        "prefix_counts": {"criterion": 1, "model": 2},
        "sha256": digest,
        "state_dict_key_count": 3,
        "status": "CHECKPOINT_STRUCTURE_PASS",
        "tensor_dtype_counts": {"torch.float16": 1, "torch.float32": 2},
        "tensor_element_count": 12,
        "top_level_keys": [
            "epoch",
            "global_step",
            "hyper_parameters",
            "state_dict",
        ],
    }
    assert set(state_dict) == {
        "model.encoder.weight",
        "model.decoder.bias",
        "criterion.empty_weight",
    }


def test_checkpoint_hash_mismatch_fails_before_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "rescene.ckpt"
    _checkpoint(path, {"model.weight": torch.ones(1)})

    with pytest.raises(CheckpointAuditError, match="SHA-256 mismatch"):
        audit_checkpoint_structure(path, "f" * 64)


def test_checkpoint_nonfinite_tensor_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "rescene.ckpt"
    digest = _checkpoint(path, {"model.weight": torch.tensor([float("nan")])})

    with pytest.raises(CheckpointAuditError, match="non-finite"):
        audit_checkpoint_structure(path, digest)


def test_topology_requires_exact_key_and_shape_consumption() -> None:
    checkpoint = {
        "model.encoder.weight": torch.ones((2, 3)),
        "model.decoder.bias": torch.zeros((2,)),
        "criterion.empty_weight": torch.ones((4,)),
    }
    model = {
        "encoder.weight": torch.empty((2, 3)),
        "decoder.bias": torch.empty((2,)),
    }

    assert compare_model_topology(checkpoint, model) == {
        "checkpoint_model_key_count": 2,
        "ignored_or_dropped_model_keys": [],
        "missing_keys": [],
        "model_key_count": 2,
        "shape_mismatches": [],
        "status": "MODEL_TOPOLOGY_STRICT_COMPATIBLE",
        "unexpected_keys": [],
    }

    bad_model = {
        "encoder.weight": torch.empty((3, 2)),
        "new.bias": torch.empty((1,)),
    }
    failed = compare_model_topology(checkpoint, bad_model)
    assert failed["status"] == "UPSTREAM_TOPOLOGY_COMPATIBILITY_FAIL"
    assert failed["missing_keys"] == ["new.bias"]
    assert failed["unexpected_keys"] == ["decoder.bias"]
    assert failed["shape_mismatches"] == [
        {
            "checkpoint": [2, 3],
            "key": "encoder.weight",
            "model": [3, 2],
        }
    ]
    assert failed["ignored_or_dropped_model_keys"] == []


def test_bound_git_source_rejects_dirty_or_wrong_commit(tmp_path: Path) -> None:
    checkout = tmp_path / "source"
    checkout.mkdir()
    source = checkout / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    _git(checkout, "config", "user.email", "test@example.invalid")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", "model.py")
    _git(checkout, "commit", "-q", "-m", "source")
    commit = _git(checkout, "rev-parse", "HEAD")
    content = source.read_bytes()
    binding = {
        "path": "model.py",
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }

    result = verify_bound_git_source(checkout, commit, [binding])
    assert result["status"] == "SOURCE_IDENTITY_PASS"
    assert result["commit"] == commit
    assert result["required_file_count"] == 1

    with pytest.raises(CheckpointAuditError, match="commit mismatch"):
        verify_bound_git_source(checkout, "0" * 40, [binding])

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(CheckpointAuditError, match="working tree is dirty"):
        verify_bound_git_source(checkout, commit, [binding])


def test_checkpoint_provenance_enforces_local_reproduction_classification(
    tmp_path: Path,
) -> None:
    config, payload = _provenance_fixture(tmp_path)

    loaded = load_checkpoint_provenance(config)
    assert loaded["artifact_id"] == "PERSIST4D_RESCENE4D_C_T2_REPRO_V1"
    assert loaded["canonical_checkpoint"] == payload["canonical_checkpoint"]
    assert loaded["official_author_checkpoint"] is False

    payload["official_author_checkpoint"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointAuditError, match="official author"):
        load_checkpoint_provenance(config)


def test_audit_checkpoint_writes_canonical_strict_topology_receipt(
    tmp_path: Path,
) -> None:
    config, _ = _provenance_fixture(tmp_path)
    output = tmp_path / "summary.json"

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty((2, 3)))

    result = audit_checkpoint(
        config,
        output,
        _model_factory=lambda _hyperparameters, _concerto, _upstream: TinyModel(),
    )

    assert result["status"] == "C0_C1_PASS"
    assert result["checkpoint_structure"]["status"] == "CHECKPOINT_STRUCTURE_PASS"
    assert result["model_topology"]["status"] == "MODEL_TOPOLOGY_STRICT_COMPATIBLE"
    assert result["model_topology"]["strict_load"] is True
    assert result["source_identity"]["persist4d"]["status"] == "SOURCE_IDENTITY_PASS"
    assert result["canonical_checkpoint"]["path"].endswith("model.ckpt")
    assert result["provenance_config"]["path"] == str(config)
    assert len(result["producer_git_commit"]) == 40
    assert result["producer_command"]
    assert json.loads(output.read_text(encoding="utf-8")) == result
