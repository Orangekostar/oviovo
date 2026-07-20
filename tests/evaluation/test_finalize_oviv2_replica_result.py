from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation import finalize_oviv2_replica_result as finalizer
from scripts.evaluation.finalize_oviv2_replica_result import (
    _dense_cache_prefix,
    _evaluation_contract,
    _load_verified_batch_config,
    _repeat_scene_evaluation,
    _validate_dense_provenance,
    _verify_run_artifacts,
    _verify_scene_config_binding,
)
from src.evaluation.oviv2_result import REPLICA8_SCENES


FROZEN_DENSE_PROVENANCE = {
    "backend": "radseg",
    "source_commit": "a" * 40,
    "radio_commit": "b" * 40,
    "model_id": "radseg:model",
    "model_sha256": "c" * 64,
    "auxiliary_model_sha256": "d" * 64,
    "vocabulary_sha256": "e" * 64,
    "prompt_sha256": "f" * 64,
    "inference_config_sha256": "1" * 64,
    "language_model_id": "siglip2",
    "language_model_revision": "2" * 40,
    "language_model_sha256": "3" * 64,
}


def _dense_audit(index: int) -> dict:
    consumed_hashes = {"frame000000.npz": f"{index + 1:064x}"}
    producer_hashes = {
        **consumed_hashes,
        "frame000001.npz": f"{index + 101:064x}",
    }
    consumed_prefix = _dense_cache_prefix(consumed_hashes)
    producer_prefix = _dense_cache_prefix(producer_hashes)
    return {
        "mode": "cached_probabilities",
        "cache_prefix_sha256": consumed_prefix,
        "producer_cache_prefix_sha256": producer_prefix,
        "cache_files_sha256": consumed_hashes,
        "producer_cache_files_sha256": producer_hashes,
        "provenance": {
            **FROZEN_DENSE_PROVENANCE,
            "cache_prefix_sha256": consumed_prefix,
        },
        "producer_provenance": {
            **FROZEN_DENSE_PROVENANCE,
            "cache_prefix_sha256": producer_prefix,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_legacy_batch_defaults_to_owner_evaluation_directory() -> None:
    assert _evaluation_contract({}, {}) == ("owner_authoritative", "evaluation", 0.5)


def test_stage3_batch_selects_fused_directory_and_frozen_weight() -> None:
    assert _evaluation_contract(
        {
            "semantic_head": "fused_uncertainty",
            "fusion_entity_weight_scale": 0.49,
        },
        {
            "dense_semantic_mode": "cached_probabilities",
            "fusion_semantic_mode": "uncertainty_linear",
            "fusion_entity_weight_scale": 0.49,
        },
    ) == ("fused_uncertainty", "evaluation_fused", 0.49)


def test_stage3_batch_rejects_nonfrozen_fusion_weight() -> None:
    with pytest.raises(ValueError, match="frozen fusion_entity_weight_scale"):
        _evaluation_contract(
            {
                "semantic_head": "fused_uncertainty",
                "fusion_entity_weight_scale": 0.49,
            },
            {
                "dense_semantic_mode": "cached_probabilities",
                "fusion_semantic_mode": "uncertainty_linear",
                "fusion_entity_weight_scale": 0.5,
            },
        )


@pytest.mark.parametrize("head", ["dense_only", "fused_uncertainty", "unknown"])
def test_nonlegacy_head_requires_matching_scene_config(head: str) -> None:
    with pytest.raises(ValueError, match="semantic_head|dense|fusion"):
        _evaluation_contract({"semantic_head": head}, {})


def test_batch_config_bytes_must_match_recorded_hash(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    _write_json(base_path, {"fusion_entity_weight_scale": 0.49})
    config_path = tmp_path / "batch.json"
    _write_json(
        config_path,
        {
            "semantic_head": "fused_uncertainty",
            "base_runner_config": str(base_path),
        },
    )
    expected = hashlib.sha256(config_path.read_bytes()).hexdigest()
    base_expected = hashlib.sha256(base_path.read_bytes()).hexdigest()
    batch = {
        "config_path": str(config_path),
        "config_sha256": expected,
        "base_runner_config_path": str(base_path.resolve()),
        "base_runner_config_sha256": base_expected,
    }

    assert _load_verified_batch_config(batch)["semantic_head"] == "fused_uncertainty"

    config_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="batch config hash mismatch"):
        _load_verified_batch_config(batch)

    _write_json(
        config_path,
        {
            "semantic_head": "fused_uncertainty",
            "base_runner_config": str(base_path),
        },
    )
    base_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base runner config hash mismatch"):
        _load_verified_batch_config(batch)


def test_scene_config_is_bound_to_record_and_run_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "room0.json"
    config = {"scene": "room0", "fusion_entity_weight_scale": 0.49}
    _write_json(config_path, config)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(
        json.dumps(config, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    record = {
        "scene_config_sha256": config_sha256,
        "config_hash": config_hash,
    }
    run = {"config_hash": config_hash}

    assert _verify_scene_config_binding(config_path, record, run) == config

    run["config_hash"] = "0" * 64
    with pytest.raises(ValueError, match="canonical config hash mismatch"):
        _verify_scene_config_binding(config_path, record, run)


def test_dense_provenance_must_match_across_scenes_and_frozen_config() -> None:
    runs = {
        scene: {"dense_semantics": _dense_audit(index)}
        for index, scene in enumerate(REPLICA8_SCENES)
    }

    digest = _validate_dense_provenance(runs, FROZEN_DENSE_PROVENANCE)
    assert len(digest) == 64

    runs["office4"]["dense_semantics"]["producer_provenance"][
        "model_sha256"
    ] = "9" * 64
    runs["office4"]["dense_semantics"]["provenance"]["model_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="dense producer provenance differs"):
        _validate_dense_provenance(runs, FROZEN_DENSE_PROVENANCE)


def test_dense_cache_prefix_must_be_internally_bound_per_scene() -> None:
    runs = {
        scene: {"dense_semantics": _dense_audit(index)}
        for index, scene in enumerate(REPLICA8_SCENES)
    }
    runs["office4"]["dense_semantics"]["cache_prefix_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="dense cache prefix mismatch"):
        _validate_dense_provenance(runs, FROZEN_DENSE_PROVENANCE)


def test_run_artifact_manifest_must_cover_exact_scene_file_set(
    tmp_path: Path,
) -> None:
    scene_root = tmp_path / "room0"
    artifact = scene_root / "final" / "snapshot.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"snapshot")

    with pytest.raises(ValueError, match="artifact file contract mismatch"):
        _verify_run_artifacts(scene_root, {"artifact_checksums": {}})

    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _verify_run_artifacts(
        scene_root,
        {"artifact_checksums": {"final/snapshot.bin": checksum}},
    )

    (scene_root / "unrecorded.bin").write_bytes(b"unrecorded")
    with pytest.raises(ValueError, match="artifact file contract mismatch"):
        _verify_run_artifacts(
            scene_root,
            {"artifact_checksums": {"final/snapshot.bin": checksum}},
        )


def test_repeat_evaluation_ignores_stale_directory_and_runs_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_root = tmp_path / "room0"
    repeat_root = tmp_path / "repeated_evaluation" / "room0"
    repeat_root.mkdir(parents=True)
    (repeat_root / "stale.txt").write_text("stale", encoding="utf-8")
    calls: list[Path] = []

    def fake_evaluate(args) -> None:
        calls.append(Path(args.output))
        Path(args.output).mkdir(parents=True)

    def fake_verify(original: Path, repeated: Path) -> dict[str, str]:
        assert original == scene_root / "evaluation_fused"
        assert repeated.is_dir()
        assert repeated != repeat_root
        return {"metrics.json": "a" * 64}

    monkeypatch.setattr(finalizer, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        finalizer,
        "verify_byte_identical_evaluation_dirs",
        fake_verify,
    )

    hashes = _repeat_scene_evaluation(
        "room0",
        scene_root,
        {"gt_mesh": "/gt.ply", "gt_info": "/info.json", "manifest": "/m.json"},
        repeat_root,
        "fused_uncertainty",
        0.49,
    )

    assert hashes == {"metrics.json": "a" * 64}
    assert len(calls) == 1
    assert (repeat_root / "stale.txt").read_text(encoding="utf-8") == "stale"
