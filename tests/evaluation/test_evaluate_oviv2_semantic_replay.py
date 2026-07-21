from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_oviv2_replica import (
    SEMANTIC_REPLAY_SOURCE_PATHS,
    evaluate,
)
from scripts.run_oviv2_replica import _DenseCachePreflight
from src.evaluation.oviv2_semantic_replay import semantic_replay_algorithm_hash
from src.oviv2.snapshot import VoxelMapSnapshot
from tests.evaluation.test_evaluate_oviv2_replica_cli import (
    _dense_provenance,
    _fixture,
)


def _args(paths: dict[str, Path], output: Path, **overrides) -> argparse.Namespace:
    values = {
        "snapshot": paths["snapshot"],
        "entity_info": None,
        "gt_mesh": paths["gt_mesh"],
        "gt_info": paths["gt_info"],
        "manifest": paths["manifest"],
        "scene": "fixture",
        "output": output,
        "min_instance_vertices": 1,
        "semantic_head": "fused_uncertainty",
        "fusion_entity_weight_scale": 0.49,
        "semantic_evidence": None,
        "semantic_replay_manifest": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _replay_artifacts(paths: dict[str, Path], tmp_path: Path) -> tuple[Path, Path]:
    snapshot = VoxelMapSnapshot.load(paths["snapshot"])
    overlay = tmp_path / "semantic_evidence.npz"
    snapshot.evidence.save(overlay)
    overlay_hash = hashlib.sha256(overlay.read_bytes()).hexdigest()
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in SEMANTIC_REPLAY_SOURCE_PATHS.items()
    }
    dense_cache = tmp_path / "dense-cache"
    dense_cache.mkdir()
    dense_frame = dense_cache / "frame000000.npz"
    dense_frame.write_bytes(b"fixture-dense-cache")
    dense_hashes = {dense_frame.name: hashlib.sha256(dense_frame.read_bytes()).hexdigest()}
    dense_manifest_payload = {
        "source_frame_ids": [0],
        "cache_files_sha256": dense_hashes,
    }
    dense_manifest = dense_cache / "dense_manifest.json"
    dense_manifest.write_text(json.dumps(dense_manifest_payload), encoding="utf-8")
    config = tmp_path / "replay_config.json"
    config.write_text("{}", encoding="utf-8")
    base_run_manifest = tmp_path / "base_run_manifest.json"
    base_run_manifest.write_text("{}", encoding="utf-8")
    dense_config = {"entropy_power": 1.0}
    structure_config = {"enabled": True}
    class_powers: dict[str, float] = {}
    semantic_support_scale = 1.0
    provenance = _dense_provenance()
    algorithm_hash = semantic_replay_algorithm_hash(
        dense_config=dense_config,
        entropy_power_by_class=class_powers,
        semantic_support_scale=semantic_support_scale,
        source_hashes=source_hashes,
        structure_config=structure_config,
    )
    vocabulary_payload = {"classes": ["wall", "chair"], "aliases": {}}
    manifest = tmp_path / "replay_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "OVIV2-semantic-evidence-replay",
                "scene": "fixture",
                "structure_replayed": True,
                "base_snapshot_checksums": snapshot.checksums,
                "base_snapshot_revision": snapshot.metadata.revision,
                "overlay_sha256": overlay_hash,
                "algorithm_hash": algorithm_hash,
                "dense_config": dense_config,
                "entropy_power_by_class": class_powers,
                "semantic_support_scale": semantic_support_scale,
                "structure_config": structure_config,
                "source_hashes": source_hashes,
                "frame_count": 1,
                "source_frame_ids": [0],
                "source_frame_ids_hash": hashlib.sha256(b"[0]").hexdigest(),
                "benchmark_manifest_sha256": hashlib.sha256(
                    paths["manifest"].read_bytes()
                ).hexdigest(),
                "vocabulary_hash": hashlib.sha256(
                    json.dumps(
                        vocabulary_payload,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "dense_cache": {
                    "path": str(dense_cache),
                    "manifest_sha256": hashlib.sha256(
                        dense_manifest.read_bytes()
                    ).hexdigest(),
                    "cache_files_sha256": dense_hashes,
                    "producer_provenance": asdict(provenance),
                    "consumed_provenance": asdict(provenance),
                },
                "frontend_cache_hash": "f" * 64,
                "frontend_manifest": None,
                "dataset": {"root": "fixture"},
                "config_path": str(config),
                "config_hash": hashlib.sha256(b"{}").hexdigest(),
                "base_run_manifest": str(base_run_manifest),
                "base_run_manifest_sha256": hashlib.sha256(
                    base_run_manifest.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return overlay, manifest


def _stub_preflight(
    monkeypatch: pytest.MonkeyPatch,
    replay_manifest: Path,
) -> None:
    payload = json.loads(replay_manifest.read_text(encoding="utf-8"))
    dense = payload["dense_cache"]
    provenance = _dense_provenance()
    cache = _DenseCachePreflight(
        cache_dir=Path(dense["path"]),
        manifest_sha256=dense["manifest_sha256"],
        frame_count=1,
        source_frame_ids=(0,),
        image_shape=(1, 1),
        sample_stride=1,
        top_k=1,
        class_count=2,
        vocabulary_sha256="d" * 64,
        producer_provenance=provenance,
        consumed_provenance=provenance,
        cache_files_sha256=dense["cache_files_sha256"],
    )
    monkeypatch.setattr(
        "scripts.run_oviv2_replica._preflight",
        lambda *_args, **_kwargs: (
            object(),
            {},
            [0],
            payload["frontend_cache_hash"],
            payload["frontend_manifest"],
            cache,
        ),
    )
    monkeypatch.setattr(
        "scripts.evaluation.replay_oviv2_semantic_evidence._replica_dataset_binding",
        lambda *_args, **_kwargs: payload["dataset"],
    )


def test_semantic_overlay_control_preserves_all_metrics_and_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(
        tmp_path,
        schema_version=3,
        entity_probabilities=((2, 1.0),),
    )
    baseline_output = tmp_path / "baseline"
    baseline = evaluate(_args(paths, baseline_output))
    overlay, replay_manifest = _replay_artifacts(paths, tmp_path)
    _stub_preflight(monkeypatch, replay_manifest)
    replay_output = tmp_path / "replay"

    replayed = evaluate(
        _args(
            paths,
            replay_output,
            semantic_evidence=overlay,
            semantic_replay_manifest=replay_manifest,
        )
    )

    for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5"):
        assert replayed[name] == baseline[name]
    assert (
        replay_output / "gt_aligned_semantic_ids.npy"
    ).read_bytes() == (
        baseline_output / "gt_aligned_semantic_ids.npy"
    ).read_bytes()
    assert (
        replay_output / "gt_aligned_instance_ids.npy"
    ).read_bytes() == (
        baseline_output / "gt_aligned_instance_ids.npy"
    ).read_bytes()
    protocol = replayed["protocol"]["semantic_replay"]
    assert protocol["algorithm_hash"] == json.loads(
        replay_manifest.read_text(encoding="utf-8")
    )["algorithm_hash"]
    assert protocol["overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()
    assert len(protocol["manifest_sha256"]) == 64
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate(
            _args(
                paths,
                replay_output,
                semantic_evidence=overlay,
                semantic_replay_manifest=replay_manifest,
            )
        )


def test_semantic_overlay_requires_paired_manifest_and_valid_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(
        tmp_path,
        schema_version=3,
        entity_probabilities=((2, 1.0),),
    )
    overlay, replay_manifest = _replay_artifacts(paths, tmp_path)
    _stub_preflight(monkeypatch, replay_manifest)
    with pytest.raises(ValueError, match="together"):
        evaluate(
            _args(
                paths,
                tmp_path / "missing-manifest",
                semantic_evidence=overlay,
            )
        )


def test_semantic_overlay_rejects_unbound_source_and_benchmark(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        schema_version=3,
        entity_probabilities=((2, 1.0),),
    )
    overlay, replay_manifest = _replay_artifacts(paths, tmp_path)
    payload = json.loads(replay_manifest.read_text(encoding="utf-8"))
    payload["source_hashes"]["replay_core"] = "0" * 64
    replay_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        evaluate(
            _args(
                paths,
                tmp_path / "bad-source",
                semantic_evidence=overlay,
                semantic_replay_manifest=replay_manifest,
            )
        )

    overlay, replay_manifest = _replay_artifacts(paths, tmp_path / "second")
    payload = json.loads(replay_manifest.read_text(encoding="utf-8"))
    payload["benchmark_manifest_sha256"] = "0" * 64
    replay_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark manifest hash"):
        evaluate(
            _args(
                paths,
                tmp_path / "bad-benchmark",
                semantic_evidence=overlay,
                semantic_replay_manifest=replay_manifest,
            )
        )

    overlay.write_bytes(overlay.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        evaluate(
            _args(
                paths,
                tmp_path / "bad-hash",
                semantic_evidence=overlay,
                semantic_replay_manifest=replay_manifest,
            )
        )
