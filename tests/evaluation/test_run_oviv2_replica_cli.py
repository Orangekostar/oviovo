from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import scripts.run_oviv2_replica as runner_module
from scripts.run_oviv2_replica import _runtime_config, parse_args, run
from src.oviv2.snapshot import VoxelMapSnapshot


_ABSENT = object()
_PRECISION_CONFIG = Path("configs/oviv2_replica_room0_precision_stage1.json")
_STAGE1_CONFIG_FIELDS = (
    "semantic_mode",
    "feature_mode",
    "association_min_directed_overlap",
    "association_bounds_expansion_m",
    "association_max_centroid_distance_m",
    "association_minimum_score",
    "association_geometry_weight",
    "association_overlap_weight",
    "association_visual_weight",
    "association_semantic_weight",
    "association_temporal_weight",
    "semantic_conflict_confidence",
    "semantic_conflict_visual_override",
    "ambiguous_edge_score",
    "third_view_min_score",
    "prototype_top_k",
    "prototype_merge_cosine",
    "view_top_k",
    "view_minimum_novelty_cosine",
)


def _write_fixture(
    tmp_path: Path,
    *,
    cache_frames: int = 2,
    frontend_manifest_frames: int | None = None,
    cache_features: bool = False,
    frontend_feature_model_id: object = _ABSENT,
    frontend_clip_model_sha256: object = _ABSENT,
    config_feature_model_id: object = _ABSENT,
) -> Path:
    dataset = tmp_path / "dataset"
    results = dataset / "results"
    cache = tmp_path / "cache"
    results.mkdir(parents=True)
    cache.mkdir()
    poses: list[str] = []
    for frame_id in range(2):
        Image.fromarray(np.full((32, 32, 3), 100 + frame_id, dtype=np.uint8)).save(
            results / f"frame{frame_id:06d}.jpg"
        )
        Image.fromarray(np.full((32, 32), 6554, dtype=np.uint16)).save(
            results / f"depth{frame_id:06d}.png"
        )
        pose = np.eye(4)
        pose[0, 3] = frame_id * 0.02
        poses.append(" ".join(str(value) for value in pose.reshape(-1)))
    (dataset / "traj.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")
    for cache_id in range(cache_frames):
        mask = np.zeros((1, 32, 32), dtype=bool)
        mask[:, 8:24, 8:24] = True
        payload = {
            "mask": mask,
            "xyxy": np.asarray([[8, 8, 24, 24]], dtype=np.float32),
            "confidence": np.asarray([0.9], dtype=np.float32),
            "class_id": np.asarray([0], dtype=np.int64),
            "classes": ["chair"],
        }
        if cache_features:
            payload["image_feats"] = np.asarray([[3.0, 4.0]], dtype=np.float32)
            payload["text_feats"] = np.asarray([[0.0, 5.0]], dtype=np.float32)
        with gzip.open(cache / f"frame{cache_id:06d}.pkl.gz", "wb") as stream:
            pickle.dump(payload, stream)
    if frontend_manifest_frames is not None:
        cache_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(cache.glob("frame*.pkl.gz"))
        }
        frontend_manifest = {
            "method": "OVIV2",
            "scene": "room0",
            "frame_count": frontend_manifest_frames,
            "algorithm_hash": "fixture-frontend",
            "cache_files_sha256": cache_hashes,
        }
        if frontend_feature_model_id is not _ABSENT:
            frontend_manifest["feature_model_id"] = frontend_feature_model_id
        if frontend_clip_model_sha256 is not _ABSENT:
            frontend_manifest["provenance_sha256"] = {
                "clip_model": frontend_clip_model_sha256,
            }
        (cache / "frontend_manifest.json").write_text(
            json.dumps(frontend_manifest),
            encoding="utf-8",
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "fixture",
                "dataset": "Replica",
                "frame_selection": {"start": 0, "stop_exclusive": 20, "stride": 10},
                "vocabulary": {"classes": ["wall", "floor", "ceiling", "chair"]},
                "aliases": {},
                "scenes": [{"scene": "room0"}],
                "protocol": {"geometry_primary_threshold_m": 0.05},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config_payload = {
        "scene": "room0",
        "dataset_root": str(dataset),
        "frontend_cache_dir": str(cache),
        "manifest": str(manifest),
        "gt_mesh": str(tmp_path / "unused.ply"),
        "gt_info": str(tmp_path / "unused.json"),
        "source_start": 0,
        "source_stride": 10,
        "voxel_size_m": 0.05,
        "block_resolution": 8,
        "pixel_stride": 2,
        "min_valid_points": 1,
        "confirm_hits": 2,
        "checkpoint_interval": 1,
        "structure_enabled": True,
        "structure_min_component_pixels": 4,
        "structure_min_component_fraction": 0.0,
        "structure_object_exclusion_dilation": 1,
    }
    if config_feature_model_id is not _ABSENT:
        config_payload["feature_model_id"] = config_feature_model_id
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    return config


def _args(config: Path, output: Path, *extra: str, num_frames: int = 2):
    return parse_args(
        [
            "--config",
            str(config),
            "--output",
            str(output),
            "--num-frames",
            str(num_frames),
            "--skip-evaluation",
            *extra,
        ]
    )


def _enable_stage1(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    precision = json.loads(_PRECISION_CONFIG.read_text(encoding="utf-8"))
    config.update({name: precision[name] for name in _STAGE1_CONFIG_FIELDS})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config


def _remove_cached_field(config_path: Path, cache_index: int, field: str) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_dir = Path(config["frontend_cache_dir"])
    cache_path = cache_dir / f"frame{cache_index:06d}.pkl.gz"
    with gzip.open(cache_path, "rb") as stream:
        payload = pickle.load(stream)
    payload.pop(field)
    with gzip.open(cache_path, "wb") as stream:
        pickle.dump(payload, stream)
    manifest_path = cache_dir / "frontend_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_files_sha256"][cache_path.name] = hashlib.sha256(
        cache_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_precision_config_enables_feature_aware_owner_semantics() -> None:
    base = json.loads(Path("configs/oviv2_replica_room0.json").read_text())
    config = json.loads(_PRECISION_CONFIG.read_text())
    runtime = _runtime_config(config)

    assert all(config[name] == value for name, value in base.items())
    assert config["semantic_mode"] == "owner_authoritative"
    assert config["feature_mode"] == "cached_image"
    assert runtime.tracker.association is runtime.registry.association
    assert runtime.tracker.association.visual_weight > 0.0
    assert runtime.registry.association.semantic_weight > 0.0
    assert runtime.tracker.ambiguous_edge_score == pytest.approx(0.60)
    assert runtime.tracker.third_view_min_score == pytest.approx(0.70)
    assert runtime.registry.prototype_top_k == 3
    assert runtime.registry.prototype_merge_cosine == pytest.approx(0.90)
    assert runtime.registry.view_top_k == 10
    assert runtime.registry.view_minimum_novelty_cosine == pytest.approx(0.10)


def test_runtime_config_without_stage1_fields_keeps_legacy_association() -> None:
    runtime = _runtime_config(
        json.loads(Path("configs/oviv2_replica_room0.json").read_text())
    )

    assert runtime.tracker.association.max_centroid_distance_m == pytest.approx(0.5)
    assert runtime.registry.association.max_centroid_distance_m == pytest.approx(0.6)
    assert runtime.tracker.association is not runtime.registry.association


@pytest.mark.parametrize(
    ("field", "value"),
    [("semantic_mode", "evidence"), ("feature_mode", "uncached")],
)
def test_runtime_config_rejects_unsupported_stage1_modes(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        _runtime_config({field: value})


def test_stage1_preflight_rejects_cache_without_image_features(tmp_path: Path) -> None:
    config_path = _write_fixture(
        tmp_path,
        frontend_manifest_frames=2,
        cache_features=False,
        frontend_clip_model_sha256="a" * 64,
    )
    _enable_stage1(config_path)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=r"frame000000\.pkl\.gz.*image_feats"):
        run(_args(config_path, output))

    assert not output.exists()


def test_stage1_preflight_checks_image_features_in_every_frame(tmp_path: Path) -> None:
    config_path = _write_fixture(
        tmp_path,
        frontend_manifest_frames=2,
        cache_features=True,
        frontend_clip_model_sha256="a" * 64,
    )
    _enable_stage1(config_path)
    _remove_cached_field(config_path, 1, "image_feats")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=r"frame000001\.pkl\.gz.*image_feats"):
        run(_args(config_path, output))

    assert not output.exists()


def test_stage1_preflight_requires_feature_model_provenance(tmp_path: Path) -> None:
    config_path = _write_fixture(
        tmp_path,
        frontend_manifest_frames=2,
        cache_features=True,
    )
    _enable_stage1(config_path)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="feature_model_id"):
        run(_args(config_path, output))

    assert not output.exists()


def test_stage1_fields_change_algorithm_hash() -> None:
    config = json.loads(_PRECISION_CONFIG.read_text(encoding="utf-8"))
    changed = dict(config)
    changed["association_visual_weight"] = 0.31

    assert runner_module.algorithm_hash(changed) != runner_module.algorithm_hash(config)


def test_runner_rejects_stale_algorithm_hash_before_output(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["association_visual_weight"] = 0.31
    config["algorithm_hash"] = "stale"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="algorithm_hash"):
        run(_args(config_path, output))

    assert not output.exists()


def test_preflight_fails_before_creating_output_for_missing_cache(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, cache_frames=1)
    output = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match="frame000001"):
        run(_args(config, output))

    assert not output.exists()


def test_preflight_accepts_prefix_of_verified_frontend_manifest(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, frontend_manifest_frames=2)
    output = tmp_path / "run"

    manifest = run(_args(config, output, num_frames=1))

    assert manifest["final_revision"] == 1
    assert manifest["frontend_algorithm_hash"] == "fixture-frontend"


def test_runner_derives_feature_model_id_from_legacy_frontend_manifest(tmp_path: Path) -> None:
    clip_hash = "a" * 64
    config = _write_fixture(
        tmp_path,
        frontend_manifest_frames=2,
        cache_features=True,
        frontend_clip_model_sha256=clip_hash,
    )

    manifest = run(_args(config, tmp_path / "run", num_frames=1))

    assert manifest["frontend_feature_model_id"] == f"clip-sha256:{clip_hash}"


@pytest.mark.parametrize(
    ("frontend_feature_model_id", "frontend_clip_model_sha256"),
    [
        ("clip-explicit:manifest", _ABSENT),
        (_ABSENT, "a" * 64),
    ],
    ids=("explicit", "derived"),
)
def test_runner_rejects_config_feature_model_id_conflict(
    tmp_path: Path,
    frontend_feature_model_id: object,
    frontend_clip_model_sha256: object,
) -> None:
    config = _write_fixture(
        tmp_path,
        frontend_manifest_frames=2,
        frontend_feature_model_id=frontend_feature_model_id,
        frontend_clip_model_sha256=frontend_clip_model_sha256,
        config_feature_model_id="clip-explicit:config",
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="feature_model_id.*conflict"):
        run(_args(config, output, num_frames=1))

    assert not output.exists()


@pytest.mark.parametrize(
    ("frontend_feature_model_id", "frontend_clip_model_sha256"),
    [
        ("   ", _ABSENT),
        (_ABSENT, "not-a-sha256"),
    ],
    ids=("blank-explicit", "invalid-derived-hash"),
)
def test_runner_rejects_invalid_frontend_feature_model_source(
    tmp_path: Path,
    frontend_feature_model_id: object,
    frontend_clip_model_sha256: object,
) -> None:
    config = _write_fixture(
        tmp_path,
        frontend_manifest_frames=2,
        frontend_feature_model_id=frontend_feature_model_id,
        frontend_clip_model_sha256=frontend_clip_model_sha256,
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="feature_model_id|clip_model"):
        run(_args(config, output, num_frames=1))

    assert not output.exists()


def test_runner_writes_restoreable_voxel_contract_and_exact_frame_selection(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    output = tmp_path / "run"

    manifest = run(_args(config, output))

    final_snapshot = output / "final" / "oviv2_voxel_snapshot.npz"
    checkpoint = output / "checkpoints" / "latest_voxel_snapshot.npz"
    assert VoxelMapSnapshot.load(final_snapshot).metadata.frame_id == 10
    assert VoxelMapSnapshot.load(checkpoint).metadata.revision == 2
    assert manifest["frame_selection"]["source_frame_ids"] == [0, 10]
    assert (output / "final" / "oviv2_entities.jsonl").is_file()
    assert (output / "final" / "oviv2_instance_mesh.ply").is_file()
    assert (output / "timing.json").is_file()
    assert manifest["authoritative_state"] == "sparse_voxel_layers"
    assert manifest["dense_point_cloud_state"] is False
    assert manifest["semantic_mode"] == "owner_authoritative"
    assert manifest["feature_mode"] == "cached_optional"


def test_runner_passes_current_registry_semantics_to_final_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_fixture(tmp_path)
    captured: list[object] = []
    original = runner_module.derive_labeled_mesh

    def record_semantics(*args, **kwargs):
        captured.append(kwargs.get("entity_semantics"))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "derive_labeled_mesh", record_semantics)

    run(_args(config, tmp_path / "run"))

    assert len(captured) == 1
    assert isinstance(captured[0], dict)
    assert captured[0]
    assert all(semantic_id == 4 for semantic_id, _ in captured[0].values())


def test_runner_fuses_structure_without_allocating_structure_entities(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    output = tmp_path / "run"

    manifest = run(_args(config, output))

    with np.load(
        output / "final" / "oviv2_voxel_snapshot.npz" / "evidence.npz",
        allow_pickle=False,
    ) as evidence:
        semantic_ids = set(int(value) for value in evidence["semantic_ids"].reshape(-1))
    entity_lines = (
        output / "final" / "oviv2_entities.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    entities = [
        payload
        for line in entity_lines
        if line.strip()
        for payload in (json.loads(line),)
        if payload.get("record_type") == "entity"
    ]
    timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))

    assert semantic_ids & {1, 2, 3}
    assert all(entity["semantic_id"] not in {1, 2, 3} for entity in entities)
    assert all(record["structure_observation_count"] > 0 for record in timing["frames"])
    assert manifest["structure_frontend"]["enabled"] is True


def test_manifest_artifact_checksums_recompute_and_resume_is_idempotent(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    output = tmp_path / "run"
    original = run(_args(config, output))
    manifest_bytes = (output / "run_manifest.json").read_bytes()

    resumed = run(_args(config, output, "--resume"))

    assert resumed == original
    assert (output / "run_manifest.json").read_bytes() == manifest_bytes
    for relative, expected in original["artifact_checksums"].items():
        path = output / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_stage1_runner_records_precision_backend_and_frame_counters(
    tmp_path: Path,
) -> None:
    clip_hash = "a" * 64
    config_path = _write_fixture(
        tmp_path,
        frontend_manifest_frames=2,
        cache_features=True,
        frontend_clip_model_sha256=clip_hash,
    )
    config = _enable_stage1(config_path)
    runtime = _runtime_config(config)
    output = tmp_path / "run"

    manifest = run(_args(config_path, output))

    timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))
    snapshot = VoxelMapSnapshot.load(
        output / "final" / "oviv2_voxel_snapshot.npz"
    )
    assert all(
        isinstance(frame[name], int)
        for frame in timing["frames"]
        for name in (
            "matched_entity_count",
            "new_entity_count",
            "association_conflict_count",
            "revoked_edge_count",
        )
    )
    assert manifest["semantic_mode"] == "owner_authoritative"
    assert manifest["feature_mode"] == "cached_image"
    assert manifest["frontend_feature_model_id"] == f"clip-sha256:{clip_hash}"
    assert manifest["precision_backend"] == {
        "association": asdict(runtime.tracker.association),
        "tracker": {
            "ambiguous_edge_score": runtime.tracker.ambiguous_edge_score,
            "third_view_min_score": runtime.tracker.third_view_min_score,
        },
        "memory": {
            "prototype_top_k": runtime.registry.prototype_top_k,
            "prototype_merge_cosine": runtime.registry.prototype_merge_cosine,
            "view_top_k": runtime.registry.view_top_k,
            "view_minimum_novelty_cosine": (
                runtime.registry.view_minimum_novelty_cosine
            ),
        },
    }
    assert snapshot.metadata.schema_version == 2
    assert snapshot.registry is not None
