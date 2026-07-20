from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

import scripts.run_oviv2_replica as runner_module
from scripts.run_oviv2_replica import _runtime_config, parse_args, run
from src.oviv2.dense_semantics import (
    DenseSemanticFrame,
    DenseSemanticProvenance,
    load_dense_frame,
    sha256_file,
    write_dense_frame,
)
from src.oviv2.dense_projection import DenseSemanticConfig
from src.oviv2.snapshot import VoxelMapSnapshot


_ABSENT = object()
_PRECISION_CONFIG = Path("configs/oviv2_replica_room0_precision_stage1.json")
_STAGE2_CONFIG = Path("configs/oviv2_replica_room0_precision_stage2_radseg.json")
_SELECTED_STAGE2_CONFIG = Path(
    "configs/oviv2_replica_room0_precision_stage2_selected.json"
)
_STAGE3_CONFIG = Path("configs/oviv2_replica_room0_precision_stage3_fused.json")
_MODEL_HASH = "c" * 64
_DENSE_MANIFEST_KEYS = {
    "schema_version",
    "method",
    "scene",
    "frame_count",
    "source_frame_ids",
    "image_shape",
    "sample_stride",
    "top_k",
    "class_count",
    "vocabulary_sha256",
    "provenance",
    "cache_files_sha256",
}
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


def _args_with_evaluation(config: Path, output: Path, *, num_frames: int = 2):
    return parse_args(
        [
            "--config",
            str(config),
            "--output",
            str(output),
            "--num-frames",
            str(num_frames),
        ]
    )


def _enable_stage1(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    precision = json.loads(_PRECISION_CONFIG.read_text(encoding="utf-8"))
    config.update({name: precision[name] for name in _STAGE1_CONFIG_FIELDS})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config


def _write_dense_cache(config_path: Path, *, frame_count: int = 2) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    benchmark = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes_json = manifest_path.parent / "classes.json"
    classes_json.write_text(
        json.dumps(
            {"classes": benchmark["vocabulary"]["classes"], "aliases": {}},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    benchmark["vocabulary"]["source_path"] = str(classes_json)
    benchmark["vocabulary"]["source_sha256"] = hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(benchmark), encoding="utf-8")
    vocabulary_sha256 = hashlib.sha256(classes_json.read_bytes()).hexdigest()

    dense_dir = config_path.parent / "dense-cache"
    dense_dir.mkdir()
    cache_hashes: dict[str, str] = {}
    source_ids = [index * 10 for index in range(frame_count)]
    for cache_index, source_id in enumerate(source_ids):
        sampled_shape = (16, 16)
        class_ids = np.empty((*sampled_shape, 2), dtype=np.int64)
        class_ids[..., 0] = 4
        class_ids[..., 1] = 1
        probabilities = np.empty((*sampled_shape, 2), dtype=np.float32)
        probabilities[..., 0] = 0.7
        probabilities[..., 1] = 0.2
        frame = DenseSemanticFrame(
            cache_frame_id=cache_index,
            source_frame_id=source_id,
            image_shape=(32, 32),
            sample_stride=2,
            class_count=4,
            class_ids=class_ids,
            probabilities=probabilities,
            entropy=np.full(sampled_shape, 0.5, dtype=np.float32),
            margin=np.full(sampled_shape, 0.5, dtype=np.float32),
        )
        name = f"frame{cache_index:06d}.npz"
        write_dense_frame(dense_dir / name, frame)
        cache_hashes[name] = sha256_file(dense_dir / name)
    prefix = hashlib.sha256()
    for cache_index, name in enumerate(cache_hashes):
        prefix.update(cache_index.to_bytes(8, "little", signed=False))
        prefix.update(bytes.fromhex(cache_hashes[name]))
    provenance = DenseSemanticProvenance(
        backend="radseg",
        source_commit="a" * 40,
        radio_commit="b" * 40,
        model_id="radseg:fixture",
        model_sha256=_MODEL_HASH,
        auxiliary_model_sha256="d" * 64,
        vocabulary_sha256=vocabulary_sha256,
        prompt_sha256="e" * 64,
        inference_config_sha256="f" * 64,
        cache_prefix_sha256=prefix.hexdigest(),
        language_model_id="fixture/language-model",
        language_model_revision="1" * 40,
        language_model_sha256="2" * 64,
    )
    dense_manifest = {
        "schema_version": 1,
        "method": "OVIV2-dense-semantic-cache",
        "scene": "room0",
        "frame_count": frame_count,
        "source_frame_ids": source_ids,
        "image_shape": [32, 32],
        "sample_stride": 2,
        "top_k": 2,
        "class_count": 4,
        "vocabulary_sha256": vocabulary_sha256,
        "provenance": asdict(provenance),
        "cache_files_sha256": cache_hashes,
    }
    assert set(dense_manifest) == _DENSE_MANIFEST_KEYS
    (dense_dir / "dense_manifest.json").write_text(
        json.dumps(dense_manifest),
        encoding="utf-8",
    )
    config.update(
        {
            "dense_semantic_mode": "cached_probabilities",
            "dense_cache_dir": str(dense_dir),
            "dense_integration_radius_m": 6.0,
            "dense_minimum_probability": 0.01,
            "dense_minimum_quality": 0.01,
            "dense_entropy_power": 1.0,
            "dense_view_angle_power": 1.0,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return dense_dir


def _rewrite_dense_manifest(dense_dir: Path, mutate) -> None:
    path = dense_dir / "dense_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rehash_dense_manifest(dense_dir: Path) -> None:
    path = dense_dir / "dense_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cache_hashes = {
        name: sha256_file(dense_dir / name)
        for name in manifest["cache_files_sha256"]
    }
    prefix = hashlib.sha256()
    for cache_index, name in enumerate(cache_hashes):
        prefix.update(cache_index.to_bytes(8, "little", signed=False))
        prefix.update(bytes.fromhex(cache_hashes[name]))
    manifest["cache_files_sha256"] = cache_hashes
    manifest["provenance"]["cache_prefix_sha256"] = prefix.hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _replace_dense_cache_hash(
    dense_dir: Path,
    cache_index: int,
    checksum: str,
) -> None:
    path = dense_dir / "dense_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    name = f"frame{cache_index:06d}.npz"
    manifest["cache_files_sha256"][name] = checksum
    prefix = hashlib.sha256()
    for index, cache_name in enumerate(manifest["cache_files_sha256"]):
        prefix.update(index.to_bytes(8, "little", signed=False))
        prefix.update(bytes.fromhex(manifest["cache_files_sha256"][cache_name]))
    manifest["provenance"]["cache_prefix_sha256"] = prefix.hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


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


def test_stage2_config_is_stage1_plus_only_frozen_dense_settings() -> None:
    stage1 = json.loads(_PRECISION_CONFIG.read_text(encoding="utf-8"))
    stage2 = json.loads(_STAGE2_CONFIG.read_text(encoding="utf-8"))

    assert stage2 == {
        **stage1,
        "dense_semantic_mode": "cached_probabilities",
        "dense_cache_dir": (
            "/home/ww/oviovo_dense_cache/room0_radseg_l_sam_s4_k4"
        ),
        "dense_integration_radius_m": 6.0,
        "dense_minimum_probability": 0.01,
        "dense_minimum_quality": 0.01,
        "dense_entropy_power": 1.0,
        "dense_view_angle_power": 1.0,
    }


def test_selected_stage2_config_is_stage1_plus_frozen_winning_dense_settings() -> None:
    stage1 = json.loads(_PRECISION_CONFIG.read_text(encoding="utf-8"))
    selected = json.loads(_SELECTED_STAGE2_CONFIG.read_text(encoding="utf-8"))

    assert selected == {
        **stage1,
        "dense_semantic_mode": "cached_probabilities",
        "dense_cache_dir": (
            "/home/ww/oviovo_dense_cache/room0_radseg_b_sam_s4_k4_200f"
        ),
        "dense_integration_radius_m": 6.0,
        "dense_minimum_probability": 0.01,
        "dense_minimum_quality": 0.01,
        "dense_entropy_power": 16.0,
        "dense_view_angle_power": 0.0,
    }


def test_stage3_config_is_selected_stage2_plus_only_frozen_fusion_settings() -> None:
    stage2 = json.loads(_SELECTED_STAGE2_CONFIG.read_text(encoding="utf-8"))
    stage3 = json.loads(_STAGE3_CONFIG.read_text(encoding="utf-8"))

    assert stage3 == {
        **stage2,
        "fusion_semantic_mode": "uncertainty_linear",
        "fusion_entity_weight_scale": 0.5,
    }


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


def test_disabled_dense_mode_never_reads_dense_cache(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dense_cache_dir"] = str(tmp_path / "hostile-dense-cache")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"

    manifest = run(_args(config_path, output, num_frames=1))

    assert "dense_semantics" not in manifest
    assert (output / "final" / "oviv2_instance_mesh.ply").is_file()
    assert not (output / "final" / "oviv2_owner_mesh.ply").exists()
    timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))
    assert not {
        "dense_sampled_pixel_count",
        "dense_valid_pixel_count",
        "dense_updated_voxel_count",
    } & timing["frames"][0].keys()


def test_cached_dense_mode_requires_cache_dir_before_output(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dense_semantic_mode"] = "cached_probabilities"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="dense_cache_dir"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_cached_dense_mode_requires_completed_manifest_before_output(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dense_dir = tmp_path / "empty-dense-cache"
    dense_dir.mkdir()
    config.update(
        {
            "dense_semantic_mode": "cached_probabilities",
            "dense_cache_dir": str(dense_dir),
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match="dense_manifest"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_oversized_manifest_before_output(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    manifest_path = dense_dir / "dense_manifest.json"
    manifest_path.write_bytes(
        manifest_path.read_bytes() + b" " * (8 * 1024 * 1024)
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="size limit"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_duplicate_manifest_key_before_output(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    manifest_path = dense_dir / "dense_manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        raw.replace(
            '"method": "OVIV2-dense-semantic-cache"',
            '"method": "wrong", "method": "OVIV2-dense-semantic-cache"',
            1,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="duplicate key.*method"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_nonfinite_manifest_constant_before_output(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    manifest_path = dense_dir / "dense_manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        raw.replace('"top_k": 2', '"top_k": NaN', 1),
        encoding="utf-8",
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="invalid JSON constant NaN"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_stage2_audits_exact_validated_manifest_bytes_after_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    manifest_path = dense_dir / "dense_manifest.json"
    validated_bytes = manifest_path.read_bytes()
    original_process = runner_module.Oviv2Runtime.process_frame
    replaced = False

    def replace_manifest_after_preflight(self, frame, observations, dense_semantics=None):
        nonlocal replaced
        if not replaced:
            replacement = dense_dir / "replacement.json"
            replacement.write_bytes(validated_bytes + b"\n")
            replacement.replace(manifest_path)
            replaced = True
        return original_process(self, frame, observations, dense_semantics)

    monkeypatch.setattr(
        runner_module.Oviv2Runtime,
        "process_frame",
        replace_manifest_after_preflight,
    )
    output = tmp_path / "run"

    manifest = run(_args(config_path, output, num_frames=1))

    assert manifest["dense_semantics"]["manifest_sha256"] == hashlib.sha256(
        validated_bytes
    ).hexdigest()
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest[
        "dense_semantics"
    ]["manifest_sha256"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.__setitem__("unexpected", True), "keys"),
        (lambda manifest: manifest.__setitem__("method", "wrong"), "method"),
        (lambda manifest: manifest.__setitem__("source_frame_ids", [1, 10]), "source"),
        (lambda manifest: manifest.__setitem__("image_shape", [31, 32]), "image_shape"),
        (lambda manifest: manifest.__setitem__("class_count", 3), "class_count"),
        (lambda manifest: manifest.__setitem__("sample_stride", 0), "sample_stride"),
        (
            lambda manifest: manifest.__setitem__("vocabulary_sha256", "0" * 64),
            "vocabulary",
        ),
        (
            lambda manifest: manifest["provenance"].__setitem__("unexpected", True),
            "provenance",
        ),
        (
            lambda manifest: manifest["cache_files_sha256"].__setitem__(
                "not-canonical.npz",
                "0" * 64,
            ),
            "canonical|keys",
        ),
    ],
)
def test_dense_preflight_rejects_manifest_mismatch_before_output(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    _rewrite_dense_manifest(dense_dir, mutate)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=message):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_tampered_requested_frame(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    frame = dense_dir / "frame000000.npz"
    frame.write_bytes(frame.read_bytes() + b"tampered")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="checksum"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_tampered_unrequested_frame(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    frame = dense_dir / "frame000001.npz"
    frame.write_bytes(frame.read_bytes() + b"tampered")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="checksum"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_duplicate_member_in_unrequested_frame(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    frame = dense_dir / "frame000001.npz"
    with zipfile.ZipFile(frame, mode="a") as archive:
        duplicate = archive.read("schema_version.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("schema_version.npy", duplicate)
    _rehash_dense_manifest(dense_dir)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="member count|duplicates=.*schema_version"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_bounds_unrequested_archive_before_checksum(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    frame = dense_dir / "frame000001.npz"
    with frame.open("wb") as stream:
        stream.truncate(512 * 1024 * 1024 + 1)
    _replace_dense_cache_hash(dense_dir, 1, "0" * 64)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="archive exceeds.*size limit"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_never_hashes_cache_before_bounded_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    original_sha256 = runner_module._sha256

    def reject_dense_cache_hash(path: Path) -> str:
        if path.parent == dense_dir and path.suffix == ".npz":
            raise AssertionError("dense cache was hashed before bounded load")
        return original_sha256(path)

    monkeypatch.setattr(runner_module, "_sha256", reject_dense_cache_hash)

    run(_args(config_path, tmp_path / "run", num_frames=1))


@pytest.mark.parametrize("replace_on_load", [1, 2])
def test_dense_cache_rejects_symlink_replacement_during_each_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_on_load: int,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    frame = dense_dir / "frame000000.npz"
    target = tmp_path / "frame-copy.npz"
    target.write_bytes(frame.read_bytes())
    original_load = runner_module.load_dense_frame
    frame_load_count = 0

    def replace_before_loader_open(path, *args, **kwargs):
        nonlocal frame_load_count
        cache_path = Path(path)
        if cache_path.name == frame.name:
            frame_load_count += 1
            if frame_load_count == replace_on_load:
                cache_path.unlink()
                cache_path.symlink_to(target)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(runner_module, "load_dense_frame", replace_before_loader_open)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="changed|symlink|regular"):
        run(_args(config_path, output, num_frames=1))

    assert not (output / "run_manifest.json").exists()


def test_dense_preflight_binds_benchmark_vocabulary_source_hash(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    benchmark = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark["vocabulary"]["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(benchmark), encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="source_sha256"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_mismatched_unrequested_source_id(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    _rewrite_dense_manifest(
        dense_dir,
        lambda manifest: manifest.__setitem__("source_frame_ids", [0, 11]),
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="source frame"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_dense_preflight_rejects_symlink_requested_frame(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    frame = dense_dir / "frame000000.npz"
    target = tmp_path / "dense-copy.npz"
    target.write_bytes(frame.read_bytes())
    frame.unlink()
    frame.symlink_to(target)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="symlink|regular"):
        run(_args(config_path, output, num_frames=1))

    assert not output.exists()


def test_runtime_config_builds_exact_dense_semantic_config(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "dense_integration_radius_m": 5.5,
            "dense_minimum_probability": 0.12,
            "dense_minimum_quality": 0.23,
            "dense_entropy_power": 1.5,
            "dense_view_angle_power": 2.5,
        }
    )

    runtime = _runtime_config(config)

    assert runtime.dense_semantics == DenseSemanticConfig(
        voxel_size_m=runtime.tsdf.voxel_size_m,
        integration_radius_m=5.5,
        minimum_probability=0.12,
        minimum_quality=0.23,
        entropy_power=1.5,
        view_angle_power=2.5,
    )


def test_stage2_wires_dense_frames_counters_and_schema3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    observed_dense: list[DenseSemanticFrame | None] = []
    original = runner_module.Oviv2Runtime.process_frame

    def record_dense(self, frame, observations, dense_semantics=None):
        observed_dense.append(dense_semantics)
        return original(self, frame, observations, dense_semantics)

    monkeypatch.setattr(runner_module.Oviv2Runtime, "process_frame", record_dense)
    output = tmp_path / "run"

    run(_args(config_path, output))

    assert [frame.cache_frame_id for frame in observed_dense if frame is not None] == [0, 1]
    assert [frame.source_frame_id for frame in observed_dense if frame is not None] == [0, 10]
    timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))
    assert all(
        type(record[name]) is int and record[name] >= 0
        for record in timing["frames"]
        for name in (
            "dense_sampled_pixel_count",
            "dense_valid_pixel_count",
            "dense_updated_voxel_count",
        )
    )
    assert all(record["dense_sampled_pixel_count"] > 0 for record in timing["frames"])
    final_snapshot = VoxelMapSnapshot.load(
        output / "final" / "oviv2_voxel_snapshot.npz"
    )
    checkpoint = VoxelMapSnapshot.load(
        output / "checkpoints" / "latest_voxel_snapshot.npz"
    )
    assert final_snapshot.metadata.schema_version == 3
    assert checkpoint.metadata.schema_version == 3
    assert final_snapshot.metadata.dense_semantic_provenance is not None
    assert final_snapshot.metadata.dense_semantic_provenance.model_sha256 == _MODEL_HASH


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("cache_frame_id", 1), ("source_frame_id", 1)],
)
def test_stage2_rejects_dense_frame_internal_id_mismatch(
    tmp_path: Path,
    field: str,
    invalid: int,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    path = dense_dir / "frame000000.npz"
    original = load_dense_frame(path)
    values = {
        "cache_frame_id": original.cache_frame_id,
        "source_frame_id": original.source_frame_id,
    }
    values[field] = invalid
    write_dense_frame(
        path,
        DenseSemanticFrame(
            cache_frame_id=values["cache_frame_id"],
            source_frame_id=values["source_frame_id"],
            image_shape=original.image_shape,
            sample_stride=original.sample_stride,
            class_count=original.class_count,
            class_ids=original.class_ids,
            probabilities=original.probabilities,
            entropy=original.entropy,
            margin=original.margin,
        ),
    )
    _rehash_dense_manifest(dense_dir)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match=field):
        run(_args(config_path, output, num_frames=1))


def test_stage2_skip_evaluation_writes_dual_mesh_and_auditable_manifest(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    output = tmp_path / "run"

    manifest = run(_args(config_path, output))

    assert (output / "final" / "oviv2_owner_mesh.ply").is_file()
    assert (output / "final" / "oviv2_dense_mesh.ply").is_file()
    assert not (output / "final" / "oviv2_instance_mesh.ply").exists()
    assert not (output / "evaluation").exists()
    assert not (output / "evaluation_owner").exists()
    assert not (output / "evaluation_dense").exists()
    dense_audit = manifest["dense_semantics"]
    assert dense_audit["mode"] == "cached_probabilities"
    assert dense_audit["cache_dir"] == str(dense_dir)
    assert dense_audit["cache_prefix_sha256"] == dense_audit["provenance"][
        "cache_prefix_sha256"
    ]
    assert dense_audit["provenance"]["model_sha256"] == _MODEL_HASH
    assert dense_audit["config"] == asdict(_runtime_config(
        json.loads(config_path.read_text(encoding="utf-8"))
    ).dense_semantics)
    timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))
    assert dense_audit["counters"] == {
        name: sum(record[name] for record in timing["frames"])
        for name in (
            "dense_sampled_pixel_count",
            "dense_valid_pixel_count",
            "dense_updated_voxel_count",
        )
    }
    assert manifest["model_weights"]["dense_model_sha256"] == _MODEL_HASH
    assert manifest["model_weights"]["dense_language_model_sha256"] == "2" * 64
    assert "final/oviv2_owner_mesh.ply" in manifest["artifact_checksums"]
    assert "final/oviv2_dense_mesh.ply" in manifest["artifact_checksums"]


def test_stage3_skip_evaluation_writes_fused_mesh_and_policy_manifest(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "fusion_semantic_mode": "uncertainty_linear",
            "fusion_entity_weight_scale": 0.5,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"

    manifest = run(_args(config_path, output))

    assert (output / "final" / "oviv2_owner_mesh.ply").is_file()
    assert (output / "final" / "oviv2_dense_mesh.ply").is_file()
    assert (output / "final" / "oviv2_fused_mesh.ply").is_file()
    assert manifest["semantic_fusion"] == {
        "entity_weight_scale": 0.5,
        "mode": "uncertainty_linear",
    }
    assert "final/oviv2_fused_mesh.ply" in manifest["artifact_checksums"]


def test_stage2_partial_run_separates_producer_and_consumed_prefixes(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)
    dense_dir = _write_dense_cache(config_path)
    producer_manifest = json.loads(
        (dense_dir / "dense_manifest.json").read_text(encoding="utf-8")
    )
    first_checksum = producer_manifest["cache_files_sha256"]["frame000000.npz"]
    consumed_digest = hashlib.sha256()
    consumed_digest.update((0).to_bytes(8, "little", signed=False))
    consumed_digest.update(bytes.fromhex(first_checksum))
    consumed_prefix = consumed_digest.hexdigest()
    producer_prefix = producer_manifest["provenance"]["cache_prefix_sha256"]
    output = tmp_path / "run"

    manifest = run(_args(config_path, output, num_frames=1))
    snapshot = VoxelMapSnapshot.load(output / "final" / "oviv2_voxel_snapshot.npz")
    audit = manifest["dense_semantics"]

    assert consumed_prefix != producer_prefix
    assert snapshot.metadata.dense_semantic_provenance is not None
    assert snapshot.metadata.dense_semantic_provenance.cache_prefix_sha256 == consumed_prefix
    assert audit["cache_prefix_sha256"] == consumed_prefix
    assert audit["provenance"]["cache_prefix_sha256"] == consumed_prefix
    assert audit["producer_cache_prefix_sha256"] == producer_prefix
    assert audit["producer_provenance"]["cache_prefix_sha256"] == producer_prefix
    assert list(audit["cache_files_sha256"]) == ["frame000000.npz"]
    assert list(audit["producer_cache_files_sha256"]) == [
        "frame000000.npz",
        "frame000001.npz",
    ]


def test_stage2_evaluates_owner_and_dense_heads_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["gt_mesh"]).write_bytes(b"runner must not parse GT mesh content")
    Path(config["gt_info"]).write_bytes(b"runner must not parse GT info content")
    calls: list[tuple[list[str], dict[str, object]]] = []
    original_run = runner_module.subprocess.run
    evaluator = runner_module.REPO_ROOT / "scripts/evaluation/evaluate_oviv2_replica.py"

    def fake_subprocess_run(command, *args, **kwargs):
        normalized = [str(value) for value in command]
        if len(normalized) > 1 and normalized[1] == str(evaluator):
            calls.append((normalized, dict(kwargs)))
            output_arg = Path(normalized[normalized.index("--output") + 1])
            semantic_head = normalized[normalized.index("--semantic-head") + 1]
            output_arg.mkdir(parents=True)
            (output_arg / "metrics.json").write_text(
                json.dumps({"protocol": {"semantic_head": semantic_head}}),
                encoding="utf-8",
            )
            return runner_module.subprocess.CompletedProcess(normalized, 0)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "run", fake_subprocess_run)
    output = tmp_path / "run"

    manifest = run(_args_with_evaluation(config_path, output))

    assert not hasattr(runner_module, "evaluate_snapshot")
    assert len(calls) == 2
    for (command, kwargs), semantic_head, evaluation_output in zip(
        calls,
        ("owner_authoritative", "dense_only"),
        (output / "evaluation_owner", output / "evaluation_dense"),
    ):
        assert command == [
            runner_module.sys.executable,
            str(evaluator),
            "--snapshot",
            str(output / "final" / "oviv2_voxel_snapshot.npz"),
            "--entity-info",
            str(output / "final" / "oviv2_entities.jsonl"),
            "--gt-mesh",
            config["gt_mesh"],
            "--gt-info",
            config["gt_info"],
            "--manifest",
            config["manifest"],
            "--scene",
            "room0",
            "--output",
            str(evaluation_output),
            "--min-instance-vertices",
            "100",
            "--semantic-head",
            semantic_head,
        ]
        assert kwargs == {
            "cwd": runner_module.REPO_ROOT,
            "check": True,
            "stdout": runner_module.subprocess.PIPE,
            "text": True,
        }
        assert json.loads(
            (evaluation_output / "metrics.json").read_text(encoding="utf-8")
        ) == {
            "protocol": {"semantic_head": semantic_head}
        }
    assert (output / "evaluation_owner" / "metrics.json").is_file()
    assert (output / "evaluation_dense" / "metrics.json").is_file()
    assert not (output / "evaluation").exists()
    assert "evaluation_owner/metrics.json" in manifest["artifact_checksums"]
    assert "evaluation_dense/metrics.json" in manifest["artifact_checksums"]


def test_stage3_adds_fused_evaluator_with_exact_frozen_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "fusion_semantic_mode": "uncertainty_linear",
            "fusion_entity_weight_scale": 0.5,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    Path(config["gt_mesh"]).write_bytes(b"runner must not parse GT mesh content")
    Path(config["gt_info"]).write_bytes(b"runner must not parse GT info content")
    calls: list[list[str]] = []
    original_run = runner_module.subprocess.run
    evaluator = runner_module.REPO_ROOT / "scripts/evaluation/evaluate_oviv2_replica.py"

    def fake_subprocess_run(command, *args, **kwargs):
        normalized = [str(value) for value in command]
        if len(normalized) > 1 and normalized[1] == str(evaluator):
            calls.append(normalized)
            output_arg = Path(normalized[normalized.index("--output") + 1])
            semantic_head = normalized[normalized.index("--semantic-head") + 1]
            output_arg.mkdir(parents=True)
            (output_arg / "metrics.json").write_text(
                json.dumps({"protocol": {"semantic_head": semantic_head}}),
                encoding="utf-8",
            )
            return runner_module.subprocess.CompletedProcess(normalized, 0)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "run", fake_subprocess_run)
    output = tmp_path / "run"

    manifest = run(_args_with_evaluation(config_path, output))

    assert [call[call.index("--semantic-head") + 1] for call in calls] == [
        "owner_authoritative",
        "dense_only",
        "fused_uncertainty",
    ]
    fused = calls[-1]
    assert fused[fused.index("--fusion-entity-weight-scale") + 1] == "0.5"
    assert (output / "evaluation_fused" / "metrics.json").is_file()
    assert "evaluation_fused/metrics.json" in manifest["artifact_checksums"]


def test_stage2_cli_stdout_remains_one_json_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["gt_mesh"]).write_bytes(b"runner must not parse GT mesh content")
    Path(config["gt_info"]).write_bytes(b"runner must not parse GT info content")
    original_run = runner_module.subprocess.run
    evaluator = runner_module.REPO_ROOT / "scripts/evaluation/evaluate_oviv2_replica.py"

    def noisy_evaluator(command, *args, **kwargs):
        normalized = [str(value) for value in command]
        if len(normalized) > 1 and normalized[1] == str(evaluator):
            if kwargs.get("stdout") is None:
                print(json.dumps({"evaluator": "leaked"}))
            output_arg = Path(normalized[normalized.index("--output") + 1])
            output_arg.mkdir(parents=True)
            (output_arg / "metrics.json").write_text("{}", encoding="utf-8")
            return runner_module.subprocess.CompletedProcess(normalized, 0)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "run", noisy_evaluator)
    output = tmp_path / "run"

    exit_code = runner_module.main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--num-frames",
            "1",
        ]
    )

    stdout_lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert len(stdout_lines) == 1
    payload = json.loads(stdout_lines[0])
    assert payload["frames"] == 1
    assert payload["output"] == str(output)
    assert payload["revision"] == 1
    assert type(payload["entities"]) is int and payload["entities"] >= 0


def test_stage2_propagates_evaluator_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["gt_mesh"]).write_bytes(b"runner must not parse GT mesh content")
    Path(config["gt_info"]).write_bytes(b"runner must not parse GT info content")
    original_run = runner_module.subprocess.run
    evaluator = runner_module.REPO_ROOT / "scripts/evaluation/evaluate_oviv2_replica.py"

    def fail_evaluator(command, *args, **kwargs):
        normalized = [str(value) for value in command]
        if len(normalized) > 1 and normalized[1] == str(evaluator):
            raise runner_module.subprocess.CalledProcessError(7, normalized)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "run", fail_evaluator)
    output = tmp_path / "run"

    with pytest.raises(runner_module.subprocess.CalledProcessError) as error:
        run(_args_with_evaluation(config_path, output))

    assert error.value.returncode == 7
    assert not (output / "timing.json").exists()
    assert not (output / "run_manifest.json").exists()


def test_stage2_resume_is_byte_and_mtime_idempotent(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    output = tmp_path / "run"
    original = run(_args(config_path, output))
    before = {
        path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }

    resumed = run(_args(config_path, output, "--resume"))

    after = {
        path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert resumed == original
    assert after == before


def test_stage2_two_frame_final_replay_is_deterministic(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    _write_dense_cache(config_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = run(_args(config_path, first_output))
    second = run(_args(config_path, second_output))

    assert first["dense_semantics"] == second["dense_semantics"]
    semantic_files = (
        "oviv2_entities.jsonl",
        "oviv2_owner_mesh.ply",
        "oviv2_dense_mesh.ply",
        "oviv2_voxel_snapshot.npz/metadata.json",
        "oviv2_voxel_snapshot.npz/evidence.npz",
        "oviv2_voxel_snapshot.npz/ownership.npz",
        "oviv2_voxel_snapshot.npz/entities.jsonl",
    )
    for relative_path in semantic_files:
        assert (first_output / "final" / relative_path).read_bytes() == (
            second_output / "final" / relative_path
        ).read_bytes()


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
