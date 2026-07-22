from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.evaluation.evaluate_oviv2_tesse_occlusion as occlusion_evaluator
from scripts.evaluation.derive_tesse_cd_occlusion_v1 import (
    _preregistered_parameters,
    write_occlusion_package,
)
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    REPO_ROOT,
    _load_checkpoint_snapshots,
    _load_target_package,
    _source_path,
    build_evaluation_checkpoint_plan,
    evaluate_occlusion_package,
    main,
)
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.compact_checkpoint import (
    COMPACT_OWNERSHIP_FORMAT,
    CompactOwnershipCheckpoint,
    CompactOwnershipMetadata,
)
from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_sources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for role in ("source_manifest", "schedule", "rgbd_lock", "camera"):
        path = root / f"{role}.bin"
        path.write_bytes(role.encode("ascii"))
        paths[role] = path
    for scene in ("apartment", "office"):
        for role in (
            "changes",
            "dsg_with_mesh",
            "export_manifest",
            "timestamps",
            "trajectory",
        ):
            path = root / f"{scene}.{role}.bin"
            if role == "timestamps":
                path.write_text(
                    "frame_index,sensor_timestamp_ns,relative_timestamp_ns\n"
                    "0,0,0\n"
                    "1,1,1\n",
                    encoding="utf-8",
                )
            else:
                path.write_bytes(f"{scene}.{role}".encode("ascii"))
            paths[f"{scene}.{role}"] = path
        for frame_index in range(2):
            role = f"{scene}.depth.{frame_index:06d}"
            path = root / f"{role}.png"
            path.write_bytes(role.encode("ascii"))
            paths[role] = path
    return paths


def _episode(scene: str, base: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    episode_id = f"{scene}_occlusion_0000"
    keys = np.asarray(
        [(base + index, 0, 20) for index in range(2)], dtype=np.int64
    )
    anchor_name = f"{episode_id}.anchor"
    checkpoint_name = f"{episode_id}.frame_000001.occluded"
    return (
        {anchor_name: keys, checkpoint_name: keys.copy()},
        {
            "episode_id": episode_id,
            "scene": scene,
            "object_id": f"gt-{scene}",
            "object_name": "chair",
            "semantic_label": 5,
            "lifecycle": {
                "index": 0,
                "first_timestamp_ns": 0,
                "last_timestamp_ns": 10,
            },
            "anchor": {
                "frame_index": 0,
                "relative_timestamp_ns": 0,
                "array": anchor_name,
                "voxel_count": 2,
            },
            "start_frame_index": 1,
            "end_frame_index": 1,
            "checkpoints": [
                {
                    "frame_index": 1,
                    "relative_timestamp_ns": 1,
                    "array": checkpoint_name,
                    "occluded_voxel_count": 2,
                    "occlusion_fraction": 1.0,
                }
            ],
            "occlusion_fraction": 1.0,
        },
    )


def _write_targets(root: Path) -> tuple[Path, dict[str, Path]]:
    source_paths = _target_sources(root / "sources")
    apartment_arrays, apartment = _episode("apartment", 0)
    office_arrays, office = _episode("office", 10)
    arrays = {**apartment_arrays, **office_arrays}
    episodes = [apartment, office]
    metadata = {
        "prediction_inputs_used": False,
        "parameters": _preregistered_parameters(),
        "scene_frame_indices": {"apartment": [0, 1], "office": [0, 1]},
        "scenes": {
            scene: {"episode_count": 1, "headline_episode_count": 1}
            for scene in ("apartment", "office")
        },
        "stress_layers": {
            "all": {
                "episode_count": 2,
                "episode_ids": [item["episode_id"] for item in episodes],
            },
            **{
                f"{threshold:.2f}": {
                    "episode_count": 2,
                    "episode_ids": [item["episode_id"] for item in episodes],
                    "minimum_occlusion_fraction": threshold,
                }
                for threshold in (0.50, 0.75, 0.90)
            },
        },
        "episodes": episodes,
    }
    target_dir = root / "targets"
    write_occlusion_package(
        target_dir,
        arrays=arrays,
        metadata=metadata,
        source_paths=source_paths,
        status="FIXTURE",
    )
    return target_dir, source_paths


def _write_snapshot(
    path: Path,
    *,
    scene: str,
    frame_index: int,
    owners: dict[tuple[int, int, int], int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = SparseTsdfVolume()
    evidence = SparseEvidenceStore()
    ownership = ReversibleOwnershipStore()
    for key, entity_id in sorted(owners.items()):
        evidence.update_entity(key, entity_id, 1.0, 0.0, 0.0, frame_index + 1)
        ownership.assign(key, entity_id, 1.0, frame_index + 1)
    VoxelMapSnapshot.commit_new(
        path,
        VoxelSnapshotMetadata(
            scene_id=scene,
            frame_id=frame_index,
            timestamp=frame_index / 1_000_000_000,
            revision=frame_index + 1,
            voxel_size_m=geometry.config.voxel_size_m,
            block_resolution=geometry.config.block_resolution,
        ),
        geometry,
        evidence,
        ownership,
    )


def _compact_provenance() -> DenseSemanticProvenance:
    return DenseSemanticProvenance(
        backend="radseg",
        source_commit="1" * 40,
        radio_commit="2" * 40,
        model_id="radseg:test",
        model_sha256="3" * 64,
        auxiliary_model_sha256="4" * 64,
        vocabulary_sha256="5" * 64,
        prompt_sha256="6" * 64,
        inference_config_sha256="7" * 64,
        cache_prefix_sha256="8" * 64,
        language_model_id="clip:test",
        language_model_revision="9" * 40,
        language_model_sha256="a" * 64,
    )


def _convert_index_to_mixed(
    checkpoint_index: Path,
    *,
    compact_keys: set[tuple[str, int]],
) -> Path:
    payload = json.loads(checkpoint_index.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    for record in payload["snapshots"]:
        key = (record["scene"], record["frame_index"])
        if key not in compact_keys:
            record["format"] = "oviv2_voxel_map_snapshot"
            continue
        full = VoxelMapSnapshot.load(checkpoint_index.parent / record["path"])
        compact_path = (
            checkpoint_index.parent
            / "compact"
            / key[0]
            / f"frame_{key[1]:06d}"
        )
        compact_path.parent.mkdir(parents=True, exist_ok=True)
        CompactOwnershipCheckpoint.commit_new(
            compact_path,
            CompactOwnershipMetadata(
                scene_id=full.metadata.scene_id,
                frame_id=full.metadata.frame_id,
                timestamp=full.metadata.timestamp,
                revision=full.metadata.revision,
                voxel_size_m=full.metadata.voxel_size_m,
                block_resolution=full.metadata.block_resolution,
                dense_semantic_provenance=_compact_provenance(),
            ),
            full.ownership,
        )
        record["format"] = COMPACT_OWNERSHIP_FORMAT
        record["path"] = compact_path.relative_to(checkpoint_index.parent).as_posix()
        record["checksums_sha256"] = _sha256(compact_path / "checksums.json")
    checkpoint_index.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return checkpoint_index


def _write_checkpoint_index(
    root: Path,
    targets: Path,
    *,
    missing_observation_policy: str = "signed_depth",
) -> Path:
    snapshot_root = root / "snapshots"
    records = []
    for scene, base in (("apartment", 0), ("office", 10)):
        keys = [(base + index, 0, 20) for index in range(2)]
        for frame_index in (0, 1):
            path = snapshot_root / scene / f"frame_{frame_index:06d}"
            _write_snapshot(
                path,
                scene=scene,
                frame_index=frame_index,
                owners={key: 7 for key in keys},
            )
            records.append(
                {
                    "scene": scene,
                    "frame_index": frame_index,
                    "timestamp_ns": frame_index,
                    "relative_timestamp_ns": frame_index,
                    "consumed_through_frame": frame_index,
                    "consumed_through_frame_exclusive": frame_index + 1,
                    "path": str(path.relative_to(root)),
                    "checksums_sha256": _sha256(path / "checksums.json"),
                }
            )
    manifest = targets / "manifest.json"
    target_payload = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(targets / "targets.npz", allow_pickle=False) as target_arrays:
        checkpoint_plan = build_evaluation_checkpoint_plan(
            arrays={
                name: np.array(target_arrays[name], copy=True)
                for name in target_arrays.files
            },
            metadata=target_payload["metadata"],
        )
    normalized_config = {
        "config_id": "oviv2-tessecd-test-v1",
        "missing_observation_policy": missing_observation_policy,
    }
    config_path = root / "normalized_run_config.json"
    config_path.write_text(
        json.dumps(
            normalized_config,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_occlusion_checkpoints_v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "run_config": {
            "path": config_path.name,
            "sha256": _sha256(config_path),
            "byte_count": config_path.stat().st_size,
        },
        "target_manifest": {
            "sha256": _sha256(manifest),
            "byte_count": manifest.stat().st_size,
        },
        "evaluation_checkpoint_frames_sha256": checkpoint_plan[
            "evaluation_checkpoint_frames_sha256"
        ],
        "snapshots": records,
    }
    path = root / "checkpoint_index.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _split_schema2_indexes_by_scene(
    checkpoint_index: Path,
) -> tuple[Path, Path]:
    payload = json.loads(checkpoint_index.read_text(encoding="utf-8"))
    indexes: list[Path] = []
    for scene in ("apartment", "office"):
        config = {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "missing_observation_policy": "signed_depth",
            "occlusion_target_manifest_sha256": payload["target_manifest"]["sha256"],
            "evaluation_checkpoint_frames": [0, 1],
            "evaluation_checkpoint_frames_sha256": payload[
                "evaluation_checkpoint_frames_sha256"
            ],
        }
        algorithm_hash = occlusion_evaluator.canonical_algorithm_hash(config)
        config["algorithm_hash"] = algorithm_hash
        config_path = checkpoint_index.parent / f"normalized_run_config.{scene}.json"
        config_path.write_text(
            json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        scene_payload = {
            **payload,
            "schema_version": 2,
            "scene": scene,
            "algorithm_hash": algorithm_hash,
            "run_config": {
                "path": config_path.name,
                "sha256": _sha256(config_path),
                "byte_count": config_path.stat().st_size,
            },
            "snapshots": [
                {**record, "format": "oviv2_voxel_map_snapshot"}
                for record in payload["snapshots"]
                if record["scene"] == scene
            ],
        }
        path = checkpoint_index.parent / f"checkpoint_index.{scene}.json"
        path.write_text(
            json.dumps(scene_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        indexes.append(path)
    return indexes[0], indexes[1]


def test_builds_canonical_target_bound_evaluation_checkpoint_plan(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    manifest = json.loads((targets / "manifest.json").read_text(encoding="utf-8"))
    with np.load(targets / "targets.npz", allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}

    plan = build_evaluation_checkpoint_plan(
        arrays=arrays,
        metadata=manifest["metadata"],
    )

    assert plan["evaluation_checkpoint_frames"] == {
        "apartment": [0, 1],
        "office": [0, 1],
    }
    binding = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_occlusion_v1_checkpoint_frames",
        "evaluation_checkpoint_frames": plan["evaluation_checkpoint_frames"],
    }
    expected = hashlib.sha256(
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert plan["evaluation_checkpoint_frames_sha256"] == expected


def test_checkpoint_plan_consumes_already_validated_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    manifest = json.loads((targets / "manifest.json").read_text(encoding="utf-8"))
    with np.load(targets / "targets.npz", allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}

    def reject_revalidation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("checkpoint plan must not revalidate the frozen target")

    monkeypatch.setattr(
        "scripts.evaluation.evaluate_oviv2_tesse_occlusion.validate_generated_target",
        reject_revalidation,
    )
    plan = build_evaluation_checkpoint_plan(
        arrays=arrays,
        metadata=manifest["metadata"],
    )

    assert plan["evaluation_checkpoint_frames"] == {
        "apartment": [0, 1],
        "office": [0, 1],
    }


def test_checkpoint_snapshots_are_loaded_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    arrays, metadata, manifest_witness, _, source_frame_times = _load_target_package(
        targets,
        dataset_root=targets.parent / "sources",
    )
    plan = build_evaluation_checkpoint_plan(arrays=arrays, metadata=metadata)
    original_load = VoxelMapSnapshot.load
    loaded: list[Path] = []

    def counted_load(path: str | Path) -> VoxelMapSnapshot:
        loaded.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(VoxelMapSnapshot, "load", classmethod(lambda _, path: counted_load(path)))
    snapshots, _, _, _, _ = _load_checkpoint_snapshots(
        checkpoints,
        metadata=metadata,
        target_manifest_witness=manifest_witness,
        checkpoint_plan=plan,
        source_frame_times=source_frame_times,
    )

    assert loaded == []
    assert snapshots[("apartment", 0)].metadata.frame_id == 0
    assert len(loaded) == 1


def test_rejects_internally_consistent_snapshot_replacement_after_index_check(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    arrays, metadata, manifest_witness, _, source_frame_times = _load_target_package(
        targets,
        dataset_root=targets.parent / "sources",
    )
    plan = build_evaluation_checkpoint_plan(arrays=arrays, metadata=metadata)
    snapshots, _, _, _, _ = _load_checkpoint_snapshots(
        checkpoints,
        metadata=metadata,
        target_manifest_witness=manifest_witness,
        checkpoint_plan=plan,
        source_frame_times=source_frame_times,
    )
    index = json.loads(checkpoints.read_text(encoding="utf-8"))
    record = next(
        item
        for item in index["snapshots"]
        if item["scene"] == "apartment" and item["frame_index"] == 0
    )
    original = checkpoints.parent / record["path"]
    original.rename(original.with_name(f"{original.name}.original"))
    _write_snapshot(
        original,
        scene="apartment",
        frame_index=0,
        owners={(0, 0, 20): 99, (1, 0, 20): 99},
    )

    with pytest.raises(ValueError, match="snapshot.*changed|identity"):
        snapshots[("apartment", 0)]
    with pytest.raises(ValueError, match="snapshot.*changed|identity"):
        snapshots.revalidate_all()  # type: ignore[attr-defined]


def test_rejects_checksum_manifest_change_during_snapshot_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    index = json.loads(checkpoints.read_text(encoding="utf-8"))
    snapshot = checkpoints.parent / index["snapshots"][0]["path"]
    checksums_path = snapshot / "checksums.json"
    original_read = occlusion_evaluator._read_bound_file
    changed = False

    def change_after_read(*args: object, **kwargs: object):
        nonlocal changed
        content, witness = original_read(*args, **kwargs)
        if Path(args[0]) == checksums_path and not changed:
            checksums_path.write_bytes(content.rstrip() + b"\n\n")
            changed = True
        return content, witness

    monkeypatch.setattr(
        occlusion_evaluator,
        "_read_bound_file",
        change_after_read,
    )
    with pytest.raises(ValueError, match="snapshot.*changed|identity"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )


def test_formal_source_paths_use_explicit_roots_and_exact_role_allowlist(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    assert _source_path(
        "configs/evaluation/manifests/tesse_cd.json",
        role="source_manifest",
        dataset_root=dataset_root,
        formal=True,
    ) == REPO_ROOT / "configs/evaluation/manifests/tesse_cd.json"
    assert _source_path(
        "ground_truth/apartment/gt_changes.csv",
        role="apartment.changes",
        dataset_root=dataset_root,
        formal=True,
    ) == dataset_root / "ground_truth/apartment/gt_changes.csv"
    with pytest.raises(ValueError, match="must be relative"):
        _source_path(
            str((dataset_root / "ground_truth/apartment/gt_changes.csv").resolve()),
            role="apartment.changes",
            dataset_root=dataset_root,
            formal=True,
        )

    targets, _ = _write_targets(tmp_path / "fixture")
    manifest_path = targets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"]["prediction"] = manifest["sources"]["camera"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact role allowlist"):
        _load_target_package(
            targets,
            dataset_root=targets.parent / "sources",
        )


def test_package_evaluation_is_byte_identical_and_passes_headline_gate(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    result = evaluate_occlusion_package(
        target_dir=targets,
        checkpoint_index=checkpoints,
        output_path=first,
        dataset_root=targets.parent / "sources",
    )
    assert main(
        [
            "--targets",
            str(targets),
            "--checkpoints",
            str(checkpoints),
            "--dataset-root",
            str(targets.parent / "sources"),
            "--output",
            str(second),
        ]
    ) == 0

    assert result["headline_gate"]["passed"] is True
    assert result["missing_observation_policy"] == "signed_depth"
    assert result["checkpoint_index"] == {
        "sha256": _sha256(checkpoints),
        "byte_count": checkpoints.stat().st_size,
    }
    config_path = checkpoints.parent / "normalized_run_config.json"
    assert result["run_config"] == {
        "sha256": _sha256(config_path),
        "byte_count": config_path.stat().st_size,
    }
    assert first.read_bytes() == second.read_bytes()
    assert b"NaN" not in first.read_bytes()


def test_two_per_scene_indexes_are_consumed_with_independent_run_configs(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    indexes = _split_schema2_indexes_by_scene(
        _write_checkpoint_index(tmp_path / "fixture", targets)
    )

    result = evaluate_occlusion_package(
        target_dir=targets,
        checkpoint_index=indexes,
        dataset_root=targets.parent / "sources",
    )

    assert result["checkpoint_count"] == 4
    assert result["headline_gate"]["passed"] is True
    assert result["checkpoint_index"] == [
        {"sha256": _sha256(path), "byte_count": path.stat().st_size}
        for path in indexes
    ]
    assert len(result["run_config"]) == 2


def test_cli_accepts_two_per_scene_checkpoint_indexes(tmp_path: Path) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    apartment, office = _split_schema2_indexes_by_scene(
        _write_checkpoint_index(tmp_path / "fixture", targets)
    )

    assert main(
        [
            "--targets",
            str(targets),
            "--checkpoints",
            str(apartment),
            "--checkpoints",
            str(office),
            "--dataset-root",
            str(targets.parent / "sources"),
            "--output",
            str(tmp_path / "result.json"),
        ]
    ) == 0


def test_two_index_bundle_rejects_missing_scene_and_algorithm_drift(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    apartment, office = _split_schema2_indexes_by_scene(
        _write_checkpoint_index(tmp_path / "fixture", targets)
    )

    with pytest.raises(ValueError, match="apartment and office|missing checkpoint"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=[apartment],
            dataset_root=targets.parent / "sources",
        )

    office_payload = json.loads(office.read_text(encoding="utf-8"))
    office_payload["algorithm_hash"] = "b" * 64
    office.write_text(json.dumps(office_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="algorithm hash"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=[apartment, office],
            dataset_root=targets.parent / "sources",
        )


def test_schema2_index_rejects_cross_scene_record_and_config_relabel(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    apartment, office = _split_schema2_indexes_by_scene(
        _write_checkpoint_index(tmp_path / "fixture", targets)
    )
    apartment_payload = json.loads(apartment.read_text(encoding="utf-8"))
    apartment_payload["snapshots"][0]["scene"] = "office"
    apartment.write_text(json.dumps(apartment_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scene membership|duplicate checkpoint"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=[apartment, office],
            dataset_root=targets.parent / "sources",
        )

    apartment, office = _split_schema2_indexes_by_scene(
        _write_checkpoint_index(tmp_path / "second", targets)
    )
    payload = json.loads(apartment.read_text(encoding="utf-8"))
    config_path = apartment.parent / payload["run_config"]["path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["scene"] = "office"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    payload["run_config"]["sha256"] = _sha256(config_path)
    payload["run_config"]["byte_count"] = config_path.stat().st_size
    apartment.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run config.*scene"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=[apartment, office],
            dataset_root=targets.parent / "sources",
        )


def test_schema2_rejects_synchronized_declared_algorithm_hash_tamper(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    apartment, office = _split_schema2_indexes_by_scene(
        _write_checkpoint_index(tmp_path / "fixture", targets)
    )
    payload = json.loads(apartment.read_text(encoding="utf-8"))
    config_path = apartment.parent / payload["run_config"]["path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload["algorithm_hash"] = "b" * 64
    config["algorithm_hash"] = "b" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    payload["run_config"]["sha256"] = _sha256(config_path)
    payload["run_config"]["byte_count"] = config_path.stat().st_size
    apartment.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="algorithm hash"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=[apartment, office],
            dataset_root=targets.parent / "sources",
        )


@pytest.mark.parametrize("drift", ["policy", "config_binding"])
def test_formal_schema2_index_rejects_policy_or_config_binding_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    apartment, office = _split_schema2_indexes_by_scene(
        _write_checkpoint_index(tmp_path / "fixture", targets)
    )
    payload = json.loads(apartment.read_text(encoding="utf-8"))
    config_path = apartment.parent / payload["run_config"]["path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if drift == "policy":
        config["missing_observation_policy"] = "missing_as_absence"
    config["algorithm_hash"] = occlusion_evaluator.canonical_algorithm_hash(config)
    config_path.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    payload["algorithm_hash"] = config["algorithm_hash"]
    payload["run_config"] = {
        "path": config_path.name,
        "sha256": _sha256(config_path),
        "byte_count": config_path.stat().st_size,
    }
    root_status = apartment.parent.stat()
    execution = {
        "schema_version": 1,
        "run_slot": "apartment_run1",
        "output_root": str(apartment.parent.resolve()),
        "root_device": root_status.st_dev,
        "root_inode": root_status.st_ino,
    }
    execution["execution_id"] = hashlib.sha256(
        json.dumps(
            execution,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload["frozen_run_identity"] = {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": "apartment",
        "freeze_manifest": {"sha256": "1" * 64, "byte_count": 1},
        "repository": {"commit": "2" * 40, "tree": "3" * 40},
        "config": {
            "sha256": (
                "4" * 64 if drift == "config_binding" else _sha256(config_path)
            ),
            "byte_count": (
                1 if drift == "config_binding" else config_path.stat().st_size
            ),
        },
        "algorithm_hash": config["algorithm_hash"],
        "missing_observation_policy": "signed_depth",
        "input_bindings_sha256": "5" * 64,
    }
    payload["run_execution"] = execution
    apartment.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen run identity"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=[apartment, office],
            dataset_root=targets.parent / "sources",
        )


def test_mixed_full_and_compact_checkpoints_share_one_evaluator_interface(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = tuple(
        _convert_index_to_mixed(
            index,
            compact_keys={("apartment", 1), ("office", 0)},
        )
        for index in _split_schema2_indexes_by_scene(
            _write_checkpoint_index(tmp_path / "fixture", targets)
        )
    )

    result = evaluate_occlusion_package(
        target_dir=targets,
        checkpoint_index=checkpoints,
        dataset_root=targets.parent / "sources",
    )

    assert result["headline_gate"]["passed"] is True
    assert result["checkpoint_count"] == 4


@pytest.mark.parametrize(
    "declared_format, compact",
    [
        (COMPACT_OWNERSHIP_FORMAT, False),
        ("oviv2_voxel_map_snapshot", True),
    ],
)
def test_rejects_checkpoint_format_disguise(
    tmp_path: Path,
    declared_format: str,
    compact: bool,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = tuple(
        _convert_index_to_mixed(
            index,
            compact_keys={("apartment", 0)} if compact else set(),
        )
        for index in _split_schema2_indexes_by_scene(
            _write_checkpoint_index(tmp_path / "fixture", targets)
        )
    )
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    payload["snapshots"][0]["format"] = declared_format
    checkpoints[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="format|inventory|checksum"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )


def test_compact_checkpoint_final_barrier_rejects_replacement(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = tuple(
        _convert_index_to_mixed(
            index,
            compact_keys={("apartment", 0)},
        )
        for index in _split_schema2_indexes_by_scene(
            _write_checkpoint_index(tmp_path / "fixture", targets)
        )
    )
    arrays, metadata, manifest_witness, _, source_frame_times = _load_target_package(
        targets,
        dataset_root=targets.parent / "sources",
    )
    plan = build_evaluation_checkpoint_plan(arrays=arrays, metadata=metadata)
    snapshots, _, _, _, _ = _load_checkpoint_snapshots(
        checkpoints,
        metadata=metadata,
        target_manifest_witness=manifest_witness,
        checkpoint_plan=plan,
        source_frame_times=source_frame_times,
    )
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    record = next(
        item
        for item in payload["snapshots"]
        if item["scene"] == "apartment" and item["frame_index"] == 0
    )
    original = checkpoints[0].parent / record["path"]
    replacement = original.with_name("replacement")
    loaded = CompactOwnershipCheckpoint.load(original)
    CompactOwnershipCheckpoint.commit_new(
        replacement,
        loaded.metadata,
        loaded.ownership,
    )
    original.rename(original.with_name("original"))
    replacement.rename(original)

    with pytest.raises(ValueError, match="snapshot.*changed|identity"):
        snapshots.revalidate_all()  # type: ignore[attr-defined]


def test_headline_rejects_ablation_policy_from_bound_run_config(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(
        tmp_path / "fixture",
        targets,
        missing_observation_policy="missing_as_absence",
    )

    result = evaluate_occlusion_package(
        target_dir=targets,
        checkpoint_index=checkpoints,
        dataset_root=targets.parent / "sources",
    )

    assert result["missing_observation_policy"] == "missing_as_absence"
    assert result["headline_gate"]["missing_observation_policy"] == (
        "missing_as_absence"
    )
    assert result["headline_gate"]["passed"] is False


def test_rejects_run_config_relabel(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    config_path = checkpoints.parent / "normalized_run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["missing_observation_policy"] = "missing_as_absence"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="run config.*hash|binding"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )


def test_rejects_strict_json_timestamp_and_boolean_causal_values(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    index = json.loads(checkpoints.read_text(encoding="utf-8"))
    snapshot = checkpoints.parent / index["snapshots"][0]["path"]
    checksums_path = snapshot / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    first_name = sorted(checksums)[0]
    serialized = json.dumps(checksums, sort_keys=True)[1:]
    checksums_path.write_text(
        "{"
        + json.dumps(first_name)
        + ":"
        + json.dumps(checksums[first_name])
        + ","
        + serialized,
        encoding="utf-8",
    )
    index["snapshots"][0]["checksums_sha256"] = _sha256(checksums_path)
    checkpoints.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )

    checkpoints = _write_checkpoint_index(tmp_path / "nonfinite", targets)
    index = json.loads(checkpoints.read_text(encoding="utf-8"))
    config_path = checkpoints.parent / index["run_config"]["path"]
    config_path.write_text(
        '{"missing_observation_policy":"signed_depth","threshold":NaN}',
        encoding="utf-8",
    )
    index["run_config"]["sha256"] = _sha256(config_path)
    index["run_config"]["byte_count"] = config_path.stat().st_size
    checkpoints.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )

    checkpoints = _write_checkpoint_index(tmp_path / "timestamp", targets)
    payload = json.loads(checkpoints.read_text(encoding="utf-8"))
    payload["snapshots"][0]["timestamp_ns"] = 99
    checkpoints.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )

    checkpoints = _write_checkpoint_index(tmp_path / "boolean", targets)
    payload = json.loads(checkpoints.read_text(encoding="utf-8"))
    payload["snapshots"][0]["consumed_through_frame"] = True
    checkpoints.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="causal boundary|integer"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )


def test_rejects_boolean_target_and_snapshot_schema_versions(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "target-schema")
    manifest_path = targets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="target manifest identity"):
        _load_target_package(
            targets,
            dataset_root=targets.parent / "sources",
        )

    targets, _ = _write_targets(tmp_path / "snapshot-schema")
    checkpoints = _write_checkpoint_index(tmp_path / "snapshot-schema", targets)
    index = json.loads(checkpoints.read_text(encoding="utf-8"))
    snapshot = checkpoints.parent / index["snapshots"][0]["path"]
    metadata_path = snapshot / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    checksums_path = snapshot / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["metadata.json"] = _sha256(metadata_path)
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
    index["snapshots"][0]["checksums_sha256"] = _sha256(checksums_path)
    checkpoints.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )


@pytest.mark.parametrize("field", ["shape", "element_count"])
def test_rejects_noninteger_target_array_declarations(
    tmp_path: Path,
    field: str,
) -> None:
    targets, _ = _write_targets(tmp_path / field)
    manifest_path = targets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration = next(iter(manifest["target_arrays"]["arrays"].values()))
    if field == "shape":
        declaration["shape"][0] = float(declaration["shape"][0])
    else:
        declaration["element_count"] = float(declaration["element_count"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="target array declaration"):
        _load_target_package(
            targets,
            dataset_root=targets.parent / "sources",
        )


def test_rejects_snapshot_and_index_timestamp_relabel_against_frozen_source(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    index = json.loads(checkpoints.read_text(encoding="utf-8"))
    record = index["snapshots"][0]
    snapshot = checkpoints.parent / record["path"]
    metadata_path = snapshot / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["timestamp"] = 999.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    checksums_path = snapshot / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["metadata.json"] = _sha256(metadata_path)
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
    record["checksums_sha256"] = _sha256(checksums_path)
    record["timestamp_ns"] = 999_000_000_000
    checkpoints.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="source timestamp"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )


def test_rejects_missing_checkpoint_future_boundary_and_output_reuse(
    tmp_path: Path,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    payload = json.loads(checkpoints.read_text(encoding="utf-8"))
    payload["snapshots"] = payload["snapshots"][:-1]
    checkpoints.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing checkpoint"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )

    checkpoints = _write_checkpoint_index(tmp_path / "second", targets)
    payload = json.loads(checkpoints.read_text(encoding="utf-8"))
    payload["snapshots"][1]["consumed_through_frame"] = 2
    checkpoints.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="future|causal boundary"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )

    checkpoints = _write_checkpoint_index(tmp_path / "third", targets)
    output = tmp_path / "existing.json"
    output.write_text("do not replace", encoding="utf-8")
    with pytest.raises(ValueError, match="output already exists"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            output_path=output,
            dataset_root=targets.parent / "sources",
        )
    assert output.read_text(encoding="utf-8") == "do not replace"


def test_rejects_target_source_and_snapshot_drift(tmp_path: Path) -> None:
    targets, sources = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    sources["office.depth.000001"].write_bytes(b"drift")
    with pytest.raises(ValueError, match="source hash mismatch"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )

    targets, _ = _write_targets(tmp_path / "second")
    checkpoints = _write_checkpoint_index(tmp_path / "second", targets)
    index = json.loads(checkpoints.read_text(encoding="utf-8"))
    snapshot = checkpoints.parent / index["snapshots"][0]["path"]
    (snapshot / "ownership.npz").write_bytes(b"drift")
    with pytest.raises(ValueError, match="snapshot|checksum"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            dataset_root=targets.parent / "sources",
        )
