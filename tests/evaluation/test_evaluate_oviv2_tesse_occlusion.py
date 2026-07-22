from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.derive_tesse_cd_occlusion_v1 import (
    _preregistered_parameters,
    write_occlusion_package,
)
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    _load_checkpoint_snapshots,
    _load_target_package,
    build_evaluation_checkpoint_plan,
    evaluate_occlusion_package,
    main,
)
from src.oviv2.evidence import SparseEvidenceStore
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


def _write_checkpoint_index(root: Path, targets: Path) -> Path:
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
    payload = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_occlusion_checkpoints_v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "missing_observation_policy": "signed_depth",
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


def test_checkpoint_snapshots_are_loaded_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, _ = _write_targets(tmp_path / "fixture")
    checkpoints = _write_checkpoint_index(tmp_path / "fixture", targets)
    arrays, metadata, manifest_witness, _ = _load_target_package(targets)
    plan = build_evaluation_checkpoint_plan(arrays=arrays, metadata=metadata)
    original_load = VoxelMapSnapshot.load
    loaded: list[Path] = []

    def counted_load(path: str | Path) -> VoxelMapSnapshot:
        loaded.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(VoxelMapSnapshot, "load", classmethod(lambda _, path: counted_load(path)))
    snapshots, _, _ = _load_checkpoint_snapshots(
        checkpoints,
        metadata=metadata,
        target_manifest_witness=manifest_witness,
        checkpoint_plan=plan,
    )

    assert loaded == []
    assert snapshots[("apartment", 0)].metadata.frame_id == 0
    assert len(loaded) == 1


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
    )
    assert main(
        [
            "--targets",
            str(targets),
            "--checkpoints",
            str(checkpoints),
            "--output",
            str(second),
        ]
    ) == 0

    assert result["headline_gate"]["passed"] is True
    assert first.read_bytes() == second.read_bytes()
    assert b"NaN" not in first.read_bytes()


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
        )

    checkpoints = _write_checkpoint_index(tmp_path / "second", targets)
    payload = json.loads(checkpoints.read_text(encoding="utf-8"))
    payload["snapshots"][1]["consumed_through_frame"] = 2
    checkpoints.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="future|causal boundary"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
        )

    checkpoints = _write_checkpoint_index(tmp_path / "third", targets)
    output = tmp_path / "existing.json"
    output.write_text("do not replace", encoding="utf-8")
    with pytest.raises(ValueError, match="output already exists"):
        evaluate_occlusion_package(
            target_dir=targets,
            checkpoint_index=checkpoints,
            output_path=output,
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
        )
