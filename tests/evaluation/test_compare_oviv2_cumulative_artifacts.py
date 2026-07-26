from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.evaluation.compare_oviv2_cumulative_artifacts import (
    ArtifactMismatch,
    compare_cumulative_artifacts,
)
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    canonical_algorithm_hash,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_record(root: Path, run_root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        size += len(data)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha(data)))
        digest.update(b"\n")
    return {
        "path": root.relative_to(run_root).as_posix(),
        "sha256": digest.hexdigest(),
        "byte_count": size,
    }


def _file_record(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(data),
        "byte_count": len(data),
    }


def _production_provenance(profile: str, *, commit: str = "a" * 40) -> dict[str, object]:
    return {
        "repository_commit": commit,
        "repository_tree": "b" * 40,
        "dirty_state_digest": _sha(b""),
        "command": ["python", "run_oviv2_tesse_cd_v2.py", profile],
        "hostname": "fixture-host",
        "platform": "fixture-platform",
        "machine": "x86_64",
        "cuda_visible_devices": None,
        "torch_cuda_version": "unavailable",
        "cudnn_version": None,
        "nvcc_version": [],
        "gpu_inventory": [],
        "library_versions": {
            name: "fixture"
            for name in ("numpy", "open3d", "torch", "scipy", "pillow")
        },
    }


def _run(
    root: Path, frames: tuple[int, ...] = (2, 7), *, profile: str = "a0"
) -> Path:
    checkpoints = []
    inventory = []
    for frame in frames:
        audit = root / "checkpoints" / f"{frame:08d}-100" / "cumulative_audit"
        artifact = audit / "artifact"
        voxel = audit / "voxel_snapshot"
        (artifact / "snapshots").mkdir(parents=True)
        (artifact / "entities").mkdir()
        voxel.mkdir()
        snapshot = artifact / "snapshots" / "neutral.npz"
        entities = artifact / "entities" / "neutral.jsonl"
        ownership = voxel / "ownership.npz"
        geometry = voxel / "geometry.npz"
        metadata = voxel / "metadata.json"
        cumulative_manifest = artifact / "manifest.json"
        cumulative_status = artifact / "checkpoint_status.json"
        cumulative_final = artifact / "final.bin"
        snapshot.write_bytes(b"points-and-mesh")
        entities.write_bytes(b'{"entity_id":1,"metadata":{"x":1}}\n')
        ownership.write_bytes(b"ownership")
        geometry.write_bytes(b"geometry")
        metadata.write_bytes(b'{"frame_id":%d}\n' % frame)
        cumulative_manifest.write_bytes(b'{"format":"cumulative-v1"}\n')
        cumulative_status.write_bytes(b'{"status":"PASS"}\n')
        cumulative_final.write_bytes(b"final-cumulative")
        status = audit.parent / "checkpoint_status.json"
        neutral = audit.parent / "neutral.jsonl"
        status.write_bytes(f'{{"profile":"{profile}","status":"PASS"}}\n'.encode())
        neutral.write_bytes(f'{{"entity_id":1,"profile":"{profile}"}}\n'.encode())
        record = {
            "format": "oviv2_cumulative_audit_v1",
            "artifact": _tree_record(artifact, root),
            "snapshot": _file_record(snapshot, root),
            "entities": _file_record(entities, root),
            "voxel_snapshot": _tree_record(voxel, root),
        }
        checkpoints.append({
            "frame_index": frame,
            "checkpoint_status": _file_record(status, root),
            "neutral_entities": _file_record(neutral, root),
            "cumulative_audit": record,
        })
        inventory.extend(
            path.relative_to(root).as_posix()
            for path in audit.rglob("*")
            if path.is_file()
        )
        inventory.extend((status.relative_to(root).as_posix(), neutral.relative_to(root).as_posix()))
    final = root / "final.bin"
    final.write_bytes(f"profile-final:{profile}".encode())
    inventory.append(final.relative_to(root).as_posix())
    normalized = root / "normalized_run_config.json"
    normalized_config = {
        "missing_observation_policy": "signed_depth",
        "temporal_readout": {
            "execution_profile": profile,
            "lifecycle": {"confirm_hits": 2},
        },
    }
    algorithm_hash = canonical_algorithm_hash(normalized_config)
    normalized_config["algorithm_hash"] = algorithm_hash
    normalized.write_bytes(json.dumps(normalized_config, sort_keys=True).encode() + b"\n")
    inventory.append(normalized.relative_to(root).as_posix())
    manifest = {
        "schema_version": 2,
        "protocol_id": "oviv2-tessecd-v2",
        "algorithm_hash": algorithm_hash,
        "code_commit": "a" * 40,
        "source_bindings": {"dataset": "fixture"},
        "normalized_run_config": _file_record(normalized, root),
        "checkpoints": checkpoints,
        "final_artifact": _file_record(final, root),
        "artifact_inventory": sorted(inventory),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (root / "execution_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": _production_provenance(profile),
                "environment": {"pid": len(profile)},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return root


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    left = _run(tmp_path / "left")
    right = tmp_path / "right"
    shutil.copytree(left, right)
    return left, right


def _v1_run(root: Path) -> Path:
    checkpoint_root = root / "checkpoints/00000002-100"
    artifact = checkpoint_root / "artifact"
    voxel = checkpoint_root / "voxel_snapshot"
    artifact.mkdir(parents=True)
    voxel.mkdir()
    neutral_snapshot = artifact / "neutral.npz"
    neutral_entities = artifact / "neutral.jsonl"
    status = checkpoint_root / "checkpoint_status.json"
    neutral_snapshot.write_bytes(b"neutral-points")
    neutral_entities.write_bytes(b'{"entity_id":1}\n')
    status.write_bytes(b'{"status":"PASS"}\n')
    (voxel / "ownership.npz").write_bytes(b"ownership")
    final = root / "final.bin"
    final.write_bytes(b"final")
    manifest = {
        "schema_version": 1,
        "checkpoints": [{
            "frame_index": 2,
            "artifact": _tree_record(artifact, root),
            "voxel_snapshot": _tree_record(voxel, root),
            "checkpoint_status": _file_record(status, root),
            "neutral_snapshot": _file_record(neutral_snapshot, root),
            "neutral_entities": _file_record(neutral_entities, root),
        }],
        "final_artifact": _file_record(final, root),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return root


def _v1_projection_of_schema2(root: Path, schema2: Path) -> Path:
    checkpoints = []
    for audit in sorted(schema2.glob("checkpoints/*/cumulative_audit")):
        source_checkpoint = audit.parent
        frame = int(source_checkpoint.name.split("-", 1)[0])
        checkpoint = root / "checkpoints" / source_checkpoint.name
        artifact = checkpoint / "artifact"
        voxel = checkpoint / "voxel_snapshot"
        shutil.copytree(audit / "artifact", artifact)
        shutil.copytree(audit / "voxel_snapshot", voxel)
        status = checkpoint / "checkpoint_status.json"
        status.write_bytes(b'{"status":"PASS"}\n')
        checkpoints.append(
            {
                "frame_index": frame,
                "artifact": _tree_record(artifact, root),
                "voxel_snapshot": _tree_record(voxel, root),
                "neutral_snapshot": _file_record(artifact / "snapshots/neutral.npz", root),
                "neutral_entities": _file_record(artifact / "entities/neutral.jsonl", root),
                "checkpoint_status": _file_record(status, root),
            }
        )
    (root / "run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "checkpoints": checkpoints}, sort_keys=True) + "\n"
    )
    return root


def test_identical_cumulative_artifacts_have_stable_root(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    result = compare_cumulative_artifacts(left, right)
    assert set(result) == {"format", "checkpoint_frames", "inventory", "root_sha256"}
    assert result["format"] == "oviv2_cumulative_exact_v1"
    assert result["checkpoint_frames"] == [2, 7]
    assert len(result["root_sha256"]) == 64
    assert any(item["path"].endswith("ownership.npz") for item in result["inventory"])


def test_schema2_projects_only_cumulative_audit_across_profiles(tmp_path: Path) -> None:
    left = _run(tmp_path / "a0", profile="a0")
    right = _run(tmp_path / "a1", profile="a1")

    result = compare_cumulative_artifacts(left, right)

    assert result["checkpoint_frames"] == [2, 7]
    assert all(item["path"].startswith("checkpoint/") for item in result["inventory"])
    assert not any(
        item["path"].endswith(("run_manifest.json", "execution_receipt.json", "normalized_run_config.json"))
        for item in result["inventory"]
    )


def test_reference_v1_projects_to_schema2_cumulative_audit(tmp_path: Path) -> None:
    schema2 = _run(tmp_path / "a0", profile="a0")
    reference = _v1_projection_of_schema2(tmp_path / "reference", schema2)

    cross = compare_cumulative_artifacts(reference, schema2)

    assert cross == compare_cumulative_artifacts(schema2, schema2)


@pytest.mark.parametrize("mutation", ["normalized", "receipt", "source", "extra"])
def test_schema2_strictly_self_validates_identity_and_full_inventory(
    tmp_path: Path, mutation: str
) -> None:
    left, right = _pair(tmp_path)
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "normalized":
        normalized = right / "normalized_run_config.json"
        payload = json.loads(normalized.read_text())
        payload["algorithm_hash"] = "f" * 64
        normalized.write_text(json.dumps(payload, sort_keys=True) + "\n")
        manifest["normalized_run_config"] = _file_record(normalized, right)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    elif mutation == "receipt":
        receipt = json.loads((right / "execution_receipt.json").read_text())
        del receipt["environment"]
        (right / "execution_receipt.json").write_text(json.dumps(receipt) + "\n")
    elif mutation == "source":
        manifest["source_bindings"] = []
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    else:
        (right / "profile-only-extra.bin").write_bytes(b"undeclared")
    with pytest.raises(ArtifactMismatch, match="identity|receipt|source|inventory"):
        compare_cumulative_artifacts(left, right)


@pytest.mark.parametrize("mode", ["root_schema", "self", "cross_profile"])
def test_schema2_rejects_production_repository_commit_drift(
    tmp_path: Path, mode: str
) -> None:
    left = _run(tmp_path / "a0", profile="a0")
    right = _run(tmp_path / "a1", profile="a1")
    receipt_path = right / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if mode == "root_schema":
        receipt["provenance"]["unexpected"] = "field"
    else:
        receipt["provenance"]["repository_commit"] = "f" * 40
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(ArtifactMismatch, match="receipt.*identity|provenance"):
        compare_cumulative_artifacts(
            right if mode in {"root_schema", "self"} else left,
            right,
        )


def test_schema2_recomputes_algorithm_hash_from_normalized_config(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", profile="a1")
    normalized_path = run / "normalized_run_config.json"
    normalized = json.loads(normalized_path.read_text())
    stale_hash = normalized["algorithm_hash"]
    normalized["temporal_readout"]["lifecycle"]["confirm_hits"] += 1
    normalized_path.write_text(json.dumps(normalized, sort_keys=True) + "\n")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalized_run_config"] = _file_record(normalized_path, run)
    assert manifest["algorithm_hash"] == stale_hash
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(ArtifactMismatch, match="algorithm identity"):
        compare_cumulative_artifacts(run, run)


def test_rejects_missing_or_extra_checkpoint(tmp_path: Path) -> None:
    left = _run(tmp_path / "left")
    right = _run(tmp_path / "right", (2,))
    with pytest.raises(ArtifactMismatch, match="checkpoint inventory"):
        compare_cumulative_artifacts(left, right)
    _run(tmp_path / "extra", (2, 7, 9))
    with pytest.raises(ArtifactMismatch, match="checkpoint inventory"):
        compare_cumulative_artifacts(left, tmp_path / "extra")


def test_rejects_metadata_normalization_even_if_semantically_equal(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    entities = next(right.glob("checkpoints/*/cumulative_audit/artifact/entities/*.jsonl"))
    entities.write_bytes(b'{"metadata":{"x":1},"entity_id":1}\n')
    manifest = json.loads((right / "run_manifest.json").read_text())
    checkpoint = manifest["checkpoints"][0]["cumulative_audit"]
    checkpoint["entities"] = _file_record(entities, right)
    checkpoint["artifact"] = _tree_record(entities.parents[1], right)
    (right / "run_manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="raw bytes"):
        compare_cumulative_artifacts(left, right)


def test_rejects_manifest_hash_lie_path_escape_and_symlink(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"][0]["cumulative_audit"]["entities"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="manifest record"):
        compare_cumulative_artifacts(left, right)

    shutil.rmtree(right)
    shutil.copytree(left, right)
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"][0]["cumulative_audit"]["entities"]["path"] = "../escape"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="path"):
        compare_cumulative_artifacts(left, right)

    shutil.rmtree(right)
    shutil.copytree(left, right)
    target = next(right.glob("checkpoints/*/cumulative_audit/artifact/entities/*.jsonl"))
    data = target.read_bytes()
    target.unlink()
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(data)
    target.symlink_to(outside)
    with pytest.raises(ArtifactMismatch, match="symlink"):
        compare_cumulative_artifacts(left, right)


def test_rejects_unmanifested_cumulative_file_and_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    left, right = _pair(tmp_path)
    extra = next(right.glob("checkpoints/*/cumulative_audit")) / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ArtifactMismatch, match="inventory"):
        compare_cumulative_artifacts(left, right)

    extra.unlink()
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ArtifactMismatch, match="symlink"):
        compare_cumulative_artifacts(left, alias / "right")


def test_accepts_profile_specific_top_level_manifest_serialization(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    manifest = json.loads((right / "run_manifest.json").read_text())
    (right / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert compare_cumulative_artifacts(left, right)["checkpoint_frames"] == [2, 7]


@pytest.mark.parametrize(
    "relative",
    [
        "artifact/manifest.json",
        "artifact/entities/neutral.jsonl",
        "artifact/checkpoint_status.json",
        "artifact/final.bin",
        "voxel_snapshot/ownership.npz",
    ],
)
def test_rejects_every_cumulative_manifest_neutral_status_final_and_file_difference(
    tmp_path: Path, relative: str
) -> None:
    left, right = _pair(tmp_path)
    target = next(right.glob("checkpoints/*/cumulative_audit")) / relative
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ArtifactMismatch, match="raw bytes|manifest record|inventory"):
        compare_cumulative_artifacts(left, right)


@pytest.mark.parametrize("name", ["checkpoint_status.json", "neutral.jsonl", "final.bin"])
def test_v1_inventory_includes_status_neutral_and_final(tmp_path: Path, name: str) -> None:
    left = _v1_run(tmp_path / "v1-left")
    right = tmp_path / "v1-right"
    shutil.copytree(left, right)
    target = next(right.rglob(name))
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ArtifactMismatch, match="manifest record|raw bytes"):
        compare_cumulative_artifacts(left, right)
