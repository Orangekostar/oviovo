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


def _run(root: Path, frames: tuple[int, ...] = (2, 7)) -> Path:
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
        snapshot.write_bytes(b"points-and-mesh")
        entities.write_bytes(b'{"entity_id":1,"metadata":{"x":1}}\n')
        ownership.write_bytes(b"ownership")
        geometry.write_bytes(b"geometry")
        metadata.write_bytes(b'{"frame_id":%d}\n' % frame)
        record = {
            "format": "oviv2_cumulative_audit_v1",
            "artifact": _tree_record(artifact, root),
            "snapshot": _file_record(snapshot, root),
            "entities": _file_record(entities, root),
            "voxel_snapshot": _tree_record(voxel, root),
        }
        checkpoints.append({"frame_index": frame, "cumulative_audit": record})
        inventory.extend(
            path.relative_to(root).as_posix()
            for path in audit.rglob("*")
            if path.is_file()
        )
    manifest = {
        "schema_version": 2,
        "protocol_id": "oviv2-tessecd-v2",
        "checkpoints": checkpoints,
        "artifact_inventory": sorted(inventory),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return root


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    left = _run(tmp_path / "left")
    right = tmp_path / "right"
    shutil.copytree(left, right)
    return left, right


def test_identical_cumulative_artifacts_have_stable_root(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    result = compare_cumulative_artifacts(left, right)
    assert set(result) == {"format", "checkpoint_frames", "inventory", "root_sha256"}
    assert result["format"] == "oviv2_cumulative_exact_v1"
    assert result["checkpoint_frames"] == [2, 7]
    assert len(result["root_sha256"]) == 64
    assert any(item["path"].endswith("ownership.npz") for item in result["inventory"])


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
