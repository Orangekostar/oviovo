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
        status = audit.parent / "checkpoint_status.json"
        neutral = audit.parent / "neutral.jsonl"
        status.write_bytes(b'{"status":"PASS"}\n')
        neutral.write_bytes(b'{"entity_id":1}\n')
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
    final.write_bytes(b"final-cumulative")
    inventory.append(final.relative_to(root).as_posix())
    manifest = {
        "schema_version": 2,
        "protocol_id": "oviv2-tessecd-v2",
        "checkpoints": checkpoints,
        "final_artifact": _file_record(final, root),
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


@pytest.mark.parametrize("mutation", ["manifest_indent", "final", "neutral", "status"])
def test_rejects_every_raw_manifest_final_and_checkpoint_difference(
    tmp_path: Path, mutation: str
) -> None:
    left, right = _pair(tmp_path)
    if mutation == "manifest_indent":
        manifest = json.loads((right / "run_manifest.json").read_text())
        (right / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    elif mutation == "final":
        (right / "final.bin").write_bytes(b"other-final")
    elif mutation == "neutral":
        next(right.glob("checkpoints/*/neutral.jsonl")).write_bytes(b'{"entity_id":2}\n')
    else:
        next(right.glob("checkpoints/*/checkpoint_status.json")).write_bytes(
            b'{"status": "PASS"}\n'
        )
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
