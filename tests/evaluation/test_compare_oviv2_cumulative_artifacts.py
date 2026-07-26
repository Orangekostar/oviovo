from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tracemalloc

import pytest

from scripts.evaluation import compare_oviv2_cumulative_artifacts as compare_module
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


def _add_v2_source_index(root: Path) -> None:
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = manifest["checkpoints"][0]
    sidecars: dict[str, dict[str, object]] = {}
    for field, relative in (
        ("schedule", "inputs/schedule.json"),
        ("capture_status", "capture_status.json"),
        ("trajectories", "trajectories.jsonl"),
        ("frame_coverage", "temporal_frame_coverage.jsonl"),
        ("lifecycle_transitions", "lifecycle_transitions.jsonl"),
    ):
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((field + "\n").encode())
        sidecars[field] = _file_record(path, root)
        manifest["artifact_inventory"].append(relative)
    source_index = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "mode": "causal_checkpoint_exports",
        "method": "OVIV2",
        "scene": "fixture",
        **sidecars,
        "checkpoints": [
            {
                "frame_index": checkpoint["frame_index"],
                "timestamp_ns": 100,
                "consumed_through_frame": checkpoint["frame_index"],
                "consumed_through_frame_exclusive": checkpoint["frame_index"] + 1,
                "checkpoint_status": checkpoint["checkpoint_status"],
                "snapshot": checkpoint["cumulative_audit"]["snapshot"],
                "entities": checkpoint["neutral_entities"],
            }
        ],
    }
    source_path = root / "source_index.json"
    source_path.write_text(json.dumps(source_index, sort_keys=True) + "\n")
    manifest["source_index"] = _file_record(source_path, root)
    manifest["artifact_inventory"].append("source_index.json")
    manifest["artifact_inventory"].sort()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")


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


def _add_t1_receipt(root: Path) -> Path:
    audit = compare_cumulative_artifacts(root, root)
    receipt = root / "t1_exact_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "oviv2_t1_exact_execution_receipt_v1",
                "execution": {
                    "profile": "reference",
                    "argv": ["/usr/bin/python3", "/repo/run_reference.py"],
                    "pid": 1234,
                    "code_commit": "a" * 40,
                    "source_manifest_sha256": "b" * 64,
                    "input_fingerprints": {"config": "c" * 64},
                    "output_root": str(root.absolute()),
                },
                "source_manifest": {
                    "path": str((root.parent / "sources.json").absolute()),
                    "sha256": "b" * 64,
                    "byte_count": 10,
                },
                "artifact_inventory": audit["inventory"],
                "checkpoint_frames": audit["checkpoint_frames"],
                "cumulative_root_sha256": audit["root_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return receipt


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


def test_schema2_rejects_t1_receipt_as_an_unmanifested_extra(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    (right / "t1_exact_receipt.json").write_text("{}\n")
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_inventory"].append("t1_exact_receipt.json")
    manifest["artifact_inventory"].sort()
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="inventory|schema|pseudo-record"):
        compare_cumulative_artifacts(left, right)


@pytest.mark.parametrize("location", ["checkpoint", "source_bindings"])
def test_schema2_declared_inventory_cannot_be_extended_by_unknown_records(
    tmp_path: Path, location: str
) -> None:
    left, right = _pair(tmp_path)
    extra = right / "extra.bin"
    extra.write_bytes(b"extra")
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = _file_record(extra, right)
    if location == "checkpoint":
        manifest["checkpoints"][0]["unexpected_extra"] = record
    else:
        manifest["source_bindings"]["unexpected_extra"] = record
    manifest["artifact_inventory"].append("extra.bin")
    manifest["artifact_inventory"].sort()
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="inventory|schema|pseudo-record"):
        compare_cumulative_artifacts(left, right)


@pytest.mark.parametrize("location", ["checkpoint_alias", "deep_artifact_alias"])
def test_schema2_rejects_unknown_nested_records_that_alias_declared_files(
    tmp_path: Path, location: str
) -> None:
    left, right = _pair(tmp_path)
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = manifest["checkpoints"][0]
    alias = checkpoint["checkpoint_status"]
    if location == "checkpoint_alias":
        checkpoint["unexpected_alias"] = alias
    else:
        checkpoint["artifacts"] = {
            "temporal_current": {
                "format": "fixture",
                "artifact": checkpoint["cumulative_audit"]["artifact"],
                "checksums_sha256": "a" * 64,
                "unexpected": {"alias": alias},
            }
        }
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="schema|checkpoint"):
        compare_cumulative_artifacts(left, right)


@pytest.mark.parametrize("location", ["manifest_scalar", "source_index_scalar"])
def test_schema2_rejects_record_alias_in_a_known_scalar_slot(
    tmp_path: Path, location: str
) -> None:
    left, right = _pair(tmp_path)
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    alias = manifest["normalized_run_config"]
    if location == "manifest_scalar":
        manifest["protocol_id"] = alias
    else:
        _add_v2_source_index(right)
        manifest = json.loads(manifest_path.read_text())
        source_path = right / "source_index.json"
        source = json.loads(source_path.read_text())
        source["scene"] = alias
        source_path.write_text(json.dumps(source) + "\n")
        manifest["source_index"] = _file_record(source_path, right)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="schema|identity"):
        compare_cumulative_artifacts(left, right)


def test_schema2_source_index_cannot_alias_an_existing_file(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_index"] = manifest["final_artifact"]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="source index|path|schema"):
        compare_cumulative_artifacts(left, right)


def test_schema2_accepts_exact_production_source_index_sidecars(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    _add_v2_source_index(left)
    _add_v2_source_index(right)
    assert compare_cumulative_artifacts(left, right)["checkpoint_frames"] == [2, 7]


def test_schema1_accepts_only_a_strictly_bound_optional_t1_receipt(
    tmp_path: Path,
) -> None:
    left = _v1_run(tmp_path / "left")
    right = tmp_path / "right"
    shutil.copytree(left, right)
    left_receipt = _add_t1_receipt(left)
    right_receipt = _add_t1_receipt(right)

    assert compare_cumulative_artifacts(left, right)["checkpoint_frames"] == [2]

    value = json.loads(right_receipt.read_text())
    valid_root = value["cumulative_root_sha256"]
    value["cumulative_root_sha256"] = "f" * 64
    right_receipt.write_text(json.dumps(value) + "\n")
    with pytest.raises(ArtifactMismatch, match="T1 exact receipt"):
        compare_cumulative_artifacts(left, right)

    value["cumulative_root_sha256"] = valid_root
    value["schema_version"] = True
    right_receipt.write_text(json.dumps(value) + "\n")
    with pytest.raises(ArtifactMismatch, match="T1 exact receipt"):
        compare_cumulative_artifacts(left, right)

    right_receipt.unlink()
    left_receipt.unlink()
    extra = right / "undeclared.bin"
    extra.write_bytes(b"extra")
    manifest_path = right / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected_extra"] = _file_record(extra, right)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ArtifactMismatch, match="inventory"):
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


@pytest.mark.parametrize("relative", [False, True])
def test_opens_root_one_component_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool
) -> None:
    left, right = _pair(tmp_path)
    original = compare_module.os.open
    roots = {os.fspath(left.absolute()), os.fspath(right.absolute())}
    nondirfd_opens: list[str] = []

    def reject_complete_root(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        assert os.fspath(path) not in roots
        if "dir_fd" not in kwargs:
            nondirfd_opens.append(os.fspath(path))
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(compare_module.os, "open", reject_complete_root)
    if relative:
        monkeypatch.chdir(tmp_path)
        left_arg: Path | str = "left"
        right_arg: Path | str = "right"
    else:
        left_arg = left
        right_arg = right
    assert compare_cumulative_artifacts(left_arg, right_arg)[
        "checkpoint_frames"
    ] == [2, 7]
    assert nondirfd_opens == ["/", "/"]


def test_rejects_ancestor_exchange_after_component_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = tmp_path / "container"
    left, right = _pair(container)
    original = compare_module.os.open
    exchanged = False
    old = tmp_path / "old-container"

    def exchange_after_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal exchanged
        descriptor = original(path, flags, *args, **kwargs)
        if os.fspath(path) == "container" and not exchanged:
            exchanged = True
            container.replace(old)
            shutil.copytree(old, container)
        return descriptor

    monkeypatch.setattr(compare_module.os, "open", exchange_after_open)
    with pytest.raises(ArtifactMismatch, match="changed|replaced|unsafe"):
        compare_cumulative_artifacts(left, right)


def test_allows_unrelated_ancestor_directory_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    original = compare_module.os.open
    changed = False

    def add_unrelated_sibling(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal changed
        descriptor = original(path, flags, *args, **kwargs)
        if os.fspath(path) == "left" and not changed:
            changed = True
            (tmp_path / "unrelated").mkdir()
        return descriptor

    monkeypatch.setattr(compare_module.os, "open", add_unrelated_sibling)
    assert compare_cumulative_artifacts(left, right)["checkpoint_frames"] == [2, 7]


def test_scans_each_distinct_root_once_per_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    original = compare_module._all_regular_files
    scans: list[Path] = []

    def count_scan(
        root: compare_module._RootHandle,
    ) -> dict[str, compare_module.FileEntry]:
        scans.append(root.path)
        return original(root)

    monkeypatch.setattr(compare_module, "_all_regular_files", count_scan)
    compare_cumulative_artifacts(left, right)
    assert scans.count(left.absolute()) == 1
    assert scans.count(right.absolute()) == 1

    scans.clear()
    compare_cumulative_artifacts(left, left)
    assert scans == [left.absolute()]


def test_accepts_profile_specific_top_level_manifest_serialization(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    manifest = json.loads((right / "run_manifest.json").read_text())
    (right / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert compare_cumulative_artifacts(left, right)["checkpoint_frames"] == [2, 7]


def test_streams_large_cumulative_file(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    payload = b"0123456789abcdef" * (1024 * 1024)
    payload_size = len(payload)
    for root in (left, right):
        target = next(
            root.glob("checkpoints/*/cumulative_audit/artifact/entities/*.jsonl")
        )
        target.write_bytes(payload)
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        audit = manifest["checkpoints"][0]["cumulative_audit"]
        audit["entities"] = _file_record(target, root)
        audit["artifact"] = _tree_record(target.parents[1], root)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    del payload
    tracemalloc.start()
    try:
        result = compare_cumulative_artifacts(left, right)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert any(item["byte_count"] == payload_size for item in result["inventory"])
    assert peak < 16 * 1024 * 1024


def test_rejects_same_byte_inode_replacement_before_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    original = compare_module._open_regular
    replaced = False

    def replace_then_open(
        root: compare_module._RootHandle, relative: object, label: str
    ) -> int:
        nonlocal replaced
        if (
            root.path == right.absolute()
            and label.startswith("checkpoint/")
            and not replaced
        ):
            replaced = True
            target = root.path.joinpath(*relative.parts)
            replacement = target.with_name(target.name + ".replacement")
            replacement.write_bytes(target.read_bytes())
            replacement.replace(target)
        return original(root, relative, label)

    monkeypatch.setattr(compare_module, "_open_regular", replace_then_open)
    with pytest.raises(ArtifactMismatch, match="replaced"):
        compare_cumulative_artifacts(left, right)


def test_rejects_intermediate_directory_exchange_during_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    original = compare_module.os.listdir
    exchanged = False

    def exchange_after_listing(descriptor: int) -> list[str]:
        nonlocal exchanged
        names = original(descriptor)
        current = Path(compare_module.os.readlink(f"/proc/self/fd/{descriptor}"))
        if current == right and not exchanged:
            exchanged = True
            old = tmp_path / "old-checkpoints"
            (right / "checkpoints").replace(old)
            shutil.copytree(old, right / "checkpoints")
        return names

    monkeypatch.setattr(compare_module.os, "listdir", exchange_after_listing)
    with pytest.raises(ArtifactMismatch, match="changed"):
        compare_cumulative_artifacts(left, right)


def test_rejects_directory_entry_addition_during_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    original = compare_module.os.listdir
    added = False

    def add_after_listing(descriptor: int) -> list[str]:
        nonlocal added
        names = original(descriptor)
        current = Path(compare_module.os.readlink(f"/proc/self/fd/{descriptor}"))
        if current.name == "artifact" and str(current).startswith(str(right)) and not added:
            added = True
            (current / "late.bin").write_bytes(b"late")
        return names

    monkeypatch.setattr(compare_module.os, "listdir", add_after_listing)
    with pytest.raises(ArtifactMismatch, match="changed"):
        compare_cumulative_artifacts(left, right)


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


def test_cli_atomically_publishes_without_clobbering_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    output = tmp_path / "comparison.json"
    original = compare_module.os.open
    owner = b"owner-created-during-publication\n"
    created = False

    def create_owner_then_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal created
        if os.fspath(path) != output.name or created:
            return original(path, flags, mode, dir_fd=dir_fd)
        created = True
        descriptor = original(path, flags, mode, dir_fd=dir_fd)
        try:
            os.write(descriptor, owner)
        finally:
            os.close(descriptor)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(compare_module.os, "open", create_owner_then_open)
    with pytest.raises(FileExistsError):
        compare_module.main([str(left), str(right), "--output", str(output)])
    assert output.read_bytes() == owner


def test_cli_rejects_existing_symlink_and_symlinked_parent(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    owner = tmp_path / "owner.json"
    owner.write_bytes(b"owner\n")
    output = tmp_path / "comparison.json"
    output.symlink_to(owner)
    with pytest.raises(FileExistsError):
        compare_module.main([str(left), str(right), "--output", str(output)])
    assert output.is_symlink()
    assert owner.read_bytes() == b"owner\n"

    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ArtifactMismatch, match="symlink|unsafe"):
        compare_module.main(
            [str(left), str(right), "--output", str(alias / "new.json")]
        )


def test_cli_publishes_complete_output_and_fsyncs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    output = tmp_path / "comparison.json"
    original = compare_module.os.fsync
    directory_fsyncs = 0

    def count_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
        original(descriptor)

    monkeypatch.setattr(compare_module.os, "fsync", count_fsync)
    assert compare_module.main([str(left), str(right), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["checkpoint_frames"] == [2, 7]
    assert directory_fsyncs >= 1


def test_cli_preserves_a_replacement_after_output_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = _pair(tmp_path)
    output = tmp_path / "comparison.json"
    original = compare_module.os.fsync
    attacker = b"not-owned-by-comparator\n"
    saved = tmp_path / "saved-comparator-output.json"
    swapped = False

    def swap_after_file_fsync(descriptor: int) -> None:
        nonlocal swapped
        original(descriptor)
        if stat.S_ISREG(os.fstat(descriptor).st_mode) and not swapped:
            swapped = True
            output.replace(saved)
            output.write_bytes(attacker)

    monkeypatch.setattr(compare_module.os, "fsync", swap_after_file_fsync)
    with pytest.raises(RuntimeError, match="publication state is uncertain"):
        compare_module.main([str(left), str(right), "--output", str(output)])
    assert output.read_bytes() == attacker
    assert json.loads(saved.read_text())["checkpoint_frames"] == [2, 7]
