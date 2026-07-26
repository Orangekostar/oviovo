from __future__ import annotations

import hashlib
import gc
import json
import os
from pathlib import Path
import stat
import weakref

import numpy as np
import pytest

from scripts.evaluation.derive_tesse_cd_common_v2 import deterministic_npz_bytes
from scripts.evaluation.evaluate_oviv2_tesse_temporal_occlusion import (
    _publish,
    evaluate_temporal_occlusion_package,
)
from src.oviv2.temporal_snapshot import TemporalCompactCheckpoint, TemporalSnapshotMetadata


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _record(path: Path, root: Path | None = None) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path if root is None else path.relative_to(root)),
            "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}


def _tree(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256(); size = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        data = item.read_bytes(); size += len(data)
        digest.update(item.relative_to(path).as_posix().encode()); digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest()); digest.update(b"\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "byte_count": size}


def _install_mechanism_sources(
    run: Path, *, profile: str = "a4", scene: str = "apartment"
) -> dict[str, object]:
    trajectories = run / "trajectories.jsonl"
    trajectories.write_text(
        "\n".join(
            json.dumps(
                {
                    "frame_index": frame,
                    "timestamp_ns": 100 + frame * 100_000_000,
                    "entity_id": "7",
                    "centroid_xyz": [0.0, 0.0, 0.0],
                    "observation_count": frame + 1,
                    "dynamic_state": "dynamic",
                    "motion_confidence": 1.0,
                    "geometry_epoch": frame,
                    "readout_valid": frame == 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for frame in (0, 1)
        )
        + "\n"
    )
    lifecycle = run / "lifecycle_transitions.jsonl"
    lifecycle.write_text(
        json.dumps(
            {
                "frame_index": 1,
                "timestamp_ns": 100_000_100,
                "entity_id": "7",
                "before": "active",
                "after": "uncertain",
                "evidence": "visible_absent",
                "geometry_epoch": 1,
                "readout_valid": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    coverage = run / "temporal_frame_coverage.jsonl"
    coverage.write_text(
        "\n".join(
            json.dumps(
                {
                    "frame_index": frame,
                    "timestamp_ns": 100 + frame * 100_000_000,
                    "record_count": 1,
                    "event_count": frame,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for frame in (0, 1)
        )
        + "\n"
    )
    counters = {
        "proposal_opportunity_count": 3,
        "proposal_trigger_count": 2,
        "reid_opportunity_count": 4,
        "reid_trigger_count": 3,
        "motion_rejection_count": 1,
        "ledger_rejection_count": 0,
        "identity_expiry_count": 1,
        "geometry_reclaim_count": 1,
        "epoch_reset_opportunity_count": 2,
        "epoch_reset_trigger_count": 1,
        "icp_opportunity_count": 2,
        "icp_accept_count": 1,
        "icp_reject_count": 1,
        "ledger_stage_count": 2,
        "ledger_commit_count": 2,
        "ledger_reclaim_count": 1,
    }
    diagnostics = run / "runtime_diagnostics.json"
    _json(
        diagnostics,
        {
            "schema_version": 1,
            "execution_profile": profile,
            "processed_frame_count": 2,
            "counters": counters,
        },
    )
    source_index = run / "source_index.json"
    payload = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "mode": "causal_checkpoint_exports",
        "method": "OVIV2",
        "scene": scene,
        "trajectories": _record(trajectories, run),
        "frame_coverage": _record(coverage, run),
        "lifecycle_transitions": _record(lifecycle, run),
        "runtime_diagnostics": _record(diagnostics, run),
        "cached_mechanism_passed": True,
    }
    _json(source_index, payload)
    return _record(source_index, run)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"; dataset.mkdir()
    schedule = dataset / "schedule.json"; _json(schedule, {"schedule": "fixture"})
    arrays: dict[str, np.ndarray] = {}
    episodes = []
    sources: dict[str, dict[str, object]] = {}
    for role in ("source_manifest", "schedule", "rgbd_lock", "camera"):
        path = schedule if role == "schedule" else dataset / f"{role}.bin"
        if role != "schedule": path.write_bytes(role.encode())
        sources[role] = _record(path)
    for scene in ("apartment", "office"):
        for role in ("changes", "dsg_with_mesh", "export_manifest", "trajectory"):
            path = dataset / f"{scene}.{role}.bin"; path.write_bytes(role.encode())
            sources[f"{scene}.{role}"] = _record(path)
        timestamps = dataset / f"{scene}.timestamps.csv"
        timestamps.write_text("frame_index,sensor_timestamp_ns,relative_timestamp_ns\n0,100,0\n1,100000100,100000000\n")
        sources[f"{scene}.timestamps"] = _record(timestamps)
        for frame in (0, 1):
            path = dataset / f"{scene}.depth.{frame}.bin"; path.write_bytes(bytes([frame]))
            sources[f"{scene}.depth.{frame:06d}"] = _record(path)
        prefix = f"{scene}_e"
        arrays[f"{prefix}.anchor"] = np.asarray([[0, 0, 0]], dtype=np.int64)
        arrays[f"{prefix}.occluded"] = np.asarray([[0, 0, 0]], dtype=np.int64)
        episodes.append({
            "episode_id": prefix, "scene": scene, "object_id": 1,
            "object_name": "O(1)", "semantic_label": 1,
            "lifecycle": {"index": 0, "first_timestamp_ns": 0, "last_timestamp_ns": 2**64 - 1},
            "anchor": {"frame_index": 0, "relative_timestamp_ns": 0,
                       "array": f"{prefix}.anchor", "voxel_count": 1},
            "start_frame_index": 1, "end_frame_index": 1,
            "checkpoints": [{"frame_index": 1, "relative_timestamp_ns": 100_000_000,
                             "array": f"{prefix}.occluded", "occluded_voxel_count": 1,
                             "occlusion_fraction": 1.0}],
            "occlusion_fraction": 1.0,
        })
    target_dir = tmp_path / "target"; target_dir.mkdir()
    archive = deterministic_npz_bytes(arrays); (target_dir / "targets.npz").write_bytes(archive)
    params = json.loads(Path("configs/evaluation/manifests/tesse_cd_occlusion_v1.json").read_text())["parameters"]
    layers = {label: {"episode_count": 2, "episode_ids": [item["episode_id"] for item in episodes],
                      **({} if label == "all" else {"minimum_occlusion_fraction": float(label)})}
              for label in ("all", "0.50", "0.75", "0.90")}
    metadata = {"prediction_inputs_used": False, "parameters": params,
                "scene_frame_indices": {"apartment": [0, 1], "office": [0, 1]},
                "scenes": {scene: {"episode_count": 1, "headline_episode_count": 1}
                           for scene in ("apartment", "office")},
                "stress_layers": layers, "episodes": episodes}
    target = {"schema_version": 1, "manifest_id": "tesse_cd_occlusion_v1_targets",
              "dataset": "TESSE-CD", "status": "FIXTURE", "targets_generated": True,
              "prediction_inputs_used": False, "sources": sources, "metadata": metadata,
              "target_arrays": {**_record(target_dir / "targets.npz", target_dir),
                  "count": len(arrays), "arrays": {name: {"shape": list(value.shape),
                  "dtype": str(value.dtype), "element_count": int(value.size)}
                  for name, value in arrays.items()}}}
    target_path = target_dir / "manifest.json"; _json(target_path, target)
    run = tmp_path / "run"; (run / "checkpoints").mkdir(parents=True)
    records = []
    for frame in (0, 1):
        (run / "checkpoints" / str(frame)).mkdir()
        checkpoint = TemporalCompactCheckpoint(
            TemporalSnapshotMetadata("apartment", frame, (100 + frame * 100_000_000) / 1e9,
                                     frame + 1, 0.05, "a" * 64),
            np.asarray([7], np.int64), np.asarray([0 if frame == 0 else 1], np.uint8),
            np.asarray([1.0]), np.asarray([0], np.int64), np.asarray([0], np.int64),
            np.asarray([np.eye(4)]), np.asarray([[0, 0, 0]], np.int64), np.asarray([0, 1], np.int64),
        ).commit_new(run / "checkpoints" / str(frame) / "temporal_compact",
                     maximum_entities=4, maximum_object_voxels=4)
        records.append({"scene": "apartment", "frame_index": frame,
                        "timestamp_ns": 100 + frame * 100_000_000,
                        "relative_timestamp_ns": frame * 100_000_000,
                        "consumed_through_frame": frame,
                        "consumed_through_frame_exclusive": frame + 1,
                        "event_ids": ["overlap"] if frame == 1 else [],
                        "roles": ["official", "occlusion_v1"] if frame == 1 else ["occlusion_v1"],
                        "format": "oviv2_temporal_compact_checkpoint",
                        "maximum_entities": 4, "maximum_object_voxels": 4,
                        "artifact": _tree(checkpoint.path, run),
                        "checksums_sha256": hashlib.sha256((checkpoint.path / "checksums.json").read_bytes()).hexdigest()})
    binding = _record(target_path); binding.pop("path")
    common = {"algorithm_hash": "a" * 64, "schedule": {k: v for k, v in _record(schedule).items() if k != "path"},
              "target_manifest": binding, "input_sha256": "b" * 64,
              "code_commit": "c" * 40, "source_bindings": {"fixture": "bound"}}
    index = {"schema_version": 1, "format": "oviv2_temporal_compact_v1",
             "protocol_id": "oviv2-tessecd-v2", "dataset": "TESSE-CD", "method_id": "OVIV2",
             "scene": "apartment", **common, "checkpoints": records}
    index_path = run / "occlusion_checkpoint_index.json"; _json(index_path, index)
    manifest = {"schema_version": 2, "protocol_id": "oviv2-tessecd-v2", "dataset": "TESSE-CD",
                "method_id": "OVIV2", "scene": "apartment", **common,
                "occlusion_checkpoint_index": _record(index_path, run),
                "source_index": _install_mechanism_sources(run)}
    _json(run / "run_manifest.json", manifest)
    return target_path, index_path, dataset


def _office_index_from_fixture(
    tmp_path: Path, apartment_index: Path
) -> Path:
    source = json.loads(apartment_index.read_text())
    run = tmp_path / "office-run"
    (run / "checkpoints").mkdir(parents=True)
    records = []
    for source_record in source["checkpoints"]:
        frame = source_record["frame_index"]
        (run / "checkpoints" / str(frame)).mkdir()
        checkpoint = TemporalCompactCheckpoint(
            TemporalSnapshotMetadata(
                "office", frame, (100 + frame * 100_000_000) / 1e9,
                frame + 1, 0.05, "a" * 64,
            ),
            np.asarray([7], np.int64),
            np.asarray([0 if frame == 0 else 1], np.uint8),
            np.asarray([1.0]), np.asarray([0], np.int64), np.asarray([0], np.int64),
            np.asarray([np.eye(4)]), np.asarray([[0, 0, 0]], np.int64),
            np.asarray([0, 1], np.int64),
        ).commit_new(
            run / "checkpoints" / str(frame) / "temporal_compact",
            maximum_entities=4,
            maximum_object_voxels=4,
        )
        record = dict(source_record)
        record["scene"] = "office"
        record["artifact"] = _tree(checkpoint.path, run)
        record["checksums_sha256"] = hashlib.sha256(
            (checkpoint.path / "checksums.json").read_bytes()
        ).hexdigest()
        records.append(record)
    index = dict(source)
    index["scene"] = "office"
    index["checkpoints"] = records
    index_path = run / "occlusion_checkpoint_index.json"
    _json(index_path, index)
    manifest = {
        key: value for key, value in index.items() if key != "checkpoints"
    }
    manifest["schema_version"] = 2
    manifest["occlusion_checkpoint_index"] = _record(index_path, run)
    manifest["source_index"] = _install_mechanism_sources(run, scene="office")
    _json(run / "run_manifest.json", manifest)
    return index_path


def test_cli_evaluates_overlap_compact_and_publishes_canonical_no_replace(tmp_path: Path) -> None:
    target, index, dataset = _fixture(tmp_path)
    output = tmp_path / "result.json"
    result = evaluate_temporal_occlusion_package(
        targets=target, checkpoints=[index], dataset_root=dataset, output=output
    )
    assert result["events"][0]["counts"]["retained"] == 1
    assert result["input_bindings"]["maximum_cached_checkpoints"] == 1
    assert result["input_bindings"]["source_indexes"] == [
        {
            "scene": "apartment",
            **_record(index.parent / "source_index.json", index.parent),
        }
    ]
    assert result["mechanism_telemetry"] == result["macro"]["mechanism_telemetry"]
    assert result["mechanism_telemetry"]["readout_invalidation"] == {
        "opportunities": 1,
        "triggers": 1,
        "available": True,
        "passed": True,
        "reason": None,
        "source": _record(index.parent / "lifecycle_transitions.jsonl", index.parent),
    }
    assert set(result["mechanism_telemetry"]) == {
        "absence",
        "readout_invalidation",
        "proposal_recovery",
        "epoch_reset",
        "background_release",
        "background_reclaim",
        "eligible_reid",
        "icp_attempt",
        "icp_accept",
        "motion_rejection",
    }
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        evaluate_temporal_occlusion_package(
            targets=target, checkpoints=[index], dataset_root=dataset, output=output
        )


def test_mechanism_zero_opportunity_is_unavailable_and_cannot_pass(
    tmp_path: Path,
) -> None:
    target, index, dataset = _fixture(tmp_path)
    run = index.parent
    diagnostics_path = run / "runtime_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["counters"] = {
        key: 0 for key in diagnostics["counters"]
    }
    _json(diagnostics_path, diagnostics)
    source_index_path = run / "source_index.json"
    source_index = json.loads(source_index_path.read_text())
    source_index["runtime_diagnostics"] = _record(diagnostics_path, run)
    source_index["cached_mechanism_passed"] = True
    _json(source_index_path, source_index)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_index"] = _record(source_index_path, run)
    _json(manifest_path, manifest)

    result = evaluate_temporal_occlusion_package(
        targets=target, checkpoints=[index], dataset_root=dataset
    )

    assert result["mechanism_telemetry"]["proposal_recovery"] == {
        "opportunities": 0,
        "triggers": 0,
        "available": False,
        "passed": False,
        "reason": "no_opportunity",
        "source": _record(diagnostics_path, run),
    }


def test_package_rejects_runtime_diagnostics_hash_drift(tmp_path: Path) -> None:
    target, index, dataset = _fixture(tmp_path)
    (index.parent / "runtime_diagnostics.json").write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="runtime_diagnostics source binding mismatch"):
        evaluate_temporal_occlusion_package(
            targets=target, checkpoints=[index], dataset_root=dataset
        )


def test_package_loads_each_unique_checkpoint_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, index, dataset = _fixture(tmp_path)
    original = TemporalCompactCheckpoint.load.__func__
    calls = 0

    def counted(cls: type[TemporalCompactCheckpoint], *args: object, **kwargs: object) -> TemporalCompactCheckpoint:
        nonlocal calls
        calls += 1
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(TemporalCompactCheckpoint, "load", classmethod(counted))
    evaluate_temporal_occlusion_package(
        targets=target, checkpoints=[index], dataset_root=dataset
    )
    assert calls == 2


def test_package_rejects_coherent_artifact_tamper_after_first_load_without_reloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluation.evaluate_oviv2_tesse_temporal_occlusion as module

    target, index, dataset = _fixture(tmp_path)
    payload = json.loads(index.read_text())
    artifact = index.parent / payload["checkpoints"][0]["artifact"]["path"]
    original_evaluate = module.evaluate_temporal_occlusion
    original_load = TemporalCompactCheckpoint.load.__func__
    loads = 0

    def counted(cls: type[TemporalCompactCheckpoint], *args: object, **kwargs: object) -> TemporalCompactCheckpoint:
        nonlocal loads
        loads += 1
        return original_load(cls, *args, **kwargs)

    def tamper_after_evaluation(*args: object, **kwargs: object) -> dict[str, object]:
        result = original_evaluate(*args, **kwargs)
        manifest = artifact / "manifest.json"
        manifest_payload = json.loads(manifest.read_text())
        manifest_payload["metadata"]["config_sha256"] = "d" * 64
        manifest_content = (
            json.dumps(
                manifest_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        manifest.write_bytes(manifest_content)
        checksums = artifact / "checksums.json"
        checksum_payload = json.loads(checksums.read_text())
        checksum_payload["manifest.json"] = hashlib.sha256(manifest_content).hexdigest()
        checksums.write_text(
            json.dumps(
                checksum_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        return result

    monkeypatch.setattr(TemporalCompactCheckpoint, "load", classmethod(counted))
    monkeypatch.setattr(module, "evaluate_temporal_occlusion", tamper_after_evaluation)
    with pytest.raises(ValueError, match="compact artifact changed"):
        evaluate_temporal_occlusion_package(
            targets=target, checkpoints=[index], dataset_root=dataset
        )
    assert loads == 2


@pytest.mark.parametrize("mutation", ["symlink", "fifo", "parent"])
def test_validated_index_is_never_path_reread_after_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    import scripts.evaluation.evaluate_oviv2_tesse_temporal_occlusion as module

    target, index, dataset = _fixture(tmp_path)
    original_evaluate = module.evaluate_temporal_occlusion
    original_read_bytes = Path.read_bytes

    def mutate_after_evaluation(*args: object, **kwargs: object) -> dict[str, object]:
        result = original_evaluate(*args, **kwargs)
        if mutation == "parent":
            displaced = index.parent.with_name("displaced-run")
            index.parent.rename(displaced)
            index.parent.mkdir()
            for child in list(displaced.iterdir()):
                child.rename(index.parent / child.name)
        else:
            index.unlink()
            if mutation == "symlink":
                replacement = index.parent / "replacement"
                replacement.write_bytes(b"replacement")
                index.symlink_to(replacement)
            else:
                os.mkfifo(index)
        return result

    def forbid_index_reread(path: Path) -> bytes:
        if path.absolute() == index.absolute():
            raise AssertionError("validated index was path-reread")
        return original_read_bytes(path)

    monkeypatch.setattr(module, "evaluate_temporal_occlusion", mutate_after_evaluation)
    monkeypatch.setattr(Path, "read_bytes", forbid_index_reread)
    expected = "validated directory changed" if mutation == "parent" else None
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        evaluate_temporal_occlusion_package(
            targets=target, checkpoints=[index], dataset_root=dataset
        )


@pytest.mark.parametrize("field,replacement", [
    ("algorithm_hash", "d" * 64),
    ("code_commit", "e" * 40),
])
def test_multi_scene_indexes_require_identical_common_authority(
    tmp_path: Path, field: str, replacement: object,
) -> None:
    target, apartment, dataset = _fixture(tmp_path)
    office = _office_index_from_fixture(tmp_path, apartment)
    index = json.loads(office.read_text())
    index[field] = replacement
    _json(office, index)
    manifest_path = office.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = replacement
    manifest["occlusion_checkpoint_index"] = _record(office, office.parent)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="cross-scene authority mismatch"):
        evaluate_temporal_occlusion_package(
            targets=target,
            checkpoints=[apartment, office],
            dataset_root=dataset,
        )


def test_publish_fails_closed_if_parent_is_exchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "result.json"
    displaced = tmp_path / "displaced"
    original_link = os.link

    def exchange_parent(*args: object, **kwargs: object) -> None:
        parent.rename(displaced)
        parent.mkdir()
        original_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", exchange_parent)
    with pytest.raises(ValueError, match="output parent changed"):
        _publish(output, b"{}\n")
    assert not output.exists()


def test_publish_fails_closed_if_temporary_name_is_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    original_link = os.link

    def substitute_temporary(
        source: str, destination: str, **kwargs: object
    ) -> None:
        source_fd = kwargs["src_dir_fd"]
        os.unlink(source, dir_fd=source_fd)
        descriptor = os.open(
            source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
            dir_fd=source_fd,
        )
        os.write(descriptor, b"evil\n")
        os.close(descriptor)
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", substitute_temporary)
    with pytest.raises(ValueError, match="temporary output changed"):
        _publish(output, b"{}\n")
    assert not output.exists()


def test_publish_rechecks_parent_after_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "result.json"
    displaced = tmp_path / "displaced-after-fsync"
    original_fsync = os.fsync
    exchanged = False

    def exchange_after_directory_fsync(descriptor: int) -> None:
        nonlocal exchanged
        original_fsync(descriptor)
        if not exchanged and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            exchanged = True
            parent.rename(displaced)
            parent.mkdir()

    monkeypatch.setattr(os, "fsync", exchange_after_directory_fsync)
    with pytest.raises(ValueError, match="output parent changed"):
        _publish(output, b"{}\n")
    assert not output.exists()


@pytest.mark.parametrize("mutation", [
    "format", "missing", "duplicate", "future", "capacity", "traversal",
    "changed_index", "changed_source", "changed_artifact", "changed_target", "changed_schedule",
])
def test_package_fails_closed_on_contract_drift(tmp_path: Path, mutation: str) -> None:
    target, index, dataset = _fixture(tmp_path)
    payload = json.loads(index.read_text())
    if mutation == "format": payload["format"] = "oviv2_temporal_current_checkpoint"
    elif mutation == "missing": payload["checkpoints"].pop()
    elif mutation == "duplicate": payload["checkpoints"].append(payload["checkpoints"][0])
    elif mutation == "future": payload["checkpoints"][1]["consumed_through_frame"] = 2
    elif mutation == "capacity": payload["checkpoints"][0]["maximum_entities"] = 100_000
    elif mutation == "traversal": payload["checkpoints"][0]["artifact"]["path"] = "../escape"
    elif mutation == "changed_source": (dataset / "camera.bin").write_bytes(b"drift")
    elif mutation == "changed_artifact":
        path = index.parent / payload["checkpoints"][0]["artifact"]["path"] / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "changed_target": target.write_bytes(target.read_bytes() + b" ")
    elif mutation == "changed_schedule": (dataset / "schedule.json").write_bytes(b"drift")
    if mutation not in {"changed_index", "changed_source", "changed_artifact", "changed_target", "changed_schedule"}:
        _json(index, payload)
    elif mutation == "changed_index": index.write_bytes(index.read_bytes() + b" ")
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate_temporal_occlusion_package(
            targets=target, checkpoints=[index], dataset_root=dataset,
            output=tmp_path / "result.json"
        )


def test_package_rejects_symlink_extra_entry_and_nonfinite_json(tmp_path: Path) -> None:
    target, index, dataset = _fixture(tmp_path)
    payload = json.loads(index.read_text())
    artifact = index.parent / payload["checkpoints"][0]["artifact"]["path"]
    (artifact / "extra").write_bytes(b"x")
    with pytest.raises(ValueError):
        evaluate_temporal_occlusion_package(targets=target, checkpoints=[index], dataset_root=dataset)
    (artifact / "extra").unlink()
    real = artifact.parent / "real"; artifact.rename(real); artifact.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        evaluate_temporal_occlusion_package(targets=target, checkpoints=[index], dataset_root=dataset)
    artifact.unlink(); real.rename(artifact)
    index.write_text(index.read_text().replace('"schema_version":1', '"schema_version":NaN'))
    with pytest.raises(ValueError):
        evaluate_temporal_occlusion_package(targets=target, checkpoints=[index], dataset_root=dataset)


def test_lazy_evaluation_releases_each_checkpoint_before_loading_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluation.evaluate_oviv2_tesse_temporal_occlusion as module

    target, index, dataset = _fixture(tmp_path)
    original = module._CompactBinding.load
    live: list[weakref.ReferenceType[TemporalCompactCheckpoint]] = []

    def tracked(binding: object) -> TemporalCompactCheckpoint:
        gc.collect()
        assert not any(reference() is not None for reference in live)
        value = original(binding)
        live.append(weakref.ref(value))
        return value

    monkeypatch.setattr(module._CompactBinding, "load", tracked)
    evaluate_temporal_occlusion_package(
        targets=target, checkpoints=[index], dataset_root=dataset
    )


def test_real_formal_target_size_schema_and_scene_root_preflight() -> None:
    import scripts.evaluation.evaluate_oviv2_tesse_temporal_occlusion as module

    if os.environ.get("OVIV2_RUN_FORMAL_SMOKE") != "1":
        pytest.skip("set OVIV2_RUN_FORMAL_SMOKE=1 for the full formal asset hash smoke")
    asset_root = Path("/home/ww/oviovo_benchmark_assets/tesse_cd")
    target = asset_root / "derived/occlusion_v1_targets/20260722-stage3-fb97-a/manifest.json"
    scene_root = asset_root / "derived/rgbd_v1/apartment"
    if not target.is_file() or not scene_root.is_dir():
        pytest.skip("formal TESSE-CD assets are unavailable")
    assert 64 * 1024 * 1024 < target.stat().st_size < module._MAX_JSON_BYTES
    assert module._asset_root(scene_root) == asset_root
    arrays, metadata, target_record, _, sources, frame_times = module._load_target(
        target, scene_root
    )
    assert arrays and metadata["episodes"]
    assert target_record["byte_count"] == target.stat().st_size
    assert "schedule" in sources
    assert ("apartment", 0) in frame_times
