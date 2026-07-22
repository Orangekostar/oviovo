from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.evaluation.run_oviv2_tesse_cd as runner_module
from scripts.evaluation.export_tesse_temporal_artifact import export_temporal_artifact
from scripts.evaluation.run_oviv2_tesse_cd import (
    RunnerDependencies,
    _cache_prefix_sha256,
    _load_causal_checkpoints,
    _validate_dense_provenance,
    algorithm_hash,
    apply_frozen_visibility_policy,
    parse_args,
    run,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot
from src.oviv2.compact_checkpoint import (
    COMPACT_OWNERSHIP_FORMAT,
    CompactOwnershipCheckpoint,
    CompactOwnershipMetadata,
)
from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.ownership import ReversibleOwnershipStore


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _entries() -> list[dict[str, object]]:
    return [
        {
            "event_ids": ["event-1"],
            "frame_index": 1,
            "relative_timestamp_ns": 10,
            "roles": ["official", "common_v2"],
            "timestamp_ns": 110,
        },
        {
            "event_ids": ["event-2"],
            "frame_index": 3,
            "relative_timestamp_ns": 30,
            "roles": ["common_v2"],
            "timestamp_ns": 130,
        },
    ]


def _write_schedule(
    path: Path,
    entries: list[dict[str, object]] | None = None,
) -> None:
    if entries is None:
        entries = _entries()
    _write_json(
        path,
        {
            "dataset": "TESSE-CD",
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "method_predictions_used": False,
            "parameters": {"frame_indexing": "zero_based"},
            "scenes": {
                "apartment": {"entries": entries, "frame_count": 5},
                "office": {"entries": entries, "frame_count": 5},
            },
            "schema_version": 2,
        },
    )


def _write_config(tmp_path: Path) -> Path:
    schedule = tmp_path / "schedule.json"
    _write_schedule(schedule)
    target_manifest = tmp_path / "occlusion-target-manifest.json"
    _write_json(target_manifest, {"manifest_id": "test-occlusion-target"})
    config = {
        "dataset": "TESSE-CD",
        "frame_count": 5,
        "method_id": "OVIV2",
        "missing_observation_policy": "signed_depth",
        "occlusion_target_manifest": str(target_manifest),
        "occlusion_target_manifest_sha256": runner_module._sha256(target_manifest),
        "evaluation_checkpoint_frames_sha256": "f" * 64,
        "scene": "apartment",
        "schedule_manifest": str(schedule),
        "schema_version": 1,
        "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
    }
    path = tmp_path / "runner.json"
    _write_json(path, config)
    return path


class _Dataset:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(frame_id=index, timestamp=(100 + 10 * index) / 1e9)

    def timestamp_ns(self, index: int) -> int:
        return 100 + 10 * index


class _Caches:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.bindings = {
            "frontend": {"byte_count": 8, "sha256": "a" * 64},
            "dense": {"byte_count": 5, "sha256": "b" * 64},
        }

    def load(self, frame_index: int, frame: object) -> tuple[tuple[object, ...], object]:
        assert getattr(frame, "frame_id") == frame_index
        self.calls.append(f"load:{frame_index}")
        return (), f"dense:{frame_index}"


class _Runtime:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.revision = 0
        self.last_frame = -1
        self.last_timestamp = 0.0
        self.ownership = ReversibleOwnershipStore(block_resolution=8)

    def process_frame(
        self,
        frame: object,
        *,
        observations: tuple[object, ...],
        dense_semantics: object,
    ) -> SimpleNamespace:
        del observations, dense_semantics
        frame_index = int(getattr(frame, "frame_id"))
        self.calls.append(f"process:{frame_index}")
        self.last_frame = frame_index
        self.last_timestamp = float(getattr(frame, "timestamp"))
        self.revision += 1
        return SimpleNamespace(revision=self.revision)

    def commit_new(self, target: Path) -> SimpleNamespace:
        frame_index = self.last_frame
        self.calls.append(f"commit:{frame_index}")
        target.mkdir()
        state = f"state:{frame_index}\n".encode()
        (target / "state.bin").write_bytes(state)
        _write_json(
            target / "checksums.json",
            {"state.bin": hashlib.sha256(state).hexdigest()},
        )
        return SimpleNamespace(
            path=target,
            metadata=SimpleNamespace(
                frame_id=frame_index,
                timestamp=self.last_timestamp,
            ),
        )

    def commit_compact_ownership_new(
        self, target: Path
    ) -> CompactOwnershipCheckpoint:
        frame_index = self.last_frame
        self.calls.append(f"compact:{frame_index}")
        return CompactOwnershipCheckpoint.commit_new(
            target,
            CompactOwnershipMetadata(
                scene_id="apartment",
                frame_id=frame_index,
                timestamp=self.last_timestamp,
                revision=self.revision,
                voxel_size_m=0.05,
                block_resolution=8,
                dense_semantic_provenance=DenseSemanticProvenance(
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
                ),
            ),
            self.ownership,
        )


def _dependencies(calls: list[str]) -> RunnerDependencies:
    def export_checkpoint(
        snapshot: object,
        checkpoint: object,
        destination: Path,
        context: object,
    ) -> dict[str, object]:
        del snapshot, context
        frame_index = int(getattr(checkpoint, "frame_index"))
        calls.append(f"export:{frame_index}")
        neutral = MapSnapshot(
            method="OVIV2",
            scene_id="apartment",
            timestamp=float(getattr(checkpoint, "timestamp_ns")),
            entities=[
                EntityPrediction(
                    entity_id=f"entity-{frame_index}",
                    points_xyz=np.asarray(
                        [[float(frame_index), 0.0, 1.0]],
                        dtype=np.float32,
                    ),
                    semantic_embedding=None,
                    semantic_label="chair",
                    semantic_score=1.0,
                    lifecycle_state="active",
                    first_seen=float(getattr(checkpoint, "timestamp_ns")),
                    last_seen=float(getattr(checkpoint, "timestamp_ns")),
                    metadata={"entity_type": "object"},
                )
            ],
            background_xyz=None,
            scope="current",
        )
        paths = write_map_snapshot(neutral, destination / "artifact")
        return {
            "snapshot": paths["snapshot"],
            "entities": paths["entities"],
            "trajectory_rows": [
                {
                    "centroid_xyz": [float(frame_index), 0.0, 1.0],
                    "entity_id": f"entity-{frame_index}",
                    "frame_index": frame_index,
                    "timestamp_ns": 100 + 10 * frame_index,
                }
            ],
        }

    return RunnerDependencies(
        dataset_factory=lambda _config: _Dataset(calls),
        cache_loader_factory=lambda _config, _dataset: _Caches(calls),
        runtime_factory=lambda _config, _caches: _Runtime(calls),
        checkpoint_exporter=export_checkpoint,
        provenance_factory=lambda: {"hostname": "test-host"},
    )


def _deterministic_files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"run_provenance.json", "timing.json"}
    }


def test_five_frame_end_to_end_is_byte_identical(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    first_calls: list[str] = []
    second_calls: list[str] = []

    first = run(config, tmp_path / "run-a", dependencies=_dependencies(first_calls))
    second = run(config, tmp_path / "run-b", dependencies=_dependencies(second_calls))

    expected_calls = [
        "load:0", "process:0",
        "load:1", "process:1", "commit:1", "export:1",
        "load:2", "process:2",
        "load:3", "process:3", "commit:3", "export:3",
        "load:4", "process:4",
    ]
    assert first_calls == expected_calls
    assert second_calls == expected_calls
    assert first["processed_frame_count"] == 5
    assert second["processed_frame_count"] == 5
    assert [item["frame_index"] for item in first["checkpoints"]] == [1, 3]
    expected_checkpoint = {
        "consumed_through_frame": 1,
        "consumed_through_frame_exclusive": 2,
        "event_ids": ["event-1"],
        "frame_index": 1,
        "relative_timestamp_ns": 10,
        "roles": ["official", "common_v2"],
        "scene": "apartment",
        "timestamp_ns": 110,
    }
    for key, value in expected_checkpoint.items():
        assert first["checkpoints"][0][key] == value
    assert first["checkpoints"][0]["voxel_snapshot"]["path"] == (
        "checkpoints/00000001-110/voxel_snapshot"
    )
    assert first["checkpoints"][0]["voxel_snapshot"]["byte_count"] > 0
    assert len(first["checkpoints"][0]["voxel_snapshot"]["sha256"]) == 64
    assert first["checkpoints"][0]["artifact"]["path"] == (
        "checkpoints/00000001-110/artifact"
    )
    assert first["checkpoints"][0]["artifact"]["byte_count"] > 0
    assert len(first["checkpoints"][0]["artifact"]["sha256"]) == 64
    source_index = json.loads(
        (tmp_path / "run-a/source_index.json").read_text(encoding="utf-8")
    )
    assert source_index["method"] == "OVIV2"
    assert [item["frame_index"] for item in source_index["checkpoints"]] == [1, 3]
    assert source_index["checkpoints"][0]["snapshot"]["path"] == (
        "checkpoints/00000001-110/artifact/snapshots/110.000000_current.npz"
    )
    status = json.loads(
        (
            tmp_path
            / "run-a/checkpoints/00000001-110/checkpoint_status.json"
        ).read_text(encoding="utf-8")
    )
    assert status == {
        "checkpoint_frame": 1,
        "consumed_through_frame": 1,
        "consumed_through_frame_exclusive": 2,
        "event_ids": ["event-1"],
        "roles": ["official", "common_v2"],
        "schema_version": 1,
        "status": "PASS",
        "timestamp_ns": 110,
    }
    temporal_manifest = export_temporal_artifact(
        tmp_path / "run-a/source_index.json",
        tmp_path / "temporal",
    )
    assert temporal_manifest.is_file()
    assert _deterministic_files(tmp_path / "run-a") == _deterministic_files(
        tmp_path / "run-b"
    )


def test_frozen_evaluation_checkpoints_are_union_with_official_schedule(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evaluation_checkpoint_frames"] = [2, 3]
    _write_json(config_path, config)
    calls: list[str] = []

    manifest = run(config_path, tmp_path / "run", dependencies=_dependencies(calls))

    assert manifest["official_schedule_frame_indices"] == [1, 3]
    assert manifest["evaluation_checkpoint_frames"] == [2, 3]
    assert manifest["scheduled_frame_indices"] == [1, 2, 3]
    assert [item["frame_index"] for item in manifest["checkpoints"]] == [1, 2, 3]
    assert manifest["checkpoints"][1]["timestamp_ns"] == 120
    assert manifest["checkpoints"][1]["relative_timestamp_ns"] == 20
    assert manifest["checkpoints"][1]["roles"] == ["occlusion_v1"]
    source_index = json.loads(
        (tmp_path / "run/source_index.json").read_text(encoding="utf-8")
    )
    assert [item["frame_index"] for item in source_index["checkpoints"]] == [1, 3]
    trajectory_frames = [
        json.loads(line)["frame_index"]
        for line in (tmp_path / "run/trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert trajectory_frames == [1, 3]
    assert calls == [
        "load:0", "process:0",
        "load:1", "process:1", "commit:1", "export:1",
        "load:2", "process:2", "compact:2",
        "load:3", "process:3", "commit:3", "export:3",
        "load:4", "process:4",
    ]

    evaluation_root = tmp_path / "run/checkpoints/00000002-120"
    assert {path.name for path in evaluation_root.rglob("*")} == {
        "ownership_checkpoint",
        "metadata.json",
        "ownership.npz",
        "checksums.json",
        "checkpoint_status.json",
    }
    assert not any(
        path.name in {"geometry.npz", "evidence.npz", "entities.jsonl", "artifact"}
        for path in evaluation_root.rglob("*")
    )
    index_path = tmp_path / "run/occlusion_checkpoint_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    assert index["evaluation_checkpoint_frames_sha256"] == "f" * 64
    assert index["target_manifest"] == {
        "byte_count": Path(config["occlusion_target_manifest"]).stat().st_size,
        "sha256": config["occlusion_target_manifest_sha256"],
    }
    assert [record["frame_index"] for record in index["snapshots"]] == [2, 3]
    assert index["snapshots"][0]["format"] == COMPACT_OWNERSHIP_FORMAT
    assert index["snapshots"][0]["path"].endswith("/ownership_checkpoint")
    assert index["snapshots"][1]["format"] == "oviv2_voxel_map_snapshot"
    assert index["snapshots"][1]["path"].endswith("/voxel_snapshot")
    normalized = tmp_path / "run/normalized_run_config.json"
    assert index["run_config"] == {
        "path": "normalized_run_config.json",
        "sha256": runner_module._sha256(normalized),
        "byte_count": normalized.stat().st_size,
    }


def test_mixed_checkpoint_run_is_byte_identical_and_compact_is_bounded(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evaluation_checkpoint_frames"] = [0, 2, 3, 4]
    _write_json(config_path, config)

    run(config_path, tmp_path / "first", dependencies=_dependencies([]))
    run(config_path, tmp_path / "second", dependencies=_dependencies([]))

    assert _deterministic_files(tmp_path / "first") == _deterministic_files(
        tmp_path / "second"
    )
    for frame_index, timestamp_ns in ((0, 100), (2, 120), (4, 140)):
        root = (
            tmp_path
            / "first/checkpoints"
            / f"{frame_index:08d}-{timestamp_ns}"
        )
        assert sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ) < 64 * 1024
        assert not any(
            path.name in {
                "geometry.npz",
                "evidence.npz",
                "entities.jsonl",
                "artifact",
            }
            for path in root.rglob("*")
        )


def test_official_only_checkpoints_keep_full_snapshot_and_neutral_inventory(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)

    run(config_path, tmp_path / "run", dependencies=_dependencies([]))

    for frame_index, timestamp_ns in ((1, 110), (3, 130)):
        root = (
            tmp_path
            / "run/checkpoints"
            / f"{frame_index:08d}-{timestamp_ns}"
        )
        assert {path.name for path in root.iterdir()} == {
            "voxel_snapshot",
            "artifact",
            "checkpoint_status.json",
        }
        assert (root / "voxel_snapshot/checksums.json").is_file()
        assert not (root / "ownership_checkpoint").exists()
    index = json.loads(
        (tmp_path / "run/occlusion_checkpoint_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["snapshots"] == []


def test_frozen_visibility_policy_overrides_runtime_parser_default() -> None:
    @dataclass(frozen=True)
    class RuntimeConfig:
        missing_observation_policy: str = "signed_depth"

    frozen = apply_frozen_visibility_policy(
        RuntimeConfig(),
        "missing_as_absence",
    )

    assert frozen.missing_observation_policy == "missing_as_absence"


def test_dense_provenance_binds_worker_vocabulary_and_cache_prefix() -> None:
    cache_hashes = {"frame000000.npz": "a" * 64}
    provenance = {
        "backend": "radseg",
        "source_commit": "1" * 40,
        "radio_commit": "2" * 40,
        "model_id": "radseg:model",
        "model_sha256": "3" * 64,
        "auxiliary_model_sha256": "4" * 64,
        "language_model_id": "language:model",
        "language_model_revision": "5" * 40,
        "language_model_sha256": "6" * 64,
        "vocabulary_sha256": "7" * 64,
        "prompt_sha256": "8" * 64,
        "inference_config_sha256": "9" * 64,
        "cache_prefix_sha256": _cache_prefix_sha256(cache_hashes),
    }

    _validate_dense_provenance(provenance, cache_hashes)

    provenance["cache_prefix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        _validate_dense_provenance(provenance, cache_hashes)


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "extra"])
def test_rejects_missing_duplicate_or_extra_checkpoint(
    tmp_path: Path,
    invalid: str,
) -> None:
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schedule = Path(config["schedule_manifest"])
    entries = _entries()
    if invalid == "missing":
        entries = []
    elif invalid == "duplicate":
        entries.append(dict(entries[0]))
    else:
        entries[0] = {**entries[0], "frame_index": 5}
    _write_schedule(schedule, entries)

    with pytest.raises(ValueError, match="non-empty|duplicate|outside"):
        run(config_path, tmp_path / "run", dependencies=_dependencies([]))

    assert not (tmp_path / "run").exists()


def test_rejects_extra_checkpoint_directory_and_cleans_staging(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    calls: list[str] = []
    dependencies = _dependencies(calls)

    def exporter(
        snapshot: object,
        checkpoint: object,
        destination: Path,
        context: object,
    ) -> object:
        result = dependencies.checkpoint_exporter(
            snapshot, checkpoint, destination, context
        )
        if int(getattr(checkpoint, "frame_index")) == 1:
            (destination.parent / "99999999-999").mkdir()
        return result

    corrupted = RunnerDependencies(
        dataset_factory=dependencies.dataset_factory,
        cache_loader_factory=dependencies.cache_loader_factory,
        runtime_factory=dependencies.runtime_factory,
        checkpoint_exporter=exporter,
        provenance_factory=dependencies.provenance_factory,
    )

    with pytest.raises(ValueError, match="checkpoint directories"):
        run(config, tmp_path / "run", dependencies=corrupted)

    assert not (tmp_path / "run").exists()
    assert list(tmp_path.glob(".run.staging-*")) == []


def test_output_reuse_is_rejected_without_touching_existing_data(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("owned\n", encoding="utf-8")
    calls: list[str] = []

    with pytest.raises(FileExistsError):
        run(config, output, dependencies=_dependencies(calls))

    assert marker.read_text(encoding="utf-8") == "owned\n"
    assert calls == []


def test_frame_failure_removes_all_partial_output(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    calls: list[str] = []
    dependencies = _dependencies(calls)

    class FailingCaches(_Caches):
        def load(self, frame_index: int, frame: object) -> tuple[tuple[object, ...], object]:
            if frame_index == 2:
                raise RuntimeError("cache failed")
            return super().load(frame_index, frame)

    failed = RunnerDependencies(
        dataset_factory=dependencies.dataset_factory,
        cache_loader_factory=lambda _config, _dataset: FailingCaches(calls),
        runtime_factory=dependencies.runtime_factory,
        checkpoint_exporter=dependencies.checkpoint_exporter,
        provenance_factory=dependencies.provenance_factory,
    )

    with pytest.raises(RuntimeError, match="cache failed"):
        run(config, tmp_path / "run", dependencies=failed)

    assert not (tmp_path / "run").exists()
    assert list(tmp_path.glob(".run.staging-*")) == []


def test_config_change_during_run_fails_closed_and_cleans_output(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    calls: list[str] = []
    dependencies = _dependencies(calls)

    class MutatingRuntime(_Runtime):
        def process_frame(
            self,
            frame: object,
            *,
            observations: tuple[object, ...],
            dense_semantics: object,
        ) -> SimpleNamespace:
            result = super().process_frame(
                frame,
                observations=observations,
                dense_semantics=dense_semantics,
            )
            if int(getattr(frame, "frame_id")) == 2:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                payload["max_age_frames"] = 99
                _write_json(config_path, payload)
            return result

    mutated = RunnerDependencies(
        dataset_factory=dependencies.dataset_factory,
        cache_loader_factory=dependencies.cache_loader_factory,
        runtime_factory=lambda _config, _caches: MutatingRuntime(calls),
        checkpoint_exporter=dependencies.checkpoint_exporter,
        provenance_factory=dependencies.provenance_factory,
    )

    with pytest.raises(ValueError, match="config changed during run"):
        run(config_path, tmp_path / "run", dependencies=mutated)

    assert not (tmp_path / "run").exists()
    assert list(tmp_path.glob(".run.staging-*")) == []


def test_schedule_swap_back_cannot_change_executed_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    schedule_path = tmp_path / "schedule.json"
    original_parser = runner_module._load_causal_checkpoints_bytes
    swapped_entries = [
        {
            "event_ids": ["swapped-0"],
            "frame_index": 0,
            "relative_timestamp_ns": 0,
            "roles": ["common_v2"],
            "timestamp_ns": 100,
        },
        {
            "event_ids": ["swapped-4"],
            "frame_index": 4,
            "relative_timestamp_ns": 40,
            "roles": ["common_v2"],
            "timestamp_ns": 140,
        },
    ]

    def swap_back_parser(
        data: bytes,
        path: Path,
        *,
        scene: str,
        frame_count: int,
    ) -> tuple[object, ...]:
        _write_schedule(schedule_path, swapped_entries)
        try:
            return original_parser(
                data,
                path,
                scene=scene,
                frame_count=frame_count,
            )
        finally:
            _write_schedule(schedule_path)

    monkeypatch.setattr(
        runner_module,
        "_load_causal_checkpoints_bytes",
        swap_back_parser,
    )

    manifest = run(config_path, tmp_path / "run", dependencies=_dependencies([]))

    assert manifest["official_schedule_frame_indices"] == [1, 3]
    assert manifest["captured_frame_indices"] == [1, 3]


def test_production_cli_accepts_only_config_and_output() -> None:
    parsed = parse_args(["--config", "scene.json", "--output", "run"])
    assert parsed.config == Path("scene.json")
    assert parsed.output == Path("run")

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--config",
                "scene.json",
                "--output",
                "run",
                "--scene",
                "office",
            ]
        )


def test_runner_imports_only_stage3_oviv2_runtime_path() -> None:
    source_path = Path("scripts/evaluation/run_oviv2_tesse_cd.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    lowered = "\n".join(sorted(imports)).lower()
    assert "route3" not in lowered
    assert "scannet200" not in lowered
    assert "stage4" not in lowered


def test_checked_scene_configs_are_same_frozen_stage3_algorithm() -> None:
    root = Path(__file__).resolve().parents[2]
    configs = {
        scene: json.loads(
            (root / f"configs/oviv2_tesse_cd_{scene}_v1.json").read_text(
                encoding="utf-8"
            )
        )
        for scene in ("apartment", "office")
    }

    assert {config["scene"] for config in configs.values()} == {
        "apartment",
        "office",
    }
    assert configs["apartment"]["frame_count"] == 1745
    assert configs["office"]["frame_count"] == 4346
    assert {algorithm_hash(config) for config in configs.values()} == {
        configs["apartment"]["algorithm_hash"]
    }
    expected_checkpoint_counts = {"apartment": 1652, "office": 4257}
    for scene, config in configs.items():
        assert config["method_id"] == "OVIV2"
        assert config["dataset"] == "TESSE-CD"
        assert config["source_stride"] == 1
        assert config["missing_observation_policy"] == "signed_depth"
        assert len(config["evaluation_checkpoint_frames"]) == (
            expected_checkpoint_counts[scene]
        )
        assert config["evaluation_checkpoint_frames"] == sorted(
            set(config["evaluation_checkpoint_frames"])
        )
        assert config["evaluation_checkpoint_frames_sha256"] == (
            "03dd2f35becf5bd5fcc4a9fdceb2e459d28e1eb2729eb1622090337dba02d9fa"
        )
        assert config["occlusion_target_manifest_sha256"] == (
            "0ce990a32af971be52f4b2a1d1bf8e8862cf18bb9d22362091411b4210b35361"
        )
        assert config["occlusion_target_manifest"].endswith(
            "/occlusion_v1_targets/20260722-stage3-fb97-a/manifest.json"
        )
        assert config["stage3_lineage_commit"] == (
            "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
        )
        assert config["schedule_manifest"] == (
            "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
        )
        assert config["input_manifest"] == (
            "configs/evaluation/manifests/oviv2_tesse_cd_cache.json"
        )
        assert config["frontend_cache_dir"] == (
            "/home/ww/oviovo_frontend_cache/tesse_cd_native_v1/"
            f"{scene}/gsa_detections_oviv2_tesse_{scene}_stage3_v1"
        )
        assert config["dense_cache_dir"] == (
            "/home/ww/oviovo_dense_cache/"
            f"tesse_cd_radseg_b_sam_s4_k4_native/{scene}"
        )
        assert config["dense_sample_stride"] == 4
        assert config["dense_top_k"] == 4
        serialized = json.dumps(config, sort_keys=True).lower()
        assert "ground_truth" not in serialized
        assert "route3" not in serialized
        assert "scannet200" not in serialized
        assert "stage4" not in serialized


def test_checked_schedule_loads_with_full_frozen_parameter_set() -> None:
    root = Path(__file__).resolve().parents[2]
    schedule = root / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"

    apartment = _load_causal_checkpoints(
        schedule,
        scene="apartment",
        frame_count=1745,
    )
    office = _load_causal_checkpoints(
        schedule,
        scene="office",
        frame_count=4346,
    )

    assert apartment
    assert office
    assert all(item.frame_index < 1745 for item in apartment)
    assert all(item.frame_index < 4346 for item in office)
