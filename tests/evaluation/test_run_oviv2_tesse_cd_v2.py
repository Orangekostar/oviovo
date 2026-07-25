from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import gc
import shutil
import sys
from types import SimpleNamespace

import pytest

from scripts.evaluation.run_oviv2_tesse_cd import RunPublicationUncertainError
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_config import temporal_config_from_json


V1_FILES = (
    Path("scripts/evaluation/run_oviv2_tesse_cd.py"),
    Path("configs/oviv2_tesse_cd_apartment_v1.json"),
    Path("configs/oviv2_tesse_cd_office_v1.json"),
)
V1_BYTES = {path: path.read_bytes() for path in V1_FILES}

TEST_ENVIRONMENT = {
    "python": "3.fixture",
    "python_implementation": "CPython",
    "platform": "fixture-platform",
    "machine": "x86_64",
    "host": "fixture-host",
    "cuda": ["fixture-cuda"],
    "cuda_visible_devices": None,
    "gpu": ["fixture-gpu"],
    "libraries": {
        "numpy": "fixture-numpy",
        "open3d": "fixture-open3d",
        "scipy": "fixture-scipy",
        "torch": "fixture-torch",
        "pillow": "fixture-pillow",
    },
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _temporal_readout() -> dict[str, object]:
    return {
        "lifecycle": {
            "initial_log_odds": 0.0,
            "present_log_likelihood": 1.2,
            "absent_log_likelihood": -0.8,
            "log_odds_limit": 6.0,
            "decay_half_life_seconds": 30.0,
            "active_on_probability": 0.75,
            "dormant_off_probability": 0.35,
            "minimum_absent_streak": 2,
            "minimum_distinct_view_bins": 3,
            "visibility_depth_tolerance_m": 0.1,
            "minimum_visible_pixel_count": 10,
            "minimum_visible_fraction": 0.05,
            "view_bin_azimuth_count": 8,
            "view_bin_elevation_count": 4,
        },
        "association": {
            "visual_weight": 0.35,
            "semantic_weight": 0.25,
            "size_weight": 0.1,
            "motion_weight": 0.1,
            "geometry_weight": 0.2,
            "minimum_score": 0.55,
            "maximum_centroid_distance_m": 1.5,
            "semantic_conflict_probability": 0.9,
            "conflict_override_visual": 0.95,
            "conflict_override_geometry": 0.85,
        },
        "geometry": {
            "voxel_size_m": 0.05,
            "depth_max_m": 10.0,
            "maximum_entities": 16,
            "maximum_object_voxels": 32,
            "maximum_visibility_points_per_entity": 32,
            "background_block_count": 64,
            "background_mask_dilation_px": 2,
            "minimum_icp_points": 20,
            "minimum_icp_fitness": 0.5,
            "maximum_icp_rmse_m": 0.2,
            "maximum_motion_m": 2.0,
        },
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    schedule = tmp_path / "schedule.json"
    _write_json(
        schedule,
        {
            "dataset": "TESSE-CD",
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "method_predictions_used": False,
            "parameters": {"frame_indexing": "zero_based"},
            "scenes": {
                scene: {
                    "frame_count": 5,
                    "entries": [
                        {
                            "event_ids": ["event-official"],
                            "frame_index": 1,
                            "relative_timestamp_ns": 10,
                            "roles": ["official"],
                            "timestamp_ns": 110,
                        },
                        {
                            "event_ids": ["event-common"],
                            "frame_index": 3,
                            "relative_timestamp_ns": 30,
                            "roles": ["common_v2"],
                            "timestamp_ns": 130,
                        },
                    ],
                }
                for scene in ("apartment", "office")
            },
            "schema_version": 2,
        },
    )
    target = tmp_path / "targets.json"
    evaluation_frames = {"apartment": [2, 4], "office": [2, 4]}
    _write_json(
        target,
        {
            "dataset": "TESSE-CD",
            "manifest_id": "tesse_cd_occlusion_v1_targets",
            "metadata": {
                "scene_frame_indices": evaluation_frames,
                "episodes": [
                    {
                        "scene": scene,
                        "anchor": {"frame_index": 2, "relative_timestamp_ns": 20},
                        "checkpoints": [
                            {"frame_index": 4, "relative_timestamp_ns": 40}
                        ],
                    }
                    for scene in ("apartment", "office")
                ],
            },
            "schema_version": 1,
        },
    )
    config: dict[str, object] = json.loads(
        Path("configs/oviv2_tesse_cd_apartment_v2.json").read_text()
    )
    config.update(
        {
            "evaluation_checkpoint_frames": [2, 4],
            "evaluation_checkpoint_frames_sha256": "",
            "frame_count": 5,
            "occlusion_target_manifest": str(target),
            "occlusion_target_manifest_sha256": _sha256(target),
            "schedule_manifest": str(schedule),
            "temporal_readout": _temporal_readout(),
        }
    )
    return tmp_path / "runner.json", config


@dataclass
class _Caches:
    temporal_config: object
    bindings: dict[str, object]
    class_names: tuple[str, ...] = ("unknown", "chair")
    dense_provenance: object = None

    def load(self, frame_index: int, frame: object) -> tuple[tuple[()], None]:
        assert frame_index == frame.frame_id
        return (), None

    def assert_inputs_unchanged(self) -> None:
        return None


class _Dataset:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(frame_id=index, timestamp=(100 + index * 10) / 1e9)

    def timestamp_ns(self, index: int) -> int:
        return 100 + index * 10


class _DualRuntime:
    def __init__(self, temporal_config: object, *, fail_frame: int | None = None):
        self.fail_frame = fail_frame
        self.calls: list[int] = []
        self.temporal = SimpleNamespace(
            state=SimpleNamespace(
                scene_id="apartment",
                revision=0,
                last_frame_id=-1,
                last_timestamp=-1.0,
                entities=(),
                background=TemporalBackgroundVolume(temporal_config.geometry),
            )
        )

    def process_frame(self, frame: object, observations: object, dense_semantics: object) -> None:
        assert observations == () and dense_semantics is None
        if frame.frame_id == self.fail_frame:
            raise RuntimeError("injected runtime failure")
        self.calls.append(frame.frame_id)
        self.temporal.state = SimpleNamespace(
            scene_id="apartment",
            revision=frame.frame_id + 1,
            last_frame_id=frame.frame_id,
            last_timestamp=frame.timestamp,
            entities=(),
            background=self.temporal.state.background,
        )


def _dependencies(
    module: object,
    *,
    fail_frame: int | None = None,
    provenance: dict[str, object] | None = None,
    environment: dict[str, object] | None = None,
    cache_bindings: dict[str, object] | None = None,
):
    holder: dict[str, object] = {}

    def dataset(config: dict[str, object]) -> _Dataset:
        holder["dataset_config"] = dict(config)
        return _Dataset()

    def caches(config: dict[str, object], dataset: object) -> _Caches:
        del dataset
        holder["cache_config"] = dict(config)
        parsed = temporal_config_from_json({"temporal_readout": _temporal_readout()})
        return _Caches(parsed, cache_bindings or {"stub": "sha256-bound"})

    def runtime(config: dict[str, object], cache: _Caches) -> _DualRuntime:
        holder["runtime_config"] = dict(config)
        value = _DualRuntime(cache.temporal_config, fail_frame=fail_frame)
        holder["runtime"] = value
        return value

    return module.RunnerDependencies(
        dataset_factory=dataset,
        cache_loader_factory=caches,
        runtime_factory=runtime,
        provenance_factory=lambda: provenance or {"repository_commit": "a" * 40},
        environment_factory=lambda: environment or TEST_ENVIRONMENT,
    ), holder


def _materialize_config(module: object, tmp_path: Path) -> Path:
    path, config = _write_inputs(tmp_path)
    plan = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_occlusion_v1_checkpoint_frames",
        "evaluation_checkpoint_frames": {"apartment": [2, 4], "office": [2, 4]},
    }
    config["evaluation_checkpoint_frames_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    config["algorithm_hash"] = module.algorithm_hash(config)
    _write_json(path, config)
    return path


def _materialize_overlap_config(module: object, tmp_path: Path) -> Path:
    path = _materialize_config(module, tmp_path)
    config = json.loads(path.read_text())
    schedule_path = Path(config["schedule_manifest"])
    schedule = json.loads(schedule_path.read_text())
    for scene in ("apartment", "office"):
        official = schedule["scenes"][scene]["entries"][0]
        official.update(
            {"frame_index": 2, "relative_timestamp_ns": 20, "timestamp_ns": 120}
        )
    _write_json(schedule_path, schedule)
    return path


def test_five_frame_dual_readout_is_causal_role_aware_and_deterministic(tmp_path: Path) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    dependencies, holder = _dependencies(module)
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    manifest = module.run(config, first, dependencies=dependencies)
    dependencies2, _ = _dependencies(module)
    module.run(config, second, dependencies=dependencies2)

    assert manifest["protocol_id"] == "oviv2-tessecd-v2"
    assert holder["runtime"].calls == [0, 1, 2, 3, 4]
    assert manifest["scheduled_frame_indices"] == [1, 2, 3, 4]
    assert manifest["captured_frame_indices"] == [1, 2, 3, 4]
    records = {item["frame_index"]: item for item in manifest["checkpoints"]}
    assert records[1]["roles"] == ["official"]
    assert records[3]["roles"] == ["common_v2"]
    assert records[1]["format"] == "oviv2_temporal_current_checkpoint"
    assert records[3]["format"] == "oviv2_temporal_current_checkpoint"
    assert records[2]["roles"] == ["occlusion_v1"]
    assert records[4]["format"] == "oviv2_temporal_compact_checkpoint"
    for frame_index, record in records.items():
        assert record["consumed_through_frame"] == frame_index
        assert record["consumed_through_frame_exclusive"] == frame_index + 1
        checkpoint_manifest = json.loads(
            (first / record["artifact"]["path"] / "manifest.json").read_text()
        )
        if "metadata" in checkpoint_manifest:
            assert checkpoint_manifest["metadata"]["frame_id"] == frame_index
        else:
            assert checkpoint_manifest["consumed_through_frame"] == frame_index
    assert _tree_hashes(first) == _tree_hashes(second)
    for root in (first, second):
        published = json.loads((root / "run_manifest.json").read_text())
        for name in ("normalized_run_config",):
            record = published[name]
            path = root / record["path"]
            assert record == {
                "path": record["path"],
                "sha256": _sha256(path),
                "byte_count": path.stat().st_size,
            }
        expected_inventory = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in {"run_manifest.json", "execution_receipt.json"}
        )
        assert published["artifact_inventory"] == expected_inventory
    assert {path: path.read_bytes() for path in V1_FILES} == V1_BYTES


def test_overlap_checkpoint_publishes_full_and_compact_with_exact_index(
    tmp_path: Path,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_overlap_config(module, tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = module.run(config, first, dependencies=_dependencies(module)[0])
    second_manifest = module.run(config, second, dependencies=_dependencies(module)[0])
    overlap = next(item for item in first_manifest["checkpoints"] if item["frame_index"] == 2)
    assert set(overlap["artifacts"]) == {"temporal_current", "temporal_compact"}
    assert (first / overlap["artifacts"]["temporal_current"]["artifact"]["path"]).is_dir()
    assert (first / overlap["artifacts"]["temporal_compact"]["artifact"]["path"]).is_dir()
    index_path = first / "occlusion_checkpoint_index.json"
    index = json.loads(index_path.read_text())
    assert set(index) == {
        "schema_version",
        "format",
        "protocol_id",
        "dataset",
        "method_id",
        "scene",
        "algorithm_hash",
        "schedule",
        "target_manifest",
        "input_sha256",
        "code_commit",
        "source_bindings",
        "checkpoints",
    }
    assert index["format"] == "oviv2_temporal_compact_v1"
    assert [item["frame_index"] for item in index["checkpoints"]] == [2, 4]
    assert all(
        item["format"] == module.TEMPORAL_COMPACT_FORMAT
        and item["artifact"]["path"].endswith("/temporal_compact")
        for item in index["checkpoints"]
    )
    assert first_manifest["occlusion_checkpoint_index"] == {
        "path": "occlusion_checkpoint_index.json",
        "sha256": _sha256(index_path),
        "byte_count": index_path.stat().st_size,
    }
    assert "occlusion_checkpoint_index.json" in first_manifest["artifact_inventory"]
    assert first_manifest == second_manifest
    assert index_path.read_bytes() == (second / "occlusion_checkpoint_index.json").read_bytes()


def test_checkpoint_relative_timestamp_must_match_dataset_origin(
    tmp_path: Path,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_overlap_config(module, tmp_path)
    payload = json.loads(config.read_text())
    schedule_path = Path(payload["schedule_manifest"])
    schedule = json.loads(schedule_path.read_text())
    schedule["scenes"]["apartment"]["entries"][0]["relative_timestamp_ns"] = 21
    _write_json(schedule_path, schedule)
    output = tmp_path / "published" / "run"
    with pytest.raises(ValueError, match="relative timestamp"):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert not output.exists()
    assert list(output.parent.glob(".run.staging-*")) == []


def test_occlusion_index_binds_stable_source_authorities(tmp_path: Path) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_overlap_config(module, tmp_path)
    first = tmp_path / "first-authority"
    second = tmp_path / "second-authority"
    first_bindings = {"cache": {"sha256": "1" * 64}}
    second_bindings = {"cache": {"sha256": "2" * 64}}
    module.run(
        config,
        first,
        dependencies=_dependencies(
            module,
            provenance={"repository_commit": "a" * 40},
            cache_bindings=first_bindings,
        )[0],
    )
    module.run(
        config,
        second,
        dependencies=_dependencies(
            module,
            provenance={"repository_commit": "b" * 40},
            cache_bindings=second_bindings,
        )[0],
    )
    first_index = json.loads((first / "occlusion_checkpoint_index.json").read_text())
    second_index = json.loads((second / "occlusion_checkpoint_index.json").read_text())
    first_manifest = json.loads((first / "run_manifest.json").read_text())
    second_manifest = json.loads((second / "run_manifest.json").read_text())
    for index, manifest, bindings, commit in (
        (first_index, first_manifest, first_bindings, "a" * 40),
        (second_index, second_manifest, second_bindings, "b" * 40),
    ):
        assert index["input_sha256"] == manifest["input_sha256"]
        assert index["code_commit"] == manifest["code_commit"] == commit
        assert index["source_bindings"] == manifest["source_bindings"] == bindings
    assert (first / "occlusion_checkpoint_index.json").read_bytes() != (
        second / "occlusion_checkpoint_index.json"
    ).read_bytes()


@pytest.mark.parametrize("mutation", ["missing_overlap_compact", "mutated_index"])
def test_overlap_compact_and_index_are_revalidated_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_overlap_config(module, tmp_path)
    original_write = module._write_json

    def inject(path: Path, value: object) -> None:
        original_write(path, value)
        if path.name != "execution_receipt.json":
            return
        if mutation == "missing_overlap_compact":
            compact = next(
                path.parent.glob("checkpoints/00000002-*/temporal_compact"), None
            )
            if compact is not None:
                shutil.rmtree(compact)
        else:
            index = path.parent / "occlusion_checkpoint_index.json"
            if index.exists():
                index.write_text('{"mutated":true}\n', encoding="utf-8")

    monkeypatch.setattr(module, "_write_json", inject)
    output = tmp_path / "published" / "run"
    with pytest.raises((FileNotFoundError, ValueError)):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert not output.exists()


def test_checked_in_manifests_have_coverable_full_compact_overlap() -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    expected = {"apartment": 38, "office": 40}
    for scene, overlap_count in expected.items():
        config = json.loads(
            Path(f"configs/oviv2_tesse_cd_{scene}_v2.json").read_text()
        )
        schedule_path = module._resolve_path(config["schedule_manifest"])
        official = module._load_causal_checkpoints_bytes(
            schedule_path.read_bytes(),
            schedule_path,
            scene=scene,
            frame_count=config["frame_count"],
        )
        target_path = module._resolve_path(config["occlusion_target_manifest"])
        plan = module._checkpoint_plan_from_target_manifest(
            target_path.read_bytes(), target_path
        )
        target_frames = set(plan["evaluation_checkpoint_frames"][scene])
        full_frames = {
            item.frame_index
            for item in official
            if {"official", "common_v2"} & set(item.roles)
        }
        assert target_frames == set(config["evaluation_checkpoint_frames"])
        assert len(target_frames & full_frames) == overlap_count
        assert target_frames & full_frames


def test_runtime_factory_receives_only_explicit_algorithm_whitelist(tmp_path: Path) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    dependencies, holder = _dependencies(module)
    module.run(config, tmp_path / "run", dependencies=dependencies)
    received = holder["runtime_config"]
    assert set(received) == module._RUNTIME_CONFIG_KEYS
    forbidden = {
        "schedule_manifest",
        "evaluation_checkpoint_frames",
        "occlusion_target_manifest",
        "dataset_root",
        "input_manifest",
        "frontend_manifest",
        "dense_manifest",
        "method_id",
        "protocol_id",
        "algorithm_hash",
        "stage3_lineage_commit",
    }
    assert set(received).isdisjoint(forbidden)


def test_dataset_and_cache_factories_receive_only_exact_allowlist_views(
    tmp_path: Path,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    dependencies, holder = _dependencies(module)
    module.run(config, tmp_path / "run", dependencies=dependencies)
    assert set(holder["dataset_config"]) == module._DATASET_CONFIG_KEYS
    assert set(holder["cache_config"]) == module._CACHE_CONFIG_KEYS
    forbidden = {
        "occlusion_target_manifest",
        "occlusion_target_manifest_sha256",
        "evaluation_checkpoint_frames",
        "evaluation_checkpoint_frames_sha256",
        "schedule_manifest",
        "dataset_root",
        "protocol_id",
        "method_id",
        "algorithm_hash",
    }
    assert set(holder["cache_config"]).isdisjoint(forbidden)
    assert set(holder["dataset_config"]).isdisjoint(
        forbidden - {"schedule_manifest", "dataset_root"}
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("structure_wall_confidence", 2.0),
        ("fusion_entity_weight_scale", 2.0),
        ("dense_sample_stride", 0),
        ("frame_count", 1_000_000_000),
    ],
)
def test_invalid_runner_ranges_are_rejected_before_output_creation(
    tmp_path: Path, field: str, value: object
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config_path = _materialize_config(module, tmp_path)
    config = json.loads(config_path.read_text())
    config[field] = value
    config["algorithm_hash"] = module.algorithm_hash(config)
    _write_json(config_path, config)
    output = tmp_path / "must-not-exist" / "run"
    with pytest.raises(ValueError):
        module.run(config_path, output, dependencies=_dependencies(module)[0])
    assert not output.parent.exists()


def test_volatile_provenance_does_not_change_deterministic_manifest(
    tmp_path: Path,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = module.run(
        config,
        first,
        dependencies=_dependencies(
            module, provenance={"repository_commit": "a" * 40, "host": "one"}
        )[0],
    )
    second_manifest = module.run(
        config,
        second,
        dependencies=_dependencies(
            module, provenance={"repository_commit": "a" * 40, "host": "two"}
        )[0],
    )
    assert first_manifest == second_manifest
    assert (first / "run_manifest.json").read_bytes() == (
        second / "run_manifest.json"
    ).read_bytes()
    assert first_manifest["artifact_inventory"] == second_manifest["artifact_inventory"]
    assert (first / "execution_receipt.json").read_bytes() != (
        second / "execution_receipt.json"
    ).read_bytes()


def test_staging_directory_replacement_is_rejected_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    dependencies, _ = _dependencies(module)
    captured: dict[str, Path] = {}
    original_mkdtemp = module.tempfile.mkdtemp

    def recording_mkdtemp(*args, **kwargs) -> str:
        value = original_mkdtemp(*args, **kwargs)
        captured["staging"] = Path(value)
        return value

    class ReplacingRuntime(_DualRuntime):
        def process_frame(self, frame, observations, dense_semantics) -> None:
            if frame.frame_id == 0:
                staging = captured["staging"]
                staging.rename(staging.with_name(staging.name + "-stolen"))
                staging.mkdir()
            super().process_frame(frame, observations, dense_semantics)

    monkeypatch.setattr(module.tempfile, "mkdtemp", recording_mkdtemp)
    dependencies = module.RunnerDependencies(
        dataset_factory=dependencies.dataset_factory,
        cache_loader_factory=dependencies.cache_loader_factory,
        runtime_factory=lambda config, cache: ReplacingRuntime(cache.temporal_config),
        provenance_factory=dependencies.provenance_factory,
        environment_factory=dependencies.environment_factory,
    )
    output = tmp_path / "published" / "run"
    with pytest.raises(ValueError, match="staging root identity"):
        module.run(config, output, dependencies=dependencies)
    assert not output.exists()


def test_checkpoint_inventory_rejects_late_unexpected_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    original_write = module._write_json

    def inject(path: Path, value: object) -> None:
        original_write(path, value)
        if path.name == "normalized_run_config.json":
            checkpoint = next((path.parent / "checkpoints").glob("*/temporal_current"))
            (checkpoint / "unexpected.bin").write_bytes(b"unexpected")

    monkeypatch.setattr(module, "_write_json", inject)
    output = tmp_path / "published" / "run"
    with pytest.raises(ValueError, match="inventory"):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert not output.exists()


def test_publication_inventory_rejects_late_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    original_write = module._write_json

    def inject(path: Path, value: object) -> None:
        original_write(path, value)
        if path.name == "run_manifest.json":
            checkpoint = next((path.parent / "checkpoints").glob("*/temporal_current"))
            (checkpoint / "unexpected-empty-dir").mkdir()

    monkeypatch.setattr(module, "_write_json", inject)
    output = tmp_path / "published" / "run"
    with pytest.raises(ValueError, match="inventory"):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert not output.exists()


def test_publisher_boundary_empty_directory_is_publication_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    original_publish = module._publish_run

    def inject(staging: Path, destination: Path) -> None:
        checkpoint = next((staging / "checkpoints").glob("*/temporal_current"))
        (checkpoint / "late-unexpected-empty-dir").mkdir()
        original_publish(staging, destination)

    monkeypatch.setattr(module, "_publish_run", inject)
    output = tmp_path / "published" / "run"
    with pytest.raises(RunPublicationUncertainError):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert output.is_dir()
    assert next(output.glob("checkpoints/*/temporal_current/late-unexpected-empty-dir")).is_dir()


def test_publisher_boundary_content_change_is_publication_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    original_publish = module._publish_run

    def inject(staging: Path, destination: Path) -> None:
        (staging / "normalized_run_config.json").write_text(
            '{"changed":true}\n', encoding="utf-8"
        )
        original_publish(staging, destination)

    monkeypatch.setattr(module, "_publish_run", inject)
    output = tmp_path / "published" / "run"
    with pytest.raises(RunPublicationUncertainError):
        module.run(config, output, dependencies=_dependencies(module)[0])
    published = json.loads((output / "run_manifest.json").read_text())
    normalized = output / published["normalized_run_config"]["path"]
    assert published["normalized_run_config"]["sha256"] != _sha256(normalized)


def test_publisher_return_without_destination_is_uncertain_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    monkeypatch.setattr(module, "_publish_run", lambda staging, destination: None)
    output = tmp_path / "published" / "run"
    with pytest.raises(RunPublicationUncertainError):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert not output.exists()
    assert list(output.parent.glob(".run.staging-*")) == []


def test_environment_schema_and_production_authority_are_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    invalid = json.loads(json.dumps(TEST_ENVIRONMENT))
    invalid["libraries"]["extra"] = "1"
    with pytest.raises(ValueError):
        module._validate_environment(invalid)
    monkeypatch.setattr(
        module,
        "_production_provenance",
        lambda: {
            "hostname": "host",
            "platform": "platform",
            "machine": "machine",
            "cuda_visible_devices": "2",
            "torch_cuda_version": "12.4",
            "cudnn_version": 90100,
            "nvcc_version": ["Cuda compilation tools, release 12.4"],
            "gpu_inventory": ["2, NVIDIA H100, 550.54"],
            "library_versions": {
                "numpy": "2.0",
                "open3d": "0.18",
                "scipy": "1.14",
                "torch": "2.5",
                "pillow": "11.0",
            },
        },
    )
    environment = module._production_environment()
    assert set(environment) == module.V2_FREEZE_ENVIRONMENT_KEYS
    assert set(environment["libraries"]) == {
        "numpy",
        "open3d",
        "scipy",
        "torch",
        "pillow",
    }
    assert environment["gpu"] == ["2, NVIDIA H100, 550.54"]
    assert environment["cuda_visible_devices"] == "2"
    assert environment["cuda"] == [
        "torch_cuda=12.4",
        "cudnn=90100",
        "nvcc=Cuda compilation tools, release 12.4",
    ]


def test_production_environment_smoke_has_complete_current_schema() -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    environment = module._validate_environment(module._production_environment())
    assert set(environment) == module.V2_FREEZE_ENVIRONMENT_KEYS
    assert set(environment["libraries"]) == module._ENVIRONMENT_LIBRARY_KEYS
    assert environment["gpu"]
    assert environment["cuda"][0].startswith("torch_cuda=")
    assert environment["cuda"][1].startswith("cudnn=")
    assert all(item.startswith("nvcc=") for item in environment["cuda"][2:])


def test_compact_checkpoint_objects_are_released_between_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    original_from_snapshot = module.TemporalCompactCheckpoint.from_snapshot
    live = 0
    maximum_live = 0

    class Proxy:
        def __init__(self, value) -> None:
            nonlocal live, maximum_live
            self.value = value
            self.path = value.path
            live += 1
            maximum_live = max(maximum_live, live)

        def revalidate_source(self) -> None:
            self.value.revalidate_source()

        def __del__(self) -> None:
            nonlocal live
            live -= 1

    class Builder:
        def __init__(self, value) -> None:
            self.value = value

        def commit_new(self, *args, **kwargs):
            gc.collect()
            return Proxy(self.value.commit_new(*args, **kwargs))

    monkeypatch.setattr(
        module.TemporalCompactCheckpoint,
        "from_snapshot",
        lambda *args, **kwargs: Builder(original_from_snapshot(*args, **kwargs)),
    )
    module.run(config, tmp_path / "run", dependencies=_dependencies(module)[0])
    gc.collect()
    assert maximum_live == 1
    assert live == 0


@pytest.mark.parametrize("failure_kind", ["runtime", "checkpoint", "export"])
def test_known_failures_remove_entire_staging_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    dependencies, _ = _dependencies(module, fail_frame=2 if failure_kind == "runtime" else None)
    if failure_kind == "checkpoint":
        monkeypatch.setattr(
            module.TemporalCompactCheckpoint,
            "commit_new",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("checkpoint")),
        )
    if failure_kind == "export":
        monkeypatch.setattr(
            module,
            "publish_temporal_current_checkpoint",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("export")),
        )
    output = tmp_path / "parent" / "result"
    with pytest.raises(RuntimeError):
        module.run(config, output, dependencies=dependencies)
    assert not output.exists()
    assert list(output.parent.glob(".result.staging-*")) == []


def test_protocol_temporal_and_freeze_validation_precede_output_parent_creation(
    tmp_path: Path,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config_path = _materialize_config(module, tmp_path)
    config = json.loads(config_path.read_text())
    parent = tmp_path / "must-not-exist"
    for mutate in (
        lambda value: value.update({"protocol_id": "oviv2-tessecd-v1"}),
        lambda value: value.update({"block_count": True}),
        lambda value: value.pop("temporal_readout"),
        lambda value: value["temporal_readout"].update({"unknown": True}),
    ):
        changed = json.loads(json.dumps(config))
        mutate(changed)
        changed["algorithm_hash"] = module.algorithm_hash(changed)
        _write_json(config_path, changed)
        with pytest.raises((TypeError, ValueError)):
            module.run(config_path, parent / "result", dependencies=_dependencies(module)[0])
        assert not parent.exists()

    _write_json(config_path, config)
    with pytest.raises(ValueError, match="provided together"):
        module.run(config_path, parent / "result", freeze_manifest=tmp_path / "freeze.json")
    assert not parent.exists()

    freeze = tmp_path / "freeze.json"
    _write_json(
        freeze,
        {
            "schema_version": 1,
            "freeze_id": "oviv2-tessecd-v1",
            "status": "FROZEN",
            "method": "OVIV2",
            "dataset": "TESSE-CD",
        },
    )
    with pytest.raises(ValueError, match="oviv2-tessecd-v2"):
        module.run(
            config_path,
            parent / "result",
            freeze_manifest=freeze,
            run_slot="apartment_run1",
            dependencies=_dependencies(module)[0],
        )
    assert not parent.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("unknown", 1),
        lambda value: value.pop("block_count"),
        lambda value: value.__setitem__("block_count", 1.5),
        lambda value: value.__setitem__("block_count", "100"),
        lambda value: value.__setitem__("block_count", True),
        lambda value: value.__setitem__("voxel_size_m", "0.05"),
        lambda value: value.__setitem__("voxel_size_m", float("nan")),
        lambda value: value.__setitem__("voxel_size_m", float("inf")),
    ],
)
def test_v2_config_has_exact_keys_and_exact_scalar_types(
    tmp_path: Path, mutation
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config_path = _materialize_config(module, tmp_path)
    config = json.loads(config_path.read_text())
    mutation(config)
    try:
        config["algorithm_hash"] = module.algorithm_hash(config)
    except ValueError:
        pass
    config_path.write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    output_parent = tmp_path / "must-not-exist"
    with pytest.raises((TypeError, ValueError)):
        module.run(
            config_path,
            output_parent / "result",
            dependencies=_dependencies(module)[0],
        )
    assert not output_parent.exists()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _formal_fixture(
    module: object, tmp_path: Path
) -> tuple[Path, Path, Path, dict[str, str]]:
    apartment_path = _materialize_config(module, tmp_path)
    apartment = json.loads(apartment_path.read_text())
    shared_files: dict[str, Path] = {
        "input_manifest": tmp_path / "input-manifest.json",
        "schedule": Path(apartment["schedule_manifest"]),
        "occlusion_target_manifest": Path(apartment["occlusion_target_manifest"]),
    }
    _write_json(shared_files["input_manifest"], {"stub": "input"})
    apartment["input_manifest"] = str(shared_files["input_manifest"])
    scene_configs: dict[str, tuple[Path, dict[str, object]]] = {}
    scene_fields = (
        "export_manifest",
        "frontend_manifest",
        "dense_manifest",
        "vocabulary_json",
        "vocabulary_txt",
    )
    for scene in ("apartment", "office"):
        config = json.loads(json.dumps(apartment))
        config["scene"] = scene
        for field in scene_fields:
            source = tmp_path / scene / f"{field}.json"
            source.parent.mkdir(parents=True, exist_ok=True)
            if field == "frontend_manifest":
                _write_json(
                    source,
                    {
                        "feature_model_id": f"clip-sha256:{'f' * 64}",
                        "provenance_sha256": {"clip_model": "f" * 64},
                    },
                )
            elif field == "dense_manifest":
                _write_json(
                    source,
                    {
                        "provenance": {
                            "model_id": "fixture-dense-model",
                            "model_sha256": "d" * 64,
                        }
                    },
                )
            else:
                source.write_text(f"{scene}:{field}\n", encoding="utf-8")
            config[field] = str(source)
        config["algorithm_hash"] = module.algorithm_hash(config)
        path = apartment_path if scene == "apartment" else tmp_path / "office.json"
        _write_json(path, config)
        scene_configs[scene] = (path, config)
    assert (
        scene_configs["apartment"][1]["algorithm_hash"]
        == scene_configs["office"][1]["algorithm_hash"]
    )
    output = tmp_path / "formal" / "apartment" / "run1"
    roots = {
        f"{scene}_run{repeat}": str(
            (tmp_path / "formal" / scene / f"run{repeat}").resolve()
        )
        for scene in ("apartment", "office")
        for repeat in (1, 2)
    }
    scenes = {
        scene: {
            "frozen_config": _record(path),
            **{field: _record(Path(config[field])) for field in scene_fields},
        }
        for scene, (path, config) in scene_configs.items()
    }
    freeze = tmp_path / "freeze.json"
    algorithm_hash = scene_configs["apartment"][1]["algorithm_hash"]
    selected_config_sha256 = module._json_hash(scene_configs["apartment"][1])
    selection_artifact = tmp_path / "selection.json"
    _write_json(
        selection_artifact,
        {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_cd_v2_selection",
            "development_scene": "apartment",
            "selected_config_sha256": selected_config_sha256,
            "algorithm_hash": algorithm_hash,
        },
    )
    release_files = {
        "temporal_evaluator": tmp_path / "temporal-evaluator.py",
        "result_finalizer": tmp_path / "result-finalizer.py",
    }
    for role, path in release_files.items():
        path.write_text(f"{role}\n", encoding="utf-8")
    mapping = []
    runner_path = Path(module.__file__).resolve()
    for scene in ("apartment", "office"):
        config_path = scene_configs[scene][0].resolve()
        for repeat in (1, 2):
            slot = f"{scene}_run{repeat}"
            mapping.append(
                {
                    "scene": scene,
                    "run_slot": slot,
                    "output": roots[slot],
                    "argv": [
                        str(Path(sys.executable).resolve()),
                        str(runner_path),
                        "--config",
                        str(config_path),
                        "--output",
                        roots[slot],
                        "--freeze-manifest",
                        str(freeze.resolve()),
                        "--run-slot",
                        slot,
                    ],
                }
            )
    models = {
        branch: {
            scene: {
                "manifest_sha256": scenes[scene][f"{branch}_manifest"]["sha256"],
                "model_id": (
                    "fixture-dense-model"
                    if branch == "dense"
                    else f"clip-sha256:{'f' * 64}"
                ),
                "model_sha256": ("d" if branch == "dense" else "f") * 64,
            }
            for scene in ("apartment", "office")
        }
        for branch in ("frontend", "dense")
    }
    _write_json(
        freeze,
        {
            "schema_version": 1,
            "freeze_id": "oviv2-tessecd-v2",
            "status": "FROZEN",
            "method": "OVIV2",
            "dataset": "TESSE-CD",
            "repository": {
                "clean": True,
                "commit": "a" * 40,
                "parents": ["c" * 40],
                "tree": "b" * 40,
                "commit_time_utc": "2026-07-25T00:00:00+00:00",
                "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
                "stage3_is_ancestor": True,
            },
            "algorithm": {
                "sha256": algorithm_hash,
                "normalized_config": module.algorithm_config(
                    scene_configs["apartment"][1]
                ),
            },
            "selection": {
                "artifact": _record(selection_artifact),
                "development_scene": "apartment",
                "selected_config_sha256": selected_config_sha256,
                "selected_algorithm_hash": algorithm_hash,
            },
            "scenes": scenes,
            "shared_bindings": {
                role: _record(path) for role, path in shared_files.items()
            },
            "environment": TEST_ENVIRONMENT,
            "commands": {
                "cwd": str(Path(module.REPO_ROOT).resolve()),
                "python": str(Path(sys.executable).resolve()),
                "mapping": mapping,
            },
            "models": models,
            "release_bindings": {
                role: _record(path) for role, path in release_files.items()
            },
            "office_pre_freeze_audit": {
                "selection_scene": "apartment",
                "metric_sources_found": [],
                "office_outputs_read": False,
            },
            "output_roots": roots,
        },
    )
    return apartment_path, freeze, output, roots


def _clean_repository_state() -> dict[str, str]:
    return {
        "repository_commit": "a" * 40,
        "repository_tree": "b" * 40,
        "dirty_state_digest": hashlib.sha256(b"").hexdigest(),
    }


def _patch_formal_authorities(module, monkeypatch, payload) -> None:
    monkeypatch.setattr(module, "_repository_provenance", _clean_repository_state)
    monkeypatch.setattr(
        module,
        "V2_RELEASE_EXPECTED_PATHS",
        {
            role: Path(record["path"])
            for role, record in payload["release_bindings"].items()
        },
        raising=False,
    )


def test_task14_can_import_the_complete_v2_freeze_contract() -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    assert module.V2_FREEZE_TOP_KEYS == {
        "schema_version",
        "freeze_id",
        "status",
        "method",
        "dataset",
        "repository",
        "algorithm",
        "selection",
        "scenes",
        "shared_bindings",
        "environment",
        "commands",
        "models",
        "release_bindings",
        "output_roots",
        "office_pre_freeze_audit",
    }
    assert callable(module.load_v2_frozen_run_context)


def test_complete_v2_formal_freeze_runs_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config, freeze, output, _ = _formal_fixture(module, tmp_path)
    freeze_payload = json.loads(freeze.read_text())
    _patch_formal_authorities(module, monkeypatch, freeze_payload)
    manifest = module.run(
        config,
        output,
        freeze_manifest=freeze,
        run_slot="apartment_run1",
        dependencies=_dependencies(module)[0],
    )
    assert manifest["frozen_run_identity"]["freeze_id"] == "oviv2-tessecd-v2"
    assert manifest["frozen_run_identity"]["formal_evidence_sha256"] == module._json_hash(
        freeze_payload
    )


@pytest.mark.parametrize(
    "scope,role,mutation",
    [
        ("apartment", "frozen_config", "delete"),
        ("apartment", "export_manifest", "delete"),
        ("apartment", "frontend_manifest", "path"),
        ("apartment", "dense_manifest", "hash"),
        ("apartment", "vocabulary_json", "delete"),
        ("apartment", "vocabulary_txt", "path"),
        ("office", "frozen_config", "hash"),
        ("office", "export_manifest", "path"),
        ("office", "frontend_manifest", "delete"),
        ("office", "dense_manifest", "path"),
        ("office", "vocabulary_json", "hash"),
        ("office", "vocabulary_txt", "delete"),
        ("shared_bindings", "input_manifest", "delete"),
        ("shared_bindings", "schedule", "path"),
        ("shared_bindings", "occlusion_target_manifest", "hash"),
    ],
)
def test_formal_freeze_requires_every_exact_input_binding_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    role: str,
    mutation: str,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config, freeze, output, _ = _formal_fixture(module, tmp_path)
    payload = json.loads(freeze.read_text())
    bindings = payload["scenes"][scope] if scope in {"apartment", "office"} else payload[scope]
    if mutation == "delete":
        del bindings[role]
    elif mutation == "path":
        wrong = tmp_path / f"wrong-{scope}-{role}"
        wrong.write_bytes(Path(bindings[role]["path"]).read_bytes())
        bindings[role]["path"] = str(wrong.resolve())
    else:
        bindings[role]["sha256"] = "0" * 64
    _write_json(freeze, payload)
    _patch_formal_authorities(module, monkeypatch, payload)
    with pytest.raises(ValueError):
        module.run(
            config,
            output,
            freeze_manifest=freeze,
            run_slot="apartment_run1",
            dependencies=_dependencies(module)[0],
        )
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("unknown", {}),
        lambda value: value["repository"].pop("parents"),
        lambda value: value["algorithm"].__setitem__("unknown", 1),
        lambda value: value["output_roots"].pop("office_run2"),
    ],
)
def test_formal_freeze_top_level_objects_are_exact_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config, freeze, output, _ = _formal_fixture(module, tmp_path)
    payload = json.loads(freeze.read_text())
    mutation(payload)
    _write_json(freeze, payload)
    _patch_formal_authorities(module, monkeypatch, payload)
    with pytest.raises(ValueError):
        module.run(
            config,
            output,
            freeze_manifest=freeze,
            run_slot="apartment_run1",
            dependencies=_dependencies(module)[0],
        )
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "selection_missing",
        "selection_path",
        "selection_hash",
        "selection_payload_hash",
        "selection_payload_identity",
        "selection_algorithm_hash",
        "environment_missing",
        "environment_malformed",
        "environment_mismatch",
        "environment_gpu_mismatch",
        "environment_cuda_mismatch",
        "commands_empty",
        "models_missing",
        "models_empty",
        "models_manifest",
        "models_forged",
        "dense_model_forged",
        "commands_python",
        "release_replacement",
        "release_missing",
        "office_audit",
    ],
)
def test_complete_formal_evidence_is_strict_and_bound_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config, freeze, output, _ = _formal_fixture(module, tmp_path)
    payload = json.loads(freeze.read_text())
    if mutation == "selection_missing":
        del payload["selection"]
    elif mutation == "selection_path":
        payload["selection"]["artifact"]["path"] = str(
            (tmp_path / "wrong-selection.json").resolve()
        )
    elif mutation == "selection_hash":
        payload["selection"]["artifact"]["sha256"] = "0" * 64
    elif mutation == "selection_payload_hash":
        artifact = Path(payload["selection"]["artifact"]["path"])
        selection = json.loads(artifact.read_text())
        selection["selected_config_sha256"] = "0" * 64
        _write_json(artifact, selection)
        payload["selection"]["artifact"] = _record(artifact)
    elif mutation == "selection_payload_identity":
        artifact = Path(payload["selection"]["artifact"]["path"])
        selection = json.loads(artifact.read_text())
        selection["manifest_id"] = "other_selection"
        _write_json(artifact, selection)
        payload["selection"]["artifact"] = _record(artifact)
    elif mutation == "selection_algorithm_hash":
        payload["selection"]["selected_algorithm_hash"] = "0" * 64
    elif mutation == "environment_missing":
        del payload["environment"]
    elif mutation == "environment_malformed":
        payload["environment"]["cuda"] = []
    elif mutation == "environment_mismatch":
        payload["environment"]["libraries"]["numpy"] = "999"
    elif mutation == "environment_gpu_mismatch":
        payload["environment"]["gpu"] = ["other-gpu"]
    elif mutation == "environment_cuda_mismatch":
        payload["environment"]["cuda"] = ["other-cuda"]
    elif mutation == "commands_empty":
        payload["commands"]["mapping"] = []
    elif mutation == "models_missing":
        del payload["models"]["frontend"]["office"]
    elif mutation == "models_empty":
        payload["models"]["dense"]["apartment"] = {}
    elif mutation == "models_manifest":
        payload["models"]["frontend"]["apartment"]["manifest_sha256"] = "0" * 64
    elif mutation == "models_forged":
        payload["models"]["frontend"]["apartment"]["model_id"] = "forged"
        payload["models"]["frontend"]["apartment"]["model_sha256"] = "a" * 64
    elif mutation == "dense_model_forged":
        payload["models"]["dense"]["office"]["model_id"] = "forged"
        payload["models"]["dense"]["office"]["model_sha256"] = "a" * 64
    elif mutation == "commands_python":
        payload["commands"]["python"] = "/invented/python"
        for command in payload["commands"]["mapping"]:
            command["argv"][0] = "/invented/python"
    elif mutation == "release_replacement":
        replacement = tmp_path / "unrelated-release.py"
        replacement.write_text("unrelated\n", encoding="utf-8")
        payload["release_bindings"]["temporal_evaluator"] = _record(replacement)
    elif mutation == "release_missing":
        del payload["release_bindings"]["result_finalizer"]
    else:
        payload["office_pre_freeze_audit"]["office_outputs_read"] = True
    _write_json(freeze, payload)
    expected_release_paths = {
        role: Path(record["path"])
        for role, record in json.loads(freeze.read_text())["release_bindings"].items()
    }
    if mutation == "release_replacement":
        expected_release_paths["temporal_evaluator"] = tmp_path / "temporal-evaluator.py"
    monkeypatch.setattr(module, "_repository_provenance", _clean_repository_state)
    monkeypatch.setattr(
        module, "V2_RELEASE_EXPECTED_PATHS", expected_release_paths, raising=False
    )
    with pytest.raises((FileNotFoundError, TypeError, ValueError)):
        module.run(
            config,
            output,
            freeze_manifest=freeze,
            run_slot="apartment_run1",
            dependencies=_dependencies(module)[0],
        )
    assert not output.parent.exists()


@pytest.mark.parametrize("failure", ["dataset", "cache"])
def test_dataset_and_cache_failures_clean_staging(
    tmp_path: Path, failure: str
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    dependencies, _ = _dependencies(module)
    if failure == "dataset":
        dependencies = module.RunnerDependencies(
            dataset_factory=lambda value: (_ for _ in ()).throw(
                RuntimeError("dataset")
            ),
            cache_loader_factory=dependencies.cache_loader_factory,
            runtime_factory=dependencies.runtime_factory,
            provenance_factory=dependencies.provenance_factory,
            environment_factory=dependencies.environment_factory,
        )
    else:
        dependencies = module.RunnerDependencies(
            dataset_factory=dependencies.dataset_factory,
            cache_loader_factory=lambda config, dataset: (
                _ for _ in ()
            ).throw(RuntimeError("cache")),
            runtime_factory=dependencies.runtime_factory,
            provenance_factory=dependencies.provenance_factory,
            environment_factory=dependencies.environment_factory,
        )
    output = tmp_path / "published" / "result"
    with pytest.raises(RuntimeError):
        module.run(config, output, dependencies=dependencies)
    assert not output.exists()
    assert list(output.parent.glob(".result.staging-*")) == []


@pytest.mark.parametrize("failure", ["dataset_frame", "cache_frame"])
def test_frame_loading_failures_clean_staging(tmp_path: Path, failure: str) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    dependencies, _ = _dependencies(module)

    class FailingDataset(_Dataset):
        def __getitem__(self, index: int) -> SimpleNamespace:
            if index == 2:
                raise RuntimeError("dataset frame")
            return super().__getitem__(index)

    class FailingCache(_Caches):
        def load(self, frame_index: int, frame: object):
            if frame_index == 2:
                raise RuntimeError("cache frame")
            return super().load(frame_index, frame)

    if failure == "dataset_frame":
        dependencies = module.RunnerDependencies(
            dataset_factory=lambda value: FailingDataset(),
            cache_loader_factory=dependencies.cache_loader_factory,
            runtime_factory=dependencies.runtime_factory,
            provenance_factory=dependencies.provenance_factory,
            environment_factory=dependencies.environment_factory,
        )
    else:
        original_cache_factory = dependencies.cache_loader_factory

        def failing_cache_factory(config: dict[str, object], dataset: object):
            cache = original_cache_factory(config, dataset)
            return FailingCache(
                cache.temporal_config,
                cache.bindings,
                cache.class_names,
                cache.dense_provenance,
            )

        dependencies = module.RunnerDependencies(
            dataset_factory=dependencies.dataset_factory,
            cache_loader_factory=failing_cache_factory,
            runtime_factory=dependencies.runtime_factory,
            provenance_factory=dependencies.provenance_factory,
            environment_factory=dependencies.environment_factory,
        )
    output = tmp_path / "published" / "result"
    with pytest.raises(RuntimeError):
        module.run(config, output, dependencies=dependencies)
    assert not output.exists()
    assert list(output.parent.glob(".result.staging-*")) == []


def test_manifest_serialization_and_pre_rename_failure_clean_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    output = tmp_path / "published" / "result"
    original = Path.write_bytes

    def fail_manifest(path: Path, value: bytes) -> int:
        if path.name == "run_manifest.json":
            raise RuntimeError("manifest serialization")
        return original(path, value)

    monkeypatch.setattr(Path, "write_bytes", fail_manifest)
    with pytest.raises(RuntimeError, match="manifest serialization"):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert not output.exists()
    assert list(output.parent.glob(".result.staging-*")) == []

    monkeypatch.setattr(Path, "write_bytes", original)
    monkeypatch.setattr(
        module,
        "_publish_run",
        lambda staging, destination: (_ for _ in ()).throw(
            RuntimeError("pre-rename")
        ),
    )
    with pytest.raises(RuntimeError, match="pre-rename"):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert not output.exists()
    assert list(output.parent.glob(".result.staging-*")) == []


def test_existing_and_symlink_outputs_are_rejected(tmp_path: Path) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        module.run(config, existing, dependencies=_dependencies(module)[0])
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        module.run(config, link, dependencies=_dependencies(module)[0])


def test_post_rename_uncertain_failure_retains_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    config = _materialize_config(module, tmp_path)
    output = tmp_path / "published" / "result"

    def uncertain(staging: Path, destination: Path) -> None:
        staging.rename(destination)
        raise RunPublicationUncertainError("fsync uncertain")

    monkeypatch.setattr(module, "_publish_run", uncertain)
    with pytest.raises(RunPublicationUncertainError):
        module.run(config, output, dependencies=_dependencies(module)[0])
    assert output.is_dir()
    assert (output / "run_manifest.json").is_file()


def test_cli_requires_config_output_and_pairs_optional_formal_arguments() -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    with pytest.raises(SystemExit):
        module.parse_args([])
    args = module.parse_args(["--config", "config.json", "--output", "output"])
    assert args.freeze_manifest is None and args.run_slot is None
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--config",
                "config.json",
                "--output",
                "output",
                "--freeze-manifest",
                "freeze.json",
            ]
        )


def test_checked_in_v2_configs_share_algorithm_and_preserve_v1_bindings() -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as module

    configs = []
    for scene in ("apartment", "office"):
        v1 = json.loads(Path(f"configs/oviv2_tesse_cd_{scene}_v1.json").read_text())
        v2 = json.loads(Path(f"configs/oviv2_tesse_cd_{scene}_v2.json").read_text())
        assert v2["protocol_id"] == "oviv2-tessecd-v2"
        assert len(v2) == 85
        assert set(v2) == module._V2_CONFIG_KEYS
        assert v2["schema_version"] == 2
        assert v2["method_id"] == "OVIV2"
        assert v2["algorithm_hash"] == module.algorithm_hash(v2)
        temporal_config_from_json({"temporal_readout": v2["temporal_readout"]})
        assert v2["temporal_readout"]["geometry"]["voxel_size_m"] == v1["voxel_size_m"]
        assert v2["temporal_readout"]["geometry"]["depth_max_m"] == v1["depth_max_m"]
        for key in module.SCENE_CONFIG_FIELDS - {"algorithm_hash"}:
            if key in v1:
                assert v2[key] == v1[key]
        configs.append(v2)
    assert module.algorithm_config(configs[0]) == module.algorithm_config(configs[1])
    assert configs[0]["algorithm_hash"] == configs[1]["algorithm_hash"]
    assert {path: path.read_bytes() for path in V1_FILES} == V1_BYTES
