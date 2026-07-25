from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_config import temporal_config_from_json


V1_FILES = (
    Path("scripts/evaluation/run_oviv2_tesse_cd.py"),
    Path("configs/oviv2_tesse_cd_apartment_v1.json"),
    Path("configs/oviv2_tesse_cd_office_v1.json"),
)
V1_BYTES = {path: path.read_bytes() for path in V1_FILES}


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
    config: dict[str, object] = {
        "dataset": "TESSE-CD",
        "evaluation_checkpoint_frames": [2, 4],
        "evaluation_checkpoint_frames_sha256": "",
        "frame_count": 5,
        "method_id": "OVIV2",
        "missing_observation_policy": "signed_depth",
        "occlusion_target_manifest": str(target),
        "occlusion_target_manifest_sha256": _sha256(target),
        "protocol_id": "oviv2-tessecd-v2",
        "scene": "apartment",
        "schedule_manifest": str(schedule),
        "schema_version": 2,
        "temporal_readout": _temporal_readout(),
        "voxel_size_m": 0.05,
        "depth_max_m": 10.0,
    }
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


def _dependencies(module: object, *, fail_frame: int | None = None):
    holder: dict[str, object] = {}

    def caches(config: dict[str, object], dataset: object) -> _Caches:
        del dataset
        parsed = temporal_config_from_json({"temporal_readout": config["temporal_readout"]})
        return _Caches(parsed, {"stub": "sha256-bound"})

    def runtime(config: dict[str, object], cache: _Caches) -> _DualRuntime:
        assert "occlusion_target_manifest" not in config
        assert "evaluation_checkpoint_frames" not in config
        value = _DualRuntime(cache.temporal_config, fail_frame=fail_frame)
        holder["runtime"] = value
        return value

    return module.RunnerDependencies(
        dataset_factory=lambda config: _Dataset(),
        cache_loader_factory=caches,
        runtime_factory=runtime,
        provenance_factory=lambda: {"repository_commit": "a" * 40},
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
        assert checkpoint_manifest["metadata"]["frame_id"] == frame_index if "metadata" in checkpoint_manifest else checkpoint_manifest["consumed_through_frame"] == frame_index
    assert _tree_hashes(first) == _tree_hashes(second)
    assert {path: path.read_bytes() for path in V1_FILES} == V1_BYTES


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


def test_protocol_temporal_and_freeze_validation_precede_output_parent_creation(tmp_path: Path) -> None:
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
