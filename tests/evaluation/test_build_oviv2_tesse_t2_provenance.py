from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import scripts.evaluation.build_oviv2_tesse_t2_provenance as builder
from scripts.evaluation.finalize_tesse_t2 import _build_result_payload


CLEAN_DIGEST = hashlib.sha256(b"").hexdigest()
STAGE3_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
ADAPTER_COMMIT = "a" * 40
RUN_ID = "oviv2-tessecd-v1"


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_json(path: Path, payload: object) -> Path:
    return _write(
        path,
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _relative_record(path: Path, base: Path) -> dict[str, object]:
    record = _record(path)
    record["path"] = str(path.relative_to(base))
    return record


class Fixture:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = tmp_path / "package"
        self.freeze_root = self.root / "frozen"
        self.dataset_manifest = _write_json(
            self.root / "dataset/tesse_cd.json",
            {"schema_version": 1, "dataset": "TESSE-CD"},
        )
        self.input_manifest = _write_json(
            self.root / "dataset/oviv2_tesse_cd_cache.json",
            {"schema_version": 1, "dataset": "TESSE-CD", "method": "OVIV2"},
        )
        self.schedule = _write_json(
            self.root / "dataset/tesse_cd_causal_schedule_v2.json",
            {"schema_version": 2, "dataset": "TESSE-CD", "status": "FROZEN"},
        )
        self.camera = _write_json(
            self.root / "dataset/camera.json",
            {"width": 720, "height": 480},
        )
        self.configs: dict[str, Path] = {}
        self.frozen_sources: dict[str, dict[str, Path]] = {}
        for scene in ("apartment", "office"):
            self.configs[scene] = _write_json(
                self.root / f"configs/{scene}.json",
                {
                    "schema_version": 1,
                    "method_id": "OVIV2",
                    "dataset": "TESSE-CD",
                    "scene": scene,
                    "stage3_lineage_commit": STAGE3_COMMIT,
                    "algorithm_hash": "b" * 64,
                    "missing_observation_policy": "signed_depth",
                },
            )
            self.frozen_sources[scene] = {
                "export_manifest": _write_json(
                    self.root / f"inputs/{scene}/export_manifest.json",
                    {"scene": scene, "combined_output_sha256": "c" * 64},
                ),
                "trajectory": _write(
                    self.root / f"inputs/{scene}/trajectory.txt", b"trajectory\n"
                ),
                "timestamps": _write(
                    self.root / f"inputs/{scene}/timestamps.csv", b"timestamps\n"
                ),
                "vocabulary_json": _write_json(
                    self.root / f"inputs/{scene}/vocabulary.json", {"classes": []}
                ),
                "vocabulary_txt": _write(
                    self.root / f"inputs/{scene}/vocabulary.txt", b"unknown\n"
                ),
                "frontend_manifest": _write_json(
                    self.root / f"inputs/{scene}/frontend_manifest.json",
                    {"scene": scene, "cache_prefix_sha256": "d" * 64},
                ),
                "dense_manifest": _write_json(
                    self.root / f"inputs/{scene}/dense_manifest.json",
                    {"scene": scene, "cache_prefix_sha256": "e" * 64},
                ),
                "source_database": _write(
                    self.root / f"inputs/{scene}/source.db", b"database\n"
                ),
            }

        self.weights: dict[str, Path] = {}
        roles = (
            "frontend.clip_model",
            "frontend.mobile_sam_model",
            "frontend.yolo_clip_model",
            "frontend.yolo_model",
            "dense.auxiliary_model",
            "dense.language_model",
            "dense.model",
        )
        for index, role in enumerate(roles):
            self.weights[role] = _write(
                self.root / f"weights/{role}.bin", f"weight-{index}\n".encode()
            )

        frontend_hashes = {
            "clip_model": _sha256(self.weights["frontend.clip_model"]),
            "mobile_sam_model": _sha256(
                self.weights["frontend.mobile_sam_model"]
            ),
            "yolo_clip_model": _sha256(self.weights["frontend.yolo_clip_model"]),
            "yolo_model": _sha256(self.weights["frontend.yolo_model"]),
            "script": "1" * 64,
            "hydra_config": "2" * 64,
            "dataset_config": "3" * 64,
        }
        dense_hashes = {
            "auxiliary_model_sha256": _sha256(
                self.weights["dense.auxiliary_model"]
            ),
            "language_model_sha256": _sha256(
                self.weights["dense.language_model"]
            ),
            "model_sha256": _sha256(self.weights["dense.model"]),
        }

        self.mapping_roots: dict[str, Path] = {}
        self.official_roots: dict[str, Path] = {}
        for scene in ("apartment", "office"):
            for repeat in (1, 2):
                key = f"{scene}_run{repeat}"
                mapping = self.freeze_root / scene / f"run{repeat}"
                self.mapping_roots[key] = mapping
                self._write_mapping_run(key, scene, mapping)
                official = self.root / "official" / scene / f"run{repeat}"
                self.official_roots[key] = official
                self._write_official_run(key, scene, mapping, official)

        finalizer = _write(
            self.root / "scripts/finalize_tesse_t2.py", b"# finalizer fixture\n"
        )
        common_finalizer = _write(
            self.root / "scripts/finalize_tesse_common_v2.py",
            b"# common finalizer fixture\n",
        )
        evaluator = _write(
            self.root / "scripts/evaluate_tesse_cd_common_v2.py",
            b"# evaluator fixture\n",
        )
        aliases = _write(self.root / "dataset/aliases.yaml", b"aliases: {}\n")
        target_arrays = _write(self.root / "dataset/targets.npz", b"targets\n")
        target_manifest = _write_json(
            self.root / "dataset/target_manifest.json",
            {"schema_version": 1, "dataset": "TESSE-CD"},
        )
        selection_path = _write_json(
            self.root / "tuning/selection.json",
            {"status": "PASS", "selected_config": _sha256(self.configs["apartment"])},
        )
        prepared_manifest = _write_json(
            self.freeze_root / "freeze_manifest.prepared.json",
            {"status": "PREPARED", "freeze_id": RUN_ID},
        )
        self.freeze_manifest = _write_json(
            self.freeze_root / "freeze_manifest.json",
            {
                "schema_version": 1,
                "freeze_id": RUN_ID,
                "status": "FROZEN",
                "method": "OVIV2",
                "dataset": "TESSE-CD",
                "repository": {
                    "commit": ADAPTER_COMMIT,
                    "parents": ["9" * 40],
                    "tree": "8" * 40,
                    "commit_time_utc": "2026-07-22T00:00:00+00:00",
                    "clean": True,
                    "stage3_lineage_commit": STAGE3_COMMIT,
                    "stage3_is_ancestor": True,
                },
                "algorithm": {"sha256": "b" * 64},
                "selection": {
                    **_record(selection_path),
                    "candidate_count": 18,
                    "selected_config_sha256": _sha256(
                        self.configs["apartment"]
                    ),
                    "selected_parameters": {
                        "visibility_depth_tolerance_m": 0.1,
                        "absence_negative_support": 1.0,
                        "ownership_min_net_support": 0.5,
                    },
                    "selection_rule": [
                        "maximize_current_miou",
                        "minimize_ghost_rate",
                        "maximize_background_f5_cm",
                        "minimize_recovery_frames",
                        "minimize_config_sha256",
                    ],
                    "candidates": [{"candidate_id": index} for index in range(18)],
                },
                "scenes": {
                    scene: {
                        "source_config": _record(self.configs[scene]),
                        "frozen_config": _record(self.configs[scene]),
                        "rgbd": {
                            "root": str((self.root / f"inputs/{scene}").resolve()),
                            "frame_count": 1,
                            "export_manifest": _record(
                                self.frozen_sources[scene]["export_manifest"]
                            ),
                            "trajectory": _record(
                                self.frozen_sources[scene]["trajectory"]
                            ),
                            "timestamps": _record(
                                self.frozen_sources[scene]["timestamps"]
                            ),
                            "camera": _record(self.camera),
                            "combined_output_sha256": "c" * 64,
                            "file_hash_count": 2,
                            "source_database": _record(
                                self.frozen_sources[scene]["source_database"]
                            ),
                        },
                        "vocabulary": {
                            "json": _record(
                                self.frozen_sources[scene]["vocabulary_json"]
                            ),
                            "txt": _record(
                                self.frozen_sources[scene]["vocabulary_txt"]
                            ),
                        },
                        "cache": {
                            "frontend_manifest": _record(
                                self.frozen_sources[scene]["frontend_manifest"]
                            ),
                            "frontend_algorithm_sha256": "b" * 64,
                            "frontend_cache_prefix_sha256": "d" * 64,
                            "dense_manifest": _record(
                                self.frozen_sources[scene]["dense_manifest"]
                            ),
                            "dense_cache_prefix_sha256": "e" * 64,
                        },
                    }
                    for scene in ("apartment", "office")
                },
                "shared_bindings": {
                    "input_manifest": _record(self.input_manifest),
                    "source_manifest": _record(self.dataset_manifest),
                    "schedule": _record(self.schedule),
                    "camera": _record(self.camera),
                    "common_target_manifest": _record(target_manifest),
                    "common_target_arrays": _record(target_arrays),
                    "alias_map": _record(aliases),
                    "evaluator": _record(evaluator),
                    "finalizers": {
                        "common_v2": _record(common_finalizer),
                        "official_t2": _record(finalizer),
                    },
                },
                "models": {
                    "frontend": {
                        scene: {
                            "feature_model_id": "clip:test",
                            "provenance_sha256": frontend_hashes,
                        }
                        for scene in ("apartment", "office")
                    },
                    "dense": {
                        scene: {
                            **dense_hashes,
                            "backend": "radseg",
                            "source_commit": "4" * 40,
                            "radio_commit": "3" * 40,
                            "model_id": "radseg:test",
                            "language_model_id": "clip:test",
                            "language_model_revision": "fixture",
                            "vocabulary_sha256": "2" * 64,
                            "prompt_sha256": "1" * 64,
                            "inference_config_sha256": "0" * 64,
                        }
                        for scene in ("apartment", "office")
                    },
                },
                "environment": {"python": "3.12.fixture"},
                "commands": {
                    "cwd": str(self.root),
                    "python": "/usr/bin/python3",
                    "mapping": [
                        f"python run.py --config {self.configs[scene]} --output "
                        f"{self.mapping_roots[f'{scene}_run{repeat}']}"
                        for scene in ("apartment", "office")
                        for repeat in (1, 2)
                    ],
                },
                "output_roots": {
                    key: str(path.resolve())
                    for key, path in sorted(self.mapping_roots.items())
                },
                "office_pre_freeze_audit": {
                    "metric_sources_found": [],
                    "output_root_was_empty": False,
                    "output_root_had_only_preparation": True,
                    "scope": {
                        "office_config_recursive": True,
                        "output_root": str(self.freeze_root.resolve()),
                        "selection_scene": "apartment",
                    },
                },
                "preparation": {
                    "manifest": _record(prepared_manifest),
                    "repository": {
                        "commit": "7" * 40,
                        "parents": ["6" * 40],
                        "tree": "5" * 40,
                        "commit_time_utc": "2026-07-21T00:00:00+00:00",
                        "clean": True,
                        "stage3_lineage_commit": STAGE3_COMMIT,
                        "stage3_is_ancestor": True,
                    },
                },
            },
        )
        frozen_payload = json.loads(self.freeze_manifest.read_text(encoding="utf-8"))
        prepared_payload = dict(frozen_payload)
        prepared_payload.pop("preparation")
        prepared_payload["status"] = "PREPARED"
        prepared_payload["repository"] = frozen_payload["preparation"]["repository"]
        _write_json(prepared_manifest, prepared_payload)
        frozen_payload["preparation"]["manifest"] = _record(prepared_manifest)
        _write_json(self.freeze_manifest, frozen_payload)

        monkeypatch.setattr(
            builder,
            "_repository_state",
            lambda: {
                "commit": ADAPTER_COMMIT,
                "dirty_state_digest": CLEAN_DIGEST,
            },
        )
        monkeypatch.setattr(
            builder,
            "validate_khronos_run_status",
            lambda path, scene: json.loads(path.read_text(encoding="utf-8")),
        )
        monkeypatch.setattr(
            builder,
            "validate_temporal_bridge_manifest",
            lambda path: json.loads(path.read_text(encoding="utf-8")),
        )

    def _write_mapping_run(self, key: str, scene: str, root: Path) -> None:
        snapshot = _write(root / "checkpoints/snapshot.npz", f"{scene}-snapshot\n".encode())
        entities = _write(root / "checkpoints/entities.jsonl", b"{}\n")
        status = _write_json(root / "checkpoints/status.json", {"status": "PASS"})
        schedule = _write(root / "inputs/schedule.json", self.schedule.read_bytes())
        capture = _write_json(root / "capture_status.json", {"status": "PASS"})
        trajectories = _write(root / "trajectories.jsonl", b"")
        source_index = {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": scene,
            "schedule": _relative_record(schedule, root),
            "capture_status": _relative_record(capture, root),
            "trajectories": _relative_record(trajectories, root),
            "checkpoints": [
                {
                    "frame_index": 0,
                    "timestamp_ns": 100,
                    "consumed_through_frame": 0,
                    "consumed_through_frame_exclusive": 1,
                    "checkpoint_status": _relative_record(status, root),
                    "snapshot": _relative_record(snapshot, root),
                    "entities": _relative_record(entities, root),
                }
            ],
        }
        _write_json(root / "source_index.json", source_index)
        occlusion_index = _write_json(
            root / "occlusion_checkpoint_index.json", {"scene": scene}
        )
        _write_json(
            root / "run_manifest.json",
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "mode": "causal_checkpoints",
                "scene": scene,
                "algorithm_hash": "b" * 64,
                "stage3_lineage_commit": STAGE3_COMMIT,
                "config": {
                    "sha256": _sha256(self.configs[scene]),
                    "byte_count": self.configs[scene].stat().st_size,
                },
                "schedule": {
                    "sha256": _sha256(self.schedule),
                    "byte_count": self.schedule.stat().st_size,
                },
                "source_bindings": {
                    "input_manifest": self._content_record(self.input_manifest),
                    "export_manifest": self._content_record(
                        self.frozen_sources[scene]["export_manifest"]
                    ),
                    "camera": self._content_record(self.camera),
                    "trajectory": self._content_record(
                        self.frozen_sources[scene]["trajectory"]
                    ),
                    "timestamps": self._content_record(
                        self.frozen_sources[scene]["timestamps"]
                    ),
                    "vocabulary_json": self._content_record(
                        self.frozen_sources[scene]["vocabulary_json"]
                    ),
                    "vocabulary_txt": self._content_record(
                        self.frozen_sources[scene]["vocabulary_txt"]
                    ),
                    "frontend_manifest": self._content_record(
                        self.frozen_sources[scene]["frontend_manifest"]
                    ),
                    "dense_manifest": self._content_record(
                        self.frozen_sources[scene]["dense_manifest"]
                    ),
                    "rgbd_combined_output_sha256": "c" * 64,
                },
                "occlusion_checkpoint_index": {
                    **_record(occlusion_index),
                    "path": "occlusion_checkpoint_index.json",
                },
                "checkpoints": [
                    {
                        "scene": scene,
                        "frame_index": 0,
                        "timestamp_ns": 100,
                        "consumed_through_frame": 0,
                        "consumed_through_frame_exclusive": 1,
                    }
                ],
            },
        )
        _write_json(
            root / "run_provenance.json",
            {
                "repository_commit": ADAPTER_COMMIT,
                "dirty_state_digest": CLEAN_DIGEST,
                "command": ["python", "run.py", "--output", str(root.resolve())],
                "hostname": "fixture-host",
                "platform": "fixture-platform",
                "machine": "x86_64",
                "python": "3.12.fixture",
                "library_versions": {"numpy": "fixture"},
                "torch_cuda_version": "12.fixture",
                "cudnn_version": 9000,
                "nvcc_version": ["fixture nvcc"],
                "gpu_inventory": ["fixture gpu"],
                "cuda_visible_devices": "0",
                "config_path": str(self.configs[scene].resolve()),
                "output": str(root.resolve()),
            },
        )

    @staticmethod
    def _content_record(path: Path) -> dict[str, object]:
        record = _record(path)
        record.pop("path")
        return record

    def _write_official_run(
        self, key: str, scene: str, mapping: Path, root: Path
    ) -> None:
        mapping_index = json.loads(
            (mapping / "source_index.json").read_text(encoding="utf-8")
        )
        checkpoint = mapping_index["checkpoints"][0]
        temporal_snapshot = _write(
            root / "temporal/checkpoints/snapshot.npz",
            (mapping / checkpoint["snapshot"]["path"]).read_bytes(),
        )
        temporal_entities = _write(
            root / "temporal/checkpoints/entities.jsonl",
            (mapping / checkpoint["entities"]["path"]).read_bytes(),
        )
        temporal_manifest = _write_json(
            root / "temporal/temporal_manifest.json",
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "method": "OVIV2",
                "mode": "causal_checkpoints",
                "scene": scene,
                "checkpoints": [
                    {
                        "frame_index": 0,
                        "timestamp_ns": 100,
                        "consumed_through_frame": 0,
                        "consumed_through_frame_exclusive": 1,
                        "snapshot": _relative_record(
                            temporal_snapshot, root / "temporal"
                        ),
                        "entities": _relative_record(
                            temporal_entities, root / "temporal"
                        ),
                    }
                ],
            },
        )
        bridge = _write_json(
            root / "bridge_input/bridge_manifest.json",
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "method": "OVIV2",
                "mode": "temporal_checkpoints",
                "display_mode": "online",
                "scene_id": scene,
                "hashed_inputs": [_record(temporal_manifest)],
            },
        )
        build = _write_json(root / "build_manifest.json", {"status": "PASS"})
        final_map = _write(root / "map/final.4dmap", b"map\n")
        timestamps = _write_json(root / "map/map_timestamps.json", [100])
        experiment = _write(root / "map/experiment_log.txt", b"finished\n")
        config_record = _record(self.configs[scene])
        run_identity = {
            "run_id": RUN_ID,
            "config_sha256": config_record["sha256"],
        }
        status = {
            "schema_version": 1,
            "status": "PASS",
            "dataset": "TESSE-CD",
            "scene": scene,
            "method": "OVIV2",
            "mode": "causal_checkpoints",
            "bridge_mode": "temporal_checkpoints",
            "run_identity": run_identity,
            "config": config_record,
            "build_command": ["build", key],
            "command": ["bridge", key],
            "sources": [
                _record(path)
                for path in (
                    self.configs[scene],
                    bridge,
                    build,
                    final_map,
                    timestamps,
                    experiment,
                )
            ],
        }
        _write_json(root / "run_status.json", status)
        results = root / "map/results"
        result_paths = {
            "static_objects.csv": _write(
                results / "static_objects.csv",
                (
                    "Name,Query,AppearedTP,DisappearedTP,AppearedFP,DisappearedFP,"
                    "AppearedFN,DisappearedFN,NumObjDetected,NumObjMissed,"
                    "NumObjHallucinated\n0,10,1,0,0,0,0,0,1,0,0\n"
                ).encode(),
            ),
            "dynamic_objects.csv": _write(
                results / "dynamic_objects.csv",
                (
                    "Name,Query,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
                    "0,10,1,0,0\n"
                ).encode(),
            ),
            "background_mesh.csv": _write(
                results / "background_mesh.csv",
                b"Name,Accuracy@0.2,Completeness@0.2\n0,1.0,1.0\n",
            ),
        }
        metrics = {
            "status": "PASS",
            "dataset": "TESSE-CD",
            "scene": scene,
            "split": f"{scene}_test",
            "method": "OVIV2",
            "mode": "causal_checkpoints",
            "run_identity": run_identity,
            "metrics": {
                "state_count": 1,
                "object_f1": 1.0,
                "dynamic_f1": 1.0,
                "change_f1": 1.0,
                "background_f1_at_0_2": 1.0,
            },
            "sources": [
                {
                    **_record(result_paths[name]),
                    "path": f"../map/results/{name}",
                }
                for name in (
                    "static_objects.csv",
                    "dynamic_objects.csv",
                    "background_mesh.csv",
                )
            ],
        }
        metrics_path = _write_json(root / "evaluation/official_metrics.json", metrics)
        metrics_repeat = _write(
            root / "evaluation/official_metrics.repeat.json", metrics_path.read_bytes()
        )
        metrics_record = _record(metrics_path)
        metrics_record.pop("byte_count")
        metrics_repeat_record = _record(metrics_repeat)
        metrics_repeat_record.pop("byte_count")
        evaluator_config = _write(
            root / f"evaluation/{scene}.yaml", b"evaluation: fixture\n"
        )
        evaluator_log = _write(root / "evaluation/evaluate.log", b"fixture log\n")
        evaluator_time = _write(
            root / "evaluation/evaluate.time.log", b"fixture timing\n"
        )

        def compact_record(path: Path) -> dict[str, object]:
            record = _record(path)
            record.pop("byte_count")
            return record

        _write_json(
            root / "evaluation/evaluation_status.json",
            {
                "status": "PASS",
                "scene": scene,
                "method": "OVIV2",
                "mode": "causal_checkpoints",
                "command": ["evaluate", key],
                "config": compact_record(evaluator_config),
                "log": compact_record(evaluator_log),
                "process_time": compact_record(evaluator_time),
                "official_metrics": metrics_record,
                "official_metrics_repeat": metrics_repeat_record,
            },
        )

    def build(self) -> dict[str, Any]:
        return builder.build_provenance(
            self.freeze_manifest,
            official_runs=self.official_roots,
            weights=self.weights,
        )

    def mutate_json(self, path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutation(payload)
        _write_json(path, payload)


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    return Fixture(tmp_path, monkeypatch)


def _status(scene: str, config_sha256: str) -> dict[str, Any]:
    return {
        "status": "PASS",
        "scene": scene,
        "method": "OVIV2",
        "mode": "causal_checkpoints",
        "run_identity": {"run_id": RUN_ID, "config_sha256": config_sha256},
    }


def test_builds_finalizer_ready_provenance_deterministically(package: Fixture) -> None:
    first = package.build()
    second = package.build()

    assert first == second
    assert first["run_id"] == RUN_ID
    assert first["upstream_commit"] == STAGE3_COMMIT
    assert first["adapter_commit"] == ADAPTER_COMMIT
    assert first["dirty_state_digest"] == CLEAN_DIGEST
    assert [entry["name"] for entry in first["configs"]] == [
        "apartment_frozen_config",
        "office_frozen_config",
    ]
    assert {entry["role"] for entry in first["weights"]} == set(package.weights)
    assert len(first["raw_outputs"]) >= 28
    assert [
        entry["name"]
        for entry in first["raw_outputs"]
        if entry["name"] == "official_temporal_manifest"
    ] == ["official_temporal_manifest"] * 4
    assert set(first["environment"]["mapping_runs"]) == set(package.mapping_roots)
    assert set(first["hardware"]["mapping_runs"]) == set(package.mapping_roots)

    scene_payloads = {
        scene: json.loads(
            (
                package.official_roots[f"{scene}_run1"]
                / "evaluation/official_metrics.json"
            ).read_text(encoding="utf-8")
        )
        for scene in ("apartment", "office")
    }
    statuses = {
        scene: _status(scene, _sha256(package.configs[scene]))
        for scene in ("apartment", "office")
    }
    result = _build_result_payload(
        scene_payloads["apartment"],
        scene_payloads["office"],
        statuses["apartment"],
        statuses["office"],
        first,
        method_key="OVIV2",
        mode="causal_checkpoints",
    )
    assert result["run_id"] == RUN_ID


def test_write_provenance_is_no_clobber(package: Fixture, tmp_path: Path) -> None:
    output = tmp_path / "provenance.json"
    builder.write_provenance(
        package.freeze_manifest,
        official_runs=package.official_roots,
        weights=package.weights,
        output=output,
    )
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        builder.write_provenance(
            package.freeze_manifest,
            official_runs=package.official_roots,
            weights=package.weights,
            output=output,
        )

    assert output.read_bytes() == original


def test_rejects_prepared_manifest(package: Fixture) -> None:
    package.mutate_json(
        package.freeze_manifest, lambda payload: payload.__setitem__("status", "PREPARED")
    )
    with pytest.raises(ValueError, match="FROZEN"):
        package.build()


def test_rejects_frozen_manifest_without_preparation(package: Fixture) -> None:
    package.mutate_json(
        package.freeze_manifest, lambda payload: payload.pop("preparation")
    )
    with pytest.raises(ValueError, match="preparation"):
        package.build()


def test_rejects_dirty_mapping_run(package: Fixture) -> None:
    provenance = package.mapping_roots["apartment_run1"] / "run_provenance.json"
    package.mutate_json(
        provenance,
        lambda payload: payload.__setitem__("dirty_state_digest", "f" * 64),
    )
    with pytest.raises(ValueError, match="dirty"):
        package.build()


def test_rejects_mapping_source_not_bound_by_freeze(package: Fixture) -> None:
    manifest = package.mapping_roots["apartment_run1"] / "run_manifest.json"
    package.mutate_json(
        manifest,
        lambda payload: payload["source_bindings"]["frontend_manifest"].__setitem__(
            "sha256", "f" * 64
        ),
    )
    with pytest.raises(ValueError, match="source bindings"):
        package.build()


def test_rejects_non_identical_mapping_run_manifests(package: Fixture) -> None:
    manifest = package.mapping_roots["office_run2"] / "run_manifest.json"
    package.mutate_json(manifest, lambda payload: payload.__setitem__("nonce", 1))
    with pytest.raises(ValueError, match="mapping run manifests"):
        package.build()


def test_rejects_non_identical_mapping_source_indexes(package: Fixture) -> None:
    source_index = package.mapping_roots["office_run2"] / "source_index.json"
    package.mutate_json(source_index, lambda payload: payload.__setitem__("nonce", 1))
    with pytest.raises(ValueError, match="mapping run manifests"):
        package.build()


def test_rejects_non_identical_temporal_manifests(package: Fixture) -> None:
    root = package.official_roots["office_run2"]
    temporal = root / "temporal/temporal_manifest.json"
    package.mutate_json(temporal, lambda payload: payload.__setitem__("nonce", 1))
    bridge = root / "bridge_input/bridge_manifest.json"

    def rebind_temporal(payload: dict[str, Any]) -> None:
        payload["hashed_inputs"][0].update(_record(temporal))

    package.mutate_json(bridge, rebind_temporal)
    status = root / "run_status.json"

    def rebind_bridge(payload: dict[str, Any]) -> None:
        for source in payload["sources"]:
            if Path(source["path"]) == bridge.resolve():
                source.update(_record(bridge))

    package.mutate_json(status, rebind_bridge)
    with pytest.raises(ValueError, match="temporal manifests"):
        package.build()


def test_rejects_reused_official_run_directory(package: Fixture) -> None:
    package.official_roots["apartment_run2"] = package.official_roots[
        "apartment_run1"
    ]
    with pytest.raises(ValueError, match="distinct"):
        package.build()


def test_rejects_reused_mapping_run_directory(package: Fixture) -> None:
    package.mutate_json(
        package.freeze_manifest,
        lambda payload: payload["output_roots"].__setitem__(
            "apartment_run2", payload["output_roots"]["apartment_run1"]
        ),
    )
    with pytest.raises(ValueError, match="distinct"):
        package.build()


def test_rejects_dirty_builder_repository(
    package: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        builder,
        "_repository_state",
        lambda: {"commit": ADAPTER_COMMIT, "dirty_state_digest": "f" * 64},
    )
    with pytest.raises(ValueError, match="dirty"):
        package.build()


def test_rejects_non_string_frozen_command(package: Fixture) -> None:
    package.mutate_json(
        package.freeze_manifest,
        lambda payload: payload["commands"]["mapping"].__setitem__(0, 7),
    )

    with pytest.raises(ValueError, match="command"):
        package.build()


def test_revalidates_all_hashed_inputs_before_return(
    package: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = builder._commands

    def mutate_after_validation(values: list[str]) -> list[str]:
        package.weights["dense.model"].write_bytes(b"changed after validation\n")
        return original(values)

    monkeypatch.setattr(builder, "_commands", mutate_after_validation)

    with pytest.raises(ValueError, match="changed after provenance capture"):
        package.build()


def test_revalidates_frozen_sources_before_return(
    package: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = builder._commands

    def mutate_after_validation(values: list[str]) -> list[str]:
        package.camera.write_bytes(b"changed after validation\n")
        return original(values)

    monkeypatch.setattr(builder, "_commands", mutate_after_validation)

    with pytest.raises(ValueError, match="hash binding mismatch"):
        package.build()


def test_large_raw_outputs_are_stream_hashed(
    package: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[bool] = []
    original = builder._stable_regular_file

    def record_capture(path: Path, *, label: str, capture: bool):
        if path.name == "final.4dmap":
            observed.append(capture)
        return original(path, label=label, capture=capture)

    monkeypatch.setattr(builder, "_stable_regular_file", record_capture)

    package.build()

    assert observed
    assert observed == [False] * len(observed)


def test_rejects_metrics_even_when_json_and_declared_hash_are_changed_together(
    package: Fixture,
) -> None:
    for key in ("apartment_run1", "apartment_run2"):
        root = package.official_roots[key]
        metrics = root / "evaluation/official_metrics.json"
        repeat = root / "evaluation/official_metrics.repeat.json"
        package.mutate_json(
            metrics,
            lambda payload: payload["metrics"].__setitem__("change_f1", 0.125),
        )
        repeat.write_bytes(metrics.read_bytes())
        evaluation = root / "evaluation/evaluation_status.json"

        def rebind(payload: dict[str, Any]) -> None:
            payload["official_metrics"]["sha256"] = _sha256(metrics)
            payload["official_metrics_repeat"]["sha256"] = _sha256(repeat)

        package.mutate_json(evaluation, rebind)

    with pytest.raises(ValueError, match="recomputed"):
        package.build()


@pytest.mark.parametrize("case", ["config", "run_id", "repeat", "weight"])
def test_rejects_mismatched_final_inputs(package: Fixture, case: str) -> None:
    if case == "config":
        status = package.official_roots["office_run1"] / "run_status.json"
        package.mutate_json(
            status,
            lambda payload: payload["run_identity"].__setitem__(
                "config_sha256", "f" * 64
            ),
        )
    elif case == "run_id":
        status = package.official_roots["office_run1"] / "run_status.json"
        package.mutate_json(
            status,
            lambda payload: payload["run_identity"].__setitem__(
                "run_id", "different-run"
            ),
        )
    elif case == "repeat":
        metrics = (
            package.official_roots["apartment_run2"]
            / "evaluation/official_metrics.json"
        )
        package.mutate_json(
            metrics,
            lambda payload: payload["metrics"].__setitem__("change_f1", 0.1),
        )
    else:
        package.weights.pop("dense.model")

    with pytest.raises(ValueError):
        package.build()
