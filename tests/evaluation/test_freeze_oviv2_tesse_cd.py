from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.evaluation.freeze_oviv2_tesse_cd import (
    ALGORITHM_EXCLUDED_FIELDS,
    STAGE3_LINEAGE_COMMIT,
    _canonical_json_bytes,
    freeze,
)


GRID = {
    "visibility_depth_tolerance_m": [0.05, 0.10, 0.15],
    "absence_negative_support": [0.5, 1.0],
    "ownership_min_net_support": [0.000001, 0.5, 1.0],
}
SCENES = ("apartment", "office")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    )


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _write_json(path: Path, value: Any) -> Path:
    return _write_bytes(path, _canonical_json_bytes(value))


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _algorithm_hash(config: dict[str, Any]) -> str:
    normalized = {
        key: value
        for key, value in sorted(config.items())
        if key not in ALGORITHM_EXCLUDED_FIELDS
    }
    return _json_hash(normalized)


def _candidate_config(base: dict[str, Any], parameters: dict[str, float]) -> dict[str, Any]:
    result = dict(base)
    result.update(parameters)
    result["algorithm_hash"] = _algorithm_hash(result)
    return result


class Fixture:
    def __init__(self, tmp_path: Path) -> None:
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        self.outputs = self.repo / "outputs/oviv2-tessecd-v1"
        self.output_apartment = self.repo / "configs/apartment_frozen.json"
        self.output_office = self.repo / "configs/office_frozen.json"
        self.output_manifest = self.outputs / "freeze_manifest.json"
        self.environment = {
            "python": "3.fixture",
            "cuda": "fixture-cuda",
            "gpu": ["fixture-gpu"],
            "host": "fixture-host",
            "libraries": {"numpy": "fixture-numpy"},
        }
        self.repository_state = {
            "commit": "a" * 40,
            "parents": ["b" * 40],
            "tree": "c" * 40,
            "commit_time_utc": "2026-07-22T00:00:00+00:00",
            "clean": True,
            "stage3_lineage_commit": STAGE3_LINEAGE_COMMIT,
            "stage3_is_ancestor": True,
        }

        self.source_manifest = _write_json(
            self.repo / "configs/evaluation/manifests/tesse_cd.json",
            {"schema_version": 1, "dataset": "TESSE-CD"},
        )
        self.schedule = _write_json(
            self.repo / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json",
            {
                "schema_version": 2,
                "manifest_id": "tesse_cd_causal_schedule_v2",
                "dataset": "TESSE-CD",
                "method_predictions_used": False,
                "scenes": {
                    "apartment": {"frame_count": 2, "entries": []},
                    "office": {"frame_count": 3, "entries": []},
                },
            },
        )
        self.alias_map = _write_bytes(
            self.repo / "configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml",
            b"aliases: {}\n",
        )
        self.evaluator = _write_bytes(
            self.repo / "scripts/evaluation/evaluate_tesse_cd_common_v2.py",
            b"# evaluator fixture\n",
        )
        self.common_finalizer = _write_bytes(
            self.repo / "scripts/evaluation/finalize_tesse_common_v2.py",
            b"# common finalizer fixture\n",
        )
        self.official_finalizer = _write_bytes(
            self.repo / "scripts/evaluation/finalize_tesse_t2.py",
            b"# official finalizer fixture\n",
        )

        self.camera = _write_json(
            self.repo / "assets/rgbd/cam_params.json",
            {"height": 480, "width": 720, "fx": 1.0, "fy": 1.0},
        )
        self.scene_artifacts: dict[str, dict[str, Path]] = {}
        self.configs: dict[str, dict[str, Any]] = {}
        self.config_paths: dict[str, Path] = {}
        input_scenes: dict[str, Any] = {}
        for scene, frame_count in (("apartment", 2), ("office", 3)):
            root = self.repo / f"assets/rgbd/{scene}"
            database = _write_bytes(root / f"{scene}.db3", f"db-{scene}\n".encode())
            timestamps = _write_bytes(root / "timestamps.csv", b"0,0\n")
            trajectory = _write_bytes(root / "traj.txt", b"0 0 0 0 0 0 1\n")
            vocabulary_json = _write_json(
                self.repo / f"configs/evaluation/vocabularies/tesse_cd_{scene}.json",
                {"scene": scene, "classes": ["chair", "wall"]},
            )
            vocabulary_txt = _write_bytes(
                self.repo / f"configs/evaluation/vocabularies/tesse_cd_{scene}.txt",
                b"chair\nwall\n",
            )
            export_manifest = _write_json(
                root / "export_manifest.json",
                {
                    "schema_version": 1,
                    "dataset": "TESSE-CD",
                    "scene": scene,
                    "frame_count": frame_count,
                    "source_database": str(database),
                    "source_database_sha256": _sha256(database),
                    "source_manifest": str(self.source_manifest),
                    "combined_output_sha256": ("1" if scene == "apartment" else "2") * 64,
                    "file_hash_count": frame_count * 2 + 3,
                },
            )
            frontend_manifest = _write_json(
                self.repo / f"cache/frontend/{scene}/frontend_manifest.json",
                {
                    "schema_version": 1,
                    "method": "OVIV2",
                    "dataset": "TESSE-CD",
                    "scene": scene,
                    "frame_count": frame_count,
                    "algorithm_hash": "3" * 64,
                    "cache_prefix_sha256": ("4" if scene == "apartment" else "5") * 64,
                    "feature_model_id": "clip-sha256:" + "6" * 64,
                    "provenance_sha256": {
                        "script": "7" * 64,
                        "hydra_config": "8" * 64,
                        "dataset_config": "9" * 64,
                        "yolo_model": "a" * 64,
                        "yolo_clip_model": "b" * 64,
                        "mobile_sam_model": "c" * 64,
                        "clip_model": "6" * 64,
                    },
                },
            )
            dense_manifest = _write_json(
                self.repo / f"cache/dense/{scene}/dense_manifest.json",
                {
                    "schema_version": 1,
                    "method": "OVIV2-dense-semantic-cache",
                    "scene": scene,
                    "frame_count": frame_count,
                    "provenance": {
                        "backend": "radseg",
                        "model_id": "radseg:fixture",
                        "model_sha256": "d" * 64,
                        "auxiliary_model_sha256": "e" * 64,
                        "language_model_id": "siglip-fixture",
                        "language_model_revision": "f" * 40,
                        "language_model_sha256": "f" * 64,
                        "vocabulary_sha256": ("0" if scene == "apartment" else "1") * 64,
                        "prompt_sha256": ("2" if scene == "apartment" else "3") * 64,
                        "inference_config_sha256": "4" * 64,
                        "cache_prefix_sha256": ("5" if scene == "apartment" else "6") * 64,
                    },
                },
            )
            input_scenes[scene] = {
                "root": str(root),
                "frame_count": frame_count,
                "source_database_sha256": _sha256(database),
                "export_manifest": {
                    **_record(export_manifest),
                    "combined_output_sha256": json.loads(export_manifest.read_text())[
                        "combined_output_sha256"
                    ],
                    "file_hash_count": frame_count * 2 + 3,
                },
                "timestamps": _record(timestamps),
                "trajectory": _record(trajectory),
                "vocabulary": {
                    "json_path": str(vocabulary_json),
                    "json_sha256": _sha256(vocabulary_json),
                    "txt_path": str(vocabulary_txt),
                    "txt_sha256": _sha256(vocabulary_txt),
                },
            }
            self.scene_artifacts[scene] = {
                "root": root,
                "database": database,
                "timestamps": timestamps,
                "trajectory": trajectory,
                "export_manifest": export_manifest,
                "frontend_manifest": frontend_manifest,
                "dense_manifest": dense_manifest,
                "vocabulary_json": vocabulary_json,
                "vocabulary_txt": vocabulary_txt,
            }

        self.input_manifest = _write_json(
            self.repo / "configs/evaluation/manifests/oviv2_tesse_cd_cache.json",
            {
                "schema_version": 1,
                "manifest_id": "oviv2_tesse_cd_cache_v1",
                "dataset": "TESSE-CD",
                "stage3_lineage_commit": STAGE3_LINEAGE_COMMIT,
                "camera": _record(self.camera),
                "source_manifest": _record(self.source_manifest),
                "schedule_manifest": _record(self.schedule),
                "scenes": input_scenes,
            },
        )

        for scene, frame_count in (("apartment", 2), ("office", 3)):
            artifact = self.scene_artifacts[scene]
            config = {
                "schema_version": 1,
                "method_id": "OVIV2",
                "dataset": "TESSE-CD",
                "scene": scene,
                "frame_count": frame_count,
                "stage3_lineage_commit": STAGE3_LINEAGE_COMMIT,
                "input_manifest": str(self.input_manifest),
                "dataset_root": str(artifact["root"]),
                "export_manifest": str(artifact["export_manifest"]),
                "schedule_manifest": str(self.schedule),
                "frontend_cache_dir": str(artifact["frontend_manifest"].parent),
                "frontend_manifest": str(artifact["frontend_manifest"]),
                "dense_cache_dir": str(artifact["dense_manifest"].parent),
                "dense_manifest": str(artifact["dense_manifest"]),
                "vocabulary_json": str(artifact["vocabulary_json"]),
                "vocabulary_txt": str(artifact["vocabulary_txt"]),
                "evaluation_checkpoint_frames": [1] if scene == "apartment" else [1, 2],
                "missing_observation_policy": "signed_depth",
                "visibility_depth_tolerance_m": 0.10,
                "absence_negative_support": 1.0,
                "ownership_min_net_support": 0.000001,
                "voxel_size_m": 0.05,
                "association_minimum_score": 0.45,
            }
            config["algorithm_hash"] = _algorithm_hash(config)
            self.configs[scene] = config
            self.config_paths[scene] = _write_json(
                self.repo / f"configs/{scene}.json", config
            )

        self.targets = _write_bytes(
            self.repo / "targets/common-v2/targets.npz", b"target fixture\n"
        )
        self.target_manifest = _write_json(
            self.targets.parent / "manifest.json",
            {
                "schema_version": 1,
                "manifest_id": "tesse_cd_common_v2_targets_v1",
                "dataset": "TESSE-CD",
                "status": "GENERATED",
                "targets_generated": True,
                "prediction_inputs_used": False,
                "metadata": {
                    "protocol_complete": True,
                    "schedule": _record(self.schedule),
                },
                "target_arrays": {
                    "path": self.targets.name,
                    "sha256": _sha256(self.targets),
                    "byte_count": self.targets.stat().st_size,
                    "count": 4,
                },
            },
        )
        self.selection_path = self.repo / "tuning/apartment/selection.json"
        self._write_selection()

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            apartment_selection=self.selection_path,
            apartment_config=self.config_paths["apartment"],
            office_config=self.config_paths["office"],
            output_apartment_config=self.output_apartment,
            output_office_config=self.output_office,
            output_manifest=self.output_manifest,
        )

    def _write_selection(
        self,
        *,
        all_metrics_equal: bool = False,
        selected_override: str | None = None,
    ) -> None:
        candidates = []
        for index, values in enumerate(itertools.product(*GRID.values())):
            parameters = dict(zip(GRID, values, strict=True))
            candidate_config = _candidate_config(self.configs["apartment"], parameters)
            config_sha256 = _json_hash(candidate_config)
            metrics = (
                {
                    "current_miou": 0.5,
                    "ghost_rate": 0.1,
                    "background_f5_cm": 0.7,
                    "recovery_frames": 5.0,
                }
                if all_metrics_equal
                else {
                    "current_miou": 0.40 + index / 100.0,
                    "ghost_rate": 0.30 - index / 1000.0,
                    "background_f5_cm": 0.50 + index / 1000.0,
                    "recovery_frames": float(20 - index),
                }
            )
            summary_payload = {
                "schema_version": 1,
                "status": "PASS",
                "scene": "apartment",
                "parameters": parameters,
                "config_sha256": config_sha256,
                "common_target_manifest_sha256": _sha256(self.target_manifest),
                "metrics": metrics,
            }
            summary_path = _write_json(
                self.repo / f"tuning/apartment/candidate-{index:02d}.json",
                summary_payload,
            )
            candidates.append({**summary_payload, "summary": _record(summary_path)})
        winner = min(
            candidates,
            key=lambda item: (
                -item["metrics"]["current_miou"],
                item["metrics"]["ghost_rate"],
                -item["metrics"]["background_f5_cm"],
                item["metrics"]["recovery_frames"],
                item["config_sha256"],
            ),
        )
        selection = {
            "schema_version": 1,
            "method": "OVIV2",
            "scene": "apartment",
            "parameter_grid": GRID,
            "common_target_manifest": _record(self.target_manifest),
            "freeze_bindings": {
                "shared": {
                    "input_manifest": _record(self.input_manifest),
                    "source_manifest": _record(self.source_manifest),
                    "schedule": _record(self.schedule),
                    "camera": _record(self.camera),
                    "alias_map": _record(self.alias_map),
                    "evaluator": _record(self.evaluator),
                    "common_finalizer": _record(self.common_finalizer),
                    "official_finalizer": _record(self.official_finalizer),
                },
                "scenes": {
                    scene: {
                        role: _record(path)
                        for role, path in self.scene_artifacts[scene].items()
                        if role != "root"
                    }
                    for scene in SCENES
                },
            },
            "candidates": candidates,
            "selected_config_sha256": selected_override or winner["config_sha256"],
        }
        _write_json(self.selection_path, selection)

    def rewrite_config(self, scene: str, mutate: Any) -> None:
        config = json.loads(self.config_paths[scene].read_text())
        mutate(config)
        _write_json(self.config_paths[scene], config)

    def run(self, **overrides: Any) -> dict[str, Any]:
        repository_state = dict(self.repository_state)
        repository_state.update(overrides.pop("repository_state", {}))
        return freeze(
            self.args(),
            repo_root=self.repo,
            repository_state=repository_state,
            environment=overrides.pop("environment", self.environment),
            **overrides,
        )


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_cli_direct_execution_resolves_repository_imports() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluation/freeze_oviv2_tesse_cd.py", "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_freeze_writes_selected_configs_and_complete_manifest(fixture: Fixture) -> None:
    manifest = fixture.run()

    apartment = json.loads(fixture.output_apartment.read_text())
    office = json.loads(fixture.output_office.read_text())
    selection = json.loads(fixture.selection_path.read_text())
    selected = next(
        item
        for item in selection["candidates"]
        if item["config_sha256"] == selection["selected_config_sha256"]
    )
    for name, value in selected["parameters"].items():
        assert apartment[name] == value
        assert office[name] == value
    assert apartment["algorithm_hash"] == office["algorithm_hash"]
    assert apartment["algorithm_hash"] == manifest["algorithm"]["sha256"]
    assert manifest["status"] == "FROZEN"
    assert manifest["freeze_id"] == "oviv2-tessecd-v1"
    assert manifest["repository"]["commit"] == "a" * 40
    assert manifest["repository"]["stage3_is_ancestor"] is True
    assert manifest["selection"]["candidate_count"] == 18
    assert manifest["selection"]["selected_config_sha256"] == _json_hash(apartment)
    assert manifest["scenes"]["apartment"]["rgbd"]["source_database"]["sha256"]
    assert manifest["scenes"]["office"]["rgbd"]["combined_output_sha256"]
    assert manifest["scenes"]["apartment"]["cache"]["frontend_manifest"]["sha256"]
    assert manifest["scenes"]["office"]["cache"]["dense_manifest"]["sha256"]
    assert manifest["models"]["dense"]["apartment"]["model_sha256"] == "d" * 64
    assert manifest["shared_bindings"]["common_target_arrays"]["sha256"] == _sha256(
        fixture.targets
    )
    assert manifest["shared_bindings"]["schedule"]["sha256"] == _sha256(
        fixture.schedule
    )
    assert manifest["shared_bindings"]["alias_map"]["sha256"] == _sha256(
        fixture.alias_map
    )
    assert manifest["shared_bindings"]["evaluator"]["sha256"] == _sha256(
        fixture.evaluator
    )
    assert set(manifest["shared_bindings"]["finalizers"]) == {"common_v2", "official_t2"}
    assert manifest["environment"] == fixture.environment
    assert len(manifest["commands"]["mapping"]) == 4
    assert manifest["office_pre_freeze_audit"]["metric_sources_found"] == []
    assert set(manifest["output_roots"]) == {"apartment_run1", "apartment_run2", "office_run1", "office_run2"}

    for path in (
        fixture.output_apartment,
        fixture.output_office,
        fixture.output_manifest,
    ):
        parsed = json.loads(path.read_text())
        assert path.read_bytes() == _canonical_json_bytes(parsed)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value["candidates"].pop(), "18 candidates"),
        (
            lambda value: value["candidates"][0]["parameters"].__setitem__(
                "visibility_depth_tolerance_m", 0.2
            ),
            "parameter grid",
        ),
        (
            lambda value: value["parameter_grid"].__setitem__(
                "absence_negative_support", [0.25, 1.0]
            ),
            "predeclared parameter grid",
        ),
    ],
)
def test_freeze_rejects_non_predeclared_sweep(
    fixture: Fixture, mutation: Any, message: str
) -> None:
    selection = json.loads(fixture.selection_path.read_text())
    mutation(selection)
    _write_json(fixture.selection_path, selection)

    with pytest.raises(ValueError, match=message):
        fixture.run()

    assert not fixture.output_manifest.exists()


def test_freeze_enforces_lexicographic_selection_and_hash_tiebreak(fixture: Fixture) -> None:
    fixture._write_selection(all_metrics_equal=True)
    selection = json.loads(fixture.selection_path.read_text())
    winner = min(item["config_sha256"] for item in selection["candidates"])
    wrong = max(item["config_sha256"] for item in selection["candidates"])
    assert winner != wrong
    selection["selected_config_sha256"] = wrong
    _write_json(fixture.selection_path, selection)

    with pytest.raises(ValueError, match="lexicographic winner"):
        fixture.run()


def test_freeze_rejects_candidate_summary_hash_or_content_mismatch(fixture: Fixture) -> None:
    selection = json.loads(fixture.selection_path.read_text())
    summary = Path(selection["candidates"][0]["summary"]["path"])
    summary.write_text(summary.read_text() + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate summary"):
        fixture.run()


def test_freeze_rejects_non_allowlisted_office_algorithm_difference(fixture: Fixture) -> None:
    def mutate(value: dict[str, Any]) -> None:
        value["association_minimum_score"] = 0.99
        value["algorithm_hash"] = _algorithm_hash(value)

    fixture.rewrite_config("office", mutate)

    with pytest.raises(ValueError, match="algorithm configuration"):
        fixture.run()


def test_freeze_rejects_stale_config_algorithm_hash(fixture: Fixture) -> None:
    fixture.rewrite_config("office", lambda value: value.__setitem__("algorithm_hash", "0" * 64))

    with pytest.raises(ValueError, match="algorithm_hash"):
        fixture.run()


def test_freeze_rejects_cross_scene_model_weight_change(fixture: Fixture) -> None:
    dense_path = fixture.scene_artifacts["office"]["dense_manifest"]
    dense = json.loads(dense_path.read_text())
    dense["provenance"]["model_sha256"] = "0" * 64
    _write_json(dense_path, dense)
    selection = json.loads(fixture.selection_path.read_text())
    selection["freeze_bindings"]["scenes"]["office"]["dense_manifest"] = _record(
        dense_path
    )
    _write_json(fixture.selection_path, selection)

    with pytest.raises(ValueError, match="same frozen model"):
        fixture.run()


def test_freeze_rejects_cache_directory_that_does_not_own_manifest(
    fixture: Fixture,
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        value["dense_cache_dir"] = str(fixture.repo / "cache/dense/wrong")
        value["algorithm_hash"] = _algorithm_hash(value)

    fixture.rewrite_config("office", mutate)

    with pytest.raises(ValueError, match="cache directory"):
        fixture.run()


@pytest.mark.parametrize("forbidden", ["route3-surface-observation-v5", "stage4", "ScanNet200"])
def test_freeze_rejects_route3_stage4_or_scannet_config(
    fixture: Fixture, forbidden: str
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        value["implementation_source"] = forbidden
        value["algorithm_hash"] = _algorithm_hash(value)

    fixture.rewrite_config("apartment", mutate)
    fixture.rewrite_config("office", mutate)

    with pytest.raises(ValueError, match="forbidden"):
        fixture.run()


@pytest.mark.parametrize(
    "repository_state, message",
    [
        ({"clean": False}, "clean repository"),
        ({"stage3_is_ancestor": False}, "47962fb"),
        ({"stage3_lineage_commit": "0" * 40}, "47962fb"),
    ],
)
def test_freeze_requires_clean_stage3_repository_lineage(
    fixture: Fixture, repository_state: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fixture.run(repository_state=repository_state)


@pytest.mark.parametrize(
    "artifact",
    ["source_manifest", "schedule", "camera", "database", "timestamps", "trajectory", "export_manifest", "frontend_manifest", "dense_manifest", "vocabulary_json", "vocabulary_txt", "target_manifest", "targets", "alias_map", "evaluator", "common_finalizer", "official_finalizer"],
)
def test_freeze_rehashes_every_declared_artifact(fixture: Fixture, artifact: str) -> None:
    if artifact in fixture.scene_artifacts["apartment"]:
        path = fixture.scene_artifacts["apartment"][artifact]
    else:
        path = getattr(fixture, artifact)
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="hash|checksum|binding|target"):
        fixture.run()


def test_freeze_rejects_missing_declared_hash(fixture: Fixture) -> None:
    manifest = json.loads(fixture.input_manifest.read_text())
    del manifest["scenes"]["apartment"]["trajectory"]["sha256"]
    _write_json(fixture.input_manifest, manifest)

    with pytest.raises(ValueError, match="trajectory.*sha256|hash"):
        fixture.run()


def test_freeze_rejects_office_metric_source_before_freeze(fixture: Fixture) -> None:
    fixture.rewrite_config(
        "office",
        lambda value: value.__setitem__("metric_source", "outputs/office/metrics.json"),
    )

    with pytest.raises(ValueError, match="Office metric source"):
        fixture.run()


@pytest.mark.parametrize("existing", ["apartment", "office", "manifest", "prior_run"])
def test_freeze_is_no_clobber_and_rejects_prior_run_output(
    fixture: Fixture, existing: str
) -> None:
    if existing == "apartment":
        path = _write_bytes(fixture.output_apartment, b"keep-apartment\n")
    elif existing == "office":
        path = _write_bytes(fixture.output_office, b"keep-office\n")
    elif existing == "manifest":
        path = _write_bytes(fixture.output_manifest, b"keep-manifest\n")
    else:
        path = _write_bytes(fixture.outputs / "office/run1/metrics.json", b"keep-run\n")
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        fixture.run()

    assert path.read_bytes() == before


def test_freeze_rolls_back_its_publications_on_atomic_link_failure(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_link = os.link

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_link(source, destination)

    monkeypatch.setattr(
        "scripts.evaluation.freeze_oviv2_tesse_cd._link_no_replace", fail_second
    )

    with pytest.raises(OSError, match="injected"):
        fixture.run()

    assert not fixture.output_apartment.exists()
    assert not fixture.output_office.exists()
    assert not fixture.output_manifest.exists()
    assert list(fixture.repo.rglob("*.freeze-tmp")) == []


def test_freeze_rolls_back_if_prior_run_appears_during_publication(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_link = os.link

    def inject_prior_run(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            _write_bytes(fixture.outputs / "office/run1/metrics.json", b"raced\n")
        real_link(source, destination)

    monkeypatch.setattr(
        "scripts.evaluation.freeze_oviv2_tesse_cd._link_no_replace",
        inject_prior_run,
    )

    with pytest.raises(FileExistsError, match="prior run output"):
        fixture.run()

    assert not fixture.output_apartment.exists()
    assert not fixture.output_office.exists()
    assert not fixture.output_manifest.exists()


@pytest.mark.parametrize(
    "invalid, message",
    [
        (math.nan, "non-finite JSON constant"),
        (math.inf, "non-finite JSON constant"),
        (True, "finite numeric metric"),
        ("0.5", "finite numeric metric"),
    ],
)
def test_freeze_rejects_nonfinite_or_non_numeric_metrics(
    fixture: Fixture, invalid: Any, message: str
) -> None:
    selection = json.loads(fixture.selection_path.read_text())
    selection["candidates"][0]["metrics"]["current_miou"] = invalid
    fixture.selection_path.write_text(
        json.dumps(selection, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        fixture.run()


def test_freeze_rejects_duplicate_json_keys(fixture: Fixture) -> None:
    fixture.selection_path.write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        fixture.run()
