from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import pytest

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    canonical_algorithm_hash,
)
from scripts.evaluation.freeze_oviv2_tesse_cd_v2 import (
    FreezeDependencies,
    _canonical_json_bytes,
    freeze,
)


SCENES = ("apartment", "office")
CANDIDATES = tuple(f"a{index}" for index in range(5))
SOURCE_ROLES = (
    "search_manifest",
    "search_status",
    "candidate_config",
    "run_manifest",
    "common_v2_summary",
    "temporal_occlusion_result",
    "official_metrics",
    "t1_exact_evidence",
    "determinism_evidence",
)
SOURCE_CONFIGS = {
    scene: Path(f"configs/oviv2_tesse_cd_{scene}_v2.json") for scene in SCENES
}


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _write_json(path: Path, value: object) -> Path:
    return _write_bytes(path, _canonical_json_bytes(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_hash(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value).rstrip(b"\n"))


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


class Fixture:
    def __init__(self, tmp_path: Path) -> None:
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        self.output = self.repo / "formal" / "freeze.json"
        self.release_paths = {
            role: _write_bytes(self.repo / f"release/{role}.py", role.encode())
            for role in ("temporal_evaluator", "result_finalizer")
        }
        self.shared = {
            "input_manifest": _write_json(
                self.repo / "inputs/input.json", {"schema_version": 1}
            ),
            "schedule_manifest": _write_json(
                self.repo / "inputs/schedule.json", {"schema_version": 2}
            ),
            "occlusion_target_manifest": _write_json(
                self.repo / "inputs/targets.json", {"schema_version": 1}
            ),
        }
        self.configs: dict[str, dict[str, Any]] = {}
        self.config_paths: dict[str, Path] = {}
        for scene in SCENES:
            config = json.loads(SOURCE_CONFIGS[scene].read_text(encoding="utf-8"))
            scene_dir = self.repo / "inputs" / scene
            export = _write_json(scene_dir / "export.json", {"scene": scene})
            frontend = _write_json(
                scene_dir / "frontend.json",
                {
                    "scene": scene,
                    "feature_model_id": f"clip-sha256:{'f' * 64}",
                    "provenance_sha256": {"clip_model": "f" * 64},
                },
            )
            dense = _write_json(
                scene_dir / "dense.json",
                {
                    "scene": scene,
                    "provenance": {
                        "model_id": "fixture-dense-model",
                        "model_sha256": "d" * 64,
                    }
                },
            )
            vocabulary_json = _write_json(
                scene_dir / "vocabulary.json", {"scene": scene}
            )
            vocabulary_txt = _write_bytes(
                scene_dir / "vocabulary.txt", f"{scene}\n".encode()
            )
            config.update(
                {
                    "protocol_id": "oviv2-tessecd-v2",
                    "input_manifest": str(self.shared["input_manifest"]),
                    "schedule_manifest": str(self.shared["schedule_manifest"]),
                    "occlusion_target_manifest": str(
                        self.shared["occlusion_target_manifest"]
                    ),
                    "occlusion_target_manifest_sha256": _sha256(
                        self.shared["occlusion_target_manifest"]
                    ),
                    "export_manifest": str(export),
                    "frontend_manifest": str(frontend),
                    "dense_manifest": str(dense),
                    "vocabulary_json": str(vocabulary_json),
                    "vocabulary_txt": str(vocabulary_txt),
                }
            )
            config["algorithm_hash"] = canonical_algorithm_hash(config)
            path = _write_json(self.repo / f"configs/{scene}.json", config)
            self.configs[scene] = config
            self.config_paths[scene] = path
        assert self.configs["apartment"]["algorithm_hash"] == self.configs["office"][
            "algorithm_hash"
        ]

        self.search_manifest = _write_json(
            self.repo / "development/search_manifest.json",
            {"schema_version": 1, "candidates": list(CANDIDATES)},
        )
        ledger = []
        result_files = []
        selected_config_record: dict[str, object] | None = None
        for candidate in CANDIDATES:
            sources = {
                role: _record(
                    _write_json(
                        self.repo
                        / (
                            f"development/candidates/{candidate}/apartment/"
                            f"evidence/{role}.json"
                        ),
                        (
                            self.configs["apartment"]
                            if role == "candidate_config" and candidate == "a3"
                            else (
                                {
                                    "schema_version": 1,
                                    "status": "PASS",
                                    "office_binding": {
                                        "scene": "office",
                                        "executed": False,
                                    },
                                    "candidates": [
                                        {
                                            "candidate_id": candidate,
                                            "scene": "apartment",
                                        }
                                    ],
                                }
                                if role == "search_status"
                                else {
                                    "schema_version": 1,
                                    "candidate_id": candidate,
                                    "role": role,
                                }
                            )
                        ),
                    )
                )
                for role in SOURCE_ROLES
            }
            result = _write_json(
                self.repo
                / f"development/candidates/{candidate}/apartment/result.json",
                {
                    "schema_version": 1,
                    "manifest_id": "oviv2-tesse-dual-readout-candidate-result-v1",
                    "candidate_id": candidate,
                    "scene": "apartment",
                    "status": "PASS",
                    "sources": sources,
                    "config_sha256": (
                        _json_hash(self.configs["apartment"])
                        if candidate == "a3"
                        else candidate[-1] * 64
                    ),
                },
            )
            result_files.append({"candidate_id": candidate, **_record(result)})
            if candidate == "a3":
                selected_config_record = sources["candidate_config"]
            ledger.append(
                {
                    "candidate_id": candidate,
                    "eligible": candidate == "a3",
                    "selected": candidate == "a3",
                    "reasons": [] if candidate == "a3" else ["not_selected"],
                    "notes": [],
                    "config_sha256": (
                        _json_hash(self.configs["apartment"])
                        if candidate == "a3"
                        else candidate[-1] * 64
                    ),
                    "algorithm_hash": (
                        self.configs["apartment"]["algorithm_hash"]
                        if candidate == "a3"
                        else candidate[-1] * 64
                    ),
                    "gates": {"t1_exact": True},
                    "metrics": {"ghost_rate": 0.1},
                    "sources": sources,
                }
            )
        assert selected_config_record is not None
        self.selection_value = {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_cd_v2_selection",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "protocol_id": "oviv2-tessecd-v2",
            "status": "PASS",
            "development_scene": "apartment",
            "transfer_scene": "office",
            "scenes_read": ["apartment"],
            "office_results_read": False,
            "selected_candidate_id": "a3",
            "selected_config": self.configs["apartment"],
            "selected_config_record": selected_config_record,
            "selected_config_sha256": _json_hash(self.configs["apartment"]),
            "algorithm_hash": self.configs["apartment"]["algorithm_hash"],
            "manifest": _record(self.search_manifest),
            "result_contract": "candidates/<candidate_id>/apartment/result.json",
            "results_root": str((self.repo / "development").resolve()),
            "result_files": result_files,
            "promotion_order": ["hard_gates", "ghost_rate"],
            "metric_policy": {"required": ["ghost_rate"]},
            "floors": {"ghost_rate_exclusive_maximum": 0.218},
            "skipped_optional_tie_axes": {},
            "rejection_ledger": ledger,
        }
        self.selection = _write_json(
            self.repo / "development/selection.json", self.selection_value
        )
        self.t1_source = _write_json(
            self.repo / "development/t1/source.json", {"manifest_id": "sources"}
        )
        t1_root = "e" * 64
        self.t1_evidence = _write_json(
            self.repo / "development/t1_evidence.json",
            {
                "schema_version": 1,
                "manifest_id": "oviv2_dual_readout_development_gates_v1",
                "deterministic_evidence": {
                    "source_manifest": _record(self.t1_source),
                    "cumulative_exact": {
                        "format": "oviv2_t1_exact_transaction_v1",
                        "sequence": [
                            "reference", "a0", "a1", "a0", "a2",
                            "a0", "a3", "a0", "a4",
                        ],
                        "executions": [{"profile": "reference"}],
                        "profiles": {
                            profile: {
                                "cumulative_root_sha256": t1_root,
                                "checkpoint_frames": [2, 4],
                                "inventory": [],
                            }
                            for profile in CANDIDATES
                        },
                    },
                    "gates": {
                        name: {"status": "PASS"}
                        for name in ("t1_exact", "determinism")
                    },
                },
                "receipt": {"created_at_utc": "2026-07-26T00:00:00Z"},
            },
        )
        self.t4_shortlist = _write_json(
            self.repo / "development/t4/shortlist.json",
            {
                "schema_version": 1,
                "manifest_id": "oviv2_tesse_dual_readout_shortlist_v1",
                "phase": "shortlist",
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "protocol_id": "oviv2-tessecd-v2",
                "status": "PASS",
                "development_scene": "apartment",
                "transfer_scene": "office",
                "office_results_read": False,
                "scenes_read": ["apartment"],
                "result_contract": "candidates/<candidate_id>/apartment/result.json",
                "results_root": str((self.repo / "development").resolve()),
                "manifest": _record(self.search_manifest),
                "result_files": [],
                "profile_fallback_order": ["a4", "a3", "a2"],
                "floors": {"current_miou_from_a0": 0.0, "object_f1_from_a0": 0.0},
                "shortlisted_candidate_ids": ["a3"],
                "shortlisted_candidates": [
                    {
                        "candidate_id": "a3",
                        "profile": "a3",
                        "config_sha256": _json_hash(self.configs["apartment"]),
                        "algorithm_hash": self.configs["apartment"]["algorithm_hash"],
                        "result": _record(self.search_manifest),
                        "selected_config": self.configs["apartment"],
                        "selected_config_record": selected_config_record,
                    }
                ],
                "rejection_ledger": [],
            },
        )
        t4_metrics = {
            name: 1.0
            for name in (
                "total_runtime_s_per_frame", "query_mean_ms", "query_p95_ms",
                "peak_gpu_gb", "peak_ram_gb", "final_map_mb",
            )
        }
        t4_run = _write_json(
            self.repo / "development/t4/a3-run.json",
            {
                "schema_version": 2,
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "protocol_id": "oviv2-tessecd-v2",
                "scene": "apartment",
                "candidate_id": "a3",
                "config_sha256": _json_hash(self.configs["apartment"]),
            },
        )
        t4_metric_sources = {
            name: _record(
                _write_json(
                    self.repo / f"development/t4/a3-{name}.json",
                    {
                        "schema_version": 1,
                        "manifest_id": "oviv2_tesse_t4_metric_v1",
                        "scene": "apartment",
                        "candidate_id": "a3",
                        "config_sha256": _json_hash(self.configs["apartment"]),
                        "run_manifest_sha256": _sha256(t4_run),
                        "metric": name,
                        "value": value,
                    },
                )
            )
            for name, value in t4_metrics.items()
        }
        self.t4_protocol = _write_json(
            self.repo / "development/t4/protocol.json",
            {
                "schema_version": 1,
                "manifest_id": "oviv2_tesse_t4_protocol_v1",
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "protocol_id": "oviv2-tessecd-v2",
                "scene": "apartment",
                "bounds": {
                    "total_runtime_s_per_frame": 6.42,
                    "query_mean_ms": 11.92,
                    "query_p95_ms": 12.12,
                    "peak_gpu_gb": 12.76,
                    "peak_ram_gb": 9.36,
                    "final_map_mb": 46.77,
                },
                "candidates": {
                    "a3": {
                        "config_sha256": _json_hash(self.configs["apartment"]),
                        "run_manifest": _record(t4_run),
                        "metric_sources": t4_metric_sources,
                    }
                },
            },
        )
        t4_without_root = {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_t4_matrix_v1",
            "status": "PASS",
            "shortlist": _record(self.t4_shortlist),
            "protocol": _record(self.t4_protocol),
            "candidates": {
                "a3": {
                    "status": "PASS",
                    "config_sha256": _json_hash(self.configs["apartment"]),
                    "run_manifest_sha256": _sha256(t4_run),
                    "metrics": t4_metrics,
                    "gates": {
                        name: True
                        for name in (
                            "total_runtime_s_per_frame", "query_mean_ms",
                            "query_p95_ms", "peak_gpu_gb", "peak_ram_gb",
                            "final_map_mb",
                        )
                    },
                }
            },
        }
        self.t4_evidence = _write_json(
            self.repo / "development/t4_evidence.json",
            {
                **t4_without_root,
                "root_sha256": _json_hash(t4_without_root),
            },
        )
        self.selection_value = {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_dual_readout_selection_v2",
            "phase": "final",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "protocol_id": "oviv2-tessecd-v2",
            "status": "PASS",
            "development_scene": "apartment",
            "transfer_scene": "office",
            "scenes_read": ["apartment"],
            "office_results_read": False,
            "manifest": _record(self.search_manifest),
            "shortlist": _record(self.t4_shortlist),
            "t4_matrix": _record(self.t4_evidence),
            "t4_protocol": _record(self.t4_protocol),
            "t4_root_sha256": _json_hash(t4_without_root),
            "profile_fallback_order": ["a4", "a3", "a2"],
            "selected_candidate_id": "a3",
            "selected_config": self.configs["apartment"],
            "selected_config_record": selected_config_record,
            "selected_config_sha256": _json_hash(self.configs["apartment"]),
            "algorithm_hash": self.configs["apartment"]["algorithm_hash"],
            "t4_ledger": [
                {
                    "candidate_id": "a3",
                    "profile": "a3",
                    "passed": True,
                    "selected": True,
                    "failed_gates": [],
                    "config_sha256": _json_hash(self.configs["apartment"]),
                    "run_manifest_sha256": _sha256(t4_run),
                }
            ],
        }
        self.selection = _write_json(
            self.repo / "development/selection.json", self.selection_value
        )
        self.repository_state = {
            "clean": True,
            "commit": "a" * 40,
            "parents": ["b" * 40],
            "tree": "c" * 40,
            "commit_time_utc": "2026-07-25T00:00:00+00:00",
            "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
            "stage3_is_ancestor": True,
        }
        self.environment = {
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
        self.seed_policy: object = None

    def dependencies(
        self,
        *,
        repository_inspector: Callable[[Path], dict[str, Any]] | None = None,
        t1_transaction_verifier: Callable[[list[dict[str, Any]]], dict[str, Any]]
        | None = None,
    ) -> FreezeDependencies:
        return FreezeDependencies(
            repository_inspector=repository_inspector
            or (lambda _: dict(self.repository_state)),
            environment_collector=lambda: dict(self.environment),
            release_paths=dict(self.release_paths),
            python_executable=Path(sys.executable).resolve(),
            t1_transaction_verifier=t1_transaction_verifier
            or (lambda executions: json.loads(self.t1_evidence.read_text())[
                "deterministic_evidence"
            ]["cumulative_exact"]),
        )

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            apartment_config=self.config_paths["apartment"],
            office_config=self.config_paths["office"],
            selection=self.selection,
            t1_evidence=self.t1_evidence,
            t4_evidence=self.t4_evidence,
            seed_policy=self.seed_policy,
            output=self.output,
        )

    def run(self, **dependency_overrides: Any) -> dict[str, Any]:
        dependencies = self.dependencies(**dependency_overrides)
        return freeze(self.args(), repo_root=self.repo, dependencies=dependencies)

    def rewrite_selection(self, mutate: Callable[[dict[str, Any]], None]) -> None:
        mutate(self.selection_value)
        _write_json(self.selection, self.selection_value)

    def rewrite_config(self, scene: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        mutate(self.configs[scene])
        _write_json(self.config_paths[scene], self.configs[scene])


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_freeze_publishes_complete_runner_compatible_v2_contract(
    fixture: Fixture,
) -> None:
    manifest = fixture.run()

    assert manifest["freeze_id"] == "oviv2-tessecd-v2"
    assert manifest["status"] == "FROZEN"
    assert manifest["repository"] == fixture.repository_state
    assert manifest["algorithm"]["sha256"] == fixture.configs["apartment"][
        "algorithm_hash"
    ]
    assert manifest["selection"] == {
        "artifact": _record(fixture.selection),
        "development_scene": "apartment",
        "selected_config_sha256": _json_hash(fixture.configs["apartment"]),
        "selected_algorithm_hash": fixture.configs["apartment"]["algorithm_hash"],
    }
    assert set(manifest["scenes"]) == set(SCENES)
    assert set(manifest["shared_bindings"]) == {
        "input_manifest",
        "schedule",
        "occlusion_target_manifest",
    }
    assert manifest["office_pre_freeze_audit"] == {
        "selection_scene": "apartment",
        "metric_sources_found": [],
        "office_outputs_read": False,
    }
    assert set(manifest["output_roots"]) == {
        "apartment_run1",
        "apartment_run2",
        "office_seed_0",
    }
    assert len(set(manifest["output_roots"].values())) == 3
    assert set(manifest["evidence"]) == {"t1", "t4"}
    assert manifest["seed_policy"] == {
        "behavior": "deterministic",
        "seeds": [0],
        "sha256": _json_hash({"behavior": "deterministic", "seeds": [0]}),
    }
    authorization = manifest["office_authorizations"]["office_seed_0"]
    assert authorization["authorization_id"] == "office_seed_0"
    assert authorization["scene"] == "office"
    assert authorization["seed"] == 0
    assert authorization["config_sha256"] == _json_hash(fixture.configs["office"])
    assert authorization["algorithm_hash"] == fixture.configs["office"]["algorithm_hash"]
    assert authorization["output_root"] == manifest["output_roots"]["office_seed_0"]
    assert authorization["t1_root_sha256"] == manifest["evidence"]["t1"]["root_sha256"]
    assert authorization["t4_root_sha256"] == manifest["evidence"]["t4"]["root_sha256"]
    assert manifest["environment"] == fixture.environment
    assert set(manifest["models"]) == {"frontend", "dense"}
    assert set(manifest["release_bindings"]) == set(fixture.release_paths)
    assert json.loads(fixture.output.read_text(encoding="utf-8")) == manifest
    assert not list(fixture.output.parent.glob(".*.tmp-*"))


def test_freeze_requires_t1_and_t4_evidence(fixture: Fixture) -> None:
    for field in ("t1_evidence", "t4_evidence"):
        args = fixture.args()
        setattr(args, field, None)
        with pytest.raises(ValueError, match="T1 and T4 evidence are required"):
            freeze(args, repo_root=fixture.repo, dependencies=fixture.dependencies())
        assert not fixture.output.exists()


def test_freeze_recomputes_task11_exact_transaction(fixture: Fixture) -> None:
    with pytest.raises(ValueError, match="T1 exact transaction recomputation"):
        fixture.run(t1_transaction_verifier=lambda _: {"format": "forged"})


@pytest.mark.parametrize("kind", ["t1_gate", "t1_root", "t4_gate", "t4_root", "t4_source"])
def test_freeze_recomputes_t1_t4_evidence_and_requires_selected_pass(
    fixture: Fixture, kind: str
) -> None:
    if kind.startswith("t1"):
        payload = json.loads(fixture.t1_evidence.read_text())
        if kind == "t1_gate":
            payload["deterministic_evidence"]["gates"]["t1_exact"]["status"] = "FAIL"
        else:
            payload["deterministic_evidence"]["cumulative_exact"]["profiles"]["a4"][
                "cumulative_root_sha256"
            ] = "0" * 64
        _write_json(fixture.t1_evidence, payload)
    else:
        payload = json.loads(fixture.t4_evidence.read_text())
        if kind == "t4_gate":
            payload["candidates"]["a3"]["gates"]["query_p95_ms"] = False
            unhashed = dict(payload)
            unhashed.pop("root_sha256")
            payload["root_sha256"] = _json_hash(unhashed)
        elif kind == "t4_root":
            payload["root_sha256"] = "0" * 64
        else:
            fixture.t4_protocol.write_text("tampered\n", encoding="utf-8")
        _write_json(fixture.t4_evidence, payload)

    with pytest.raises(ValueError):
        fixture.run()
    assert not fixture.output.exists()


def test_freeze_stochastic_policy_requires_exact_five_seed_authorizations(
    fixture: Fixture,
) -> None:
    seeds = [17, 29, 43, 71, 101]
    fixture.seed_policy = {
        "behavior": "stochastic",
        "seeds": seeds,
    }

    manifest = fixture.run()

    assert manifest["seed_policy"]["seeds"] == seeds
    assert set(manifest["office_authorizations"]) == {
        f"office_seed_{seed}" for seed in seeds
    }
    assert set(manifest["output_roots"]) == {
        "apartment_run1", "apartment_run2", *(f"office_seed_{seed}" for seed in seeds)
    }


def test_freeze_rejects_nonregistered_seed_list(fixture: Fixture) -> None:
    fixture.seed_policy = {
        "behavior": "stochastic",
        "seeds": [17],
    }

    with pytest.raises(ValueError, match="pre-registered seeds"):
        fixture.run()


def test_frozen_manifest_loads_through_the_existing_v2_runner(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.run_oviv2_tesse_cd_v2 as runner

    manifest = fixture.run()
    monkeypatch.setattr(runner, "REPO_ROOT", fixture.repo)
    monkeypatch.setattr(
        runner,
        "V2_RELEASE_EXPECTED_PATHS",
        fixture.release_paths,
    )
    monkeypatch.setattr(
        runner,
        "_repository_provenance",
        lambda: {
            "repository_commit": fixture.repository_state["commit"],
            "repository_tree": fixture.repository_state["tree"],
            "dirty_state_digest": hashlib.sha256(b"").hexdigest(),
        },
    )
    config_path = fixture.config_paths["apartment"]
    context = runner.load_v2_frozen_run_context(
        fixture.output,
        run_slot="apartment_run1",
        source_config=config_path,
        source_config_bytes=config_path.read_bytes(),
        config=fixture.configs["apartment"],
        destination=Path(manifest["output_roots"]["apartment_run1"]),
    )

    assert context.frozen_run_identity["freeze_id"] == "oviv2-tessecd-v2"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("manifest_id", "oviv2-tessecd-v1"),
        lambda value: value.__setitem__("protocol_id", "oviv2-tessecd-v1"),
        lambda value: value.__setitem__("development_scene", "office"),
        lambda value: value.__setitem__("scenes_read", ["apartment", "office"]),
        lambda value: value.__setitem__("office_results_read", True),
        lambda value: value.__setitem__("unknown", 1),
        lambda value: value["t4_ledger"].pop(),
        lambda value: value["t4_ledger"].append(
            dict(value["t4_ledger"][0])
        ),
        lambda value: value["t4_ledger"][0].__setitem__(
            "candidate_id", "a5"
        ),
        lambda value: value["t4_ledger"][0].__setitem__(
            "failed_gates", ["query_p95_ms"]
        ),
        lambda value: value["t4_ledger"][0].__setitem__("unknown", 1),
    ],
)
def test_freeze_rejects_v1_office_undeclared_or_surplus_selection_evidence(
    fixture: Fixture, mutation: Callable[[dict[str, Any]], None]
) -> None:
    fixture.rewrite_selection(mutation)

    with pytest.raises(ValueError):
        fixture.run()

    assert not fixture.output.parent.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("selected_config_sha256", "0" * 64),
        lambda value: value.__setitem__("algorithm_hash", "0" * 64),
        lambda value: value["selected_config"].__setitem__(
            "visibility_depth_tolerance_m", 0.123
        ),
        lambda value: value.__setitem__("selected_candidate_id", "a2"),
        lambda value: value["t4_ledger"][0].__setitem__("selected", False),
        lambda value: value["t4_matrix"].__setitem__(
            "sha256", "0" * 64
        ),
    ],
)
def test_freeze_rejects_selection_or_ledger_drift(
    fixture: Fixture, mutation: Callable[[dict[str, Any]], None]
) -> None:
    fixture.rewrite_selection(mutation)

    with pytest.raises(ValueError):
        fixture.run()


def test_freeze_recomputes_t4_bounds_despite_forged_pass_and_fresh_hashes(
    fixture: Fixture,
) -> None:
    metric = "total_runtime_s_per_frame"
    protocol = json.loads(fixture.t4_protocol.read_text())
    metric_path = Path(protocol["candidates"]["a3"]["metric_sources"][metric]["path"])
    metric_payload = json.loads(metric_path.read_text())
    metric_payload["value"] = 999.0
    _write_json(metric_path, metric_payload)
    protocol["candidates"]["a3"]["metric_sources"][metric] = _record(metric_path)
    _write_json(fixture.t4_protocol, protocol)

    matrix = json.loads(fixture.t4_evidence.read_text())
    matrix["protocol"] = _record(fixture.t4_protocol)
    matrix["candidates"]["a3"]["metrics"][metric] = 999.0
    matrix.pop("root_sha256")
    matrix["root_sha256"] = _json_hash(matrix)
    _write_json(fixture.t4_evidence, matrix)

    fixture.selection_value["t4_protocol"] = _record(fixture.t4_protocol)
    fixture.selection_value["t4_matrix"] = _record(fixture.t4_evidence)
    fixture.selection_value["t4_root_sha256"] = matrix["root_sha256"]
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="stale gates|ledger disagrees"):
        fixture.run()


@pytest.mark.parametrize("scene", SCENES)
def test_freeze_rejects_config_drift_and_surplus(fixture: Fixture, scene: str) -> None:
    fixture.rewrite_config(scene, lambda value: value.__setitem__("unknown", 1))

    with pytest.raises(ValueError):
        fixture.run()


def test_freeze_rejects_algorithm_or_cross_scene_config_drift(fixture: Fixture) -> None:
    fixture.rewrite_config(
        "office", lambda value: value.__setitem__("algorithm_hash", "0" * 64)
    )

    with pytest.raises(ValueError):
        fixture.run()


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite"])
def test_freeze_rejects_duplicate_or_nonfinite_json(fixture: Fixture, kind: str) -> None:
    if kind == "duplicate":
        fixture.selection.write_text(
            '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
        )
    else:
        fixture.selection.write_text('{"metric":NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        fixture.run()


def test_freeze_rejects_duplicate_json_in_bound_shared_manifest(
    fixture: Fixture,
) -> None:
    fixture.shared["schedule_manifest"].write_text(
        '{"schema_version":2,"schema_version":2}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        fixture.run()


@pytest.mark.parametrize("role", ["apartment_config", "selection"])
def test_freeze_rejects_symlink_or_nonregular_inputs(
    fixture: Fixture, role: str
) -> None:
    args = fixture.args()
    source = Path(getattr(args, role))
    replacement = source.with_suffix(".replacement")
    source.rename(replacement)
    if role == "selection":
        source.mkdir()
    else:
        source.symlink_to(replacement)

    with pytest.raises(ValueError):
        freeze(args, repo_root=fixture.repo, dependencies=fixture.dependencies())


@pytest.mark.parametrize("clean", [False, 0])
def test_freeze_requires_an_exact_clean_repository(fixture: Fixture, clean: object) -> None:
    state = dict(fixture.repository_state)
    state["clean"] = clean

    with pytest.raises(ValueError):
        fixture.run(repository_inspector=lambda _: state)


def test_freeze_rejects_repository_change_during_freeze(fixture: Fixture) -> None:
    changed = dict(fixture.repository_state, tree="e" * 40)
    states = iter((fixture.repository_state, changed))

    with pytest.raises(ValueError, match="repository"):
        fixture.run(repository_inspector=lambda _: dict(next(states)))

    assert not fixture.output.exists()


@pytest.mark.parametrize("occupied", ["output", "run", "staging"])
def test_freeze_is_no_overwrite_and_requires_new_output_roots(
    fixture: Fixture, occupied: str
) -> None:
    if occupied == "output":
        _write_bytes(fixture.output, b"occupied")
    elif occupied == "run":
        (fixture.output.parent / "office/run2").mkdir(parents=True)
    else:
        _write_bytes(fixture.output.parent / ".freeze.json.tmp-attacker", b"occupied")

    with pytest.raises((FileExistsError, ValueError)):
        fixture.run()

    if occupied == "output":
        assert fixture.output.read_bytes() == b"occupied"


def test_freeze_rejects_output_alias_and_parent_child_overlap(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.freeze_oviv2_tesse_cd_v2 as module

    monkeypatch.setattr(
        module,
        "_output_roots",
        lambda _, seeds: {
            "apartment_run1": fixture.output.parent / "runs",
            "apartment_run2": fixture.output.parent / "runs/child",
            "office_seed_0": fixture.config_paths["office"],
        },
    )

    with pytest.raises(ValueError):
        fixture.run()


def test_freeze_rolls_back_staging_when_atomic_publication_fails(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "link", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("injected link failure")
    ))

    with pytest.raises(OSError, match="injected"):
        fixture.run()

    assert not fixture.output.exists()
    assert not list(fixture.output.parent.glob(".*.tmp-*"))


def test_freeze_detects_input_swap_during_atomic_publication(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link

    def mutate_then_link(source: object, destination: object, *args: Any, **kwargs: Any):
        fixture.selection.write_bytes(fixture.selection.read_bytes() + b" ")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", mutate_then_link)

    with pytest.raises(ValueError, match="changed"):
        fixture.run()

    assert not fixture.output.exists()
    assert not list(fixture.output.parent.glob(".*.tmp-*"))


def test_freeze_rolls_back_when_output_parent_is_swapped(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link
    displaced = fixture.repo / "displaced-formal"

    def swap_parent_then_link(
        source: object, destination: object, *args: Any, **kwargs: Any
    ):
        fixture.output.parent.rename(displaced)
        fixture.output.parent.mkdir()
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", swap_parent_then_link)

    with pytest.raises((OSError, ValueError)):
        fixture.run()

    assert not fixture.output.exists()
    assert not (displaced / fixture.output.name).exists()
    assert not list(displaced.glob(".*.tmp-*"))


def test_freeze_rejects_office_result_evidence_in_config(fixture: Fixture) -> None:
    fixture.rewrite_config(
        "office",
        lambda value: value.__setitem__(
            "dataset_root", str(fixture.repo / "development/office/results")
        ),
    )
    fixture.configs["office"]["algorithm_hash"] = canonical_algorithm_hash(
        fixture.configs["office"]
    )
    _write_json(fixture.config_paths["office"], fixture.configs["office"])

    with pytest.raises(ValueError, match="Office"):
        fixture.run()


def test_freeze_rejects_self_consistent_office_selection_source(
    fixture: Fixture,
) -> None:
    _write_json(fixture.t4_protocol, {"schema_version": 1, "scene": "office"})
    protocol = _record(fixture.t4_protocol)
    matrix = json.loads(fixture.t4_evidence.read_text())
    matrix["protocol"] = protocol
    matrix.pop("root_sha256")
    matrix["root_sha256"] = _json_hash(matrix)
    _write_json(fixture.t4_evidence, matrix)
    fixture.selection_value["t4_protocol"] = protocol
    fixture.selection_value["t4_matrix"] = _record(fixture.t4_evidence)
    fixture.selection_value["t4_root_sha256"] = matrix["root_sha256"]
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="Office"):
        fixture.run()


def test_freeze_rejects_generic_payload_at_office_metric_path(
    fixture: Fixture,
) -> None:
    path = _write_json(
        fixture.repo / "development/office_metrics.json",
        {"schema_version": 1, "status": "PASS"},
    )
    protocol = _record(path)
    matrix = json.loads(fixture.t4_evidence.read_text())
    matrix["protocol"] = protocol
    matrix.pop("root_sha256")
    matrix["root_sha256"] = _json_hash(matrix)
    _write_json(fixture.t4_evidence, matrix)
    fixture.selection_value["t4_protocol"] = protocol
    fixture.selection_value["t4_matrix"] = _record(fixture.t4_evidence)
    fixture.selection_value["t4_root_sha256"] = matrix["root_sha256"]
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="Office"):
        fixture.run()


def test_freeze_rejects_reused_final_selection_source_record(
    fixture: Fixture,
) -> None:
    fixture.selection_value["t4_protocol"] = fixture.selection_value["shortlist"]
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="distinct|binding"):
        fixture.run()
