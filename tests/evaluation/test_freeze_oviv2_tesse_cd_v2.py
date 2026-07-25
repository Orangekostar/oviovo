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

    def dependencies(
        self,
        *,
        repository_inspector: Callable[[Path], dict[str, Any]] | None = None,
    ) -> FreezeDependencies:
        return FreezeDependencies(
            repository_inspector=repository_inspector
            or (lambda _: dict(self.repository_state)),
            environment_collector=lambda: dict(self.environment),
            release_paths=dict(self.release_paths),
            python_executable=Path(sys.executable).resolve(),
        )

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            apartment_config=self.config_paths["apartment"],
            office_config=self.config_paths["office"],
            selection=self.selection,
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
        "office_run1",
        "office_run2",
    }
    assert len(set(manifest["output_roots"].values())) == 4
    assert manifest["environment"] == fixture.environment
    assert set(manifest["models"]) == {"frontend", "dense"}
    assert set(manifest["release_bindings"]) == set(fixture.release_paths)
    assert json.loads(fixture.output.read_text(encoding="utf-8")) == manifest
    assert not list(fixture.output.parent.glob(".*.tmp-*"))


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
        lambda value: value["rejection_ledger"].pop(),
        lambda value: value["rejection_ledger"].append(
            dict(value["rejection_ledger"][0])
        ),
        lambda value: value["rejection_ledger"][0].__setitem__(
            "candidate_id", "a5"
        ),
        lambda value: value["rejection_ledger"][0]["sources"].__setitem__(
            "office_result", value["rejection_ledger"][0]["sources"]["run_manifest"]
        ),
        lambda value: value["rejection_ledger"][0].__setitem__("unknown", 1),
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
        lambda value: value["rejection_ledger"][3].__setitem__("selected", False),
        lambda value: value["result_files"][0].__setitem__(
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
        lambda _: {
            "apartment_run1": fixture.output.parent / "runs",
            "apartment_run2": fixture.output.parent / "runs/child",
            "office_run1": fixture.config_paths["office"],
            "office_run2": fixture.output.parent / "other",
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
    source = fixture.selection_value["rejection_ledger"][0]["sources"][
        "official_metrics"
    ]
    path = Path(source["path"])
    _write_json(path, {"schema_version": 1, "scene": "office"})
    fixture.selection_value["rejection_ledger"][0]["sources"][
        "official_metrics"
    ] = _record(path)
    result_path = Path(fixture.selection_value["result_files"][0]["path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["sources"] = fixture.selection_value["rejection_ledger"][0]["sources"]
    _write_json(result_path, result)
    fixture.selection_value["result_files"][0] = {
        "candidate_id": "a0",
        **_record(result_path),
    }
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="Office"):
        fixture.run()


def test_freeze_rejects_generic_payload_at_office_metric_path(
    fixture: Fixture,
) -> None:
    candidate = "a0"
    ledger = fixture.selection_value["rejection_ledger"][0]
    path = _write_json(
        fixture.repo / "development/office_metrics.json",
        {"schema_version": 1, "status": "PASS"},
    )
    ledger["sources"]["official_metrics"] = _record(path)
    result_record = fixture.selection_value["result_files"][0]
    result_path = Path(result_record["path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["sources"] = ledger["sources"]
    _write_json(result_path, result)
    fixture.selection_value["result_files"][0] = {
        "candidate_id": candidate,
        **_record(result_path),
    }
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="Office|candidate evidence root"):
        fixture.run()


def test_freeze_rejects_reused_result_file_across_candidates(
    fixture: Fixture,
) -> None:
    first = fixture.selection_value["result_files"][0]
    fixture.selection_value["result_files"] = [
        {"candidate_id": candidate, **{key: first[key] for key in ("path", "sha256", "byte_count")}}
        for candidate in CANDIDATES
    ]
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="result|candidate"):
        fixture.run()


@pytest.mark.parametrize("reuse", ["cross_candidate", "cross_role"])
def test_freeze_rejects_reused_candidate_specific_source(
    fixture: Fixture, reuse: str
) -> None:
    target_index = 1 if reuse == "cross_candidate" else 0
    target = fixture.selection_value["rejection_ledger"][target_index]
    if reuse == "cross_candidate":
        replacement = fixture.selection_value["rejection_ledger"][0]["sources"][
            "run_manifest"
        ]
        role = "run_manifest"
    else:
        replacement = target["sources"]["common_v2_summary"]
        role = "official_metrics"
    target["sources"][role] = replacement
    result_record = fixture.selection_value["result_files"][target_index]
    result_path = Path(result_record["path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["sources"] = target["sources"]
    _write_json(result_path, result)
    fixture.selection_value["result_files"][target_index] = {
        "candidate_id": target["candidate_id"],
        **_record(result_path),
    }
    _write_json(fixture.selection, fixture.selection_value)

    with pytest.raises(ValueError, match="source|candidate evidence root|reused"):
        fixture.run()
