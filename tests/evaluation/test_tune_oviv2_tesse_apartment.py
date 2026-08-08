from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import scripts.evaluation.tune_oviv2_tesse_apartment as tuning_module
from scripts.evaluation.tune_oviv2_tesse_apartment import (
    CandidateExecution,
    TUNABLE_FIELDS,
    build_candidate_configs,
    run_tuning,
    selection_key,
)
from scripts.evaluation.run_oviv2_tesse_cd import algorithm_hash


ROOT = Path(__file__).resolve().parents[2]
ALIASES = ROOT / "configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml"
TUNING_SCRIPT = ROOT / "scripts/evaluation/tune_oviv2_tesse_apartment.py"


def _base_config() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/oviv2_tesse_cd_apartment_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_base_config(path: Path) -> Path:
    _write_json(path, _base_config())
    return path


@pytest.fixture
def isolated_freeze_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    bindings = {
        "shared": {"input_manifest": {"sha256": "a" * 64}},
        "scenes": {
            "apartment": {"database": {"sha256": "b" * 64}},
            "office": {"database": {"sha256": "c" * 64}},
        },
    }
    monkeypatch.setattr(
        tuning_module,
        "_build_freeze_bindings",
        lambda *args, **kwargs: bindings,
    )


def _fake_executor(
    *,
    target_sha256: str = "d" * 64,
    bad_candidate: str | None = None,
    bad_kind: str | None = None,
    concurrency: dict[str, object] | None = None,
):
    def execute(item: CandidateExecution) -> None:
        if concurrency is not None:
            lock = concurrency["lock"]
            assert isinstance(lock, type(threading.Lock()))
            with lock:
                concurrency["active"] = int(concurrency["active"]) + 1
                concurrency["maximum"] = max(
                    int(concurrency["maximum"]), int(concurrency["active"])
                )
            time.sleep(0.005)
        try:
            if bad_kind == "missing" and item.candidate.candidate_id == bad_candidate:
                return
            config = item.candidate.config
            run_manifest = {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "mode": "causal_checkpoints",
                "scene": "apartment",
                "algorithm_hash": config["algorithm_hash"],
                "config": {
                    "sha256": item.candidate.config_sha256,
                    "byte_count": len(item.candidate.config_bytes),
                },
                "maintenance_parameters": {
                    field: config[field] for field in sorted(TUNABLE_FIELDS)
                },
            }
            _write_json(item.artifact_root / "run/run_manifest.json", run_manifest)
            index = int(item.candidate.candidate_id.rsplit("-", 1)[1])
            metrics = {
                "current_miou": 1.0 if index == 5 else 0.5,
                "ghost_rate": 0.01 + index / 1000.0,
                "background_f5": 0.8,
                "recovery_frames": 50.0,
            }
            status = "PASS"
            scene = "apartment"
            summary_target_sha = target_sha256
            if item.candidate.candidate_id == bad_candidate:
                if bad_kind == "status":
                    status = "FAIL"
                elif bad_kind == "scene":
                    scene = "office"
                elif bad_kind == "nonfinite":
                    metrics["current_miou"] = math.inf
                elif bad_kind == "target":
                    summary_target_sha = "e" * 64
            summary = {
                "schema_version": 1,
                "manifest_id": "tesse_cd_common_v2_scene_summary",
                "dataset": "TESSE-CD",
                "protocol": "tesse_cd_common_v2",
                "status": status,
                "method": "OVIV2",
                "mode": "causal_checkpoints",
                "scene": scene,
                "metrics": metrics,
                "sources": {
                    "target_manifest": {
                        "path": "unread-target-manifest.json",
                        "sha256": summary_target_sha,
                        "byte_count": 1,
                    }
                },
            }
            summary_path = item.artifact_root / "evaluation/summary.json"
            if bad_kind == "nonfinite" and item.candidate.candidate_id == bad_candidate:
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(
                    json.dumps(summary, sort_keys=True).replace("Infinity", "NaN") + "\n",
                    encoding="utf-8",
                )
            else:
                _write_json(summary_path, summary)
        finally:
            if concurrency is not None:
                lock = concurrency["lock"]
                assert isinstance(lock, type(threading.Lock()))
                with lock:
                    concurrency["active"] = int(concurrency["active"]) - 1

    return execute


def test_builds_exact_predeclared_grid_from_apartment_stage3_base() -> None:
    base = _base_config()

    candidates = build_candidate_configs(base)

    assert len(candidates) == 18
    assert len({candidate.config_sha256 for candidate in candidates}) == 18
    assert [candidate.candidate_id for candidate in candidates] == [
        f"candidate-{index:02d}" for index in range(18)
    ]
    assert {
        tuple(candidate.config[field] for field in sorted(TUNABLE_FIELDS))
        for candidate in candidates
    } == {
        (absence, ownership, tolerance)
        for absence in (0.5, 1.0)
        for ownership in (0.000001, 0.5, 1.0)
        for tolerance in (0.05, 0.10, 0.15)
    }
    for candidate in candidates:
        changed = {
            key
            for key in base
            if candidate.config.get(key) != base.get(key)
        }
        assert changed <= TUNABLE_FIELDS | {"algorithm_hash"}
        assert candidate.config["scene"] == "apartment"
        assert candidate.config["stage3_lineage_commit"].startswith("47962fb")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"scene": "office"}, "Apartment Stage3"),
        ({"stage3_lineage_commit": "0" * 40}, "Apartment Stage3"),
        ({"implementation": "route3-surface-observation-v5"}, "forbidden"),
    ],
)
def test_rejects_non_apartment_or_non_stage3_base(
    mutation: dict[str, object],
    match: str,
) -> None:
    base = {**_base_config(), **mutation}

    with pytest.raises(ValueError, match=match):
        build_candidate_configs(base)


def test_rejects_self_consistent_change_outside_tuning_grid() -> None:
    base = _base_config()
    base["association_geometry_weight"] = 0.99
    base["algorithm_hash"] = algorithm_hash(base)

    with pytest.raises(ValueError, match="canonical Stage3 base"):
        build_candidate_configs(base)


def test_commands_mode_is_canonical_and_does_not_read_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _write_base_config(tmp_path / "base.json")
    target = tmp_path / "must-not-be-read.json"
    target.write_text("target bytes\n", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == target:
            raise AssertionError("tuning orchestrator read GT target")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    output = tmp_path / "commands"

    result = run_tuning(
        base_config=base,
        output=output,
        target_manifest=target,
        aliases=tmp_path / "aliases.yaml",
        label_space=tmp_path / "labels.yaml",
        mode="commands",
    )

    assert result == output / "commands.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "COMMANDS"
    assert payload["scene"] == "apartment"
    assert payload["candidate_count"] == 18
    assert payload["max_parallel"] == 3
    assert len(payload["candidates"]) == 18
    assert all(len(item["commands"]) == 3 for item in payload["candidates"])
    for item in payload["candidates"]:
        for command in item["commands"]:
            assert Path(command[0]).is_absolute()
            assert Path(command[0]).is_file()
            assert Path(command[1]).is_absolute()
            assert Path(command[1]).is_file()
    assert len(list((output / "configs").glob("candidate-*.json"))) == 18
    serialized = result.read_text(encoding="utf-8").lower()
    assert "ground_truth" not in serialized


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "dry"

    payload = run_tuning(
        base_config=_write_base_config(tmp_path / "base.json"),
        output=output,
        target_manifest=tmp_path / "target.json",
        aliases=ALIASES,
        label_space=tmp_path / "labels.yaml",
        mode="run",
        dry_run=True,
    )

    assert isinstance(payload, dict)
    assert payload["status"] == "DRY_RUN"
    assert payload["candidate_count"] == 18
    assert not output.exists()


def test_direct_script_help_runs_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(TUNING_SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--base-config" in completed.stdout


def test_direct_script_dry_run_and_commands_work_outside_repository(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(TUNING_SCRIPT),
            "--base-config",
            str(ROOT / "configs/oviv2_tesse_cd_apartment_v1.json"),
            "--office-config",
            str(ROOT / "configs/oviv2_tesse_cd_office_v1.json"),
            "--output",
            str(output),
            "--target-manifest",
            str(tmp_path / "not-read-target.json"),
            "--aliases",
            str(ALIASES),
            "--label-space",
            str(tmp_path / "not-read-labels.yaml"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_RUN"
    assert payload["candidate_count"] == 18
    assert not output.exists()
    for candidate in payload["candidates"]:
        for command in candidate["commands"]:
            assert command[0] == str(Path(sys.executable).absolute())
            assert Path(command[1]).is_absolute()
            assert Path(command[1]).is_file()
            assert "-m" not in command


def test_run_batches_at_most_three_and_writes_deterministic_selection(
    tmp_path: Path,
    isolated_freeze_bindings: None,
) -> None:
    concurrency: dict[str, object] = {
        "lock": threading.Lock(),
        "active": 0,
        "maximum": 0,
    }
    output = tmp_path / "run"

    result = run_tuning(
        base_config=_write_base_config(tmp_path / "base.json"),
        output=output,
        target_manifest=tmp_path / "target.json",
        aliases=ALIASES,
        label_space=tmp_path / "labels.yaml",
        max_parallel=3,
        execute_candidate=_fake_executor(concurrency=concurrency),
    )

    assert result == output / "selection.json"
    assert concurrency["maximum"] == 3
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["method"] == "OVIV2"
    assert payload["scene"] == "apartment"
    assert payload["candidate_count"] == 18
    assert payload["selected_candidate_id"] == "candidate-05"
    assert len(payload["selected_config_sha256"]) == 64
    assert payload["common_v2_target_manifest_sha256"] == "d" * 64
    assert payload["common_target_manifest"] == {
        "path": "unread-target-manifest.json",
        "sha256": "d" * 64,
        "byte_count": 1,
    }
    assert payload["parameter_grid"] == {
        "visibility_depth_tolerance_m": [0.05, 0.10, 0.15],
        "absence_negative_support": [0.5, 1.0],
        "ownership_min_net_support": [0.000001, 0.5, 1.0],
    }
    assert set(payload["freeze_bindings"]) == {"shared", "scenes"}
    assert set(payload["freeze_bindings"]["scenes"]) == {"apartment", "office"}
    assert len(payload["candidates"]) == 18
    for record in payload["candidates"]:
        assert record["schema_version"] == 1
        assert record["status"] == "PASS"
        assert record["scene"] == "apartment"
        assert len(record["config_sha256"]) == 64
        assert record["common_target_manifest_sha256"] == "d" * 64
        assert set(record["metrics"]) == {
            "current_miou",
            "ghost_rate",
            "background_f5_cm",
            "recovery_frames",
        }
        assert len(record["run_identity"]["sha256"]) == 64
        assert len(record["evaluator_summary"]["sha256"]) == 64
        summary_path = Path(record["summary"]["path"])
        assert summary_path.is_file()
        assert json.loads(summary_path.read_text(encoding="utf-8")) == {
            key: value for key, value in record.items() if key != "summary"
        }
    assert result.read_bytes() == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    second = tmp_path / "run-second"
    second_result = run_tuning(
        base_config=tmp_path / "base.json",
        output=second,
        target_manifest=tmp_path / "target.json",
        aliases=ALIASES,
        label_space=tmp_path / "labels.yaml",
        max_parallel=3,
        execute_candidate=_fake_executor(),
    )
    second_payload = json.loads(second_result.read_text(encoding="utf-8"))
    assert payload["selected_config_sha256"] == second_payload[
        "selected_config_sha256"
    ]
    assert payload["selected_metrics"] == second_payload["selected_metrics"]


def test_evaluator_summary_is_parsed_and_hashed_from_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_freeze_bindings: None,
) -> None:
    original_read_bytes = Path.read_bytes
    read_counts: dict[Path, int] = {}

    def swap_second_read(path: Path) -> bytes:
        if path.name == "summary.json" and path.parent.name == "evaluation":
            read_counts[path] = read_counts.get(path, 0) + 1
            if read_counts[path] == 2:
                return original_read_bytes(path) + b"swapped"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_second_read)
    output = tmp_path / "run"
    selection_path = run_tuning(
        base_config=_write_base_config(tmp_path / "base.json"),
        output=output,
        target_manifest=tmp_path / "target.json",
        aliases=ALIASES,
        label_space=tmp_path / "labels.yaml",
        execute_candidate=_fake_executor(),
    )
    assert isinstance(selection_path, Path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for candidate in selection["candidates"]:
        summary_path = output / candidate["evaluator_summary"]["path"]
        assert candidate["evaluator_summary"]["sha256"] == hashlib.sha256(
            original_read_bytes(summary_path)
        ).hexdigest()


def test_final_output_is_not_visible_until_sweep_succeeds(
    tmp_path: Path,
    isolated_freeze_bindings: None,
) -> None:
    output = tmp_path / "run"
    fake = _fake_executor()

    def assert_hidden(item: CandidateExecution) -> None:
        assert not output.exists()
        fake(item)

    result = run_tuning(
        base_config=_write_base_config(tmp_path / "base.json"),
        output=output,
        target_manifest=tmp_path / "target.json",
        aliases=ALIASES,
        label_space=tmp_path / "labels.yaml",
        execute_candidate=assert_hidden,
    )

    assert result == output / "selection.json"
    assert output.is_dir()
    assert list(tmp_path.glob(".run.staging-*")) == []


def test_rejects_candidate_directory_symlink_without_writing_outside(
    tmp_path: Path,
    isolated_freeze_bindings: None,
) -> None:
    output = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    fake = _fake_executor()

    def symlink_first(item: CandidateExecution) -> None:
        if item.candidate.candidate_id == "candidate-00":
            item.artifact_root.symlink_to(outside, target_is_directory=True)
        fake(item)

    with pytest.raises(ValueError, match="symlink"):
        run_tuning(
            base_config=_write_base_config(tmp_path / "base.json"),
            output=output,
            target_manifest=tmp_path / "target.json",
            aliases=ALIASES,
            label_space=tmp_path / "labels.yaml",
            execute_candidate=symlink_first,
        )

    assert not output.exists()
    assert not (outside / "candidate_summary.json").exists()


@pytest.mark.parametrize(
    ("bad_kind", "match"),
    [
        ("status", "PASS"),
        ("scene", "Apartment"),
        ("nonfinite", "non-finite"),
        ("target", "target manifest hash"),
        ("missing", "missing"),
    ],
)
def test_rejects_invalid_or_incomplete_candidate_results_atomically(
    tmp_path: Path,
    bad_kind: str,
    match: str,
    isolated_freeze_bindings: None,
) -> None:
    output = tmp_path / "run"

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        run_tuning(
            base_config=_write_base_config(tmp_path / "base.json"),
            output=output,
            target_manifest=tmp_path / "target.json",
            aliases=ALIASES,
            label_space=tmp_path / "labels.yaml",
            execute_candidate=_fake_executor(
                bad_candidate="candidate-17",
                bad_kind=bad_kind,
            ),
        )

    assert not output.exists()


def test_rejects_output_reuse_without_touching_existing_data(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("owned\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_tuning(
            base_config=_write_base_config(tmp_path / "base.json"),
            output=output,
            target_manifest=tmp_path / "target.json",
            aliases=tmp_path / "aliases.yaml",
            label_space=tmp_path / "labels.yaml",
            mode="commands",
        )

    assert marker.read_text(encoding="utf-8") == "owned\n"


def test_selection_key_uses_predeclared_lexicographic_order() -> None:
    records = [
        {
            "metrics": {
                "current_miou": 0.8,
                "ghost_rate": 0.2,
                "background_f5_cm": 0.9,
                "recovery_frames": 1.0,
            },
            "config_sha256": "a" * 64,
        },
        {
            "metrics": {
                "current_miou": 0.8,
                "ghost_rate": 0.1,
                "background_f5_cm": 0.5,
                "recovery_frames": 100.0,
            },
            "config_sha256": "b" * 64,
        },
    ]

    assert min(records, key=selection_key) is records[1]
