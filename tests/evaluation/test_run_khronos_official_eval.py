from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluation.run_khronos_official_eval as official_eval
from scripts.evaluation.run_khronos_official_eval import (
    CANONICAL_TESSE_MANIFEST,
    build_evaluation_command,
    parse_args,
    patch_evaluation_config,
    run,
    validate_ground_truth_files,
    validate_khronos_run_status,
    validate_tesse_manifest,
    write_repeated_metrics,
)


RUN_ID = "oviv2-tessecd-v1"


def _source_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _write_valid_results(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "static_objects.csv").write_text(
        "Name,Query,AppearedTP,DisappearedTP,AppearedFP,DisappearedFP,"
        "AppearedFN,DisappearedFN,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
        "0,10,1,0,0,0,0,0,1,0,0\n",
        encoding="utf-8",
    )
    (root / "dynamic_objects.csv").write_text(
        "Name,Query,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
        "0,10,1,0,0\n",
        encoding="utf-8",
    )
    (root / "background_mesh.csv").write_text(
        "Name,Accuracy@0.2,Completeness@0.2\n0,1.0,1.0\n",
        encoding="utf-8",
    )
    return root


def _write_required_run_status(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    identity_extras: dict[str, object] | None = None,
) -> tuple[Path, dict[str, Path]]:
    required = {
        "final": root / "map/final.4dmap",
        "timestamps": root / "map/map_timestamps.json",
        "experiment": root / "map/experiment_log.txt",
        "bridge": root / "bridge_input/bridge_manifest.json",
        "build": root / "build_manifest.json",
        "config": root / "oviv2_tesse_cd_apartment_v1.json",
    }
    for name, path in required.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    monkeypatch.setattr(
        official_eval, "validate_temporal_bridge_manifest", lambda _: {}
    )
    monkeypatch.setattr(
        official_eval, "validate_build_manifest", lambda _: {}
    )
    status = root / "run_status.json"
    config = _source_record(required["config"])
    status.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "dataset": "TESSE-CD",
                "scene": "apartment",
                "method": "OVIV2",
                "mode": "causal_checkpoints",
                "bridge_mode": "temporal_checkpoints",
                "run_identity": {
                    "run_id": RUN_ID,
                    "config_sha256": config["sha256"],
                    **(identity_extras or {}),
                },
                "config": config,
                "sources": [_source_record(path) for path in required.values()],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return status, required


def _validated_run_status(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    status, _ = _write_required_run_status(root, monkeypatch)
    return status, validate_khronos_run_status(status, scene="apartment")


def test_patch_evaluation_config_binds_local_gt_paths() -> None:
    config = {
        "ground_truth_dsg_file": "/upstream/dsg.json",
        "ground_truth_background_file": "/upstream/background.ply",
        "gt_changes_file": "/upstream/changes.csv",
        "evaluation": {"object_evaluation": {"changes_file": "/upstream/changes.csv"}},
    }
    files = {
        "dsg": {"path": "/local/dsg.json"},
        "background_mesh": {"path": "/local/background.ply"},
        "changes": {"path": "/local/changes.csv"},
    }

    patched = patch_evaluation_config(config, files)

    assert patched["ground_truth_dsg_file"] == "/local/dsg.json"
    assert patched["ground_truth_background_file"] == "/local/background.ply"
    assert patched["gt_changes_file"] == "/local/changes.csv"
    assert patched["evaluation"]["object_evaluation"]["changes_file"] == "/local/changes.csv"
    assert config["ground_truth_dsg_file"] == "/upstream/dsg.json"


def test_ground_truth_validation_uses_only_patched_files(tmp_path: Path) -> None:
    files = {}
    for name in ("dsg", "background_mesh", "changes"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    files["dsg_with_mesh"] = {
        "path": str(tmp_path / "unused-missing.json"),
        "sha256": "0" * 64,
        "size_bytes": 1,
    }

    selected = validate_ground_truth_files(files)

    assert set(selected) == {"dsg", "background_mesh", "changes"}
    files["dsg"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ground-truth provenance"):
        validate_ground_truth_files(files)


def test_build_evaluation_command_uses_full_state_sequence() -> None:
    command = build_evaluation_command(
        workspace=Path("/ws"),
        map_dir=Path("/run/map"),
        config=Path("/run/evaluation/apartment.yaml"),
    )

    assert command == [
        "/ws/install/khronos_eval/lib/khronos_eval/evaluate_pipeline.sh",
        "/run/map",
        "/run/evaluation/apartment.yaml",
        "true",
        "false",
    ]


def test_parser_accepts_oviv2_causal_identity() -> None:
    args = parse_args(
        [
            "--workspace",
            "/ws",
            "--manifest",
            "/data/manifest.json",
            "--run-root",
            "/run",
            "--scene",
            "apartment",
            "--method",
            "OVIV2",
            "--mode",
            "causal_checkpoints",
        ]
    )

    assert args.method == "OVIV2"
    assert args.mode == "causal_checkpoints"

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--workspace", "/ws",
                "--manifest", "/data/manifest.json",
                "--run-root", "/run",
                "--scene", "apartment",
                "--method", "DUALMAP",
                "--mode", "causal_checkpoints",
            ]
        )


def test_repeated_metrics_records_consistent_unavailable_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        (results / name).write_text(f"{name}\n", encoding="utf-8")

    def unavailable(_: Path) -> None:
        raise ValueError("Khronos change F1 has no finite states")

    monkeypatch.setattr(
        official_eval, "summarize_khronos_official_metrics", unavailable
    )
    monkeypatch.setattr(
        official_eval,
        "summarize_khronos_official_metrics_partial",
        lambda _: {
            "state_count": 1,
            "metrics": {"object_f1": 1.0, "dynamic_f1": None, "change_f1": None},
            "unavailable": {
                "dynamic_f1": "upstream omitted dynamic states",
                "change_f1": "upstream omitted change states",
            },
        },
    )
    status_path, _ = _validated_run_status(tmp_path / "run", monkeypatch)
    summary = write_repeated_metrics(
        results_dir=results,
        scene="apartment",
        method="OVIV2",
        mode="causal_checkpoints",
        metrics_path=tmp_path / "official_metrics.json",
        repeat_path=tmp_path / "official_metrics.repeat.json",
        run_status_path=status_path,
    )

    assert summary["status"] == "UNAVAILABLE"
    assert summary["reason"] == "Khronos change F1 has no finite states"
    assert summary["repeat_reason"] == summary["reason"]
    assert summary["official_metrics_partial"]["sha256"] == summary[
        "official_metrics_partial_repeat"
    ]["sha256"]
    assert [Path(entry["path"]).name for entry in summary["sources"]] == [
        "static_objects.csv",
        "dynamic_objects.csv",
        "background_mesh.csv",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in summary["sources"])


def test_repeated_metrics_refuses_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        (results / name).write_text(f"{name}\n", encoding="utf-8")
    metrics = tmp_path / "official_metrics.json"
    metrics.write_text("preserve\n", encoding="utf-8")
    status_path, _ = _validated_run_status(tmp_path / "run", monkeypatch)

    with pytest.raises(FileExistsError, match="already exists"):
        write_repeated_metrics(
            results_dir=results,
            scene="apartment",
            method="OVIV2",
            mode="causal_checkpoints",
            metrics_path=metrics,
            repeat_path=tmp_path / "official_metrics.repeat.json",
            run_status_path=status_path,
        )

    assert metrics.read_text(encoding="utf-8") == "preserve\n"


def test_repeated_metrics_refuses_dangling_symlink_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        (results / name).write_text(f"{name}\n", encoding="utf-8")
    victim = tmp_path / "victim.json"
    metrics = tmp_path / "official_metrics.json"
    metrics.symlink_to(victim.name)
    status_path, _ = _validated_run_status(tmp_path / "run", monkeypatch)

    with pytest.raises(FileExistsError, match="already exists"):
        write_repeated_metrics(
            results_dir=results,
            scene="apartment",
            method="OVIV2",
            mode="causal_checkpoints",
            metrics_path=metrics,
            repeat_path=tmp_path / "official_metrics.repeat.json",
            run_status_path=status_path,
        )

    assert not victim.exists()


def test_canonical_tesse_manifest_is_hash_bound(tmp_path: Path) -> None:
    copied = tmp_path / "tesse_cd.json"
    copied.write_bytes(CANONICAL_TESSE_MANIFEST.read_bytes())

    manifest = validate_tesse_manifest(copied, scene="apartment")

    assert manifest["dataset"] == "TESSE-CD"
    payload = json.loads(copied.read_text(encoding="utf-8"))
    payload["manifest_id"] = "tampered"
    copied.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        validate_tesse_manifest(copied, scene="apartment")


def test_canonical_tesse_manifest_rejects_symlink(tmp_path: Path) -> None:
    linked = tmp_path / "tesse_cd.json"
    linked.symlink_to(CANONICAL_TESSE_MANIFEST)

    with pytest.raises(ValueError, match="symlink"):
        validate_tesse_manifest(linked, scene="apartment")


def test_khronos_run_status_revalidates_hashed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, required = _write_required_run_status(tmp_path / "run", monkeypatch)

    validate_khronos_run_status(status, scene="apartment")
    required["final"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="source"):
        validate_khronos_run_status(status, scene="apartment")


def test_khronos_run_status_normalizes_bound_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, required = _write_required_run_status(
        tmp_path / "run",
        monkeypatch,
        identity_extras={"note": "must not propagate"},
    )

    validated = validate_khronos_run_status(status, scene="apartment")

    assert validated["run_identity"] == {
        "run_id": RUN_ID,
        "config_sha256": hashlib.sha256(required["config"].read_bytes()).hexdigest(),
    }


def test_khronos_run_status_rejects_missing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _ = _write_required_run_status(tmp_path / "run", monkeypatch)
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload.pop("run_identity")
    status.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run identity.*required"):
        validate_khronos_run_status(status, scene="apartment")


def test_khronos_run_status_rejects_noncanonical_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _ = _write_required_run_status(tmp_path / "run", monkeypatch)
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["run_identity"]["run_id"] = "tampered/run"
    status.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run_id.*canonical"):
        validate_khronos_run_status(status, scene="apartment")


def test_khronos_run_status_rejects_config_hash_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _ = _write_required_run_status(tmp_path / "run", monkeypatch)
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["run_identity"]["config_sha256"] = "f" * 64
    status.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config.*binding"):
        validate_khronos_run_status(status, scene="apartment")


def test_khronos_run_status_rejects_config_record_not_in_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, required = _write_required_run_status(tmp_path / "run", monkeypatch)
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["sources"] = [
        entry
        for entry in payload["sources"]
        if Path(entry["path"]) != required["config"].resolve()
    ]
    status.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config.*source"):
        validate_khronos_run_status(status, scene="apartment")


def test_khronos_run_status_rejects_symlinked_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, required = _write_required_run_status(tmp_path / "run", monkeypatch)
    target = tmp_path / "target.4dmap"
    target.write_bytes(b"map")
    source = required["final"]
    source.unlink()
    source.symlink_to(target.name)

    with pytest.raises(ValueError, match="symlink"):
        validate_khronos_run_status(status, scene="apartment")


def test_khronos_run_status_rejects_missing_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, required = _write_required_run_status(tmp_path / "run", monkeypatch)
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["sources"] = [
        entry
        for entry in payload["sources"]
        if Path(entry["path"]) != required["timestamps"].resolve()
    ]
    status.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required artifact"):
        validate_khronos_run_status(status, scene="apartment")


def test_run_refuses_existing_evaluation_directory_before_external_work(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "evaluation").mkdir(parents=True)
    args = parse_args(
        [
            "--workspace", str(tmp_path / "ws"),
            "--manifest", str(CANONICAL_TESSE_MANIFEST),
            "--run-root", str(run_root),
            "--scene", "apartment",
            "--method", "OVIV2",
            "--mode", "causal_checkpoints",
        ]
    )

    with pytest.raises(FileExistsError, match="evaluation output already exists"):
        run(args)


def test_repeated_metrics_rejects_unvalidated_status(tmp_path: Path) -> None:
    results = _write_valid_results(tmp_path / "map/results")
    forged_status = tmp_path / "run_status.json"
    forged_status.write_text(
        json.dumps(
            {
                "run_identity": {
                    "run_id": "caller-controlled",
                    "config_sha256": "f" * 64,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid OVIV2 Khronos run identity"):
        write_repeated_metrics(
            results_dir=results,
            scene="apartment",
            method="OVIV2",
            mode="causal_checkpoints",
            metrics_path=tmp_path / "official_metrics.json",
            repeat_path=tmp_path / "official_metrics.repeat.json",
            run_status_path=forged_status,
        )


def test_official_eval_exposes_no_constructible_validation_token() -> None:
    assert not hasattr(official_eval, "_ValidatedKhronosRunStatus")


def test_run_passes_status_path_to_revalidating_metric_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    base = workspace / "src/khronos/khronos_eval/config/pipeline/apartment.yaml"
    base.parent.mkdir(parents=True)
    base.write_text(
        "ground_truth_dsg_file: old\n"
        "ground_truth_background_file: old\n"
        "gt_changes_file: old\n"
        "evaluation:\n  object_evaluation:\n    changes_file: old\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_status.json").write_text(
        "not raw-readable json\n", encoding="utf-8"
    )
    validated = {
        "run_identity": {
            "run_id": RUN_ID,
            "config_sha256": "a" * 64,
        }
    }
    monkeypatch.setattr(
        official_eval,
        "validate_khronos_run_status",
        lambda *_args, **_kwargs: validated,
    )
    files = {
        "dsg": {"path": str(tmp_path / "dsg")},
        "background_mesh": {"path": str(tmp_path / "mesh")},
        "changes": {"path": str(tmp_path / "changes")},
    }
    monkeypatch.setattr(
        official_eval,
        "validate_tesse_manifest",
        lambda *_args, **_kwargs: {
            "sequences": {"apartment": {"ground_truth": {"files": files}}}
        },
    )
    monkeypatch.setattr(official_eval, "validate_ground_truth_files", lambda _: files)

    def completed(command: list[str], **_: object) -> object:
        time_path = Path(command[command.index("-o") + 1])
        time_path.write_text("time\n", encoding="utf-8")
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "", "stderr": ""},
        )()

    monkeypatch.setattr(official_eval.subprocess, "run", completed)
    captured: dict[str, object] = {}

    def write_metrics(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "PASS",
            "official_metrics": {"path": "metrics", "sha256": "b" * 64},
            "official_metrics_repeat": {"path": "repeat", "sha256": "b" * 64},
        }

    monkeypatch.setattr(official_eval, "write_repeated_metrics", write_metrics)
    args = parse_args(
        [
            "--workspace", str(workspace),
            "--manifest", str(tmp_path / "manifest.json"),
            "--run-root", str(run_root),
            "--scene", "apartment",
        ]
    )

    run(args)

    assert captured["run_status_path"] == run_root / "run_status.json"


def test_partial_metric_record_uses_online_display_mode_and_hashed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        (results / name).write_text(f"{name}\n", encoding="utf-8")
    monkeypatch.setattr(
        official_eval,
        "summarize_khronos_official_metrics_partial",
        lambda _: {
            "state_count": 1,
            "metrics": {"object_f1": 1.0, "dynamic_f1": None, "change_f1": None},
            "unavailable": {
                "dynamic_f1": "upstream omitted dynamic states",
                "change_f1": "upstream omitted change states",
            },
        },
    )
    output = tmp_path / "partial.json"
    repeat = tmp_path / "partial.repeat.json"
    status_path, validated = _validated_run_status(tmp_path / "run", monkeypatch)

    def unavailable(_: Path) -> None:
        raise ValueError("Khronos change F1 has no finite states")

    monkeypatch.setattr(
        official_eval, "summarize_khronos_official_metrics", unavailable
    )

    write_repeated_metrics(
        results_dir=results,
        scene="apartment",
        method="OVIV2",
        mode="causal_checkpoints",
        metrics_path=output,
        repeat_path=repeat,
        run_status_path=status_path,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "PARTIAL"
    assert payload["run_identity"] == validated["run_identity"]
    assert payload["display_mode"] == "online"
    assert all(source["byte_count"] > 0 for source in payload["sources"])
    assert output.read_bytes() == repeat.read_bytes()


def test_metric_json_is_byte_identical_across_run_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_results = _write_valid_results(tmp_path / "first/map/results")
    second_results = _write_valid_results(tmp_path / "second/map/results")
    first = tmp_path / "first/evaluation/official_metrics.json"
    second = tmp_path / "second/evaluation/official_metrics.json"
    first_repeat = tmp_path / "first/evaluation/official_metrics.repeat.json"
    second_repeat = tmp_path / "second/evaluation/official_metrics.repeat.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first_status_path, first_status = _validated_run_status(
        tmp_path / "first/run", monkeypatch
    )
    second_status_path, second_status = _validated_run_status(
        tmp_path / "second/run", monkeypatch
    )

    write_repeated_metrics(
        results_dir=first_results,
        scene="apartment",
        method="OVIV2",
        mode="causal_checkpoints",
        metrics_path=first,
        repeat_path=first_repeat,
        run_status_path=first_status_path,
    )
    write_repeated_metrics(
        results_dir=second_results,
        scene="apartment",
        method="OVIV2",
        mode="causal_checkpoints",
        metrics_path=second,
        repeat_path=second_repeat,
        run_status_path=second_status_path,
    )

    assert first.read_bytes() == first_repeat.read_bytes()
    assert second.read_bytes() == second_repeat.read_bytes()
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text())
    assert payload["status"] == "PASS"
    assert payload["run_identity"] == first_status["run_identity"]
    assert payload["run_identity"] == second_status["run_identity"]
    assert set(payload["run_identity"]) == {"run_id", "config_sha256"}
    assert [entry["path"] for entry in payload["sources"]] == [
        "../map/results/static_objects.csv",
        "../map/results/dynamic_objects.csv",
        "../map/results/background_mesh.csv",
    ]


def test_missing_metric_csv_is_recorded_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _write_valid_results(tmp_path / "results")
    (results / "dynamic_objects.csv").unlink()
    metrics = tmp_path / "official_metrics.json"
    repeat = tmp_path / "official_metrics.repeat.json"
    status_path, validated = _validated_run_status(tmp_path / "run", monkeypatch)

    summary = write_repeated_metrics(
        results_dir=results,
        scene="apartment",
        method="OVIV2",
        mode="causal_checkpoints",
        metrics_path=metrics,
        repeat_path=repeat,
        run_status_path=status_path,
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert summary["status"] == "UNAVAILABLE"
    assert metrics.read_bytes() == repeat.read_bytes()
    assert payload["metrics"]["dynamic_f1"] is None
    assert payload["run_identity"] == validated["run_identity"]
    assert "dynamic_objects.csv" in payload["unavailable"]["dynamic_f1"]
    missing = next(
        source
        for source in payload["sources"]
        if Path(source["path"]).name == "dynamic_objects.csv"
    )
    assert missing == {"path": "results/dynamic_objects.csv", "status": "MISSING"}
