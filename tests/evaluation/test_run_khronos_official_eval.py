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
    validate_khronos_run_status,
    validate_tesse_manifest,
    write_repeated_metrics,
)


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

    def unavailable(**_: object) -> None:
        raise ValueError("Khronos change F1 has no finite states")

    def partial(*, output: Path, **_: object) -> None:
        output.write_text('{"status":"PARTIAL"}\n', encoding="utf-8")

    monkeypatch.setattr(official_eval, "_write_metrics", unavailable)
    monkeypatch.setattr(official_eval, "_write_partial_metrics", partial)
    summary = write_repeated_metrics(
        results_dir=results,
        scene="apartment",
        method="OVIV2",
        mode="causal_checkpoints",
        metrics_path=tmp_path / "official_metrics.json",
        repeat_path=tmp_path / "official_metrics.repeat.json",
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


def test_repeated_metrics_refuses_existing_outputs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        (results / name).write_text(f"{name}\n", encoding="utf-8")
    metrics = tmp_path / "official_metrics.json"
    metrics.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_repeated_metrics(
            results_dir=results,
            scene="apartment",
            method="OVIV2",
            mode="causal_checkpoints",
            metrics_path=metrics,
            repeat_path=tmp_path / "official_metrics.repeat.json",
        )

    assert metrics.read_text(encoding="utf-8") == "preserve\n"


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


def test_khronos_run_status_revalidates_hashed_sources(tmp_path: Path) -> None:
    source = tmp_path / "final.4dmap"
    source.write_bytes(b"map")
    status = tmp_path / "run_status.json"
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
                "sources": [
                    {
                        "path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "byte_count": source.stat().st_size,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    validate_khronos_run_status(status, scene="apartment")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source"):
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

    official_eval._write_partial_metrics(
        results_dir=results,
        scene="apartment",
        method="OVIV2",
        mode="causal_checkpoints",
        output=output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["display_mode"] == "online"
    assert all(source["byte_count"] > 0 for source in payload["sources"])
