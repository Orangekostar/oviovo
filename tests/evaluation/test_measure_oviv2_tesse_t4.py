from __future__ import annotations

import hashlib
import importlib
import json
import os
import signal
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.evaluation.measure_oviv2_tesse_t4 import (
    T4CollectionError,
    build_final_map_inventory,
    collect_shortlist,
    collect_gpu_sample,
    measure_t4,
    _validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path.absolute()), "sha256": _sha256(path), "byte_count": path.stat().st_size}


def _relative_record(path: Path, root: Path) -> dict[str, object]:
    return {**_record(path), "path": path.relative_to(root).as_posix()}


def _fixture(tmp_path: Path) -> dict[str, object]:
    from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash

    checkpoint = tmp_path / "clip.bin"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    config_value = {
        "clip_pretrained_path": str(checkpoint),
        "temporal_readout": {"execution_profile": "a4"},
    }
    config_value["algorithm_hash"] = canonical_algorithm_hash(config_value)
    config = _write(tmp_path / "config.json", config_value)
    config_sha = hashlib.sha256(
        json.dumps(config_value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_root = tmp_path / "run"
    snapshot = run_root / "final" / "snapshot.npz"
    entities = run_root / "final" / "entities.jsonl"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snapshot,
        background_xyz=np.asarray([[1.0, 2.0, 3.0]]),
        timestamp=np.asarray(4_000_000_000.0),
        scope=np.asarray("current"),
        scene_id=np.asarray("apartment"),
    )
    entities.write_bytes(b"entities")
    normalized = _write(run_root / "normalized_run_config.json", config_value)
    coverage = run_root / "temporal_frame_coverage.jsonl"
    coverage.write_text(
        "".join(
            json.dumps({"frame_index": frame, "timestamp_ns": (frame + 1) * 1_000_000_000}) + "\n"
            for frame in range(4)
        ),
        encoding="utf-8",
    )
    source_index = _write(run_root / "source_index.json", {
        "frame_coverage": _relative_record(coverage, run_root),
    })
    final_current_map = {
        "frame_index": 3,
        "timestamp_ns": 4_000_000_000,
        "scope": "current",
        "snapshot": _relative_record(snapshot, run_root),
        "entities": _relative_record(entities, run_root),
        "background_storage": "snapshot.npz:background_xyz",
    }
    run_manifest = _write(
        run_root / "run_manifest.json",
        {
            "schema_version": 2,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment",
            "algorithm_hash": config_value["algorithm_hash"],
            "processed_frame_count": 4,
            "first_frame_index": 0,
            "last_frame_index": 3,
            "config": {"sha256": _sha256(config), "byte_count": config.stat().st_size},
            "normalized_run_config": {
                "path": normalized.name,
                "sha256": _sha256(normalized),
                "byte_count": normalized.stat().st_size,
            },
            "artifact_inventory": [
                "final/entities.jsonl", "final/snapshot.npz", "normalized_run_config.json",
                "source_index.json", "temporal_frame_coverage.jsonl",
            ],
            "source_index": _relative_record(source_index, run_root),
            "final_current_map": final_current_map,
        },
    )
    time_log = tmp_path / "time.txt"
    time_log.write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:08.00\n"
        "Maximum resident set size (kbytes): 2500000\n",
        encoding="utf-8",
    )
    gpu = _write(
        tmp_path / "gpu.json",
        {
            "sample_interval_ms": 200,
            "process_group_id": 17,
            "phase_events": [
                {"phase": "mapping", "event": "start", "timestamp_ns": 900_000_000, "pid": 20, "process_group_id": 17, "proc_starttime_ticks": 200},
                {"phase": "mapping", "event": "end", "timestamp_ns": 1_200_000_000, "pid": 20, "process_group_id": 17, "proc_starttime_ticks": 200},
                {"phase": "queries", "event": "start", "timestamp_ns": 1_300_000_000, "pid": 21, "process_group_id": 17, "proc_starttime_ticks": 210},
                {"phase": "queries", "event": "end", "timestamp_ns": 1_600_000_000, "pid": 21, "process_group_id": 17, "proc_starttime_ticks": 210},
            ],
            "samples": [
                {"timestamp_ns": 1_000_000_000, "phase": "mapping", "pid": 20, "process_group_id": 17, "used_memory_mib": 3000,
                 "processes": [{"gpu_uuid": "GPU-fixture", "pid": 20, "proc_starttime_ticks": 200,
                                "process_group_id": 17, "used_memory_mib": 3000}]},
                {"timestamp_ns": 1_100_000_000, "phase": "mapping", "pid": 20, "process_group_id": 17, "used_memory_mib": 3000,
                 "processes": [{"gpu_uuid": "GPU-fixture", "pid": 20, "proc_starttime_ticks": 200,
                                "process_group_id": 17, "used_memory_mib": 3000}]},
                {"timestamp_ns": 1_400_000_000, "phase": "queries", "pid": 21, "process_group_id": 17, "used_memory_mib": 4000,
                 "processes": [{"gpu_uuid": "GPU-fixture", "pid": 21, "proc_starttime_ticks": 210,
                                "process_group_id": 17, "used_memory_mib": 4000}]},
                {"timestamp_ns": 1_500_000_000, "phase": "queries", "pid": 21, "process_group_id": 17, "used_memory_mib": 4000,
                 "processes": [{"gpu_uuid": "GPU-fixture", "pid": 21, "proc_starttime_ticks": 210,
                                "process_group_id": 17, "used_memory_mib": 4000}]},
            ],
        },
    )
    queries = _write(tmp_path / "queries.json", {"classes": ["Chair"]})
    protocol = _write(tmp_path / "protocol.json", {"manifest_id": "fixture"})
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    query = _write(
        tmp_path / "query.json",
        {
            "schema_version": 1, "manifest_id": "oviv2-query-measurement-v1", "baseline": "oviv2",
            "scene_id": "apartment", "query_count": 1, "warmup_count": 10,
            "measured_repeats": 5, "repeat_count": 5, "entity_count": 1,
            "model": "ViT-H-14", "text_model": {"model_id": "ViT-H-14", "checkpoint_sha256": _sha256(checkpoint)},
            "query_vocabulary_sha256": _sha256(queries), "query_protocol": _record(protocol), "device": "cuda:0",
            "initialization_s": 1.0, "evaluation_io_s": 0.1, "latencies_ms": latencies,
            "samples": [{"repeat_index": index, "query_index": 0, "latency_ms": value, "top_entity_index": 0} for index, value in enumerate(latencies)],
            "query_mean_ms": 3.0, "query_p50_ms": 3.0, "query_p95_ms": 4.8,
            "summary": {"query_mean_ms": 3.0, "query_p50_ms": 3.0, "query_p95_ms": 4.8},
            "sources": {"snapshot": _record(snapshot), "entities": _record(entities), "queries": _record(queries), "checkpoint": _record(checkpoint)},
            "protocol": "full tokenization, text encoding, and entity ranking",
        },
    )
    candidate_result = _write(tmp_path / "candidate-result.json", {"candidate_id": "a4"})
    shortlist = _write(tmp_path / "shortlist.json", {
        "shortlisted_candidate_ids": ["a4"],
        "shortlisted_candidates": [{"candidate_id": "a4", "config_sha256": config_sha,
                                     "profile": "a4", "algorithm_hash": config_value["algorithm_hash"],
                                     "result": _record(candidate_result),
                                     "selected_config": config_value, "selected_config_record": _record(config)}],
    })
    return {
        "candidate_id": "a4",
        "config": config,
        "config_sha256": config_sha,
        "config_source_sha256": _sha256(config),
        "run_manifest": run_manifest,
        "time_log": time_log,
        "gpu_samples": gpu,
        "query_measurements": query,
        "protocol": protocol,
        "shortlist": shortlist,
        "output": tmp_path / "evidence.json",
    }


def test_measurement_binds_all_metrics_to_one_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = measure_t4(fixture)

    assert set(evidence["metrics"]) == {
        "total_runtime_s_per_frame", "query_mean_ms", "query_p95_ms",
        "peak_gpu_gb", "peak_ram_gb", "final_map_mb",
    }
    assert evidence["sources"]["run_manifest_sha256"] == _sha256(fixture["run_manifest"])
    assert evidence["process_group_id"] == 17
    assert evidence["metrics"]["peak_gpu_gb"] == 4.0
    assert evidence["metrics"]["peak_ram_gb"] == 2.5
    assert evidence["config_sha256"] != json.loads(
        Path(fixture["config"]).read_text()
    )["algorithm_hash"]


def test_gpu_collection_keeps_only_requested_process_group() -> None:
    rows = [
        {"pid": 11, "process_group_id": 7, "used_memory_mib": 100},
        {"pid": 12, "process_group_id": 8, "used_memory_mib": 9999},
    ]
    assert collect_gpu_sample(rows, process_group_id=7, phase="mapping") == {
        "phase": "mapping", "pid": 11, "process_group_id": 7, "used_memory_mib": 100.0,
    }


def test_final_map_inventory_is_exact_and_rejects_aliases(tmp_path: Path) -> None:
    root = tmp_path / "run"
    paths = []
    for name in ("snapshot", "entities"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        paths.append(path)
    inventory = build_final_map_inventory(root, dict(zip(("snapshot", "entities"), paths)))
    assert [item["role"] for item in inventory["files"]] == ["snapshot", "entities"]
    assert inventory["total_bytes"] == sum(path.stat().st_size for path in paths)
    with pytest.raises(T4CollectionError, match="distinct"):
        build_final_map_inventory(root, {"snapshot": paths[0], "entities": paths[0]})


def test_measurement_rejects_config_hash_drift_and_no_clobber(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["config_sha256"] = "0" * 64
    with pytest.raises(T4CollectionError, match="candidate identity|config hash"):
        measure_t4(fixture)
    fixture = _fixture(tmp_path / "second")
    Path(fixture["output"]).write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        measure_t4(fixture)


def test_measurement_rejects_stale_final_snapshot_and_empty_directory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "stale")
    run_path = Path(fixture["run_manifest"])
    run = json.loads(run_path.read_text())
    snapshot = run_path.parent / run["final_current_map"]["snapshot"]["path"]
    np.savez_compressed(
        snapshot,
        background_xyz=np.asarray([[1.0, 2.0, 3.0]]),
        timestamp=np.asarray(3_000_000_000.0),
        scope=np.asarray("current"),
        scene_id=np.asarray("apartment"),
    )
    run["final_current_map"]["snapshot"] = _relative_record(snapshot, run_path.parent)
    run["final_current_map"]["timestamp_ns"] = 3_000_000_000
    _write(run_path, run)
    with pytest.raises(T4CollectionError, match="run-end frame coverage"):
        measure_t4(fixture)

    fixture = _fixture(tmp_path / "external")
    run_path = Path(fixture["run_manifest"])
    run_root = run_path.parent
    run = json.loads(run_path.read_text())
    outside_coverage = run_root.parent / "outside-coverage.jsonl"
    outside_coverage.write_text(
        json.dumps({"frame_index": 3, "timestamp_ns": 3_000_000_000}) + "\n",
        encoding="utf-8",
    )
    outside_index = _write(run_root.parent / "outside-source-index.json", {
        "frame_coverage": {
            **_record(outside_coverage),
            "path": "../outside-coverage.jsonl",
        },
    })
    run["source_index"] = {
        **_record(outside_index),
        "path": "../outside-source-index.json",
    }
    _write(run_path, run)
    with pytest.raises(T4CollectionError, match="source index path is invalid"):
        measure_t4(fixture)

    fixture = _fixture(tmp_path / "external-final")
    run_path = Path(fixture["run_manifest"])
    run = json.loads(run_path.read_text())
    outside_snapshot = run_path.parent.parent / "outside-final.npz"
    np.savez_compressed(
        outside_snapshot,
        background_xyz=np.asarray([[1.0, 2.0, 3.0]]),
        timestamp=np.asarray(4_000_000_000.0),
        scope=np.asarray("current"),
        scene_id=np.asarray("apartment"),
    )
    run["final_current_map"]["snapshot"] = {
        **_record(outside_snapshot),
        "path": "../outside-final.npz",
    }
    _write(run_path, run)
    with pytest.raises(T4CollectionError, match="final current-map snapshot path is invalid"):
        measure_t4(fixture)

    fixture = _fixture(tmp_path / "directory")
    (Path(fixture["run_manifest"]).parent / "undeclared-empty").mkdir()
    with pytest.raises(T4CollectionError, match="directory inventory"):
        measure_t4(fixture)

def test_frozen_protocol_pins_query_resources_units_and_bounds() -> None:
    path = REPO_ROOT / "configs/evaluation/manifests/oviv2_tesse_t4_v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert protocol["query"]["text_model_id"] == "ViT-H-14"
    assert len(protocol["query"]["checkpoint_sha256"]) == 64
    assert protocol["query"]["warmup_count"] == 10
    assert protocol["query"]["measured_repeats"] == 5
    assert protocol["query"]["operation_order"] == [
        "full_tokenization", "text_encoding", "l2_normalization", "current_map_entity_ranking"
    ]
    assert protocol["sample_interval_ms"] == 200
    assert protocol["units"] == {
        "total_runtime_s_per_frame": "seconds/frame", "query_mean_ms": "milliseconds",
        "query_p95_ms": "milliseconds", "peak_gpu_gb": "GB (decimal)",
        "peak_ram_gb": "GB (decimal)", "final_map_mb": "MB (decimal)",
    }
    assert set(protocol["bounds"]) == set(protocol["units"])
    assert protocol["final_map_inventory"] == {
        "scope": "final_current_map_after_all_processed_frames",
        "roles": ["snapshot", "entities"],
        "background_storage": "snapshot.npz:background_xyz",
        "include": "only_unique_regular_source_files_for_snapshot_and_entities",
        "exclude": ["cumulative", "checkpoints", "diagnostics", "logs", "metrics", "office"],
    }
    assert protocol["mapping_argv"][:2] == [
        "{python}", "{repo_root}/scripts/evaluation/run_oviv2_tesse_cd_v2.py",
    ]
    assert protocol["query_argv"][:2] == [
        "{python}", "{repo_root}/scripts/evaluation/measure_baseline_queries.py",
    ]
    vocabulary = REPO_ROOT / protocol["query"]["vocabulary_path"]
    assert _sha256(vocabulary) == protocol["query"]["vocabulary_sha256"]


def test_gpu_phase_events_reject_single_sparse_foreign_and_reused_pid(tmp_path: Path) -> None:
    cases = (
        ("single", lambda gpu: gpu["samples"].pop(1), "at least two"),
        ("sparse", lambda gpu: gpu["samples"][1].update(timestamp_ns=1_300_000_001), "sampling interval"),
        ("foreign", lambda gpu: gpu["samples"][0]["processes"][0].update(process_group_id=999), "foreign process"),
        ("reuse", lambda gpu: gpu["samples"][0]["processes"][0].update(proc_starttime_ticks=999), "starttime"),
    )
    for name, mutate, match in cases:
        fixture = _fixture(tmp_path / name)
        path = Path(fixture["gpu_samples"])
        gpu = json.loads(path.read_text())
        mutate(gpu)
        _write(path, gpu)
        with pytest.raises(T4CollectionError, match=match):
            measure_t4(fixture)


def test_protocol_copy_is_not_a_controlled_trust_anchor(tmp_path: Path, monkeypatch) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    frozen = REPO_ROOT / "configs/evaluation/manifests/oviv2_tesse_t4_v1.json"
    copied = tmp_path / "protocol.json"
    copied.write_bytes(frozen.read_bytes())
    monkeypatch.setattr(collector, "FROZEN_PROTOCOL_PATH", frozen)
    with pytest.raises(T4CollectionError, match="controlled frozen"):
        _validate_protocol(json.loads(copied.read_text()), copied)


def test_fake_collector_covers_mapping_and_query_process_groups(tmp_path: Path, monkeypatch) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector
    from scripts.evaluation.freeze_oviv2_tesse_cd_v2 import _freeze_t4_evidence
    from scripts.evaluation.tune_oviv2_tesse_dual_readout import tune_final
    from scripts.evaluation.verify_oviv2_tesse_t4_gate import verify_t4_gate

    checkpoint = tmp_path / "clip.bin"
    checkpoint.write_bytes(b"checkpoint")
    protocol = json.loads(
        (REPO_ROOT / "configs/evaluation/manifests/oviv2_tesse_t4_v1.json").read_text()
    )
    protocol["query"]["checkpoint_sha256"] = _sha256(checkpoint)
    protocol_path = _write(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(collector, "FROZEN_PROTOCOL_PATH", protocol_path)
    from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash

    config_value = {
        "clip_pretrained_path": str(checkpoint),
        "temporal_readout": {"execution_profile": "a4"},
    }
    config_value["algorithm_hash"] = canonical_algorithm_hash(config_value)
    config = _write(tmp_path / "config.json", config_value)
    config_sha = hashlib.sha256(
        json.dumps(config_value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    algorithm_hash = config_value["algorithm_hash"]
    manifest_path = REPO_ROOT / "configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json"
    shortlist = {
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
        "results_root": str(tmp_path),
        "manifest": _record(manifest_path),
        "result_files": [],
        "profile_fallback_order": ["a4", "a3", "a2"],
        "floors": {},
        "shortlisted_candidate_ids": ["a4"],
        "shortlisted_candidates": [{
            "candidate_id": "a4", "config_sha256": config_sha,
            "profile": "a4", "algorithm_hash": algorithm_hash,
            "selected_config": config_value,
            "selected_config_record": _record(config),
        }],
        "rejection_ledger": [],
    }
    candidate_result = _write(tmp_path / "candidate-result.json", {"candidate_id": "a4"})
    shortlist["shortlisted_candidates"][0]["result"] = _record(candidate_result)
    shortlist_path = _write(tmp_path / "shortlist.json", shortlist)

    def fake_run(argv, *, gpu, phase, time_log=None):
        assert argv[0] == str(Path(sys.executable).resolve())
        assert Path(argv[1]).is_absolute()
        assert Path(argv[1]).is_relative_to(REPO_ROOT)
        del gpu
        if phase == "mapping":
            run_root = Path(argv[argv.index("--output") + 1])
            final = run_root / "final"
            final.mkdir(parents=True)
            snapshot = final / "snapshot.npz"
            entities = final / "entities.jsonl"
            np.savez_compressed(
                snapshot,
                background_xyz=np.asarray([[1.0, 2.0, 3.0]]),
                timestamp=np.asarray(2_000_000_000.0),
                scope=np.asarray("current"),
                scene_id=np.asarray("apartment"),
            )
            entities.write_bytes(b"entities")
            normalized = _write(run_root / "normalized_run_config.json", config_value)
            coverage = run_root / "temporal_frame_coverage.jsonl"
            coverage.write_text(
                json.dumps({"frame_index": 0, "timestamp_ns": 1_000_000_000}) + "\n"
                + json.dumps({"frame_index": 1, "timestamp_ns": 2_000_000_000}) + "\n",
                encoding="utf-8",
            )
            source_index = _write(run_root / "source_index.json", {
                "frame_coverage": _relative_record(coverage, run_root),
            })
            _write(run_root / "run_manifest.json", {
                "schema_version": 2, "dataset": "TESSE-CD", "method_id": "OVIV2",
                "protocol_id": "oviv2-tessecd-v2", "scene": "apartment",
                "algorithm_hash": algorithm_hash,
                "processed_frame_count": 2,
                "first_frame_index": 0, "last_frame_index": 1,
                "config": {"sha256": _sha256(config), "byte_count": config.stat().st_size},
                "normalized_run_config": {
                    "path": normalized.name, "sha256": _sha256(normalized),
                    "byte_count": normalized.stat().st_size,
                },
                "artifact_inventory": [
                    "final/entities.jsonl", "final/snapshot.npz",
                    "normalized_run_config.json", "source_index.json",
                    "temporal_frame_coverage.jsonl",
                ],
                "source_index": _relative_record(source_index, run_root),
                "final_current_map": {
                    "frame_index": 1, "timestamp_ns": 2_000_000_000,
                    "scope": "current", "snapshot": _relative_record(snapshot, run_root),
                    "entities": _relative_record(entities, run_root),
                    "background_storage": "snapshot.npz:background_xyz",
                },
            })
            Path(time_log).write_text(
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:04.00\n"
                "Maximum resident set size (kbytes): 1000000\n", encoding="utf-8"
            )
            samples = [
                {"timestamp_ns": timestamp, "phase": "mapping", "pid": 101, "process_group_id": 101, "used_memory_mib": 2000,
                 "processes": [{"gpu_uuid": "GPU-fake", "pid": 101, "proc_starttime_ticks": 1,
                                "process_group_id": 101, "used_memory_mib": 2000}]}
                for timestamp in (1_000_000_000, 1_100_000_000)
            ]
            events = [
                {"phase": "mapping", "event": event, "timestamp_ns": timestamp, "pid": 101,
                 "process_group_id": 101, "proc_starttime_ticks": 1}
                for event, timestamp in (("start", 900_000_000), ("end", 1_200_000_000))
            ]
            return 0, samples, events
        query_output = Path(argv[argv.index("--output") + 1])
        snapshot = Path(argv[argv.index("--snapshot") + 1])
        entities = Path(argv[argv.index("--entities") + 1])
        queries = Path(argv[argv.index("--queries") + 1])
        clip = Path(argv[argv.index("--clip-weight") + 1])
        query_protocol = Path(argv[argv.index("--protocol") + 1])
        query_count = len(json.loads(queries.read_text())["classes"])
        latencies = [1.0] * (query_count * 5)
        _write(query_output, {
            "schema_version": 1, "manifest_id": "oviv2-query-measurement-v1", "baseline": "oviv2",
            "scene_id": "apartment", "query_count": query_count, "warmup_count": 10,
            "measured_repeats": 5, "repeat_count": 5, "entity_count": 1, "model": "ViT-H-14",
            "text_model": {"model_id": "ViT-H-14", "checkpoint_sha256": _sha256(clip)},
            "query_vocabulary_sha256": _sha256(queries), "query_protocol": _record(query_protocol), "device": "cuda:0",
            "initialization_s": 0.1, "evaluation_io_s": 0.1, "latencies_ms": latencies,
            "samples": [{"repeat_index": index // query_count, "query_index": index % query_count,
                         "latency_ms": 1.0, "top_entity_index": 0} for index in range(len(latencies))],
            "query_mean_ms": 1.0, "query_p50_ms": 1.0, "query_p95_ms": 1.0,
            "summary": {"query_mean_ms": 1.0, "query_p50_ms": 1.0, "query_p95_ms": 1.0},
            "sources": {"snapshot": _record(snapshot), "entities": _record(entities),
                        "queries": _record(queries), "checkpoint": _record(clip)},
            "protocol": "full tokenization, text encoding, and entity ranking",
        })
        samples = [
            {"timestamp_ns": timestamp, "phase": "queries", "pid": 202, "process_group_id": 202, "used_memory_mib": 3000,
             "processes": [{"gpu_uuid": "GPU-fake", "pid": 202, "proc_starttime_ticks": 2,
                            "process_group_id": 202, "used_memory_mib": 3000}]}
            for timestamp in (1_400_000_000, 1_500_000_000)
        ]
        events = [
            {"phase": "queries", "event": event, "timestamp_ns": timestamp, "pid": 202,
             "process_group_id": 202, "proc_starttime_ticks": 2}
            for event, timestamp in (("start", 1_300_000_000), ("end", 1_600_000_000))
        ]
        return 0, samples, events

    monkeypatch.setattr("scripts.evaluation.measure_oviv2_tesse_t4._run_sampled", fake_run)
    outside = tmp_path / "outside-cwd"
    outside.mkdir()
    monkeypatch.chdir(outside)
    paths = collect_shortlist(protocol_path, shortlist_path, "0", tmp_path / "collected")

    assert len(paths) == 1
    evidence = json.loads(paths[0].read_text())
    assert evidence["metrics"]["peak_gpu_gb"] == 3.0
    wrapper = json.loads((paths[0].parent / "measurement-run.json").read_text())
    assert wrapper["collection_protocol"]["sha256"] == _sha256(protocol_path)
    assert wrapper["runner_manifest"]["sha256"] == _sha256(paths[0].parent / "run" / "run_manifest.json")
    matrix_path = tmp_path / "matrix.json"
    matrix = verify_t4_gate(paths, shortlist_path, matrix_path)
    assert matrix["candidates"]["a4"]["status"] == "PASS"
    selection_path = tmp_path / "selection.json"
    selection = tune_final(manifest_path, shortlist_path, matrix_path, selection_path)
    assert selection["selected_candidate_id"] == "a4"
    frozen = _freeze_t4_evidence(
        matrix_path,
        selected_candidate="a4",
        selected_config_sha256=config_sha,
        repo_root=tmp_path,
        snapshots={},
    )
    assert frozen["root_sha256"] == matrix["root_sha256"]
    assert frozen["artifact"]["path"] != frozen["protocol"]["path"]
    assert evidence["sources"]["protocol"]["path"] != frozen["protocol"]["path"]


def test_collector_staging_failure_preserves_artifacts_and_uses_new_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    destination = tmp_path / "collected"
    calls = 0

    def fail(protocol_path, shortlist_path, gpu, staging):
        nonlocal calls
        calls += 1
        assert staging.is_dir()
        (staging / "partial").write_text("partial", encoding="utf-8")
        raise OSError("injected collection failure")

    monkeypatch.setattr(collector, "_collect_shortlist_unpublished", fail)
    preserved_inodes = []
    for expected_count in (1, 2):
        before_fds = len(list(Path("/proc/self/fd").iterdir()))
        with pytest.raises(collector.T4PublicationUncertain, match="preserved") as raised:
            collect_shortlist(Path("protocol"), Path("shortlist"), "0", destination)
        assert isinstance(raised.value.__cause__, OSError)
        assert not destination.exists()
        staging = sorted(tmp_path.glob(".collected.staging-*"))
        assert len(staging) == expected_count
        newest = next(path for path in staging if path.stat().st_ino not in preserved_inodes)
        preserved_inodes.append(newest.stat().st_ino)
        assert (newest / "partial").read_text(encoding="utf-8") == "partial"
        record = next(item for item in raised.value.preserved if item.inode == newest.stat().st_ino)
        assert record.name == newest.name
        assert record.ownership == "owned"
        assert len(list(Path("/proc/self/fd").iterdir())) == before_fds
    assert calls == 2


def test_collector_fsyncs_staging_root_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    events: list[tuple[str, int | None]] = []
    staging_fd = -1
    original_fsync = collector.os.fsync
    original_rename = collector._rename_directory_new

    def collect_empty(protocol_path, shortlist_path, gpu, staging):
        nonlocal staging_fd
        staging_fd = int(staging.name)
        return []

    def record_fsync(fd: int) -> None:
        events.append(("fsync", fd))
        original_fsync(fd)

    def record_rename(source, destination, *, parent_fd=None):
        events.append(("rename", None))
        return original_rename(source, destination, parent_fd=parent_fd)

    monkeypatch.setattr(collector, "_collect_shortlist_unpublished", collect_empty)
    monkeypatch.setattr(collector.os, "fsync", record_fsync)
    monkeypatch.setattr(collector, "_rename_directory_new", record_rename)

    collect_shortlist(Path("protocol"), Path("shortlist"), "0", tmp_path / "collected")

    assert events.index(("fsync", staging_fd)) < events.index(("rename", None))


def test_sample_collector_terminates_and_waits_for_process_group_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    class Process:
        pid = 321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -signal.SIGTERM
            return self.returncode

    process = Process()
    killed = []
    monkeypatch.setattr(collector.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(collector.os, "getpgid", lambda pid: 321)
    monkeypatch.setattr(collector, "_proc_starttime", lambda pid: 7)
    monkeypatch.setattr(collector, "_gpu_rows", lambda gpu: (_ for _ in ()).throw(OSError("nvidia failed")))
    monkeypatch.setattr(collector.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    monkeypatch.setattr(collector, "_process_group_exists", lambda pgid: False)

    with pytest.raises(OSError, match="nvidia failed"):
        collector._run_sampled([sys.executable, "-c", "pass"], gpu="0", phase="mapping")
    assert killed == [(321, signal.SIGTERM)]
    assert process.returncode == -signal.SIGTERM


def test_sample_collector_cleans_group_when_starttime_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    class Process:
        pid = 654
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -signal.SIGTERM
            return self.returncode

    process = Process()
    killed = []
    monkeypatch.setattr(collector.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(collector, "_proc_starttime", lambda pid: (_ for _ in ()).throw(OSError("proc failed")))
    monkeypatch.setattr(collector.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    monkeypatch.setattr(collector, "_process_group_exists", lambda pgid: False)

    with pytest.raises(OSError, match="proc failed"):
        collector._run_sampled([sys.executable, "-c", "pass"], gpu="0", phase="mapping")
    assert killed == [(654, signal.SIGTERM)]
    assert process.returncode == -signal.SIGTERM


def test_collector_failure_preserves_replacement_and_owned_renamed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    destination = tmp_path / "collected"
    moved = tmp_path / "owned-moved"

    def replace_then_fail(protocol_path, shortlist_path, gpu, staging):
        actual_staging = Path(os.readlink(staging))
        (actual_staging / "owned").write_text("owned", encoding="utf-8")
        actual_staging.rename(moved)
        actual_staging.mkdir()
        (actual_staging / "replacement").write_text("replacement", encoding="utf-8")
        raise OSError("injected replacement")

    monkeypatch.setattr(collector, "_collect_shortlist_unpublished", replace_then_fail)
    with pytest.raises(collector.T4PublicationUncertain, match="preserved") as raised:
        collect_shortlist(Path("protocol"), Path("shortlist"), "0", destination)
    assert isinstance(raised.value.__cause__, OSError)
    replacements = list(tmp_path.glob(".collected.staging-*/replacement"))
    assert len(replacements) == 1
    assert replacements[0].read_text(encoding="utf-8") == "replacement"
    assert (moved / "owned").read_text(encoding="utf-8") == "owned"
    identities = {(item.name, item.ownership, item.inode) for item in raised.value.preserved}
    assert (moved.name, "owned", moved.stat().st_ino) in identities
    assert (replacements[0].parent.name, "unknown", replacements[0].parent.stat().st_ino) in identities


@pytest.mark.parametrize("failure", ["mkdir", "stat", "open"])
def test_collector_staging_initialization_failure_closes_fds_and_preserves_if_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    destination = tmp_path / "collected"
    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    original_mkdir = collector.os.mkdir
    original_open = collector.os.open
    original_stat = collector.os.stat
    injected = False

    def is_staging(path, dir_fd) -> bool:
        return (
            dir_fd is not None
            and isinstance(path, str)
            and path.startswith(".collected.staging-")
        )

    def fail_mkdir(path, *args, **kwargs):
        nonlocal injected
        if failure == "mkdir" and is_staging(path, kwargs.get("dir_fd")) and not injected:
            injected = True
            raise OSError("injected mkdir failure")
        return original_mkdir(path, *args, **kwargs)

    def fail_stat(path, *args, **kwargs):
        nonlocal injected
        if failure == "stat" and is_staging(path, kwargs.get("dir_fd")) and not injected:
            injected = True
            raise OSError("injected stat failure")
        return original_stat(path, *args, **kwargs)

    def fail_open(path, *args, **kwargs):
        nonlocal injected
        if failure == "open" and is_staging(path, kwargs.get("dir_fd")) and not injected:
            injected = True
            raise OSError("injected open failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(collector.os, "mkdir", fail_mkdir)
    monkeypatch.setattr(collector.os, "stat", fail_stat)
    monkeypatch.setattr(collector.os, "open", fail_open)
    if failure == "mkdir":
        with pytest.raises(OSError, match="injected mkdir failure"):
            collect_shortlist(Path("protocol"), Path("shortlist"), "0", destination)
    else:
        with pytest.raises(collector.T4PublicationUncertain, match="preserved") as raised:
            collect_shortlist(Path("protocol"), Path("shortlist"), "0", destination)
        assert isinstance(raised.value.__cause__, OSError)
        staging = list(tmp_path.glob(".collected.staging-*"))
        assert len(staging) == 1
        record = next(item for item in raised.value.preserved if item.name == staging[0].name)
        assert record.inode == staging[0].stat().st_ino
        assert record.ownership == ("unbound" if failure == "stat" else "owned")
    assert injected
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds
    if failure == "mkdir":
        assert not list(tmp_path.glob(".collected.staging-*"))


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.evaluation.measure_oviv2_tesse_t4",
        "scripts.evaluation.verify_oviv2_tesse_t4_gate",
    ],
)
def test_directory_publication_parent_swap_preserves_published_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    module = importlib.import_module(module_name)
    parent = tmp_path / module_name.rsplit(".", 1)[-1]
    moved = parent.with_name(f"{parent.name}-moved")
    parent.mkdir()
    source = parent / "staging"
    destination = parent / "published"
    source.mkdir()
    (source / "payload").write_text("owned", encoding="utf-8")
    original_stat = module.os.stat
    swapped = False

    def swap_after_destination_stat(path, *args, **kwargs):
        nonlocal swapped
        result = original_stat(path, *args, **kwargs)
        if path == destination.name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            parent.rename(moved)
            parent.mkdir()
            (parent / "replacement").write_text("replacement", encoding="utf-8")
        return result

    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "stat", swap_after_destination_stat)
        with pytest.raises(module.T4PublicationUncertain, match="preserved") as raised:
            module._rename_directory_new(source, destination)
    published = moved / destination.name
    assert (published / "payload").read_text(encoding="utf-8") == "owned"
    assert (parent / "replacement").read_text(encoding="utf-8") == "replacement"
    record = next(item for item in raised.value.preserved if item.inode == published.stat().st_ino)
    assert record.name == destination.name and record.ownership == "owned"
    assert (record.parent_device, record.parent_inode) == (
        moved.stat().st_dev,
        moved.stat().st_ino,
    )
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds


def test_collector_parent_swap_fails_without_publishing_to_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    parent = tmp_path / "publish"
    moved = tmp_path / "publish-moved"
    destination = parent / "collected"

    def swap_parent(protocol_path, shortlist_path, gpu, staging):
        parent.rename(moved)
        parent.mkdir()
        (parent / "replacement").write_text("replacement", encoding="utf-8")
        return []

    monkeypatch.setattr(collector, "_collect_shortlist_unpublished", swap_parent)
    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(collector.T4PublicationUncertain, match="preserved") as raised:
        collect_shortlist(Path("protocol"), Path("shortlist"), "0", destination)
    assert (parent / "replacement").read_text(encoding="utf-8") == "replacement"
    assert not destination.exists()
    staging = list(moved.glob(".collected.staging-*"))
    assert len(staging) == 1
    assert next(item for item in raised.value.preserved if item.inode == staging[0].stat().st_ino)
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds


def test_time_wrapper_child_and_cuda_startup_gap_produce_continuous_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.measure_oviv2_tesse_t4 as collector

    leader = None
    original_popen = collector.subprocess.Popen

    def start(*args, **kwargs):
        nonlocal leader
        leader = original_popen(*args, **kwargs)
        return leader

    calls = 0

    def rows(gpu):
        nonlocal calls
        calls += 1
        if calls == 1 or leader is None:
            return []
        children_path = Path(f"/proc/{leader.pid}/task/{leader.pid}/children")
        try:
            children = children_path.read_text(encoding="ascii").split()
        except FileNotFoundError:
            return []
        if not children:
            return []
        child = int(children[0])
        return [{
            "gpu_uuid": "GPU-test", "pid": child,
            "proc_starttime_ticks": collector._proc_starttime(child),
            "process_group_id": os.getpgid(child), "used_memory_mib": 128.0,
        }]

    monkeypatch.setattr(collector.subprocess, "Popen", start)
    monkeypatch.setattr(collector, "_gpu_rows", rows)
    code, samples, events = collector._run_sampled(
        [sys.executable, "-c", "import time; time.sleep(0.45)"],
        gpu="0", phase="mapping", time_log=tmp_path / "time.txt",
    )

    assert code == 0
    assert any(not sample["processes"] for sample in samples)
    assert any(
        process["pid"] != events[0]["pid"]
        for sample in samples for process in sample["processes"]
    )
    _, peaks = collector._validate_gpu_evidence({
        "sample_interval_ms": 200,
        "process_group_id": events[0]["process_group_id"],
        "process_group_ids": [events[0]["process_group_id"]],
        "phase_events": [
            *events,
            {**events[0], "phase": "queries", "timestamp_ns": events[-1]["timestamp_ns"] + 1},
            {**events[-1], "phase": "queries", "timestamp_ns": events[-1]["timestamp_ns"] + 200_000_000},
        ],
        "samples": [
            *samples,
            {**samples[-2], "phase": "queries", "timestamp_ns": events[-1]["timestamp_ns"] + 50_000_000},
            {**samples[-1], "phase": "queries", "timestamp_ns": events[-1]["timestamp_ns"] + 150_000_000},
        ],
    })
    assert max(peaks) == 128.0
