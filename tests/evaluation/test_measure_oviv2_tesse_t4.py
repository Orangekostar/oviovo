from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
            "samples": [
                {"timestamp_ns": 1_000_000_000, "phase": "mapping", "pid": 20, "process_group_id": 17, "used_memory_mib": 3000,
                 "processes": [{"gpu_uuid": "GPU-fixture", "pid": 20, "proc_starttime_ticks": 200,
                                "process_group_id": 17, "used_memory_mib": 3000}]},
                {"timestamp_ns": 1_200_000_000, "phase": "queries", "pid": 21, "process_group_id": 17, "used_memory_mib": 4000,
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
    vocabulary = REPO_ROOT / protocol["query"]["vocabulary_path"]
    assert _sha256(vocabulary) == protocol["query"]["vocabulary_sha256"]


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
            return 0, [{"timestamp_ns": 1_000_000_000, "phase": "mapping", "pid": 101, "process_group_id": 101, "used_memory_mib": 2000,
                        "processes": [{"gpu_uuid": "GPU-fake", "pid": 101, "proc_starttime_ticks": 1,
                                       "process_group_id": 101, "used_memory_mib": 2000}]}]
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
        return 0, [{"timestamp_ns": 1_200_000_000, "phase": "queries", "pid": 202, "process_group_id": 202, "used_memory_mib": 3000,
                    "processes": [{"gpu_uuid": "GPU-fake", "pid": 202, "proc_starttime_ticks": 2,
                                   "process_group_id": 202, "used_memory_mib": 3000}]}]

    monkeypatch.setattr("scripts.evaluation.measure_oviv2_tesse_t4._run_sampled", fake_run)
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
