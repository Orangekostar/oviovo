from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluation.tune_oviv2_tesse_dual_readout as tuner


PROFILES = ("a0", "a1", "a2", "a3", "a4")
T4_METRICS = (
    "total_runtime_s_per_frame",
    "query_mean_ms",
    "query_p95_ms",
    "peak_gpu_gb",
    "peak_ram_gb",
    "final_map_mb",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_id": "oviv2-tesse-dual-readout-search-v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "protocol_id": "oviv2-tessecd-v2",
        "development_scene": "apartment",
        "transfer_scene": "office",
        "profile_fallback_order": ["a4", "a3", "a2"],
        "metric_gates": {
            "dynamic_f1": "strictly_increase_from_a0",
            "change_f1": "strictly_increase_from_a0",
            "ghost_rate": "strictly_decrease_from_a0",
            "background_f5_cm": "strictly_increase_from_a0",
            "recovery_frames": "strictly_decrease_from_a0",
        },
        "t4_bounds": {
            "total_runtime_s_per_frame": 6.42,
            "query_mean_ms": 11.92,
            "query_p95_ms": 12.12,
            "peak_gpu_gb": 12.76,
            "peak_ram_gb": 9.36,
            "final_map_mb": 46.77,
        },
        "candidates": [
            {
                "candidate_id": profile,
                "components": {"profile": profile},
                "temporal_readout": {"execution_profile": profile},
            }
            for profile in PROFILES
        ],
        "diagnostic_candidates": [
            {
                "candidate_id": "a4_translation_only",
                "base_profile": "a4",
                "diagnostic": True,
                "selectable": False,
            }
        ],
    }


def _metric(value: float) -> dict[str, object]:
    return {
        "available": True,
        "value": value,
        "reason": None,
        "source": "fixture",
    }


def _candidate(
    candidate_id: str,
    *,
    profile: str | None = None,
    selectable: bool = True,
    metrics: dict[str, float] | None = None,
    opportunities: int = 1,
    triggers: int = 1,
) -> dict[str, object]:
    profile = profile or candidate_id
    values = {
        "current_miou": 0.50,
        "object_f1": 0.60,
        "dynamic_f1": 0.55,
        "change_f1": 0.50,
        "ghost_rate": 0.10,
        "background_f5_cm": 0.50,
        "recovery_frames": 80.0,
        "runtime_seconds": 100.0,
    }
    if candidate_id == "a0":
        values.update(
            dynamic_f1=0.50,
            change_f1=0.45,
            ghost_rate=0.20,
            background_f5_cm=0.40,
            recovery_frames=100.0,
        )
    values.update(metrics or {})
    mechanism = {
        "mechanism": "profile_claim",
        "opportunities": opportunities,
        "triggers": triggers,
        "available": opportunities > 0,
        "passed": opportunities > 0 and triggers > 0,
        "reason": None if opportunities > 0 else "no_opportunity",
        "source": {"path": "/fixture/telemetry.json", "sha256": "1" * 64, "byte_count": 1},
    }
    return {
        "schema_version": 1,
        "manifest_id": "oviv2-tesse-dual-readout-candidate-result-v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "protocol_id": "oviv2-tessecd-v2",
        "scene": "apartment",
        "status": "PASS",
        "candidate_id": candidate_id,
        "profile": {
            "candidate_id": candidate_id,
            "kind": "diagnostic" if not selectable else "main",
            "execution_profile": profile,
            "selectable": selectable,
        },
        "component_map": {"profile": profile},
        "parameter_values": {},
        "bindings": {
            "config_sha256": candidate_id[0] * 63 + candidate_id[-1],
            "non_temporal_config_sha256": "f" * 64,
            "input_hashes": {"dataset": "d" * 64},
            "candidate": {"config": {"temporal_readout": {"execution_profile": profile}}},
        },
        "run_identity": {"algorithm_hash": candidate_id[0] * 63 + candidate_id[-1]},
        "gates": {
            "correctness": {"passed": True},
            "causality": {"passed": True},
            "determinism": {"passed": True},
            "t1_exact": {"passed": True},
            "mechanisms": {"profile_claim": mechanism},
            "anchor_coverage": {
                "scene": "apartment",
                "eligible_count": 66,
                "uniquely_mapped_count": 53,
                "zero_overlap_count": 13,
                "ambiguous_count": 0,
                "required_eligible_count": 66,
                "required_mapped_count": 53,
                "available": True,
                "passed": True,
                "reason": None,
            },
            "t2_metrics": {},
        },
        "metrics": {name: _metric(value) for name, value in values.items()},
        "mechanism_telemetry": {"profile_claim": mechanism},
        "anchor_coverage_gate": {},
        "promotion_evidence": {"selectable": selectable},
        "sources": {"candidate_config": {"path": "/fixture/config.json", "sha256": "2" * 64, "byte_count": 1}},
    } | {
        "gates": {
            "correctness": {"passed": True},
            "causality": {"passed": True},
            "determinism": {"passed": True},
            "t1_exact": {"passed": True},
            "mechanisms": {"profile_claim": mechanism},
            "anchor_coverage": {
                "scene": "apartment",
                "eligible_count": 66,
                "uniquely_mapped_count": 53,
                "zero_overlap_count": 13,
                "ambiguous_count": 0,
                "required_eligible_count": 66,
                "required_mapped_count": 53,
                "available": True,
                "passed": True,
                "reason": None,
            },
            "t2_metrics": {
                name: {
                    "direction": "maximize_strict" if name in {"dynamic_f1", "change_f1", "background_f5_cm"} else "minimize_strict" if name in {"ghost_rate", "recovery_frames"} else "maximize_noninferior",
                    "baseline": (0.50 if name == "dynamic_f1" else 0.45 if name == "change_f1" else 0.20 if name == "ghost_rate" else 0.40 if name == "background_f5_cm" else 100.0 if name == "recovery_frames" else 0.50 if name == "current_miou" else 0.60),
                    "value": values[name],
                    "available": True,
                    "passed": (
                        values[name] >= (0.50 if name == "current_miou" else 0.60)
                        if name in {"current_miou", "object_f1"}
                        else values[name] > (0.50 if name == "dynamic_f1" else 0.45 if name == "change_f1" else 0.40)
                        if name in {"dynamic_f1", "change_f1", "background_f5_cm"}
                        else values[name] < (0.20 if name == "ghost_rate" else 100.0)
                    ),
                    "source": {"path": "/fixture/t2.json", "sha256": "3" * 64, "byte_count": 1},
                }
                for name in ("dynamic_f1", "change_f1", "ghost_rate", "background_f5_cm", "recovery_frames", "current_miou", "object_f1")
            },
        },
        "anchor_coverage_gate": {
            "scene": "apartment", "eligible_count": 66, "uniquely_mapped_count": 53,
            "zero_overlap_count": 13, "ambiguous_count": 0,
            "required_eligible_count": 66, "required_mapped_count": 53,
            "available": True, "passed": True, "reason": None,
        },
    }


@pytest.fixture
def bound_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = _manifest()
    manifest_path = _write(tmp_path / "manifest.json", manifest)
    loaded: dict[Path, dict[str, object]] = {}

    def add(candidate: dict[str, object]) -> Path:
        path = _write(tmp_path / f"{candidate['candidate_id']}.json", {"fixture": candidate["candidate_id"]})
        loaded[path.resolve()] = candidate
        return path

    monkeypatch.setattr(tuner, "load_search_manifest", lambda path: manifest)
    monkeypatch.setattr(
        tuner,
        "load_and_revalidate_result",
        lambda path, *, manifest: loaded[Path(path).resolve()],
    )
    return manifest, manifest_path, add


def test_shortlist_uses_independent_hard_gates_without_compensation(
    tmp_path: Path, bound_inputs
) -> None:
    _, manifest_path, add = bound_inputs
    candidates = [_candidate(profile) for profile in PROFILES]
    candidates[4] = _candidate(
        "a4",
        metrics={
            "dynamic_f1": 0.99,
            "change_f1": 0.99,
            "ghost_rate": 0.20,
            "background_f5_cm": 0.99,
            "recovery_frames": 1.0,
        },
    )
    result = tuner.tune_shortlist(
        manifest_path,
        [add(item) for item in candidates],
        tmp_path / "shortlist.json",
    )

    assert result["phase"] == "shortlist"
    assert result["shortlisted_candidate_ids"] == ["a3", "a2"]
    ledger = {row["candidate_id"]: row for row in result["rejection_ledger"]}
    assert ledger["a4"]["reasons"] == ["ghost_rate_t2_hard_gate_failed"]
    assert "score" not in result
    assert "weighted" not in json.dumps(result).lower()


def test_shortlist_requires_exact_t1_anchor_53_of_66_and_a0_floors(
    tmp_path: Path, bound_inputs
) -> None:
    _, manifest_path, add = bound_inputs
    candidates = [_candidate(profile) for profile in PROFILES]
    candidates[2]["gates"]["t1_exact"]["passed"] = False
    candidates[3]["gates"]["anchor_coverage"]["uniquely_mapped_count"] = 52
    candidates[3]["gates"]["anchor_coverage"]["passed"] = False
    candidates[4]["metrics"]["object_f1"] = _metric(0.59)

    result = tuner.tune_shortlist(
        manifest_path,
        [add(item) for item in candidates],
        tmp_path / "shortlist.json",
    )

    assert result["status"] == "NO_ELIGIBLE_CANDIDATE"
    ledger = {row["candidate_id"]: row for row in result["rejection_ledger"]}
    assert "t1_exact_gate_failed" in ledger["a2"]["reasons"]
    assert "anchor_coverage_gate_failed" in ledger["a3"]["reasons"]
    assert "object_f1_below_a0_floor" in ledger["a4"]["reasons"]


def test_zero_opportunity_cannot_support_claim_and_diagnostic_is_never_selectable(
    tmp_path: Path, bound_inputs
) -> None:
    manifest, manifest_path, add = bound_inputs
    candidates = [_candidate(profile) for profile in PROFILES]
    candidates[4] = _candidate("a4", opportunities=0, triggers=0)
    diagnostic = _candidate(
        "a4_translation_only", profile="a4", selectable=False
    )
    result = tuner.tune_shortlist(
        manifest_path,
        [add(item) for item in [*candidates, diagnostic]],
        tmp_path / "shortlist.json",
    )

    assert result["shortlisted_candidate_ids"] == ["a3", "a2"]
    ledger = {row["candidate_id"]: row for row in result["rejection_ledger"]}
    assert "mechanism_profile_claim_no_opportunity" in ledger["a4"]["reasons"]
    assert ledger["a4_translation_only"]["eligible"] is False
    assert "diagnostic_candidate_not_selectable" in ledger["a4_translation_only"]["reasons"]


def test_shortlist_preserves_fixed_profile_fallback_order(
    tmp_path: Path, bound_inputs
) -> None:
    _, manifest_path, add = bound_inputs
    candidates = [_candidate(profile) for profile in PROFILES]
    result = tuner.tune_shortlist(
        manifest_path,
        [add(item) for item in reversed(candidates)],
        tmp_path / "shortlist.json",
    )
    assert result["shortlisted_candidate_ids"] == ["a4", "a3", "a2"]
    assert result["profile_fallback_order"] == ["a4", "a3", "a2"]


def test_shortlist_records_unpublished_failed_profile_and_continues_fallback(
    tmp_path: Path, bound_inputs
) -> None:
    _, manifest_path, add = bound_inputs
    candidates = [_candidate(profile) for profile in PROFILES if profile != "a4"]

    result = tuner.tune_shortlist(
        manifest_path,
        [add(item) for item in candidates],
        tmp_path / "shortlist.json",
    )

    assert result["shortlisted_candidate_ids"] == ["a3", "a2"]
    ledger = {row["candidate_id"]: row for row in result["rejection_ledger"]}
    assert ledger["a4"]["reasons"] == ["candidate_result_unavailable"]


def _t4_matrix(
    tmp_path: Path,
    manifest: dict[str, object],
    shortlist_path: Path,
    shortlisted: list[dict[str, object]],
    *,
    failed: set[str] = frozenset(),
) -> tuple[Path, dict[str, Path]]:
    sources: dict[str, Path] = {}
    candidates_matrix = {}
    protocol_candidates = {}
    for candidate in shortlisted:
        candidate_id = str(candidate["candidate_id"])
        source = _write(
            tmp_path / f"{candidate_id}-t4-run.json",
            {
                "schema_version": 2,
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "protocol_id": "oviv2-tessecd-v2",
                "scene": "apartment",
                "candidate_id": candidate_id,
                "config_sha256": candidate["config_sha256"],
            },
        )
        sources[candidate_id] = source
        run_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        metrics = {}
        gates = {}
        metric_sources = {}
        for name, bound in manifest["t4_bounds"].items():
            value = float(bound) + (1.0 if candidate_id in failed and name == "final_map_mb" else -0.01)
            metrics[name] = value
            gates[name] = value <= float(bound)
            metric_source = _write(
                tmp_path / f"{candidate_id}-{name}.json",
                {
                    "schema_version": 1,
                    "manifest_id": "oviv2_tesse_t4_metric_v1",
                    "scene": "apartment",
                    "candidate_id": candidate_id,
                    "config_sha256": candidate["config_sha256"],
                    "run_manifest_sha256": run_sha256,
                    "metric": name,
                    "value": value,
                },
            )
            sources[f"{candidate_id}:{name}"] = metric_source
            metric_sources[name] = _record(metric_source)
        candidates_matrix[candidate_id] = {
                "config_sha256": candidate["config_sha256"],
                "run_manifest_sha256": run_sha256,
                "metrics": metrics,
                "gates": gates,
                "status": "FAIL" if candidate_id in failed else "PASS",
            }
        protocol_candidates[candidate_id] = {
            "config_sha256": candidate["config_sha256"],
            "run_manifest": _record(source),
            "metric_sources": metric_sources,
        }
    protocol_path = _write(
        tmp_path / "t4-protocol.json",
        {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_t4_protocol_v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment",
            "bounds": manifest["t4_bounds"],
            "candidates": protocol_candidates,
        },
    )
    matrix = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_t4_matrix_v1",
        "status": "PASS",
        "shortlist": _record(shortlist_path),
        "protocol": _record(protocol_path),
        "candidates": candidates_matrix,
    }
    matrix["root_sha256"] = hashlib.sha256(_canonical(matrix)).hexdigest()
    return _write(tmp_path / "t4-matrix.json", matrix), sources


@pytest.mark.parametrize(
    ("failed", "selected"),
    [({"a4"}, "a3"), ({"a4", "a3"}, "a2")],
)
def test_final_uses_whole_profile_t4_fallback(
    tmp_path: Path, bound_inputs, failed: set[str], selected: str
) -> None:
    manifest, manifest_path, add = bound_inputs
    shortlist_path = tmp_path / "shortlist.json"
    shortlist = tuner.tune_shortlist(
        manifest_path,
        [add(_candidate(profile)) for profile in PROFILES],
        shortlist_path,
    )
    matrix_path, _ = _t4_matrix(
        tmp_path, manifest, shortlist_path, shortlist["shortlisted_candidates"], failed=failed
    )

    final = tuner.tune_final(manifest_path, shortlist_path, matrix_path, tmp_path / "final.json")

    assert final["phase"] == "final"
    assert final["selected_candidate_id"] == selected
    assert final["office_results_read"] is False
    assert final["scenes_read"] == ["apartment"]


def test_final_rejects_cell_splicing_and_unbound_sources(
    tmp_path: Path, bound_inputs
) -> None:
    manifest, manifest_path, add = bound_inputs
    shortlist_path = tmp_path / "shortlist.json"
    shortlist = tuner.tune_shortlist(
        manifest_path,
        [add(_candidate(profile)) for profile in PROFILES],
        shortlist_path,
    )
    matrix_path, sources = _t4_matrix(
        tmp_path, manifest, shortlist_path, shortlist["shortlisted_candidates"]
    )
    matrix = json.loads(matrix_path.read_text())
    matrix["candidates"]["a4"]["run_manifest_sha256"] = hashlib.sha256(
        sources["a3"].read_bytes()
    ).hexdigest()
    matrix.pop("root_sha256")
    matrix["root_sha256"] = hashlib.sha256(_canonical(matrix)).hexdigest()
    matrix_path.write_bytes(_canonical(matrix) + b"\n")

    with pytest.raises(ValueError, match="whole-profile run"):
        tuner.tune_final(manifest_path, shortlist_path, matrix_path, tmp_path / "final.json")


def test_final_rejects_shortlist_hash_drift_and_no_clobber(
    tmp_path: Path, bound_inputs
) -> None:
    manifest, manifest_path, add = bound_inputs
    shortlist_path = tmp_path / "shortlist.json"
    shortlist = tuner.tune_shortlist(
        manifest_path,
        [add(_candidate(profile)) for profile in PROFILES],
        shortlist_path,
    )
    matrix_path, _ = _t4_matrix(
        tmp_path, manifest, shortlist_path, shortlist["shortlisted_candidates"]
    )
    shortlist_path.write_bytes(shortlist_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="shortlist source binding mismatch"):
        tuner.tune_final(manifest_path, shortlist_path, matrix_path, tmp_path / "final.json")

    occupied = tmp_path / "occupied.json"
    occupied.write_text("occupied")
    with pytest.raises(FileExistsError):
        tuner.tune_final(manifest_path, shortlist_path, matrix_path, occupied)


def test_final_rejects_nonhex_run_manifest_hash(
    tmp_path: Path, bound_inputs
) -> None:
    manifest, manifest_path, add = bound_inputs
    shortlist_path = tmp_path / "shortlist.json"
    shortlist = tuner.tune_shortlist(
        manifest_path,
        [add(_candidate(profile)) for profile in PROFILES],
        shortlist_path,
    )
    matrix_path, _ = _t4_matrix(
        tmp_path, manifest, shortlist_path, shortlist["shortlisted_candidates"]
    )
    matrix = json.loads(matrix_path.read_text())
    matrix["candidates"]["a4"]["run_manifest_sha256"] = "z" * 64
    matrix.pop("root_sha256")
    matrix["root_sha256"] = hashlib.sha256(_canonical(matrix)).hexdigest()
    matrix_path.write_bytes(_canonical(matrix) + b"\n")

    with pytest.raises(ValueError, match="run manifest hash is invalid"):
        tuner.tune_final(manifest_path, shortlist_path, matrix_path, tmp_path / "final.json")


def test_final_rejects_t4_metric_source_drift(
    tmp_path: Path, bound_inputs
) -> None:
    manifest, manifest_path, add = bound_inputs
    shortlist_path = tmp_path / "shortlist.json"
    shortlist = tuner.tune_shortlist(
        manifest_path,
        [add(_candidate(profile)) for profile in PROFILES],
        shortlist_path,
    )
    matrix_path, sources = _t4_matrix(
        tmp_path, manifest, shortlist_path, shortlist["shortlisted_candidates"]
    )
    sources["a4:query_mean_ms"].write_text('{"tampered":true}\n')

    with pytest.raises(ValueError, match="T4 metric.*source binding mismatch"):
        tuner.tune_final(manifest_path, shortlist_path, matrix_path, tmp_path / "final.json")


def test_phase_cli_has_conditionally_required_inputs() -> None:
    shortlist = tuner.parse_args(
        [
            "--phase", "shortlist", "--manifest", "manifest.json",
            "--results-root", "results", "--output", "shortlist.json",
        ]
    )
    assert shortlist.phase == "shortlist"
    final = tuner.parse_args(
        [
            "--phase", "final", "--manifest", "manifest.json",
            "--shortlist", "shortlist.json", "--t4-matrix", "t4.json",
            "--output", "final.json",
        ]
    )
    assert final.phase == "final"
    with pytest.raises(SystemExit):
        tuner.parse_args(
            ["--phase", "final", "--manifest", "manifest.json", "--output", "final.json"]
        )


def test_atomic_publish_handles_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "selection.json"
    real_write = tuner.os.write

    def partial_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, data[:3])

    monkeypatch.setattr(tuner.os, "write", partial_write)
    tuner._atomic_write_new(output, {"status": "PASS", "candidate": "a4"})

    assert json.loads(output.read_text()) == {"status": "PASS", "candidate": "a4"}


def test_atomic_publish_cleans_temporary_file_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        tuner.os, "write", lambda descriptor, data: (_ for _ in ()).throw(OSError("write failure"))
    )

    with pytest.raises(OSError, match="write failure"):
        tuner._atomic_write_new(output, {"status": "PASS"})

    assert not output.exists()
    assert not list(tmp_path.glob(".selection.json.tmp-*"))
