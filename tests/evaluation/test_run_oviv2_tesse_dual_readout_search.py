from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

import scripts.evaluation.run_oviv2_tesse_dual_readout_search as search_runner
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    load_search_manifest,
    run_search,
)
from src.oviv2.temporal_config import ExecutionProfile, temporal_config_from_json


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json"
)
APARTMENT_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_apartment_v2.json"
OFFICE_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_office_v2.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _preflight(tmp_path: Path, candidate_ids: tuple[str, ...]) -> Path:
    path = tmp_path / f"preflight-{'-'.join(candidate_ids)}.json"
    manifest = json.loads(MANIFEST.read_text())
    base = json.loads(APARTMENT_CONFIG.read_text())
    declarations = {item["candidate_id"]: item for item in manifest["candidates"]}
    mechanisms = {
        "a0": [],
        "a1": ["absence", "readout_invalidation"],
        "a2": [
            "absence",
            "readout_invalidation",
            "proposal_recovery",
            "epoch_reset",
            "motion_rejection",
        ],
        "a3": [
            "absence",
            "readout_invalidation",
            "proposal_recovery",
            "epoch_reset",
            "motion_rejection",
            "background_release",
            "background_reclaim",
        ],
        "a4": [
            "absence",
            "readout_invalidation",
            "proposal_recovery",
            "epoch_reset",
            "motion_rejection",
            "background_release",
            "background_reclaim",
            "eligible_reid",
            "icp",
        ],
    }
    candidates = []
    for candidate_id in candidate_ids:
        materialized = search_runner._materialize_config(base, declarations[candidate_id])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_config_sha256": hashlib.sha256(
                    _canonical(materialized)
                ).hexdigest(),
                "frame_coverage": {
                    "expected_frame_indices": [0, 1, 2],
                    "observed_frame_indices": [0, 1, 2],
                    "expected_count": 3,
                    "observed_count": 3,
                },
                "future_leakage": {"count": 0, "records": []},
                "anchor_coverage": {
                    "eligible_anchor_ids": list(range(66)),
                    "mapped_anchor_ids": list(range(53)),
                    "eligible_count": 66,
                    "mapped_count": 53,
                },
                "mechanisms": {
                    name: {
                        "opportunity_count": 1,
                        "trigger_count": 1,
                        "opportunity_records": [f"{candidate_id}:{name}:opportunity"],
                        "trigger_records": [f"{candidate_id}:{name}:trigger"],
                    }
                    for name in mechanisms[candidate_id]
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "oviv2_dual_readout_search_preflight_v1",
                "scene": "apartment",
                "bindings": {
                    "search_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                    "apartment_base_config_file_sha256": hashlib.sha256(
                        APARTMENT_CONFIG.read_bytes()
                    ).hexdigest(),
                },
                "candidates": candidates,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_manifest_predeclares_exact_bounded_causal_matrix_and_gates() -> None:
    manifest = load_search_manifest(MANIFEST)

    assert [candidate["candidate_id"] for candidate in manifest["candidates"]] == [
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
    ]
    assert manifest["development_scene"] == "apartment"
    assert manifest["transfer_scene"] == "office"
    assert manifest["transfer_policy"] == "bind_only_never_execute"
    assert manifest["metric_directions"] == {
        "background_f5_cm": "maximize",
        "change_f1": "maximize",
        "current_miou": "maximize",
        "dynamic_f1": "maximize",
        "ghost_rate": "minimize",
        "object_f1": "maximize",
        "recovery_frames": "minimize",
        "runtime_seconds": "minimize",
    }
    assert manifest["metric_policy"]["selection_required"] == [
        "current_miou",
        "object_f1",
        "ghost_rate",
        "background_f5_cm",
        "recovery_frames",
        "runtime_seconds",
    ]
    assert manifest["promotion_order"] == [
        "t1_exact_gate",
        "current_miou_floor",
        "object_f1_floor",
        "ghost_rate",
        "background_f5_cm",
        "recovery_frames",
        "dynamic_f1",
        "change_f1",
        "runtime_seconds",
        "config_sha256",
    ]
    for candidate in manifest["candidates"]:
        profile = ExecutionProfile.from_id(candidate["candidate_id"])
        parsed = temporal_config_from_json(
            {"temporal_readout": candidate["temporal_readout"]}
        )
        assert parsed.execution_profile is profile
        assert candidate["components"] == profile.components
        assert set(candidate["temporal_readout"]) == {
            "execution_profile",
            "components",
            "lifecycle",
            "association",
            "geometry",
            "proposal",
            "identity",
            "dynamic_state",
            "motion",
            "geometry_epoch",
            "background_ledger",
        }
    spaces = manifest["parameter_spaces"]
    assert set(spaces) == {
        "lifecycle.minimum_absent_streak",
        "proposal.minimum_depth_residual_m",
        "dynamic_state.minimum_motion_confidence",
        "identity.minimum_reid_similarity",
        "identity.maximum_reid_distance_m",
        "motion.minimum_translation_confidence",
        "motion.maximum_translation_residual_m",
        "background_ledger.commit_support_frames",
        "background_ledger.commit_distinct_view_bins",
    }
    assert all(1 <= len(space["values"]) <= 3 for space in spaces.values())
    assert all(len(space["values"]) == len(set(space["values"])) for space in spaces.values())
    for candidate in manifest["candidates"]:
        profile_id = candidate["candidate_id"]
        temporal = candidate["temporal_readout"]
        for field, space in spaces.items():
            if profile_id not in space["profiles"]:
                continue
            group, name = field.split(".", 1)
            assert temporal[group][name] in space["values"]

    assert manifest["gpu_lanes"] == {
        "lane0": ["a0", "a1"],
        "lane1": ["a2"],
        "lane2": ["a3", "a4"],
    }
    assert manifest["profile_fallback_order"] == ["a4", "a3", "a2"]
    assert manifest["metric_gates"]["apartment_anchor_coverage"] == {
        "minimum_mapped_anchors": 53,
        "eligible_anchors": 66,
    }


def test_manifest_registers_micro_ablations_as_fail_closed_diagnostics() -> None:
    manifest = load_search_manifest(MANIFEST)
    diagnostics = manifest["diagnostic_candidates"]
    assert [item["candidate_id"] for item in diagnostics] == [
        "diag_a2_no_proposal_recovery",
        "diag_a3_masking_only_no_ledger",
        "diag_a4_no_dormant_candidates",
        "diag_a4_translation_only_no_icp",
    ]
    assert [item["base_profile"] for item in diagnostics] == ["a2", "a3", "a4", "a4"]
    assert all(item["diagnostic"] is True for item in diagnostics)
    assert all(item["selectable"] is False for item in diagnostics)
    assert all(item["runnable"] is False for item in diagnostics)
    assert all(item["reason"] == "unsupported_runtime_control" for item in diagnostics)
    assert all(item["required_runtime_control"] for item in diagnostics)
    assert not (
        {item["candidate_id"] for item in diagnostics}
        & {item["candidate_id"] for item in manifest["candidates"]}
    )


def test_manifest_rejects_unknown_and_profile_incompatible_parameter_spaces(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["parameter_spaces"]["proposal.unknown"] = {
        "profiles": ["a2"],
        "values": [1],
    }
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parameter_spaces.*unknown"):
        load_search_manifest(unknown)

    payload = json.loads(MANIFEST.read_text())
    payload["parameter_spaces"]["proposal.minimum_depth_residual_m"]["profiles"] = [
        "a1",
        "a2",
    ]
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="profile-incompatible"):
        load_search_manifest(incompatible)


def test_execution_profiles_have_distinct_canonical_behavior() -> None:
    components = {
        tuple(profile.components.items())
        for profile in ExecutionProfile
    }
    assert len(components) == 5


def test_manifest_rejects_duplicate_json_keys_and_undeclared_candidate(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_search_manifest(duplicate)

    with pytest.raises(ValueError, match="undeclared candidate"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "out",
            gpu_ids=("0",),
            max_parallel=1,
            candidate_ids=("a5",),
            available_ram_bytes=10**15,
        )


def test_runner_materializes_unique_apartment_configs_and_never_runs_office(
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, ...]] = []

    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        command = (
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ok')",
            str(output.parent / "executed.txt"),
            candidate_id,
        )
        commands.append(command)
        return command

    root = tmp_path / "search"
    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=root,
        gpu_ids=("2", "4", "6"),
        max_parallel=3,
        preflight_gate_evidence=_preflight(tmp_path, ("a0", "a1", "a2", "a3", "a4")),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    status = json.loads(status_path.read_text())

    assert status["status"] == "PASS"
    assert [record["candidate_id"] for record in status["candidates"]] == [
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
    ]
    assert all(record["scene"] == "apartment" for record in status["candidates"])
    assert all(record["pid"] > 0 and record["exit_code"] == 0 for record in status["candidates"])
    assert all(record["runtime_seconds"] >= 0.0 for record in status["candidates"])
    assert all(
        record["config_file"]["path"] == record["config_path"]
        for record in status["candidates"]
    )
    assert all(
        record["stdout_file"]["path"] == record["stdout_path"]
        for record in status["candidates"]
    )
    assert all(
        record["stderr_file"]["path"] == record["stderr_path"]
        for record in status["candidates"]
    )
    assert {
        record["candidate_id"]: record["cuda_visible_devices"]
        for record in status["candidates"]
    } == {
        "a0": "2",
        "a1": "2",
        "a2": "4",
        "a3": "6",
        "a4": "6",
    }
    assert len({record["output_root"] for record in status["candidates"]}) == 5
    assert len({record["config_path"] for record in status["candidates"]}) == 5
    assert status["office_binding"]["scene"] == "office"
    assert status["office_binding"]["executed"] is False
    assert status["preflight_gate_evidence"]["path"].endswith(
        "preflight-a0-a1-a2-a3-a4.json"
    )
    assert len(status["preflight_gate_evidence"]["sha256"]) == 64
    assert all("office" not in " ".join(command).lower() for command in commands)
    for record in status["candidates"]:
        config_path = Path(record["config_path"])
        config = json.loads(config_path.read_text())
        assert config["scene"] == "apartment"
        assert config["temporal_readout"]["execution_profile"] == record["candidate_id"]
        assert (
            temporal_config_from_json(
                {"temporal_readout": config["temporal_readout"]}
            ).execution_profile.profile_id
            == record["candidate_id"]
        )
        assert hashlib.sha256(_canonical(config)).hexdigest() == record["config_sha256"]

    with pytest.raises(FileExistsError):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=root,
            gpu_ids=("0", "1", "2"),
            max_parallel=3,
            preflight_gate_evidence=_preflight(
                tmp_path, ("a0", "a1", "a2", "a3", "a4")
            ),
            available_ram_bytes=10**15,
        )


def test_runner_stops_new_scheduling_after_first_failure(tmp_path: Path) -> None:
    root = tmp_path / "failed"

    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-c",
            "import sys; sys.exit(7 if sys.argv[1] == 'a1' else 0)",
            candidate_id,
        )

    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=root,
        gpu_ids=("0",),
        max_parallel=1,
        preflight_gate_evidence=_preflight(tmp_path, ("a0", "a1")),
        candidate_ids=("a0", "a1"),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    status = json.loads(status_path.read_text())
    assert status["status"] == "FAIL"
    assert [record["candidate_id"] for record in status["candidates"]] == ["a0", "a1"]
    assert status["candidates"][-1]["exit_code"] == 7
    assert status["unscheduled_candidate_ids"] == []


def test_parallel_runtime_has_no_head_of_line_wait_inflation(tmp_path: Path) -> None:
    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        del config, output
        delay = "0.35" if candidate_id == "a0" else "0.05" if candidate_id == "a2" else "0"
        return (sys.executable, "-c", "import sys,time; time.sleep(float(sys.argv[1]))", delay)

    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=tmp_path / "parallel",
        gpu_ids=("0", "1", "2"),
        max_parallel=3,
        candidate_ids=("a0", "a2"),
        preflight_gate_evidence=_preflight(tmp_path, ("a0", "a2")),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    records = {
        item["candidate_id"]: item for item in json.loads(status_path.read_text())["candidates"]
    }
    assert records["a2"]["runtime_seconds"] < 0.25
    assert records["a2"]["runtime_seconds"] < records["a0"]["runtime_seconds"]


def test_fixed_gpu_lanes_serialize_a0_a1_and_a3_a4(tmp_path: Path) -> None:
    lane0_done = tmp_path / "lane0.done"
    lane2_done = tmp_path / "lane2.done"

    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        del config, output
        if candidate_id in {"a0", "a3"}:
            marker = lane0_done if candidate_id == "a0" else lane2_done
            return (
                sys.executable,
                "-c",
                "import pathlib,sys,time; time.sleep(.1); pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            )
        if candidate_id in {"a1", "a4"}:
            marker = lane0_done if candidate_id == "a1" else lane2_done
            return (
                sys.executable,
                "-c",
                "import pathlib,sys; "
                "raise SystemExit(0 if pathlib.Path(sys.argv[1]).is_file() else 9)",
                str(marker),
            )
        return (sys.executable, "-c", "raise SystemExit(0)")

    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=tmp_path / "serialized",
        gpu_ids=("0", "1", "2"),
        max_parallel=3,
        preflight_gate_evidence=_preflight(
            tmp_path, ("a0", "a1", "a2", "a3", "a4")
        ),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    assert json.loads(status_path.read_text())["status"] == "PASS"


def test_preflight_recomputes_source_coverage_and_mechanism_records(
    tmp_path: Path,
) -> None:
    gate_path = _preflight(tmp_path, ("a2",))
    payload = json.loads(gate_path.read_text())
    payload["bindings"]["search_manifest_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source bindings mismatch"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "bad-binding",
            gpu_ids=("0", "1"),
            candidate_ids=("a2",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )

    gate_path = _preflight(tmp_path, ("a2",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["frame_coverage"]["observed_frame_indices"] = [0, 1]
    payload["candidates"][0]["frame_coverage"]["observed_count"] = 2
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frame_coverage did not PASS"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "bad-coverage",
            gpu_ids=("0", "1"),
            candidate_ids=("a2",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )

    gate_path = _preflight(tmp_path, ("a2",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["mechanisms"]["proposal_recovery"][
        "opportunity_count"
    ] = 2
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="opportunity_count does not match"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "bad-mechanism",
            gpu_ids=("0", "1"),
            candidate_ids=("a2",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )


def test_default_runner_requires_run_manifest_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        search_runner,
        "_default_command",
        lambda config, output, candidate_id: (
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ),
    )

    with pytest.raises(ValueError, match="run manifest"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "missing-run-manifest",
            gpu_ids=("0",),
            max_parallel=1,
            candidate_ids=("a0",),
            preflight_gate_evidence=_preflight(tmp_path, ("a0",)),
            available_ram_bytes=10**15,
        )


def test_runner_requires_all_preflight_gates_before_output_creation(tmp_path: Path) -> None:
    output = tmp_path / "missing-gates"
    with pytest.raises(ValueError, match="preflight gate evidence is required"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=output,
            gpu_ids=("0",),
            candidate_ids=("a0",),
            available_ram_bytes=10**15,
        )
    assert not output.exists()

    gate_path = _preflight(tmp_path, ("a0",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["future_leakage"] = {
        "count": 1,
        "records": ["a0:frame2:query3"],
    }
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="future_leakage.*PASS"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=output,
            gpu_ids=("0",),
            candidate_ids=("a0",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )
    assert not output.exists()


def test_runner_enforces_apartment_anchor_53_of_66_before_launch(tmp_path: Path) -> None:
    gate_path = _preflight(tmp_path, ("a0",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["anchor_coverage"]["mapped_anchor_ids"] = list(range(52))
    payload["candidates"][0]["anchor_coverage"]["mapped_count"] = 52
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="anchor coverage.*53/66"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "anchor-fail",
            gpu_ids=("0",),
            candidate_ids=("a0",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )


def test_runner_rejects_diagnostic_candidates_and_office_search(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="diagnostic candidate.*not runnable"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "diagnostic",
            gpu_ids=("0", "1"),
            candidate_ids=("diag_a2_no_proposal_recovery",),
            available_ram_bytes=10**15,
        )
    with pytest.raises(ValueError, match="Office search requires frozen authorization"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "office",
            gpu_ids=("0",),
            scene="office",
            available_ram_bytes=10**15,
        )


@pytest.mark.parametrize("max_parallel", [0, 4, True])
def test_runner_rejects_invalid_parallelism(tmp_path: Path, max_parallel: object) -> None:
    with pytest.raises(ValueError, match="max_parallel"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "out",
            gpu_ids=("0",),
            max_parallel=max_parallel,  # type: ignore[arg-type]
            available_ram_bytes=10**15,
        )


def test_runner_checks_available_ram_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="available RAM"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=output,
            gpu_ids=("0",),
            max_parallel=1,
            candidate_ids=("a0",),
            available_ram_bytes=0,
        )
    assert not output.exists()


def test_available_ram_uses_linux_memavailable_including_reclaimable_cache(
    tmp_path: Path,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       1000000 kB\n"
        "MemFree:          12000 kB\n"
        "MemAvailable:    800000 kB\n"
        "Cached:          700000 kB\n"
    )

    assert search_runner._available_ram_bytes(meminfo) == 800000 * 1024
