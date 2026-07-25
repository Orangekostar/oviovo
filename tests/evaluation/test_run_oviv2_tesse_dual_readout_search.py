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
        assert candidate["components"] == {
            "association_mode": profile.association_mode,
            "background_mode": profile.background_mode,
            "geometry_mode": profile.geometry_mode,
            "lifecycle_mode": profile.lifecycle_mode,
            "motion_mode": profile.motion_mode,
        }
    bounds = manifest["parameter_bounds"]
    for candidate in manifest["candidates"]:
        for group, values in candidate["temporal_readout"].items():
            if group == "execution_profile":
                continue
            for name, value in values.items():
                lower, upper = bounds[f"{group}.{name}"]
                assert lower <= value <= upper


def test_execution_profiles_have_distinct_canonical_behavior() -> None:
    components = {
        (
            profile.lifecycle_mode,
            profile.geometry_mode,
            profile.background_mode,
            profile.association_mode,
            profile.motion_mode,
        )
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
    assert all(record["config_file"]["path"] == record["config_path"] for record in status["candidates"])
    assert all(record["stdout_file"]["path"] == record["stdout_path"] for record in status["candidates"])
    assert all(record["stderr_file"]["path"] == record["stderr_path"] for record in status["candidates"])
    assert {record["cuda_visible_devices"] for record in status["candidates"]} <= {
        "2",
        "4",
        "6",
    }
    assert len({record["output_root"] for record in status["candidates"]}) == 5
    assert len({record["config_path"] for record in status["candidates"]}) == 5
    assert status["office_binding"]["scene"] == "office"
    assert status["office_binding"]["executed"] is False
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
            gpu_ids=("0",),
            max_parallel=1,
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
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    status = json.loads(status_path.read_text())
    assert status["status"] == "FAIL"
    assert [record["candidate_id"] for record in status["candidates"]] == ["a0", "a1"]
    assert status["candidates"][-1]["exit_code"] == 7
    assert status["unscheduled_candidate_ids"] == ["a2", "a3", "a4"]


def test_parallel_runtime_has_no_head_of_line_wait_inflation(tmp_path: Path) -> None:
    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        del config, output
        delay = "0.35" if candidate_id == "a0" else "0.05" if candidate_id == "a1" else "0"
        return (sys.executable, "-c", "import sys,time; time.sleep(float(sys.argv[1]))", delay)

    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=tmp_path / "parallel",
        gpu_ids=("0", "1"),
        max_parallel=2,
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    records = {
        item["candidate_id"]: item for item in json.loads(status_path.read_text())["candidates"]
    }
    assert records["a1"]["runtime_seconds"] < 0.25
    assert records["a1"]["runtime_seconds"] < records["a0"]["runtime_seconds"]


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
