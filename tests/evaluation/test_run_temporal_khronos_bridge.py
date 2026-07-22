from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluation.run_temporal_khronos_bridge as bridge_runner
from scripts.evaluation.run_temporal_khronos_bridge import (
    build_bridge_command,
    build_package_command,
    parse_args,
    run,
    snapshot_bridge_input,
    stage_importer_source,
    validate_build_manifest,
    write_build_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
CPP = (
    ROOT
    / "scripts/evaluation/compat/khronos_temporal_bridge/import_temporal_baseline.cpp"
)


def _fake_workspace(root: Path) -> Path:
    package = root / "src/khronos/khronos_eval"
    (package / "app").mkdir(parents=True)
    (package / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.22.1)\n"
        "project(khronos_eval)\n"
        "set(PROJECT_NAME khronos_eval)\n"
        "set(gflags_LIBRARIES gflags)\n"
        "# ##############################################################################\n"
        "# Export #\n"
        "# ##############################################################################\n",
        encoding="utf-8",
    )
    return root


def test_cpp_importer_uses_stable_symbols_intervals_trajectory_and_ordered_updates() -> None:
    text = CPP.read_text(encoding="utf-8")

    assert "#include <map>" in text
    assert "#include <vector>" in text
    assert "NodeSymbol('O', entry.at(\"node_index\")" in text
    assert "first_observed_ns" in text
    assert "last_observed_ns" in text
    assert "presence interval endpoint vectors are invalid" in text
    assert "trajectory_timestamps" in text
    assert "trajectory_positions" in text
    assert "dynamic_track_eligible" in text
    assert 'manifest.value("dataset", "") != "TESSE-CD"' in text
    assert 'manifest.value("method", "") != "OVIV2"' in text
    assert "resolveBridgePath" in text
    assert "map.update(buildGraph(checkpoint), timestamp_ns);" in text
    assert "map_timestamps.json" in text


def test_stage_and_build_command_target_only_khronos_eval(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")

    audit = stage_importer_source(CPP, workspace)
    command = build_package_command(
        conda=Path("/conda"), environment="khronos", workspace=workspace
    )

    staged = workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp"
    assert staged.read_bytes() == CPP.read_bytes()
    cmake = (workspace / "src/khronos/khronos_eval/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    assert cmake.count("OVIV2_KHRONOS_TEMPORAL_BRIDGE_BEGIN") == 1
    assert "add_executable(import_temporal_baseline" in cmake
    assert audit["source"]["sha256"] == hashlib.sha256(CPP.read_bytes()).hexdigest()
    joined = " ".join(command)
    assert "colcon build" in joined
    assert "--packages-select khronos_eval" in joined


def test_build_manifest_and_run_command_are_hash_bound(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    executable = workspace / "install/khronos_eval/lib/khronos_eval/import_temporal_baseline"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"executable")
    staged = workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp"
    build_manifest = write_build_manifest(
        CPP,
        staged,
        workspace / "src/khronos/khronos_eval/CMakeLists.txt",
        executable,
        tmp_path / "build_manifest.json",
    )

    payload = validate_build_manifest(build_manifest, expected_source=CPP)
    command = build_bridge_command(
        conda=Path("/conda"),
        environment="khronos",
        workspace=workspace,
        manifest=Path("/bridge/manifest.json"),
        output=Path("/run/map"),
    )

    assert payload["executable"]["sha256"] == hashlib.sha256(b"executable").hexdigest()
    assert payload["staged_source"]["sha256"] == payload["source"]["sha256"]
    assert command[-2:] == ["/bridge/manifest.json", "/run/map"]
    assert str(executable) in command

    executable.write_bytes(b"tamperable")
    with pytest.raises(ValueError, match="executable SHA256"):
        validate_build_manifest(build_manifest, expected_source=CPP)


def test_stage_migrates_exact_legacy_cmake_block_once(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    cmake_path = workspace / "src/khronos/khronos_eval/CMakeLists.txt"
    legacy = (
        "# OVIOVO_KHRONOS_TEMPORAL_BRIDGE_BEGIN\n"
        "add_executable(import_temporal_baseline app/import_temporal_baseline.cpp)\n"
        "target_link_libraries(import_temporal_baseline ${PROJECT_NAME} ${gflags_LIBRARIES})\n"
        "install(TARGETS import_temporal_baseline RUNTIME DESTINATION lib/${PROJECT_NAME})\n"
        "# OVIOVO_KHRONOS_TEMPORAL_BRIDGE_END\n\n"
    )
    cmake_path.write_text(
        cmake_path.read_text(encoding="utf-8").replace(
            "# Export #\n", legacy + "# Export #\n"
        ),
        encoding="utf-8",
    )

    stage_importer_source(CPP, workspace)
    content = cmake_path.read_text(encoding="utf-8")

    assert "OVIOVO_KHRONOS_TEMPORAL_BRIDGE" not in content
    assert content.count("OVIV2_KHRONOS_TEMPORAL_BRIDGE_BEGIN") == 1
    assert content.count("add_executable(import_temporal_baseline") == 1


def test_build_manifest_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.cpp"
    cmake = tmp_path / "CMakeLists.txt"
    executable = tmp_path / "import_temporal_baseline"
    for path in (source, cmake, executable):
        path.write_text(path.name, encoding="utf-8")
    destination = tmp_path / "build_manifest.json"
    destination.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_build_manifest(source, source, cmake, executable, destination)

    assert destination.read_text(encoding="utf-8") == "preserve\n"


def test_stage_rejects_symlinked_importer_target(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    victim = tmp_path / "victim.cpp"
    victim.write_text("preserve\n", encoding="utf-8")
    target = (
        workspace
        / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp"
    )
    target.symlink_to(victim)

    with pytest.raises(ValueError, match="symlink"):
        stage_importer_source(CPP, workspace)

    assert victim.read_text(encoding="utf-8") == "preserve\n"


def test_build_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "build_manifest.json"
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,"status":"PASS",'
        '"mode":"khronos_temporal_importer_build","build_scope":["khronos_eval"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_build_manifest(manifest, expected_source=CPP)


def test_build_manifest_rejects_symlinked_build_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.cpp"
    cmake = tmp_path / "CMakeLists.txt"
    real_executable = tmp_path / "real_importer"
    executable = tmp_path / "import_temporal_baseline"
    for path in (source, cmake, real_executable):
        path.write_text(path.name, encoding="utf-8")
    executable.symlink_to(real_executable.name)

    with pytest.raises(ValueError, match="symlink"):
        write_build_manifest(
            source,
            source,
            cmake,
            executable,
            tmp_path / "build_manifest.json",
        )


def test_parser_fixes_oviv2_causal_identity() -> None:
    args = parse_args(
        [
            "--manifest", "/bridge/manifest.json",
            "--scene", "apartment",
            "--output", "/run",
        ]
    )

    assert args.method == "OVIV2"
    assert args.mode == "causal_checkpoints"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--manifest", "/bridge/manifest.json",
                "--scene", "apartment",
                "--output", "/run",
                "--method", "DUALMAP",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--manifest", "/bridge/manifest.json",
                "--scene", "apartment",
                "--output", "/run",
                "--source", "/tmp/unreviewed.cpp",
            ]
        )


def test_snapshot_bridge_input_copies_then_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    artifact = source_root / "checkpoints/00000000/object.ply"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"points")
    manifest = source_root / "bridge_manifest.json"
    manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")
    validated: list[Path] = []

    def validate(path: Path) -> dict[str, object]:
        validated.append(path.resolve())
        return {"path": str(path)}

    monkeypatch.setattr(
        bridge_runner, "validate_temporal_bridge_manifest", validate
    )
    copied = snapshot_bridge_input(manifest, tmp_path / "run/bridge_input")

    assert copied == tmp_path / "run/bridge_input/bridge_manifest.json"
    assert copied.read_bytes() == manifest.read_bytes()
    assert (copied.parent / "checkpoints/00000000/object.ply").read_bytes() == b"points"
    assert validated == [manifest.resolve(), copied.resolve()]


def test_run_refuses_dangling_output_symlink_before_external_work(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim"
    output = tmp_path / "run"
    output.symlink_to(victim.name)
    args = parse_args(
        [
            "--manifest", str(tmp_path / "missing.json"),
            "--scene", "apartment",
            "--output", str(output),
        ]
    )

    with pytest.raises(ValueError, match="output already exists"):
        run(args)

    assert not victim.exists()
