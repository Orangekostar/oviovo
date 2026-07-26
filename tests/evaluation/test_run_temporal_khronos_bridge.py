from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

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


def _write_elf(path: Path, payload: bytes = b"approved") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF" + payload)
    path.chmod(0o755)
    return path


def _install_symlink(workspace: Path, target: Path) -> Path:
    executable = (
        workspace / "install/khronos_eval/lib/khronos_eval/import_temporal_baseline"
    )
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.symlink_to(target)
    return executable


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
    assert "dynamic_state_at_query" in text
    assert "eligible != explicit_dynamic" in text
    assert "trajectory.size() < 2" not in text
    assert 'manifest.value("dataset", "") != "TESSE-CD"' in text
    assert 'manifest.value("method", "") != "OVIV2"' in text
    assert "resolveBridgePath" in text
    assert "map.update(buildGraph(checkpoint), timestamp_ns);" in text
    assert "map_timestamps.json" in text


def test_cpp_importer_independently_rejects_sample_event_epoch_conflicts() -> None:
    text = CPP.read_text(encoding="utf-8")

    assert 'manifest.at("temporal_consistency_json")' in text
    assert 'sample.at("geometry_epoch")' in text
    assert 'event.at("geometry_epoch")' in text
    assert "sample_state.geometry_epoch != event_epoch" in text
    assert "sample_state.readout_valid != event_readout_valid" in text
    assert "sample/event temporal state conflict" in text


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
    resolved_executable = _write_elf(
        workspace / "build/khronos_eval/import_temporal_baseline"
    )
    executable = _install_symlink(workspace, resolved_executable)
    staged = workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp"
    build_manifest = write_build_manifest(
        CPP,
        staged,
        workspace / "src/khronos/khronos_eval/CMakeLists.txt",
        executable,
        tmp_path / "build_manifest.json",
        trusted_workspace=workspace,
    )

    payload = validate_build_manifest(
        build_manifest, expected_source=CPP, expected_workspace=workspace
    )
    executable_fd = os.open(resolved_executable, os.O_RDONLY)
    try:
        command = build_bridge_command(
            conda=Path("/conda"),
            environment="khronos",
            workspace=workspace,
            executable_fd=executable_fd,
            manifest=Path("/bridge/manifest.json"),
            output=Path("/run/map"),
        )
    finally:
        os.close(executable_fd)

    assert payload["schema_version"] == 2
    assert payload["executable"]["declared_path"] == str(executable)
    assert payload["executable"]["symlink_chain"][0]["target"] == str(
        resolved_executable
    )
    assert payload["executable"]["resolved"]["sha256"] == hashlib.sha256(
        resolved_executable.read_bytes()
    ).hexdigest()
    assert payload["staged_source"]["sha256"] == payload["source"]["sha256"]
    assert command[-2:] == ["/bridge/manifest.json", "/run/map"]
    assert f"/proc/self/fd/{executable_fd}" in command
    assert str(resolved_executable) not in command
    assert str(executable) not in command

    resolved_executable.write_bytes(b"\x7fELFtampered")
    with pytest.raises(ValueError, match="resolved ELF"):
        validate_build_manifest(
            build_manifest, expected_source=CPP, expected_workspace=workspace
        )


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
        write_build_manifest(
            source,
            source,
            cmake,
            executable,
            destination,
            trusted_workspace=tmp_path,
        )

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


def test_build_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    outside = _write_elf(tmp_path / "outside/import_temporal_baseline")
    executable = _install_symlink(workspace, outside)

    with pytest.raises(ValueError, match="trusted workspace"):
        write_build_manifest(
            CPP,
            workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
            workspace / "src/khronos/khronos_eval/CMakeLists.txt",
            executable,
            tmp_path / "build_manifest.json",
            trusted_workspace=workspace,
        )


def test_build_manifest_rejects_symlink_loop(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    executable = (
        workspace / "install/khronos_eval/lib/khronos_eval/import_temporal_baseline"
    )
    other = executable.with_name("import_temporal_baseline.other")
    executable.parent.mkdir(parents=True)
    executable.symlink_to(other.name)
    other.symlink_to(executable.name)

    with pytest.raises(ValueError, match="loop"):
        write_build_manifest(
            CPP,
            workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
            workspace / "src/khronos/khronos_eval/CMakeLists.txt",
            executable,
            tmp_path / "build_manifest.json",
            trusted_workspace=workspace,
        )


def test_build_manifest_rejects_symlink_identity_swap(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    resolved = _write_elf(workspace / "build/khronos_eval/import_temporal_baseline")
    executable = _install_symlink(workspace, resolved)
    manifest = write_build_manifest(
        CPP,
        workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
        workspace / "src/khronos/khronos_eval/CMakeLists.txt",
        executable,
        tmp_path / "build_manifest.json",
        trusted_workspace=workspace,
    )
    executable.rename(executable.with_name("held-link"))
    executable.symlink_to(resolved)

    with pytest.raises(ValueError, match="symlink chain"):
        validate_build_manifest(
            manifest, expected_source=CPP, expected_workspace=workspace
        )


def test_build_manifest_rejects_symlink_target_swap(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    first = _write_elf(workspace / "build/khronos_eval/importer-first")
    second = _write_elf(workspace / "build/khronos_eval/importer-second")
    executable = _install_symlink(workspace, first)
    manifest = write_build_manifest(
        CPP,
        workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
        workspace / "src/khronos/khronos_eval/CMakeLists.txt",
        executable,
        tmp_path / "build_manifest.json",
        trusted_workspace=workspace,
    )
    executable.unlink()
    executable.symlink_to(second)

    with pytest.raises(ValueError, match="symlink chain"):
        validate_build_manifest(
            manifest, expected_source=CPP, expected_workspace=workspace
        )


def test_build_manifest_rejects_earlier_link_swap_during_chain_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    resolved = _write_elf(workspace / "build/khronos_eval/import_temporal_baseline")
    hop = workspace / "install/khronos_eval/lib/khronos_eval/importer-hop"
    hop.parent.mkdir(parents=True)
    hop.symlink_to(resolved)
    executable = _install_symlink(workspace, hop)
    held = executable.with_name("held-link")
    original_readlink = bridge_runner.os.readlink
    swapped = False

    def swap_first_link(path: Path) -> str:
        nonlocal swapped
        target = original_readlink(path)
        if Path(path) == hop and not swapped:
            executable.rename(held)
            executable.symlink_to(hop)
            swapped = True
        return target

    monkeypatch.setattr(bridge_runner.os, "readlink", swap_first_link)

    with pytest.raises(ValueError, match="symlink chain changed"):
        write_build_manifest(
            CPP,
            workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
            workspace / "src/khronos/khronos_eval/CMakeLists.txt",
            executable,
            tmp_path / "build_manifest.json",
            trusted_workspace=workspace,
        )


def test_build_manifest_rejects_non_elf_target(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    target = workspace / "build/khronos_eval/import_temporal_baseline"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not-elf")
    target.chmod(0o755)
    executable = _install_symlink(workspace, target)

    with pytest.raises(ValueError, match="ELF"):
        write_build_manifest(
            CPP,
            workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
            workspace / "src/khronos/khronos_eval/CMakeLists.txt",
            executable,
            tmp_path / "build_manifest.json",
            trusted_workspace=workspace,
        )


def test_build_manifest_rejects_noncanonical_install_entry(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    stage_importer_source(CPP, workspace)
    resolved = _write_elf(workspace / "build/khronos_eval/unreviewed_importer")
    alternate = workspace / "install/khronos_eval/lib/khronos_eval/unreviewed_importer"
    alternate.parent.mkdir(parents=True)
    alternate.symlink_to(resolved)

    with pytest.raises(ValueError, match="canonical install executable"):
        write_build_manifest(
            CPP,
            workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
            workspace / "src/khronos/khronos_eval/CMakeLists.txt",
            alternate,
            tmp_path / "rejected-at-write.json",
            trusted_workspace=workspace,
        )

    canonical_resolved = _write_elf(
        workspace / "build/khronos_eval/import_temporal_baseline"
    )
    canonical = _install_symlink(workspace, canonical_resolved)
    manifest = write_build_manifest(
        CPP,
        workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
        workspace / "src/khronos/khronos_eval/CMakeLists.txt",
        canonical,
        tmp_path / "build_manifest.json",
        trusted_workspace=workspace,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["executable"] = bridge_runner._executable_provenance(
        alternate, trusted_workspace=workspace
    )
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="canonical install executable"):
        validate_build_manifest(
            manifest, expected_source=CPP, expected_workspace=workspace
        )


def test_run_revalidates_build_before_and_after_using_resolved_elf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _fake_workspace(tmp_path / "ws")
    staged = workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp"
    staged.write_bytes(CPP.read_bytes())
    resolved = _write_elf(workspace / "build/khronos_eval/import_temporal_baseline")
    declared = _install_symlink(workspace, resolved)
    manifest = tmp_path / "bridge/bridge_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "oviv2_tesse_cd_apartment_v1.json"
    config.write_text('{"mode":"causal_checkpoints"}\n', encoding="utf-8")
    output = tmp_path / "run"
    bridge = {
        "dataset": "TESSE-CD",
        "method": "OVIV2",
        "scene_id": "apartment",
        "query_timestamps_ns": [100],
    }
    validation_calls: list[Path] = []
    commands: list[list[str]] = []
    importer_kwargs: dict[str, object] = {}
    expected_record = bridge_runner._resolved_elf_record(resolved)

    monkeypatch.setattr(bridge_runner, "stage_importer_source", lambda *_: {})
    monkeypatch.setattr(
        bridge_runner, "validate_temporal_bridge_manifest", lambda _: bridge
    )

    def snapshot(_: Path, destination: Path) -> Path:
        destination.mkdir(parents=True)
        copied = destination / "bridge_manifest.json"
        copied.write_text("{}\n", encoding="utf-8")
        return copied

    def write_manifest(*args: object, **_: object) -> Path:
        destination = Path(args[4])
        destination.write_text("{}\n", encoding="utf-8")
        return destination

    def validate_manifest(path: Path, **_: object) -> dict[str, object]:
        validation_calls.append(path)
        return {"executable": {"resolved": expected_record}}

    def execute(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        if len(commands) == 2:
            importer_kwargs.update(kwargs)
            inherited = kwargs.get("pass_fds")
            assert isinstance(inherited, tuple) and len(inherited) == 1
            descriptor = inherited[0]
            assert f"/proc/self/fd/{descriptor}" in command
            os.lseek(descriptor, 0, os.SEEK_SET)
            assert os.read(descriptor, 64) == b"\x7fELFapproved"
            map_root = output / "map"
            map_root.mkdir()
            (map_root / "final.4dmap").write_bytes(b"map")
            (map_root / "experiment_log.txt").write_text(
                "[FLAG] [Experiment Finished Cleanly] ", encoding="utf-8"
            )
            (map_root / "map_timestamps.json").write_text(
                "[100]\n", encoding="utf-8"
            )
            (output / "bridge.time.log").write_text("timing\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bridge_runner, "snapshot_bridge_input", snapshot)
    monkeypatch.setattr(bridge_runner, "write_build_manifest", write_manifest)
    monkeypatch.setattr(bridge_runner, "validate_build_manifest", validate_manifest)
    monkeypatch.setattr(bridge_runner.subprocess, "run", execute)
    args = parse_args(
        [
            "--manifest", str(manifest),
            "--scene", "apartment",
            "--output", str(output),
            "--workspace", str(workspace),
            "--config", str(config),
        ]
    )

    status_path = run(args)
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert validation_calls == [
        output / "build_manifest.json",
        output / "build_manifest.json",
        output / "build_manifest.json",
    ]
    assert len(importer_kwargs["pass_fds"]) == 1
    assert any(item.startswith("/proc/self/fd/") for item in status["command"])
    assert str(resolved) not in status["command"]
    assert str(declared) not in status["command"]
    assert status["run_identity"] == {
        "run_id": "oviv2-tessecd-v1",
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    assert status["config"] in status["sources"]
    assert Path(status["config"]["path"]) == config.resolve()


def test_verified_descriptor_cannot_be_redirected_by_path_swap(tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path / "workspace")
    stage_importer_source(CPP, workspace)
    resolved = _write_elf(workspace / "build/khronos_eval/import_temporal_baseline")
    declared = _install_symlink(workspace, resolved)
    manifest = write_build_manifest(
        CPP,
        workspace / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp",
        workspace / "src/khronos/khronos_eval/CMakeLists.txt",
        declared,
        tmp_path / "build_manifest.json",
        trusted_workspace=workspace,
    )
    payload = validate_build_manifest(
        manifest, expected_source=CPP, expected_workspace=workspace
    )
    expected = payload["executable"]["resolved"]
    descriptor = bridge_runner._open_verified_executable(resolved, expected)
    held = resolved.with_name("held-approved-importer")
    try:
        resolved.rename(held)
        _write_elf(resolved, payload=b"tampered")
        os.lseek(descriptor, 0, os.SEEK_SET)
        assert os.read(descriptor, 64) == b"\x7fELFapproved"
        with pytest.raises(ValueError, match="provenance mismatch"):
            validate_build_manifest(
                manifest, expected_source=CPP, expected_workspace=workspace
            )
    finally:
        os.close(descriptor)


def test_run_rejects_symlinked_workspace_before_write_or_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_workspace = _fake_workspace(tmp_path / "real-workspace")
    linked_workspace = tmp_path / "linked-workspace"
    linked_workspace.symlink_to(real_workspace.name)
    manifest = tmp_path / "bridge_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(
        bridge_runner,
        "validate_temporal_bridge_manifest",
        lambda _: {
            "dataset": "TESSE-CD",
            "method": "OVIV2",
            "scene_id": "apartment",
            "query_timestamps_ns": [100],
        },
    )
    monkeypatch.setattr(
        bridge_runner,
        "stage_importer_source",
        lambda *_: pytest.fail("workspace was written before trust validation"),
    )
    monkeypatch.setattr(
        bridge_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "external command ran before workspace trust validation"
        ),
    )
    args = parse_args(
        [
            "--manifest", str(manifest),
            "--scene", "apartment",
            "--output", str(output),
            "--workspace", str(linked_workspace),
            "--config", str(tmp_path / "missing-config.json"),
        ]
    )

    with pytest.raises(ValueError, match="trusted workspace"):
        run(args)

    assert not output.exists()


def test_parser_fixes_oviv2_causal_identity() -> None:
    args = parse_args(
        [
            "--manifest", "/bridge/manifest.json",
            "--scene", "apartment",
            "--output", "/run",
            "--config", "/configs/oviv2_tesse_cd_apartment_v1.json",
        ]
    )

    assert args.method == "OVIV2"
    assert args.mode == "causal_checkpoints"
    assert args.run_id == "oviv2-tessecd-v1"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--manifest", "/bridge/manifest.json",
                "--scene", "apartment",
                "--output", "/run",
                "--config", "/configs/oviv2_tesse_cd_apartment_v1.json",
                "--method", "DUALMAP",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--manifest", "/bridge/manifest.json",
                "--scene", "apartment",
                "--output", "/run",
                "--config", "/configs/oviv2_tesse_cd_apartment_v1.json",
                "--source", "/tmp/unreviewed.cpp",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--manifest", "/bridge/manifest.json",
                "--scene", "apartment",
                "--output", "/run",
            ]
        )


def test_run_rejects_noncanonical_run_id_before_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _fake_workspace(tmp_path / "workspace")
    manifest = tmp_path / "bridge_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(
        bridge_runner,
        "validate_temporal_bridge_manifest",
        lambda _: pytest.fail("manifest validation followed an invalid run id"),
    )
    args = parse_args(
        [
            "--manifest", str(manifest),
            "--scene", "apartment",
            "--output", str(output),
            "--workspace", str(workspace),
            "--config", str(config),
            "--run-id", "tampered/run",
        ]
    )

    with pytest.raises(ValueError, match="run_id.*canonical"):
        run(args)

    assert not output.exists()


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
            "--config", str(tmp_path / "missing-config.json"),
        ]
    )

    with pytest.raises(ValueError, match="output already exists"):
        run(args)

    assert not victim.exists()
