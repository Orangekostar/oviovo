#!/usr/bin/env python3
"""Build and run the hash-bound Khronos temporal baseline importer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.prepare_temporal_khronos_bridge import (
    validate_temporal_bridge_manifest,
)
from src.evaluation.json_contracts import loads_strict


CPP_SOURCE = (
    REPO_ROOT
    / "scripts/evaluation/compat/khronos_temporal_bridge/import_temporal_baseline.cpp"
)
DEFAULT_WORKSPACE = Path("/home/ww/oviovo_baseline_builds/khronos-jazzy-ws")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CMAKE_BEGIN = "# OVIV2_KHRONOS_TEMPORAL_BRIDGE_BEGIN"
CMAKE_END = "# OVIV2_KHRONOS_TEMPORAL_BRIDGE_END"
CMAKE_BLOCK = f"""{CMAKE_BEGIN}
add_executable(import_temporal_baseline app/import_temporal_baseline.cpp)
target_link_libraries(import_temporal_baseline ${{PROJECT_NAME}} ${{gflags_LIBRARIES}})
install(TARGETS import_temporal_baseline RUNTIME DESTINATION lib/${{PROJECT_NAME}})
{CMAKE_END}

"""
LEGACY_CMAKE_BEGIN = "# OVIOVO_KHRONOS_TEMPORAL_BRIDGE_BEGIN"
LEGACY_CMAKE_END = "# OVIOVO_KHRONOS_TEMPORAL_BRIDGE_END"
LEGACY_CMAKE_BLOCK = CMAKE_BLOCK.replace(
    CMAKE_BEGIN, LEGACY_CMAKE_BEGIN
).replace(CMAKE_END, LEGACY_CMAKE_END)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"build artifact must not be a symlink: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _validate_entry(entry: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(entry.get("path", ""))).resolve()
    raw_path = Path(str(entry.get("path", "")))
    if raw_path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if path.stat().st_size != int(entry.get("byte_count", -1)):
        raise ValueError(f"{label} byte count mismatch")
    if _sha256(path) != str(entry.get("sha256", "")):
        raise ValueError(f"{label} SHA256 mismatch")
    return path


def _run_identity(run_id: str, config: Mapping[str, Any]) -> dict[str, str]:
    if type(run_id) is not str or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("temporal bridge run identity run_id is not canonical")
    config_sha256 = config.get("sha256")
    if (
        type(config_sha256) is not str
        or len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise ValueError("temporal bridge run identity config SHA256 is invalid")
    return {"run_id": run_id, "config_sha256": config_sha256}


def _identity(status: os.stat_result) -> dict[str, int]:
    return {
        "device": status.st_dev,
        "inode": status.st_ino,
        "mode": status.st_mode,
        "uid": status.st_uid,
        "gid": status.st_gid,
        "size": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
    }


def _trusted_workspace_root(workspace: Path) -> Path:
    declared = Path(os.path.abspath(workspace))
    try:
        resolved = declared.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"trusted workspace is invalid: {declared}") from error
    if declared != resolved or not resolved.is_dir():
        raise ValueError("trusted workspace must be a real directory without symlinks")
    return resolved


def _assert_trusted_parents(path: Path, workspace: Path) -> None:
    try:
        relative = path.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"executable symlink escapes trusted workspace: {path}") from error
    current = workspace
    for part in relative.parts[:-1]:
        current /= part
        try:
            status = current.lstat()
        except OSError as error:
            raise ValueError(f"executable parent is invalid: {current}") from error
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"executable parent symlink is not trusted: {current}")
        if not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"executable parent is not a directory: {current}")


def _elf_record_from_descriptor(descriptor: int, path: Path) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"resolved ELF is not a regular file: {path}")
    if not before.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        raise ValueError(f"resolved ELF is not executable: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb", closefd=False) as handle:
        magic = handle.read(4)
        if magic != b"\x7fELF":
            raise ValueError(f"resolved executable is not an ELF file: {path}")
        digest.update(magic)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after):
        raise ValueError(f"resolved ELF changed while hashing: {path}")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "byte_count": after.st_size,
        "identity": _identity(after),
    }


def _open_elf(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        return os.open(path, flags)
    except OSError as error:
        raise ValueError(f"resolved ELF is not a readable regular file: {path}") from error


def _resolved_elf_record(path: Path) -> dict[str, Any]:
    descriptor = _open_elf(path)
    try:
        record = _elf_record_from_descriptor(descriptor, path)
    finally:
        os.close(descriptor)
    try:
        path_status = path.lstat()
    except OSError as error:
        raise ValueError(f"resolved ELF changed while hashing: {path}") from error
    if record["identity"] != _identity(path_status):
        raise ValueError(f"resolved ELF changed while hashing: {path}")
    return record


def _open_verified_executable(path: Path, expected: Mapping[str, Any]) -> int:
    descriptor = _open_elf(path)
    try:
        if _elf_record_from_descriptor(descriptor, path) != dict(expected):
            raise ValueError("resolved ELF does not match verified build provenance")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_open_executable(
    descriptor: int, path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    record = _elf_record_from_descriptor(descriptor, path)
    if record != dict(expected):
        raise ValueError("executed ELF changed while the verified descriptor was open")
    return record


def _revalidate_symlink_chain(chain: list[dict[str, Any]]) -> None:
    for expected in chain:
        path = Path(expected["path"])
        try:
            before = path.lstat()
            target = os.readlink(path)
            after = path.lstat()
        except OSError as error:
            raise ValueError(f"executable symlink chain changed: {path}") from error
        if (
            _identity(before) != expected["identity"]
            or _identity(after) != expected["identity"]
            or target != expected["target"]
        ):
            raise ValueError(f"executable symlink chain changed: {path}")


def _executable_provenance(
    declared_path: Path, *, trusted_workspace: Path
) -> dict[str, Any]:
    workspace = _trusted_workspace_root(trusted_workspace)
    declared = Path(os.path.abspath(declared_path))
    current = declared
    seen: set[Path] = set()
    chain: list[dict[str, Any]] = []
    for _ in range(32):
        _assert_trusted_parents(current, workspace)
        if current in seen:
            raise ValueError("executable symlink chain contains a loop")
        seen.add(current)
        try:
            before = current.lstat()
        except OSError as error:
            raise ValueError(f"executable symlink target is missing: {current}") from error
        if not stat.S_ISLNK(before.st_mode):
            if not chain:
                raise ValueError("declared install executable must be a symlink")
            resolved = _resolved_elf_record(current)
            _revalidate_symlink_chain(chain)
            return {
                "declared_path": str(declared),
                "symlink_chain": chain,
                "resolved": resolved,
            }
        target = os.readlink(current)
        after = current.lstat()
        if _identity(before) != _identity(after):
            raise ValueError(f"executable symlink changed while reading: {current}")
        chain.append(
            {
                "path": str(current),
                "target": target,
                "identity": _identity(after),
            }
        )
        raw_target = Path(target)
        current = Path(
            os.path.abspath(
                raw_target
                if raw_target.is_absolute()
                else current.parent / raw_target
            )
        )
    raise ValueError("executable symlink chain exceeds 32 links")


def stage_importer_source(source: Path, workspace: Path) -> dict[str, Any]:
    if not source.is_file():
        raise ValueError(f"temporal importer source is missing: {source}")
    package = workspace / "src/khronos/khronos_eval"
    cmake_path = package / "CMakeLists.txt"
    app = package / "app"
    if not cmake_path.is_file() or not app.is_dir():
        raise ValueError("workspace has no Khronos khronos_eval source package")
    target = app / "import_temporal_baseline.cpp"
    if target.is_symlink():
        raise ValueError("temporal importer target must not be a symlink")
    shutil.copyfile(source, target)

    before = cmake_path.read_text(encoding="utf-8")
    if CMAKE_BEGIN in before:
        if before.count(CMAKE_BEGIN) != 1 or CMAKE_BLOCK not in before:
            raise ValueError("existing temporal importer CMake block disagrees")
        after = before
    elif LEGACY_CMAKE_BEGIN in before:
        if (
            before.count(LEGACY_CMAKE_BEGIN) != 1
            or before.count(LEGACY_CMAKE_END) != 1
            or LEGACY_CMAKE_BLOCK not in before
        ):
            raise ValueError("existing legacy temporal importer CMake block disagrees")
        after = before.replace(LEGACY_CMAKE_BLOCK, CMAKE_BLOCK)
        cmake_path.write_text(after, encoding="utf-8")
    else:
        marker = (
            "# ##############################################################################\n"
            "# Export #\n"
            "# ##############################################################################\n"
        )
        if before.count(marker) != 1:
            raise ValueError("khronos_eval CMake export marker mismatch")
        after = before.replace(marker, CMAKE_BLOCK + marker)
        cmake_path.write_text(after, encoding="utf-8")
    return {
        "source": _entry(source),
        "staged_source": _entry(target),
        "cmake_sha256_before": hashlib.sha256(before.encode("utf-8")).hexdigest(),
        "cmake": _entry(cmake_path),
    }


def build_package_command(
    *, conda: Path, environment: str, workspace: Path
) -> list[str]:
    return [
        str(conda),
        "run",
        "--no-capture-output",
        "-n",
        environment,
        "bash",
        "-lc",
        'source "$1/install/setup.bash"; shift; exec "$@"',
        "bash",
        str(workspace),
        "colcon",
        "build",
        "--packages-select",
        "khronos_eval",
        "--symlink-install",
        "--cmake-args",
        "-DCMAKE_BUILD_TYPE=Release",
    ]


def build_bridge_command(
    *,
    conda: Path,
    environment: str,
    workspace: Path,
    executable_fd: int,
    manifest: Path,
    output: Path,
) -> list[str]:
    if type(executable_fd) is not int or executable_fd < 0:
        raise ValueError("bridge command requires a verified executable descriptor")
    try:
        descriptor_status = os.fstat(executable_fd)
    except OSError as error:
        raise ValueError(
            "bridge command requires an open verified executable descriptor"
        ) from error
    if not stat.S_ISREG(descriptor_status.st_mode):
        raise ValueError("bridge command executable descriptor is not regular")
    conda_profile = conda.parent.parent / "etc/profile.d/conda.sh"
    return [
        "/bin/bash",
        "-lc",
        (
            'source "$1"; conda activate "$2"; '
            'source "$3/install/setup.bash"; shift 3; exec "$@"'
        ),
        "bash",
        str(conda_profile),
        environment,
        str(workspace),
        f"/proc/self/fd/{executable_fd}",
        str(manifest),
        str(output),
    ]


def _canonical_install_executable(workspace: Path) -> Path:
    return workspace / "install/khronos_eval/lib/khronos_eval/import_temporal_baseline"


def write_build_manifest(
    source: Path,
    staged_source: Path,
    cmake: Path,
    executable: Path,
    destination: Path,
    *,
    trusted_workspace: Path,
) -> Path:
    if os.path.lexists(destination):
        raise FileExistsError(f"build manifest already exists: {destination}")
    for path in (source, staged_source, cmake):
        if not path.is_file():
            raise ValueError(f"temporal bridge build input is missing: {path}")
    workspace = _trusted_workspace_root(trusted_workspace)
    declared_executable = Path(os.path.abspath(executable))
    if declared_executable != _canonical_install_executable(workspace):
        raise ValueError("build manifest requires the canonical install executable")
    payload = {
        "schema_version": 2,
        "status": "PASS",
        "mode": "khronos_temporal_importer_build",
        "source": _entry(source),
        "staged_source": _entry(staged_source),
        "cmake": _entry(cmake),
        "trusted_workspace": str(workspace),
        "executable": _executable_provenance(
            declared_executable, trusted_workspace=workspace
        ),
        "build_scope": ["khronos_eval"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def validate_build_manifest(
    path: Path,
    *,
    expected_source: Path = CPP_SOURCE,
    expected_workspace: Path = DEFAULT_WORKSPACE,
) -> dict[str, Any]:
    payload = loads_strict(
        path.read_text(encoding="utf-8"), label="temporal importer build manifest"
    )
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "PASS"
        or payload.get("mode") != "khronos_temporal_importer_build"
        or payload.get("build_scope") != ["khronos_eval"]
    ):
        raise ValueError("invalid temporal importer build manifest")
    workspace = _trusted_workspace_root(expected_workspace)
    if payload.get("trusted_workspace") != str(workspace):
        raise ValueError("temporal importer trusted workspace mismatch")
    source = _validate_entry(payload.get("source", {}), label="source")
    staged_source = _validate_entry(
        payload.get("staged_source", {}), label="staged source"
    )
    cmake = _validate_entry(payload.get("cmake", {}), label="cmake")
    executable = payload.get("executable")
    if not isinstance(executable, Mapping):
        raise ValueError("temporal importer executable provenance is invalid")
    declared_path = Path(str(executable.get("declared_path", "")))
    if declared_path != _canonical_install_executable(workspace):
        raise ValueError("build manifest requires the canonical install executable")
    observed_executable = _executable_provenance(
        declared_path, trusted_workspace=workspace
    )
    if dict(executable) != observed_executable:
        raise ValueError("executable symlink chain or resolved ELF provenance mismatch")
    if source != expected_source.resolve() or _sha256(source) != _sha256(expected_source):
        raise ValueError("temporal importer source does not match expected source")
    if staged_source.read_bytes() != source.read_bytes():
        raise ValueError("staged temporal importer source does not match reviewed source")
    expected_cmake = workspace / "src/khronos/khronos_eval/CMakeLists.txt"
    if cmake != expected_cmake:
        raise ValueError("temporal importer CMake path is invalid")
    expected_staged_source = (
        workspace
        / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp"
    )
    if staged_source != expected_staged_source.resolve():
        raise ValueError("staged temporal importer source path is invalid")
    return payload


def _bridge_tree_records(root: Path) -> list[tuple[str, str, int]]:
    records: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"bridge input snapshot rejects symlink: {path}")
        if path.is_file():
            records.append(
                (str(path.relative_to(root)), _sha256(path), path.stat().st_size)
            )
        elif not path.is_dir():
            raise ValueError(f"bridge input snapshot rejects special file: {path}")
    return records


def snapshot_bridge_input(manifest: Path, destination: Path) -> Path:
    if manifest.name != "bridge_manifest.json":
        raise ValueError("temporal bridge manifest must be named bridge_manifest.json")
    validate_temporal_bridge_manifest(manifest)
    source_root = manifest.parent
    if os.path.lexists(destination):
        raise FileExistsError(f"bridge input snapshot already exists: {destination}")
    before = _bridge_tree_records(source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, destination)
    copied_manifest = destination / manifest.name
    if _bridge_tree_records(source_root) != before:
        raise ValueError("bridge input changed while snapshotting")
    if _bridge_tree_records(destination) != before:
        raise ValueError("bridge input snapshot byte mismatch")
    validate_temporal_bridge_manifest(copied_manifest)
    return copied_manifest


def run(args: argparse.Namespace) -> Path:
    if os.path.lexists(args.output):
        raise ValueError(f"output already exists: {args.output}")
    workspace = _trusted_workspace_root(args.workspace)
    config_record = _entry(args.config)
    run_identity = _run_identity(args.run_id, config_record)
    bridge = validate_temporal_bridge_manifest(args.manifest)
    if (
        bridge.get("dataset") != "TESSE-CD"
        or bridge.get("method") != "OVIV2"
        or args.method != "OVIV2"
        or args.mode != "causal_checkpoints"
    ):
        raise ValueError("temporal bridge requires OVIV2 causal identity")
    if bridge.get("scene_id") != args.scene:
        raise ValueError("temporal bridge manifest scene mismatch")
    args.output.mkdir(parents=True)
    stage_importer_source(CPP_SOURCE, workspace)
    build_log = args.output / "build.log"
    build_command = build_package_command(
        conda=args.conda, environment=args.environment, workspace=workspace
    )
    with build_log.open("w", encoding="utf-8") as output:
        built = subprocess.run(
            build_command,
            cwd=workspace,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.build_timeout_s,
            check=False,
        )
    executable = _canonical_install_executable(workspace)
    if built.returncode != 0 or not executable.is_file():
        raise RuntimeError(f"temporal importer build failed: {build_log}")
    cmake = workspace / "src/khronos/khronos_eval/CMakeLists.txt"
    staged_source = (
        workspace
        / "src/khronos/khronos_eval/app/import_temporal_baseline.cpp"
    )
    build_manifest = write_build_manifest(
        CPP_SOURCE,
        staged_source,
        cmake,
        executable,
        args.output / "build_manifest.json",
        trusted_workspace=workspace,
    )
    build_payload = validate_build_manifest(
        build_manifest,
        expected_source=CPP_SOURCE,
        expected_workspace=workspace,
    )
    resolved_executable = Path(build_payload["executable"]["resolved"]["path"])

    copied_manifest = snapshot_bridge_input(
        args.manifest, args.output / "bridge_input"
    )
    bridge = validate_temporal_bridge_manifest(copied_manifest)
    if _entry(args.config) != config_record:
        raise ValueError("frozen OVIV2 run config changed before importer execution")

    pre_execute_build = validate_build_manifest(
        build_manifest,
        expected_source=CPP_SOURCE,
        expected_workspace=workspace,
    )
    if pre_execute_build["executable"]["resolved"] != build_payload["executable"][
        "resolved"
    ]:
        raise ValueError("resolved temporal importer changed before execution")

    map_output = args.output / "map"
    log_path = args.output / "bridge.log"
    timing_path = args.output / "bridge.time.log"
    started_at = datetime.now(timezone.utc).isoformat()
    executable_record = build_payload["executable"]["resolved"]
    executable_fd = _open_verified_executable(
        resolved_executable, executable_record
    )
    try:
        command = build_bridge_command(
            conda=args.conda,
            environment=args.environment,
            workspace=workspace,
            executable_fd=executable_fd,
            manifest=copied_manifest,
            output=map_output,
        )
        with log_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                ["/usr/bin/time", "-v", "-o", str(timing_path), *command],
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_s,
                check=False,
                pass_fds=(executable_fd,),
            )
        executed_executable = _revalidate_open_executable(
            executable_fd, resolved_executable, executable_record
        )
    finally:
        os.close(executable_fd)
    final_map = map_output / "final.4dmap"
    experiment_log = map_output / "experiment_log.txt"
    timestamps_path = map_output / "map_timestamps.json"
    expected_timestamps = bridge["query_timestamps_ns"]
    passed = (
        completed.returncode == 0
        and final_map.is_file()
        and experiment_log.is_file()
        and "[FLAG] [Experiment Finished Cleanly] "
        in experiment_log.read_text(encoding="utf-8")
        and timestamps_path.is_file()
        and json.loads(timestamps_path.read_text(encoding="utf-8"))
        == expected_timestamps
    )
    validate_temporal_bridge_manifest(copied_manifest)
    post_execute_build = validate_build_manifest(
        build_manifest,
        expected_source=CPP_SOURCE,
        expected_workspace=workspace,
    )
    if post_execute_build["executable"]["resolved"] != executable_record:
        raise ValueError("resolved temporal importer changed during execution")
    if _entry(args.config) != config_record:
        raise ValueError("frozen OVIV2 run config changed before status publication")
    sources = [
        args.config,
        copied_manifest,
        CPP_SOURCE,
        staged_source,
        cmake,
        resolved_executable,
        build_manifest,
        build_log,
        log_path,
        timing_path,
    ]
    for path in (final_map, experiment_log, timestamps_path):
        if path.is_file():
            sources.append(path)
    status = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "dataset": "TESSE-CD",
        "scene": args.scene,
        "method": args.method,
        "mode": args.mode,
        "bridge_mode": "temporal_checkpoints",
        "display_mode": "online",
        "run_identity": run_identity,
        "config": config_record,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_status": completed.returncode,
        "query_timestamps_ns": expected_timestamps,
        "build_command": build_command,
        "command": command,
        "executed_executable": executed_executable,
        "sources": [_entry(path) for path in sources],
    }
    status_path = args.output / "run_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"temporal Khronos bridge failed: {status_path}")
    return status_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default="oviv2-tessecd-v1")
    parser.add_argument("--scene", choices=("apartment", "office"), required=True)
    parser.add_argument("--method", choices=("OVIV2",), default="OVIV2")
    parser.add_argument(
        "--mode", choices=("causal_checkpoints",), default="causal_checkpoints"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
    )
    parser.add_argument(
        "--conda", type=Path, default=Path("/home/ww/miniconda3/bin/conda")
    )
    parser.add_argument("--environment", default="oviovo-khronos-jazzy")
    parser.add_argument("--build-timeout-s", type=float, default=1800.0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
