#!/usr/bin/env python3
"""Build and run the hash-bound Khronos temporal baseline importer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    manifest: Path,
    output: Path,
) -> list[str]:
    executable = workspace / "install/khronos_eval/lib/khronos_eval/import_temporal_baseline"
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
        str(executable),
        str(manifest),
        str(output),
    ]


def write_build_manifest(
    source: Path, cmake: Path, executable: Path, destination: Path
) -> Path:
    if os.path.lexists(destination):
        raise FileExistsError(f"build manifest already exists: {destination}")
    for path in (source, cmake, executable):
        if not path.is_file():
            raise ValueError(f"temporal bridge build input is missing: {path}")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "khronos_temporal_importer_build",
        "source": _entry(source),
        "cmake": _entry(cmake),
        "executable": _entry(executable),
        "build_scope": ["khronos_eval"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def validate_build_manifest(
    path: Path, *, expected_source: Path = CPP_SOURCE
) -> dict[str, Any]:
    payload = loads_strict(
        path.read_text(encoding="utf-8"), label="temporal importer build manifest"
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
        or payload.get("mode") != "khronos_temporal_importer_build"
        or payload.get("build_scope") != ["khronos_eval"]
    ):
        raise ValueError("invalid temporal importer build manifest")
    source = _validate_entry(payload.get("source", {}), label="source")
    _validate_entry(payload.get("cmake", {}), label="cmake")
    _validate_entry(payload.get("executable", {}), label="executable")
    if source != expected_source.resolve() or _sha256(source) != _sha256(expected_source):
        raise ValueError("temporal importer source does not match expected source")
    return payload


def run(args: argparse.Namespace) -> Path:
    if os.path.lexists(args.output):
        raise ValueError(f"output already exists: {args.output}")
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
    stage_importer_source(args.source, args.workspace)
    build_log = args.output / "build.log"
    build_command = build_package_command(
        conda=args.conda, environment=args.environment, workspace=args.workspace
    )
    with build_log.open("w", encoding="utf-8") as output:
        built = subprocess.run(
            build_command,
            cwd=args.workspace,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.build_timeout_s,
            check=False,
        )
    executable = (
        args.workspace
        / "install/khronos_eval/lib/khronos_eval/import_temporal_baseline"
    )
    if built.returncode != 0 or not executable.is_file():
        raise RuntimeError(f"temporal importer build failed: {build_log}")
    cmake = args.workspace / "src/khronos/khronos_eval/CMakeLists.txt"
    build_manifest = write_build_manifest(
        args.source, cmake, executable, args.output / "build_manifest.json"
    )
    validate_build_manifest(build_manifest, expected_source=args.source)

    map_output = args.output / "map"
    command = build_bridge_command(
        conda=args.conda,
        environment=args.environment,
        workspace=args.workspace,
        manifest=args.manifest,
        output=map_output,
    )
    log_path = args.output / "bridge.log"
    timing_path = args.output / "bridge.time.log"
    started_at = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(timing_path), *command],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_s,
            check=False,
        )
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
    sources = [
        args.manifest,
        args.source,
        cmake,
        executable,
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
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_status": completed.returncode,
        "query_timestamps_ns": expected_timestamps,
        "build_command": build_command,
        "command": command,
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
    parser.add_argument("--scene", choices=("apartment", "office"), required=True)
    parser.add_argument("--method", choices=("OVIV2",), default="OVIV2")
    parser.add_argument(
        "--mode", choices=("causal_checkpoints",), default="causal_checkpoints"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=CPP_SOURCE
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/home/ww/oviovo_baseline_builds/khronos-jazzy-ws"),
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
