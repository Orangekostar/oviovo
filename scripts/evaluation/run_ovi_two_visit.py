#!/usr/bin/env python3
"""Plan and execute two independent OVI-MAP visits for a frozen protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.build_tesse_ovimap_static_anchor import (
    TesseNativeCommands,
    TesseNativeEnvironment,
    build_tesse_native_commands,
    run_tesse_native_mapping,
)
from scripts.evaluation.freeze_tesse_two_visit_protocol import (
    ProtocolError,
    authorize_scene_run,
    compute_window_input_sha256,
    protocol_content_sha256,
)


@dataclass(frozen=True, slots=True)
class OviVisitPlan:
    visit_id: str
    run_id: str
    source_start_frame: int
    source_end_frame: int
    source_input_sha256: str
    materialized_rgbd_root: Path
    commands: TesseNativeCommands


@dataclass(frozen=True, slots=True)
class TwoVisitOviPlan:
    protocol_content_sha256: str
    scene: str
    t0: OviVisitPlan
    t1: OviVisitPlan


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _file_record(path: Path) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = absolute.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProtocolError("file binding must be a regular non-symlink file")
    data = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(data) != after.st_size
    ):
        raise ProtocolError("file binding changed while reading")
    return {
        "path": str(absolute),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _relative_record(path: Path, *, root: Path) -> dict[str, object]:
    record = _file_record(path)
    record["path"] = path.relative_to(root).as_posix()
    return record


def _atomic_json(path: Path, value: object) -> None:
    data = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_visit_window(
    *,
    source_rgbd_root: Path,
    materialized_rgbd_root: Path,
    scene: str,
    visit_id: str,
    start_frame: int,
    end_frame: int,
    expected_input_sha256: str,
) -> Path:
    """Copy and reindex one immutable source window into Replica layout."""

    if visit_id not in {"t0", "t1"}:
        raise ProtocolError("visit_id must be t0 or t1")
    if start_frame < 0 or end_frame < start_frame:
        raise ProtocolError("visit frame interval is invalid")
    observed_input_hash = compute_window_input_sha256(
        source_rgbd_root,
        scene=scene,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if observed_input_hash != expected_input_sha256:
        raise ProtocolError("source window input hash mismatch")
    destination = Path(materialized_rgbd_root)
    if destination.exists():
        raise ProtocolError(f"materialized visit root already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    source_scene = Path(source_rgbd_root) / scene
    try:
        staged_scene = staging / scene
        staged_results = staged_scene / "results"
        staged_results.mkdir(parents=True)
        shutil.copy2(Path(source_rgbd_root) / "cam_params.json", staging / "cam_params.json")
        source_trajectory = (source_scene / "traj.txt").read_text(encoding="utf-8").splitlines()
        with (source_scene / "timestamps.csv").open("r", encoding="utf-8", newline="") as handle:
            source_timestamps = tuple(csv.DictReader(handle))
        selected_trajectory = []
        selected_timestamps = []
        first_relative = int(source_timestamps[start_frame]["relative_timestamp_ns"])
        source_bindings: dict[str, dict[str, object]] = {}
        for target_index, source_index in enumerate(range(start_frame, end_frame + 1)):
            source_rgb = source_scene / "results" / f"frame{source_index:06d}.jpg"
            source_depth = source_scene / "results" / f"depth{source_index:06d}.png"
            target_rgb = staged_results / f"frame{target_index:06d}.jpg"
            target_depth = staged_results / f"depth{target_index:06d}.png"
            shutil.copy2(source_rgb, target_rgb)
            shutil.copy2(source_depth, target_depth)
            selected_trajectory.append(source_trajectory[source_index] + "\n")
            row = source_timestamps[source_index]
            selected_timestamps.append(
                {
                    "frame_index": target_index,
                    "sensor_timestamp_ns": int(row["sensor_timestamp_ns"]),
                    "relative_timestamp_ns": int(row["relative_timestamp_ns"])
                    - first_relative,
                }
            )
            source_bindings[f"rgb/{target_index:06d}"] = _file_record(source_rgb)
            source_bindings[f"depth/{target_index:06d}"] = _file_record(source_depth)
        (staged_scene / "traj.txt").write_text(
            "".join(selected_trajectory), encoding="utf-8"
        )
        with (staged_scene / "timestamps.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "frame_index",
                    "sensor_timestamp_ns",
                    "relative_timestamp_ns",
                ),
            )
            writer.writeheader()
            writer.writerows(selected_timestamps)
        manifest = {
            "schema_version": 1,
            "status": "MATERIALIZED_INPUT_PASS",
            "dataset": "TESSE-CD",
            "scene": scene,
            "visit_id": visit_id,
            "frame_count": end_frame - start_frame + 1,
            "source_frame_interval": [start_frame, end_frame],
            "source_input_sha256": expected_input_sha256,
            "source_bindings": source_bindings,
            "outputs": {
                "camera": _relative_record(staging / "cam_params.json", root=staging),
                "trajectory": _relative_record(staged_scene / "traj.txt", root=staging),
                "timestamps": _relative_record(staged_scene / "timestamps.csv", root=staging),
            },
        }
        manifest_path = staged_scene / "export_manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        if compute_window_input_sha256(
            source_rgbd_root,
            scene=scene,
            start_frame=start_frame,
            end_frame=end_frame,
        ) != expected_input_sha256:
            raise ProtocolError("source window changed during materialization")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / scene / "export_manifest.json"


def _visit_plan(
    *,
    protocol_hash: str,
    scene: str,
    visit_id: str,
    visit: Mapping[str, Any],
    run_root: Path,
    native: TesseNativeEnvironment,
) -> OviVisitPlan:
    count = int(visit["frame_count"])
    run_id = f"{protocol_hash[:12]}-{scene}-{visit_id}"
    materialized = run_root / "materialized-inputs" / scene / visit_id
    attempt = run_root / "native-runs" / scene / visit_id / run_id
    commands = build_tesse_native_commands(
        scene=scene,
        rgbd_root=materialized,
        frame_ids=tuple(range(count)),
        native=native,
        attempt_root=attempt,
    )
    return OviVisitPlan(
        visit_id=visit_id,
        run_id=run_id,
        source_start_frame=int(visit["start_frame"]),
        source_end_frame=int(visit["end_frame"]),
        source_input_sha256=str(visit["input_sha256"]),
        materialized_rgbd_root=materialized,
        commands=commands,
    )


def build_independent_ovi_plan(
    protocol: Mapping[str, Any],
    *,
    scene: str,
    source_rgbd_root: Path,
    run_root: Path,
    native: TesseNativeEnvironment,
    office_release: Mapping[str, Any] | None = None,
) -> TwoVisitOviPlan:
    """Build separate zero-based OVI commands without sharing visit state."""

    method_input = authorize_scene_run(
        protocol, scene, office_release=office_release
    )
    visits = method_input.get("visits")
    if not isinstance(visits, Mapping) or set(visits) != {"t0", "t1"}:
        raise ProtocolError("method input visits are invalid")
    digest = protocol_content_sha256(protocol)
    t0 = _visit_plan(
        protocol_hash=digest,
        scene=scene,
        visit_id="t0",
        visit=visits["t0"],
        run_root=run_root,
        native=native,
    )
    t1 = _visit_plan(
        protocol_hash=digest,
        scene=scene,
        visit_id="t1",
        visit=visits["t1"],
        run_root=run_root,
        native=native,
    )
    if t0.run_id == t1.run_id or t0.commands.attempt_root == t1.commands.attempt_root:
        raise ProtocolError("OVI visits must have independent run identities and roots")
    t0_markers = (str(t0.commands.attempt_root), "/t0/")
    t1_arguments = (
        *t1.commands.geometry,
        *t1.commands.frontend,
        *t1.commands.mapping,
    )
    if any(marker in argument for marker in t0_markers for argument in t1_arguments):
        raise ProtocolError("t1 OVI command leaks t0 state path")
    return TwoVisitOviPlan(
        protocol_content_sha256=digest,
        scene=scene,
        t0=t0,
        t1=t1,
    )


def execute_independent_ovi_plan(
    plan: TwoVisitOviPlan,
    *,
    protocol_path: Path,
    source_rgbd_root: Path,
    native: TesseNativeEnvironment,
    output_manifest: Path,
    native_executor: Callable[..., Path] = run_tesse_native_mapping,
    office_release: Mapping[str, Any] | None = None,
) -> Path:
    """Materialize and run exactly two source-bound native OVI mappings."""

    protocol = _load_json(protocol_path)
    if protocol_content_sha256(protocol) != plan.protocol_content_sha256:
        raise ProtocolError("OVI plan protocol hash mismatch")
    output = Path(output_manifest)
    if output.exists():
        raise ProtocolError(f"two-visit OVI manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    method_input_path = output.parent / "method_input_manifest.json"
    if method_input_path.exists():
        raise ProtocolError(f"method input manifest already exists: {method_input_path}")
    method_input = authorize_scene_run(
        protocol, plan.scene, office_release=office_release
    )
    declared_visits = method_input.get("visits")
    if not isinstance(declared_visits, Mapping):
        raise ProtocolError("method input visits are invalid")
    for visit in (plan.t0, plan.t1):
        declared = declared_visits.get(visit.visit_id)
        if (
            not isinstance(declared, Mapping)
            or declared.get("start_frame") != visit.source_start_frame
            or declared.get("end_frame") != visit.source_end_frame
            or declared.get("input_sha256") != visit.source_input_sha256
        ):
            raise ProtocolError("OVI plan visit differs from frozen method input")
    _atomic_json(method_input_path, method_input)
    visit_records: dict[str, dict[str, object]] = {}
    for visit in (plan.t0, plan.t1):
        materialized_manifest = materialize_visit_window(
            source_rgbd_root=source_rgbd_root,
            materialized_rgbd_root=visit.materialized_rgbd_root,
            scene=plan.scene,
            visit_id=visit.visit_id,
            start_frame=visit.source_start_frame,
            end_frame=visit.source_end_frame,
            expected_input_sha256=visit.source_input_sha256,
        )
        native_manifest = native_executor(
            scene=plan.scene,
            rgbd_root=visit.materialized_rgbd_root,
            commands=visit.commands,
            native=native,
        )
        visit_records[visit.visit_id] = {
            "run_id": visit.run_id,
            "source_frame_interval": [
                visit.source_start_frame,
                visit.source_end_frame,
            ],
            "source_input_sha256": visit.source_input_sha256,
            "materialized_manifest": _file_record(materialized_manifest),
            "native_manifest": _file_record(native_manifest),
        }
    if set(visit_records) != {"t0", "t1"}:
        raise ProtocolError("native executor did not produce both visits")
    payload = {
        "schema_version": 1,
        "status": "TWO_VISIT_OVI_PASS",
        "protocol_id": protocol["protocol_id"],
        "protocol_content_sha256": plan.protocol_content_sha256,
        "scene": plan.scene,
        "protocol": _file_record(protocol_path),
        "method_input_manifest": _file_record(method_input_path),
        "visits": visit_records,
    }
    _atomic_json(output, payload)
    return output


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError("protocol must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene", choices=("apartment", "office"), required=True)
    parser.add_argument("--source-rgbd-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--office-release", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args(argv)
    build_root = Path("/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native")
    native = TesseNativeEnvironment(
        build_root=build_root,
        frontend_python=Path("/home/ww/miniconda3/envs/ovimap-cropformer/bin/python"),
        mapping_python=Path("/home/ww/miniconda3/envs/ovimap-map/bin/python"),
        entity_root=build_root / "Entity",
        ovimap_root=build_root / "OVI-MAP",
        cropformer_weights=build_root / "Entity/checkpoints/CropFormer_hornet_3x_03823a.pth",
        siglip_model=build_root / "siglip-large-patch16-384",
        source_hashes=build_root / "environment/source-hashes.json",
    )
    release = _load_json(args.office_release) if args.office_release else None
    protocol = _load_json(args.protocol)
    plan = build_independent_ovi_plan(
        protocol,
        scene=args.scene,
        source_rgbd_root=args.source_rgbd_root,
        run_root=args.run_root,
        native=native,
        office_release=release,
    )
    if args.plan_only:
        print(json.dumps({"scene": plan.scene, "t0": plan.t0.run_id, "t1": plan.t1.run_id}, sort_keys=True))
        return 0
    if args.output_manifest is None:
        raise ProtocolError("--output-manifest is required with --execute")
    execute_independent_ovi_plan(
        plan,
        protocol_path=args.protocol,
        source_rgbd_root=args.source_rgbd_root,
        native=native,
        output_manifest=args.output_manifest,
        office_release=release,
    )
    print("TWO_VISIT_OVI_PASS")
    return 0


__all__ = [
    "OviVisitPlan",
    "TwoVisitOviPlan",
    "build_independent_ovi_plan",
    "execute_independent_ovi_plan",
    "main",
    "materialize_visit_window",
]


if __name__ == "__main__":
    raise SystemExit(main())
