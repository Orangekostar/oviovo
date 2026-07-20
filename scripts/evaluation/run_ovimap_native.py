#!/usr/bin/env python3
"""Run hash-bound native CropFormer and OVI-MAP mapping gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines.ovimap_native import audit_native_frame


REPLICA8_SCENES = (
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
)
GATE_TRANSITIONS = {
    "ENV_PASS": "FRONTEND_PASS",
    "FRONTEND_PASS": "MAPPING_PASS",
    "MAPPING_PASS": "ROOM0_PASS",
    "ROOM0_PASS": "REPLICA8_PASS",
    "REPLICA8_PASS": "EVAL_PASS",
}


class GateFailure(RuntimeError):
    """Raised when a native reproduction gate cannot advance."""


@dataclass(frozen=True)
class NativeConfig:
    build_root: Path
    run_root: Path
    data_root: Path
    frontend_python: Path
    mapping_python: Path
    entity_root: Path
    ovimap_root: Path
    cropformer_weights: Path
    siglip_model: Path

    @property
    def cropformer_root(self) -> Path:
        return self.entity_root / "Entityv2/CropFormer"

    @property
    def cropformer_config(self) -> Path:
        return (
            self.cropformer_root
            / "configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml"
        )

    @property
    def mapping_workspace(self) -> Path:
        return self.ovimap_root / "mapping_ros_ws"


@dataclass(frozen=True)
class SceneCommands:
    frame_ids: list[int]
    geometry: tuple[str, ...]
    frontend: tuple[str, ...]
    mapping: tuple[str, ...]
    attempt_root: Path

    @property
    def commands(self) -> tuple[tuple[str, ...], ...]:
        return (self.geometry, self.frontend, self.mapping)


def default_config(run_root: Path) -> NativeConfig:
    build_root = Path("/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native")
    return NativeConfig(
        build_root=build_root,
        run_root=run_root,
        data_root=Path("/home/ww/vv/dataset/Replica"),
        frontend_python=Path("/home/ww/miniconda3/envs/ovimap-cropformer/bin/python"),
        mapping_python=Path("/home/ww/miniconda3/envs/ovimap-map/bin/python"),
        entity_root=build_root / "Entity",
        ovimap_root=build_root / "OVI-MAP",
        cropformer_weights=Path(
            "/home/ww/oviovo_benchmark_assets/weights/"
            "CropFormer_hornet_3x_03823a.pth"
        ),
        siglip_model=Path(
            "/home/ww/oviovo_benchmark_assets/weights/siglip-large-patch16-384"
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": stat.st_size,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def advance_state(current: str, *, event: dict[str, Any]) -> str:
    gate = str(event.get("gate", "gate"))
    if event.get("status") != "PASS":
        raise GateFailure(f"{gate} failed")
    expected = GATE_TRANSITIONS.get(current)
    requested = event.get("next_state")
    if expected is None or requested != expected:
        raise GateFailure(f"invalid transition: {current} -> {requested}")
    return str(requested)


def _frame_ids(start: int, end: int, step: int) -> list[int]:
    if start < 0 or end <= start or step <= 0:
        raise ValueError("frame range requires 0 <= start < end and step > 0")
    return list(range(start, end, step))


def build_scene_commands(
    config: NativeConfig,
    *,
    scene: str,
    start: int = 0,
    end: int = 2000,
    step: int = 10,
    attempt_root: Path | None = None,
) -> SceneCommands:
    if scene not in REPLICA8_SCENES:
        raise ValueError(f"unknown Replica scene: {scene}")
    frame_ids = _frame_ids(start, end, step)
    root = attempt_root or config.run_root / "plan" / scene
    frontend_output = root / "frontend"
    geometry_output = root / "geometric_segments"
    mapping_output = root / "mapping"
    intermediate_output = root / "intermediate_segments"
    audit_output = root / "native_audit"
    images = [
        str(config.data_root / scene / "results" / f"frame{frame_id:06d}.jpg")
        for frame_id in frame_ids
    ]

    geometry = (
        str(config.mapping_python),
        str(Path(__file__).resolve()),
        "--internal-geometry",
        "--scene-data",
        str(config.data_root / scene),
        "--geometry-output",
        str(geometry_output),
        "--frame-ids",
        ",".join(str(value) for value in frame_ids),
        "--ovimap-root",
        str(config.ovimap_root),
    )
    frontend = (
        str(config.frontend_python),
        str(config.cropformer_root / "demo_cropformer/demo_from_dirs.py"),
        "--config-file",
        str(config.cropformer_config),
        "--input",
        *images,
        "--output",
        str(frontend_output),
        "--confidence-threshold",
        "0.5",
        "--out-type",
        "0",
        "--opts",
        "MODEL.WEIGHTS",
        str(config.cropformer_weights),
    )
    mapping = (
        str(config.mapping_python),
        str(config.ovimap_root / "scripts/panoptic_mapping_.py"),
        "--dataset",
        "replica",
        "--task",
        "Nyu40",
        "--scene_num",
        scene,
        "--data_folder",
        str(config.data_root),
        "--result_folder",
        str(mapping_output),
        "--start",
        str(start),
        "--end",
        str(end),
        "--step",
        str(step),
        "--data_association",
        "2",
        "--inst_association",
        "4",
        "--seg_graph_confidence",
        "3",
        "--use_temp_results",
        "--save_temp_results",
        "--intermediate_seg_folder",
        str(intermediate_output),
        "--temp_panoptics_folder",
        str(frontend_output),
        "--use_temp_geometrics",
        "--save_temp_geometrics",
        "--temp_geometrics_folder",
        str(geometry_output),
        "--num_threads",
        "10",
        "--siglip_model_path",
        str(config.siglip_model),
        "--native_audit_dir",
        str(audit_output),
        "--log",
        "ubuntu24-native",
    )
    return SceneCommands(frame_ids, geometry, frontend, mapping, root)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise GateFailure(f"{label} missing: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise GateFailure(f"{label} missing: {path}")


def preflight(config: NativeConfig, commands: SceneCommands, *, scene: str) -> dict[str, Any]:
    for path, label in (
        (config.frontend_python, "frontend python"),
        (config.mapping_python, "mapping python"),
        (config.cropformer_config, "CropFormer config"),
        (config.cropformer_weights, "CropFormer weights"),
        (config.ovimap_root / "scripts/panoptic_mapping_.py", "OVI-MAP mapper"),
        (
            config.mapping_workspace / "devel/lib/consistent_gsm.cpython-311-x86_64-linux-gnu.so",
            "consistent_gsm extension",
        ),
        (
            config.mapping_workspace
            / "devel/lib/depth_segmentation_py.cpython-311-x86_64-linux-gnu.so",
            "depth_segmentation_py extension",
        ),
        (config.data_root / scene / "traj.txt", f"{scene} trajectory"),
        (config.data_root / "cam_params.json", "Replica camera parameters"),
    ):
        _require_file(path, label)
    _require_dir(config.siglip_model, "local SigLIP model")
    for frame_id in commands.frame_ids:
        _require_file(
            config.data_root / scene / "results" / f"frame{frame_id:06d}.jpg",
            f"{scene} RGB frame {frame_id}",
        )
        _require_file(
            config.data_root / scene / "results" / f"depth{frame_id:06d}.png",
            f"{scene} depth frame {frame_id}",
        )
    return {
        "status": "PASS",
        "timestamp": _utc_now(),
        "scene": scene,
        "frame_ids": commands.frame_ids,
        "sources": {
            "cropformer_config": _artifact(config.cropformer_config),
            "cropformer_weights": _artifact(config.cropformer_weights),
            "mapper": _artifact(config.ovimap_root / "scripts/panoptic_mapping_.py"),
            "source_hashes": _artifact(
                config.build_root / "environment/source-hashes.json"
            ),
        },
    }


def _frontend_env(config: NativeConfig) -> dict[str, str]:
    env = os.environ.copy()
    prefix = config.frontend_python.parents[1]
    env["PATH"] = f"{prefix / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{prefix / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{config.cropformer_root}:{env.get('PYTHONPATH', '')}"
    return env


def _mapping_env(config: NativeConfig) -> dict[str, str]:
    env = os.environ.copy()
    prefix = config.mapping_python.parents[1]
    devel = config.mapping_workspace / "devel"
    env["PATH"] = f"{prefix / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{devel / 'lib'}:{prefix / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["PYTHONPATH"] = (
        f"{devel / 'lib'}:{devel / 'lib/python3.11/site-packages'}:"
        f"{config.ovimap_root / 'scripts'}:{env.get('PYTHONPATH', '')}"
    )
    env["OVIMAP_SIGLIP_MODEL"] = str(config.siglip_model)
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _run_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record = {
        "argv": list(argv),
        "cwd": str(cwd.resolve()),
        "started_at": started,
        "finished_at": _utc_now(),
        "exit_code": completed.returncode,
        "log": _artifact(log),
    }
    if completed.returncode != 0:
        raise GateFailure(f"command failed with exit code {completed.returncode}: {log}")
    return record


def run_frontend(config: NativeConfig, commands: SceneCommands) -> dict[str, Any]:
    record = _run_command(
        commands.frontend,
        cwd=config.cropformer_root,
        env=_frontend_env(config),
        log=commands.attempt_root / "logs/frontend.log",
    )
    output = commands.attempt_root / "frontend"
    for frame_id in commands.frame_ids:
        _require_file(output / f"frame{frame_id:06d}.png", "CropFormer instance mask")
        _require_file(
            output / "frame_diagnostics" / f"frame{frame_id:06d}.json",
            "CropFormer frame diagnostics",
        )
    record["status"] = "PASS"
    record["mask_count"] = len(commands.frame_ids)
    return record


def run_mapping(config: NativeConfig, commands: SceneCommands) -> dict[str, Any]:
    env = _mapping_env(config)
    geometry = _run_command(
        commands.geometry,
        cwd=REPO_ROOT,
        env=env,
        log=commands.attempt_root / "logs/geometry.log",
    )
    mapping = _run_command(
        commands.mapping,
        cwd=config.ovimap_root,
        env=env,
        log=commands.attempt_root / "logs/mapping.log",
    )
    return {"status": "PASS", "geometry": geometry, "mapping": mapping}


def audit_scene(commands: SceneCommands) -> dict[str, Any]:
    frontend = commands.attempt_root / "frontend"
    native = commands.attempt_root / "native_audit"
    colors_path = native / "color_pairs.json"
    _require_file(colors_path, "native color pairs")
    color_pairs = json.loads(colors_path.read_text(encoding="utf-8"))
    if not isinstance(color_pairs, list) or not color_pairs:
        raise GateFailure("native frame audit failed: no color pairs")
    colors = {int(pair["instance_id"]): pair for pair in color_pairs}
    frame_records: list[dict[str, Any]] = []
    total_boxes = 0
    total_full_frame = 0
    for frame_id in commands.frame_ids:
        mask_path = frontend / f"frame{frame_id:06d}.png"
        raycast_path = native / f"frame{frame_id:06d}.raycast.npy"
        mapper_path = native / f"frame{frame_id:06d}.json"
        for path, label in (
            (mask_path, "instance mask"),
            (raycast_path, "raycast IDs"),
            (mapper_path, "mapper frame audit"),
        ):
            _require_file(path, label)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise GateFailure(f"native frame audit failed: unreadable mask {mask_path}")
        raycast = np.load(raycast_path, allow_pickle=False)
        mapper = json.loads(mapper_path.read_text(encoding="utf-8"))
        raycast_ids = {int(value) for value in np.unique(raycast) if value > 0}
        if not raycast_ids <= set(colors):
            raise GateFailure("native frame audit failed: raycast color IDs are incomplete")
        try:
            audit = audit_native_frame(
                mask,
                raycast,
                [colors[value] for value in sorted(raycast_ids)],
            )
        except ValueError as error:
            raise GateFailure(f"native frame audit failed: {error}") from error
        expected_boxes = {
            str(instance_id): box
            for instance_id, box in zip(sorted(raycast_ids), audit["boxes"], strict=True)
        }
        if mapper.get("box_2d") != expected_boxes:
            raise GateFailure("native frame audit failed: mapper bbox serialization mismatch")
        for instance_id in raycast_ids:
            if mapper.get("mapper_rgb", {}).get(str(instance_id)) != colors[instance_id][
                "mapper_rgb"
            ]:
                raise GateFailure("native frame audit failed: mapper RGB serialization mismatch")
        total_boxes += int(audit["raycast_instance_count"])
        total_full_frame += int(audit["full_frame_bbox_count"])
        frame_record = {
            "frame_id": frame_id,
            "audit": audit,
            "inputs": {
                "mask": _artifact(mask_path),
                "raycast": _artifact(raycast_path),
                "mapper": _artifact(mapper_path),
            },
        }
        _atomic_json(native / "validated" / f"frame{frame_id:06d}.json", frame_record)
        frame_records.append(frame_record)
    if total_boxes == 0:
        raise GateFailure("native frame audit failed: no non-empty raycast boxes")
    if total_full_frame == total_boxes:
        raise GateFailure("native frame audit failed: all bboxes are full-frame")
    summary = {
        "status": "PASS",
        "frame_ids": commands.frame_ids,
        "frame_count": len(frame_records),
        "raycast_bbox_count": total_boxes,
        "full_frame_bbox_count": total_full_frame,
        "full_frame_bbox_ratio": total_full_frame / total_boxes,
        "color_pairs": _artifact(colors_path),
    }
    _atomic_json(native / "scene_audit.json", summary)
    return summary


def _mapping_artifacts(commands: SceneCommands) -> dict[str, Any]:
    count = len(commands.frame_ids)
    root = commands.attempt_root / "mapping/cropformer_inst"
    expected = {
        "instance_mesh": root / f"instance_mesh_{count}.ply",
        "semantic_features": root / f"inst_sem_siglip-l-16-384_{count}_incre_combine.pkl",
    }
    for name, path in expected.items():
        _require_file(path, f"mapping {name}")
    return {name: _artifact(path) for name, path in expected.items()}


def write_scene_manifest(
    commands: SceneCommands,
    *,
    scene: str,
    stage: str,
    state: str,
    preflight_record: dict[str, Any],
    frontend_record: dict[str, Any],
    mapping_record: dict[str, Any],
    audit_record: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "state": state,
        "stage": stage,
        "scene": scene,
        "frame_ids": commands.frame_ids,
        "created_at": _utc_now(),
        "preflight": preflight_record,
        "frontend": frontend_record,
        "mapping": mapping_record,
        "audit": audit_record,
        "artifacts": _mapping_artifacts(commands),
    }
    path = commands.attempt_root / "native_mapping_manifest.json"
    _atomic_json(path, payload)
    return path


def _new_attempt(run_root: Path, scene: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = run_root / "scenes" / scene / f"attempt-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "frontend",
        "geometric_segments",
        "intermediate_segments",
        "mapping",
        "native_audit",
        "logs",
    ):
        (root / name).mkdir()
    return root


def run_scene(
    config: NativeConfig,
    *,
    scene: str,
    stage: str,
    start: int,
    end: int,
    step: int,
) -> Path:
    attempt = _new_attempt(config.run_root, scene)
    commands = build_scene_commands(
        config,
        scene=scene,
        start=start,
        end=end,
        step=step,
        attempt_root=attempt,
    )
    preflight_record = preflight(config, commands, scene=scene)
    state = "ENV_PASS"
    frontend_record = run_frontend(config, commands)
    state = advance_state(
        state,
        event={"status": "PASS", "next_state": "FRONTEND_PASS", "gate": "frontend"},
    )
    mapping_record = run_mapping(config, commands)
    audit_record = audit_scene(commands)
    state = advance_state(
        state,
        event={
            "status": audit_record["status"],
            "next_state": "MAPPING_PASS",
            "gate": "native frame audit",
        },
    )
    if stage == "room" and scene == "room0":
        state = advance_state(
            state,
            event={"status": "PASS", "next_state": "ROOM0_PASS", "gate": "room0"},
        )
    return write_scene_manifest(
        commands,
        scene=scene,
        stage=stage,
        state=state,
        preflight_record=preflight_record,
        frontend_record=frontend_record,
        mapping_record=mapping_record,
        audit_record=audit_record,
    )


def _parse_frame_ids(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("frame IDs must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("frame IDs cannot be empty")
    return values


def _run_internal_geometry(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(args.ovimap_root / "scripts"))
    import depth_segmentation_py

    camera = json.loads(
        (args.scene_data.parent / "cam_params.json").read_text(encoding="utf-8")
    )["camera"]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = camera["fx"]
    intrinsic[1, 1] = camera["fy"]
    intrinsic[0, 2] = camera["cx"]
    intrinsic[1, 2] = camera["cy"]
    segmenter = depth_segmentation_py.DepthSegmentation_py(
        int(camera["h"]), int(camera["w"]), cv2.CV_32FC1, intrinsic
    )
    args.geometry_output.mkdir(parents=True, exist_ok=True)
    for frame_id in args.frame_ids:
        rgb = cv2.imread(
            str(args.scene_data / "results" / f"frame{frame_id:06d}.jpg"),
            cv2.IMREAD_COLOR,
        )
        depth = cv2.imread(
            str(args.scene_data / "results" / f"depth{frame_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if rgb is None or depth is None:
            raise GateFailure(f"geometry input missing for frame {frame_id}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        segmenter.depthSegment(
            depth.astype(np.float32) / float(camera["scale"]),
            rgb.astype(np.float32),
        )
        masks = np.asarray(segmenter.get_segmentMasks(), dtype=bool)
        if masks.shape[0] > np.iinfo(np.uint8).max:
            raise GateFailure(f"too many geometric segments in frame {frame_id}")
        output = np.zeros(depth.shape, dtype=np.uint8)
        for index, mask in enumerate(masks, start=1):
            output[mask] = index
        path = args.geometry_output / f"{frame_id:05d}_mask.png"
        if not cv2.imwrite(str(path), output):
            raise OSError(f"failed to write geometric segmentation: {path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mapping", "room"))
    parser.add_argument("--scene", choices=REPLICA8_SCENES)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--internal-geometry", action="store_true")
    parser.add_argument("--scene-data", type=Path)
    parser.add_argument("--geometry-output", type=Path)
    parser.add_argument("--frame-ids", type=_parse_frame_ids)
    parser.add_argument("--ovimap-root", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_geometry:
        required = (args.scene_data, args.geometry_output, args.frame_ids, args.ovimap_root)
        if any(value is None for value in required):
            raise GateFailure("internal geometry command is incomplete")
        return _run_internal_geometry(args)
    if args.stage is None or args.scene is None or args.run_root is None:
        raise GateFailure("--stage, --scene, and --run-root are required")
    config = default_config(args.run_root)
    try:
        manifest = run_scene(
            config,
            scene=args.scene,
            stage=args.stage,
            start=args.start,
            end=args.end,
            step=args.step,
        )
    except (OSError, ValueError, GateFailure, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
