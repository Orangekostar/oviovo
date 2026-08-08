#!/usr/bin/env python3
"""Run checkpointed fast-eval for the selected Replica scenes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

SCENES: dict[str, str] = {
    "room0": "room_0",
    "room1": "room_1",
    "room2": "room_2",
    "office0": "office_0",
    "office1": "office_1",
    "office2": "office_2",
    "office3": "office_3",
    "office4": "office_4",
}

DEFAULT_PYTHON_EXECUTABLE = Path("/home/ww/miniconda3/envs/oviovo/bin/python")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "tmp_validation"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "room0_surface_gate_fast_high_iou_4090.yaml"
DEFAULT_ONLINE_CONFIG_PATH = REPO_ROOT / "configs" / "replica_yoloworld_sam_online_baseline_4090.yaml"
DEFAULT_SAM3_CONFIG_PATH = REPO_ROOT / "configs" / "replica_sam3_concept_room0_experiment_4090.yaml"
DEFAULT_DATASET_ROOT_BASE = Path("/home/ww/vv/dataset/Replica")
DEFAULT_GT_ORIGINAL_BASE = Path("/home/ww/vv/dataset/Replica-Dataset/Replica_original")
DEFAULT_GT_LABEL_DIR = REPO_ROOT / "data" / "input" / "replica_semantic_gt"
DEFAULT_RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_room0_checkpointed_eval.py"
DEFAULT_SAM_REPO_ROOT = Path("/home/ww/vv/paper2/OVO/thirdParty/segment-anything-2")
DEFAULT_SAM_CKPT_PATH = Path("/home/ww/vv/oviovo/data/input/sam_ckpts")
DEFAULT_NUM_FRAMES = 200
DEFAULT_FRAME_STRIDE = 10
DEFAULT_PROPOSAL_BACKEND = "precomputed"
DEFAULT_PROPOSAL_DEVICE = "cuda"

RUN_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "HF_HUB_OFFLINE": "1",
}


@dataclass(frozen=True)
class ScenePaths:
    dataset_root: Path
    gt_labels: Path
    gt_mesh_ply: Path
    gt_info_json: Path


@dataclass(frozen=True)
class ProposalCacheOverride:
    manifest_path: Path | None = None
    cache_dir: Path | None = None


def selected_scene_names(scenes: Sequence[str] | None) -> list[str]:
    if scenes is None:
        return list(SCENES)
    selected = list(dict.fromkeys(str(scene).strip() for scene in scenes if str(scene).strip()))
    unknown = [scene for scene in selected if scene not in SCENES]
    if unknown:
        raise ValueError(f"Unknown scene(s): {', '.join(unknown)}. Valid scenes: {', '.join(SCENES)}")
    return selected


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _repo_relative(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _default_online_config_path() -> Path:
    return REPO_ROOT / "configs" / DEFAULT_ONLINE_CONFIG_PATH.name


def _expand_scene_template(template: str, *, scene: str) -> Path:
    return _repo_relative(Path(template.format(scene=scene, gt_scene=SCENES[scene])))


def scene_paths(
    scene: str,
    *,
    dataset_root_base: Path = DEFAULT_DATASET_ROOT_BASE,
    gt_original_base: Path = DEFAULT_GT_ORIGINAL_BASE,
    gt_label_dir: Path = DEFAULT_GT_LABEL_DIR,
) -> ScenePaths:
    gt_scene = SCENES[scene]
    habitat_dir = Path(gt_original_base) / gt_scene / "habitat"
    return ScenePaths(
        dataset_root=Path(dataset_root_base) / scene,
        gt_labels=Path(gt_label_dir) / f"{scene}.txt",
        gt_mesh_ply=habitat_dir / "mesh_semantic.ply",
        gt_info_json=habitat_dir / "info_semantic.json",
    )


def experiment_name_for_scene(batch_name: str, scene: str, *, frame_stride: int, num_frames: int) -> str:
    return f"{batch_name}_{scene}_s{int(frame_stride)}_{int(num_frames)}f_fast_eval"


def build_scene_command(
    *,
    scene: str,
    experiment_name: str,
    paths: ScenePaths,
    python_executable: Path,
    runner_script: Path,
    config_path: Path,
    output_root: Path,
    num_frames: int,
    frame_stride: int,
    proposal_backend: str,
    proposal_device: str,
    proposal_cache: ProposalCacheOverride | None = None,
    sam3_mode: str | None = None,
    sam3_worker_python: Path | None = None,
    sam3_worker_script: Path | None = None,
    sam3_repo_root: Path | None = None,
    sam3_checkpoint_path: Path | None = None,
) -> list[str]:
    command = [
        str(python_executable),
        str(runner_script),
        "--mode",
        "build-export",
        "--scene-name",
        scene,
        "--experiment-name",
        experiment_name,
        "--dataset-root",
        str(paths.dataset_root),
        "--gt-labels",
        str(paths.gt_labels),
        "--gt-mesh-ply",
        str(paths.gt_mesh_ply),
        "--gt-info-json",
        str(paths.gt_info_json),
        "--config-path",
        str(config_path),
        "--output-root",
        str(output_root),
        "--num-frames",
        str(int(num_frames)),
        "--frame-stride",
        str(int(frame_stride)),
        "--fast-eval",
        "--proposal-backend",
        proposal_backend,
        "--proposal-device",
        proposal_device,
        "--quiet",
    ]
    if proposal_backend == "sam2":
        command.extend(
            [
                "--sam-version",
                "2.1",
                "--sam-repo-root",
                str(DEFAULT_SAM_REPO_ROOT),
                "--sam-ckpt-path",
                str(DEFAULT_SAM_CKPT_PATH),
                "--points-per-side",
                "16",
                "--max-proposals",
                "50",
                "--confidence-threshold",
                "0.5",
                "--min-mask-area",
                "100",
                "--stability-score-th",
                "0.95",
                "--nms-iou-th",
                "0.8",
                "--min-mask-region-area",
                "100",
            ]
        )
    if proposal_backend == "sam3_concept":
        if sam3_mode:
            command.extend(["--sam3-mode", str(sam3_mode)])
        if sam3_worker_python is not None:
            command.extend(["--sam3-worker-python", str(sam3_worker_python)])
        if sam3_worker_script is not None:
            command.extend(["--sam3-worker-script", str(sam3_worker_script)])
        if sam3_repo_root is not None:
            command.extend(["--sam3-repo-root", str(sam3_repo_root)])
        if sam3_checkpoint_path is not None:
            command.extend(["--sam3-checkpoint-path", str(sam3_checkpoint_path)])
    if proposal_backend == "precomputed" and proposal_cache is not None:
        manifest_path = proposal_cache.manifest_path or _manifest_for_cache_dir(proposal_cache.cache_dir)
        if manifest_path is not None:
            command.extend(["--proposal-cache-manifest", str(manifest_path)])
        if proposal_cache.cache_dir is not None:
            command.extend(["--proposal-cache-dir", str(proposal_cache.cache_dir)])
    return command


def _load_precomputed_config(config_path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"_error": f"failed to read config precomputed proposal settings from {config_path}: {exc}"}
    if not isinstance(payload, dict):
        return {}
    proposal = payload.get("proposal")
    if not isinstance(proposal, dict):
        return {}
    precomputed = proposal.get("precomputed")
    return precomputed if isinstance(precomputed, dict) else {}


def _manifest_for_cache_dir(cache_dir: Path | None) -> Path | None:
    return None if cache_dir is None else Path(cache_dir) / "manifest.json"


def _validate_manifest_scene(
    *,
    scene: str,
    manifest_path: Path,
    num_frames: int,
    frame_stride: int,
) -> list[str]:
    if not manifest_path.exists():
        return [f"{scene}: missing precomputed cache manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{scene}: invalid precomputed cache manifest JSON: {manifest_path}: {exc}"]
    dataset_summary = manifest.get("dataset_summary") if isinstance(manifest, dict) else None
    dataset_root = dataset_summary.get("root") if isinstance(dataset_summary, dict) else None
    if not dataset_root:
        return [f"{scene}: precomputed cache manifest missing dataset_summary.root: {manifest_path}"]
    manifest_scene = Path(str(dataset_root)).name
    if manifest_scene != scene:
        return [
            f"{scene}: precomputed cache manifest scene mismatch: {manifest_path} "
            f"has dataset_summary.root basename '{manifest_scene}'"
        ]

    frames = manifest.get("frames") if isinstance(manifest, dict) else None
    if not isinstance(frames, list):
        return [f"{scene}: precomputed cache manifest missing frames list: {manifest_path}"]
    frame_index: dict[int, dict[str, object]] = {}
    for entry in frames:
        if not isinstance(entry, dict) or "frame_id" not in entry:
            continue
        try:
            frame_index[int(entry["frame_id"])] = entry
        except (TypeError, ValueError):
            continue

    required_frame_ids = [idx * int(frame_stride) for idx in range(int(num_frames))]
    errors: list[str] = []
    for frame_id in required_frame_ids:
        entry = frame_index.get(frame_id)
        if entry is None:
            errors.append(f"{scene}: precomputed cache manifest missing requested frame id {frame_id}: {manifest_path}")
            continue
        frame_file = entry.get("file")
        if not frame_file:
            errors.append(f"{scene}: precomputed cache manifest frame {frame_id} missing file: {manifest_path}")
            continue
        frame_path = manifest_path.parent / str(frame_file)
        if not frame_path.exists():
            errors.append(f"{scene}: missing precomputed cache frame file for frame {frame_id}: {frame_path}")
    if errors:
        return errors
    return []


def effective_proposal_cache_override(
    *,
    scene: str,
    manifest_template: str | None,
    dir_template: str | None,
) -> ProposalCacheOverride | None:
    manifest_path = _expand_scene_template(manifest_template, scene=scene) if manifest_template else None
    cache_dir = _expand_scene_template(dir_template, scene=scene) if dir_template else None
    if manifest_path is None and cache_dir is None:
        return None
    return ProposalCacheOverride(manifest_path=manifest_path, cache_dir=cache_dir)


def validate_precomputed_cache_for_scene(
    *,
    scene: str,
    config_path: Path,
    override: ProposalCacheOverride | None,
    num_frames: int,
    frame_stride: int,
) -> list[str]:
    if override is not None:
        manifest_path = override.manifest_path or _manifest_for_cache_dir(override.cache_dir)
        if manifest_path is None:
            return [f"{scene}: precomputed proposal cache override did not provide a manifest or cache dir"]
        return _validate_manifest_scene(
            scene=scene,
            manifest_path=manifest_path,
            num_frames=num_frames,
            frame_stride=frame_stride,
        )

    precomputed = _load_precomputed_config(config_path)
    config_error = precomputed.get("_error") if isinstance(precomputed, dict) else None
    if config_error:
        return [f"{scene}: {config_error}"]
    manifest_value = precomputed.get("manifest_path") if isinstance(precomputed, dict) else None
    cache_dir_value = precomputed.get("cache_dir") if isinstance(precomputed, dict) else None
    manifest_path = _repo_relative(Path(manifest_value)) if manifest_value else _manifest_for_cache_dir(
        _repo_relative(Path(cache_dir_value)) if cache_dir_value else None
    )
    if manifest_path is None:
        return [f"{scene}: precomputed proposal backend requires a cache manifest or cache dir in {config_path}"]
    return _validate_manifest_scene(
        scene=scene,
        manifest_path=manifest_path,
        num_frames=num_frames,
        frame_stride=frame_stride,
    )


def validate_scene_inputs(
    *,
    scene: str,
    paths: ScenePaths,
    python_executable: Path,
    config_path: Path,
    runner_script: Path,
) -> list[str]:
    errors: list[str] = []
    path_checks = [
        ("dataset root", paths.dataset_root),
        ("GT labels", paths.gt_labels),
        ("GT mesh PLY", paths.gt_mesh_ply),
        ("GT info JSON", paths.gt_info_json),
        ("config path", Path(config_path)),
        ("runner script", Path(runner_script)),
    ]
    errors.extend(f"{scene}: missing {label}: {path}" for label, path in path_checks if not Path(path).exists())
    python_executable = Path(python_executable)
    if not python_executable.exists():
        errors.append(f"{scene}: missing python executable: {python_executable}")
    elif not python_executable.is_file():
        errors.append(f"{scene}: python executable is not a file: {python_executable}")
    elif not os.access(python_executable, os.X_OK):
        errors.append(f"{scene}: python executable is not executable: {python_executable}")
    return errors


def _json_status_is_complete(status_path: Path) -> bool:
    if not status_path.exists():
        return False
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "complete"


def _line_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def scene_is_complete(run_root: Path, scene: str, *, expected_frames: int) -> bool:
    run_root = Path(run_root)
    scene_dir = run_root / scene
    report_paths = [scene_dir / "run_report.md", run_root / "room0" / "run_report.md"]
    return all(
        [
            _json_status_is_complete(run_root / "status.json"),
            _line_count(scene_dir / "frame_metrics.jsonl") == int(expected_frames),
            (scene_dir / "mapping_state.pkl").exists(),
            (run_root / "replica" / "results.json").exists(),
            any(path.exists() for path in report_paths),
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--scenes", nargs="+", default=None, help="Scene names to run. Defaults to all selected scenes.")
    parser.add_argument("--num-frames", type=_positive_int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--frame-stride", type=_positive_int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON_EXECUTABLE)
    parser.add_argument("--runner-script", type=Path, default=DEFAULT_RUNNER_SCRIPT)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--dataset-root-base", type=Path, default=DEFAULT_DATASET_ROOT_BASE)
    parser.add_argument("--gt-original-base", type=Path, default=DEFAULT_GT_ORIGINAL_BASE)
    parser.add_argument("--gt-label-dir", type=Path, default=DEFAULT_GT_LABEL_DIR)
    parser.add_argument("--proposal-backend", default=None)
    parser.add_argument("--proposal-device", default=DEFAULT_PROPOSAL_DEVICE)
    parser.add_argument("--online-yoloworld-sam", action="store_true")
    parser.add_argument("--online-sam3-concept", action="store_true")
    parser.add_argument("--sam3-worker-python", type=Path, default=None)
    parser.add_argument("--sam3-worker-script", type=Path, default=None)
    parser.add_argument("--sam3-mode", type=str, default=None)
    parser.add_argument("--sam3-repo-root", type=Path, default=None)
    parser.add_argument("--sam3-checkpoint-path", type=Path, default=None)
    parser.add_argument("--proposal-cache-manifest-template", default=None)
    parser.add_argument("--proposal-cache-dir-template", default=None)
    args = parser.parse_args(argv)
    if args.scenes is not None:
        try:
            args.scenes = selected_scene_names(args.scenes)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def format_scene_command(command: Sequence[str]) -> str:
    env_prefix = [f"{key}={value}" for key, value in RUN_ENV.items()]
    return shlex.join([*env_prefix, *command])


def _print_command(command: Sequence[str]) -> None:
    print(format_scene_command(command))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    online_yoloworld_sam = bool(args.online_yoloworld_sam)
    online_sam3_concept = bool(args.online_sam3_concept)
    if online_yoloworld_sam and online_sam3_concept:
        raise SystemExit("--online-yoloworld-sam and --online-sam3-concept are mutually exclusive")
    if online_sam3_concept:
        default_config = REPO_ROOT / "configs" / DEFAULT_SAM3_CONFIG_PATH.name
    elif online_yoloworld_sam:
        default_config = _default_online_config_path()
    else:
        default_config = DEFAULT_CONFIG_PATH
    proposal_backend = args.proposal_backend or (
        "sam3_concept" if online_sam3_concept else ("sam2" if online_yoloworld_sam else DEFAULT_PROPOSAL_BACKEND)
    )
    scenes = selected_scene_names(args.scenes)
    output_root = _repo_relative(args.output_root)
    python_executable = _repo_relative(args.python_executable)
    runner_script = _repo_relative(args.runner_script)
    config_path = _repo_relative(args.config_path or default_config)
    dataset_root_base = _repo_relative(args.dataset_root_base)
    gt_original_base = _repo_relative(args.gt_original_base)
    gt_label_dir = _repo_relative(args.gt_label_dir)

    jobs: list[tuple[str, str, Path, ScenePaths, list[str]]] = []
    errors: list[str] = []
    for scene in scenes:
        paths = scene_paths(
            scene,
            dataset_root_base=dataset_root_base,
            gt_original_base=gt_original_base,
            gt_label_dir=gt_label_dir,
        )
        experiment_name = experiment_name_for_scene(
            args.batch_name,
            scene,
            frame_stride=args.frame_stride,
            num_frames=args.num_frames,
        )
        run_root = output_root / experiment_name
        proposal_cache = effective_proposal_cache_override(
            scene=scene,
            manifest_template=args.proposal_cache_manifest_template,
            dir_template=args.proposal_cache_dir_template,
        )
        command = build_scene_command(
            scene=scene,
            experiment_name=experiment_name,
            paths=paths,
            python_executable=python_executable,
            runner_script=runner_script,
            config_path=config_path,
            output_root=output_root,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            proposal_backend=proposal_backend,
            proposal_device=args.proposal_device,
            proposal_cache=proposal_cache,
            sam3_mode=args.sam3_mode,
            sam3_worker_python=_repo_relative(args.sam3_worker_python)
            if args.sam3_worker_python is not None
            else None,
            sam3_worker_script=_repo_relative(args.sam3_worker_script)
            if args.sam3_worker_script is not None
            else None,
            sam3_repo_root=_repo_relative(args.sam3_repo_root)
            if args.sam3_repo_root is not None
            else None,
            sam3_checkpoint_path=_repo_relative(args.sam3_checkpoint_path)
            if args.sam3_checkpoint_path is not None
            else None,
        )
        errors.extend(
            validate_scene_inputs(
                scene=scene,
                paths=paths,
                python_executable=python_executable,
                config_path=config_path,
                runner_script=runner_script,
            )
        )
        if proposal_backend == "precomputed":
            errors.extend(
                validate_precomputed_cache_for_scene(
                    scene=scene,
                    config_path=config_path,
                    override=proposal_cache,
                    num_frames=args.num_frames,
                    frame_stride=args.frame_stride,
                )
            )
        jobs.append((scene, experiment_name, run_root, paths, command))

    if args.dry_run:
        for scene, experiment_name, _run_root, _paths, command in jobs:
            print(f"[dry-run] {scene}: {experiment_name}")
            _print_command(command)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.update(RUN_ENV)
    for scene, experiment_name, run_root, _paths, command in jobs:
        if scene_is_complete(run_root, scene, expected_frames=args.num_frames):
            print(f"[skip] {scene}: complete ({experiment_name})")
            continue
        print(f"[run] {scene}: {experiment_name}")
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        print(f"[done] {scene}: {experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
