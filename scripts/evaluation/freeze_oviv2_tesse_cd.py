#!/usr/bin/env python3
"""Freeze the selected OVIV2 Stage3 TESSE-CD configuration and its inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3_LINEAGE_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
SCENES = ("apartment", "office")
TUNED_PARAMETERS = {
    "visibility_depth_tolerance_m": (0.05, 0.10, 0.15),
    "absence_negative_support": (0.5, 1.0),
    "ownership_min_net_support": (0.000001, 0.5, 1.0),
}
ALGORITHM_EXCLUDED_FIELDS = frozenset(
    {
        "algorithm_hash",
        "scene",
        "frame_count",
        "dataset_root",
        "export_manifest",
        "frontend_cache_dir",
        "frontend_manifest",
        "dense_cache_dir",
        "dense_manifest",
        "evaluation_checkpoint_frames",
        "vocabulary_json",
        "vocabulary_txt",
    }
)
SCENE_DIFFERENCE_FIELDS = ALGORITHM_EXCLUDED_FIELDS - {"algorithm_hash"}
_SHA256_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_CONFIG_MARKERS = ("route3", "surface-observation", "stage4", "scannet200")
_OFFICE_METRIC_KEYS = frozenset(
    {"metric_source", "metrics_path", "evaluation_result", "official_metrics"}
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, role: str) -> dict[str, Any]:
    _require_regular_file(path, role)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _require_regular_file(path: Path, role: str) -> None:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{role} must be a regular non-symlink file: {path}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file: {path}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value).rstrip(b"\n"))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX
    )


def _resolve_path(value: object, repo_root: Path, role: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} path must be a non-empty string")
    path = Path(value).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _verify_binding(
    value: object,
    *,
    repo_root: Path,
    role: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{role} binding must be an object")
    path = _resolve_path(value.get("path"), repo_root, role)
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{role} path binding mismatch")
    _require_regular_file(path, role)
    expected_hash = value.get("sha256")
    if not _is_sha256(expected_hash):
        raise ValueError(f"{role} binding requires a sha256 hash")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{role} checksum binding mismatch")
    byte_count = path.stat().st_size
    if "byte_count" in value and (
        type(value["byte_count"]) is not int or value["byte_count"] != byte_count
    ):
        raise ValueError(f"{role} byte_count binding mismatch")
    return {"path": str(path), "sha256": actual_hash, "byte_count": byte_count}


def _git(repo_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _inspect_repository(repo_root: Path) -> dict[str, Any]:
    commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    parents = _git(repo_root, "show", "-s", "--format=%P", "HEAD").stdout.split()
    clean = not _git(
        repo_root, "status", "--porcelain", "--untracked-files=all"
    ).stdout.strip()
    lineage = _git(repo_root, "rev-parse", STAGE3_LINEAGE_COMMIT).stdout.strip()
    ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        STAGE3_LINEAGE_COMMIT,
        "HEAD",
        check=False,
    ).returncode == 0
    return {
        "commit": commit,
        "parents": parents,
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}").stdout.strip(),
        "commit_time_utc": _git(repo_root, "show", "-s", "--format=%cI", "HEAD").stdout.strip(),
        "clean": clean,
        "stage3_lineage_commit": lineage,
        "stage3_is_ancestor": ancestry,
    }


def _validate_repository(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("clean") is not True:
        raise ValueError("freeze requires a clean repository commit")
    if (
        value.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
        or value.get("stage3_is_ancestor") is not True
    ):
        raise ValueError("freeze commit must descend from Stage3 commit 47962fb")
    for field in ("commit", "tree"):
        raw = value.get(field)
        if not isinstance(raw, str) or len(raw) != 40:
            raise ValueError(f"repository {field} must be a full git object ID")
    parents = value.get("parents")
    if not isinstance(parents, list) or any(
        not isinstance(item, str) or len(item) != 40 for item in parents
    ):
        raise ValueError("repository parents must be full git object IDs")
    if not isinstance(value.get("commit_time_utc"), str) or not value["commit_time_utc"]:
        raise ValueError("repository commit time is missing")
    return {
        "clean": True,
        "commit": value["commit"],
        "parents": list(value["parents"]),
        "tree": value["tree"],
        "commit_time_utc": value["commit_time_utc"],
        "stage3_lineage_commit": STAGE3_LINEAGE_COMMIT,
        "stage3_is_ancestor": True,
    }


def _default_environment() -> dict[str, Any]:
    libraries: dict[str, str] = {}
    for package in ("numpy", "open3d", "torch"):
        try:
            libraries[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            libraries[package] = "UNAVAILABLE"
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": sorted(line.strip() for line in gpu_query.stdout.splitlines() if line.strip()),
        "libraries": dict(sorted(libraries.items())),
    }


def _algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in sorted(config)
        if key not in ALGORITHM_EXCLUDED_FIELDS
    }


def _algorithm_hash(config: Mapping[str, Any]) -> str:
    return _json_hash(_algorithm_config(config))


def _find_forbidden(value: object, path: str = "config") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_forbidden(key, f"{path}.<key>") or _find_forbidden(
                item, f"{path}.{key}"
            )
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_forbidden(item, f"{path}[{index}]")
            if found is not None:
                return found
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _FORBIDDEN_CONFIG_MARKERS):
            return path
    return None


def _validate_source_configs(
    apartment: dict[str, Any], office: dict[str, Any]
) -> None:
    for scene, config in zip(SCENES, (apartment, office), strict=True):
        if (
            config.get("schema_version") != 1
            or config.get("method_id") != "OVIV2"
            or config.get("dataset") != "TESSE-CD"
            or config.get("scene") != scene
            or config.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
        ):
            raise ValueError(f"{scene} config identity or Stage3 lineage mismatch")
        forbidden = _find_forbidden(config)
        if forbidden is not None:
            raise ValueError(f"forbidden route3/Stage4/ScanNet200 reference at {forbidden}")
        calculated = _algorithm_hash(config)
        if config.get("algorithm_hash") != calculated:
            raise ValueError(f"{scene} config algorithm_hash is stale")
    if set(apartment) != set(office):
        raise ValueError("scene configs must expose the same configuration keys")
    differences = {
        key for key in apartment if apartment.get(key) != office.get(key)
    }
    if not differences <= SCENE_DIFFERENCE_FIELDS:
        invalid = sorted(differences - SCENE_DIFFERENCE_FIELDS)
        raise ValueError(f"Apartment and Office algorithm configuration differs: {invalid}")
    if _algorithm_config(apartment) != _algorithm_config(office):
        raise ValueError("Apartment and Office normalized algorithm configuration differs")


def _finite_metric(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric metric")
    return float(value)


def _candidate_from_config(
    base_config: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = dict(base_config)
    candidate.update(parameters)
    candidate["algorithm_hash"] = _algorithm_hash(candidate)
    return candidate


def _validate_target_manifest(
    binding: object, repo_root: Path, schedule_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_binding = _verify_binding(
        binding, repo_root=repo_root, role="common target manifest"
    )
    target_path = Path(target_binding["path"])
    target = _load_json(target_path, "common target manifest")
    metadata = target.get("metadata")
    arrays = target.get("target_arrays")
    if (
        target.get("dataset") != "TESSE-CD"
        or target.get("status") != "GENERATED"
        or target.get("targets_generated") is not True
        or target.get("prediction_inputs_used") is not False
        or not isinstance(metadata, dict)
        or metadata.get("protocol_complete") is not True
        or not isinstance(metadata.get("schedule"), dict)
        or metadata["schedule"].get("sha256") != schedule_sha256
        or not isinstance(arrays, dict)
        or not _is_sha256(arrays.get("sha256"))
        or type(arrays.get("byte_count")) is not int
        or type(arrays.get("count")) is not int
        or arrays["count"] <= 0
    ):
        raise ValueError("common target manifest is incomplete or unbound")
    arrays_path = _resolve_path(
        arrays.get("path"), target_path.parent, "common target arrays"
    )
    _require_regular_file(arrays_path, "common target arrays")
    if (
        _sha256(arrays_path) != arrays["sha256"]
        or arrays_path.stat().st_size != arrays["byte_count"]
    ):
        raise ValueError("common target arrays checksum binding mismatch")
    arrays_binding = {
        "path": str(arrays_path),
        "sha256": arrays["sha256"],
        "byte_count": arrays["byte_count"],
        "count": arrays["count"],
    }
    return target_binding, arrays_binding


def _validate_selection(
    selection_path: Path,
    selection: dict[str, Any],
    *,
    apartment_config: dict[str, Any],
    repo_root: Path,
    schedule_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    expected_grid = {key: list(values) for key, values in TUNED_PARAMETERS.items()}
    if (
        selection.get("schema_version") != 1
        or selection.get("method") != "OVIV2"
        or selection.get("scene") != "apartment"
    ):
        raise ValueError("Apartment selection identity mismatch")
    if selection.get("parameter_grid") != expected_grid:
        raise ValueError("selection does not use the predeclared parameter grid")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 18:
        raise ValueError("Apartment selection must contain exactly 18 candidates")
    target_binding, arrays_binding = _validate_target_manifest(
        selection.get("common_target_manifest"), repo_root, schedule_sha256
    )

    expected_combinations = {
        tuple((name, value) for name, value in zip(TUNED_PARAMETERS, values, strict=True))
        for values in __import__("itertools").product(*TUNED_PARAMETERS.values())
    }
    observed_combinations: set[tuple[tuple[str, Any], ...]] = set()
    validated: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} must be an object")
        parameters = candidate.get("parameters")
        if not isinstance(parameters, dict) or set(parameters) != set(TUNED_PARAMETERS):
            raise ValueError(f"candidate {index} parameter grid is invalid")
        combination = tuple((name, parameters[name]) for name in TUNED_PARAMETERS)
        if combination not in expected_combinations or combination in observed_combinations:
            raise ValueError(f"candidate {index} parameter grid is outside the predeclared sweep")
        observed_combinations.add(combination)
        candidate_config = _candidate_from_config(apartment_config, parameters)
        config_sha256 = _json_hash(candidate_config)
        if candidate.get("config_sha256") != config_sha256:
            raise ValueError(f"candidate {index} config hash mismatch")
        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != {
            "current_miou",
            "ghost_rate",
            "background_f5_cm",
            "recovery_frames",
        }:
            raise ValueError(f"candidate {index} metric set mismatch")
        normalized_metrics = {
            key: _finite_metric(value, f"candidate {index} {key}")
            for key, value in metrics.items()
        }
        if (
            candidate.get("schema_version") != 1
            or candidate.get("status") != "PASS"
            or candidate.get("scene") != "apartment"
            or candidate.get("common_target_manifest_sha256") != target_binding["sha256"]
        ):
            raise ValueError(f"candidate {index} summary identity mismatch")
        summary_binding = _verify_binding(
            candidate.get("summary"),
            repo_root=repo_root,
            role=f"candidate summary {index}",
        )
        summary = _load_json(Path(summary_binding["path"]), f"candidate summary {index}")
        expected_summary = {key: value for key, value in candidate.items() if key != "summary"}
        if summary != expected_summary:
            raise ValueError(f"candidate summary {index} content mismatch")
        validated.append(
            {
                "parameters": dict(parameters),
                "metrics": normalized_metrics,
                "config_sha256": config_sha256,
                "summary": summary_binding,
            }
        )
    if observed_combinations != expected_combinations:
        raise ValueError("Apartment selection does not cover the full parameter grid")
    winner = min(
        validated,
        key=lambda item: (
            -item["metrics"]["current_miou"],
            item["metrics"]["ghost_rate"],
            -item["metrics"]["background_f5_cm"],
            item["metrics"]["recovery_frames"],
            item["config_sha256"],
        ),
    )
    if selection.get("selected_config_sha256") != winner["config_sha256"]:
        raise ValueError("selected configuration is not the predeclared lexicographic winner")
    return winner, validated, target_binding, arrays_binding


def _validate_declared_freeze_bindings(
    selection: Mapping[str, Any],
    *,
    repo_root: Path,
    expected: Mapping[str, Mapping[str, Path]],
) -> dict[str, dict[str, dict[str, Any]]]:
    raw = selection.get("freeze_bindings")
    if not isinstance(raw, dict) or set(raw) != {"shared", "scenes"}:
        raise ValueError("selection freeze_bindings are incomplete")
    raw_shared = raw.get("shared")
    raw_scenes = raw.get("scenes")
    if not isinstance(raw_shared, dict) or not isinstance(raw_scenes, dict):
        raise ValueError("selection freeze_bindings are invalid")
    if set(raw_shared) != set(expected["shared"]) or set(raw_scenes) != set(SCENES):
        raise ValueError("selection freeze_bindings do not cover exact artifacts")
    result: dict[str, dict[str, dict[str, Any]]] = {"shared": {}, "scenes": {}}
    for role, path in expected["shared"].items():
        result["shared"][role] = _verify_binding(
            raw_shared.get(role),
            repo_root=repo_root,
            role=role.replace("_", " "),
            expected_path=path,
        )
    for scene in SCENES:
        declared = raw_scenes.get(scene)
        scene_expected = expected[scene]
        if not isinstance(declared, dict) or set(declared) != set(scene_expected):
            raise ValueError(f"{scene} freeze bindings do not cover exact artifacts")
        result["scenes"][scene] = {}
        for role, path in scene_expected.items():
            result["scenes"][scene][role] = _verify_binding(
                declared.get(role),
                repo_root=repo_root,
                role=f"{scene} {role.replace('_', ' ')}",
                expected_path=path,
            )
    return result


def _expected_binding_paths(
    configs: Mapping[str, Mapping[str, Any]], repo_root: Path
) -> dict[str, Mapping[str, Path]]:
    input_paths = {
        _resolve_path(config.get("input_manifest"), repo_root, f"{scene} input manifest")
        for scene, config in configs.items()
    }
    if len(input_paths) != 1:
        raise ValueError("Apartment and Office must share one input manifest")
    input_manifest = next(iter(input_paths))
    result: dict[str, Mapping[str, Path]] = {
        "shared": {
            "input_manifest": input_manifest,
            "source_manifest": repo_root / "configs/evaluation/manifests/tesse_cd.json",
            "schedule": _resolve_path(
                configs["apartment"].get("schedule_manifest"), repo_root, "schedule"
            ),
            "camera": repo_root / "configs/evaluation/manifests/oviv2_tesse_cd_cache.json",
            "alias_map": repo_root
            / "configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml",
            "evaluator": repo_root / "scripts/evaluation/evaluate_tesse_cd_common_v2.py",
            "common_finalizer": repo_root
            / "scripts/evaluation/finalize_tesse_common_v2.py",
            "official_finalizer": repo_root / "scripts/evaluation/finalize_tesse_t2.py",
        }
    }
    for scene, config in configs.items():
        result[scene] = {
            "database": Path("."),
            "timestamps": Path("."),
            "trajectory": Path("."),
            "export_manifest": _resolve_path(
                config.get("export_manifest"), repo_root, f"{scene} export manifest"
            ),
            "frontend_manifest": _resolve_path(
                config.get("frontend_manifest"), repo_root, f"{scene} frontend manifest"
            ),
            "dense_manifest": _resolve_path(
                config.get("dense_manifest"), repo_root, f"{scene} dense manifest"
            ),
            "vocabulary_json": _resolve_path(
                config.get("vocabulary_json"), repo_root, f"{scene} vocabulary JSON"
            ),
            "vocabulary_txt": _resolve_path(
                config.get("vocabulary_txt"), repo_root, f"{scene} vocabulary TXT"
            ),
        }
    return result


def _validate_input_manifest_and_bindings(
    configs: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _expected_binding_paths(configs, repo_root)
    input_path = paths["shared"]["input_manifest"]
    input_manifest = _load_json(input_path, "runner input manifest")
    if (
        input_manifest.get("schema_version") != 1
        or input_manifest.get("dataset") != "TESSE-CD"
        or input_manifest.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
    ):
        raise ValueError("runner input manifest identity mismatch")
    scenes = input_manifest.get("scenes")
    if not isinstance(scenes, dict) or set(scenes) != set(SCENES):
        raise ValueError("runner input manifest scene set mismatch")
    source_record = input_manifest.get("source_manifest")
    schedule_record = input_manifest.get("schedule_manifest")
    camera_record = input_manifest.get("camera")
    source = _verify_binding(
        source_record, repo_root=repo_root, role="source manifest"
    )
    schedule = _verify_binding(
        schedule_record,
        repo_root=repo_root,
        role="schedule",
        expected_path=paths["shared"]["schedule"],
    )
    camera = _verify_binding(camera_record, repo_root=repo_root, role="camera")
    paths["shared"] = {**paths["shared"], "source_manifest": Path(source["path"]), "camera": Path(camera["path"])}

    scene_records: dict[str, Any] = {}
    for scene in SCENES:
        config = configs[scene]
        locked = scenes[scene]
        if not isinstance(locked, dict):
            raise ValueError(f"{scene} runner input lock must be an object")
        if (
            type(locked.get("frame_count")) is not int
            or locked["frame_count"] != config.get("frame_count")
            or _resolve_path(locked.get("root"), repo_root, f"{scene} RGB-D root")
            != _resolve_path(config.get("dataset_root"), repo_root, f"{scene} dataset root")
        ):
            raise ValueError(f"{scene} RGB-D source identity mismatch")
        export = _verify_binding(
            locked.get("export_manifest"),
            repo_root=repo_root,
            role=f"{scene} export manifest",
            expected_path=paths[scene]["export_manifest"],
        )
        export_payload = _load_json(Path(export["path"]), f"{scene} export manifest")
        database_path = _resolve_path(
            export_payload.get("source_database"), repo_root, f"{scene} source database"
        )
        _require_regular_file(database_path, f"{scene} source database")
        database_sha256 = _sha256(database_path)
        if (
            export_payload.get("dataset") != "TESSE-CD"
            or export_payload.get("scene") != scene
            or export_payload.get("frame_count") != locked["frame_count"]
            or not _is_sha256(export_payload.get("combined_output_sha256"))
            or type(export_payload.get("file_hash_count")) is not int
            or export_payload.get("source_database_sha256") != database_sha256
            or locked.get("source_database_sha256") != database_sha256
            or locked["export_manifest"].get("combined_output_sha256")
            != export_payload["combined_output_sha256"]
            or locked["export_manifest"].get("file_hash_count")
            != export_payload["file_hash_count"]
        ):
            raise ValueError(f"{scene} RGB-D export or source database hash mismatch")
        export_source = _resolve_path(
            export_payload.get("source_manifest"), repo_root, f"{scene} export source manifest"
        )
        _require_regular_file(export_source, f"{scene} export source manifest")
        if _sha256(export_source) != source["sha256"]:
            raise ValueError(f"{scene} export source manifest hash mismatch")
        timestamps = _verify_binding(
            locked.get("timestamps"), repo_root=repo_root, role=f"{scene} timestamps"
        )
        trajectory = _verify_binding(
            locked.get("trajectory"), repo_root=repo_root, role=f"{scene} trajectory"
        )
        vocabulary = locked.get("vocabulary")
        if not isinstance(vocabulary, dict):
            raise ValueError(f"{scene} vocabulary lock is missing")
        vocab_json_binding = {
            "path": vocabulary.get("json_path"),
            "sha256": vocabulary.get("json_sha256"),
        }
        vocab_txt_binding = {
            "path": vocabulary.get("txt_path"),
            "sha256": vocabulary.get("txt_sha256"),
        }
        vocab_json = _verify_binding(
            vocab_json_binding,
            repo_root=repo_root,
            role=f"{scene} vocabulary JSON",
            expected_path=paths[scene]["vocabulary_json"],
        )
        vocab_txt = _verify_binding(
            vocab_txt_binding,
            repo_root=repo_root,
            role=f"{scene} vocabulary TXT",
            expected_path=paths[scene]["vocabulary_txt"],
        )
        paths[scene] = {
            **paths[scene],
            "database": database_path,
            "timestamps": Path(timestamps["path"]),
            "trajectory": Path(trajectory["path"]),
        }
        scene_records[scene] = {
            "rgbd": {
                "root": str(_resolve_path(locked["root"], repo_root, f"{scene} root")),
                "frame_count": locked["frame_count"],
                "export_manifest": export,
                "combined_output_sha256": export_payload["combined_output_sha256"],
                "file_hash_count": export_payload["file_hash_count"],
                "source_database": {
                    "path": str(database_path),
                    "sha256": database_sha256,
                    "byte_count": database_path.stat().st_size,
                },
                "timestamps": timestamps,
                "trajectory": trajectory,
                "camera": camera,
            },
            "vocabulary": {"json": vocab_json, "txt": vocab_txt},
        }

    declared = _validate_declared_freeze_bindings(
        selection, repo_root=repo_root, expected=paths
    )
    return (
        {"input_manifest": declared["shared"]["input_manifest"], "source": source, "schedule": schedule, "camera": camera},
        scene_records,
        declared,
    )


def _validate_cache_manifest(
    scene: str,
    config: Mapping[str, Any],
    scene_binding: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frontend_path = _resolve_path(config.get("frontend_manifest"), repo_root, "frontend manifest")
    dense_path = _resolve_path(config.get("dense_manifest"), repo_root, "dense manifest")
    frontend_dir = _resolve_path(
        config.get("frontend_cache_dir"), repo_root, "frontend cache directory"
    )
    dense_dir = _resolve_path(
        config.get("dense_cache_dir"), repo_root, "dense cache directory"
    )
    if frontend_path.parent != frontend_dir or dense_path.parent != dense_dir:
        raise ValueError(f"{scene} cache directory does not own its declared manifest")
    frontend = _load_json(frontend_path, f"{scene} frontend manifest")
    dense = _load_json(dense_path, f"{scene} dense manifest")
    if (
        frontend.get("method") != "OVIV2"
        or frontend.get("dataset") != "TESSE-CD"
        or frontend.get("scene") != scene
        or frontend.get("frame_count") != config.get("frame_count")
        or not _is_sha256(frontend.get("algorithm_hash"))
        or not _is_sha256(frontend.get("cache_prefix_sha256"))
        or not isinstance(frontend.get("feature_model_id"), str)
        or not isinstance(frontend.get("provenance_sha256"), dict)
        or any(not _is_sha256(value) for value in frontend["provenance_sha256"].values())
    ):
        raise ValueError(f"{scene} frontend cache manifest is invalid")
    provenance = dense.get("provenance")
    if (
        dense.get("method") != "OVIV2-dense-semantic-cache"
        or dense.get("scene") != scene
        or dense.get("frame_count") != config.get("frame_count")
        or not isinstance(provenance, dict)
        or not isinstance(provenance.get("model_id"), str)
        or not isinstance(provenance.get("language_model_id"), str)
        or not isinstance(provenance.get("language_model_revision"), str)
        or any(
            not _is_sha256(provenance.get(key))
            for key in (
                "model_sha256",
                "auxiliary_model_sha256",
                "language_model_sha256",
                "vocabulary_sha256",
                "prompt_sha256",
                "inference_config_sha256",
                "cache_prefix_sha256",
            )
        )
    ):
        raise ValueError(f"{scene} dense cache manifest is invalid")
    cache = {
        "frontend_manifest": scene_binding["frontend_manifest"],
        "frontend_algorithm_sha256": frontend["algorithm_hash"],
        "frontend_cache_prefix_sha256": frontend["cache_prefix_sha256"],
        "dense_manifest": scene_binding["dense_manifest"],
        "dense_cache_prefix_sha256": provenance["cache_prefix_sha256"],
    }
    frontend_models = {
        "feature_model_id": frontend["feature_model_id"],
        "provenance_sha256": dict(sorted(frontend["provenance_sha256"].items())),
    }
    dense_models = {
        key: provenance[key]
        for key in (
            "model_id",
            "model_sha256",
            "auxiliary_model_sha256",
            "language_model_id",
            "language_model_revision",
            "language_model_sha256",
            "vocabulary_sha256",
            "prompt_sha256",
            "inference_config_sha256",
        )
    }
    return cache, frontend_models, dense_models


def _preflight_outputs(args: argparse.Namespace) -> None:
    destinations = (
        args.output_apartment_config.resolve(),
        args.output_office_config.resolve(),
        args.output_manifest.resolve(),
    )
    if len(set(destinations)) != 3:
        raise ValueError("freeze output paths must be distinct")
    for path in destinations:
        if os.path.lexists(path):
            raise FileExistsError(f"freeze output already exists: {path}")
    output_root = args.output_manifest.resolve().parent
    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing:
            raise FileExistsError(f"prior run output exists under {output_root}")


def _office_metric_sources(config: Mapping[str, Any], output_root: Path) -> list[str]:
    found = sorted(key for key in config if key.lower() in _OFFICE_METRIC_KEYS)
    if output_root.exists():
        found.extend(
            str(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file() and "metric" in path.name.lower()
        )
    return found


def _mapping_commands(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    commands: list[str] = []
    roots: dict[str, str] = {}
    root = args.output_manifest.resolve().parent
    config_paths = {
        "apartment": args.output_apartment_config.resolve(),
        "office": args.output_office_config.resolve(),
    }
    for scene in SCENES:
        for repeat in (1, 2):
            output = root / scene / f"run{repeat}"
            roots[f"{scene}_run{repeat}"] = str(output)
            commands.append(
                shlex.join(
                    [
                        "python",
                        "scripts/evaluation/run_oviv2_tesse_cd.py",
                        "--config",
                        str(config_paths[scene]),
                        "--output",
                        str(output),
                    ]
                )
            )
    return commands, roots


def _link_no_replace(source: Path, destination: Path) -> None:
    os.link(source, destination)


def _publish_json_transaction(
    publications: Sequence[tuple[Path, bytes]], *, empty_root_guard: Path
) -> None:
    temporary: list[Path] = []
    published: list[tuple[Path, int, int]] = []
    try:
        for destination, content in publications:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_path = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".freeze-tmp",
            )
            temp_path = Path(raw_path)
            temporary.append(temp_path)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for (destination, _), temp_path in zip(publications, temporary, strict=True):
            expected_root_entries = {
                path.resolve()
                for path in temporary
                if path.exists() and path.parent.resolve() == empty_root_guard.resolve()
            }
            observed_root_entries = {
                path.resolve() for path in empty_root_guard.iterdir()
            }
            if observed_root_entries != expected_root_entries:
                raise FileExistsError(
                    f"prior run output appeared under {empty_root_guard}"
                )
            _link_no_replace(temp_path, destination)
            status = destination.stat(follow_symlinks=False)
            published.append((destination, status.st_dev, status.st_ino))
            temp_path.unlink()
        for parent in {destination.parent for destination, _ in publications}:
            descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for destination, device, inode in reversed(published):
            try:
                status = destination.stat(follow_symlinks=False)
                if status.st_dev == device and status.st_ino == inode:
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for path in temporary:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def freeze(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    repository_state: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = (repo_root or REPO_ROOT).resolve()
    _preflight_outputs(args)
    repository = _validate_repository(
        repository_state if repository_state is not None else _inspect_repository(root)
    )
    apartment_path = args.apartment_config.resolve()
    office_path = args.office_config.resolve()
    selection_path = args.apartment_selection.resolve()
    apartment = _load_json(apartment_path, "Apartment config")
    office = _load_json(office_path, "Office config")
    selection = _load_json(selection_path, "Apartment selection")
    office_metrics = _office_metric_sources(office, args.output_manifest.resolve().parent)
    if office_metrics:
        raise ValueError(f"Office metric source exists before freeze: {office_metrics}")
    _validate_source_configs(apartment, office)

    shared, scene_records, declared = _validate_input_manifest_and_bindings(
        {"apartment": apartment, "office": office}, selection, repo_root=root
    )
    winner, candidates, target_binding, target_arrays = _validate_selection(
        selection_path,
        selection,
        apartment_config=apartment,
        repo_root=root,
        schedule_sha256=shared["schedule"]["sha256"],
    )
    frozen_configs: dict[str, dict[str, Any]] = {}
    for scene, source in (("apartment", apartment), ("office", office)):
        frozen = dict(source)
        frozen.update(winner["parameters"])
        frozen["algorithm_hash"] = _algorithm_hash(frozen)
        frozen_configs[scene] = frozen
    if frozen_configs["apartment"]["algorithm_hash"] != frozen_configs["office"]["algorithm_hash"]:
        raise ValueError("frozen normalized algorithm hashes differ across scenes")
    if _json_hash(frozen_configs["apartment"]) != winner["config_sha256"]:
        raise ValueError("selected candidate does not reproduce the frozen Apartment config")

    models: dict[str, dict[str, Any]] = {"frontend": {}, "dense": {}}
    for scene, config in frozen_configs.items():
        cache, frontend_models, dense_models = _validate_cache_manifest(
            scene, config, declared["scenes"][scene], root
        )
        scene_records[scene]["cache"] = cache
        models["frontend"][scene] = frontend_models
        models["dense"][scene] = dense_models
    if models["frontend"]["apartment"] != models["frontend"]["office"]:
        raise ValueError("Apartment and Office must use the same frozen model provenance")
    dense_shared_fields = (
        "model_id",
        "model_sha256",
        "auxiliary_model_sha256",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
        "inference_config_sha256",
    )
    if any(
        models["dense"]["apartment"][field]
        != models["dense"]["office"][field]
        for field in dense_shared_fields
    ):
        raise ValueError("Apartment and Office must use the same frozen model weights")

    apartment_bytes = _canonical_json_bytes(frozen_configs["apartment"])
    office_bytes = _canonical_json_bytes(frozen_configs["office"])
    mapping_commands, output_roots = _mapping_commands(args)
    shared_bindings = {
        "input_manifest": shared["input_manifest"],
        "source_manifest": shared["source"],
        "schedule": shared["schedule"],
        "camera": shared["camera"],
        "common_target_manifest": target_binding,
        "common_target_arrays": target_arrays,
        "alias_map": declared["shared"]["alias_map"],
        "evaluator": declared["shared"]["evaluator"],
        "finalizers": {
            "common_v2": declared["shared"]["common_finalizer"],
            "official_t2": declared["shared"]["official_finalizer"],
        },
    }
    manifest = {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v1",
        "status": "FROZEN",
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "repository": repository,
        "algorithm": {
            "sha256": frozen_configs["apartment"]["algorithm_hash"],
            "normalized_config": _algorithm_config(frozen_configs["apartment"]),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "byte_count": selection_path.stat().st_size,
            "candidate_count": len(candidates),
            "selected_config_sha256": winner["config_sha256"],
            "selected_parameters": winner["parameters"],
            "selection_rule": [
                "maximize_current_miou",
                "minimize_ghost_rate",
                "maximize_background_f5_cm",
                "minimize_recovery_frames",
                "minimize_config_sha256",
            ],
            "candidates": candidates,
        },
        "scenes": {
            "apartment": {
                **scene_records["apartment"],
                "source_config": {
                    "path": str(apartment_path),
                    "sha256": _sha256(apartment_path),
                    "byte_count": apartment_path.stat().st_size,
                },
                "frozen_config": {
                    "path": str(args.output_apartment_config.resolve()),
                    "sha256": _sha256_bytes(apartment_bytes),
                    "byte_count": len(apartment_bytes),
                },
            },
            "office": {
                **scene_records["office"],
                "source_config": {
                    "path": str(office_path),
                    "sha256": _sha256(office_path),
                    "byte_count": office_path.stat().st_size,
                },
                "frozen_config": {
                    "path": str(args.output_office_config.resolve()),
                    "sha256": _sha256_bytes(office_bytes),
                    "byte_count": len(office_bytes),
                },
            },
        },
        "shared_bindings": shared_bindings,
        "models": models,
        "environment": dict(environment) if environment is not None else _default_environment(),
        "commands": {"mapping": mapping_commands},
        "output_roots": output_roots,
        "office_pre_freeze_audit": {
            "metric_sources_found": [],
            "output_root_was_empty": True,
        },
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    _publish_json_transaction(
        (
            (args.output_apartment_config.resolve(), apartment_bytes),
            (args.output_office_config.resolve(), office_bytes),
            (args.output_manifest.resolve(), manifest_bytes),
        ),
        empty_root_guard=args.output_manifest.resolve().parent,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apartment-selection", type=Path, required=True)
    parser.add_argument("--apartment-config", type=Path, required=True)
    parser.add_argument("--office-config", type=Path, required=True)
    parser.add_argument("--output-apartment-config", type=Path, required=True)
    parser.add_argument("--output-office-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = freeze(args)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"FROZEN {manifest['freeze_id']} "
        f"algorithm={manifest['algorithm']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
