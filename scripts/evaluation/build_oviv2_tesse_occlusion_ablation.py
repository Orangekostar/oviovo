#!/usr/bin/env python3
"""Build the parent-frozen missing-as-absence TESSE-CD occlusion ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


SCENES = ("apartment", "office")
STAGE3_LINEAGE_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
SCENE_CONFIG_FIELDS = frozenset(
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
        "occlusion_target_manifest",
        "vocabulary_json",
        "vocabulary_txt",
    }
)
MUTATED_FIELDS = frozenset({"algorithm_hash", "missing_observation_policy"})
_FORBIDDEN_MARKERS = ("route3", "surface-observation", "stage4", "scannet200")
_SHA256_HEX = frozenset("0123456789abcdef")
_GIT_HEX = _SHA256_HEX


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX
    )


def _is_git_oid(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _GIT_HEX


def _complete_binding(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256", "byte_count"}
        and isinstance(value.get("path"), str)
        and value["path"]
        and _is_sha256(value.get("sha256"))
        and type(value.get("byte_count")) is int
        and value["byte_count"] >= 0
    )


def _contains_complete_binding(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and {"path", "sha256", "byte_count"} <= set(value)
        and isinstance(value.get("path"), str)
        and value["path"]
        and _is_sha256(value.get("sha256"))
        and type(value.get("byte_count")) is int
        and value["byte_count"] >= 0
    )


def _valid_repository(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "commit",
        "parents",
        "tree",
        "commit_time_utc",
        "clean",
        "stage3_lineage_commit",
        "stage3_is_ancestor",
    }:
        return False
    parents = value.get("parents")
    return bool(
        _is_git_oid(value.get("commit"))
        and isinstance(parents, list)
        and all(_is_git_oid(parent) for parent in parents)
        and _is_git_oid(value.get("tree"))
        and isinstance(value.get("commit_time_utc"), str)
        and value["commit_time_utc"]
        and value.get("clean") is True
        and value.get("stage3_lineage_commit") == STAGE3_LINEAGE_COMMIT
        and value.get("stage3_is_ancestor") is True
    )


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        try:
            status = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{role} must not contain a symlink component: {path}")


def _read_regular_file(path: Path, role: str) -> bytes:
    _reject_symlink_components(path, role)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{role} must be a readable regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{role} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{role} changed while being read: {path}") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or _fingerprint(before) != _fingerprint(after)
        or _fingerprint(after) != _fingerprint(current)
    ):
        raise ValueError(f"{role} changed while being read: {path}")
    return b"".join(chunks)


def _load_json(path: Path, role: str, *, require_canonical: bool = True) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file(path, role)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object")
    if require_canonical and raw != _canonical_bytes(payload):
        raise ValueError(f"{role} must use canonical JSON encoding")
    return payload, raw


def _resolve_binding_path(raw: object, parent: Path, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{role} path is invalid")
    path = Path(raw)
    return path if path.is_absolute() else parent / path


def _load_bound_config(
    record: object,
    *,
    manifest_path: Path,
    role: str,
) -> tuple[dict[str, Any], bytes, Path]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{role} binding is invalid")
    path = _resolve_binding_path(record["path"], manifest_path.parent, role)
    config, raw = _load_json(path, role)
    if not (
        _is_sha256(record["sha256"])
        and record["sha256"] == _sha256_bytes(raw)
        and type(record["byte_count"]) is int
        and record["byte_count"] == len(raw)
    ):
        raise ValueError(f"{role} hash binding mismatch")
    return config, raw, path


def canonical_algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in sorted(config)
        if key not in SCENE_CONFIG_FIELDS
    }


def canonical_algorithm_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_algorithm_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _find_forbidden(value: object, path: str = "config") -> str | None:
    if isinstance(value, Mapping):
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
        if any(marker in lowered for marker in _FORBIDDEN_MARKERS):
            return path
    return None


def _validate_parent(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    expected_fields = {
        "schema_version",
        "freeze_id",
        "status",
        "method",
        "dataset",
        "repository",
        "algorithm",
        "selection",
        "scenes",
        "shared_bindings",
        "models",
        "environment",
        "commands",
        "output_roots",
        "office_pre_freeze_audit",
        "preparation",
    }
    if set(manifest) != expected_fields:
        raise ValueError("parent freeze fields do not match the FROZEN v1 contract")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("freeze_id") == "oviv2-tessecd-v1"
        and manifest.get("status") == "FROZEN"
        and manifest.get("method") == "OVIV2"
        and manifest.get("dataset") == "TESSE-CD"
    ):
        raise ValueError("parent freeze must be the FROZEN oviv2-tessecd-v1 manifest")
    repository = manifest.get("repository")
    if not _valid_repository(repository):
        raise ValueError("parent freeze repository identity is invalid")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "path",
        "sha256",
        "byte_count",
        "candidate_count",
        "selected_config_sha256",
        "selected_parameters",
        "selection_rule",
        "candidates",
    } or not (
        _complete_binding(
            {key: selection.get(key) for key in ("path", "sha256", "byte_count")}
        )
        and selection.get("candidate_count") == 18
        and _is_sha256(selection.get("selected_config_sha256"))
        and isinstance(selection.get("selected_parameters"), Mapping)
        and bool(selection["selected_parameters"])
        and isinstance(selection.get("selection_rule"), list)
        and bool(selection["selection_rule"])
        and isinstance(selection.get("candidates"), list)
        and len(selection["candidates"]) == 18
    ):
        raise ValueError("parent freeze selection identity is invalid")
    shared = manifest.get("shared_bindings")
    shared_fields = {
        "input_manifest",
        "source_manifest",
        "schedule",
        "camera",
        "common_target_manifest",
        "common_target_arrays",
        "alias_map",
        "evaluator",
        "finalizers",
    }
    if not isinstance(shared, Mapping) or set(shared) != shared_fields:
        raise ValueError("parent freeze shared bindings are invalid")
    if any(
        not _contains_complete_binding(shared[field])
        for field in shared_fields - {"finalizers"}
    ):
        raise ValueError("parent freeze shared bindings are incomplete")
    finalizers = shared.get("finalizers")
    if not isinstance(finalizers, Mapping) or set(finalizers) != {
        "common_v2",
        "official_t2",
    } or any(not _complete_binding(binding) for binding in finalizers.values()):
        raise ValueError("parent freeze finalizer bindings are invalid")
    models = manifest.get("models")
    if not isinstance(models, Mapping) or set(models) != {"frontend", "dense"}:
        raise ValueError("parent freeze model identity is invalid")
    if any(
        not isinstance(models[role], Mapping)
        or set(models[role]) != set(SCENES)
        or any(
            not isinstance(models[role][scene], Mapping) or not models[role][scene]
            for scene in SCENES
        )
        for role in ("frontend", "dense")
    ):
        raise ValueError("parent freeze model identity is incomplete")
    preparation = manifest.get("preparation")
    if not isinstance(preparation, Mapping) or set(preparation) != {
        "manifest",
        "repository",
    } or not (
        _complete_binding(preparation.get("manifest"))
        and _valid_repository(preparation.get("repository"))
    ):
        raise ValueError("parent freeze preparation identity is invalid")
    for field in (
        "environment",
        "commands",
        "output_roots",
        "office_pre_freeze_audit",
    ):
        if not isinstance(manifest.get(field), Mapping) or not manifest[field]:
            raise ValueError(f"parent freeze {field} is invalid")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(SCENES):
        raise ValueError("parent freeze must contain Apartment and Office exactly")

    configs: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        scene_record = scenes[scene]
        if not isinstance(scene_record, Mapping) or set(scene_record) != {
            "rgbd",
            "vocabulary",
            "cache",
            "source_config",
            "frozen_config",
        }:
            raise ValueError(f"parent {scene} scene record is invalid")
        if any(
            not isinstance(scene_record[field], Mapping) or not scene_record[field]
            for field in ("rgbd", "vocabulary", "cache")
        ) or not (
            _complete_binding(scene_record.get("source_config"))
            and _complete_binding(scene_record.get("frozen_config"))
            and {"frontend_manifest", "dense_manifest"}
            <= set(scene_record["cache"])
        ):
            raise ValueError(f"parent {scene} scene record is incomplete")
        config, _, _ = _load_bound_config(
            scene_record.get("frozen_config"),
            manifest_path=manifest_path,
            role=f"parent {scene} frozen config",
        )
        forbidden = _find_forbidden(config)
        if forbidden is not None:
            raise ValueError(f"forbidden non-Stage3 reference at {forbidden}")
        if not (
            config.get("schema_version") == 1
            and config.get("dataset") == "TESSE-CD"
            and config.get("method_id") == "OVIV2"
            and config.get("scene") == scene
            and config.get("stage3_lineage_commit") == STAGE3_LINEAGE_COMMIT
            and config.get("missing_observation_policy") == "signed_depth"
        ):
            raise ValueError(f"parent {scene} config must be signed_depth Stage3 OVIV2")
        if config.get("algorithm_hash") != canonical_algorithm_hash(config):
            raise ValueError(f"parent {scene} config algorithm_hash is stale")
        configs[scene] = config

    apartment = configs["apartment"]
    office = configs["office"]
    if set(apartment) != set(office):
        raise ValueError("parent scene configs expose different fields")
    differences = {key for key in apartment if apartment[key] != office[key]}
    if not differences <= SCENE_CONFIG_FIELDS - {"algorithm_hash"}:
        raise ValueError("parent scene configs differ outside frozen scene inputs")
    if canonical_algorithm_config(apartment) != canonical_algorithm_config(office):
        raise ValueError("parent scene algorithm configurations differ")
    for field in (
        "algorithm_hash",
        "occlusion_target_manifest_sha256",
        "evaluation_checkpoint_frames_sha256",
    ):
        if apartment.get(field) != office.get(field):
            raise ValueError(f"parent scene {field} values differ")
    if not all(
        _is_sha256(configs["apartment"].get(field))
        for field in (
            "occlusion_target_manifest_sha256",
            "evaluation_checkpoint_frames_sha256",
        )
    ):
        raise ValueError("parent occlusion target or checkpoint plan hash is invalid")
    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, Mapping) or not (
        algorithm.get("sha256") == apartment["algorithm_hash"]
        and algorithm.get("normalized_config") == canonical_algorithm_config(apartment)
    ):
        raise ValueError("parent freeze algorithm binding mismatch")
    return configs


def _binding(raw: bytes, *, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "path_base": "manifest",
        "sha256": _sha256_bytes(raw),
        "byte_count": len(raw),
    }


def _write_exclusive(path: Path, raw: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_ablation(*, parent_freeze: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate the signed parent and publish the sole allowed policy mutation."""
    parent_path = Path(parent_freeze)
    output = Path(output_dir)
    if os.path.lexists(output):
        raise ValueError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")

    parent, parent_raw = _load_json(parent_path, "parent freeze manifest")
    parent_configs = _validate_parent(parent, manifest_path=parent_path)
    ablation_configs: dict[str, dict[str, Any]] = {}
    config_bytes: dict[str, bytes] = {}
    for scene, signed_config in parent_configs.items():
        ablation = dict(signed_config)
        ablation["missing_observation_policy"] = "missing_as_absence"
        ablation["algorithm_hash"] = canonical_algorithm_hash(ablation)
        changed = {
            key
            for key in signed_config
            if signed_config.get(key) != ablation.get(key)
        }
        if set(ablation) != set(signed_config) or changed != MUTATED_FIELDS:
            raise ValueError("ablation mutation escaped the two-field allowlist")
        ablation_configs[scene] = ablation
        config_bytes[scene] = _canonical_bytes(ablation)
    ablation_hashes = {
        config["algorithm_hash"] for config in ablation_configs.values()
    }
    if len(ablation_hashes) != 1:
        raise ValueError("ablation algorithm hashes differ across scenes")
    ablation_hash = next(iter(ablation_hashes))
    parent_hash = parent_configs["apartment"]["algorithm_hash"]
    if ablation_hash == parent_hash:
        raise ValueError("ablation algorithm hash did not change")

    manifest = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_occlusion_ablation_v1",
        "status": "FROZEN",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "ablation_id": "missing_as_absence",
        "parent_freeze": {
            "sha256": _sha256_bytes(parent_raw),
            "byte_count": len(parent_raw),
            "freeze_id": parent["freeze_id"],
            "repository_commit": parent["repository"]["commit"],
            "algorithm_hash": parent_hash,
        },
        "mutation": {
            "only_changed_fields": sorted(MUTATED_FIELDS),
            "missing_observation_policy": {
                "from": "signed_depth",
                "to": "missing_as_absence",
            },
        },
        "algorithm": {
            "parent_sha256": parent_hash,
            "ablation_sha256": ablation_hash,
            "normalized_config": canonical_algorithm_config(
                ablation_configs["apartment"]
            ),
        },
        "occlusion_target_manifest_sha256": parent_configs["apartment"][
            "occlusion_target_manifest_sha256"
        ],
        "evaluation_checkpoint_frames_sha256": parent_configs["apartment"][
            "evaluation_checkpoint_frames_sha256"
        ],
        "frozen_identity": {
            "selection_sha256": _json_hash(parent.get("selection")),
            "shared_bindings_sha256": _json_hash(parent.get("shared_bindings")),
            "models_sha256": _json_hash(parent.get("models")),
        },
        "scenes": {
            scene: {
                "parent_config": parent["scenes"][scene]["frozen_config"],
                "ablation_config": _binding(
                    config_bytes[scene], path=f"{scene}.json"
                ),
                "invariant_config_sha256": _json_hash(
                    {
                        key: value
                        for key, value in ablation_configs[scene].items()
                        if key not in MUTATED_FIELDS
                    }
                ),
            }
            for scene in SCENES
        },
    }
    manifest_bytes = _canonical_bytes(manifest)
    try:
        os.mkdir(output, 0o755)
    except FileExistsError as error:
        raise ValueError(f"output directory already exists: {output}") from error
    for scene in SCENES:
        _write_exclusive(output / f"{scene}.json", config_bytes[scene])
    _write_exclusive(output / "manifest.json", manifest_bytes)
    descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_ablation(
        parent_freeze=args.parent_freeze,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "parent_freeze_sha256": manifest["parent_freeze"]["sha256"],
                "algorithm_hash": manifest["algorithm"]["ablation_sha256"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
