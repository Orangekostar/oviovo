#!/usr/bin/env python3
"""Build strict OVIV2 TESSE-CD provenance for ``finalize_tesse_t2.py``."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.finalize_tesse_t2 import (  # noqa: E402
    _atomic_json_no_replace,
    _open_directory_no_symlinks,
    _stable_regular_file,
    _strict_json_object,
    build_scene_evidence,
)
from scripts.evaluation.canonicalize_tesse_common_v2_summary import (  # noqa: E402
    _temporal_identity_projection,
)
from scripts.evaluation.prepare_temporal_khronos_bridge import (  # noqa: E402
    validate_temporal_bridge_manifest,
)
from scripts.evaluation.run_khronos_official_eval import (  # noqa: E402
    validate_khronos_run_status,
)
from src.evaluation.baselines.tesse_cd import (  # noqa: E402
    summarize_khronos_official_metrics,
)


SCENES = ("apartment", "office")
RUN_KEYS = tuple(
    f"{scene}_run{repeat}" for scene in SCENES for repeat in (1, 2)
)
CLEAN_DIRTY_STATE_DIGEST = hashlib.sha256(b"").hexdigest()
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
STAGE3_LINEAGE_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
_WEIGHT_FIELDS = {
    "frontend.clip_model": ("frontend", "clip_model"),
    "frontend.mobile_sam_model": ("frontend", "mobile_sam_model"),
    "frontend.yolo_clip_model": ("frontend", "yolo_clip_model"),
    "frontend.yolo_model": ("frontend", "yolo_model"),
    "dense.auxiliary_model": ("dense", "auxiliary_model_sha256"),
    "dense.language_model": ("dense", "language_model_sha256"),
    "dense.model": ("dense", "model_sha256"),
}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    descriptor, _ = _open_directory_no_symlinks(absolute, label=label)
    os.close(descriptor)
    return absolute


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    absolute = _absolute(path)
    descriptor, _ = _open_directory_no_symlinks(absolute, label=label)
    try:
        status = os.fstat(descriptor)
        return status.st_dev, status.st_ino
    finally:
        os.close(descriptor)


def _file(
    path: Path, *, label: str, capture: bool = False
) -> tuple[bytes | None, dict[str, Any]]:
    absolute = _absolute(path)
    digest, byte_count, content = _stable_regular_file(
        absolute, label=label, capture=capture
    )
    if capture:
        assert content is not None
    return content, {
        "path": str(absolute),
        "sha256": digest,
        "byte_count": byte_count,
    }


def _json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    content, record = _file(path, label=label, capture=True)
    assert content is not None
    return _strict_json_object(content, label=label), record


def _declared_path(entry: Mapping[str, Any], *, base: Path, label: str) -> Path:
    if not isinstance(entry, Mapping) or type(entry.get("path")) is not str:
        raise ValueError(f"{label} requires a string path")
    raw = Path(entry["path"])
    if not raw.is_absolute() and ".." in raw.parts:
        raise ValueError(f"{label} path cannot contain parent traversal")
    return _absolute(raw if raw.is_absolute() else base / raw)


def _declared_record(
    entry: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    expected_path: Path | None = None,
    require_byte_count: bool = True,
) -> dict[str, Any]:
    path = _declared_path(entry, base=base, label=label)
    if expected_path is not None and path != _absolute(expected_path):
        raise ValueError(f"{label} path identity mismatch")
    _, observed = _file(path, label=label)
    declared_bytes = entry.get("byte_count")
    if entry.get("sha256") != observed["sha256"] or (
        (require_byte_count or declared_bytes is not None)
        and declared_bytes != observed["byte_count"]
    ):
        raise ValueError(f"{label} hash binding mismatch")
    return observed


def _named_record(path: Path, *, name: str, **fields: str) -> dict[str, Any]:
    _, record = _file(path, label=name)
    return {"name": name, **fields, **record}


def _repository_state() -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip().decode("ascii")
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "commit": commit,
        "dirty_state_digest": hashlib.sha256(dirty).hexdigest(),
    }


def _command(value: object, *, label: str) -> str:
    if not isinstance(value, list) or not value or any(
        type(item) is not str or not item for item in value
    ):
        raise ValueError(f"{label} must be a non-empty argv list")
    return shlex.join(value)


def _commands(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if type(value) is not str or not value:
            raise ValueError("provenance command must be a non-empty string")
        if value not in seen:
            seen.add(value)
            result.append(value)
    if not result:
        raise ValueError("provenance commands cannot be empty")
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


_FROZEN_RUN_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "freeze_id",
        "dataset",
        "method_id",
        "scene",
        "freeze_manifest",
        "repository",
        "config",
        "algorithm_hash",
        "missing_observation_policy",
        "input_bindings_sha256",
    }
)
_RUN_EXECUTION_FIELDS = frozenset(
    {
        "schema_version",
        "run_slot",
        "execution_id",
        "output_root",
        "root_device",
        "root_inode",
    }
)


def _content_only(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not (
        _is_sha256(record.get("sha256"))
        and type(record.get("byte_count")) is int
        and record["byte_count"] >= 0
    ):
        raise ValueError(f"{label} content binding is invalid")
    return {"sha256": record["sha256"], "byte_count": record["byte_count"]}


def _expected_frozen_run_identity(
    *,
    freeze: Mapping[str, Any],
    freeze_record: Mapping[str, Any],
    scene: str,
    config: Mapping[str, Any],
    config_record: Mapping[str, Any],
) -> dict[str, Any]:
    repository = _repository_binding(freeze.get("repository"), label="frozen")
    scenes = freeze.get("scenes")
    shared = freeze.get("shared_bindings")
    if not isinstance(scenes, Mapping) or not isinstance(shared, Mapping):
        raise ValueError("freeze input bindings are incomplete")
    scene_binding = scenes.get(scene)
    if not isinstance(scene_binding, Mapping):
        raise ValueError(f"frozen {scene} source binding is missing")
    identity_inputs = {
        "shared_bindings": dict(shared),
        "scene": dict(scene_binding),
    }
    algorithm = freeze.get("algorithm")
    if not isinstance(algorithm, Mapping) or not _is_sha256(
        algorithm.get("sha256")
    ):
        raise ValueError("frozen algorithm identity is invalid")
    policy = config.get("missing_observation_policy")
    if policy != "signed_depth":
        raise ValueError("frozen main run policy must be signed_depth")
    return {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": scene,
        "freeze_manifest": _content_only(
            freeze_record, label="freeze manifest"
        ),
        "repository": {
            "commit": repository["commit"],
            "tree": repository["tree"],
        },
        "config": _content_only(config_record, label=f"frozen {scene} config"),
        "algorithm_hash": algorithm["sha256"],
        "missing_observation_policy": policy,
        "input_bindings_sha256": hashlib.sha256(
            _canonical_json(identity_inputs)
        ).hexdigest(),
    }


def _validate_formal_run_fields(
    payload: Mapping[str, Any],
    *,
    expected_frozen: Mapping[str, Any],
    key: str,
    root: Path,
    label: str,
) -> dict[str, Any]:
    frozen = payload.get("frozen_run_identity")
    execution = payload.get("run_execution")
    if (
        not isinstance(frozen, Mapping)
        or set(frozen) != _FROZEN_RUN_IDENTITY_FIELDS
        or dict(frozen) != dict(expected_frozen)
    ):
        raise ValueError(f"{label} frozen run identity mismatch")
    if not isinstance(execution, Mapping) or set(execution) != _RUN_EXECUTION_FIELDS:
        raise ValueError(f"{label} run execution fields are invalid")
    status = root.stat()
    expected_base = {
        "schema_version": 1,
        "run_slot": key,
        "output_root": str(root),
        "root_device": status.st_dev,
        "root_inode": status.st_ino,
    }
    expected_execution = {
        **expected_base,
        "execution_id": hashlib.sha256(_canonical_json(expected_base)).hexdigest(),
    }
    if dict(execution) != expected_execution:
        raise ValueError(f"{label} run execution mismatch")
    return expected_execution


def _mapping_identity_projection(
    manifest: Mapping[str, Any],
    source_index: Mapping[str, Any],
    occlusion: Mapping[str, Any],
) -> tuple[str, str]:
    projected_occlusion = copy.deepcopy(dict(occlusion))
    projected_occlusion.pop("run_execution")
    occlusion_bytes = _canonical_json(projected_occlusion)
    projected_manifest = copy.deepcopy(dict(manifest))
    projected_manifest.pop("run_execution")
    declaration = projected_manifest.get("occlusion_checkpoint_index")
    if not isinstance(declaration, Mapping) or type(declaration.get("path")) is not str:
        raise ValueError("mapping occlusion index declaration is invalid")
    projected_manifest["occlusion_checkpoint_index"] = {
        "path": declaration["path"],
        "sha256": hashlib.sha256(occlusion_bytes).hexdigest(),
        "byte_count": len(occlusion_bytes),
    }
    projected_source_index = copy.deepcopy(dict(source_index))
    projected_source_index.pop("run_execution")
    return (
        hashlib.sha256(_canonical_json(projected_manifest)).hexdigest(),
        hashlib.sha256(_canonical_json(projected_source_index)).hexdigest(),
    )


def _revalidate_records(value: object) -> None:
    bindings: dict[Path, tuple[str, int]] = {}

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            if (
                type(item.get("path")) is str
                and _is_sha256(item.get("sha256"))
                and type(item.get("byte_count")) is int
            ):
                path = _absolute(item["path"])
                expected = (item["sha256"], item["byte_count"])
                previous = bindings.setdefault(path, expected)
                if previous != expected:
                    raise ValueError(f"conflicting provenance bindings for {path}")
                _, observed = _file(path, label=f"revalidated provenance input {path}")
                if expected != (observed["sha256"], observed["byte_count"]):
                    raise ValueError(
                        f"input changed after provenance capture: {path}"
                    )
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)


def _expected_weights(models: Mapping[str, Any]) -> dict[str, str]:
    sections: dict[str, Mapping[str, Any]] = {}
    for section in ("frontend", "dense"):
        by_scene = models.get(section)
        if not isinstance(by_scene, Mapping) or set(by_scene) != set(SCENES):
            raise ValueError(f"frozen {section} model bindings are incomplete")
        apartment = by_scene["apartment"]
        office = by_scene["office"]
        if not isinstance(apartment, Mapping) or apartment != office:
            raise ValueError(f"frozen {section} model bindings differ by scene")
        sections[section] = apartment
    frontend = sections["frontend"].get("provenance_sha256")
    if not isinstance(frontend, Mapping):
        raise ValueError("frozen frontend model hashes are missing")
    expected: dict[str, str] = {}
    for role, (section, field) in _WEIGHT_FIELDS.items():
        source = frontend if section == "frontend" else sections["dense"]
        digest = source.get(field)
        if not _is_sha256(digest):
            raise ValueError(f"frozen weight hash is invalid: {role}")
        expected[role] = digest
    return expected


def _repository_binding(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} repository identity is missing")
    parents = value.get("parents")
    if (
        not _is_commit(value.get("commit"))
        or not _is_commit(value.get("tree"))
        or not isinstance(parents, list)
        or any(not _is_commit(parent) for parent in parents)
        or type(value.get("commit_time_utc")) is not str
        or not value["commit_time_utc"]
        or value.get("clean") is not True
        or value.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
        or value.get("stage3_is_ancestor") is not True
    ):
        raise ValueError(f"{label} repository identity is invalid")
    return value


def _content_binding(
    entry: object, *, base: Path, label: str
) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{label} binding is missing")
    record = _declared_record(entry, base=base, label=label)
    return {
        "sha256": record["sha256"],
        "byte_count": record["byte_count"],
    }


def _freeze_source_bindings(
    freeze: Mapping[str, Any], *, freeze_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    required = {
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
    missing = sorted(required - set(freeze))
    if missing:
        raise ValueError(f"frozen manifest is missing required fields: {missing}")

    selection = freeze.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("frozen Apartment selection is missing")
    _content_binding(
        selection,
        base=freeze_path.parent,
        label="frozen Apartment selection",
    )
    candidates = selection.get("candidates")
    if (
        selection.get("candidate_count") != 18
        or not isinstance(candidates, list)
        or len(candidates) != 18
        or not isinstance(selection.get("selected_parameters"), Mapping)
        or not _is_sha256(selection.get("selected_config_sha256"))
    ):
        raise ValueError("frozen Apartment selection identity is invalid")

    audit = freeze.get("office_pre_freeze_audit")
    audit_scope = audit.get("scope") if isinstance(audit, Mapping) else None
    if (
        not isinstance(audit, Mapping)
        or audit.get("metric_sources_found") != []
        or audit.get("output_root_was_empty") is not False
        or audit.get("output_root_had_only_preparation") is not True
        or not isinstance(audit_scope, Mapping)
        or audit_scope.get("office_config_recursive") is not True
        or audit_scope.get("selection_scene") != "apartment"
        or _absolute(str(audit_scope.get("output_root", "")))
        != freeze_path.parent
    ):
        raise ValueError("Office pre-freeze audit is invalid")

    preparation = freeze.get("preparation")
    if not isinstance(preparation, Mapping):
        raise ValueError("frozen preparation evidence is missing")
    prepared_entry = preparation.get("manifest")
    if not isinstance(prepared_entry, Mapping):
        raise ValueError("frozen preparation manifest binding is missing")
    prepared_path = _declared_path(
        prepared_entry,
        base=freeze_path.parent,
        label="prepared freeze manifest",
    )
    _declared_record(
        prepared_entry,
        base=freeze_path.parent,
        label="prepared freeze manifest",
        expected_path=prepared_path,
    )
    prepared, _ = _json(prepared_path, label="prepared freeze manifest")
    prepared_repository = _repository_binding(
        preparation.get("repository"), label="prepared freeze"
    )
    if (
        prepared.get("status") != "PREPARED"
        or prepared.get("freeze_id") != freeze.get("freeze_id")
        or prepared.get("repository") != prepared_repository
        or prepared.get("algorithm") != freeze.get("algorithm")
        or prepared.get("selection") != selection
        or prepared.get("shared_bindings") != freeze.get("shared_bindings")
        or prepared.get("models") != freeze.get("models")
    ):
        raise ValueError("prepared freeze manifest does not match final inputs")
    prepared_scenes = prepared.get("scenes")
    frozen_scenes = freeze.get("scenes")
    if not isinstance(prepared_scenes, Mapping) or not isinstance(
        frozen_scenes, Mapping
    ):
        raise ValueError("prepared freeze scene bindings are missing")
    for scene in SCENES:
        prepared_scene = prepared_scenes.get(scene)
        frozen_scene = frozen_scenes.get(scene)
        if (
            not isinstance(prepared_scene, Mapping)
            or not isinstance(frozen_scene, Mapping)
            or prepared_scene.get("frozen_config")
            != frozen_scene.get("frozen_config")
        ):
            raise ValueError("prepared frozen configs do not match final inputs")

    shared = freeze.get("shared_bindings")
    if not isinstance(shared, Mapping):
        raise ValueError("frozen shared bindings are missing")
    required_shared = {
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
    if not required_shared.issubset(shared):
        raise ValueError("frozen shared bindings are incomplete")
    for name in (
        "source_manifest",
        "common_target_manifest",
        "common_target_arrays",
        "alias_map",
        "evaluator",
    ):
        _content_binding(
            shared.get(name),
            base=freeze_path.parent,
            label=f"frozen shared {name}",
        )
    finalizers = shared.get("finalizers")
    if not isinstance(finalizers, Mapping) or not {
        "common_v2",
        "official_t2",
    }.issubset(finalizers):
        raise ValueError("frozen finalizer bindings are incomplete")
    for name in ("common_v2", "official_t2"):
        _content_binding(
            finalizers.get(name),
            base=freeze_path.parent,
            label=f"frozen {name} finalizer",
        )
    shared_input = _content_binding(
        shared.get("input_manifest"),
        base=freeze_path.parent,
        label="frozen input manifest",
    )
    schedule = _content_binding(
        shared.get("schedule"),
        base=freeze_path.parent,
        label="frozen causal schedule",
    )
    shared_camera = _content_binding(
        shared.get("camera"),
        base=freeze_path.parent,
        label="frozen camera manifest",
    )
    expected: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        scene_binding = frozen_scenes.get(scene)
        if not isinstance(scene_binding, Mapping):
            raise ValueError(f"frozen {scene} source binding is missing")
        if not {
            "source_config",
            "frozen_config",
            "rgbd",
            "vocabulary",
            "cache",
        }.issubset(scene_binding):
            raise ValueError(f"frozen {scene} source binding is incomplete")
        _content_binding(
            scene_binding.get("source_config"),
            base=freeze_path.parent,
            label=f"frozen {scene} source config",
        )
        rgbd = scene_binding.get("rgbd")
        vocabulary = scene_binding.get("vocabulary")
        cache = scene_binding.get("cache")
        if not all(isinstance(value, Mapping) for value in (rgbd, vocabulary, cache)):
            raise ValueError(f"frozen {scene} source binding is incomplete")
        assert isinstance(rgbd, Mapping)
        assert isinstance(vocabulary, Mapping)
        assert isinstance(cache, Mapping)
        rgbd_root = rgbd.get("root")
        if (
            type(rgbd_root) is not str
            or type(rgbd.get("frame_count")) is not int
            or rgbd["frame_count"] <= 0
            or type(rgbd.get("file_hash_count")) is not int
            or rgbd["file_hash_count"] <= 0
        ):
            raise ValueError(f"frozen {scene} RGB-D inventory is invalid")
        _assert_directory(
            _absolute(rgbd_root), label=f"frozen {scene} RGB-D root"
        )
        _content_binding(
            rgbd.get("source_database"),
            base=freeze_path.parent,
            label=f"frozen {scene} source database",
        )
        if any(
            not _is_sha256(cache.get(field))
            for field in (
                "frontend_algorithm_sha256",
                "frontend_cache_prefix_sha256",
                "dense_cache_prefix_sha256",
            )
        ):
            raise ValueError(f"frozen {scene} cache identity is invalid")
        camera = _content_binding(
            rgbd.get("camera"),
            base=freeze_path.parent,
            label=f"frozen {scene} camera",
        )
        combined = rgbd.get("combined_output_sha256")
        if camera != shared_camera or not _is_sha256(combined):
            raise ValueError(f"frozen {scene} RGB-D identity is invalid")
        expected[scene] = {
            "input_manifest": shared_input,
            "export_manifest": _content_binding(
                rgbd.get("export_manifest"),
                base=freeze_path.parent,
                label=f"frozen {scene} export manifest",
            ),
            "camera": camera,
            "trajectory": _content_binding(
                rgbd.get("trajectory"),
                base=freeze_path.parent,
                label=f"frozen {scene} trajectory",
            ),
            "timestamps": _content_binding(
                rgbd.get("timestamps"),
                base=freeze_path.parent,
                label=f"frozen {scene} timestamps",
            ),
            "vocabulary_json": _content_binding(
                vocabulary.get("json"),
                base=freeze_path.parent,
                label=f"frozen {scene} vocabulary JSON",
            ),
            "vocabulary_txt": _content_binding(
                vocabulary.get("txt"),
                base=freeze_path.parent,
                label=f"frozen {scene} vocabulary TXT",
            ),
            "frontend_manifest": _content_binding(
                cache.get("frontend_manifest"),
                base=freeze_path.parent,
                label=f"frozen {scene} frontend cache manifest",
            ),
            "dense_manifest": _content_binding(
                cache.get("dense_manifest"),
                base=freeze_path.parent,
                label=f"frozen {scene} dense cache manifest",
            ),
            "rgbd_combined_output_sha256": combined,
        }
    return expected, schedule


def _mapping_checkpoint_identity(
    source_index: Mapping[str, Any], *, base: Path, scene: str
) -> list[tuple[int, int, int, int, str, int, str, int]]:
    if (
        source_index.get("schema_version") != 1
        or source_index.get("dataset") != "TESSE-CD"
        or source_index.get("method") != "OVIV2"
        or source_index.get("mode") != "causal_checkpoint_exports"
        or source_index.get("scene") != scene
    ):
        raise ValueError(f"{scene} mapping source index identity mismatch")
    checkpoints = source_index.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"{scene} mapping source index has no checkpoints")
    result: list[tuple[int, int, int, int, str, int, str, int]] = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{scene} mapping checkpoint {index} is invalid")
        frame = checkpoint.get("frame_index")
        timestamp = checkpoint.get("timestamp_ns")
        consumed = checkpoint.get("consumed_through_frame")
        exclusive = checkpoint.get("consumed_through_frame_exclusive")
        if (
            type(frame) is not int
            or type(timestamp) is not int
            or consumed != frame
            or exclusive != frame + 1
        ):
            raise ValueError(f"{scene} mapping checkpoint boundary is invalid")
        snapshot = _declared_record(
            checkpoint.get("snapshot", {}),
            base=base,
            label=f"{scene} mapping checkpoint {index} snapshot",
        )
        entities = _declared_record(
            checkpoint.get("entities", {}),
            base=base,
            label=f"{scene} mapping checkpoint {index} entities",
        )
        result.append(
            (
                frame,
                timestamp,
                consumed,
                exclusive,
                snapshot["sha256"],
                snapshot["byte_count"],
                entities["sha256"],
                entities["byte_count"],
            )
        )
    if result != sorted(set(result)):
        raise ValueError(f"{scene} mapping checkpoints are not canonical")
    return result


def _temporal_checkpoint_identity(
    temporal: Mapping[str, Any], *, base: Path, scene: str
) -> list[tuple[int, int, int, int, str, int, str, int]]:
    if (
        temporal.get("schema_version") != 1
        or temporal.get("dataset") != "TESSE-CD"
        or temporal.get("method") != "OVIV2"
        or temporal.get("mode") != "causal_checkpoints"
        or temporal.get("scene") != scene
    ):
        raise ValueError(f"{scene} temporal artifact identity mismatch")
    checkpoints = temporal.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"{scene} temporal artifact has no checkpoints")
    result: list[tuple[int, int, int, int, str, int, str, int]] = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{scene} temporal checkpoint {index} is invalid")
        frame = checkpoint.get("frame_index")
        timestamp = checkpoint.get("timestamp_ns")
        consumed = checkpoint.get("consumed_through_frame")
        exclusive = checkpoint.get("consumed_through_frame_exclusive")
        if (
            type(frame) is not int
            or type(timestamp) is not int
            or consumed != frame
            or exclusive != frame + 1
        ):
            raise ValueError(f"{scene} temporal checkpoint boundary is invalid")
        snapshot = _declared_record(
            checkpoint.get("snapshot", {}),
            base=base,
            label=f"{scene} temporal checkpoint {index} snapshot",
        )
        entities = _declared_record(
            checkpoint.get("entities", {}),
            base=base,
            label=f"{scene} temporal checkpoint {index} entities",
        )
        result.append(
            (
                frame,
                timestamp,
                consumed,
                exclusive,
                snapshot["sha256"],
                snapshot["byte_count"],
                entities["sha256"],
                entities["byte_count"],
            )
        )
    return result


def _mapping_run(
    *,
    key: str,
    scene: str,
    root: Path,
    config_path: Path,
    config_record: Mapping[str, Any],
    algorithm_hash: str,
    adapter_commit: str,
    source_bindings: Mapping[str, Any],
    schedule_binding: Mapping[str, Any],
    frozen_run_identity: Mapping[str, Any],
    repository_tree: str,
) -> dict[str, Any]:
    root = _assert_directory(root, label=f"{key} mapping root")
    manifest_path = root / "run_manifest.json"
    provenance_path = root / "run_provenance.json"
    source_index_path = root / "source_index.json"
    manifest, manifest_record = _json(
        manifest_path, label=f"{key} mapping manifest"
    )
    provenance, _ = _json(provenance_path, label=f"{key} mapping provenance")
    source_index, source_index_record = _json(
        source_index_path, label=f"{key} source index"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset") != "TESSE-CD"
        or manifest.get("method_id") != "OVIV2"
        or manifest.get("mode") != "causal_checkpoints"
        or manifest.get("scene") != scene
        or manifest.get("algorithm_hash") != algorithm_hash
        or manifest.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
        or manifest.get("config")
        != {
            "sha256": config_record["sha256"],
            "byte_count": config_record["byte_count"],
        }
        or manifest.get("schedule") != schedule_binding
    ):
        raise ValueError(f"{key} mapping manifest does not match the freeze")
    if manifest.get("source_bindings") != source_bindings:
        raise ValueError(f"{key} mapping source bindings do not match the freeze")
    occlusion_path = root / "occlusion_checkpoint_index.json"
    _declared_record(
        manifest.get("occlusion_checkpoint_index", {}),
        base=root,
        label=f"{key} mapping occlusion index",
        expected_path=occlusion_path,
    )
    occlusion, _ = _json(
        occlusion_path, label=f"{key} mapping occlusion index"
    )
    execution = _validate_formal_run_fields(
        manifest,
        expected_frozen=frozen_run_identity,
        key=key,
        root=root,
        label=f"{key} mapping manifest",
    )
    _validate_formal_run_fields(
        source_index,
        expected_frozen=frozen_run_identity,
        key=key,
        root=root,
        label=f"{key} source index",
    )
    _validate_formal_run_fields(
        occlusion,
        expected_frozen=frozen_run_identity,
        key=key,
        root=root,
        label=f"{key} occlusion index",
    )
    if provenance.get("repository_commit") != adapter_commit:
        raise ValueError(f"{key} mapping adapter commit mismatch")
    if provenance.get("repository_tree") != repository_tree:
        raise ValueError(f"{key} mapping repository tree mismatch")
    if provenance.get("dirty_state_digest") != CLEAN_DIRTY_STATE_DIGEST:
        raise ValueError(f"{key} mapping run was dirty")
    if _absolute(str(provenance.get("config_path", ""))) != config_path:
        raise ValueError(f"{key} mapping config path mismatch")
    if _absolute(str(provenance.get("output", ""))) != root:
        raise ValueError(f"{key} mapping output path mismatch")
    command = _command(provenance.get("command"), label=f"{key} mapping command")
    required_environment = ("python", "platform", "library_versions")
    if any(field not in provenance for field in required_environment) or not isinstance(
        provenance.get("library_versions"), Mapping
    ):
        raise ValueError(f"{key} mapping environment is incomplete")
    required_hardware = ("hostname", "machine", "gpu_inventory")
    if any(field not in provenance for field in required_hardware) or not isinstance(
        provenance.get("gpu_inventory"), list
    ):
        raise ValueError(f"{key} mapping hardware is incomplete")
    checkpoints = _mapping_checkpoint_identity(
        source_index, base=source_index_path.parent, scene=scene
    )
    manifest_checkpoints = manifest.get("checkpoints")
    if not isinstance(manifest_checkpoints, list) or not manifest_checkpoints:
        raise ValueError(f"{key} mapping manifest has no checkpoints")
    manifest_boundaries: list[tuple[int, int, int, int]] = []
    for index, checkpoint in enumerate(manifest_checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{key} mapping manifest checkpoint {index} is invalid")
        boundary = (
            checkpoint.get("frame_index"),
            checkpoint.get("timestamp_ns"),
            checkpoint.get("consumed_through_frame"),
            checkpoint.get("consumed_through_frame_exclusive"),
        )
        frame, timestamp, consumed, exclusive = boundary
        if (
            type(frame) is not int
            or type(timestamp) is not int
            or consumed != frame
            or exclusive != frame + 1
            or checkpoint.get("scene") != scene
        ):
            raise ValueError(f"{key} mapping manifest checkpoint boundary is invalid")
        manifest_boundaries.append(boundary)
    if manifest_boundaries != sorted(set(manifest_boundaries)):
        raise ValueError(f"{key} mapping manifest checkpoints are not canonical")
    if any(checkpoint[:4] not in manifest_boundaries for checkpoint in checkpoints):
        raise ValueError(f"{key} mapping source checkpoints are not in the run manifest")
    raw_paths = (
        ("mapping_run_manifest", manifest_path),
        ("mapping_run_provenance", provenance_path),
        ("mapping_source_index", source_index_path),
        ("mapping_capture_status", root / "capture_status.json"),
        ("mapping_occlusion_index", occlusion_path),
    )
    manifest_projection, source_index_projection = _mapping_identity_projection(
        manifest,
        source_index,
        occlusion,
    )
    return {
        "root": root,
        "checkpoints": checkpoints,
        "manifest_sha256": manifest_record["sha256"],
        "source_index_sha256": source_index_record["sha256"],
        "manifest_identity_sha256": manifest_projection,
        "source_index_identity_sha256": source_index_projection,
        "frozen_run_identity": dict(frozen_run_identity),
        "run_execution": execution,
        "command": command,
        "environment": {
            field: provenance[field]
            for field in (
                "python",
                "platform",
                "library_versions",
                "torch_cuda_version",
                "cudnn_version",
                "nvcc_version",
            )
            if field in provenance
        },
        "hardware": {
            field: provenance[field]
            for field in (
                "hostname",
                "machine",
                "cuda_visible_devices",
                "gpu_inventory",
            )
            if field in provenance
        },
        "raw_outputs": [
            _named_record(path, name=name, run_key=key, scene=scene)
            for name, path in raw_paths
        ],
    }


def _temporal_manifest_from_bridge(
    bridge: Mapping[str, Any], *, bridge_path: Path, scene: str
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    inputs = bridge.get("hashed_inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"{scene} bridge hashed inputs are missing")
    for index, entry in enumerate(inputs):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{scene} bridge hashed input {index} is invalid")
        record = _declared_record(
            entry,
            base=bridge_path.parent,
            label=f"{scene} bridge hashed input {index}",
        )
        try:
            payload, _ = _json(
                Path(record["path"]), label=f"{scene} bridge JSON input {index}"
            )
        except (UnicodeError, ValueError):
            continue
        if (
            payload.get("schema_version") == 1
            and payload.get("dataset") == "TESSE-CD"
            and payload.get("method") == "OVIV2"
            and payload.get("mode") == "causal_checkpoints"
            and payload.get("scene") == scene
        ):
            candidates.append((payload, Path(record["path"]), record))
    if len(candidates) != 1:
        raise ValueError(f"{scene} bridge must bind exactly one temporal artifact")
    return candidates[0]


def _official_run(
    *,
    key: str,
    scene: str,
    root: Path,
    run_id: str,
    config_path: Path,
    config_record: Mapping[str, Any],
    mapping_checkpoints: Sequence[tuple[int, int, int, int, str, int, str, int]],
    mapping_root: Path,
    frozen_run_identity: Mapping[str, Any],
    run_execution: Mapping[str, Any],
) -> dict[str, Any]:
    root = _assert_directory(root, label=f"{key} official root")
    status_path = root / "run_status.json"
    status = validate_khronos_run_status(status_path, scene=scene)
    if (
        status.get("status") != "PASS"
        or status.get("dataset") != "TESSE-CD"
        or status.get("scene") != scene
        or status.get("method") != "OVIV2"
        or status.get("mode") != "causal_checkpoints"
        or status.get("bridge_mode") != "temporal_checkpoints"
    ):
        raise ValueError(f"{key} official run status is not usable")
    identity = status.get("run_identity")
    expected_identity = {
        "run_id": run_id,
        "config_sha256": config_record["sha256"],
    }
    if identity != expected_identity:
        raise ValueError(f"{key} official run identity mismatch")
    _declared_record(
        status.get("config", {}),
        base=status_path.parent,
        label=f"{key} official config",
        expected_path=config_path,
    )
    sources = status.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{key} official source bindings are missing")
    source_paths = {
        Path(
            _declared_record(
                entry,
                base=status_path.parent,
                label=f"{key} official source {index}",
            )["path"]
        )
        for index, entry in enumerate(sources)
    }
    bridge_path = root / "bridge_input/bridge_manifest.json"
    if bridge_path not in source_paths:
        raise ValueError(f"{key} bridge manifest is not a run-status source")
    bridge = validate_temporal_bridge_manifest(bridge_path)
    if (
        bridge.get("dataset") != "TESSE-CD"
        or bridge.get("method") != "OVIV2"
        or bridge.get("mode") != "temporal_checkpoints"
        or bridge.get("scene_id") != scene
    ):
        raise ValueError(f"{key} temporal bridge identity mismatch")
    temporal, temporal_path, temporal_record = _temporal_manifest_from_bridge(
        bridge, bridge_path=bridge_path, scene=scene
    )
    observed_execution = _validate_formal_run_fields(
        temporal,
        expected_frozen=frozen_run_identity,
        key=key,
        root=mapping_root,
        label=f"{key} temporal manifest",
    )
    if observed_execution != dict(run_execution):
        raise ValueError(f"{key} temporal run execution mismatch")
    temporal_content, _ = _file(
        temporal_path, label=f"{key} temporal manifest", capture=True
    )
    assert temporal_content is not None
    temporal_projection = _temporal_identity_projection(
        temporal_content,
        temporal_path=temporal_path,
    )
    if temporal_projection is None:
        raise ValueError(f"{key} temporal manifest has no frozen run identity")
    temporal_checkpoints = _temporal_checkpoint_identity(
        temporal, base=temporal_path.parent, scene=scene
    )
    if list(mapping_checkpoints) != temporal_checkpoints:
        raise ValueError(f"{key} temporal artifact does not bind its mapping run")

    evaluation_status_path = root / "evaluation/evaluation_status.json"
    evaluation, _ = _json(
        evaluation_status_path, label=f"{key} evaluation status"
    )
    if (
        evaluation.get("status") != "PASS"
        or evaluation.get("scene") != scene
        or evaluation.get("method") != "OVIV2"
        or evaluation.get("mode") != "causal_checkpoints"
    ):
        raise ValueError(f"{key} official evaluator status is not PASS")
    metrics_path = root / "evaluation/official_metrics.json"
    repeat_path = root / "evaluation/official_metrics.repeat.json"
    _declared_record(
        evaluation.get("official_metrics", {}),
        base=evaluation_status_path.parent,
        label=f"{key} official metrics",
        expected_path=metrics_path,
        require_byte_count=False,
    )
    _declared_record(
        evaluation.get("official_metrics_repeat", {}),
        base=evaluation_status_path.parent,
        label=f"{key} repeated official metrics",
        expected_path=repeat_path,
        require_byte_count=False,
    )
    metrics_bytes, _ = _file(
        metrics_path, label=f"{key} official metrics", capture=True
    )
    repeat_bytes, _ = _file(
        repeat_path, label=f"{key} repeated official metrics", capture=True
    )
    assert metrics_bytes is not None and repeat_bytes is not None
    if metrics_bytes != repeat_bytes:
        raise ValueError(f"{key} official metric repeat is not byte-identical")
    metrics_payload, _ = _json(metrics_path, label=f"{key} official metrics")
    results_dir = root / "map/results"
    source_names = (
        "static_objects.csv",
        "dynamic_objects.csv",
        "background_mesh.csv",
    )
    metric_sources = metrics_payload.get("sources")
    if not isinstance(metric_sources, list) or len(metric_sources) != len(source_names):
        raise ValueError(f"{key} official metric source bindings are incomplete")
    for index, (entry, name) in enumerate(zip(metric_sources, source_names, strict=True)):
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"{key} official metric source {index} is invalid")
        expected_path = results_dir / name
        expected_relative = os.path.relpath(expected_path, start=metrics_path.parent)
        if entry.get("path") != expected_relative:
            raise ValueError(f"{key} official metric source path is not canonical")
        _, observed = _file(
            expected_path, label=f"{key} official metric source {name}"
        )
        if (
            entry.get("sha256") != observed["sha256"]
            or entry.get("byte_count") != observed["byte_count"]
        ):
            raise ValueError(f"{key} official metric source hash mismatch")
    recomputed = summarize_khronos_official_metrics(results_dir)
    if _canonical_json(metrics_payload.get("metrics")) != _canonical_json(recomputed):
        raise ValueError(f"{key} official metrics differ from recomputed CSV values")

    evaluator_inputs = {
        "official_evaluator_config": (
            evaluation.get("config"),
            root / f"evaluation/{scene}.yaml",
        ),
        "official_evaluator_log": (
            evaluation.get("log"),
            root / "evaluation/evaluate.log",
        ),
        "official_evaluator_time": (
            evaluation.get("process_time"),
            root / "evaluation/evaluate.time.log",
        ),
    }
    for name, (entry, expected_path) in evaluator_inputs.items():
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"{key} {name} binding is invalid")
        _declared_record(
            entry,
            base=evaluation_status_path.parent,
            label=f"{key} {name}",
            expected_path=expected_path,
            require_byte_count=False,
        )
    evidence = build_scene_evidence(
        metrics_path,
        status_path,
        method_key="OVIV2",
        mode="causal_checkpoints",
    )
    if evidence.get("run_identity") != expected_identity:
        raise ValueError(f"{key} finalizer input identity mismatch")

    raw_paths = (
        ("official_run_status", status_path),
        ("official_bridge_manifest", bridge_path),
        ("official_temporal_manifest", temporal_path),
        ("official_build_manifest", root / "build_manifest.json"),
        ("official_final_map", root / "map/final.4dmap"),
        ("official_map_timestamps", root / "map/map_timestamps.json"),
        ("official_experiment_log", root / "map/experiment_log.txt"),
        ("official_evaluation_status", evaluation_status_path),
        *(
            (name, path)
            for name, (_, path) in evaluator_inputs.items()
        ),
        ("official_metrics", metrics_path),
        ("official_metrics_repeat", repeat_path),
        ("official_static_objects", root / "map/results/static_objects.csv"),
        ("official_dynamic_objects", root / "map/results/dynamic_objects.csv"),
        ("official_background_mesh", root / "map/results/background_mesh.csv"),
    )
    return {
        "root": root,
        "metrics_bytes": metrics_bytes,
        "temporal_sha256": temporal_record["sha256"],
        "temporal_identity_sha256": temporal_projection[0],
        "commands": [
            _command(status.get("build_command"), label=f"{key} build command"),
            _command(status.get("command"), label=f"{key} bridge command"),
            _command(evaluation.get("command"), label=f"{key} evaluator command"),
        ],
        "raw_outputs": [
            _named_record(path, name=name, run_key=key, scene=scene)
            for name, path in raw_paths
        ],
    }


def build_provenance(
    freeze_manifest: str | Path,
    *,
    official_runs: Mapping[str, Path],
    weights: Mapping[str, Path],
) -> dict[str, Any]:
    freeze_path = _absolute(freeze_manifest)
    freeze, freeze_record = _json(freeze_path, label="OVIV2 freeze manifest")
    if (
        freeze.get("schema_version") != 1
        or freeze.get("freeze_id") != "oviv2-tessecd-v1"
        or freeze.get("status") != "FROZEN"
        or freeze.get("method") != "OVIV2"
        or freeze.get("dataset") != "TESSE-CD"
    ):
        raise ValueError("provenance requires the FROZEN oviv2-tessecd-v1 manifest")
    frozen_commands = freeze.get("commands")
    mapping_commands = (
        frozen_commands.get("mapping") if isinstance(frozen_commands, Mapping) else None
    )
    if not isinstance(mapping_commands, list) or len(mapping_commands) != 4:
        raise ValueError("frozen mapping commands are incomplete")
    if any(type(value) is not str or not value for value in mapping_commands):
        raise ValueError("frozen mapping command must be a non-empty string")
    source_bindings, schedule_binding = _freeze_source_bindings(
        freeze, freeze_path=freeze_path
    )
    repository = _repository_binding(freeze.get("repository"), label="frozen")
    adapter_commit = repository.get("commit")
    upstream_commit = repository.get("stage3_lineage_commit")
    if (
        not _is_commit(adapter_commit)
        or upstream_commit != STAGE3_LINEAGE_COMMIT
        or repository.get("clean") is not True
        or repository.get("stage3_is_ancestor") is not True
    ):
        raise ValueError("frozen repository identity is invalid")
    current = _repository_state()
    if current.get("commit") != adapter_commit:
        raise ValueError("builder repository commit differs from the freeze")
    if current.get("dirty_state_digest") != CLEAN_DIRTY_STATE_DIGEST:
        raise ValueError("builder repository is dirty")

    scenes = freeze.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(SCENES):
        raise ValueError("frozen scene bindings are incomplete")
    algorithm = freeze.get("algorithm")
    algorithm_hash = algorithm.get("sha256") if isinstance(algorithm, Mapping) else None
    if not _is_sha256(algorithm_hash):
        raise ValueError("frozen algorithm hash is invalid")
    config_paths: dict[str, Path] = {}
    config_records: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        scene_binding = scenes[scene]
        if not isinstance(scene_binding, Mapping):
            raise ValueError(f"frozen {scene} binding is invalid")
        frozen_config = scene_binding.get("frozen_config")
        if not isinstance(frozen_config, Mapping):
            raise ValueError(f"frozen {scene} config binding is missing")
        config_path = _declared_path(
            frozen_config, base=freeze_path.parent, label=f"frozen {scene} config"
        )
        config_record = _declared_record(
            frozen_config,
            base=freeze_path.parent,
            label=f"frozen {scene} config",
            expected_path=config_path,
        )
        config, _ = _json(config_path, label=f"frozen {scene} config")
        if (
            config.get("method_id") != "OVIV2"
            or config.get("dataset") != "TESSE-CD"
            or config.get("scene") != scene
            or config.get("algorithm_hash") != algorithm_hash
            or config.get("stage3_lineage_commit") != upstream_commit
        ):
            raise ValueError(f"frozen {scene} config identity mismatch")
        config_paths[scene] = config_path
        config_records[scene] = config_record
        configs[scene] = config
    selection = freeze["selection"]
    assert isinstance(selection, Mapping)
    if selection.get("selected_config_sha256") != config_records["apartment"]["sha256"]:
        raise ValueError("frozen Apartment selection does not bind its config")

    shared = freeze.get("shared_bindings")
    if not isinstance(shared, Mapping):
        raise ValueError("frozen shared bindings are missing")
    dataset_manifest = _declared_record(
        shared.get("source_manifest", {}),
        base=freeze_path.parent,
        label="frozen TESSE-CD source manifest",
    )
    finalizers = shared.get("finalizers")
    if not isinstance(finalizers, Mapping):
        raise ValueError("frozen finalizer bindings are missing")
    official_finalizer = _declared_record(
        finalizers.get("official_t2", {}),
        base=freeze_path.parent,
        label="frozen T2 finalizer",
    )

    outputs = freeze.get("output_roots")
    if not isinstance(outputs, Mapping) or set(outputs) != set(RUN_KEYS):
        raise ValueError("freeze must declare exactly four mapping output roots")
    if set(official_runs) != set(RUN_KEYS):
        raise ValueError("exactly four official run roots are required")
    mapping_roots: dict[str, Path] = {}
    mapping_identities: dict[str, tuple[int, int]] = {}
    official_roots: dict[str, Path] = {}
    official_identities: dict[str, tuple[int, int]] = {}
    for key in RUN_KEYS:
        if type(outputs[key]) is not str:
            raise ValueError(f"{key} mapping output root must be a string")
        mapping_roots[key] = _absolute(outputs[key])
        mapping_identities[key] = _directory_identity(
            mapping_roots[key], label=f"{key} mapping root"
        )
        official_roots[key] = _absolute(official_runs[key])
        official_identities[key] = _directory_identity(
            official_roots[key], label=f"{key} official root"
        )
    if len(set(mapping_identities.values())) != len(RUN_KEYS):
        raise ValueError("mapping run directories must be distinct")
    if len(set(official_identities.values())) != len(RUN_KEYS):
        raise ValueError("official run directories must be distinct")
    mapping: dict[str, dict[str, Any]] = {}
    frozen_run_identities = {
        scene: _expected_frozen_run_identity(
            freeze=freeze,
            freeze_record=freeze_record,
            scene=scene,
            config=configs[scene],
            config_record=config_records[scene],
        )
        for scene in SCENES
    }
    for key in RUN_KEYS:
        scene = key.rsplit("_run", 1)[0]
        mapping[key] = _mapping_run(
            key=key,
            scene=scene,
            root=mapping_roots[key],
            config_path=config_paths[scene],
            config_record=config_records[scene],
            algorithm_hash=algorithm_hash,
            adapter_commit=adapter_commit,
            source_bindings=source_bindings[scene],
            schedule_binding=schedule_binding,
            frozen_run_identity=frozen_run_identities[scene],
            repository_tree=repository["tree"],
        )

    official: dict[str, dict[str, Any]] = {}
    for key in RUN_KEYS:
        scene = key.rsplit("_run", 1)[0]
        official[key] = _official_run(
            key=key,
            scene=scene,
            root=official_roots[key],
            run_id=freeze["freeze_id"],
            config_path=config_paths[scene],
            config_record=config_records[scene],
            mapping_checkpoints=mapping[key]["checkpoints"],
            mapping_root=mapping[key]["root"],
            frozen_run_identity=mapping[key]["frozen_run_identity"],
            run_execution=mapping[key]["run_execution"],
        )
    for scene in SCENES:
        first_mapping = mapping[f"{scene}_run1"]
        second_mapping = mapping[f"{scene}_run2"]
        if (
            first_mapping["manifest_identity_sha256"]
            != second_mapping["manifest_identity_sha256"]
            or first_mapping["source_index_identity_sha256"]
            != second_mapping["source_index_identity_sha256"]
            or first_mapping["checkpoints"] != second_mapping["checkpoints"]
        ):
            raise ValueError(
                f"{scene} mapping run manifests identity projections are not byte-identical"
            )
        if (
            official[f"{scene}_run1"]["temporal_identity_sha256"]
            != official[f"{scene}_run2"]["temporal_identity_sha256"]
        ):
            raise ValueError(
                f"{scene} temporal manifests identity projections are not byte-identical"
            )
        if (
            official[f"{scene}_run1"]["metrics_bytes"]
            != official[f"{scene}_run2"]["metrics_bytes"]
        ):
            raise ValueError(f"{scene} full-run metric repeats are not byte-identical")

    expected_weights = _expected_weights(freeze.get("models", {}))
    if set(weights) != set(expected_weights):
        raise ValueError("weight bindings do not cover the exact frozen model roles")
    weight_records: list[dict[str, Any]] = []
    for role in sorted(expected_weights):
        record = _named_record(
            _absolute(weights[role]),
            name=role.replace(".", "_"),
            role=role,
        )
        if record["sha256"] != expected_weights[role]:
            raise ValueError(f"weight hash does not match the freeze: {role}")
        weight_records.append(record)

    command_values = list(mapping_commands)
    command_values.extend(mapping[key]["command"] for key in RUN_KEYS)
    for key in RUN_KEYS:
        command_values.extend(official[key]["commands"])

    raw_outputs = [
        {"name": "freeze_manifest", **freeze_record},
        {"name": "frozen_t2_finalizer", **official_finalizer},
    ]
    for key in RUN_KEYS:
        raw_outputs.extend(mapping[key]["raw_outputs"])
        raw_outputs.extend(official[key]["raw_outputs"])

    payload = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_t2_provenance_v1",
        "run_id": freeze["freeze_id"],
        "upstream_commit": upstream_commit,
        "adapter_commit": adapter_commit,
        "dirty_state_digest": CLEAN_DIRTY_STATE_DIGEST,
        "frozen_run_identities": frozen_run_identities,
        "run_executions": {
            key: mapping[key]["run_execution"] for key in RUN_KEYS
        },
        "dataset_manifest": dataset_manifest,
        "commands": _commands(command_values),
        "environment": {
            "freeze": dict(freeze.get("environment", {})),
            "mapping_runs": {
                key: mapping[key]["environment"] for key in RUN_KEYS
            },
        },
        "hardware": {
            "mapping_runs": {key: mapping[key]["hardware"] for key in RUN_KEYS}
        },
        "configs": [
            {
                "name": f"{scene}_frozen_config",
                "scene": scene,
                **config_records[scene],
            }
            for scene in SCENES
        ],
        "weights": weight_records,
        "raw_outputs": raw_outputs,
        "protocol_deviations": [],
        "freeze_manifest": freeze_record,
        "run_roots": {
            key: {
                "mapping": str(mapping[key]["root"]),
                "official": str(official[key]["root"]),
            }
            for key in RUN_KEYS
        },
    }
    final_sources, final_schedule = _freeze_source_bindings(
        freeze, freeze_path=freeze_path
    )
    if final_sources != source_bindings or final_schedule != schedule_binding:
        raise ValueError("frozen source bindings changed during provenance capture")
    _revalidate_records(payload)
    final_state = _repository_state()
    if final_state != current:
        raise ValueError("builder repository changed during provenance capture")
    return payload


def write_provenance(
    freeze_manifest: str | Path,
    *,
    official_runs: Mapping[str, Path],
    weights: Mapping[str, Path],
    output: Path,
) -> Path:
    payload = build_provenance(
        freeze_manifest, official_runs=official_runs, weights=weights
    )
    return _atomic_json_no_replace(_absolute(output), payload)


def _weight_binding(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or not role or not raw_path:
        raise argparse.ArgumentTypeError("weight binding must be ROLE=PATH")
    return role, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    for key in RUN_KEYS:
        parser.add_argument(
            f"--{key.replace('_', '-')}", dest=key, type=Path, required=True
        )
    parser.add_argument(
        "--weight", action="append", type=_weight_binding, default=[], metavar="ROLE=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    weights = dict(args.weight)
    if len(weights) != len(args.weight):
        parser.error("weight roles must not be duplicated")
    args.weights = weights
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    official_runs = {key: getattr(args, key) for key in RUN_KEYS}
    write_provenance(
        args.freeze_manifest,
        official_runs=official_runs,
        weights=args.weights,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
