#!/usr/bin/env python3
"""Finalize signed-depth versus missing-as-absence TESSE-CD occlusion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.build_oviv2_tesse_occlusion_ablation import (
    MUTATED_FIELDS,
    SCENES,
    _canonical_bytes,
    _is_sha256,
    _json_hash,
    _load_json,
    _read_regular_file,
    _sha256_bytes,
    _strict_object,
    _validate_parent,
    _write_exclusive,
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    evaluate_occlusion_package,
)


_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_id",
        "missing_observation_policy",
        "target_manifest",
        "checkpoint_count",
        "checkpoint_index",
        "run_config",
        "evaluation_checkpoint_frames_sha256",
        "fixed_anchor_mapping_rule",
        "fixed_anchor_mappings",
        "stress_layers",
        "headline_gate",
    }
)
_BASE_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_id",
        "scene",
        "algorithm_hash",
        "run_config",
        "target_manifest",
        "evaluation_checkpoint_frames_sha256",
        "snapshots",
    }
)
_FORMAL_INDEX_FIELDS = _BASE_INDEX_FIELDS | {
    "frozen_run_identity",
    "run_execution",
}
_RUN_EXECUTION_FIELDS = frozenset(
    {
        "schema_version",
        "run_slot",
        "output_root",
        "root_device",
        "root_inode",
        "execution_id",
    }
)


def _result_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_result(path: Path, role: str) -> tuple[dict[str, Any], bytes]:
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
    if not isinstance(payload, dict) or raw != _result_bytes(payload):
        raise ValueError(f"{role} must be a canonical evaluator JSON object")
    return payload, raw


def _content_record(raw: bytes) -> dict[str, int | str]:
    return {"sha256": _sha256_bytes(raw), "byte_count": len(raw)}


def _binding_record(value: object, role: str) -> dict[str, int | str]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "byte_count"}:
        raise ValueError(f"{role} binding is invalid")
    if not (
        _is_sha256(value["sha256"])
        and type(value["byte_count"]) is int
        and value["byte_count"] >= 0
    ):
        raise ValueError(f"{role} binding is invalid")
    return {"sha256": value["sha256"], "byte_count": value["byte_count"]}


def _relative_path(value: object, role: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{role} path must be canonical and relative")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{role} path must be canonical and relative")
    if path.as_posix() != value:
        raise ValueError(f"{role} path must be canonical and relative")
    return path


def _load_ablation_manifest(
    path: Path,
    *,
    parent: Mapping[str, Any],
    parent_raw: bytes,
    parent_configs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes, dict[str, dict[str, Any]], list[tuple[Path, bytes, str]]]:
    manifest, raw = _load_json(path, "ablation manifest")
    expected_fields = {
        "schema_version",
        "manifest_id",
        "status",
        "dataset",
        "method_id",
        "ablation_id",
        "parent_freeze",
        "mutation",
        "algorithm",
        "occlusion_target_manifest_sha256",
        "evaluation_checkpoint_frames_sha256",
        "frozen_identity",
        "scenes",
    }
    if set(manifest) != expected_fields or not (
        manifest.get("schema_version") == 1
        and manifest.get("manifest_id")
        == "oviv2_tesse_cd_occlusion_ablation_v1"
        and manifest.get("status") == "FROZEN"
        and manifest.get("dataset") == "TESSE-CD"
        and manifest.get("method_id") == "OVIV2"
        and manifest.get("ablation_id") == "missing_as_absence"
    ):
        raise ValueError("ablation manifest identity is invalid")
    expected_parent = {
        "sha256": _sha256_bytes(parent_raw),
        "byte_count": len(parent_raw),
        "freeze_id": parent["freeze_id"],
        "repository_commit": parent["repository"]["commit"],
        "algorithm_hash": parent_configs["apartment"]["algorithm_hash"],
    }
    if manifest.get("parent_freeze") != expected_parent:
        raise ValueError("ablation manifest parent freeze binding mismatch")
    if manifest.get("mutation") != {
        "only_changed_fields": sorted(MUTATED_FIELDS),
        "missing_observation_policy": {
            "from": "signed_depth",
            "to": "missing_as_absence",
        },
    }:
        raise ValueError("ablation mutation contract is invalid")
    expected_frozen_identity = {
        "selection_sha256": _json_hash(parent.get("selection")),
        "shared_bindings_sha256": _json_hash(parent.get("shared_bindings")),
        "models_sha256": _json_hash(parent.get("models")),
    }
    if manifest.get("frozen_identity") != expected_frozen_identity:
        raise ValueError("ablation frozen identity binding mismatch")
    if not (
        manifest.get("occlusion_target_manifest_sha256")
        == parent_configs["apartment"]["occlusion_target_manifest_sha256"]
        and manifest.get("evaluation_checkpoint_frames_sha256")
        == parent_configs["apartment"]["evaluation_checkpoint_frames_sha256"]
    ):
        raise ValueError("ablation target or checkpoint plan binding mismatch")

    scenes = manifest.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(SCENES):
        raise ValueError("ablation manifest scene set is invalid")
    configs: dict[str, dict[str, Any]] = {}
    witnesses: list[tuple[Path, bytes, str]] = [(path, raw, "ablation manifest")]
    for scene in SCENES:
        record = scenes[scene]
        if not isinstance(record, Mapping) or set(record) != {
            "parent_config",
            "ablation_config",
            "invariant_config_sha256",
        }:
            raise ValueError(f"ablation {scene} scene record is invalid")
        if record["parent_config"] != parent["scenes"][scene]["frozen_config"]:
            raise ValueError(f"ablation {scene} parent config binding mismatch")
        binding = record["ablation_config"]
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "path_base",
            "sha256",
            "byte_count",
        } or binding.get("path_base") != "manifest":
            raise ValueError(f"ablation {scene} config binding is invalid")
        config_path = path.parent / _relative_path(
            binding.get("path"), f"ablation {scene} config"
        )
        config, config_raw = _load_json(config_path, f"ablation {scene} config")
        if not (
            binding.get("sha256") == _sha256_bytes(config_raw)
            and type(binding.get("byte_count")) is int
            and binding.get("byte_count") == len(config_raw)
        ):
            raise ValueError(f"ablation {scene} config hash binding mismatch")
        parent_config = parent_configs[scene]
        changed = {
            key
            for key in parent_config
            if parent_config.get(key) != config.get(key)
        }
        if set(config) != set(parent_config) or changed != MUTATED_FIELDS:
            raise ValueError(f"ablation {scene} config changed frozen inputs or maintenance")
        if not (
            config.get("missing_observation_policy") == "missing_as_absence"
            and config.get("algorithm_hash") == canonical_algorithm_hash(config)
        ):
            raise ValueError(f"ablation {scene} algorithm_hash or policy is invalid")
        invariant_hash = _json_hash(
            {key: value for key, value in config.items() if key not in MUTATED_FIELDS}
        )
        if record.get("invariant_config_sha256") != invariant_hash:
            raise ValueError(f"ablation {scene} invariant config binding mismatch")
        configs[scene] = config
        witnesses.append((config_path, config_raw, f"ablation {scene} config"))
    ablation_hashes = {config["algorithm_hash"] for config in configs.values()}
    if len(ablation_hashes) != 1:
        raise ValueError("ablation algorithm hash differs across scenes")
    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, Mapping) or algorithm != {
        "parent_sha256": parent_configs["apartment"]["algorithm_hash"],
        "ablation_sha256": next(iter(ablation_hashes)),
        "normalized_config": canonical_algorithm_config(configs["apartment"]),
    }:
        raise ValueError("ablation algorithm binding mismatch")
    return manifest, raw, configs, witnesses


def _validate_result(payload: Mapping[str, Any], *, policy: str, role: str) -> None:
    if set(payload) != _RESULT_FIELDS or not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id")
        == "oviv2_tesse_cd_occlusion_evaluation_v1"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("method_id") == "OVIV2"
        and payload.get("missing_observation_policy") == policy
        and type(payload.get("checkpoint_count")) is int
        and payload["checkpoint_count"] > 0
        and payload.get("fixed_anchor_mapping_rule")
        == "majority_owner_then_lowest_entity_id"
        and isinstance(payload.get("fixed_anchor_mappings"), list)
    ):
        raise ValueError(f"{role} identity or policy is invalid")
    _binding_record(payload.get("target_manifest"), f"{role} target")
    if not _is_sha256(payload.get("evaluation_checkpoint_frames_sha256")):
        raise ValueError(f"{role} checkpoint plan hash is invalid")
    if not (
        isinstance(payload.get("checkpoint_index"), list)
        and len(payload["checkpoint_index"]) == 2
        and isinstance(payload.get("run_config"), list)
        and len(payload["run_config"]) == 2
    ):
        raise ValueError(f"{role} must bind two scene indexes and run configs")
    for index, record in enumerate(payload["checkpoint_index"]):
        _binding_record(record, f"{role} checkpoint index {index}")
    for index, record in enumerate(payload["run_config"]):
        _binding_record(record, f"{role} run config {index}")

    headline = payload.get("headline_gate")
    if not isinstance(headline, Mapping):
        raise ValueError(f"{role} headline gate is invalid")
    coverage = headline.get("scene_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != set(SCENES):
        raise ValueError(f"{role} scene coverage is invalid")
    both_scenes_covered = True
    for scene in SCENES:
        item = coverage[scene]
        if not isinstance(item, Mapping) or set(item) != {
            "episode_count",
            "anchor_mapped_episode_count",
        }:
            raise ValueError(f"{role} scene coverage is invalid")
        episode_count = item["episode_count"]
        mapped_count = item["anchor_mapped_episode_count"]
        if not (
            type(episode_count) is int
            and episode_count > 0
            and type(mapped_count) is int
            and mapped_count == episode_count
        ):
            both_scenes_covered = False
    if not both_scenes_covered:
        raise ValueError(f"{role} scene coverage is incomplete")

    integer_fields = (
        "episode_count",
        "anchor_mapped_episode_count",
        "anchor_owned_target_voxels",
        "false_release_count",
        "false_reassignment_count",
    )
    if any(
        type(headline.get(field)) is not int or headline[field] < 0
        for field in integer_fields
    ):
        raise ValueError(f"{role} headline counts are invalid")
    recall = headline.get("retained_ownership_recall")
    if not (
        type(recall) in (int, float)
        and math.isfinite(float(recall))
        and 0.0 <= float(recall) <= 1.0
    ):
        raise ValueError(f"{role} retained ownership recall is invalid")
    if not (
        headline.get("stress_layer") == "0.90"
        and headline.get("missing_observation_policy") == policy
        and type(headline.get("passed")) is bool
    ):
        raise ValueError(f"{role} headline identity is invalid")
    expected_pass = bool(
        policy == "signed_depth"
        and headline["anchor_owned_target_voxels"] > 0
        and headline["anchor_mapped_episode_count"] == headline["episode_count"]
        and headline["false_release_count"] == 0
        and headline["false_reassignment_count"] == 0
        and float(recall) == 1.0
    )
    if headline["passed"] is not expected_pass:
        raise ValueError(f"{role} headline gate is internally inconsistent")
    layers = payload.get("stress_layers")
    layer = layers.get("0.90") if isinstance(layers, Mapping) else None
    if not isinstance(layer, Mapping) or any(
        layer.get(field) != headline[field]
        for field in (*integer_fields, "retained_ownership_recall")
    ):
        raise ValueError(f"{role} headline does not match stress layer 0.90")


def _load_bundle(
    paths: Sequence[str | Path],
    *,
    expected_configs: Mapping[str, Mapping[str, Any]],
    expected_policy: str,
    result: Mapping[str, Any],
    role: str,
    expected_frozen_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[
    tuple[Path, Path],
    list[dict[str, int | str]],
    list[dict[str, int | str]],
    list[tuple[Path, bytes, str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if len(paths) != 2:
        raise ValueError(f"{role} requires exactly two checkpoint indexes")
    loaded: dict[str, tuple[Path, dict[str, Any], bytes, Path, bytes]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        index, index_raw = _load_json(path, f"{role} checkpoint index")
        expected_fields = (
            _FORMAL_INDEX_FIELDS
            if expected_frozen_identities is not None
            else _BASE_INDEX_FIELDS
        )
        if (
            expected_frozen_identities is None
            and set(index) == _FORMAL_INDEX_FIELDS
        ):
            raise ValueError(
                f"{role} {index.get('scene')} run config does not match its frozen config"
            )
        if set(index) != expected_fields or not (
            type(index.get("schema_version")) is int
            and index.get("schema_version") == 2
            and index.get("manifest_id")
            == "oviv2_tesse_cd_occlusion_checkpoints_v1"
            and index.get("dataset") == "TESSE-CD"
            and index.get("method_id") == "OVIV2"
            and isinstance(index.get("snapshots"), list)
        ):
            raise ValueError(f"{role} checkpoint index identity is invalid")
        scene = index.get("scene")
        if scene not in SCENES or scene in loaded:
            raise ValueError(f"{role} checkpoint indexes must cover both scenes exactly")
        if expected_frozen_identities is not None:
            frozen_identity = index.get("frozen_run_identity")
            if not isinstance(frozen_identity, Mapping) or _canonical_bytes(
                frozen_identity
            ) != _canonical_bytes(expected_frozen_identities[scene]):
                raise ValueError(f"{role} {scene} frozen run identity mismatch")
            execution = index.get("run_execution")
            if not isinstance(execution, Mapping) or set(execution) != _RUN_EXECUTION_FIELDS:
                raise ValueError(f"{role} {scene} run execution identity is invalid")
            output_root = Path(os.path.abspath(path.parent))
            status = os.stat(output_root, follow_symlinks=False)
            execution_base = {
                field: execution.get(field)
                for field in _RUN_EXECUTION_FIELDS
                if field != "execution_id"
            }
            if not (
                type(execution.get("schema_version")) is int
                and execution.get("schema_version") == 1
                and execution.get("run_slot")
                in {f"{scene}_run1", f"{scene}_run2"}
                and execution.get("output_root") == os.fspath(output_root)
                and type(execution.get("root_device")) is int
                and execution.get("root_device") == status.st_dev
                and type(execution.get("root_inode")) is int
                and execution.get("root_inode") == status.st_ino
                and execution.get("execution_id") == _json_hash(execution_base)
            ):
                raise ValueError(f"{role} {scene} run execution identity mismatch")
        run_binding = index.get("run_config")
        if not isinstance(run_binding, Mapping) or set(run_binding) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"{role} {scene} run config binding is invalid")
        config_path = path.parent / _relative_path(
            run_binding.get("path"), f"{role} {scene} run config"
        )
        config, config_raw = _load_json(config_path, f"{role} {scene} run config")
        if not (
            run_binding.get("sha256") == _sha256_bytes(config_raw)
            and type(run_binding.get("byte_count")) is int
            and run_binding.get("byte_count") == len(config_raw)
            and config == expected_configs[scene]
            and config.get("missing_observation_policy") == expected_policy
            and config.get("algorithm_hash") == canonical_algorithm_hash(config)
            and index.get("algorithm_hash") == config["algorithm_hash"]
        ):
            raise ValueError(f"{role} {scene} run config does not match its frozen config")
        target = _binding_record(index.get("target_manifest"), f"{role} {scene} target")
        if target != result["target_manifest"]:
            raise ValueError(f"{role} {scene} target binding differs from result")
        if (
            index.get("evaluation_checkpoint_frames_sha256")
            != result["evaluation_checkpoint_frames_sha256"]
        ):
            raise ValueError(f"{role} {scene} checkpoint plan differs from result")
        loaded[scene] = (path, index, index_raw, config_path, config_raw)

    ordered = tuple(loaded[scene][0] for scene in SCENES)
    index_records = [_content_record(loaded[scene][2]) for scene in SCENES]
    config_records = [_content_record(loaded[scene][4]) for scene in SCENES]
    if result["checkpoint_index"] != index_records:
        raise ValueError(f"{role} result checkpoint bundle hash mismatch")
    if result["run_config"] != config_records:
        raise ValueError(f"{role} result run config hash mismatch")
    witnesses = [
        witness
        for scene in SCENES
        for witness in (
            (loaded[scene][0], loaded[scene][2], f"{role} {scene} checkpoint index"),
            (loaded[scene][3], loaded[scene][4], f"{role} {scene} run config"),
        )
    ]
    frozen_identities = {
        scene: dict(loaded[scene][1]["frozen_run_identity"])
        for scene in SCENES
        if "frozen_run_identity" in loaded[scene][1]
    }
    run_executions = {
        scene: dict(loaded[scene][1]["run_execution"])
        for scene in SCENES
        if "run_execution" in loaded[scene][1]
    }
    return (
        ordered,
        index_records,
        config_records,
        witnesses,
        frozen_identities,
        run_executions,
    )


def _headline_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    gate = result["headline_gate"]
    return {
        "stress_layer": gate["stress_layer"],
        "episode_count": gate["episode_count"],
        "anchor_mapped_episode_count": gate["anchor_mapped_episode_count"],
        "anchor_owned_target_voxels": gate["anchor_owned_target_voxels"],
        "false_release_count": gate["false_release_count"],
        "false_reassignment_count": gate["false_reassignment_count"],
        "retained_ownership_recall": gate["retained_ownership_recall"],
        "scene_coverage": gate["scene_coverage"],
        "passed": gate["passed"],
    }


def _expected_frozen_run_identities(
    parent: Mapping[str, Any],
    *,
    parent_raw: bytes,
    parent_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    shared = parent.get("shared_bindings")
    scenes = parent.get("scenes")
    repository = parent.get("repository")
    if not all(isinstance(value, Mapping) for value in (shared, scenes, repository)):
        raise ValueError("parent freeze identity bindings are invalid")
    assert isinstance(shared, Mapping)
    assert isinstance(scenes, Mapping)
    assert isinstance(repository, Mapping)
    expected: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        scene_binding = scenes.get(scene)
        if not isinstance(scene_binding, Mapping):
            raise ValueError(f"parent {scene} freeze identity binding is invalid")
        config = scene_binding.get("frozen_config")
        if not isinstance(config, Mapping):
            raise ValueError(f"parent {scene} frozen config binding is invalid")
        expected[scene] = {
            "schema_version": 1,
            "freeze_id": "oviv2-tessecd-v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "freeze_manifest": _content_record(parent_raw),
            "repository": {
                "commit": repository["commit"],
                "tree": repository["tree"],
            },
            "config": _binding_record(
                {
                    "sha256": config.get("sha256"),
                    "byte_count": config.get("byte_count"),
                },
                f"parent {scene} frozen config",
            ),
            "algorithm_hash": parent_configs[scene]["algorithm_hash"],
            "missing_observation_policy": "signed_depth",
            "input_bindings_sha256": _json_hash(
                {
                    "shared_bindings": dict(shared),
                    "scene": dict(scene_binding),
                }
            ),
        }
    return expected


def finalize_occlusion_ablation(
    *,
    parent_freeze: str | Path,
    ablation_manifest: str | Path,
    targets: str | Path,
    dataset_root: str | Path,
    signed_result: str | Path,
    signed_checkpoints: Sequence[str | Path],
    ablation_result: str | Path,
    ablation_checkpoints: Sequence[str | Path],
    output_path: str | Path,
    reevaluate: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Revalidate both bundles and publish claim eligibility with a fixed scope."""
    output = Path(output_path)
    if os.path.lexists(output):
        raise ValueError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")

    parent_path = Path(parent_freeze)
    parent, parent_raw = _load_json(parent_path, "parent freeze manifest")
    parent_configs = _validate_parent(parent, manifest_path=parent_path)
    ablation_path = Path(ablation_manifest)
    (
        ablation,
        ablation_raw,
        ablation_configs,
        ablation_witnesses,
    ) = _load_ablation_manifest(
        ablation_path,
        parent=parent,
        parent_raw=parent_raw,
        parent_configs=parent_configs,
    )
    signed_result_path = Path(signed_result)
    ablation_result_path = Path(ablation_result)
    signed, signed_raw = _load_result(signed_result_path, "signed-depth result")
    missing, missing_raw = _load_result(
        ablation_result_path, "missing-as-absence result"
    )
    _validate_result(signed, policy="signed_depth", role="signed-depth result")
    _validate_result(
        missing,
        policy="missing_as_absence",
        role="missing-as-absence result",
    )
    expected_frozen_identities = _expected_frozen_run_identities(
        parent,
        parent_raw=parent_raw,
        parent_configs=parent_configs,
    )
    (
        signed_paths,
        signed_indexes,
        signed_configs,
        signed_witnesses,
        signed_frozen_identities,
        signed_run_executions,
    ) = _load_bundle(
        signed_checkpoints,
        expected_configs=parent_configs,
        expected_policy="signed_depth",
        result=signed,
        role="signed-depth bundle",
        expected_frozen_identities=expected_frozen_identities,
    )
    (
        missing_paths,
        missing_indexes,
        missing_configs,
        missing_witnesses,
        missing_frozen_identities,
        missing_run_executions,
    ) = _load_bundle(
        ablation_checkpoints,
        expected_configs=ablation_configs,
        expected_policy="missing_as_absence",
        result=missing,
        role="missing-as-absence bundle",
    )
    if missing_frozen_identities or missing_run_executions:
        raise ValueError("missing-as-absence bundle cannot use the signed frozen identity")

    if signed["target_manifest"] != missing["target_manifest"]:
        raise ValueError("signed and ablation target bindings differ")
    if (
        signed["target_manifest"]["sha256"]
        != ablation["occlusion_target_manifest_sha256"]
    ):
        raise ValueError("result target does not match the parent-frozen target")
    if (
        signed["evaluation_checkpoint_frames_sha256"]
        != missing["evaluation_checkpoint_frames_sha256"]
    ):
        raise ValueError("signed and ablation checkpoint plan bindings differ")
    if (
        signed["evaluation_checkpoint_frames_sha256"]
        != ablation["evaluation_checkpoint_frames_sha256"]
    ):
        raise ValueError("result checkpoint plan does not match the parent-frozen plan")
    if not set(item["sha256"] for item in signed_indexes).isdisjoint(
        item["sha256"] for item in missing_indexes
    ):
        raise ValueError("signed and ablation checkpoint bundles are not independent")
    if not set(item["sha256"] for item in signed_configs).isdisjoint(
        item["sha256"] for item in missing_configs
    ):
        raise ValueError("signed and ablation run configs are not independent")
    if _sha256_bytes(signed_raw) == _sha256_bytes(missing_raw):
        raise ValueError("signed and ablation result artifacts are not independent")

    evaluator = reevaluate or evaluate_occlusion_package
    fresh_signed = evaluator(
        target_dir=Path(targets),
        checkpoint_index=signed_paths,
        dataset_root=Path(dataset_root),
    )
    fresh_missing = evaluator(
        target_dir=Path(targets),
        checkpoint_index=missing_paths,
        dataset_root=Path(dataset_root),
    )
    if _result_bytes(fresh_signed) != signed_raw:
        raise ValueError("signed-depth result does not match fresh evaluator output")
    if _result_bytes(fresh_missing) != missing_raw:
        raise ValueError("missing-as-absence result does not match fresh evaluator output")

    signed_headline = signed["headline_gate"]
    missing_headline = missing["headline_gate"]
    false_release_worse = (
        missing_headline["false_release_count"]
        > signed_headline["false_release_count"]
    )
    retained_recall_lower = float(
        missing_headline["retained_ownership_recall"]
    ) < float(signed_headline["retained_ownership_recall"])
    eligible = bool(
        signed_headline["passed"]
        and (false_release_worse or retained_recall_lower)
    )
    supported_outcomes = [
        name
        for name, supported in (
            ("false_release_count", false_release_worse),
            ("retained_ownership_recall", retained_recall_lower),
        )
        if supported
    ]
    allowed_claims = (
        [
            {
                "claim_id": "tesse_cd_occlusion_retention_vs_missing_as_absence_v1",
                "scope": {
                    "dataset": "TESSE-CD",
                    "scenes": list(SCENES),
                    "stress_layer": "0.90",
                    "headline_policy": "signed_depth",
                    "comparator_policy": "missing_as_absence",
                },
                "supported_outcomes": supported_outcomes,
            }
        ]
        if eligible
        else []
    )
    final = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_occlusion_ablation_final_v1",
        "status": "PASS",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "parent_freeze": {
            **_content_record(parent_raw),
            "freeze_id": parent["freeze_id"],
            "repository_commit": parent["repository"]["commit"],
            "algorithm_hash": parent_configs["apartment"]["algorithm_hash"],
        },
        "ablation_manifest": {
            **_content_record(ablation_raw),
            "manifest_id": ablation["manifest_id"],
            "algorithm_hash": ablation["algorithm"]["ablation_sha256"],
            "parent_freeze_sha256": ablation["parent_freeze"]["sha256"],
        },
        "target_manifest": signed["target_manifest"],
        "evaluation_checkpoint_frames_sha256": signed[
            "evaluation_checkpoint_frames_sha256"
        ],
        "bundles": {
            "signed_depth": {
                "algorithm_hash": parent_configs["apartment"]["algorithm_hash"],
                "result": _content_record(signed_raw),
                "checkpoint_indexes": signed_indexes,
                "run_configs": signed_configs,
                "frozen_run_identities": signed_frozen_identities,
                "run_executions": signed_run_executions,
            },
            "missing_as_absence": {
                "algorithm_hash": ablation["algorithm"]["ablation_sha256"],
                "result": _content_record(missing_raw),
                "checkpoint_indexes": missing_indexes,
                "run_configs": missing_configs,
            },
        },
        "headline_comparison": {
            "signed_depth": _headline_projection(signed),
            "missing_as_absence": _headline_projection(missing),
            "directional_differences": {
                "ablation_minus_signed_false_release_count": (
                    missing_headline["false_release_count"]
                    - signed_headline["false_release_count"]
                ),
                "signed_minus_ablation_retained_ownership_recall": (
                    float(signed_headline["retained_ownership_recall"])
                    - float(missing_headline["retained_ownership_recall"])
                ),
            },
        },
        "claim_eligibility": {
            "eligible": eligible,
            "criteria": {
                "signed_headline_gate_passed": signed_headline["passed"],
                "ablation_false_release_count_greater": false_release_worse,
                "ablation_retained_ownership_recall_lower": retained_recall_lower,
            },
            "allowed_claims": allowed_claims,
        },
    }

    witnesses = [
        (parent_path, parent_raw, "parent freeze manifest"),
        (signed_result_path, signed_raw, "signed-depth result"),
        (ablation_result_path, missing_raw, "missing-as-absence result"),
        *ablation_witnesses,
        *signed_witnesses,
        *missing_witnesses,
    ]
    for path, raw, role in witnesses:
        if _read_regular_file(path, role) != raw:
            raise ValueError(f"{role} changed during finalization")
    _write_exclusive(output, _canonical_bytes(final))
    directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-freeze", type=Path, required=True)
    parser.add_argument("--ablation-manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--signed-result", type=Path, required=True)
    parser.add_argument(
        "--signed-checkpoints", type=Path, action="append", required=True
    )
    parser.add_argument("--ablation-result", type=Path, required=True)
    parser.add_argument(
        "--ablation-checkpoints", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    final = finalize_occlusion_ablation(
        parent_freeze=args.parent_freeze,
        ablation_manifest=args.ablation_manifest,
        targets=args.targets,
        dataset_root=args.dataset_root,
        signed_result=args.signed_result,
        signed_checkpoints=args.signed_checkpoints,
        ablation_result=args.ablation_result,
        ablation_checkpoints=args.ablation_checkpoints,
        output_path=args.output,
    )
    print(json.dumps(final["claim_eligibility"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
