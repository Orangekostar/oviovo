#!/usr/bin/env python3
"""Finalize OVIV2 common-v2 metrics from canonical independent repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import finalize_tesse_common_v2 as pinned_common
from scripts.evaluation.canonicalize_tesse_common_v2_summary import (
    CapturedArtifact,
    EXTERNAL_SOURCE_ROLES,
    FileIdentity,
    canonical_summary_bytes,
    capture_and_canonicalize_summary,
    _stable_regular_file_with_identity,
)
from scripts.evaluation.finalize_tesse_t2 import (
    _absolute_lexical,
    _atomic_json_no_replace,
    _open_directory_no_symlinks,
    _stable_regular_file,
)
from scripts.evaluation.freeze_oviv2_tesse_cd import (
    STAGE3_LINEAGE_COMMIT,
    _algorithm_config,
    _algorithm_hash,
    _validate_repository,
)
from src.evaluation.json_contracts import loads_strict


SCENES = ("apartment", "office")
REPEATS = (1, 2)
_REQUIRED_SHARED_BINDINGS = frozenset(
    {
        "common_target_manifest",
        "common_target_arrays",
        "schedule",
        "alias_map",
        "evaluator",
        "finalizers",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, object]]:
    digest, byte_count, content = _stable_regular_file(path, label=label, capture=True)
    assert content is not None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    payload = loads_strict(text, label=label)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(payload), {
        "role": "freeze_manifest",
        "sha256": digest,
        "byte_count": byte_count,
    }


def _verified_binding(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    path, record, _ = _verified_binding_data(
        value,
        label=label,
        expected_path=expected_path,
        capture=False,
    )
    return path, record


def _verified_binding_data(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
    capture: bool,
) -> tuple[Path, dict[str, object], bytes | None]:
    declaration = _mapping(value, label=f"{label} binding")
    if set(declaration) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} binding fields are invalid")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = _absolute_lexical(Path(raw_path))
    if raw_path != os.fspath(path):
        raise ValueError(f"{label} path must be canonical")
    if expected_path is not None and path != _absolute_lexical(expected_path):
        raise ValueError(f"{label} path mismatch")
    digest, byte_count, content = _stable_regular_file(
        path, label=f"{label} file", capture=capture
    )
    if (
        not isinstance(declaration.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(declaration.get("sha256")))
        or type(declaration.get("byte_count")) is not int
        or declaration.get("sha256") != digest
        or declaration.get("byte_count") != byte_count
    ):
        raise ValueError(f"{label} content mismatch")
    return (
        path,
        {
            "role": label,
            "sha256": digest,
            "byte_count": byte_count,
        },
        content,
    )


def _direct_root(path: Path, *, label: str) -> tuple[Path, FileIdentity]:
    absolute = _absolute_lexical(path)
    descriptor, _ = _open_directory_no_symlinks(absolute, label=label)
    try:
        status = os.fstat(descriptor)
        identity = (status.st_dev, status.st_ino)
    finally:
        os.close(descriptor)
    return absolute, identity


def _canonical_record(payload: Mapping[str, Any], *, role: str) -> dict[str, object]:
    data = canonical_summary_bytes(payload)
    return {
        "role": role,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _compact_json_hash(payload: Mapping[str, Any]) -> str:
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _captured_json_artifact(
    path: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, object], FileIdentity]:
    digest, byte_count, content, identity = _stable_regular_file_with_identity(
        path, label=label, capture=True
    )
    assert content is not None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    payload = loads_strict(text, label=label)
    return (
        dict(_mapping(payload, label=label)),
        {"sha256": digest, "byte_count": byte_count},
        identity,
    )


def _validate_run_execution(
    value: object,
    *,
    scene: str,
    repeat: int,
    root: Path,
    root_identity: FileIdentity,
) -> dict[str, Any]:
    execution = dict(_mapping(value, label=f"{scene}.run{repeat} execution"))
    base = {
        "schema_version": 1,
        "run_slot": f"{scene}_run{repeat}",
        "output_root": os.fspath(root),
        "root_device": root_identity[0],
        "root_inode": root_identity[1],
    }
    expected = {**base, "execution_id": _compact_json_hash(base)}
    if execution != expected:
        raise ValueError(f"{scene}.run{repeat} execution identity mismatch")
    return execution


def _validate_captured_summary(
    payload: Mapping[str, Any], *, scene: str
) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "tesse_cd_common_v2_scene_summary"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("protocol") == "tesse_cd_common_v2"
        and payload.get("status") == "PASS"
        and payload.get("method") == "OVIV2"
        and payload.get("mode") == pinned_common.METHODS["OVIV2"]["summary_mode"]
        and payload.get("scene") == scene
    ):
        raise ValueError(f"{scene} scene summary identity mismatch")
    metrics = _mapping(payload.get("metrics"), label=f"{scene} metrics")
    for name in pinned_common.METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{scene}.{name} must be numeric")
        number = float(value)
        upper = 450.0 if name == "recovery_frames" else 1.0
        if not math.isfinite(number) or not 0.0 <= number <= upper:
            raise ValueError(f"{scene}.{name} is outside its valid range")
    pinned_common._validate_metric_details(metrics, scene=scene)
    sources = _mapping(payload.get("sources"), label=f"{scene} sources")
    if not pinned_common.REQUIRED_SUMMARY_SOURCES <= set(sources):
        raise ValueError(f"{scene} summary source coverage is incomplete")


def _target_declared_record(
    value: object,
    *,
    label: str,
    cache: dict[Path, dict[str, object]],
    base: Path | None = None,
) -> dict[str, object]:
    declaration = _mapping(value, label=f"{label} declaration")
    raw = Path(str(declaration.get("path", "")))
    path = _absolute_lexical(
        raw if raw.is_absolute() else (base / raw if base is not None else raw)
    )
    record = cache.get(path)
    if record is None:
        digest, byte_count, _ = _stable_regular_file(
            path, label=label, capture=False
        )
        record = {
            "path": os.fspath(path),
            "sha256": digest,
            "byte_count": byte_count,
        }
        cache[path] = record
    if (
        declaration.get("sha256") != record["sha256"]
        or declaration.get("byte_count") != record["byte_count"]
    ):
        raise ValueError(f"{label} hash mismatch")
    return dict(record)


def _validate_captured_target_package(
    content: bytes,
    *,
    manifest_path: Path,
    arrays_path: Path,
    arrays_record: Mapping[str, object],
    schedule_path: Path,
    schedule_record: Mapping[str, object],
) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("common-v2 target package is not UTF-8") from error
    payload = loads_strict(text, label="common-v2 target package")
    target = _mapping(payload, label="target package")
    metadata = _mapping(target.get("metadata"), label="target metadata")
    if not (
        target.get("schema_version") == 1
        and target.get("manifest_id") == "tesse_cd_common_v2_targets"
        and target.get("dataset") == "TESSE-CD"
        and target.get("status") == "GENERATED"
        and target.get("targets_generated") is True
        and target.get("prediction_inputs_used") is False
        and metadata.get("prediction_inputs_used") is False
        and metadata.get("protocol_complete") is True
        and metadata.get("window_frames") == 450
        and metadata.get("voxel_size_m") == 0.05
        and set(metadata.get("scenes", ())) == set(SCENES)
    ):
        raise ValueError("target package is not complete and prediction-independent")

    cache = {
        arrays_path: {
            "path": os.fspath(arrays_path),
            "sha256": arrays_record["sha256"],
            "byte_count": arrays_record["byte_count"],
        },
        schedule_path: {
            "path": os.fspath(schedule_path),
            "sha256": schedule_record["sha256"],
            "byte_count": schedule_record["byte_count"],
        },
    }
    observed_arrays = _target_declared_record(
        target.get("target_arrays"),
        label="target arrays",
        cache=cache,
        base=manifest_path.parent,
    )
    if observed_arrays != cache[arrays_path]:
        raise ValueError("scene summaries do not bind the target arrays")

    target_sources = target.get("sources")
    if not isinstance(target_sources, list) or not target_sources:
        raise ValueError("target package sources must be non-empty")
    validated_sources = [
        _target_declared_record(
            declaration,
            label="target source",
            cache=cache,
        )
        for declaration in target_sources
    ]
    _target_declared_record(
        metadata.get("source_manifest"),
        label="target source manifest",
        cache=cache,
    )
    declared_schedule = _target_declared_record(
        metadata.get("schedule"),
        label="target schedule",
        cache=cache,
    )
    if declared_schedule != cache[schedule_path]:
        raise ValueError("target package schedule binding mismatch")
    declared_sources = _mapping(
        metadata.get("declared_source_records"),
        label="target declared source records",
    )
    if not declared_sources:
        raise ValueError("target declared source records must be non-empty")
    validated_declared_sources = {
        str(label): _target_declared_record(
            declaration,
            label="target source",
            cache=cache,
        )
        for label, declaration in declared_sources.items()
    }

    observability = _mapping(
        metadata.get("background_observable_by_event"),
        label="target observability",
    )
    if not observability or any(
        type(value) is not bool for value in observability.values()
    ):
        raise ValueError(
            "target observability must contain prediction-independent booleans"
        )
    arrays = _mapping(
        _mapping(target.get("target_arrays"), label="target arrays").get("arrays"),
        label="target array declarations",
    )
    revealed = {
        str(name).removesuffix(".revealed_background"): _mapping(
            declaration, label=f"target array {name}"
        )
        for name, declaration in arrays.items()
        if str(name).endswith(".revealed_background")
    }
    if set(revealed) != set(observability) or any(
        bool(
            pinned_common._nonnegative_int(
                declaration.get("element_count"),
                label=f"{event_id} target elements",
            )
        )
        is not observability[event_id]
        for event_id, declaration in revealed.items()
    ):
        raise ValueError("target observability disagrees with target arrays")
    observable_count = sum(observability.values())
    if (
        metadata.get("background_observable_event_count") != observable_count
        or metadata.get("unobservable_revealed_target_event_count")
        != len(observability) - observable_count
        or any(
            not any(
                value
                for event_id, value in observability.items()
                if str(event_id).startswith(f"{scene}_event_")
            )
            for scene in SCENES
        )
    ):
        raise ValueError("target observability counts are inconsistent")
    return {
        "target_arrays": cache[arrays_path],
        "prediction_inputs_used": False,
        "background_observable_by_event": dict(observability),
        "sources": validated_sources,
        "declared_source_records": validated_declared_sources,
    }


def finalize_oviv2_common_v2_release(
    freeze_manifest: Path,
    *,
    run_id: str,
    output: Path,
) -> Path:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    freeze, freeze_record = _read_json(freeze_manifest, label="freeze manifest")
    if not (
        freeze.get("schema_version") == 1
        and freeze.get("freeze_id") == "oviv2-tessecd-v1"
        and freeze.get("status") == "FROZEN"
        and freeze.get("method") == "OVIV2"
        and freeze.get("dataset") == "TESSE-CD"
    ):
        raise ValueError("freeze manifest identity mismatch")
    repository = _validate_repository(
        _mapping(freeze.get("repository"), label="freeze repository")
    )
    if repository["stage3_lineage_commit"] != STAGE3_LINEAGE_COMMIT:
        raise ValueError("freeze manifest Stage3 lineage mismatch")
    algorithm = _mapping(freeze.get("algorithm"), label="freeze algorithm")
    if not re.fullmatch(r"[0-9a-f]{64}", str(algorithm.get("sha256", ""))):
        raise ValueError("freeze algorithm hash is invalid")

    shared = _mapping(freeze.get("shared_bindings"), label="shared bindings")
    if not _REQUIRED_SHARED_BINDINGS <= set(shared):
        raise ValueError("freeze shared bindings are incomplete")
    target_manifest_path, target_manifest_record, target_manifest_content = (
        _verified_binding_data(
            shared["common_target_manifest"],
            label="common_target_manifest",
            capture=True,
        )
    )
    assert target_manifest_content is not None
    target_arrays_path, target_arrays_record = _verified_binding(
        shared["common_target_arrays"], label="common_target_arrays"
    )
    schedule_path, schedule_record = _verified_binding(
        shared["schedule"], label="schedule"
    )
    aliases_path, aliases_record = _verified_binding(
        shared["alias_map"], label="alias_map"
    )
    evaluator_path, evaluator_record = _verified_binding(
        shared["evaluator"],
        label="evaluator",
        expected_path=ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py",
    )
    finalizers = _mapping(shared["finalizers"], label="frozen finalizers")
    if set(finalizers) != {"common_v2", "official_t2"}:
        raise ValueError("frozen finalizer bindings are incomplete")
    _, common_finalizer_record = _verified_binding(
        finalizers["common_v2"],
        label="common_v2_finalizer",
        expected_path=ROOT / "scripts/evaluation/finalize_tesse_common_v2.py",
    )
    _, official_finalizer_record = _verified_binding(
        finalizers["official_t2"],
        label="official_t2_finalizer",
        expected_path=ROOT / "scripts/evaluation/finalize_tesse_t2.py",
    )

    release_bindings = _mapping(
        freeze.get("release_bindings"), label="release bindings"
    )
    if set(release_bindings) != {
        "canonical_summary_generator",
        "release_finalizer",
        "label_spaces",
    }:
        raise ValueError("freeze release bindings are incomplete")
    _, canonicalizer_record = _verified_binding(
        release_bindings["canonical_summary_generator"],
        label="canonical_summary_generator",
        expected_path=ROOT
        / "scripts/evaluation/canonicalize_tesse_common_v2_summary.py",
    )
    _, release_finalizer_record = _verified_binding(
        release_bindings["release_finalizer"],
        label="release_finalizer",
        expected_path=Path(__file__),
    )
    label_bindings = _mapping(
        release_bindings["label_spaces"], label="label space bindings"
    )
    if set(label_bindings) != set(SCENES):
        raise ValueError("label space bindings must cover Apartment and Office")
    label_spaces: dict[str, Path] = {}
    label_records: dict[str, dict[str, object]] = {}
    for scene in SCENES:
        label_spaces[scene], label_records[scene] = _verified_binding(
            label_bindings[scene], label=f"{scene}_label_space"
        )

    scene_bindings = _mapping(freeze.get("scenes"), label="freeze scenes")
    if set(scene_bindings) != set(SCENES):
        raise ValueError("freeze scenes must cover Apartment and Office")
    frozen_config_records: dict[str, dict[str, object]] = {}
    expected_run_identities: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        selected = _mapping(
            scene_bindings[scene], label=f"{scene} freeze scene binding"
        )
        _, config_record, config_content = _verified_binding_data(
            selected.get("frozen_config"),
            label=f"{scene}_frozen_config",
            capture=True,
        )
        assert config_content is not None
        try:
            config_text = config_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{scene} frozen config is not UTF-8") from error
        config = _mapping(
            loads_strict(config_text, label=f"{scene} frozen config"),
            label=f"{scene} frozen config",
        )
        if not (
            config.get("missing_observation_policy") == "signed_depth"
            and config.get("algorithm_hash") == algorithm["sha256"]
            and _algorithm_hash(config) == algorithm["sha256"]
            and _algorithm_config(config) == algorithm.get("normalized_config")
        ):
            raise ValueError(f"{scene} frozen config differs from frozen algorithm")
        frozen_config_records[scene] = config_record
        input_bindings = {
            "shared_bindings": dict(shared),
            "scene": dict(selected),
        }
        expected_run_identities[scene] = {
            "schema_version": 1,
            "freeze_id": "oviv2-tessecd-v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "freeze_manifest": {
                "sha256": freeze_record["sha256"],
                "byte_count": freeze_record["byte_count"],
            },
            "repository": {
                "commit": repository["commit"],
                "tree": repository["tree"],
            },
            "config": {
                "sha256": config_record["sha256"],
                "byte_count": config_record["byte_count"],
            },
            "algorithm_hash": algorithm["sha256"],
            "missing_observation_policy": "signed_depth",
            "input_bindings_sha256": _compact_json_hash(input_bindings),
        }

    raw_roots = _mapping(freeze.get("output_roots"), label="output roots")
    expected_root_roles = {
        f"{scene}_run{repeat}" for scene in SCENES for repeat in REPEATS
    }
    if set(raw_roots) != expected_root_roles:
        raise ValueError("freeze output roots do not cover exact independent runs")
    roots: dict[tuple[str, int], Path] = {}
    root_identities: dict[tuple[str, int], FileIdentity] = {}
    for scene in SCENES:
        for repeat in REPEATS:
            role = f"{scene}_run{repeat}"
            raw = raw_roots[role]
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError(f"{role} root must be absolute")
            if raw != os.fspath(_absolute_lexical(Path(raw))):
                raise ValueError(f"{role} root must be canonical")
            roots[(scene, repeat)], root_identities[(scene, repeat)] = _direct_root(
                Path(raw), label=f"{role} root"
            )
    if len(set(roots.values())) != len(roots) or len(
        set(root_identities.values())
    ) != len(root_identities):
        raise ValueError("release runs must use physically independent artifact roots")

    external_common = {
        "target_manifest": target_manifest_path,
        "target_arrays": target_arrays_path,
        "aliases": aliases_path,
        "evaluator": evaluator_path,
    }
    raw_summaries: dict[tuple[str, int], dict[str, Any]] = {}
    canonical: dict[tuple[str, int], dict[str, Any]] = {}
    raw_records: dict[tuple[str, int], dict[str, object]] = {}
    captured_identities: dict[
        tuple[str, int], dict[str, FileIdentity]
    ] = {}
    formal_temporal_artifacts: dict[
        tuple[str, int], dict[str, CapturedArtifact]
    ] = {}
    for scene in SCENES:
        for repeat in REPEATS:
            key = (scene, repeat)
            summary_path = roots[key] / "evaluation/summary.json"
            (
                raw_payload,
                canonical_payload,
                raw_content_record,
                identities,
                temporal_artifacts,
            ) = capture_and_canonicalize_summary(
                summary_path,
                artifact_root=roots[key],
                external_sources={
                    **external_common,
                    "label_space": label_spaces[scene],
                },
            )
            _validate_captured_summary(raw_payload, scene=scene)
            raw_summaries[key] = raw_payload
            canonical[key] = canonical_payload
            raw_records[key] = {
                "role": f"{scene}.run{repeat}.raw_summary",
                **raw_content_record,
            }
            captured_identities[key] = identities
            formal_temporal_artifacts[key] = temporal_artifacts

    formal_run_records: dict[
        tuple[str, int], dict[str, dict[str, object]]
    ] = {}
    run_executions: dict[tuple[str, int], dict[str, Any]] = {}
    root_artifact_paths = {
        "run_manifest": Path("run_manifest.json"),
        "source_index": Path("source_index.json"),
        "occlusion_checkpoint_index": Path("occlusion_checkpoint_index.json"),
    }
    for scene in SCENES:
        for repeat in REPEATS:
            key = (scene, repeat)
            temporal_artifacts = formal_temporal_artifacts[key]
            if set(temporal_artifacts) != {
                "temporal_manifest",
                "temporal_source_index",
            }:
                raise ValueError(
                    f"{scene}.run{repeat} formal temporal identity is missing"
                )
            payloads: dict[str, dict[str, Any]] = {}
            records: dict[str, dict[str, object]] = {}
            for role, relative in root_artifact_paths.items():
                payload, record, identity = _captured_json_artifact(
                    roots[key] / relative,
                    label=f"{scene}.run{repeat} {role}",
                )
                payloads[role] = payload
                records[role] = {
                    "role": f"{scene}.run{repeat}.{role}",
                    **record,
                }
                captured_identities[key][f"formal_{role}"] = identity
            for role, (payload, record) in temporal_artifacts.items():
                payloads[role] = payload
                records[role] = {
                    "role": f"{scene}.run{repeat}.{role}",
                    **record,
                }
            expected_identity = expected_run_identities[scene]
            if any(
                payload.get("frozen_run_identity") != expected_identity
                for payload in payloads.values()
            ):
                raise ValueError(f"{scene}.run{repeat} frozen run identity mismatch")
            execution = _validate_run_execution(
                payloads["run_manifest"].get("run_execution"),
                scene=scene,
                repeat=repeat,
                root=roots[key],
                root_identity=root_identities[key],
            )
            if any(
                payload.get("run_execution") != execution
                for payload in payloads.values()
            ):
                raise ValueError(f"{scene}.run{repeat} execution identity mismatch")
            formal_run_records[key] = records
            run_executions[key] = execution

    if len(
        {execution["execution_id"] for execution in run_executions.values()}
    ) != len(run_executions):
        raise ValueError("release runs must have distinct execution identities")

    observed_run_local_identities: dict[FileIdentity, tuple[str, int, str]] = {}
    for (scene, repeat), identities in captured_identities.items():
        for role, identity in identities.items():
            if role in EXTERNAL_SOURCE_ROLES:
                continue
            previous = observed_run_local_identities.setdefault(
                identity, (scene, repeat, role)
            )
            if previous != (scene, repeat, role):
                raise ValueError(
                    "run-local sources must be independent files: "
                    f"{previous[0]}.run{previous[1]}.{previous[2]} and "
                    f"{scene}.run{repeat}.{role}"
                )

    for scene in SCENES:
        if canonical_summary_bytes(canonical[(scene, 1)]) != canonical_summary_bytes(
            canonical[(scene, 2)]
        ):
            raise ValueError(
                f"{scene} canonical summaries must be byte-identical"
            )
    shared_source_roles = (
        "target_manifest",
        "target_arrays",
        "schedule",
        "aliases",
        "evaluator",
    )
    for role in shared_source_roles:
        apartment_source = canonical[("apartment", 1)]["sources"][role]
        office_source = canonical[("office", 1)]["sources"][role]
        if apartment_source != office_source:
            raise ValueError(f"Apartment and Office {role} bindings differ")
    if canonical[("apartment", 1)]["sources"]["schedule"] != {
        "role": "schedule",
        "sha256": schedule_record["sha256"],
        "byte_count": schedule_record["byte_count"],
    }:
        raise ValueError("scene summaries disagree with frozen schedule")

    target_evidence = _validate_captured_target_package(
        target_manifest_content,
        manifest_path=target_manifest_path,
        arrays_path=target_arrays_path,
        arrays_record=target_arrays_record,
        schedule_path=schedule_path,
        schedule_record=schedule_record,
    )
    observability = _mapping(
        target_evidence.get("background_observable_by_event"),
        label="target observability",
    )
    for scene in SCENES:
        pinned_common._validate_scene_observability(
            raw_summaries[(scene, 1)],
            scene=scene,
            observability=observability,
        )

    scene_metrics = {
        scene: {
            name: float(raw_summaries[(scene, 1)]["metrics"][name])
            for name in pinned_common.METRICS
        }
        for scene in SCENES
    }
    metrics = {
        name: fmean(scene_metrics[scene][name] for scene in SCENES)
        for name in pinned_common.METRICS
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("release metrics must be finite")
    token_bindings = [
        {
            "token": f"T2_OVIV2_{suffix}",
            "json_pointer": f"/metrics/{name}",
            "precision": 3,
        }
        for name, suffix in pinned_common.METRICS.items()
    ]
    method_config = pinned_common.METHODS["OVIV2"]
    result = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_common_v2_release_v1",
        "status": "VERIFIED",
        "run_id": run_id,
        "method": {
            "key": "OVIV2",
            "display_label": method_config["display_label"],
            "mode": method_config["mode"],
            "eligible_for_ranking": method_config["eligible_for_ranking"],
        },
        "dataset": {"name": "TESSE-CD", "splits": ["macro_test"]},
        "protocol": {
            "name": "tesse_cd_common_v2",
            "aggregation": "macro mean over Apartment and Office scene summaries",
            "deterministic_repeat": "canonical-role-content-byte-identical",
            "recovery": "F@5cm >= 0.9 for three checkpoints; 450-frame restricted censor",
        },
        "metrics": metrics,
        "scene_metrics": scene_metrics,
        "scene_details": {
            scene: dict(raw_summaries[(scene, 1)]["metrics"])
            for scene in SCENES
        },
        "scene_summaries": {
            scene: {
                "primary": _canonical_record(
                    canonical[(scene, 1)], role=f"{scene}.run1.canonical_summary"
                ),
                "repeat": _canonical_record(
                    canonical[(scene, 2)], role=f"{scene}.run2.canonical_summary"
                ),
                "raw_primary": raw_records[(scene, 1)],
                "raw_repeat": raw_records[(scene, 2)],
            }
            for scene in SCENES
        },
        "scene_sources": {
            scene: dict(canonical[(scene, 1)]["sources"]) for scene in SCENES
        },
        "formal_run_artifacts": {
            scene: {
                "primary": formal_run_records[(scene, 1)],
                "repeat": formal_run_records[(scene, 2)],
            }
            for scene in SCENES
        },
        "target_package": {
            "manifest": target_manifest_record,
            "target_arrays": target_arrays_record,
            "schedule": schedule_record,
            "prediction_inputs_used": False,
            "background_observable_by_event": dict(observability),
        },
        "frozen_identity": {
            **freeze_record,
            "freeze_id": "oviv2-tessecd-v1",
            "repository_commit": repository["commit"],
            "algorithm_sha256": algorithm["sha256"],
            "scene_run_identities": expected_run_identities,
        },
        "validation_tools": {
            "evaluator": evaluator_record,
            "common_v2_finalizer": common_finalizer_record,
            "official_t2_finalizer": official_finalizer_record,
            "canonical_summary_generator": canonicalizer_record,
            "release_finalizer": release_finalizer_record,
            "aliases": aliases_record,
            "label_spaces": label_records,
            "frozen_configs": frozen_config_records,
        },
        "token_bindings": token_bindings,
        "unavailable_bindings": [],
        "protocol_deviations": [],
    }
    _atomic_json_no_replace(output, result)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    finalize_oviv2_common_v2_release(
        args.freeze_manifest,
        run_id=args.run_id,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
