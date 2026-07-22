#!/usr/bin/env python3
"""Finalize official TESSE-CD T2 metrics with explicit partial availability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping


SCENE_METRICS = ("object_f1", "dynamic_f1", "change_f1")
METHOD_MODES = {
    "OVIMAP_FROZEN": "frozen",
    "CONCEPTGRAPHS_FROZEN": "frozen",
    "DUALMAP": "native",
    "PANOPTIC_SHARED": "composed",
    "KHRONOS_OPEN": "open-set",
    "OVIV2": "causal_checkpoints",
}
TABLE_MODES = {
    **METHOD_MODES,
    "KHRONOS_OPEN": "online",
    "OVIV2": "online",
}
METHOD_DISPLAY_LABELS = {"OVIV2": "OVIV2"}
SOURCE_METHOD_KEYS = {"KHRONOS_OPEN": "KHRONOS"}
METRIC_SOURCE_FILES = {
    "object_f1": "static_objects.csv",
    "dynamic_f1": "dynamic_objects.csv",
    "change_f1": "static_objects.csv",
}
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _is_lower_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _run_identity(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    raw = payload.get("run_identity")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} run identity is required")
    run_id = raw.get("run_id")
    if type(run_id) is not str or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(f"{label} run identity run_id is not canonical")
    if not _is_lower_sha256(raw.get("config_sha256")):
        raise ValueError(f"{label} run identity config_sha256 is invalid")
    return {
        "run_id": run_id,
        "config_sha256": raw["config_sha256"],
    }


def _paired_run_identity(
    metrics: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    scene: str,
) -> dict[str, Any]:
    metrics_identity = _run_identity(metrics, label=f"{scene} metrics")
    status_identity = _run_identity(status, label=f"{scene} status")
    if metrics_identity != status_identity:
        raise ValueError(f"{scene} metrics/status run identity mismatch")
    return metrics_identity


def _scene_metrics(
    payload: Mapping[str, Any],
    *,
    scene: str,
    method_key: str,
    mode: str,
) -> tuple[dict[str, float | None], dict[str, str]]:
    if payload.get("status") not in {None, "PASS", "PARTIAL"}:
        raise ValueError(f"{scene} official metric source is not usable")
    if payload.get("dataset") != "TESSE-CD":
        raise ValueError(f"{scene} source is not TESSE-CD")
    if payload.get("scene") != scene or payload.get("split") != f"{scene}_test":
        raise ValueError(f"expected {scene} official metric source")
    source_method = SOURCE_METHOD_KEYS.get(method_key, method_key)
    if payload.get("method") != source_method or payload.get("mode") != mode:
        raise ValueError(f"{scene} method or mode mismatch")
    source = payload.get("metrics")
    if not isinstance(source, Mapping):
        raise ValueError(f"{scene} source has no metrics")
    unavailable_source = payload.get("unavailable", {})
    if not isinstance(unavailable_source, Mapping):
        raise ValueError(f"{scene} unavailable reasons must be a mapping")

    metrics: dict[str, float | None] = {}
    unavailable: dict[str, str] = {}
    for name in SCENE_METRICS:
        value = source.get(name)
        if value is None:
            reason = str(unavailable_source.get(name, "")).strip()
            if not reason:
                raise ValueError(f"{scene}.{name} requires an unavailable reason")
            metrics[name] = None
            unavailable[name] = reason
            continue
        if type(value) not in {int, float}:
            raise ValueError(f"{scene}.{name} must be an int or float")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{scene}.{name} must be finite and within [0, 1]")
        metrics[name] = number
    return metrics, unavailable


def build_partial_official_metrics(
    apartment: Mapping[str, Any],
    office: Mapping[str, Any],
    *,
    method_key: str,
    mode: str,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, str]]]:
    if METHOD_MODES.get(method_key) != mode:
        raise ValueError("unsupported TESSE-CD method or mode")
    metrics: dict[str, dict[str, float | None]] = {}
    unavailable: dict[str, dict[str, str]] = {}
    for scene, payload in (("apartment", apartment), ("office", office)):
        scene_metrics, scene_unavailable = _scene_metrics(
            payload,
            scene=scene,
            method_key=method_key,
            mode=mode,
        )
        metrics[scene] = scene_metrics
        unavailable[scene] = scene_unavailable
    return metrics, unavailable


def partial_official_token_bindings(
    method_key: str,
    metrics: Mapping[str, Mapping[str, float | None]],
) -> list[dict[str, object]]:
    if method_key not in METHOD_MODES:
        raise ValueError("unsupported TESSE-CD method")
    return [
        {
            "token": f"T2_{method_key}_{scene.upper()}_{metric.upper()}",
            "json_pointer": f"/metrics/{scene}/{metric}",
            "precision": 3,
        }
        for scene in ("apartment", "office")
        for metric in SCENE_METRICS
        if metrics[scene][metric] is not None
    ]


def partial_official_unavailable_bindings(
    method_key: str,
    metrics: Mapping[str, Mapping[str, float | None]],
) -> list[dict[str, str]]:
    if method_key not in METHOD_MODES:
        raise ValueError("unsupported TESSE-CD method")
    return [
        {
            "token": f"T2_{method_key}_{scene.upper()}_{metric.upper()}",
            "reason_pointer": f"/unavailable/{scene}/{metric}",
            "evidence_pointer": f"/unavailable_evidence/{scene}/{metric}",
        }
        for scene in ("apartment", "office")
        for metric in SCENE_METRICS
        if metrics[scene][metric] is None
    ]


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_no_symlinks(
    path: Path, *, label: str, terminal_directory: bool
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    absolute = _absolute_lexical(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0) | no_follow
    descriptors: list[int] = []
    identities: list[tuple[int, int, int]] = []
    try:
        current = os.open(absolute.anchor, directory_flags)
        descriptors.append(current)
        root_status = os.fstat(current)
        identities.append(
            (root_status.st_dev, root_status.st_ino, stat.S_IFMT(root_status.st_mode))
        )
        for index, component in enumerate(absolute.parts[1:]):
            observed = os.stat(component, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(
                    f"{label} regular file path contains a symbolic link component"
                )
            terminal = index == len(absolute.parts[1:]) - 1
            expected = (
                stat.S_ISDIR
                if not terminal or terminal_directory
                else stat.S_ISREG
            )
            if not expected(observed.st_mode):
                kind = "directory" if not terminal or terminal_directory else "regular file"
                raise ValueError(f"{label} path component is not a {kind}")
            opened = os.open(
                component,
                flags
                | no_follow
                | (
                    getattr(os, "O_DIRECTORY", 0)
                    if not terminal or terminal_directory
                    else 0
                ),
                dir_fd=current,
            )
            confirmed = os.fstat(opened)
            identity = (
                confirmed.st_dev,
                confirmed.st_ino,
                stat.S_IFMT(confirmed.st_mode),
            )
            if identity != (
                observed.st_dev,
                observed.st_ino,
                stat.S_IFMT(observed.st_mode),
            ):
                os.close(opened)
                raise ValueError(f"{label} changed while resolving")
            descriptors.append(opened)
            identities.append(identity)
            current = opened
        terminal_fd = descriptors.pop()
        return terminal_fd, tuple(identities)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"{label} is missing or cannot be opened safely") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _open_regular_no_symlinks(
    path: Path, *, label: str
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    return _open_no_symlinks(path, label=label, terminal_directory=False)


def _open_directory_no_symlinks(
    path: Path, *, label: str
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    return _open_no_symlinks(path, label=label, terminal_directory=True)


def _stable_regular_file(
    path: Path, *, label: str, capture: bool
) -> tuple[str, int, bytes | None]:
    descriptor, identities = _open_regular_no_symlinks(path, label=label)
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        snapshot = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            byte_count != before.st_size
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != snapshot
        ):
            raise ValueError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)

    reopened, reopened_identities = _open_regular_no_symlinks(path, label=label)
    try:
        current = os.fstat(reopened)
        if reopened_identities != identities or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != snapshot:
            raise ValueError(f"{label} changed while it was read")
    finally:
        os.close(reopened)
    return digest.hexdigest(), byte_count, None if chunks is None else b"".join(chunks)


def _canonical_relative_source(raw: str, *, label: str) -> Path:
    path = Path(raw)
    if (
        not raw
        or "\\" in raw
        or path.is_absolute()
        or "." in path.parts
        or path.as_posix() != raw
    ):
        raise ValueError(f"{label} must be a canonical artifact-relative path")
    return path


def _resolved_relative_source(
    relative: Path,
    *,
    artifact_base: Path | None,
    artifact_root: Path | None,
    label: str,
) -> Path:
    if artifact_base is None:
        raise ValueError(f"{label} relative source requires an artifact base")
    base = _absolute_lexical(artifact_base)
    if artifact_root is None:
        if ".." in relative.parts:
            raise ValueError(
                f"{label} parent traversal requires an explicit artifact root"
            )
        root = base
    else:
        root = _absolute_lexical(artifact_root)
    try:
        base.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{label} artifact base must be beneath the artifact root"
        ) from error
    target = _absolute_lexical(base / relative)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} source escapes the artifact root") from error
    if target == root:
        raise ValueError(
            f"{label} source must identify a file beneath the artifact root"
        )
    return target


def _validated_declared_source(
    entry: Mapping[str, Any],
    *,
    label: str,
    artifact_base: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    raw_path = entry.get("path")
    if type(raw_path) is not str or not raw_path:
        raise ValueError(f"{label} source path must be a non-empty string")
    declared = Path(raw_path)
    if declared.is_absolute():
        path = _absolute_lexical(declared)
    else:
        relative = _canonical_relative_source(raw_path, label=f"{label} source path")
        path = _resolved_relative_source(
            relative,
            artifact_base=artifact_base,
            artifact_root=artifact_root,
            label=label,
        )
    byte_count = entry.get("byte_count")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{label} source byte count mismatch")
    declared_sha256 = entry.get("sha256")
    if not _is_lower_sha256(declared_sha256):
        raise ValueError(f"{label} source SHA256 is invalid")
    observed, observed_bytes, _ = _stable_regular_file(
        path, label=f"{label} source", capture=False
    )
    if byte_count != observed_bytes:
        raise ValueError(f"{label} source byte count mismatch")
    if declared_sha256 != observed:
        raise ValueError(f"{label} source SHA256 mismatch")
    return {
        "path": str(path),
        "sha256": observed,
        "byte_count": byte_count,
    }


def _validated_missing_source(
    entry: Mapping[str, Any],
    *,
    label: str,
    artifact_base: Path | None,
    artifact_root: Path | None = None,
) -> dict[str, str]:
    if set(entry) != {"path", "status"} or entry.get("status") != "MISSING":
        raise ValueError(
            f"{label} MISSING record must contain exactly path and status=MISSING"
        )
    raw_path = entry.get("path")
    if type(raw_path) is not str:
        raise ValueError(f"{label} MISSING record path must be a string")
    relative = _canonical_relative_source(raw_path, label=f"{label} source path")
    target = _resolved_relative_source(
        relative,
        artifact_base=artifact_base,
        artifact_root=artifact_root,
        label=label,
    )
    parent = target.parent
    descriptor, identities = _open_directory_no_symlinks(
        parent, label=f"{label} missing source parent"
    )
    try:
        try:
            observed = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            observed = None
        if observed is not None:
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(
                    f"{label} missing source path contains a symbolic link component"
                )
            raise ValueError(f"{label} source is marked MISSING but exists")
    finally:
        os.close(descriptor)

    reopened, reopened_identities = _open_directory_no_symlinks(
        parent, label=f"{label} missing source parent"
    )
    try:
        if reopened_identities != identities:
            raise ValueError(f"{label} missing source parent changed while validating")
        try:
            observed = os.stat(target.name, dir_fd=reopened, follow_symlinks=False)
        except FileNotFoundError:
            observed = None
        if observed is not None:
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(
                    f"{label} missing source path contains a symbolic link component"
                )
            raise ValueError(f"{label} source appeared while validating MISSING record")
    finally:
        os.close(reopened)
    return {"path": relative.as_posix(), "status": "MISSING"}


def _source_bound_unavailable(
    payload: Mapping[str, Any],
    unavailable: Mapping[str, str],
    *,
    scene: str,
    artifact_base: Path | None = None,
    declaration_source: Mapping[str, Any] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if not unavailable:
        return {}
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{scene} unavailable metrics require source records")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{scene} metric source record must be a mapping")
        raw_path = raw.get("path")
        if type(raw_path) is not str:
            raise ValueError(f"{scene} metric source path must be a string")
        if "status" in raw or set(raw) == {"path"}:
            _validated_missing_source(
                raw,
                label=f"{scene} metric source",
                artifact_base=artifact_base,
                artifact_root=artifact_root,
            )
        name = Path(raw_path).name
        if not name or name in by_name:
            raise ValueError(f"{scene} metric source names must be unique")
        by_name[name] = raw

    evidence: dict[str, dict[str, Any]] = {}
    for metric, reason in unavailable.items():
        filename = METRIC_SOURCE_FILES[metric]
        if filename not in by_name:
            raise ValueError(f"{scene}.{metric} source is missing: {filename}")
        source = by_name[filename]
        if source.get("status") == "MISSING":
            if declaration_source is None:
                raise ValueError(
                    f"{scene}.{metric} MISSING record requires its metrics JSON source"
                )
            missing = _validated_missing_source(
                source,
                label=f"{scene}.{metric}",
                artifact_base=artifact_base,
                artifact_root=artifact_root,
            )
            metrics_source = _validated_declared_source(
                declaration_source,
                label=f"{scene}.{metric} metrics declaration",
            )
            evidence[metric] = {
                "reason": reason,
                "source": metrics_source,
                "missing_source": missing,
            }
        else:
            record = {
                "reason": reason,
                "source": _validated_declared_source(
                    source,
                    label=f"{scene}.{metric}",
                    artifact_base=artifact_base,
                    artifact_root=artifact_root,
                ),
            }
            evidence[metric] = record
    return evidence


def _hashed_entry(
    entry: Mapping[str, Any],
    *,
    label: str,
    artifact_base: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(entry, Mapping) or type(entry.get("path")) is not str:
        raise ValueError(f"{label} provenance record requires a string path")
    declared = Path(entry["path"])
    if declared.is_absolute():
        path = _absolute_lexical(declared)
    else:
        relative = _canonical_relative_source(
            entry["path"], label=f"{label} path"
        )
        if ".." in relative.parts:
            raise ValueError(f"{label} path must not contain parent traversal")
        path = _absolute_lexical(
            relative if artifact_base is None else artifact_base / relative
        )
    observed, byte_count, _ = _stable_regular_file(
        path, label=label, capture=False
    )
    result = dict(entry)
    result.update(
        path=str(path),
        sha256=observed,
        byte_count=byte_count,
    )
    return result


def _hashed_entries(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
    artifact_base: Path | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"provenance {label} must be a list")
    return [
        _hashed_entry(
            entry,
            label=f"provenance {label}[{index}]",
            artifact_base=artifact_base,
        )
        for index, entry in enumerate(value)
    ]


def _validate_run_status(
    status: Mapping[str, Any],
    *,
    scene: str,
    method_key: str,
    mode: str,
) -> None:
    if status.get("status") != "PASS":
        raise ValueError(f"{scene} run status must be PASS")
    if status.get("scene") != scene or status.get("mode") != mode:
        raise ValueError(f"{scene} run identity mismatch")
    source_method = SOURCE_METHOD_KEYS.get(method_key, method_key)
    if (
        type(status.get("method")) is not str
        or status["method"] != source_method
    ):
        raise ValueError(f"{scene} run method mismatch")
    if mode == "frozen" and int(status.get("updates_after_freeze", -1)) != 0:
        raise ValueError(f"{scene} updates_after_freeze must be zero")


def build_scene_evidence(
    metrics_path: Path,
    status_path: Path,
    *,
    method_key: str,
    mode: str,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    if METHOD_MODES.get(method_key) != mode:
        raise ValueError("unsupported TESSE-CD method or mode")
    metrics_payload, metrics_source, _ = _json_source(
        metrics_path, label="scene metrics"
    )
    status_payload, status_source, _ = _json_source(
        status_path, label="scene status"
    )
    scene = str(metrics_payload.get("scene", ""))
    if scene not in {"apartment", "office"}:
        raise ValueError("scene evidence requires apartment or office metrics")
    metrics, unavailable = _scene_metrics(
        metrics_payload,
        scene=scene,
        method_key=method_key,
        mode=mode,
    )
    _validate_run_status(
        status_payload,
        scene=scene,
        method_key=method_key,
        mode=mode,
    )
    run_identity = _paired_run_identity(
        metrics_payload, status_payload, scene=scene
    )
    unavailable_evidence = _source_bound_unavailable(
        metrics_payload,
        unavailable,
        scene=scene,
        artifact_base=Path(metrics_source["path"]).parent,
        declaration_source=metrics_source,
        artifact_root=artifact_root,
    )
    result = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_t2_scene_evidence",
        "status": "PASS",
        "dataset": "TESSE-CD",
        "scene": scene,
        "method_key": method_key,
        "mode": mode,
        "run_identity": run_identity,
        "metrics": metrics,
        "unavailable": unavailable,
        "unavailable_evidence": unavailable_evidence,
        "official_metrics_source": metrics_source,
        "run_status_source": status_source,
        "official_metrics": dict(metrics_payload),
        "run_status": dict(status_payload),
    }
    return result


def _build_result_payload(
    apartment: Mapping[str, Any],
    office: Mapping[str, Any],
    apartment_status: Mapping[str, Any],
    office_status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    method_key: str,
    mode: str,
    source_bases: Mapping[str, Path] | None = None,
    metrics_sources: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_root: Path | None = None,
    provenance_base: Path | None = None,
) -> dict[str, Any]:
    metrics, unavailable = build_partial_official_metrics(
        apartment,
        office,
        method_key=method_key,
        mode=mode,
    )
    unavailable_evidence = {
        "apartment": _source_bound_unavailable(
            apartment,
            unavailable["apartment"],
            scene="apartment",
            artifact_base=None if source_bases is None else source_bases.get("apartment"),
            declaration_source=(
                None if metrics_sources is None else metrics_sources.get("apartment")
            ),
            artifact_root=artifact_root,
        ),
        "office": _source_bound_unavailable(
            office,
            unavailable["office"],
            scene="office",
            artifact_base=None if source_bases is None else source_bases.get("office"),
            declaration_source=(
                None if metrics_sources is None else metrics_sources.get("office")
            ),
            artifact_root=artifact_root,
        ),
    }
    _validate_run_status(
        apartment_status,
        scene="apartment",
        method_key=method_key,
        mode=mode,
    )
    _validate_run_status(
        office_status,
        scene="office",
        method_key=method_key,
        mode=mode,
    )
    run_identity = {
        "apartment": _paired_run_identity(
            apartment, apartment_status, scene="apartment"
        ),
        "office": _paired_run_identity(office, office_status, scene="office"),
    }
    dirty_digest = str(provenance.get("dirty_state_digest", ""))
    if len(dirty_digest) != 64 or any(
        char not in "0123456789abcdef" for char in dirty_digest
    ):
        raise ValueError("dirty_state_digest must be a lowercase SHA256")
    commands = provenance.get("commands")
    if not isinstance(commands, list) or not commands or not all(commands):
        raise ValueError("provenance commands must be a non-empty list")
    provenance_run_id = provenance.get("run_id")
    if (
        type(provenance_run_id) is not str
        or RUN_ID_PATTERN.fullmatch(provenance_run_id) is None
    ):
        raise ValueError("provenance run_id must be canonical")
    configs = _hashed_entries(
        provenance["configs"],
        "configs",
        artifact_base=provenance_base,
    )
    config_sha256 = {record["sha256"] for record in configs}
    for scene, identity in run_identity.items():
        if identity["run_id"] != provenance_run_id:
            raise ValueError(f"{scene} run identity disagrees with provenance run_id")
        if identity["config_sha256"] not in config_sha256:
            raise ValueError(f"{scene} run identity has no provenance config binding")

    result = {
        "run_id": provenance_run_id,
        "run_identity": run_identity,
        "method": {
            "key": method_key,
            "display_label": METHOD_DISPLAY_LABELS.get(
                method_key, method_key.replace("_", " ").title()
            ),
            "mode": TABLE_MODES[method_key],
            "eligible_for_ranking": True,
        },
        "upstream_commit": str(provenance["upstream_commit"]),
        "adapter_commit": str(provenance["adapter_commit"]),
        "dirty_state_digest": dirty_digest,
        "dataset": {
            "name": "TESSE-CD",
            "splits": ["apartment_test", "office_test"],
            "manifest": _hashed_entry(
                provenance["dataset_manifest"],
                label="dataset manifest",
                artifact_base=provenance_base,
            ),
        },
        "metrics": metrics,
        "unavailable": unavailable,
        "unavailable_evidence": unavailable_evidence,
        "protocol": {
            "name": "Khronos upstream TESSE-CD evaluator via neutral-map bridge",
            "aggregation": "macro over unique online state/query rows per sequence",
            "binding_policy": "only finite official metrics are importable",
        },
        "run_status": {
            "apartment": dict(apartment_status),
            "office": dict(office_status),
        },
        "commands": [str(command) for command in commands],
        "environment": dict(provenance["environment"]),
        "hardware": dict(provenance["hardware"]),
        "seed": int(provenance.get("seed", 0)),
        "configs": configs,
        "weights": _hashed_entries(
            provenance.get("weights", []),
            "weights",
            allow_empty=True,
            artifact_base=provenance_base,
        ),
        "raw_outputs": _hashed_entries(
            provenance["raw_outputs"],
            "raw_outputs",
            artifact_base=provenance_base,
        ),
        "protocol_deviations": [
            str(value) for value in provenance.get("protocol_deviations", ())
        ],
        "token_bindings": partial_official_token_bindings(method_key, metrics),
        "unavailable_bindings": partial_official_unavailable_bindings(
            method_key, metrics
        ),
    }
    return result


def _strict_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _json_source(
    path: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    resolved = _absolute_lexical(path)
    observed, byte_count, captured = _stable_regular_file(
        resolved, label=label, capture=True
    )
    if captured is None:
        raise AssertionError("captured JSON bytes are required")
    payload = _strict_json_object(captured, label=label)
    return (
        payload,
        {
            "path": str(resolved),
            "sha256": observed,
            "byte_count": byte_count,
        },
        captured,
    )


def _json_repeat_pair(
    primary: Path,
    repeat: Path,
    *,
    label: str,
    require_byte_identical: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    payload, primary_record, primary_bytes = _json_source(primary, label=label)
    repeat_payload, repeat_record, repeat_bytes = _json_source(
        repeat, label=f"{label} repeat"
    )
    if os.path.samefile(primary, repeat):
        raise ValueError(f"{label} repeat must use independent files")
    if require_byte_identical and primary_bytes != repeat_bytes:
        raise ValueError(f"{label} repeat must be byte-identical")
    return (
        payload,
        repeat_payload,
        {"primary": primary_record, "repeat": repeat_record},
    )


def build_result(
    apartment_metrics: Path,
    apartment_metrics_repeat: Path,
    office_metrics: Path,
    office_metrics_repeat: Path,
    apartment_status: Path,
    apartment_status_repeat: Path,
    office_status: Path,
    office_status_repeat: Path,
    provenance: Path,
    *,
    method_key: str,
    mode: str,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    apartment_payload, _, apartment_metrics_sources = _json_repeat_pair(
        apartment_metrics,
        apartment_metrics_repeat,
        label="apartment metrics",
        require_byte_identical=True,
    )
    office_payload, _, office_metrics_sources = _json_repeat_pair(
        office_metrics,
        office_metrics_repeat,
        label="office metrics",
        require_byte_identical=True,
    )
    (
        apartment_status_payload,
        apartment_status_repeat_payload,
        apartment_status_sources,
    ) = _json_repeat_pair(
        apartment_status,
        apartment_status_repeat,
        label="apartment status",
        require_byte_identical=False,
    )
    (
        office_status_payload,
        office_status_repeat_payload,
        office_status_sources,
    ) = _json_repeat_pair(
        office_status,
        office_status_repeat,
        label="office status",
        require_byte_identical=False,
    )
    provenance_payload, provenance_source, _ = _json_source(
        provenance, label="provenance"
    )
    result = _build_result_payload(
        apartment_payload,
        office_payload,
        apartment_status_payload,
        office_status_payload,
        provenance_payload,
        method_key=method_key,
        mode=mode,
        source_bases={
            "apartment": Path(apartment_metrics_sources["primary"]["path"]).parent,
            "office": Path(office_metrics_sources["primary"]["path"]).parent,
        },
        metrics_sources={
            "apartment": apartment_metrics_sources["primary"],
            "office": office_metrics_sources["primary"],
        },
        artifact_root=artifact_root,
        provenance_base=Path(provenance_source["path"]).parent,
    )
    result["status"] = "VERIFIED"
    result["protocol"]["deterministic_repeat"] = "metrics-byte-identical"
    for scene, metrics_payload, status_payload in (
        ("apartment", apartment_payload, apartment_status_repeat_payload),
        ("office", office_payload, office_status_repeat_payload),
    ):
        _validate_run_status(
            status_payload,
            scene=scene,
            method_key=method_key,
            mode=mode,
        )
        _paired_run_identity(metrics_payload, status_payload, scene=scene)
    evidence_sources = {
        "apartment": {
            "metrics": apartment_metrics_sources,
            "status": apartment_status_sources,
        },
        "office": {
            "metrics": office_metrics_sources,
            "status": office_status_sources,
        },
        "provenance": provenance_source,
    }
    result["evidence_sources"] = evidence_sources
    return result


def _atomic_json_no_replace(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    directory_descriptor: int | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        os.fsync(directory_descriptor)
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        temporary.unlink(missing_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apartment-metrics", type=Path)
    parser.add_argument("--apartment-metrics-repeat", type=Path)
    parser.add_argument("--office-metrics", type=Path)
    parser.add_argument("--office-metrics-repeat", type=Path)
    parser.add_argument("--apartment-status", type=Path)
    parser.add_argument("--apartment-status-repeat", type=Path)
    parser.add_argument("--office-status", type=Path)
    parser.add_argument("--office-status-repeat", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--scene-metrics", type=Path)
    parser.add_argument("--scene-status", type=Path)
    parser.add_argument("--method", choices=tuple(METHOD_MODES), required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.scene_metrics is not None or args.scene_status is not None:
        if args.scene_metrics is None or args.scene_status is None:
            parser.error("scene evidence requires --scene-metrics and --scene-status")
        result = build_scene_evidence(
            args.scene_metrics,
            args.scene_status,
            method_key=args.method,
            mode=METHOD_MODES[args.method],
            artifact_root=args.artifact_root,
        )
        _atomic_json_no_replace(args.output, result)
        return 0

    required = {
        "--apartment-metrics": args.apartment_metrics,
        "--apartment-metrics-repeat": args.apartment_metrics_repeat,
        "--office-metrics": args.office_metrics,
        "--office-metrics-repeat": args.office_metrics_repeat,
        "--apartment-status": args.apartment_status,
        "--apartment-status-repeat": args.apartment_status_repeat,
        "--office-status": args.office_status,
        "--office-status-repeat": args.office_status_repeat,
        "--provenance": args.provenance,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("full result requires " + ", ".join(missing))

    result = build_result(
        args.apartment_metrics,
        args.apartment_metrics_repeat,
        args.office_metrics,
        args.office_metrics_repeat,
        args.apartment_status,
        args.apartment_status_repeat,
        args.office_status,
        args.office_status_repeat,
        args.provenance,
        method_key=args.method,
        mode=METHOD_MODES[args.method],
        artifact_root=args.artifact_root,
    )
    _atomic_json_no_replace(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
