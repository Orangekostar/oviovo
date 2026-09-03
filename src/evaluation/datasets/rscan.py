"""Strict metadata adapter for the causal 3RScan pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.json_contracts import loads_strict

CHANGE_TYPES = ("rigid", "nonrigid", "removed")
REQUIRED_RUNTIME_MEMBERS = (
    "sequence.zip",
    "mesh.refined.v2.obj",
    "labels.instances.annotated.v2.ply",
    "semseg.v2.json",
)


class RScanDatasetError(ValueError):
    """Raised when 3RScan metadata or selection evidence is ambiguous."""


@dataclass(frozen=True, slots=True)
class RScanChange:
    change_type: str
    reference_instance_id: int
    rescan_instance_id: int | None
    transform: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class RScanSession:
    scan_id: str
    session_index: int
    changes: tuple[RScanChange, ...]
    evaluator_global_transform: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class RScanEnvironment:
    reference_id: str
    split: str
    sessions: tuple[RScanSession, ...]


def _scan_id(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RScanDatasetError(f"{label} must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise RScanDatasetError(f"{label} must be a UUID") from exc
    if str(parsed) != value:
        raise RScanDatasetError(f"{label} must be a canonical lowercase UUID")
    return value


def _instance_id(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RScanDatasetError(f"{label} must be a non-negative integer")
    return value


def _transform(value: object, *, required: bool) -> tuple[float, ...] | None:
    if value is None and not required:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 16:
        raise RScanDatasetError("3RScan transform must contain 16 finite values")
    result = tuple(float(number) for number in value)
    if not all(math.isfinite(number) for number in result):
        raise RScanDatasetError("3RScan transform must contain 16 finite values")
    return result


def _change(value: object, *, change_type: str) -> RScanChange:
    if isinstance(value, int) and not isinstance(value, bool):
        reference = _instance_id(value, label=f"{change_type} instance")
        return RScanChange(
            change_type,
            reference,
            None if change_type == "removed" else reference,
            None,
        )
    if not isinstance(value, Mapping):
        raise RScanDatasetError(f"invalid {change_type} change record")
    reference = _instance_id(
        value.get("instance_reference"), label=f"{change_type} reference instance"
    )
    rescan_value = value.get("instance_rescan")
    rescan = (
        None
        if change_type == "removed" and rescan_value is None
        else _instance_id(rescan_value, label=f"{change_type} rescan instance")
    )
    transform = _transform(value.get("transform"), required=False)
    return RScanChange(change_type, reference, rescan, transform)


def load_rscan_metadata(path: str | Path) -> tuple[RScanEnvironment, ...]:
    """Parse official metadata while keeping evaluator-only transforms explicit."""

    source = Path(path).resolve()
    try:
        payload = loads_strict(source.read_text(encoding="utf-8"), label="3RScan metadata")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RScanDatasetError("3RScan metadata is unreadable") from exc
    if not isinstance(payload, list) or not payload:
        raise RScanDatasetError("3RScan metadata must be a non-empty list")
    environments: list[RScanEnvironment] = []
    seen_scans: set[str] = set()
    for raw_environment in payload:
        if not isinstance(raw_environment, Mapping):
            raise RScanDatasetError("3RScan environment must be a mapping")
        reference = _scan_id(raw_environment.get("reference"), label="reference")
        split = raw_environment.get("type")
        scans = raw_environment.get("scans")
        if not isinstance(split, str) or not split:
            raise RScanDatasetError("3RScan split is missing")
        if not isinstance(scans, list) or not scans:
            raise RScanDatasetError("3RScan environment has no rescan")
        sessions = [RScanSession(reference, 0, (), None)]
        for session_index, raw_scan in enumerate(scans, start=1):
            if not isinstance(raw_scan, Mapping):
                raise RScanDatasetError("3RScan rescan must be a mapping")
            scan_id = _scan_id(raw_scan.get("reference"), label="rescan")
            changes: list[RScanChange] = []
            for change_type in CHANGE_TYPES:
                values = raw_scan.get(change_type, [])
                if not isinstance(values, list):
                    raise RScanDatasetError(f"{change_type} changes must be a list")
                changes.extend(_change(value, change_type=change_type) for value in values)
            seen_changes: set[RScanChange] = set()
            normalized_changes: list[RScanChange] = []
            for change in changes:
                if change in seen_changes:
                    continue
                seen_changes.add(change)
                normalized_changes.append(change)
            sessions.append(
                RScanSession(
                    scan_id,
                    session_index,
                    tuple(normalized_changes),
                    _transform(raw_scan.get("transform"), required=False),
                )
            )
        scan_ids = [session.scan_id for session in sessions]
        if len(scan_ids) != len(set(scan_ids)) or seen_scans.intersection(scan_ids):
            raise RScanDatasetError("duplicate 3RScan scan UUID")
        seen_scans.update(scan_ids)
        environments.append(RScanEnvironment(reference, split, tuple(sessions)))
    return tuple(environments)


def _validation_ids(path: str | Path) -> frozenset[str]:
    source = Path(path).resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RScanDatasetError("3RScan validation list is unreadable") from exc
    values = tuple(_scan_id(line, label="validation scan") for line in lines if line)
    if not values or len(values) != len(set(values)):
        raise RScanDatasetError("3RScan validation list is empty or duplicated")
    return frozenset(values)


def select_pilot(
    metadata: str | Path,
    validation_list: str | Path,
    *,
    count: int = 10,
) -> tuple[RScanEnvironment, ...]:
    """Apply the frozen result-independent pilot-selection rule."""

    if type(count) is not int or count <= 0:
        raise RScanDatasetError("pilot count must be a positive integer")
    eligible = _eligible_environments(
        load_rscan_metadata(metadata), _validation_ids(validation_list)
    )
    if len(eligible) < count:
        raise RScanDatasetError(
            f"only {len(eligible)} environments satisfy the frozen pilot rule"
        )
    return tuple(eligible[:count])


def _eligible_environments(
    environments: Sequence[RScanEnvironment], allowed: frozenset[str]
) -> list[RScanEnvironment]:
    eligible = [
        environment
        for environment in environments
        if environment.split == "validation"
        and all(session.scan_id in allowed for session in environment.sessions)
        and len(environment.sessions) >= 2
        and any(session.changes for session in environment.sessions[1:])
    ]
    eligible.sort(key=lambda environment: environment.reference_id)
    return eligible


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"sha256": _sha256(path), "byte_count": path.stat().st_size}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_selection_manifest(
    metadata: str | Path,
    validation_list: str | Path,
    *,
    asset_root: str | Path,
    count: int,
    output: str | Path,
) -> dict[str, Any]:
    """Freeze selected visits and inventory runtime assets without reading results."""

    metadata_path = Path(metadata).resolve()
    validation_path = Path(validation_list).resolve()
    environments = load_rscan_metadata(metadata_path)
    eligible = _eligible_environments(environments, _validation_ids(validation_path))
    if len(eligible) < count:
        raise RScanDatasetError(
            f"only {len(eligible)} environments satisfy the frozen pilot rule"
        )
    selected = tuple(eligible[:count])
    root = Path(asset_root).resolve()
    missing: list[dict[str, object]] = []
    for environment in selected:
        for session in environment.sessions:
            directory = root / session.scan_id
            members = [
                member for member in REQUIRED_RUNTIME_MEMBERS if not (directory / member).is_file()
            ]
            if members:
                missing.append({"scan_id": session.scan_id, "members": members})
    visit_histogram = Counter(len(environment.sessions) for environment in environments)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "3rscan_causal_pilot_v1",
        "status": "BLOCKED_DATASET_ACCESS" if missing else "READY",
        "metadata_status": "PASS",
        "selection_rule": {
            "split": "validation",
            "all_visits_in_validation_list": True,
            "minimum_visit_count": 2,
            "required_change_types": list(CHANGE_TYPES),
            "sort_key": "reference_scan_uuid",
            "take_first": count,
            "result_independent": True,
        },
        "source_bindings": {
            "metadata": _binding(metadata_path),
            "validation_list": _binding(validation_path),
        },
        "environment_count": len(environments),
        "visit_statistics": {
            "exact": {str(key): visit_histogram[key] for key in sorted(visit_histogram)},
            "at_least": {
                str(threshold): sum(
                    value for visits, value in visit_histogram.items() if visits >= threshold
                )
                for threshold in range(2, 7)
            },
        },
        "selected_environment_count": len(selected),
        "eligible_environment_count": len(eligible),
        "selected_environments": [
            {
                "reference_id": environment.reference_id,
                "session_ids": [session.scan_id for session in environment.sessions],
                "change_counts": dict(
                    Counter(
                        change.change_type
                        for session in environment.sessions
                        for change in session.changes
                    )
                ),
            }
            for environment in selected
        ],
        "runtime_assets": {
            "status": "BLOCKED_DATASET_ACCESS" if missing else "PASS",
            "required_members": list(REQUIRED_RUNTIME_MEMBERS),
            "missing": missing,
        },
        "protocol": {
            "causal_prefix_only": True,
            "gt_transforms_and_cross_time_ids_are_evaluator_only": True,
            "community_metrics_separate_from_exact_id_diagnostics": True,
        },
    }
    _atomic_json(Path(output), manifest)
    return manifest
