"""Build the development-only, two-session 3RScan evidence selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.datasets.rscan import load_rscan_metadata
from src.evaluation.json_contracts import loads_strict

RAW_RUNTIME_MEMBERS = (
    "sequence.zip",
    "mesh.refined.v2.obj",
    "labels.instances.annotated.v2.ply",
    "semseg.v2.json",
    "mesh.refined.0.010000.segs.v2.json",
)

_PROCESSED_RAW_PATHS = {
    "raw_filepath": "mesh.refined.v2.obj",
    "raw_instance_filepath": "semseg.v2.json",
    "raw_label_filepath": "labels.instances.annotated.v2.ply",
    "raw_segmentation_filepath": "mesh.refined.0.010000.segs.v2.json",
}


class RScanT2DevelopmentError(ValueError):
    """Raised when the frozen development selection cannot be justified."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise RScanT2DevelopmentError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _reject_nonfinite(value: object, *, label: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise RScanT2DevelopmentError(f"{label} contains a non-finite value")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label=label)


def _load_yaml(path: Path, *, label: str) -> object:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RScanT2DevelopmentError(f"{label} is unreadable") from error
    _reject_nonfinite(value, label=label)
    return value


def _canonical_uuid(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RScanT2DevelopmentError(f"{label} must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise RScanT2DevelopmentError(f"{label} must be a UUID") from error
    if str(parsed) != value:
        raise RScanT2DevelopmentError(f"{label} must be a canonical lowercase UUID")
    return value


def _regular_file(path: str | Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        record = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise RScanT2DevelopmentError(f"{label} is unavailable") from error
    if not stat.S_ISREG(record.st_mode):
        raise RScanT2DevelopmentError(f"{label} must be a regular non-symlink file")
    return absolute


def _file_record(path: str | Path, *, label: str) -> dict[str, object]:
    absolute = _regular_file(path, label=label)
    before = absolute.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with absolute.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise RScanT2DevelopmentError(f"{label} cannot be read") from error
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or byte_count != before.st_size:
        raise RScanT2DevelopmentError(f"{label} changed while being read")
    return {
        "path": str(absolute),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _validate_file_record(record: object, *, label: str) -> dict[str, object]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise RScanT2DevelopmentError(f"{label} binding schema is invalid")
    path = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise RScanT2DevelopmentError(f"{label} binding value is invalid")
    observed = _file_record(path, label=label)
    if dict(record) != observed:
        raise RScanT2DevelopmentError(f"{label} binding mismatch")
    return observed


def _validation_ids(path: Path) -> frozenset[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RScanT2DevelopmentError("validation list is unreadable") from error
    values = tuple(
        _canonical_uuid(line, label="validation scan") for line in lines if line
    )
    if not values or len(values) != len(set(values)):
        raise RScanT2DevelopmentError("validation list is empty or duplicated")
    return frozenset(values)


def _safe_relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RScanT2DevelopmentError(f"{label} must be a relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RScanT2DevelopmentError(f"{label} must be a safe relative path")
    return path


def _scan_id_from_database_record(record: Mapping[str, object], *, label: str) -> str:
    raw = record.get("raw_filepath")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise RScanT2DevelopmentError(f"{label} raw_filepath must be absolute")
    return _canonical_uuid(Path(raw).parent.name, label=f"{label} scan")


def _database_by_scan(value: object, *, label: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise RScanT2DevelopmentError(f"{label} must contain a non-empty list")
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise RScanT2DevelopmentError(f"{label}[{index}] must be a mapping")
        scan_id = _scan_id_from_database_record(raw, label=f"{label}[{index}]")
        if scan_id in result:
            raise RScanT2DevelopmentError(f"{label} duplicates scan {scan_id}")
        result[scan_id] = raw
    return result


def _existing_regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        return False


def _processed_paths(
    record: Mapping[str, object],
    *,
    scan_id: str,
    raw_root: Path,
    processed_root: Path,
) -> tuple[Path, Path] | None:
    scene = record.get("scene")
    sub_scene = record.get("sub_scene")
    if (
        type(scene) is not int
        or scene < 0
        or type(sub_scene) is not int
        or sub_scene < 0
    ):
        return None
    for field, member in _PROCESSED_RAW_PATHS.items():
        value = record.get(field)
        if not isinstance(value, str):
            return None
        expected = raw_root / scan_id / member
        if Path(os.path.abspath(value)) != expected:
            return None
    try:
        points = processed_root / _safe_relative(
            record.get("filepath"), label="processed points"
        )
        instance_gt = processed_root / _safe_relative(
            record.get("instance_gt_filepath"), label="processed instance GT"
        )
    except RScanT2DevelopmentError:
        return None
    if not _existing_regular(points) or not _existing_regular(instance_gt):
        return None
    return points, instance_gt


def _sequence_record(
    value: object,
    *,
    key: str,
    scene: int,
    sub_scenes: tuple[int, int],
    processed_root: Path,
) -> tuple[Mapping[str, object], Path] | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get(key)
    if (
        not isinstance(raw, Mapping)
        or raw.get("type") != "validation"
        or raw.get("scene") != scene
        or raw.get("sub_scenes") != list(sub_scenes)
    ):
        return None
    try:
        change_path = processed_root / _safe_relative(
            raw.get("filepath"), label="sequence change GT"
        )
    except RScanT2DevelopmentError:
        return None
    if not _existing_regular(change_path):
        return None
    return raw, change_path


def _raw_complete(raw_root: Path, scan_ids: Sequence[str]) -> bool:
    return all(
        _existing_regular(raw_root / scan_id / member)
        for scan_id in scan_ids
        for member in RAW_RUNTIME_MEMBERS
    )


def _atomic_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RScanT2DevelopmentError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RScanT2DevelopmentError(f"output already exists: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_t2_development_manifest(
    *,
    metadata_path: str | Path,
    validation_list_path: str | Path,
    raw_root: str | Path,
    processed_root: str | Path,
    validation_database_path: str | Path,
    train_database_path: str | Path,
    sequence_database_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_provenance_path: str | Path,
    count: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Select and hash-bind complete T=2 pairs without consulting model results."""

    if type(count) is not int or count < 3:
        raise RScanT2DevelopmentError("development selection requires at least three pairs")
    output = Path(os.path.abspath(os.fspath(output_path)))
    if output.exists() or output.is_symlink():
        raise RScanT2DevelopmentError(f"output already exists: {output}")
    metadata = _regular_file(metadata_path, label="3RScan metadata")
    validation_list = _regular_file(validation_list_path, label="validation list")
    validation_database = _regular_file(
        validation_database_path, label="validation database"
    )
    train_database = _regular_file(train_database_path, label="train database")
    sequence_database = _regular_file(
        sequence_database_path, label="sequence database"
    )
    checkpoint = _regular_file(checkpoint_path, label="checkpoint")
    checkpoint_provenance = _regular_file(
        checkpoint_provenance_path, label="checkpoint provenance"
    )
    raw = Path(os.path.abspath(os.fspath(raw_root)))
    processed = Path(os.path.abspath(os.fspath(processed_root)))

    try:
        parsed_environments = load_rscan_metadata(metadata)
        raw_payload = loads_strict(
            metadata.read_text(encoding="utf-8"), label="3RScan metadata"
        )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as error:
        raise RScanT2DevelopmentError("3RScan metadata is invalid") from error
    if not isinstance(raw_payload, list):
        raise RScanT2DevelopmentError("3RScan metadata must contain a list")
    raw_by_reference = {
        item.get("reference"): item
        for item in raw_payload
        if isinstance(item, Mapping)
    }
    if len(raw_by_reference) != len(raw_payload):
        raise RScanT2DevelopmentError("3RScan environment identity is ambiguous")

    validation_ids = _validation_ids(validation_list)
    validation_by_scan = _database_by_scan(
        _load_yaml(validation_database, label="validation database"),
        label="validation database",
    )
    train_by_scan = _database_by_scan(
        _load_yaml(train_database, label="train database"),
        label="train database",
    )
    sequences = _load_yaml(sequence_database, label="sequence database")
    if not isinstance(sequences, Mapping) or not sequences:
        raise RScanT2DevelopmentError("sequence database must contain a mapping")

    candidates: list[tuple[object, Mapping[str, object], tuple[object, object], str, Path]] = []
    excluded = Counter()
    for environment in parsed_environments:
        sessions = environment.sessions
        scan_ids = tuple(session.scan_id for session in sessions)
        if environment.split != "validation":
            excluded["non_validation"] += 1
            continue
        if len(sessions) != 2:
            excluded["not_exact_t2"] += 1
            continue
        if not all(scan_id in validation_ids for scan_id in scan_ids):
            excluded["validation_list_incomplete"] += 1
            continue
        if not sessions[1].changes:
            excluded["no_declared_change"] += 1
            continue
        if sessions[1].evaluator_global_transform is None:
            excluded["global_alignment_missing"] += 1
            continue
        processed_records = tuple(validation_by_scan.get(scan_id) for scan_id in scan_ids)
        if any(record is None for record in processed_records):
            excluded["processed_database_incomplete"] += 1
            continue
        assert all(record is not None for record in processed_records)
        records = tuple(processed_records)
        normalized_paths = tuple(
            _processed_paths(
                record,
                scan_id=scan_id,
                raw_root=raw,
                processed_root=processed,
            )
            for scan_id, record in zip(scan_ids, records, strict=True)
        )
        if any(value is None for value in normalized_paths):
            excluded["processed_assets_incomplete"] += 1
            continue
        if not _raw_complete(raw, scan_ids):
            excluded["raw_assets_incomplete"] += 1
            continue
        first, second = records
        scene = first.get("scene")
        sub_scenes = (first.get("sub_scene"), second.get("sub_scene"))
        if (
            type(scene) is not int
            or second.get("scene") != scene
            or any(type(value) is not int for value in sub_scenes)
        ):
            excluded["processed_pair_identity_mismatch"] += 1
            continue
        key = (
            f"scene{scene:04d}_{sub_scenes[0]:02d}-"
            f"scene{scene:04d}_{sub_scenes[1]:02d}"
        )
        selected_sequence = _sequence_record(
            sequences,
            key=key,
            scene=scene,
            sub_scenes=sub_scenes,
            processed_root=processed,
        )
        if selected_sequence is None:
            excluded["sequence_database_incomplete"] += 1
            continue
        raw_environment = raw_by_reference.get(environment.reference_id)
        if not isinstance(raw_environment, Mapping):
            raise RScanT2DevelopmentError("raw and parsed metadata disagree")
        candidates.append(
            (
                environment,
                raw_environment,
                records,
                key,
                selected_sequence[1],
            )
        )

    candidates.sort(key=lambda value: value[0].reference_id)
    if len(candidates) < count:
        raise RScanT2DevelopmentError(
            f"only {len(candidates)} complete T=2 development pairs are available"
        )
    chosen = candidates[:count]
    selected_scan_ids = {
        session.scan_id
        for environment, _, _, _, _ in chosen
        for session in environment.sessions
    }
    overlap = selected_scan_ids.intersection(train_by_scan)
    if overlap:
        raise RScanT2DevelopmentError(
            "checkpoint training database overlap: " + ", ".join(sorted(overlap))
        )

    pairs: list[dict[str, object]] = []
    for environment, raw_environment, records, key, change_gt_path in chosen:
        raw_scan = raw_environment.get("scans")
        if not isinstance(raw_scan, list) or len(raw_scan) != 1:
            raise RScanT2DevelopmentError("raw T=2 metadata disagrees with parsed metadata")
        rescan_metadata = raw_scan[0]
        if not isinstance(rescan_metadata, Mapping):
            raise RScanT2DevelopmentError("raw rescan metadata is invalid")
        sessions_payload: list[dict[str, object]] = []
        for session, record in zip(environment.sessions, records, strict=True):
            scan_root = raw / session.scan_id
            points, instance_gt = _processed_paths(
                record,
                scan_id=session.scan_id,
                raw_root=raw,
                processed_root=processed,
            ) or (None, None)
            if points is None or instance_gt is None:
                raise RScanT2DevelopmentError("selected processed asset disappeared")
            sessions_payload.append(
                {
                    "visit_index": session.session_index,
                    "scan_id": session.scan_id,
                    "scene": record["scene"],
                    "sub_scene": record["sub_scene"],
                    "raw_assets": {
                        member: _file_record(
                            scan_root / member,
                            label=f"{session.scan_id}/{member}",
                        )
                        for member in RAW_RUNTIME_MEMBERS
                    },
                    "processed_assets": {
                        "points": _file_record(points, label="processed points"),
                        "instance_gt": _file_record(
                            instance_gt, label="processed instance GT"
                        ),
                    },
                }
            )
        transform = rescan_metadata.get("transform")
        ambiguity = raw_environment.get("ambiguity", [])
        pairs.append(
            {
                "pair_id": key,
                "environment_id": environment.reference_id,
                "split": "validation",
                "role": "development_selection",
                "sessions": sessions_payload,
                "common_method_inputs": {
                    "global_alignment": {
                        "direction": "rescan_row_vector_to_reference",
                        "storage": "row_major_flat_4x4",
                        "application": "homogeneous_row_vector_right_multiply",
                        "matrix": transform,
                    }
                },
                "sequence": {
                    "key": key,
                    "change_gt_binding": _file_record(
                        change_gt_path, label="sequence change GT"
                    ),
                },
                "evaluator_only": {
                    "changes": {
                        change_type: rescan_metadata.get(change_type, [])
                        for change_type in ("rigid", "nonrigid", "removed")
                    },
                    "ambiguity": ambiguity,
                    "cross_time_identity_source": "official_3rscan_metadata",
                    "symmetry_source": "official_rigid_change_records",
                },
            }
        )

    source_bindings = {
        "metadata": _file_record(metadata, label="3RScan metadata"),
        "validation_list": _file_record(validation_list, label="validation list"),
        "validation_database": _file_record(
            validation_database, label="validation database"
        ),
        "train_database": _file_record(train_database, label="train database"),
        "sequence_database": _file_record(
            sequence_database, label="sequence database"
        ),
        "checkpoint": _file_record(checkpoint, label="checkpoint"),
        "checkpoint_provenance": _file_record(
            checkpoint_provenance, label="checkpoint provenance"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "RSCAN_T2_DEV_V1",
        "status": "PASS",
        "role": "DEVELOPMENT_ONLY",
        "selection_rule": {
            "split": "validation",
            "exact_visit_count": 2,
            "complete_raw_and_processed_assets": True,
            "declared_change_required": True,
            "sort_key": "reference_scan_uuid",
            "take_first": count,
            "result_independent": True,
        },
        "checkpoint_provenance": {
            "training_environment_overlap": False,
            "validation_selection_overlap": True,
            "claim_boundary": "development_reproduction_not_unseen_generalization",
        },
        "source_bindings": source_bindings,
        "eligible_pair_count": len(candidates),
        "selected_pair_count": len(pairs),
        "excluded_environment_counts": dict(sorted(excluded.items())),
        "pairs": pairs,
    }
    _atomic_no_clobber(output, manifest)
    return manifest


def audit_t2_development_manifest(path: str | Path) -> dict[str, Any]:
    """Revalidate the frozen manifest schema and every external file binding."""

    source = _regular_file(path, label="RSCAN_T2_DEV_V1 manifest")
    try:
        manifest = loads_strict(
            source.read_text(encoding="utf-8"), label="RSCAN_T2_DEV_V1 manifest"
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RScanT2DevelopmentError("RSCAN_T2_DEV_V1 manifest is invalid") from error
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "role",
        "selection_rule",
        "checkpoint_provenance",
        "source_bindings",
        "eligible_pair_count",
        "selected_pair_count",
        "excluded_environment_counts",
        "pairs",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "RSCAN_T2_DEV_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("role") != "DEVELOPMENT_ONLY"
    ):
        raise RScanT2DevelopmentError("RSCAN_T2_DEV_V1 identity is invalid")
    provenance = manifest.get("checkpoint_provenance")
    if provenance != {
        "training_environment_overlap": False,
        "validation_selection_overlap": True,
        "claim_boundary": "development_reproduction_not_unseen_generalization",
    }:
        raise RScanT2DevelopmentError("checkpoint provenance is invalid")
    source_bindings = manifest.get("source_bindings")
    expected_sources = {
        "metadata",
        "validation_list",
        "validation_database",
        "train_database",
        "sequence_database",
        "checkpoint",
        "checkpoint_provenance",
    }
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != expected_sources:
        raise RScanT2DevelopmentError("source binding roles are invalid")
    for role in sorted(expected_sources):
        _validate_file_record(source_bindings[role], label=role)

    pairs = manifest.get("pairs")
    selected_count = manifest.get("selected_pair_count")
    eligible_count = manifest.get("eligible_pair_count")
    if (
        not isinstance(pairs, list)
        or type(selected_count) is not int
        or selected_count < 3
        or selected_count != len(pairs)
        or type(eligible_count) is not int
        or eligible_count < selected_count
    ):
        raise RScanT2DevelopmentError("selected pair counts are invalid")
    environment_ids: list[str] = []
    pair_ids: list[str] = []
    seen_scans: set[str] = set()
    for pair_index, pair in enumerate(pairs):
        label = f"pair[{pair_index}]"
        if not isinstance(pair, Mapping) or set(pair) != {
            "pair_id",
            "environment_id",
            "split",
            "role",
            "sessions",
            "common_method_inputs",
            "sequence",
            "evaluator_only",
        }:
            raise RScanT2DevelopmentError(f"{label} schema is invalid")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise RScanT2DevelopmentError(f"{label} identity is invalid")
        pair_ids.append(pair_id)
        environment_ids.append(
            _canonical_uuid(pair.get("environment_id"), label=f"{label} environment")
        )
        if pair.get("split") != "validation" or pair.get("role") != "development_selection":
            raise RScanT2DevelopmentError(f"{label} split or role is invalid")
        sessions = pair.get("sessions")
        if not isinstance(sessions, list) or len(sessions) != 2:
            raise RScanT2DevelopmentError(f"{label} must contain exactly two sessions")
        for visit_index, session in enumerate(sessions):
            session_label = f"{label}.sessions[{visit_index}]"
            if not isinstance(session, Mapping) or set(session) != {
                "visit_index",
                "scan_id",
                "scene",
                "sub_scene",
                "raw_assets",
                "processed_assets",
            }:
                raise RScanT2DevelopmentError(f"{session_label} schema is invalid")
            if session.get("visit_index") != visit_index:
                raise RScanT2DevelopmentError(f"{session_label} visit order is invalid")
            scan_id = _canonical_uuid(
                session.get("scan_id"), label=f"{session_label} scan"
            )
            if scan_id in seen_scans:
                raise RScanT2DevelopmentError("selected scan UUID is duplicated")
            seen_scans.add(scan_id)
            raw_assets = session.get("raw_assets")
            processed_assets = session.get("processed_assets")
            if not isinstance(raw_assets, Mapping) or set(raw_assets) != set(
                RAW_RUNTIME_MEMBERS
            ):
                raise RScanT2DevelopmentError(f"{session_label} raw assets are invalid")
            if not isinstance(processed_assets, Mapping) or set(processed_assets) != {
                "points",
                "instance_gt",
            }:
                raise RScanT2DevelopmentError(
                    f"{session_label} processed assets are invalid"
                )
            for member in RAW_RUNTIME_MEMBERS:
                record = _validate_file_record(
                    raw_assets[member], label=f"{session_label}.{member}"
                )
                if Path(str(record["path"])).parent.name != scan_id:
                    raise RScanT2DevelopmentError(
                        f"{session_label}.{member} scan path mismatch"
                    )
            for role in ("points", "instance_gt"):
                _validate_file_record(
                    processed_assets[role], label=f"{session_label}.{role}"
                )
        method_inputs = pair.get("common_method_inputs")
        if not isinstance(method_inputs, Mapping) or set(method_inputs) != {
            "global_alignment"
        }:
            raise RScanT2DevelopmentError(f"{label} method inputs are invalid")
        alignment = method_inputs.get("global_alignment")
        if (
            not isinstance(alignment, Mapping)
            or set(alignment) != {"direction", "storage", "application", "matrix"}
            or alignment.get("direction") != "rescan_row_vector_to_reference"
            or alignment.get("storage") != "row_major_flat_4x4"
            or alignment.get("application")
            != "homogeneous_row_vector_right_multiply"
        ):
            raise RScanT2DevelopmentError(f"{label} global alignment is invalid")
        matrix = alignment.get("matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 16
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in matrix
            )
        ):
            raise RScanT2DevelopmentError(f"{label} global alignment matrix is invalid")
        sequence = pair.get("sequence")
        if (
            not isinstance(sequence, Mapping)
            or set(sequence) != {"key", "change_gt_binding"}
            or sequence.get("key") != pair_id
        ):
            raise RScanT2DevelopmentError(f"{label} sequence is invalid")
        _validate_file_record(
            sequence["change_gt_binding"], label=f"{label}.sequence.change_gt"
        )
        evaluator = pair.get("evaluator_only")
        if not isinstance(evaluator, Mapping) or set(evaluator) != {
            "changes",
            "ambiguity",
            "cross_time_identity_source",
            "symmetry_source",
        }:
            raise RScanT2DevelopmentError(f"{label} evaluator-only schema is invalid")
        changes = evaluator.get("changes")
        if (
            not isinstance(changes, Mapping)
            or set(changes) != {"rigid", "nonrigid", "removed"}
            or any(not isinstance(changes[key], list) for key in changes)
            or not isinstance(evaluator.get("ambiguity"), list)
            or evaluator.get("cross_time_identity_source")
            != "official_3rscan_metadata"
            or evaluator.get("symmetry_source") != "official_rigid_change_records"
        ):
            raise RScanT2DevelopmentError(f"{label} evaluator-only values are invalid")
    if environment_ids != sorted(environment_ids) or len(pair_ids) != len(set(pair_ids)):
        raise RScanT2DevelopmentError("pair ordering or identity is invalid")
    return manifest


__all__ = [
    "RAW_RUNTIME_MEMBERS",
    "RScanT2DevelopmentError",
    "audit_t2_development_manifest",
    "build_t2_development_manifest",
]
