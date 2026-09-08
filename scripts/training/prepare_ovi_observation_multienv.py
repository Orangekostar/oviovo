#!/usr/bin/env python3
"""Freeze V3 TRAIN pairs into the existing OVI/D2 selection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.datasets.rscan import RScanDatasetError, load_rscan_metadata
from src.training.ovi_observation_data import (
    ObservationTrainingDataError,
    load_split_pair,
)


class MultiEnvironmentAssetPreparationError(ValueError):
    """Raised when a frozen V3 TRAIN pair cannot be materialized safely."""


def _json_object(path: str | Path, *, label: str) -> dict[str, object]:
    source = Path(path).absolute()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultiEnvironmentAssetPreparationError(f"{label} is unavailable") from error
    if not isinstance(payload, dict):
        raise MultiEnvironmentAssetPreparationError(f"{label} must be an object")
    return payload


def _yaml_value(path: str | Path, *, label: str) -> object:
    source = Path(path).absolute()
    try:
        return yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise MultiEnvironmentAssetPreparationError(f"{label} is unavailable") from error


def _file_record(path: str | Path) -> dict[str, object]:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise MultiEnvironmentAssetPreparationError(
            f"source file is unavailable: {source}"
        )
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return {
        "path": str(source),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _validate_file_record(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise MultiEnvironmentAssetPreparationError(f"{label} binding is invalid")
    path = value.get("path")
    if not isinstance(path, str):
        raise MultiEnvironmentAssetPreparationError(f"{label} path is invalid")
    if dict(value) != _file_record(path):
        raise MultiEnvironmentAssetPreparationError(f"{label} binding mismatch")
    return Path(path).absolute()


def _resolve_under(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MultiEnvironmentAssetPreparationError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MultiEnvironmentAssetPreparationError(f"{label} path must be relative")
    return root / relative


def _database_by_scan(value: object, *, raw_root: Path) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list):
        raise MultiEnvironmentAssetPreparationError("processed database must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for record in value:
        raw_path = record.get("raw_filepath") if isinstance(record, Mapping) else None
        if not isinstance(raw_path, str):
            raise MultiEnvironmentAssetPreparationError(
                "processed database scan binding is invalid"
            )
        absolute = Path(raw_path).absolute()
        scan_id = absolute.parent.name
        if absolute.parent != raw_root / scan_id or scan_id in result:
            raise MultiEnvironmentAssetPreparationError(
                "processed database repeats or misbinds a scan"
            )
        result[scan_id] = record
    return result


def _raw_environment_by_reference(metadata_path: Path) -> dict[str, Mapping[str, object]]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultiEnvironmentAssetPreparationError("3RScan metadata is unavailable") from error
    if not isinstance(payload, list):
        raise MultiEnvironmentAssetPreparationError("3RScan metadata must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for value in payload:
        reference = value.get("reference") if isinstance(value, Mapping) else None
        if not isinstance(reference, str) or reference in result:
            raise MultiEnvironmentAssetPreparationError(
                "3RScan environment identity is invalid"
            )
        result[reference] = value
    return result


def build_training_selection_manifest(
    *,
    split_manifest: str | Path,
    metadata_path: str | Path,
    processed_database_path: str | Path,
    sequence_database_path: str | Path,
    raw_root: str | Path,
    processed_root: str | Path,
    pair_ids: Sequence[str],
) -> dict[str, object]:
    """Build the existing pair-view contract for explicitly frozen V3 TRAIN pairs."""

    requested = tuple(pair_ids)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(not isinstance(value, str) or not value for value in requested)
    ):
        raise MultiEnvironmentAssetPreparationError(
            "TRAIN pair IDs must be non-empty and unique"
        )
    split_path = Path(split_manifest).absolute()
    metadata = Path(metadata_path).absolute()
    database_path = Path(processed_database_path).absolute()
    sequence_path = Path(sequence_database_path).absolute()
    raw = Path(raw_root).absolute()
    processed = Path(processed_root).absolute()
    try:
        environments = {
            environment.reference_id: environment
            for environment in load_rscan_metadata(metadata)
        }
    except RScanDatasetError as error:
        raise MultiEnvironmentAssetPreparationError(str(error)) from error
    raw_environments = _raw_environment_by_reference(metadata)
    database = _database_by_scan(
        _yaml_value(database_path, label="processed database"), raw_root=raw
    )
    sequences = _yaml_value(sequence_path, label="sequence database")
    if not isinstance(sequences, Mapping):
        raise MultiEnvironmentAssetPreparationError(
            "sequence database must be a mapping"
        )

    pairs: list[dict[str, object]] = []
    for pair_id in requested:
        try:
            split_pair = load_split_pair(
                split_path, pair_id=pair_id, required_role="TRAIN"
            )
        except ObservationTrainingDataError as error:
            raise MultiEnvironmentAssetPreparationError(
                f"requested pair is not a valid TRAIN pair: {pair_id}"
            ) from error
        if split_pair.get("official_split") != "train":
            raise MultiEnvironmentAssetPreparationError(
                f"requested pair is not in official TRAIN: {pair_id}"
            )
        environment_id = split_pair.get("environment_uuid")
        sessions = split_pair.get("sessions")
        if not isinstance(environment_id, str) or not isinstance(sessions, list):
            raise MultiEnvironmentAssetPreparationError("split TRAIN pair is invalid")
        environment = environments.get(environment_id)
        raw_environment = raw_environments.get(environment_id)
        if (
            environment is None
            or environment.split != "train"
            or not isinstance(raw_environment, Mapping)
        ):
            raise MultiEnvironmentAssetPreparationError(
                "split and official TRAIN metadata disagree"
            )
        official_sessions = {session.scan_id: session for session in environment.sessions}

        session_rows: list[dict[str, object]] = []
        scene_values: list[int] = []
        sub_scene_values: list[int] = []
        selected_scan_ids: list[str] = []
        for visit_id, split_session in enumerate(sessions):
            if not isinstance(split_session, Mapping):
                raise MultiEnvironmentAssetPreparationError("split session is invalid")
            scan_id = split_session.get("scan_uuid")
            if (
                split_session.get("visit_id") != visit_id
                or not isinstance(scan_id, str)
                or scan_id not in official_sessions
                or scan_id not in database
            ):
                raise MultiEnvironmentAssetPreparationError(
                    "split session identity is not source-backed"
                )
            point_path = _validate_file_record(
                split_session.get("processed_points"), label="processed points"
            )
            database_record = database[scan_id]
            database_point = _resolve_under(
                processed, database_record.get("filepath"), label="processed points"
            ).absolute()
            instance_gt = _resolve_under(
                processed,
                database_record.get("instance_gt_filepath"),
                label="processed instance GT",
            )
            scene = database_record.get("scene")
            sub_scene = database_record.get("sub_scene")
            if (
                point_path != database_point
                or type(scene) is not int
                or type(sub_scene) is not int
            ):
                raise MultiEnvironmentAssetPreparationError(
                    "split and processed database disagree"
                )
            scene_values.append(scene)
            sub_scene_values.append(sub_scene)
            selected_scan_ids.append(scan_id)
            session_rows.append(
                {
                    "visit_index": visit_id,
                    "scan_id": scan_id,
                    "scene": scene,
                    "sub_scene": sub_scene,
                    "processed_assets": {
                        "points": _file_record(point_path),
                        "instance_gt": _file_record(instance_gt),
                    },
                }
            )
        if selected_scan_ids[0] != environment_id:
            raise MultiEnvironmentAssetPreparationError(
                "TRAIN pair reference is not its environment UUID"
            )
        rescan = official_sessions[selected_scan_ids[1]]
        if rescan.evaluator_global_transform is None:
            raise MultiEnvironmentAssetPreparationError(
                "selected TRAIN pair has no official global alignment"
            )
        sequence = sequences.get(pair_id)
        if (
            not isinstance(sequence, Mapping)
            or sequence.get("type") != "train"
            or scene_values[0] != scene_values[1]
            or sequence.get("scene") != scene_values[0]
            or sequence.get("sub_scenes") != sub_scene_values
        ):
            raise MultiEnvironmentAssetPreparationError(
                "selected TRAIN sequence identity mismatch"
            )
        change_gt = _resolve_under(
            processed, sequence.get("filepath"), label="sequence change GT"
        )
        changes = tuple(rescan.changes)
        pairs.append(
            {
                "pair_id": pair_id,
                "environment_id": environment_id,
                "split": "train",
                "role": "training_selection",
                "sessions": session_rows,
                "common_method_inputs": {
                    "global_alignment": {
                        "direction": "rescan_row_vector_to_reference",
                        "storage": "row_major_flat_4x4",
                        "application": "homogeneous_row_vector_right_multiply",
                        "matrix": list(rescan.evaluator_global_transform),
                    }
                },
                "sequence": {
                    "key": pair_id,
                    "change_gt_binding": _file_record(change_gt),
                },
                "evaluator_only": {
                    "changes": {
                        f"{change_type}_reference_ids": sorted(
                            {
                                change.reference_instance_id
                                for change in changes
                                if change.change_type == change_type
                            }
                        )
                        for change_type in ("rigid", "nonrigid", "removed")
                    },
                    "ambiguity": raw_environment.get("ambiguity", []),
                    "cross_time_identity_source": "official_3rscan_metadata",
                },
            }
        )

    return {
        "schema_version": 3,
        "artifact_id": "OVI_RESCENE_OBSERVATION_QUERY_TRAIN_SELECTION_MULTIENV_V3",
        "status": "PASS",
        "role": "TRAIN",
        "source_bindings": {
            "split_manifest": _file_record(split_path),
            "metadata": _file_record(metadata),
            "processed_database": _file_record(database_path),
            "sequence_database": _file_record(sequence_path),
        },
        "selected_pair_count": len(pairs),
        "pairs": pairs,
    }


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise MultiEnvironmentAssetPreparationError(f"output already exists: {output}")
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
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--processed-database", type=Path, required=True)
    parser.add_argument("--sequence-database", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--pair-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        payload = build_training_selection_manifest(
            split_manifest=args.split_manifest,
            metadata_path=args.metadata,
            processed_database_path=args.processed_database,
            sequence_database_path=args.sequence_database,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            pair_ids=args.pair_ids,
        )
        _write_atomic(args.output, payload)
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps({"status": payload["status"], "output": str(args.output.absolute())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MultiEnvironmentAssetPreparationError",
    "build_training_selection_manifest",
    "main",
]
