#!/usr/bin/env python3
"""Build source-bound observation-query supervision for one official TRAIN pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluation.prepare_ovi_observations import load_model_observation_bundle
from src.oviv2.observation_query.contracts import load_observation_bank
from src.training.ovi_observation_data import (
    LabelTransferConfig,
    ObservationTrainingDataError,
    build_observation_training_sample,
    load_native_processed_labels,
    load_official_pair_training_metadata,
    load_rescene_class_mapping,
    load_split_pair,
    save_observation_training_sample,
)


class TrainingPreparationError(ValueError):
    """Raised when TRAIN supervision inputs are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class TrainingPairSources:
    pair_id: str
    environment_id: str
    scan_ids: tuple[str, str]
    processed_points: tuple[Path, Path]


def _file_record(path: str | Path) -> dict[str, object]:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise TrainingPreparationError(f"source file is unavailable: {source}")
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


def _bound_path(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise TrainingPreparationError(f"{label} binding schema is invalid")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise TrainingPreparationError(f"{label} binding path is invalid")
    observed = _file_record(path_value)
    if dict(value) != observed:
        raise TrainingPreparationError(f"{label} binding mismatch")
    return Path(path_value)


def _load_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, object]]:
    source = Path(path).absolute()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingPreparationError(f"{label} is unavailable") from error
    if not isinstance(payload, dict):
        raise TrainingPreparationError(f"{label} must be a JSON object")
    return source, payload


def resolve_training_sources(pair: Mapping[str, object]) -> TrainingPairSources:
    """Validate and resolve both processed endpoints of one official TRAIN pair."""

    if pair.get("role") != "TRAIN" or pair.get("official_split") != "train":
        raise TrainingPreparationError("pair must be an official TRAIN environment")
    pair_id = pair.get("pair_id")
    environment_id = pair.get("environment_uuid")
    sessions = pair.get("sessions")
    if (
        not isinstance(pair_id, str)
        or not pair_id
        or not isinstance(environment_id, str)
        or not environment_id
        or not isinstance(sessions, list)
        or len(sessions) != 2
    ):
        raise TrainingPreparationError("TRAIN pair schema is invalid")
    scan_ids: list[str] = []
    processed: list[Path] = []
    for visit_id, session in enumerate(sessions):
        if not isinstance(session, Mapping) or session.get("visit_id") != visit_id:
            raise TrainingPreparationError("TRAIN sessions must be ordered by visit")
        scan_id = session.get("scan_uuid")
        if not isinstance(scan_id, str) or not scan_id:
            raise TrainingPreparationError("TRAIN scan identity is invalid")
        scan_ids.append(scan_id)
        processed.append(
            _bound_path(session.get("processed_points"), label="processed points")
        )
    if scan_ids[0] != environment_id or scan_ids[0] == scan_ids[1]:
        raise TrainingPreparationError("TRAIN environment and scan identities disagree")
    return TrainingPairSources(
        pair_id=pair_id,
        environment_id=environment_id,
        scan_ids=(scan_ids[0], scan_ids[1]),
        processed_points=(processed[0], processed[1]),
    )


def _training_diagnostics(sample) -> dict[str, object]:
    model_visits = sample.model_input.model_visit_ids
    region_visits = sample.observations.region_visit_ids
    edge_rows = np.repeat(
        np.arange(len(region_visits)), np.diff(sample.observations.csr_indptr)
    )
    per_visit: list[dict[str, object]] = []
    for visit_id in (0, 1):
        model_mask = model_visits == visit_id
        region_mask = region_visits == visit_id
        supported = np.unique(
            sample.observations.csr_model_indices[region_visits[edge_rows] == visit_id]
        )
        per_visit.append(
            {
                "visit_id": visit_id,
                "model_count": int(np.count_nonzero(model_mask)),
                "observation_region_count": int(np.count_nonzero(region_mask)),
                "observation_supported_model_count": len(supported),
                "label_valid_count": int(
                    sample.label_valid[model_mask].count_nonzero().item()
                ),
                "region_valid_count": int(
                    sample.region_label_valid[region_mask].count_nonzero().item()
                ),
            }
        )
    masks = sample.instance_masks.detach().cpu().numpy() > 0.0
    represented = [
        np.any(masks[:, model_visits == visit_id], axis=1) for visit_id in (0, 1)
    ]
    persistent = int(np.count_nonzero(represented[0] & represented[1]))
    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_DIAGNOSTICS_V2",
        "status": "PASS",
        "pair_id": sample.observations.pair_id,
        "model_count": len(model_visits),
        "observation_region_count": len(region_visits),
        "label_valid_count": int(sample.label_valid.count_nonzero().item()),
        "class_valid_target_count": int(sample.class_valid.count_nonzero().item()),
        "region_valid_count": int(sample.region_label_valid.count_nonzero().item()),
        "temporal_identity_count": len(sample.temporal_identity_keys),
        "persistent_identity_count": persistent,
        "per_visit": per_visit,
    }


def prepare_training_artifact(
    *,
    runtime_config: str | Path,
    pilot_config: str | Path,
    split_manifest: str | Path,
    pair_id: str,
    model_bundle: str | Path,
    observation_bank: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    runtime_path, runtime = _load_json(runtime_config, label="runtime config")
    pilot_path, pilot = _load_json(pilot_config, label="pilot config")
    split_path = Path(split_manifest).absolute()
    try:
        pair = load_split_pair(split_path, pair_id=pair_id, required_role="TRAIN")
    except ObservationTrainingDataError as error:
        raise TrainingPreparationError(str(error)) from error
    sources = resolve_training_sources(pair)
    assets = runtime.get("assets")
    target_config = pilot.get("training_targets")
    if not isinstance(assets, Mapping) or not isinstance(target_config, Mapping):
        raise TrainingPreparationError("runtime assets or training targets are invalid")
    raw_root = assets.get("raw_3rscan_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise TrainingPreparationError("3RScan raw root is invalid")
    metadata_path = Path(raw_root).absolute() / "3RScan.json"

    model_input, model_points, _center, _model_manifest = load_model_observation_bundle(
        model_bundle
    )
    observations = load_observation_bank(observation_bank)
    if observations.pair_id != sources.pair_id:
        raise TrainingPreparationError("ObservationBank pair identity mismatch")
    labels = tuple(
        load_native_processed_labels(path, visit_id=visit_id)
        for visit_id, path in enumerate(sources.processed_points)
    )
    official = load_official_pair_training_metadata(
        metadata_path,
        reference_scan_uuid=sources.scan_ids[0],
        rescan_uuid=sources.scan_ids[1],
        reference_instance_ids={int(value) for value in labels[0].instance_ids if value > 0},
        rescan_instance_ids={int(value) for value in labels[1].instance_ids if value > 0},
    )
    label_database = target_config.get("label_database")
    if not isinstance(label_database, str) or not label_database:
        raise TrainingPreparationError("label database path is invalid")
    class_mapping = load_rescene_class_mapping(
        label_database,
        expected_validation_label_count=int(
            target_config["expected_validation_label_count"]
        ),
        label_offset=int(target_config["label_offset"]),
    )
    transfer = LabelTransferConfig(
        maximum_distance_m=float(target_config["maximum_distance_m"]),
        minimum_neighbors=int(target_config["minimum_neighbors"]),
        minimum_consensus_fraction=float(
            target_config["minimum_consensus_fraction"]
        ),
        maximum_neighbors=int(target_config["maximum_neighbors"]),
        trusted_background_raw_ids=tuple(
            int(value)
            for value in target_config["trusted_background_raw_semantic_ids"]
        ),
        ignored_raw_semantic_ids=tuple(
            int(value) for value in target_config["ignored_raw_semantic_ids"]
        ),
        minimum_region_valid_fraction=float(
            target_config["minimum_region_valid_fraction"]
        ),
    )
    target_sources = {
        "runtime_config": _file_record(runtime_path),
        "pilot_config": _file_record(pilot_path),
        "split_manifest": _file_record(split_path),
        "official_metadata": _file_record(metadata_path),
        "processed_points": [
            _file_record(path) for path in sources.processed_points
        ],
        "label_database": _file_record(label_database),
        "official_split": "train",
        "environment_id": sources.environment_id,
    }
    sample = build_observation_training_sample(
        model_input=model_input,
        observations=observations,
        model_points_reference_xyz=model_points,
        visit_labels=(labels[0], labels[1]),
        identity_rules=official.identity_rules,
        raw_semantic_to_model_class=class_mapping,
        config=transfer,
        expected_pair_id=sources.pair_id,
        target_source_manifest=target_sources,
    )
    paths = save_observation_training_sample(sample, output_root)
    diagnostics = _training_diagnostics(sample)
    diagnostics_path = paths.root / "diagnostics.json"
    with diagnostics_path.open("xb") as stream:
        stream.write(
            (
                json.dumps(
                    diagnostics,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        stream.flush()
        os.fsync(stream.fileno())
    return {**diagnostics, "output_root": str(paths.root)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--observation-bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_training_artifact(
            runtime_config=args.runtime_config,
            pilot_config=args.pilot_config,
            split_manifest=args.split_manifest,
            pair_id=args.pair_id,
            model_bundle=args.model_bundle,
            observation_bank=args.observation_bank,
            output_root=args.output_root,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TrainingPairSources",
    "TrainingPreparationError",
    "main",
    "prepare_training_artifact",
    "resolve_training_sources",
]
