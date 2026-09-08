"""Validated multi-environment pair collections and deterministic sampling."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class MultiEnvironmentTrainingError(ValueError):
    """Raised when a multi-environment training contract is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingPairEntry:
    environment_id: str
    pair_id: str
    artifact_root: Path
    split_record: Mapping[str, object]
    runtime_record: Mapping[str, object]


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TrainingPairIdentity:
    environment_id: str
    pair_id: str
    model_input_sha256: str
    observation_sha256: str
    training_target_sha256: str
    backbone_cache_sha256: str

    def __post_init__(self) -> None:
        if not self.environment_id or not self.pair_id:
            raise MultiEnvironmentTrainingError("training pair identity is empty")
        for value in (
            self.model_input_sha256,
            self.observation_sha256,
            self.training_target_sha256,
            self.backbone_cache_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise MultiEnvironmentTrainingError("training pair hash is invalid")

    def as_dict(self) -> dict[str, str]:
        return {
            "environment_id": self.environment_id,
            "pair_id": self.pair_id,
            "model_input_sha256": self.model_input_sha256,
            "observation_sha256": self.observation_sha256,
            "training_target_sha256": self.training_target_sha256,
            "backbone_cache_sha256": self.backbone_cache_sha256,
        }


def _pair_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise MultiEnvironmentTrainingError(
            "requested TRAIN pair IDs must be non-empty and unique"
        )
    return result


def collect_training_pair_entries(
    *,
    split: Mapping[str, object],
    runtime: Mapping[str, object],
    requested_pair_ids: Sequence[str],
) -> tuple[TrainingPairEntry, ...]:
    """Bind requested TRAIN pairs to exactly one split and runtime record."""

    requested = _pair_ids(requested_pair_ids)
    environments = split.get("environments")
    runtime_pairs = runtime.get("pairs")
    if not isinstance(environments, list) or not isinstance(runtime_pairs, Mapping):
        raise MultiEnvironmentTrainingError(
            "split or runtime pair collection is invalid"
        )
    entries: list[TrainingPairEntry] = []
    environment_ids: set[str] = set()
    for pair_id in requested:
        split_matches = [
            value
            for value in environments
            if isinstance(value, Mapping)
            and value.get("role") == "TRAIN"
            and value.get("pair_id") == pair_id
        ]
        runtime_matches = [
            value
            for key, value in runtime_pairs.items()
            if isinstance(key, str)
            and isinstance(value, Mapping)
            and value.get("role", key.upper()) == "TRAIN"
            and value.get("pair_id") == pair_id
        ]
        if len(split_matches) != 1 or len(runtime_matches) != 1:
            raise MultiEnvironmentTrainingError(
                f"TRAIN pair is absent or ambiguous: {pair_id}"
            )
        split_record = split_matches[0]
        runtime_record = runtime_matches[0]
        environment_id = split_record.get("environment_uuid")
        artifact_root = runtime_record.get("artifact_root")
        if not isinstance(environment_id, str) or not environment_id:
            raise MultiEnvironmentTrainingError("TRAIN environment identity is invalid")
        if environment_id in environment_ids:
            raise MultiEnvironmentTrainingError(
                "requested TRAIN pairs repeat an environment"
            )
        if not isinstance(artifact_root, str) or not artifact_root:
            raise MultiEnvironmentTrainingError("TRAIN artifact root is invalid")
        environment_ids.add(environment_id)
        entries.append(
            TrainingPairEntry(
                environment_id=environment_id,
                pair_id=pair_id,
                artifact_root=Path(artifact_root).absolute(),
                split_record=dict(split_record),
                runtime_record=dict(runtime_record),
            )
        )
    return tuple(entries)


def build_training_dataset_manifest(
    identities: Sequence[TrainingPairIdentity],
    *,
    data_seed: int,
    gradient_accumulation: int,
) -> dict[str, object]:
    records = tuple(identities)
    if (
        not records
        or any(not isinstance(value, TrainingPairIdentity) for value in records)
        or len({value.pair_id for value in records}) != len(records)
        or len({value.environment_id for value in records}) != len(records)
    ):
        raise MultiEnvironmentTrainingError("training dataset identities are invalid")
    if type(data_seed) is not int or data_seed <= 0:
        raise MultiEnvironmentTrainingError("data seed must be positive")
    if type(gradient_accumulation) is not int or gradient_accumulation <= 0:
        raise MultiEnvironmentTrainingError("gradient accumulation must be positive")
    return {
        "schema_version": 3,
        "artifact_id": "OVI_OBSERVATION_TRAINING_DATASET_MULTIENV_V3",
        "environment_ids": [value.environment_id for value in records],
        "pairs": [value.as_dict() for value in records],
        "sampling": {
            "policy": "environment_balanced_epoch_shuffle",
            "data_seed": data_seed,
            "gradient_accumulation": gradient_accumulation,
        },
    }


def aggregate_pair_binding_sha256(
    identities: Sequence[TrainingPairIdentity], field: str
) -> str:
    if field not in {"observation_sha256", "backbone_cache_sha256"}:
        raise MultiEnvironmentTrainingError("aggregate pair binding field is invalid")
    records = tuple(identities)
    if not records or any(
        not isinstance(value, TrainingPairIdentity) for value in records
    ):
        raise MultiEnvironmentTrainingError("training dataset identities are invalid")
    payload = [
        {"pair_id": value.pair_id, field: getattr(value, field)} for value in records
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EnvironmentBalancedPairSampler:
    """Visit every selected environment once per independently shuffled epoch."""

    def __init__(self, *, pair_ids: Sequence[str], seed: int) -> None:
        self._pair_ids = _pair_ids(pair_ids)
        if type(seed) is not int or seed <= 0:
            raise MultiEnvironmentTrainingError("sampler seed must be positive")
        self._seed = seed
        self._random = random.Random(seed)
        self._epoch = -1
        self._cursor = 0
        self._order: list[str] = []
        self._micro_steps_completed = 0
        self._exposure_counts = {pair_id: 0 for pair_id in self._pair_ids}
        self._start_epoch()

    def _start_epoch(self) -> None:
        self._epoch += 1
        self._order = list(self._pair_ids)
        self._random.shuffle(self._order)
        self._cursor = 0

    def next_pair_id(self) -> str:
        if self._cursor == len(self._order):
            self._start_epoch()
        pair_id = self._order[self._cursor]
        self._cursor += 1
        self._micro_steps_completed += 1
        self._exposure_counts[pair_id] += 1
        return pair_id

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "pair_ids": list(self._pair_ids),
            "seed": self._seed,
            "epoch": self._epoch,
            "cursor": self._cursor,
            "order": list(self._order),
            "micro_steps_completed": self._micro_steps_completed,
            "exposure_counts": dict(self._exposure_counts),
            "random_state": self._random.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise MultiEnvironmentTrainingError("sampler state is invalid")
        state_pair_ids = state.get("pair_ids")
        if (
            not isinstance(state_pair_ids, list)
            or tuple(state_pair_ids) != self._pair_ids
        ):
            raise MultiEnvironmentTrainingError("sampler dataset changed on resume")
        order = state.get("order")
        epoch = state.get("epoch")
        cursor = state.get("cursor")
        micro_steps_completed = state.get("micro_steps_completed")
        exposure_counts = state.get("exposure_counts")
        if (
            state.get("schema_version") != 1
            or state.get("seed") != self._seed
            or not isinstance(order, list)
            or len(order) != len(self._pair_ids)
            or set(order) != set(self._pair_ids)
            or type(epoch) is not int
            or epoch < 0
            or type(cursor) is not int
            or not 0 <= cursor <= len(order)
            or type(micro_steps_completed) is not int
            or micro_steps_completed < 0
            or not isinstance(exposure_counts, Mapping)
            or set(exposure_counts) != set(self._pair_ids)
            or any(
                type(value) is not int or value < 0
                for value in exposure_counts.values()
            )
            or sum(exposure_counts.values()) != micro_steps_completed
        ):
            raise MultiEnvironmentTrainingError("sampler state is invalid")
        try:
            self._random.setstate(state["random_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise MultiEnvironmentTrainingError(
                "sampler random state is invalid"
            ) from error
        self._epoch = epoch
        self._cursor = cursor
        self._order = list(order)
        self._micro_steps_completed = micro_steps_completed
        self._exposure_counts = dict(exposure_counts)


__all__ = [
    "EnvironmentBalancedPairSampler",
    "MultiEnvironmentTrainingError",
    "TrainingPairEntry",
    "TrainingPairIdentity",
    "aggregate_pair_binding_sha256",
    "build_training_dataset_manifest",
    "collect_training_pair_entries",
]
