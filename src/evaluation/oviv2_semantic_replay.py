from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Real

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.dense_projection import DenseSemanticConfig, DenseSemanticIntegrator
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.observations import FrameObservation, ObservationKind


SEMANTIC_REPLAY_SOURCE_RELATIVE_PATHS = {
    "data_structures": "src/core/data_structures.py",
    "dense_projection": "src/oviv2/dense_projection.py",
    "dense_semantics": "src/oviv2/dense_semantics.py",
    "evidence": "src/oviv2/evidence.py",
    "observations": "src/oviv2/observations.py",
    "replica_dataset": "src/datasets/replica.py",
    "replay_core": "src/evaluation/oviv2_semantic_replay.py",
    "replay_cli": "scripts/evaluation/replay_oviv2_semantic_evidence.py",
    "runner": "scripts/run_oviv2_replica.py",
    "structure": "src/oviv2/structure.py",
}


def semantic_replay_algorithm_hash(
    *,
    dense_config: Mapping[str, object],
    entropy_power_by_class: Mapping[str, object],
    semantic_support_scale: object,
    source_hashes: Mapping[str, object],
    structure_config: Mapping[str, object],
) -> str:
    payload = {
        "dense_config": dict(dense_config),
        "entropy_power_by_class": dict(entropy_power_by_class),
        "semantic_support_scale": semantic_support_scale,
        "source_hashes": dict(source_hashes),
        "structure_config": dict(structure_config),
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("semantic replay algorithm payload must be valid JSON") from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SemanticReplayInput:
    frame: Frame
    dense_semantics: DenseSemanticFrame
    structure_observations: tuple[FrameObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")
        if not isinstance(self.dense_semantics, DenseSemanticFrame):
            raise TypeError("dense_semantics must be a DenseSemanticFrame")
        if not isinstance(self.structure_observations, tuple):
            raise TypeError("structure_observations must be a tuple")
        for observation in self.structure_observations:
            if not isinstance(observation, FrameObservation):
                raise TypeError(
                    "structure_observations must contain FrameObservation values"
                )
            if observation.kind is not ObservationKind.STRUCTURE:
                raise ValueError("semantic replay accepts only structure observations")
            if observation.frame_id != self.frame.frame_id:
                raise ValueError("structure observation frame does not match replay frame")


@dataclass(frozen=True)
class SemanticReplayFrameResult:
    frame_id: int
    revision: int
    dense_sampled_pixel_count: int
    dense_valid_pixel_count: int
    dense_updated_voxel_count: int
    structure_observation_count: int
    structure_updated_voxel_count: int


@dataclass(frozen=True)
class SemanticReplayResult:
    evidence: SparseEvidenceStore
    frame_results: tuple[SemanticReplayFrameResult, ...]
    structure_replayed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, SparseEvidenceStore):
            raise TypeError("evidence must be a SparseEvidenceStore")
        if not isinstance(self.frame_results, tuple):
            raise TypeError("frame_results must be a tuple")
        if self.structure_replayed is not True:
            raise ValueError("semantic replay requires structure evidence")


def semantic_evidence_equal(
    left: SparseEvidenceStore,
    right: SparseEvidenceStore,
) -> bool:
    """Return whether two stores contain byte-identical semantic evidence."""
    if not isinstance(left, SparseEvidenceStore) or not isinstance(
        right,
        SparseEvidenceStore,
    ):
        raise TypeError("semantic evidence values must be SparseEvidenceStore instances")
    if left.config != right.config:
        return False
    for block_key in left._blocks.keys() | right._blocks.keys():
        left_block = left._blocks.get(block_key)
        right_block = right._blocks.get(block_key)
        if left_block is None:
            left_block = left._new_block()
        if right_block is None:
            right_block = right._new_block()
        for name in (
            "semantic_ids",
            "semantic_support",
            "semantic_revisions",
        ):
            if not np.array_equal(getattr(left_block, name), getattr(right_block, name)):
                return False
    return True


def replay_semantic_evidence(
    inputs: Iterable[SemanticReplayInput],
    *,
    evidence_config: EvidenceConfig,
    dense_config: DenseSemanticConfig,
    semantic_support_scale: float = 1.0,
    entropy_power_by_class: Mapping[int, float] | None = None,
) -> SemanticReplayResult:
    if not isinstance(evidence_config, EvidenceConfig):
        raise TypeError("evidence_config must be an EvidenceConfig")
    if not isinstance(dense_config, DenseSemanticConfig):
        raise TypeError("dense_config must be a DenseSemanticConfig")
    if isinstance(semantic_support_scale, (bool, np.bool_)) or not isinstance(
        semantic_support_scale,
        Real,
    ):
        raise TypeError("semantic_support_scale must be a real number")
    support_scale = float(semantic_support_scale)
    if not np.isfinite(support_scale) or support_scale <= 0.0:
        raise ValueError("semantic_support_scale must be finite and positive")

    evidence = SparseEvidenceStore(evidence_config)
    integrator = DenseSemanticIntegrator(dense_config)
    frame_results: list[SemanticReplayFrameResult] = []
    previous_frame_id = -1
    for revision, item in enumerate(inputs, start=1):
        if not isinstance(item, SemanticReplayInput):
            raise TypeError("inputs must contain SemanticReplayInput values")
        frame_id = int(item.frame.frame_id)
        if frame_id <= previous_frame_id:
            raise ValueError("replay frame IDs must increase monotonically")
        previous_frame_id = frame_id
        dense_result = integrator.integrate(
            item.frame,
            item.dense_semantics,
            evidence,
            revision,
            entropy_power_by_class=entropy_power_by_class,
        )
        structure_keys: set[tuple[int, int, int]] = set()
        for observation in item.structure_observations:
            support = max(observation.confidence * support_scale, 1e-9)
            for voxel_key in observation.voxel_keys:
                evidence.update_semantic(
                    voxel_key,
                    observation.semantic_id,
                    support,
                    revision,
                )
                structure_keys.add(voxel_key)
        frame_results.append(
            SemanticReplayFrameResult(
                frame_id=frame_id,
                revision=revision,
                dense_sampled_pixel_count=dense_result.sampled_pixel_count,
                dense_valid_pixel_count=dense_result.valid_pixel_count,
                dense_updated_voxel_count=dense_result.updated_voxel_count,
                structure_observation_count=len(item.structure_observations),
                structure_updated_voxel_count=len(structure_keys),
            )
        )
    return SemanticReplayResult(evidence=evidence, frame_results=tuple(frame_results))
