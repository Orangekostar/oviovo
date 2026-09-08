from __future__ import annotations

import hashlib

import pytest

from scripts.training.prepare_ovi_observation_training import (
    TrainingPreparationError,
    resolve_training_sources,
)


def _binding(path):
    content = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def test_training_sources_require_bound_official_train_endpoints(tmp_path) -> None:
    left = tmp_path / "left.npy"
    right = tmp_path / "right.npy"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    pair = {
        "role": "TRAIN",
        "official_split": "train",
        "environment_uuid": "scan-a",
        "pair_id": "pair-a",
        "sessions": [
            {
                "visit_id": 0,
                "scan_uuid": "scan-a",
                "processed_points": _binding(left),
            },
            {
                "visit_id": 1,
                "scan_uuid": "scan-b",
                "processed_points": _binding(right),
            },
        ],
    }

    sources = resolve_training_sources(pair)

    assert sources.environment_id == "scan-a"
    assert sources.scan_ids == ("scan-a", "scan-b")
    assert sources.processed_points == (left, right)

    pair["sessions"][1]["processed_points"]["sha256"] = "0" * 64
    with pytest.raises(TrainingPreparationError, match="binding mismatch"):
        resolve_training_sources(pair)


def test_training_sources_reject_dev_role(tmp_path) -> None:
    pair = {
        "role": "DEV",
        "official_split": "validation",
        "environment_uuid": "scan-a",
        "pair_id": "pair-a",
        "sessions": [],
    }

    with pytest.raises(TrainingPreparationError, match="official TRAIN"):
        resolve_training_sources(pair)
