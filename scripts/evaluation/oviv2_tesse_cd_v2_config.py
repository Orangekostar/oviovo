"""Algorithm identity normalization for the TESSE-CD v2 protocol."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    canonical_algorithm_config as _canonical_t1_algorithm_config,
)


V2_ONLY_SCENE_CONFIG_FIELDS = frozenset(
    {"temporal_frontend_cache_dir", "temporal_frontend_manifest"}
)


def canonical_algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonical_t1_algorithm_config(config)
    return {
        key: value
        for key, value in normalized.items()
        if key not in V2_ONLY_SCENE_CONFIG_FIELDS
    }


def canonical_algorithm_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_algorithm_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
