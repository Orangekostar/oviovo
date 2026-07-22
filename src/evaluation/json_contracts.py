"""Strict JSON parsing for hash-bound evaluation contracts."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} contains non-finite JSON number")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label=label)


def loads_strict(content: str, *, label: str) -> Any:
    """Parse JSON while rejecting duplicate keys and all non-finite numbers."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"{label} contains duplicate JSON key {key}")
            payload[key] = value
        return payload

    payload = json.loads(
        content,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    _reject_nonfinite(payload, label=label)
    return payload
