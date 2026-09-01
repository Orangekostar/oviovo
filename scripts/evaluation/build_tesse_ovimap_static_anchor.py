#!/usr/bin/env python3
"""Build a causal TESSE-CD OVI-MAP static anchor package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import stat
from typing import Any


@dataclass(frozen=True)
class CausalPrefixContract:
    scene: str
    first_intervention_frame: int
    frame_ids: tuple[int, ...]
    maximum_source_frame: int
    schedule_sha256: str
    rgbd_export_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene, str) or not self.scene.strip():
            raise ValueError("scene must be non-empty")
        if self.scene != self.scene.strip():
            raise ValueError("scene must not contain surrounding whitespace")
        if (
            isinstance(self.first_intervention_frame, bool)
            or not isinstance(self.first_intervention_frame, Integral)
            or int(self.first_intervention_frame) <= 0
        ):
            raise ValueError("first_intervention_frame must be positive")
        expected = tuple(range(int(self.first_intervention_frame)))
        if self.frame_ids != expected:
            raise ValueError("frame_ids must be the complete causal prefix")
        if self.maximum_source_frame != expected[-1]:
            raise ValueError("maximum_source_frame must end the causal prefix")
        for value in (
            self.schedule_sha256,
            self.rgbd_export_manifest_sha256,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("source hashes must be lowercase SHA-256 values")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks")


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    _reject_symlink_components(path, label=label)
    try:
        status = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    data = path.read_bytes()
    final = path.stat(follow_symlinks=False)
    if (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns) != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise ValueError(f"{label} changed while being read")
    return data


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _integer(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return normalized


def _scene_events(schedule: Mapping[str, Any], scene: str) -> tuple[Mapping[str, Any], ...]:
    scenes = schedule.get("scenes")
    if not isinstance(scenes, Mapping) or scene not in scenes:
        raise ValueError(f"scene is not declared by causal schedule: {scene}")
    scene_payload = scenes[scene]
    if not isinstance(scene_payload, Mapping):
        raise ValueError("causal schedule scene record must be an object")
    raw_events = scene_payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("causal schedule scene must declare events")
    if any(not isinstance(item, Mapping) for item in raw_events):
        raise ValueError("causal schedule events must be objects")
    events = tuple(raw_events)
    event_ids = tuple(item.get("event_id") for item in events)
    if any(not isinstance(value, str) or not value.strip() for value in event_ids):
        raise ValueError("causal schedule event IDs must be non-empty strings")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("causal schedule event IDs must be unique")
    return events


def load_causal_prefix_contract(
    *,
    scene: str,
    schedule_path: Path,
    rgbd_root: Path,
    configured_cutoff: int,
) -> CausalPrefixContract:
    """Validate the complete RGB-D prefix strictly before the first intervention."""

    if not isinstance(scene, str) or not scene.strip() or scene != scene.strip():
        raise ValueError("scene must be a normalized non-empty string")
    cutoff = _integer(configured_cutoff, label="configured_cutoff", minimum=0)
    schedule_path = Path(schedule_path)
    rgbd_root = Path(rgbd_root)
    schedule_bytes = _regular_file_bytes(
        schedule_path, label="causal schedule"
    )
    schedule = _json_object(schedule_bytes, label="causal schedule")
    if schedule.get("dataset") != "TESSE-CD":
        raise ValueError("causal schedule dataset must be TESSE-CD")
    events = _scene_events(schedule, scene)
    interventions = tuple(
        _integer(
            item.get("intervention_frame_index"),
            label="intervention_frame_index",
            minimum=1,
        )
        for item in events
    )
    first_intervention = min(interventions)
    if cutoff != first_intervention - 1:
        raise ValueError(
            "configured cutoff must be strictly before first intervention "
            "and include the complete prefix"
        )

    scene_root = rgbd_root / scene
    export_manifest = scene_root / "export_manifest.json"
    export_bytes = _regular_file_bytes(
        export_manifest, label="RGB-D export manifest"
    )
    export_payload = _json_object(export_bytes, label="RGB-D export manifest")
    if (
        export_payload.get("dataset") != "TESSE-CD"
        or export_payload.get("scene") != scene
    ):
        raise ValueError("RGB-D export manifest does not match dataset and scene")

    frame_ids = tuple(range(first_intervention))
    results = scene_root / "results"
    for frame_index in frame_ids:
        for prefix, suffix in (("frame", ".jpg"), ("depth", ".png")):
            path = results / f"{prefix}{frame_index:06d}{suffix}"
            try:
                _regular_file_bytes(path, label="causal RGB-D frame")
            except ValueError as error:
                raise ValueError(f"causal RGB-D frame is missing or invalid: {path}") from error

    return CausalPrefixContract(
        scene=scene,
        first_intervention_frame=first_intervention,
        frame_ids=frame_ids,
        maximum_source_frame=cutoff,
        schedule_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
        rgbd_export_manifest_sha256=hashlib.sha256(export_bytes).hexdigest(),
    )


__all__ = ["CausalPrefixContract", "load_causal_prefix_contract"]
