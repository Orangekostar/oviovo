#!/usr/bin/env python3
"""Attribute frozen two-visit evaluator failures to exact source decisions."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.exporters.oviovo import read_map_snapshot
from src.evaluation.json_contracts import loads_strict
from src.evaluation.two_visit_attribution import (
    EvaluatorFailure,
    attribute_two_visit_failures,
    write_two_visit_attribution,
)
from src.oviv2.two_visit_contracts import CurrentCompositionDecision, PairRelation
from src.oviv2.two_visit_current_map import (
    CompositionPointGroup,
    TwoVisitCurrentMap,
)

_MANIFEST_KEYS = {
    "schema_version",
    "artifact_id",
    "status",
    "method",
    "current_map_sha256",
    "source_visit_map_sha256",
    "source_manifest_sha256",
    "visibility_source_sha256",
    "relation_ids",
    "geometry_sources",
    "artifacts",
}
_GROUP_KEYS = {
    "decision",
    "source_point_indices",
    "source_snapshot_sha256",
    "output_entity_id",
    "output_point_start",
    "output_point_count",
}
_DECISION_KEYS = {
    "source_entity_id",
    "source_visit",
    "decision",
    "visibility_status",
    "visibility_score",
    "relation_id",
    "geometry_source",
    "identity_source",
    "state_source",
    "semantic_source",
}
_RELATION_KEYS = {
    "temporal_query_id",
    "t0_entity_ids",
    "t1_entity_ids",
    "state",
    "query_confidence",
    "evidence",
    "identity_source",
}
_FAILURE_KEYS = {
    "failure_id",
    "failure_type",
    "predicted_entity_id",
    "predicted_point_index",
    "target_entity_id",
    "evaluator_visibility",
    "visibility_depth_correct",
    "observed_relation_id",
    "expected_relation_id",
    "relation_correct",
}


def _direct_bytes(path: str | Path, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    try:
        metadata = current.lstat()
        for component in absolute.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} cannot contain symlinks")
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    before = absolute.stat(follow_symlinks=False)
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(content) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return content


def _object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = loads_strict(content.decode("utf-8"), label=label)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _bound_artifact(
    root: Path, record: object, *, expected_path: str, label: str
) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} binding schema is invalid")
    relative = record.get("path")
    if relative != expected_path:
        raise ValueError(f"{label} path binding is invalid")
    path = root / expected_path
    content = _direct_bytes(path, label)
    if (
        record.get("sha256") != hashlib.sha256(content).hexdigest()
        or record.get("byte_count") != len(content)
    ):
        raise ValueError(f"{label} binding mismatch")
    return path, content


def _load_current_map(root: Path) -> TwoVisitCurrentMap:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("current-map root must be a direct directory")
    manifest_content = _direct_bytes(root / "manifest.json", "current-map manifest")
    manifest = _object(manifest_content, "current-map manifest")
    if (
        set(manifest) != _MANIFEST_KEYS
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "OVI_TWO_VISIT_CURRENT_MAP_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("geometry_sources") != ["ovi_t0", "ovi_t1"]
    ):
        raise ValueError("current-map manifest identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "snapshot",
        "entities",
        "provenance",
    }:
        raise ValueError("current-map artifact bindings are invalid")
    timestamp = None
    snapshot_record = artifacts["snapshot"]
    if isinstance(snapshot_record, Mapping):
        raw_path = snapshot_record.get("path")
        if isinstance(raw_path, str) and raw_path.startswith("snapshots/"):
            timestamp = raw_path.removeprefix("snapshots/").removesuffix("_current.npz")
    if timestamp is None:
        raise ValueError("current-map snapshot path is invalid")
    snapshot_relative = f"snapshots/{timestamp}_current.npz"
    entities_relative = f"entities/{timestamp}_current.jsonl"
    snapshot_path, snapshot_content = _bound_artifact(
        root,
        snapshot_record,
        expected_path=snapshot_relative,
        label="current-map snapshot",
    )
    entities_path, entities_content = _bound_artifact(
        root,
        artifacts["entities"],
        expected_path=entities_relative,
        label="current-map entities",
    )
    _, provenance_content = _bound_artifact(
        root,
        artifacts["provenance"],
        expected_path="provenance.jsonl",
        label="current-map provenance",
    )
    snapshot = read_map_snapshot(snapshot_path, entities_path)
    if snapshot.method != manifest.get("method"):
        raise ValueError("current-map method identity mismatch")
    groups: list[CompositionPointGroup] = []
    for line_number, raw_line in enumerate(provenance_content.splitlines(), start=1):
        if not raw_line:
            raise ValueError(f"current-map provenance line {line_number} is empty")
        payload = _object(raw_line, f"current-map provenance line {line_number}")
        if set(payload) != _GROUP_KEYS:
            raise ValueError("current-map provenance group schema is invalid")
        decision = payload.get("decision")
        if not isinstance(decision, Mapping) or set(decision) != _DECISION_KEYS:
            raise ValueError("current-map composition decision schema is invalid")
        groups.append(
            CompositionPointGroup(
                decision=CurrentCompositionDecision(**decision),
                source_point_indices=np.asarray(payload["source_point_indices"]),
                source_snapshot_sha256=payload["source_snapshot_sha256"],
                output_entity_id=payload["output_entity_id"],
                output_point_start=payload["output_point_start"],
                output_point_count=payload["output_point_count"],
            )
        )
    source_hashes = manifest.get("source_visit_map_sha256")
    relation_ids = manifest.get("relation_ids")
    if not isinstance(source_hashes, list) or not isinstance(relation_ids, list):
        raise TypeError("current-map source identities are invalid")
    current = TwoVisitCurrentMap(
        snapshot=snapshot,
        provenance=tuple(groups),
        source_visit_map_sha256=tuple(source_hashes),
        source_manifest_sha256=manifest["source_manifest_sha256"],
        visibility_source_sha256=manifest["visibility_source_sha256"],
        relation_ids=tuple(relation_ids),
    )
    if current.content_sha256() != manifest.get("current_map_sha256"):
        raise ValueError("current-map content SHA-256 mismatch")
    if (
        _direct_bytes(snapshot_path, "current-map snapshot") != snapshot_content
        or _direct_bytes(entities_path, "current-map entities") != entities_content
        or _direct_bytes(root / "manifest.json", "current-map manifest")
        != manifest_content
    ):
        raise ValueError("current-map inputs changed during loading")
    return current


def _load_relations(content: bytes) -> tuple[PairRelation, ...]:
    payload = _object(content, "pair relations")
    if set(payload) != {"schema_version", "status", "relations"} or (
        payload.get("schema_version"), payload.get("status")
    ) != (1, "PASS"):
        raise ValueError("pair relation artifact identity is invalid")
    values = payload.get("relations")
    if not isinstance(values, list):
        raise TypeError("pair relations must be a list")
    result: list[PairRelation] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != _RELATION_KEYS:
            raise ValueError("pair relation schema is invalid")
        result.append(PairRelation(**value))
    return tuple(result)


def _load_failures(content: bytes) -> tuple[EvaluatorFailure, ...]:
    payload = _object(content, "evaluator failures")
    if set(payload) != {"schema_version", "status", "failures"} or (
        payload.get("schema_version"), payload.get("status")
    ) != (1, "PASS"):
        raise ValueError("evaluator failure artifact identity is invalid")
    values = payload.get("failures")
    if not isinstance(values, list):
        raise TypeError("evaluator failures must be a list")
    result: list[EvaluatorFailure] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != _FAILURE_KEYS:
            raise ValueError("evaluator failure schema is invalid")
        result.append(EvaluatorFailure(**value))
    return tuple(result)


def run_two_visit_attribution(
    *,
    current_root: str | Path,
    failures_path: str | Path,
    relations_path: str | Path,
    output_root: str | Path,
) -> Path:
    """Load exact inputs, attribute every failure, and publish sidecars."""

    current_path = Path(os.path.abspath(os.fspath(current_root)))
    current = _load_current_map(current_path)
    failure_content = _direct_bytes(failures_path, "evaluator failures")
    relation_content = _direct_bytes(relations_path, "pair relations")
    result = attribute_two_visit_failures(
        current,
        _load_failures(failure_content),
        _load_relations(relation_content),
        evaluator_source_sha256=hashlib.sha256(failure_content).hexdigest(),
        relation_source_sha256=hashlib.sha256(relation_content).hexdigest(),
    )
    if (
        _direct_bytes(failures_path, "evaluator failures") != failure_content
        or _direct_bytes(relations_path, "pair relations") != relation_content
    ):
        raise ValueError("attribution inputs changed during loading")
    return write_two_visit_attribution(result, output_root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-map-root", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        manifest = run_two_visit_attribution(
            current_root=arguments.current_map_root,
            failures_path=arguments.failures,
            relations_path=arguments.relations,
            output_root=arguments.output,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
