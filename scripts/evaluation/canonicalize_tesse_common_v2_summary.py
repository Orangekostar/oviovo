#!/usr/bin/env python3
"""Canonicalize one OVIV2 common-v2 summary without serializing run paths."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.finalize_tesse_t2 import (
    _absolute_lexical,
    _atomic_json_no_replace,
    _open_directory_no_symlinks,
    _stable_regular_file,
)
from src.evaluation.json_contracts import loads_strict


CANONICAL_MANIFEST_ID = "tesse_cd_common_v2_canonical_scene_summary_v1"
EXTERNAL_SOURCE_ROLES = frozenset(
    {"target_manifest", "target_arrays", "aliases", "label_space", "evaluator"}
)
_CHECKPOINT_ROLE = re.compile(r"^(snapshot|entities)\.(\d{6})$")


def _direct_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute_lexical(path)
    descriptor, _ = _open_directory_no_symlinks(absolute, label=label)
    os.close(descriptor)
    return absolute


def _expected_internal_path(role: str, *, artifact_root: Path) -> Path:
    if role == "temporal_index":
        relative = Path("temporal/temporal_manifest.json")
    elif role == "schedule":
        relative = Path("temporal/sidecars/schedule.json")
    else:
        match = _CHECKPOINT_ROLE.fullmatch(role)
        if match is None:
            raise ValueError(f"unsupported summary source role: {role}")
        kind, raw_frame = match.groups()
        suffix = "snapshot.npz" if kind == "snapshot" else "entities.jsonl"
        relative = Path(
            f"temporal/checkpoints/{int(raw_frame):08d}/{suffix}"
        )
    return artifact_root / relative


def _canonical_source(
    role: str,
    declaration: object,
    *,
    expected_path: Path,
) -> tuple[dict[str, object], str]:
    if not isinstance(declaration, Mapping) or set(declaration) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{role} source record fields are invalid")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{role} source path must be absolute")
    observed_path = _absolute_lexical(Path(raw_path))
    if raw_path != os.fspath(observed_path):
        raise ValueError(f"{role} source path must be canonical")
    expected = _absolute_lexical(expected_path)
    if observed_path != expected:
        raise ValueError(f"{role} path mismatch")
    digest, byte_count, _ = _stable_regular_file(
        observed_path, label=f"{role} source", capture=False
    )
    declared_digest = declaration.get("sha256")
    declared_bytes = declaration.get("byte_count")
    if (
        not isinstance(declared_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", declared_digest)
        or type(declared_bytes) is not int
        or declared_bytes < 0
        or declared_digest != digest
        or declared_bytes != byte_count
    ):
        raise ValueError(f"{role} content mismatch")
    return (
        {
            "role": role,
            "sha256": digest,
            "byte_count": byte_count,
        },
        raw_path,
    )


def canonicalize_summary(
    summary: Path,
    *,
    artifact_root: Path,
    external_sources: Mapping[str, Path],
) -> dict[str, Any]:
    _, canonical, _ = capture_and_canonicalize_summary(
        summary,
        artifact_root=artifact_root,
        external_sources=external_sources,
    )
    return canonical


def capture_and_canonicalize_summary(
    summary: Path,
    *,
    artifact_root: Path,
    external_sources: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, object]]:
    root = _direct_directory(artifact_root, label="artifact root")
    expected_summary = root / "evaluation/summary.json"
    if _absolute_lexical(summary) != expected_summary:
        raise ValueError("summary path must be artifact_root/evaluation/summary.json")
    raw_digest, raw_byte_count, raw_bytes = _stable_regular_file(
        expected_summary, label="raw common-v2 summary", capture=True
    )
    assert raw_bytes is not None
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("raw common-v2 summary is not UTF-8") from error
    payload = loads_strict(content, label="raw common-v2 summary")
    if not isinstance(payload, Mapping):
        raise ValueError("raw common-v2 summary must be a mapping")
    if not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "tesse_cd_common_v2_scene_summary"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("protocol") == "tesse_cd_common_v2"
        and payload.get("status") == "PASS"
        and payload.get("method") == "OVIV2"
        and payload.get("mode") == "causal_checkpoints"
        and payload.get("scene") in {"apartment", "office"}
    ):
        raise ValueError("raw common-v2 summary identity mismatch")
    if {"canonicalization", "source_manifest_id"} & set(payload):
        raise ValueError("raw summary contains reserved canonicalization fields")
    if set(external_sources) != EXTERNAL_SOURCE_ROLES:
        raise ValueError("external source bindings must cover exact OVIV2 roles")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("raw common-v2 summary sources must be a mapping")
    checkpoint_roles = {
        str(role) for role in sources if _CHECKPOINT_ROLE.fullmatch(str(role))
    }
    snapshot_frames = {
        role.removeprefix("snapshot.")
        for role in checkpoint_roles
        if role.startswith("snapshot.")
    }
    entity_frames = {
        role.removeprefix("entities.")
        for role in checkpoint_roles
        if role.startswith("entities.")
    }
    expected_roles = (
        EXTERNAL_SOURCE_ROLES
        | {"temporal_index", "schedule"}
        | checkpoint_roles
    )
    if (
        not checkpoint_roles
        or snapshot_frames != entity_frames
        or set(sources) != expected_roles
    ):
        raise ValueError("raw common-v2 source roles are incomplete or inconsistent")

    canonical_sources: dict[str, dict[str, object]] = {}
    physical_paths: list[str] = []
    for raw_role, declaration in sources.items():
        role = str(raw_role)
        expected_path = (
            external_sources[role]
            if role in EXTERNAL_SOURCE_ROLES
            else _expected_internal_path(role, artifact_root=root)
        )
        canonical, physical_path = _canonical_source(
            role, declaration, expected_path=expected_path
        )
        canonical_sources[role] = canonical
        physical_paths.append(physical_path)

    canonical_payload = copy.deepcopy(dict(payload))
    canonical_payload["manifest_id"] = CANONICAL_MANIFEST_ID
    canonical_payload["source_manifest_id"] = "tesse_cd_common_v2_scene_summary"
    canonical_payload["canonicalization"] = {
        "schema_version": 1,
        "source_identity": "stable_role_sha256_byte_count",
        "physical_paths_serialized": False,
    }
    canonical_payload["sources"] = canonical_sources
    encoded = canonical_summary_bytes(canonical_payload)
    for physical_path in physical_paths:
        if physical_path.encode("utf-8") in encoded:
            raise ValueError("physical source path leaked into canonical summary")
    if os.fspath(root).encode("utf-8") in encoded:
        raise ValueError("artifact root leaked into canonical summary")
    return (
        dict(payload),
        canonical_payload,
        {"sha256": raw_digest, "byte_count": raw_byte_count},
    )


def canonical_summary_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _external_source(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or role not in EXTERNAL_SOURCE_ROLES or not raw_path:
        raise argparse.ArgumentTypeError(
            "external source must be one of the required ROLE=/absolute/path bindings"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("external source path must be absolute")
    return role, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--external-source", type=_external_source, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    external_sources: dict[str, Path] = {}
    for role, path in args.external_source:
        if role in external_sources:
            parser.error(f"duplicate external source role: {role}")
        external_sources[role] = path
    payload = canonicalize_summary(
        args.summary,
        artifact_root=args.artifact_root,
        external_sources=external_sources,
    )
    _atomic_json_no_replace(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
