"""Audit CROVE map provenance and geometry authority without changing state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Ruff's TRY004 rule does not distinguish malformed external artifacts from
# programmer type errors. Artifact schema violations intentionally use ValueError.
# ruff: noqa: TRY004

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PLY_INTEGER_TYPES = {
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
}
_PLY_FLOAT_TYPES = {"float", "double", "float32", "float64"}


def parse_instance_color_log(path: Path) -> dict[int, tuple[int, int, int]]:
    from src.evaluation.baselines.ovimap import parse_instance_color_log as parse

    return parse(path)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} rejects symlink component: {current}")


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    _reject_symlink_components(path, label=label)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.read_bytes()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stream_file_observation(path: Path, *, label: str) -> tuple[int, str]:
    _reject_symlink_components(path, label=label)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def verify_file_record(
    record: Mapping[str, object],
    *,
    base_dir: Path,
    label: str,
) -> dict[str, object]:
    """Verify one exact path/hash/byte-count record and return its observation."""

    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} must contain exactly path, sha256, and byte_count")
    raw_path = record["path"]
    expected_sha256 = record["sha256"]
    expected_byte_count = record["byte_count"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path must be a non-empty string")
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise ValueError(f"{label} sha256 must be a lowercase SHA-256 digest")
    if (
        isinstance(expected_byte_count, bool)
        or not isinstance(expected_byte_count, int)
        or expected_byte_count < 0
    ):
        raise ValueError(f"{label} byte_count must be a non-negative integer")

    base = _absolute_lexical(Path(base_dir))
    declared = Path(raw_path)
    if declared.is_absolute():
        path = _absolute_lexical(declared)
    else:
        path = _absolute_lexical(base / declared)
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ValueError(f"{label} path escapes manifest directory") from error
    byte_count, observed_sha256 = _stream_file_observation(path, label=label)
    if byte_count != expected_byte_count:
        raise ValueError(f"{label} byte-count mismatch")
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "byte_count": byte_count,
    }


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_pass_manifest(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Load one regular JSON manifest and require top-level status PASS."""

    manifest_path = _absolute_lexical(Path(path))
    content = _regular_file_bytes(manifest_path, label=label)
    try:
        payload = json.loads(content, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must contain valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if payload.get("status") != "PASS":
        raise ValueError(f"{label} must have status PASS")
    source = {
        "path": str(manifest_path),
        "sha256": _sha256_bytes(content),
        "byte_count": len(content),
    }
    return payload, source


def _ply_header(handle) -> tuple[int, int, tuple[tuple[str, str], ...]]:
    first = handle.readline()
    if first != b"ply\n" and first != b"ply\r\n":
        raise ValueError("mesh must start with a PLY header")
    format_name: str | None = None
    vertex_count: int | None = None
    face_count = 0
    current_element: str | None = None
    vertex_properties: list[tuple[str, str]] = []
    header_bytes = len(first)
    while True:
        line_bytes = handle.readline()
        header_bytes += len(line_bytes)
        if not line_bytes or header_bytes > 65536:
            raise ValueError("PLY header is missing end_header")
        try:
            line = line_bytes.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("PLY header must be ASCII") from error
        if not line or line.startswith(("comment ", "obj_info ")):
            continue
        fields = line.split()
        if fields[0] == "format":
            if len(fields) != 3 or fields[2] != "1.0":
                raise ValueError("PLY format declaration is invalid")
            format_name = f"{fields[1]} {fields[2]}"
        elif fields[0] == "element":
            if len(fields) != 3:
                raise ValueError("PLY element declaration is invalid")
            try:
                count = int(fields[2])
            except ValueError as error:
                raise ValueError("PLY element count must be an integer") from error
            if count < 0:
                raise ValueError("PLY element count must be non-negative")
            current_element = fields[1]
            if current_element == "vertex":
                if vertex_count is not None:
                    raise ValueError("PLY declares the vertex element more than once")
                vertex_count = count
            elif current_element == "face":
                face_count = count
        elif fields[0] == "property":
            if current_element == "vertex":
                if len(fields) != 3:
                    raise ValueError("PLY audit requires scalar vertex properties")
                value_type, name = fields[1], fields[2]
                if value_type not in _PLY_INTEGER_TYPES | _PLY_FLOAT_TYPES:
                    raise ValueError(
                        f"unsupported PLY vertex property type: {value_type}"
                    )
                vertex_properties.append((name, value_type))
        elif fields[0] == "end_header":
            if len(fields) != 1:
                raise ValueError("PLY end_header declaration is invalid")
            break
        elif fields[0] not in {"ply"}:
            raise ValueError(f"unsupported PLY header declaration: {fields[0]}")
    if format_name != "ascii 1.0":
        raise ValueError("map-quality audit currently requires an ASCII PLY")
    if vertex_count is None:
        raise ValueError("PLY header is missing the vertex element")
    names = [name for name, _ in vertex_properties]
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = required - set(names)
    if missing:
        raise ValueError(f"PLY vertex element lacks properties: {sorted(missing)}")
    if len(names) != len(set(names)):
        raise ValueError("PLY vertex property names must be unique")
    types = dict(vertex_properties)
    if any(
        types[channel] not in {"uchar", "uint8"} for channel in ("red", "green", "blue")
    ):
        raise ValueError("PLY RGB values must use uchar vertex properties")
    return vertex_count, face_count, tuple(vertex_properties)


def _parse_ply_scalar(token: str, value_type: str, *, row_index: int) -> int | float:
    try:
        if value_type in _PLY_INTEGER_TYPES:
            return int(token)
        return float(token)
    except ValueError as error:
        raise ValueError(
            f"PLY vertex {row_index} contains an invalid scalar"
        ) from error


def audit_ascii_instance_ply(
    path: Path,
    colors_by_instance: Mapping[int, tuple[int, int, int]],
) -> dict[str, object]:
    """Stream an OVI-MAP ASCII PLY and prove its registered instance palette."""

    mesh_path = _absolute_lexical(Path(path))
    _reject_symlink_components(mesh_path, label="instance mesh")
    if not mesh_path.is_file():
        raise ValueError(f"instance mesh must be a regular file: {mesh_path}")
    palette: dict[int, tuple[int, int, int]] = {}
    instances_by_color: dict[tuple[int, int, int], int] = {}
    for raw_instance_id, raw_color in colors_by_instance.items():
        if (
            isinstance(raw_instance_id, bool)
            or not isinstance(raw_instance_id, int)
            or raw_instance_id <= 0
        ):
            raise ValueError("instance palette IDs must be positive integers")
        if not isinstance(raw_color, tuple) or len(raw_color) != 3:
            raise ValueError("instance palette colors must be RGB triples")
        color = tuple(int(value) for value in raw_color)
        if any(value < 0 or value > 255 for value in color):
            raise ValueError("instance palette RGB values must lie in [0, 255]")
        if color in instances_by_color and instances_by_color[color] != raw_instance_id:
            raise ValueError("instance palette color is reused by multiple IDs")
        palette[raw_instance_id] = color
        instances_by_color[color] = raw_instance_id
    if not palette:
        raise ValueError("instance palette must not be empty")

    color_counts: Counter[tuple[int, int, int]] = Counter()
    with mesh_path.open("rb") as handle:
        vertex_count, face_count, properties = _ply_header(handle)
        property_names = [name for name, _ in properties]
        channel_indices = tuple(
            property_names.index(channel) for channel in ("red", "green", "blue")
        )
        for row_index in range(vertex_count):
            line = handle.readline()
            if not line:
                raise ValueError("PLY contains truncated vertex data")
            try:
                tokens = line.decode("ascii").split()
            except UnicodeDecodeError as error:
                raise ValueError(f"PLY vertex {row_index} must be ASCII") from error
            if len(tokens) != len(properties):
                raise ValueError(f"PLY vertex {row_index} property count mismatch")
            values = [
                _parse_ply_scalar(token, value_type, row_index=row_index)
                for token, (_, value_type) in zip(tokens, properties, strict=True)
            ]
            color = tuple(int(values[index]) for index in channel_indices)
            if any(value < 0 or value > 255 for value in color):
                raise ValueError("PLY RGB values must lie in [0, 255]")
            color_counts[color] += 1

    instances = [
        {
            "instance_id": instance_id,
            "rgb": list(color),
            "vertex_count": color_counts[color],
        }
        for instance_id, color in sorted(palette.items())
        if color_counts[color] > 0
    ]
    missing_instance_ids = [
        instance_id
        for instance_id, color in sorted(palette.items())
        if color_counts[color] == 0
    ]
    if missing_instance_ids:
        raise ValueError(
            f"registered instance colors are absent from PLY: {missing_instance_ids}"
        )
    instance_vertex_count = sum(item["vertex_count"] for item in instances)
    return {
        "format": "ascii 1.0",
        "vertex_count": vertex_count,
        "face_count": face_count,
        "vertex_properties": [name for name, _ in properties],
        "unique_color_count": len(color_counts),
        "registered_instance_count": len(palette),
        "present_instance_count": len(instances),
        "instance_vertex_count": instance_vertex_count,
        "background_or_unregistered_vertex_count": vertex_count - instance_vertex_count,
        "missing_instance_ids": missing_instance_ids,
        "visualization_mode": "instance_palette",
        "instances": instances,
    }


def _observed_file_record(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    absolute = _absolute_lexical(path)
    content = _regular_file_bytes(absolute, label=label)
    return (
        {
            "path": str(absolute),
            "sha256": _sha256_bytes(content),
            "byte_count": len(content),
        },
        content,
    )


def _mapping_command(native_payload: Mapping[str, object]) -> tuple[Path, Path]:
    commands = native_payload.get("commands")
    if not isinstance(commands, Mapping):
        raise ValueError("native manifest lacks commands")
    mapping = commands.get("mapping")
    if not isinstance(mapping, Mapping):
        raise ValueError("native manifest lacks mapping command")
    cwd = mapping.get("cwd")
    argv = mapping.get("argv")
    if not isinstance(cwd, str) or not cwd or not Path(cwd).is_absolute():
        raise ValueError("native mapping cwd must be an absolute path")
    if (
        not isinstance(argv, list)
        or len(argv) < 2
        or not all(isinstance(value, str) for value in argv)
    ):
        raise ValueError("native mapping argv must contain string arguments")
    if mapping.get("exit_code") != 0:
        raise ValueError("native mapping command did not exit successfully")
    root = _absolute_lexical(Path(cwd))
    _reject_symlink_components(root, label="OVI-MAP root")
    if not root.is_dir():
        raise ValueError(f"OVI-MAP root must be a directory: {root}")
    entrypoint = _absolute_lexical(Path(argv[1]))
    try:
        entrypoint.relative_to(root)
    except ValueError as error:
        raise ValueError("native mapping entrypoint is outside mapping cwd") from error
    expected_entrypoint = root / "scripts/panoptic_mapping_.py"
    if entrypoint != expected_entrypoint:
        raise ValueError(
            "native mapping entrypoint is not scripts/panoptic_mapping_.py"
        )
    return root, entrypoint


def _normalized_source(content: bytes, *, label: str) -> str:
    try:
        return " ".join(content.decode("utf-8").split())
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 source text") from error


def bind_producer_chain(native_payload: Mapping[str, object]) -> dict[str, object]:
    """Bind the exact OVI-MAP source chain that produces an instance PLY."""

    if (
        not isinstance(native_payload, Mapping)
        or native_payload.get("status") != "PASS"
    ):
        raise ValueError("native manifest must have status PASS")
    preflight = native_payload.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("native manifest lacks preflight")
    ovimap_commit = preflight.get("ovimap_commit")
    if (
        not isinstance(ovimap_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", ovimap_commit) is None
    ):
        raise ValueError("native manifest lacks a valid OVI-MAP commit")
    root, entrypoint = _mapping_command(native_payload)
    paths = {
        "python_entrypoint": entrypoint,
        "instance_mesh_writer": root
        / "mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/global_segment_map_py.cpp",
        "instance_colorizer": root
        / "mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/meshing/label_tsdf_mesh_integrator.cc",
    }
    sources: dict[str, dict[str, object]] = {}
    source_text: dict[str, str] = {}
    for role, path in paths.items():
        record, content = _observed_file_record(path, label=role.replace("_", " "))
        sources[role] = record
        source_text[role] = _normalized_source(content, label=role.replace("_", " "))

    preflight_sources = _required_mapping(
        preflight, "sources", label="native preflight"
    )
    mapper_binding = _verified_named_record(
        preflight_sources,
        "mapper",
        base_dir=root,
        label="native mapper",
    )
    extension_binding = _verified_named_record(
        preflight_sources,
        "consistent_gsm_extension",
        base_dir=root,
        label="consistent_gsm extension",
    )
    source_hashes_binding = _verified_named_record(
        preflight_sources,
        "source_hashes",
        base_dir=root,
        label="native source hashes",
    )
    _require_same_source(
        mapper_binding,
        sources["python_entrypoint"],
        label="native mapper binding",
    )
    source_hashes = _read_json_object(
        Path(str(source_hashes_binding["path"])),
        label="native source hashes",
    )
    ovimap_source = source_hashes.get("OVI-MAP")
    if (
        not isinstance(ovimap_source, Mapping)
        or ovimap_source.get("commit") != ovimap_commit
    ):
        raise ValueError("native source hashes OVI-MAP commit disagrees with preflight")

    python_text = source_text["python_entrypoint"]
    if (
        "gsm_node.generateMesh(" not in python_text
        or "False, False, True" not in python_text
    ):
        raise ValueError("Python entrypoint lacks the instance-only generateMesh call")
    writer_text = source_text["instance_mesh_writer"]
    if (
        "mesh_instance_integrator_->generateMesh" not in writer_text
        or "outputMeshLayerAsPly" not in writer_text
        or '"/instance_mesh_"' not in writer_text
    ):
        raise ValueError("C++ instance mesh writer lacks the required output chain")
    colorizer_text = source_text["instance_colorizer"]
    if (
        "case kInstance:" not in colorizer_text
        or "instance_color_map_.getColor" not in colorizer_text
    ):
        raise ValueError("C++ instance colorizer lacks kInstance palette lookup")
    return {
        "ovimap_root": str(root),
        "ovimap_commit": ovimap_commit,
        "runtime_bindings": {
            "mapper": mapper_binding,
            "consistent_gsm_extension": extension_binding,
            "source_hashes": source_hashes_binding,
        },
        "sources": sources,
        "chain": [
            "panoptic_mapping.generateMesh(instance_only)",
            "GlobalSegmentMap_py.generateMesh",
            "MeshLabelIntegrator.kInstance",
            "instance_color_map",
            "outputMeshLayerAsPly",
        ],
    }


def _read_entity_records(path: Path, *, label: str) -> dict[str, dict[str, object]]:
    content = _regular_file_bytes(_absolute_lexical(path), label=label)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 JSONL") from error
    records: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label} contains a blank JSONL record")
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        entity_id = record.get("entity_id")
        point_count = record.get("point_count")
        metadata = record.get("metadata")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(f"{label} line {line_number} has an invalid entity ID")
        if entity_id in records:
            raise ValueError(f"duplicate {label} entity ID: {entity_id}")
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count < 0
        ):
            raise ValueError(f"{label} line {line_number} has an invalid point count")
        if not isinstance(metadata, dict):
            raise ValueError(f"{label} line {line_number} has invalid metadata")
        records[entity_id] = record
    if not records:
        raise ValueError(f"{label} must contain at least one entity")
    return records


def _diagnostic_id_sets(
    diagnostics: Mapping[str, object],
) -> tuple[set[str], set[str], set[str], set[str], set[int]]:
    anchor_roles = (
        "unchanged_anchor_ids",
        "occluded_anchor_ids",
        "moved_anchor_ids",
        "removed_anchor_ids",
    )
    anchor_sets: list[set[str]] = []
    for role in anchor_roles:
        values = diagnostics.get(role)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"diagnostics {role} must be a list of entity IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"diagnostics {role} contains duplicate entity IDs")
        anchor_sets.append(set(values))
    for index, left in enumerate(anchor_sets):
        for right in anchor_sets[index + 1 :]:
            if left & right:
                raise ValueError("diagnostics anchor roles must be disjoint")
    new_values = diagnostics.get("new_temporal_ids")
    if not isinstance(new_values, list) or not all(
        not isinstance(value, bool) and isinstance(value, int) and value > 0
        for value in new_values
    ):
        raise ValueError("diagnostics new_temporal_ids must contain positive integers")
    if len(new_values) != len(set(new_values)):
        raise ValueError("diagnostics new_temporal_ids contains duplicates")
    return (*anchor_sets, set(new_values))


def _point_count(record: Mapping[str, object]) -> int:
    value = record["point_count"]
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _metadata(record: Mapping[str, object]) -> dict[str, object]:
    value = record["metadata"]
    assert isinstance(value, dict)
    return value


def summarize_geometry_authority(
    anchor_entities_path: Path,
    current_entities_path: Path,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Compare one composed checkpoint with its immutable anchor entities."""

    if not isinstance(diagnostics, Mapping):
        raise ValueError("checkpoint diagnostics must be a mapping")
    anchor = _read_entity_records(anchor_entities_path, label="anchor")
    current = _read_entity_records(current_entities_path, label="current")
    for entity_id, record in anchor.items():
        if _metadata(record).get("authority") != "ovimap_anchor":
            raise ValueError(
                f"anchor entity must use ovimap_anchor authority: {entity_id}"
            )
    unchanged, occluded, moved, removed, new_temporal = _diagnostic_id_sets(diagnostics)
    categorized_anchor_ids = unchanged | occluded | moved | removed
    if categorized_anchor_ids != set(anchor):
        raise ValueError(
            "diagnostics anchor roles do not exactly cover anchor entities"
        )

    expected_present_anchor_ids = unchanged | occluded | moved
    actual_present_anchor_ids = {
        entity_id for entity_id in current if entity_id.startswith("ovimap:")
    }
    if actual_present_anchor_ids != expected_present_anchor_ids:
        missing_moved = moved - actual_present_anchor_ids
        if missing_moved:
            raise ValueError(
                f"moved anchor replacement is missing: {sorted(missing_moved)}"
            )
        raise ValueError("current anchor entities disagree with checkpoint diagnostics")
    for entity_id in sorted(unchanged | occluded):
        metadata = _metadata(current[entity_id])
        expected_overlay = "occluded" if entity_id in occluded else "unchanged"
        if metadata.get("authority") != "ovimap_anchor":
            raise ValueError(
                "unchanged or occluded anchor must use ovimap_anchor authority"
            )
        if metadata.get("overlay_state") != expected_overlay:
            raise ValueError("anchor overlay metadata disagrees with diagnostics")
        if metadata.get("anchor_entity_id") != entity_id:
            raise ValueError("anchor metadata contains a mismatched anchor ID")

    moved_entities: list[dict[str, object]] = []
    for entity_id in sorted(moved):
        current_record = current[entity_id]
        metadata = _metadata(current_record)
        if metadata.get("authority") != "crove_temporal":
            raise ValueError("moved anchor must use crove_temporal authority")
        if metadata.get("overlay_state") != "moved":
            raise ValueError("moved anchor overlay metadata disagrees with diagnostics")
        if metadata.get("anchor_entity_id") != entity_id:
            raise ValueError("moved replacement contains a mismatched anchor ID")
        temporal_entity_id = metadata.get("temporal_entity_id")
        if (
            isinstance(temporal_entity_id, bool)
            or not isinstance(temporal_entity_id, int)
            or temporal_entity_id <= 0
        ):
            raise ValueError("moved replacement lacks a valid temporal entity ID")
        anchor_points = _point_count(anchor[entity_id])
        current_points = _point_count(current_record)
        moved_entities.append(
            {
                "anchor_entity_id": entity_id,
                "anchor_points": anchor_points,
                "current_points": current_points,
                "retained_point_ratio": current_points / anchor_points
                if anchor_points
                else None,
                "temporal_entity_id": temporal_entity_id,
            }
        )

    expected_new_ids = {f"temporal:{entity_id}" for entity_id in new_temporal}
    actual_new_ids = {
        entity_id for entity_id in current if entity_id.startswith("temporal:")
    }
    if actual_new_ids != expected_new_ids:
        raise ValueError("new temporal entities disagree with checkpoint diagnostics")
    new_point_counts: list[int] = []
    for entity_id in sorted(
        actual_new_ids, key=lambda value: int(value.split(":", 1)[1])
    ):
        record = current[entity_id]
        metadata = _metadata(record)
        numeric_id = int(entity_id.split(":", 1)[1])
        if (
            metadata.get("authority") != "crove_temporal"
            or metadata.get("overlay_state") != "new"
            or metadata.get("anchor_entity_id") is not None
            or metadata.get("temporal_entity_id") != numeric_id
        ):
            raise ValueError("new temporal metadata disagrees with diagnostics")
        new_point_counts.append(_point_count(record))
    if set(current) != actual_present_anchor_ids | actual_new_ids:
        raise ValueError("current entities contain an unsupported identity namespace")

    authority_counts: dict[str, list[int]] = {}
    overlay_counts: dict[str, list[int]] = {}
    for record in current.values():
        metadata = _metadata(record)
        authority = metadata.get("authority")
        overlay_state = metadata.get("overlay_state")
        if not isinstance(authority, str) or not isinstance(overlay_state, str):
            raise ValueError("current entity lacks authority or overlay state")
        authority_counts.setdefault(authority, []).append(_point_count(record))
        overlay_counts.setdefault(overlay_state, []).append(_point_count(record))
    moved_anchor_points = sum(item["anchor_points"] for item in moved_entities)
    moved_current_points = sum(item["current_points"] for item in moved_entities)
    assert isinstance(moved_anchor_points, int) and isinstance(
        moved_current_points, int
    )
    anchor_total = sum(_point_count(record) for record in anchor.values())
    current_total = sum(_point_count(record) for record in current.values())
    return {
        "anchor": {"entity_count": len(anchor), "total_points": anchor_total},
        "current": {
            "entity_count": len(current),
            "total_points": current_total,
            "by_authority": [
                {
                    "authority": authority,
                    "entity_count": len(counts),
                    "total_points": sum(counts),
                }
                for authority, counts in sorted(authority_counts.items())
            ],
            "by_overlay_state": [
                {
                    "overlay_state": state,
                    "entity_count": len(counts),
                    "total_points": sum(counts),
                }
                for state, counts in sorted(overlay_counts.items())
            ],
        },
        "moved": {
            "entity_count": len(moved_entities),
            "anchor_points": moved_anchor_points,
            "current_points": moved_current_points,
            "retained_point_ratio": (
                moved_current_points / moved_anchor_points
                if moved_anchor_points
                else None
            ),
            "entities": moved_entities,
        },
        "new_temporal": {
            "entity_count": len(new_point_counts),
            "total_points": sum(new_point_counts),
            "minimum_points": min(new_point_counts) if new_point_counts else None,
            "median_points": statistics.median(new_point_counts)
            if new_point_counts
            else None,
            "maximum_points": max(new_point_counts) if new_point_counts else None,
        },
        "removed": {
            "entity_count": len(removed),
            "anchor_points": sum(
                _point_count(anchor[entity_id]) for entity_id in removed
            ),
        },
        "conclusion": (
            "dense_anchor_to_sparse_temporal_switch_measured"
            if moved_anchor_points and moved_current_points < moved_anchor_points
            else "geometry_authority_transition_measured"
        ),
    }


def _required_mapping(
    parent: Mapping[str, object], key: str, *, label: str
) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a {key} mapping")
    return value


def _verified_named_record(
    parent: Mapping[str, object],
    key: str,
    *,
    base_dir: Path,
    label: str,
) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing")
    return verify_file_record(value, base_dir=base_dir, label=label)


def _require_same_source(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if dict(observed) != dict(expected):
        raise ValueError(f"{label} does not bind the same source")


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    content = _regular_file_bytes(path, label=label)
    try:
        payload = json.loads(content, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must contain valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _final_checkpoint(composed_payload: Mapping[str, object]) -> Mapping[str, object]:
    checkpoints = composed_payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("composed manifest must contain checkpoints")
    previous_frame = -1
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("composed checkpoints must be mappings")
        frame_index = checkpoint.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index <= previous_frame
        ):
            raise ValueError("composed checkpoint frames must be strictly increasing")
        previous_frame = frame_index
    return checkpoints[-1]


def audit_map_quality(
    *,
    native_manifest_path: Path,
    anchor_manifest_path: Path,
    composed_manifest_path: Path,
) -> dict[str, object]:
    """Verify the P0 provenance chain and return a deterministic audit payload."""

    native, native_source = load_pass_manifest(
        native_manifest_path, label="native manifest"
    )
    anchor, anchor_source = load_pass_manifest(
        anchor_manifest_path, label="anchor manifest"
    )
    composed, composed_source = load_pass_manifest(
        composed_manifest_path,
        label="composed manifest",
    )
    scenes = {native.get("scene"), anchor.get("scene"), composed.get("scene")}
    if len(scenes) != 1 or not all(
        isinstance(scene, str) and scene for scene in scenes
    ):
        raise ValueError("native, anchor, and composed manifests must bind one scene")
    scene = next(iter(scenes))
    assert isinstance(scene, str)

    native_dir = _absolute_lexical(Path(native_manifest_path)).parent
    anchor_dir = _absolute_lexical(Path(anchor_manifest_path)).parent
    composed_dir = _absolute_lexical(Path(composed_manifest_path)).parent
    native_artifacts = _required_mapping(native, "artifacts", label="native manifest")
    instance_mesh = _verified_named_record(
        native_artifacts,
        "instance_mesh",
        base_dir=native_dir,
        label="native instance mesh",
    )
    instance_color_log = _verified_named_record(
        native_artifacts,
        "instance_color_log",
        base_dir=native_dir,
        label="native instance color log",
    )

    anchor_sources = _required_mapping(anchor, "sources", label="anchor manifest")
    anchor_native = _verified_named_record(
        anchor_sources,
        "native_manifest",
        base_dir=anchor_dir,
        label="anchor native manifest source",
    )
    anchor_mesh = _verified_named_record(
        anchor_sources,
        "instance_mesh",
        base_dir=anchor_dir,
        label="anchor instance mesh source",
    )
    anchor_color_log = _verified_named_record(
        anchor_sources,
        "instance_color_log",
        base_dir=anchor_dir,
        label="anchor instance color log source",
    )
    _require_same_source(anchor_native, native_source, label="anchor native manifest")
    _require_same_source(anchor_mesh, instance_mesh, label="anchor instance mesh")
    _require_same_source(anchor_color_log, instance_color_log, label="anchor color log")

    anchor_outputs = _required_mapping(anchor, "outputs", label="anchor manifest")
    anchor_entities = _verified_named_record(
        anchor_outputs,
        "entities",
        base_dir=anchor_dir,
        label="anchor entities",
    )
    anchor_snapshot = _verified_named_record(
        anchor_outputs,
        "snapshot",
        base_dir=anchor_dir,
        label="anchor snapshot",
    )

    composed_inputs = _required_mapping(composed, "inputs", label="composed manifest")
    composed_anchor_manifest = _verified_named_record(
        composed_inputs,
        "anchor_manifest",
        base_dir=composed_dir,
        label="composed anchor manifest input",
    )
    composed_anchor_entities = _verified_named_record(
        composed_inputs,
        "anchor_entities",
        base_dir=composed_dir,
        label="composed anchor entities input",
    )
    composed_anchor_snapshot = _verified_named_record(
        composed_inputs,
        "anchor_snapshot",
        base_dir=composed_dir,
        label="composed anchor snapshot input",
    )
    _require_same_source(
        composed_anchor_manifest, anchor_source, label="composed anchor manifest"
    )
    _require_same_source(
        composed_anchor_entities, anchor_entities, label="composed anchor entities"
    )
    _require_same_source(
        composed_anchor_snapshot, anchor_snapshot, label="composed anchor snapshot"
    )

    final_checkpoint = _final_checkpoint(composed)
    current_entities = _verified_named_record(
        final_checkpoint,
        "entities",
        base_dir=composed_dir,
        label="final current entities",
    )
    current_snapshot = _verified_named_record(
        final_checkpoint,
        "snapshot",
        base_dir=composed_dir,
        label="final current snapshot",
    )
    diagnostics_source = _verified_named_record(
        final_checkpoint,
        "diagnostics_file",
        base_dir=composed_dir,
        label="final checkpoint diagnostics",
    )
    diagnostics = final_checkpoint.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("final checkpoint lacks inline diagnostics")
    diagnostics_file = _read_json_object(
        Path(str(diagnostics_source["path"])),
        label="final checkpoint diagnostics",
    )
    if dict(diagnostics) != diagnostics_file:
        raise ValueError("inline and file checkpoint diagnostics disagree")
    if diagnostics.get("frame_index") != final_checkpoint.get("frame_index"):
        raise ValueError("final checkpoint diagnostics frame disagrees")

    producer_chain = bind_producer_chain(native)
    colors_by_instance = parse_instance_color_log(Path(str(instance_color_log["path"])))
    ply_visualization = audit_ascii_instance_ply(
        Path(str(instance_mesh["path"])),
        colors_by_instance,
    )
    geometry_authority = summarize_geometry_authority(
        Path(str(anchor_entities["path"])),
        Path(str(current_entities["path"])),
        diagnostics,
    )
    if (
        ply_visualization["registered_instance_count"]
        != geometry_authority["anchor"]["entity_count"]
        or ply_visualization["instance_vertex_count"]
        != geometry_authority["anchor"]["total_points"]
    ):
        raise ValueError(
            "native PLY palette groups do not match anchor entity geometry"
        )

    return {
        "schema_version": 1,
        "status": "PASS",
        "scene": scene,
        "sources": {
            "native_manifest": native_source,
            "anchor_manifest": anchor_source,
            "composed_manifest": composed_source,
            "instance_mesh": instance_mesh,
            "instance_color_log": instance_color_log,
            "anchor_entities": anchor_entities,
            "anchor_snapshot": anchor_snapshot,
            "current_entities": current_entities,
            "current_snapshot": current_snapshot,
            "checkpoint_diagnostics": diagnostics_source,
        },
        "evidence": {
            "producer_chain": {
                "label": "CODE_EVIDENCE",
                "result": producer_chain,
            },
            "ply_visualization": {
                "label": "MEASURED_EVIDENCE",
                "result": ply_visualization,
            },
            "geometry_authority": {
                "label": "MEASURED_EVIDENCE",
                "result": geometry_authority,
            },
            "hypothesis": {
                "label": "HYPOTHESIS",
                "statement": (
                    "The dense-anchor to sparse-temporal geometry switch is the primary "
                    "cause of degraded moved and new object surfaces."
                ),
            },
        },
    }


def _markdown_report(result: Mapping[str, object]) -> str:
    evidence = result["evidence"]
    assert isinstance(evidence, Mapping)
    ply = evidence["ply_visualization"]["result"]
    geometry = evidence["geometry_authority"]["result"]
    assert isinstance(ply, Mapping) and isinstance(geometry, Mapping)
    moved = geometry["moved"]
    new_temporal = geometry["new_temporal"]
    anchor = geometry["anchor"]
    current = geometry["current"]
    assert all(
        isinstance(value, Mapping) for value in (moved, new_temporal, anchor, current)
    )
    lines = [
        "# CROVE Map-Quality Audit",
        "",
        f"- Status: `{result['status']}`",
        f"- Scene: `{result['scene']}`",
        f"- PLY visualization: `{ply['visualization_mode']}`",
        f"- Conclusion: `{geometry['conclusion']}`",
        "",
        "## Geometry Authority",
        "",
        "| Population | Entities | Points |",
        "| --- | ---: | ---: |",
        f"| Anchor | {anchor['entity_count']} | {anchor['total_points']} |",
        f"| Current | {current['entity_count']} | {current['total_points']} |",
        f"| Moved: anchor source | {moved['entity_count']} | {moved['anchor_points']} |",
        f"| Moved: current source | {moved['entity_count']} | {moved['current_points']} |",
        f"| New temporal | {new_temporal['entity_count']} | {new_temporal['total_points']} |",
        "",
        f"Moved retained-point ratio: `{moved['retained_point_ratio']}`.",
        "",
        "## Moved Entities",
        "",
        "| Anchor ID | Temporal ID | Anchor points | Current points | Retained ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    entities = moved["entities"]
    assert isinstance(entities, list)
    for entity in entities:
        assert isinstance(entity, Mapping)
        lines.append(
            "| {anchor_entity_id} | {temporal_entity_id} | {anchor_points} | "
            "{current_points} | {retained_point_ratio} |".format(**entity)
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            (
                "The producer chain is `CODE_EVIDENCE`; PLY and point-count summaries are "
                "`MEASURED_EVIDENCE`; the causal explanation remains a `HYPOTHESIS` until "
                "later phase comparisons test it."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _ensure_output_path(
    path: Path, *, input_roots: tuple[Path, ...], label: str
) -> Path:
    absolute = _absolute_lexical(path)
    for root in input_roots:
        try:
            absolute.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"{label} is inside an input artifact root: {root}")
    if absolute.exists() or absolute.is_symlink():
        raise ValueError(f"{label} already exists: {absolute}")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(absolute.parent, label=f"{label} parent")
    return absolute


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--composed-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_map_quality(
        native_manifest_path=args.native_manifest,
        anchor_manifest_path=args.anchor_manifest,
        composed_manifest_path=args.composed_manifest,
    )
    producer_root = Path(result["evidence"]["producer_chain"]["result"]["ovimap_root"])
    input_roots = tuple(
        _absolute_lexical(path).parent
        for path in (
            args.native_manifest,
            args.anchor_manifest,
            args.composed_manifest,
        )
    ) + (_absolute_lexical(producer_root),)
    output_json = _ensure_output_path(
        args.output_json,
        input_roots=input_roots,
        label="JSON output",
    )
    output_markdown = _ensure_output_path(
        args.output_markdown,
        input_roots=input_roots,
        label="Markdown output",
    )
    if output_json == output_markdown:
        raise ValueError("JSON and Markdown outputs must be different paths")
    json_bytes = (
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    markdown_bytes = _markdown_report(result).encode("utf-8")
    _atomic_write(output_json, json_bytes)
    try:
        _atomic_write(output_markdown, markdown_bytes)
    except Exception:
        output_json.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
