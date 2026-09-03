"""Hash-bound sidecar attribution for official CROVE dynamic metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from src.evaluation.exporters.oviovo import read_map_snapshot
from src.evaluation.json_contracts import loads_strict


_DYNAMIC_COLUMNS = (
    "Name",
    "Query",
    "NumObjDetected",
    "NumObjMissed",
    "NumObjHallucinated",
)
_OBJECT_COLUMNS = (
    "MapName",
    "QueryTime",
    "IsGT",
    "ID",
    "Label",
    "CentroidX",
    "CentroidY",
    "CentroidZ",
    "BBPosX",
    "BBPosY",
    "BBPosZ",
    "BBDimX",
    "BBDimY",
    "BBDimZ",
    "Present",
    "HasAppeared",
    "HasDisappeared",
)
_ASSOCIATION_COLUMNS = (
    "MapName",
    "QueryTime",
    "FromGT",
    "FromID",
    "ToID",
)
_NODE_ID = re.compile(r"O\([0-9]+\)\Z")


@dataclass(frozen=True, slots=True)
class OfficialDynRow:
    map_name: int
    query_time_ns: int
    detected: int
    missed: int
    hallucinated: int

    @property
    def total_mass(self) -> int:
        return self.detected + self.missed + self.hallucinated

    def to_json_record(self) -> dict[str, int]:
        return {
            "map_name": self.map_name,
            "query_time_ns": self.query_time_ns,
            "detected": self.detected,
            "missed": self.missed,
            "hallucinated": self.hallucinated,
            "total_mass": self.total_mass,
        }


def _regular_file(value: str | Path, *, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    try:
        current = Path(path.anchor)
        metadata = current.lstat()
        for component in path.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} must be a direct regular file")
    except FileNotFoundError as error:
        raise ValueError(f"{label} must be a direct regular file") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a direct regular file")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label=label)
    payload = loads_strict(source.read_text(encoding="utf-8"), label=label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return source, payload


def _bound_file(
    root: Path, record: object, *, label: str
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} binding schema is invalid")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} binding path is invalid")
    declared = Path(raw)
    if not declared.is_absolute() and ".." in declared.parts:
        raise ValueError(f"{label} binding escapes its root")
    path = _regular_file(
        declared if declared.is_absolute() else root / declared,
        label=label,
    )
    observed = _record(path)
    if (
        record.get("sha256") != observed["sha256"]
        or record.get("byte_count") != observed["byte_count"]
    ):
        raise ValueError(f"{label} binding mismatch")
    return path, observed


def _csv_rows(
    path: str | Path,
    *,
    label: str,
    required_columns: Sequence[str],
    key_columns: Sequence[str],
) -> tuple[Path, list[dict[str, str]]]:
    source = _regular_file(path, label=label)
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(required_columns).issubset(
            columns
        ):
            raise ValueError(f"{label} columns are invalid")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError(f"{label} rows are invalid")
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[name] for name in key_columns)
        previous = unique.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"{label} has conflicting duplicate row {key}")
        unique[key] = row
    return source, list(unique.values())


def _count(value: str, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def parse_official_dynamic_rows(path: str | Path) -> tuple[OfficialDynRow, ...]:
    _, rows = _csv_rows(
        path,
        label="official dynamic objects",
        required_columns=_DYNAMIC_COLUMNS,
        key_columns=("Name", "Query"),
    )
    parsed = tuple(
        OfficialDynRow(
            map_name=_count(row["Name"], label="dynamic map name"),
            query_time_ns=_count(row["Query"], label="dynamic query time"),
            detected=_count(row["NumObjDetected"], label="detected mass"),
            missed=_count(row["NumObjMissed"], label="missed mass"),
            hallucinated=_count(
                row["NumObjHallucinated"], label="hallucinated mass"
            ),
        )
        for row in rows
    )
    if parsed != tuple(sorted(parsed, key=lambda item: (item.map_name, item.query_time_ns))):
        raise ValueError("official dynamic rows are not canonical")
    return parsed


def classify_official_dyn_mass(
    official_mass: int,
    *,
    candidate_count: int,
    exact_event_count: int | None,
) -> str:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (official_mass, candidate_count)
    ):
        raise ValueError("official mass and candidate count must be nonnegative")
    if exact_event_count is not None:
        if (
            isinstance(exact_event_count, bool)
            or not isinstance(exact_event_count, int)
            or exact_event_count < 0
            or exact_event_count != official_mass
        ):
            raise ValueError("exact event mass must equal official mass")
        return "exact"
    return "ambiguous" if candidate_count else "unavailable"


def _trajectory_rows(path: Path) -> dict[tuple[int, object], dict[str, Any]]:
    result: dict[tuple[int, object], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = loads_strict(line, label=f"trajectory line {line_number}")
        if not isinstance(row, dict):
            raise ValueError("trajectory row must be an object")
        frame = row.get("frame_index")
        entity_id = row.get("entity_id")
        if type(frame) is not int or frame < 0 or not isinstance(entity_id, (int, str)):
            raise ValueError("trajectory row identity is invalid")
        key = (frame, entity_id)
        if key in result:
            raise ValueError("duplicate trajectory identity")
        result[key] = row
    if not result:
        raise ValueError("source trajectories are empty")
    return result


def _latest_trajectory(
    trajectories: Mapping[tuple[int, object], dict[str, Any]],
    *,
    frame_index: int,
    temporal_entity_id: object,
    source_entity_id: object,
) -> dict[str, Any] | None:
    candidates = [
        row
        for (frame, entity_id), row in trajectories.items()
        if frame <= frame_index
        and entity_id in {temporal_entity_id, source_entity_id}
    ]
    return (
        max(candidates, key=lambda item: int(item["frame_index"]))
        if candidates
        else None
    )


def _visualization_inputs(
    objects_path: str | Path,
    associations_path: str | Path,
) -> tuple[Path, Path, set[tuple[int, int, str]]]:
    objects_file, object_rows = _csv_rows(
        objects_path,
        label="official visualization objects",
        required_columns=_OBJECT_COLUMNS,
        key_columns=("MapName", "QueryTime", "IsGT", "ID"),
    )
    associations_file, association_rows = _csv_rows(
        associations_path,
        label="official visualization associations",
        required_columns=_ASSOCIATION_COLUMNS,
        key_columns=("MapName", "QueryTime", "FromGT", "FromID", "ToID"),
    )
    identities: set[tuple[int, int, str]] = set()
    for row in object_rows:
        map_name = _count(row["MapName"], label="visualization map")
        query = _count(row["QueryTime"], label="visualization query")
        if row["IsGT"] not in {"0", "1"} or _NODE_ID.fullmatch(row["ID"]) is None:
            raise ValueError("visualization object identity is invalid")
        if row["IsGT"] == "0":
            identities.add((map_name, query, row["ID"]))
    for row in association_rows:
        _count(row["MapName"], label="association map")
        _count(row["QueryTime"], label="association query")
        from_gt = row["FromGT"]
        if from_gt not in {"0", "1"}:
            raise ValueError("association FromGT is invalid")
        if _NODE_ID.fullmatch(row["FromID"]) is None or (
            row["ToID"] != "Failed" and _NODE_ID.fullmatch(row["ToID"]) is None
        ):
            raise ValueError("association node identity is invalid")
    return objects_file, associations_file, identities


def _candidate_record(
    *,
    entity: Any,
    bridge_object: Mapping[str, Any],
    trajectory: Mapping[str, Any] | None,
) -> dict[str, object]:
    metadata = entity.metadata
    anchor_id = metadata.get("anchor_entity_id")
    temporal_id = metadata.get("temporal_entity_id")
    return {
        "prediction_entity_id": entity.entity_id,
        "node_symbol": bridge_object.get("node_symbol"),
        "authority": metadata.get("authority"),
        "anchor_id": anchor_id,
        "temporal_entity_id": temporal_id,
        "overlay_state": metadata.get("overlay_state"),
        "dynamic_state": (
            bridge_object.get("dynamic_state_at_query")
            if trajectory is None
            else trajectory.get("dynamic_state")
        ),
        "motion_confidence": (
            None if trajectory is None else trajectory.get("motion_confidence")
        ),
        "geometry_epoch": (
            metadata.get("geometry_epoch")
            if trajectory is None
            else trajectory.get("geometry_epoch")
        ),
        "readout_valid": (
            metadata.get("readout_valid")
            if trajectory is None
            else trajectory.get("readout_valid")
        ),
        "lifecycle": entity.lifecycle_state,
        "bound": isinstance(anchor_id, str) and type(temporal_id) is int,
        "association_diagnostic": None,
        "motion_evidence_source": None,
        "geometry_accepted": metadata.get("geometry_gate_accepted"),
        "identity_qualified": isinstance(anchor_id, str)
        or type(temporal_id) is int,
        "proposal_recovery_status": "unavailable",
        "provenance_status": "exact",
    }


def _build_payload(
    *,
    composition_path: Path,
    composition: Mapping[str, Any],
    bridge: Mapping[str, Any],
    dynamic_rows: Sequence[OfficialDynRow],
    visualization_prediction_ids: set[tuple[int, int, str]],
    sources: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    if (
        composition.get("schema_version") != 1
        or composition.get("manifest_id")
        != "crove_ovimap_static_anchor_composition_v1"
        or composition.get("status") != "PASS"
        or composition.get("dataset") != "TESSE-CD"
        or composition.get("scene") != "apartment"
    ):
        raise ValueError("composition identity is invalid")
    if (
        bridge.get("schema_version") != 1
        or bridge.get("dataset") != "TESSE-CD"
        or bridge.get("method") != "OVIV2"
        or bridge.get("mode") != "temporal_checkpoints"
        or bridge.get("scene_id") != "apartment"
    ):
        raise ValueError("bridge identity is invalid")
    composition_checkpoints = composition.get("checkpoints")
    bridge_checkpoints = bridge.get("checkpoints")
    assignments = bridge.get("symbol_assignments")
    if not all(
        isinstance(value, list)
        for value in (composition_checkpoints, bridge_checkpoints, assignments)
    ):
        raise ValueError("checkpoint or symbol assignments are invalid")
    assert isinstance(composition_checkpoints, list)
    assert isinstance(bridge_checkpoints, list)
    assert isinstance(assignments, list)
    assignment_by_entity: dict[str, Mapping[str, Any]] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise ValueError("symbol assignment is invalid")
        entity_id = assignment.get("entity_id")
        if not isinstance(entity_id, str) or entity_id in assignment_by_entity:
            raise ValueError("symbol assignment identity is invalid")
        assignment_by_entity[entity_id] = assignment
    composition_by_timestamp = {
        item.get("timestamp_ns"): item
        for item in composition_checkpoints
        if isinstance(item, Mapping)
    }
    if len(composition_by_timestamp) != len(composition_checkpoints):
        raise ValueError("composition checkpoint identity is invalid")
    trajectory_path, trajectory_record = _bound_file(
        composition_path.parent,
        composition.get("inputs", {}).get("source_trajectories")
        if isinstance(composition.get("inputs"), Mapping)
        else None,
        label="source trajectories",
    )
    if dict(sources.get("source_trajectories", {})) != trajectory_record:
        raise ValueError("trajectory source record disagrees")
    trajectories = _trajectory_rows(trajectory_path)
    events: list[dict[str, object]] = []
    status_mass = {name: 0 for name in ("exact", "ambiguous", "unavailable")}
    for row in dynamic_rows:
        if row.map_name >= len(bridge_checkpoints):
            raise ValueError("official dynamic map index has no bridge checkpoint")
        bridge_checkpoint = bridge_checkpoints[row.map_name]
        if (
            not isinstance(bridge_checkpoint, Mapping)
            or bridge_checkpoint.get("timestamp_ns") != row.query_time_ns
            or type(bridge_checkpoint.get("frame_index")) is not int
        ):
            raise ValueError("official dynamic row and bridge checkpoint disagree")
        composition_checkpoint = composition_by_timestamp.get(row.query_time_ns)
        if not isinstance(composition_checkpoint, Mapping) or (
            composition_checkpoint.get("frame_index")
            != bridge_checkpoint["frame_index"]
        ):
            raise ValueError("bridge checkpoint and composition disagree")
        snapshot_path, _ = _bound_file(
            composition_path.parent,
            composition_checkpoint.get("snapshot"),
            label="composition snapshot",
        )
        entities_path, _ = _bound_file(
            composition_path.parent,
            composition_checkpoint.get("entities"),
            label="composition entities",
        )
        snapshot = read_map_snapshot(snapshot_path, entities_path)
        entity_by_id = {entity.entity_id: entity for entity in snapshot.entities}
        if len(entity_by_id) != len(snapshot.entities):
            raise ValueError("composition contains duplicate entities")
        bridge_objects = bridge_checkpoint.get("objects")
        if not isinstance(bridge_objects, list):
            raise ValueError("bridge checkpoint objects are invalid")
        candidates: list[dict[str, object]] = []
        for bridge_object in bridge_objects:
            if not isinstance(bridge_object, Mapping):
                raise ValueError("bridge object is invalid")
            entity_id = bridge_object.get("entity_id")
            node_symbol = bridge_object.get("node_symbol")
            node_index = bridge_object.get("node_index")
            assignment = assignment_by_entity.get(str(entity_id))
            if (
                not isinstance(entity_id, str)
                or type(node_index) is not int
                or node_symbol != f"O{node_index}"
                or assignment is None
                or assignment.get("node_symbol") != node_symbol
                or assignment.get("node_index") != node_index
                or (
                    row.map_name,
                    row.query_time_ns,
                    f"O({node_index})",
                )
                not in visualization_prediction_ids
            ):
                raise ValueError("bridge object symbol identity is invalid")
            entity = entity_by_id.get(entity_id)
            if entity is None:
                raise ValueError(f"composition entity is missing: {entity_id}")
            temporal_id = entity.metadata.get("temporal_entity_id")
            trajectory = _latest_trajectory(
                trajectories,
                frame_index=int(bridge_checkpoint["frame_index"]),
                temporal_entity_id=temporal_id,
                source_entity_id=assignment.get("source_entity_id"),
            )
            candidates.append(
                _candidate_record(
                    entity=entity,
                    bridge_object=bridge_object,
                    trajectory=trajectory,
                )
            )
        candidates.sort(key=lambda item: str(item["node_symbol"]))
        status = classify_official_dyn_mass(
            row.total_mass,
            candidate_count=len(candidates),
            exact_event_count=0 if row.total_mass == 0 else None,
        )
        status_mass[status] += row.total_mass
        events.append(
            {
                "event_id": f"dyn:{row.map_name}:{row.query_time_ns}",
                "checkpoint_frame": bridge_checkpoint["frame_index"],
                "official": row.to_json_record(),
                "candidate_predictions": candidates,
                "prediction_entity_id": None,
                "authority": None,
                "anchor_id": None,
                "temporal_entity_id": None,
                "overlay_state": None,
                "dynamic_state": None,
                "motion_confidence": None,
                "geometry_epoch": None,
                "readout_valid": None,
                "lifecycle": None,
                "bound": None,
                "association_diagnostic": None,
                "motion_evidence_source": None,
                "geometry_accepted": None,
                "identity_qualified": None,
                "proposal_recovery_status": "unavailable",
                "provenance_status": status,
                "provenance_reason": (
                    "zero_official_dynamic_mass"
                    if row.total_mass == 0
                    else "official_dynamic_metrics_expose_aggregate_trajectory_mass_only"
                ),
                "static_visualization_associations_used_for_dyn": False,
            }
        )
    official_mass = sum(item.total_mass for item in dynamic_rows)
    if sum(status_mass.values()) != official_mass:
        raise RuntimeError("dynamic provenance mass is not conserved")

    def fraction(status: str) -> float:
        if official_mass:
            return status_mass[status] / official_mass
        return 1.0 if status == "exact" else 0.0

    return {
        "schema_version": 1,
        "manifest_id": "crove_official_dyn_provenance_v1",
        "status": "PASS",
        "dataset": "TESSE-CD",
        "scene": "apartment",
        "metric_modification": False,
        "dynamic_association_source": "unavailable_in_official_outputs",
        "coverage": {
            "official_mass": official_mass,
            "exact_mass": status_mass["exact"],
            "ambiguous_mass": status_mass["ambiguous"],
            "unavailable_mass": status_mass["unavailable"],
            "exact_fraction": fraction("exact"),
            "ambiguous_fraction": fraction("ambiguous"),
            "unavailable_fraction": fraction("unavailable"),
        },
        "sources": dict(sources),
        "events": events,
    }


def publish_crove_dyn_provenance(
    *,
    composition_manifest: str | Path,
    bridge_manifest: str | Path,
    official_dynamic_csv: str | Path,
    visualization_objects_csv: str | Path,
    visualization_associations_csv: str | Path,
    output_root: str | Path,
) -> Path:
    output = Path(os.path.abspath(os.fspath(output_root)))
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    composition_path, composition = _json(
        composition_manifest, label="composition manifest"
    )
    bridge_path, bridge = _json(bridge_manifest, label="official bridge manifest")
    dynamic_path = _regular_file(
        official_dynamic_csv, label="official dynamic objects"
    )
    dynamic_rows = parse_official_dynamic_rows(dynamic_path)
    objects_path, associations_path, prediction_ids = _visualization_inputs(
        visualization_objects_csv,
        visualization_associations_csv,
    )
    inputs = {
        "composition_manifest": _record(composition_path),
        "official_bridge_manifest": _record(bridge_path),
        "official_dynamic_objects": _record(dynamic_path),
        "official_visualization_objects": _record(objects_path),
        "official_visualization_associations": _record(associations_path),
    }
    trajectory_path, trajectory_record = _bound_file(
        composition_path.parent,
        composition.get("inputs", {}).get("source_trajectories")
        if isinstance(composition.get("inputs"), Mapping)
        else None,
        label="source trajectories",
    )
    inputs["source_trajectories"] = trajectory_record
    payload = _build_payload(
        composition_path=composition_path,
        composition=composition,
        bridge=bridge,
        dynamic_rows=dynamic_rows,
        visualization_prediction_ids=prediction_ids,
        sources=inputs,
    )
    observed = {
        name: _record(path)
        for name, path in (
            ("composition_manifest", composition_path),
            ("official_bridge_manifest", bridge_path),
            ("official_dynamic_objects", dynamic_path),
            ("official_visualization_objects", objects_path),
            ("official_visualization_associations", associations_path),
            ("source_trajectories", trajectory_path),
        )
    }
    if observed != inputs:
        raise RuntimeError("provenance input changed before publication")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        json_path = staging / "dyn_provenance.json"
        json_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        csv_path = staging / "dyn_provenance.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "event_id",
                    "checkpoint_frame",
                    "detected",
                    "missed",
                    "hallucinated",
                    "total_mass",
                    "candidate_prediction_count",
                    "provenance_status",
                    "provenance_reason",
                ),
            )
            writer.writeheader()
            for event in payload["events"]:  # type: ignore[index]
                official = event["official"]
                writer.writerow(
                    {
                        "event_id": event["event_id"],
                        "checkpoint_frame": event["checkpoint_frame"],
                        "detected": official["detected"],
                        "missed": official["missed"],
                        "hallucinated": official["hallucinated"],
                        "total_mass": official["total_mass"],
                        "candidate_prediction_count": len(
                            event["candidate_predictions"]
                        ),
                        "provenance_status": event["provenance_status"],
                        "provenance_reason": event["provenance_reason"],
                    }
                )
        os.replace(staging, output)
        return output / "dyn_provenance.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "OfficialDynRow",
    "classify_official_dyn_mass",
    "parse_official_dynamic_rows",
    "publish_crove_dyn_provenance",
]
