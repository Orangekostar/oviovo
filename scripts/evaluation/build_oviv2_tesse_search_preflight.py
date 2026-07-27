#!/usr/bin/env python3
"""Build source-recomputed OVIV2 TESSE-CD search preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (  # noqa: E402
    _MECHANISMS_BY_PROFILE,
    _materialize_config,
)
from src.evaluation.oviv2_temporal_occlusion import (  # noqa: E402
    mechanism_telemetry_from_sources,
)


_SOURCE_INDEX_ROLES = (
    "trajectories",
    "lifecycle_transitions",
    "frame_coverage",
    "runtime_diagnostics",
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_symlinks(path: Path, label: str) -> None:
    for component in (path.absolute(), *path.absolute().parents):
        try:
            status = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} contains a symlink")


def _read_bytes(path: Path, label: str) -> bytes:
    absolute = path.absolute()
    _reject_symlinks(absolute, label)
    descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            size += len(chunk)
            if size > 128 * 1024 * 1024:
                raise ValueError(f"{label} exceeds size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    current = os.stat(absolute, follow_symlinks=False)
    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(current):
        raise ValueError(f"{label} changed while reading")
    return b"".join(chunks)


def _json(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _jsonl(content: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.strip():
            rows.append(_json(line, f"{label} line {line_number}"))
    return rows


def _bound_source(
    record: object, *, base: Path, label: str, parse_json: bool = True
) -> tuple[Path, bytes, dict[str, Any] | None]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} source record is not exact")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} source path is invalid")
    relative = Path(raw)
    if not relative.is_absolute() and (".." in relative.parts or relative == Path(".")):
        raise ValueError(f"{label} source path is unsafe")
    path = (relative if relative.is_absolute() else base / relative).absolute()
    content = _read_bytes(path, label)
    if (
        record.get("sha256") != hashlib.sha256(content).hexdigest()
        or type(record.get("byte_count")) is not int
        or record["byte_count"] != len(content)
    ):
        raise ValueError(f"{label} source binding mismatch")
    return path, content, _json(content, label) if parse_json else None


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_record(path: Path, content: bytes) -> dict[str, object]:
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _source_tag(record: Mapping[str, Any], role: str, index: int | str) -> str:
    return f"{role}:{record['sha256']}:{index}"


def _mechanisms(
    *,
    candidate_id: str,
    source_records: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = payloads["runtime_diagnostics"]
    if set(runtime) != {
        "schema_version",
        "execution_profile",
        "processed_frame_count",
        "counters",
        "mechanism_records",
    }:
        raise ValueError("runtime diagnostics mechanism_records are required")
    counters = runtime["counters"]
    explicit = runtime["mechanism_records"]
    if not isinstance(counters, Mapping) or not isinstance(explicit, Mapping) or set(
        explicit
    ) != set(counters):
        raise ValueError("runtime diagnostics mechanism_records inventory mismatch")
    for name, count in counters.items():
        records = explicit[name]
        if (
            type(count) is not int
            or count < 0
            or not isinstance(records, list)
            or len(records) != count
            or any(not isinstance(record, str) or not record for record in records)
            or len(records) != len(set(records))
        ):
            raise ValueError(f"mechanism_records do not match counter: {name}")
    subset_pairs = (
        ("proposal_trigger_count", "proposal_opportunity_count"),
        ("reid_trigger_count", "reid_opportunity_count"),
        ("epoch_reset_trigger_count", "epoch_reset_opportunity_count"),
        ("icp_accept_count", "icp_opportunity_count"),
        ("icp_reject_count", "icp_opportunity_count"),
        ("motion_rejection_count", "icp_opportunity_count"),
        ("ledger_commit_count", "ledger_stage_count"),
        ("ledger_reclaim_count", "ledger_commit_count"),
    )
    if any(not set(explicit[child]) <= set(explicit[parent]) for child, parent in subset_pairs):
        raise ValueError("runtime diagnostics mechanism_records relation mismatch")
    if set(explicit["icp_accept_count"]) & set(explicit["icp_reject_count"]) or set(
        explicit["icp_accept_count"]
    ) | set(explicit["icp_reject_count"]) != set(explicit["icp_opportunity_count"]):
        raise ValueError("runtime diagnostics ICP mechanism_records are not a partition")
    legacy_runtime = {key: runtime[key] for key in (
        "schema_version", "execution_profile", "processed_frame_count", "counters"
    )}
    telemetry = mechanism_telemetry_from_sources(
        trajectories=payloads["trajectories"],
        lifecycle_transitions=payloads["lifecycle_transitions"],
        frame_coverage=payloads["frame_coverage"],
        runtime_diagnostics=legacy_runtime,
        source_records=source_records,
    )
    lifecycle = payloads["lifecycle_transitions"]
    visible_absent = [
        index for index, row in enumerate(lifecycle) if row["evidence"] == "visible_absent"
    ]
    invalidated = [index for index in visible_absent if lifecycle[index]["readout_valid"] is False]
    record_pairs: dict[str, tuple[list[str], list[str], str]] = {
        "proposal_recovery": (
            explicit["proposal_opportunity_count"],
            explicit["proposal_trigger_count"],
            "runtime_diagnostics",
        ),
        "epoch_reset": (
            explicit["epoch_reset_opportunity_count"],
            explicit["epoch_reset_trigger_count"],
            "runtime_diagnostics",
        ),
        "background_release": (
            explicit["ledger_stage_count"],
            explicit["ledger_commit_count"],
            "runtime_diagnostics",
        ),
        "background_reclaim": (
            explicit["ledger_commit_count"],
            explicit["ledger_reclaim_count"],
            "runtime_diagnostics",
        ),
        "eligible_reid": (
            explicit["reid_opportunity_count"],
            explicit["reid_trigger_count"],
            "runtime_diagnostics",
        ),
        "icp": (
            explicit["icp_opportunity_count"],
            explicit["icp_opportunity_count"],
            "runtime_diagnostics",
        ),
        "motion_rejection": (
            explicit["icp_opportunity_count"],
            explicit["motion_rejection_count"],
            "runtime_diagnostics",
        ),
    }
    result: dict[str, Any] = {}
    for name in _MECHANISMS_BY_PROFILE[candidate_id]:
        if name in {"absence", "readout_invalidation"}:
            opportunities = visible_absent
            triggers = visible_absent if name == "absence" else invalidated
            role = "lifecycle_transitions"
        else:
            opportunities, triggers, role = record_pairs[name]
        record = source_records[role]
        result[name] = {
            "opportunity_count": len(opportunities),
            "trigger_count": len(triggers),
            "opportunity_records": [
                _source_tag(record, f"{name}:opportunity", event_id)
                for event_id in opportunities
            ],
            "trigger_records": [
                _source_tag(record, f"{name}:trigger", event_id) for event_id in triggers
            ],
        }
        if not opportunities or not triggers:
            raise ValueError(f"{name} mechanism has no usable opportunity/trigger")
    # Force validation of the evaluator-owned profile projection as well.
    if set(telemetry) - {"icp_attempt", "icp_accept"} - set(result):
        raise ValueError("mechanism telemetry profile differs from candidate")
    return result


def _anchor_coverage(occlusion: Mapping[str, Any]) -> dict[str, Any]:
    mappings = occlusion.get("anchor_mappings")
    macro = occlusion.get("macro")
    declared = macro.get("anchor_coverage_gate") if isinstance(macro, Mapping) else None
    if not isinstance(mappings, list) or not isinstance(declared, Mapping):
        raise ValueError("anchor mapping evidence is missing")
    eligible_ids: list[int] = []
    mapped_ids: list[int] = []
    zero_overlap = 0
    ambiguous = 0
    identities: set[tuple[str, int]] = set()
    expected_fields = {
        "scene",
        "object_id",
        "lifecycle_index",
        "anchor_frame_index",
        "anchor_relative_timestamp_ns",
        "eligible",
        "target_voxel_count",
        "mapped_temporal_id",
        "overlap_voxel_count",
        "ambiguous",
    }
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, Mapping) or set(mapping) != expected_fields:
            raise ValueError("anchor mapping schema is not exact")
        object_id = mapping.get("object_id")
        lifecycle_index = mapping.get("lifecycle_index")
        if not (
            mapping.get("scene") == "apartment"
            and not isinstance(object_id, bool)
            and isinstance(object_id, (str, int))
            and str(object_id).strip()
            and type(lifecycle_index) is int
            and lifecycle_index >= 0
            and type(mapping.get("anchor_frame_index")) is int
            and mapping["anchor_frame_index"] >= 0
            and type(mapping.get("anchor_relative_timestamp_ns")) is int
            and mapping["anchor_relative_timestamp_ns"] >= 0
            and type(mapping.get("eligible")) is bool
            and type(mapping.get("target_voxel_count")) is int
            and mapping["target_voxel_count"] >= 0
            and mapping["eligible"] == (mapping["target_voxel_count"] > 0)
        ):
            raise ValueError("anchor mapping evidence is invalid")
        identity = (str(object_id), lifecycle_index)
        if identity in identities:
            raise ValueError("duplicate anchor mapping identity")
        identities.add(identity)
        if mapping.get("eligible") is not True:
            continue
        overlap = mapping.get("overlap_voxel_count")
        temporal_id = mapping.get("mapped_temporal_id")
        is_ambiguous = mapping.get("ambiguous")
        if type(overlap) is not int or overlap < 0 or type(is_ambiguous) is not bool:
            raise ValueError("anchor mapping evidence value is invalid")
        if temporal_id is not None and (type(temporal_id) is not int or temporal_id < 0):
            raise ValueError("anchor temporal ID is invalid")
        eligible_ids.append(index)
        if temporal_id is not None and overlap > 0 and not is_ambiguous:
            mapped_ids.append(index)
        zero_overlap += int(overlap == 0)
        ambiguous += int(is_ambiguous)
    recomputed = {
        "scene": "apartment",
        "eligible_count": len(eligible_ids),
        "uniquely_mapped_count": len(mapped_ids),
        "zero_overlap_count": zero_overlap,
        "ambiguous_count": ambiguous,
        "required_eligible_count": 66,
        "required_mapped_count": 53,
        "available": bool(eligible_ids),
        "passed": len(eligible_ids) >= 66 and len(mapped_ids) >= 53,
        "reason": (
            "no_eligible_anchor_mappings"
            if not eligible_ids
            else "insufficient_eligible_anchor_mappings"
            if len(eligible_ids) < 66
            else "insufficient_unique_anchor_mappings"
            if len(mapped_ids) < 53
            else None
        ),
    }
    if dict(declared) != recomputed:
        raise ValueError("anchor coverage gate differs from source mappings")
    if len(eligible_ids) != 66:
        raise ValueError("Apartment anchor coverage requires exactly 66 eligible anchors")
    return {
        "eligible_anchor_ids": eligible_ids,
        "mapped_anchor_ids": mapped_ids,
        "eligible_count": len(eligible_ids),
        "mapped_count": len(mapped_ids),
    }


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.absolute()
    _reject_symlinks(output.parent, "output parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinks(output.parent, "output parent")
    content = _canonical(payload) + b"\n"
    temporary = output.parent / f".{output.name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644
    )
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, output, follow_symlinks=False)
    except FileExistsError:
        raise FileExistsError(f"output already exists: {output}")
    finally:
        temporary.unlink(missing_ok=True)


def _revalidate_record(record: Mapping[str, Any], label: str) -> None:
    path = Path(record["path"])
    content = _read_bytes(path, label)
    if (
        hashlib.sha256(content).hexdigest() != record["sha256"]
        or len(content) != record["byte_count"]
    ):
        raise ValueError(f"{label} changed before publication")


def build_preflight(
    *,
    search_manifest: str | Path,
    apartment_base_config: str | Path,
    candidate_sources: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(search_manifest).absolute()
    base_path = Path(apartment_base_config).absolute()
    sources_path = Path(candidate_sources).absolute()
    manifest_bytes = _read_bytes(manifest_path, "search manifest")
    base_bytes = _read_bytes(base_path, "Apartment base config")
    sources_bytes = _read_bytes(sources_path, "candidate source index")
    manifest = _json(manifest_bytes, "search manifest")
    base = _json(base_bytes, "Apartment base config")
    sources = _json(sources_bytes, "candidate source index")
    input_records = {
        "search manifest": _file_record(manifest_path, manifest_bytes),
        "Apartment base config": _file_record(base_path, base_bytes),
        "candidate source index": _file_record(sources_path, sources_bytes),
    }
    if set(sources) != {"schema_version", "manifest_id", "candidates"} or not (
        sources.get("schema_version") == 1
        and sources.get("manifest_id") == "oviv2_tesse_search_preflight_sources_v1"
        and isinstance(sources.get("candidates"), list)
        and sources["candidates"]
    ):
        raise ValueError("candidate source index identity/schema is invalid")
    declarations = {
        item["candidate_id"]: item for item in manifest.get("candidates", [])
    }
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in sources["candidates"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "candidate_id",
            "run_manifest",
            "temporal_occlusion_result",
            "future_leakage_evidence",
        }:
            raise ValueError("candidate source entry is invalid")
        candidate_id = entry.get("candidate_id")
        if candidate_id in seen or candidate_id not in declarations:
            raise ValueError(f"candidate source identity is invalid: {candidate_id}")
        seen.add(candidate_id)
        run_path, run_content, run = _bound_source(
            entry["run_manifest"], base=sources_path.parent, label=f"{candidate_id} run manifest"
        )
        assert run is not None and isinstance(candidate_id, str)
        materialized = _materialize_config(base, declarations[candidate_id])
        if not (
            run.get("dataset") == "TESSE-CD"
            and run.get("protocol_id") == "oviv2-tessecd-v2"
            and run.get("scene") == "apartment"
            and run.get("method_id") == "OVIV2"
            and run.get("algorithm_hash") == materialized["algorithm_hash"]
        ):
            raise ValueError(f"{candidate_id} run identity mismatch")
        source_path, source_content, source_index = _bound_source(
            run.get("source_index"), base=run_path.parent, label=f"{candidate_id} source_index"
        )
        assert source_index is not None
        if not (
            source_index.get("dataset") == "TESSE-CD"
            and source_index.get("method") == "OVIV2"
            and source_index.get("scene") == "apartment"
        ):
            raise ValueError(f"{candidate_id} source_index identity mismatch")
        source_records: dict[str, Mapping[str, Any]] = {}
        payloads: dict[str, Any] = {}
        source_evidence: dict[str, dict[str, object]] = {
            "run_manifest": _file_record(run_path, run_content),
            "source_index": _file_record(source_path, source_content),
        }
        for role in _SOURCE_INDEX_ROLES:
            role_path, role_content, role_json = _bound_source(
                source_index.get(role),
                base=source_path.parent,
                label=role,
                parse_json=role == "runtime_diagnostics",
            )
            source_records[role] = source_index[role]
            source_evidence[role] = _file_record(role_path, role_content)
            payloads[role] = (
                role_json if role == "runtime_diagnostics" else _jsonl(role_content, role)
            )
        if payloads["runtime_diagnostics"].get("execution_profile") != candidate_id:
            raise ValueError(f"{candidate_id} runtime diagnostics profile mismatch")
        expected_frames = run.get("scheduled_frame_indices")
        observed_frames = [row.get("frame_index") for row in payloads["frame_coverage"]]
        if not (
            isinstance(expected_frames, list)
            and expected_frames
            and all(type(item) is int and item >= 0 for item in expected_frames)
            and len(expected_frames) == len(set(expected_frames))
            and expected_frames == sorted(expected_frames)
            and observed_frames == expected_frames
        ):
            raise ValueError(f"{candidate_id} frame coverage differs from run schedule")
        occlusion_path, occlusion_content, occlusion = _bound_source(
            entry["temporal_occlusion_result"],
            base=sources_path.parent,
            label=f"{candidate_id} temporal occlusion result",
        )
        assert occlusion is not None
        source_evidence["temporal_occlusion_result"] = _file_record(
            occlusion_path, occlusion_content
        )
        bindings = occlusion.get("input_bindings")
        bound_indexes = bindings.get("source_indexes") if isinstance(bindings, Mapping) else None
        expected_source_record = {
            "scene": "apartment",
            "path": run["source_index"]["path"],
            "sha256": hashlib.sha256(source_content).hexdigest(),
            "byte_count": len(source_content),
        }
        if bound_indexes != [expected_source_record]:
            raise ValueError(f"{candidate_id} occlusion source_index binding mismatch")
        leakage_path, leakage_content, leakage = _bound_source(
            entry["future_leakage_evidence"],
            base=sources_path.parent,
            label=f"{candidate_id} future leakage evidence",
        )
        assert leakage is not None
        source_evidence["future_leakage_evidence"] = _file_record(
            leakage_path, leakage_content
        )
        if set(leakage) != {
            "schema_version",
            "manifest_id",
            "scene",
            "candidate_id",
            "source_index",
            "records",
        } or not (
            leakage.get("schema_version") == 1
            and leakage.get("manifest_id") == "oviv2_tesse_future_leakage_evidence_v1"
            and leakage.get("scene") == "apartment"
            and leakage.get("candidate_id") == candidate_id
            and leakage.get("source_index") == run.get("source_index")
            and isinstance(leakage.get("records"), list)
        ):
            raise ValueError(f"{candidate_id} future leakage evidence is invalid")
        if leakage["records"]:
            raise ValueError(f"{candidate_id} future leakage detected")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_config_sha256": hashlib.sha256(
                    _canonical(materialized)
                ).hexdigest(),
                "frame_coverage": {
                    "expected_frame_indices": expected_frames,
                    "observed_frame_indices": observed_frames,
                    "expected_count": len(expected_frames),
                    "observed_count": len(observed_frames),
                },
                "future_leakage": {
                    "count": len(leakage["records"]),
                    "records": leakage["records"],
                },
                "anchor_coverage": _anchor_coverage(occlusion),
                "mechanisms": _mechanisms(
                    candidate_id=candidate_id,
                    source_records=source_records,
                    payloads=payloads,
                ),
                "source_evidence": source_evidence,
            }
        )
    canonical_order = [item["candidate_id"] for item in manifest["candidates"]]
    if [item["candidate_id"] for item in candidates] != [
        name for name in canonical_order if name in seen
    ]:
        raise ValueError("candidate sources must preserve canonical order")
    evidence = {
        "schema_version": 1,
        "manifest_id": "oviv2_dual_readout_search_preflight_v1",
        "scene": "apartment",
        "bindings": {
            "search_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "apartment_base_config_file_sha256": hashlib.sha256(base_bytes).hexdigest(),
        },
        "candidates": candidates,
    }
    for label, record in input_records.items():
        _revalidate_record(record, label)
    for candidate in candidates:
        for role, record in candidate["source_evidence"].items():
            _revalidate_record(record, f"{candidate['candidate_id']} {role}")
    _publish(Path(output), evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-manifest", required=True, type=Path)
    parser.add_argument("--apartment-base-config", required=True, type=Path)
    parser.add_argument("--candidate-sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    build_preflight(
        search_manifest=args.search_manifest,
        apartment_base_config=args.apartment_base_config,
        candidate_sources=args.candidate_sources,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
