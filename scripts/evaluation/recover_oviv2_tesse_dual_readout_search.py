#!/usr/bin/env python3
"""Recover the externally killed A4 OVIV2 search candidate immutably."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (  # noqa: E402
    _validate_preflight_gate_evidence,
    _validate_temporal_frontend_source_binding,
    input_binding_values_sha256,
    load_search_manifest,
    non_temporal_config_sha256,
)
from scripts.evaluation.run_oviv2_tesse_cd_v2 import _input_sha256  # noqa: E402
from scripts.evaluation.oviv2_tesse_cd_v2_config import (  # noqa: E402
    canonical_algorithm_config,
    canonical_algorithm_hash,
)


CANDIDATE_IDS = ("a0", "a1", "a2", "a3", "a4")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_STATUS_KEYS = {
    "schema_version",
    "manifest",
    "apartment_base_config",
    "office_binding",
    "preflight_gate_evidence",
    "max_parallel",
    "gpu_ids",
    "required_available_ram_bytes",
    "observed_available_ram_bytes",
    "status",
    "candidates",
    "unscheduled_candidate_ids",
}
_CANDIDATE_RECORD_KEYS = {
    "candidate_id",
    "scene",
    "status",
    "command",
    "pid",
    "exit_code",
    "cuda_visible_devices",
    "config_path",
    "config_file",
    "output_root",
    "stdout_path",
    "stderr_path",
    "stdout_file",
    "stderr_file",
    "config_sha256",
    "algorithm_hash",
    "non_temporal_config_sha256",
    "input_binding_values_sha256",
    "input_hashes",
    "run_identity",
    "runtime_seconds",
    "failure_reason",
}
_BASE_SOURCE_BINDING_KEYS = {
    "input_manifest",
    "export_manifest",
    "camera",
    "trajectory",
    "timestamps",
    "vocabulary_json",
    "vocabulary_txt",
    "frontend_manifest",
    "dense_manifest",
    "rgbd_combined_output_sha256",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} path contains a symlink: {current}")


@dataclass(frozen=True)
class Snapshot:
    path: Path
    data: bytes
    payload: dict[str, Any] | None
    witness: tuple[int, int, int, int, int]

    @property
    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "byte_count": len(self.data),
        }

    def revalidate(self, label: str) -> None:
        _reject_symlink_components(self.path, label)
        current = os.stat(self.path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or _identity(current) != self.witness:
            raise ValueError(f"{label} changed after snapshot")


@dataclass(frozen=True)
class ContentWitness:
    path: Path
    sha256: str
    byte_count: int
    witness: tuple[int, int, int, int, int]

    @property
    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_count": self.byte_count,
        }

    def revalidate(self, label: str) -> None:
        _reject_symlink_components(self.path, label)
        current = os.stat(self.path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or _identity(current) != self.witness:
            raise ValueError(f"{label} changed after snapshot")


def _snapshot(path: str | Path, label: str, *, parse_json: bool = True) -> Snapshot:
    absolute = Path(path).absolute()
    _reject_symlink_components(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise FileNotFoundError(f"missing {label}: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if parse_json and before.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{label} exceeds 16 MiB")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(absolute, follow_symlinks=False)
    if _identity(before) != _identity(after) or _identity(after) != _identity(current):
        raise ValueError(f"{label} changed while it was read")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise ValueError(f"{label} size changed while it was read")
    payload: dict[str, Any] | None = None
    if parse_json:
        try:
            payload = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"JSON must be UTF-8: {absolute}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSON root must be an object: {absolute}")
    return Snapshot(absolute, data, payload, _identity(after))


def _content_witness(path: str | Path, label: str) -> ContentWitness:
    absolute = Path(path).absolute()
    _reject_symlink_components(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise FileNotFoundError(f"missing {label}: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(absolute, follow_symlinks=False)
    if _identity(before) != _identity(after) or _identity(after) != _identity(current):
        raise ValueError(f"{label} changed while it was read")
    if byte_count != after.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return ContentWitness(absolute, digest.hexdigest(), byte_count, _identity(after))


def _require_record(record: object, snapshot: Snapshot, label: str) -> None:
    if not isinstance(record, Mapping) or dict(record) != snapshot.record:
        raise ValueError(f"{label} record mismatch")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _require_status_manifest(
    status: Mapping[str, Any], manifest: Snapshot, label: str
) -> None:
    expected = {"path": str(manifest.path), "sha256": manifest.record["sha256"]}
    if status.get("manifest") != expected:
        raise ValueError(f"{label} manifest record mismatch")


def _snapshot_preflight_gate_evidence(
    original: Mapping[str, Any],
    retry: Mapping[str, Any],
    manifest: Snapshot,
    apartment: Snapshot,
) -> list[Any]:
    original_record = original.get("preflight_gate_evidence")
    retry_record = retry.get("preflight_gate_evidence")
    if not (
        isinstance(original_record, Mapping)
        and isinstance(original_record.get("path"), str)
        and isinstance(retry_record, Mapping)
        and isinstance(retry_record.get("path"), str)
    ):
        raise ValueError("preflight gate evidence record is invalid")
    if dict(original_record) != dict(retry_record):
        raise ValueError("preflight gate evidence records differ")

    original_preflight = _snapshot(
        original_record["path"], "original preflight gate evidence"
    )
    retry_preflight = _snapshot(
        retry_record["path"], "retry preflight gate evidence"
    )
    _require_record(
        original_record, original_preflight, "original preflight gate evidence"
    )
    _require_record(retry_record, retry_preflight, "retry preflight gate evidence")
    if original_preflight.data != retry_preflight.data:
        raise ValueError("original and retry preflight gate evidence bytes differ")

    assert manifest.payload is not None and apartment.payload is not None
    declarations = {
        item["candidate_id"]: item for item in manifest.payload["candidates"]
    }
    validated = _validate_preflight_gate_evidence(
        original_preflight.path,
        manifest_bytes=manifest.data,
        apartment_bytes=apartment.data,
        apartment=apartment.payload,
        declarations=declarations,
        selected_ids=CANDIDATE_IDS,
    )
    if dict(validated) != original_preflight.record:
        raise ValueError("preflight gate evidence validation record mismatch")
    return [original_preflight, retry_preflight, *validated.witnesses]


def _snapshot_bound_inputs(
    original: Mapping[str, Any], retry: Mapping[str, Any], manifest: Snapshot
) -> list[Any]:
    _require_status_manifest(original, manifest, "original")
    _require_status_manifest(retry, manifest, "retry")
    original_apartment = original.get("apartment_base_config")
    retry_apartment = retry.get("apartment_base_config")
    if not isinstance(original_apartment, Mapping) or not isinstance(
        original_apartment.get("path"), str
    ):
        raise ValueError("Apartment base config record mismatch")
    apartment = _snapshot(original_apartment["path"], "Apartment base config")
    expected_apartment = {
        "path": str(apartment.path),
        "file_sha256": apartment.record["sha256"],
    }
    if dict(original_apartment) != expected_apartment or retry_apartment != expected_apartment:
        raise ValueError("Apartment base config record mismatch")

    original_office = original.get("office_binding")
    retry_office = retry.get("office_binding")
    if not isinstance(original_office, Mapping) or not isinstance(
        original_office.get("config_path"), str
    ):
        raise ValueError("Office binding record mismatch")
    office = _snapshot(original_office["config_path"], "Office config")
    assert office.payload is not None
    expected_office = {
        "scene": "office",
        "executed": False,
        "config_path": str(office.path),
        "file_sha256": office.record["sha256"],
        "config_sha256": hashlib.sha256(_canonical(office.payload)).hexdigest(),
        "algorithm_hash": office.payload.get("algorithm_hash"),
    }
    if dict(original_office) != expected_office or retry_office != expected_office:
        raise ValueError("Office binding record mismatch")
    if office.payload.get("algorithm_hash") != canonical_algorithm_hash(office.payload):
        raise ValueError("Office binding algorithm hash mismatch")
    assert apartment.payload is not None
    if apartment.payload.get("algorithm_hash") != canonical_algorithm_hash(
        apartment.payload
    ):
        raise ValueError("Apartment base config algorithm hash mismatch")
    apartment_algorithm = canonical_algorithm_config(apartment.payload)
    office_algorithm = canonical_algorithm_config(office.payload)
    apartment_algorithm.pop("temporal_readout", None)
    office_algorithm.pop("temporal_readout", None)
    if apartment_algorithm != office_algorithm:
        raise ValueError(
            "Apartment base config mismatch; Office binding non-temporal mismatch"
        )
    preflight_witnesses = _snapshot_preflight_gate_evidence(
        original, retry, manifest, apartment
    )
    return [apartment, office, *preflight_witnesses]


def _validate_candidate_config(
    record: Mapping[str, Any],
    declaration: Mapping[str, Any],
    apartment_base: Mapping[str, Any],
    label: str,
) -> Snapshot:
    if (
        record.get("scene") != "apartment"
        or record.get("status") not in {"PASS", "FAIL"}
        or (record.get("status") == "PASS" and record.get("exit_code") != 0)
    ):
        raise ValueError(f"{label} candidate status mismatch")
    path = record.get("config_path")
    if not isinstance(path, str):
        raise ValueError(f"{label} config path is missing")
    config = _snapshot(path, f"{label} config")
    assert config.payload is not None
    _require_record(record.get("config_file"), config, f"{label} config")
    canonical_hash = hashlib.sha256(_canonical(config.payload)).hexdigest()
    if record.get("config_sha256") != canonical_hash:
        raise ValueError(f"{label} canonical config hash mismatch")
    algorithm_hash = canonical_algorithm_hash(config.payload)
    if (
        config.payload.get("algorithm_hash") != algorithm_hash
        or record.get("algorithm_hash") != algorithm_hash
    ):
        raise ValueError(f"{label} algorithm hash mismatch")
    if config.payload.get("temporal_readout") != declaration.get("temporal_readout"):
        raise ValueError(f"{label} config differs from manifest declaration")
    expected = dict(apartment_base)
    expected["temporal_readout"] = declaration["temporal_readout"]
    expected["algorithm_hash"] = canonical_algorithm_hash(expected)
    if config.payload != expected:
        raise ValueError(f"Apartment base config candidate mismatch for {label}")
    non_temporal = non_temporal_config_sha256(config.payload)
    if record.get("non_temporal_config_sha256") != non_temporal:
        raise ValueError(f"{label} non-temporal hash mismatch")
    input_binding = input_binding_values_sha256(config.payload)
    if record.get("input_binding_values_sha256") != input_binding:
        raise ValueError(f"{label} input binding values hash mismatch")
    return config


def _snapshot_temporal_frontend_manifest(
    config: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    label: str,
) -> Snapshot | None:
    temporal_binding = source_bindings.get("temporal_frontend_manifest")
    if temporal_binding is None:
        return None
    raw_path = config.get("temporal_frontend_manifest")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} temporal frontend manifest path is invalid")
    temporal_path = Path(raw_path)
    if not temporal_path.is_absolute():
        temporal_path = REPO_ROOT / temporal_path
    temporal_manifest = _snapshot(
        temporal_path, f"{label} temporal frontend manifest", parse_json=False
    )
    expected = {
        "sha256": temporal_manifest.record["sha256"],
        "byte_count": temporal_manifest.record["byte_count"],
    }
    if temporal_binding != expected:
        raise ValueError(f"{label} temporal frontend source binding mismatch")
    return temporal_manifest


def _config_path(config: Mapping[str, Any], field: str, label: str) -> Path:
    raw_path = config.get(field)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} {field} path is invalid")
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _dataset_source_paths(
    config: Mapping[str, Any], label: str
) -> dict[str, Path]:
    dataset_root = _config_path(config, "dataset_root", label)
    scene = config.get("scene")
    if not isinstance(scene, str) or not scene:
        raise ValueError(f"{label} scene is invalid")
    if (dataset_root / "results").is_dir():
        scene_root = dataset_root
        camera = dataset_root.parent / "cam_params.json"
    elif (dataset_root / scene / "results").is_dir():
        scene_root = dataset_root / scene
        camera = dataset_root / "cam_params.json"
    else:
        raise ValueError(f"{label} dataset root is invalid")
    return {
        "camera": camera,
        "trajectory": scene_root / "traj.txt",
        "timestamps": scene_root / "timestamps.csv",
    }


def _snapshot_base_source_bindings(
    config: Mapping[str, Any], source_bindings: Mapping[str, Any], label: str
) -> list[Snapshot | ContentWitness]:
    expected_keys = set(_BASE_SOURCE_BINDING_KEYS)
    if "temporal_frontend_manifest" in source_bindings:
        expected_keys.add("temporal_frontend_manifest")
    if set(source_bindings) != expected_keys:
        raise ValueError(f"{label} source binding schema mismatch")

    paths = {
        field: _config_path(config, field, label)
        for field in (
            "input_manifest",
            "export_manifest",
            "vocabulary_json",
            "vocabulary_txt",
            "frontend_manifest",
            "dense_manifest",
        )
    }
    paths.update(_dataset_source_paths(config, label))
    snapshots: list[Snapshot | ContentWitness] = []
    snapshots_by_role: dict[str, Snapshot] = {}
    export_manifest: Snapshot | None = None
    for role, path in paths.items():
        snapshot = _snapshot(
            path,
            f"{label} {role}",
            parse_json=role == "export_manifest",
        )
        expected = {
            "sha256": snapshot.record["sha256"],
            "byte_count": snapshot.record["byte_count"],
        }
        if source_bindings.get(role) != expected:
            raise ValueError(f"{label} {role} source binding mismatch")
        snapshots.append(snapshot)
        snapshots_by_role[role] = snapshot
        if role == "export_manifest":
            export_manifest = snapshot

    assert export_manifest is not None and export_manifest.payload is not None
    frame_count = config.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f"{label} frame_count is invalid")
    results_root = snapshots_by_role["trajectory"].path.parent / "results"
    rgb_paths = {
        results_root / f"frame{index:06d}.jpg" for index in range(frame_count)
    }
    depth_paths = {
        results_root / f"depth{index:06d}.png" for index in range(frame_count)
    }
    if set(results_root.glob("frame*.jpg")) != rgb_paths:
        raise ValueError(f"{label} RGB frame inventory mismatch")
    if set(results_root.glob("depth*.png")) != depth_paths:
        raise ValueError(f"{label} depth frame inventory mismatch")
    frame_snapshots = [
        _content_witness(path, f"{label} RGB-D frame")
        for index in range(frame_count)
        for path in (
            results_root / f"frame{index:06d}.jpg",
            results_root / f"depth{index:06d}.png",
        )
    ]
    export_inputs = [
        *frame_snapshots,
        snapshots_by_role["trajectory"],
        snapshots_by_role["timestamps"],
        snapshots_by_role["camera"],
    ]
    camera_parent = snapshots_by_role["camera"].path.parent
    digest = hashlib.sha256()
    for snapshot in sorted(
        export_inputs, key=lambda item: str(item.path.relative_to(camera_parent))
    ):
        relative = str(snapshot.path.relative_to(camera_parent))
        digest.update(
            relative.encode("utf-8")
            + b"\0"
            + snapshot.record["sha256"].encode("ascii")
            + b"\n"
        )
    combined = digest.hexdigest()
    if (
        export_manifest.payload.get("combined_output_sha256") != combined
        or export_manifest.payload.get("file_hash_count") != len(export_inputs)
    ):
        raise ValueError(f"{label} RGB-D combined output mismatch")
    if source_bindings.get("rgbd_combined_output_sha256") != combined:
        raise ValueError(f"{label} RGB-D source binding mismatch")
    snapshots.extend(frame_snapshots)
    return snapshots


def _snapshot_runner_inputs(
    config: Snapshot,
    source_bindings: Mapping[str, Any],
    input_sha256: object,
    label: str,
) -> list[Snapshot]:
    assert config.payload is not None
    schedule = _snapshot(
        _config_path(config.payload, "schedule_manifest", label),
        f"{label} schedule",
        parse_json=False,
    )
    target = _snapshot(
        _config_path(config.payload, "occlusion_target_manifest", label),
        f"{label} target",
        parse_json=False,
    )
    expected = _input_sha256(config.data, schedule.data, target.data, source_bindings)
    if input_sha256 != expected:
        raise ValueError(f"{label} input_sha256 mismatch")
    return [schedule, target]


def _validate_pass_run(
    record: Mapping[str, Any], config: Snapshot, label: str
) -> tuple[list[Snapshot | ContentWitness], str, Mapping[str, Any]]:
    if not (
        record.get("status") == "PASS"
        and record.get("exit_code") == 0
        and record.get("failure_reason") is None
    ):
        raise ValueError(f"{label} must be PASS")
    snapshots: list[Snapshot | ContentWitness] = []
    for stream in ("stdout", "stderr"):
        path = record.get(f"{stream}_path")
        if not isinstance(path, str):
            raise ValueError(f"{label} {stream} path is missing")
        stream_snapshot = _snapshot(path, f"{label} {stream}", parse_json=False)
        _require_record(record.get(f"{stream}_file"), stream_snapshot, f"{label} {stream}")
        snapshots.append(stream_snapshot)
    output_root = record.get("output_root")
    if not isinstance(output_root, str):
        raise ValueError(f"{label} output root is missing")
    run_manifest = _snapshot(Path(output_root) / "run_manifest.json", f"{label} run manifest")
    assert run_manifest.payload is not None and config.payload is not None
    run = run_manifest.payload
    _require_record(run.get("config"), config, f"{label} run config")
    identity = {
        key: run.get(key)
        for key in ("algorithm_hash", "input_sha256", "code_commit", "source_bindings")
    }
    if (
        run.get("algorithm_hash") != config.payload.get("algorithm_hash")
        or record.get("run_identity") != identity
        or not isinstance(run.get("input_sha256"), str)
    ):
        raise ValueError(f"{label} run algorithm/input identity mismatch")
    source_bindings = run.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or record.get("input_hashes") != source_bindings:
        raise ValueError(f"{label} run source bindings mismatch")
    _validate_temporal_frontend_source_binding(config.payload, source_bindings)
    snapshots.extend(
        _snapshot_base_source_bindings(config.payload, source_bindings, label)
    )
    temporal_manifest = _snapshot_temporal_frontend_manifest(
        config.payload, source_bindings, label
    )
    if temporal_manifest is not None:
        snapshots.append(temporal_manifest)
    snapshots.extend(
        _snapshot_runner_inputs(config, source_bindings, run.get("input_sha256"), label)
    )
    code_commit = run.get("code_commit")
    if not isinstance(code_commit, str):
        raise ValueError(f"{label} run code commit is invalid")
    snapshots.append(run_manifest)
    return snapshots, code_commit, source_bindings


def _validate_artifacts(
    original: Mapping[str, Any],
    retry: Mapping[str, Any],
    manifest: Snapshot,
) -> list[Any]:
    assert manifest.payload is not None
    witnesses = _snapshot_bound_inputs(original, retry, manifest)
    apartment_base = witnesses[0].payload
    assert apartment_base is not None
    declarations = {
        item["candidate_id"]: item for item in manifest.payload["candidates"]
    }
    config_snapshots: list[Snapshot] = []
    successful: list[tuple[Mapping[str, Any], Snapshot, str]] = []
    for record in original["candidates"]:
        candidate_id = record["candidate_id"]
        config = _validate_candidate_config(
            record,
            declarations[candidate_id],
            apartment_base,
            f"original {candidate_id.upper()}",
        )
        config_snapshots.append(config)
        if candidate_id != "a4":
            successful.append((record, config, f"original {candidate_id.upper()}"))
    retry_record = retry["candidates"][0]
    retry_config = _validate_candidate_config(
        retry_record, declarations["a4"], apartment_base, "retry A4"
    )
    config_snapshots.append(retry_config)
    if config_snapshots[4].data != retry_config.data:
        raise ValueError("original and retry A4 config bytes differ")
    non_temporal = {record["non_temporal_config_sha256"] for record in original["candidates"]}
    non_temporal.add(retry_record["non_temporal_config_sha256"])
    if len(non_temporal) != 1:
        raise ValueError("candidate non-temporal hashes differ")
    input_bindings = {record["input_binding_values_sha256"] for record in original["candidates"]}
    input_bindings.add(retry_record["input_binding_values_sha256"])
    if len(input_bindings) != 1:
        raise ValueError("candidate input binding values hashes differ")
    successful.append((retry_record, retry_config, "retry A4"))
    commits: list[str] = []
    sources: list[Mapping[str, Any]] = []
    for record, config, label in successful:
        run_witnesses, commit, bindings = _validate_pass_run(record, config, label)
        witnesses.extend(run_witnesses)
        commits.append(commit)
        sources.append(bindings)
    if any(commit != commits[0] for commit in commits[1:]):
        raise ValueError("successful run code commit differs")
    common_sources = [
        {
            name: value
            for name, value in bindings.items()
            if name != "temporal_frontend_manifest"
        }
        for bindings in sources
    ]
    if any(bindings != common_sources[0] for bindings in common_sources[1:]):
        raise ValueError("successful run source bindings differ")
    witnesses.extend(config_snapshots)
    return witnesses


def _candidate_ids(status: Mapping[str, Any]) -> list[Any]:
    candidates = status.get("candidates")
    if not isinstance(candidates, list) or any(
        not isinstance(item, Mapping) for item in candidates
    ):
        raise ValueError("search candidates must be a list of objects")
    if any(set(item) != _CANDIDATE_RECORD_KEYS for item in candidates):
        raise ValueError("candidate record keys mismatch")
    return [item.get("candidate_id") for item in candidates]


def _require_empty_unscheduled(status: Mapping[str, Any], label: str) -> None:
    if status.get("unscheduled_candidate_ids") != []:
        raise ValueError(f"{label} has unscheduled candidates")


def _validate_status_shapes(
    original: Mapping[str, Any], retry: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if set(original) != _STATUS_KEYS or set(retry) != _STATUS_KEYS:
        raise ValueError("search status keys mismatch")
    if (
        type(original.get("schema_version")) is not int
        or original["schema_version"] != 1
        or type(retry.get("schema_version")) is not int
        or retry["schema_version"] != 1
    ):
        raise ValueError("schema_version must be integer 1")
    if original.get("status") != "FAIL":
        raise ValueError("original search status must be FAIL")
    if _candidate_ids(original) != list(CANDIDATE_IDS):
        raise ValueError("original candidates must be exact A0-A4 order")
    if retry.get("status") != "PASS" or _candidate_ids(retry) != ["a4"]:
        raise ValueError("retry must contain exactly one PASS A4")
    _require_empty_unscheduled(original, "original search status")
    _require_empty_unscheduled(retry, "retry search status")
    return original["candidates"][4], retry["candidates"][0]


def _validate_original_sigkill(record: Mapping[str, Any]) -> list[Snapshot]:
    if record.get("scene") != "apartment" or record.get("status") != "FAIL":
        raise ValueError("original A4 must be an Apartment FAIL")
    if record.get("exit_code") != -9:
        raise ValueError("original A4 exit code must be -9")
    if record.get("failure_reason") is not None:
        raise ValueError("original A4 failure reason must be null")
    if record.get("input_hashes") is not None:
        raise ValueError("original A4 input hashes must be null")
    if record.get("run_identity") is not None:
        raise ValueError("original A4 run identity must be null")
    streams: list[Snapshot] = []
    for stream in ("stdout", "stderr"):
        path = record.get(f"{stream}_path")
        if not isinstance(path, str):
            raise ValueError(f"original A4 {stream} path is missing")
        snapshot = _snapshot(path, f"original A4 {stream}", parse_json=False)
        _require_record(record.get(f"{stream}_file"), snapshot, f"original A4 {stream}")
        if snapshot.data:
            raise ValueError(f"original A4 {stream} must be empty")
        streams.append(snapshot)
    return streams


def _validate_retry_a4(record: Mapping[str, Any]) -> None:
    if (
        record.get("candidate_id") != "a4"
        or record.get("scene") != "apartment"
        or record.get("status") != "PASS"
        or record.get("exit_code") != 0
    ):
        raise ValueError("retry must contain exactly one PASS A4")


def _status_source_record(snapshot: Snapshot) -> dict[str, Any]:
    return snapshot.record


def _revalidate_sources(witnesses: list[Any]) -> None:
    for witness in witnesses:
        witness.revalidate(f"recovery source {witness.path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_composite(
    staging: Path,
    composite: Mapping[str, Any],
) -> None:
    candidates = staging / "candidates"
    candidates.mkdir()
    for candidate_id in CANDIDATE_IDS:
        apartment = candidates / candidate_id / "apartment"
        apartment.mkdir(parents=True)
        _fsync_directory(apartment)
        _fsync_directory(apartment.parent)
    _fsync_directory(candidates)
    data = _canonical(composite) + b"\n"
    _write_fsynced(staging / "search_status.json", data)
    _fsync_directory(staging)


def _publish(
    staging: Path, destination: Path, witnesses: list[Any]
) -> None:
    _revalidate_sources(witnesses)
    try:
        destination.mkdir()
    except FileExistsError:
        raise FileExistsError(f"output root already exists: {destination}")
    try:
        os.rename(staging, destination)
    except BaseException:
        try:
            destination.rmdir()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(destination.parent)


def recover_search(
    *,
    manifest_path: str | Path,
    original_status: str | Path,
    retry_status: str | Path,
    output_root: str | Path,
) -> Path:
    manifest = _snapshot(manifest_path, "search manifest")
    assert manifest.payload is not None
    load_search_manifest(manifest.path)
    original = _snapshot(original_status, "original search status")
    retry = _snapshot(retry_status, "retry search status")
    assert original.payload is not None and retry.payload is not None
    original_a4, retry_a4 = _validate_status_shapes(original.payload, retry.payload)
    witnesses = [manifest, original, retry, *_validate_original_sigkill(original_a4)]
    _validate_retry_a4(retry_a4)
    witnesses.extend(_validate_artifacts(original.payload, retry.payload, manifest))

    destination = Path(output_root).absolute()
    if destination.exists():
        raise FileExistsError(f"output root already exists: {destination}")
    _reject_symlink_components(destination.parent, "output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    composite = dict(original.payload)
    composite["status"] = "PASS"
    composite["candidates"] = [*original.payload["candidates"][:4], retry_a4]
    composite["recovery"] = {
        "strategy": "immutable_single_candidate_retry_v1",
        "replaced_candidate_id": "a4",
        "original_status": _status_source_record(original),
        "retry_status": _status_source_record(retry),
        "accepted_original_failure": {
            "candidate_id": "a4",
            "scene": "apartment",
            "status": "FAIL",
            "exit_code": -9,
            "stdout_file": original_a4["stdout_file"],
            "stderr_file": original_a4["stderr_file"],
            "failure_reason": None,
            "input_hashes": None,
            "run_identity": None,
        },
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        _build_composite(staging, composite)
        _publish(staging, destination, witnesses)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination / "search_status.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--original-status", required=True, type=Path)
    parser.add_argument("--retry-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    status_path = recover_search(
        manifest_path=args.manifest,
        original_status=args.original_status,
        retry_status=args.retry_status,
        output_root=args.output,
    )
    print(json.dumps({"status_path": str(status_path.absolute())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
