#!/usr/bin/env python3
"""Select a dual-readout profile from packaged immutable Apartment evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.package_oviv2_tesse_dual_readout_result import (  # noqa: E402
    load_and_revalidate_result,
)
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (  # noqa: E402
    load_search_manifest,
)


RESULT_CONTRACT = "candidates/<candidate_id>/apartment/result.json"
_CANDIDATE_IDS = ("a0", "a1", "a2", "a3", "a4")
_REQUIRED_METRICS = (
    "current_miou",
    "object_f1",
    "ghost_rate",
    "background_f5_cm",
    "recovery_frames",
    "runtime_seconds",
)
_OPTIONAL_TIE_AXES = ("dynamic_f1", "change_f1")
_REQUIRED_GATES = ("correctness", "causality", "determinism", "t1_exact")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def _stable_bytes(path: Path, label: str) -> bytes:
    absolute = path.absolute()
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
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    witness = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    current = os.stat(absolute, follow_symlinks=False)
    if witness(before) != witness(after) or witness(after) != witness(current):
        raise ValueError(f"{label} changed while it was read")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return data


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    _reject_symlink_components(path.parent, "selection output parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, "selection output parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = _canonical_json(payload) + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _metric_value(result: Mapping[str, Any], name: str) -> float:
    metric = result["metrics"][name]
    if metric["available"] is not True or not isinstance(metric["value"], (int, float)):
        raise ValueError(f"metric {name} is unavailable")
    return float(metric["value"])


def _rank_key(
    result: Mapping[str, Any], optional_axes: tuple[str, ...]
) -> tuple[float | str, ...]:
    key: list[float | str] = [
        _metric_value(result, "ghost_rate"),
        -_metric_value(result, "background_f5_cm"),
        _metric_value(result, "recovery_frames"),
    ]
    key.extend(-_metric_value(result, name) for name in optional_axes)
    key.extend(
        (
            _metric_value(result, "runtime_seconds"),
            result["bindings"]["config_sha256"],
        )
    )
    return tuple(key)


def _result_file_record(path: Path, data: bytes, candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def tune(
    manifest_path: str | Path,
    result_paths: Sequence[str | Path],
    output: str | Path,
    *,
    results_root: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output).absolute()
    _reject_symlink_components(destination.parent, "selection output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest_file = Path(manifest_path).absolute()
    manifest = load_search_manifest(manifest_file)
    manifest_bytes = _stable_bytes(manifest_file, "search manifest")
    if isinstance(result_paths, (str, bytes)) or not result_paths:
        raise ValueError("result_paths must be a non-empty sequence")

    loaded: dict[str, dict[str, Any]] = {}
    result_records: dict[str, dict[str, Any]] = {}
    for raw_path in result_paths:
        path = Path(raw_path).absolute()
        before = _stable_bytes(path, "packaged candidate result")
        result = load_and_revalidate_result(path, manifest=manifest)
        after = _stable_bytes(path, "packaged candidate result")
        if before != after:
            raise ValueError("packaged candidate result changed during ingestion")
        candidate_id = result["candidate_id"]
        if candidate_id not in _CANDIDATE_IDS:
            raise ValueError(f"undeclared candidate: {candidate_id!r}")
        if candidate_id in loaded:
            raise ValueError(f"duplicate candidate result: {candidate_id}")
        if results_root is not None:
            expected_path = (
                Path(results_root).absolute()
                / f"candidates/{candidate_id}/apartment/result.json"
            )
            if path != expected_path:
                raise ValueError(
                    f"candidate result path does not match bound identity: {path}"
                )
        loaded[candidate_id] = result
        result_records[candidate_id] = _result_file_record(path, after, candidate_id)

    if "a0" not in loaded:
        raise ValueError("A0 result is required to establish promotion floors")
    a0 = loaded["a0"]
    for name in _REQUIRED_METRICS:
        if a0["metrics"][name]["available"] is not True:
            raise ValueError(f"A0 required metric is unavailable: {name}")
    if any(a0["gates"][name]["passed"] is not True for name in _REQUIRED_GATES):
        raise ValueError("A0 must pass every evidence gate")

    baseline_non_temporal = a0["bindings"]["non_temporal_config_sha256"]
    baseline_inputs = a0["bindings"]["input_hashes"]
    for candidate_id, result in loaded.items():
        if result["bindings"]["non_temporal_config_sha256"] != baseline_non_temporal:
            raise ValueError(
                f"candidate {candidate_id} non-temporal config hash differs from A0"
            )
        if result["bindings"]["input_hashes"] != baseline_inputs:
            raise ValueError(f"candidate {candidate_id} input hash bindings differ from A0")

    optional_axes: list[str] = []
    skipped_optional: dict[str, str] = {}
    for name in _OPTIONAL_TIE_AXES:
        pattern = {result["metrics"][name]["available"] for result in loaded.values()}
        if len(pattern) != 1:
            raise ValueError(f"optional metric availability differs across candidates: {name}")
        if pattern == {True}:
            optional_axes.append(name)
        else:
            skipped_optional[name] = "all_candidates_unavailable_matching_a0"

    current_floor = _metric_value(a0, "current_miou")
    object_floor = _metric_value(a0, "object_f1")
    eligible: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for candidate_id in _CANDIDATE_IDS:
        if candidate_id not in loaded:
            continue
        result = loaded[candidate_id]
        reasons: list[str] = []
        notes = [
            f"{name}_axis_skipped_all_unavailable" for name in skipped_optional
        ]
        for name in _REQUIRED_GATES:
            if result["gates"][name]["passed"] is not True:
                reasons.append(f"{name}_gate_failed")
        for name in _REQUIRED_METRICS:
            if result["metrics"][name]["available"] is not True:
                reasons.append(f"{name}_unavailable")
        if not any(reason.endswith("_unavailable") for reason in reasons):
            if _metric_value(result, "current_miou") < current_floor:
                reasons.append("current_miou_below_a0_floor")
            if _metric_value(result, "object_f1") < object_floor:
                reasons.append("object_f1_below_a0_floor")
        entry = {
            "candidate_id": candidate_id,
            "eligible": not reasons,
            "selected": False,
            "reasons": reasons,
            "notes": notes,
            "config_sha256": result["bindings"]["config_sha256"],
            "algorithm_hash": result["run_identity"]["algorithm_hash"],
            "gates": result["gates"],
            "metrics": result["metrics"],
            "sources": result["sources"],
        }
        ledger.append(entry)
        if not reasons:
            eligible.append(result)

    selected = (
        min(eligible, key=lambda value: _rank_key(value, tuple(optional_axes)))
        if eligible
        else None
    )
    selected_id = selected["candidate_id"] if selected is not None else None
    for entry in ledger:
        entry["selected"] = entry["candidate_id"] == selected_id
        if entry["eligible"] and not entry["selected"]:
            entry["reasons"].append("lost_lexicographic_promotion")

    selected_config = (
        selected["bindings"]["candidate"]["config"] if selected is not None else None
    )
    selected_record = (
        selected["sources"]["candidate_config"] if selected is not None else None
    )
    selection: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_v2_selection",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "protocol_id": "oviv2-tessecd-v2",
        "status": "PASS" if selected is not None else "NO_ELIGIBLE_CANDIDATE",
        "development_scene": "apartment",
        "transfer_scene": "office",
        "office_results_read": False,
        "scenes_read": ["apartment"],
        "result_contract": RESULT_CONTRACT,
        "results_root": str(Path(results_root).absolute()) if results_root else None,
        "manifest": {
            "path": str(manifest_file),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "byte_count": len(manifest_bytes),
        },
        "result_files": [
            result_records[candidate_id]
            for candidate_id in _CANDIDATE_IDS
            if candidate_id in result_records
        ],
        "promotion_order": manifest["promotion_order"],
        "metric_policy": manifest["metric_policy"],
        "floors": {
            "current_miou_from_a0": current_floor,
            "object_f1_from_a0": object_floor,
        },
        "skipped_optional_tie_axes": skipped_optional,
        "selected_candidate_id": selected_id,
        "selected_config": selected_config,
        "selected_config_record": selected_record,
        "selected_config_sha256": (
            selected["bindings"]["config_sha256"] if selected is not None else None
        ),
        "algorithm_hash": (
            selected["run_identity"]["algorithm_hash"] if selected is not None else None
        ),
        "rejection_ledger": ledger,
    }

    current_manifest_bytes = _stable_bytes(manifest_file, "search manifest")
    if current_manifest_bytes != manifest_bytes:
        raise ValueError("search manifest changed during selection")
    if load_search_manifest(manifest_file) != manifest:
        raise ValueError("search manifest changed during selection")
    for candidate_id, original in loaded.items():
        record = result_records[candidate_id]
        path = Path(record["path"])
        before = _stable_bytes(path, "packaged candidate result")
        if (
            hashlib.sha256(before).hexdigest() != record["sha256"]
            or len(before) != record["byte_count"]
        ):
            raise ValueError("packaged candidate result changed during selection")
        revalidated = load_and_revalidate_result(path, manifest=manifest)
        after = _stable_bytes(path, "packaged candidate result")
        if before != after or revalidated != original:
            raise ValueError("packaged candidate result changed during selection")
    _atomic_write_new(destination, selection)
    return selection


def tune_results_root(
    manifest_path: str | Path,
    results_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest = load_search_manifest(manifest_path)
    root = Path(results_root).absolute()
    _reject_symlink_components(root, "results root")
    if not root.is_dir():
        raise FileNotFoundError(f"missing results root: {root}")
    paths: list[Path] = []
    for candidate in manifest["candidates"]:
        relative = Path(
            f"candidates/{candidate['candidate_id']}/apartment/result.json"
        )
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing immutable candidate result: {relative}")
        paths.append(path)
    return tune(manifest_path, paths, output, results_root=root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = tune_results_root(args.manifest, args.results_root, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "selected_candidate_id": result["selected_candidate_id"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
