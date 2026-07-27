#!/usr/bin/env python3
"""Freeze source-bound Apartment baseline evidence for OVIV2 T2 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Mapping, Sequence


_METRICS = (
    "dynamic_f1",
    "change_f1",
    "ghost_rate",
    "background_f5_cm",
    "recovery_frames",
    "current_miou",
    "object_f1",
)
_PACKAGE_FIELDS = {
    "schema_version",
    "manifest_id",
    "dataset",
    "protocol_id",
    "scene",
    "status",
    "method_id",
    "oracle",
    "metrics",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_symlinks(path: Path, label: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            status = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} contains a symlink")


def _read(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    absolute = path.absolute()
    _reject_symlinks(absolute, label)
    descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        content = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            content += chunk
            if len(content) > 8 * 1024 * 1024:
                raise ValueError(f"{label} exceeds size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    current = os.stat(absolute, follow_symlinks=False)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError(f"{label} changed while reading")
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload, content


def _record(path: Path, content: bytes) -> dict[str, object]:
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.absolute()
    _reject_symlinks(output.parent, "output parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinks(output.parent, "output parent")
    content = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
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


def freeze_baseline_evidence(
    *, packages: Sequence[str | Path], output: str | Path
) -> dict[str, Any]:
    if not packages:
        raise ValueError("baseline packages are required")
    seen_paths: set[Path] = set()
    package_records: list[dict[str, object]] = []
    metric_sources: dict[str, dict[str, object]] = {}
    for supplied in packages:
        path = Path(supplied).absolute()
        if path in seen_paths:
            raise ValueError(f"duplicate baseline package: {path}")
        seen_paths.add(path)
        payload, content = _read(path, "baseline package")
        if set(payload) != _PACKAGE_FIELDS or not (
            payload.get("schema_version") == 1
            and payload.get("manifest_id")
            == "tesse-cd-frozen-baseline-scene-result-v1"
            and payload.get("dataset") == "TESSE-CD"
            and payload.get("protocol_id") == "oviv2-tessecd-v2"
            and payload.get("scene") == "apartment"
            and payload.get("status") == "PASS"
            and isinstance(payload.get("method_id"), str)
            and bool(payload["method_id"])
            and payload.get("oracle") is False
            and isinstance(payload.get("metrics"), Mapping)
        ):
            raise ValueError("baseline package identity/schema is invalid")
        source = _record(path, content)
        package_records.append(source)
        for name, value in payload["metrics"].items():
            if name not in _METRICS:
                continue
            if name in metric_sources:
                raise ValueError(f"duplicate baseline metric: {name}")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"baseline metric is not finite: {name}")
            numeric = float(value)
            if (name == "recovery_frames" and numeric < 0.0) or (
                name != "recovery_frames" and not 0.0 <= numeric <= 1.0
            ):
                raise ValueError(f"baseline metric domain is invalid: {name}")
            metric_sources[name] = source
    missing = [name for name in _METRICS if name not in metric_sources]
    if missing:
        raise ValueError(f"missing baseline metrics: {missing}")
    evidence = {
        "schema_version": 1,
        "manifest_id": "oviv2-tesse-dual-readout-baseline-evidence-v1",
        "dataset": "TESSE-CD",
        "protocol_id": "oviv2-tessecd-v2",
        "scene": "apartment",
        "baselines": {
            name: {"metric": name, "source": metric_sources[name]}
            for name in _METRICS
        },
    }
    for record in package_records:
        _, content = _read(Path(record["path"]), "baseline package revalidation")
        if (
            hashlib.sha256(content).hexdigest() != record["sha256"]
            or len(content) != record["byte_count"]
        ):
            raise ValueError("baseline package changed before publication")
    _publish(Path(output), evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    freeze_baseline_evidence(packages=args.package, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
