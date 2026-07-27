#!/usr/bin/env python3
"""Run the OVIV2 v1 path in a narrow reference worker."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.compare_oviv2_cumulative_artifacts import (
    compare_cumulative_artifacts,
)
from scripts.evaluation.run_oviv2_tesse_cd import run as run_v1


DEFAULT_SOURCE_MANIFEST = Path(
    "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
)
DEVELOPMENT_MODE = "apartment_development_unfrozen"
ReferenceRunner = Callable[..., Mapping[str, Any]]


def _regular_bytes(path: Path, label: str) -> bytes:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink")
    metadata = os.lstat(absolute)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    return absolute.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _commit(repo: Path) -> str:
    value = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip().decode("ascii")
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("repository commit is invalid")
    return value


def _input_fingerprints(config_data: bytes) -> dict[str, str]:
    config = json.loads(config_data.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("reference config root is not an object")
    result = {"config": _sha256(config_data)}
    for name, value in sorted(config.items()):
        if name.endswith("_sha256"):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"reference input fingerprint is invalid: {name}")
            result[name] = value
    return result


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_reference(
    *,
    config: str | Path,
    output: str | Path,
    freeze_manifest: str | Path,
    run_slot: str,
    receipt: str | Path,
    argv: Sequence[str],
    repo: Path = REPO_ROOT,
    source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
    runner: ReferenceRunner = run_v1,
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    config_path = Path(config).absolute()
    output_path = Path(output).absolute()
    receipt_path = Path(receipt).absolute()
    source_path = Path(source_manifest)
    if not source_path.is_absolute():
        source_path = repo / source_path
    config_data = _regular_bytes(config_path, "reference config")
    source_data = _regular_bytes(source_path, "T1 source manifest")
    exact_argv = list(argv)
    if not exact_argv or any(not isinstance(item, str) or not item for item in exact_argv):
        raise ValueError("reference argv is invalid")
    expected_argv = [
        exact_argv[0], str(Path(__file__).resolve()),
        "--config", str(config_path), "--output", str(output_path),
        "--freeze-manifest", str(Path(freeze_manifest).absolute()),
        "--run-slot", run_slot, "--receipt", str(receipt_path),
        "--source-manifest", str(source_path.absolute()),
    ]
    if (
        not Path(exact_argv[0]).is_absolute()
        or exact_argv != expected_argv
        or receipt_path != output_path / "t1_exact_receipt.json"
    ):
        raise ValueError("reference argv/receipt binding is not canonical")
    code_commit = _commit(repo)
    input_fingerprints = _input_fingerprints(config_data)
    runner(
        config_path,
        output_path,
        freeze_manifest=Path(freeze_manifest),
        run_slot=run_slot,
    )
    audit = compare_cumulative_artifacts(output_path, output_path)
    if _commit(repo) != code_commit:
        raise ValueError("repository commit changed during reference run")
    if _regular_bytes(config_path, "reference config") != config_data:
        raise ValueError("reference config changed during run")
    if _regular_bytes(source_path, "T1 source manifest") != source_data:
        raise ValueError("T1 source manifest changed during run")
    execution = {
        "profile": "reference",
        "argv": exact_argv,
        "pid": os.getpid(),
        "code_commit": code_commit,
        "source_manifest_sha256": _sha256(source_data),
        "input_fingerprints": input_fingerprints,
        "output_root": str(output_path),
    }
    payload = {
        "schema_version": 1,
        "format": "oviv2_t1_exact_execution_receipt_v1",
        "execution": execution,
        "source_manifest": {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_data),
            "byte_count": len(source_data),
        },
        "artifact_inventory": audit["inventory"],
        "checkpoint_frames": audit["checkpoint_frames"],
        "cumulative_root_sha256": audit["root_sha256"],
    }
    _atomic_receipt(receipt_path, payload)
    return payload


def run_development_reference(
    *,
    config: str | Path,
    output: str | Path,
    receipt: str | Path,
    argv: Sequence[str],
    repo: Path = REPO_ROOT,
    source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
    runner: ReferenceRunner = run_v1,
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    config_path = Path(config).absolute()
    output_path = Path(output).absolute()
    receipt_path = Path(receipt).absolute()
    source_path = Path(source_manifest)
    if not source_path.is_absolute():
        source_path = repo / source_path
    config_data = _regular_bytes(config_path, "reference config")
    source_data = _regular_bytes(source_path, "T1 source manifest")
    try:
        config_value = json.loads(config_data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference config is invalid") from exc
    if not isinstance(config_value, dict) or config_value.get("scene") != "apartment":
        raise ValueError("development reference requires Apartment config")
    exact_argv = list(argv)
    if not exact_argv or any(not isinstance(item, str) or not item for item in exact_argv):
        raise ValueError("reference argv is invalid")
    expected_argv = [
        exact_argv[0], str(Path(__file__).resolve()),
        "--config", str(config_path), "--output", str(output_path),
        "--receipt", str(receipt_path),
        "--source-manifest", str(source_path.absolute()),
    ]
    if (
        not Path(exact_argv[0]).is_absolute()
        or exact_argv != expected_argv
        or receipt_path != output_path / "t1_exact_receipt.json"
    ):
        raise ValueError("reference argv/receipt binding is not canonical")
    code_commit = _commit(repo)
    input_fingerprints = _input_fingerprints(config_data)
    runner(config_path, output_path, freeze_manifest=None, run_slot=None)
    audit = compare_cumulative_artifacts(output_path, output_path)
    if _commit(repo) != code_commit:
        raise ValueError("repository commit changed during reference run")
    if _regular_bytes(config_path, "reference config") != config_data:
        raise ValueError("reference config changed during run")
    if _regular_bytes(source_path, "T1 source manifest") != source_data:
        raise ValueError("T1 source manifest changed during run")
    execution = {
        "profile": "reference",
        "mode": DEVELOPMENT_MODE,
        "argv": exact_argv,
        "pid": os.getpid(),
        "code_commit": code_commit,
        "source_manifest_sha256": _sha256(source_data),
        "input_fingerprints": input_fingerprints,
        "output_root": str(output_path),
    }
    payload = {
        "schema_version": 1,
        "format": "oviv2_t1_exact_execution_receipt_v1",
        "execution": execution,
        "source_manifest": {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_data),
            "byte_count": len(source_data),
        },
        "artifact_inventory": audit["inventory"],
        "checkpoint_frames": audit["checkpoint_frames"],
        "cumulative_root_sha256": audit["root_sha256"],
    }
    _atomic_receipt(receipt_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exact_argv = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    run_development_reference(
        config=args.config,
        output=args.output,
        receipt=args.receipt,
        argv=exact_argv,
        source_manifest=args.source_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
