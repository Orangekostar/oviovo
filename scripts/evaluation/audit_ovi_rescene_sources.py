#!/usr/bin/env python3
"""Verify pinned OVI-MAP, ReScene4D, and voxel-contract source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_EXPECTED_REPRESENTATIONS = {
    "ovi_mapping": (0.01, True, False, True),
    "rescene_neural": (0.02, False, True, False),
    "crove_legacy_temporal": (0.05, True, False, False),
    "evaluator": (0.05, False, True, False),
}


class SourceAuditError(ValueError):
    """Raised when a declared source or representation identity is invalid."""


@dataclass(frozen=True, slots=True)
class FileBinding:
    source_id: str
    path: str
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class SourceAuditResult:
    status: str
    voxel_status: str
    checkpoint_status: str
    source_commits: Mapping[str, str]
    voxel_sizes_m: Mapping[str, float]
    verified_files: tuple[FileBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "voxel_contract_status": self.voxel_status,
            "checkpoint_status": self.checkpoint_status,
            "source_commits": dict(sorted(self.source_commits.items())),
            "voxel_sizes_m": dict(sorted(self.voxel_sizes_m.items())),
            "verified_files": [asdict(binding) for binding in self.verified_files],
        }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceAuditError(f"{label} must be a mapping")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceAuditError(f"{label} must be a non-empty string")
    return value.strip()


def _git(checkout: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise SourceAuditError(f"Git source identity check failed: {detail}") from error


def _resolve_checkout(value: object, source_id: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise SourceAuditError(f"{source_id} checkout must not be a symlink")
    path = path.resolve()
    if not path.is_dir():
        raise SourceAuditError(f"{source_id} checkout is missing: {path}")
    return path


def _source_file(checkout: Path, relative: str) -> Path:
    path = checkout / relative
    if path.is_symlink() or not path.is_file():
        raise SourceAuditError(f"required source file is missing or a symlink: {relative}")
    resolved = path.resolve()
    try:
        resolved.relative_to(checkout)
    except ValueError as error:
        raise SourceAuditError(f"required source file escapes checkout: {relative}") from error
    return resolved


def _verify_source(
    source_id: str,
    declaration: Mapping[str, Any],
    checkout: Path,
) -> tuple[str, tuple[FileBinding, ...], dict[str, bytes]]:
    commit = _string(declaration.get("commit"), f"{source_id}.commit")
    if _GIT_SHA.fullmatch(commit) is None:
        raise SourceAuditError(f"{source_id}.commit must be a full lowercase Git SHA")
    resolved = _git(checkout, "rev-parse", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise SourceAuditError(f"{source_id} commit resolved unexpectedly: {resolved}")

    remote_name = declaration.get("remote_name")
    if remote_name is not None:
        remote = _string(remote_name, f"{source_id}.remote_name")
        expected_url = _string(
            declaration.get("repository_url"), f"{source_id}.repository_url"
        )
        observed_url = _git(checkout, "remote", "get-url", remote).decode().strip()
        if observed_url != expected_url:
            raise SourceAuditError(
                f"{source_id} remote URL mismatch: expected {expected_url}, got {observed_url}"
            )

    required = declaration.get("required_files")
    if not isinstance(required, list) or not required:
        raise SourceAuditError(f"{source_id}.required_files must be a non-empty list")
    bindings: list[FileBinding] = []
    content_by_path: dict[str, bytes] = {}
    for index, raw_binding in enumerate(required):
        binding = _mapping(raw_binding, f"{source_id}.required_files[{index}]")
        relative = _string(binding.get("path"), "required source path")
        if relative in content_by_path:
            raise SourceAuditError(f"duplicate required source path: {source_id}:{relative}")
        declared_sha256 = _string(binding.get("sha256"), f"{relative}.sha256")
        if _SHA256.fullmatch(declared_sha256) is None:
            raise SourceAuditError(f"{relative} SHA-256 must be 64 lowercase hex digits")
        byte_count = binding.get("byte_count")
        if type(byte_count) is not int or byte_count <= 0:
            raise SourceAuditError(f"{relative} byte_count must be a positive integer")
        frozen = _git(checkout, "show", f"{commit}:{relative}")
        working = _source_file(checkout, relative).read_bytes()
        if working != frozen:
            raise SourceAuditError(
                f"{source_id}:{relative} differs from pinned commit {commit}"
            )
        observed_sha256 = hashlib.sha256(frozen).hexdigest()
        if observed_sha256 != declared_sha256:
            raise SourceAuditError(
                f"{source_id}:{relative} SHA-256 mismatch: "
                f"expected {declared_sha256}, got {observed_sha256}"
            )
        if len(frozen) != byte_count:
            raise SourceAuditError(
                f"{source_id}:{relative} byte count mismatch: "
                f"expected {byte_count}, got {len(frozen)}"
            )
        content_by_path[relative] = frozen
        bindings.append(
            FileBinding(source_id, relative, observed_sha256, len(frozen))
        )

    license_name = declaration.get("license")
    if not isinstance(license_name, str) or not license_name.strip():
        raise SourceAuditError(f"{source_id}.license must be a non-empty string")
    license_path = declaration.get("license_path")
    if license_path is not None:
        relative = _string(license_path, f"{source_id}.license_path")
        if relative not in content_by_path:
            raise SourceAuditError(f"{source_id} license file is not source-bound")
        if license_name == "MIT" and b"MIT License" not in content_by_path[relative]:
            raise SourceAuditError(f"{source_id} license content is not MIT")
    return commit, tuple(bindings), content_by_path


def _verify_evidence(
    declaration: Mapping[str, Any],
    contents: Mapping[str, Mapping[str, bytes]],
    label: str,
) -> None:
    source_id = _string(declaration.get("source_id"), f"{label}.source_id")
    if source_id not in contents:
        raise SourceAuditError(f"{label} references unknown source: {source_id}")
    evidence = _mapping(declaration.get("evidence"), f"{label}.evidence")
    path = _string(evidence.get("path"), f"{label}.evidence.path")
    expected = _string(evidence.get("contains"), f"{label}.evidence.contains")
    content = contents[source_id].get(path)
    if content is None:
        raise SourceAuditError(f"{label} evidence file is not source-bound: {source_id}:{path}")
    if expected not in content.decode("utf-8", errors="strict"):
        raise SourceAuditError(f"{label} evidence text is absent from {source_id}:{path}")


def audit_sources(
    manifest: Mapping[str, Any],
    checkouts: Mapping[str, Path],
) -> SourceAuditResult:
    """Audit exact Git/file identities and the four-role voxel contract."""

    manifest = _mapping(manifest, "manifest")
    if manifest.get("schema_version") != 1:
        raise SourceAuditError("manifest schema_version must be 1")
    if manifest.get("status") != "EXTERNAL_SOURCE_PASS":
        raise SourceAuditError("manifest status must be EXTERNAL_SOURCE_PASS")
    if manifest.get("voxel_contract_status") != "VOXEL_CONTRACT_PASS":
        raise SourceAuditError("manifest voxel contract must be VOXEL_CONTRACT_PASS")

    sources = _mapping(manifest.get("sources"), "sources")
    if set(sources) != {"ovimap", "rescene", "current"}:
        raise SourceAuditError("sources must contain exactly ovimap, rescene, and current")
    if set(checkouts) != set(sources):
        raise SourceAuditError("checkout IDs must exactly match manifest source IDs")

    commits: dict[str, str] = {}
    contents: dict[str, Mapping[str, bytes]] = {}
    verified: list[FileBinding] = []
    for source_id in ("ovimap", "rescene", "current"):
        checkout = _resolve_checkout(checkouts[source_id], source_id)
        commit, bindings, source_contents = _verify_source(
            source_id, _mapping(sources[source_id], source_id), checkout
        )
        commits[source_id] = commit
        contents[source_id] = source_contents
        verified.extend(bindings)

    representations = _mapping(manifest.get("representations"), "representations")
    if set(representations) != set(_EXPECTED_REPRESENTATIONS):
        raise SourceAuditError("representations must contain exactly the four voxel roles")
    voxel_sizes: dict[str, float] = {}
    for name, expected in _EXPECTED_REPRESENTATIONS.items():
        declaration = _mapping(representations[name], f"representations.{name}")
        resolution = declaration.get("resolution_m")
        if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
            raise SourceAuditError(f"{name}.resolution_m must be numeric")
        observed = (
            float(resolution),
            declaration.get("persistent"),
            declaration.get("input_only"),
            declaration.get("final_map_authoritative"),
        )
        if observed != expected:
            if name == "rescene_neural" and observed[-1] is True:
                raise SourceAuditError("ReScene neural tokens cannot be final map geometry")
            raise SourceAuditError(f"{name} representation contract mismatch")
        _verify_evidence(declaration, contents, f"representations.{name}")
        voxel_sizes[name] = float(resolution)

    adapter = _mapping(manifest.get("adapter_contract"), "adapter_contract")
    if adapter.get("spatial_quantization_axes") != ["x", "y", "z"]:
        raise SourceAuditError("adapter spatial quantization must exclude visit/time")
    if adapter.get("temporal_coordinate") != "exact_visit_index_0_or_1":
        raise SourceAuditError("adapter temporal coordinate must be the exact visit index")
    if adapter.get("reverse_index") != "csr_all_contributing_ovi_samples":
        raise SourceAuditError("adapter reverse index must preserve every OVI sample")
    if adapter.get("final_geometry_sources") != ["ovi_t0", "ovi_t1"]:
        raise SourceAuditError("final geometry sources must be OVI t0 and OVI t1 only")

    checkpoint = _mapping(manifest.get("checkpoint"), "checkpoint")
    checkpoint_status = _string(checkpoint.get("status"), "checkpoint.status")
    if checkpoint_status not in {
        "EXTERNAL_SOURCE_PASS",
        "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
    }:
        raise SourceAuditError("unsupported checkpoint status")
    if checkpoint_status == "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT":
        _verify_evidence(checkpoint, contents, "checkpoint")
    else:
        artifact = _mapping(checkpoint.get("artifact"), "checkpoint.artifact")
        sha256 = _string(artifact.get("sha256"), "checkpoint.artifact.sha256")
        byte_count = artifact.get("byte_count")
        if _SHA256.fullmatch(sha256) is None or type(byte_count) is not int or byte_count <= 0:
            raise SourceAuditError("available checkpoint must have a valid byte binding")

    return SourceAuditResult(
        status="EXTERNAL_SOURCE_PASS",
        voxel_status="VOXEL_CONTRACT_PASS",
        checkpoint_status=checkpoint_status,
        source_commits=dict(sorted(commits.items())),
        voxel_sizes_m=dict(sorted(voxel_sizes.items())),
        verified_files=tuple(verified),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "configs/external/ovi_rescene_sources.json",
    )
    parser.add_argument("--ovimap-checkout", type=Path, required=True)
    parser.add_argument("--rescene-checkout", type=Path, required=True)
    parser.add_argument("--repository-checkout", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = audit_sources(
            payload,
            {
                "ovimap": args.ovimap_checkout,
                "rescene": args.rescene_checkout,
                "current": args.repository_checkout,
            },
        )
        _write_json_atomic(args.output, result.to_dict())
    except (OSError, json.JSONDecodeError, SourceAuditError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), **result.to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
