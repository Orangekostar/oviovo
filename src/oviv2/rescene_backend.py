"""Fail-closed boundary for an optional external ReScene inference process."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import NeuralSampleMap, TemporalQueryEvidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPOSITORY_ROOT / "configs/external/ovi_rescene_sources.json"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ReSceneBackendError(ValueError):
    """Raised for malformed or contradictory ReScene backend identities."""


def _digest(payload: object) -> str:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _git(checkout: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _file_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReSceneBackendError(f"{label} is missing or a symlink")
    before = path.stat(follow_symlinks=False)
    content = path.read_bytes()
    after = path.stat(follow_symlinks=False)
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
        raise ReSceneBackendError(f"{label} changed while being read")
    return content


class ReSceneBackend:
    """Validate external identities before any learned dependency is imported."""

    def __init__(
        self,
        *,
        checkout: str | Path,
        checkpoint: str | Path,
        source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
        checkpoint_sha256: str | None = None,
        backend_name: str = "concerto",
    ) -> None:
        self.checkout = Path(checkout)
        self.checkpoint = Path(checkpoint)
        self.source_manifest = Path(source_manifest)
        self.checkpoint_sha256 = checkpoint_sha256
        self.backend_name = backend_name.strip().lower()
        if self.backend_name != "concerto":
            raise ReSceneBackendError("only the source-bound Concerto backend is enabled")

    def _backend_config_sha256(self) -> str:
        manifest_hash = (
            hashlib.sha256(self.source_manifest.read_bytes()).hexdigest()
            if self.source_manifest.is_file() and not self.source_manifest.is_symlink()
            else "missing"
        )
        return _digest(
            {
                "backend_name": self.backend_name,
                "source_manifest_sha256": manifest_hash,
                "checkpoint_sha256": self.checkpoint_sha256,
            }
        )

    def _blocked(
        self, pair: NeuralSampleMap, status: str, reason: str
    ) -> TemporalQueryEvidence:
        result = TemporalQueryEvidence.blocked(
            status=status,
            backend_name=f"rescene:{self.backend_name}",
            backend_config_sha256=self._backend_config_sha256(),
            pair_sha256=pair.content_sha256(),
            diagnostics={"reason": reason},
        )
        validate_query_evidence(pair, result)
        return result

    def _validate_source(self) -> bool:
        try:
            if self.checkout.is_symlink() or not self.checkout.is_dir():
                return False
            manifest = json.loads(
                _file_bytes(self.source_manifest, "ReScene source manifest").decode(
                    "utf-8"
                )
            )
            declaration = manifest["sources"]["rescene"]
            commit = declaration["commit"]
            required_files = declaration["required_files"]
            if (
                manifest.get("status") != "EXTERNAL_SOURCE_PASS"
                or not isinstance(commit, str)
                or _GIT_SHA.fullmatch(commit) is None
                or not isinstance(required_files, list)
                or not required_files
            ):
                return False
            if _git(self.checkout, "rev-parse", "HEAD").decode().strip() != commit:
                return False
            if (
                _git(self.checkout, "rev-parse", f"{commit}^{{commit}}")
                .decode()
                .strip()
                != commit
            ):
                return False
            seen: set[str] = set()
            for binding in required_files:
                if not isinstance(binding, dict):
                    return False
                relative = binding.get("path")
                sha256 = binding.get("sha256")
                byte_count = binding.get("byte_count")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or relative in seen
                    or not isinstance(sha256, str)
                    or _SHA256.fullmatch(sha256) is None
                    or type(byte_count) is not int
                    or byte_count <= 0
                ):
                    return False
                seen.add(relative)
                path = self.checkout / relative
                working = _file_bytes(path, f"ReScene source {relative}")
                frozen = _git(self.checkout, "show", f"{commit}:{relative}")
                if (
                    working != frozen
                    or len(frozen) != byte_count
                    or hashlib.sha256(frozen).hexdigest() != sha256
                ):
                    return False
            if "conf/backbone/concerto.yaml" not in seen:
                return False
            checkpoint = manifest.get("checkpoint")
            return isinstance(checkpoint, dict) and checkpoint.get(
                "selected_backbone"
            ) == "Concerto"
        except (
            KeyError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
            ReSceneBackendError,
        ):
            return False

    def infer(self, pair: NeuralSampleMap) -> TemporalQueryEvidence:
        if not isinstance(pair, NeuralSampleMap):
            raise TypeError("pair must be a NeuralSampleMap")
        if not self._validate_source():
            return self._blocked(
                pair, "BLOCKED_EXTERNAL_SOURCE", "source_identity_mismatch"
            )
        if self.checkpoint.is_symlink() or not self.checkpoint.is_file():
            return self._blocked(
                pair,
                "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
                "checkpoint_missing",
            )
        if (
            not isinstance(self.checkpoint_sha256, str)
            or _SHA256.fullmatch(self.checkpoint_sha256) is None
        ):
            raise ReSceneBackendError(
                "checkpoint SHA-256 binding must be 64 lowercase hexadecimal digits"
            )
        checkpoint = _file_bytes(self.checkpoint, "ReScene checkpoint")
        if hashlib.sha256(checkpoint).hexdigest() != self.checkpoint_sha256:
            raise ReSceneBackendError("checkpoint SHA-256 mismatch")
        return self._blocked(
            pair, "BLOCKED_EXTERNAL_SOURCE", "backend_executor_unavailable"
        )
