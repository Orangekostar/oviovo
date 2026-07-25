#!/usr/bin/env python3
"""Produce shared T1-exact and determinism development-gate evidence."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


PLANNED_BASE_COMMIT = "e556767cc3750ff592ff29c288851b33e2089c8d"
SCOPE = "shared_code_and_A0-A4_fixture"
SCHEMA_VERSION = 1
MANIFEST_ID = "oviv2_dual_readout_development_gates_v1"
DEFAULT_MAX_INPUT_BYTES = 16 * 1024 * 1024

PROTECTED_FILES = (
    "src/oviv2/runtime.py",
    "src/oviv2/runner_config.py",
    "scripts/run_oviv2_replica.py",
    "scripts/evaluation/run_oviv2_tesse_cd.py",
    "configs/oviv2_tesse_cd_apartment_v1.json",
    "configs/oviv2_tesse_cd_office_v1.json",
)

T1_TEST_FILES = ("tests/oviv2/test_t1_noninterference.py",)
DETERMINISM_TEST_FILES = (
    "tests/oviv2/test_temporal_config.py",
    "tests/oviv2/test_temporal_lifecycle.py",
    "tests/oviv2/test_temporal_association.py",
    "tests/oviv2/test_temporal_geometry.py",
    "tests/oviv2/test_temporal_background.py",
    "tests/oviv2/test_temporal_runtime.py",
    "tests/oviv2/test_temporal_snapshot.py",
    "tests/oviv2/test_dual_readout.py",
    "tests/oviv2/test_reference_readout.py",
    "tests/evaluation/test_oviv2_temporal_tesse.py",
    "tests/evaluation/test_run_oviv2_tesse_cd_v2.py",
)
TEST_FILES = tuple(dict.fromkeys((*T1_TEST_FILES, *DETERMINISM_TEST_FILES)))
SMOKE_TEST = "tests/oviv2/test_temporal_config.py::test_serialization_is_deterministic"

GitRunner = Callable[[tuple[str, ...], Path], bytes]
CommandRunner = Callable[
    [tuple[str, ...], Path], subprocess.CompletedProcess[bytes]
]


class GateVerificationError(RuntimeError):
    """Raised when evidence cannot be established without ambiguity."""


def _default_git(argv: tuple[str, ...], cwd: Path) -> bytes:
    try:
        return subprocess.run(
            ("git", *argv),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise GateVerificationError(
            f"git {' '.join(argv)} failed: {detail or error.returncode}"
        ) from error


def _default_run(
    argv: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_file_record(repo: Path, relative: str, maximum: int) -> dict[str, Any]:
    path = repo / relative
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise GateVerificationError(f"required file is missing: {relative}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise GateVerificationError(f"required file must be a regular non-symlink: {relative}")
    if before.st_size > maximum:
        raise GateVerificationError(
            f"required file exceeds maximum {maximum} bytes: {relative}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise GateVerificationError(f"required file changed while opening: {relative}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise GateVerificationError(
                    f"required file exceeds maximum {maximum} bytes: {relative}"
                )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise GateVerificationError(f"required file changed while hashing: {relative}")
    payload = b"".join(chunks)
    return {"path": relative, "sha256": _sha256(payload), "bytes": len(payload)}


def _snapshot_files(
    repo: Path, paths: tuple[str, ...], maximum: int
) -> list[dict[str, Any]]:
    return [_regular_file_record(repo, path, maximum) for path in paths]


def _git_state(repo: Path, git: GitRunner) -> tuple[str, str]:
    if git(("status", "--porcelain", "--untracked-files=all"), repo):
        raise GateVerificationError("repository must be clean")
    commit = git(("rev-parse", "HEAD"), repo).decode("ascii").strip()
    tree = git(("rev-parse", "HEAD^{tree}"), repo).decode("ascii").strip()
    if len(commit) != 40 or len(tree) != 40:
        raise GateVerificationError("git returned an invalid commit or tree identity")
    return commit, tree


def _assert_protected_diff(repo: Path, base_commit: str, git: GitRunner) -> None:
    try:
        git(("diff", "--quiet", base_commit, "--", *PROTECTED_FILES), repo)
    except (subprocess.CalledProcessError, GateVerificationError) as error:
        raise GateVerificationError("protected files differ from planned base") from error


def _command_record(
    *, repo: Path, argv: tuple[str, ...], run: CommandRunner
) -> dict[str, Any]:
    completed = run(argv, repo)
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    if isinstance(stdout, str) or isinstance(stderr, str):
        raise GateVerificationError("test runner must capture stdout and stderr as bytes")
    combined = (stdout + b"\n" + stderr).decode("utf-8", errors="replace").lower()
    if completed.returncode != 0:
        raise GateVerificationError(f"test command failed with return code {completed.returncode}")
    forbidden = (" skipped", " xfailed", " xpassed", "skip=", "xfail=", "xpass=")
    progress_has_nonpass = False
    for line in combined.splitlines():
        prefix = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if prefix and set(prefix) <= set(".efxs") and any(
            marker in prefix for marker in "sx"
        ):
            progress_has_nonpass = True
            break
    if any(token in combined for token in forbidden) or progress_has_nonpass:
        raise GateVerificationError("test command reported skip or xfail/xpass outcomes")
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout_sha256": _sha256(stdout),
        "stdout_bytes": len(stdout),
        "stderr_sha256": _sha256(stderr),
        "stderr_bytes": len(stderr),
    }


def _atomic_json_no_replace(
    path: Path,
    payload: dict[str, Any],
    validate_before_publish: Callable[[], None] = lambda: None,
) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise GateVerificationError("output parent must be a regular directory")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    validate_before_publish()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    directory_descriptor: int | None = None
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(directory_descriptor)
    except BaseException:
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        temporary.unlink(missing_ok=True)


def generate_evidence(
    *,
    repo: Path,
    output: Path,
    base_commit: str,
    python_executable: str,
    git: GitRunner = _default_git,
    run: CommandRunner = _default_run,
    now_utc: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    ),
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> Path:
    repo = repo.resolve(strict=True)
    if base_commit != PLANNED_BASE_COMMIT:
        raise GateVerificationError(
            f"base commit must equal planned base {PLANNED_BASE_COMMIT}"
        )
    if output.is_symlink() or output.exists():
        raise FileExistsError(output)

    code_commit, code_tree = _git_state(repo, git)
    _assert_protected_diff(repo, base_commit, git)
    protected = _snapshot_files(repo, PROTECTED_FILES, max_input_bytes)
    test_sources = _snapshot_files(repo, TEST_FILES, max_input_bytes)

    pytest_prefix = (
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-rA",
        "-o",
        "addopts=",
    )
    t1_argv = (*pytest_prefix, *T1_TEST_FILES)
    deterministic_argv = (
        *pytest_prefix,
        *DETERMINISM_TEST_FILES,
    )
    t1_record = _command_record(repo=repo, argv=t1_argv, run=run)
    determinism_record = _command_record(repo=repo, argv=deterministic_argv, run=run)

    final_commit, final_tree = _git_state(repo, git)
    _assert_protected_diff(repo, base_commit, git)
    final_protected = _snapshot_files(repo, PROTECTED_FILES, max_input_bytes)
    final_tests = _snapshot_files(repo, TEST_FILES, max_input_bytes)
    if (final_commit, final_tree) != (code_commit, code_tree):
        raise GateVerificationError("git commit or tree changed during verification")
    if final_protected != protected or final_tests != test_sources:
        raise GateVerificationError("protected or test source file changed during verification")

    common = {
        "scope": SCOPE,
        "status": "PASS",
        "code_commit": code_commit,
        "code_tree": code_tree,
        "protected_records": protected,
    }
    deterministic_evidence = {
        "base_commit": base_commit,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "protected_files": protected,
        "test_sources": test_sources,
        "gates": {
            "t1_exact": {**common, "test_records": [t1_record]},
            "determinism": {**common, "test_records": [determinism_record]},
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "deterministic_evidence": deterministic_evidence,
        "receipt": {"created_at_utc": now_utc()},
    }

    def validate_before_publish() -> None:
        publish_commit, publish_tree = _git_state(repo, git)
        _assert_protected_diff(repo, base_commit, git)
        if (publish_commit, publish_tree) != (code_commit, code_tree):
            raise GateVerificationError("git commit or tree changed before publication")
        if (
            _snapshot_files(repo, PROTECTED_FILES, max_input_bytes) != protected
            or _snapshot_files(repo, TEST_FILES, max_input_bytes) != test_sources
        ):
            raise GateVerificationError(
                "protected or test source file changed before publication"
            )

    _atomic_json_no_replace(output, payload, validate_before_publish)
    return output


def run_focused_smoke(
    *,
    repo: Path,
    python_executable: str = sys.executable,
    run: CommandRunner = _default_run,
) -> subprocess.CompletedProcess[bytes]:
    return run((python_executable, "-m", "pytest", "-q", SMOKE_TEST), repo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-commit", default=PLANNED_BASE_COMMIT)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        generate_evidence(
            repo=args.repo,
            output=args.output,
            base_commit=args.base_commit,
            python_executable=sys.executable,
        )
    except (GateVerificationError, FileExistsError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
