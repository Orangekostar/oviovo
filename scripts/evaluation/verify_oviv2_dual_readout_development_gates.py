#!/usr/bin/env python3
"""Produce shared T1-exact and determinism development-gate evidence."""

from __future__ import annotations

import argparse
import ast
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


CUMULATIVE_BASE_COMMIT = "8034e79d9cb853166222610981a7e6893f6cca70"
SCOPE = "shared_code_and_A0-A4_fixture"
SCHEMA_VERSION = 1
MANIFEST_ID = "oviv2_dual_readout_development_gates_v1"
DEFAULT_MAX_INPUT_BYTES = 16 * 1024 * 1024

CUMULATIVE_ROOTS = (
    "src/oviv2/runtime.py",
    "src/oviv2/runner_config.py",
    "scripts/run_oviv2_replica.py",
    "scripts/evaluation/run_oviv2_tesse_cd.py",
    "configs/oviv2_tesse_cd_apartment_v1.json",
    "configs/oviv2_tesse_cd_office_v1.json",
)
DEFAULT_SOURCE_MANIFEST = Path(
    "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
)
SOURCE_MANIFEST_ID = "oviv2_t1_transitive_sources_v1"
LOCAL_IMPORT_PREFIXES = ("src", "scripts", "configs")

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
    environment = os.environ.copy()
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_PLUGINS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _module_candidates(module: str) -> tuple[str, str]:
    stem = module.replace(".", "/")
    return f"{stem}.py", f"{stem}/__init__.py"


def _module_for_path(path: str) -> tuple[str, bool]:
    if path.endswith("/__init__.py"):
        return path[: -len("/__init__.py")].replace("/", "."), True
    if path.endswith(".py"):
        return path[:-3].replace("/", "."), False
    return "", False


def _local_import_closure(
    roots: tuple[str, ...], read: Callable[[str], bytes | None]
) -> dict[str, bytes]:
    pending = list(roots)
    closure: dict[str, bytes] = {}

    def resolve(module: str, *, required: bool) -> str | None:
        if not module or module.split(".", 1)[0] not in LOCAL_IMPORT_PREFIXES:
            return None
        for candidate in _module_candidates(module):
            payload = read(candidate)
            if payload is not None:
                if candidate not in closure and candidate not in pending:
                    pending.append(candidate)
                return candidate
        if required:
            raise GateVerificationError(f"unresolved local import: {module}")
        return None

    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        payload = read(relative)
        if payload is None:
            raise GateVerificationError(f"required source is missing: {relative}")
        closure[relative] = payload
        module, is_package = _module_for_path(relative)
        if not module:
            continue
        parts = module.split(".")
        for index in range(1, len(parts) if is_package else len(parts)):
            resolve(".".join(parts[:index]), required=False)
        try:
            tree = ast.parse(payload, filename=relative)
        except (SyntaxError, ValueError) as error:
            raise GateVerificationError(f"cannot parse source imports: {relative}") from error
        package = module if is_package else module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name, required=True)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".") if package else []
                    if node.level > len(package_parts):
                        raise GateVerificationError(
                            f"unresolved local import in {relative}"
                        )
                    prefix = package_parts[: len(package_parts) - node.level + 1]
                    base = ".".join((*prefix, *(node.module or "").split("."))).strip(".")
                else:
                    base = node.module or ""
                if not base or base.split(".", 1)[0] not in LOCAL_IMPORT_PREFIXES:
                    continue
                base_found = resolve(base, required=False)
                child_found = False
                for alias in node.names:
                    if alias.name != "*":
                        child_found = resolve(
                            f"{base}.{alias.name}", required=False
                        ) is not None or child_found
                if base_found is None and not child_found:
                    raise GateVerificationError(f"unresolved local import: {base}")
    return dict(sorted(closure.items()))


def _git_source_reader(
    *, repo: Path, base_commit: str, git: GitRunner
) -> Callable[[str], bytes | None]:
    cache: dict[str, bytes | None] = {}

    def read(relative: str) -> bytes | None:
        if relative not in cache:
            try:
                cache[relative] = git(("show", f"{base_commit}:{relative}"), repo)
            except (GateVerificationError, subprocess.CalledProcessError):
                cache[relative] = None
        return cache[relative]

    return read


def _worktree_source_reader(
    *, repo: Path, maximum: int
) -> Callable[[str], bytes | None]:
    def read(relative: str) -> bytes | None:
        path = repo / relative
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        return _regular_file_bytes(repo, relative, maximum)

    return read


def build_source_manifest(
    *,
    repo: Path,
    base_commit: str,
    roots: tuple[str, ...] = CUMULATIVE_ROOTS,
    git: GitRunner = _default_git,
) -> dict[str, Any]:
    if base_commit != CUMULATIVE_BASE_COMMIT:
        raise GateVerificationError(
            f"base commit must equal cumulative base {CUMULATIVE_BASE_COMMIT}"
        )
    closure = _local_import_closure(
        roots, _git_source_reader(repo=repo, base_commit=base_commit, git=git)
    )
    return {
        "schema_version": 1,
        "manifest_id": SOURCE_MANIFEST_ID,
        "base_commit": base_commit,
        "roots": list(roots),
        "files": {path: _sha256(payload) for path, payload in closure.items()},
    }


def verify_source_manifest(
    manifest: dict[str, Any],
    *,
    repo: Path,
    git: GitRunner = _default_git,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> list[dict[str, Any]]:
    if set(manifest) != {"schema_version", "manifest_id", "base_commit", "roots", "files"}:
        raise GateVerificationError("source manifest has invalid fields")
    if manifest["schema_version"] != 1 or manifest["manifest_id"] != SOURCE_MANIFEST_ID:
        raise GateVerificationError("source manifest identity is invalid")
    if manifest["base_commit"] != CUMULATIVE_BASE_COMMIT:
        raise GateVerificationError("source manifest cumulative base is invalid")
    roots_value = manifest["roots"]
    files_value = manifest["files"]
    if not isinstance(roots_value, list) or not all(isinstance(item, str) for item in roots_value):
        raise GateVerificationError("source manifest roots are invalid")
    if tuple(roots_value) != CUMULATIVE_ROOTS:
        raise GateVerificationError("source manifest cumulative roots are invalid")
    if not isinstance(files_value, dict) or not all(
        isinstance(path, str)
        and isinstance(digest, str)
        and len(digest) == 64
        and set(digest) <= set("0123456789abcdef")
        for path, digest in files_value.items()
    ):
        raise GateVerificationError("source manifest hashes are invalid")
    for path in files_value:
        current_path = repo / path
        try:
            metadata = current_path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise GateVerificationError(
                f"required file must be a regular non-symlink: {path}"
            )
        if metadata.st_size > max_input_bytes:
            raise GateVerificationError(
                f"required file exceeds maximum {max_input_bytes} bytes: {path}"
            )
    roots = tuple(roots_value)
    trusted = build_source_manifest(
        repo=repo,
        base_commit=CUMULATIVE_BASE_COMMIT,
        roots=roots,
        git=git,
    )
    trusted_files = trusted["files"]
    if set(files_value) != set(trusted_files):
        raise GateVerificationError("trusted source path set does not match manifest")
    if files_value != trusted_files:
        raise GateVerificationError("trusted source hash does not match manifest")
    current = _local_import_closure(
        roots, _worktree_source_reader(repo=repo, maximum=max_input_bytes)
    )
    if set(current) != set(files_value):
        raise GateVerificationError("current source path set does not match manifest")
    records = []
    for path, payload in current.items():
        digest = _sha256(payload)
        if digest != files_value[path]:
            raise GateVerificationError(f"current source hash does not match: {path}")
        records.append({"path": path, "sha256": digest, "bytes": len(payload)})
    return records


def _load_source_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateVerificationError(f"cannot read source manifest: {path}") from error
    if not isinstance(payload, dict):
        raise GateVerificationError("source manifest root must be an object")
    return payload


def _regular_file_bytes(repo: Path, relative: str, maximum: int) -> bytes:
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
    return b"".join(chunks)


def _regular_file_record(repo: Path, relative: str, maximum: int) -> dict[str, Any]:
    payload = _regular_file_bytes(repo, relative, maximum)
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
    forbidden = (
        " skipped",
        " xfailed",
        " xpassed",
        " deselected",
        "skip=",
        "xfail=",
        "xpass=",
    )
    progress_has_nonpass = False
    for line in combined.splitlines():
        prefix = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if prefix and set(prefix) <= set(".efxs") and any(
            marker in prefix for marker in "sx"
        ):
            progress_has_nonpass = True
            break
    if any(token in combined for token in forbidden) or progress_has_nonpass:
        raise GateVerificationError(
            "test command reported skip, xfail/xpass, or deselected outcomes"
        )
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
        os.close(directory_descriptor)
        directory_descriptor = None
        temporary.unlink()
    except BaseException:
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        if linked:
            for _ in range(2):
                try:
                    path.unlink(missing_ok=True)
                    break
                except OSError:
                    continue
        for _ in range(2):
            try:
                temporary.unlink(missing_ok=True)
                break
            except OSError:
                continue
        raise


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
    if base_commit != CUMULATIVE_BASE_COMMIT:
        raise GateVerificationError(
            f"base commit must equal cumulative base {CUMULATIVE_BASE_COMMIT}"
        )
    if output.is_symlink() or output.exists():
        raise FileExistsError(output)

    code_commit, code_tree = _git_state(repo, git)
    source_manifest = _load_source_manifest(repo / DEFAULT_SOURCE_MANIFEST)
    protected = verify_source_manifest(
        source_manifest, repo=repo, git=git, max_input_bytes=max_input_bytes
    )
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
    final_protected = verify_source_manifest(
        source_manifest, repo=repo, git=git, max_input_bytes=max_input_bytes
    )
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
        publish_protected = verify_source_manifest(
            source_manifest, repo=repo, git=git, max_input_bytes=max_input_bytes
        )
        if (publish_commit, publish_tree) != (code_commit, code_tree):
            raise GateVerificationError("git commit or tree changed before publication")
        if (
            publish_protected != protected
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--write-source-manifest", type=Path)
    mode.add_argument("--verify-source-manifest", type=Path)
    parser.add_argument("--base-commit", default=CUMULATIVE_BASE_COMMIT)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        repo = args.repo.resolve(strict=True)
        if args.write_source_manifest is not None:
            destination = args.write_source_manifest
            if not destination.is_absolute():
                destination = repo / destination
            relative = destination.relative_to(repo).as_posix()
            if _default_git(("status", "--porcelain", "--", relative), repo):
                raise GateVerificationError("source manifest destination is dirty")
            manifest = build_source_manifest(repo=repo, base_commit=args.base_commit)
            _atomic_json_no_replace(destination, manifest)
        elif args.verify_source_manifest is not None:
            source = args.verify_source_manifest
            if not source.is_absolute():
                source = repo / source
            verify_source_manifest(_load_source_manifest(source), repo=repo)
        else:
            generate_evidence(
                repo=repo,
                output=args.output,
                base_commit=args.base_commit,
                python_executable=sys.executable,
            )
    except (GateVerificationError, FileExistsError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
