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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.compare_oviv2_cumulative_artifacts import (
    ArtifactMismatch,
    compare_cumulative_artifacts,
)


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


EXACT_PROFILE_SEQUENCE = (
    "reference",
    "a0",
    "a1",
    "a0",
    "a2",
    "a0",
    "a3",
    "a0",
    "a4",
)
EXACT_EXECUTION_FIELDS = {
    "profile",
    "argv",
    "pid",
    "code_commit",
    "source_manifest_sha256",
    "input_fingerprints",
    "output_root",
    "receipt_sha256",
}


def _exact_execution(record: object) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != EXACT_EXECUTION_FIELDS:
        raise GateVerificationError("exact execution binding fields are invalid")
    argv = record.get("argv")
    pid = record.get("pid")
    commit = record.get("code_commit")
    source = record.get("source_manifest_sha256")
    inputs = record.get("input_fingerprints")
    output = record.get("output_root")
    receipt_sha256 = record.get("receipt_sha256")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise GateVerificationError("exact execution argv is invalid")
    if type(pid) is not int or pid <= 0:
        raise GateVerificationError("exact execution PID is invalid")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(source, str)
        or len(source) != 64
        or any(character not in "0123456789abcdef" for character in source)
    ):
        raise GateVerificationError("exact execution binding is invalid")
    if not isinstance(inputs, dict) or not inputs or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for key, value in inputs.items()
    ):
        raise GateVerificationError("exact execution input binding is invalid")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise GateVerificationError("exact execution output binding is invalid")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise GateVerificationError("exact execution receipt binding is invalid")
    return dict(record)


def _validate_exact_argv(record: Mapping[str, Any], root: Path) -> None:
    profile = str(record["profile"])
    argv = record["argv"]
    expected_runner = (
        REPO_ROOT / "scripts/evaluation/run_oviv2_t1_reference.py"
        if profile == "reference"
        else REPO_ROOT / "scripts/evaluation/run_oviv2_tesse_cd_v2.py"
    ).resolve()
    expected_flags = ("--config", "--output", "--freeze-manifest", "--run-slot")
    expected_length = 14 if profile == "reference" else 10
    if (
        len(argv) != expected_length
        or not Path(argv[0]).is_absolute()
        or Path(argv[1]) != expected_runner
        or tuple(argv[2:10:2]) != expected_flags
        or argv[5] != str(root)
        or any(not Path(argv[index]).is_absolute() for index in (3, 5, 7))
        or argv[9] not in {
            "apartment_run1", "apartment_run2", "office_run1", "office_run2"
        }
    ):
        raise GateVerificationError("exact execution argv is not canonical")
    if profile == "reference" and (
        tuple(argv[10:14:2]) != ("--receipt", "--source-manifest")
        or argv[11] != str(root / "t1_exact_receipt.json")
        or not Path(argv[13]).is_absolute()
    ):
        raise GateVerificationError("reference execution argv is not canonical")


def _bind_exact_receipt(
    record: dict[str, Any],
    *,
    compare: Callable[[Path, Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_root = record["output_root"]
    try:
        root = Path(raw_root).resolve(strict=True)
    except OSError as exc:
        raise GateVerificationError("exact execution root is missing") from exc
    if str(root) != raw_root or not root.is_dir():
        raise GateVerificationError("exact execution root is not canonical")
    _validate_exact_argv(record, root)
    receipt_path = root / "t1_exact_receipt.json"
    try:
        receipt_data = _regular_file_bytes(
            root, receipt_path.name, DEFAULT_MAX_INPUT_BYTES
        )
        receipt = json.loads(
            receipt_data.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateVerificationError("exact execution receipt is invalid") from exc
    if _sha256(receipt_data) != record["receipt_sha256"]:
        raise GateVerificationError("exact execution receipt hash mismatch")
    expected_execution = {
        key: value for key, value in record.items() if key != "receipt_sha256"
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {
            "schema_version", "format", "execution", "artifact_inventory",
            "checkpoint_frames", "cumulative_root_sha256", "source_manifest",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("format") != "oviv2_t1_exact_execution_receipt_v1"
        or receipt.get("execution") != expected_execution
    ):
        raise GateVerificationError("exact execution receipt binding mismatch")
    source_record = receipt.get("source_manifest")
    if not isinstance(source_record, dict):
        raise GateVerificationError("exact execution source manifest binding mismatch")
    source_path = Path(
        record["argv"][13]
        if record["profile"] == "reference"
        else source_record.get("path", "")
    )
    try:
        canonical_source = source_path.resolve(strict=True)
    except OSError as exc:
        raise GateVerificationError("exact execution source manifest is missing") from exc
    if (
        set(source_record) != {"path", "sha256", "byte_count"}
        or str(canonical_source) != str(source_path)
        or source_record.get("path") != str(source_path)
        or source_record.get("sha256") != record["source_manifest_sha256"]
        or type(source_record.get("byte_count")) is not int
        or source_record["byte_count"] < 0
    ):
        raise GateVerificationError("exact execution source manifest binding mismatch")
    source_data = _regular_file_bytes(
        canonical_source.parent, canonical_source.name, DEFAULT_MAX_INPUT_BYTES
    )
    if (
        len(source_data) != source_record["byte_count"]
        or _sha256(source_data) != source_record["sha256"]
    ):
        raise GateVerificationError("exact execution source manifest hash mismatch")
    try:
        audit = compare(root, root)
    except (ArtifactMismatch, KeyError, TypeError) as exc:
        raise GateVerificationError(f"exact execution artifact mismatch: {exc}") from exc
    if (
        receipt.get("artifact_inventory") != audit.get("inventory")
        or receipt.get("checkpoint_frames") != audit.get("checkpoint_frames")
        or receipt.get("cumulative_root_sha256") != audit.get("root_sha256")
    ):
        raise GateVerificationError("exact execution receipt artifact mismatch")
    return record, audit


def verify_exact_profile_runs(
    executions: list[dict[str, Any]],
    *,
    compare: Callable[[Path, Path], dict[str, Any]] = compare_cumulative_artifacts,
) -> dict[str, Any]:
    """Verify an independently executed, interleaved T1/A0-A4 transaction."""
    records_and_audits = [
        _bind_exact_receipt(_exact_execution(record), compare=compare)
        for record in executions
    ]
    records = [record for record, _ in records_and_audits]
    sequence = tuple(record["profile"] for record in records)
    if sequence != EXACT_PROFILE_SEQUENCE:
        raise GateVerificationError("exact execution profile sequence is invalid")
    if len({record["pid"] for record in records}) != len(records):
        raise GateVerificationError("exact executions must use independent PIDs")
    roots = [Path(record["output_root"]) for record in records]
    if len(set(roots)) != len(roots) or any(
        left in right.parents or right in left.parents
        for position, left in enumerate(roots)
        for right in roots[position + 1 :]
    ):
        raise GateVerificationError("exact execution roots must be independent")
    binding = (
        records[0]["code_commit"],
        records[0]["source_manifest_sha256"],
        records[0]["input_fingerprints"],
    )
    if any(
        (
            record["code_commit"],
            record["source_manifest_sha256"],
            record["input_fingerprints"],
        )
        != binding
        for record in records[1:]
    ):
        raise GateVerificationError("exact execution binding disagreement")

    reference_root = Path(records[0]["output_root"])
    profiles: dict[str, Any] = {}
    try:
        for anchor_index, candidate_index in ((1, 2), (3, 4), (5, 6), (7, 8)):
            anchor = records[anchor_index]
            candidate = records[candidate_index]
            reference_audit = compare(
                reference_root, Path(anchor["output_root"])
            )
            profile_audit = compare(
                Path(anchor["output_root"]), Path(candidate["output_root"])
            )
            if reference_audit["root_sha256"] != profile_audit["root_sha256"]:
                raise GateVerificationError("cumulative audit root disagreement")
            profiles["a0"] = {
                "cumulative_root_sha256": reference_audit["root_sha256"],
                "checkpoint_frames": reference_audit["checkpoint_frames"],
                "inventory": reference_audit["inventory"],
            }
            profiles[candidate["profile"]] = {
                "cumulative_root_sha256": profile_audit["root_sha256"],
                "checkpoint_frames": profile_audit["checkpoint_frames"],
                "inventory": profile_audit["inventory"],
            }
    except (ArtifactMismatch, KeyError, TypeError) as exc:
        raise GateVerificationError(f"exact cumulative artifact mismatch: {exc}") from exc
    return {
        "format": "oviv2_t1_exact_transaction_v1",
        "sequence": list(sequence),
        "executions": records,
        "profiles": profiles,
    }


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


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(item) for item in target.elts))
    return set()


def _top_level_exports(payload: bytes, *, path: str) -> tuple[set[str], set[str] | None]:
    try:
        tree = ast.parse(payload, filename=path)
    except (SyntaxError, ValueError) as error:
        raise GateVerificationError(f"cannot parse source imports: {path}") from error
    exports: set[str] = set()
    explicit_all: set[str] | None = None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                exports.update(_assigned_names(target))
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)) and all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in node.value.elts
                ):
                    explicit_all = {item.value for item in node.value.elts}
        elif isinstance(node, ast.AnnAssign):
            exports.update(_assigned_names(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                exports.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    exports.add(alias.asname or alias.name)
    return exports, explicit_all


def _dynamic_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    functions = {"__import__"}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    modules.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    functions.add(alias.asname or alias.name)
    return functions, modules


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
        dynamic_functions, importlib_modules = _dynamic_import_aliases(tree)
        package = module if is_package else module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                recognized = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in dynamic_functions
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in importlib_modules
                )
                if not recognized:
                    continue
                if (
                    not node.args
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)
                ):
                    raise GateVerificationError(
                        f"non-literal dynamic import in {relative}"
                    )
                target = node.args[0].value
                if target.startswith("."):
                    raise GateVerificationError(
                        f"unresolved dynamic import in {relative}: {target}"
                    )
                if target.split(".", 1)[0] in LOCAL_IMPORT_PREFIXES:
                    resolve(target, required=True)
            elif isinstance(node, ast.Import):
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
                base_exports: set[str] = set()
                explicit_all: set[str] | None = None
                if base_found is not None:
                    base_payload = read(base_found)
                    if base_payload is None:
                        raise GateVerificationError(
                            f"source disappeared while resolving import: {base}"
                        )
                    base_exports, explicit_all = _top_level_exports(
                        base_payload, path=base_found
                    )
                for alias in node.names:
                    if alias.name == "*":
                        if (
                            base_found is None
                            or explicit_all is None
                            or not explicit_all <= base_exports
                        ):
                            raise GateVerificationError(
                                f"unresolved local import: {base}.*"
                            )
                        continue
                    child_found = resolve(
                        f"{base}.{alias.name}", required=False
                    )
                    if child_found is None and (
                        base_found is None or alias.name not in base_exports
                    ):
                        raise GateVerificationError(
                            f"unresolved local import: {base}.{alias.name}"
                        )
    return dict(sorted(closure.items()))


def _git_source_reader(
    *, repo: Path, base_commit: str, git: GitRunner
) -> Callable[[str], bytes | None]:
    cache: dict[str, bytes | None] = {}
    raw_paths = git(("ls-tree", "-rz", "--name-only", base_commit), repo)
    try:
        paths = {
            item.decode("utf-8")
            for item in raw_paths.split(b"\0")
            if item
        }
    except UnicodeDecodeError as error:
        raise GateVerificationError("git tree contains a non-UTF-8 path") from error

    def read(relative: str) -> bytes | None:
        if relative not in paths:
            return None
        if relative not in cache:
            cache[relative] = git(("show", f"{base_commit}:{relative}"), repo)
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
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["manifest_id"] != SOURCE_MANIFEST_ID
    ):
        raise GateVerificationError("source manifest identity is invalid")
    if manifest["base_commit"] != CUMULATIVE_BASE_COMMIT:
        raise GateVerificationError("source manifest cumulative base is invalid")
    roots_value = manifest["roots"]
    files_value = manifest["files"]
    if not isinstance(roots_value, list) or not all(isinstance(item, str) for item in roots_value):
        raise GateVerificationError("source manifest roots are invalid")
    if not all(_is_canonical_repo_path(path) for path in roots_value):
        raise GateVerificationError(
            "source manifest paths must be canonical relative POSIX paths"
        )
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
    if not all(_is_canonical_repo_path(path) for path in files_value):
        raise GateVerificationError(
            "source manifest paths must be canonical relative POSIX paths"
        )
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
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateVerificationError(f"cannot read source manifest: {path}") from error
    if not isinstance(payload, dict):
        raise GateVerificationError("source manifest root must be an object")
    return payload


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_canonical_repo_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path or "\0" in path:
        return False
    parts = path.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _regular_file_bytes(repo: Path, relative: str, maximum: int) -> bytes:
    if not _is_canonical_repo_path(relative):
        raise GateVerificationError(
            f"required file path must be canonical relative POSIX: {relative!r}"
        )
    parts = relative.split("/")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_descriptor = os.open(repo, directory_flags)
    except FileNotFoundError as error:
        raise GateVerificationError(f"repository root is missing: {repo}") from error
    except OSError as error:
        raise GateVerificationError(
            f"repository root must be a regular non-symlink directory: {repo}"
        ) from error
    directory_descriptor = root_descriptor
    descriptor: int | None = None
    try:
        for part in parts[:-1]:
            try:
                next_descriptor = os.open(
                    part, directory_flags, dir_fd=directory_descriptor
                )
            except FileNotFoundError as error:
                raise GateVerificationError(
                    f"required file is missing: {relative}"
                ) from error
            except OSError as error:
                raise GateVerificationError(
                    f"required file ancestors must be regular non-symlink directories: {relative}"
                ) from error
            if directory_descriptor != root_descriptor:
                os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        try:
            descriptor = os.open(
                parts[-1], file_flags, dir_fd=directory_descriptor
            )
        except FileNotFoundError as error:
            raise GateVerificationError(
                f"required file is missing: {relative}"
            ) from error
        except OSError as error:
            raise GateVerificationError(
                f"required file must be a regular non-symlink: {relative}"
            ) from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GateVerificationError(
                f"required file must be a regular non-symlink: {relative}"
            )
        if before.st_size > maximum:
            raise GateVerificationError(
                f"required file exceeds maximum {maximum} bytes: {relative}"
            )
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
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor != root_descriptor:
            os.close(directory_descriptor)
        os.close(root_descriptor)
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
    encoded = _canonical_json_bytes(payload)
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


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_json_replace(
    path: Path,
    payload: dict[str, Any],
    validate_before_replace: Callable[[], None],
) -> None:
    encoded = _canonical_json_bytes(payload)
    destination_mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    directory_descriptor: int | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), destination_mode)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        validate_before_replace()
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)
        directory_descriptor = None
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        temporary.unlink(missing_ok=True)


def _checked_manifest_destination(repo: Path, destination: Path) -> tuple[Path, str]:
    raw = destination if destination.is_absolute() else repo / destination
    if raw.is_symlink():
        raise GateVerificationError("source manifest destination must not be a symlink")
    try:
        lexical_relative = raw.relative_to(repo)
    except ValueError as error:
        raise GateVerificationError("source manifest destination must be inside repo") from error
    current = repo
    for part in lexical_relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise GateVerificationError(
                "source manifest destination ancestor must not be a symlink"
            )
    resolved = raw.resolve(strict=False)
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError as error:
        raise GateVerificationError("source manifest destination must be inside repo") from error
    return resolved, relative


def write_source_manifest(
    *,
    repo: Path,
    destination: Path,
    payload: dict[str, Any],
    git: GitRunner = _default_git,
) -> Path:
    repo = repo.resolve(strict=True)
    destination, relative = _checked_manifest_destination(repo, destination)
    if git(("status", "--porcelain", "--", relative), repo):
        raise GateVerificationError("source manifest destination is dirty")
    encoded = _canonical_json_bytes(payload)
    if destination.exists():
        current = _regular_file_bytes(
            destination.parent, destination.name, DEFAULT_MAX_INPUT_BYTES
        )
        if current == encoded:
            return destination

        def validate_before_replace() -> None:
            if git(("status", "--porcelain", "--", relative), repo):
                raise GateVerificationError("source manifest destination became dirty")
            if (
                _regular_file_bytes(
                    destination.parent, destination.name, DEFAULT_MAX_INPUT_BYTES
                )
                != current
            ):
                raise GateVerificationError(
                    "source manifest destination changed before replacement"
                )

        _atomic_json_replace(destination, payload, validate_before_replace)
        return destination
    _atomic_json_no_replace(destination, payload)
    return destination


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
    exact_profile_runs: list[dict[str, Any]] | None = None,
    compare: Callable[[Path, Path], dict[str, Any]] = compare_cumulative_artifacts,
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
    source_manifest_bytes = _regular_file_bytes(
        repo, DEFAULT_SOURCE_MANIFEST.as_posix(), max_input_bytes
    )
    source_manifest_record = {
        "path": str((repo / DEFAULT_SOURCE_MANIFEST).resolve()),
        "sha256": _sha256(source_manifest_bytes),
        "byte_count": len(source_manifest_bytes),
    }
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
    if exact_profile_runs is None:
        raise GateVerificationError("exact profile run transaction is required")
    cumulative_exact = verify_exact_profile_runs(
        exact_profile_runs, compare=compare
    )

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
        "source_manifest": source_manifest_record,
        "cumulative_exact": cumulative_exact,
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
    parser.add_argument("--exact-profile-runs", type=Path)
    args = parser.parse_args()
    try:
        repo = args.repo.resolve(strict=True)
        if args.write_source_manifest is not None:
            manifest = build_source_manifest(repo=repo, base_commit=args.base_commit)
            write_source_manifest(
                repo=repo,
                destination=args.write_source_manifest,
                payload=manifest,
            )
        elif args.verify_source_manifest is not None:
            source = args.verify_source_manifest
            if not source.is_absolute():
                source = repo / source
            verify_source_manifest(_load_source_manifest(source), repo=repo)
        else:
            if args.exact_profile_runs is None:
                raise GateVerificationError(
                    "--exact-profile-runs is required for development evidence"
                )
            executions = _load_source_manifest(args.exact_profile_runs)
            if not isinstance(executions, dict) or set(executions) != {"executions"} or not isinstance(executions["executions"], list):
                raise GateVerificationError("exact profile run transaction is invalid")
            generate_evidence(
                repo=repo,
                output=args.output,
                base_commit=args.base_commit,
                python_executable=sys.executable,
                exact_profile_runs=executions["executions"],
            )
    except (GateVerificationError, FileExistsError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
