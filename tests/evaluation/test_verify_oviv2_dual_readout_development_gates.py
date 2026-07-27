from __future__ import annotations

import gc
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from scripts.evaluation import run_oviv2_t1_reference as reference_worker


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts/evaluation/verify_oviv2_dual_readout_development_gates.py"
)
SPEC = importlib.util.spec_from_file_location("development_gates", SCRIPT)
assert SPEC and SPEC.loader
gates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gates
SPEC.loader.exec_module(gates)

PRODUCTION_SCHEMA1_VARIANTS = {
    profile: "production" for profile in set(gates.EXACT_PROFILE_SEQUENCE)
}


def _make_repo(root: Path) -> None:
    for relative in (*gates.CUMULATIVE_ROOTS, *gates.TEST_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# " + relative + "\n").encode())
    manifest = {
        "schema_version": 1,
        "manifest_id": "oviv2_t1_transitive_sources_v1",
        "base_commit": gates.CUMULATIVE_BASE_COMMIT,
        "roots": list(gates.CUMULATIVE_ROOTS),
        "files": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in gates.CUMULATIVE_ROOTS
        },
    }
    manifest_path = root / gates.DEFAULT_SOURCE_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class FakeGit:
    def __init__(self, *, dirty: bool = False, protected_diff: bool = False) -> None:
        self.dirty = dirty
        self.protected_diff = protected_diff

    def __call__(self, argv: tuple[str, ...], cwd: Path) -> bytes:
        if argv == ("status", "--porcelain", "--untracked-files=all"):
            return b" M dirty\n" if self.dirty else b""
        if argv == ("rev-parse", "HEAD"):
            return ("a" * 40 + "\n").encode()
        if argv == ("rev-parse", "HEAD^{tree}"):
            return ("b" * 40 + "\n").encode()
        if argv[:3] == ("ls-tree", "-rz", "--name-only"):
            return b"\0".join(
                str(path.relative_to(cwd)).encode()
                for path in sorted(cwd.rglob("*"))
                if path.is_file()
            ) + b"\0"
        if argv[:2] == ("diff", "--quiet"):
            if self.protected_diff:
                raise subprocess.CalledProcessError(1, ("git", *argv))
            return b""
        if argv[0] == "show":
            relative = argv[1].split(":", 1)[1]
            path = cwd / relative
            if not path.is_file():
                raise gates.GateVerificationError(f"missing git object: {relative}")
            return path.read_bytes()
        raise AssertionError(argv)


class SourceGit:
    def __init__(
        self,
        objects: dict[str, bytes],
        *,
        dirty: bool = False,
        fail_show: set[str] | None = None,
    ) -> None:
        self.objects = objects
        self.dirty = dirty
        self.fail_show = fail_show or set()
        self.show_calls: list[str] = []

    def __call__(self, argv: tuple[str, ...], cwd: Path) -> bytes:
        del cwd
        if argv[:3] == ("ls-tree", "-rz", "--name-only"):
            return b"\0".join(path.encode() for path in sorted(self.objects)) + b"\0"
        if argv[0] == "show":
            relative = argv[1].split(":", 1)[1]
            self.show_calls.append(relative)
            if relative in self.fail_show:
                raise gates.GateVerificationError(
                    f"operational show failure: {relative}"
                )
            try:
                return self.objects[relative]
            except KeyError as error:
                raise gates.GateVerificationError(f"missing git object: {relative}") from error
        if argv[:3] == ("status", "--porcelain", "--"):
            return b" M destination\n" if self.dirty else b""
        raise AssertionError(argv)


def _source_fixture(tmp_path: Path) -> tuple[Path, SourceGit, tuple[str, ...]]:
    repo = tmp_path / "source-repo"
    objects = {
        "src/__init__.py": b"",
        "src/pkg/__init__.py": b"",
        "src/pkg/dep.py": b"VALUE = 1\n",
        **{relative: ("# " + relative + "\n").encode() for relative in gates.CUMULATIVE_ROOTS},
    }
    objects[gates.CUMULATIVE_ROOTS[0]] = b"from src.pkg import dep\n"
    for relative, payload in objects.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return repo, SourceGit(objects), gates.CUMULATIVE_ROOTS


def _valid_source_manifest(tmp_path: Path) -> tuple[dict[str, object], Path, SourceGit]:
    repo, git, roots = _source_fixture(tmp_path)
    manifest = gates.build_source_manifest(
        repo=repo,
        base_commit=gates.CUMULATIVE_BASE_COMMIT,
        roots=roots,
        git=git,
    )
    return manifest, repo, git


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_source_manifest_rejects_path_set_drift(tmp_path: Path, mutation: str) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    files = manifest["files"]
    assert isinstance(files, dict)
    if mutation == "missing":
        files.pop("src/pkg/dep.py")
    else:
        files["src/pkg/extra.py"] = "0" * 64
    with pytest.raises(gates.GateVerificationError, match="source path set"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


def test_source_manifest_rejects_symlinked_dependency(tmp_path: Path) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    dependency = repo / "src/pkg/dep.py"
    dependency.unlink()
    dependency.symlink_to(repo / "src/pkg/root.py")
    with pytest.raises(gates.GateVerificationError, match="non-symlink"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


def test_source_manifest_rejects_symlinked_dependency_ancestor(
    tmp_path: Path,
) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    package = repo / "src/pkg"
    outside = tmp_path / "outside-pkg"
    package.rename(outside)
    package.symlink_to(outside, target_is_directory=True)
    with pytest.raises(gates.GateVerificationError, match="non-symlink"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


def test_source_manifest_rejects_current_tree_only_dependency(tmp_path: Path) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    (repo / gates.CUMULATIVE_ROOTS[0]).write_text(
        "from src.pkg import dep, current_only\n", encoding="utf-8"
    )
    (repo / "src/pkg/current_only.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(gates.GateVerificationError, match="current source path set"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


def test_source_manifest_must_match_trusted_git_objects(tmp_path: Path) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    files = manifest["files"]
    assert isinstance(files, dict)
    files["src/pkg/dep.py"] = "0" * 64
    with pytest.raises(gates.GateVerificationError, match="trusted source hash"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


def test_source_manifest_rejects_current_hash_mismatch(tmp_path: Path) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    (repo / "src/pkg/dep.py").write_text("VALUE = 9\n", encoding="utf-8")
    with pytest.raises(gates.GateVerificationError, match="current source hash"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


def test_source_manifest_fails_closed_on_unresolved_local_import(tmp_path: Path) -> None:
    repo, git, roots = _source_fixture(tmp_path)
    git.objects[gates.CUMULATIVE_ROOTS[0]] = b"from src.pkg.missing import VALUE\n"
    with pytest.raises(gates.GateVerificationError, match="unresolved local import"):
        gates.build_source_manifest(
            repo=repo,
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            roots=roots,
            git=git,
        )


def test_source_manifest_rejects_missing_symbol_from_existing_package(
    tmp_path: Path,
) -> None:
    repo, git, roots = _source_fixture(tmp_path)
    git.objects[gates.CUMULATIVE_ROOTS[0]] = (
        b"from src.pkg import definitely_missing\n"
    )
    with pytest.raises(gates.GateVerificationError, match="unresolved local import"):
        gates.build_source_manifest(
            repo=repo,
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            roots=roots,
            git=git,
        )


@pytest.mark.parametrize(
    ("symbol", "raises"),
    [("ExportedClass", False), ("EXPORTED_VALUE", False), ("MissingClass", True)],
)
def test_source_manifest_proves_symbols_exported_by_regular_module(
    tmp_path: Path, symbol: str, raises: bool
) -> None:
    repo, git, roots = _source_fixture(tmp_path)
    git.objects["src/pkg/dep.py"] = (
        b"class ExportedClass:\n    pass\n\nEXPORTED_VALUE = 1\n"
    )
    git.objects[gates.CUMULATIVE_ROOTS[0]] = (
        f"from src.pkg.dep import {symbol}\n".encode()
    )
    if raises:
        with pytest.raises(gates.GateVerificationError, match="unresolved local import"):
            gates.build_source_manifest(
                repo=repo,
                base_commit=gates.CUMULATIVE_BASE_COMMIT,
                roots=roots,
                git=git,
            )
    else:
        manifest = gates.build_source_manifest(
            repo=repo,
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            roots=roots,
            git=git,
        )
        assert "src/pkg/dep.py" in manifest["files"]


def test_source_manifest_accepts_package_reexport_and_child_module(
    tmp_path: Path,
) -> None:
    repo, git, roots = _source_fixture(tmp_path)
    git.objects["src/pkg/dep.py"] = b"class ExportedClass:\n    pass\n"
    git.objects["src/pkg/__init__.py"] = (
        b"from src.pkg.dep import ExportedClass\n__all__ = ['ExportedClass']\n"
    )
    git.objects[gates.CUMULATIVE_ROOTS[0]] = (
        b"from src.pkg import ExportedClass, dep\n"
    )
    manifest = gates.build_source_manifest(
        repo=repo,
        base_commit=gates.CUMULATIVE_BASE_COMMIT,
        roots=roots,
        git=git,
    )
    assert {"src/pkg/__init__.py", "src/pkg/dep.py"} <= set(manifest["files"])


@pytest.mark.parametrize(
    "source",
    [
        "from importlib import import_module\nimport_module('src.pkg.dynamic')\n",
        "from importlib import import_module as load\nload('src.pkg.dynamic')\n",
        "import importlib as loader\nloader.import_module('src.pkg.dynamic')\n",
        "__import__('src.pkg.dynamic')\n",
    ],
)
def test_source_manifest_follows_literal_dynamic_local_import(
    tmp_path: Path, source: str
) -> None:
    repo, git, roots = _source_fixture(tmp_path)
    git.objects[gates.CUMULATIVE_ROOTS[0]] = source.encode()
    git.objects["src/pkg/dynamic.py"] = b"from src.pkg import dep\n"
    manifest = gates.build_source_manifest(
        repo=repo,
        base_commit=gates.CUMULATIVE_BASE_COMMIT,
        roots=roots,
        git=git,
    )
    assert {"src/pkg/dynamic.py", "src/pkg/dep.py"} <= set(manifest["files"])


@pytest.mark.parametrize(
    "source",
    [
        "from importlib import import_module\nimport_module(module_name)\n",
        "import importlib\nimportlib.import_module(module_name)\n",
        "__import__(module_name)\n",
    ],
)
def test_source_manifest_rejects_nonliteral_dynamic_import(
    tmp_path: Path, source: str
) -> None:
    repo, git, roots = _source_fixture(tmp_path)
    git.objects[gates.CUMULATIVE_ROOTS[0]] = source.encode()
    with pytest.raises(gates.GateVerificationError, match="non-literal dynamic import"):
        gates.build_source_manifest(
            repo=repo,
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            roots=roots,
            git=git,
        )


def test_git_reader_distinguishes_absent_path_from_show_failure(tmp_path: Path) -> None:
    repo, git, roots = _source_fixture(tmp_path)
    git.objects[gates.CUMULATIVE_ROOTS[0]] = b"import src.pkg.absent\n"
    with pytest.raises(gates.GateVerificationError, match="unresolved local import"):
        gates.build_source_manifest(
            repo=repo,
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            roots=roots,
            git=git,
        )
    assert "src/pkg/absent.py" not in git.show_calls

    repo, git, roots = _source_fixture(tmp_path / "failure")
    git.fail_show.add("src/pkg/dep.py")
    with pytest.raises(gates.GateVerificationError, match="operational show failure"):
        gates.build_source_manifest(
            repo=repo,
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            roots=roots,
            git=git,
        )


def test_source_manifest_rejects_reduced_cumulative_roots(tmp_path: Path) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    removed = manifest["roots"].pop()
    manifest["files"].pop(removed)
    with pytest.raises(gates.GateVerificationError, match="cumulative roots"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_source_manifest_requires_integer_schema_version(
    tmp_path: Path, schema_version: object
) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    manifest["schema_version"] = schema_version
    with pytest.raises(gates.GateVerificationError, match="identity"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


def test_write_source_manifest_existing_clean_same_is_noop(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    payload = {"schema_version": 1, "files": {"a.py": "0" * 64}}
    destination.write_bytes(gates._canonical_json_bytes(payload))
    before = destination.stat()

    gates.write_source_manifest(
        repo=tmp_path, destination=destination, payload=payload, git=SourceGit({})
    )

    after = destination.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_write_source_manifest_atomically_replaces_clean_different(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old\n", encoding="utf-8")
    destination.chmod(0o640)
    original_mode = stat.S_IMODE(destination.stat().st_mode)
    payload = {"schema_version": 2}
    gates.write_source_manifest(
        repo=tmp_path, destination=destination, payload=payload, git=SourceGit({})
    )
    assert destination.read_bytes() == gates._canonical_json_bytes(payload)
    assert stat.S_IMODE(destination.stat().st_mode) == original_mode
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_write_source_manifest_rejects_dirty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("keep\n", encoding="utf-8")
    with pytest.raises(gates.GateVerificationError, match="dirty"):
        gates.write_source_manifest(
            repo=tmp_path,
            destination=destination,
            payload={"ok": True},
            git=SourceGit({}, dirty=True),
        )
    assert destination.read_text(encoding="utf-8") == "keep\n"


def test_write_source_manifest_creates_new_file_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "nested/manifest.json"
    payload = {"ok": True}
    gates.write_source_manifest(
        repo=tmp_path, destination=destination, payload=payload, git=SourceGit({})
    )
    assert destination.read_bytes() == gates._canonical_json_bytes(payload)
    assert not list(destination.parent.glob(".manifest.json.*.tmp"))


def test_write_source_manifest_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("keep\n", encoding="utf-8")
    destination = tmp_path / "manifest.json"
    destination.symlink_to(target)
    with pytest.raises((gates.GateVerificationError, FileExistsError), match="symlink"):
        gates.write_source_manifest(
            repo=tmp_path,
            destination=destination,
            payload={"ok": True},
            git=SourceGit({}),
        )
    assert target.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize("kind", ["parent", "absolute", "nested_symlink"])
def test_write_source_manifest_rejects_destination_escape(
    tmp_path: Path, kind: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "parent":
        destination = repo / ".." / "escaped.json"
    elif kind == "absolute":
        destination = outside / "escaped.json"
    else:
        (repo / "linked").symlink_to(outside, target_is_directory=True)
        destination = repo / "linked/escaped.json"
    with pytest.raises(gates.GateVerificationError, match="inside repo|symlink"):
        gates.write_source_manifest(
            repo=repo,
            destination=destination,
            payload={"ok": True},
            git=SourceGit({}),
        )
    assert not (outside / "escaped.json").exists()


def test_load_source_manifest_rejects_recursive_duplicate_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"files": {"a.py": "one", "a.py": "two"}}\n')
    with pytest.raises(gates.GateVerificationError, match="duplicate JSON key"):
        gates._load_source_manifest(manifest)


@pytest.mark.parametrize(
    "invalid",
    [
        "/absolute.py",
        "../escape.py",
        "src/../escape.py",
        "./src/a.py",
        "src\\a.py",
        "src//a.py",
        "src/a\0.py",
    ],
)
def test_source_manifest_rejects_noncanonical_paths(
    tmp_path: Path, invalid: str
) -> None:
    manifest, repo, git = _valid_source_manifest(tmp_path)
    manifest["files"][invalid] = "0" * 64
    with pytest.raises(gates.GateVerificationError, match="canonical relative POSIX"):
        gates.verify_source_manifest(manifest, repo=repo, git=git)


class PassingRunner:
    def __init__(self, repo: Path, mutate: str | None = None) -> None:
        self.repo = repo
        self.mutate = mutate
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[bytes]:
        assert cwd == self.repo
        self.calls.append(argv)
        if self.mutate and len(self.calls) == 1:
            (self.repo / self.mutate).write_text("changed\n")
        return subprocess.CompletedProcess(argv, 0, b"2 passed in 0.01s\n", b"")


def _generate(tmp_path: Path, **overrides: object) -> tuple[Path, PassingRunner]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)
    output = tmp_path / "gate.json"
    runner = PassingRunner(repo)
    sequence = gates.EXACT_PROFILE_SEQUENCE
    source_sha = hashlib.sha256((repo / gates.DEFAULT_SOURCE_MANIFEST).read_bytes()).hexdigest()
    exact_runs = [
        _exact_execution(
            profile,
            tmp_path / f"exact-{position}",
            100 + position,
            source_sha=source_sha,
            source_path=repo / gates.DEFAULT_SOURCE_MANIFEST,
        )
        for position, profile in enumerate(sequence)
    ]
    exact_specs = [{"profile": profile} for profile in sequence]

    def execute_transaction(
        specs: list[dict[str, object]], **kwargs: object
    ) -> dict[str, object]:
        assert specs == exact_specs
        assert kwargs["repo"] == repo.resolve()
        return gates.verify_exact_profile_runs(
            exact_runs,
            compare=kwargs["compare"],
            schema1_variants=PRODUCTION_SCHEMA1_VARIANTS,
        )

    kwargs = {
        "repo": repo,
        "output": output,
        "base_commit": gates.CUMULATIVE_BASE_COMMIT,
        "python_executable": "/env/bin/python",
        "git": FakeGit(),
        "run": runner,
        "now_utc": lambda: "2026-07-25T00:00:00Z",
        "exact_profile_specs": exact_specs,
        "transaction_dir": tmp_path / "transaction",
        "execute_transaction": execute_transaction,
        "compare": lambda left, right, **kwargs: {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2, 7],
            "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
            "root_sha256": "e" * 64,
        },
    }
    kwargs.update(overrides)
    gates.generate_evidence(**kwargs)
    return output, runner


def test_generates_packager_shared_exact_and_determinism_evidence(tmp_path: Path) -> None:
    output, runner = _generate(tmp_path)
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 1
    assert payload["manifest_id"] == "oviv2_dual_readout_development_gates_v1"
    deterministic = payload["deterministic_evidence"]
    assert deterministic["base_commit"] == gates.CUMULATIVE_BASE_COMMIT
    assert deterministic["code_commit"] == "a" * 40
    assert deterministic["code_tree"] == "b" * 40
    assert len(deterministic["protected_files"]) == 6
    assert len(deterministic["test_sources"]) == len(gates.TEST_FILES)
    assert payload["receipt"] == {"created_at_utc": "2026-07-25T00:00:00Z"}

    for name in ("t1_exact", "determinism"):
        gate = deterministic["gates"][name]
        assert gate["scope"] == "shared_code_and_A0-A4_fixture"
        assert gate["status"] == "PASS"
        assert gate["code_commit"] == "a" * 40
        assert gate["code_tree"] == "b" * 40
        assert gate["protected_records"] == deterministic["protected_files"]
        assert gate["test_records"]
        record = gate["test_records"][0]
        assert record["argv"][:3] == ["/env/bin/python", "-m", "pytest"]
        assert record["returncode"] == 0
        assert record["stdout_bytes"] == len(b"2 passed in 0.01s\n")
        assert len(record["stdout_sha256"]) == 64
        assert record["stderr_bytes"] == 0
    assert len(runner.calls) == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_commit": "c" * 40}, "cumulative base"),
        ({"git": FakeGit(dirty=True)}, "clean"),
    ],
)
def test_fails_closed_before_tests(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(gates.GateVerificationError, match=message):
        _generate(tmp_path, **overrides)
    assert not (tmp_path / "gate.json").exists()


@pytest.mark.parametrize(
    "summary",
    [
        b"1 passed, 1 skipped\n",
        b"1 passed, 1 xfailed\n",
        b".s\n",
        b".x\n",
        b"1 passed, 9 deselected\n",
    ],
)
def test_rejects_skip_or_xfail_and_cleans_output(tmp_path: Path, summary: bytes) -> None:
    def run(argv: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[bytes]:
        del cwd
        return subprocess.CompletedProcess(argv, 0, summary, b"")

    with pytest.raises(gates.GateVerificationError, match="skip|xfail|deselect"):
        _generate(tmp_path, run=run)
    assert not (tmp_path / "gate.json").exists()


def test_rejects_nonzero_test_and_input_toctou(tmp_path: Path) -> None:
    def fail(argv: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[bytes]:
        del cwd
        return subprocess.CompletedProcess(argv, 1, b"", b"failed")

    with pytest.raises(gates.GateVerificationError, match="failed"):
        _generate(tmp_path, run=fail)
    assert not (tmp_path / "gate.json").exists()

    repo = tmp_path / "other-repo"
    repo.mkdir()
    _make_repo(repo)
    runner = PassingRunner(repo, gates.TEST_FILES[0])
    with pytest.raises(gates.GateVerificationError, match="changed"):
        gates.generate_evidence(
            repo=repo,
            output=tmp_path / "other.json",
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            python_executable=sys.executable,
            git=FakeGit(),
            run=runner,
            now_utc=lambda: "2026-07-25T00:00:00Z",
        )
    assert not (tmp_path / "other.json").exists()


def test_rejects_symlink_oversize_and_existing_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)
    protected = repo / gates.CUMULATIVE_ROOTS[0]
    protected.unlink()
    protected.symlink_to(repo / gates.CUMULATIVE_ROOTS[1])
    with pytest.raises(gates.GateVerificationError, match="regular non-symlink"):
        gates.generate_evidence(
            repo=repo,
            output=tmp_path / "gate.json",
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            python_executable=sys.executable,
            git=FakeGit(),
            run=PassingRunner(repo),
            now_utc=lambda: "2026-07-25T00:00:00Z",
        )

    protected.unlink()
    protected.write_bytes(b"x" * 32)
    with pytest.raises(gates.GateVerificationError, match="maximum"):
        gates.generate_evidence(
            repo=repo,
            output=tmp_path / "gate.json",
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            python_executable=sys.executable,
            git=FakeGit(),
            run=PassingRunner(repo),
            now_utc=lambda: "2026-07-25T00:00:00Z",
            max_input_bytes=16,
        )

    output = tmp_path / "gate.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        gates.generate_evidence(
            repo=repo,
            output=output,
            base_commit=gates.CUMULATIVE_BASE_COMMIT,
            python_executable=sys.executable,
            git=FakeGit(),
            run=PassingRunner(repo),
            now_utc=lambda: "2026-07-25T00:00:00Z",
            max_input_bytes=64,
        )
    assert output.read_text() == "keep"


def test_focused_real_smoke_uses_fast_fixture_command(tmp_path: Path) -> None:
    del tmp_path
    repo = Path(__file__).parents[2]
    completed = gates.run_focused_smoke(
        repo=repo,
        python_executable=sys.executable,
    )
    assert completed.returncode == 0
    assert gates.SMOKE_TEST in completed.args


def test_real_runner_clears_pytest_addopts_that_would_deselect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del tmp_path
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k definitely_not_a_real_test")
    completed = gates.run_focused_smoke(
        repo=Path(__file__).parents[2],
        python_executable=sys.executable,
    )
    assert completed.returncode == 0
    assert b"deselected" not in completed.stdout


def test_atomic_publication_failure_leaves_no_output_or_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_link(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("link failed")

    monkeypatch.setattr(gates.os, "link", fail_link)
    with pytest.raises(OSError, match="link failed"):
        _generate(tmp_path)
    assert not (tmp_path / "gate.json").exists()
    assert not list(tmp_path.glob(".gate.json.*.tmp"))


def test_atomic_post_link_failure_removes_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = gates.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(gates.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        _generate(tmp_path)
    assert not (tmp_path / "gate.json").exists()
    assert not list(tmp_path.glob(".gate.json.*.tmp"))


def test_atomic_directory_close_failure_removes_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_close = gates.os.close
    failed = False

    def fail_once(descriptor: int) -> None:
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            failed = True
            raise OSError("directory close failed")
        real_close(descriptor)

    monkeypatch.setattr(gates.os, "close", fail_once)
    with pytest.raises(OSError, match="directory close failed"):
        _generate(tmp_path)
    assert not (tmp_path / "gate.json").exists()
    assert not list(tmp_path.glob(".gate.json.*.tmp"))


def test_rechecks_git_at_publication_boundary(tmp_path: Path) -> None:
    class BecomesDirtyGit(FakeGit):
        def __init__(self) -> None:
            super().__init__()
            self.status_calls = 0

        def __call__(self, argv: tuple[str, ...], cwd: Path) -> bytes:
            if argv == ("status", "--porcelain", "--untracked-files=all"):
                self.status_calls += 1
                if self.status_calls == 3:
                    return b" M changed-at-publication\n"
            return super().__call__(argv, cwd)

    with pytest.raises(gates.GateVerificationError, match="clean"):
        _generate(tmp_path, git=BecomesDirtyGit())
    assert not (tmp_path / "gate.json").exists()
    assert not list(tmp_path.glob(".gate.json.*.tmp"))


def test_publication_boundary_validation_runs_before_staging_creation(
    tmp_path: Path,
) -> None:
    seen_staging: list[Path] = []

    def validate() -> None:
        seen_staging.extend(tmp_path.glob(".gate.json.*.tmp"))

    gates._atomic_json_no_replace(tmp_path / "gate.json", {"ok": True}, validate)
    assert seen_staging == []


def _exact_execution(
    profile: str,
    root: Path,
    pid: int,
    *,
    commit: str = "a" * 40,
    source_sha: str | None = None,
    source_path: Path | None = None,
    inputs: dict[str, str] | None = None,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    source_path = (
        source_path
        if source_path is not None
        else Path(__file__).parents[2] / gates.DEFAULT_SOURCE_MANIFEST
    ).resolve()
    source_data = source_path.read_bytes()
    actual_source_sha = hashlib.sha256(source_data).hexdigest()
    if source_sha is not None and source_sha != actual_source_sha:
        raise ValueError("test source digest does not match canonical source manifest")
    runner = (
        Path(reference_worker.__file__).resolve()
        if profile == "reference"
        else (Path(__file__).parents[2] / "scripts/evaluation/run_oviv2_tesse_cd_v2.py").resolve()
    )
    input_path = (root.parent / "input.json").resolve()
    schedule_path = (root.parent / "schedule.json").resolve()
    target_path = (root.parent / "target.json").resolve()
    for path, payload in (
        (input_path, {"input": "shared"}),
        (schedule_path, {"schedule": "shared"}),
        (target_path, {"target": "shared"}),
    ):
        if not path.exists():
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    config_path = (root.parent / f"{profile}.json").resolve()
    config_payload = {
        "scene": "apartment",
        "algorithm_hash": "d" * 64,
        "input_manifest": str(input_path),
        "schedule_manifest": str(schedule_path),
        "occlusion_target_manifest": str(target_path),
        "occlusion_target_manifest_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
    }
    config_path.write_text(json.dumps(config_payload, sort_keys=True) + "\n")
    argv = [
        "/env/bin/python", str(runner), "--config", str(config_path),
        "--output", str(root.resolve()),
    ]
    if profile == "reference":
        argv += ["--receipt", str((root / "t1_exact_receipt.json").resolve()),
                 "--source-manifest", str(source_path)]
    run_manifest = {
        "schema_version": 1 if profile == "reference" else 2,
        "algorithm_hash": "d" * 64,
        "code_commit": commit,
        "scene": "apartment",
        "mode": "dual_readout_causal_checkpoints",
        "config": {
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "byte_count": len(config_path.read_bytes()),
        },
        "schedule": {
            "sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
            "byte_count": len(schedule_path.read_bytes()),
        },
        "target_manifest": {
            "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "byte_count": len(target_path.read_bytes()),
        },
        "source_bindings": {
            "input_manifest": {
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "byte_count": len(input_path.read_bytes()),
            },
            "dataset": "fixture",
            "cache": "shared",
        },
        "checkpoints": [{"frame_index": 2}],
    }
    (root / "run_manifest.json").write_text(
        json.dumps(run_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    audit = {
        "checkpoint_frames": [2, 7],
        "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
        "root_sha256": "e" * 64,
    }
    if profile == "reference":
        worker_execution = {
            "profile": profile,
            "mode": gates.DEVELOPMENT_MODE,
            "argv": argv,
            "pid": pid,
            "code_commit": commit,
            "source_manifest_sha256": actual_source_sha,
            "input_fingerprints": inputs or {
                "config": hashlib.sha256(config_path.read_bytes()).hexdigest()
            },
            "output_root": str(root.resolve()),
        }
        receipt = {
            "schema_version": 1,
            "format": "oviv2_t1_exact_execution_receipt_v1",
            "execution": worker_execution,
            "source_manifest": {
                "path": str(source_path),
                "sha256": actual_source_sha,
                "byte_count": len(source_data),
            },
            "artifact_inventory": audit["inventory"],
            "checkpoint_frames": audit["checkpoint_frames"],
            "cumulative_root_sha256": audit["root_sha256"],
        }
        receipt_path = root / "t1_exact_receipt.json"
    else:
        receipt = {
            "schema_version": 1,
            "provenance": {
                "repository_commit": commit,
                "command": argv[1:],
            },
            "environment": {},
        }
        receipt_path = root / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    completed = gates._reopen_completed_execution(
        profile, root.resolve(), argv, pid, 0, source_path
    )
    status = root.stat()
    observation = {
        "schema_version": 1,
        "format": "oviv2_exact_process_observation_v1",
        "position": int(root.name.rsplit("-", 1)[-1]) if root.name.rsplit("-", 1)[-1].isdigit() else pid,
        "profile": profile,
        "argv": argv,
        "pid": pid,
        "returncode": 0,
        "config": gates._absolute_file_record(config_path),
        "source_manifest": gates._absolute_file_record(source_path),
        "output_root": str(root.resolve()),
        "root_device": status.st_dev,
        "root_inode": status.st_ino,
        "trust_model": gates.LOCAL_PROCESS_TRUST_MODEL,
        "execution_context": gates.DEVELOPMENT_EXECUTION_CONTEXT,
        "run_manifest": gates._absolute_file_record(root / "run_manifest.json"),
        "production_receipt": gates._absolute_file_record(receipt_path),
        "completed_execution": completed,
    }
    observation_dir = root.parent / "observations"
    observation_dir.mkdir(exist_ok=True)
    observation_path = observation_dir / f"{root.name}.json"
    observation_path.write_text(json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n")
    return {**completed, "observation_receipt": gates._absolute_file_record(observation_path)}


def test_exact_profile_gate_requires_interleaved_independent_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = ("reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4")
    executions = [
        _exact_execution(profile, tmp_path / f"run-{index}", 100 + index)
        for index, profile in enumerate(profiles)
    ]
    calls: list[tuple[Path, Path, dict[str, object]]] = []

    def compare(left: Path, right: Path, **kwargs: object) -> dict[str, object]:
        calls.append((left, right, kwargs))
        if kwargs != {
            "left_schema1_variant": "production",
            "right_schema1_variant": "production",
        }:
            raise gates.ArtifactMismatch("wrong caller-trusted variant")
        return {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2, 7],
            "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
            "root_sha256": "e" * 64,
        }

    with pytest.raises(gates.GateVerificationError, match="schema1 variant"):
        gates.verify_exact_profile_runs(executions, compare=compare)
    evidence = gates.verify_exact_profile_runs(
        executions,
        compare=compare,
        schema1_variants=PRODUCTION_SCHEMA1_VARIANTS,
    )
    assert evidence["sequence"] == list(profiles)
    assert set(evidence["profiles"]) == {"a0", "a1", "a2", "a3", "a4"}
    assert all(
        profile["cumulative_root_sha256"] == "e" * 64
        for profile in evidence["profiles"].values()
    )
    assert len(calls) == 9
    assert all(left == right for left, right, _ in calls)
    assert len({left for left, _, _ in calls}) == 9

    wrong = {**PRODUCTION_SCHEMA1_VARIANTS, "reference": "t1_transaction"}
    with pytest.raises(gates.GateVerificationError, match="wrong caller-trusted"):
        gates.verify_exact_profile_runs(
            executions,
            compare=compare,
            schema1_variants=wrong,
        )

    executions[1]["pid"] = executions[0]["pid"]
    observation_path = Path(executions[1]["observation_receipt"]["path"])
    observation = json.loads(observation_path.read_text())
    observation["pid"] = executions[1]["pid"]
    observation["completed_execution"]["pid"] = executions[1]["pid"]
    observation_path.write_text(json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n")
    executions[1]["observation_receipt"] = gates._absolute_file_record(observation_path)
    with pytest.raises(gates.GateVerificationError, match="PID"):
        gates.verify_exact_profile_runs(
            executions,
            compare=compare,
            schema1_variants=PRODUCTION_SCHEMA1_VARIANTS,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "minimal"))
def test_exact_profile_gate_requires_strict_caller_variant_mapping(
    mutation: str,
) -> None:
    variants = dict(PRODUCTION_SCHEMA1_VARIANTS)
    if mutation == "missing":
        variants.pop("a4")
    elif mutation == "extra":
        variants["unexpected"] = "production"
    else:
        variants["reference"] = "minimal"
    with pytest.raises(gates.GateVerificationError, match="schema1 variant mapping"):
        gates.verify_exact_profile_runs([], schema1_variants=variants)


def _exact_audits() -> list[dict[str, object]]:
    return [
        {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2, 7],
            "inventory": [
                {"path": "x", "sha256": "d" * 64, "byte_count": 1}
            ],
            "root_sha256": "e" * 64,
        }
        for _ in gates.EXACT_PROFILE_SEQUENCE
    ]


def test_preinjected_audits_cannot_bypass_caller_variant_revalidation(
    tmp_path: Path,
) -> None:
    executions = [
        _exact_execution(profile, tmp_path / f"run-{position}", 100 + position)
        for position, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE)
    ]
    calls = 0

    def compare(left: Path, right: Path, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        del left, right
        calls += 1
        if kwargs["left_schema1_variant"] != "production":
            raise gates.ArtifactMismatch("wrong caller-trusted variant")
        return _exact_audits()[0]

    wrong = {**PRODUCTION_SCHEMA1_VARIANTS, "reference": "t1_transaction"}
    with pytest.raises(gates.GateVerificationError, match="wrong caller-trusted"):
        gates.verify_exact_profile_runs(
            executions,
            compare=compare,
            audits=_exact_audits(),
            schema1_variants=wrong,
        )
    assert calls == 1


@pytest.mark.parametrize("mutation", ("root", "inventory", "extra", "type"))
def test_preinjected_audits_must_exactly_match_recomputed_roots(
    tmp_path: Path, mutation: str
) -> None:
    executions = [
        _exact_execution(profile, tmp_path / f"run-{position}", 100 + position)
        for position, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE)
    ]
    expected = _exact_audits()[0]
    audits = _exact_audits()
    if mutation == "root":
        audits[0]["root_sha256"] = "f" * 64
    elif mutation == "inventory":
        audits[0]["inventory"] = []
    elif mutation == "extra":
        audits[0]["unexpected"] = True
    else:
        audits[0]["inventory"][0]["byte_count"] = True

    with pytest.raises(gates.GateVerificationError, match="audit.*recomputed"):
        gates.verify_exact_profile_runs(
            executions,
            compare=lambda left, right, **kwargs: dict(expected),
            audits=audits,
            schema1_variants=PRODUCTION_SCHEMA1_VARIANTS,
        )


def test_matching_preinjected_audits_are_recomputed_and_accepted(
    tmp_path: Path,
) -> None:
    executions = [
        _exact_execution(profile, tmp_path / f"run-{position}", 100 + position)
        for position, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE)
    ]
    calls = 0
    expected = _exact_audits()[0]

    def compare(left: Path, right: Path, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        del left, right, kwargs
        calls += 1
        return dict(expected)

    result = gates.verify_exact_profile_runs(
        executions,
        compare=compare,
        audits=_exact_audits(),
        schema1_variants=PRODUCTION_SCHEMA1_VARIANTS,
    )
    assert result["sequence"] == list(gates.EXACT_PROFILE_SEQUENCE)
    assert calls == len(gates.EXACT_PROFILE_SEQUENCE)


def test_dual_receipt_command_must_equal_parent_popen_argv(tmp_path: Path) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    root = Path(execution["output_root"])
    receipt_path = root / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["provenance"]["command"] = ["different-runner.py"]
    receipt_path.write_text(json.dumps(receipt) + "\n")
    observation = json.loads(
        Path(execution["observation_receipt"]["path"]).read_text()
    )

    with pytest.raises(gates.GateVerificationError, match="command"):
        gates._reopen_completed_execution(
            "a1",
            root,
            execution["argv"],
            execution["pid"],
            0,
            Path(observation["source_manifest"]["path"]),
        )


def test_observation_declares_local_unsigned_pid_trust_model(tmp_path: Path) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    observation_path = Path(execution["observation_receipt"]["path"])
    observation = json.loads(observation_path.read_text())
    assert observation["trust_model"] == {
        "pid_semantics": "trusted_local_orchestrator_observation",
        "observation_basis": "parent_popen_and_waitpid",
        "root_ownership": "after_successful_receipt_manifest_observation_reopen_only",
        "audit_authentication": "unsigned_local_audit",
        "stability_scope": "verification_interval_only",
        "excluded_adversaries": ["same_uid_process", "root"],
    }
    observation["trust_model"]["audit_authentication"] = "cryptographic_proof"
    observation_path.write_text(json.dumps(observation) + "\n")
    execution["observation_receipt"] = gates._absolute_file_record(observation_path)
    with pytest.raises(gates.GateVerificationError, match="trust|observation"):
            gates._bind_exact_receipt(
                execution,
                expected_position=2,
                schema1_variant="production",
            compare=lambda left, right, **kwargs: {
                "checkpoint_frames": [2, 7],
                "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
                "root_sha256": "e" * 64,
            },
        )


def test_exact_transaction_uses_popen_pid_argv_and_returncode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    source = tmp_path / "source.json"
    config.write_text('{"scene":"apartment"}\n')
    source.write_text("{}\n")
    specs = [
        {
            "profile": profile,
            "config": str(config.resolve()),
            "output_root": str((tmp_path / f"run-{position}").resolve()),
            "source_manifest": str(source.resolve()),
        }
        for position, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE)
    ]
    launched: list[tuple[list[str], int]] = []

    class Process:
        def __init__(self, argv: list[str], pid: int) -> None:
            self.args = argv
            self.pid = pid
            self.returncode = 0

        def communicate(self) -> tuple[bytes, bytes]:
            return b"ok\n", b""

    def popen(argv: list[str], **kwargs: object) -> Process:
        assert kwargs["cwd"] == gates.REPO_ROOT
        pid = 7000 + len(launched)
        launched.append((argv, pid))
        output = Path(argv[5])
        output.mkdir()
        (output / "run_manifest.json").write_text("{}\n")
        receipt_name = (
            "t1_exact_receipt.json"
            if Path(argv[1]).name == "run_oviv2_t1_reference.py"
            else "execution_receipt.json"
        )
        (output / receipt_name).write_text("{}\n")
        return Process(argv, pid)

    def reopen(
        profile: str,
        root: Path,
        argv: list[str],
        pid: int,
        returncode: int,
        source_manifest: Path,
    ) -> dict[str, object]:
        assert source_manifest == source.resolve()
        return {
            "profile": profile,
            "argv": argv,
            "pid": pid,
            "returncode": returncode,
            "code_commit": "a" * 40,
            "source_manifest_sha256": "b" * 64,
            "per_run_fingerprints": {"config": "c" * 64, "algorithm": "d" * 64, "profile": "e" * 64},
            "common_input_fingerprints": {"dataset": "f" * 64},
            "output_root": str(root),
            "receipt_sha256": "1" * 64,
        }

    seen: list[dict[str, object]] = []
    monkeypatch.setattr(gates, "_reopen_completed_execution", reopen)
    monkeypatch.setattr(
        gates,
        "verify_exact_profile_runs",
        lambda executions, **kwargs: seen.extend(executions) or {"executions": executions},
    )
    compare_calls: list[tuple[Path, Path]] = []

    def compare(left: Path, right: Path, **kwargs: object) -> dict[str, object]:
        compare_calls.append((left, right))
        return {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2],
            "inventory": [{"path": "x", "sha256": "2" * 64, "byte_count": 1}],
            "root_sha256": "3" * 64,
        }

    result = gates.execute_exact_profile_transaction(
        specs,
        repo=gates.REPO_ROOT,
        python_executable="/env/bin/python",
        transaction_dir=tmp_path / "transaction",
        popen_factory=popen,
        compare=compare,
    )

    assert result["executions"] == seen
    assert [record["pid"] for record in seen] == [pid for _, pid in launched]
    assert all(record["returncode"] == 0 for record in seen)
    assert [record["profile"] for record in seen] == list(gates.EXACT_PROFILE_SEQUENCE)
    assert len(compare_calls) == 9
    assert all(left == right for left, right in compare_calls)
    assert len({left for left, _ in compare_calls}) == 9
    observations = sorted((tmp_path / "transaction/receipts").glob("*.json"))
    assert len(observations) == 9
    assert [json.loads(path.read_text())["pid"] for path in observations] == [
        pid for _, pid in launched
    ]
    assert all(
        json.loads(path.read_text())["trust_model"]
        == gates.LOCAL_PROCESS_TRUST_MODEL
        for path in observations
    )
    assert all("freeze_manifest" not in json.loads(path.read_text()) for path in observations)
    assert all(
        json.loads(path.read_text())["execution_context"]
        == gates.DEVELOPMENT_EXECUTION_CONTEXT
        for path in observations
    )
    assert launched[0][0] == [
        "/env/bin/python",
        str((gates.REPO_ROOT / "scripts/evaluation/run_oviv2_t1_reference.py").resolve()),
        "--config", str(config.resolve()),
        "--output", str((tmp_path / "run-0").resolve()),
        "--receipt", str((tmp_path / "run-0/t1_exact_receipt.json").resolve()),
        "--source-manifest", str(source.resolve()),
    ]
    assert all(len(argv) == 6 for argv, _ in launched[1:])


@pytest.mark.parametrize("case", ["old_fields", "office", "duplicate_root", "existing_root"])
def test_exact_transaction_preflights_all_specs_before_creating_or_launching(
    tmp_path: Path, case: str
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    specs: list[dict[str, object]] = []
    for position, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE):
        config = tmp_path / f"config-{position}.json"
        config.write_text(json.dumps({"scene": "apartment"}) + "\n")
        spec: dict[str, object] = {
            "profile": profile,
            "config": str(config.resolve()),
            "output_root": str((tmp_path / f"rejected-{position}").resolve()),
            "source_manifest": str(source.resolve()),
        }
        if position == 4 and case == "old_fields":
            spec.update(
                freeze_manifest=str((tmp_path / "freeze.json").resolve()),
                run_slot="apartment_run1",
            )
        specs.append(spec)
    if case == "office":
        Path(specs[4]["config"]).write_text('{"scene":"office"}\n')
    elif case == "duplicate_root":
        specs[4]["output_root"] = specs[1]["output_root"]
    elif case == "existing_root":
        Path(specs[4]["output_root"]).mkdir()
    popen_calls = 0

    def popen(*args: object, **kwargs: object) -> object:
        nonlocal popen_calls
        popen_calls += 1
        raise RuntimeError((args, kwargs))

    transaction = tmp_path / "rejected-transaction"
    with pytest.raises(
        gates.GateVerificationError,
        match="fields|Apartment|apartment|independent|clobber",
    ):
        gates.execute_exact_profile_transaction(
            specs,
            repo=gates.REPO_ROOT,
            python_executable="/env/bin/python",
            transaction_dir=transaction,
            popen_factory=popen,
        )
    assert popen_calls == 0
    assert not transaction.exists()


@pytest.mark.parametrize(
    "relationship",
    ["root_equals_transaction", "root_inside_transaction", "transaction_inside_root"],
)
def test_exact_transaction_rejects_output_root_overlapping_transaction_before_launch(
    tmp_path: Path, relationship: str
) -> None:
    source = tmp_path / "source.json"
    config = tmp_path / "config.json"
    source.write_text("{}\n")
    config.write_text('{"scene":"apartment"}\n')
    overlap_root = (tmp_path / "overlap").resolve()
    transaction = (
        overlap_root / "transaction"
        if relationship == "transaction_inside_root"
        else overlap_root
    )
    first_root = {
        "root_equals_transaction": transaction,
        "root_inside_transaction": transaction / "receipts" / "run",
        "transaction_inside_root": overlap_root,
    }[relationship]
    specs = [
        {
            "profile": profile,
            "config": str(config.resolve()),
            "output_root": str(
                first_root
                if position == 0
                else (tmp_path / f"independent-run-{position}").resolve()
            ),
            "source_manifest": str(source.resolve()),
        }
        for position, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE)
    ]
    popen_calls = 0

    def popen(*args: object, **kwargs: object) -> object:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError((args, kwargs))

    with pytest.raises(gates.GateVerificationError, match="transaction|receipt|overlap"):
        gates.execute_exact_profile_transaction(
            specs,
            repo=gates.REPO_ROOT,
            python_executable="/env/bin/python",
            transaction_dir=transaction,
            popen_factory=popen,
        )

    assert popen_calls == 0
    assert not transaction.exists()


def test_exact_argv_rejects_injected_freeze_flag(tmp_path: Path) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    execution["argv"] = [
        *execution["argv"],
        "--freeze-manifest",
        str((tmp_path / "freeze.json").resolve()),
    ]
    with pytest.raises(gates.GateVerificationError, match="canonical"):
        gates._validate_exact_argv(execution, Path(execution["output_root"]))


@pytest.mark.parametrize("mutation", ["missing_source", "mode"])
def test_exact_execution_rejects_missing_source_hash_or_mode_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    if mutation == "missing_source":
        execution["source_manifest_sha256"] = None
    else:
        execution["profile_config_binding"]["mode"] = "formal_final_freeze"
    with pytest.raises(gates.GateVerificationError, match="binding|mode"):
        gates._exact_execution(execution)


def test_completed_execution_rejects_source_bindings_mismatch(tmp_path: Path) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    root = Path(execution["output_root"])
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_bindings"]["dataset"] = "different"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(gates.GateVerificationError, match="source bindings|reopened|changed"):
            gates._bind_exact_receipt(
                execution,
                expected_position=2,
                schema1_variant="production",
            compare=lambda left, right, **kwargs: {
                "checkpoint_frames": [2, 7],
                "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
                "root_sha256": "e" * 64,
            },
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_input_binding", "forged_input_binding", "schedule", "target"],
)
def test_completed_execution_cross_checks_common_input_content_records(
    tmp_path: Path, mutation: str
) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    root = Path(execution["output_root"])
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "missing_input_binding":
        manifest["source_bindings"].pop("input_manifest")
    elif mutation == "forged_input_binding":
        manifest["source_bindings"]["input_manifest"]["sha256"] = "f" * 64
    elif mutation == "schedule":
        manifest["schedule"]["sha256"] = "f" * 64
    else:
        manifest["target_manifest"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    observation = json.loads(Path(execution["observation_receipt"]["path"]).read_text())
    with pytest.raises(gates.GateVerificationError, match="input|identity|schedule|target"):
        gates._reopen_completed_execution(
            "a1",
            root,
            execution["argv"],
            execution["pid"],
            0,
            Path(observation["source_manifest"]["path"]),
        )


def test_common_input_fingerprint_schema_and_values_are_exact(tmp_path: Path) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    common = execution["common_input_fingerprints"]
    assert set(common) == {
        "source_manifest_sha256",
        "input_manifest_sha256",
        "schedule_sha256",
        "ground_truth_sha256",
        "source_bindings_sha256",
    }
    assert gates.COMMON_INPUT_FINGERPRINT_FIELDS == set(common)
    config = json.loads(Path(execution["argv"][3]).read_text())
    assert common["input_manifest_sha256"] == hashlib.sha256(
        Path(config["input_manifest"]).read_bytes()
    ).hexdigest()
    assert common["schedule_sha256"] == hashlib.sha256(
        Path(config["schedule_manifest"]).read_bytes()
    ).hexdigest()
    assert common["ground_truth_sha256"] == hashlib.sha256(
        Path(config["occlusion_target_manifest"]).read_bytes()
    ).hexdigest()


def test_exact_receipt_rejects_config_changed_after_run(tmp_path: Path) -> None:
    execution = _exact_execution("a1", tmp_path / "run-2", 102)
    config_path = Path(execution["argv"][3])
    config = json.loads(config_path.read_text())
    config["algorithm_hash"] = "f" * 64
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")

    with pytest.raises(gates.GateVerificationError, match="config.*changed|observed config"):
            gates._bind_exact_receipt(
                execution,
                expected_position=2,
                schema1_variant="production",
            compare=lambda left, right, **kwargs: {
                "checkpoint_frames": [2, 7],
                "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
                "root_sha256": "e" * 64,
            },
        )


def _failure_cleanup_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str,
    position: int,
) -> tuple[list[dict[str, object]], Path, object, object]:
    config = tmp_path / "config.json"
    source = tmp_path / "source.json"
    config.write_text('{"scene":"apartment"}\n')
    source.write_text("{}\n")
    specs = [
        {
            "profile": profile,
            "config": str(config.resolve()),
            "output_root": str((tmp_path / f"cleanup-run-{index}").resolve()),
            "source_manifest": str(source.resolve()),
        }
        for index, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE)
    ]
    transaction = tmp_path / "cleanup-transaction"
    failed = False
    launches = 0

    class Process:
        def __init__(self, returncode: int, pid: int) -> None:
            self.returncode = returncode
            self.pid = pid

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"child failed" if self.returncode else b""

    def popen(argv: list[str], **kwargs: object) -> Process:
        nonlocal failed, launches
        del kwargs
        current = launches % len(specs)
        launches += 1
        should_fail = failure == "child" and current == position and not failed
        if should_fail:
            failed = True
            return Process(9, 20_000 + launches)
        output = Path(argv[5])
        output.mkdir()
        (output / "run_manifest.json").write_text("{}\n")
        receipt = output / (
            "t1_exact_receipt.json"
            if Path(argv[1]).name == "run_oviv2_t1_reference.py"
            else "execution_receipt.json"
        )
        receipt.write_text("{}\n")
        return Process(0, 20_000 + launches)

    def reopen(
        profile: str,
        root: Path,
        argv: list[str],
        pid: int,
        returncode: int,
        source_manifest: Path,
    ) -> dict[str, object]:
        nonlocal failed
        del source_manifest
        current = int(root.name.rsplit("-", 1)[1])
        if failure == "receipt" and current == position and not failed:
            failed = True
            raise gates.GateVerificationError("receipt failed")
        return {
            "profile": profile,
            "argv": argv,
            "pid": pid,
            "returncode": returncode,
            "code_commit": "a" * 40,
            "source_manifest_sha256": "b" * 64,
            "profile_config_binding": {
                "mode": gates.DEVELOPMENT_MODE,
                "config_sha256": "c" * 64,
                "algorithm_hash": "e" * 64,
                "profile_sha256": "f" * 64,
            },
            "common_input_fingerprints": {
                "source_manifest_sha256": "1" * 64,
                "input_manifest_sha256": "2" * 64,
                "schedule_sha256": "3" * 64,
                "ground_truth_sha256": "4" * 64,
                "source_bindings_sha256": "5" * 64,
            },
            "output_root": str(root),
            "receipt_sha256": "6" * 64,
            "run_manifest_sha256": "7" * 64,
        }

    compare_calls = 0

    def compare(left: Path, right: Path, **kwargs: object) -> dict[str, object]:
        nonlocal failed, compare_calls
        compare_calls += 1
        if failure == "compare" and compare_calls > position and not failed:
            failed = True
            raise gates.ArtifactMismatch("compare failed")
        return {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2],
            "inventory": [{"path": "x", "sha256": "8" * 64, "byte_count": 1}],
            "root_sha256": "9" * 64,
        }

    monkeypatch.setattr(gates, "_reopen_completed_execution", reopen)
    monkeypatch.setattr(
        gates,
        "verify_exact_profile_runs",
        lambda executions, **kwargs: {"executions": executions},
    )
    return specs, transaction, popen, compare


@pytest.mark.parametrize(
    ("recovery", "expected_events"),
    [
        (
            "terminate",
            ["communicate", "poll", "terminate", "wait", "cleanup"],
        ),
        (
            "kill",
            [
                "communicate", "poll", "terminate", "wait", "kill", "wait",
                "cleanup",
            ],
        ),
        ("exited", ["communicate", "poll", "wait", "cleanup"]),
    ],
)
def test_exact_transaction_reaps_interrupted_process_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery: str,
    expected_events: list[str],
) -> None:
    specs, transaction, _, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )
    events: list[str] = []

    class InterruptedProcess:
        pid = 31_338
        returncode = None

        def __init__(self) -> None:
            self.running = recovery != "exited"
            self.reaped = False
            self.wait_calls = 0

        def communicate(self) -> tuple[bytes, bytes]:
            events.append("communicate")
            raise KeyboardInterrupt("stop requested")

        def poll(self) -> int | None:
            events.append("poll")
            return None if self.running else 0

        def terminate(self) -> None:
            events.append("terminate")
            if recovery == "terminate":
                self.running = False

        def kill(self) -> None:
            events.append("kill")
            self.running = False

        def wait(self, timeout: float | None = None) -> int:
            events.append("wait")
            assert timeout is not None
            self.wait_calls += 1
            if recovery == "kill" and self.wait_calls == 1:
                raise subprocess.TimeoutExpired(["worker"], timeout)
            assert not self.running
            self.reaped = True
            self.returncode = 0
            return 0

    process = InterruptedProcess()
    original_cleanup = gates._cleanup_exact_transaction

    def cleanup(*args: object, **kwargs: object) -> list[str]:
        events.append("cleanup")
        assert process.reaped
        assert not process.running
        return original_cleanup(*args, **kwargs)

    monkeypatch.setattr(gates, "_cleanup_exact_transaction", cleanup)
    with pytest.raises(KeyboardInterrupt, match="stop requested"):
        gates.execute_exact_profile_transaction(
            specs,
            repo=gates.REPO_ROOT,
            python_executable="/env/bin/python",
            transaction_dir=transaction,
            popen_factory=lambda *args, **kwargs: process,
            compare=compare,
        )

    assert events == expected_events
    assert process.reaped
    assert not transaction.exists()


def test_exact_transaction_reports_termination_failure_with_original_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, transaction, _, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )
    events: list[str] = []

    class StopSignal(BaseException):
        pass

    class TerminateFailureProcess:
        pid = 31_339
        returncode = None

        def __init__(self) -> None:
            self.running = True
            self.reaped = False

        def communicate(self) -> tuple[bytes, bytes]:
            events.append("communicate")
            raise StopSignal("original interrupt")

        def poll(self) -> int | None:
            events.append("poll")
            return None if self.running else 0

        def terminate(self) -> None:
            events.append("terminate")
            raise OSError("terminate denied")

        def kill(self) -> None:
            events.append("kill")
            self.running = False

        def wait(self, timeout: float | None = None) -> int:
            events.append("wait")
            assert timeout is not None
            assert not self.running
            self.reaped = True
            self.returncode = 0
            return 0

    process = TerminateFailureProcess()
    original_cleanup = gates._cleanup_exact_transaction

    def cleanup(*args: object, **kwargs: object) -> list[str]:
        events.append("cleanup")
        assert process.reaped
        return original_cleanup(*args, **kwargs)

    monkeypatch.setattr(gates, "_cleanup_exact_transaction", cleanup)
    with pytest.raises(
        gates.GateVerificationError, match="process.*recovery|terminate denied"
    ) as caught:
        gates.execute_exact_profile_transaction(
            specs,
            repo=gates.REPO_ROOT,
            python_executable="/env/bin/python",
            transaction_dir=transaction,
            popen_factory=lambda *args, **kwargs: process,
            compare=compare,
        )

    assert isinstance(caught.value.__cause__, StopSignal)
    assert events == ["communicate", "poll", "terminate", "kill", "wait", "cleanup"]
    assert process.reaped
    assert not transaction.exists()


def test_exact_transaction_reap_failure_preserves_paths_and_closes_all_witness_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, transaction, successful_popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )
    baseline_fds = len(os.listdir("/proc/self/fd"))
    launches = 0
    owned_fds: set[int] = set()
    close_calls: list[int] = []
    original_close = os.close
    preserved_paths = {
        str(transaction),
        str(transaction / "receipts"),
        str(Path(specs[0]["output_root"])),
        str(transaction / "receipts/000-reference.json"),
    }

    class StopSignal(BaseException):
        pass

    class UnreapableProcess:
        pid = 31_340
        returncode = None

        def communicate(self) -> tuple[bytes, bytes]:
            raise StopSignal("communication interrupted")

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            raise OSError("terminate denied")

        def kill(self) -> None:
            for entry in os.listdir("/proc/self/fd"):
                try:
                    target = os.readlink(f"/proc/self/fd/{entry}")
                except OSError:
                    continue
                if target in preserved_paths:
                    owned_fds.add(int(entry))
            assert len(owned_fds) == len(preserved_paths)
            injected_failure_fd = min(owned_fds)

            def close_with_one_reported_failure(descriptor: int) -> None:
                close_calls.append(descriptor)
                original_close(descriptor)
                if descriptor == injected_failure_fd:
                    raise OSError("injected witness close failure")

            monkeypatch.setattr(gates.os, "close", close_with_one_reported_failure)
            raise OSError("kill denied")

    def popen(argv: list[str], **kwargs: object) -> object:
        nonlocal launches
        launches += 1
        if launches == 1:
            return successful_popen(argv, **kwargs)
        return UnreapableProcess()

    try:
        with pytest.raises(
            gates.GateVerificationError,
            match="process.*reap|witness close.*injected witness close failure",
        ) as caught:
            gates.execute_exact_profile_transaction(
                specs,
                repo=gates.REPO_ROOT,
                python_executable="/env/bin/python",
                transaction_dir=transaction,
                popen_factory=popen,
                compare=compare,
            )

        assert "injected witness close failure" in str(caught.value)
        assert isinstance(caught.value.__cause__, StopSignal)
        assert len(close_calls) == len(owned_fds) == 4
        assert set(close_calls) == owned_fds
        assert transaction.is_dir()
        assert (transaction / "receipts").is_dir()
        assert Path(specs[0]["output_root"]).is_dir()
        assert (transaction / "receipts/000-reference.json").is_file()
        gc.collect()
        assert len(os.listdir("/proc/self/fd")) - baseline_fds == 0
    finally:
        monkeypatch.setattr(gates.os, "close", original_close)
        for descriptor in owned_fds:
            try:
                target = os.readlink(f"/proc/self/fd/{descriptor}")
            except OSError:
                continue
            if target in preserved_paths:
                original_close(descriptor)


@pytest.mark.parametrize("position", [0, 4, 8])
def test_exact_transaction_child_failure_cleans_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, position: int
) -> None:
    specs, transaction, popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=position
    )
    kwargs = {
        "repo": gates.REPO_ROOT,
        "python_executable": "/env/bin/python",
        "transaction_dir": transaction,
        "popen_factory": popen,
        "compare": compare,
    }
    with pytest.raises(gates.GateVerificationError, match="child failed"):
        gates.execute_exact_profile_transaction(specs, **kwargs)
    assert not transaction.exists()
    assert all(not Path(spec["output_root"]).exists() for spec in specs)

    result = gates.execute_exact_profile_transaction(specs, **kwargs)
    assert len(result["executions"]) == len(specs)


def test_exact_transaction_compare_failure_cleans_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, transaction, popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="compare", position=4
    )
    kwargs = {
        "repo": gates.REPO_ROOT,
        "python_executable": "/env/bin/python",
        "transaction_dir": transaction,
        "popen_factory": popen,
        "compare": compare,
    }
    with pytest.raises((gates.GateVerificationError, gates.ArtifactMismatch)):
        gates.execute_exact_profile_transaction(specs, **kwargs)
    assert not transaction.exists()
    assert all(not Path(spec["output_root"]).exists() for spec in specs)
    assert len(gates.execute_exact_profile_transaction(specs, **kwargs)["executions"]) == 9


def test_exact_transaction_receipt_failure_preserves_unproven_root_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, transaction, popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="receipt", position=4
    )
    with pytest.raises(gates.GateVerificationError, match="cleanup unsafe"):
        gates.execute_exact_profile_transaction(
            specs, repo=gates.REPO_ROOT, python_executable="/env/bin/python",
            transaction_dir=transaction, popen_factory=popen, compare=compare,
        )
    assert transaction.exists()
    assert Path(specs[4]["output_root"]).exists()
    assert all(Path(specs[index]["output_root"]).exists() for index in range(5))


def test_exact_transaction_preserves_preexisting_transaction_and_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, transaction, popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=0
    )
    transaction.mkdir()
    (transaction / "owner.txt").write_text("keep")
    with pytest.raises(gates.GateVerificationError, match="clobber"):
        gates.execute_exact_profile_transaction(
            specs, repo=gates.REPO_ROOT, python_executable="/env/bin/python",
            transaction_dir=transaction, popen_factory=popen, compare=compare,
        )
    assert (transaction / "owner.txt").read_text() == "keep"

    shutil.rmtree(transaction)
    root = Path(specs[0]["output_root"])
    root.mkdir()
    (root / "owner.txt").write_text("keep")
    with pytest.raises(gates.GateVerificationError, match="clobber"):
        gates.execute_exact_profile_transaction(
            specs, repo=gates.REPO_ROOT, python_executable="/env/bin/python",
            transaction_dir=transaction, popen_factory=popen, compare=compare,
        )
    assert (root / "owner.txt").read_text() == "keep"
    assert not transaction.exists()


@pytest.mark.parametrize("replacement_kind", ["directory", "symlink"])
def test_exact_transaction_does_not_delete_replaced_root_and_preserves_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement_kind: str
) -> None:
    specs, transaction, popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="receipt", position=1
    )
    original_reopen = gates._reopen_completed_execution
    identities: dict[str, tuple[int, int, int]] = {}

    def replace_then_fail(*args: object, **kwargs: object) -> dict[str, object]:
        root = args[1]
        assert isinstance(root, Path)
        if root.name.endswith("-1"):
            shutil.rmtree(root)
            if replacement_kind == "directory":
                root.mkdir()
                (root / "replacement.txt").write_text("keep")
            else:
                outside = tmp_path / "replacement-target"
                outside.mkdir()
                (outside / "replacement.txt").write_text("keep")
                root.symlink_to(outside, target_is_directory=True)
            metadata = os.lstat(root)
            identities["replacement"] = (
                metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)
            )
        else:
            metadata = os.lstat(root)
            identities["earlier"] = (
                metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)
            )
        return original_reopen(*args, **kwargs)

    monkeypatch.setattr(gates, "_reopen_completed_execution", replace_then_fail)
    with pytest.raises(gates.GateVerificationError, match="cleanup unsafe"):
        gates.execute_exact_profile_transaction(
            specs, repo=gates.REPO_ROOT, python_executable="/env/bin/python",
            transaction_dir=transaction, popen_factory=popen, compare=compare,
        )
    replacement = Path(specs[1]["output_root"])
    assert (replacement / "replacement.txt").read_text() == "keep"
    roots = [Path(specs[index]["output_root"]) for index in (0, 1)]
    assert all(root.exists() for root in roots)
    assert tuple(
        (item.st_dev, item.st_ino, stat.S_IFMT(item.st_mode))
        for item in map(os.lstat, roots)
    ) == (identities["earlier"], identities["replacement"])
    assert transaction.exists()
    observations = sorted((transaction / "receipts").glob("*.json"))
    assert [path.name for path in observations] == ["000-reference.json"]


@pytest.mark.parametrize("replacement_kind", ["file", "symlink"])
def test_exact_transaction_preserves_replaced_observation_and_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement_kind: str
) -> None:
    specs, transaction, popen, original_compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )
    replaced = False
    root_identities: dict[Path, tuple[int, int, int]] = {}
    observation_identities: dict[Path, tuple[int, int, int]] = {}

    def compare(left: Path, right: Path, **kwargs: object) -> dict[str, object]:
        nonlocal replaced
        result = original_compare(left, right)
        observation = transaction / "receipts/000-reference.json"
        later_observation = transaction / "receipts/001-a0.json"
        if later_observation.exists() and not replaced:
            replaced = True
            observation.unlink()
            if replacement_kind == "file":
                observation.write_text("replacement")
            else:
                outside = tmp_path / "outside-observation.json"
                outside.write_text("outside")
                observation.symlink_to(outside)
            for root in (Path(specs[0]["output_root"]), Path(specs[1]["output_root"])):
                metadata = os.lstat(root)
                root_identities[root] = (
                    metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)
                )
            for receipt in (observation, later_observation):
                metadata = os.lstat(receipt)
                observation_identities[receipt] = (
                    metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)
                )
            raise gates.ArtifactMismatch("compare failed after receipt replacement")
        return result

    with pytest.raises(gates.GateVerificationError, match="cleanup unsafe"):
        gates.execute_exact_profile_transaction(
            specs, repo=gates.REPO_ROOT, python_executable="/env/bin/python",
            transaction_dir=transaction, popen_factory=popen, compare=compare,
        )
    observation = transaction / "receipts/000-reference.json"
    assert observation.exists()
    assert observation.read_text() in {"replacement", "outside"}
    assert transaction.exists()
    assert len(list((transaction / "receipts").iterdir())) == 2
    for path, identity in {**root_identities, **observation_identities}.items():
        metadata = os.lstat(path)
        assert (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)) == identity


@pytest.mark.parametrize("mutation", ["insert", "replace"])
def test_exact_transaction_nested_root_mutation_preserves_every_owned_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    specs, transaction, original_popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )

    def popen(argv: list[str], **kwargs: object) -> object:
        process = original_popen(argv, **kwargs)
        nested = Path(argv[5]) / "nested"
        nested.mkdir()
        (nested / "known.bin").write_bytes(b"known")
        return process

    compare_calls = 0

    def mutate_then_fail(
        left: Path, right: Path, **kwargs: object
    ) -> dict[str, object]:
        nonlocal compare_calls
        result = compare(left, right, **kwargs)
        compare_calls += 1
        if compare_calls == 2:
            nested = Path(specs[0]["output_root"]) / "nested"
            target = nested / ("extra.bin" if mutation == "insert" else "known.bin")
            if mutation == "replace":
                target.unlink()
            target.write_bytes(b"attacker")
            raise gates.ArtifactMismatch("compare failed after nested mutation")
        return result

    with pytest.raises(gates.GateVerificationError, match="cleanup unsafe"):
        gates.execute_exact_profile_transaction(
            specs, repo=gates.REPO_ROOT, python_executable="/env/bin/python",
            transaction_dir=transaction, popen_factory=popen, compare=mutate_then_fail,
        )

    roots = [Path(specs[index]["output_root"]) for index in (0, 1)]
    assert sum(root.exists() for root in roots) == 2
    assert transaction.exists()
    assert sorted(path.name for path in (transaction / "receipts").iterdir()) == [
        "000-reference.json", "001-a0.json",
    ]
    nested_files = sorted(path.name for path in (roots[0] / "nested").iterdir())
    assert nested_files == (["extra.bin", "known.bin"] if mutation == "insert" else ["known.bin"])


def test_exact_transaction_popen_start_failure_cleans_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, transaction, successful_popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )
    failed = False

    def popen(argv: list[str], **kwargs: object) -> object:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("popen failed")
        return successful_popen(argv, **kwargs)

    kwargs = {
        "repo": gates.REPO_ROOT,
        "python_executable": "/env/bin/python",
        "transaction_dir": transaction,
        "popen_factory": popen,
        "compare": compare,
    }
    with pytest.raises(OSError, match="popen failed"):
        gates.execute_exact_profile_transaction(specs, **kwargs)
    assert not transaction.exists()
    assert all(not Path(spec["output_root"]).exists() for spec in specs)
    assert len(gates.execute_exact_profile_transaction(specs, **kwargs)["executions"]) == 9


@pytest.mark.parametrize("marker", ["partial-child", "other-owner-race"])
def test_exact_transaction_nonzero_unproven_root_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    specs, transaction, successful_popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )
    failed = False

    class FailedProcess:
        returncode = 9
        pid = 31_337

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"child failed"

    def popen(argv: list[str], **kwargs: object) -> object:
        nonlocal failed
        if not failed:
            failed = True
            root = Path(argv[5])
            root.mkdir()
            (root / "owner.txt").write_text(marker)
            return FailedProcess()
        return successful_popen(argv, **kwargs)

    with pytest.raises(gates.GateVerificationError, match="cleanup unsafe"):
        gates.execute_exact_profile_transaction(
            specs, repo=gates.REPO_ROOT, python_executable="/env/bin/python",
            transaction_dir=transaction, popen_factory=popen, compare=compare,
        )
    root = Path(specs[0]["output_root"])
    assert (root / "owner.txt").read_text() == marker
    assert transaction.exists()


def test_exact_transaction_mid_spec_failure_cleans_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs, transaction, popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure="child", position=99
    )
    source = specs[4].pop("source_manifest")
    kwargs = {
        "repo": gates.REPO_ROOT,
        "python_executable": "/env/bin/python",
        "transaction_dir": transaction,
        "popen_factory": popen,
        "compare": compare,
    }
    with pytest.raises(gates.GateVerificationError, match="spec fields"):
        gates.execute_exact_profile_transaction(specs, **kwargs)
    assert not transaction.exists()
    assert all(not Path(spec["output_root"]).exists() for spec in specs)
    specs[4]["source_manifest"] = source
    assert len(gates.execute_exact_profile_transaction(specs, **kwargs)["executions"]) == 9


@pytest.mark.parametrize("mutation", ["duplicate_root", "ancestor_root", "argv", "receipt"])
def test_exact_profile_gate_rejects_root_alias_argv_and_receipt_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    executions = [
        _exact_execution(profile, tmp_path / f"run-{index}", 100 + index)
        for index, profile in enumerate(gates.EXACT_PROFILE_SEQUENCE)
    ]
    compare = lambda left, right, **kwargs: {
        "format": "oviv2_cumulative_exact_v1",
        "checkpoint_frames": [2, 7],
        "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
        "root_sha256": "e" * 64,
    }
    if mutation == "duplicate_root":
        executions[2]["output_root"] = executions[1]["output_root"]
    elif mutation == "ancestor_root":
        executions[2]["output_root"] = str(
            (Path(executions[1]["output_root"]) / "child").resolve()
        )
    elif mutation == "argv":
        executions[2]["argv"] = ["python", "runner.py"]
    else:
        receipt = Path(executions[2]["output_root"]) / "execution_receipt.json"
        receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(gates.GateVerificationError, match="root|argv|receipt"):
        gates.verify_exact_profile_runs(
            executions,
            compare=compare,
            schema1_variants=PRODUCTION_SCHEMA1_VARIANTS,
        )


@pytest.mark.parametrize(
    "field",
    ["argv", "code_commit", "source_manifest_sha256", "common_input_fingerprints"],
)
def test_exact_profile_gate_rejects_incomplete_or_disagreeing_bindings(
    tmp_path: Path, field: str
) -> None:
    profiles = ("reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4")
    executions = [
        _exact_execution(profile, tmp_path / f"run-{index}", 100 + index)
        for index, profile in enumerate(profiles)
    ]
    compare = lambda left, right, **kwargs: {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2, 7],
            "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
            "root_sha256": "e" * 64,
        }
    if field == "argv":
        executions[3][field] = []
    elif field == "common_input_fingerprints":
        executions[3][field] = {"source_manifest_sha256": "f" * 64}
    else:
        executions[3][field] = "f" * (40 if field == "code_commit" else 64)
    with pytest.raises(gates.GateVerificationError, match="binding|argv"):
        gates.verify_exact_profile_runs(
            executions,
            compare=compare,
            schema1_variants=PRODUCTION_SCHEMA1_VARIANTS,
        )


def test_reference_worker_records_exact_process_and_artifact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schedule_sha256": "1" * 64}))
    source = tmp_path / "sources.json"
    source.write_text("{}\n")
    output = tmp_path / "run"
    receipt = output / "t1_exact_receipt.json"
    calls: list[tuple[Path, Path, Path, str]] = []

    def runner(
        config_path: Path,
        output_path: Path,
        *,
        freeze_manifest: Path,
        run_slot: str,
    ) -> dict[str, object]:
        calls.append((config_path, output_path, freeze_manifest, run_slot))
        output_path.mkdir()
        return {}

    monkeypatch.setattr(reference_worker, "_commit", lambda repo: "a" * 40)
    monkeypatch.setattr(reference_worker.os, "getpid", lambda: 4321)
    def compare_reference(
        left: Path, right: Path, **kwargs: object
    ) -> dict[str, object]:
        assert (left, right) == (output.resolve(), output.resolve())
        assert kwargs == {
            "validation_stage": "pre_legacy",
            "left_schema1_variant": "production",
            "right_schema1_variant": "production",
        }
        return {
                "format": "oviv2_cumulative_exact_v1",
                "checkpoint_frames": [2, 7],
                "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
                "root_sha256": "e" * 64,
        }

    monkeypatch.setattr(
        reference_worker, "compare_cumulative_artifacts", compare_reference
    )
    argv = [
        "/env/bin/python", str(Path(reference_worker.__file__).resolve()),
        "--config", str(config.resolve()), "--output", str(output.resolve()),
        "--freeze-manifest", str((tmp_path / "freeze.json").resolve()),
        "--run-slot", "apartment_run1", "--receipt", str(receipt.resolve()),
        "--source-manifest", str(source.resolve()),
    ]
    payload = reference_worker.run_reference(
        config=config,
        output=output,
        freeze_manifest=tmp_path / "freeze.json",
        run_slot="apartment_run1",
        receipt=receipt,
        argv=argv,
        repo=tmp_path,
        source_manifest=source,
        runner=runner,
    )
    assert calls == [(config, output, tmp_path / "freeze.json", "apartment_run1")]
    assert payload["execution"]["argv"] == argv
    assert payload["execution"]["pid"] == 4321
    assert payload["execution"]["input_fingerprints"] == {
        "config": hashlib.sha256(config.read_bytes()).hexdigest(),
        "schedule_sha256": "1" * 64,
    }
    assert payload["format"] == "oviv2_t1_exact_execution_receipt_v1"
    assert payload["cumulative_root_sha256"] == "e" * 64
    assert json.loads(receipt.read_text()) == payload


@pytest.mark.parametrize("field", ["path", "sha256", "byte_count"])
def test_exact_receipt_rejects_source_manifest_record_drift(
    tmp_path: Path, field: str
) -> None:
    record = _exact_execution("reference", tmp_path / "run", 101)
    receipt_path = Path(record["output_root"]) / "t1_exact_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if field == "path":
        receipt["source_manifest"][field] = str((tmp_path / "other.json").resolve())
    elif field == "sha256":
        receipt["source_manifest"][field] = "f" * 64
    else:
        receipt["source_manifest"][field] += 1
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    record["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(
        gates.GateVerificationError, match="source manifest|production_receipt"
    ):
            gates._bind_exact_receipt(
                record,
                schema1_variant="t1_transaction",
            compare=lambda left, right, **kwargs: {
                "format": "oviv2_cumulative_exact_v1",
                "checkpoint_frames": [2, 7],
                "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
                "root_sha256": "e" * 64,
            },
        )
