from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts/evaluation/verify_oviv2_dual_readout_development_gates.py"
)
SPEC = importlib.util.spec_from_file_location("development_gates", SCRIPT)
assert SPEC and SPEC.loader
gates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gates
SPEC.loader.exec_module(gates)


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
    kwargs = {
        "repo": repo,
        "output": output,
        "base_commit": gates.CUMULATIVE_BASE_COMMIT,
        "python_executable": "/env/bin/python",
        "git": FakeGit(),
        "run": runner,
        "now_utc": lambda: "2026-07-25T00:00:00Z",
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
