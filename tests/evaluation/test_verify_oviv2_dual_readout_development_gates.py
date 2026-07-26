from __future__ import annotations

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
            exact_runs, compare=kwargs["compare"]
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
        "compare": lambda left, right: {
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
    config_path = (root.parent / f"{profile}.json").resolve()
    config_path.write_text(json.dumps({"algorithm_hash": "d" * 64, "profile": profile}) + "\n")
    freeze_path = (root.parent / "freeze.json").resolve()
    if not freeze_path.exists():
        freeze_path.write_text(
            json.dumps(
                {
                    "shared_bindings": {
                        "input_manifest": {"sha256": "1" * 64},
                        "schedule": {"sha256": "2" * 64},
                        "occlusion_target_manifest": {"sha256": "3" * 64},
                    }
                },
                sort_keys=True,
            )
            + "\n"
        )
    argv = [
        "/env/bin/python", str(runner), "--config", str(config_path),
        "--output", str(root.resolve()), "--freeze-manifest", str(freeze_path),
        "--run-slot", "apartment_run1",
    ]
    if profile == "reference":
        argv += ["--receipt", str((root / "t1_exact_receipt.json").resolve()),
                 "--source-manifest", str(source_path)]
    run_manifest = {
        "schema_version": 1 if profile == "reference" else 2,
        "algorithm_hash": "d" * 64,
        "code_commit": commit,
        "source_bindings": {"dataset": "fixture", "cache": "shared"},
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
            "argv": argv,
            "pid": pid,
            "code_commit": commit,
            "source_manifest_sha256": actual_source_sha,
            "input_fingerprints": inputs or {"dataset": "c" * 64},
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
            "provenance": {"repository_commit": commit},
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
        "freeze_manifest": gates._absolute_file_record(freeze_path),
        "source_manifest": gates._absolute_file_record(source_path),
        "output_root": str(root.resolve()),
        "root_device": status.st_dev,
        "root_inode": status.st_ino,
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
    calls: list[tuple[Path, Path]] = []

    def compare(left: Path, right: Path) -> dict[str, object]:
        calls.append((left, right))
        return {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2, 7],
            "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
            "root_sha256": "e" * 64,
        }

    evidence = gates.verify_exact_profile_runs(executions, compare=compare)
    assert evidence["sequence"] == list(profiles)
    assert set(evidence["profiles"]) == {"a0", "a1", "a2", "a3", "a4"}
    assert all(
        profile["cumulative_root_sha256"] == "e" * 64
        for profile in evidence["profiles"].values()
    )
    assert len(calls) == 17
    assert calls[9] == (tmp_path / "run-0", tmp_path / "run-1")

    executions[1]["pid"] = executions[0]["pid"]
    observation_path = Path(executions[1]["observation_receipt"]["path"])
    observation = json.loads(observation_path.read_text())
    observation["pid"] = executions[1]["pid"]
    observation["completed_execution"]["pid"] = executions[1]["pid"]
    observation_path.write_text(json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n")
    executions[1]["observation_receipt"] = gates._absolute_file_record(observation_path)
    with pytest.raises(gates.GateVerificationError, match="PID"):
        gates.verify_exact_profile_runs(executions, compare=compare)


def test_exact_transaction_uses_popen_pid_argv_and_returncode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    freeze = tmp_path / "freeze.json"
    source = tmp_path / "source.json"
    for path in (config, freeze, source):
        path.write_text("{}\n")
    specs = [
        {
            "profile": profile,
            "config": str(config.resolve()),
            "output_root": str((tmp_path / f"run-{position}").resolve()),
            "freeze_manifest": str(freeze.resolve()),
            "run_slot": "apartment_run1",
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
        receipt_name = "t1_exact_receipt.json" if len(argv) == 14 else "execution_receipt.json"
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

    result = gates.execute_exact_profile_transaction(
        specs,
        repo=gates.REPO_ROOT,
        python_executable="/env/bin/python",
        transaction_dir=tmp_path / "transaction",
        popen_factory=popen,
        compare=lambda left, right: {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2],
            "inventory": [{"path": "x", "sha256": "2" * 64, "byte_count": 1}],
            "root_sha256": "3" * 64,
        },
    )

    assert result["executions"] == seen
    assert [record["pid"] for record in seen] == [pid for _, pid in launched]
    assert all(record["returncode"] == 0 for record in seen)
    assert [record["profile"] for record in seen] == list(gates.EXACT_PROFILE_SEQUENCE)
    observations = sorted((tmp_path / "transaction/receipts").glob("*.json"))
    assert len(observations) == 9
    assert [json.loads(path.read_text())["pid"] for path in observations] == [
        pid for _, pid in launched
    ]


def _failure_cleanup_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str,
    position: int,
) -> tuple[list[dict[str, object]], Path, object, object]:
    config = tmp_path / "config.json"
    freeze = tmp_path / "freeze.json"
    source = tmp_path / "source.json"
    for path in (config, freeze, source):
        path.write_text("{}\n")
    specs = [
        {
            "profile": profile,
            "config": str(config.resolve()),
            "output_root": str((tmp_path / f"cleanup-run-{index}").resolve()),
            "freeze_manifest": str(freeze.resolve()),
            "run_slot": "apartment_run1",
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
        output = Path(argv[5])
        output.mkdir()
        (output / "run_manifest.json").write_text("{}\n")
        receipt = output / (
            "t1_exact_receipt.json" if len(argv) == 14 else "execution_receipt.json"
        )
        receipt.write_text("{}\n")
        should_fail = failure == "child" and current == position and not failed
        if should_fail:
            failed = True
        return Process(9 if should_fail else 0, 20_000 + launches)

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
                "config_sha256": "c" * 64,
                "freeze_manifest_sha256": "d" * 64,
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

    def compare(left: Path, right: Path) -> dict[str, object]:
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


@pytest.mark.parametrize("failure", ["compare", "receipt"])
def test_exact_transaction_validation_failure_cleans_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    specs, transaction, popen, compare = _failure_cleanup_harness(
        tmp_path, monkeypatch, failure=failure, position=4
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

    def compare(left: Path, right: Path) -> dict[str, object]:
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
            output = Path(argv[5])
            output.mkdir()
            (output / "partial.txt").write_text("partial")
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
    compare = lambda left, right: {
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
        gates.verify_exact_profile_runs(executions, compare=compare)


@pytest.mark.parametrize("field", ["argv", "code_commit", "source_manifest_sha256", "input_fingerprints"])
def test_exact_profile_gate_rejects_incomplete_or_disagreeing_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    profiles = ("reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4")
    executions = [
        _exact_execution(profile, tmp_path / f"run-{index}", 100 + index)
        for index, profile in enumerate(profiles)
    ]
    compare = lambda left, right: {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2, 7],
            "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
            "root_sha256": "e" * 64,
        }
    if field == "argv":
        executions[3][field] = []
    elif field == "input_fingerprints":
        executions[3][field] = {"dataset": "f" * 64}
    else:
        executions[3][field] = "f" * (40 if field == "code_commit" else 64)
    with pytest.raises(gates.GateVerificationError, match="binding|argv"):
        gates.verify_exact_profile_runs(executions, compare=compare)


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
    monkeypatch.setattr(
        reference_worker,
        "compare_cumulative_artifacts",
        lambda left, right: {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": [2, 7],
            "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
            "root_sha256": "e" * 64,
        },
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
            compare=lambda left, right: {
                "format": "oviv2_cumulative_exact_v1",
                "checkpoint_frames": [2, 7],
                "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
                "root_sha256": "e" * 64,
            },
        )
