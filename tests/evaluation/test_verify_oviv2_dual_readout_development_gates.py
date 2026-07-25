from __future__ import annotations

import importlib.util
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
    for relative in (*gates.PROTECTED_FILES, *gates.TEST_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode())


class FakeGit:
    def __init__(self, *, dirty: bool = False, protected_diff: bool = False) -> None:
        self.dirty = dirty
        self.protected_diff = protected_diff

    def __call__(self, argv: tuple[str, ...], cwd: Path) -> bytes:
        del cwd
        if argv == ("status", "--porcelain", "--untracked-files=all"):
            return b" M dirty\n" if self.dirty else b""
        if argv == ("rev-parse", "HEAD"):
            return ("a" * 40 + "\n").encode()
        if argv == ("rev-parse", "HEAD^{tree}"):
            return ("b" * 40 + "\n").encode()
        if argv[:2] == ("diff", "--quiet"):
            if self.protected_diff:
                raise subprocess.CalledProcessError(1, ("git", *argv))
            return b""
        raise AssertionError(argv)


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
        "base_commit": gates.PLANNED_BASE_COMMIT,
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
    assert deterministic["base_commit"] == gates.PLANNED_BASE_COMMIT
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
        ({"base_commit": "c" * 40}, "planned base"),
        ({"git": FakeGit(dirty=True)}, "clean"),
        ({"git": FakeGit(protected_diff=True)}, "protected"),
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
            base_commit=gates.PLANNED_BASE_COMMIT,
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
    protected = repo / gates.PROTECTED_FILES[0]
    protected.unlink()
    protected.symlink_to(repo / gates.PROTECTED_FILES[1])
    with pytest.raises(gates.GateVerificationError, match="regular non-symlink"):
        gates.generate_evidence(
            repo=repo,
            output=tmp_path / "gate.json",
            base_commit=gates.PLANNED_BASE_COMMIT,
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
            base_commit=gates.PLANNED_BASE_COMMIT,
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
            base_commit=gates.PLANNED_BASE_COMMIT,
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
