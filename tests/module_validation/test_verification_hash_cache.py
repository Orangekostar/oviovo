"""Verification may reuse unchanged bytes but must detect file mutations."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.evaluation.verification_cache.file_hash_cache import StatAwareSha256


def _reader(calls):
    def read(path):
        calls.append(Path(path))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return read


def test_unchanged_file_is_hashed_once_and_deleted_file_is_not_reused(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"aaaa")
    time.sleep(2.01)
    calls = []
    cache = StatAwareSha256(_reader(calls))
    assert cache(path) == cache(str(path)) == hashlib.sha256(b"aaaa").hexdigest()
    assert calls == [path]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        cache(path)


def test_same_size_edit_with_restored_mtime_invalidates_cache(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"aaaa")
    time.sleep(2.01)
    before = path.stat()
    calls = []
    cache = StatAwareSha256(_reader(calls))
    original = cache(path)
    path.write_bytes(b"bbbb")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cache(path) == hashlib.sha256(b"bbbb").hexdigest() != original
    assert len(calls) == 3


def test_recent_files_are_rechecked_even_when_stat_can_share_a_clock_tick(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"aaaa")
    before = path.stat()
    calls = []
    cache = StatAwareSha256(_reader(calls))
    original = cache(path)
    path.write_bytes(b"bbbb")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cache(path) == hashlib.sha256(b"bbbb").hexdigest() != original
    assert len(calls) == 4


def test_atomic_replacement_with_same_size_and_mtime_invalidates_cache(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"aaaa")
    time.sleep(2.01)
    before = path.stat()
    calls = []
    cache = StatAwareSha256(_reader(calls))
    cache(path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"bbbb")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    assert cache(path) == hashlib.sha256(b"bbbb").hexdigest()
    assert len(calls) == 3


def test_mutation_during_hashing_fails_without_caching_partial_identity(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"aaaa")
    calls = []

    def changing_reader(source):
        result = _reader(calls)(source)
        if len(calls) == 1:
            path.write_bytes(b"bbbb")
        return result

    cache = StatAwareSha256(changing_reader)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        cache(path)
    reads_after_mutation = len(calls)
    assert cache(path) == hashlib.sha256(b"bbbb").hexdigest()
    assert len(calls) == reads_after_mutation + 2


def test_cache_has_bounded_lru_storage(tmp_path):
    paths = [tmp_path / name for name in ("a", "b", "c")]
    for path in paths:
        path.write_text(path.name)
    time.sleep(2.01)
    calls = []
    cache = StatAwareSha256(_reader(calls), max_entries=2)
    for index in (0, 1, 0, 2, 1):
        cache(paths[index])
    assert calls == [paths[index] for index in (0, 1, 2, 1)]


@pytest.mark.parametrize("enabled", ("0", "1"))
def test_bootstrap_is_explicit_and_available_to_child_processes(enabled):
    root = Path(__file__).resolve().parents[2]
    bootstrap = root / "scripts/evaluation/verification_cache"
    env = {**os.environ, "OVIMAP_VERIFY_HASH_CACHE": enabled,
           "PYTHONPATH": os.pathsep.join((str(bootstrap), str(root)))}
    code = "from src.static_ovmap.module_validation.assets import sha256_file; print(type(sha256_file).__name__)"
    output = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=root, text=True).strip()
    assert output == ("StatAwareSha256" if enabled == "1" else "function")
