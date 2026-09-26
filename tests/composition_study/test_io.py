"""Content mutation and immutable output guards for the new study."""

import hashlib

import pytest


def test_source_identity_rechecks_changed_same_size_file(tmp_path):
    from src.static_ovmap.composition_study.io import SourceIndex

    path = tmp_path / "source"
    path.write_bytes(b"first")
    index = SourceIndex()
    original = index.identity(path)
    assert original["sha256"] == hashlib.sha256(b"first").hexdigest()
    path.write_bytes(b"other")
    assert index.identity(path)["sha256"] == hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError, match="identity"):
        index.identity(path, expected=original)


def test_write_once_preserves_completed_results(tmp_path):
    from src.static_ovmap.composition_study.io import read_json, write_once

    path = tmp_path / "receipt.json"
    write_once(path, {"candidate": "M1"})
    write_once(path, {"candidate": "M1"})
    with pytest.raises(ValueError, match="immutable"):
        write_once(path, {"candidate": "M2"})
    assert read_json(path) == {"candidate": "M1"}
