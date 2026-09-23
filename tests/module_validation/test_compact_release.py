"""Lossless packaging checks; these fixtures are not scientific measurements."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/static_ovmap/module_validation_v1"
    / "finalization-20260923-native-v10/compact_release.py"
)


def load_compactor():
    spec = importlib.util.spec_from_file_location("compact_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compact_release


def make_release(root):
    root.mkdir()
    request = b'{"mapping":{"label":"chair"},"padding":"' + b"x" * 4000 + b'"}'
    payloads = {
        "study/requests/one.json": request,
        "selection.json": b'{"candidate":"N0"}',
        "study/query/Q_GAIN/checkpoint.pt": b"exact checkpoint bytes",
    }
    for name, blob in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    source = {
        "source": {
            "path": "/external/original.json",
            "bytes": 5000,
            "sha256": "a" * 64,
        },
        "transformation": "request_ledger_without_class_similarity_vector",
    }
    manifest = {
        "schema_version": 2,
        "source_attempt": "/external/attempt",
        "cache_key": "frozen-cache-key",
        "files": {
            name: hashlib.sha256(blob).hexdigest() for name, blob in payloads.items()
        },
        "source_files": {"study/requests/one.json": source},
        "maximum_bytes": 256 * 1024 * 1024,
        "total_bytes": 0,
    }
    size = sum(map(len, payloads.values()))
    while manifest["total_bytes"] != size + len(json.dumps(manifest).encode()):
        manifest["total_bytes"] = size + len(json.dumps(manifest).encode())
    (root / "export_manifest.json").write_text(json.dumps(manifest))
    return payloads, source


def test_small_ledgers_compress_losslessly_with_source_provenance(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    payloads, provenance = make_release(source)
    manifest = load_compactor()(source, destination)
    compressed = "study/requests/one.json.gz"
    assert (
        gzip.decompress((destination / compressed).read_bytes())
        == payloads["study/requests/one.json"]
    )
    assert not (destination / "study/requests/one.json").exists()
    assert manifest["source_files"][compressed]["source"] == provenance["source"]
    assert (
        manifest["source_files"][compressed]["transformation"]
        == provenance["transformation"]
    )
    assert (
        manifest["source_files"][compressed]["decoded_sha256"]
        == hashlib.sha256(payloads["study/requests/one.json"]).hexdigest()
    )
    for name in ("selection.json", "study/query/Q_GAIN/checkpoint.pt"):
        assert (destination / name).read_bytes() == payloads[name]
    assert manifest["total_bytes"] == sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    assert manifest["maximum_bytes"] == 100 * 1024 * 1024


def test_changed_input_rejected_before_destination_creation(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    make_release(source)
    (source / "selection.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_compactor()(source, destination)
    assert not destination.exists()


def test_size_cap_rejected_before_destination_creation(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    make_release(source)
    with pytest.raises(ValueError, match="size limit"):
        load_compactor()(source, destination, maximum_bytes=100)
    assert not destination.exists()
