"""Read-only relocation and memoized input changes at the filesystem boundary."""

import json

import pytest


def test_path_resolution_returns_new_paths_without_editing_old_receipts(tmp_path):
    from static_ovmap.recovery_wave2.binding import PathResolver

    receipt = tmp_path / "historical.json"
    original = {"array": "/old/data/x.npz", "identity": "unchanged", "label": "old"}
    receipt.write_text(json.dumps(original))
    before = receipt.read_bytes()
    resolver = PathResolver({"/old": str(tmp_path / "new"), "/old/data": str(tmp_path / "dense")})
    assert resolver.rewrite(original)["array"] == str(tmp_path / "dense/x.npz")
    assert resolver.resolve("/other/data") == "/other/data"
    assert resolver.resolve("/older/data") == "/older/data"
    assert receipt.read_bytes() == before
    assert original["array"] == "/old/data/x.npz"


def test_consumption_memo_revalidates_modified_file_against_original_hash(tmp_path):
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    source = tmp_path / "array.bin"
    source.write_bytes(b"original")
    first = ConsumptionIndex()
    original = first.identity(source)
    memo = tmp_path / "memo.json"
    first.write_memo(memo)
    second = ConsumptionIndex(memo)
    assert second.identity(source, original) == original
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="bound input changed"):
        second.identity(source, original)
