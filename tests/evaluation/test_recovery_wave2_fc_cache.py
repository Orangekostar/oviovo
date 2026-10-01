"""A region miss must write only the new task cache."""

import json

import numpy as np

from static_ovmap.module_validation.assets import sha256_file


def test_parent_feature_lookup_is_read_only_and_miss_is_task_local(tmp_path):
    from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache

    parent = tmp_path / "parent"
    directory = parent / "model/regions"
    directory.mkdir(parents=True)
    array_path = directory / "first.npz"
    np.savez_compressed(array_path, feature=np.array([1., 2.], np.float64))
    receipt_path = directory / "first.json"
    receipt_path.write_text(json.dumps({"content_identity": "first", "arrays": {
        "path": str(array_path), "bytes": array_path.stat().st_size, "sha256": sha256_file(array_path)}}))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    cache = ContentCache([parent], tmp_path / "new", "model")
    found = cache.lookup("regions", "first")
    np.testing.assert_array_equal(np.load(found["arrays"]["path"])["feature"], [1., 2.])
    created = cache.write("regions", "second", {"feature": np.array([3., 4.])},
                          {"content_identity": "second"})
    assert str(tmp_path / "new") in created["arrays"]["path"]
    assert not (directory / "second.npz").exists()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
