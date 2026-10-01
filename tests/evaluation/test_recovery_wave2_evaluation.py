"""Recovery changes the emitted universe without fabricating native evidence."""

from static_ovmap.recovery_wave2.evaluation import expanded_native_registry


def test_recovered_owner_has_mask_metadata_but_no_fabricated_native_score():
    source = {"objects": {"4": {"owner_id": 4, "available": True, "scores": [1., 0.]}},
              "valid_ids": [2, 3], "identity": "original"}
    expanded = expanded_native_registry(source, {4: 2, 9: 3})
    assert set(expanded["objects"]) == {"4", "9"}
    assert expanded["objects"]["4"]["scores"] == [1., 0.]
    assert not expanded["objects"]["9"]["available"]
    assert expanded["objects"]["9"]["scores"] is None
    assert set(source["objects"]) == {"4"}
