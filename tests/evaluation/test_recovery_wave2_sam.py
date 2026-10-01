"""Protected CropFormer groups and causal addition-only conflict handling."""

import numpy as np


def test_attached_winners_complete_background_without_changing_cropformer_groups():
    from static_ovmap.recovery_wave2.sam_completion import crop_priority_raster

    crop = np.zeros((20, 20), np.int32)
    crop[:10, :10], crop[:10, 10:] = 4, 9
    masks = np.zeros((2, 20, 20), bool)
    masks[0, :, :10], masks[1, :, 10:] = True, True
    raster, rows = crop_priority_raster(crop, masks, masks.copy())
    np.testing.assert_array_equal(raster[crop > 0], crop[crop > 0])
    assert np.all(raster[10:, :10] == 4) and np.all(raster[10:, 10:] == 9)
    assert [row["action"] for row in rows] == ["ATTACHED", "ATTACHED"]


def test_mutual_best_tie_is_unattached_and_standalone_uses_binary_overlap():
    from static_ovmap.recovery_wave2.sam_completion import crop_priority_raster

    crop = np.zeros((20, 20), np.int32)
    crop[:5] = 4
    mask = np.zeros_like(crop, bool)
    mask[:10] = True
    winners = np.zeros((2, 20, 20), bool)
    winners[0, :10, :10], winners[1, :10, 10:] = True, True
    raster, rows = crop_priority_raster(crop, np.stack([mask, mask]), winners)
    np.testing.assert_array_equal(raster, crop)
    assert all(row["ambiguous_best"] for row in rows)
    masks = np.zeros((2, 20, 20), bool)
    masks[0].flat[:40] = True
    masks[0].flat[100:260] = True
    masks[1] = crop > 0
    winners[0] = masks[0] & (crop == 0)
    winners[1] = crop > 0
    raster, rows = crop_priority_raster(crop, masks, winners)
    np.testing.assert_array_equal(raster, crop)
    assert rows[0]["crop_positive_overlap"] == .2


def test_s2_known_stability_requires_two_factor_zero_owner_snapshots():
    from static_ovmap.recovery_wave2.sam_completion import known_stable_support

    labels = np.array([[3, 4, 5, 6]])
    owner = np.array([[7, 7, 7, 7]])
    probe = {"prior_label": labels, "prior_owner": owner, "validity": np.ones_like(labels),
             "hit_depth": np.ones_like(labels, float)}
    stable = known_stable_support(probe, np.ones_like(labels, float), [{3: 7, 4: 8, 5: 7}, {3: 7, 4: 7, 6: 7}])
    np.testing.assert_array_equal(stable, [[True, False, False, False]])
    assert not known_stable_support(probe, np.ones_like(labels, float), [{3: 7}]).any()


def test_s2_conflicts_use_proposed_support_and_remove_only_sam_additions():
    from static_ovmap.recovery_wave2.sam_completion import filter_additions

    crop = np.zeros((20, 20), np.int32)
    crop[:10, :10] = 4
    additions = np.zeros_like(crop, bool)
    additions[10:, :10] = True
    proposed = (crop == 4) | additions
    owner = np.full_like(crop, 2)
    owner[proposed] = 1
    owner[10:13, :10] = 2
    stable = np.ones_like(crop, bool)
    kept, row = filter_additions(additions, proposed, stable, owner, 1)
    np.testing.assert_array_equal(kept, additions)
    assert row["known_other"] == 30 and row["q_other"] == .15
    owner[10:16, :10] = 2
    kept, row = filter_additions(additions, proposed, stable, owner, 1)
    assert not kept.any() and row["removed_additions"] == 100
    assert row["q_other"] == .3


def test_causal_s2_uses_previous_binary_mask_and_preserves_crop_pixels(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from static_ovmap.recovery_wave2 import sam_completion as module

    crop = np.zeros((20, 20), np.int32)
    crop[:10, :10] = 4
    mask = np.zeros_like(crop, bool)
    mask[:, :10] = True
    s1 = crop.copy()
    s1[mask] = 4
    owner = np.ones_like(crop)
    owner[10:16, :10] = 2
    probe = {"prior_label": owner.copy(), "prior_owner": owner, "validity": np.ones_like(crop),
             "hit_depth": np.ones_like(crop, float), "state_before": {"n": 1}, "state_after": {"n": 1}}
    tracks = {"crop": crop, "winners": mask[None], "masks": mask[None], "ids": [1]}
    row = {"track_index": 0, "track_id": 1, "seed_key": [0, "key"], "output_group": 4,
           "attached_crop_group": 4, "action": "ATTACHED"}
    source = {"chunk": 0, "tracks": [row]}
    monkeypatch.setattr(module, "_hook_tracks", lambda hook, frame: (tracks, source))
    snapshot = {"frame_id": 1, "chunk": 0, "membership": {1: 1, 2: 2},
                "depth": np.ones_like(crop, float), "pose": np.eye(4), "intrinsics": np.eye(3),
                "binary_tracks": {1: crop > 0}}
    hook = SimpleNamespace(history=[snapshot, snapshot], frontend_context={"frame_id": 2, "original": crop, "s1": s1},
                           gsm=SimpleNamespace(exportAssociationProbe=lambda pose, depth: probe))
    result = module.causal_s2_raster(hook, 2, np.ones_like(crop, float), np.eye(3), np.eye(4), s1, tmp_path)
    np.testing.assert_array_equal(result, crop)
    diagnostic = module.read(tmp_path / "s2_conflict.json")["tracks"][0]
    assert diagnostic["self_owner"] == 1 and diagnostic["known_visible_past_pixels"] == 100
    assert diagnostic["q_other"] == .3 and diagnostic["removed_additions"] == 100
    source["chunk"] = 1
    result = module.causal_s2_raster(hook, 2, np.ones_like(crop, float), np.eye(3), np.eye(4), s1, tmp_path)
    np.testing.assert_array_equal(result, s1)
