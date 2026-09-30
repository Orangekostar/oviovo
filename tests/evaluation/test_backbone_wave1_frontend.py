import numpy as np

from static_ovmap.backbone_wave1.frontend_sam2 import current_output, paired_rasters, warp_visible_support


def test_past_rgbd_visibility_abstains_on_missing_depth_and_rejects_occlusion():
    mask = np.ones((12, 12), bool)
    depth = np.full(mask.shape, 2., np.float32)
    k = np.array([[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]])
    pose = np.eye(4)
    np.testing.assert_array_equal(warp_visible_support(mask, depth, pose, k, depth, pose, k), mask)
    assert not warp_visible_support(mask, depth, pose, k, depth / 2, pose, k).any()
    assert not warp_visible_support(mask, depth * 0, pose, k, depth, pose, k).any()


def test_current_only_propagation_and_raw_defined_shared_discovery():
    class Predictor:
        def propagate_in_video(self, state, **kwargs):
            assert kwargs == dict(start_frame_idx=2, max_frame_num_to_track=0, reverse=False)
            yield 2, [1], np.ones((1, 1, 12, 20), np.float32)
    ids, logits = current_output(Predictor(), {}, 2)
    assert ids == [1] and logits.shape == (1, 12, 20)
    logits[:, :, 10:] = -1
    crop = np.zeros((12, 20), int)
    crop[:, :10] = 11
    crop[:, 10:] = 22
    past_visible = [np.pad(np.ones((12, 10), bool), ((0, 0), (10, 0)))]
    raw, geom, diagnostic = paired_rasters(logits, ["seed"], crop, past_visible)
    assert diagnostic["discovery_groups"] == [22]
    assert diagnostic["tracks"][0]["decision"] == "REPLACE"
    assert set(raw[raw > 0]) == {1, 2}
    assert set(geom[geom > 0]) == {1, 2}
    assert raw[:, :10].all() and geom[:, 10:].all()
    assert diagnostic["geom_discovery_groups"] == diagnostic["discovery_groups"]
