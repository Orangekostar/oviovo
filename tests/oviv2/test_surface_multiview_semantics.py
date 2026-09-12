import numpy as np


def test_thin_rgb_crop_preprocessing_does_not_infer_height_as_channels():
    from types import SimpleNamespace

    from transformers import SiglipImageProcessor

    from src.oviv2.surface_multiview_semantics import native_crop_inputs

    rgb = np.zeros((3, 30, 3), np.uint8)
    rgb[:, :, 0] = 255
    processor = SimpleNamespace(
        image_processor=SiglipImageProcessor(size={"height": 8, "width": 8})
    )
    inputs = native_crop_inputs(processor, [rgb])
    np.testing.assert_allclose(
        inputs["pixel_values"].mean((0, 2, 3)).numpy(), [1, -1, -1]
    )


def test_native_six_crops_keep_rgb_and_masked_pairs_at_three_scales():
    from src.oviv2.surface_multiview_semantics import native_six_crops

    rgb = np.full((20, 20, 3), 127, np.uint8)
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    crops = native_six_crops(rgb, mask)
    assert [c.shape[:2] for c in crops] == [(10, 10)] * 2 + [(12, 12)] * 2 + [
        (14, 14)
    ] * 2
    assert crops[2][0, 0].tolist() == [127, 127, 127]
    assert crops[3][0, 0].tolist() == [0, 0, 0]
    assert native_six_crops(rgb, np.zeros_like(mask)) == []


def test_owner_posterior_lookup_keeps_visit_offsets_and_missing_rows():
    from src.oviv2.surface_multiview_semantics import owner_posteriors_to_rows

    ids, confidence, covered = owner_posteriors_to_rows(
        np.array([1, 1000001, 0, 2]),
        np.array([1000001, 1]),
        np.array([[0.1, 0.9], [0.8, 0.2]]),
        np.array([3, 5]),
    )
    assert ids.tolist() == [3, 5, 0, 0]
    np.testing.assert_allclose(confidence, [0.8, 0.9, 0, 0])
    assert covered.tolist() == [True, True, False, False]


def test_diverse_selection_avoids_redundant_camera_direction():
    from src.oviv2.surface_multiview_semantics import diverse_views

    directions = np.array([[1, 0, 0], [1, 0.01, 0], [-1, 0, 0], [0, 1, 0]])
    got = diverse_views(directions, np.array([100, 90, 40, 30]), k=3)
    assert got.tolist() == [0, 2, 3]


def test_zero_quality_is_missing_evidence_and_view_norm_does_not_weight_votes():
    from src.oviv2.surface_multiview_semantics import aggregate_views

    features = np.array([[100, 0], [0, 1]], dtype=float)
    np.testing.assert_allclose(aggregate_views(features, [1, 1]), [2**-0.5] * 2)
    np.testing.assert_allclose(aggregate_views(features, [0, 1]), [0, 1])
    assert aggregate_views(features, [0, 0]) is None


def test_equal_direction_ties_use_visibility_and_do_not_repeat_views():
    from src.oviv2.surface_multiview_semantics import diverse_views

    got = diverse_views(np.ones((3, 3)), np.array([3, 8, 5]), k=4)
    assert got.tolist() == [1, 2, 0]
