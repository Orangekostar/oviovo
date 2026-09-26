from types import SimpleNamespace

import numpy as np
import pytest


def test_native_scores_preserve_fp32_canonical_scale_and_saved_order():
    from src.static_ovmap.composition_study.object_evidence import native_scores

    scores = native_scores(
        np.array([1, 0], np.float32),
        np.array([[1, 0], [0, 1], [1, 0]], np.float32),
        np.array([[0, 1], [-1, 0]], np.float32),
    )
    np.testing.assert_allclose(
        scores, [1 / (1 + np.exp(-1)), 0.5, 1 / (1 + np.exp(-1))], rtol=1e-7
    )
    assert scores.dtype == np.float32
    assert np.argmax(scores) == 0


def test_native_request_provenance_requires_unique_actual_paid_observation():
    from src.static_ovmap.composition_study.object_evidence import native_request_ids

    request = {
        "request_id": "real",
        "bbox_xyxy": [2, 3, 7, 8],
        "visible_target_pixels": 13,
        "target_id": "owner:99",
    }
    frame = {
        "frame_id": 11,
        "requests": [request],
        "native_selected_request_ids": ["real"],
        "pose_c2w": np.eye(4).tolist(),
    }
    saved = {
        "frame_id": [11],
        "box_2d": [(2, 3, 7, 8)],
        "vis_area": [13],
        "pose": [np.eye(4)],
    }
    assert native_request_ids(saved, {11: frame}) == ["real"]
    frame["native_selected_request_ids"] = []
    with pytest.raises(ValueError, match="unique paid"):
        native_request_ids(saved, {11: frame})
    frame["requests"].append({**request, "request_id": "duplicate"})
    frame["native_selected_request_ids"] = ["real", "duplicate"]
    with pytest.raises(ValueError, match="unique paid"):
        native_request_ids(saved, {11: frame})


def test_static_evidence_preserves_raw_mean_magnitudes_and_unavailable_owner():
    from src.static_ovmap.composition_study.object_evidence import static_objects

    manifest = {
        "views": {"owner:1": ["a", "b"], "owner:2": ["c"]},
        "requests": {key: {"visible_target_pixels": 10} for key in "abc"},
    }
    records = {
        "a": {"status": "COMPLETE", "feature": np.array([0.1, 0])},
        "b": {"status": "COMPLETE", "feature": np.array([0, 1.0])},
        "c": {"status": "FAILED", "feature": None},
    }
    objects = static_objects(manifest, records, np.eye(2), (4, 19), {1: 4, 2: 19})
    assert objects[1]["label"] == 19
    np.testing.assert_allclose(objects[1]["scores"], np.array([0.1, 1]) / np.sqrt(1.01))
    assert objects[1]["used_request_ids"] == ["a", "b"]
    assert not objects[2]["available"] and objects[2]["scores"] is None
    assert objects[2]["label"] == 19
    assert objects[2]["attempted_request_ids"] == ["c"]


def test_matching_geometry_does_not_allow_different_owner_rows():
    from src.static_ovmap.composition_study.object_evidence import same_source_surface

    native = SimpleNamespace(
        locked=True,
        scene_id="scene",
        geometry="same",
        owner_ids=np.array([1, 2]),
        instance_ranks=((1, 0.7), (2, 0.3)),
    )
    source = SimpleNamespace(**vars(native))
    same_source_surface(native, source)
    source.owner_ids = np.array([2, 1])
    with pytest.raises(ValueError, match="owner"):
        same_source_surface(native, source)
