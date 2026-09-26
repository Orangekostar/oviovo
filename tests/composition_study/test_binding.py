"""Attempt isolation and metadata-only holdout exposure detection."""

import json


def test_attempt_selection_resumes_only_matching_bound_inputs(tmp_path):
    from src.static_ovmap.composition_study.binding import select_attempt

    first = select_attempt(tmp_path, "source-a")
    (first / "resolved_config.json").write_text(json.dumps({"binding_key": "source-a"}))
    assert select_attempt(tmp_path, "source-a") == first
    second = select_attempt(tmp_path, "source-b")
    assert second != first
    assert (
        json.loads((first / "resolved_config.json").read_text())["binding_key"]
        == "source-a"
    )


def test_raw_holdout_is_not_exposure_but_prepared_scene_blocks_both(tmp_path):
    from src.static_ovmap.composition_study.binding import confirmation_exposure

    scenes = ["scene0553_00", "scene0064_00"]
    raw = tmp_path / "data/scannet/scans/scene0553_00"
    raw.mkdir(parents=True)
    (raw / "sensor.sens").write_bytes(b"must not inspect raw images")
    clean = confirmation_exposure([tmp_path], scenes)
    assert clean["status"] == "NO_PRIOR_PREPARED_CONFIRMATION_FOUND"
    prepared = tmp_path / "scannet_study_v1/scenes/scene0064_00"
    prepared.mkdir(parents=True)
    (prepared / "receipt.json").write_text('{"status":"COMPLETE"}')
    result = confirmation_exposure([tmp_path], scenes)
    assert result["status"] == "BLOCKED_CONFIRMATION_EXPOSURE"
    assert result["blocked_scenes"] == scenes
    assert result["prepared_scene_paths"] == [str(prepared)]
