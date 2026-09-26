from types import SimpleNamespace

import pytest


def test_evaluation_reuse_ignores_method_but_not_any_protocol_content(tmp_path):
    from src.static_ovmap.composition_study.evaluation import evaluation_identity

    files = {}
    for name in ("evaluator", "projection", "ground_truth", "vocabulary"):
        path = tmp_path / name
        path.write_bytes(b"a")
        files[name] = [path]
    payload = SimpleNamespace(
        locked=True, prediction_key="whole_prediction", scene_id="cal", method_id="A"
    )
    original = evaluation_identity(payload, files)
    payload.method_id = "B"
    assert evaluation_identity(payload, files) == original
    for paths in files.values():
        paths[0].write_bytes(b"b")
        assert evaluation_identity(payload, files) != original
        paths[0].write_bytes(b"a")
    payload.prediction_key = "changed_classes"
    assert evaluation_identity(payload, files) != original
    payload.locked = False
    with pytest.raises(ValueError, match="locked"):
        evaluation_identity(payload, files)
