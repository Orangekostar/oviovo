import numpy as np

from src.static_ovmap.m2_reviewer_study.evaluation import (
    confusion,
    official_view,
    semantic_summary,
)


def test_official_view_excludes_small_and_zero_and_ranks_current_class():
    owners = np.array([1] * 100 + [2] * 200 + [3] * 99 + [4] * 100)
    labels = {1: 5, 2: 5, 3: 7, 4: 0}
    before = labels.copy()
    view = official_view(owners, labels, 100)
    assert view == {1: {"label": 5, "rank": "0.500000", "area": 100}, 2: {"label": 5, "rank": "1.000000", "area": 200}}
    assert labels == before
    assert official_view(owners, {**labels, 1: 6}, 100)[1]["rank"] == "1.000000"


def test_pooled_confusion_is_not_scene_macro_and_ignores_invalid_gt():
    a = confusion(np.array([1, 0, 99]), np.array([2, 1, 1]), [1, 2])
    b = confusion(np.array([2]), np.array([2]), [1, 2])
    assert a.sum() == 1
    assert semantic_summary(a)["miou"] == 0.
    assert semantic_summary(b)["miou"] == 1.
    assert semantic_summary(a + b)["miou"] == .25
    assert semantic_summary(a + b)["macc"] == .5


def test_export_manifest_uses_relative_paths_accepted_by_released_parser(tmp_path, capsys):
    from src.static_ovmap.m2_reviewer_study.evaluation import write_manifest
    from src.static_ovmap.released_loader import load_released_module

    masks = tmp_path / "masks"
    masks.mkdir()
    path = masks / "owner_1.npy"
    np.save(path, np.ones(100, bool))
    dest = tmp_path / "evaluation" / "mapping.txt"
    dest.parent.mkdir()
    write_manifest(dest, {1: str(path)}, {1: {"label": 5, "rank": "1.000000"}})
    evaluator = load_released_module('/home/ww/crove/ovimap-module-validation-upstream/scripts/eval_utils.py')
    parsed = evaluator['read_instance_prediction_file'](str(dest), str(tmp_path))
    assert parsed[str(path)] == {"label_id": 5, "conf": 1.}
    assert capsys.readouterr().out == ""
