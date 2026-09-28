import pytest

from src.static_ovmap.replica_transfer.reporting import summarize


def row(scene, method, value):
    return {"status": "COMPLETE", "role": "replica_transfer", "scene_id": scene,
            "method_id": method, "metrics": {key: value for key in ("uap", "ap50", "ap25", "miou", "macc")}}


def test_missing_rows_are_not_zeros_or_complete():
    result = summarize([row("room0", "N0", 0.5)], ["room0", "room1"], ["N0", "Q_GAIN"])
    assert result["status"] == "INCOMPLETE"
    assert len(result["missing"]) == 3
    assert result["means"]["N0"]["means"]["uap"] == 0.5
    assert result["means"]["Q_GAIN"]["means"]["uap"] is None


def test_paired_deltas_include_scene_losses():
    rows = [row("room0", "N0", 0.2), row("room1", "N0", 0.2),
            row("room0", "Q_GAIN", 0.5), row("room1", "Q_GAIN", 0.1)]
    result = summarize(rows, ["room0", "room1"], ["N0", "Q_GAIN"])
    assert result["status"] == "COMPLETE"
    delta = result["comparisons"]["Q_GAIN_vs_N0"]
    assert delta["mean_deltas"]["uap"] == pytest.approx(0.1)
    assert delta["worst_scene_deltas"]["uap"] == pytest.approx(-0.1)
    assert delta["positive_scene_counts"]["uap"] == 1
    for key in ("ap25", "ap50", "macc"):
        assert delta["mean_deltas"][key] == pytest.approx(0.1)
        assert delta["worst_scene_deltas"][key] == pytest.approx(-0.1)
        assert delta["positive_scene_counts"][key] == 1


def test_duplicate_scene_method_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row("room0", "N0", 0.5)] * 2, ["room0"], ["N0"])


def test_undefined_secondary_metric_is_not_complete():
    measured = row("room0", "N0", 0.5)
    measured["metrics"]["ap25"] = None
    result = summarize([measured], ["room0"], ["N0"])
    assert result["status"] == "INCOMPLETE"
    assert result["means"]["N0"]["means"]["ap25"] is None
