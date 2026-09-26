def test_partial_scene_means_keep_defined_and_required_denominators():
    from src.static_ovmap.composition_study.reporting import metric_summary

    rows = [
        {
            "scene_id": "a",
            "role": "compose_cal",
            "method_id": "M",
            "status": "COMPLETE",
            "metrics": {"uap": 0.2, "miou": 0.4},
        },
        {
            "scene_id": "b",
            "role": "compose_cal",
            "method_id": "M",
            "status": "COMPLETE",
            "metrics": {"uap": None, "miou": 0.6},
        },
        {
            "scene_id": "c",
            "role": "regression_only",
            "method_id": "M",
            "status": "COMPLETE",
            "metrics": {"uap": 0.9, "miou": 0.9},
        },
    ]
    result = metric_summary(rows, "compose_cal", "M", ["a", "b"])
    assert result["means"]["uap"] == 0.2
    assert result["means"]["miou"] == 0.5
    assert result["denominators"]["uap"] == {"defined": 1, "total": 2}
    assert result["status"] == "INCONCLUSIVE_UNDEFINED_METRIC"
    assert (
        metric_summary([], "compose_cal", "M", ["a", "b"])["status"] == "MISSING_ROWS"
    )


def test_scene_tradeoff_is_separate_from_mean_gain():
    from src.static_ovmap.composition_study.reporting import compare_methods

    rows = []
    for scene, candidate in (("a", (0.4, 0.5)), ("b", (0.15, 0.3))):
        for method, metrics in (("N0", (0.2, 0.3)), ("M", candidate)):
            rows.append(
                {
                    "scene_id": scene,
                    "role": "confirmation",
                    "method_id": method,
                    "status": "COMPLETE",
                    "metrics": dict(zip(("uap", "miou"), metrics, strict=True)),
                }
            )
    result = compare_methods(rows, "confirmation", "M", "N0", ["a", "b"])
    assert result["status"] == "MEAN_GAIN_WITH_SCENE_TRADEOFF"
    assert not result["every_scene_nonnegative"]
    assert abs(result["worst_scene_deltas"]["uap"] + 0.05) < 1e-10
