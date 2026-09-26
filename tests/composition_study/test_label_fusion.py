import pytest

VALID_IDS = (4, 7, 19)


def _source(label, scores, *, available=True, valid_ids=VALID_IDS):
    return {
        "valid_ids": list(valid_ids),
        "objects": {
            1: {
                "label": label,
                "available": available,
                "scores": None if scores is None else list(scores),
                "fallback_reason": None if available else "TECHNICAL_FALLBACK",
            }
        },
    }


def _sources(n0, q_gain, s2):
    return {"N0": n0, "Q_GAIN": q_gain, "S_SIGLIP2_AREA": s2}


def _probability_sum(probabilities):
    values = (
        list(probabilities.values())
        if hasattr(probabilities, "values")
        else list(probabilities)
    )
    if values and isinstance(values[0], (tuple, list)):
        values = (item[1] for item in values)
    return sum(values)


def test_m1_keeps_native_when_s2_is_unavailable():
    from src.static_ovmap.composition_study.label_fusion import fuse_labels

    labels, decisions = fuse_labels(
        {1: 4},
        _sources(
            _source(4, [1, 0, 0]),
            _source(7, [0, 1, 0]),
            _source(7, None, available=False),
        ),
        VALID_IDS,
        "CP_M1_AGREE_KEEP",
    )

    assert labels == {1: 4}
    decision = decisions[1]
    assert decision["label"] == 4
    assert decision["changed"] is False
    assert decision["available_sources"] == ["N0", "Q_GAIN"]
    assert decision["reason"] == "SOURCE_UNAVAILABLE"
    assert decision["probabilities"] is None


def test_m1_replaces_on_real_agreement_and_records_unchanged_agreement():
    from src.static_ovmap.composition_study.label_fusion import fuse_labels

    sources = _sources(
        _source(4, [1, 0, 0]),
        _source(7, [0, 1, 0]),
        _source(7, [0, 1, 0]),
    )
    labels, decisions = fuse_labels({1: 4}, sources, VALID_IDS, "CP_M1_AGREE_KEEP")
    assert labels == {1: 7}
    assert decisions[1]["label"] == 7
    assert decisions[1]["reason"] == "AGREEMENT_REPLACE"
    assert decisions[1]["changed"] is True

    labels, decisions = fuse_labels(
        {1: 7},
        _sources(
            _source(7, [0, 1, 0]),
            _source(7, [0, 1, 0]),
            _source(7, [0, 1, 0]),
        ),
        VALID_IDS,
        "CP_M1_AGREE_KEEP",
    )
    assert labels == {1: 7}
    assert decisions[1]["label"] == 7
    assert decisions[1]["reason"] == "AGREEMENT_UNCHANGED"
    assert decisions[1]["changed"] is False


def test_m2_equal_raw_renormalizes_to_one_available_source():
    from src.static_ovmap.composition_study.label_fusion import fuse_labels

    labels, decisions = fuse_labels(
        {1: 4},
        _sources(
            _source(4, None, available=False),
            _source(7, [0, 1, 0]),
            _source(7, None, available=False),
        ),
        VALID_IDS,
        "CP_M2_EQUAL_RAW",
    )

    assert labels == {1: 7}
    decision = decisions[1]
    assert decision["available_sources"] == ["Q_GAIN"]
    assert decision["source_weights"] == {"Q_GAIN": 1.0}
    assert len(decision["probabilities"]) == len(VALID_IDS)
    assert _probability_sum(decision["probabilities"]) == pytest.approx(1.0)


def test_m2_equal_raw_keeps_native_without_any_available_source():
    from src.static_ovmap.composition_study.label_fusion import fuse_labels

    labels, decisions = fuse_labels(
        {1: 4},
        _sources(
            _source(4, None, available=False),
            _source(7, None, available=False),
            _source(7, None, available=False),
        ),
        VALID_IDS,
        "CP_M2_EQUAL_RAW",
    )

    assert labels == {1: 4}
    decision = decisions[1]
    assert decision["label"] == 4
    assert decision["changed"] is False
    assert decision["available_sources"] == []
    assert decision["source_weights"] == {}
    assert decision["reason"] == "NO_AVAILABLE_SOURCE"
    assert decision["probabilities"] is None


def test_m2_equal_raw_aligns_ids_before_breaking_an_exact_tie():
    from src.static_ovmap.composition_study.label_fusion import fuse_labels

    sources = _sources(
        _source(4, [1, 0, 0]),
        _source(7, [0, 1, 0]),
        _source(7, None, available=False),
    )
    labels, _ = fuse_labels({1: 4}, sources, VALID_IDS, "CP_M2_EQUAL_RAW")
    assert labels == {1: 4}

    reordered_q = _sources(
        _source(4, [1, 0, 0]),
        _source(7, [0, 0, 1], valid_ids=(19, 4, 7)),
        _source(7, None, available=False),
    )
    labels, _ = fuse_labels({1: 4}, reordered_q, VALID_IDS, "CP_M2_EQUAL_RAW")
    assert labels == {1: 4}


def test_m2_cal_rejects_invalid_temperatures_and_missing_vocabulary_id():
    from src.static_ovmap.composition_study.label_fusion import fuse_labels

    sources = _sources(
        _source(4, [1, 0, 0]),
        _source(7, None, available=False),
        _source(7, None, available=False),
    )
    for invalid_temperature in (0.0, float("nan")):
        with pytest.raises(ValueError):
            fuse_labels(
                {1: 4},
                sources,
                VALID_IDS,
                "CP_M2_EQUAL_CAL",
                temperatures={
                    "N0": invalid_temperature,
                    "Q_GAIN": 0.07,
                    "S_SIGLIP2_AREA": 0.07,
                },
            )

    missing_id = _sources(
        _source(4, [1, 0], valid_ids=(4, 7)),
        _source(7, None, available=False),
        _source(7, None, available=False),
    )
    with pytest.raises(ValueError):
        fuse_labels(
            {1: 4},
            missing_id,
            VALID_IDS,
            "CP_M2_EQUAL_CAL",
            temperatures={
                "N0": 0.07,
                "Q_GAIN": 0.07,
                "S_SIGLIP2_AREA": 0.07,
            },
        )
