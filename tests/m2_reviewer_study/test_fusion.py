import pytest

VALID_IDS = (3, 1, 2)
SOURCE_NAMES = ("N0", "Q_GAIN", "S_SIGLIP2_AREA")
TEMPERATURES = {name: 0.07 for name in SOURCE_NAMES}


def _row(label, scores, *, available=True):
    return {
        "label": label,
        "available": available,
        "scores": None if scores is None else list(scores),
    }


def _source(objects, *, valid_ids=VALID_IDS):
    return {"valid_ids": list(valid_ids), "objects": objects}


def _soft_sources(*, q_valid_ids=VALID_IDS, q_scores=(0.1, 0.4, 0.2)):
    n0_scores = (0.3, 0.1, 0.2)
    return {
        "N0": _source(
            {
                1: _row(3, n0_scores),
                "2": _row(3, n0_scores),
            }
        ),
        "Q_GAIN": _source(
            {
                "1": _row(1, q_scores),
                2: _row(1, None, available=False),
            },
            valid_ids=q_valid_ids,
        ),
        "S_SIGLIP2_AREA": _source(
            {
                1: _row(2, (0.2, 0.1, 0.6)),
                "2": _row(1, None, available=False),
            }
        ),
    }


def _assert_legacy_parity(fuse, sources, native_labels):
    from src.static_ovmap.composition_study.label_fusion import fuse_labels

    expected_labels, expected_details = fuse_labels(
        native_labels,
        sources,
        VALID_IDS,
        "CP_M2_EQUAL_CAL",
        temperatures=TEMPERATURES,
    )
    labels, details = fuse(
        native_labels,
        sources,
        VALID_IDS,
        SOURCE_NAMES,
        TEMPERATURES,
    )

    assert labels == expected_labels
    for owner in native_labels:
        assert details[owner]["label"] == expected_details[owner]["label"]
        assert details[owner]["available_sources"] == expected_details[owner]["available_sources"]
        assert details[owner]["probabilities"] == pytest.approx(
            expected_details[owner]["probabilities"]
        )

    return labels, details


def test_three_source_soft_fusion_matches_legacy_and_aligns_permuted_ids():
    from src.static_ovmap.m2_reviewer_study.fusion import fuse

    native_labels = {1: 3, 2: 1}
    baseline_sources = _soft_sources()
    baseline_labels, baseline_details = _assert_legacy_parity(
        fuse, baseline_sources, native_labels
    )

    permuted_sources = _soft_sources(
        q_valid_ids=(1, 2, 3), q_scores=(0.4, 0.2, 0.1)
    )
    permuted_labels, permuted_details = _assert_legacy_parity(
        fuse, permuted_sources, native_labels
    )

    assert permuted_labels == baseline_labels
    for owner in native_labels:
        assert permuted_details[owner]["probabilities"] == pytest.approx(
            baseline_details[owner]["probabilities"]
        )


def test_selected_subset_supports_a3_empty_class0_and_native_fallback():
    from src.static_ovmap.m2_reviewer_study.fusion import fuse

    native_labels = {1: 3, 2: 1}
    sources = _soft_sources()
    selected = ("Q_GAIN", "S_SIGLIP2_AREA")

    labels, details = fuse(
        native_labels,
        sources,
        VALID_IDS,
        selected,
        TEMPERATURES,
        empty_class0=True,
    )
    assert labels[1] == 2
    assert details[1]["available_sources"] == list(selected)
    assert labels[2] == 0
    assert details[2]["label"] == 0
    assert details[2]["available_sources"] == []
    assert details[2]["probabilities"] is None

    fallback_labels, fallback_details = fuse(
        native_labels,
        sources,
        VALID_IDS,
        selected,
        TEMPERATURES,
    )
    assert fallback_labels[2] == native_labels[2]
    assert fallback_details[2]["label"] == native_labels[2]
    assert fallback_details[2]["probabilities"] is None


def test_hard_vote_excludes_unavailable_label_and_breaks_ties_by_valid_id_order():
    from src.static_ovmap.m2_reviewer_study.fusion import fuse

    native_labels = {1: 3}
    sources = {
        "N0": _source({1: _row(3, (1.0, 0.0, 0.0))}),
        "Q_GAIN": _source({"1": _row(3, None, available=False)}),
        "S_SIGLIP2_AREA": _source({1: _row(2, (0.0, 0.0, 1.0))}),
    }

    labels, details = fuse(
        native_labels,
        sources,
        VALID_IDS,
        SOURCE_NAMES,
        TEMPERATURES,
        hard=True,
    )

    assert labels == {1: 3}
    assert details[1]["label"] == 3
    assert details[1]["available_sources"] == ["N0", "S_SIGLIP2_AREA"]


def test_hard_vote_uses_only_available_sources_and_valid_id_tie_order():
    from src.static_ovmap.m2_reviewer_study.fusion import fuse

    native_labels = {1: 3}
    sources = {
        "N0": _source({1: _row(3, None, available=False)}),
        "Q_GAIN": _source({"1": _row(2, (0.0, 0.0, 1.0))}),
        "S_SIGLIP2_AREA": _source({1: _row(1, (0.0, 1.0, 0.0))}),
    }

    labels, details = fuse(
        native_labels,
        sources,
        VALID_IDS,
        SOURCE_NAMES,
        TEMPERATURES,
        hard=True,
    )

    assert labels == {1: 1}
    assert details[1]["label"] == 1
    assert details[1]["available_sources"] == ["Q_GAIN", "S_SIGLIP2_AREA"]


def test_hard_vote_returns_available_majority():
    from src.static_ovmap.m2_reviewer_study.fusion import fuse

    native_labels = {1: 3}
    sources = {
        "N0": _source({1: _row(3, (1.0, 0.0, 0.0))}),
        "Q_GAIN": _source({"1": _row(1, (0.0, 1.0, 0.0))}),
        "S_SIGLIP2_AREA": _source({1: _row(1, (0.0, 1.0, 0.0))}),
    }

    labels, details = fuse(
        native_labels,
        sources,
        VALID_IDS,
        SOURCE_NAMES,
        TEMPERATURES,
        hard=True,
    )

    assert labels == {1: 1}
    assert details[1]["label"] == 1
    assert details[1]["available_sources"] == list(SOURCE_NAMES)


def test_available_score_argmax_must_reproduce_source_label():
    from src.static_ovmap.m2_reviewer_study.fusion import fuse

    native_labels = {1: 1}
    sources = {
        "N0": _source({1: _row(1, (0.3, 0.1, 0.2))}),
        "Q_GAIN": _source({1: _row(1, None, available=False)}),
        "S_SIGLIP2_AREA": _source({1: _row(1, None, available=False)}),
    }

    with pytest.raises(ValueError):
        fuse(native_labels, sources, VALID_IDS, ("N0",), TEMPERATURES)


def test_unavailable_source_must_not_contain_fabricated_scores():
    from src.static_ovmap.m2_reviewer_study.fusion import fuse

    native_labels = {1: 1}
    sources = {
        "N0": _source({1: _row(1, None, available=False)}),
        "Q_GAIN": _source({"1": _row(1, (0.0, 1.0, 0.0), available=False)}),
        "S_SIGLIP2_AREA": _source({1: _row(1, None, available=False)}),
    }

    with pytest.raises(ValueError):
        fuse(native_labels, sources, VALID_IDS, ("Q_GAIN",), TEMPERATURES)
