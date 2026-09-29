import numpy as np


def test_common_view_failure_zero_detection_and_tie_rules():
    from src.static_ovmap.a7_evidence_upgrade.localized import decide

    # Failure for class2 removes its entire high-area view, including class1.
    r = decide([1, 2], 1, [.8, .2], [[1., None], [.1, .9]], [100., 1.])
    assert r["common_view_indices"] == [1]
    assert (r["SHORTLIST"], r["SPATIAL"], r["MIX50"]) == (1, 2, 2)
    np.testing.assert_allclose(r["q"], [.1, .9])
    r = decide([1, 2], 2, [.5, .5], [[.3, .3]], [1.])
    assert (r["SHORTLIST"], r["SPATIAL"], r["MIX50"]) == (2, 2, 2)
    r = decide([1, 2], 1, [.8, .2], [[0., 0.]], [1.])
    assert r["status"] == "NO_LOCALIZED_EVIDENCE" and r["SPATIAL"] == 1
    assert r["common_view_indices"] == [0]


def test_shortlist_includes_base_winner_and_hash_cap_is_order_independent():
    from src.static_ovmap.a7_evidence_upgrade.localized import freeze_candidates

    ids = [1, 2, 3, 4]
    labels = {i: 4 for i in range(70)}
    details = {i: {"probabilities": [.2, .2, .1, .5]} for i in labels}
    views = {f"owner:{i}": [f"request:{i}"] for i in labels}
    sources = {name: {"objects": {str(i): {"available": available, "label": label} for i in labels}}
               for name, available, label in [("N0", True, 1), ("Q_GAIN", True, 2), ("S_SIGLIP2_AREA", False, 3)]}
    first = freeze_candidates("scene", ids, sources, labels, details, views)
    second = freeze_candidates("scene", ids, sources, dict(reversed(list(labels.items()))), details, views)
    assert first == second and first["eligible_count"] == 70 and len(first["selected"]) == 64
    assert all(r["candidates"] == [1, 2, 4] for r in first["selected"])
    assert [r["hash"] for r in first["selected"]] == sorted(r["hash"] for r in first["selected"])
