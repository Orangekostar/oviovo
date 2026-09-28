from dataclasses import replace

import numpy as np
import pytest

from src.static_ovmap.module_validation.query_state import (
    AcquisitionPayload,
    FeatureStore,
)
from tests.module_validation.test_query_gain_policy import _candidate
from tests.module_validation.test_query_lineage import _snapshot


def test_random_prefix_is_deterministic_and_split_evidence_never_resurrects():
    from src.static_ovmap.m2_reviewer_study.query_replay import replay_study

    class Frames:
        scene_id = "test"
        schedule = tuple(range(200))

        def load(self, index):
            return {"candidates": [replace(_candidate(1, "mixed", 12), frame_index=0)] if index == 0 else [],
                    "snapshot": _snapshot({11: 1, 22: 2} if index == 1 else {11: 1, 22: 1})}

    def run(sentinel):
        calls = []
        store = FeatureStore(lambda c: calls.append(c.request_id) or AcquisitionPayload(np.array([1., 0.]), 6, .01),
                             initial_cache={"forbidden_future": AcquisitionPayload(sentinel, 6, 0.)})
        result = replay_study(Frames(), "RV_Q_RANDOM", store, np.eye(2), budget=200, seed=17)
        assert calls == ["mixed"]
        assert result["state"].logical_ledger.attempts == 1
        assert result["lineage"].dropped_feature_ids == {"mixed"}
        assert not result["state"].object_state(1).features
        return result["decisions"]

    assert run(np.array([0., 1.])) == run(np.array([1e30, -1e30]))


def test_combine_updates_coverage_on_quota_zero_frame():
    from src.static_ovmap.m2_reviewer_study.query_replay import replay_study

    # Native COMBINE requires at least 500 valid-depth pixels before it
    # initializes coverage. The small ranking fixture does not meet this gate.
    mask = np.ones((40, 40), dtype=bool)
    candidate = replace(_candidate(1, "a", 12), frame_index=0,
                        global_mask=mask, local_mask=mask, union_mask=mask,
                        bbox_xyxy=(0, 0, 39, 39), global_pixels=1600,
                        local_pixels=1600, overlap_pixels=1600)

    class Frames:
        scene_id = "test"
        schedule = tuple(range(200))

        def load(self, index):
            return {"candidates": [candidate] if index == 0 else [],
                    "snapshot": _snapshot({11: 1})}

    def forbidden(_):
        raise AssertionError("quota-zero frame must not read features")

    result = replay_study(Frames(), "Q_COMBINE", FeatureStore(forbidden), np.eye(2), budget=100)
    assert result["decisions"][0]["quota"] == 0
    assert result["combine"].coverage_by_owner[1] == set(candidate.spherical_cells)
    assert result["state"].logical_ledger.attempts == 0


def test_query_input_index_checks_receipt_outputs_after_file_changes(tmp_path):
    from src.static_ovmap.m2_reviewer_study.binding import InputIndex

    path = tmp_path / "feature.bin"
    path.write_bytes(b"original")
    index = InputIndex()
    expected = index.identity(path)
    receipt = {"outputs": [expected]}
    assert index.expected_output(path, receipt) == expected
    assert index.manifest() == {"entries": [expected]}
    with pytest.raises(ValueError, match="missing from source receipt"):
        index.expected_output(path, {"outputs": []})
    path.write_bytes(b"changed feature")
    with pytest.raises(ValueError, match="bound input changed"):
        index.expected_output(path, receipt)


def test_study_query_access_rejects_unfrozen_curves_and_invalid_seeds(tmp_path):
    from src.static_ovmap.m2_reviewer_study.query_jobs import authorize_query

    binding = {"output_root": str(tmp_path), "scenes": {"cal": {"role": "CAL"},
                                                       "room0": {"role": "HISTORICAL_TRANSFER"}}}
    assert authorize_query(binding, "cal", "RV_Q_RANDOM", 200, 17) == "RV_Q_RANDOM_s17_B200"
    with pytest.raises(ValueError):
        authorize_query(binding, "cal", "RV_Q_RANDOM", 200, 42)
    with pytest.raises(ValueError):
        authorize_query(binding, "room0", "Q_COMBINE", 100, None)
    with pytest.raises(ValueError):
        authorize_query(binding, "cal", "Q_GAIN", 100, None)


def test_query_cache_import_accepts_json_tuple_roundtrip_but_checks_hash(tmp_path):
    from src.static_ovmap.composition_study.io import write_once
    from src.static_ovmap.composition_study.visual_requests import operation_identity
    from src.static_ovmap.m2_reviewer_study.binding import InputIndex
    from src.static_ovmap.m2_reviewer_study.query_jobs import StudyLoader

    request = {"image_sha256": "a", "target_mask_sha256": "b", "native_union_mask_sha256": "c",
               "bbox_xyxy": (0, 0, 10, 10), "crop_convention": "six"}
    identity = operation_identity("model", request)
    path = tmp_path / (identity + ".json")
    write_once(path, {"input_identity": identity, "model_identity": "model", "request": request})
    loader = StudyLoader.__new__(StudyLoader)
    loader.index = InputIndex()
    loader.historical = {str(path): loader.index.identity(path)}
    loader.bound, loader.old_cache = {"identity": "model"}, tmp_path
    assert loader._import_native(request)[0] == path
    path.write_text("{}")
    with pytest.raises(ValueError, match="bound input changed"):
        loader._import_native(request)


def test_curve_gate_requires_strict_ap_improvement_over_seed_mean():
    from src.static_ovmap.m2_reviewer_study.budget_controls import curve_decision

    gain = {"uap": .4, "miou": .5}
    combine = {"uap": .3, "miou": .5}
    random = [{"uap": .2, "miou": .4}, {"uap": .3, "miou": .5}, {"uap": .4, "miou": .6}]
    assert curve_decision(gain, combine, random)["triggered"]
    assert not curve_decision(gain, {**combine, "uap": .4}, random)["triggered"]
    assert not curve_decision(gain, {**combine, "miou": .500001}, random)["triggered"]
