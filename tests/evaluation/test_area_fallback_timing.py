import copy
import numpy as np
import pytest
import torch

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import read


def test_cold_session_shares_only_inside_one_call(monkeypatch):
    from static_ovmap.cvpr_compact.area_fallback_timing import AreaFallbackSession
    import static_ovmap.a7_evidence_upgrade.region_adapter as adapter

    monkeypatch.setattr(adapter, "image_tensor", lambda image, device: (
        torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None], (8, 8)))
    session = object.__new__(AreaFallbackSession)
    session.model_key, session.model, session.cache, session.device = "fixed", object(), None, "cpu"
    session.operators = {
        "extract_features_convnext": lambda model, image: {"clip_vis_dense": torch.ones(1, 3, 2, 2)},
        "MaskPooling": lambda: lambda dense, signed: torch.ones(1, 1, 3),
        "visual_prediction_forward_convnext": lambda model, pooled, signed: pooled,
    }
    requests = {rid: {"image_sha256": "rgb", "frame_id": "f"} for rid in ("first", "alias")}
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    values = {"image": np.ones((8, 8, 3), dtype=np.uint8), "target": mask}
    loaders = {rid: lambda request_id: values for rid in requests}
    for _ in range(2):
        features, stats = session.encode(requests, loaders)
        assert set(features) == set(requests)
        assert stats["physical_image_encodings"] == 1
        assert stats["physical_region_poolings"] == 1
        assert stats["area_fallback_poolings"] == 1
        assert not stats["persistent_feature_cache_enabled"]
        assert stats["dense_cache_hits"] == stats["region_cache_hits"] == 0
        assert stats["requests"]["first"]["original_support"] == 0
        assert stats["requests"]["alias"]["fallback"] is True
        np.testing.assert_array_equal(features["first"], features["alias"])
    session.cache = object()
    with pytest.raises(ValueError, match="persistent"):
        session.encode(requests, loaders)


def timing_rows():
    rows = []
    for scene in [f"scene{i}" for i in range(8)]:
        for arm in ("G1_FC", "G3_FC"):
            row = {"scene": scene, "arm": arm, "status": "COMPLETE",
                "protocol": "PROJECTED_FC_EMPTY_AREA_FALLBACK_V2", "resident_model": True,
                "model_load_included": False,
                "production_callable": "static_ovmap.cvpr_compact.recovery_run.recover_fc",
                "region_session": "static_ovmap.cvpr_compact.area_fallback_timing.AreaFallbackSession",
                "persistent_feature_cache_enabled": False, "persistent_view_cache_enabled": False,
                "persistent_result_cache_enabled": False, "dense_cache_hits": 0, "region_cache_hits": 0,
                "measured_seconds": 2. if arm == "G1_FC" else 6., "parity": {"status": "PASS"},
                "hardware": {"uuid": "GPU-fixed", "name": "A40", "compute_capability": "8.6"}}
            row["identity"] = canonical_digest(row)
            rows.append(row)
    return rows


def test_pool_requires_all_eight_scenes_and_both_arms():
    from static_ovmap.cvpr_compact.area_fallback_timing import aggregate_v2_timings
    spec = {"cohorts": {"replica8": [f"scene{i}" for i in range(8)]}, "methods": [],
            "selection": {"primary_method": "unused"}, "timing": {"cohort": "replica8", "arms": ["unused"]}}
    rows = timing_rows()
    result = aggregate_v2_timings(spec, rows)
    assert result["leaf_count"] == 16
    assert result["arms"]["G1_FC"]["mean_seconds"] == 2.
    assert result["arms"]["G3_FC"]["mean_seconds"] == 6.
    for invalid in (rows[:-1], [*rows, rows[0]]):
        with pytest.raises(ValueError):
            aggregate_v2_timings(spec, invalid)
    invalid = copy.deepcopy(rows)
    invalid[0]["dense_cache_hits"] = 1
    invalid[0]["identity"] = canonical_digest({k: v for k, v in invalid[0].items() if k != "identity"})
    with pytest.raises(ValueError, match="cache"):
        aggregate_v2_timings(spec, invalid)


def test_parent_memo_writes_are_redirected(tmp_path):
    from static_ovmap.cvpr_compact.area_fallback_timing import IndependentMemoIndex
    parent = tmp_path / "parent.json"
    atomic_write_json(parent, {"immutable": True})
    before = parent.read_bytes()
    destination = tmp_path / "v2" / "memo.json"
    index = IndependentMemoIndex(destination)
    index.identity(parent)
    index.write_memo(parent)
    assert parent.read_bytes() == before
    assert read(destination)["entries"][0]["identity"]["path"] == str(parent.resolve())


@pytest.mark.parametrize("changed", ["target_mask_sha256", "input_tensor_key", "fallback", "original_support"])
def test_mask_parity_rejects_changed_support_or_inputs(changed):
    from static_ovmap.cvpr_compact.area_fallback_timing import validate_mask_parity
    first = {"target_mask_sha256": "mask", "input_tensor_key": "tensor", "image_content_key": "rgb",
             "fallback": True, "original_support": 0}
    scientific = {"plan": {"G1": {"owner": ["request"]}}, "requests": {"request": first}}
    cold = {"requests": {"request": copy.deepcopy(first)}}
    validate_mask_parity(scientific, cold, "G1")
    cold["requests"]["request"][changed] = "changed"
    with pytest.raises(ValueError, match="cold recovery changed"):
        validate_mask_parity(scientific, cold, "G1")


def test_report_fills_timing_cells_with_eight_receipt_identities(tmp_path):
    from static_ovmap.cvpr_compact.area_fallback_report import typed_tables
    from static_ovmap.cvpr_compact.area_fallback_experiment import seal
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    parent, output = tmp_path / "parent", tmp_path / "v2"
    tables = seal({"tables": {"table1": [], "table2": [], "table3": [
        {"arm": arm, "row_id": arm, "cells": [{"metric": "cold_feature_incremental_seconds_mean",
            "source_kind": "UNAVAILABLE", "value_seconds": None, "completed_coverage": 0}]}
        for arm in ("G1", "G3")]}})
    atomic_write_json(parent / "tables" / "tables_main.json", tables)
    timing = {"identity": "timing", "arms": {arm + "_FC": {"mean_seconds": seconds,
        "receipt_identities": [f"receipt-{arm}-{i}" for i in range(8)]} for arm, seconds in (("G1", 2.), ("G3", 6.))}}
    filled = typed_tables(parent, output, {"identity": "store"},
        {"identity": "diagnosis", "arms": {"G1": {}, "G3": {}}}, ConsumptionIndex(), timing)
    assert filled["status"] == "COMPLETE"
    for row, expected in zip(filled["tables"]["table3"], (2., 6.)):
        cell = row["cells"][0]
        assert cell["source_kind"] == "MEASURED"
        assert cell["value_seconds"] == expected
        assert cell["completed_coverage"] == len(cell["measurement_receipt_identities"]) == 8
        assert cell["receipt_identity"] == "timing"
        assert cell["unavailable_reason"] is None
    assert read(output / "timing_scope.json")["status"] == "COMPLETE"
