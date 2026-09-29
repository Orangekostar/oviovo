import json

from src.static_ovmap.a7_evidence_upgrade.report_data import request_intervention


def test_request_accounting_ignores_text_cache_but_keeps_failed_visual_views(tmp_path):
    folder = tmp_path / "c0/requests/office0"
    folder.mkdir(parents=True)
    records = {
        "text": {"status": "AVAILABLE"},
        "a" * 64: {"status": "COMPLETE", "processor_support": [4, 3, 2], "elapsed_seconds": 1.5},
        "b" * 64: {"status": "UNAVAILABLE", "error": "EMPTY_SUPPORT", "elapsed_seconds": .5},
    }
    for name, value in records.items():
        (folder / (name + ".json")).write_text(json.dumps(value))
    result = request_intervention(tmp_path, "office0", "AW_C0_SO400M", ["a" * 64, "b" * 64])
    assert result["requests"] == 2
    assert result["status_counts"] == {"COMPLETE": 1, "UNAVAILABLE": 1}
    assert result["failure_counts"] == {"EMPTY_SUPPORT": 1}
    assert result["nonempty_final_support_views"] == 1
    assert result["recorded_request_elapsed_seconds"] == 2
