from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULTS = (
    Path(__file__).resolve().parents[2]
    / "configs/evaluation/results/ovi_rescene_c2_repair_b5_v2"
)
EVALUATED_CODE_SHA = "85248e62b4a0e780ed72ea339abb911806a564ef"


def _load(name: str) -> dict[str, object]:
    payload = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_c2_blocker_preserves_null_results_and_zero_attempts() -> None:
    coverage = _load("attribute_coverage.json")
    decision = _load("decision.json")
    backend = _load("b5_backend_receipt.json")
    b4 = _load("b4_relations.json")
    b5 = _load("b5_relations.json")
    delta = _load("relation_delta.json")

    assert decision["verdict"] == "C2_REPAIR_BLOCKED"
    assert decision["evaluated_code_sha"] == EVALUATED_CODE_SHA
    assert decision["gt_identity_evidence_available"] is False
    assert decision["paper_superiority_established"] is False
    assert decision["gpu_attempt_count"] == 0
    assert decision["b4_reconstruction_attempt_count"] == 0
    assert decision["office_attempt_count"] == 0
    assert coverage["unsupported_model_count"] == 36_738
    assert coverage["model_attribute_coverage_fraction"] < 1.0
    assert backend["status"] == "NOT_RUN_C2_BLOCKED"
    assert backend["metrics"] is None
    assert b4["relations"] is None
    assert b5["relations"] is None
    assert delta["topology_delta"] is None


def test_artifact_manifest_covers_exact_nonself_file_set() -> None:
    manifest = _load("artifact_manifest.json")
    expected = {
        "attribute_coverage.json",
        "b4_relations.json",
        "b5_backend_receipt.json",
        "b5_relations.json",
        "decision.json",
        "input_contract_v2.json",
        "relation_delta.json",
        "reuse_ledger.json",
        "sampling_map_summary.json",
    }
    assert manifest["evaluated_code_sha"] == EVALUATED_CODE_SHA
    assert set(manifest["artifacts"]) == expected
    assert {path.name for path in RESULTS.iterdir() if path.name != "artifact_manifest.json"} == expected
    for name in sorted(expected):
        content = (RESULTS / name).read_bytes()
        assert manifest["artifacts"][name] == {
            "byte_count": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
