from __future__ import annotations

import json
from pathlib import Path

RESULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "evaluation"
    / "results"
    / "ovi_rescene_b5_identity"
)


def _load(name: str) -> dict[str, object]:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


def test_blocked_result_package_is_complete_and_internally_consistent() -> None:
    checkpoint = _load("checkpoint_audit_summary.json")
    input_contract = _load("input_contract_summary.json")
    b4 = _load("b4_relation_summary.json")
    b5 = _load("b5_relation_summary.json")
    delta = _load("b5_vs_b4_relation_diff.json")
    decision = _load("decision.json")

    checkpoint_sha = checkpoint["canonical_checkpoint"]["sha256"]
    assert checkpoint["status"] == "C0_C1_PASS"
    assert checkpoint["model_topology"]["status"] == (
        "MODEL_TOPOLOGY_STRICT_COMPATIBLE"
    )
    assert checkpoint["model_topology"]["ignored_or_dropped_model_keys"] == []

    assert input_contract["status"] == "BLOCKED_INPUT_FEATURE_CONTRACT"
    assert input_contract["gpu_authorized"] is False
    assert input_contract["blocking_reasons"] == [
        "COLOR_NORMALIZATION_MISMATCH",
        "MISSING_SOURCE_GEOMETRY_NORMALS",
        "BLOCKED_RESCENE_TOKEN_CONSERVATION",
    ]

    assert b4["status"] == "PASS_FROZEN_OUTPUT"
    assert b4["relation_summary"]["total"] == 120
    for field in (
        "persistent_static",
        "persistent_moved",
        "appeared",
        "removed_candidate",
        "split",
        "merge",
        "uncertain",
        "one_to_one_persistent_total",
    ):
        assert b4["relation_summary"][field] is None
        assert field in b4["unavailable"]

    assert b5["status"] == "NOT_RUN_C2_BLOCKED"
    assert b5["checkpoint_sha256"] == checkpoint_sha
    assert b5["gpu_attempt_count"] == 0
    for field in (
        "backend_status",
        "query_count",
        "nonempty_query_count",
        "runtime_s",
        "peak_gpu_memory_bytes",
    ):
        assert b5[field] is None
        assert field in b5["unavailable"]
    assert b5["relation_summary"]["total"] is None

    assert delta["status"] == "NOT_RUN_C2_BLOCKED"
    for field in (
        "exact_topology_intersection",
        "b4_only",
        "b5_only",
        "b4_only_one_to_one",
        "b5_only_one_to_one",
        "b5_only_persistent_static",
        "b5_only_persistent_moved",
        "b5_only_registration_accepted",
    ):
        assert delta["topology_delta"][field] is None
        assert field in delta["unavailable"]

    assert decision["verdict"] == "RESCENE_B5_BLOCKED"
    assert decision["gpu_command_count"] == 0
    assert decision["gates"] == {
        "C0_checkpoint_structure": "PASS",
        "C1_model_topology": "PASS",
        "C2_input_and_token_contract": "FAIL",
        "C3_one_pair_backend": "NOT_RUN_C2_BLOCKED",
    }
    assert decision["learned_b7"] == "NOT_RUN_BY_SCOPE"
    assert decision["office"] == {"attempt_count": 0, "status": "HELD_OUT"}
    assert decision["three_rscan"]["selected_visit_count"] == 44
    assert decision["three_rscan"]["complete_visit_count"] == 27
    assert decision["three_rscan"]["final_ranking"] == "NOT_RUN"


def test_unavailable_measurements_are_null_not_zero_or_predictions() -> None:
    b5 = _load("b5_relation_summary.json")
    delta = _load("b5_vs_b4_relation_diff.json")

    assert all(value is None for value in b5["relation_summary"].values())
    assert all(value is None for value in delta["topology_delta"].values())
