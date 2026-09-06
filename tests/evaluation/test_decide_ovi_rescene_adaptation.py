from __future__ import annotations

import pytest

from scripts.evaluation.decide_ovi_rescene_adaptation import (
    AdaptationDecisionError,
    build_transfer_evidence_from_rows,
    decide_adaptation,
    decide_transfer_adaptation,
)


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "native_r": {
            "status": "MEASURED",
            "temporal_ap": 0.2,
            "temporal_recall": 0.4,
        },
        "d2": {"status": "MISSING_ASSET", "d0_paired_f1": None, "d2_paired_f1": None},
        "raw_resolver": {
            "status": "MEASURED",
            "raw_nonempty_fraction": 0.9,
            "legacy_paired_f1": 0.1,
            "supported_paired_f1": 0.0,
        },
        "registration_oracle": {
            "status": "MISSING_ASSET",
            "o1_f_score": None,
            "o2_f_score": None,
        },
        "completion_opportunity": {
            "status": "MISSING_ASSET",
            "opportunity_fraction": None,
        },
        "identity_geometry": {
            "status": "MISSING_ASSET",
            "identity_gain": None,
            "geometry_gain": None,
        },
    }


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda value: value["native_r"].update(
                temporal_ap=0.01, temporal_recall=0.02
            ),
            "REPORT_MODEL_LIMIT",
        ),
        (
            lambda value: value.update(
                d2={
                    "status": "MEASURED",
                    "d0_paired_f1": 0.4,
                    "d2_paired_f1": 0.1,
                },
                raw_resolver={"status": "MISSING_ASSET", "raw_nonempty_fraction": None, "legacy_paired_f1": None, "supported_paired_f1": None},
            ),
            "ADAPT_DECODER_MASK_HEAD",
        ),
        (lambda value: None, "TUNE_RESOLVER"),
        (
            lambda value: value.update(
                raw_resolver={"status": "MISSING_ASSET", "raw_nonempty_fraction": None, "legacy_paired_f1": None, "supported_paired_f1": None},
                registration_oracle={
                    "status": "MEASURED",
                    "o1_f_score": 0.05,
                    "o2_f_score": 0.3,
                },
            ),
            "FIX_REGISTRATION",
        ),
        (
            lambda value: value.update(
                raw_resolver={"status": "MISSING_ASSET", "raw_nonempty_fraction": None, "legacy_paired_f1": None, "supported_paired_f1": None},
                completion_opportunity={
                    "status": "MEASURED",
                    "opportunity_fraction": 0.005,
                },
            ),
            "CHANGE_PAIR_BUDGET",
        ),
        (
            lambda value: value.update(
                raw_resolver={"status": "MISSING_ASSET", "raw_nonempty_fraction": None, "legacy_paired_f1": None, "supported_paired_f1": None},
                identity_geometry={
                    "status": "MEASURED",
                    "identity_gain": 0.1,
                    "geometry_gain": 0.0,
                },
            ),
            "RETAIN_IDENTITY_ONLY",
        ),
    ],
)
def test_six_ordered_decision_rules(mutate, expected: str) -> None:
    evidence = _evidence()
    mutate(evidence)

    decision = decide_adaptation(evidence)

    assert decision["action"] == expected
    assert decision["evidence_status"] == "MEASURED_ONLY"


def test_missing_evidence_cannot_carry_a_numeric_value() -> None:
    evidence = _evidence()
    evidence["d2"]["d2_paired_f1"] = 0.0

    with pytest.raises(AdaptationDecisionError, match="missing evidence"):
        decide_adaptation(evidence)


def _transfer_evidence() -> dict[str, object]:
    return {
        "schema_version": 2,
        "pair_scope": "IDENTICAL_UUID_PAIRS",
        "domains": {
            "D0_NATIVE_PROCESSED": {
                "status": "MEASURED",
                "sensor_support_fraction": 1.0,
                "raw_nonempty_fraction": 0.90,
                "confident_query_fraction": 0.70,
                "proposal_representation_coverage": 0.80,
                "fixed_pool_association_f1": 0.60,
            },
            "D1_NATIVE_SENSOR_SUPPORT": {
                "status": "MEASURED",
                "sensor_support_fraction": 0.75,
                "raw_nonempty_fraction": 0.82,
                "confident_query_fraction": 0.62,
                "proposal_representation_coverage": 0.72,
                "fixed_pool_association_f1": 0.55,
            },
            "D2_OVI_RECONSTRUCTION": {
                "status": "MEASURED",
                "sensor_support_fraction": 0.70,
                "raw_nonempty_fraction": 0.80,
                "confident_query_fraction": 0.50,
                "proposal_representation_coverage": 0.70,
                "fixed_pool_association_f1": 0.50,
            },
        },
    }


def _transfer_thresholds() -> dict[str, float]:
    return {
        "maximum_d1_association_drop": 0.10,
        "maximum_d1_proposal_drop": 0.10,
        "minimum_d2_raw_nonempty_fraction": 0.50,
        "minimum_d2_confident_query_fraction": 0.10,
        "minimum_d2_proposal_representation_coverage": 0.25,
        "minimum_usable_association_f1": 0.05,
        "minimum_proposal_association_gap": 0.10,
    }


def test_transfer_gate_does_not_train_when_d1_sensor_support_degrades() -> None:
    evidence = _transfer_evidence()
    evidence["domains"]["D1_NATIVE_SENSOR_SUPPORT"].update(
        proposal_representation_coverage=0.20,
        fixed_pool_association_f1=0.55,
    )

    decision = decide_transfer_adaptation(evidence, _transfer_thresholds())

    assert decision["action"] == "NO_ADAPTATION"
    assert decision["selected_rule"] == "d1_sensor_support_degradation"


def test_transfer_gate_adapts_decoder_for_d2_only_raw_query_degradation() -> None:
    evidence = _transfer_evidence()
    evidence["domains"]["D2_OVI_RECONSTRUCTION"].update(
        raw_nonempty_fraction=0.20,
        confident_query_fraction=0.05,
        proposal_representation_coverage=0.70,
        fixed_pool_association_f1=0.40,
    )

    decision = decide_transfer_adaptation(evidence, _transfer_thresholds())

    assert decision["action"] == "ADAPT_DECODER_MASK_HEAD"
    assert decision["selected_rule"] == "d2_raw_query_domain_gap"


def test_transfer_gate_does_not_call_shared_raw_weakness_a_d2_domain_gap() -> None:
    evidence = _transfer_evidence()
    for domain_id in (
        "D0_NATIVE_PROCESSED",
        "D1_NATIVE_SENSOR_SUPPORT",
        "D2_OVI_RECONSTRUCTION",
    ):
        evidence["domains"][domain_id].update(
            raw_nonempty_fraction=0.20,
            confident_query_fraction=0.05,
            fixed_pool_association_f1=0.65,
        )

    decision = decide_transfer_adaptation(evidence, _transfer_thresholds())

    assert decision["action"] == "NO_ADAPTATION"
    assert decision["selected_rule"] == "no_actionable_transfer_gap"


def test_transfer_gate_repairs_resolver_when_raw_and_proposals_are_usable() -> None:
    evidence = _transfer_evidence()
    evidence["domains"]["D2_OVI_RECONSTRUCTION"].update(
        raw_nonempty_fraction=0.85,
        confident_query_fraction=0.45,
        proposal_representation_coverage=0.70,
        fixed_pool_association_f1=0.02,
    )

    decision = decide_transfer_adaptation(evidence, _transfer_thresholds())

    assert decision["action"] == "UPSTREAM_PROPOSAL_GROUPING_REPAIR"
    assert decision["selected_rule"] == "d2_resolver_bottleneck"


def test_transfer_gate_repairs_proposals_before_considering_decoder_adaptation() -> None:
    evidence = _transfer_evidence()
    evidence["domains"]["D2_OVI_RECONSTRUCTION"].update(
        raw_nonempty_fraction=0.20,
        confident_query_fraction=0.05,
        proposal_representation_coverage=0.05,
        fixed_pool_association_f1=0.0,
    )

    decision = decide_transfer_adaptation(evidence, _transfer_thresholds())

    assert decision["action"] == "UPSTREAM_PROPOSAL_GROUPING_REPAIR"
    assert decision["selected_rule"] == "d2_proposal_limited"


def test_transfer_evidence_is_derived_from_same_uuid_resolver_rows() -> None:
    rows = tuple(
        {
            "pair_id": "pair-a",
            "domain_id": domain_id,
            "status": "MEASURED",
            "method_id": method_id,
            "sensor_support_fraction": support,
            "raw_nonempty_fraction": raw,
            "confident_query_fraction": confident,
            "proposal_representation_coverage": proposal,
            "association_f1": association,
        }
        for domain_id, method_id, support, raw, confident, proposal, association in (
            ("D0_NATIVE_PROCESSED", "R_legacy", 1.0, 0.9, 0.7, 0.8, 0.6),
            ("D1_NATIVE_SENSOR_SUPPORT", "R_legacy", 0.7, 0.8, 0.6, 0.7, 0.5),
            ("D2_OVI_RECONSTRUCTION", "R_obj", 0.6, 0.7, 0.5, 0.4, 0.3),
        )
    )

    evidence = build_transfer_evidence_from_rows(rows)

    assert evidence == {
        "schema_version": 2,
        "pair_scope": "IDENTICAL_UUID_PAIRS",
        "domains": {
            "D0_NATIVE_PROCESSED": {
                "status": "MEASURED",
                "sensor_support_fraction": 1.0,
                "raw_nonempty_fraction": 0.9,
                "confident_query_fraction": 0.7,
                "proposal_representation_coverage": 0.8,
                "fixed_pool_association_f1": 0.6,
            },
            "D1_NATIVE_SENSOR_SUPPORT": {
                "status": "MEASURED",
                "sensor_support_fraction": 0.7,
                "raw_nonempty_fraction": 0.8,
                "confident_query_fraction": 0.6,
                "proposal_representation_coverage": 0.7,
                "fixed_pool_association_f1": 0.5,
            },
            "D2_OVI_RECONSTRUCTION": {
                "status": "MEASURED",
                "sensor_support_fraction": 0.6,
                "raw_nonempty_fraction": 0.7,
                "confident_query_fraction": 0.5,
                "proposal_representation_coverage": 0.4,
                "fixed_pool_association_f1": 0.3,
            },
        },
    }

    mixed = tuple({**row, "pair_id": "pair-b"} if index == 2 else row for index, row in enumerate(rows))
    with pytest.raises(AdaptationDecisionError, match="same UUID"):
        build_transfer_evidence_from_rows(mixed)
