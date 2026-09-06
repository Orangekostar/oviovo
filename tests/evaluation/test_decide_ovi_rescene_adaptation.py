from __future__ import annotations

import pytest

from scripts.evaluation.decide_ovi_rescene_adaptation import (
    AdaptationDecisionError,
    decide_adaptation,
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
