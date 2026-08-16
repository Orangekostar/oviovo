from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.oviv2.temporal_config import (
    DiagnosticControl,
    ExecutionProfile,
    TemporalReadoutConfig,
    temporal_config_from_json,
)


RUNTIME_DIAGNOSTIC_KEYS = (
    "proposal_opportunity_count",
    "proposal_trigger_count",
    "reid_opportunity_count",
    "reid_trigger_count",
    "identity_expiry_count",
    "geometry_reclaim_count",
    "motion_rejection_count",
    "ledger_rejection_count",
    "epoch_reset_opportunity_count",
    "epoch_reset_trigger_count",
    "icp_opportunity_count",
    "icp_accept_count",
    "icp_reject_count",
    "ledger_stage_count",
    "ledger_commit_count",
    "ledger_reclaim_count",
)

_A2_COUNTERS = frozenset({
    "proposal_opportunity_count",
    "proposal_trigger_count",
    "identity_expiry_count",
    "geometry_reclaim_count",
    "motion_rejection_count",
    "epoch_reset_opportunity_count",
    "epoch_reset_trigger_count",
})
_LEDGER_COUNTERS = frozenset({
    "ledger_rejection_count",
    "ledger_stage_count",
    "ledger_commit_count",
    "ledger_reclaim_count",
})
_REID_COUNTERS = frozenset({"reid_opportunity_count", "reid_trigger_count"})
_ICP_COUNTERS = frozenset({
    "icp_opportunity_count", "icp_accept_count", "icp_reject_count",
})
_ALLOWED_COUNTERS_BY_PROFILE = {
    "a0": frozenset(),
    "a1": frozenset(),
    "a2": _A2_COUNTERS,
    "a3": _A2_COUNTERS | _LEDGER_COUNTERS,
    "a4": _A2_COUNTERS | _LEDGER_COUNTERS | _REID_COUNTERS | _ICP_COUNTERS,
}


def canonical_diagnostic_claim(
    config: TemporalReadoutConfig,
) -> dict[str, object] | None:
    control = config.diagnostic_control
    if control is None:
        return None
    component_enabled = {
        "proposal_recovery": (
            config.execution_profile.profile_id in {"a2", "a3", "a4"}
            and config.proposal_recovery_enabled
        ),
        "background_masking": config.execution_profile.profile_id in {"a3", "a4"},
        "background_ledger": (
            config.execution_profile.profile_id in {"a3", "a4"}
            and config.background_ledger_enabled
        ),
        "dormant_reid": (
            config.execution_profile.profile_id == "a4"
            and config.dormant_reid_enabled
        ),
        "icp": config.execution_profile.profile_id == "a4" and config.icp_enabled,
    }
    disabled_component = {
        DiagnosticControl.A2_NO_PROPOSAL_RECOVERY: "proposal_recovery",
        DiagnosticControl.A3_MASKING_ONLY_NO_LEDGER: "background_ledger",
        DiagnosticControl.A4_NO_DORMANT_CANDIDATES: "dormant_reid",
        DiagnosticControl.A4_TRANSLATION_ONLY_NO_ICP: "icp",
    }[control]
    return {
        "identity": control.diagnostic_identity,
        "controls": control.controls,
        "component_enabled": component_enabled,
        "positive_claim_available": {disabled_component: False},
    }


def _expected_claim_without_config(
    runtime: Mapping[str, Any], label: str
) -> tuple[DiagnosticControl | None, dict[str, object] | None]:
    diagnostic = runtime.get("diagnostic")
    if diagnostic is None:
        return None, None
    if not isinstance(diagnostic, Mapping):
        raise ValueError(f"{label} diagnostic claim is invalid")
    profile = ExecutionProfile.from_id(runtime.get("execution_profile"))
    controls = diagnostic.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError(f"{label} diagnostic claim is invalid")
    control = DiagnosticControl.from_controls(profile, controls)
    enabled = {
        "proposal_recovery": profile.profile_id in {"a2", "a3", "a4"},
        "background_masking": profile.profile_id in {"a3", "a4"},
        "background_ledger": profile.profile_id in {"a3", "a4"},
        "dormant_reid": profile.profile_id == "a4",
        "icp": profile.profile_id == "a4",
    }
    disabled_component = {
        DiagnosticControl.A2_NO_PROPOSAL_RECOVERY: "proposal_recovery",
        DiagnosticControl.A3_MASKING_ONLY_NO_LEDGER: "background_ledger",
        DiagnosticControl.A4_NO_DORMANT_CANDIDATES: "dormant_reid",
        DiagnosticControl.A4_TRANSLATION_ONLY_NO_ICP: "icp",
    }[control]
    enabled[disabled_component] = False
    return control, {
        "identity": control.diagnostic_identity,
        "controls": control.controls,
        "component_enabled": enabled,
        "positive_claim_available": {disabled_component: False},
    }


def validate_runtime_diagnostics(
    runtime: Mapping[str, Any],
    *,
    label: str = "runtime diagnostics",
    expected_temporal_readout: Mapping[str, Any] | None = None,
    expected_candidate_id: str | None = None,
    expected_processed_frame_count: int | None = None,
) -> None:
    required = {
        "schema_version", "execution_profile", "processed_frame_count",
        "counters", "mechanism_records", "diagnostic",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != required:
        raise ValueError(f"{label} fields/diagnostic claim are not exact")
    try:
        profile = ExecutionProfile.from_id(runtime.get("execution_profile"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} execution profile is invalid") from error
    if type(runtime.get("schema_version")) is not int or runtime["schema_version"] != 1:
        raise ValueError(f"{label} schema_version is invalid")
    if (
        type(runtime.get("processed_frame_count")) is not int
        or runtime["processed_frame_count"] < 0
        or (
            expected_processed_frame_count is not None
            and runtime["processed_frame_count"] != expected_processed_frame_count
        )
    ):
        raise ValueError(f"{label} identity/frame count is invalid")

    if expected_temporal_readout is not None:
        parsed = temporal_config_from_json(
            {"temporal_readout": dict(expected_temporal_readout)}
        )
        if parsed.execution_profile is not profile:
            raise ValueError(f"{label} execution profile claim mismatch")
        expected_claim = canonical_diagnostic_claim(parsed)
        control = parsed.diagnostic_control
        expected_identity = parsed.diagnostic_identity or profile.profile_id
        if expected_candidate_id is not None and expected_candidate_id != expected_identity:
            raise ValueError(f"{label} diagnostic candidate declaration mismatch")
    else:
        control, expected_claim = _expected_claim_without_config(runtime, label)
    if runtime["diagnostic"] != expected_claim:
        raise ValueError(f"{label} diagnostic claim mismatch")

    counters = runtime["counters"]
    records = runtime["mechanism_records"]
    names = set(RUNTIME_DIAGNOSTIC_KEYS)
    if (
        not isinstance(counters, Mapping)
        or not isinstance(records, Mapping)
        or set(counters) != names
        or set(records) != names
    ):
        raise ValueError(f"{label} counter/record inventory mismatch")
    for name in RUNTIME_DIAGNOSTIC_KEYS:
        count = counters[name]
        values = records[name]
        if (
            type(count) is not int
            or count < 0
            or not isinstance(values, list)
            or len(values) != count
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"{label} records do not match counter: {name}")

    allowed = set(_ALLOWED_COUNTERS_BY_PROFILE[profile.profile_id])
    if control is DiagnosticControl.A2_NO_PROPOSAL_RECOVERY:
        allowed -= {"proposal_opportunity_count", "proposal_trigger_count"}
    elif control is DiagnosticControl.A3_MASKING_ONLY_NO_LEDGER:
        allowed -= _LEDGER_COUNTERS
    elif control is DiagnosticControl.A4_NO_DORMANT_CANDIDATES:
        allowed -= _REID_COUNTERS
    elif control is DiagnosticControl.A4_TRANSLATION_ONLY_NO_ICP:
        allowed -= _ICP_COUNTERS
    impossible = names - allowed
    if any(counters[name] != 0 or records[name] != [] for name in impossible):
        raise ValueError(f"{label} profile-impossible counters/records must be zero")
    if profile in {ExecutionProfile.A2, ExecutionProfile.A3, ExecutionProfile.A4}:
        epoch_records = records["epoch_reset_opportunity_count"]
        motion_epoch_records = [
            value for value in epoch_records if value.startswith("motion:")
        ]
        if (
            any(
                not value.startswith(("motion:", "dynamic:"))
                for value in epoch_records
            )
            or motion_epoch_records != records["motion_rejection_count"]
        ):
            raise ValueError(
                f"{label} epoch reset motion records and motion rejection "
                "records must be identical"
            )

    subset_pairs = (
        ("proposal_trigger_count", "proposal_opportunity_count"),
        ("reid_trigger_count", "reid_opportunity_count"),
        ("epoch_reset_trigger_count", "epoch_reset_opportunity_count"),
        ("icp_accept_count", "icp_opportunity_count"),
        ("icp_reject_count", "icp_opportunity_count"),
        ("ledger_commit_count", "ledger_stage_count"),
        ("ledger_reclaim_count", "ledger_commit_count"),
    )
    if any(not set(records[child]) <= set(records[parent]) for child, parent in subset_pairs):
        raise ValueError(f"{label} mechanism record relation mismatch")
    opportunities = set(records["icp_opportunity_count"])
    accepts = set(records["icp_accept_count"])
    rejects = set(records["icp_reject_count"])
    if accepts & rejects or accepts | rejects != opportunities:
        raise ValueError(f"{label} ICP mechanism records are not a partition")
    icp_enabled = profile is ExecutionProfile.A4 and (
        control is not DiagnosticControl.A4_TRANSLATION_ONLY_NO_ICP
    )
    if icp_enabled and not set(records["motion_rejection_count"]) <= opportunities:
        raise ValueError(f"{label} motion/ICP record relation mismatch")
