from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

import scripts.evaluation.run_oviv2_tesse_dual_readout_search as search_runner
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    load_search_manifest,
    run_search,
)
from src.oviv2.temporal_config import ExecutionProfile, temporal_config_from_json
from src.evaluation.oviv2_runtime_diagnostics import canonical_diagnostic_claim


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json"
)
APARTMENT_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_apartment_v2.json"
OFFICE_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_office_v2.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(value) + b"\n" for value in values))
    return path


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path.absolute())
        ),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _source_path(preflight: Path, record: dict[str, object]) -> Path:
    return preflight.parent / str(record["path"])


def _preflight(tmp_path: Path, candidate_ids: tuple[str, ...]) -> Path:
    path = tmp_path / f"preflight-{'-'.join(candidate_ids)}.json"
    manifest = json.loads(MANIFEST.read_text())
    base = json.loads(APARTMENT_CONFIG.read_text())
    declarations = {item["candidate_id"]: item for item in manifest["candidates"]}
    diagnostics = {
        item["candidate_id"]: item for item in manifest["diagnostic_candidates"]
    }
    mechanisms = {
        "a0": [],
        "a1": ["absence", "readout_invalidation"],
        "a2": [
            "absence",
            "readout_invalidation",
            "proposal_recovery",
            "epoch_reset",
            "motion_rejection",
        ],
        "a3": [
            "absence",
            "readout_invalidation",
            "proposal_recovery",
            "epoch_reset",
            "motion_rejection",
            "background_release",
            "background_reclaim",
        ],
        "a4": [
            "absence",
            "readout_invalidation",
            "proposal_recovery",
            "epoch_reset",
            "motion_rejection",
            "background_release",
            "background_reclaim",
            "eligible_reid",
            "icp",
        ],
    }
    candidates = []
    evidence_root = tmp_path / f"preflight-sources-{'-'.join(candidate_ids)}"
    for candidate_id in candidate_ids:
        declaration = declarations.get(candidate_id)
        if declaration is None:
            diagnostic = diagnostics[candidate_id]
            declaration = {
                **declarations[diagnostic["base_profile"]],
                "diagnostic_controls": diagnostic["diagnostic_controls"],
            }
        materialized = search_runner._materialize_config(base, declaration)
        base_profile = diagnostics.get(candidate_id, {}).get("base_profile", candidate_id)
        candidate_mechanisms = list(mechanisms[base_profile])
        disabled = {
            "diag_a2_no_proposal_recovery": {"proposal_recovery"},
            "diag_a3_masking_only_no_ledger": {"background_release", "background_reclaim"},
            "diag_a4_no_dormant_candidates": {"eligible_reid"},
            "diag_a4_translation_only_no_icp": {"icp"},
        }.get(candidate_id, set())
        source_root = evidence_root / candidate_id
        coverage_rows = [
            {
                "frame_index": frame,
                "timestamp_ns": 100 + frame,
                "record_count": 1,
                "event_count": int(frame == 1),
            }
            for frame in range(3)
        ]
        trajectories = _write_jsonl(
            source_root / "trajectories.jsonl",
            [
                {
                    "frame_index": frame,
                    "timestamp_ns": 100 + frame,
                    "entity_id": "7",
                    "centroid_xyz": [0.0, 0.0, 0.0],
                    "observation_count": frame + 1,
                    "dynamic_state": "dynamic",
                    "motion_confidence": 1.0,
                    "geometry_epoch": frame,
                    "readout_valid": frame != 1,
                }
                for frame in range(3)
            ],
        )
        lifecycle = _write_jsonl(
            source_root / "lifecycle_transitions.jsonl",
            [
                {
                    "frame_index": 1,
                    "timestamp_ns": 101,
                    "entity_id": "7",
                    "before": "active",
                    "after": "uncertain",
                    "evidence": "visible_absent",
                    "geometry_epoch": 1,
                    "readout_valid": False,
                }
            ],
        )
        coverage = _write_jsonl(source_root / "frame_coverage.jsonl", coverage_rows)
        counter_names = (
            "proposal_opportunity_count",
            "proposal_trigger_count",
            "reid_opportunity_count",
            "reid_trigger_count",
            "motion_rejection_count",
            "ledger_rejection_count",
            "identity_expiry_count",
            "geometry_reclaim_count",
            "epoch_reset_opportunity_count",
            "epoch_reset_trigger_count",
            "icp_opportunity_count",
            "icp_accept_count",
            "icp_reject_count",
            "ledger_stage_count",
            "ledger_commit_count",
            "ledger_reclaim_count",
        )
        zero_counters = {"ledger_rejection_count", "icp_accept_count"}
        allowed_by_profile = {
            "a0": set(),
            "a1": set(),
            "a2": {
                "proposal_opportunity_count", "proposal_trigger_count",
                "identity_expiry_count", "geometry_reclaim_count",
                "motion_rejection_count", "epoch_reset_opportunity_count",
                "epoch_reset_trigger_count",
            },
            "a3": {
                "proposal_opportunity_count", "proposal_trigger_count",
                "identity_expiry_count", "geometry_reclaim_count",
                "motion_rejection_count", "epoch_reset_opportunity_count",
                "epoch_reset_trigger_count", "ledger_rejection_count",
                "ledger_stage_count", "ledger_commit_count",
                "ledger_reclaim_count",
            },
            "a4": set(counter_names),
        }
        zero_counters |= set(counter_names) - allowed_by_profile[base_profile]
        zero_counters |= {
            "diag_a2_no_proposal_recovery": {
                "proposal_opportunity_count",
                "proposal_trigger_count",
            },
            "diag_a3_masking_only_no_ledger": {
                "ledger_stage_count",
                "ledger_commit_count",
                "ledger_reclaim_count",
            },
            "diag_a4_no_dormant_candidates": {
                "reid_opportunity_count",
                "reid_trigger_count",
            },
            "diag_a4_translation_only_no_icp": {
                "icp_opportunity_count",
                "icp_accept_count",
                "icp_reject_count",
            },
        }.get(candidate_id, set())
        counters = {name: int(name not in zero_counters) for name in counter_names}
        record_id = {
            "proposal_opportunity_count": "proposal:0",
            "proposal_trigger_count": "proposal:0",
            "reid_opportunity_count": "reid:0",
            "reid_trigger_count": "reid:0",
            "motion_rejection_count": (
                "motion:0"
                if candidate_id == "diag_a4_translation_only_no_icp"
                or base_profile in {"a2", "a3"}
                else "icp:0"
            ),
            "identity_expiry_count": "identity:0",
            "geometry_reclaim_count": "geometry:0",
            "epoch_reset_opportunity_count": (
                "motion:0"
                if candidate_id == "diag_a4_translation_only_no_icp"
                or base_profile in {"a2", "a3"}
                else "icp:0"
            ),
            "epoch_reset_trigger_count": (
                "motion:0"
                if candidate_id == "diag_a4_translation_only_no_icp"
                or base_profile in {"a2", "a3"}
                else "icp:0"
            ),
            "icp_opportunity_count": "icp:0",
            "icp_reject_count": "icp:0",
            "ledger_stage_count": "ledger:0",
            "ledger_commit_count": "ledger:0",
            "ledger_reclaim_count": "ledger:0",
        }
        runtime = _write_json(
            source_root / "runtime_diagnostics.json",
            {
                "schema_version": 1,
                "execution_profile": base_profile,
                "processed_frame_count": 3,
                "counters": counters,
                "mechanism_records": {
                    name: ([] if counters[name] == 0 else [record_id[name]])
                    for name in counter_names
                },
                "diagnostic": canonical_diagnostic_claim(
                    temporal_config_from_json({
                        "temporal_readout": materialized["temporal_readout"]
                    })
                ),
            },
        )
        source_index = _write_json(
            source_root / "source_index.json",
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "mode": "causal_checkpoint_exports",
                "method": "OVIV2",
                "scene": "apartment",
                "trajectories": _record(trajectories, relative_to=source_root),
                "lifecycle_transitions": _record(lifecycle, relative_to=source_root),
                "frame_coverage": _record(coverage, relative_to=source_root),
                "runtime_diagnostics": _record(runtime, relative_to=source_root),
            },
        )
        run_manifest = _write_json(
            source_root / "run_manifest.json",
            {
                "schema_version": 2,
                "dataset": "TESSE-CD",
                "protocol_id": "oviv2-tessecd-v2",
                "scene": "apartment",
                "method_id": "OVIV2",
                "algorithm_hash": materialized["algorithm_hash"],
                "processed_frame_count": 3,
                "covered_frame_count": 3,
                "first_frame_index": 0,
                "last_frame_index": 2,
                "scheduled_frame_indices": [1, 2],
                "source_index": _record(source_index, relative_to=source_root),
            },
        )
        mappings = [
            {
                "scene": "apartment",
                "object_id": index,
                "lifecycle_index": 0,
                "anchor_frame_index": 0,
                "anchor_relative_timestamp_ns": 100,
                "eligible": True,
                "target_voxel_count": 1,
                "mapped_temporal_id": index if index < 53 else None,
                "overlap_voxel_count": 1 if index < 53 else 0,
                "ambiguous": False,
            }
            for index in range(66)
        ]
        occlusion = _write_json(
            source_root / "temporal_occlusion_result.json",
            {
                "format": "oviv2_temporal_compact_v1",
                "anchor_mappings": mappings,
                "macro": {
                    "anchor_coverage_gate": {
                        "scene": "apartment",
                        "eligible_count": 66,
                        "uniquely_mapped_count": 53,
                        "zero_overlap_count": 13,
                        "ambiguous_count": 0,
                        "required_eligible_count": 66,
                        "required_mapped_count": 53,
                        "available": True,
                        "passed": True,
                        "reason": None,
                    }
                },
                "input_bindings": {
                    "source_indexes": [
                        {"scene": "apartment", **_record(source_index, relative_to=source_root)}
                    ]
                },
            },
        )
        leakage = _write_json(
            source_root / "future_leakage.json",
            {
                "schema_version": 1,
                "manifest_id": "oviv2_tesse_future_leakage_evidence_v1",
                "scene": "apartment",
                "candidate_id": candidate_id,
                "source_index": _record(source_index, relative_to=source_root),
                "records": [],
            },
        )
        source_evidence = {
            "run_manifest": _record(run_manifest, relative_to=path.parent),
            "source_index": _record(source_index, relative_to=path.parent),
            "trajectories": _record(trajectories, relative_to=path.parent),
            "lifecycle_transitions": _record(lifecycle, relative_to=path.parent),
            "frame_coverage": _record(coverage, relative_to=path.parent),
            "runtime_diagnostics": _record(runtime, relative_to=path.parent),
            "temporal_occlusion_result": _record(occlusion, relative_to=path.parent),
            "future_leakage_evidence": _record(leakage, relative_to=path.parent),
        }
        runtime_hash = source_evidence["runtime_diagnostics"]["sha256"]
        lifecycle_hash = source_evidence["lifecycle_transitions"]["sha256"]
        mechanism_records = {
            "absence": ([0], [0], lifecycle_hash),
            "readout_invalidation": ([0], [0], lifecycle_hash),
            "proposal_recovery": (["proposal:0"], ["proposal:0"], runtime_hash),
            "epoch_reset": (
                [record_id["epoch_reset_opportunity_count"]],
                [record_id["epoch_reset_trigger_count"]],
                runtime_hash,
            ),
            "motion_rejection": (
                [record_id["motion_rejection_count"]],
                [record_id["motion_rejection_count"]],
                runtime_hash,
            ),
            "background_release": (["ledger:0"], ["ledger:0"], runtime_hash),
            "background_reclaim": (["ledger:0"], ["ledger:0"], runtime_hash),
            "eligible_reid": (["reid:0"], ["reid:0"], runtime_hash),
            "icp": (["icp:0"], ["icp:0"], runtime_hash),
        }
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_config_sha256": hashlib.sha256(
                    _canonical(materialized)
                ).hexdigest(),
                "frame_coverage": {
                    "expected_frame_indices": [0, 1, 2],
                    "observed_frame_indices": [0, 1, 2],
                    "expected_count": 3,
                    "observed_count": 3,
                },
                "future_leakage": {"count": 0, "records": []},
                "anchor_coverage": {
                    "eligible_anchor_ids": list(range(66)),
                    "mapped_anchor_ids": list(range(53)),
                    "eligible_count": 66,
                    "mapped_count": 53,
                },
                "mechanisms": {
                    name: {
                        "opportunity_count": 1,
                        "trigger_count": 1,
                        "opportunity_records": [
                            f"{name}:opportunity:{mechanism_records[name][2]}:"
                            f"{mechanism_records[name][0][0]}"
                        ],
                        "trigger_records": [
                            f"{name}:trigger:{mechanism_records[name][2]}:"
                            f"{mechanism_records[name][1][0]}"
                        ],
                    }
                    for name in candidate_mechanisms
                    if name not in disabled
                },
                "source_evidence": source_evidence,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "oviv2_dual_readout_search_preflight_v1",
                "scene": "apartment",
                "bindings": {
                    "search_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                    "apartment_base_config_file_sha256": hashlib.sha256(
                        APARTMENT_CONFIG.read_bytes()
                    ).hexdigest(),
                },
                "candidates": candidates,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_manifest_predeclares_exact_bounded_causal_matrix_and_gates() -> None:
    manifest = load_search_manifest(MANIFEST)

    assert [candidate["candidate_id"] for candidate in manifest["candidates"]] == [
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
    ]
    assert manifest["development_scene"] == "apartment"
    assert manifest["transfer_scene"] == "office"
    assert manifest["transfer_policy"] == "bind_only_never_execute"
    assert manifest["metric_directions"] == {
        "background_f5_cm": "maximize",
        "change_f1": "maximize",
        "current_miou": "maximize",
        "dynamic_f1": "maximize",
        "ghost_rate": "minimize",
        "object_f1": "maximize",
        "recovery_frames": "minimize",
        "runtime_seconds": "minimize",
    }
    assert manifest["metric_policy"]["selection_required"] == [
        "current_miou",
        "object_f1",
        "ghost_rate",
        "background_f5_cm",
        "recovery_frames",
        "runtime_seconds",
    ]
    assert manifest["promotion_order"] == [
        "t1_exact_gate",
        "current_miou_floor",
        "object_f1_floor",
        "ghost_rate",
        "background_f5_cm",
        "recovery_frames",
        "dynamic_f1",
        "change_f1",
        "runtime_seconds",
        "config_sha256",
    ]
    for candidate in manifest["candidates"]:
        profile = ExecutionProfile.from_id(candidate["candidate_id"])
        parsed = temporal_config_from_json(
            {"temporal_readout": candidate["temporal_readout"]}
        )
        assert parsed.execution_profile is profile
        assert candidate["components"] == profile.components
        assert set(candidate["temporal_readout"]) == {
            "execution_profile",
            "components",
            "lifecycle",
            "association",
            "geometry",
            "proposal",
            "identity",
            "dynamic_state",
            "motion",
            "geometry_epoch",
            "background_ledger",
        }
    spaces = manifest["parameter_spaces"]
    assert set(spaces) == {
        "lifecycle.minimum_absent_streak",
        "proposal.minimum_depth_residual_m",
        "dynamic_state.minimum_motion_confidence",
        "identity.minimum_reid_similarity",
        "identity.maximum_reid_distance_m",
        "motion.minimum_translation_confidence",
        "motion.maximum_translation_residual_m",
        "background_ledger.commit_support_frames",
        "background_ledger.commit_distinct_view_bins",
    }
    assert all(1 <= len(space["values"]) <= 3 for space in spaces.values())
    assert all(len(space["values"]) == len(set(space["values"])) for space in spaces.values())
    for candidate in manifest["candidates"]:
        profile_id = candidate["candidate_id"]
        temporal = candidate["temporal_readout"]
        for field, space in spaces.items():
            if profile_id not in space["profiles"]:
                continue
            group, name = field.split(".", 1)
            assert temporal[group][name] in space["values"]

    assert manifest["gpu_lanes"] == {
        "lane0": ["a0", "a1"],
        "lane1": ["a2"],
        "lane2": ["a3", "a4"],
    }
    assert manifest["profile_fallback_order"] == ["a4", "a3", "a2"]
    assert manifest["metric_gates"]["apartment_anchor_coverage"] == {
        "minimum_mapped_anchors": 53,
        "eligible_anchors": 66,
    }


def test_manifest_registers_runnable_nonselectable_micro_ablations() -> None:
    manifest = load_search_manifest(MANIFEST)
    diagnostics = manifest["diagnostic_candidates"]
    assert [item["candidate_id"] for item in diagnostics] == [
        "diag_a2_no_proposal_recovery",
        "diag_a3_masking_only_no_ledger",
        "diag_a4_no_dormant_candidates",
        "diag_a4_translation_only_no_icp",
    ]
    assert [item["base_profile"] for item in diagnostics] == ["a2", "a3", "a4", "a4"]
    assert all(item["diagnostic"] is True for item in diagnostics)
    assert all(item["selectable"] is False for item in diagnostics)
    assert all(item["runnable"] is True for item in diagnostics)
    assert [item["lane"] for item in diagnostics] == ["lane1", "lane2", "lane2", "lane2"]
    assert [item["diagnostic_controls"] for item in diagnostics] == [
        {"proposal_recovery_enabled": False},
        {"background_mode": "masking_only"},
        {"dormant_reid_enabled": False},
        {"icp_enabled": False},
    ]
    assert not (
        {item["candidate_id"] for item in diagnostics}
        & {item["candidate_id"] for item in manifest["candidates"]}
    )


@pytest.mark.parametrize(
    ("candidate_id", "gpu", "control", "zero_names"),
    [
        (
            "diag_a2_no_proposal_recovery",
            "1",
            {"proposal_recovery_enabled": False},
            {"proposal_opportunity_count", "proposal_trigger_count"},
        ),
        (
            "diag_a3_masking_only_no_ledger",
            "2",
            {"background_mode": "masking_only"},
            {"ledger_stage_count", "ledger_commit_count", "ledger_reclaim_count"},
        ),
        (
            "diag_a4_no_dormant_candidates",
            "2",
            {"dormant_reid_enabled": False},
            {"reid_opportunity_count", "reid_trigger_count"},
        ),
        (
            "diag_a4_translation_only_no_icp",
            "2",
            {"icp_enabled": False},
            {
                "icp_opportunity_count",
                "icp_accept_count",
                "icp_reject_count",
            },
        ),
    ],
)
def test_runner_executes_diagnostic_on_fixed_lane_outside_main_inventory(
    tmp_path: Path,
    candidate_id: str,
    gpu: str,
    control: dict[str, object],
    zero_names: set[str],
) -> None:
    preflight = _preflight(tmp_path, (candidate_id,))
    preflight_payload = json.loads(preflight.read_text())
    runtime_path = _source_path(
        preflight, preflight_payload["candidates"][0]["source_evidence"]["runtime_diagnostics"]
    )
    runtime = json.loads(runtime_path.read_text())
    assert all(runtime["counters"][name] == 0 for name in zero_names)
    assert all(runtime["mechanism_records"][name] == [] for name in zero_names)
    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=tmp_path / "diagnostic",
        gpu_ids=("0", "1", "2"),
        max_parallel=1,
        candidate_ids=(candidate_id,),
        preflight_gate_evidence=preflight,
        available_ram_bytes=10**15,
        command_builder=lambda config, output, identity: (
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ),
    )

    status = json.loads(status_path.read_text())
    assert status["candidates"] == []
    assert [item["candidate_id"] for item in status["diagnostic_candidates"]] == [candidate_id]
    record = status["diagnostic_candidates"][0]
    assert record["selectable"] is False
    assert record["cuda_visible_devices"] == gpu
    config = json.loads(Path(record["config_path"]).read_text())
    assert config["temporal_readout"]["execution_profile"] == status[
        "diagnostic_candidates"
    ][0]["base_profile"]
    assert config["temporal_readout"]["diagnostic_controls"] == control


def test_diagnostic_rejects_nonzero_disabled_mechanism_self_report(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path, ("diag_a2_no_proposal_recovery",))
    payload = json.loads(preflight.read_text())
    runtime_path = _source_path(
        preflight, payload["candidates"][0]["source_evidence"]["runtime_diagnostics"]
    )
    runtime = json.loads(runtime_path.read_text())
    runtime["counters"]["proposal_opportunity_count"] = 1
    runtime["counters"]["proposal_trigger_count"] = 1
    runtime["mechanism_records"]["proposal_opportunity_count"] = ["proposal:0"]
    runtime["mechanism_records"]["proposal_trigger_count"] = ["proposal:0"]

    with pytest.raises(ValueError, match="disabled mechanism|profile-impossible"):
        search_runner._diagnostic_recompute_payloads(
            {"runtime_diagnostics": runtime},
            {"proposal_recovery"},
            "diagnostic",
        )


def test_translation_only_diagnostic_disables_icp_but_retains_motion_evidence(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path, ("diag_a4_translation_only_no_icp",))
    payload = json.loads(preflight.read_text())
    runtime_path = _source_path(
        preflight, payload["candidates"][0]["source_evidence"]["runtime_diagnostics"]
    )
    runtime = json.loads(runtime_path.read_text())
    runtime["counters"]["motion_rejection_count"] = 1
    runtime["mechanism_records"]["motion_rejection_count"] = ["motion:1:2:7"]
    runtime["mechanism_records"]["epoch_reset_opportunity_count"] = [
        "motion:1:2:7"
    ]
    runtime["mechanism_records"]["epoch_reset_trigger_count"] = ["motion:1:2:7"]

    prepared = search_runner._diagnostic_recompute_payloads(
        {"runtime_diagnostics": runtime}, {"icp"}, "diagnostic"
    )

    assert prepared["runtime_diagnostics"]["counters"]["motion_rejection_count"] == 1


def _a2_diagnostic_claim() -> dict[str, object]:
    return {
        "identity": "diag_a2_no_proposal_recovery",
        "controls": {"proposal_recovery_enabled": False},
        "component_enabled": {
            "proposal_recovery": False,
            "background_masking": False,
            "background_ledger": False,
            "dormant_reid": False,
            "icp": False,
        },
        "positive_claim_available": {"proposal_recovery": False},
    }


@pytest.mark.parametrize(
    "mutation",
    ("missing", "identity", "controls", "component_enabled", "claim"),
)
def test_search_rejects_noncanonical_diagnostic_claim(
    tmp_path: Path, mutation: str
) -> None:
    preflight = _preflight(tmp_path, ("diag_a2_no_proposal_recovery",))
    payload = json.loads(preflight.read_text())
    runtime = json.loads(_source_path(
        preflight, payload["candidates"][0]["source_evidence"]["runtime_diagnostics"]
    ).read_text())
    runtime["diagnostic"] = _a2_diagnostic_claim()
    if mutation == "missing":
        runtime.pop("diagnostic")
    elif mutation == "identity":
        runtime["diagnostic"]["identity"] = "diag_a4_no_dormant_candidates"
    elif mutation == "controls":
        runtime["diagnostic"]["controls"] = {"icp_enabled": False}
    elif mutation == "component_enabled":
        runtime["diagnostic"]["component_enabled"]["proposal_recovery"] = True
    else:
        runtime["diagnostic"]["positive_claim_available"] = {"icp": False}
    manifest = json.loads(MANIFEST.read_text())
    base = json.loads(APARTMENT_CONFIG.read_text())
    main = {item["candidate_id"]: item for item in manifest["candidates"]}
    diagnostic = next(
        item for item in manifest["diagnostic_candidates"]
        if item["candidate_id"] == "diag_a2_no_proposal_recovery"
    )
    declaration = {**main["a2"], **diagnostic}
    temporal = search_runner._materialize_config(base, declaration)["temporal_readout"]

    with pytest.raises(ValueError, match="diagnostic"):
        search_runner._validate_runtime_mechanism_records(
            runtime,
            "diagnostic",
            expected_temporal_readout=temporal,
            expected_candidate_id=diagnostic["candidate_id"],
        )


def test_search_rejects_main_profile_with_diagnostic_claim(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path, ("a2",))
    payload = json.loads(preflight.read_text())
    runtime = json.loads(_source_path(
        preflight, payload["candidates"][0]["source_evidence"]["runtime_diagnostics"]
    ).read_text())
    runtime["diagnostic"] = _a2_diagnostic_claim()
    manifest = json.loads(MANIFEST.read_text())
    declaration = next(
        item for item in manifest["candidates"] if item["candidate_id"] == "a2"
    )

    with pytest.raises(ValueError, match="diagnostic"):
        search_runner._validate_runtime_mechanism_records(
            runtime,
            "main",
            expected_temporal_readout=declaration["temporal_readout"],
            expected_candidate_id="a2",
        )


def test_search_rejects_main_profile_impossible_counter(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path, ("a2",))
    payload = json.loads(preflight.read_text())
    runtime = json.loads(_source_path(
        preflight, payload["candidates"][0]["source_evidence"]["runtime_diagnostics"]
    ).read_text())
    runtime["counters"]["ledger_stage_count"] = 1
    runtime["mechanism_records"]["ledger_stage_count"] = ["ledger:0"]
    manifest = json.loads(MANIFEST.read_text())
    declaration = next(
        item for item in manifest["candidates"] if item["candidate_id"] == "a2"
    )

    with pytest.raises(ValueError, match="profile-impossible"):
        search_runner._validate_runtime_mechanism_records(
            runtime,
            "main",
            expected_temporal_readout=declaration["temporal_readout"],
            expected_candidate_id="a2",
        )


def test_manifest_rejects_unknown_and_profile_incompatible_parameter_spaces(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["parameter_spaces"]["proposal.unknown"] = {
        "profiles": ["a2"],
        "values": [1],
    }
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parameter_spaces.*unknown"):
        load_search_manifest(unknown)

    payload = json.loads(MANIFEST.read_text())
    payload["parameter_spaces"]["proposal.minimum_depth_residual_m"]["profiles"] = [
        "a1",
        "a2",
    ]
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="profile-incompatible"):
        load_search_manifest(incompatible)


def test_execution_profiles_have_distinct_canonical_behavior() -> None:
    components = {
        tuple(profile.components.items())
        for profile in ExecutionProfile
    }
    assert len(components) == 5


def test_manifest_rejects_duplicate_json_keys_and_undeclared_candidate(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_search_manifest(duplicate)

    with pytest.raises(ValueError, match="undeclared candidate"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "out",
            gpu_ids=("0",),
            max_parallel=1,
            candidate_ids=("a5",),
            available_ram_bytes=10**15,
        )


def test_runner_materializes_unique_apartment_configs_and_never_runs_office(
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, ...]] = []

    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        command = (
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ok')",
            str(output.parent / "executed.txt"),
            candidate_id,
        )
        commands.append(command)
        return command

    root = tmp_path / "search"
    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=root,
        gpu_ids=("2", "4", "6"),
        max_parallel=3,
        preflight_gate_evidence=_preflight(tmp_path, ("a0", "a1", "a2", "a3", "a4")),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    status = json.loads(status_path.read_text())

    assert status["status"] == "PASS"
    assert [record["candidate_id"] for record in status["candidates"]] == [
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
    ]
    assert all(record["scene"] == "apartment" for record in status["candidates"])
    assert all(record["pid"] > 0 and record["exit_code"] == 0 for record in status["candidates"])
    assert all(record["runtime_seconds"] >= 0.0 for record in status["candidates"])
    assert all(
        record["config_file"]["path"] == record["config_path"]
        for record in status["candidates"]
    )
    assert all(
        record["stdout_file"]["path"] == record["stdout_path"]
        for record in status["candidates"]
    )
    assert all(
        record["stderr_file"]["path"] == record["stderr_path"]
        for record in status["candidates"]
    )
    assert {
        record["candidate_id"]: record["cuda_visible_devices"]
        for record in status["candidates"]
    } == {
        "a0": "2",
        "a1": "2",
        "a2": "4",
        "a3": "6",
        "a4": "6",
    }
    assert len({record["output_root"] for record in status["candidates"]}) == 5
    assert len({record["config_path"] for record in status["candidates"]}) == 5
    assert status["office_binding"]["scene"] == "office"
    assert status["office_binding"]["executed"] is False
    assert status["preflight_gate_evidence"]["path"].endswith(
        "preflight-a0-a1-a2-a3-a4.json"
    )
    assert len(status["preflight_gate_evidence"]["sha256"]) == 64
    assert all("office" not in " ".join(command).lower() for command in commands)
    for record in status["candidates"]:
        config_path = Path(record["config_path"])
        config = json.loads(config_path.read_text())
        assert config["scene"] == "apartment"
        assert config["temporal_readout"]["execution_profile"] == record["candidate_id"]
        assert (
            temporal_config_from_json(
                {"temporal_readout": config["temporal_readout"]}
            ).execution_profile.profile_id
            == record["candidate_id"]
        )
        assert hashlib.sha256(_canonical(config)).hexdigest() == record["config_sha256"]

    with pytest.raises(FileExistsError):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=root,
            gpu_ids=("0", "1", "2"),
            max_parallel=3,
            preflight_gate_evidence=_preflight(
                tmp_path, ("a0", "a1", "a2", "a3", "a4")
            ),
            available_ram_bytes=10**15,
        )


def test_runner_stops_new_scheduling_after_first_failure(tmp_path: Path) -> None:
    root = tmp_path / "failed"

    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-c",
            "import sys; sys.exit(7 if sys.argv[1] == 'a1' else 0)",
            candidate_id,
        )

    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=root,
        gpu_ids=("0",),
        max_parallel=1,
        preflight_gate_evidence=_preflight(tmp_path, ("a0", "a1")),
        candidate_ids=("a0", "a1"),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    status = json.loads(status_path.read_text())
    assert status["status"] == "FAIL"
    assert [record["candidate_id"] for record in status["candidates"]] == ["a0", "a1"]
    assert status["candidates"][-1]["exit_code"] == 7
    assert status["unscheduled_candidate_ids"] == []


def test_parallel_runtime_has_no_head_of_line_wait_inflation(tmp_path: Path) -> None:
    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        del config, output
        delay = "0.35" if candidate_id == "a0" else "0.05" if candidate_id == "a2" else "0"
        return (sys.executable, "-c", "import sys,time; time.sleep(float(sys.argv[1]))", delay)

    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=tmp_path / "parallel",
        gpu_ids=("0", "1", "2"),
        max_parallel=3,
        candidate_ids=("a0", "a2"),
        preflight_gate_evidence=_preflight(tmp_path, ("a0", "a2")),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    records = {
        item["candidate_id"]: item for item in json.loads(status_path.read_text())["candidates"]
    }
    assert records["a2"]["runtime_seconds"] < 0.25
    assert records["a2"]["runtime_seconds"] < records["a0"]["runtime_seconds"]


def test_fixed_gpu_lanes_serialize_a0_a1_and_a3_a4(tmp_path: Path) -> None:
    lane0_done = tmp_path / "lane0.done"
    lane2_done = tmp_path / "lane2.done"

    def command_builder(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
        del config, output
        if candidate_id in {"a0", "a3"}:
            marker = lane0_done if candidate_id == "a0" else lane2_done
            return (
                sys.executable,
                "-c",
                "import pathlib,sys,time; time.sleep(.1); pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            )
        if candidate_id in {"a1", "a4"}:
            marker = lane0_done if candidate_id == "a1" else lane2_done
            return (
                sys.executable,
                "-c",
                "import pathlib,sys; "
                "raise SystemExit(0 if pathlib.Path(sys.argv[1]).is_file() else 9)",
                str(marker),
            )
        return (sys.executable, "-c", "raise SystemExit(0)")

    status_path = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=tmp_path / "serialized",
        gpu_ids=("0", "1", "2"),
        max_parallel=3,
        preflight_gate_evidence=_preflight(
            tmp_path, ("a0", "a1", "a2", "a3", "a4")
        ),
        available_ram_bytes=10**15,
        command_builder=command_builder,
    )
    assert json.loads(status_path.read_text())["status"] == "PASS"


def test_preflight_recomputes_source_coverage_and_mechanism_records(
    tmp_path: Path,
) -> None:
    gate_path = _preflight(tmp_path, ("a2",))
    payload = json.loads(gate_path.read_text())
    payload["bindings"]["search_manifest_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source bindings mismatch"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "bad-binding",
            gpu_ids=("0", "1"),
            candidate_ids=("a2",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )

    gate_path = _preflight(tmp_path, ("a2",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["frame_coverage"]["observed_frame_indices"] = [0, 1]
    payload["candidates"][0]["frame_coverage"]["observed_count"] = 2
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frame_coverage differs from raw source"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "bad-coverage",
            gpu_ids=("0", "1"),
            candidate_ids=("a2",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )

    gate_path = _preflight(tmp_path, ("a2",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["mechanisms"]["proposal_recovery"][
        "opportunity_count"
    ] = 2
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mechanism records/counts differ from raw source"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "bad-mechanism",
            gpu_ids=("0", "1"),
            candidate_ids=("a2",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )


def test_runner_accepts_producer_source_evidence_end_to_end(tmp_path: Path) -> None:
    from scripts.evaluation.build_oviv2_tesse_search_preflight import build_preflight

    fixture = _preflight(tmp_path, ("a1",))
    fixture_payload = json.loads(fixture.read_text())
    evidence = fixture_payload["candidates"][0]["source_evidence"]
    candidate_sources = _write_json(
        tmp_path / "producer-input.json",
        {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_search_preflight_sources_v1",
            "candidates": [
                {
                    "candidate_id": "a1",
                    "run_manifest": evidence["run_manifest"],
                    "temporal_occlusion_result": evidence["temporal_occlusion_result"],
                    "future_leakage_evidence": evidence["future_leakage_evidence"],
                }
            ],
        },
    )
    produced = tmp_path / "producer-preflight.json"
    build_preflight(
        search_manifest=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        candidate_sources=candidate_sources,
        output=produced,
    )

    status = run_search(
        manifest_path=MANIFEST,
        apartment_base_config=APARTMENT_CONFIG,
        office_base_config=OFFICE_CONFIG,
        output_root=tmp_path / "producer-consumer",
        gpu_ids=("0",),
        candidate_ids=("a1",),
        preflight_gate_evidence=produced,
        available_ram_bytes=10**15,
        command_builder=lambda config, output, identity: (
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ),
    )

    assert json.loads(status.read_text())["status"] == "PASS"


def test_runner_rejects_raw_hash_drift_even_when_summary_still_passes(
    tmp_path: Path,
) -> None:
    gate = _preflight(tmp_path, ("a1",))
    payload = json.loads(gate.read_text())
    raw = _source_path(gate, payload["candidates"][0]["source_evidence"]["frame_coverage"])
    raw.write_bytes(raw.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="frame_coverage source binding mismatch"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "hash-drift",
            gpu_ids=("0",),
            candidate_ids=("a1",),
            preflight_gate_evidence=gate,
            available_ram_bytes=10**15,
        )


@pytest.mark.parametrize("attack", ["absolute", "dotdot", "symlink"])
def test_runner_rejects_source_path_aliases_and_symlinks(
    tmp_path: Path, attack: str
) -> None:
    gate = _preflight(tmp_path, ("a1",))
    payload = json.loads(gate.read_text())
    record = payload["candidates"][0]["source_evidence"]["trajectories"]
    original = _source_path(gate, record)
    if attack == "absolute":
        record["path"] = str(original)
    elif attack == "dotdot":
        record["path"] = str(Path(record["path"]).parent / "nested" / ".." / original.name)
    else:
        alias = original.with_name("trajectories-alias.jsonl")
        alias.symlink_to(original)
        record["path"] = alias.relative_to(gate.parent).as_posix()
    gate.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="alias|symlink"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / f"path-{attack}",
            gpu_ids=("0",),
            candidate_ids=("a1",),
            preflight_gate_evidence=gate,
            available_ram_bytes=10**15,
        )


def test_runner_rejects_source_inventory_inode_alias(tmp_path: Path) -> None:
    gate = _preflight(tmp_path, ("a1",))
    payload = json.loads(gate.read_text())
    evidence = payload["candidates"][0]["source_evidence"]
    original = _source_path(gate, evidence["trajectories"])
    alias = original.with_name("trajectories-hardlink.jsonl")
    os.link(original, alias)
    evidence["trajectories"]["path"] = alias.relative_to(gate.parent).as_posix()
    evidence["lifecycle_transitions"] = dict(evidence["trajectories"])
    gate.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="same inode"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "inode-alias",
            gpu_ids=("0",),
            candidate_ids=("a1",),
            preflight_gate_evidence=gate,
            available_ram_bytes=10**15,
        )


def test_runner_rejects_nested_source_record_path_alias(tmp_path: Path) -> None:
    gate = _preflight(tmp_path, ("a1",))
    payload = json.loads(gate.read_text())
    evidence = payload["candidates"][0]["source_evidence"]
    source_index_path = _source_path(gate, evidence["source_index"])
    source_index = json.loads(source_index_path.read_text())
    source_index["trajectories"]["path"] = str(
        source_index_path.parent / "nested" / ".." / "trajectories.jsonl"
    )
    _write_json(source_index_path, source_index)
    evidence["source_index"] = _record(source_index_path, relative_to=gate.parent)
    run_path = _source_path(gate, evidence["run_manifest"])
    run = json.loads(run_path.read_text())
    run["source_index"] = _record(source_index_path, relative_to=source_index_path.parent)
    _write_json(run_path, run)
    evidence["run_manifest"] = _record(run_path, relative_to=gate.parent)
    gate.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="source path is an alias"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "nested-path-alias",
            gpu_ids=("0",),
            candidate_ids=("a1",),
            preflight_gate_evidence=gate,
            available_ram_bytes=10**15,
        )


def test_runner_rejects_raw_occlusion_source_index_splice(tmp_path: Path) -> None:
    gate = _preflight(tmp_path, ("a1",))
    payload = json.loads(gate.read_text())
    evidence = payload["candidates"][0]["source_evidence"]
    original_index = _source_path(gate, evidence["source_index"])
    spliced_index = original_index.with_name("spliced_source_index.json")
    spliced_index.write_bytes(original_index.read_bytes())
    occlusion_path = _source_path(gate, evidence["temporal_occlusion_result"])
    occlusion = json.loads(occlusion_path.read_text())
    occlusion["input_bindings"]["source_indexes"][0] = {
        "scene": "apartment",
        **_record(spliced_index, relative_to=original_index.parent),
    }
    _write_json(occlusion_path, occlusion)
    evidence["temporal_occlusion_result"] = _record(occlusion_path, relative_to=gate.parent)
    gate.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="occlusion source_index binding"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "raw-splice",
            gpu_ids=("0",),
            candidate_ids=("a1",),
            preflight_gate_evidence=gate,
            available_ram_bytes=10**15,
        )


def test_runner_revalidates_all_raw_witnesses_immediately_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _preflight(tmp_path, ("a1",))
    payload = json.loads(gate.read_text())
    raw = _source_path(gate, payload["candidates"][0]["source_evidence"]["runtime_diagnostics"])
    original_content = raw.read_bytes()
    original_status = os.stat(raw, follow_symlinks=False)
    original_stat = search_runner.os.stat

    def pinned_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if Path(path) == raw:
            return original_status
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(search_runner.os, "stat", pinned_stat)

    def mutate_during_command_build(
        config: Path, output: Path, candidate_id: str
    ) -> tuple[str, ...]:
        del config, output, candidate_id
        replacement = bytearray(original_content)
        replacement[-2] = ord(" ") if replacement[-2] != ord(" ") else ord("x")
        raw.write_bytes(replacement)
        assert len(raw.read_bytes()) == len(original_content)
        return (sys.executable, "-c", "raise SystemExit(0)")

    with pytest.raises(ValueError, match="changed before launch"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "toctou",
            gpu_ids=("0",),
            candidate_ids=("a1",),
            preflight_gate_evidence=gate,
            available_ram_bytes=10**15,
            command_builder=mutate_during_command_build,
        )


def test_default_runner_requires_run_manifest_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        search_runner,
        "_default_command",
        lambda config, output, candidate_id: (
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ),
    )

    with pytest.raises(ValueError, match="run manifest"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "missing-run-manifest",
            gpu_ids=("0",),
            max_parallel=1,
            candidate_ids=("a0",),
            preflight_gate_evidence=_preflight(tmp_path, ("a0",)),
            available_ram_bytes=10**15,
        )


def test_runner_requires_all_preflight_gates_before_output_creation(tmp_path: Path) -> None:
    output = tmp_path / "missing-gates"
    with pytest.raises(ValueError, match="preflight gate evidence is required"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=output,
            gpu_ids=("0",),
            candidate_ids=("a0",),
            available_ram_bytes=10**15,
        )
    assert not output.exists()

    gate_path = _preflight(tmp_path, ("a0",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["future_leakage"] = {
        "count": 1,
        "records": ["a0:frame2:query3"],
    }
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="future_leakage.*PASS"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=output,
            gpu_ids=("0",),
            candidate_ids=("a0",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )
    assert not output.exists()


def test_runner_enforces_apartment_anchor_53_of_66_before_launch(tmp_path: Path) -> None:
    gate_path = _preflight(tmp_path, ("a0",))
    payload = json.loads(gate_path.read_text())
    payload["candidates"][0]["anchor_coverage"]["mapped_anchor_ids"] = list(range(52))
    payload["candidates"][0]["anchor_coverage"]["mapped_count"] = 52
    gate_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="anchor coverage.*53/66"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "anchor-fail",
            gpu_ids=("0",),
            candidate_ids=("a0",),
            preflight_gate_evidence=gate_path,
            available_ram_bytes=10**15,
        )


def test_runner_rejects_mixed_main_diagnostic_and_office_search(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="main and diagnostic candidates"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "diagnostic",
            gpu_ids=("0", "1", "2"),
            candidate_ids=("a2", "diag_a2_no_proposal_recovery"),
            available_ram_bytes=10**15,
        )
    with pytest.raises(ValueError, match="Office search requires frozen authorization"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "office",
            gpu_ids=("0",),
            scene="office",
            available_ram_bytes=10**15,
        )


@pytest.mark.parametrize("max_parallel", [0, 4, True])
def test_runner_rejects_invalid_parallelism(tmp_path: Path, max_parallel: object) -> None:
    with pytest.raises(ValueError, match="max_parallel"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=tmp_path / "out",
            gpu_ids=("0",),
            max_parallel=max_parallel,  # type: ignore[arg-type]
            available_ram_bytes=10**15,
        )


def test_runner_checks_available_ram_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="available RAM"):
        run_search(
            manifest_path=MANIFEST,
            apartment_base_config=APARTMENT_CONFIG,
            office_base_config=OFFICE_CONFIG,
            output_root=output,
            gpu_ids=("0",),
            max_parallel=1,
            candidate_ids=("a0",),
            available_ram_bytes=0,
        )
    assert not output.exists()


def test_available_ram_uses_linux_memavailable_including_reclaimable_cache(
    tmp_path: Path,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       1000000 kB\n"
        "MemFree:          12000 kB\n"
        "MemAvailable:    800000 kB\n"
        "Cached:          700000 kB\n"
    )

    assert search_runner._available_ram_bytes(meminfo) == 800000 * 1024
