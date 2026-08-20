from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash
from scripts.evaluation.build_oviv2_tesse_search_preflight import (
    _anchor_coverage,
    _build_preflight_at,
    _mechanisms,
    build_preflight,
)
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    _materialize_config,
    _temporal_frontend_manifest_binding,
    _validate_preflight_gate_evidence,
    _validate_preflight_gate_evidence_at,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_MANIFEST = REPO_ROOT / "configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json"
BASE_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_apartment_v2.json"


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))
    return path


@pytest.fixture(autouse=True)
def _hermetic_base_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporal_manifest = _write(
        tmp_path / "scene-inputs/temporal-frontend.json",
        {"schema_version": 1, "cache": "temporal"},
    )
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["temporal_frontend_manifest"] = str(temporal_manifest)
    config["algorithm_hash"] = canonical_algorithm_hash(config)
    hermetic_config = _write(tmp_path / "scene-inputs/apartment.json", config)
    monkeypatch.setattr(sys.modules[__name__], "BASE_CONFIG", hermetic_config)


def _record(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _candidate_sources(
    tmp_path: Path,
    candidate_id: str = "a1",
    *,
    include_source_bindings: bool = True,
) -> Path:
    manifest = json.loads(SEARCH_MANIFEST.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    main = {item["candidate_id"]: item for item in manifest["candidates"]}
    diagnostics_by_id = {
        item["candidate_id"]: item for item in manifest["diagnostic_candidates"]
    }
    diagnostic = diagnostics_by_id.get(candidate_id)
    base_profile = diagnostic["base_profile"] if diagnostic else candidate_id
    declaration = (
        {**main[base_profile], **diagnostic}
        if diagnostic is not None
        else main[candidate_id]
    )
    root = tmp_path / candidate_id
    coverage = _write(
        root / "temporal_frame_coverage.jsonl",
        [
            {
                "frame_index": frame,
                "timestamp_ns": 100 + frame,
                "record_count": 1,
                "event_count": int(frame == 1),
            }
            for frame in range(5)
        ],
    )
    # Convert the JSON array helper output into canonical JSONL.
    coverage.write_bytes(
        b"".join(
            _bytes(
                {
                    "frame_index": frame,
                    "timestamp_ns": 100 + frame,
                    "record_count": 1,
                    "event_count": int(frame == 1),
                }
            )
            for frame in range(5)
        )
    )
    trajectories = root / "trajectories.jsonl"
    trajectories.write_bytes(
        b"".join(
            _bytes(
                {
                    "frame_index": frame,
                    "timestamp_ns": 100 + frame,
                    "entity_id": "7",
                    "centroid_xyz": [0.0, 0.0, 0.0],
                    "observation_count": frame + 1,
                    "dynamic_state": "dynamic",
                    "motion_confidence": 1.0,
                    "geometry_epoch": frame,
                    "readout_valid": frame == 0,
                }
            )
            for frame in range(5)
        )
    )
    lifecycle = root / "lifecycle_transitions.jsonl"
    lifecycle.write_bytes(
        _bytes(
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
        )
    )
    counters = {
        "proposal_opportunity_count": 1,
        "proposal_trigger_count": 1,
        "reid_opportunity_count": 1,
        "reid_trigger_count": 1,
        "motion_rejection_count": 1,
        "ledger_rejection_count": 0,
        "identity_expiry_count": 1,
        "geometry_reclaim_count": 1,
        "epoch_reset_opportunity_count": 1,
        "epoch_reset_trigger_count": 1,
        "icp_opportunity_count": 1,
        "icp_accept_count": 0,
        "icp_reject_count": 1,
        "ledger_stage_count": 1,
        "ledger_commit_count": 1,
        "ledger_reclaim_count": 1,
    }
    disabled_counters = {
        "diag_a2_no_proposal_recovery": {
            "proposal_opportunity_count", "proposal_trigger_count",
        },
        "diag_a3_masking_only_no_ledger": {
            "ledger_stage_count", "ledger_commit_count", "ledger_reclaim_count",
        },
        "diag_a4_no_dormant_candidates": {
            "reid_opportunity_count", "reid_trigger_count",
        },
        "diag_a4_translation_only_no_icp": {
            "icp_opportunity_count", "icp_accept_count", "icp_reject_count",
        },
    }.get(candidate_id, set())
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
            "ledger_stage_count", "ledger_commit_count", "ledger_reclaim_count",
        },
        "a4": set(counters),
    }
    disabled_counters |= set(counters) - allowed_by_profile[base_profile]
    counters.update({name: 0 for name in disabled_counters})
    mechanism_records = {
        "proposal_opportunity_count": ["proposal:0"],
        "proposal_trigger_count": ["proposal:0"],
        "reid_opportunity_count": ["reid:0"],
        "reid_trigger_count": ["reid:0"],
        "motion_rejection_count": ["motion:0"],
        "ledger_rejection_count": [],
        "identity_expiry_count": ["identity:0"],
        "geometry_reclaim_count": ["geometry:0"],
        "epoch_reset_opportunity_count": ["motion:0"],
        "epoch_reset_trigger_count": ["motion:0"],
        "icp_opportunity_count": ["motion:0"],
        "icp_accept_count": [],
        "icp_reject_count": ["motion:0"],
        "ledger_stage_count": ["ledger:0"],
        "ledger_commit_count": ["ledger:0"],
        "ledger_reclaim_count": ["ledger:0"],
    }
    for name in disabled_counters:
        mechanism_records[name] = []
    diagnostic_payload = None
    if diagnostic is not None:
        enabled = {
            "proposal_recovery": base_profile in {"a2", "a3", "a4"},
            "background_masking": base_profile in {"a3", "a4"},
            "background_ledger": base_profile in {"a3", "a4"},
            "dormant_reid": base_profile == "a4",
            "icp": base_profile == "a4",
        }
        disabled_component = {
            "diag_a2_no_proposal_recovery": "proposal_recovery",
            "diag_a3_masking_only_no_ledger": "background_ledger",
            "diag_a4_no_dormant_candidates": "dormant_reid",
            "diag_a4_translation_only_no_icp": "icp",
        }[candidate_id]
        enabled[disabled_component] = False
        diagnostic_payload = {
            "identity": candidate_id,
            "controls": diagnostic["diagnostic_controls"],
            "component_enabled": enabled,
            "positive_claim_available": {disabled_component: False},
        }
    diagnostics = _write(
        root / "runtime_diagnostics.json",
        {
            "schema_version": 1,
            "execution_profile": base_profile,
            "processed_frame_count": 5,
            "counters": counters,
            "mechanism_records": mechanism_records,
            "diagnostic": diagnostic_payload,
        },
    )
    source_index = _write(
        root / "source_index.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": "apartment",
            "trajectories": _record(trajectories, root),
            "lifecycle_transitions": _record(lifecycle, root),
            "frame_coverage": _record(coverage, root),
            "runtime_diagnostics": _record(diagnostics, root),
        },
    )
    materialized = _materialize_config(base, declaration)
    run_payload = {
        "schema_version": 2,
        "dataset": "TESSE-CD",
        "protocol_id": "oviv2-tessecd-v2",
        "scene": "apartment",
        "method_id": "OVIV2",
        "algorithm_hash": materialized["algorithm_hash"],
        "processed_frame_count": 5,
        "covered_frame_count": 5,
        "first_frame_index": 0,
        "last_frame_index": 4,
        "scheduled_frame_indices": [1, 2, 3, 4],
        "source_index": _record(source_index, root),
    }
    if include_source_bindings:
        temporal_binding = _temporal_frontend_manifest_binding(materialized)
        run_payload["source_bindings"] = (
            {}
            if temporal_binding is None
            else {"temporal_frontend_manifest": temporal_binding}
        )
    run_manifest = _write(root / "run_manifest.json", run_payload)
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
    gate = {
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
    occlusion = _write(
        root / "temporal_occlusion_result.json",
        {
            "format": "oviv2_temporal_compact_v1",
            "anchor_mappings": mappings,
            "macro": {"anchor_coverage_gate": gate},
            "input_bindings": {
                "source_indexes": [
                    {"scene": "apartment", **_record(source_index, root)}
                ]
            },
        },
    )
    leakage = _write(
        root / "future_leakage.json",
        {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_future_leakage_evidence_v1",
            "scene": "apartment",
            "candidate_id": candidate_id,
            "source_index": _record(source_index, root),
            "records": [],
        },
    )
    return _write(
        tmp_path / "candidate_sources.json",
        {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_search_preflight_sources_v1",
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "run_manifest": _record(run_manifest, tmp_path),
                    "temporal_occlusion_result": _record(occlusion, tmp_path),
                    "future_leakage_evidence": _record(leakage, tmp_path),
                }
            ],
        },
    )


def test_builds_source_recomputed_preflight_consumable_by_search(tmp_path: Path) -> None:
    sources = _candidate_sources(tmp_path)
    output = tmp_path / "preflight.json"

    result = build_preflight(
        search_manifest=SEARCH_MANIFEST,
        apartment_base_config=BASE_CONFIG,
        candidate_sources=sources,
        output=output,
    )

    source_evidence = result["candidates"][0]["source_evidence"]
    assert set(source_evidence) == {
        "run_manifest",
        "source_index",
        "trajectories",
        "lifecycle_transitions",
        "frame_coverage",
        "runtime_diagnostics",
        "temporal_occlusion_result",
        "future_leakage_evidence",
    }
    for record in source_evidence.values():
        source = output.parent / record["path"]
        content = source.read_bytes()
        assert record == {
            "path": source.relative_to(output.parent).as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        }

    manifest = json.loads(SEARCH_MANIFEST.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    validated = _validate_preflight_gate_evidence(
        output,
        manifest_bytes=SEARCH_MANIFEST.read_bytes(),
        apartment_bytes=BASE_CONFIG.read_bytes(),
        apartment=base,
        declarations={item["candidate_id"]: item for item in manifest["candidates"]},
        selected_ids=("a1",),
    )
    assert validated["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["candidates"][0]["frame_coverage"]["observed_frame_indices"] == [0, 1, 2, 3, 4]
    assert result["candidates"][0]["anchor_coverage"]["mapped_count"] == 53
    invalidation = result["candidates"][0]["mechanisms"]["readout_invalidation"]
    assert invalidation["opportunity_count"] == 1
    assert invalidation["trigger_count"] == 1
    assert hashlib.sha256(
        (tmp_path / "a1/lifecycle_transitions.jsonl").read_bytes()
    ).hexdigest() in invalidation["trigger_records"][0]


def test_builder_rejects_run_manifest_without_source_bindings(tmp_path: Path) -> None:
    sources = _candidate_sources(tmp_path, include_source_bindings=False)

    with pytest.raises(ValueError, match="source_bindings"):
        build_preflight(
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources,
            output=tmp_path / "preflight.json",
        )


def test_builder_rejects_wrong_temporal_frontend_source_binding(
    tmp_path: Path,
) -> None:
    sources = _candidate_sources(tmp_path, "a2")
    source_index = json.loads(sources.read_text(encoding="utf-8"))
    run_record = source_index["candidates"][0]["run_manifest"]
    run_path = sources.parent / run_record["path"]
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["source_bindings"]["temporal_frontend_manifest"]["sha256"] = "0" * 64
    _write(run_path, run)
    source_index["candidates"][0]["run_manifest"] = _record(
        run_path, sources.parent
    )
    _write(sources, source_index)

    with pytest.raises(ValueError, match="temporal frontend"):
        build_preflight(
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources,
            output=tmp_path / "preflight.json",
        )


def test_fd_backed_builder_and_validator_use_held_bundle_directory(
    tmp_path: Path,
) -> None:
    sources = _candidate_sources(tmp_path)
    manifest = json.loads(SEARCH_MANIFEST.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result = _build_preflight_at(
            root_fd=descriptor,
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources.relative_to(tmp_path),
            output=Path("preflight-fd.json"),
        )
        validated = _validate_preflight_gate_evidence_at(
            descriptor,
            Path("preflight-fd.json"),
            manifest_bytes=SEARCH_MANIFEST.read_bytes(),
            apartment_bytes=BASE_CONFIG.read_bytes(),
            apartment=base,
            declarations={item["candidate_id"]: item for item in manifest["candidates"]},
            selected_ids=("a1",),
        )
    finally:
        os.close(descriptor)

    output = tmp_path / "preflight-fd.json"
    assert result["manifest_id"] == "oviv2_dual_readout_search_preflight_v1"
    assert validated["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_relative_source_evidence_survives_whole_bundle_rename_without_rewrite(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging-bundle"
    sources = _candidate_sources(staging)
    preflight = staging / "preflight.json"
    build_preflight(
        search_manifest=SEARCH_MANIFEST,
        apartment_base_config=BASE_CONFIG,
        candidate_sources=sources,
        output=preflight,
    )
    manifest = json.loads(SEARCH_MANIFEST.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    kwargs = {
        "manifest_bytes": SEARCH_MANIFEST.read_bytes(),
        "apartment_bytes": BASE_CONFIG.read_bytes(),
        "apartment": base,
        "declarations": {item["candidate_id"]: item for item in manifest["candidates"]},
        "selected_ids": ("a1",),
    }
    _validate_preflight_gate_evidence(preflight, **kwargs)
    original = preflight.read_bytes()

    published = tmp_path / "published-bundle"
    staging.rename(published)

    moved = published / "preflight.json"
    assert moved.read_bytes() == original
    _validate_preflight_gate_evidence(moved, **kwargs)


@pytest.mark.parametrize(
    ("candidate_id", "disabled_counters", "disabled_mechanisms"),
    (
        (
            "diag_a2_no_proposal_recovery",
            {"proposal_opportunity_count", "proposal_trigger_count"},
            {"proposal_recovery"},
        ),
        (
            "diag_a3_masking_only_no_ledger",
            {"ledger_stage_count", "ledger_commit_count", "ledger_reclaim_count"},
            {"background_release", "background_reclaim"},
        ),
        (
            "diag_a4_no_dormant_candidates",
            {"reid_opportunity_count", "reid_trigger_count"},
            {"eligible_reid"},
        ),
        (
            "diag_a4_translation_only_no_icp",
            {"icp_opportunity_count", "icp_accept_count", "icp_reject_count"},
            {"icp"},
        ),
    ),
)
def test_builds_diagnostic_preflight_from_raw_sources_for_search_consumer(
    tmp_path: Path,
    candidate_id: str,
    disabled_counters: set[str],
    disabled_mechanisms: set[str],
) -> None:
    sources = _candidate_sources(tmp_path, candidate_id)
    output = tmp_path / "preflight.json"

    result = build_preflight(
        search_manifest=SEARCH_MANIFEST,
        apartment_base_config=BASE_CONFIG,
        candidate_sources=sources,
        output=output,
    )

    runtime = json.loads(
        (tmp_path / candidate_id / "runtime_diagnostics.json").read_text()
    )
    assert all(runtime["counters"][name] == 0 for name in disabled_counters)
    assert all(runtime["mechanism_records"][name] == [] for name in disabled_counters)
    assert disabled_mechanisms.isdisjoint(result["candidates"][0]["mechanisms"])

    manifest = json.loads(SEARCH_MANIFEST.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    declarations = {item["candidate_id"]: item for item in manifest["candidates"]}
    for diagnostic in manifest["diagnostic_candidates"]:
        declarations[diagnostic["candidate_id"]] = {
            **declarations[diagnostic["base_profile"]],
            **diagnostic,
        }
    validated = _validate_preflight_gate_evidence(
        output,
        manifest_bytes=SEARCH_MANIFEST.read_bytes(),
        apartment_bytes=BASE_CONFIG.read_bytes(),
        apartment=base,
        declarations=declarations,
        selected_ids=(candidate_id,),
    )
    assert validated["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_rejects_spliced_occlusion_source_index(tmp_path: Path) -> None:
    sources = _candidate_sources(tmp_path)
    occlusion_path = tmp_path / "a1/temporal_occlusion_result.json"
    payload = json.loads(occlusion_path.read_text())
    payload["input_bindings"]["source_indexes"][0]["sha256"] = "0" * 64
    _write(occlusion_path, payload)
    index = json.loads(sources.read_text())
    index["candidates"][0]["temporal_occlusion_result"] = _record(occlusion_path, tmp_path)
    _write(sources, index)

    with pytest.raises(ValueError, match="occlusion source_index binding"):
        build_preflight(
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources,
            output=tmp_path / "preflight.json",
        )


def test_rejects_nonzero_future_leakage_before_publication(tmp_path: Path) -> None:
    sources = _candidate_sources(tmp_path)
    leakage_path = tmp_path / "a1/future_leakage.json"
    leakage = json.loads(leakage_path.read_text())
    leakage["records"] = [{"consumer_frame": 0, "source_frame": 1}]
    _write(leakage_path, leakage)
    index = json.loads(sources.read_text())
    index["candidates"][0]["future_leakage_evidence"] = _record(
        leakage_path, tmp_path
    )
    _write(sources, index)

    with pytest.raises(ValueError, match="future leakage detected"):
        build_preflight(
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources,
            output=tmp_path / "preflight.json",
        )
    assert not (tmp_path / "preflight.json").exists()


def test_rejects_nonexact_anchor_mapping_schema(tmp_path: Path) -> None:
    sources = _candidate_sources(tmp_path)
    occlusion_path = tmp_path / "a1/temporal_occlusion_result.json"
    occlusion = json.loads(occlusion_path.read_text())
    occlusion["anchor_mappings"][0]["cached_pass"] = True
    _write(occlusion_path, occlusion)
    index = json.loads(sources.read_text())
    index["candidates"][0]["temporal_occlusion_result"] = _record(
        occlusion_path, tmp_path
    )
    _write(sources, index)

    with pytest.raises(ValueError, match="anchor mapping schema"):
        build_preflight(
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources,
            output=tmp_path / "preflight.json",
        )


def test_rejects_more_than_exact_66_eligible_anchors(tmp_path: Path) -> None:
    _candidate_sources(tmp_path)
    occlusion = json.loads((tmp_path / "a1/temporal_occlusion_result.json").read_text())
    extra = dict(occlusion["anchor_mappings"][0])
    extra["object_id"] = 66
    occlusion["anchor_mappings"].append(extra)
    occlusion["macro"]["anchor_coverage_gate"].update(
        eligible_count=67, uniquely_mapped_count=54
    )

    with pytest.raises(ValueError, match="exactly 66"):
        _anchor_coverage(occlusion)


def test_rejects_required_mechanism_without_opportunity(tmp_path: Path) -> None:
    _candidate_sources(tmp_path)
    root = tmp_path / "a1"
    source_index = json.loads((root / "source_index.json").read_text())
    payloads = {
        "trajectories": [json.loads(line) for line in (root / "trajectories.jsonl").read_text().splitlines()],
        "lifecycle_transitions": [
            {
                **json.loads((root / "lifecycle_transitions.jsonl").read_text()),
                "evidence": "occluded",
            }
        ],
        "frame_coverage": [json.loads(line) for line in (root / "temporal_frame_coverage.jsonl").read_text().splitlines()],
        "runtime_diagnostics": json.loads((root / "runtime_diagnostics.json").read_text()),
    }
    with pytest.raises(ValueError, match="absence.*opportunity"):
        _mechanisms(
            candidate_id="a1",
            source_records={name: source_index[name] for name in (
                "trajectories", "lifecycle_transitions", "frame_coverage", "runtime_diagnostics"
            )},
            payloads=payloads,
        )


def test_rejects_runtime_counter_without_explicit_event_records(tmp_path: Path) -> None:
    _candidate_sources(tmp_path, "a2")
    root = tmp_path / "a2"
    source_index = json.loads((root / "source_index.json").read_text())
    diagnostics = json.loads((root / "runtime_diagnostics.json").read_text())
    diagnostics["mechanism_records"]["proposal_trigger_count"] = []
    payloads = {
        "trajectories": [json.loads(line) for line in (root / "trajectories.jsonl").read_text().splitlines()],
        "lifecycle_transitions": [json.loads(line) for line in (root / "lifecycle_transitions.jsonl").read_text().splitlines()],
        "frame_coverage": [json.loads(line) for line in (root / "temporal_frame_coverage.jsonl").read_text().splitlines()],
        "runtime_diagnostics": diagnostics,
    }
    with pytest.raises(ValueError, match="mechanism_records.*counter"):
        _mechanisms(
            candidate_id="a1",
            source_records={name: source_index[name] for name in (
                "trajectories", "lifecycle_transitions", "frame_coverage", "runtime_diagnostics"
            )},
            payloads=payloads,
        )


@pytest.mark.parametrize("candidate_id", ("a2", "a3"))
def test_translation_motion_records_do_not_require_icp_records(
    tmp_path: Path, candidate_id: str
) -> None:
    _candidate_sources(tmp_path, candidate_id)
    root = tmp_path / candidate_id
    source_index = json.loads((root / "source_index.json").read_text())
    diagnostics = json.loads((root / "runtime_diagnostics.json").read_text())
    for name in ("icp_opportunity_count", "icp_accept_count", "icp_reject_count"):
        diagnostics["counters"][name] = 0
        diagnostics["mechanism_records"][name] = []
    diagnostics["counters"]["motion_rejection_count"] = 1
    diagnostics["mechanism_records"]["motion_rejection_count"] = ["motion:1:2:7"]
    diagnostics["mechanism_records"]["epoch_reset_opportunity_count"] = [
        "motion:1:2:7"
    ]
    diagnostics["mechanism_records"]["epoch_reset_trigger_count"] = [
        "motion:1:2:7"
    ]
    payloads = {
        "trajectories": [json.loads(line) for line in (root / "trajectories.jsonl").read_text().splitlines()],
        "lifecycle_transitions": [json.loads(line) for line in (root / "lifecycle_transitions.jsonl").read_text().splitlines()],
        "frame_coverage": [json.loads(line) for line in (root / "temporal_frame_coverage.jsonl").read_text().splitlines()],
        "runtime_diagnostics": diagnostics,
    }

    mechanisms = _mechanisms(
        candidate_id=candidate_id,
        source_records={name: source_index[name] for name in (
            "trajectories", "lifecycle_transitions", "frame_coverage", "runtime_diagnostics"
        )},
        payloads=payloads,
    )

    assert mechanisms["motion_rejection"]["opportunity_count"] == 1
    assert mechanisms["motion_rejection"]["trigger_count"] == 1


def test_rejects_drifted_runtime_diagnostics_and_no_clobber(tmp_path: Path) -> None:
    sources = _candidate_sources(tmp_path)
    (tmp_path / "a1/runtime_diagnostics.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="runtime_diagnostics source binding"):
        build_preflight(
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources,
            output=tmp_path / "preflight.json",
        )

    sources = _candidate_sources(tmp_path / "fresh")
    output = tmp_path / "existing.json"
    output.write_bytes(b"keep\n")
    with pytest.raises(FileExistsError):
        build_preflight(
            search_manifest=SEARCH_MANIFEST,
            apartment_base_config=BASE_CONFIG,
            candidate_sources=sources,
            output=output,
        )
    assert output.read_bytes() == b"keep\n"
