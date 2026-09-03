from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/external/khronos_source_manifest.json"
REPORT = (
    ROOT
    / "docs/superpowers/reports/2026-09-03-khronos-protocol-reproduction.md"
)

RSS_RELEASE = "742227a88de8b2ac23ac54d719b321c3af88dc75"
LATEST_PUBLIC = "63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e"
ALLOWED_STATUSES = {
    "PAPER_PROTOCOL_EXACT",
    "PAPER_PROTOCOL_APPROX_PUBLIC",
    "BLOCKED_PAPER_ASSET",
    "BENCHMARK_REPRODUCTION_NO_GO",
}
REFERENCE_METRICS = {
    "background_f1": 0.912,
    "object_f1": 0.753,
    "dynamic_f1": 0.841,
    "change_f1": 0.646,
}


def _load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_source_manifest_distinguishes_release_latest_and_used_artifact() -> None:
    payload = _load_manifest()

    assert payload["status"] in ALLOWED_STATUSES
    assert payload["rss_release"]["commit"] == RSS_RELEASE
    assert payload["rss_release"]["khronos_eval_file_count"] == 0
    assert payload["rss_release"]["evaluator_identity"] == "NOT_PRESENT_IN_RELEASE"
    assert payload["latest_public"]["commit"] == LATEST_PUBLIC
    assert payload["latest_public"]["khronos_eval_file_count"] > 0
    assert payload["artifact_evaluator_identity"]["status"] in {
        "PROVEN",
        "UNPROVEN_ARTIFACT_EVALUATOR_IDENTITY",
    }


def test_paper_protocol_and_sanity_gate_are_frozen() -> None:
    payload = _load_manifest()
    protocol = payload["paper_protocol"]

    assert protocol["scene"] == "Apartment"
    assert protocol["pose"] == "GROUND_TRUTH"
    assert protocol["semantics"] == "SIMULATOR_GROUND_TRUTH"
    assert protocol["voxel_resolution_m"] == 0.08
    assert protocol["sensing_range_m"] == 5.0
    assert protocol["reference_metrics"] == REFERENCE_METRICS
    assert protocol["absolute_f1_tolerance"] == 0.02
    assert protocol["tolerance_frozen_before_reproduction"] is True


def test_blocked_status_cannot_claim_a_local_paper_reproduction() -> None:
    payload = _load_manifest()
    reproduction = payload["reproduction"]

    assert reproduction["status"] == payload["status"]
    if payload["status"] == "BLOCKED_PAPER_ASSET":
        assert reproduction["blocking_evidence"]
        assert reproduction["executed"] is False
        assert "observed_metrics" not in reproduction
        assert reproduction["comparison_result"] == "NOT_RUN"


def test_source_and_artifact_bindings_are_content_addressed() -> None:
    payload = _load_manifest()

    for identity_name in ("rss_release", "latest_public"):
        identity = payload[identity_name]
        assert re.fullmatch(r"[0-9a-f]{40}", identity["commit"])
        for binding in identity["file_bindings"]:
            assert binding["path"].strip()
            assert re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])
            assert binding["byte_count"] > 0

    artifact = payload["artifact_evaluator_identity"]
    for sha256 in artifact["bound_evidence"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", sha256)


def test_report_states_protocol_identity_boundary_and_all_reference_metrics() -> None:
    payload = _load_manifest()
    report = REPORT.read_text(encoding="utf-8")

    assert payload["status"] in report
    assert RSS_RELEASE in report
    assert LATEST_PUBLIC in report
    assert "0.02" in report
    for metric_name, metric_value in REFERENCE_METRICS.items():
        assert metric_name in report
        assert f"{metric_value:.3f}" in report
    assert "PAPER_PROTOCOL_EXACT" in report
    assert "post-release" in report


def test_public_artifacts_do_not_publish_machine_local_paths() -> None:
    for path in (MANIFEST, REPORT):
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "file://" not in text
