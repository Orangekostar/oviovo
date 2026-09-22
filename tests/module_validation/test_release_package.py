"""Publication includes evidence and small heads without copying raw tensors."""

import gzip
import json
from types import SimpleNamespace

import pytest

from scripts.evaluation.run_ovimap_module_study import export_attempt
from src.static_ovmap.module_validation.boundary_jobs import file_identity


def setup_release(tmp_path, files):
    study, runtime, attempt = (tmp_path / name for name in ("study", "runtime", "attempt"))
    attempt.mkdir()
    identities = []
    for relative, contents in files.items():
        path = study / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents if isinstance(contents, bytes) else json.dumps(contents).encode())
        identities.append(file_identity(path))
    (attempt / "external_artifacts.json").write_text(json.dumps({"entries": identities}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"study_root": str(study)}))
    return SimpleNamespace(attempt_dir=attempt, cache_key="frozen-key", resolved_config={
        "scannet_study_config": str(config), "scannet_runtime": {"output_root": str(runtime)}})


def test_export_preserves_head_and_diagnostics_but_excludes_logits_and_surfaces(tmp_path):
    runner = setup_release(tmp_path, {
        "semantic/heads/S_SIMPLE/checkpoint.pt": b"safe-state-dict-fixture",
        "semantic/heads/S_SIMPLE/calibration.json": {"threshold": .02, "status": "COMPLETE"},
        "scenes/s1/semantic_models/wow/requests/r1.json": {
            "request_id": "r1", "inference_status": "COMPLETE", "raw_generation": "chair",
            "original_mask_support": 99, "final_mask_support": 3,
            "mapping": {"label_id": 2, "method": "EXACT", "similarities": [.4, .8], "top1_top2_gap": .4}},
        "scenes/s1/geometry/pool/hypotheses.json": {
            "groups": [{"group_id": "g1", "owner_ids": [1], "leaf_ids": [11, 12]}],
            "hypotheses": {"g1": [{"hypothesis_id": "h1", "leaf_ids": [11, 12],
                "components": [0, 0], "kind": "ORIGINAL", "owner_ids": [1]}]},
            "agreements": {"h1": {"score": .8}}, "GT_input": False},
        "scenes/s1/baseline/N0/prediction.npz": b"large-external-surface",
        "scenes/s1/semantic/direct_suggestions.json": {"all_logits": [1, 2]},
    })
    destination = tmp_path / "export"
    export_attempt(runner, destination)
    assert (destination / "study/semantic/heads/S_SIMPLE/checkpoint.pt").read_bytes() == b"safe-state-dict-fixture"
    assert json.loads((destination / "study/semantic/heads/S_SIMPLE/calibration.json").read_text())["threshold"] == .02
    ledger = json.loads((destination / "study/scenes/s1/semantic_models/wow/requests/r1.json").read_text())
    assert ledger["raw_generation"] == "chair"
    assert ledger["final_mask_support"] == 3
    assert ledger["mapping"] == {"label_id": 2, "method": "EXACT", "top1_top2_gap": .4}
    pool = json.loads((destination / "study/scenes/s1/geometry/pool/hypotheses_summary.json").read_text())
    assert pool["groups"][0]["leaf_count"] == 2
    assert pool["hypotheses"]["g1"][0]["component_count"] == 1
    assert "components" not in pool["hypotheses"]["g1"][0]
    assert not list(destination.rglob("*.npz"))
    assert not list(destination.rglob("direct_suggestions.json"))
    manifest = json.loads((destination / "export_manifest.json").read_text())
    assert manifest["total_bytes"] == sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    assert manifest["total_bytes"] <= 100 * 1024 * 1024


def test_export_rejects_changed_evidence_before_creating_package(tmp_path):
    runner = setup_release(tmp_path, {"query/checkpoint.pt": b"original"})
    (tmp_path / "study/query/checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        export_attempt(runner, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_rejects_oversized_head_before_creating_package(tmp_path):
    runner = setup_release(tmp_path, {"geometry/checkpoint.pt": b"x" * (10 * 1024 * 1024 + 1)})
    with pytest.raises(ValueError, match="10 MiB"):
        export_attempt(runner, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_total_limit_includes_ledgers_and_manifest_before_writing(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import release_package

    runner = setup_release(tmp_path, {"semantic/training.json": {"detail": "x" * 4096}})
    monkeypatch.setattr(release_package, "MAX_RELEASE_BYTES", 4096)
    with pytest.raises(ValueError, match="release exceeds"):
        export_attempt(runner, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_large_diagnostic_json_is_losslessly_compressed_with_stable_hash(tmp_path):
    content = {"frame_barrier": True, "frames": [{"paid": True, "request": "r" * 100}] * 3000}
    runner = setup_release(tmp_path, {"scenes/s1/query/trace/decisions.json": content})
    export_attempt(runner, tmp_path / "export-one")
    export_attempt(runner, tmp_path / "export-two")
    relative = "study/scenes/s1/query/trace/decisions.json.gz"
    one, two = (tmp_path / name / relative for name in ("export-one", "export-two"))
    assert json.loads(gzip.decompress(one.read_bytes())) == content
    assert one.read_bytes() == two.read_bytes()
    assert one.stat().st_size < 5000
    manifest = json.loads((tmp_path / "export-one/export_manifest.json").read_text())
    assert manifest["source_files"][relative]["encoding"] == "gzip"
    assert manifest["source_files"][relative]["source"]["path"].endswith("decisions.json")
