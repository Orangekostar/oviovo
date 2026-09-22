"""Prevent holdout leakage and stale native executions on real ScanNet inputs."""

import json

import pytest

from src.static_ovmap.module_validation import boundary_jobs
from src.static_ovmap.module_validation.assets import sha256_file
from src.static_ovmap.module_validation.contracts import canonical_digest


def test_capture_missing_slots_are_explicit_not_backfilled(tmp_path):
    surface = tmp_path / "surface.npz"
    surface.write_bytes(b"surface")
    manifest = {
        "artifact_type": "OVIMAP_NATIVE_CAPTURE",
        "scheduled_frame_ids": [0, 10, 20],
        "completed_frame_ids": [0, 20],
        "frames": [{"frame_id": 0}, {"frame_id": 20}],
        "surface": {"path": surface.name, "sha256": sha256_file(surface)},
    }
    manifest["identity"] = canonical_digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert boundary_jobs.verify_capture(path, allow_skipped=True)["completed_frame_ids"] == [0, 20]
    manifest.pop("identity")
    manifest["completed_frame_ids"] = [0, 30]
    manifest["frames"][1]["frame_id"] = 30
    manifest["identity"] = canonical_digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schedule"):
        boundary_jobs.verify_capture(path, allow_skipped=True)


def test_only_locked_development_scenes_can_start_capture():
    from src.static_ovmap.module_validation.scannet_runtime import development_rows

    locked = {"selected": [
        {"scene_id": "scene0001_00", "family_id": "scene0001", "role": "fit"},
        {"scene_id": "scene0002_00", "family_id": "scene0002", "role": "confirm"},
    ]}
    assert [r["scene_id"] for r in development_rows(locked)] == ["scene0001_00"]
    for scene in ("scene0002_00", "scene0003_00"):
        with pytest.raises(ValueError, match="development"):
            development_rows(locked, [scene])


def test_execution_cache_rejects_changed_output(tmp_path):
    from src.static_ovmap.module_validation.scannet_runtime import reusable_job

    result = tmp_path / "mesh.ply"
    result.write_bytes(b"mesh")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "COMPLETE", "input_identity": "input",
        "outputs": [{"path": str(result), "sha256": sha256_file(result)}]}))
    assert reusable_job(receipt, "input")
    assert not reusable_job(receipt, "different")
    result.write_bytes(b"corrupt")
    assert not reusable_job(receipt, "input")


def test_busy_gpu_fails_before_starting_any_visual_job(tmp_path):
    from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu

    with pytest.raises(RuntimeError, match="GPU_BUSY"):
        require_idle_gpu("2", tmp_path, sample="35375, 100")
    assert json.loads((tmp_path / "resource_status.json").read_text())["status"] == "BLOCKED_GPU_BUSY"
    require_idle_gpu("2", tmp_path, sample="8, 0")


def test_gpu_rechecks_transient_teardown_without_accepting_a_busy_sample(tmp_path, monkeypatch):
    from src.static_ovmap.module_validation import scannet_runtime as runtime

    samples = iter(["1489, 55", "0, 0"])
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    runtime.require_idle_gpu("0", tmp_path)
    assert json.loads((tmp_path / "resource_status.json").read_text())["status"] == "AVAILABLE"


def test_interrupted_native_replay_preserves_partial_before_retry(tmp_path):
    from src.static_ovmap.module_validation.scannet_runtime import (
        preserve_interrupted_replay,
    )

    (tmp_path / "capture/scene0001_00").mkdir(parents=True)
    (tmp_path / "capture/scene0001_00/frame.npz").write_bytes(b"captured")
    (tmp_path / "mapping_job").mkdir()
    (tmp_path / "mapping_job/running.json").write_text(json.dumps({"input_identity": "same"}))
    with pytest.raises(ValueError, match="identity"):
        preserve_interrupted_replay(tmp_path, "changed")
    assert (tmp_path / "capture/scene0001_00/frame.npz").exists()
    archive = preserve_interrupted_replay(tmp_path, "same")
    assert (archive / "capture/scene0001_00/frame.npz").read_bytes() == b"captured"
    assert not (tmp_path / "capture").exists()
    assert preserve_interrupted_replay(tmp_path, "same") is None
