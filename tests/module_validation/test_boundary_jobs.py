"""A historical receipt alone must never stand in for verified execution."""

import json

import pytest

from src.static_ovmap.module_validation.assets import sha256_file
from src.static_ovmap.module_validation.contracts import canonical_digest


def test_capture_reader_rejects_changed_payload_and_incomplete_schedule(tmp_path):
    from src.static_ovmap.module_validation.boundary_jobs import verify_capture

    surface = tmp_path / "surface.npz"
    surface.write_bytes(b"payload")
    manifest = {
        "artifact_type": "OVIMAP_NATIVE_CAPTURE",
        "scheduled_frame_ids": [0],
        "completed_frame_ids": [0],
        "frames": [{"frame_id": 0}],
        "surface": {"path": surface.name, "sha256": sha256_file(surface)},
    }
    manifest["identity"] = canonical_digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify_capture(path)["identity"] == manifest["identity"]
    surface.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_capture(path)
    surface.write_bytes(b"payload")
    manifest.pop("identity")
    manifest["scheduled_frame_ids"] = [0, 10]
    manifest["identity"] = canonical_digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        verify_capture(path)
