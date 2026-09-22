"""Confirmation/refinement executes only preselected semantic models and settings."""

import json

import numpy as np

from src.static_ovmap.module_validation import frozen_semantic as frozen
from src.static_ovmap.module_validation.boundary_jobs import file_identity
from src.static_ovmap.module_validation.contracts import (
    atomic_write_json,
    canonical_digest,
)
from src.static_ovmap.module_validation.evaluation import (
    GeometryIdentity,
    PredictionPayload,
)
from src.static_ovmap.module_validation.scannet_study import save_prediction
from tests.module_validation.test_semantic_models import make_inputs


def test_frozen_raw_teacher_uses_only_its_model_and_preserves_masks_ranks(tmp_path, monkeypatch):
    manifest_path, config = make_inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["views"] = {"owner:7": list(manifest["requests"])}
    manifest.pop("identity")
    manifest["identity"] = canonical_digest(manifest)
    atomic_write_json(manifest_path, manifest)
    text = tmp_path / "text.npz"
    np.savez(text, text_embeddings=np.eye(2), valid_ids=[1, 2], class_names=["chair", "table"])
    config.update(siglip2_model=config["native_model"], siglip2_text_cache=str(text))
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    config["runtime_config"] = str(config_path)
    native = PredictionPayload("N0", "N0", "sceneA", GeometryIdentity("a" * 64, "b" * 64, "c" * 64, "projection", 3),
        np.array([7, 7, 0]), np.array([1, 1, 0]), ((7, .7),), {"attempts": 5, "added_attempts": 2, "added_crop_inputs": 12})
    native.lock()
    native_path = save_prediction(native, tmp_path / "native")
    calibration = tmp_path / "calibration_receipt.json"
    calibration.write_text(json.dumps({"status": "COMPLETE", "input_identity": "cal",
        "outputs": [file_identity(config_path)], "teacher_id": "S_SIGLIP2_AREA", "heads": {}}))
    calls = []

    class Encoder:
        def encode_images(self, images):
            return np.tile([0., 1.], (len(images), 1))

    original_run = frozen.subprocess.run

    def run(command, **kwargs):
        if "--model" not in command:
            return original_run(command, **kwargs)
        from src.static_ovmap.module_validation import semantic_models as models
        model = command[command.index("--model") + 1]
        calls.append(model)
        monkeypatch.setattr(models.FrozenSiglipBackend, "from_local", lambda *args, **kwargs: Encoder())
        monkeypatch.setattr(models, "_mask_support", lambda *args: [4, 4, 4])
        models.encode_semantic_requests(manifest_path, model, config, command[-1], device="cpu")

    monkeypatch.setattr(frozen.subprocess, "run", run)
    data = {"native": native, "native_manifest_path": native_path, "output": tmp_path, "capture_path": manifest_path}
    runtime = {"semantic_python": "unused", "native_perception_python": "unused"}
    payload, result = frozen.prepare_frozen_semantic(data, "S_SIGLIP2_AREA", runtime, config, config_path,
        calibration, tmp_path / "evidence", tmp_path / "prediction", manifest_path=manifest_path)
    assert calls == ["siglip2"]
    np.testing.assert_array_equal(payload.owner_ids, native.owner_ids)
    np.testing.assert_array_equal(payload.semantic_labels, [2, 2, 0])
    assert payload.instance_ranks == native.instance_ranks
    assert result["effective_changes"] == 1
    assert result["logical_cost"]["added_attempts"] == 3
    assert result["logical_cost"]["added_crop_inputs"] == 18
