import json

import numpy as np
import pytest
from PIL import Image

from scripts.evaluation.complete_crove_frontend_gpu import validated_frame


def test_incomplete_frame_is_not_reused_and_model_binding_is_enforced(tmp_path):
    root = tmp_path / "run"
    scene = tmp_path / "room1"
    (root / "frontend/frame_diagnostics").mkdir(parents=True)
    mask = root / "frontend/frame000000.png"
    Image.fromarray(np.array([[0, 1], [1, 0]], np.uint8)).save(mask)
    assert validated_frame(root, scene, 0, "weights", "config") is None
    diagnostic = root / "frontend/frame_diagnostics/frame000000.json"
    record = {
        "input": str(scene / "results/frame000000.jpg"),
        "weights_sha256": "weights",
        "config_sha256": "config",
        "confidence_threshold": 0.5,
        "shape": [2, 2],
        "dtype": "uint8",
        "instance_ids": [1],
    }
    diagnostic.write_text(json.dumps(record))
    assert len(validated_frame(root, scene, 0, "weights", "config")) == 2
    with pytest.raises(ValueError, match="binding"):
        validated_frame(root, scene, 0, "different weights", "config")
    Image.fromarray(np.array([[0, 2], [2, 0]], np.uint8)).save(mask)
    with pytest.raises(ValueError, match="IDs"):
        validated_frame(root, scene, 0, "weights", "config")
