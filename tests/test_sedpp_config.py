"""Config and vocabulary checks for the SED++ Replica frontend."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_replica_sedpp_online_config_disables_yoloworld_sam_fusion() -> None:
    config_path = Path("configs/replica_sedpp_online_4090.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "sedpp"
    assert config["proposal"]["sedpp"] == {
        "device": "cuda",
        "repo_root": "/home/ww/vv/paper2/SED",
        "config_path": "/home/ww/vv/paper2/SED/configs/convnextL_768.yaml",
        "checkpoint_path": "/home/ww/vv/paper2/SED/weights/sed_model_large.pth",
        "openclip_weight_path": (
            "/home/ww/vv/paper2/SED/weights/"
            "CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin"
        ),
        "hf_endpoint": "https://hf-mirror.com",
        "class_json": "data/input/sedpp_replica_classes.json",
        "fast_inference": True,
        "topk": 8,
        "sem_seg_output": "logits",
        "min_pixel_confidence": 0.35,
        "strong_confidence_threshold": 0.50,
        "structural_confidence_threshold": 0.60,
        "structure_classes": ["wall", "floor", "ceiling", "door", "window", "blinds"],
        "min_component_area": 100,
        "max_proposals": 80,
        "depth_split_enabled": True,
    }
    assert config["anchor_frontend"]["enabled"] is False
    assert config["anchor_guided_sam"]["enabled"] is False
    assert config["pipeline"]["yoloworld_sam_parallel_frontend_enabled"] is False
    assert config["pipeline"]["proposal_prefetch_enabled"] is False
    assert config["proposal"]["precomputed"]["cache_dir"] == ""
    assert config["proposal"]["precomputed"]["manifest_path"] == ""


def test_replica_yoloworld_anchor_first_sam_config_uses_anchor_prompted_sam2() -> None:
    config_path = Path("configs/replica_yoloworld_anchor_first_sam_4090.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["proposal"]["backend"] == "sam2"
    assert config["anchor_frontend"]["enabled"] is True
    assert config["anchor_frontend"]["backend"] == "yoloworld"
    assert config["anchor_guided_sam"]["enabled"] is False
    assert config["pipeline"]["yoloworld_sam_parallel_frontend_enabled"] is True
    assert config["pipeline"]["yoloworld_sam_mode"] == "anchor_first_sam"
    assert config["pipeline"]["proposal_prefetch_enabled"] is False

    sam2_config = config["proposal"]["sam2"]
    assert sam2_config["anchor_prompt"] == {
        "enabled": True,
        "box_padding_px": 0,
        "multimask_output": False,
    }
    assert config["proposal"]["precomputed"]["cache_dir"] == ""
    assert config["proposal"]["precomputed"]["manifest_path"] == ""


def test_sedpp_replica_class_table_has_key_canonical_labels_and_aliases() -> None:
    class_path = Path("data/input/sedpp_replica_classes.json")
    payload = json.loads(class_path.read_text(encoding="utf-8"))
    classes = payload["classes"]
    aliases = payload["aliases"]

    required_classes = {
        "wall",
        "floor",
        "ceiling",
        "door",
        "window",
        "blinds",
        "cabinet",
        "chair",
        "table",
        "sofa",
        "rug",
        "bed",
        "basket",
        "blanket",
        "book",
        "bottle",
        "bowl",
        "cushion",
        "lamp",
        "picture",
        "indoor-plant",
        "plant-stand",
        "pillow",
        "shelf",
        "stool",
        "vase",
    }
    assert required_classes <= set(classes)
    assert len(classes) == len(set(classes))

    expected_aliases = {
        "couch": "sofa",
        "potted plant": "indoor-plant",
        "plant": "indoor-plant",
        "windowpane": "window",
        "window-blind": "blinds",
        "floor-wood": "floor",
        "floor-tile": "floor",
        "wall-other": "wall",
        "dining table": "table",
        "coffee table": "table",
    }
    for alias, canonical in expected_aliases.items():
        assert aliases[alias] == canonical
        assert canonical in classes

    assert not (set(aliases) & set(classes))
