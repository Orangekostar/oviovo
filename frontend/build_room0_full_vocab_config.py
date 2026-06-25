#!/usr/bin/env python3
"""Build a room0 full-vocabulary config for anchor-based fast validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


INVALID_CLASS_IDS = {-2, -1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("/home/phl/vv/oviovo/configs/midrecall_local_memory_boost.yaml"),
    )
    parser.add_argument(
        "--gt-labels",
        type=Path,
        default=Path("/home/phl/vv/oviovo/data/input/replica_semantic_gt/room0.txt"),
    )
    parser.add_argument(
        "--gt-info-json",
        type=Path,
        default=Path("/home/phl/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json"),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--yoloworld-model-path",
        type=Path,
        default=Path("/home/phl/vv/DualMapV1/model/yolov8l-world.pt"),
    )
    parser.add_argument(
        "--yoloworld-python",
        type=Path,
        default=Path("/home/phl/anaconda3/envs/dualmap/bin/python"),
    )
    parser.add_argument(
        "--yoloe-python",
        type=Path,
        default=Path("/home/phl/anaconda3/envs/oviovo/bin/python"),
    )
    parser.add_argument(
        "--yoloe-repo-root",
        type=Path,
        default=Path("/home/phl/vv/yoloe_repo_probe"),
    )
    parser.add_argument(
        "--yoloe-checkpoint-path",
        type=Path,
        default=Path("/home/phl/vv/yoloe_repo_probe/pretrain/yoloe-v8s-seg-pf.pt"),
    )
    parser.add_argument(
        "--yoloworld-confidence",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--yoloe-confidence",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--primary-device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--supplemental-device",
        type=str,
        default="cuda:0",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Base config must be a mapping: {path}")
    return payload


def read_room0_vocab(gt_labels_path: Path, gt_info_path: Path) -> list[str]:
    labels = [
        int(line.strip())
        for line in gt_labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    info = json.loads(gt_info_path.read_text(encoding="utf-8"))
    class_name_by_id = {int(item["id"]): str(item["name"]).strip() for item in info.get("classes", [])}

    vocab: list[str] = []
    for class_id in sorted(set(labels)):
        if class_id in INVALID_CLASS_IDS:
            continue
        class_name = class_name_by_id.get(class_id, "").strip()
        if class_name:
            vocab.append(class_name)
    return vocab


def main() -> None:
    args = parse_args()
    config = load_yaml(args.base_config)
    vocab = read_room0_vocab(args.gt_labels, args.gt_info_json)
    if not vocab:
        raise RuntimeError("Resolved empty room0 vocabulary from GT labels.")

    anchor_cfg = dict(config.get("anchor_frontend", {}))
    anchor_cfg["enabled"] = True
    anchor_cfg["backend"] = "yoloworld"
    anchor_cfg["device"] = args.primary_device
    anchor_cfg["python_executable"] = str(args.yoloworld_python)
    anchor_cfg["model_path"] = str(args.yoloworld_model_path)
    anchor_cfg["confidence_threshold"] = float(args.yoloworld_confidence)
    anchor_cfg["classes"] = vocab
    anchor_cfg["supplemental_enabled"] = True
    anchor_cfg["supplemental"] = {
        "backend": "yoloe_seg_pf",
        "device": args.supplemental_device,
        "python_executable": str(args.yoloe_python),
        "repo_root": str(args.yoloe_repo_root),
        "checkpoint_path": str(args.yoloe_checkpoint_path),
        "confidence_threshold": float(args.yoloe_confidence),
        "max_detections": int(anchor_cfg.get("max_detections", 128)),
    }
    config["anchor_frontend"] = anchor_cfg

    logging_cfg = dict(config.get("logging", {}))
    logging_cfg.setdefault("level", "INFO")
    config["logging"] = logging_cfg

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    summary = {
        "output_config": str(args.output_config),
        "class_count": len(vocab),
        "classes": vocab,
        "yoloworld_model_path": str(args.yoloworld_model_path),
        "yoloe_checkpoint_path": str(args.yoloe_checkpoint_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
