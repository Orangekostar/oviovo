#!/usr/bin/env python3
"""Helper process for YOLO-World anchor inference under an external Python env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--classes-json", type=str, default="[]")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--max-detections", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if str(args.device).strip().lower().startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from ultralytics import YOLO

    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image_path}")

    model = YOLO(str(args.model_path))
    try:
        model.to(str(args.device))
    except Exception:
        pass
    classes = [str(item).strip() for item in json.loads(args.classes_json) if str(item).strip()]
    if classes:
        try:
            model.set_classes(classes)
        except Exception:
            pass

    results = model.predict(
        source=image,
        conf=float(args.confidence_threshold),
        verbose=False,
        max_det=int(args.max_detections),
        device=str(args.device),
    )

    anchors = []
    if results:
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        if boxes is not None:
            for anchor_id, (xyxy, conf, cls_idx) in enumerate(
                zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
            ):
                class_name = names.get(int(cls_idx), str(int(cls_idx))) if isinstance(names, dict) else str(int(cls_idx))
                anchors.append(
                    {
                        "anchor_id": int(anchor_id),
                        "bbox_xyxy": [float(v) for v in xyxy.tolist()],
                        "class_name": str(class_name),
                        "confidence": float(conf),
                        "metadata": {"source": "yoloworld_helper"},
                    }
                )

    args.output_json.write_text(json.dumps({"anchors": anchors}), encoding="utf-8")


if __name__ == "__main__":
    main()
