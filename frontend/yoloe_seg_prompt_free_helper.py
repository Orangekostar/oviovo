#!/usr/bin/env python3
"""Helper process for YOLOE-Seg prompt-free anchor inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--classes-json", type=str, default="[]")
    parser.add_argument("--vocab-source-checkpoint-path", type=Path)
    parser.add_argument("--vocab-cache-path", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--max-detections", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def resolve_vocab_source_checkpoint_path(args: argparse.Namespace) -> Path | None:
    if args.vocab_source_checkpoint_path:
        return args.vocab_source_checkpoint_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    if checkpoint_path.stem.endswith("-pf"):
        candidate = checkpoint_path.with_name(f"{checkpoint_path.stem[:-3]}{checkpoint_path.suffix}")
        if candidate.exists():
            return candidate
    return None


def resolve_vocab_cache_path(args: argparse.Namespace, repo_root: Path, classes: list[str]) -> Path:
    if args.vocab_cache_path:
        return args.vocab_cache_path.resolve()
    source_checkpoint_path = resolve_vocab_source_checkpoint_path(args)
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else "",
                "classes": classes,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return repo_root / ".cache" / "oviovo_yoloe_vocab" / f"{digest}.pt"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    if str(args.device).strip().lower().startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    third_party_clip = repo_root / "third_party" / "CLIP"
    if third_party_clip.exists() and str(third_party_clip) not in sys.path:
        sys.path.insert(0, str(third_party_clip))
    third_party_mobileclip = repo_root / "third_party" / "ml-mobileclip"
    if third_party_mobileclip.exists() and str(third_party_mobileclip) not in sys.path:
        sys.path.insert(0, str(third_party_mobileclip))

    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.torch_utils import select_device
    from ultralytics.utils import DEFAULT_CFG

    class NoFuseSegmentationPredictor(SegmentationPredictor):
        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        def setup_model(self, model, verbose=True):
            self.model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, verbose=verbose),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
                batch=self.args.batch,
                fuse=False,
                verbose=verbose,
            )
            self.device = self.model.device
            self.args.half = self.model.fp16
            self.model.eval()

    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image_path}")

    classes = [str(item).strip() for item in json.loads(args.classes_json) if str(item).strip()]
    vocab = None
    if classes:
        vocab_source_checkpoint_path = resolve_vocab_source_checkpoint_path(args)
        if vocab_source_checkpoint_path is None or not vocab_source_checkpoint_path.exists():
            raise FileNotFoundError(
                "YOLOE vocabulary-aware prompt-free inference requires a non-PF source checkpoint."
            )
        vocab_cache_path = resolve_vocab_cache_path(args, repo_root, classes)
        if vocab_cache_path.exists():
            vocab = torch.load(vocab_cache_path, map_location=str(args.device))
        else:
            unfused_model = YOLOE(str(vocab_source_checkpoint_path))
            unfused_model.to(str(args.device))
            vocab = unfused_model.get_vocab(classes)
            vocab_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vocab, vocab_cache_path)
            del unfused_model
    model = YOLOE(str(args.checkpoint_path))
    model.to(str(args.device))
    if vocab is not None:
        model.set_vocab(vocab, names=classes)
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = float(args.confidence_threshold)
        model.model.model[-1].max_det = int(args.max_detections)
    results = model.predict(
        image,
        verbose=False,
        conf=float(args.confidence_threshold),
        max_det=int(args.max_detections),
        device=str(args.device),
        predictor=NoFuseSegmentationPredictor,
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
                        "metadata": {
                            "source": "yoloe_seg_pf_helper",
                            "vocab_applied": bool(classes),
                        },
                    }
                )

    args.output_json.write_text(json.dumps({"anchors": anchors}), encoding="utf-8")


if __name__ == "__main__":
    main()
