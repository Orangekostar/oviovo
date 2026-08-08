#!/usr/bin/env python3
"""JSONL worker for experimental SAM3 concept inference."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    return parser.parse_args()


def _load_sam3(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.sam3_repo_root is not None:
        sys.path.insert(0, str(args.sam3_repo_root.resolve()))
    try:
        importlib.import_module("sam3")
        builder_module = importlib.import_module("sam3.model_builder")
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
    except Exception as exc:
        raise RuntimeError(
            "SAM3 package is not importable; install official facebookresearch/sam3 "
            "in an isolated env first."
        ) from exc

    build_model = getattr(builder_module, "build_sam3_image_model", None)
    processor_cls = getattr(processor_module, "Sam3Processor", None)
    if build_model is None or processor_cls is None:
        raise RuntimeError(
            "SAM3 package does not expose official image API "
            "build_sam3_image_model/Sam3Processor."
        )

    checkpoint_path = str(args.checkpoint_path) if args.checkpoint_path is not None else None
    try:
        model = build_model(device=args.device, checkpoint_path=checkpoint_path)
        processor = processor_cls(model, device=args.device, confidence_threshold=0.0)
    except Exception as exc:
        checkpoint_note = (
            f" checkpoint_path={checkpoint_path!r}."
            if checkpoint_path is not None
            else " no checkpoint_path was provided, so official Hugging Face loading was used."
        )
        raise RuntimeError(
            "Failed to load official SAM3 image model;"
            f"{checkpoint_note} Verify Python>=3.12, PyTorch>=2.7, CUDA>=12.6, "
            "GPU memory, and Hugging Face access/authentication for facebook/sam3."
        ) from exc
    return model, processor


def _decode_image(request: dict[str, Any]) -> Image.Image:
    image = request.get("image")
    if not isinstance(image, dict):
        raise ValueError("Request image must be an object.")
    if image.get("dtype") != "uint8" or image.get("encoding") != "base64":
        raise ValueError("Request image must use dtype=uint8 and encoding=base64.")
    shape = image.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int for value in shape)
        or shape[2] != 3
        or shape[0] <= 0
        or shape[1] <= 0
    ):
        raise ValueError("Request image shape must be [H, W, 3].")
    try:
        data = base64.b64decode(str(image.get("data", "")), validate=True)
        array = np.frombuffer(data, dtype=np.uint8).reshape(tuple(shape))
    except Exception as exc:
        raise ValueError("Request image data is not valid base64 uint8 RGB data.") from exc
    return Image.fromarray(array, mode="RGB")


def _normalize_classes(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(label).strip() for label in value if str(label).strip()]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _iter_output_candidates(output: dict[str, Any], label: str) -> tuple[int, list[dict[str, Any]]]:
    masks = _to_numpy(output.get("masks", []))
    scores = _to_numpy(output.get("scores", []))
    boxes = _to_numpy(output.get("boxes", []))
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim < 3:
        return 0, []
    candidates: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        score = float(scores.reshape(-1)[index]) if scores.size > index else 1.0
        candidate: dict[str, Any] = {
            "label": label,
            "confidence": score,
            "mask": mask,
            "metadata": {"sam3_score": score, "prompt": label},
        }
        if boxes.size >= (index + 1) * 4:
            candidate["bbox_xyxy"] = [float(value) for value in boxes.reshape(-1, 4)[index].tolist()]
        candidates.append(candidate)
    return int(masks.shape[0]), candidates


def _materialize_proposal(candidate: dict[str, Any]) -> dict[str, Any]:
    proposal = dict(candidate)
    proposal["mask"] = np.asarray(candidate["mask"], dtype=bool).tolist()
    return proposal


def _coerce_nonnegative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_request(processor: Any, request: dict[str, Any]) -> dict[str, Any]:
    image = _decode_image(request)
    classes = _normalize_classes(request.get("classes", []))
    max_prompts_per_frame = _coerce_nonnegative_int(
        request.get("max_prompts_per_frame"),
        len(classes),
    )
    classes = classes[:max_prompts_per_frame]
    max_proposals = _coerce_nonnegative_int(request.get("max_proposals"), 64)
    confidence_threshold = _coerce_float(request.get("confidence_threshold"), 0.0)

    proposals: list[dict[str, Any]] = []
    raw_mask_count = 0
    state = processor.set_image(image)
    for label in classes:
        output = processor.set_text_prompt(state=state, prompt=label)
        if not isinstance(output, dict):
            raise RuntimeError("SAM3 processor returned a non-dict output.")
        prompt_mask_count, prompt_items = _iter_output_candidates(output, label)
        raw_mask_count += prompt_mask_count
        proposals.extend(
            proposal
            for proposal in prompt_items
            if float(proposal["confidence"]) >= confidence_threshold
        )
    proposals.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return {
        "raw_mask_count": raw_mask_count,
        "prompt_count": len(classes),
        "proposals": [_materialize_proposal(proposal) for proposal in proposals[:max_proposals]],
    }


def _error_response(request_id: int | None, error: str) -> dict[str, Any]:
    return {"id": request_id, "ok": False, "error": error}


def _success_response(request_id: int, result: dict[str, Any]) -> dict[str, Any]:
    return dict({"id": request_id, "ok": True}, **result)


def _read_requests() -> Any:
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(
                json.dumps({"id": None, "ok": False, "error": "Malformed JSON request."}),
                flush=True,
            )
            continue
        if not isinstance(request, dict):
            print(
                json.dumps({"id": None, "ok": False, "error": "Request must be a JSON object."}),
                flush=True,
            )
            continue
        yield request


def _request_id(request: dict[str, Any]) -> int | None:
    request_id = request.get("id")
    return request_id if type(request_id) is int else None


def main() -> int:
    args = parse_args()
    load_error: Exception | None = None
    processor: Any | None = None
    try:
        _, processor = _load_sam3(args)
    except RuntimeError as exc:
        load_error = exc

    for request in _read_requests():
        request_id = _request_id(request)
        if request_id is None:
            print(
                json.dumps(_error_response(None, "Request id must be an integer.")),
                flush=True,
            )
            continue
        if load_error is not None:
            response = _error_response(request_id, str(load_error))
        else:
            try:
                assert processor is not None
                response = _success_response(request_id, _run_request(processor, request))
            except Exception as exc:
                response = _error_response(request_id, f"SAM3 inference failed: {exc}")
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
