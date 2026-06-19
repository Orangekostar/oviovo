"""CropFormer proposal backend wrapper with configurable local integration."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List
import sys

import numpy as np

from src.core.data_structures import Frame, Proposal2D
from src.models.proposal_backend import ProposalBackend


class CropFormerProposalBackend(ProposalBackend):
    """Detectron-style CropFormer wrapper that normalizes outputs into Proposal2D.

    The current machine does not appear to have a ready-to-use CropFormer
    installation. This backend therefore focuses on:

    - clean config/path plumbing
    - support for detectron-style predictors when available
    - support for a custom factory hook for local adapters
    - explicit, actionable errors when assets are missing
    """

    def __init__(self) -> None:
        self.device = "cuda"
        self.max_proposals = 50
        self.confidence_threshold = 0.0
        self.input_format = "BGR"
        self.predictor: Any | None = None
        self.config_path: str | None = None
        self.checkpoint_path: str | None = None
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: dict) -> None:
        self.device = str(config.get("device", "cuda"))
        self.max_proposals = int(config.get("max_proposals", 50))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
        self.input_format = str(config.get("input_format", "BGR")).upper()

        extra_python_paths = config.get("python_paths", [])
        if isinstance(extra_python_paths, (str, Path)):
            extra_python_paths = [extra_python_paths]
        for extra_path in extra_python_paths:
            path = str(Path(extra_path).expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)

        self.config_path = self._resolve_optional_path(config.get("config_path"))
        self.checkpoint_path = self._resolve_optional_path(config.get("checkpoint_path"))

        factory_spec = config.get("factory")
        if factory_spec:
            self.predictor = self._build_from_factory(factory_spec, config)
            self.last_generation_info = {
                "backend": "cropformer",
                "loader": "factory",
                "factory": factory_spec,
                "device": self.device,
                "config_path": self.config_path,
                "checkpoint_path": self.checkpoint_path,
            }
            return

        self.predictor = self._build_detectron_predictor(config)
        self.last_generation_info = {
            "backend": "cropformer",
            "loader": "detectron2_default_predictor",
            "device": self.device,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "config_hooks": list(config.get("config_hooks", [])),
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        if self.predictor is None:
            raise RuntimeError("CropFormerProposalBackend.initialize() must be called before inference.")

        image = rgb
        if self.input_format == "BGR":
            image = rgb[..., ::-1]

        outputs = self.predictor(image)
        proposals = self._normalize_outputs(outputs)
        proposals.sort(key=lambda proposal: (proposal.confidence, proposal.area), reverse=True)
        proposals = [proposal for proposal in proposals if proposal.confidence >= self.confidence_threshold]
        proposals = proposals[: self.max_proposals]

        for proposal_id, proposal in enumerate(proposals):
            proposal.proposal_id = proposal_id

        self.last_generation_info = {
            **self.last_generation_info,
            "kept_mask_count": len(proposals),
        }
        return proposals

    def _build_from_factory(self, factory_spec: str, config: dict) -> Any:
        factory = self._import_symbol(factory_spec)
        predictor = factory(config)
        if not callable(predictor):
            raise TypeError(
                f"CropFormer factory '{factory_spec}' returned a non-callable predictor: {type(predictor)!r}"
            )
        return predictor

    def _build_detectron_predictor(self, config: dict) -> Any:
        if self.config_path is None:
            raise RuntimeError(
                "CropFormer config_path is required. No local CropFormer assets were auto-detected."
            )
        if self.checkpoint_path is None:
            raise RuntimeError(
                "CropFormer checkpoint_path is required. No local CropFormer checkpoint was auto-detected."
            )

        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CropFormer backend requires detectron2 or a custom factory. "
                "Current machine has no detectable CropFormer/Detectron2 installation."
            ) from exc

        cfg = get_cfg()
        for hook_spec in config.get("config_hooks", []):
            hook = self._import_symbol(hook_spec)
            hook(cfg)

        cfg.merge_from_file(self.config_path)
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.WEIGHTS = self.checkpoint_path
        score_threshold = float(config.get("score_threshold", self.confidence_threshold))
        if hasattr(cfg.MODEL, "ROI_HEADS"):
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        if hasattr(cfg.MODEL, "RETINANET"):
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold
        if hasattr(cfg.TEST, "DETECTIONS_PER_IMAGE") and config.get("max_detections"):
            cfg.TEST.DETECTIONS_PER_IMAGE = int(config["max_detections"])
        return DefaultPredictor(cfg)

    def _normalize_outputs(self, outputs: Any) -> List[Proposal2D]:
        if isinstance(outputs, dict) and "instances" in outputs:
            return self._normalize_detectron_instances(outputs["instances"])
        if isinstance(outputs, dict) and "proposals" in outputs:
            return self._normalize_iterable(outputs["proposals"])
        if isinstance(outputs, (list, tuple)):
            return self._normalize_iterable(outputs)
        raise RuntimeError(
            "Unsupported CropFormer output format. Expected detectron2 instances or iterable proposals."
        )

    def _normalize_detectron_instances(self, instances: Any) -> List[Proposal2D]:
        if hasattr(instances, "to"):
            instances = instances.to("cpu")
        if not hasattr(instances, "pred_masks"):
            raise RuntimeError("CropFormer detectron output has no pred_masks field.")

        masks = instances.pred_masks.numpy()
        boxes = None
        if hasattr(instances, "pred_boxes"):
            boxes = instances.pred_boxes.tensor.numpy()
        scores = None
        if hasattr(instances, "scores"):
            scores = instances.scores.numpy()
        classes = None
        if hasattr(instances, "pred_classes"):
            classes = instances.pred_classes.numpy()

        proposals: List[Proposal2D] = []
        for index, mask in enumerate(masks):
            segmentation = np.asarray(mask, dtype=bool)
            area = int(segmentation.sum())
            if area <= 0:
                continue
            bbox_xyxy = self._mask_to_bbox_xyxy(segmentation)
            if boxes is not None and index < len(boxes):
                bbox_xyxy = boxes[index].astype(np.float32)
            confidence = float(scores[index]) if scores is not None and index < len(scores) else 1.0
            metadata = {"backend": "cropformer", "raw_mask_index": index}
            if classes is not None and index < len(classes):
                metadata["pred_class"] = int(classes[index])
            proposals.append(
                Proposal2D(
                    proposal_id=index,
                    mask=segmentation,
                    bbox_xyxy=bbox_xyxy,
                    area=area,
                    confidence=confidence,
                    backend_name="cropformer",
                    metadata=metadata,
                )
            )
        return proposals

    def _normalize_iterable(self, items: Iterable[Any]) -> List[Proposal2D]:
        proposals: List[Proposal2D] = []
        for index, item in enumerate(items):
            if isinstance(item, dict):
                segmentation = item["segmentation"] if "segmentation" in item else item.get("mask")
                bbox = item["bbox_xyxy"] if "bbox_xyxy" in item else item.get("bbox")
                score = item.get("score", item.get("confidence", 1.0))
                metadata = {
                    key: value
                    for key, value in item.items()
                    if key not in {"segmentation", "mask", "bbox_xyxy", "bbox", "score", "confidence"}
                }
            else:
                raise RuntimeError(
                    "Unsupported iterable CropFormer output item. Expected dict-like proposal entries."
                )

            if segmentation is None:
                raise RuntimeError("CropFormer iterable output item has no segmentation/mask field.")

            mask = np.asarray(segmentation, dtype=bool)
            area = int(mask.sum())
            if area <= 0:
                continue
            bbox_xyxy = self._mask_to_bbox_xyxy(mask)
            if bbox is not None:
                bbox_xyxy = np.asarray(bbox, dtype=np.float32)
                if bbox_xyxy.shape[0] != 4:
                    raise RuntimeError("CropFormer bbox must have four coordinates.")

            proposals.append(
                Proposal2D(
                    proposal_id=index,
                    mask=mask,
                    bbox_xyxy=bbox_xyxy,
                    area=area,
                    confidence=float(score),
                    backend_name="cropformer",
                    metadata={"backend": "cropformer", **metadata},
                )
            )
        return proposals

    @staticmethod
    def _mask_to_bbox_xyxy(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            dtype=np.float32,
        )

    @staticmethod
    def _import_symbol(spec: str) -> Callable[..., Any]:
        module_name, symbol_name = spec.split(":", 1)
        module = import_module(module_name)
        return getattr(module, symbol_name)

    @staticmethod
    def _resolve_optional_path(value: Any) -> str | None:
        if value in (None, "", "null"):
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / path
        return str(path.resolve())
