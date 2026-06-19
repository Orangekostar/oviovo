"""SAM2 proposal backend adapted from the OVO frontend setup."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List
import os
import sys
import time

import numpy as np

from src.core.data_structures import Anchor2D, Frame, Proposal2D
from src.models.proposal_backend import ProposalBackend

logger = logging.getLogger("oviovo.models.sam2_proposal_backend")


class SAM2ProposalBackend(ProposalBackend):
    """Class-agnostic proposal backend powered by SAM2 automatic mask generation."""

    def __init__(self) -> None:
        self.mask_generator: Any | None = None
        self.device = "cuda"
        self.max_proposals = 50
        self.confidence_threshold = 0.0
        self.model_cfg = ""
        self.checkpoint_path = ""
        self.last_generation_info: dict[str, Any] = {}
        self.last_generation_timings: dict[str, float] = {}
        self.sam2_repo_root: str | None = None
        self.cuda_postprocess_available = False
        self.image_predictor: Any | None = None
        self.anchor_prompt_enabled = False
        self.anchor_prompt_multimask_output = False
        self.anchor_prompt_max_masks_per_anchor = 1
        self.anchor_prompt_min_mask_area = 25
        self.anchor_prompt_min_anchor_box_coverage = 0.0
        self.anchor_prompt_min_depth_valid_coverage = 0.0
        self.anchor_prompt_box_padding_px = 0.0
        self.anchor_prompt_batch_boxes = True

    def initialize(self, config: Dict[str, Any]) -> None:
        torch = self._import_torch()

        self.device = config.get("device", "cuda")
        self.max_proposals = int(config.get("max_proposals", 50))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
        self._configure_anchor_prompts(config)
        sam2_repo_root = self._resolve_sam2_repo_root(config)
        self.sam2_repo_root = str(sam2_repo_root) if sam2_repo_root is not None else None

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM2 backend requested CUDA, but torch.cuda.is_available() is False.")

        sam_version = str(config.get("sam_version", "2.1"))
        sam_encoder = str(config.get("sam_encoder", "hiera_l"))
        checkpoint_root = self._resolve_checkpoint_root(config)
        model_cards = {
            "hiera_t": "hiera_tiny.pt",
            "hiera_s": "hiera_small.pt",
            "hiera_b+": "hiera_base_plus.pt",
            "hiera_l": "hiera_large.pt",
        }
        if sam_encoder not in model_cards:
            raise ValueError(f"Unsupported SAM2 encoder '{sam_encoder}'.")

        self.checkpoint_path = str(checkpoint_root / f"sam{sam_version}_{model_cards[sam_encoder]}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint_path}")

        build_sam2, SamAutomaticMaskGenerator, sam2_origin = self._import_sam2(sam2_repo_root)
        SAM2ImagePredictor = self._import_sam2_image_predictor()
        self.model_cfg = self._resolve_model_cfg(
            sam_version=sam_version,
            sam_encoder=sam_encoder,
            sam2_origin=sam2_origin,
        )
        self.cuda_postprocess_available = self._sam2_cuda_postprocess_available()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        sam = build_sam2(
            self.model_cfg,
            self.checkpoint_path,
            device=self.device,
            mode="eval",
            apply_postprocessing=False,
        )
        sam_config = {
            "points_per_side": int(config.get("points_per_side", 16)),
            "pred_iou_thresh": float(config.get("nms_iou_th", 0.8)),
            "stability_score_thresh": float(config.get("stability_score_th", 0.95)),
            "min_mask_region_area": int(config.get("min_mask_region_area", 100)),
            "use_m2m": bool(config.get("use_m2m", False)),
        }
        self.mask_generator = SamAutomaticMaskGenerator(model=sam, **sam_config)
        self.image_predictor = getattr(self.mask_generator, "predictor", None)
        if self.image_predictor is None and SAM2ImagePredictor is not None:
            self.image_predictor = SAM2ImagePredictor(
                sam,
                max_hole_area=sam_config["min_mask_region_area"],
                max_sprinkle_area=sam_config["min_mask_region_area"],
            )
        self._disable_runtime_postprocess_if_needed()
        self.last_generation_info = {
            "device": self.device,
            "model_cfg": self.model_cfg,
            "checkpoint_path": self.checkpoint_path,
            "sam2_origin": sam2_origin,
            "sam2_repo_root": self.sam2_repo_root,
            "cuda_postprocess_available": self.cuda_postprocess_available,
            "sam_config": sam_config,
            "anchor_prompt_enabled": self.anchor_prompt_enabled,
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        if self.mask_generator is None:
            raise RuntimeError("SAM2ProposalBackend.initialize() must be called before inference.")

        torch = self._import_torch()
        autocast_context = nullcontext()
        if self.device.startswith("cuda"):
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        start = time.perf_counter()
        with torch.inference_mode():
            with autocast_context:
                masks = self.mask_generator.generate(np.array(rgb, copy=True))

        proposals: List[Proposal2D] = []
        raw_count = len(masks)
        for mask_idx, mask_data in enumerate(masks):
            segmentation = np.asarray(mask_data["segmentation"], dtype=bool)
            area = int(segmentation.sum())
            confidence = float(
                mask_data.get(
                    "predicted_iou",
                    mask_data.get("stability_score", 1.0),
                )
            )
            if area <= 0 or confidence < self.confidence_threshold:
                continue

            bbox_xyxy = self._mask_to_bbox_xyxy(segmentation, mask_data.get("bbox"))
            proposals.append(
                Proposal2D(
                    proposal_id=mask_idx,
                    mask=segmentation,
                    bbox_xyxy=bbox_xyxy,
                    area=area,
                    confidence=confidence,
                    backend_name="sam2",
                    metadata={
                        "backend": "sam2",
                        "raw_mask_index": mask_idx,
                        "predicted_iou": float(mask_data.get("predicted_iou", 1.0)),
                        "stability_score": float(mask_data.get("stability_score", 1.0)),
                    },
                )
            )

        proposals.sort(key=lambda proposal: (proposal.confidence, proposal.area), reverse=True)
        proposals = proposals[: self.max_proposals]
        for proposal_id, proposal in enumerate(proposals):
            proposal.proposal_id = proposal_id

        self.last_generation_info = {
            **self.last_generation_info,
            "raw_mask_count": raw_count,
            "kept_mask_count": len(proposals),
            "proposal_source": "full_frame_sam",
        }
        self.last_generation_timings = {"sam2_proposals": max(0.0, time.perf_counter() - start)}
        return proposals

    def generate_proposals_for_anchors(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        anchors: List[Anchor2D],
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        """Generate SAM2 masks directly from detector anchor boxes.

        The baseline path keeps using full-frame automatic masks unless
        ``anchor_prompt.enabled`` is explicitly enabled in the SAM2 config.
        """
        if not self.anchor_prompt_enabled:
            return self.generate_proposals(rgb, depth, frame=frame)

        start = time.perf_counter()
        image = np.array(rgb, copy=True)
        height, width = image.shape[:2]
        valid_records: list[tuple[Anchor2D, np.ndarray]] = []
        skipped_invalid = 0
        for anchor in anchors:
            clipped_box = self._clip_anchor_box(anchor.bbox_xyxy, width=width, height=height)
            if clipped_box is None:
                skipped_invalid += 1
                continue
            valid_records.append((anchor, clipped_box))

        if not valid_records:
            self.last_generation_info = {
                **self.last_generation_info,
                "proposal_source": "anchor_prompted_sam",
                "anchor_count": int(len(anchors)),
                "prompted_anchor_count": 0,
                "raw_mask_count": 0,
                "kept_mask_count": 0,
                "skipped_anchor_count": int(skipped_invalid),
                "mask_count_per_anchor": [],
            }
            self.last_generation_timings = {"sam2_anchor_prompts": max(0.0, time.perf_counter() - start)}
            return []
        if self.image_predictor is None:
            raise RuntimeError("SAM2 anchor prompts require SAM2ImagePredictor support.")

        torch = self._import_torch()
        autocast_context = nullcontext()
        if self.device.startswith("cuda"):
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            with autocast_context:
                self.image_predictor.set_image(image)
                predictions = self._predict_anchor_boxes(valid_records)

        proposals: list[Proposal2D] = []
        raw_mask_count = 0
        mask_count_per_anchor: list[int] = []
        skipped_filter = 0
        for anchor, box, masks, scores in predictions:
            candidate_masks = self._normalize_predicted_masks(masks)
            candidate_scores = self._normalize_scores(scores, len(candidate_masks))
            selected = self._select_anchor_masks(
                candidate_masks=candidate_masks,
                candidate_scores=candidate_scores,
                anchor_box=box,
                depth=depth,
            )
            raw_mask_count += len(candidate_masks)
            mask_count_per_anchor.append(len(selected))
            if not selected:
                skipped_filter += 1
                continue
            for mask, score, coverage, depth_valid_coverage in selected:
                proposal_id = len(proposals)
                proposals.append(
                    self._build_anchor_prompted_proposal(
                        proposal_id=proposal_id,
                        mask=mask,
                        score=score,
                        anchor=anchor,
                        anchor_box=box,
                        anchor_box_coverage=coverage,
                        depth_valid_coverage=depth_valid_coverage,
                    )
                )
                if len(proposals) >= self.max_proposals:
                    break
            if len(proposals) >= self.max_proposals:
                break

        self.last_generation_info = {
            **self.last_generation_info,
            "proposal_source": "anchor_prompted_sam",
            "anchor_count": int(len(anchors)),
            "prompted_anchor_count": int(len(valid_records)),
            "raw_mask_count": int(raw_mask_count),
            "kept_mask_count": int(len(proposals)),
            "skipped_anchor_count": int(skipped_invalid + skipped_filter),
            "mask_count_per_anchor": [int(value) for value in mask_count_per_anchor],
        }
        self.last_generation_timings = {"sam2_anchor_prompts": max(0.0, time.perf_counter() - start)}
        return proposals

    def _configure_anchor_prompts(self, config: Dict[str, Any]) -> None:
        prompt_cfg = config.get("anchor_prompt", {})
        if not isinstance(prompt_cfg, dict):
            prompt_cfg = {}
        self.anchor_prompt_enabled = bool(prompt_cfg.get("enabled", False))
        self.anchor_prompt_multimask_output = bool(prompt_cfg.get("multimask_output", False))
        default_max_masks = 3 if self.anchor_prompt_multimask_output else 1
        self.anchor_prompt_max_masks_per_anchor = max(
            1,
            int(prompt_cfg.get("max_masks_per_anchor", default_max_masks) or default_max_masks),
        )
        self.anchor_prompt_min_mask_area = int(prompt_cfg.get("min_mask_area", config.get("min_mask_area", 25)))
        self.anchor_prompt_min_anchor_box_coverage = float(
            np.clip(prompt_cfg.get("min_anchor_box_coverage", 0.0), 0.0, 1.0)
        )
        self.anchor_prompt_min_depth_valid_coverage = float(
            np.clip(prompt_cfg.get("min_depth_valid_coverage", 0.0), 0.0, 1.0)
        )
        self.anchor_prompt_box_padding_px = float(prompt_cfg.get("box_padding_px", 0.0) or 0.0)
        self.anchor_prompt_batch_boxes = bool(prompt_cfg.get("batch_boxes", True))

    def _predict_anchor_boxes(
        self,
        records: list[tuple[Anchor2D, np.ndarray]],
    ) -> list[tuple[Anchor2D, np.ndarray, Any, Any]]:
        if self.anchor_prompt_batch_boxes and len(records) > 1:
            try:
                boxes = np.stack([box for _, box in records], axis=0).astype(np.float32)
                masks, scores, _ = self.image_predictor.predict(
                    box=boxes,
                    multimask_output=self.anchor_prompt_multimask_output,
                    return_logits=False,
                    normalize_coords=True,
                )
                masks_by_anchor = self._split_batched_masks(masks, len(records))
                scores_by_anchor = self._split_batched_scores(scores, len(records))
                return [
                    (anchor, box, anchor_masks, anchor_scores)
                    for (anchor, box), anchor_masks, anchor_scores in zip(
                        records,
                        masks_by_anchor,
                        scores_by_anchor,
                    )
                ]
            except Exception as exc:
                logger.debug("SAM2 batched box prompts failed; falling back to per-anchor prompts: %s", exc)

        predictions = []
        for anchor, box in records:
            masks, scores, _ = self.image_predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                multimask_output=self.anchor_prompt_multimask_output,
                return_logits=False,
                normalize_coords=True,
            )
            predictions.append((anchor, box, masks, scores))
        return predictions

    @staticmethod
    def _split_batched_masks(masks: Any, count: int) -> list[np.ndarray]:
        masks_array = np.asarray(masks)
        if masks_array.ndim == 4 and masks_array.shape[0] == count:
            return [np.asarray(masks_array[index]) for index in range(count)]
        if masks_array.ndim == 3 and masks_array.shape[0] == count:
            return [np.asarray(masks_array[index][None, ...]) for index in range(count)]
        if count == 1:
            return [masks_array]
        raise ValueError(f"Unexpected batched SAM2 mask shape {masks_array.shape} for {count} boxes.")

    @staticmethod
    def _split_batched_scores(scores: Any, count: int) -> list[np.ndarray]:
        scores_array = np.asarray(scores, dtype=np.float32)
        if scores_array.ndim == 2 and scores_array.shape[0] == count:
            return [np.asarray(scores_array[index], dtype=np.float32) for index in range(count)]
        if scores_array.ndim == 1 and scores_array.shape[0] == count:
            return [np.asarray([scores_array[index]], dtype=np.float32) for index in range(count)]
        if count == 1:
            return [scores_array.reshape(-1)]
        raise ValueError(f"Unexpected batched SAM2 score shape {scores_array.shape} for {count} boxes.")

    @staticmethod
    def _normalize_predicted_masks(masks: Any) -> list[np.ndarray]:
        masks_array = np.asarray(masks)
        if masks_array.ndim == 2:
            masks_array = masks_array[None, ...]
        if masks_array.ndim == 4 and masks_array.shape[0] == 1:
            masks_array = masks_array[0]
        if masks_array.ndim != 3:
            return []
        return [np.asarray(mask, dtype=bool) for mask in masks_array]

    @staticmethod
    def _normalize_scores(scores: Any, count: int) -> list[float]:
        scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores_array.size == 0:
            return [1.0] * count
        if scores_array.size < count:
            scores_array = np.pad(scores_array, (0, count - scores_array.size), constant_values=float(scores_array[-1]))
        return [float(value) for value in scores_array[:count]]

    def _select_anchor_masks(
        self,
        *,
        candidate_masks: list[np.ndarray],
        candidate_scores: list[float],
        anchor_box: np.ndarray,
        depth: np.ndarray,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        selected: list[tuple[np.ndarray, float, float, float, float]] = []
        anchor_mask = self._bbox_mask(depth.shape, anchor_box)
        anchor_area = max(int(anchor_mask.sum()), 1)
        valid_depth = np.isfinite(depth) & (np.asarray(depth) > 0)
        for mask, score in zip(candidate_masks, candidate_scores):
            mask = np.asarray(mask, dtype=bool)
            area = int(mask.sum())
            if area < self.anchor_prompt_min_mask_area or score < self.confidence_threshold:
                continue
            anchor_box_coverage = float(np.logical_and(mask, anchor_mask).sum() / anchor_area)
            if anchor_box_coverage < self.anchor_prompt_min_anchor_box_coverage:
                continue
            depth_valid_coverage = float(np.logical_and(mask, valid_depth).sum() / max(area, 1))
            if depth_valid_coverage < self.anchor_prompt_min_depth_valid_coverage:
                continue
            rank_score = float(score) + 0.25 * anchor_box_coverage + 0.10 * depth_valid_coverage
            selected.append((rank_score, mask, float(score), anchor_box_coverage, depth_valid_coverage))
        selected.sort(key=lambda item: item[0], reverse=True)
        return [
            (mask.copy(), score, anchor_box_coverage, depth_valid_coverage)
            for _, mask, score, anchor_box_coverage, depth_valid_coverage in selected[
                : self.anchor_prompt_max_masks_per_anchor
            ]
        ]

    def _build_anchor_prompted_proposal(
        self,
        *,
        proposal_id: int,
        mask: np.ndarray,
        score: float,
        anchor: Anchor2D,
        anchor_box: np.ndarray,
        anchor_box_coverage: float,
        depth_valid_coverage: float,
    ) -> Proposal2D:
        metadata = dict(getattr(anchor, "metadata", {}) or {})
        class_name = str(anchor.class_name)
        metadata.update(
            {
                "source": "anchor_prompted_sam",
                "proposal_source": "anchor_prompted_sam",
                "geometry_source": "sam2_anchor_prompt",
                "mask_source": "sam2_box_prompt",
                "observation_layer": "fine",
                "anchor_id": int(anchor.anchor_id),
                "anchor_bbox": np.asarray(anchor_box, dtype=np.float32).copy(),
                "anchor_class": class_name,
                "anchor_class_name": class_name,
                "anchor_confidence": float(anchor.confidence),
                "anchor_source_bbox_xyxy": np.asarray(anchor.bbox_xyxy, dtype=np.float32).copy(),
                "anchor_source_class_name": class_name,
                "anchor_source_confidence": float(anchor.confidence),
                "anchor_label_strength": "strong",
                "anchor_keepalive": True,
                "anchor_center_inside": True,
                "anchor_bbox_iou": float(self._bbox_iou(self._mask_to_bbox_xyxy(mask), anchor_box)),
                "anchor_vote_score": float(anchor.confidence),
                "anchor_label_votes": {class_name: float(anchor.confidence)},
                "anchor_candidate_classes": [class_name],
                "anchor_candidate_ids": [int(anchor.anchor_id)],
                "semantic_commit_allowed": True,
                "source_raw_proposal_id": int(proposal_id),
                "source_raw_proposal_ids": [int(proposal_id)],
                "sam_score": float(score),
                "predicted_iou": float(score),
                "anchor_box_coverage": float(anchor_box_coverage),
                "depth_valid_coverage": float(depth_valid_coverage),
            }
        )
        return Proposal2D(
            proposal_id=int(proposal_id),
            mask=np.asarray(mask, dtype=bool).copy(),
            bbox_xyxy=self._mask_to_bbox_xyxy(mask),
            area=int(np.asarray(mask, dtype=bool).sum()),
            confidence=float(score),
            backend_name="sam2_anchor_prompt",
            metadata=metadata,
        )

    def _clip_anchor_box(self, bbox_xyxy: Any, *, width: int, height: int) -> np.ndarray | None:
        box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4).copy()
        if self.anchor_prompt_box_padding_px:
            pad = float(self.anchor_prompt_box_padding_px)
            box += np.array([-pad, -pad, pad, pad], dtype=np.float32)
        box[0::2] = np.clip(box[0::2], 0.0, float(width))
        box[1::2] = np.clip(box[1::2], 0.0, float(height))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return box.astype(np.float32)

    @staticmethod
    def _bbox_mask(shape: tuple[int, int], bbox_xyxy: np.ndarray) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
        x1_i = int(np.floor(np.clip(x1, 0, width)))
        y1_i = int(np.floor(np.clip(y1, 0, height)))
        x2_i = int(np.ceil(np.clip(x2, 0, width)))
        y2_i = int(np.ceil(np.clip(y2, 0, height)))
        mask = np.zeros((height, width), dtype=bool)
        if x2_i > x1_i and y2_i > y1_i:
            mask[y1_i:y2_i, x1_i:x2_i] = True
        return mask

    @staticmethod
    def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
        a = np.asarray(first, dtype=np.float32).reshape(4)
        b = np.asarray(second, dtype=np.float32).reshape(4)
        inter_x1 = max(float(a[0]), float(b[0]))
        inter_y1 = max(float(a[1]), float(b[1]))
        inter_x2 = min(float(a[2]), float(b[2]))
        inter_y2 = min(float(a[3]), float(b[3]))
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter
        return float(inter / union) if union > 0.0 else 0.0

    @staticmethod
    def _mask_to_bbox_xyxy(mask: np.ndarray, bbox_xywh: Any | None = None) -> np.ndarray:
        if bbox_xywh is not None:
            x, y, w, h = [float(value) for value in bbox_xywh]
            return np.array([x, y, x + w, y + h], dtype=np.float32)

        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            dtype=np.float32,
        )

    @staticmethod
    def _resolve_checkpoint_root(config: Dict[str, Any]) -> Path:
        root = Path(config.get("sam_ckpt_path", "data/input/sam_ckpts"))
        if root.is_absolute():
            return root
        project_root = Path(__file__).resolve().parents[2]
        return (project_root / root).resolve()

    @staticmethod
    def _resolve_sam2_repo_root(config: Dict[str, Any]) -> Path | None:
        configured = config.get("sam_repo_root") or config.get("repo_root")
        if configured:
            root = Path(configured).expanduser()
            return root.resolve() if root.exists() else None
        return None

    @staticmethod
    def _resolve_model_cfg(
        sam_version: str,
        sam_encoder: str,
        sam2_origin: str,
    ) -> str:
        origin_path = Path(sam2_origin).resolve()
        package_root = origin_path.parent
        suffix = sam_encoder.split("_", 1)[-1]
        flat_config = package_root / "configs" / f"sam{sam_version}_{sam_encoder}.yaml"
        nested_config = package_root / "configs" / f"sam{sam_version}" / f"sam{sam_version}_{sam_encoder}.yaml"
        root_link_config = package_root / f"sam2_hiera_{suffix}.yaml"

        if flat_config.exists():
            return f"configs/sam{sam_version}_{sam_encoder}.yaml"
        if nested_config.exists():
            return os.path.join("configs", f"sam{sam_version}", f"sam{sam_version}_{sam_encoder}.yaml")
        if root_link_config.exists():
            return f"sam2_hiera_{sam_encoder.split('_', 1)[-1]}.yaml"
        return os.path.join("configs", f"sam{sam_version}", f"sam{sam_version}_{sam_encoder}.yaml")

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SAM2 backend requires torch. Run it with the OVO conda environment."
            ) from exc
        return torch

    @staticmethod
    def _import_sam2(preferred_root: Path | None = None):
        fallback_roots = []
        if preferred_root is not None and preferred_root.exists():
            fallback_roots.append(preferred_root.resolve())
        for candidate in [
            Path("/home/ww/vv/paper2/OVO/thirdParty/segment-anything-2"),
            Path("/home/phl/vv/paper2/OVO/thirdParty/segment-anything-2"),
        ]:
            try:
                if candidate.exists():
                    fallback_roots.append(candidate.resolve())
            except OSError:
                continue

        for fallback_root in fallback_roots:
            fallback_root_str = str(fallback_root)
            if fallback_root_str in sys.path:
                sys.path.remove(fallback_root_str)
            sys.path.insert(0, fallback_root_str)

            for module_name in list(sys.modules):
                if module_name == "sam2" or module_name.startswith("sam2."):
                    del sys.modules[module_name]

            try:
                sam2_pkg = importlib.import_module("sam2")
                build_sam_mod = importlib.import_module("sam2.build_sam")
                amg_mod = importlib.import_module("sam2.automatic_mask_generator")
                return (
                    build_sam_mod.build_sam2,
                    amg_mod.SAM2AutomaticMaskGenerator,
                    str(Path(sam2_pkg.__file__).resolve()),
                )
            except ModuleNotFoundError:
                continue

        try:
            sam2_pkg = importlib.import_module("sam2")
            build_sam_mod = importlib.import_module("sam2.build_sam")
            amg_mod = importlib.import_module("sam2.automatic_mask_generator")
            return (
                build_sam_mod.build_sam2,
                amg_mod.SAM2AutomaticMaskGenerator,
                str(Path(sam2_pkg.__file__).resolve()),
            )
        except ModuleNotFoundError:
            pass

        raise RuntimeError(
            "SAM2 package not found. Run with 'conda run -n ovopro ...' or install SAM2."
        )

    @staticmethod
    def _import_sam2_image_predictor():
        try:
            predictor_mod = importlib.import_module("sam2.sam2_image_predictor")
            return predictor_mod.SAM2ImagePredictor
        except ModuleNotFoundError:
            return None

    @staticmethod
    def _sam2_cuda_postprocess_available() -> bool:
        try:
            importlib.import_module("sam2._C")
            return True
        except Exception:
            return False

    def _disable_runtime_postprocess_if_needed(self) -> None:
        """Disable SAM2 CUDA-dependent postprocessing when the extension is unavailable."""
        if self.mask_generator is None or self.cuda_postprocess_available:
            return

        predictor = getattr(self.mask_generator, "predictor", None)
        transforms = getattr(predictor, "_transforms", None)
        if transforms is not None:
            transforms.max_hole_area = 0.0
            transforms.max_sprinkle_area = 0.0

        logger.warning(
            "SAM2 CUDA postprocessing extension unavailable; hole/sprinkle postprocess disabled."
        )
