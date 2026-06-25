"""SAM2 proposal backend adapted from the OVO frontend setup."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List
import os
import sys

import numpy as np

from src.core.data_structures import Frame, Proposal2D
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
        self.sam2_repo_root: str | None = None
        self.cuda_postprocess_available = False

    def initialize(self, config: Dict[str, Any]) -> None:
        torch = self._import_torch()

        self.device = config.get("device", "cuda")
        self.max_proposals = int(config.get("max_proposals", 50))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
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
        self._disable_runtime_postprocess_if_needed()
        self.last_generation_info = {
            "device": self.device,
            "model_cfg": self.model_cfg,
            "checkpoint_path": self.checkpoint_path,
            "sam2_origin": sam2_origin,
            "sam2_repo_root": self.sam2_repo_root,
            "cuda_postprocess_available": self.cuda_postprocess_available,
            "sam_config": sam_config,
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
        }
        return proposals

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
