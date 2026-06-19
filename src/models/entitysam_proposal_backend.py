"""EntitySAM proposal backend using the local EntitySAM repo.

EntitySAM is a video entity segmentation model, not a plain single-image
automatic mask generator. This adapter keeps the frontend switchable while
using EntitySAM in the role it was designed for: a temporal proposal frontend
that maintains video state and exposes current-frame masks as Proposal2D.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List
import os
import sys

import numpy as np

from src.core.data_structures import Frame, Proposal2D
from src.models.proposal_backend import ProposalBackend


class EntitySAMProposalBackend(ProposalBackend):
    """Proposal backend powered by EntitySAM video query inference."""

    def __init__(self) -> None:
        self.predictor: Any | None = None
        self.device = "cuda"
        self.max_proposals = 50
        self.confidence_threshold = 0.0
        self.repo_root = ""
        self.model_cfg = ""
        self.model_cfg_path = ""
        self.checkpoint_path = ""
        self.sequence_dir = ""
        self.mask_decoder_depth = 8
        self.mask_binary_threshold = 0.5
        self.object_mask_threshold = 0.05
        self.topk_per_frame = 100

        self._frame_names: List[str] = []
        self._inference_state: Any | None = None
        self._generator: Any | None = None
        self._current_step = 0
        self._output_queue: Deque[dict[str, Any]] = deque()
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        torch = self._import_torch()

        self.device = str(config.get("device", "cuda"))
        self.max_proposals = int(config.get("max_proposals", 50))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
        self.mask_decoder_depth = int(config.get("mask_decoder_depth", 8))
        self.mask_binary_threshold = float(config.get("mask_binary_threshold", 0.5))
        self.object_mask_threshold = float(config.get("object_mask_threshold", 0.05))
        self.topk_per_frame = int(config.get("topk_per_frame", 100))

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("EntitySAM backend requested CUDA, but torch.cuda.is_available() is False.")

        self.repo_root = str(config.get("repo_root", "/home/phl/vv/paper2/entitysam"))
        repo_root_path = Path(self.repo_root).expanduser().resolve()
        if not repo_root_path.is_dir():
            raise NotADirectoryError(
                f"EntitySAM repo_root not found: {repo_root_path}. "
                "Clone from https://github.com/ymq2017/entitysam or update proposal.entitysam.repo_root."
            )

        sequence_dir_value = config.get("sequence_dir", "")
        if not sequence_dir_value:
            raise RuntimeError(
                "EntitySAM backend requires proposal.entitysam.sequence_dir to point at an RGB frame folder."
            )
        sequence_dir_path = Path(sequence_dir_value).expanduser()
        if not sequence_dir_path.is_absolute():
            sequence_dir_path = repo_root_path / sequence_dir_path
        sequence_dir_path = sequence_dir_path.resolve()
        if not sequence_dir_path.is_dir():
            raise NotADirectoryError(
                f"EntitySAM sequence_dir not found: {sequence_dir_path}. "
                "Point it at a directory of ordered RGB frames, e.g. Replica room0/rgb."
            )
        self.sequence_dir = str(sequence_dir_path)

        config_path_value = str(config.get("config_path", "configs/sam2.1_hiera_l.yaml"))
        config_name, config_path = self._resolve_hydra_config(repo_root_path, config_path_value)

        checkpoint_rel = str(config.get("checkpoint_path", "checkpoints/vit-l/model_0009999.pth"))
        checkpoint_path = Path(checkpoint_rel)
        if not checkpoint_path.is_absolute():
            checkpoint_path = repo_root_path / checkpoint_rel
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"EntitySAM checkpoint not found: {checkpoint_path}. "
                "Download from https://huggingface.co/mqye/entitysam/tree/main "
                "or update proposal.entitysam.checkpoint_path."
            )

        self.model_cfg = config_name
        self.model_cfg_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)

        build_video_predictor = self._import_entitysam(repo_root_path)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        with self._working_directory(repo_root_path):
            with self._autocast():
                self.predictor = build_video_predictor(
                    self.model_cfg,
                    self.checkpoint_path,
                    device=self.device,
                    mode="eval",
                    apply_postprocessing=False,
                    mask_decoder_depth=self.mask_decoder_depth,
                )
        self._frame_names = self._list_frame_names(sequence_dir_path)
        self._reset_video_state()

        self.last_generation_info = {
            "backend": "entitysam",
            "device": self.device,
            "model_cfg": self.model_cfg,
            "model_cfg_path": self.model_cfg_path,
            "checkpoint_path": self.checkpoint_path,
            "repo_root": self.repo_root,
            "sequence_dir": self.sequence_dir,
            "frame_count": len(self._frame_names),
            "mask_decoder_depth": self.mask_decoder_depth,
            "mask_binary_threshold": self.mask_binary_threshold,
            "object_mask_threshold": self.object_mask_threshold,
            "topk_per_frame": self.topk_per_frame,
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        if self.predictor is None:
            raise RuntimeError("EntitySAMProposalBackend.initialize() must be called before inference.")
        if frame is None:
            raise RuntimeError("EntitySAM backend requires frame context to keep temporal state aligned.")

        if frame.frame_id == 0 and self._current_step != 0:
            self._reset_video_state()
        if frame.frame_id != self._current_step:
            raise RuntimeError(
                f"EntitySAM backend expects sequential frame ids starting from 0. "
                f"Expected frame_id {self._current_step}, got {frame.frame_id}."
            )

        output = self._next_frame_output()
        if output["frame_id"] != frame.frame_id:
            raise RuntimeError(
                f"EntitySAM output/frame mismatch. Backend yielded frame {output['frame_id']}, "
                f"but pipeline requested frame {frame.frame_id}."
            )

        proposals = self._normalize_frame_output(output, target_size=rgb.shape[:2])
        raw_proposals = [self._clone_proposal(proposal) for proposal in proposals]

        proposals = [proposal for proposal in proposals if proposal.confidence >= self.confidence_threshold]
        proposals.sort(key=lambda proposal: (proposal.confidence, proposal.area), reverse=True)
        proposals = proposals[: self.max_proposals]

        for proposal_id, proposal in enumerate(proposals):
            proposal.proposal_id = proposal_id

        self._current_step += 1
        self.last_generation_info = {
            **self.last_generation_info,
            "last_frame_id": frame.frame_id,
            "raw_mask_count": len(raw_proposals),
            "raw_proposals": raw_proposals,
            "kept_mask_count": len(proposals),
        }
        return proposals

    def _reset_video_state(self) -> None:
        assert self.predictor is not None
        repo_root_path = Path(self.repo_root)
        with self._working_directory(repo_root_path):
            with self._autocast():
                self._inference_state = self.predictor.init_state(video_path=self.sequence_dir)
                self.predictor.reset_state(self._inference_state)
                self._generator = self.predictor.propagate_in_video(self._inference_state, start_frame_idx=0)
        self._current_step = 0
        self._output_queue.clear()

    def _next_frame_output(self) -> dict[str, Any]:
        if self._generator is None:
            raise RuntimeError("EntitySAM generator is not initialized.")
        try:
            with self._autocast():
                out_frame_idx, obj_ids, pred_masks, pred_eiou = next(self._generator)
        except StopIteration as exc:
            raise RuntimeError("EntitySAM sequence ended before the requested frame count.") from exc

        frame_name = self._frame_names[out_frame_idx] if out_frame_idx < len(self._frame_names) else None
        return {
            "frame_id": int(out_frame_idx),
            "obj_ids": list(obj_ids) if isinstance(obj_ids, Iterable) else [],
            "pred_masks": pred_masks,
            "pred_eiou": pred_eiou,
            "frame_name": frame_name,
            "mask_count": int(pred_masks.shape[0]) if hasattr(pred_masks, "shape") else 0,
        }

    def _normalize_frame_output(
        self,
        output: dict[str, Any],
        target_size: tuple[int, int],
    ) -> List[Proposal2D]:
        torch = self._import_torch()
        pred_masks = output["pred_masks"]
        pred_eiou = output["pred_eiou"]

        if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks[:, 0]
        elif pred_masks.ndim != 3:
            raise RuntimeError(f"Unexpected EntitySAM pred_masks shape: {tuple(pred_masks.shape)}")

        target_h, target_w = target_size
        masks_resized = torch.nn.functional.interpolate(
            pred_masks.unsqueeze(1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        scores = pred_eiou
        if scores.ndim > 1:
            scores = scores.squeeze()
        scores = scores.float()

        keep = scores >= max(
            self.object_mask_threshold,
            float(scores.topk(k=min(len(scores), self.topk_per_frame))[0][-1]) if len(scores) > 0 else 1.0,
        )
        masks_resized = masks_resized[keep]
        scores = scores[keep]

        masks_sigmoid = masks_resized.sigmoid()
        proposals: List[Proposal2D] = []
        for query_index in range(masks_sigmoid.shape[0]):
            mask = (masks_sigmoid[query_index] >= self.mask_binary_threshold).detach().cpu().numpy().astype(bool)
            area = int(mask.sum())
            if area <= 0:
                continue
            bbox_xyxy = self._mask_to_bbox_xyxy(mask)
            confidence = float(scores[query_index].item())
            entity_id = None
            if output["obj_ids"] and query_index < len(output["obj_ids"]):
                entity_id = output["obj_ids"][query_index]

            proposals.append(
                Proposal2D(
                    proposal_id=query_index,
                    mask=mask,
                    bbox_xyxy=bbox_xyxy,
                    area=area,
                    confidence=confidence,
                    backend_name="entitysam",
                    metadata={
                        "backend": "entitysam",
                        "frame_id": output["frame_id"],
                        "frame_name": output["frame_name"],
                        "query_id": query_index,
                        "entity_id": entity_id,
                        "pred_eiou": confidence,
                    },
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
    def _list_frame_names(sequence_dir: Path) -> List[str]:
        frame_names = [
            path.name
            for path in sorted(sequence_dir.iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not frame_names:
            raise RuntimeError(f"No RGB frames found in EntitySAM sequence_dir: {sequence_dir}")
        return frame_names

    @staticmethod
    def _clone_proposal(proposal: Proposal2D) -> Proposal2D:
        return Proposal2D(
            proposal_id=proposal.proposal_id,
            mask=proposal.mask.copy(),
            bbox_xyxy=proposal.bbox_xyxy.copy(),
            area=proposal.area,
            confidence=proposal.confidence,
            backend_name=proposal.backend_name,
            metadata=dict(proposal.metadata),
        )

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "EntitySAM backend requires torch. Install PyTorch or use the appropriate conda environment."
            ) from exc
        return torch

    def _autocast(self):
        torch = self._import_torch()
        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _import_entitysam(repo_root: Path):
        """Import the video query predictor builder from the EntitySAM repo."""
        repo_str = str(repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from sam2.build_sam import build_sam2_video_query_iou_predictor
            return build_sam2_video_query_iou_predictor
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import EntitySAM modules from {repo_root}. "
                "Ensure the repo is properly set up: pip install -e . in the EntitySAM directory. "
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    @contextmanager
    def _working_directory(path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    @staticmethod
    def _resolve_hydra_config(repo_root: Path, config_value: str) -> tuple[str, Path]:
        candidate = Path(config_value)
        candidates: list[tuple[str, Path]] = []
        if candidate.is_absolute():
            resolved = candidate.resolve()
            candidates.append((config_value, resolved))
        else:
            candidates.append((config_value, (repo_root / candidate).resolve()))
            candidates.append((config_value, (repo_root / "sam2" / candidate).resolve()))
            if config_value.startswith("sam2/"):
                trimmed = config_value[len("sam2/") :]
                candidates.append((trimmed, (repo_root / "sam2" / trimmed).resolve()))

        for config_name, path in candidates:
            if path.exists():
                if path.is_relative_to((repo_root / "sam2").resolve()):
                    rel = path.relative_to((repo_root / "sam2").resolve())
                    return str(rel).replace(os.sep, "/"), path
                return config_name, path

        raise FileNotFoundError(
            f"EntitySAM config file not found for '{config_value}'. "
            f"Tried relative to {repo_root} and {repo_root / 'sam2'}."
        )
