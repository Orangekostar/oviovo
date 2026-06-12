"""Experimental SAM3 concept proposal backend."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.core.data_structures import Frame, Proposal2D
from src.models.json_line_worker_client import JsonLineWorkerClient
from src.models.proposal_backend import ProposalBackend


class SAM3ConceptProposalBackend(ProposalBackend):
    """SAM3 concept backend with disabled, mock, and JSONL worker modes."""

    _SUPPORTED_MODES = {"mock", "worker", "disabled"}
    _SUPPORTED_CANDIDATE_STRATEGIES = {"all", "fixed_subset", "round_robin"}

    def __init__(self) -> None:
        self.mode = "disabled"
        self.classes: list[str] = []
        self.always_include_classes: list[str] = []
        self.candidate_strategy = "all"
        self.max_prompts_per_frame = 0
        self.max_proposals = 64
        self.confidence_threshold = 0.0
        self.worker_python = sys.executable
        self.worker_script = ""
        self.worker_env: dict[str, str] = {}
        self.request_timeout_sec = 30.0
        self.device: str | None = None
        self.sam3_repo_root: str | None = None
        self.checkpoint_path: str | None = None
        self.worker_client: JsonLineWorkerClient | None = None
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: dict) -> None:
        self.mode = str(config.get("mode", "disabled")).strip().lower()
        if self.mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported sam3_concept mode '{self.mode}'. "
                "Expected one of: disabled, mock, worker."
            )

        self.classes = self._normalize_classes(config.get("classes", []))
        self.always_include_classes = self._normalize_classes(config.get("always_include_classes", []))
        self.candidate_strategy = str(config.get("candidate_strategy", "all")).strip().lower() or "all"
        if self.candidate_strategy not in self._SUPPORTED_CANDIDATE_STRATEGIES:
            raise ValueError(
                f"Unsupported sam3_concept candidate_strategy '{self.candidate_strategy}'. "
                "Expected one of: all, fixed_subset, round_robin."
            )
        self.max_prompts_per_frame = int(config.get("max_prompts_per_frame", 0))
        if self.max_prompts_per_frame < 0:
            raise ValueError("sam3_concept max_prompts_per_frame must be >= 0.")
        self.max_proposals = int(config.get("max_proposals", 64))
        if self.max_proposals < 0:
            raise ValueError("sam3_concept max_proposals must be >= 0.")
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
        self.worker_python = str(config.get("worker_python", sys.executable)).strip() or sys.executable
        self.worker_script = str(config.get("worker_script", "")).strip()
        self.request_timeout_sec = float(config.get("request_timeout_sec", 30.0))
        if self.request_timeout_sec <= 0:
            raise ValueError("sam3_concept request_timeout_sec must be > 0.")
        self.device = self._optional_config_str(config, "device")
        self.sam3_repo_root = self._optional_config_str(config, "sam3_repo_root")
        if self.sam3_repo_root is None:
            self.sam3_repo_root = self._optional_config_str(config, "repo_root")
        self.checkpoint_path = self._optional_config_str(config, "checkpoint_path")
        if self.checkpoint_path is None:
            self.checkpoint_path = self._optional_config_str(config, "sam3_checkpoint_path")
        self.worker_env = {
            str(key): str(value)
            for key, value in dict(config.get("worker_env", {}) or {}).items()
        }

        if self.mode == "worker":
            self.worker_client = self._create_worker_client()

        self.last_generation_info = {
            "backend": "sam3_concept",
            "mode": self.mode,
            "class_count": len(self.classes),
            "candidate_strategy": self.candidate_strategy,
            "max_prompts_per_frame": self.max_prompts_per_frame,
            "always_include_class_count": len(self.always_include_classes),
            "max_proposals": self.max_proposals,
            "confidence_threshold": self.confidence_threshold,
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> list[Proposal2D]:
        if self.mode == "disabled":
            return []
        if self.mode == "mock":
            return self._mock_proposals(rgb, frame)
        response = self._request_worker(rgb, frame)
        return self._normalize_response(response, rgb.shape[:2])

    def close(self) -> None:
        if self.worker_client is None:
            return
        try:
            self.worker_client.close()
        finally:
            self.worker_client = None

    def _create_worker_client(self) -> JsonLineWorkerClient:
        if not self.worker_script:
            raise ValueError("sam3_concept worker mode requires worker_script.")
        worker_script_path = self._resolve_path(self.worker_script)
        if not worker_script_path.is_file():
            raise FileNotFoundError(f"sam3_concept worker_script does not exist: {worker_script_path}")
        command = [self.worker_python, str(worker_script_path)]
        if self.device is not None:
            command.extend(["--device", self.device])
        if self.sam3_repo_root is not None:
            command.extend(["--sam3-repo-root", self.sam3_repo_root])
        if self.checkpoint_path is not None:
            command.extend(["--checkpoint-path", self.checkpoint_path])
        env = os.environ.copy()
        env.update(self.worker_env)
        return JsonLineWorkerClient(
            command=command,
            env=env,
            cwd=str(self._repo_root()),
            request_timeout_sec=self.request_timeout_sec,
        )

    def _request_worker(self, rgb: np.ndarray, frame: Frame | None) -> dict[str, Any]:
        if self.worker_client is None:
            raise RuntimeError("SAM3 concept worker is not running.")

        selected_classes = self._select_classes_for_frame(frame)
        rgb_uint8 = np.ascontiguousarray(rgb, dtype=np.uint8)
        request = {
            "frame_id": frame.frame_id if frame is not None else None,
            "source_frame_id": frame.source_frame_id if frame is not None else None,
            "image": {
                "shape": list(rgb_uint8.shape),
                "dtype": "uint8",
                "encoding": "base64",
                "data": base64.b64encode(rgb_uint8.tobytes()).decode("ascii"),
            },
            "classes": selected_classes,
            "max_prompts_per_frame": len(selected_classes),
            "max_proposals": self.max_proposals,
            "confidence_threshold": self.confidence_threshold,
            "prompt_selection": {
                "candidate_strategy": self.candidate_strategy,
                "full_class_count": len(self.classes),
                "selected_class_count": len(selected_classes),
                "selected_classes": list(selected_classes),
                "max_prompts_per_frame": self.max_prompts_per_frame,
            },
        }
        self.last_generation_info = {
            **self.last_generation_info,
            "full_class_count": len(self.classes),
            "selected_class_count": len(selected_classes),
            "selected_classes": list(selected_classes),
        }
        response = self.worker_client.request(request)
        if response.get("ok") is False:
            error = str(response.get("error", "worker reported failure"))
            raise RuntimeError(f"SAM3 concept worker reported failure: {error}")
        if response.get("error"):
            raise RuntimeError(f"SAM3 concept worker error: {response['error']}")
        return response

    def _select_classes_for_frame(self, frame: Frame | None) -> list[str]:
        classes = self._dedupe_preserve_order(self.classes)
        if not classes:
            return []
        budget = int(self.max_prompts_per_frame)
        if budget <= 0 or budget >= len(classes):
            return classes

        always = [
            label
            for label in self._dedupe_preserve_order(self.always_include_classes)
            if label in classes
        ]
        selected = always[:budget]
        remaining_budget = budget - len(selected)
        if remaining_budget <= 0:
            return selected

        pool = [label for label in classes if label not in selected]
        if not pool:
            return selected
        if self.candidate_strategy in {"all", "fixed_subset"}:
            return selected + pool[:remaining_budget]

        frame_index = 0
        if frame is not None:
            frame_index = int(frame.frame_id)
        offset = frame_index % len(pool)
        rotated = pool[offset:] + pool[:offset]
        return selected + rotated[:remaining_budget]

    def _mock_proposals(self, rgb: np.ndarray, frame: Frame | None) -> list[Proposal2D]:
        height, width = rgb.shape[:2]
        labels = self._select_classes_for_frame(frame) or ["object"]
        proposals: list[Proposal2D] = []
        limit = min(3, self.max_proposals, len(labels))
        for index, label in enumerate(labels[:limit]):
            x1 = min(width - 1, max(0, int(width * (0.12 + 0.24 * index))))
            y1 = min(height - 1, max(0, int(height * (0.12 + 0.18 * index))))
            x2 = min(width, max(x1 + 1, x1 + max(1, width // 4)))
            y2 = min(height, max(y1 + 1, y1 + max(1, height // 4)))
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            confidence = max(self.confidence_threshold, 0.9 - 0.05 * index)
            proposals.append(self._make_proposal(index, mask, confidence, label, {"mode": "mock"}))
        self.last_generation_info = {
            **self.last_generation_info,
            "full_class_count": len(self.classes),
            "selected_class_count": len(labels),
            "selected_classes": list(labels),
            "raw_mask_count": len(proposals),
            "kept_mask_count": len(proposals),
        }
        return proposals

    def _normalize_response(
        self,
        response: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> list[Proposal2D]:
        raw_items = response.get("proposals", response.get("masks", []))
        if not isinstance(raw_items, list):
            return []

        proposals: list[Proposal2D] = []
        for item_index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            mask = self._mask_from_item(item, image_shape)
            if mask is None:
                continue
            area = int(mask.sum())
            if area <= 0:
                continue
            confidence = self._coerce_float(
                item.get("confidence", item.get("score", item.get("anchor_confidence", 1.0))),
                default=1.0,
            )
            if confidence < self.confidence_threshold:
                continue
            label = self._label_from_item(item)
            metadata = item.get("metadata", {})
            sam3_metadata = dict(metadata) if isinstance(metadata, dict) else {}
            if "anchor_id" in item:
                sam3_metadata.setdefault("anchor_id", item["anchor_id"])
            proposals.append(
                self._make_proposal(
                    item_index,
                    mask,
                    confidence,
                    label,
                    sam3_metadata,
                )
            )

        proposals.sort(key=lambda proposal: (proposal.confidence, proposal.area), reverse=True)
        proposals = proposals[: self.max_proposals]
        for proposal_id, proposal in enumerate(proposals):
            proposal.proposal_id = proposal_id
            proposal.metadata["anchor_id"] = proposal_id
        self.last_generation_info = {
            **self.last_generation_info,
            "raw_mask_count": int(response.get("raw_mask_count", len(raw_items))),
            "worker_prompt_count": int(response.get("prompt_count", 0) or 0),
            "kept_mask_count": len(proposals),
        }
        return proposals

    def _make_proposal(
        self,
        proposal_id: int,
        mask: np.ndarray,
        confidence: float,
        label: str,
        sam3_metadata: dict[str, Any],
        anchor_id: Any | None = None,
    ) -> Proposal2D:
        label = str(label).strip() or "object"
        confidence = float(confidence)
        anchor_id_value = self._coerce_int(anchor_id, default=proposal_id)
        metadata = {
            "anchor_id": anchor_id_value,
            "anchor_class_name": label,
            "anchor_confidence": confidence,
            "anchor_label_strength": "strong",
            "semantic_commit_allowed": True,
            "anchor_label_votes": {label: confidence},
            "source": "sam3_concept",
            "backend": "sam3_concept",
            "mask_source": "sam3_concept",
            "observation_layer": "fine",
            "sam3_metadata": dict(sam3_metadata),
        }
        return Proposal2D(
            proposal_id=proposal_id,
            mask=mask,
            bbox_xyxy=self._mask_to_bbox_xyxy(mask),
            area=int(mask.sum()),
            confidence=confidence,
            backend_name="sam3_concept",
            metadata=metadata,
        )

    def _mask_from_item(
        self,
        item: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        mask_value = item.get("mask", item.get("segmentation", item.get("binary_mask")))
        if mask_value is not None:
            mask = np.asarray(mask_value, dtype=bool)
            if mask.shape == image_shape:
                return mask

        if item.get("mask_encoding") == "base64":
            shape = item.get("shape", image_shape)
            try:
                data = base64.b64decode(str(item.get("data", "")))
                mask = np.frombuffer(data, dtype=np.uint8).reshape(tuple(shape)).astype(bool)
            except Exception:
                return None
            if mask.shape == image_shape:
                return mask

        bbox = item.get("bbox_xyxy", item.get("bbox"))
        if bbox is None:
            return None
        try:
            coords = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None
        if len(coords) != 4:
            return None
        x1, y1, x2, y2 = coords
        if "bbox_xyxy" not in item and (x2 <= x1 or y2 <= y1):
            x2 = x1 + max(0.0, x2)
            y2 = y1 + max(0.0, y2)
        height, width = image_shape
        ix1 = min(width, max(0, int(np.floor(x1))))
        iy1 = min(height, max(0, int(np.floor(y1))))
        ix2 = min(width, max(ix1, int(np.ceil(x2))))
        iy2 = min(height, max(iy1, int(np.ceil(y2))))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        mask = np.zeros(image_shape, dtype=bool)
        mask[iy1:iy2, ix1:ix2] = True
        return mask

    def _label_from_item(self, item: dict[str, Any]) -> str:
        for key in ("class_name", "label", "anchor_class_name", "concept"):
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return self.classes[0] if self.classes else "object"

    @staticmethod
    def _normalize_classes(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        return [str(label).strip() for label in value if str(label).strip()]

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            label = str(value).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            deduped.append(label)
        return deduped

    @staticmethod
    def _optional_config_str(config: dict[str, Any], key: str) -> str | None:
        value = config.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_path(cls, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = cls._repo_root() / path
        return path.resolve()

    @staticmethod
    def _mask_to_bbox_xyxy(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            if isinstance(value, bool):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default
