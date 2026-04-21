"""E-SAM proposal backend wrapper with configurable local integration.

This adapter keeps E-SAM details behind the ProposalBackend interface,
normalizes outputs into Proposal2D, and surfaces explicit setup failures.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List
import sys

import numpy as np

from src.core.data_structures import Frame, Proposal2D
from src.models.proposal_backend import ProposalBackend


class ESAMProposalBackend(ProposalBackend):
    """Configurable E-SAM proposal backend.

    The local /home/phl/vv/E-SAM checkout currently appears to be a project
    page repository without an obvious Python inference entrypoint. This
    backend therefore focuses on:

    - clean path/config plumbing for local E-SAM code
    - a configurable predictor/factory hook
    - robust output normalization into Proposal2D
    - actionable errors when integration assets are missing
    """

    def __init__(self) -> None:
        self.device = "cuda"
        self.max_proposals = 50
        self.confidence_threshold = 0.0
        self.call_mode = "auto"
        self.bbox_format = "auto"

        self.repo_root: str | None = None
        self.config_path: str | None = None
        self.checkpoint_path: str | None = None
        self.predictor: Any | None = None
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        self.device = str(config.get("device", "cuda"))
        self.max_proposals = int(config.get("max_proposals", 50))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
        self.call_mode = str(config.get("call_mode", "auto")).lower()
        self.bbox_format = str(config.get("bbox_format", "auto")).lower()

        repo_root_value = config.get("repo_root", "/home/phl/vv/E-SAM")
        self.repo_root = self._resolve_optional_path(repo_root_value, allow_missing=False)
        repo_root_path = Path(self.repo_root)
        if not repo_root_path.is_dir():
            raise NotADirectoryError(f"E-SAM repo_root is not a directory: {repo_root_path}")

        extra_python_paths = config.get("python_paths", [])
        if isinstance(extra_python_paths, (str, Path)):
            extra_python_paths = [extra_python_paths]
        python_paths = [repo_root_path] + [Path(path) for path in extra_python_paths]
        for python_path in python_paths:
            resolved_path = str(Path(python_path).expanduser().resolve())
            if resolved_path not in sys.path:
                sys.path.insert(0, resolved_path)

        self.config_path = self._resolve_optional_path(config.get("config_path"), allow_missing=True)
        self.checkpoint_path = self._resolve_optional_path(config.get("checkpoint_path"), allow_missing=True)

        factory_spec = str(config.get("factory", "")).strip()
        predictor_spec = str(config.get("predictor", "")).strip()
        predictor_is_factory = bool(config.get("predictor_is_factory", False))

        loader = ""
        if factory_spec:
            self.predictor = self._build_from_factory(factory_spec, config)
            loader = "factory"
        elif predictor_spec:
            predictor_or_factory = self._import_symbol(predictor_spec)
            if predictor_is_factory:
                self.predictor = predictor_or_factory(config)
                loader = "predictor_factory"
            else:
                self.predictor = predictor_or_factory
                loader = "predictor_symbol"
        else:
            self._raise_missing_entrypoint(repo_root_path)

        if not callable(self.predictor):
            raise TypeError(
                f"Configured E-SAM predictor is not callable: {type(self.predictor)!r}."
            )

        self.last_generation_info = {
            "backend": "esam",
            "loader": loader,
            "device": self.device,
            "repo_root": self.repo_root,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "call_mode": self.call_mode,
            "bbox_format": self.bbox_format,
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        if self.predictor is None:
            raise RuntimeError("ESAMProposalBackend.initialize() must be called before inference.")

        outputs = self._invoke_predictor(rgb, depth)
        proposals = self._normalize_outputs(outputs)

        proposals = [proposal for proposal in proposals if proposal.confidence >= self.confidence_threshold]
        proposals.sort(key=lambda proposal: (proposal.confidence, proposal.area), reverse=True)
        proposals = proposals[: self.max_proposals]

        for proposal_id, proposal in enumerate(proposals):
            proposal.proposal_id = proposal_id

        self.last_generation_info = {
            **self.last_generation_info,
            "raw_mask_count": len(self._as_list_like(outputs)),
            "kept_mask_count": len(proposals),
        }
        return proposals

    def _invoke_predictor(self, rgb: np.ndarray, depth: np.ndarray) -> Any:
        if self.predictor is None:
            raise RuntimeError("ESAM predictor is not initialized.")

        rgb_for_infer = np.array(rgb, copy=True)
        depth_for_infer = np.array(depth, copy=True)

        if self.call_mode == "rgb":
            return self.predictor(rgb_for_infer)
        if self.call_mode == "rgbd":
            return self.predictor(rgb_for_infer, depth_for_infer)
        if self.call_mode == "kwargs":
            return self.predictor(rgb=rgb_for_infer, depth=depth_for_infer)
        if self.call_mode != "auto":
            raise ValueError(
                f"Unsupported proposal.esam.call_mode='{self.call_mode}'. "
                "Expected one of: auto, rgb, rgbd, kwargs."
            )

        attempts = [
            ("predictor(rgb=..., depth=...)", lambda: self.predictor(rgb=rgb_for_infer, depth=depth_for_infer)),
            ("predictor(image=..., depth=...)", lambda: self.predictor(image=rgb_for_infer, depth=depth_for_infer)),
            ("predictor(rgb=...)", lambda: self.predictor(rgb=rgb_for_infer)),
            ("predictor(image=...)", lambda: self.predictor(image=rgb_for_infer)),
            ("predictor(rgb, depth)", lambda: self.predictor(rgb_for_infer, depth_for_infer)),
            ("predictor(rgb)", lambda: self.predictor(rgb_for_infer)),
        ]

        type_errors: list[str] = []
        for label, invoke in attempts:
            try:
                return invoke()
            except TypeError as exc:
                type_errors.append(f"- {label}: {exc}")

        raise RuntimeError(
            "Unable to invoke E-SAM predictor in auto mode. "
            "Set proposal.esam.call_mode to match the predictor signature.\n"
            + "\n".join(type_errors[-3:])
        )

    def _normalize_outputs(self, outputs: Any) -> List[Proposal2D]:
        if isinstance(outputs, dict):
            if "annotations" in outputs:
                return self._normalize_iterable(outputs["annotations"])
            if "proposals" in outputs:
                return self._normalize_iterable(outputs["proposals"])
            if "masks" in outputs:
                return self._normalize_mask_stack(
                    outputs["masks"],
                    outputs.get("scores"),
                    outputs.get("boxes"),
                )
            if any(key in outputs for key in ("segmentation", "mask", "binary_mask")):
                return self._normalize_iterable([outputs])

        if isinstance(outputs, np.ndarray):
            if outputs.ndim == 2:
                return self._normalize_mask_stack(outputs[np.newaxis, ...], None, None)
            if outputs.ndim == 3:
                return self._normalize_mask_stack(outputs, None, None)

        if isinstance(outputs, (list, tuple)):
            return self._normalize_iterable(outputs)

        raise RuntimeError(
            "Unsupported E-SAM output format. "
            "Expected dict/list/tuple/mask ndarray with segmentation proposals."
        )

    def _normalize_mask_stack(
        self,
        masks: Any,
        scores: Any,
        boxes: Any,
    ) -> List[Proposal2D]:
        masks_np = np.asarray(masks)
        if masks_np.ndim != 3:
            raise RuntimeError("E-SAM masks output must be a (N, H, W) array.")

        proposals: List[Proposal2D] = []
        scores_np = np.asarray(scores) if scores is not None else None
        boxes_np = np.asarray(boxes) if boxes is not None else None

        for index in range(masks_np.shape[0]):
            mask = np.asarray(masks_np[index], dtype=bool)
            area = int(mask.sum())
            if area <= 0:
                continue

            bbox_xyxy = self._mask_to_bbox_xyxy(mask)
            if boxes_np is not None and index < len(boxes_np):
                bbox_xyxy = self._coerce_bbox(np.asarray(boxes_np[index], dtype=np.float32), source_key="bbox")

            confidence = 1.0
            if scores_np is not None and index < len(scores_np):
                confidence = float(scores_np[index])

            proposals.append(
                Proposal2D(
                    proposal_id=index,
                    mask=mask,
                    bbox_xyxy=bbox_xyxy,
                    area=area,
                    confidence=confidence,
                    backend_name="esam",
                    metadata={
                        "backend": "esam",
                        "source": "esam",
                        "raw_mask_index": index,
                    },
                )
            )
        return proposals

    def _normalize_iterable(self, items: Iterable[Any]) -> List[Proposal2D]:
        proposals: List[Proposal2D] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(
                    "Unsupported E-SAM iterable output item. Expected dict-like entries."
                )

            segmentation = (
                item.get("segmentation")
                if "segmentation" in item
                else item.get("mask", item.get("binary_mask"))
            )
            if segmentation is None:
                raise RuntimeError("E-SAM output item has no segmentation/mask field.")

            mask = np.asarray(segmentation, dtype=bool)
            if mask.ndim != 2:
                raise RuntimeError("E-SAM segmentation mask must be a 2D array.")

            area = int(mask.sum())
            if area <= 0:
                continue

            bbox_xyxy = self._mask_to_bbox_xyxy(mask)
            if "bbox_xyxy" in item and item["bbox_xyxy"] is not None:
                bbox_xyxy = self._coerce_bbox(np.asarray(item["bbox_xyxy"], dtype=np.float32), source_key="bbox_xyxy")
            elif "bbox" in item and item["bbox"] is not None:
                bbox_xyxy = self._coerce_bbox(np.asarray(item["bbox"], dtype=np.float32), source_key="bbox")

            confidence = float(
                item.get(
                    "score",
                    item.get(
                        "confidence",
                        item.get("predicted_iou", item.get("stability_score", 1.0)),
                    ),
                )
            )

            metadata = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "segmentation",
                    "mask",
                    "binary_mask",
                    "bbox",
                    "bbox_xyxy",
                    "score",
                    "confidence",
                    "predicted_iou",
                    "stability_score",
                }
            }

            proposals.append(
                Proposal2D(
                    proposal_id=index,
                    mask=mask,
                    bbox_xyxy=bbox_xyxy,
                    area=area,
                    confidence=confidence,
                    backend_name="esam",
                    metadata={"backend": "esam", "source": "esam", **metadata},
                )
            )

        return proposals

    def _coerce_bbox(self, bbox: np.ndarray, source_key: str) -> np.ndarray:
        if bbox.shape[0] != 4:
            raise RuntimeError("E-SAM bbox must have four coordinates.")

        if source_key == "bbox_xyxy":
            return bbox.astype(np.float32)

        if self.bbox_format == "xyxy":
            return bbox.astype(np.float32)
        if self.bbox_format == "xywh":
            x, y, w, h = [float(value) for value in bbox.tolist()]
            return np.array([x, y, x + w, y + h], dtype=np.float32)

        # auto mode: SAM-like outputs often use xywh under key "bbox".
        x, y, w_or_x2, h_or_y2 = [float(value) for value in bbox.tolist()]
        if source_key == "bbox":
            return np.array([x, y, x + w_or_x2, y + h_or_y2], dtype=np.float32)
        return np.array([x, y, w_or_x2, h_or_y2], dtype=np.float32)

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
        if ":" not in spec:
            raise ValueError(
                f"Invalid symbol spec '{spec}'. Expected format 'module:function'."
            )
        module_name, symbol_name = spec.split(":", 1)
        module = import_module(module_name)
        return getattr(module, symbol_name)

    def _build_from_factory(self, factory_spec: str, config: Dict[str, Any]) -> Any:
        factory = self._import_symbol(factory_spec)
        predictor = factory(config)
        if not callable(predictor):
            raise TypeError(
                f"E-SAM factory '{factory_spec}' returned non-callable predictor: {type(predictor)!r}"
            )
        return predictor

    def _raise_missing_entrypoint(self, repo_root: Path) -> None:
        python_file_count = sum(1 for _ in repo_root.rglob("*.py"))
        has_index_html = (repo_root / "index.html").exists()
        repo_hint = (
            "The configured repo_root looks like a project-page repo (index.html exists) "
            "and no Python inference entrypoint was auto-detected. "
            if has_index_html and python_file_count == 0
            else ""
        )
        raise RuntimeError(
            "E-SAM backend requires an explicit inference entrypoint. "
            "Set proposal.esam.factory or proposal.esam.predictor (module:function), "
            "and optionally proposal.esam.call_mode. "
            f"repo_root={repo_root} contains {python_file_count} Python files. "
            + repo_hint
            + "Example: proposal.esam.factory='my_esam_adapter:build_predictor'."
        )

    @staticmethod
    def _resolve_optional_path(value: Any, allow_missing: bool) -> str | None:
        if value in (None, "", "null"):
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / path
        resolved = path.resolve()
        if not allow_missing and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        return str(resolved)

    @staticmethod
    def _as_list_like(outputs: Any) -> List[Any]:
        if isinstance(outputs, dict):
            if "annotations" in outputs and isinstance(outputs["annotations"], list):
                return outputs["annotations"]
            if "proposals" in outputs and isinstance(outputs["proposals"], list):
                return outputs["proposals"]
            if "masks" in outputs:
                masks = np.asarray(outputs["masks"])
                if masks.ndim == 3:
                    return [None] * int(masks.shape[0])
                if masks.ndim == 2:
                    return [None]
            if any(key in outputs for key in ("segmentation", "mask", "binary_mask")):
                return [outputs]
            return []

        if isinstance(outputs, np.ndarray):
            if outputs.ndim == 3:
                return [None] * int(outputs.shape[0])
            if outputs.ndim == 2:
                return [None]
            return []

        if isinstance(outputs, (list, tuple)):
            return list(outputs)
        return []
