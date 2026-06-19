"""SED/SED++ semantic-segmentation proposal backend.

This backend converts class-aware semantic segmentation into the existing
``Proposal2D`` stream by extracting connected components per canonical class.
The postprocess path is intentionally pure and testable without Detectron2 or
SED weights.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.core.data_structures import Frame, Proposal2D
from src.models.proposal_backend import ProposalBackend


DEFAULT_CANONICAL_MAP = {
    "couch": "sofa",
    "potted plant": "indoor-plant",
    "plant": "indoor-plant",
    "windowpane": "window",
    "window-blind": "blinds",
    "floor-wood": "floor",
    "floor-tile": "floor",
    "wall-other": "wall",
    "dining table": "table",
    "coffee table": "table",
}


def _normalize_label(label: Any) -> str:
    return str(label).strip().lower().replace("_", "-")


def _as_numpy_sem_seg(sem_seg: Any) -> np.ndarray:
    if hasattr(sem_seg, "detach"):
        sem_seg = sem_seg.detach().cpu().numpy()
    arr = np.asarray(sem_seg, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"sem_seg must have shape (C,H,W) or (H,W,C), got {arr.shape}")
    return arr


def _looks_like_probabilities(arr: np.ndarray) -> bool:
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return False
    return float(arr.min()) >= -1e-6 and float(arr.max()) <= 1.0 + 1e-6


def _softmax_channel_first(logits: np.ndarray) -> np.ndarray:
    if logits.shape[0] == 1:
        return 1.0 / (1.0 + np.exp(-logits))
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=0, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _ensure_channel_first(arr: np.ndarray, class_count: int) -> np.ndarray:
    if arr.shape[0] == class_count:
        return arr
    if arr.shape[-1] == class_count:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(
        f"sem_seg class dimension does not match class_names: sem_seg={arr.shape}, classes={class_count}"
    )


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    try:
        import cv2  # type: ignore

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return [labels == idx for idx in range(1, int(count))]
    except Exception:
        pass

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        component = np.zeros_like(mask, dtype=bool)
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        seen[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            component[cy, cx] = True
            for dx, dy in neighbors:
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        components.append(component)
    return components


def _depth_split_components(
    component: np.ndarray,
    depth: np.ndarray | None,
    *,
    enabled: bool,
    max_depth_gap: float,
) -> list[np.ndarray]:
    if not enabled or depth is None or max_depth_gap <= 0.0:
        return [component]
    valid_depth = np.asarray(depth)
    if valid_depth.shape != component.shape:
        return [component]
    values = valid_depth[component]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        return [component]
    # Conservative split by depth bands. Spatial components are recomputed inside
    # each band so disconnected regions do not merge.
    bins = np.floor(valid_depth / float(max_depth_gap)).astype(np.int32)
    out: list[np.ndarray] = []
    for bin_id in np.unique(bins[component]):
        band = component & (bins == int(bin_id))
        out.extend(_connected_components(band))
    return out or [component]


def semantic_components_to_proposals(
    sem_seg: Any,
    class_names: Sequence[str],
    *,
    canonical_map: Mapping[str, str] | None = None,
    depth: np.ndarray | None = None,
    sem_seg_output: str = "logits",
    min_pixel_confidence: float = 0.35,
    strong_confidence_threshold: float = 0.50,
    structural_confidence_threshold: float = 0.60,
    structure_classes: Sequence[str] | None = None,
    min_component_area: int = 100,
    max_proposals: int | None = 80,
    depth_split_enabled: bool = False,
    depth_split_max_gap: float = 0.25,
) -> list[Proposal2D]:
    """Convert semantic logits/probabilities into class-aware proposals."""
    names = [_normalize_label(name) for name in class_names]
    if not names:
        return []
    raw_map = {**DEFAULT_CANONICAL_MAP, **dict(canonical_map or {})}
    canonical = {_normalize_label(k): _normalize_label(v) for k, v in raw_map.items()}

    arr = _ensure_channel_first(_as_numpy_sem_seg(sem_seg), len(names))
    output_mode = str(sem_seg_output or "logits").strip().lower()
    if output_mode == "probabilities":
        probs = arr
    elif output_mode == "auto":
        probs = arr if _looks_like_probabilities(arr) else _softmax_channel_first(arr)
    elif output_mode == "logits":
        probs = _softmax_channel_first(arr)
    else:
        raise ValueError("sem_seg_output must be one of: logits, probabilities, auto")
    probs = np.asarray(probs, dtype=np.float32)
    winners = np.argmax(probs, axis=0)
    confidence = np.max(probs, axis=0)
    structural_labels = {_normalize_label(value) for value in (structure_classes or [])}

    candidates: list[tuple[float, int, Proposal2D]] = []
    for class_index, sedpp_class_name in enumerate(names):
        canonical_name = canonical.get(sedpp_class_name, sedpp_class_name)
        if not canonical_name:
            continue
        class_mask = (winners == class_index) & (confidence >= float(min_pixel_confidence))
        for component in _connected_components(class_mask):
            for split in _depth_split_components(
                component,
                depth,
                enabled=bool(depth_split_enabled),
                max_depth_gap=float(depth_split_max_gap),
            ):
                area = int(split.sum())
                if area < int(min_component_area):
                    continue
                score = float(np.mean(confidence[split]))
                threshold = (
                    float(structural_confidence_threshold)
                    if canonical_name in structural_labels
                    else float(strong_confidence_threshold)
                )
                semantic_commit_allowed = bool(score >= threshold)
                label_strength = "strong" if semantic_commit_allowed else "contextual"
                metadata = {
                    "source": "sedpp_semantic_component",
                    "mask_source": "sedpp_semantic_component",
                    "anchor_class_name": canonical_name,
                    "anchor_confidence": score,
                    "anchor_label_votes": {canonical_name: score},
                    "anchor_label_strength": label_strength,
                    "semantic_commit_allowed": semantic_commit_allowed,
                    "sedpp_commit_threshold": threshold,
                    "sedpp_class_index": int(class_index),
                    "sedpp_class_name": sedpp_class_name,
                }
                proposal = Proposal2D(
                    proposal_id=-1,
                    mask=split.astype(bool, copy=True),
                    bbox_xyxy=_bbox_from_mask(split),
                    area=area,
                    confidence=score,
                    backend_name="sedpp",
                    metadata=metadata,
                )
                candidates.append((score, area, proposal))

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2].metadata["sedpp_class_index"]))
    if max_proposals is not None and int(max_proposals) > 0:
        candidates = candidates[: int(max_proposals)]
    proposals = [proposal for _, _, proposal in candidates]
    for proposal_id, proposal in enumerate(proposals):
        proposal.proposal_id = int(proposal_id)
    return proposals


def load_sedpp_class_config(path: str | Path) -> tuple[list[str], dict[str, str]]:
    """Load class names and optional aliases/canonical map from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    aliases: dict[str, str] = {}
    if isinstance(data, list):
        classes = data
    elif isinstance(data, dict):
        classes = data.get("classes") or data.get("class_names") or data.get("labels")
        aliases.update(data.get("aliases") or {})
        aliases.update(data.get("canonical_map") or {})
    else:
        raise ValueError(f"Unsupported SED++ class JSON format in {path}")
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError(f"SED++ class JSON must provide a list of class names: {path}")
    return list(classes), aliases


def write_sed_class_list_json(class_names: Sequence[str]) -> str:
    """Write the plain class-list JSON expected by the SED repository."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="sedpp_classes_",
        delete=False,
    )
    with handle:
        json.dump(list(class_names), handle)
    return str(handle.name)


OPENCLIP_LOCAL_WEIGHT_DEFAULTS = {
    ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup"): (
        "weights/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin"
    ),
    ("convnext_base_w_320", "laion_aesthetic_s13b_b82k"): (
        "weights/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K/open_clip_pytorch_model.bin"
    ),
}


def _resolve_existing_path(raw_path: Any, *, repo_path: Path, project_root: Path) -> str:
    if not raw_path:
        return ""
    candidate = Path(str(raw_path)).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [repo_path / candidate, project_root / candidate]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def resolve_openclip_weight_path(
    config: Mapping[str, Any],
    *,
    repo_path: Path,
    project_root: Path,
    model_name: str,
    pretrained_tag: str,
) -> str:
    """Resolve local OpenCLIP checkpoint paths before SED falls back to HF downloads."""
    explicit = _resolve_existing_path(
        config.get("openclip_weight_path"),
        repo_path=repo_path,
        project_root=project_root,
    )
    if explicit:
        return explicit

    configured_paths = config.get("openclip_weight_paths") or {}
    if isinstance(configured_paths, Mapping):
        lookup_keys = (
            f"{model_name}:{pretrained_tag}",
            model_name,
            pretrained_tag,
        )
        for key in lookup_keys:
            resolved = _resolve_existing_path(
                configured_paths.get(key),
                repo_path=repo_path,
                project_root=project_root,
            )
            if resolved:
                return resolved

    default_relpath = OPENCLIP_LOCAL_WEIGHT_DEFAULTS.get(
        (str(model_name).lower(), str(pretrained_tag).lower())
    )
    return _resolve_existing_path(default_relpath, repo_path=repo_path, project_root=project_root)


def install_openclip_local_weight_patch(
    open_clip_module: Any,
    *,
    model_name: str,
    pretrained_tag: str,
    weight_path: str,
) -> None:
    """Patch OpenCLIP calls used by SED so hard-coded HF aliases load local weights."""
    if not weight_path:
        return
    original = getattr(open_clip_module, "_sedpp_original_create_model_and_transforms", None)
    if original is None:
        original = open_clip_module.create_model_and_transforms
        setattr(open_clip_module, "_sedpp_original_create_model_and_transforms", original)

    alias_map = dict(getattr(open_clip_module, "_sedpp_local_weight_aliases", {}))
    alias_map[(str(model_name).lower(), str(pretrained_tag).lower())] = str(weight_path)
    setattr(open_clip_module, "_sedpp_local_weight_aliases", alias_map)

    def create_model_and_transforms(model_name_arg, pretrained=None, *args, **kwargs):
        local_weight = alias_map.get((str(model_name_arg).lower(), str(pretrained).lower()))
        if local_weight and Path(local_weight).exists():
            pretrained = local_weight
        return original(model_name_arg, pretrained=pretrained, *args, **kwargs)

    open_clip_module.create_model_and_transforms = create_model_and_transforms


def configure_hf_endpoint(config: Mapping[str, Any]) -> str:
    endpoint = str(config.get("hf_endpoint") or "").strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        return endpoint
    if bool(config.get("use_hf_mirror", True)) and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    return os.environ.get("HF_ENDPOINT", "")


class SEDPPProposalBackend(ProposalBackend):
    """Proposal backend backed by SED semantic segmentation."""

    def __init__(self) -> None:
        self.predictor: Any = None
        self.class_names: list[str] = []
        self.canonical_map: dict[str, str] = {}
        self.min_pixel_confidence = 0.35
        self.sem_seg_output = "logits"
        self.strong_confidence_threshold = 0.50
        self.structural_confidence_threshold = 0.60
        self.structure_classes = ["wall", "floor", "ceiling", "door", "window", "blinds"]
        self.min_component_area = 100
        self.max_proposals = 80
        self.depth_split_enabled = False
        self.depth_split_max_gap = 0.25
        self.last_generation_timings: dict[str, float] = {}
        self.last_debug: dict[str, Any] = {}
        self._sed_class_json_path = ""

    def initialize(self, config: dict) -> None:
        self.min_pixel_confidence = float(config.get("min_pixel_confidence", 0.35))
        self.sem_seg_output = str(config.get("sem_seg_output", "logits"))
        self.strong_confidence_threshold = float(config.get("strong_confidence_threshold", 0.50))
        self.structural_confidence_threshold = float(config.get("structural_confidence_threshold", 0.60))
        self.structure_classes = [
            _normalize_label(value)
            for value in config.get(
                "structure_classes",
                ["wall", "floor", "ceiling", "door", "window", "blinds"],
            )
        ]
        self.min_component_area = int(config.get("min_component_area", 100))
        self.max_proposals = int(config.get("max_proposals", 80))
        self.depth_split_enabled = bool(config.get("depth_split_enabled", False))
        self.depth_split_max_gap = float(config.get("depth_split_max_gap", 0.25))

        class_json = config.get("class_json")
        if not class_json:
            raise ValueError("SED++ backend requires proposal.sedpp.class_json")
        class_json_path = Path(class_json).expanduser()
        if not class_json_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            class_json_path = project_root / class_json_path
        self.class_names, file_aliases = load_sedpp_class_config(class_json_path)
        self._sed_class_json_path = write_sed_class_list_json(self.class_names)
        self.canonical_map = {
            **DEFAULT_CANONICAL_MAP,
            **file_aliases,
            **dict(config.get("canonical_map") or {}),
        }

        repo_root = config.get("repo_root")
        config_path = config.get("config_path")
        checkpoint_path = config.get("checkpoint_path")
        if not repo_root or not config_path or not checkpoint_path:
            raise ValueError(
                "SED++ backend requires repo_root, config_path, checkpoint_path, and class_json"
            )

        repo_path = Path(repo_root).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"SED++ repo_root does not exist: {repo_path}")
        for required_path, label in [(config_path, "config_path"), (checkpoint_path, "checkpoint_path")]:
            if not Path(required_path).expanduser().exists():
                raise FileNotFoundError(f"SED++ {label} does not exist: {required_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            import open_clip
            from detectron2.config import get_cfg
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.projects.deeplab import add_deeplab_config
            from sed import add_sed_config
        except Exception as exc:
            raise ImportError(
                "Failed to import Detectron2/SED for SED++ backend. "
                f"Check repo_root={repo_path} and environment dependencies. Original error: {exc}"
            ) from exc

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_sed_config(cfg)
        cfg.merge_from_file(str(config_path))
        cfg.defrost()
        hf_endpoint = configure_hf_endpoint(config)
        project_root = Path(__file__).resolve().parents[2]
        openclip_model_name = str(cfg.MODEL.ENC.CLIP_MODEL_NAME)
        openclip_pretrained_tag = str(cfg.MODEL.ENC.CLIP_PRETRAINED_WEIGHTS)
        openclip_weight_path = resolve_openclip_weight_path(
            config,
            repo_path=repo_path,
            project_root=project_root,
            model_name=openclip_model_name,
            pretrained_tag=openclip_pretrained_tag,
        )
        if openclip_weight_path:
            install_openclip_local_weight_patch(
                open_clip,
                model_name=openclip_model_name,
                pretrained_tag=openclip_pretrained_tag,
                weight_path=openclip_weight_path,
            )
            cfg.MODEL.ENC.CLIP_PRETRAINED_WEIGHTS = openclip_weight_path
        cfg.MODEL.WEIGHTS = str(checkpoint_path)
        cfg.MODEL.DEVICE = str(config.get("device", "cuda"))
        cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = self._sed_class_json_path
        cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = self._sed_class_json_path
        cfg.TEST.FAST_INFERENCE = bool(config.get("fast_inference", True))
        cfg.TEST.TOPK = int(config.get("topk", 8))
        cfg.freeze()

        try:
            self.predictor = DefaultPredictor(cfg)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize SED++ DefaultPredictor. "
                f"config_path={config_path}, checkpoint_path={checkpoint_path}, device={cfg.MODEL.DEVICE}. "
                f"openclip_weight_path={openclip_weight_path or '<download>'}, hf_endpoint={hf_endpoint or '<default HF>'}. "
                f"Original error: {exc}"
            ) from exc
        self.last_debug = {
            "class_count": len(self.class_names),
            "sed_class_json_path": self._sed_class_json_path,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "openclip_weight_path": openclip_weight_path,
            "hf_endpoint": hf_endpoint,
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> list[Proposal2D]:
        if self.predictor is None:
            raise RuntimeError("SED++ backend is not initialized")
        self.last_generation_timings = {}
        start = time.perf_counter()
        predictions = self.predictor(np.asarray(rgb)[:, :, ::-1])
        inference_elapsed = float(time.perf_counter() - start)
        if "sem_seg" not in predictions:
            raise RuntimeError("SED++ predictor output did not contain 'sem_seg'")

        post_start = time.perf_counter()
        proposals = semantic_components_to_proposals(
            predictions["sem_seg"],
            self.class_names,
            canonical_map=self.canonical_map,
            depth=depth,
            sem_seg_output=self.sem_seg_output,
            min_pixel_confidence=self.min_pixel_confidence,
            strong_confidence_threshold=self.strong_confidence_threshold,
            structural_confidence_threshold=self.structural_confidence_threshold,
            structure_classes=self.structure_classes,
            min_component_area=self.min_component_area,
            max_proposals=self.max_proposals,
            depth_split_enabled=self.depth_split_enabled,
            depth_split_max_gap=self.depth_split_max_gap,
        )
        post_elapsed = float(time.perf_counter() - post_start)
        self.last_generation_timings = {
            "sedpp_inference": inference_elapsed,
            "sedpp_postprocess": post_elapsed,
        }
        self.last_debug = {
            **self.last_debug,
            "raw_class_count": len(self.class_names),
            "kept_component_count": len(proposals),
            "inference_time": inference_elapsed,
            "postprocess_time": post_elapsed,
            "frame_id": None if frame is None else int(frame.frame_id),
        }
        return proposals
