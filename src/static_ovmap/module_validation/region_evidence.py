"""Leakage-safe target manifests and frozen region-model boundaries."""

from __future__ import annotations

import hashlib
import inspect
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .native_capture import FrameObservation, RegionRequest

WOW_PROMPT = (
    "This is the original image: <image>. Please classify the specified target "
    "area. Reply with only the category name of the target area, without "
    "explanation. Specified target area:<image>"
)
WOW_GENERATION = MappingProxyType(
    {
        "do_sample": False,
        "max_dynamic_patches": 12,
        "max_new_tokens": 32,
        "num_beams": 1,
        "thumbnail": True,
    }
)
_WOW_MODES = {"official_combined", "combined_with_patch_list"}


def _readonly(value: Any, *, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _unit_rows(value: Any, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError(f"{name} must be a finite matrix")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} rows must have nonzero norm")
    return rows / norms


def stable_target_subset(
    scene_id: str,
    stable_target_ids: Sequence[str],
    *,
    limit: int = 128,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply the protocol's label-free scene/target hash cap."""

    if not scene_id or limit <= 0:
        raise ValueError("scene_id and a positive target limit are required")
    targets = tuple(str(target_id) for target_id in stable_target_ids)
    if any(not target_id for target_id in targets) or len(set(targets)) != len(targets):
        raise ValueError("stable target IDs must be nonempty and unique")
    ordered = tuple(
        sorted(
            targets,
            key=lambda target_id: (
                hashlib.sha256(f"{scene_id}|S|{target_id}".encode()).hexdigest(),
                target_id,
            ),
        )
    )
    return ordered[:limit], ordered[limit:]


@dataclass(frozen=True)
class UnavailableRequest:
    request_id: str
    frame_id: int
    captured_target_id: str
    reconciled_target_id: str | None
    reason: str


@dataclass(frozen=True)
class TargetViewManifest:
    scene_id: str
    selected_targets: tuple[str, ...]
    excluded_targets: tuple[str, ...]
    views: Mapping[str, tuple[RegionRequest, ...]]
    unavailable: tuple[UnavailableRequest, ...]
    duplicate_request_ids: tuple[str, ...]


def build_target_view_manifest(
    scene_id: str,
    stable_target_ids: Sequence[str],
    frames: Sequence[FrameObservation],
    reconciled_targets: Mapping[str, str | None],
    *,
    max_views: int = 3,
    max_targets: int = 128,
) -> TargetViewManifest:
    """Bind requests to final targets only through an explicit lineage decision."""

    if max_views <= 0:
        raise ValueError("max_views must be positive")
    selected, excluded = stable_target_subset(
        scene_id, stable_target_ids, limit=max_targets
    )
    selected_set = set(selected)
    candidates: dict[str, list[RegionRequest]] = {target: [] for target in selected}
    unavailable: list[UnavailableRequest] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    for frame in frames:
        if frame.scene_id != scene_id:
            raise ValueError("frame scene does not match target manifest")
        for request in frame.requests:
            if request.request_id in seen:
                duplicates.append(request.request_id)
                continue
            seen.add(request.request_id)
            if request.request_id not in reconciled_targets:
                unavailable.append(
                    UnavailableRequest(
                        request.request_id,
                        request.frame_id,
                        request.target_id,
                        None,
                        "MISSING_TARGET_RECONCILIATION",
                    )
                )
                continue
            resolved = reconciled_targets[request.request_id]
            if resolved is None:
                unavailable.append(
                    UnavailableRequest(
                        request.request_id,
                        request.frame_id,
                        request.target_id,
                        None,
                        "AMBIGUOUS_OR_INACTIVE_TARGET",
                    )
                )
                continue
            resolved = str(resolved)
            if resolved != request.target_id and resolved not in request.lineage:
                unavailable.append(
                    UnavailableRequest(
                        request.request_id,
                        request.frame_id,
                        request.target_id,
                        resolved,
                        "UNPROVEN_TARGET_LINEAGE",
                    )
                )
                continue
            if resolved not in selected_set:
                reason = (
                    "TARGET_CAP_EXCLUDED" if resolved in set(excluded)
                    else "TARGET_OUTSIDE_FINAL_REGISTRY"
                )
                unavailable.append(
                    UnavailableRequest(
                        request.request_id,
                        request.frame_id,
                        request.target_id,
                        resolved,
                        reason,
                    )
                )
                continue
            candidates[resolved].append(request)

    views = {
        target: tuple(
            sorted(
                requests,
                key=lambda request: (
                    -request.visible_target_pixels,
                    request.frame_id,
                    request.request_id,
                ),
            )[:max_views]
        )
        for target, requests in candidates.items()
    }
    return TargetViewManifest(
        scene_id=scene_id,
        selected_targets=selected,
        excluded_targets=excluded,
        views=MappingProxyType(views),
        unavailable=tuple(unavailable),
        duplicate_request_ids=tuple(duplicates),
    )


@dataclass(frozen=True)
class NativeCropBatch:
    raw: tuple[np.ndarray, ...]
    foreground: tuple[np.ndarray, ...]
    background: tuple[np.ndarray, ...]
    geometries: tuple[tuple[int, int, int, int], ...]
    identities: tuple[str, ...]
    target_pixels: int
    union_pixels: int

    @property
    def legacy_six(self) -> tuple[np.ndarray, ...]:
        return tuple(
            image
            for scale in range(3)
            for image in (self.raw[scale], self.foreground[scale])
        )

    @property
    def all_nine(self) -> tuple[np.ndarray, ...]:
        return (*self.legacy_six, *self.background)


def native_crops(
    rgb_image: Any,
    target_mask: Any,
    union_mask: Any,
    bbox_xyxy: Sequence[int],
    *,
    expansion: float = 0.1,
) -> NativeCropBatch:
    """Reproduce OVI's three raw/foreground crops and add three backgrounds."""

    rgb = np.asarray(rgb_image)
    target = np.asarray(target_mask, dtype=bool)
    union = np.asarray(union_mask, dtype=bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("native RGB must be an HxWx3 uint8 array")
    if target.shape != rgb.shape[:2] or union.shape != target.shape:
        raise ValueError("native target and union masks must align with RGB")
    if not np.any(target) or not np.all(union[target]):
        raise ValueError("native union mask must contain a nonempty target")
    if not np.isfinite(expansion) or expansion < 0.0:
        raise ValueError("crop expansion must be finite and nonnegative")
    bbox = tuple(int(value) for value in bbox_xyxy)
    if len(bbox) != 4:
        raise ValueError("native bbox must contain four values")
    x1, y1, x2, y2 = bbox
    height, width = target.shape
    if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
        raise ValueError("native bbox must be a nondegenerate in-image min/max box")

    foreground_image = np.array(rgb, copy=True)
    foreground_image[~union] = 0
    background_image = np.array(rgb, copy=True)
    background_image[target] = 0
    raw: list[np.ndarray] = []
    foreground: list[np.ndarray] = []
    background: list[np.ndarray] = []
    geometries: list[tuple[int, int, int, int]] = []
    for layer in range(3):
        x_pad = int(expansion * layer * (x2 - x1))
        y_pad = int(expansion * layer * (y2 - y1))
        x_min = max(0, x1 - x_pad)
        y_min = max(0, y1 - y_pad)
        x_max = min(width - 1, x2 + x_pad)
        y_max = min(height - 1, y2 + y_pad)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("native exclusive crop became empty")
        geometry = (x_min, y_min, x_max, y_max)
        geometries.append(geometry)
        raw.append(_readonly(rgb[y_min:y_max, x_min:x_max], dtype=np.uint8))
        foreground.append(
            _readonly(
                foreground_image[y_min:y_max, x_min:x_max], dtype=np.uint8
            )
        )
        background.append(
            _readonly(
                background_image[y_min:y_max, x_min:x_max], dtype=np.uint8
            )
        )

    identities = tuple(
        [
            identity
            for scale in range(3)
            for identity in (f"raw:scale{scale}", f"foreground:scale{scale}")
        ]
        + [f"background:scale{scale}" for scale in range(3)]
    )
    return NativeCropBatch(
        raw=tuple(raw),
        foreground=tuple(foreground),
        background=tuple(background),
        geometries=tuple(geometries),
        identities=identities,
        target_pixels=int(np.count_nonzero(target)),
        union_pixels=int(np.count_nonzero(union)),
    )


class ImageBatchBackend(Protocol):
    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


class FrozenSiglipBackend:
    """Concrete local-only image/text backend for native SigLIP or SigLIP2."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        tokenizer: Any,
        device: str,
    ) -> None:
        if not device:
            raise ValueError("SigLIP device must be nonempty")
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_local(cls, model_path: str, *, device: str = "cuda") -> FrozenSiglipBackend:
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(device)
        processor = AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        return cls(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
        )

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        if not images:
            raise ValueError("SigLIP image batch must not be empty")
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {
            name: value.to(device=self.device, dtype=torch.float32)
            for name, value in inputs.items()
        }
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.float().cpu().numpy()

    def encode_texts(self, values: Sequence[str]) -> np.ndarray:
        import torch

        texts = tuple(str(value) for value in values)
        if not texts or any(not value for value in texts):
            raise ValueError("SigLIP text batch must contain nonempty strings")
        inputs = self.tokenizer(
            list(texts),
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.float().cpu().numpy()


@dataclass(frozen=True)
class EncodedNativeRequest:
    vectors: np.ndarray
    identities: tuple[str, ...]
    legacy_mean: np.ndarray

    def __post_init__(self) -> None:
        vectors = _readonly(self.vectors, dtype=np.float64)
        mean = _readonly(self.legacy_mean, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != 9:
            raise ValueError("native request must contain nine crop vectors")
        if mean.shape != (vectors.shape[1],):
            raise ValueError("native legacy mean must match crop-vector width")
        if len(self.identities) != 9:
            raise ValueError("native crop identities must align with vectors")
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "legacy_mean", mean)


class NativeRegionEncoder:
    """Adds crop-level readout without changing the historical encoder."""

    def __init__(self, backend: ImageBatchBackend) -> None:
        self._backend = backend

    def encode(self, crops: NativeCropBatch) -> EncodedNativeRequest:
        vectors = _unit_rows(
            self._backend.encode_images(crops.all_nine), name="native crop features"
        )
        if vectors.shape[0] != 9:
            raise ValueError("native backend must return one row per crop")
        return EncodedNativeRequest(
            vectors=vectors,
            identities=crops.identities,
            legacy_mean=vectors[:6].mean(axis=0),
        )


class WowRunner(Protocol):
    def prepare_region(
        self,
        image: np.ndarray,
        region_mask: np.ndarray,
        *,
        scale: float,
        image_size: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def generate(
        self,
        *,
        images: tuple[np.ndarray, np.ndarray],
        region_mask_16: np.ndarray,
        prompt: str,
        generation: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _wow_crop_box(mask: Any, resized_size_wh: tuple[int, int], scale: float):
    import torch

    coordinates = torch.nonzero(mask)
    if coordinates.numel() == 0:
        return None
    min_row, max_row = coordinates[:, 0].min(), coordinates[:, 0].max()
    min_col, max_col = coordinates[:, 1].min(), coordinates[:, 1].max()
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2
    box_width = max_col - min_col + 1
    box_height = max_row - min_row + 1
    resized_width, resized_height = resized_size_wh
    new_width = min(max(box_width, box_height) * scale, resized_width)
    new_height = min(max(box_width, box_height) * scale, resized_height)
    new_min_col = max(0, int(center_col - new_width / 2))
    new_max_col = min(resized_width - 1, int(center_col + new_width / 2))
    new_min_row = max(0, int(center_row - new_height / 2))
    new_max_row = min(resized_height - 1, int(center_row + new_height / 2))

    if new_max_col - new_min_col + 1 < new_width:
        if new_min_col == 0:
            new_max_col = min(resized_width - 1, int(new_min_col + new_width))
        else:
            new_min_col = max(0, int(new_max_col - new_width))
    if new_max_row - new_min_row + 1 < new_height:
        if new_min_row == 0:
            new_max_row = min(resized_height - 1, int(new_min_row + new_height))
        else:
            new_min_row = max(0, int(new_max_row - new_height))
    return new_min_col, new_min_row, new_max_col + 1, new_max_row + 1


def _official_wow_dynamic_preprocess(
    image: Any,
    *,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    """Static-image subset of the pinned WOW-Seg preprocessing function."""

    width, height = image.size
    aspect_ratio = width / height
    target_ratios = {
        (columns, rows)
        for count in range(min_num, max_num + 1)
        for columns in range(1, count + 1)
        for rows in range(1, count + 1)
        if min_num <= columns * rows <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    best_difference = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized = image.resize((target_width, target_height))
    patches = []
    for index in range(best_ratio[0] * best_ratio[1]):
        column = index % (target_width // image_size)
        row = index // (target_width // image_size)
        patches.append(
            resized.crop(
                (
                    column * image_size,
                    row * image_size,
                    (column + 1) * image_size,
                    (row + 1) * image_size,
                )
            )
        )
    if use_thumbnail and len(patches) != 1:
        patches.append(image.resize((image_size, image_size)))
    return patches, resized


def _official_wow_transform(image_size: int = 448):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.convert("RGB") if image.mode != "RGB" else image
            ),
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


class OfficialWowRunner:
    """Run the pinned WOW-Seg mask-selection path without interface fallbacks."""

    _MASK_SELECTION_PATTERN = re.compile(
        r"mask_token_idx\s*\[\s*vaild_region_idx\s*\]\s*=\s*"
        r"pixel_masks\.reshape\(\s*-1\s*,\s*256\s*\)"
    )

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        transform: Callable[[Any], Any],
        dynamic_preprocess: Callable[..., Any],
        device: str,
    ) -> None:
        import torch

        if not device:
            raise ValueError("WOW device must be nonempty")
        parameters = inspect.signature(model.chat).parameters
        required = {"pixel_masks", "vaild_region_idx"}
        if not required.issubset(parameters):
            raise RuntimeError("BLOCKED_WOW_MASK_INTERFACE: chat lacks mask inputs")
        source = inspect.getsource(model.generate)
        if (
            self._MASK_SELECTION_PATTERN.search(source) is None
            or "vit_embeds_with_mask_token" not in source
        ):
            raise RuntimeError(
                "BLOCKED_WOW_MASK_INTERFACE: generate lacks 16x16 mask selection"
            )
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.transform = transform
        self.dynamic_preprocess = dynamic_preprocess
        self.device = torch.device(device)
        self.generate_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_local(
        cls,
        model_path: str,
        *,
        official_repo_path: str,
        device: str = "cuda",
    ) -> OfficialWowRunner:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dataset_source = (
            Path(official_repo_path).resolve()
            / "wow_eval"
            / "internvl"
            / "train"
            / "dataset.py"
        )
        if not dataset_source.is_file():
            raise FileNotFoundError("pinned WOW-Seg preprocessing source is missing")

        target = torch.device(device)
        dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(target)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )
        runner = cls(
            model=model,
            tokenizer=tokenizer,
            transform=_official_wow_transform(448),
            dynamic_preprocess=_official_wow_dynamic_preprocess,
            device=str(target),
        )
        runner.preprocess_source_sha256 = hashlib.sha256(
            dataset_source.read_bytes()
        ).hexdigest()
        return runner

    def _dynamic_full_image(self, image: np.ndarray, *, max_patches: int, thumbnail: bool):
        from PIL import Image

        return self.dynamic_preprocess(
            Image.fromarray(image, mode="RGB"),
            min_num=1,
            max_num=max_patches,
            image_size=448,
            use_thumbnail=thumbnail,
        )

    def prepare_region(
        self,
        image: np.ndarray,
        region_mask: np.ndarray,
        *,
        scale: float,
        image_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from torch.nn import functional

        if scale != 2.5 or image_size != 448:
            raise ValueError("WOW preprocessing is fixed to scale=2.5 and image_size=448")
        _patches, resized_image = self._dynamic_full_image(
            image, max_patches=12, thumbnail=True
        )
        mask = torch.from_numpy(np.asarray(region_mask, dtype=np.float32))[None, None]
        resized_mask = functional.interpolate(
            mask,
            size=(resized_image.size[1], resized_image.size[0]),
            mode="nearest",
        )[0, 0]
        crop_box = _wow_crop_box(resized_mask, resized_image.size, scale)
        if crop_box is None:
            raise ValueError("WOW region mask became empty before cropping")
        min_col, min_row, max_col, max_row = crop_box
        crop = resized_image.crop(crop_box)
        cropped_mask = resized_mask[None, min_row:max_row, min_col:max_col]
        mask16 = functional.interpolate(
            cropped_mask[None], size=(16, 16), mode="nearest"
        )[0, 0]
        return np.asarray(crop, dtype=np.uint8), mask16.cpu().numpy().astype(bool)

    def generate(
        self,
        *,
        images: tuple[np.ndarray, np.ndarray],
        region_mask_16: np.ndarray,
        prompt: str,
        generation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import torch

        if prompt != WOW_PROMPT or dict(generation) != dict(WOW_GENERATION):
            raise ValueError("WOW prompt and generation settings are protocol-fixed")
        full_patches, _resized = self._dynamic_full_image(
            images[0],
            max_patches=int(generation["max_dynamic_patches"]),
            thumbnail=bool(generation["thumbnail"]),
        )
        tensors = [self.transform(patch) for patch in full_patches]
        from PIL import Image

        tensors.append(self.transform(Image.fromarray(images[1], mode="RGB")))
        dtype = getattr(self.model, "dtype", None)
        if dtype is None:
            dtype = next(self.model.parameters()).dtype
        pixel_values = torch.stack(tensors).to(device=self.device, dtype=dtype)
        pixel_masks = torch.from_numpy(np.asarray(region_mask_16, dtype=bool))[
            None
        ].to(self.device)
        region_index = torch.tensor(
            len(full_patches), dtype=torch.long, device=self.device
        )
        generation_config = {
            "do_sample": False,
            "max_new_tokens": int(generation["max_new_tokens"]),
            "num_beams": int(generation["num_beams"]),
        }
        observed: dict[str, Any] = {}
        original_generate = self.model.generate

        def traced_generate(*args: Any, **kwargs: Any):
            actual_mask = kwargs.get("pixel_masks")
            actual_index = kwargs.get("vaild_region_idx")
            observed["mask_identity"] = actual_mask is pixel_masks
            observed["region_index"] = (
                int(actual_index.item()) if isinstance(actual_index, torch.Tensor) else None
            )
            result = original_generate(*args, **kwargs)
            observed["returned"] = True
            return result

        self.model.generate = traced_generate
        try:
            raw_generation = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config,
                vaild_region_idx=region_index,
                pixel_masks=pixel_masks,
            )
        finally:
            self.model.generate = original_generate
        consumed = (
            observed.get("mask_identity") is True
            and observed.get("region_index") == len(full_patches)
            and observed.get("returned") is True
        )
        return {
            "raw_generation": str(raw_generation),
            "trace": {
                "boundary": "official_generate.visual_token_mask_selection",
                "mask_consumed": consumed,
                "mask_shape": [16, 16],
                "mask_support": int(pixel_masks.sum().item()),
                "full_patch_count": len(full_patches),
                "combined_patch_count": len(tensors),
                "region_tile_index": len(full_patches),
                "generate_source_sha256": self.generate_source_sha256,
            },
        }


@dataclass(frozen=True)
class WowResult:
    status: str
    raw_generation: str | None
    original_mask_support: int
    final_mask_support: int
    mode: str
    trace: Mapping[str, Any]


class WowRegionAdapter:
    def __init__(self, runner: WowRunner, *, mode: str = "official_combined") -> None:
        if mode not in _WOW_MODES:
            raise ValueError("unsupported WOW input mode")
        self._runner = runner
        self.mode = mode

    def classify(self, image: Any, region_mask: Any) -> WowResult:
        rgb = np.asarray(image)
        mask = np.asarray(region_mask, dtype=bool)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("WOW image must be HxWx3 uint8")
        if mask.shape != rgb.shape[:2] or not np.any(mask):
            raise ValueError("WOW region mask must be aligned and nonempty")
        region_image, mask16_value = self._runner.prepare_region(
            rgb, mask, scale=2.5, image_size=448
        )
        mask16 = np.asarray(mask16_value, dtype=bool)
        if mask16.shape != (16, 16):
            raise ValueError("WOW official region mask must have shape 16x16")
        support = int(np.count_nonzero(mask16))
        if support == 0:
            return WowResult(
                status="UNAVAILABLE_EMPTY_WOW_MASK",
                raw_generation=None,
                original_mask_support=int(np.count_nonzero(mask)),
                final_mask_support=0,
                mode=self.mode,
                trace=MappingProxyType({}),
            )
        response = self._runner.generate(
            images=(rgb, np.asarray(region_image)),
            region_mask_16=mask16,
            prompt=WOW_PROMPT,
            generation=WOW_GENERATION,
        )
        trace = response.get("trace")
        if not isinstance(trace, Mapping):
            raise TypeError("WOW runner did not return a mask-consumption trace")
        if (
            trace.get("boundary") is None
            or trace.get("mask_consumed") is not True
            or tuple(trace.get("mask_shape", ())) != (16, 16)
            or int(trace.get("mask_support", -1)) != support
        ):
            raise RuntimeError("WOW visual-token path did not consume the region mask")
        raw_generation = response.get("raw_generation")
        if not isinstance(raw_generation, str) or not raw_generation.strip():
            return WowResult(
                status="UNAVAILABLE_TECHNICAL_FAILURE",
                raw_generation=None,
                original_mask_support=int(np.count_nonzero(mask)),
                final_mask_support=support,
                mode=self.mode,
                trace=MappingProxyType(dict(trace)),
            )
        return WowResult(
            status="COMPLETE",
            raw_generation=raw_generation,
            original_mask_support=int(np.count_nonzero(mask)),
            final_mask_support=support,
            mode=self.mode,
            trace=MappingProxyType(dict(trace)),
        )


def clean_wow_category_response(text: str) -> str:
    """Author-provided first-line category cleaner from the pinned WOW demo."""

    cleaned = (text or "").replace("\r", "\n").strip()
    if not cleaned:
        return cleaned
    first_line = cleaned.split("\n")[0].strip()
    lowered = first_line.lower()
    for prefix in ("category:", "label:", "class:", "answer:"):
        if lowered.startswith(prefix):
            first_line = first_line[len(prefix) :].strip()
            break
    if ". " in first_line:
        first_line = first_line.split(". ", 1)[0].strip()
    return first_line.strip(" .")


def _normalize_name(value: str) -> str:
    replaced = re.sub(r"[-_]", " ", value.strip().lower())
    return " ".join(replaced.split())


@dataclass(frozen=True)
class NameMapping:
    raw_generation: str
    cleaned_generation: str
    class_index: int
    class_name: str
    method: str
    similarities: tuple[float, ...]
    top1_top2_gap: float
    interpretation_flag: bool


def map_generated_name(
    raw_generation: str,
    class_names: Sequence[str],
    *,
    clean_response: Callable[[str], str],
    embed_text: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> NameMapping:
    """Apply official cleaning, exact matching, then frozen MiniLM cosine."""

    if not isinstance(raw_generation, str) or not raw_generation.strip():
        raise ValueError("WOW generation must be nonempty")
    classes = tuple(str(name) for name in class_names)
    if not classes or any(not name.strip() for name in classes):
        raise ValueError("class names must be nonempty")
    cleaned_raw = clean_response(raw_generation)
    if not isinstance(cleaned_raw, str):
        raise TypeError("official response cleaner must return text")
    cleaned = _normalize_name(cleaned_raw)
    if not cleaned:
        raise ValueError("cleaned WOW generation is empty")
    normalized_classes = tuple(_normalize_name(name) for name in classes)
    exact = [index for index, name in enumerate(normalized_classes) if name == cleaned]
    if len(exact) == 1:
        index = exact[0]
        similarities = tuple(1.0 if row == index else 0.0 for row in range(len(classes)))
        return NameMapping(
            raw_generation=raw_generation,
            cleaned_generation=cleaned,
            class_index=index,
            class_name=classes[index],
            method="EXACT",
            similarities=similarities,
            top1_top2_gap=1.0,
            interpretation_flag=False,
        )
    if embed_text is None:
        raise ValueError("non-exact WOW names require a frozen text embedder")
    rows = _unit_rows(
        embed_text((cleaned, *classes)), name="WOW name-mapping embeddings"
    )
    if rows.shape[0] != len(classes) + 1:
        raise ValueError("name mapper must return one row per supplied string")
    similarities_array = rows[1:] @ rows[0]
    index = int(np.argmax(similarities_array))
    ordered_scores = np.sort(similarities_array)[::-1]
    gap = (
        float(ordered_scores[0] - ordered_scores[1])
        if len(ordered_scores) > 1
        else float(ordered_scores[0])
    )
    return NameMapping(
        raw_generation=raw_generation,
        cleaned_generation=cleaned,
        class_index=index,
        class_name=classes[index],
        method="MINILM_COSINE",
        similarities=tuple(float(value) for value in similarities_array),
        top1_top2_gap=gap,
        interpretation_flag=(len(cleaned.split()) > 4 or Counter(cleaned).get(",", 0) > 0),
    )
