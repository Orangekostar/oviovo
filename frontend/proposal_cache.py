"""Proposal cache helpers shared by offline precompute and replay backends."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.core.data_structures import Proposal2D


CACHE_FORMAT_VERSION = 1
FRAME_FILENAME_TEMPLATE = "frames/frame{frame_id:06d}_proposals.npz"
MANIFEST_FILENAME = "manifest.json"


def frame_relpath(frame_id: int) -> str:
    """Return the canonical per-frame cache relative path."""
    return FRAME_FILENAME_TEMPLATE.format(frame_id=int(frame_id))


def frame_path(cache_dir: str | Path, frame_id: int) -> Path:
    """Return the absolute per-frame cache path."""
    return Path(cache_dir) / frame_relpath(frame_id)


def sanitize_json_value(value: Any) -> Any:
    """Convert numpy/path objects into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_proposals(
    cache_dir: str | Path,
    *,
    frame_id: int,
    image_shape: tuple[int, int],
    proposals: list[Proposal2D],
    source_backend: str,
    generation_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one frame of normalized proposals into a compressed NPZ file."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    npz_path = frame_path(cache_root, frame_id)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = [int(value) for value in image_shape]
    proposal_count = len(proposals)

    if proposal_count > 0:
        masks = np.stack([np.asarray(proposal.mask, dtype=np.bool_) for proposal in proposals], axis=0)
        bbox_xyxy = np.stack(
            [np.asarray(proposal.bbox_xyxy, dtype=np.float32).reshape(4) for proposal in proposals],
            axis=0,
        ).astype(np.float32, copy=False)
        areas = np.asarray([int(proposal.area) for proposal in proposals], dtype=np.int32)
        confidences = np.asarray([float(proposal.confidence) for proposal in proposals], dtype=np.float32)
        proposal_ids = np.asarray([int(proposal.proposal_id) for proposal in proposals], dtype=np.int32)
        backend_names = np.asarray([str(proposal.backend_name) for proposal in proposals], dtype=np.str_)
        metadata_jsons = np.asarray(
            [
                json.dumps(
                    sanitize_json_value(proposal.metadata),
                    ensure_ascii=True,
                    sort_keys=True,
                )
                for proposal in proposals
            ],
            dtype=np.str_,
        )
    else:
        masks = np.zeros((0, height, width), dtype=np.bool_)
        bbox_xyxy = np.zeros((0, 4), dtype=np.float32)
        areas = np.zeros((0,), dtype=np.int32)
        confidences = np.zeros((0,), dtype=np.float32)
        proposal_ids = np.zeros((0,), dtype=np.int32)
        backend_names = np.asarray([], dtype=np.str_)
        metadata_jsons = np.asarray([], dtype=np.str_)

    generation_payload = sanitize_json_value(generation_info or {})
    np.savez_compressed(
        npz_path,
        cache_format_version=np.asarray([CACHE_FORMAT_VERSION], dtype=np.int32),
        frame_id=np.asarray([int(frame_id)], dtype=np.int32),
        image_shape=np.asarray([height, width], dtype=np.int32),
        proposal_ids=proposal_ids,
        masks=masks,
        bbox_xyxy=bbox_xyxy,
        areas=areas,
        confidences=confidences,
        backend_names=backend_names,
        metadata_jsons=metadata_jsons,
        source_backend=np.asarray([str(source_backend)], dtype=np.str_),
        generation_info_json=np.asarray(
            [json.dumps(generation_payload, ensure_ascii=True, sort_keys=True)],
            dtype=np.str_,
        ),
    )

    return {
        "frame_id": int(frame_id),
        "file": frame_relpath(frame_id),
        "proposal_count": int(proposal_count),
        "image_shape": [height, width],
        "source_backend": str(source_backend),
        "generation_info": generation_payload,
    }


def load_proposals(
    cache_dir: str | Path,
    *,
    frame_id: int,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[list[Proposal2D], dict[str, Any]]:
    """Load one cached frame of normalized proposals."""
    npz_path = frame_path(cache_dir, frame_id)
    if not npz_path.exists():
        raise FileNotFoundError(f"Cached proposals not found for frame {frame_id}: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as payload:
        image_shape = tuple(int(value) for value in payload["image_shape"].tolist())
        if expected_shape is not None and tuple(expected_shape) != image_shape:
            raise ValueError(
                "Cached proposal shape mismatch for frame "
                f"{frame_id}: cache={image_shape}, expected={tuple(expected_shape)}"
            )

        proposal_ids = payload["proposal_ids"].astype(np.int32, copy=False)
        masks = payload["masks"].astype(np.bool_, copy=False)
        bbox_xyxy = payload["bbox_xyxy"].astype(np.float32, copy=False)
        areas = payload["areas"].astype(np.int32, copy=False)
        confidences = payload["confidences"].astype(np.float32, copy=False)
        backend_names = payload["backend_names"].astype(np.str_)
        metadata_jsons = payload["metadata_jsons"].astype(np.str_)
        source_backend = str(payload["source_backend"][0]) if payload["source_backend"].size else "unknown"
        generation_info_json = (
            str(payload["generation_info_json"][0]) if payload["generation_info_json"].size else "{}"
        )

    proposals: list[Proposal2D] = []
    for idx in range(len(proposal_ids)):
        metadata = json.loads(str(metadata_jsons[idx])) if len(metadata_jsons) > idx else {}
        metadata = sanitize_json_value(metadata)
        metadata.setdefault("cache_source", "precomputed_npz")
        metadata.setdefault("cache_frame_id", int(frame_id))
        metadata.setdefault("cache_relpath", frame_relpath(frame_id))
        proposals.append(
            Proposal2D(
                proposal_id=int(proposal_ids[idx]),
                mask=np.asarray(masks[idx], dtype=np.bool_),
                bbox_xyxy=np.asarray(bbox_xyxy[idx], dtype=np.float32),
                area=int(areas[idx]),
                confidence=float(confidences[idx]),
                backend_name=str(backend_names[idx]),
                metadata=dict(metadata),
            )
        )

    frame_info = {
        "frame_id": int(frame_id),
        "file": frame_relpath(frame_id),
        "proposal_count": int(len(proposals)),
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "source_backend": source_backend,
        "generation_info": json.loads(generation_info_json),
    }
    return proposals, frame_info


def write_manifest(
    cache_dir: str | Path,
    *,
    dataset_summary: dict[str, Any],
    backend_config: dict[str, Any],
    frame_entries: list[dict[str, Any]],
    notes: dict[str, Any] | None = None,
) -> Path:
    """Write the top-level cache manifest."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": sanitize_json_value(dataset_summary),
        "backend_config": sanitize_json_value(backend_config),
        "frame_count": int(len(frame_entries)),
        "frames": [sanitize_json_value(entry) for entry in frame_entries],
        "notes": sanitize_json_value(notes or {}),
    }
    manifest_path = cache_root / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Load and parse a cache manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", -1)) != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported proposal cache format_version={payload.get('format_version')} "
            f"(expected {CACHE_FORMAT_VERSION})"
        )
    return payload
