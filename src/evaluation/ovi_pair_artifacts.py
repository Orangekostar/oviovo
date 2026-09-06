"""Deterministic geometry-preserving artifacts for a dense OVI pair view."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from src.evaluation.ovi_pair_views import OviObjectPairView, OviObjectVisitView

_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("normal_x", "<f4"),
        ("normal_y", "<f4"),
        ("normal_z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("entity_index", "<u4"),
        ("normal_valid", "u1"),
        ("appearance_valid", "u1"),
        ("appearance_frame_id", "<i4"),
        ("appearance_source_frame_id", "<i4"),
        ("appearance_row", "<i4"),
        ("appearance_column", "<i4"),
        ("appearance_camera_depth_m", "<f4"),
        ("appearance_observed_depth_m", "<f4"),
        ("appearance_depth_residual_m", "<f4"),
        ("source_vertex_index", "<u4"),
    ]
)


@dataclass(frozen=True, slots=True)
class OviPairArtifactExport:
    output_dir: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class _PreviewCamera:
    center: np.ndarray
    view_direction: np.ndarray
    right: np.ndarray
    up: np.ndarray
    projected_center: tuple[float, float]
    meters_per_pixel: float
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        return {
            "projection": "orthographic",
            "center_xyz": self.center.tolist(),
            "view_direction_xyz": self.view_direction.tolist(),
            "right_xyz": self.right.tolist(),
            "up_xyz": self.up.tolist(),
            "projected_center_xy": list(self.projected_center),
            "meters_per_pixel": self.meters_per_pixel,
            "width": self.width,
            "height": self.height,
            "point_radius_px": 0,
            "shared_across_visits_and_modes": True,
        }


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _colors(visit: OviObjectVisitView, *, mode: str) -> np.ndarray:
    if mode == "rgb":
        result = np.full((visit.point_count, 3), 160, dtype=np.uint8)
        result[visit.appearance_valid] = visit.camera_rgb_uint8[visit.appearance_valid]
        return result
    if mode != "instance":
        raise ValueError("preview mode must be rgb or instance")
    result = np.full((visit.point_count, 3), 96, dtype=np.uint8)
    owned = visit.entity_owner_indices >= 0
    result[owned] = visit.palette_rgb_uint8[owned]
    return result


def _write_ply(path: Path, visit: OviObjectVisitView, colors: np.ndarray) -> None:
    if colors.shape != (visit.point_count, 3) or colors.dtype != np.uint8:
        raise ValueError("PLY colors must align with the dense visit")
    if visit.point_count > np.iinfo(np.uint32).max:
        raise ValueError("PLY source vertex index exceeds uint32")
    records = np.empty(visit.point_count, dtype=_PLY_DTYPE)
    for column, name in enumerate(("x", "y", "z")):
        records[name] = visit.points_xyz[:, column]
    for column, name in enumerate(("normal_x", "normal_y", "normal_z")):
        records[name] = visit.normals_xyz[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, column]
    records["entity_index"] = (visit.entity_owner_indices + 1).astype(np.uint32)
    records["normal_valid"] = visit.normal_valid.astype(np.uint8)
    records["appearance_valid"] = visit.appearance_valid.astype(np.uint8)
    records["appearance_frame_id"] = visit.appearance_frame_ids.astype(np.int32)
    records["appearance_source_frame_id"] = visit.appearance_source_frame_ids.astype(
        np.int32
    )
    records["appearance_row"] = visit.appearance_rows.astype(np.int32)
    records["appearance_column"] = visit.appearance_columns.astype(np.int32)
    records["appearance_camera_depth_m"] = visit.appearance_camera_depth_m
    records["appearance_observed_depth_m"] = visit.appearance_observed_depth_m
    records["appearance_depth_residual_m"] = visit.appearance_depth_residual_m
    records["source_vertex_index"] = visit.source_vertex_indices.astype(np.uint32)
    PlyData(
        [PlyElement.describe(records, "vertex")],
        text=False,
        byte_order="<",
    ).write(path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _preview_camera(
    pair: OviObjectPairView, *, width: int, height: int
) -> _PreviewCamera:
    if isinstance(width, bool) or not isinstance(width, int) or width < 32:
        raise ValueError("preview width must be an integer of at least 32")
    if isinstance(height, bool) or not isinstance(height, int) or height < 32:
        raise ValueError("preview height must be an integer of at least 32")
    points = np.concatenate([visit.points_xyz for visit in pair.visits], axis=0).astype(
        np.float64, copy=False
    )
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    direction = np.asarray([1.0, -1.0, 0.75], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    up_seed = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(direction, up_seed)
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    up /= np.linalg.norm(up)
    horizontal = points @ right
    vertical = points @ up
    projected_center = (
        float((horizontal.min() + horizontal.max()) / 2.0),
        float((vertical.min() + vertical.max()) / 2.0),
    )
    span_x = float(horizontal.max() - horizontal.min())
    span_y = float(vertical.max() - vertical.min())
    meters_per_pixel = (
        max(
            span_x / max(width - 3, 1),
            span_y / max(height - 3, 1),
            np.finfo(np.float64).eps,
        )
        * 1.03
    )
    return _PreviewCamera(
        center=center,
        view_direction=direction,
        right=right,
        up=up,
        projected_center=projected_center,
        meters_per_pixel=meters_per_pixel,
        width=width,
        height=height,
    )


def _render_preview(
    path: Path,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    camera: _PreviewCamera,
) -> None:
    points = np.asarray(points_xyz, dtype=np.float64)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    horizontal = points @ camera.right
    vertical = points @ camera.up
    depth = (points - camera.center) @ camera.view_direction
    columns = np.rint(
        (horizontal - camera.projected_center[0]) / camera.meters_per_pixel
        + (camera.width - 1) / 2.0
    ).astype(np.int64)
    rows = np.rint(
        (camera.projected_center[1] - vertical) / camera.meters_per_pixel
        + (camera.height - 1) / 2.0
    ).astype(np.int64)
    inside = (
        (rows >= 0)
        & (rows < camera.height)
        & (columns >= 0)
        & (columns < camera.width)
        & np.isfinite(depth)
    )
    rows = rows[inside]
    columns = columns[inside]
    depth = depth[inside]
    colors = colors[inside]
    pixels = rows * camera.width + columns
    order = np.lexsort((depth, pixels))
    ordered_pixels = pixels[order]
    nearest = np.empty(len(order), dtype=bool)
    nearest[0] = True
    nearest[1:] = ordered_pixels[1:] != ordered_pixels[:-1]
    selected = order[nearest]
    image = np.full((camera.height, camera.width, 3), 255, dtype=np.uint8)
    image[rows[selected], columns[selected]] = colors[selected]
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write preview: {path}")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def export_ovi_object_pair_artifacts(
    pair: OviObjectPairView,
    output_dir: Path,
    *,
    preview_width: int = 960,
    preview_height: int = 720,
) -> OviPairArtifactExport:
    """Export complete measured geometry and visualization-only fixed views."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = _preview_camera(pair, width=preview_width, height=preview_height)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    output_paths: dict[str, Path] = {}
    try:
        for visit in pair.visits:
            visit_dir = staging / f"t{visit.visit_id}"
            visit_dir.mkdir()
            for mode in ("rgb", "instance"):
                colors = _colors(visit, mode=mode)
                ply = visit_dir / f"{mode}.ply"
                preview = visit_dir / f"{mode}.png"
                _write_ply(ply, visit, colors)
                _render_preview(preview, visit.points_xyz, colors, camera)
                output_paths[f"t{visit.visit_id}_{mode}_ply"] = ply
                output_paths[f"t{visit.visit_id}_{mode}_preview"] = preview
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_D2_PAIR_VIEW_V1",
            "status": "D2_PAIR_ARTIFACT_PASS",
            "pair_id": pair.pair_id,
            "domain_id": pair.domain_id,
            "pair_content_sha256": pair.content_sha256(),
            "source_manifest_sha256": pair.source_manifest_sha256,
            "coordinate_frame_id": pair.coordinate_frame_id,
            "geometry": {
                "source": "native OVI-MAP dense instance PLY",
                "xyz_changed": False,
                "point_order_changed": False,
                "point_count_changed": False,
                "smoothing_or_hole_filling": False,
            },
            "rgb": {
                "supported_source": "same_visit_rgbd_reprojected",
                "unsupported_color_rgb": [160, 160, 160],
                "unsupported_is_measured_rgb": False,
                "validity_property": "appearance_valid",
                "pixel_provenance_properties": [
                    "appearance_frame_id",
                    "appearance_source_frame_id",
                    "appearance_row",
                    "appearance_column",
                    "appearance_camera_depth_m",
                    "appearance_observed_depth_m",
                    "appearance_depth_residual_m",
                ],
            },
            "instance": {
                "entity_index_zero": "unowned_background",
                "entity_index_positive": "one_based_index_into_sorted_visit_entities",
                "color_source": "native OVI instance palette",
            },
            "visits": [
                {
                    "visit_id": visit.visit_id,
                    "scan_id": visit.scan_id,
                    "frame_count": visit.frame_count,
                    "point_count": visit.point_count,
                    "entity_count": len(visit.entities),
                    "entity_point_count": visit.entity_point_count,
                    "background_point_count": visit.background_point_count,
                    "normal_valid_count": int(np.count_nonzero(visit.normal_valid)),
                    "appearance_support_count": visit.appearance_support_count,
                    "appearance_support_fraction": visit.appearance_support_fraction,
                    "bounds_xyz": {
                        "minimum": visit.points_xyz.min(axis=0).tolist(),
                        "maximum": visit.points_xyz.max(axis=0).tolist(),
                    },
                    "native_manifest_sha256": visit.native_manifest_sha256,
                    "materialized_manifest_sha256": visit.materialized_manifest_sha256,
                    "source_artifact_sha256": dict(visit.source_artifact_sha256),
                    "global_alignment_application": visit.global_alignment_application,
                    "entities": [
                        {
                            "entity_index": entity_index,
                            "entity_id": entity.entity_id,
                            "source_instance_id": entity.source_instance_id,
                            "palette_rgb": list(entity.palette_rgb),
                            "point_count": entity.point_count,
                        }
                        for entity_index, entity in enumerate(visit.entities, start=1)
                    ],
                }
                for visit in pair.visits
            ],
            "preview_camera": camera.to_json(),
            "outputs": {
                role: _file_record(path, root=staging)
                for role, path in sorted(output_paths.items())
            },
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return OviPairArtifactExport(output_dir=output, manifest=output / "manifest.json")


__all__ = ["OviPairArtifactExport", "export_ovi_object_pair_artifacts"]
