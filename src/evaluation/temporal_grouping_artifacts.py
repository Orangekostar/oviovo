"""Geometry-preserving artifacts for one dense temporal object grouping."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from src.evaluation.ovi_pair_artifacts import (
    OviPairArtifactExport,
    _file_record,
    _preview_camera,
    _render_preview,
    _write_json,
)
from src.evaluation.ovi_pair_views import OviObjectPairView, OviObjectVisitView
from src.evaluation.temporal_object_groups import TemporalObjectGrouping

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


def _instance_colors(
    visit: OviObjectVisitView,
    grouping: TemporalObjectGrouping,
) -> np.ndarray:
    colors = np.full((visit.point_count, 3), 96, dtype=np.uint8)
    for value in grouping.objects[visit.visit_id]:
        digest = hashlib.sha256(value.object_id.encode("utf-8")).digest()
        color = np.maximum(np.frombuffer(digest[:3], dtype=np.uint8), 48)
        colors[value.source_point_indices] = color
    return colors


def _rgb_colors(visit: OviObjectVisitView) -> np.ndarray:
    colors = np.full((visit.point_count, 3), 160, dtype=np.uint8)
    colors[visit.appearance_valid] = visit.camera_rgb_uint8[visit.appearance_valid]
    return colors


def _write_grouping_ply(
    path: Path,
    visit: OviObjectVisitView,
    owner_indices: np.ndarray,
    colors: np.ndarray,
) -> None:
    owners = np.asarray(owner_indices)
    if (
        owners.shape != (visit.point_count,)
        or not np.issubdtype(owners.dtype, np.integer)
        or np.any(owners < -1)
        or colors.shape != (visit.point_count, 3)
        or colors.dtype != np.uint8
    ):
        raise ValueError("grouping PLY arrays must align with the dense visit")
    records = np.empty(visit.point_count, dtype=_PLY_DTYPE)
    for column, name in enumerate(("x", "y", "z")):
        records[name] = visit.points_xyz[:, column]
    for column, name in enumerate(("normal_x", "normal_y", "normal_z")):
        records[name] = visit.normals_xyz[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, column]
    records["entity_index"] = (owners + 1).astype(np.uint32)
    records["normal_valid"] = visit.normal_valid.astype(np.uint8)
    records["appearance_valid"] = visit.appearance_valid.astype(np.uint8)
    records["appearance_frame_id"] = visit.appearance_frame_ids.astype(np.int32)
    records["appearance_source_frame_id"] = (
        visit.appearance_source_frame_ids.astype(np.int32)
    )
    records["appearance_row"] = visit.appearance_rows.astype(np.int32)
    records["appearance_column"] = visit.appearance_columns.astype(np.int32)
    records["appearance_camera_depth_m"] = visit.appearance_camera_depth_m
    records["appearance_observed_depth_m"] = visit.appearance_observed_depth_m
    records["appearance_depth_residual_m"] = visit.appearance_depth_residual_m
    records["source_vertex_index"] = visit.source_vertex_indices.astype(np.uint32)
    PlyData(
        [PlyElement.describe(records, "vertex")], text=False, byte_order="<"
    ).write(path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def export_temporal_grouping_artifacts(
    pair: OviObjectPairView,
    grouping: TemporalObjectGrouping,
    output_dir: Path,
    *,
    preview_width: int = 960,
    preview_height: int = 720,
) -> OviPairArtifactExport:
    """Export dense current ownership without changing OVI geometry or provenance."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    if not isinstance(grouping, TemporalObjectGrouping):
        raise TypeError("grouping must be TemporalObjectGrouping")
    if (
        grouping.pair_content_sha256 != pair.content_sha256()
        or grouping.source_xyz_multiset_sha256
        != grouping.output_xyz_multiset_sha256
    ):
        raise ValueError("grouping does not bind unchanged pair geometry")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = _preview_camera(pair, width=preview_width, height=preview_height)
    current = pair.visits[1]
    instance_colors = _instance_colors(current, grouping)
    rgb_colors = _rgb_colors(current)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    outputs: dict[str, Path] = {}
    try:
        outputs["current_instance_ply"] = staging / "current_instance.ply"
        outputs["current_rgb_preview"] = staging / "current_rgb.png"
        outputs["current_instance_preview"] = staging / "current_instance.png"
        _write_grouping_ply(
            outputs["current_instance_ply"],
            current,
            grouping.owner_object_indices[1],
            instance_colors,
        )
        _render_preview(
            outputs["current_rgb_preview"], current.points_xyz, rgb_colors, camera
        )
        _render_preview(
            outputs["current_instance_preview"],
            current.points_xyz,
            instance_colors,
            camera,
        )
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_DENSE_CURRENT_GROUPING_V1",
            "status": "DENSE_CURRENT_GROUPING_PASS",
            "pair_id": pair.pair_id,
            "variant_id": grouping.variant_id,
            "pair_content_sha256": pair.content_sha256(),
            "grouping_sha256": grouping.content_sha256(),
            "coordinate_frame_id": pair.coordinate_frame_id,
            "geometry": {
                "source": "native OVI-MAP dense t1 instance PLY",
                "xyz_changed": False,
                "point_order_changed": False,
                "point_count_changed": False,
                "smoothing_or_hole_filling": False,
            },
            "instance": {
                "entity_index_zero": "unowned_background",
                "entity_index_positive": "one_based_index_into_objects",
                "color_source": "sha256_of_composite_object_id_for_visualization_only",
                "objects": [
                    {
                        "entity_index": index,
                        "object_id": value.object_id,
                        "member_entity_ids": list(value.member_entity_ids),
                        "temporal_identity_id": value.temporal_identity_id,
                        "query_ids": list(value.query_ids),
                        "query_confidence": value.query_confidence,
                        "query_support": value.query_support,
                        "point_count": len(value.source_point_indices),
                    }
                    for index, value in enumerate(grouping.objects[1], start=1)
                ],
            },
            "current_visit": {
                "visit_id": current.visit_id,
                "scan_id": current.scan_id,
                "frame_count": current.frame_count,
                "point_count": current.point_count,
                "object_count": len(grouping.objects[1]),
                "background_point_count": current.background_point_count,
                "appearance_support_count": current.appearance_support_count,
                "native_manifest_sha256": current.native_manifest_sha256,
                "materialized_manifest_sha256": current.materialized_manifest_sha256,
            },
            "preview_camera": camera.to_json(),
            "outputs": {
                role: _file_record(path, root=staging)
                for role, path in sorted(outputs.items())
            },
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return OviPairArtifactExport(output_dir=output, manifest=output / "manifest.json")


__all__ = ["export_temporal_grouping_artifacts"]
