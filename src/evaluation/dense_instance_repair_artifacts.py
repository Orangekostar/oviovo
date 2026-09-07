"""Deterministic full-density artifacts for P0/P1/P2 method views."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from src.evaluation.dense_instance_repair_metrics import DenseMethodView
from src.evaluation.ovi_pair_artifacts import (
    _colors,
    _file_record,
    _preview_camera,
    _render_preview,
    _write_json,
)
from src.evaluation.ovi_pair_views import OviObjectPairView, OviObjectVisitView
from src.oviv2.rescene_dense_instance_readout import (
    OWNER_BACKGROUND,
    OWNER_UNKNOWN,
)

_DENSE_PLY_DTYPE = np.dtype(
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
        ("instance_index", "<u4"),
        ("owner_source", "u1"),
        ("neural_valid", "u1"),
        ("source_vertex_index", "<u4"),
    ]
)

_BACKGROUND_RGB = np.asarray([96, 96, 96], dtype=np.uint8)
_UNKNOWN_RGB = np.asarray([255, 0, 255], dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class DenseMethodArtifactExport:
    output_dir: Path
    manifest: Path


def _candidate_color(method_id: str, candidate_id: str) -> np.ndarray:
    digest = hashlib.sha256(f"{method_id}\0{candidate_id}".encode("utf-8")).digest()
    return np.asarray([48 + value % 160 for value in digest[:3]], dtype=np.uint8)


def _instance_colors(
    view: DenseMethodView, *, visit_id: int
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    sources = view.owner_source_codes[visit_id]
    colors = np.tile(_BACKGROUND_RGB, (len(sources), 1))
    colors[sources == OWNER_UNKNOWN] = _UNKNOWN_RGB
    records = []
    for candidate_index, candidate in enumerate(view.candidates[visit_id]):
        color = _candidate_color(view.method_id, candidate.candidate_id)
        colors[view.owner_instance_indices[visit_id] == candidate_index] = color
        records.append(
            {
                "instance_index": candidate_index + 1,
                "candidate_id": candidate.candidate_id,
                "rgb": color.tolist(),
            }
        )
    return colors, tuple(records)


def _write_dense_ply(
    path: Path,
    visit: OviObjectVisitView,
    *,
    owner_instance_indices: np.ndarray,
    owner_source_codes: np.ndarray,
    neural_valid: np.ndarray,
    colors: np.ndarray,
) -> None:
    if visit.point_count > np.iinfo(np.uint32).max:
        raise ValueError("dense source vertex index exceeds uint32")
    if any(
        value.shape != (visit.point_count,)
        for value in (owner_instance_indices, owner_source_codes, neural_valid)
    ) or colors.shape != (visit.point_count, 3):
        raise ValueError("dense artifact arrays do not align with visit rows")
    records = np.empty(visit.point_count, dtype=_DENSE_PLY_DTYPE)
    for column, name in enumerate(("x", "y", "z")):
        records[name] = visit.points_xyz[:, column]
    for column, name in enumerate(("normal_x", "normal_y", "normal_z")):
        records[name] = visit.normals_xyz[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, column]
    records["instance_index"] = (owner_instance_indices + 1).astype(np.uint32)
    records["owner_source"] = owner_source_codes
    records["neural_valid"] = neural_valid.astype(np.uint8)
    records["source_vertex_index"] = visit.source_vertex_indices.astype(np.uint32)
    PlyData(
        [PlyElement.describe(records, "vertex")], text=False, byte_order="<"
    ).write(path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def export_dense_method_artifacts(
    pair: OviObjectPairView,
    view: DenseMethodView,
    output_dir: str | Path,
    *,
    preview_width: int = 960,
    preview_height: int = 720,
) -> DenseMethodArtifactExport:
    """Write both visits and the t1 current readout without changing dense XYZ."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    if not isinstance(view, DenseMethodView):
        raise TypeError("view must be DenseMethodView")
    if (
        pair.pair_id != view.pair_id
        or pair.content_sha256() != view.pair_content_sha256
        or view.source_xyz_sha256 != view.output_xyz_sha256
    ):
        raise ValueError("dense method view does not bind the OVI pair")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = _preview_camera(pair, width=preview_width, height=preview_height)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    paths: dict[str, Path] = {}
    candidate_colors: dict[str, tuple[dict[str, object], ...]] = {}
    try:
        for scope, visit_id in (("t0", 0), ("t1", 1), ("current", 1)):
            visit = pair.visits[visit_id]
            scope_dir = staging / scope
            scope_dir.mkdir()
            instance_colors, records = _instance_colors(view, visit_id=visit_id)
            candidate_colors.setdefault(f"t{visit_id}", records)
            instance_ply = scope_dir / "instances.ply"
            instance_preview = scope_dir / "instances.png"
            rgb_preview = scope_dir / "rgb.png"
            _write_dense_ply(
                instance_ply,
                visit,
                owner_instance_indices=view.owner_instance_indices[visit_id],
                owner_source_codes=view.owner_source_codes[visit_id],
                neural_valid=view.neural_valid[visit_id],
                colors=instance_colors,
            )
            _render_preview(
                instance_preview, visit.points_xyz, instance_colors, camera
            )
            _render_preview(
                rgb_preview,
                visit.points_xyz,
                _colors(visit, mode="rgb"),
                camera,
            )
            paths[f"{scope}_instance_ply"] = instance_ply
            paths[f"{scope}_instance_preview"] = instance_preview
            paths[f"{scope}_rgb_preview"] = rgb_preview
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_DENSE_INSTANCE_ARTIFACT_V1",
            "status": "DENSE_INSTANCE_ARTIFACT_PASS",
            "pair_id": pair.pair_id,
            "method_id": view.method_id,
            "pair_content_sha256": pair.content_sha256(),
            "method_view_sha256": view.content_sha256(),
            "geometry": {
                "xyz_changed": False,
                "point_order_changed": False,
                "point_count_changed": False,
            },
            "instance_encoding": {
                "instance_index_zero": "background_or_unknown",
                "instance_index_positive": "one_based_index_into_visit_candidates",
                "owner_source_codes": {
                    "1": "query",
                    "2": "ovi_residual",
                    "3": "background",
                    "4": "unknown",
                },
                "background_rgb": _BACKGROUND_RGB.tolist(),
                "unknown_rgb": _UNKNOWN_RGB.tolist(),
            },
            "candidate_colors": candidate_colors,
            "preview_camera": camera.to_json(),
            "outputs": {
                role: _file_record(path, root=staging)
                for role, path in sorted(paths.items())
            },
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return DenseMethodArtifactExport(output, output / "manifest.json")


__all__ = ["DenseMethodArtifactExport", "export_dense_method_artifacts"]
