from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

from src.oviv2.current_surface import (
    CurrentEvidenceState,
    CurrentSurfaceView,
    SemanticSource,
    export_current_surface,
    select_current_surface,
)


def _surface() -> CurrentSurfaceView:
    return CurrentSurfaceView(
        surface_id="fixture",
        vertices_xyz=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        normals_xyz=np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (5, 1)
        ),
        triangles=np.asarray([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int64),
        source_surface_indices=np.zeros(5, dtype=np.uint16),
        source_vertex_indices=np.arange(5, dtype=np.int64),
        source_visit_ids=np.asarray([0, 0, 0, 0, 1], dtype=np.int16),
        geometry_epochs=np.asarray([0, 0, 0, 0, 1], dtype=np.int32),
        observed_rgb_uint8=np.asarray(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [1, 2, 3], [4, 5, 6]],
            dtype=np.uint8,
        ),
        rgb_valid=np.asarray([True, True, False, True, True]),
        current_valid=np.asarray([True, True, True, False, True]),
        evidence_state_codes=np.asarray(
            [
                CurrentEvidenceState.CURRENT_OBSERVED,
                CurrentEvidenceState.HISTORICAL_OCCLUDED,
                CurrentEvidenceState.HISTORICAL_UNOBSERVED,
                CurrentEvidenceState.REVOKED_VISIBLE_FREE,
                CurrentEvidenceState.CURRENT_OBSERVED,
            ],
            dtype=np.uint8,
        ),
        last_supported_frames=np.asarray([10, 3, -1, 8, 12], dtype=np.int32),
        owner_entity_ids=np.asarray([11, 11, 0, 12, 13], dtype=np.int64),
        owner_confidences=np.asarray([0.9, 0.8, 0.0, 0.7, 0.95], dtype=np.float32),
        semantic_ids=np.asarray([2, 2, 0, 3, 4], dtype=np.int32),
        semantic_confidences=np.asarray([0.8, 0.7, 0.0, 0.6, 0.9], dtype=np.float32),
        semantic_support_reliabilities=np.asarray(
            [0.9, 0.8, 0.0, 0.5, 0.95], dtype=np.float32
        ),
        semantic_source_codes=np.asarray(
            [
                SemanticSource.LOCAL_SURFACE,
                SemanticSource.LOCAL_SURFACE,
                SemanticSource.UNKNOWN,
                SemanticSource.OWNER_ENTITY,
                SemanticSource.BLENDED,
            ],
            dtype=np.uint8,
        ),
    )


def _ply_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = PlyData.read(path)
    vertices = payload["vertex"].data
    faces = np.asarray(
        [row[0] for row in payload["face"].data], dtype=np.int64
    )
    return vertices, faces


def test_current_selection_keeps_only_all_valid_source_triangles() -> None:
    selected = select_current_surface(_surface())

    assert selected.source_row_indices.tolist() == [0, 1, 2, 4]
    assert selected.triangles.tolist() == [[0, 1, 2]]


def test_four_exports_share_geometry_and_numeric_state(tmp_path: Path) -> None:
    exported = export_current_surface(
        _surface(),
        tmp_path / "surface",
        semantic_palette={0: (96, 96, 96), 2: (220, 20, 60), 4: (0, 128, 0)},
        owner_id_table={0: "background", 11: "ovimap:11", 13: "ovimap:13"},
        source_surfaces=(
            {
                "surface_index": 0,
                "surface_id": "source",
                "path": "/source/mesh.ply",
                "sha256": "a" * 64,
                "surface_voxel_size_m": 0.01,
            },
        ),
    )

    modes = ("rgb", "instance", "semantic", "state")
    payloads = {
        mode: _ply_payload(exported.output_dir / f"current_{mode}.ply")
        for mode in modes
    }
    first_vertices, first_faces = payloads["rgb"]
    numeric = (
        "x",
        "y",
        "z",
        "normal_x",
        "normal_y",
        "normal_z",
        "source_surface_index",
        "source_vertex_index",
        "source_visit",
        "geometry_epoch",
        "rgb_valid",
        "evidence_state",
        "last_supported_frame",
        "owner_entity_id",
        "owner_confidence",
        "semantic_id",
        "semantic_confidence",
        "semantic_support_reliability",
        "semantic_source",
    )
    for mode in modes[1:]:
        vertices, faces = payloads[mode]
        np.testing.assert_array_equal(faces, first_faces)
        for name in numeric:
            np.testing.assert_array_equal(vertices[name], first_vertices[name])
    assert any(
        not np.array_equal(
            payloads[mode][0]["red"], first_vertices["red"]
        )
        for mode in modes[1:]
    )

    with np.load(exported.output_dir / "current_surface.npz") as sidecar:
        assert sidecar["current_valid"].tolist() == [True, True, True, False, True]
        assert sidecar["source_vertex_indices"].tolist() == [0, 1, 2, 3, 4]
    manifest = json.loads(exported.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["canonical_vertex_count"] == 5
    assert manifest["current_vertex_count"] == 4
    assert manifest["current_face_count"] == 1
    assert manifest["geometry_shared_across_views"] is True
    assert manifest["source_surfaces"][0]["sha256"] == "a" * 64
    assert json.loads(
        (exported.output_dir / "owner_id_table.json").read_text(encoding="utf-8")
    ) == {"0": "background", "11": "ovimap:11", "13": "ovimap:13"}
    assert "owner_id_table.json" in manifest["artifacts"]


def test_export_refuses_overwrite_and_missing_semantic_palette(tmp_path: Path) -> None:
    output = tmp_path / "surface"
    with pytest.raises(ValueError, match="semantic palette"):
        export_current_surface(
            _surface(),
            output,
            semantic_palette={0: (96, 96, 96)},
            owner_id_table={0: "background", 11: "ovimap:11", 13: "ovimap:13"},
            source_surfaces=(),
        )

    export_current_surface(
        _surface(),
        output,
        semantic_palette={0: (96, 96, 96), 2: (1, 2, 3), 4: (4, 5, 6)},
        owner_id_table={0: "background", 11: "ovimap:11", 13: "ovimap:13"},
        source_surfaces=(),
    )
    with pytest.raises(FileExistsError):
        export_current_surface(
            _surface(),
            output,
            semantic_palette={0: (96, 96, 96), 2: (1, 2, 3), 4: (4, 5, 6)},
            owner_id_table={0: "background", 11: "ovimap:11", 13: "ovimap:13"},
            source_surfaces=(),
        )


def test_export_requires_traceable_current_owner_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="owner ID table"):
        export_current_surface(
            _surface(),
            tmp_path / "surface",
            semantic_palette={0: (96, 96, 96), 2: (1, 2, 3), 4: (4, 5, 6)},
            owner_id_table={0: "background", 11: "ovimap:11"},
            source_surfaces=(),
        )
