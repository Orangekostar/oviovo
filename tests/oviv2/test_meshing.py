from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from src.oviv2.addressing import point_to_voxel
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.meshing import derive_labeled_mesh, write_labeled_mesh
from src.oviv2.ownership import ReversibleOwnershipStore

from tests.oviv2.test_geometry import _integrate_twice, _plane_frame


def _geometry() -> SparseTsdfVolume:
    depth, rgb, intrinsics = _plane_frame()
    geometry = SparseTsdfVolume()
    _integrate_twice(geometry, depth, rgb, intrinsics, np.eye(4))
    return geometry


def test_synthetic_plane_yields_shape_consistent_read_only_arrays() -> None:
    geometry = _geometry()
    mesh = derive_labeled_mesh(
        geometry,
        SparseEvidenceStore(),
        ReversibleOwnershipStore(),
    )
    vertex_count = mesh.vertices_xyz.shape[0]

    assert vertex_count > 0
    assert mesh.vertices_xyz.shape == (vertex_count, 3)
    assert mesh.triangles.ndim == 2 and mesh.triangles.shape[1] == 3
    assert mesh.colors_rgb.shape == (vertex_count, 3)
    assert mesh.semantic_ids.shape == (vertex_count,)
    assert mesh.entity_ids.shape == (vertex_count,)
    assert mesh.semantic_confidence.shape == (vertex_count,)
    assert mesh.ownership_confidence.shape == (vertex_count,)
    for array in vars(mesh).values():
        assert array.flags.c_contiguous
        assert not array.flags.writeable


def test_vertices_take_strongest_semantic_candidate_and_current_owner() -> None:
    geometry = _geometry()
    raw_vertices = geometry.extract_mesh().vertex.positions.numpy()
    target_key = point_to_voxel(raw_vertices[0], geometry.config.voxel_size_m)
    evidence = SparseEvidenceStore()
    evidence.update_semantic(target_key, 5, 1.0, 1)
    evidence.update_semantic(target_key, 3, 2.0, 2)
    evidence.update_entity(target_key, 9, 2.0, 0.0, 1.0, 2)
    ownership = ReversibleOwnershipStore()
    ownership.assign(target_key, 9, 0.8, 2)

    mesh = derive_labeled_mesh(geometry, evidence, ownership)
    keys = np.asarray(
        [point_to_voxel(vertex, geometry.config.voxel_size_m) for vertex in mesh.vertices_xyz]
    )
    labeled = np.all(keys == np.asarray(target_key), axis=1)

    assert labeled.any()
    assert np.all(mesh.semantic_ids[labeled] == 3)
    assert np.all(mesh.semantic_confidence[labeled] == pytest.approx(2.0 / 3.0))
    assert np.all(mesh.entity_ids[labeled] == 9)
    assert np.all(mesh.ownership_confidence[labeled] == pytest.approx(0.8))
    assert np.all(mesh.semantic_ids[~labeled] == 0)
    assert np.all(mesh.entity_ids[~labeled] == 0)


def test_derivation_and_write_do_not_mutate_voxel_state(tmp_path: Path) -> None:
    geometry = _geometry()
    key = point_to_voxel(
        geometry.extract_mesh().vertex.positions.numpy()[0],
        geometry.config.voxel_size_m,
    )
    evidence = SparseEvidenceStore()
    evidence.update_semantic(key, 2, 1.0, 1)
    evidence.update_entity(key, 4, 1.0, 0.0, 1.0, 1)
    ownership = ReversibleOwnershipStore()
    ownership.assign(key, 4, 0.7, 1)
    block_count = geometry.active_block_count
    semantic_before = evidence.semantic_candidates(key)
    entity_before = evidence.entity_candidates(key)
    owner_before = ownership.owner_of(key)

    mesh = derive_labeled_mesh(geometry, evidence, ownership)
    write_labeled_mesh(tmp_path / "mesh.ply", mesh)

    assert geometry.active_block_count == block_count
    assert evidence.semantic_candidates(key) == semantic_before
    assert evidence.entity_candidates(key) == entity_before
    assert ownership.owner_of(key) == owner_before


def test_written_ply_is_readable_and_contains_label_properties(tmp_path: Path) -> None:
    mesh = derive_labeled_mesh(
        _geometry(),
        SparseEvidenceStore(),
        ReversibleOwnershipStore(),
    )
    path = tmp_path / "mesh.ply"

    write_labeled_mesh(path, mesh)
    restored = o3d.t.io.read_triangle_mesh(str(path))
    header = path.read_bytes().split(b"end_header", 1)[0]

    assert restored.vertex.positions.shape[0] == mesh.vertices_xyz.shape[0]
    assert b"semantic_id" in header
    assert b"entity_id" in header
