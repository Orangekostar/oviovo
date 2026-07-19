from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata


FORBIDDEN_NAMES = {
    "SystemState",
    "ObjectMap",
    "local_pcd",
    "DenseSurfaceMap",
    "tsdf_instance_map",
}


def test_oviv2_runtime_has_no_legacy_mutable_map_dependency() -> None:
    violations: list[str] = []
    for path in sorted(Path("src/oviv2").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            else:
                continue
            for name in names:
                if any(part in FORBIDDEN_NAMES for part in name.split(".")):
                    violations.append(f"{path}:{getattr(node, 'lineno', 0)}:{name}")
    assert violations == []


def test_snapshot_contains_no_dense_point_cloud_payload(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(
        snapshot_dir,
        VoxelSnapshotMetadata("fixture", 0, 0.0, 0, 0.05, 8),
        SparseTsdfVolume(),
        SparseEvidenceStore(),
        ReversibleOwnershipStore(),
    )
    forbidden_payloads: list[str] = []
    for path in snapshot_dir.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            for name in payload.files:
                array = payload[name]
                if array.ndim == 2 and array.shape[1:] == (3,) and any(
                    token in name.lower() for token in ("point", "cloud", "xyz")
                ):
                    forbidden_payloads.append(f"{path.name}:{name}")
    assert forbidden_payloads == []
