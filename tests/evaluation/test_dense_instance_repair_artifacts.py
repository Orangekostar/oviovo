from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from src.evaluation.dense_instance_repair_artifacts import (
    export_dense_method_artifacts,
)
from src.evaluation.dense_instance_repair_metrics import build_p2_method_view
from src.evaluation.ovi_pair_views import OviObjectPairView
from src.oviv2.rescene_dense_instance_readout import (
    OWNER_BACKGROUND,
    OWNER_UNKNOWN,
)
from tests.evaluation.test_dense_instance_repair_metrics import _p2_readout, _visit


def _pair() -> OviObjectPairView:
    return OviObjectPairView(
        pair_id="dense-pair",
        visits=(_visit(0), _visit(1)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dense_artifacts_preserve_rows_and_encode_all_owner_states(tmp_path: Path) -> None:
    pair = _pair()
    view = build_p2_method_view(pair, _p2_readout(pair))

    first = export_dense_method_artifacts(
        pair, view, tmp_path / "first", preview_width=64, preview_height=48
    )
    second = export_dense_method_artifacts(
        pair, view, tmp_path / "second", preview_width=64, preview_height=48
    )

    first_ply = first.output_dir / "t0" / "instances.ply"
    second_ply = second.output_dir / "t0" / "instances.ply"
    vertices = PlyData.read(first_ply)["vertex"].data
    np.testing.assert_array_equal(
        np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
        pair.visits[0].points_xyz,
    )
    np.testing.assert_array_equal(vertices["source_vertex_index"], np.arange(5))
    np.testing.assert_array_equal(vertices["instance_index"], [1, 2, 1, 3, 0])
    np.testing.assert_array_equal(
        vertices["owner_source"],
        view.owner_source_codes[0],
    )
    assert vertices["owner_source"][4] == OWNER_BACKGROUND
    current = PlyData.read(first.output_dir / "current" / "instances.ply")["vertex"].data
    np.testing.assert_array_equal(
        np.column_stack((current["x"], current["y"], current["z"])),
        pair.visits[1].points_xyz,
    )
    assert current["owner_source"][4] == OWNER_UNKNOWN
    assert _sha256(first_ply) == _sha256(second_ply)


def test_dense_artifact_manifest_binds_deterministic_colors_and_outputs(
    tmp_path: Path,
) -> None:
    pair = _pair()
    view = build_p2_method_view(pair, _p2_readout(pair))

    exported = export_dense_method_artifacts(
        pair, view, tmp_path / "artifacts", preview_width=64, preview_height=48
    )
    manifest = json.loads(exported.manifest.read_text(encoding="utf-8"))

    assert manifest["status"] == "DENSE_INSTANCE_ARTIFACT_PASS"
    assert manifest["method_id"] == "P2"
    assert manifest["method_view_sha256"] == view.content_sha256()
    assert manifest["geometry"] == {
        "xyz_changed": False,
        "point_order_changed": False,
        "point_count_changed": False,
    }
    assert manifest["instance_encoding"]["instance_index_zero"] == (
        "background_or_unknown"
    )
    assert manifest["instance_encoding"]["owner_source_codes"]["4"] == "unknown"
    assert len(manifest["candidate_colors"]["t0"]) == 3
    assert set(manifest["outputs"]) == {
        f"{scope}_{kind}"
        for scope in ("t0", "t1", "current")
        for kind in ("instance_ply", "instance_preview", "rgb_preview")
    }
    assert all(
        (exported.output_dir / record["path"]).is_file()
        for record in manifest["outputs"].values()
    )
