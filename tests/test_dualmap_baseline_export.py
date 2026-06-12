"""Tests for exporting OVIOVO results in DualMap benchmark layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from export_dualmap_baseline_bundle import (
    dualmap_label_color,
    export_dualmap_baseline_bundle,
    high_contrast_label_color_table,
    read_vertex_ply,
)
from run_room0_full_eval import PLY_DTYPE, write_binary_ply


def _records(rows: list[tuple[float, float, float, int, int, int, int, int, float]]) -> np.ndarray:
    return np.asarray(rows, dtype=PLY_DTYPE)


def test_export_dualmap_baseline_bundle_groups_instances_and_eval_labels(tmp_path: Path) -> None:
    source_dir = tmp_path / "scene"
    exports_dir = source_dir / "exports"
    exports_dir.mkdir(parents=True)

    write_binary_ply(
        exports_dir / "room0_instance_map.ply",
        _records(
            [
                (0.0, 0.0, 0.0, 10, 20, 30, 2, 1, 0.9),
                (0.1, 0.0, 0.0, 11, 21, 31, 2, 1, 0.8),
                (1.0, 0.0, 0.0, 40, 50, 60, 5, 2, 0.7),
                (2.0, 0.0, 0.0, 90, 90, 90, -1, 0, 0.0),
            ]
        ),
    )
    write_binary_ply(
        exports_dir / "room0_instance_map_dense_surface.ply",
        _records(
            [
                (0.0, 1.0, 0.0, 10, 20, 30, 2, 1, 0.9),
                (1.0, 1.0, 0.0, 40, 50, 60, 5, 2, 0.7),
            ]
        ),
    )
    (source_dir / "final_object_semantic_audit.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"object_id": 2, "exported_label": "chair", "observation_count": 3},
                    {"object_id": 5, "exported_label": "rug", "observation_count": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = export_dualmap_baseline_bundle(
        source_exports_dir=exports_dir,
        audit_path=source_dir / "final_object_semantic_audit.json",
        output_dir=exports_dir / "dualmap_baseline_export",
    )

    benchmark_dir = exports_dir / "dualmap_baseline_export" / "benchmark"
    assert summary["instance_pcd"]["point_count"] == 3
    assert summary["instance_pcd"]["object_count"] == 2
    assert summary["instance_pcd"]["instance_ranges"] == [
        {"object_id": "object_000002", "start": 0, "count": 2, "label": "chair"},
        {"object_id": "object_000005", "start": 2, "count": 1, "label": "rug"},
    ]

    pred_info = json.loads((benchmark_dir / "pred_info.json").read_text(encoding="utf-8"))
    assert pred_info == {"mapping": {"chair": 1, "rug": 2}}

    final_eval = read_vertex_ply(benchmark_dir / "final_map_for_eval.ply")
    assert final_eval.dtype.names == ("x", "y", "z", "label")
    assert final_eval["label"].tolist() == [1, 1, 2]

    instance_summary = json.loads((benchmark_dir / "instance_summary.json").read_text(encoding="utf-8"))
    assert instance_summary[0]["bbox_min"] == [0.0, 0.0, 0.0]
    assert instance_summary[0]["bbox_max"] == [0.10000000149011612, 0.0, 0.0]
    assert instance_summary[0]["observed_count"] == 3
    assert instance_summary[1]["label"] == "rug"

    persisted_summary = json.loads((benchmark_dir / "benchmark_export_summary.json").read_text(encoding="utf-8"))
    assert persisted_summary["source_exports_dir"] == str(exports_dir)
    assert Path(persisted_summary["instance_pcd"]["path"]).name == "instance_pcd.ply"

    color_table = json.loads((benchmark_dir / "semantic_color_table.json").read_text(encoding="utf-8"))
    assert set(color_table) == {"chair", "rug"}
    assert color_table["chair"] == dualmap_label_color("chair")
    assert color_table["rug"] == dualmap_label_color("rug")

    instance_pcd = read_vertex_ply(benchmark_dir / "instance_pcd.ply")
    instance_colors = np.stack([instance_pcd["red"], instance_pcd["green"], instance_pcd["blue"]], axis=1)
    chair_u8 = [int(channel * 255.0) for channel in color_table["chair"]]
    rug_u8 = [int(channel * 255.0) for channel in color_table["rug"]]
    assert instance_colors[0].tolist() == chair_u8
    assert instance_colors[1].tolist() == chair_u8
    assert instance_colors[2].tolist() == rug_u8
    assert color_table["chair"] != color_table["rug"]

    instance_surface = read_vertex_ply(benchmark_dir / "instance_surface.ply")
    surface_colors = np.stack([instance_surface["red"], instance_surface["green"], instance_surface["blue"]], axis=1)
    assert surface_colors[0].tolist() == chair_u8
    assert surface_colors[1].tolist() == rug_u8

    viz_dir = exports_dir / "dualmap_baseline_export" / "viz_ply"
    assert (viz_dir / "object_000002.ply").exists()
    assert (viz_dir / "object_000005.ply").exists()

    viz_summary = json.loads((viz_dir / "viz_summary.json").read_text(encoding="utf-8"))
    assert Path(viz_summary["scene_ply"]).name == "scene_all_objects.ply"
    assert Path(viz_summary["scene_semantic_ply"]).name == "scene_all_objects_semantic_color.ply"
    assert Path(viz_summary["scene_high_contrast_semantic_ply"]).name == (
        "scene_all_objects_high_contrast_semantic_color.ply"
    )
    assert viz_summary["num_objects_with_points"] == 2
    assert viz_summary["num_per_object_ply"] == 2
    assert viz_summary["num_scene_points"] == 3
    assert viz_summary["label_color_table"] == color_table

    high_contrast_table = json.loads((viz_dir / "high_contrast_color_table.json").read_text(encoding="utf-8"))
    assert viz_summary["high_contrast_color_table"] == high_contrast_table
    assert high_contrast_table == high_contrast_label_color_table(["chair", "rug"])
    assert high_contrast_table["chair"] != color_table["chair"]
    assert high_contrast_table["rug"] != color_table["rug"]

    scene_all_objects = read_vertex_ply(viz_dir / "scene_all_objects.ply")
    scene_true_colors = np.stack(
        [scene_all_objects["red"], scene_all_objects["green"], scene_all_objects["blue"]],
        axis=1,
    )
    assert scene_true_colors.tolist() == [[10, 20, 30], [11, 21, 31], [40, 50, 60]]

    semantic_scene = read_vertex_ply(viz_dir / "scene_all_objects_semantic_color.ply")
    semantic_scene_colors = np.stack([semantic_scene["red"], semantic_scene["green"], semantic_scene["blue"]], axis=1)
    assert semantic_scene_colors[0].tolist() == chair_u8
    assert semantic_scene_colors[1].tolist() == chair_u8
    assert semantic_scene_colors[2].tolist() == rug_u8

    high_contrast_scene = read_vertex_ply(viz_dir / "scene_all_objects_high_contrast_semantic_color.ply")
    high_contrast_scene_colors = np.stack(
        [high_contrast_scene["red"], high_contrast_scene["green"], high_contrast_scene["blue"]],
        axis=1,
    )
    chair_high_contrast_u8 = [int(channel * 255.0) for channel in high_contrast_table["chair"]]
    rug_high_contrast_u8 = [int(channel * 255.0) for channel in high_contrast_table["rug"]]
    assert high_contrast_scene_colors[0].tolist() == chair_high_contrast_u8
    assert high_contrast_scene_colors[1].tolist() == chair_high_contrast_u8
    assert high_contrast_scene_colors[2].tolist() == rug_high_contrast_u8
    assert high_contrast_scene_colors[0].tolist() != semantic_scene_colors[0].tolist()
    assert high_contrast_scene_colors[2].tolist() != semantic_scene_colors[2].tolist()

    persisted_summary = json.loads((benchmark_dir / "benchmark_export_summary.json").read_text(encoding="utf-8"))
    assert Path(persisted_summary["viz_export"]["scene_semantic_ply"]).name == "scene_all_objects_semantic_color.ply"
    assert Path(persisted_summary["viz_export"]["scene_high_contrast_semantic_ply"]).name == (
        "scene_all_objects_high_contrast_semantic_color.ply"
    )
