#!/usr/bin/env python3
"""Evaluate a deterministic voxel-selected readout of a CROVE fine surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_replica import (  # noqa: E402
    NON_INSTANCE_CLASSES,
    load_replica_ground_truth,
)
from scripts.evaluation.run_crove_fine_current_map import (  # noqa: E402
    _configured_path,
    _headline,
    _json,
    _sha256,
    load_ovimap_semantic_mapping,
)
from src.evaluation.oviv2_replica import (  # noqa: E402
    EntityEvaluationInfo,
    evaluate_replica_voxel_map,
)
from src.oviv2.fine_surface_io import select_source_surface_voxels  # noqa: E402
from src.oviv2.meshing import LabeledMesh  # noqa: E402


def evaluate_control(
    *,
    config_path: Path,
    static_run: Path,
    voxel_size_m: float,
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    if config_path.is_symlink() or not config_path.is_file():
        raise FileNotFoundError(config_path)
    if static_run.is_symlink() or not static_run.is_dir():
        raise FileNotFoundError(static_run)

    config = _json(config_path)
    case = config["cases"]["replica_room0_static"]
    run_receipt = _json(static_run / "run_receipt.json")
    run_metrics = _json(static_run / "metrics_summary.json")
    surface_manifest = _json(
        static_run / "current_map" / "current_surface_manifest.json"
    )
    if run_receipt.get("status") != "PASS" or surface_manifest.get("status") != "PASS":
        raise ValueError("static source run must be PASS")
    sidecar = static_run / "current_map" / "current_surface.npz"
    expected_sidecar = surface_manifest["artifacts"]["current_surface.npz"]
    if sidecar.stat().st_size != int(expected_sidecar["byte_count"]):
        raise ValueError("current surface byte count differs from its manifest")
    sidecar_sha256 = _sha256(sidecar)
    if sidecar_sha256 != expected_sidecar["sha256"]:
        raise ValueError("current surface hash differs from its manifest")

    started = time.perf_counter()
    with np.load(sidecar, allow_pickle=False) as payload:
        surface_id = str(payload["surface_id"].item())
        vertices = payload["vertices_xyz"]
        current_valid = payload["current_valid"]
        current_rows = np.flatnonzero(current_valid)
        selected_local = select_source_surface_voxels(
            vertices[current_rows],
            voxel_size_m=voxel_size_m,
            origin_xyz=np.zeros(3, dtype=np.float64),
        )
        selected_rows = current_rows[selected_local]
        selection_seconds = time.perf_counter() - started
        selected_vertices = np.asarray(vertices[selected_rows], dtype=np.float32)
        semantic_ids = np.asarray(payload["semantic_ids"][selected_rows], dtype=np.int32)
        owner_ids = np.asarray(payload["owner_entity_ids"][selected_rows], dtype=np.int64)
        semantic_confidences = np.asarray(
            payload["semantic_confidences"][selected_rows], dtype=np.float32
        )
        owner_confidences = np.asarray(
            payload["owner_confidences"][selected_rows], dtype=np.float32
        )

    benchmark = _json(_configured_path(case["benchmark_manifest"]))
    vocabulary = tuple(str(value) for value in benchmark["vocabulary"]["classes"])
    semantic_mapping = load_ovimap_semantic_mapping(
        _configured_path(case["ovimap_semantic_mapping"]),
        source_vocabulary=tuple(str(value) for value in case["ovimap_source_vocabulary"]),
        target_vocabulary=vocabulary,
    )
    scene = next(
        value for value in benchmark["scenes"] if value["scene"] == case["scene"]
    )
    gt_root = (
        Path(benchmark["ground_truth_root"])
        / scene["ground_truth_scene"]
        / "habitat"
    )
    class_to_id = {name: index + 1 for index, name in enumerate(vocabulary)}
    ground_truth = load_replica_ground_truth(
        gt_root / "mesh_semantic.ply",
        gt_root / "info_semantic.json",
        class_to_id=class_to_id,
        aliases={str(key): str(value) for key, value in benchmark["aliases"].items()},
    )
    entity_info = [
        EntityEvaluationInfo(
            entity_id=int(owner_id),
            semantic_id=int(semantic_mapping[int(owner_id)][0]),
            accepted_view_count=2,
            semantic_confidence=float(semantic_mapping[int(owner_id)][1]),
        )
        for owner_id in np.unique(owner_ids)
        if int(owner_id) in semantic_mapping and semantic_mapping[int(owner_id)][0] > 0
    ]
    mesh = LabeledMesh(
        vertices_xyz=selected_vertices,
        triangles=np.empty((0, 3), dtype=np.int64),
        colors_rgb=np.zeros((len(selected_vertices), 3), dtype=np.float32),
        semantic_ids=semantic_ids,
        entity_ids=owner_ids,
        semantic_confidence=semantic_confidences,
        ownership_confidence=owner_confidences,
    )
    evaluation_started = time.perf_counter()
    metrics = evaluate_replica_voxel_map(
        mesh,
        ground_truth,
        entity_info,
        valid_semantic_ids=set(class_to_id.values()),
        instance_semantic_ids={
            semantic_id
            for name, semantic_id in class_to_id.items()
            if name not in NON_INSTANCE_CLASSES
        },
        min_instance_vertices=int(case["min_instance_vertices"]),
        distance_threshold_m=float(
            benchmark["protocol"]["geometry_primary_threshold_m"]
        ),
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    headline = _headline(metrics)
    selected_strategy = str(run_metrics["selected_strategy"])
    reference = {
        key: float(value)
        for key, value in run_metrics["trials"][selected_strategy].items()
    }
    selected_sha256 = hashlib.sha256(
        np.asarray(selected_rows, dtype="<i8").tobytes()
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "condition": "SOURCE_SURFACE_VOXEL_SELECTION_NOT_RECONSTRUCTION",
        "protocol_id": "crove_fine_current_map_v1_replica_room0_2cm_source_selection",
        "scene": case["scene"],
        "split_role": scene["split_role"],
        "source_surface_id": surface_id,
        "source_surface_voxel_size_m": 0.01,
        "selection_voxel_size_m": float(voxel_size_m),
        "selection_origin_xyz_m": [0.0, 0.0, 0.0],
        "selection_index_rule": "floor_world_coordinate_first_source_row",
        "source_current_vertex_count": len(current_rows),
        "selected_vertex_count": len(selected_rows),
        "selected_vertex_ratio": float(len(selected_rows) / len(current_rows)),
        "selected_rows_sha256": selected_sha256,
        "selected_evaluation_array_bytes": int(
            selected_vertices.nbytes
            + semantic_ids.nbytes
            + owner_ids.nbytes
            + semantic_confidences.nbytes
            + owner_confidences.nbytes
        ),
        "metrics": headline,
        "reference_1cm_metrics": reference,
        "delta_2cm_minus_1cm": {
            key: float(headline[key] - reference[key]) for key in headline
        },
        "timing_seconds": {
            "selection": selection_seconds,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - started,
        },
        "ground_truth_role": "evaluator_only",
        "ground_truth_used_for_selection": False,
        "source_binding": {
            "path": str(sidecar),
            "byte_count": sidecar.stat().st_size,
            "sha256": sidecar_sha256,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--static-run", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_control(
        config_path=args.config,
        static_run=args.static_run,
        voxel_size_m=args.voxel_size_m,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
