#!/usr/bin/env python3
"""Render deterministic fixed-view plates from saved V3 dense predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_INSTANCE_COLORS = np.asarray(
    [
        [230, 159, 0],
        [86, 180, 233],
        [0, 158, 115],
        [240, 228, 66],
        [0, 114, 178],
        [213, 94, 0],
        [204, 121, 167],
        [34, 34, 34],
    ],
    dtype=np.uint8,
)
_BACKGROUND = np.asarray([242, 242, 242], dtype=np.uint8)
_AMBIGUOUS = np.asarray([166, 166, 166], dtype=np.uint8)


class QualitativeRenderError(ValueError):
    """Raised when a qualitative source does not match its declared case."""


def _file_record(path: Path, *, recorded_path: str | None = None) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": recorded_path if recorded_path is not None else path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _labels(value: object, *, row_count: int, label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (row_count,) or not np.issubdtype(result.dtype, np.integer):
        raise QualitativeRenderError(f"{label} must contain one integer per point")
    return result.astype(np.int64, copy=False)


def ground_truth_labels_for_points(
    *,
    points_xyz: np.ndarray,
    targets: Sequence[object],
    voxel_size_m: float,
) -> np.ndarray:
    """Assign evaluator GT IDs to dense points using its floor-quantized grid."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise QualitativeRenderError("points_xyz must have finite shape (N, 3)")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise QualitativeRenderError("voxel_size_m must be finite and positive")
    voxel_owner: dict[tuple[int, int, int], int] = {}
    for target in targets:
        instance_id = getattr(target, "instance_id", None)
        voxels = getattr(target, "voxels", None)
        if type(instance_id) is not int or instance_id <= 0 or not isinstance(
            voxels, (set, frozenset)
        ):
            raise QualitativeRenderError("ground-truth target is invalid")
        for voxel in voxels:
            key = tuple(int(axis) for axis in voxel)
            previous = voxel_owner.get(key)
            if previous is None:
                voxel_owner[key] = instance_id
            elif previous != instance_id:
                voxel_owner[key] = -2
    quantized = np.floor(points / float(voxel_size_m)).astype(np.int64)
    return np.fromiter(
        (voxel_owner.get(tuple(int(axis) for axis in row), -1) for row in quantized),
        dtype=np.int64,
        count=len(points),
    )


def _projection_pixels(points: np.ndarray, panel_size: int) -> tuple[np.ndarray, np.ndarray]:
    xy = points[:, :2].astype(np.float64, copy=False)
    lower = np.min(xy, axis=0)
    upper = np.max(xy, axis=0)
    extent = upper - lower
    extent[extent == 0.0] = 1.0
    padding = extent * 0.025
    lower -= padding
    upper += padding
    normalized = (xy - lower) / (upper - lower)
    columns = np.clip(np.rint(normalized[:, 0] * (panel_size - 1)), 0, panel_size - 1)
    rows = np.clip(
        np.rint((1.0 - normalized[:, 1]) * (panel_size - 1)), 0, panel_size - 1
    )
    return rows.astype(np.int64), columns.astype(np.int64)


def _visible_rows(points: np.ndarray, rows: np.ndarray, columns: np.ndarray, panel_size: int) -> np.ndarray:
    flat = rows * panel_size + columns
    order = np.lexsort((np.arange(len(points), dtype=np.int64), points[:, 2], flat))
    ordered_flat = flat[order]
    keep = np.ones(len(order), dtype=np.bool_)
    keep[:-1] = ordered_flat[:-1] != ordered_flat[1:]
    return order[keep]


def _instance_rgb(labels: np.ndarray) -> np.ndarray:
    colors = np.broadcast_to(_BACKGROUND, (len(labels), 3)).copy()
    colors[labels == -2] = _AMBIGUOUS
    foreground = labels >= 0
    colors[foreground] = _INSTANCE_COLORS[labels[foreground] % len(_INSTANCE_COLORS)]
    return colors


def _raster(
    colors: np.ndarray,
    *,
    rows: np.ndarray,
    columns: np.ndarray,
    visible: np.ndarray,
    panel_size: int,
) -> Image.Image:
    canvas = np.broadcast_to(_BACKGROUND, (panel_size, panel_size, 3)).copy()
    canvas[rows[visible], columns[visible]] = colors[visible]
    image = Image.fromarray(canvas, mode="RGB")
    return image.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST)


def render_topdown_plate(
    *,
    points_xyz: np.ndarray,
    camera_rgb_uint8: np.ndarray,
    ground_truth_labels: np.ndarray,
    prediction_labels: Mapping[str, np.ndarray],
    output_path: str | Path,
    title: str,
    panel_size: int = 320,
) -> dict[str, Any]:
    """Render RGB support, GT, and saved prediction ownership in one fixed view."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
        or not np.all(np.isfinite(points))
    ):
        raise QualitativeRenderError("points_xyz must have non-empty finite shape (N, 3)")
    rgb = np.asarray(camera_rgb_uint8)
    if rgb.shape != (len(points), 3) or rgb.dtype != np.uint8:
        raise QualitativeRenderError("camera RGB must be uint8 with shape (N, 3)")
    gt = _labels(ground_truth_labels, row_count=len(points), label="ground-truth labels")
    if not isinstance(prediction_labels, Mapping) or not prediction_labels:
        raise QualitativeRenderError("prediction labels must be a non-empty mapping")
    predictions = {
        str(name): _labels(value, row_count=len(points), label="prediction labels")
        for name, value in prediction_labels.items()
    }
    if any(not name for name in predictions):
        raise QualitativeRenderError("prediction panel names must be non-empty")
    if type(panel_size) is not int or panel_size < 64:
        raise QualitativeRenderError("panel_size must be an integer of at least 64")
    if not isinstance(title, str) or not title:
        raise QualitativeRenderError("title must be non-empty")

    rows, columns = _projection_pixels(points, panel_size)
    visible = _visible_rows(points, rows, columns, panel_size)
    panels = {"RGB support": rgb, "GT": _instance_rgb(gt)}
    panels.update({name: _instance_rgb(labels) for name, labels in predictions.items()})

    gap = 10
    outer = 14
    title_height = 30
    label_height = 28
    width = outer * 2 + len(panels) * panel_size + (len(panels) - 1) * gap
    height = outer * 2 + title_height + panel_size + label_height
    plate = Image.new("RGB", (width, height), tuple(int(v) for v in _BACKGROUND))
    draw = ImageDraw.Draw(plate)
    font = ImageFont.load_default(size=16)
    label_bbox = font.getbbox("RGB support")
    label_text_height = label_bbox[3] - label_bbox[1]
    draw.text((outer, outer), title, fill=(34, 34, 34), font=font)
    top = outer + title_height
    for index, (name, colors) in enumerate(panels.items()):
        left = outer + index * (panel_size + gap)
        plate.paste(
            _raster(
                colors,
                rows=rows,
                columns=columns,
                visible=visible,
                panel_size=panel_size,
            ),
            (left, top),
        )
        draw.text((left, top + panel_size + 5), name, fill=(34, 34, 34), font=font)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    plate.save(temporary, format="PNG", optimize=False, compress_level=9)
    os.replace(temporary, destination)
    return {
        "panels": list(panels),
        "point_count": len(points),
        "visible_pixel_count": len(visible),
        "projection": "XY_TOPDOWN_MAX_Z",
        "prediction_color_semantics": "PANEL_LOCAL_INSTANCE_ID",
        "ground_truth_ambiguous_label": -2,
        "ground_truth_ambiguous_point_count": int(np.count_nonzero(gt == -2)),
        "ground_truth_used_for_prediction": False,
        "panel_size_px": panel_size,
        "label_text_height_px": label_text_height,
    }


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualitativeRenderError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise QualitativeRenderError(f"{label} must contain an object")
    return value


def _load_run_labels(
    *, evaluation_root: Path, run_id: str, pair_id: str, visit_id: int, row_count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    root = evaluation_root / "evaluation_runs" / run_id
    summary_path = root / "summary.json"
    predictions_path = root / "predictions.npz"
    summary = _json_object(summary_path, label=f"{run_id} summary")
    if (
        summary.get("status") != "PASS"
        or summary.get("run_id") != run_id
        or summary.get("pair_id") != pair_id
    ):
        raise QualitativeRenderError(f"{run_id} summary identity is invalid")
    expected = summary.get("artifacts", {}).get("predictions", {})
    observed = _file_record(predictions_path, recorded_path="predictions.npz")
    if expected != observed:
        raise QualitativeRenderError(f"{run_id} prediction binding mismatch")
    try:
        with np.load(predictions_path, allow_pickle=False) as arrays:
            labels = _labels(
                arrays[f"visit{visit_id}_owner_instance_indices"],
                row_count=row_count,
                label="prediction labels",
            ).copy()
    except (OSError, KeyError, ValueError) as error:
        raise QualitativeRenderError(f"{run_id} predictions are invalid") from error
    return labels, {
        "run_id": run_id,
        "method_id": summary.get("method_id"),
        "checkpoint_id": summary.get("checkpoint_id"),
        "training_updates": summary.get("training_updates"),
        "summary": _file_record(summary_path, recorded_path=f"{run_id}/summary.json"),
        "predictions": {
            **observed,
            "path": f"{run_id}/predictions.npz",
        },
    }


def render_cases(
    *,
    cases_path: str | Path,
    evaluation_root: str | Path,
    runtime_root: str | Path,
    output_root: str | Path,
    panel_size: int = 320,
) -> dict[str, Any]:
    from scripts.evaluation.run_ovi_observation_query import _load_dense_inputs

    cases_source = Path(cases_path)
    cases = _json_object(cases_source, label="qualitative case manifest")
    if cases.get("schema_version") != 1 or not isinstance(cases.get("cases"), list):
        raise QualitativeRenderError("qualitative case manifest schema is invalid")
    eval_root = Path(evaluation_root).absolute()
    local_runtime_root = Path(runtime_root).absolute()
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases["cases"]:
        if not isinstance(case, Mapping):
            raise QualitativeRenderError("qualitative case must be an object")
        case_id = str(case.get("case_id", ""))
        pair_id = str(case.get("pair_id", ""))
        visit_id = case.get("visit_id")
        runtime_name = str(case.get("runtime", ""))
        raw_runs = case.get("runs")
        if (
            not case_id
            or not pair_id
            or visit_id not in (0, 1)
            or not runtime_name
            or not isinstance(raw_runs, list)
            or not raw_runs
        ):
            raise QualitativeRenderError("qualitative case fields are invalid")
        runtime_path = local_runtime_root / runtime_name
        runtime = _json_object(runtime_path, label=f"{case_id} runtime")
        legacy = runtime.get("assets", {}).get("legacy_dense_config")
        if not isinstance(legacy, str):
            raise QualitativeRenderError(f"{case_id} runtime lacks legacy dense config")
        pair, _bundle, ground_truth = _load_dense_inputs(Path(legacy))
        if pair.pair_id != pair_id or ground_truth.pair_id != pair_id:
            raise QualitativeRenderError(f"{case_id} dense inputs name another pair")
        visit = pair.visits[int(visit_id)]
        prediction_labels: dict[str, np.ndarray] = {}
        source_runs = []
        for raw_run in raw_runs:
            if not isinstance(raw_run, Mapping):
                raise QualitativeRenderError("qualitative run must be an object")
            label = str(raw_run.get("label", ""))
            run_id = str(raw_run.get("run_id", ""))
            if not label or not run_id or label in prediction_labels:
                raise QualitativeRenderError("qualitative run fields are invalid")
            labels, source = _load_run_labels(
                evaluation_root=eval_root,
                run_id=run_id,
                pair_id=pair_id,
                visit_id=int(visit_id),
                row_count=visit.point_count,
            )
            prediction_labels[label] = labels
            source_runs.append(source)
        camera_rgb = visit.camera_rgb_uint8.copy()
        camera_rgb[~visit.appearance_valid] = _BACKGROUND
        gt_labels = ground_truth_labels_for_points(
            points_xyz=visit.points_xyz,
            targets=ground_truth.visits[int(visit_id)],
            voxel_size_m=ground_truth.voxel_size_m,
        )
        figure_path = destination / f"{case_id}.png"
        render_record = render_topdown_plate(
            points_xyz=visit.points_xyz,
            camera_rgb_uint8=camera_rgb,
            ground_truth_labels=gt_labels,
            prediction_labels=prediction_labels,
            output_path=figure_path,
            title=f"{pair_id} / visit {visit_id}",
            panel_size=panel_size,
        )
        records.append(
            {
                "case_id": case_id,
                "pair_id": pair_id,
                "visit_id": visit_id,
                "selection": case.get("selection"),
                "figure": _file_record(figure_path),
                "render": render_record,
                "runtime": _file_record(runtime_path, recorded_path=runtime_name),
                "source_runs": source_runs,
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_id": "OVI_OBSERVATION_QUERY_V3_QUALITATIVE_V1",
        "status": "PASS",
        "case_manifest": _file_record(cases_source, recorded_path=cases_source.name),
        "color_palette": "OKABE_ITO_PANEL_LOCAL",
        "claim_boundary": (
            "Qualitative fixed-view inspection only; GT did not alter saved predictions."
        ),
        "cases": records,
    }
    manifest_path = destination / "manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--panel-size", type=int, default=320)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    render_cases(
        cases_path=args.cases,
        evaluation_root=args.evaluation_root,
        runtime_root=args.runtime_root,
        output_root=args.output,
        panel_size=args.panel_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "QualitativeRenderError",
    "ground_truth_labels_for_points",
    "render_cases",
    "render_topdown_plate",
]
