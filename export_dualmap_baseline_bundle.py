#!/usr/bin/env python3
"""Export OVIOVO room0 maps in DualMap benchmark bundle layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


OVIOVO_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("object_id", "<i4"),
        ("state_id", "u1"),
        ("support", "<f4"),
    ]
)

DUALMAP_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)

EVAL_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("label", "<i4"),
    ]
)

HIGH_CONTRAST_LABEL_COLORS = {
    "blanket": ("red", [0.95, 0.05, 0.05]),
    "blinds": ("lime", [0.10, 0.90, 0.05]),
    "book": ("orange", [1.00, 0.45, 0.00]),
    "cabinet": ("brown", [0.55, 0.25, 0.05]),
    "ceiling": ("violet", [0.55, 0.20, 1.00]),
    "chair": ("yellow", [1.00, 0.95, 0.00]),
    "cushion": ("hot pink", [1.00, 0.05, 0.55]),
    "door": ("cyan", [0.00, 0.90, 1.00]),
    "floor": ("deep purple", [0.35, 0.00, 0.85]),
    "indoor-plant": ("green", [0.00, 0.70, 0.10]),
    "lamp": ("gold", [1.00, 0.65, 0.00]),
    "monitor": ("blue", [0.00, 0.20, 1.00]),
    "picture": ("sky blue", [0.25, 0.65, 1.00]),
    "plate": ("mint", [0.00, 1.00, 0.55]),
    "rug": ("turquoise", [0.00, 0.75, 0.75]),
    "sofa": ("magenta", [0.85, 0.00, 1.00]),
    "stool": ("coral", [1.00, 0.25, 0.20]),
    "switch": ("white", [1.00, 1.00, 1.00]),
    "table": ("navy", [0.00, 0.05, 0.55]),
    "vase": ("royal blue", [0.05, 0.35, 1.00]),
    "window": ("aqua", [0.00, 1.00, 0.90]),
}

HIGH_CONTRAST_FALLBACK_PALETTE = [
    ("red", [0.95, 0.05, 0.05]),
    ("blue", [0.00, 0.20, 1.00]),
    ("green", [0.00, 0.70, 0.10]),
    ("yellow", [1.00, 0.95, 0.00]),
    ("magenta", [0.85, 0.00, 1.00]),
    ("cyan", [0.00, 0.90, 1.00]),
    ("orange", [1.00, 0.45, 0.00]),
    ("deep purple", [0.35, 0.00, 0.85]),
    ("lime", [0.10, 0.90, 0.05]),
    ("hot pink", [1.00, 0.05, 0.55]),
    ("turquoise", [0.00, 0.75, 0.75]),
    ("brown", [0.55, 0.25, 0.05]),
    ("gold", [1.00, 0.65, 0.00]),
    ("sky blue", [0.25, 0.65, 1.00]),
    ("coral", [1.00, 0.25, 0.20]),
    ("mint", [0.00, 1.00, 0.55]),
    ("navy", [0.00, 0.05, 0.55]),
    ("royal blue", [0.05, 0.35, 1.00]),
    ("aqua", [0.00, 1.00, 0.90]),
    ("violet", [0.55, 0.20, 1.00]),
    ("maroon", [0.60, 0.00, 0.15]),
    ("olive", [0.55, 0.60, 0.00]),
    ("white", [1.00, 1.00, 1.00]),
    ("black", [0.00, 0.00, 0.00]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-exports-dir",
        type=Path,
        required=True,
        help="OVIOVO room0 exports directory containing room0_instance_map.ply.",
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=None,
        help="final_object_semantic_audit.json. Defaults to ../final_object_semantic_audit.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to source exports dir / dualmap_baseline_export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_exports_dir = args.source_exports_dir.resolve()
    audit_path = args.audit_path.resolve() if args.audit_path else source_exports_dir.parent / "final_object_semantic_audit.json"
    output_dir = args.output_dir.resolve() if args.output_dir else source_exports_dir / "dualmap_baseline_export"
    summary = export_dualmap_baseline_bundle(
        source_exports_dir=source_exports_dir,
        audit_path=audit_path,
        output_dir=output_dir,
    )
    benchmark_dir = output_dir / "benchmark"
    print("Saved DualMap baseline export bundle:")
    print(f"- {benchmark_dir / 'instance_pcd.ply'}")
    print(f"- {benchmark_dir / 'instance_surface.ply'}")
    print(f"- {benchmark_dir / 'final_map_for_eval.ply'}")
    print(f"- {benchmark_dir / 'pred_info.json'}")
    print(f"- {output_dir / 'viz_ply' / 'scene_all_objects_semantic_color.ply'}")
    print(f"- {output_dir / 'viz_ply' / 'scene_all_objects_high_contrast_semantic_color.ply'}")
    print(f"- objects: {summary['instance_pcd']['object_count']}")
    print(f"- points: {summary['instance_pcd']['point_count']}")


def export_dualmap_baseline_bundle(
    *,
    source_exports_dir: Path,
    audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_exports_dir = Path(source_exports_dir)
    audit_path = Path(audit_path)
    output_dir = Path(output_dir)
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    audit_by_object = load_audit_by_object(audit_path)
    instance_records = read_vertex_ply(source_exports_dir / "room0_instance_map.ply")
    surface_path = source_exports_dir / "room0_instance_map_dense_surface.ply"
    surface_records = read_vertex_ply(surface_path) if surface_path.exists() else np.zeros(0, dtype=OVIOVO_VERTEX_DTYPE)
    labels = sorted(
        {
            object_label(int(object_id), audit_by_object)
            for object_id in np.unique(instance_records[instance_records["object_id"] >= 0]["object_id"])
        }
    )
    semantic_color_table = {label: dualmap_label_color(label) for label in labels}

    instance_export, instance_summaries, diagnostics = build_instance_export(
        instance_records,
        audit_by_object,
        semantic_color_table,
    )
    surface_export = build_surface_export(surface_records, audit_by_object, semantic_color_table)
    viz_export = write_viz_export(output_dir / "viz_ply", instance_records, audit_by_object, semantic_color_table)

    instance_pcd_path = benchmark_dir / "instance_pcd.ply"
    instance_surface_path = benchmark_dir / "instance_surface.ply"
    final_eval_path = benchmark_dir / "final_map_for_eval.ply"
    pred_info_path = benchmark_dir / "pred_info.json"
    summary_path = benchmark_dir / "instance_summary.json"
    diagnostics_path = benchmark_dir / "instance_diagnostics.json"
    color_table_path = benchmark_dir / "semantic_color_table.json"
    benchmark_summary_path = benchmark_dir / "benchmark_export_summary.json"

    write_dualmap_ascii_ply(instance_pcd_path, instance_export["records"])
    write_dualmap_ascii_ply(instance_surface_path, surface_export["records"])
    pred_info = write_eval_files(
        final_eval_path,
        pred_info_path,
        instance_export["records"],
        instance_export["instance_ranges"],
    )
    summary_path.write_text(json.dumps(instance_summaries, indent=2), encoding="utf-8")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    color_table_path.write_text(json.dumps(semantic_color_table, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "source_exports_dir": str(source_exports_dir),
        "source_audit_path": str(audit_path),
        "instance_pcd": {
            "path": str(instance_pcd_path),
            "point_count": int(len(instance_export["records"])),
            "object_count": int(len(instance_export["instance_ranges"])),
            "instance_ranges": instance_export["instance_ranges"],
        },
        "instance_surface": {
            "path": str(instance_surface_path),
            "point_count": int(len(surface_export["records"])),
            "object_count": int(len(surface_export["object_ids"])),
        },
        "instance_summary": str(summary_path),
        "instance_diagnostics": str(diagnostics_path),
        "semantic_color_table": str(color_table_path),
        "eval_export": {
            "final_map_for_eval": str(final_eval_path),
            "pred_info": str(pred_info_path),
            "mapping": pred_info["mapping"],
        },
        "viz_export": viz_export,
    }
    benchmark_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_audit_by_object(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(record["object_id"]): record for record in payload.get("objects", [])}


def build_instance_export(
    records: np.ndarray,
    audit_by_object: dict[int, dict[str, Any]],
    semantic_color_table: dict[str, list[int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    valid = records[records["object_id"] >= 0]
    chunks: list[np.ndarray] = []
    instance_ranges: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    offset = 0

    for object_id in sorted(int(value) for value in np.unique(valid["object_id"])):
        object_records = valid[valid["object_id"] == object_id]
        if len(object_records) == 0:
            continue
        label = object_label(object_id, audit_by_object)
        dualmap_records = to_dualmap_records(object_records, semantic_color_table[label])
        chunks.append(dualmap_records)
        instance_ranges.append(
            {
                "object_id": dualmap_object_id(object_id),
                "start": int(offset),
                "count": int(len(dualmap_records)),
                "label": label,
            }
        )
        summaries.append(instance_summary(object_id, label, object_records, audit_by_object))
        diagnostics.append(instance_diagnostics(object_id, label, object_records))
        offset += int(len(dualmap_records))

    merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=DUALMAP_VERTEX_DTYPE)
    return {"records": merged, "instance_ranges": instance_ranges}, summaries, diagnostics


def build_surface_export(
    records: np.ndarray,
    audit_by_object: dict[int, dict[str, Any]],
    semantic_color_table: dict[str, list[int]],
) -> dict[str, Any]:
    valid = records[records["object_id"] >= 0]
    chunks: list[np.ndarray] = []
    object_ids: list[int] = []
    for object_id in sorted(int(value) for value in np.unique(valid["object_id"])):
        object_records = valid[valid["object_id"] == object_id]
        if len(object_records) == 0:
            continue
        label = object_label(object_id, audit_by_object)
        chunks.append(to_dualmap_records(object_records, semantic_color_table.get(label, dualmap_label_color(label))))
        object_ids.append(object_id)
    merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=DUALMAP_VERTEX_DTYPE)
    return {"records": merged, "object_ids": object_ids}


def write_viz_export(
    viz_dir: Path,
    records: np.ndarray,
    audit_by_object: dict[int, dict[str, Any]],
    semantic_color_table: dict[str, list[float]],
) -> dict[str, Any]:
    viz_dir.mkdir(parents=True, exist_ok=True)
    valid = records[records["object_id"] >= 0]
    true_chunks: list[np.ndarray] = []
    semantic_chunks: list[np.ndarray] = []
    high_contrast_chunks: list[np.ndarray] = []
    object_entries: list[dict[str, Any]] = []
    labels = sorted({object_label(int(object_id), audit_by_object) for object_id in np.unique(valid["object_id"])})
    high_contrast_table = high_contrast_label_color_table(labels)
    high_contrast_color_names = high_contrast_label_color_names(labels)

    for object_id in sorted(int(value) for value in np.unique(valid["object_id"])):
        object_records = valid[valid["object_id"] == object_id]
        if len(object_records) == 0:
            continue
        label = object_label(object_id, audit_by_object)
        true_records = to_dualmap_true_records(object_records)
        semantic_records = to_dualmap_records(object_records, semantic_color_table.get(label, dualmap_label_color(label)))
        high_contrast_records = to_dualmap_records(object_records, high_contrast_table[label])
        object_name = dualmap_object_id(object_id)
        object_path = viz_dir / f"{object_name}.ply"
        write_dualmap_ascii_ply(object_path, true_records)
        true_chunks.append(true_records)
        semantic_chunks.append(semantic_records)
        high_contrast_chunks.append(high_contrast_records)
        object_entries.append(
            {
                "object_id": object_name,
                "source_object_id": int(object_id),
                "label": label,
                "point_count": int(len(object_records)),
                "path": str(object_path),
            }
        )

    merged_true = np.concatenate(true_chunks) if true_chunks else np.zeros(0, dtype=DUALMAP_VERTEX_DTYPE)
    merged_semantic = np.concatenate(semantic_chunks) if semantic_chunks else np.zeros(0, dtype=DUALMAP_VERTEX_DTYPE)
    merged_high_contrast = (
        np.concatenate(high_contrast_chunks) if high_contrast_chunks else np.zeros(0, dtype=DUALMAP_VERTEX_DTYPE)
    )

    scene_ply = viz_dir / "scene_all_objects.ply"
    scene_semantic_ply = viz_dir / "scene_all_objects_semantic_color.ply"
    scene_high_contrast_semantic_ply = viz_dir / "scene_all_objects_high_contrast_semantic_color.ply"
    viz_summary_path = viz_dir / "viz_summary.json"
    high_contrast_color_table_path = viz_dir / "high_contrast_color_table.json"
    write_dualmap_ascii_ply(scene_ply, merged_true)
    write_dualmap_ascii_ply(scene_semantic_ply, merged_semantic)
    write_dualmap_ascii_ply(scene_high_contrast_semantic_ply, merged_high_contrast)
    high_contrast_color_table_path.write_text(
        json.dumps(high_contrast_table, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "viz_dir": str(viz_dir),
        "scene_ply": str(scene_ply),
        "scene_semantic_ply": str(scene_semantic_ply),
        "scene_high_contrast_semantic_ply": str(scene_high_contrast_semantic_ply),
        "num_objects_with_points": int(len(object_entries)),
        "num_per_object_ply": int(len(object_entries)),
        "num_scene_points": int(len(merged_true)),
        "label_color_table": semantic_color_table,
        "high_contrast_color_table": high_contrast_table,
        "high_contrast_color_names": high_contrast_color_names,
        "high_contrast_color_table_path": str(high_contrast_color_table_path),
        "objects": object_entries,
    }
    viz_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["viz_summary"] = str(viz_summary_path)
    return summary


def object_label(object_id: int, audit_by_object: dict[int, dict[str, Any]]) -> str:
    record = audit_by_object.get(int(object_id), {})
    label = str(record.get("exported_label", "")).strip()
    return label or "unknown"


def dualmap_object_id(object_id: int) -> str:
    return f"object_{int(object_id):06d}"


def to_dualmap_records(records: np.ndarray, color: list[int]) -> np.ndarray:
    out = np.zeros(len(records), dtype=DUALMAP_VERTEX_DTYPE)
    for field in ["x", "y", "z"]:
        out[field] = records[field]
    color_u8 = dualmap_color_to_uint8(color)
    out["red"] = int(color_u8[0])
    out["green"] = int(color_u8[1])
    out["blue"] = int(color_u8[2])
    return out


def high_contrast_label_color_table(labels: list[str]) -> dict[str, list[float]]:
    return {label: color for label, _name, color in high_contrast_label_color_entries(labels)}


def high_contrast_label_color_names(labels: list[str]) -> dict[str, str]:
    return {label: name for label, name, _color in high_contrast_label_color_entries(labels)}


def high_contrast_label_color_entries(labels: list[str]) -> list[tuple[str, str, list[float]]]:
    clean_labels = sorted({str(label).strip() or "unknown" for label in labels})
    used_names = {HIGH_CONTRAST_LABEL_COLORS[label][0] for label in clean_labels if label in HIGH_CONTRAST_LABEL_COLORS}
    fallback_iter = (
        (name, color)
        for name, color in HIGH_CONTRAST_FALLBACK_PALETTE
        if name not in used_names
    )
    entries: list[tuple[str, str, list[float]]] = []
    for label in clean_labels:
        if label in HIGH_CONTRAST_LABEL_COLORS:
            name, color = HIGH_CONTRAST_LABEL_COLORS[label]
        else:
            name, color = next(fallback_iter)
        entries.append((label, name, color))
    return entries


def to_dualmap_true_records(records: np.ndarray) -> np.ndarray:
    out = np.zeros(len(records), dtype=DUALMAP_VERTEX_DTYPE)
    for field in ["x", "y", "z", "red", "green", "blue"]:
        out[field] = records[field]
    return out


def dualmap_label_color(label: str) -> list[float]:
    text = str(label).strip() or "unknown"
    digest = hashlib.md5(text.encode("utf-8")).digest()
    rgb = np.array([digest[0], digest[1], digest[2]], dtype=np.float32) / 255.0
    color = np.clip(0.35 + 0.65 * rgb, 0.0, 1.0)
    return [float(value) for value in color.tolist()]


def dualmap_color_to_uint8(color: list[float]) -> list[int]:
    values = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
    return [int(value * 255.0) for value in values.tolist()]


def instance_summary(
    object_id: int,
    label: str,
    records: np.ndarray,
    audit_by_object: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    points = points_from_records(records)
    bbox_min = points.min(axis=0).astype(float).tolist()
    bbox_max = points.max(axis=0).astype(float).tolist()
    centroid = points.mean(axis=0).astype(float).tolist()
    audit = audit_by_object.get(int(object_id), {})
    return {
        "object_id": dualmap_object_id(object_id),
        "source_object_id": int(object_id),
        "label": label,
        "label_type": "object_like",
        "observed_count": int(audit.get("observation_count", 0)),
        "geometry_point_count": int(len(records)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "centroid": centroid,
        "footprint_point_count": int(len(records)),
        "geometry_update_count": int(audit.get("update_count", audit.get("observation_count", 0))),
        "semantic_state": "exported",
        "lifecycle_state": str(audit.get("state", "unknown")),
        "interpretation": "oviovo_export",
    }


def instance_diagnostics(object_id: int, label: str, records: np.ndarray) -> dict[str, Any]:
    points = points_from_records(records)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    size = np.maximum(bbox_max - bbox_min, 0.0)
    bbox_volume = float(size[0] * size[1] * size[2])
    point_count = int(len(records))
    return {
        "object_id": dualmap_object_id(object_id),
        "source_object_id": int(object_id),
        "label": label,
        "instance_point_count": point_count,
        "observation_point_count": point_count,
        "soft_owned_surface_count": point_count,
        "hard_owned_surface_count": point_count,
        "instance_bbox_volume": bbox_volume,
        "geometry_completeness_proxy": float(point_count / max(1.0, bbox_volume * 1000.0)),
        "neighbor_overlap_ratio": 0.0,
        "geometry_update_events": 0,
    }


def points_from_records(records: np.ndarray) -> np.ndarray:
    return np.stack([records["x"], records["y"], records["z"]], axis=1).astype(np.float32)


def write_eval_files(
    ply_path: Path,
    pred_info_path: Path,
    records: np.ndarray,
    instance_ranges: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    labels = np.zeros(len(records), dtype=np.int32)
    mapping: dict[str, int] = {}
    next_id = 1
    for rec in instance_ranges:
        label = str(rec["label"])
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        start = int(rec["start"])
        count = int(rec["count"])
        labels[start : start + count] = mapping[label]

    out = np.zeros(len(records), dtype=EVAL_VERTEX_DTYPE)
    out["x"] = records["x"]
    out["y"] = records["y"]
    out["z"] = records["z"]
    out["label"] = labels
    write_binary_eval_ply(ply_path, out)

    payload = {"mapping": mapping}
    pred_info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_vertex_ply(path: Path) -> np.ndarray:
    path = Path(path)
    header, fmt, dtype, count = read_ply_layout(path)
    with path.open("rb") as handle:
        for _ in header:
            handle.readline()
        if fmt == "binary_little_endian":
            return np.fromfile(handle, dtype=dtype, count=count)
        if fmt == "ascii":
            return np.loadtxt(handle, dtype=dtype, max_rows=count)
    raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")


def read_ply_layout(path: Path) -> tuple[list[str], str, np.dtype, int]:
    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "ushort": "<u2",
        "uint16": "<u2",
    }
    with Path(path).open("rb") as handle:
        header = []
        fmt = ""
        count = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            raw = handle.readline()
            if not raw:
                raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
            line = raw.decode("ascii", errors="ignore").strip()
            header.append(line)
            parts = line.split()
            if parts[:2] == ["format", "ascii"]:
                fmt = "ascii"
            elif parts[:2] == ["format", "binary_little_endian"]:
                fmt = "binary_little_endian"
            elif parts[:2] == ["element", "vertex"]:
                in_vertex = True
                count = int(parts[2])
            elif in_vertex and parts and parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"] and len(parts) == 3:
                props.append((parts[2], type_map[parts[1]]))
            elif line == "end_header":
                break
    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported or missing PLY format in {path}")
    return header, fmt, np.dtype(props), count


def write_dualmap_ascii_ply(path: Path, records: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(records)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for row in records:
            handle.write(
                f"{float(row['x']):.6f} {float(row['y']):.6f} {float(row['z']):.6f} "
                f"{int(row['red'])} {int(row['green'])} {int(row['blue'])}\n"
            )


def write_binary_eval_ply(path: Path, records: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(records)}",
            "property float x",
            "property float y",
            "property float z",
            "property int label",
            "end_header",
            "",
        ]
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(records.astype(EVAL_VERTEX_DTYPE, copy=False).tobytes())


if __name__ == "__main__":
    main()
