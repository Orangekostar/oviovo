#!/usr/bin/env python3
"""Evaluate one external baseline map against frozen Replica ground truth."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.baselines.adapters import adapt_conceptgraphs, adapt_dualmap, adapt_ovimap
from src.evaluation.baselines.artifacts import write_baseline_artifact
from src.evaluation.baselines.contracts import RuntimeBreakdown
from src.evaluation.baselines.ovimap import load_instance_mesh, relative_similarity_labels
from src.evaluation.baselines.static_metrics import evaluate_static_snapshot
from src.evaluation.contracts import GroundTruthSnapshot

NON_INSTANCE_CLASSES = {"ceiling", "floor", "wall"}


def _normalize_label(label: str, aliases: dict[str, str]) -> str:
    normalized = str(label).strip().lower().replace("_", "-")
    return aliases.get(normalized, normalized)


def load_replica_ground_truth(
    *,
    scene_id: str,
    mesh_path: Path,
    semantic_labels_path: Path,
    instance_labels_path: Path,
    semantic_info_path: Path,
    aliases: dict[str, str],
    timestamp: float,
) -> GroundTruthSnapshot:
    from plyfile import PlyData

    vertices = PlyData.read(mesh_path)["vertex"]
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    semantic_ids = np.loadtxt(semantic_labels_path, dtype=np.int32)
    instance_ids = np.loadtxt(instance_labels_path, dtype=np.int64)
    if len(points) != len(semantic_ids) or len(points) != len(instance_ids):
        raise ValueError("Replica GT mesh and label arrays must have equal length")

    semantic_info = json.loads(semantic_info_path.read_text(encoding="utf-8"))
    id_to_name = {
        int(item["id"]): _normalize_label(str(item["name"]), aliases)
        for item in semantic_info["classes"]
    }
    semantic_labels = np.asarray(
        [id_to_name.get(int(class_id), "__ignored__") for class_id in semantic_ids],
        dtype=object,
    )
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    instance_ids[instance_ids < 1000] = -1
    return GroundTruthSnapshot(
        scene_id=scene_id,
        timestamp=timestamp,
        points_xyz=points,
        semantic_labels=semantic_labels,
        instance_ids=instance_ids,
    )


def _dualmap_runtime(path: Path, map_dir: Path, frame_count: int) -> RuntimeBreakdown:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Step"]: float(row["Avg Time (s)"]) for row in csv.DictReader(handle)}
    total_per_frame = rows["Time Per Frame"]
    frontend_per_frame = min(rows.get("Process Detection", total_per_frame), total_per_frame)
    backend_per_frame = max(total_per_frame - frontend_per_frame, 0.0)
    final_map_mb = sum(item.stat().st_size for item in map_dir.iterdir() if item.is_file()) / 1e6
    return RuntimeBreakdown(
        frame_count=frame_count,
        frontend_s=frontend_per_frame * frame_count,
        backend_s=backend_per_frame * frame_count,
        final_map_mb=final_map_mb,
        source_labels={
            "frontend_s": "DualMap system_time.csv Process Detection",
            "backend_s": "DualMap system_time.csv Time Per Frame residual",
            "final_map_mb": "DualMap serialized map files",
        },
    )


def _elapsed_seconds(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
    if match is None:
        raise ValueError(f"elapsed wall-clock time missing from {log_path}")
    parts = [float(value) for value in match.group(1).split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"invalid elapsed wall-clock value in {log_path}")


def _peak_rss_gb(*log_paths: Path) -> float:
    peaks = []
    for log_path in log_paths:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        if match is None:
            raise ValueError(f"peak RSS missing from {log_path}")
        peaks.append(int(match.group(1)) / 1e6)
    return max(peaks)


def _peak_gpu_gb(*paths: Path) -> float:
    values = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields or fields[0].startswith("#") or len(fields) < 4:
                continue
            values.append(float(fields[3]) / 1000.0)
    if not values:
        raise ValueError("GPU memory samples missing from dmon logs")
    return max(values)


def _conceptgraphs_runtime(args: argparse.Namespace) -> RuntimeBreakdown:
    frontend_s = _elapsed_seconds(args.detection_log)
    backend_s = _elapsed_seconds(args.mapping_log)
    return RuntimeBreakdown(
        frame_count=args.frame_count,
        frontend_s=frontend_s,
        backend_s=backend_s,
        peak_gpu_gb=_peak_gpu_gb(*args.gpu_dmon),
        peak_ram_gb=_peak_rss_gb(args.detection_log, args.mapping_log),
        final_map_mb=args.map_file.stat().st_size / 1e6,
        source_labels={
            "frontend_s": "ConceptGraphs detector process wall clock",
            "backend_s": "ConceptGraphs mapping process wall clock",
            "peak_gpu_gb": "exclusive-device nvidia-smi dmon fb samples",
            "peak_ram_gb": "GNU time process peak RSS",
            "final_map_mb": "ConceptGraphs serialized final map",
        },
    )


def _ovimap_runtime(args: argparse.Namespace) -> RuntimeBreakdown:
    return RuntimeBreakdown(
        frame_count=args.frame_count,
        frontend_s=_elapsed_seconds(args.frontend_log),
        backend_s=_elapsed_seconds(args.backend_log),
        peak_gpu_gb=_peak_gpu_gb(*args.gpu_dmon),
        peak_ram_gb=_peak_rss_gb(args.frontend_log, args.backend_log),
        final_map_mb=(args.instances_file.stat().st_size + args.instance_mesh.stat().st_size) / 1e6,
        source_labels={
            "frontend_s": "OVI-MAP official CropFormer process wall clock",
            "backend_s": "OVI-MAP mapping process wall clock",
            "peak_gpu_gb": "exclusive-device nvidia-smi dmon fb samples",
            "peak_ram_gb": "GNU time process peak RSS",
            "final_map_mb": "OVI-MAP instance feature pickle and instance mesh",
        },
    )


def _load_dualmap(args: argparse.Namespace, manifest: dict[str, Any]):
    import open3d as o3d

    objects = []
    for path in sorted(args.map_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            objects.append(pickle.load(handle))
    class_names = {
        int(key): _normalize_label(str(value), manifest["aliases"])
        for key, value in json.loads(args.class_names.read_text(encoding="utf-8")).items()
    }
    _classify_dualmap_objects(objects, class_names, args.clip_weight, args.device)
    background = np.asarray(o3d.io.read_point_cloud(str(args.layout)).points, dtype=np.float32)
    return adapt_dualmap(
        objects,
        class_id_names=class_names,
        scene_id=args.scene_id,
        timestamp=args.timestamp,
        upstream_commit=args.upstream_commit,
        runtime=_dualmap_runtime(args.runtime_csv, args.map_dir, args.frame_count),
        background_xyz=background,
        semantic_label_source="method_output.clip_ft MobileCLIP-S2 zero-shot",
    )


def _classify_dualmap_objects(
    objects: list[Any],
    class_names: dict[int, str],
    clip_weight: Path,
    device: str,
) -> None:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP-S2",
        pretrained=str(clip_weight),
    )
    model = model.to(device).eval()
    from mobileclip.modules.common.mobileone import reparameterize_model

    model = reparameterize_model(model)
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
    ordered_ids = list(class_names)
    prompts = [
        template.format(class_names[class_id])
        for class_id in ordered_ids
        for template in ("{}", "There is the {} in the scene.")
    ]
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.reshape(len(ordered_ids), 2, -1).mean(dim=1)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    object_features = torch.as_tensor(
        np.vstack([obj.clip_ft for obj in objects]),
        dtype=torch.float32,
        device=device,
    )
    object_features /= object_features.norm(dim=-1, keepdim=True)
    predicted = torch.argmax(object_features @ text_features.T, dim=1).cpu().numpy()
    for obj, class_index in zip(objects, predicted, strict=True):
        obj.class_id = ordered_ids[int(class_index)]


def _classify_conceptgraphs_objects(
    objects: list[dict[str, Any]],
    class_names: tuple[str, ...],
    clip_weight: Path,
    device: str,
) -> tuple[str, ...]:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained=str(clip_weight),
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    with torch.no_grad():
        tokens = tokenizer([f"an image of {label}" for label in class_names]).to(device)
        text_features = model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        object_features = torch.as_tensor(
            np.vstack([obj["clip_ft"] for obj in objects]),
            dtype=torch.float32,
            device=device,
        )
        object_features /= object_features.norm(dim=-1, keepdim=True)
        predicted = torch.argmax(object_features @ text_features.T, dim=1).cpu().numpy()
    return tuple(class_names[int(index)] for index in predicted)


def _load_conceptgraphs(
    args: argparse.Namespace,
    semantic_vocabulary: tuple[str, ...],
):
    with gzip.open(args.map_file, "rb") as handle:
        payload = pickle.load(handle)
    objects = list(payload.get("objects", ()))
    if not objects:
        raise ValueError("ConceptGraphs map contains no objects")
    semantic_labels = _classify_conceptgraphs_objects(
        objects,
        semantic_vocabulary,
        args.clip_weight,
        args.device,
    )
    payload = dict(payload)
    payload["objects"] = objects
    return adapt_conceptgraphs(
        payload,
        scene_id=args.scene_id,
        timestamp=args.timestamp,
        upstream_commit=args.upstream_commit,
        runtime=_conceptgraphs_runtime(args),
        semantic_labels=semantic_labels,
        semantic_label_source="method_output.clip_ft ViT-H-14 zero-shot",
    )


def _load_ovimap(
    args: argparse.Namespace,
    semantic_vocabulary: tuple[str, ...],
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    with args.instances_file.open("rb") as handle:
        instances = pickle.load(handle)
    if not instances:
        raise ValueError("OVI-MAP instance feature file contains no instances")
    colors = {
        tuple(int(value) for value in np.asarray(instance["color"]).reshape(-1))
        for instance in instances.values()
    }
    points_by_color, background_xyz = load_instance_mesh(args.instance_mesh, colors)
    semantic_source = "method_output.feat SigLIP-L/16-384 official canonical zero-shot"
    artifact = adapt_ovimap(
        instances,
        points_by_color=points_by_color,
        background_xyz=background_xyz,
        scene_id=args.scene_id,
        timestamp=args.timestamp,
        upstream_commit=args.upstream_commit,
        runtime=_ovimap_runtime(args),
        semantic_label_source=semantic_source,
        protocol_notes=(
            "semantic classification uses the frozen Replica-41 vocabulary instead of upstream Replica-51",
            "class-agnostic AP uses equal confidence with deterministic entity-ID tie ordering",
        ),
    )

    eligible = [
        index
        for index, entity in enumerate(artifact.snapshot.entities)
        if entity.semantic_embedding is not None
        and int(entity.metadata.get("observation_count", 0)) >= 2
    ]
    if eligible:
        model = AutoModel.from_pretrained(
            str(args.siglip_model),
            local_files_only=True,
        ).eval().to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.siglip_model),
            local_files_only=True,
        )

        def encode_text(values: list[str]) -> np.ndarray:
            tokens = tokenizer(
                values,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(args.device)
            with torch.no_grad():
                return model.get_text_features(**tokens).float().cpu().numpy()

        entity_features = np.vstack(
            [artifact.snapshot.entities[index].semantic_embedding for index in eligible]
        )
        text_features = encode_text(list(semantic_vocabulary))
        canonical_features = encode_text(["object", "things", "stuff", "texture"])
        labels, scores = relative_similarity_labels(
            entity_features,
            text_features,
            canonical_features,
            semantic_vocabulary,
        )
        for index, label, score in zip(eligible, labels, scores, strict=True):
            entity = artifact.snapshot.entities[index]
            entity.semantic_label = label
            entity.metadata["semantic_match_score"] = score
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("conceptgraphs", "dualmap", "ovimap"), required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--timestamp", type=float, default=1990.0)
    parser.add_argument("--frame-count", type=int, default=200)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--instance-labels", type=Path, required=True)
    parser.add_argument("--semantic-info", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--runtime-csv", type=Path)
    parser.add_argument("--map-file", type=Path)
    parser.add_argument("--detection-log", type=Path)
    parser.add_argument("--mapping-log", type=Path)
    parser.add_argument("--frontend-log", type=Path)
    parser.add_argument("--backend-log", type=Path)
    parser.add_argument("--gpu-dmon", type=Path, action="append")
    parser.add_argument("--clip-weight", type=Path)
    parser.add_argument("--instances-file", type=Path)
    parser.add_argument("--instance-mesh", type=Path)
    parser.add_argument("--siglip-model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--upstream-commit",
        default="",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    aliases = {
        _normalize_label(key, {}): _normalize_label(value, {})
        for key, value in manifest["aliases"].items()
    }
    manifest["aliases"] = aliases
    semantic_vocabulary = tuple(
        _normalize_label(label, aliases) for label in manifest["vocabulary"]["classes"]
    )
    instance_vocabulary = tuple(
        label for label in semantic_vocabulary if label not in NON_INSTANCE_CLASSES
    )

    required_by_baseline = {
        "dualmap": ("map_dir", "class_names", "layout", "runtime_csv", "clip_weight"),
        "conceptgraphs": (
            "map_file",
            "detection_log",
            "mapping_log",
            "gpu_dmon",
            "clip_weight",
        ),
        "ovimap": (
            "instances_file",
            "instance_mesh",
            "frontend_log",
            "backend_log",
            "gpu_dmon",
            "siglip_model",
        ),
    }
    missing = [name for name in required_by_baseline[args.baseline] if getattr(args, name) is None]
    if missing:
        parser.error(f"{args.baseline} requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    if not args.upstream_commit:
        args.upstream_commit = {
            "dualmap": "157235ec49e6a1f439babbc571c4c02ad1f06aa9",
            "conceptgraphs": "93277a02bd89171f8121e84203121cf7af9ebb5d",
            "ovimap": "58a804e2d7c8cfac6040639701abeb9d45b86537",
        }[args.baseline]

    artifact_loaders = {
        "dualmap": lambda: _load_dualmap(args, manifest),
        "conceptgraphs": lambda: _load_conceptgraphs(args, semantic_vocabulary),
        "ovimap": lambda: _load_ovimap(args, semantic_vocabulary),
    }
    artifact = artifact_loaders[args.baseline]()
    ground_truth = load_replica_ground_truth(
        scene_id=args.scene_id,
        mesh_path=args.mesh,
        semantic_labels_path=args.semantic_labels,
        instance_labels_path=args.instance_labels,
        semantic_info_path=args.semantic_info,
        aliases=aliases,
        timestamp=args.timestamp,
    )
    metrics = evaluate_static_snapshot(
        artifact.snapshot,
        ground_truth,
        semantic_vocabulary=semantic_vocabulary,
        instance_vocabulary=instance_vocabulary,
        distance_threshold_m=0.05,
        min_instance_points=100,
    )
    artifact_paths = write_baseline_artifact(artifact, args.artifact_output)
    payload = {
        "method": artifact.metadata.to_json(),
        "scene_id": args.scene_id,
        "metrics": metrics,
        "runtime": artifact.runtime.to_json(),
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
