"""Adapters from upstream baseline outputs to neutral evaluation artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.evaluation.baselines.contracts import (
    BaselineArtifact,
    BaselineMetadata,
    RuntimeBreakdown,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot


def _embedding(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array if array.size else None


def adapt_conceptgraphs(
    payload: Mapping[str, Any],
    *,
    scene_id: str,
    timestamp: float,
    upstream_commit: str,
    runtime: RuntimeBreakdown,
    semantic_labels: Sequence[str | None] | None = None,
    semantic_scores: Sequence[float] | None = None,
    semantic_label_source: str = "method_output.class_name",
) -> BaselineArtifact:
    objects = tuple(payload.get("objects", ()))
    if semantic_labels is not None and len(semantic_labels) != len(objects):
        raise ValueError("ConceptGraphs semantic labels must match object count")
    if semantic_scores is not None and len(semantic_scores) != len(objects):
        raise ValueError("ConceptGraphs semantic scores must match object count")
    entities: list[EntityPrediction] = []
    for index, obj in enumerate(objects):
        labels = [str(label) for label in obj.get("class_name", ()) if str(label).strip()]
        native_label = Counter(labels).most_common(1)[0][0] if labels else None
        label = native_label if semantic_labels is None else semantic_labels[index]
        confidences = [float(value) for value in obj.get("conf", ())]
        native_score = sum(confidences) / len(confidences) if confidences else 0.0
        score = native_score if semantic_scores is None else float(semantic_scores[index])
        image_indices = [float(value) for value in obj.get("image_idx", ())]
        first_seen = min(image_indices) if image_indices else float(timestamp)
        last_seen = max(image_indices) if image_indices else float(timestamp)
        entities.append(
            EntityPrediction(
                entity_id=f"conceptgraphs:{index:06d}",
                points_xyz=np.asarray(obj["pcd_np"], dtype=np.float32),
                semantic_embedding=_embedding(obj.get("clip_ft")),
                semantic_label=label,
                semantic_score=score,
                lifecycle_state="active",
                first_seen=first_seen,
                last_seen=last_seen,
                metadata={
                    "semantic_label_source": semantic_label_source,
                    "observation_count": len(image_indices),
                },
            )
        )
    metadata = BaselineMetadata(
        "CONCEPTGRAPHS",
        "ConceptGraphs",
        "native",
        upstream_commit,
        "artifact-local",
        semantic_label_source=semantic_label_source,
    )
    snapshot = MapSnapshot(
        method=metadata.display_label,
        scene_id=scene_id,
        timestamp=timestamp,
        entities=entities,
        background_xyz=None,
        scope="current",
        runtime=runtime.to_snapshot_runtime(),
    )
    return BaselineArtifact(snapshot=snapshot, runtime=runtime, metadata=metadata)


def adapt_dualmap(
    objects: Sequence[Any],
    *,
    class_id_names: Mapping[int, str],
    scene_id: str,
    timestamp: float,
    upstream_commit: str,
    runtime: RuntimeBreakdown,
    background_xyz: np.ndarray | None = None,
    semantic_label_source: str = "method_output.class_id",
) -> BaselineArtifact:
    entities: list[EntityPrediction] = []
    for obj in objects:
        class_id = int(obj.class_id)
        semantic_label = class_id_names.get(class_id)
        entities.append(
            EntityPrediction(
                entity_id=str(obj.uid),
                points_xyz=np.asarray(obj.pcd.points, dtype=np.float32),
                semantic_embedding=_embedding(getattr(obj, "clip_ft", None)),
                semantic_label=None if semantic_label is None else str(semantic_label),
                semantic_score=1.0 if semantic_label is not None else 0.0,
                lifecycle_state="active",
                first_seen=0.0,
                last_seen=float(timestamp),
                metadata={
                    "semantic_label_source": semantic_label_source,
                    "class_id": class_id,
                    "observation_count": int(getattr(obj, "observed_num", 0)),
                },
            )
        )
    metadata = BaselineMetadata(
        "DUALMAP",
        "DualMap",
        "native",
        upstream_commit,
        "persistent-within-run",
        semantic_label_source=semantic_label_source,
    )
    snapshot = MapSnapshot(
        method=metadata.display_label,
        scene_id=scene_id,
        timestamp=timestamp,
        entities=entities,
        background_xyz=background_xyz,
        scope="current",
        runtime=runtime.to_snapshot_runtime(),
    )
    return BaselineArtifact(snapshot=snapshot, runtime=runtime, metadata=metadata)


def adapt_ovimap(
    instances: Mapping[int, Mapping[str, Any]],
    *,
    points_by_color: Mapping[tuple[int, int, int], np.ndarray],
    scene_id: str,
    timestamp: float,
    upstream_commit: str,
    runtime: RuntimeBreakdown,
    protocol_notes: Sequence[str] = (),
) -> BaselineArtifact:
    entities: list[EntityPrediction] = []
    for instance_id, instance in sorted(instances.items()):
        color_array = np.asarray(instance["color"], dtype=np.uint8).reshape(-1)
        if color_array.shape != (3,):
            raise ValueError("OVI-MAP instance color must have three channels")
        color = tuple(int(value) for value in color_array)
        features = np.asarray(instance["feat"], dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        visibility = np.asarray(instance.get("vis_area", ()), dtype=np.float32).reshape(-1)
        if len(visibility) == len(features) and len(visibility):
            features = features[-8:]
            weights = visibility[-8:]
            weights = weights / (float(weights.sum()) + 1e-6)
            embedding = (features * weights[:, None]).sum(axis=0)
        else:
            embedding = features.mean(axis=0)
        frames = [float(value) for value in instance.get("frame_id", ())]
        entities.append(
            EntityPrediction(
                entity_id=f"ovimap:{int(instance_id)}",
                points_xyz=np.asarray(points_by_color.get(color, np.empty((0, 3))), dtype=np.float32),
                semantic_embedding=embedding,
                semantic_label=None,
                semantic_score=1.0,
                lifecycle_state="active",
                first_seen=min(frames) if frames else float(timestamp),
                last_seen=max(frames) if frames else float(timestamp),
                metadata={
                    "semantic_label_source": "method_output.feat",
                    "instance_color_rgb": list(color),
                    "observation_count": len(frames),
                },
            )
        )
    metadata = BaselineMetadata(
        "OVIMAP",
        "OVI-MAP",
        "native",
        upstream_commit,
        "persistent-within-run",
        protocol_notes=tuple(protocol_notes),
    )
    snapshot = MapSnapshot(
        method=metadata.display_label,
        scene_id=scene_id,
        timestamp=timestamp,
        entities=entities,
        background_xyz=None,
        scope="current",
        runtime=runtime.to_snapshot_runtime(),
    )
    return BaselineArtifact(snapshot=snapshot, runtime=runtime, metadata=metadata)
