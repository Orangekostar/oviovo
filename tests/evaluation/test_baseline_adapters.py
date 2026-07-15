from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.baselines.adapters import adapt_conceptgraphs, adapt_dualmap, adapt_ovimap
from src.evaluation.baselines.artifacts import write_baseline_artifact
from src.evaluation.baselines.contracts import (
    BaselineArtifact,
    BaselineMetadata,
    FrozenUpdateAudit,
    RuntimeBreakdown,
    evaluate_identity_assignments,
)


def _runtime() -> RuntimeBreakdown:
    return RuntimeBreakdown(
        frame_count=2,
        initialization_s=1.0,
        frontend_s=0.4,
        backend_s=0.2,
        maintenance_s=0.0,
        finalization_s=0.1,
        evaluation_io_s=0.3,
        query_latencies_ms=(2.0, 4.0),
        peak_gpu_gb=1.5,
        peak_ram_gb=2.5,
        final_map_mb=3.5,
        source_labels={"frontend_s": "Detection Process", "backend_s": "Local Mapping"},
    )


def test_runtime_breakdown_uses_explicit_units_and_rejects_gt_sources() -> None:
    runtime = _runtime()
    payload = runtime.to_json()

    assert payload["units"]["frontend_s"] == "seconds"
    assert payload["units"]["query_p95_ms"] == "milliseconds"
    assert payload["metrics"]["total_s_per_frame"] == pytest.approx(0.3)
    assert payload["metrics"]["processed_hz"] == pytest.approx(10.0 / 3.0)

    with pytest.raises(ValueError, match="ground truth"):
        RuntimeBreakdown(frame_count=1, source_labels={"frontend_s": "GT runtime"})


def test_method_modes_are_disclosed_and_oracle_is_not_ranked() -> None:
    with pytest.raises(ValueError, match="composed"):
        BaselineMetadata("PANOPTIC_SHARED", "Panoptic Mapping", "composed", "abc", "artifact-local")
    with pytest.raises(ValueError, match="offline"):
        BaselineMetadata("OPEN3DIS", "Open3DIS", "offline", "abc", "artifact-local")
    with pytest.raises(ValueError, match="oracle"):
        BaselineMetadata("KHRONOS_ORACLE", "Khronos", "oracle", "abc", "persistent-within-run")

    oracle = BaselineMetadata(
        "KHRONOS_ORACLE",
        "Khronos (GT semantics oracle)",
        "oracle",
        "abc",
        "persistent-within-run",
    )
    frozen = BaselineMetadata(
        "OVIMAP_FROZEN",
        "OVI-MAP (frozen)",
        "frozen",
        "abc",
        "artifact-local",
    )
    assert oracle.eligible_for_ranking is False
    assert frozen.eligible_for_ranking is True


def test_conceptgraphs_adapter_uses_runtime_labels_and_declares_local_ids() -> None:
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
                "clip_ft": np.array([0.1, 0.2], dtype=np.float32),
                "class_name": ["chair", "chair", "table"],
                "conf": [0.9, 0.8, 1.0],
                "image_idx": [0, 10, 20],
                "ground_truth_label": "table",
            }
        ]
    }
    artifact = adapt_conceptgraphs(
        payload,
        scene_id="room0",
        timestamp=20.0,
        upstream_commit="93277a0",
        runtime=_runtime(),
    )

    entity = artifact.snapshot.entities[0]
    assert entity.entity_id == "conceptgraphs:000000"
    assert entity.semantic_label == "chair"
    assert entity.metadata["semantic_label_source"] == "method_output.class_name"
    assert artifact.metadata.entity_id_stability == "artifact-local"


def test_conceptgraphs_adapter_accepts_runtime_clip_classification() -> None:
    payload = {
        "objects": [
            {
                "pcd_np": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                "clip_ft": np.array([0.1, 0.2], dtype=np.float32),
                "class_name": ["chair"],
                "conf": [0.9],
                "image_idx": [0],
            }
        ]
    }
    artifact = adapt_conceptgraphs(
        payload,
        scene_id="room0",
        timestamp=0.0,
        upstream_commit="93277a0",
        runtime=_runtime(),
        semantic_labels=("table",),
        semantic_scores=(0.75,),
        semantic_label_source="method_output.clip_ft ViT-H-14 zero-shot",
    )

    entity = artifact.snapshot.entities[0]
    assert entity.semantic_label == "table"
    assert entity.semantic_score == pytest.approx(0.75)
    assert entity.metadata["semantic_label_source"] == "method_output.clip_ft ViT-H-14 zero-shot"
    assert artifact.metadata.semantic_label_source == "method_output.clip_ft ViT-H-14 zero-shot"


def test_dualmap_adapter_uses_uid_and_runtime_class_map_not_gt() -> None:
    obj = SimpleNamespace(
        uid="stable-uuid",
        pcd=SimpleNamespace(points=np.array([[1.0, 0.0, 1.0]], dtype=np.float64)),
        clip_ft=np.array([0.3, 0.4], dtype=np.float32),
        class_id=7,
        observed_num=4,
        ground_truth_label="table",
    )
    artifact = adapt_dualmap(
        [obj],
        class_id_names={7: "chair"},
        background_xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scene_id="room0",
        timestamp=20.0,
        upstream_commit="157235e",
        runtime=_runtime(),
    )

    entity = artifact.snapshot.entities[0]
    assert entity.entity_id == "stable-uuid"
    assert entity.semantic_label == "chair"
    assert entity.metadata["semantic_label_source"] == "method_output.class_id"
    assert artifact.metadata.entity_id_stability == "persistent-within-run"
    np.testing.assert_allclose(artifact.snapshot.background_xyz, [[0.0, 0.0, 0.0]])


def test_ovimap_adapter_uses_runtime_embedding_and_color_instance_id() -> None:
    artifact = adapt_ovimap(
        {
            7: {
                "feat": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                "vis_area": np.array([1.0, 3.0], dtype=np.float32),
                "frame_id": [0, 10],
                "color": np.array([10, 20, 30], dtype=np.uint8),
                "ground_truth_label": "table",
            }
        },
        points_by_color={(10, 20, 30): np.array([[0.0, 0.0, 1.0]], dtype=np.float32)},
        scene_id="room0",
        timestamp=10.0,
        upstream_commit="58a804e",
        runtime=_runtime(),
    )

    entity = artifact.snapshot.entities[0]
    assert entity.entity_id == "ovimap:7"
    assert entity.semantic_label is None
    assert entity.semantic_embedding == pytest.approx([0.25, 0.75])
    assert entity.metadata["semantic_label_source"] == "method_output.feat"
    assert artifact.metadata.entity_id_stability == "persistent-within-run"


def test_frozen_update_audit_rejects_updates_after_intervention() -> None:
    audit = FrozenUpdateAudit()
    audit.record_update(1.0)
    audit.freeze(2.0)
    audit.record_query(3.0)

    with pytest.raises(RuntimeError, match="frozen"):
        audit.record_update(2.0)
    assert audit.to_json()["updates_after_freeze"] == 0


def test_identity_switch_fixture_reports_changed_prediction_id() -> None:
    perfect = evaluate_identity_assignments([{"gt-1": "pred-a"}, {"gt-1": "pred-a"}])
    switched = evaluate_identity_assignments([{"gt-1": "pred-a"}, {"gt-1": "pred-b"}])

    assert perfect == {"id_switches": 0, "tracked_gt_entities": 1, "assignment_count": 2}
    assert switched["id_switches"] == 1


def test_artifact_writer_emits_snapshot_runtime_and_metadata_json(tmp_path) -> None:
    artifact = adapt_conceptgraphs(
        {
            "objects": [
                {
                    "pcd_np": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    "clip_ft": np.array([1.0], dtype=np.float32),
                    "class_name": ["chair"],
                    "conf": [0.75],
                    "image_idx": [0],
                }
            ]
        },
        scene_id="room0",
        timestamp=0.0,
        upstream_commit="93277a0",
        runtime=_runtime(),
    )
    paths = write_baseline_artifact(artifact, tmp_path / "fixture")

    assert set(paths) == {"snapshot", "entities", "runtime", "metadata"}
    runtime = json.loads(paths["runtime"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert runtime["units"]["peak_gpu_gb"] == "GB"
    assert metadata["semantic_label_source"] == "method_output.class_name"
    assert metadata["snapshot_scope"] == "current"
    assert BaselineArtifact.from_json_files(paths["snapshot"], paths["entities"], paths["runtime"], paths["metadata"])
