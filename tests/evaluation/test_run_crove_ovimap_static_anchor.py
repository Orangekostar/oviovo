from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.export_tesse_temporal_artifact import export_temporal_artifact
from scripts.evaluation.run_crove_ovimap_static_anchor import (
    _official_visibility_schedule,
    compose_run,
)
from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.crove_anchor_counterfactual import CounterfactualVariant
from src.evaluation.exporters.oviovo import read_map_snapshot, write_map_snapshot

_TIMESTAMP_ORIGIN_NS = 4_000_000_000


def _timestamp_ns(frame: int) -> int:
    return _TIMESTAMP_ORIGIN_NS + frame * 1_000_000_000


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path if root is None else path.relative_to(root)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def test_official_visibility_schedule_accepts_only_identical_worktree_alias(
    tmp_path: Path,
) -> None:
    canonical = (
        Path(__file__).resolve().parents[2]
        / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
    )
    alias = tmp_path / "schedule.json"
    alias.write_bytes(canonical.read_bytes())

    assert _official_visibility_schedule(alias) == canonical.resolve()

    alias.write_bytes(alias.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="official schedule binding"):
        _official_visibility_schedule(alias)


def _prediction(
    entity_id: str,
    x: float,
    *,
    temporal_id: int | None = None,
    dense: bool = False,
    z: float = 0.0,
) -> EntityPrediction:
    metadata: dict[str, object] = {"authority": "ovimap_anchor"}
    if temporal_id is not None:
        metadata = {
            "temporal_entity_id": temporal_id,
            "semantic_id": 0,
            "geometry_epoch": int(x > 0.5),
            "readout_valid": True,
        }
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(
            tuple(
                (x + offset, 0.0, z)
                for offset in (
                    (-0.2, -0.1, 0.0, 0.1, 0.2)
                    if dense
                    else (-0.05, 0.05)
                )
            ),
            dtype=np.float32,
        ),
        semantic_embedding=np.asarray((1.0, 0.0), dtype=np.float32),
        semantic_label="Chair",
        semantic_score=1.0,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=3.0,
        metadata=metadata,
    )


def _write_anchor_package(
    tmp_path: Path,
    *,
    anchor_x: float = 0.0,
    anchor_z: float = 0.0,
    dense: bool = False,
    visibility_export: bool = False,
) -> Path:
    root = tmp_path / "anchor"
    config = tmp_path / "anchor_config.json"
    vocabulary = tmp_path / "vocabulary.json"
    schedule = tmp_path / "schedule.json"
    _write_json(
        config,
        {
            "minimum_spatial_iou": 0.01,
            "maximum_centroid_distance_m": 0.75,
            "minimum_semantic_cosine": 0.65,
            "moved_displacement_m": 0.2,
            "background_voxel_size_m": 0.05,
        },
    )
    _write_json(
        vocabulary,
        {"dataset": "TESSE-CD", "scene": "apartment", "classes": ["Chair"]},
    )
    _write_json(
        schedule,
        {
            "schema_version": 2,
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "dataset": "TESSE-CD",
            "method_predictions_used": False,
            "parameters": {"frame_indexing": "zero_based"},
            "scenes": {
                "apartment": {
                    "frame_count": 4,
                    "entries": [
                        {"frame_index": frame, "timestamp_ns": _timestamp_ns(frame)}
                        for frame in (2, 3)
                    ],
                }
            },
        },
    )
    sources = {
        "config": _record(config),
        "vocabulary": _record(vocabulary),
        "schedule": _record(schedule),
    }
    if visibility_export:
        export_manifest = tmp_path / "rgbd" / "export_manifest.json"
        _write_json(
            export_manifest,
            {
                "dataset": "TESSE-CD",
                "scene": "apartment",
                "combined_output_sha256": "c" * 64,
                "file_hash_count": 1,
            },
        )
        sources["rgbd_export_manifest"] = _record(export_manifest)
    written = write_map_snapshot(
        MapSnapshot(
            method="OVI-MAP causal static anchor",
            scene_id="apartment",
            timestamp=1.0,
            entities=[
                _prediction("ovimap:1", anchor_x, dense=dense, z=anchor_z)
            ],
            background_xyz=np.asarray(((4.0, 0.0, 0.0),), dtype=np.float32),
            scope="current",
        ),
        root,
    )
    manifest = root / "anchor_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_v1",
            "status": "PASS",
            "method": "OVI-MAP causal static anchor",
            "scene": "apartment",
            "anchor_id": "a" * 64,
            "causality": {
                "first_source_frame": 0,
                "last_source_frame": 1,
                "maximum_source_frame": 1,
                "strictly_pre_intervention": True,
            },
            "sources": sources,
            "outputs": {
                "snapshot": _record(written["snapshot"], root=root),
                "entities": _record(written["entities"], root=root),
            },
        },
    )
    return manifest


class _VisibilityDataset:
    def __len__(self) -> int:
        return 4

    def timestamp_ns(self, index: int) -> int:
        return _timestamp_ns(index)

    def __getitem__(self, index: int) -> Frame:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 0.3 * max(0, index - 2)
        return Frame(
            frame_id=index,
            source_frame_id=index,
            rgb=np.zeros((5, 5, 3), dtype=np.uint8),
            depth=np.full((5, 5), 2.0, dtype=np.float32),
            pose=pose,
            intrinsics=CameraIntrinsics(3.0, 3.0, 2.0, 2.0, 5, 5),
            timestamp=_timestamp_ns(index) / 1_000_000_000,
        )


def _write_visibility_policy(tmp_path: Path) -> Path:
    path = tmp_path / "visibility_policy.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "policy_id": "crove_ovimap_unbound_visibility_v1",
            "voxel_sampling": "sorted_even_spacing",
            "voxel_size_m": 1.0,
            "depth_tolerance_m": 0.1,
            "depth_max_m": 10.0,
            "maximum_voxels_per_anchor": 1000,
            "minimum_tested_voxels": 1,
            "minimum_absent_fraction": 0.8,
            "minimum_present_fraction": 0.8,
            "minimum_absent_observations": 2,
            "minimum_distinct_viewpoints": 2,
            "minimum_viewpoint_baseline_m": 0.25,
            "minimum_present_streak": 2,
        },
    )
    return path


def _sample(frame: int, x: float) -> dict[str, object]:
    return {
        "frame_index": frame,
        "timestamp_ns": _timestamp_ns(frame),
        "entity_id": 7,
        "centroid_xyz": [x, 0.0, 0.0],
        "observation_count": frame + 1,
        "dynamic_state": "dynamic" if x > 0.5 else "static",
        "motion_confidence": 0.9 if x > 0.5 else 0.0,
        "geometry_epoch": int(x > 0.5),
        "readout_valid": True,
    }


def _write_source_run(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    trajectories = root / "trajectories.jsonl"
    lifecycle = root / "lifecycle_transitions.jsonl"
    coverage = root / "temporal_frame_coverage.jsonl"
    samples = [_sample(frame, 0.0 if frame < 3 else 1.0) for frame in range(4)]
    _write_jsonl(trajectories, samples)
    _write_jsonl(lifecycle, [])
    _write_jsonl(
        coverage,
        [
            {
                "frame_index": frame,
                "timestamp_ns": _timestamp_ns(frame),
                "record_count": 1,
                "event_count": 0,
            }
            for frame in range(4)
        ],
    )
    source_index = root / "source_index.json"
    _write_json(
        source_index,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": "apartment",
            "trajectories": _record(trajectories, root=root),
            "lifecycle_transitions": _record(lifecycle, root=root),
            "frame_coverage": _record(coverage, root=root),
        },
    )
    checkpoints = []
    for frame, x in ((2, 0.0), (3, 1.0)):
        written = write_map_snapshot(
            MapSnapshot(
                method="OVIV2-temporal",
                scene_id="apartment",
                timestamp=float(_timestamp_ns(frame)),
                entities=[_prediction("temporal:7", x, temporal_id=7)],
                background_xyz=None,
                scope="current",
            ),
            root / "checkpoints" / f"{frame:08d}-{_timestamp_ns(frame)}",
        )
        checkpoints.append(
            {
                "scene": "apartment",
                "frame_index": frame,
                "timestamp_ns": _timestamp_ns(frame),
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "neutral_snapshot": _record(written["snapshot"], root=root),
                "neutral_entities": _record(written["entities"], root=root),
            }
        )
    manifest = root / "run_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 2,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": "apartment",
            "mode": "dual_readout_causal_checkpoints",
            "processed_frame_count": 4,
            "covered_frame_count": 4,
            "first_frame_index": 0,
            "last_frame_index": 3,
            "temporal_export_schema_version": 1,
            "captured_frame_indices": [2, 3],
            "source_index": _record(source_index, root=root),
            "checkpoints": checkpoints,
        },
    )
    return manifest


def test_composed_run_publishes_hash_bound_checkpoints(tmp_path: Path) -> None:
    result = compose_run(
        source_run_manifest=_write_source_run(tmp_path),
        anchor_manifest=_write_anchor_package(tmp_path),
        output_root=tmp_path / "composed",
    )

    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["method"] == "CROVE + OVI-MAP static anchor (composed)"
    assert manifest["integration"] == "composed"
    assert manifest["execution_mode"] == "online_after_causal_initialization"
    assert manifest["readout_contract"] == {
        "moved_geometry_mode": "temporal_compact",
        "readout_role": "formal_baseline",
    }
    assert manifest["processed_frame_count"] == 4
    assert manifest["official_state_count"] == 2
    assert [item["frame_index"] for item in manifest["checkpoints"]] == [2, 3]
    assert manifest["checkpoints"][0]["diagnostics"]["unchanged_anchor_ids"] == ["ovimap:1"]
    assert manifest["checkpoints"][1]["diagnostics"]["moved_anchor_ids"] == ["ovimap:1"]
    for record in manifest["inputs"].values():
        assert set(record) == {"path", "sha256", "byte_count"}
    for checkpoint in manifest["checkpoints"]:
        for role in ("snapshot", "entities", "diagnostics_file"):
            path = result.parent / checkpoint[role]["path"]
            assert _record(path, root=result.parent) == checkpoint[role]

    source_index = result.parent / manifest["source_index"]["path"]
    exported = export_temporal_artifact(source_index, tmp_path / "exported")
    temporal = json.loads(exported.read_text(encoding="utf-8"))
    assert temporal["method"] == "OVIV2"
    assert [item["frame_index"] for item in temporal["checkpoints"]] == [2, 3]
    assert {item["entity_id"] for item in temporal["entity_lifecycles"]} == {
        "ovimap:1"
    }


def test_composed_run_publishes_dense_moved_visualization_shadow(
    tmp_path: Path,
) -> None:
    anchor_manifest = _write_anchor_package(tmp_path, dense=True)
    result = compose_run(
        source_run_manifest=_write_source_run(tmp_path),
        anchor_manifest=anchor_manifest,
        output_root=tmp_path / "dense-shadow",
        moved_geometry_mode="anchor_centroid_translation",
        readout_role="visualization_shadow",
    )

    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["readout_contract"] == {
        "moved_geometry_mode": "anchor_centroid_translation",
        "readout_role": "visualization_shadow",
    }
    checkpoint = manifest["checkpoints"][-1]
    dense = read_map_snapshot(
        result.parent / checkpoint["snapshot"]["path"],
        result.parent / checkpoint["entities"]["path"],
    )
    entity = dense.entities[0]
    np.testing.assert_allclose(
        entity.points_xyz[:, 0],
        (0.8, 0.9, 1.0, 1.1, 1.2),
        rtol=0.0,
        atol=1e-6,
    )
    assert entity.semantic_label == "Chair"
    assert entity.lifecycle_state == "active"
    assert entity.metadata["geometry_authority"] == "ovimap_anchor_template"
    assert entity.metadata["state_authority"] == "crove_temporal"


def test_composed_run_publishes_causal_unbound_visibility_candidate(
    tmp_path: Path,
) -> None:
    result = compose_run(
        source_run_manifest=_write_source_run(tmp_path),
        anchor_manifest=_write_anchor_package(
            tmp_path,
            anchor_z=1.5,
            visibility_export=True,
        ),
        output_root=tmp_path / "visibility-candidate",
        readout_role="causal_visibility_candidate",
        visibility_policy=_write_visibility_policy(tmp_path),
        dataset_factory=lambda *_: _VisibilityDataset(),
    )

    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["readout_contract"] == {
        "moved_geometry_mode": "temporal_compact",
        "readout_role": "causal_visibility_candidate",
        "unbound_anchor_mode": "causal_visibility",
    }
    source_index_path = result.parent / manifest["source_index"]["path"]
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    assert source_index["runtime_diagnostics"]["path"] == "runtime_diagnostics.json"
    diagnostics_path = result.parent / source_index["runtime_diagnostics"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["policy"]["sha256"] == manifest["inputs"][
        "visibility_policy"
    ]["sha256"]
    assert diagnostics["transition_count"] == 1
    assert diagnostics["transitions"] == [
        {
            "after": "dormant",
            "before": "active",
            "entity_id": "anchor:ovimap:1",
            "evidence": "visible_absent",
            "frame_index": 3,
            "readout_valid": False,
            "timestamp_ns": _timestamp_ns(3),
        }
    ]
    trajectories = [
        json.loads(line)
        for line in (result.parent / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    anchor_rows = [
        row for row in trajectories if row["entity_id"] == "anchor:ovimap:1"
    ]
    assert [row["readout_valid"] for row in anchor_rows] == [True, True, True, False]
    lifecycle = [
        json.loads(line)
        for line in (result.parent / "lifecycle_transitions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert lifecycle[-1] == {
        "after": "dormant",
        "before": "active",
        "entity_id": "anchor:ovimap:1",
        "evidence": "visible_absent",
        "frame_index": 3,
        "geometry_epoch": 0,
        "readout_valid": False,
        "timestamp_ns": _timestamp_ns(3),
    }
    final = manifest["checkpoints"][-1]
    snapshot = read_map_snapshot(
        result.parent / final["snapshot"]["path"],
        result.parent / final["entities"]["path"],
    )
    assert snapshot.entities == []
    assert final["diagnostics"]["visibility_suppressed_anchor_ids"] == [
        "ovimap:1"
    ]

    exported = export_temporal_artifact(source_index_path, tmp_path / "exported")
    temporal = json.loads(exported.read_text(encoding="utf-8"))
    assert temporal["sources"]["runtime_diagnostics"]["path"] == (
        "sidecars/runtime_diagnostics.json"
    )


def test_counterfactual_diagnostic_filters_the_same_causal_visibility_timeline(
    tmp_path: Path,
) -> None:
    source = _write_source_run(tmp_path)
    anchor = _write_anchor_package(
        tmp_path,
        anchor_z=1.5,
        visibility_export=True,
    )
    policy = _write_visibility_policy(tmp_path)
    variants = (
        CounterfactualVariant("CF0", "no_suppression", ()),
        CounterfactualVariant("CF3_00", "only_one", ("ovimap:1",), "ovimap:1"),
    )

    manifests = []
    for variant in variants:
        manifests.append(
            compose_run(
                source_run_manifest=source,
                anchor_manifest=anchor,
                output_root=tmp_path / variant.variant_id,
                readout_role="counterfactual_diagnostic",
                visibility_policy=policy,
                dataset_factory=lambda *_: _VisibilityDataset(),
                counterfactual_variant=variant,
            )
        )

    no_suppression = json.loads(manifests[0].read_text(encoding="utf-8"))
    only_anchor = json.loads(manifests[1].read_text(encoding="utf-8"))
    assert no_suppression["readout_contract"] == {
        "diagnostic_only": True,
        "moved_geometry_mode": "temporal_compact",
        "promotion_eligible": False,
        "readout_role": "counterfactual_diagnostic",
        "unbound_anchor_mode": "causal_visibility_filtered",
    }
    assert no_suppression["counterfactual_variant"] == variants[0].to_json_record()
    assert only_anchor["counterfactual_variant"] == variants[1].to_json_record()

    def final_entities(manifest: dict[str, object]) -> list[str]:
        checkpoint = manifest["checkpoints"][-1]
        snapshot = read_map_snapshot(
            manifests[0].parent.parent
            / manifest["counterfactual_variant"]["variant_id"]
            / checkpoint["snapshot"]["path"],
            manifests[0].parent.parent
            / manifest["counterfactual_variant"]["variant_id"]
            / checkpoint["entities"]["path"],
        )
        return [entity.entity_id for entity in snapshot.entities]

    assert final_entities(no_suppression) == ["ovimap:1"]
    assert final_entities(only_anchor) == []
    no_diagnostics = json.loads(
        (manifests[0].parent / "runtime_diagnostics.json").read_text(encoding="utf-8")
    )
    only_diagnostics = json.loads(
        (manifests[1].parent / "runtime_diagnostics.json").read_text(encoding="utf-8")
    )
    assert no_diagnostics["transition_count"] == 0
    assert only_diagnostics["transition_count"] == 1
    assert no_diagnostics["p5_reference_transition_count"] == 1
    assert only_diagnostics["p5_reference_transition_count"] == 1
    reference = only_diagnostics["p5_reference_transitions"][0]
    assert reference["absence_observation_count"] == 2
    assert reference["distinct_absence_viewpoint_count"] == 2
    entity = only_diagnostics["entities"][0]
    assert entity["first_absence_frame"] == 2
    assert entity["transition"] == {
        "absence_observation_count": 2,
        "distinct_absence_viewpoint_count": 2,
        "frame_index": 3,
    }


def test_counterfactual_variant_is_rejected_by_production_readout_role(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match="counterfactual"):
        compose_run(
            source_run_manifest=tmp_path / "not-read.json",
            anchor_manifest=tmp_path / "not-read-anchor.json",
            output_root=output,
            readout_role="causal_visibility_candidate",
            visibility_policy=tmp_path / "not-read-policy.json",
            counterfactual_variant=CounterfactualVariant(
                "CF0", "no_suppression", ()
            ),
        )

    assert not output.exists()


def test_counterfactual_variant_rejects_nontransitioned_anchor_without_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unknown-counterfactual"

    with pytest.raises(ValueError, match="transitioned unbound"):
        compose_run(
            source_run_manifest=_write_source_run(tmp_path),
            anchor_manifest=_write_anchor_package(
                tmp_path,
                anchor_z=1.5,
                visibility_export=True,
            ),
            output_root=output,
            readout_role="counterfactual_diagnostic",
            visibility_policy=_write_visibility_policy(tmp_path),
            dataset_factory=lambda *_: _VisibilityDataset(),
            counterfactual_variant=CounterfactualVariant(
                "CF3_00", "only_one", ("ovimap:missing",), "ovimap:missing"
            ),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("readout_role", "with_policy"),
    (
        ("causal_visibility_candidate", False),
        ("formal_baseline", True),
        ("evaluation_candidate", True),
    ),
)
def test_composed_run_rejects_visibility_role_policy_mismatch_before_output(
    tmp_path: Path,
    readout_role: str,
    with_policy: bool,
) -> None:
    output = tmp_path / f"rejected-visibility-{readout_role}"

    with pytest.raises(ValueError, match="visibility"):
        compose_run(
            source_run_manifest=tmp_path / "not-read.json",
            anchor_manifest=tmp_path / "not-read-anchor.json",
            output_root=output,
            readout_role=readout_role,
            visibility_policy=(
                tmp_path / "not-read-policy.json" if with_policy else None
            ),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("moved_geometry_mode", "readout_role"),
    (
        ("temporal_compact", "visualization_shadow"),
        ("temporal_compact", "evaluation_candidate"),
        ("anchor_centroid_translation", "formal_baseline"),
        ("unknown", "formal_baseline"),
        ("temporal_compact", "unknown"),
    ),
)
def test_composed_run_rejects_incompatible_readout_contract_before_output(
    tmp_path: Path,
    moved_geometry_mode: str,
    readout_role: str,
) -> None:
    output = tmp_path / f"rejected-{moved_geometry_mode}-{readout_role}"

    with pytest.raises(ValueError, match="incompatible"):
        compose_run(
            source_run_manifest=tmp_path / "not-read.json",
            anchor_manifest=tmp_path / "not-read-anchor.json",
            output_root=output,
            moved_geometry_mode=moved_geometry_mode,
            readout_role=readout_role,
        )

    assert not output.exists()


def test_composed_run_never_overwrites_existing_output(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    anchor = _write_anchor_package(tmp_path)
    output = tmp_path / "composed"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        compose_run(
            source_run_manifest=source,
            anchor_manifest=anchor,
            output_root=output,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_unbound_anchor_has_explicit_causal_presence_for_official_export(
    tmp_path: Path,
) -> None:
    result = compose_run(
        source_run_manifest=_write_source_run(tmp_path),
        anchor_manifest=_write_anchor_package(tmp_path, anchor_x=5.0),
        output_root=tmp_path / "composed",
    )
    manifest = json.loads(result.read_text(encoding="utf-8"))
    exported = export_temporal_artifact(
        result.parent / manifest["source_index"]["path"],
        tmp_path / "exported",
    )
    temporal = json.loads(exported.read_text(encoding="utf-8"))

    assert {item["entity_id"] for item in temporal["entity_lifecycles"]} == {
        "ovimap:1"
    }


def test_composed_run_rejects_checkpoint_ahead_of_temporal_export(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["timestamp_ns"] += 1
    _write_json(source, payload)

    with pytest.raises(ValueError, match="checkpoint.*temporal export"):
        compose_run(
            source_run_manifest=source,
            anchor_manifest=_write_anchor_package(tmp_path),
            output_root=tmp_path / "composed",
        )


def test_composed_run_rejects_tampered_temporal_sidecar(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    trajectories = source.parent / "trajectories.jsonl"
    trajectories.write_bytes(trajectories.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="trajectories binding mismatch"):
        compose_run(
            source_run_manifest=source,
            anchor_manifest=_write_anchor_package(tmp_path),
            output_root=tmp_path / "composed",
        )

    assert not (tmp_path / "composed").exists()
