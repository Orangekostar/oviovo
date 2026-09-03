from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.crove_dyn_provenance import (
    classify_official_dyn_mass,
    parse_official_dynamic_rows,
    publish_crove_dyn_provenance,
)
from src.evaluation.exporters.oviovo import write_map_snapshot


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path if root is None else path.relative_to(root)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    composition_root = tmp_path / "composition"
    snapshot = MapSnapshot(
        method="OVIV2",
        scene_id="apartment",
        timestamp=100.0,
        entities=[
            EntityPrediction(
                entity_id="ovimap:1",
                points_xyz=np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
                semantic_embedding=np.asarray((1.0, 0.0), dtype=np.float32),
                semantic_label="Chair",
                semantic_score=1.0,
                lifecycle_state="active",
                first_seen=0.0,
                last_seen=10.0,
                metadata={
                    "authority": "crove_temporal",
                    "anchor_entity_id": "ovimap:1",
                    "temporal_entity_id": 7,
                    "overlay_state": "moved",
                    "geometry_epoch": 2,
                    "readout_valid": True,
                    "geometry_gate_accepted": False,
                },
            )
        ],
        background_xyz=None,
        scope="current",
    )
    written = write_map_snapshot(snapshot, composition_root / "checkpoint/current")
    trajectories = composition_root / "trajectories.jsonl"
    trajectories.write_text(
        json.dumps(
            {
                "frame_index": 10,
                "timestamp_ns": 100,
                "entity_id": 7,
                "centroid_xyz": [1.0, 0.0, 0.0],
                "dynamic_state": "dynamic",
                "motion_confidence": 0.8,
                "geometry_epoch": 2,
                "readout_valid": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    composition = _json(
        composition_root / "run_manifest.json",
        {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_composition_v1",
            "status": "PASS",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "identity_bindings": [
                {"anchor_entity_id": "ovimap:1", "temporal_entity_id": 7}
            ],
            "inputs": {"source_trajectories": _record(trajectories)},
            "checkpoints": [
                {
                    "frame_index": 10,
                    "timestamp_ns": 100,
                    "snapshot": _record(written["snapshot"], root=composition_root),
                    "entities": _record(written["entities"], root=composition_root),
                    "diagnostics": {
                        "moved_anchor_ids": ["ovimap:1"],
                        "new_temporal_ids": [],
                        "removed_anchor_ids": [],
                        "unchanged_anchor_ids": [],
                        "occluded_anchor_ids": [],
                    },
                }
            ],
        },
    )
    bridge = _json(
        tmp_path / "bridge/bridge_manifest.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "method": "OVIV2",
            "mode": "temporal_checkpoints",
            "scene_id": "apartment",
            "query_timestamps_ns": [100],
            "symbol_assignments": [
                {
                    "entity_id": "ovimap:1",
                    "node_index": 0,
                    "node_symbol": "O0",
                    "source_entity_id": "anchor:ovimap:1",
                }
            ],
            "checkpoints": [
                {
                    "frame_index": 10,
                    "timestamp_ns": 100,
                    "objects": [
                        {
                            "entity_id": "ovimap:1",
                            "node_index": 0,
                            "node_symbol": "O0",
                            "dynamic_state_at_query": "dynamic",
                        }
                    ],
                }
            ],
        },
    )
    dynamic = tmp_path / "dynamic_objects.csv"
    dynamic.write_text(
        "Name,Query,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
        "0,100,1,2,3\n",
        encoding="utf-8",
    )
    objects = tmp_path / "objects.csv"
    objects.write_text(
        "MapName,QueryTime,IsGT,ID,Label,CentroidX,CentroidY,CentroidZ,"
        "BBPosX,BBPosY,BBPosZ,BBDimX,BBDimY,BBDimZ,Present,HasAppeared,"
        "HasDisappeared\n0,100,0,O(0),5,1,0,0,1,0,0,1,1,1,1,0,0\n"
        "0,100,1,O(9),5,1,0,0,1,0,0,1,1,1,1,0,0\n"
        "0,100,1,O(10),5,1,0,0,1,0,0,1,1,1,1,0,0\n",
        encoding="utf-8",
    )
    associations = tmp_path / "associations.csv"
    associations.write_text(
        "MapName,QueryTime,FromGT,FromID,ToID\n"
        "0,100,0,O(0),O(9)\n"
        "0,100,0,O(0),O(10)\n"
        "0,100,1,O(68),Failed\n",
        encoding="utf-8",
    )
    return {
        "composition": composition,
        "bridge": bridge,
        "dynamic": dynamic,
        "objects": objects,
        "associations": associations,
    }


def test_parse_official_dynamic_rows_is_unique_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "dynamic.csv"
    path.write_text(
        "Name,Query,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
        "0,100,1,2,3\n0,100,1,2,3\n",
        encoding="utf-8",
    )

    assert parse_official_dynamic_rows(path)[0].total_mass == 6
    path.write_text(path.read_text(encoding="utf-8") + "0,100,2,2,3\n")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        parse_official_dynamic_rows(path)


def test_parse_official_dynamic_rows_uses_numeric_identity_order(tmp_path: Path) -> None:
    path = tmp_path / "dynamic.csv"
    path.write_text(
        "Name,Query,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
        "0,100,1,0,0\n"
        "2,100,2,0,0\n"
        "10,100,3,0,0\n",
        encoding="utf-8",
    )

    assert [row.map_name for row in parse_official_dynamic_rows(path)] == [0, 2, 10]


def test_dyn_mass_status_exact_ambiguous_and_unavailable() -> None:
    assert classify_official_dyn_mass(0, candidate_count=0, exact_event_count=0) == (
        "exact"
    )
    assert classify_official_dyn_mass(6, candidate_count=2, exact_event_count=None) == (
        "ambiguous"
    )
    assert classify_official_dyn_mass(6, candidate_count=0, exact_event_count=None) == (
        "unavailable"
    )
    assert classify_official_dyn_mass(6, candidate_count=2, exact_event_count=6) == (
        "exact"
    )
    with pytest.raises(ValueError, match="event mass"):
        classify_official_dyn_mass(6, candidate_count=2, exact_event_count=5)


def test_zero_official_dyn_mass_has_one_exact_fraction(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["dynamic"].write_text(
        "Name,Query,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
        "0,100,0,0,0\n",
        encoding="utf-8",
    )

    output = publish_crove_dyn_provenance(
        composition_manifest=paths["composition"],
        bridge_manifest=paths["bridge"],
        official_dynamic_csv=paths["dynamic"],
        visualization_objects_csv=paths["objects"],
        visualization_associations_csv=paths["associations"],
        output_root=tmp_path / "output",
    )

    assert json.loads(output.read_text(encoding="utf-8"))["coverage"] == {
        "ambiguous_fraction": 0.0,
        "ambiguous_mass": 0,
        "exact_fraction": 1.0,
        "exact_mass": 0,
        "official_mass": 0,
        "unavailable_fraction": 0.0,
        "unavailable_mass": 0,
    }


def test_dyn_provenance_cli_exposes_all_required_inputs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/build_crove_dyn_provenance.py",
            "--help",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    for option in (
        "--composition-manifest",
        "--bridge-manifest",
        "--official-dynamic-csv",
        "--visualization-objects-csv",
        "--visualization-associations-csv",
        "--output-root",
    ):
        assert option in result.stdout


def test_publish_dyn_provenance_binds_candidates_and_conserves_mass(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    output = publish_crove_dyn_provenance(
        composition_manifest=paths["composition"],
        bridge_manifest=paths["bridge"],
        official_dynamic_csv=paths["dynamic"],
        visualization_objects_csv=paths["objects"],
        visualization_associations_csv=paths["associations"],
        output_root=tmp_path / "output",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["coverage"] == {
        "ambiguous_fraction": 1.0,
        "ambiguous_mass": 6,
        "exact_fraction": 0.0,
        "exact_mass": 0,
        "official_mass": 6,
        "unavailable_fraction": 0.0,
        "unavailable_mass": 0,
    }
    row = payload["events"][0]
    assert row["event_id"] == "dyn:0:100"
    assert row["checkpoint_frame"] == 10
    assert row["provenance_status"] == "ambiguous"
    assert row["provenance_reason"] == (
        "official_dynamic_metrics_expose_aggregate_trajectory_mass_only"
    )
    assert row["static_visualization_associations_used_for_dyn"] is False
    candidate = row["candidate_predictions"][0]
    assert candidate["prediction_entity_id"] == "ovimap:1"
    assert candidate["authority"] == "crove_temporal"
    assert candidate["anchor_id"] == "ovimap:1"
    assert candidate["temporal_entity_id"] == 7
    assert candidate["overlay_state"] == "moved"
    assert candidate["dynamic_state"] == "dynamic"
    assert candidate["motion_confidence"] == 0.8
    assert candidate["geometry_epoch"] == 2
    assert candidate["readout_valid"] is True
    assert candidate["geometry_accepted"] is False
    assert candidate["identity_qualified"] is True
    assert sum(
        payload["coverage"][f"{name}_mass"]
        for name in ("exact", "ambiguous", "unavailable")
    ) == payload["coverage"]["official_mass"]
    assert set(payload["sources"]) == {
        "composition_manifest",
        "official_bridge_manifest",
        "official_dynamic_objects",
        "official_visualization_associations",
        "official_visualization_objects",
        "source_trajectories",
    }


def test_publish_dyn_provenance_rejects_coordinated_bridge_identity_tamper(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    bridge = json.loads(paths["bridge"].read_text(encoding="utf-8"))
    bridge["checkpoints"][0]["objects"][0]["entity_id"] = "ovimap:9"
    bridge["symbol_assignments"][0]["entity_id"] = "ovimap:9"
    _json(paths["bridge"], bridge)

    with pytest.raises(ValueError, match="composition entity"):
        publish_crove_dyn_provenance(
            composition_manifest=paths["composition"],
            bridge_manifest=paths["bridge"],
            official_dynamic_csv=paths["dynamic"],
            visualization_objects_csv=paths["objects"],
            visualization_associations_csv=paths["associations"],
            output_root=tmp_path / "output",
        )


def test_publish_dyn_provenance_rejects_malformed_visualization_node(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths["associations"].write_text(
        paths["associations"].read_text(encoding="utf-8").replace("O(68)", "bad"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="association node identity"):
        publish_crove_dyn_provenance(
            composition_manifest=paths["composition"],
            bridge_manifest=paths["bridge"],
            official_dynamic_csv=paths["dynamic"],
            visualization_objects_csv=paths["objects"],
            visualization_associations_csv=paths["associations"],
            output_root=tmp_path / "output",
        )
