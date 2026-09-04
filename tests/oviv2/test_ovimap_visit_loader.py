from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.oviv2.ovimap_visit_loader import load_ovimap_visit

_SOURCE_MANIFEST_SHA256 = "a" * 64
_OVIMAP_COMMIT = "58a804e2d7c82ba05a489eb071aba3367301fed8"


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_mesh(path: Path) -> None:
    path.write_text(
        """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 1 10 20 30
0.01 0 1 10 20 30
1 0 1 40 50 60
2 0 1 0 0 0
""",
        encoding="ascii",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    artifacts = {
        "instance_mesh": tmp_path / "instance_mesh_3.ply",
        "semantic_features": tmp_path / "features.pkl",
        "instance_color_log": tmp_path / "mapping.log",
    }
    _write_mesh(artifacts["instance_mesh"])
    with artifacts["semantic_features"].open("wb") as handle:
        pickle.dump(
            {
                7: {
                    "feat": np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
                    "vis_area": [1.0, 3.0],
                    "frame_id": [0, 2],
                },
                8: {
                    "feat": np.asarray([[0.0, 1.0]], dtype=np.float32),
                    "frame_id": [1],
                },
                99: {"feat": np.asarray([[1.0, 1.0]], dtype=np.float32)},
            },
            handle,
        )
    artifacts["instance_color_log"].write_text(
        "Instance: 7 Color: (10,20,30)\nInstance: 8 Color: (40,50,60)\n",
        encoding="utf-8",
    )

    copied = tmp_path / "materialized"
    copied.mkdir()
    materialized = copied / "export_manifest.json"
    materialized_payload = {
        "schema_version": 1,
        "status": "MATERIALIZED_INPUT_PASS",
        "dataset": "TESSE-CD",
        "scene": "apartment",
        "visit_id": "t1",
        "frame_count": 3,
        "source_frame_interval": [20, 22],
        "source_input_sha256": "b" * 64,
        "source_bindings": {},
        "outputs": {},
    }
    _write_json(materialized, materialized_payload)

    preflight = {
        "status": "PASS",
        "scene": "apartment",
        "ovimap_commit": _OVIMAP_COMMIT,
        "frame_ids": [0, 1, 2],
        "sources": {},
        "frames": [],
    }
    native = tmp_path / "native_mapping_manifest.json"
    _write_json(
        native,
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "MAPPING_PASS",
            "scene": "apartment",
            "frame_ids": [0, 1, 2],
            "preflight": preflight,
            "commands": {},
            "artifacts": {name: _record(path) for name, path in artifacts.items()},
            "postflight": preflight,
        },
    )
    return native, materialized, artifacts


def test_loads_hash_bound_native_visit_with_ovi_geometry_and_semantics(
    tmp_path: Path,
) -> None:
    native, materialized, _artifacts = _fixture(tmp_path)
    seen: list[tuple[str, int]] = []

    def labeler(evidence):
        seen.extend((item.entity_id, item.observation_count) for item in evidence)
        return {"ovimap:7": ("chair", 0.75)}

    visit = load_ovimap_visit(
        native_manifest=native,
        materialized_manifest=materialized,
        visit_id=1,
        scene="apartment",
        coordinate_frame_id="tesse_cd_world",
        source_manifest_sha256=_SOURCE_MANIFEST_SHA256,
        observed_frame_start=20,
        observed_frame_end=22,
        semantic_labeler=labeler,
    )

    assert visit.visit_id == 1
    assert visit.map_voxel_size_m == 0.01
    assert visit.observed_frame_start == 20
    assert visit.observed_frame_end == 22
    assert seen == [("ovimap:7", 2), ("ovimap:8", 1)]
    assert [entity.entity_id for entity in visit.snapshot.entities] == [
        "ovimap:7",
        "ovimap:8",
    ]
    first, second = visit.snapshot.entities
    assert first.semantic_label == "chair"
    assert first.semantic_score == pytest.approx(0.75)
    assert first.first_seen == 20.0
    assert first.last_seen == 22.0
    assert first.metadata["geometry_authority"] == "ovi_t1"
    assert first.metadata["source_instance_id"] == 7
    assert len(first.points_xyz) == 2
    assert second.semantic_label is None
    assert len(second.points_xyz) == 1
    assert visit.snapshot.background_xyz is not None
    assert visit.snapshot.background_xyz.tolist() == [[2.0, 0.0, 1.0]]


def test_rejects_native_artifact_changed_after_manifest_binding(tmp_path: Path) -> None:
    native, materialized, artifacts = _fixture(tmp_path)
    artifacts["semantic_features"].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact binding mismatch"):
        load_ovimap_visit(
            native_manifest=native,
            materialized_manifest=materialized,
            visit_id=1,
            scene="apartment",
            coordinate_frame_id="tesse_cd_world",
            source_manifest_sha256=_SOURCE_MANIFEST_SHA256,
            observed_frame_start=20,
            observed_frame_end=22,
        )


def test_rejects_visit_or_native_identity_mismatch(tmp_path: Path) -> None:
    native, materialized, _artifacts = _fixture(tmp_path)
    payload = json.loads(native.read_text(encoding="utf-8"))
    payload["postflight"]["ovimap_commit"] = "c" * 40
    _write_json(native, payload)

    with pytest.raises(ValueError, match="preflight and postflight"):
        load_ovimap_visit(
            native_manifest=native,
            materialized_manifest=materialized,
            visit_id=1,
            scene="apartment",
            coordinate_frame_id="tesse_cd_world",
            source_manifest_sha256=_SOURCE_MANIFEST_SHA256,
            observed_frame_start=20,
            observed_frame_end=22,
        )


def test_rejects_semantic_labels_for_unknown_or_ineligible_entities(
    tmp_path: Path,
) -> None:
    native, materialized, _artifacts = _fixture(tmp_path)

    def labeler(_evidence):
        return {"ovimap:999": ("chair", 0.9)}

    with pytest.raises(ValueError, match="unknown OVI entity"):
        load_ovimap_visit(
            native_manifest=native,
            materialized_manifest=materialized,
            visit_id=1,
            scene="apartment",
            coordinate_frame_id="tesse_cd_world",
            source_manifest_sha256=_SOURCE_MANIFEST_SHA256,
            observed_frame_start=20,
            observed_frame_end=22,
            semantic_labeler=labeler,
        )
