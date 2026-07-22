from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scripts.evaluation import export_tesse_temporal_artifact as exporter_module
from scripts.evaluation.export_tesse_temporal_artifact import (
    export_temporal_artifact,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SCHEDULE = (
    ROOT / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
)
COMMON_V2_MANIFEST = ROOT / "configs/evaluation/manifests/tesse_cd_common_v2.json"
OFFICIAL_SCHEDULE_SHA256 = (
    "fb97bacee377f9fd67ee9dae8064dc6f33ac32d129ee633629fec4164d5003e0"
)


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _entity(
    entity_id: str,
    label: str,
    x: float,
    *,
    entity_type: str = "object",
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray([[x, 0.0, 1.0]], dtype=np.float32),
        semantic_embedding=None,
        semantic_label=label,
        semantic_score=1.0,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=x,
        metadata={"entity_type": entity_type},
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_fixture(
    root: Path,
    *,
    panoptic: bool = False,
) -> tuple[Path, dict[str, Any]]:
    root.mkdir()
    scene = "apartment"
    checkpoints = [(0, 100), (2, 300), (4, 500)]
    schedule_path = root / "schedule.json"
    _write_json(
        schedule_path,
        {
            "schema_version": 2,
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "dataset": "TESSE-CD",
            "method_predictions_used": False,
            "parameters": {"frame_indexing": "zero_based"},
            "scenes": {
                scene: {
                    "frame_count": 5,
                    "entries": [
                        {"frame_index": frame, "timestamp_ns": timestamp}
                        for frame, timestamp in checkpoints
                    ],
                }
            },
        },
    )

    trajectory_path = root / "trajectories.jsonl"
    if panoptic:
        trajectory_rows = [
            {
                "frame_index": frame,
                "source_timestamp_ns": timestamp,
                "entities": [
                    {
                        "native_submap_id": 41,
                        "centroid_xyz": [float(frame), 0.0, 1.0],
                    }
                ],
            }
            for frame, timestamp in enumerate((100, 200, 300, 400, 500))
        ]
    else:
        trajectory_rows = [
            {
                "frame_index": frame,
                "timestamp_ns": timestamp,
                "entity_id": entity_id,
                "centroid_xyz": [float(frame), 0.0, 1.0],
                "observation_count": frame + 1,
            }
            for frame, timestamp, entity_id in (
                (0, 100, "entity-a"),
                (0, 100, "entity-b"),
                (1, 200, "entity-b"),
                (2, 300, "entity-b"),
                (4, 500, "entity-a"),
            )
        ]
    trajectory_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in trajectory_rows
        ),
        encoding="utf-8",
    )

    checkpoint_records = []
    artifact_entities = {
        0: (
            [_entity("panoptic:41", "chair", 0.0)]
            if panoptic
            else [_entity("entity-a", "chair", 0.0), _entity("entity-b", "table", 0.0)]
        ),
        2: (
            [_entity("panoptic:41", "chair", 2.0)]
            if panoptic
            else [_entity("entity-b", "table", 2.0)]
        ),
        4: (
            [_entity("panoptic:41", "chair", 4.0)]
            if panoptic
            else [_entity("entity-a", "chair", 4.0)]
        ),
    }
    checkpoint_status_records = []
    for frame, timestamp in checkpoints:
        checkpoint_root = root / "checkpoints" / str(timestamp)
        checkpoint_root.mkdir(parents=True)
        status_path = checkpoint_root / "checkpoint_status.json"
        status_payload = (
            {
                "schema_version": 1,
                "frame_index": frame,
                "source_timestamp_ns": timestamp,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "panmap_file": f"frame_{frame:08d}.panmap",
            }
            if panoptic
            else {
                "schema_version": 1,
                "status": "PASS",
                "checkpoint_frame": frame,
                "timestamp_ns": timestamp,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
            }
        )
        _write_json(status_path, status_payload)
        checkpoint_status_records.append(_record(status_path))

        snapshot = MapSnapshot(
            method="Panoptic Mapping + shared masks" if panoptic else "DualMap",
            scene_id=scene,
            timestamp=float(timestamp),
            entities=artifact_entities[frame],
            background_xyz=None,
            scope="current",
        )
        artifact_paths = write_map_snapshot(snapshot, checkpoint_root / "artifact")
        checkpoint_records.append(
            {
                "frame_index": frame,
                "timestamp_ns": timestamp,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "checkpoint_status": _record(status_path),
                "snapshot": _record(artifact_paths["snapshot"]),
                "entities": _record(artifact_paths["entities"]),
            }
        )

    capture_status_path = root / "run_status.json"
    _write_json(
        capture_status_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "scene": scene,
            "mode": "causal_checkpoints",
            "scheduled_frame_indices": [frame for frame, _ in checkpoints],
            "captured_frame_indices": [frame for frame, _ in checkpoints],
            "schedule": _record(schedule_path),
            "trajectories": _record(trajectory_path),
            "checkpoint_statuses": checkpoint_status_records,
        },
    )
    index_payload = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "mode": "causal_checkpoint_exports",
        "method": "PANOPTIC_SHARED" if panoptic else "DUALMAP",
        "scene": scene,
        "schedule": _record(schedule_path),
        "capture_status": _record(capture_status_path),
        "trajectories": _record(trajectory_path),
        "checkpoints": checkpoint_records,
    }
    index_path = root / "source_index.json"
    _write_json(index_path, index_payload)
    return index_path, index_payload


def _rewrite_index(index_path: Path, payload: dict[str, Any]) -> None:
    _write_json(index_path, payload)


def _artifact_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _rewrite_trajectories(
    index_path: Path,
    payload: dict[str, Any],
    content: str,
) -> None:
    trajectory_path = Path(payload["trajectories"]["path"])
    trajectory_path.write_text(content, encoding="utf-8")
    payload["trajectories"] = _record(trajectory_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["trajectories"] = payload["trajectories"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)


def _rewrite_checkpoint_entities(
    index_path: Path,
    payload: dict[str, Any],
    *,
    checkpoint: int,
    content: str,
) -> None:
    entities_path = Path(payload["checkpoints"][checkpoint]["entities"]["path"])
    entities_path.write_text(content, encoding="utf-8")
    payload["checkpoints"][checkpoint]["entities"] = _record(entities_path)
    _rewrite_index(index_path, payload)


def _rewrite_schedule(
    index_path: Path,
    payload: dict[str, Any],
    schedule: dict[str, Any],
) -> None:
    schedule_path = Path(payload["schedule"]["path"])
    _write_json(schedule_path, schedule)
    payload["schedule"] = _record(schedule_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schedule"] = payload["schedule"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)


def test_cli_help_runs_from_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/export_tesse_temporal_artifact.py"),
            "--help",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--source-index" in completed.stdout


def test_exports_presence_intervals_and_byte_identical_repeat(tmp_path: Path) -> None:
    index_path, source_index = _build_fixture(tmp_path / "source")

    first = export_temporal_artifact(index_path, tmp_path / "first")
    second = export_temporal_artifact(index_path, tmp_path / "second")

    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["mode"] == "causal_checkpoints"
    assert manifest["scene"] == "apartment"
    assert manifest["sources"]["trajectories"] == manifest["trajectories"]
    assert [item["frame_index"] for item in manifest["checkpoints"]] == [0, 2, 4]
    assert [item["timestamp_ns"] for item in manifest["checkpoints"]] == [100, 300, 500]
    assert [
        item["consumed_through_frame_exclusive"]
        for item in manifest["checkpoints"]
    ] == [1, 3, 5]
    lifecycles = {item["entity_id"]: item for item in manifest["entity_lifecycles"]}
    assert lifecycles["entity-a"]["presence_intervals"] == [
        {
            "first_frame_index": 0,
            "first_timestamp_ns": 100,
            "last_frame_index": 0,
            "last_timestamp_ns": 100,
        },
        {
            "first_frame_index": 4,
            "first_timestamp_ns": 500,
            "last_frame_index": 4,
            "last_timestamp_ns": 500,
        },
    ]
    assert lifecycles["entity-b"]["presence_intervals"] == [
        {
            "first_frame_index": 0,
            "first_timestamp_ns": 100,
            "last_frame_index": 2,
            "last_timestamp_ns": 300,
        }
    ]
    for checkpoint in manifest["checkpoints"]:
        for key in ("snapshot", "entities"):
            path = first.parent / checkpoint[key]["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint[key]["sha256"]
            assert path.stat().st_size == checkpoint[key]["byte_count"]
    first_files = {
        path.relative_to(first.parent): path.read_bytes()
        for path in first.parent.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.parent): path.read_bytes()
        for path in second.parent.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files

    assert first.read_bytes().endswith(b"\n")
    assert not first.read_bytes().endswith(b"\n\n")
    assert b"\n " not in first.read_bytes()
    exported_schedule = first.parent / manifest["sources"]["schedule"]["path"]
    source_schedule = Path(source_index["schedule"]["path"])
    assert exported_schedule.read_bytes() == source_schedule.read_bytes()
    for source in (
        manifest["sources"]["source_index"],
        manifest["sources"]["schedule"],
        manifest["sources"]["capture_status"],
        manifest["sources"]["trajectories"],
        *manifest["sources"]["checkpoint_statuses"],
    ):
        assert not Path(source["path"]).is_absolute()
        copied = first.parent / source["path"]
        assert copied.is_file()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == source["sha256"]
        assert copied.stat().st_size == source["byte_count"]


def test_artifact_is_byte_identical_across_distinct_source_roots(
    tmp_path: Path,
) -> None:
    first_index, _ = _build_fixture(tmp_path / "source-a")
    second_index, second_payload = _build_fixture(tmp_path / "source-b")
    for checkpoint_index, checkpoint in enumerate(second_payload["checkpoints"]):
        entities_path = Path(checkpoint["entities"]["path"])
        records = [
            json.loads(line)
            for line in entities_path.read_text(encoding="utf-8").splitlines()
        ]
        differently_formatted = "".join(
            json.dumps(dict(reversed(tuple(record.items())))) + "\n"
            for record in records
        )
        _rewrite_checkpoint_entities(
            second_index,
            second_payload,
            checkpoint=checkpoint_index,
            content=differently_formatted,
        )

    first_manifest = export_temporal_artifact(first_index, tmp_path / "output-a")
    second_manifest = export_temporal_artifact(second_index, tmp_path / "output-b")

    first_files = _artifact_files(first_manifest.parent)
    second_files = _artifact_files(second_manifest.parent)
    assert first_files == second_files
    forbidden = (
        str(first_index.parent.resolve()).encode(),
        str(second_index.parent.resolve()).encode(),
    )
    assert all(
        source_root not in content
        for content in first_files.values()
        for source_root in forbidden
    )
    for relative_path, content in first_files.items():
        if relative_path.suffix == ".jsonl":
            for line in content.splitlines():
                payload = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant: {value}")
                    ),
                )
                assert line == json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode(), relative_path
        if relative_path.suffix != ".json" or relative_path == Path(
            "sidecars/schedule.json"
        ):
            continue
        payload = json.loads(
            content,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
        expected = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        assert content == expected, relative_path


def test_official_schedule_preserves_checked_bytes_and_common_v2_hash(
    tmp_path: Path,
) -> None:
    from scripts.evaluation.evaluate_tesse_cd_common_v2 import _load_schedule

    common_v2 = json.loads(COMMON_V2_MANIFEST.read_text(encoding="utf-8"))
    assert common_v2["schedule"]["sha256"] == OFFICIAL_SCHEDULE_SHA256
    assert hashlib.sha256(OFFICIAL_SCHEDULE.read_bytes()).hexdigest() == (
        OFFICIAL_SCHEDULE_SHA256
    )
    source = exporter_module._direct_source(OFFICIAL_SCHEDULE, label="schedule")
    payload = exporter_module._read_json(source, label="schedule")
    output = tmp_path / "schedule.json"

    exporter_module._write_schedule_sidecar(
        source,
        payload,
        selected_scene="apartment",
        output=output,
    )

    assert output.read_bytes() == OFFICIAL_SCHEDULE.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == OFFICIAL_SCHEDULE_SHA256
    assert exporter_module._output_record(output, output=tmp_path)["sha256"] == (
        common_v2["schedule"]["sha256"]
    )
    events, common = _load_schedule(output, scene="apartment")
    assert events
    assert common


def test_rejects_unknown_trajectory_path_field(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    rows = trajectory_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["cache_path"] = "/tmp/not-part-of-the-contract"
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_trajectories(index_path, payload, "\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="trajectory .* fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_nonfinite_trajectory_json(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    rows = trajectory_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["centroid_xyz"][0] = float("nan")
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_trajectories(index_path, payload, "\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="non-finite JSON"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_unknown_entity_path_field(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][0]["entities"]["path"])
    rows = entities_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["cache_path"] = "/tmp/not-part-of-the-contract"
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_checkpoint_entities(
        index_path,
        payload,
        checkpoint=0,
        content="\n".join(rows) + "\n",
    )

    with pytest.raises(ValueError, match="entity .* fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_nonfinite_entity_json(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][0]["entities"]["path"])
    rows = entities_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["semantic_score"] = float("nan")
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_checkpoint_entities(
        index_path,
        payload,
        checkpoint=0,
        content="\n".join(rows) + "\n",
    )

    with pytest.raises(ValueError, match="non-finite JSON"):
        export_temporal_artifact(index_path, tmp_path / "output")


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_copied_json_sidecar(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schema_version"] = nonfinite
    capture_path.write_text(
        json.dumps(capture, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="non-finite JSON"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_unknown_source_index_path_field(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["unreviewed_cache_path"] = "/tmp/not-part-of-the-contract"
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="source index fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_unregistered_index_method(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["method"] = "OVIOVO"
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="unsupported.*method"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_index_method_that_does_not_match_snapshot(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["method"] = "OVIV2"
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="method.*match"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_wrong_schedule_schema_version(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    schedule_path = Path(payload["schedule"]["path"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["schema_version"] = 3
    _rewrite_schedule(index_path, payload, schedule)

    with pytest.raises(ValueError, match="schedule identity"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_wrong_capture_schema_version(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schema_version"] = 2
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="capture status"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_duplicate_capture_json_keys(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture.pop("schema_version")
    capture_path.write_text(
        '{"schema_version":1,"schema_version":1,'
        + json.dumps(capture, sort_keys=True)[1:],
        encoding="utf-8",
    )
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        export_temporal_artifact(index_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_index", -1, "non-negative"),
        ("timestamp_ns", 0, "positive"),
        ("centroid_xyz", [True, 0.0, 1.0], "numeric"),
        ("centroid_xyz", ["0.0", 0.0, 1.0], "numeric"),
        ("centroid_xyz", [10**400, 0.0, 1.0], "numeric"),
    ],
)
def test_rejects_invalid_trajectory_scalar_types_and_domains(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    row = json.loads(trajectory_path.read_text(encoding="utf-8").splitlines()[0])
    row[field] = value
    _rewrite_trajectories(index_path, payload, json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=message):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_invalid_trajectory_entity_after_last_checkpoint(
    tmp_path: Path,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    content = trajectory_path.read_text(encoding="utf-8")
    trailing = {
        "frame_index": 6,
        "timestamp_ns": 600,
        "entity_id": "entity-a",
        "centroid_xyz": [False, 0.0, 1.0],
        "observation_count": 1,
    }
    _rewrite_trajectories(
        index_path,
        payload,
        content + json.dumps(trailing) + "\n",
    )

    with pytest.raises(ValueError, match="numeric"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_accepts_panoptic_checkpoint_and_trajectory_fixture(tmp_path: Path) -> None:
    index_path, _ = _build_fixture(tmp_path / "panoptic", panoptic=True)

    manifest_path = export_temporal_artifact(index_path, tmp_path / "output")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["method"] == "PANOPTIC_SHARED"
    assert payload["entity_lifecycles"][0]["entity_id"] == "panoptic:41"
    rows = [
        json.loads(line)
        for line in (manifest_path.parent / payload["trajectories"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["entity_id"] == "panoptic:41"
    assert rows[-1]["timestamp_ns"] == 500


def test_rejects_missing_schedule_checkpoint_without_final_fallback(
    tmp_path: Path,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["checkpoints"].pop(1)
    _rewrite_index(index_path, payload)
    final = index_path.parent / "final"
    final.mkdir()
    (final / "snapshot.npz").write_bytes(b"must-not-be-read")

    with pytest.raises(ValueError, match="exactly cover schedule"):
        export_temporal_artifact(index_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("consumed_through_frame", 1),
        ("consumed_through_frame_exclusive", 2),
    ],
)
def test_rejects_checkpoint_that_does_not_end_exactly_at_t(
    tmp_path: Path, field: str, value: int
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["checkpoints"][1][field] = value
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="freeze boundary"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_checkpoint_without_exclusive_boundary(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["checkpoints"][1].pop("consumed_through_frame_exclusive")
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="freeze boundary"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_nonincreasing_checkpoint_timestamps(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    schedule_path = Path(payload["schedule"]["path"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["scenes"]["apartment"]["entries"][1]["timestamp_ns"] = 100
    _write_json(schedule_path, schedule)
    payload["schedule"] = _record(schedule_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schedule"] = payload["schedule"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="strictly increasing"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_checkpoint_outside_declared_frame_count(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    schedule_path = Path(payload["schedule"]["path"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["scenes"]["apartment"]["frame_count"] = 4
    _write_json(schedule_path, schedule)
    payload["schedule"] = _record(schedule_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schedule"] = payload["schedule"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="outside declared frame count"):
        export_temporal_artifact(index_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("semantic_label", "sofa", "semantic conflict"),
        ("entity_type", "region", "type conflict"),
    ],
)
def test_rejects_stable_id_semantic_or_type_conflict(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][2]["entities"]["path"])
    record = json.loads(entities_path.read_text(encoding="utf-8"))
    if field == "semantic_label":
        record["semantic_label"] = value
    else:
        record["metadata"]["entity_type"] = value
    entities_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    payload["checkpoints"][2]["entities"] = _record(entities_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match=message):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_duplicate_id_within_checkpoint(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][0]["entities"]["path"])
    records = [
        json.loads(line)
        for line in entities_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["entity_id"] = records[0]["entity_id"]
    entities_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    payload["checkpoints"][0]["entities"] = _record(entities_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="unique|duplicate"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_future_trajectory_sample(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    rows = trajectory_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[3])
    assert row["frame_index"] == 2
    row["timestamp_ns"] = 301
    rows[3] = json.dumps(row, sort_keys=True)
    trajectory_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload["trajectories"] = _record(trajectory_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["trajectories"] = payload["trajectories"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="later than checkpoint"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_exact_source_hash_mismatch(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    snapshot_path = Path(payload["checkpoints"][1]["snapshot"]["path"])
    changed = bytearray(snapshot_path.read_bytes())
    changed[-1] ^= 1
    snapshot_path.write_bytes(changed)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_copy_failure_cleans_staging_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path, _ = _build_fixture(tmp_path / "source")
    output = tmp_path / "output"
    original_copyfile = exporter_module.shutil.copyfile
    copy_count = 0

    def fail_second_copy(source: object, target: object) -> object:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("injected copy failure")
        return original_copyfile(source, target)

    monkeypatch.setattr(exporter_module.shutil, "copyfile", fail_second_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        export_temporal_artifact(index_path, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))

    monkeypatch.setattr(exporter_module.shutil, "copyfile", original_copyfile)
    manifest = export_temporal_artifact(index_path, output)
    assert manifest.is_file()


def test_temporal_output_is_atomic_no_clobber(tmp_path: Path) -> None:
    index_path, _ = _build_fixture(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker"
    marker.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        export_temporal_artifact(index_path, output)

    assert marker.read_bytes() == b"sentinel"
    assert set(output.iterdir()) == {marker}


def test_temporal_publish_failure_after_reservation_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path, _ = _build_fixture(tmp_path / "source")
    output = tmp_path / "output"

    def fail_rename(source: object, target: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "rename", fail_rename)
    with pytest.raises(RuntimeError, match="publication-uncertain"):
        export_temporal_artifact(index_path, output)

    assert output.is_dir()
