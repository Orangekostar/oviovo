from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.build_tesse_ovimap_static_anchor import (
    load_causal_prefix_contract,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_schedule(
    root: Path,
    *,
    scene: str = "apartment",
    interventions: tuple[int, ...] = (3, 7),
    event_ids: tuple[str, ...] | None = None,
) -> Path:
    identifiers = event_ids or tuple(
        f"{scene}_event_{index:02d}"
        for index in range(1, len(interventions) + 1)
    )
    return _write_json(
        root / "schedule.json",
        {
            "dataset": "TESSE-CD",
            "manifest_id": "fixture_schedule",
            "scenes": {
                scene: {
                    "events": [
                        {
                            "event_id": event_id,
                            "intervention_frame_index": frame_index,
                        }
                        for event_id, frame_index in zip(
                            identifiers, interventions, strict=True
                        )
                    ]
                }
            },
            "schema_version": 1,
        },
    )


def _write_rgbd(root: Path, *, scene: str = "apartment", frame_count: int) -> Path:
    rgbd_root = root / "rgbd_v1"
    scene_root = rgbd_root / scene
    results = scene_root / "results"
    results.mkdir(parents=True)
    for frame_index in range(frame_count):
        (results / f"frame{frame_index:06d}.jpg").write_bytes(
            f"rgb:{frame_index}".encode("ascii")
        )
        (results / f"depth{frame_index:06d}.png").write_bytes(
            f"depth:{frame_index}".encode("ascii")
        )
    _write_json(
        scene_root / "export_manifest.json",
        {
            "dataset": "TESSE-CD",
            "frame_count": frame_count,
            "scene": scene,
            "schema_version": 1,
        },
    )
    return rgbd_root


def test_prefix_stops_before_first_intervention(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, interventions=(3, 7))
    rgbd_root = _write_rgbd(tmp_path, frame_count=8)

    contract = load_causal_prefix_contract(
        scene="apartment",
        schedule_path=schedule,
        rgbd_root=rgbd_root,
        configured_cutoff=2,
    )

    assert contract.scene == "apartment"
    assert contract.first_intervention_frame == 3
    assert contract.frame_ids == (0, 1, 2)
    assert contract.maximum_source_frame == 2
    assert contract.schedule_sha256 == _sha256(schedule)
    assert contract.rgbd_export_manifest_sha256 == _sha256(
        rgbd_root / "apartment" / "export_manifest.json"
    )


def test_prefix_rejects_intervention_frame(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, interventions=(3,))
    rgbd_root = _write_rgbd(tmp_path, frame_count=4)

    with pytest.raises(ValueError, match="strictly before first intervention"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=3,
        )


def test_prefix_rejects_scene_not_declared_by_schedule(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, scene="office", interventions=(3,))
    rgbd_root = _write_rgbd(tmp_path, scene="apartment", frame_count=4)

    with pytest.raises(ValueError, match="scene is not declared"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=2,
        )


@pytest.mark.parametrize("missing_name", ("frame000001.jpg", "depth000001.png"))
def test_prefix_rejects_incomplete_rgbd_inventory(
    tmp_path: Path, missing_name: str
) -> None:
    schedule = _write_schedule(tmp_path, interventions=(3,))
    rgbd_root = _write_rgbd(tmp_path, frame_count=4)
    (rgbd_root / "apartment" / "results" / missing_name).unlink()

    with pytest.raises(ValueError, match="causal RGB-D frame is missing"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=2,
        )


def test_prefix_rejects_duplicate_event_ids(tmp_path: Path) -> None:
    schedule = _write_schedule(
        tmp_path,
        interventions=(3, 7),
        event_ids=("duplicate", "duplicate"),
    )
    rgbd_root = _write_rgbd(tmp_path, frame_count=8)

    with pytest.raises(ValueError, match="event IDs must be unique"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=rgbd_root,
            configured_cutoff=2,
        )
