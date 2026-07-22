from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.datasets.tesse_cd import TesseCdFrameRecord, TesseCdRgbdDataset


TIMESTAMPS_NS = (
    4_204_107_999,
    4_254_107_999,
    4_304_108_000,
    4_354_108_000,
    4_404_107_999,
)
CAMERA = {
    "cx": 360.0,
    "cy": 240.0,
    "fx": 415.69219381653056,
    "fy": 415.69219381653056,
    "h": 480,
    "scale": 1000.0,
    "w": 720,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _exported_content_digest(
    dataset_root: Path,
    scene_root: Path,
) -> tuple[str, int]:
    paths = [
        *scene_root.joinpath("results").glob("frame*.jpg"),
        *scene_root.joinpath("results").glob("depth*.png"),
        scene_root / "traj.txt",
        scene_root / "timestamps.csv",
        dataset_root / "cam_params.json",
    ]
    output_hashes = [
        (str(path.relative_to(dataset_root)), _sha256(path)) for path in paths
    ]
    digest = hashlib.sha256()
    for relative, file_hash in sorted(output_hashes):
        digest.update(
            relative.encode("utf-8")
            + b"\0"
            + file_hash.encode("ascii")
            + b"\n"
        )
    return digest.hexdigest(), len(output_hashes)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "rgbd_v1"
    scene_root = dataset_root / "apartment"
    results = scene_root / "results"
    results.mkdir(parents=True)

    _write_json(dataset_root / "cam_params.json", {"camera": CAMERA})
    for index in range(5):
        rgb = np.full((480, 720, 3), index + 20, dtype=np.uint8)
        depth_mm = np.full((480, 720), 1_000 + index, dtype=np.uint16)
        Image.fromarray(rgb, mode="RGB").save(
            results / f"frame{index:06d}.jpg",
            quality=95,
        )
        Image.fromarray(depth_mm).save(results / f"depth{index:06d}.png")

    timestamps_path = scene_root / "timestamps.csv"
    with timestamps_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("frame_index", "sensor_timestamp_ns", "relative_timestamp_ns")
        )
        first = TIMESTAMPS_NS[0]
        for index, timestamp_ns in enumerate(TIMESTAMPS_NS):
            writer.writerow((index, timestamp_ns, timestamp_ns - first))

    poses = []
    for index in range(5):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.1
        poses.append(" ".join(str(value) for value in pose.reshape(-1)))
    (scene_root / "traj.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")

    source_manifest = tmp_path / "tesse_cd.json"
    database_sha256 = "a" * 64
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "camera": {
                "width": 720,
                "height": 480,
                "fx": CAMERA["fx"],
                "fy": CAMERA["fy"],
                "cx": CAMERA["cx"],
                "cy": CAMERA["cy"],
            },
            "sequences": {
                "apartment": {
                    "bag": {"database": {"sha256": database_sha256}},
                    "timeline": {
                        "depth_frame_count": 5,
                        "first_depth_timestamp_ns": TIMESTAMPS_NS[0],
                        "last_depth_timestamp_ns": TIMESTAMPS_NS[-1],
                    },
                }
            },
        },
    )

    combined_output_sha256, file_hash_count = _exported_content_digest(
        dataset_root,
        scene_root,
    )
    export_manifest = scene_root / "export_manifest.json"
    _write_json(
        export_manifest,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "frame_count": 5,
            "source_manifest": str(source_manifest),
            "source_database_sha256": database_sha256,
            "combined_output_sha256": combined_output_sha256,
            "file_hash_count": file_hash_count,
        },
    )

    schedule_manifest = tmp_path / "schedule.json"
    _write_json(
        schedule_manifest,
        {
            "schema_version": 2,
            "dataset": "TESSE-CD",
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "source_manifest": {
                "path": str(source_manifest),
                "sha256": _sha256(source_manifest),
            },
            "scenes": {
                "apartment": {
                    "frame_count": 5,
                    "first_depth_timestamp_ns": TIMESTAMPS_NS[0],
                    "last_depth_timestamp_ns": TIMESTAMPS_NS[-1],
                    "sources": {"database": {"sha256": database_sha256}},
                }
            },
        },
    )
    return scene_root, export_manifest, schedule_manifest


@pytest.fixture(autouse=True)
def _five_frame_apartment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(TesseCdRgbdDataset.EXPECTED_FRAMES, "apartment", 5)


def _dataset(tmp_path: Path) -> TesseCdRgbdDataset:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    return TesseCdRgbdDataset(
        root,
        scene="apartment",
        export_manifest=export_manifest,
        schedule_manifest=schedule_manifest,
    )


def test_loads_five_frame_rgbd_sequence_with_true_nanosecond_timestamps(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)

    assert len(dataset) == 5
    assert dataset.timestamp_ns(2) == TIMESTAMPS_NS[2]
    assert isinstance(dataset.records[2], TesseCdFrameRecord)
    assert dataset.records[2].relative_timestamp_ns == 100_000_001

    frame = dataset[2]
    assert frame.frame_id == 2
    assert frame.source_frame_id == 2
    assert frame.timestamp == TIMESTAMPS_NS[2] / 1_000_000_000
    assert frame.rgb.shape == (480, 720, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (480, 720)
    assert frame.depth.dtype == np.float32
    np.testing.assert_allclose(frame.depth, 1.002)
    assert frame.intrinsics.width == 720
    assert frame.intrinsics.height == 480
    assert frame.intrinsics.fx == CAMERA["fx"]
    np.testing.assert_allclose(frame.pose[0, 3], 0.2)


def test_accepts_dataset_root_and_resolves_scene_subdirectory(tmp_path: Path) -> None:
    scene_root, export_manifest, schedule_manifest = _write_fixture(tmp_path)

    dataset = TesseCdRgbdDataset(
        scene_root.parent,
        scene="apartment",
        export_manifest=export_manifest,
        schedule_manifest=schedule_manifest,
    )

    assert len(dataset) == 5


def test_rejects_unknown_scene(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="scene"):
        TesseCdRgbdDataset(
            root,
            scene="warehouse",
            export_manifest=export_manifest,
            schedule_manifest=schedule_manifest,
        )


def test_rejects_non_increasing_timestamps(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    rows = list(csv.reader((root / "timestamps.csv").read_text().splitlines()))
    rows[3][1] = rows[2][1]
    with (root / "timestamps.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    with pytest.raises(ValueError, match="strictly increasing"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_rejects_wrong_rgb_shape(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    Image.fromarray(np.zeros((480, 719, 3), dtype=np.uint8)).save(
        root / "results" / "frame000003.jpg"
    )

    with pytest.raises(ValueError, match="RGB.*shape|RGB.*size"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_rejects_non_uint16_source_depth(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    Image.fromarray(np.zeros((480, 720), dtype=np.uint8)).save(
        root / "results" / "depth000003.png"
    )

    with pytest.raises(ValueError, match="depth.*uint16"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


@pytest.mark.parametrize(
    "bad_pose",
    [
        pytest.param("nan " + "0 " * 15, id="nonfinite"),
        pytest.param("0 " * 15, id="too-few-tokens"),
        pytest.param("0 " * 17, id="too-many-tokens"),
        pytest.param("not-a-number " + "0 " * 15, id="nonnumeric"),
        pytest.param(
            "2 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
            id="nonrigid",
        ),
        pytest.param(
            "1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1",
            id="nonhomogeneous",
        ),
    ],
)
def test_rejects_invalid_pose_with_source_frame_context(
    tmp_path: Path,
    bad_pose: str,
) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    lines = (root / "traj.txt").read_text(encoding="utf-8").splitlines()
    lines[3] = bad_pose.strip()
    (root / "traj.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"source frame 3 pose"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_record_pose_cannot_be_made_writeable(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError):
        dataset.records[0].camera_to_world.setflags(write=True)


def test_intrinsics_mutation_does_not_change_dataset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    exposed = dataset.intrinsics
    exposed.fx = 1.0

    assert dataset.intrinsics.fx == CAMERA["fx"]
    assert dataset[0].intrinsics.fx == CAMERA["fx"]


def test_rejects_camera_mismatch(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    camera_path = root.parent / "cam_params.json"
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    payload["camera"]["fx"] = 400.0
    _write_json(camera_path, payload)

    with pytest.raises(ValueError, match="camera|intrinsics"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_rejects_missing_frame_in_contiguous_sequence(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    (root / "results" / "frame000003.jpg").unlink()

    with pytest.raises(ValueError, match="frame|RGB"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_rejects_same_shape_uint16_depth_content_tamper(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    Image.fromarray(np.full((480, 720), 9_999, dtype=np.uint16)).save(
        root / "results" / "depth000003.png"
    )

    with pytest.raises(ValueError, match="combined output hash mismatch"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_rejects_export_file_hash_count_drift(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    payload = json.loads(export_manifest.read_text(encoding="utf-8"))
    payload["file_hash_count"] += 1
    _write_json(export_manifest, payload)

    with pytest.raises(ValueError, match="file hash count"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_rejects_export_manifest_hash_chain_drift(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    payload = json.loads(export_manifest.read_text(encoding="utf-8"))
    payload["source_database_sha256"] = "c" * 64
    _write_json(export_manifest, payload)

    with pytest.raises(ValueError, match="export.*hash|database.*hash"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)


def test_rejects_schedule_source_manifest_hash_drift(tmp_path: Path) -> None:
    root, export_manifest, schedule_manifest = _write_fixture(tmp_path)
    payload = json.loads(schedule_manifest.read_text(encoding="utf-8"))
    payload["source_manifest"]["sha256"] = "d" * 64
    _write_json(schedule_manifest, payload)

    with pytest.raises(ValueError, match="schedule.*hash|source manifest.*hash"):
        TesseCdRgbdDataset(root, "apartment", export_manifest, schedule_manifest)
