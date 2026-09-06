from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.evaluation.materialize_3rscan_ovi_visit import (
    MaterializationError,
    main,
    materialize_sequence_archive,
    parse_sequence_info,
)


SCAN_ID = "00000000-0000-0000-0000-000000000123"
PAIR_ID = "fixture-pair"
VISIT_INDEX = 1

COLOR_INTRINSIC = np.asarray(
    [
        [120.0, 0.0, 2.0, 0.0],
        [0.0, 110.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEPTH_INTRINSIC = np.asarray(
    [
        [200.0, 0.0, 1.0, 0.0],
        [0.0, 210.0, 0.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
IDENTITY = np.eye(4, dtype=np.float64)


def _matrix_text(matrix: np.ndarray) -> str:
    return "\n".join(
        " ".join(f"{float(value):.17g}" for value in row)
        for row in np.asarray(matrix, dtype=np.float64)
    ) + "\n"


def _info_text(
    *,
    color_extrinsic: np.ndarray = IDENTITY,
    depth_extrinsic: np.ndarray = IDENTITY,
) -> str:
    return (
        "m_versionNumber = 4\n"
        "m_sensorName = fixtureSensor\n"
        "m_colorWidth = 6\n"
        "m_colorHeight = 4\n"
        "m_depthWidth = 3\n"
        "m_depthHeight = 2\n"
        "m_depthShift = 1000\n"
        f"m_calibrationColorIntrinsic = {' '.join(f'{value:.17g}' for value in COLOR_INTRINSIC.reshape(-1))}\n"
        f"m_calibrationColorExtrinsic = {' '.join(f'{value:.17g}' for value in color_extrinsic.reshape(-1))}\n"
        f"m_calibrationDepthIntrinsic = {' '.join(f'{value:.17g}' for value in DEPTH_INTRINSIC.reshape(-1))}\n"
        f"m_calibrationDepthExtrinsic = {' '.join(f'{value:.17g}' for value in depth_extrinsic.reshape(-1))}\n"
        "m_frames.size = 3\n"
    )


def _encoded(extension: str, array: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(extension, array)
    assert success
    return encoded.tobytes()


def _sequence_zip(
    tmp_path: Path,
    *,
    missing_members: frozenset[str] = frozenset(),
    malformed_pose_frame: int | None = None,
    color_extrinsic: np.ndarray = IDENTITY,
    depth_extrinsic: np.ndarray = IDENTITY,
) -> tuple[dict[str, object], dict[int, np.ndarray]]:
    colors = {
        0: np.full((4, 6, 3), (12, 34, 56), dtype=np.uint8),
        1: np.full((4, 6, 3), (78, 90, 102), dtype=np.uint8),
        2: np.full((4, 6, 3), (144, 156, 168), dtype=np.uint8),
    }
    depths = {
        0: np.asarray([[0, 1, 255], [256, 1000, 65535]], dtype=np.uint16),
        1: np.asarray([[65535, 1000, 256], [255, 1, 0]], dtype=np.uint16),
        2: np.asarray([[1, 0, 1000], [65535, 256, 255]], dtype=np.uint16),
    }
    archive_path = tmp_path / "sequence.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "_info.txt",
            _info_text(
                color_extrinsic=color_extrinsic,
                depth_extrinsic=depth_extrinsic,
            ).encode("ascii"),
        )
        for frame_id in range(3):
            members = {
                f"frame-{frame_id:06d}.color.jpg": _encoded(".jpg", colors[frame_id]),
                f"frame-{frame_id:06d}.depth.pgm": _encoded(".pgm", depths[frame_id]),
            }
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = float(frame_id)
            pose_data = _matrix_text(pose).encode("ascii")
            if malformed_pose_frame == frame_id:
                pose_data = _matrix_text(np.eye(3, dtype=np.float64)).encode("ascii")
            members[f"frame-{frame_id:06d}.pose.txt"] = pose_data
            for name, data in members.items():
                if name not in missing_members:
                    archive.writestr(name, data)
    archive_bytes = archive_path.read_bytes()
    return (
        {
            "path": str(archive_path.resolve()),
            "sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "byte_count": len(archive_bytes),
        },
        depths,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_pgm(archive: zipfile.ZipFile, frame_id: int) -> np.ndarray:
    member = f"frame-{frame_id:06d}.depth.pgm"
    encoded = np.frombuffer(archive.read(member), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    return decoded


def test_parse_sequence_info_preserves_calibration_dimensions_and_frame_count() -> None:
    info = parse_sequence_info(_info_text())

    assert info.version_number == 4
    assert info.sensor_name == "fixtureSensor"
    assert info.color_width == 6
    assert info.color_height == 4
    assert info.depth_width == 3
    assert info.depth_height == 2
    assert info.depth_shift == 1000
    np.testing.assert_array_equal(info.color_intrinsic, COLOR_INTRINSIC)
    np.testing.assert_array_equal(info.depth_intrinsic, DEPTH_INTRINSIC)
    np.testing.assert_array_equal(info.color_extrinsic, IDENTITY)
    np.testing.assert_array_equal(info.depth_extrinsic, IDENTITY)
    assert info.frame_count == 3


def test_materialize_step_two_preserves_source_content_and_manifest_bindings(
    tmp_path: Path,
) -> None:
    binding, _depths = _sequence_zip(tmp_path)
    output_dir = tmp_path / "visit"

    result = materialize_sequence_archive(
        sequence_zip_binding=binding,
        scan_id=SCAN_ID,
        pair_id=PAIR_ID,
        visit_index=VISIT_INDEX,
        output_dir=output_dir,
        frame_step=2,
    )

    manifest_path = output_dir / "materialized_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result == manifest
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "MATERIALIZED_INPUT_PASS"
    assert manifest["dataset"] == "3RScan"
    assert manifest["dataset_adapter"] == "scannet_nyu"
    assert manifest["scan_id"] == SCAN_ID
    assert manifest["pair_id"] == PAIR_ID
    assert manifest["visit_index"] == VISIT_INDEX
    assert manifest["frame_step"] == 2
    assert manifest["frame_map"] == [
        {"source_frame_id": 0, "target_frame_id": 0},
        {"source_frame_id": 2, "target_frame_id": 1},
    ]

    color_zero = cv2.imread(str(output_dir / "color" / "0.jpg"), cv2.IMREAD_UNCHANGED)
    color_one = cv2.imread(str(output_dir / "color" / "1.jpg"), cv2.IMREAD_UNCHANGED)
    assert color_zero is not None
    assert color_one is not None
    assert color_zero.shape == (4, 6, 3)
    assert color_one.shape == (4, 6, 3)

    with zipfile.ZipFile(binding["path"]) as archive:
        source_depth_zero = _decode_pgm(archive, 0)
        source_depth_two = _decode_pgm(archive, 2)
    assert source_depth_zero.dtype == np.uint16
    assert source_depth_two.dtype == np.uint16
    assert set(int(value) for value in source_depth_zero.reshape(-1)) == {
        0,
        1,
        255,
        256,
        1000,
        65535,
    }
    depth_zero = cv2.imread(
        str(output_dir / "depth" / "0.png"), cv2.IMREAD_UNCHANGED
    )
    depth_one = cv2.imread(
        str(output_dir / "depth" / "1.png"), cv2.IMREAD_UNCHANGED
    )
    assert depth_zero is not None
    assert depth_one is not None
    assert depth_zero.dtype == np.uint16
    assert depth_one.dtype == np.uint16
    np.testing.assert_array_equal(depth_zero, source_depth_zero)
    np.testing.assert_array_equal(depth_one, source_depth_two)

    pose_zero = np.loadtxt(output_dir / "pose" / "0.txt")
    pose_one = np.loadtxt(output_dir / "pose" / "1.txt")
    assert pose_zero.shape == (4, 4)
    assert pose_one.shape == (4, 4)
    assert pose_one[0, 3] == pytest.approx(2.0)

    color_intrinsic = np.loadtxt(output_dir / "intrinsic" / "intrinsic_color.txt")
    depth_intrinsic = np.loadtxt(output_dir / "intrinsic" / "intrinsic_depth.txt")
    assert color_intrinsic.shape == (4, 4)
    assert depth_intrinsic.shape == (4, 4)
    np.testing.assert_array_equal(color_intrinsic, COLOR_INTRINSIC)
    np.testing.assert_array_equal(depth_intrinsic, DEPTH_INTRINSIC)
    assert not np.array_equal(color_intrinsic, depth_intrinsic)

    expected_files = {
        "color/0.jpg",
        "color/1.jpg",
        "depth/0.png",
        "depth/1.png",
        "pose/0.txt",
        "pose/1.txt",
        "intrinsic/intrinsic_color.txt",
        "intrinsic/intrinsic_depth.txt",
    }
    output_tree = manifest["output_tree"]
    assert output_tree["file_count"] == len(expected_files)
    records = output_tree["files"]
    assert {record["path"] for record in records} == expected_files
    assert output_tree["byte_count"] == sum(
        (output_dir / relative).stat().st_size for relative in expected_files
    )
    tree_digest = hashlib.sha256()
    for relative in sorted(expected_files):
        path = output_dir / relative
        record = next(item for item in records if item["path"] == relative)
        assert record["byte_count"] == path.stat().st_size
        assert record["sha256"] == _sha256(path)
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(record["sha256"].encode("ascii"))
        tree_digest.update(b"\n")
    assert output_tree["sha256"] == tree_digest.hexdigest()


def test_materialize_rejects_missing_pose_from_complete_contiguous_frame_set(
    tmp_path: Path,
) -> None:
    binding, _depths = _sequence_zip(
        tmp_path,
        missing_members=frozenset({"frame-000002.pose.txt"}),
    )

    with pytest.raises(MaterializationError, match="complete contiguous frame set"):
        materialize_sequence_archive(
            sequence_zip_binding=binding,
            scan_id=SCAN_ID,
            pair_id=PAIR_ID,
            visit_index=VISIT_INDEX,
            output_dir=tmp_path / "visit",
        )


def test_materialize_rejects_pose_that_is_not_exactly_4x4(tmp_path: Path) -> None:
    binding, _depths = _sequence_zip(tmp_path, malformed_pose_frame=1)

    with pytest.raises(MaterializationError, match="4x4"):
        materialize_sequence_archive(
            sequence_zip_binding=binding,
            scan_id=SCAN_ID,
            pair_id=PAIR_ID,
            visit_index=VISIT_INDEX,
            output_dir=tmp_path / "visit",
        )


@pytest.mark.parametrize("extrinsic", ("color", "depth"))
def test_materialize_rejects_non_identity_rgb_depth_extrinsics(
    tmp_path: Path,
    extrinsic: str,
) -> None:
    bad_extrinsic = IDENTITY.copy()
    bad_extrinsic[0, 3] = 0.1
    kwargs = {
        "color_extrinsic": bad_extrinsic if extrinsic == "color" else IDENTITY,
        "depth_extrinsic": bad_extrinsic if extrinsic == "depth" else IDENTITY,
    }
    binding, _depths = _sequence_zip(tmp_path, **kwargs)

    with pytest.raises(MaterializationError, match="identity RGB-depth extrinsics"):
        materialize_sequence_archive(
            sequence_zip_binding=binding,
            scan_id=SCAN_ID,
            pair_id=PAIR_ID,
            visit_index=VISIT_INDEX,
            output_dir=tmp_path / "visit",
        )


def test_materialize_rejects_existing_output_without_modifying_content(
    tmp_path: Path,
) -> None:
    binding, _depths = _sequence_zip(tmp_path)
    output_dir = tmp_path / "visit"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="ascii")
    before = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }

    with pytest.raises(MaterializationError, match="already exists"):
        materialize_sequence_archive(
            sequence_zip_binding=binding,
            scan_id=SCAN_ID,
            pair_id=PAIR_ID,
            visit_index=VISIT_INDEX,
            output_dir=output_dir,
        )

    after = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_cli_materializes_exact_bound_archive(tmp_path: Path, capsys) -> None:
    binding, _depths = _sequence_zip(tmp_path)
    output_dir = tmp_path / "cli-visit"

    exit_code = main(
        [
            "--sequence-zip",
            str(binding["path"]),
            "--sequence-sha256",
            str(binding["sha256"]),
            "--sequence-byte-count",
            str(binding["byte_count"]),
            "--scan-id",
            SCAN_ID,
            "--pair-id",
            PAIR_ID,
            "--visit-index",
            str(VISIT_INDEX),
            "--output-dir",
            str(output_dir),
            "--frame-step",
            "2",
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "MATERIALIZED_INPUT_PASS"
    assert printed["frame_count"] == 2
    assert json.loads(
        (output_dir / "materialized_manifest.json").read_text(encoding="utf-8")
    ) == printed
