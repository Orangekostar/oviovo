from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from scripts.evaluation.derive_tesse_cd_common_v2 import (
    active_dsg_records,
    build_contract_manifest,
    deterministic_npz_bytes,
    derive_target_arrays,
    dsg_interval_world_points,
    dilate_voxels,
    match_event_dsg_records,
    load_generation_contract,
    load_derived_rgbd_frames,
    main,
    occlusion_shadow_voxels,
    observed_background_voxels,
    ray_free_voxels,
    reject_prediction_input_paths,
    render_manifest,
    revealed_background_voxels,
    semantic_voxel_labels,
    validate_generation_inputs,
    visible_surface_voxels,
    voxel_keys,
    world_object_points,
    write_contract_manifest,
    write_target_package,
)


Voxel = tuple[int, int, int]
ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "configs/evaluation/manifests/tesse_cd.json"
SCHEDULE = ROOT / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
CONTRACT_MANIFEST = ROOT / "configs/evaluation/manifests/tesse_cd_common_v2.json"


def _dsg_record(
    symbol: str,
    *,
    first: int,
    last: int,
    semantic_label: int = 1,
    point: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict[str, object]:
    return {
        "id": int(symbol[2:-1]),
        "attributes": {
            "name": symbol,
            "semantic_label": semantic_label,
            "first_observed_ns": [first],
            "last_observed_ns": [last],
            "dynamic_object_points": [[list(point)]],
        },
    }


def _changes(count: int = 14) -> list[dict[str, str]]:
    return [
        {
            "ObjectSymbol": f"O({index})",
            "AppearedAt": str(index * 10),
            "DisappearedAt": str(index * 10 + 100),
        }
        for index in range(count)
    ]


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        serialized_path = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        serialized_path = str(resolved)
    return {
        "path": serialized_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _declared_file(path: Path) -> dict[str, object]:
    record = _file_record(path)
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "size_bytes": record["byte_count"],
    }


def _write_generation_fixture(root: Path) -> tuple[Path, Path]:
    sequences: dict[str, object] = {}
    schedule_scenes: dict[str, object] = {}
    object_index = 0
    for scene, object_count, event_times in (
        ("apartment", 6, (100, 200, 300, 400)),
        ("office", 8, (500, 600, 700, 800)),
    ):
        scene_root = root / scene
        scene_root.mkdir()
        changes = scene_root / "gt_changes.csv"
        rows = [
            {
                "ObjectSymbol": f"O({object_index + index})",
                "AppearedAt": "0",
                "DisappearedAt": str(event_times[index % 4]),
            }
            for index in range(object_count)
        ]
        object_index += object_count
        changes.write_text(
            "ObjectSymbol,AppearedAt,DisappearedAt\n"
            + "".join(
                f'{row["ObjectSymbol"]},{row["AppearedAt"]},{row["DisappearedAt"]}\n'
                for row in rows
            ),
            encoding="utf-8",
        )
        dsg = scene_root / "gt_dsg_consolidated.json"
        dsg.write_text(
            json.dumps(
                {
                    "nodes": [
                        _dsg_record(
                            row["ObjectSymbol"],
                            first=0,
                            last=int(row["DisappearedAt"]),
                        )
                        for row in rows
                    ]
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        dsg_with_mesh = scene_root / "gt_dsg_with_mesh_consolidated.json"
        mesh_nodes = []
        for row in rows:
            record = _dsg_record(
                row["ObjectSymbol"],
                first=0,
                last=int(row["DisappearedAt"]),
            )
            attributes = record["attributes"]
            attributes["dynamic_object_points"] = []
            attributes["position"] = [0.0, 0.0, 1.0]
            attributes["bounding_box"] = {
                "type": "AABB",
                "world_R_center": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            }
            attributes["mesh"] = {"points": [[0.0, 0.0, 0.0]]}
            mesh_nodes.append(record)
        dsg_with_mesh.write_text(
            json.dumps({"nodes": mesh_nodes}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        background = scene_root / "gt_background.ply"
        background.write_bytes(
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            b"element vertex 1\n"
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"end_header\n"
            + struct.pack("<fff", 0.0, 0.0, 1.06)
        )
        database = scene_root / "bag.db3"
        database.write_bytes(b"fixture bag")
        sequences[scene] = {
            "bag": {"database": _declared_file(database), "directory": str(scene_root)},
            "timeline": {
                "depth_frame_count": 20,
                "first_depth_timestamp_ns": 1_000,
                "last_depth_timestamp_ns": 2_000,
                "change_times_relative_ns": list(event_times),
            },
            "ground_truth": {
                "files": {
                    "background_mesh": _declared_file(background),
                    "changes": _declared_file(changes),
                    "dsg": _declared_file(dsg),
                    "dsg_with_mesh": _declared_file(dsg_with_mesh),
                }
            },
        }
        schedule_scenes[scene] = {
            "events": [
                {
                    "event_id": f"{scene}_event_{index:02d}",
                    "event_relative_timestamp_ns": timestamp,
                    "intervention_frame_index": index + 2,
                    "common_checkpoint_frame_indices": list(range(index + 2, index + 12)),
                }
                for index, timestamp in enumerate(event_times, start=1)
            ],
            "entries": [],
        }

    source_manifest = root / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "tesse_cd_dynamic_v1",
                "dataset": "TESSE-CD",
                "source": {"owner": "MIT-SPARK Khronos official release"},
                "camera": {
                    "width": 3,
                    "height": 3,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 1.0,
                    "cy": 1.0,
                },
                "topics": {"depth": "/depth", "pose": "/pose"},
                "sequences": sequences,
                "protocol": {
                    "future_frames_allowed": False,
                    "ground_truth_evaluator_only": True,
                    "runtime_ground_truth_access": False,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    schedule = root / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_id": "tesse_cd_causal_schedule_v2",
                "dataset": "TESSE-CD",
                "method_predictions_used": False,
                "source_manifest": _file_record(source_manifest),
                "scenes": schedule_scenes,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_manifest, schedule


def _write_derived_rgbd_fixture(
    root: Path,
    source_manifest: Path,
    *,
    frame_count: int = 20,
    source_role: str = "official_rgbd_export",
) -> tuple[Path, Path]:
    from PIL import Image

    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    derived = root / "derived_rgbd"
    camera_path = derived / "cam_params.json"
    camera_path.parent.mkdir(parents=True)
    camera_path.write_text(
        json.dumps({"camera": source["camera"]}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock_scenes: dict[str, object] = {}
    for scene in ("apartment", "office"):
        scene_root = derived / scene
        results = scene_root / "results"
        results.mkdir(parents=True)
        timestamps = []
        poses = []
        first = int(source["sequences"][scene]["timeline"]["first_depth_timestamp_ns"])
        last = int(source["sequences"][scene]["timeline"]["last_depth_timestamp_ns"])
        for index in range(frame_count):
            timestamp = first + round((last - first) * index / (frame_count - 1))
            timestamps.append((index, timestamp, timestamp - first))
            poses.append(" ".join(str(value) for value in np.eye(4).reshape(-1)))
            depth = np.full((3, 3), 1000 if index < 3 else 1049, dtype=np.uint16)
            Image.fromarray(depth).save(results / f"depth{index:06d}.png")
            rgb = np.full((3, 3, 3), index, dtype=np.uint8)
            Image.fromarray(rgb).save(results / f"frame{index:06d}.jpg", quality=95)
        (scene_root / "timestamps.csv").write_text(
            "frame_index,sensor_timestamp_ns,relative_timestamp_ns\n"
            + "".join(f"{a},{b},{c}\n" for a, b, c in timestamps),
            encoding="utf-8",
        )
        (scene_root / "traj.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")
        database = source["sequences"][scene]["bag"]["database"]
        output_files = [
            path
            for index in range(frame_count)
            for path in (
                results / f"frame{index:06d}.jpg",
                results / f"depth{index:06d}.png",
            )
        ] + [
            scene_root / "traj.txt",
            scene_root / "timestamps.csv",
            camera_path,
        ]
        output_hashes = [
            (
                path.relative_to(derived).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in output_files
        ]
        digest = hashlib.sha256()
        for relative, file_hash in sorted(output_hashes):
            digest.update(
                relative.encode("utf-8")
                + b"\0"
                + file_hash.encode("ascii")
                + b"\n"
            )
        export_manifest = scene_root / "export_manifest.json"
        export_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset": "TESSE-CD",
                    "scene": scene,
                    "frame_count": frame_count,
                    "source_database": database["path"],
                    "source_database_sha256": database["sha256"],
                    "source_manifest": str(source_manifest.resolve()),
                    "rgb_encoding": "JPEG quality 95 decoded from official rgb8",
                    "depth_encoding": "uint16 millimeters decoded from official 32FC1 meters",
                    "pose": "world_T_base_link_gt multiplied by bag tf_static base_link_gt_T_left_cam",
                    "combined_output_sha256": digest.hexdigest(),
                    "file_hash_count": len(output_hashes),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        lock_scenes[scene] = {
            "export_manifest": _file_record(export_manifest),
            "combined_output_sha256": digest.hexdigest(),
            "file_hash_count": len(output_hashes),
            "source_database_sha256": database["sha256"],
        }
    lock = root / "tesse_cd_rgbd_v1.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "tesse_cd_rgbd_v1",
                "dataset": "TESSE-CD",
                "source_role": source_role,
                "source_manifest": _file_record(source_manifest),
                "derived_root": str(derived.resolve()),
                "scenes": lock_scenes,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return derived, lock


def test_match_event_dsg_records_requires_exact_fourteen_known_objects() -> None:
    changes = _changes()
    records = [
        _dsg_record(
            row["ObjectSymbol"],
            first=int(row["AppearedAt"]),
            last=int(row["DisappearedAt"]),
        )
        for row in changes
    ]

    matched = match_event_dsg_records(changes, records)

    assert len(matched) == 14
    assert [item["symbol"] for item in matched] == [
        f"O({index})" for index in range(14)
    ]
    assert all(item["interval_index"] == 0 for item in matched)

    with pytest.raises(ValueError, match="exactly 14 event object records"):
        match_event_dsg_records(changes, records[:-1])

    records[-1]["attributes"]["semantic_label"] = 4_294_967_295
    with pytest.raises(ValueError, match="unknown semantic label"):
        match_event_dsg_records(changes, records)


def test_event_interval_matching_and_active_domain_are_half_open() -> None:
    record = _dsg_record("O(0)", first=10, last=20)
    changes = [
        {"ObjectSymbol": "O(0)", "AppearedAt": "10", "DisappearedAt": "20"}
    ]

    matched = match_event_dsg_records(changes, [record], expected_count=1)

    assert matched[0]["first_timestamp_ns"] == 10
    assert matched[0]["last_timestamp_ns"] == 20
    assert active_dsg_records([record], timestamp_ns=10) == (record,)
    assert active_dsg_records([record], timestamp_ns=19) == (record,)
    assert active_dsg_records([record], timestamp_ns=20) == ()


def test_world_object_points_applies_finite_homogeneous_transform() -> None:
    local = np.asarray([[0.0, 0.0, 0.0], [0.05, 0.10, 0.15]])
    world_from_object = np.asarray(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.array_equal(
        world_object_points(local, world_from_object),
        np.asarray([[1.0, 2.0, 3.0], [0.9, 2.05, 3.15]]),
    )

    invalid = world_from_object.copy()
    invalid[3, 3] = 2.0
    with pytest.raises(ValueError, match="homogeneous"):
        world_object_points(local, invalid)


def test_voxelization_is_floor_based_unique_and_deterministically_sorted() -> None:
    points = np.asarray(
        [
            [0.099, 0.0, 0.0],
            [-0.001, 0.0, 0.0],
            [0.050, 0.0, 0.0],
            [0.050, 0.0, 0.0],
        ]
    )

    assert voxel_keys(points) == ((-1, 0, 0), (1, 0, 0))


def test_one_voxel_dilation_uses_full_twenty_six_neighbor_connectivity() -> None:
    dilated = dilate_voxels(((0, 0, 0),))

    assert len(dilated) == 27
    assert (-1, -1, -1) in dilated
    assert (1, 1, 1) in dilated
    assert dilated == tuple(sorted(dilated))


def test_ray_free_voxels_stop_five_centimeters_before_endpoint() -> None:
    free = ray_free_voxels(
        np.asarray([0.025, 0.025, 0.025]),
        np.asarray([[0.225, 0.025, 0.025]]),
    )

    assert free == ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
    assert (4, 0, 0) not in free


def test_visible_surface_voxels_keep_only_z_buffer_front_surface() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ]
    )

    visible = visible_surface_voxels(
        points,
        camera_from_world=np.eye(4),
        intrinsics=(10.0, 10.0, 1.0, 1.0),
        image_size=(3, 3),
    )

    assert visible == ((0, 0, 20), (2, 0, 20))
    assert (0, 0, 40) not in visible


def test_occlusion_shadow_requires_background_behind_object_in_same_pixel() -> None:
    object_points = np.asarray([[0.0, 0.0, 1.0]])
    background_points = np.asarray([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]])

    shadow = occlusion_shadow_voxels(
        object_points,
        background_points,
        camera_from_world=np.eye(4),
        intrinsics=(10.0, 10.0, 1.0, 1.0),
        image_size=(3, 4),
    )

    assert shadow == ((0, 0, 40),)


def test_revealed_background_is_shadowed_pre_unobserved_post_observed() -> None:
    background: tuple[Voxel, ...] = ((0, 0, 0), (1, 0, 0), (2, 0, 0))
    shadow: tuple[Voxel, ...] = ((0, 0, 0), (1, 0, 0))
    pre_observed: tuple[Voxel, ...] = ((1, 0, 0),)
    post_observed: tuple[Voxel, ...] = ((0, 0, 0), (1, 0, 0), (2, 0, 0))

    assert revealed_background_voxels(
        background,
        shadow,
        pre_observed,
        post_observed,
    ) == ((0, 0, 0),)


def test_background_observation_requires_at_most_five_centimeter_residual() -> None:
    background = np.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]])
    observed = np.asarray([[0.049, 0.0, 1.0], [0.251, 0.0, 1.0]])

    assert observed_background_voxels(background, observed) == ((0, 0, 20),)


def test_semantic_voxel_collisions_use_point_majority_then_lowest_label() -> None:
    labeled_points = [
        (np.asarray([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0]]), 2),
        (np.asarray([[0.003, 0.0, 0.0]]), 3),
        (np.asarray([[0.051, 0.0, 0.0]]), 2),
        (np.asarray([[0.052, 0.0, 0.0]]), 1),
    ]

    assert semantic_voxel_labels(labeled_points) == (
        (0, 0, 0, 2),
        (1, 0, 0, 1),
    )


@pytest.mark.parametrize(
    "path",
    [
        "/runs/method_output/snapshot.npz",
        "/runs/predictions/frame_001.npz",
        "/results/baselines/dualmap/map.ply",
    ],
)
def test_method_output_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="prediction or method output path"):
        reject_prediction_input_paths([path])


def test_benign_source_under_prediction_named_worktree_is_not_rejected(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "worktrees" / "predictions-hygiene" / "official"
    source_root.mkdir(parents=True)
    source = source_root / "ground_truth.csv"
    source.write_text("official\n", encoding="utf-8")

    validate_generation_inputs(
        source_paths=[source],
        target_arrays={"event": np.ones((1, 3))},
    )


def test_explicit_prediction_role_is_rejected_without_suspicious_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.bin"
    prediction = tmp_path / "benign.bin"
    source.write_bytes(b"official")
    prediction.write_bytes(b"prediction")

    with pytest.raises(ValueError, match="prediction or method output path"):
        validate_generation_inputs(
            source_paths=[source],
            target_arrays={"event": np.ones((1, 3))},
            prediction_input_paths=[prediction],
        )


def test_contract_manifest_is_prediction_independent_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    schedule = tmp_path / "schedule.json"
    changes = tmp_path / "gt_changes.csv"
    source.write_bytes(b'{"dataset":"TESSE-CD"}\n')
    schedule.write_bytes(b'{"manifest_id":"schedule"}\n')
    changes.write_bytes(b"ObjectSymbol,AppearedAt,DisappearedAt\n")

    first = build_contract_manifest(
        source_manifest=source,
        schedule=schedule,
        ground_truth_sources={"changes": changes},
    )
    second = build_contract_manifest(
        source_manifest=source,
        schedule=schedule,
        ground_truth_sources={"changes": changes},
    )

    assert first == second
    assert first["status"] == "CONTRACT_ONLY"
    assert first["targets_generated"] is False
    assert first["fixture_tested"] is True
    assert first["expected_event_object_count"] == 14
    assert first["prediction_inputs_used"] is False
    assert first["parameters"] == {
        "voxel_size_m": 0.05,
        "dilation_voxels": 1,
        "ray_endpoint_margin_m": 0.05,
        "pre_event_frames": 450,
        "post_event_frames": 450,
        "z_buffer": "nearest_positive_depth_per_pixel",
        "active_interval": "first <= timestamp < last",
        "semantic_voxel_label_rule": "point_majority_then_lowest_label_id",
        "unobservable_background_rule": "exclude_from_background_f5_and_recovery",
    }
    assert first["schedule"]["sha256"] == hashlib.sha256(
        schedule.read_bytes()
    ).hexdigest()
    assert render_manifest(first) == render_manifest(second)

    output_a = tmp_path / "a.json"
    output_b = tmp_path / "b.json"
    write_contract_manifest(output_a, first)
    write_contract_manifest(output_b, second)
    assert output_a.read_bytes() == output_b.read_bytes()
    assert json.loads(output_a.read_text(encoding="utf-8")) == first


def test_contract_generation_hard_fails_on_missing_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    schedule = tmp_path / "schedule.json"
    source.write_text("{}\n", encoding="utf-8")
    schedule.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source is not a file"):
        build_contract_manifest(
            source_manifest=source,
            schedule=schedule,
            ground_truth_sources={"changes": tmp_path / "missing.csv"},
        )

def test_real_generation_input_gate_rejects_missing_sources_and_empty_targets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"official")

    with pytest.raises(ValueError, match="source is not a file"):
        validate_generation_inputs(
            source_paths=[tmp_path / "missing.bin"],
            target_arrays={"event": np.ones((1, 3))},
        )

    with pytest.raises(ValueError, match="target arrays must be non-empty"):
        validate_generation_inputs(source_paths=[source], target_arrays={})

    with pytest.raises(ValueError, match="target array is empty"):
        validate_generation_inputs(
            source_paths=[source], target_arrays={"event": np.empty((0, 3))}
        )

    with pytest.raises(ValueError, match="prediction or method output path"):
        validate_generation_inputs(
            source_paths=[source],
            target_arrays={"event": np.ones((1, 3))},
            prediction_input_paths=[tmp_path / "method_output" / "map.npz"],
        )


def test_checked_in_manifest_is_explicitly_contract_only_and_hash_bound() -> None:
    assert CONTRACT_MANIFEST.is_file()
    payload = json.loads(CONTRACT_MANIFEST.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    assert payload["status"] == "CONTRACT_ONLY"
    assert payload["targets_generated"] is False
    assert payload["fixture_tested"] is True
    assert payload["expected_event_object_count"] == 14
    assert payload["prediction_inputs_used"] is False
    assert "verified_event_object_count" not in payload
    assert "targets" not in payload
    assert payload["source_manifest"] == _file_record(SOURCE_MANIFEST)
    assert payload["schedule"] == _file_record(SCHEDULE)
    assert not Path(payload["source_manifest"]["path"]).is_absolute()
    assert not Path(payload["schedule"]["path"]).is_absolute()
    assert str(ROOT) not in CONTRACT_MANIFEST.read_text(encoding="utf-8")

    expected_ground_truth: dict[str, dict[str, object]] = {}
    for scene in ("apartment", "office"):
        files = source["sequences"][scene]["ground_truth"]["files"]
        for source_name, manifest_name in (
            ("background_mesh", "background_mesh"),
            ("changes", "changes"),
            ("dsg", "dsg"),
            ("dsg_with_mesh", "dsg_with_mesh"),
        ):
            declaration = files[manifest_name]
            expected_ground_truth[f"{scene}.{source_name}"] = {
                "path": str(Path(declaration["path"]).resolve()),
                "sha256": declaration["sha256"],
                "byte_count": declaration["size_bytes"],
            }
    assert payload["ground_truth_sources"] == expected_ground_truth

    binding = {
        "source_manifest": payload["source_manifest"],
        "schedule": payload["schedule"],
        "ground_truth": payload["ground_truth_sources"],
    }
    canonical = json.dumps(
        binding, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert payload["input_binding_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert CONTRACT_MANIFEST.read_bytes() == render_manifest(payload)
    rebuilt = build_contract_manifest(
        source_manifest=SOURCE_MANIFEST,
        schedule=SCHEDULE,
        ground_truth_sources={
            f"{scene}.{name}": Path(declaration["path"])
            for scene in ("apartment", "office")
            for name, declaration in source["sequences"][scene]["ground_truth"][
                "files"
            ].items()
        },
    )
    assert render_manifest(rebuilt) == CONTRACT_MANIFEST.read_bytes()


def test_generation_contract_parses_sources_schedule_and_exact_object_coverage(
    tmp_path: Path,
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)

    contract = load_generation_contract(source, schedule)

    assert contract["event_object_count"] == 14
    assert contract["scene_event_object_counts"] == {"apartment": 6, "office": 8}
    assert contract["scene_event_counts"] == {"apartment": 4, "office": 4}
    assert contract["prediction_inputs_used"] is False
    assert len(contract["source_records"]) == 10


def test_real_dsg_mesh_points_are_transformed_from_object_to_world() -> None:
    record = _dsg_record("O(0)", first=0, last=10)
    attributes = record["attributes"]
    attributes["dynamic_object_points"] = []
    attributes["position"] = [1.0, 2.0, 3.0]
    attributes["bounding_box"] = {
        "type": "AABB",
        "world_R_center": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    attributes["mesh"] = {"points": [[0.1, 0.2, 0.3]]}
    matched = match_event_dsg_records(
        [{"ObjectSymbol": "O(0)", "AppearedAt": "0", "DisappearedAt": "10"}],
        [record],
        expected_count=1,
    )[0]

    assert np.allclose(
        dsg_interval_world_points(matched), np.asarray([[1.1, 2.2, 3.3]])
    )


def test_deterministic_target_package_has_equal_npz_and_manifest_hashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.bin"
    source.write_bytes(b"official")
    arrays = {
        "office.event.region": np.asarray([[2, 0, 0], [1, 0, 0]], dtype=np.int64),
        "apartment_event_01.revealed_background": np.asarray(
            [[0, 0, 1]], dtype=np.int64
        ),
    }
    metadata = {
        "event_object_count": 14,
        "background_observable_by_event": {"apartment_event_01": True},
        "background_observable_event_count": 1,
        "unobservable_revealed_target_event_count": 0,
    }

    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(arrays)
    first = write_target_package(
        tmp_path / "first",
        arrays=arrays,
        source_paths=[source],
        metadata=metadata,
    )
    second = write_target_package(
        tmp_path / "second",
        arrays=arrays,
        source_paths=[source],
        metadata=metadata,
    )

    assert (first.parent / "targets.npz").read_bytes() == (
        second.parent / "targets.npz"
    ).read_bytes()
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["status"] == "GENERATED"
    assert payload["targets_generated"] is True
    assert payload["prediction_inputs_used"] is False
    assert payload["target_arrays"]["count"] == 2
    assert payload["target_arrays"]["sha256"] == hashlib.sha256(
        (first.parent / "targets.npz").read_bytes()
    ).hexdigest()


def test_confirmed_free_checkpoint_may_be_empty_before_space_is_observed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.bin"
    source.write_bytes(b"official")
    arrays = {
        "apartment_event_01.region": np.asarray([[0, 0, 0]], dtype=np.int64),
        "apartment_event_01.confirmed_free.000263": np.empty((0, 3), dtype=np.int64),
    }

    manifest = write_target_package(
        tmp_path / "package",
        arrays=arrays,
        source_paths=[source],
        metadata={"event_object_count": 14},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["target_arrays"]["arrays"][
        "apartment_event_01.confirmed_free.000263"
    ]["element_count"] == 0


def test_unobservable_revealed_target_requires_explicit_prediction_independent_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.bin"
    source.write_bytes(b"official")
    arrays = {
        "apartment_event_01.region": np.asarray([[0, 0, 0]], dtype=np.int64),
        "apartment_event_01.revealed_background": np.asarray(
            [[0, 0, 1]], dtype=np.int64
        ),
        "apartment_event_02.region": np.asarray([[1, 0, 0]], dtype=np.int64),
        "apartment_event_02.revealed_background": np.empty((0, 3), dtype=np.int64),
    }
    metadata = {
        "event_object_count": 14,
        "background_observable_by_event": {
            "apartment_event_01": True,
            "apartment_event_02": False,
        },
        "background_observable_event_count": 1,
        "unobservable_revealed_target_event_count": 1,
    }

    manifest = write_target_package(
        tmp_path / "package",
        arrays=arrays,
        source_paths=[source],
        metadata=metadata,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["metadata"]["background_observable_by_event"] == {
        "apartment_event_01": True,
        "apartment_event_02": False,
    }


def test_revealed_target_observability_must_agree_in_both_directions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.bin"
    source.write_bytes(b"official")
    arrays = {
        "apartment_event_01.region": np.asarray([[0, 0, 0]], dtype=np.int64),
        "apartment_event_01.revealed_background": np.asarray(
            [[0, 0, 1]], dtype=np.int64
        ),
    }
    metadata = {
        "event_object_count": 14,
        "background_observable_by_event": {"apartment_event_01": False},
        "background_observable_event_count": 0,
        "unobservable_revealed_target_event_count": 1,
    }

    with pytest.raises(ValueError, match="disagrees with arrays"):
        write_target_package(
            tmp_path / "package",
            arrays=arrays,
            source_paths=[source],
            metadata=metadata,
        )


def test_fixture_integration_derives_nonempty_prediction_independent_targets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    contract = load_generation_contract(source, schedule)
    frames: dict[str, list[dict[str, object]]] = {}
    backgrounds: dict[str, np.ndarray] = {}
    for scene in ("apartment", "office"):
        intervention = int(
            contract["scenes"][scene]["schedule"]["events"][0][
                "intervention_frame_index"
            ]
        )
        frames[scene] = [
            {
                "frame_index": index,
                "depth": np.full(
                    (3, 3), 1.0 if index < intervention else 1.049, dtype=np.float32
                ),
                "world_from_camera": np.eye(4),
                "relative_timestamp_ns": index * 50,
            }
            for index in range(15)
        ]
        backgrounds[scene] = np.asarray([[0.0, 0.0, 1.06]])

    arrays, metadata = derive_target_arrays(
        contract,
        frames_by_scene=frames,
        background_points_by_scene=backgrounds,
        window_frames=2,
        event_limit=1,
    )
    progress = capsys.readouterr().out

    assert metadata["event_object_count"] == 14
    assert metadata["derived_event_count"] == 2
    assert metadata["window_frames"] == 2
    assert "COMMON_V2_PROGRESS event apartment_event_01" in progress
    assert "COMMON_V2_PROGRESS scene apartment semantic_complete" in progress
    assert "COMMON_V2_PROGRESS event office_event_01" in progress
    assert "COMMON_V2_PROGRESS scene office semantic_complete" in progress
    assert all(
        array.size > 0
        for name, array in arrays.items()
        if ".current_semantic." not in name
    )
    for scene in ("apartment", "office"):
        event_id = f"{scene}_event_01"
        assert f"{event_id}.region" in arrays
        assert f"{event_id}.revealed_background" in arrays
        assert any(
            name.startswith(f"{event_id}.confirmed_free.") for name in arrays
        )
        semantic_names = [
            name for name in arrays if name.startswith(f"{scene}.current_semantic.")
        ]
        assert semantic_names
        assert any(arrays[name].size > 0 for name in semantic_names)


def test_cli_fixture_bundle_is_deterministic_and_cannot_claim_full_targets(
    tmp_path: Path,
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    bundle = tmp_path / "official_fixture_bundle.npz"
    bundle_arrays: dict[str, np.ndarray] = {}
    for scene in ("apartment", "office"):
        intervention = 3
        bundle_arrays[f"{scene}.depth"] = np.stack(
            [
                np.full(
                    (3, 3), 1.0 if index < intervention else 1.049, dtype=np.float32
                )
                for index in range(15)
            ]
        )
        bundle_arrays[f"{scene}.world_from_camera"] = np.repeat(
            np.eye(4)[None, :, :], 15, axis=0
        )
        bundle_arrays[f"{scene}.background_points"] = np.asarray(
            [[0.0, 0.0, 1.06]]
        )
    np.savez(bundle, **bundle_arrays)

    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        assert (
            main(
                [
                    "--source-manifest",
                    str(source),
                    "--schedule",
                    str(schedule),
                    "--output-dir",
                    str(output),
                    "--fixture-bundle",
                    str(bundle),
                    "--window-frames",
                    "2",
                    "--event-limit",
                    "1",
                ]
            )
            == 0
        )
        outputs.append(output)

    assert (outputs[0] / "targets.npz").read_bytes() == (
        outputs[1] / "targets.npz"
    ).read_bytes()
    assert (outputs[0] / "manifest.json").read_bytes() == (
        outputs[1] / "manifest.json"
    ).read_bytes()
    payload = json.loads((outputs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FIXTURE"
    assert payload["targets_generated"] is True
    assert payload["metadata"]["protocol_complete"] is False


def test_derived_rgbd_loader_and_single_scene_smoke_cli(
    tmp_path: Path,
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    derived, lock = _write_derived_rgbd_fixture(tmp_path, source)
    contract = load_generation_contract(source, schedule)

    frames = load_derived_rgbd_frames(
        contract, "apartment", derived, lock, maximum_frame_index=14
    )

    assert len(frames) == 15
    assert frames[0]["depth"].dtype == np.float32
    assert np.allclose(frames[0]["depth"], 1.0)
    assert np.allclose(frames[3]["depth"], 1.049)
    output = tmp_path / "smoke"
    assert (
        main(
            [
                "--source-manifest",
                str(source),
                "--schedule",
                str(schedule),
                "--output-dir",
                str(output),
                "--derived-rgbd-root",
                str(derived),
                "--derived-rgbd-lock",
                str(lock),
                "--scene",
                "apartment",
                "--window-frames",
                "2",
                "--event-limit",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "SMOKE"
    assert payload["metadata"]["scenes"] == ["apartment"]
    with np.load(output / "targets.npz", allow_pickle=False) as targets:
        assert targets.files
        assert all(
            name.startswith("apartment_event_")
            or name.startswith("apartment.current_semantic.")
            for name in targets.files
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "apartment/results/frame000000.jpg",
        "apartment/results/depth000000.png",
        "apartment/traj.txt",
        "apartment/timestamps.csv",
        "cam_params.json",
    ],
)
def test_derived_rgbd_loader_rejects_mutated_export_bytes(
    tmp_path: Path, relative_path: str
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    derived, lock = _write_derived_rgbd_fixture(tmp_path, source)
    contract = load_generation_contract(source, schedule)
    artifact = derived / relative_path
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="combined output SHA256"):
        load_derived_rgbd_frames(
            contract, "apartment", derived, lock, maximum_frame_index=14
        )


def test_full_generation_rejects_depth_png_mutated_after_export(
    tmp_path: Path,
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    derived, lock = _write_derived_rgbd_fixture(tmp_path, source)
    depth = derived / "apartment/results/depth000003.png"
    depth.write_bytes(depth.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="combined output SHA256"):
        main(
            [
                "--source-manifest",
                str(source),
                "--schedule",
                str(schedule),
                "--output-dir",
                str(tmp_path / "output"),
                "--derived-rgbd-root",
                str(derived),
                "--derived-rgbd-lock",
                str(lock),
                "--scene",
                "apartment",
                "--window-frames",
                "2",
                "--event-limit",
                "1",
            ]
        )


@pytest.mark.parametrize("source_role", ["prediction", "unregistered"])
def test_derived_rgbd_loader_rejects_untrusted_lock_role(
    tmp_path: Path, source_role: str
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    derived, lock = _write_derived_rgbd_fixture(
        tmp_path, source, source_role=source_role
    )
    contract = load_generation_contract(source, schedule)

    with pytest.raises(ValueError, match="source role"):
        load_derived_rgbd_frames(
            contract,
            "apartment",
            derived,
            lock,
            maximum_frame_index=14,
        )


def test_generation_with_derived_rgbd_requires_checked_lock(tmp_path: Path) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    derived, _ = _write_derived_rgbd_fixture(tmp_path, source)

    with pytest.raises(ValueError, match="RGB-D lock"):
        main(
            [
                "--source-manifest",
                str(source),
                "--schedule",
                str(schedule),
                "--output-dir",
                str(tmp_path / "output"),
                "--derived-rgbd-root",
                str(derived),
                "--scene",
                "apartment",
                "--window-frames",
                "2",
                "--event-limit",
                "1",
            ]
        )


def test_derived_rgbd_lock_rejects_export_manifest_self_report_change(
    tmp_path: Path,
) -> None:
    source, schedule = _write_generation_fixture(tmp_path)
    derived, lock = _write_derived_rgbd_fixture(tmp_path, source)
    export_manifest = derived / "apartment/export_manifest.json"
    payload = json.loads(export_manifest.read_text(encoding="utf-8"))
    payload["pose"] = "untrusted self-report"
    export_manifest.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="export manifest SHA256"):
        load_derived_rgbd_frames(
            load_generation_contract(source, schedule),
            "apartment",
            derived,
            lock,
            maximum_frame_index=14,
        )


def test_derived_rgbd_parses_the_same_depth_bytes_used_for_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    from scripts.evaluation import export_tesse_cd_rgbd

    source, schedule = _write_generation_fixture(tmp_path)
    derived, lock = _write_derived_rgbd_fixture(tmp_path, source)
    target = (derived / "apartment/results/depth000000.png").resolve()
    replacement = io.BytesIO()
    Image.fromarray(np.full((3, 3), 9000, dtype=np.uint16)).save(
        replacement, format="PNG"
    )
    replacement_bytes = replacement.getvalue()
    original_read = export_tesse_cd_rgbd._read_bound_file

    def read_then_mutate(path: Path) -> bytes:
        content = original_read(path)
        if path.resolve() == target:
            path.write_bytes(replacement_bytes)
        return content

    monkeypatch.setattr(export_tesse_cd_rgbd, "_read_bound_file", read_then_mutate)
    frames = load_derived_rgbd_frames(
        load_generation_contract(source, schedule),
        "apartment",
        derived,
        lock,
        maximum_frame_index=14,
    )

    assert np.allclose(frames[0]["depth"], 1.0)
