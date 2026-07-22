#!/usr/bin/env python3
"""Export official TESSE-CD ROS2 bags to one Replica-style RGB-D layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(quaternion @ quaternion)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    quaternion *= math.sqrt(2.0 / norm)
    outer = np.outer(quaternion, quaternion)
    return np.asarray(
        [
            [1.0 - outer[2, 2] - outer[3, 3], outer[1, 2] - outer[3, 0], outer[1, 3] + outer[2, 0]],
            [outer[1, 2] + outer[3, 0], 1.0 - outer[1, 1] - outer[3, 3], outer[2, 3] - outer[1, 0]],
            [outer[1, 3] - outer[2, 0], outer[2, 3] + outer[1, 0], 1.0 - outer[1, 1] - outer[2, 2]],
        ],
        dtype=np.float64,
    )


def _matrix(translation: Any, rotation: Any) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_matrix(
        rotation.x, rotation.y, rotation.z, rotation.w
    )
    result[:3, 3] = [translation.x, translation.y, translation.z]
    return result


def camera_pose_matrix(body_pose: Any, body_from_sensor: Any) -> np.ndarray:
    world_from_body = _matrix(body_pose.position, body_pose.orientation)
    body_from_camera = _matrix(
        body_from_sensor.translation, body_from_sensor.rotation
    )
    return world_from_body @ body_from_camera


def depth_meters_to_mm(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0)
    output = np.zeros(values.shape, dtype=np.uint16)
    millimeters = np.rint(values[valid] * 1000.0)
    output[valid] = np.clip(millimeters, 0.0, 65535.0).astype(np.uint16)
    return output


def _decode_image(message: Any, *, dtype: np.dtype[Any], channels: int) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    if message.step % itemsize:
        raise ValueError(f"image step is not aligned to {np.dtype(dtype)}")
    values_per_row = message.step // itemsize
    values = np.frombuffer(message.data, dtype=dtype).reshape(
        message.height, values_per_row
    )
    used = values[:, : message.width * channels]
    shape = (
        (message.height, message.width)
        if channels == 1
        else (message.height, message.width, channels)
    )
    return np.asarray(used.reshape(shape))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_export_output_binding(
    output_root: Path, scene: str, frame_count: int
) -> dict[str, object]:
    """Recompute the exact per-file binding recorded by ``export_scene``."""
    if not scene or type(frame_count) is not int or frame_count < 0:
        raise ValueError("export binding requires a scene and non-negative frame count")
    scene_dir = output_root / scene
    results_dir = scene_dir / "results"
    paths = [
        path
        for index in range(frame_count)
        for path in (
            results_dir / f"frame{index:06d}.jpg",
            results_dir / f"depth{index:06d}.png",
        )
    ] + [
        scene_dir / "traj.txt",
        scene_dir / "timestamps.csv",
        output_root / "cam_params.json",
    ]
    output_hashes: list[tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"exported RGB-D artifact is not a file: {path}")
        output_hashes.append(
            (path.relative_to(output_root).as_posix(), _sha256(path))
        )
    digest = hashlib.sha256()
    for relative, file_hash in sorted(output_hashes):
        digest.update(
            relative.encode("utf-8")
            + b"\0"
            + file_hash.encode("ascii")
            + b"\n"
        )
    return {
        "combined_output_sha256": digest.hexdigest(),
        "file_hash_count": len(output_hashes),
    }


def _load_pose_inputs(bag: Path) -> tuple[dict[int, Any], Any]:
    from rosbags.highlevel import AnyReader

    poses: dict[int, Any] = {}
    extrinsic = None
    with AnyReader([bag]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in {"/tesse/odom", "/tf_static"}
        ]
        for connection, timestamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic == "/tesse/odom":
                poses[int(timestamp)] = message.pose.pose
                continue
            for transform in message.transforms:
                if (
                    transform.header.frame_id == "base_link_gt"
                    and transform.child_frame_id == "left_cam"
                ):
                    candidate = transform.transform
                    if extrinsic is not None:
                        previous = _matrix(extrinsic.translation, extrinsic.rotation)
                        current = _matrix(candidate.translation, candidate.rotation)
                        if not np.allclose(previous, current, atol=1e-9):
                            raise ValueError("TESSE-CD left camera extrinsics disagree")
                    extrinsic = candidate
    if not poses:
        raise ValueError("TESSE-CD odometry topic is empty")
    if extrinsic is None:
        raise ValueError("TESSE-CD left camera extrinsics are missing")
    return poses, extrinsic


def export_scene(manifest_path: Path, scene: str, output_root: Path) -> Path:
    import cv2
    from rosbags.highlevel import AnyReader

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence = manifest["sequences"][scene]
    bag = Path(sequence["bag"]["directory"])
    expected_frames = int(sequence["timeline"]["depth_frame_count"])
    output_root.mkdir(parents=True, exist_ok=True)
    scene_dir = output_root / scene
    scene_dir.mkdir(parents=False, exist_ok=False)
    results_dir = scene_dir / "results"
    results_dir.mkdir()

    camera = manifest["camera"]
    camera_payload = {
        "camera": {
            "h": int(camera["height"]),
            "w": int(camera["width"]),
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "scale": 1000.0,
        }
    }
    camera_path = output_root / "cam_params.json"
    camera_text = json.dumps(camera_payload, indent=2, sort_keys=True) + "\n"
    if camera_path.exists() and camera_path.read_text(encoding="utf-8") != camera_text:
        raise ValueError("existing TESSE-CD camera parameters disagree")
    camera_path.write_text(camera_text, encoding="utf-8")

    poses, body_from_camera = _load_pose_inputs(bag)
    pending_rgb: dict[int, Any] = {}
    pending_depth: dict[int, Any] = {}
    rows: list[tuple[int, int, int]] = []
    trajectory_path = scene_dir / "traj.txt"
    with trajectory_path.open("w", encoding="utf-8") as trajectory:
        with AnyReader([bag]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic
                in {
                    "/tesse/left_cam/rgb/image_raw",
                    "/tesse/depth_cam/mono/image_raw",
                }
            ]
            for connection, timestamp, raw in reader.messages(
                connections=connections
            ):
                message = reader.deserialize(raw, connection.msgtype)
                timestamp = int(timestamp)
                if connection.topic == "/tesse/left_cam/rgb/image_raw":
                    if message.encoding.lower() != "rgb8":
                        raise ValueError(f"unexpected TESSE-CD RGB encoding: {message.encoding}")
                    pending_rgb[timestamp] = message
                else:
                    if message.encoding.upper() != "32FC1":
                        raise ValueError(f"unexpected TESSE-CD depth encoding: {message.encoding}")
                    pending_depth[timestamp] = message
                if timestamp not in pending_rgb or timestamp not in pending_depth:
                    continue
                if timestamp not in poses:
                    raise ValueError(f"TESSE-CD frame has no exact odometry: {timestamp}")

                index = len(rows)
                rgb = _decode_image(
                    pending_rgb.pop(timestamp), dtype=np.dtype("u1"), channels=3
                )
                depth = _decode_image(
                    pending_depth.pop(timestamp), dtype=np.dtype("<f4"), channels=1
                )
                rgb_path = results_dir / f"frame{index:06d}.jpg"
                depth_path = results_dir / f"depth{index:06d}.png"
                if not cv2.imwrite(
                    str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]
                ):
                    raise RuntimeError(f"failed to write {rgb_path}")
                if not cv2.imwrite(str(depth_path), depth_meters_to_mm(depth)):
                    raise RuntimeError(f"failed to write {depth_path}")
                pose = camera_pose_matrix(poses[timestamp], body_from_camera)
                trajectory.write(" ".join(f"{value:.12g}" for value in pose.reshape(-1)) + "\n")
                rows.append((index, timestamp, timestamp - int(sequence["timeline"]["first_depth_timestamp_ns"])))

    if pending_rgb or pending_depth:
        raise ValueError("TESSE-CD RGB/depth timestamps do not match exactly")
    if len(rows) != expected_frames:
        raise ValueError(
            f"TESSE-CD exported {len(rows)} frames; expected {expected_frames}"
        )
    timestamps_path = scene_dir / "timestamps.csv"
    with timestamps_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame_index", "sensor_timestamp_ns", "relative_timestamp_ns"))
        writer.writerows(rows)
    output_binding = compute_export_output_binding(output_root, scene, len(rows))
    export_manifest = scene_dir / "export_manifest.json"
    export_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "scene": scene,
                "frame_count": len(rows),
                "source_manifest": str(manifest_path.resolve()),
                "source_database": str(Path(sequence["bag"]["database"]["path"]).resolve()),
                "source_database_sha256": sequence["bag"]["database"]["sha256"],
                "rgb_encoding": "JPEG quality 95 decoded from official rgb8",
                "depth_encoding": "uint16 millimeters decoded from official 32FC1 meters",
                "pose": "world_T_base_link_gt multiplied by bag tf_static base_link_gt_T_left_cam",
                **output_binding,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return export_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", choices=("apartment", "office"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    export_scene(args.manifest, args.scene, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
