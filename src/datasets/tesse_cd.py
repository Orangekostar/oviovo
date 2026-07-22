"""Validated read-only RGB-D adapter for the TESSE-CD benchmark export."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np
from PIL import Image

from src.core.data_structures import CameraIntrinsics, Frame


@dataclass(frozen=True)
class TesseCdFrameRecord:
    """Resolved immutable metadata for one exported TESSE-CD RGB-D frame."""

    frame_index: int
    timestamp_ns: int
    relative_timestamp_ns: int
    rgb_path: Path
    depth_path: Path
    camera_to_world: np.ndarray


class TesseCdRgbdDataset:
    """Load one complete, frozen TESSE-CD scene without evaluation inputs."""

    EXPECTED_FRAMES = {"apartment": 1745, "office": 4346}
    OFFICIAL_FRAME_COUNTS = {"apartment": 1745, "office": 4346}
    WIDTH = 720
    HEIGHT = 480
    DEPTH_SCALE = 1000.0
    CAMERA_SHA256 = "73fe7357e25e4708bc6554bb2a0fc925de00e8368d06a18f9eb3eadd9cb808eb"
    SCHEDULE_SHA256 = "fb97bacee377f9fd67ee9dae8064dc6f33ac32d129ee633629fec4164d5003e0"
    EXPORT_SHA256 = {
        "apartment": "ee35d1843540fc88ae20b5dae5ad34caf192d50f0cab7c5564c47c454d4d5b81",
        "office": "16068d00033c8f8407d143d9e50f2008a5f4feb22c134abd0059fc551637f1f6",
    }
    INPUT_SHA256 = {
        "apartment": {
            "timestamps": "bc369c8c5b4bca30b87c5617bc8c17076c90e5efc00167490cf5c9361af5e83d",
            "trajectory": "5553fdfc32e0289965d19f4735316445fc07c2eee0ffe57b3f637a3e6e53bc45",
            "database": "43a30e694b8deabf4861b2b20a8f32a5c18c6288f37a886289783af0f26dab07",
        },
        "office": {
            "timestamps": "a73591a0703d94c6228de1e34f0d93e9ec2892ec72f88ebc35c5e5a945514491",
            "trajectory": "06a31ee46540b4e67f42e373592c746bad097b1024ffdc94196642787d0d968e",
            "database": "4305b56149378799bfa468da2ac0acd84716bef59aba004f8e4feb34ec0d3b51",
        },
    }

    def __init__(
        self,
        root: str | Path,
        scene: str,
        export_manifest: str | Path,
        schedule_manifest: str | Path,
    ) -> None:
        if scene not in self.EXPECTED_FRAMES:
            raise ValueError("scene must be exactly 'apartment' or 'office'")
        self.scene = scene
        self.root, self.camera_path = self._resolve_roots(Path(root), scene)
        self.results_dir = self.root / "results"
        self.export_manifest_path = Path(export_manifest)
        self.schedule_manifest_path = Path(schedule_manifest)
        self.timestamps_path = self.root / "timestamps.csv"
        self.trajectory_path = self.root / "traj.txt"

        for path, role in (
            (self.export_manifest_path, "export manifest"),
            (self.schedule_manifest_path, "schedule manifest"),
            (self.camera_path, "camera manifest"),
            (self.timestamps_path, "timestamps"),
            (self.trajectory_path, "trajectory"),
        ):
            self._require_regular_file(path, role)
        if self.export_manifest_path.resolve() != (self.root / "export_manifest.json").resolve():
            raise ValueError("export manifest must be the selected scene export_manifest.json")

        export = self._read_json(self.export_manifest_path, "export manifest")
        schedule = self._read_json(self.schedule_manifest_path, "schedule manifest")
        source = self._validate_manifest_chain(export, schedule)
        self.intrinsics = self._load_camera(source)

        expected_count = self.EXPECTED_FRAMES[scene]
        rgb_paths, depth_paths = self._validate_frame_files(expected_count)
        timestamps = self._load_timestamps(expected_count, schedule)
        poses = self._load_poses(expected_count)
        self._validate_official_hashes(expected_count)

        self.records = tuple(
            TesseCdFrameRecord(
                frame_index=index,
                timestamp_ns=timestamp_ns,
                relative_timestamp_ns=relative_timestamp_ns,
                rgb_path=rgb_paths[index],
                depth_path=depth_paths[index],
                camera_to_world=poses[index],
            )
            for index, (timestamp_ns, relative_timestamp_ns) in enumerate(timestamps)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Frame:
        record = self.records[index]
        with Image.open(record.rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(record.depth_path) as image:
            depth_mm = np.asarray(image, dtype=np.float32)
        depth_m = np.ascontiguousarray(depth_mm / self.DEPTH_SCALE, dtype=np.float32)
        return Frame(
            frame_id=record.frame_index,
            source_frame_id=record.frame_index,
            rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
            depth=depth_m,
            pose=record.camera_to_world.copy(),
            intrinsics=self.intrinsics,
            timestamp=record.timestamp_ns / 1_000_000_000,
        )

    def timestamp_ns(self, index: int) -> int:
        return self.records[index].timestamp_ns

    @classmethod
    def _resolve_roots(cls, root: Path, scene: str) -> tuple[Path, Path]:
        if (root / "results").is_dir():
            return root, root.parent / "cam_params.json"
        scene_root = root / scene
        if (scene_root / "results").is_dir():
            return scene_root, root / "cam_params.json"
        raise FileNotFoundError(
            f"TESSE-CD scene results directory does not exist under {root}"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _require_regular_file(path: Path, role: str) -> None:
        try:
            status = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"{role} must be a regular non-symlink file: {path}") from exc
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"{role} must be a regular non-symlink file: {path}")

    @classmethod
    def _read_json(cls, path: Path, role: str) -> Mapping[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{role} must be valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{role} must contain a JSON object")
        return payload

    @staticmethod
    def _mapping(value: object, role: str) -> Mapping[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{role} must be a mapping")
        return value

    @staticmethod
    def _hash_text(value: object, role: str) -> str:
        digest = str(value).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{role} must be a SHA-256 digest")
        return digest

    def _validate_manifest_chain(
        self,
        export: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        expected_count = self.EXPECTED_FRAMES[self.scene]
        if (
            export.get("dataset") != "TESSE-CD"
            or export.get("scene") != self.scene
            or export.get("frame_count") != expected_count
        ):
            raise ValueError("export manifest identity or frame count mismatch")
        self._hash_text(export.get("combined_output_sha256"), "export output hash")

        if schedule.get("dataset") != "TESSE-CD":
            raise ValueError("schedule manifest dataset mismatch")
        scene_schedule = self._mapping(
            self._mapping(schedule.get("scenes"), "schedule scenes").get(self.scene),
            f"schedule scene {self.scene}",
        )
        if scene_schedule.get("frame_count") != expected_count:
            raise ValueError("schedule frame count mismatch")

        source_binding = self._mapping(
            schedule.get("source_manifest"), "schedule source manifest"
        )
        source_hash = self._hash_text(
            source_binding.get("sha256"), "schedule source manifest hash"
        )
        source_path = self._resolve_source_manifest(
            str(source_binding.get("path", "")),
            source_hash,
        )
        export_source_name = Path(str(export.get("source_manifest", ""))).name
        if export_source_name != source_path.name:
            raise ValueError("export and schedule source manifests disagree")
        source = self._read_json(source_path, "source manifest")
        if source.get("dataset") != "TESSE-CD":
            raise ValueError("source manifest dataset mismatch")

        sequence = self._mapping(
            self._mapping(source.get("sequences"), "source sequences").get(self.scene),
            f"source sequence {self.scene}",
        )
        source_database = self._hash_text(
            self._mapping(
                self._mapping(sequence.get("bag"), "source bag").get("database"),
                "source database",
            ).get("sha256"),
            "source database hash",
        )
        schedule_database = self._hash_text(
            self._mapping(
                self._mapping(scene_schedule.get("sources"), "schedule sources").get(
                    "database"
                ),
                "schedule database",
            ).get("sha256"),
            "schedule database hash",
        )
        export_database = self._hash_text(
            export.get("source_database_sha256"), "export database hash"
        )
        if len({source_database, schedule_database, export_database}) != 1:
            raise ValueError("export database hash disagrees with frozen manifest chain")

        timeline = self._mapping(sequence.get("timeline"), "source timeline")
        if timeline.get("depth_frame_count") != expected_count:
            raise ValueError("source timeline frame count mismatch")
        return source

    def _resolve_source_manifest(self, declared: str, expected_hash: str) -> Path:
        candidates = (
            Path(declared),
            self.schedule_manifest_path.parent / Path(declared).name,
            self.export_manifest_path.parent / Path(declared).name,
        )
        for candidate in candidates:
            try:
                self._require_regular_file(candidate, "source manifest")
            except ValueError:
                continue
            if self._sha256(candidate) == expected_hash:
                return candidate
        raise ValueError("schedule source manifest hash mismatch")

    def _load_camera(self, source: Mapping[str, Any]) -> CameraIntrinsics:
        camera_document = self._read_json(self.camera_path, "camera manifest")
        camera = self._mapping(camera_document.get("camera"), "camera parameters")
        source_camera = self._mapping(source.get("camera"), "source camera")
        values = {
            "width": int(camera.get("w", -1)),
            "height": int(camera.get("h", -1)),
            "fx": float(camera.get("fx", float("nan"))),
            "fy": float(camera.get("fy", float("nan"))),
            "cx": float(camera.get("cx", float("nan"))),
            "cy": float(camera.get("cy", float("nan"))),
        }
        expected = {
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "fx": 415.69219381653056,
            "fy": 415.69219381653056,
            "cx": 360.0,
            "cy": 240.0,
        }
        if float(camera.get("scale", float("nan"))) != self.DEPTH_SCALE:
            raise ValueError("camera depth scale mismatch")
        source_mismatch = any(
            source_camera.get(key) != value for key, value in expected.items()
        )
        if values != expected or source_mismatch:
            raise ValueError("camera intrinsics mismatch")
        if self._sha256(self.camera_path) != self.CAMERA_SHA256:
            raise ValueError("camera manifest hash mismatch")
        return CameraIntrinsics(**values)

    def _validate_frame_files(
        self,
        expected_count: int,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        rgb_paths = tuple(
            self.results_dir / f"frame{index:06d}.jpg" for index in range(expected_count)
        )
        depth_paths = tuple(
            self.results_dir / f"depth{index:06d}.png" for index in range(expected_count)
        )
        discovered_rgb = set(self.results_dir.glob("frame*.jpg"))
        discovered_depth = set(self.results_dir.glob("depth*.png"))
        if discovered_rgb != set(rgb_paths):
            raise ValueError("RGB frame files must form the complete contiguous sequence")
        if discovered_depth != set(depth_paths):
            raise ValueError("depth frame files must form the complete contiguous sequence")

        for index, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
            self._require_regular_file(rgb_path, f"RGB frame {index}")
            self._require_regular_file(depth_path, f"depth frame {index}")
            with Image.open(rgb_path) as image:
                if image.size != (self.WIDTH, self.HEIGHT):
                    raise ValueError(f"RGB frame {index} size mismatch")
                if len(image.getbands()) != 3:
                    raise ValueError(f"RGB frame {index} shape mismatch")
            with Image.open(depth_path) as image:
                raw_mode = image.tile[0][3] if image.tile else image.mode
                if image.size != (self.WIDTH, self.HEIGHT):
                    raise ValueError(f"depth frame {index} shape mismatch")
                if raw_mode not in {"I;16", "I;16B", "I;16L"}:
                    raise ValueError(f"depth frame {index} source dtype must be uint16")
        return rgb_paths, depth_paths

    def _load_timestamps(
        self,
        expected_count: int,
        schedule: Mapping[str, Any],
    ) -> tuple[tuple[int, int], ...]:
        rows: list[tuple[int, int]] = []
        with self.timestamps_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_fields = [
                "frame_index",
                "sensor_timestamp_ns",
                "relative_timestamp_ns",
            ]
            if reader.fieldnames != expected_fields:
                raise ValueError("timestamps CSV header mismatch")
            for expected_index, row in enumerate(reader):
                try:
                    frame_index = int(row["frame_index"])
                    timestamp_ns = int(row["sensor_timestamp_ns"])
                    relative_timestamp_ns = int(row["relative_timestamp_ns"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("timestamps CSV contains an invalid row") from exc
                if frame_index != expected_index:
                    raise ValueError("timestamp frame indices must be contiguous")
                rows.append((timestamp_ns, relative_timestamp_ns))
        if len(rows) != expected_count:
            raise ValueError("timestamp frame count mismatch")
        if any(current[0] >= following[0] for current, following in zip(rows, rows[1:])):
            raise ValueError("timestamps must be strictly increasing")
        first_timestamp = rows[0][0]
        if any(relative != timestamp - first_timestamp for timestamp, relative in rows):
            raise ValueError("relative timestamps do not match sensor timestamps")

        scene_schedule = self._mapping(
            self._mapping(schedule.get("scenes"), "schedule scenes").get(self.scene),
            f"schedule scene {self.scene}",
        )
        if (
            scene_schedule.get("first_depth_timestamp_ns") != rows[0][0]
            or scene_schedule.get("last_depth_timestamp_ns") != rows[-1][0]
        ):
            raise ValueError("timestamps do not match the causal schedule")
        return tuple(rows)

    def _load_poses(self, expected_count: int) -> tuple[np.ndarray, ...]:
        lines = self.trajectory_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != expected_count:
            raise ValueError("pose frame count mismatch")
        poses: list[np.ndarray] = []
        for index, line in enumerate(lines):
            values = np.fromstring(line, sep=" ", dtype=np.float64)
            if values.size != 16:
                raise ValueError(f"pose {index} must contain 16 values")
            pose = values.reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                raise ValueError(f"pose {index} must be finite")
            if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
                raise ValueError(f"pose {index} must be a homogeneous transform")
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-5
            ):
                raise ValueError(f"pose {index} rotation must be rigid")
            pose = np.asarray(pose, dtype=np.float64)
            pose.setflags(write=False)
            poses.append(pose)
        return tuple(poses)

    def _validate_official_hashes(self, expected_count: int) -> None:
        if expected_count != self.OFFICIAL_FRAME_COUNTS[self.scene]:
            return
        if self._sha256(self.export_manifest_path) != self.EXPORT_SHA256[self.scene]:
            raise ValueError("export manifest hash mismatch")
        if self._sha256(self.schedule_manifest_path) != self.SCHEDULE_SHA256:
            raise ValueError("schedule manifest hash mismatch")
        expected = self.INPUT_SHA256[self.scene]
        if self._sha256(self.timestamps_path) != expected["timestamps"]:
            raise ValueError("timestamps hash mismatch")
        if self._sha256(self.trajectory_path) != expected["trajectory"]:
            raise ValueError("trajectory hash mismatch")


__all__ = ["TesseCdFrameRecord", "TesseCdRgbdDataset"]
