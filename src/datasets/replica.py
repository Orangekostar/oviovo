"""Replica dataset adapters."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

from src.core.data_structures import CameraIntrinsics, Frame

REPLICA_NICESLAM_INTRINSICS = CameraIntrinsics(
    fx=600.0,
    fy=600.0,
    cx=599.5,
    cy=339.5,
    width=1200,
    height=680,
)
REPLICA_NICESLAM_DEPTH_SCALE = 6553.5

_FRAME_INDEX_PATTERN = re.compile(r"(\d+)")


@dataclass(frozen=True)
class _FrameRecord:
    """Resolved on-disk paths and pose for one Replica frame."""

    frame_index: int
    rgb_path: Path
    depth_path: Path
    pose: np.ndarray


class ReplicaRoom0Dataset:
    """Minimal adapter for Replica room0 RGB-D frames.

    The loader supports both the documented NICE-SLAM Replica structure:

    - ``room0/results/frameXXXXXX.jpg``
    - ``room0/results/depthXXXXXX.png``
    - ``room0/traj.txt``

    and the flattened structure found in this workspace:

    - ``room0/rgb/frameXXXXXX.jpg``
    - ``room0/depth/depthXXXXXX.png``
    - ``room0/traj.txt``
    - optional ``room0/frame_manifest.txt``
    """

    def __init__(
        self,
        root: str | Path,
        intrinsics: CameraIntrinsics | None = None,
        depth_scale: float = REPLICA_NICESLAM_DEPTH_SCALE,
        timestamp_step: float = 0.0,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Replica root does not exist: {self.root}")

        self.layout, self.rgb_dir, self.depth_dir = self._resolve_layout(self.root)
        self.traj_path = self.root / "traj.txt"
        self.manifest_path = self.root / "frame_manifest.txt"
        self.depth_scale = float(depth_scale)
        self.timestamp_step = float(timestamp_step)

        self._records, self.pose_source = self._build_records()
        if not self._records:
            raise ValueError(f"No Replica frames discovered under {self.root}")

        width, height = self._inspect_image_size(self._records[0].rgb_path)
        if intrinsics is None:
            self.intrinsics = self._build_default_intrinsics(width=width, height=height)
            self.intrinsics_source = "NICE-SLAM Replica defaults"
        else:
            if intrinsics.width != width or intrinsics.height != height:
                raise ValueError(
                    "Explicit intrinsics resolution does not match RGB image size: "
                    f"got {(intrinsics.width, intrinsics.height)}, expected {(width, height)}"
                )
            self.intrinsics = intrinsics
            self.intrinsics_source = "explicit override"

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, item: int) -> Frame:
        record = self._records[item]
        with Image.open(record.rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(record.depth_path) as image:
            depth_raw = np.asarray(image, dtype=np.float32)
        depth_m = (depth_raw / self.depth_scale).astype(np.float32)

        timestamp = 0.0
        if self.timestamp_step > 0.0:
            timestamp = record.frame_index * self.timestamp_step

        return Frame(
            frame_id=record.frame_index,
            rgb=rgb,
            depth=depth_m,
            pose=record.pose.copy(),
            intrinsics=self.intrinsics,
            timestamp=timestamp,
        )

    @property
    def frame_indices(self) -> list[int]:
        """Dataset-order frame indices resolved from filenames."""
        return [record.frame_index for record in self._records]

    def iter_frames(self, limit: int | None = None) -> Iterator[Frame]:
        """Iterate over frames in dataset order, optionally truncated."""
        count = len(self)
        if limit is not None:
            count = min(count, max(limit, 0))
        for index in range(count):
            yield self[index]

    def summary(self) -> dict[str, object]:
        """Return a compact summary of the discovered dataset layout."""
        return {
            "root": str(self.root),
            "layout": self.layout,
            "rgb_dir": str(self.rgb_dir),
            "depth_dir": str(self.depth_dir),
            "traj_path": str(self.traj_path) if self.traj_path.exists() else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path.exists() else None,
            "frame_count": len(self),
            "pose_source": self.pose_source,
            "intrinsics_source": self.intrinsics_source,
            "image_size": (self.intrinsics.height, self.intrinsics.width),
            "depth_scale": self.depth_scale,
        }

    @staticmethod
    def _resolve_layout(root: Path) -> tuple[str, Path, Path]:
        if (root / "rgb").is_dir() and (root / "depth").is_dir():
            return "flat_rgb_depth", root / "rgb", root / "depth"

        results_dir = root / "results"
        if results_dir.is_dir():
            return "results_dir", results_dir, results_dir

        raise FileNotFoundError(
            "Unsupported Replica layout. Expected either 'rgb/ + depth/' or 'results/'."
        )

    def _build_records(self) -> tuple[list[_FrameRecord], str]:
        manifest_records = self._build_records_from_manifest()
        if manifest_records:
            return manifest_records, "frame_manifest.txt"

        traj_records = self._build_records_from_traj()
        if traj_records:
            return traj_records, "traj.txt"

        return [], "unresolved"

    def _build_records_from_manifest(self) -> list[_FrameRecord]:
        if not self.manifest_path.exists():
            return []

        rgb_files = self._index_files(self._list_rgb_files())
        depth_files = self._index_files(self._list_depth_files())
        records: list[_FrameRecord] = []

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame_index = int(row["idx"])
                rgb_path = rgb_files.get(frame_index)
                depth_path = depth_files.get(frame_index)
                if rgb_path is None or depth_path is None:
                    continue
                pose = self._parse_pose16(row["pose16"])
                records.append(
                    _FrameRecord(
                        frame_index=frame_index,
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        pose=pose,
                    )
                )

        return records

    def _build_records_from_traj(self) -> list[_FrameRecord]:
        if not self.traj_path.exists():
            return []

        poses = self._load_traj_poses(self.traj_path)
        rgb_files = self._index_files(self._list_rgb_files())
        depth_files = self._index_files(self._list_depth_files())

        common_indices = sorted(set(rgb_files) & set(depth_files))
        records: list[_FrameRecord] = []
        for frame_index in common_indices:
            if frame_index >= len(poses):
                continue
            records.append(
                _FrameRecord(
                    frame_index=frame_index,
                    rgb_path=rgb_files[frame_index],
                    depth_path=depth_files[frame_index],
                    pose=poses[frame_index],
                )
            )
        return records

    def _list_rgb_files(self) -> Sequence[Path]:
        return sorted(self.rgb_dir.glob("frame*.jpg"))

    def _list_depth_files(self) -> Sequence[Path]:
        return sorted(self.depth_dir.glob("depth*.png"))

    @staticmethod
    def _index_files(paths: Sequence[Path]) -> dict[int, Path]:
        indexed: dict[int, Path] = {}
        for path in paths:
            indexed[ReplicaRoom0Dataset._extract_frame_index(path)] = path
        return indexed

    @staticmethod
    def _extract_frame_index(path: Path) -> int:
        match = _FRAME_INDEX_PATTERN.search(path.stem)
        if match is None:
            raise ValueError(f"Could not extract frame index from {path.name}")
        return int(match.group(1))

    @staticmethod
    def _parse_pose16(pose16: str) -> np.ndarray:
        values = np.fromstring(pose16, sep=" ", dtype=np.float64)
        if values.size != 16:
            raise ValueError(f"Expected 16 pose values, found {values.size}")
        return values.reshape(4, 4)

    @staticmethod
    def _load_traj_poses(path: Path) -> list[np.ndarray]:
        poses: list[np.ndarray] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                poses.append(ReplicaRoom0Dataset._parse_pose16(stripped))
        return poses

    @staticmethod
    def _inspect_image_size(path: Path) -> tuple[int, int]:
        with Image.open(path) as image:
            width, height = image.size
        return width, height

    @staticmethod
    def _build_default_intrinsics(width: int, height: int) -> CameraIntrinsics:
        scale_x = width / REPLICA_NICESLAM_INTRINSICS.width
        scale_y = height / REPLICA_NICESLAM_INTRINSICS.height
        return CameraIntrinsics(
            fx=REPLICA_NICESLAM_INTRINSICS.fx * scale_x,
            fy=REPLICA_NICESLAM_INTRINSICS.fy * scale_y,
            cx=REPLICA_NICESLAM_INTRINSICS.cx * scale_x,
            cy=REPLICA_NICESLAM_INTRINSICS.cy * scale_y,
            width=width,
            height=height,
        )
