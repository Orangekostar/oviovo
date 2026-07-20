"""ScanNet200 RGB-D dataset adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
import stat
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from src.core.data_structures import CameraIntrinsics, Frame


class ScanNet200Dataset:
    """Load an explicit ordered selection of exported ScanNet RGB-D frames."""

    def __init__(
        self,
        root: str | Path,
        *,
        source_frame_ids: Sequence[int],
        input_hashes: Mapping[int, Mapping[str, str]] | None = None,
        expected_image_shape: tuple[int, int] = (480, 640),
        depth_scale: float = 1000.0,
    ) -> None:
        self.root = Path(root)
        self._frame_indices = tuple(int(value) for value in source_frame_ids)
        self.expected_image_shape = tuple(int(value) for value in expected_image_shape)
        self.depth_scale = float(depth_scale)
        if not self._frame_indices or len(set(self._frame_indices)) != len(
            self._frame_indices
        ):
            raise ValueError("source_frame_ids must contain unique source frame IDs")
        if any(value < 0 for value in self._frame_indices):
            raise ValueError("source_frame_ids must be non-negative")
        if (
            len(self.expected_image_shape) != 2
            or any(value <= 0 for value in self.expected_image_shape)
        ):
            raise ValueError("expected_image_shape must contain two positive dimensions")
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0.0:
            raise ValueError("depth_scale must be finite and positive")
        self.input_hashes = (
            None
            if input_hashes is None
            else {int(key): dict(value) for key, value in input_hashes.items()}
        )

        intrinsic_path = self.root / "intrinsic" / "intrinsic_depth.txt"
        self._require_regular_file(intrinsic_path, "intrinsic")
        intrinsic = np.loadtxt(
            intrinsic_path,
            dtype=np.float64,
        )
        if intrinsic.shape != (4, 4) or not np.all(np.isfinite(intrinsic)):
            raise ValueError("depth intrinsic must be a finite 4x4 matrix")
        height, width = self.expected_image_shape
        self.intrinsics = CameraIntrinsics(
            fx=float(intrinsic[0, 0]),
            fy=float(intrinsic[1, 1]),
            cx=float(intrinsic[0, 2]),
            cy=float(intrinsic[1, 2]),
            width=width,
            height=height,
        )
        if self.intrinsics.fx <= 0.0 or self.intrinsics.fy <= 0.0:
            raise ValueError("depth focal lengths must be positive")
        for source_frame_id in self._frame_indices:
            self._validate_frame_inputs(source_frame_id)
        self._poses = {
            source_frame_id: self._load_pose(source_frame_id)
            for source_frame_id in self._frame_indices
        }

    def __len__(self) -> int:
        return len(self._frame_indices)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return self._frame_indices

    def __getitem__(self, index: int) -> Frame:
        source_frame_id = self._frame_indices[index]
        with Image.open(self.root / "color" / f"{source_frame_id}.jpg") as image:
            rgb = np.asarray(
                image.convert("RGB").resize(
                    (self.intrinsics.width, self.intrinsics.height),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
        with Image.open(self.root / "depth" / f"{source_frame_id}.png") as image:
            depth = np.asarray(image, dtype=np.float32) / self.depth_scale
        return Frame(
            frame_id=source_frame_id,
            source_frame_id=source_frame_id,
            rgb=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth, dtype=np.float32),
            pose=self._poses[source_frame_id].copy(),
            intrinsics=self.intrinsics,
            timestamp=float(source_frame_id),
        )

    def _load_pose(self, source_frame_id: int) -> np.ndarray:
        pose = np.loadtxt(
            self.root / "pose" / f"{source_frame_id}.txt",
            dtype=np.float64,
        )
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"source frame {source_frame_id} must have a finite pose")
        if not np.isfinite(np.linalg.det(pose)) or abs(np.linalg.det(pose)) <= 1e-12:
            raise ValueError(f"source frame {source_frame_id} must have an invertible pose")
        return np.asarray(pose, dtype=np.float64)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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

    def _validate_frame_inputs(self, source_frame_id: int) -> None:
        paths = {
            "color": self.root / "color" / f"{source_frame_id}.jpg",
            "depth": self.root / "depth" / f"{source_frame_id}.png",
            "pose": self.root / "pose" / f"{source_frame_id}.txt",
        }
        for role, path in paths.items():
            self._require_regular_file(path, role)
        if self.input_hashes is not None:
            expected = self.input_hashes.get(source_frame_id)
            if expected is None or set(expected) != set(paths):
                raise ValueError(f"source frame {source_frame_id} input hashes are incomplete")
            for role, path in paths.items():
                if self._sha256(path) != expected[role]:
                    raise ValueError(f"source frame {source_frame_id} {role} hash mismatch")
        with Image.open(paths["depth"]) as image:
            if image.size != (self.intrinsics.width, self.intrinsics.height):
                raise ValueError(f"source frame {source_frame_id} depth shape mismatch")
