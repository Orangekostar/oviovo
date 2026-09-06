#!/usr/bin/env python3
"""Materialize one manifest-bound 3RScan visit for OVI-MAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


class MaterializationError(RuntimeError):
    """Raised when a 3RScan visit cannot be materialized faithfully."""


@dataclass(frozen=True)
class SequenceInfo:
    version_number: int
    sensor_name: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    color_intrinsic: np.ndarray
    color_extrinsic: np.ndarray
    depth_intrinsic: np.ndarray
    depth_extrinsic: np.ndarray
    frame_count: int


_INFO_FIELDS = {
    "m_versionNumber",
    "m_sensorName",
    "m_colorWidth",
    "m_colorHeight",
    "m_depthWidth",
    "m_depthHeight",
    "m_depthShift",
    "m_calibrationColorIntrinsic",
    "m_calibrationColorExtrinsic",
    "m_calibrationDepthIntrinsic",
    "m_calibrationDepthExtrinsic",
    "m_frames.size",
}


def _parse_positive_int(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise MaterializationError(f"{field} must be an integer") from error
    if parsed <= 0:
        raise MaterializationError(f"{field} must be positive")
    return parsed


def _parse_matrix(value: str, *, field: str) -> np.ndarray:
    tokens = value.split()
    if len(tokens) != 16:
        raise MaterializationError(f"{field} must contain one 4x4 matrix")
    try:
        matrix = np.asarray([float(token) for token in tokens], dtype=np.float64).reshape(4, 4)
    except ValueError as error:
        raise MaterializationError(f"{field} must contain finite numeric values") from error
    if not np.isfinite(matrix).all():
        raise MaterializationError(f"{field} must contain finite numeric values")
    return matrix


def parse_sequence_info(text: str) -> SequenceInfo:
    """Parse the calibration and frame metadata embedded in sequence.zip."""

    if not isinstance(text, str):
        raise TypeError("sequence info must be text")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or key in fields:
            raise MaterializationError("sequence _info.txt contains malformed or duplicate fields")
        fields[key] = value.strip()
    missing = _INFO_FIELDS - fields.keys()
    if missing:
        raise MaterializationError(
            "sequence _info.txt is missing required fields: " + ", ".join(sorted(missing))
        )
    try:
        version_number = int(fields["m_versionNumber"])
        depth_shift = float(fields["m_depthShift"])
    except ValueError as error:
        raise MaterializationError("sequence scalar metadata is invalid") from error
    if version_number <= 0 or not np.isfinite(depth_shift) or depth_shift <= 0.0:
        raise MaterializationError("sequence scalar metadata must be positive and finite")
    sensor_name = fields["m_sensorName"]
    if not sensor_name:
        raise MaterializationError("m_sensorName must be non-empty")
    return SequenceInfo(
        version_number=version_number,
        sensor_name=sensor_name,
        color_width=_parse_positive_int(fields["m_colorWidth"], field="m_colorWidth"),
        color_height=_parse_positive_int(fields["m_colorHeight"], field="m_colorHeight"),
        depth_width=_parse_positive_int(fields["m_depthWidth"], field="m_depthWidth"),
        depth_height=_parse_positive_int(fields["m_depthHeight"], field="m_depthHeight"),
        depth_shift=depth_shift,
        color_intrinsic=_parse_matrix(
            fields["m_calibrationColorIntrinsic"],
            field="m_calibrationColorIntrinsic",
        ),
        color_extrinsic=_parse_matrix(
            fields["m_calibrationColorExtrinsic"],
            field="m_calibrationColorExtrinsic",
        ),
        depth_intrinsic=_parse_matrix(
            fields["m_calibrationDepthIntrinsic"],
            field="m_calibrationDepthIntrinsic",
        ),
        depth_extrinsic=_parse_matrix(
            fields["m_calibrationDepthExtrinsic"],
            field="m_calibrationDepthExtrinsic",
        ),
        frame_count=_parse_positive_int(fields["m_frames.size"], field="m_frames.size"),
    )


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MaterializationError(f"cannot read bound sequence archive: {path}") from error
    return digest.hexdigest(), byte_count


def _validate_sequence_binding(binding: Mapping[str, object]) -> tuple[Path, dict[str, object]]:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256", "byte_count"}:
        raise MaterializationError("sequence.zip binding must contain path, sha256, and byte_count")
    raw_path = binding.get("path")
    expected_hash = binding.get("sha256")
    expected_size = binding.get("byte_count")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise MaterializationError("sequence.zip binding path must be absolute")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise MaterializationError("sequence.zip binding sha256 is invalid")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise MaterializationError("sequence.zip binding byte_count is invalid")
    path = Path(raw_path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise MaterializationError(f"bound sequence archive is missing: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise MaterializationError("bound sequence archive must be a regular non-symlink file")
    observed_hash, observed_size = _sha256_and_size(path)
    observed = {
        "path": str(path),
        "sha256": observed_hash,
        "byte_count": observed_size,
    }
    if observed != dict(binding):
        raise MaterializationError("sequence.zip binding does not match source content")
    return path, observed


def _decode_image(data: bytes, *, label: str, flag: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
    if image is None:
        raise MaterializationError(f"{label} is not a readable image")
    return image


def _parse_pose(data: bytes, *, frame_id: int) -> np.ndarray:
    try:
        tokens = data.decode("ascii").split()
        values = [float(token) for token in tokens]
    except (UnicodeDecodeError, ValueError) as error:
        raise MaterializationError(f"frame {frame_id} pose must be a numeric 4x4 matrix") from error
    if len(values) != 16:
        raise MaterializationError(f"frame {frame_id} pose must be exactly 4x4")
    pose = np.asarray(values, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    if (
        not np.isfinite(pose).all()
        or not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=5e-3)
        or float(np.linalg.det(rotation)) <= 0.0
    ):
        raise MaterializationError(f"frame {frame_id} pose must be a finite rigid 4x4 matrix")
    return pose


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return (
        "".join(
            " ".join(f"{float(value):.17g}" for value in row) + "\n"
            for row in np.asarray(matrix, dtype=np.float64)
        )
    ).encode("ascii")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise MaterializationError(f"cannot write materialized file: {path}") from error


def _relative_file_record(path: Path, *, root: Path) -> dict[str, object]:
    digest, byte_count = _sha256_and_size(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "byte_count": byte_count,
    }


def _output_tree(root: Path, paths: list[Path]) -> dict[str, object]:
    records = sorted(
        (_relative_file_record(path, root=root) for path in paths),
        key=lambda item: str(item["path"]),
    )
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "byte_count": sum(int(record["byte_count"]) for record in records),
        "file_count": len(records),
        "files": records,
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _round_trip_error(depth: np.ndarray, intrinsic: np.ndarray, pose: np.ndarray) -> float:
    positive = np.argwhere(depth > 0)
    if not len(positive):
        raise MaterializationError("first selected depth frame has no positive sample")
    v, u = (int(value) for value in positive[0])
    z = float(depth[v, u])
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise MaterializationError("depth intrinsic focal lengths must be non-zero")
    camera = np.asarray(((u - cx) * z / fx, (v - cy) * z / fy, z, 1.0))
    try:
        recovered = np.linalg.inv(pose) @ (pose @ camera)
    except np.linalg.LinAlgError as error:
        raise MaterializationError("first selected pose is singular") from error
    error = float(np.linalg.norm(recovered[:3] - camera[:3]))
    if not np.isfinite(error) or error > 1e-6:
        raise MaterializationError("camera/world round-trip validation failed")
    return error


def materialize_sequence_archive(
    *,
    sequence_zip_binding: Mapping[str, object],
    scan_id: str,
    pair_id: str,
    visit_index: int,
    output_dir: Path,
    frame_step: int = 1,
) -> dict[str, Any]:
    """Validate and publish one sequence.zip as an OVI ScanNet-style scene."""

    if not isinstance(scan_id, str) or not scan_id or Path(scan_id).name != scan_id:
        raise MaterializationError("scan_id must be one safe path component")
    if not isinstance(pair_id, str) or not pair_id:
        raise MaterializationError("pair_id must be non-empty")
    if isinstance(visit_index, bool) or visit_index not in {0, 1}:
        raise MaterializationError("visit_index must be 0 or 1")
    if isinstance(frame_step, bool) or not isinstance(frame_step, int) or frame_step <= 0:
        raise MaterializationError("frame_step must be a positive integer")
    output = Path(output_dir)
    if os.path.lexists(output):
        raise MaterializationError(f"materialized output already exists: {output}")
    archive_path, source_record = _validate_sequence_binding(sequence_zip_binding)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    written: list[Path] = []
    first_depth: np.ndarray | None = None
    first_pose: np.ndarray | None = None
    try:
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as error:
            raise MaterializationError("sequence.zip is not a readable ZIP archive") from error
        with archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise MaterializationError("sequence.zip contains duplicate member names")
            try:
                info = parse_sequence_info(archive.read("_info.txt").decode("utf-8"))
            except KeyError as error:
                raise MaterializationError("sequence.zip is missing _info.txt") from error
            except UnicodeDecodeError as error:
                raise MaterializationError("sequence _info.txt must be UTF-8") from error
            if not np.allclose(info.color_extrinsic, np.eye(4), rtol=0.0, atol=1e-9) or not np.allclose(
                info.depth_extrinsic, np.eye(4), rtol=0.0, atol=1e-9
            ):
                raise MaterializationError(
                    "OVI ScannetLoader currently requires identity RGB-depth extrinsics"
                )
            required = {
                f"frame-{frame_id:06d}.{suffix}"
                for frame_id in range(info.frame_count)
                for suffix in ("color.jpg", "depth.pgm", "pose.txt")
            }
            if not required.issubset(names):
                raise MaterializationError(
                    "sequence.zip does not contain a complete contiguous frame set"
                )

            for name, matrix in (
                ("intrinsic/intrinsic_color.txt", info.color_intrinsic),
                ("intrinsic/intrinsic_depth.txt", info.depth_intrinsic),
            ):
                target = staging / name
                _write_bytes(target, _matrix_bytes(matrix))
                written.append(target)

            frame_map: list[dict[str, int]] = []
            for target_frame_id, source_frame_id in enumerate(
                range(0, info.frame_count, frame_step)
            ):
                prefix = f"frame-{source_frame_id:06d}"
                color_data = archive.read(f"{prefix}.color.jpg")
                color = _decode_image(
                    color_data,
                    label=f"frame {source_frame_id} color",
                    flag=cv2.IMREAD_COLOR,
                )
                if color.shape != (info.color_height, info.color_width, 3):
                    raise MaterializationError(
                        f"frame {source_frame_id} color dimensions do not match _info.txt"
                    )
                color_target = staging / "color" / f"{target_frame_id}.jpg"
                _write_bytes(color_target, color_data)
                written.append(color_target)

                depth_data = archive.read(f"{prefix}.depth.pgm")
                depth = _decode_image(
                    depth_data,
                    label=f"frame {source_frame_id} depth",
                    flag=cv2.IMREAD_UNCHANGED,
                )
                if depth.dtype != np.uint16 or depth.shape != (
                    info.depth_height,
                    info.depth_width,
                ):
                    raise MaterializationError(
                        f"frame {source_frame_id} depth must match 16-bit _info.txt dimensions"
                    )
                encoded, png = cv2.imencode(".png", depth)
                if not encoded:
                    raise MaterializationError(f"frame {source_frame_id} depth PNG encoding failed")
                depth_target = staging / "depth" / f"{target_frame_id}.png"
                _write_bytes(depth_target, png.tobytes())
                written.append(depth_target)

                pose = _parse_pose(archive.read(f"{prefix}.pose.txt"), frame_id=source_frame_id)
                pose_target = staging / "pose" / f"{target_frame_id}.txt"
                _write_bytes(pose_target, _matrix_bytes(pose))
                written.append(pose_target)
                frame_map.append(
                    {
                        "source_frame_id": source_frame_id,
                        "target_frame_id": target_frame_id,
                    }
                )
                if first_depth is None:
                    first_depth = depth
                    first_pose = pose

        observed_hash, observed_size = _sha256_and_size(archive_path)
        if (observed_hash, observed_size) != (
            source_record["sha256"],
            source_record["byte_count"],
        ):
            raise MaterializationError("sequence.zip changed during materialization")
        assert first_depth is not None and first_pose is not None
        round_trip_error = _round_trip_error(first_depth, info.depth_intrinsic, first_pose)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_id": "RSCAN_OVI_VISIT_V1",
            "status": "MATERIALIZED_INPUT_PASS",
            "dataset": "3RScan",
            "dataset_adapter": "scannet_nyu",
            "scan_id": scan_id,
            "pair_id": pair_id,
            "visit_index": visit_index,
            "frame_step": frame_step,
            "source_frame_count": info.frame_count,
            "frame_count": len(frame_map),
            "frame_map": frame_map,
            "depth_shift": info.depth_shift,
            "dimensions": {
                "color": {"width": info.color_width, "height": info.color_height},
                "depth": {"width": info.depth_width, "height": info.depth_height},
            },
            "calibration": {
                "color_intrinsic": info.color_intrinsic.tolist(),
                "color_extrinsic": info.color_extrinsic.tolist(),
                "depth_intrinsic": info.depth_intrinsic.tolist(),
                "depth_extrinsic": info.depth_extrinsic.tolist(),
            },
            "sequence_zip": source_record,
            "preflight": {
                "camera_world_round_trip_error": round_trip_error,
                "depth_encoding": "uint16",
                "depth_units_per_meter": info.depth_shift,
                "rgb_depth_registration": "K_depth @ inv(K_color)",
            },
            "output_tree": _output_tree(staging, written),
        }
        _write_bytes(staging / "materialized_manifest.json", _canonical_json(manifest))
        if os.path.lexists(output):
            raise MaterializationError(f"materialized output already exists: {output}")
        try:
            os.rename(staging, output)
        except OSError as error:
            raise MaterializationError(f"cannot publish materialized output: {output}") from error
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-zip", required=True, type=Path)
    parser.add_argument("--sequence-sha256", required=True)
    parser.add_argument("--sequence-byte-count", required=True, type=int)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--visit-index", required=True, type=int, choices=(0, 1))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = materialize_sequence_archive(
            sequence_zip_binding={
                "path": str(args.sequence_zip.resolve()),
                "sha256": args.sequence_sha256,
                "byte_count": args.sequence_byte_count,
            },
            scan_id=args.scan_id,
            pair_id=args.pair_id,
            visit_index=args.visit_index,
            output_dir=args.output_dir,
            frame_step=args.frame_step,
        )
    except MaterializationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "MaterializationError",
    "SequenceInfo",
    "main",
    "materialize_sequence_archive",
    "parse_sequence_info",
]


if __name__ == "__main__":
    raise SystemExit(main())
