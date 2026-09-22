"""Streaming ScanNet v4 sensor export for the pinned native ScannetLoader.

Binary format: ScanNet/ScanNet@3830fce7f8b2e48ef047ef7fd76ea5f62903f51c,
SensReader/python/SensorData.py. Original JPEG bytes are preserved without
resizing/recompression; the native loader performs its own BGR/depth alignment.
"""

from __future__ import annotations

import fcntl
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

from .assets import sha256_file
from .contracts import atomic_write_json, canonical_digest
from .scannet_download import SCANNET_REVISION, native_schedule, verified_download


def _read(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"truncated sensor stream at byte {handle.tell()}")
    return data


def _unpack(handle, fmt):
    return struct.unpack("<" + fmt, _read(handle, struct.calcsize("<" + fmt)))


def _matrix(handle):
    return np.frombuffer(_read(handle, 64), dtype="<f4").reshape(4, 4)


def sensor_header(handle) -> dict:
    version, = _unpack(handle, "I")
    if version != 4:
        raise ValueError(f"unsupported ScanNet sensor version {version}")
    length, = _unpack(handle, "Q")
    if length > 4096:
        raise ValueError("invalid sensor name length")
    name = _read(handle, length).decode("utf-8", errors="replace")
    matrices = {key: _matrix(handle) for key in (
        "intrinsic_color", "extrinsic_color", "intrinsic_depth", "extrinsic_depth")}
    color_type, depth_type, cw, ch, dw, dh, shift, count = _unpack(handle, "iiIIIIfQ")
    if (color_type, depth_type) != (2, 1):
        raise ValueError("only native JPEG + zlib uint16 sensor streams are supported")
    if shift != 1000.0:
        raise ValueError("sensor depth units incompatible with native ScannetLoader's fixed 1000 scale")
    if min(cw, ch, dw, dh) <= 0 or max(cw, ch, dw, dh) > 16384:
        raise ValueError("invalid sensor image dimensions")
    if not all(np.isfinite(m).all() for m in matrices.values()):
        raise ValueError("nonfinite calibration matrix")
    return {"sensor_name": name, "matrices": matrices, "color_width": cw, "color_height": ch,
            "depth_width": dw, "depth_height": dh, "depth_shift": shift, "source_frame_count": count}


def _completed(output: Path, identity: dict) -> dict | None:
    receipt_path = output / "export_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt["input_identity"] != identity:
        raise ValueError("export input changed; use a separate output directory")
    for relative, digest in receipt["file_hashes"].items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"export file changed: {path}")
    return receipt


def export_sensor(sensor: Path, output: Path, schedule: dict) -> dict:
    identity = {"sensor_path": str(sensor.resolve()), "sensor_sha256": sha256_file(sensor),
                "schedule": schedule, "export_schema": 1, "format_reference": SCANNET_REVISION}
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = _completed(output, identity)
        if receipt is not None:
            return receipt
        binding = output / "export_input.json"
        if (output.exists() and any(output.iterdir())
                and (not binding.exists() or json.loads(binding.read_text()) != identity)):
            raise ValueError(f"refusing to overwrite unrelated export: {output}")
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(binding, identity)
        return _export(sensor, output, schedule, identity)


def _export(sensor: Path, output: Path, schedule: dict, identity: dict) -> dict:
    file_hashes, invalid_poses = {}, []
    for folder in ("color", "depth", "pose", "intrinsic"):
        (output / folder).mkdir(exist_ok=True)
    with sensor.open("rb") as handle:
        header = sensor_header(handle)
        if schedule != native_schedule(header["source_frame_count"]):
            raise ValueError("sensor frame count disagrees with the locked native schedule")
        for key, matrix in header["matrices"].items():
            path = output / "intrinsic" / f"{key}.txt"
            np.savetxt(path, matrix, fmt="%f")
            file_hashes[str(path.relative_to(output))] = sha256_file(path)
        selected = set(schedule["frame_ids"])
        file_size = sensor.stat().st_size
        for index in range(header["source_frame_count"]):
            pose = _matrix(handle)
            _, _, color_size, depth_size = _unpack(handle, "QQQQ")
            if handle.tell() + color_size + depth_size > file_size:
                raise ValueError(f"truncated sensor frame {index}")
            if index not in selected:
                handle.seek(color_size + depth_size, 1)
                continue
            color_bytes, depth_bytes = _read(handle, color_size), _read(handle, depth_size)
            decoded = cv2.imdecode(np.frombuffer(color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[:2] != (header["color_height"], header["color_width"]):
                raise ValueError(f"invalid color payload in frame {index}")
            pixels = zlib.decompress(depth_bytes)
            if len(pixels) != header["depth_width"] * header["depth_height"] * 2:
                raise ValueError(f"invalid depth payload in frame {index}")
            depth = np.frombuffer(pixels, dtype="<u2").reshape(header["depth_height"], header["depth_width"])
            color_path = output / "color" / f"{index}.jpg"
            depth_path = output / "depth" / f"{index}.png"
            pose_path = output / "pose" / f"{index}.txt"
            color_path.write_bytes(color_bytes)
            if not cv2.imwrite(str(depth_path), depth):
                raise OSError(f"failed to write depth PNG: {depth_path}")
            np.savetxt(pose_path, pose, fmt="%f")
            if not np.isfinite(pose).all():
                invalid_poses.append(index)
            for path in (color_path, depth_path, pose_path):
                file_hashes[str(path.relative_to(output))] = sha256_file(path)
    if sha256_file(sensor) != identity["sensor_sha256"]:
        raise ValueError("sensor input changed during export")
    receipt = {"schema_version": 1, "status": "EXPORTED_NATIVE_INPUTS", "input_identity": identity,
               "frame_count": len(selected), "schedule": schedule,
               "depth_scale": header["depth_shift"], "invalid_pose_frame_ids": invalid_poses,
               "file_hashes": file_hashes,
               "color_convention": "original JPEG payload; native ScannetLoader decodes BGR and warps to depth intrinsics",
               "registration": "native loader K_depth @ inverse(K_color); no extra registration or axis alignment"}
    atomic_write_json(output / "export_receipt.json", receipt)
    return receipt


def export_development(root: Path) -> dict:
    locked = json.loads((root / "acquisition_lock.json").read_text())
    raw_receipt = json.loads((root / "download_receipt.json").read_text())
    if raw_receipt["status"] != "RAW_DOWNLOADS_COMPLETE" or raw_receipt["lock_sha256"] != sha256_file(root / "acquisition_lock.json"):
        raise ValueError("raw downloads are not complete for this acquisition lock")
    receipts = {}
    for row in locked["selected"]:
        if row["role"] == "confirm":
            continue
        scene = row["scene_id"]
        sensor = root / "scans" / scene / f"{scene}.sens"
        if not verified_download(sensor, row["files"][".sens"]):
            raise ValueError(f"sensor download no longer matches acquisition identity: {sensor}")
        receipt = export_sensor(sensor, root / "exported" / scene, row["schedule"])
        receipts[scene] = {"path": str(root / "exported" / scene / "export_receipt.json"),
                           "digest": canonical_digest(receipt)}
    result = {"status": "DEVELOPMENT_INPUTS_EXPORTED", "scenes": receipts,
              "confirmation_processing": "DEFERRED_UNTIL_FROZEN_SELECTION"}
    atomic_write_json(root / "export_receipt.json", result)
    return result
