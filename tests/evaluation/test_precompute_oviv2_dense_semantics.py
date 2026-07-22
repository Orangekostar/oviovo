from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import stat
import sys

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

import scripts.precompute_oviv2_dense_semantics as precompute_module
from scripts.precompute_oviv2_dense_semantics import (
    _atomic_json,
    _load_rgb_frame,
    _load_json,
    _materialized_worker_config,
    _preflight,
    parse_args,
    run,
)
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.oviv2.dense_semantics import (
    DenseSemanticProvenance,
    load_dense_frame,
    sha256_file,
)


CLASS_COUNT = 41
CLASSES = [f"class-{index:02d}" for index in range(1, CLASS_COUNT + 1)]
MANIFEST_KEYS = {
    "schema_version",
    "method",
    "scene",
    "frame_count",
    "source_frame_ids",
    "image_shape",
    "sample_stride",
    "top_k",
    "class_count",
    "vocabulary_sha256",
    "provenance",
    "cache_files_sha256",
}


FAKE_WORKER = r'''#!/usr/bin/env python3
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def log(event):
    path = os.environ.get("FAKE_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")


def block(array):
    values = np.ascontiguousarray(array)
    return {
        "encoding": "base64",
        "dtype": values.dtype.name,
        "shape": list(values.shape),
        "data": base64.b64encode(values.tobytes(order="C")).decode("ascii"),
    }


def argument(name):
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    return sys.argv[index + 1]


classes_path = Path(
    argument("--classes-json") or os.environ.get("FAKE_CLASSES_JSON")
)
classes_raw = classes_path.read_bytes()
classes_payload = json.loads(classes_raw)
classes = classes_payload["classes"]
vocabulary_sha256 = hashlib.sha256(classes_raw).hexdigest()
mode = os.environ.get("FAKE_MODE", "ok")
sample_stride = 1 if mode == "bool_infer_stride" else (4 if mode == "tesse" else 2)
top_k = 4 if mode == "tesse" else 2
infer_count = 0
provenance = {
    "backend": "fake-radseg",
    "source_commit": "a" * 40,
    "radio_commit": "b" * 40,
    "model_id": "fake:model",
    "model_sha256": "c" * 64,
    "auxiliary_model_sha256": "d" * 64,
    "language_model_id": "fake/language-model",
    "language_model_revision": "e" * 40,
    "language_model_sha256": "f" * 64,
    "vocabulary_sha256": vocabulary_sha256,
    "prompt_sha256": "1" * 64,
    "inference_config_sha256": "2" * 64,
}
log({
    "event": "start",
    "pid": os.getpid(),
    "argv": sys.argv[1:],
    "classes_keys": sorted(classes_payload),
    "classes_sha256": vocabulary_sha256,
})

for line in sys.stdin:
    request = json.loads(line)
    request_id = request.get("id")
    operation = request.get("operation")
    log({"event": operation, "id": request_id})
    if operation == "metadata":
        if mode == "metadata_error":
            response = {"id": request_id, "ok": False, "error": "metadata rejected"}
        else:
            metadata_classes = list(classes)
            metadata_provenance = dict(provenance)
            if mode == "bad_metadata_classes":
                metadata_classes[0] = "wrong-class"
            if mode == "bad_metadata_vocab":
                metadata_provenance["vocabulary_sha256"] = "0" * 64
            if mode == "missing_language_provenance":
                metadata_provenance.pop("language_model_sha256")
            response = {
                "id": request_id,
                "ok": True,
                "class_count": len(metadata_classes),
                "classes": metadata_classes,
                "sample_stride": sample_stride,
                "top_k": top_k,
                "provenance": metadata_provenance,
            }
    elif operation == "infer":
        infer_count += 1
        rgb = request.get("rgb")
        expected_keys = {"encoding", "dtype", "shape", "data"}
        if not isinstance(rgb, dict) or set(rgb) != expected_keys:
            response = {"id": request_id, "ok": False, "error": "bad RGB keys"}
        elif rgb["encoding"] != "base64" or rgb["dtype"] != "uint8":
            response = {"id": request_id, "ok": False, "error": "bad RGB contract"}
        else:
            raw = base64.b64decode(rgb["data"], validate=True)
            shape = tuple(rgb["shape"])
            if shape[-1] != 3 or len(raw) != math.prod(shape):
                response = {"id": request_id, "ok": False, "error": "bad RGB bytes"}
            elif mode == "second_error" and infer_count == 2:
                response = {"id": request_id, "ok": False, "error": "second frame failed"}
            else:
                height, width = shape[:2]
                sampled = (
                    math.ceil(height / sample_stride),
                    math.ceil(width / sample_stride),
                )
                class_ids = np.empty((*sampled, top_k), dtype=np.int64)
                probabilities = np.empty((*sampled, top_k), dtype=np.float32)
                for top_index in range(top_k):
                    class_ids[..., top_index] = top_index + 1
                    probabilities[..., top_index] = (
                        [0.4, 0.3, 0.2, 0.1][top_index]
                        if top_k == 4
                        else [0.75, 0.20][top_index]
                    )
                entropy = np.full(sampled, 0.5, dtype=np.float32)
                margin = np.full(
                    sampled,
                    0.1 if top_k == 4 else 0.55,
                    dtype=np.float32,
                )
                blocks = {
                    "class_ids": block(class_ids),
                    "probabilities": block(probabilities),
                    "entropy": block(entropy),
                    "margin": block(margin),
                }
                if mode == "bad_shape":
                    blocks["entropy"]["shape"] = [sampled[0], sampled[1] + 1]
                elif mode == "bad_dtype":
                    blocks["probabilities"]["dtype"] = "float64"
                elif mode == "bad_base64":
                    blocks["class_ids"]["data"] = "!!!"
                response = {
                    "id": request_id,
                    "ok": True,
                    "image_shape": (
                        [float(height), width]
                        if mode == "float_infer_image_shape"
                        else [height, width]
                    ),
                    "class_count": len(classes),
                    "sample_stride": (
                        True if mode == "bool_infer_stride" else sample_stride
                    ),
                    **blocks,
                }
    else:
        response = {"id": request_id, "ok": False, "error": "unknown operation"}
    print(json.dumps(response, sort_keys=True, allow_nan=False), flush=True)

log({"event": "closed"})
'''


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_fixture(tmp_path: Path, *, mode: str = "ok") -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    results = dataset_root / "results"
    results.mkdir(parents=True)
    poses: list[str] = []
    for cache_index in range(2):
        rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3) + cache_index
        Image.fromarray(rgb).save(results / f"frame{cache_index:06d}.jpg")
        Image.fromarray(np.full((4, 5), 6554, dtype=np.uint16)).save(
            results / f"depth{cache_index:06d}.png"
        )
        poses.append(" ".join(str(value) for value in np.eye(4).reshape(-1)))
    (dataset_root / "traj.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")

    classes_json = tmp_path / "classes.json"
    classes_json.write_text(
        json.dumps({"classes": CLASSES, "aliases": {}}, separators=(",", ":")),
        encoding="utf-8",
    )
    benchmark = tmp_path / "manifest.json"
    _write_json(
        benchmark,
        {
            "schema_version": 1,
            "manifest_id": "fixture",
            "dataset": "Replica",
            "frame_selection": {
                "start": 10,
                "stop_exclusive": 16,
                "stride": 3,
                "sampled_frames_per_scene": 2,
            },
            "vocabulary": {
                "classes": CLASSES,
                "source_path": str(classes_json),
            },
            "aliases": {},
            "scenes": [{"scene": "room0"}],
        },
    )
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    log_path = tmp_path / "worker.jsonl"
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "scene": "room0",
            "dataset_root": str(dataset_root),
            "manifest": str(benchmark),
            "num_frames": 2,
            "source_start": 10,
            "source_stride": 3,
            "dense_semantics": {
                "classes_json": str(classes_json),
                "worker_command": [sys.executable, str(worker)],
                "worker_cwd": str(tmp_path),
                "request_timeout_sec": 5.0,
                "worker_env": {
                    "FAKE_CLASSES_JSON": str(classes_json),
                    "FAKE_LOG": str(log_path),
                    "FAKE_MODE": mode,
                },
            },
        },
    )
    return config, classes_json, log_path


def _write_scannet_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "scannet" / "scene0011_00"
    for name in ("color", "depth", "pose", "intrinsic"):
        (dataset_root / name).mkdir(parents=True)
    intrinsic = np.eye(4)
    intrinsic[0, 0] = 4.0
    intrinsic[1, 1] = 4.0
    intrinsic[0, 2] = 2.0
    intrinsic[1, 2] = 1.5
    np.savetxt(dataset_root / "intrinsic" / "intrinsic_depth.txt", intrinsic)
    frame_inputs: dict[str, dict[str, str]] = {}
    for cache_index, source_id in enumerate((0, 20)):
        color = dataset_root / "color" / f"{source_id}.jpg"
        depth = dataset_root / "depth" / f"{source_id}.png"
        pose = dataset_root / "pose" / f"{source_id}.txt"
        Image.fromarray(
            np.full((4, 5, 3), 30 + cache_index, dtype=np.uint8)
        ).save(color)
        Image.fromarray(np.full((4, 5), 1000, dtype=np.uint16)).save(depth)
        np.savetxt(pose, np.eye(4))
        frame_inputs[str(source_id)] = {
            "color_sha256": hashlib.sha256(color.read_bytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(depth.read_bytes()).hexdigest(),
            "pose_sha256": hashlib.sha256(pose.read_bytes()).hexdigest(),
        }
    classes = [f"scannet-class-{index:03d}" for index in range(200)]
    classes_json = tmp_path / "scannet_classes.json"
    classes_json.write_text(
        json.dumps({"classes": classes, "aliases": {}}, separators=(",", ":")),
        encoding="utf-8",
    )
    benchmark = tmp_path / "scannet_manifest.json"
    _write_json(
        benchmark,
        {
            "schema_version": 1,
            "manifest_id": "oviv2_scannet200_fixture",
            "dataset": "ScanNet200",
            "frame_selection": {"start": 0, "stride": 10},
            "vocabulary": {"classes": classes, "source_path": str(classes_json)},
            "scenes": [
                {
                    "scene": "scene0011_00",
                    "frame_count": 2,
                    "source_frame_ids": [0, 20],
                    "frame_inputs": frame_inputs,
                    "image_shape": [4, 5],
                    "depth_scale": 1000.0,
                }
            ],
        },
    )
    worker = tmp_path / "fake_scannet_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    log_path = tmp_path / "scannet_worker.jsonl"
    config = tmp_path / "scannet_config.json"
    _write_json(
        config,
        {
            "scene": "scene0011_00",
            "dataset_root": str(dataset_root),
            "manifest": str(benchmark),
            "num_frames": 2,
            "dense_semantics": {
                "classes_json": str(classes_json),
                "worker_command": [sys.executable, str(worker)],
                "worker_cwd": str(tmp_path),
                "request_timeout_sec": 5.0,
                "worker_env": {
                    "FAKE_CLASSES_JSON": str(classes_json),
                    "FAKE_LOG": str(log_path),
                },
            },
        },
    )
    return config, classes_json, log_path


TESSE_CLASSES = [
    "Fridge",
    "Books",
    "Chair",
    "Vase",
    "Couch",
    "Drawer",
    "Objects",
    "Table",
    "Bin",
    "Humans",
]
TESSE_OBJECT_IDS = [1, 2, 5, 6, 7, 9, 10, 16, 18, 20]
TESSE_CAMERA = {
    "cx": 360.0,
    "cy": 240.0,
    "fx": 415.69219381653056,
    "fy": 415.69219381653056,
    "h": 480,
    "scale": 1000.0,
    "w": 720,
}


def _tesse_export_digest(dataset_root: Path, scene_root: Path) -> tuple[str, int]:
    paths = [
        *scene_root.joinpath("results").glob("frame*.jpg"),
        *scene_root.joinpath("results").glob("depth*.png"),
        scene_root / "traj.txt",
        scene_root / "timestamps.csv",
        dataset_root / "cam_params.json",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(dataset_root))):
        relative = str(path.relative_to(dataset_root))
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")
    return digest.hexdigest(), len(paths)


def _write_tesse_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    monkeypatch.setitem(TesseCdRgbdDataset.EXPECTED_FRAMES, "apartment", 2)
    dataset_root = tmp_path / "rgbd_v1"
    scene_root = dataset_root / "apartment"
    results = scene_root / "results"
    results.mkdir(parents=True)
    camera_path = dataset_root / "cam_params.json"
    _write_json(camera_path, {"camera": TESSE_CAMERA})
    timestamps = (4_204_107_999, 4_254_107_999)
    for index in range(2):
        Image.fromarray(
            np.full((480, 720, 3), 20 + index, dtype=np.uint8),
            mode="RGB",
        ).save(results / f"frame{index:06d}.jpg", quality=95)
        Image.fromarray(np.full((480, 720), 1000 + index, dtype=np.uint16)).save(
            results / f"depth{index:06d}.png"
        )
    with (scene_root / "timestamps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame_index", "sensor_timestamp_ns", "relative_timestamp_ns"))
        writer.writerow((0, timestamps[0], 0))
        writer.writerow((1, timestamps[1], timestamps[1] - timestamps[0]))
    pose = " ".join(str(value) for value in np.eye(4).reshape(-1))
    (scene_root / "traj.txt").write_text(f"{pose}\n{pose}\n", encoding="utf-8")

    source_manifest = tmp_path / "tesse_cd.json"
    database_hash = "a" * 64
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "camera": {
                "width": 720,
                "height": 480,
                "fx": TESSE_CAMERA["fx"],
                "fy": TESSE_CAMERA["fy"],
                "cx": TESSE_CAMERA["cx"],
                "cy": TESSE_CAMERA["cy"],
            },
            "sequences": {
                "apartment": {
                    "bag": {"database": {"sha256": database_hash}},
                    "timeline": {"depth_frame_count": 2},
                }
            },
        },
    )
    schedule = tmp_path / "schedule.json"
    _write_json(
        schedule,
        {
            "schema_version": 2,
            "dataset": "TESSE-CD",
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "source_manifest": {
                "path": str(source_manifest),
                "sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            },
            "scenes": {
                "apartment": {
                    "frame_count": 2,
                    "first_depth_timestamp_ns": timestamps[0],
                    "last_depth_timestamp_ns": timestamps[-1],
                    "sources": {"database": {"sha256": database_hash}},
                }
            },
        },
    )
    combined_hash, file_count = _tesse_export_digest(dataset_root, scene_root)
    export = scene_root / "export_manifest.json"
    _write_json(
        export,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "frame_count": 2,
            "source_manifest": str(source_manifest),
            "source_database_sha256": database_hash,
            "combined_output_sha256": combined_hash,
            "file_hash_count": file_count,
        },
    )
    classes_json = tmp_path / "tesse_apartment.json"
    _write_json(
        classes_json,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "classes": TESSE_CLASSES,
            "object_semantic_ids": TESSE_OBJECT_IDS,
            "unknown_semantic_id": 0,
        },
    )
    worker = tmp_path / "fake_tesse_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    log_path = tmp_path / "tesse_worker.jsonl"
    benchmark = tmp_path / "oviv2_tesse_cd_cache.json"
    _write_json(
        benchmark,
        {
            "schema_version": 1,
            "manifest_id": "oviv2_tesse_cd_cache_v1",
            "dataset": "TESSE-CD",
            "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
            "camera": {
                "path": str(camera_path),
                "sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
                "width": 720,
                "height": 480,
                "fx": TESSE_CAMERA["fx"],
                "fy": TESSE_CAMERA["fy"],
                "cx": TESSE_CAMERA["cx"],
                "cy": TESSE_CAMERA["cy"],
                "depth_scale": 1000.0,
            },
            "source_manifest": {
                "path": str(source_manifest),
                "sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            },
            "schedule_manifest": {
                "path": str(schedule),
                "sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
            },
            "scenes": {
                "apartment": {
                    "root": str(scene_root),
                    "frame_count": 2,
                    "image_shape": [480, 720],
                    "depth_unit": "millimeter",
                    "source_depth_dtype": "uint16",
                    "pose_convention": "camera_to_world",
                    "source_frame_ids": {"start": 0, "stop_exclusive": 2, "stride": 1},
                    "export_manifest": {
                        "path": str(export),
                        "sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
                        "combined_output_sha256": combined_hash,
                        "file_hash_count": file_count,
                    },
                    "vocabulary": {
                        "json_path": str(classes_json),
                        "json_sha256": hashlib.sha256(classes_json.read_bytes()).hexdigest(),
                    },
                }
            },
        },
    )
    config = tmp_path / "tesse_dense.json"
    _write_json(
        config,
        {
            "scene": "apartment",
            "dataset_root": str(scene_root),
            "manifest": str(benchmark),
            "num_frames": 2,
            "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
            "dense_semantics": {
                "classes_json": str(classes_json),
                "worker_command": [sys.executable, str(worker)],
                "worker_cwd": str(tmp_path),
                "request_timeout_sec": 5.0,
                "worker_env": {
                    "FAKE_CLASSES_JSON": str(classes_json),
                    "FAKE_LOG": str(log_path),
                    "FAKE_MODE": "tesse",
                },
            },
        },
    )
    return config, classes_json, benchmark


def _args(config: Path, output: Path, *, num_frames: int = 2, resume: bool = False):
    values = [
        "--config",
        str(config),
        "--output",
        str(output),
        "--num-frames",
        str(num_frames),
    ]
    if resume:
        values.append("--resume")
    return parse_args(values)


def _explicit_args(
    config: Path,
    output: Path,
    worker: Path,
    tmp_path: Path,
    *,
    classes_json: Path | None = None,
):
    values = [
        "--config",
        str(config),
        "--output",
        str(output),
        "--num-frames",
        "1",
        "--worker-python",
        sys.executable,
        "--worker-script",
        str(worker),
        "--backend",
        "radseg",
        "--source-root",
        str(tmp_path / "radseg-source"),
        "--radio-root",
        str(tmp_path / "radio-source"),
        "--model-version",
        "c-radio_v3-b",
        "--lang-model",
        "siglip2",
        "--language-model-root",
        str(tmp_path / "language-model"),
        "--language-model-id",
        "google/siglip2-so400m-patch14-384",
        "--language-model-revision",
        "a" * 40,
        "--language-model-sha256",
        "b" * 64,
        "--device",
        "cpu",
        "--sample-stride",
        "2",
        "--top-k",
        "2",
        "--amp",
        "--sam-refinement",
        "--sam-checkpoint",
        str(tmp_path / "sam.pth"),
    ]
    if classes_json is not None:
        values.extend(["--classes-json", str(classes_json)])
    return parse_args(values)


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_precompute_writes_two_frame_manifest_hashes_and_causal_ids(tmp_path: Path) -> None:
    config, classes_json, log_path = _write_fixture(tmp_path)
    output = tmp_path / "dense"

    manifest = run(_args(config, output))

    assert set(manifest) == MANIFEST_KEYS
    assert manifest["schema_version"] == 1
    assert manifest["method"] == "OVIV2-dense-semantic-cache"
    assert manifest["scene"] == "room0"
    assert manifest["frame_count"] == 2
    assert manifest["source_frame_ids"] == [10, 13]
    assert manifest["image_shape"] == [4, 5]
    assert manifest["sample_stride"] == 2
    assert manifest["top_k"] == 2
    assert manifest["class_count"] == CLASS_COUNT
    assert manifest["vocabulary_sha256"] == hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()
    assert list(manifest["cache_files_sha256"]) == [
        "frame000000.npz",
        "frame000001.npz",
    ]
    for cache_index, source_id in enumerate((10, 13)):
        name = f"frame{cache_index:06d}.npz"
        path = output / name
        digest = manifest["cache_files_sha256"][name]
        assert sha256_file(path) == digest
        dense = load_dense_frame(path, expected_sha256=digest)
        assert dense.cache_frame_id == cache_index
        assert dense.source_frame_id == source_id
        assert dense.image_shape == (4, 5)
    expected_prefix = hashlib.sha256()
    for cache_index, name in enumerate(
        ("frame000000.npz", "frame000001.npz")
    ):
        expected_prefix.update(cache_index.to_bytes(8, "little", signed=False))
        expected_prefix.update(bytes.fromhex(manifest["cache_files_sha256"][name]))
    assert manifest["provenance"]["cache_prefix_sha256"] == expected_prefix.hexdigest()
    assert DenseSemanticProvenance(**manifest["provenance"])
    assert manifest["provenance"]["language_model_id"] == "fake/language-model"
    assert manifest["provenance"]["language_model_revision"] == "e" * 40
    assert manifest["provenance"]["language_model_sha256"] == "f" * 64
    assert json.loads((output / "dense_manifest.json").read_text()) == manifest
    assert [event["event"] for event in _events(log_path)] == [
        "start",
        "metadata",
        "infer",
        "infer",
        "closed",
    ]


def test_precompute_supports_scannet200_explicit_source_frames(tmp_path: Path) -> None:
    config, classes_json, _ = _write_scannet_fixture(tmp_path)

    manifest = run(_args(config, tmp_path / "dense_scannet"))

    assert manifest["scene"] == "scene0011_00"
    assert manifest["frame_count"] == 2
    assert manifest["source_frame_ids"] == [0, 20]
    assert manifest["image_shape"] == [4, 5]
    assert manifest["class_count"] == 200
    assert manifest["vocabulary_sha256"] == hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()


def test_tesse_preflight_uses_checked_scene_contract_without_materializing_rgb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, classes_json, _ = _write_tesse_fixture(tmp_path, monkeypatch)

    preflight = _preflight(config, 2, _args(config, tmp_path / "dense"))

    assert isinstance(preflight.dataset, TesseCdRgbdDataset)
    assert not hasattr(preflight, "rgb_frames")
    assert preflight.dataset_name == "TESSE-CD"
    assert preflight.scene == "apartment"
    assert preflight.source_frame_ids == (0, 1)
    assert preflight.image_shape == (480, 720)
    assert preflight.classes == tuple(TESSE_CLASSES)
    assert preflight.object_semantic_ids == tuple(TESSE_OBJECT_IDS)
    assert preflight.vocabulary_sha256 == hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()
    assert preflight.worker_vocabulary_sha256 != preflight.vocabulary_sha256


def test_tesse_load_rgb_frame_streams_only_requested_rgb_without_depth_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_tesse_fixture(tmp_path, monkeypatch)
    preflight = _preflight(config, 2, _args(config, tmp_path / "dense"))
    converted: list[str | None] = []
    original_convert = Image.Image.convert

    def tracked_convert(image, *args, **kwargs):
        converted.append(getattr(image, "filename", None))
        return original_convert(image, *args, **kwargs)

    def reject_depth_load(self, *args, **kwargs):
        raise AssertionError("streaming RGB must not decode depth")

    monkeypatch.setattr(Image.Image, "convert", tracked_convert)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", reject_depth_load)

    rgb = _load_rgb_frame(preflight.dataset, 1, preflight.rgb_bindings[1])

    assert rgb.shape == (480, 720, 3)
    assert rgb.dtype == np.uint8
    assert len(converted) == 1
    assert int(rgb[0, 0, 0]) == 21


def test_tesse_precompute_preserves_frozen_radseg_output_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_tesse_fixture(tmp_path, monkeypatch)
    output = tmp_path / "dense"

    manifest = run(_args(config, output, num_frames=1))
    frame = load_dense_frame(
        output / "frame000000.npz",
        expected_sha256=manifest["cache_files_sha256"]["frame000000.npz"],
    )

    assert manifest["class_count"] == 10
    assert manifest["sample_stride"] == 4
    assert manifest["top_k"] == 4
    assert frame.class_ids.shape == (120, 180, 4)
    assert frame.probabilities.shape == (120, 180, 4)
    assert frame.entropy.shape == (120, 180)
    assert frame.margin.shape == (120, 180)


def test_tesse_worker_receives_canonical_projection_and_manifest_binds_both_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, classes_json, _ = _write_tesse_fixture(tmp_path, monkeypatch)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    log_path = Path(config_payload["dense_semantics"]["worker_env"]["FAKE_LOG"])

    manifest = run(_args(config, tmp_path / "dense", num_frames=1))

    projected = json.dumps(
        {"aliases": {}, "classes": TESSE_CLASSES},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    projection_hash = hashlib.sha256(projected).hexdigest()
    source_hash = hashlib.sha256(classes_json.read_bytes()).hexdigest()
    start = _events(log_path)[0]
    assert start["classes_keys"] == ["aliases", "classes"]
    assert start["classes_sha256"] == projection_hash
    assert manifest["vocabulary_sha256"] == source_hash
    assert manifest["provenance"]["vocabulary_sha256"] == projection_hash
    classes_index = start["argv"].index("--classes-json") + 1
    assert not Path(start["argv"][classes_index]).exists()


@pytest.mark.parametrize(("scene", "class_count"), [("apartment", 10), ("office", 7)])
def test_formal_tesse_command_projection_passes_real_worker_parser_preflight(
    tmp_path: Path,
    scene: str,
    class_count: int,
) -> None:
    required = [
        Path("/home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1"),
        Path("/home/ww/oviovo_references/modules/RADSeg"),
        Path("/home/ww/oviovo_references/modules/RADIO"),
        Path("/home/ww/oviovo_benchmark_assets/weights/segment_anything/sam_vit_h_4b8939.pth"),
    ]
    if not all(path.exists() for path in required):
        pytest.skip("formal TESSE/RADSeg assets are unavailable")
    repo_root = Path(__file__).resolve().parents[2]
    config = repo_root / f"configs/oviv2_tesse_{scene}_dense_stage3.json"
    worker_python = "/home/ww/miniconda3/envs/oviovo-radseg/bin/python"
    args = parse_args(
        [
            "--config",
            str(config),
            "--output",
            str(tmp_path / scene),
            "--num-frames",
            "1",
            "--classes-json",
            str(repo_root / f"configs/evaluation/vocabularies/tesse_cd_{scene}.json"),
            "--worker-python",
            worker_python,
            "--worker-script",
            "scripts/radseg_dense_worker.py",
            "--backend",
            "radseg",
            "--source-root",
            "/home/ww/oviovo_references/modules/RADSeg",
            "--radio-root",
            "/home/ww/oviovo_references/modules/RADIO",
            "--model-version",
            "/home/ww/oviovo_benchmark_assets/weights/radio/c-radio_v3-b_half-44653a.pth.tar",
            "--lang-model",
            "siglip2",
            "--language-model-root",
            "/home/ww/oviovo_benchmark_assets/weights/siglip2-so400m-patch16-naflex-cc24074",
            "--language-model-id",
            "google/siglip2-so400m-patch16-naflex",
            "--language-model-revision",
            "cc24074f717b612951c2dead130904ab9b65a81e",
            "--language-model-sha256",
            "0e5dbd4cd9511c4335ae4a144ac5187841a2f7fed2a07df9c36668516e02cafe",
            "--device",
            "cuda",
            "--sample-stride",
            "4",
            "--top-k",
            "4",
            "--amp",
            "--sam-refinement",
            "--sam-checkpoint",
            "/home/ww/oviovo_benchmark_assets/weights/segment_anything/sam_vit_h_4b8939.pth",
        ]
    )
    preflight = _preflight(config, 1, args)
    parser_script = """
import json
import sys
from scripts.radseg_dense_worker import load_frozen_classes, parse_args, validate_cli_args
args = parse_args(sys.argv[1:])
classes, digest = load_frozen_classes(args.classes_json)
validate_cli_args(args, classes)
print(json.dumps({"class_count": len(classes), "sha256": digest}, sort_keys=True))
"""

    with _materialized_worker_config(preflight) as worker:
        result = subprocess.run(
            [worker_python, "-c", parser_script, *worker.command[2:]],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed == {
        "class_count": class_count,
        "sha256": preflight.worker_vocabulary_sha256,
    }


@pytest.mark.parametrize(
    ("binding_path", "field", "bad_value", "message"),
    [
        (("camera",), "fx", 400.0, "camera"),
        (("schedule_manifest",), "sha256", "0" * 64, "schedule"),
        (("scenes", "apartment", "export_manifest"), "sha256", "0" * 64, "export"),
        (
            ("scenes", "apartment", "source_frame_ids"),
            "stride",
            2,
            "source frame",
        ),
    ],
)
def test_tesse_preflight_rejects_drift_from_checked_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_path: tuple[str, ...],
    field: str,
    bad_value: object,
    message: str,
) -> None:
    config, _, benchmark_path = _write_tesse_fixture(tmp_path, monkeypatch)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    binding = benchmark
    for key in binding_path:
        binding = binding[key]
    binding[field] = bad_value
    _write_json(benchmark_path, benchmark)

    with pytest.raises(ValueError, match=message):
        _preflight(config, 1, _args(config, tmp_path / "dense", num_frames=1))


def test_checked_tesse_dense_configs_are_stage3_only() -> None:
    expected = {"apartment": 1745, "office": 4346}
    for scene, frame_count in expected.items():
        path = (
            Path(__file__).resolve().parents[2]
            / f"configs/oviv2_tesse_{scene}_dense_stage3.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {
            "scene": scene,
            "dataset_root": (
                f"/home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1/{scene}"
            ),
            "manifest": "configs/evaluation/manifests/oviv2_tesse_cd_cache.json",
            "num_frames": frame_count,
            "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
        }
        serialized = json.dumps(payload, sort_keys=True).lower()
        assert all(token not in serialized for token in ("stage4", "route3", "scannet200"))
        assert all(
            token not in serialized
            for token in ("ground_truth", "prediction", "target")
        )


def test_cli_entrypoint_uses_configured_real_worker(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/precompute_oviv2_dense_semantics.py",
            "--config",
            str(config),
            "--output",
            str(output),
            "--num-frames",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["frame_count"] == 1


def test_stage1_config_supports_complete_explicit_worker_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, classes_json, log_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    worker = Path(config["dense_semantics"]["worker_command"][1])
    config.pop("dense_semantics")
    _write_json(config_path, config)
    monkeypatch.setenv("FAKE_LOG", str(log_path))

    manifest = run(
        _explicit_args(config_path, tmp_path / "dense", worker, tmp_path)
    )

    assert manifest["vocabulary_sha256"] == hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()
    argv = _events(log_path)[0]["argv"]
    assert argv == [
        "--backend",
        "radseg",
        "--source-root",
        str(tmp_path / "radseg-source"),
        "--radio-root",
        str(tmp_path / "radio-source"),
        "--model-version",
        "c-radio_v3-b",
        "--lang-model",
        "siglip2",
        "--language-model-root",
        str(tmp_path / "language-model"),
        "--language-model-id",
        "google/siglip2-so400m-patch14-384",
        "--language-model-revision",
        "a" * 40,
        "--language-model-sha256",
        "b" * 64,
        "--classes-json",
        str(classes_json),
        "--device",
        "cpu",
        "--sample-stride",
        "2",
        "--top-k",
        "2",
        "--amp",
        "--sam-refinement",
        "--sam-checkpoint",
        str(tmp_path / "sam.pth"),
    ]


def test_explicit_classes_json_overrides_manifest_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, log_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    worker = Path(config["dense_semantics"]["worker_command"][1])
    config.pop("dense_semantics")
    benchmark_path = Path(config["manifest"])
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["vocabulary"]["source_path"] = str(tmp_path / "missing.json")
    _write_json(benchmark_path, benchmark)
    override = tmp_path / "override.json"
    override.write_text(
        json.dumps({"aliases": {}, "classes": CLASSES}, indent=2),
        encoding="utf-8",
    )
    _write_json(config_path, config)
    monkeypatch.setenv("FAKE_LOG", str(log_path))

    manifest = run(
        _explicit_args(
            config_path,
            tmp_path / "dense",
            worker,
            tmp_path,
            classes_json=override,
        )
    )

    assert manifest["vocabulary_sha256"] == hashlib.sha256(
        override.read_bytes()
    ).hexdigest()


def test_partial_explicit_worker_cli_is_rejected_as_a_group(tmp_path: Path) -> None:
    config_path, _, _ = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config.pop("dense_semantics")
    _write_json(config_path, config)
    args = parse_args(
        [
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "dense"),
            "--num-frames",
            "1",
            "--backend",
            "radseg",
        ]
    )

    with pytest.raises(ValueError, match="explicit worker arguments.*source-root"):
        run(args)


def test_fresh_preflight_decodes_only_requested_rgb_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    original_convert = Image.Image.convert
    converted: list[str | None] = []

    def tracked_convert(image, *args, **kwargs):
        converted.append(getattr(image, "filename", None))
        return original_convert(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", tracked_convert)

    run(_args(config, tmp_path / "dense", num_frames=1))

    assert len(converted) == 1


def test_replica_preflight_binds_paths_without_rgb_pixels(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)

    preflight = _preflight(config, 2, _args(config, tmp_path / "dense"))

    assert not hasattr(preflight, "rgb_frames")
    assert [binding.source_frame_id for binding in preflight.rgb_bindings] == [10, 13]
    assert [binding.dataset_frame_id for binding in preflight.rgb_bindings] == [0, 1]
    assert all(isinstance(binding.path, Path) for binding in preflight.rgb_bindings)
    assert all(binding.size > 0 for binding in preflight.rgb_bindings)
    assert all(binding.device >= 0 and binding.inode > 0 for binding in preflight.rgb_bindings)
    assert all(binding.mtime_ns > 0 and binding.ctime_ns > 0 for binding in preflight.rgb_bindings)


def test_rgb_binding_rejects_path_replacement_after_preflight(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    preflight = _preflight(config, 1, _args(config, tmp_path / "dense", num_frames=1))
    binding = preflight.rgb_bindings[0]
    replacement = tmp_path / "replacement.jpg"
    Image.fromarray(np.full((4, 5, 3), 99, dtype=np.uint8)).save(replacement)
    os.replace(replacement, binding.path)

    with pytest.raises(ValueError, match="identity"):
        _load_rgb_frame(preflight.dataset, 0, binding)


def test_rgb_binding_rejects_in_place_content_change_after_preflight(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    preflight = _preflight(config, 1, _args(config, tmp_path / "dense", num_frames=1))
    binding = preflight.rgb_bindings[0]
    payload = bytearray(binding.path.read_bytes())
    payload[-1] ^= 1
    with binding.path.open("r+b") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.utime(
        binding.path,
        ns=(binding.mtime_ns + 1, binding.mtime_ns + 1),
    )

    with pytest.raises(ValueError, match="identity|changed"):
        _load_rgb_frame(preflight.dataset, 0, binding)


def test_rgb_loader_decodes_only_bytes_read_from_bound_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    preflight = _preflight(config, 1, _args(config, tmp_path / "dense", num_frames=1))
    binding = preflight.rgb_bindings[0]
    original_open = Image.open
    opened: list[object] = []

    def tracked_open(source, *args, **kwargs):
        opened.append(source)
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Image, "open", tracked_open)

    rgb = _load_rgb_frame(preflight.dataset, 0, binding)

    assert rgb.shape == (4, 5, 3)
    assert opened and all(isinstance(source, io.BytesIO) for source in opened)


def test_multiframe_replica_run_indexes_each_frame_directory_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    results = tmp_path / "dataset" / "results"
    original = Path.glob
    indexed: list[str] = []

    def tracked_glob(directory: Path, pattern: str):
        if directory == results and pattern in {"frame*.jpg", "depth*.png"}:
            indexed.append(pattern)
        return original(directory, pattern)

    monkeypatch.setattr(Path, "glob", tracked_glob)

    run(_args(config, tmp_path / "dense"))

    assert indexed.count("frame*.jpg") == 1
    assert indexed.count("depth*.png") == 1


def test_fresh_preflight_does_not_decode_depth_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_fixture(tmp_path)

    def reject_depth_load(self, *args, **kwargs):
        raise AssertionError("depth pixels must not be decoded")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", reject_depth_load)

    manifest = run(_args(config, tmp_path / "dense", num_frames=1))

    assert manifest["image_shape"] == [4, 5]


def test_resume_does_not_decode_dataset_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    expected = run(_args(config, output))

    def reject_rgb_convert(self, *args, **kwargs):
        raise AssertionError("resume must not decode RGB pixels")

    def reject_depth_load(self, *args, **kwargs):
        raise AssertionError("resume must not decode depth pixels")

    monkeypatch.setattr(Image.Image, "convert", reject_rgb_convert)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", reject_depth_load)

    assert run(_args(config, output, resume=True)) == expected


def test_load_json_rejects_oversize_from_stat_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversize.json"
    with path.open("wb") as stream:
        stream.seek(8 * 1024 * 1024)
        stream.write(b"x")

    def forbidden_read_bytes(self):
        raise AssertionError("oversize JSON must be rejected before allocation")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    with pytest.raises(ValueError, match="JSON size limit"):
        _load_json(path, "fixture")


def test_load_json_rechecks_size_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "growing.json"
    path.write_text("{}", encoding="utf-8")
    real_open = os.open
    grew = False

    def grow_before_open(candidate, flags, *args, **kwargs):
        nonlocal grew
        if not grew and Path(candidate) == path:
            grew = True
            with path.open("ab") as stream:
                stream.write(b"x" * (8 * 1024 * 1024))
        return real_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", grow_before_open)

    with pytest.raises(ValueError, match="JSON size limit"):
        _load_json(path, "fixture")


def test_producer_sets_strict_jsonl_response_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    real_client = precompute_module.JsonLineWorkerClient
    captured: dict[str, object] = {}

    def recording_client(**kwargs):
        captured.update(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr(precompute_module, "JsonLineWorkerClient", recording_client)

    run(_args(config, tmp_path / "dense", num_frames=1))

    assert captured["max_response_chars"] == 16 * 1024 * 1024 + 1


def test_output_directory_creation_fsyncs_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "new-parent" / "dense"
    fsynced: list[Path] = []
    real_fsync = precompute_module._fsync_directory

    def tracking_fsync(path: Path) -> None:
        fsynced.append(Path(path))
        real_fsync(path)

    monkeypatch.setattr(precompute_module, "_fsync_directory", tracking_fsync)

    run(_args(config, output, num_frames=1))

    assert output.parent in fsynced


def test_atomic_manifest_publish_never_overwrites_concurrent_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dense_manifest.json"
    concurrent = b'{"owner":"concurrent"}\n'
    real_link = os.link

    def create_concurrent_target(source, target, *args, **kwargs):
        destination.write_bytes(concurrent)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", create_concurrent_target)

    with pytest.raises(FileExistsError):
        _atomic_json(destination, {"owner": "producer"})

    assert destination.read_bytes() == concurrent
    assert list(tmp_path.glob(".dense_manifest.json.*.tmp")) == []


def test_atomic_manifest_rejects_parent_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "dense"
    parent.mkdir()
    destination = parent / "dense_manifest.json"
    displaced = tmp_path / "displaced"
    real_link = os.link
    replaced = False

    def replace_parent_before_link(source, target, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(displaced)
            parent.mkdir()
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", replace_parent_before_link)

    with pytest.raises(
        precompute_module.DenseManifestPublicationUncertainError,
        match="uncertain",
    ):
        _atomic_json(destination, {"owner": "producer"})

    assert not destination.exists()
    assert json.loads((displaced / destination.name).read_text(encoding="utf-8")) == {
        "owner": "producer"
    }
    assert list(parent.iterdir()) == []
    assert [path.name for path in displaced.iterdir()] == [destination.name]


def test_atomic_manifest_post_link_fsync_failure_is_uncertain_and_leaks_no_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dense_manifest.json"
    payload = {"complete": True, "frames": 2}
    real_fsync = os.fsync
    before_fds = len(list(Path("/proc/self/fd").iterdir()))

    def fail_published_directory_fsync(descriptor: int) -> None:
        status = os.fstat(descriptor)
        if stat.S_ISDIR(status.st_mode) and destination.exists():
            raise OSError("injected post-link directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_published_directory_fsync)

    with pytest.raises(
        precompute_module.DenseManifestPublicationUncertainError,
        match="uncertain",
    ):
        _atomic_json(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(".dense_manifest.json.*.tmp")) == []
    assert len(list(Path("/proc/self/fd").iterdir())) == before_fds


def test_atomic_manifest_pre_link_failure_has_no_target_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dense_manifest.json"
    real_link = os.link
    attempts = 0

    def fail_first_link(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected pre-link failure")
        return real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", fail_first_link)

    with pytest.raises(OSError, match="pre-link"):
        _atomic_json(destination, {"attempt": 1})

    assert not destination.exists()
    assert list(tmp_path.glob(".dense_manifest.json.*.tmp")) == []

    _atomic_json(destination, {"attempt": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"attempt": 2}
    assert attempts == 2


def test_metadata_failure_does_not_create_output_and_closes_worker(tmp_path: Path) -> None:
    config, _, log_path = _write_fixture(tmp_path, mode="metadata_error")
    output = tmp_path / "dense"

    with pytest.raises(RuntimeError, match="metadata rejected"):
        run(_args(config, output))

    assert not output.exists()
    assert [event["event"] for event in _events(log_path)] == [
        "start",
        "metadata",
        "closed",
    ]


@pytest.mark.parametrize(
    "mode",
    ["bad_metadata_classes", "bad_metadata_vocab", "missing_language_provenance"],
)
def test_invalid_metadata_never_creates_output(tmp_path: Path, mode: str) -> None:
    config, _, _ = _write_fixture(tmp_path, mode=mode)
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match="classes|vocabulary|provenance|language"):
        run(_args(config, output))

    assert not output.exists()


def test_second_frame_failure_leaves_partial_directory_without_manifest(
    tmp_path: Path,
) -> None:
    config, _, log_path = _write_fixture(tmp_path, mode="second_error")
    output = tmp_path / "dense"

    with pytest.raises(RuntimeError, match="second frame failed"):
        run(_args(config, output))

    assert (output / "frame000000.npz").is_file()
    assert not (output / "frame000001.npz").exists()
    assert not (output / "dense_manifest.json").exists()
    assert _events(log_path)[-1]["event"] == "closed"


def test_existing_output_is_rejected_without_starting_worker(tmp_path: Path) -> None:
    config, _, log_path = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run(_args(config, output))

    assert not log_path.exists()


def test_existing_symlink_output_is_rejected(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "dense"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError):
        run(_args(config, output))


def test_output_rejects_symlink_in_existing_parent_chain(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    target = tmp_path / "target-parent"
    target.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(target, target_is_directory=True)
    output = linked_parent / "nested" / "dense"

    with pytest.raises(ValueError, match="symlink"):
        run(_args(config, output, num_frames=1))

    assert not (target / "nested" / "dense").exists()


def test_verified_resume_is_byte_and_mtime_idempotent(tmp_path: Path) -> None:
    config, _, log_path = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    first = run(_args(config, output))
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(output.iterdir())
    }

    resumed = run(_args(config, output, num_frames=1, resume=True))

    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(output.iterdir())
    }
    assert resumed == first
    assert after == before
    assert [event["event"] for event in _events(log_path)].count("start") == 2
    assert [event["event"] for event in _events(log_path)].count("infer") == 2


def test_resume_rejects_tampered_cache_without_modifying_output(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    run(_args(config, output))
    frame_path = output / "frame000001.npz"
    frame_path.write_bytes(frame_path.read_bytes() + b"tampered")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ValueError, match="checksum|hash|corrupt"):
        run(_args(config, output, resume=True))

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_resume_rejects_partial_directory(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    output.mkdir()
    (output / "frame000000.npz").write_bytes(b"partial")

    with pytest.raises(ValueError, match="manifest|complete|partial"):
        run(_args(config, output, resume=True))


def test_resume_rejects_manifest_with_extra_key(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    run(_args(config, output))
    manifest_path = output / "dense_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="manifest.*keys|unsupported"):
        run(_args(config, output, resume=True))


def test_resume_rejects_provenance_with_extra_key(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    run(_args(config, output))
    manifest_path = output / "dense_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["unexpected"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="provenance.*keys|unsupported"):
        run(_args(config, output, resume=True))


@pytest.mark.parametrize("mode", ["bad_shape", "bad_dtype", "bad_base64"])
def test_invalid_array_response_leaves_no_manifest(tmp_path: Path, mode: str) -> None:
    config, _, _ = _write_fixture(tmp_path, mode=mode)
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match="shape|dtype|base64|byte"):
        run(_args(config, output, num_frames=1))

    assert output.is_dir()
    assert not (output / "dense_manifest.json").exists()


def test_inference_response_rejects_boolean_integer_fields(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path, mode="bool_infer_stride")
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match="sample_stride.*integer"):
        run(_args(config, output, num_frames=1))

    assert not (output / "dense_manifest.json").exists()


def test_inference_response_rejects_float_image_shape(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path, mode="float_infer_image_shape")
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match="image_shape.*integer"):
        run(_args(config, output, num_frames=1))

    assert not (output / "dense_manifest.json").exists()


def test_preflight_rejects_non_object_benchmark_vocabulary(tmp_path: Path) -> None:
    config_path, _, log_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    benchmark_path = Path(config["manifest"])
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["vocabulary"] = []
    _write_json(benchmark_path, benchmark)
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match="vocabulary"):
        run(_args(config_path, output))

    assert not output.exists()
    assert not log_path.exists()


def test_preflight_rejects_float_sampled_frame_count(tmp_path: Path) -> None:
    config_path, _, log_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    benchmark_path = Path(config["manifest"])
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["frame_selection"]["sampled_frames_per_scene"] = 2.0
    _write_json(benchmark_path, benchmark)
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match="sampled_frames_per_scene.*integer"):
        run(_args(config_path, output))

    assert not output.exists()
    assert not log_path.exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("image_shape", [4.0, 5]),
        ("sample_stride", 2.0),
        ("top_k", 2.0),
        ("class_count", 41.0),
    ],
)
def test_resume_rejects_float_integer_schema_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    run(_args(config, output))
    manifest_path = output / "dense_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = invalid
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=f"manifest {field}.*integer"):
        run(_args(config, output, resume=True))


def test_worker_command_can_derive_classes_json_argument(tmp_path: Path) -> None:
    config_path, classes_json, _ = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    dense = config["dense_semantics"]
    dense.pop("classes_json")
    dense["worker_command"].extend(["--classes-json", str(classes_json)])
    _write_json(config_path, config)

    manifest = run(_args(config_path, tmp_path / "dense", num_frames=1))

    assert manifest["vocabulary_sha256"] == hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()


def test_config_classes_json_does_not_require_manifest_source_path(
    tmp_path: Path,
) -> None:
    config_path, classes_json, _ = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    benchmark_path = Path(config["manifest"])
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["vocabulary"].pop("source_path")
    _write_json(benchmark_path, benchmark)

    manifest = run(_args(config_path, tmp_path / "dense", num_frames=1))

    assert manifest["vocabulary_sha256"] == hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()


def test_cli_classes_json_overrides_conflicting_config_sources(
    tmp_path: Path,
) -> None:
    config_path, classes_json, _ = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    first_stale = tmp_path / "first-stale.json"
    second_stale = tmp_path / "second-stale.json"
    first_stale.write_text(
        json.dumps({"classes": CLASSES, "aliases": {}}, indent=1),
        encoding="utf-8",
    )
    second_stale.write_text(
        json.dumps({"aliases": {}, "classes": CLASSES}, indent=3),
        encoding="utf-8",
    )
    dense = config["dense_semantics"]
    dense["classes_json"] = str(first_stale)
    dense["worker_command"].extend(["--classes-json", str(second_stale)])
    _write_json(config_path, config)
    args = parse_args(
        [
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "dense"),
            "--num-frames",
            "1",
            "--classes-json",
            str(classes_json),
        ]
    )

    manifest = run(args)

    assert manifest["vocabulary_sha256"] == hashlib.sha256(
        classes_json.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("worker_command", []),
        ("worker_command", "python worker.py"),
        ("worker_command", [sys.executable, ""]),
        ("worker_command", ["   "]),
        ("worker_env", {"A": 1}),
        ("worker_cwd", 3),
        ("request_timeout_sec", True),
        ("request_timeout_sec", 0),
        ("request_timeout_sec", 10**400),
    ],
)
def test_invalid_worker_config_is_rejected_before_output(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    config_path, _, _ = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["dense_semantics"][field] = invalid
    _write_json(config_path, config)
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match=field):
        run(_args(config_path, output))

    assert not output.exists()


@pytest.mark.parametrize("invalid", [True, 0, -1, 3])
def test_config_num_frames_must_match_frozen_selection(
    tmp_path: Path,
    invalid: object,
) -> None:
    config_path, _, _ = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["num_frames"] = invalid
    _write_json(config_path, config)
    output = tmp_path / "dense"

    with pytest.raises(ValueError, match="num_frames|selection"):
        run(_args(config_path, output))

    assert not output.exists()


def test_source_selection_must_exactly_match_benchmark_manifest(tmp_path: Path) -> None:
    config_path, _, _ = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["source_stride"] = 4
    _write_json(config_path, config)

    with pytest.raises(ValueError, match="frame_selection|source_stride"):
        run(_args(config_path, tmp_path / "dense"))


def test_resume_rejects_recorded_symlink_frame(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    output = tmp_path / "dense"
    run(_args(config, output))
    frame = output / "frame000001.npz"
    target = tmp_path / "copied.npz"
    target.write_bytes(frame.read_bytes())
    frame.unlink()
    frame.symlink_to(target)

    with pytest.raises(ValueError, match="symlink|regular"):
        run(_args(config, output, resume=True))
