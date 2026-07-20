from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.precompute_oviv2_dense_semantics import (
    _cache_prefix_sha256,
    parse_args,
    run,
)
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


classes_path = Path(os.environ["FAKE_CLASSES_JSON"])
classes_raw = classes_path.read_bytes()
classes = json.loads(classes_raw)["classes"]
vocabulary_sha256 = hashlib.sha256(classes_raw).hexdigest()
mode = os.environ.get("FAKE_MODE", "ok")
sample_stride = 1 if mode == "bool_infer_stride" else 2
top_k = 2
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
log({"event": "start", "pid": os.getpid()})

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
                class_ids[..., 0] = 1
                class_ids[..., 1] = 2
                probabilities = np.empty((*sampled, top_k), dtype=np.float32)
                probabilities[..., 0] = 0.75
                probabilities[..., 1] = 0.20
                entropy = np.full(sampled, 0.5, dtype=np.float32)
                margin = np.full(sampled, 0.55, dtype=np.float32)
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
            "vocabulary": {"classes": CLASSES},
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
    assert manifest["provenance"]["cache_prefix_sha256"] == _cache_prefix_sha256(
        [10, 13], manifest["cache_files_sha256"]
    )
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
