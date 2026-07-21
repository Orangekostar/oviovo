from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

import scripts.build_oviv2_hybrid_frontend_cache as builder
from scripts.build_oviv2_hybrid_frontend_cache import parse_args, run
from scripts.run_oviv2_replica import algorithm_hash
from src.oviv2.dense_semantics import (
    DenseSemanticFrame,
    sha256_file,
    write_dense_frame,
)
from src.oviv2.hybrid_cache import load_frontend_batch
from src.oviv2.hybrid_frontend import HybridFrontendConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUTE2_SHARED_SOURCES = {
    "yolo_cache_dir": "/home/ww/vv/dataset/Replica/room0_s10_200f/gsa_detections_yolo_room0_s10_200f",
    "sam_cache_dir": "/home/ww/oviovo_baseline_runs/20260719_conceptgraphs_canonical/room0/input/room0/gsa_detections_none_canonical_room0_s10_200f",
    "sam_gate": "/home/ww/oviovo_baseline_runs/20260719_conceptgraphs_canonical/room0/gate.json",
    "sam_generator_script": "/home/ww/oviovo_baseline_builds/conceptgraphs-canonical-runtime/conceptgraph/scripts/generate_gsa_results.py",
    "sam_runtime_patch": "/home/ww/oviovo_baseline_runs/20260719_conceptgraphs_canonical/room0/runtime_path_localization.patch",
    "clip_checkpoint": "/home/ww/vv/paper2/DovSG/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
}
_ROUTE2_CONFIGS = {
    "sam_labeled": (
        "oviv2_replica_room0_route2_sam_labeled.json",
        "/home/ww/oviovo_experiments/20260721_route2_frontend/sam_labeled/cache",
    ),
    "yolo_novel_sam": (
        "oviv2_replica_room0_route2_yolo_novel_sam.json",
        "/home/ww/oviovo_experiments/20260721_route2_frontend/yolo_novel_sam/cache",
    ),
    "quota_nms_ensemble": (
        "oviv2_replica_room0_route2_quota_nms.json",
        "/home/ww/oviovo_experiments/20260721_route2_frontend/quota_nms_ensemble/cache",
    ),
}


def _load_repo_json(relative_path: str) -> dict[str, object]:
    payload = json.loads((_REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(("variant", "config_record"), _ROUTE2_CONFIGS.items())
def test_route2_config_only_changes_permitted_base_fields(
    variant: str, config_record: tuple[str, str]
) -> None:
    filename, output = config_record
    baseline = _load_repo_json(
        "configs/oviv2_replica_room0_precision_stage3_fused.json"
    )
    config = _load_repo_json(f"configs/{filename}")

    excluded = {"frontend_cache_dir", "algorithm_hash", "hybrid_frontend"}
    assert {key: value for key, value in config.items() if key not in excluded} == {
        key: value for key, value in baseline.items() if key not in excluded
    }
    assert set(config) == set(baseline) | {"algorithm_hash", "hybrid_frontend"}
    assert config["frontend_cache_dir"] == output
    assert config["hybrid_frontend"]["variant"] == variant


@pytest.mark.parametrize(("variant", "config_record"), _ROUTE2_CONFIGS.items())
def test_route2_config_freezes_shared_sources_and_every_policy_default(
    variant: str, config_record: tuple[str, str]
) -> None:
    filename, _ = config_record
    hybrid = _load_repo_json(f"configs/{filename}")["hybrid_frontend"]

    assert set(hybrid) == {
        "variant",
        *_ROUTE2_SHARED_SOURCES,
        *_ROUTE2_CONFIGS,
    }
    assert {name: hybrid[name] for name in _ROUTE2_SHARED_SOURCES} == (
        _ROUTE2_SHARED_SOURCES
    )
    for policy_variant in _ROUTE2_CONFIGS:
        assert hybrid[policy_variant] == asdict(
            HybridFrontendConfig(variant=policy_variant)
        )
    assert hybrid["quota_nms_ensemble"]["maximum_proposals"] == 64
    assert hybrid["quota_nms_ensemble"]["maximum_per_class"] == 12
    assert hybrid["quota_nms_ensemble"]["maximum_compact_rescues"] == 4


@pytest.mark.parametrize(("variant", "config_record"), _ROUTE2_CONFIGS.items())
def test_route2_config_has_runner_derived_algorithm_hash(
    variant: str, config_record: tuple[str, str]
) -> None:
    filename, _ = config_record
    config = _load_repo_json(f"configs/{filename}")

    assert config["hybrid_frontend"]["variant"] == variant
    assert config["algorithm_hash"] == algorithm_hash(config)


@dataclass(frozen=True)
class BuilderFixture:
    config: Path
    output: Path
    yolo_manifest: Path
    dense_manifest: Path
    frame_manifest: Path
    checkpoint: Path
    generator: Path
    runtime_patch: Path
    sam_dir: Path


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_payload(*, mask: np.ndarray, label: str, confidence: float) -> dict[str, object]:
    ys, xs = np.nonzero(mask)
    box = np.asarray(
        [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float32
    )
    return {
        "xyxy": box,
        "confidence": np.asarray([confidence], dtype=np.float32),
        "class_id": np.asarray([0], dtype=np.int64),
        "mask": mask[None],
        "classes": [label],
        "image_crops": [None],
        "image_feats": np.asarray([[3.0, 4.0]], dtype=np.float32),
        "text_feats": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }


def _write_legacy(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=path, mode="wb", mtime=0) as stream:
        pickle.dump(payload, stream, protocol=4)


def _dense_frame(cache_index: int, source_id: int) -> DenseSemanticFrame:
    shape = (4, 5)
    class_ids = np.full((*shape, 1), 4, dtype=np.int64)
    probabilities = np.full((*shape, 1), 0.9, dtype=np.float32)
    return DenseSemanticFrame(
        cache_frame_id=cache_index,
        source_frame_id=source_id,
        image_shape=shape,
        sample_stride=1,
        class_count=5,
        class_ids=class_ids,
        probabilities=probabilities,
        entropy=np.zeros(shape, dtype=np.float32),
        margin=probabilities[..., 0].copy(),
    )


def write_builder_fixture(tmp_path: Path, *, frame_count: int = 2) -> BuilderFixture:
    dataset = tmp_path / "dataset"
    results = dataset / "results"
    yolo_dir = tmp_path / "yolo"
    sam_dir = tmp_path / "sam"
    dense_dir = tmp_path / "dense"
    source_depths = tmp_path / "source-depths"
    for directory in (results, yolo_dir, sam_dir, dense_dir, source_depths):
        directory.mkdir(parents=True)

    source_ids = [index * 10 for index in range(frame_count)]
    frame_records: list[dict[str, object]] = []
    yolo_hashes: dict[str, str] = {}
    dense_hashes: dict[str, str] = {}
    for cache_index, source_id in enumerate(source_ids):
        depth_path = results / f"depth{cache_index:06d}.png"
        source_depth = source_depths / f"depth{source_id:06d}.png"
        depth = np.ones((4, 5), dtype=np.uint16)
        depth[0, 0] = 0
        Image.fromarray(depth).save(source_depth)
        depth_path.symlink_to(source_depth)
        frame_records.append(
            {
                "sampled_frame_id": cache_index,
                "source_frame_id": source_id,
                "depth_sha256": _sha(depth_path),
            }
        )

        yolo_mask = np.zeros((4, 5), dtype=bool)
        yolo_mask[:2, :2] = True
        yolo_path = yolo_dir / f"frame{cache_index:06d}.pkl.gz"
        _write_legacy(
            yolo_path,
            _legacy_payload(mask=yolo_mask, label="table", confidence=0.8),
        )
        yolo_hashes[yolo_path.name] = _sha(yolo_path)

        sam_mask = np.zeros((4, 5), dtype=bool)
        sam_mask[2:, 3:] = True
        _write_legacy(
            sam_dir / f"frame{source_id:06d}.pkl.gz",
            _legacy_payload(mask=sam_mask, label="item", confidence=0.9),
        )

        dense_path = dense_dir / f"frame{cache_index:06d}.npz"
        write_dense_frame(dense_path, _dense_frame(cache_index, source_id))
        dense_hashes[dense_path.name] = sha256_file(dense_path)

    frame_manifest = dataset / "frame_manifest.json"
    _json(
        frame_manifest,
        {
            "dataset": "Replica",
            "frame_count": frame_count,
            "source_frame_ids": source_ids,
            "frames": frame_records,
        },
    )

    benchmark = tmp_path / "benchmark.json"
    vocabulary_sha = "a" * 64
    classes = ["wall", "floor", "ceiling", "chair", "table"]
    _json(
        benchmark,
        {
            "dataset": "Replica",
            "vocabulary": {
                "source_sha256": vocabulary_sha,
                "classes": classes,
            },
            "scenes": [{"scene": "room0"}],
        },
    )

    checkpoint = tmp_path / "clip.pt"
    generator = tmp_path / "generator.py"
    runtime_patch = tmp_path / "runtime.patch"
    checkpoint.write_bytes(b"clip checkpoint")
    generator.write_bytes(b"canonical generator")
    runtime_patch.write_bytes(b"runtime patch")

    yolo_manifest = yolo_dir / "frontend_manifest.json"
    _json(
        yolo_manifest,
        {
            "schema_version": 1,
            "method": "OVIV2",
            "scene": "room0",
            "frame_count": frame_count,
            "mask_shape": [4, 5],
            "classes": ["table"],
            "provenance_sha256": {"clip_model": _sha(checkpoint)},
            "cache_files_sha256": yolo_hashes,
        },
    )

    dense_manifest = dense_dir / "dense_manifest.json"
    _json(
        dense_manifest,
        {
            "schema_version": 1,
            "method": "OVIV2-dense-semantic-cache",
            "scene": "room0",
            "frame_count": frame_count,
            "source_frame_ids": source_ids,
            "image_shape": [4, 5],
            "sample_stride": 1,
            "top_k": 1,
            "class_count": len(classes),
            "vocabulary_sha256": vocabulary_sha,
            "cache_files_sha256": dense_hashes,
        },
    )

    gate = tmp_path / "sam_gate.json"
    _json(gate, {"status": "VERIFIED", "headline_result_eligible": True})
    policies = {
        name: {
            "minimum_area_fraction": 0.01,
            "minimum_valid_depth_fraction": 0.5,
            "minimum_dense_probability": 0.25,
            "minimum_dense_margin": 0.05,
            "maximum_structure_probability": 0.5,
        }
        for name in ("sam_labeled", "yolo_novel_sam", "quota_nms_ensemble")
    }
    config = tmp_path / "runner.json"
    _json(
        config,
        {
            "scene": "room0",
            "dataset_root": str(dataset),
            "dense_cache_dir": str(dense_dir),
            "manifest": str(benchmark),
            "source_start": 0,
            "source_stride": 10,
            "hybrid_frontend": {
                "variant": "yolo_novel_sam",
                "yolo_cache_dir": str(yolo_dir),
                "sam_cache_dir": str(sam_dir),
                "sam_gate": str(gate),
                "sam_generator_script": str(generator),
                "sam_runtime_patch": str(runtime_patch),
                "clip_checkpoint": str(checkpoint),
                **policies,
            },
        },
    )
    return BuilderFixture(
        config=config,
        output=tmp_path / "output",
        yolo_manifest=yolo_manifest,
        dense_manifest=dense_manifest,
        frame_manifest=frame_manifest,
        checkpoint=checkpoint,
        generator=generator,
        runtime_patch=runtime_patch,
        sam_dir=sam_dir,
    )


def _patch_required_asset_hashes(
    monkeypatch: pytest.MonkeyPatch, fixture: BuilderFixture
) -> None:
    monkeypatch.setattr(builder, "REQUIRED_CLIP_SHA256", _sha(fixture.checkpoint))
    monkeypatch.setattr(builder, "REQUIRED_GENERATOR_SHA256", _sha(fixture.generator))
    monkeypatch.setattr(builder, "REQUIRED_RUNTIME_PATCH_SHA256", _sha(fixture.runtime_patch))


def _run(fixture: BuilderFixture, *extra: str) -> dict[str, object]:
    return run(
        parse_args(
            [
                "--config",
                str(fixture.config),
                "--variant",
                "yolo_novel_sam",
                "--output",
                str(fixture.output),
                *extra,
            ]
        )
    )


def _mutate_json(path: Path, mutate: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    _json(path, payload)


def test_builder_maps_cache_index_to_sam_source_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)

    manifest = _run(fixture)

    assert manifest["frame_count"] == 2
    assert manifest["source_frame_ids"] == [0, 10]
    assert manifest["diagnostics"]["accepted_novel_sam"] == 2
    assert "sam/frame000010.pkl.gz" in manifest["source_sha256"]
    assert "depth/depth000001.png" in manifest["source_sha256"]
    assert (fixture.output / "frontend_manifest.json").is_file()
    restored = load_frontend_batch(fixture.output / "frame000001.pkl.gz")
    assert restored.labels == ("table", "chair")
    assert manifest["classes"] == ["table", "wall", "floor", "ceiling", "chair"]


def test_builder_rejects_cli_variant_mismatch_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    args = parse_args(
        [
            "--config", str(fixture.config),
            "--variant", "sam_labeled",
            "--output", str(fixture.output),
        ]
    )

    with pytest.raises(ValueError, match="variant"):
        run(args)
    assert not os.path.lexists(fixture.output)


def test_builder_rejects_clip_manifest_and_checkpoint_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    _mutate_json(
        fixture.yolo_manifest,
        lambda value: value["provenance_sha256"].__setitem__("clip_model", "0" * 64),
    )

    with pytest.raises(ValueError, match="CLIP"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


def test_builder_rejects_dense_vocabulary_mismatch_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    _mutate_json(
        fixture.dense_manifest,
        lambda value: value.__setitem__("vocabulary_sha256", "b" * 64),
    )

    with pytest.raises(ValueError, match="vocabulary"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


def test_builder_rejects_cross_source_image_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    _mutate_json(
        fixture.dense_manifest,
        lambda value: value.__setitem__("image_shape", [4, 6]),
    )

    with pytest.raises(ValueError, match="image shape"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


def test_builder_rejects_dense_source_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    _mutate_json(
        fixture.dense_manifest,
        lambda value: value.__setitem__("source_frame_ids", [0, 11]),
    )

    with pytest.raises(ValueError, match="source_frame_id"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


def test_builder_rejects_missing_selected_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    _mutate_json(
        fixture.yolo_manifest,
        lambda value: value["cache_files_sha256"].pop("frame000001.pkl.gz"),
    )

    with pytest.raises(ValueError, match="source hash"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


def test_builder_preflights_all_selected_checksums_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    (fixture.sam_dir / "frame000010.pkl.gz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SAM"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


def test_builder_removes_partial_output_when_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    real_select = builder.select_hybrid_proposals
    call_count = 0

    def fail_second_conversion(**kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("second conversion failed")
        return real_select(**kwargs)

    monkeypatch.setattr(builder, "select_hybrid_proposals", fail_second_conversion)

    with pytest.raises(RuntimeError, match="second conversion failed"):
        _run(fixture)
    assert call_count == 2
    assert not os.path.lexists(fixture.output)


def test_builder_refuses_existing_partial_output_and_dangling_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    fixture.output.mkdir()
    marker = fixture.output / "frame000000.pkl.gz"
    marker.write_bytes(b"partial")

    with pytest.raises(FileExistsError):
        _run(fixture)
    assert marker.read_bytes() == b"partial"

    other = write_builder_fixture(tmp_path / "other")
    _patch_required_asset_hashes(monkeypatch, other)
    other.output.symlink_to(tmp_path / "missing-output")
    with pytest.raises(FileExistsError):
        _run(other)


def test_num_frames_limits_preflight_to_selected_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    (fixture.sam_dir / "frame000010.pkl.gz").unlink()

    manifest = _run(fixture, "--num-frames", "1")

    assert manifest["frame_count"] == 1
    assert manifest["source_frame_ids"] == [0]


def test_asset_hashing_does_not_materialize_the_checkpoint_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "large-checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint" * 1024)
    expected = _sha(checkpoint)

    monkeypatch.setattr(
        builder,
        "_read_snapshot",
        lambda *_args, **_kwargs: pytest.fail("asset hashing read a full snapshot"),
    )

    assert builder._asset_hash(checkpoint, expected, "checkpoint") == expected


def test_builder_rejects_manifest_frame_count_disagreement_even_for_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    _mutate_json(
        fixture.dense_manifest,
        lambda value: value.__setitem__("frame_count", 1),
    )

    with pytest.raises(ValueError, match="frame_count"):
        _run(fixture, "--num-frames", "1")
    assert not os.path.lexists(fixture.output)


@pytest.mark.parametrize("manifest_name", ["dense", "frame"])
def test_builder_rejects_manifest_lists_longer_than_declared_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_name: str,
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    if manifest_name == "dense":
        _mutate_json(
            fixture.dense_manifest,
            lambda value: value["source_frame_ids"].append(20),
        )
    else:
        _mutate_json(
            fixture.frame_manifest,
            lambda value: value["frames"].append(dict(value["frames"][-1])),
        )

    with pytest.raises(ValueError, match="declared frame_count"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


@pytest.mark.parametrize("manifest_name", ["yolo", "dense"])
def test_builder_rejects_cache_hash_names_outside_declared_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_name: str,
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    manifest_path = (
        fixture.yolo_manifest if manifest_name == "yolo" else fixture.dense_manifest
    )
    _mutate_json(
        manifest_path,
        lambda value: value["cache_files_sha256"].__setitem__(
            "frame999999.pkl.gz" if manifest_name == "yolo" else "frame999999.npz",
            "0" * 64,
        ),
    )

    with pytest.raises(ValueError, match="exactly cover declared frame_count"):
        _run(fixture, "--num-frames", "1")
    assert not os.path.lexists(fixture.output)


def test_builder_rejects_top_level_frame_source_ids_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_builder_fixture(tmp_path)
    _patch_required_asset_hashes(monkeypatch, fixture)
    _mutate_json(
        fixture.frame_manifest,
        lambda value: value.__setitem__("source_frame_ids", [0, 11]),
    )

    with pytest.raises(ValueError, match="frame manifest source_frame_ids"):
        _run(fixture)
    assert not os.path.lexists(fixture.output)


def _legacy_snapshot(payload: object) -> bytes:
    destination = io.BytesIO()
    with gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as stream:
        pickle.dump(payload, stream, protocol=4)
    return destination.getvalue()


_UNSAFE_UNPICKLE_EXECUTED = False


def _mark_unsafe_unpickle() -> None:
    global _UNSAFE_UNPICKLE_EXECUTED
    _UNSAFE_UNPICKLE_EXECUTED = True


class _UnsafePickle:
    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_mark_unsafe_unpickle, ())


def test_legacy_loader_rejects_unapproved_pickle_globals_before_execution() -> None:
    global _UNSAFE_UNPICKLE_EXECUTED
    _UNSAFE_UNPICKLE_EXECUTED = False

    with pytest.raises(ValueError, match="unsafe pickle global"):
        builder._load_legacy_snapshot(_legacy_snapshot(_UnsafePickle()), field="legacy")

    assert _UNSAFE_UNPICKLE_EXECUTED is False


def test_legacy_loader_wraps_zero_dimensional_mask_as_value_error() -> None:
    mask = np.ones((4, 5), dtype=bool)
    payload = _legacy_payload(mask=mask, label="chair", confidence=0.9)
    payload["mask"] = np.asarray(True)

    with pytest.raises(ValueError):
        builder._load_legacy_snapshot(_legacy_snapshot(payload), field="legacy")


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("image_crops", []),
        ("text_feats", np.ones((2, 2), dtype=np.float32)),
        ("text_feats", np.asarray([[np.nan, 0.0]], dtype=np.float32)),
        ("text_feats", np.asarray([["unsafe", "object"]], dtype=object)),
    ],
)
def test_legacy_loader_validates_ignored_extras(
    field: str,
    invalid: object,
) -> None:
    mask = np.ones((4, 5), dtype=bool)
    payload = _legacy_payload(mask=mask, label="chair", confidence=0.9)
    payload[field] = invalid

    with pytest.raises(ValueError, match=field):
        builder._load_legacy_snapshot(_legacy_snapshot(payload), field="legacy")


def test_legacy_loader_clips_canonical_sam_confidence_roundoff() -> None:
    mask = np.ones((4, 5), dtype=bool)
    payload = _legacy_payload(mask=mask, label="item", confidence=1.05)

    batch = builder._load_legacy_snapshot(_legacy_snapshot(payload), field="SAM frame")

    assert batch.confidences.tolist() == [1.0]


def test_legacy_loader_rejects_confidence_outside_canonical_tolerance() -> None:
    mask = np.ones((4, 5), dtype=bool)
    payload = _legacy_payload(mask=mask, label="item", confidence=1.2)

    with pytest.raises(ValueError, match="confidence"):
        builder._load_legacy_snapshot(_legacy_snapshot(payload), field="SAM frame")


def test_builder_script_starts_outside_repository_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(builder.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
