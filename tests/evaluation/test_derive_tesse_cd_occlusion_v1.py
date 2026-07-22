from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

import scripts.evaluation.derive_tesse_cd_occlusion_v1 as occlusion_deriver
from scripts.evaluation.derive_tesse_cd_common_v2 import deterministic_npz_bytes
from scripts.evaluation.derive_tesse_cd_occlusion_v1 import (
    OcclusionPublicationUncertainError,
    StreamingDiagnostics,
    build_contract_manifest,
    depth_collection_binding,
    derive_occlusion_targets,
    render_manifest,
    resolve_dataset_source,
    validate_source_allowlist,
    validate_source_bindings,
    validate_generated_target,
    write_occlusion_package,
)
from src.evaluation.oviv2_occlusion import classify_depth


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/evaluation/manifests/tesse_cd_occlusion_v1.json"


def _frame(
    index: int, depth_m: float, *, timestamp_ns: int | None = None
) -> dict[str, object]:
    return {
        "frame_index": index,
        "relative_timestamp_ns": index if timestamp_ns is None else timestamp_ns,
        "depth": np.asarray([[depth_m]], dtype=np.float32),
        "world_from_camera": np.eye(4, dtype=np.float64),
    }


def _record(object_id: int, *, first: int = 0, last: int = 10) -> dict[str, object]:
    return {
        "id": object_id,
        "attributes": {
            "name": f"O({object_id})",
            "semantic_label": 3,
            "first_observed_ns": [first],
            "last_observed_ns": [last],
            "dynamic_object_points": [[[0.0, 0.0, 2.0]]],
        },
    }


def _fixture_inputs() -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
]:
    frames = {
        scene: [
            _frame(0, 2.0),
            _frame(1, 1.0),
            _frame(2, 1.0),
            _frame(3, 2.0),
        ]
        for scene in ("apartment", "office")
    }
    records = {scene: [_record(index)] for index, scene in enumerate(frames)}
    return frames, records


def _source_paths(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role in ("source_manifest", "schedule", "rgbd_lock", "camera"):
        path = root / f"{role}.bin"
        path.write_bytes(role.encode("ascii"))
        paths[role] = path
    for scene in ("apartment", "office"):
        for role in (
            "changes",
            "dsg_with_mesh",
            "export_manifest",
            "timestamps",
            "trajectory",
        ):
            path = root / f"{scene}.{role}.bin"
            path.write_bytes(f"{scene}.{role}".encode("ascii"))
            paths[f"{scene}.{role}"] = path
        for index in range(4):
            path = root / f"{scene}.depth.{index:06d}.png"
            path.write_bytes(f"{scene}.{index}".encode("ascii"))
            paths[f"{scene}.depth.{index:06d}"] = path
    return paths


def _bound_file(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def test_classify_depth_has_explicit_ten_centimeter_boundaries() -> None:
    assert classify_depth(d_obs=2.00, d_gt=2.05, tolerance_m=0.10) == "present"
    assert classify_depth(d_obs=1.50, d_gt=2.05, tolerance_m=0.10) == "occluded"
    assert classify_depth(d_obs=2.50, d_gt=2.05, tolerance_m=0.10) == "absent"
    assert classify_depth(d_obs=1.95, d_gt=2.05, tolerance_m=0.10) == "present"
    assert classify_depth(d_obs=2.15, d_gt=2.05, tolerance_m=0.10) == "present"
    assert classify_depth(d_obs=float("nan"), d_gt=2.05, tolerance_m=0.10) == "unobserved"
    assert classify_depth(d_obs=0.0, d_gt=2.05, tolerance_m=0.10) == "unobserved"

    with pytest.raises(ValueError, match="d_gt"):
        classify_depth(d_obs=2.0, d_gt=float("nan"), tolerance_m=0.10)
    with pytest.raises(ValueError, match="tolerance"):
        classify_depth(d_obs=2.0, d_gt=2.0, tolerance_m=0.0)


def test_derivation_uses_prior_present_anchor_and_merges_consecutive_occlusion() -> None:
    frames, records = _fixture_inputs()

    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )

    assert metadata["prediction_inputs_used"] is False
    assert metadata["parameters"] == json.loads(CONTRACT.read_text())["parameters"]
    assert metadata["scene_frame_indices"] == {
        "apartment": [0, 1, 2, 3],
        "office": [0, 1, 2, 3],
    }
    assert len(metadata["episodes"]) == 2
    for episode in metadata["episodes"]:
        assert episode["lifecycle"] == {
            "index": 0,
            "first_timestamp_ns": 0,
            "last_timestamp_ns": 10,
        }
        assert episode["anchor"]["frame_index"] == 0
        assert episode["start_frame_index"] == 1
        assert episode["end_frame_index"] == 2
        assert [item["frame_index"] for item in episode["checkpoints"]] == [1, 2]
        assert episode["occlusion_fraction"] == 1.0
        assert arrays[episode["anchor"]["array"]].dtype == np.int64
        assert arrays[episode["anchor"]["array"]].shape == (1, 3)
        for checkpoint in episode["checkpoints"]:
            assert np.array_equal(
                arrays[checkpoint["array"]], np.asarray([[0, 0, 40]], dtype=np.int64)
            )

    assert metadata["stress_layers"]["all"]["episode_count"] == 2
    assert metadata["stress_layers"]["0.50"]["episode_count"] == 2
    assert metadata["stress_layers"]["0.75"]["episode_count"] == 2
    assert metadata["stress_layers"]["0.90"]["episode_count"] == 2
    assert metadata["scenes"]["apartment"]["headline_episode_count"] == 1
    assert metadata["scenes"]["office"]["headline_episode_count"] == 1


def test_derivation_streams_one_depth_and_one_scene_dsg_and_inverts_once_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, records = _fixture_inputs()
    diagnostics = StreamingDiagnostics()
    original_inverse = occlusion_deriver.np.linalg.inv
    inverse_calls = 0

    def counted_inverse(transform: np.ndarray) -> np.ndarray:
        nonlocal inverse_calls
        inverse_calls += 1
        return original_inverse(transform)

    monkeypatch.setattr(occlusion_deriver.np.linalg, "inv", counted_inverse)
    derive_occlusion_targets(
        frames_by_scene={scene: iter(values) for scene, values in frames.items()},
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        diagnostics=diagnostics,
    )

    assert diagnostics.max_live_depth_frames == 1
    assert diagnostics.live_depth_frames == 0
    assert diagnostics.max_live_scene_dsgs == 1
    assert diagnostics.live_scene_dsgs == 0
    assert diagnostics.pose_inverse_count == 8
    assert inverse_calls == 8


def test_derivation_never_uses_unknown_or_future_lifecycle_as_anchor() -> None:
    frames, records = _fixture_inputs()
    records["apartment"][0]["attributes"]["semantic_label"] = 4_294_967_295
    records["office"][0]["attributes"]["first_observed_ns"] = [20]
    records["office"][0]["attributes"]["last_observed_ns"] = [30]

    with pytest.raises(ValueError, match="no qualifying occlusion episode"):
        derive_occlusion_targets(
            frames_by_scene=frames,
            dsg_records_by_scene=records,
            camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        )


def test_headline_layer_fails_closed_when_either_scene_has_no_ninety_percent_episode() -> None:
    frames, records = _fixture_inputs()
    frames["office"] = [_frame(0, 2.0), _frame(1, 2.0)]

    with pytest.raises(ValueError, match="office has no 0.90 occlusion episode"):
        derive_occlusion_targets(
            frames_by_scene=frames,
            dsg_records_by_scene=records,
            camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        )


@pytest.mark.parametrize(
    ("extra_role", "name"),
    [
        ("apartment.rgb.000000", "frame000000.jpg"),
        ("method.snapshot", "snapshot.npz"),
        ("prediction", "prediction.json"),
        ("frontend", "detections.pkl.gz"),
    ],
)
def test_source_allowlist_rejects_rgb_and_method_inputs(
    tmp_path: Path, extra_role: str, name: str
) -> None:
    paths = _source_paths(tmp_path)
    candidate = tmp_path / name
    candidate.write_bytes(b"forbidden")
    paths[extra_role] = candidate

    with pytest.raises(ValueError, match="exact source allowlist"):
        validate_source_allowlist(
            paths,
            scene_frame_indices={
                "apartment": [0, 1, 2, 3],
                "office": [0, 1, 2, 3],
            },
        )


def test_source_allowlist_rejects_method_path_hidden_under_allowed_role(
    tmp_path: Path,
) -> None:
    paths = _source_paths(tmp_path / "official")
    hidden = tmp_path / "method_output" / "camera.json"
    hidden.parent.mkdir()
    hidden.write_text("{}\n", encoding="utf-8")
    paths["camera"] = hidden

    with pytest.raises(ValueError, match="prediction or method output path"):
        validate_source_allowlist(
            paths,
            scene_frame_indices={
                "apartment": range(4),
                "office": range(4),
            },
        )


def test_source_allowlist_binds_every_file_sha_and_byte_count(tmp_path: Path) -> None:
    paths = _source_paths(tmp_path)
    first = validate_source_allowlist(
        paths,
        scene_frame_indices={"apartment": range(4), "office": range(4)},
    )
    assert list(first) == sorted(first)
    assert all(set(record) == {"path", "sha256", "byte_count"} for record in first.values())

    paths["office.depth.000003"].write_bytes(b"drift")
    second = validate_source_allowlist(
        paths,
        scene_frame_indices={"apartment": range(4), "office": range(4)},
    )
    assert first["office.depth.000003"] != second["office.depth.000003"]


@pytest.mark.parametrize("drift_role", ["camera", "office.depth.000003"])
def test_frozen_source_bindings_reject_hash_drift(
    tmp_path: Path, drift_role: str
) -> None:
    paths = _source_paths(tmp_path)
    indices = {"apartment": range(4), "office": range(4)}
    records = validate_source_allowlist(paths, scene_frame_indices=indices)
    expected_files = {
        role: record for role, record in records.items() if ".depth." not in role
    }
    expected_depth = {
        scene: depth_collection_binding(
            records, scene=scene, frame_indices=indices[scene]
        )
        for scene in ("apartment", "office")
    }
    paths[drift_role].write_bytes(b"source drift")

    with pytest.raises(ValueError, match="hash or byte count drift"):
        validate_source_bindings(
            paths,
            scene_frame_indices=indices,
            expected_source_records=expected_files,
            expected_depth_bindings=expected_depth,
        )


def test_manifest_and_npz_are_byte_identical_across_output_roots(tmp_path: Path) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    first = write_occlusion_package(
        tmp_path / "run-a",
        arrays=arrays,
        metadata=metadata,
        source_paths=paths,
        status="FIXTURE",
    )
    second = write_occlusion_package(
        tmp_path / "nested" / "run-b",
        arrays=arrays,
        metadata=metadata,
        source_paths=paths,
        status="FIXTURE",
    )

    assert first.read_bytes() == second.read_bytes()
    assert (first.parent / "targets.npz").read_bytes() == (
        second.parent / "targets.npz"
    ).read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["prediction_inputs_used"] is False
    assert payload["target_arrays"]["sha256"] == hashlib.sha256(
        deterministic_npz_bytes(arrays)
    ).hexdigest()
    assert payload["sources"] == validate_source_allowlist(
        paths, scene_frame_indices=metadata["scene_frame_indices"]
    )


def test_write_rejects_empty_episode_package(tmp_path: Path) -> None:
    paths = _source_paths(tmp_path / "sources")
    with pytest.raises(ValueError, match="no qualifying occlusion episode"):
        write_occlusion_package(
            tmp_path / "output",
            arrays={"unused": np.asarray([[0, 0, 1]], dtype=np.int64)},
            metadata={
                "prediction_inputs_used": False,
                "episodes": [],
                "scene_frame_indices": {
                    "apartment": [0, 1, 2, 3],
                    "office": [0, 1, 2, 3],
                },
            },
            source_paths=paths,
            status="FIXTURE",
        )


def test_writer_revalidates_frozen_bindings_before_publish(tmp_path: Path) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={
            "width": 1,
            "height": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        },
    )
    paths = _source_paths(tmp_path / "sources")
    indices = metadata["scene_frame_indices"]
    bound = validate_source_allowlist(paths, scene_frame_indices=indices)
    expected_files = {
        role: record for role, record in bound.items() if ".depth." not in role
    }
    expected_depth = {
        scene: depth_collection_binding(
            bound, scene=scene, frame_indices=indices[scene]
        )
        for scene in ("apartment", "office")
    }
    paths["camera"].write_bytes(b"changed after derivation")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="hash or byte count drift"):
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            expected_source_records=expected_files,
            expected_depth_bindings=expected_depth,
            status="FIXTURE",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "attack",
    [
        "single_scene",
        "missing_stratum",
        "zero_scene_headline",
        "duplicate_episode_id",
        "future_anchor",
        "bad_bounds",
        "empty_checkpoints",
        "missing_array",
        "dangling_array",
        "count_mismatch",
        "outside_anchor",
        "missing_object_id",
        "tolerance_99",
        "policy_drift",
        "extra_prediction_metadata",
        "extra_method_parameter",
    ],
)
def test_generated_target_validation_rejects_forged_or_incomplete_packages(
    attack: str,
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={
            "width": 1,
            "height": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        },
    )
    forged_arrays = {name: value.copy() for name, value in arrays.items()}
    forged = copy.deepcopy(metadata)
    if attack == "single_scene":
        forged["scenes"].pop("office")
    elif attack == "missing_stratum":
        forged["stress_layers"].pop("0.90")
    elif attack == "zero_scene_headline":
        forged["scenes"]["office"]["headline_episode_count"] = 0
    elif attack == "duplicate_episode_id":
        forged["episodes"].append(copy.deepcopy(forged["episodes"][0]))
    elif attack == "future_anchor":
        forged["episodes"][0]["anchor"]["relative_timestamp_ns"] = -1
    elif attack == "bad_bounds":
        forged["episodes"][0]["end_frame_index"] += 1
    elif attack == "empty_checkpoints":
        forged["episodes"][0]["checkpoints"] = []
    elif attack == "missing_array":
        forged["episodes"][0]["anchor"]["array"] = "missing"
    elif attack == "dangling_array":
        forged_arrays["dangling"] = np.asarray([[0, 0, 1]], dtype=np.int64)
    elif attack == "count_mismatch":
        forged["episodes"][0]["checkpoints"][0]["occluded_voxel_count"] += 1
    elif attack == "outside_anchor":
        array_name = forged["episodes"][0]["checkpoints"][0]["array"]
        forged_arrays[array_name] = np.asarray([[9, 9, 9]], dtype=np.int64)
    elif attack == "missing_object_id":
        forged["episodes"][0]["object_id"] = None
    elif attack == "tolerance_99":
        forged["parameters"]["depth_tolerance_m"] = 99.0
    elif attack == "policy_drift":
        forged["parameters"]["episode_rule"] = "merge arbitrary gaps"
    elif attack == "extra_prediction_metadata":
        forged["method_predictions_used"] = False
    elif attack == "extra_method_parameter":
        forged["parameters"]["method_mode"] = "signed_depth"

    with pytest.raises(ValueError, match="generated occlusion target"):
        validate_generated_target(forged_arrays, forged)


def test_generated_target_validation_accepts_complete_two_scene_package() -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={
            "width": 1,
            "height": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        },
    )

    validate_generated_target(arrays, metadata)


def test_derivation_rejects_non_preregistered_depth_tolerance() -> None:
    frames, records = _fixture_inputs()

    with pytest.raises(ValueError, match="depth_tolerance_m"):
        derive_occlusion_targets(
            frames_by_scene=frames,
            dsg_records_by_scene=records,
            camera={
                "width": 1,
                "height": 1,
                "fx": 1.0,
                "fy": 1.0,
                "cx": 0.0,
                "cy": 0.0,
            },
            depth_tolerance_m=99.0,
        )


@pytest.mark.parametrize("mutation", ["content", "same_bytes_new_inode"])
def test_source_change_after_initial_validation_prevents_publish_and_retry_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={
            "width": 1,
            "height": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        },
    )
    paths = _source_paths(tmp_path / "sources")
    output = tmp_path / "output"
    original_render = occlusion_deriver.render_manifest

    def mutate_after_serialization(payload: object) -> bytes:
        content = original_render(payload)
        if mutation == "content":
            paths["camera"].write_bytes(b"changed after initial validation")
        else:
            replacement = paths["camera"].with_suffix(".replacement")
            replacement.write_bytes(paths["camera"].read_bytes())
            replacement.replace(paths["camera"])
        return content

    monkeypatch.setattr(
        occlusion_deriver, "render_manifest", mutate_after_serialization
    )
    with pytest.raises(ValueError, match="source changed before publication"):
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))

    monkeypatch.setattr(occlusion_deriver, "render_manifest", original_render)
    manifest = write_occlusion_package(
        output,
        arrays=arrays,
        metadata=metadata,
        source_paths=paths,
        status="FIXTURE",
    )
    assert manifest.is_file()


def test_cross_file_drift_after_early_rehash_is_rejected_by_identity_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    output = tmp_path / "output"
    first_role = sorted(paths)[0]
    original_revalidate = occlusion_deriver._revalidate_source_witness
    revalidated = 0

    def mutate_after_first_rehash(witness: object) -> None:
        nonlocal revalidated
        original_revalidate(witness)
        revalidated += 1
        if revalidated == 1:
            paths[first_role].write_bytes(b"changed after its full rehash")

    monkeypatch.setattr(
        occlusion_deriver,
        "_revalidate_source_witness",
        mutate_after_first_rehash,
    )

    with pytest.raises(ValueError, match="source changed before publication"):
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )

    assert revalidated == len(paths)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))


def test_publication_identity_barrier_rejects_symlink_source(tmp_path: Path) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    camera_target = paths["camera"]
    camera_link = camera_target.with_suffix(".link")
    camera_link.symlink_to(camera_target)
    paths["camera"] = camera_link
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="ordinary non-symlink file"):
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))


def test_publish_pre_reservation_fsync_failure_cleans_staging_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    output = tmp_path / "output"
    original = occlusion_deriver._fsync_tree

    monkeypatch.setattr(
        occlusion_deriver,
        "_fsync_tree",
        lambda _path: (_ for _ in ()).throw(OSError("injected tree fsync failure")),
    )
    with pytest.raises(OSError, match="injected tree fsync failure"):
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))

    monkeypatch.setattr(occlusion_deriver, "_fsync_tree", original)
    assert write_occlusion_package(
        output,
        arrays=arrays,
        metadata=metadata,
        source_paths=paths,
        status="FIXTURE",
    ).is_file()


def test_publish_replace_failure_after_reservation_is_uncertain_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    output = tmp_path / "output"
    monkeypatch.setattr(
        occlusion_deriver.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("injected replace failure")),
    )

    with pytest.raises(OcclusionPublicationUncertainError) as raised:
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )

    assert raised.value.published is None
    assert raised.value.staging.is_dir()
    assert output.is_dir()
    assert not any(output.iterdir())


def test_publish_parent_fsync_failure_preserves_published_output_as_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    output = tmp_path / "output"
    original_fsync_directory = occlusion_deriver._fsync_directory

    def fail_published_parent(path: Path) -> None:
        if path == output.parent and output.exists():
            raise OSError("injected parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        occlusion_deriver,
        "_fsync_directory",
        fail_published_parent,
    )

    with pytest.raises(OcclusionPublicationUncertainError) as raised:
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )

    assert raised.value.published is True
    assert (output / "manifest.json").is_file()
    assert (output / "targets.npz").is_file()


def test_publish_never_overwrites_existing_output(tmp_path: Path) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")

    with pytest.raises(ValueError, match="output already exists"):
        write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )

    assert sentinel.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".output.*"))


def test_concurrent_publishers_have_exactly_one_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    output = tmp_path / "output"
    barrier = threading.Barrier(2)
    original_fsync_tree = occlusion_deriver._fsync_tree

    def synchronize_staging(path: Path) -> None:
        original_fsync_tree(path)
        barrier.wait(timeout=10)

    monkeypatch.setattr(occlusion_deriver, "_fsync_tree", synchronize_staging)

    def publish() -> Path:
        return write_occlusion_package(
            output,
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            status="FIXTURE",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish), executor.submit(publish)]
        outcomes = [future.exception() for future in futures]

    assert sum(error is None for error in outcomes) == 1
    assert sum(isinstance(error, ValueError) for error in outcomes) == 1
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".output.*"))


def test_writer_uses_captured_first_pass_without_independent_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )
    paths = _source_paths(tmp_path / "sources")
    captured_records, captured_witnesses = occlusion_deriver._capture_source_allowlist(
        paths, scene_frame_indices=metadata["scene_frame_indices"]
    )
    monkeypatch.setattr(
        occlusion_deriver,
        "_capture_source_bindings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("writer performed an independent binding pass")
        ),
    )
    monkeypatch.setattr(
        occlusion_deriver,
        "_capture_source_allowlist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("writer performed an independent allowlist pass")
        ),
    )

    manifest = write_occlusion_package(
        tmp_path / "output",
        arrays=arrays,
        metadata=metadata,
        source_paths=paths,
        captured_source_records=captured_records,
        captured_source_witnesses=captured_witnesses,
        status="FIXTURE",
    )

    assert manifest.is_file()


def test_generated_publish_requires_matching_frozen_contract(tmp_path: Path) -> None:
    frames, records = _fixture_inputs()
    arrays, metadata = derive_occlusion_targets(
        frames_by_scene=frames,
        dsg_records_by_scene=records,
        camera={
            "width": 1,
            "height": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        },
    )
    paths = _source_paths(tmp_path / "sources")
    indices = metadata["scene_frame_indices"]
    bound = validate_source_allowlist(paths, scene_frame_indices=indices)
    expected_files = {
        role: record for role, record in bound.items() if ".depth." not in role
    }
    expected_depth = {
        scene: depth_collection_binding(
            bound, scene=scene, frame_indices=indices[scene]
        )
        for scene in ("apartment", "office")
    }
    contract = tmp_path / "contract.json"
    contract.write_bytes(
        render_manifest(
            build_contract_manifest(
                source_records=expected_files,
                depth_bindings=expected_depth,
            )
        )
    )
    metadata["contract"] = _bound_file(contract)

    with pytest.raises(ValueError, match="frozen CONTRACT_ONLY"):
        write_occlusion_package(
            tmp_path / "missing-contract",
            arrays=arrays,
            metadata=metadata,
            source_paths=paths,
            expected_source_records=expected_files,
            expected_depth_bindings=expected_depth,
            status="GENERATED",
        )

    manifest = write_occlusion_package(
        tmp_path / "generated",
        arrays=arrays,
        metadata=metadata,
        source_paths=paths,
        expected_source_records=expected_files,
        expected_depth_bindings=expected_depth,
        contract_path=contract,
        status="GENERATED",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "GENERATED"
    assert payload["metadata"]["contract"] == _bound_file(contract)


def test_build_contract_is_input_only_and_detects_hash_drift(tmp_path: Path) -> None:
    paths = _source_paths(tmp_path)
    records = validate_source_allowlist(
        paths, scene_frame_indices={"apartment": range(4), "office": range(4)}
    )
    depth_bindings = {
        scene: depth_collection_binding(
            records, scene=scene, frame_indices=range(4)
        )
        for scene in ("apartment", "office")
    }
    payload = build_contract_manifest(source_records=records, depth_bindings=depth_bindings)

    assert payload["status"] == "CONTRACT_ONLY"
    assert payload["targets_generated"] is False
    assert payload["prediction_inputs_used"] is False
    assert "episodes" not in payload
    assert render_manifest(payload) == render_manifest(
        build_contract_manifest(source_records=records, depth_bindings=depth_bindings)
    )

    changed = dict(records)
    changed["source_manifest"] = dict(changed["source_manifest"])
    changed["source_manifest"]["sha256"] = "0" * 64
    assert build_contract_manifest(source_records=changed, depth_bindings=depth_bindings) != payload


def test_checked_contract_preregisters_inputs_and_does_not_claim_results() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert CONTRACT.read_bytes() == render_manifest(payload)
    assert payload["manifest_id"] == "tesse_cd_occlusion_v1"
    assert payload["status"] == "CONTRACT_ONLY"
    assert payload["targets_generated"] is False
    assert payload["dataset_root_id"] == "tesse_cd_official_root_v1"
    assert payload["prediction_inputs_used"] is False
    assert payload["parameters"]["depth_tolerance_m"] == 0.10
    assert payload["parameters"]["stress_thresholds"] == [0.50, 0.75, 0.90]
    assert payload["parameters"]["headline_stress_threshold"] == 0.90
    assert set(payload["input_roles"]) == {
        "source_manifest",
        "schedule",
        "rgbd_lock",
        "camera",
        "scene.changes",
        "scene.dsg_with_mesh",
        "scene.export_manifest",
        "scene.timestamps",
        "scene.trajectory",
        "scene.depth.NNNNNN",
    }
    serialized = CONTRACT.read_text(encoding="utf-8").lower()
    assert "/home/ww" not in serialized
    assert "snapshot" not in serialized
    assert "frontend" not in serialized
    assert "prediction" not in serialized.replace('"prediction_inputs_used": false', "")
    assert "frame000000.jpg" not in serialized
    rebuilt = build_contract_manifest(
        source_records=payload["source_records"],
        depth_bindings=payload["depth_collections"],
        dataset_root_id=payload["dataset_root_id"],
    )
    assert render_manifest(rebuilt) == CONTRACT.read_bytes()


def test_dataset_source_resolution_rejects_absolute_parent_and_symlink_escape(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    inside = dataset_root / "derived" / "camera.json"
    inside.parent.mkdir()
    inside.write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (dataset_root / "escape.json").symlink_to(outside)

    assert resolve_dataset_source("derived/camera.json", dataset_root) == inside.resolve()
    for unsafe in (str(outside.resolve()), "../outside.json", "escape.json"):
        with pytest.raises(ValueError, match="dataset-root"):
            resolve_dataset_source(unsafe, dataset_root)


@pytest.mark.parametrize("symlink_position", ["intermediate", "terminal"])
def test_dataset_source_resolution_rejects_every_relative_symlink_component(
    tmp_path: Path, symlink_position: str
) -> None:
    dataset_root = tmp_path / "dataset"
    actual = dataset_root / "actual"
    actual.mkdir(parents=True)
    source = actual / "camera.json"
    source.write_text("{}\n", encoding="utf-8")
    if symlink_position == "intermediate":
        (dataset_root / "logical").symlink_to(actual, target_is_directory=True)
        relative = "logical/camera.json"
    else:
        (dataset_root / "camera.json").symlink_to(source)
        relative = "camera.json"

    with pytest.raises(ValueError, match="symbolic link"):
        resolve_dataset_source(relative, dataset_root)


def test_dataset_source_rejects_retargetable_symlink_before_initial_read(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    first = dataset_root / "first"
    second = dataset_root / "second"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "camera.json").write_text('{"source": 1}\n', encoding="utf-8")
    (second / "camera.json").write_text('{"source": 2}\n', encoding="utf-8")
    logical = dataset_root / "logical"
    logical.symlink_to(first, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        path = resolve_dataset_source("logical/camera.json", dataset_root)
        _, witness = occlusion_deriver._read_source_bytes(
            path, serialized_path="logical/camera.json"
        )
        logical.unlink()
        logical.symlink_to(second, target_is_directory=True)
        occlusion_deriver._revalidate_source_witness(witness)


def test_dataset_source_witness_rejects_intermediate_redirect_after_read(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    logical = dataset_root / "logical"
    alternate = dataset_root / "alternate"
    logical.mkdir(parents=True)
    alternate.mkdir()
    (logical / "camera.json").write_text('{"source": 1}\n', encoding="utf-8")
    (alternate / "camera.json").write_text('{"source": 2}\n', encoding="utf-8")
    path = resolve_dataset_source("logical/camera.json", dataset_root)
    _, witness = occlusion_deriver._read_source_bytes(
        path, serialized_path="logical/camera.json"
    )

    logical.rename(dataset_root / "original")
    logical.symlink_to(alternate, target_is_directory=True)

    with pytest.raises(ValueError, match="source changed before publication"):
        occlusion_deriver._revalidate_source_witness(witness)


def test_deriver_cli_can_run_directly_from_repository_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/derive_tesse_cd_occlusion_v1.py",
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--contract" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--dataset-root" in completed.stdout
    assert "--preflight-only" in completed.stdout
