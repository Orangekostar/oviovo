from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.build_tesse_ovimap_static_anchor import (
    TesseNativeEnvironment,
)
from scripts.evaluation.evaluate_ovi_two_visit_current import (
    validate_evaluation_inputs,
)
from scripts.evaluation.freeze_tesse_two_visit_protocol import (
    OFFICE_HELD_OUT_STATUS,
    PROTOCOL_ID,
    ProtocolError,
    authorize_scene_run,
    candidate_visit_windows,
    compute_window_input_sha256,
    freeze_protocol,
    protocol_content_sha256,
)
from scripts.evaluation.run_ovi_two_visit import (
    _file_record,
    build_independent_ovi_plan,
    execute_independent_ovi_plan,
    materialize_visit_window,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _binding(name: str) -> dict[str, object]:
    data = name.encode("ascii")
    return {
        "path": f"/{name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _window(visit: str, start: int, end: int, seed: str) -> dict[str, object]:
    return {
        "visit_id": visit,
        "start_frame": start,
        "end_frame": end,
        "frame_count": end - start + 1,
        "source_frame_ids_sha256": hashlib.sha256(
            _canonical(list(range(start, end + 1)))
        ).hexdigest(),
        "input_sha256": hashlib.sha256(seed.encode("ascii")).hexdigest(),
    }


def _candidate(
    identifier: str,
    *,
    t0: tuple[int, int],
    t1: tuple[int, int],
    trajectory: float,
    observable: float,
    old_visibility: float,
    camera: float = 0.7,
) -> dict[str, object]:
    return {
        "candidate_id": identifier,
        "visits": {
            "t0": _window("t0", *t0, f"{identifier}-t0"),
            "t1": _window("t1", *t1, f"{identifier}-t1"),
        },
        "diagnostics": {
            "no_temporal_overlap": t0[1] < t1[0],
            "trajectory_overlap_fraction": trajectory,
            "common_observable_volume_fraction": observable,
            "camera_viewpoint_histogram_intersection": camera,
            "changed_object_count": 4,
            "changed_object_mass_voxels": 120,
            "old_location_visibility_fraction": old_visibility,
        },
        "evaluator_only": {
            "event_ids": ["event_01", "event_02"],
            "changed_object_count": 4,
            "changed_object_mass_voxels": 120,
            "old_location_visibility_fraction": old_visibility,
        },
    }


def _candidate_metadata() -> dict[str, object]:
    source_bindings = {
        "dataset_manifest": _binding("dataset.json"),
        "rgbd_lock": _binding("rgbd-lock.json"),
        "causal_schedule": _binding("schedule.json"),
        "common_v2_target_manifest": _binding("targets-manifest.json"),
        "common_v2_targets": _binding("targets.npz"),
    }
    scene_sources = {
        "rgbd_export_manifest": _binding("export.json"),
        "camera": _binding("camera.json"),
        "trajectory": _binding("traj.txt"),
        "timestamps": _binding("timestamps.csv"),
    }
    apartment_candidates = [
        _candidate(
            "apartment-late",
            t0=(7, 262),
            t1=(1117, 1372),
            trajectory=0.70,
            observable=0.75,
            old_visibility=0.80,
        ),
        _candidate(
            "apartment-best",
            t0=(7, 262),
            t1=(1167, 1422),
            trajectory=0.80,
            observable=0.85,
            old_visibility=0.90,
        ),
    ]
    office_candidates = [
        _candidate(
            "office-frozen",
            t0=(1745, 2000),
            t1=(3646, 3901),
            trajectory=0.65,
            observable=0.70,
            old_visibility=0.75,
        )
    ]
    return {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "selection_inputs": ["source_metadata", "causal_schedule"],
        "method_predictions_used": False,
        "selection_policy": {
            "window_frame_count": 256,
            "minimum_trajectory_overlap_fraction": 0.05,
            "minimum_common_observable_volume_fraction": 0.05,
            "minimum_camera_viewpoint_histogram_intersection": 0.05,
            "minimum_changed_object_count": 1,
            "minimum_changed_object_mass_voxels": 1,
            "minimum_old_location_visibility_fraction": 0.05,
            "selection_key": [
                "old_location_visibility_fraction",
                "common_observable_volume_fraction",
                "trajectory_overlap_fraction",
                "camera_viewpoint_histogram_intersection",
                "earliest_t1_end_frame",
            ],
            "trajectory_match_radius_m": 3.0,
            "observable_voxel_size_m": 0.25,
            "diagnostic_frame_stride": 16,
            "diagnostic_pixel_stride": 48,
            "diagnostic_ray_samples": 16,
            "maximum_depth_m": 10.0,
            "viewpoint_azimuth_bins": 16,
            "viewpoint_elevation_bins": 4,
        },
        "source_bindings": source_bindings,
        "scenes": {
            "apartment": {
                "role": "development",
                "source_bindings": scene_sources,
                "candidates": apartment_candidates,
            },
            "office": {
                "role": "held_out",
                "source_bindings": scene_sources,
                "candidates": office_candidates,
            },
        },
    }


def test_visit_windows_are_non_overlapping_source_bound_and_method_independent() -> None:
    protocol = freeze_protocol(_candidate_metadata())

    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["selection_inputs"] == ["source_metadata", "causal_schedule"]
    assert protocol["method_predictions_used"] is False
    assert "method_result_bindings" not in protocol
    selected = protocol["scenes"]["apartment"]
    assert selected["selected_candidate_id"] == "apartment-best"
    visits = selected["method_input_manifest"]["visits"]
    assert visits["t0"]["end_frame"] < visits["t1"]["start_frame"]
    assert set(selected["method_input_manifest"]["source_bindings"]) == {
        "rgbd_export_manifest",
        "camera",
        "trajectory",
        "timestamps",
    }
    assert "evaluator_only" not in selected["method_input_manifest"]


def test_checked_in_protocol_replays_exact_freeze_and_keeps_office_held_out() -> None:
    path = REPO_ROOT / "configs/evaluation/tesse_two_visit_current_v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        key: protocol[key]
        for key in (
            "schema_version",
            "dataset",
            "selection_inputs",
            "method_predictions_used",
            "selection_policy",
            "source_bindings",
        )
    }
    metadata["scenes"] = {
        scene: {
            "role": record["role"],
            "source_bindings": record["method_input_manifest"]["source_bindings"],
            "candidates": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "visits",
                        "diagnostics",
                        "evaluator_only",
                    )
                }
                for candidate in record["candidates"]
            ],
        }
        for scene, record in protocol["scenes"].items()
    }

    assert freeze_protocol(metadata) == protocol
    assert len(protocol["scenes"]["apartment"]["candidates"]) == 16
    assert len(protocol["scenes"]["office"]["candidates"]) == 16
    assert protocol["scenes"]["apartment"]["selected_candidate_id"] == (
        "apartment-t0-000766-001021-t1-001217-001472"
    )
    assert protocol["scenes"]["office"]["selected_candidate_id"] == (
        "office-t0-002145-002400-t1-002446-002701"
    )
    assert protocol["scenes"]["office"]["execution_status"] == OFFICE_HELD_OUT_STATUS
    for scene in ("apartment", "office"):
        method_input = protocol["scenes"][scene]["method_input_manifest"]
        assert method_input["runtime_ground_truth_inputs"] == []
        serialized = json.dumps(method_input, sort_keys=True).lower()
        assert "evaluator_only" not in serialized
        assert "method_result" not in serialized


def test_freeze_rejects_method_result_leakage_and_overlapping_windows() -> None:
    metadata = _candidate_metadata()
    metadata["scenes"]["apartment"]["candidates"][0]["ghost"] = 0.01
    with pytest.raises(ProtocolError, match="candidate fields"):
        freeze_protocol(metadata)

    protocol = freeze_protocol(_candidate_metadata())
    protocol["scenes"]["apartment"]["selected_candidate_id"] = "apartment-late"
    with pytest.raises(ProtocolError, match="replay"):
        authorize_scene_run(protocol, "apartment")

    metadata = _candidate_metadata()
    candidate = metadata["scenes"]["apartment"]["candidates"][0]
    candidate["visits"]["t1"] = _window("t1", 200, 455, "overlap")
    with pytest.raises(ProtocolError, match="overlap"):
        freeze_protocol(metadata)


def test_office_cannot_run_before_all_freeze_bindings_exist() -> None:
    protocol = freeze_protocol(_candidate_metadata())
    assert protocol["scenes"]["office"]["execution_status"] == OFFICE_HELD_OUT_STATUS

    with pytest.raises(ProtocolError, match=OFFICE_HELD_OUT_STATUS):
        authorize_scene_run(protocol, "office")

    incomplete_release = {
        "status": "OFFICE_RELEASE_AUTHORIZED",
        "protocol_content_sha256": protocol_content_sha256(protocol),
        "office_attempt_count": 0,
        "frozen_bindings": {},
    }
    with pytest.raises(ProtocolError, match="freeze bindings"):
        authorize_scene_run(protocol, "office", office_release=incomplete_release)


def test_candidate_windows_fix_t0_before_first_change_and_t1_after_last_change() -> None:
    windows = candidate_visit_windows(
        {
            "frame_count": 30,
            "events": [
                {
                    "event_id": "event_01",
                    "intervention_frame_index": 10,
                    "common_checkpoint_frame_indices": [10, 14, 18],
                },
                {
                    "event_id": "event_02",
                    "intervention_frame_index": 16,
                    "common_checkpoint_frame_indices": [16, 20, 24, 28],
                },
            ],
        },
        window_frame_count=4,
    )

    assert windows == (
        ((6, 9), (11, 14)),
        ((6, 9), (15, 18)),
        ((12, 15), (17, 20)),
        ((12, 15), (21, 24)),
        ((12, 15), (25, 28)),
    )


def _native(root: Path) -> TesseNativeEnvironment:
    return TesseNativeEnvironment(
        build_root=root / "native",
        frontend_python=root / "frontend/bin/python",
        mapping_python=root / "mapping/bin/python",
        entity_root=root / "native/Entity",
        ovimap_root=root / "native/OVI-MAP",
        cropformer_weights=root / "weights/cropformer.pth",
        siglip_model=root / "weights/siglip",
        source_hashes=root / "native/source-hashes.json",
    )


def test_two_ovi_visits_have_distinct_roots_ids_and_no_state_path_leakage(
    tmp_path: Path,
) -> None:
    protocol = freeze_protocol(_candidate_metadata())
    plan = build_independent_ovi_plan(
        protocol,
        scene="apartment",
        source_rgbd_root=tmp_path / "source-rgbd",
        run_root=tmp_path / "runs",
        native=_native(tmp_path),
    )

    assert plan.t0.run_id != plan.t1.run_id
    assert plan.t0.commands.attempt_root != plan.t1.commands.attempt_root
    assert plan.t0.materialized_rgbd_root != plan.t1.materialized_rgbd_root
    t0_root = str(plan.t0.commands.attempt_root)
    t1_argv = "\n".join(
        (
            *plan.t1.commands.geometry,
            *plan.t1.commands.frontend,
            *plan.t1.commands.mapping,
        )
    )
    assert t0_root not in t1_argv
    assert "/t0/" not in t1_argv


def _write_rgbd_fixture(root: Path, *, scene: str, frame_count: int) -> Path:
    scene_root = root / scene
    results = scene_root / "results"
    results.mkdir(parents=True)
    (root / "cam_params.json").write_bytes(b'{"camera":{}}\n')
    (scene_root / "export_manifest.json").write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "scene": scene,
                "frame_count": frame_count,
            }
        )
    )
    trajectory = []
    timestamps = ["frame_index,sensor_timestamp_ns,relative_timestamp_ns\n"]
    for index in range(frame_count):
        (results / f"frame{index:06d}.jpg").write_bytes(f"rgb-{index}".encode())
        (results / f"depth{index:06d}.png").write_bytes(f"depth-{index}".encode())
        trajectory.append(
            f"1 0 0 {index} 0 1 0 0 0 0 1 0 0 0 0 1\n"
        )
        timestamps.append(f"{index},{1000 + index},{index}\n")
    (scene_root / "traj.txt").write_text("".join(trajectory), encoding="utf-8")
    (scene_root / "timestamps.csv").write_text("".join(timestamps), encoding="utf-8")
    return root


def test_materialized_visit_is_reindexed_copied_and_source_hash_bound(
    tmp_path: Path,
) -> None:
    source = _write_rgbd_fixture(tmp_path / "source", scene="apartment", frame_count=5)
    input_hash = compute_window_input_sha256(
        source, scene="apartment", start_frame=2, end_frame=4
    )
    destination = tmp_path / "materialized"

    manifest = materialize_visit_window(
        source_rgbd_root=source,
        materialized_rgbd_root=destination,
        scene="apartment",
        visit_id="t1",
        start_frame=2,
        end_frame=4,
        expected_input_sha256=input_hash,
    )

    assert manifest.is_file()
    scene_root = destination / "apartment"
    assert (scene_root / "results/frame000000.jpg").read_bytes() == b"rgb-2"
    assert (scene_root / "results/depth000002.png").read_bytes() == b"depth-4"
    assert not (scene_root / "results/frame000003.jpg").exists()
    assert len((scene_root / "traj.txt").read_text().splitlines()) == 3
    rows = (scene_root / "timestamps.csv").read_text().splitlines()
    assert rows[1].startswith("0,1002,0")
    assert rows[3].startswith("2,1004,2")
    payload = json.loads(manifest.read_text())
    assert payload["source_frame_interval"] == [2, 4]
    assert payload["source_input_sha256"] == input_hash
    assert all(not path.is_symlink() for path in destination.rglob("*"))

    (source / "apartment/results/frame000003.jpg").write_bytes(b"changed")
    with pytest.raises(ProtocolError, match="input hash"):
        materialize_visit_window(
            source_rgbd_root=source,
            materialized_rgbd_root=tmp_path / "second",
            scene="apartment",
            visit_id="t1",
            start_frame=2,
            end_frame=4,
            expected_input_sha256=input_hash,
        )


def test_runner_file_bindings_reject_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}\n")
    link = tmp_path / "manifest.json"
    link.symlink_to(target)

    with pytest.raises(ProtocolError, match="regular non-symlink"):
        _file_record(link)


def test_execute_runs_exactly_two_independent_native_mappings(tmp_path: Path) -> None:
    source = _write_rgbd_fixture(
        tmp_path / "source", scene="apartment", frame_count=1423
    )
    native = _native(tmp_path)
    metadata = _candidate_metadata()
    for candidate in metadata["scenes"]["apartment"]["candidates"]:
        for visit in candidate["visits"].values():
            visit["input_sha256"] = compute_window_input_sha256(
                source,
                scene="apartment",
                start_frame=visit["start_frame"],
                end_frame=visit["end_frame"],
            )
    protocol = freeze_protocol(metadata)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(_canonical(protocol))
    plan = build_independent_ovi_plan(
        protocol,
        scene="apartment",
        source_rgbd_root=source,
        run_root=tmp_path / "runs",
        native=native,
    )
    calls: list[str] = []

    def fake_executor(*, scene, rgbd_root, commands, native):
        assert scene == "apartment"
        assert rgbd_root in {
            plan.t0.materialized_rgbd_root,
            plan.t1.materialized_rgbd_root,
        }
        calls.append(commands.attempt_root.name)
        commands.attempt_root.mkdir(parents=True)
        manifest = commands.attempt_root / "native_mapping_manifest.json"
        manifest.write_bytes(_canonical({"status": "PASS", "scene": scene}))
        return manifest

    output = tmp_path / "runs" / "two_visit_ovi_manifest.json"
    result = execute_independent_ovi_plan(
        plan,
        protocol_path=protocol_path,
        source_rgbd_root=source,
        native=native,
        output_manifest=output,
        native_executor=fake_executor,
    )

    assert result == output
    assert calls == [plan.t0.run_id, plan.t1.run_id]
    payload = json.loads(output.read_text())
    assert payload["status"] == "TWO_VISIT_OVI_PASS"
    assert set(payload["visits"]) == {"t0", "t1"}
    assert payload["visits"]["t0"]["native_manifest"]["sha256"]
    assert payload["visits"]["t1"]["native_manifest"]["sha256"]


def test_evaluator_validates_hashes_before_loading_predictions(tmp_path: Path) -> None:
    metadata = _candidate_metadata()
    rgbd = _write_rgbd_fixture(
        tmp_path / "rgbd", scene="apartment", frame_count=1423
    )
    for candidate in metadata["scenes"]["apartment"]["candidates"]:
        for visit in candidate["visits"].values():
            visit["input_sha256"] = compute_window_input_sha256(
                rgbd,
                scene="apartment",
                start_frame=visit["start_frame"],
                end_frame=visit["end_frame"],
            )
    source_files: list[Path] = []
    for role in sorted(metadata["source_bindings"]):
        path = tmp_path / "sources" / role
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{role}".encode())
        metadata["source_bindings"][role] = _record(path)
        source_files.append(path)
    apartment_source_paths = {
        "rgbd_export_manifest": rgbd / "apartment/export_manifest.json",
        "camera": rgbd / "cam_params.json",
        "trajectory": rgbd / "apartment/traj.txt",
        "timestamps": rgbd / "apartment/timestamps.csv",
    }
    metadata["scenes"]["apartment"]["source_bindings"] = {
        role: _record(path) for role, path in apartment_source_paths.items()
    }
    for scene in ("office",):
        for role in sorted(metadata["scenes"][scene]["source_bindings"]):
            path = tmp_path / "sources" / scene / role
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"source-{scene}-{role}".encode())
            metadata["scenes"][scene]["source_bindings"][role] = _record(path)
            source_files.append(path)
    protocol = freeze_protocol(metadata)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(_canonical(protocol))
    method_input_path = tmp_path / "method-input.json"
    method_input_path.write_bytes(
        _canonical(protocol["scenes"]["apartment"]["method_input_manifest"])
    )
    output = tmp_path / "current-map.npz"
    output.write_bytes(b"prediction")
    config = tmp_path / "method.json"
    config.write_bytes(b"{}\n")
    prediction_manifest = tmp_path / "prediction-manifest.json"
    prediction_manifest.write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "status": "PASS",
                "protocol_id": PROTOCOL_ID,
                "protocol_content_sha256": protocol_content_sha256(protocol),
                "scene": "apartment",
                "method_input_manifest": _record(method_input_path),
                "method_config": _record(config),
                "outputs": {"current_map": _record(output)},
            }
        )
    )

    called = False

    def load_prediction(_: Path) -> object:
        nonlocal called
        called = True
        return object()

    receipt = validate_evaluation_inputs(
        protocol_path=protocol_path,
        scene="apartment",
        method_input_manifest_path=method_input_path,
        prediction_manifest_path=prediction_manifest,
        prediction_loader=load_prediction,
    )
    assert called is True
    assert receipt["status"] == "EVALUATION_INPUTS_PASS"

    called = False
    selected_start = protocol["scenes"]["apartment"]["method_input_manifest"][
        "visits"
    ]["t0"]["start_frame"]
    selected_rgb = (
        rgbd / "apartment/results" / f"frame{selected_start:06d}.jpg"
    )
    selected_rgb.write_bytes(b"tampered-window")
    with pytest.raises(ProtocolError, match="visit input hash"):
        validate_evaluation_inputs(
            protocol_path=protocol_path,
            scene="apartment",
            method_input_manifest_path=method_input_path,
            prediction_manifest_path=prediction_manifest,
            prediction_loader=load_prediction,
        )
    assert called is False
    selected_rgb.write_bytes(f"rgb-{selected_start}".encode())

    called = False
    source_files[0].write_bytes(b"tampered-source")
    with pytest.raises(ProtocolError, match="source binding"):
        validate_evaluation_inputs(
            protocol_path=protocol_path,
            scene="apartment",
            method_input_manifest_path=method_input_path,
            prediction_manifest_path=prediction_manifest,
            prediction_loader=load_prediction,
        )
    assert called is False
    source_files[0].write_bytes(
        f"source-{source_files[0].name}".encode()
    )

    called = False
    output.write_bytes(b"tampered")
    with pytest.raises(ProtocolError, match="output binding"):
        validate_evaluation_inputs(
            protocol_path=protocol_path,
            scene="apartment",
            method_input_manifest_path=method_input_path,
            prediction_manifest_path=prediction_manifest,
            prediction_loader=load_prediction,
        )
    assert called is False
