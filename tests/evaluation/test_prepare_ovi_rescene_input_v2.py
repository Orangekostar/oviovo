from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.evaluation.prepare_ovi_rescene_input_v2 import (
    C2PreparationError,
    RecoveryResult,
    audit_recovered_input,
    build_recovered_input,
    capture_native_sampling,
    load_bound_surface_groups,
    load_depth_calibration_artifact,
    load_materialized_visit,
    load_static_input_artifact,
    main,
    measure_calibration_residuals,
    prepare_static_contract,
    publish_recovered_input,
    recover_model_support,
    resolve_frozen_static_sources,
    select_calibration_indices,
    select_depth_tolerance,
    write_depth_calibration_artifact,
    write_static_input_artifact,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.rescene_input_bridge import (
    AdapterGeometry,
    ModelCandidateMap,
    NativeSamplingMap,
    RecoveredModelSupport,
    SurfaceAttributeBundle,
    load_model_input_artifact,
)
from src.oviv2.two_visit_contracts import OviEntitySemanticEvidence, VisitMap


class _RecordedGridSample:
    def __init__(self, calls: list[dict[str, object]], **options: object) -> None:
        self._calls = calls
        self._options = options

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        adapter_indices = np.asarray(data["adapter_index"], dtype=np.int64)
        self._calls.append(
            {
                "options": self._options,
                "adapter_indices": adapter_indices.copy(),
            }
        )
        np.random.randint(0, 1000, size=5)
        if adapter_indices.tolist() == [0, 1, 2]:
            selected = np.asarray([1, 2], dtype=np.int64)
            inverse = np.asarray([0, 0, 1], dtype=np.int64)
            grid = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
        elif adapter_indices.tolist() == [3, 4]:
            selected = np.asarray([3], dtype=np.int64)
            inverse = np.asarray([0, 0], dtype=np.int64)
            grid = np.asarray([[0, 0, 0]], dtype=np.int64)
        else:  # pragma: no cover - a wrong visit split should fail loudly
            raise AssertionError("unexpected adapter visit split")
        return {
            "adapter_index": selected,
            "inverse": inverse,
            "grid_coord": grid,
        }


class _NativeWitnessGridSample:
    def __init__(self, **_options: object) -> None:
        pass

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        adapter_indices = np.asarray(data["adapter_index"], dtype=np.int64)
        if adapter_indices.tolist() == [0, 1, 2]:
            return {
                "adapter_index": np.asarray([2, 0], dtype=np.int64),
                "inverse": np.asarray([1, 1, 0], dtype=np.int64),
                "grid_coord": np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.int64),
            }
        return {
            "adapter_index": np.asarray([4], dtype=np.int64),
            "inverse": np.asarray([0, 0], dtype=np.int64),
            "grid_coord": np.asarray([[0, 0, 0]], dtype=np.int64),
        }


class _IdentityGridSample:
    def __init__(self, *, grid_size: float, **_options: object) -> None:
        self._grid_size = grid_size

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        coord = np.asarray(data["coord"], dtype=np.float64)
        grid = np.floor(coord / self._grid_size).astype(np.int64)
        grid -= grid.min(axis=0)
        return {
            "adapter_index": np.asarray(data["adapter_index"], dtype=np.int64),
            "inverse": np.arange(len(coord), dtype=np.int64),
            "grid_coord": grid,
        }


def test_native_capture_composes_visit_local_inverse_and_restores_rng(
    tmp_path: Path,
) -> None:
    source = tmp_path / "transform.py"
    source.write_bytes(b"pinned native grid sampler\n")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    calls: list[dict[str, object]] = []

    np.random.seed(917)
    state_before = np.random.get_state()
    sampling = capture_native_sampling(
        coordinates_xyzt=np.asarray(
            [
                [0.000, 0.0, 0.000, 0.0],
                [0.001, 0.0, 0.0, 0.0],
                [0.021, 0.0, 0.0, 0.0],
                [1.000, 0.0, 0.0, 1.0],
                [1.001, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        visit_ids=np.asarray([0, 0, 0, 1, 1], dtype=np.int8),
        shared_center_xyz=np.zeros(3, dtype=np.float64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_source_path=source,
        expected_sampler_source_sha256=source_hash,
        grid_sample_factory=lambda **options: _RecordedGridSample(calls, **options),
    )
    state_after = np.random.get_state()

    assert state_after[0] == state_before[0]
    assert np.array_equal(state_after[1], state_before[1])
    assert state_after[2:] == state_before[2:]
    assert np.array_equal(sampling.adapter_to_model, [0, 0, 1, 2, 2])
    assert np.array_equal(sampling.selected_adapter_indices, [1, 2, 3])
    assert np.array_equal(sampling.model_visit_ids, [0, 0, 1])
    assert np.array_equal(sampling.visit_model_offsets, [0, 2, 3])
    assert np.array_equal(sampling.model_grid_coordinates, [[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    assert np.array_equal(sampling.visit_grid_origins, [[0, 0, 0], [50, 0, 0]])
    assert sampling.sampler_source_sha256 == source_hash
    assert [call["adapter_indices"].tolist() for call in calls] == [[0, 1, 2], [3, 4]]
    assert all(
        call["options"]
        == {
            "grid_size": 0.02,
            "hash_type": "fnv",
            "mode": "train",
            "return_inverse": True,
            "return_grid_coord": True,
        }
        for call in calls
    )


def test_native_capture_rejects_sampler_source_mismatch_before_sampling(
    tmp_path: Path,
) -> None:
    source = tmp_path / "transform.py"
    source.write_bytes(b"unexpected source\n")

    with pytest.raises(C2PreparationError, match="sampler source SHA-256 mismatch"):
        capture_native_sampling(
            coordinates_xyzt=np.asarray(
                [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
            ),
            visit_ids=np.asarray([0, 1], dtype=np.int8),
            shared_center_xyz=np.zeros(3),
            voxel_size_m=0.02,
            sampler_seed=45,
            sampler_source_path=source,
            expected_sampler_source_sha256="0" * 64,
            grid_sample_factory=lambda **_options: pytest.fail(
                "sampler must not run after a source mismatch"
            ),
        )


def test_native_self_test_cli_publishes_exact_inverse_witness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "transform.py"
    source.write_bytes(b"pinned native grid sampler\n")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    exit_code = main(
        [
            "native-self-test",
            "--sampler-source",
            str(source),
            "--sampler-source-sha256",
            source_hash,
        ],
        grid_sample_factory=_NativeWitnessGridSample,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {
        "adapter_to_model": [1, 1, 0, 2, 2],
        "artifact_id": "OVI_RESCENE_NATIVE_GRID_SAMPLE_WITNESS_V2",
        "model_grid_coordinates": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        "sampler_seed": 45,
        "sampler_source_sha256": source_hash,
        "selected_adapter_indices": [2, 0, 4],
        "status": "PASS",
        "visit_model_offsets": [0, 2, 3],
    }


def _write_surface_ply(path: Path, color: tuple[int, int, int]) -> None:
    vertices = np.empty(
        3,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("normal_x", "f4"), ("normal_y", "f4"), ("normal_z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices[:] = [
        (9, 9, 9, 0, 0, 1, 0, 0, 0),
        (1, 2, 3, 2, 0, 0, *color),
        (4, 5, 6, 0, 3, 0, *color),
    ]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _surface_visit(visit_id: int, entity_id: str, instance_id: int) -> VisitMap:
    start = 10 * visit_id
    return VisitMap(
        visit_id=visit_id,
        snapshot=MapSnapshot(
            method="OVI-MAP",
            scene_id="apartment",
            timestamp=float(start + 1),
            entities=[
                EntityPrediction(
                    entity_id=entity_id,
                    points_xyz=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                    semantic_embedding=None,
                    semantic_label=None,
                    semantic_score=0.0,
                    lifecycle_state="current",
                    first_seen=float(start),
                    last_seen=float(start + 1),
                    metadata={"source_instance_id": instance_id},
                )
            ],
            background_xyz=np.empty((0, 3), dtype=np.float32),
            scope="current",
        ),
        coordinate_frame_id="world",
        source_manifest_sha256="a" * 64,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 1,
    )


def test_bound_surface_groups_follow_instance_log_and_original_ply_rows(
    tmp_path: Path,
) -> None:
    visits = (_surface_visit(0, "ovimap:1", 1), _surface_visit(1, "ovimap:2", 2))
    manifests: list[Path] = []
    for visit_id, color in enumerate(((7, 8, 9), (10, 11, 12))):
        mesh = tmp_path / f"t{visit_id}.ply"
        log = tmp_path / f"t{visit_id}.log"
        semantics = tmp_path / f"t{visit_id}.pkl"
        _write_surface_ply(mesh, color)
        log.write_text(
            f"INFO Instance: {visit_id + 1} Color: ({color[0]},{color[1]},{color[2]})\n",
            encoding="utf-8",
        )
        semantics.write_bytes(b"not read by the sidecar loader")
        manifest = tmp_path / f"t{visit_id}.json"
        manifest.write_text(
            json.dumps(
                {
                    "artifacts": {
                        "instance_mesh": _file_binding(mesh),
                        "instance_color_log": _file_binding(log),
                        "semantic_features": _file_binding(semantics),
                    }
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)

    groups = load_bound_surface_groups(visits, tuple(manifests))

    assert set(groups) == {(0, "ovimap:1"), (1, "ovimap:2")}
    assert groups[(0, "ovimap:1")].palette_rgb == (7, 8, 9)
    assert groups[(1, "ovimap:2")].palette_rgb == (10, 11, 12)
    assert groups[(0, "ovimap:1")].original_vertex_indices.tolist() == [1, 2]
    assert groups[(0, "ovimap:1")].normals_xyz.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_frozen_static_source_resolution_revalidates_every_required_record(
    tmp_path: Path,
) -> None:
    records: dict[str, dict[str, object]] = {}
    for name in ("protocol", "two_visit_ovi_manifest", "b0_output_manifest", "b2_output_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
        records[name] = _file_binding(path)
    config = tmp_path / "source-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "FROZEN_BEFORE_FIRST_B7_SCORE",
                "frozen_inputs": {"apartment": records},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sampler = tmp_path / "transform.py"
    sampler.write_bytes(b"sampler\n")
    sampler_hash = hashlib.sha256(sampler.read_bytes()).hexdigest()

    sources = resolve_frozen_static_sources(
        source_config_path=config,
        sampler_source_path=sampler,
        expected_sampler_source_sha256=sampler_hash,
    )

    assert sources.protocol_path == Path(records["protocol"]["path"])
    assert sources.two_visit_ovi_manifest_path == Path(
        records["two_visit_ovi_manifest"]["path"]
    )
    assert sources.b0_output_manifest_record == records["b0_output_manifest"]
    assert sources.b2_output_manifest_record == records["b2_output_manifest"]
    assert set(sources.input_bindings) == {
        "b0_output_manifest",
        "b2_output_manifest",
        "source_config",
        "sampler_source",
    }

    Path(records["b2_output_manifest"]["path"]).write_bytes(b"changed\n")
    with pytest.raises(C2PreparationError, match="b2_output_manifest binding mismatch"):
        resolve_frozen_static_sources(
            source_config_path=config,
            sampler_source_path=sampler,
            expected_sampler_source_sha256=sampler_hash,
        )


def test_prepare_static_contract_runs_complete_d_to_a_to_m_chain(
    tmp_path: Path,
) -> None:
    visits = (_surface_visit(0, "ovimap:1", 1), _surface_visit(1, "ovimap:2", 2))
    manifests: list[Path] = []
    for visit_id, color in enumerate(((7, 8, 9), (10, 11, 12))):
        mesh = tmp_path / f"prepare-t{visit_id}.ply"
        log = tmp_path / f"prepare-t{visit_id}.log"
        semantics = tmp_path / f"prepare-t{visit_id}.pkl"
        _write_surface_ply(mesh, color)
        log.write_text(
            f"Instance: {visit_id + 1} Color: ({color[0]},{color[1]},{color[2]})\n",
            encoding="utf-8",
        )
        semantics.write_bytes(b"semantic source\n")
        manifest = tmp_path / f"prepare-t{visit_id}.json"
        manifest.write_text(
            json.dumps(
                {
                    "artifacts": {
                        "instance_mesh": _file_binding(mesh),
                        "instance_color_log": _file_binding(log),
                        "semantic_features": _file_binding(semantics),
                    }
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)
    sampler = tmp_path / "sampler.py"
    sampler.write_bytes(b"identity sampler\n")
    parent = tmp_path / "parent.json"
    parent.write_bytes(b"frozen parent\n")
    output = tmp_path / "prepared-static"

    summary = prepare_static_contract(
        visits=visits,
        native_manifest_paths=tuple(manifests),
        sampler_source_path=sampler,
        expected_sampler_source_sha256=hashlib.sha256(sampler.read_bytes()).hexdigest(),
        input_bindings={"parent": _file_binding(parent)},
        output_root=output,
        neural_voxel_size_m=0.02,
        sampler_seed=45,
        maximum_candidates=8,
        grid_sample_factory=_IdentityGridSample,
    )
    loaded = load_static_input_artifact(output)

    assert summary == {
        "adapter_count": 4,
        "artifact_id": "OVI_RESCENE_STATIC_PREPARATION_SUMMARY_V2",
        "candidate_width": 8,
        "cross_entity_model_token_count": 0,
        "merge_count": 0,
        "model_count": 4,
        "source_point_count": 4,
        "status": "PASS",
    }
    assert loaded.geometry.adapter_count == 4
    assert loaded.surface.normal_valid.tolist() == [True, True, True, True]
    assert loaded.sampling.adapter_to_model.tolist() == [0, 1, 2, 3]
    assert loaded.candidates.candidate_counts.tolist() == [1, 1, 1, 1]


def test_calibration_indices_are_evenly_spaced_within_each_visit() -> None:
    selected = select_calibration_indices(
        np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8),
        sample_count_per_visit=3,
    )

    assert np.array_equal(selected[0], [0, 2, 4])
    assert np.array_equal(selected[1], [5, 7, 9])


def test_depth_tolerance_uses_per_visit_q99_millimetre_ceiling_and_maximum() -> None:
    decision = select_depth_tolerance(
        {
            0: np.asarray([0.001, 0.0091, 0.0101, 0.081]),
            1: np.asarray([0.002, 0.0201, 0.0491, np.nan]),
        },
        minimum_inlier_count=3,
        residual_ceiling_m=0.08,
        minimum_tolerance_m=0.01,
        maximum_tolerance_m=0.05,
    )

    assert decision.inlier_counts == (3, 3)
    assert decision.visit_q99_m == pytest.approx((0.0101, 0.0491))
    assert decision.visit_tolerances_m == pytest.approx((0.011, 0.05))
    assert decision.selected_tolerance_m == pytest.approx(0.05)


def test_depth_tolerance_fails_closed_when_one_visit_lacks_inliers() -> None:
    with pytest.raises(C2PreparationError, match="visit 1 has 1 calibration inliers"):
        select_depth_tolerance(
            {0: np.asarray([0.01, 0.02]), 1: np.asarray([0.01, 0.2])},
            minimum_inlier_count=2,
        )


def _file_binding(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _static_contract() -> tuple[
    AdapterGeometry,
    SurfaceAttributeBundle,
    NativeSamplingMap,
    ModelCandidateMap,
]:
    geometry = AdapterGeometry(
        coordinates_xyzt=np.asarray(
            [
                [0.000, 0.0, 2.000, 0.0],
                [0.001, 0.0, 2.000, 0.0],
                [1.000, 0.0, 2.000, 1.0],
                [1.021, 0.0, 2.000, 1.0],
            ],
            dtype=np.float64,
        ),
        visit_ids=np.asarray([0, 0, 1, 1], dtype=np.int8),
        source_visit_ids=np.asarray([0, 0, 1, 1], dtype=np.int8),
        source_entity_indices=np.asarray([0, 0, 1, 1], dtype=np.int32),
        source_entity_local_indices=np.asarray([0, 1, 0, 1], dtype=np.int64),
        source_point_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        source_to_adapter_offsets=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        entity_keys=((0, "a"), (1, "b")),
        entity_semantics=(
            OviEntitySemanticEvidence(0, "a", "chair", 0.9, np.asarray([1.0, 0.0])),
            OviEntitySemanticEvidence(1, "b", "table", 0.8, None),
        ),
        neural_voxel_size_m=0.02,
        coordinate_frame_id="world",
        source_manifest_sha256="1" * 64,
        source_visit_map_sha256=("2" * 64, "3" * 64),
        shared_center_xyz=np.zeros(3),
    )
    surface = SurfaceAttributeBundle(
        points_xyz=geometry.coordinates_xyzt[:, :3].astype(np.float32),
        normals_xyz=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)),
        normal_valid=np.ones(4, dtype=np.bool_),
        original_vertex_indices=np.asarray([10, 11, 20, 21], dtype=np.int64),
        source_visit_ids=np.asarray([0, 0, 1, 1], dtype=np.int8),
        source_entity_indices=np.asarray([0, 0, 1, 1], dtype=np.int32),
        adapter_geometry_sha256=geometry.content_sha256(),
    )
    sampling = NativeSamplingMap(
        adapter_to_model=np.asarray([0, 0, 1, 2], dtype=np.int64),
        selected_adapter_indices=np.asarray([0, 2, 3], dtype=np.int64),
        model_grid_coordinates=np.asarray(
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int64
        ),
        model_visit_ids=np.asarray([0, 1, 1], dtype=np.int8),
        visit_model_offsets=np.asarray([0, 1, 3], dtype=np.int64),
        visit_grid_origins=np.asarray([[0, 0, 100], [50, 0, 100]], dtype=np.int64),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_mode="train",
        sampler_source_sha256="4" * 64,
    )
    candidates = ModelCandidateMap(
        source_point_indices=np.asarray([[0, 1], [2, -1], [3, -1]], dtype=np.int64),
        candidate_counts=np.asarray([2, 1, 1], dtype=np.int16),
        maximum_candidates=2,
    )
    return geometry, surface, sampling, candidates


def test_static_artifact_roundtrip_is_hash_bound_atomic_and_no_clobber(
    tmp_path: Path,
) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent.json"
    parent.write_bytes(b'{"frozen":true}\n')
    output = tmp_path / "prepared"

    paths = write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=output,
    )
    loaded = load_static_input_artifact(output)

    assert paths.root == output.absolute()
    assert paths.manifest == output.absolute() / "manifest.json"
    assert loaded.geometry.content_sha256() == geometry.content_sha256()
    assert loaded.surface.content_sha256() == surface.content_sha256()
    assert loaded.sampling.content_sha256() == sampling.content_sha256()
    assert loaded.candidates.content_sha256() == candidates.content_sha256()
    assert loaded.input_bindings == {"parent": _file_binding(parent)}
    assert sorted(path.name for path in output.glob("*.npy")) == [
        "adapter_to_model.npy",
        "candidate_counts.npy",
        "candidate_source_point_indices.npy",
        "coordinates_xyzt.npy",
        "model_grid_coordinates.npy",
        "model_visit_ids.npy",
        "normal_valid.npy",
        "normals_xyz.npy",
        "original_vertex_indices.npy",
        "selected_adapter_indices.npy",
        "shared_center_xyz.npy",
        "source_entity_indices.npy",
        "source_entity_local_indices.npy",
        "source_point_indices.npy",
        "source_points_xyz.npy",
        "source_to_adapter_offsets.npy",
        "source_visit_ids.npy",
        "surface_source_entity_indices.npy",
        "surface_source_visit_ids.npy",
        "visit_grid_origins.npy",
        "visit_ids.npy",
        "visit_model_offsets.npy",
    ]
    with pytest.raises(C2PreparationError, match="already exists"):
        write_static_input_artifact(
            geometry=geometry,
            surface=surface,
            sampling=sampling,
            candidates=candidates,
            input_bindings={"parent": _file_binding(parent)},
            output_root=output,
        )


def test_static_artifact_rejects_changed_parent_binding(tmp_path: Path) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent.json"
    parent.write_bytes(b"original\n")
    output = tmp_path / "prepared"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=output,
    )

    parent.write_bytes(b"changed\n")

    with pytest.raises(C2PreparationError, match="input binding mismatch: parent"):
        load_static_input_artifact(output)


def _relative_binding(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _write_materialized_visit(
    root: Path,
    *,
    visit_id: int,
    global_start: int,
    rgb: tuple[int, int, int],
    depth_mm: int,
) -> Path:
    visit_root = root / f"t{visit_id}"
    scene_root = visit_root / "apartment"
    results = scene_root / "results"
    results.mkdir(parents=True)
    camera_path = visit_root / "cam_params.json"
    camera_path.write_text(
        json.dumps(
            {
                "camera": {
                    "cx": 360.0,
                    "cy": 240.0,
                    "fx": 415.69219381653056,
                    "fy": 415.69219381653056,
                    "h": 480,
                    "scale": 1000.0,
                    "w": 720,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory_path = scene_root / "traj.txt"
    trajectory_path.write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n", encoding="utf-8"
    )
    timestamps_path = scene_root / "timestamps.csv"
    timestamps_path.write_text(
        "frame_index,sensor_timestamp_ns,relative_timestamp_ns\n"
        "0,1000000000,0\n",
        encoding="utf-8",
    )
    rgb_path = results / "frame000000.jpg"
    Image.fromarray(np.full((480, 720, 3), rgb, dtype=np.uint8)).save(
        rgb_path, format="PNG"
    )
    depth_path = results / "depth000000.png"
    Image.fromarray(np.full((480, 720), depth_mm, dtype=np.uint16)).save(depth_path)
    manifest = {
        "schema_version": 1,
        "status": "MATERIALIZED_INPUT_PASS",
        "dataset": "TESSE-CD",
        "scene": "apartment",
        "visit_id": f"t{visit_id}",
        "frame_count": 1,
        "source_frame_interval": [global_start, global_start],
        "source_input_sha256": f"{visit_id + 5:x}" * 64,
        "source_bindings": {
            "rgb/000000": _file_binding(rgb_path),
            "depth/000000": _file_binding(depth_path),
        },
        "outputs": {
            "camera": _relative_binding(camera_path, visit_root),
            "trajectory": _relative_binding(trajectory_path, visit_root),
            "timestamps": _relative_binding(timestamps_path, visit_root),
        },
    }
    manifest_path = scene_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return manifest_path


def test_materialized_visit_loader_binds_local_and_global_frames(tmp_path: Path) -> None:
    manifest = _write_materialized_visit(
        tmp_path, visit_id=0, global_start=766, rgb=(10, 20, 30), depth_mm=2000
    )

    window = load_materialized_visit(
        manifest_path=manifest,
        expected_visit_id=0,
        expected_global_start=766,
        expected_global_end=766,
        expected_source_input_sha256="5" * 64,
    )
    frame = window.read_frame(0)

    assert window.visit_id == 0
    assert window.global_frame_indices == (766,)
    assert frame.frame_id == 0
    assert frame.source_frame_id == 766
    assert frame.rgb[240, 360].tolist() == [10, 20, 30]
    assert frame.depth[240, 360] == pytest.approx(2.0)


def test_recovery_uses_first_supported_candidate_and_same_visit_rgb(tmp_path: Path) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    points = surface.points_xyz.copy()
    points[0] = [0.0, 0.0, 2.009]
    surface = SurfaceAttributeBundle(
        points_xyz=points,
        normals_xyz=surface.normals_xyz,
        normal_valid=surface.normal_valid,
        original_vertex_indices=surface.original_vertex_indices,
        source_visit_ids=surface.source_visit_ids,
        source_entity_indices=surface.source_entity_indices,
        adapter_geometry_sha256=geometry.content_sha256(),
    )
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    artifact = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=artifact,
    )
    prepared = load_static_input_artifact(artifact)
    t0 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=0, global_start=766,
            rgb=(10, 20, 30), depth_mm=2000,
        ),
        expected_visit_id=0,
        expected_global_start=766,
        expected_global_end=766,
        expected_source_input_sha256="5" * 64,
    )
    t1 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=1, global_start=1217,
            rgb=(40, 50, 60), depth_mm=2000,
        ),
        expected_visit_id=1,
        expected_global_start=1217,
        expected_global_end=1217,
        expected_source_input_sha256="6" * 64,
    )

    result = recover_model_support(
        prepared,
        {0: t0, 1: t1},
        depth_tolerance_m=0.005,
    )

    assert result.status == "C2_V2_PASS"
    assert result.unsupported_model_indices.tolist() == []
    assert result.support.representative_source_point_indices.tolist() == [1, 2, 3]
    assert result.support.rgb_uint8.tolist() == [[10, 20, 30], [40, 50, 60], [40, 50, 60]]
    assert result.support.local_frame_indices.tolist() == [0, 0, 0]
    assert result.support.global_frame_indices.tolist() == [766, 1217, 1217]

    with pytest.raises(C2PreparationError, match="visit window mapping is inconsistent"):
        recover_model_support(
            prepared,
            {0: t1, 1: t0},
            depth_tolerance_m=0.005,
        )


def test_recovery_reports_partial_scope_without_fabricated_support(tmp_path: Path) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    artifact = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=artifact,
    )
    prepared = load_static_input_artifact(artifact)
    t0 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=0, global_start=766,
            rgb=(10, 20, 30), depth_mm=2000,
        ),
        expected_visit_id=0,
        expected_global_start=766,
        expected_global_end=766,
        expected_source_input_sha256="5" * 64,
    )
    t1 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=1, global_start=1217,
            rgb=(40, 50, 60), depth_mm=0,
        ),
        expected_visit_id=1,
        expected_global_start=1217,
        expected_global_end=1217,
        expected_source_input_sha256="6" * 64,
    )

    result = recover_model_support(prepared, {0: t0, 1: t1}, depth_tolerance_m=0.02)

    assert result.status == "PARTIAL_INPUT_SCOPE"
    assert result.unsupported_model_indices.tolist() == [1, 2]
    assert result.support.support_valid.tolist() == [True, False, False]
    assert result.support.representative_source_point_indices.tolist() == [0, -1, -1]
    assert np.isnan(result.support.depth_residual_m[1:]).all()


def test_calibration_measures_minimum_same_visit_residual_for_first_candidate(
    tmp_path: Path,
) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    artifact = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=artifact,
    )
    prepared = load_static_input_artifact(artifact)
    t0 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=0, global_start=766,
            rgb=(10, 20, 30), depth_mm=2000,
        ),
        expected_visit_id=0,
        expected_global_start=766,
        expected_global_end=766,
        expected_source_input_sha256="5" * 64,
    )
    t1 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=1, global_start=1217,
            rgb=(40, 50, 60), depth_mm=2001,
        ),
        expected_visit_id=1,
        expected_global_start=1217,
        expected_global_end=1217,
        expected_source_input_sha256="6" * 64,
    )

    residuals = measure_calibration_residuals(
        prepared,
        {0: t0, 1: t1},
        calibration_indices={
            0: np.asarray([0], dtype=np.int64),
            1: np.asarray([1, 2], dtype=np.int64),
        },
        residual_ceiling_m=0.08,
    )

    assert residuals[0].tolist() == pytest.approx([0.0])
    assert residuals[1].tolist() == pytest.approx([0.001, 0.001], abs=1e-6)


def test_calibration_records_missing_valid_normal_candidate_as_nan(tmp_path: Path) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    candidate_rows = candidates.source_point_indices.copy()
    candidate_counts = candidates.candidate_counts.copy()
    candidate_rows[2] = -1
    candidate_counts[2] = 0
    candidates = ModelCandidateMap(
        source_point_indices=candidate_rows,
        candidate_counts=candidate_counts,
        maximum_candidates=candidates.maximum_candidates,
    )
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    artifact = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=artifact,
    )
    prepared = load_static_input_artifact(artifact)
    t0 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=0, global_start=766,
            rgb=(10, 20, 30), depth_mm=2000,
        ),
        expected_visit_id=0,
        expected_global_start=766,
        expected_global_end=766,
        expected_source_input_sha256="5" * 64,
    )
    t1 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=1, global_start=1217,
            rgb=(40, 50, 60), depth_mm=2000,
        ),
        expected_visit_id=1,
        expected_global_start=1217,
        expected_global_end=1217,
        expected_source_input_sha256="6" * 64,
    )

    residuals = measure_calibration_residuals(
        prepared,
        {0: t0, 1: t1},
        calibration_indices={
            0: np.asarray([0], dtype=np.int64),
            1: np.asarray([1, 2], dtype=np.int64),
        },
    )

    assert residuals[0].tolist() == pytest.approx([0.0])
    assert residuals[1][0] == pytest.approx(0.0)
    assert np.isnan(residuals[1][1])


def test_depth_calibration_artifact_is_parent_bound_and_no_clobber(
    tmp_path: Path,
) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    static_root = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=static_root,
    )
    t0 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=0, global_start=766,
            rgb=(10, 20, 30), depth_mm=2000,
        ),
        expected_visit_id=0,
        expected_global_start=766,
        expected_global_end=766,
        expected_source_input_sha256="5" * 64,
    )
    t1 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=1, global_start=1217,
            rgb=(40, 50, 60), depth_mm=2000,
        ),
        expected_visit_id=1,
        expected_global_start=1217,
        expected_global_end=1217,
        expected_source_input_sha256="6" * 64,
    )
    indices = {
        0: np.asarray([0], dtype=np.int64),
        1: np.asarray([1, 2], dtype=np.int64),
    }
    residuals = {
        0: np.asarray([0.0091], dtype=np.float64),
        1: np.asarray([0.0201, 0.0491], dtype=np.float64),
    }
    decision = select_depth_tolerance(residuals, minimum_inlier_count=1)
    output = tmp_path / "calibration"

    paths = write_depth_calibration_artifact(
        static_input_root=static_root,
        prepared=load_static_input_artifact(static_root),
        visit_windows={0: t0, 1: t1},
        calibration_indices=indices,
        residuals_by_visit=residuals,
        decision=decision,
        output_root=output,
        minimum_inlier_count=1,
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == "OVI_RESCENE_DEPTH_CALIBRATION_V2"
    assert manifest["status"] == "PASS"
    assert manifest["sample_count_per_visit"] == {"t0": 1, "t1": 2}
    assert manifest["selected_tolerance_m"] == pytest.approx(0.05)
    assert manifest["static_sampling_sha256"] == sampling.content_sha256()
    assert set(manifest["visit_manifests"]) == {"t0", "t1"}
    with np.load(paths.arrays, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "t0_min_depth_residual_m",
            "t0_model_indices",
            "t1_min_depth_residual_m",
            "t1_model_indices",
        }
        assert arrays["t1_model_indices"].tolist() == [1, 2]
    with pytest.raises(C2PreparationError, match="already exists"):
        write_depth_calibration_artifact(
            static_input_root=static_root,
            prepared=load_static_input_artifact(static_root),
            visit_windows={0: t0, 1: t1},
            calibration_indices=indices,
            residuals_by_visit=residuals,
            decision=decision,
            output_root=output,
            minimum_inlier_count=1,
        )
    loaded = load_depth_calibration_artifact(
        output,
        static_input_root=static_root,
        prepared=load_static_input_artifact(static_root),
    )
    assert loaded.decision == decision
    assert loaded.calibration_indices[0].tolist() == [0]
    assert loaded.residuals_by_visit[1].tolist() == pytest.approx([0.0201, 0.0491])

    paths.arrays.write_bytes(paths.arrays.read_bytes() + b"tampered")
    with pytest.raises(C2PreparationError, match="calibration arrays binding mismatch"):
        load_depth_calibration_artifact(
            output,
            static_input_root=static_root,
            prepared=load_static_input_artifact(static_root),
        )


def test_complete_recovery_publishes_model_input_and_compatible_adapter_pair(
    tmp_path: Path,
) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    static_root = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=static_root,
    )
    prepared = load_static_input_artifact(static_root)
    support = RecoveredModelSupport(
        support_valid=np.ones(3, dtype=np.bool_),
        representative_source_point_indices=np.asarray([0, 2, 3], dtype=np.int64),
        rgb_uint8=np.asarray([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8),
        local_frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
        global_frame_indices=np.asarray([766, 1218, 1219], dtype=np.int64),
        rows=np.asarray([1, 2, 3], dtype=np.int64),
        columns=np.asarray([4, 5, 6], dtype=np.int64),
        camera_depth_m=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        observed_depth_m=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        depth_residual_m=np.zeros(3, dtype=np.float32),
    )
    recovery = RecoveryResult(
        status="C2_V2_PASS",
        support=support,
        unsupported_model_indices=np.empty(0, dtype=np.int64),
    )
    calibration_manifest = tmp_path / "calibration.json"
    calibration_manifest.write_bytes(b'{"status":"PASS"}\n')
    output = tmp_path / "built-input"

    receipt_path = publish_recovered_input(
        static_input_root=static_root,
        calibration_manifest_path=calibration_manifest,
        prepared=prepared,
        recovery=recovery,
        depth_tolerance_m=0.02,
        output_root=output,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    model_input = load_model_input_artifact(output / "model_input")
    pair = load_neural_sample_artifact(output / "adapter_pair")

    assert receipt["status"] == "C2_V2_PASS"
    assert receipt["model_count"] == 3
    assert receipt["adapter_count"] == 4
    assert receipt["attribute_coverage"]["model_rgb_fraction"] == 1.0
    assert receipt["attribute_coverage"]["model_normal_fraction"] == 1.0
    assert model_input.adapter_to_model.tolist() == [0, 0, 1, 2]
    assert model_input.features[0, 3:6].tolist() == pytest.approx(
        np.asarray([10, 20, 30]) / 255.0
    )
    assert pair.coordinates_xyzt.tolist() == geometry.coordinates_xyzt.tolist()
    assert pair.features.shape == (4, 6)
    assert pair.features[0].tolist() == pytest.approx(pair.features[1].tolist())

    audited = audit_recovered_input(
        output_root=output,
        static_input_root=static_root,
        calibration_manifest_path=calibration_manifest,
    )
    assert audited == receipt


def test_partial_recovery_publishes_only_explicit_support_evidence(tmp_path: Path) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    static_root = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=static_root,
    )
    prepared = load_static_input_artifact(static_root)
    support = RecoveredModelSupport(
        support_valid=np.asarray([True, False, False]),
        representative_source_point_indices=np.asarray([0, -1, -1]),
        rgb_uint8=np.asarray([[10, 20, 30], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
        local_frame_indices=np.asarray([0, -1, -1]),
        global_frame_indices=np.asarray([766, -1, -1]),
        rows=np.asarray([1, -1, -1]),
        columns=np.asarray([4, -1, -1]),
        camera_depth_m=np.asarray([2.0, np.nan, np.nan], dtype=np.float32),
        observed_depth_m=np.asarray([2.0, np.nan, np.nan], dtype=np.float32),
        depth_residual_m=np.asarray([0.0, np.nan, np.nan], dtype=np.float32),
    )
    calibration_manifest = tmp_path / "calibration.json"
    calibration_manifest.write_bytes(b'{"status":"PASS"}\n')
    output = tmp_path / "partial-input"

    receipt_path = publish_recovered_input(
        static_input_root=static_root,
        calibration_manifest_path=calibration_manifest,
        prepared=prepared,
        recovery=RecoveryResult(
            status="PARTIAL_INPUT_SCOPE",
            support=support,
            unsupported_model_indices=np.asarray([1, 2]),
        ),
        depth_tolerance_m=0.02,
        output_root=output,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "PARTIAL_INPUT_SCOPE"
    assert receipt["unsupported_model_count"] == 2
    assert receipt["model_input_manifest"] is None
    assert receipt["adapter_pair_manifest"] is None
    assert not (output / "model_input").exists()
    assert not (output / "adapter_pair").exists()
    assert audit_recovered_input(
        output_root=output,
        static_input_root=static_root,
        calibration_manifest_path=calibration_manifest,
    ) == receipt

    support_path = output / "recovered_support.npz"
    support_path.write_bytes(support_path.read_bytes() + b"tampered")
    with pytest.raises(C2PreparationError, match="recovered support binding mismatch"):
        audit_recovered_input(
            output_root=output,
            static_input_root=static_root,
            calibration_manifest_path=calibration_manifest,
        )


def test_build_recovered_input_uses_bound_calibration_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent"
    parent.write_bytes(b"source\n")
    static_root = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=static_root,
    )
    prepared = load_static_input_artifact(static_root)
    t0 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=0, global_start=766,
            rgb=(10, 20, 30), depth_mm=2000,
        ),
        expected_visit_id=0,
        expected_global_start=766,
        expected_global_end=766,
        expected_source_input_sha256="5" * 64,
    )
    t1 = load_materialized_visit(
        manifest_path=_write_materialized_visit(
            tmp_path / "frames", visit_id=1, global_start=1217,
            rgb=(40, 50, 60), depth_mm=2000,
        ),
        expected_visit_id=1,
        expected_global_start=1217,
        expected_global_end=1217,
        expected_source_input_sha256="6" * 64,
    )
    decision = select_depth_tolerance(
        {0: np.asarray([0.0191]), 1: np.asarray([0.0091])},
        minimum_inlier_count=1,
    )
    calibration_root = tmp_path / "calibration"
    write_depth_calibration_artifact(
        static_input_root=static_root,
        prepared=prepared,
        visit_windows={0: t0, 1: t1},
        calibration_indices={0: np.asarray([0]), 1: np.asarray([1])},
        residuals_by_visit={
            0: np.asarray([0.0191]),
            1: np.asarray([0.0091]),
        },
        decision=decision,
        output_root=calibration_root,
        minimum_inlier_count=1,
    )
    monkeypatch.setattr(
        "scripts.evaluation.prepare_ovi_rescene_input_v2.load_bound_visit_windows",
        lambda _prepared: {0: t0, 1: t1},
    )

    receipt = build_recovered_input(
        static_input_root=static_root,
        calibration_root=calibration_root,
        output_root=tmp_path / "built",
    )

    assert receipt["status"] == "C2_V2_PASS"
    assert receipt["depth_tolerance_m"] == pytest.approx(0.02)
