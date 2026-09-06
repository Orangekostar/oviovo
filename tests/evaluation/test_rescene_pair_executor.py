from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.prepare_ovi_rescene_input_v2 import (
    RecoveryResult,
    load_static_input_artifact,
    publish_recovered_input,
    write_static_input_artifact,
)
from scripts.evaluation.prepare_ovi_rescene_supported_v3 import build_supported_input
from scripts.evaluation.rescene_pair_executor import (
    ExecutorError,
    NativeForwardResult,
    extract_model_state_dict,
    postprocess_native_predictions,
    run_rescene_pair_executor,
)
from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.rescene_input_bridge import (
    AdapterGeometry,
    ModelCandidateMap,
    NativeSamplingMap,
    RecoveredModelSupport,
    SurfaceAttributeBundle,
    build_model_input,
    load_model_input_artifact,
    materialize_neural_sample_map,
)
from src.oviv2.two_visit_contracts import OviEntitySemanticEvidence
from tests.evaluation.test_prepare_ovi_rescene_supported_v3 import _partial_v2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.absolute()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _input_sidecar(tmp_path: Path) -> tuple[Path, object, object]:
    geometry = AdapterGeometry(
        coordinates_xyzt=np.asarray(
            [
                [0.000, 0.0, 2.0, 0.0],
                [0.001, 0.0, 2.0, 0.0],
                [1.000, 0.0, 2.0, 1.0],
                [1.021, 0.0, 2.0, 1.0],
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
            OviEntitySemanticEvidence(0, "a", "chair", 0.9, None),
            OviEntitySemanticEvidence(1, "b", "table", 0.8, None),
        ),
        neural_voxel_size_m=0.02,
        coordinate_frame_id="world",
        source_manifest_sha256="1" * 64,
        source_visit_map_sha256=("2" * 64, "3" * 64),
        shared_center_xyz=np.zeros(3),
    )
    surface = SurfaceAttributeBundle(
        points_xyz=geometry.coordinates_xyzt[:, :3],
        normals_xyz=np.tile(np.asarray([[0.0, 0.0, 1.0]]), (4, 1)),
        normal_valid=np.ones(4, dtype=np.bool_),
        original_vertex_indices=np.asarray([10, 11, 20, 21]),
        source_visit_ids=geometry.source_visit_ids,
        source_entity_indices=geometry.source_entity_indices,
        adapter_geometry_sha256=geometry.content_sha256(),
    )
    sampling = NativeSamplingMap(
        adapter_to_model=np.asarray([0, 0, 1, 2]),
        selected_adapter_indices=np.asarray([0, 2, 3]),
        model_grid_coordinates=np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        model_visit_ids=np.asarray([0, 1, 1]),
        visit_model_offsets=np.asarray([0, 1, 3]),
        visit_grid_origins=np.asarray([[0, 0, 100], [50, 0, 100]]),
        voxel_size_m=0.02,
        sampler_seed=45,
        sampler_mode="train",
        sampler_source_sha256="4" * 64,
    )
    candidates = ModelCandidateMap(
        source_point_indices=np.asarray([[0, 1], [2, -1], [3, -1]]),
        candidate_counts=np.asarray([2, 1, 1]),
        maximum_candidates=2,
    )
    support = RecoveredModelSupport(
        support_valid=np.ones(3, dtype=np.bool_),
        representative_source_point_indices=np.asarray([0, 2, 3]),
        rgb_uint8=np.asarray([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
        local_frame_indices=np.asarray([0, 1, 2]),
        global_frame_indices=np.asarray([766, 1218, 1219]),
        rows=np.asarray([1, 2, 3]),
        columns=np.asarray([4, 5, 6]),
        camera_depth_m=np.asarray([2.0, 2.0, 2.0]),
        observed_depth_m=np.asarray([2.0, 2.0, 2.0]),
        depth_residual_m=np.asarray([0.0, 0.0, 0.0]),
    )
    parent = tmp_path / "parent"
    parent.write_bytes(b"parent\n")
    static_root = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_record(parent)},
        output_root=static_root,
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"status":"PASS"}\n', encoding="utf-8")
    sidecar = tmp_path / "input-sidecar"
    publish_recovered_input(
        static_input_root=static_root,
        calibration_manifest_path=calibration,
        prepared=load_static_input_artifact(static_root),
        recovery=RecoveryResult(
            status="C2_V2_PASS",
            support=support,
            unsupported_model_indices=np.empty(0, dtype=np.int64),
        ),
        depth_tolerance_m=0.02,
        output_root=sidecar,
    )
    model_input = build_model_input(geometry, sampling, surface, candidates, support)
    pair = materialize_neural_sample_map(geometry, model_input)
    return sidecar, model_input, pair


def _pair_arrays(path: Path, pair: object) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            coordinates_xyzt=pair.coordinates_xyzt,
            features=pair.features,
            visit_ids=pair.visit_ids,
            source_visit_ids=pair.source_visit_ids,
            source_point_indices=pair.source_point_indices,
            source_to_token_offsets=pair.source_to_token_offsets,
        )


def _supported_input_sidecar(tmp_path: Path) -> tuple[Path, object, object]:
    input_v2, sidecar = _partial_v2(tmp_path)
    build_supported_input(input_v2_root=input_v2, output_root=sidecar)
    model_input = load_model_input_artifact(sidecar / "model_input")
    pair = load_neural_sample_artifact(sidecar / "adapter_pair")
    return sidecar, model_input, pair


def _checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "source",
        ],
        check=True,
    )
    return checkout


def test_postprocess_preserves_raw_query_identity_and_expands_exactly() -> None:
    result = postprocess_native_predictions(
        pred_masks_mq=np.asarray(
            [[2.0, -2.0, -1.0], [0.0, 3.0, -2.0], [-4.0, 1.0, -3.0]],
            dtype=np.float32,
        ),
        pred_logits_qc=np.asarray(
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 1.0, 2.0]],
            dtype=np.float32,
        ),
        adapter_to_model=np.asarray([0, 0, 1, 2]),
    )

    assert result.raw_query_indices.tolist() == [0, 1, 2]
    assert result.retained_query_indices.tolist() == [0, 1]
    assert result.temporal_query_ids == ("query_0000", "query_0001")
    assert result.query_masks_qa.tolist() == [
        [True, True, False, False],
        [False, False, True, True],
    ]
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
    assert result.token_scores_qa[0].tolist() == pytest.approx(
        [sigmoid(2.0), sigmoid(2.0), sigmoid(0.0), sigmoid(-4.0)]
    )
    assert result.query_scores.shape == (2,)
    assert np.all((0.0 <= result.query_scores) & (result.query_scores <= 1.0))


@pytest.mark.parametrize(
    ("masks", "logits", "mapping", "message"),
    [
        (np.zeros((2, 3)), np.zeros((2, 4)), np.asarray([0, 1]), "query count"),
        (np.zeros((2, 2)), np.zeros((2, 1)), np.asarray([0, 1]), "class count"),
        (np.zeros((2, 2)), np.zeros((2, 3)), np.asarray([0, 2]), "mapping"),
    ],
)
def test_postprocess_rejects_shape_or_mapping_ambiguity(
    masks: np.ndarray, logits: np.ndarray, mapping: np.ndarray, message: str
) -> None:
    with pytest.raises(ExecutorError, match=message):
        postprocess_native_predictions(
            pred_masks_mq=masks,
            pred_logits_qc=logits,
            adapter_to_model=mapping,
        )


def test_checkpoint_extraction_consumes_exactly_model_keys() -> None:
    state = {f"model.layer_{index:04d}": index for index in range(796)}
    state["criterion.empty_weight"] = "excluded"
    state["criterion.change_weights"] = "excluded"

    extracted = extract_model_state_dict({"state_dict": state})

    assert len(extracted) == 796
    assert extracted["layer_0000"] == 0
    assert extracted["layer_0795"] == 795
    del state["model.layer_0795"]
    with pytest.raises(ExecutorError, match="796 model tensors"):
        extract_model_state_dict({"state_dict": state})


def test_executor_publishes_existing_schema_and_bound_raw_output(
    tmp_path: Path,
) -> None:
    sidecar, model_input, pair = _input_sidecar(tmp_path)
    pair_arrays = tmp_path / "pair_arrays.npz"
    _pair_arrays(pair_arrays, pair)
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checkout = _checkout(tmp_path)
    output_arrays = tmp_path / "query_evidence.npz"
    output_manifest = tmp_path / "query_evidence.json"
    raw_output = tmp_path / "raw-output"
    calls: list[object] = []

    def runner(observed: object, *_args: object) -> NativeForwardResult:
        calls.append(observed)
        return NativeForwardResult(
            pred_masks_mq=np.asarray(
                [[2.0, -2.0, -1.0], [0.0, 3.0, -2.0], [-4.0, 1.0, -3.0]],
                dtype=np.float32,
            ),
            pred_logits_qc=np.asarray(
                [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 1.0, 2.0]],
                dtype=np.float32,
            ),
            runtime_s=1.25,
            peak_memory_bytes=4096,
            peak_reserved_memory_bytes=8192,
            rss_peak_bytes=16384,
            device_name="cpu-stub",
            model_tensor_count=796,
        )

    paths = run_rescene_pair_executor(
        input_sidecar=sidecar,
        pair_arrays=pair_arrays,
        output_arrays=output_arrays,
        output_manifest=output_manifest,
        checkpoint=checkpoint,
        checkout=checkout,
        pair_sha256=pair.content_sha256(),
        checkpoint_sha256=_sha256(checkpoint),
        feature_schema="rgb_normals",
        neural_voxel_size_m=0.02,
        raw_output_root=raw_output,
        native_runner=runner,
        source_validator=lambda _path: "f" * 40,
    )

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    with np.load(output_arrays, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "token_indices",
            "query_masks",
            "token_scores",
            "query_scores",
        }
        assert arrays["token_indices"].tolist() == [0, 1, 2, 3]
        assert arrays["query_masks"].shape == (2, 4)
    raw_manifest = json.loads(
        (raw_output / "manifest.json").read_text(encoding="utf-8")
    )
    with np.load(raw_output / "arrays.npz", allow_pickle=False) as raw:
        assert raw["raw_query_indices"].tolist() == [0, 1, 2]
        assert raw["retained_query_indices"].tolist() == [0, 1]
        assert raw["pred_masks_qm"].shape == (3, 3)

    assert paths.output_manifest == output_manifest.absolute()
    assert paths.raw_manifest == (raw_output / "manifest.json").absolute()
    assert set(manifest) == {
        "schema_version",
        "status",
        "backend_name",
        "pair_sha256",
        "checkpoint_sha256",
        "temporal_query_ids",
        "runtime_s",
        "peak_memory_bytes",
        "output_arrays",
    }
    assert manifest["temporal_query_ids"] == ["query_0000", "query_0001"]
    assert raw_manifest["model_input_sha256"] == model_input.content_sha256()
    assert raw_manifest["empty_support_query_count"] == 1
    assert calls and calls[0].content_sha256() == model_input.content_sha256()
    assert (
        load_neural_sample_artifact(sidecar / "adapter_pair").content_sha256()
        == pair.content_sha256()
    )

    with pytest.raises(ExecutorError, match="already exists"):
        run_rescene_pair_executor(
            input_sidecar=sidecar,
            pair_arrays=pair_arrays,
            output_arrays=output_arrays,
            output_manifest=output_manifest,
            checkpoint=checkpoint,
            checkout=checkout,
            pair_sha256=pair.content_sha256(),
            checkpoint_sha256=_sha256(checkpoint),
            feature_schema="rgb_normals",
            neural_voxel_size_m=0.02,
            raw_output_root=raw_output,
            native_runner=runner,
            source_validator=lambda _path: "f" * 40,
        )


def test_executor_accepts_audited_supported_v3_and_preserves_v2_compatibility(
    tmp_path: Path,
) -> None:
    sidecar, model_input, pair = _supported_input_sidecar(tmp_path)
    pair_arrays = tmp_path / "supported_pair.npz"
    _pair_arrays(pair_arrays, pair)
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[object] = []

    def runner(observed: object, *_args: object) -> NativeForwardResult:
        calls.append(observed)
        return NativeForwardResult(
            pred_masks_mq=np.asarray([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32),
            pred_logits_qc=np.asarray(
                [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=np.float32
            ),
            runtime_s=0.5,
            peak_memory_bytes=1024,
            peak_reserved_memory_bytes=2048,
            rss_peak_bytes=4096,
            device_name="cpu-stub",
            model_tensor_count=796,
        )

    run_rescene_pair_executor(
        input_sidecar=sidecar,
        pair_arrays=pair_arrays,
        output_arrays=tmp_path / "supported_evidence.npz",
        output_manifest=tmp_path / "supported_evidence.json",
        checkpoint=checkpoint,
        checkout=_checkout(tmp_path),
        pair_sha256=pair.content_sha256(),
        checkpoint_sha256=_sha256(checkpoint),
        feature_schema="rgb_normals",
        neural_voxel_size_m=0.02,
        raw_output_root=tmp_path / "supported_raw",
        native_runner=runner,
        source_validator=lambda _path: "f" * 40,
    )

    assert len(calls) == 1
    assert calls[0].content_sha256() == model_input.content_sha256()
    with np.load(tmp_path / "supported_evidence.npz", allow_pickle=False) as arrays:
        assert arrays["query_masks"].shape == (2, 3)


def test_executor_rejects_supported_v3_mapping_tamper_before_forward(
    tmp_path: Path,
) -> None:
    sidecar, _, pair = _supported_input_sidecar(tmp_path)
    pair_arrays = tmp_path / "supported_pair.npz"
    _pair_arrays(pair_arrays, pair)
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    mappings = sidecar / "mappings.npz"
    mappings.write_bytes(mappings.read_bytes() + b"tampered")

    with pytest.raises(ExecutorError, match="input sidecar audit failed"):
        run_rescene_pair_executor(
            input_sidecar=sidecar,
            pair_arrays=pair_arrays,
            output_arrays=tmp_path / "supported_evidence.npz",
            output_manifest=tmp_path / "supported_evidence.json",
            checkpoint=checkpoint,
            checkout=_checkout(tmp_path),
            pair_sha256=pair.content_sha256(),
            checkpoint_sha256=_sha256(checkpoint),
            feature_schema="rgb_normals",
            neural_voxel_size_m=0.02,
            raw_output_root=tmp_path / "supported_raw",
            native_runner=lambda *_args: pytest.fail("forward must not run"),
            source_validator=lambda _path: "f" * 40,
        )
