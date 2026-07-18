from __future__ import annotations

import gzip
import json
import pickle
from types import SimpleNamespace

import numpy as np

from scripts.evaluation import evaluate_replica_static
from src.evaluation.baselines.contracts import RuntimeBreakdown


def test_conceptgraphs_loader_still_returns_artifact(tmp_path, monkeypatch) -> None:
    map_file = tmp_path / "map.pkl.gz"
    with gzip.open(map_file, "wb") as handle:
        pickle.dump(
            {
                "objects": [
                    {
                        "pcd_np": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                        "clip_ft": np.asarray([1.0, 0.0], dtype=np.float32),
                        "image_idx": [0],
                    }
                ]
            },
            handle,
        )
    monkeypatch.setattr(
        evaluate_replica_static,
        "_classify_conceptgraphs_objects",
        lambda objects, class_names, clip_weight, device: ("chair",),
    )
    monkeypatch.setattr(
        evaluate_replica_static,
        "_conceptgraphs_runtime",
        lambda args: RuntimeBreakdown(frame_count=1),
    )
    args = SimpleNamespace(
        map_file=map_file,
        clip_weight=tmp_path / "weight.bin",
        device="cpu",
        scene_id="room0",
        timestamp=0.0,
        upstream_commit="93277a0",
    )

    artifact = evaluate_replica_static._load_conceptgraphs(args, ("chair",))

    assert artifact.metadata.method_key == "CONCEPTGRAPHS"
    assert artifact.snapshot.entities[0].semantic_label == "chair"


def test_openfusion_loader_reads_exported_semantic_query(tmp_path) -> None:
    prediction = tmp_path / "semantic_prediction.npz"
    np.savez_compressed(
        prediction,
        points=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        class_ids=np.asarray([1, 0], dtype=np.int16),
        vocabulary=np.asarray(["wall", "chair"]),
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "frame_count": 200,
                "initialization_s": 1.0,
                "mapping_s": 2.0,
                "semantic_query_s": 3.0,
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "vlfusion.npz"
    state_path.write_bytes(b"state")
    trajectory_path = tmp_path / "traj.txt"
    np.savetxt(trajectory_path, np.eye(4).reshape(1, 16))
    args = SimpleNamespace(
        openfusion_prediction=prediction,
        openfusion_runtime=runtime_path,
        openfusion_state=state_path,
        openfusion_trajectory=trajectory_path,
        gpu_dmon=None,
        scene_id="room0",
        timestamp=1990.0,
        upstream_commit="86bdc1a",
    )

    artifact = evaluate_replica_static._load_openfusion(args)

    assert artifact.metadata.method_key == "OPENFUSION"
    assert artifact.runtime.initialization_s == 1.0
    assert artifact.runtime.backend_s == 2.0
    assert artifact.runtime.finalization_s == 3.0
