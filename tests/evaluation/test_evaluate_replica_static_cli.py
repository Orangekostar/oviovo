from __future__ import annotations

import gzip
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
