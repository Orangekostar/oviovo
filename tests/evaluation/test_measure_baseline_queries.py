import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation.measure_baseline_queries import (
    _TorchQueryBackend,
    _entity_embeddings,
    _load_queries,
    _query_model_name,
    _query_protocol,
    main,
)


def test_ovimap_query_uses_siglip_and_skips_single_observation_entities() -> None:
    entities = [
        SimpleNamespace(
            semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            metadata={"observation_count": 1},
        ),
        SimpleNamespace(
            semantic_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            metadata={"observation_count": 2},
        ),
        SimpleNamespace(semantic_embedding=None, metadata={"observation_count": 3}),
    ]

    embeddings = _entity_embeddings(entities, "ovimap")

    assert _query_model_name("ovimap") == "SigLIP-L/16-384"
    assert "fewer than two frames" in _query_protocol("ovimap")
    np.testing.assert_array_equal(embeddings, [[0.0, 1.0]])


def test_non_ovimap_queries_keep_all_embedded_entities() -> None:
    entities = [
        SimpleNamespace(
            semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            metadata={"observation_count": 1},
        ),
        SimpleNamespace(
            semantic_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            metadata={},
        ),
    ]

    embeddings = _entity_embeddings(entities, "conceptgraphs")

    assert _query_model_name("conceptgraphs") == "ViT-H-14"
    assert "OVI-MAP" not in _query_protocol("conceptgraphs")
    np.testing.assert_array_equal(embeddings, [[1.0, 0.0], [0.0, 1.0]])


def test_query_loader_accepts_frozen_tesse_top_level_classes() -> None:
    content = json.dumps(
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "classes": ["Fridge", "Books"],
        }
    ).encode("utf-8")

    assert _load_queries(content) == ["Fridge", "Books"]


def test_torch_backend_synchronizes_the_selected_cuda_device() -> None:
    calls: list[str] = []
    backend = object.__new__(_TorchQueryBackend)
    backend._device = "cuda:3"
    backend._torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: calls.append(device))
    )

    backend.synchronize()

    assert calls == ["cuda:3"]


class _FakeQueryBackend:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def prepare_entities(self, embeddings: np.ndarray) -> str:
        self.events.append(("prepare_entities", embeddings.copy()))
        return "entities"

    def synchronize(self) -> None:
        self.events.append(("synchronize",))

    def tokenize(self, text: str) -> str:
        self.events.append(("tokenize", text))
        return f"tokens:{text}"

    def encode_text(self, tokens: str) -> str:
        self.events.append(("encode_text", tokens))
        return f"feature:{tokens}"

    def rank_entities(self, feature: str, entity_features: str) -> int:
        self.events.append(("rank_entities", feature, entity_features))
        return 1 if "chair" in feature else 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oviv2_args(tmp_path: Path, *, warmup: int = 10, repeats: int = 5) -> list[str]:
    snapshot = tmp_path / "snapshot.npz"
    entities = tmp_path / "entities.jsonl"
    queries = tmp_path / "queries.json"
    checkpoint = tmp_path / "open_clip_pytorch_model.bin"
    protocol = tmp_path / "protocol.json"
    snapshot.write_bytes(b"current-map-arrays")
    entities.write_bytes(b"current-map-entities\n")
    queries.write_text(
        json.dumps({"vocabulary": {"classes": ["chair", "table"]}}) + "\n",
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"frozen-vit-h-14")
    protocol.write_text(
        json.dumps(
            {
                "query": {
                    "vocabulary_path": str(queries),
                    "vocabulary_sha256": _sha256(queries),
                    "text_model_id": "ViT-H-14",
                    "checkpoint_sha256": _sha256(checkpoint),
                    "operation_order": [
                        "full_tokenization",
                        "text_encoding",
                        "l2_normalization",
                        "current_map_entity_ranking",
                    ],
                    "warmup_count": 10,
                    "measured_repeats": 5,
                    "cuda_synchronize_each_query": True,
                    "precomputed_query_embeddings": False,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        "--baseline", "oviv2",
        "--snapshot", str(snapshot),
        "--entities", str(entities),
        "--queries", str(queries),
        "--clip-weight", str(checkpoint),
        "--device", "cuda:3",
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--protocol", str(protocol),
        "--output", str(tmp_path / "query.json"),
    ]


def _current_snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        method="OVIV2",
        scene_id="apartment",
        scope="current",
        entities=[
            SimpleNamespace(
                semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                metadata={"readout_valid": True},
            ),
            SimpleNamespace(
                semantic_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
                metadata={"readout_valid": True},
            ),
        ],
    )


def test_oviv2_main_measures_full_queries_and_binds_all_sources(tmp_path: Path) -> None:
    backend = _FakeQueryBackend()
    factory_calls: list[tuple[object, ...]] = []

    def backend_factory(
        baseline: str, model_name: str, checkpoint: Path, device: str
    ) -> _FakeQueryBackend:
        factory_calls.append((baseline, model_name, checkpoint, device))
        return backend

    ticks = iter(index / 1000.0 for index in range(1000))
    args = _oviv2_args(tmp_path)

    assert main(
        args,
        backend_factory=backend_factory,
        snapshot_loader=lambda _snapshot, _entities: _current_snapshot(),
        clock=lambda: next(ticks),
    ) == 0

    output = Path(args[-1])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert factory_calls == [
        (
            "oviv2",
            "ViT-H-14",
            tmp_path / "open_clip_pytorch_model.bin",
            "cuda:3",
        )
    ]
    assert payload["baseline"] == "oviv2"
    assert payload["model"] == "ViT-H-14"
    assert payload["warmup_count"] == 10
    assert payload["measured_repeats"] == 5
    assert payload["query_count"] == 2
    assert payload["entity_count"] == 2
    assert len(payload["latencies_ms"]) == 10
    assert payload["query_mean_ms"] == pytest.approx(1.0)
    assert payload["query_p50_ms"] == pytest.approx(1.0)
    assert payload["query_p95_ms"] == pytest.approx(1.0)
    assert [
        (sample["repeat_index"], sample["query_index"], sample["top_entity_index"])
        for sample in payload["samples"]
    ] == [
        (repeat, query, 1 if query == 0 else 0)
        for repeat in range(5)
        for query in range(2)
    ]
    for role, name in {
        "snapshot": "snapshot.npz",
        "entities": "entities.jsonl",
        "queries": "queries.json",
        "checkpoint": "open_clip_pytorch_model.bin",
    }.items():
        path = tmp_path / name
        assert payload["sources"][role] == {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "byte_count": path.stat().st_size,
        }
    assert payload["text_model"]["checkpoint_sha256"] == _sha256(
        tmp_path / "open_clip_pytorch_model.bin"
    )
    assert payload["query_vocabulary_sha256"] == _sha256(tmp_path / "queries.json")

    tokenizations = [event for event in backend.events if event[0] == "tokenize"]
    encodings = [event for event in backend.events if event[0] == "encode_text"]
    rankings = [event for event in backend.events if event[0] == "rank_entities"]
    synchronizations = [event for event in backend.events if event[0] == "synchronize"]
    assert len(tokenizations) == len(encodings) == len(rankings) == 20
    assert all(event[1].startswith("an image of ") for event in tokenizations)
    assert len(synchronizations) == 41


@pytest.mark.parametrize(
    ("warmup", "repeats", "message"),
    [(9, 5, "warmup"), (10, 4, "repeats")],
)
def test_oviv2_rejects_non_protocol_measurement_counts(
    tmp_path: Path, warmup: int, repeats: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        main(
            _oviv2_args(tmp_path, warmup=warmup, repeats=repeats),
            backend_factory=lambda *_args: _FakeQueryBackend(),
            snapshot_loader=lambda _snapshot, _entities: _current_snapshot(),
        )


def test_oviv2_rejects_non_current_snapshot(tmp_path: Path) -> None:
    snapshot = _current_snapshot()
    snapshot.scope = "history"
    with pytest.raises(ValueError, match="current-map"):
        main(
            _oviv2_args(tmp_path),
            backend_factory=lambda *_args: _FakeQueryBackend(),
            snapshot_loader=lambda _snapshot, _entities: snapshot,
        )


def test_oviv2_rejects_source_mutation_during_measurement(tmp_path: Path) -> None:
    args = _oviv2_args(tmp_path)

    def mutating_loader(_snapshot: Path, entities: Path) -> SimpleNamespace:
        entities.write_bytes(b"mutated\n")
        return _current_snapshot()

    with pytest.raises(ValueError, match="entities source changed"):
        main(
            args,
            backend_factory=lambda *_args: _FakeQueryBackend(),
            snapshot_loader=mutating_loader,
        )


def test_oviv2_rejects_checkpoint_hash_outside_frozen_protocol(tmp_path: Path) -> None:
    args = _oviv2_args(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["query"]["checkpoint_sha256"] = "0" * 64
    protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        main(
            args,
            backend_factory=lambda *_args: _FakeQueryBackend(),
            snapshot_loader=lambda _snapshot, _entities: _current_snapshot(),
        )


def test_existing_ovimap_baseline_accepts_local_model_directory(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.npz"
    entities = tmp_path / "entities.jsonl"
    queries = tmp_path / "queries.json"
    checkpoint = tmp_path / "siglip"
    snapshot.write_bytes(b"snapshot")
    entities.write_bytes(b"entities")
    queries.write_text(
        json.dumps({"vocabulary": {"classes": ["chair"]}}) + "\n",
        encoding="utf-8",
    )
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(b"{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    output = tmp_path / "query.json"
    backend = _FakeQueryBackend()
    current = _current_snapshot()
    for entity in current.entities:
        entity.metadata["observation_count"] = 2

    assert main(
        [
            "--baseline", "ovimap",
            "--snapshot", str(snapshot),
            "--entities", str(entities),
            "--queries", str(queries),
            "--clip-weight", str(checkpoint),
            "--device", "cpu",
            "--warmup", "1",
            "--output", str(output),
        ],
        backend_factory=lambda *_args: backend,
        snapshot_loader=lambda _snapshot, _entities: current,
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["repeat_count"] == 1
    assert payload["sources"]["checkpoint"]["path"] == str(checkpoint.resolve())
    assert payload["sources"]["checkpoint"]["byte_count"] == 10
    assert len(payload["sources"]["checkpoint"]["sha256"]) == 64
    assert [event for event in backend.events if event[0] == "tokenize"] == [
        ("tokenize", "chair"),
        ("tokenize", "chair"),
    ]
