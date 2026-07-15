from types import SimpleNamespace

import numpy as np

from scripts.evaluation.measure_baseline_queries import _entity_embeddings, _query_model_name


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
    np.testing.assert_array_equal(embeddings, [[1.0, 0.0], [0.0, 1.0]])
