from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.evaluate_ovi_rescene_apartment_v3 import (
    ApartmentEvaluationError,
    _project_points_to_pixels,
    evaluate_apartment_artifact,
    load_query_evidence_artifact,
)
from scripts.evaluation.prepare_ovi_rescene_supported_v3 import build_supported_input
from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.query_entity_resolver import ResolverConfig
from src.oviv2.query_instance_projection import ProjectionConfig
from src.oviv2.two_visit_contracts import PairRelation
from tests.evaluation.test_prepare_ovi_rescene_supported_v3 import _partial_v2


def _file_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _evidence_artifact(root: Path, pair_sha256: str) -> Path:
    root.mkdir()
    arrays_path = root / "query_evidence.npz"
    with arrays_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            token_indices=np.asarray([2, 0, 1], dtype=np.int64),
            query_masks=np.asarray(
                [[True, True, True], [False, True, False]], dtype=np.bool_
            ),
            token_scores=np.asarray(
                [[0.9, 0.8, 0.7], [0.1, 0.8, 0.2]], dtype=np.float32
            ),
            query_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        )
    manifest_path = root / "query_evidence.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "backend_name": "concerto",
                "pair_sha256": pair_sha256,
                "checkpoint_sha256": "f" * 64,
                "temporal_query_ids": ["q0", "q1"],
                "runtime_s": 0.5,
                "peak_memory_bytes": 1024,
                "output_arrays": {
                    "path": arrays_path.name,
                    "sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
                    "byte_count": arrays_path.stat().st_size,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _b4() -> tuple[PairRelation, ...]:
    return (
        PairRelation(
            temporal_query_id="geom:0",
            t0_entity_ids=("a",),
            t1_entity_ids=("b",),
            state="persistent_moved",
            query_confidence=0.75,
            evidence={"pair_score": 0.75},
            identity_source="geometric_baseline",
        ),
    )


def test_visual_projection_uses_entity_geometry_as_shared_reference() -> None:
    entity = np.asarray(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    bounds = (20, 40, 220, 240)

    entity_pixels = _project_points_to_pixels(entity, entity, bounds=bounds)
    selected_pixels = _project_points_to_pixels(entity[[1]], entity, bounds=bounds)

    assert selected_pixels.tolist() == entity_pixels[[1]].tolist()


def test_apartment_report_serializes_three_readouts_and_non_gt_visuals(
    tmp_path: Path,
) -> None:
    input_root, supported_root = _partial_v2(tmp_path)
    receipt = build_supported_input(
        input_v2_root=input_root, output_root=supported_root
    )
    pair = load_neural_sample_artifact(supported_root / "adapter_pair")
    evidence_manifest = _evidence_artifact(tmp_path / "evidence", pair.content_sha256())
    b4_source = tmp_path / "b4-source.json"
    b4_source.write_text('{"source":"frozen"}\n', encoding="utf-8")
    output = tmp_path / "report"

    summary = evaluate_apartment_artifact(
        supported_root=supported_root,
        evidence_manifest_path=evidence_manifest,
        b4_relations=_b4(),
        b4_source_record=_file_record(b4_source),
        output_root=output,
        projection_config=ProjectionConfig(
            minimum_entity_token_coverage=0.25,
            minimum_source_point_coverage=0.25,
        ),
        resolver_config=ResolverConfig(
            minimum_supported_entity_coverage=0.25,
            minimum_query_precision=0.25,
            minimum_query_confidence=0.2,
        ),
    )

    assert summary["artifact_id"] == "OVI_RESCENE_APARTMENT_EVIDENCE_V3"
    assert summary["status"] == "PASS"
    assert summary["pair_sha256"] == pair.content_sha256()
    assert summary["counts"] == {
        "b4_raw": 1,
        "b4_unique": 1,
        "legacy_raw": 2,
        "legacy_unique": 2,
        "supported_raw": 1,
        "supported_unique": 1,
        "supported_fallback": 0,
        "supported_conflict": 0,
        "supported_one_sided": 1,
        "model_collision": 0,
        "duplicate_topology": 0,
    }
    assert summary["coverage"] == {
        "source_fraction": 0.75,
        "adapter_fraction": 0.75,
        "model_fraction": 2 / 3,
        "entity_fraction": 1.0,
        "conditional_query_coverage": 0.5,
    }
    assert receipt["status"] == "SUPPORTED_INFERENCE_PASS"
    assert [case["role"] for case in summary["visual_selection"]] == [
        "representative",
        "low_support",
    ]
    assert [case["query_id"] for case in summary["visual_selection"]] == [
        "q0",
        "q1",
    ]
    assert set(summary["artifacts"]) == {
        "b4_relations",
        "legacy_relations",
        "supported_relations",
        "resolver_diagnostics",
        "representative_visual",
        "low_support_visual",
    }
    for record in summary["artifacts"].values():
        artifact = output / record["path"]
        assert artifact.stat().st_size == record["byte_count"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["sha256"]
    assert (output / "visuals/representative.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    supported = json.loads(
        (output / "supported_relations.json").read_text(encoding="utf-8")
    )
    assert supported["relations"][0]["temporal_query_id"] == "q0"

    with pytest.raises(ApartmentEvaluationError, match="already exists"):
        evaluate_apartment_artifact(
            supported_root=supported_root,
            evidence_manifest_path=evidence_manifest,
            b4_relations=_b4(),
            b4_source_record=_file_record(b4_source),
            output_root=output,
            projection_config=ProjectionConfig(),
            resolver_config=ResolverConfig(),
        )


def test_apartment_evidence_loader_restores_permuted_tokens_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    input_root, supported_root = _partial_v2(tmp_path)
    build_supported_input(input_v2_root=input_root, output_root=supported_root)
    pair = load_neural_sample_artifact(supported_root / "adapter_pair")
    manifest = _evidence_artifact(tmp_path / "evidence", pair.content_sha256())

    evidence = load_query_evidence_artifact(manifest, pair)

    assert evidence.query_masks is not None
    assert evidence.query_masks.tolist() == [
        [True, True, True],
        [True, False, False],
    ]
    arrays = manifest.parent / "query_evidence.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tampered")
    with pytest.raises(ApartmentEvaluationError, match="binding mismatch"):
        load_query_evidence_artifact(manifest, pair)
