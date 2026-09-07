from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.run_ovi_rescene_dense_instance_repair import (
    DenseInstanceRepairRunError,
    cached_forward_outputs,
    csv_bytes,
)


def _arrays_bytes(
    *,
    t0_rows: int = 2,
    t1_rows: int = 3,
    model_rows: int = 5,
    query_count: int = 4,
) -> bytes:
    stream = io.BytesIO()
    np.savez(
        stream,
        independent_t0=np.ones((t0_rows, 8), dtype=np.float32),
        independent_t1=np.ones((t1_rows, 8), dtype=np.float32),
        pred_masks_mq=np.ones((model_rows, query_count), dtype=np.float32),
        pred_logits_qc=np.ones((query_count, 3), dtype=np.float32),
    )
    return stream.getvalue()


def _metadata(arrays: bytes) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "OVI_RESCENE_OBJECT_LEVEL_NATIVE_FORWARDS_V1",
                "status": "PASS",
                "pair_id": "pair",
                "pair_content_sha256": "1" * 64,
                "sample_content_sha256": "2" * 64,
                "config_sha256": "3" * 64,
                "checkpoint_sha256": "4" * 64,
                "candidate_counts": [2, 3],
                "domains": {"supported_model": 5},
                "npz": {
                    "path": "/bound/native_forwards.npz",
                    "sha256": hashlib.sha256(arrays).hexdigest(),
                    "byte_count": len(arrays),
                },
                "forward": {
                    "visit_forward_sha256": ["5" * 64, "6" * 64],
                    "independent_runtime_s": [0.1, 0.2],
                    "joint_runtime_s": 0.3,
                    "peak_memory_bytes": 10,
                    "peak_reserved_memory_bytes": 20,
                    "rss_peak_bytes": 30,
                    "device_name": "fake",
                    "model_tensor_count": 40,
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load(metadata: bytes, arrays: bytes):
    return cached_forward_outputs(
        metadata,
        arrays,
        pair_id="pair",
        pair_content_sha256="1" * 64,
        sample_content_sha256="2" * 64,
        candidate_counts=(2, 3),
        independent_model_counts=(2, 3),
        supported_model_count=5,
        expected_query_count=4,
        config_sha256="3" * 64,
        checkpoint_sha256="4" * 64,
    )


def test_cached_forward_accepts_exact_bound_identity_and_dimensions() -> None:
    arrays = _arrays_bytes()

    output = _load(_metadata(arrays), arrays)

    assert output.joint_forward.pred_masks_mq.shape == (5, 4)
    assert output.joint_forward.pred_logits_qc.shape == (4, 3)
    assert output.independent_model_features[0].shape == (2, 8)
    assert output.independent_model_features[1].shape == (3, 8)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("pair_content_sha256", "9" * 64, "pair identity mismatch"),
        ("sample_content_sha256", "9" * 64, "sample identity mismatch"),
        ("candidate_counts", [9, 3], "candidate counts mismatch"),
        ("config_sha256", "9" * 64, "config identity mismatch"),
        ("checkpoint_sha256", "9" * 64, "checkpoint identity mismatch"),
    ),
)
def test_cached_forward_rejects_identity_mismatch_before_publication(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    arrays = _arrays_bytes()
    payload = json.loads(_metadata(arrays))
    payload[field] = replacement
    publication = tmp_path / "published"

    with pytest.raises(DenseInstanceRepairRunError, match=message):
        _load((json.dumps(payload) + "\n").encode(), arrays)

    assert not publication.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"model_rows": 6}, "joint mask M dimension mismatch"),
        ({"query_count": 5}, "joint query dimensions mismatch"),
        ({"t0_rows": 4}, "independent visit row count mismatch"),
    ),
)
def test_cached_forward_rejects_m_q_or_independent_row_mismatch(
    tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    arrays = _arrays_bytes(**kwargs)
    publication = tmp_path / "published"

    with pytest.raises(DenseInstanceRepairRunError, match=message):
        _load(_metadata(arrays), arrays)

    assert not publication.exists()


def test_cached_forward_rejects_arrays_that_do_not_match_metadata_digest() -> None:
    expected = _arrays_bytes()
    observed = _arrays_bytes(query_count=5)

    with pytest.raises(DenseInstanceRepairRunError, match="array binding mismatch"):
        _load(_metadata(expected), observed)


def test_csv_bytes_preserves_stable_columns_and_null_values() -> None:
    rows = (
        {"method": "P0", "conditional_recall": None, "status": "NOT_COMPUTED"},
        {"method": "P2", "conditional_recall": 0.5, "status": "PASS"},
    )

    content = csv_bytes(
        rows,
        fieldnames=("method", "conditional_recall", "status"),
    )

    assert content == (
        b"method,conditional_recall,status\n"
        b"P0,,NOT_COMPUTED\n"
        b"P2,0.5,PASS\n"
    )


def test_csv_bytes_rejects_missing_or_extra_columns() -> None:
    with pytest.raises(DenseInstanceRepairRunError, match="CSV row schema"):
        csv_bytes(
            ({"method": "P0", "extra": 1},),
            fieldnames=("method", "conditional_recall"),
        )
