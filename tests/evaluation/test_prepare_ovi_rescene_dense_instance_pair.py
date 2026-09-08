from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.prepare_ovi_rescene_dense_instance_pair import (
    DensePairPreparationError,
    write_native_forward_cache,
)
from scripts.evaluation.write_ovi_mapping_receipt import write_mapping_receipt
from scripts.evaluation.rescene_pair_executor import NativeForwardResult
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    AssociationForwardOutputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _outputs() -> AssociationForwardOutputs:
    return AssociationForwardOutputs(
        independent_model_features=(
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ),
        visit_forward_sha256=("1" * 64, "2" * 64),
        independent_runtime_s=(0.1, 0.2),
        joint_forward=NativeForwardResult(
            pred_masks_mq=np.asarray(
                [[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]], dtype=np.float32
            ),
            pred_logits_qc=np.asarray(
                [[1.0, 0.0, -1.0], [0.0, 2.0, -2.0]], dtype=np.float32
            ),
            runtime_s=0.3,
            peak_memory_bytes=10,
            peak_reserved_memory_bytes=20,
            rss_peak_bytes=30,
            device_name="test cuda:0",
            model_tensor_count=796,
        ),
    )


def test_script_entrypoint_resolves_repository_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts/evaluation/prepare_ovi_rescene_dense_instance_pair.py"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_write_mapping_receipt_binds_two_native_visits(tmp_path: Path) -> None:
    pair_id = "scene0001_00-scene0001_01"
    scan_ids = ("scan-a", "scan-b")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": pair_id,
                        "sessions": [
                            {"visit_index": visit_id, "scan_id": scan_id}
                            for visit_id, scan_id in enumerate(scan_ids)
                        ],
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    native_paths: list[Path] = []
    materialized_paths: list[Path] = []
    for visit_id, scan_id in enumerate(scan_ids):
        native = tmp_path / f"native-{visit_id}.json"
        native.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "state": "MAPPING_PASS",
                    "dataset": "scannet_nyu",
                    "scene": scan_id,
                    "frame_ids": [0, 1],
                    "artifacts": {"visit": visit_id},
                    "audit": {
                        "frame_count": 2,
                        "requested_frame_count": 2,
                        "mapper_skipped_frame_count": 0,
                        "raycast_bbox_count": 1,
                        "full_frame_bbox_count": 0,
                        "full_frame_bbox_ratio": 0.0,
                        "stale_raycast_ids": [],
                    },
                }
            ),
            encoding="ascii",
        )
        materialized = tmp_path / f"materialized-{visit_id}.json"
        materialized.write_text(
            json.dumps(
                {
                    "status": "MATERIALIZED_INPUT_PASS",
                    "pair_id": pair_id,
                    "scan_id": scan_id,
                    "visit_index": visit_id,
                    "frame_count": 2,
                    "output_tree": {
                        "sha256": str(visit_id) * 64,
                        "byte_count": 10,
                        "file_count": 6,
                    },
                }
            ),
            encoding="ascii",
        )
        native_paths.append(native)
        materialized_paths.append(materialized)

    output = tmp_path / "mapping_receipt.json"
    result = write_mapping_receipt(
        pair_id=pair_id,
        selection_manifest=selection,
        native_manifests=tuple(native_paths),
        materialized_manifests=tuple(materialized_paths),
        output=output,
        repository_root=tmp_path,
    )

    assert result["status"] == "REAL_D2_PAIR_PASS"
    assert [visit["scan_id"] for visit in result["visits"]] == list(scan_ids)
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_write_native_forward_cache_is_bound_and_exclusive(tmp_path: Path) -> None:
    arrays_path = tmp_path / "native_forwards.npz"
    metadata_path = tmp_path / "native_forwards.json"
    write_native_forward_cache(
        arrays_path=arrays_path,
        metadata_path=metadata_path,
        outputs=_outputs(),
        pair_id="pair",
        pair_content_sha256="a" * 64,
        sample_content_sha256="b" * 64,
        candidate_counts=(4, 5),
        domain_counts={
            "full_source": 10,
            "full_adapter": 8,
            "full_model": 4,
            "supported_source": 9,
            "supported_adapter": 7,
            "supported_model": 3,
        },
        config_sha256="c" * 64,
        checkpoint_sha256="d" * 64,
    )

    content = arrays_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "PASS"
    assert metadata["pair_id"] == "pair"
    assert metadata["candidate_counts"] == [4, 5]
    assert metadata["domains"]["supported_model"] == 3
    assert metadata["npz"] == {
        "path": str(arrays_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    with np.load(arrays_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "independent_t0",
            "independent_t1",
            "pred_masks_mq",
            "pred_logits_qc",
        }
        np.testing.assert_array_equal(
            archive["pred_masks_mq"], _outputs().joint_forward.pred_masks_mq
        )

    before = (arrays_path.read_bytes(), metadata_path.read_bytes())
    with pytest.raises(DensePairPreparationError, match="already exists"):
        write_native_forward_cache(
            arrays_path=arrays_path,
            metadata_path=metadata_path,
            outputs=_outputs(),
            pair_id="pair",
            pair_content_sha256="a" * 64,
            sample_content_sha256="b" * 64,
            candidate_counts=(4, 5),
            domain_counts={
                "full_source": 10,
                "full_adapter": 8,
                "full_model": 4,
                "supported_source": 9,
                "supported_adapter": 7,
                "supported_model": 3,
            },
            config_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
        )
    assert (arrays_path.read_bytes(), metadata_path.read_bytes()) == before
