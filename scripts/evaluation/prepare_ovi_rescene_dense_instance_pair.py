#!/usr/bin/env python3
"""Prepare one independently mapped 3RScan pair for dense repair evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_ovi_rescene_dense_instance_repair import (
    _source_bound_grid_sample_factory,
)
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    AssociationForwardOutputs,
    build_d2_inference_bundle,
    load_object_transfer_config,
    run_native_association_forwards,
)
from src.evaluation.ovi_pair_artifact_loader import restore_bound_ovi_pair_artifact
from src.evaluation.ovi_pair_artifacts import export_ovi_object_pair_artifacts
from src.evaluation.ovi_pair_views import OviObjectPairView, build_ovi_object_pair_view

_DOMAIN_KEYS = {
    "full_source",
    "full_adapter",
    "full_model",
    "supported_source",
    "supported_adapter",
    "supported_model",
}


class DensePairPreparationError(ValueError):
    """Raised when a pair preparation input or output is inconsistent."""


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DensePairPreparationError("SHA-256 value is invalid")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DensePairPreparationError(f"{label} is unavailable") from error
    if not isinstance(value, dict):
        raise DensePairPreparationError(f"{label} must be a JSON object")
    return content, value


def _file_record(path: Path, *, recorded_path: str | None = None) -> dict[str, object]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DensePairPreparationError(f"artifact is unavailable: {path}") from error
    return {
        "path": str(path.resolve()) if recorded_path is None else recorded_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _write_or_verify(path: Path, content: bytes, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise DensePairPreparationError(f"existing {label} cannot be read") from error
        if observed != content:
            raise DensePairPreparationError(f"existing {label} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _pair_record(selection: Mapping[str, object], pair_id: str) -> Mapping[str, object]:
    pairs = selection.get("pairs")
    if not isinstance(pairs, list):
        raise DensePairPreparationError("selection pair list is invalid")
    matches = [
        item
        for item in pairs
        if isinstance(item, Mapping) and item.get("pair_id") == pair_id
    ]
    if len(matches) != 1:
        raise DensePairPreparationError("selected pair is not unique")
    return matches[0]


def _mapping_receipt(
    *,
    pair_id: str,
    pair_record: Mapping[str, object],
    selection_manifest: Path,
    native_manifests: Sequence[Path],
    materialized_manifests: Sequence[Path],
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    sessions = pair_record.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 2:
        raise DensePairPreparationError("pair sessions are invalid")
    if len(native_manifests) != 2 or len(materialized_manifests) != 2:
        raise DensePairPreparationError("pair preparation requires exactly two visits")
    visits: list[dict[str, object]] = []
    for visit_id, (session, native_path, materialized_path) in enumerate(
        zip(sessions, native_manifests, materialized_manifests, strict=True)
    ):
        if not isinstance(session, Mapping):
            raise DensePairPreparationError("pair session is invalid")
        scan_id = session.get("scan_id")
        native_content, native = _read_json(native_path, label="native mapping manifest")
        materialized_content, materialized = _read_json(
            materialized_path, label="materialized manifest"
        )
        if (
            session.get("visit_index") != visit_id
            or not isinstance(scan_id, str)
            or native.get("status") != "PASS"
            or native.get("state") != "MAPPING_PASS"
            or native.get("dataset") != "scannet_nyu"
            or native.get("scene") != scan_id
            or materialized.get("status") != "MATERIALIZED_INPUT_PASS"
            or materialized.get("pair_id") != pair_id
            or materialized.get("scan_id") != scan_id
            or materialized.get("visit_index") != visit_id
        ):
            raise DensePairPreparationError("mapping visit identity mismatch")
        frame_ids = native.get("frame_ids")
        frame_count = materialized.get("frame_count")
        if (
            not isinstance(frame_ids, list)
            or not isinstance(frame_count, int)
            or frame_ids != list(range(frame_count))
        ):
            raise DensePairPreparationError("mapping frame domain mismatch")
        output_tree = materialized.get("output_tree")
        audit = native.get("audit")
        if not isinstance(output_tree, Mapping) or not isinstance(audit, Mapping):
            raise DensePairPreparationError("mapping summary metadata is invalid")
        visits.append(
            {
                "visit_index": visit_id,
                "scan_id": scan_id,
                "frame_count": frame_count,
                "materialized": {
                    "manifest": {
                        "path": str(materialized_path.resolve()),
                        "sha256": hashlib.sha256(materialized_content).hexdigest(),
                        "byte_count": len(materialized_content),
                    },
                    "output_tree": {
                        key: output_tree.get(key)
                        for key in ("sha256", "byte_count", "file_count")
                    },
                },
                "native_mapping": {
                    "manifest": {
                        "path": str(native_path.resolve()),
                        "sha256": hashlib.sha256(native_content).hexdigest(),
                        "byte_count": len(native_content),
                    },
                    "artifacts": native.get("artifacts"),
                    "audit": {
                        key: audit.get(key)
                        for key in (
                            "frame_count",
                            "requested_frame_count",
                            "mapper_skipped_frame_count",
                            "raycast_bbox_count",
                            "full_frame_bbox_count",
                            "full_frame_bbox_ratio",
                            "stale_raycast_ids",
                        )
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_D2_OVIMAP_SMOKE_V1",
        "status": "REAL_D2_PAIR_PASS",
        "pair_id": pair_id,
        "selection": {
            "manifest": _file_record(
                selection_manifest,
                recorded_path=str(selection_manifest.relative_to(repository_root)),
            ),
            "pair_id": pair_id,
            "role": "D2_EVAL",
            "selected_before_method_results": True,
        },
        "visits": visits,
        "compute": {
            "frontend_forward_count_per_visit": 1,
            "native_mapping_count_per_visit": 1,
            "visit_mapping": "independent",
        },
    }


def _pair_receipt(
    *,
    pair: OviObjectPairView,
    artifact_root: Path,
    mapping_receipt: Path,
    selection_manifest: Path,
) -> dict[str, object]:
    manifest_path = artifact_root / "manifest.json"
    _, manifest = _read_json(manifest_path, label="OVI pair artifact manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise DensePairPreparationError("OVI pair artifact outputs are invalid")
    full_outputs: dict[str, dict[str, object]] = {}
    for role in ("t0_instance_ply", "t0_rgb_ply", "t1_instance_ply", "t1_rgb_ply"):
        record = outputs.get(role)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise DensePairPreparationError(f"OVI pair artifact {role} is missing")
        full_outputs[role] = _file_record(artifact_root / str(record["path"]))
    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_D2_PAIR_VIEW_RECEIPT_V1",
        "status": "REAL_D2_PAIR_VIEW_PASS",
        "pair_id": pair.pair_id,
        "domain_id": pair.domain_id,
        "pair_content_sha256": pair.content_sha256(),
        "coordinate_frame_id": pair.coordinate_frame_id,
        "source_pair_manifest": _file_record(
            selection_manifest,
            recorded_path=str(selection_manifest.relative_to(REPO_ROOT)),
        ),
        "parent_mapping_receipt": _file_record(mapping_receipt),
        "visits": manifest.get("visits"),
        "geometry_audit": manifest.get("geometry"),
        "local_artifact_manifest": _file_record(manifest_path),
        "local_full_outputs": full_outputs,
    }


def write_native_forward_cache(
    *,
    arrays_path: Path,
    metadata_path: Path,
    outputs: AssociationForwardOutputs,
    pair_id: str,
    pair_content_sha256: str,
    sample_content_sha256: str,
    candidate_counts: tuple[int, int],
    domain_counts: Mapping[str, int],
    config_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Publish the exact arrays/metadata consumed by the dense repair runner."""

    if arrays_path.parent != metadata_path.parent:
        raise DensePairPreparationError("forward outputs must share one directory")
    if any(path.exists() or path.is_symlink() for path in (arrays_path, metadata_path)):
        raise DensePairPreparationError("forward output already exists")
    if not isinstance(pair_id, str) or not pair_id:
        raise DensePairPreparationError("pair ID is invalid")
    for value in (
        pair_content_sha256,
        sample_content_sha256,
        config_sha256,
        checkpoint_sha256,
        *outputs.visit_forward_sha256,
    ):
        _digest(value)
    if set(domain_counts) != _DOMAIN_KEYS or any(
        type(value) is not int or value < 0 for value in domain_counts.values()
    ):
        raise DensePairPreparationError("forward domain counts are invalid")
    arrays = {
        "independent_t0": np.asarray(outputs.independent_model_features[0]),
        "independent_t1": np.asarray(outputs.independent_model_features[1]),
        "pred_masks_mq": np.asarray(outputs.joint_forward.pred_masks_mq),
        "pred_logits_qc": np.asarray(outputs.joint_forward.pred_logits_qc),
    }
    if any(
        value.ndim != 2
        or not np.issubdtype(value.dtype, np.floating)
        or not np.all(np.isfinite(value))
        for value in arrays.values()
    ):
        raise DensePairPreparationError("forward arrays are invalid")
    if (
        arrays["independent_t0"].shape[0] + arrays["independent_t1"].shape[0]
        != domain_counts["supported_model"]
        or arrays["pred_masks_mq"].shape[0] != domain_counts["supported_model"]
        or arrays["pred_masks_mq"].shape[1] != arrays["pred_logits_qc"].shape[0]
    ):
        raise DensePairPreparationError("forward array domains differ")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".native-forwards.", dir=arrays_path.parent))
    try:
        staged_arrays = staging / arrays_path.name
        with staged_arrays.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        arrays_record = _file_record(staged_arrays, recorded_path=str(arrays_path.resolve()))
        metadata = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBJECT_LEVEL_NATIVE_FORWARDS_V1",
            "status": "PASS",
            "pair_id": pair_id,
            "pair_content_sha256": pair_content_sha256,
            "sample_content_sha256": sample_content_sha256,
            "candidate_counts": list(candidate_counts),
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "domains": dict(domain_counts),
            "forward": {
                "visit_forward_sha256": list(outputs.visit_forward_sha256),
                "independent_runtime_s": list(outputs.independent_runtime_s),
                "joint_runtime_s": outputs.joint_forward.runtime_s,
                "peak_memory_bytes": outputs.joint_forward.peak_memory_bytes,
                "peak_reserved_memory_bytes": outputs.joint_forward.peak_reserved_memory_bytes,
                "rss_peak_bytes": outputs.joint_forward.rss_peak_bytes,
                "device_name": outputs.joint_forward.device_name,
                "model_tensor_count": outputs.joint_forward.model_tensor_count,
            },
            "npz": arrays_record,
        }
        staged_metadata = staging / metadata_path.name
        with staged_metadata.open("xb") as stream:
            stream.write(_json_bytes(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_arrays, arrays_path)
        os.replace(staged_metadata, metadata_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_dense_pair(
    *,
    object_config_path: Path,
    pair_id: str,
    native_manifests: tuple[Path, Path],
    materialized_manifests: tuple[Path, Path],
    mapping_receipt_path: Path,
    pair_receipt_path: Path,
    pair_artifact_root: Path,
    forward_root: Path,
) -> dict[str, object]:
    config = load_object_transfer_config(object_config_path)
    if (
        config.d2_mapping_receipt != mapping_receipt_path.resolve()
        or config.d2_pair_view_receipt != pair_receipt_path.resolve()
    ):
        raise DensePairPreparationError("object config output paths differ")
    selection_content, selection = _read_json(
        config.selection_manifest, label="selection manifest"
    )
    if hashlib.sha256(selection_content).hexdigest() != config.selection_manifest_sha256:
        raise DensePairPreparationError("selection manifest binding mismatch")
    pair_record = _pair_record(selection, pair_id)
    mapping = _mapping_receipt(
        pair_id=pair_id,
        pair_record=pair_record,
        selection_manifest=config.selection_manifest,
        native_manifests=native_manifests,
        materialized_manifests=materialized_manifests,
    )
    _write_or_verify(
        mapping_receipt_path,
        _json_bytes(mapping),
        label="mapping receipt",
    )
    pair = build_ovi_object_pair_view(
        pair_record=pair_record,
        source_manifest_sha256=config.selection_manifest_sha256,
        native_manifests=native_manifests,
        materialized_manifests=materialized_manifests,
    )
    if pair_artifact_root.exists() or pair_artifact_root.is_symlink():
        _, receipt = _read_json(pair_receipt_path, label="pair receipt")
        pair = restore_bound_ovi_pair_artifact(pair, receipt)
    else:
        export_ovi_object_pair_artifacts(pair, pair_artifact_root)
        receipt = _pair_receipt(
            pair=pair,
            artifact_root=pair_artifact_root,
            mapping_receipt=mapping_receipt_path,
            selection_manifest=config.selection_manifest,
        )
        _write_or_verify(pair_receipt_path, _json_bytes(receipt), label="pair receipt")
        pair = restore_bound_ovi_pair_artifact(pair, receipt)
    bundle = build_d2_inference_bundle(
        pair,
        sampler_source_path=config.native_sampler_source.path,
        sampler_source_sha256=config.native_sampler_source.sha256,
        sampler_seed=config.sampler_seed,
        maximum_candidates=config.maximum_candidates,
        grid_sample_factory=_source_bound_grid_sample_factory(
            config.native_sampler_source.path
        ),
    )
    outputs = run_native_association_forwards(bundle, config)
    forward_root.mkdir(parents=True, exist_ok=True)
    arrays_path = forward_root / "native_forwards.npz"
    metadata_path = forward_root / "native_forwards.json"
    supported_visits = np.asarray(bundle.model_input.model_visit_ids)
    write_native_forward_cache(
        arrays_path=arrays_path,
        metadata_path=metadata_path,
        outputs=outputs,
        pair_id=pair_id,
        pair_content_sha256=pair.content_sha256(),
        sample_content_sha256=bundle.sample.content_sha256(),
        candidate_counts=tuple(len(value) for value in pair.candidate_ids),
        domain_counts={
            "full_source": bundle.geometry.source_point_count,
            "full_adapter": bundle.geometry.adapter_count,
            "full_model": bundle.sampling.model_count,
            "supported_source": bundle.sample.source_point_count,
            "supported_adapter": len(bundle.sample.visit_ids),
            "supported_model": len(supported_visits),
        },
        config_sha256=config.source_sha256,
        checkpoint_sha256=config.rescene_checkpoint.sha256,
    )
    return {
        "status": "PASS",
        "pair_id": pair_id,
        "pair_content_sha256": pair.content_sha256(),
        "sample_content_sha256": bundle.sample.content_sha256(),
        "candidate_counts": [len(value) for value in pair.candidate_ids],
        "supported_model_count": len(supported_visits),
        "mapping_receipt": str(mapping_receipt_path),
        "pair_receipt": str(pair_receipt_path),
        "forward_metadata": str(metadata_path),
        "forward_arrays": str(arrays_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-config", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--native-manifests", type=Path, nargs=2, required=True)
    parser.add_argument("--materialized-manifests", type=Path, nargs=2, required=True)
    parser.add_argument("--mapping-receipt", type=Path, required=True)
    parser.add_argument("--pair-receipt", type=Path, required=True)
    parser.add_argument("--pair-artifact-root", type=Path, required=True)
    parser.add_argument("--forward-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_dense_pair(
            object_config_path=args.object_config.resolve(),
            pair_id=args.pair_id,
            native_manifests=tuple(path.resolve() for path in args.native_manifests),
            materialized_manifests=tuple(
                path.resolve() for path in args.materialized_manifests
            ),
            mapping_receipt_path=args.mapping_receipt.resolve(),
            pair_receipt_path=args.pair_receipt.resolve(),
            pair_artifact_root=args.pair_artifact_root.resolve(),
            forward_root=args.forward_root.resolve(),
        )
    except (DensePairPreparationError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DensePairPreparationError",
    "prepare_dense_pair",
    "write_native_forward_cache",
]
