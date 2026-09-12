"""Replay authorized legacy CROVE dense observations into per-visit local fields."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.oviv2.dense_projection import DenseSemanticConfig, DenseSemanticIntegrator
from src.oviv2.dense_semantics import load_dense_frame
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    pair = json.loads(path(config["state_config"]).read_text())["splits"]["dev"][
        "pairs"
    ][0]
    old_config_file = path(
        "$HOME/oviovo_baseline_runs/20260901_crove_ovimap_anchor/crove_a6_c069510_source/normalized_run_config.json"
    )
    old = json.loads(old_config_file.read_text())
    cache = path(
        "$HOME/oviovo_dense_cache/tesse_cd_radseg_b_sam_s4_k4_native/apartment"
    )
    manifest = json.loads((cache / "dense_manifest.json").read_text())
    vocabulary_file = ROOT / "configs/evaluation/vocabularies/tesse_cd_apartment.json"
    vocabulary = json.loads(vocabulary_file.read_text())
    mapping = np.array([0] + vocabulary["object_semantic_ids"], np.int32)
    if manifest["class_count"] != len(mapping) - 1 or manifest[
        "source_frame_ids"
    ] != list(range(manifest["frame_count"])):
        raise ValueError("legacy dense frame/class binding differs")
    params = {
        "voxel_size_m": old["voxel_size_m"],
        "integration_radius_m": old["dense_integration_radius_m"],
        "minimum_probability": old["dense_minimum_probability"],
        "minimum_quality": old["dense_minimum_quality"],
        "entropy_power": old["dense_entropy_power"],
        "view_angle_power": old["dense_view_angle_power"],
    }
    settings = {
        "case": "apartment",
        "frame_windows": pair["visit_frame_ranges"],
        "legacy_run_config_sha256": file_hash(old_config_file),
        "dense_manifest_sha256": file_hash(cache / "dense_manifest.json"),
        "vocabulary_sha256": file_hash(vocabulary_file),
        "legacy_class_to_common_v2": mapping.tolist(),
        "projection_parameters": params,
        "semantic_top_k": old["semantic_top_k"],
        "implementation": "existing DenseSemanticIntegrator and SparseEvidenceStore; one fresh store per authorized visit",
        "reference": "occupied semantic evidence voxel centers, not a reconstructed TSDF surface; exact old 10-class cache is explicitly a legacy baseline",
        "reference_support": "maximum accumulated candidate support divided by sum of retained top-k supports; uncalibrated scalar, no reconstructed full posterior",
        "source_geometry_changes": False,
        "cross_visit_evidence_union": False,
        "GT_read": False,
    }
    compact = path(config["compact_output_root"])
    registry = compact / "apartment_local_semantic_replay_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen dense replay policy changed")
    _atomic_json(registry, settings)
    dataset = TesseCdRgbdDataset(
        path(pair["rgbd_root"]),
        pair["scene"],
        path(pair["rgbd_export_manifest"]),
        path(pair["causal_schedule"]),
    )
    evidence_config = EvidenceConfig(
        block_resolution=old["block_resolution"], semantic_top_k=old["semantic_top_k"]
    )
    integrator = DenseSemanticIntegrator(DenseSemanticConfig(**params))
    for visit in (0, 1):
        root = (
            path(config["run_root"])
            / "dev/apartment/local_semantic_replay"
            / f"t{visit}"
        )
        root.mkdir(parents=True, exist_ok=True)
        target = root / "reference.npz"
        if target.exists():
            print("visit", visit, "local reference already exists", flush=True)
            continue
        store = SparseEvidenceStore(evidence_config)
        start, stop = pair["visit_frame_ranges"][f"t{visit}"]
        started = time.monotonic()
        receipts = []
        for frame_id in range(start, stop + 1):
            filename = f"frame{frame_id:06d}.npz"
            digest = manifest["cache_files_sha256"][filename]
            dense = load_dense_frame(cache / filename, expected_sha256=digest)
            if (
                dense.source_frame_id != frame_id
                or dense.class_count != len(mapping) - 1
            ):
                raise ValueError("legacy cache frame identity differs")
            result = integrator.integrate(dataset[frame_id], dense, store, frame_id)
            receipts.append(
                {
                    "frame_id": frame_id,
                    "dense_sha256": digest,
                    "valid_pixels": result.valid_pixel_count,
                    "updated_voxels": result.updated_voxel_count,
                }
            )
            if frame_id % 16 == 0:
                print(
                    "visit",
                    visit,
                    "frame",
                    frame_id,
                    "blocks",
                    store.allocated_block_count,
                    flush=True,
                )
        snapshot = root / "evidence.npz"
        store.save(snapshot)
        del store
        with np.load(snapshot) as data:
            support = data["semantic_support"]
            total = support.sum(-1)
            index = np.argwhere(total > 0)
            index_tuple = tuple(index.T)
            candidates = data["semantic_ids"][index_tuple]
            values = support[index_tuple]
            chosen = values.argmax(1)
            labels = candidates[np.arange(len(index)), chosen]
            confidence = values[np.arange(len(index)), chosen] / total[index_tuple]
            keys = (
                data["block_keys"][index[:, 0]] * evidence_config.block_resolution
                + index[:, 1:]
            )
        atomic_npz(
            target,
            xyz=((keys + 0.5) * params["voxel_size_m"]).astype(np.float32),
            semantic_ids=mapping[labels],
            semantic_support=confidence.astype(np.float32),
            voxel_keys=keys,
            visit_id=visit,
        )
        _atomic_json(root / "frame_receipts.json", receipts)
        record = {
            "visit": visit,
            "authorized_frames": len(receipts),
            "reference_voxels": len(keys),
            "replay_seconds": time.monotonic() - started,
            "reference_sha256": file_hash(target),
            "status": "LOCAL_REFERENCE_COMPLETE_TRANSFER_PENDING",
            "GT_read": False,
        }
        _atomic_json(compact / f"apartment_t{visit}_local_semantic_replay.json", record)
        print(record, flush=True)


if __name__ == "__main__":
    main()
