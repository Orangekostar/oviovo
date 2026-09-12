"""S0/S2 transfer from actual authorized legacy dense replay, with native fallback."""

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
from scripts.evaluation.run_crove_fine_current_map import _semantic_configs
from scripts.evaluation.run_crove_multiview_apartment import evaluate_saved_readouts
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.surface_readout import resolve_semantic_update
from src.oviv2.surface_semantics import (
    SurfaceSemanticStrategy,
    transfer_surface_semantics,
)


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    fine = json.loads(path(config["source_config"]).read_text())
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    room = path(config["run_root"]) / "dev/apartment"
    output = room / "local_semantic_controls"
    output.mkdir(exist_ok=True)
    compact = path(config["compact_output_root"])
    methods = {
        SurfaceSemanticStrategy.S0: "B_SEM_CROVE_S0_DENSE_REPLAY",
        SurfaceSemanticStrategy.S2: "B_SEM_CROVE_S2_DENSE_REPLAY",
    }
    references = [
        room / "local_semantic_replay" / f"t{v}" / "reference.npz" for v in (0, 1)
    ]
    if not all(p.exists() for p in references):
        raise RuntimeError("both actual per-visit local references must be complete")
    settings = {
        "methods": list(methods.values()),
        "states": ["B3", "H2"],
        "reference_sha256": [file_hash(p) for p in references],
        "transfer_parameters": fine["semantic_transfer"],
        "source": "replayed legacy 10-class RADSeg observations at 5cm evidence voxel centers; not the old room0 reference mesh",
        "implementation": "existing transfer_surface_semantics S0/S2 with unchanged thresholds",
        "missing": "native KEEP_SOURCE for zero-output/no-evidence transfer, never a new unknown rejection",
        "S2_unary": "top-1 confidence scalar only, no invented posterior and no second local reliability factor",
        "cross_visit_reference_query": False,
        "GT_before_all_predictions": False,
    }
    registry = compact / "apartment_local_semantic_controls_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen local transfer policy changed")
    _atomic_json(registry, settings)
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz, visits = data["vertices_xyz"], data["source_visit_ids"]
    for state in ("B3", "H2"):
        baseline_file = room / "native_cached_batch" / f"{state}_MV_NATIVE_CACHED.npz"
        with np.load(baseline_file) as data:
            baseline = {k: data[k] for k in data.files}
        source = baseline["source_indices"]
        binding_file = output / f"{state}_input_binding.json"
        binding = {
            "native_prediction": file_hash(baseline_file),
            "policy": file_hash(registry),
        }
        if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
            raise ValueError("frozen local transfer input changed")
        _atomic_json(binding_file, binding)
        started = time.monotonic()
        buffers = {
            strategy: {
                "ids": baseline["semantic_ids"].copy(),
                "confidence": baseline["semantic_confidence"].copy(),
                "reliability": np.zeros(len(source), np.float32),
                "source_codes": np.zeros(len(source), np.uint8),
                "kind": np.zeros(len(source), np.uint8),
            }
            for strategy in methods
        }
        for visit in (0, 1):
            rows = np.flatnonzero(visits[source] == visit)
            with np.load(references[visit]) as ref:
                if int(ref["visit_id"]) != visit:
                    raise ValueError("reference visit identity differs")
                result = transfer_surface_semantics(
                    fine_vertices_xyz=xyz[source[rows]],
                    reference_vertices_xyz=ref["xyz"],
                    reference_semantic_ids=ref["semantic_ids"],
                    reference_supports=ref["semantic_support"],
                    entity_semantic_ids=baseline["semantic_ids"][rows],
                    entity_confidences=baseline["semantic_confidence"][rows],
                    configs=tuple(
                        c for c in _semantic_configs(fine) if c.strategy in methods
                    ),
                    neighbor_count=fine["semantic_transfer"]["neighbor_count"],
                    point_batch_size=fine["semantic_transfer"]["point_batch_size"],
                )
            for strategy in methods:
                value = result[strategy]
                valid = value.semantic_ids > 0
                dest = buffers[strategy]
                dest["ids"][rows[valid]] = value.semantic_ids[valid]
                dest["confidence"][rows[valid]] = value.semantic_confidences[valid]
                dest["reliability"][rows] = value.support_reliabilities
                dest["source_codes"][rows] = value.semantic_source_codes
                dest["kind"][rows[valid]] = 1
            print(
                state, "visit", visit, "local transfer complete", len(rows), flush=True
            )
        for strategy, method in methods.items():
            value = buffers[strategy]
            ids, roles = resolve_semantic_update(
                baseline["semantic_ids"],
                baseline["eval_role"],
                value["ids"],
                value["kind"],
                crosswalk,
            )
            prediction = dict(baseline)
            prediction.update(
                semantic_ids=ids,
                semantic_confidence=value["confidence"],
                eval_role=roles,
                support_reliability=value["reliability"],
                local_semantic_source_codes=value["source_codes"],
                semantic_update_kind=value["kind"],
                feature_covered=value["kind"].astype(bool),
            )
            atomic_npz(output / f"{state}_{method}.npz", **prediction)
            _atomic_json(
                output / f"{state}_{method}_prediction.json",
                {
                    "state": state,
                    "method": method,
                    "source_rows": len(source),
                    "changed_semantic_rows": int(
                        np.count_nonzero(ids != baseline["semantic_ids"])
                    ),
                    "changed_role_rows": int(
                        np.count_nonzero(roles != baseline["eval_role"])
                    ),
                    "transfer_source_counts": {
                        str(code): int(np.count_nonzero(value["source_codes"] == code))
                        for code in np.unique(value["source_codes"])
                    },
                    "geometry_and_owner_fixed": True,
                    "prediction_and_write_seconds_shared_pair": time.monotonic()
                    - started,
                    "adaptation": "per-visit local evidence voxel reference plus native missing fallback; legacy 10-class cache remapped to public common-v2 IDs",
                },
            )
            print(state, method, "full prediction saved", flush=True)
    evaluate_saved_readouts(
        config, state_config, pair, crosswalk, output, list(methods.values())
    )


if __name__ == "__main__":
    main()
