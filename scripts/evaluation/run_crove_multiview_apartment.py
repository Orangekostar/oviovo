"""Frozen B3/H2 whole-source M1 cached-view controls through the common bridge."""

from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_fine_current_map import _entity_records
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.evaluation.two_visit_snapshot_metrics import (
    TwoVisitEvaluationContext,
    evaluate_two_visit_points,
)
from src.oviv2.fine_dynamic_policy import CoarseSurfaceState, load_b3_surface_policy
from src.oviv2.surface_multiview_semantics import (
    aggregate_views,
    owner_posteriors_to_rows,
)
from src.oviv2.surface_readout import resolve_semantic_update


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    vocabulary = {
        item.semantic_id: item.native_name
        for item in crosswalk.aliases.values()
        if item.matched
    }
    class_ids = np.array(sorted(vocabulary), np.int32)
    names = [vocabulary[key] for key in class_ids]
    run = path(config["run_root"])
    output = run / "dev/apartment/native_cached_batch"
    output.mkdir(parents=True, exist_ok=True)
    compact = path(config["compact_output_root"])
    methods = ["MV_NATIVE_CACHED", "MV_SINGLE", "MV_TOPK_MEAN"]
    settings = {
        "scene": pair["scene"],
        "pair_id": pair["pair_id"],
        "states": ["B3", "H2"],
        "methods": methods,
        "class_ids": class_ids.tolist(),
        "class_names": names,
        "text": "bare complete public common-v2 vocabulary; shared SigLIP text encoder and scale",
        "candidate_pool": "same native retained per-visit six-crop mean cache",
        "native": "last eight cached views; native visible-area weighted raw features then normalize",
        "single_topk": "one/four highest-visible-area views; normalize each cached six-crop mean then uniform mean",
        "frame_windows": pair["visit_frame_ranges"],
        "local_to_original_offsets": [766, 1217],
        "t1_owner_offset": pair["t1_owner_offset"],
        "new_image_forwards": 0,
        "source_state": "exact saved B3/H2 current-valid masks; no geometry or owner edits",
        "missing_observation": "KEEP_SOURCE semantic and role; never explicit unknown rejection",
        "state_policy": "same source-local cached view rule in both states; current-aware fusion remains pending",
    }
    registry = compact / "multiview_apartment_registry.json"
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen Apartment M1 settings differ")
    else:
        _atomic_json(registry, settings)
    native_root = path(
        "$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment"
    )
    banks, provenance = {}, []
    record_sets = []
    for key in ("b0_entities", "b2_entities"):
        entity_path = path(pair[key])
        records = _entity_records(entity_path)
        for line in entity_path.read_text().splitlines():
            item = json.loads(line)
            metadata = item["metadata"]
            records[int(metadata["source_instance_id"])]["metadata"] = metadata
        record_sets.append(records)
    owner_confidence = {}
    for visit, record_set in enumerate(record_sets):
        native_manifest = (
            native_root
            / f"t{visit}"
            / f"2f638911509f-apartment-t{visit}"
            / "native_mapping_manifest.json"
        )
        manifest = json.loads(native_manifest.read_text())
        manifest_hash = file_hash(native_manifest)
        feature_path = Path(manifest["artifacts"]["semantic_features"]["path"])
        expected = manifest["artifacts"]["semantic_features"]["sha256"]
        if file_hash(feature_path) != expected:
            raise ValueError(
                "native feature hash differs from original mapping manifest"
            )
        with feature_path.open("rb") as handle:
            bank = pickle.load(handle)
        offset = int(pair["t1_owner_offset"]) if visit else 0
        start, stop = pair["visit_frame_ranges"][f"t{visit}"]
        export_path = path(
            f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/materialized-inputs/apartment/t{visit}/apartment/export_manifest.json"
        )
        export = json.loads(export_path.read_text())
        if export["source_frame_interval"] != [start, stop]:
            raise ValueError("materialized local frame interval differs")
        for owner, record in record_set.items():
            if record["metadata"]["native_manifest_sha256"] != manifest_hash:
                raise ValueError("lifted entity and native manifest differ")
            owner_confidence[owner + offset] = float(record["semantic_score"])
            if owner not in bank:
                continue
            entry = bank[owner]
            if not np.array_equal(
                entry["color"], record["metadata"]["instance_color_rgb"]
            ):
                raise ValueError(
                    "feature owner color does not match lifted source owner"
                )
            frames = np.asarray(entry["frame_id"], np.int64)
            if len(frames) != len(np.unique(frames)) or np.any(
                (frames < 0) | (frames > stop - start)
            ):
                raise ValueError("duplicate or unauthorized native local frame")
            banks[owner + offset] = (entry, visit, frames + start)
        provenance.append(
            {
                "visit": visit,
                "native_manifest_sha256": manifest_hash,
                "feature_sha256": expected,
                "materialized_manifest_sha256": file_hash(export_path),
            }
        )
    _atomic_json(output / "native_feature_binding.json", provenance)
    text_cache = output / "text_features.npz"
    if not text_cache.exists():
        model_path = path(
            "$HOME/oviovo_baseline_builds/ovimap-ubuntu24-native/siglip-large-patch16-384"
        )
        model = (
            AutoModel.from_pretrained(model_path, local_files_only=True)
            .eval()
            .to("cuda:1")
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokens = tokenizer(
            names, padding="max_length", max_length=64, return_tensors="pt"
        ).to("cuda:1")
        with torch.inference_mode():
            text = model.get_text_features(**tokens)
            if not isinstance(text, torch.Tensor):
                text = text.pooler_output
            text = torch.nn.functional.normalize(text, dim=-1).cpu().numpy()
            scale = float(model.logit_scale.exp().cpu())
        atomic_npz(text_cache, text=text, logit_scale=scale, class_ids=class_ids)
        del model
        torch.cuda.empty_cache()
    with np.load(text_cache) as data:
        text, scale = data["text"], float(data["logit_scale"])
        if not np.array_equal(data["class_ids"], class_ids):
            raise ValueError("text class IDs differ")
    for method in methods:
        target = output / f"{method}_owner_features.npz"
        if target.exists():
            continue
        feature_owners, posteriors, refs = [], [], []
        for owner, (entry, visit, original_frames) in sorted(banks.items()):
            if len(original_frames) < 2:
                continue
            area, features = np.asarray(entry["vis_area"]), np.asarray(entry["feat"])
            if method == "MV_NATIVE_CACHED":
                selected = np.arange(max(0, len(area) - 8), len(area))
                if area[selected].sum() <= 0:
                    continue
                z = np.average(features[selected], axis=0, weights=area[selected])
                z = z / max(float(np.linalg.norm(z)), 1e-12)
            else:
                selected = np.argsort(-area, kind="stable")[
                    : 1 if method == "MV_SINGLE" else 4
                ]
                z = aggregate_views(features[selected], np.ones(len(selected)))
            if z is None or np.linalg.norm(z) < 1e-12:
                continue
            logits = scale * (z @ text.T)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            feature_owners.append(owner)
            posteriors.append(p)
            refs.append(
                {
                    "owner": owner,
                    "visit": visit,
                    "original_frames": original_frames[selected].tolist(),
                }
            )
        atomic_npz(
            target,
            feature_owners=np.asarray(feature_owners),
            owner_posterior=np.asarray(posteriors),
            class_ids=class_ids,
        )
        _atomic_json(
            output / f"{method}_views.json",
            {"observations": refs, "new_image_forwards": 0},
        )
    apply_and_evaluate_owner_features(
        config, state_config, pair, crosswalk, output, methods, owner_confidence
    )


def apply_and_evaluate_owner_features(
    config, state_config, pair, crosswalk, output, methods, owner_confidence
):
    """Shared full-state bridge for complete owner posteriors from each family."""
    run = path(config["run_root"])
    class_ids = np.array(
        sorted({v.semantic_id for v in crosswalk.aliases.values() if v.matched}),
        np.int32,
    )
    # Write both full frozen states for every method before opening evaluator targets.
    for state in ("B3", "H2"):
        with np.load(run / "bridge" / f"{state}_legacy_prediction.npz") as base:
            source, owners = base["source_indices"], base["owner_ids"]
            old_ids, old_roles = base["semantic_ids"], base["eval_role"]
            original_sources, order = (
                base["legacy_eval_source"],
                base["legacy_eval_order"],
            )
            if not base["legacy_covered"].all() or len(order) != len(source):
                raise ValueError(
                    "bridge does not cover the complete current source set"
                )
        keys = np.array(sorted(owner_confidence))
        values = np.array([owner_confidence[key] for key in keys])
        baseline_conf = np.zeros(len(owners), np.float32)
        known = np.isin(owners, keys)
        baseline_conf[known] = values[np.searchsorted(keys, owners[known])]
        for method in methods:
            target = output / f"{state}_{method}.npz"
            if target.exists():
                continue
            started = time.monotonic()
            feature_file = output / f"{state}_{method}_owner_features.npz"
            if not feature_file.exists():
                feature_file = output / f"{method}_owner_features.npz"
            with np.load(feature_file) as data:
                if not np.array_equal(data["class_ids"], class_ids):
                    raise ValueError(
                        "owner posterior class order differs from fixed vocabulary"
                    )
                proposed, conf, covered = owner_posteriors_to_rows(
                    owners, data["feature_owners"], data["owner_posterior"], class_ids
                )
            kind = covered.astype(np.uint8)
            ids, roles = resolve_semantic_update(
                old_ids, old_roles, proposed, kind, crosswalk
            )
            conf[~covered] = baseline_conf[~covered]
            atomic_npz(
                target,
                source_indices=source,
                owner_ids=owners,
                semantic_ids=ids,
                semantic_confidence=conf,
                eval_role=roles,
                legacy_eval_source=original_sources,
                semantic_update_kind=kind,
                legacy_eval_order=order,
                feature_covered=covered,
            )
            _atomic_json(
                output / f"{state}_{method}_prediction.json",
                {
                    "state": state,
                    "method": method,
                    "source_rows": len(source),
                    "feature_coverage": float(covered.mean()),
                    "changed_semantic_rows": int(np.count_nonzero(ids != old_ids)),
                    "changed_role_rows": int(np.count_nonzero(roles != old_roles)),
                    "prediction_and_write_seconds": time.monotonic() - started,
                    "geometry_and_owner_fixed": True,
                },
            )
            print(state, method, "full prediction saved", len(source), flush=True)
    evaluate_saved_readouts(config, state_config, pair, crosswalk, output, methods)


def evaluate_saved_readouts(
    config, state_config, pair, crosswalk, output, methods, *, allow_owner_changes=False
):
    """Evaluate already-saved complete point readouts through the frozen bridge."""
    run = path(config["run_root"])
    compact = path(config["compact_output_root"])
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz, original_owners, visits, vertex_indices = (
            data["vertices_xyz"],
            data["owner_entity_ids"],
            data["source_visit_ids"],
            data["source_vertex_indices"],
        )
    n0 = int(np.count_nonzero(visits == 0))
    if not np.all(visits[:n0] == 0) or not np.all(visits[n0:] == 1):
        raise ValueError("canonical visit order differs")
    policy = load_b3_surface_policy(path(pair["b3_provenance"]), original_owners[:n0])
    observed = policy.states != int(CoarseSurfaceState.RETAINED_UNOBSERVED)
    with np.load(run / "bridge/evaluation_context.npz") as data:
        context_values = {
            f.name: data[f.name] for f in fields(TwoVisitEvaluationContext)
        }
    for key in ("event_id", "frame_id", "intervention_frame_id"):
        context_values[key] = context_values[key].item()
    context = TwoVisitEvaluationContext(**context_values)
    hypotheses = (
        path(state_config["run_root"]) / "dev/pairs" / pair["pair_id"] / "hypotheses"
    )
    for state, hypothesis, variant in [
        ("B3", "H1_B3", "D1_B3"),
        ("H2", "H2_INHERIT", "D2_INHERIT"),
    ]:
        with np.load(
            hypotheses
            / hypothesis
            / variant
            / "entity_epoch_state/entity_epoch_state.npz"
        ) as data:
            if not np.array_equal(
                data["source_visit_ids"], visits
            ) or not np.array_equal(data["source_vertex_indices"], vertex_indices):
                raise ValueError("frozen state source keys differ")
            expected_rows = np.flatnonzero(data["current_valid_after"])
        retained_rows = expected_rows[expected_rows < n0]
        retained = {
            "retained_t0_xyz": xyz[retained_rows],
            "retained_t0_t1_observed_mask": observed[retained_rows],
        }
        for method in methods:
            target = compact / f"apartment_{state}_{method}.json"
            if target.exists():
                continue
            with np.load(output / f"{state}_{method}.npz") as data:
                canonical, order = data["source_indices"], data["legacy_eval_order"]
                if not np.array_equal(canonical, expected_rows) or (
                    not allow_owner_changes
                    and not np.array_equal(
                        data["owner_ids"], original_owners[canonical]
                    )
                ):
                    raise ValueError("readout changed state geometry/owner")
                inverse = np.searchsorted(canonical, order)
                if not np.array_equal(canonical[inverse], order):
                    raise ValueError(
                        "legacy evaluator ordering is not an exact inverse"
                    )
                ids, roles, owner_ids = (
                    data["semantic_ids"][inverse],
                    data["eval_role"][inverse],
                    data["owner_ids"][inverse],
                )
            started = time.monotonic()
            measured = evaluate_two_visit_points(
                points_xyz=xyz[order],
                semantic_ids=ids,
                eval_role=roles,
                owner_ids=owner_ids,
                scene_id=pair["scene"],
                timestamp=float(context.frame_id),
                context=context,
                crosswalk=crosswalk,
                **retained,
            )
            gates = config["selection_policy"]["apartment_gates"]
            passed = (
                measured["ghost"] <= gates["maximum_ghost"]
                and measured["background_f1_at_5cm"]
                >= gates["minimum_background_f1_at_5cm"]
                and measured["surface_precision_at_5cm"]
                >= gates["minimum_surface_precision_at_5cm"]
                and measured["t1_observed_region_stale_precision"]
                >= gates["minimum_observed_stale_precision"]
            )
            record = json.loads(
                (output / f"{state}_{method}_prediction.json").read_text()
            )
            record.update(
                case=pair["pair_id"],
                split="dev",
                status="FULL_MAP_EVALUATED",
                metrics=measured,
                passes_original_per_case_gates=passed,
                evaluation_seconds=time.monotonic() - started,
                interpretation=(
                    "Source geometry stays frozen; owner IDs and supported semantic roles may change according to the declared readout."
                    if allow_owner_changes
                    else "Role-dependent object/background metrics may change from supported semantics; frozen source geometry and owner do not change."
                ),
            )
            _atomic_json(target, record)
            print(state, method, measured, "gates", passed, flush=True)


if __name__ == "__main__":
    main()
