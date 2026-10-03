"""Shared production FC recovery, including real views and fixed output exports."""

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np

from static_ovmap.backbone_wave1.runtime import check_gpu_once, exclusive_lock
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.module_validation.scannet_study import load_prediction, save_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
from static_ovmap.recovery_wave2.recovery_registry import build_registry

from .costs import load_base_cost_inventory, recovery_usage
from .outputs import build_method_outputs
from .projected_views import _verified_identity, build_projected_views, load_projected_request
from .protocol import load_spec
from .region_worker import (FCSession, archived_loader, archived_u2_plan, classify_regions,
                            projected_plan, validate_request_outcomes)
from .runtime import require_frozen_execution


ARM_METHOD = {"G1": "CT_A3_ER", "G3": "CT_G3", "U2": "CT_H_U2"}
ARM_SOURCE = {"G1": "G1_FC", "G3": "G3_FC", "U2": "ARCHIVED_U2_FC"}


@dataclass
class RecoveryInputs:
    scene: str
    binding_identity: str
    data: dict
    baseline: object
    xyz: np.ndarray
    faces: np.ndarray
    raw: np.ndarray
    sources: dict
    valid_ids: list
    nearest: np.ndarray
    matched: np.ndarray
    identity: str
    cost_inventory: dict = None


@dataclass
class GPULease:
    binding_identity: str
    path: str
    pid: int
    active: bool = True


@contextmanager
def gpu_lease(binding, receipt_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(binding["gpu"]):
        raise ValueError("production FC recovery requires the single bound GPU environment")
    with exclusive_lock(binding["gpu_lock"]):
        resource = check_gpu_once(binding["gpu"], Path(receipt_root) / "gpu_check.json")
        foreign = [line for line in resource["occupants"] if int(line.split(",")[1].strip()) != os.getpid()]
        if foreign:
            raise RuntimeError("RESOURCE_BLOCK: the bound recovery GPU is occupied")
        lease = GPULease(binding["identity"], binding["gpu_lock"], os.getpid())
        try:
            yield lease
        finally:
            lease.active = False


def load_recovery_inputs(binding, scene, *, context=None, index=None):
    require_frozen_execution(binding, scene)
    root = Path(binding["output_root"])
    index = index or ConsumptionIndex(root / "validation/input_verifications.json")
    context_path = root / "contexts" / (scene + ".json")
    data = context or (read(context_path) if context_path.is_file() else binding["scenes"][scene])
    if "identity" in data:
        _verified_identity(data)
    path = Path(data["predictions"]["NATIVE_READOUT"])
    index.identity(path)
    manifest = read(path)
    index.identity(path.parent / manifest["arrays"]["path"], manifest["arrays"])
    baseline = load_prediction(path)
    capture_path = Path(data["capture_manifest"])
    index.identity(capture_path)
    capture = read(capture_path)
    _verified_identity(capture)
    surface = capture_path.parent / capture["surface"]["path"]
    index.identity(surface, capture["surface"])
    with np.load(surface, allow_pickle=False) as arrays:
        xyz, faces, raw = arrays["surface_xyz"], arrays["surface_faces"], arrays["original_owner"]
    if (baseline.scene_id != scene or capture["scene_id"] != scene
            or _array_digest(xyz) != baseline.geometry.xyz_sha256
            or _array_digest(faces) != baseline.geometry.faces_sha256
            or capture["tsdf"]["sha256"] != baseline.geometry.tsdf_sha256):
        raise ValueError("resident recovery anchor differs from the actual same-scene Native capture")
    index.identity(data["config"])
    config = PathResolver(binding["path_map"]).rewrite(read(data["config"]))
    projection_path = Path(config["scenes"][scene]["projection"])
    index.identity(projection_path)
    projection = read(projection_path)
    index.identity(projection_path.with_name("projection.npz"), {"sha256": projection["sha256"]})
    with np.load(projection_path.with_name("projection.npz"), allow_pickle=False) as arrays:
        nearest, matched = arrays["nearest"], arrays["matched"]
    if projection["identity"] != baseline.geometry.projection_identity:
        raise ValueError("resident export projection differs from the common Native geometry")
    sources = {}
    ids = list(map(int, data["models"]["native"]["valid_ids"]))
    for name, item in data["sources"].items():
        index.identity(item["path"])
        source = read(item["path"])
        _verified_identity(source)
        if source["identity"] != item["identity"] or source["valid_ids"] != ids:
            raise ValueError("resident existing N/Q/F evidence or vocabulary changed")
        sources[name] = source
    if set(sources) != {"N", "Q", "F"}:
        raise ValueError("resident common baseline requires the exact existing N/Q/F")
    cost_inventory = load_base_cost_inventory(binding, data, index=index)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene,
        "native": baseline.record_key, "capture": capture["identity"], "projection": projection["identity"],
        "sources": {name: source["identity"] for name, source in sources.items()},
        "base_cost_inventory": cost_inventory["identity"]})
    index.write_memo(root / "validation/input_verifications.json")
    return RecoveryInputs(scene, binding["identity"], data, baseline, xyz, faces, raw,
                          sources, ids, nearest, matched, identity, cost_inventory)


def export_conditions(binding, inputs, registry, region_sources, arms, output_root, index, *, region_receipt=None):
    spec = load_spec(binding["spec"])
    methods = {row["id"]: row for row in spec["methods"]}
    selected = [methods[ARM_METHOD[arm]] for arm in arms]
    recovery = {ARM_SOURCE[arm]: region_sources[arm] for arm in arms}
    costs = None
    if inputs.cost_inventory is not None:
        if region_receipt is None:
            raise ValueError("production recovery export lacks its actual selected-request cost receipt")
        costs = {ARM_SOURCE[arm]: recovery_usage(region_sources[arm], region_receipt, ARM_SOURCE[arm]) for arm in arms}
    outputs = build_method_outputs(inputs.baseline, inputs.raw, registry, inputs.sources,
        binding["final_temperatures"], inputs.valid_ids, inputs.nearest, inputs.matched, recovery, selected,
        cost_inventory=inputs.cost_inventory, recovery_costs=costs)
    exported = {}
    for method, payload in outputs.items():
        path = save_prediction(payload, Path(output_root) / method)
        manifest = read(path)
        exported[method] = {"manifest": str(path), "record_key": payload.record_key,
            "prediction_key": payload.prediction_key, "files": [index.identity(path),
                index.identity(path.parent / manifest["arrays"]["path"], manifest["arrays"])]}
    return exported


def recover_fc(binding, inputs, output_root, *, arms=None, session=None, cold=False,
               lease=None, views_root=None):
    freeze = require_frozen_execution(binding, inputs.scene)
    spec = load_spec(binding["spec"])
    replica = inputs.scene in spec["cohorts"]["replica8"] or inputs.scene == spec["smoke_scene"]
    permitted = {"G1", "G3", "U2"} if replica else {"G1"}
    arms = tuple(arms if arms is not None else (("G1", "G3", "U2") if replica else ("G1",)))
    if (not arms or len(set(arms)) != len(arms) or not set(arms) <= permitted
            or inputs.binding_identity != binding["identity"] or not inputs.baseline.locked):
        raise ValueError("production recovery leaves its fixed cohort, common anchor or bound arms")
    root = Path(output_root).resolve()
    if cold and (len(arms) != 1 or session is None or session.cache is not None or session.model is None):
        raise ValueError("cold recovery requires one arm, resident model/text and no persistent feature cache")
    if cold and root.exists() and any(root.iterdir()):
        raise ValueError("cold recovery requires a fresh isolated view/result/output directory")
    index = ConsumptionIndex(root / "input_verifications.json")
    identity = canonical_digest({"binding": binding["identity"], "resident_inputs": inputs.identity,
        "arms": arms, "cold": cold, "producer": index.identity(__file__),
        "region_operator": index.identity(Path(__file__).with_name("region_worker.py")),
        "projector": index.identity(Path(__file__).with_name("projected_views.py")), "freeze": freeze})
    path = root / "receipt.json"
    if path.is_file():
        old = read(path)
        _verified_identity(old)
        if old["status"] == "COMPLETE":
            if old["input_identity"] != identity:
                raise ValueError("completed standalone recovery inputs changed; invalidate explicitly")
            for item in old["inputs"] + old["outputs"]:
                index.identity(item["path"], item)
            return old
        path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    result = {"status": "RUNNING", "scene": inputs.scene, "arms": arms, "input_identity": identity,
        "resident_input_identity": inputs.identity, "freeze": freeze, "cold": cold, "GT_input": False,
        "persistent_feature_cache_enabled": not cold, "failed_view_replacement": False,
        "production_callable": "static_ovmap.cvpr_compact.recovery_run.recover_fc"}
    atomic_write_json(path, result)
    stats, outputs, encoding_started = {}, [], False
    try:
        scope = gpu_lease(binding, root) if lease is None else nullcontext(lease)
        with scope as active_lease:
            if (not active_lease.active or active_lease.pid != os.getpid()
                    or active_lease.binding_identity != binding["identity"] or active_lease.path != binding["gpu_lock"]):
                raise ValueError("standalone recovery requires its active own-process bound GPU lease")
            plan, requests, loaders, view_receipt, archival = {}, {}, {}, None, None
            projected_arms = tuple(arm for arm in arms if arm != "U2")
            if projected_arms:
                selected_root = root / "views" if cold else Path(views_root or Path(binding["output_root"]) / "projected_views" / inputs.scene)
                if cold and views_root is not None:
                    raise ValueError("cold projected recovery cannot consume a persistent view cache")
                view_receipt = build_projected_views(inputs.data["capture_manifest"], inputs.baseline, selected_root, index=index)
                registry, manifest = read(view_receipt["registry"]), read(view_receipt["manifest"])
                _verified_identity(registry)
                _verified_identity(manifest)
                plan.update(projected_plan(registry, manifest, projected_arms))
                selected = dict.fromkeys(rid for arm in projected_arms for rows in plan[arm].values() for rid in rows)
                for rid in selected:
                    requests[rid] = manifest["requests"][rid]
                    loaders[rid] = lambda request_id: load_projected_request(view_receipt["manifest"], request_id, index=index)
            else:
                registry = build_registry(inputs.xyz, inputs.raw, inputs.baseline.owner_ids)
            if "U2" in arms:
                archived_plan, archived_requests, archival = archived_u2_plan(binding, inputs.scene, registry, index=index)
                plan.update(archived_plan)
                requests.update(archived_requests)
                loader = archived_loader(archival, index=index)
                loaders.update({rid: loader for rid in archived_requests})
            registry_path = root / "registry.json"
            atomic_write_json(registry_path, registry)
            outputs.append(index.identity(registry_path))
            if session is None:
                cache = ContentCache(binding["parent_fc_cache_roots"], Path(binding["writable_content_cache"]) / "fc",
                    inputs.data["FC_physical_model_identity"], index=index, resolver=PathResolver(binding["path_map"]))
                session = FCSession(binding, inputs.data, index, cache=cache)
            if (session.model_key != inputs.data["FC_physical_model_identity"] or list(map(int, session.ids)) != inputs.valid_ids
                    or session.text_identity["sha256"] != inputs.data["FC_text"]["sha256"] or session.device != "cuda"):
                raise ValueError("resident FC session changed its original model/text/vocabulary or physical device")
            if cold and session.cache is not None:
                raise ValueError("cold recovery forbids persistent dense or pooled feature reads")
            session.index = index
            encoding_started = True
            features, stats = session.encode(requests, loaders)
            validate_request_outcomes(requests, features, stats)
            for targets in plan.values():
                ids = {rid for selected in targets.values() for rid in selected}
                validate_request_outcomes(dict.fromkeys(ids), {rid: features[rid] for rid in ids if rid in features}, stats)
            if any(vector.dtype != np.float32 for vector in features.values()):
                raise ValueError("standalone recovery vectors must retain original FP32 precision")
            classified = classify_regions(registry, plan, requests, features, session.text, session.ids)
            sources, source_files = {}, {}
            for arm, objects in classified.items():
                source = {"source": arm, "scene": inputs.scene, "objects": objects, "valid_ids": inputs.valid_ids,
                    "model_identity": session.model_key, "text_identity": session.text_identity,
                    "registry_identity": registry["identity"], "score_kind": "ORIGINAL_FC_COSINE",
                    "request_manifest_identity": archival["identity"] if arm == "U2" else manifest["identity"], "GT_input": False}
                source["identity"] = canonical_digest(source)
                source_path = root / "regions" / (arm + "_FC.json")
                atomic_write_json(source_path, source)
                sources[arm], source_files[arm] = source, index.identity(source_path)
                outputs.append(source_files[arm])
            feature_path = root / "feature_vectors.npz"
            ordered = sorted(features)
            _write_npz(feature_path, {"request_ids": np.asarray(ordered, dtype="U64"),
                "features": np.stack([features[rid] for rid in ordered]) if ordered else np.empty((0, session.text.shape[1]), np.float32)})
            outputs.append(index.identity(feature_path))
            exports = export_conditions(binding, inputs, registry, sources, arms, root / "predictions", index,
                                        region_receipt=stats)
            for export in exports.values():
                outputs.extend(export["files"])
            regions = {"status": "COMPLETE", "scene": inputs.scene, "arms": arms, "sources": source_files,
                "plan": plan, "outputs": list(source_files.values()), "features": index.identity(feature_path),
                "input_identity": identity, "physical_model_identity": session.model_key, **stats,
                "model_load_seconds": session.model_load_seconds, "inputs": index.entries(), "GT_input": False}
            regions["identity"] = canonical_digest(regions)
            regions_path = root / "regions/receipt.json"
            atomic_write_json(regions_path, regions)
            outputs.append(index.identity(regions_path))
            result.update(status="COMPLETE", registry=str(registry_path), registry_identity=registry["identity"],
                candidate_count=len(registry["candidates"]), sources=source_files, plan=plan, regions=regions,
                regions_receipt=str(regions_path), projected_views=view_receipt, features=index.identity(feature_path),
                archived_request_identity=archival["identity"] if archival is not None else None,
                exports=exports, outputs=outputs, **stats, model_load_seconds=session.model_load_seconds,
                source_available_additions={arm: sum(row["available"] for row in source["objects"].values())
                                           for arm, source in sources.items()})
    except BaseException as exc:
        if encoding_started:
            stats.update(getattr(session, "last_stats", {}))
            result["model_load_seconds"] = session.model_load_seconds
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", outputs=outputs, **stats)
        raise
    finally:
        result.update(inputs=index.entries(), elapsed_seconds=time.monotonic() - started)
        result["identity"] = canonical_digest({k: v for k, v in result.items() if k != "identity"})
        atomic_write_json(path, result)
        index.write_memo(root / "input_verifications.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--arm", action="append", choices=tuple(ARM_METHOD))
    args = parser.parse_args()
    binding = read(args.binding)
    inputs = load_recovery_inputs(binding, args.scene)
    result = recover_fc(binding, inputs, args.output_root, arms=args.arm)
    print(args.scene, result["status"], result["source_available_additions"], flush=True)
