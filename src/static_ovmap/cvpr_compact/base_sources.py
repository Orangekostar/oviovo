"""Original N/Q/F evidence on new BB00 captures under the compact-task gate."""

import argparse
import os
from pathlib import Path
import pickle
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import native_readout, load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read

from .projected_views import _verified_identity
from .runtime import execute_leaf, require_frozen_execution


def existing_sources(incumbents, saved, query, text, canonical, valid_ids,
                     native_record_key, model_identity, query_receipt):
    import torch

    ids = list(map(int, valid_ids))
    if not ids or len(set(ids)) != len(ids) or any(value <= 0 for value in ids):
        raise ValueError("existing evidence requires the complete positive vocabulary")
    labels = {int(owner): int(label) for owner, label in incumbents.items()}
    if any(owner <= 0 or label not in ids for owner, label in labels.items()):
        raise ValueError("existing evidence leaves the Native incumbent registry or vocabulary")
    owners, scores = np.asarray(query["owner_ids"]), np.asarray(query["scores"])
    available = np.asarray(query["available"])
    if list(map(int, query["valid_ids"])) != ids:
        raise ValueError("final reconciled Q vocabulary differs from the frozen classifier")
    if (owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer)
            or len(np.unique(owners)) != len(owners) or np.any(owners <= 0)
            or scores.shape != (len(owners), len(ids)) or not np.isfinite(scores).all()
            or available.shape != owners.shape or available.dtype != bool):
        raise ValueError("final reconciled Q arrays are malformed")
    readout = native_readout(saved, text, canonical, tuple(ids))
    positions = {int(owner): position for position, owner in enumerate(owners)}
    tensor_text = torch.as_tensor(text, dtype=torch.float32)
    sources = {name: {"source": name, "objects": {}, "map_id": "BB00_NATIVE", "valid_ids": ids,
        "native_record_key": native_record_key, "model_identity": model_identity,
        "source_receipt": query_receipt} for name in ("N", "Q")}
    for owner in sorted(labels):
        native = readout.get(owner)
        has_native = native is not None and native["status"] == "AVAILABLE"
        native_scores = None
        if has_native:
            native_scores = torch.nn.functional.cosine_similarity(
                torch.as_tensor(native["feature"], dtype=torch.float32), tensor_text, dim=-1).numpy()
            if not np.isfinite(native_scores).all():
                raise ValueError("original Native cosine scores are not finite")
        sources["N"]["objects"][str(owner)] = {
            "available": has_native, "scores": native_scores.tolist() if has_native else None,
            "label": ids[int(native_scores.argmax())] if has_native else None,
            "used_frames": native["used_frames"] if native is not None else [],
            "retained_observations": native["retained_observations"] if native is not None else 0,
            "reason": native["status"] if native is not None else "NO_SUCCESSFUL_NATIVE_OBSERVATION",
            "score_kind": "ORIGINAL_NATIVE_COSINE_FOR_D2"}
        position = positions.get(owner)
        has_query = position is not None and bool(available[position])
        query_scores = scores[position] if has_query else None
        sources["Q"]["objects"][str(owner)] = {
            "available": has_query, "scores": query_scores.tolist() if has_query else None,
            "label": ids[int(query_scores.argmax())] if has_query else None,
            "reason": "FINAL_RECONCILED_Q_GAIN" if has_query else "NO_FINAL_RECONCILED_Q_FEATURE",
            "score_kind": "ORIGINAL_FINAL_RECONCILED_Q_COSINE"}
    for source in sources.values():
        source["identity"] = canonical_digest(source)
    return sources


def static_fc_plan(manifest, incumbents):
    _verified_identity(manifest)
    selected, excluded = set(manifest["selected_targets"]), set(manifest["excluded_targets"])
    targets = {"owner:" + str(owner) for owner in incumbents}
    if (selected & excluded or selected | excluded != targets or len(selected) > 128
            or set(manifest["views"]) != selected):
        raise ValueError("original static F cap/selection differs from the Native registry")
    plan, requests = {}, {}
    for owner in sorted(incumbents):
        target = "owner:" + str(owner)
        chosen = list(manifest["views"].get(target, []))
        if len(chosen) > 3 or len(set(chosen)) != len(chosen):
            raise ValueError("original static F must retain at most three selected requests")
        plan[str(owner)] = chosen
        for rid in chosen:
            row = manifest["requests"][rid]
            if (manifest["lineage_proofs"][rid]["target_id"] != target
                    or row["visible_target_pixels"] <= 0 or rid in requests):
                raise ValueError("original static F request leaves its proven final owner")
            requests[rid] = {**row, "visible_pixels": row["visible_target_pixels"]}
    if set(requests) != set(manifest["requests"]):
        raise ValueError("original static F manifest contains an unselected request")
    return {"F": plan}, requests


def run_native_query(job):
    from static_ovmap.backbone_wave1 import semantic_readout
    from static_ovmap.backbone_wave1.runtime import check_gpu_once
    from static_ovmap.recovery_wave2.native_worker import (
        ReadOnlyNativeCache, _adapt_native_function, validate_query_budget)
    from .native_region_worker import NATIVE_VL_SHA256

    binding = read(job["binding"])
    freeze = require_frozen_execution(binding, job["scene"])
    if job["scene"] not in binding["cohorts"]["scannet_cf18"]:
        raise ValueError("fresh N/Q acquisition is limited to the fixed missing CF18 anchors")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(binding["gpu"]):
        raise ValueError("N/Q worker must use the single bound GPU")
    bound = binding["scenes"][job["scene"]]
    if (job["model"] != bound["models"]["native"] or job["checkpoint"] != bound["checkpoint"]
            or job["gpu_lock"] != binding["gpu_lock"]):
        raise ValueError("N/Q job differs from the bound original model, scaler or GPU lock")
    import torch
    import transformers
    if (job["model"]["dtype"] != "float32" or job["model"]["torch_version"] != torch.__version__
            or job["model"]["transformers_version"] != transformers.__version__):
        raise ValueError("N/Q runtime or precision differs from the inherited actual encoder")
    root = Path(job["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(root / "input_verifications.json")
    mapping = read(job["map_receipt"])
    _verified_identity(mapping)
    if (mapping["status"] != "COMPLETE" or mapping["scene"] != job["scene"]
            or mapping["map_id"] != "BB00_NATIVE" or mapping["GT_input"]
            or not mapping["geometry_locked_before_semantics_and_labels"]
            or job["capture_manifest"] != mapping["capture_manifest"]
            or job["deferred_metadata"] != mapping["deferred_metadata"]):
        raise ValueError("N/Q requires the complete label-free same-scene Native map")
    for item in mapping["outputs"]:
        index.identity(item["path"], item)
    for path in (job["map_receipt"], job["capture_manifest"], job["deferred_metadata"],
                 __file__, semantic_readout.__file__, Path(semantic_readout.__file__).with_name("features.py")):
        index.identity(path)
    index.identity(Path(job["upstream"]) / "scripts/vl_models.py", {"sha256": NATIVE_VL_SHA256})
    for item in job["model"]["files"] + [job["model"]["text"], job["checkpoint"]]:
        index.identity(item["path"], item)
    identity = canonical_digest({"job": job, "freeze": freeze, "inputs": index.entries()})
    receipt_path = root / "native_query_receipt.json"
    if receipt_path.is_file():
        old = read(receipt_path)
        if old["status"] == "COMPLETE":
            _verified_identity(old)
            if old.get("compact_input_identity") != identity:
                raise ValueError("completed N/Q inputs changed; invalidate affected descendants explicitly")
            for item in old["compact_inputs"] + [old["native_features"], old["query_scores"], old["query_decisions"]]:
                index.identity(item["path"], item)
            return old
        receipt_path.rename(root / ("native_query_receipt.failed_" + str(time.time_ns()) + ".json"))
    resource = check_gpu_once(binding["gpu"], root / "gpu_check.json")
    if resource["occupants"]:
        raise RuntimeError("RESOURCE_BLOCK: bound N/Q GPU is occupied")
    encoders, accounting = [], []

    def factory(model, model_identity, cache_root):
        encoder = ReadOnlyNativeCache(model, model_identity, cache_root, binding["parent_native_cache_roots"],
            resolver=PathResolver(binding["path_map"]), memo=root / "input_verifications.json")
        encoders.append(encoder)
        return encoder

    def validate(result, frames):
        accounting.append(validate_query_budget(result, frames))

    run, adapter = _adapt_native_function(factory, validate)
    receipt = run(job)
    if len(encoders) != 1 or len(accounting) != 1:
        raise RuntimeError("N/Q lacks exact encoder or prescribed-budget accounting")
    for item in encoders[0].index.entries():
        index.identity(item["path"], item)
    receipt.update(compact_input_identity=identity, compact_inputs=index.entries(), freeze=freeze,
        effective_adapter=adapter, effective_worker=index.identity(__file__),
        query_budget_accounting=accounting[0], parent_caches_read_only=True,
        parent_physical_cache_hits=encoders[0].parent_hits,
        physical_forward_attempts=encoders[0].forward_attempts, new_text_forwards=0,
        final_surface_read_after_last_barrier=True)
    receipt["identity"] = canonical_digest({k: v for k, v in receipt.items() if k != "identity"})
    atomic_write_json(receipt_path, receipt)
    index.write_memo(root / "input_verifications.json")
    return receipt


def run_existing_fc(job):
    from static_ovmap.backbone_wave1.runtime import check_gpu_once, exclusive_lock
    from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
    from .region_worker import FCSession, archived_loader, classify_regions, validate_request_outcomes

    binding = read(job["binding"])
    freeze = require_frozen_execution(binding, job["scene"])
    data = read(job["context"])
    _verified_identity(data)
    if job["scene"] not in binding["cohorts"]["scannet_cf18"]:
        raise ValueError("fresh existing F is limited to the fixed missing CF18 anchors")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(binding["gpu"]):
        raise ValueError("existing F worker must use the single bound GPU")
    root = Path(job["output_root"])
    index = ConsumptionIndex(root / "input_verifications.json")
    index.identity(job["context"])
    index.identity(job["request_manifest"])
    manifest = read(job["request_manifest"])
    _verified_identity(manifest)
    native = load_prediction(data["predictions"]["NATIVE_READOUT"])
    if (native.scene_id != job["scene"] or manifest["scene_id"] != native.scene_id
            or manifest["native_record_key"] != native.record_key):
        raise ValueError("existing F manifest differs from its same-scene Native anchor")
    from static_ovmap.composition_study.object_evidence import owner_labels
    incumbents = owner_labels(native)
    plan, requests = static_fc_plan(manifest, incumbents)
    identity = canonical_digest({"binding": binding["identity"], "freeze": freeze,
        "context": data["identity"], "manifest": manifest["identity"], "plan": plan,
        "producer": index.identity(__file__), "operators": index.identity(Path(__file__).with_name("region_worker.py"))})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        old = read(receipt_path)
        _verified_identity(old)
        if old["status"] == "COMPLETE":
            if old["input_identity"] != identity:
                raise ValueError("completed static F inputs changed; invalidate explicitly")
            for item in old["inputs"] + old["outputs"]:
                index.identity(item["path"], item)
            return old
        receipt_path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    receipt = {"status": "RUNNING", "scene": job["scene"], "input_identity": identity,
        "freeze": freeze, "plan": plan, "GT_input": False, "failed_view_replacement": False}
    atomic_write_json(receipt_path, receipt)
    session = None
    try:
        cache = ContentCache(binding["parent_fc_cache_roots"], Path(binding["writable_content_cache"]) / "fc",
            data["FC_physical_model_identity"], index=index, resolver=PathResolver(binding["path_map"]))
        loader = archived_loader(manifest, index=index)
        with exclusive_lock(binding["gpu_lock"]):
            resource = check_gpu_once(binding["gpu"], root / "gpu_check.json")
            if resource["occupants"]:
                raise RuntimeError("RESOURCE_BLOCK: bound static F GPU is occupied")
            session = FCSession(binding, data, index, cache=cache)
            features, stats = session.encode(requests, {rid: loader for rid in requests})
        receipt.update(stats, model_load_seconds=session.model_load_seconds)
        validate_request_outcomes(requests, features, stats)
        registry = {"candidates": [{"raw_owner": owner} for owner in sorted(incumbents)]}
        objects = classify_regions(registry, plan, requests, features, session.text, session.ids)["F"]
        for target in manifest["excluded_targets"]:
            objects[target.removeprefix("owner:")]["reason"] = "ORIGINAL_STATIC_F_TARGET_CAP_EXCLUDED"
        source = {"source": "F", "scene": job["scene"], "map_id": "BB00_NATIVE", "objects": objects,
            "valid_ids": list(map(int, session.ids)), "native_record_key": native.record_key,
            "model_identity": session.model_key, "text_identity": session.text_identity,
            "request_manifest_identity": manifest["identity"], "target_cap": 128, "view_cap": 3,
            "target_cap_exclusions": manifest["excluded_targets"], "score_kind": "ORIGINAL_FC_COSINE"}
        source["identity"] = canonical_digest(source)
        source_path = root / "F.json"
        atomic_write_json(source_path, source)
        outputs = [index.identity(source_path)]
        if session.weight_audit is not None:
            audit_path = root / "weight_audit.json"
            atomic_write_json(audit_path, session.weight_audit)
            outputs.append(index.identity(audit_path))
        receipt.update(status="COMPLETE", source=outputs[0], outputs=outputs,
            selected_targets=manifest["selected_targets"], excluded_targets=manifest["excluded_targets"],
            selected_requests=len(requests), source_availability=sum(row["available"] for row in objects.values()),
            physical_model_identity=session.model_key, required_text_identity=session.text_identity)
    except BaseException as exc:
        if session is not None:
            receipt.update(getattr(session, "last_stats", {}), model_load_seconds=session.model_load_seconds)
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", outputs=[])
        raise
    finally:
        receipt.update(inputs=index.entries(), elapsed_seconds=time.monotonic() - started)
        receipt["identity"] = canonical_digest({k: v for k, v in receipt.items() if k != "identity"})
        atomic_write_json(receipt_path, receipt)
        index.write_memo(root / "input_verifications.json")
    return receipt


def _prepare_baseline(binding, scene, data, native_receipt, root, index):
    from plyfile import PlyData
    from static_ovmap.module_validation.scannet_study import build_native_prediction, freeze_projection
    from static_ovmap.module_validation.semantic_study import prepare_semantic_manifest

    preparation = read(data["preparation_receipt"])
    _verified_identity(preparation)
    target = preparation["raw_files"]["_vh_clean_2.ply"]
    index.identity(target["path"], target)
    vertices = PlyData.read(target["path"])["vertex"].data
    if "label" in vertices.dtype.names:
        raise ValueError("preprediction target coordinates must come from the geometry-only mesh")
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    capture_path = Path(data["capture_manifest"])
    capture = read(capture_path)
    _verified_identity(capture)
    surface_path = capture_path.parent / capture["surface"]["path"]
    index.identity(surface_path, capture["surface"])
    with np.load(surface_path, allow_pickle=False) as arrays:
        projection = freeze_projection(arrays["surface_xyz"], xyz, root / "projection")
    native_path = build_native_prediction(capture_path, Path(native_receipt["native_features"]["path"]),
        Path(data["models"]["native"]["text"]["path"]), projection, root / "anchor")
    native = load_prediction(native_path)
    manifest_path = root / "fc_requests.json"
    prepare_semantic_manifest(capture_path, native, manifest_path)
    config = {"schema_version": 1, "task_id": binding["task_id"], "attempt_root": str(root),
        "repository_root": binding["repository_root"], "cpu_threads": 4, "models": data["models"],
        "runtime": data["runtime"], "scenes": {scene: {
            "annotations": str(root / "annotations/receipt.json"), "capture": str(capture_path),
            "native_prediction": str(native_path), "native_features": native_receipt["native_features"]["path"],
            "native_geometry": native.geometry.to_dict(), "projection": str(root / "projection/manifest.json"),
            "surface": str(surface_path), "source_directory": str(root), "schedule": data["schedule"],
            "role": "fixed_cf18", "status": "BASELINE_LOCKED"}}}
    config_path = root / "config.json"
    if config_path.exists() and read(config_path) != config:
        raise ValueError("same-scene baseline configuration changed")
    atomic_write_json(config_path, config)
    context = {k: v for k, v in data.items() if k != "identity"}
    context.update(config=str(config_path), native_query_root=str(root / "native_query"),
        native_query_receipt=str(root / "native_query/native_query_receipt.json"),
        fc_root=str(root / "fc"), predictions={"NATIVE_READOUT": str(native_path)},
        availability="NATIVE_BASELINE_LOCKED_STATIC_F_REQUIRED")
    context["identity"] = canonical_digest(context)
    context_path = root / "baseline_context.json"
    atomic_write_json(context_path, context)
    return context, context_path, manifest_path


def _run_worker(binding, python, job, kind, index):
    if kind not in ("native-query", "existing-fc"):
        raise ValueError("unknown fixed base readout worker")
    root = Path(job["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    job_path = root / "job.json"
    environment = {**os.environ, "PYTHONPATH": binding["repository_root"] + "/src:" + binding["repository_root"],
        "CUDA_VISIBLE_DEVICES": str(binding["gpu"]), "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    argv = [python, "-m", "static_ovmap.cvpr_compact.base_sources", "--worker", kind, "--job", str(job_path)]
    identity = canonical_digest({"job": job, "producer": index.identity(__file__), "argv": argv})
    ledger_path = root / ("command_" + identity + ".json")
    if ledger_path.is_file():
        ledger = read(ledger_path)
        if ledger["input_identity"] != identity:
            raise ValueError("base worker command ledger identity changed")
        previous = ledger["attempts"][-1] if ledger["attempts"] else None
        if previous is not None and previous["status"] == "COMPLETE":
            if previous["argv"] != argv or previous["cwd"] != binding["repository_root"] or previous["exit_code"] != 0:
                raise ValueError("successful base worker command differs from its exact job")
            path = root / ("native_query_receipt.json" if kind == "native-query" else "receipt.json")
            receipt = read(path)
            _verified_identity(receipt)
            if receipt["status"] != "COMPLETE":
                raise RuntimeError("successful base command lacks its complete own result")
            files = (receipt["compact_inputs"] + [receipt[name] for name in
                ("native_features", "query_scores", "query_decisions")]) if kind == "native-query" else receipt["inputs"] + receipt["outputs"]
            for item in files:
                index.identity(item["path"], item)
            return {**previous, "reused_successful_command": True}
    return execute_leaf(argv, binding["repository_root"], root / "run.log", env=environment,
                        input_identity=identity, prepare=lambda: atomic_write_json(job_path, job))


def run_base_sources(binding, scene):
    freeze = require_frozen_execution(binding, scene)
    if scene not in binding["cohorts"]["scannet_cf18"]:
        return {"status": "NQF_READY", "scene": scene, "new_visual_inference": 0,
                "alias_kind": "EXACT_VERIFIED_IMMUTABLE_PARENT_NQF", "context": binding["scenes"][scene]}
    output = Path(binding["output_root"])
    root = output / "readouts" / scene / "BB00_NATIVE"
    root.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(root / "input_verifications.json")
    context_path = output / "anchor_contexts" / (scene + ".json")
    data = read(context_path)
    _verified_identity(data)
    if data["availability"] != "LABEL_FREE_CAPTURE_COMPLETE":
        raise ValueError("CF18 readouts require their own complete label-free Native context")
    for path in (context_path, __file__, data["parent_map_receipt"]):
        index.identity(path)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "freeze": freeze,
        "anchor_context": data["identity"], "inputs": index.entries()})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        old = read(receipt_path)
        _verified_identity(old)
        if old["status"] == "NQF_READY":
            if old["input_identity"] != identity:
                raise ValueError("completed base readout inputs changed; invalidate explicitly")
            for item in old["outputs"]:
                index.identity(item["path"], item)
            return old
        receipt_path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    receipt = {"status": "RUNNING", "scene": scene, "input_identity": identity, "freeze": freeze,
               "GT_input": False, "new_temperature_fit": False}
    atomic_write_json(receipt_path, receipt)
    try:
        native_root = root / "native_query"
        native_job = {"binding": str(output / "resolved_inputs.json"), "scene": scene, "map_id": "BB00_NATIVE",
            "output_root": str(native_root), "map_receipt": data["parent_map_receipt"],
            "capture_manifest": data["capture_manifest"], "deferred_metadata": data["deferred_metadata"],
            "model": data["models"]["native"], "checkpoint": data["checkpoint"],
            "upstream": data["runtime"]["upstream"], "gpu_lock": binding["gpu_lock"],
            "encoder_cache_root": str(Path(binding["writable_content_cache"]) / "native")}
        _run_worker(binding, data["runtime"]["native_perception_python"], native_job, "native-query", index)
        nq = read(native_root / "native_query_receipt.json")
        if nq["status"] != "COMPLETE":
            raise RuntimeError("new CF18 N/Q evidence is incomplete")
        context, base_context_path, request_path = _prepare_baseline(binding, scene, data, nq, root, index)
        fc_job = {"binding": str(output / "resolved_inputs.json"), "scene": scene,
            "context": str(base_context_path), "request_manifest": str(request_path), "output_root": str(root / "fc")}
        _run_worker(binding, binding["fc"]["python"], fc_job, "existing-fc", index)
        fc = read(root / "fc/receipt.json")
        _verified_identity(fc)
        if fc["status"] != "COMPLETE":
            raise RuntimeError("new CF18 static F evidence is incomplete")
        from static_ovmap.composition_study.object_evidence import owner_labels
        native = load_prediction(context["predictions"]["NATIVE_READOUT"])
        with Path(nq["native_features"]["path"]).open("rb") as handle:
            saved = pickle.load(handle)
        with np.load(data["models"]["native"]["text"]["path"], allow_pickle=False) as arrays:
            text, canonical = arrays["text_embeddings"], arrays["canonical_embeddings"]
        with np.load(nq["query_scores"]["path"], allow_pickle=False) as arrays:
            query = {name: arrays[name] for name in ("valid_ids", "owner_ids", "available", "scores")}
        sources = existing_sources(owner_labels(native), saved, query, text, canonical,
            data["models"]["native"]["valid_ids"], native.record_key, data["models"]["native"]["identity"],
            str(native_root / "native_query_receipt.json"))
        sources["F"] = read(fc["source"]["path"])
        _verified_identity(sources["F"])
        source_paths = {}
        for name, source in sources.items():
            path = root / "sources" / (name + ".json")
            atomic_write_json(path, source)
            source_paths[name] = {"path": str(path), "identity": source["identity"]}
        context.pop("identity")
        context.update(sources=source_paths, availability="NQF_READY",
            parent_readout_receipt=str(receipt_path), temperatures=binding["final_temperatures"])
        context["identity"] = canonical_digest(context)
        final_context_path = output / "contexts" / (scene + ".json")
        atomic_write_json(final_context_path, context)
        paths = [final_context_path, root / "config.json", request_path, native_root / "native_query_receipt.json",
            root / "fc/receipt.json", root / "projection/manifest.json", root / "projection/projection.npz",
            *[Path(item["path"]) for item in source_paths.values()], Path(context["predictions"]["NATIVE_READOUT"])]
        prediction_manifest = read(context["predictions"]["NATIVE_READOUT"])
        paths.append(Path(context["predictions"]["NATIVE_READOUT"]).parent / prediction_manifest["arrays"]["path"])
        paths += [Path(nq[name]["path"]) for name in ("native_features", "query_scores", "query_decisions")]
        receipt.update(status="NQF_READY", context=str(final_context_path), sources=source_paths,
            predictions=context["predictions"], outputs=[index.identity(path) for path in paths],
            query_budget_accounting=nq["query_budget_accounting"], fc_target_cap_exclusions=fc["excluded_targets"],
            source_availability={name: sum(row["available"] for row in source["objects"].values())
                                 for name, source in sources.items()},
            physical_native_Q_image_inputs=nq["physical_image_encodings"],
            physical_FC_image_inputs=fc["physical_image_encodings"], physical_FC_region_poolings=fc["physical_region_poolings"],
            geometry_only_rank_target=index.identity(read(data["preparation_receipt"])["raw_files"]["_vh_clean_2.ply"]["path"]))
    except BaseException as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", outputs=[])
        raise
    finally:
        receipt.update(inputs=index.entries(), elapsed_seconds=time.monotonic() - started)
        receipt["identity"] = canonical_digest({k: v for k, v in receipt.items() if k != "identity"})
        atomic_write_json(receipt_path, receipt)
        index.write_memo(root / "input_verifications.json")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("native-query", "existing-fc"))
    parser.add_argument("--job")
    parser.add_argument("--binding")
    parser.add_argument("--scene")
    args = parser.parse_args()
    if args.worker:
        if not args.job:
            parser.error("--worker requires --job")
        result = (run_native_query if args.worker == "native-query" else run_existing_fc)(read(args.job))
    else:
        if not args.binding or not args.scene:
            parser.error("controller requires --binding and --scene")
        result = run_base_sources(read(args.binding), args.scene)
    print(result.get("scene", result.get("map_id")), result["status"], flush=True)
