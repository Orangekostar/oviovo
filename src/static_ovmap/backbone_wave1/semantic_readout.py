"""Deferred native processing and frozen causal Q on each fresh map."""

import argparse
from dataclasses import asdict
import pickle
from pathlib import Path
import sys
import subprocess
import time

import numpy as np

from static_ovmap.module_validation.boundary_jobs import file_identity
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _write_npz
from static_ovmap.module_validation.semantic_models import _load_request
from .binding import read
from .features import TensorEncoderCache
from .runtime import exclusive_lock


def reconstruct_native(metadata, records):
    """Filter successful observations, then apply the original saved top-10 order."""
    saved = {}
    for row in metadata["rows"]:
        owner = int(row["owner"])
        valid = [i for i, frame in enumerate(row["frame_id"]) if (frame, owner) in records]
        if not valid:
            continue
        areas = np.asarray(row["vis_area"], np.int64)[valid]
        features = np.stack([records[(row["frame_id"][i], owner)] for i in valid]).astype(np.float32)
        if metadata["view_select_strategy"] == "viewcov":
            retained = np.arange(len(valid))
        else:
            retained = np.argsort(areas)[-metadata["max_top_views"]:]
        selected = [valid[i] for i in retained]
        saved[owner] = {"feat": features[retained], "vis_area": areas[retained],
            "frame_id": [row["frame_id"][i] for i in selected],
            "pose": [np.asarray(row["pose"][i], np.float32) for i in selected],
            "box_2d": [tuple(row["box_2d"][i]) for i in selected],
            "color": np.asarray(row["color"], np.uint8)}
    return saved


def run_native_query(job):
    import torch
    from transformers import AutoImageProcessor

    from static_ovmap.module_validation.query_gain_policy import QueryFeatureStandardizer, predict_query_gain
    from static_ovmap.module_validation.query_models import NativeQueryLoader
    from static_ovmap.module_validation.query_pipeline import reconcile_export
    from static_ovmap.module_validation.query_state import AcquisitionPayload, FeatureStore, update_cached_class_scores
    from static_ovmap.module_validation.query_study import CapturedFrames, replay_captured
    from static_ovmap.module_validation.rgb_siglip import FrozenSiglipBackend

    root, capture_path = Path(job["output_root"]), Path(job["capture_manifest"])
    root.mkdir(parents=True, exist_ok=True)
    capture, metadata = read(capture_path), read(job["deferred_metadata"])
    replay_frames = CapturedFrames(capture_path)
    loader = NativeQueryLoader(replay_frames, {"native_model": job["model"]["path"]}, root / "query_lineage_cache")
    actual_files = {item["path"]: item for item in loader.inputs}
    for expected_file in job["model"]["files"]:
        if actual_files.get(expected_file["path"]) != expected_file:
            raise ValueError("frozen native model file identity changed")
    if file_identity(job["model"]["text"]["path"]) != job["model"]["text"]:
        raise ValueError("frozen native text identity changed")
    sys.path.insert(0, str(Path(job["upstream"]) / "scripts"))
    import vl_models
    vl_models.siglip_model_list["siglip-l-16-384"] = job["model"]["path"]
    model_identity = canonical_digest({"files": job["model"]["files"], "torch": torch.__version__,
                                      "precision": "float32", "encoder": "SIGLIP_LARGE_PATCH16_384"})
    receipt = {"status": "RUNNING", "capture": file_identity(capture_path),
               "map_id": job["map_id"], "native_requests": [], "Q_budget": 200,
               "GT_input": False, "new_temperature_fit": False, "model_identity": model_identity}
    receipt["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3], text=True).strip()
    started = time.monotonic()
    with exclusive_lock(job["gpu_lock"]), torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        encoder = None
        try:
            begin = time.monotonic()
            height, width = capture["frames"][0]["image_size_hw"]
            native_model = vl_models.VLModel("siglip-l-16-384", (height, width), "cuda", precision="fp32")
            native_model.siglip_model.requires_grad_(False)
            if any(p.dtype != torch.float32 for p in native_model.siglip_model.parameters() if p.is_floating_point()):
                raise ValueError("deferred native model changed FP32 precision")
            receipt["model_load_seconds"] = time.monotonic() - begin
            encoder = TensorEncoderCache(native_model.siglip_model, model_identity, job["encoder_cache_root"])
            observations = {}
            expected = {(int(frame), int(row["owner"])): tuple(row["box_2d"][i])
                        for row in metadata["rows"] for i, frame in enumerate(row["frame_id"])}
            selected = []
            for frame in capture["frames"]:
                requests = {request["request_id"]: request for request in frame["requests"]}
                for request_id in frame["native_selected_request_ids"]:
                    selected.append((frame, request_id, requests[request_id]))
            for frame, request_id, request in selected:
                owner = int(request["target_id"].removeprefix("owner:"))
                if expected[(frame["frame_id"], owner)] != tuple(request["bbox_xyxy"]):
                    raise ValueError("native metadata/request crop mismatch")
                value = _load_request(capture_path.parent, frame, request)
                begin = time.monotonic()
                row = {"request_id": request_id, "frame_id": frame["frame_id"], "owner": owner,
                       "lineage": request["lineage"], "source_map_version": request["source_map_version"]}
                try:
                    feature = native_model.encode_image_with_bbox(value["rgb"], value["union"], tuple(request["bbox_xyxy"]))
                    feature = np.asarray(feature, np.float32)
                    if feature.ndim != 1 or not np.isfinite(feature).all():
                        raise RuntimeError("native feature vector is malformed")
                    observations[(frame["frame_id"], owner)] = feature
                    row.update(status="COMPLETE", encoder_content=encoder.last_call)
                except RuntimeError as exc:
                    row.update(status="UNAVAILABLE_TECHNICAL_FAILURE", error=str(exc))
                    torch.cuda.empty_cache()
                row["elapsed_seconds"] = time.monotonic() - begin
                receipt["native_requests"].append(row)
            if selected and not observations:
                raise RuntimeError("all native visual requests failed; core evidence is unavailable")
            saved = reconstruct_native(metadata, observations)
            native_feature_path = root / "native_features.pkl"
            native_feature_path.write_bytes(pickle.dumps(saved, protocol=pickle.HIGHEST_PROTOCOL))
            receipt["native_features"] = file_identity(native_feature_path)
            receipt["native_required_content"] = dict(encoder.required_content)
            receipt["native_physical_image_encodings"] = encoder.physical_encodings
            receipt["native_elapsed_seconds"] = time.monotonic() - started

            # The Q ranker sees only its current frame and debited acquisitions.
            processor = AutoImageProcessor.from_pretrained(job["model"]["path"], local_files_only=True, use_fast=False)
            loader.backend = FrozenSiglipBackend(model=native_model.siglip_model,
                processor=processor, tokenizer=None, device="cuda")
            checkpoint = torch.load(job["checkpoint"]["path"], map_location="cpu", weights_only=False)
            if checkpoint["status"] != "COMPLETE":
                raise ValueError("Q_GAIN checkpoint is not the frozen complete parent")
            if file_identity(job["checkpoint"]["path"]) != job["checkpoint"]:
                raise ValueError("Q_GAIN checkpoint identity changed")
            scaler = QueryFeatureStandardizer(**checkpoint["scaler_state_dict"])
            predictor = lambda features: predict_query_gain(checkpoint["state_dict"], features)
            with np.load(job["model"]["text"]["path"], allow_pickle=False) as arrays:
                text = arrays["text_embeddings"]
                ids = arrays["valid_ids"]
            query_aliases = {}
            def paid_load(candidate):
                encoder.last_call = None
                payload = loader(candidate)
                lineage_path = Path(loader.used_receipts[candidate.request_id]["path"])
                lineage_row = read(lineage_path)
                if payload.physical_cache_hit:
                    content = lineage_row.get("backbone_encoder_content")
                    if payload.feature is not None and content is None:
                        raise ValueError("query cache lacks its physical-content alias")
                    if content is not None:
                        content_receipt = read(content["cache_receipt"])
                        encoder.index.identity(content_receipt["arrays"]["path"], content_receipt["arrays"])
                        encoder.required_content[content["content_identity"]] = content["required_crop_encodings"]
                else:
                    content = encoder.last_call
                    lineage_row["backbone_encoder_content"] = content
                    atomic_write_json(lineage_path, lineage_row)
                query_aliases[candidate.request_id] = {"frame_id": candidate.frame_id,
                    "lineage": replay_frames.current["requests"][candidate.request_id].lineage,
                    "encoder_content": content,
                    "base_loader_cache_hit": payload.physical_cache_hit,
                    "loader_elapsed_seconds": lineage_row["elapsed_seconds"]}
                return AcquisitionPayload(payload.feature, payload.attempted_crop_inputs,
                                          payload.inference_seconds, payload.failure_reason,
                                          physical_cache_hit=bool(payload.physical_cache_hit or
                                              (content is not None and content["physical_cache_hit"])))
            store = FeatureStore(paid_load)
            result = replay_captured(replay_frames, "Q_GAIN", store, text, budget=200,
                                     predictor=predictor, standardizer=scaler)
            if result["state"].logical_ledger.attempts != 200:
                raise ValueError("Q_GAIN did not spend the prescribed 200 attempted requests")
            with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as surface:
                reconcile_export(result, surface, text)
            owners = sorted(result["state"].objects)
            scores, features, available = [], [], []
            for owner in owners:
                obj = result["state"].objects[owner]
                update_cached_class_scores(result["state"], owner, text)
                scores.append(np.zeros(len(ids)) if obj.cached_scores is None else obj.cached_scores)
                vectors = [item.feature for item in obj.features]
                if vectors:
                    vector = np.average(np.stack(vectors), axis=0,
                                        weights=[item.overlap_pixels for item in obj.features])
                    vector /= np.linalg.norm(vector)
                else:
                    vector = np.zeros(text.shape[1])
                features.append(vector)
                available.append(obj.cached_scores is not None)
            query_path = root / "query_scores.npz"
            _write_npz(query_path, {"owner_ids": np.asarray(owners, np.int64), "scores": np.asarray(scores),
                "features": np.asarray(features), "available": np.asarray(available, bool), "valid_ids": ids})
            atomic_write_json(root / "query_decisions.json", {"frames": result["decisions"],
                "lineage": result["lineage"].frame_diagnostics, "logical": asdict(result["state"].logical_ledger),
                "aliases": query_aliases, "final_surface_read_after_last_barrier": True,
                "retained_feature_requests": {str(k): [f.request_id for f in obj.features]
                                               for k, obj in result["state"].objects.items()}})
            receipt.update(status="COMPLETE", query_scores=file_identity(query_path),
                query_decisions=file_identity(root / "query_decisions.json"),
                query_logical_ledger=asdict(result["state"].logical_ledger),
                standalone_required_content=dict(encoder.required_content))
            required_seconds = {key: read(Path(job["encoder_cache_root"]) / model_identity / (key + ".json"))["elapsed_seconds"]
                                for key in encoder.required_content}
            cpu_seconds = sum(max(0., row["elapsed_seconds"] - row.get("encoder_content", {}).get("call_elapsed_seconds", 0.))
                              for row in receipt["native_requests"] if row["status"] == "COMPLETE")
            cpu_seconds += sum(max(0., row["loader_elapsed_seconds"] -
                (row["encoder_content"] or {}).get("call_elapsed_seconds", 0.)) for row in query_aliases.values())
            receipt["required_content_encoder_seconds"] = required_seconds
            receipt["attributable_standalone_seconds"] = receipt["model_load_seconds"] + cpu_seconds + sum(required_seconds.values())
            receipt["standalone_timing_missing_components"] = ["causal_ranker_CPU_time_not_separately_instrumented"]
        except BaseException as exc:
            receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            receipt["physical_image_encodings"] = 0 if encoder is None else encoder.physical_encodings
            receipt["physical_encoder_calls"] = 0 if encoder is None else encoder.physical_calls
            receipt["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            receipt["worker_identity"] = file_identity(__file__)
            atomic_write_json(root / "native_query_receipt.json", receipt)
            if encoder is not None:
                encoder.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    run_native_query(read(parser.parse_args().job))
