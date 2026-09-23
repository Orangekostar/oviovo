"""Bound per-request frozen visual inference, shared by semantic study scenes."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import sha256_file
from .boundary_jobs import file_identity, verify_capture
from .contracts import atomic_write_json, canonical_digest
from .native_capture import _array_digest, _write_npz
from .region_evidence import (
    OfficialWowRunner,
    WowRegionAdapter,
    clean_wow_category_response,
    map_generated_name,
    native_crops,
)
from .rgb_siglip import FrozenSiglipBackend
from .scannet_runtime import _tree_inputs, reusable_job


def _load_request(root: Path, frame: dict, request: dict, *, fresh_masks=None) -> dict:
    rgb = np.array(Image.open(root / frame["rgb_path"]).convert("RGB"))
    request_id = request["request_id"]
    if fresh_masks is None:
        with np.load(root / frame["request_arrays"]["path"], allow_pickle=False) as arrays:
            target, union = arrays[f"{request_id}_target"], arrays[f"{request_id}_union"]
    else:
        if file_identity(fresh_masks["path"]) != fresh_masks:
            raise ValueError("fresh final-mask request payload changed")
        with np.load(fresh_masks["path"], allow_pickle=False) as arrays:
            target, union = arrays["target"], arrays["union"]
    if (_array_digest(target) != request["target_mask_sha256"]
            or _array_digest(union) != request["native_union_mask_sha256"]):
        raise ValueError("captured request mask hash changed")
    with np.load(root / frame["depth_path"], allow_pickle=False) as arrays:
        depth = arrays["depth_m"]
    entities = np.array(Image.open(root / frame["panoptic_path"]))
    entity_ids, counts = np.unique(entities[target], return_counts=True)
    local = entities == entity_ids[np.argmax(counts)]
    if not np.array_equal(target | local, union):
        raise ValueError("native local/global union cannot be reproduced")
    x1, y1, x2, y2 = request["bbox_xyxy"]
    return {"rgb": rgb, "target": target, "union": union, "bbox": (x1, y1, x2, y2),
        "stats": {"request_mask_pixels": int(target.sum()),
            # The native bbox stores inclusive extrema; image slicing remains
            # its original exclusive-upper convention in native_crops().
            "bbox_pixels": (x2 - x1 + 1) * (y2 - y1 + 1),
            "depth_valid_fraction": float(np.mean(np.isfinite(depth[target]) & (depth[target] > 0))),
            "local_global_iou": float(np.count_nonzero(target & local) / np.count_nonzero(union)),
            "foreground_fraction": float(np.count_nonzero(union[y1:y2, x1:x2]) / ((x2 - x1) * (y2 - y1)))}}


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise RuntimeError("visual encoder returned invalid crop features")
    return vectors / norms


def _mask_support(backend, mask: np.ndarray, geometries) -> list[int]:
    images = [np.repeat(mask[y1:y2, x1:x2, None].astype(np.uint8) * 255, 3, axis=2)
              for x1, y1, x2, y2 in geometries]
    processed = backend.processor(images=images, return_tensors="np", do_normalize=False, do_rescale=False)
    return [int(np.any(image > 0, axis=0).sum()) for image in processed["pixel_values"]]


def _name_embedder(model_path: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    cache = {}

    def embed(values):
        missing = list(dict.fromkeys(value for value in values if value not in cache))
        for start in range(0, len(missing), 32):
            texts = missing[start:start + 32]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
                weights = tokens["attention_mask"].unsqueeze(-1)
                features = ((hidden * weights).sum(1) / weights.sum(1)).numpy()
            cache.update(zip(texts, features, strict=True))
        return np.stack([cache[value] for value in values])

    return embed


def wow_mapping_with_strengths(response: str, names: tuple[str, ...], embed) -> dict:
    mapping = asdict(map_generated_name(response, names, clean_response=clean_wow_category_response, embed_text=embed))
    if mapping["method"] == "EXACT":
        # Exact-name recognition fixes the label, while selector strengths still
        # need real within-MiniLM cosines rather than fabricated one-hot scores.
        vectors = _unit(embed((mapping["cleaned_generation"], *names)))
        scores = vectors[1:] @ vectors[0]
        ordered = np.sort(scores)
        mapping["similarities"] = scores.tolist()
        mapping["top1_top2_gap"] = float(ordered[-1] - ordered[-2]) if len(scores) > 1 else 0.0
    return mapping


def encode_semantic_requests(manifest_path: Path, model_id: str, config: dict, output: Path,
                             *, device: str = "cuda") -> dict:
    """One loaded visual model; exact input identities and failures persist per request."""
    import torch
    import transformers

    if model_id not in {"native", "siglip2", "wow"}:
        raise ValueError("unlisted semantic model")
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if manifest["identity"] != canonical_digest({key: value for key, value in manifest.items() if key != "identity"}):
        raise ValueError("semantic manifest identity changed")
    capture_path = Path(manifest["capture"]["path"])
    if sha256_file(capture_path) != manifest["capture"]["sha256"]:
        raise ValueError("semantic capture identity changed")
    capture = verify_capture(capture_path, allow_skipped=True)
    root = capture_path.parent
    frames = {frame["frame_id"]: frame for frame in capture["frames"]}
    static = manifest.get("artifact_type") == "OVIMAP_STATIC_FINAL_MASK_REQUESTS"
    requests = {key: value["request"] if static else value for key, value in manifest["requests"].items()}
    model_path = Path(config[model_id + "_model"])
    inputs = _tree_inputs(model_path)
    inputs += [file_identity(path) for path in (manifest_path, Path(__file__),
        Path(__file__).with_name("region_evidence.py"), Path(__file__).with_name("rgb_siglip.py"))]
    if model_id == "wow":
        inputs += _tree_inputs(Path(config["wow_code"]), "*.py") + _tree_inputs(Path(config["name_model"]))
    native_text_path = Path(config["native_text_cache"])
    inputs.append(file_identity(native_text_path))
    with np.load(native_text_path, allow_pickle=False) as text:
        names = tuple(text["class_names"].tolist())
    identity = canonical_digest({"inputs": inputs, "model_id": model_id, "device": device,
        "schema": 1, "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "precision": "bfloat16" if model_id == "wow" else "float32"})
    receipt_path = output / "receipt.json"
    if reusable_job(receipt_path, identity):
        return json.loads(receipt_path.read_text())
    if receipt_path.exists():
        raise ValueError("completed semantic inference changed; choose a new output root")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if model_id == "wow" and manifest["requests"]:
        runner = OfficialWowRunner.from_local(str(model_path), official_repo_path=config["wow_code"], device=device)
        backend = WowRegionAdapter(runner, mode="official_combined")
        embed = _name_embedder(config["name_model"])
    elif manifest["requests"]:
        backend = FrozenSiglipBackend.from_local(str(model_path), device=device)
    load_seconds = time.monotonic() - started if manifest["requests"] else 0.0
    index_path = output / "request_index.json"
    atomic_write_json(index_path, {"input_identity": identity, "request_ids": sorted(manifest["requests"])})
    outputs, ledger = [file_identity(index_path)], []
    physical_attempts, physical_crops, physical_generations = 0, 0, 0
    for request_id, request in sorted(requests.items(), key=lambda item: (item[1]["frame_id"], item[0])):
        path = output / "requests" / f"{request_id}.json"
        request_identity = canonical_digest({"model_identity": identity, "request": request})
        if reusable_job(path, request_identity):
            row = json.loads(path.read_text())
            ledger.append(row)
            outputs += [file_identity(path), *row["outputs"]]
            continue
        if path.exists():
            raise ValueError("partial semantic request changed; choose a new output root")
        value = _load_request(root, frames[request["frame_id"]], request,
            fresh_masks=manifest["requests"][request_id]["masks"] if static else None)
        physical_attempts += 1
        row = {"status": "COMPLETE", "input_identity": request_identity, "request_id": request_id,
               "stats": value["stats"], "crop_inputs": 0, "background_crop_inputs": 0, "generations": 0}
        arrays = {}
        attempt_started = time.monotonic()
        try:
            if model_id == "wow":
                # Count actual generated token IDs, including EOS, at the
                # author's inputs_embeds-only language-model boundary.
                original = runner.model.language_model.generate
                tokens = {}

                def counted_generate(*args, _tokens=tokens, _original=original, **kwargs):
                    _tokens["attempted"] = True
                    result = _original(*args, **kwargs)
                    sequence = result.sequences if hasattr(result, "sequences") else result
                    _tokens["generated_tokens"] = int(sequence.shape[-1])
                    _tokens["prompt_embedding_tokens"] = int(kwargs["inputs_embeds"].shape[1])
                    return result

                runner.model.language_model.generate = counted_generate
                try:
                    result = backend.classify(value["rgb"], value["target"])
                finally:
                    runner.model.language_model.generate = original
                    row["generations"] = int(tokens.get("attempted", False))
                    row.update(tokens)
                row.update(status=result.status, raw_generation=result.raw_generation,
                    original_mask_support=result.original_mask_support, final_mask_support=result.final_mask_support,
                    trace=dict(result.trace), representation_survived=result.final_mask_support > 0)
                if result.status == "COMPLETE":
                    if not clean_wow_category_response(result.raw_generation).strip():
                        row["status"] = "UNAVAILABLE_TECHNICAL_FAILURE"
                        row["error"] = "EMPTY_CLEANED_CATEGORY_RESPONSE"
                    else:
                        row["mapping"] = wow_mapping_with_strengths(result.raw_generation, names, embed)
            else:
                crops = native_crops(value["rgb"], value["target"], value["union"], value["bbox"])
                row["crop_inputs"] = 6
                six_started = time.monotonic()
                vectors = _unit(backend.encode_images(crops.legacy_six))
                row["six_crop_seconds"] = time.monotonic() - six_started
                arrays["feature"] = vectors.mean(axis=0)
                background_available = True
                if model_id == "native":
                    row["background_crop_inputs"] = 3
                    background_started = time.monotonic()
                    try:
                        background = _unit(backend.encode_images(crops.background))
                    except RuntimeError as error:
                        background_available = False
                        row["background_error"] = str(error)
                        background = np.zeros((3, vectors.shape[1]), dtype=vectors.dtype)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    row["background_seconds"] = time.monotonic() - background_started
                    vectors = np.concatenate((vectors, background))
                arrays["vectors"] = vectors
                row.update(crop_geometries=[list(box) for box in crops.geometries],
                    target_processor_support=_mask_support(backend, value["target"], crops.geometries),
                    union_processor_support=_mask_support(backend, value["union"], crops.geometries),
                    background_scale_usable=[background_available and bool(np.any(~value["target"][y1:y2, x1:x2]))
                        for x1, y1, x2, y2 in crops.geometries],
                    background_usable=background_available and bool(any(np.any(~value["target"][y1:y2, x1:x2])
                        for x1, y1, x2, y2 in crops.geometries)), representation_survived=True)
        except RuntimeError as error:
            row.update(status="UNAVAILABLE_TECHNICAL_FAILURE", error=str(error))
            arrays = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row["elapsed_seconds"] = time.monotonic() - attempt_started
        physical_crops += row["crop_inputs"] + row["background_crop_inputs"]
        physical_generations += row["generations"]
        array_path = path.with_suffix(".npz")
        _write_npz(array_path, arrays)
        row["outputs"] = [file_identity(array_path)]
        row["arrays_path"] = str(array_path)
        # Transport COMPLETE means this attempt is fully recorded; inference
        # availability remains separate and never becomes teacher success.
        row["inference_status"] = row["status"]
        row["status"] = "COMPLETE"
        atomic_write_json(path, row)
        ledger.append(row)
        outputs += [file_identity(path), *row["outputs"]]
    result = {"status": "COMPLETE", "input_identity": identity, "inputs": inputs, "outputs": outputs,
        "model_id": model_id, "model_load_seconds": load_seconds,
        "request_count": len(ledger), "successful_requests": sum(row["inference_status"] == "COMPLETE" for row in ledger),
        "crop_inputs": sum(row["crop_inputs"] for row in ledger),
        "background_crop_inputs": sum(row["background_crop_inputs"] for row in ledger),
        "generations": sum(row["generations"] for row in ledger),
        "request_seconds": sum(row["elapsed_seconds"] for row in ledger),
        "physical_attempts_this_invocation": physical_attempts,
        "physical_crop_inputs_this_invocation": physical_crops,
        "physical_generations_this_invocation": physical_generations,
        "elapsed_seconds_this_invocation": time.monotonic() - started, "GT_input": False}
    atomic_write_json(receipt_path, result)
    return result


def load_semantic_records(output: Path) -> dict:
    output = Path(output)
    receipt = json.loads((output / "receipt.json").read_text())
    if not reusable_job(output / "receipt.json", receipt["input_identity"]):
        raise ValueError("semantic model output is incomplete or changed")
    index = json.loads((output / "request_index.json").read_text())
    records = {}
    for request_id in index["request_ids"]:
        path = output / "requests" / f"{request_id}.json"
        row = json.loads(path.read_text())
        if not reusable_job(path, row["input_identity"]):
            raise ValueError("semantic request payload changed")
        with np.load(row["arrays_path"], allow_pickle=False) as arrays:
            records[row["request_id"]] = {**row, "status": row["inference_status"],
                **{key: arrays[key] for key in arrays.files}}
    return records
