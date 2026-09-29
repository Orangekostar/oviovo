"""Pinned box-only SAM2 image worker, sharing one encoding per selected frame."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from src.static_ovmap.module_validation.region_evidence import native_crops
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.semantic_models import _load_request

from .crops import recognition_crops


def run_masks(binding, scene, assets, gpu, *, smoke=False):
    import torch

    assets, index = Path(assets), InputIndex()
    spec = read_json(binding["spec"])
    if scene not in binding["scenes"] or (smoke and scene not in spec["datasets"]["calibration"]):
        raise ValueError("SAM2 scene outside authorized scope")
    code = assets / "code"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if commit != spec["sam2"]["commit"]:
        raise ValueError("SAM2 source pin differs")
    download = read_json(assets / "download_receipt.json")
    weight = index.identity(download["checkpoint"]["path"], download["checkpoint"])
    sys.path.insert(0, str(code))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    index.identity(__file__)
    index.identity(Path(__file__).with_name("crops.py"))
    bound = binding["scenes"][scene]
    manifest_path = Path(bound["static_manifest"]["path"])
    index.identity(manifest_path, bound["static_manifest"])
    manifest = read_json(manifest_path)
    if canonical_digest({k: v for k, v in manifest.items() if k != "identity"}) != manifest["identity"]:
        raise ValueError("original static request manifest changed")
    index.identity(manifest["capture"]["path"], manifest["capture"])
    capture = read_json(manifest["capture"]["path"])
    capture_root = Path(manifest["capture"]["path"]).parent
    frames = {row["frame_id"]: row for row in capture["frames"]}
    requests = {k: v.get("request", v) for k, v in manifest["requests"].items()}
    ordered = sorted(requests, key=lambda k: (requests[k]["frame_id"], k))
    if smoke:
        ordered = ordered[:1]
    output = Path(binding["output_root"]) / "e02" / ("sam2_smoke" if smoke else "sam2_masks") / scene
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "requests": ordered,
        "manifest": manifest["identity"], "model": weight, "commit": commit,
        "precision": "bfloat16_autocast", "torch": torch.__version__, "code": index.identity(__file__),
        "crop_code": index.identity(Path(__file__).with_name("crops.py"))})
    final_path = output / "receipt.json"
    if final_path.exists():
        previous = read_json(final_path)
        if previous["identity"] != identity:
            raise ValueError("SAM2 completed run changed")
        for row in previous["inputs"] + previous["outputs"]:
            index.identity(row["path"], row)
        return previous
    config = read_json(bound["config"])
    lock_path = Path(config["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock")
    outputs, records = [], []
    images, decodes, model_load_seconds, inference_seconds = 0, 0, 0., 0.
    started = time.monotonic()
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), output)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        model = build_sam2(spec["sam2"]["config"], weight["path"], device="cuda")
        predictor = SAM2ImagePredictor(model)
        torch.cuda.synchronize()
        model_load_seconds = time.monotonic() - load_start
        parameter_count = sum(p.numel() for p in model.parameters())
        current_frame = None
        for request_id in ordered:
            request = requests[request_id]
            path = output / "requests" / (request_id + ".json")
            if path.exists():
                row = read_json(path)
                if row["job_identity"] != identity:
                    raise ValueError("SAM2 partial request identity changed")
                for artifact in row["outputs"]:
                    index.identity(artifact["path"], artifact)
                records.append(row)
                outputs += [index.identity(path), *row["outputs"]]
                continue
            frame = frames[request["frame_id"]]
            index.identity(capture_root / frame["rgb_path"], {"sha256": request["image_sha256"]})
            value = _load_request(capture_root, frame, request,
                fresh_masks=manifest["requests"][request_id].get("masks"))
            old = native_crops(value["rgb"], value["target"], value["union"], value["bbox"])
            unchanged = recognition_crops(value["rgb"], value["bbox"], value["union"])
            if any(not np.array_equal(a, b) for a, b in zip(old.legacy_six, unchanged["six"], strict=True)):
                raise ValueError("real original union crop parity failed")
            inference_start = time.monotonic()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if current_frame != request["frame_id"]:
                        predictor.set_image(value["rgb"])
                        current_frame = request["frame_id"]
                        images += 1
                    decodes += 1
                    masks, quality, _ = predictor.predict(box=np.asarray(value["bbox"], np.float32),
                        multimask_output=True, return_logits=False, normalize_coords=True)
                torch.cuda.synchronize()
                quality = np.asarray(quality, np.float64).reshape(-1)
                if masks.shape != (len(quality), *value["target"].shape) or not np.isfinite(quality).any():
                    raise RuntimeError("SAM2 output shape or quality invalid")
                selected = int(np.where(np.isfinite(quality), quality, -np.inf).argmax())
                semantic = np.asarray(masks[selected], bool)
                pixels = int(semantic.sum())
                union = int(np.count_nonzero(semantic | value["target"]))
                try:
                    crops = recognition_crops(value["rgb"], value["bbox"], semantic)
                    support, rejection = crops["foreground_pixels"], None
                except ValueError as error:
                    support, rejection = None, str(error)
                mask_path = path.with_suffix(".npz")
                _write_npz(mask_path, {"mask_bits": np.packbits(semantic), "shape": np.asarray(semantic.shape), "quality": quality})
                row = {"status": "MASK_COMPLETE", "job_identity": identity, "request_id": request_id,
                    "original_request": request, "selected_index": selected,
                    "quality": [float(v) if np.isfinite(v) else None for v in quality],
                    "semantic_mask_sha256": _array_digest(semantic), "mask_pixels": pixels,
                    "original_pixels": int(value["target"].sum()),
                    "iou_with_original": float(np.count_nonzero(semantic & value["target"]) / union) if union else 0.,
                    "foreground_pixels": support, "recognition_rejection": rejection,
                    "original_union_crop_parity": True, "outputs": [index.identity(mask_path)]}
            except RuntimeError as error:
                torch.cuda.synchronize()
                row = {"status": "UNAVAILABLE_TECHNICAL_FAILURE", "job_identity": identity,
                       "request_id": request_id, "error": str(error), "outputs": []}
                torch.cuda.empty_cache()
            elapsed = time.monotonic() - inference_start
            inference_seconds += elapsed
            row["elapsed_seconds"] = elapsed
            write_once(path, row)
            records.append(row)
            outputs += [index.identity(path), *row["outputs"]]
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    receipt = {"status": "SMOKE_COMPLETE" if smoke else "MASK_REQUESTS_COMPLETE", "identity": identity,
        "scene": scene, "requests": len(records), "successful_masks": sum(r["status"] == "MASK_COMPLETE" for r in records),
        "model": weight, "code_commit": commit, "parameter_count": parameter_count,
        "precision": "bfloat16_autocast", "physical_image_encodings": images, "physical_box_decodes": decodes,
        "model_load_seconds": model_load_seconds, "inference_seconds": inference_seconds,
        "wall_seconds": time.monotonic() - started, "gpu": str(gpu),
        "peak_gpu_allocated_bytes": allocated, "peak_gpu_reserved_bytes": reserved,
        "inputs": [v for v in index.entries() if v["path"] not in {o["path"] for o in outputs}], "outputs": outputs,
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}
    if smoke and receipt["successful_masks"] != 1:
        raise RuntimeError("real SAM2 smoke failed; inspect persisted request failure")
    write_once(final_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = run_masks(read_json(args.binding), args.scene, args.assets, args.gpu, smoke=args.smoke)
    print({k: result[k] for k in ("status", "scene", "requests", "successful_masks", "physical_image_encodings", "physical_box_decodes")}, flush=True)


if __name__ == "__main__":
    main()
