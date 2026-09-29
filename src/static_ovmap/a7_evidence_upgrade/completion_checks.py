"""Final adapter contract checks over the new wave's real receipts only."""

from collections import Counter
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _array_digest


def check(binding):
    root, index = Path(binding["output_root"]), InputIndex()
    index.identity(__file__)
    summary, total_requests = {}, 0

    def read(path):
        index.identity(path)
        return read_json(path)

    for scene, bound in binding["scenes"].items():
        manifest = read(bound["static_manifest"]["path"])
        if len(manifest["selected_targets"]) > 128 or any(len(v) > 3 for v in manifest["views"].values()):
            raise ValueError("frozen static target/view budget exceeded")
        requests = {k: v.get("request", v) for k, v in manifest["requests"].items()}
        total_requests += len(requests)
        original_q = read(root / "e01" / scene / "receipt.json")
        if original_q["physical_image_forwards"] != 0 or original_q["retention_order_exact"] is not True or original_q["maximum_original_score_error"] > 1e-12:
            raise ValueError("Q retention/zero-inference/original score parity failed")
        sam = read(root / "e02/sam2_masks" / scene / "receipt.json")
        selected_frames = {v["frame_id"] for v in requests.values()}
        if sam["physical_image_encodings"] > len(selected_frames) or sam["physical_box_decodes"] > len(requests):
            raise ValueError("SAM2 image/box cache budget exceeded")
        sam_ok, parity, dense = 0, Counter(), {}
        for request_id, request in requests.items():
            row = read(root / "e02/sam2_masks" / scene / "requests" / (request_id + ".json"))
            if row["status"] == "MASK_COMPLETE":
                quality = np.asarray(row["quality"], np.float64)
                finite = np.isfinite(quality)
                if not finite.any() or row["selected_index"] != int(np.where(finite, quality, -np.inf).argmax()):
                    raise ValueError("SAM2 mask choice differs from fixed quality/tie rule")
                if row["original_request"] != request or row["original_union_crop_parity"] is not True:
                    raise ValueError("SAM2 changed original request or historical crop pixels")
                sam_ok += 1
            for mode in ("GLOBAL", "SAM2"):
                row = read(root / "e02" / scene / ("AW_E02_" + mode) / "requests" / (request_id + ".json"))
                if row["status"] == "COMPLETE":
                    if row["raw_crop_pixel_parity"] is not True:
                        raise ValueError("raw crop pixel reuse parity failed")
                    parity[mode] += 1
        for branch in ("FC_FROZEN", "OVR"):
            folder = root / "e03" / scene / ("AW_E03_" + branch)
            weights = read(folder / "weight_audit.json")
            if not weights["strict"] or weights["random_active_parameters"] or weights["loaded_key_count"] != 543 or weights["active_trunk_changed_elements"] <= 0:
                raise ValueError("region checkpoint branch/strict loading audit failed")
            frames, successful, empty = Counter(), 0, 0
            for request_id, request in requests.items():
                row = read(folder / "requests" / (request_id + ".json"))
                frames[request["frame_id"]] += row.get("image_encodings", 0)
                if row["status"] == "COMPLETE":
                    if row["dense_support"] <= 0 or row["region_poolings"] != 1:
                        raise ValueError("successful region lacks original target or nonempty dense support")
                    index.expected_output(row["arrays_path"], row)
                    with np.load(row["arrays_path"], allow_pickle=False) as arrays:
                        shape = tuple(arrays["original_shape"])
                        original = np.unpackbits(arrays["original_bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)
                    if _array_digest(original) != request["target_mask_sha256"] or int(original.sum()) != row["original_pixels"]:
                        raise ValueError("actual packed region mask differs from original target")
                    successful += 1
                empty += row.get("error") == "EMPTY_DENSE_MASK_SUPPORT"
            if any(n > 1 for n in frames.values()):
                raise ValueError("region backbone recomputed a frame for multiple targets")
            dense[branch] = {"successful_regions": successful, "empty_dense_support": empty,
                             "frame_encodings": sum(frames.values()), "strict_loaded_keys": weights["loaded_key_count"]}
        targets = read(root / "e04" / scene / "targets.json")
        if targets["new_source_outputs_used"] or len(targets["selected"]) > 64:
            raise ValueError("E04 target freeze used new-source predictions or exceeded budget")
        for row in targets["selected"]:
            if len(row["candidates"]) > 4 or row["a7_winner"] not in row["candidates"] or row["request_ids"] != manifest["views"][f"owner:{row['owner_id']}"]:
                raise ValueError("E04 frozen shortlist/view contract differs")
        summary[scene] = {"static_requests": len(requests), "SAM2_verified_choices": sam_ok,
                          "raw_pixel_parity": dict(parity), "region": dense, "E04_targets": len(targets["selected"])}
    if total_requests > 3840:
        raise ValueError("per-encoder static request allowance exceeded")
    result = {"status": "REAL_ADAPTER_CONTRACTS_VERIFIED", "scenes": summary, "static_requests_per_encoder": total_requests,
              "E01_new_image_forwards": 0, "inputs": index.entries()}
    result["identity"] = canonical_digest(result)
    write_once(root / "review/adapter_contracts.json", result)
    return result
