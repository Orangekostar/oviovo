"""Input-identified required visual-operation unions, distinct from cache work."""

from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest


def crop_operations(model, request, shape, mask):
    height, width = shape
    x1, y1, x2, y2 = request["bbox_xyxy"]
    operations = {}
    for layer in range(3):
        xp, yp = int(.1 * layer * (x2 - x1)), int(.1 * layer * (y2 - y1))
        box = [max(0, x1 - xp), max(0, y1 - yp), min(width - 1, x2 + xp), min(height - 1, y2 + yp)]
        for foreground in (False, True):
            operation = {"kind": "image_crop", "model_processor": model, "precision": "float32",
                         "rgb": request["image_sha256"], "exclusive_box": box,
                         "foreground_mask": mask if foreground else None}
            operations[canonical_digest(operation)] = operation
    return operations


def scene_operations(binding, scene, variants):
    config = read_json(binding["scenes"][scene]["config"])
    data, root = config["scenes"][scene], Path(binding["output_root"])
    capture = read_json(data["capture"])
    frames = {f["frame_id"]: f for f in capture["frames"]}
    all_requests = {r["request_id"]: r for f in capture["frames"] for r in f["requests"]}
    native_requests = [all_requests[r] for f in capture["frames"] for r in f["native_selected_request_ids"]]
    query = read_json(Path(config["attempt_root"]) / "query" / scene / "Q_GAIN/decisions.json")
    manifest = read_json(binding["scenes"][scene]["static_manifest"]["path"])
    static = {k: v.get("request", v) for k, v in manifest["requests"].items()}
    sources = {"N0": {}, "Q_GAIN": {}, "S_SIGLIP2_AREA": {}}

    def crops(destination, model, requests, mask_key):
        for request in requests:
            destination.update(crop_operations(model, request, frames[request["frame_id"]]["image_size_hw"], request[mask_key]))

    crops(sources["N0"], config["models"]["native"]["identity"], native_requests, "native_union_mask_sha256")
    crops(sources["Q_GAIN"], config["models"]["native"]["identity"],
          [r["request"] for r in query["paid_requests"]], "native_union_mask_sha256")
    crops(sources["S_SIGLIP2_AREA"], config["models"]["siglip2"]["identity"], static.values(), "native_union_mask_sha256")
    for variant in variants:
        operations = {}
        if variant.startswith("AW_E01_"):
            operations.update(sources["Q_GAIN"])
        elif variant == "AW_E02_GLOBAL":
            crops(operations, config["models"]["siglip2"]["identity"], static.values(), "target_mask_sha256")
        elif variant == "AW_E02_SAM2":
            sam_root = root / "e02/sam2_masks" / scene
            sam = read_json(sam_root / "receipt.json")
            model = {"checkpoint": sam["model"], "code_commit": sam["code_commit"], "precision": sam["precision"]}
            for key, request in static.items():
                image = {"kind": "sam2_image", "model": model, "rgb": request["image_sha256"]}
                image_key = canonical_digest(image)
                operations[image_key] = image
                box = {"kind": "sam2_box_decode", "image": image_key, "box": list(map(float, request["bbox_xyxy"])),
                       "multimask_output": True, "normalize_coords": True}
                operations[canonical_digest(box)] = box
                row = read_json(sam_root / "requests" / (key + ".json"))
                recognition = read_json(root / "e02" / scene / variant / "requests" / (key + ".json"))
                if recognition["physical_crop_inputs"]:
                    operations.update(crop_operations(config["models"]["siglip2"]["identity"], request,
                        frames[request["frame_id"]]["image_size_hw"], row["semantic_mask_sha256"]))
        elif variant == "AW_C0_SO400M":
            source = read_json(root / "c0" / scene / (variant + ".json"))
            crops(operations, source["model_identity"], static.values(), "native_union_mask_sha256")
        elif variant.startswith("AW_E03_"):
            branch = variant.removeprefix("AW_E03_")
            asset = root.parent / "assets" / ("ovrcoat" if branch == "OVR" else "fc_frozen") / "download_receipt.json"
            checkpoint = read_json(asset)
            model = {"branch": branch, "weights": checkpoint.get("checkpoint", checkpoint.get("files")),
                     "precision": "float32", "adapter": "RGB_bilinear800_max1333_pad32_signed_global_mask"}
            for key, request in static.items():
                image = {"kind": "dense_image", "model": model, "rgb": request["image_sha256"]}
                image_key = canonical_digest(image)
                operations[image_key] = image
                row = read_json(root / "e03" / scene / variant / "requests" / (key + ".json"))
                if row["region_poolings"]:
                    pool = {"kind": "region_pool_projection", "image": image_key, "mask": request["target_mask_sha256"]}
                    operations[canonical_digest(pool)] = pool
        else:
            raise ValueError("unregistered source operation family")
        sources[variant] = operations
    result = {"scene": scene, "binding": binding["identity"], "sources": sources,
              "unit": "one crop, dense/SAM image encoding, box decode or region pooling/projection",
              "common_geometry_frontend_required_for_every_method": True,
              "text_operations_reported_separately": True, "physical_cache_savings_are_not_free_logical_cost": True}
    result["identity"] = canonical_digest(result)
    write_once(root / "operations" / scene / (result["identity"] + ".json"), result)
    return result


def method_operations(source_operations, method, pair=None):
    required = ["N0"]
    if method in {"RV_A7_COS_FIXED", "RV_A7_COS_REFIT", "AW_E04_SHORTLIST"}:
        required += ["Q_GAIN", "S_SIGLIP2_AREA"]
    elif method in {"Q_GAIN", "S_SIGLIP2_AREA"}:
        required.append(method)
    elif method == "AW_COMBO_QR":
        if not pair:
            raise ValueError("composition cost requires frozen pair")
        required += [pair["q_variant"], pair["region_variant"]]
    elif method.endswith(("_DIRECT", "_A7")):
        variant, mode = method.rsplit("_", 1)
        required.append(variant)
        if mode == "A7":
            required.append("S_SIGLIP2_AREA" if variant.startswith("AW_E01_") else "Q_GAIN")
    elif method != "N0":
        raise ValueError("unregistered method operation union")
    union = {}
    for source in required:
        union.update(source_operations[source])
    return union
