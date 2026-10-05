"""Post-call scientific checks; parent recovery handles stay in this lane only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from static_ovmap.cvpr_compact.area_fallback_experiment import document, seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .views import scientific_key


@dataclass
class VerificationContext:
    reference: dict


def compare_source_objects(reference, selected, mapping):
    if reference["valid_ids"] != selected["valid_ids"] or set(reference["objects"]) != set(selected["objects"]):
        raise ValueError("source candidate registry or ordered vocabulary changed")
    maximum = 0.
    for owner,first in reference["objects"].items():
        second = selected["objects"][owner]
        for field in ("available","label","reason"):
            if first[field] != second[field]:
                raise ValueError("source "+field+" parity failed for owner "+owner)
        for field in ("attempted_request_ids","used_request_ids","failed_request_ids"):
            if [mapping[rid] for rid in first[field]] != second[field]:
                raise ValueError("source ordered request outcome changed")
        if first["available"]:
            scores_a,scores_b = np.asarray(first["scores"]),np.asarray(second["scores"])
            if not np.allclose(scores_a,scores_b,atol=1e-5,rtol=1e-5):
                raise ValueError("source score parity failed")
            maximum = max(maximum,float(np.max(np.abs(scores_a-scores_b))))
    return maximum


def load_mask(request, index):
    item = request["mask"]
    index.identity(item["path"],item)
    with np.load(item["path"],allow_pickle=False) as arrays:
        shape = tuple(map(int,arrays["shape"]))
        mask = np.unpackbits(arrays["packed"],bitorder="little",count=int(np.prod(shape))).reshape(shape).astype(bool)
    if _array_digest(mask) != request["target_mask_sha256"]:
        raise ValueError("selected full-resolution mask digest changed")
    return mask


def compare_views(context, scene, current_root, *, reference_diagnostics=None):
    if not isinstance(context,VerificationContext):
        raise TypeError("parent views are confined to a verification context")
    reference = context.reference
    index = ConsumptionIndex()
    parent_root = Path(reference["parent_root"])/"projected_views"/scene
    parent,current = (document(path/"manifest.json",index) for path in (parent_root,Path(current_root)))
    if parent["registry_identity"] != current["registry_identity"] or parent["geometry_identity"] != current["geometry_identity"]:
        raise ValueError("projected candidate registry or final raw geometry changed")
    capture = read(parent["capture_manifest"])
    shapes = {int(frame["frame_id"]):frame["image_size_hw"] for frame in capture["frames"]}
    model = reference["contexts"][scene]["model_identity"]
    ids = reference["contexts"][scene]["valid_ids"]
    mapping,keys = {},[]
    if set(parent["views"]) != set(current["views"]):
        raise ValueError("candidate owners with no selected views changed")
    for owner,old_ids in parent["views"].items():
        new_ids = current["views"][owner]
        if len(old_ids) != len(new_ids):
            raise ValueError("ordered Top-3 view count changed")
        for old,new in zip(old_ids,new_ids,strict=True):
            first,second = parent["requests"][old],current["requests"][new]
            key_a = scientific_key(first,ids,model,shapes[int(first["frame_id"])])
            key_b = scientific_key(second,ids,model,shapes[int(second["frame_id"])])
            if key_a != key_b or not np.array_equal(load_mask(first,index),load_mask(second,index)):
                raise ValueError("selected mask/frame/bbox/rank scientific content changed")
            mapping[old] = new
            keys.append({"old_request_id":old,"new_request_id":new,"scientific_key":key_a})
        if current["g1"][owner] != new_ids[:1] or parent["g1"][owner] != old_ids[:1]:
            raise ValueError("G1 is no longer the exact ordered G3 prefix")
    if len(mapping) != len(set(mapping.values())):
        raise ValueError("scientific request mapping is not bijective")
    diagnostic_path = Path(reference_diagnostics) if reference_diagnostics else parent_root/"frame_diagnostics.json"
    if not diagnostic_path.is_file():
        raise FileNotFoundError("MISSING_PARENT_FRAME_DIAGNOSTICS: "+str(diagnostic_path))
    first = read(diagnostic_path)["frames"]
    second = read(Path(current_root)/"frame_diagnostics.json")["frames"]
    if len(first) != len(second):
        raise ValueError("completed camera count changed")
    for old,new in zip(first,second,strict=True):
        if old["frame_id"] != new["frame_id"] or old["counts"] != new["counts"]:
            raise ValueError("candidate per-frame count/bbox/admissibility parity failed at frame "+str(old["frame_id"]))
    return seal({"status":"PASS","scene":scene,"registry_identity":parent["registry_identity"],
        "all_candidate_frame_statistics_exact":True,"ordered_top3_exact":True,"selected_mask_bytes_exact":True,
        "frames":len(first),"mapping":mapping,"scientific_keys":keys,
        "diagnostic_scope":list(dict.fromkeys(row.get("diagnostic_scope","full_frame") for row in second))})


def compare_prediction(first_path, second_path):
    first,second = load_prediction(first_path),load_prediction(second_path)
    if (first.prediction_key != second.prediction_key or first.geometry != second.geometry or not np.array_equal(first.owner_ids,second.owner_ids)
            or not np.array_equal(first.semantic_labels,second.semantic_labels)
            or first.instance_ranks != second.instance_ranks
            or first.metadata["recovered_labels"] != second.metadata["recovered_labels"]
            or first.metadata["owner_semantic_decisions"] != second.metadata["owner_semantic_decisions"]
            or first.metadata["final_temperatures"] != second.metadata["final_temperatures"]):
        raise ValueError("whole prediction owners/semantics/ranks/export context parity failed")
    return {"parent_record_key":first.record_key,"current_record_key":second.record_key,
        "parent_prediction_key":first.prediction_key,"current_prediction_key":second.prediction_key,
        "owners_sha256":_array_digest(second.owner_ids),"semantic_sha256":_array_digest(second.semantic_labels),
        "ranks_exact":True,"rank_strings_exact":all(f"{a[1]:.6f}"==f"{b[1]:.6f}" for a,b in zip(first.instance_ranks,second.instance_ranks,strict=True)),
        "export_context_exact":True}


def compare_cold(context, result, current_root):
    reference,scene,arm = context.reference,result["scene"],result["arm"]
    parent = Path(reference["parent_root"])
    v2 = Path(reference["v2_results_root"])
    index = ConsumptionIndex()
    if arm == "U2":
        original_path = parent/"recovery"/scene/"receipt.json"
        original = read(original_path)
        if "U2" not in original.get("sources",{}):
            original_path = parent/"recovery"/(scene+"_U2")/"receipt.json"
            original = document(original_path,index)
        mapping = {rid:rid for rid in result["requests"]}
        projection_parity = {"status":"GENUINE_ARCHIVED_U2_POLICY","mapping":mapping}
        first_stats = original["requests"]
        sources = original["sources"]
        feature_item = original["features"]
    else:
        projection_parity = compare_views(context,scene,Path(current_root)/"views")
        mapping = projection_parity["mapping"]
        original = document(v2/"regions"/scene/"receipt.json",index)
        first_stats,sources,feature_item = original["requests"],original["sources"],original["features"]
    first_source = document(sources[arm]["path"],index,sources[arm])
    second_source = document(result["sources"][arm]["path"],index,result["sources"][arm])
    max_scores = compare_source_objects(first_source,second_source,mapping)
    attempted = {rid for rows in result["plan"][arm].values() for rid in rows}
    expected_old = {rid for row in first_source["objects"].values() for rid in row["attempted_request_ids"]}
    if {mapping[rid] for rid in expected_old} != attempted or set(result["stats"]["requests"]) != attempted:
        raise ValueError("cold successful/failed/attempted request sets changed")
    index.identity(feature_item["path"],feature_item)
    with np.load(feature_item["path"],allow_pickle=False) as arrays:
        old_features = dict(zip(arrays["request_ids"].tolist(),arrays["features"],strict=True))
    item = result["features"]
    index.identity(item["path"],item)
    with np.load(item["path"],allow_pickle=False) as arrays:
        new_features = dict(zip(arrays["request_ids"].tolist(),arrays["features"],strict=True))
    maximum = {"normal":0.,"fallback":0.}
    max_norm_error = 0.
    for old in expected_old:
        new = mapping[old]
        before,after = first_stats[old],result["stats"]["requests"][new]
        for key in ("status","target_mask_sha256","input_tensor_key","image_content_key"):
            if before[key] != after[key]:
                raise ValueError("cold request "+key+" changed")
        if bool(before.get("fallback",False)) != bool(after.get("fallback",False)):
            raise ValueError("original normal/fallback outcome changed")
        support = before.get("original_support",before.get("dense_mask_support"))
        if support is None and arm != "U2":
            raise ValueError("v2 projected source omitted its hard-support audit")
        if support is not None and support != after.get("original_support",after.get("dense_mask_support")):
            raise ValueError("original hard dense-mask support changed")
        if (old in old_features) != (new in new_features):
            raise ValueError("cold feature success set changed")
        if old not in old_features:
            continue
        first,second = old_features[old],new_features[new]
        if second.dtype != np.float32 or not np.allclose(first,second,atol=1e-5,rtol=1e-5):
            raise ValueError("cold original FP32 unit-vector parity failed")
        group = "fallback" if before.get("fallback",False) else "normal"
        maximum[group] = max(maximum[group],float(np.max(np.abs(first-second))))
        max_norm_error = max(max_norm_error,abs(float(np.linalg.norm(second))-1.))
    lock = document(v2/"predictions"/scene/"receipt.json",index)
    predictions = {method:compare_prediction(lock["predictions"][method],row["manifest"])
        for method,row in result["exports"].items()}
    return seal({"status":"PASS","scene":scene,"arm":arm,"implementation":result["implementation"],
        "discrete_output_parity":"EXACT","max_feature_absolute_error":maximum,"max_unit_norm_error":max_norm_error,
        "max_score_absolute_error":max_scores,"selected_requests":len(attempted),"projection":projection_parity,
        "predictions":predictions,"new_GPU_inference_in_verification":0,
        "legacy_U2_hard_support":"not recorded by cached parent; exact full mask/input/operator/vector checks" if arm=="U2" else None})
