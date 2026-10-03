"""Fixed-support predictions including the matched-evidence FC-only unknowns."""

import numpy as np

from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.module_validation.evaluation import PredictionPayload
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import native_ranks

from .costs import method_costs


def fc_only_labels(source, valid_ids, incumbents):
    ids = np.asarray(valid_ids, np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("FC-only classification requires a positive ordered vocabulary")
    if set(map(int, source["objects"])) != set(incumbents):
        raise ValueError("existing F evidence must cover exactly the incumbent registry")
    labels = {}
    for owner in incumbents:
        row = source["objects"][str(owner)]
        if not row["available"]:
            labels[owner] = 0
            continue
        scores = np.asarray(row["scores"])
        if scores.shape != ids.shape or not np.isfinite(scores).all():
            raise ValueError("available F evidence has invalid full-vocabulary scores")
        labels[owner] = int(ids[int(scores.argmax())])
    return labels


def construct_output(baseline, raw, registry, recovered_labels, valid_ids, nearest,
                     matched, method, *, incumbent_labels=None, costs=None, metadata=None):
    if not baseline.locked:
        raise ValueError("the baseline must lock before constructing outputs")
    raw = np.asarray(raw)
    if (raw.shape != baseline.owner_ids.shape
            or _array_digest(raw) != registry["raw_owner_sha256"]
            or _array_digest(baseline.owner_ids) != registry["painted_owner_sha256"]
            or canonical_digest({k: v for k, v in registry.items() if k != "identity"}) != registry["identity"]):
        raise ValueError("candidate registry differs from the fixed raw/incumbent support")
    ids = set(map(int, valid_ids))
    if not ids or any(i <= 0 for i in ids):
        raise ValueError("recovery requires a positive frozen vocabulary")
    candidates = {int(row["raw_owner"]): row for row in registry["candidates"]}
    recovered = {int(owner): int(label) for owner, label in recovered_labels.items()}
    if not set(recovered) <= set(candidates):
        raise ValueError("recovered owners leave the prelocked candidate registry")
    if not set(recovered.values()) <= ids:
        raise ValueError("recovered labels must be positive classes in the frozen vocabulary")
    labels = owner_labels(baseline)
    if incumbent_labels is not None:
        replacement = {int(owner): int(label) for owner, label in incumbent_labels.items()}
        allowed = ids | ({0} if method == "CT_A5_FC_ONLY" else set())
        if set(replacement) != set(labels) or not set(replacement.values()) <= allowed:
            raise ValueError("incumbent decisions leave the original registry or allowed classes")
        labels = replacement
    owners, semantic = baseline.owner_ids.copy(), baseline.semantic_labels.copy()
    for owner, label in labels.items():
        semantic[owners == owner] = label
    for owner, label in recovered.items():
        if owner in labels:
            raise ValueError("recovery collides with incumbent support")
        support = (raw == owner) & (baseline.owner_ids == 0)
        if int(support.sum()) != candidates[owner]["residual_source_rows"]:
            raise ValueError("recovery support differs from its locked candidate registry")
        owners[support], semantic[support] = owner, label
        labels[owner] = label
    active = baseline.owner_ids > 0
    if not np.array_equal(owners[active], baseline.owner_ids[active]):
        raise ValueError("an output changed incumbent instance support")
    ranks = native_ranks(owners, labels, nearest, matched)
    payload = PredictionPayload(method, "COMBO", baseline.scene_id, baseline.geometry,
        owners, semantic, ranks, costs or {}, {
            "baseline_record_key": baseline.record_key,
            "recovery_registry_identity": registry["identity"],
            "recovered_labels": {str(k): v for k, v in recovered.items()},
            "owner_semantic_decisions": {str(k): v for k, v in sorted(labels.items())},
            "old_masks_preserved": True, "unknown_incumbent_semantics_preserve_support": True,
            "official_current_class_ranks_recomputed": True, **(metadata or {})})
    payload.lock()
    return payload


def build_method_outputs(baseline, raw, registry, sources, temperatures, valid_ids,
                         nearest, matched, recovery_sources, methods, *, cost_inventory=None,
                         recovery_costs=None):
    if set(sources) != {"N", "Q", "F"} or set(temperatures) != {"N", "Q", "F"}:
        raise ValueError("existing E must consume the original N/Q/F and final scalar vector")
    if cost_inventory is not None and cost_inventory["scene"] != baseline.scene_id:
        raise ValueError("method dependency costs belong to a different Native scene")
    incumbents = owner_labels(baseline)
    candidates = {int(row["raw_owner"]) for row in registry["candidates"]}
    ids = list(map(int, valid_ids))
    for source in [*sources.values(), *recovery_sources.values()]:
        if (canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]
                or list(map(int, source["valid_ids"])) != ids):
            raise ValueError("semantic source content or complete vocabulary changed")
    for source in sources.values():
        if set(map(int, source["objects"])) != set(incumbents):
            raise ValueError("existing E sources differ from the native incumbent registry")
    recovered = {"NONE": {}}
    for name, source in recovery_sources.items():
        if set(map(int, source["objects"])) != candidates:
            raise ValueError("recovery source differs from the prelocked common candidate registry")
        labels = {}
        for owner, row in source["objects"].items():
            if not row["available"]:
                continue
            scores = np.asarray(row["scores"])
            if scores.shape != (len(ids),) or not np.isfinite(scores).all():
                raise ValueError("available recovery evidence has invalid complete-vocabulary scores")
            label = ids[int(scores.argmax())]
            if int(row["label"]) != label:
                raise ValueError("recovery label differs from its original source argmax")
            labels[int(owner)] = label
        recovered[name] = labels
    d2, _ = fuse_readout(sources, temperatures, "D2", ids, incumbents)
    decision_identity = canonical_digest({"labels": d2, "valid_ids": ids, "temperatures": temperatures,
                                         "source_identities": {key: value["identity"] for key, value in sources.items()}})
    existing = {"NATIVE": incumbents, "D2": d2,
                "FC_ONLY_SAME_EVIDENCE": fc_only_labels(sources["F"], ids, incumbents)}
    outputs = {}
    for method in methods:
        name, recovery = method["id"], method["recovery"]
        if name in outputs or recovery not in recovered or method["existing"] not in existing:
            raise ValueError("method recipes must be unique members of the fixed evidence definitions")
        attempted = sum(len(row.get("attempted_request_ids", []))
                        for row in recovery_sources[recovery]["objects"].values()) if recovery != "NONE" else 0
        costs = {**baseline.logical_cost, "recovery_attempted_requests": attempted}
        dependency = {"status": "NOT_ACCOUNTED_SYNTHETIC_FIXTURE_ONLY"}
        if cost_inventory is not None:
            dependency = method_costs(method, cost_inventory,
                None if recovery == "NONE" else recovery_costs[recovery])
            costs.update({key: dependency[key] for key in ("native_required_crop_inputs", "FC_required_image_inputs",
                "FC_selected_mask_inputs", "known_required_image_encodings_lower_bound", "logical_attempted_requests",
                "failed_request_count", "query_attempts")})
            if dependency["required_image_encodings"] is not None:
                costs["required_image_encodings"] = dependency["required_image_encodings"]
        outputs[name] = construct_output(baseline, raw, registry, recovered[recovery], ids,
            nearest, matched, name, incumbent_labels=existing[method["existing"]],
            costs=costs, metadata={
                "dependency_costs": dependency,
                "method_recipe": method, "existing_source_identities": {key: value["identity"] for key, value in sources.items()},
                "final_temperatures": temperatures, "existing_D2_decision_identity": decision_identity,
                "recovery_source_identity": None if recovery == "NONE" else recovery_sources[recovery]["identity"]})
    shared = [outputs[name].metadata["recovered_labels"] for name in ("CT_A2_R", "CT_A3_ER", "CT_A5_FC_ONLY") if name in outputs]
    if shared and any(value != shared[0] for value in shared):
        raise RuntimeError("A2/A3/A5 must consume the identical successful G1_FC recovered set")
    return outputs
