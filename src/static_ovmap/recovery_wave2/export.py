"""New expanded payloads on fixed coordinates; no old-mask replacement."""

import numpy as np

from static_ovmap.module_validation.evaluation import PredictionPayload
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import native_ranks


def expanded_prediction(baseline, raw, registry, recovered_labels, valid_ids,
                        nearest, matched, method, costs, *, incumbent_labels=None, metadata=None):
    if not baseline.locked:
        raise ValueError("baseline prediction must lock before recovery")
    raw = np.asarray(raw)
    if (raw.shape != baseline.owner_ids.shape or _array_digest(raw) != registry["raw_owner_sha256"]
            or _array_digest(baseline.owner_ids) != registry["painted_owner_sha256"]):
        raise ValueError("recovery registry does not match the frozen baseline support")
    candidate_ids = {row["raw_owner"] for row in registry["candidates"]}
    recovered_labels = {int(owner): int(label) for owner, label in recovered_labels.items()}
    if not set(recovered_labels) <= candidate_ids:
        raise ValueError("recovered owner is outside the locked candidate registry")
    ids = set(map(int, valid_ids))
    if 0 in ids or not set(recovered_labels.values()) <= ids:
        raise ValueError("recovered classes leave the frozen vocabulary")
    owners, semantic = baseline.owner_ids.copy(), baseline.semantic_labels.copy()
    labels = {int(owner): int(semantic[np.flatnonzero(owners == owner)[0]])
              for owner in np.unique(owners) if owner > 0}
    if incumbent_labels is not None:
        if set(incumbent_labels) != set(labels) or not set(incumbent_labels.values()) <= ids:
            raise ValueError("incumbent weight decisions leave the original registry")
        labels.update({int(owner): int(label) for owner, label in incumbent_labels.items()})
        for owner, label in labels.items():
            semantic[owners == owner] = label
    mapping = {}
    for owner, label in recovered_labels.items():
        if owner in labels:
            raise ValueError("recovery candidate collides with an incumbent owner")
        support = (raw == owner) & (baseline.owner_ids == 0)
        if not support.any():
            raise ValueError("recovery candidate has no residual support")
        owners[support], semantic[support] = owner, label
        labels[owner], mapping[str(owner)] = label, owner
    ranks = native_ranks(owners, labels, nearest, matched)
    payload = PredictionPayload(method, "COMBO", baseline.scene_id, baseline.geometry,
        owners, semantic, ranks, costs, {
            "baseline_record_key": baseline.record_key, "recovery_registry": registry["identity"],
            "recovered_raw_to_output": mapping, "old_masks_preserved": True,
            "old_classes_preserved": incumbent_labels is None,
            "official_current_class_ranks_recomputed": True, **(metadata or {})})
    payload.lock()
    return payload
