"""Thin fixed-owner construction allowing unknown without removing support."""
import numpy as np
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.evaluation import PredictionPayload
from static_ovmap.module_validation.scannet_study import native_ranks

def validate_eligible_replacement(before,after,eligible):
    if set(before)!=set(after) or any(after[o]!=before[o] for o in before if o not in set(eligible)):
        raise ValueError('Replacement leaves the fixed eligible owner domain')

def fixed_owner_output(baseline,labels,ids,nearest,matched,method):
    original=owner_labels(baseline)
    labels={int(o):int(label) for o,label in labels.items()}
    if not baseline.locked or set(labels)!=set(original) or not set(labels.values())<=set(map(int,ids))|{0}: raise ValueError('Fixed-owner output has invalid labels/domain')
    semantic=baseline.semantic_labels.copy()
    for owner,label in labels.items(): semantic[baseline.owner_ids==owner]=label
    ranks=native_ranks(baseline.owner_ids,labels,nearest,matched)
    payload=PredictionPayload(method,'COMBO',baseline.scene_id,baseline.geometry,baseline.owner_ids,semantic,ranks,dict(baseline.logical_cost),dict(source_record_key=baseline.record_key,owner_partition_preserved=True,unknown_support_preserved=True))
    payload.lock()
    return payload
