"""Prediction-only frozen candidates and isolated fusion interventions.

No GT, evaluator class filter, or model inference belongs in this module.
Canonical index is stable within one run; it is not cross-run object identity.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _readonly(array):
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FrozenCandidates:
    masks: np.ndarray
    areas: np.ndarray
    original_labels: np.ndarray
    released_scores: np.ndarray
    native_owners: np.ndarray
    records: tuple
    ovi_count: int
    ious: np.ndarray


def build_frozen_candidates(owners, readout, masks, labels, scores, query_ids, *,
                            scene_id='unspecified', prediction_run='unspecified',
                            ovi_readout_id='unbound_legacy_readout'):
    owners, masks, labels, scores, query_ids = map(np.asarray, (owners, masks, labels, scores, query_ids))
    if (owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer) or np.any(owners < 0)
            or labels.ndim != 1 or masks.shape != (len(labels), len(owners)) or masks.dtype != bool
            or scores.shape != labels.shape or query_ids.shape != labels.shape
            or not np.isfinite(scores).all() or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels <= 0) or not np.issubdtype(query_ids.dtype, np.integer)
            or len(np.unique(query_ids)) != len(query_ids)):
        raise ValueError('aligned finite proposals and unique integer query IDs required')
    ids, counts = np.unique(owners, return_counts=True)
    eligible = {int(i): int(n) for i, n in zip(ids, counts) if i > 0 and readout.get(str(i)) is not None}
    order = sorted(eligible, key=lambda i: (-eligible[i], i))
    sf_order = np.argsort(-scores, kind='stable')
    bank = np.empty((len(order)+len(labels), len(owners)), dtype=bool)
    records, class_ids, release = [], [], []
    for index, owner in enumerate(order):
        observation = readout[str(owner)]
        if int(observation['class_id']) <= 0:
            raise ValueError('qualified OVI candidates require a positive class')
        bank[index] = owners == owner
        records.append({'source': 'OVI', 'native_owner_id': owner, 'spaceformer_query_id': None,
            'original_t0_row': None, 'source_query_ids': tuple(observation['selected_query_ids']),
            'source_mask_reference': f'native_owners=={owner}', 'best_owner_id': None, 'best_owner_iou': 0.})
        class_ids.append(int(observation['class_id'])); release.append(float(eligible[owner]))
    for row in sf_order:
        index = len(records); bank[index] = masks[row]
        area = int(masks[row].sum(dtype=np.int64))
        overlap_owners, sizes = np.unique(owners[masks[row]], return_counts=True)
        matches = [(int(i), int(n)/(eligible[int(i)]+area-int(n)))
                   for i, n in zip(overlap_owners, sizes) if int(i) in eligible]
        best, iou = max(matches, key=lambda x: (x[1], -x[0])) if matches else (None, 0.)
        records.append({'source': 'SpaCeFormer', 'native_owner_id': None,
            'spaceformer_query_id': int(query_ids[row]), 'original_t0_row': int(row),
            'source_query_ids': (), 'source_mask_reference': f'T0.masks[{int(row)}]',
            'best_owner_id': best, 'best_owner_iou': iou})
        class_ids.append(int(labels[row])); release.append(float(scores[row]))
    areas = bank.sum(axis=1, dtype=np.int64)
    ious = np.zeros((len(bank), len(bank)), dtype=np.float64)
    for i in range(len(bank)):
        ious[i, i] = float(areas[i] > 0)
        for j in range(i):
            if i < len(order):  # qualified native owner masks are disjoint
                continue
            intersection = int(np.count_nonzero(bank[i] & bank[j]))
            union = int(areas[i]+areas[j]-intersection)
            ious[i, j] = ious[j, i] = intersection/union if union else 0.
    for i, record in enumerate(records):
        identity = record['native_owner_id'] if record['source'] == 'OVI' else record['spaceformer_query_id']
        record.update(scene_id=scene_id, prediction_run_id=prediction_run, canonical_index=i,
            candidate_id=f"{scene_id}/{prediction_run}/{record['source']}/{identity}",
            ovi_readout_id=ovi_readout_id, original_class_id=class_ids[i], source_area=int(areas[i]))
    return FrozenCandidates(_readonly(bank), _readonly(areas), _readonly(np.array(class_ids, dtype=np.int64)),
        _readonly(np.array(release, dtype=np.float64)), _readonly(owners.copy()),
        tuple(MappingProxyType(r) for r in records), len(order), _readonly(ious))


def apply_label_reuse(pool, *, enabled=True, reuse_iou=.5, ovi_labels=None):
    if not 0 < reuse_iou <= 1:
        raise ValueError('reuse threshold must be in (0,1]; use enabled=False to disable')
    labels = pool.original_labels.copy()
    if ovi_labels is not None:
        if set(ovi_labels) != {r['native_owner_id'] for r in pool.records[:pool.ovi_count]}:
            raise ValueError('readout control must preserve OVI eligibility')
        for i, record in enumerate(pool.records[:pool.ovi_count]):
            labels[i] = ovi_labels[record['native_owner_id']]
    owner_to_index = {r['native_owner_id']: i for i, r in enumerate(pool.records[:pool.ovi_count])}
    events = []
    for i, record in enumerate(pool.records):
        best = record['best_owner_id']
        borrowed = best if enabled and best is not None and record['best_owner_iou'] >= reuse_iou else None
        if borrowed is not None:
            labels[i] = labels[owner_to_index[borrowed]]
        events.append({'borrowed_from': None if borrowed is None else pool.records[owner_to_index[borrowed]]['candidate_id'],
            'reused_owner': borrowed, 'label_actually_changed': bool(labels[i] != pool.original_labels[i]),
            'final_class_id': int(labels[i])})
    return labels, events


def apply_fusion_nms(pool, *, enabled=True, nms_iou=.7, sf_first=False):
    if not 0 < nms_iou <= 1:
        raise ValueError('NMS threshold must be in (0,1]; use enabled=False to disable')
    visit = list(range(len(pool.records)))
    if sf_first:
        visit = visit[pool.ovi_count:]+visit[:pool.ovi_count]
    accepted, edges = [], [None]*len(visit)
    for rank, i in enumerate(visit):
        suppressor = next((j for j in accepted if pool.ious[i, j] >= nms_iou), None) if enabled else None
        kept = pool.areas[i] > 0 and suppressor is None
        if kept:
            accepted.append(i)
        edges[i] = {'nms_visit_rank': rank, 'kept': bool(kept), 'suppressed_by': suppressor,
                    'suppression_iou': None if suppressor is None else float(pool.ious[i, suppressor]),
                    'source_empty': bool(pool.areas[i] == 0)}
    return np.array(sorted(accepted), dtype=np.int64), edges


def serialize_scores(scores):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError('finite confidence required')
    strings = [f'{float(x):.6f}' for x in scores]
    return strings, np.array([float(x) for x in strings])


def assignment_scores(pool, kept, labels, rule='raw'):
    result = pool.areas.astype(np.float64)
    if rule == 'class_norm':
        for label in np.unique(labels[kept]):
            ids = kept[labels[kept] == label]
            result[ids] /= max(float(result[ids].max()), 1.)
    elif rule not in ('raw', 'ovi_fill'):
        raise ValueError('unknown assignment rule')
    return result


def resolve_native_owners(masks, kept, priorities, *, source_priority=None):
    masks, kept, priorities = map(np.asarray, (masks, kept, priorities))
    if (masks.ndim != 2 or masks.dtype != bool or priorities.shape != (len(masks),)
            or not np.isfinite(priorities).all() or kept.ndim != 1
            or not np.issubdtype(kept.dtype, np.integer) or len(np.unique(kept)) != len(kept)
            or np.any(kept < 0) or np.any(kept >= len(masks))):
        raise ValueError('aligned mask bank, canonical IDs and finite priorities required')
    group = np.zeros(len(masks), dtype=int) if source_priority is None else np.asarray(source_priority)
    if group.shape != priorities.shape:
        raise ValueError('source priority shape differs')
    order = kept[np.lexsort((kept, -priorities[kept], group[kept]))]
    owners = np.zeros(masks.shape[1], dtype=np.int64)
    for i in order:
        owners[masks[i] & (owners == 0)] = int(i)+1
    return owners


def fixed_regions(pool):
    n = pool.masks.shape[1]
    count = np.zeros(n, dtype=np.int32)
    has_ovi, has_sf = np.zeros(n, bool), np.zeros(n, bool)
    low = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    high = np.zeros(n, dtype=np.int64)
    for i, mask in enumerate(pool.masks):
        count += mask
        (has_ovi if i < pool.ovi_count else has_sf)[:] |= mask
        label = pool.original_labels[i]
        low[mask] = np.minimum(low[mask], label); high[mask] = np.maximum(high[mask], label)
    source = has_ovi.astype(np.int16)+2*has_sf.astype(np.int16)
    multiplicity = np.minimum(count, 2).astype(np.int16)
    agreement = np.where(count == 0, 0, np.where(low == high, 1, 2)).astype(np.int16)
    regions = source*9+multiplicity*3+agreement
    names, mult_names, agree_names = ['neither', 'OVI-only', 'SpaCeFormer-only', 'both'], ['zero', 'one', 'multiple'], ['none', 'unanimous', 'conflicting']
    registry = {int(code): {'source_support': names[int(code)//9],
        'candidate_multiplicity': mult_names[(int(code)%9)//3], 'class_agreement': agree_names[int(code)%3]}
        for code in np.unique(regions)}
    return regions, registry, count
