"""GT-free multiview entity evidence and component-wide constrained aggregation."""
from itertools import combinations

import numpy as np

from .geometry_support import independent_support_frames


def frame_pair_evidence(owners, masks, depth_valid, *, whole_entity_source=False,
                        min_owner_pixels=100, min_mask_pixels=500,
                        min_depth_ratio=.9, min_geometry_coverage=.7,
                        min_owner_purity=.8, min_joint_coverage=.5, min_separate_iou=.65):
    owners, masks, depth_valid = map(np.asarray, (owners, masks, depth_valid))
    if (owners.shape != masks.shape or owners.shape != depth_valid.shape
            or not np.issubdtype(owners.dtype, np.integer)
            or not np.issubdtype(masks.dtype, np.integer)
            or np.any(owners < 0) or np.any(masks < 0)):
        raise ValueError('aligned nonnegative native owner and frame-local mask images required')
    oi, inverse_o = np.unique(owners, return_inverse=True)
    mi, inverse_m = np.unique(masks, return_inverse=True)
    table = np.bincount(inverse_o.ravel()*len(mi)+inverse_m.ravel(),
                        minlength=len(oi)*len(mi)).reshape(len(oi), len(mi))
    owner_area, mask_area = table.sum(axis=1), table.sum(axis=0)
    valid_depth = np.bincount(inverse_m.ravel(), weights=depth_valid.ravel().astype(bool),
                             minlength=len(mi))
    coverage = table[oi > 0].sum(axis=0) / mask_area
    credible = ((mi > 0) & (mask_area >= min_mask_pixels)
                & (valid_depth/mask_area >= min_depth_ratio) & (coverage >= min_geometry_coverage))
    candidates = np.flatnonzero((oi > 0) & (owner_area >= min_owner_pixels))
    result = {'positive': [], 'negative': [], 'covisible': [],
              'credible_mask_ids': mi[credible].astype(int).tolist(),
              'whole_entity_source': bool(whole_entity_source)}
    for a, b in combinations(candidates, 2):
        pair = [int(oi[a]), int(oi[b])]
        result['covisible'].append(pair)
        if not whole_entity_source:
            continue
        ma, mb = int(table[a].argmax()), int(table[b].argmax())
        if not credible[ma] or not credible[mb]:
            continue
        purity_a, purity_b = table[a, ma]/owner_area[a], table[b, mb]/owner_area[b]
        if ma == mb:
            joint = (table[a, ma]+table[b, mb])/mask_area[ma]
            if min(purity_a, purity_b) >= min_owner_purity and joint >= min_joint_coverage:
                result['positive'].append({'owners': pair, 'mask_ids': [int(mi[ma])],
                    'owner_purity': [float(purity_a), float(purity_b)], 'joint_coverage': float(joint),
                    'reason': 'same_credible_whole_entity_covers_both_native_tracks'})
        else:
            iou_a = table[a, ma]/(owner_area[a]+mask_area[ma]-table[a, ma])
            iou_b = table[b, mb]/(owner_area[b]+mask_area[mb]-table[b, mb])
            if min(iou_a, iou_b) >= min_separate_iou:
                result['negative'].append({'owners': pair, 'mask_ids': [int(mi[ma]), int(mi[mb])],
                    'owner_mask_iou': [float(iou_a), float(iou_b)],
                    'reason': 'separately_delineated_whole_entities_bidirectionally_match_native_tracks'})
    return result


def aggregate_edges(frames, poses, surface_distances, *, min_independent_frames=3,
                    min_positive_agreement=.6, max_surface_distance_m=.05):
    records = {}
    for frame, evidence in sorted(frames.items()):
        for sign in ('positive', 'negative', 'covisible'):
            for item in evidence[sign]:
                pair = tuple(sorted(item if sign == 'covisible' else item['owners']))
                row = records.setdefault(pair, {'positive': [], 'negative': [], 'covisible': []})
                row[sign].append(int(frame))
    result = {'positive': [], 'negative': [], 'rejected': []}
    for pair, row in sorted(records.items()):
        counts = {}
        independent = {}
        for sign in ('positive', 'negative', 'covisible'):
            support = independent_support_frames({f: {1: 1} for f in row[sign]}, poses, min_pixels=1)
            independent[sign] = support.get(1, {}).get('independent_frame_ids', [])
            counts[sign] = len(independent[sign])
        detail = {'owners': list(pair), 'frame_ids': row, 'independent_frame_ids': independent,
                  'surface_distance_m': surface_distances.get(pair),
                  'positive_agreement': len(row['positive'])/max(len(row['covisible']), 1)}
        if counts['negative'] >= min_independent_frames:
            result['negative'].append({**detail, 'weight': counts['negative'],
                'reason': 'independent_multiview_bidirectional_whole_entity_separation'})
        distance = surface_distances.get(pair)
        if (counts['positive'] >= min_independent_frames and distance is not None
                and distance < max_surface_distance_m
                and detail['positive_agreement'] >= min_positive_agreement):
            result['positive'].append({**detail, 'weight': counts['positive']})
        elif row['positive']:
            reasons = []
            if counts['positive'] < min_independent_frames:
                reasons.append('insufficient_independent_positive_views')
            if distance is None or distance >= max_surface_distance_m:
                reasons.append('no_measured_native_surface_contact')
            if detail['positive_agreement'] < min_positive_agreement:
                reasons.append('unstable_shared_entity_support')
            result['rejected'].append({**detail, 'reason': reasons})
    return result


def constrained_components(nodes, positive, negative):
    nodes = list(map(int, nodes))
    if len(set(nodes)) != len(nodes) or any(n <= 0 for n in nodes):
        raise ValueError('unique positive native node IDs required')
    parent = {n: n for n in nodes}
    members = {n: {n} for n in nodes}
    prohibited = set()
    for edge in [*positive, *negative]:
        if len(edge['owners']) != 2 or len(set(edge['owners'])) != 2 or any(n not in parent for n in edge['owners']):
            raise ValueError('edge must connect two known distinct native owners')
    for edge in negative:
        prohibited.add(tuple(sorted(edge['owners'])))
    accepted, rejected = [], []
    for edge in sorted(positive, key=lambda e: (-e['weight'], tuple(sorted(e['owners'])))):
        a, b = (parent[n] for n in edge['owners'])
        if a == b:
            continue
        conflict = next((list(pair) for pair in sorted(prohibited)
                         if (pair[0] in members[a] and pair[1] in members[b])
                         or (pair[1] in members[a] and pair[0] in members[b])), None)
        if conflict:
            rejected.append({'owners': edge['owners'], 'reason': 'component_wide_cannot_link',
                             'conflicting_pair': conflict})
            continue
        keep, drop = min(a, b), max(a, b)
        members[keep] |= members.pop(drop)
        for node in members[keep]:
            parent[node] = keep
        accepted.append(edge)
    return {'components': [sorted(members[k]) for k in sorted(members)],
            'accepted': accepted, 'rejected': rejected}
