"""Split a native parent only with independently repeated whole-entity surface evidence."""
import numpy as np

from .geometry_support import independent_support_frames


def split_from_views(mask_ids, frame_ids, poses, *, min_seed_atoms=20, min_child_atoms=50,
                     min_child_fraction=.1, min_seed_agreement=.8, min_independent_frames=3,
                     min_atom_votes=2, min_atom_agreement=.8, min_assigned_fraction=.7):
    masks = np.asarray(mask_ids)
    if (masks.ndim != 2 or len(frame_ids) != len(masks) or len(set(frame_ids)) != len(frame_ids)
            or not np.issubdtype(masks.dtype, np.integer) or np.any(masks < -1)):
        raise ValueError('frame-aligned native atom mask IDs required; -1 means invisible')
    result = {'accepted': False, 'atom_children': np.zeros(masks.shape[1], dtype=np.int64),
              'reason': 'no_credible_multientity_anchor'}
    candidates = []
    for index, row in enumerate(masks):
        labels, counts = np.unique(row[row > 0], return_counts=True)
        visible = int((row >= 0).sum())
        keep = (counts >= min_seed_atoms) & (counts >= visible*min_child_fraction)
        labels, counts = labels[keep], counts[keep]
        if 2 <= len(labels) <= 4 and counts.sum() >= .8*visible:
            candidates.append((int(counts.sum()), float(counts.min()/counts.sum()), -int(frame_ids[index]), index, labels))
    if not candidates:
        return result
    _, _, _, anchor_index, anchor_labels = max(candidates, key=lambda x: x[:3])
    seeds = [masks[anchor_index] == label for label in anchor_labels]
    matches = {}
    for index, row in enumerate(masks):
        labels_for_children = []
        for seed in seeds:
            seen = row[seed & (row >= 0)]
            labels, counts = np.unique(seen[seen > 0], return_counts=True)
            if not len(labels):
                break
            best = int(counts.argmax())
            if counts[best] < min_seed_atoms or counts[best]/max(len(seen), 1) < min_seed_agreement:
                break
            labels_for_children.append(int(labels[best]))
        if (len(labels_for_children) == len(seeds)
                and len(set(labels_for_children)) == len(seeds)):
            matches[int(frame_ids[index])] = (index, labels_for_children)
    support = independent_support_frames({f: {1: 1} for f in matches}, poses, min_pixels=1)
    independent = support.get(1, {}).get('independent_frame_ids', [])
    result.update({'anchor_frame_id': int(frame_ids[anchor_index]),
                   'anchor_mask_ids': anchor_labels.astype(int).tolist(),
                   'support_frame_ids': sorted(matches), 'independent_frame_ids': independent})
    if len(independent) < min_independent_frames:
        result['reason'] = 'insufficient_independent_whole_entity_separation'
        return result
    votes = np.zeros((len(seeds), masks.shape[1]), dtype=np.int32)
    seen_counts = np.zeros(masks.shape[1], dtype=np.int32)
    frame_matching = {}
    for frame in independent:
        index, labels = matches[frame]
        row = masks[index]
        seen_counts += row >= 0
        for child, label in enumerate(labels):
            votes[child] += row == label
        frame_matching[str(frame)] = labels
    best = votes.argmax(axis=0)
    best_votes = votes[best, np.arange(masks.shape[1])]
    supported = ((best_votes >= min_atom_votes)
                 & (best_votes/np.maximum(seen_counts, 1) >= min_atom_agreement))
    children = np.where(supported, best+1, 0)
    sizes = [int((children == child).sum()) for child in range(1, len(seeds)+1)]
    assigned_fraction = float(np.count_nonzero(children)/max(len(children), 1))
    result.update({'frame_mask_correspondence': frame_matching, 'child_atom_counts': sizes,
                   'assigned_fraction': assigned_fraction, 'residual_atom_count': int((children == 0).sum())})
    if (assigned_fraction < min_assigned_fraction or any(n < min_child_atoms
            or n < masks.shape[1]*min_child_fraction for n in sizes)):
        result['reason'] = 'insufficient_stable_surface_coverage'
        return result
    result.update({'accepted': True, 'atom_children': children,
                   'reason': 'independent_multiview_whole_entity_surface_partition'})
    return result
