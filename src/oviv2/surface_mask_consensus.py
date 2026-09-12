"""CROVE local-patch adaptation of MaskClustering observation/containment rules.

Uses independent frame columns, excludes undersegmented observers, and keeps
source-owner residuals. This is not the original system or its benchmark output.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def construct_mask_graph(observations, *, minimum_cloud_size=8):
    labels = np.asarray(observations)
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("nonnegative patch-by-independent-frame mask IDs required")
    keys, row_parts, col_parts = [], [], []
    for frame in range(labels.shape[1]):
        nodes = np.flatnonzero(labels[:, frame] > 0)
        unique, inverse = np.unique(labels[nodes, frame], return_inverse=True)
        row_parts.append(nodes)
        col_parts.append(inverse + len(keys))
        keys.extend((frame, int(mask)) for mask in unique)
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, np.int64)
    cols = np.concatenate(col_parts) if col_parts else np.empty(0, np.int64)
    incidence = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(labels), len(keys))
    )
    sizes = np.asarray(incidence.sum(0)).ravel()
    overlap = (incidence.T @ incidence).tocsr()
    mask_frames = np.array([frame for frame, _ in keys], np.int32)
    undersegmented = sizes < minimum_cloud_size
    contain_rows, contain_cols = [], []
    for mask in range(len(keys)):
        start, stop = overlap.indptr[mask : mask + 2]
        other, counts = overlap.indices[start:stop], overlap.data[start:stop]
        frames = mask_frames[other]
        totals = np.bincount(frames, weights=counts, minlength=labels.shape[1])
        maximum = np.zeros(labels.shape[1])
        np.maximum.at(maximum, frames, counts)
        visible = totals >= 0.3 * sizes[mask]
        contained = visible & (maximum >= 0.8 * totals) & (totals > 0)
        split = visible & ~contained
        undersegmented[mask] |= (
            visible.sum() == 0 or split.sum() / max(1, visible.sum()) > 0.3
        )
        # Count each frame once, even if maxima tie; a tie cannot exceed 0.8.
        for frame in np.flatnonzero(contained):
            candidates = other[(frames == frame) & (counts == maximum[frame])]
            contain_rows.append(mask)
            contain_cols.append(int(candidates.min()))
    contained = sparse.csr_matrix(
        (np.ones(len(contain_rows), bool), (contain_rows, contain_cols)),
        shape=(len(keys), len(keys)),
    )
    contained = contained.multiply((~undersegmented)[None, :]).astype(bool).tocsr()
    contained.eliminate_zeros()
    # An undersegmented observer supplies neither support nor an opposition vote.
    visible = np.zeros((len(keys), labels.shape[1]), bool)
    row, col = contained.nonzero()
    visible[row, mask_frames[col]] = True
    return {
        "mask_keys": keys,
        "incidence": incidence,
        "overlap": overlap,
        "sizes": sizes,
        "contained": contained,
        "visible": visible,
        "undersegmented": undersegmented,
    }


def consensus_edges(contained, visible, mask_frames, *, threshold=0.9, minimum_views=2):
    """Common-frame support / common observers; OR multiple masks within a frame."""
    contained = sparse.csr_matrix(contained, dtype=bool)
    visible = np.asarray(visible, bool)
    mask_frames = np.asarray(mask_frames)
    if visible.shape[0] != contained.shape[0] or len(mask_frames) != contained.shape[1]:
        raise ValueError("cluster observations and mask frames must align")
    supporter = sparse.csr_matrix((len(visible), len(visible)), dtype=np.int32)
    for frame in range(visible.shape[1]):
        frame_masks = contained[:, mask_frames == frame]
        supporter += (frame_masks @ frame_masks.T).astype(bool).astype(np.int32)
    upper = sparse.triu(supporter, k=1).tocoo()
    a, b = upper.row, upper.col
    accepted = np.zeros(len(a), bool)
    for start in range(0, len(a), 10000):
        stop = start + 10000
        observers = (visible[a[start:stop]] & visible[b[start:stop]]).sum(1)
        accepted[start:stop] = (observers >= minimum_views) & (
            upper.data[start:stop] >= threshold * observers
        )
    a, b = a[accepted], b[accepted]
    return sparse.csr_matrix(
        (np.ones(2 * len(a), bool), (np.r_[a, b], np.r_[b, a])), shape=supporter.shape
    )


def cluster_masks(graph, *, mode, iterations=3):
    """Same filtered input masks; pairwise overlap or iterative view consensus."""
    if mode not in ("pairwise", "consensus") or iterations < 1:
        raise ValueError("explicit clustering mode and positive iterations required")
    keep = np.flatnonzero(~graph["undersegmented"])
    assignments = np.full(len(graph["mask_keys"]), -1, np.int64)
    if not len(keep):
        return assignments, []
    if mode == "pairwise":
        overlap = sparse.triu(graph["overlap"][keep][:, keep], k=1).tocoo()
        size = graph["sizes"][keep]
        accepted = overlap.data >= 0.5 * np.minimum(
            size[overlap.row], size[overlap.col]
        )
        a, b = overlap.row[accepted], overlap.col[accepted]
        edges = sparse.csr_matrix(
            (np.ones(len(a), bool), (a, b)), shape=(len(keep), len(keep))
        )
        count, assignments[keep] = connected_components(edges, directed=False)
        return assignments, [{"clusters": count, "undirected_edges": len(a)}]
    contained = graph["contained"][keep]
    visible = graph["visible"][keep]
    frames = np.array([f for f, _ in graph["mask_keys"]])
    assignment = np.arange(len(keep))
    history = []
    for _ in range(iterations):
        edges = consensus_edges(contained, visible, frames)
        count, component = connected_components(edges, directed=False)
        history.append({"clusters": count, "undirected_edges": edges.nnz // 2})
        assignment = component[assignment]
        if count == len(visible):
            break
        membership = sparse.csr_matrix(
            (np.ones(len(component), bool), (component, np.arange(len(component)))),
            shape=(count, len(component)),
        )
        contained = (membership @ contained).astype(bool).tocsr()
        visible = np.asarray(membership @ visible, bool)
    assignments[keep] = assignment
    return assignments, history


def arbitrate_candidates(
    votes,
    *,
    observer_valid=None,
    minimum_views=2,
    minimum_score=0.6,
    minimum_margin=0.15,
):
    """One candidate vote per patch/frame, with ambiguity and unsupported fallback."""
    votes = np.asarray(votes)
    if votes.ndim != 2 or votes.dtype.kind not in "iu":
        raise ValueError("integer patch-by-frame candidate votes required")
    valid = votes >= 0 if observer_valid is None else np.asarray(observer_valid, bool)
    if valid.shape != votes.shape or np.any((votes >= 0) & ~valid):
        raise ValueError("candidate votes require usable observers")
    rows, frames = np.nonzero(votes >= 0)
    result = np.full(len(votes), -1, np.int64)
    if not len(rows):
        return result
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), np.int32), (rows, votes[rows, frames])),
        shape=(len(votes), int(votes.max()) + 1),
    )
    row = np.repeat(np.arange(len(votes)), np.diff(matrix.indptr))
    top = np.zeros(len(votes), np.int32)
    np.maximum.at(top, row, matrix.data)
    best = np.full(len(votes), matrix.shape[1], np.int64)
    winner = matrix.data == top[row]
    np.minimum.at(best, row[winner], matrix.indices[winner])
    second = np.zeros(len(votes), np.int32)
    rest = matrix.indices != best[row]
    np.maximum.at(second, row[rest], matrix.data[rest])
    denominator = np.maximum(valid.sum(1), 1)
    accept = (
        (top >= minimum_views)
        & (top / denominator >= minimum_score)
        & ((top - second) / denominator >= minimum_margin)
    )
    result[accept] = best[accept]
    return result


def apply_owner_proposals(source_owners, source_patch, patch_candidate):
    """Only supported merges/splits get new IDs; all residuals keep old owners."""
    owners = np.asarray(source_owners)
    patch = np.asarray(source_patch)
    candidates = np.asarray(patch_candidate)[patch]
    if owners.shape != patch.shape:
        raise ValueError("one exact patch inverse per source row required")
    supported = (candidates >= 0) & (owners > 0)
    result = owners.astype(np.int64, copy=True)
    if not supported.any():
        return result, []
    pairs = np.unique(
        np.column_stack((candidates[supported], owners[supported])), axis=0
    )
    parent, children = np.unique(pairs[:, 1], return_counts=True)
    split_parents = parent[children >= 2]
    candidate_ids, parent_count = np.unique(pairs[:, 0], return_counts=True)
    active = np.union1d(
        candidate_ids[parent_count >= 2], pairs[np.isin(pairs[:, 1], split_parents), 0]
    )
    next_id = int(owners.max()) + 1
    lineage = []
    for offset, candidate in enumerate(active):
        new_id = next_id + offset
        assigned = supported & (candidates == candidate)
        result[assigned] = new_id
        lineage.append(
            {
                "candidate": int(candidate),
                "new_owner": new_id,
                "parents": pairs[pairs[:, 0] == candidate, 1].tolist(),
                "assigned_source_rows": int(assigned.sum()),
            }
        )
    residual_ids, residual_counts = np.unique(result, return_counts=True)
    residual = dict(zip(residual_ids.tolist(), residual_counts.tolist()))
    for entry in lineage:
        entry["parent_residual_source_rows"] = {
            str(parent): residual.get(parent, 0) for parent in entry["parents"]
        }
    return result, lineage
