"""Independent 2D-region observations for fixed source-topology patches."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse


def interior_labels(labels):
    """Discard one-pixel region/image boundaries; zero always means no vote."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("nonnegative integer 2D region image required")
    low = ndimage.minimum_filter(labels, size=3, mode="constant", cval=0)
    high = ndimage.maximum_filter(labels, size=3, mode="constant", cval=0)
    return np.where((labels > 0) & (low == high), labels, 0)


def boundary_graph(adjacency, observations, *, rgb=None, rgb_valid=None):
    """Attenuate repeated mask disagreements without inventing unseen support.

    Each observation column must be a distinct authorized frame. No RGB term is
    inferred from missing colors. Optional RGB values are means from real,
    depth-consistent source observations, never semantic colors or GT textures.
    """
    matrix = sparse.csr_matrix(adjacency, dtype=np.float32)
    labels = np.asarray(observations)
    if labels.ndim != 2 or labels.shape[0] != matrix.shape[0]:
        raise ValueError("one row per frozen patch required")
    if matrix.shape[0] != matrix.shape[1] or np.any(matrix.data < 0):
        raise ValueError("nonnegative square graph required")
    delta = matrix - matrix.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-6:
        raise ValueError("symmetric graph required")
    upper = sparse.triu(matrix, k=1).tocoo()
    a, b = upper.row, upper.col
    joint = np.zeros(len(a), np.int32)
    disagreement = np.zeros(len(a), np.int32)
    for frame in range(labels.shape[1]):
        left, right = labels[a, frame], labels[b, frame]
        valid = (left > 0) & (right > 0)
        joint += valid
        disagreement += valid & (left != right)
    # One undersegmented or erroneous view cannot change connectivity by itself.
    fraction = np.divide(
        disagreement, joint, out=np.zeros(len(a), np.float32), where=joint >= 2
    )
    factor = 1.0 - 0.8 * fraction
    rgb_edges = np.zeros(len(a), bool)
    if rgb is not None:
        colors, color_valid = np.asarray(rgb), np.asarray(rgb_valid, bool)
        if colors.shape != (len(labels), 3) or color_valid.shape != (len(labels),):
            raise ValueError("aligned real RGB and validity required")
        if not np.isfinite(colors[color_valid]).all() or np.any(
            (colors[color_valid] < 0) | (colors[color_valid] > 1)
        ):
            raise ValueError("valid RGB must be finite in [0,1]")
        rgb_edges = color_valid[a] & color_valid[b]
        distance = np.mean(
            np.square(colors[a[rgb_edges]] - colors[b[rgb_edges]]), axis=1
        )
        factor[rgb_edges] *= np.exp(-distance / 0.25**2)
    weight = upper.data * factor
    result = sparse.csr_matrix(
        (np.r_[weight, weight], (np.r_[a, b], np.r_[b, a])), shape=matrix.shape
    )
    boundary = (joint >= 2) & (disagreement > 0)
    nodes = np.unique(np.r_[a[boundary], b[boundary]])
    return result, {
        "jointly_observed_edge_count": int(np.count_nonzero(joint >= 2)),
        "boundary_edge_count": int(boundary.sum()),
        "boundary_node_fraction": len(nodes) / max(1, matrix.shape[0]),
        "rgb_supported_edge_count": int(rgb_edges.sum()),
        "frames": labels.shape[1],
    }
