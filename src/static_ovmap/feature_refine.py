"""Linear-memory RGB-D surface graph refinement; no production N-by-N matrix."""
import numpy as np
from scipy.sparse import csr_matrix


def geometry_edges(owners, depth, visible, tolerance=.05):
    owners, depth, visible = map(np.asarray, (owners, depth, visible))
    if owners.ndim != 2 or owners.shape != depth.shape or owners.shape != visible.shape:
        raise ValueError('aligned 2D native owners, depth and visibility required')
    valid = (owners > 0) & visible.astype(bool) & np.isfinite(depth) & (depth > 0)
    indices = np.arange(owners.size).reshape(owners.shape)
    sources, targets = [], []
    for first, second in [(np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])]:
        keep = (valid[first] & valid[second] & (owners[first] == owners[second])
                & (np.abs(depth[first]-depth[second]) < tolerance))
        a, b = indices[first][keep], indices[second][keep]
        sources.extend([a, b]); targets.extend([b, a])
    return np.concatenate(sources), np.concatenate(targets)


def refine_sparse(features, sources, targets, *, mixing=.2, temperature=1., channel_chunk=64):
    features = np.asarray(features, dtype=np.float32)
    sources, targets = np.asarray(sources), np.asarray(targets)
    if (features.ndim != 2 or sources.shape != targets.shape or sources.ndim != 1
            or not 0 <= mixing <= 1 or temperature <= 0 or channel_chunk < 1
            or np.any(sources < 0) or np.any(targets < 0)
            or np.any(sources >= len(features)) or np.any(targets >= len(features))
            or not np.isfinite(features).all()):
        raise ValueError('finite features, valid sparse edges and refinement parameters required')
    norms = np.maximum(np.linalg.norm(features, axis=1), 1e-8)
    weights = np.empty(len(sources), dtype=np.float32)
    for start in range(0, len(sources), 8192):
        a, b = sources[start:start+8192], targets[start:start+8192]
        cosine = (features[a]*features[b]).sum(axis=1)/(norms[a]*norms[b])
        weights[start:start+8192] = np.exp((np.clip(cosine, -1, 1)-1)/temperature)
    operator = csr_matrix((weights, (sources, targets)), shape=(len(features), len(features)))
    degree = np.asarray(operator.sum(axis=1)).reshape(-1)
    isolated = degree == 0
    operator.data /= np.repeat(np.maximum(degree, 1e-20), np.diff(operator.indptr))
    result = np.empty_like(features)
    for channel in range(0, features.shape[1], channel_chunk):
        part = features[:, channel:channel+channel_chunk]
        neighbors = operator @ part
        neighbors[isolated] = part[isolated]
        result[:, channel:channel+channel_chunk] = (1-mixing)*part+mixing*neighbors
    return result
