"""Recover native triangle-owner IDs independently of semantic query eligibility."""
import numpy as np


def native_partition(faces, vertex_rgb, colors):
    faces, rgb = np.asarray(faces), np.asarray(vertex_rgb)
    if (faces.ndim != 2 or faces.shape[1] != 3 or rgb.ndim != 2 or rgb.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer) or rgb.dtype != np.uint8
            or np.any(faces < 0) or np.any(faces >= len(rgb))):
        raise ValueError('native triangle indices and uint8 vertex RGB required')
    normalized = {int(k): tuple(map(int, v)) for k, v in colors.items()}
    if (any(k < 0 or k > np.iinfo(np.int64).max for k in normalized)
            or len(set(normalized.values())) != len(normalized)
            or any(len(v) != 3 or any(c < 0 or c > 255 for c in v) for v in normalized.values())):
        raise ValueError('unique native RGB identities and nonnegative int64 IDs required')
    first_rgb = rgb[faces[:, 0]]
    owners = np.zeros(len(rgb), dtype=np.int64)
    for owner, color in normalized.items():
        owners[faces[np.all(first_rgb == color, axis=1)].reshape(-1)] = owner
    return owners
